"""Background R2 uploader for Free-tier contributed episodes.

Per `data-collection-free-tier.md`:
- Cron-style: runs at server start, every 24h, and on graceful shutdown
- Reads `~/.reflex/contribute/queue/`, applies episode-level quality filter,
  uploads accepted JSONL → R2 via signed PUT URLs from the contribution worker
- Bandwidth-respecting: throttle to REFLEX_CONTRIB_MAX_MBPS (default 10 MB/s)
- Set REFLEX_NO_CONTRIB_UPLOAD=1 to keep collecting locally without uploading

Phase 1 scope: this module ships the queue-management + episode-quality-filter
+ bandwidth-throttle skeleton, with the actual R2 PUT stubbed (the contribution
worker has not been deployed yet — see infra/contribution-worker/, planned).
Once the worker ships:
    swap `_request_signed_url()` + `_put_to_r2()` to real httpx calls and the
    uploader becomes live. No other call sites change.

Until then, the uploader runs `--dry-run` semantics by default: it scans, it
filters, it logs what *would* upload, but it does not delete the local file.
This keeps queued data safe across the launch transition.
"""
from __future__ import annotations

import json
import logging
import math
import os
import shutil
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_DIR = "~/.reflex/contribute/queue"
DEFAULT_UPLOADED_DIR = "~/.reflex/contribute/uploaded"
DEFAULT_REJECTED_DIR = "~/.reflex/contribute/rejected"

# Episode-level quality thresholds (per spec).
MIN_EPISODE_STEPS = 30
MAX_ZERO_ACTION_FRAC = 0.90
MAX_DUP_IMAGE_HASH_FRAC = 0.50
MAX_ACTION_Z_SCORE = 100.0  # any per-channel z-score exceeding this → reject

# Bandwidth + scheduling defaults.
DEFAULT_MAX_MBPS = 10.0
DEFAULT_DAILY_INTERVAL_S = 86_400.0  # 24h between scheduled passes
KILL_SWITCH_ENV = "REFLEX_NO_CONTRIB_UPLOAD"
THROTTLE_ENV = "REFLEX_CONTRIB_MAX_MBPS"


def _kill_switch_active() -> bool:
    return os.environ.get(KILL_SWITCH_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _max_mbps() -> float:
    raw = os.environ.get(THROTTLE_ENV)
    if not raw:
        return DEFAULT_MAX_MBPS
    try:
        v = float(raw)
        if v < 0:
            return DEFAULT_MAX_MBPS
        return v
    except ValueError:
        logger.warning("invalid %s=%r, falling back to default %g", THROTTLE_ENV, raw, DEFAULT_MAX_MBPS)
        return DEFAULT_MAX_MBPS


@dataclass
class EpisodeStats:
    """Computed once per episode at filter time."""

    episode_id: str
    step_count: int
    has_nan_or_inf_action: bool
    zero_action_frac: float
    dup_image_hash_frac: float
    max_action_z_score: float

    def reject_reasons(self) -> list[str]:
        reasons: list[str] = []
        if self.step_count < MIN_EPISODE_STEPS:
            reasons.append(f"length<{MIN_EPISODE_STEPS}({self.step_count})")
        if self.has_nan_or_inf_action:
            reasons.append("nan_or_inf_in_actions")
        if self.zero_action_frac > MAX_ZERO_ACTION_FRAC:
            reasons.append(f"all_zero_actions({self.zero_action_frac:.2f})")
        if self.dup_image_hash_frac > MAX_DUP_IMAGE_HASH_FRAC:
            reasons.append(f"dup_images({self.dup_image_hash_frac:.2f})")
        if self.max_action_z_score > MAX_ACTION_Z_SCORE:
            reasons.append(f"action_z>{MAX_ACTION_Z_SCORE}({self.max_action_z_score:.1f})")
        return reasons

    @property
    def accepted(self) -> bool:
        return not self.reject_reasons()


@dataclass
class UploadOutcome:
    """One pass of the uploader. Aggregates across all queue files."""

    files_scanned: int = 0
    episodes_inspected: int = 0
    episodes_accepted: int = 0
    episodes_rejected: int = 0
    bytes_uploaded: int = 0
    files_uploaded: int = 0
    files_kept_in_queue: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "files_scanned": self.files_scanned,
            "episodes_inspected": self.episodes_inspected,
            "episodes_accepted": self.episodes_accepted,
            "episodes_rejected": self.episodes_rejected,
            "bytes_uploaded": self.bytes_uploaded,
            "files_uploaded": self.files_uploaded,
            "files_kept_in_queue": self.files_kept_in_queue,
            "rejection_reasons": dict(self.rejection_reasons),
            "errors": list(self.errors),
        }


def _iter_jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("uploader.skipping_bad_line file=%s err=%s", path, exc)


def _stats_for_episode(rows: list[dict[str, Any]]) -> EpisodeStats:
    """Compute per-episode quality stats from a list of /act event rows."""
    step_count = len(rows)
    has_nan_or_inf = False
    zero_action_count = 0
    z_max = 0.0
    image_hashes: list[str] = []

    # Pass 1: collect action vectors (flattened) for stats + check NaN/Inf.
    flat_actions: list[float] = []
    for row in rows:
        chunk = row.get("action_chunk") or []
        for action in chunk:
            if not isinstance(action, list):
                continue
            non_zero = False
            for v in action:
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isnan(fv) or math.isinf(fv):
                    has_nan_or_inf = True
                    continue
                flat_actions.append(fv)
                if fv != 0.0:
                    non_zero = True
            if not non_zero:
                zero_action_count += 1
        img_hash = row.get("image_b64")
        if isinstance(img_hash, str):
            image_hashes.append(img_hash)

    total_action_count = sum(
        1 for r in rows for a in (r.get("action_chunk") or []) if isinstance(a, list)
    )
    zero_frac = (zero_action_count / total_action_count) if total_action_count else 0.0

    if flat_actions:
        n = len(flat_actions)
        mean = sum(flat_actions) / n
        var = sum((v - mean) ** 2 for v in flat_actions) / n
        std = math.sqrt(var) if var > 0 else 0.0
        if std > 0:
            z_max = max(abs(v - mean) / std for v in flat_actions)

    if image_hashes:
        unique = len(set(image_hashes))
        dup_frac = 1.0 - (unique / len(image_hashes))
    else:
        dup_frac = 0.0

    return EpisodeStats(
        episode_id=rows[0].get("episode_id", "") if rows else "",
        step_count=step_count,
        has_nan_or_inf_action=has_nan_or_inf,
        zero_action_frac=zero_frac,
        dup_image_hash_frac=dup_frac,
        max_action_z_score=z_max,
    )


def filter_episodes(
    rows_by_episode: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, EpisodeStats]]:
    """Split rows into accepted/rejected episodes. Returns (accepted_rows_by_episode, stats_by_episode)."""
    accepted: dict[str, list[dict[str, Any]]] = {}
    stats: dict[str, EpisodeStats] = {}
    for episode_id, rows in rows_by_episode.items():
        s = _stats_for_episode(rows)
        stats[episode_id] = s
        if s.accepted:
            accepted[episode_id] = rows
    return accepted, stats


# ── R2 transport (Phase 1: stubbed — worker not yet deployed) ──────────────────


class UploadStub(Exception):
    """Marker raised by the Phase 1 stub to indicate the worker is not yet live."""


def _request_signed_url(*, contributor_id: str, file_name: str) -> str:
    """Phase 1 stub: returns a marker URL. Phase 1.5 will POST to the
    contribution worker for a real signed PUT URL."""
    raise UploadStub(
        f"contribution worker not yet deployed; would request signed URL "
        f"for contributor_id={contributor_id} file={file_name}"
    )


def _put_to_r2(*, signed_url: str, file_bytes: bytes, max_mbps: float) -> int:
    """Phase 1 stub: would PUT the bytes to R2 with bandwidth throttling.
    Phase 1.5 swaps to httpx.put with chunked upload + sleep-throttle.
    Returns the bytes uploaded on success."""
    raise UploadStub(
        f"contribution worker not yet deployed; would PUT {len(file_bytes)} bytes "
        f"to {signed_url} at max {max_mbps} MB/s"
    )


# ── The uploader (Phase 1: dry-run by default; live mode flagged on) ───────────


class Uploader:
    """Cron-driven background uploader for the Curate queue.

    Default mode is DRY-RUN — files are scanned, filtered, accepted/rejected
    decisions are made and logged, but the local queue is NOT mutated. This
    is Phase 1 behavior because the contribution worker hasn't been deployed
    yet. Set `live=True` (Phase 1.5) to enable real R2 PUT + queue removal.
    """

    __slots__ = (
        "_queue_dir", "_uploaded_dir", "_rejected_dir",
        "_contributor_id", "_live", "_max_mbps",
        "_thread", "_stop_event", "_interval_s",
        "_last_outcome", "_lock",
    )

    def __init__(
        self,
        *,
        contributor_id: str,
        queue_dir: str | Path = DEFAULT_QUEUE_DIR,
        uploaded_dir: str | Path = DEFAULT_UPLOADED_DIR,
        rejected_dir: str | Path = DEFAULT_REJECTED_DIR,
        live: bool = False,
        max_mbps: float | None = None,
        interval_s: float = DEFAULT_DAILY_INTERVAL_S,
    ):
        self._queue_dir = Path(queue_dir).expanduser()
        self._uploaded_dir = Path(uploaded_dir).expanduser()
        self._rejected_dir = Path(rejected_dir).expanduser()
        self._contributor_id = contributor_id
        self._live = bool(live)
        self._max_mbps = float(max_mbps) if max_mbps is not None else _max_mbps()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._interval_s = float(interval_s)
        self._last_outcome: UploadOutcome | None = None
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def last_outcome(self) -> UploadOutcome | None:
        return self._last_outcome

    def run_once(self) -> UploadOutcome:
        """Single pass: scan queue, filter episodes, upload accepted, archive rejected.
        Idempotent — safe to call multiple times."""
        outcome = UploadOutcome()
        if _kill_switch_active():
            logger.info("uploader.kill_switch_active queue_dir=%s", self._queue_dir)
            return outcome

        if not self._queue_dir.exists():
            return outcome

        for jsonl_path in sorted(self._queue_dir.glob("*.jsonl")):
            outcome.files_scanned += 1
            try:
                rows_by_episode = self._group_by_episode(jsonl_path)
            except Exception as exc:  # noqa: BLE001
                msg = f"failed to read {jsonl_path}: {exc}"
                logger.error("uploader.read_failed %s", msg)
                outcome.errors.append(msg)
                continue

            accepted, stats = filter_episodes(rows_by_episode)
            outcome.episodes_inspected += len(stats)
            for s in stats.values():
                if s.accepted:
                    outcome.episodes_accepted += 1
                else:
                    outcome.episodes_rejected += 1
                    for reason in s.reject_reasons():
                        outcome.rejection_reasons[reason] += 1

            if not accepted:
                # All episodes rejected — archive the file to rejected/ for audit.
                self._archive_rejected(jsonl_path)
                outcome.files_kept_in_queue += 0
                continue

            # Build accepted-rows-only payload + try upload.
            accepted_bytes = self._build_payload(accepted)
            try:
                if self._live:
                    signed = _request_signed_url(
                        contributor_id=self._contributor_id,
                        file_name=jsonl_path.name,
                    )
                    bytes_up = _put_to_r2(
                        signed_url=signed, file_bytes=accepted_bytes,
                        max_mbps=self._max_mbps,
                    )
                    outcome.bytes_uploaded += bytes_up
                    outcome.files_uploaded += 1
                    self._archive_uploaded(jsonl_path)
                else:
                    logger.info(
                        "uploader.dry_run file=%s episodes=%d bytes=%d (worker not deployed)",
                        jsonl_path.name, len(accepted), len(accepted_bytes),
                    )
                    outcome.files_kept_in_queue += 1
            except UploadStub as exc:
                logger.info("uploader.stub %s file=%s", exc, jsonl_path.name)
                outcome.files_kept_in_queue += 1
            except Exception as exc:  # noqa: BLE001
                msg = f"upload failed for {jsonl_path.name}: {exc}"
                logger.error("uploader.upload_failed %s", msg)
                outcome.errors.append(msg)
                outcome.files_kept_in_queue += 1

        with self._lock:
            self._last_outcome = outcome
        return outcome

    def start(self) -> None:
        """Spawn the background loop. Idempotent."""
        if self.is_running:
            return
        self._stop_event.clear()
        # Run an immediate pass in the loop, then wait `interval_s` between passes.
        self._thread = threading.Thread(
            target=self._loop, name="curate-uploader", daemon=True,
        )
        self._thread.start()

    def stop(self, *, drain: bool = True, timeout_s: float = 30.0) -> None:
        """Stop the background loop. If drain=True, runs one final pass before exit."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=timeout_s)
        self._thread = None
        if drain:
            try:
                self.run_once()
            except Exception as exc:  # noqa: BLE001
                logger.warning("uploader.drain_failed %s", exc)

    def _loop(self) -> None:
        # Always run an immediate pass so a fresh start picks up backlog.
        try:
            self.run_once()
        except Exception as exc:  # noqa: BLE001
            logger.error("uploader.run_failed %s", exc)
        # Then wait `interval_s` between passes; the stop_event provides a
        # responsive cancellation path.
        while not self._stop_event.wait(self._interval_s):
            try:
                self.run_once()
            except Exception as exc:  # noqa: BLE001
                logger.error("uploader.run_failed %s", exc)

    def _group_by_episode(self, path: Path) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in _iter_jsonl_rows(path):
            episode_id = row.get("episode_id") or "anon"
            out[episode_id].append(row)
        return dict(out)

    def _build_payload(self, rows_by_episode: dict[str, list[dict[str, Any]]]) -> bytes:
        """Serialize accepted rows back to a JSONL byte payload for upload."""
        lines: list[str] = []
        for episode_id, rows in rows_by_episode.items():
            for row in rows:
                lines.append(json.dumps(row))
        return ("\n".join(lines) + "\n").encode("utf-8")

    def _archive_uploaded(self, src: Path) -> None:
        self._uploaded_dir.mkdir(parents=True, exist_ok=True)
        dest = self._uploaded_dir / src.name
        if dest.exists():
            stem = dest.stem
            suffix = dest.suffix
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            dest = self._uploaded_dir / f"{stem}.{ts}{suffix}"
        shutil.move(str(src), str(dest))

    def _archive_rejected(self, src: Path) -> None:
        self._rejected_dir.mkdir(parents=True, exist_ok=True)
        dest = self._rejected_dir / src.name
        if dest.exists():
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            dest = self._rejected_dir / f"{src.stem}.{ts}{src.suffix}"
        shutil.move(str(src), str(dest))


__all__ = [
    "DEFAULT_QUEUE_DIR",
    "DEFAULT_UPLOADED_DIR",
    "DEFAULT_REJECTED_DIR",
    "DEFAULT_DAILY_INTERVAL_S",
    "DEFAULT_MAX_MBPS",
    "MIN_EPISODE_STEPS",
    "MAX_ZERO_ACTION_FRAC",
    "MAX_DUP_IMAGE_HASH_FRAC",
    "MAX_ACTION_Z_SCORE",
    "EpisodeStats",
    "UploadOutcome",
    "UploadStub",
    "Uploader",
    "filter_episodes",
]

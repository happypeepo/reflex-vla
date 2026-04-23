"""`reflex replay` CLI implementation.

Day 2 scope: load JSONL, replay each request against a target export,
compute per-request actions diff (cosine + max_abs), print human-readable
summary + optional JSON output. Latency / cache / guard diff modes land
in Day 3.

Usage (registered as a typer subcommand by src/reflex/cli.py):

    reflex replay <file.jsonl[.gz]> --model <export_dir>           \\
            [--diff actions]                                       \\
            [--n <int>]                                            \\
            [--output <json>]                                      \\
            [--fail-on actions]                                    \\
            [--no-replay]            # parse only, don't load model

Replay invokes the same predict path as `reflex serve` /act, so the
diff is a true regression-against-recorded comparison.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two flat float vectors. Returns 0.0 on
    degenerate inputs (zero-norm or length mismatch)."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def max_abs_diff(a: list[float], b: list[float]) -> float:
    """Max |a[i] - b[i]| over the shorter prefix. 0.0 on empty inputs."""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    return max(abs(a[i] - b[i]) for i in range(n))


def _flatten_actions(actions: list[list[float]]) -> list[float]:
    """Flatten a chunk-of-actions to one flat vector for cosine / max_abs.
    Returns [] on empty input."""
    out: list[float] = []
    for row in actions:
        out.extend(row)
    return out


def diff_actions(
    recorded: list[list[float]], replayed: list[list[float]],
    *, threshold_cos: float = 0.999, threshold_max_abs: float = 1e-3,
) -> dict[str, Any]:
    """Compare recorded vs replayed action chunks. Returns a dict with
    cosine, max_abs, pass flag."""
    flat_r = _flatten_actions(recorded)
    flat_p = _flatten_actions(replayed)
    cos = cosine_similarity(flat_r, flat_p)
    mad = max_abs_diff(flat_r, flat_p)
    return {
        "cosine": cos,
        "max_abs_diff": mad,
        "passed": cos >= threshold_cos and mad <= threshold_max_abs,
        "threshold_cos": threshold_cos,
        "threshold_max_abs": threshold_max_abs,
    }


def _load_target_server(model: str):
    """Load the target export for replay. Returns the same server type
    create_app() would use, but invoked outside a FastAPI lifespan."""
    from reflex.runtime.server import create_app  # noqa: F401  (deferred import)

    # Use create_app to get the same dispatch logic as `reflex serve`,
    # but bypass FastAPI — we just want the underlying server object.
    # We can't call create_app() here easily because it sets up FastAPI;
    # instead, replicate its dispatch-by-config logic directly.
    config_path = Path(model) / "reflex_config.json"
    cfg: dict[str, Any] = {}
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            cfg = {}

    if cfg.get("export_kind") == "monolithic":
        model_type = cfg.get("model_type", "smolvla")
        if model_type == "pi0":
            from reflex.runtime.pi0_onnx_server import Pi0OnnxServer
            srv = Pi0OnnxServer(model)
        elif model_type == "smolvla":
            from reflex.runtime.smolvla_onnx_server import SmolVLAOnnxServer
            srv = SmolVLAOnnxServer(model)
        else:
            raise ValueError(
                f"Replay against monolithic model_type={model_type!r} not "
                f"yet supported. Day 2 ships pi0 + smolvla only."
            )
    else:
        from reflex.runtime.server import ReflexServer
        srv = ReflexServer(model)
    srv.load()
    return srv


def run_replay(
    trace_file: str,
    model: str | None,
    *,
    diff_mode: str = "actions",
    n: int = 0,  # 0 = all
    output_json: str = "",
    fail_on: str = "",
    no_replay: bool = False,
) -> int:
    """Implementation entry point. Returns CLI exit code."""
    from reflex.replay.readers import (  # noqa: F401  (deferred import)
        ReplaySchemaUnknownError,
        load_reader,
    )

    try:
        reader = load_reader(trace_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        return 1

    header = reader.read_header()
    print(f"Replay: {trace_file}")
    print(f"  reflex_version: {header.get('reflex_version', '?')}")
    print(f"  model_hash:     {header.get('model_hash', '?')}")
    print(f"  config_hash:    {header.get('config_hash', '?')}")
    print(f"  model_type:     {header.get('model_type', '?')}")
    print(f"  export_kind:    {header.get('export_kind', '?')}")
    print(f"  embodiment:     {header.get('embodiment', '?')}")
    print(f"  redaction:      {header.get('redaction', {})}")
    print(f"  started_at:     {header.get('started_at', '?')}")

    # Parse-only mode: dump records, no model load
    if no_replay or model is None:
        if no_replay:
            print(f"\n--no-replay: parsing records only, not loading model.\n")
        records = list(reader.read_records())
        n_req = sum(1 for k, _ in records if k == "request")
        n_foot = sum(1 for k, _ in records if k == "footer")
        print(f"  records:        {len(records)} ({n_req} requests, {n_foot} footer)")
        return 0

    # Load target model
    print(f"\nLoading target model: {model}")
    try:
        srv = _load_target_server(model)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR loading target: {e}")
        return 2

    # Hash mismatch warnings (not blocking)
    target_model_hash = ""
    try:
        from reflex.runtime.record import compute_model_hash
        target_model_hash = compute_model_hash(model)
    except Exception:  # noqa: BLE001
        pass
    if target_model_hash and target_model_hash != header.get("model_hash"):
        print(
            f"WARN: model_hash mismatch — recorded={header.get('model_hash')} "
            f"vs replay={target_model_hash}"
        )

    # Replay loop
    diffs: list[dict[str, Any]] = []
    n_replayed = 0
    n_passed = 0
    image_redaction = (header.get("redaction", {}) or {}).get("image", "hash_only")

    if image_redaction != "full":
        print(
            f"WARN: trace was recorded with image redaction='{image_redaction}'; "
            f"replay needs full images. Day 2 limitation — pass --no-replay to "
            f"parse the trace without re-running inference."
        )

    print(f"\nReplaying requests (--n={n or 'all'}, --diff={diff_mode}):")
    for kind, rec in reader.read_records():
        if kind != "request":
            continue
        if n and n_replayed >= n:
            break

        req = rec.get("request", {})
        recorded_resp = rec.get("response", {})
        recorded_actions = recorded_resp.get("actions", [])

        # Day-2 limitation: we can only replay if the trace has full images
        if not req.get("image_b64"):
            n_replayed += 1
            continue

        try:
            replay_resp = srv.predict_from_base64(
                image_b64=req["image_b64"],
                instruction=req.get("instruction", ""),
                state=req.get("state"),
            )
        except Exception as e:  # noqa: BLE001
            print(f"  seq={rec.get('seq')}: replay raised {type(e).__name__}: {e}")
            n_replayed += 1
            continue

        replayed_actions = replay_resp.get("actions", [])
        d = diff_actions(recorded_actions, replayed_actions)
        d["seq"] = rec.get("seq")
        diffs.append(d)
        n_replayed += 1
        if d["passed"]:
            n_passed += 1

        marker = "PASS" if d["passed"] else "FAIL"
        print(
            f"  seq={d['seq']:4d}  cos={d['cosine']:.6f}  "
            f"max_abs={d['max_abs_diff']:.2e}  {marker}"
        )

    # Summary
    print("\nSummary:")
    print(f"  replayed: {n_replayed}")
    print(f"  diffed:   {len(diffs)}")
    print(f"  passed:   {n_passed}/{len(diffs)} (threshold cos≥0.999, max_abs<1e-3)")

    if output_json:
        Path(output_json).write_text(
            json.dumps(
                {
                    "summary": {
                        "trace_file": str(trace_file),
                        "model": model,
                        "n_replayed": n_replayed,
                        "n_diffed": len(diffs),
                        "n_passed": n_passed,
                    },
                    "header": header,
                    "per_request_diffs": diffs,
                },
                indent=2,
            )
        )
        print(f"  output:   {output_json}")

    if fail_on == "actions" and n_passed < len(diffs):
        return 3

    return 0


__all__ = [
    "cosine_similarity",
    "max_abs_diff",
    "diff_actions",
    "run_replay",
]

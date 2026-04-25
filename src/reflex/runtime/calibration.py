"""Auto-calibration substrate — Phase 1 a2u-calibration feature.

Per ADR 2026-04-25-auto-calibration-architecture:
- SELECTION not tuning: pick among pre-shipped variants + bucketed values
- Greedy resolver order: variant → provider → NFE → chunk_size →
  latency_compensation_ms (strict partial order)
- Schema v1 with `schema_version` as the FIRST field for fast-path detection
- Cache key: (hardware_fingerprint, embodiment, model_hash)
- Hardware fingerprint: gpu_uuid + gpu_name + driver_major.minor + cuda +
  kernel + cpu_count + ram_gb + reflex_version (major.minor only on driver
  to avoid invalidation on every patch)

This module is the substrate (Day 1 of the plan). The greedy resolver +
measurement harness + CLI integration are Days 2-9.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


# Schema version of the cache JSON. Bump on a breaking change; v1 readers
# refuse to load v2+. Phase 1 = v1; Phase 2 evolution is additive-only
# (new optional fields only, no rename / no remove).
SCHEMA_VERSION = 1

# Cold-start defaults for latency_compensation_ms by embodiment, used until
# the LatencyTracker has populated real samples (per ADR Lens 3 estimates).
COLD_START_LATENCY_COMP_MS_BY_EMBODIMENT: dict[str, float] = {
    "franka": 40.0,
    "so100": 60.0,
    "ur5": 40.0,
}
DEFAULT_COLD_START_LATENCY_COMP_MS = 40.0

# How long after a calibration entry was recorded before we treat it as
# stale (per ADR: 30 days OR fingerprint mismatch OR reflex_version change).
DEFAULT_STALE_AFTER_DAYS = 30


@dataclass(frozen=True)
class HardwareFingerprint:
    """Per-host identity. Stable across reboots; insensitive to driver
    patch-version bumps (only major.minor matters for cache validity).

    `current()` probes the running host. Returns sentinel "unknown" values
    on probes that fail (e.g., nvidia-smi missing on a CPU-only host) so
    operators see one consistent fingerprint shape, not None / Optional.
    """

    gpu_uuid: str
    gpu_name: str
    driver_version_major: int
    driver_version_minor: int
    cuda_version_major: int
    cuda_version_minor: int
    kernel_release: str
    cpu_count: int
    ram_gb: int
    reflex_version: str

    @classmethod
    def current(cls) -> "HardwareFingerprint":
        """Probe the running host. Always returns a valid fingerprint;
        unknown fields populated with sentinels."""
        gpu_uuid, gpu_name = _probe_gpu()
        driver_major, driver_minor = _probe_driver_version()
        cuda_major, cuda_minor = _probe_cuda_version()
        kernel = platform.release() or "unknown"
        cpu_count = os.cpu_count() or 0
        ram_gb = _probe_ram_gb()
        reflex_version = _probe_reflex_version()
        return cls(
            gpu_uuid=gpu_uuid,
            gpu_name=gpu_name,
            driver_version_major=driver_major,
            driver_version_minor=driver_minor,
            cuda_version_major=cuda_major,
            cuda_version_minor=cuda_minor,
            kernel_release=kernel,
            cpu_count=cpu_count,
            ram_gb=ram_gb,
            reflex_version=reflex_version,
        )

    def matches(self, other: "HardwareFingerprint", *, strict: bool = False) -> bool:
        """Compare two fingerprints. Default mode ignores `kernel_release` +
        `ram_gb` minor differences (kernel patch + RAM rounding). Strict
        mode requires bitwise equality."""
        if strict:
            return self == other
        return (
            self.gpu_uuid == other.gpu_uuid
            and self.gpu_name == other.gpu_name
            and self.driver_version_major == other.driver_version_major
            and self.driver_version_minor == other.driver_version_minor
            and self.cuda_version_major == other.cuda_version_major
            and self.cuda_version_minor == other.cuda_version_minor
            and self.cpu_count == other.cpu_count
            and abs(self.ram_gb - other.ram_gb) <= 1  # tolerate ±1 GB rounding
            and self.reflex_version == other.reflex_version
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HardwareFingerprint":
        return cls(**d)


@dataclass(frozen=True)
class MeasurementQuality:
    """Quality of one measurement run — recorded into each calibration entry
    so operators can see how trustworthy the calibration is."""

    warmup_iters: int
    measurement_iters: int
    median_ms: float
    p99_ms: float
    n_outliers_dropped: int
    quality_score: float  # ∈ [0, 1] — 1.0 = clean (low variance), 0.0 = noisy

    def __post_init__(self) -> None:
        if not (0.0 <= self.quality_score <= 1.0):
            raise ValueError(
                f"quality_score must be in [0, 1], got {self.quality_score}"
            )
        if self.warmup_iters < 0:
            raise ValueError(f"warmup_iters must be >= 0, got {self.warmup_iters}")
        if self.measurement_iters < 1:
            raise ValueError(
                f"measurement_iters must be >= 1, got {self.measurement_iters}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MeasurementQuality":
        return cls(**d)


@dataclass(frozen=True)
class MeasurementContext:
    """Stack-version snapshot at calibration time. Drives staleness detection
    on the next boot: any field changing invalidates the cached entry."""

    ort_version: str
    torch_version: str | None  # None when torch isn't imported
    numpy_version: str
    onnx_version: str | None  # None when onnx isn't imported

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MeasurementContext":
        return cls(**d)

    @classmethod
    def current(cls) -> "MeasurementContext":
        """Probe versions of the runtime stack. Optional imports return None."""
        ort_v = _safe_pkg_version("onnxruntime")
        torch_v = _safe_pkg_version("torch")
        numpy_v = _safe_pkg_version("numpy") or "unknown"
        onnx_v = _safe_pkg_version("onnx")
        return cls(
            ort_version=ort_v or "unknown",
            torch_version=torch_v,
            numpy_version=numpy_v,
            onnx_version=onnx_v,
        )


@dataclass(frozen=True)
class CalibrationEntry:
    """One calibration result for a (hardware × embodiment × model_hash) tuple.

    Frozen + serializable — written once per calibration pass; never mutated.
    Re-calibration produces a new entry that overwrites the old via
    CalibrationCache.record().
    """

    chunk_size: int
    nfe: int
    latency_compensation_ms: float
    provider: str
    variant: str
    measurement_quality: MeasurementQuality
    measurement_context: MeasurementContext
    timestamp: str  # ISO 8601

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")
        if not (1 <= self.nfe <= 50):
            raise ValueError(f"nfe must be in [1, 50], got {self.nfe}")
        if self.latency_compensation_ms < 0:
            raise ValueError(
                f"latency_compensation_ms must be >= 0, got "
                f"{self.latency_compensation_ms}"
            )
        if not self.provider:
            raise ValueError("provider must be non-empty")
        if not self.variant:
            raise ValueError("variant must be non-empty")

    def age_seconds(self) -> float:
        """Seconds since the entry was recorded. Returns large positive value
        on parse failure (treats unparseable timestamp as stale)."""
        try:
            ts = datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            return (now - ts).total_seconds()
        except Exception:  # noqa: BLE001
            return float("inf")

    def is_stale(self, max_age_days: float = DEFAULT_STALE_AFTER_DAYS) -> bool:
        return self.age_seconds() > max_age_days * 86_400.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "nfe": self.nfe,
            "latency_compensation_ms": self.latency_compensation_ms,
            "provider": self.provider,
            "variant": self.variant,
            "measurement_quality": self.measurement_quality.to_dict(),
            "measurement_context": self.measurement_context.to_dict(),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CalibrationEntry":
        return cls(
            chunk_size=int(d["chunk_size"]),
            nfe=int(d["nfe"]),
            latency_compensation_ms=float(d["latency_compensation_ms"]),
            provider=str(d["provider"]),
            variant=str(d["variant"]),
            measurement_quality=MeasurementQuality.from_dict(d["measurement_quality"]),
            measurement_context=MeasurementContext.from_dict(d["measurement_context"]),
            timestamp=str(d["timestamp"]),
        )


@dataclass
class CalibrationCache:
    """Top-level cache container — one per host. Backed by a JSON file at
    `~/.reflex/calibration.json` by default.

    Schema v1 layout (FIRST field is `schema_version` for fast-path version
    detection):

    {
      "schema_version": 1,
      "reflex_version": "0.5.0",
      "calibration_date": "2026-04-25T10:00:00Z",
      "hardware_fingerprint": {...},
      "entries": {
        "franka::abc123": <CalibrationEntry dict>,
        "so100::def456": <CalibrationEntry dict>,
        ...
      }
    }

    Entry keys are `{embodiment}::{model_hash}`. Model hash comes from the
    existing compute_model_hash function (server.py:1246). Embodiment comes
    from the loaded EmbodimentConfig.embodiment field.

    Phase 2 evolution: ADD-ONLY. Never rename / remove a v1 field. Schema v2
    bump only on breaking changes; v1 readers refuse to load v2+ loud.
    """

    schema_version: int = SCHEMA_VERSION
    reflex_version: str = ""
    calibration_date: str = ""
    hardware_fingerprint: HardwareFingerprint | None = None
    entries: dict[str, CalibrationEntry] = field(default_factory=dict)

    # Class constants surfaced for downstream consumers.
    SCHEMA_VERSION: ClassVar[int] = SCHEMA_VERSION

    @staticmethod
    def make_key(embodiment: str, model_hash: str) -> str:
        if not embodiment or not model_hash:
            raise ValueError(
                f"embodiment + model_hash must both be non-empty; got "
                f"{embodiment!r}, {model_hash!r}"
            )
        if "::" in embodiment or "::" in model_hash:
            raise ValueError(
                "embodiment + model_hash must not contain '::' separator"
            )
        return f"{embodiment}::{model_hash}"

    def lookup(
        self,
        *,
        embodiment: str,
        model_hash: str,
        require_fingerprint: HardwareFingerprint | None = None,
    ) -> CalibrationEntry | None:
        """Return the cached entry for this (embodiment, model_hash) tuple,
        or None if absent / hardware-mismatched.

        When `require_fingerprint` is provided, returns None unless the
        cached fingerprint matches (default tolerance — kernel patch +
        RAM rounding ignored)."""
        if (
            require_fingerprint is not None
            and self.hardware_fingerprint is not None
            and not self.hardware_fingerprint.matches(require_fingerprint)
        ):
            return None
        return self.entries.get(self.make_key(embodiment, model_hash))

    def record(
        self,
        *,
        embodiment: str,
        model_hash: str,
        entry: CalibrationEntry,
    ) -> None:
        """Insert or overwrite the entry. Updates calibration_date."""
        self.entries[self.make_key(embodiment, model_hash)] = entry
        self.calibration_date = _utcnow_iso()

    def is_stale(
        self,
        current: HardwareFingerprint,
        *,
        max_age_days: float = DEFAULT_STALE_AFTER_DAYS,
    ) -> bool:
        """Cache is stale when:
        - hardware fingerprint mismatches OR
        - calibration_date older than max_age_days
        """
        if self.hardware_fingerprint is None:
            return True
        if not self.hardware_fingerprint.matches(current):
            return True
        if not self.calibration_date:
            return True
        try:
            ts = datetime.fromisoformat(self.calibration_date.replace("Z", "+00:00"))
            age_s = (datetime.now(timezone.utc) - ts).total_seconds()
            return age_s > max_age_days * 86_400.0
        except Exception:  # noqa: BLE001
            return True

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "reflex_version": self.reflex_version,
            "calibration_date": self.calibration_date,
            "hardware_fingerprint": (
                self.hardware_fingerprint.to_dict()
                if self.hardware_fingerprint is not None else None
            ),
            "entries": {
                k: v.to_dict() for k, v in self.entries.items()
            },
        }
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CalibrationCache":
        sv = int(d.get("schema_version", 0))
        if sv > SCHEMA_VERSION:
            raise ValueError(
                f"calibration cache schema_version={sv} exceeds supported "
                f"version {SCHEMA_VERSION}. Upgrade reflex-vla or delete "
                f"the cache file."
            )
        if sv < 1:
            raise ValueError(
                f"calibration cache schema_version={sv} is invalid (must be >= 1)"
            )
        fp = d.get("hardware_fingerprint")
        return cls(
            schema_version=sv,
            reflex_version=str(d.get("reflex_version", "")),
            calibration_date=str(d.get("calibration_date", "")),
            hardware_fingerprint=(
                HardwareFingerprint.from_dict(fp) if fp is not None else None
            ),
            entries={
                str(k): CalibrationEntry.from_dict(v)
                for k, v in d.get("entries", {}).items()
            },
        )

    def save(self, path: str | Path) -> None:
        """Atomic write via temp + rename."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationCache":
        """Load + validate. Raises FileNotFoundError when path doesn't exist."""
        path = Path(path)
        data = json.loads(path.read_text())
        return cls.from_dict(data)

    @classmethod
    def load_or_empty(cls, path: str | Path) -> "CalibrationCache":
        """Load if file exists; else return a fresh empty cache."""
        path = Path(path)
        if not path.exists():
            return cls(
                schema_version=SCHEMA_VERSION,
                reflex_version=_probe_reflex_version(),
                calibration_date="",
                hardware_fingerprint=None,
                entries={},
            )
        return cls.load(path)


# ---------------------------------------------------------------------------
# Hardware probes — defensive, all failures fall back to "unknown" sentinels
# ---------------------------------------------------------------------------


def _probe_gpu() -> tuple[str, str]:
    """Probe primary GPU UUID + name via nvidia-smi. Returns ("unknown",
    "unknown") on any failure (CPU-only host, NVIDIA driver missing, etc.)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5.0,
        )
        if result.returncode != 0:
            return ("unknown", "unknown")
        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",", 1)]
        if len(parts) != 2:
            return ("unknown", "unknown")
        return (parts[0], parts[1])
    except Exception:  # noqa: BLE001
        return ("unknown", "unknown")


def _probe_driver_version() -> tuple[int, int]:
    """Probe NVIDIA driver version (major.minor only — patch ignored)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5.0,
        )
        if result.returncode != 0:
            return (0, 0)
        v = result.stdout.strip().split("\n")[0].strip()
        m = re.match(r"(\d+)\.(\d+)", v)
        if not m:
            return (0, 0)
        return (int(m.group(1)), int(m.group(2)))
    except Exception:  # noqa: BLE001
        return (0, 0)


def _probe_cuda_version() -> tuple[int, int]:
    """Probe CUDA toolkit version (major.minor)."""
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True, text=True, timeout=5.0,
        )
        if result.returncode != 0:
            return (0, 0)
        m = re.search(r"release (\d+)\.(\d+)", result.stdout)
        if not m:
            return (0, 0)
        return (int(m.group(1)), int(m.group(2)))
    except Exception:  # noqa: BLE001
        return (0, 0)


def _probe_ram_gb() -> int:
    """Probe total RAM in GB. Linux: parse /proc/meminfo. macOS: sysctl.
    Other: 0."""
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return max(1, round(kb / (1024 * 1024)))
        elif sys.platform == "darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=2.0,
            )
            if result.returncode == 0:
                bytes_total = int(result.stdout.strip())
                return max(1, round(bytes_total / (1024 ** 3)))
    except Exception:  # noqa: BLE001
        pass
    return 0


def _probe_reflex_version() -> str:
    """Probe installed reflex-vla version. Returns 'unknown' on any failure."""
    return _safe_pkg_version("reflex-vla") or _safe_pkg_version("reflex") or "unknown"


def _safe_pkg_version(pkg: str) -> str | None:
    """Return version string for an installed pkg, or None when not installed."""
    try:
        from importlib.metadata import version
        return version(pkg)
    except Exception:  # noqa: BLE001
        return None


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_STALE_AFTER_DAYS",
    "COLD_START_LATENCY_COMP_MS_BY_EMBODIMENT",
    "DEFAULT_COLD_START_LATENCY_COMP_MS",
    "HardwareFingerprint",
    "MeasurementQuality",
    "MeasurementContext",
    "CalibrationEntry",
    "CalibrationCache",
]

"""RTC adapter — wraps ``lerobot.policies.rtc.RTCProcessor`` for Reflex serve.

Implements Real-Time Chunking (arxiv 2506.07339) so the robot keeps executing
the tail of one chunk while the next chunk is being computed. Net: 2-3x
effective throughput on high-latency deployments (Jetson-class).

Status: SKELETON. Class signatures + docstrings shipped. Logic gated on open
design questions tracked in ``reflex_context/04_product/rtc_lerobot_design_review.md``.
Do not import from production code until items marked ``# TODO(rtc)`` resolve.

Design: ``reflex_context/reference/deep_dive_lerobot_rtc.md``
Plan ref: ``reflex_context/04_product/serve_technical_plan_v3.md`` §2.1
Goal: ``serve-rtc-wrapper`` (GOALS.yaml, weight 10)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from reflex.runtime.buffer import ActionChunkBuffer

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Soft import of lerobot.policies.rtc
# ──────────────────────────────────────────────────────────────────
try:
    from lerobot.policies.rtc import RTCProcessor  # type: ignore
    _RTC_AVAILABLE = True
except ImportError:  # pragma: no cover
    RTCProcessor = None  # type: ignore
    _RTC_AVAILABLE = False


def require_rtc() -> None:
    """Raise if lerobot's RTC module isn't installed in this env."""
    if not _RTC_AVAILABLE:
        raise ImportError(
            "lerobot.policies.rtc not available. Install lerobot>=0.5.1 "
            "(pinned in reflex-vla exporters extras)."
        )


# ──────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────
@dataclass
class RtcAdapterConfig:
    """Config knobs for RTC behavior. Populated from per-embodiment YAML
    (see features/serve/per-embodiment-configs.md) or CLI flags.

    Open design questions (see design review doc):
    - latency_percentile: p95 vs p99 — TBD per LeRobot maintainer
    - prefix_weight_schedule: always trailing-zero or parameterizable — TBD
    - guidance_space: 'normalized' (our default) vs 'processed' — TBD
    """

    enabled: bool = False
    replan_hz: float = 20.0
    execute_hz: float = 100.0
    rtc_execution_horizon: int = 10  # actions locked to old chunk during replan
    latency_percentile: int = 95      # p95 default; p99 if maintainer recommends
    cold_start_discard: int = 10      # first N chunks NOT recorded in latency tracker
    guidance_space: str = "normalized"  # 'normalized' | 'processed'
    gripper_dim_indices: list[int] = field(default_factory=list)
    skip_gripper_smoothing: bool = True


# ──────────────────────────────────────────────────────────────────
# Latency tracking
# ──────────────────────────────────────────────────────────────────
class LatencyTracker:
    """Rolling-window percentile estimator for inference latency.

    Feeds RTC's internal ``update_latency`` path. Conservative percentile
    (p95 default) guards against under-sizing the replan budget.
    """

    def __init__(self, window_size: int = 50, percentile: int = 95, discard_first: int = 10):
        self._samples: list[float] = []
        self.window_size = window_size
        self.percentile = percentile
        self.discard_first = discard_first
        self._seen = 0

    def record(self, latency_s: float) -> None:
        """Record one inference wall-clock. Discards the first N (cold start)."""
        self._seen += 1
        if self._seen <= self.discard_first:
            return
        self._samples.append(latency_s)
        if len(self._samples) > self.window_size:
            self._samples.pop(0)

    def estimate(self) -> float:
        """Return conservative latency estimate for the scheduler."""
        if not self._samples:
            return 0.1  # 100ms fallback before any warm samples
        return float(np.percentile(self._samples, self.percentile))

    def summary(self) -> dict[str, float]:
        if not self._samples:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "n": 0}
        arr = np.array(self._samples)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "n": len(self._samples),
        }


# ──────────────────────────────────────────────────────────────────
# Policy protocol
# ──────────────────────────────────────────────────────────────────
class RtcCompatiblePolicy(Protocol):
    """Minimum interface an RtcAdapter expects from a policy.

    Pi05DecomposedInference, Pi0OnnxServer, and SmolVLANativeServer all
    implement ``predict_action_chunk(**kwargs) -> np.ndarray`` returning
    shape ``(B, chunk_size, action_dim)``. RTC wraps this call.
    """

    def predict_action_chunk(self, **kwargs: Any) -> np.ndarray: ...


# ──────────────────────────────────────────────────────────────────
# RtcAdapter
# ──────────────────────────────────────────────────────────────────
class RtcAdapter:
    """Reflex-side wrapper around ``lerobot.policies.rtc.RTCProcessor``.

    Owns:
    - config parsing from YAML / CLI / per-embodiment overrides
    - episode-id reset hooks (SDK ``client.reset()`` calls)
    - interop with existing ``ActionChunkBuffer`` (buffer holds processed /
      denormalized actions; RTC internally holds original / normalized)
    - latency tracker feeding RTC's scheduler
    - gripper-dim bypass for binary components

    Does NOT own:
    - the RTC math itself (delegated to lerobot's processor)
    - denormalization (handled by the policy's postprocessor downstream)
    - robot-side safety clamping (Guard wedge)

    Usage::

        adapter = RtcAdapter(
            policy=decomposed_server,
            action_buffer=buf,
            config=RtcAdapterConfig(enabled=True, replan_hz=20, execute_hz=100),
        )
        actions = adapter.predict_chunk_with_rtc(batch)
        adapter.merge_and_update(actions, elapsed_time=latency_s)
    """

    def __init__(
        self,
        policy: RtcCompatiblePolicy,
        action_buffer: ActionChunkBuffer,
        config: RtcAdapterConfig | None = None,
    ):
        if config is None:
            config = RtcAdapterConfig()
        if config.enabled:
            require_rtc()

        self.policy = policy
        self.buffer = action_buffer
        self.config = config

        self.latency = LatencyTracker(
            percentile=config.latency_percentile,
            discard_first=config.cold_start_discard,
        )
        # Lerobot RTCProcessor — only constructed when enabled + dep available.
        # TODO(rtc): finalize constructor args after Part A.2 of design review.
        self._processor: Any = None
        if config.enabled:
            # TODO(rtc): real construction — signature pending maintainer confirmation
            # self._processor = RTCProcessor(
            #     execution_horizon=config.rtc_execution_horizon,
            #     prefix_weight_schedule=...,
            # )
            pass

        self._active_episode_id: str | None = None
        self._chunk_count: int = 0

    # ---- Public API ---------------------------------------------

    def predict_chunk_with_rtc(self, batch: dict[str, Any]) -> np.ndarray:
        """Run one inference with RTC guidance applied.

        Steps:
        1. Get latency estimate from tracker → scheduler input
        2. Call policy.predict_action_chunk(**batch) → raw action chunk
        3. Pass through lerobot.RTCProcessor for guidance correction
        4. Return denormalized chunk ready for ActionChunkBuffer.push

        TODO(rtc): implement steps 1-4 after Part A (all 4 questions) resolves
        from design review.
        """
        raise NotImplementedError(
            "RtcAdapter.predict_chunk_with_rtc: pending Part A design review. "
            "See reflex_context/04_product/rtc_lerobot_design_review.md"
        )

    def merge_and_update(
        self,
        actions: np.ndarray,
        elapsed_time: float,
    ) -> None:
        """Push the new chunk into the action buffer; record inference latency.

        TODO(rtc): exact merge semantics depend on Part A.3 (normalized vs
        processed guidance space). Implemented as pass-through for now.
        """
        self.latency.record(elapsed_time)
        self._chunk_count += 1
        # Pass-through — pending full implementation
        self.buffer.push_chunk(actions)

    def reset(self, episode_id: str | None = None) -> None:
        """Reset RTC state at episode boundary.

        Called by SDK ``client.reset()`` hooks or when the server detects a
        new ``episode_id`` via ``/act`` params.
        """
        self._active_episode_id = episode_id
        self._chunk_count = 0
        # TODO(rtc): call self._processor.reset() when processor is wired
        logger.info("[rtc] reset — new episode_id=%s", episode_id)

    # ---- Introspection ------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Stats for logging / metrics. Consumed by Prometheus exporter
        (Phase 0.5.8) + ``reflex monitor``."""
        return {
            "enabled": self.config.enabled,
            "chunk_count": self._chunk_count,
            "active_episode_id": self._active_episode_id,
            "latency": self.latency.summary(),
            "rtc_available": _RTC_AVAILABLE,
        }


__all__ = [
    "RtcAdapter",
    "RtcAdapterConfig",
    "LatencyTracker",
    "RtcCompatiblePolicy",
    "require_rtc",
]

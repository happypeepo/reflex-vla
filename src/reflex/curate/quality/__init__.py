"""Per-episode quality scoring for the Curate wedge.

Composes 4 signals (success / smoothness / efficiency / coverage) into a
[0, 1] quality_score that downstream consumers use to rank episodes for
inclusion in published datasets. Rule-based v1; ML classifier v2 deferred.

Spec + research sidecar:
    reflex_context/features/08_curate/_curation/quality-scoring.md
    reflex_context/features/08_curate/_curation/quality-scoring_research.md

Submodules:
    signals       — the 4 signal functions
    composite     — quality_score() composite + result type
    per_embodiment — weight overrides per embodiment
    baseline      — BASELINE_MEDIAN_STEPS table for the efficiency signal
"""
from __future__ import annotations

from reflex.curate.quality.composite import (
    QualityResult,
    QualityWeights,
    quality_from_jsonl_rows,
    quality_score,
)
from reflex.curate.quality.per_embodiment import (
    DEFAULT_WEIGHTS,
    EMBODIMENT_WEIGHTS,
    weights_for,
)
from reflex.curate.quality.signals import (
    coverage_signal,
    efficiency_signal,
    policy_smoothness_signal,
    success_signal,
)

__all__ = [
    "DEFAULT_WEIGHTS",
    "EMBODIMENT_WEIGHTS",
    "QualityResult",
    "QualityWeights",
    "coverage_signal",
    "efficiency_signal",
    "policy_smoothness_signal",
    "quality_from_jsonl_rows",
    "quality_score",
    "success_signal",
    "weights_for",
]

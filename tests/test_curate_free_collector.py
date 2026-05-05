"""Tests for src/reflex/curate/free_collector.py — Free-tier collector."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from reflex.curate import consent as curate_consent
from reflex.curate.free_collector import FreeContributorCollector, _has_nan_or_inf
from reflex.pro.data_collection import CollectedEvent


def _mk_event(**overrides: Any) -> CollectedEvent:
    defaults: dict[str, Any] = dict(
        timestamp="2026-05-05T00:00:00Z",
        episode_id="ep-001",
        state_vec=[0.1, 0.2, 0.3],
        action_chunk=[[0.1, 0.2], [0.3, 0.4]],
        reward_proxy=1.0,
        image_b64=None,
        instruction_hash="abc",
        instruction_raw=None,
        metadata={},
    )
    defaults.update(overrides)
    return CollectedEvent(**defaults)


# ── _has_nan_or_inf ──────────────────────────────────────────────────────────


def test_has_nan_or_inf_flat() -> None:
    assert _has_nan_or_inf([1.0, 2.0, math.nan])
    assert _has_nan_or_inf([1.0, math.inf, 3.0])
    assert not _has_nan_or_inf([1.0, 2.0, 3.0])


def test_has_nan_or_inf_nested() -> None:
    assert _has_nan_or_inf([[1.0, 2.0], [3.0, math.nan]])
    assert not _has_nan_or_inf([[1.0, 2.0], [3.0, 4.0]])


# ── FreeContributorCollector ─────────────────────────────────────────────────


def test_requires_contributor_id() -> None:
    with pytest.raises(ValueError, match="contributor_id"):
        FreeContributorCollector(contributor_id="", data_dir="/tmp/x")


def test_record_tags_with_contributor_id(tmp_path: Path) -> None:
    collector = FreeContributorCollector(
        contributor_id="free_abc_123",
        tier="free",
        data_dir=tmp_path / "queue",
    )
    collector.start()
    try:
        ev = _mk_event(metadata={"existing_key": "existing_value"})
        collector.record(ev)
        # Force a flush
        collector.stop()
    finally:
        # idempotent stop
        pass

    files = list((tmp_path / "queue").glob("*.jsonl"))
    assert len(files) == 1
    content = files[0].read_text()
    assert "contributor_id" in content
    assert "free_abc_123" in content
    assert "existing_key" in content  # metadata pre-existing entries are preserved
    assert "existing_value" in content


def test_record_drops_nan_actions(tmp_path: Path) -> None:
    collector = FreeContributorCollector(
        contributor_id="free_abc_123",
        data_dir=tmp_path / "queue",
    )
    collector.start()
    try:
        good = _mk_event(episode_id="ep-good")
        bad = _mk_event(
            episode_id="ep-bad",
            action_chunk=[[1.0, math.nan], [3.0, 4.0]],
        )
        collector.record(good)
        collector.record(bad)
    finally:
        collector.stop()

    assert collector.events_recorded == 1  # only "good"
    assert collector.events_dropped >= 1  # "bad" dropped


def test_record_drops_inf_state(tmp_path: Path) -> None:
    collector = FreeContributorCollector(
        contributor_id="free_abc_123",
        data_dir=tmp_path / "queue",
    )
    collector.start()
    try:
        bad = _mk_event(
            episode_id="ep-bad",
            state_vec=[1.0, math.inf, 3.0],
        )
        collector.record(bad)
    finally:
        collector.stop()

    assert collector.events_recorded == 0
    assert collector.events_dropped >= 1


def test_from_consent_loads_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    consent_path = tmp_path / "consent.json"
    queue_dir = tmp_path / "queue"
    curate_consent.save(tier="free", contributor_id="free_aaa_bbb", path=consent_path)

    collector = FreeContributorCollector.from_consent(
        consent_path=consent_path,
        data_dir=queue_dir,
    )
    assert collector.contributor_id == "free_aaa_bbb"
    assert collector.tier == "free"


def test_from_consent_raises_when_not_opted_in(tmp_path: Path) -> None:
    consent_path = tmp_path / "consent.json"
    queue_dir = tmp_path / "queue"
    with pytest.raises(curate_consent.ConsentNotFound):
        FreeContributorCollector.from_consent(
            consent_path=consent_path,
            data_dir=queue_dir,
        )

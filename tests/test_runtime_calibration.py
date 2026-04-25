"""Tests for src/reflex/runtime/calibration.py — Phase 1 auto-calibration Day 1.

Covers: HardwareFingerprint construction + matching + serialization,
MeasurementQuality + MeasurementContext invariants, CalibrationEntry
validation + age + staleness, CalibrationCache schema-v1 invariants
(first field, key shape), lookup with/without fingerprint guard,
load/save round-trip + atomic-write semantics, schema_version refusal
on out-of-range values.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from reflex.runtime.calibration import (
    DEFAULT_STALE_AFTER_DAYS,
    SCHEMA_VERSION,
    CalibrationCache,
    CalibrationEntry,
    HardwareFingerprint,
    MeasurementContext,
    MeasurementQuality,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk_fp(**overrides) -> HardwareFingerprint:
    defaults = dict(
        gpu_uuid="GPU-abc-123",
        gpu_name="NVIDIA A10G",
        driver_version_major=535,
        driver_version_minor=129,
        cuda_version_major=12,
        cuda_version_minor=2,
        kernel_release="6.1.0",
        cpu_count=8,
        ram_gb=32,
        reflex_version="0.5.0",
    )
    defaults.update(overrides)
    return HardwareFingerprint(**defaults)


def _mk_quality(**overrides) -> MeasurementQuality:
    defaults = dict(
        warmup_iters=10,
        measurement_iters=100,
        median_ms=42.0,
        p99_ms=51.0,
        n_outliers_dropped=2,
        quality_score=0.92,
    )
    defaults.update(overrides)
    return MeasurementQuality(**defaults)


def _mk_context() -> MeasurementContext:
    return MeasurementContext(
        ort_version="1.20.1",
        torch_version="2.5.1",
        numpy_version="1.26.4",
        onnx_version="1.16.0",
    )


def _mk_entry(**overrides) -> CalibrationEntry:
    defaults = dict(
        chunk_size=50,
        nfe=4,
        latency_compensation_ms=42.0,
        provider="TensorrtExecutionProvider",
        variant="fp16",
        measurement_quality=_mk_quality(),
        measurement_context=_mk_context(),
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
    )
    defaults.update(overrides)
    return CalibrationEntry(**defaults)


# ---------------------------------------------------------------------------
# HardwareFingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_matches_default_tolerance_ignores_kernel_release():
    fp1 = _mk_fp(kernel_release="6.1.0")
    fp2 = _mk_fp(kernel_release="6.1.5")
    assert fp1.matches(fp2)


def test_fingerprint_matches_default_tolerates_ram_rounding():
    fp1 = _mk_fp(ram_gb=32)
    fp2 = _mk_fp(ram_gb=33)
    assert fp1.matches(fp2)


def test_fingerprint_matches_default_rejects_ram_drift():
    fp1 = _mk_fp(ram_gb=32)
    fp2 = _mk_fp(ram_gb=64)
    assert not fp1.matches(fp2)


def test_fingerprint_strict_mode_requires_bitwise_equality():
    fp1 = _mk_fp(kernel_release="6.1.0")
    fp2 = _mk_fp(kernel_release="6.1.5")
    assert not fp1.matches(fp2, strict=True)


def test_fingerprint_matches_rejects_driver_major_drift():
    fp1 = _mk_fp(driver_version_major=535)
    fp2 = _mk_fp(driver_version_major=545)
    assert not fp1.matches(fp2)


def test_fingerprint_matches_rejects_gpu_uuid_change():
    fp1 = _mk_fp(gpu_uuid="GPU-abc-123")
    fp2 = _mk_fp(gpu_uuid="GPU-different")
    assert not fp1.matches(fp2)


def test_fingerprint_matches_rejects_reflex_version_drift():
    fp1 = _mk_fp(reflex_version="0.5.0")
    fp2 = _mk_fp(reflex_version="0.6.0")
    assert not fp1.matches(fp2)


def test_fingerprint_round_trip_via_dict():
    fp = _mk_fp()
    d = fp.to_dict()
    fp2 = HardwareFingerprint.from_dict(d)
    assert fp == fp2


def test_fingerprint_current_returns_a_valid_object():
    fp = HardwareFingerprint.current()
    assert isinstance(fp, HardwareFingerprint)
    # Sentinels populate when probes fail — never None / empty
    assert fp.gpu_uuid != ""
    assert fp.kernel_release != ""


# ---------------------------------------------------------------------------
# MeasurementQuality
# ---------------------------------------------------------------------------


def test_quality_rejects_score_below_zero():
    with pytest.raises(ValueError, match="quality_score"):
        MeasurementQuality(
            warmup_iters=10, measurement_iters=100, median_ms=42.0,
            p99_ms=51.0, n_outliers_dropped=0, quality_score=-0.1,
        )


def test_quality_rejects_score_above_one():
    with pytest.raises(ValueError, match="quality_score"):
        MeasurementQuality(
            warmup_iters=10, measurement_iters=100, median_ms=42.0,
            p99_ms=51.0, n_outliers_dropped=0, quality_score=1.5,
        )


def test_quality_accepts_boundary_scores():
    MeasurementQuality(warmup_iters=0, measurement_iters=1, median_ms=0.1,
                       p99_ms=0.1, n_outliers_dropped=0, quality_score=0.0)
    MeasurementQuality(warmup_iters=0, measurement_iters=1, median_ms=0.1,
                       p99_ms=0.1, n_outliers_dropped=0, quality_score=1.0)


def test_quality_rejects_zero_measurement_iters():
    with pytest.raises(ValueError, match="measurement_iters"):
        MeasurementQuality(
            warmup_iters=0, measurement_iters=0, median_ms=42.0,
            p99_ms=51.0, n_outliers_dropped=0, quality_score=1.0,
        )


def test_quality_rejects_negative_warmup_iters():
    with pytest.raises(ValueError, match="warmup_iters"):
        MeasurementQuality(
            warmup_iters=-1, measurement_iters=100, median_ms=42.0,
            p99_ms=51.0, n_outliers_dropped=0, quality_score=1.0,
        )


def test_quality_round_trip():
    q = _mk_quality()
    q2 = MeasurementQuality.from_dict(q.to_dict())
    assert q == q2


# ---------------------------------------------------------------------------
# MeasurementContext
# ---------------------------------------------------------------------------


def test_context_round_trip():
    c = _mk_context()
    c2 = MeasurementContext.from_dict(c.to_dict())
    assert c == c2


def test_context_current_returns_valid_object():
    c = MeasurementContext.current()
    assert isinstance(c, MeasurementContext)
    assert c.numpy_version != ""


# ---------------------------------------------------------------------------
# CalibrationEntry
# ---------------------------------------------------------------------------


def test_entry_rejects_zero_chunk_size():
    with pytest.raises(ValueError, match="chunk_size"):
        _mk_entry(chunk_size=0)


def test_entry_rejects_nfe_out_of_range():
    with pytest.raises(ValueError, match="nfe"):
        _mk_entry(nfe=0)
    with pytest.raises(ValueError, match="nfe"):
        _mk_entry(nfe=51)


def test_entry_rejects_negative_latency_comp():
    with pytest.raises(ValueError, match="latency_compensation_ms"):
        _mk_entry(latency_compensation_ms=-1.0)


def test_entry_rejects_empty_provider():
    with pytest.raises(ValueError, match="provider"):
        _mk_entry(provider="")


def test_entry_rejects_empty_variant():
    with pytest.raises(ValueError, match="variant"):
        _mk_entry(variant="")


def test_entry_round_trip_via_dict():
    e = _mk_entry()
    e2 = CalibrationEntry.from_dict(e.to_dict())
    assert e == e2


def test_entry_age_seconds_recent():
    e = _mk_entry()
    assert e.age_seconds() < 60


def test_entry_age_seconds_unparseable_timestamp_returns_inf():
    e = _mk_entry(timestamp="not-an-iso-date")
    assert e.age_seconds() == float("inf")


def test_entry_is_stale_after_threshold():
    old_ts = (datetime.now(timezone.utc) - timedelta(days=31)).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    e = _mk_entry(timestamp=old_ts)
    assert e.is_stale(max_age_days=DEFAULT_STALE_AFTER_DAYS)


def test_entry_is_not_stale_when_recent():
    e = _mk_entry()
    assert not e.is_stale()


# ---------------------------------------------------------------------------
# CalibrationCache — schema v1 invariants
# ---------------------------------------------------------------------------


def test_cache_to_dict_has_schema_version_field():
    cache = CalibrationCache(reflex_version="0.5.0")
    d = cache.to_dict()
    assert "schema_version" in d
    assert d["schema_version"] == SCHEMA_VERSION


def test_cache_make_key_format():
    assert CalibrationCache.make_key("franka", "abc123") == "franka::abc123"


def test_cache_make_key_rejects_empty_args():
    with pytest.raises(ValueError):
        CalibrationCache.make_key("", "abc123")
    with pytest.raises(ValueError):
        CalibrationCache.make_key("franka", "")


def test_cache_make_key_rejects_separator_in_args():
    with pytest.raises(ValueError, match="'::'"):
        CalibrationCache.make_key("fran::ka", "abc")
    with pytest.raises(ValueError, match="'::'"):
        CalibrationCache.make_key("franka", "abc::def")


def test_cache_record_then_lookup_roundtrip():
    cache = CalibrationCache(reflex_version="0.5.0")
    entry = _mk_entry()
    cache.record(embodiment="franka", model_hash="abc", entry=entry)
    out = cache.lookup(embodiment="franka", model_hash="abc")
    assert out == entry


def test_cache_lookup_returns_none_on_miss():
    cache = CalibrationCache(reflex_version="0.5.0")
    out = cache.lookup(embodiment="franka", model_hash="abc")
    assert out is None


def test_cache_lookup_with_fingerprint_guard():
    fp_a = _mk_fp(gpu_uuid="GPU-a")
    fp_b = _mk_fp(gpu_uuid="GPU-b")
    cache = CalibrationCache(
        reflex_version="0.5.0", hardware_fingerprint=fp_a,
    )
    cache.record(embodiment="franka", model_hash="abc", entry=_mk_entry())
    assert cache.lookup(
        embodiment="franka", model_hash="abc",
        require_fingerprint=fp_a,
    ) is not None
    assert cache.lookup(
        embodiment="franka", model_hash="abc",
        require_fingerprint=fp_b,
    ) is None


def test_cache_record_updates_calibration_date():
    cache = CalibrationCache(reflex_version="0.5.0", calibration_date="")
    cache.record(embodiment="franka", model_hash="abc", entry=_mk_entry())
    assert cache.calibration_date != ""


def test_cache_is_stale_when_fingerprint_mismatches():
    cache = CalibrationCache(
        reflex_version="0.5.0",
        calibration_date=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        hardware_fingerprint=_mk_fp(gpu_uuid="GPU-a"),
    )
    different = _mk_fp(gpu_uuid="GPU-b")
    assert cache.is_stale(different)


def test_cache_is_stale_when_no_fingerprint():
    cache = CalibrationCache(reflex_version="0.5.0")
    assert cache.is_stale(_mk_fp())


def test_cache_is_not_stale_when_fingerprint_matches_and_recent():
    fp = _mk_fp()
    cache = CalibrationCache(
        reflex_version="0.5.0",
        calibration_date=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        hardware_fingerprint=fp,
    )
    assert not cache.is_stale(fp)


def test_cache_is_stale_when_old():
    fp = _mk_fp()
    old_date = (datetime.now(timezone.utc) - timedelta(days=31)).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    cache = CalibrationCache(
        reflex_version="0.5.0",
        calibration_date=old_date,
        hardware_fingerprint=fp,
    )
    assert cache.is_stale(fp)


# ---------------------------------------------------------------------------
# Save / Load round-trip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path):
    fp = _mk_fp()
    cache = CalibrationCache(
        reflex_version="0.5.0",
        calibration_date=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        hardware_fingerprint=fp,
    )
    cache.record(embodiment="franka", model_hash="abc", entry=_mk_entry())
    cache.record(embodiment="so100", model_hash="def", entry=_mk_entry(chunk_size=30))

    path = tmp_path / "cache.json"
    cache.save(path)
    assert path.exists()
    loaded = CalibrationCache.load(path)
    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.hardware_fingerprint == fp
    assert loaded.lookup(embodiment="franka", model_hash="abc").chunk_size == 50
    assert loaded.lookup(embodiment="so100", model_hash="def").chunk_size == 30


def test_save_writes_atomically(tmp_path):
    cache = CalibrationCache(reflex_version="0.5.0")
    path = tmp_path / "cache.json"
    cache.save(path)
    assert path.exists()
    assert not path.with_suffix(".json.tmp").exists()


def test_load_or_empty_returns_fresh_cache_when_missing(tmp_path):
    cache = CalibrationCache.load_or_empty(tmp_path / "nonexistent.json")
    assert cache.schema_version == SCHEMA_VERSION
    assert cache.entries == {}


def test_load_or_empty_returns_loaded_cache_when_exists(tmp_path):
    path = tmp_path / "cache.json"
    cache = CalibrationCache(
        reflex_version="0.5.0",
        calibration_date=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        hardware_fingerprint=_mk_fp(),
    )
    cache.record(embodiment="franka", model_hash="abc", entry=_mk_entry())
    cache.save(path)
    loaded = CalibrationCache.load_or_empty(path)
    assert loaded.lookup(embodiment="franka", model_hash="abc") is not None


def test_load_refuses_future_schema_version(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION + 1,
        "reflex_version": "0.6.0",
        "entries": {},
    }))
    with pytest.raises(ValueError, match="schema_version"):
        CalibrationCache.load(path)


def test_load_refuses_zero_schema_version(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({
        "schema_version": 0,
        "entries": {},
    }))
    with pytest.raises(ValueError, match="schema_version"):
        CalibrationCache.load(path)


def test_save_creates_parent_directory(tmp_path):
    nested = tmp_path / "deeply" / "nested" / "path"
    cache = CalibrationCache(reflex_version="0.5.0")
    cache.save(nested / "cache.json")
    assert (nested / "cache.json").exists()

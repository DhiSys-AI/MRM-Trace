"""Unit tests for classify_suitability."""

import pandas as pd
import pytest

from mrm_trace.analyser.suitability import classify_suitability


def _retention_summary(region_type, p50_s, p99_s):
    return pd.DataFrame([{
        "region_type": region_type,
        "n_regions": 1,
        "retention_p50_s": p50_s,
        "retention_p90_s": p99_s,
        "retention_p99_s": p99_s,
        "retention_mean_s": (p50_s + p99_s) / 2,
    }])


def _write_once(region_type, ratio):
    return pd.DataFrame([{
        "region_id": 1,
        "region_type": region_type,
        "n_written_addresses": 1,
        "n_write_once_addresses": 1 if ratio >= 1.0 else 0,
        "write_once_ratio": ratio,
    }])


def _read_freq(region_type):
    return pd.DataFrame([{
        "region_id": 1,
        "region_type": region_type,
        "total_reads": 10,
        "total_writes": 1,
        "total_accesses": 11,
        "read_fraction": 10 / 11,
        "reads_per_write": 10.0,
    }])


# ── classification thresholds ─────────────────────────────────────────────────

def test_weight_like_region_is_high_mrm():
    ret = _retention_summary("weight", p50_s=30.0, p99_s=30.0)
    wo  = _write_once("weight", 1.0)
    rf  = _read_freq("weight")
    result = classify_suitability(ret, wo, rf)
    assert result[result["region_type"] == "weight"].iloc[0]["suitability"] == "high_mrm"


def test_medium_retention_is_medium_mrm():
    ret = _retention_summary("activation", p50_s=2.0, p99_s=5.0)
    wo  = _write_once("activation", 0.6)
    rf  = _read_freq("activation")
    result = classify_suitability(ret, wo, rf)
    assert result[result["region_type"] == "activation"].iloc[0]["suitability"] == "medium_mrm"


def test_short_retention_is_low_mrm():
    ret = _retention_summary("kv_cache", p50_s=0.05, p99_s=0.1)
    wo  = _write_once("kv_cache", 1.0)
    rf  = _read_freq("kv_cache")
    result = classify_suitability(ret, wo, rf)
    assert result[result["region_type"] == "kv_cache"].iloc[0]["suitability"] == "low_mrm"


def test_low_write_once_with_long_retention_is_low_mrm():
    ret = _retention_summary("runtime", p50_s=20.0, p99_s=30.0)
    wo  = _write_once("runtime", 0.1)
    rf  = _read_freq("runtime")
    result = classify_suitability(ret, wo, rf)
    assert result[result["region_type"] == "runtime"].iloc[0]["suitability"] == "low_mrm"


def test_required_columns_present():
    ret = _retention_summary("weight", p50_s=30.0, p99_s=30.0)
    wo  = _write_once("weight", 1.0)
    rf  = _read_freq("weight")
    result = classify_suitability(ret, wo, rf)
    for col in ("region_type", "write_once_ratio_mean",
                "retention_p99_s", "retention_p50_s", "suitability"):
        assert col in result.columns


def test_boundary_high_mrm_exact_thresholds():
    # Exactly at threshold — should be high_mrm
    ret = _retention_summary("weight", p50_s=10.0, p99_s=10.0)
    wo  = _write_once("weight", 0.8)
    rf  = _read_freq("weight")
    result = classify_suitability(ret, wo, rf)
    assert result.iloc[0]["suitability"] == "high_mrm"

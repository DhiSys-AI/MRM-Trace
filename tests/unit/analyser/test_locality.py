"""Unit tests for compute_locality."""

import pandas as pd
import pytest

from mrm_trace.analyser.locality import compute_locality


def _trace(*rows):
    return pd.DataFrame(rows)


def _row(region_id, region_type, address, timestamp_ns=0):
    return {
        "region_id": region_id,
        "region_type": region_type,
        "op_type": "load",
        "timestamp_ns": timestamp_ns,
        "address": address,
        "address_page": address >> 12,
    }


def test_single_page_accesses_have_zero_stride():
    trace = _trace(
        _row(1, "weight", 0x1000, 0),
        _row(1, "weight", 0x1000, 1),
        _row(1, "weight", 0x1000, 2),
    )
    per_region, _ = compute_locality(trace)
    assert per_region.iloc[0]["mean_stride_bytes"] == pytest.approx(0.0)
    assert per_region.iloc[0]["same_page_fraction"] == pytest.approx(1.0)


def test_stride_computed_as_absolute_difference():
    trace = _trace(
        _row(1, "weight", 0x1000, 0),
        _row(1, "weight", 0x2000, 1),
    )
    per_region, _ = compute_locality(trace)
    # |0x2000 - 0x1000| = 0x1000 = 4096
    assert per_region.iloc[0]["mean_stride_bytes"] == pytest.approx(4096.0)


def test_same_page_fraction_zero_when_always_different_pages():
    trace = _trace(
        _row(1, "weight", 0x0000, 0),
        _row(1, "weight", 0x1000, 1),
        _row(1, "weight", 0x2000, 2),
    )
    per_region, _ = compute_locality(trace)
    assert per_region.iloc[0]["same_page_fraction"] == pytest.approx(0.0)


def test_single_access_has_zero_stats():
    trace = _trace(_row(1, "weight", 0x1000))
    per_region, _ = compute_locality(trace)
    r = per_region.iloc[0]
    assert r["mean_stride_bytes"] == 0.0
    assert r["same_page_fraction"] == 0.0


def test_type_summary_averages_per_region_stats():
    trace = _trace(
        _row(1, "weight", 0x1000, 0),
        _row(1, "weight", 0x1000, 1),  # same page — stride=0
        _row(2, "weight", 0x1000, 0),
        _row(2, "weight", 0x2000, 1),  # different page — stride=4096
    )
    _, type_summary = compute_locality(trace)
    ws = type_summary[type_summary["region_type"] == "weight"].iloc[0]
    # mean of (0.0 + 4096.0)/2 = 2048
    assert ws["mean_stride_bytes"] == pytest.approx(2048.0)


def test_required_columns_per_region():
    trace = _trace(_row(1, "weight", 0x1000, 0), _row(1, "weight", 0x2000, 1))
    per_region, _ = compute_locality(trace)
    for col in ("region_id", "region_type", "n_accesses",
                "mean_stride_bytes", "median_stride_bytes", "same_page_fraction"):
        assert col in per_region.columns

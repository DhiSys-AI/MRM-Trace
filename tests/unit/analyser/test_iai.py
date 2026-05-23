"""Unit tests for compute_iai."""

import pandas as pd
import pytest

from mrm_trace.analyser.iai import compute_iai


def _trace(*rows):
    return pd.DataFrame(rows)


def _row(region_type, address, timestamp_ns, op_type="load", region_id=1):
    return {
        "region_id": region_id,
        "region_type": region_type,
        "op_type": op_type,
        "timestamp_ns": timestamp_ns,
        "address": address,
        "address_page": address >> 12,
    }


def test_single_address_two_accesses_one_interval():
    trace = _trace(
        _row("weight", 0x1000, 0),
        _row("weight", 0x1000, 1_000_000),
    )
    result = compute_iai(trace)
    assert len(result) == 1
    assert result.iloc[0]["n_intervals"] == 1
    assert result.iloc[0]["iai_p50_ns"] == 1_000_000


def test_iai_groups_by_exact_address_not_page():
    # Two KV blocks on the same 4K page, 256 bytes apart
    trace = _trace(
        _row("kv_cache", 0xAA00_0000_0000,       0,         region_id=2),
        _row("kv_cache", 0xAA00_0000_0000,       10_000_000, region_id=2),
        _row("kv_cache", 0xAA00_0000_0100,       5_000_000,  region_id=3),
        _row("kv_cache", 0xAA00_0000_0100,       15_000_000, region_id=3),
    )
    result = compute_iai(trace)
    kv = result[result["region_type"] == "kv_cache"].iloc[0]
    # Each address has 1 interval: address 0 → 10ms, address +0x100 → 10ms
    assert kv["n_intervals"] == 2
    assert kv["iai_p50_ns"] == 10_000_000


def test_single_access_per_address_produces_no_intervals():
    trace = _trace(
        _row("weight", 0x1000, 0),
        _row("weight", 0x2000, 0),
    )
    result = compute_iai(trace)
    assert len(result) == 0


def test_multiple_region_types_separated():
    trace = _trace(
        _row("weight",   0x1000, 0),
        _row("weight",   0x1000, 100),
        _row("kv_cache", 0x2000, 0,   region_id=2),
        _row("kv_cache", 0x2000, 200, region_id=2),
    )
    result = compute_iai(trace)
    types = set(result["region_type"])
    assert "weight" in types
    assert "kv_cache" in types


def test_required_columns_present():
    trace = _trace(
        _row("weight", 0x1000, 0),
        _row("weight", 0x1000, 100),
    )
    result = compute_iai(trace)
    for col in ("region_type", "n_intervals", "iai_p50_ns",
                "iai_p90_ns", "iai_p99_ns", "iai_mean_ns"):
        assert col in result.columns


def test_zero_delta_timestamps_ignored():
    # Duplicate timestamps should not produce zero-width intervals
    trace = _trace(
        _row("weight", 0x1000, 0),
        _row("weight", 0x1000, 0),   # same timestamp
        _row("weight", 0x1000, 1_000_000),
    )
    result = compute_iai(trace)
    # Only 1 non-zero interval: 0 → 1_000_000
    assert result.iloc[0]["n_intervals"] == 1

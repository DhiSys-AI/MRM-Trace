"""Unit tests for compute_read_freq."""

import pandas as pd
import pytest

from mrm_trace.analyser.read_freq import compute_read_freq


def _trace(*rows):
    return pd.DataFrame(rows)


def _row(region_id, region_type, op_type, timestamp_ns=0, address=0x1000):
    return {
        "region_id": region_id,
        "region_type": region_type,
        "op_type": op_type,
        "timestamp_ns": timestamp_ns,
        "address": address,
        "address_page": address >> 12,
    }


def test_counts_reads_and_writes():
    trace = _trace(
        _row(1, "weight", "store"),
        _row(1, "weight", "load"),
        _row(1, "weight", "load"),
    )
    result = compute_read_freq(trace)
    r = result.iloc[0]
    assert r["total_writes"] == 1
    assert r["total_reads"] == 2
    assert r["total_accesses"] == 3


def test_read_fraction():
    trace = _trace(
        _row(1, "weight", "store"),
        _row(1, "weight", "load"),
    )
    result = compute_read_freq(trace)
    assert result.iloc[0]["read_fraction"] == pytest.approx(0.5)


def test_reads_per_write():
    trace = _trace(
        _row(1, "weight", "store"),
        _row(1, "weight", "load"),
        _row(1, "weight", "load"),
        _row(1, "weight", "load"),
    )
    result = compute_read_freq(trace)
    assert result.iloc[0]["reads_per_write"] == pytest.approx(3.0)


def test_read_only_region_reads_per_write_zero():
    trace = _trace(
        _row(1, "weight", "load"),
        _row(1, "weight", "load"),
    )
    result = compute_read_freq(trace)
    assert result.iloc[0]["reads_per_write"] == pytest.approx(0.0)


def test_write_only_region_read_fraction_zero():
    trace = _trace(_row(1, "weight", "store"))
    result = compute_read_freq(trace)
    assert result.iloc[0]["read_fraction"] == pytest.approx(0.0)


def test_multiple_regions():
    trace = _trace(
        _row(1, "weight",   "store", address=0x1000),
        _row(1, "weight",   "load",  address=0x1000),
        _row(2, "kv_cache", "store", address=0x2000),
        _row(2, "kv_cache", "load",  address=0x2000),
        _row(2, "kv_cache", "load",  address=0x2000),
        _row(2, "kv_cache", "load",  address=0x2000),
        _row(2, "kv_cache", "load",  address=0x2000),
        _row(2, "kv_cache", "load",  address=0x2000),
    )
    result = compute_read_freq(trace).set_index("region_id")
    assert result.loc[1, "reads_per_write"] == pytest.approx(1.0)
    assert result.loc[2, "reads_per_write"] == pytest.approx(5.0)


def test_required_columns_present():
    trace = _trace(_row(1, "weight", "store"))
    result = compute_read_freq(trace)
    for col in ("region_id", "region_type", "total_reads", "total_writes",
                "total_accesses", "read_fraction", "reads_per_write"):
        assert col in result.columns

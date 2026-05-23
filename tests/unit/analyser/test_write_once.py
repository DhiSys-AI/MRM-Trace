"""Unit tests for compute_write_once."""

import pandas as pd
import pytest

from mrm_trace.analyser.write_once import compute_write_once


def _trace(*rows):
    return pd.DataFrame(rows)


def _row(region_id, region_type, op_type, address):
    return {
        "region_id": region_id,
        "region_type": region_type,
        "op_type": op_type,
        "address": address,
        "address_page": address >> 12,
        "timestamp_ns": 0,
    }


def test_single_store_is_write_once():
    trace = _trace(_row(1, "weight", "store", 0x1000))
    result = compute_write_once(trace)
    assert result.iloc[0]["write_once_ratio"] == pytest.approx(1.0)


def test_two_stores_to_same_address_is_not_write_once():
    trace = _trace(
        _row(1, "weight", "store", 0x1000),
        _row(1, "weight", "store", 0x1000),
    )
    result = compute_write_once(trace)
    assert result.iloc[0]["write_once_ratio"] == pytest.approx(0.0)


def test_two_addresses_one_written_twice():
    trace = _trace(
        _row(1, "weight", "store", 0x1000),
        _row(1, "weight", "store", 0x2000),
        _row(1, "weight", "store", 0x2000),  # 0x2000 written twice
    )
    result = compute_write_once(trace)
    # 0x1000 write-once, 0x2000 not → ratio = 1/2
    assert result.iloc[0]["write_once_ratio"] == pytest.approx(0.5)


def test_loads_are_ignored():
    trace = _trace(
        _row(1, "weight", "store", 0x1000),
        _row(1, "weight", "load",  0x1000),
        _row(1, "weight", "load",  0x1000),
    )
    result = compute_write_once(trace)
    assert result.iloc[0]["write_once_ratio"] == pytest.approx(1.0)
    assert result.iloc[0]["n_written_addresses"] == 1


def test_no_stores_returns_empty():
    trace = _trace(_row(1, "weight", "load", 0x1000))
    result = compute_write_once(trace)
    assert len(result) == 0


def test_multiple_regions():
    trace = _trace(
        _row(1, "weight",   "store", 0x1000),
        _row(2, "kv_cache", "store", 0x2000),
        _row(2, "kv_cache", "store", 0x2000),  # region 2 writes same addr twice
    )
    result = compute_write_once(trace)
    by_id = result.set_index("region_id")["write_once_ratio"]
    assert by_id[1] == pytest.approx(1.0)
    assert by_id[2] == pytest.approx(0.0)


def test_required_columns_present():
    trace = _trace(_row(1, "weight", "store", 0x1000))
    result = compute_write_once(trace)
    for col in ("region_id", "region_type", "n_written_addresses",
                "n_write_once_addresses", "write_once_ratio"):
        assert col in result.columns

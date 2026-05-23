"""Unit tests for compute_working_set."""

import pandas as pd
import pytest

from mrm_trace.analyser.working_set import compute_working_set

_PAGE = 4096


def _trace(*rows):
    return pd.DataFrame(rows)


def _row(region_id, region_type, address):
    return {
        "region_id": region_id,
        "region_type": region_type,
        "op_type": "load",
        "timestamp_ns": 0,
        "address": address,
        "address_page": address >> 12,
    }


def test_unique_pages_counted():
    trace = _trace(
        _row(1, "weight", 0x0000),
        _row(1, "weight", 0x0000),  # same page
        _row(1, "weight", 0x1000),  # different page
    )
    per_region, _ = compute_working_set(trace)
    assert per_region.iloc[0]["unique_pages"] == 2


def test_working_set_bytes_is_pages_times_4096():
    trace = _trace(
        _row(1, "weight", 0x0000),
        _row(1, "weight", 0x1000),
        _row(1, "weight", 0x2000),
    )
    per_region, _ = compute_working_set(trace)
    assert per_region.iloc[0]["working_set_bytes"] == 3 * _PAGE


def test_type_summary_totals():
    trace = _trace(
        _row(1, "weight", 0x0000),
        _row(1, "weight", 0x1000),
        _row(2, "weight", 0x2000),
    )
    _, type_summary = compute_working_set(trace)
    ws = type_summary[type_summary["region_type"] == "weight"].iloc[0]
    assert ws["total_pages"] == 3
    assert ws["total_bytes"] == 3 * _PAGE
    assert ws["n_regions"] == 2


def test_multiple_types_in_summary():
    trace = _trace(
        _row(1, "weight",   0x1000),
        _row(2, "kv_cache", 0x2000),
    )
    _, type_summary = compute_working_set(trace)
    assert set(type_summary["region_type"]) == {"weight", "kv_cache"}


def test_per_region_required_columns():
    trace = _trace(_row(1, "weight", 0x1000))
    per_region, _ = compute_working_set(trace)
    for col in ("region_id", "region_type", "unique_pages", "working_set_bytes"):
        assert col in per_region.columns

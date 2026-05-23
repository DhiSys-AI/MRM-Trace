"""
Property-based tests for the analyser — invariants that must hold for all inputs.
"""

import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from mrm_trace.analyser.retention import compute_retention
from mrm_trace.analyser.write_once import compute_write_once
from mrm_trace.analyser.read_freq import compute_read_freq


def _rows(region_id, region_type, n_stores, n_loads, base_address=0x1000):
    rows = []
    for i in range(n_stores):
        rows.append({
            "region_id": region_id, "region_type": region_type,
            "op_type": "store", "timestamp_ns": i,
            "address": base_address + i * 64,
            "address_page": (base_address + i * 64) >> 12,
        })
    for j in range(n_loads):
        rows.append({
            "region_id": region_id, "region_type": region_type,
            "op_type": "load", "timestamp_ns": n_stores + j * 100,
            "address": base_address,
            "address_page": base_address >> 12,
        })
    return rows


@given(
    n_stores=st.integers(min_value=1, max_value=50),
    n_loads=st.integers(min_value=0, max_value=50),
)
@settings(max_examples=200, deadline=None)
def test_retention_ns_always_nonnegative(n_stores, n_loads):
    """retention_ns must never be negative regardless of access pattern."""
    trace = pd.DataFrame(_rows(1, "weight", n_stores, n_loads))
    per_region, _ = compute_retention(trace)
    assert (per_region["retention_ns"] >= 0).all()


@given(
    n_stores=st.integers(min_value=1, max_value=50),
    n_loads=st.integers(min_value=0, max_value=50),
)
@settings(max_examples=200, deadline=None)
def test_retention_s_consistent_with_ns(n_stores, n_loads):
    """retention_s must equal retention_ns / 1e9 (within floating-point tolerance)."""
    trace = pd.DataFrame(_rows(1, "weight", n_stores, n_loads))
    per_region, _ = compute_retention(trace)
    for _, row in per_region.iterrows():
        assert abs(row["retention_s"] - row["retention_ns"] / 1e9) < 1e-6


@given(
    n_stores=st.integers(min_value=1, max_value=30),
    n_loads=st.integers(min_value=0, max_value=30),
)
@settings(max_examples=200, deadline=None)
def test_write_once_ratio_between_0_and_1(n_stores, n_loads):
    """write_once_ratio must always be in [0, 1]."""
    trace = pd.DataFrame(_rows(1, "weight", n_stores, n_loads))
    result = compute_write_once(trace)
    if len(result) > 0:
        assert (result["write_once_ratio"] >= 0.0).all()
        assert (result["write_once_ratio"] <= 1.0).all()


@given(
    n_stores=st.integers(min_value=1, max_value=30),
    n_loads=st.integers(min_value=0, max_value=30),
)
@settings(max_examples=200, deadline=None)
def test_read_freq_total_accesses_is_sum_of_reads_and_writes(n_stores, n_loads):
    """total_accesses must equal total_reads + total_writes."""
    trace = pd.DataFrame(_rows(1, "weight", n_stores, n_loads))
    result = compute_read_freq(trace)
    for _, row in result.iterrows():
        assert row["total_accesses"] == row["total_reads"] + row["total_writes"]


@given(
    n_stores=st.integers(min_value=1, max_value=30),
    n_loads=st.integers(min_value=0, max_value=30),
)
@settings(max_examples=200, deadline=None)
def test_read_fraction_between_0_and_1(n_stores, n_loads):
    """read_fraction must always be in [0, 1]."""
    trace = pd.DataFrame(_rows(1, "weight", n_stores, n_loads))
    result = compute_read_freq(trace)
    assert (result["read_fraction"] >= 0.0).all()
    assert (result["read_fraction"] <= 1.0).all()

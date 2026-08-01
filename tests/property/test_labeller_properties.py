"""
Property-based tests for the labeller — uses Hypothesis.

These tests verify invariants that must hold for ALL valid inputs,
not just the specific cases tested in unit tests.
"""

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from mrm_trace.labeller.labeller import TraceLabeller

_VALID_TYPES = {"weight", "kv_cache", "activation", "runtime", "unknown"}
_VALID_OPS = {"load", "store"}
_VALID_POLICIES = {"keep", "skip", "error"}


def _row(address, symbol, op_type, phase, region_type="unknown"):
    page = address >> 12
    return {
        "run_id": "prop", "sample_id": 0, "timestamp_ns": 0,
        "phase": phase, "address": address, "address_page": page,
        "op_type": op_type, "symbol": symbol, "dso": "lib.so",
        "region_type": region_type, "region_id": 0,
        "access_latency_ns": None, "cache_level": None, "numa_node": None,
        "cpu_id": 0, "thread_id": 1, "token_index": None, "context_tokens": None,
    }


@given(
    address=st.integers(min_value=0, max_value=2**48 - 1),
    symbol=st.text(max_size=64),
    op_type=st.sampled_from(["load", "store"]),
    phase=st.one_of(
        st.none(),
        st.sampled_from(["weight_load", "generation", "teardown"]),
    ),
)
@settings(max_examples=200)
def test_label_always_yields_valid_region_type(address, symbol, op_type, phase):
    """Every labelled row must have a region_type from the valid set."""
    row = _row(address, symbol, op_type, phase)
    labeller = TraceLabeller(unknown_policy="keep")
    result = list(labeller.label([row]))
    assert len(result) == 1
    assert result[0]["region_type"] in _VALID_TYPES


@given(
    address=st.integers(min_value=0, max_value=2**48 - 1),
    symbol=st.text(max_size=64),
    op_type=st.sampled_from(["load", "store"]),
    phase=st.one_of(st.none(), st.sampled_from(["weight_load", "generation"])),
)
@settings(max_examples=200)
def test_region_id_always_nonnegative(address, symbol, op_type, phase):
    """region_id must never be negative."""
    row = _row(address, symbol, op_type, phase)
    labeller = TraceLabeller(unknown_policy="keep")
    result = list(labeller.label([row]))
    assert result[0]["region_id"] >= 0


@given(
    n=st.integers(min_value=1, max_value=20),
    base_address=st.integers(min_value=4096, max_value=2**40),
)
@settings(max_examples=100)
def test_weight_pages_get_one_region_each(n, base_address):
    """
    Distinct weight pages must get distinct region_ids, so retention has one
    sample per page rather than one sample in total (spec §6.1 asks for a
    distribution: p50/p90/p99/max/mean/std).
    """
    rows = [
        _row(base_address + i * 4096, "ggml_init", "store", "weight_load")
        for i in range(n)
    ]
    labeller = TraceLabeller()
    labelled = [r for r in labeller.label(rows) if r["region_type"] == "weight"]
    pages = {r["address_page"] for r in labelled}
    ids = {r["region_id"] for r in labelled}
    assert len(ids) == len(pages)


@given(
    n=st.integers(min_value=2, max_value=20),
    base_address=st.integers(min_value=4096, max_value=2**40),
)
@settings(max_examples=100)
def test_page_propagation_is_stable(n, base_address):
    """
    Once a page is classified, all subsequent rows on that page must have
    the same region_type (first-classification-wins invariant).
    """
    page_addr = (base_address >> 12) << 12  # align to page
    rows = [
        _row(page_addr, "ggml_init", "store", "weight_load"),  # classifies as weight
    ] + [
        _row(page_addr, "malloc", "load", None)  # would be runtime if unclassified
        for _ in range(n - 1)
    ]
    labeller = TraceLabeller()
    labelled = list(labeller.label(rows))
    types = [r["region_type"] for r in labelled]
    assert len(set(types)) == 1, (
        f"Page classification changed mid-stream: {types}. "
        "First-wins invariant violated."
    )


@given(
    n=st.integers(min_value=0, max_value=50),
    symbol=st.text(max_size=64),
    op_type=st.sampled_from(["load", "store"]),
)
@settings(max_examples=100)
def test_skip_policy_never_yields_unknown_rows(n, symbol, op_type):
    """With unknown_policy='skip', no unknown rows must appear in output."""
    rows = [
        _row(0x1000 + i * 4096, symbol, op_type, None)
        for i in range(n)
    ]
    labeller = TraceLabeller(unknown_policy="skip")
    result = list(labeller.label(rows))
    assert all(r["region_type"] != "unknown" for r in result)


@given(
    address=st.integers(min_value=4096, max_value=2**40),
    pre_type=st.sampled_from(["weight", "kv_cache", "activation", "runtime"]),
)
@settings(max_examples=100)
def test_pre_labelled_rows_pass_through_unchanged(address, pre_type):
    """Pre-labelled rows (region_type != 'unknown') must not be reclassified."""
    row = _row(address, "any_symbol", "load", None, region_type=pre_type)
    row["region_id"] = 5  # arbitrary pre-assigned id
    labeller = TraceLabeller()
    result = list(labeller.label([row]))
    assert result[0]["region_type"] == pre_type

"""Unit tests for TraceLabeller."""

import types

import pytest

from mrm_trace.labeller.labeller import TraceLabeller


def _row(**overrides):
    """Minimal canonical trace row with region_type=unknown."""
    base = {
        "run_id": "r1",
        "sample_id": 0,
        "timestamp_ns": 1_000,
        "phase": None,
        "address": 0x7F00_0000_0000,
        "address_page": 0x7F00_0000_0000 >> 12,
        "op_type": "load",
        "symbol": "",
        "dso": "",
        "region_type": "unknown",
        "region_id": 0,
        "access_latency_ns": None,
        "cache_level": None,
        "numa_node": None,
        "cpu_id": 0,
        "thread_id": 1,
        "token_index": None,
        "context_tokens": None,
    }
    base.update(overrides)
    return base


# ── constructor ───────────────────────────────────────────────────────────────

def test_invalid_unknown_policy_raises():
    with pytest.raises(ValueError, match="unknown_policy"):
        TraceLabeller(unknown_policy="ignore")


def test_valid_policies_accepted():
    for p in ("keep", "skip", "error"):
        TraceLabeller(unknown_policy=p)


# ── label() is a generator ────────────────────────────────────────────────────

def test_label_returns_generator():
    result = TraceLabeller().label(iter([]))
    assert isinstance(result, types.GeneratorType)


def test_empty_input_yields_nothing():
    rows = list(TraceLabeller().label(iter([])))
    assert rows == []


# ── phase-based weight classification ────────────────────────────────────────

def test_weight_load_store_classified_weight():
    row = _row(phase="weight_load", op_type="store", symbol="ggml_init",
               address=0x7F00_0000_0000, address_page=0x7F00_0000_0000 >> 12)
    rows = list(TraceLabeller().label([row]))
    assert rows[0]["region_type"] == "weight"


def test_weight_load_store_gets_region_id_1():
    row = _row(phase="weight_load", op_type="store", symbol="ggml_init",
               address=0x7F00_0000_0000, address_page=0x7F00_0000_0000 >> 12)
    rows = list(TraceLabeller().label([row]))
    assert rows[0]["region_id"] == 1


# ── symbol-based KV classification ───────────────────────────────────────────

def test_kv_cache_symbol_classified():
    row = _row(phase="generation", op_type="store", symbol="llama_kv_cache_update",
               address=0xAA00_0000_0000, address_page=0xAA00_0000_0000 >> 12)
    rows = list(TraceLabeller().label([row]))
    assert rows[0]["region_type"] == "kv_cache"


def test_kv_cache_region_id_is_2_when_no_weight_region():
    row = _row(phase="generation", op_type="store", symbol="llama_kv_cache_update",
               address=0xAA00_0000_0000, address_page=0xAA00_0000_0000 >> 12)
    rows = list(TraceLabeller().label([row]))
    assert rows[0]["region_id"] == 2


# ── address propagation ───────────────────────────────────────────────────────

def test_second_access_to_classified_page_inherits_type():
    page = 0x7F00_0000_0000 >> 12
    labeller = TraceLabeller()
    # First row: symbol classifies as weight
    rows = list(labeller.label([
        _row(phase="weight_load", op_type="store", symbol="ggml_init",
             address=0x7F00_0000_0000, address_page=page, timestamp_ns=100),
        # Second row: no matching symbol but same page
        _row(phase="generation", op_type="load", symbol="unknown_sym",
             address=0x7F00_0000_0000, address_page=page, sample_id=1, timestamp_ns=200),
    ]))
    assert rows[1]["region_type"] == "weight"
    assert rows[1]["region_id"] == 1


def test_propagated_region_id_matches_original():
    page = 0xBB00_0000_0000 >> 12
    labeller = TraceLabeller()
    rows = list(labeller.label([
        _row(symbol="llama_kv_cache_update", phase="generation", op_type="store",
             address=0xBB00_0000_0000, address_page=page, timestamp_ns=100),
        _row(symbol="", phase="generation", op_type="load",
             address=0xBB00_0000_0000, address_page=page, sample_id=1, timestamp_ns=200),
    ]))
    assert rows[0]["region_id"] == rows[1]["region_id"]


# ── pass-through for pre-labelled rows ───────────────────────────────────────

def test_pre_labelled_weight_row_passes_through():
    row = _row(region_type="weight", region_id=1, symbol="any")
    rows = list(TraceLabeller().label([row]))
    assert rows[0]["region_type"] == "weight"


def test_pre_labelled_region_type_not_changed():
    row = _row(region_type="kv_cache", region_id=5, symbol="malloc")
    rows = list(TraceLabeller().label([row]))
    assert rows[0]["region_type"] == "kv_cache"


# ── unknown_policy ────────────────────────────────────────────────────────────

def test_unknown_policy_keep_returns_unknown_rows():
    row = _row(symbol="totally_unknown_func", phase=None)
    rows = list(TraceLabeller(unknown_policy="keep").label([row]))
    assert len(rows) == 1
    assert rows[0]["region_type"] == "unknown"
    assert rows[0]["region_id"] == 0


def test_unknown_policy_skip_omits_unknown_rows():
    row = _row(symbol="totally_unknown_func", phase=None)
    rows = list(TraceLabeller(unknown_policy="skip").label([row]))
    assert rows == []


def test_unknown_policy_error_raises_on_unknown():
    row = _row(symbol="totally_unknown_func", phase=None)
    with pytest.raises(ValueError):
        list(TraceLabeller(unknown_policy="error").label([row]))


# ── other columns preserved ──────────────────────────────────────────────────

def test_all_original_columns_present_in_output():
    row = _row(phase="weight_load", op_type="store", symbol="ggml_init",
               address=0x7F00_0000_0000, address_page=0x7F00_0000_0000 >> 12)
    rows = list(TraceLabeller().label([row]))
    for key in row:
        assert key in rows[0]


# ── region_map() and kv_lifecycle() ──────────────────────────────────────────

def test_region_map_available_after_consuming_label():
    labeller = TraceLabeller()
    row = _row(phase="weight_load", op_type="store", symbol="ggml_init",
               address=0x7F00_0000_0000, address_page=0x7F00_0000_0000 >> 12)
    list(labeller.label([row]))
    region_map = labeller.region_map()
    assert len(region_map) == 1
    assert region_map[0]["region_type"] == "weight"


def test_kv_lifecycle_available_after_consuming_label():
    labeller = TraceLabeller()
    rows = [
        _row(symbol="llama_kv_cache_update", phase="generation", op_type="store",
             address=0xCC00_0000_0000, address_page=0xCC00_0000_0000 >> 12, timestamp_ns=100),
        _row(symbol="llama_kv_cache_update", phase="generation", op_type="load",
             address=0xCC00_0000_0000, address_page=0xCC00_0000_0000 >> 12,
             sample_id=1, timestamp_ns=500),
    ]
    list(labeller.label(rows))
    lc = labeller.kv_lifecycle()
    assert len(lc) == 1
    assert lc[0]["total_reads"] == 1
    assert lc[0]["lifetime_ns"] == 400

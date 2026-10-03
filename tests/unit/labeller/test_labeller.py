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


# ── weight comes from the GGUF mapping, never from a symbol ──────────────────

def _weight_labeller():
    lab = TraceLabeller()
    lab.register_address_ranges([{
        "region_type": "weight", "source": "mmap_gguf",
        "addr_start": 0x7F00_0000_0000, "addr_end": 0x7F00_0000_1000,
    }])
    return lab


def test_weight_range_classifies_weight():
    row = _row(phase="weight_load", op_type="store", symbol="ggml_init",
               address=0x7F00_0000_0000, address_page=0x7F00_0000_0000 >> 12)
    rows = list(_weight_labeller().label([row]))
    assert rows[0]["region_type"] == "weight"


def test_weight_range_row_gets_region_id_1():
    row = _row(phase="weight_load", op_type="store", symbol="ggml_init",
               address=0x7F00_0000_0000, address_page=0x7F00_0000_0000 >> 12)
    rows = list(_weight_labeller().label([row]))
    assert rows[0]["region_id"] == 1


def test_engine_symbol_alone_cannot_claim_weight():
    """
    Measured on a real run: of 26,617,751 rows a ggml_vec_dot rule labelled
    `weight`, only 114 were inside the mmap'd GGUF file.  The mapping is
    read-only, so a store can never be a weight address at all.
    """
    row = _row(phase="weight_load", op_type="store", symbol="ggml_init",
               address=0xDDDD_0000_0000, address_page=0xDDDD_0000_0000 >> 12)
    rows = list(TraceLabeller().label([row]))
    assert rows[0]["region_type"] != "weight"


# ── kv_cache comes from the engine's allocation, never from a symbol ─────────

def test_kv_symbol_alone_cannot_claim_kv_cache():
    """
    llama_kv_cache_* manages slot metadata, not the K/V tensors.  On a real run
    that rule tagged four granules — one in the main stack — which then
    absorbed 279,186 accesses from nearest_int, memmove and softmax.
    """
    row = _row(phase="generation", op_type="store", symbol="llama_kv_cache_update",
               address=0xAA00_0000_0000, address_page=0xAA00_0000_0000 >> 12)
    rows = list(TraceLabeller().label([row]))
    assert rows[0]["region_type"] != "kv_cache"


def test_region_ids_are_allocated_from_1():
    """Region ids are sequential per granule; no type-specific reserved ids."""
    lab = TraceLabeller()
    lab.register_address_ranges([{
        "region_type": "kv_cache", "source": "kv_allocation",
        "addr_start": 0xAA00_0000_0000, "addr_end": 0xAA00_0000_1000,
    }])
    row = _row(phase="generation", op_type="store",
               address=0xAA00_0000_0000, address_page=0xAA00_0000_0000 >> 12)
    rows = list(lab.label([row]))
    assert rows[0]["region_id"] == 1


# ── address propagation ───────────────────────────────────────────────────────

def test_second_access_to_classified_page_inherits_type():
    page = 0x7F00_0000_0000 >> 12
    labeller = _weight_labeller()
    # First row: the registered weight range classifies it
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
    labeller = _weight_labeller()
    row = _row(phase="weight_load", op_type="store", symbol="ggml_init",
               address=0x7F00_0000_0000, address_page=0x7F00_0000_0000 >> 12)
    list(labeller.label([row]))
    region_map = labeller.region_map()
    assert len(region_map) == 1
    assert region_map[0]["region_type"] == "weight"


def test_kv_lifecycle_available_after_consuming_label():
    labeller = TraceLabeller()
    labeller.register_address_ranges([{
        "region_type": "kv_cache", "source": "kv_allocation",
        "addr_start": 0xCC00_0000_0000, "addr_end": 0xCC00_0000_1000,
    }])
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


# ── evidence-ranked precedence ────────────────────────────────────────────────
#
# Address ranges from /proc/self/maps cover the whole address space.  With a
# first-source-wins rule every page is classified before symbol rules are
# consulted, so symbol-based labelling never runs — silently discarding the most
# specific evidence available.  These tests pin the ranking that keeps both
# mechanisms live.

_ADDR = 0x7F00_0000_0000
_PAGE = _ADDR >> 12


def _ranges(region_type, source, addr=_ADDR, size=0x1000):
    return [{
        "region_type": region_type, "source": source,
        "addr_start": addr, "addr_end": addr + size,
    }]


def _label_one(labeller, **overrides):
    return list(labeller.label([_row(**overrides)]))[0]


def test_a_kv_symbol_does_not_override_the_address_map():
    """
    This inversion was the bug: EV_SYMBOL_KV outranked the address map, so one
    llama_kv_cache_* sample claimed a whole granule of whatever it landed in.
    """
    lab = TraceLabeller()
    lab.register_address_ranges(_ranges("runtime", "anon_post_load"))
    row = _label_one(lab, symbol="llama_kv_cache_update", op_type="store")
    assert row["region_type"] == "runtime"


def test_gguf_mmap_range_beats_symbol():
    """A file-backed GGUF mapping is unambiguous — no symbol should override it."""
    lab = TraceLabeller()
    lab.register_address_ranges(_ranges("weight", "mmap_gguf"))
    row = _label_one(lab, symbol="llama_kv_cache_update", op_type="store")
    assert row["region_type"] == "weight"


def test_kv_symbol_beats_geometry_matched_kv_range():
    """Both say kv_cache; the point is the stronger source is recorded."""
    lab = TraceLabeller()
    lab.register_address_ranges(_ranges("kv_cache", "anon_kv_sized"))
    row = _label_one(lab, symbol="llama_kv_cache_update", op_type="store")
    assert row["region_type"] == "kv_cache"


def test_geometry_kv_range_beats_generic_symbol():
    lab = TraceLabeller()
    lab.register_address_ranges(_ranges("kv_cache", "anon_kv_sized"))
    row = _label_one(lab, symbol="malloc")
    assert row["region_type"] == "kv_cache"


def test_generic_symbol_beats_coarse_range():
    """A kernel's store destination is an activation buffer — that much holds."""
    lab = TraceLabeller()
    lab.register_address_ranges(_ranges("runtime", "anon"))
    row = _label_one(lab, symbol="ggml_vec_dot_q4_K", op_type="store")
    assert row["region_type"] == "activation"


def test_coarse_range_used_when_no_symbol_matches():
    lab = TraceLabeller()
    lab.register_address_ranges(_ranges("runtime", "thread_stack"))
    row = _label_one(lab, symbol="_some_unknown_symbol")
    assert row["region_type"] == "runtime"


def test_equal_evidence_keeps_first_classification():
    """Ties must not flip, or labelling becomes order-dependent."""
    lab = TraceLabeller()
    rows = [
        _row(symbol="ggml_vec_dot_q4_K", op_type="store", sample_id=0),  # → activation
        _row(symbol="malloc", op_type="store", sample_id=1),             # → runtime
    ]
    out = list(lab.label(rows))
    assert [r["region_type"] for r in out] == ["activation", "activation"]


def test_stronger_range_upgrades_weaker_range():
    lab = TraceLabeller()
    lab.register_address_ranges(_ranges("runtime", "anon"))
    lab.register_address_ranges(_ranges("weight", "mmap_gguf"))
    assert _label_one(lab)["region_type"] == "weight"


def test_weaker_range_does_not_downgrade_stronger():
    lab = TraceLabeller()
    lab.register_address_ranges(_ranges("weight", "mmap_gguf"))
    lab.register_address_ranges(_ranges("runtime", "anon"))
    assert _label_one(lab)["region_type"] == "weight"


# ── prescan_symbols ───────────────────────────────────────────────────────────

def test_prescan_makes_labelling_order_independent():
    """
    Without a prescan the first row on a page fixes its type, so a decisive
    symbol arriving later has no effect and the result depends on row order.
    """
    def _run(order, prescan):
        lab = TraceLabeller()
        lab.register_address_ranges(_ranges("runtime", "anon"))
        rows = [_row(**kw) for kw in order]
        if prescan:
            lab.prescan_symbols(rows)
        return [r["region_type"] for r in lab.label(rows)]

    symbol_first = [{"symbol": "ggml_vec_dot_q4_K", "op_type": "store", "sample_id": 0},
                    {"symbol": "", "op_type": "store", "sample_id": 1}]
    symbol_last = [{"symbol": "", "op_type": "store", "sample_id": 0},
                   {"symbol": "ggml_vec_dot_q4_K", "op_type": "store", "sample_id": 1}]

    # Without prescan the two orderings disagree on the first row.
    assert _run(symbol_first, prescan=False)[0] != _run(symbol_last, prescan=False)[0]

    # With prescan both orderings classify the whole page as kv_cache.
    assert _run(symbol_first, prescan=True) == ["activation", "activation"]
    assert _run(symbol_last, prescan=True) == ["activation", "activation"]


def test_prescan_returns_upgrade_count():
    lab = TraceLabeller()
    lab.register_address_ranges(_ranges("runtime", "anon"))
    n = lab.prescan_symbols([_row(symbol="ggml_vec_dot_q4_K", op_type="store")])
    assert n == 1


def test_prescan_ignores_rows_without_symbol_or_dso():
    lab = TraceLabeller()
    assert lab.prescan_symbols([_row(symbol="", dso="")]) == 0


def test_register_address_ranges_accepts_regions_without_source():
    """
    pytorch_regions.TensorRegion has no `source` field — reading it with [] instead
    of .get() raised KeyError and crashed the whole PyTorch collection path.
    """
    lab = TraceLabeller()
    n = lab.register_address_ranges([
        {"region_type": "weight", "addr_start": _ADDR, "addr_end": _ADDR + 0x1000},
    ])
    assert n == 1
    assert _label_one(lab)["region_type"] == "weight"


def test_sourceless_weight_range_outranks_coarse_runtime_range():
    """Tensor-derived weights must win over the /proc/maps runtime baseline."""
    lab = TraceLabeller()
    lab.register_address_ranges(_ranges("runtime", "file_mapping"))
    lab.register_address_ranges([
        {"region_type": "weight", "addr_start": _ADDR, "addr_end": _ADDR + 0x1000},
    ])
    assert _label_one(lab)["region_type"] == "weight"

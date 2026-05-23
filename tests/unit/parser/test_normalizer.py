"""Unit tests for the trace row normaliser."""

import pytest

from mrm_trace.parser.normalizer import ALL_COLUMNS, REQUIRED_COLUMNS, normalise_rows


def _minimal(**overrides):
    row = {
        "run_id": "r1",
        "sample_id": 0,
        "timestamp_ns": 1_000,
        "address": 0x7F00_0000_0000,
        "address_page": 0x7F00_0000_0000 >> 12,
        "op_type": "load",
    }
    row.update(overrides)
    return row


# ── happy path ────────────────────────────────────────────────────────────────

def test_normalise_minimal_row_succeeds():
    rows = list(normalise_rows([_minimal()]))
    assert len(rows) == 1


def test_all_canonical_columns_present():
    rows = list(normalise_rows([_minimal()]))
    assert set(rows[0].keys()) == set(ALL_COLUMNS)


def test_output_column_order_matches_all_columns():
    rows = list(normalise_rows([_minimal()]))
    assert list(rows[0].keys()) == ALL_COLUMNS


# ── defaults ──────────────────────────────────────────────────────────────────

def test_default_region_type_is_unknown():
    rows = list(normalise_rows([_minimal()]))
    assert rows[0]["region_type"] == "unknown"


def test_default_region_id_is_zero():
    rows = list(normalise_rows([_minimal()]))
    assert rows[0]["region_id"] == 0


def test_default_phase_is_none():
    rows = list(normalise_rows([_minimal()]))
    assert rows[0]["phase"] is None


def test_default_symbol_is_empty_string():
    rows = list(normalise_rows([_minimal()]))
    assert rows[0]["symbol"] == ""


def test_default_optional_fields_are_none():
    rows = list(normalise_rows([_minimal()]))
    for col in ("access_latency_ns", "cache_level", "numa_node", "cpu_id",
                "thread_id", "token_index", "context_tokens"):
        assert rows[0][col] is None


# ── provided values take priority ─────────────────────────────────────────────

def test_provided_region_type_not_overwritten():
    rows = list(normalise_rows([_minimal(region_type="weight")]))
    assert rows[0]["region_type"] == "weight"


def test_provided_phase_not_overwritten():
    rows = list(normalise_rows([_minimal(phase="generation")]))
    assert rows[0]["phase"] == "generation"


# ── address_page derivation ───────────────────────────────────────────────────

def test_address_page_derived_when_zero():
    addr = 0x7F00_0123_4567
    row = _minimal(address=addr, address_page=0)
    rows = list(normalise_rows([row]))
    assert rows[0]["address_page"] == addr >> 12


def test_address_page_preserved_when_set():
    addr = 0x7F00_0123_4567
    page = addr >> 12
    rows = list(normalise_rows([_minimal(address=addr, address_page=page)]))
    assert rows[0]["address_page"] == page


# ── error handling ────────────────────────────────────────────────────────────

def test_missing_required_column_raises():
    bad = {"run_id": "r1", "sample_id": 0}
    with pytest.raises(ValueError, match="required"):
        list(normalise_rows([bad]))


def test_missing_run_id_raises():
    bad = {k: v for k, v in _minimal().items() if k != "run_id"}
    with pytest.raises(ValueError):
        list(normalise_rows([bad]))


# ── streaming guarantee ───────────────────────────────────────────────────────

def test_normalise_is_a_generator():
    import types
    result = normalise_rows(iter([_minimal()]))
    assert isinstance(result, types.GeneratorType)


def test_normalise_large_input_without_materialising():
    def _gen(n):
        for i in range(n):
            yield _minimal(sample_id=i, timestamp_ns=i)

    count = sum(1 for _ in normalise_rows(_gen(100_000)))
    assert count == 100_000

"""Unit tests for the perf script line parser."""

import io

import pytest

from mrm_trace.parser.perf_script_parser import (
    _event_to_op_type,
    _time_to_ns,
    parse_perf_script,
)

_LOAD_LINE = (
    "          llama.cpp 12345 [001] 1000.000001:          1 "
    "cpu/mem-loads,ldlat=20/P:      7f0001234567 ggml_init+0x123 /usr/lib/libggml.so\n"
)
_STORE_LINE = (
    "          llama.cpp 12345 [001] 1000.000002:          1 "
    "cpu/mem-stores/P:             7f0001234568 ggml_store+0x0 /usr/lib/libggml.so\n"
)


# ── _time_to_ns ───────────────────────────────────────────────────────────────

def test_time_to_ns_microsecond_precision():
    assert _time_to_ns("1234.567890") == 1_234_567_890_000


def test_time_to_ns_short_fraction():
    # 6-digit fraction: "000001" → 1 µs → 1000 ns
    assert _time_to_ns("1000.000001") == 1_000_000_001_000


def test_time_to_ns_zero_fraction():
    assert _time_to_ns("5.000000") == 5_000_000_000


def test_time_to_ns_long_fraction_truncated():
    # More than 9 digits → truncated to nanoseconds
    assert _time_to_ns("0.123456789999") == 123_456_789


# ── _event_to_op_type ─────────────────────────────────────────────────────────

def test_event_to_op_type_load():
    assert _event_to_op_type("cpu/mem-loads,ldlat=20/P") == "load"


def test_event_to_op_type_store():
    assert _event_to_op_type("cpu/mem-stores/P") == "store"


def test_event_to_op_type_unknown_defaults_to_load():
    assert _event_to_op_type("cpu/cycles/P") == "load"


def test_event_to_op_type_case_insensitive():
    assert _event_to_op_type("CPU/MEM-STORES/P") == "store"


# ── parse_perf_script — single line parsing ───────────────────────────────────

def test_parse_single_load_line():
    rows = list(parse_perf_script(io.StringIO(_LOAD_LINE), run_id="r1"))
    assert len(rows) == 1
    assert rows[0]["op_type"] == "load"
    assert rows[0]["run_id"] == "r1"


def test_parse_single_store_line():
    rows = list(parse_perf_script(io.StringIO(_STORE_LINE), run_id="r1"))
    assert len(rows) == 1
    assert rows[0]["op_type"] == "store"


def test_symbol_offset_stripped():
    rows = list(parse_perf_script(io.StringIO(_LOAD_LINE), run_id="r1"))
    assert rows[0]["symbol"] == "ggml_init"
    assert "+" not in rows[0]["symbol"]


def test_address_parsed_as_int():
    rows = list(parse_perf_script(io.StringIO(_LOAD_LINE), run_id="r1"))
    assert isinstance(rows[0]["address"], int)
    assert rows[0]["address"] == 0x7F0001234567


def test_address_page_is_address_right_shifted_12():
    rows = list(parse_perf_script(io.StringIO(_LOAD_LINE), run_id="r1"))
    assert rows[0]["address_page"] == 0x7F0001234567 >> 12


def test_timestamp_ns_parsed():
    rows = list(parse_perf_script(io.StringIO(_LOAD_LINE), run_id="r1"))
    assert rows[0]["timestamp_ns"] == 1_000_000_001_000


def test_cpu_id_parsed():
    rows = list(parse_perf_script(io.StringIO(_LOAD_LINE), run_id="r1"))
    assert rows[0]["cpu_id"] == 1


def test_thread_id_is_pid():
    rows = list(parse_perf_script(io.StringIO(_LOAD_LINE), run_id="r1"))
    assert rows[0]["thread_id"] == 12345


def test_dso_parsed():
    rows = list(parse_perf_script(io.StringIO(_LOAD_LINE), run_id="r1"))
    assert rows[0]["dso"] == "/usr/lib/libggml.so"


def test_default_region_type_unknown():
    rows = list(parse_perf_script(io.StringIO(_LOAD_LINE), run_id="r1"))
    assert rows[0]["region_type"] == "unknown"


def test_all_canonical_columns_present():
    rows = list(parse_perf_script(io.StringIO(_LOAD_LINE), run_id="r1"))
    expected = {
        "run_id", "sample_id", "timestamp_ns", "phase", "address", "address_page",
        "op_type", "symbol", "dso", "region_type", "region_id", "access_latency_ns",
        "cache_level", "numa_node", "cpu_id", "thread_id", "token_index", "context_tokens",
    }
    assert expected.issubset(rows[0].keys())


# ── parse_perf_script — multi-line and edge cases ────────────────────────────

def test_sample_id_sequential():
    rows = list(parse_perf_script(io.StringIO(_LOAD_LINE + _STORE_LINE), run_id="r1"))
    assert rows[0]["sample_id"] == 0
    assert rows[1]["sample_id"] == 1


def test_malformed_lines_skipped():
    content = "this is garbage\n# comment line\n" + _LOAD_LINE
    rows = list(parse_perf_script(io.StringIO(content), run_id="r1"))
    assert len(rows) == 1


def test_empty_input_yields_nothing():
    assert list(parse_perf_script(io.StringIO(""), run_id="r1")) == []


def test_phase_annotation_by_boundary():
    rows = list(parse_perf_script(
        io.StringIO(_LOAD_LINE), run_id="r1",
        phase_boundaries=[{"phase": "generation", "start_ns": 0, "end_ns": 2_000_000_000_000}],
    ))
    assert rows[0]["phase"] == "generation"


def test_phase_none_when_no_boundaries():
    rows = list(parse_perf_script(io.StringIO(_LOAD_LINE), run_id="r1"))
    assert rows[0]["phase"] is None


def test_phase_none_when_outside_all_boundaries():
    rows = list(parse_perf_script(
        io.StringIO(_LOAD_LINE), run_id="r1",
        phase_boundaries=[{"phase": "weight_load", "start_ns": 0, "end_ns": 1}],
    ))
    assert rows[0]["phase"] is None

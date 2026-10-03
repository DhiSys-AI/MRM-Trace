"""
Chunked parsing must not change what the pipeline produces.

A 30M-sample perf-mem run holds 26 GB of row dicts when parsed with list()
(863 bytes a row, measured on a real trace), and the pipeline needs two passes
over the rows — symbol prescan, then labelling — so the eager path peaked near
55 GB and was OOM-killed on a 30 GB machine.  _parse_and_label now streams the
text file twice in chunks whose size comes from free memory.

Streaming is only safe if the result does not depend on where the chunk
boundaries fall.  Region classification is stateful (first evidence wins, and
region ids are handed out in arrival order), so a chunking bug would not raise
— it would quietly relabel regions.  These tests pin that invariant.
"""

import importlib.util
import pathlib
import time

import pandas as pd
import pytest

from mrm_trace.reporter.trace_qa import check_trace_quality, check_trace_quality_counts

pytestmark = pytest.mark.validity

_DEMO = pathlib.Path(__file__).resolve().parents[2] / "notebooks" / "scripts" / "demo_perf_mem.py"

_WEIGHT_BASE = 0x7B6C_0000_0000
_KV_BASE = 0x7B6D_0000_0000
_GRANULARITY = 65536


def _load_demo():
    """Import demo_perf_mem.py by path (notebooks/ is not an installed package)."""
    spec = importlib.util.spec_from_file_location("demo_perf_mem", _DEMO)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _perf_line(mono_s: float, addr: int, symbol: str, store: bool = True) -> str:
    event = "cpu/mem-stores/P" if store else "cpu/mem-loads,ldlat=30/P"
    op = "STORE" if store else "LOAD"
    weight = 0 if store else 97
    return (
        f"          python    9608  {mono_s:.6f}:         {event}: {addr:x}"
        f"      1e05080144 |OP {op}|LVL L1 hit|SNP N/A|TLB N/A|LCK N/A|BLK  N/A"
        f"          {weight}     719f96f6e33d {symbol} (/lib/libggml-cpu.so.0)\n"
    )


def _phase_boundaries(demo, mono_base: int):
    events = [
        ("model_load_start", 0), ("model_load_end", 2_000_000_000),
        ("generation_start", 2_000_000_000), ("generation_end", 9_000_000_000),
    ]
    wall_base = time.time_ns()
    return demo.build_phase_boundaries([
        {"event": ev, "wall_ns": wall_base + off, "mono_ns": mono_base + off}
        for ev, off in events
    ])


@pytest.fixture
def trace_file(tmp_path):
    """A synthetic perf script covering several regions, symbols and both ops."""
    mono_base = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    lines = []
    for i in range(240):
        offset_ns = 100_000_000 + i * 30_000_000
        # Spread across granules so region ids are handed out progressively.
        lines.append(_perf_line(
            (mono_base + offset_ns) / 1e9,
            _WEIGHT_BASE + (i % 7) * _GRANULARITY + i * 64,
            "ggml_gemv_q4_K_8x8_q8_K",
            store=(i % 3 != 0),
        ))
        lines.append(_perf_line(
            (mono_base + offset_ns + 1_000_000) / 1e9,
            _KV_BASE + (i % 5) * _GRANULARITY + i * 128,
            "llama_kv_cache_find_slot",
            store=(i % 2 == 0),
        ))
    path = tmp_path / "perf_script.txt"
    path.write_text("".join(lines))
    return path, mono_base


@pytest.fixture
def address_regions():
    return [
        {"region_type": "weight", "addr_start": _WEIGHT_BASE,
         "addr_end": _WEIGHT_BASE + 16 * _GRANULARITY,
         "size_bytes": 16 * _GRANULARITY, "source": "mmap_gguf"},
        {"region_type": "kv_cache", "addr_start": _KV_BASE,
         "addr_end": _KV_BASE + 16 * _GRANULARITY,
         "size_bytes": 16 * _GRANULARITY, "source": "anon_kv_sized"},
    ]


def _run(demo, trace_file, address_regions, chunk_rows):
    path, mono_base = trace_file
    return demo._parse_and_label(
        path,
        run_id="r1",
        phase_boundaries=_phase_boundaries(demo, mono_base),
        address_regions=address_regions,
        granularity=_GRANULARITY,
        chunk_rows=chunk_rows,
        sample_period=32_000,
    )


@pytest.mark.parametrize("chunk_rows", [1, 2, 7, 100, 10_000])
def test_labelled_trace_is_independent_of_chunk_size(
    trace_file, address_regions, chunk_rows
):
    """Chunk boundaries must not shift a single label or region id."""
    demo = _load_demo()
    baseline, _, _ = _run(demo, trace_file, address_regions, chunk_rows=10_000)
    chunked, _, _ = _run(demo, trace_file, address_regions, chunk_rows=chunk_rows)

    assert len(chunked) == len(baseline)
    pd.testing.assert_frame_equal(chunked, baseline)


@pytest.mark.parametrize("chunk_rows", [1, 7, 100])
def test_region_map_is_independent_of_chunk_size(
    trace_file, address_regions, chunk_rows
):
    demo = _load_demo()
    _, baseline, _ = _run(demo, trace_file, address_regions, chunk_rows=10_000)
    _, chunked, _ = _run(demo, trace_file, address_regions, chunk_rows=chunk_rows)

    pd.testing.assert_frame_equal(
        chunked.sort_values("region_id").reset_index(drop=True),
        baseline.sort_values("region_id").reset_index(drop=True),
    )


def test_kv_lifecycle_is_independent_of_chunk_size(trace_file, address_regions):
    demo = _load_demo()
    _, _, baseline = _run(demo, trace_file, address_regions, chunk_rows=10_000)
    _, _, chunked = _run(demo, trace_file, address_regions, chunk_rows=3)

    assert len(chunked) == len(baseline)
    pd.testing.assert_frame_equal(chunked, baseline)


def test_regions_are_actually_classified(trace_file, address_regions):
    """Guards the fixture: a trace labelled entirely 'unknown' proves nothing."""
    demo = _load_demo()
    labelled, _, _ = _run(demo, trace_file, address_regions, chunk_rows=50)
    assert set(labelled["region_type"]) >= {"weight", "kv_cache"}


def test_a_file_object_is_rejected(trace_file, address_regions):
    """
    Pass 1 would exhaust a stream and pass 2 would label nothing, producing an
    empty trace with no error — so the type is refused up front.
    """
    import io
    demo = _load_demo()
    path, mono_base = trace_file
    with pytest.raises(TypeError, match="file path"):
        demo._parse_and_label(
            io.StringIO(path.read_text()), run_id="r1",
            phase_boundaries=_phase_boundaries(demo, mono_base),
            address_regions=address_regions,
        )


def test_chunks_yields_every_row_exactly_once():
    demo = _load_demo()
    for size in (1, 3, 10, 999):
        got = [row for chunk in demo._chunks(range(37), size) for row in chunk]
        assert got == list(range(37))
        assert all(len(c) <= size for c in demo._chunks(range(37), size))


def test_auto_chunk_rows_stays_within_bounds(monkeypatch):
    """A tiny VM must not chunk down to nothing; a big host must not over-allocate."""
    demo = _load_demo()

    class _VM:
        def __init__(self, available):
            self.available = available

    monkeypatch.setattr(demo.psutil, "virtual_memory", lambda: _VM(64 * 1024**2))
    assert demo._auto_chunk_rows() == demo._CHUNK_ROWS_MIN

    monkeypatch.setattr(demo.psutil, "virtual_memory", lambda: _VM(512 * 1024**3))
    assert demo._auto_chunk_rows() == demo._CHUNK_ROWS_MAX

    monkeypatch.setattr(demo.psutil, "virtual_memory", lambda: _VM(25 * 1024**3))
    rows = demo._auto_chunk_rows()
    assert demo._CHUNK_ROWS_MIN < rows < demo._CHUNK_ROWS_MAX
    # The chunk itself must stay a small fraction of free memory.
    assert rows * demo._ROW_DICT_BYTES < 25 * 1024**3 * 0.05


def test_counts_gate_agrees_with_dataframe_gate():
    """Both entry points must reach the same verdict on the same trace."""
    trace = pd.DataFrame({
        "access_latency_ns": [10.0, None, 30.0, 40.0],
        "cache_level": ["L1", "L3", None, "DRAM"],
        "symbol": ["ggml_mul", "", "ggml_mul", "x"],
        "region_type": ["weight", "unknown", "kv_cache", "unknown"],
        "phase": ["generation", None, "generation", "generation"],
    })
    from_frame = check_trace_quality(
        trace, collector_type="perf_mem_labelled", print_report=False)
    from_counts = check_trace_quality_counts(
        rows=4, latency_non_null=3, cache_level_non_null=3,
        symbol_non_empty=3, region_classified=2, phase_non_null=3,
        collector_type="perf_mem_labelled", print_report=False)

    assert from_counts == from_frame


def test_counts_gate_fails_an_empty_trace():
    qa = check_trace_quality_counts(
        rows=0, latency_non_null=0, cache_level_non_null=0,
        symbol_non_empty=0, region_classified=0, print_report=False)
    assert qa.status == "FAIL"
    assert not qa.ok


@pytest.mark.parametrize("chunk_rows", [1, 7, 100, 10_000])
def test_repeated_string_columns_stay_dictionary_encoded(
    trace_file, address_regions, chunk_rows
):
    """
    pandas 3 stores a str column as inline bytes, so on a real trace dso costs
    90 bytes a row and symbol 63 — 70% of the frame.  Dictionary encoding only
    survives pd.concat when every chunk shares one explicit CategoricalDtype;
    a per-chunk .astype("category") silently degrades back to str, which is a
    2.4x memory regression that no other assertion would catch.
    """
    demo = _load_demo()
    labelled, _, _ = _run(demo, trace_file, address_regions, chunk_rows=chunk_rows)

    for col in demo._CATEGORICAL_COLS:
        assert isinstance(labelled[col].dtype, pd.CategoricalDtype), (
            f"{col} is {labelled[col].dtype}, not dictionary-encoded"
        )

    # The columns the analysers group by must NOT be encoded — a categorical
    # grouper's handling of unobserved values is pandas-version dependent.
    for col in ("op_type", "region_type"):
        assert not isinstance(labelled[col].dtype, pd.CategoricalDtype)


def test_dictionary_encoding_preserves_every_value(trace_file, address_regions):
    """A value missing from the dtype would become NaN, wiping the column."""
    demo = _load_demo()
    labelled, _, _ = _run(demo, trace_file, address_regions, chunk_rows=50)

    assert not labelled["symbol"].isna().any()
    assert not labelled["dso"].isna().any()
    assert set(labelled["symbol"].astype(str)) == {
        "ggml_gemv_q4_K_8x8_q8_K", "llama_kv_cache_find_slot"
    }
    assert set(labelled["dso"].astype(str)) == {"/lib/libggml-cpu.so.0"}
    assert labelled["cache_level"].astype(str).eq("L1").all()


def test_encoding_is_leaner_than_plain_strings(trace_file, address_regions):
    """Pins the direction of the optimisation, not a specific byte count."""
    demo = _load_demo()
    labelled, _, _ = _run(demo, trace_file, address_regions, chunk_rows=100)

    encoded = labelled.memory_usage(deep=False).sum()
    plain = labelled.assign(**{
        col: labelled[col].astype(str) for col in demo._CATEGORICAL_COLS
    }).memory_usage(deep=False).sum()
    assert encoded < plain

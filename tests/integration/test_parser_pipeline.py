"""
Integration tests for the full parser pipeline.

Verifies that synthetic perf script text can be parsed, normalised, and
written to trace.parquet without schema drift and within expected row counts.
"""

import io

import pandas as pd
import pytest

from mrm_trace.parser.normalizer import ALL_COLUMNS, normalise_rows
from mrm_trace.parser.perf_script_parser import parse_perf_script
from mrm_trace.parser.writer import write_parquet


def _make_perf_lines(n: int, run_id: str = "r1") -> str:
    lines = []
    for i in range(n):
        event = "cpu/mem-loads,ldlat=20/P" if i % 2 == 0 else "cpu/mem-stores/P"
        ts = f"{1000 + i // 1_000_000}.{(i % 1_000_000):06d}"
        addr = f"{0x7F00_0000_0000 + i * 64:016x}"
        sym = "ggml_matmul" if i % 2 == 0 else "ggml_store"
        lines.append(
            f"          llama 12345 [001] {ts}:          1 {event}:      "
            f"{addr} {sym}+0x0 /usr/lib/libggml.so\n"
        )
    return "".join(lines)


class TestPerfScriptToParquet:

    def test_round_trip_small(self, tmp_path):
        rows = normalise_rows(parse_perf_script(io.StringIO(_make_perf_lines(100)), run_id="r1"))
        out = tmp_path / "trace.parquet"
        n = write_parquet(rows, out)
        assert n == 100
        assert out.exists()

    def test_output_has_all_canonical_columns(self, tmp_path):
        rows = normalise_rows(parse_perf_script(io.StringIO(_make_perf_lines(10)), run_id="r1"))
        out = tmp_path / "trace.parquet"
        write_parquet(rows, out)
        df = pd.read_parquet(out)
        for col in ALL_COLUMNS:
            assert col in df.columns, f"Column '{col}' missing from trace.parquet"

    def test_row_count_matches_input(self, tmp_path):
        n = 500
        rows = normalise_rows(parse_perf_script(io.StringIO(_make_perf_lines(n)), run_id="r1"))
        out = tmp_path / "trace.parquet"
        count = write_parquet(rows, out)
        df = pd.read_parquet(out)
        assert count == n
        assert len(df) == n

    def test_sample_ids_are_sequential(self, tmp_path):
        rows = normalise_rows(parse_perf_script(io.StringIO(_make_perf_lines(50)), run_id="r1"))
        out = tmp_path / "trace.parquet"
        write_parquet(rows, out)
        df = pd.read_parquet(out)
        assert list(df["sample_id"]) == list(range(50))

    def test_op_type_values_are_valid(self, tmp_path):
        rows = normalise_rows(parse_perf_script(io.StringIO(_make_perf_lines(100)), run_id="r1"))
        out = tmp_path / "trace.parquet"
        write_parquet(rows, out)
        df = pd.read_parquet(out)
        assert set(df["op_type"].unique()).issubset({"load", "store"})

    def test_alternating_op_types_from_alternating_events(self, tmp_path):
        rows = normalise_rows(parse_perf_script(io.StringIO(_make_perf_lines(4)), run_id="r1"))
        out = tmp_path / "trace.parquet"
        write_parquet(rows, out)
        df = pd.read_parquet(out)
        assert list(df["op_type"]) == ["load", "store", "load", "store"]

    def test_garbage_lines_skipped(self, tmp_path):
        garbage = "not a valid perf line\n# comment\n\n"
        good = _make_perf_lines(5)
        rows = normalise_rows(parse_perf_script(io.StringIO(garbage + good), run_id="r1"))
        out = tmp_path / "trace.parquet"
        n = write_parquet(rows, out)
        assert n == 5

    def test_phase_annotation_written_to_parquet(self, tmp_path):
        boundaries = [{"phase": "generation", "start_ns": 0, "end_ns": 10**15}]
        rows = normalise_rows(parse_perf_script(
            io.StringIO(_make_perf_lines(10)), run_id="r1",
            phase_boundaries=boundaries,
        ))
        out = tmp_path / "trace.parquet"
        write_parquet(rows, out)
        df = pd.read_parquet(out)
        assert all(df["phase"] == "generation")

    def test_large_batch_streaming(self, tmp_path):
        n = 1_000_000
        rows = normalise_rows(parse_perf_script(
            io.StringIO(_make_perf_lines(n)), run_id="r1",
        ))
        out = tmp_path / "trace.parquet"
        count = write_parquet(rows, out, batch_size=50_000)
        assert count == n

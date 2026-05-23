"""
Tier-3 validity tests for Phase 4 — Parsing.

These tests verify scientific correctness properties that, if violated,
would corrupt the canonical trace output or break the streaming guarantee (ADR-2).
"""

import io
import tracemalloc

import pandas as pd
import pytest

from mrm_trace.parser.normalizer import ALL_COLUMNS, normalise_rows
from mrm_trace.parser.perf_script_parser import parse_perf_script
from mrm_trace.parser.writer import write_parquet

_LOAD_LINE = (
    "          llama 99 [000]  1234.000001:          1 "
    "cpu/mem-loads,ldlat=20/P:      7f0001234567 test_sym+0x0 /usr/lib/libtest.so\n"
)
_STORE_LINE = (
    "          llama 99 [000]  1234.000002:          1 "
    "cpu/mem-stores/P:             7f0001234568 test_store+0x0 /usr/lib/libtest.so\n"
)


def _make_n_lines(n: int) -> str:
    lines = []
    for i in range(n):
        event = "cpu/mem-loads,ldlat=20/P" if i % 2 == 0 else "cpu/mem-stores/P"
        ts = f"{1000 + i // 1_000_000}.{(i % 1_000_000):06d}"
        addr = f"{0x7F00_0000_0000 + i * 64:016x}"
        lines.append(
            f"          llama 12345 [001] {ts}:          1 {event}:      "
            f"{addr} sym+0x0 /lib/lib.so\n"
        )
    return "".join(lines)


def test_load_event_produces_load_op_type():
    """A mem-loads event must always produce op_type='load'."""
    rows = list(parse_perf_script(io.StringIO(_LOAD_LINE), run_id="r1"))
    assert rows[0]["op_type"] == "load", (
        f"mem-loads event produced op_type={rows[0]['op_type']!r}. "
        "This would corrupt write-once ratio calculations — stores would be undercounted."
    )


def test_store_event_produces_store_op_type():
    """A mem-stores event must always produce op_type='store'."""
    rows = list(parse_perf_script(io.StringIO(_STORE_LINE), run_id="r1"))
    assert rows[0]["op_type"] == "store", (
        f"mem-stores event produced op_type={rows[0]['op_type']!r}. "
        "This would corrupt the write-once ratio — stores counted as loads."
    )


def test_address_page_is_exactly_address_right_shifted_12():
    """address_page must equal address >> 12 for every row."""
    rows = list(normalise_rows(parse_perf_script(
        io.StringIO(_LOAD_LINE * 50 + _STORE_LINE * 50), run_id="r1"
    )))
    for row in rows:
        expected = row["address"] >> 12
        assert row["address_page"] == expected, (
            f"address_page={row['address_page']:#x} != address >> 12 = {expected:#x}. "
            "Incorrect page mapping corrupts working-set and locality analysis."
        )


def test_sample_ids_are_contiguous_from_zero():
    """sample_id must be 0-based and contiguous — no gaps or duplicates."""
    n = 200
    rows = list(parse_perf_script(io.StringIO(_make_n_lines(n)), run_id="r1"))
    ids = [r["sample_id"] for r in rows]
    assert ids == list(range(n)), (
        "sample_id sequence has gaps or duplicates. "
        "Non-contiguous IDs corrupt join operations in the labeller and analyser."
    )


def test_canonical_schema_all_columns_present_in_parquet(tmp_path):
    """trace.parquet must contain every column in the canonical schema."""
    rows = normalise_rows(parse_perf_script(io.StringIO(_make_n_lines(10)), run_id="r1"))
    out = tmp_path / "trace.parquet"
    write_parquet(rows, out)
    df = pd.read_parquet(out)
    missing = set(ALL_COLUMNS) - set(df.columns)
    assert not missing, (
        f"trace.parquet is missing columns: {missing}. "
        "Missing columns will cause KeyErrors in downstream labeller/analyser phases."
    )


def test_parser_does_not_accumulate_rows_in_memory(tmp_path):
    """
    Parsing 200k lines must not require holding all rows in memory simultaneously.
    This verifies ADR-2: streaming parser — never load full trace into RAM.
    """
    n = 200_000
    # Allocate the source text before measuring so it doesn't inflate the delta
    text = _make_n_lines(n)
    out = tmp_path / "trace.parquet"

    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    count = write_parquet(
        normalise_rows(parse_perf_script(io.StringIO(text), run_id="r1")),
        out,
        batch_size=10_000,
    )

    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    top_stats = snapshot_after.compare_to(snapshot_before, "lineno")
    total_increase_mb = sum(s.size_diff for s in top_stats) / 1_048_576

    assert count == n
    # Two batches of 10k rows in flight at once; 200 MB is a generous ceiling
    assert total_increase_mb < 200, (
        f"Memory increased by {total_increase_mb:.1f} MB while parsing {n} rows. "
        "The streaming parser must not accumulate all rows in memory (ADR-2). "
        "Check that parse_perf_script and normalise_rows are proper generators."
    )


def test_no_schema_drift_across_batch_boundaries(tmp_path):
    """All rows must have the same schema regardless of batch boundaries."""
    n = 250_000
    rows = normalise_rows(parse_perf_script(io.StringIO(_make_n_lines(n)), run_id="r1"))
    out = tmp_path / "trace.parquet"
    # Batch size intentionally not aligned to n to test cross-boundary writes
    write_parquet(rows, out, batch_size=99_999)
    df = pd.read_parquet(out)
    assert len(df) == n
    assert set(df.columns) == set(ALL_COLUMNS)


def test_timestamp_precision_not_truncated():
    """
    Timestamps must preserve sub-microsecond precision from perf script output.
    Precision loss would corrupt access latency histograms and ordering.
    """
    line = (
        "          llama 1 [000]  1000.123456789:          1 "
        "cpu/mem-loads,ldlat=20/P:      7f0000000000 sym /lib/lib.so\n"
    )
    rows = list(parse_perf_script(io.StringIO(line), run_id="r1"))
    assert rows[0]["timestamp_ns"] == 1_000_123_456_789, (
        f"Expected 1_000_123_456_789 ns, got {rows[0]['timestamp_ns']}. "
        "Timestamp truncation corrupts fine-grained latency ordering."
    )

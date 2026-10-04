"""
Parsing in parallel must produce exactly the serial result.

Parsing is the expensive half of building a trace (38.1 s per million samples,
measured) and it is embarrassingly parallel, but splitting a file by byte offset
can silently drop or duplicate the line a boundary falls in.  A trace short by a
few rows would pass every other check in this project.

Labelling is deliberately *not* parallelised: region ids are handed out on
arrival and classification is first-evidence-wins, so splitting it would change
the labels rather than just the speed.
"""

import pathlib
import time

import pandas as pd
import pytest

from mrm_trace.parser.parallel import parse_trace_parallel
from mrm_trace.parser.perf_script_parser import (
    line_aligned_ranges,
    parse_perf_script,
)

pytestmark = pytest.mark.validity


def _perf_line(mono_s: float, addr: int, symbol: str, store: bool = True) -> str:
    event = "cpu/mem-stores/P" if store else "cpu/mem-loads,ldlat=3/P"
    op = "STORE" if store else "LOAD"
    weight = 0 if store else 97
    return (
        f"          python    9608  {mono_s:.6f}:         {event}: {addr:x}"
        f"      1e05080144 |OP {op}|LVL L1 hit|SNP N/A|TLB N/A|LCK N/A|BLK  N/A"
        f"          {weight}     719f96f6e33d {symbol} (/lib/libggml-cpu.so.0)\n"
    )


@pytest.fixture
def trace_file(tmp_path):
    lines = []
    for i in range(5000):
        lines.append(_perf_line(
            1000.0 + i * 0.001,
            0x7B6C_0000_0000 + (i % 97) * 4096 + i * 8,
            "ggml_vec_dot_q6_K_q8_K" if i % 3 else "llama_kv_cache_find_slot",
            store=(i % 4 != 0),
        ))
    path = tmp_path / "perf_script.txt"
    path.write_text("".join(lines))
    return path


@pytest.mark.parametrize("workers", [1, 2, 3, 4, 8, 16])
def test_parallel_parse_matches_serial(trace_file, workers):
    serial = pd.DataFrame(list(parse_perf_script(trace_file, run_id="r1")))
    parallel = parse_trace_parallel(trace_file, "r1", None, workers)

    assert len(parallel) == len(serial), f"{workers} workers changed the row count"
    # Compare on the raw trace columns; sample_id is renumbered by construction
    # and must come out identical because order is preserved.
    cols = [c for c in serial.columns if c in parallel.columns]
    pd.testing.assert_frame_equal(
        parallel[cols].reset_index(drop=True),
        serial[cols].reset_index(drop=True),
        check_dtype=False,
    )


def test_sample_ids_are_contiguous_and_ordered(trace_file):
    parallel = parse_trace_parallel(trace_file, "r1", None, 4)
    assert parallel["sample_id"].tolist() == list(range(len(parallel)))
    # Time must not go backwards: the ranges were concatenated in file order.
    assert parallel["timestamp_ns"].is_monotonic_increasing


def test_more_workers_than_lines_is_safe(tmp_path):
    path = tmp_path / "tiny.txt"
    path.write_text(_perf_line(1000.0, 0x7B6C_0000_0000, "ggml_vec_dot"))
    serial = pd.DataFrame(list(parse_perf_script(path, run_id="r1")))
    assert len(parse_trace_parallel(path, "r1", None, 32)) == len(serial)


def test_an_empty_file_yields_an_empty_frame(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("")
    frame = parse_trace_parallel(path, "r1", None, 4)
    assert frame.empty


def test_unparsable_lines_are_dropped_the_same_way(tmp_path):
    """A boundary must not turn a skipped line into a kept one."""
    lines = []
    for i in range(500):
        lines.append("this line does not match the perf format\n")
        lines.append(_perf_line(1000.0 + i * 0.001, 0x7B6C_0000_0000 + i * 64,
                                "ggml_vec_dot"))
    path = tmp_path / "mixed.txt"
    path.write_text("".join(lines))

    serial = pd.DataFrame(list(parse_perf_script(path, run_id="r1")))
    for workers in (2, 5):
        parallel = parse_trace_parallel(path, "r1", None, workers)
        assert len(parallel) == len(serial) == 500


def test_ranges_partition_the_file_exactly(trace_file):
    """The union of the ranges must be every byte, with no overlap."""
    size = trace_file.stat().st_size
    for workers in (1, 2, 4, 9):
        ranges = line_aligned_ranges(trace_file, workers)
        assert ranges[0][0] == 0
        assert ranges[-1][1] == size
        for (_, end), (start, _) in zip(ranges, ranges[1:]):
            assert end == start, "ranges overlap or leave a gap"

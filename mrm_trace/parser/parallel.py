"""
Parallel parsing of a perf script text file.

Parsing is the expensive half of turning a capture into a trace — 38.1 s per
million samples, measured, mostly regex over several gigabytes of text — and it
is embarrassingly parallel.  Labelling is not: region ids are handed out on
arrival and classification is first-evidence-wins, so it must stay serial and
in file order.

This lives in the package rather than in a collection script because a
multiprocessing worker has to be importable by the child process; a function
defined in a script loaded by path cannot be pickled.
"""

from __future__ import annotations

import multiprocessing
import pathlib
import tempfile
from typing import List, Optional

import pandas as pd

from mrm_trace.parser.perf_script_parser import line_aligned_ranges, parse_perf_script
from mrm_trace.parser.writer import TRACE_SCHEMA

# Column order for an empty result, so a worker whose range held no parsable
# lines still returns a frame the concatenation can align.
TRACE_COLUMNS = tuple(field.name for field in TRACE_SCHEMA)

# Rows a worker accumulates before converting to a frame.  Row dicts cost 863
# bytes each (measured), so this bounds a worker's transient memory.
_WORKER_CHUNK_ROWS = 250_000


def _parse_range_to_parquet(job) -> str:
    """
    Worker: parse one byte range and write it to a temporary Parquet file.

    The frame goes to disk rather than back through the pool because a range of
    a 10M-row trace is a few hundred MB as a frame, and pickling several of
    those costs more than writing them.
    """
    path, byte_range, run_id, phase_boundaries, out_path = job
    frames: List[pd.DataFrame] = []
    chunk: list = []
    for row in parse_perf_script(path, run_id=run_id,
                                 phase_boundaries=phase_boundaries,
                                 byte_range=byte_range):
        chunk.append(row)
        if len(chunk) >= _WORKER_CHUNK_ROWS:
            frames.append(pd.DataFrame(chunk))
            chunk = []
    if chunk:
        frames.append(pd.DataFrame(chunk))
    frame = (pd.concat(frames, ignore_index=True) if frames
             else pd.DataFrame(columns=list(TRACE_COLUMNS)))
    frame.to_parquet(out_path, index=False)
    return out_path


def parse_trace_parallel(path, run_id: str,
                         phase_boundaries: Optional[list] = None,
                         workers: int = 1) -> pd.DataFrame:
    """
    Parse a perf script file once, across *workers* processes.

    Returns the raw (unlabelled) trace as one frame, in file order, with
    sample_id renumbered over the whole trace — the parser restarts it at 0 in
    each byte range.

    Falls back to a single-process parse when only one range results, so a
    small file or workers=1 costs nothing extra.
    """
    path = pathlib.Path(path)
    ranges = line_aligned_ranges(path, workers)
    if len(ranges) <= 1:
        rows = list(parse_perf_script(path, run_id=run_id,
                                      phase_boundaries=phase_boundaries))
        frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=list(TRACE_COLUMNS))
    else:
        with tempfile.TemporaryDirectory(prefix="mrm-parse-") as tmpdir:
            jobs = [
                (str(path), byte_range, run_id, phase_boundaries,
                 str(pathlib.Path(tmpdir) / f"part{i:03d}.parquet"))
                for i, byte_range in enumerate(ranges)
            ]
            with multiprocessing.Pool(processes=len(jobs)) as pool:
                parts = pool.map(_parse_range_to_parquet, jobs)
            # sorted() restores file order, so the trace stays time-ordered.
            frame = pd.concat([pd.read_parquet(part) for part in sorted(parts)],
                              ignore_index=True)

    if "sample_id" in frame.columns and len(frame):
        frame["sample_id"] = range(len(frame))
    return frame

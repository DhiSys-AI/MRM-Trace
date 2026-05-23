"""
Streaming parser for `perf script` text output.

ADR-2: never accumulates the full trace in memory.
Each parsed line is yielded immediately as a dict.
"""

import re
from pathlib import Path
from typing import Generator, IO, List, Optional, Union

from mrm_trace.parser.schema import OpType

# perf script line format (with -F time,pid,addr,sym,dso):
#   <comm> <pid> [<cpu>] <time>: [<period>] <event>:  <addr> <sym>[+off] [<dso>]
_LINE_RE = re.compile(
    r"^\s*(?P<comm>.+?)\s+(?P<pid>\d+)\s+\[(?P<cpu>\d+)\]\s+"
    r"(?P<time>\d+\.\d+):\s+(?:\d+\s+)?(?P<event>[^:]+):\s+"
    r"(?P<addr>[0-9a-fA-F]+)\s+(?P<sym>\S+)(?:\s+(?P<dso>\S+))?",
    re.ASCII,
)

_STORE_RE = re.compile(r"mem-store", re.IGNORECASE)


def _event_to_op_type(event: str) -> str:
    if _STORE_RE.search(event):
        return OpType.STORE.value
    return OpType.LOAD.value


def _time_to_ns(time_str: str) -> int:
    """Convert 'seconds.fraction' string to integer nanoseconds."""
    seconds, _, frac = time_str.partition(".")
    # pad or truncate fraction to exactly 9 digits (nanosecond precision)
    frac_ns = int(frac.ljust(9, "0")[:9])
    return int(seconds) * 1_000_000_000 + frac_ns


def parse_perf_script(
    source: Union[str, Path, IO],
    run_id: str,
    phase_boundaries: Optional[List[dict]] = None,
) -> Generator[dict, None, None]:
    """
    Stream-parse perf script text output, yielding one canonical row dict per sample.

    Args:
        source: file path or readable file object (supports StringIO)
        run_id: run identifier written into every row
        phase_boundaries: optional list of {phase, start_ns, end_ns} dicts;
                          timestamps in [start_ns, end_ns) receive that phase label
    """
    sample_id = 0

    def _annotate_phase(ts_ns: int) -> Optional[str]:
        if not phase_boundaries:
            return None
        for pb in phase_boundaries:
            if pb["start_ns"] <= ts_ns < pb["end_ns"]:
                return pb["phase"]
        return None

    def _process_fh(fh: IO) -> Generator[dict, None, None]:
        nonlocal sample_id
        for line in fh:
            m = _LINE_RE.match(line)
            if not m:
                continue

            addr = int(m.group("addr"), 16)
            ts_ns = _time_to_ns(m.group("time"))
            sym_raw = m.group("sym")
            sym = sym_raw.split("+")[0] if "+" in sym_raw else sym_raw

            yield {
                "run_id": run_id,
                "sample_id": sample_id,
                "timestamp_ns": ts_ns,
                "phase": _annotate_phase(ts_ns),
                "address": addr,
                "address_page": addr >> 12,
                "op_type": _event_to_op_type(m.group("event")),
                "symbol": sym,
                "dso": m.group("dso") or "",
                "region_type": "unknown",
                "region_id": 0,
                "access_latency_ns": None,
                "cache_level": None,
                "numa_node": None,
                "cpu_id": int(m.group("cpu")),
                "thread_id": int(m.group("pid")),
                "token_index": None,
                "context_tokens": None,
            }
            sample_id += 1

    if hasattr(source, "read"):
        yield from _process_fh(source)
    else:
        with open(source, "r", encoding="utf-8", errors="replace") as fh:
            yield from _process_fh(fh)

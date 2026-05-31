"""
Streaming parser for `perf script` text output.

ADR-2: never accumulates the full trace in memory.
Each parsed line is yielded immediately as a dict.

Handles two output formats produced by different perf versions / flag sets:

  Full format  (perf script -F comm,pid,cpu,time,event,addr,sym,dso):
    python  1234 [003]  12345.678901:  cpu/mem-loads/P:  7f00dead sym+0x10 /lib/foo.so

  Compact format  (perf script -F time,pid,addr or default with no symbols):
    1234  12345.678901:  7f00dead

The parser tries the full format first; if that fails it falls back to compact.
Lines that match neither are silently skipped and counted.  Parse statistics are
logged at INFO level when iteration finishes.
"""

import logging
import re
from pathlib import Path
from typing import Generator, IO, List, Optional, Union

from mrm_trace.parser.schema import OpType

log = logging.getLogger(__name__)

# ── Regex patterns ────────────────────────────────────────────────────────────

# Full format: comm  pid [cpu]  time:  [period ]  event:  addr  sym  [dso]
# [cpu] is absent when the cpu attribute is not set for the event type.
# [cpu] absent when cpu attr not in event; sym/dso absent when no debug symbols.
_FULL_RE = re.compile(
    r"^\s*\S.*?\s+(?P<pid>\d+)(?:\s+\[(?P<cpu>\d+)\])?\s+"
    r"(?P<time>\d+\.\d+):\s+(?:\d+\s+)?(?P<event>[^:]+):\s+"
    r"(?P<addr>[0-9a-fA-F]+)(?:\s+(?P<sym>\S+)(?:\s+(?P<dso>\S+))?)?",
    re.ASCII,
)

# Compact format: pid  time:  addr   (sym and dso absent)
_COMPACT_RE = re.compile(
    r"^\s*(?P<pid>\d+)\s+(?P<time>\d+\.\d+):\s+(?P<addr>[0-9a-fA-F]+)\s*$",
    re.ASCII,
)

_STORE_RE = re.compile(r"mem-store", re.IGNORECASE)


# ── Stateless helpers (no nesting → lower cognitive complexity) ───────────────

def _event_to_op_type(event: str) -> str:
    return OpType.STORE.value if _STORE_RE.search(event) else OpType.LOAD.value


def _time_to_ns(time_str: str) -> int:
    seconds, _, frac = time_str.partition(".")
    return int(seconds) * 1_000_000_000 + int(frac.ljust(9, "0")[:9])


def _annotate_phase(ts_ns: int, phase_boundaries: Optional[List[dict]]) -> Optional[str]:
    if not phase_boundaries:
        return None
    for pb in phase_boundaries:
        if pb["start_ns"] <= ts_ns < pb["end_ns"]:
            return pb["phase"]
    return None


def _parse_line(line: str) -> "Optional[tuple]":
    """
    Try to parse one perf script line.

    Returns (pid, cpu_or_None, time_str, event, addr_hex, sym, dso) or None.
    """
    m = _FULL_RE.match(line)
    if m:
        cpu_str = m.group("cpu")
        return (
            int(m.group("pid")), int(cpu_str) if cpu_str else None, m.group("time"),
            m.group("event"), m.group("addr"),
            m.group("sym") or "", m.group("dso") or "",
        )
    m = _COMPACT_RE.match(line)
    if m:
        return (int(m.group("pid")), None, m.group("time"), "", m.group("addr"), "", "")
    return None


def _build_row(run_id: str, sample_id: int, phase_boundaries, parsed: tuple) -> dict:
    pid, cpu, time_str, event, addr_hex, sym, dso = parsed
    addr = int(addr_hex, 16)
    ts_ns = _time_to_ns(time_str)
    sym_base = sym.split("+")[0] if "+" in sym else sym
    return {
        "run_id": run_id,
        "sample_id": sample_id,
        "timestamp_ns": ts_ns,
        "phase": _annotate_phase(ts_ns, phase_boundaries),
        "address": addr,
        "address_page": addr >> 12,
        "op_type": _event_to_op_type(event),
        "symbol": sym_base,
        "dso": dso,
        "region_type": "unknown",
        "region_id": 0,
        "access_latency_ns": None,
        "cache_level": None,
        "numa_node": None,
        "cpu_id": cpu,
        "thread_id": pid,
        "token_index": None,
        "context_tokens": None,
    }


def _iter_source(source: Union[str, Path, IO]) -> Generator[str, None, None]:
    if hasattr(source, "read"):
        yield from source
    else:
        with open(source, "r", encoding="utf-8", errors="replace") as fh:
            yield from fh


# ── Public API ────────────────────────────────────────────────────────────────

def parse_perf_script(
    source: Union[str, Path, IO],
    run_id: str,
    phase_boundaries: Optional[List[dict]] = None,
) -> Generator[dict, None, None]:
    """
    Stream-parse perf script output, yielding one canonical row dict per sample.

    Accepts both full and compact formats (see module docstring).  Lines that
    match neither format are skipped and counted; statistics are logged at INFO.

    Args:
        source: file path, Path, or readable IO (supports StringIO)
        run_id: written into every yielded row
        phase_boundaries: optional [{phase, start_ns, end_ns}]
    """
    sample_id = 0
    total_lines = 0
    skipped_lines = 0
    first_skipped: list = []

    for line in _iter_source(source):
        total_lines += 1
        parsed = _parse_line(line)
        if parsed is None:
            skipped_lines += 1
            if len(first_skipped) < 5:
                first_skipped.append(line.rstrip())
            continue
        yield _build_row(run_id, sample_id, phase_boundaries, parsed)
        sample_id += 1

    parsed_count = total_lines - skipped_lines
    log.info(
        "perf_script_parser: %d lines, %d parsed (%.1f%%), %d skipped",
        total_lines, parsed_count,
        100.0 * parsed_count / total_lines if total_lines else 0.0,
        skipped_lines,
    )
    if first_skipped:
        log.debug("First skipped lines: %s", first_skipped)

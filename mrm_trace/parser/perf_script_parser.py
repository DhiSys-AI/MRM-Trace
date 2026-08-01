"""
Streaming parser for `perf script` text output.

ADR-2: never accumulates the full trace in memory.
Each parsed line is yielded immediately as a dict.

perf 6.8 format with -F comm,pid,time,event,addr,sym,dso,weight,data_src:

  python 1234 12345.678901: cpu/mem-loads,ldlat=30/P: 7f00deadbeef  [sym+0x10 /lib/foo.so]  1a05080144 |OP LOAD|LVL L1 hit|SNP None|TLB L1 or L2 hit|LCK No|BLK N/A  61

Fields after addr:
  [sym+offset /dso]  — present only when the address resolves to a known symbol
  data_src_raw       — hex without 0x prefix
  |decoded_str|      — human-readable data_src (cache level, snoop, TLB, ...)
  weight             — PEBS load-latency cycles (0 for stores)

Older format without weight/data_src (still supported):
  python  1234 12345.678901: cpu/mem-loads/P: 7f00deadbeef sym+0x10 /lib/foo.so
"""

import logging
import re
from pathlib import Path
from typing import Generator, IO, List, Optional, Union

from mrm_trace.parser.schema import OpType

log = logging.getLogger(__name__)

# ── Regex patterns ────────────────────────────────────────────────────────────

# Captures: pid, cpu (optional), time, event, addr.
# Everything after addr is handled by _parse_mem_tail().
_FULL_RE = re.compile(
    r"^\s*\S.*?\s+(?P<pid>\d+)(?:\s+\[(?P<cpu>\d+)\])?\s+"
    r"(?P<time>\d+\.\d+):\s+(?:\d+\s+)?(?P<event>[^:]+):\s+"
    r"(?P<addr>[0-9a-fA-F]+)",
    re.ASCII,
)

# Compact format: pid  time:  addr
_COMPACT_RE = re.compile(
    r"^\s*(?P<pid>\d+)\s+(?P<time>\d+\.\d+):\s+(?P<addr>[0-9a-fA-F]+)\s*$",
    re.ASCII,
)

_STORE_RE = re.compile(r"mem-store", re.IGNORECASE)

# Cache level from the decoded data_src string: |LVL L1 hit|, |LVL LFB/MAB hit|, etc.
_LVL_DECODED_RE = re.compile(r"\|LVL\s+([\w/]+)", re.IGNORECASE)

# Normalise decoded level label to canonical name (checked in order — first match wins)
_LVL_CANONICAL = [
    ("DRAM", "DRAM"), ("RAM", "DRAM"),
    ("L4", "L4"), ("L3", "L3"), ("L2", "L2"),
    ("LFB", "LFB"), ("MAB", "LFB"),
    ("L1", "L1"),
]



# ── Stateless helpers ─────────────────────────────────────────────────────────

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


def _lvl_from_decoded(decoded: str) -> Optional[str]:
    """Extract canonical cache level from perf decoded data_src string."""
    m = _LVL_DECODED_RE.search(decoded)
    if not m:
        return None
    lvl_raw = m.group(1).upper()
    for key, canonical in _LVL_CANONICAL:
        if key in lvl_raw:
            return canonical
    return None


def _numa_from_hex(hex_str: str) -> Optional[int]:
    """Extract NUMA remote bit (bit 37) from raw data_src hex without 0x prefix."""
    try:
        return (int(hex_str, 16) >> 37) & 1
    except ValueError:
        return None


def _parse_before_pipe(tokens: list) -> "tuple[str, str, Optional[float]]":
    """
    Parse the tokens appearing before the decoded |...| section.

    Three-token case: sym+offset  /dso  data_src_hex
    One-token case:  data_src_hex  (address unresolved)
    """
    if len(tokens) >= 3:
        return tokens[0].split("+")[0], tokens[1], _numa_from_hex(tokens[2])
    if len(tokens) == 1:
        return "", "", _numa_from_hex(tokens[0])
    return "", "", None


_UNRESOLVED = "[unknown]"

# End of the decoded data_src block.  Its final "|" precedes the BLK field, so
# rfind("|") lands *inside* the block rather than after it — the fields that
# follow (weight, ip, sym, dso) must be measured from the end of BLK instead.
_DECODED_END_RE = re.compile(r"\|BLK\s+\S+")


def _parse_after_data_src(text: str) -> "tuple[Optional[int], str, str]":
    """
    Parse the fields following the decoded data_src: ``weight [ip sym (dso)]``.

    ``perf script`` only emits sym and dso when ``ip`` is also in the ``-F``
    list, and appends ``ip sym (dso)`` *after* the weight:

        with ip:     ``   39     5d6a80 _PyEval_EvalFrameDefault (/usr/bin/python3.12)``
        without ip:  ``   39``

    Returns (weight_cycles, sym, dso); unresolved ``[unknown]`` entries become
    empty strings so symbol coverage reflects genuine resolution.
    """
    parts = text.split()
    if not parts:
        return None, "", ""

    weight = int(parts[0]) if parts[0].isdigit() else None
    if len(parts) < 3:
        return weight, "", ""

    # parts[1] is the ip.  The dso is the final parenthesised token; anything
    # between it and the ip is the symbol, which may contain spaces once C++
    # names are demangled.
    sym_tokens, dso = parts[2:], ""
    if sym_tokens[-1].startswith("(") and sym_tokens[-1].endswith(")"):
        dso = sym_tokens[-1][1:-1]
        sym_tokens = sym_tokens[:-1]

    sym = " ".join(sym_tokens).split("+")[0]
    return weight, "" if sym == _UNRESOLVED else sym, "" if dso == _UNRESOLVED else dso


def _parse_mem_tail(
    tail: str,
) -> "tuple[str, str, Optional[int], Optional[str], Optional[float]]":
    """
    Parse everything after the addr field in a perf mem script line.

    Handles three formats:
      With ip (current):
        data_src_hex  |decoded|  weight  ip  sym (dso)
      Without ip (traces collected before ip was added to -F):
        [sym+off /dso]  data_src_hex  |decoded|  weight
      Old (no weight/data_src):
        [sym+off /dso]

    Returns (sym, dso, weight_cycles, cache_level, numa_node).
    """
    pipe_idx = tail.find("|")

    if pipe_idx < 0:
        # Old format — just sym and optional dso
        parts = tail.split()
        sym = parts[0].split("+")[0] if parts else ""
        dso = parts[1] if len(parts) > 1 else ""
        return sym, dso, None, None, None

    end_match = _DECODED_END_RE.search(tail)
    decoded_end = end_match.end() if end_match else tail.rfind("|") + 1
    decoded = tail[pipe_idx:decoded_end]
    cache_level = _lvl_from_decoded(decoded)

    weight, sym, dso = _parse_after_data_src(tail[decoded_end:])

    before_sym, before_dso, numa_node = _parse_before_pipe(tail[:pipe_idx].split())
    # Pre-ip traces carry sym/dso ahead of the pipe; keep them when present.
    return sym or before_sym, dso or before_dso, weight, cache_level, numa_node


def _parse_line(line: str) -> "Optional[tuple]":
    """
    Parse one perf script line.

    Returns (pid, cpu, time_str, event, addr_hex, sym, dso, weight, cache_level, numa_node)
    or None if the line matches neither known format.
    """
    m = _FULL_RE.match(line)
    if m:
        cpu_str = m.group("cpu")
        tail = line[m.end():]
        sym, dso, weight, cache_level, numa_node = _parse_mem_tail(tail)
        return (
            int(m.group("pid")), int(cpu_str) if cpu_str else None, m.group("time"),
            m.group("event"), m.group("addr"),
            sym, dso, weight, cache_level, numa_node,
        )
    m = _COMPACT_RE.match(line)
    if m:
        return (int(m.group("pid")), None, m.group("time"), "", m.group("addr"),
                "", "", None, None, None)
    return None


# x86-64 kernel addresses have bit 47+ set: 0xffff800000000000 and above.
# Perf mem can capture kernel events even in spawn mode; skip them so the
# trace contains only user-space access samples.
_KERNEL_ADDR_THRESHOLD = 0x0000_8000_0000_0000


def _build_row(run_id: str, sample_id: int, phase_boundaries, parsed: tuple) -> dict:
    pid, cpu, time_str, event, addr_hex, sym, dso, weight, cache_level, numa_node = parsed
    addr = int(addr_hex, 16)
    if addr >= _KERNEL_ADDR_THRESHOLD:
        return {}
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
        "access_latency_ns": weight,
        "cache_level": cache_level,
        "numa_node": numa_node,
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

    Handles the perf 6.8 mem-event format with decoded data_src and trailing
    weight field, as well as older formats without those fields.

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
        row = _build_row(run_id, sample_id, phase_boundaries, parsed)
        if not row:
            skipped_lines += 1
            continue
        yield row
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

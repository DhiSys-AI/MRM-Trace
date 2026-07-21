"""
Post-parse quality gate for trace DataFrames.

Call check_trace_quality() immediately after parsing to catch broken
collectors/parsers before the expensive labelling/analysis/export steps.

Exit codes written to the returned TraceQA namedtuple:
  PASS  — columns populated at expected levels for the collector type
  WARN  — partial coverage; usable but some metrics will be impaired
  FAIL  — critical columns empty; run should be discarded
"""

import sys
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

# Minimum non-null fraction required for each key column, by collector type.
# perf mem should deliver 100% latency and ~100% cache_level (stores are 0-weight
# but still have a cache_level entry; only a tiny minority are truly null).
_THRESHOLDS = {
    "perf_mem": {
        "access_latency_ns": (0.95, 0.50),   # (pass_min, warn_min)
        "cache_level":       (0.85, 0.50),
    },
    "memray": {
        "access_latency_ns": (0.0, 0.0),      # always 0 for memray — not a failure
        "cache_level":       (0.0, 0.0),
    },
    "default": {
        "access_latency_ns": (0.50, 0.10),
        "cache_level":       (0.50, 0.10),
    },
}

_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


@dataclass
class TraceQA:
    status: str                          # "PASS" | "WARN" | "FAIL"
    row_count: int
    latency_coverage: float              # fraction non-null
    cache_level_coverage: float
    symbol_coverage: float               # fraction non-empty
    region_classified: float             # fraction region_type != "unknown"
    issues: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status != "FAIL"


def check_trace_quality(
    trace: pd.DataFrame,
    collector_type: str = "perf_mem",
    *,
    print_report: bool = True,
    fail_fast: bool = False,
) -> TraceQA:
    """
    Validate key column coverage in *trace* and print a quality report.

    Args:
        trace:          parsed trace DataFrame
        collector_type: "perf_mem" | "memray" | "default"
        print_report:   print the report to stdout
        fail_fast:      call sys.exit(1) if status is FAIL

    Returns:
        TraceQA with status and per-column coverage fractions.
    """
    n = len(trace)
    issues = []

    if n == 0:
        qa = TraceQA("FAIL", 0, 0.0, 0.0, 0.0, 0.0, ["No rows parsed"])
        if print_report:
            _print(qa, collector_type)
        if fail_fast:
            sys.exit(1)
        return qa

    thresholds = _THRESHOLDS.get(collector_type, _THRESHOLDS["default"])

    lat_cov = float(trace["access_latency_ns"].notna().sum()) / n
    cl_cov  = float(trace["cache_level"].notna().sum()) / n
    sym_cov = float((trace["symbol"] != "").sum()) / n if "symbol" in trace.columns else 0.0
    reg_cov = (
        float((trace["region_type"] != "unknown").sum()) / n
        if "region_type" in trace.columns else 0.0
    )

    status = "PASS"

    for col, cov, key in [
        ("access_latency_ns", lat_cov, "access_latency_ns"),
        ("cache_level",       cl_cov,  "cache_level"),
    ]:
        pass_min, warn_min = thresholds.get(key, (0.5, 0.1))
        if cov < warn_min:
            issues.append(f"{col}: {cov:.0%} coverage (need ≥{warn_min:.0%})")
            status = "FAIL"
        elif cov < pass_min:
            issues.append(f"{col}: {cov:.0%} coverage (expected ≥{pass_min:.0%})")
            if status != "FAIL":
                status = "WARN"

    qa = TraceQA(status, n, lat_cov, cl_cov, sym_cov, reg_cov, issues)

    if print_report:
        _print(qa, collector_type)
    if fail_fast and not qa.ok:
        sys.exit(1)
    return qa


def _colour(status: str) -> str:
    return {
        "PASS": f"{_GREEN}{_BOLD}PASS{_RESET}",
        "WARN": f"{_YELLOW}{_BOLD}WARN{_RESET}",
        "FAIL": f"{_RED}{_BOLD}FAIL{_RESET}",
    }.get(status, status)


def _bar(frac: float, width: int = 20) -> str:
    filled = round(frac * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {frac:5.1%}"


def _print(qa: TraceQA, collector_type: str) -> None:
    print(f"\n  Trace quality gate [{collector_type}] — {_colour(qa.status)}")
    print(f"    Rows parsed          : {qa.row_count:,}")
    print(f"    access_latency_ns    : {_bar(qa.latency_coverage)}")
    print(f"    cache_level          : {_bar(qa.cache_level_coverage)}")
    print(f"    symbol (non-empty)   : {_bar(qa.symbol_coverage)}")
    print(f"    region classified    : {_bar(qa.region_classified)}")
    if qa.issues:
        for msg in qa.issues:
            print(f"    {_RED}✗ {msg}{_RESET}")
    print()

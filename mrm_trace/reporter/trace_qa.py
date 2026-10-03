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

import pandas as pd

# Minimum non-null fraction required for each key column, by collector type.
# perf mem should deliver 100% latency and ~100% cache_level (stores are 0-weight
# but still have a cache_level entry; only a tiny minority are truly null).
_THRESHOLDS = {
    "perf_mem": {
        "access_latency_ns": (0.95, 0.50),   # (pass_min, warn_min)
        "cache_level":       (0.85, 0.50),
        "phase":             (0.30, 0.001),
    },
    # Post-labelling check: region classification must be non-trivial.
    # weight + kv_cache rows depend on /proc/maps injection; < 5% means
    # address_regions didn't register or the model backend isn't llama.cpp.
    "perf_mem_labelled": {
        "access_latency_ns": (0.95, 0.50),
        "cache_level":       (0.85, 0.50),
        "region_classified": (0.05, 0.001),  # FAIL if literally zero
        "phase":             (0.30, 0.001),
    },
    "memray": {
        "access_latency_ns": (0.0, 0.0),      # always 0 for memray — not a failure
        "cache_level":       (0.0, 0.0),
        "phase":             (0.0, 0.0),      # memray timestamps are 0; phase can't align
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
    # Declared after issues so existing positional construction keeps working.
    issues: list = field(default_factory=list)
    phase_coverage: float = 0.0          # fraction with a non-null phase label

    @property
    def ok(self) -> bool:
        return self.status != "FAIL"


def _check_column(col: str, cov: float, thresholds: dict, issues: list) -> str:
    """Return updated status after checking one coverage value against thresholds."""
    limits = thresholds.get(col)
    if limits is None:
        return "PASS"
    pass_min, warn_min = limits
    if cov < warn_min:
        issues.append(f"{col}: {cov:.1%} coverage (need ≥{warn_min:.1%})")
        return "FAIL"
    if cov < pass_min:
        issues.append(f"{col}: {cov:.1%} coverage (expected ≥{pass_min:.1%})")
        return "WARN"
    return "PASS"


def _merge_status(current: str, new: str) -> str:
    order = {"PASS": 0, "WARN": 1, "FAIL": 2}
    return current if order[current] >= order[new] else new


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
        collector_type: "perf_mem" | "perf_mem_labelled" | "memray" | "default"
        print_report:   print the report to stdout
        fail_fast:      call sys.exit(1) if status is FAIL

    Returns:
        TraceQA with status and per-column coverage fractions.
    """
    n = len(trace)
    # A phase column that is present but entirely NULL means the phase boundaries
    # and the sample timestamps are in different clock domains — phase-aware
    # labelling (ADR-3, spec §5) is then silently disabled for the whole run.
    # A trace with no phase column at all is a different thing (synthetic
    # fixtures, collectors with no phase concept) and is not gated.
    has_phase_col = "phase" in trace.columns
    return check_trace_quality_counts(
        rows=n,
        latency_non_null=int(trace["access_latency_ns"].notna().sum()) if n else 0,
        cache_level_non_null=int(trace["cache_level"].notna().sum()) if n else 0,
        symbol_non_empty=(
            int((trace["symbol"] != "").sum()) if n and "symbol" in trace.columns else 0
        ),
        region_classified=(
            int((trace["region_type"] != "unknown").sum())
            if n and "region_type" in trace.columns else 0
        ),
        phase_non_null=int(trace["phase"].notna().sum()) if n and has_phase_col else 0,
        has_phase_col=has_phase_col,
        collector_type=collector_type,
        print_report=print_report,
        fail_fast=fail_fast,
    )


def check_trace_quality_counts(
    *,
    rows: int,
    latency_non_null: int,
    cache_level_non_null: int,
    symbol_non_empty: int,
    region_classified: int,
    phase_non_null: int = 0,
    has_phase_col: bool = True,
    collector_type: str = "perf_mem",
    print_report: bool = True,
    fail_fast: bool = False,
) -> TraceQA:
    """
    Same gate as :func:`check_trace_quality`, driven by counts not a DataFrame.

    A trace large enough to need chunked parsing (ADR-2) is never held in one
    frame, so its coverage has to be tallied while streaming.  Both entry points
    share this evaluation so thresholds and the printed report have one
    definition.

    Args:
        rows:                 total rows seen
        latency_non_null:     rows with a non-null access_latency_ns
        cache_level_non_null: rows with a non-null cache_level
        symbol_non_empty:     rows with a non-empty symbol
        region_classified:    rows whose region_type is not "unknown"
        phase_non_null:       rows with a non-null phase
        has_phase_col:        False when the trace has no phase column at all,
                              which is not gated (see check_trace_quality)
    """
    issues: list = []

    if rows == 0:
        qa = TraceQA("FAIL", 0, 0.0, 0.0, 0.0, 0.0, ["No rows parsed"])
        if print_report:
            _print(qa, collector_type)
        if fail_fast:
            sys.exit(1)
        return qa

    thresholds = _THRESHOLDS.get(collector_type, _THRESHOLDS["default"])

    lat_cov   = latency_non_null / rows
    cl_cov    = cache_level_non_null / rows
    sym_cov   = symbol_non_empty / rows
    reg_cov   = region_classified / rows
    phase_cov = phase_non_null / rows if has_phase_col else 0.0

    status = "PASS"
    status = _merge_status(status, _check_column("access_latency_ns", lat_cov, thresholds, issues))
    status = _merge_status(status, _check_column("cache_level",       cl_cov,  thresholds, issues))
    status = _merge_status(status, _check_column("region_classified", reg_cov, thresholds, issues))
    if has_phase_col:
        status = _merge_status(status, _check_column("phase", phase_cov, thresholds, issues))

    qa = TraceQA(status, rows, lat_cov, cl_cov, sym_cov, reg_cov, issues, phase_cov)

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
    print(f"    phase labelled       : {_bar(qa.phase_coverage)}")
    if qa.issues:
        for msg in qa.issues:
            print(f"    {_RED}✗ {msg}{_RESET}")
    print()

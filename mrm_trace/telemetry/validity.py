"""
Run validity classification.

Observer-effect thresholds (from CLAUDE.md):
  clean       : observer CPU < 10%  AND observer memory within the clean bound
                AND throttle == 0 AND baseline_cpu < 15%
  marginal    : observer CPU < 20%  AND observer memory within the marginal bound
                AND throttle ≤ 2
  contaminated: anything worse than marginal

The observer memory bound depends on where the collector runs, because the two
cases perturb the target by different mechanisms:

  in-process (memray)       : the tracer allocates inside the target's address
                              space, so its cost is naturally expressed as a
                              fraction of target RSS — < 0.05 clean, < 0.15
                              marginal.
  out-of-process (perf mem) : the collector is a separate process whose
                              footprint is set by CPU count, ring-buffer size
                              and flush rate — not by model size.  Measured:
                              perf peaked at 2512 MB against a 1478 MB target,
                              i.e. a ratio of 1.70, on a 31 GB machine with
                              ~27 GB free.  Nothing was evicted and nothing
                              swapped, yet the ratio rule marked every run in
                              the sweep contaminated.  What actually matters
                              here is whether the collector consumed enough of
                              *system* memory to pressure the target, so the
                              bound is a fraction of total RAM — < 0.10 clean,
                              < 0.25 marginal.

Temperature thresholds (from experiment-workplan.md §9.2):
  TEMP_RUNNING_WARN_C  = 75  : running warm — log but continue
  TEMP_WARN_C          = 85  : thermal warning — at most marginal (potentially contaminated)
  TEMP_CONTAMINATED_C  = 90  : hard throttle likely — contaminated, discard from primary dataset
"""

from dataclasses import dataclass
from typing import Optional

_CLEAN_CPU = 10.0
_CLEAN_MEM_FRAC = 0.05
_CLEAN_THROTTLE = 0
_CLEAN_BASELINE_CPU = 15.0

_MARGINAL_CPU = 20.0
_MARGINAL_MEM_FRAC = 0.15
_MARGINAL_THROTTLE = 2
_MARGINAL_BASELINE_CPU = 30.0

# Out-of-process collectors: bound against total system memory instead (see
# the module docstring for why the target-RSS ratio does not apply to them).
# Fraction of the sampling cap the achieved rate must reach before a mid-run
# cap reduction is treated as having clipped the trace.
_CAP_BINDING_FRACTION = 0.5

_CLEAN_SYSTEM_MEM_FRAC = 0.10
_MARGINAL_SYSTEM_MEM_FRAC = 0.25

# Temperature thresholds — exported so experiment_runner.py can use the same values.
TEMP_RUNNING_WARN_C: float = 75.0   # running warm; log but continue
TEMP_WARN_C: float = 85.0           # thermal warning; run is at most marginal
TEMP_CONTAMINATED_C: float = 90.0   # hard throttle likely; run is contaminated


@dataclass
class ValidityResult:
    """Output of run validity classification."""

    run_validity: str                        # "clean" | "marginal" | "contaminated"
    observer_cpu_overhead_pct_max: float
    observer_memory_frac: float
    thermal_throttle_events: int
    baseline_cpu_pct: float
    io_contention_detected: bool
    failure_reasons: list                    # why it degraded below clean/marginal


def classify_run(
    observer_cpu_overhead_pct_max: float,
    observer_memory_frac: float,
    thermal_throttle_events: int,
    baseline_cpu_pct: float = 0.0,
    io_contention_detected: bool = False,
    peak_temp_c: float = 0.0,
    observer_memory_system_frac: Optional[float] = None,
    sample_rate_lowered: bool = False,
) -> ValidityResult:
    """
    Classify a run as clean, marginal, or contaminated.

    Parameters
    ----------
    observer_cpu_overhead_pct_max : peak CPU % of the observer/tracer process
    observer_memory_frac          : peak observer RSS / peak target RSS.  The
                                    memory gate for an in-process tracer.
    thermal_throttle_events       : CPU thermal throttle events during run
    baseline_cpu_pct              : mean system CPU before the run started
    io_contention_detected        : True if observer caused significant I/O
    peak_temp_c                   : peak CPU package temperature during run (°C)
    observer_memory_system_frac   : peak observer RSS / total system RAM.  Pass
                                    this for an out-of-process collector such
                                    as perf mem and it replaces the target-RSS
                                    ratio as the memory gate; see the module
                                    docstring.  None keeps the ratio rule.
    sample_rate_lowered           : True if the kernel reduced
                                    perf_event_max_sample_rate during the run
                                    *and* the achieved rate was close enough to
                                    the cap for the reduction to have clipped
                                    it.  A cap reduction alone is not evidence:
                                    measured on a real run, the cap fell
                                    100,000 -> 76,000 while the achieved rate
                                    was 5,049/s/CPU — 6.6% of the cap — and the
                                    sample count differed from a steady-cap run
                                    by 2.4%.  Flagging every reduction would
                                    mark nearly every run marginal for no
                                    effect.  See classify_from_metadata.

    Returns
    -------
    ValidityResult
    """
    failures = []
    by_system = observer_memory_system_frac is not None

    if observer_cpu_overhead_pct_max >= _CLEAN_CPU:
        failures.append(
            f"observer_cpu {observer_cpu_overhead_pct_max:.1f}% ≥ {_CLEAN_CPU}% (clean threshold)"
        )
    if by_system:
        if observer_memory_system_frac >= _CLEAN_SYSTEM_MEM_FRAC:
            failures.append(
                f"observer_mem {observer_memory_system_frac:.1%} of system RAM "
                f"≥ {_CLEAN_SYSTEM_MEM_FRAC:.0%} (clean threshold)"
            )
    elif observer_memory_frac >= _CLEAN_MEM_FRAC:
        failures.append(
            f"observer_mem_frac {observer_memory_frac:.3f} ≥ {_CLEAN_MEM_FRAC} (clean threshold)"
        )
    if thermal_throttle_events > _CLEAN_THROTTLE:
        failures.append(f"thermal_throttle_events {thermal_throttle_events} > 0")
    if baseline_cpu_pct >= _CLEAN_BASELINE_CPU:
        failures.append(
            f"baseline_cpu {baseline_cpu_pct:.1f}% ≥ {_CLEAN_BASELINE_CPU}%"
        )
    if io_contention_detected:
        failures.append("io_contention_detected=True")
    if sample_rate_lowered:
        failures.append(
            "kernel lowered perf_event_max_sample_rate during the run while "
            "sampling near the cap — density was not uniform over the "
            "collection, biasing retention"
        )
    if peak_temp_c > TEMP_CONTAMINATED_C:
        failures.append(
            f"peak_temp {peak_temp_c:.1f}°C > {TEMP_CONTAMINATED_C}°C "
            f"(hard throttle likely — contaminated)"
        )
    elif peak_temp_c > TEMP_WARN_C:
        failures.append(
            f"peak_temp {peak_temp_c:.1f}°C > {TEMP_WARN_C}°C (thermal warning — at most marginal)"
        )

    is_clean = len(failures) == 0

    memory_within_marginal = (
        observer_memory_system_frac < _MARGINAL_SYSTEM_MEM_FRAC if by_system
        else observer_memory_frac < _MARGINAL_MEM_FRAC
    )
    is_marginal = (
        observer_cpu_overhead_pct_max < _MARGINAL_CPU
        and memory_within_marginal
        and thermal_throttle_events <= _MARGINAL_THROTTLE
        and baseline_cpu_pct < _MARGINAL_BASELINE_CPU
        and not io_contention_detected
        and peak_temp_c <= TEMP_CONTAMINATED_C
    )

    if is_clean:
        verdict = "clean"
    elif is_marginal:
        verdict = "marginal"
    else:
        verdict = "contaminated"

    return ValidityResult(
        run_validity=verdict,
        observer_cpu_overhead_pct_max=observer_cpu_overhead_pct_max,
        observer_memory_frac=observer_memory_frac,
        thermal_throttle_events=thermal_throttle_events,
        baseline_cpu_pct=baseline_cpu_pct,
        io_contention_detected=io_contention_detected,
        failure_reasons=failures,
    )


def classify_from_summary(summary: dict, target_rss_mb: float = 0.0) -> ValidityResult:
    """
    Classify from an observer_effect_summary dict (as stored in metadata.json).

    If *target_rss_mb* is provided, memory fraction is computed; otherwise the
    pre-computed value from the summary is used if present.
    """
    # Use mean for the gate — it reflects sustained overhead, not startup/teardown spikes.
    # Fall back to max for old metadata.json files that pre-date the mean key.
    cpu = summary.get(
        "observer_cpu_overhead_pct_mean",
        summary.get("observer_cpu_overhead_pct_max", 0.0),
    )
    mem_mb = summary.get("observer_memory_overhead_mb_peak", 0.0)
    throttle = summary.get("thermal_throttle_events", 0)
    baseline = summary.get("baseline_cpu_pct", 0.0)
    io_c = summary.get("io_contention_detected", False)

    if target_rss_mb > 0:
        mem_frac = mem_mb / target_rss_mb
    else:
        mem_frac = summary.get("observer_memory_overhead_frac", 0.0)

    # Present only for out-of-process collectors; absent for memray and for
    # metadata written before the field existed, which keeps the ratio rule.
    system_frac = summary.get("observer_memory_system_frac")
    if system_frac is None:
        total_mb = summary.get("system_memory_mb", 0.0)
        if total_mb > 0 and mem_mb > 0:
            system_frac = mem_mb / total_mb

    return classify_run(
        observer_cpu_overhead_pct_max=cpu,
        observer_memory_frac=mem_frac,
        thermal_throttle_events=throttle,
        baseline_cpu_pct=baseline,
        io_contention_detected=io_c,
        peak_temp_c=summary.get("peak_temp_c", 0.0),
        observer_memory_system_frac=system_frac,
    )


def classify_from_metadata(metadata: dict) -> ValidityResult:
    """
    Re-derive a run's validity from a metadata.json dict.

    Validity is a label computed from numbers the run already recorded, so it
    can always be re-derived without re-collecting anything.  A fresh run and a
    reclassification of its metadata must reach the same verdict, which is why
    both go through this function.
    """
    thermal = metadata.get("thermal") or {}
    oe = metadata.get("observer_effect") or {}
    hardware = metadata.get("hardware") or {}
    collector = metadata.get("collector") or {}

    # A mid-run reduction of the sampling cap only matters if the cap was
    # binding.  Compare the achieved per-CPU rate against the lower of the two
    # caps: below _CAP_BINDING_FRACTION of it, clipping is implausible and the
    # reduction changed nothing.
    rate_before = collector.get("perf_event_max_sample_rate_before")
    rate_after = collector.get("perf_event_max_sample_rate_after")
    achieved = collector.get("achieved_sample_rate_per_cpu")
    cap_fell = bool(rate_before and rate_after and rate_after < rate_before)
    if cap_fell and achieved:
        sample_rate_lowered = achieved >= _CAP_BINDING_FRACTION * min(
            rate_before, rate_after)
    else:
        # No achieved rate recorded (older metadata): fall back to the cap
        # change alone rather than silently passing a run that may be clipped.
        sample_rate_lowered = cap_fell

    system_frac = oe.get("observer_memory_system_frac")
    if system_frac is None:
        # Metadata written before the field existed: reconstruct it from the
        # absolute peak and whichever total-RAM figure the run recorded.
        total_mb = oe.get("system_memory_mb") or hardware.get("mem_total_mb") or 0.0
        peak_mb = oe.get("observer_memory_overhead_mb_peak", 0.0)
        if total_mb > 0 and peak_mb > 0:
            system_frac = peak_mb / total_mb

    baseline_cpu = oe.get("baseline_cpu_pct", hardware.get("baseline_cpu_pct", 0.0))

    return classify_run(
        observer_cpu_overhead_pct_max=oe.get("observer_cpu_overhead_pct_mean", 0.0),
        observer_memory_frac=oe.get("observer_memory_overhead_frac", 0.0),
        thermal_throttle_events=thermal.get("throttle_events", 0),
        baseline_cpu_pct=baseline_cpu,
        io_contention_detected=oe.get("io_contention_detected", False),
        peak_temp_c=thermal.get("peak_temp_c", 0.0),
        observer_memory_system_frac=system_frac,
        sample_rate_lowered=sample_rate_lowered,
    )

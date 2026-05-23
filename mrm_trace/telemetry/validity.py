"""
Run validity classification.

Thresholds (from CLAUDE.md):
  clean       : observer CPU < 10%  AND observer mem_frac < 0.05 AND throttle == 0
                AND baseline_cpu < 15%
  marginal    : observer CPU < 20%  AND observer mem_frac < 0.15 AND throttle ≤ 2
  contaminated: anything worse than marginal
"""

from dataclasses import dataclass

_CLEAN_CPU = 10.0
_CLEAN_MEM_FRAC = 0.05
_CLEAN_THROTTLE = 0
_CLEAN_BASELINE_CPU = 15.0

_MARGINAL_CPU = 20.0
_MARGINAL_MEM_FRAC = 0.15
_MARGINAL_THROTTLE = 2


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
) -> ValidityResult:
    """
    Classify a run as clean, marginal, or contaminated.

    Parameters
    ----------
    observer_cpu_overhead_pct_max : peak CPU % of the observer/tracer process
    observer_memory_frac          : peak observer RSS / peak target RSS
    thermal_throttle_events       : CPU thermal throttle events during run
    baseline_cpu_pct              : mean system CPU before the run started
    io_contention_detected        : True if observer caused significant I/O

    Returns
    -------
    ValidityResult
    """
    failures = []

    if observer_cpu_overhead_pct_max >= _CLEAN_CPU:
        failures.append(
            f"observer_cpu {observer_cpu_overhead_pct_max:.1f}% ≥ {_CLEAN_CPU}% (clean threshold)"
        )
    if observer_memory_frac >= _CLEAN_MEM_FRAC:
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

    is_clean = len(failures) == 0

    is_marginal = (
        observer_cpu_overhead_pct_max < _MARGINAL_CPU
        and observer_memory_frac < _MARGINAL_MEM_FRAC
        and thermal_throttle_events <= _MARGINAL_THROTTLE
        and not io_contention_detected
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
    cpu = summary.get("observer_cpu_overhead_pct_max", 0.0)
    mem_mb = summary.get("observer_memory_overhead_mb_peak", 0.0)
    throttle = summary.get("thermal_throttle_events", 0)
    baseline = summary.get("baseline_cpu_pct", 0.0)
    io_c = summary.get("io_contention_detected", False)

    if target_rss_mb > 0:
        mem_frac = mem_mb / target_rss_mb
    else:
        mem_frac = summary.get("observer_memory_overhead_frac", 0.0)

    return classify_run(
        observer_cpu_overhead_pct_max=cpu,
        observer_memory_frac=mem_frac,
        thermal_throttle_events=throttle,
        baseline_cpu_pct=baseline,
        io_contention_detected=io_c,
    )

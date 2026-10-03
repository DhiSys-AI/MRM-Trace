"""
Observer memory must be judged by the mechanism that could perturb the target.

An in-process tracer (memray) allocates inside the target's address space, so
its cost is naturally a fraction of target RSS.  perf mem is a separate process
whose footprint comes from CPU count, ring-buffer size and flush rate — nothing
to do with model size.  Measured on a real run: perf peaked at 2512 MB against
a 1478 MB target, a ratio of 1.70, on a 31 GB machine with ~27 GB free. Nothing
was evicted and nothing swapped, yet the target-RSS ratio marked every run in
the sweep contaminated — 8 of 8 — and would keep doing so for every model, since
even an 8B target at ~5.5 GB gives 0.45.

A rule that excludes 100% of runs discriminates nothing. These tests pin the
corrected criterion, and that both rules still exist for their own collector.
"""

import pytest

from mrm_trace.telemetry.validity import (
    TEMP_CONTAMINATED_C,
    TEMP_WARN_C,
    classify_from_metadata,
    classify_run,
)

pytestmark = pytest.mark.validity

# The real run that was misclassified.
_PERF_RUN = {
    "thermal": {"peak_temp_c": 73.0, "throttle_events": 0},
    "observer_effect": {
        "observer_cpu_overhead_pct_mean": 4.03,
        "observer_memory_overhead_mb_peak": 2512.2,
        "observer_memory_overhead_frac": 1.699,
        "target_rss_mb_peak": 1478.5,
        "baseline_cpu_pct": 0.06,
        "io_contention_detected": False,
        "thermal_throttle_events": 0,
    },
    "hardware": {"mem_total_mb": 31742.0},
}


def test_the_real_perf_run_is_clean():
    """2512 MB of 31742 MB is 7.9% of system RAM — no pressure, no eviction."""
    result = classify_from_metadata(_PERF_RUN)
    assert result.run_validity == "clean"
    assert result.failure_reasons == []


def test_the_target_ratio_alone_would_have_condemned_it():
    """Documents the bug: same numbers, ratio rule, contaminated."""
    result = classify_run(
        observer_cpu_overhead_pct_max=4.03, observer_memory_frac=1.699,
        thermal_throttle_events=0, baseline_cpu_pct=0.06, peak_temp_c=73.0,
    )
    assert result.run_validity == "contaminated"


def test_the_ratio_rule_still_applies_to_an_in_process_tracer():
    """memray allocates inside the target, so the ratio is the right gate."""
    result = classify_run(
        observer_cpu_overhead_pct_max=2.0, observer_memory_frac=0.40,
        thermal_throttle_events=0,
    )
    assert result.run_validity == "contaminated"
    assert any("observer_mem_frac" in r for r in result.failure_reasons)


def test_a_collector_that_really_does_hog_memory_is_still_caught():
    """The corrected rule must not be a rubber stamp."""
    result = classify_run(
        observer_cpu_overhead_pct_max=2.0, observer_memory_frac=0.01,
        thermal_throttle_events=0, observer_memory_system_frac=0.60,
    )
    assert result.run_validity == "contaminated"
    assert any("system RAM" in r for r in result.failure_reasons)


@pytest.mark.parametrize("system_frac,expected", [
    (0.05, "clean"),
    (0.099, "clean"),
    (0.15, "marginal"),
    (0.249, "marginal"),
    (0.25, "contaminated"),
])
def test_system_memory_thresholds(system_frac, expected):
    result = classify_run(
        observer_cpu_overhead_pct_max=2.0, observer_memory_frac=0.01,
        thermal_throttle_events=0, observer_memory_system_frac=system_frac,
    )
    assert result.run_validity == expected


# ── the thermal rules must be untouched ──────────────────────────────────────

def test_a_genuinely_hot_run_is_still_contaminated():
    result = classify_from_metadata({
        **_PERF_RUN,
        "thermal": {"peak_temp_c": TEMP_CONTAMINATED_C + 1, "throttle_events": 0},
    })
    assert result.run_validity == "contaminated"


def test_a_warm_run_is_still_marginal():
    result = classify_from_metadata({
        **_PERF_RUN,
        "thermal": {"peak_temp_c": TEMP_WARN_C + 1, "throttle_events": 0},
    })
    assert result.run_validity == "marginal"


def test_throttling_still_costs_a_run_its_clean_status():
    result = classify_from_metadata({
        **_PERF_RUN,
        "thermal": {"peak_temp_c": 73.0, "throttle_events": 1},
    })
    assert result.run_validity == "marginal"
    assert any("throttle" in r for r in result.failure_reasons)


# ── provenance ───────────────────────────────────────────────────────────────

def test_a_verdict_always_carries_its_reasons():
    """
    A "contaminated" with no stated cause cannot be acted on, and it is what
    the data card needs to justify excluding a run.
    """
    result = classify_from_metadata({
        **_PERF_RUN,
        "thermal": {"peak_temp_c": 95.0, "throttle_events": 3},
    })
    assert result.run_validity == "contaminated"
    assert result.failure_reasons
    assert all(isinstance(r, str) and r for r in result.failure_reasons)


def test_metadata_without_the_new_field_is_reconstructed():
    """
    The 8 runs already on disk predate observer_memory_system_frac; their
    verdicts must still be derivable from the absolute peak and total RAM.
    """
    assert "observer_memory_system_frac" not in _PERF_RUN["observer_effect"]
    assert classify_from_metadata(_PERF_RUN).run_validity == "clean"


def test_the_new_field_wins_when_present():
    meta = {
        **_PERF_RUN,
        "observer_effect": {**_PERF_RUN["observer_effect"],
                            "observer_memory_system_frac": 0.80},
    }
    assert classify_from_metadata(meta).run_validity == "contaminated"


def test_missing_sections_do_not_raise():
    assert classify_from_metadata({}).run_validity in {"clean", "marginal", "contaminated"}

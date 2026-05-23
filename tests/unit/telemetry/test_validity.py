"""Unit tests for the run validity classifier."""

import pytest

from mrm_trace.telemetry.validity import classify_run, classify_from_summary


# ── clean classification ──────────────────────────────────────────────────────

def test_low_overhead_is_clean():
    result = classify_run(
        observer_cpu_overhead_pct_max=4.2,
        observer_memory_frac=0.01,
        thermal_throttle_events=0,
        baseline_cpu_pct=5.0,
    )
    assert result.run_validity == "clean"


def test_zero_overhead_is_clean():
    result = classify_run(
        observer_cpu_overhead_pct_max=0.0,
        observer_memory_frac=0.0,
        thermal_throttle_events=0,
        baseline_cpu_pct=0.0,
    )
    assert result.run_validity == "clean"


def test_clean_has_no_failure_reasons():
    result = classify_run(
        observer_cpu_overhead_pct_max=4.0,
        observer_memory_frac=0.02,
        thermal_throttle_events=0,
        baseline_cpu_pct=5.0,
    )
    assert result.failure_reasons == []


# ── clean boundary ────────────────────────────────────────────────────────────

def test_cpu_exactly_at_clean_threshold_is_not_clean():
    result = classify_run(
        observer_cpu_overhead_pct_max=10.0,
        observer_memory_frac=0.0,
        thermal_throttle_events=0,
    )
    assert result.run_validity != "clean"


def test_cpu_just_below_clean_threshold_is_clean():
    result = classify_run(
        observer_cpu_overhead_pct_max=9.99,
        observer_memory_frac=0.0,
        thermal_throttle_events=0,
    )
    assert result.run_validity == "clean"


def test_one_thermal_event_breaks_clean():
    result = classify_run(
        observer_cpu_overhead_pct_max=5.0,
        observer_memory_frac=0.01,
        thermal_throttle_events=1,
        baseline_cpu_pct=5.0,
    )
    assert result.run_validity != "clean"


def test_high_baseline_cpu_breaks_clean():
    result = classify_run(
        observer_cpu_overhead_pct_max=5.0,
        observer_memory_frac=0.01,
        thermal_throttle_events=0,
        baseline_cpu_pct=15.0,  # exactly at threshold — not clean
    )
    assert result.run_validity != "clean"


# ── marginal classification ───────────────────────────────────────────────────

def test_moderate_cpu_is_marginal():
    result = classify_run(
        observer_cpu_overhead_pct_max=15.0,
        observer_memory_frac=0.05,
        thermal_throttle_events=1,
        baseline_cpu_pct=5.0,
    )
    assert result.run_validity == "marginal"


def test_two_throttle_events_is_still_marginal():
    result = classify_run(
        observer_cpu_overhead_pct_max=5.0,
        observer_memory_frac=0.02,
        thermal_throttle_events=2,
    )
    assert result.run_validity == "marginal"


def test_three_throttle_events_is_contaminated():
    result = classify_run(
        observer_cpu_overhead_pct_max=5.0,
        observer_memory_frac=0.02,
        thermal_throttle_events=3,
    )
    assert result.run_validity == "contaminated"


# ── contaminated classification ───────────────────────────────────────────────

def test_high_cpu_is_contaminated():
    result = classify_run(
        observer_cpu_overhead_pct_max=28.5,
        observer_memory_frac=0.5,
        thermal_throttle_events=5,
    )
    assert result.run_validity == "contaminated"


def test_io_contention_triggers_contaminated():
    result = classify_run(
        observer_cpu_overhead_pct_max=5.0,
        observer_memory_frac=0.02,
        thermal_throttle_events=0,
        io_contention_detected=True,
    )
    assert result.run_validity == "contaminated"


def test_contaminated_records_failure_reasons():
    result = classify_run(
        observer_cpu_overhead_pct_max=28.5,
        observer_memory_frac=0.5,
        thermal_throttle_events=5,
    )
    assert len(result.failure_reasons) > 0


# ── classify_from_summary ─────────────────────────────────────────────────────

def test_classify_from_summary_clean():
    summary = {
        "observer_cpu_overhead_pct_max": 4.2,
        "observer_memory_overhead_mb_peak": 0.0,
        "observer_memory_overhead_frac": 0.01,
        "thermal_throttle_events": 0,
        "baseline_cpu_pct": 5.0,
        "io_contention_detected": False,
    }
    result = classify_from_summary(summary)
    assert result.run_validity == "clean"


def test_classify_from_summary_with_target_rss():
    summary = {
        "observer_cpu_overhead_pct_max": 4.2,
        "observer_memory_overhead_mb_peak": 100.0,
        "thermal_throttle_events": 0,
        "baseline_cpu_pct": 0.0,
        "io_contention_detected": False,
    }
    # target_rss=10000 → mem_frac = 100/10000 = 0.01 → clean
    result = classify_from_summary(summary, target_rss_mb=10_000.0)
    assert result.run_validity == "clean"


def test_classify_from_summary_contaminated():
    summary = {
        "observer_cpu_overhead_pct_max": 28.5,
        "observer_memory_overhead_mb_peak": 4800.0,
        "observer_memory_overhead_frac": 0.5,
        "thermal_throttle_events": 5,
        "baseline_cpu_pct": 0.0,
        "io_contention_detected": True,
    }
    result = classify_from_summary(summary)
    assert result.run_validity == "contaminated"

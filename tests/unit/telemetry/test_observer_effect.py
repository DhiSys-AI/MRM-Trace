"""Unit tests for compute_observer_effect."""

import pytest

from mrm_trace.collector.process_monitor import ProcessSample
from mrm_trace.telemetry.observer_effect import compute_observer_effect


def _sample(pid, cpu_pct, rss_mb, write_bytes=0):
    return ProcessSample(
        t_s=0.0, pid=pid, cpu_pct=cpu_pct,
        rss_mb=rss_mb, vms_mb=rss_mb * 2,
        shared_mb=0.0, threads=1, open_fds=0,
        read_bytes=0, write_bytes=write_bytes,
        voluntary_ctx_switches=0, involuntary_ctx_switches=0,
    )


def test_peak_cpu_is_p95_not_abs_max():
    # 3 samples sorted: [5.0, 8.0, 15.0]  p95_idx = max(0, int(3*0.95)-1) = max(0,1) = 1 → 8.0
    # abs_max is still stored separately for diagnostics.
    obs = [_sample(1, 5.0, 100), _sample(1, 15.0, 110), _sample(1, 8.0, 105)]
    tgt = [_sample(2, 80.0, 5000)]
    result = compute_observer_effect(obs, tgt)
    assert result["observer_cpu_overhead_pct_max"] == pytest.approx(8.0)
    assert result["observer_cpu_overhead_pct_abs_max"] == pytest.approx(15.0)


def test_peak_memory_is_max_rss():
    obs = [_sample(1, 0.0, 100), _sample(1, 0.0, 350), _sample(1, 0.0, 200)]
    tgt = [_sample(2, 0.0, 5000)]
    result = compute_observer_effect(obs, tgt)
    assert result["observer_memory_overhead_mb_peak"] == pytest.approx(350.0)


def test_memory_frac_computed_from_target_rss():
    obs = [_sample(1, 0.0, 500)]
    tgt = [_sample(2, 0.0, 10_000)]
    result = compute_observer_effect(obs, tgt)
    assert result["observer_memory_overhead_frac"] == pytest.approx(0.05)


def test_target_rss_peak_captured():
    obs = [_sample(1, 0.0, 100)]
    tgt = [_sample(2, 0.0, 8000), _sample(2, 0.0, 12000), _sample(2, 0.0, 9000)]
    result = compute_observer_effect(obs, tgt)
    assert result["target_rss_mb_peak"] == pytest.approx(12000.0)


def test_thermal_events_passed_through():
    obs = [_sample(1, 0.0, 100)]
    tgt = [_sample(2, 0.0, 5000)]
    result = compute_observer_effect(obs, tgt, thermal_throttle_events=3)
    assert result["thermal_throttle_events"] == 3


def test_no_io_contention_when_small_writes():
    obs = [
        _sample(1, 0.0, 100, write_bytes=0),
        _sample(1, 0.0, 100, write_bytes=1_000),  # 1 KB delta
    ]
    tgt = [_sample(2, 0.0, 5000)]
    result = compute_observer_effect(obs, tgt)
    assert result["io_contention_detected"] is False


def test_io_contention_when_large_writes():
    obs = [
        _sample(1, 0.0, 100, write_bytes=0),
        _sample(1, 0.0, 100, write_bytes=20 * 1_048_576),  # 20 MB delta
    ]
    tgt = [_sample(2, 0.0, 5000)]
    result = compute_observer_effect(obs, tgt)
    assert result["io_contention_detected"] is True


def test_empty_observer_samples_handled():
    tgt = [_sample(2, 0.0, 5000)]
    result = compute_observer_effect([], tgt)
    assert result["observer_cpu_overhead_pct_max"] == pytest.approx(0.0)
    assert result["observer_memory_overhead_mb_peak"] == pytest.approx(0.0)


def test_required_keys_present():
    obs = [_sample(1, 5.0, 200)]
    tgt = [_sample(2, 80.0, 8000)]
    result = compute_observer_effect(obs, tgt)
    for key in (
        "observer_cpu_overhead_pct_max", "observer_cpu_overhead_pct_abs_max",
        "observer_cpu_overhead_pct_mean",
        "observer_memory_overhead_mb_peak", "observer_memory_overhead_frac",
        "target_rss_mb_peak", "thermal_throttle_events",
        "baseline_cpu_pct", "io_contention_detected",
        "n_observer_samples", "n_target_samples",
    ):
        assert key in result, f"Missing key: {key}"

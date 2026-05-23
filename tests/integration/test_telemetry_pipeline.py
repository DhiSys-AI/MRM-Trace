"""
Integration tests for the telemetry pipeline.

Exercises the full chain: baseline → observer samples → observer_effect →
validity classification, using real psutil and in-process mocked samples.
"""

import os

import pytest

from mrm_trace.collector.process_monitor import ProcessMonitor, ProcessSample
from mrm_trace.telemetry.baseline import capture_baseline
from mrm_trace.telemetry.observer_effect import compute_observer_effect
from mrm_trace.telemetry.thermal import ThermalMonitor
from mrm_trace.telemetry.validity import classify_run, classify_from_summary


def _sample(cpu_pct, rss_mb, write_bytes=0):
    return ProcessSample(
        t_s=0.0, pid=os.getpid(), cpu_pct=cpu_pct,
        rss_mb=rss_mb, vms_mb=rss_mb,
        shared_mb=0.0, threads=1, open_fds=0,
        read_bytes=0, write_bytes=write_bytes,
        voluntary_ctx_switches=0, involuntary_ctx_switches=0,
    )


class TestBaselineCaptureIntegration:

    def test_baseline_runs_without_error(self):
        snap = capture_baseline(duration_s=0.2, poll_interval_s=0.05)
        assert snap.cpu_pct >= 0.0
        assert snap.mem_total_mb > 0

    def test_baseline_cpu_max_ge_mean(self):
        snap = capture_baseline(duration_s=0.3, poll_interval_s=0.05)
        assert snap.cpu_pct_max >= snap.cpu_pct


class TestThermalMonitorIntegration:

    def test_begin_end_returns_nonnegative_int(self):
        monitor = ThermalMonitor()
        monitor.begin()
        events = monitor.end()
        assert isinstance(events, int)
        assert events >= 0


class TestObserverEffectIntegration:

    def test_full_chain_produces_all_keys(self):
        obs = [_sample(5.0, 200), _sample(8.0, 220)]
        tgt = [_sample(70.0, 8000), _sample(75.0, 8500)]
        summary = compute_observer_effect(obs, tgt, thermal_throttle_events=0)
        for key in (
            "observer_cpu_overhead_pct_max", "observer_memory_overhead_mb_peak",
            "observer_memory_overhead_frac", "target_rss_mb_peak",
            "thermal_throttle_events", "io_contention_detected",
        ):
            assert key in summary

    def test_classify_from_observer_effect_summary(self):
        obs = [_sample(4.0, 100)]
        tgt = [_sample(80.0, 10_000)]
        summary = compute_observer_effect(obs, tgt, thermal_throttle_events=0)
        result = classify_from_summary(summary, target_rss_mb=10_000.0)
        assert result.run_validity in ("clean", "marginal", "contaminated")


class TestProcessMonitorIntegration:

    def test_monitor_collects_samples_for_self(self):
        monitor = ProcessMonitor([os.getpid()], poll_interval_ms=50)
        monitor.start()
        import time; time.sleep(0.25)
        samples = monitor.stop()
        own = [s for s in samples if s.pid == os.getpid()]
        assert len(own) >= 2

    def test_monitor_samples_feed_into_observer_effect(self):
        monitor = ProcessMonitor([os.getpid()], poll_interval_ms=50)
        monitor.start()
        import time; time.sleep(0.25)
        samples = monitor.stop()
        summary = compute_observer_effect(samples, samples, thermal_throttle_events=0)
        assert summary["n_observer_samples"] >= 1
        assert summary["n_target_samples"] >= 1

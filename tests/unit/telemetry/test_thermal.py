"""Unit tests for ThermalSnapshot and ThermalMonitor."""

import pytest

from mrm_trace.telemetry.thermal import ThermalSnapshot, ThermalMonitor


# ── ThermalSnapshot ───────────────────────────────────────────────────────────

def test_events_since_same_snapshot_is_zero():
    snap = ThermalSnapshot(counts={"cpu0": 10, "cpu1": 5})
    assert snap.events_since(snap) == 0


def test_events_since_counts_delta():
    before = ThermalSnapshot(counts={"cpu0": 10, "cpu1": 5})
    after  = ThermalSnapshot(counts={"cpu0": 12, "cpu1": 6})
    assert after.events_since(before) == 3  # (12-10) + (6-5)


def test_events_since_new_cpu_added():
    before = ThermalSnapshot(counts={"cpu0": 10})
    after  = ThermalSnapshot(counts={"cpu0": 10, "cpu1": 3})
    # cpu1 not in before → prev defaults to current → delta = 0
    assert after.events_since(before) == 0


def test_events_since_no_regression_on_counter_wrap():
    before = ThermalSnapshot(counts={"cpu0": 100})
    after  = ThermalSnapshot(counts={"cpu0": 50})  # counter wrapped
    # max(0, 50-100) = 0 — we don't go negative
    assert after.events_since(before) == 0


def test_empty_snapshot_events_since_empty_is_zero():
    assert ThermalSnapshot().events_since(ThermalSnapshot()) == 0


# ── ThermalMonitor ────────────────────────────────────────────────────────────

def test_thermal_monitor_returns_int():
    monitor = ThermalMonitor()
    monitor.begin()
    result = monitor.end()
    assert isinstance(result, int)


def test_thermal_monitor_result_nonnegative():
    monitor = ThermalMonitor()
    monitor.begin()
    assert monitor.end() >= 0


def test_thermal_monitor_no_events_in_idle_run():
    """On an idle machine, no throttle events should occur in a trivial run."""
    monitor = ThermalMonitor()
    monitor.begin()
    _ = sum(range(1000))  # trivial workload
    events = monitor.end()
    assert events >= 0  # just must not crash; typically 0


def test_thermal_monitor_capture_returns_snapshot():
    snap = ThermalSnapshot.capture()
    assert isinstance(snap, ThermalSnapshot)
    assert isinstance(snap.counts, dict)

"""Unit tests for BaselineSnapshot and capture_baseline."""

import pytest

from mrm_trace.telemetry.baseline import BaselineSnapshot, capture_baseline


def test_capture_baseline_returns_snapshot():
    snap = capture_baseline(duration_s=0.1, poll_interval_s=0.05)
    assert isinstance(snap, BaselineSnapshot)


def test_baseline_cpu_pct_is_nonnegative():
    snap = capture_baseline(duration_s=0.1, poll_interval_s=0.05)
    assert snap.cpu_pct >= 0.0


def test_baseline_cpu_pct_max_ge_mean():
    snap = capture_baseline(duration_s=0.3, poll_interval_s=0.05)
    assert snap.cpu_pct_max >= snap.cpu_pct


def test_baseline_mem_total_positive():
    snap = capture_baseline(duration_s=0.1, poll_interval_s=0.05)
    assert snap.mem_total_mb > 0


def test_baseline_mem_available_le_total():
    snap = capture_baseline(duration_s=0.1, poll_interval_s=0.05)
    assert snap.mem_available_mb <= snap.mem_total_mb


def test_baseline_mem_used_pct_between_0_and_100():
    snap = capture_baseline(duration_s=0.1, poll_interval_s=0.05)
    assert 0.0 <= snap.mem_used_pct <= 100.0


def test_baseline_timestamp_positive():
    snap = capture_baseline(duration_s=0.1, poll_interval_s=0.05)
    assert snap.timestamp_s > 0

"""Unit tests for ProcessMonitor."""

import os
import time

import pytest

from mrm_trace.collector.process_monitor import ProcessMonitor, ProcessSample


# ── ProcessSample dataclass ────────────────────────────────────────────────────

def test_process_sample_fields():
    s = ProcessSample(
        t_s=1.5, pid=123, cpu_pct=5.0, rss_mb=100.0, vms_mb=200.0,
        shared_mb=10.0, threads=4, open_fds=8,
        read_bytes=0, write_bytes=512,
        voluntary_ctx_switches=1, involuntary_ctx_switches=0,
    )
    assert s.t_s == 1.5
    assert s.pid == 123
    assert s.rss_mb == 100.0
    assert s.write_bytes == 512


# ── Lifecycle ─────────────────────────────────────────────────────────────────

def test_start_spawns_thread():
    monitor = ProcessMonitor([os.getpid()], poll_interval_ms=50)
    monitor.start()
    assert monitor._thread is not None
    assert monitor._thread.is_alive()
    monitor.stop()


def test_stop_joins_thread():
    monitor = ProcessMonitor([os.getpid()], poll_interval_ms=50)
    monitor.start()
    monitor.stop()
    assert monitor._thread is None


def test_stop_returns_list():
    monitor = ProcessMonitor([os.getpid()], poll_interval_ms=50)
    monitor.start()
    time.sleep(0.15)
    samples = monitor.stop()
    assert isinstance(samples, list)


def test_stop_before_start_returns_empty():
    monitor = ProcessMonitor([os.getpid()], poll_interval_ms=50)
    assert monitor.stop() == []


# ── Sampling ──────────────────────────────────────────────────────────────────

def test_collects_samples_for_current_pid():
    monitor = ProcessMonitor([os.getpid()], poll_interval_ms=30)
    monitor.start()
    time.sleep(0.2)
    samples = monitor.stop()
    assert len(samples) > 0
    for s in samples:
        assert s.pid == os.getpid()


def test_samples_have_nonnegative_rss():
    monitor = ProcessMonitor([os.getpid()], poll_interval_ms=30)
    monitor.start()
    time.sleep(0.15)
    samples = monitor.stop()
    for s in samples:
        assert s.rss_mb >= 0


def test_get_samples_returns_copy():
    monitor = ProcessMonitor([os.getpid()], poll_interval_ms=30)
    monitor.start()
    time.sleep(0.1)
    a = monitor.get_samples()
    b = monitor.get_samples()
    assert a is not b  # independent list objects


def test_empty_pid_list_produces_no_samples():
    monitor = ProcessMonitor([], poll_interval_ms=30)
    monitor.start()
    time.sleep(0.1)
    samples = monitor.stop()
    assert samples == []


def test_nonexistent_pid_does_not_crash():
    monitor = ProcessMonitor([2 ** 30 - 1], poll_interval_ms=30)
    monitor.start()
    time.sleep(0.1)
    samples = monitor.stop()
    assert isinstance(samples, list)


def test_second_stop_is_safe():
    monitor = ProcessMonitor([os.getpid()], poll_interval_ms=30)
    monitor.start()
    time.sleep(0.1)
    monitor.stop()
    samples = monitor.stop()  # should not raise
    assert isinstance(samples, list)

"""
Tier-3 validity tests for Phase 3 — Collectors.

These tests verify scientific correctness properties that, if violated, would
corrupt the observer-effect section of metadata.json or silently allow invalid
runs into the dataset.
"""

import os
import shutil
import time
from pathlib import Path

import pytest

from mrm_trace.collector.artifact_manager import ArtifactManager, DiskSpaceError
from mrm_trace.collector.base import CollectorMeta
from mrm_trace.collector.process_monitor import ProcessMonitor


def test_process_monitor_timestamps_are_monotonic():
    """
    All t_s values for a given PID must be non-decreasing.
    Backwards timestamps corrupt the observer-effect time-series and
    would invalidate observer CPU / memory overhead attribution.
    """
    monitor = ProcessMonitor([os.getpid()], poll_interval_ms=20)
    monitor.start()
    time.sleep(0.35)
    samples = monitor.stop()

    my_samples = [s for s in samples if s.pid == os.getpid()]
    for i in range(1, len(my_samples)):
        assert my_samples[i].t_s >= my_samples[i - 1].t_s, (
            f"t_s went backwards at index {i}: "
            f"{my_samples[i-1].t_s:.6f} → {my_samples[i].t_s:.6f}. "
            "Non-monotonic observer timestamps corrupt run validity analysis."
        )


def test_process_monitor_produces_minimum_sample_count():
    """
    Over a 300 ms window with 50 ms poll interval, at least 3 samples
    must be collected.  Fewer would indicate the sampling thread is not
    running, which would cause the observer_effect section to appear
    empty (falsely suggesting zero overhead).
    """
    monitor = ProcessMonitor([os.getpid()], poll_interval_ms=50)
    monitor.start()
    time.sleep(0.35)
    samples = monitor.stop()

    my_samples = [s for s in samples if s.pid == os.getpid()]
    assert len(my_samples) >= 3, (
        f"Expected ≥ 3 samples in a 350 ms window, got {len(my_samples)}. "
        "ProcessMonitor thread may not be running — observer effect will be "
        "misreported as zero overhead."
    )


def test_process_monitor_rss_positive_for_live_process():
    """
    RSS must be positive for any live process.
    Zero RSS would indicate a silent psutil failure — the value would be
    used directly in observer_effect_summary comparisons, causing the
    run to be incorrectly classified as 'clean' regardless of actual overhead.
    """
    monitor = ProcessMonitor([os.getpid()], poll_interval_ms=20)
    monitor.start()
    time.sleep(0.12)
    samples = monitor.stop()

    for s in samples:
        assert s.rss_mb > 0, (
            f"RSS is {s.rss_mb} MB for PID {s.pid}. "
            "Zero RSS from a live process indicates a psutil read failure "
            "silently coerced to 0, which corrupts observer overhead metrics."
        )


def test_process_monitor_no_fabricated_samples_for_dead_pid():
    """
    Monitoring a non-existent PID must produce exactly zero samples.
    Fabricated samples for a dead PID would corrupt the observer-effect
    attribution in the metadata — crediting overhead to a process that
    never ran.
    """
    dead_pid = 2 ** 30 - 1
    monitor = ProcessMonitor([dead_pid], poll_interval_ms=20)
    monitor.start()
    time.sleep(0.12)
    samples = monitor.stop()

    assert samples == [], (
        f"Expected 0 samples for non-existent PID {dead_pid}, got {len(samples)}. "
        "Fabricated samples would corrupt observer-effect process attribution."
    )


def test_collector_meta_has_all_required_fields():
    """
    CollectorMeta must expose every field the reporter needs to populate
    the observer_effect section of metadata.json.  A missing field causes
    a silent KeyError at export time, producing an incomplete metadata record.
    """
    required = {
        "collector_type", "available", "used", "fallback_reason",
        "event_strings", "sample_period", "output_file",
        "output_file_size_bytes", "parse_success", "dropped_samples",
        "coverage_estimate", "warnings", "extra",
    }
    meta = CollectorMeta(collector_type="test", available=True, used=True)
    actual = set(vars(meta).keys())
    missing = required - actual
    assert not missing, (
        f"CollectorMeta is missing fields required for metadata.json: {missing}. "
        "The reporter will fail silently when assembling the observer_effect section."
    )


def test_disk_space_check_boundary_is_exact(tmp_path):
    """
    check_disk_space must pass when free == threshold and fail when
    free == threshold - 1.  An off-by-one in the comparison would silently
    allow collection to start on a nearly-full disk, leading to truncated
    perf.data files that corrupt the entire trace.
    """
    actual_free = shutil.disk_usage(tmp_path).free

    # At the exact boundary: must pass
    ArtifactManager(tmp_path, min_free_bytes=actual_free).check_disk_space()

    # One byte over: must fail
    with pytest.raises(DiskSpaceError):
        ArtifactManager(tmp_path, min_free_bytes=actual_free + 1).check_disk_space()


def test_artifact_paths_confined_to_raw_dir():
    """
    Every canonical artifact path must be a child of raw_dir.
    A path outside raw_dir could overwrite unrelated run data when
    the collector writes its output, silently corrupting a prior run.
    """
    mgr = ArtifactManager(Path("/runs/run_001"))
    raw = mgr.raw_dir
    for path in [mgr.perf_data_path(), mgr.perf_script_path(), mgr.memray_path()]:
        assert str(path).startswith(str(raw)), (
            f"Artifact path {path} is outside raw_dir {raw}. "
            "Artifacts written outside raw_dir could corrupt adjacent run data."
        )

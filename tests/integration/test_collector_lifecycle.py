"""
Integration tests for the collector subsystem.

Tests real lifecycle behaviour using the current process as target.
No root or PMU required — focuses on ProcessMonitor and ArtifactManager,
with mocked perf/memray for the primary collector paths.
"""

import os
import time
from unittest.mock import MagicMock, patch

import pytest

from mrm_trace.collector import (
    ArtifactManager,
    CollectorMeta,
    MemrayRunner,
    PerfMemRunner,
    ProcessMonitor,
    select_collector,
)


# ── ProcessMonitor ─────────────────────────────────────────────────────────────

class TestProcessMonitorLifecycle:

    def test_collects_real_samples_from_current_process(self):
        monitor = ProcessMonitor([os.getpid()], poll_interval_ms=30)
        monitor.start()
        time.sleep(0.25)
        samples = monitor.stop()
        assert len(samples) >= 1
        assert all(s.pid == os.getpid() for s in samples)

    def test_samples_are_temporally_monotonic(self):
        monitor = ProcessMonitor([os.getpid()], poll_interval_ms=30)
        monitor.start()
        time.sleep(0.25)
        samples = monitor.stop()
        t_vals = [s.t_s for s in samples]
        for i in range(1, len(t_vals)):
            assert t_vals[i] >= t_vals[i - 1], (
                f"t_s went backwards at index {i}: {t_vals[i-1]} → {t_vals[i]}"
            )

    def test_rss_is_positive_for_live_process(self):
        monitor = ProcessMonitor([os.getpid()], poll_interval_ms=30)
        monitor.start()
        time.sleep(0.15)
        samples = monitor.stop()
        assert all(s.rss_mb > 0 for s in samples)

    def test_can_start_stop_multiple_times(self):
        monitor = ProcessMonitor([os.getpid()], poll_interval_ms=50)
        monitor.start()
        time.sleep(0.1)
        samples1 = monitor.stop()
        # Reinitialise for a second run
        monitor2 = ProcessMonitor([os.getpid()], poll_interval_ms=50)
        monitor2.start()
        time.sleep(0.1)
        samples2 = monitor2.stop()
        assert len(samples1) > 0
        assert len(samples2) > 0


# ── ArtifactManager ───────────────────────────────────────────────────────────

class TestArtifactManagerLifecycle:

    def test_full_setup_workflow(self, tmp_path):
        run_dir = tmp_path / "run_001"
        mgr = ArtifactManager(run_dir)
        mgr.check_disk_space()
        mgr.ensure_directories()
        assert mgr.raw_dir.exists()

    def test_get_file_size_after_write(self, tmp_path):
        mgr = ArtifactManager(tmp_path)
        mgr.ensure_directories()
        path = mgr.perf_data_path()
        path.write_bytes(b"synthetic" * 100)
        assert mgr.get_file_size(path) == 900

    def test_raw_artifacts_survive_independently(self, tmp_path):
        mgr = ArtifactManager(tmp_path)
        mgr.ensure_directories()
        for path in [mgr.perf_data_path(), mgr.memray_path()]:
            path.write_bytes(b"data")
        # Both files exist and are independent
        assert mgr.perf_data_path().exists()
        assert mgr.memray_path().exists()


# ── select_collector ──────────────────────────────────────────────────────────

class TestSelectCollector:

    def test_returns_perf_when_available(self):
        with patch("mrm_trace.collector.perf_runner.shutil.which",
                   return_value="/usr/bin/perf"):
            collector = select_collector()
        assert isinstance(collector, PerfMemRunner)

    def test_falls_back_to_memray_when_perf_unavailable(self):
        mock_memray = MagicMock()
        with patch("mrm_trace.collector.perf_runner.shutil.which", return_value=None):
            with patch("mrm_trace.collector.memray_runner._try_import_memray",
                       return_value=mock_memray):
                collector = select_collector()
        assert isinstance(collector, MemrayRunner)

    def test_prefer_memray_skips_perf(self):
        mock_memray = MagicMock()
        with patch("mrm_trace.collector.memray_runner._try_import_memray",
                   return_value=mock_memray):
            collector = select_collector(prefer="memray")
        assert isinstance(collector, MemrayRunner)

    def test_raises_when_neither_available(self):
        with patch("mrm_trace.collector.perf_runner.shutil.which", return_value=None):
            with patch("mrm_trace.collector.memray_runner._try_import_memray",
                       return_value=None):
                with pytest.raises(RuntimeError):
                    select_collector()


# ── MemrayRunner with mocked tracker ─────────────────────────────────────────

class TestMemrayRunnerLifecycle:

    def test_full_mock_lifecycle(self, tmp_path):
        mock_mod = MagicMock()
        mock_tracker = MagicMock()
        mock_mod.Tracker.return_value = mock_tracker
        runner = MemrayRunner()
        with patch("mrm_trace.collector.memray_runner._try_import_memray",
                   return_value=mock_mod):
            runner.start([os.getpid()], tmp_path)
        (tmp_path / "memray.bin").write_bytes(b"fake data")
        meta = runner.stop()
        assert isinstance(meta, CollectorMeta)
        assert meta.used is True
        assert meta.parse_success is True
        mock_tracker.__exit__.assert_called_once()


# ── PerfMemRunner with mocked subprocess ─────────────────────────────────────

class TestPerfRunnerLifecycle:

    def test_full_mock_lifecycle(self, tmp_path):
        runner = PerfMemRunner()
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 0
        (tmp_path / "perf.data").write_bytes(b"fake perf" * 200)
        with patch("mrm_trace.collector.perf_runner.shutil.which",
                   return_value="/usr/bin/perf"):
            with patch("subprocess.Popen", return_value=mock_proc):
                runner.start([os.getpid()], tmp_path)
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stderr=b"")
                meta = runner.stop()
        assert meta.output_file is not None
        assert meta.output_file_size_bytes > 0
        assert meta.parse_success is True

    def test_stop_without_start_safe(self):
        meta = PerfMemRunner().stop()
        assert meta.collector_type == "perf_mem"

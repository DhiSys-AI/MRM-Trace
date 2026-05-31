"""Unit tests for PerfMemRunner."""

import signal
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mrm_trace.collector.perf_runner import PerfMemRunner


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mock_proc(returncode=0):
    proc = MagicMock()
    proc.communicate.return_value = (b"", b"")
    proc.returncode = returncode
    return proc


# ── Availability ───────────────────────────────────────────────────────────────

def test_name_is_perf_mem():
    assert PerfMemRunner().name == "perf_mem"


def test_is_available_false_when_perf_not_found():
    with patch("mrm_trace.collector.perf_runner.shutil.which", return_value=None):
        assert PerfMemRunner().is_available() is False


def test_is_available_true_when_perf_found():
    with patch("mrm_trace.collector.perf_runner.shutil.which", return_value="/usr/bin/perf"):
        assert PerfMemRunner().is_available() is True


# ── start() ───────────────────────────────────────────────────────────────────

def test_start_raises_when_perf_not_found(tmp_path):
    with patch("mrm_trace.collector.perf_runner.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="perf"):
            PerfMemRunner().start([1234], tmp_path)


def test_start_never_uses_shell(tmp_path):
    with patch("mrm_trace.collector.perf_runner.shutil.which", return_value="/usr/bin/perf"):
        with patch("subprocess.Popen", return_value=_mock_proc()) as mock_popen:
            PerfMemRunner().start([1], tmp_path)
    _, kwargs = mock_popen.call_args
    assert kwargs.get("shell") is not True


def test_start_passes_pid_to_p_flag(tmp_path):
    with patch("mrm_trace.collector.perf_runner.shutil.which", return_value="/usr/bin/perf"):
        with patch("subprocess.Popen", return_value=_mock_proc()) as mock_popen:
            PerfMemRunner().start([9876], tmp_path)
    cmd = mock_popen.call_args[0][0]
    p_idx = cmd.index("-p")
    assert "9876" in cmd[p_idx + 1]


def test_start_joins_multiple_pids(tmp_path):
    with patch("mrm_trace.collector.perf_runner.shutil.which", return_value="/usr/bin/perf"):
        with patch("subprocess.Popen", return_value=_mock_proc()) as mock_popen:
            PerfMemRunner().start([111, 222, 333], tmp_path)
    cmd = mock_popen.call_args[0][0]
    p_idx = cmd.index("-p")
    pid_arg = cmd[p_idx + 1]
    assert "111" in pid_arg and "222" in pid_arg and "333" in pid_arg


def test_start_includes_output_flag_with_perf_data(tmp_path):
    with patch("mrm_trace.collector.perf_runner.shutil.which", return_value="/usr/bin/perf"):
        with patch("subprocess.Popen", return_value=_mock_proc()) as mock_popen:
            PerfMemRunner().start([1], tmp_path)
    cmd = mock_popen.call_args[0][0]
    assert "-o" in cmd
    assert "perf.data" in cmd[cmd.index("-o") + 1]


def test_start_no_event_flags_by_default(tmp_path):
    # Default events list is empty — let perf auto-select CPU-appropriate events.
    # Explicit -e flags are omitted; perf mem record picks the right events itself.
    with patch("mrm_trace.collector.perf_runner.shutil.which", return_value="/usr/bin/perf"):
        with patch("subprocess.Popen", return_value=_mock_proc()) as mock_popen:
            PerfMemRunner().start([1], tmp_path)
    cmd = mock_popen.call_args[0][0]
    assert "-e" not in cmd


def test_start_uses_custom_events(tmp_path):
    runner = PerfMemRunner(events=["cpu/mem-loads/P"])
    with patch("mrm_trace.collector.perf_runner.shutil.which", return_value="/usr/bin/perf"):
        with patch("subprocess.Popen", return_value=_mock_proc()) as mock_popen:
            runner.start([1], tmp_path)
    cmd = mock_popen.call_args[0][0]
    assert "cpu/mem-loads/P" in cmd


# ── stop() ────────────────────────────────────────────────────────────────────

def test_stop_sends_sigint(tmp_path):
    runner = PerfMemRunner()
    proc = _mock_proc()
    with patch("mrm_trace.collector.perf_runner.shutil.which", return_value="/usr/bin/perf"):
        with patch("subprocess.Popen", return_value=proc):
            runner.start([1], tmp_path)
    runner.stop()
    proc.send_signal.assert_called_once_with(signal.SIGINT)


def test_stop_returns_collector_meta(tmp_path):
    runner = PerfMemRunner()
    with patch("mrm_trace.collector.perf_runner.shutil.which", return_value="/usr/bin/perf"):
        with patch("subprocess.Popen", return_value=_mock_proc()):
            runner.start([1], tmp_path)
    meta = runner.stop()
    assert meta.collector_type == "perf_mem"
    assert meta.available is True
    assert meta.used is True


def test_stop_without_start_is_safe():
    meta = PerfMemRunner().stop()
    assert meta.collector_type == "perf_mem"
    assert meta.used is True


def test_stop_records_file_size_when_perf_data_exists(tmp_path):
    runner = PerfMemRunner()
    proc = _mock_proc()
    (tmp_path / "perf.data").write_bytes(b"x" * 512)
    with patch("mrm_trace.collector.perf_runner.shutil.which", return_value="/usr/bin/perf"):
        with patch("subprocess.Popen", return_value=proc):
            runner.start([1], tmp_path)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr=b"")
            meta = runner.stop()
    assert meta.output_file_size_bytes == 512
    assert meta.output_file is not None


def test_stop_warns_when_perf_data_missing(tmp_path):
    runner = PerfMemRunner()
    with patch("mrm_trace.collector.perf_runner.shutil.which", return_value="/usr/bin/perf"):
        with patch("subprocess.Popen", return_value=_mock_proc()):
            runner.start([1], tmp_path)
    meta = runner.stop()
    assert any("perf.data" in w for w in meta.warnings)


def test_stop_records_event_strings():
    custom = ["cpu/mem-loads,ldlat=30/P"]
    meta = PerfMemRunner(events=custom).stop()
    assert meta.event_strings == custom


def test_stop_kills_on_timeout(tmp_path):
    runner = PerfMemRunner()
    proc = MagicMock()
    proc.communicate.side_effect = subprocess.TimeoutExpired("perf", 30)
    with patch("mrm_trace.collector.perf_runner.shutil.which", return_value="/usr/bin/perf"):
        with patch("subprocess.Popen", return_value=proc):
            runner.start([1], tmp_path)
    meta = runner.stop()
    proc.kill.assert_called_once()
    assert any("timeout" in w.lower() for w in meta.warnings)

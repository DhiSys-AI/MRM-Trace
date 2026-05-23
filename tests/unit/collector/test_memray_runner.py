"""Unit tests for MemrayRunner."""

from unittest.mock import MagicMock, patch

import pytest

from mrm_trace.collector.memray_runner import MemrayRunner


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mock_memray():
    mod = MagicMock()
    mod.Tracker.return_value = MagicMock()
    return mod


# ── Availability ───────────────────────────────────────────────────────────────

def test_name_is_memray():
    assert MemrayRunner().name == "memray"


def test_is_available_false_when_not_installed():
    with patch("mrm_trace.collector.memray_runner._try_import_memray", return_value=None):
        assert MemrayRunner().is_available() is False


def test_is_available_true_when_installed():
    with patch("mrm_trace.collector.memray_runner._try_import_memray", return_value=_mock_memray()):
        assert MemrayRunner().is_available() is True


# ── start() ───────────────────────────────────────────────────────────────────

def test_start_raises_when_not_installed(tmp_path):
    with patch("mrm_trace.collector.memray_runner._try_import_memray", return_value=None):
        with pytest.raises(RuntimeError, match="memray"):
            MemrayRunner().start([1234], tmp_path)


def test_start_creates_tracker(tmp_path):
    mock_mod = _mock_memray()
    mock_tracker = mock_mod.Tracker.return_value
    with patch("mrm_trace.collector.memray_runner._try_import_memray", return_value=mock_mod):
        MemrayRunner().start([1], tmp_path)
    mock_mod.Tracker.assert_called_once()
    mock_tracker.__enter__.assert_called_once()


def test_start_output_file_under_output_dir(tmp_path):
    mock_mod = _mock_memray()
    with patch("mrm_trace.collector.memray_runner._try_import_memray", return_value=mock_mod):
        MemrayRunner().start([1], tmp_path)
    tracker_path = mock_mod.Tracker.call_args[0][0]
    assert str(tmp_path) in tracker_path
    assert "memray.bin" in tracker_path


def test_start_passes_native_traces_true(tmp_path):
    mock_mod = _mock_memray()
    with patch("mrm_trace.collector.memray_runner._try_import_memray", return_value=mock_mod):
        MemrayRunner(native_traces=True).start([1], tmp_path)
    _, kwargs = mock_mod.Tracker.call_args
    assert kwargs.get("native_traces") is True


def test_start_passes_native_traces_false(tmp_path):
    mock_mod = _mock_memray()
    with patch("mrm_trace.collector.memray_runner._try_import_memray", return_value=mock_mod):
        MemrayRunner(native_traces=False).start([1], tmp_path)
    _, kwargs = mock_mod.Tracker.call_args
    assert kwargs.get("native_traces") is False


# ── stop() ────────────────────────────────────────────────────────────────────

def test_stop_exits_tracker_context(tmp_path):
    mock_mod = _mock_memray()
    mock_tracker = mock_mod.Tracker.return_value
    runner = MemrayRunner()
    with patch("mrm_trace.collector.memray_runner._try_import_memray", return_value=mock_mod):
        runner.start([1], tmp_path)
    runner.stop()
    mock_tracker.__exit__.assert_called_once_with(None, None, None)


def test_stop_returns_collector_meta(tmp_path):
    mock_mod = _mock_memray()
    runner = MemrayRunner()
    with patch("mrm_trace.collector.memray_runner._try_import_memray", return_value=mock_mod):
        runner.start([1], tmp_path)
    meta = runner.stop()
    assert meta.collector_type == "memray"
    assert meta.available is True
    assert meta.used is True


def test_stop_without_start_is_safe():
    meta = MemrayRunner().stop()
    assert meta.collector_type == "memray"
    assert meta.output_file is None
    assert meta.output_file_size_bytes == 0


def test_stop_parse_success_false_when_no_output_file(tmp_path):
    mock_mod = _mock_memray()
    runner = MemrayRunner()
    with patch("mrm_trace.collector.memray_runner._try_import_memray", return_value=mock_mod):
        runner.start([1], tmp_path)
    meta = runner.stop()
    assert meta.parse_success is False


def test_stop_parse_success_true_when_output_file_exists(tmp_path):
    mock_mod = _mock_memray()
    runner = MemrayRunner()
    with patch("mrm_trace.collector.memray_runner._try_import_memray", return_value=mock_mod):
        runner.start([1], tmp_path)
    (tmp_path / "memray.bin").write_bytes(b"fake memray data")
    meta = runner.stop()
    assert meta.parse_success is True
    assert meta.output_file_size_bytes > 0


def test_stop_extra_has_native_traces_flag(tmp_path):
    mock_mod = _mock_memray()
    runner = MemrayRunner(native_traces=True)
    with patch("mrm_trace.collector.memray_runner._try_import_memray", return_value=mock_mod):
        runner.start([1], tmp_path)
    meta = runner.stop()
    assert meta.extra.get("native_traces") is True


def test_second_stop_is_idempotent(tmp_path):
    mock_mod = _mock_memray()
    runner = MemrayRunner()
    with patch("mrm_trace.collector.memray_runner._try_import_memray", return_value=mock_mod):
        runner.start([1], tmp_path)
    runner.stop()
    meta = runner.stop()  # should not raise
    assert meta.collector_type == "memray"

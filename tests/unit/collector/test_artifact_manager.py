"""Unit tests for ArtifactManager."""

import shutil
from pathlib import Path

import pytest

from mrm_trace.collector.artifact_manager import ArtifactManager, DiskSpaceError


# ── Directory structure ────────────────────────────────────────────────────────

def test_run_dir_property(tmp_path):
    run_dir = tmp_path / "my_run"
    assert ArtifactManager(run_dir).run_dir == run_dir


def test_raw_dir_is_under_run_dir(tmp_path):
    run_dir = tmp_path / "run_01"
    assert ArtifactManager(run_dir).raw_dir == run_dir / "raw"


def test_ensure_directories_creates_raw_dir(tmp_path):
    run_dir = tmp_path / "run_01"
    mgr = ArtifactManager(run_dir)
    mgr.ensure_directories()
    assert (run_dir / "raw").is_dir()


def test_ensure_directories_creates_nested_path(tmp_path):
    run_dir = tmp_path / "results" / "model_01" / "run_001"
    mgr = ArtifactManager(run_dir)
    mgr.ensure_directories()
    assert mgr.raw_dir.is_dir()


def test_ensure_directories_idempotent(tmp_path):
    mgr = ArtifactManager(tmp_path / "run_01")
    mgr.ensure_directories()
    mgr.ensure_directories()  # should not raise


# ── Canonical paths ────────────────────────────────────────────────────────────

def test_perf_data_path(tmp_path):
    mgr = ArtifactManager(tmp_path)
    assert mgr.perf_data_path() == tmp_path / "raw" / "perf.data"


def test_perf_script_path(tmp_path):
    mgr = ArtifactManager(tmp_path)
    assert mgr.perf_script_path() == tmp_path / "raw" / "perf_script.txt"


def test_memray_path(tmp_path):
    mgr = ArtifactManager(tmp_path)
    assert mgr.memray_path() == tmp_path / "raw" / "memray.bin"


def test_all_paths_under_raw_dir(tmp_path):
    mgr = ArtifactManager(tmp_path)
    raw = mgr.raw_dir
    for p in [mgr.perf_data_path(), mgr.perf_script_path(), mgr.memray_path()]:
        assert str(p).startswith(str(raw))


# ── File size ─────────────────────────────────────────────────────────────────

def test_get_file_size_zero_for_missing(tmp_path):
    mgr = ArtifactManager(tmp_path)
    assert mgr.get_file_size(tmp_path / "nonexistent.dat") == 0


def test_get_file_size_correct_for_existing(tmp_path):
    f = tmp_path / "test.bin"
    f.write_bytes(b"x" * 1024)
    assert ArtifactManager(tmp_path).get_file_size(f) == 1024


def test_get_file_size_zero_byte_file(tmp_path):
    f = tmp_path / "empty.bin"
    f.touch()
    assert ArtifactManager(tmp_path).get_file_size(f) == 0


# ── Disk space ────────────────────────────────────────────────────────────────

def test_check_disk_space_passes_with_tiny_threshold(tmp_path):
    ArtifactManager(tmp_path, min_free_bytes=1).check_disk_space()


def test_check_disk_space_raises_with_impossibly_large_threshold(tmp_path):
    with pytest.raises(DiskSpaceError):
        ArtifactManager(tmp_path, min_free_bytes=999_999_999_999_999).check_disk_space()


def test_disk_space_error_message_contains_gib(tmp_path):
    with pytest.raises(DiskSpaceError, match="GiB"):
        ArtifactManager(tmp_path, min_free_bytes=999_999_999_999_999).check_disk_space()


def test_check_disk_space_walks_to_existing_ancestor(tmp_path):
    deep = tmp_path / "a" / "b" / "c"
    ArtifactManager(deep, min_free_bytes=1).check_disk_space()


def test_disk_space_error_is_runtime_error(tmp_path):
    with pytest.raises(RuntimeError):
        ArtifactManager(tmp_path, min_free_bytes=999_999_999_999_999).check_disk_space()

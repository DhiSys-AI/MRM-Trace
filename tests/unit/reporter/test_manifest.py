import json

import pytest

from mrm_trace.reporter.manifest import (
    MANIFEST_SCHEMA_VERSION,
    build_manifest,
    write_manifest,
)


def test_manifest_has_required_keys(tmp_path):
    f = tmp_path / "trace.parquet"
    f.write_bytes(b"fake parquet")
    manifest = build_manifest(tmp_path, "run_001", files=[f])
    assert "manifest_schema_version" in manifest
    assert "run_id" in manifest
    assert "created_at_utc" in manifest
    assert "files" in manifest


def test_manifest_schema_version_correct(tmp_path):
    manifest = build_manifest(tmp_path, "run_001", files=[])
    assert manifest["manifest_schema_version"] == MANIFEST_SCHEMA_VERSION


def test_manifest_run_id_set(tmp_path):
    manifest = build_manifest(tmp_path, "my_run_123", files=[])
    assert manifest["run_id"] == "my_run_123"


def test_manifest_file_entry_has_checksum(tmp_path):
    f = tmp_path / "metrics.csv"
    f.write_text("col1,col2\n1,2\n")
    manifest = build_manifest(tmp_path, "run", files=[f])
    entry = manifest["files"][0]
    assert "sha256" in entry
    assert len(entry["sha256"]) == 64


def test_manifest_file_entry_has_size(tmp_path):
    content = b"x" * 1000
    f = tmp_path / "data.bin"
    f.write_bytes(content)
    manifest = build_manifest(tmp_path, "run", files=[f])
    assert manifest["files"][0]["size_bytes"] == 1000


def test_manifest_file_entry_has_path(tmp_path):
    f = tmp_path / "trace.parquet"
    f.write_bytes(b"d")
    manifest = build_manifest(tmp_path, "run", files=[f])
    assert manifest["files"][0]["path"] == "trace.parquet"


def test_manifest_known_file_has_schema_version(tmp_path):
    for fname in ("trace.parquet", "region_map.parquet", "metrics.csv", "metadata.json"):
        f = tmp_path / fname
        f.write_bytes(b"d")
        manifest = build_manifest(tmp_path, "run", files=[f])
        assert "schema_version" in manifest["files"][0], f"Missing schema_version for {fname}"
        f.unlink()


def test_manifest_unknown_file_no_schema_version(tmp_path):
    f = tmp_path / "custom_output.bin"
    f.write_bytes(b"d")
    manifest = build_manifest(tmp_path, "run", files=[f])
    assert "schema_version" not in manifest["files"][0]


def test_manifest_empty_files(tmp_path):
    manifest = build_manifest(tmp_path, "empty_run", files=[])
    assert manifest["files"] == []


def test_manifest_auto_discovers_files(tmp_path):
    (tmp_path / "trace.parquet").write_bytes(b"d1")
    (tmp_path / "metrics.csv").write_text("a\n")
    (tmp_path / "manifest.json").write_text("{}")  # must be excluded
    manifest = build_manifest(tmp_path, "run_001")
    names = {e["path"] for e in manifest["files"]}
    assert "trace.parquet" in names
    assert "metrics.csv" in names
    assert "manifest.json" not in names


def test_write_manifest_creates_file(tmp_path):
    f = tmp_path / "metrics.csv"
    f.write_text("a,b\n")
    out = write_manifest(tmp_path, "run_xyz", files=[f])
    assert out.exists()
    assert out.name == "manifest.json"


def test_write_manifest_valid_json(tmp_path):
    f = tmp_path / "trace.parquet"
    f.write_bytes(b"data")
    write_manifest(tmp_path, "run_001", files=[f])
    manifest_path = tmp_path / "manifest.json"
    data = json.loads(manifest_path.read_text())
    assert data["run_id"] == "run_001"


def test_write_manifest_returns_path(tmp_path):
    out = write_manifest(tmp_path, "run_001", files=[])
    assert isinstance(out, type(tmp_path))


def test_manifest_checksum_is_reproducible(tmp_path):
    content = b"consistent content"
    f = tmp_path / "file.bin"
    f.write_bytes(content)
    m1 = build_manifest(tmp_path, "run", files=[f])
    m2 = build_manifest(tmp_path, "run", files=[f])
    assert m1["files"][0]["sha256"] == m2["files"][0]["sha256"]


def test_manifest_files_sorted_by_name(tmp_path):
    for name in ("z_file.txt", "a_file.txt", "m_file.txt"):
        (tmp_path / name).write_bytes(b"x")
    manifest = build_manifest(tmp_path, "run")
    names = [e["path"] for e in manifest["files"]]
    assert names == sorted(names)

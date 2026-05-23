"""Unit tests for schema_version module."""

import pytest

from mrm_trace.schema_version import (
    CURRENT_VERSIONS,
    REQUIRED_COLUMNS,
    SchemaVersionError,
    assert_compatible,
    check_parquet_schema,
    is_compatible,
    validate_columns,
)


# ── is_compatible ─────────────────────────────────────────────────────────────

def test_same_version_is_compatible():
    assert is_compatible("trace", "1.0") is True


def test_minor_bump_is_compatible():
    assert is_compatible("trace", "1.5") is True


def test_major_bump_is_incompatible():
    assert is_compatible("trace", "2.0") is False


def test_older_major_is_incompatible():
    assert is_compatible("trace", "0.9") is False


def test_unknown_schema_is_compatible():
    assert is_compatible("nonexistent_schema", "99.0") is True


# ── assert_compatible ─────────────────────────────────────────────────────────

def test_assert_compatible_passes_for_same_major():
    assert_compatible("trace", "1.0")   # must not raise
    assert_compatible("trace", "1.99")  # minor bump OK


def test_assert_compatible_raises_for_major_mismatch():
    with pytest.raises(SchemaVersionError, match="version mismatch"):
        assert_compatible("trace", "2.0")


def test_schema_version_error_message_contains_schema_name():
    with pytest.raises(SchemaVersionError, match="trace"):
        assert_compatible("trace", "2.0")


# ── validate_columns ──────────────────────────────────────────────────────────

def test_all_required_columns_present_returns_empty_list():
    cols = list(REQUIRED_COLUMNS["trace"])
    assert validate_columns("trace", cols) == []


def test_missing_column_reported():
    cols = list(REQUIRED_COLUMNS["trace"] - {"region_type"})
    missing = validate_columns("trace", cols)
    assert "region_type" in missing


def test_extra_columns_are_allowed():
    cols = list(REQUIRED_COLUMNS["trace"]) + ["extra_col", "another_col"]
    assert validate_columns("trace", cols) == []


def test_unknown_schema_returns_no_missing():
    assert validate_columns("nonexistent", ["any_col"]) == []


def test_missing_columns_sorted():
    missing = validate_columns("trace", [])
    assert missing == sorted(missing)


# ── CURRENT_VERSIONS registry ─────────────────────────────────────────────────

def test_all_known_schemas_have_version():
    for name in ("trace", "region_map", "kv_block_lifecycle", "metrics"):
        assert name in CURRENT_VERSIONS, f"'{name}' missing from CURRENT_VERSIONS"


def test_versions_are_valid_format():
    for name, ver in CURRENT_VERSIONS.items():
        parts = ver.split(".")
        assert len(parts) == 2, f"Invalid version {ver!r} for '{name}'"
        assert all(p.isdigit() for p in parts), f"Non-numeric version {ver!r}"


# ── check_parquet_schema ──────────────────────────────────────────────────────

def test_check_parquet_schema_passes_for_valid_trace(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([(c, pa.string()) for c in REQUIRED_COLUMNS["trace"]])
    table = pa.table({c: ["val"] for c in REQUIRED_COLUMNS["trace"]}, schema=schema)
    path = tmp_path / "trace.parquet"
    pq.write_table(table, str(path))
    check_parquet_schema(path, "trace")  # must not raise


def test_check_parquet_schema_raises_on_missing_column(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = list(REQUIRED_COLUMNS["trace"] - {"region_type"})
    schema = pa.schema([(c, pa.string()) for c in cols])
    table = pa.table({c: ["val"] for c in cols}, schema=schema)
    path = tmp_path / "trace_bad.parquet"
    pq.write_table(table, str(path))
    with pytest.raises(ValueError, match="region_type"):
        check_parquet_schema(path, "trace")


def test_check_parquet_schema_raises_on_incompatible_version(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = list(REQUIRED_COLUMNS["trace"])
    schema = pa.schema(
        [(c, pa.string()) for c in cols],
        metadata={b"mrm_trace_schema_version": b"2.0"},
    )
    table = pa.table({c: ["val"] for c in cols}, schema=schema)
    path = tmp_path / "trace_v2.parquet"
    pq.write_table(table, str(path))
    with pytest.raises(SchemaVersionError):
        check_parquet_schema(path, "trace")

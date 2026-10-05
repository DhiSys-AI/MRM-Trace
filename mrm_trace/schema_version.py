"""
Schema version registry and compatibility utilities.

Every canonical output file has a declared schema version (major.minor).
  - Same major  → backward-compatible; readers should handle it.
  - Major bump  → breaking change; readers must reject.
  - Minor bump  → additive change (new optional columns); readers may ignore.

Versions are also written into Parquet file-level metadata under the key
"mrm_trace_schema_version" so files are self-describing.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List

# ── Current schema versions ───────────────────────────────────────────────────

CURRENT_VERSIONS: Dict[str, str] = {
    "trace":              "1.0",
    "region_map":         "1.0",
    "kv_block_lifecycle": "1.1",
    "metrics":            "1.2",
    "metadata":           "1.0",
    "manifest":           "1.0",
}

# ── Required columns per schema ───────────────────────────────────────────────

REQUIRED_COLUMNS: Dict[str, FrozenSet[str]] = {
    "trace": frozenset({
        "run_id", "sample_id", "timestamp_ns", "phase",
        "address", "address_page", "op_type",
        "symbol", "dso", "region_type", "region_id",
    }),
    "region_map": frozenset({
        "region_id", "region_type",
        "page_min", "page_max",
        "first_seen_ns", "last_seen_ns", "access_count",
    }),
    "kv_block_lifecycle": frozenset({
        "block_id", "address", "address_page",
        "write_ns", "last_read_ns", "evict_ns",
        "total_reads", "write_observed", "lifetime_ns",
    }),
    "metrics": frozenset({
        "region_type", "n_regions", "n_regions_observable",
        "retention_support_frac",
        "retention_p99_s", "write_once_ratio_mean", "suitability",
    }),
}


class SchemaVersionError(Exception):
    """Raised when a file's schema version is incompatible with this reader."""


def _parse(version: str):
    """Return (major, minor) from 'X.Y' string."""
    parts = version.split(".")
    if len(parts) != 2:
        raise ValueError(f"Invalid version string {version!r} — expected 'X.Y'")
    return int(parts[0]), int(parts[1])


def is_compatible(schema_name: str, file_version: str) -> bool:
    """
    Return True if *file_version* is readable by the current reader.

    Compatible when file_major == current_major (minor bumps are additive).
    """
    if schema_name not in CURRENT_VERSIONS:
        return True  # unknown schema — let caller decide
    current_major, _ = _parse(CURRENT_VERSIONS[schema_name])
    file_major, _ = _parse(file_version)
    return file_major == current_major


def assert_compatible(schema_name: str, file_version: str) -> None:
    """Raise SchemaVersionError if the file is not compatible with this reader."""
    if not is_compatible(schema_name, file_version):
        current = CURRENT_VERSIONS.get(schema_name, "unknown")
        raise SchemaVersionError(
            f"Schema '{schema_name}' version mismatch: "
            f"file={file_version!r}, reader supports major={current.split('.')[0]}. "
            "Upgrade mrm-trace to read this file."
        )


def validate_columns(
    schema_name: str,
    present_columns: List[str],
) -> List[str]:
    """
    Return a list of required columns that are missing from *present_columns*.

    An empty list means the schema is satisfied.
    """
    required = REQUIRED_COLUMNS.get(schema_name, frozenset())
    return sorted(required - set(present_columns))


def check_parquet_schema(path, schema_name: str) -> None:
    """
    Validate that a Parquet file has the required columns for *schema_name*
    and a compatible schema version (if metadata is present).

    Raises
    ------
    SchemaVersionError : version incompatibility
    ValueError         : missing required columns
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(path))
    meta = pf.schema_arrow.metadata or {}
    version_key = b"mrm_trace_schema_version"
    if version_key in meta:
        file_version = meta[version_key].decode()
        assert_compatible(schema_name, file_version)

    present = pf.schema_arrow.names
    missing = validate_columns(schema_name, present)
    if missing:
        raise ValueError(
            f"Parquet file {path} is missing required columns for schema "
            f"'{schema_name}': {missing}"
        )

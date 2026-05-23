"""
Normalises parsed trace rows into the canonical schema.

- Fills default values for optional columns
- Validates that required columns are present
- Derives address_page when absent or zero
- Does NOT accumulate rows — accepts and returns generators
"""

from typing import Any, Dict, Generator, Iterable

REQUIRED_COLUMNS = frozenset({
    "run_id", "sample_id", "timestamp_ns", "address", "address_page", "op_type",
})

# Canonical column order from CLAUDE.md trace schema
ALL_COLUMNS = [
    "run_id", "sample_id", "timestamp_ns", "phase", "address", "address_page",
    "op_type", "symbol", "dso", "region_type", "region_id", "access_latency_ns",
    "cache_level", "numa_node", "cpu_id", "thread_id", "token_index", "context_tokens",
]

_DEFAULTS: Dict[str, Any] = {
    "phase": None,
    "symbol": "",
    "dso": "",
    "region_type": "unknown",
    "region_id": 0,
    "access_latency_ns": None,
    "cache_level": None,
    "numa_node": None,
    "cpu_id": None,
    "thread_id": None,
    "token_index": None,
    "context_tokens": None,
}


def normalise_rows(
    rows: Iterable[Dict[str, Any]],
) -> Generator[Dict[str, Any], None, None]:
    """
    Apply schema defaults and ensure address_page is set, yielding normalised rows.

    Raises:
        ValueError: if a row is missing a required column
    """
    for row in rows:
        missing = REQUIRED_COLUMNS - row.keys()
        if missing:
            raise ValueError(
                f"Row missing required columns: {missing}. Row keys: {set(row.keys())}"
            )

        normalised = {**_DEFAULTS, **row}

        # Derive address_page if not set or zero
        if not normalised.get("address_page") and normalised.get("address"):
            normalised["address_page"] = normalised["address"] >> 12

        yield {col: normalised[col] for col in ALL_COLUMNS}

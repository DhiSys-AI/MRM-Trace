"""
Streaming Parquet writer for canonical trace rows.

Uses PyArrow batched writing to keep memory bounded (ADR-2).
ADR-8: zstd compression.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import pyarrow as pa
import pyarrow.parquet as pq

# Canonical trace schema (ADR-8 / spec §4).
# Explicit types are required: auto-inferred int64 overflows on kernel-space
# addresses (> 2^63) which are valid uint64 values on x86-64.
TRACE_SCHEMA = pa.schema([
    pa.field("run_id",            pa.string()),
    pa.field("sample_id",         pa.uint64()),
    pa.field("timestamp_ns",      pa.uint64()),
    pa.field("phase",             pa.string()),
    pa.field("address",           pa.uint64()),
    pa.field("address_page",      pa.uint64()),
    pa.field("op_type",           pa.string()),
    pa.field("symbol",            pa.string()),
    pa.field("dso",               pa.string()),
    pa.field("region_type",       pa.string()),
    pa.field("region_id",         pa.uint32()),
    pa.field("access_latency_ns", pa.uint32()),
    pa.field("cache_level",       pa.string()),
    pa.field("numa_node",         pa.uint8()),
    pa.field("cpu_id",            pa.uint16()),
    pa.field("thread_id",         pa.uint32()),
    pa.field("token_index",       pa.uint32()),
    pa.field("context_tokens",    pa.uint32()),
])


def write_parquet(
    rows: Iterable[Dict[str, Any]],
    path: Union[str, Path],
    batch_size: int = 100_000,
    schema: Optional[pa.Schema] = None,
) -> int:
    """
    Stream rows to a Parquet file using zstd compression.

    Writes in batches of ``batch_size`` so memory is bounded regardless of
    trace size (ADR-2).  Pass ``schema=TRACE_SCHEMA`` when writing trace rows
    to enforce uint64 for address/timestamp columns.

    Returns total rows written (0 if input is empty, no file created).
    """
    path = Path(path)
    total = 0
    batch: list = []
    writer_ref: list = []  # holds at most one ParquetWriter; avoids nonlocal

    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            _write_batch(batch, path, schema, writer_ref)
            total += len(batch)
            batch.clear()

    if batch:
        _write_batch(batch, path, schema, writer_ref)
        total += len(batch)

    if writer_ref:
        writer_ref[0].close()

    return total


def _write_batch(
    batch: list,
    path: Path,
    schema: Optional[pa.Schema],
    writer_ref: list,
) -> None:
    table = pa.Table.from_pylist(batch, schema=schema)
    if not writer_ref:
        writer_ref.append(pq.ParquetWriter(str(path), table.schema, compression="zstd"))
    writer_ref[0].write_table(table)

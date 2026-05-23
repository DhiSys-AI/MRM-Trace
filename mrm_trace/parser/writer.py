"""
Streaming Parquet writer for canonical trace rows.

Uses PyArrow batched writing to keep memory bounded (ADR-2).
ADR-8: zstd compression.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import pyarrow as pa
import pyarrow.parquet as pq


def write_parquet(
    rows: Iterable[Dict[str, Any]],
    path: Union[str, Path],
    batch_size: int = 100_000,
) -> int:
    """
    Stream rows to a Parquet file using zstd compression.

    Writes in batches of batch_size so memory is bounded regardless of trace size.

    Returns:
        Total number of rows written. Returns 0 and does not create the file
        if the input is empty.
    """
    path = Path(path)
    writer: Optional[pq.ParquetWriter] = None
    total = 0
    batch = []

    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            table = pa.Table.from_pylist(batch)
            if writer is None:
                writer = pq.ParquetWriter(str(path), table.schema, compression="zstd")
            writer.write_table(table)
            total += len(batch)
            batch.clear()

    if batch:
        table = pa.Table.from_pylist(batch)
        if writer is None:
            writer = pq.ParquetWriter(str(path), table.schema, compression="zstd")
        writer.write_table(table)
        total += len(batch)

    if writer is not None:
        writer.close()

    return total

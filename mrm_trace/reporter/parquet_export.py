"""
Parquet export helpers for the three canonical output tables.

All files use zstd compression (ADR-8).  Streaming write is used so memory
stays bounded regardless of trace size (ADR-2).
"""

import math
from pathlib import Path
from typing import List, Union

import pandas as pd

from mrm_trace.parser.writer import TRACE_SCHEMA, write_parquet

_TRACE_FILE = "trace.parquet"
_REGION_MAP_FILE = "region_map.parquet"
_KV_LIFECYCLE_FILE = "kv_block_lifecycle.parquet"


def export_trace(
    trace: pd.DataFrame,
    run_dir: Union[str, Path],
) -> Path:
    """Write trace.parquet into run_dir.  Returns the file path."""
    run_dir = Path(run_dir)
    out = run_dir / _TRACE_FILE
    # pandas stores missing values as float(nan) even in object/string columns.
    # pyarrow string arrays reject floats — replace nan with None in each record.
    def _clean(records):
        for row in records:
            yield {k: (None if isinstance(v, float) and math.isnan(v) else v)
                   for k, v in row.items()}

    write_parquet(_clean(trace.to_dict("records")), out, schema=TRACE_SCHEMA)
    return out


def export_region_map(
    region_map: Union[pd.DataFrame, List[dict]],
    run_dir: Union[str, Path],
) -> Path:
    """Write region_map.parquet into run_dir.  Returns the file path."""
    run_dir = Path(run_dir)
    out = run_dir / _REGION_MAP_FILE
    rows = region_map.to_dict("records") if isinstance(region_map, pd.DataFrame) else region_map
    write_parquet(iter(rows), out)
    return out


def export_kv_lifecycle(
    lifecycle: Union[pd.DataFrame, List[dict]],
    run_dir: Union[str, Path],
) -> Path:
    """Write kv_block_lifecycle.parquet into run_dir.  Returns the file path."""
    run_dir = Path(run_dir)
    out = run_dir / _KV_LIFECYCLE_FILE
    rows = lifecycle.to_dict("records") if isinstance(lifecycle, pd.DataFrame) else lifecycle
    write_parquet(iter(rows), out)
    return out

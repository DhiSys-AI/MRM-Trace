"""
write_parquet_frame must stream a DataFrame without materialising row dicts.

export_trace() used to call frame.to_dict("records"), which allocates every
row up front — 26 GB on a 30M-row trace, measured — so the export was
OOM-killed after parsing, labelling and analysis had all succeeded.  The
batched writer underneath was never the problem; the conversion above it was.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mrm_trace.parser.writer import TRACE_SCHEMA, write_parquet, write_parquet_frame

pytestmark = pytest.mark.unit


def _trace_frame(n: int = 250) -> pd.DataFrame:
    return pd.DataFrame({
        "run_id": ["r1"] * n,
        "sample_id": list(range(n)),
        "timestamp_ns": [1_000 + i for i in range(n)],
        "phase": ["generation"] * n,
        "address": [0x7B6C_0000_0000 + i * 64 for i in range(n)],
        "address_page": [(0x7B6C_0000_0000 + i * 64) >> 12 for i in range(n)],
        "op_type": ["load" if i % 2 else "store" for i in range(n)],
        "symbol": ["ggml_vec_dot_q6_K_q8_K"] * n,
        "dso": ["/lib/libggml-cpu.so.0"] * n,
        "region_type": ["weight"] * n,
        "region_id": [1] * n,
        "access_latency_ns": [97] * n,
        "cache_level": ["L1"] * n,
        "numa_node": [0] * n,
        "cpu_id": [3] * n,
        "thread_id": [9608] * n,
        "token_index": [None] * n,
        "context_tokens": [None] * n,
    })


def test_writes_every_row(tmp_path):
    df = _trace_frame(250)
    out = tmp_path / "trace.parquet"
    assert write_parquet_frame(df, out, batch_rows=37, schema=TRACE_SCHEMA) == 250
    assert pq.read_table(out).num_rows == 250


def test_schema_is_enforced(tmp_path):
    """uint64 addresses must survive; auto-inference would overflow int64."""
    out = tmp_path / "trace.parquet"
    write_parquet_frame(_trace_frame(10), out, schema=TRACE_SCHEMA)
    assert pq.read_table(out).schema.equals(TRACE_SCHEMA)


def test_uses_zstd(tmp_path):
    out = tmp_path / "trace.parquet"
    write_parquet_frame(_trace_frame(10), out, schema=TRACE_SCHEMA)
    meta = pq.ParquetFile(out).metadata.row_group(0).column(0)
    assert meta.compression.lower() == "zstd"


def test_batch_size_does_not_change_contents(tmp_path):
    df = _trace_frame(300)
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    write_parquet_frame(df, a, batch_rows=7, schema=TRACE_SCHEMA)
    write_parquet_frame(df, b, batch_rows=100_000, schema=TRACE_SCHEMA)
    assert pq.read_table(a).equals(pq.read_table(b))


def test_matches_the_row_dict_writer(tmp_path):
    """The frame path must produce the same table as the row-dict path."""
    df = _trace_frame(120)
    frame_out, dict_out = tmp_path / "frame.parquet", tmp_path / "dict.parquet"
    write_parquet_frame(df, frame_out, batch_rows=13, schema=TRACE_SCHEMA)
    write_parquet(iter(df.to_dict("records")), dict_out, schema=TRACE_SCHEMA)
    assert pq.read_table(frame_out).equals(pq.read_table(dict_out))


def test_empty_frame_writes_nothing(tmp_path):
    """Matches write_parquet: 0 rows, no file, so manifests skip it."""
    out = tmp_path / "trace.parquet"
    assert write_parquet_frame(_trace_frame(0), out, schema=TRACE_SCHEMA) == 0
    assert not out.exists()


def test_nan_in_a_string_column_becomes_null(tmp_path):
    """
    pandas represents a missing string as float nan, which pyarrow string
    arrays reject.  from_pandas coerces it; a plain from_pylist would raise.
    """
    df = _trace_frame(2)
    df.loc[1, "phase"] = np.nan
    df.loc[0, "symbol"] = np.nan
    out = tmp_path / "trace.parquet"
    write_parquet_frame(df, out, schema=TRACE_SCHEMA)
    table = pq.read_table(out)
    assert table.column("phase").to_pylist() == ["generation", None]
    assert table.column("symbol").to_pylist() == [None, "ggml_vec_dot_q6_K_q8_K"]


def test_categorical_columns_are_written_as_strings(tmp_path):
    """
    The labelled trace dictionary-encodes its repeated string columns; the
    Parquet schema still declares them pa.string().
    """
    df = _trace_frame(50)
    for col in ("run_id", "symbol", "dso", "cache_level"):
        df[col] = df[col].astype("category")
    out = tmp_path / "trace.parquet"
    write_parquet_frame(df, out, batch_rows=11, schema=TRACE_SCHEMA)
    table = pq.read_table(out)
    assert table.schema.field("symbol").type == pa.string()
    assert set(table.column("dso").to_pylist()) == {"/lib/libggml-cpu.so.0"}


def test_categorical_null_round_trips(tmp_path):
    df = _trace_frame(2)
    df["cache_level"] = pd.Series(["L1", None]).astype("category")
    out = tmp_path / "trace.parquet"
    write_parquet_frame(df, out, schema=TRACE_SCHEMA)
    assert pq.read_table(out).column("cache_level").to_pylist() == ["L1", None]


def test_a_frame_missing_schema_columns_is_written_with_nulls(tmp_path):
    """
    The row-dict path filled an absent column with nulls; from_pandas raises
    on it.  Partial traces are legitimate — fixtures and the memray collector
    both produce them — so the frame must not be rejected.
    """
    df = _trace_frame(5).drop(columns=["access_latency_ns", "cache_level", "numa_node"])
    out = tmp_path / "trace.parquet"
    assert write_parquet_frame(df, out, schema=TRACE_SCHEMA) == 5

    table = pq.read_table(out)
    assert table.schema.equals(TRACE_SCHEMA)
    assert table.column("access_latency_ns").null_count == 5
    assert table.column("cache_level").to_pylist() == [None] * 5
    # The columns that were present must be untouched.
    assert table.column("symbol").to_pylist() == ["ggml_vec_dot_q6_K_q8_K"] * 5


def test_extra_columns_are_dropped_to_match_the_schema(tmp_path):
    df = _trace_frame(5).assign(scratch=[1, 2, 3, 4, 5])
    out = tmp_path / "trace.parquet"
    write_parquet_frame(df, out, schema=TRACE_SCHEMA)
    assert pq.read_table(out).schema.equals(TRACE_SCHEMA)

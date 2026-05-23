"""Unit tests for the streaming Parquet writer."""

import pandas as pd
import pyarrow.parquet as pq
import pytest

from mrm_trace.parser.writer import write_parquet


def _make_rows(n):
    return [
        {
            "run_id": "r1",
            "sample_id": i,
            "timestamp_ns": i * 1_000,
            "phase": None,
            "address": 0x7F00_0000_0000 + i * 64,
            "address_page": (0x7F00_0000_0000 + i * 64) >> 12,
            "op_type": "load",
            "symbol": "test_sym",
            "dso": "/test/lib.so",
            "region_type": "unknown",
            "region_id": 0,
            "access_latency_ns": None,
            "cache_level": None,
            "numa_node": None,
            "cpu_id": 0,
            "thread_id": 12345,
            "token_index": None,
            "context_tokens": None,
        }
        for i in range(n)
    ]


def test_returns_correct_row_count(tmp_path):
    count = write_parquet(iter(_make_rows(100)), tmp_path / "trace.parquet")
    assert count == 100


def test_written_file_is_readable(tmp_path):
    out = tmp_path / "trace.parquet"
    write_parquet(iter(_make_rows(10)), out)
    df = pd.read_parquet(out)
    assert len(df) == 10


def test_row_data_roundtrips(tmp_path):
    out = tmp_path / "trace.parquet"
    write_parquet(iter(_make_rows(5)), out)
    df = pd.read_parquet(out)
    assert list(df["sample_id"]) == [0, 1, 2, 3, 4]


def test_batch_boundary_preserves_row_count(tmp_path):
    n = 350
    out = tmp_path / "trace.parquet"
    write_parquet(iter(_make_rows(n)), out, batch_size=100)
    df = pd.read_parquet(out)
    assert len(df) == n


def test_empty_input_returns_zero_and_no_file(tmp_path):
    out = tmp_path / "trace.parquet"
    count = write_parquet(iter([]), out)
    assert count == 0
    assert not out.exists()


def test_zstd_compression_used(tmp_path):
    out = tmp_path / "trace.parquet"
    write_parquet(iter(_make_rows(10)), out)
    meta = pq.read_metadata(str(out))
    col_meta = meta.row_group(0).column(0)
    assert col_meta.compression == "ZSTD"


def test_single_row_written_correctly(tmp_path):
    out = tmp_path / "trace.parquet"
    count = write_parquet(iter(_make_rows(1)), out)
    assert count == 1
    df = pd.read_parquet(out)
    assert len(df) == 1
    assert df.iloc[0]["run_id"] == "r1"

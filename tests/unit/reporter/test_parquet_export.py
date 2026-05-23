"""Unit tests for parquet_export helpers."""

import pandas as pd
import pyarrow.parquet as pq
import pytest

from mrm_trace.reporter.parquet_export import (
    export_kv_lifecycle,
    export_region_map,
    export_trace,
)


def _minimal_trace():
    return pd.DataFrame([{
        "run_id": "r1", "sample_id": 0, "timestamp_ns": 0,
        "address": 0x1000, "address_page": 1, "op_type": "load",
        "region_type": "weight", "region_id": 1,
        "phase": "generation", "symbol": "ggml_init", "dso": "lib.so",
        "access_latency_ns": None, "cache_level": None, "numa_node": None,
        "cpu_id": 0, "thread_id": 1, "token_index": None, "context_tokens": None,
    }])


def _minimal_region_map():
    return [{"region_id": 1, "region_type": "weight", "page_min": 1, "page_max": 1,
             "first_seen_ns": 0, "last_seen_ns": 0, "access_count": 1}]


def _minimal_lifecycle():
    return [{"block_id": 1, "address": 0x1000, "address_page": 1, "write_ns": 0,
             "last_read_ns": 100, "evict_ns": None, "total_reads": 1, "lifetime_ns": 100}]


# ── export_trace ──────────────────────────────────────────────────────────────

def test_export_trace_creates_file(tmp_path):
    p = export_trace(_minimal_trace(), tmp_path)
    assert p.exists()
    assert p.name == "trace.parquet"


def test_export_trace_readable(tmp_path):
    export_trace(_minimal_trace(), tmp_path)
    df = pd.read_parquet(tmp_path / "trace.parquet")
    assert "region_type" in df.columns
    assert len(df) == 1


def test_export_trace_uses_zstd(tmp_path):
    export_trace(_minimal_trace(), tmp_path)
    pf = pq.ParquetFile(tmp_path / "trace.parquet")
    rg = pf.metadata.row_group(0)
    col = rg.column(0)
    assert col.compression == "ZSTD"


# ── export_region_map ─────────────────────────────────────────────────────────

def test_export_region_map_creates_file(tmp_path):
    p = export_region_map(_minimal_region_map(), tmp_path)
    assert p.name == "region_map.parquet"
    assert p.exists()


def test_export_region_map_accepts_dataframe(tmp_path):
    df = pd.DataFrame(_minimal_region_map())
    p = export_region_map(df, tmp_path)
    assert p.exists()


def test_export_region_map_readable(tmp_path):
    export_region_map(_minimal_region_map(), tmp_path)
    df = pd.read_parquet(tmp_path / "region_map.parquet")
    assert "region_id" in df.columns


# ── export_kv_lifecycle ───────────────────────────────────────────────────────

def test_export_kv_lifecycle_creates_file(tmp_path):
    p = export_kv_lifecycle(_minimal_lifecycle(), tmp_path)
    assert p.name == "kv_block_lifecycle.parquet"
    assert p.exists()


def test_export_kv_lifecycle_readable(tmp_path):
    export_kv_lifecycle(_minimal_lifecycle(), tmp_path)
    df = pd.read_parquet(tmp_path / "kv_block_lifecycle.parquet")
    assert "block_id" in df.columns

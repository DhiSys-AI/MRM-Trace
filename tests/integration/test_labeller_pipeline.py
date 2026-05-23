"""
Integration tests for the labeller pipeline.

Uses raw (unlabelled) trace rows that mirror what the perf script parser
produces — region_type='unknown', region_id=0 — and verifies that the
labeller classifies them correctly end-to-end.
"""

import pytest

from mrm_trace.labeller.labeller import TraceLabeller
from mrm_trace.parser.writer import write_parquet
import pandas as pd


def _raw_row(
    address: int,
    symbol: str,
    dso: str = "libmodel.so",
    phase: str = None,
    op_type: str = "load",
    timestamp_ns: int = 0,
    sample_id: int = 0,
) -> dict:
    return {
        "run_id": "r1",
        "sample_id": sample_id,
        "timestamp_ns": timestamp_ns,
        "phase": phase,
        "address": address,
        "address_page": address >> 12,
        "op_type": op_type,
        "symbol": symbol,
        "dso": dso,
        "region_type": "unknown",
        "region_id": 0,
        "access_latency_ns": None,
        "cache_level": None,
        "numa_node": None,
        "cpu_id": 1,
        "thread_id": 100,
        "token_index": None,
        "context_tokens": None,
    }


# ── weight classification ─────────────────────────────────────────────────────

class TestWeightLabelling:

    def test_weight_load_phase_classified_as_weight(self):
        row = _raw_row(0x7F00_0000_0000, "ggml_init", phase="weight_load", op_type="store")
        rows = list(TraceLabeller().label([row]))
        assert rows[0]["region_type"] == "weight"

    def test_weight_rows_all_get_region_id_1(self):
        raw = [
            _raw_row(0x7F00_0000_0000 + i * 4096, "ggml_init",
                     phase="weight_load", op_type="store",
                     timestamp_ns=i, sample_id=i)
            for i in range(10)
        ]
        rows = list(TraceLabeller().label(raw))
        assert all(r["region_id"] == 1 for r in rows)

    def test_weight_compute_kernel_classified_as_weight(self):
        row = _raw_row(0x7F00_0000_0000, "ggml_vec_dot_q4_K",
                       phase="generation", op_type="load")
        rows = list(TraceLabeller().label([row]))
        assert rows[0]["region_type"] == "weight"


# ── KV cache classification ───────────────────────────────────────────────────

class TestKVCacheLabelling:

    def test_kv_symbol_classified_as_kv_cache(self):
        row = _raw_row(0xAA00_0000_0000, "llama_kv_cache_update",
                       phase="generation", op_type="store")
        rows = list(TraceLabeller().label([row]))
        assert rows[0]["region_type"] == "kv_cache"

    def test_kv_blocks_get_distinct_region_ids(self):
        labeller = TraceLabeller()
        kv_rows = [
            _raw_row(0xAA00_0000_0000 + i * 4096, "llama_kv_cache_update",
                     phase="generation", op_type="store",
                     timestamp_ns=i * 10, sample_id=i)
            for i in range(5)
        ]
        labelled = list(labeller.label(kv_rows))
        ids = [r["region_id"] for r in labelled]
        assert len(set(ids)) == 5  # all distinct

    def test_kv_region_ids_start_at_2_after_weight(self):
        labeller = TraceLabeller()
        rows = [
            _raw_row(0x7F00_0000_0000, "ggml_init", phase="weight_load",
                     op_type="store", sample_id=0, timestamp_ns=0),
            _raw_row(0xAA00_0000_0000, "llama_kv_cache_update", phase="generation",
                     op_type="store", sample_id=1, timestamp_ns=100),
        ]
        labelled = list(labeller.label(rows))
        assert labelled[0]["region_id"] == 1   # weight
        assert labelled[1]["region_id"] == 2   # first KV block


# ── address propagation ───────────────────────────────────────────────────────

class TestAddressPropagation:

    def test_unknown_access_to_known_weight_page_inherits_weight(self):
        page = 0x7F00_0000_0000
        labeller = TraceLabeller()
        rows = list(labeller.label([
            _raw_row(page, "ggml_init", phase="weight_load", op_type="store",
                     timestamp_ns=0, sample_id=0),
            _raw_row(page, "", phase="generation", op_type="load",
                     timestamp_ns=100, sample_id=1),
        ]))
        assert rows[1]["region_type"] == "weight"
        assert rows[1]["region_id"] == 1

    def test_unknown_access_to_known_kv_page_inherits_kv_cache(self):
        page = 0xAA00_0000_0000
        labeller = TraceLabeller()
        rows = list(labeller.label([
            _raw_row(page, "llama_kv_cache_update", phase="generation",
                     op_type="store", timestamp_ns=0, sample_id=0),
            _raw_row(page, "", phase="generation", op_type="load",
                     timestamp_ns=50, sample_id=1),
        ]))
        assert rows[1]["region_type"] == "kv_cache"


# ── region_map and kv_lifecycle ───────────────────────────────────────────────

class TestRegionMapAndLifecycle:

    def test_region_map_has_weight_and_kv_entries(self):
        labeller = TraceLabeller()
        rows = [
            _raw_row(0x7F00_0000_0000, "ggml_init", phase="weight_load",
                     op_type="store", sample_id=0, timestamp_ns=0),
            _raw_row(0xAA00_0000_0000, "llama_kv_cache_update", phase="generation",
                     op_type="store", sample_id=1, timestamp_ns=100),
        ]
        list(labeller.label(rows))
        rmap = labeller.region_map()
        types = {r["region_type"] for r in rmap}
        assert "weight" in types
        assert "kv_cache" in types

    def test_kv_lifecycle_captures_write_and_reads(self):
        labeller = TraceLabeller()
        page = 0xAA00_0000_0000
        rows = [
            _raw_row(page, "llama_kv_cache_update", phase="generation",
                     op_type="store", timestamp_ns=100, sample_id=0),
            _raw_row(page, "llama_kv_cache_update", phase="generation",
                     op_type="load", timestamp_ns=200, sample_id=1),
            _raw_row(page, "llama_kv_cache_update", phase="generation",
                     op_type="load", timestamp_ns=300, sample_id=2),
        ]
        list(labeller.label(rows))
        lc = labeller.kv_lifecycle()
        assert len(lc) == 1
        assert lc[0]["write_ns"] == 100
        assert lc[0]["total_reads"] == 2
        assert lc[0]["lifetime_ns"] == 200

    def test_region_map_written_to_parquet(self, tmp_path):
        labeller = TraceLabeller()
        rows = [
            _raw_row(0x7F00_0000_0000, "ggml_init", phase="weight_load",
                     op_type="store", sample_id=0, timestamp_ns=0),
        ]
        list(labeller.label(rows))
        out = tmp_path / "region_map.parquet"
        write_parquet(iter(labeller.region_map()), out)
        df = pd.read_parquet(out)
        assert "region_type" in df.columns
        assert len(df) >= 1

    def test_kv_lifecycle_written_to_parquet(self, tmp_path):
        labeller = TraceLabeller()
        rows = [
            _raw_row(0xAA00_0000_0000, "llama_kv_cache_update", phase="generation",
                     op_type="store", timestamp_ns=0, sample_id=0),
        ]
        list(labeller.label(rows))
        out = tmp_path / "kv_block_lifecycle.parquet"
        write_parquet(iter(labeller.kv_lifecycle()), out)
        df = pd.read_parquet(out)
        assert "block_id" in df.columns

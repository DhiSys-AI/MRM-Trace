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


_W_BASE = 0x7F00_0000_0000
_KV_BASE = 0xAA00_0000_0000


def _labeller(**kwargs):
    """
    TraceLabeller with the fixture's weight and KV address ranges registered.

    weight and kv_cache are established by address range only.  A symbol says
    which code was running, not which region the sampled address belongs to —
    measured on a real run, 100% of the rows a ggml_vec_dot rule labelled
    `weight` were outside the mmap'd GGUF file.  The real pipeline takes the
    weight range from that mapping and the KV range from the engine's own
    allocation; here they are the fixture's known bases.
    """
    labeller = TraceLabeller(**kwargs)
    labeller.register_address_ranges([
        {"region_type": "weight", "addr_start": _W_BASE,
         "addr_end": _W_BASE + 64 * 4096, "source": "mmap_gguf"},
        {"region_type": "kv_cache", "addr_start": _KV_BASE,
         "addr_end": _KV_BASE + 64 * 4096, "source": "kv_allocation"},
    ])
    return labeller


# ── weight classification ─────────────────────────────────────────────────────

class TestWeightLabelling:

    def test_weight_load_phase_classified_as_weight(self):
        row = _raw_row(0x7F00_0000_0000, "ggml_init", phase="weight_load", op_type="store")
        rows = list(_labeller().label([row]))
        assert rows[0]["region_type"] == "weight"

    def test_weight_pages_get_one_region_each(self):
        """One region per page — retention needs a distribution, not one sample."""
        raw = [
            _raw_row(0x7F00_0000_0000 + i * 4096, "ggml_init",
                     phase="weight_load", op_type="store",
                     timestamp_ns=i, sample_id=i)
            for i in range(10)
        ]
        rows = list(_labeller().label(raw))
        assert all(r["region_type"] == "weight" for r in rows)
        assert len({r["region_id"] for r in rows}) == 10

    def test_weight_comes_from_the_range_not_the_kernel_symbol(self):
        """Inside the weight range the label is weight — because of the range."""
        row = _raw_row(_W_BASE, "ggml_vec_dot_q4_K",
                       phase="generation", op_type="load")
        rows = list(_labeller().label([row]))
        assert rows[0]["region_type"] == "weight"

    def test_a_compute_kernel_cannot_claim_weight_outside_the_range(self):
        """
        The defect this replaces: ggml_vec_dot reads weights but writes
        activations, so its store address is an output buffer.  On a real run
        that rule labelled 26,617,751 rows `weight`, of which 114 were actually
        inside the weight mapping.
        """
        row = _raw_row(0xBBBB_0000_0000, "ggml_vec_dot_q4_K",
                       phase="generation", op_type="store")
        rows = list(_labeller().label([row]))
        assert rows[0]["region_type"] != "weight"
        assert rows[0]["region_type"] == "activation"


# ── KV cache classification ───────────────────────────────────────────────────

class TestKVCacheLabelling:

    def test_kv_comes_from_the_allocation_range(self):
        row = _raw_row(_KV_BASE, "llama_kv_cache_update",
                       phase="generation", op_type="store")
        rows = list(_labeller().label([row]))
        assert rows[0]["region_type"] == "kv_cache"

    def test_a_kv_symbol_cannot_claim_kv_cache_outside_the_range(self):
        """
        llama_kv_cache_* manages slot metadata, not the K/V tensors, so a
        sample taken inside it is not a KV address.  On a real run that rule
        tagged four granules — one of them in the main stack — which then
        absorbed 279,186 accesses from nearest_int, memmove and softmax.
        """
        row = _raw_row(0xCCCC_0000_0000, "llama_kv_cache_update",
                       phase="generation", op_type="store")
        rows = list(_labeller().label([row]))
        assert rows[0]["region_type"] != "kv_cache"

    def test_kv_blocks_get_distinct_region_ids(self):
        labeller = _labeller()
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
        labeller = _labeller()
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
        labeller = _labeller()
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
        labeller = _labeller()
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
        labeller = _labeller()
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
        labeller = _labeller()
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
        labeller = _labeller()
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
        labeller = _labeller()
        rows = [
            _raw_row(0xAA00_0000_0000, "llama_kv_cache_update", phase="generation",
                     op_type="store", timestamp_ns=0, sample_id=0),
        ]
        list(labeller.label(rows))
        out = tmp_path / "kv_block_lifecycle.parquet"
        write_parquet(iter(labeller.kv_lifecycle()), out)
        df = pd.read_parquet(out)
        assert "block_id" in df.columns

"""
Tier-3 validity tests for Phase 5 — Labelling.

These tests verify scientific correctness properties that, if violated, would
corrupt the region_map, kv_block_lifecycle, or the labelled trace.parquet,
invalidating downstream retention / write-once / locality analysis.
"""

import pytest

from mrm_trace.labeller.labeller import TraceLabeller
from mrm_trace.parser.schema import OpType, Phase, RegionType


def _raw(address, symbol, dso="lib.so", phase=None, op_type="load",
         timestamp_ns=0, sample_id=0):
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
        "cpu_id": 0,
        "thread_id": 1,
        "token_index": None,
        "context_tokens": None,
    }


def test_weight_pages_get_distinct_region_ids():
    """
    Each weight page must get its own region_id.

    Retention is reported per region_type as p50/p90/p99/max/mean/std (spec §6.1).
    Collapsing every weight page into a single region_id yields exactly one
    retention sample, so all percentiles coincide and the reported "distribution"
    is a point — which is what the pre-2026-08 output showed for weight.
    """
    labeller = TraceLabeller()
    n = 20
    weight_rows = [
        _raw(0x7F00_0000_0000 + i * 4096, "ggml_init",
             phase="weight_load", op_type="store", dso="libggml.so",
             timestamp_ns=i, sample_id=i)
        for i in range(n)
    ]
    labelled = list(labeller.label(weight_rows))
    assert all(r["region_type"] == "weight" for r in labelled)
    ids = {r["region_id"] for r in labelled}
    assert len(ids) == n, (
        f"Expected {n} distinct weight region ids (one per page), got {len(ids)}. "
        "Merged ids collapse the retention distribution to a single sample."
    )


def test_same_page_accesses_share_a_region_id():
    """Retention needs several samples per region — same page must mean same region."""
    labeller = TraceLabeller()
    rows = [
        _raw(0x7F00_0000_0000 + off, "ggml_init",
             phase="weight_load", op_type="store", dso="libggml.so",
             timestamp_ns=i, sample_id=i)
        for i, off in enumerate((0, 8, 64, 4095))
    ]
    ids = {r["region_id"] for r in labeller.label(rows)}
    assert len(ids) == 1, (
        "Byte offsets within one page must map to one region — perf reports "
        "arbitrary addresses inside an object, and splitting them gives every "
        "region a single access and therefore retention == 0."
    )


def test_kv_cache_blocks_have_unique_region_ids():
    """
    Each distinct KV cache write address must receive a unique region_id.
    Shared region_ids would merge KV block lifecycles and corrupt per-block
    retention statistics.
    """
    labeller = TraceLabeller()
    n = 50
    kv_rows = [
        _raw(0xAA00_0000_0000 + i * 4096, "llama_kv_cache_update",
             phase="generation", op_type="store",
             timestamp_ns=i * 10, sample_id=i)
        for i in range(n)
    ]
    labelled = list(labeller.label(kv_rows))
    ids = [r["region_id"] for r in labelled]
    assert len(set(ids)) == n, (
        f"Expected {n} distinct KV region_ids, got {len(set(ids))}. "
        "Duplicate region_ids merge KV block lifecycles and corrupt per-block retention."
    )


def test_address_propagation_never_contradicts_symbol_classification():
    """
    Once a page is classified by a symbol rule, subsequent accesses to that page
    must always inherit the same region_type, even if later symbol matches differ.
    Inconsistent classification would split one physical region across multiple
    region_ids and corrupt working-set size calculations.
    """
    page = 0x7F00_0000_0000
    labeller = TraceLabeller()
    rows = [
        # First access: symbol rule fires → weight
        _raw(page, "ggml_init", phase="weight_load", op_type="store",
             timestamp_ns=0, sample_id=0),
        # Second access: same page, no symbol → propagation
        _raw(page, "", phase="generation", op_type="load",
             timestamp_ns=100, sample_id=1),
        # Third access: same page, a different (non-weight) symbol
        _raw(page, "malloc", phase=None, op_type="load",
             timestamp_ns=200, sample_id=2),
    ]
    labelled = list(labeller.label(rows))
    types = [r["region_type"] for r in labelled]
    assert all(t == "weight" for t in types), (
        f"Inconsistent region_types across same page: {types}. "
        "Address propagation must be stable — first classification must win."
    )


def test_kv_lifecycle_write_ns_precedes_all_reads():
    """
    For every KV block, write_ns must be ≤ all load timestamps.
    write_ns > last_read_ns would produce negative lifetime_ns, corrupting
    KV retention histograms.
    """
    labeller = TraceLabeller()
    base = 0xBB00_0000_0000
    rows = [
        _raw(base + i * 4096, "llama_kv_cache_update",
             phase="generation", op_type="store",
             timestamp_ns=100 + i * 50, sample_id=i * 3)
        for i in range(10)
    ] + [
        _raw(base + i * 4096, "llama_kv_cache_update",
             phase="generation", op_type="load",
             timestamp_ns=100 + i * 50 + 30, sample_id=i * 3 + 1)
        for i in range(10)
    ]
    rows.sort(key=lambda r: r["timestamp_ns"])
    for i, r in enumerate(rows):
        r["sample_id"] = i

    list(labeller.label(rows))
    for block in labeller.kv_lifecycle():
        assert block["write_ns"] <= (block["last_read_ns"] or block["write_ns"]), (
            f"Block {block['block_id']}: write_ns={block['write_ns']} > "
            f"last_read_ns={block['last_read_ns']}. "
            "Negative lifetime_ns corrupts KV retention histograms."
        )


def test_labelling_synthetic_trace_fixture(synthetic_trace_factory):
    """
    Labelling the unlabelled equivalent of the conftest synthetic fixture must
    produce the same region_type assignments as the fixture's ground truth.

    Acceptance criteria for Phase 5: 'Test fixtures classify weight + KV regions.'
    """
    # Build raw (unlabelled) rows matching conftest fixture symbols and phases
    n_weight = 50
    n_kv = 20
    raw_rows = []
    sid = 0

    # Weight: written during weight_load, read during generation
    raw_rows.append(_raw(0x7F00_0000_0000, "ggml_init",
                         phase="weight_load", op_type="store",
                         timestamp_ns=0, sample_id=sid))
    sid += 1
    for i in range(n_weight):
        raw_rows.append(_raw(0x7F00_0000_0000 + i * 64, "ggml_vec_dot_q4_K",
                             phase="generation", op_type="load",
                             timestamp_ns=1000 + i, sample_id=sid))
        sid += 1

    # KV cache: store then read
    for i in range(n_kv):
        addr = 0x7F10_0000_0000 + i * 256
        raw_rows.append(_raw(addr, "llama_kv_cache_update",
                             phase="generation", op_type="store",
                             timestamp_ns=2000 + i * 10, sample_id=sid))
        sid += 1
        raw_rows.append(_raw(addr, "llama_kv_cache_update",
                             phase="generation", op_type="load",
                             timestamp_ns=2000 + i * 10 + 5, sample_id=sid))
        sid += 1

    labeller = TraceLabeller()
    labelled = list(labeller.label(raw_rows))

    weight_rows = [r for r in labelled if r["sample_id"] <= n_weight]
    kv_rows = [r for r in labelled if r["sample_id"] > n_weight]

    assert all(r["region_type"] == "weight" for r in weight_rows), (
        "Weight rows not classified as 'weight'. "
        "Phase 5 acceptance criterion: test fixtures must classify weight regions."
    )
    assert all(r["region_type"] == "kv_cache" for r in kv_rows), (
        "KV rows not classified as 'kv_cache'. "
        "Phase 5 acceptance criterion: test fixtures must classify KV regions."
    )
    # The 50 weight rows span 50 * 64 B = 3200 B — one page, so one region.
    assert len({r["region_id"] for r in weight_rows}) == 1

    # The 20 KV rows are 256 B apart, spanning 5120 B — two pages at the default
    # 4 KiB granularity.  Region identity is per granule, not per address: perf
    # reports arbitrary byte addresses inside a block, so per-address ids give
    # every region exactly one access and retention == 0 by construction.
    kv_ids = {r["region_id"] for r in kv_rows if r["op_type"] == "store"}
    assert len(kv_ids) == 2, f"Expected 2 KV granules at 4 KiB, got {len(kv_ids)}"

    # Finer granularity resolves the individual 256 B blocks — the parameter is
    # what makes this an auditable choice rather than a hidden assumption.
    fine = TraceLabeller(region_granularity_bytes=256)
    fine_labelled = list(fine.label(raw_rows))
    fine_kv = {
        r["region_id"] for r in fine_labelled
        if r["sample_id"] > n_weight and r["op_type"] == "store"
    }
    assert len(fine_kv) == n_kv, (
        f"Expected {n_kv} distinct KV blocks at 256 B granularity, got {len(fine_kv)}"
    )

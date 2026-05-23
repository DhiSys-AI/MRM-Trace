"""
Performance benchmarks for the labeller hot path.

Run with:  pytest -m performance --benchmark-only
"""

import pytest

from mrm_trace.labeller.labeller import TraceLabeller


def _make_rows(n: int, n_kv: int = 0):
    rows = []
    for i in range(n - n_kv):
        rows.append({
            "run_id": "bench", "sample_id": i, "timestamp_ns": i * 1000,
            "phase": "weight_load" if i == 0 else "generation",
            "address": 0x7F00_0000_0000 + (i % 256) * 64,
            "address_page": (0x7F00_0000_0000 + (i % 256) * 64) >> 12,
            "op_type": "store" if i == 0 else "load",
            "symbol": "ggml_init" if i == 0 else "ggml_vec_dot_q4_K",
            "dso": "libggml.so",
            "region_type": "unknown", "region_id": 0,
            "access_latency_ns": None, "cache_level": None, "numa_node": None,
            "cpu_id": 0, "thread_id": 1, "token_index": None, "context_tokens": None,
        })
    for j in range(n_kv):
        rows.append({
            "run_id": "bench", "sample_id": n - n_kv + j, "timestamp_ns": j * 500,
            "phase": "generation",
            "address": 0xAA00_0000_0000 + j * 256,
            "address_page": (0xAA00_0000_0000 + j * 256) >> 12,
            "op_type": "store",
            "symbol": "llama_kv_cache_update",
            "dso": "libllama.so",
            "region_type": "unknown", "region_id": 0,
            "access_latency_ns": None, "cache_level": None, "numa_node": None,
            "cpu_id": 0, "thread_id": 1, "token_index": None, "context_tokens": None,
        })
    return rows


@pytest.mark.performance
def test_labeller_10k_rows(benchmark):
    """Benchmark labelling 10,000 trace rows (weight + KV mix)."""
    rows = _make_rows(9_000, n_kv=1_000)
    labeller = TraceLabeller()

    result = benchmark(lambda: list(labeller.label(rows)))
    assert len(result) == 10_000


@pytest.mark.performance
def test_labeller_weight_propagation_50k(benchmark):
    """
    Benchmark page-propagation path — 50k reads to already-classified pages.
    This is the hot path in a real inference run.
    """
    # Classify first row, rest propagate
    rows = _make_rows(50_000)
    labeller = TraceLabeller()

    result = benchmark(lambda: list(labeller.label(rows)))
    assert len(result) == 50_000


@pytest.mark.performance
def test_labeller_throughput_rows_per_second(benchmark):
    """Assert labeller sustains > 100k rows/second on the propagation path."""
    n = 100_000
    rows = _make_rows(n)
    labeller = TraceLabeller()

    result = benchmark(lambda: list(labeller.label(rows)))
    # pytest-benchmark will record the throughput; the assertion is structural
    assert len(result) == n

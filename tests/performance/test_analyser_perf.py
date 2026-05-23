"""
Performance benchmarks for the analyser hot path.

Run with:  pytest -m performance --benchmark-only
"""

import pytest

from mrm_trace.analyser import compute_all


@pytest.mark.performance
def test_compute_all_default_fixture(benchmark, synthetic_trace_factory):
    """Benchmark compute_all() on the default synthetic fixture (~1600 rows)."""
    trace = synthetic_trace_factory()
    result = benchmark(lambda: compute_all(trace))
    assert "suitability" in result


@pytest.mark.performance
def test_compute_all_large_trace(benchmark, synthetic_trace_factory):
    """Benchmark compute_all() on a larger synthetic trace (~6000 rows)."""
    trace = synthetic_trace_factory(n_weight_reads=5000, n_kv_writes=200, n_kv_reads=1000)
    result = benchmark(lambda: compute_all(trace))
    assert "retention_summary" in result


@pytest.mark.performance
def test_retention_computation(benchmark, synthetic_trace_factory):
    """Benchmark just the retention module on 1k+ regions."""
    from mrm_trace.analyser.retention import compute_retention
    trace = synthetic_trace_factory(n_kv_writes=500, n_kv_reads=500)
    result = benchmark(lambda: compute_retention(trace))
    assert len(result[0]) > 0

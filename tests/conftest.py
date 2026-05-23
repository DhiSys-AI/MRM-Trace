import numpy as np
import pandas as pd
import pytest

from mrm_trace.parser.schema import OpType, Phase, RegionType


@pytest.fixture
def synthetic_trace_factory():
    """
    Factory fixture for synthetic traces with fully known ground truth.

    Usage:
        trace = synthetic_trace_factory(n_weight_reads=1000, n_kv_writes=50)
    """

    def _factory(
        n_weight_reads: int = 1000,
        n_kv_writes: int = 100,
        n_kv_reads: int = 500,
        weight_retention_s: float = 30.0,
        kv_retention_s: float = 5.0,
        seed: int = 42,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rows = []
        sample_id = 0
        t = 0.0

        # Weight region: one store, many reads, long retention.
        weight_addr_base = 0x7F0000000000
        rows.append(
            dict(
                run_id="test_run",
                sample_id=sample_id,
                timestamp_ns=int(t * 1e9),
                address=weight_addr_base,
                address_page=weight_addr_base >> 12,
                op_type=OpType.STORE.value,
                region_type=RegionType.WEIGHT.value,
                phase=Phase.WEIGHT_LOAD.value,
                symbol="ggml_init",
                dso="libggml.so",
                region_id=1,
            )
        )
        sample_id += 1

        for i in range(n_weight_reads):
            t += weight_retention_s / n_weight_reads
            addr = weight_addr_base + int(rng.integers(0, 4096)) * 64
            rows.append(
                dict(
                    run_id="test_run",
                    sample_id=sample_id,
                    timestamp_ns=int(t * 1e9),
                    address=addr,
                    address_page=addr >> 12,
                    op_type=OpType.LOAD.value,
                    region_type=RegionType.WEIGHT.value,
                    phase=Phase.GENERATION.value,
                    symbol="ggml_vec_dot_q4_K",
                    dso="libggml.so",
                    region_id=1,
                )
            )
            sample_id += 1

        # KV cache regions: many stores, many reads, short retention windows.
        kv_addr_base = 0x7F1000000000
        t_kv_start = 1.0
        reads_per_block = max(1, n_kv_reads // n_kv_writes)
        for i in range(n_kv_writes):
            t_kv = t_kv_start + i * (kv_retention_s / n_kv_writes)
            addr = kv_addr_base + i * 256
            rows.append(
                dict(
                    run_id="test_run",
                    sample_id=sample_id,
                    timestamp_ns=int(t_kv * 1e9),
                    address=addr,
                    address_page=addr >> 12,
                    op_type=OpType.STORE.value,
                    region_type=RegionType.KV_CACHE.value,
                    phase=Phase.GENERATION.value,
                    symbol="llama_kv_cache_update",
                    dso="libllama.so",
                    region_id=2 + i,
                )
            )
            sample_id += 1
            for j in range(reads_per_block):
                rows.append(
                    dict(
                        run_id="test_run",
                        sample_id=sample_id,
                        timestamp_ns=int((t_kv + (j + 1) * 0.01) * 1e9),
                        address=addr,
                        address_page=addr >> 12,
                        op_type=OpType.LOAD.value,
                        region_type=RegionType.KV_CACHE.value,
                        phase=Phase.GENERATION.value,
                        symbol="llama_kv_cache_update",
                        dso="libllama.so",
                        region_id=2 + i,
                    )
                )
                sample_id += 1

        return pd.DataFrame(rows)

    return _factory


@pytest.fixture
def clean_metadata():
    return {
        "run_id": "test_run_001",
        "observer_effect": {
            "observer_effect_summary": {
                "run_validity": "clean",
                "observer_cpu_overhead_pct_max": 4.2,
                "observer_memory_overhead_mb_peak": 312.0,
                "thermal_throttle_events": 0,
                "io_contention_detected": False,
            }
        },
    }


@pytest.fixture
def contaminated_metadata():
    return {
        "run_id": "test_run_002",
        "observer_effect": {
            "observer_effect_summary": {
                "run_validity": "contaminated",
                "observer_cpu_overhead_pct_max": 28.5,
                "observer_memory_overhead_mb_peak": 4800.0,
                "thermal_throttle_events": 5,
                "io_contention_detected": True,
            }
        },
    }

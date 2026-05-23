## Testing Philosophy for a Measurement Tool

Standard software testing asks: *"Does the code do what I intended?"* For `mrm-trace`, you need a harder question: *"Are the measurements scientifically valid?"* This means tests fall into three tiers beyond the usual unit/integration split:

```
Tier 1: Unit Tests          — individual functions behave correctly
Tier 2: Integration Tests   — components work together correctly
Tier 3: Validity Tests      — measurements are scientifically sound
         (unique to measurement tools)
```


***

## Full Test Structure

```
tests/
│
├── conftest.py                     ← shared fixtures, synthetic trace generators
├── fixtures/
│   ├── traces/
│   │   ├── minimal_trace.parquet   ← 100-row synthetic trace, known ground truth
│   │   ├── weight_only.parquet     ← only weight_load phase samples
│   │   ├── kvcache_only.parquet    ← only kv_cache samples
│   │   ├── mixed_phases.parquet    ← weight_load then generation interleaved
│   │   └── contaminated.parquet    ← high observer overhead, should flag invalid
│   ├── perf_output/
│   │   ├── sample_perf_script.txt  ← real perf script stdout snippet
│   │   ├── malformed_lines.txt     ← corrupt lines, missing fields
│   │   └── empty_output.txt
│   ├── memray/
│   │   └── sample_memray.bin       ← small memray capture
│   └── metadata/
│       ├── clean_run.json
│       └── contaminated_run.json
│
├── unit/
│   ├── engines/
│   │   ├── test_base_engine.py
│   │   ├── test_llamacpp_engine.py
│   │   └── test_vllm_engine.py
│   ├── collector/
│   │   ├── test_perf_runner.py
│   │   ├── test_memray_runner.py
│   │   └── test_process_monitor.py
│   ├── parser/
│   │   ├── test_perf_script_parser.py
│   │   ├── test_memray_parser.py
│   │   └── test_schema.py
│   ├── labeller/
│   │   ├── test_symbol_labeller.py
│   │   ├── test_address_tracker.py
│   │   └── test_region_map.py
│   ├── analyser/
│   │   ├── test_retention.py
│   │   ├── test_write_once.py
│   │   ├── test_read_frequency.py
│   │   └── test_working_set.py
│   ├── telemetry/
│   │   ├── test_hardware.py
│   │   ├── test_software.py
│   │   ├── test_thermal.py
│   │   ├── test_observer.py
│   │   └── test_baseline.py
│   └── reporter/
│       ├── test_csv_exporter.py
│       ├── test_plot.py
│       └── test_summary.py
│
├── integration/
│   ├── test_collector_to_parser.py     ← perf output → parsed DataFrame
│   ├── test_parser_to_labeller.py      ← parsed → labelled region types
│   ├── test_labeller_to_analyser.py    ← labelled → MRM metrics
│   ├── test_analyser_to_reporter.py    ← metrics → CSV + figures
│   ├── test_full_pipeline_synthetic.py ← end-to-end with synthetic trace
│   └── test_metadata_linkage.py        ← trace.parquet run_id → metadata.json
│
├── validity/                           ← Tier 3: scientific correctness
│   ├── test_ground_truth_metrics.py    ← known-input → known-output
│   ├── test_observer_effect_bounds.py  ← overhead within acceptable limits
│   ├── test_retention_monotonicity.py  ← retention can't be negative
│   ├── test_write_once_consistency.py  ← write_once_ratio ∈ [0.0, 1.0]
│   ├── test_phase_coverage.py          ← all phases present in labelled trace
│   ├── test_no_cross_run_leakage.py    ← run_ids don't mix across outputs
│   └── test_thermal_invalidation.py    ← throttled run → validity = contaminated
│
└── performance/                        ← parser must handle real trace sizes
    ├── test_parser_throughput.py        ← ≥ 500K rows/sec parse rate
    ├── test_streaming_memory.py         ← parser RAM stays under 2 GB for 50M rows
    └── test_parquet_write_speed.py
```


***

## `conftest.py` — Shared Fixtures

The most important file in the test suite — defines reusable synthetic trace generators that every test can use without touching real inference:

```python
# tests/conftest.py
import pytest
import pandas as pd
import numpy as np
from mrm_trace.parser.schema import RegionType, OpType, Phase

@pytest.fixture
def synthetic_trace_factory():
    """
    Returns a factory that generates traces with fully known ground truth.
    Usage: trace = synthetic_trace_factory(n_weight_reads=1000, n_kv_writes=50)
    """
    def _factory(
        n_weight_reads: int = 1000,
        n_kv_writes: int = 100,
        n_kv_reads: int = 500,
        weight_retention_s: float = 30.0,
        kv_retention_s: float = 5.0,
        seed: int = 42
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rows = []
        t = 0

        # Weight region: one store, many reads, long retention
        weight_addr_base = 0x7f0000000000
        rows.append(dict(
            timestamp_ns=int(t * 1e9), address=weight_addr_base,
            op_type=OpType.STORE, region_type=RegionType.WEIGHT,
            phase=Phase.WEIGHT_LOAD, symbol="ggml_init", region_id=1
        ))
        for i in range(n_weight_reads):
            t += weight_retention_s / n_weight_reads
            rows.append(dict(
                timestamp_ns=int(t * 1e9),
                address=weight_addr_base + rng.integers(0, 4096) * 64,
                op_type=OpType.LOAD, region_type=RegionType.WEIGHT,
                phase=Phase.GENERATION, symbol="ggml_vec_dot_q4_K",
                region_id=1
            ))

        # KV region: many stores, many reads, short retention
        kv_addr_base = 0x7f1000000000
        t_kv_start = 1.0
        for i in range(n_kv_writes):
            t_kv = t_kv_start + i * (kv_retention_s / n_kv_writes)
            rows.append(dict(
                timestamp_ns=int(t_kv * 1e9),
                address=kv_addr_base + i * 256,
                op_type=OpType.STORE, region_type=RegionType.KV_CACHE,
                phase=Phase.GENERATION, symbol="llama_kv_cache_update",
                region_id=2 + i
            ))
            for j in range(n_kv_reads // n_kv_writes):
                rows.append(dict(
                    timestamp_ns=int((t_kv + j * 0.01) * 1e9),
                    address=kv_addr_base + i * 256,
                    op_type=OpType.LOAD, region_type=RegionType.KV_CACHE,
                    phase=Phase.GENERATION, symbol="llama_kv_cache_update",
                    region_id=2 + i
                ))

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
                "thermal_throttle_events": 0
            }
        }
    }
```


***

## Key Unit Tests

### `tests/unit/parser/test_perf_script_parser.py`

```python
from mrm_trace.parser.perf_script_parser import PerfScriptParser

SAMPLE_OUTPUT = """
48823 llama-cli 12345678901 0x7f0000001000 ggml_vec_dot_q4_K /lib/libggml.so
48823 llama-cli 12345678950 0x7f0000002000 llama_kv_cache_update /lib/libllama.so
"""

def test_parses_valid_line():
    parser = PerfScriptParser()
    records = list(parser.parse_lines(SAMPLE_OUTPUT.strip().splitlines()))
    assert len(records) == 2
    assert records[0].address == 0x7f0000001000
    assert records[0].symbol == "ggml_vec_dot_q4_K"
    assert records[0].timestamp_ns == 12345678901

def test_skips_malformed_lines(caplog):
    parser = PerfScriptParser()
    records = list(parser.parse_lines(["this is garbage", "also bad"]))
    assert len(records) == 0
    assert "malformed" in caplog.text.lower()

def test_handles_empty_input():
    parser = PerfScriptParser()
    assert list(parser.parse_lines([])) == []

def test_streams_without_loading_all_into_memory():
    """Parser must be a generator — not accumulate all rows in RAM."""
    import inspect
    parser = PerfScriptParser()
    result = parser.parse_lines(["dummy"])
    assert inspect.isgenerator(result)
```


***

### `tests/unit/analyser/test_retention.py`

```python
from mrm_trace.analyser.retention import RetentionAnalyser
import pytest

def test_retention_known_value(synthetic_trace_factory):
    trace = synthetic_trace_factory(
        n_weight_reads=100, weight_retention_s=30.0
    )
    analyser = RetentionAnalyser()
    result = analyser.compute(trace)
    weight_row = result[result.region_type == "weight"].iloc[0]

    # Ground truth: retention should be ~30 seconds
    assert abs(weight_row["retention_p99_s"] - 30.0) < 0.5

def test_retention_never_negative(synthetic_trace_factory):
    trace = synthetic_trace_factory()
    analyser = RetentionAnalyser()
    result = analyser.compute(trace)
    assert (result["retention_p50_s"] >= 0).all()
    assert (result["retention_p99_s"] >= result["retention_p50_s"]).all()

def test_kv_cache_shorter_than_weights(synthetic_trace_factory):
    trace = synthetic_trace_factory(
        weight_retention_s=30.0, kv_retention_s=3.0
    )
    analyser = RetentionAnalyser()
    result = analyser.compute(trace)
    weight_ret = result[result.region_type=="weight"]["retention_p50_s"].values[0]
    kv_ret = result[result.region_type=="kv_cache"]["retention_p50_s"].values[0]
    assert weight_ret > kv_ret  # fundamental MRM assumption

def test_empty_trace_returns_empty_frame():
    import pandas as pd
    analyser = RetentionAnalyser()
    result = analyser.compute(pd.DataFrame())
    assert result.empty
```


***

### `tests/unit/analyser/test_write_once.py`

```python
from mrm_trace.analyser.write_once import WriteOnceAnalyser

def test_write_once_ratio_known_value(synthetic_trace_factory):
    # weights: 1 store, N reads → 100% write-once for weight regions
    trace = synthetic_trace_factory(n_weight_reads=1000, n_kv_writes=100)
    analyser = WriteOnceAnalyser()
    result = analyser.compute(trace)
    weight_row = result[result.region_type == "weight"].iloc[0]
    assert weight_row["write_once_ratio"] == pytest.approx(1.0)

def test_ratio_bounded_zero_to_one(synthetic_trace_factory):
    trace = synthetic_trace_factory()
    result = WriteOnceAnalyser().compute(trace)
    assert (result["write_once_ratio"] >= 0.0).all()
    assert (result["write_once_ratio"] <= 1.0).all()

def test_multi_write_region_not_write_once():
    import pandas as pd
    # A region written 5 times should NOT be write-once
    rows = [
        {"address": 0x1000, "op_type": "store", "region_type": "kv_cache",
         "region_id": 1, "timestamp_ns": i * 1000}
        for i in range(5)
    ]
    trace = pd.DataFrame(rows)
    result = WriteOnceAnalyser().compute(trace)
    assert result.iloc[0]["write_once_ratio"] == 0.0
```


***

## Tier 3: Validity Tests

These are unique to measurement tools and the most important tests in the suite:

```python
# tests/validity/test_ground_truth_metrics.py

def test_full_pipeline_known_input_known_output(synthetic_trace_factory):
    """
    End-to-end: a trace with precisely constructed properties must
    produce metrics that match ground truth within tolerance.
    This is the single most important test in the suite.
    """
    from mrm_trace.analyser.retention import RetentionAnalyser
    from mrm_trace.analyser.write_once import WriteOnceAnalyser

    trace = synthetic_trace_factory(
        n_weight_reads=1000,
        weight_retention_s=30.0,
        kv_retention_s=5.0,
        n_kv_writes=50,
        n_kv_reads=500
    )

    ret = RetentionAnalyser().compute(trace)
    wo = WriteOnceAnalyser().compute(trace)

    # Retention ground truth
    w = ret[ret.region_type == "weight"].iloc[0]
    kv = ret[ret.region_type == "kv_cache"].iloc[0]
    assert abs(w["retention_p99_s"] - 30.0) < 1.0,   "Weight retention wrong"
    assert abs(kv["retention_p99_s"] - 5.0) < 0.5,   "KV retention wrong"

    # Write-once ground truth
    w_wo = wo[wo.region_type == "weight"].iloc[0]
    assert w_wo["write_once_ratio"] == pytest.approx(1.0, abs=0.01)


# tests/validity/test_no_cross_run_leakage.py

def test_run_ids_isolated(tmp_path, synthetic_trace_factory):
    """Two runs must never share region_ids or addresses in merged output."""
    from mrm_trace.reporter.csv_exporter import CsvExporter

    trace_a = synthetic_trace_factory(seed=1)
    trace_b = synthetic_trace_factory(seed=2)
    trace_a["run_id"] = "run_a"
    trace_b["run_id"] = "run_b"

    exporter = CsvExporter(output_dir=tmp_path)
    exporter.write(trace_a, run_id="run_a")
    exporter.write(trace_b, run_id="run_b")

    merged = exporter.load_all()
    assert set(merged[merged.run_id == "run_a"]["region_id"]).isdisjoint(
           set(merged[merged.run_id == "run_b"]["region_id"])
    )


# tests/validity/test_thermal_invalidation.py

def test_throttled_run_marked_contaminated():
    from mrm_trace.telemetry.observer import ObserverEffectSummary
    summary = ObserverEffectSummary(
        observer_cpu_overhead_pct_max=5.0,
        observer_memory_overhead_mb_peak=200.0,
        thermal_throttle_events=3,       # ← throttled
        io_contention_detected=False
    )
    assert summary.run_validity == "contaminated"

def test_clean_run_marked_clean():
    from mrm_trace.telemetry.observer import ObserverEffectSummary
    summary = ObserverEffectSummary(
        observer_cpu_overhead_pct_max=3.1,
        observer_memory_overhead_mb_peak=180.0,
        thermal_throttle_events=0,
        io_contention_detected=False
    )
    assert summary.run_validity == "clean"
```


***

## Performance Tests

```python
# tests/performance/test_parser_throughput.py
import time, io
from mrm_trace.parser.perf_script_parser import PerfScriptParser

def test_parser_exceeds_500k_rows_per_second():
    # Generate 1M synthetic perf script lines
    line = "48823 llama-cli 12345678901 0x7f0000001000 ggml_vec_dot /lib/libggml.so\n"
    data = (line * 1_000_000).splitlines()

    parser = PerfScriptParser()
    start = time.perf_counter()
    count = sum(1 for _ in parser.parse_lines(data))
    elapsed = time.perf_counter() - start

    rate = count / elapsed
    assert rate >= 500_000, f"Parser too slow: {rate:.0f} rows/s"


# tests/performance/test_streaming_memory.py
import tracemalloc
from mrm_trace.parser.perf_script_parser import PerfScriptParser

def test_parser_ram_stays_under_2gb_for_50m_rows():
    line = "48823 llama-cli 12345678901 0x7f0000001000 ggml_vec_dot /lib/libggml.so\n"
    data = (line * 50_000_000).splitlines()

    tracemalloc.start()
    parser = PerfScriptParser()
    # consume generator without storing results
    for _ in parser.parse_lines(data):
        pass
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_gb = peak / 1e9
    assert peak_gb < 2.0, f"Parser used {peak_gb:.2f} GB — streaming broken"
```


***

## How to Add Them — Tooling Setup

```toml
# pyproject.toml additions

[project.optional-dependencies]
test = [
    "pytest>=8.0",
    "pytest-cov",           # coverage reports
    "pytest-xdist",         # parallel test execution
    "pytest-benchmark",     # performance test runner
    "pytest-mock",          # mock perf/memray subprocesses
    "hypothesis",           # property-based testing for schema validation
    "freezegun",            # freeze time for timestamp tests
]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "unit: fast, no I/O",
    "integration: may write temp files",
    "validity: scientific correctness assertions",
    "performance: throughput and memory benchmarks",
    "slow: requires model files on disk",
]
addopts = "-v --tb=short -m 'not slow'"   # exclude slow by default
```

```bash
# Run tiers selectively
pytest -m unit                  # fast, CI on every commit
pytest -m "unit or integration" # pre-merge
pytest -m validity              # before any dataset release
pytest -m performance           # weekly / before paper submission
pytest -m slow                  # only on machines with model files
```


***

## One Additional Technique: Property-Based Tests with Hypothesis

For the schema and metric bounds, Hypothesis generates adversarial inputs automatically — far better than hand-written edge cases:

```python
# tests/unit/analyser/test_retention_hypothesis.py
from hypothesis import given, strategies as st
from mrm_trace.analyser.retention import RetentionAnalyser
import pandas as pd

@given(st.lists(
    st.fixed_dictionaries({
        "timestamp_ns": st.integers(min_value=0, max_value=10**12),
        "address": st.integers(min_value=0x1000, max_value=0xFFFFFFFFFFFF),
        "op_type": st.sampled_from(["load", "store"]),
        "region_type": st.sampled_from(["weight", "kv_cache", "activation"]),
        "region_id": st.integers(min_value=1, max_value=100),
    }),
    min_size=1, max_size=10_000
))
def test_retention_always_non_negative_any_input(rows):
    trace = pd.DataFrame(rows)
    result = RetentionAnalyser().compute(trace)
    if not result.empty:
        assert (result["retention_p50_s"] >= 0).all()
        assert (result["retention_p99_s"] >= result["retention_p50_s"]).all()
```

Hypothesis will find timestamp orderings, duplicate addresses, and edge cases you would never think to write manually — and it shrinks failing cases to the minimal reproducer automatically.


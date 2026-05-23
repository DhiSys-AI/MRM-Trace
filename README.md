# mrm-trace

A Python research package for collecting, parsing, labelling, and analysing LLM inference
memory access traces. Designed as scientific instrumentation for
**Managed-Retention Memory (MRM)** research - it characterises how model weights, KV cache,
activations, and runtime allocations are actually accessed during inference.

**Primary metrics:** retention duration · write-once ratio · read frequency · working set size

---

## Install from PyPI

```bash
pip install mrm-trace
```

> Linux (or WSL2) is required for `perf mem` collection. See [Requirements](#requirements) below.

---

## Requirements

| Requirement | Notes |
|---|---|
| Linux (WSL2 supported) | `perf mem` requires Linux PMU; WSL2 works |
| Python ≥ 3.11 | Tested on 3.11 and 3.12 |
| sudo / CAP_PERFMON | Required for `perf mem` collection |

---

## Install

```bash
# Clone and set up a virtual environment
git clone https://github.com/DhiSys-AI/MRM-Trace
cd MRM-Trace
python -m venv venv
source venv/bin/activate      # Windows WSL: same command

# Install package + test dependencies
pip install -e ".[test]"

# Optional: install matplotlib/seaborn for figures
pip install -e ".[test,plots]"
```

---

## Quick start

```bash
# Validate a config file
mrm-trace validate --config config/default_experiment.yaml

# Preview what a run would do (dry run)
mrm-trace plan --config config/default_experiment.yaml

# Run a full experiment (requires model files + sudo for perf)
mrm-trace run --config config/default_experiment.yaml
```

---

## Notebooks

| Notebook | Description | Run on Colab |
|---|---|---|
| [001 - Getting Started](notebooks/001%20-%20Getting%20Started.ipynb) | Install, synthetic trace, label, analyse, export, schema versioning, validity | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DhiSys-AI/MRM-Trace/blob/main/notebooks/001%20-%20Getting%20Started.ipynb) |
| [002 - YAML Config & Experiment Planning](notebooks/002%20-%20YAML%20Config%20%26%20Experiment%20Planning.ipynb) | Write & validate configs, sweep expansion, multi-model runs, collector tuning | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DhiSys-AI/MRM-Trace/blob/main/notebooks/002%20-%20YAML%20Config%20%26%20Experiment%20Planning.ipynb) |
| [003 - Real Collection Walkthrough](notebooks/003%20-%20Real%20Collection%20Walkthrough.ipynb) | Real memray capture, parse raw trace, understand symbols, real-model guide | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DhiSys-AI/MRM-Trace/blob/main/notebooks/003%20-%20Real%20Collection%20Walkthrough.ipynb) |

All three notebooks run without root or model files (001 and 002 use synthetic data; 003 uses memray on a simulated workload). They are a good first stop for new contributors and researchers.

---

## Running tests

```bash
# Every commit - fast, no I/O
pytest -m unit

# Pre-merge - includes integration tests
pytest -m "unit or integration"

# Before dataset release - scientific correctness checks
pytest -m validity

# Property-based invariant tests (Hypothesis)
pytest tests/property/

# Performance benchmarks (excluded from default run)
pytest -m performance --benchmark-only

# Full suite (excludes slow + performance)
pytest
```

The test suite has three tiers:

| Tier | Marker | Purpose |
|---|---|---|
| 1 | `unit` | Individual functions behave correctly |
| 2 | `integration` | Components work together |
| 3 | `validity` | Measurements are scientifically sound |

Tier-3 validity tests are the most important: they verify that known synthetic inputs produce
known metric outputs (e.g. a 30s weight retention window must yield `retention_p99_s ≈ 30.0`).

---

## Output layout

Each run writes to `results/<model_id>/<run_id>/`:

```
results/llama-7b/run_20240101_120000/
├── trace.parquet                  ← labelled memory access trace
├── region_map.parquet             ← one row per region (weight, kv_cache, …)
├── kv_block_lifecycle.parquet     ← per-block write / read / eviction timestamps
├── metrics.csv                    ← per-region-type summary (human-readable)
├── metadata.json                  ← hardware, software, observer effect, run validity
├── manifest.json                  ← SHA-256 checksums for all files
└── raw/
    ├── perf.data
    ├── perf_script.txt
    └── memray.bin
```

---

## Run validity classification

Every run is automatically classified based on observer overhead:

| Class | Criteria |
|---|---|
| `clean` | observer CPU < 10 %, observer mem < 5 % of target RSS, no throttle, baseline CPU < 15 % |
| `marginal` | observer CPU < 20 %, observer mem < 15 % of target RSS, ≤ 2 throttle events |
| `contaminated` | anything worse than marginal |

Contaminated runs are archived but excluded from aggregated metrics and paper figures.

---

## Architecture

```
mrm_trace/
├── cli.py              CLI (typer)
├── api.py              Python API (Experiment class)
├── schema_version.py   Schema version registry and compatibility checking
├── engines/            llama.cpp / vLLM wrappers
├── collector/          perf mem / memray / process_monitor
├── parser/             perf script + memray parsers → trace.parquet
├── labeller/           symbol + address-range region classification
├── analyser/           retention / write-once / read-freq / working-set / IAI / suitability
├── telemetry/          baseline capture / thermal / observer effect / validity classifier
├── reporter/           CSV + Parquet export / figures / manifest / RunExporter
└── utils/              logging / IDs / file helpers
```

Key design decisions:
- **Streaming parser** - generators throughout; never loads full trace into RAM (ADR-2)
- **Phase-aware tracing** - `weight_load` / `generation` / `teardown` phases distinguish weight from KV (ADR-3)
- **Observer effect as mandatory output** - every run records overhead and validity class (ADR-4)
- **Parquet + zstd** - column-oriented, ~3× better compression than gzip (ADR-8)

---

## MRM suitability labels

| Label | Criteria |
|---|---|
| `high_mrm` | write-once ratio ≥ 0.8 **and** retention p99 ≥ 10 s |
| `medium_mrm` | write-once ratio ≥ 0.5 **and** retention p50 ≥ 1 s |
| `low_mrm` | everything else |

In practice: model weights → `high_mrm`, short-lived KV blocks → `low_mrm`.

---

## Schema versioning

All output files carry a `mrm_trace_schema_version` in their Parquet metadata.
The version registry is in `mrm_trace/schema_version.py`. Readers validate
major-version compatibility on load; a major bump is a breaking change.

```python
from mrm_trace.schema_version import check_parquet_schema
check_parquet_schema("results/.../trace.parquet", "trace")  # raises on incompatibility
```

---

## Python API

```python
from mrm_trace.labeller import TraceLabeller
from mrm_trace.analyser import compute_all
from mrm_trace.reporter import RunExporter

# Label a stream of raw trace rows
labeller = TraceLabeller()
labelled = list(labeller.label(raw_rows))
region_map   = labeller.region_map()    # call after consuming label()
kv_lifecycle = labeller.kv_lifecycle()

# Analyse
import pandas as pd
trace = pd.DataFrame(labelled)
results = compute_all(trace)
# results keys: retention_per_region, retention_summary, write_once,
#               read_freq, working_set_per_region, working_set_summary,
#               locality_per_region, locality_summary, iai, suitability

# Export a publication-ready run directory
exporter = RunExporter("results/llama-7b/run_001")
exporter.export(trace, region_map, kv_lifecycle, results,
                metadata={"run_id": "run_001"}, run_id="run_001")
```

---

## Collector hierarchy

1. `perf mem` - primary; requires Linux PMU + root/sudo; WSL2 supported
2. `memray` - fallback; Python-level allocations; no root needed
3. `process_monitor` - always runs in parallel as coarse baseline (psutil)

---

## Reporting issues and contact

- **Bug reports / feature requests:** [GitHub Issues](https://github.com/DhiSys-AI/MRM-Trace/issues)
- **Email:** [info@dhisys.co.uk](mailto:info@dhisys.co.uk)

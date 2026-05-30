# CLAUDE.md — mrm-trace Developer Guide

## Dev Environment
WSL  
Run full test suite via WSL  
`wsl -e bash -c "cd /mnt/c/proj/mrm-trace && source venv/bin/activate && python -m pytest --tb=short -q 2>&1"`  

## Project Summary

`mrm-trace` is a Python research package for collecting, parsing, labelling, and analysing
LLM inference memory access traces. It is designed as **scientific instrumentation** for
Managed-Retention Memory (MRM) research — the goal is to characterise how different memory
regions (model weights, KV cache, activations) are actually accessed during inference.

Primary metrics: retention duration, write-once ratio, read frequency, working set size.

## Environment

- **Runtime**: WSL2 Ubuntu 24.04 (kernel `5.15.167.4-microsoft-standard-WSL2`)
- **Python**: Python venv at `venv/` — activate with `source venv/bin/activate`
- **Install (dev)**: `pip install -e ".[test]"`
- **Run tests**: `pytest` (excludes `slow` tests by default)

### Quick start

```bash
source venv/bin/activate
pip install -e ".[test]"
pytest -m "unit or integration"
mrm-trace validate --config config/default_experiment.yaml
mrm-trace plan --config config/default_experiment.yaml
```

### Verify installed version and fixes

```bash
python -c "import mrm_trace; print(mrm_trace.__version__)"
python -c "import mrm_trace; print(mrm_trace.__file__)"
# should print the source tree path, not site-packages
```

If either fix below says STALE, run `pip install -e ".[test]"`:

```bash
python -c "
from mrm_trace.reporter.manifest import build_manifest
from mrm_trace.engines.llamacpp import LlamaCppEngine
import inspect
print('manifest fix:', 'OK' if \"not f.exists()\" in inspect.getsource(build_manifest) else 'STALE')
print('engine reset:', 'OK' if 'reset' in inspect.getsource(LlamaCppEngine._generate_via_bindings) else 'STALE')
"
```

## Three-Layer Data Model

```
Layer 1: OBSERVED DATA       → trace.parquet, region_map.parquet, kv_block_lifecycle.parquet
Layer 2: OBSERVER EFFECT     → metadata.json / observer_effect section
Layer 3: MACHINE CONTEXT     → metadata.json / hardware + software + baseline + thermal
```

Every run produces all three layers. Layer 2 and 3 are what allow trace values (latency,
retention) to be independently audited and compared across machines and papers.

## Output Layout per Run

```
results/<model_id>/<run_id>/
├── trace.parquet
├── region_map.parquet
├── kv_block_lifecycle.parquet
├── metrics.csv
├── metadata.json
├── manifest.json
├── logs/
│   ├── run.log
│   └── parser.log
└── raw/
    ├── perf.data
    ├── perf_script.txt
    └── memray.bin
```

## Architecture Overview

```
mrm_trace/
├── cli.py              CLI entry point (typer)
├── api.py              Python API (Experiment class)
├── config/             YAML schema + loader + validators
├── engines/            llama.cpp / vLLM wrappers (Phase 2)
├── collector/          perf mem / memray / process_monitor (Phase 3)
├── parser/             perf script + memray parsers (Phase 4)
├── labeller/           symbol + address-range region classification (Phase 5)
├── analyser/           retention / write-once / read-freq / working-set (Phase 6)
├── telemetry/          hardware / software / thermal / observer effect (Phase 7)
├── reporter/           CSV/Parquet export + manifest + plots (Phase 8)
└── utils/              logging / IDs / file helpers
```

## Phase Plan

| Phase | Deliverables | Acceptance Criteria |
|-------|-------------|---------------------|
| **1 — Foundations** (done) | Scaffolding, schemas, YAML loader, logging, CLI skeleton, manifest | `validate` + `plan` commands work |
| **2 — Engine adapters** (done) | llama.cpp + vLLM wrappers, phase events, PID discovery | Both engines run a prompt |
| **3 — Collectors** (done) | perf mem runner, memray runner, process monitor | Raw artifacts + telemetry snapshots produced |
| **4 — Parsing** (done) | perf script parser, memray parser, canonical schema output | Synthetic raw data → trace.parquet |
| **5 — Labelling** (done) | Symbol labeller, address tracker, region map, KV lifecycle | Test fixtures classify weight + KV regions |
| **6 — Analysis** (done) | Retention, write-once, read-freq, working-set, locality, IAI, suitability | Synthetic traces yield expected metrics |
| **7 — Telemetry + validity** (done) | Baseline capture, thermal, observer effect sampling, validity classifier | Contaminated/clean fixture runs classify correctly |
| **8 — Reporting + export** (done) | CSV/Parquet exports, figures, dataset manifest, export helpers | Run dirs are publication-ready |
| **9 — Hardening** (done) | Full tests, perf tuning, schema versioning, docs | CI passes, ready for internal pilot |

## Testing Philosophy

This is a **measurement system** — tests have three tiers:

```
Tier 1: Unit         — individual functions behave correctly (fast, no I/O)
Tier 2: Integration  — components work together (may write temp files)
Tier 3: Validity     — measurements are scientifically sound (unique to this project)
```

Run selectively:
```bash
pytest -m unit                   # every commit
pytest -m "unit or integration"  # pre-merge
pytest -m validity               # before dataset release
pytest -m performance            # weekly / before paper submission
pytest -m slow                   # only on machines with model files
```

The Tier 3 validity tests are **the most important tests in the suite**. They verify that
known synthetic inputs produce known metric outputs — e.g. a trace with a 30s weight
retention window must produce `retention_p99_s ≈ 30.0`.

## Key Design Decisions (ADRs)

### ADR-1: `perf mem` as primary collector
Hardware PMU-assisted collection captures load latency and NUMA node at hardware speed.
Memray and process_monitor are fallbacks for environments without PMU (VMs, containers).

### ADR-2: Streaming parser — never load full trace into RAM
A 70B model inference run can produce 50+ GB of `perf.data`. The parser must be a
generator that streams rows; it must never accumulate all rows in memory.

### ADR-3: Phase-aware tracing
Without phase tagging (`weight_load`, `generation`, `teardown`), weight and KV cache
accesses are indistinguishable by address alone. Engine wrappers emit phase boundaries.

### ADR-4: Observer effect as mandatory output
The tracer's own CPU/memory overhead disturbs the measurement. Every run records observer
overhead, and runs are classified `clean / marginal / contaminated` based on it.

### ADR-5: CLI framework — `typer`
Chosen over `argparse` for auto-generated help, type-checked arguments, and Rich integration.

### ADR-6: Schema-first with Pydantic v2
All internal data objects have Pydantic models. The YAML schema is validated at load time;
invalid configs are rejected before any work starts.

### ADR-7: Output directory layout is deterministic
Layout: `results/<model_id>/<run_id>/`. Run ID encodes model, context, batch, tokens, rep,
and timestamp — making results self-describing and sortable.

### ADR-8: Parquet with zstd compression
Column-oriented storage handles billions of trace rows efficiently. zstd gives ~3x better
compression than gzip at higher speed. Compatible with HuggingFace Datasets for release.

## Run Validity Classification

| Class | Criteria |
|-------|----------|
| `clean` | observer CPU < 10%, observer mem < 5% of target RSS, no throttle, baseline CPU < 15% |
| `marginal` | observer CPU < 20%, observer mem < 15% of target RSS, ≤ 2 throttle events |
| `contaminated` | anything worse than marginal |

Contaminated runs are archived but excluded from aggregated metrics and paper figures.

## Collector Hierarchy

```
1. perf mem   — primary, requires Linux PMU + root; DOES NOT WORK on WSL2
               (Microsoft WSL2 kernel does not expose hardware PMU counters)
2. memray     — correct collector for WSL2
               non-root: Python-level only, region_map empty, timestamps = 0
               root + native_traces=True: C-level alloc events, timestamps still 0
               region_map STILL EMPTY with pip-installed llama-cpp-python (C symbols stripped)
               region_map populated only with debug-symbol build of llama-cpp-python
3. process_monitor — always runs in parallel as coarse RSS/CPU baseline (psutil)
```

### What each collector gives you

| Capability | memray (non-root) | memray (root, native) | perf mem |
|---|---|---|---|
| Allocation events | Python-level only | Python + C-level | load/store events |
| `region_map` populated | No (all unknown) | No† (pip strips symbols) | Yes |
| `timestamp_ns` | Always 0 | Always 0 | Nanosecond HW timestamps |
| Retention duration | Always 0 | Always 0 | Real values |
| Cache level (L1/L2/L3) | No | No | Yes |
| Works on WSL2 | Yes | Yes (sudo) | No |
| Works on bare-metal | Yes | Yes | Yes |

† pip-installed `llama-cpp-python` strips C debug symbols. `ggml_*` names won't appear even with `native_traces=True`. Use `CMAKE_BUILD_TYPE=Debug pip install llama-cpp-python` for a debug build, or use `perf mem` on bare-metal.

### WSL2 limitation — perf mem does not work

The development machine runs `5.15.167.4-microsoft-standard-WSL2`. This kernel does not
forward hardware PMU counters to the guest. `perf mem` will not produce data. Even after
installing `linux-tools-common linux-tools-virtual`, the PMU events
(`cpu/mem-loads,ldlat=20/P`) are unsupported.

**For WSL2 development:** use `demo_tinyllama.py --native-traces` (with sudo).
**For publication data:** run on bare-metal Linux with `demo_perf_mem.py`.

### Live demo scripts (`notebooks/scripts/`)

| Script | Collector | Root needed | Region map | Timestamps |
|---|---|---|---|---|
| `demo_tinyllama.py` | memray | No (or --native-traces) | Empty (pip strips C symbols) | Always 0 |
| `demo_qwen_hf.py` | memray | No (or --native-traces) | Empty (PyTorch internals, no ggml_ names) | Always 0 |
| `demo_perf_mem.py` | perf mem | Yes | Populated | Nanoseconds |

```bash
# WSL2 — best available (C-level symbols, populated region_map)
sudo -E python notebooks/scripts/demo_tinyllama.py \
  --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --native-traces

# WSL2 — no root (Python-level only, region_map empty)
python notebooks/scripts/demo_tinyllama.py \
  --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# WSL2 — Qwen (no GGUF needed, auto-downloads ~1 GB)
pip install transformers torch accelerate
python notebooks/scripts/demo_qwen_hf.py

# Bare-metal Linux only
sudo -E python notebooks/scripts/demo_perf_mem.py \
  --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

Model download (one-time):

```bash
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
mkdir -p models
wget -P models/ https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

## Important Constraints

- Collector commands are built from validated arguments only — no shell interpolation of user config values
- Sensitive env vars must not be logged
- If disk space is low, fail safely and mark run incomplete
- If parse fails, retain raw collector files untouched
- Unknown symbol policy is configurable; never silently coerce unknowns

## Canonical Trace Schema (trace.parquet)

Key columns: `run_id, sample_id, timestamp_ns, phase, address, address_page, op_type,
symbol, dso, region_type, region_id, access_latency_ns, cache_level, numa_node, cpu_id,
thread_id, token_index, context_tokens`

## Dependencies

Core: `pydantic>=2`, `pyyaml`, `typer`, `rich`, `pandas`, `pyarrow`, `psutil`, `numpy`
Test: `pytest`, `pytest-cov`, `pytest-xdist`, `pytest-benchmark`, `pytest-mock`, `hypothesis`, `freezegun`
Optional: `matplotlib`, `seaborn` (Phase 8 plots)

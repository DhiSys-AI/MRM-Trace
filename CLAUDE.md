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
| `clean` | observer CPU < 10%, observer memory within the clean bound, no throttle, baseline CPU < 15%, peak temp ≤ 85°C |
| `marginal` | observer CPU < 20%, observer memory within the marginal bound, ≤ 2 throttle events, peak temp ≤ 90°C |
| `contaminated` | anything worse than marginal |

The observer memory bound depends on where the collector runs, because the two
cases perturb the target by different mechanisms:

| Collector | Bound | clean | marginal |
|---|---|---|---|
| memray (in-process) | observer RSS / **target** RSS | < 0.05 | < 0.15 |
| perf mem (out-of-process) | observer RSS / **total system** RAM | < 0.10 | < 0.25 |

An out-of-process collector's footprint comes from CPU count, ring-buffer size
and flush rate — not from model size — so a ratio to target RSS measures
nothing. Measured: perf peaked at 2512 MB against a 1478 MB target (ratio 1.70)
on a 31 GB machine with ~27 GB free; nothing was evicted or swapped, yet the
ratio rule marked 8 of 8 runs contaminated, and would have marked all 151.

Validity is a **label over numbers each run already recorded**
(`metadata.json → observer_effect` + `thermal`), so correcting the rule never
requires re-collecting a trace: `experiment_runner.py --reclassify` re-derives
every verdict in place. Both a fresh run and a reclassification go through
`classify_from_metadata()` so they cannot disagree. Every verdict carries
`thermal.validity_failure_reasons` stating what failed.

Contaminated runs are archived but excluded from aggregated metrics and paper figures.

## Collector Hierarchy

```
1. perf mem   — primary, requires Linux PMU + root
               WORKS: bare-metal Linux (x86-64, Intel/AMD with PEBS/IBS support)
               DOES NOT WORK: WSL2 (Microsoft kernel does not forward PMU counters)
               MODE: spawn (perf launches the Python process); attach mode (-p PID) fails
                     on Ubuntu 24.04 / kernel 6.8 for PEBS load-latency events
               EVENTS: auto-selected by perf (no explicit -e flags; ldlat syntax
                       varies across kernel versions and is not universally available)
2. memray     — fallback for WSL2, non-root, or when PMU probe fails
               non-root: Python-level only, region_map empty, timestamps = 0
               root + native_traces=True: C-level alloc events, timestamps still 0
               region_map STILL EMPTY with pip-installed llama-cpp-python (C symbols stripped)
               region_map populated only with debug-symbol build of llama-cpp-python
3. process_monitor — always runs in parallel as coarse RSS/CPU baseline (psutil)
```

### What each collector gives you

| Capability | memray (non-root) | memray (root, native) | perf mem |
|---|---|---|---|
| Allocation events | Python-level only | Python + C-level | load/store events only |
| `region_map` populated | No | No† (pip strips symbols) | Only with llama.cpp + debug build |
| `timestamp_ns` | Always 0 | Always 0 | Nanosecond HW timestamps |
| Retention duration | Always 0 | Always 0 | Real values |
| Cache level (L1/L2/L3) | No | No | Yes |
| Works on WSL2 | Yes | Yes (sudo) | No |
| Works on bare-metal Linux | Yes | Yes | Yes |
| Works on Google Colab / VMs | Yes | Partial | No |

† pip-installed `llama-cpp-python` strips C debug symbols. `ggml_*` names won't appear even
with `native_traces=True`. Use `CMAKE_BUILD_TYPE=Debug pip install llama-cpp-python` for a
debug build, or use `perf mem` on bare-metal.

### Why region_map is empty for PyTorch/transformers models

This is a fundamental limitation, not a bug.

The labeller classifies addresses two ways:
1. **Symbol-based**: regex rules match `ggml_*` → weight, `llama_kv_cache_*` → kv_cache.
   These symbols come from llama.cpp's C library. PyTorch symbols are
   `at::native::mm_cpu_kernel`, `torch::autograd::...` — none match.
2. **Address-range tracking**: allocation events from memray (malloc/free pairs) let the
   labeller track which region each address belongs to. `perf mem` captures load/store
   events only — no allocation events, so the address tracker has nothing to work with.

**Impact on MRM metrics**: `region_type = unknown` for all records. Retention and write-once
metrics are computed correctly but cannot be separated by region type. You cannot answer
"do weights have higher retention than KV blocks?" from this data alone.

**Roadmap fix**: Before perf collection, enumerate `model.named_parameters()` to get each
weight tensor's `data_ptr()` and size, and inject these as pre-classified address ranges into
the labeller. This will classify weight regions without needing debug symbols.

**For publication data with classified regions**: use `demo_perf_mem.py` with a GGUF model
and `CMAKE_BUILD_TYPE=Debug pip install llama-cpp-python`. The `ggml_*` symbols will then be
visible in perf script output and the labeller rules will classify regions correctly.

### WSL2 limitation — perf mem does not work

The Microsoft WSL2 kernel (`*-microsoft-standard-WSL2`) does not forward hardware PMU
counters to the guest. `perf mem` will not produce data on any WSL2 kernel.

**For WSL2 development:** use `demo_tinyllama.py --native-traces` (with sudo).
**For publication data:** run on bare-metal Linux with `demo_perf_mem.py`.

### perf mem attach vs spawn mode

`perf mem record -p PID` (attach to running process) fails on Ubuntu 24.04 / kernel 6.8
with `"Failed to collect cpu/mem-loads,ldlat=30/P ... No such file or directory"` even as
root. Intel PEBS load-latency events require the target process to start under perf.

**Fix**: `demo_qwen_hf.py` uses **spawn mode** — it re-invokes itself with `--headless`
under `perf mem record -- python script.py --headless`. The full process lifetime (model
load → generation) is instrumented from the first instruction.

### sudo and venv

`sudo -E python` resolves to the **system Python**, not the venv.
Always use the full venv path:
```bash
sudo -E venv/bin/python notebooks/scripts/demo_qwen_hf.py
```

### Live demo scripts (`notebooks/scripts/`)

| Script | Collector | Root needed | Region map | Timestamps |
|---|---|---|---|---|
| `demo_tinyllama.py` | memray | No (or --native-traces) | Empty (pip strips C symbols) | Always 0 |
| `demo_qwen_hf.py` | **auto-detect** | No (memray) / Yes (perf mem) | Empty‡ | 0 (memray) / ns (perf mem) |
| `demo_perf_mem.py` | perf mem | Yes | Populated (with debug llama.cpp) | Nanoseconds |

‡ PyTorch backend: region_map empty regardless of collector. See "Why region_map is empty" above.

`demo_qwen_hf.py` **auto-selects its collector** at startup:
- Bare-metal Linux + root + PMU probe passes → **perf mem** (spawn mode, real timestamps)
- WSL2 / no root / PMU probe fails → **memray** (Python-level allocations)

Override with `--collector memray` or `--collector perf-mem`.

```bash
# WSL2 — best available (C-level allocs, still no region_map without debug symbols)
sudo -E venv/bin/python notebooks/scripts/demo_tinyllama.py \
  --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --native-traces

# WSL2 — no root (Python-level only)
python notebooks/scripts/demo_tinyllama.py \
  --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Qwen — WSL2 or non-root (auto → memray, timestamps=0)
pip install transformers torch accelerate
python notebooks/scripts/demo_qwen_hf.py

# Qwen — bare-metal Linux as root (auto → perf mem spawn mode, real timestamps)
sudo -E venv/bin/python notebooks/scripts/demo_qwen_hf.py

# TinyLlama — bare-metal Linux, perf mem + GGUF + classified regions
sudo -E venv/bin/python notebooks/scripts/demo_perf_mem.py \
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

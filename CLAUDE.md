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

- **Runtime**: WSL2 (Linux kernel) — required for `perf mem` (PMU-based collection)
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
1. perf mem   — primary, requires Linux PMU + root/sudo, WSL2 supported
2. memray     — fallback, Python-level allocations, no root needed
3. process_monitor — always runs in parallel as coarse baseline (psutil)
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

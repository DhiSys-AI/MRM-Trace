# MRM-Trace: Dataset & Publication Plan

*Last updated: 2026-06-28*

---

## Competitive Landscape and Window

No MRM simulator and no MRM-framed retention trace study has been published as of 2026-06-28. The idea is starting to leak into adjacent work - an edge-NPU paper is now doing "layer retention analysis" with a deliberately refresh-less workspace for transient mantissas. That is the same conceptual territory arriving from a different door. The gap is real and still open.

**The window is approximately 12–18 months** before this space gets crowded. When one group notices a gap, usually two or three have. A well-resourced group with GPU access could produce HBM data faster, but:
- They would need to build the collection + labelling tooling that is already done here
- HBM trace collection has its own infrastructure problems (no `perf mem` equivalent, proprietary PMU tooling)
- First-mover on the *open dataset + open tool* framing has durable value even if a closed-infrastructure GPU paper appears later

**The adjacent NPU paper helps rather than hurts.** It validates the research direction and gives a reviewer who might ask "why does this matter?" a ready answer: independent convergence from a different architecture confirms the question is real. Cite it as corroboration.

**The hard part is already done.** Building a working trace collection, parsing, labelling, and analysis pipeline for LLM inference is what stops most researchers. The MRM authors published the *idea* 18+ months ago with no dataset release. That silence is the opening.

### Critical path to filling the gap

The bottleneck is not code - it is moving from "code complete" to "data published." An arXiv preprint with results from 2 models, 3 context lengths, honest proxy caveat, and the open tool linked establishes priority even before the full sweep finishes. Reviewers cite the preprint date.

**Sequence that matters:**
1. Verify PMU works on this machine (15 minutes)
2. Confirm region fix on TinyLlama (30 minutes)
3. Download 2–3 more GGUF models (1 hour)
4. Start overnight sweep runs
5. **arXiv preprint with initial results** - this is the priority claim

---

## Research Tracks

Two separate papers from this codebase:

| Track | Title | Status |
|---|---|---|
| **A** (original) | LLM inference memory characterisation → MRM silicon contribution + Kaggle dataset | Code complete (Phase 9). Needs measurement sweep. |
| **B** (new) | Retention-Aware Memory Placement for Batch LLM Inference on Reused Hardware | See `review-discuss.md`. Needs energy measurement + crossover model. |

Track B layers on top of Track A - it does not break or replace it.

---

## Track A: Original MRM-Trace Goal

### What the codebase produces (already built)

Per-run derived artifacts in `results/<model_id>/<run_id>/`:

- `metrics.csv` - retention p50/p90/p99, write-once ratio, read frequency, working-set size, by region type
- `region_map.parquet` - address → region type mapping (weight / kv_cache / activation / unknown)
- `kv_block_lifecycle.parquet` - per-KV-block birth/death/reuse events
- `trace.parquet` (zstd) - canonical trace schema with timestamps, addresses, cache levels, NUMA

### Value proposition for MRM researchers

The three metrics Microsoft's MRM paper calls for but does not measure:
1. **Retention duration** - how long each region must stay valid under a batch schedule
2. **Write-once ratio** - fraction of bytes written exactly once then only read
3. **Read frequency** - per region type, as a function of model size / context / batch

These are *logical* properties of the inference algorithm, not the memory substrate. A CPU PEBS trace captures them faithfully. Frame the dataset as "logical memory-access characterisation - first-order evidence for MRM design, not a hardware-timing model." Put the proxy caveat in the data card.

---

## Hardware Assessment (this machine)

| Requirement | Status |
|---|---|
| Bare-metal Linux (not WSL2) | ✓ kernel `6.8.0-107-generic` |
| Intel PEBS-capable CPU | ✓ i5-8300H (Coffee Lake) |
| RAPL power measurement | ✓ `/sys/class/powercap/intel-rapl/` |
| `perf` binary | ✓ `/usr/bin/perf` |
| Root for collection | ✓ already using `sudo -E venv/bin/python` |
| Disk space | ✓ 267 GB free |
| RAM | ✓ 31 GB (fits 1B–7B Q4 models) |

**`perf_event_paranoid = 4`** - Ubuntu 24.04 default. Does NOT block collection. Root overrides it. Every collection run needs `sudo`, which the demo scripts already require.

---

## Steps to a Publishable Dataset

### Step 1 - Verify PMU produces real samples (15 min, do first)

```bash
sudo perf mem record -- sleep 2
sudo perf mem report --stdio | head -20
```

Must show non-zero samples with addresses. If "0 samples", stop - the PMU isn't working and no further collection is possible on this machine.

**CONFIRMED on dev machine (2026-06-28):** PMU is working. `perf mem record -- sleep 2` produced 3 samples on `cpu/mem-loads,ldlat=30/P` (Intel PEBS confirmed active). The partner machine's Tiger Lake PMU v5 is stronger and will produce denser samples. This machine is ready for collection.

### Step 2 - Verify TinyLlama region fix

The KV lifecycle and region map were previously zero for llama-cpp-python backends. Fix implemented (2026-06-28) using `/proc/self/maps` before/after diffing.

```bash
sudo -E venv/bin/python notebooks/scripts/demo_tinyllama.py \
  --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

Expected output:
```
Extracted N weight + M KV cache memory regions
Pre-registered N weight + M KV regions → K pages
Region map     : (N, 8)   ← must be non-zero
KV lifecycle   : (N, 8)   ← must be non-zero
```

### Step 3 - Download models for the sweep

Currently available: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`

Need for a useful dataset:

| Model | Format | Size (Q4_K_M) | Why |
|---|---|---|---|
| Llama-3.2-3B | GGUF | ~2 GB | Small model, strong baseline |
| Phi-3.5-mini (3.8B) | GGUF | ~2.2 GB | Efficient architecture, different access pattern |
| Mistral-7B | GGUF | ~4.1 GB | Mid-size, widely cited |
| Small MoE (optional) | GGUF | varies | Exposes expert-routing access patterns |

Model size range (1B → 7B) is what makes the dataset useful - shows how metrics scale.

Download from HuggingFace TheBloke or bartowski GGUF repos:
```bash
mkdir -p models
# example
wget -P models/ https://huggingface.co/TheBloke/<model>/resolve/main/<file>.gguf
```

### Step 4 - Measurement sweep

Matrix: models × context lengths × batch sizes × 2–3 reps

| Dimension | Values |
|---|---|
| Models | TinyLlama-1.1B, Llama-3.2-3B, Phi-3.5-mini, Mistral-7B |
| Context lengths | 2K, 8K, 32K |
| Batch sizes | 1, 4, 8 |
| Reps | 2–3 (discard 1 warmup) |

Estimated time: 3–4 weeks of actual runs on this hardware.

Engine: `llama.cpp` (CPU path, reliable on this hardware). `vLLM` as secondary if GPU available.

### Step 5 - Kaggle packaging

**What to publish on Kaggle** (derived artifacts only - raw perf.data is too large):
- `metrics.csv` - the headline numbers, one row per run
- `region_map.parquet` - address classification per run
- `kv_block_lifecycle.parquet` - KV block events per run
- `trace.parquet` - *sampled* (e.g. 10% random sample, zstd compressed)
- `metadata.json` per run - hardware, software, config, phase timestamps

**What NOT to publish on Kaggle**:
- `raw/perf.data` - multi-GB per run, Kaggle isn't the right home
- `raw/perf_script.txt` - intermediate artifact, not the dataset

**For bulk archival with DOI** (if needed for citation): Zenodo. Kaggle for discoverability.

**Data card must include**:
- Hardware spec (CPU model, RAM, storage)
- Collector used (perf mem, kernel version, perf version)
- Proxy caveat: "Retention durations are *observed lifetimes under a batch schedule* on this CPU's cache hierarchy, not hardware retention requirements. The logical access-pattern structure (write-once ratio, read frequency ordering) transfers across substrates."
- Schema documentation for each parquet file
- Citation for Microsoft MRM paper (Legtchenko et al., 2025)

---

## Codebase Status

| Phase | Deliverable | Status |
|---|---|---|
| 1 | Foundations, YAML loader, CLI | Done |
| 2 | Engine adapters (llama.cpp, vLLM) | Done |
| 3 | Collectors (perf mem, memray, process monitor) | Done |
| 4 | Parsers (perf script, memray) | Done |
| 5 | Labeller, region map, KV lifecycle | Done - KV fix for llama.cpp 2026-06-28 |
| 6 | Analyser (retention, write-once, read-freq, working-set) | Done |
| 7 | Telemetry + validity classifier | Done |
| 8 | Reporting + export (CSV/Parquet, figures, manifest) | Done |
| 9 | Hardening (full tests, schema versioning) | Done - 603 tests passing |
| - | PMU validation in `mrm-trace validate` | Pending |
| - | Actual measurement sweep | Pending (needs Step 1–3 above) |
| - | Kaggle dataset packaging | Pending (needs sweep) |

---

## Pending Code Work (small, before sweep)

1. **Extend `mrm-trace validate` with PMU checks** - detect `perf_event_paranoid`, CPU PMU type (Intel PEBS/AMD IBS/ARM SPE), run a live `perf mem record` probe and report go/no-go. Researchers using the tool need this on day one.

2. **Config templates for model matrix** - YAML experiment configs for each model in the sweep, with the 2K/8K/32K × 1/4/8 sweep pre-configured.

Neither is required before running Step 1 and 2 above.

## Full Experiment Guide

See [experiment-workplan.md](experiment-workplan.md) for the complete partner machine reproduction guide, including:
- Ubuntu installation steps
- All 6 GGUF models with exact download commands
- Full experiment matrix (186 runs: 162 CPU + 24 GPU)
- Thermal protocol (mandatory for no-AC setup on Alienware x17)
- The experiment runner script (`notebooks/scripts/experiment_runner.py`) with resume, retry, and thermal management

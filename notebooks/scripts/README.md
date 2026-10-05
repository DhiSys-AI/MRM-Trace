# mrm-trace — Live Demo Scripts

These scripts run **real inference** against small models and capture full mrm-trace output:
`trace.parquet`, `region_map.parquet`, `kv_block_lifecycle.parquet`, `metrics.csv`, `metadata.json`.

---

## Scripts

| Script | Collector | Root needed | Timestamps | Cache level | Region map |
|---|---|---|---|---|---|
| `demo_tinyllama.py` | memray | No (or `--native-traces`) | Always 0 | No | Empty |
| `demo_qwen_hf.py` | auto-detect | No (memray) / Yes (perf mem) | 0 or ns | perf mem only | Empty (PyTorch) |
| `demo_perf_mem.py` | perf mem | Yes | Nanoseconds | L1/L2/L3/DRAM | Populated (GGUF) |

**For publication-quality data use `demo_perf_mem.py` on bare-metal Linux.** It is the only
script that produces real hardware timestamps and a populated cache hierarchy breakdown.

---

## Setup

### 1. Install the package (from repo root)

```bash
# Linux / WSL2
cd /mnt/c/proj/mrm-trace
source venv/bin/activate
pip install -e ".[test]"
pip install memray
```

### 2a. TinyLlama — download model

```bash
mkdir -p notebooks/scripts/models
wget -P notebooks/scripts/models/ \
  https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
pip install llama-cpp-python
```

### 2b. Qwen — install transformers (model downloads automatically)

```bash
pip install transformers torch accelerate
```

---

## Running

```bash
# From the repo root
source venv/bin/activate

# TinyLlama — memray (Python-level only, no timestamps)
python notebooks/scripts/demo_tinyllama.py \
  --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Qwen2.5-0.5B — auto-detects collector
python notebooks/scripts/demo_qwen_hf.py

# TinyLlama — perf mem (bare-metal Linux, root, real timestamps + cache levels)
# IMPORTANT: sudo -E python resolves to system Python — always use the full venv path
sudo -E venv/bin/python notebooks/scripts/demo_perf_mem.py \
  --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# More output tokens (slower, richer KV trace)
python notebooks/scripts/demo_qwen_hf.py --tokens 256
```

Each run prints an inline quality gate immediately after parsing:

```
  Trace quality gate [perf_mem] — PASS
    Rows parsed          : 342,162
    access_latency_ns    : [████████████████████] 100.0%
    cache_level          : [████████████████████] 100.0%
    symbol (non-empty)   : [░░░░░░░░░░░░░░░░░░░░]   0.0%
    region classified    : [░░░░░░░░░░░░░░░░░░░░]   0.0%
```

A `FAIL` status exits immediately before export so bad runs are never written to disk.

---

## Output

Each run writes to `results/<model_id>/<run_id>/`:

```
results/tinyllama-1.1b-q4km/
└── tinyllama-1.1b-q4km_ctx2048_batch1_out128_rep0_<ts>/
    ├── trace.parquet
    ├── region_map.parquet
    ├── kv_block_lifecycle.parquet
    ├── metrics.csv
    ├── metadata.json
    ├── manifest.json
    ├── retention_histogram.png   (if matplotlib installed)
    ├── working_set.png
    └── raw/
        └── memray.bin
```

---

## Expected output (Qwen2.5-0.5B, ~40s on CPU)

```
mrm-trace live demo — Qwen2.5-0.5B-Instruct
  Context    : 2048 tokens
  Output     : results/qwen2.5-0.5b-instruct/...
  native_traces: True

Loading model (Qwen/Qwen2.5-0.5B-Instruct) ...
  [1/5] generating ... 128 tokens
  [2/5] generating ... 128 tokens
  ...
Parsed 2,847 allocation records
  Labelled trace : (2847, 18)
  Region map     : (0, 0)      ← expected for Python-level trace (no ggml_ symbols)
  Suitability    : insufficient_data
Done. Written to results/qwen2.5-0.5b-instruct/<run_id>/
```

The `region_map` is empty because `memray` captures Python/C-extension allocations —
real LLM symbol names (`ggml_init`, `llama_kv_cache_update`) only appear with llama-cpp-python
at `native_traces=True` (root) or when traced at the C level with `perf mem`.

---


### Packaging for Kaggle

`prepare_kaggle_dataset.py` turns a finished sweep in `results/` into a Kaggle-ready folder
under `datasets/` (git-ignored): stacked tables, per-run metadata and prompts, stratified
trace subsamples, a generated data card (`README.md`), `dataset-metadata.json` and
`SHA256SUMS`. `raw/perf.data` is not included. The username and home paths are rewritten,
and the run fails if either survives anywhere. What is included and why:
`context-files/06 - Kaggle Dataset Selection.md`.

```bash
set -a; . ./.env; set +a; export KAGGLE_API_TOKEN="$KAGGLE_TOKEN"
venv/bin/python notebooks/scripts/prepare_kaggle_dataset.py --force
# -t keeps trace.parquet as Parquet; without it Kaggle converts it to CSV
venv/bin/kaggle datasets version -p datasets/mrm-trace-kaggle -t -m "<notes>"
venv/bin/kaggle datasets metadata --update -p datasets/mrm-trace-kaggle \
  "$KAGGLE_USERNAME/mrm-trace-llm-memory-access-traces"
```

## Tips

- Run as root (`sudo`) or in Colab to enable `native_traces=True`, which captures C-level
  numpy/torch allocations and gives more meaningful symbol data.
- Increase `--tokens` for a richer generation trace (more KV allocations).
- Pass `--ctx 512` for a faster run.

# mrm-trace — Live Demo Scripts

These scripts run **real inference** against small models and capture full mrm-trace output:
`trace.parquet`, `region_map.parquet`, `kv_block_lifecycle.parquet`, `metrics.csv`, `metadata.json`.

---

## Scripts

| Script | Model | Backend | Notes |
|---|---|---|---|
| `demo_tinyllama.py` | TinyLlama-1.1B Q4_K_M | llama-cpp-python | Needs GGUF download |
| `demo_qwen_hf.py` | Qwen2.5-0.5B-Instruct | transformers | Auto-downloads from HF |

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
cd /mnt/c/proj/mrm-trace/notebooks/scripts

# TinyLlama
python demo_tinyllama.py --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Qwen2.5-0.5B
python demo_qwen_hf.py

# Custom output directory
python demo_tinyllama.py --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --out /tmp/mrm_results

# More output tokens (slower, richer KV trace)
python demo_qwen_hf.py --tokens 256
```

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

## Tips

- Run as root (`sudo`) or in Colab to enable `native_traces=True`, which captures C-level
  numpy/torch allocations and gives more meaningful symbol data.
- Increase `--tokens` for a richer generation trace (more KV allocations).
- Pass `--ctx 512` for a faster run.

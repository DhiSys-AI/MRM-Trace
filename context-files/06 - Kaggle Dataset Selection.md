# Kaggle Dataset Selection

*Which parts of `results/` go into the Kaggle release, and why — one sentence each.
Based on the occupancy sweep of 2026-10-04/05: 30 runs, 5 GGUF models under llama.cpp
on CPU (Llama 3.2 1B, Llama 3.2 3B, Phi-3.5 mini, Mistral 7B, Llama 3.1 8B), output
tokens 512 / 1024 / 1977, 2 repetitions each, context 2048, all 30 `clean`. The
results folder is 16 GB, 14 GB of which is `raw/perf.data`.*

The packaging script is `notebooks/scripts/prepare_kaggle_dataset.py`; it writes to
`datasets/` (git-ignored).

---

## Must include (small, high value)

| Item | Why it matters |
|---|---|
| `results/manifest.csv` | The table of contents: one row per run with its model, settings, validity, temperature and speed, so users can find and filter runs before opening anything else. |
| `results/analysis/region_contrast.csv` | The headline result, comparing how long weights, KV cache and runtime memory keep their data across all 30 runs. |
| `results/analysis/model_scaling.csv` | Shows how memory retention changes as models get bigger (1B to 8B parameters). |
| `results/analysis/context_scaling.csv` | Shows how memory retention changes as the model generates more tokens (512 to 1977). |
| `results/analysis/kv_lifecycle_blocks.csv` | The pooled life story of about 90,000 KV-cache blocks; most have no observed write, so their lifetime is null. |
| per-run `metrics.csv` | The key numbers for each memory region in one run: retention, write-once ratio, working set and MRM suitability. |
| per-run `metadata.json` | Records the machine, CPU frequency, temperature, software versions and the tracer's own overhead, so anyone can check whether a run was trustworthy. |
| per-run `region_map.parquet` | Says which address ranges are weights, KV cache or runtime memory, which is what lets the trace be split by memory type. |
| per-run `kv_block_lifecycle.parquet` | The per-run KV block table: when each block was written, last read and evicted. |
| per-run `prompt_outputs.json` | The real prompt and generated text, proving the model was doing meaningful work while being traced. |
| `results/.sample_calibration.json` | The sampling rate used for each model and length, which matters because the trace is a sample, not every memory access. |

## Include, but shrink first

| Item | Why it matters |
|---|---|
| per-run `trace.parquet` (2.6 GB total, ~5M rows per run) | The raw-ish evidence of every sampled memory access, so researchers can compute their own metrics, but it must be subsampled or limited to a few runs to fit Kaggle. |

Subsampling must not thin the rare regions further: `kv_cache` (~2.5k rows/run) and
`weight` (~24k rows/run) are already sparse, while `runtime` is ~98% of rows. The
script therefore keeps every non-`runtime` row and samples `runtime` only, recording
the inverse keep probability per row so counts can be re-weighted.

## Include in the documentation (data card), not as data

| Item | Why it matters |
|---|---|
| `04 - Region Attribution Validity.md` | Explains how each address was labelled and how reliable each label is, so users don't over-trust it. |
| `05 - Imputed Values in Sampled Measurements.md` | Warns that missing observations are null rather than 0, and that `kv_cache` write-once = 1.0 is an artefact of sparse store sampling, not a finding. |
| `dataset-plan.md` (Step 5) | Already lists what the data card must say, including the proxy caveat that retention is observed lifetime on this CPU, not a hardware requirement. |

## Leave out

| Item | Why not |
|---|---|
| `raw/perf.data` (14 GB) | The huge unprocessed recording, readable only with `perf`; everything useful from it is already in `trace.parquet`. |
| `generation_results.json`, `raw/alloc.log` | Internal pipeline working files whose useful content is already in `region_map` and `metadata`. |
| per-run `manifest.json` | Checksums of the unmodified run directory; the release gets its own `SHA256SUMS`. |
| `.experiment_state.json`, `logs/runner.log` | Scheduler bookkeeping, meaningless to dataset users. |
| Other context files (spec, review, testing philosophy, YAML guide) | About building the tool, not about the data. |

## Privacy

`machine_id`, `software.hostname` and `collector.output_file` contain the
developer's name. The script replaces them with a neutral machine alias; hardware
specs (CPU model, RAM, kernel) are kept because they are needed to interpret the data.

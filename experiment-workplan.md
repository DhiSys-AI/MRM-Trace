# MRM-Trace: Full Experiment Workplan

*This document is a standalone reproduction guide. A partner with no prior context should be able to follow it start to finish and produce the complete dataset.*

*Last updated: 2026-10-03*

---

## 0. What This Document Covers

This guide covers everything needed to run the complete MRM-trace experiment sweep and produce a publishable dataset for two research tracks simultaneously:

- **Track A** - Empirical LLM inference memory characterisation (retention duration, write-once ratio, read frequency) to contribute to the MRM silicon research space and publish as an open Kaggle dataset.
- **Track B** - Retention-aware placement across reused hardware tiers (adds energy measurement; feeds the carbon crossover model described in `review-discuss.md`).

Both tracks collect the same raw traces. Track B adds a RAPL energy overlay on top.

---

## 1. Machine Roles

| Machine | Role | Hardware |
|---|---|---|
| **Dev machine** | Code development, testing, small validation runs only | i5-8300H, 31 GB RAM, bare-metal Linux |
| **Experiment machine** | Full sweep - all models, all contexts, all batch sizes | i9-11980HK, 31 GB RAM, RTX 3080 16 GB, ~380 GB free |

As of 2026-10-03 development and the pilot runs are both happening on the
i9-11980HK box, so "dev" and "partner" are the same machine. Where the text
below says "partner machine", read "whichever machine runs the sweep".

The dev machine runs TinyLlama (GGUF) and Qwen2.5-0.5B (HuggingFace) only - enough to confirm the pipeline is working end-to-end before handing off to the partner machine.

The partner machine runs the full matrix described in Section 7.

---

## 2. Partner Machine Hardware Profile

| Property | Value |
|---|---|
| CPU | Intel Core i9-11980HK (Tiger Lake, 8c/16t) |
| Intel PMU version | **5** - full PEBS with load-latency sampling (`ldlat`) |
| RAM | 32 GB DDR4 |
| GPU | NVIDIA RTX 3080 Laptop, **16 GB VRAM** |
| Storage | ~3.4 TB free NVMe |
| NUMA | Single socket - trivial (all memory node 0) |
| Thermal risk | **HIGH** - Alienware x17 throttles under sustained load, no AC |
| perf mem event | PEBS load-latency available; the collector passes no `-e` and lets perf auto-select, since `ldlat` syntax varies across kernels. The event actually used is recorded per run in `metadata.json` as `collector.event_strings`. |

**Critical note on thermals:** This machine WILL heat-soak under sustained perf collection. See Section 9 for the mandatory thermal protocol. Runs collected while throttled are scientifically invalid and must be discarded.

---

## 3. Ubuntu Installation

Ubuntu is not yet installed on the partner machine. Follow these steps.

### 3.1 Recommended OS

**Ubuntu 22.04.4 LTS** (Jammy) - stable, well-supported perf-tools packages, proven with Tiger Lake and RTX 3080.

Download: `https://releases.ubuntu.com/22.04/ubuntu-22.04.4-desktop-amd64.iso`

### 3.2 Create bootable USB

On Windows (before wipe):
```
Rufus → select ISO → GPT partition scheme → UEFI only → write
```

On Linux:
```bash
sudo dd if=ubuntu-22.04.4-desktop-amd64.iso of=/dev/sdX bs=4M status=progress && sync
```

### 3.3 BIOS settings (Alienware x17)

Enter BIOS (F2 at boot):
- **Secure Boot**: Disable (required for NVIDIA drivers and perf kernel modules)
- **Intel Virtualization**: Leave enabled (no effect on PMU)
- **Turbo Boost**: Leave for now (we disable it in software per-run)
- **Thermal**: Set fan to "Full Speed" or "Performance" mode

### 3.4 Installation options

- Partition: use entire disk, ext4, no LVM (simpler for trace file management)
- Swap: 8 GB (for large model loads)
- Username: keep simple, e.g. `mrm`
- **Do not** encrypt disk (encryption overhead affects perf traces)

### 3.5 First boot

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git curl wget python3-pip python3-venv \
  linux-tools-$(uname -r) linux-tools-generic \
  lm-sensors cpufrequtils htop nvtop screen tmux \
  libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
  libffi-dev liblzma-dev
```

---

## 4. System Configuration

### 4.1 Kernel perf paranoia

```bash
# Check current value
cat /proc/sys/kernel/perf_event_paranoid

# Set to 0 for this session (root can override anyway, but this is cleaner)
sudo sysctl -w kernel.perf_event_paranoid=0

# Make permanent
echo 'kernel.perf_event_paranoid=0' | sudo tee /etc/sysctl.d/99-perf.conf
sudo sysctl --system
```

### 4.2 NVIDIA drivers and CUDA

```bash
# Auto-install recommended driver
sudo ubuntu-drivers autoinstall
sudo reboot

# After reboot - verify
nvidia-smi
# Should show RTX 3080 with driver version 525+ and CUDA 12.x

# Install CUDA toolkit for vLLM
sudo apt install nvidia-cuda-toolkit
nvcc --version
```

### 4.3 Verify perf mem works (MUST DO before anything else)

```bash
sudo perf mem record -- sleep 2
sudo perf mem report --stdio | head -25
```

Expected output - look for non-zero samples under `cpu/mem-loads,ldlat=...`.
The exact threshold perf picks varies by kernel; any value is fine, and the one
used is recorded per run in `metadata.json` as `collector.event_strings`:
```
# Samples: N  of event 'cpu/mem-loads,ldlat=30/P'
# Total weight : ...
```

If samples = 0 or the command errors: stop. The PMU is not accessible. Check `dmesg` for perf errors.

**This step is a hard gate. Do not proceed if it fails.**

**Confirmed on dev machine (2026-06-28):** `perf mem record -- sleep 2` produced 3 samples on `cpu/mem-loads,ldlat=30/P` on i5-8300H. The partner machine's Tiger Lake PMU v5 is stronger and will produce denser samples.

### 4.4 Sensors (for thermal monitoring)

```bash
sudo sensors-detect   # accept all defaults
sensors               # should show CPU core temps
```

---

## 5. Python Environment Setup

Every command in this guide, on either machine, resolves paths through a single
`MRM_ROOT` variable set once per machine. Nothing below is hardcoded to a
particular username or clone location — set `MRM_ROOT` correctly for the box
you're on and every subsequent command (including Sections 7, 10, and 13) just works.

```bash
# 1. Clone the repo (anywhere you like)
git clone <mrm-trace-repo-url> mrm-trace
cd mrm-trace

# 2. Set MRM_ROOT once for this machine/shell — $(pwd) is always absolute.
#    Add this export to ~/.bashrc (or re-run it each new shell) so it persists.
export MRM_ROOT="$(pwd)"

# 3. Create the venv INSIDE the repo and install mrm-trace in editable/dev mode.
#    This is a local, easy-to-test install: source edits under mrm_trace/ take
#    effect immediately, no reinstall or separate package venv needed.
python3 -m venv "$MRM_ROOT/venv"
source "$MRM_ROOT/venv/bin/activate"
pip install --upgrade pip
pip install -e "$MRM_ROOT[test]"

# Verify — should print the source tree path, not site-packages
python -c "import mrm_trace; print(mrm_trace.__version__); print(mrm_trace.__file__)"
mrm-trace --help
```

`$MRM_ROOT/venv` provides both the mrm-trace package/CLI and the scripts in
`notebooks/scripts/` from the same clone — one venv, one repo, per machine.

### 5.1 llama-cpp-python (CPU path) — MUST be a debug-symbol source build

**Do not install the pre-built wheel.** This is a hard requirement, not an optimisation.

Region labelling has two independent mechanisms (spec §5): symbol rules matching
`ggml_*` / `llama_kv_cache_*`, and address-range injection from `/proc/self/maps`.
The pip wheel — including 0.3.34 — ships **stripped** shared libraries, so `perf
script` resolves no symbols and the symbol mechanism is dead. Measured on a real
run with the wheel installed: `symbol (non-empty): 0.0%` across 10,082,976 samples.

> An earlier revision of this document claimed 0.3.34+ needed no debug build
> because it exports symbols "at link time". That was wrong. Verified on the
> partner machine 2026-08-01: the wheel's `libllama.so` is 3.6 MB and reports
> `stripped`; the source build is 66 MB and reports `not stripped`.

```bash
# Source build with debug symbols. Takes 5–15 min — it compiles llama.cpp.
CMAKE_BUILD_TYPE=Debug pip install --no-binary llama-cpp-python \
  --force-reinstall llama-cpp-python
```

`--no-binary` is as important as `CMAKE_BUILD_TYPE`. Without it pip resolves to a
pre-built wheel, the env var is **silently ignored**, and you get a stripped
library with no error.

Verify — all three must hold:
```bash
# 1. Import works and version is >= 0.3.34
"$MRM_ROOT/venv/bin/python" -c "
import importlib.metadata
print('llama-cpp-python', importlib.metadata.version('llama-cpp-python'))
from llama_cpp import Llama; print('import OK')"

# 2. Libraries are NOT stripped  (this is the check that matters)
find "$MRM_ROOT/venv" -name 'libllama.so' -o -name 'libggml*.so' | while read f; do
  printf '%-20s %s  %s\n' "$(basename $f)" "$(file -b "$f" | grep -o 'not stripped\|stripped')" "$(du -h "$f" | cut -f1)"
done
# Expected: every line reads "not stripped"; libllama.so ~66 MB (NOT 3.6 MB)

# 3. The symbols the labeller keys on are present
L=$(find "$MRM_ROOT/venv" -name libllama.so | head -1)
nm -C --defined-only "$L" | grep -ci kv_cache      # expect > 1000
```

If step 2 prints `stripped`, the build fell back to a wheel — re-run with
`--no-binary` and confirm compilation output scrolls past.

The source build also installs a second copy of the libraries into
`site-packages/lib/` (~445 MB) which nothing loads; the copy that matters is
`site-packages/llama_cpp/lib/`. It is harmless, but it is why the Section 7
check asks the module for its library directory instead of taking the first
`find` hit — with two copies present, `find | head -1` can inspect the one that
is never loaded. The stray copy is safe to delete if disk is tight:

```bash
"$MRM_ROOT/venv/bin/python" -c "import llama_cpp, pathlib; print(pathlib.Path(llama_cpp.__file__).parent / 'lib')"
# anything under site-packages/lib/ that is NOT that path is unused
```

### 5.1.1 Why `kv_cache` may be empty on the llama.cpp path

The KV cache is **not separable from `/proc/self/maps` alone**. llama.cpp carves
it out of a larger ggml backend arena, so the mapping boundaries do not
correspond to the KV buffer. Measured on Llama-3.2-1B / ctx=512:

| | |
|---|---|
| KV size implied by model geometry | **16.8 MB** |
| Actual new anonymous mappings after load | 262.7 MB, 778.8 MB, 536.9 MB, 2.1 MB |

An earlier rule labelled *any* new anonymous mapping ≥ 2 MB as `kv_cache`. That
tagged 1,580 MB of backend arena and thread-stack memory as KV cache and produced
48,000 bogus "KV regions" with `retention_p99_s = 0.0` and
`reads_per_write = 0.0007` — a number that would not survive review.

A mapping is now claimed as `kv_cache` only when its size matches
`2 × n_layer × n_ctx × n_head_kv × head_dim × 2 B` from the GGUF metadata. When
nothing matches, `kv_cache` and `kv_block_lifecycle.parquet` are **empty, by
design**. An absent measurement is publishable; a wrong one is not.

To actually obtain KV regions, in order of preference:

1. **Symbol-based labelling** via the Section 5.1 debug build (`llama_kv_cache_*`
   rules in `mrm_trace/labeller/symbol_rules.py`).
2. **Phase-aware separation** inside the arena — possible now that the `phase`
   column is populated (spec §5, ADR-3).
3. **The GPU/vLLM path**, where KV blocks are explicit PyTorch tensors and
   `mrm_trace/labeller/kv_address_monitor.py` captures exact addresses.
   review-discuss §3.5 already requires validating weight-vs-KV lifetime ordering
   here, so this is on the critical path regardless.

### 5.1.2 Region granularity — a methodology choice the paper must state

Retention is `last access − first access` **of a region**, so it can only be
estimated when the same region is sampled more than once. `perf mem` samples
sparsely, and its load samples are PEBS cache-miss samples — a re-read that hits
in cache is never recorded. Both granularity extremes therefore produce artifacts
that look like data. Measured on Llama-3.2-1B / ctx=512 / 10.1M samples:

| granule | samples/region | retention > 0 | p50 (s) | p99 (s) | verdict |
|---|---|---|---|---|---|
| 4 KB | 1.07 | 7% | 0.000 | 13.587 | undersampled |
| 16 KB | 1.27 | 23% | 0.000 | 17.471 | undersampled |
| **64 KB** | **2.27** | **66%** | **5.117** | **19.547** | **usable** |
| 256 KB | 7.50 | 97% | 16.099 | 20.506 | usable |
| 1 MB | 27.3 | 92% | 19.769 | 20.868 | saturated |
| 16 MB | 213 | 80% | 20.831 | 21.526 | saturated |

At 4 KB nearly every region holds one sample, so retention is 0 by construction.
At ≥ 1 MB every granule is touched near the start and near the end, so retention
converges on the ~21 s **run duration** — a property of how long you ran, not of
the data. Only around 64 KB is there real spread.

The table above is **one run at one sample density** — do not carry 64 KB
forward as a setting. A denser 30.6M-sample run of the same model recommended
very different granules per region type:

| region type | recommended granule | 64 KB verdict |
|---|---|---|
| `activation` | 4 KB | saturated |
| `kv_cache` | 64 KB | usable |
| `runtime` | 1024 KB | undersampled |
| `weight` | 4096 KB | undersampled |

There is no universally correct value: it depends on sample density and
working-set size, so it varies with model, context length and `--sample-period`.
Every run therefore prints a sweep with a `usable` / `undersampled` / `saturated`
verdict per region type and a recommended granule. **Read it before trusting any
retention number**, and set `--region-granularity` accordingly:

```bash
sudo -E "$MRM_ROOT/venv/bin/python" notebooks/scripts/demo_perf_mem.py \
  --model models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --ctx 512 --sample-period 128000 --region-granularity 65536
```

The granule used is recorded in `metadata.json` as
`collector.region_granularity_bytes`, so a published retention figure can
always be traced back to it. The experiment runner passes
`--region-granularity` on every run (default 65536, override with the same flag
on the runner).

If the sweep reports no usable granule, that trace cannot support a per-region
retention distribution — report aggregate figures only, or re-collect with a
smaller `--sample-period`. Record the granule used alongside every retention
figure in the paper; it is not a free parameter.

### 5.1.3 Sampling period — the parameter that decides whether a run completes

`--sample-period` is a **period**, not a frequency: one sample per N memory
accesses. Sample count therefore scales with how long a run takes, not with how
big the model is. Measured on the experiment machine:

| | |
|---|---|
| Samples | 30,608,373 |
| Generation time | 83.1 s |
| `--sample-period` | 32,000 |
| **Sampled-access rate** | **368k samples/s** = 1.18 x 10^10 memory accesses/s |

That rate is roughly model-independent — it is bandwidth-bound, not
parameter-bound — so one fixed period across the matrix is the wrong
instrument. At `--sample-period 32000`:

| run length | samples | labelled frame | `perf_script.txt` |
|---|---|---|---|
| 1.5 min (the pilot above) | 31M | 4.3 GB | 8.9 GB |
| 20 min | 442M | 61 GB | 141 GB |
| 40 min | 884M | 123 GB | 282 GB |
| 90 min | 1,989M | 277 GB | 634 GB |

A 31 GB machine analyses roughly **30M rows** (measured: 15 GB peak RSS, parse
through export). Past ~90M rows a run cannot be labelled or analysed at all,
whatever the disk situation.

**The period is therefore derived from a fixed sample budget**, not chosen per
model: `experiment_runner.py` sets `SAMPLE_BUDGET = 30_000_000` and solves for
the period. Two things set run length, and both feed it — context length
(decode slows as attention scans more of it: 11.2 tok/s at ctx 512 vs 8.38 at
2048, measured) and prompts x output tokens. Batch size does not: batch 1, 4
and 8 all ran 395-455 s, because the prompts are processed sequentially.

At `PROMPTS_PER_RUN = 4` and `OUTPUT_TOKENS = 512`:

| model | ctx 2048 | ctx 8192 | ctx 32768 |
|---|---|---|---|
| llama-3.2-1b | 99,000 (4 min) | 199,000 (8 min) | 597,000 (24 min) |
| llama-3.2-3b | 273,000 (11 min) | 547,000 (22 min) | 1,640,000 (67 min) |
| phi-3.5-mini | 340,000 (14 min) | 680,000 (28 min) | 2,041,000 (84 min) |
| mistral-7b | 641,000 (26 min) | 1,282,000 (53 min) | 3,847,000 (158 min) |
| llama-3.1-8b | 725,000 (30 min) | 1,450,000 (59 min) | 4,349,000 (178 min) |

**The estimate only seeds the first run of each (model, ctx).** It is guesswork,
and it was wrong: the first 8 runs assumed 11.0 tok/s where the truth was 8.38,
and came in at 40.5M samples against the 30M budget — 35% over, silently. Each
run therefore records what its period actually produced in
`results/.sample_calibration.json`, and the next run of the same configuration
corrects proportionally:

```
period_next = period_used x samples_observed / SAMPLE_BUDGET
```

No duration model is involved once a configuration has been measured once, and
rep 0 is already a warmup, so its measurement calibrates the reps that carry
data. The runner prints the correction after each run:

```
30,142,939 samples (1.00x budget) at -c 99,000  ->  next run uses -c 99,000
```

The calibration key includes context, output tokens and prompt count, so
changing any of them discards stale measurements rather than reusing them.

The runner logs the period and timeout for every configuration before starting.
For manual runs, **always pass `--sample-period` explicitly** — the script's own
default is 4,000, far denser than any value above, and will produce a trace too
large to analyse.

### 5.2 PyTorch + transformers (GPU path)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate

# Verify GPU is visible
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: NVIDIA GeForce RTX 3080 Laptop GPU
```

### 5.3 HuggingFace access (for gated models)

Llama-3.x and some other models require a HuggingFace account and license acceptance.

```bash
pip install huggingface_hub
hf auth login
# Paste your HF token (read-only scope is fine)

# To verify if you're already logged in
hf auth whoami
```

Then go to each model's HuggingFace page and click "Agree and access repository":
- `https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct`
- `https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct`
- `https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct`

---

## 6. Model Inventory

All models for the full experiment matrix. Download before starting the runner.

### 6.1 CPU path - GGUF models (perf mem primary collector)

```bash
cd "$MRM_ROOT"
mkdir -p models
```

| # | Model ID | GGUF File | HF Repo | Size | Arch |
|---|---|---|---|---|---|
| 1 | `tinyllama-1.1b` | `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` | `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` | 668 MB | LLaMA |
| 2 | `llama-3.2-1b` | `Llama-3.2-1B-Instruct-Q4_K_M.gguf` | `bartowski/Llama-3.2-1B-Instruct-GGUF` | 770 MB | LLaMA-3 |
| 3 | `llama-3.2-3b` | `Llama-3.2-3B-Instruct-Q4_K_M.gguf` | `bartowski/Llama-3.2-3B-Instruct-GGUF` | 1.9 GB | LLaMA-3 |
| 4 | `phi-3.5-mini` | `Phi-3.5-mini-instruct-Q4_K_M.gguf` | `bartowski/Phi-3.5-mini-instruct-GGUF` | 2.2 GB | Phi-3 |
| 5 | `mistral-7b` | `Mistral-7B-Instruct-v0.3-Q4_K_M.gguf` | `bartowski/Mistral-7B-Instruct-v0.3-GGUF` | 4.1 GB | Mistral |
| 6 | `llama-3.1-8b` | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` | 4.9 GB | LLaMA-3 |

**Download commands.** All six GGUF repos below are ungated community
conversions — no token or license click-through is needed. (The gated
`meta-llama/*` repos in Section 5.3 matter only for the GPU path, which loads
the original weights.)

```bash
cd "$MRM_ROOT"
hf download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --local-dir models/
hf download bartowski/Llama-3.2-1B-Instruct-GGUF \
  Llama-3.2-1B-Instruct-Q4_K_M.gguf --local-dir models/
hf download bartowski/Llama-3.2-3B-Instruct-GGUF \
  Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir models/
hf download bartowski/Phi-3.5-mini-instruct-GGUF \
  Phi-3.5-mini-instruct-Q4_K_M.gguf --local-dir models/
hf download bartowski/Mistral-7B-Instruct-v0.3-GGUF \
  Mistral-7B-Instruct-v0.3-Q4_K_M.gguf --local-dir models/
hf download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
  Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --local-dir models/
```

`hf download` resumes partial transfers; `wget` restarts them, which matters on
the 4–5 GB files.

**Verify all models present:**
```bash
ls -lh models/*.gguf
# Expected total: ~14 GB
```

### 6.2 GPU path - HuggingFace models (transformers, PyTorch GPU)

These are downloaded automatically on first use by `transformers`. They are cached in `~/.cache/huggingface/hub/`. Pre-downloading is optional but recommended to avoid network timeouts during experiments.

| # | HF Model ID | Size (bf16) | VRAM required | Notes |
|---|---|---|---|---|
| 1 | `Qwen/Qwen2.5-0.5B-Instruct` | 1 GB | ~2 GB | No login required |
| 2 | `meta-llama/Llama-3.2-3B-Instruct` | 6 GB | ~8 GB | Requires HF login + license |
| 3 | `mistralai/Mistral-7B-Instruct-v0.3` | 14 GB | ~16 GB | Requires HF login; tight on 16 GB VRAM |

Pre-download (optional — `transformers` fetches these on first use):
```bash
hf download Qwen/Qwen2.5-0.5B-Instruct
hf download meta-llama/Llama-3.2-3B-Instruct        # needs login + license
hf download mistralai/Mistral-7B-Instruct-v0.3      # needs login
```

**VRAM note for Mistral-7B:** With 16 GB VRAM, bf16 weights use ~14 GB leaving ~2 GB for KV cache. Use `--ctx 4096` (not 8192 or 32768) for the GPU path Mistral runs to avoid OOM.

### 6.3 Dev machine models (testing only)

```bash
# Already downloaded
models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Qwen2.5-0.5B - auto-downloaded by demo_qwen_hf.py on first run
"$MRM_ROOT/venv/bin/python" notebooks/scripts/demo_qwen_hf.py \
  --model Qwen/Qwen2.5-0.5B-Instruct
```

---

## 7. Verification Checklist

Run every check below on the **partner machine** before starting the sweep. The first check is a hard gate — do not proceed if it exits non-zero.

```bash
cd "$MRM_ROOT"


# 1. HARD GATE — full automated machine validation
#    Checks: OS, WSL2, kernel, perf binary, perf version, root, paranoid,
#    CPU vendor + PEBS sysfs, dmesg PMU, live smoke test, thermal, memray, disk.
#    Exit 0 = publication-grade perf mem. Exit 1 = degraded. Exit 2 = stop.
sudo "$MRM_ROOT/venv/bin/mrm-trace" validate --machine
# Tiger Lake PMU v5 should exit 0 with no action required.
#
# WARN governor: means frequency is not pinned (governor != performance, or
# turbo on) — a reproducibility risk, not a blocker. Apply the Section 9.1
# thermal protocol commands (cpupower + no_turbo) before starting the sweep,
# then this check should read PASS.

# Save the validation report to the results directory (for paper provenance)
mkdir -p results
sudo "$MRM_ROOT/venv/bin/mrm-trace" validate --machine --json > results/machine_validation.json
cat results/machine_validation.json | python -m json.tool | grep verdict

# 2. Python and mrm-trace
"$MRM_ROOT/venv/bin/python" -c "import mrm_trace; print(mrm_trace.__version__)"
"$MRM_ROOT/venv/bin/mrm-trace" validate --config config/default_experiment.yaml

# 3. llama-cpp-python — import AND debug symbols in the libraries actually
#    loaded (see 5.1; a stripped build silently disables symbol-based region
#    labelling). Ask the module where its libs are rather than using
#    `find | head -1`: a source build can leave a second, unused copy elsewhere
#    in site-packages, and head -1 may inspect that one instead. libggml-cpu.so
#    is checked too — it supplies most of the symbols that appear in a trace
#    (ggml_vec_dot_*, ggml_gemv_*), not libllama.so.
LIBDIR=$("$MRM_ROOT/venv/bin/python" -c \
  "import llama_cpp, pathlib; print(pathlib.Path(llama_cpp.__file__).parent / 'lib')")
"$MRM_ROOT/venv/bin/python" -c "from llama_cpp import Llama; print('llama-cpp OK')"
echo "loaded from: $LIBDIR"
fail=0
for f in "$LIBDIR"/libllama.so "$LIBDIR"/libggml-cpu.so; do
  if file -b "$f" | grep -q 'not stripped'; then
    printf '  %-16s not stripped  %s  symbols: %s\n' \
      "$(basename "$f")" "$(du -h "$f" | cut -f1)" \
      "$(nm -C --defined-only "$f" 2>/dev/null | grep -cE ' [Tt] (ggml_|llama_kv_cache)')"
  else
    echo "  FAIL: $f is stripped — redo Section 5.1 with --no-binary"; fail=1
  fi
done
[ "$fail" -eq 0 ] && echo "debug symbols OK" || exit 1
# Expected on a correct source build (measured 2026-10-03):
#   libllama.so      not stripped  66M   symbols: 305
#   libggml-cpu.so   not stripped  4.4M  symbols: 791

# 4. PyTorch GPU
"$MRM_ROOT/venv/bin/python" -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

# 5. Model files
ls -lh models/*.gguf | awk '{print $5, $9}'
# Expected: 6 files totalling ~14 GB

# 6. Temperature baseline (must be < 60°C before starting)
sensors | grep -E "Core|Package|temp"

# 7. Disk space
df -h ~   # need > 400 GB free: ~340 GB of retained artifacts for the full
          # sweep (Section 11.2) plus headroom for one run's perf_script.txt

# 8. Quick end-to-end test (TinyLlama, 1 prompt, ~1 min)
#    --num-prompts 1 is what makes it quick; without it the script runs all 7.
#    --sample-period is mandatory, see Section 5.1.3. 16000 suits a run this
#    short; the sweep's real values are much coarser.
sudo -E "$MRM_ROOT/venv/bin/python" notebooks/scripts/demo_perf_mem.py \
  --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --ctx 512 --tokens 32 --batch-size 1 --rep 0 \
  --num-prompts 1 --sample-period 16000 --region-granularity 65536
# Expected console output (two inline quality gates printed by the pipeline):
#
#   Expected KV cache from model geometry: 11.5 MB   ← TinyLlama @ ctx 512
#                                                      (2 x 22 layers x 512 ctx
#                                                       x 4 KV heads x 64 dim
#                                                       x 2 B); other models
#                                                      differ — the script
#                                                      computes it from the GGUF
#   Pre-registered N pages from /proc/maps:
#     weight    mmap_gguf         ...  ~the GGUF file size
#     runtime   anon_post_load    ...  ggml backend arenas
#     runtime   thread_stack      ...  one 8 MB stack per ggml worker
#     ...      (sizes vary by model and thread count; `weight` must be present)
#
#   Trace quality gate [perf_mem] — PASS        ← Gate 1: latency + cache_level
#     access_latency_ns    : [████████████████████] 100.0%   ← PASS needs ≥ 95%
#     cache_level          : [████████████████████] 100.0%   ← PASS needs ≥ 85%
#     symbol (non-empty)   : > 0%     ← REQUIRES the 5.1 debug build; 0.0% = stripped
#     phase labelled       : [████████████████████] 100.0%   ← must NOT be 0.0%
#
#   Trace quality gate [perf_mem_labelled] — PASS  ← Gate 2: + region classified
#     region classified    : [████████████████████] 100.0%   ← PASS needs ≥ 5%
#   region classified (load ):  ~99%
#   region classified (store): ~100%
#
# Two failure modes that produce a healthy-looking run with unusable data:
#   phase labelled = 0.0%   → collector/engine clock domain mismatch; every
#                             phase is NULL and phase-aware labelling is dead.
#                             Gate 2 now FAILs on this.
#   symbol = 0.0%           → stripped llama-cpp build; redo Section 5.1.
#
# kv_cache may legitimately be absent on the llama.cpp path — see Section 5.1.1.
#
# If either gate prints FAIL in red, the pipeline exits immediately — the run is
# NOT exported. Fix the issue before continuing.
#
# Layout is results/<machine_id>/<model_id>/<run_id>/ (ADR-7), so the glob needs
# the machine level.
RUN_DIR=$(ls -td results/*/tinyllama-1.1b-q4km-perf/*/ 2>/dev/null | head -1)
cat "$RUN_DIR/metrics.csv"   # one row per region_type: weight, kv_cache,
                             # activation, runtime, unknown — with retention,
                             # write_once and read_freq columns

# 9. Validity test suite — run before dataset release
#    Verifies that known synthetic inputs produce known metric outputs.
#    These are the most important tests: a 30 s weight-retention trace must
#    yield retention_p99_s ≈ 30.0, a write-once trace must yield ratio ≈ 1.0, etc.
#    All must pass before any data is exported or uploaded.
"$MRM_ROOT/venv/bin/python" -m pytest -m validity -q
```

---

## 8. Experiment Matrix

### 8.1 CPU path (primary - perf mem, llama.cpp)

| Dimension | Values | Count |
|---|---|---|
| Models | llama-3.2-1b, llama-3.2-3b, phi-3.5-mini, mistral-7b, llama-3.1-8b | 5 |
| Context lengths | 2048, 8192, 32768 | 3 |
| Batch sizes | 1, 4, 8 | 3 |
| Reps per config | 3 (rep 0 = warmup, reps 1–2 = data) | 3 |
| **Total CPU runs** | 5 × 3 × 3 × 3 | **135** |

TinyLlama is commented out in `CPU_MODELS` — it is a pipeline smoke test
(Section 7 check 8), not a dataset model. Uncomment it for 6 models / 162 runs.

Each run generates `PROMPTS_PER_RUN = 4` prompts of `OUTPUT_TOKENS = 512`
tokens, both set in `experiment_runner.py` and fixed across the matrix for
comparability. 4 rather than all 7 prompts: run length is linear in prompt
count, and 7 put the sweep at ~201 h of active runtime against ~115 h at 4,
while repetition still comes from 3 batch sizes x 3 reps per configuration.
Note the `demo_perf_mem.py` defaults differ (128 tokens, all 7 prompts) — the
runner passes both explicitly, so a manual run needs to as well:
```bash
sudo -E "$MRM_ROOT/venv/bin/python" notebooks/scripts/demo_perf_mem.py \
  --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --tokens 512 --sample-period 128000 --region-granularity 65536
```
The configured ceiling is recorded in `metadata.json` as `generation.output_tokens` and in
`manifest.csv` as `max_tokens_per_prompt`. The actual tokens generated (which may be lower
if the model hits EOS before the ceiling) appears as `total_output_tokens` in `manifest.csv`.

### 8.2 GPU path (cross-validation - transformers, PyTorch GPU)

| Dimension | Values | Count |
|---|---|---|
| Models | qwen2.5-0.5b, llama-3.2-3b, mistral-7b | 3 |
| Context lengths | 2048, 8192 — capped per model by VRAM | 1–2 |
| Batch sizes | 1, 4 | 2 |
| Reps | 2 (rep 0 = warmup, rep 1 = data) | 2 |
| **Total GPU runs** | see below | **16** |

Not 24: each model's `max_ctx` in `GPU_MODELS` prunes the context dimension.
Qwen2.5-0.5B is capped at 2048 and Mistral-7B at 4096 (VRAM), so only
Llama-3.2-3B runs both context lengths — 4 + 8 + 4 = 16.

### 8.3 Total sweep

Derived from the measured 8.38 tok/s at ctx 2048 and the context scaling in
Section 5.1.3, at 4 prompts x 512 tokens:

| Path | Runs | Est. time per run | Est. total active |
|---|---|---|---|
| CPU - llama-3.2-1b | 27 | 4–24 min | ~6 hrs |
| CPU - llama-3.2-3b | 27 | 11–67 min | ~15 hrs |
| CPU - phi-3.5-mini | 27 | 14–84 min | ~19 hrs |
| CPU - mistral-7b | 27 | 26–158 min | ~36 hrs |
| CPU - llama-3.1-8b | 27 | 30–178 min | ~40 hrs |
| GPU path | 16 | 5–20 min | ~3 hrs |
| **Total** | **151** | - | **~118 hrs** |

**Expected calendar time:** ~15 nights at 8 hrs/night, plus cooldowns. The
ctx-32768 figures are extrapolated from two measured points (ctx 512 and 2048)
and are the least reliable; llama-3.2-1b at ctx 32768 runs on night 1 and
replaces the extrapolation with a measurement.

Run `experiment_runner.py --dry-run` for the authoritative count and the
per-configuration periods and timeouts — both are computed from `CPU_MODELS`,
and this table will drift from them.

Start with small models first - they validate the pipeline and produce usable
data fast even if larger runs are interrupted.

### 8.4 Execution order (recommended)

Run in this order to get publishable data as early as possible:

1. Llama-3.2-1B - all contexts, all batches, all reps (validates full pipeline)
2. Llama-3.2-3B - same
3. Phi-3.5-mini - same
4. GPU path: Qwen2.5-0.5B, Llama-3.2-3B (GPU)
5. Mistral-7B - same
6. Llama-3.1-8B - same
7. GPU path: Mistral-7B (GPU, if VRAM allows)

TinyLlama is the Section 7 check-8 smoke test, not part of the matrix.

The experiment runner handles this order automatically.

---

## 9. Thermal Protocol (CRITICAL - no AC)

The PC chassis will heat-soak under sustained load. A throttled run is a contaminated run - its retention duration and latency values are artifacts of the thermal state, not the memory access pattern.

### 9.1 Pre-experiment setup

```bash
# Disable turbo boost for reproducibility (prevents burst → throttle cycle)
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

# Verify: base clock is 2.6 GHz for i9-11980HK
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq
# Should read ~2600000 (2.6 GHz)

# Set performance governor (locks to max non-turbo freq)
sudo cpupower frequency-set -g performance

# Set NVIDIA to persistence mode (avoids init overhead during runs)
sudo nvidia-smi -pm 1

# Power-limit GPU during CPU-path runs (GPU should be idle; this prevents
# passive thermal contribution from GPU power leakage).
# Check the allowed range first — values outside it are rejected:
nvidia-smi -q -d POWER | grep -E "Min Power Limit|Max Power Limit|Default Power Limit"
# Measured on the RTX 3080 Laptop in this machine: min 1 W, default 115 W,
# max 165 W (not the 150 W an earlier revision of this document assumed).
sudo nvidia-smi -pl 50      # during CPU-path runs
sudo nvidia-smi -pl 115     # restore the default before GPU-path runs
```

### 9.2 Temperature thresholds

| Threshold | Meaning |
|---|---|
| < 60°C | Safe to start a new run |
| 60–75°C | Running, acceptable |
| 75–85°C | Running warm - log but continue |
| > 85°C | **Thermal warning** - run is at most `marginal` |
| > 90°C | **Hard throttle likely** - run is `contaminated`, discard from primary dataset |

Temperature is only one input to `run_validity`. Observer CPU, observer memory,
baseline CPU, throttle events and I/O contention all feed the same verdict, so
a `contaminated` run is **not necessarily a hot run** — check
`thermal.validity_failure_reasons` in `metadata.json`, which names the
criterion that failed. The runner prints the same reasons under the DONE line.

Validity is a label over numbers the run already recorded, so it can always be
re-derived without re-collecting anything:

```bash
# Preview what would change
sudo -E "$MRM_ROOT/venv/bin/python" notebooks/scripts/experiment_runner.py \
  --reclassify --dry-run

# Apply: rewrites metadata.json, .experiment_state.json and manifest.csv
sudo -E "$MRM_ROOT/venv/bin/python" notebooks/scripts/experiment_runner.py --reclassify
```

### 9.3 Cooling between runs

The experiment runner enforces:
- **Mandatory 60-second cooldown** after every run (sensor-settle time, not a thermal wait)
- **Adaptive wait** until CPU package temp < 55°C before starting the next run
- Checks temperature every 60 seconds during the adaptive wait

For large model runs (7B–8B): an extra 60-second mandatory pause, then the same adaptive wait until < 55°C.

The 60-second floor exists only to let the thermal sensor stabilise after the subprocess exits. The `wait_for_cool` loop is the real thermal gate — it extends the pause as long as needed regardless of model size.

### 9.4 Practical advice

- Run overnight or in the coolest part of the day
- Place a desk fan aimed at the laptop vents if possible (even without AC)
- Do not run other applications during the sweep (browser, IDE, etc.)
- The runner will display current temperature before each run

---

## 10. The Experiment Runner

The experiment runner script lives at:
```
notebooks/scripts/experiment_runner.py
```

### 10.1 Features

- **State-tracked**: every run's status (pending / running / done / failed / warmup) is recorded in `results/.experiment_state.json`
- **Crash-safe**: state is written atomically after each run completes; a crash mid-run leaves the run as `failed` (retryable), not corrupt
- **Resume**: re-running the script skips completed runs and picks up where it left off
- **Thermal management**: checks CPU temperature before each run, waits until safe
- **Timeout**: per run, 3x its estimated collection time with a 2-hour floor and a 12-hour ceiling. A flat 2 hours would have hard-killed every ctx-32768 run of phi, mistral and llama-3.1-8b — 27 of the 135 CPU runs — for a reason unrelated to the science.
- **Progress display**: live table showing completed / pending / failed counts and estimated time remaining
- **Logging**: full output of each run logged to `results/logs/runner.log`

### 10.2 Commands

```bash
# Run or resume the sweep (standard usage)
sudo -E "$MRM_ROOT/venv/bin/python" notebooks/scripts/experiment_runner.py

# Show current status without running
sudo -E "$MRM_ROOT/venv/bin/python" notebooks/scripts/experiment_runner.py --status

# Retry only runs that previously failed
sudo -E "$MRM_ROOT/venv/bin/python" notebooks/scripts/experiment_runner.py --retry-failed

# Restart from scratch (WARNING: discards all state, does NOT delete results)
sudo -E "$MRM_ROOT/venv/bin/python" notebooks/scripts/experiment_runner.py --restart

# Dry run: print what would be run without executing
sudo -E "$MRM_ROOT/venv/bin/python" notebooks/scripts/experiment_runner.py --dry-run

# Fast validation mode: 1 model, 1 context, 1 batch, 1 rep
sudo -E "$MRM_ROOT/venv/bin/python" notebooks/scripts/experiment_runner.py --fast

# CPU path only (skip GPU runs)
sudo -E "$MRM_ROOT/venv/bin/python" notebooks/scripts/experiment_runner.py --cpu-only

# GPU path only
sudo -E "$MRM_ROOT/venv/bin/python" notebooks/scripts/experiment_runner.py --gpu-only

# Re-derive run_validity for every exported run from its own metadata.json.
# Nothing is re-collected; add --dry-run to preview. Use after changing a
# validity threshold (Section 9.2).
sudo -E "$MRM_ROOT/venv/bin/python" notebooks/scripts/experiment_runner.py --reclassify

# Override the region granularity for every run (default 65536; see Section 5.1.2)
sudo -E "$MRM_ROOT/venv/bin/python" notebooks/scripts/experiment_runner.py \
  --region-granularity 1048576
```

The sampling period is **not** a flag — the runner derives it per model from
`SAMPLE_BUDGET` (Section 5.1.3) and logs its choice before the first run. To
change the budget, edit `SAMPLE_BUDGET` in `experiment_runner.py`; it is set
from a measured memory ceiling, so re-measure before raising it.

#### Low End Machines Only
```bash
sudo -E "$MRM_ROOT/venv/bin/python" notebooks/scripts/experiment_runner.py --low-end

sudo -E "$MRM_ROOT/venv/bin/python" notebooks/scripts/experiment_runner.py --low-end --cpu-only

sudo -E "$MRM_ROOT/venv/bin/python" notebooks/scripts/experiment_runner.py --low-end --dry-run   # preview first
```

### 10.3 Stopping and resuming

To stop gracefully: press `Ctrl+C` once. The runner will finish the current run, save state, and exit cleanly. Press `Ctrl+C` twice to abort mid-run (that run is marked `failed` and will be retried on next resume).

To resume: simply re-run the same command. Completed runs are skipped.

### 10.4 State file

`results/.experiment_state.json` - human-readable, do not delete. Contains:
- Status of every run (pending/running/done/failed)
- Timestamps, peak temperature, throttle flag
- Output directory path for each run
- Error message for failed runs

To inspect progress:
```bash
python notebooks/scripts/experiment_runner.py --status
```

Or manually:
```bash
python -c "
import json, pathlib
state = json.loads(pathlib.Path('results/.experiment_state.json').read_text())
runs = state['runs']
from collections import Counter
counts = Counter(v['status'] for v in runs.values())
print(counts)
"
```

### 10.5 Estimated run schedule (8 hrs/night, no AC)

| Night | Models | Runs | Cumulative |
|---|---|---|---|
| 1 | Llama-3.2-1B (all) | 27 | 27 |
| 2 | Llama-3.2-3B (all) | 27 | 54 |
| 3 | Phi-3.5-mini (all) + GPU small | 27 + 8 | 89 |
| 4 | Mistral-7B (partial) | 20 | 109 |
| 5 | Mistral-7B (remainder) + Llama-3.1-8B (partial) | 7 + 14 | 130 |
| 6 | Llama-3.1-8B (remainder) | 13 | 143 |
| 7 | GPU path (remainder) | 8 | 151 |

---

## 11. Data Safety and Checkpointing

### 11.1 What is saved and when

The runner saves state after every completed run. Each run's output directory (`results/<model_id>/<run_id>/`) contains all derived artifacts before the next run starts. A crash loses at most one in-progress run.

Raw artifacts (`raw/perf.data`, `raw/perf_script.txt`) are the most expensive to re-collect. They are written by the collector before parsing begins. If the parse fails, raw files are preserved.

On a **successful** run, `raw/perf_script.txt` is deleted automatically once
export completes — it is intermediate state (`perf script` regenerates it from
`perf.data`, and `trace.parquet` already holds the parsed result) and it is by
far the largest artifact of a run. Pass `--keep-perf-script` to retain it.
`raw/perf.data` is never touched: it is the primary raw artifact, cited by
`metadata.json` and bound for the Zenodo archive (Section 12.3).

### 11.2 Disk space planning

Measured on a 30.6M-sample run (llama-3.2-1b, ctx 512, 83 s of generation).
Every run now targets the same ~30M-sample budget (Section 5.1.3), so these
figures hold across the matrix rather than growing with model size:

| Artifact | Size per run (measured) | 151 runs | Retained? |
|---|---|---|---|
| `trace.parquet` | ~300 MB | ~45 GB | yes |
| `raw/perf.data` | ~1.9 GB | ~290 GB | yes |
| `raw/perf_script.txt` | **~8.9 GB** | **~8.9 GB** | **no — deleted after export** |
| `region_map.parquet` | ~50 KB | < 100 MB | yes |
| `kv_block_lifecycle.parquet` | < 10 KB | < 2 MB | yes |
| `metrics.csv` + JSON | < 20 KB | < 5 MB | yes |

**Steady-state total: ~340 GB**, plus one run's `perf_script.txt` in flight.

Two things make this work, and both matter:

1. **The sampling budget.** Without it a 90-minute 8B run alone produces a
   634 GB text dump and a 277 GB frame (Section 5.1.3). No disk plan survives
   that; the fix is the period, not the storage.
2. **Deleting the text dump after export.** It is the single largest artifact
   and it is regenerable, so only one needs to exist at a time. This is
   automatic — see Section 11.1.

If `perf.data` also becomes a problem, it compresses well (`zstd -3` gets
~3x) and only the runs selected for the Zenodo archive need the original.

### 11.3 Incremental backup (recommended)

After each night's run, rsync derived artifacts to a safe location:
```bash
# Sync only derived artifacts (skip raw, which is re-generatable)
rsync -av --exclude='raw/' results/ /path/to/backup/mrm-results/
```

---

## 12. After the Sweep: Publishing

### 12.1 What goes to Kaggle

Publish only derived artifacts - raw perf.data is too large for Kaggle.

```
kaggle_dataset/
├── metrics.csv              (all runs, one row each)
├── region_map/              (region_map.parquet per run, zstd)
├── kv_lifecycle/            (kv_block_lifecycle.parquet per run, zstd)
├── trace_samples/           (10% random sample of trace.parquet per run, zstd)
├── metadata/                (metadata.json per run)
└── datacard.md              (schema, hardware, proxy caveat)
```

### 12.2 Data card must include

- Hardware: CPU model, RAM, GPU, OS, kernel version
- Collector: perf mem, perf version, sampling period, and the PMU event strings perf auto-selected (`collector.sample_period` and `collector.event_strings` in `metadata.json`). The period is mandatory: a sample count means nothing without it, and it varies per model (Section 5.1.3).
- Region granularity used for retention (`collector.region_granularity_bytes`), with the per-run sweep verdict that justified it (Section 5.1.2)
- Region classification: every `/proc/self/maps` entry is classified, snapshotted both after `Llama()` construction and after the first generation (llama.cpp creates its threadpool lazily). GGUF file mmap → `weight`; thread stacks (identified by the adjacent PROT_NONE guard page), malloc arenas, heap, and shared libraries → `runtime`; a new anonymous mapping matching the KV size implied by GGUF geometry → `kv_cache`. Size alone never implies `kv_cache` (see Section 5.1.1). Addresses in no mapping remain `unknown`.
- Run validity: `clean` / `marginal` / `contaminated` per run, from observer CPU and memory overhead, baseline CPU, thermal throttle events and peak temperature. Observer memory is bounded against **total system RAM** for perf mem (out-of-process: < 10% clean, < 25% marginal) and against **target RSS** for memray (in-process: < 0.05 clean, < 0.15 marginal) — a ratio to target RSS does not describe an out-of-process collector, whose footprint is set by CPU count and ring-buffer size. Each run records the criteria it failed in `thermal.validity_failure_reasons`. Only `clean` and `marginal` runs enter aggregated metrics.
- Quality gate: every run passed two inline gates: (1) pre-labelling — `access_latency_ns` ≥ 95% populated, `cache_level` ≥ 85%, `phase` ≥ 30% non-NULL; (2) post-labelling — the same plus `region_classified` ≥ 5%. Thresholds are defined in `mrm_trace/reporter/trace_qa.py`. Runs that failed either gate were discarded and not re-run until the fault was corrected.
- Sample composition caveat: `perf mem` samples stores at a fixed period but samples loads only above the PEBS `ldlat` threshold, so traces are ~99.6% stores. Report load- and store-derived metrics separately; a blended figure is dominated by store samples, most of which land on thread stacks rather than model data.
- Proxy caveat: *"Retention durations are observed lifetimes under a batch schedule on this CPU/GPU's cache hierarchy. The logical access-pattern structure (write-once ratio, read frequency ordering by region type) transfers across substrates and constitutes first-order evidence for MRM silicon design questions. Absolute timing values are machine-specific. Regions are classified per-run via /proc/maps address-range injection and symbol rules; addresses outside every known mapping appear as region_type=unknown and are excluded from region-level retention comparisons. Where the KV cache could not be isolated from the engine's backend arena, kv_cache is absent rather than approximated."*
- Schema: column definitions for all parquet files
- Citations: Legtchenko et al. (2025) "Managed-Retention Memory"

### 12.3 For Zenodo (full raw archive with DOI)

Raw `perf.data` files go to Zenodo, not Kaggle. Zenodo gives a DOI for formal citation and handles multi-GB files. Upload after paper is accepted or pre-print is live.

### 12.4 arXiv pre-print (priority claim)

Do not wait for the full 186-run sweep. After nights 1–2 (TinyLlama + Llama-3.2-1B complete), the data is sufficient for an initial pre-print. This establishes the priority date.

---

## 13. Quick Reference

```bash
# Smoke test (tinyllama, 1 prompt, ~1 min)
#   --num-prompts 1      otherwise all 7 prompts run
#   --sample-period      MANDATORY — the 4000 default is ~30x too dense (§5.1.3)
#   --region-granularity otherwise retention is computed at 4 KB (§5.1.2)
# --tokens is the max output tokens per prompt (script default 128; the sweep
# uses 512). Recorded in manifest.csv as max_tokens_per_prompt.
sudo -E "$MRM_ROOT/venv/bin/python" notebooks/scripts/demo_perf_mem.py \
  --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --ctx 2048 --tokens 128 --batch-size 1 --rep 1 \
  --num-prompts 1 --sample-period 16000 --region-granularity 65536

# Partner machine - start the full sweep
sudo -E "$MRM_ROOT/venv/bin/python" notebooks/scripts/experiment_runner.py

# Check progress
sudo -E "$MRM_ROOT/venv/bin/python" notebooks/scripts/experiment_runner.py --status

# Resume after interruption
sudo -E "$MRM_ROOT/venv/bin/python" notebooks/scripts/experiment_runner.py

# Retry failed runs
sudo -E "$MRM_ROOT/venv/bin/python" notebooks/scripts/experiment_runner.py --retry-failed

# Check temperature
sensors | grep -E "Package|Core 0"

# Watch disk usage
watch -n 30 "du -sh results/ && df -h ."

# Find runs that never produced a trace (crashed / killed). Dry run first —
# a run still in progress also has no trace.parquet.
find results -mindepth 3 -maxdepth 3 -type d \
  '!' -exec test -e '{}/trace.parquet' ';' -print

# Reclaim text dumps left behind by older runs or by --keep-perf-script
find results -name perf_script.txt -size +1G -printf '%s %p\n' | sort -rn
```

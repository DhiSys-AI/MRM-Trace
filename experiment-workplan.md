# MRM-Trace: Full Experiment Workplan

*This document is a standalone reproduction guide. A partner with no prior context should be able to follow it start to finish and produce the complete dataset.*

*Last updated: 2026-07-21*

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
| **Dev machine** (you) | Code development, testing, small validation runs only | i5-8300H, 31 GB RAM, bare-metal Linux |
| **Partner machine** (experiment machine) | Full sweep - all models, all contexts, all batch sizes | i9-11980HK, 32 GB RAM, RTX 3080 16 GB, ~3.4 TB free |

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
| perf mem event | `cpu/mem-loads,ldlat=20/P` confirmed available on Tiger Lake |

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

Expected output - look for non-zero samples under `cpu/mem-loads,ldlat=...`:
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

### 5.1 llama-cpp-python (CPU path)

**Minimum version: 0.3.34.** Versions before 0.3.34 used a monolithic shared library and stripped debug symbols, leaving `region_map` empty. Version 0.3.34+ ships split shared libraries (`libllama.so`, `libggml.so`, `libggml-cpu.so`) that export 1,334 `ggml_*` and 281 `llama_kv_cache*` symbols at link time — no debug build required.

```bash
pip install "llama-cpp-python>=0.3.34" \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

Verify:
```bash
python -c "
import llama_cpp, importlib.metadata
ver = importlib.metadata.version('llama-cpp-python')
print(f'llama-cpp-python {ver} OK')
from llama_cpp import Llama
print('import OK')
"
# Expected: llama-cpp-python 0.3.34 (or newer) OK
```

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
hf models ls --search "gemma"
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

**Download commands:**

```bash
BASE_HF="https://huggingface.co"

# 1. TinyLlama 1.1B
wget -P models/ "$BASE_HF/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# 2. Llama-3.2-1B (requires HF login)
wget -P models/ --header="Authorization: Bearer $(cat ~/.cache/huggingface/token)" \
  "$BASE_HF/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"

# 3. Llama-3.2-3B (requires HF login)
wget -P models/ --header="Authorization: Bearer $(cat ~/.cache/huggingface/token)" \
  "$BASE_HF/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

# 4. Phi-3.5-mini (no gate)
wget -P models/ "$BASE_HF/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf"

# 5. Mistral-7B (no gate)
wget -P models/ "$BASE_HF/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"

# 6. Llama-3.1-8B (requires HF login + license)
wget -P models/ --header="Authorization: Bearer $(cat ~/.cache/huggingface/token)" \
  "$BASE_HF/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
```

Or use the HuggingFace CLI (cleaner for gated models):
```bash
huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF \
  Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir models/
```

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

Pre-download:
```bash

python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
  AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct'); \
  print('tokenizer OK')"
```

**VRAM note for Mistral-7B:** With 16 GB VRAM, bf16 weights use ~14 GB leaving ~2 GB for KV cache. Use `--ctx 4096` (not 8192 or 32768) for the GPU path Mistral runs to avoid OOM.

### 6.3 Dev machine models (testing only)

```bash
# Already downloaded
models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Qwen2.5-0.5B - auto-downloaded by demo_qwen_hf.py on first run
python notebooks/scripts/demo_qwen_hf.py --model Qwen/Qwen2.5-0.5B-Instruct
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

# 3. llama-cpp-python
"$MRM_ROOT/venv/bin/python" -c "from llama_cpp import Llama; print('llama-cpp OK')"

# 4. PyTorch GPU
"$MRM_ROOT/venv/bin/python" -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

# 5. Model files
ls -lh models/*.gguf | awk '{print $5, $9}'
# Expected: 6 files totalling ~14 GB

# 6. Temperature baseline (must be < 50°C before starting)
sensors | grep -E "Core|Package|temp"

# 7. Disk space
df -h ~   # must have > 200 GB free for safe operation

# 8. Quick end-to-end test (TinyLlama, 1 prompt, fast)
sudo -E "$MRM_ROOT/venv/bin/python" notebooks/scripts/demo_perf_mem.py \
  --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --ctx 512 --tokens 32 --batch-size 1 --rep 0
# Expected console output (two inline quality gates printed by the pipeline):
#
#   Pre-registered N pages from /proc/maps (M weight mmap, K kv_cache anon)
#
#   Trace quality gate [perf_mem] — PASS        ← Gate 1: latency + cache_level
#     Rows parsed          : 300,000+
#     access_latency_ns    : [████████████████████] 100.0%
#     cache_level          : [████████████████████] 100.0%
#     symbol (non-empty)   : [░░░░░░░░░░░░░░░░░░░░]   0.0%   ← expected
#     region classified    : [░░░░░░░░░░░░░░░░░░░░]   0.0%   ← expected pre-label
#
#   Trace quality gate [perf_mem_labelled] — PASS  ← Gate 2: + region classified
#     region classified    : [████████...] > 5%   ← weight + kv_cache via /proc/maps
#
#   Region map     : (N, 8)   ← non-zero (weight + kv_cache regions)
#   KV lifecycle   : (N, 8)   ← non-zero
#
# If either gate prints FAIL in red, the pipeline exits immediately — the run is
# NOT exported. Fix the issue (check perf version, PMU availability, llama-cpp ≥ 0.3.34)
# before continuing.
RUN_DIR=$(ls -td results/tinyllama-1.1b-q4km-perf/*/ 2>/dev/null | head -1)
cat "$RUN_DIR/metrics.csv"   # should show retention, write_once, read_freq columns,
                              # with rows for region_type: weight, kv_cache, unknown
```

---

## 8. Experiment Matrix

### 8.1 CPU path (primary - perf mem, llama.cpp)

| Dimension | Values | Count |
|---|---|---|
| Models | tinyllama-1.1b, llama-3.2-1b, llama-3.2-3b, phi-3.5-mini, mistral-7b, llama-3.1-8b | 6 |
| Context lengths | 2048, 8192, 32768 | 3 |
| Batch sizes | 1, 4, 8 | 3 |
| Reps per config | 3 (rep 0 = warmup, reps 1–2 = data) | 3 |
| **Total CPU runs** | 6 × 3 × 3 × 3 | **162** |

Output tokens per run: **256** (fixed across all configs).

### 8.2 GPU path (cross-validation - transformers, PyTorch GPU)

| Dimension | Values | Count |
|---|---|---|
| Models | qwen2.5-0.5b, llama-3.2-3b, mistral-7b | 3 |
| Context lengths | 2048, 8192 (Mistral: 4096 max due to VRAM) | 2 |
| Batch sizes | 1, 4 | 2 |
| Reps | 2 (rep 0 = warmup, rep 1 = data) | 2 |
| **Total GPU runs** | 3 × 2 × 2 × 2 | **24** |

### 8.3 Total sweep

| Path | Runs | Est. time per run | Est. total active | With cooling |
|---|---|---|---|---|
| CPU - small (1B) | 54 | 5–15 min | ~9 hrs | ~18 hrs |
| CPU - mid (3B–4B) | 54 | 15–40 min | ~22 hrs | ~44 hrs |
| CPU - large (7B–8B) | 54 | 40–90 min | ~50 hrs | ~100 hrs |
| GPU path | 24 | 5–20 min | ~5 hrs | ~10 hrs |
| **Total** | **186** | - | **~86 hrs** | **~172 hrs** |

**Expected calendar time:** 7–10 days of overnight running (8 hrs/night). Start with small models first - they validate the pipeline and produce usable data fast even if larger runs are interrupted.

### 8.4 Execution order (recommended)

Run in this order to get publishable data as early as possible:

1. TinyLlama 1.1B - all contexts, all batches, all reps (validates full pipeline)
2. Llama-3.2-1B - same
3. Llama-3.2-3B - same
4. Phi-3.5-mini - same
5. GPU path: Qwen2.5-0.5B, Llama-3.2-3B (GPU)
6. Mistral-7B - same
7. Llama-3.1-8B - same
8. GPU path: Mistral-7B (GPU, if VRAM allows)

The experiment runner handles this order automatically.

---

## 9. Thermal Protocol (CRITICAL - no AC)

The i9-11980HK in the Alienware x17 chassis will heat-soak under sustained load. A throttled run is a contaminated run - its retention duration and latency values are artifacts of the thermal state, not the memory access pattern.

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
# passive thermal contribution from GPU power leakage)
sudo nvidia-smi -pl 50   # 50 W out of 150 W max during CPU runs
# For GPU-path runs: sudo nvidia-smi -pl 130
```

### 9.2 Temperature thresholds

| Threshold | Meaning |
|---|---|
| < 60°C | Safe to start a new run |
| 60–75°C | Running, acceptable |
| 75–85°C | Running warm - log but continue |
| > 85°C | **Thermal warning** - run flagged as potentially contaminated |
| > 90°C | **Hard throttle likely** - run flagged as contaminated, discard from primary dataset |

### 9.3 Cooling between runs

The experiment runner enforces:
- **Mandatory 5-minute cooldown** after every run regardless of temperature
- **Extended wait** until CPU package temp < 55°C before starting the next run
- Checks temperature every 60 seconds during the wait

For large model runs (7B–8B @ 32K context): allow 5 minutes between runs.

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
- **Timeout**: hard kill after 2 hours per run (prevents infinite hangs on large models)
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
| 1 | TinyLlama-1.1B (all) | 27 | 27 |
| 2 | Llama-3.2-1B (all) | 27 | 54 |
| 3 | Llama-3.2-3B (all) + GPU small | 27 + 8 | 89 |
| 4 | Phi-3.5-mini (all) | 27 | 116 |
| 5 | Mistral-7B (all) | 27 | 143 |
| 6 | Llama-3.1-8B (partial) | 20 | 163 |
| 7 | Llama-3.1-8B (remainder) + GPU large | 7 + 16 | 186 |

---

## 11. Data Safety and Checkpointing

### 11.1 What is saved and when

The runner saves state after every completed run. Each run's output directory (`results/<model_id>/<run_id>/`) contains all derived artifacts before the next run starts. A crash loses at most one in-progress run.

Raw artifacts (`raw/perf.data`, `raw/perf_script.txt`) are the most expensive to re-collect. They are written by the collector before parsing begins. If the parse fails, raw files are preserved.

### 11.2 Disk space planning

| Artifact | Size per run (approx) | 186 runs |
|---|---|---|
| `trace.parquet` | 50–500 MB (model/context dependent) | 10–50 GB |
| `raw/perf.data` | 200 MB – 2 GB | 50–200 GB |
| `raw/perf_script.txt` | 500 MB – 5 GB | 100–500 GB |
| `metrics.csv` | < 1 MB | < 200 MB |
| `region_map.parquet` | 1–10 MB | < 2 GB |

**Conservative total: 200–700 GB** - the partner's 3.4 TB free is adequate with room to spare.

If disk space becomes tight, `raw/perf_script.txt` can be deleted after parsing completes (it's reproducible from `raw/perf.data` via `perf script`).

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
- Collector: perf mem, perf version, ldlat threshold used
- Region classification: weight regions are identified by diffing `/proc/self/maps` before and after `Llama()` construction — GGUF file mmap ranges → `weight`, new anonymous RW regions ≥ 2 MB → `kv_cache`. Runs where `region_classified < 5%` are rejected by the post-labelling quality gate and excluded from the dataset.
- Quality gate: every run passed two inline gates: (1) pre-labelling — `access_latency_ns` and `cache_level` ≥ 95% populated; (2) post-labelling — `region_classified` ≥ 5%. Runs that failed either gate were discarded and not re-run until the fault was corrected.
- Proxy caveat: *"Retention durations are observed lifetimes under a batch schedule on this CPU/GPU's cache hierarchy. The logical access-pattern structure (write-once ratio, read frequency ordering by region type) transfers across substrates and constitutes first-order evidence for MRM silicon design questions. Absolute timing values are machine-specific. Weight and KV-cache regions are classified per-run via /proc/maps address-range injection; Python runtime allocations appear as region_type=unknown and are excluded from region-level retention comparisons."*
- Schema: column definitions for all parquet files
- Citations: Legtchenko et al. (2025) "Managed-Retention Memory"

### 12.3 For Zenodo (full raw archive with DOI)

Raw `perf.data` files go to Zenodo, not Kaggle. Zenodo gives a DOI for formal citation and handles multi-GB files. Upload after paper is accepted or pre-print is live.

### 12.4 arXiv pre-print (priority claim)

Do not wait for the full 186-run sweep. After nights 1–2 (TinyLlama + Llama-3.2-1B complete), the data is sufficient for an initial pre-print. This establishes the priority date.

---

## 13. Quick Reference

```bash
# Dev machine - test run (tinyllama, fast)
sudo -E "$MRM_ROOT/venv/bin/python" notebooks/scripts/demo_perf_mem.py \
  --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --ctx 2048 --tokens 128 --batch-size 1 --rep 1

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
```

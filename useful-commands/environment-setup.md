# mrm-trace - Environment Setup & Common Commands

---

## Initial setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[test]"
pip install memray
```

---

## Verify installation

```bash
# Check version
python -c "import mrm_trace; print(mrm_trace.__version__)"

# Confirm editable install (should print source tree path, not site-packages)
python -c "import mrm_trace; print(mrm_trace.__file__)"

# Verify key dependencies
python -c "import pydantic, yaml, typer, pandas, pyarrow, pytest; print('OK')"
```

---

## Machine validation (day-one PMU check)

Run this once on any new machine before attempting to collect data.

```bash
# Human-readable report - exits 0 (classified), 1 (degraded/memray only), or 2 (cannot collect)
sudo -E venv/bin/python -m mrm_trace.cli validate --machine

# Or via the CLI entry point (same thing)
sudo mrm-trace validate --machine

# Machine-readable JSON - for CI gates or embedding in run manifests
sudo mrm-trace validate --machine --json

# Check disk space in a specific directory
sudo mrm-trace validate --machine --out results/
```

Exit codes:
- `0` - perf mem collection available (publication-grade)
- `1` - degraded: only memray fallback (no timestamps, empty region_map)
- `2` - cannot collect (non-Linux, WSL2, hidden PMU)

The check runs: OS/kernel, WSL2 detection, perf binary, perf version match, root/paranoid,
CPU vendor + PEBS sysfs markers, dmesg PMU driver, live smoke test (32 MiB strided access),
AC power, governor/turbo, thermal sensors, memray, nvidia-smi, disk space.

---

## Step-by-step test sequence - dev machine

Run through these in order. Each step depends on the previous.

```bash
# --- 0. Activate venv (always first) ---
source venv/bin/activate

# --- 1. PMU gate check ---
sudo mrm-trace validate --machine
# Must exit 0. If exit 1: memray only, no real timestamps. If exit 2: stop.

# --- 2. Unit tests (< 30 s, no I/O) ---
pytest -m unit -q
# Expected: all pass

# --- 3. Integration tests (writes temp files, ~2–5 min) ---
pytest -m "unit or integration" -q
# Expected: all pass

# --- 4. Quick end-to-end demo (TinyLlama, 1 short prompt) ---
sudo -E venv/bin/python notebooks/scripts/demo_perf_mem.py \
  --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --ctx 512 --tokens 32 --batch-size 1 --rep 0
# Expected output includes:
#   Region map     : (N, 8)   ← non-zero row count
#   KV lifecycle   : (N, 8)   ← non-zero row count
#   Wrote results to results/tinyllama-1.1b/<run_id>/

# --- 5. Inspect outputs ---
RUN_DIR=$(ls -td results/tinyllama-1.1b/*/ | head -1)
echo "Run dir: $RUN_DIR"
cat "$RUN_DIR/metrics.csv"
ls -lh "$RUN_DIR"
# Expected files: trace.parquet, region_map.parquet, kv_block_lifecycle.parquet,
#                 metrics.csv, metadata.json, manifest.json, logs/, raw/

# --- 6. Dry-run the experiment runner ---
python notebooks/scripts/experiment_runner.py --dry-run --fast
# Expected: prints the 1 run that --fast mode would execute, exits without running

# --- 7. Run fast experiment mode (1 real run, smallest config) ---
sudo -E venv/bin/python notebooks/scripts/experiment_runner.py --fast
# Expected: completes 1 TinyLlama run and shows status summary
```

---

## Step-by-step test sequence - partner machine

Follow this after completing the Ubuntu + environment setup in `experiment-workplan.md`.

```bash
# --- 1. PMU hard gate (MUST be exit 0 before any collection) ---
sudo mrm-trace validate --machine
# Exit 0 required. Tiger Lake PMU v5 should pass easily.

# --- 2. Basic sanity ---
python -m pytest -m unit -q
mrm-trace validate --config config/default_experiment.yaml

# --- 3. Quick end-to-end (TinyLlama, fast) ---
sudo -E venv/bin/python notebooks/scripts/demo_perf_mem.py \
  --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --ctx 512 --tokens 32 --batch-size 1 --rep 0
# Must show non-zero Region map and KV lifecycle rows, and valid metrics.csv.

# --- 4. Check temperature before starting sweep ---
sensors | grep -E "Package|Core 0"
# Must be < 50°C

# --- 5. Start the full sweep ---
sudo -E venv/bin/python notebooks/scripts/experiment_runner.py
# Runs all 186 runs with thermal management and crash-safe state.

# --- 6. Check progress (any time) ---
python notebooks/scripts/experiment_runner.py --status

# --- 7. Resume after interruption ---
sudo -E venv/bin/python notebooks/scripts/experiment_runner.py
# Completed runs are skipped automatically.
```

---

## Running tests

```bash
pytest -m unit -q                   # every commit
pytest -m "unit or integration" -q  # pre-merge
pytest -m validity -q               # before dataset release
pytest tests/property/ -q           # property-based invariant tests
pytest -m performance --benchmark-only  # weekly / before paper submission
```

---

## Demo scripts

### Install backends

```bash
# memray (required for fallback collector)
pip install memray

# TinyLlama via llama-cpp-python (GGUF - needs model download, see below)
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Qwen via HuggingFace transformers (auto-downloads ~1 GB on first run)
pip install transformers torch accelerate
```

### Download TinyLlama model (one-time, ~670 MB)

```bash
mkdir -p models
wget -P models/ \
  https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

---

### demo_qwen_hf.py - Qwen2.5 (auto-detect collector)

The script auto-selects the best available collector at startup:
- Bare-metal Linux + root + PMU available → **perf mem** (spawn mode, real timestamps)
- WSL2 / non-root / PMU unavailable → **memray** (Python-level, timestamps = 0)

```bash
# Non-root or WSL2 → memray (timestamps = 0, region_map empty)
python notebooks/scripts/demo_qwen_hf.py

# Bare-metal Linux as root → perf mem (real hw timestamps, region_map still empty†)
# IMPORTANT: sudo -E python resolves to SYSTEM Python, not venv - use full path:
sudo -E venv/bin/python notebooks/scripts/demo_qwen_hf.py
# or absolute path:
sudo -E /home/oj/projects/mrm-trace/venv/bin/python notebooks/scripts/demo_qwen_hf.py

# Force a specific collector:
python notebooks/scripts/demo_qwen_hf.py --collector memray
sudo -E venv/bin/python notebooks/scripts/demo_qwen_hf.py --collector perf-mem

# Larger model:
python notebooks/scripts/demo_qwen_hf.py --model Qwen/Qwen2.5-1.5B-Instruct
```

† region_map is empty because PyTorch symbols don't match ggml_* labeller rules.
  Retention and write-once metrics ARE real (from hardware PMU timestamps).
  For classified region_map, use demo_perf_mem.py with a GGUF model.

---

### demo_tinyllama.py - TinyLlama (memray)

```bash
# Non-root (Python-level only)
python notebooks/scripts/demo_tinyllama.py \
  --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Root - C-level allocs (region_map still empty with pip llama-cpp-python)
sudo -E venv/bin/python notebooks/scripts/demo_tinyllama.py \
  --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --native-traces
```

---

### demo_perf_mem.py - TinyLlama (perf mem, populated region_map)

Requires: bare-metal Linux, root, hardware PMU, GGUF model.
Does NOT work on WSL2.

```bash
# Check PMU availability first (should exit 0 with no error)
sudo perf mem record -o /dev/null -- true && echo "PMU OK"

# Run (use full venv path - sudo -E python resolves to system Python)
sudo -E venv/bin/python notebooks/scripts/demo_perf_mem.py \
  --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

For a populated region_map, you need a debug-symbol build of llama-cpp-python:
```bash
CMAKE_BUILD_TYPE=Debug pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

---

## perf mem troubleshooting

```bash
# Full automated check (preferred - covers all failure modes)
sudo mrm-trace validate --machine

# Manual checks if you need to isolate a specific failure:

# Install perf if missing
sudo apt update && sudo apt install linux-perf -y

# Verify perf version matches kernel (mismatch = harmless warning usually)
perf --version
uname -r

# Check if spawn mode works on this machine
sudo perf mem record -o /dev/null -- sleep 0.1 && echo "spawn mode OK"
# WSL2: will fail - Microsoft kernel does not expose PMU
# Bare-metal Intel/AMD: should pass

# Check what PMU events are available
perf list | grep -i "mem-load\|mem-store"

# Check paranoid level (root bypasses this, but good to know)
cat /proc/sys/kernel/perf_event_paranoid

# Check PEBS sysfs markers (Intel)
ls /sys/bus/event_source/devices/cpu/format/ldlat 2>/dev/null && echo "ldlat OK"
ls /sys/bus/event_source/devices/cpu/events/mem-loads 2>/dev/null && echo "mem-loads OK"
```

---

## Build and release

```bash
# Lint
ruff check mrm_trace/ --select E,F,W --ignore E501

# Build wheel
pip install build && python -m build

# Tag release (version must match pyproject.toml)
git tag v0.1.8
git push origin v0.1.8
```

# mrm-trace — Environment Setup & Common Commands

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

# TinyLlama via llama-cpp-python (GGUF — needs model download, see below)
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

### demo_qwen_hf.py — Qwen2.5 (auto-detect collector)

The script auto-selects the best available collector at startup:
- Bare-metal Linux + root + PMU available → **perf mem** (spawn mode, real timestamps)
- WSL2 / non-root / PMU unavailable → **memray** (Python-level, timestamps = 0)

```bash
# Non-root or WSL2 → memray (timestamps = 0, region_map empty)
python notebooks/scripts/demo_qwen_hf.py

# Bare-metal Linux as root → perf mem (real hw timestamps, region_map still empty†)
# IMPORTANT: sudo -E python resolves to SYSTEM Python, not venv — use full path:
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

### demo_tinyllama.py — TinyLlama (memray)

```bash
# Non-root (Python-level only)
python notebooks/scripts/demo_tinyllama.py \
  --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Root — C-level allocs (region_map still empty with pip llama-cpp-python)
sudo -E venv/bin/python notebooks/scripts/demo_tinyllama.py \
  --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --native-traces
```

---

### demo_perf_mem.py — TinyLlama (perf mem, populated region_map)

Requires: bare-metal Linux, root, hardware PMU, GGUF model.
Does NOT work on WSL2.

```bash
# Check PMU availability first (should exit 0 with no error)
sudo perf mem record -o /dev/null -- true && echo "PMU OK"

# Run (use full venv path — sudo -E python resolves to system Python)
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
# Install perf if missing
sudo apt update && sudo apt install linux-perf -y

# Check if perf mem spawn mode works on this machine
sudo perf mem record -o /dev/null -- sleep 0.1 && echo "spawn mode OK"

# WSL2: will fail — Microsoft kernel does not expose PMU
# Bare-metal Intel/AMD: should pass

# Check what PMU events are available
perf list | grep -i "mem-load\|mem-store"

# perf_event_paranoid must be ≤ 2 for non-root perf (root bypasses this)
cat /proc/sys/kernel/perf_event_paranoid
```

---

## Build and release

```bash
# Lint
ruff check mrm_trace/ --select E,F,W --ignore E501

# Build wheel
pip install build && python -m build

# Tag release (version must match pyproject.toml)
git tag v0.1.7
git push origin v0.1.7
```

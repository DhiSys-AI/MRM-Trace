python3 -m venv venv
source venv/bin/activate

pip install pydantic pyyaml typer rich pandas pyarrow psutil numpy pytest pytest-cov pytest-mock freezegun hypothesis


# Verify key dependencies are installed
python -c 'import pydantic, yaml, typer, pandas, pytest;

# Run full test suite
python -m pytest tests/ -v --tb=short

python -m pytest tests/unit/config/test_validators.py::test_expand_sweep_correct_total -v --tb=long

# lint
ruff check mrm_trace/ --select E,F,W --ignore E501

# CI-equivalent test runs
pytest tests/unit/ -q --tb=short
pytest tests/integration/ -q --tb=short
pytest tests/validity/ -q --tb=short
pytest tests/property/ -q --tb=short

# Tag the version (must match pyproject.toml version field)
git tag v0.1.6
git push origin v0.1.6


# build options:
Goal	Command
Dev testing (editable, changes instant)	already done: 
`pip install -e ".[test]"`

## Run specific tests	
pytest tests/unit/parser/test_memray_parser.py -v

## Run full unit suite	
pytest -m unit -q

## Build wheel	
pip install build && python -m build

## Install wheel locally	
pip install dist/mrm_trace-0.1.2-py3-none-any.whl --force-reinstall


# To run on Linux / WSL:
## One-time setup
pip install memray

## TinyLlama (needs GGUF download first — see README)
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

### download model
mkdir -p models
wget -P models/ \
  https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf


python notebooks/scripts/demo_tinyllama.py --native-traces \
  --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf



## Qwen (auto-downloads ~1 GB on first run)
pip install transformers torch accelerate
python notebooks/scripts/demo_qwen_hf.py

pip install transformers torch accelerate

python -c \"from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', low_cpu_mem_usage=True)\""


# Check version (should show 0.1.3)
python -c "import mrm_trace; print(mrm_trace.__version__)"

# Check which file (editable = source tree, not site-packages)
python -c "import mrm_trace; print(mrm_trace.__file__)"

# Verify the manifest fix (the key change from this session)
python -c "
from mrm_trace.reporter.manifest import build_manifest
import inspect
src = inspect.getsource(build_manifest)
print('manifest fix: OK' if 'not f.exists()' in src else 'STALE — run: pip install -e .[test]')
"

# Verify the engine reset fix
python -c "
from mrm_trace.engines.llamacpp import LlamaCppEngine
import inspect
src = inspect.getsource(LlamaCppEngine._generate_via_bindings)
print('engine reset fix: OK' if 'reset' in src else 'STALE — run: pip install -e .[test]')
"


# Q2 — Running with perf mem:

## perf not found — install it:
sudo apt update
sudo apt install linux-perf -y

## Check if perf mem is supported
## Then test PMU availability:
sudo perf stat true

## What to expect on WSL2:
WSL2 uses Microsoft's custom kernel. Hardware PMU counters are almost never forwarded to the guest, so even after installing perf you'll likely see:
sudo perf stat -e cpu/mem-loads,ldlat=20/P -- sleep 0.1


## 1. Check PMU is available (bare-metal Linux or WSL2 with PMU forwarding)
sudo perf stat -e cpu/mem-loads,ldlat=20/P true

## 2. Run the demo (must be root; -E preserves the venv)
sudo -E python notebooks/scripts/demo_perf_mem.py \
  --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

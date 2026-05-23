## Package Name: `mrm-trace`

```
pip install mrm-trace
```

A research toolchain for collecting, parsing, labelling, and analysing LLM inference memory access traces, designed specifically to produce the MRM-relevant metrics (retention duration, write-once ratio, read frequency) that no existing tool reports.

***

## Package Structure

```
mrm-trace/
│
├── pyproject.toml                  ← pip-installable, declares CLI entry points
├── README.md
├── docs/
│
└── mrm_trace/
    ├── __init__.py
    │
    ├── cli.py                      ← `mrm-trace run`, `mrm-trace analyse`, `mrm-trace report`
    │
    ├── engines/                    ← Inference engine drivers
    │   ├── __init__.py
    │   ├── base.py                 ← Abstract BaseEngine class
    │   ├── llamacpp.py             ← Wraps llama-cpp-python [web:292]
    │   └── vllm.py                 ← Wraps vLLM Python API
    │
    ├── collector/                  ← Memory trace capture
    │   ├── __init__.py
    │   ├── perf_runner.py          ← Shells out to `perf mem record`, manages lifecycle
    │   ├── memray_runner.py        ← Bloomberg Memray fallback for Python-level allocations [web:285]
    │   └── process_monitor.py      ← psutil RSS/VMS polling (lightweight fallback)
    │
    ├── parser/                     ← Raw trace → structured data
    │   ├── __init__.py
    │   ├── perf_script_parser.py   ← Parses `perf script` stdout into DataFrames [web:288]
    │   ├── memray_parser.py        ← Parses Memray binary output
    │   └── schema.py               ← Shared Pydantic schema: TraceRecord, RegionType
    │
    ├── labeller/                   ← Assign region types to addresses
    │   ├── __init__.py
    │   ├── symbol_labeller.py      ← Maps symbol names → weight/kv_cache/activation/other
    │   ├── address_tracker.py      ← Tracks allocation/free events per address range
    │   └── region_map.py           ← Configurable regex rules per engine/model family
    │
    ├── analyser/                   ← Compute MRM-specific metrics
    │   ├── __init__.py
    │   ├── retention.py            ← Retention duration distribution per region type
    │   ├── write_once.py           ← Write-once ratio per region type
    │   ├── read_frequency.py       ← Read Hz per region type, over context length
    │   └── working_set.py          ← Working set size over time
    │
    ├── reporter/                   ← Output and visualisation
    │   ├── __init__.py
    │   ├── csv_exporter.py         ← Exports Parquet + CSV datasets
    │   ├── plot.py                 ← Matplotlib/seaborn figures for paper
    │   └── summary.py              ← Prints terminal summary table
    │
    └── config/
        ├── __init__.py
        ├── models.yaml             ← Known model families and their symbol patterns
        └── experiments.yaml        ← Sweep definitions (context lengths, batch sizes)
```


***

## What Each Layer Does

### `engines/` — Unified Inference Control

Wraps both **llama-cpp-python** [^1] and **vLLM** behind a common `BaseEngine` interface. Each engine driver handles model loading, prompt submission, and — critically — **exposes timing hooks**: `on_model_load_start`, `on_model_load_end`, `on_generation_token`, `on_kv_cache_alloc`. These hooks timestamp phase boundaries so the collector can split traces into weight-load vs. inference phases automatically.

```python
# Example usage
from mrm_trace.engines.llamacpp import LlamaCppEngine

engine = LlamaCppEngine(model_path="llama-3-8b.gguf", context_length=4096)
with engine.timed_session() as session:
    session.generate("Explain quantum computing", max_tokens=512)
```


### `collector/` — Trace Capture

The `perf_runner` shells out to `sudo perf mem record` with the engine's PID, captures `perf.data`, then calls `perf script` to stream raw samples [^2][^3]. The **memray_runner** uses Bloomberg's Memray [^4] as a Python-level fallback for systems without hardware PMU support, capturing allocation/free events. The `process_monitor` polls `psutil` at configurable intervals as a coarse-grained fallback.

The collector is **phase-aware** — it receives the timing events from the engine layer and tags every sample with the phase in which it was captured.

### `parser/` — Raw → Structured

The `perf_script_parser` reads the text output of `perf script -F time,pid,addr,sym,dso` line-by-line using a regex FSM [^5], streaming into Pandas DataFrames without loading the full trace into memory at once. Output rows conform to the `TraceRecord` Pydantic schema: `(timestamp_ns, address, symbol, dso, op_type, phase)`.

### `labeller/` — Region Type Assignment

This is the most research-critical module. It assigns each memory address range to a **region type** (weight, kv_cache, activation, runtime_other) using two complementary approaches:

1. **Symbol-based**: regex rules matching `ggml_*` → weight, `llama_kv_cache_*` → kv_cache [^6]
2. **Address range tracking**: monitors allocation events from Memray to assign contiguous address ranges to region types at alloc-time, then labels all subsequent accesses to that range

The `region_map.yaml` config makes these rules extensible for new models without code changes.

### `analyser/` — MRM Metrics

This is where the novel contribution lives. Four sub-modules compute the metrics the MRM paper requires but which no prior dataset provides:

- **`retention.py`** — For each (address, region\_type), computes `last_access_time - first_access_time` and returns the distribution as a histogram and CDF
- **`write_once.py`** — Counts addresses where `write_count == 1` and `read_count > 0`, reports this as a fraction of total allocated bytes per region type
- **`read_frequency.py`** — Computes reads-per-second per region type, parameterised by context length and batch size
- **`working_set.py`** — Computes the live byte count over time (bytes resident between first write and last access)


### `reporter/` — Dataset and Paper Figures

Exports Parquet files (for dataset release) and generates the exact figures needed for the paper: retention duration CDF, write-once ratio bar chart, read frequency heat map by context length, and working set size over time.

***

## CLI Interface

```bash
# Run a full sweep (single model, multiple context lengths)
mrm-trace run \
  --engine llamacpp \
  --model llama-3-8b.gguf \
  --context-lengths 512,2048,8192,32768 \
  --prompts prompts/standard_suite.jsonl \
  --output traces/llama3-8b/

# Analyse a captured trace directory
mrm-trace analyse \
  --traces traces/llama3-8b/ \
  --metrics retention,write-once,read-frequency \
  --output results/llama3-8b/

# Generate paper figures
mrm-trace report \
  --results results/ \
  --format pdf \
  --output figures/
```


***

## Key Design Decisions

| Decision | Rationale |
| :-- | :-- |
| **Streaming parser** | `perf.data` for a 70B model inference run can exceed 50 GB — must stream, not load into RAM |
| **Phase-aware capture** | Without phase tagging, weight-load and KV-cache traces are indistinguishable by address alone |
| **Memray fallback** [^4] | `perf mem` requires hardware PMU which cloud VMs often disable; Memray works universally |
| **YAML region maps** | Different model families use different internal symbol naming; config-driven beats hardcoded |
| **Parquet output** | Column-oriented storage handles billions of trace rows efficiently; compatible with HuggingFace Datasets for release |
| **Single `pip install`** | Reproducibility — any researcher can replicate your exact dataset with one command and one YAML config file |


***

## What a Researcher Would Do With This Package in Practice

```python
from mrm_trace import Experiment

# Define the experiment
exp = Experiment.from_yaml("experiments/llama3_sweep.yaml")

# Run all engines × models × context lengths
exp.run()           # captures traces (~hours)

# Compute all MRM metrics
results = exp.analyse()

# Export dataset + figures
exp.report(output_dir="./paper_output/")
```

A single YAML config file fully specifies the experiment, making the entire dataset **reproducible by any reader** of your paper — which is now an explicit requirement at MLSys, ASPLOS, and USENIX ATC for artifact evaluation badges.
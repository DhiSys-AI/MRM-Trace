# MRM Trace Platform Specification

## Purpose

This document specifies the end-to-end design, implementation plan, data contracts, telemetry model, testing approach, and delivery requirements for a Python package tentatively named `mrm-trace`. The package is intended to collect, parse, label, analyse, validate, and export memory access traces for LLM inference workloads in a way that supports Managed-Retention Memory (MRM) research.

The package must produce scientifically defensible outputs for research use, not just engineering observability. It must therefore treat trace correctness, reproducibility, observer effect, machine context, and run validity as first-class outputs.

The intended primary use cases are:

- Collect CPU-side memory access traces for LLM inference runs.
- Distinguish and label memory behaviour for model weights, KV cache, activations, and runtime allocations.
- Compute MRM-relevant metrics such as retention duration, write-once ratio, read frequency, and working set size.
- Record complete machine context and observer overhead so that each run is independently auditable.
- Export dataset artifacts suitable for publication, reproduction, and later simulator development.

---

## Product Goals

### Primary goals

1. Provide a single installable Python package and CLI for running trace experiments.
2. Support at minimum `llama.cpp` and `vLLM` as inference engines.
3. Support `perf mem` as the primary collector where PMU support is available.
4. Support `memray` and coarse process monitoring as fallbacks or supplementary collection paths.
5. Record complete hardware, software, thermal, process, and observer telemetry per run.
6. Generate raw trace files, region maps, KV lifecycle files, aggregated metrics, metadata sidecars, and publication-ready derived outputs.
7. Include a rigorous test suite covering functional correctness, scientific validity, performance, and regression stability.

### Secondary goals

1. Make experiments reproducible through YAML configuration.
2. Make region labelling extensible through rules files rather than hardcoded logic.
3. Make all outputs machine-readable and suitable for downstream analysis.
4. Provide clean separation between trace collection, trace interpretation, and metric computation.
5. Support future extension into a simulator-backed workflow for MRM hardware studies.

### Non-goals for initial release

1. GPU HBM tracing.
2. Full kernel-level memory coherence modelling.
3. Production dashboard UI.
4. Cross-machine distributed orchestration.
5. Hardware-in-the-loop MRM emulation.

---

## Research Framing

This package exists to fill the empirical gap between existing LLM inference characterisation work and the requirements of MRM research. Existing work has examined NUMA locality, SSD offload I/O, and CPU cache behaviour, but does not provide the core metrics needed for MRM design, including retention duration distribution, write-once ratio, and read frequency by region type across engines and workloads.[cite:231][cite:260][cite:258][cite:134]

The platform therefore must be designed as a **measurement system**, not just a profiling wrapper. In practice this means:

- every run must be accompanied by full metadata,
- the tracing tool's own overhead must be observed and recorded,
- invalid or contaminated runs must be flagged,
- metric definitions must be explicit and testable,
- the raw observed data must remain available for audit.

---

## Scope of Deliverables

The Dev team is expected to deliver the following.

### Software deliverables

1. Installable Python package `mrm-trace`.
2. CLI entry point `mrm-trace`.
3. Python API for programmatic experiment execution.
4. Config-driven experiment runner.
5. Trace collection subsystem.
6. Parsing subsystem.
7. Labelling subsystem.
8. Analysis subsystem.
9. Telemetry subsystem.
10. Reporting/export subsystem.
11. Comprehensive automated test suite.
12. Developer documentation and runbook.

### Data deliverables per run

1. `trace.parquet`
2. `region_map.parquet`
3. `kv_block_lifecycle.parquet`
4. `metrics.csv`
5. `metadata.json`
6. collector-native files, e.g. `perf.data`, only if retention policy allows
7. logs and validation outputs

### Documentation deliverables

1. package architecture document
2. CLI usage guide
3. YAML schema reference
4. data schema reference
5. telemetry reference
6. run validity and contamination policy
7. testing guide
8. artifact packaging guide for publication

---

## System Overview

At a high level the platform runs a configured experiment, launches an inference workload through a supported engine, collects memory access traces and concurrent telemetry, parses and labels the resulting data, computes MRM metrics, classifies run validity, and writes reproducible artifacts.

### End-to-end flow

1. Load experiment YAML.
2. Validate schema.
3. Capture pre-run machine baseline.
4. Resolve engine, model, prompts, and collector.
5. Start telemetry background samplers.
6. Start collector.
7. Launch engine workload.
8. Record phase boundaries.
9. Stop engine workload.
10. Stop collector.
11. Stop telemetry samplers.
12. Parse raw collector output.
13. Label memory regions.
14. Compute region-level and run-level metrics.
15. Compute observer effect and run validity.
16. Export raw, intermediate, and derived outputs.
17. Emit summary logs and manifests.

---

## Package Structure

The following structure is required.

```text
mrm-trace/
├── pyproject.toml
├── README.md
├── docs/
├── config/
│   ├── models.yaml
│   ├── validity_rules.yaml
│   └── default_experiment.yaml
├── tests/
└── mrm_trace/
    ├── __init__.py
    ├── cli.py
    ├── api.py
    ├── config/
    │   ├── __init__.py
    │   ├── schema.py
    │   ├── loader.py
    │   └── validators.py
    ├── engines/
    │   ├── __init__.py
    │   ├── base.py
    │   ├── llamacpp.py
    │   └── vllm.py
    ├── collector/
    │   ├── __init__.py
    │   ├── base.py
    │   ├── perf_runner.py
    │   ├── memray_runner.py
    │   ├── process_monitor.py
    │   └── artifact_manager.py
    ├── parser/
    │   ├── __init__.py
    │   ├── schema.py
    │   ├── perf_script_parser.py
    │   ├── memray_parser.py
    │   └── normalizer.py
    ├── labeller/
    │   ├── __init__.py
    │   ├── symbol_labeller.py
    │   ├── address_tracker.py
    │   ├── phase_labeller.py
    │   └── region_map.py
    ├── analyser/
    │   ├── __init__.py
    │   ├── retention.py
    │   ├── write_once.py
    │   ├── read_frequency.py
    │   ├── working_set.py
    │   ├── locality.py
    │   ├── inter_access_interval.py
    │   └── suitability.py
    ├── telemetry/
    │   ├── __init__.py
    │   ├── hardware.py
    │   ├── software.py
    │   ├── thermal.py
    │   ├── baseline.py
    │   ├── observer.py
    │   ├── collector_meta.py
    │   └── snapshot.py
    ├── reporter/
    │   ├── __init__.py
    │   ├── csv_exporter.py
    │   ├── parquet_exporter.py
    │   ├── plot.py
    │   ├── summary.py
    │   └── manifest.py
    ├── orchestration/
    │   ├── __init__.py
    │   ├── experiment_runner.py
    │   ├── run_executor.py
    │   ├── phase_controller.py
    │   └── validity.py
    └── utils/
        ├── __init__.py
        ├── subprocess.py
        ├── clocks.py
        ├── files.py
        ├── ids.py
        └── logging.py
```

---

## Runtime Architecture

### Core architectural principles

1. **Collector isolation**: collector components must be clearly separated from analysis logic.
2. **Schema-first design**: all internal data objects must have defined schemas.
3. **Phase-awareness**: all collected signals must be alignable to workload phases.
4. **Observer accountability**: tracing overhead must be treated as measurable data.
5. **Streaming-first parsing**: large trace files must never require full in-memory loading.
6. **Deterministic output layout**: output directories and file names must be reproducible.
7. **Graceful degradation**: if `perf mem` is unavailable, the system must fall back and record why.

---

## Functional Requirements

## 1. Experiment Configuration

### Requirements

The system shall accept experiment definitions in YAML.

The YAML must define:

- experiment metadata
- model set
- engine mapping
- sweep parameters
- prompt source
- collector configuration
- telemetry configuration
- analysis metrics
- output formats
- labelling rules source
- validity policy reference

### Validation requirements

1. Reject invalid engine names.
2. Reject missing model paths or unresolved repositories.
3. Reject invalid context lengths, batch sizes, or repetitions.
4. Reject unknown metrics.
5. Reject unsupported collector events.
6. Validate all file paths before run scheduling.
7. Materialise each sweep combination as an explicit run plan.

### Example YAML

```yaml
experiment:
  name: "llama3_mrm_sweep_v1"
  description: "MRM retention characterisation across LLM engines"
  version: "1.0.0"
  date_created: "2026-05-23"

models:
  - id: "llama3-8b-q4"
    engine: "llamacpp"
    path: "models/llama-3-8b.Q4_K_M.gguf"
    family: "llama3"
    params_billions: 8
    quantisation: "Q4_K_M"
    dtype: "float16"

  - id: "llama3-8b-vllm"
    engine: "vllm"
    hf_repo: "meta-llama/Meta-Llama-3-8B"
    family: "llama3"
    params_billions: 8
    dtype: "float16"

sweep:
  context_lengths: [512, 2048, 8192, 32768]
  batch_sizes: [1, 4, 8]
  output_tokens: [128, 512]
  repetitions: 3
  warmup_runs: 1

prompts:
  source: "prompts/standard_suite.jsonl"

collector:
  primary: "perf_mem"
  fallback: "memray"
  fallback_2: "process_monitor"
  perf_mem:
    sample_period: 4000
    ldlat_threshold: 20
    events:
      - "cpu/mem-loads,ldlat=20/P"
      - "cpu/mem-stores/P"
    extra_flags: ["--call-graph", "dwarf"]
    require_root: true
  memray:
    native_traces: true
    follow_fork: true
    file_size_limit_gb: 20
  process_monitor:
    poll_interval_ms: 100
    metrics: ["rss", "vms", "shared", "text", "data"]

telemetry:
  capture: true
  thermal_sampling_interval_s: 5
  thermal_source: "auto"
  invalidate_on_throttle: true
  software_versions: true
  hardware_topology: true
  run_fio_benchmark: true

labeller:
  rules_file: "config/models.yaml"
  unknown_policy: "log_and_skip"
  min_sample_count: 10

analyser:
  metrics:
    - retention_duration
    - write_once_ratio
    - read_frequency
    - working_set_size
    - access_locality
    - inter_access_interval
  histogram_bins: 100
  cdf_resolution: 1000
  group_by: ["region_type", "context_length", "model_family", "batch_size"]

output:
  base_dir: "results/"
  formats:
    raw_traces: "parquet"
    metrics: "csv"
    figures: "pdf"
    metadata: "json"
  compression: "zstd"
```

---

## 2. Engine Support

### Shared engine interface

All supported engines must implement a shared interface.

#### Required methods

- `prepare()`
- `load_model()`
- `warmup()`
- `generate()`
- `shutdown()`
- `get_target_pids()`
- `get_runtime_metadata()`

#### Required emitted phase events

- `model_load_start`
- `model_load_end`
- `generation_start`
- `generation_end`
- `teardown_start`
- `teardown_end`
- optional `kv_cache_alloc`
- optional `token_emitted`

### `llama.cpp` requirements

1. Support local GGUF models.
2. Support CPU-only mode.
3. Support prompt-driven generation with configurable context length and output tokens.
4. Expose process ID and child process IDs if applicable.
5. Record model runtime flags used.
6. Support both CLI invocation and Python bindings where appropriate.[cite:292]

### `vLLM` requirements

1. Support Python API invocation for local or fetched model weights.
2. Support batch workloads.
3. Record vLLM, torch, CUDA status, even for CPU-only scenarios.
4. Expose worker processes where used.
5. Emit timing boundaries around generate calls.[cite:243]

### Engine constraints

1. Engine wrappers must not embed analysis logic.
2. Engine wrappers must produce enough metadata to correlate trace windows to inference phases.
3. All engine-specific output must be normalised into common schemas.

---

## 3. Collector Subsystem

### Primary collector: `perf mem`

`perf mem` is the preferred collector where hardware PMU support is available because it captures memory loads and stores with hardware assistance.[cite:230][cite:236]

#### Requirements

1. Detect `perf` availability and version.
2. Detect PMU support before run.
3. Detect permissions or root requirement.
4. Start collector before engine enters traced phase.
5. Stop collector cleanly after workload completion.
6. Persist raw `perf.data`.
7. Convert or expose scriptable form via `perf script` and/or `perf mem report`.
8. Record dropped samples, event definitions, sampling rate, and collector success status.

#### Standard invocation pattern

```bash
sudo perf mem record -t load -e cpu/mem-loads,ldlat=20/P -o weight_load_trace.perf.data -- <target command>
```

#### Data to record about collector

- collector type used
- event strings
- sample period
- `ldlat` threshold
- root requirement
- fallback trigger reason if primary collector not used
- output file size
- parse success/failure
- coverage estimate
- dropped samples

### Secondary collector: `memray`

Use Memray when PMU is unavailable or when Python-level allocation visibility is needed.[cite:285]

#### Requirements

1. Must support native trace mode where feasible.
2. Must support follow-fork where collector can observe child processes.
3. Must record file size growth and buffer pressure.
4. Must expose allocation/free events and timestamps where possible.
5. Must record limitations relative to `perf mem`.

### Supplementary collector: process monitor

Always run a coarse process monitor in parallel, even if `perf mem` or `memray` is active.

#### Requirements

Sample at configurable interval and record:

- RSS
n- VMS
- shared memory
- thread count
- open file descriptors
- read bytes
- write bytes
- CPU percent
- page faults if available
- voluntary/involuntary context switches

---

## 4. Parsing Subsystem

### Goals

Convert collector-native output into a common trace schema suitable for downstream labelling and analysis.

### Requirements

1. Support streaming parse of `perf script` output.[cite:290][cite:299]
2. Support malformed line handling and logging.
3. Support schema normalisation from multiple collector sources.
4. Support enrichment with phase windows and run identifiers.
5. Avoid loading full trace into memory where possible.

### Canonical trace schema

Each row in `trace.parquet` must contain the following canonical fields.

| Column | Type | Description |
|---|---|---|
| run_id | string | Unique run identifier |
| sample_id | uint64 | Monotonic sample counter |
| timestamp_ns | uint64 | Nanoseconds since run start |
| phase | category | `weight_load`, `generation`, `teardown` |
| address | uint64 | Virtual address |
| address_page | uint64 | Page-aligned address |
| op_type | category | `load`, `store` |
| symbol | string | Resolved symbol |
| dso | string | Binary or shared object |
| region_type | category | `weight`, `kv_cache`, `activation`, `runtime`, `unknown` |
| region_id | uint32 | Region mapping identifier |
| access_latency_ns | uint32/null | Latency if available |
| cache_level | category/null | `L1`, `L2`, `L3`, `DRAM`, `null` |
| numa_node | uint8/null | Memory node |
| cpu_id | uint16/null | Logical CPU |
| thread_id | uint32/null | Thread identifier |
| token_index | uint32/null | Token index if derivable |
| context_tokens | uint32/null | Context size at sample time |

### Parsing acceptance criteria

1. Parse one million synthetic rows without schema drift.
2. Maintain streaming memory profile.
3. Correctly reject or log malformed rows without crashing the run.
4. Preserve source-specific metadata needed for debugging.

---

## 5. Labelling Subsystem

### Purpose

Assign semantic meaning to addresses and symbols so that analysis can distinguish model weights from KV cache and other memory regions.

### Labelling outputs

1. `region_type` on trace rows.
2. `region_map.parquet` containing allocation-level mappings.
3. `kv_block_lifecycle.parquet` for KV-specific lifecycles.

### Required region classes

- `weight`
- `kv_cache`
- `activation`
- `runtime`
- `unknown`

### Labelling methods

#### Symbol-based labelling

Map symbols and DSOs using configurable regex rules.

Examples:

- `ggml_*` patterns may indicate weight reads in `llama.cpp`.
- `llama_kv_cache_*` patterns indicate KV operations.
- engine/runtime internals may map to `runtime`.

#### Address-range tracking

Where allocation events are visible, assign address intervals to semantic region types and use interval matching for subsequent accesses.

#### Phase-aware labelling

Use workload phases to disambiguate.

Examples:

- accesses during `weight_load` are likely weight-related,
- short-lived writes during `generation` may be KV or activation,
- teardown-only accesses should not pollute generation analysis.

### `region_map.parquet` schema

| Column | Type | Description |
|---|---|---|
| region_id | uint32 | Primary key |
| region_type | category | Region classification |
| addr_start | uint64 | Allocation start |
| addr_end | uint64 | Allocation end |
| size_bytes | uint64 | Allocation size |
| alloc_timestamp_ns | uint64/null | Allocation time |
| free_timestamp_ns | uint64/null | Free time |
| first_access_ns | uint64 | First observed access |
| last_access_ns | uint64 | Last observed access |
| retention_duration_ns | uint64 | Last minus first access |
| write_count | uint64 | Total stores |
| read_count | uint64 | Total loads |
| is_write_once | bool | True when written once then read |
| symbol_dominant | string/null | Dominant symbol |
| layer_index | int32/null | Transformer layer if derivable |
| weight_type | category/null | `q_proj`, `k_proj`, `v_proj`, `ffn`, `embed`, `other` |

### `kv_block_lifecycle.parquet` schema

| Column | Type | Description |
|---|---|---|
| block_id | string | Unique block ID |
| layer_index | int32 | Transformer layer |
| head_index | int32/null | Attention head |
| token_start | int32 | Token range start |
| token_end | int32 | Token range end |
| alloc_ns | uint64 | Allocation time |
| first_read_ns | uint64/null | First read |
| last_read_ns | uint64/null | Last read |
| free_ns | uint64/null | Free time |
| lifetime_ns | uint64/null | Lifetime |
| active_read_window_ns | uint64/null | Last read minus first read |
| read_count | uint64 | Read count |
| context_length_at_alloc | uint32 | Context depth |

### Unknown handling policy

Unknown or weakly labelled symbols must not be silently coerced. The YAML must allow one of:

- `log_and_skip`
- `assign_runtime`
- `assign_unknown`
- `fail_run`

Initial default should be `log_and_skip`.

---

## 6. Analysis Subsystem

### Purpose

Compute the metrics needed for MRM-oriented research and export them in machine-readable form.

### Required metrics

#### 6.1 Retention duration

For each region or block, compute:

\[
\text{retention duration} = \text{last access timestamp} - \text{first access timestamp}
\]

Aggregate by region type, context length, model, batch size, engine, and run validity.

Required summary outputs:

- p50
- p90
- p99
- max
- mean
- standard deviation

#### 6.2 Write-once ratio

For each region set, compute:

- fraction of regions with exactly one write and one or more reads
- fraction of bytes belonging to such regions

#### 6.3 Read frequency

Compute reads per second and reads per output token for each region type.

#### 6.4 Working set size

Track bytes considered live over time based on allocation and last access.

#### 6.5 Access locality

Measure sequential versus random access behaviour. At minimum compute:

- sequential access ratio
- dominant stride size
- random access ratio

#### 6.6 Inter-access interval

Compute time between consecutive accesses to same address or region.

#### 6.7 MRM suitability score

Compute a derived heuristic score for each region class or run based on retention duration, write-once ratio, read intensity, and locality. This score is for prioritisation only and must be clearly marked as heuristic.

### Aggregated metrics output

`metrics.csv` must contain at minimum the following fields.

| Column | Description |
|---|---|
| run_id | Unique run ID |
| model_id | Model identifier |
| engine | Engine name |
| context_length | Context length |
| batch_size | Batch size |
| output_tokens | Output token count |
| region_type | Region class |
| rep_index | Repetition index |
| run_validity | `clean`, `marginal`, `contaminated` |
| retention_p50_s | Median retention |
| retention_p90_s | p90 retention |
| retention_p99_s | p99 retention |
| retention_max_s | Max retention |
| retention_mean_s | Mean retention |
| retention_std_s | Standard deviation |
| write_once_ratio | Region-level write-once ratio |
| write_once_bytes_ratio | Byte-level write-once ratio |
| read_frequency_mean_hz | Mean read frequency |
| read_frequency_p99_hz | p99 read frequency |
| read_frequency_per_token_hz | Read frequency normalised by token |
| working_set_peak_mb | Peak live bytes |
| working_set_mean_mb | Mean live bytes |
| working_set_growth_rate_mb_per_token | Working set growth |
| sequential_access_ratio | Locality metric |
| stride_dominant_bytes | Dominant stride |
| random_access_ratio | Locality metric |
| inter_access_interval_p50_us | Interval metric |
| inter_access_interval_p99_us | Interval metric |
| mrm_suitability_score | Heuristic score |
| mrm_retention_class | `seconds`, `minutes`, `hours`, `session` |

---

## 7. Telemetry Subsystem

### Purpose

Record everything required to interpret a run scientifically.

### Telemetry categories

1. hardware topology
2. software versions
3. baseline machine state
4. thermal and power state
5. target process resource usage
6. observer process resource usage
7. collector process resource usage
8. system-wide resource usage during run
9. collector metadata
10. timestamps and phase boundaries

---

## 7.1 Hardware telemetry

Capture once per experiment start, and attach per run via metadata snapshot.

Required fields include:

- CPU model
- architecture
- microarchitecture if derivable
- socket count
- core count
- thread count
- frequency range
- cache hierarchy sizes
- NUMA topology
- total installed memory
- memory type and speed if derivable
- huge page settings
- storage device identity and optional quick throughput benchmark
- PMU capability and supported events

Example structure:

```json
{
  "hardware": {
    "cpu": {
      "model": "Intel Xeon Gold 6338",
      "architecture": "x86_64",
      "socket_count": 2,
      "total_logical_cpus": 128,
      "pmu_available": true
    },
    "memory": {
      "total_gb": 256,
      "type": "DDR4",
      "numa_nodes": 2
    },
    "storage": {
      "interface": "NVMe PCIe 4.0"
    }
  }
}
```

---

## 7.2 Software telemetry

Record at minimum:

- OS and version
- kernel version
- Python version
- package version
- `perf` version
- `memray` version
- engine versions
- torch version if present
- CUDA version if present
- driver version if present
- build flags for `llama.cpp`
- ulimit information
- NUMA policy in effect

---

## 7.3 Baseline machine state

Capture an idle pre-run baseline for a configurable duration, default 30 seconds.

Required fields:

- mean and max system-wide CPU utilisation
- per-core CPU utilisation
- used and available memory
- page cache and buffers
- swap usage
- disk read/write rates
- network rates
- load averages
- top existing processes by RSS and CPU
- kernel threads CPU usage if derivable
- IRQ rate
- context switch rate
- page fault rate
- idle cache or memory bandwidth baseline if available

This baseline must be stored so later analyses can compare run load against pre-existing machine activity.

---

## 7.4 Thermal and power telemetry

Thermal state matters because throttling can distort latency and effective memory behaviour.

Required behaviour:

1. Sample thermal data at configurable interval.
2. Support best-effort autodetection from `lm_sensors`, `intel_rapl`, IPMI, or available OS interfaces.
3. Record source used.
4. Record peak temperatures and throttle counts.
5. Flag or invalidate runs where throttling occurs if configured.

Required fields:

- sample timestamp
- per-core or package temperature
- package power if available
- memory power if available
- throttle event count
- peak temperature
- sensor source

---

## 7.5 Observer effect telemetry

The platform must measure itself and the collector while tracing. This is mandatory.

### Processes to observe

1. target inference process group
2. observer process (`mrm-trace` Python process)
3. collector process (`perf`, and optionally `memray` if separate)
4. whole machine

### Sampling interval

Default 250 ms. Must be configurable.

### Required fields per sampled process

- timestamp relative to run start
- CPU percent
- RSS
- VMS
- shared memory
- page faults if available
- read bytes
- write bytes
- thread count
- open file descriptors
- voluntary context switches
- involuntary context switches

### Whole-machine sampled fields

- total CPU utilisation
- available memory
- page cache
- swap used
- disk read/write throughput
- context switch rate
- IRQ rate
- perf sampling interrupt rate if derivable

### Derived observer summary

Compute at minimum:

- observer CPU overhead mean and max
- observer memory overhead mean and peak
- collector CPU overhead mean and max
- collector write throughput
- I/O contention detected true/false
- cache pollution estimate low/medium/high
- run validity class
- validity notes list

---

## 7.6 Collector metadata

Store collector-specific metadata in `metadata.json`.

Required fields:

- primary collector attempted
- primary collector succeeded
- fallback triggered
- fallback reason
- collector actually used
- event list active
- sample rate or period
- estimated coverage percentage
- dropped samples
- parse outcome
- raw artifact paths

---

## 8. Metadata File Contract

Each run must produce a `metadata.json` file with the following top-level structure.

```json
{
  "run_id": "...",
  "experiment_ref": "experiments/...yaml",
  "run_config": {},
  "hardware": {},
  "software": {},
  "baseline": {},
  "thermal": {},
  "observer_effect": {},
  "collector_meta": {},
  "timestamps": {}
}
```

### Timestamp fields

Must include:

- run start UTC
- run end UTC
- model load start
- model load end
- generation start
- generation end
- teardown start
- teardown end

---

## 9. Output Directory Layout

The system must create deterministic output layouts.

### Required layout

```text
results/
└── <model_id>/
    └── <run_id>/
        ├── trace.parquet
        ├── region_map.parquet
        ├── kv_block_lifecycle.parquet
        ├── metrics.csv
        ├── metadata.json
        ├── logs/
        │   ├── run.log
        │   └── parser.log
        ├── raw/
        │   ├── perf.data
        │   ├── perf_script.txt
        │   └── memray.bin
        └── manifest.json
```

### Manifest requirements

`manifest.json` must include checksums, file sizes, and schema versions for all artifacts.

---

## 10. Run Validity Policy

The system must classify runs into validity states.

### States

- `clean`
- `marginal`
- `contaminated`

### Inputs to validity classification

- observer CPU overhead
- observer memory overhead as percentage of target RSS
- collector overhead
- thermal throttle events
- I/O contention
- baseline CPU activity level
- parse completeness
- dropped samples

### Example policy

```yaml
validity_rules:
  clean:
    observer_cpu_overhead_pct_max: 10.0
    observer_memory_overhead_pct_of_target_rss_max: 5.0
    thermal_throttle_events: 0
    io_contention_detected: false
    baseline_cpu_utilisation_max: 15.0

  marginal:
    observer_cpu_overhead_pct_max: 20.0
    observer_memory_overhead_pct_of_target_rss_max: 15.0
    thermal_throttle_events: 2
```

### Required behaviour

1. Runs marked `contaminated` must remain archived for transparency.
2. `contaminated` runs must be excluded from default aggregated metrics and plots.
3. Validity state must appear in `metrics.csv` and metadata.
4. Validity notes must explain why the run was not clean.

---

## 11. CLI Specification

### Top-level commands

The CLI must support at minimum:

- `mrm-trace validate`
- `mrm-trace plan`
- `mrm-trace run`
- `mrm-trace analyse`
- `mrm-trace report`
- `mrm-trace export`

### Command definitions

#### `validate`

Validate YAML config and environment.

Example:

```bash
mrm-trace validate --config experiments/llama3_sweep.yaml
```

#### `plan`

Expand sweep combinations and show planned runs.

```bash
mrm-trace plan --config experiments/llama3_sweep.yaml
```

#### `run`

Execute runs and produce raw artifacts.

```bash
mrm-trace run --config experiments/llama3_sweep.yaml
```

#### `analyse`

Parse, label, and compute metrics from existing raw artifacts.

```bash
mrm-trace analyse --input results/ --metrics retention,write-once,read-frequency
```

#### `report`

Generate aggregated CSVs and publication figures.

```bash
mrm-trace report --input results/ --output reports/
```

#### `export`

Package outputs for dataset release.

```bash
mrm-trace export --input results/ --format hf,zenodo --output dist/
```

---

## 12. Python API Specification

Expose a programmatic API alongside the CLI.

### Minimum API surface

```python
from mrm_trace.api import Experiment

exp = Experiment.from_yaml("experiments/llama3_sweep.yaml")
exp.validate()
plan = exp.plan()
exp.run()
results = exp.analyse()
exp.report(output_dir="reports/")
```

### API requirements

1. API must be thin wrapper over orchestration layer, not separate logic path.
2. API and CLI must produce equivalent outputs.
3. API exceptions must be typed and documented.

---

## 13. Logging and Diagnostics

### Logging requirements

1. Structured logs with timestamps.
2. Log levels: DEBUG, INFO, WARNING, ERROR.
3. Per-run log file plus optional global orchestrator log.
4. Collector command invocations must be logged.
5. Fallbacks and validity triggers must be logged clearly.
6. Parser skips of malformed lines must be counted and logged.
7. Long-running phases must emit progress updates.

### Debug artifacts

Optional debug outputs should include:

- first N parsed rows
- unknown symbol report
- collector stderr capture
- telemetry summary snapshot

---

## 14. Performance and Scalability Requirements

### Parser performance

The parser must be streaming and performant enough for very large trace files.

Target non-functional goals:

- parse at least 500k rows/second on a capable modern workstation for synthetic text traces,
- keep parser memory below 2 GB when processing 50M rows in streaming mode,
- write compressed Parquet efficiently.

### Storage behaviour

1. Support zstd compression for Parquet.
2. Optionally purge large raw collector artifacts after verified conversion, governed by retention policy.
3. Record raw artifact checksums before purge.

### Failure handling

1. If disk space becomes low, fail safely and mark run incomplete.
2. If parse step fails, retain raw collector files.
3. If analysis fails, raw and parsed artifacts must remain untouched.

---

## 15. Security and Safety Considerations

### Required safeguards

1. Collector commands must be built from validated arguments only.
2. Avoid arbitrary shell interpolation of user-provided config values.
3. Sensitive environment variables must not be logged.
4. If running with `sudo`, the package must clearly document operational risks.
5. Prompt files and model paths must be treated as external input and validated.

---

## 16. Testing Specification

The package is a measurement system, so testing must cover both software correctness and measurement validity.

### Test suite structure

```text
tests/
├── conftest.py
├── fixtures/
├── unit/
├── integration/
├── validity/
└── performance/
```

### 16.1 Unit tests

Unit tests must cover each module in isolation.

#### Areas to cover

- config loading and validation
- engine wrappers
- collector command construction
- parser line parsing
- schema validators
- symbol labelling
- address range matching
- metric computation
- telemetry samplers
- manifest generation

### 16.2 Integration tests

Integration tests must cover component boundaries.

Required scenarios:

1. collector output to parser.
2. parser output to labeller.
3. labeller output to analyser.
4. analyser output to exporter.
5. metadata linkage from run IDs across artifacts.
6. fallback collector flow.
7. incomplete raw artifacts handling.

### 16.3 Validity tests

These are mandatory and distinct from normal tests.

Required validity tests:

1. known synthetic trace yields known retention metrics.
2. known write-once pattern yields correct write-once ratio.
3. retention values never negative.
4. p99 retention not below p50 retention.
5. write-once ratios remain in [0,1].
6. contaminated thermal run is classified correctly.
7. high observer overhead run is classified correctly.
8. clean run remains clean.
9. region IDs do not leak across runs.
10. unknown symbol policy behaves as configured.

### 16.4 Property-based tests

Use property-based testing for schema resilience and metric bounds.

Recommended library: `hypothesis`.

### 16.5 Performance tests

Required performance tests:

1. parser throughput benchmark.
2. parser memory profile benchmark.
3. Parquet write speed benchmark.
4. experiment planning scale test with many sweep combinations.

### 16.6 Synthetic fixtures

The team must implement synthetic trace fixtures with known ground truth.

Examples:

- weight-only trace
- KV-only trace
- mixed phase trace
- malformed trace
- contaminated run metadata
- clean run metadata

### Example synthetic trace requirement

A synthetic trace must be able to encode:

- one weight region written once, read many times, long retention
- several KV regions written and read over shorter retention windows
- clear phase boundaries
- deterministic expected metric outcomes

---

## 17. Recommended Dependencies

Suggested dependencies, subject to team validation:

- `pydantic` for schemas
- `pyyaml` for config
- `pandas` and `pyarrow` for data handling
- `psutil` for process telemetry
- `matplotlib` or `seaborn` for plots
- `hypothesis` for property-based tests
- `pytest`, `pytest-cov`, `pytest-xdist`, `pytest-benchmark`, `pytest-mock`
- optional `rich` or `typer` for CLI UX

The package should avoid excessive dependency weight beyond what is necessary.

---

## 18. Implementation Phases

The work should be delivered in phases.

### Phase 1: Foundations

Deliver:

- project scaffolding
- schema definitions
- YAML loader and validation
- logging framework
- CLI skeleton
- manifest generator

Acceptance criteria:

- team can validate configs and produce empty run plans.

### Phase 2: Engine adapters

Deliver:

- `llama.cpp` wrapper
- `vLLM` wrapper
- phase event emission
- target PID discovery

Acceptance criteria:

- both engines can run a trivial prompt and return runtime metadata.

### Phase 3: Collectors

Deliver:

- `perf mem` runner
- `memray` runner
- process monitor
- collector metadata capture

Acceptance criteria:

- package can produce raw collector artifacts and telemetry snapshots.

### Phase 4: Parsing and normalisation

Deliver:

- `perf script` parser
- Memray parser
- canonical trace schema output

Acceptance criteria:

- synthetic raw data converts to canonical trace format.

### Phase 5: Labelling

Deliver:

- symbol labeller
- address tracker
- region map writer
- KV lifecycle derivation

Acceptance criteria:

- test fixtures correctly classify known weight and KV regions.

### Phase 6: Analysis

Deliver:

- retention analyser
- write-once analyser
- read frequency analyser
- working set analyser
- locality and interval analysers
- heuristic suitability score

Acceptance criteria:

- synthetic traces yield expected metrics.

### Phase 7: Telemetry completeness and validity engine

Deliver:

- baseline capture
- thermal monitoring
- observer effect sampling
- validity classifier

Acceptance criteria:

- contaminated and clean fixture runs classify correctly.

### Phase 8: Reporting and export

Deliver:

- CSV/Parquet exports
- figures
- dataset package manifest
- export helpers

Acceptance criteria:

- run directories are publication-ready.

### Phase 9: Hardening

Deliver:

- full tests
- performance tuning
- schema versioning
- docs completion

Acceptance criteria:

- CI passes and package is ready for internal pilot.

---

## 19. CI/CD Requirements

### CI jobs

1. lint and formatting
2. unit tests
3. integration tests
4. validity tests
5. performance smoke tests
6. packaging build

### Recommended gates

- merge blocked if unit or validity tests fail
- release blocked if schema version changed without migration notes
- release blocked if parser throughput regression exceeds defined threshold

---

## 20. Open Technical Decisions for Dev Team

These decisions should be finalised early and documented in ADRs.

1. CLI framework choice, e.g. `typer` versus `argparse`.
2. whether to use `pandas` eagerly or more streaming-focused intermediates.
3. exact method for access interval storage at large scale.
4. rule engine format for symbol matching.
5. retention policy for raw collector artifacts.
6. whether report figure generation is in-core or separate job.
7. whether to support SQLite manifest indexing for large experiments.

---

## 21. Acceptance Criteria Summary

The MVP is complete only when all of the following are true.

1. A YAML experiment file can be validated and expanded into concrete runs.
2. `llama.cpp` and `vLLM` runs can be launched through one common API.
3. `perf mem` collection works where supported and records its own metadata.[cite:230]
4. Fallback to `memray` or process monitor is supported and recorded.[cite:285]
5. Every run produces `trace.parquet`, `region_map.parquet`, `metrics.csv`, and `metadata.json`.
6. Observer process and existing machine resource usage are recorded alongside target process usage.
7. Thermal state and throttling are captured where available.
8. Run validity is computed and surfaced in outputs.
9. Synthetic test traces prove metric correctness for retention and write-once logic.
10. Parser scales to large traces in streaming mode.
11. Documentation is sufficient for another engineer to run and reproduce an experiment.

---

## 22. Suggested Initial ADR List

The Dev team should create Architecture Decision Records for at least the following.

1. Why `perf mem` is the primary collector.
2. Why `memray` is fallback and not replacement.
3. Canonical trace schema design.
4. Run validity classification thresholds.
5. Raw artifact retention and purge policy.
6. Output directory layout and naming strategy.
7. Choice of plotting and report generation tools.
8. Choice of CLI and config validation libraries.

---

## 23. Handover Notes for Engineering Lead

The most important instruction for the team is that this package must be built like scientific instrumentation. Convenience, elegance, and raw speed matter, but **measurement trustworthiness matters more**. Every shortcut that obscures provenance, collector overhead, or machine context will weaken the resulting dataset and undercut its usefulness for MRM research.[cite:134][cite:231][cite:258]

The implementation should therefore prefer:

- explicit schemas over implicit conventions,
- raw-data preservation over lossy convenience,
- validity flags over hidden assumptions,
- deterministic outputs over ad hoc notebooks,
- measured observer effect over assumed negligible overhead.

If implemented according to this specification, the result should be suitable as the foundation for a publishable trace benchmark dataset and a later MRM simulator pipeline.[cite:134][cite:161]

## The Experiment YAML File

A complete `experiments/llama3_sweep.yaml` would look like this:

```yaml
# mrm-trace experiment definition
# Version-pinned for full reproducibility

experiment:
  name: "llama3_mrm_sweep_v1"
  description: "MRM retention characterisation across Llama-3 family, variable context"
  version: "1.0.0"
  authors: ["Patrick O. Ogbuitepu"]
  institution: "Independent Research"
  date_created: "2026-05-23"
  paper_ref: "TBD — arXiv submission"

# ─── Models ───────────────────────────────────────────────────────────────────
models:
  - id: "llama3-8b-q4"
    engine: "llamacpp"
    path: "models/llama-3-8b.Q4_K_M.gguf"
    family: "llama3"
    params_billions: 8
    quantisation: "Q4_K_M"
    dtype: "float16"

  - id: "llama3-70b-q4"
    engine: "llamacpp"
    path: "models/llama-3-70b.Q4_K_M.gguf"
    family: "llama3"
    params_billions: 70
    quantisation: "Q4_K_M"
    dtype: "float16"

  - id: "mistral-7b-q4"
    engine: "llamacpp"
    path: "models/mistral-7b-v0.3.Q4_K_M.gguf"
    family: "mistral"
    params_billions: 7
    quantisation: "Q4_K_M"
    dtype: "float16"

  - id: "llama3-8b-vllm"
    engine: "vllm"
    hf_repo: "meta-llama/Meta-Llama-3-8B"
    family: "llama3"
    params_billions: 8
    quantisation: null
    dtype: "float16"

# ─── Sweep Parameters ─────────────────────────────────────────────────────────
sweep:
  context_lengths: [512, 2048, 8192, 32768, 65536, 131072]
  batch_sizes: [1, 4, 8, 16]
  output_tokens: [128, 512, 1024]
  repetitions: 3           # repeat each config for statistical stability
  warmup_runs: 1           # discarded, warms model into memory

# ─── Prompts ──────────────────────────────────────────────────────────────────
prompts:
  source: "prompts/standard_suite.jsonl"
  # Each entry in the JSONL has: {id, text, category, token_count}
  categories:
    - "factual_short"      # low KV cache pressure
    - "long_reasoning"     # high KV cache pressure
    - "code_generation"    # mixed pattern
    - "document_summary"   # large input, short output (weight-read heavy)

# ─── Collector Configuration ──────────────────────────────────────────────────
collector:
  primary: "perf_mem"               # preferred method
  fallback: "memray"                # used if perf PMU unavailable
  fallback_2: "process_monitor"     # coarse-grained last resort

  perf_mem:
    sample_period: 4000             # sample every N memory events
    ldlat_threshold: 20             # load latency threshold (cycles)
    events:
      - "cpu/mem-loads,ldlat=20/P"
      - "cpu/mem-stores/P"
    extra_flags: ["--call-graph", "dwarf"]
    require_root: true

  memray:
    native_traces: true             # capture C extension allocations
    follow_fork: true               # track child processes (llama.cpp)
    file_size_limit_gb: 20

  process_monitor:
    poll_interval_ms: 100
    metrics: ["rss", "vms", "shared", "text", "data"]

# ─── Labelling Rules ──────────────────────────────────────────────────────────
labeller:
  rules_file: "config/models.yaml"  # per-model-family symbol → region_type map
  unknown_policy: "log_and_skip"    # what to do with unlabelled symbols
  min_sample_count: 10              # discard regions with fewer samples

# ─── Analysis ─────────────────────────────────────────────────────────────────
analyser:
  metrics:
    - retention_duration
    - write_once_ratio
    - read_frequency
    - working_set_size
    - access_locality           # sequential vs random stride
    - inter_access_interval     # time between consecutive accesses to same addr
  histogram_bins: 100
  cdf_resolution: 1000
  group_by: ["region_type", "context_length", "model_family", "batch_size"]

# ─── Output ───────────────────────────────────────────────────────────────────
output:
  base_dir: "results/"
  formats:
    raw_traces: "parquet"          # full address-level trace per run
    metrics: "csv"                 # computed MRM metrics
    figures: "pdf"                 # paper-ready plots
    metadata: "json"               # machine + run telemetry (see below)
  compression: "zstd"
  hf_dataset_push: false          # set true to push to HuggingFace Hub on completion
  hf_repo: "pat2echo/mrm-llm-traces"
```


***

## Machine Telemetry — The Current Design Gap

**No — the current design as described does not yet capture full machine telemetry.** This is an important omission. Reviewers at USENIX ATC, MLSys, and ASPLOS will reject a trace dataset paper without comprehensive machine metadata, because trace values (latency, access frequency) are entirely machine-specific and meaningless without it. Here is what the design needs and where it should live:

***

### New Module: `mrm_trace/telemetry/`

```
mrm_trace/
└── telemetry/
    ├── __init__.py
    ├── hardware.py       ← CPU, memory, cache hierarchy, NUMA topology
    ├── software.py       ← OS, kernel, Python, library versions
    ├── thermal.py        ← CPU temperature, power draw, throttle events
    ├── collector_meta.py ← Which collector was used, why fallback triggered
    └── snapshot.py       ← Assembles full telemetry JSON, taken before/during/after
```


***

### What Each Sub-Module Captures

**`hardware.py`**

```yaml
hardware:
  cpu:
    model: "Intel Xeon Gold 6338 @ 2.00GHz"
    architecture: "x86_64"
    microarchitecture: "Ice Lake"
    socket_count: 2
    cores_per_socket: 32
    threads_per_core: 2
    total_logical_cpus: 128
    base_frequency_mhz: 2000
    max_boost_mhz: 3200
    l1d_cache_kb: 48
    l1i_cache_kb: 32
    l2_cache_kb: 1280
    l3_cache_mb: 48
    pmu_available: true         # determines if perf_mem is usable
    pmu_events_supported: ["mem-loads", "mem-stores", "cache-misses"]

  memory:
    total_gb: 256
    channels: 8
    speed_mt_s: 3200
    type: "DDR4"
    numa_nodes: 2
    numa_topology:
      node_0: {cpus: "0-63", memory_gb: 128}
      node_1: {cpus: "64-127", memory_gb: 128}
    huge_pages_enabled: true
    transparent_huge_pages: "madvise"

  storage:
    model: "Samsung 990 Pro 2TB"
    interface: "NVMe PCIe 4.0"
    sequential_read_gb_s: 7.45   # from fio benchmark run at start
    sequential_write_gb_s: 6.9
```

**`software.py`**

```yaml
software:
  os: "Ubuntu 24.04.2 LTS"
  kernel: "6.8.0-51-generic"
  python_version: "3.11.9"
  mrm_trace_version: "0.3.1"
  perf_version: "6.8.1"
  memray_version: "1.13.0"
  llama_cpp_python_version: "0.3.4"
  llama_cpp_build_flags: ["GGML_NATIVE=ON", "GGML_AVX512=ON"]
  vllm_version: "0.5.3"
  torch_version: "2.3.1+cu121"
  cuda_version: "12.1"          # null if CPU-only
  driver_version: "535.104.05"  # null if CPU-only
  numa_policy: "default"        # libnuma policy in effect
  ulimits:
    open_files: 1048576
    memory_bytes: "unlimited"
```

**`thermal.py`** — sampled at configurable intervals throughout the run:

```yaml
thermal:
  sampling_interval_s: 5
  samples:
    - timestamp_s: 0.0
      cpu_temp_c: [42.0, 44.0, 41.5, ...]   # per-core
      package_power_w: 85.2
      memory_power_w: 12.4
      throttle_events: 0
    - timestamp_s: 5.0
      cpu_temp_c: [68.0, 71.0, 69.5, ...]
      package_power_w: 210.8
      throttle_events: 0
  peak_temp_c: 78.0
  thermal_throttle_count: 0     # non-zero invalidates run
  source: "lm_sensors"          # or "intel_rapl", "IPMI"
```

Thermal data is critical because memory access latency degrades under thermal throttling — a throttled run would produce misleading retention duration measurements and must be flagged or discarded.

**`collector_meta.py`** — records exactly which collection method was used and why:

```yaml
collector_meta:
  primary_attempted: "perf_mem"
  primary_succeeded: true
  fallback_triggered: false
  fallback_reason: null         # e.g. "PMU unavailable in VM", "insufficient privileges"
  collector_used: "perf_mem"
  perf_events_active:
    - "cpu/mem-loads,ldlat=20/P"
    - "cpu/mem-stores/P"
  perf_sample_rate_hz: 1000
  estimated_trace_coverage_pct: 94.2   # fraction of memory ops sampled
  dropped_samples: 1243                # perf buffer overflows
  memray_native_mode: null
  process_monitor_active: true         # always runs as a coarse parallel baseline
```


***

### Updated YAML — Telemetry Section

The experiment YAML gains a `telemetry` section:

```yaml
# ─── Telemetry ────────────────────────────────────────────────────────────────
telemetry:
  capture: true
  thermal_sampling_interval_s: 5
  thermal_source: "auto"          # auto-detects lm_sensors / intel_rapl / IPMI
  invalidate_on_throttle: true    # abort and re-run if throttling detected
  software_versions: true
  hardware_topology: true
  run_fio_benchmark: true         # quick storage benchmark at experiment start
  output_file: "metadata/telemetry_{run_id}.json"
```


***

### What Gets Written Per Run

Every individual run (one model × one context length × one batch size) produces a sidecar `metadata.json` alongside its trace Parquet file:

```
results/
└── llama3-8b-q4/
    └── ctx8192_batch4_rep2/
        ├── trace.parquet           ← address-level samples
        ├── metrics.csv             ← computed MRM metrics
        └── metadata.json          ← complete telemetry snapshot
            ├── hardware/
            ├── software/
            ├── thermal/            ← time-series from this specific run
            ├── collector_meta/
            ├── run_config/         ← the exact YAML slice that produced this run
            └── timestamps/
                ├── run_start_utc: "2026-05-23T04:12:33Z"
                ├── model_load_start_s: 0.000
                ├── model_load_end_s: 14.823
                ├── generation_start_s: 14.823
                ├── generation_end_s: 47.291
                └── run_end_utc: "2026-05-23T04:13:21Z"
```

This level of telemetry makes every trace in your dataset **independently interpretable** — a researcher can look at any single Parquet file and know exactly what machine produced it, what software stack was running, whether thermal conditions were stable, and which collection method was used. That is the standard required for an artifact evaluation badge at ASPLOS or USENIX ATC.


## The Three Data Layers

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1: OBSERVED DATA                                 │
│  The actual LLM inference memory access traces          │
│  → trace.parquet (address, symbol, op, timestamp)       │
├─────────────────────────────────────────────────────────┤
│  Layer 2: OBSERVER EFFECT DATA                          │
│  Resource usage of mrm-trace itself while collecting    │
│  → metadata.json / observer_process section             │
├─────────────────────────────────────────────────────────┤
│  Layer 3: MACHINE CONTEXT DATA                          │
│  Baseline system state, hardware topology, thermals     │
│  → metadata.json / baseline + hardware + thermal        │
└─────────────────────────────────────────────────────────┘
```

the observed data (Layer 1) is the primary output, and it is currently described only at a high level. Here is what the observed data records contain in full:

***

## Layer 1: The Observed Data Files

### `trace.parquet` — Raw Address-Level Samples

Every memory access event captured by `perf mem` or Memray is stored as one row. For a 70B model inference run at 32K context this file will be **tens of millions of rows**:

```
Schema: trace.parquet
─────────────────────────────────────────────────────────
Column              Type        Description
─────────────────────────────────────────────────────────
run_id              string      Foreign key → metadata.json
sample_id           uint64      Monotonic sample counter
timestamp_ns        uint64      Nanoseconds since run start (from perf clock)
phase               category    weight_load | generation | teardown
address             uint64      Virtual memory address of access
address_page        uint64      Page-aligned address (address >> 12)
op_type             category    load | store
symbol              string      Nearest symbol from DWARF/ELF (e.g. ggml_vec_dot_q4_K)
dso                 string      Shared object / binary (e.g. libggml.so)
region_type         category    weight | kv_cache | activation | runtime | unknown
region_id           uint32      Index into region_map (groups addresses into allocations)
access_latency_ns   uint32      Load-to-use latency if perf ldlat used, else null
cache_level         category    L1 | L2 | L3 | DRAM | null
numa_node           uint8       NUMA node where memory physically resides
cpu_id              uint8       Logical CPU that performed the access
thread_id           uint32      Thread within the inference process
token_index         uint32      Which output token was being generated (null for weight_load)
context_tokens      uint32      KV cache size at time of sample
```


***

### `region_map.parquet` — Allocation Registry

Every distinct memory allocation (a contiguous address range assigned to one region type) gets one row. This is what the `labeller/address_tracker.py` builds during the run:

```
Schema: region_map.parquet
─────────────────────────────────────────────────────────
Column                  Type      Description
─────────────────────────────────────────────────────────
region_id               uint32    Primary key, FK in trace.parquet
region_type             category  weight | kv_cache | activation | runtime
addr_start              uint64    Base address of allocation
addr_end                uint64    End address of allocation
size_bytes              uint64    Allocation size
alloc_timestamp_ns      uint64    When allocated (from Memray / perf probe)
free_timestamp_ns       uint64    When freed, null if never freed in run
first_access_ns         uint64    First sample timestamp in this region
last_access_ns          uint64    Last sample timestamp in this region
retention_duration_ns   uint64    last_access_ns - first_access_ns  ← MRM metric
write_count             uint32    Total store operations to this region
read_count              uint32    Total load operations to this region
is_write_once           bool      write_count == 1  ← MRM metric
symbol_dominant         string    Most frequent symbol accessing this region
layer_index             int16     Transformer layer (if inferrable from symbol), else -1
weight_type             category  q_proj | k_proj | v_proj | ffn | embed | other | null
```


***

### `metrics.csv` — Computed MRM Metrics (Aggregated)

This is what the `analyser/` modules produce — one row per (run_id, region_type, context_length, batch_size) grouping. This is the file that goes directly into your paper tables and figures:

```
Schema: metrics.csv
─────────────────────────────────────────────────────────
run_id
model_id
engine
context_length
batch_size
output_tokens
region_type
rep_index
run_validity

# Retention duration (seconds)
retention_p50_s
retention_p90_s
retention_p99_s
retention_max_s
retention_mean_s
retention_std_s

# Write-once ratio
write_once_ratio              # fraction of regions where write_count == 1
write_once_bytes_ratio        # fraction of total bytes in write-once regions

# Read frequency
read_frequency_mean_hz
read_frequency_p99_hz
read_frequency_per_token_hz   # normalised by output token count

# Working set size
working_set_peak_mb
working_set_mean_mb
working_set_growth_rate_mb_per_token

# Access locality
sequential_access_ratio       # fraction of accesses within 64 bytes of previous
stride_dominant_bytes         # most common access stride
random_access_ratio

# Inter-access interval
inter_access_interval_p50_us
inter_access_interval_p99_us

# Derived MRM relevance scores (computed from above)
mrm_suitability_score         # 0.0–1.0, higher = better MRM candidate
mrm_retention_class           # seconds | minutes | hours | session
```


***

### `kv_block_lifecycle.parquet` — Per-Block KV Cache Detail

This is a **KV-cache-specific** supplementary file that records the lifecycle of every individual KV cache block, which is the most granular data the MRM paper needs for retention window sizing:

```
Schema: kv_block_lifecycle.parquet
─────────────────────────────────────────────────────────
block_id
layer_index             # transformer layer (0 to n_layers-1)
head_index              # attention head
token_start             # first token position this block covers
token_end               # last token position this block covers
alloc_ns                # when block was allocated
first_read_ns           # first time block was read after write
last_read_ns            # last time block was read
free_ns                 # when block was evicted/freed
lifetime_ns             # free_ns - alloc_ns
active_read_window_ns   # last_read_ns - first_read_ns
read_count              # how many times this block was read
context_length_at_alloc # KV cache depth when this block was created
```

This file directly answers: *"Given a context of N tokens, how long does a KV block at layer L remain live?"* — which is the exact input an MRM hardware designer needs to set the retention window of their memory cells.

***

## Complete Per-Run Output

Every single experiment run therefore produces:

```
results/llama3-8b-q4/ctx8192_batch4_rep2/
│
├── trace.parquet                  ← every memory sample (Layer 1, primary)
├── region_map.parquet             ← allocation registry with retention durations
├── kv_block_lifecycle.parquet     ← per-KV-block lifecycle detail
├── metrics.csv                    ← aggregated MRM metrics for this run
│
└── metadata.json                  ← all context for interpreting Layer 1
    ├── run_id
    ├── run_config/                 ← YAML slice
    ├── hardware/                   ← Layer 3
    ├── software/                   ← Layer 3
    ├── baseline/                   ← Layer 3 (idle snapshot before run)
    ├── thermal/                    ← Layer 3 (time-series during run)
    ├── observer_effect/            ← Layer 2 (mrm-trace + perf resource usage)
    │   ├── target_process/         ← inference engine resource time-series
    │   ├── observer_process/       ← mrm-trace resource time-series
    │   ├── perf_process/           ← perf daemon resource time-series
    │   ├── system_wide/            ← whole-machine resource time-series
    │   └── observer_effect_summary/
    │       └── run_validity: "clean"
    ├── collector_meta/             ← which collector, coverage %, dropped samples
    └── timestamps/                 ← phase boundaries
```

Every row in `trace.parquet` has a `run_id` that links back to the `metadata.json`, so any trace sample is fully auditable — you know exactly what machine state, observer overhead, and thermal conditions existed at the moment it was captured. That traceability chain is what earns an artifact evaluation **Artifacts Available + Artifacts Evaluated Functional** badge, which is now effectively mandatory for systems papers at top venues.


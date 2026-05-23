## The Observer Effect Problem

The `mrm-trace` package is running **concurrently** with the inference engine. It introduces:

- **CPU overhead** from `perf` sampling interrupt handlers
- **Memory overhead** from Memray's allocation hooks and trace buffers
- **I/O overhead** from writing `perf.data` to disk
- **Cache pollution** from the tracer's own memory accesses disturbing the very patterns you are measuring

All of this must be measured, recorded, and disclosed. Some runs may need to be flagged as high-observer-effect and excluded from retention duration analysis.

***

## New Module: `mrm_trace/telemetry/observer.py`

This module runs in a **dedicated background thread** throughout each experiment run, continuously sampling both the target process (the inference engine) and the observer process (mrm-trace itself), plus the total machine state:

```
mrm_trace/telemetry/
├── hardware.py
├── software.py
├── thermal.py
├── collector_meta.py
├── observer.py          ← NEW: observer effect + machine baseline
├── baseline.py          ← NEW: pre-run idle machine snapshot
└── snapshot.py          ← assembles everything
```


***

## `baseline.py` — Pre-Run Idle Snapshot

Before any inference or tracing starts, a **30-second idle baseline** is captured. This establishes what the machine looks like at rest — existing processes, background resource usage, kernel threads, and ambient memory pressure that your traces will sit on top of:

```yaml
baseline:
  duration_s: 30
  captured_at_utc: "2026-05-23T04:11:58Z"

  system_wide:
    cpu_utilisation_pct:
      mean: 3.2
      max: 8.1
      per_core: [1.1, 2.3, 0.8, ...]     # one per logical CPU
    memory:
      total_gb: 256.0
      available_gb: 218.4
      used_gb: 37.6
      cached_gb: 29.1
      buffers_gb: 2.3
      swap_used_gb: 0.0
    io:
      read_mb_s: 0.4
      write_mb_s: 1.2
    network:
      recv_mb_s: 0.1
      sent_mb_s: 0.0
    load_average: [0.42, 0.38, 0.35]    # 1m, 5m, 15m

  existing_processes:
    # snapshot of top-20 processes by RSS at baseline
    - pid: 1842
      name: "systemd-journald"
      cpu_pct: 0.1
      rss_mb: 42.3
      status: "sleeping"
    - pid: 2201
      name: "postgres"
      cpu_pct: 0.2
      rss_mb: 312.8
      status: "sleeping"
    # ... etc

  kernel_threads_cpu_pct: 1.4
  irq_rate_per_s: 4821
  context_switches_per_s: 12043
  page_faults_per_s: 0.3
  cache_state:
    l3_hit_rate_pct: 94.2        # from perf stat at idle
    memory_bandwidth_utilisation_pct: 2.1
```

This baseline is **subtracted or compared against** during analysis — any metric during inference is meaningful only relative to this idle state.

***

## `observer.py` — Continuous Dual-Process Monitoring

During the actual run, a background thread samples **three separate process groups** at configurable intervals, writing to a time-series buffer that is flushed to the sidecar `metadata.json`:

```yaml
observer_effect:
  sampling_interval_ms: 250     # 4 samples/second throughout the run

  # ── Target process (the inference engine) ───────────────────────
  target_process:
    pid: 48823
    name: "llama-cli"
    samples:
      # one entry per 250ms interval
      - t_s: 0.00
        cpu_pct: 0.1              # pre-load
        rss_mb: 48.2
        vms_mb: 312.0
        shared_mb: 18.4
        page_faults_minor: 12
        page_faults_major: 0
        voluntary_ctx_switches: 3
        involuntary_ctx_switches: 0
        read_bytes: 0
        write_bytes: 0
        open_file_descriptors: 24
        threads: 1

      - t_s: 1.25                 # during weight load
        cpu_pct: 98.4
        rss_mb: 4821.3            # weights loading into RAM
        vms_mb: 9204.1
        page_faults_major: 8241   # pages being faulted in from mmap
        read_bytes: 4_294_967_296 # 4 GB read from disk
        write_bytes: 0
        threads: 8

      - t_s: 16.00                # during generation (KV cache growing)
        cpu_pct: 87.2
        rss_mb: 6102.4
        vms_mb: 9204.1
        page_faults_major: 0      # all weights now resident
        page_faults_minor: 1842   # KV cache allocations
        read_bytes: 0
        write_bytes: 0
        threads: 8

  # ── Observer process (mrm-trace itself) ─────────────────────────
  observer_process:
    pid: 48791
    name: "python mrm-trace"
    samples:
      - t_s: 0.00
        cpu_pct: 1.2
        rss_mb: 124.3
        write_bytes: 0

      - t_s: 1.25
        cpu_pct: 4.8              # perf parsing and writing
        rss_mb: 381.7             # trace buffer filling
        write_bytes: 524_288_000  # 500 MB/s to perf.data

      - t_s: 16.00
        cpu_pct: 3.1
        rss_mb: 892.4             # memray buffer growing
        write_bytes: 204_800_000

  # ── perf itself (kernel-space overhead) ─────────────────────────
  perf_process:
    pid: 48790
    name: "perf"
    samples:
      - t_s: 0.00
        cpu_pct: 2.3
        rss_mb: 28.1
        kernel_cpu_pct: 0.8       # perf interrupt handler overhead

  # ── System-wide during run ───────────────────────────────────────
  system_wide:
    samples:
      - t_s: 0.00
        total_cpu_pct: 6.2
        available_memory_gb: 215.1
        page_cache_gb: 29.4
        swap_used_gb: 0.0
        io_read_mb_s: 0.8
        io_write_mb_s: 501.2      # perf.data write
        context_switches_per_s: 14821
        irq_rate_per_s: 5104
        perf_sampling_interrupts_per_s: 1024  # from perf stat

  # ── Derived observer effect metrics (computed post-run) ─────────
  observer_effect_summary:
    observer_cpu_overhead_pct_mean: 3.8
    observer_cpu_overhead_pct_max: 6.2
    observer_memory_overhead_mb_mean: 612.4
    observer_memory_overhead_mb_peak: 1204.8
    perf_interrupt_overhead_pct: 0.9    # % of target CPU stolen by perf handlers
    cache_pollution_estimate: "low"     # low/medium/high based on observer RSS vs L3
    io_contention_detected: false       # true if observer write_mb_s > 10% of disk BW
    run_validity: "clean"               # clean / marginal / contaminated
    validity_notes: []
    # contaminated example: "observer RSS (1.8 GB) > 10% of L3 cache (48 MB) during generation phase"
```


***

## Validity Classification

The `run_validity` field is computed automatically from rules defined in the package config:

```yaml
# config/validity_rules.yaml

validity_rules:
  clean:
    observer_cpu_overhead_pct_max: 10.0
    observer_memory_overhead_pct_of_target_rss_max: 5.0
    thermal_throttle_events: 0
    io_contention_detected: false
    baseline_cpu_utilisation_max: 15.0   # machine must be near-idle at start

  marginal:                              # included in dataset but flagged
    observer_cpu_overhead_pct_max: 20.0
    observer_memory_overhead_pct_of_target_rss_max: 15.0
    thermal_throttle_events: 2

  contaminated:                          # excluded from analysis, kept in archive
    # anything exceeding marginal thresholds
```

Runs marked `contaminated` are still stored in the dataset archive (for transparency) but excluded from metric computations and paper figures. The validity field is a first-class column in every exported CSV, allowing downstream researchers to apply their own filtering criteria.

***

## Updated `metadata.json` Structure Per Run

```
metadata.json
├── run_id: "llama3-8b-q4_ctx8192_batch4_rep2_20260523T041233Z"
├── experiment_ref: "experiments/llama3_sweep.yaml"
├── run_config/              ← exact YAML slice for this run
├── hardware/                ← from hardware.py (static, per machine)
├── software/                ← from software.py (static, per machine)
├── baseline/                ← from baseline.py (30s idle snapshot before run)
├── thermal/                 ← from thermal.py (time-series during run)
├── observer_effect/         ← from observer.py (dual-process time-series + summary)
│   ├── target_process/
│   ├── observer_process/
│   ├── perf_process/
│   ├── system_wide/
│   └── observer_effect_summary/
│       └── run_validity: "clean"
├── collector_meta/          ← from collector_meta.py
└── timestamps/              ← phase boundaries (load / generate / teardown)
```


***

## Why This Matters for Your Paper

A reviewer seeing retention duration values of 14 seconds will immediately ask: *"Is that because the inference actually needed 14 seconds, or because your tracer's memory pressure caused page evictions that extended the apparent lifetime?"* With the observer effect records in the dataset, you can directly answer that: for all runs marked `clean`, observer memory overhead was under 5% of target RSS and no cache pollution was detected. That is the difference between a paper that passes artifact evaluation and one that gets a "concerns about measurement validity" rejection.


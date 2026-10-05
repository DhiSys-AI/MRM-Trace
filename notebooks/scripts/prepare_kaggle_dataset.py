#!/usr/bin/env python3
"""
Package a completed sweep in results/ as a Kaggle dataset under datasets/.

What goes in, and why, is decided in
context-files/06 - Kaggle Dataset Selection.md.

The layout is flat on purpose.  The Kaggle API uploads a sub-directory as a
single zip/tar archive, and a file inside an archive cannot carry a file
description or column descriptions — two of the inputs to Kaggle's usability
score.  So every published file sits at the top level, is CSV or Parquet, and
is keyed by run_id; nothing is per-run except as rows.

    <out>/
    ├── README.md                       data card (also the Kaggle description)
    ├── dataset-metadata.json           title, subtitle, description, licence,
    │                                   tags, sources, and every file and column
    ├── dataset-cover-image.png
    ├── runs.csv                        one row per run: configuration, machine,
    │                                   sampling, thermal, observer effect
    ├── metrics.csv                     per run x region type
    ├── summary_by_region.csv           headline: region types across all runs
    ├── summary_by_model.csv            region metrics by model
    ├── summary_by_output_length.csv    region metrics by generated tokens
    ├── region_map.parquet              address ranges -> region type
    ├── kv_block_lifecycle.parquet      KV-cache blocks
    ├── trace.parquet                   sampled memory accesses (see below)
    ├── prompt_outputs.csv              prompt and generated text per run
    ├── sample_calibration.csv          sampling period per configuration
    └── sha256sums.txt

Every column of every table must have a description in _COLUMN_DOCS; the
script fails, naming the columns, if one is missing.  That is what keeps the
Kaggle metadata complete when the pipeline adds a column.

raw/perf.data (14 GB of a 16 GB sweep) is not published: everything a user
can read out of it is already in trace.parquet.

Trace subsampling.  ~98% of trace rows are `runtime`; `kv_cache` and `weight`
are already sparse (a few thousand rows per run, many regions single-sample).
A uniform subsample would thin exactly the regions the dataset is about, so
by default every non-runtime row is kept and runtime rows are kept with
probability --runtime-fraction.  Each row carries `sample_weight` = 1/p so
counts can be re-weighted.  --trace-mode full ships traces unthinned.

Privacy.  machine_id, hostname, output paths and the dso column contain the
collecting user's name and home directory.  They are rewritten to a machine
alias and /home/user/, and the finished package is scanned for the username;
the script fails if it survives anywhere in a text file or in a string column
of a parquet file.

    venv/bin/python notebooks/scripts/prepare_kaggle_dataset.py \\
        --force
    kaggle datasets version -p datasets/mrm-trace-kaggle -t -m "<notes>"
    kaggle datasets metadata --update -p datasets/mrm-trace-kaggle <owner>/<slug>

-t (--keep-tabular) matters: without it Kaggle converts trace.parquet to CSV.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import pathlib
import re
import shutil
import sys
import zlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPO = pathlib.Path(__file__).resolve().parents[2]

_RUN_FILES = ("metadata.json", "prompt_outputs.json", "trace.parquet",
              "region_map.parquet", "kv_block_lifecycle.parquet", "metrics.csv")
# results/analysis/<src> -> <dst>.  kv_lifecycle_blocks.csv is not carried:
# it is the same 90k blocks as the stacked kv_block_lifecycle.parquet.
_SUMMARIES = {
    "region_contrast.csv": "summary_by_region.csv",
    "model_scaling.csv": "summary_by_model.csv",
    "context_scaling.csv": "summary_by_output_length.csv",
}
_HOME_RE = re.compile(r"/home/[^/]+/")
_TRACE_BATCH_ROWS = 1_000_000  # ADR-2: stream, never load a whole trace

# Matches the existing Kaggle dataset, which new releases are versions of.
_TITLE = "MRM-Trace: LLM Inference Memory-Access Traces"
_SLUG = "mrm-trace-llm-memory-access-traces"
_SUBTITLE = "Sampled perf mem traces of llama.cpp inference, labelled by memory region"
# Must be existing Kaggle tags; unknown ones are dropped by Kaggle on upload.
_KEYWORDS = ["computer science", "artificial intelligence", "deep learning", "nlp",
             "electronics"]


# ------------------------------------------------------------ runs.csv shape ---

# manifest.csv column -> runs.csv column.  manifest columns not listed are
# dropped (has_prompt_outputs: always true here, and prompt_outputs.csv exists).
_MANIFEST_COLS = {
    "run_id": "run_id",
    "model_id": "model_id",
    "max_tokens_per_prompt": "output_tokens",
    "rep": "rep",
    "machine_id": "machine_id",
    "collector": "collector",
    "context_length": "context_length",
    "batch_size": "batch_size",
    "run_validity": "run_validity",
    "peak_temp_c": "peak_temp_c",
    "throttle_events": "throttle_events",
    "total_wall_s": "total_wall_s",
    "avg_tok_per_s": "avg_tok_per_s",
    "total_output_tokens": "total_output_tokens",
    "retention_p99_s": "retention_p99_s",
    "write_once_ratio_mean": "write_once_ratio_mean",
    "working_set_pages_total": "working_set_pages_total",
    "cpu_model": "cpu_model",
    "os": "os",
    "recorded_at": "recorded_at",
}
# Flattened metadata.json fields that repeat a manifest column.  Each pair is
# checked equal on every run before the metadata copy is dropped; if one ever
# differs, it is kept, and the doc gate then asks for it to be described.
_METADATA_DUPLICATES = {
    "model.id": "model_id",
    "engine.model_id": "model_id",
    "engine.context_length": "context_length",
    "generation.context_length": "context_length",
    "engine.batch_size": "batch_size",
    "generation.batch_size": "batch_size",
    "generation.output_tokens": "output_tokens",
    "generation.tokens_requested_per_prompt": "output_tokens",
    "generation.rep_index": "rep",
    "generation.total_output_tokens": "total_output_tokens",
    "generation.avg_tokens_per_second": "avg_tok_per_s",
    "timing.total_wall_duration_s": "total_wall_s",
    "thermal.peak_temp_c": "peak_temp_c",
    "thermal.throttle_events": "throttle_events",
    "observer_effect.thermal_throttle_events": "throttle_events",
    "thermal.run_validity": "run_validity",
    "collector.type": "collector",
    "hardware.cpu_model": "cpu_model",
    "software.os": "os",
    "engine.dtype": "model.dtype",
    "engine.quantisation": "model.quantisation",
    "observer_effect.baseline_cpu_pct": "hardware.baseline_cpu_pct",
    "thermal.collection_wall_duration_s": "collector.collection_duration_s",
}
# Dropped outright: local paths, the hostname, constants with no information
# for this release, and GPU power fields that the driver reported implausibly.
_METADATA_DROP = {
    "engine.model_path", "collector.output_file", "software.hostname",
    "software.perf_bin", "engine.extra.cli_path", "engine.extra.build_flags",
    "gpu.power_limit_w", "gpu.power_draw_status",
}


# --------------------------------------------------------------- column docs ---

_KEY_DOCS = {
    "run_id": "Run identifier: model, context, batch, output tokens, repetition and UTC start time. Joins every table.",
    "model_id": "Model that was run (GGUF, Q4_K_M, llama.cpp on CPU).",
    "output_tokens": "Output tokens requested for the run; generation always ran to this limit (EOS ignored).",
    "rep": "Repetition index of this configuration (1 or 2).",
    "region_type": "Memory region: weight (mmap'd model file), kv_cache (engine KV allocation), runtime (stacks, heap, libraries), unknown (no known mapping).",
}

_COLUMN_DOCS: dict[str, dict[str, str]] = {
    "runs.csv": {
        "machine_id": "Alias of the machine that collected the run.",
        "collector": "Memory-access collector: perf_mem (Intel PEBS sampling via perf mem).",
        "context_length": "llama.cpp context window in tokens.",
        "batch_size": "Sequences decoded together.",
        "run_validity": "clean / marginal / contaminated, from tracer overhead, baseline load, temperature and throttling.",
        "peak_temp_c": "Peak CPU package temperature during collection, °C.",
        "throttle_events": "Thermal throttle events during collection.",
        "total_wall_s": "Wall time of the whole run including model load, seconds.",
        "avg_tok_per_s": "Generation throughput under tracing, tokens per second.",
        "total_output_tokens": "Tokens actually generated.",
        "retention_p99_s": "99th-percentile region retention across all region types in the run, seconds.",
        "write_once_ratio_mean": "Mean write-once ratio across region types in the run.",
        "working_set_pages_total": "Distinct 4 KiB pages touched in the run.",
        "cpu_model": "CPU model string.",
        "os": "Operating system and kernel.",
        "recorded_at": "UTC time the run was recorded.",
        "model.family": "Model family.",
        "model.params_billions": "Parameter count, billions.",
        "model.quantisation": "Weight quantisation of the GGUF file.",
        "model.dtype": "Compute dtype reported for the model.",
        "engine.engine_name": "Inference engine.",
        "engine.engine_version": "llama-cpp-python version.",
        "engine.extra.use_python_bindings": "Whether llama.cpp was driven through its Python bindings.",
        "generation.prompts": "Prompts in the run.",
        "generation.ignore_eos": "Whether end-of-sequence was ignored so generation reached the token limit.",
        "generation.reached_token_limit": "Whether every prompt generated the full requested tokens.",
        "generation.total_duration_s": "Time spent generating, seconds.",
        "collector.mode": "spawn: perf started the inference process, so it was traced from its first instruction.",
        "collector.sample_period": "Load sampling period: one load sampled per this many loads.",
        "collector.event_strings": "perf events used (JSON list); the store period is in the mem-stores event.",
        "collector.region_granularity_bytes": "Granule used to group addresses into regions, bytes.",
        "collector.samples_collected": "Samples in the full trace (before this release's subsampling).",
        "collector.collection_duration_s": "Time perf was recording, seconds.",
        "collector.cpu_count": "Logical CPUs.",
        "collector.achieved_sample_rate_per_cpu": "Samples per second per CPU actually achieved.",
        "collector.perf_event_max_sample_rate_before": "Kernel perf_event_max_sample_rate before the run.",
        "collector.perf_event_max_sample_rate_after": "Kernel perf_event_max_sample_rate after the run; a drop means the kernel throttled sampling.",
        "collector.output_file_size_bytes": "Size of the raw perf.data (not published), bytes.",
        "collector.parse_success": "Whether the raw trace parsed completely.",
        "collector.warnings": "Collector warnings (JSON list).",
        "software.kernel": "Kernel release.",
        "software.is_wsl2": "Whether the machine was WSL2 (perf mem cannot work there).",
        "software.perf_version": "perf version.",
        "software.perf_event_paranoid": "Kernel perf_event_paranoid setting.",
        "software.memray_installed": "Whether memray was installed (not used by these runs).",
        "software.mrm_trace_version": "Version of the mrm-trace tool.",
        "hardware.baseline_cpu_pct": "System CPU use measured before the run, percent.",
        "hardware.baseline_cpu_pct_max": "Maximum baseline CPU use, percent.",
        "hardware.mem_total_mb": "Total RAM, MB.",
        "hardware.mem_available_mb": "Available RAM before the run, MB.",
        "hardware.mem_used_pct": "RAM in use before the run, percent.",
        "hardware.load_avg_1m": "1-minute load average before the run.",
        "hardware.cpu_freq_hw_max_mhz": "Hardware maximum CPU frequency, MHz.",
        "hardware.cpu_freq_scaling_max_mhz": "Frequency-scaling ceiling in force, MHz (turbo off pins this to base clock).",
        "hardware.cpu_freq_scaling_min_mhz": "Frequency-scaling floor, MHz.",
        "hardware.cpu_freq_governor": "CPU frequency governor.",
        "hardware.cpu_freq_start_mhz": "CPU frequency at the start of the run, MHz.",
        "hardware.cpu_freq_min_mhz": "Lowest CPU frequency seen during the run, MHz.",
        "hardware.cpu_freq_mean_mhz": "Mean CPU frequency during the run, MHz.",
        "hardware.cpu_freq_end_mhz": "CPU frequency at the end of the run, MHz.",
        "hardware.cpu_vendor": "CPU vendor.",
        "hardware.pmu_type": "PMU sampling facility (PEBS on Intel).",
        "hardware.pmu_facility_available": "Whether the PMU facility was available.",
        "hardware.ac_power": "Whether the machine was on mains power.",
        "hardware.turbo_enabled": "Whether turbo boost was enabled.",
        "hardware.disk_free_gb": "Free disk space before the run, GB.",
        "gpu.name": "GPU present in the machine (unused: these are CPU runs).",
        "gpu.vram_total_mb": "GPU memory, MB.",
        "gpu.vram_start_used_mb": "GPU memory in use at the start, MB.",
        "gpu.vram_peak_used_mb": "Peak GPU memory in use, MB; flat confirms the GPU was idle.",
        "gpu.vram_end_used_mb": "GPU memory in use at the end, MB.",
        "gpu.gpu_util_pct_mean": "Mean GPU utilisation, percent.",
        "thermal.start_temp_c": "CPU temperature at the start, °C.",
        "thermal.end_temp_c": "CPU temperature at the end, °C.",
        "thermal.throttled": "Whether any thermal throttling occurred.",
        "thermal.validity_failure_reasons": "Why the run is not clean (JSON list; empty when clean).",
        "observer_effect.observer_cpu_overhead_pct_max": "Peak CPU used by the tracer, percent of the machine.",
        "observer_effect.observer_cpu_overhead_pct_abs_max": "Peak CPU used by the tracer, percent of one core.",
        "observer_effect.observer_cpu_overhead_pct_mean": "Mean CPU used by the tracer, percent of the machine.",
        "observer_effect.observer_memory_overhead_mb_peak": "Peak memory used by the tracer, MB.",
        "observer_effect.observer_memory_overhead_frac": "Tracer peak memory / target peak RSS.",
        "observer_effect.observer_memory_system_frac": "Tracer peak memory / system RAM (the bound used for perf mem validity).",
        "observer_effect.system_memory_mb": "System RAM, MB.",
        "observer_effect.target_rss_mb_peak": "Peak resident memory of the inference process, MB.",
        "observer_effect.io_contention_detected": "Whether disk contention was detected during the run.",
        "observer_effect.n_observer_samples": "Tracer resource samples taken.",
        "observer_effect.n_target_samples": "Target-process resource samples taken.",
    },
    "metrics.csv": {
        "n_regions": "Regions of this type in the run.",
        "n_regions_observable": "Regions with at least 2 samples; only these have a retention interval.",
        "retention_support_frac": "n_regions_observable / n_regions.",
        "retention_p50_s": "Median retention (last minus first sampled access of a region) over observable regions, seconds.",
        "retention_p90_s": "90th-percentile retention, seconds.",
        "retention_p99_s": "99th-percentile retention, seconds.",
        "retention_mean_s": "Mean retention, seconds.",
        "write_once_ratio_mean": "Share of written addresses with exactly one sampled store; empty when nothing was written. Unsupported for kv_cache (too few sampled stores).",
        "reads_per_write_mean": "Sampled loads per sampled store; inf when no store was sampled (weight is read-only).",
        "working_set_mb_total": "Distinct pages touched x 4 KiB, MB.",
        "unique_pages_total": "Distinct 4 KiB pages touched.",
        "n_intervals": "Inter-access intervals measured: gaps between consecutive sampled accesses to the same exact address.",
        "iai_p50_s": "Median inter-access interval, seconds. Biased short: only addresses sampled twice yield a gap, and those are the most-touched ones.",
        "iai_p90_s": "90th-percentile inter-access interval, seconds.",
        "iai_p99_s": "99th-percentile inter-access interval, seconds.",
        "suitability": "Heuristic suitability of the region for managed-retention memory: high_mrm / medium_mrm / low_mrm / unknown.",
    },
    "summary_by_region.csv": {
        "runs": "Runs aggregated.",
        "models": "Models aggregated.",
        "retention p50 (s)": "Mean over runs of the per-run median retention, seconds.",
        "retention p50 (s) sd": "Standard deviation of that across runs.",
        "retention p99 (s)": "Mean over runs of the per-run p99 retention, seconds.",
        "retention p99 (s) sd": "Standard deviation of that across runs.",
        "write-once ratio": "Mean over runs of the write-once ratio (unsupported for kv_cache).",
        "write-once ratio sd": "Standard deviation of that across runs.",
        "reads per write": "Mean over runs of reads per write, excluding runs where it is unbounded.",
        "reads per write sd": "Standard deviation of that across runs.",
        "working set (MB)": "Mean over runs of the working set, MB.",
        "working set (MB) sd": "Standard deviation of that across runs.",
        "unbounded r/w runs": "Runs where no store was sampled, so reads per write is unbounded.",
    },
    "summary_by_model.csv": {
        "params_b": "Model parameter count, billions.",
        "retention_p50_s": "Mean over runs of the per-run median retention, seconds.",
        "retention_p99_s": "Mean over runs of the per-run p99 retention, seconds.",
        "write_once": "Mean over runs of the write-once ratio (unsupported for kv_cache).",
        "working_set_mb": "Mean over runs of the working set, MB.",
        "runs": "Runs aggregated.",
    },
    "summary_by_output_length.csv": {
        "retention_p50_s": "Mean over runs of the per-run median retention, seconds.",
        "retention_p99_s": "Mean over runs of the per-run p99 retention, seconds.",
        "write_once": "Mean over runs of the write-once ratio (unsupported for kv_cache).",
        "working_set_mb": "Mean over runs of the working set, MB; for kv_cache it tracks how full the cache got.",
        "n_regions": "Mean regions of this type per run.",
        "runs": "Runs aggregated.",
    },
    "region_map.parquet": {
        "region_id": "Region within the run; joins to trace.region_id.",
        "page_min": "First 4 KiB page of the region (address >> 12).",
        "page_max": "Last 4 KiB page of the region.",
        "first_seen_ns": "Timestamp of the region's first sampled access, ns.",
        "last_seen_ns": "Timestamp of the region's last sampled access, ns.",
        "access_count": "Sampled accesses to the region in the full trace.",
    },
    "kv_block_lifecycle.parquet": {
        "block_id": "KV block within the run.",
        "address": "Block virtual address.",
        "address_page": "address >> 12.",
        "write_ns": "First sampled store to the block, ns; null when no store was sampled.",
        "last_read_ns": "Last sampled load from the block, ns.",
        "evict_ns": "Eviction time when observed, ns; usually null.",
        "total_reads": "Sampled loads from the block.",
        "write_observed": "Whether any store to the block was sampled. False for most blocks.",
        "lifetime_ns": "last_read_ns - write_ns; null when write_observed is false (never imputed as 0).",
    },
    "trace.parquet": {
        "sample_id": "Sequence number of the sample within the run's full trace.",
        "timestamp_ns": "perf sample time on the monotonic clock, ns (µs resolution).",
        "phase": "Inference phase: weight_load or generation; null before the engine reports a phase.",
        "address": "Sampled virtual data address.",
        "address_page": "address >> 12 (4 KiB page).",
        "op_type": "load or store. Stores are sampled 500x less often than loads.",
        "symbol": "Function executing the access, as resolved by perf; not used to assign region_type.",
        "dso": "Binary or shared library containing that function; home directories rewritten to /home/user/.",
        "region_id": "Region within the run; joins to region_map.region_id.",
        "access_latency_ns": "Load latency reported by PEBS (perf's weight field).",
        "cache_level": "Where the access was satisfied (L1, L2, L3, LFB, RAM, ...).",
        "numa_node": "NUMA node of the access.",
        "cpu_id": "CPU the sample was taken on, when recorded.",
        "thread_id": "OS thread id.",
        "token_index": "Output token being generated, when known.",
        "context_tokens": "Context length in tokens at the time of the sample, when known.",
        "sample_weight": "1 / keep-probability in this release: 1.0 for kept-in-full regions, 1/runtime-fraction for subsampled runtime rows.",
    },
    "prompt_outputs.csv": {
        "prompt_index": "Prompt within the run.",
        "prompt": "Prompt given to the model.",
        "output": "Text the model generated.",
        "prompt_tokens": "Prompt length in tokens.",
        "tokens_generated": "Tokens generated for this prompt.",
        "duration_s": "Generation time for this prompt, seconds.",
    },
    "sample_calibration.csv": {
        "context_length": "Context length of the calibrated configuration.",
        "prompts": "Prompts per run in the calibrated configuration.",
        "load_sample_period": "Load sampling period chosen so runs of this configuration collect a similar number of samples.",
        "calibration_samples": "Samples collected by the calibration run.",
        "recorded_at": "UTC time of calibration.",
    },
}

_FILE_DOCS = {
    "runs.csv": "One row per run: model, configuration, validity, machine, sampling parameters, thermal readings and the tracer's own overhead. Start here.",
    "metrics.csv": "Per run and region type: retention percentiles, write-once ratio, reads per write, working set and MRM suitability, computed from the full (unsubsampled) trace.",
    "summary_by_region.csv": "Headline table: each region type summarised across all runs, with spread across runs.",
    "summary_by_model.csv": "Region metrics averaged per model, ordered by parameter count.",
    "summary_by_output_length.csv": "Region metrics averaged per output length; context is fixed, so this shows the KV cache filling.",
    "region_map.parquet": "Address ranges of each run and the memory region each belongs to.",
    "kv_block_lifecycle.parquet": "KV-cache blocks per run: first sampled write, last sampled read and lifetime. Lifetime is null where no write was sampled.",
    "trace.parquet": "Sampled memory accesses of all runs, sorted by run. weight, kv_cache and unknown rows are complete; runtime rows are subsampled (see sample_weight).",
    "prompt_outputs.csv": "The prompt and the text each run generated, showing the model did real work while traced.",
    "sample_calibration.csv": "Load sampling period chosen per model and output length.",
    "README.md": "Data card: what was run, how to read the sampling, labels, caveats.",
    "dataset-cover-image.png": "Cover image: one run's sampled accesses over time, coloured by memory region.",
    "sha256sums.txt": "SHA-256 checksum of every file.",
}


def _docs_for(file: str, column: str) -> str | None:
    return _COLUMN_DOCS.get(file, {}).get(column) or _KEY_DOCS.get(column)


# ----------------------------------------------------------------- privacy ---

class Anonymiser:
    def __init__(self, machine_id: str, alias: str, usernames: set[str]):
        self.machine_id = machine_id
        self.alias = alias
        self.usernames = {u for u in usernames if u and u != "root"}

    def text(self, s: str) -> str:
        s = s.replace(self.machine_id, self.alias)
        s = _HOME_RE.sub("/home/user/", s)
        for u in self.usernames:
            s = re.sub(re.escape(u), "user", s, flags=re.IGNORECASE)
        return s

    def obj(self, o):
        if isinstance(o, dict):
            return {k: self.obj(v) for k, v in o.items()}
        if isinstance(o, list):
            return [self.obj(v) for v in o]
        if isinstance(o, str):
            return self.text(o)
        return o

    def arrow_strings(self, col):
        col = pc.replace_substring_regex(col, _HOME_RE.pattern, "/home/user/")
        for u in self.usernames:
            col = pc.replace_substring_regex(col, f"(?i){re.escape(u)}", "user")
        return col


def _usernames() -> set[str]:
    names = {getpass.getuser(), os.environ.get("SUDO_USER", "")}
    m = re.match(r"/home/([^/]+)/", str(REPO) + "/")
    if m:
        names.add(m.group(1))
    return names


def _default_alias(machine_id: str, usernames: set[str]) -> str:
    for u in sorted(usernames, key=len, reverse=True):
        if u and machine_id.lower().startswith(u.lower() + "-"):
            return machine_id[len(u) + 1:]
    return "machine-01"


# --------------------------------------------------------------- selection ---

def select_runs(results: pathlib.Path, include_contaminated: bool) -> pd.DataFrame:
    manifest = pd.read_csv(results / "manifest.csv")
    if not include_contaminated:
        manifest = manifest[manifest["run_validity"] != "contaminated"]
    dirs, missing = [], []
    for row in manifest.itertuples():
        d = results / row.machine_id / row.model_id / row.run_id
        absent = [f for f in _RUN_FILES if not (d / f).exists()]
        if absent:
            missing.append(f"{row.run_id}: {', '.join(absent)}")
        dirs.append(d)
    if missing:
        sys.exit("runs listed in manifest.csv are incomplete:\n  " + "\n  ".join(missing))

    # A half-finished `experiment_runner.py --rederive-metrics` leaves runs
    # with metrics from two code versions; publishing that mixes them.
    headers = {d: tuple(pd.read_csv(d / "metrics.csv", nrows=0).columns) for d in dirs}
    if len(set(headers.values())) > 1:
        common = max(set(headers.values()), key=list(headers.values()).count)
        odd = sorted(d.name for d, h in headers.items() if h != common)
        sys.exit(f"metrics.csv columns differ across runs ({len(odd)} of {len(dirs)} differ, "
                 "e.g. " + ", ".join(odd[:3]) + "). Finish `experiment_runner.py "
                 "--rederive-metrics` first.")
    newest_metrics = max((d / "metrics.csv").stat().st_mtime for d in dirs)
    stale = [s for s in _SUMMARIES
             if (results / "analysis" / s).stat().st_mtime < newest_metrics]
    if stale:
        sys.exit(f"results/analysis/{', '.join(stale)} predate the newest metrics.csv. Re-run "
                 "notebooks/scripts/results_exploratory_data_analysis.py first.")
    manifest = manifest.assign(run_dir=dirs)
    return manifest.sort_values(["model_id", "max_tokens_per_prompt", "rep"]).reset_index(drop=True)


def _keys(run) -> dict:
    return {"run_id": run.run_id, "model_id": run.model_id,
            "output_tokens": int(run.max_tokens_per_prompt), "rep": int(run.rep)}


def _with_keys(df: pd.DataFrame, run) -> pd.DataFrame:
    keys = pd.DataFrame(_keys(run), index=df.index)
    return pd.concat([keys, df.drop(columns=[c for c in keys if c in df])], axis=1)


# ------------------------------------------------------------------ traces ---

def write_traces(runs: pd.DataFrame, dst: pathlib.Path, mode: str,
                 runtime_fraction: float, seed: int, anon: Anonymiser) -> list[dict]:
    """Stream every run's trace into one parquet file, one run after another."""
    writer = None
    schema = None
    stats = []
    try:
        for i, run in enumerate(runs.itertuples(), 1):
            rng = np.random.default_rng([seed, zlib.crc32(run.run_id.encode())])
            k = _keys(run)
            full: dict[str, int] = {}
            kept: dict[str, int] = {}
            for batch in pq.ParquetFile(run.run_dir / "trace.parquet").iter_batches(
                    batch_size=_TRACE_BATCH_ROWS):
                tbl = pa.Table.from_batches([batch])
                region = tbl.column("region_type").to_numpy(zero_copy_only=False).astype(str)
                if mode == "stratified":
                    is_runtime = region == "runtime"
                    keep = ~is_runtime | (rng.random(tbl.num_rows) < runtime_fraction)
                    weight = np.where(is_runtime, 1.0 / runtime_fraction, 1.0)[keep]
                    tbl = tbl.filter(pa.array(keep))
                    kept_region = region[keep]
                else:
                    weight = np.ones(tbl.num_rows)
                    kept_region = region
                for counts, values in ((full, region), (kept, kept_region)):
                    for r, n in zip(*np.unique(values, return_counts=True)):
                        counts[r] = counts.get(r, 0) + int(n)
                n = tbl.num_rows
                tbl = tbl.drop_columns(["run_id"])
                tbl = tbl.set_column(tbl.schema.get_field_index("dso"), "dso",
                                     anon.arrow_strings(tbl.column("dso")))
                cols = [pa.array([k["run_id"]] * n, pa.string()),
                        pa.array([k["model_id"]] * n, pa.string()),
                        pa.array(np.full(n, k["output_tokens"], dtype="int32")),
                        pa.array(np.full(n, k["rep"], dtype="int8"))]
                tbl = pa.Table.from_arrays(
                    cols + tbl.columns + [pa.array(weight.astype("float32"))],
                    names=["run_id", "model_id", "output_tokens", "rep",
                           *tbl.column_names, "sample_weight"])
                if writer is None:
                    schema = tbl.schema
                    writer = pq.ParquetWriter(dst, schema, compression="zstd")
                writer.write_table(tbl.cast(schema))
            stats.append({"run_id": run.run_id, "full": full, "kept": kept})
            print(f"[{i:>2}/{len(runs)}] {run.run_id}: "
                  f"{sum(full.values()):,} -> {sum(kept.values()):,} trace rows")
    finally:
        if writer is not None:
            writer.close()
    return stats


# ---------------------------------------------------------------- tables ---

def build_runs_table(runs: pd.DataFrame, anon: Anonymiser) -> pd.DataFrame:
    base = runs[list(_MANIFEST_COLS)].rename(columns=_MANIFEST_COLS).copy()
    base["machine_id"] = anon.alias
    flat = []
    for r in runs.itertuples():
        md = anon.obj(json.loads((r.run_dir / "metadata.json").read_text()))
        row = pd.json_normalize(md, sep=".").iloc[0].to_dict()
        row = {k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()}
        flat.append({k: v for k, v in row.items()
                     if k not in _METADATA_DROP and k not in ("run_id", "machine_id")})
    df = pd.concat([base, pd.DataFrame(flat)], axis=1)
    for dup, kept in _METADATA_DUPLICATES.items():
        if dup in df and kept in df and (df[dup].astype(str) == df[kept].astype(str)).all():
            df = df.drop(columns=dup)
    return df


def build_prompt_outputs(runs: pd.DataFrame, anon: Anonymiser) -> pd.DataFrame:
    rows = []
    for r in runs.itertuples():
        po = anon.obj(json.loads((r.run_dir / "prompt_outputs.json").read_text()))
        for p in po["prompts"]:
            rows.append({**_keys(r), "prompt_index": p["index"], "prompt": p["prompt"],
                         "output": p["output"], "prompt_tokens": p.get("prompt_tokens"),
                         "tokens_generated": p.get("tokens_generated"),
                         "duration_s": p.get("duration_s")})
    return pd.DataFrame(rows)


def build_calibration(path: pathlib.Path) -> pd.DataFrame:
    rows = []
    for key, v in json.loads(path.read_text()).items():
        rows.append({"model_id": key.split("__")[0], "context_length": v["ctx"],
                     "output_tokens": v["output_tokens"], "prompts": v["prompts"],
                     "load_sample_period": v["period"], "calibration_samples": v["samples"],
                     "recorded_at": v["recorded_at"]})
    return pd.DataFrame(rows).sort_values(["model_id", "output_tokens"])


def write_stacked(runs: pd.DataFrame, out: pathlib.Path) -> None:
    pd.concat([_with_keys(pd.read_csv(r.run_dir / "metrics.csv"), r)
               for r in runs.itertuples()], ignore_index=True) \
        .to_csv(out / "metrics.csv", index=False)
    for name in ("region_map.parquet", "kv_block_lifecycle.parquet"):
        df = pd.concat([_with_keys(pd.read_parquet(r.run_dir / name), r)
                        for r in runs.itertuples()], ignore_index=True)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False).replace_schema_metadata(None),
                       out / name, compression="zstd")


# ------------------------------------------------------------ cover image ---

def write_cover(out: pathlib.Path, run, path: pathlib.Path, seed: int) -> None:
    """One run's sampled accesses over time, coloured by region."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Reference palette, categorical slots 1-3 (validated all-pairs), light mode.
    colours = {"runtime": "#1baf7a", "weight": "#2a78d6", "kv_cache": "#eb6834"}
    labels = {"weight": "model weights", "kv_cache": "KV cache", "runtime": "runtime"}
    surface, ink, muted = "#fcfcfb", "#0b0b0b", "#52514e"

    df = pq.read_table(path, columns=["timestamp_ns", "address_page", "region_type"],
                       filters=[("run_id", "=", run.run_id)]).to_pandas()
    df = df[df["region_type"].isin(list(colours))]
    rng = np.random.default_rng(seed)
    rt = df[df["region_type"] == "runtime"]
    df = pd.concat([rt.iloc[rng.random(len(rt)) < min(1.0, 60_000 / max(len(rt), 1))],
                    df[df["region_type"] != "runtime"]])
    # Dense page rank closes the multi-terabyte gaps between mappings.
    df["y"] = df["address_page"].rank(method="dense")
    df["t"] = (df["timestamp_ns"] - df["timestamp_ns"].min()) / 1e9

    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=160)  # 1120 x 560 px
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    for region in ("runtime", "weight", "kv_cache"):  # draw rare regions on top
        g = df[df["region_type"] == region]
        ax.scatter(g["t"], g["y"], s=1.2 if region == "runtime" else 2.0, linewidths=0,
                   color=colours[region], alpha=0.5 if region == "runtime" else 0.8,
                   label=labels[region], rasterized=True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(muted)
    ax.set_yticks([])
    ax.tick_params(colors=muted, labelsize=7)
    ax.set_xlabel("seconds into run", color=muted, fontsize=7)
    ax.set_ylabel("address (page order)", color=muted, fontsize=7)
    fig.text(0.04, 0.93, "MRM-Trace", color=ink, fontsize=15, weight="bold")
    fig.text(0.04, 0.865, f"Sampled memory accesses of LLM inference by region — {run.model_id}",
             color=muted, fontsize=8)
    leg = ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.17), ncol=3, fontsize=7,
                    frameon=False, markerscale=6, handletextpad=0.3, columnspacing=1.2)
    for t in leg.get_texts():
        t.set_color(ink)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.8, bottom=0.12)
    fig.savefig(out / "dataset-cover-image.png", facecolor=surface)
    plt.close(fig)


# -------------------------------------------------------------- data card ---

def _thin_support_note(out: pathlib.Path) -> str:
    """Name the (run, region) pairs whose retention rests on under half their regions."""
    m = pd.read_csv(out / "metrics.csv")
    thin = m[m["retention_support_frac"] < 0.5]
    if thin.empty:
        return ""
    rows = "\n".join(f"| `{r.run_id}` | {r.region_type} | {r.n_regions_observable:,} / "
                     f"{r.n_regions:,} ({r.retention_support_frac:.0%}) |"
                     for r in thin.itertuples())
    return (f"\nIn {len(thin)} (run, region) pair(s), retention rests on **under half** of the "
            "regions, so its percentiles describe the better-sampled minority:\n\n"
            "| run | region | observable regions |\n|---|---|---|\n" + rows + "\n")


def write_data_card(out: pathlib.Path, runs: pd.DataFrame, rt: pd.DataFrame,
                    trace_stats: list[dict], args, alias: str) -> str:
    """Write README.md and return the Kaggle description (README minus schemas)."""
    first = rt.iloc[0]
    full: dict[str, int] = {}
    kept: dict[str, int] = {}
    for s in trace_stats:
        for k, v in s["full"].items():
            full[k] = full.get(k, 0) + v
        for k, v in s["kept"].items():
            kept[k] = kept.get(k, 0) + v
    region_rows = "\n".join(f"| {k} | {full[k]:,} | {kept.get(k, 0):,} |"
                            for k in sorted(full, key=full.get, reverse=True))
    models = ", ".join(sorted(runs["model_id"].unique()))
    tokens = ", ".join(str(t) for t in sorted(runs["max_tokens_per_prompt"].unique()))
    validity = ", ".join(f"{n} {v}" for v, n in runs["run_validity"].value_counts().items())
    if args.trace_mode == "stratified":
        trace_note = (f"Every `weight`, `kv_cache` and `unknown` row is kept; `runtime` rows are kept "
                      f"with probability {args.runtime_fraction} (seed {args.seed}), and carry "
                      f"`sample_weight` = {1 / args.runtime_fraction:g}. `metrics.csv` was computed "
                      "from the full trace, so recomputing runtime retention or write-once ratio "
                      "from this subsample will not reproduce it exactly.")
    else:
        trace_note = "Traces are published unthinned; `sample_weight` is 1.0 throughout."

    def rows(f: str) -> str:
        p = out / f
        n = pq.ParquetFile(p).metadata.num_rows if p.suffix == ".parquet" else len(pd.read_csv(p))
        return f"{n:,}"

    files = "\n".join(f"| `{f}` | {rows(f)} | {_FILE_DOCS[f]} |" for f in _TABULAR_FILES)

    body = f"""# MRM-Trace: memory access traces of LLM inference on CPU

Hardware-sampled memory accesses from {len(runs)} LLM inference runs, each access labelled
with the memory region it touched — model weights, KV cache or runtime. The dataset measures
how long data stays live in each region and how often it is rewritten, the properties that
decide whether a region could live in Managed-Retention Memory (MRM; Legtchenko et al., 2025):
memory that holds data only as long as it is needed instead of for years.

## What was run

| | |
|---|---|
| Models | {models} (GGUF, Q4_K_M, llama.cpp on CPU) |
| Context length | 2048 |
| Output tokens | {tokens} (generated to the limit, EOS ignored) |
| Repetitions | {runs['rep'].nunique()} per configuration |
| Runs | {len(runs)} ({validity}) |
| Machine | `{alias}`: {first['cpu_model']}, {first['hardware.mem_total_mb'] / 1024:.0f} GB RAM, turbo {'on' if first['hardware.turbo_enabled'] else 'off'}, governor `{first['hardware.cpu_freq_governor']}` |
| OS | {first['os']} |
| Collector | `perf mem` (Intel PEBS), {first['software.perf_version']}, spawn mode |
| Tool | mrm-trace {first['software.mrm_trace_version']} |

## Files

| file | rows | contents |
|---|---|---|
{files}

`run_id` joins every table; `model_id`, `output_tokens` and `rep` are repeated on each for
convenience. Every column has a description in the file's column metadata.

## Sampling: read this first

`perf mem` **samples** memory accesses; it does not record every one. Loads are sampled every
`collector.sample_period` loads (calibrated per configuration so runs collect similar sample
counts), and stores 500x less often. So:

- A region seen once has no retention interval. `retention_*` covers regions with at least two
  samples; `n_regions_observable` and `retention_support_frac` say how many that was.
- A KV block's lifetime needs its write. `lifetime_ns` is **null** when no store to the block
  was sampled (`write_observed = false`), which is most blocks. Absent is never recorded as 0.
- `kv_cache` write-once ratio = 1.000 with zero variance is the signature of at most one
  sampled store per address, **not** evidence of write-once behaviour. Treat `write_once_ratio`
  and `reads_per_write` as unsupported for `kv_cache`.
- `weight` has no sampled stores because the model file is mapped read-only. That is a real
  finding, not a gap.

{_thin_support_note(out)}
## How accesses are labelled

Region types come from address ranges only, never from function names:

| region | source | status |
|---|---|---|
| `weight` | the mmap'd GGUF file, from `/proc/self/maps` | exact |
| `kv_cache` | engine allocation captured at allocation time | exact when unambiguous, else absent |
| `runtime` | thread stacks, heap, malloc arenas, shared libraries | exact, but a catch-all whose contents vary |
| `unknown` | in no known mapping | — |

## Trace subsampling

{trace_note}

| region | rows in full traces | rows published |
|---|---|---|
{region_rows}

## Validity

Each run records the tracer's own CPU and memory overhead, baseline CPU load, temperature and
throttling, and is classified `clean`, `marginal` or `contaminated` from them (`runs.csv`:
`run_validity`; reasons in `thermal.validity_failure_reasons`).
{'Contaminated runs are excluded from this release.' if not args.include_contaminated else 'Contaminated runs are included; filter on run_validity.'}

## Caveats

- Retention values are **observed lifetimes on this CPU's cache hierarchy under this schedule**,
  not hardware retention requirements. What transfers to other memory systems is the logical
  structure: which regions are written once, how read-heavy they are, their relative ordering.
- One machine, one engine (llama.cpp on CPU), one context length.
- Home directories in the trace `dso` column are rewritten to `/home/user/`, and the machine is
  identified by an alias.

## Getting started

```python
import pandas as pd
runs = pd.read_csv("runs.csv")
metrics = pd.read_csv("metrics.csv")
one_run = pd.read_parquet("trace.parquet", filters=[("run_id", "==", runs.run_id[0])])
```

## Provenance

Collected with the open-source mrm-trace tool and packaged
{datetime.now(timezone.utc).strftime('%Y-%m-%d')} by `notebooks/scripts/prepare_kaggle_dataset.py`.
Checksums are in `sha256sums.txt`.
"""
    schemas = ["## Column reference", ""]
    for f in _TABULAR_FILES:
        schemas += [f"### {f}", "", "| column | description |", "|---|---|"]
        schemas += [f"| `{c}` | {_docs_for(f, c)} |" for c in _columns(out / f)]
        schemas.append("")
    (out / "README.md").write_text(body + "\n" + "\n".join(schemas))
    return body


# ---------------------------------------------------- kaggle metadata ---

_TABULAR_FILES = ("runs.csv", "metrics.csv", "summary_by_region.csv", "summary_by_model.csv",
                  "summary_by_output_length.csv", "region_map.parquet",
                  "kv_block_lifecycle.parquet", "trace.parquet", "prompt_outputs.csv",
                  "sample_calibration.csv")


def _columns(path: pathlib.Path) -> list[str]:
    if path.suffix == ".parquet":
        return pq.read_schema(path).names
    return list(pd.read_csv(path, nrows=0).columns)


def _kaggle_type(path: pathlib.Path, column: str) -> str:
    if column in ("recorded_at",):
        return "datetime"
    if path.suffix == ".parquet":
        t = pq.read_schema(path).field(column).type
        if pa.types.is_boolean(t):
            return "boolean"
        if pa.types.is_integer(t):
            return "integer"
        if pa.types.is_floating(t):
            return "numeric"
        return "string"
    s = pd.read_csv(path, usecols=[column])[column]
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_integer_dtype(s):
        return "integer"
    if pd.api.types.is_float_dtype(s):
        return "numeric"
    return "string"


def check_docs(out: pathlib.Path) -> None:
    missing = [f"{f}: {c}" for f in _TABULAR_FILES for c in _columns(out / f)
               if not _docs_for(f, c)]
    if missing:
        sys.exit("columns without a description (add them to _COLUMN_DOCS):\n  "
                 + "\n  ".join(missing))


def write_kaggle_metadata(out: pathlib.Path, description: str, args) -> None:
    resources = []
    for f in sorted(p.name for p in out.iterdir() if p.is_file()):
        if f == "dataset-metadata.json":
            continue
        res = {"path": f, "description": _FILE_DOCS[f]}
        if f in _TABULAR_FILES:
            res["schema"] = {"fields": [
                {"name": c, "description": _docs_for(f, c), "type": _kaggle_type(out / f, c)}
                for c in _columns(out / f)]}
        resources.append(res)
    meta = {
        "title": _TITLE,
        "subtitle": _SUBTITLE,
        "description": description,
        "id": args.kaggle_id,
        "licenses": [{"name": args.license}],
        "keywords": _KEYWORDS,
        "expectedUpdateFrequency": args.update_frequency,
        "userSpecifiedSources": (
            "Collected by the dataset author on one bare-metal Linux laptop with the open-source "
            "mrm-trace tool. Each run launches llama.cpp (via llama-cpp-python) under "
            "`perf mem record` in spawn mode, so Intel PEBS samples loads (with latency and cache "
            "level) and stores from the process's first instruction. Accesses are labelled by "
            "address range: the model file's mmap from /proc/self/maps, and the KV cache from the "
            "engine's allocation. Every run records machine state, thermal readings and the "
            "tracer's own overhead, and is classified clean/marginal/contaminated. Models: public "
            "GGUF Q4_K_M quantisations of Llama 3.2 1B/3B, Llama 3.1 8B, Mistral 7B and "
            "Phi-3.5 mini. Concept: Legtchenko et al., 'Managed-Retention Memory: A New Class of "
            "Memory for the AI Era', 2025."),
        "image": "dataset-cover-image.png",
        "resources": resources,
    }
    assert 6 <= len(_TITLE) <= 50 and 20 <= len(_SUBTITLE) <= 80
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")


# --------------------------------------------------------------- finishing ---

def write_checksums(out: pathlib.Path) -> None:
    lines = []
    for f in sorted(p for p in out.iterdir() if p.is_file()
                    and p.name not in ("sha256sums.txt", "dataset-metadata.json")):
        h = hashlib.sha256()
        with f.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {f.name}")
    (out / "sha256sums.txt").write_text("\n".join(lines) + "\n")


def privacy_scan(out: pathlib.Path, needles: set[str]) -> list[str]:
    pattern = re.compile("|".join(re.escape(u) for u in needles), re.IGNORECASE)
    hits = []
    for f in sorted(p for p in out.iterdir() if p.is_file()):
        if f.suffix in (".csv", ".json", ".md", ".txt"):
            if pattern.search(f.read_text(errors="replace")):
                hits.append(f.name)
        elif f.suffix == ".parquet":
            pfile = pq.ParquetFile(f)
            str_cols = [fld.name for fld in pfile.schema_arrow
                        if pa.types.is_string(fld.type) or pa.types.is_large_string(fld.type)]
            found = set()
            for batch in pfile.iter_batches(columns=str_cols, batch_size=_TRACE_BATCH_ROWS):
                for name, col in zip(batch.schema.names, batch.columns):
                    if name not in found and any(
                            pattern.search(v) for v in pc.unique(col).drop_null().to_pylist()):
                        found.add(name)
            hits += [f"{f.name}:{n}" for n in sorted(found)]
    return hits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", type=pathlib.Path, default=REPO / "results")
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "datasets" / "mrm-trace-kaggle")
    ap.add_argument("--trace-mode", choices=("stratified", "full"), default="stratified")
    ap.add_argument("--runtime-fraction", type=float, default=0.10,
                    help="keep probability for runtime trace rows in stratified mode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-contaminated", action="store_true")
    ap.add_argument("--machine-alias", help="replaces machine_id (default: machine_id minus the username prefix)")
    ap.add_argument("--kaggle-id",
                    default=f"{os.environ.get('KAGGLE_USERNAME', 'USERNAME')}/{_SLUG}",
                    help="owner/slug; defaults to $KAGGLE_USERNAME/" + _SLUG)
    ap.add_argument("--license", default="CC0-1.0",
                    help="CC0-1.0 matches the licence the dataset was first published under")
    ap.add_argument("--update-frequency", default="never",
                    choices=("never", "annually", "quarterly", "monthly", "weekly", "daily"))
    ap.add_argument("--cover-model", default="llama-3.2-1b",
                    help="model whose first run is drawn on the cover image")
    ap.add_argument("--force", action="store_true", help="replace an existing --out directory")
    args = ap.parse_args()

    if not 0 < args.runtime_fraction <= 1:
        ap.error("--runtime-fraction must be in (0, 1]")
    if args.out.exists():
        if not args.force:
            ap.error(f"{args.out} exists; pass --force to replace it")
        shutil.rmtree(args.out)

    runs = select_runs(args.results, args.include_contaminated)
    machines = runs["machine_id"].unique()
    if len(machines) != 1:
        sys.exit(f"expected one machine, found {list(machines)}")
    usernames = _usernames()
    alias = args.machine_alias or _default_alias(machines[0], usernames)
    anon = Anonymiser(machines[0], alias, usernames)
    print(f"{len(runs)} runs from {machines[0]} -> machine alias '{alias}'")

    out = args.out
    out.mkdir(parents=True)

    trace_stats = write_traces(runs, out / "trace.parquet", args.trace_mode,
                               args.runtime_fraction, args.seed, anon)
    write_stacked(runs, out)
    rt = build_runs_table(runs, anon)
    rt.to_csv(out / "runs.csv", index=False)
    build_prompt_outputs(runs, anon).to_csv(out / "prompt_outputs.csv", index=False)
    build_calibration(args.results / ".sample_calibration.json") \
        .to_csv(out / "sample_calibration.csv", index=False)
    for src, dst in _SUMMARIES.items():
        shutil.copy2(args.results / "analysis" / src, out / dst)

    check_docs(out)
    cover = runs[runs["model_id"] == args.cover_model]
    write_cover(out, next((cover if len(cover) else runs).itertuples()),
                out / "trace.parquet", args.seed)
    description = write_data_card(out, runs, rt, trace_stats, args, alias)
    write_checksums(out)
    write_kaggle_metadata(out, description, args)

    hits = privacy_scan(out, anon.usernames | {machines[0]})
    if hits:
        sys.exit("username still present in:\n  " + "\n  ".join(hits))

    size = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    print(f"\nwrote {out} ({size / 2**20:,.0f} MB). "
          "Every file and column described; privacy scan clean.")
    if args.kaggle_id.startswith("USERNAME/"):
        print("Set --kaggle-id or $KAGGLE_USERNAME before uploading.")


if __name__ == "__main__":
    main()

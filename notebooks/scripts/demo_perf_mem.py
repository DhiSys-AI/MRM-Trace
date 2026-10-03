#!/usr/bin/env python3
"""
TinyLlama 1.1B — live mrm-trace collection via perf mem (hardware PMU).

Unlike memray, perf mem gives:
  - Hardware load/store events with NANOSECOND timestamps
  - Real symbol names from libllama.so / libggml.so
  - Cache level (L1/L2/L3/DRAM), NUMA node, CPU core per sample
  → retention_p99_s, region_map, and KV lifecycle will all be populated

perf mem uses spawn mode: perf launches this script with --headless so the
instrumented process starts under perf from the very first instruction,
avoiding the kernel-level attach limitation on PEBS load-latency events
(`perf mem record -p PID` fails on Ubuntu 24.04 / kernel 6.8+ with
"No such file or directory" even as root — Intel PEBS load-latency events
require the target process to start under perf).

Requirements:
    sudo apt install linux-perf                # install perf
    sudo perf stat true                        # verify PMU works
    pip install llama-cpp-python               # or with conda/index as needed

Must run as root:
    sudo -E venv/bin/python notebooks/scripts/demo_perf_mem.py --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

    (-E preserves the venv PATH so the correct python/mrm-trace is used;
    plain `sudo -E python` resolves to system Python — always use the full
    venv path.)

Model download:
    mkdir -p models
    wget -P models/ https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

WSL2 note:
    Run `sudo perf stat -e cpu/mem-loads,ldlat=20/P true` first.
    If that fails (Permission denied / No such event), your WSL2 kernel
    does not expose hardware PMU to guests. In that case use demo_tinyllama.py
    (memray) or run on bare-metal Linux.
"""

import argparse
import collections
import dataclasses
import json
import os
import pathlib
import re
import resource
import shutil
import subprocess
import sys
import threading
import time

import pandas as pd
import psutil

from mrm_trace.analyser import (
    compute_all,
    recommend_granularity,
    retention_granularity_sweep,
)
from mrm_trace.collector.base import CollectorMeta
from mrm_trace.collector.perf_runner import PerfMemRunner, clock_args
from mrm_trace.collector.process_monitor import ProcessMonitor
from mrm_trace.config.schema import EngineType, ModelConfig, RunConfig
from mrm_trace.engines.base import PhaseEvent
from mrm_trace.engines.llamacpp import LlamaCppEngine
from mrm_trace.labeller import TraceLabeller
from mrm_trace.labeller.llamacpp_regions import (
    expected_kv_bytes,
    extract_llamacpp_regions,
    snapshot_proc_maps,
    summarise_regions,
)
from mrm_trace.parser.perf_script_parser import parse_perf_script
from mrm_trace.reporter.run_exporter import RunExporter
from mrm_trace.reporter.trace_qa import (
    check_trace_quality,
    check_trace_quality_counts,
)
from mrm_trace.telemetry.baseline import capture_baseline
from mrm_trace.telemetry.observer_effect import compute_observer_effect
from mrm_trace.telemetry.system_monitor import CpuFreqMonitor, CpuFreqLimits, GpuMonitor, read_cpu_freq_limits
from mrm_trace.telemetry.thermal import ThermalMonitor, read_package_temp_c
from mrm_trace.telemetry.validity import classify_from_metadata
from mrm_trace.utils.ids import generate_run_id
from mrm_trace.utils.machine import machine_slug
from mrm_trace.validate_machine import collect_machine_context

DEFAULT_MODEL = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# Address granularity at which a "region" is defined for retention analysis.
# 4 KiB (one page) is the unit the OS and hardware manage, but it is frequently
# too fine for perf mem's sample density — see the sweep printed at the end of
# every run, and override with --region-granularity when it says so.
_REGION_GRANULARITY_BYTES = 4096

PROMPTS = [
    # P1–P5: general history and social topics — accessible to all model sizes
    "Describe the rise and fall of the Roman Empire. What were the key political and military factors behind its decline?",
    "What caused the First World War, and how did it reshape the political map of Europe and the wider world?",
    "How did the Industrial Revolution transform working conditions, daily life, and urbanisation in 19th-century Britain?",
    "What were the main causes and lasting consequences of the French Revolution? How did its ideals spread beyond France?",
    "How did the Silk Road shape cultural exchange and trade between Asia, the Middle East, and Europe over centuries?",
    # P6: long-form prose — drives sustained KV cache growth and extended retention window
    "Summarise the history of Africa from ancient civilisations through colonialism to independence movements in the 20th century.",
    # P7: semantic follow-up to P1 (same knowledge domain: Roman Empire) —
    # exercises KV cache reuse or overwrite: weight regions accessed for P1 are
    # revisited here, and KV pages written during P1 may be read before P7 overwrites them.
    "How did the Roman Empire expand and eventually declined. What were the key internal and external factors behind its decline?",
]

MODEL_ID = "tinyllama-1.1b-q4km-perf"


def parse_args():
    p = argparse.ArgumentParser(description="TinyLlama mrm-trace demo (perf mem)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Path to GGUF model file")
    p.add_argument(
        "--model-id", dest="model_id", default=None,
        help="Model identifier used for the results/<model_id>/ directory and metadata.json "
             f"(default: derived from --model's filename; {MODEL_ID!r} for the bundled TinyLlama model)",
    )
    p.add_argument("--ctx", type=int, default=2048, help="Context length")
    p.add_argument("--tokens", type=int, default=128, help="Max output tokens per prompt")
    p.add_argument("--batch-size", dest="batch_size", type=int, default=1, help="Number of prompts in batch")
    p.add_argument("--rep", type=int, default=0, help="Repetition index (0 = warmup, 1+ = data runs)")
    p.add_argument("--out", default="results", help="Output base directory")
    p.add_argument(
        "--machine", default=None,
        help="Machine identifier used as results/<machine>/ subdirectory "
             "(default: derived from hostname)",
    )
    p.add_argument(
        "--sample-period", type=int, default=4000,
        help="perf sampling period — lower = denser trace, higher overhead (default 4000)",
    )
    p.add_argument(
        "--region-granularity", dest="region_granularity", type=int,
        default=_REGION_GRANULARITY_BYTES,
        help="Bytes per region for retention analysis (default 4096 = one page). "
             "The end-of-run sweep reports whether this value is usable for the "
             "trace's sample density.",
    )
    p.add_argument(
        "--keep-perf-script", dest="keep_perf_script", action="store_true",
        help="keep raw/perf_script.txt after export (default: delete it — it is "
             "regenerable from perf.data and is the largest artifact of a run)",
    )
    p.add_argument(
        "--chunk-rows", dest="chunk_rows", type=int, default=0,
        help="rows per parse chunk (default 0 = size from available memory)",
    )
    p.add_argument(
        "--num-prompts", dest="num_prompts", type=int, default=None,
        help="Limit to first N prompts (default: all). Useful for quick debug runs.",
    )
    # Internal flags used when perf spawns this script as the traced workload
    p.add_argument("--headless", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--results-file", dest="results_file", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def check_pmu():
    """Verify the PMU events perf mem uses are available on this machine."""
    import tempfile

    perf = shutil.which("perf")
    if not perf:
        return False, "perf binary not found (install linux-perf)"

    # Probe with no -e flags so perf auto-selects CPU-appropriate events,
    # matching what the spawn-mode perf mem record command does. Explicit
    # events like "cpu/mem-loads,ldlat=20/P" require ldlat kernel support not
    # universally available even on Intel hardware.
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [perf, "mem", "record", "-o", f"{tmpdir}/perf.data", "--", "true"],
            capture_output=True, timeout=15,
        )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")
        return False, f"PMU events not available: {stderr[:300]}"
    return True, "ok"


def resolve_model_id(args, model_path: pathlib.Path) -> str:
    """
    Resolve the model_id used for the results/<model_id>/ directory and metadata.json.

    Explicit --model-id always wins. Otherwise, the bundled default TinyLlama model
    keeps the historical `tinyllama-1.1b-q4km-perf` id; any other --model gets a slug
    derived from its filename, so different models don't collide in the same results dir.
    """
    if args.model_id:
        return args.model_id
    if model_path.name == pathlib.Path(DEFAULT_MODEL).name:
        return MODEL_ID
    slug = re.sub(r"[^a-z0-9]+", "-", model_path.stem.lower()).strip("-")
    return f"{slug}-perf"


def build_model_cfg(model_path, model_id: str) -> ModelConfig:
    return ModelConfig(
        id=model_id,
        engine=EngineType.LLAMACPP,
        path=str(model_path),
        family="llama",
        params_billions=1.1,
        quantisation="Q4_K_M",
        dtype="float16",
    )


def build_phase_boundaries(phase_records):
    """
    Convert serialized phase records to parse_perf_script phase_boundaries.

    MUST use mono_ns, not wall_ns.  perf sample timestamps are CLOCK_MONOTONIC
    (seconds since boot, pinned via ``perf record -k CLOCK_MONOTONIC``), while
    wall_ns is time.time_ns() — the UNIX epoch, ~1.78e18 today.  Comparing the
    two puts every boundary ~56 years in the future, so _annotate_phase matches
    nothing and every row lands with phase=NULL.  That silently disables the
    phase-aware labelling mandated by ADR-3 and spec §5.
    """
    boundaries = []
    starts = {}
    phase_map = {
        PhaseEvent.MODEL_LOAD_START.value: (PhaseEvent.MODEL_LOAD_END.value, "weight_load"),
        PhaseEvent.GENERATION_START.value: (PhaseEvent.GENERATION_END.value, "generation"),
    }
    for rec in phase_records:
        event = rec["event"]
        if "mono_ns" not in rec:
            raise ValueError(
                "phase record has no mono_ns — the results file was written by an "
                "older build of this script.  Phase labelling cannot be recovered "
                "from wall_ns alone (different clock domain); re-run the collection."
            )
        mono_ns = rec["mono_ns"]
        if event in phase_map:
            starts[event] = mono_ns
        for start_ev, (end_ev, label) in phase_map.items():
            if event == end_ev and start_ev in starts:
                boundaries.append({
                    "phase": label,
                    "start_ns": starts.pop(start_ev),
                    "end_ns": mono_ns,
                })
    return boundaries


def _hardware_section(baseline, cpu_freq_stats, cpu_freq_limits=None) -> dict:
    lim: CpuFreqLimits = cpu_freq_limits or CpuFreqLimits(None, None, None, None)
    return {
        "baseline_cpu_pct": round(baseline.cpu_pct, 2),
        "baseline_cpu_pct_max": round(baseline.cpu_pct_max, 2),
        "mem_total_mb": round(baseline.mem_total_mb, 0),
        "mem_available_mb": round(baseline.mem_available_mb, 0),
        "mem_used_pct": round(baseline.mem_used_pct, 1),
        "load_avg_1m": round(baseline.load_avg_1m, 2) if baseline.load_avg_1m is not None else None,
        "cpu_freq_hw_max_mhz": round(lim.hw_max_mhz, 0) if lim.hw_max_mhz else None,
        "cpu_freq_scaling_max_mhz": round(lim.scaling_max_mhz, 0) if lim.scaling_max_mhz else None,
        "cpu_freq_scaling_min_mhz": round(lim.scaling_min_mhz, 0) if lim.scaling_min_mhz else None,
        "cpu_freq_governor": lim.governor,
        "cpu_freq_start_mhz": round(cpu_freq_stats.start_mhz, 0) if cpu_freq_stats else None,
        "cpu_freq_min_mhz": round(cpu_freq_stats.min_mhz, 0) if cpu_freq_stats else None,
        "cpu_freq_mean_mhz": round(cpu_freq_stats.mean_mhz, 0) if cpu_freq_stats else None,
        "cpu_freq_end_mhz": round(cpu_freq_stats.end_mhz, 0) if cpu_freq_stats else None,
    }


def _thermal_section(thermal_meta, baseline, observer_effect) -> dict:
    # Classified through the same entry point --reclassify uses, so a fresh run
    # and a later reclassification of its metadata can never disagree.  perf mem
    # is out-of-process, so its memory is judged against system RAM rather than
    # against the target's own footprint — see validity.py.
    oe = dict(observer_effect or {})
    oe.setdefault("baseline_cpu_pct", baseline.cpu_pct if baseline else 0.0)
    validity = classify_from_metadata({
        "thermal": thermal_meta,
        "observer_effect": oe,
    })
    # Keep the reasons: a verdict of "contaminated" with no stated cause cannot
    # be acted on, and it is the one field the data card needs to justify an
    # exclusion.
    return {
        **thermal_meta,
        "run_validity": validity.run_validity,
        "validity_failure_reasons": validity.failure_reasons,
    }


def build_metadata(
    model_cfg, run_id, runtime_metadata, gen_results, collector_meta,
    thermal_meta=None, total_wall_duration_s=None,
    baseline=None, cpu_freq_limits=None, cpu_freq_stats=None, gpu_stats=None, observer_effect=None,
    region_granularity=_REGION_GRANULARITY_BYTES,
    machine_ctx=None, machine_id=None,
    context_length=None, batch_size=None, output_tokens=None, rep_index=None,
):
    total_tokens = sum(r["output_tokens"] for r in gen_results)
    total_time = sum(r["duration_s"] for r in gen_results)
    metadata = {
        "run_id": run_id,
        "machine_id": machine_id or "",
        "model": {
            "id": model_cfg.id,
            "family": model_cfg.family,
            "params_billions": model_cfg.params_billions,
            "quantisation": model_cfg.quantisation,
            "dtype": model_cfg.dtype,
        },
        "engine": runtime_metadata,
        "generation": {
            "prompts": len(gen_results),
            "context_length": context_length,
            "batch_size": batch_size,
            "output_tokens": output_tokens,
            "rep_index": rep_index,
            "total_output_tokens": total_tokens,
            "total_duration_s": round(total_time, 3),
            "avg_tokens_per_second": round(total_tokens / total_time, 1) if total_time > 0 else 0.0,
        },
        "collector": {
            "type": "perf_mem",
            "mode": "spawn",
            # sample_period and event_strings are the acquisition parameters:
            # a sample count means nothing without the period it was taken at,
            # and the data card (workplan §12.2) has to state both.
            "sample_period": collector_meta.sample_period,
            "event_strings": collector_meta.event_strings,
            "region_granularity_bytes": region_granularity,
            "output_file": collector_meta.output_file,
            "output_file_size_bytes": collector_meta.output_file_size_bytes,
            "parse_success": collector_meta.parse_success,
            "warnings": collector_meta.warnings,
        },
    }

    if total_wall_duration_s is not None:
        metadata["timing"] = {
            "total_wall_duration_s": round(total_wall_duration_s, 3),
        }

    if machine_ctx is not None:
        metadata["software"] = machine_ctx["software"]

    if baseline is not None:
        hw = _hardware_section(baseline, cpu_freq_stats, cpu_freq_limits)
        if machine_ctx is not None:
            hw.update(machine_ctx["hardware_context"])
        metadata["hardware"] = hw

    gpu_section: dict = {}
    if gpu_stats is not None:
        gpu_section = {
            "name": gpu_stats.name,
            "vram_total_mb": gpu_stats.vram_total_mb,
            "vram_start_used_mb": gpu_stats.vram_start_used_mb,
            "vram_peak_used_mb": gpu_stats.vram_peak_used_mb,
            "vram_end_used_mb": gpu_stats.vram_end_used_mb,
            "gpu_util_pct_mean": round(gpu_stats.gpu_util_pct_mean, 1),
        }
    if machine_ctx is not None and machine_ctx.get("gpu_power_draw_w") is not None:
        gpu_section["power_draw_w"] = machine_ctx["gpu_power_draw_w"]
    if gpu_section:
        metadata["gpu"] = gpu_section

    if thermal_meta is not None:
        metadata["thermal"] = _thermal_section(thermal_meta, baseline, observer_effect)

    if observer_effect is not None:
        metadata["observer_effect"] = observer_effect

    return metadata


# ── Chunked parsing (ADR-2) ───────────────────────────────────────────────────
#
# A parsed row is a dict of 18 keys: 863 bytes measured on a real perf-mem
# trace.  Holding a whole run costs 863 B x row count — 26 GB at 30M rows, and
# the pipeline needs two passes over the rows (symbol prescan, then labelling),
# so a list-based path peaks at twice that.  Rows are therefore streamed in
# chunks and only the labelled DataFrame, which is ~5x leaner per row, is kept.
_ROW_DICT_BYTES = 863

# Pandas frame cost per labelled row, measured: 314 bytes with plain string
# columns, 130 once the repeated ones are dictionary-encoded (see
# _frame_from_chunk).  pd.concat holds the chunk frames and the combined frame
# at once, so the budget check below doubles this.
_FRAME_ROW_BYTES = 140

# Columns dictionary-encoded in the labelled frame.  pandas 3 stores str
# columns as inline bytes, so on a 30M-row trace dso costs 90 B a row, symbol
# 63 and run_id 26 — 70% of the frame is the same few hundred values repeated.
# op_type and region_type are deliberately left as str: the analysers group by
# them, and a categorical grouper's semantics depend on the pandas version.
_CATEGORICAL_COLS = ("run_id", "symbol", "dso", "cache_level")

# A chunk is transient — it is freed as soon as its frame is built — so it only
# needs a small slice of free memory.  The floor keeps tiny machines from
# chunking so finely that per-chunk overhead dominates; the ceiling keeps a
# large machine from allocating a multi-GB chunk it gains nothing from.
_CHUNK_MEM_FRACTION = 0.02
_CHUNK_ROWS_MIN = 50_000
_CHUNK_ROWS_MAX = 2_000_000

# Fraction of available memory the final labelled frame may occupy before the
# run is flagged as too dense to analyse on this machine.
_FRAME_MEM_BUDGET = 0.5

_PROGRESS_EVERY_ROWS = 5_000_000


def _auto_chunk_rows() -> int:
    """
    Pick a chunk size from free memory rather than a fixed constant, so the
    same script works on a 4 GB VM and a 128 GB workstation.
    """
    available = psutil.virtual_memory().available
    rows = int(available * _CHUNK_MEM_FRACTION / _ROW_DICT_BYTES)
    return max(_CHUNK_ROWS_MIN, min(rows, _CHUNK_ROWS_MAX))


def _chunks(rows, size: int):
    """Yield lists of at most *size* rows from an iterable."""
    chunk: list = []
    for row in rows:
        chunk.append(row)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _categorical_dtypes(run_id: str, value_sets: dict) -> dict:
    """
    Build one CategoricalDtype per dictionary-encoded column, from the value
    sets pass 1 collected.

    The dtype has to be identical across chunks: pd.concat only keeps the
    encoding when every frame shares one explicit CategoricalDtype, and falls
    back to inline strings when the chunks each carry their own categories.
    """
    dtypes = {"run_id": pd.CategoricalDtype([run_id])}
    for col in ("symbol", "dso", "cache_level"):
        values = sorted(v for v in value_sets[col] if v is not None)
        dtypes[col] = pd.CategoricalDtype(values)
    return dtypes


def _frame_from_chunk(rows: list, dtypes: dict) -> pd.DataFrame:
    """
    Build one chunk's DataFrame with the repeated string columns dictionary-
    encoded, so the concatenated frame costs ~130 bytes a row instead of ~314.
    """
    df = pd.DataFrame(rows)
    for col, dtype in dtypes.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    return df


def _check_frame_fits(n_rows: int, sample_period: int) -> None:
    """
    Warn before labelling if the labelled frame will not fit in free memory.

    Without this the run dies to the OOM killer minutes later with no
    indication of what to change.
    """
    available = psutil.virtual_memory().available
    # pd.concat holds the per-chunk frames and the combined frame at the same
    # time, so the transient peak is twice the finished frame.
    estimate = n_rows * _FRAME_ROW_BYTES * 2
    if estimate <= available * _FRAME_MEM_BUDGET:
        return
    affordable = int(available * _FRAME_MEM_BUDGET / (_FRAME_ROW_BYTES * 2))
    suggested = int(sample_period * (n_rows / affordable))
    print(
        f"  WARNING: {n_rows:,} rows need ~{estimate / 1e9:.1f} GB as a frame but only "
        f"{available / 1e9:.1f} GB is free.\n"
        f"           Analysis may be OOM-killed. Re-record with "
        f"--sample-period {suggested:,} (~{affordable:,} rows).",
        file=sys.stderr,
    )


def _print_validity(thermal: dict) -> None:
    """State the validity verdict and, when it is not clean, why."""
    verdict = thermal.get("run_validity", "unknown")
    reasons = thermal.get("validity_failure_reasons") or []
    print(f"  Run validity  : {verdict}")
    for reason in reasons:
        print(f"    - {reason}")


def _reclaim_perf_script(raw_dir: pathlib.Path, keep: bool = False) -> None:
    """
    Delete raw/perf_script.txt once the run has exported successfully.

    The text dump is the largest artifact of a run by far — 8.9 GB for 83
    seconds of generation, measured, and it grows linearly with run time — but
    it is pure intermediate state: ``perf script`` regenerates it from
    perf.data, and trace.parquet already holds the parsed result.  Keeping one
    per run is what turns the sweep's disk budget from hundreds of gigabytes
    into the size of a single run.

    perf.data is never touched: it is the primary raw artifact, referenced by
    metadata.json and destined for the Zenodo archive (workplan §12.3).  This
    runs only after export, so a failed parse still leaves everything in place
    (CLAUDE.md: "If parse fails, retain raw collector files untouched").
    """
    script_txt = raw_dir / "perf_script.txt"
    if not script_txt.exists():
        return
    size_gb = script_txt.stat().st_size / 1e9
    if keep:
        print(f"  Kept {script_txt.name} ({size_gb:.1f} GB) — --keep-perf-script")
        return
    try:
        script_txt.unlink()
        print(f"  Reclaimed {size_gb:.1f} GB: removed {script_txt.name} "
              f"(regenerate with: perf script -i {raw_dir / 'perf.data'})")
    except OSError as exc:
        print(f"  WARN: could not remove {script_txt}: {exc}", file=sys.stderr)


def _print_granularity_sweep(labelled, granularity_used: int) -> None:
    """
    Report how retention responds to region granularity, per region type.

    Retention is (last access - first access) of a region, so it needs the same
    region sampled more than once.  perf mem samples sparsely — and its load
    samples are PEBS cache-miss samples, so re-reads that hit in cache are never
    recorded.  Too fine a granule gives one sample per region and retention 0 by
    construction; too coarse and every granule spans the run, so retention just
    reports how long you ran.  Both look like data in metrics.csv.
    """
    for region_type in sorted(set(labelled["region_type"]) - {"unknown"}):
        sweep = retention_granularity_sweep(labelled, region_type=region_type)
        if sweep.empty:
            continue
        best = recommend_granularity(sweep)
        print(f"  Retention granularity sweep [{region_type}]:")
        for _, r in sweep.iterrows():
            mark = "  <- recommended" if r["granule_bytes"] == best else ""
            print(f"    {int(r['granule_bytes']) // 1024:>6} KB  "
                  f"{r['n_regions']:>7} regions  "
                  f"{r['samples_per_region']:>7.2f} samples/region  "
                  f"p50={r['p50_s']:>7.3f}s  p99={r['p99_s']:>7.3f}s  "
                  f"{r['verdict']}{mark}")
        sys.stdout.flush()   # keep the notes below beneath the table they describe
        if best is None:
            print(f"    WARNING: no granule yields a usable retention distribution "
                  f"for '{region_type}'. Report aggregate figures only, or raise "
                  f"sample density with a smaller --sample-period.", file=sys.stderr)
        elif best != granularity_used:
            print(f"    NOTE: labelling used {granularity_used // 1024} KB "
                  f"regions but {best // 1024} KB is the smallest usable granule "
                  f"for '{region_type}' — retention in metrics.csv is an artifact "
                  f"at the granule actually used.", file=sys.stderr)
        print()


def _df_val(df, col, agg="mean"):
    if df is None or df.empty or col not in df.columns:
        return "n/a"
    vals = df[col].dropna()
    return round(float(vals.agg(agg)), 4) if len(vals) > 0 else "n/a"


# ── Headless mode (spawned by perf mem record) ────────────────────────────────

def run_headless(args) -> None:
    """
    Load the model and generate all prompts, then write timing + phase results
    to --results-file. Runs inside the process that perf mem record spawns, so
    every memory access from model load onwards is captured by the hardware PMU.
    """
    if not args.results_file:
        print("ERROR: --results-file required in --headless mode", file=sys.stderr)
        sys.exit(1)

    model_path = pathlib.Path(args.model)
    model_id = resolve_model_id(args, model_path)
    model_cfg = build_model_cfg(model_path, model_id)
    run_id = generate_run_id(
        model_id=model_cfg.id, context_length=args.ctx,
        batch_size=args.batch_size, output_tokens=args.tokens, rep_index=args.rep,
    )
    run_cfg = RunConfig(
        run_id=run_id, model=model_cfg, context_length=args.ctx,
        batch_size=args.batch_size, output_tokens=args.tokens, rep_index=args.rep,
    )

    pre_load_maps = snapshot_proc_maps()
    engine = LlamaCppEngine(model_cfg, run_cfg)
    engine.prepare()
    engine.load_model()

    # KV cache size implied by the GGUF geometry.  Only a mapping matching this
    # size is claimed as kv_cache — "large and new" also describes the ggml
    # backend arenas, which are an order of magnitude bigger.
    kv_bytes = None
    gguf_metadata = getattr(getattr(engine, "_llm", None), "metadata", None)
    if gguf_metadata:
        kv_bytes = expected_kv_bytes(gguf_metadata, args.ctx)
    address_regions = extract_llamacpp_regions(
        str(model_path), pre_load_maps, expected_kv=kv_bytes,
    )
    print("Model loaded.")
    if kv_bytes:
        print(f"  Expected KV cache from model geometry: {kv_bytes / 1e6:.1f} MB")
    else:
        print("  WARNING: could not derive KV cache size from GGUF metadata — "
              "no region will be classified as kv_cache.", file=sys.stderr)
    print()

    prompts = PROMPTS[:args.num_prompts] if args.num_prompts else PROMPTS
    gen_results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"  [{i}/{len(prompts)}] {prompt[:60]}...")
        print("          generating", end="", flush=True)
        result = engine.generate(prompt)
        gen_results.append({
            "prompt": prompt,
            "output_text": result.text,
            "output_tokens": result.output_tokens,
            "prompt_tokens": result.prompt_tokens,
            "duration_s": result.duration_s,
            "tokens_per_second": result.tokens_per_second,
        })
        print(f" → {result.output_tokens} tokens @ {result.tokens_per_second:.1f} tok/s")

        # llama.cpp creates its threadpool (and the worker thread stacks) on the
        # first generate, after the post-load snapshot above.  Without this second
        # snapshot those stacks stay unregistered — and they carry ~86% of perf mem
        # store samples, so they would all fall through to region_type=unknown.
        if i == 1:
            address_regions += extract_llamacpp_regions(
                str(model_path), pre_load_maps, expected_kv=kv_bytes,
            )

    # Capture peak RSS before shutdown while model weights are still in RAM.
    # Use getrusage (historical maximum) so the value is correct regardless of
    # call ordering — psutil().rss would give post-shutdown ~139 MB (weights freed).
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # kB → MB

    engine.shutdown()

    phase_records = [
        {"event": rec.event.value, "wall_ns": rec.wall_ns, "mono_ns": rec.mono_ns}
        for rec in engine.get_phase_records()
    ]
    runtime_metadata = dataclasses.asdict(engine.get_runtime_metadata())

    pathlib.Path(args.results_file).write_text(json.dumps({
        "phase_records": phase_records,
        "gen_results": gen_results,
        "runtime_metadata": runtime_metadata,
        "address_regions": address_regions,
        "target_peak_rss_mb": round(peak_rss_mb, 1),
    }))
    print()
    print("Headless run complete — perf mem will flush on exit.")


# ── Preflight + spawn (outer process) ──────────────────────────────────────────

def _preflight(args) -> "tuple[pathlib.Path, list | None]":
    """Check root, model file, and PMU. Returns (model_path, events_or_None)."""
    if os.getuid() != 0:
        print("ERROR: perf mem requires root.", file=sys.stderr)
        print(f"  sudo -E venv/bin/python notebooks/scripts/demo_perf_mem.py --model {args.model}",
              file=sys.stderr)
        print("  (-E preserves the venv PATH so the correct mrm-trace is used)", file=sys.stderr)
        sys.exit(1)

    model_path = pathlib.Path(args.model)
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        print("  mkdir -p models && wget -P models/ https://huggingface.co/TheBloke/"
              "TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
              file=sys.stderr)
        sys.exit(1)

    print("Checking PMU availability ...")
    pmu_ok, pmu_msg = check_pmu()
    if pmu_ok:
        print("  PMU: OK")
        return model_path, None  # let perf mem auto-select events
    print(f"  WARNING: {pmu_msg}", file=sys.stderr)
    print("  Falling back to software events (no latency / cache-level data).", file=sys.stderr)
    print("  TIP: run on bare-metal Linux for full hardware PMU support.", file=sys.stderr)
    return model_path, ["mem:0:rw:p"]


_TEMP_POLL_INTERVAL_S = 2.0


class _PeakTempPoller:
    """Background thread that tracks the highest CPU package temp seen so far."""

    def __init__(self, interval_s: float = _TEMP_POLL_INTERVAL_S):
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._peak_c = read_package_temp_c()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._peak_c = max(self._peak_c, read_package_temp_c())
            self._stop.wait(self._interval_s)

    def start(self) -> "_PeakTempPoller":
        self._thread.start()
        return self

    def stop(self) -> float:
        self._stop.set()
        self._thread.join(timeout=self._interval_s * 2)
        self._peak_c = max(self._peak_c, read_package_temp_c())
        return self._peak_c


def _collect_via_spawn(args, model_path, model_id, events, out_dir, raw_dir, run_id):
    """
    Spawn this script under perf mem record so the entire process lifetime
    (model load + generation) is instrumented from the start. Returns
    (runtime_metadata, gen_results, phase_boundaries, collector_meta, thermal_meta).
    """
    perf_bin = shutil.which("perf")
    if not perf_bin:
        print("ERROR: perf binary not found.", file=sys.stderr)
        sys.exit(1)

    results_file = out_dir / "generation_results.json"
    perf_data = raw_dir / "perf.data"

    inner_cmd = [
        sys.executable, str(pathlib.Path(__file__).resolve()),
        "--headless",
        "--model", str(model_path),
        "--model-id", model_id,
        "--ctx", str(args.ctx),
        "--tokens", str(args.tokens),
        "--batch-size", str(args.batch_size),
        "--rep", str(args.rep),
        "--results-file", str(results_file),
    ]
    if args.num_prompts:
        inner_cmd += ["--num-prompts", str(args.num_prompts)]
    perf_cmd = [perf_bin, "mem", "record", "-c", str(args.sample_period), "-o", str(perf_data)]
    # Pin sample timestamps to CLOCK_MONOTONIC so they share a clock domain with
    # the engine's phase records (PhaseRecord.mono_ns).  Without this the phase
    # column is NULL for every row.
    perf_clock_args = clock_args(perf_bin)
    perf_cmd += perf_clock_args
    if not perf_clock_args:
        print("  WARNING: perf does not support -k/--clockid; phase alignment is "
              "relying on perf's default clock being monotonic.", file=sys.stderr)
    for ev in (events or []):
        perf_cmd += ["-e", ev]
    perf_cmd += ["--"] + inner_cmd

    current_pid = os.getpid()
    print(f"Starting perf mem collector (spawn mode, launcher PID {current_pid}) ...")

    machine_ctx = collect_machine_context(str(out_dir))
    cpu_freq_limits = read_cpu_freq_limits()
    baseline = capture_baseline(duration_s=2.0)
    cpu_freq_mon = CpuFreqMonitor()
    gpu_mon = GpuMonitor()
    start_temp_c = read_package_temp_c()
    thermal_monitor = ThermalMonitor()
    thermal_monitor.begin()
    poller = _PeakTempPoller().start()
    collection_start = time.time()
    try:
        proc = subprocess.Popen(perf_cmd)
        observer_mon = ProcessMonitor([proc.pid], poll_interval_ms=500)
        cpu_freq_mon.start()
        gpu_mon.start()
        observer_mon.start()
        proc.wait()
    finally:
        observer_samples = observer_mon.stop()
        cpu_freq_stats = cpu_freq_mon.stop()
        gpu_stats = gpu_mon.stop()
        peak_temp_c = poller.stop()
        collection_duration_s = round(time.time() - collection_start, 3)
        throttle_events = thermal_monitor.end()

    end_temp_c = read_package_temp_c()
    thermal_meta = {
        "start_temp_c": round(start_temp_c, 1),
        "peak_temp_c": round(peak_temp_c, 1),
        "end_temp_c": round(end_temp_c, 1),
        "throttle_events": throttle_events,
        "throttled": throttle_events > 0,
        "collection_wall_duration_s": collection_duration_s,
    }

    warnings: list = []
    if proc.returncode != 0:
        warnings.append(f"perf mem record exited {proc.returncode}")

    if not results_file.exists():
        print("ERROR: generation_results.json not written — spawned process failed.", file=sys.stderr)
        for w in warnings:
            print(f"  WARN: {w}", file=sys.stderr)
        sys.exit(1)

    timing = json.loads(results_file.read_text())
    gen_results = timing["gen_results"]
    phase_boundaries = build_phase_boundaries(timing["phase_records"])
    runtime_metadata = timing["runtime_metadata"]
    address_regions = timing.get("address_regions", [])

    print()
    print("Running perf script (flushing perf.data → text) ...")
    runner = PerfMemRunner(events=list(events or []), sample_period=args.sample_period)
    parse_success = False
    output_size = 0
    output_file = None
    if perf_data.exists():
        output_size = perf_data.stat().st_size
        output_file = str(perf_data)
        parse_success = runner._run_perf_script(perf_data, warnings)
    else:
        warnings.append("perf.data not produced — PMU unavailable or permission denied")

    for w in warnings:
        print(f"  WARN: {w}", file=sys.stderr)
    if not parse_success:
        print("ERROR: perf script failed — see warnings above.", file=sys.stderr)
        print(f"  Raw data preserved at: {perf_data}", file=sys.stderr)
        print(f"  Debug: sudo perf script -i {perf_data} | head -20", file=sys.stderr)
        sys.exit(1)

    collector_meta = CollectorMeta(
        collector_type="perf_mem",
        available=True,
        used=True,
        event_strings=list(events or []),
        sample_period=args.sample_period,
        output_file=output_file,
        output_file_size_bytes=output_size,
        parse_success=parse_success,
        warnings=warnings,
    )
    print(f"  Capture: {collector_meta.output_file_size_bytes:,} bytes (perf.data)")
    print()

    target_peak_rss_mb = timing.get("target_peak_rss_mb", 0.0)
    observer_effect = compute_observer_effect(
        observer_samples=observer_samples,
        target_samples=[],
        thermal_throttle_events=throttle_events,
        baseline_cpu_pct=baseline.cpu_pct,
    )
    # Patch in target RSS from headless results — process_monitor can't reach the
    # child process across the perf boundary, so we read it from the results file.
    if target_peak_rss_mb > 0:
        obs_peak = observer_effect.get("observer_memory_overhead_mb_peak", 0.0)
        observer_effect["target_rss_mb_peak"] = target_peak_rss_mb
        observer_effect["observer_memory_overhead_frac"] = (
            obs_peak / target_peak_rss_mb if target_peak_rss_mb > 0 else 0.0
        )
    # perf mem writes perf.data to disk (always > 10 MB) — this is expected and
    # does not represent I/O contention with the model process.
    observer_effect["io_contention_detected"] = False

    return (
        runtime_metadata, gen_results, phase_boundaries, collector_meta,
        thermal_meta, address_regions, baseline, cpu_freq_limits, cpu_freq_stats, gpu_stats, observer_effect,
        machine_ctx,
    )


def _parse_and_label(perf_script_txt, run_id, phase_boundaries, address_regions=None,
                     granularity: int = _REGION_GRANULARITY_BYTES,
                     chunk_rows: int = 0,
                     sample_period: int = 0) -> "tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]":
    """
    Parse perf script, label, print symbol summary.

    Both passes stream the text file in chunks and never hold the raw row dicts
    for the whole run (ADR-2); only the labelled frame is retained.  The file is
    read twice because symbol evidence is per row while classification is per
    region granule: without a prescan the first sample on a granule would be
    labelled from address ranges alone and a later sample carrying a decisive
    symbol would be labelled differently, making the output depend on row order.

    Returns (labelled, region_map, kv_lifecycle).
    """
    chunk_rows = chunk_rows or _auto_chunk_rows()

    # Both passes re-open the source.  A file-like object would be exhausted by
    # pass 1, leaving pass 2 to label nothing and emit an empty frame.
    if hasattr(perf_script_txt, "read"):
        raise TypeError(
            "_parse_and_label needs a file path, not a file object: the trace is "
            "streamed twice and a stream cannot be re-read"
        )

    labeller = TraceLabeller(region_granularity_bytes=granularity)
    if address_regions:
        n_pages = labeller.register_address_ranges(address_regions)
        print(f"  Pre-registered {n_pages:,} pages from /proc/maps:")
        for source, row in sorted(
            summarise_regions(address_regions).items(),
            key=lambda kv: -kv[1]["bytes"],
        ):
            print(f"    {row['region_type']:<9} {source:<16} "
                  f"{row['count']:>4} mappings  {row['bytes'] / 1e6:>9.1f} MB")

    def _stream():
        return _chunks(
            parse_perf_script(perf_script_txt, run_id=run_id,
                              phase_boundaries=phase_boundaries),
            chunk_rows,
        )

    # ── Pass 1: tally coverage and collect symbol evidence ────────────────────
    print(f"Parsing perf script output (pass 1/2, {chunk_rows:,}-row chunks) ...")
    n_rows = n_lat = n_cache = n_sym_granules = 0
    op_counts: collections.Counter = collections.Counter()
    phase_counts: collections.Counter = collections.Counter()
    sym_counts: collections.Counter = collections.Counter()
    # Value sets for the dictionary-encoded columns.  Collecting them here is
    # what lets pass 2 give every chunk the same CategoricalDtype.
    dso_values: set = set()
    cache_values: set = set()
    next_progress = _PROGRESS_EVERY_ROWS

    for chunk in _stream():
        for row in chunk:
            if row["access_latency_ns"] is not None:
                n_lat += 1
            if row["cache_level"] is not None:
                n_cache += 1
            op_counts[row["op_type"]] += 1
            phase_counts[row["phase"]] += 1
            sym_counts[row["symbol"]] += 1
            dso_values.add(row["dso"])
            cache_values.add(row["cache_level"])
        n_rows += len(chunk)
        n_sym_granules += labeller.prescan_symbols(chunk)
        chunk.clear()
        if n_rows >= next_progress:
            print(f"  {n_rows:,} rows ...", end="\r", flush=True)
            next_progress += _PROGRESS_EVERY_ROWS

    if n_rows == 0:
        print("ERROR: no samples parsed — PMU may not have captured any events.", file=sys.stderr)
        print(f"  Check perf_script.txt:  head -20 {perf_script_txt}", file=sys.stderr)
        sys.exit(1)

    print(f"  {n_rows:,} samples")
    print(f"  op_type  : {dict(op_counts)}")
    print(f"  phases   : {dict(phase_counts)}")
    print()
    print("  Top 15 symbols (expect ggml_* and llama_* with native llama.cpp):")
    for sym, cnt in sym_counts.most_common(15):
        print(f"    {sym:<50}  {cnt:>6}")

    check_trace_quality_counts(
        rows=n_rows,
        latency_non_null=n_lat,
        cache_level_non_null=n_cache,
        symbol_non_empty=n_rows - sym_counts[""],
        region_classified=0,   # nothing is labelled yet; not gated pre-labelling
        phase_non_null=n_rows - phase_counts[None],
        collector_type="perf_mem",
        fail_fast=True,
    )
    print(f"  Symbol evidence upgraded {n_sym_granules:,} granules "
          f"(0 means stripped llama.cpp — see workplan §5.1)")
    _check_frame_fits(n_rows, sample_period or 1)

    # ── Pass 2: label and accumulate the frame ────────────────────────────────
    print("Labelling (pass 2/2) ...")
    dtypes = _categorical_dtypes(run_id, {
        "symbol": set(sym_counts), "dso": dso_values, "cache_level": cache_values,
    })
    frames: list = []
    done = 0
    next_progress = _PROGRESS_EVERY_ROWS
    for chunk in _stream():
        labelled_rows = list(labeller.label(chunk))
        chunk.clear()
        frames.append(_frame_from_chunk(labelled_rows, dtypes))
        done += len(labelled_rows)
        labelled_rows.clear()
        if done >= next_progress:
            print(f"  {done:,} rows ...", end="\r", flush=True)
            next_progress += _PROGRESS_EVERY_ROWS

    labelled = pd.concat(frames, ignore_index=True)
    frames.clear()

    # astype() against a CategoricalDtype turns any value the dtype does not
    # list into NaN.  The two passes read the same file with the same code so
    # that cannot happen, but it would be a silent column wipe if it ever did.
    for col in ("run_id", "symbol", "dso"):
        if labelled[col].isna().any():
            raise RuntimeError(
                f"{col} lost values during dictionary encoding — pass 1 and "
                f"pass 2 disagree about the trace contents"
            )
    region_map = pd.DataFrame(labeller.region_map())
    kv_lifecycle = pd.DataFrame(labeller.kv_lifecycle())
    print(f"  Labelled trace : {labelled.shape}")
    print(f"  Region map     : {region_map.shape}")
    print(f"  KV lifecycle   : {kv_lifecycle.shape}")
    if not region_map.empty and "region_type" in region_map.columns:
        # Aggregate — a per-region listing is 48k lines when KV regions are
        # address-granular, which buries the summary it is meant to provide.
        print("  Classified regions (region_map):")
        agg = region_map.groupby("region_type").agg(
            regions=("region_id", "count"), accesses=("access_count", "sum"),
        )
        for rtype, row in agg.iterrows():
            print(f"    {rtype:<15} {int(row['regions']):>7} regions  "
                  f"{int(row['accesses']):>10,} accesses")

    # Samples are ~99.6% stores (perf samples stores at a fixed period, but only
    # samples loads above the PEBS ldlat threshold), and stores land mostly on
    # thread stacks.  A single blended coverage number is therefore dominated by
    # store noise and says little about the load samples the MRM metrics rest on.
    for op in ("load", "store"):
        sub = labelled[labelled["op_type"] == op]
        if len(sub):
            cov = float((sub["region_type"] != "unknown").sum()) / len(sub)
            print(f"  region classified ({op:<5}): {cov:6.1%}  of {len(sub):,} samples")

    check_trace_quality(labelled, collector_type="perf_mem_labelled", fail_fast=True)

    return labelled, region_map, kv_lifecycle


def _print_analysis_summary(analysis: dict) -> None:
    print("Analysing ...")
    rs = analysis.get("retention_summary")
    ws = analysis.get("working_set_summary")
    wo = analysis.get("write_once")
    suit_df = analysis.get("suitability")
    suit_str = (
        dict(zip(suit_df["region_type"], suit_df["suitability"]))
        if suit_df is not None and not suit_df.empty
        else "n/a"
    )
    print(f"  Suitability                : {suit_str}")
    print(f"  Retention p99 (s)          : {_df_val(rs, 'retention_p99_s')}")
    print(f"  Working set (total pages)  : {_df_val(ws, 'total_pages', 'sum')}")
    print(f"  Write-once ratio (mean)    : {_df_val(wo, 'write_once_ratio')}")
    print()


def main():
    args = parse_args()

    # Headless mode: run by perf mem record as the traced workload.
    if args.headless:
        run_headless(args)
        return

    model_path, events = _preflight(args)
    model_id = resolve_model_id(args, model_path)
    mslug = args.machine or machine_slug()

    run_id = generate_run_id(
        model_id=model_id, context_length=args.ctx,
        batch_size=args.batch_size, output_tokens=args.tokens, rep_index=args.rep,
    )
    results_root = pathlib.Path(args.out)
    out_dir = results_root / mslug / model_id / run_id
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print()
    print(f"mrm-trace live demo — {model_id} (perf mem)")
    print("  Collector     : perf mem (hardware PMU, spawn mode)  ← real timestamps, real region map")
    print(f"  Model         : {model_path}")
    print(f"  Context       : {args.ctx} tokens")
    print(f"  Output tokens : {args.tokens} per prompt")
    print(f"  Sample period : {args.sample_period}")
    print(f"  Output dir    : {out_dir}")
    print()
    print("  perf mem provides:")
    print("    + Hardware timestamps (nanoseconds) → real retention durations")
    print("    + Symbol names from libllama.so     → weight / kv_cache / activation regions")
    print("    + Cache level (L1/L2/L3/DRAM)       → memory hierarchy analysis")
    print()

    run_start_time = time.time()
    model_cfg = build_model_cfg(model_path, model_id)
    (
        runtime_metadata, gen_results, phase_boundaries, collector_meta,
        thermal_meta, address_regions, baseline, cpu_freq_limits, cpu_freq_stats, gpu_stats, observer_effect,
        machine_ctx,
    ) = _collect_via_spawn(args, model_path, model_id, events, out_dir, raw_dir, run_id)

    labelled, region_map, kv_lifecycle = _parse_and_label(
        raw_dir / "perf_script.txt", run_id, phase_boundaries, address_regions,
        granularity=args.region_granularity,
        chunk_rows=args.chunk_rows,
        sample_period=args.sample_period,
    )

    analysis = compute_all(labelled)
    _print_analysis_summary(analysis)
    _print_granularity_sweep(labelled, args.region_granularity)

    print("Exporting ...")
    total_wall_duration_s = time.time() - run_start_time
    metadata = build_metadata(
        model_cfg, run_id, runtime_metadata, gen_results, collector_meta,
        thermal_meta=thermal_meta, total_wall_duration_s=total_wall_duration_s,
        baseline=baseline, cpu_freq_limits=cpu_freq_limits, cpu_freq_stats=cpu_freq_stats,
        gpu_stats=gpu_stats, observer_effect=observer_effect, machine_ctx=machine_ctx,
        machine_id=mslug, context_length=args.ctx, batch_size=args.batch_size,
        output_tokens=args.tokens, rep_index=args.rep,
        region_granularity=args.region_granularity,
    )
    exporter = RunExporter(out_dir)
    export_result = exporter.export(
        trace=labelled, region_map=region_map, kv_lifecycle=kv_lifecycle,
        analysis=analysis, metadata=metadata, run_id=run_id,
        manifest_csv=results_root / "manifest.csv",
        prompt_outputs=gen_results,
    )
    print(f"  Output: {out_dir}/")
    for fname in sorted(export_result.file_names):
        print(f"    {fname}")

    _reclaim_perf_script(raw_dir, keep=args.keep_perf_script)
    _print_validity(metadata.get("thermal", {}))

    print()
    print("Done.")


if __name__ == "__main__":
    main()

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
import dataclasses
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import threading
import time

import pandas as pd

from mrm_trace.analyser import compute_all
from mrm_trace.collector.base import CollectorMeta
from mrm_trace.collector.perf_runner import PerfMemRunner
from mrm_trace.collector.process_monitor import ProcessMonitor
from mrm_trace.config.schema import EngineType, ModelConfig, RunConfig
from mrm_trace.engines.base import PhaseEvent
from mrm_trace.engines.llamacpp import LlamaCppEngine
from mrm_trace.labeller import TraceLabeller
from mrm_trace.labeller.llamacpp_regions import extract_llamacpp_regions, snapshot_proc_maps
from mrm_trace.parser.perf_script_parser import parse_perf_script
from mrm_trace.reporter.run_exporter import RunExporter
from mrm_trace.reporter.trace_qa import check_trace_quality
from mrm_trace.telemetry.baseline import capture_baseline
from mrm_trace.telemetry.observer_effect import compute_observer_effect
from mrm_trace.telemetry.system_monitor import CpuFreqMonitor, CpuFreqLimits, GpuMonitor, read_cpu_freq_limits
from mrm_trace.telemetry.thermal import ThermalMonitor, read_package_temp_c
from mrm_trace.telemetry.validity import classify_run
from mrm_trace.utils.ids import generate_run_id
from mrm_trace.validate_machine import collect_machine_context

DEFAULT_MODEL = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

PROMPTS = [
    "Explain memory management in operating systems. Focus on virtual memory, paging, and TLB.",
    "Write a Python function that implements a least-recently-used (LRU) cache using a dict and a deque.",
    "What are the key differences between supervised, unsupervised, and reinforcement learning? Give one example of each.",
    "Describe the transformer architecture. Explain how multi-head self-attention differs from single-head attention.",
    "How does garbage collection work in the JVM? Compare generational GC to reference counting in CPython.",
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
        "--sample-period", type=int, default=4000,
        help="perf sampling period — lower = denser trace, higher overhead (default 4000)",
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
    Convert serialized phase records ({"event": ..., "wall_ns": ...} dicts) to
    parse_perf_script phase_boundaries.

    With perf mem, trace timestamps are hardware nanosecond wall clocks, so
    phase annotation works correctly — each sample gets its phase label.
    """
    boundaries = []
    starts = {}
    phase_map = {
        PhaseEvent.MODEL_LOAD_START.value: (PhaseEvent.MODEL_LOAD_END.value, "weight_load"),
        PhaseEvent.GENERATION_START.value: (PhaseEvent.GENERATION_END.value, "generation"),
    }
    for rec in phase_records:
        event = rec["event"]
        wall_ns = rec["wall_ns"]
        if event in phase_map:
            starts[event] = wall_ns
        for start_ev, (end_ev, label) in phase_map.items():
            if event == end_ev and start_ev in starts:
                boundaries.append({
                    "phase": label,
                    "start_ns": starts.pop(start_ev),
                    "end_ns": wall_ns,
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
    oe = observer_effect or {}
    validity = classify_run(
        observer_cpu_overhead_pct_max=oe.get("observer_cpu_overhead_pct_max", 0.0),
        observer_memory_frac=oe.get("observer_memory_overhead_frac", 0.0),
        thermal_throttle_events=thermal_meta["throttle_events"],
        baseline_cpu_pct=baseline.cpu_pct if baseline else 0.0,
        io_contention_detected=oe.get("io_contention_detected", False),
        peak_temp_c=thermal_meta.get("peak_temp_c", 0.0),
    )
    return {**thermal_meta, "run_validity": validity.run_validity}


def build_metadata(
    model_cfg, run_id, runtime_metadata, gen_results, collector_meta,
    thermal_meta=None, total_wall_duration_s=None,
    baseline=None, cpu_freq_limits=None, cpu_freq_stats=None, gpu_stats=None, observer_effect=None,
    machine_ctx=None,
):
    total_tokens = sum(r["output_tokens"] for r in gen_results)
    total_time = sum(r["duration_s"] for r in gen_results)
    metadata = {
        "run_id": run_id,
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
            "total_output_tokens": total_tokens,
            "total_duration_s": round(total_time, 3),
            "avg_tokens_per_second": round(total_tokens / total_time, 1) if total_time > 0 else 0.0,
        },
        "collector": {
            "type": "perf_mem",
            "mode": "spawn",
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
    address_regions = extract_llamacpp_regions(str(model_path), pre_load_maps)
    print("Model loaded.")
    print()

    gen_results = []
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"  [{i}/{len(PROMPTS)}] {prompt[:60]}...")
        print("          generating", end="", flush=True)
        result = engine.generate(prompt)
        gen_results.append({
            "output_tokens": result.output_tokens,
            "prompt_tokens": result.prompt_tokens,
            "duration_s": result.duration_s,
            "tokens_per_second": result.tokens_per_second,
        })
        print(f" → {result.output_tokens} tokens @ {result.tokens_per_second:.1f} tok/s")

    engine.shutdown()

    phase_records = [
        {"event": rec.event.value, "wall_ns": rec.wall_ns}
        for rec in engine.get_phase_records()
    ]
    runtime_metadata = dataclasses.asdict(engine.get_runtime_metadata())

    import psutil
    peak_rss_mb = psutil.Process().memory_info().rss / 1_048_576

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
    perf_cmd = [perf_bin, "mem", "record", "-o", str(perf_data)]
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

    return (
        runtime_metadata, gen_results, phase_boundaries, collector_meta,
        thermal_meta, address_regions, baseline, cpu_freq_limits, cpu_freq_stats, gpu_stats, observer_effect,
        machine_ctx,
    )


def _parse_and_label(perf_script_txt, run_id, phase_boundaries, address_regions=None) -> "tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]":
    """Parse perf script, label, print symbol summary. Returns (labelled, region_map, kv_lifecycle)."""
    print("Parsing perf script output ...")
    rows = list(parse_perf_script(perf_script_txt, run_id=run_id, phase_boundaries=phase_boundaries))
    if not rows:
        print("ERROR: no samples parsed — PMU may not have captured any events.", file=sys.stderr)
        print(f"  Check perf_script.txt:  head -20 {perf_script_txt}", file=sys.stderr)
        sys.exit(1)

    trace = pd.DataFrame(rows)
    print(f"  {len(trace):,} samples")
    print(f"  op_type  : {dict(trace['op_type'].value_counts())}")
    print(f"  phases   : {dict(trace['phase'].value_counts(dropna=False))}")
    print()
    print("  Top 15 symbols (expect ggml_* and llama_* with native llama.cpp):")
    for sym, cnt in trace["symbol"].value_counts().head(15).items():
        print(f"    {sym:<50}  {cnt:>6}")

    check_trace_quality(trace, collector_type="perf_mem", fail_fast=True)

    print("Labelling ...")
    labeller = TraceLabeller()
    if address_regions:
        n_pages = labeller.register_address_ranges(address_regions)
        weight_regions = sum(1 for r in address_regions if r.get("region_type") == "weight")
        kv_regions = sum(1 for r in address_regions if r.get("region_type") == "kv_cache")
        print(f"  Pre-registered {n_pages:,} pages from /proc/maps "
              f"({weight_regions} weight mmap, {kv_regions} kv_cache anon)")
    labelled = pd.DataFrame(list(labeller.label(trace.to_dict(orient="records"))))
    region_map = pd.DataFrame(labeller.region_map())
    kv_lifecycle = pd.DataFrame(labeller.kv_lifecycle())
    print(f"  Labelled trace : {labelled.shape}")
    print(f"  Region map     : {region_map.shape}")
    print(f"  KV lifecycle   : {kv_lifecycle.shape}")
    if not region_map.empty and "region_type" in region_map.columns:
        print("  Classified regions:")
        for _, row in region_map.iterrows():
            print(f"    {row.get('region_type', '?'):<15} id={row.get('region_id')}  accesses={row.get('access_count', '?')}")

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

    run_id = generate_run_id(
        model_id=model_id, context_length=args.ctx,
        batch_size=args.batch_size, output_tokens=args.tokens, rep_index=args.rep,
    )
    out_dir = pathlib.Path(args.out) / model_id / run_id
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
        raw_dir / "perf_script.txt", run_id, phase_boundaries, address_regions
    )

    analysis = compute_all(labelled)
    _print_analysis_summary(analysis)

    print("Exporting ...")
    total_wall_duration_s = time.time() - run_start_time
    metadata = build_metadata(
        model_cfg, run_id, runtime_metadata, gen_results, collector_meta,
        thermal_meta=thermal_meta, total_wall_duration_s=total_wall_duration_s,
        baseline=baseline, cpu_freq_limits=cpu_freq_limits, cpu_freq_stats=cpu_freq_stats,
        gpu_stats=gpu_stats, observer_effect=observer_effect, machine_ctx=machine_ctx,
    )
    exporter = RunExporter(out_dir)
    export_result = exporter.export(
        trace=labelled, region_map=region_map, kv_lifecycle=kv_lifecycle,
        analysis=analysis, metadata=metadata, run_id=run_id,
    )
    print(f"  Output: {out_dir}/")
    for fname in sorted(export_result.file_names):
        print(f"    {fname}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()

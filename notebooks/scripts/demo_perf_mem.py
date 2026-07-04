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
import shutil
import subprocess
import sys
import time

import pandas as pd

from mrm_trace.analyser import compute_all
from mrm_trace.collector.base import CollectorMeta
from mrm_trace.collector.perf_runner import PerfMemRunner
from mrm_trace.config.schema import EngineType, ModelConfig, RunConfig
from mrm_trace.engines.base import PhaseEvent
from mrm_trace.engines.llamacpp import LlamaCppEngine
from mrm_trace.labeller import TraceLabeller
from mrm_trace.parser.perf_script_parser import parse_perf_script
from mrm_trace.reporter.run_exporter import RunExporter
from mrm_trace.utils.ids import generate_run_id

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


def build_model_cfg(model_path) -> ModelConfig:
    return ModelConfig(
        id=MODEL_ID,
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


def build_metadata(model_cfg, run_id, runtime_metadata, gen_results, collector_meta):
    total_tokens = sum(r["output_tokens"] for r in gen_results)
    total_time = sum(r["duration_s"] for r in gen_results)
    return {
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
    model_cfg = build_model_cfg(model_path)
    run_id = generate_run_id(
        model_id=model_cfg.id, context_length=args.ctx,
        batch_size=args.batch_size, output_tokens=args.tokens, rep_index=args.rep,
    )
    run_cfg = RunConfig(
        run_id=run_id, model=model_cfg, context_length=args.ctx,
        batch_size=args.batch_size, output_tokens=args.tokens, rep_index=args.rep,
    )

    engine = LlamaCppEngine(model_cfg, run_cfg)
    engine.prepare()
    engine.load_model()
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

    pathlib.Path(args.results_file).write_text(json.dumps({
        "phase_records": phase_records,
        "gen_results": gen_results,
        "runtime_metadata": runtime_metadata,
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


def _collect_via_spawn(args, model_path, events, out_dir, raw_dir, run_id):
    """
    Spawn this script under perf mem record so the entire process lifetime
    (model load + generation) is instrumented from the start. Returns
    (runtime_metadata, gen_results, phase_boundaries, collector_meta).
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
    result = subprocess.run(perf_cmd)

    warnings: list = []
    if result.returncode != 0:
        warnings.append(f"perf mem record exited {result.returncode}")

    if not results_file.exists():
        print("ERROR: generation_results.json not written — spawned process failed.", file=sys.stderr)
        for w in warnings:
            print(f"  WARN: {w}", file=sys.stderr)
        sys.exit(1)

    timing = json.loads(results_file.read_text())
    gen_results = timing["gen_results"]
    phase_boundaries = build_phase_boundaries(timing["phase_records"])
    runtime_metadata = timing["runtime_metadata"]

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
    return runtime_metadata, gen_results, phase_boundaries, collector_meta


def _parse_and_label(perf_script_txt, run_id, phase_boundaries) -> "tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]":
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
    print()

    print("Labelling ...")
    labeller = TraceLabeller()
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
    print()
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

    run_id = generate_run_id(
        model_id=MODEL_ID, context_length=args.ctx,
        batch_size=args.batch_size, output_tokens=args.tokens, rep_index=args.rep,
    )
    out_dir = pathlib.Path(args.out) / MODEL_ID / run_id
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("mrm-trace live demo — TinyLlama 1.1B Q4_K_M (perf mem)")
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

    model_cfg = build_model_cfg(model_path)
    runtime_metadata, gen_results, phase_boundaries, collector_meta = _collect_via_spawn(
        args, model_path, events, out_dir, raw_dir, run_id
    )

    labelled, region_map, kv_lifecycle = _parse_and_label(
        raw_dir / "perf_script.txt", run_id, phase_boundaries
    )

    analysis = compute_all(labelled)
    _print_analysis_summary(analysis)

    print("Exporting ...")
    metadata = build_metadata(model_cfg, run_id, runtime_metadata, gen_results, collector_meta)
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

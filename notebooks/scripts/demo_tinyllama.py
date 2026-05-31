#!/usr/bin/env python3
"""
TinyLlama 1.1B Q4_K_M — live mrm-trace collection.

Collector is selected automatically:
  - Bare-metal Linux + root + PMU available → perf mem (spawn mode)
    → real hardware timestamps, cache levels, populated region_map
      (if llama-cpp-python was built with debug symbols)
  - WSL2 / no root / no PMU              → memray (fallback)

Usage:
    python demo_tinyllama.py --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

    # Force collector:
    python demo_tinyllama.py --collector memray   --model ...
    sudo -E venv/bin/python demo_tinyllama.py --collector perf-mem --model ...

    # Bare-metal Linux, auto → perf mem:
    sudo -E venv/bin/python demo_tinyllama.py --model ...

Model download (one-time):
    mkdir -p models
    wget -P models/ https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

region_map note:
    perf mem + pip-installed llama-cpp-python → region_map empty (pip strips C symbols).
    perf mem + debug build → region_map populated with ggml_* regions.
    Build debug: CMAKE_BUILD_TYPE=Debug pip install llama-cpp-python
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
from mrm_trace.config.schema import EngineType, ModelConfig, RunConfig
from mrm_trace.engines.base import PhaseEvent
from mrm_trace.engines.llamacpp import LlamaCppEngine
from mrm_trace.labeller import TraceLabeller, snapshot_proc_maps, extract_llamacpp_regions
from mrm_trace.reporter.run_exporter import RunExporter
from mrm_trace.utils.ids import generate_run_id

DEFAULT_MODEL = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

_RAW_PROMPTS = [
    "Explain memory management in operating systems. Focus on virtual memory, paging, and TLB.",
    "Write a Python function that implements a least-recently-used (LRU) cache using a dict and a deque.",
    "What are the key differences between supervised, unsupervised, and reinforcement learning? Give one example of each.",
    "Describe the transformer architecture. Explain how multi-head self-attention differs from single-head attention.",
    "How does garbage collection work in the JVM? Compare generational GC to reference counting in CPython.",
]

def _fmt(prompt: str) -> str:
    return (
        "<|system|>\nYou are a helpful AI assistant.</s>\n"
        f"<|user|>\n{prompt}</s>\n<|assistant|>\n"
    )

PROMPTS = [_fmt(p) for p in _RAW_PROMPTS]


# ── Environment detection ─────────────────────────────────────────────────────

def _is_wsl2() -> bool:
    try:
        return "microsoft" in pathlib.Path("/proc/sys/kernel/osrelease").read_text().lower()
    except OSError:
        return False


def _check_pmu() -> "tuple[bool, str]":
    perf = shutil.which("perf")
    if not perf:
        return False, "perf binary not found (install linux-perf)"
    import tempfile
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [perf, "mem", "record", "-o", f"{tmpdir}/perf.data", "--", "true"],
                capture_output=True, timeout=15,
            )
    except Exception as e:
        return False, str(e)
    if result.returncode != 0:
        return False, result.stderr.decode(errors="replace")[:300]
    return True, "ok"


def _auto_select_collector(is_root: bool) -> str:
    if _is_wsl2():
        return "memray"
    if not is_root:
        return "memray"
    pmu_ok, _ = _check_pmu()
    return "perf-mem" if pmu_ok else "memray"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="TinyLlama mrm-trace demo")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Path to GGUF model file")
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--out", default="results")
    p.add_argument("--collector", choices=["auto", "memray", "perf-mem"], default="auto")
    p.add_argument(
        "--native-traces", dest="native_traces", action="store_true", default=None,
        help="[memray] Force native_traces=True (C-level allocs; needs root)",
    )
    p.add_argument(
        "--no-native-traces", dest="native_traces", action="store_false",
    )
    p.add_argument(
        "--sample-every-n", dest="sample_every_n", type=int, default=25,
        help="[memray] Parse every Nth allocation record (default 25)",
    )
    p.add_argument("--sample-period", dest="sample_period", type=int, default=4000,
                   help="[perf-mem] Sampling period (default 4000)")
    # Internal flags used when perf spawns this script as the traced workload
    p.add_argument("--headless", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--results-file", dest="results_file", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


# ── Engine helpers ────────────────────────────────────────────────────────────

def _make_model_cfg(model_path: pathlib.Path) -> ModelConfig:
    return ModelConfig(
        id="tinyllama-1.1b-q4km",
        engine=EngineType.LLAMACPP,
        path=str(model_path),
        family="llama",
        params_billions=1.1,
        quantisation="Q4_K_M",
        dtype="float16",
    )


def _make_run_cfg(model_cfg: ModelConfig, run_id: str, args) -> RunConfig:
    return RunConfig(
        run_id=run_id,
        model=model_cfg,
        context_length=args.ctx,
        batch_size=1,
        output_tokens=args.tokens,
        rep_index=0,
    )


def _run_generation(engine: LlamaCppEngine) -> "tuple[list, int, int]":
    gen_results = []
    gen_start_ns = time.time_ns()
    for i, (prompt, raw_prompt) in enumerate(zip(PROMPTS, _RAW_PROMPTS), 1):
        print(f"  [{i}/{len(PROMPTS)}] {raw_prompt[:60]}...")
        print("          generating", end="", flush=True)
        result = engine.generate(prompt)
        gen_results.append(result)
        print(f" → {result.output_tokens} tokens @ {result.tokens_per_second:.1f} tok/s")
    return gen_results, gen_start_ns, time.time_ns()


def _phase_boundaries_from_records(phase_records) -> list:
    boundaries = []
    starts = {}
    phase_map = {
        PhaseEvent.MODEL_LOAD_START: (PhaseEvent.MODEL_LOAD_END, "weight_load"),
        PhaseEvent.GENERATION_START: (PhaseEvent.GENERATION_END, "generation"),
    }
    for rec in phase_records:
        if rec.event in phase_map:
            starts[rec.event] = rec.wall_ns
        for start_ev, (end_ev, label) in phase_map.items():
            if rec.event == end_ev and start_ev in starts:
                boundaries.append({
                    "phase": label,
                    "start_ns": starts.pop(start_ev),
                    "end_ns": rec.wall_ns,
                })
    return boundaries


def _phase_boundaries_from_dicts(records: list) -> list:
    """Reconstruct phase_boundaries from the serialised phase_records in results JSON."""
    phase_map = {
        "model_load_start": ("model_load_end", "weight_load"),
        "generation_start": ("generation_end", "generation"),
    }
    starts: dict = {}
    boundaries = []
    for rec in records:
        ev = rec["event"]
        if ev in phase_map:
            starts[ev] = rec["wall_ns"]
        for start_ev, (end_ev, label) in phase_map.items():
            if ev == end_ev and start_ev in starts:
                boundaries.append({
                    "phase": label,
                    "start_ns": starts.pop(start_ev),
                    "end_ns": rec["wall_ns"],
                })
    return boundaries


# ── Headless mode (spawned by perf mem record) ────────────────────────────────

def run_headless(args) -> None:
    """
    Load model, run generation, write timing + phase records to --results-file.
    Runs inside the process that perf mem record spawns — the full model lifetime
    is captured by the hardware PMU from the first instruction.
    """
    if not args.results_file:
        print("ERROR: --results-file required in --headless mode", file=sys.stderr)
        sys.exit(1)

    model_path = pathlib.Path(args.model)
    model_cfg = _make_model_cfg(model_path)
    run_id = "headless"  # placeholder; outer process has the real run_id
    run_cfg = _make_run_cfg(model_cfg, run_id, args)

    # Snapshot memory maps BEFORE Llama() so we can detect new allocations
    pre_load_maps = snapshot_proc_maps()

    engine = LlamaCppEngine(model_cfg, run_cfg)
    engine.prepare()
    engine.load_model()
    print("Model loaded.")

    # Extract weight (GGUF mmap) + KV cache (new anonymous regions) after load
    llamacpp_regions = extract_llamacpp_regions(str(model_path), pre_load_maps)
    n_weight = sum(1 for r in llamacpp_regions if r["region_type"] == "weight")
    n_kv = sum(1 for r in llamacpp_regions if r["region_type"] == "kv_cache")
    print(f"  Extracted {n_weight} weight + {n_kv} KV cache memory regions")
    print()

    gen_results, gen_start_ns, gen_end_ns = _run_generation(engine)
    engine.shutdown()

    # Serialize phase records so the outer process can reconstruct phase boundaries
    phase_records_raw = [
        {"event": rec.event.value, "wall_ns": rec.wall_ns}
        for rec in engine.get_phase_records()
    ]

    results_path = pathlib.Path(args.results_file)
    results = {
        "gen_start_ns": gen_start_ns,
        "gen_end_ns": gen_end_ns,
        "gen_results": [dataclasses.asdict(r) for r in gen_results],
        "phase_records": phase_records_raw,
    }
    results_path.write_text(json.dumps(results))

    # Write regions alongside generation results for the outer process to pre-register
    regions_file = results_path.parent / "llamacpp_regions.json"
    regions_file.write_text(json.dumps(llamacpp_regions))

    print()
    print("Headless run complete — perf mem will flush on exit.")


# ── Collection paths ──────────────────────────────────────────────────────────

def collect_with_perf_mem(args, raw_dir: pathlib.Path, run_id: str):
    """
    Spawn this script under perf mem record so the full process lifetime
    (model load + generation) is instrumented from the start.
    """
    from mrm_trace.parser.perf_script_parser import parse_perf_script
    from mrm_trace.collector.perf_runner import PerfMemRunner

    perf = shutil.which("perf")
    results_file = raw_dir.parent / "generation_results.json"
    perf_data = raw_dir / "perf.data"

    inner_cmd = [
        sys.executable, str(pathlib.Path(__file__).resolve()),
        "--headless",
        "--model", args.model,
        "--ctx", str(args.ctx),
        "--tokens", str(args.tokens),
        "--results-file", str(results_file),
    ]
    perf_cmd = [perf, "mem", "record", "-o", str(perf_data), "--"] + inner_cmd

    print("Launching perf mem record (spawn mode) ...")
    print("  perf will instrument the full process: model load → generation")
    print()

    result = subprocess.run(perf_cmd)
    if result.returncode != 0:
        print(f"ERROR: perf mem record exited {result.returncode}", file=sys.stderr)
        sys.exit(1)

    if not results_file.exists():
        print("ERROR: generation_results.json not written — inner process failed.", file=sys.stderr)
        sys.exit(1)

    timing = json.loads(results_file.read_text())
    gen_results_raw = timing["gen_results"]
    phase_boundaries = _phase_boundaries_from_dicts(timing["phase_records"])

    print("Running perf script ...")
    runner = PerfMemRunner()
    runner._output_dir = raw_dir
    warnings: list = []
    ok = runner._run_perf_script(perf_data, warnings)
    for w in warnings:
        print(f"  WARN: {w}", file=sys.stderr)
    if not ok:
        print("ERROR: perf script failed.", file=sys.stderr)
        sys.exit(1)

    collector_info = {
        "type": "perf_mem",
        "mode": "spawn",
        "output_file": str(perf_data),
        "output_file_size_bytes": perf_data.stat().st_size,
        "parse_success": True,
    }

    perf_script_txt = raw_dir / "perf_script.txt"
    print("Parsing + labelling perf script output ...")
    try:
        with open(perf_script_txt) as _fh:
            print(f"  format sample: {next(_fh).rstrip()}")
    except (StopIteration, OSError):
        pass

    rows = list(parse_perf_script(perf_script_txt, run_id=run_id, phase_boundaries=phase_boundaries))
    if not rows:
        print("WARNING: no samples parsed — check format above.", file=sys.stderr)

    labeller = TraceLabeller()

    # Pre-register weight mmap + KV cache regions extracted by the headless process
    regions_file = raw_dir.parent / "llamacpp_regions.json"
    if regions_file.exists():
        regions_data = json.loads(regions_file.read_text())
        n_pages = labeller.register_address_ranges(regions_data)
        weight_r = [r for r in regions_data if r["region_type"] == "weight"]
        kv_r = [r for r in regions_data if r["region_type"] == "kv_cache"]
        print(f"  Pre-registered {len(weight_r)} weight + {len(kv_r)} KV regions → {n_pages:,} pages")

    labelled = pd.DataFrame(list(labeller.label(iter(rows)))) if rows else pd.DataFrame()
    region_map = pd.DataFrame(labeller.region_map())
    kv_lifecycle = pd.DataFrame(labeller.kv_lifecycle())
    print(f"  {len(labelled):,} labelled records")
    print(f"  Region map     : {region_map.shape}")
    print(f"  KV lifecycle   : {kv_lifecycle.shape}")
    print()

    return labelled, region_map, kv_lifecycle, gen_results_raw, phase_boundaries, collector_info, False


def collect_with_memray(args, raw_dir: pathlib.Path, run_id: str,
                        model_cfg: ModelConfig, run_cfg: RunConfig, use_native: bool):
    from mrm_trace.collector.memray_runner import MemrayRunner
    from mrm_trace.parser.memray_parser import parse_memray
    from mrm_trace.parser.writer import write_parquet

    engine = LlamaCppEngine(model_cfg, run_cfg)
    engine.prepare()
    engine.load_model()
    print("Model loaded.")
    print()

    runner = MemrayRunner(native_traces=use_native)
    if not runner.is_available():
        print("ERROR: memray is not installed. Run: pip install memray", file=sys.stderr)
        sys.exit(1)

    print("Starting memray collector (generation phase only) ...")
    runner.start(target_pids=[], output_dir=raw_dir)

    gen_results, _, _ = _run_generation(engine)
    engine.shutdown()
    collector_meta = runner.stop()

    print()
    print(f"Capture complete — {collector_meta.output_file_size_bytes:,} bytes")
    print()

    phase_boundaries = _phase_boundaries_from_records(engine.get_phase_records())
    memray_bin = raw_dir / "memray.bin"
    trace_path = raw_dir.parent / "trace.parquet"
    mb = collector_meta.output_file_size_bytes / 1e6
    print(f"Parsing + labelling ({mb:.1f} MB, 1-in-{args.sample_every_n} sample) ...")
    labeller = TraceLabeller()
    n_rows = write_parquet(
        labeller.label(
            parse_memray(
                memray_bin,
                run_id=run_id,
                phase_boundaries=phase_boundaries,
                native_traces=use_native,
                sample_every_n=args.sample_every_n,
            )
        ),
        trace_path,
    )
    region_map = pd.DataFrame(labeller.region_map())
    kv_lifecycle = pd.DataFrame(labeller.kv_lifecycle())
    print(f"  {n_rows:,} labelled records → trace.parquet")
    print(f"  Region map     : {region_map.shape}")
    print(f"  KV lifecycle   : {kv_lifecycle.shape}")
    if region_map.empty:
        print("  (Region map is empty — expected without native_traces=True + root;")
        print("   real ggml_init / llama_kv_cache_update symbols populate it.)")

    labelled = pd.read_parquet(trace_path)

    gen_results_raw = [dataclasses.asdict(r) for r in gen_results]
    collector_info = {
        "type": "memray",
        "output_file": collector_meta.output_file,
        "output_file_size_bytes": collector_meta.output_file_size_bytes,
        "native_traces": collector_meta.extra.get("native_traces"),
    }
    return labelled, region_map, kv_lifecycle, gen_results_raw, phase_boundaries, collector_info, True


# ── Main helpers ──────────────────────────────────────────────────────────────

def _df_val(df, col, agg="mean"):
    if df is None or df.empty or col not in df.columns:
        return "n/a"
    vals = df[col].dropna()
    return round(float(vals.agg(agg)), 4) if len(vals) > 0 else "n/a"


def _resolve_collector(args, is_root: bool) -> str:
    if args.collector != "auto":
        if args.collector == "perf-mem" and not is_root:
            print("ERROR: perf mem requires root. Re-run with sudo -E venv/bin/python.", file=sys.stderr)
            sys.exit(1)
        return args.collector
    collector = _auto_select_collector(is_root)
    print(f"Auto-selected collector: {collector}", end="")
    if collector == "memray" and not _is_wsl2() and is_root:
        pmu_ok, pmu_msg = _check_pmu()
        if not pmu_ok:
            print(f"  (perf mem unavailable: {pmu_msg})", end="")
    print()
    return collector


def _print_run_header(args, collector: str, use_native: bool, out_dir: pathlib.Path) -> None:
    collector_label = (
        "perf mem (hardware PMU, spawn mode)  ← real timestamps, real cache levels"
        if collector == "perf-mem"
        else "memray (Python-level allocations)  ← NO hw timestamps, NO region map"
    )
    print("mrm-trace live demo — TinyLlama 1.1B Q4_K_M")
    print(f"  Collector     : {collector_label}")
    print(f"  Model         : {args.model}")
    print(f"  Context       : {args.ctx} tokens")
    print(f"  Output tokens : {args.tokens} per prompt")
    print(f"  Output dir    : {out_dir}")
    if collector == "memray":
        note = "(C-level allocs captured)" if use_native else "(Python-level only)"
        print(f"  native_traces : {use_native}  {note}")
    print()


def _print_analysis(analysis: dict) -> None:
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


def _build_metadata(model_cfg, run_id, gen_results_raw, collector_info, phase_boundaries):
    total_tokens = sum(r.get("output_tokens", 0) for r in gen_results_raw)
    total_time = sum(r.get("duration_s", 0) for r in gen_results_raw)
    return {
        "run_id": run_id,
        "model": {
            "id": model_cfg.id,
            "family": model_cfg.family,
            "params_billions": model_cfg.params_billions,
            "quantisation": model_cfg.quantisation,
            "dtype": model_cfg.dtype,
        },
        "generation": {
            "prompts": len(gen_results_raw),
            "total_output_tokens": total_tokens,
            "total_duration_s": round(total_time, 3),
            "avg_tokens_per_second": round(total_tokens / total_time, 1) if total_time > 0 else 0.0,
        },
        "collector": collector_info,
        "phase_boundaries": phase_boundaries,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.headless:
        run_headless(args)
        return

    model_path = pathlib.Path(args.model)
    if not model_path.exists():
        print(f"ERROR: model file not found: {model_path}", file=sys.stderr)
        print("  mkdir -p models && wget -P models/ https://huggingface.co/TheBloke/"
              "TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/"
              "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", file=sys.stderr)
        sys.exit(1)

    try:
        is_root = os.getuid() == 0
    except AttributeError:
        is_root = False

    use_native = is_root if args.native_traces is None else args.native_traces
    collector = _resolve_collector(args, is_root)

    model_cfg = _make_model_cfg(model_path)
    run_id = generate_run_id(
        model_id=model_cfg.id, context_length=args.ctx,
        batch_size=1, output_tokens=args.tokens, rep_index=0,
    )
    run_cfg = _make_run_cfg(model_cfg, run_id, args)

    out_dir = pathlib.Path(args.out) / model_cfg.id / run_id
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    _print_run_header(args, collector, use_native, out_dir)

    if collector == "perf-mem":
        labelled, region_map, kv_lifecycle, gen_results_raw, phase_boundaries, collector_info, trace_prewritten = \
            collect_with_perf_mem(args, raw_dir, run_id)
    else:
        labelled, region_map, kv_lifecycle, gen_results_raw, phase_boundaries, collector_info, trace_prewritten = \
            collect_with_memray(args, raw_dir, run_id, model_cfg, run_cfg, use_native)

    print()

    if labelled.empty:
        print("WARNING: empty trace — analysis skipped.", file=sys.stderr)
        print("Done (no trace data).")
        return

    analysis = compute_all(labelled)
    _print_analysis(analysis)

    print("Exporting ...")
    metadata = _build_metadata(model_cfg, run_id, gen_results_raw, collector_info, phase_boundaries)
    exporter = RunExporter(out_dir)
    export_result = exporter.export(
        trace=labelled, region_map=region_map, kv_lifecycle=kv_lifecycle,
        analysis=analysis, metadata=metadata, run_id=run_id,
        trace_already_written=trace_prewritten,
    )
    print(f"  Output: {out_dir}/")
    for fname in sorted(export_result.file_names):
        print(f"    {fname}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()

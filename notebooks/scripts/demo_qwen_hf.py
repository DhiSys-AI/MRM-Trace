#!/usr/bin/env python3
"""
Qwen2.5-0.5B-Instruct — live mrm-trace collection.

Collector is selected automatically:
  - Bare-metal Linux + root + PMU available → perf mem (hardware timestamps, real retention)
  - WSL2 / no root / no PMU              → memray (Python-level allocations)

perf mem uses spawn mode: perf launches this script with --headless so the
instrumented process starts under perf from the beginning, avoiding the
kernel-level attach limitation on PEBS load-latency events.

Requirements:
    pip install mrm-trace memray transformers torch accelerate

Usage:
    python demo_qwen_hf.py
    python demo_qwen_hf.py --model Qwen/Qwen2.5-1.5B-Instruct
    python demo_qwen_hf.py --tokens 256 --out /tmp/demo_results
    python demo_qwen_hf.py --collector memray     # force memray
    python demo_qwen_hf.py --collector perf-mem   # force perf mem (requires root)

    sudo -E venv/bin/python demo_qwen_hf.py       # bare-metal, auto → perf mem
"""

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

import pandas as pd

from mrm_trace.analyser import compute_all
from mrm_trace.labeller import TraceLabeller
from mrm_trace.reporter.run_exporter import RunExporter
from mrm_trace.utils.ids import generate_run_id

DEFAULT_MODEL_HF = "Qwen/Qwen2.5-0.5B-Instruct"

PROMPTS = [
    "Explain memory management in operating systems. Focus on virtual memory, paging, and TLB.",
    "Write a Python function that implements a least-recently-used (LRU) cache using a dict and a deque.",
    "What are the key differences between supervised, unsupervised, and reinforcement learning? Give one example of each.",
    "Describe the transformer architecture. Explain how multi-head self-attention differs from single-head attention.",
    "How does garbage collection work in the JVM? Compare generational GC to reference counting in CPython.",
]


# ── Environment detection ─────────────────────────────────────────────────────

def _is_wsl2() -> bool:
    try:
        return "microsoft" in pathlib.Path("/proc/sys/kernel/osrelease").read_text().lower()
    except OSError:
        return False


def _check_pmu() -> "tuple[bool, str]":
    """Probe perf mem record in spawn mode — the mode actually used for collection."""
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
    p = argparse.ArgumentParser(description="Qwen2.5 mrm-trace demo (transformers)")
    p.add_argument("--model", default=DEFAULT_MODEL_HF)
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--out", default="results")
    p.add_argument("--device", default="auto")
    p.add_argument("--collector", choices=["auto", "memray", "perf-mem"], default="auto")
    p.add_argument("--sample-every-n", dest="sample_every_n", type=int, default=25)
    p.add_argument("--sample-period", dest="sample_period", type=int, default=4000)
    # Internal flag used when perf spawns this script as the traced workload
    p.add_argument("--headless", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--results-file", dest="results_file", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


# ── Model loading + generation ────────────────────────────────────────────────

def check_transformers():
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        print("ERROR: transformers and torch are required.", file=sys.stderr)
        print("Install with: pip install transformers torch accelerate", file=sys.stderr)
        sys.exit(1)


def load_model_and_tokenizer(model_id: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading model ({model_id}) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device if device != "auto" else "auto",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def generate_one(model, tokenizer, prompt: str, max_new_tokens: int) -> "tuple[str, int, int, float]":
    import torch
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    duration = time.perf_counter() - t0
    new_ids = output_ids[0][prompt_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True), prompt_len, len(new_ids), duration


def run_generation(model, tokenizer, max_new_tokens: int) -> "tuple[list, int, int]":
    gen_results = []
    gen_start_ns = time.time_ns()
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"  [{i}/{len(PROMPTS)}] {prompt[:60]}...")
        print("          generating", end="", flush=True)
        text, prompt_tokens, output_tokens, duration = generate_one(model, tokenizer, prompt, max_new_tokens)
        gen_results.append({
            "prompt": prompt,
            "output_text": text,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "duration_s": duration,
        })
        print(f" → {output_tokens} tokens @ {output_tokens / duration:.1f} tok/s")
    return gen_results, gen_start_ns, time.time_ns()


# ── Headless mode (spawned by perf mem record) ────────────────────────────────

def run_headless(args) -> None:
    """
    Load model and generate all prompts, then write timing results to --results-file.
    This function runs inside the process that perf mem record spawns, so every
    memory access from model load onwards is captured by the hardware PMU.
    """
    if not args.results_file:
        print("ERROR: --results-file required in --headless mode", file=sys.stderr)
        sys.exit(1)

    check_transformers()

    load_start_ns = time.time_ns()
    model, tokenizer = load_model_and_tokenizer(args.model, args.device)
    load_end_ns = time.time_ns()
    print(f"Model loaded in {(load_end_ns - load_start_ns) / 1e9:.1f}s")
    print()

    gen_results, gen_start_ns, gen_end_ns = run_generation(model, tokenizer, args.tokens)
    del model, tokenizer

    results = {
        "load_start_ns": load_start_ns,
        "load_end_ns": load_end_ns,
        "gen_start_ns": gen_start_ns,
        "gen_end_ns": gen_end_ns,
        "gen_results": gen_results,
    }
    pathlib.Path(args.results_file).write_text(json.dumps(results))
    print()
    print("Headless run complete — perf mem will flush on exit.")


# ── Collection paths ──────────────────────────────────────────────────────────

def collect_with_perf_mem(args, raw_dir: pathlib.Path, run_id: str):
    """
    Spawn this script under perf mem record so the entire process lifetime
    (model load + generation) is instrumented from the start.
    Returns (labelled, region_map, kv_lifecycle, gen_results, phase_boundaries, collector_info).
    """
    from mrm_trace.parser.perf_script_parser import parse_perf_script

    perf = shutil.which("perf")
    results_file = raw_dir.parent / "generation_results.json"
    perf_data = raw_dir / "perf.data"

    # Build the headless command: re-invoke this script under perf
    inner_cmd = [
        sys.executable, str(pathlib.Path(__file__).resolve()),
        "--headless",
        "--model", args.model,
        "--ctx", str(args.ctx),
        "--tokens", str(args.tokens),
        "--device", args.device,
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
    gen_results = timing["gen_results"]
    phase_boundaries = [
        {"phase": "weight_load", "start_ns": timing["load_start_ns"], "end_ns": timing["load_end_ns"]},
        {"phase": "generation",  "start_ns": timing["gen_start_ns"],  "end_ns": timing["gen_end_ns"]},
    ]

    # Run perf script to convert perf.data → text
    print("Running perf script ...")
    from mrm_trace.collector.perf_runner import PerfMemRunner
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

    # Diagnostic: show first 3 lines so format issues are immediately visible
    try:
        with open(perf_script_txt) as _fh:
            _sample = [next(_fh).rstrip() for _ in range(3)]
        print(f"  perf_script.txt format sample: {_sample[0]}")
    except (StopIteration, OSError):
        pass

    rows = list(parse_perf_script(perf_script_txt, run_id=run_id, phase_boundaries=phase_boundaries))
    if not rows:
        print("WARNING: no samples parsed from perf script output.", file=sys.stderr)
        print(f"  Check format: head -5 {perf_script_txt}", file=sys.stderr)
        print("  Continuing with empty trace — analysis metrics will be zero.", file=sys.stderr)
        rows = []

    labeller = TraceLabeller()
    labelled = pd.DataFrame(list(labeller.label(iter(rows)))) if rows else pd.DataFrame()
    region_map = pd.DataFrame(labeller.region_map())
    kv_lifecycle = pd.DataFrame(labeller.kv_lifecycle())
    print(f"  {len(labelled):,} labelled records")
    print(f"  Region map     : {region_map.shape}")
    print(f"  KV lifecycle   : {kv_lifecycle.shape}")
    if region_map.empty:
        print("  (Region map is empty — PyTorch uses internal symbols, not ggml_* names.)")
    print()

    return labelled, region_map, kv_lifecycle, gen_results, phase_boundaries, collector_info


def collect_with_memray(model, tokenizer, args, raw_dir: pathlib.Path, run_id: str,
                        load_start_ns: int, load_end_ns: int):
    from mrm_trace.collector.memray_runner import MemrayRunner
    from mrm_trace.parser.memray_parser import parse_memray
    from mrm_trace.parser.writer import write_parquet

    is_root = os.getuid() == 0
    runner = MemrayRunner(native_traces=is_root)
    if not runner.is_available():
        print("ERROR: memray is not installed. Run: pip install memray", file=sys.stderr)
        sys.exit(1)

    phase_boundaries = [
        {"phase": "weight_load", "start_ns": load_start_ns, "end_ns": load_end_ns},
    ]

    print("Starting memray collector (generation phase only) ...")
    runner.start(target_pids=[], output_dir=raw_dir)

    gen_results, gen_start_ns, gen_end_ns = run_generation(model, tokenizer, args.tokens)
    phase_boundaries.append({"phase": "generation", "start_ns": gen_start_ns, "end_ns": gen_end_ns})
    del model, tokenizer

    collector_meta = runner.stop()
    print()
    print(f"Capture complete — {collector_meta.output_file_size_bytes:,} bytes")
    print()

    out_dir = raw_dir.parent
    trace_path = out_dir / "trace.parquet"
    mb = collector_meta.output_file_size_bytes / 1e6
    print(f"Parsing + labelling ({mb:.1f} MB, 1-in-{args.sample_every_n} sample) ...")
    labeller = TraceLabeller()
    n_rows = write_parquet(
        labeller.label(
            parse_memray(
                raw_dir / "memray.bin",
                run_id=run_id,
                phase_boundaries=phase_boundaries,
                native_traces=is_root,
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
        print("  (Region map is empty — transformers uses PyTorch internals, not ggml_ symbols.)")

    labelled = pd.read_parquet(trace_path)
    collector_info = {
        "type": "memray",
        "output_file": collector_meta.output_file,
        "output_file_size_bytes": collector_meta.output_file_size_bytes,
        "native_traces": collector_meta.extra.get("native_traces"),
    }
    return labelled, region_map, kv_lifecycle, gen_results, phase_boundaries, collector_info, True


# ── Main helpers ──────────────────────────────────────────────────────────────

def build_metadata(model_id, run_id, gen_results, collector_info, phase_boundaries, args):
    total_tokens = sum(r["output_tokens"] for r in gen_results)
    total_time = sum(r["duration_s"] for r in gen_results)
    return {
        "run_id": run_id,
        "model": {"id": model_id, "hf_repo": args.model, "backend": "transformers"},
        "generation": {
            "prompts": len(gen_results),
            "total_output_tokens": total_tokens,
            "total_duration_s": round(total_time, 3),
            "avg_tokens_per_second": round(total_tokens / total_time, 1) if total_time > 0 else 0.0,
        },
        "collector": collector_info,
        "phase_boundaries": phase_boundaries,
    }


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


def _print_run_header(args, collector: str, is_root: bool, out_dir: pathlib.Path) -> None:
    collector_label = (
        "perf mem (hardware PMU, spawn mode)  ← real timestamps, real cache levels"
        if collector == "perf-mem"
        else "memray (Python-level allocations)  ← NO hw timestamps, NO region map"
    )
    print(f"mrm-trace live demo — {args.model}")
    print(f"  Collector     : {collector_label}")
    print(f"  Context       : {args.ctx} tokens")
    print(f"  Output tokens : {args.tokens} per prompt")
    print(f"  Output dir    : {out_dir}")
    if collector == "memray":
        native_note = "(C-level allocs captured)" if is_root else "(Python-level only)"
        print(f"  native_traces : {is_root}  {native_note}")
    print()


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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Headless mode: run by perf mem record as the traced workload
    if args.headless:
        run_headless(args)
        return

    check_transformers()

    try:
        is_root = os.getuid() == 0
    except AttributeError:
        is_root = False

    collector = _resolve_collector(args, is_root)

    model_slug = args.model.split("/")[-1].lower()
    run_id = generate_run_id(
        model_id=model_slug, context_length=args.ctx,
        batch_size=1, output_tokens=args.tokens, rep_index=0,
    )
    out_dir = pathlib.Path(args.out) / model_slug / run_id
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    _print_run_header(args, collector, is_root, out_dir)

    if collector == "perf-mem":
        labelled, region_map, kv_lifecycle, gen_results, phase_boundaries, collector_info = \
            collect_with_perf_mem(args, raw_dir, run_id)
        trace_prewritten = False
    else:
        load_start_ns = time.time_ns()
        model, tokenizer = load_model_and_tokenizer(args.model, args.device)
        load_end_ns = time.time_ns()
        print(f"Model loaded in {(load_end_ns - load_start_ns) / 1e9:.1f}s")
        print()
        labelled, region_map, kv_lifecycle, gen_results, phase_boundaries, collector_info, trace_prewritten = \
            collect_with_memray(model, tokenizer, args, raw_dir, run_id, load_start_ns, load_end_ns)
    print()

    if labelled.empty:
        print("WARNING: empty trace — analysis skipped. Check perf_script.txt format.", file=sys.stderr)
        print("Done (no trace data).")
        return

    analysis = compute_all(labelled)
    _print_analysis_summary(analysis)

    print("Exporting ...")
    metadata = build_metadata(model_slug, run_id, gen_results, collector_info, phase_boundaries, args)
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

    if gen_results:
        print()
        print("─" * 60)
        print("Sample output (prompt 1):")
        print(gen_results[0]["output_text"][:400])
        print("─" * 60)


if __name__ == "__main__":
    main()

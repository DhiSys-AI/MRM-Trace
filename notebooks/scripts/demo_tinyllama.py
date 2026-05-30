#!/usr/bin/env python3
"""
TinyLlama 1.1B — live mrm-trace collection via memray + llama-cpp-python.

Downloads the model once, then runs 5 real prompts and writes all mrm-trace
artifacts (trace.parquet, metrics.csv, metadata.json, …) to a timestamped
results directory.

Requirements:
    pip install mrm-trace llama-cpp-python memray

Model download (run once):
    mkdir -p models
    wget -P models/ https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

Usage:
    python demo_tinyllama.py
    python demo_tinyllama.py --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
    python demo_tinyllama.py --ctx 512 --tokens 64 --out /tmp/demo_results

Chat-template note:
    TinyLlama-1.1B-Chat is a chat-finetuned model. Without the <|system|>/<|user|>/<|assistant|>
    template, some prompts generate 0 tokens (model emits immediate EOG). Prompts are wrapped
    automatically by _fmt() below.

region_map / kv_lifecycle note:
    Both will be (0, 0) with pip-installed llama-cpp-python, even with --native-traces.
    Pip builds strip C symbols so ggml_* names never appear in the trace.
    Only perf mem (bare-metal Linux) produces a populated region_map.
"""

import argparse
import dataclasses
import os
import pathlib
import sys
import time

import pandas as pd

from mrm_trace.analyser import compute_all
from mrm_trace.collector.memray_runner import MemrayRunner
from mrm_trace.config.schema import EngineType, ModelConfig, RunConfig
from mrm_trace.engines.base import PhaseEvent
from mrm_trace.engines.llamacpp import LlamaCppEngine
from mrm_trace.labeller import TraceLabeller
from mrm_trace.parser.memray_parser import parse_memray
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

# TinyLlama-1.1B-Chat is a chat-finetuned model that expects this exact format.
# Sending raw prompts triggers immediate EOG on some prompts (0 tokens generated).
def _fmt(prompt: str) -> str:
    return (
        "<|system|>\nYou are a helpful AI assistant.</s>\n"
        f"<|user|>\n{prompt}</s>\n<|assistant|>\n"
    )

PROMPTS = [_fmt(p) for p in _RAW_PROMPTS]


def parse_args():
    p = argparse.ArgumentParser(description="TinyLlama mrm-trace demo")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Path to GGUF model file")
    p.add_argument("--ctx", type=int, default=2048, help="Context length")
    p.add_argument("--tokens", type=int, default=128, help="Max output tokens per prompt")
    p.add_argument("--out", default="results", help="Output base directory")
    p.add_argument(
        "--native-traces", dest="native_traces", action="store_true", default=None,
        help="Force native_traces=True for memray (captures C-level allocs; may need root)",
    )
    p.add_argument(
        "--no-native-traces", dest="native_traces", action="store_false",
        help="Force native_traces=False (Python-level only, always works)",
    )
    p.add_argument(
        "--sample-every-n", dest="sample_every_n", type=int, default=1,
        help="Parse every Nth allocation record (default 1 = all). Use 5-10 for large traces.",
    )
    return p.parse_args()


def build_phase_boundaries(phase_records):
    """
    Convert engine PhaseRecord list into parse_memray phase_boundaries format.

    Note: memray records allocations with timestamp=0; phase annotation via
    wall_ns only works correctly with perf mem (hardware timestamps). These
    boundaries are stored in metadata but will not annotate individual rows
    in a memray trace.
    """
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


def build_metadata(model_cfg, run_id, engine, gen_results, collector_meta):
    total_tokens = sum(r.output_tokens for r in gen_results)
    total_time = sum(r.duration_s for r in gen_results)
    runtime_meta = dataclasses.asdict(engine.get_runtime_metadata())
    return {
        "run_id": run_id,
        "model": {
            "id": model_cfg.id,
            "family": model_cfg.family,
            "params_billions": model_cfg.params_billions,
            "quantisation": model_cfg.quantisation,
            "dtype": model_cfg.dtype,
        },
        "engine": runtime_meta,
        "generation": {
            "prompts": len(gen_results),
            "total_output_tokens": total_tokens,
            "total_duration_s": round(total_time, 3),
            "avg_tokens_per_second": round(total_tokens / total_time, 1) if total_time > 0 else 0.0,
        },
        "collector": {
            "type": "memray",
            "output_file": collector_meta.output_file,
            "output_file_size_bytes": collector_meta.output_file_size_bytes,
            "native_traces": collector_meta.extra.get("native_traces"),
        },
    }


def main():
    args = parse_args()

    model_path = pathlib.Path(args.model)
    if not model_path.exists():
        print(f"ERROR: model file not found: {model_path}", file=sys.stderr)
        print("Download with:", file=sys.stderr)
        print(
            "  mkdir -p models && wget -P models/ "
            "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/"
            "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        is_root = os.getuid() == 0
    except AttributeError:
        is_root = False

    # --native-traces / --no-native-traces override; default = auto (root check)
    if args.native_traces is None:
        use_native = is_root
    else:
        use_native = args.native_traces

    model_cfg = ModelConfig(
        id="tinyllama-1.1b-q4km",
        engine=EngineType.LLAMACPP,
        path=str(model_path),
        family="llama",
        params_billions=1.1,
        quantisation="Q4_K_M",
        dtype="float16",
    )

    run_id = generate_run_id(
        model_id=model_cfg.id,
        context_length=args.ctx,
        batch_size=1,
        output_tokens=args.tokens,
        rep_index=0,
    )

    run_cfg = RunConfig(
        run_id=run_id,
        model=model_cfg,
        context_length=args.ctx,
        batch_size=1,
        output_tokens=args.tokens,
        rep_index=0,
    )

    out_dir = pathlib.Path(args.out) / model_cfg.id / run_id
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"mrm-trace live demo — TinyLlama 1.1B Q4_K_M")
    print(f"  Model         : {model_path}")
    print(f"  Context       : {args.ctx} tokens")
    print(f"  Output tokens : {args.tokens} per prompt")
    print(f"  Output dir    : {out_dir}")
    print(f"  native_traces : {use_native}  {'(C-level allocs captured)' if use_native else '(Python-level only)'}")
    print()

    engine = LlamaCppEngine(model_cfg, run_cfg)
    engine.prepare()

    runner = MemrayRunner(native_traces=use_native)
    if not runner.is_available():
        print("ERROR: memray is not installed. Run: pip install memray", file=sys.stderr)
        sys.exit(1)

    # ── Phase 1: collection ───────────────────────────────────────────────────
    print("Starting memray collector ...")
    runner.start(target_pids=[], output_dir=raw_dir)

    engine.load_model()
    print(f"Model loaded.")
    print()

    gen_results = []
    for i, (prompt, raw_prompt) in enumerate(zip(PROMPTS, _RAW_PROMPTS), 1):
        print(f"  [{i}/{len(PROMPTS)}] {raw_prompt[:60]}...")
        print(f"          generating", end="", flush=True)
        result = engine.generate(prompt)
        gen_results.append(result)
        print(f" → {result.output_tokens} tokens @ {result.tokens_per_second:.1f} tok/s")

    engine.shutdown()
    collector_meta = runner.stop()

    print()
    print(f"Capture complete — {collector_meta.output_file_size_bytes:,} bytes")
    print()

    # ── Phase 2: parse ────────────────────────────────────────────────────────
    phase_boundaries = build_phase_boundaries(engine.get_phase_records())
    memray_bin = raw_dir / "memray.bin"

    print(f"Parsing trace ({collector_meta.output_file_size_bytes / 1e6:.1f} MB) ...")
    rows = list(parse_memray(memray_bin, run_id=run_id, phase_boundaries=phase_boundaries, native_traces=use_native, sample_every_n=args.sample_every_n))
    print(f"  {len(rows):,} allocation records")
    if rows:
        op_counts: dict = {}
        phase_counts: dict = {}
        for r in rows:
            op_counts[r["op_type"]] = op_counts.get(r["op_type"], 0) + 1
            phase_counts[r["phase"]] = phase_counts.get(r["phase"], 0) + 1
        print(f"  op_type  : {op_counts}")
        print(f"  phases   : {phase_counts}")
    print()

    # ── Phase 3: label ────────────────────────────────────────────────────────
    # Pass rows directly — avoids a costly DataFrame → dict round-trip.
    print("Labelling ...")
    labeller = TraceLabeller()
    labelled = pd.DataFrame(list(labeller.label(rows)))
    region_map = pd.DataFrame(labeller.region_map())
    kv_lifecycle = pd.DataFrame(labeller.kv_lifecycle())
    print(f"  Labelled trace : {labelled.shape}")
    print(f"  Region map     : {region_map.shape}")
    print(f"  KV lifecycle   : {kv_lifecycle.shape}")
    if region_map.empty:
        print("  (Region map is empty — expected without native_traces=True + root;")
        print("   real ggml_init / llama_kv_cache_update symbols populate it.)")
    print()

    # ── Phase 4: analyse ──────────────────────────────────────────────────────
    print("Analysing ...")
    analysis = compute_all(labelled)

    def _df_val(df, col, agg="mean"):
        if df is None or df.empty or col not in df.columns:
            return "n/a"
        vals = df[col].dropna()
        return round(float(vals.agg(agg)), 4) if len(vals) > 0 else "n/a"

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

    # ── Phase 5: export ───────────────────────────────────────────────────────
    print("Exporting ...")
    metadata = build_metadata(model_cfg, run_id, engine, gen_results, collector_meta)
    exporter = RunExporter(out_dir)
    export_result = exporter.export(
        trace=labelled,
        region_map=region_map,
        kv_lifecycle=kv_lifecycle,
        analysis=analysis,
        metadata=metadata,
        run_id=run_id,
    )

    print(f"  Output: {out_dir}/")
    for fname in sorted(export_result.file_names):
        print(f"    {fname}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()

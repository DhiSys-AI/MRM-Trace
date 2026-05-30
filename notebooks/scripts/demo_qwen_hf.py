#!/usr/bin/env python3
"""
Qwen2.5-0.5B-Instruct — live mrm-trace collection via memray + transformers.

The model (~1 GB) downloads from HuggingFace automatically on first run.
No GGUF file needed. Runs 5 real prompts and writes all mrm-trace artifacts.

Requirements:
    pip install mrm-trace memray transformers torch accelerate

Usage:
    python demo_qwen_hf.py
    python demo_qwen_hf.py --model Qwen/Qwen2.5-1.5B-Instruct   # larger
    python demo_qwen_hf.py --tokens 256 --out /tmp/demo_results
"""

import argparse
import os
import pathlib
import sys
import time

import pandas as pd

from mrm_trace.analyser import compute_all
from mrm_trace.collector.memray_runner import MemrayRunner
from mrm_trace.labeller import TraceLabeller
from mrm_trace.parser.memray_parser import parse_memray
from mrm_trace.parser.writer import write_parquet
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


def parse_args():
    p = argparse.ArgumentParser(description="Qwen2.5 mrm-trace demo (transformers)")
    p.add_argument("--model", default=DEFAULT_MODEL_HF, help="HuggingFace model ID or local path")
    p.add_argument("--ctx", type=int, default=2048, help="Max context length")
    p.add_argument("--tokens", type=int, default=128, help="Max new tokens per prompt")
    p.add_argument("--out", default="results", help="Output base directory")
    p.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, mps")
    p.add_argument(
        "--sample-every-n", dest="sample_every_n", type=int, default=25,
        help="Parse every Nth allocation record (default 25). Lower = denser but slower; 1 = all (may OOM).",
    )
    return p.parse_args()


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


def generate_one(model, tokenizer, prompt: str, max_new_tokens: int) -> tuple[str, int, int, float]:
    """Returns (output_text, prompt_tokens, output_tokens, duration_s)."""
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
    output_text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return output_text, prompt_len, len(new_ids), duration


def build_metadata(model_id, run_id, gen_results, collector_meta, phase_boundaries, args):
    total_tokens = sum(r["output_tokens"] for r in gen_results)
    total_time = sum(r["duration_s"] for r in gen_results)
    return {
        "run_id": run_id,
        "model": {
            "id": model_id,
            "hf_repo": args.model,
            "backend": "transformers",
        },
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
        "phase_boundaries": phase_boundaries,
    }


def main():
    args = parse_args()
    check_transformers()

    try:
        is_root = os.getuid() == 0
    except AttributeError:
        is_root = False

    # Derive a filesystem-safe model id from the HF repo name
    model_slug = args.model.split("/")[-1].lower()

    run_id = generate_run_id(
        model_id=model_slug,
        context_length=args.ctx,
        batch_size=1,
        output_tokens=args.tokens,
        rep_index=0,
    )

    out_dir = pathlib.Path(args.out) / model_slug / run_id
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"mrm-trace live demo — {args.model}")
    print(f"  Collector     : memray (Python-level allocations)  ← NO hw timestamps, NO region map")
    print(f"  Context       : {args.ctx} tokens")
    print(f"  Output tokens : {args.tokens} per prompt")
    print(f"  Output dir    : {out_dir}")
    print(f"  native_traces : {is_root}  {'(C-level allocs captured)' if is_root else '(Python-level only)'}")
    print()

    runner = MemrayRunner(native_traces=is_root)
    if not runner.is_available():
        print("ERROR: memray is not installed. Run: pip install memray", file=sys.stderr)
        sys.exit(1)

    # ── Load model BEFORE starting collector ──────────────────────────────────
    # Model loading (290+ tensors, ~1 GB) creates tens of millions of Python-level
    # allocation events. Capturing that phase makes the memray file 10-20x larger
    # and causes parsing to take 30+ minutes. Only the generation phase is captured.
    load_start_ns = time.time_ns()
    model, tokenizer = load_model_and_tokenizer(args.model, args.device)
    load_end_ns = time.time_ns()
    print(f"Model loaded in {(load_end_ns - load_start_ns) / 1e9:.1f}s")
    print()

    phase_boundaries = [
        {"phase": "weight_load", "start_ns": load_start_ns, "end_ns": load_end_ns},
    ]

    # ── Phase 1: collection (generation only) ─────────────────────────────────
    print("Starting memray collector (generation phase only) ...")
    runner.start(target_pids=[], output_dir=raw_dir)

    gen_results = []
    gen_start_ns = time.time_ns()
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"  [{i}/{len(PROMPTS)}] {prompt[:60]}...")
        print(f"          generating", end="", flush=True)
        text, prompt_tokens, output_tokens, duration = generate_one(model, tokenizer, prompt, args.tokens)
        gen_results.append({
            "prompt": prompt,
            "output_text": text,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "duration_s": duration,
        })
        tok_s = output_tokens / duration if duration > 0 else 0
        print(f" → {output_tokens} tokens @ {tok_s:.1f} tok/s")

    gen_end_ns = time.time_ns()
    phase_boundaries.append({"phase": "generation", "start_ns": gen_start_ns, "end_ns": gen_end_ns})

    # Teardown
    del model
    del tokenizer

    collector_meta = runner.stop()

    print()
    print(f"Capture complete — {collector_meta.output_file_size_bytes:,} bytes")
    print()

    # ── Phase 2+3: parse → label → Parquet (streaming, one chunk at a time) ───
    # Records flow: parse_memray() → labeller.label() → write_parquet() → disk
    # Peak RAM = one 100k-row batch (~130 MB) not the full trace.
    memray_bin = raw_dir / "memray.bin"
    trace_path = out_dir / "trace.parquet"

    mb = collector_meta.output_file_size_bytes / 1e6
    print(f"Parsing + labelling ({mb:.1f} MB, 1-in-{args.sample_every_n} sample) ...")
    labeller = TraceLabeller()
    n_rows = write_parquet(
        labeller.label(
            parse_memray(
                memray_bin,
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
        print("  (Region map is empty — transformers uses PyTorch internals, not ggml_ symbols.")
        print("   Run with llama-cpp-python + native_traces=True for classified regions.)")
    print()

    # Read back from Parquet for analysis — sampled trace fits in RAM comfortably.
    labelled = pd.read_parquet(trace_path)

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
    metadata = build_metadata(model_slug, run_id, gen_results, collector_meta, phase_boundaries, args)
    exporter = RunExporter(out_dir)
    export_result = exporter.export(
        trace=labelled,
        region_map=region_map,
        kv_lifecycle=kv_lifecycle,
        analysis=analysis,
        metadata=metadata,
        run_id=run_id,
        trace_already_written=True,
    )

    print(f"  Output: {out_dir}/")
    for fname in sorted(export_result.file_names):
        print(f"    {fname}")

    print()
    print("Done.")

    # Print one sample output
    if gen_results:
        print()
        print("─" * 60)
        print(f"Sample output (prompt 1):")
        print(gen_results[0]["output_text"][:400])
        print("─" * 60)


if __name__ == "__main__":
    main()

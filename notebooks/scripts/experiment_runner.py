#!/usr/bin/env python3
"""
experiment_runner.py — MRM-trace full experiment sweep runner.

Runs all model × context × batch × rep combinations, with:
  - Thermal safety (waits for CPU to cool before each run)
  - Crash-safe state (state file updated atomically after every run)
  - Resume support (re-run to pick up where you left off)
  - Per-run timeout (hard-kills runs that hang)
  - Live progress display

Usage:
  sudo -E venv/bin/python notebooks/scripts/experiment_runner.py            # run or resume
  sudo -E venv/bin/python notebooks/scripts/experiment_runner.py --status   # show state, no run
  sudo -E venv/bin/python notebooks/scripts/experiment_runner.py --retry-failed
  sudo -E venv/bin/python notebooks/scripts/experiment_runner.py --restart  # discard state
  sudo -E venv/bin/python notebooks/scripts/experiment_runner.py --dry-run  # print plan only
  sudo -E venv/bin/python notebooks/scripts/experiment_runner.py --fast     # 1 model validation
  sudo -E venv/bin/python notebooks/scripts/experiment_runner.py --cpu-only
  sudo -E venv/bin/python notebooks/scripts/experiment_runner.py --gpu-only
"""

import argparse
import csv
import json
import os
import pathlib
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from mrm_trace.telemetry.thermal import read_package_temp_c
from mrm_trace.telemetry.validity import (
    TEMP_CONTAMINATED_C,
    TEMP_WARN_C,
    classify_from_metadata,
)
from mrm_trace.utils.machine import machine_slug

# ── Configuration ─────────────────────────────────────────────────────────────

RESULTS_DIR = pathlib.Path("results")
STATE_FILE = RESULTS_DIR / ".experiment_state.json"
LOG_FILE = RESULTS_DIR / "logs" / "runner.log"

# Script paths (relative to repo root)
SCRIPT_PERF_MEM = pathlib.Path("notebooks/scripts/demo_perf_mem.py")
SCRIPT_HF_GPU = pathlib.Path("notebooks/scripts/demo_qwen_hf.py")

# Python interpreter: the current venv's python (runner must be started via venv)
PYTHON = sys.executable

# Thermal thresholds (°C) — workplan §9.2
# TEMP_WARN_C and TEMP_CONTAMINATED_C are imported from validity.py (single source of truth).
TEMP_START_MAX = 60             # do not start a run above this (§9.2: < 60°C safe to start)
TEMP_COOL_TARGET = 55           # extended post-run wait target (§9.3: wait until < 55°C)
# Fallback-only: used when child's metadata.json can't be read.
# Matches TEMP_CONTAMINATED_C from validity.py (> 90°C → contaminated).
TEMP_CONTAMINATED_FALLBACK_C = TEMP_CONTAMINATED_C
COOL_CHECK_INTERVAL = 60        # seconds between temperature checks while cooling

# Timing — workplan §9.3
MIN_COOL_SECONDS = 60           # mandatory cooldown after every run (sensor-settle time)
LARGE_MODEL_EXTRA_COOL = 60     # extra time for 7B–8B models; wait_for_cool() handles the rest
# Per-run timeout. A flat 2 hours silently hard-kills every legitimate long
# run: at OUTPUT_TOKENS=512 the estimates reach 5.2 h (llama-3.1-8b, ctx
# 32768), so 27 of the 135 CPU runs would have died for a reason unrelated to
# the science. The timeout is therefore derived from each run's own estimate —
# generous enough not to interfere, tight enough to still catch a hang.
MAX_RUN_SECONDS = 7200          # floor, and the timeout for uncalibrated runs
RUN_TIMEOUT_FACTOR = 3.0        # x the estimated collection time
MAX_RUN_SECONDS_CEILING = 43200 # 12 h — beyond this something is wrong
PROGRESS_INTERVAL = 120         # re-print progress every 2 min

# Output tokens (fixed across all runs for comparability)
# 512 gives a longer KV cache growth window and cleaner retention histograms vs 256.
OUTPUT_TOKENS = 512

# ── Experiment Matrix ──────────────────────────────────────────────────────────

# ── Sampling budget ───────────────────────────────────────────────────────────
#
# perf mem's -c is a *period*: one sample per N memory accesses.  Sample count
# therefore scales with how long the run takes, not with how big the model is,
# and a fixed per-model period is the wrong instrument.  Measured on the dev
# box: 30,608,373 samples over 83.1 s of generation at -c 32000, i.e.
#
#     368k samples/s  ->  1.18e10 memory accesses/s
#
# The same period that gives a usable 1B run therefore gives ~2 billion samples
# on an 8B run, which no amount of streaming will analyse on a 32 GB machine.
# So the period is derived per run from a fixed sample budget instead.
# Measured across 8 llama-3.2-1b runs at -c 128000: 40.5M samples over a
# 414-455 s collection window => 1.221e10. Stable to ~3% across those runs.
_ACCESS_RATE_PER_S = 1.221e10

# Rows the analysis stage can hold.  A 30M-row trace was measured end to end at
# a 15 GB peak RSS, which fits a 32 GB box alongside the OS and the model.
# Raise it only after re-measuring; it is a memory limit, not a preference.
SAMPLE_BUDGET = 30_000_000

# Never go denser than this however short the run, so a fast model still
# produces a trace of manageable size.
MIN_SAMPLE_PERIOD = 10_000

# Region granularity for retention analysis (workplan §5.1.2).  No value is
# universally correct — it depends on sample density and working-set size — so
# every run prints a sweep with a usable/undersampled/saturated verdict and a
# recommendation.  Read it; this is a starting point, not a setting to trust.
REGION_GRANULARITY_BYTES = 65_536

# CPU path: GGUF models via demo_perf_mem.py + perf mem
# Each entry: (model_id, model_file, params_b, tok_per_s_at_ref_ctx)
#
# tok_per_s is generation throughput *under perf* at CTX_REF — perf mem roughly
# quarters throughput, so this is not the figure you get unloaded.  It only
# seeds the run-length estimate for the sampling budget, and only until the
# first run of each (model, ctx) measures the real duration (see the
# calibration below).  params_b is not a usable proxy: Llama-3.2-1B (1.0B) runs
# slower than TinyLlama (1.1B) on this CPU.
#
# llama-3.2-1b is measured: 8.38 tok/s over the collection window at ctx 2048,
# mean of 8 runs.  The rest are scaled from it by parameter count and will be
# replaced by measurement on their own first run.
CTX_REF = 2048

CPU_MODELS = [
    #("tinyllama-1.1b",  "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",       1.1,   9.2),
    ("llama-3.2-1b",    "Llama-3.2-1B-Instruct-Q4_K_M.gguf",          1.0,   8.38),  # measured
    ("llama-3.2-3b",    "Llama-3.2-3B-Instruct-Q4_K_M.gguf",          3.0,   3.05),
    ("phi-3.5-mini",    "Phi-3.5-mini-instruct-Q4_K_M.gguf",           3.8,   2.45),
    ("mistral-7b",      "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",        7.0,   1.30),
    ("llama-3.1-8b",    "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",      8.0,   1.15),
]

# Prompts per run, passed to demo_perf_mem.py as --num-prompts (the script
# defaults to all 7).  Run length is linear in this, and at OUTPUT_TOKENS=512
# the full 7 put the sweep at ~201 h of active runtime.  4 brings it to ~115 h
# while keeping the 512-token KV growth window and all three context lengths;
# repetition still comes from 3 batch sizes x 3 reps per configuration.
PROMPTS_PER_RUN = 4

# Decode slows as the context grows, because attention scans more of it.
# Two measured points on llama-3.2-1b: 11.2 tok/s at ctx 512 and 8.38 at 2048,
# which this linear-in-ctx form reproduces with k = 1/3.  It is an
# extrapolation beyond 2048 — deliberately one that *over*-estimates run
# length, so an uncalibrated run comes out under budget rather than over.
# Every (model, ctx) pair stops relying on it after its first run.
_CTX_SLOWDOWN_K = 1.0 / 3.0

# Measured sample counts, written after each run, keyed "<model_id>__ctx<ctx>".
# rep 0 of each configuration is a warmup, so its measurement calibrates the
# reps that carry data.
CALIBRATION_FILE = RESULTS_DIR / ".sample_calibration.json"


def run_timeout_for(est_seconds: float) -> int:
    """Hang-detection timeout for a run expected to take *est_seconds*."""
    return int(min(MAX_RUN_SECONDS_CEILING,
                   max(MAX_RUN_SECONDS, est_seconds * RUN_TIMEOUT_FACTOR)))


def tok_per_s_at(tok_per_s_ref: float, ctx: int) -> float:
    """Throughput at *ctx*, scaled from the CTX_REF measurement."""
    scale = 1.0 + _CTX_SLOWDOWN_K * (ctx / CTX_REF - 1.0)
    return tok_per_s_ref / max(scale, 0.1)


def estimate_run_seconds(tok_per_s_ref: float, ctx: int = CTX_REF,
                         output_tokens: int = OUTPUT_TOKENS,
                         prompts: int = PROMPTS_PER_RUN) -> float:
    """Collection seconds for one run.  Decode dominates at 512 output tokens."""
    return prompts * output_tokens / max(tok_per_s_at(tok_per_s_ref, ctx), 0.1)


def _load_calibration() -> Dict[str, dict]:
    try:
        return json.loads(CALIBRATION_FILE.read_text())
    except (OSError, ValueError):
        return {}


def _calibration_key(model_id: str, ctx: int, output_tokens: int, prompts: int) -> str:
    """
    Identify a calibration measurement by everything that sets run length.

    Context and prompt count both scale the run, so a measurement taken at 7
    prompts must not be reused after dropping to 4 — it would coarsen the
    period by 7/4 and halve the sample count.
    """
    return f"{model_id}__ctx{ctx}__t{output_tokens}__p{prompts}"


def record_calibration(model_id: str, ctx: int, period: int, samples: int,
                       output_tokens: int = OUTPUT_TOKENS,
                       prompts: int = PROMPTS_PER_RUN) -> None:
    """
    Remember what a period actually produced, so the next run of the same
    (model, ctx) can correct for it.

    The access rate is stable to ~3%; run length is what the estimate gets
    wrong — 8.38 tok/s measured against 11.0 assumed put the first 8 runs 35%
    over budget.  Measuring beats estimating, so each configuration estimates
    once and measures thereafter.
    """
    if samples <= 0 or period <= 0:
        return
    data = _load_calibration()
    data[_calibration_key(model_id, ctx, output_tokens, prompts)] = {
        "period": period, "samples": samples, "recorded_at": _now(),
        "ctx": ctx, "output_tokens": output_tokens, "prompts": prompts,
    }
    try:
        CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        CALIBRATION_FILE.write_text(json.dumps(data, indent=2, sort_keys=True))
    except OSError:
        pass


def sample_period_for(tok_per_s_ref: float, ctx: int = CTX_REF,
                      output_tokens: int = OUTPUT_TOKENS,
                      model_id: Optional[str] = None,
                      calibration: Optional[Dict[str, dict]] = None) -> int:
    """
    Period that keeps a run inside SAMPLE_BUDGET.

    A longer run gets a coarser period, which is the point: every run in the
    matrix then yields a comparable number of samples instead of one that
    scales with wall time.

    When a previous run of this (model, ctx) was measured, the period is
    corrected proportionally from what that run actually produced — no duration
    model involved.  Otherwise it is estimated.
    """
    if model_id is not None:
        cal = (calibration if calibration is not None else _load_calibration())
        prev = cal.get(_calibration_key(model_id, ctx, output_tokens,
                                        PROMPTS_PER_RUN))
        if prev and prev.get("samples", 0) > 0 and prev.get("period", 0) > 0:
            period = prev["period"] * prev["samples"] / SAMPLE_BUDGET
            return max(MIN_SAMPLE_PERIOD, int(round(period, -3)))

    est_s = estimate_run_seconds(tok_per_s_ref, ctx, output_tokens)
    period = _ACCESS_RATE_PER_S * est_s / SAMPLE_BUDGET
    return max(MIN_SAMPLE_PERIOD, int(round(period, -3)))

CONTEXT_LENGTHS = [2048, 8192, 32768]
BATCH_SIZES = [1, 4, 8]
REPS = [0, 1, 2]  # rep 0 = warmup; reps 1 and 2 = data

# GPU path: HuggingFace models via demo_qwen_hf.py + PyTorch
# Each entry: (model_id, hf_repo, max_ctx)
GPU_MODELS = [
    ("qwen2.5-0.5b-gpu",  "Qwen/Qwen2.5-0.5B-Instruct",                2048),
    ("llama-3.2-3b-gpu",  "meta-llama/Llama-3.2-3B-Instruct",          8192),
    ("mistral-7b-gpu",    "mistralai/Mistral-7B-Instruct-v0.3",         4096),  # VRAM limit
]

GPU_CONTEXT_LENGTHS = [2048, 8192]
GPU_BATCH_SIZES = [1, 4]
GPU_REPS = [0, 1]

# Fast validation matrix (--fast flag): 1 run only
FAST_MATRIX = [
    ("cpu", "tinyllama-1.1b", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", 1.1, 2048, 1, 0),
]

# ── Low-end machine limits (--low-end flag) ───────────────────────────────────
#
# For machines like i5-8300H (4-core, 2.3 GHz turbo-off) + GTX 1060 6 GB:
#
#   CPU path exclusions:
#   - 7B+ models: ~12–18 s/tok at 2.3 GHz → 50+ min/run, sustained thermal load
#   - 32K context: KV cache bloat + hours of inference time, throttle risk
#
#   GPU path exclusions:
#   - Llama-3.2-3B (fp16 = 6 GB, exactly GPU limit → OOM from activations)
#   - Mistral-7B (14 GB fp16, no chance)
#   Only Qwen2.5-0.5B (~1 GB fp16) is safe on a 6 GB GPU.
#
# Keeps 4 CPU models × 2 contexts × 3 batches × 3 reps = 72 runs (CPU)
# + 1 GPU model × 1 context × 2 batches × 2 reps = 4 runs (GPU)
# = 76 runs total vs 196 full-matrix runs.

LOW_END_MAX_PARAMS_B = 4.0      # exclude CPU models above this
LOW_END_MAX_CTX = 8192          # exclude context lengths above this
LOW_END_GPU_MODELS = [
    ("qwen2.5-0.5b-gpu", "Qwen/Qwen2.5-0.5B-Instruct", 2048),
]
LOW_END_GPU_CONTEXT_LENGTHS = [2048]


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class RunSpec:
    key: str
    path_type: str          # "cpu" or "gpu"
    model_id: str
    model_file: str         # GGUF filename or HF repo
    params_b: float
    ctx: int
    batch: int
    rep: int
    sample_period: int = MIN_SAMPLE_PERIOD
    region_granularity: int = REGION_GRANULARITY_BYTES
    is_warmup: bool = False
    max_runtime: int = MAX_RUN_SECONDS


@dataclass
class RunResult:
    status: str             # "done" | "failed" | "skipped"
    run_id: Optional[str] = None
    output_dir: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_s: Optional[float] = None
    peak_temp_c: float = 0.0
    throttled: bool = False
    contaminated: bool = False
    validity_reasons: list = field(default_factory=list)
    run_validity: str = "unknown"   # "clean" | "marginal" | "contaminated" | "unknown"
    error: Optional[str] = None


# ── Temperature utilities ─────────────────────────────────────────────────────

def _get_cpu_package_temp() -> float:
    """Read CPU package temperature in Celsius from sysfs."""
    return read_package_temp_c()


def _read_throttle_counts() -> dict:
    """
    Read sysfs core_throttle_count for each CPU. Returns {cpu_id: count}.
    Only increments when the CPU actually hits its thermal limit — unlike
    frequency scaling, which drops for power-management reasons at any temp.
    """
    counts = {}
    for path in pathlib.Path("/sys/devices/system/cpu").glob(
        "cpu[0-9]*/thermal_throttle/core_throttle_count"
    ):
        cpu_id = path.parts[-3]
        try:
            counts[cpu_id] = int(path.read_text())
        except OSError:
            pass
    return counts


def _new_throttle_events(before: dict, after: dict) -> int:
    """Return total new thermal throttle events between two snapshots."""
    return sum(
        max(0, after.get(cpu, 0) - before.get(cpu, 0))
        for cpu in after
    )


def wait_for_cool(target_temp: float, log_fn) -> None:
    """Block until CPU package temperature is below target_temp."""
    while True:
        temp = _get_cpu_package_temp()
        if temp <= target_temp:
            return
        log_fn(f"  Cooling... {temp:.1f}°C (target <{target_temp}°C). "
               f"Waiting {COOL_CHECK_INTERVAL}s.")
        time.sleep(COOL_CHECK_INTERVAL)


# ── State management ──────────────────────────────────────────────────────────

def _load_state() -> Dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"version": 1, "created_at": _now(), "last_updated": _now(), "runs": {}}


def _save_state(state: Dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE_FILE)
    state["last_updated"] = _now()


def _mark(state: Dict, key: str, result: RunResult) -> None:
    state["runs"][key] = {
        "status": result.status,
        "run_id": result.run_id,
        "output_dir": result.output_dir,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "duration_s": result.duration_s,
        "peak_temp_c": result.peak_temp_c,
        "throttled": result.throttled,
        "contaminated": result.contaminated,
        "run_validity": result.run_validity,
        "error": result.error,
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Run execution ─────────────────────────────────────────────────────────────

def _build_command(spec: RunSpec, models_dir: pathlib.Path, machine: str) -> List[str]:
    if spec.path_type == "cpu":
        model_path = models_dir / spec.model_file
        return [
            PYTHON,
            str(SCRIPT_PERF_MEM),
            "--model", str(model_path),
            "--model-id", spec.model_id,
            "--ctx",   str(spec.ctx),
            "--tokens", str(OUTPUT_TOKENS),
            "--batch-size", str(spec.batch),
            "--rep", str(spec.rep),
            "--machine", machine,
            "--out", str(RESULTS_DIR),
            "--sample-period", str(spec.sample_period),
            "--region-granularity", str(spec.region_granularity),
            "--num-prompts", str(PROMPTS_PER_RUN),
        ]
    else:  # gpu
        return [
            PYTHON,
            str(SCRIPT_HF_GPU),
            "--model", spec.model_file,
            "--ctx",   str(spec.ctx),
            "--tokens", str(OUTPUT_TOKENS),
            "--batch-size", str(spec.batch),
            "--rep", str(spec.rep),
            "--machine", machine,
            "--collector", "auto",
            "--out", str(RESULTS_DIR),
        ]


def run_one(spec: RunSpec, models_dir: pathlib.Path, machine: str, log_fn) -> RunResult:
    started_at = _now()
    start_time = time.time()
    start_temp = _get_cpu_package_temp()
    peak_temp = start_temp
    throttle_before = _read_throttle_counts()

    cmd = _build_command(spec, models_dir, machine)
    log_fn(f"  CMD: {' '.join(cmd)}")

    proc = None
    error = None
    returncode = -1

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            # Own session, so Ctrl+C in the terminal reaches only the runner.
            # Without this the SIGINT went to the whole foreground process
            # group: the run died with exit -2 and was recorded as failed,
            # despite _handle_sigint promising to finish it first.
            start_new_session=True,
        )

        # Read stdout on a background thread so the main loop can poll
        # temperature and enforce the timeout independently of child output.
        # Without this, blocking readline() during silent phases (model load,
        # 30–120 s) prevents temperature updates and timeout enforcement.
        line_queue: queue.Queue = queue.Queue()

        def _reader(pipe, q):
            try:
                for line in pipe:
                    q.put(line.rstrip())
            finally:
                q.put(None)  # sentinel

        reader_thread = threading.Thread(target=_reader, args=(proc.stdout, line_queue), daemon=True)
        reader_thread.start()

        output_lines = []
        reader_done = False
        while not reader_done or proc.poll() is None:
            # Drain all available lines without blocking
            while True:
                try:
                    line = line_queue.get_nowait()
                except queue.Empty:
                    break
                if line is None:
                    reader_done = True
                    break
                output_lines.append(line)
                log_fn(f"    | {line}")

            temp = _get_cpu_package_temp()
            peak_temp = max(peak_temp, temp)

            elapsed = time.time() - start_time
            if elapsed > spec.max_runtime:
                log_fn(f"  TIMEOUT after {elapsed:.0f}s — killing process")
                proc.kill()
                proc.wait()
                error = f"Timeout after {elapsed:.0f}s"
                break

            if not reader_done:
                time.sleep(0.5)

        if proc.poll() is not None:
            returncode = proc.returncode

    except Exception as exc:
        error = str(exc)
        log_fn(f"  EXCEPTION: {exc}")
        if proc and proc.poll() is None:
            proc.kill()

    elapsed = time.time() - start_time
    finished_at = _now()

    throttle_after = _read_throttle_counts()
    throttled = _new_throttle_events(throttle_before, throttle_after) > 0

    if error is None and returncode != 0:
        error = f"Non-zero exit code: {returncode}"

    status = "done" if error is None else "failed"

    # Try to extract run_id and output_dir from the output lines
    run_id = None
    output_dir = None
    for line in output_lines:
        if "Output dir" in line or "output_dir" in line:
            parts = line.split(":")
            if len(parts) > 1:
                candidate = parts[-1].strip()
                if "results" in candidate:
                    output_dir = candidate
        if "run_id" in line.lower() or "run-id" in line.lower():
            parts = line.split(":")
            if len(parts) > 1:
                run_id = parts[-1].strip()

    run_validity, validity_reasons = _resolve_run_validity(output_dir, peak_temp)
    contaminated = run_validity == "contaminated"

    return RunResult(
        status=status,
        run_id=run_id,
        output_dir=output_dir,
        started_at=started_at,
        finished_at=finished_at,
        duration_s=round(elapsed, 1),
        peak_temp_c=round(peak_temp, 1),
        throttled=throttled,
        contaminated=contaminated,
        run_validity=run_validity,
        validity_reasons=validity_reasons,
        error=error,
    )


def reclassify_runs(results_dir: pathlib.Path, state: Dict, log,
                    dry_run: bool = False) -> Dict[str, int]:
    """
    Re-derive run_validity for every exported run from its own metadata.json.

    Validity is a label over numbers the run already recorded, so correcting
    the rule never requires re-collecting a trace.  Rewrites
    thermal.run_validity and thermal.validity_failure_reasons in each
    metadata.json, the matching entries in the state file, and the
    run_validity column of manifest.csv.

    Returns a count of runs per new verdict.
    """
    metas = sorted(results_dir.glob("*/*/*/metadata.json"))
    log(f"Reclassifying {len(metas)} exported runs"
        f"{' (dry run)' if dry_run else ''} ...")

    counts: Dict[str, int] = {}
    changed = 0
    verdict_by_run_id: Dict[str, str] = {}

    for meta_path in metas:
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError) as exc:
            log(f"  SKIP {meta_path.parent.name}: {exc}")
            continue

        result = classify_from_metadata(meta)
        thermal = meta.setdefault("thermal", {})
        before = thermal.get("run_validity")
        counts[result.run_validity] = counts.get(result.run_validity, 0) + 1
        verdict_by_run_id[meta.get("run_id", meta_path.parent.name)] = result.run_validity

        if before == result.run_validity:
            continue
        changed += 1
        log(f"  {meta_path.parent.name}")
        log(f"    {before} -> {result.run_validity}")
        for reason in result.failure_reasons:
            log(f"      {reason}")
        if dry_run:
            continue
        thermal["run_validity"] = result.run_validity
        thermal["validity_failure_reasons"] = result.failure_reasons
        meta_path.write_text(json.dumps(meta, indent=2))

    if not dry_run and changed:
        _sync_state_validity(state, verdict_by_run_id, log)
        _sync_manifest_validity(results_dir / "manifest.csv", verdict_by_run_id, log)

    log(f"Reclassified: {changed} changed, "
        + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    return counts


def _sync_state_validity(state: Dict, verdict_by_run_id: Dict[str, str], log) -> None:
    """Carry the new verdicts into .experiment_state.json so --status agrees."""
    synced = 0
    for entry in state.get("runs", {}).values():
        run_id = entry.get("run_id")
        # The state file stores the run_id scraped from the child's stdout; fall
        # back to matching on the output directory name.
        verdict = verdict_by_run_id.get(run_id)
        if verdict is None and entry.get("output_dir"):
            verdict = verdict_by_run_id.get(pathlib.Path(entry["output_dir"]).name)
        if verdict is None or entry.get("run_validity") == verdict:
            continue
        entry["run_validity"] = verdict
        entry["contaminated"] = verdict == "contaminated"
        synced += 1
    if synced:
        _save_state(state)
    log(f"  state file: {synced} entries updated")


def _sync_manifest_validity(manifest_csv: pathlib.Path,
                            verdict_by_run_id: Dict[str, str], log) -> None:
    """Rewrite the run_validity column of the dataset manifest in place."""
    if not manifest_csv.exists():
        return
    with open(manifest_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or "run_validity" not in rows[0]:
        return
    updated = 0
    for row in rows:
        verdict = verdict_by_run_id.get(row.get("run_id"))
        if verdict and row.get("run_validity") != verdict:
            row["run_validity"] = verdict
            updated += 1
    if updated:
        with open(manifest_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    log(f"  manifest.csv: {updated} rows updated")


def _resolve_run_validity(output_dir: Optional[str],
                          peak_temp: float) -> "tuple[str, list]":
    """
    Prefer the classification the child run wrote into its own metadata.json
    (see demo_perf_mem.py _thermal_section()), together with the reasons that
    produced it. Falls back to a peak-temperature heuristic for runs that
    didn't write one (GPU path, failed runs, older script versions).

    Returns (verdict, reasons).
    """
    if output_dir:
        meta_path = pathlib.Path(output_dir) / "metadata.json"
        try:
            meta = json.loads(meta_path.read_text())
            thermal = meta.get("thermal", {})
            validity = thermal.get("run_validity")
            if validity:
                return validity, thermal.get("validity_failure_reasons") or []
        except (OSError, ValueError):
            pass
    if peak_temp > TEMP_CONTAMINATED_FALLBACK_C:
        return "contaminated", [f"peak_temp {peak_temp:.1f}°C > "
                                f"{TEMP_CONTAMINATED_FALLBACK_C}°C (no metadata.json)"]
    if peak_temp > TEMP_WARN_C:
        return "marginal", [f"peak_temp {peak_temp:.1f}°C > {TEMP_WARN_C}°C "
                            f"(no metadata.json)"]
    return "unknown", ["no metadata.json to classify from"]


# ── Matrix builder ─────────────────────────────────────────────────────────────

_FAST_SPEC = RunSpec(
    key="tinyllama-1.1b__ctx2048__b1__rep0__cpu",
    path_type="cpu",
    model_id="tinyllama-1.1b",
    model_file="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    params_b=1.1,
    ctx=2048,
    batch=1,
    rep=0,
    is_warmup=True,
)


def _cpu_specs(models, ctx_lengths,
               region_granularity: int = REGION_GRANULARITY_BYTES) -> List[RunSpec]:
    specs = []
    calibration = _load_calibration()
    for model_id, model_file, params_b, tok_per_s in models:
        for ctx in ctx_lengths:
            # Throughput falls with context length, so the period is per
            # (model, ctx).  Batch size does not change run length — measured
            # 395-455 s across batch 1, 4 and 8 — so it is not a factor.
            sample_period = sample_period_for(
                tok_per_s, ctx, model_id=model_id, calibration=calibration)
            max_runtime = run_timeout_for(estimate_run_seconds(tok_per_s, ctx))
            for batch in BATCH_SIZES:
                for rep in REPS:
                    specs.append(RunSpec(
                        key=f"{model_id}__ctx{ctx}__b{batch}__rep{rep}__cpu",
                        path_type="cpu",
                        model_id=model_id,
                        model_file=model_file,
                        params_b=params_b,
                        ctx=ctx,
                        batch=batch,
                        rep=rep,
                        sample_period=sample_period,
                        region_granularity=region_granularity,
                        is_warmup=(rep == 0),
                        max_runtime=max_runtime,
                    ))
    return specs


def _gpu_specs(models, ctx_lengths) -> List[RunSpec]:
    specs = []
    for model_id, hf_repo, max_ctx in models:
        for ctx in ctx_lengths:
            if ctx > max_ctx:
                continue
            for batch in GPU_BATCH_SIZES:
                for rep in GPU_REPS:
                    specs.append(RunSpec(
                        key=f"{model_id}__ctx{ctx}__b{batch}__rep{rep}__gpu",
                        path_type="gpu",
                        model_id=model_id,
                        model_file=hf_repo,
                        params_b=0.0,
                        ctx=ctx,
                        batch=batch,
                        rep=rep,
                        is_warmup=(rep == 0),
                    ))
    return specs


def build_matrix(fast: bool = False, cpu_only: bool = False, gpu_only: bool = False,
                 low_end: bool = False,
                 region_granularity: int = REGION_GRANULARITY_BYTES) -> List[RunSpec]:
    if fast:
        return [_FAST_SPEC]

    cpu_models = [(m, f, p, t) for m, f, p, t in CPU_MODELS
                  if not low_end or p <= LOW_END_MAX_PARAMS_B]
    ctx_lengths = [c for c in CONTEXT_LENGTHS
                   if not low_end or c <= LOW_END_MAX_CTX]
    gpu_models = LOW_END_GPU_MODELS if low_end else GPU_MODELS
    gpu_ctx_lengths = LOW_END_GPU_CONTEXT_LENGTHS if low_end else GPU_CONTEXT_LENGTHS

    specs = []
    if not gpu_only:
        specs += _cpu_specs(cpu_models, ctx_lengths, region_granularity)
    if not cpu_only:
        specs += _gpu_specs(gpu_models, gpu_ctx_lengths)
    return specs


# ── Logging ───────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self, log_file: pathlib.Path):
        log_file.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_file, "a", buffering=1)

    def __call__(self, msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        self._file.write(line + "\n")

    def close(self):
        self._file.close()


# ── Progress display ──────────────────────────────────────────────────────────

def print_status(state: Dict, all_specs: List[RunSpec]) -> None:
    runs = state.get("runs", {})
    total = len(all_specs)
    done = sum(1 for s in all_specs if runs.get(s.key, {}).get("status") == "done")
    failed = sum(1 for s in all_specs if runs.get(s.key, {}).get("status") == "failed")
    pending = total - done - failed
    contaminated = sum(
        1 for s in all_specs
        if runs.get(s.key, {}).get("status") == "done"
        and runs.get(s.key, {}).get("run_validity") == "contaminated"
    )
    marginal = sum(
        1 for s in all_specs
        if runs.get(s.key, {}).get("status") == "done"
        and runs.get(s.key, {}).get("run_validity") == "marginal"
    )
    warmup_done = sum(
        1 for s in all_specs
        if s.is_warmup and runs.get(s.key, {}).get("status") == "done"
    )

    print()
    print("=" * 60)
    print("  MRM-Trace Experiment Runner — Status")
    print(f"  Last updated : {state.get('last_updated', 'n/a')}")
    print("-" * 60)
    print(f"  Total runs   : {total}")
    print(f"  Done         : {done}  (warmup: {warmup_done})")
    print(f"  Failed       : {failed}")
    print(f"  Pending      : {pending}")
    print(f"  Contaminated : {contaminated}  (done but excluded from dataset)")
    print(f"  Marginal     : {marginal}  (done, thermal caveat)")
    print(f"  CPU temp now : {_get_cpu_package_temp():.1f}°C")
    print("=" * 60)

    if failed:
        print("\n  Failed runs:")
        for s in all_specs:
            r = runs.get(s.key, {})
            if r.get("status") == "failed":
                print(f"    {s.key}")
                print(f"      error: {r.get('error', 'unknown')}")

    # Per-model summary
    model_ids = []
    for s in all_specs:
        if s.model_id not in model_ids:
            model_ids.append(s.model_id)
    print("\n  Per-model:")
    print(f"  {'Model':<25} {'Done':>6} {'Pending':>8} {'Failed':>7}")
    print(f"  {'-'*25} {'-'*6} {'-'*8} {'-'*7}")
    for mid in model_ids:
        m_specs = [s for s in all_specs if s.model_id == mid]
        m_done = sum(1 for s in m_specs if runs.get(s.key, {}).get("status") == "done")
        m_fail = sum(1 for s in m_specs if runs.get(s.key, {}).get("status") == "failed")
        m_pend = len(m_specs) - m_done - m_fail
        print(f"  {mid:<25} {m_done:>6} {m_pend:>8} {m_fail:>7}")
    print()


# ── Main helpers ──────────────────────────────────────────────────────────────

_shutdown_requested = False


def _handle_sigint(sig, frame):
    global _shutdown_requested
    if _shutdown_requested:
        print("\nForce exit.")
        sys.exit(1)
    _shutdown_requested = True
    print("\n[Ctrl+C] Shutdown after current run completes. Press again to force-exit.")


def _parse_cli() -> "tuple":
    """Parse CLI arguments. Returns (args, machine_slug)."""
    parser = argparse.ArgumentParser(description="MRM-trace experiment runner")
    parser.add_argument("--status",       action="store_true", help="Show status and exit")
    parser.add_argument("--restart",      action="store_true", help="Discard state, restart from scratch")
    parser.add_argument("--retry-failed", action="store_true", dest="retry_failed", help="Retry failed runs")
    parser.add_argument("--dry-run",      action="store_true", dest="dry_run", help="Print plan without running")
    parser.add_argument("--fast",         action="store_true", help="Single validation run (TinyLlama 1 prompt)")
    parser.add_argument("--low-end",      action="store_true", dest="low_end",
                        help="Exclude models > 4B and 32K context (safe for i5-8300H + GTX 1060 6 GB)")
    parser.add_argument("--reclassify", action="store_true",
                        help="re-derive run_validity for every exported run from its "
                             "own metadata.json and exit; no run is re-collected. "
                             "Combine with --dry-run to preview.")
    parser.add_argument("--region-granularity", type=int, dest="region_granularity",
                        default=REGION_GRANULARITY_BYTES,
                        help=f"bytes per region for retention analysis "
                             f"(default {REGION_GRANULARITY_BYTES}); every run prints a "
                             f"sweep saying whether the value suited its sample density")
    parser.add_argument("--cpu-only",     action="store_true", dest="cpu_only")
    parser.add_argument("--gpu-only",     action="store_true", dest="gpu_only")
    parser.add_argument("--models-dir",   default="models", dest="models_dir",
                        help="Directory containing GGUF model files (default: models/)")
    parser.add_argument(
        "--machine", default=None,
        help="Machine identifier for results/<machine>/ subdirectory and manifest.csv "
             "(default: derived from hostname)",
    )
    args = parser.parse_args()
    return args, args.machine or machine_slug()


def _init_state(args, all_specs: List[RunSpec], log) -> Dict:
    """Load/reset state and apply --retry-failed. Returns the state dict."""
    if args.restart:
        log("--restart: discarding existing state")
        state: Dict = {"version": 1, "created_at": _now(), "last_updated": _now(), "runs": {}}
        _save_state(state)
    else:
        state = _load_state()

    if args.retry_failed:
        for key, v in list(state["runs"].items()):
            if v.get("status") == "failed":
                del state["runs"][key]
                log(f"  Reset failed run: {key}")
        _save_state(state)

    return state


def _verify_model_files(all_specs: List[RunSpec], models_dir: pathlib.Path) -> None:
    """Exit with a clear message if any required GGUF model files are missing."""
    missing = [
        str(models_dir / s.model_file)
        for s in all_specs
        if s.path_type == "cpu" and not (models_dir / s.model_file).exists()
    ]
    if not missing:
        return
    print("\nMissing model files:")
    for m in missing:
        print(f"  {m}")
    print("\nDownload them per experiment-workplan.md Section 6.1, then re-run.")
    sys.exit(1)


def _result_flag(result: RunResult) -> str:
    """Return the bracketed status tag to append to the DONE log line."""
    if result.contaminated:
        return " [CONTAMINATED — excluded from dataset]"
    if result.throttled or result.peak_temp_c > TEMP_CONTAMINATED_C:
        return " [THROTTLED — check quality]"
    if result.peak_temp_c > TEMP_WARN_C:
        return " [WARM — thermal warning, at most marginal]"
    return ""


def _trace_row_count(output_dir: Optional[str]) -> int:
    """Rows in a run's trace.parquet, from the footer — no data is read."""
    if not output_dir:
        return 0
    trace = pathlib.Path(output_dir) / "trace.parquet"
    if not trace.exists():
        return 0
    try:
        import pyarrow.parquet as pq
        return pq.ParquetFile(trace).metadata.num_rows
    except Exception:
        return 0


def _calibrate_from_run(spec: RunSpec, result: RunResult, log) -> None:
    """
    Feed a completed run's sample count back into the period calculation.

    Reports the overshoot so a drifting estimate is visible rather than silent:
    the first 8 runs of the sweep came in 35% over budget and nothing said so.
    """
    if spec.path_type != "cpu":
        return
    samples = _trace_row_count(result.output_dir)
    if samples <= 0:
        return
    ratio = samples / SAMPLE_BUDGET
    record_calibration(spec.model_id, spec.ctx, spec.sample_period, samples)
    corrected = sample_period_for(0.0, spec.ctx, model_id=spec.model_id)
    log(f"     {samples:,} samples ({ratio:.2f}x budget) at -c {spec.sample_period:,}"
        f"  ->  next {spec.model_id} ctx{spec.ctx} run uses -c {corrected:,}")


def _do_cooldown(spec: RunSpec, log) -> None:
    """Mandatory inter-run cooldown per workplan §9.3."""
    cool_time = MIN_COOL_SECONDS + (LARGE_MODEL_EXTRA_COOL if spec.params_b >= 7.0 else 0)
    log(f"   Mandatory {cool_time}s cooldown before next run...")
    time.sleep(cool_time)
    wait_for_cool(TEMP_COOL_TARGET, log)


def _run_loop(all_specs: List[RunSpec], state: Dict,
              models_dir: pathlib.Path, machine: str, log) -> None:
    """Execute all pending runs in order, respecting thermal limits and shutdown."""
    run_count = 0
    last_progress_time = time.time()

    for spec in all_specs:
        if _shutdown_requested:
            log("Shutdown requested — stopping after saving state.")
            break

        if state["runs"].get(spec.key, {}).get("status") == "done":
            continue

        if time.time() - last_progress_time > PROGRESS_INTERVAL:
            print_status(state, all_specs)
            last_progress_time = time.time()

        temp = _get_cpu_package_temp()
        log("")
        log(f"━━ [{run_count + 1}] {spec.key}")
        log(f"   Type: {spec.path_type.upper()} | Model: {spec.model_id} | "
            f"ctx={spec.ctx} | batch={spec.batch} | rep={spec.rep} | "
            f"{'WARMUP' if spec.is_warmup else 'DATA'}")
        log(f"   CPU temp: {temp:.1f}°C")

        if temp > TEMP_START_MAX:
            log(f"   Too hot ({temp:.1f}°C > {TEMP_START_MAX}°C). Waiting to cool...")
            wait_for_cool(TEMP_START_MAX, log)
            log(f"   Cool enough ({_get_cpu_package_temp():.1f}°C). Starting run.")

        state["runs"][spec.key] = {"status": "running", "started_at": _now()}
        _save_state(state)

        result = run_one(spec, models_dir, machine, log)
        _mark(state, spec.key, result)
        _save_state(state)
        run_count += 1

        if result.status == "done":
            log(f"   DONE in {result.duration_s:.0f}s, "
                f"peak {result.peak_temp_c:.1f}°C{_result_flag(result)}")
            if result.run_validity != "clean":
                for reason in result.validity_reasons:
                    log(f"     reason: {reason}")
            _calibrate_from_run(spec, result, log)
        else:
            log(f"   FAILED: {result.error}")

        if _shutdown_requested:
            break

        _do_cooldown(spec, log)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global _shutdown_requested

    args, machine = _parse_cli()

    models_dir = pathlib.Path(args.models_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGINT, _handle_sigint)

    log = Logger(LOG_FILE)
    log(f"experiment_runner.py started — pid {os.getpid()}, machine={machine}")

    all_specs = build_matrix(fast=args.fast, cpu_only=args.cpu_only,
                             gpu_only=args.gpu_only, low_end=args.low_end,
                             region_granularity=args.region_granularity)
    log(f"Matrix: {len(all_specs)} runs planned")
    cpu_specs = [s for s in all_specs if s.path_type == "cpu"]
    if cpu_specs:
        log(f"Sampling budget: {SAMPLE_BUDGET:,} samples/run | "
            f"{PROMPTS_PER_RUN} prompts x {OUTPUT_TOKENS} tokens | "
            f"--region-granularity {args.region_granularity}")
        log(f"  {'model':<16}{'ctx':>7}{'-c period':>12}{'timeout':>10}")
        seen = set()
        for s in cpu_specs:
            if (s.model_id, s.ctx) in seen:
                continue
            seen.add((s.model_id, s.ctx))
            log(f"  {s.model_id:<16}{s.ctx:>7}{s.sample_period:>12,}"
                f"{s.max_runtime / 3600:>8.1f}h")

    state = _init_state(args, all_specs, log)

    if args.reclassify:
        reclassify_runs(RESULTS_DIR, state, log, dry_run=args.dry_run)
        print_status(state, all_specs)
        log.close()
        return

    if args.status:
        print_status(state, all_specs)
        return

    if args.dry_run:
        print_status(state, all_specs)
        print("Pending runs (would execute):")
        for s in all_specs:
            if state["runs"].get(s.key, {}).get("status") != "done":
                print(f"  {s.key}")
        return

    _verify_model_files(all_specs, models_dir)
    print_status(state, all_specs)
    _run_loop(all_specs, state, models_dir, machine, log)
    print_status(state, all_specs)

    runs = state["runs"]
    done = sum(1 for s in all_specs if runs.get(s.key, {}).get("status") == "done")
    total = len(all_specs)
    if done == total:
        log("All runs complete!")
    else:
        log(f"{done}/{total} runs complete. Re-run to resume.")
    log.close()


if __name__ == "__main__":
    main()

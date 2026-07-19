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
import json
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from mrm_trace.telemetry.thermal import read_package_temp_c

# ── Configuration ─────────────────────────────────────────────────────────────

RESULTS_DIR = pathlib.Path("results")
STATE_FILE = RESULTS_DIR / ".experiment_state.json"
LOG_FILE = RESULTS_DIR / "logs" / "runner.log"

# Script paths (relative to repo root)
SCRIPT_PERF_MEM = pathlib.Path("notebooks/scripts/demo_perf_mem.py")
SCRIPT_HF_GPU = pathlib.Path("notebooks/scripts/demo_qwen_hf.py")

# Python interpreter: the current venv's python (runner must be started via venv)
PYTHON = sys.executable

# Thermal thresholds (°C)
TEMP_START_MAX = 60      # do not start a run above this
TEMP_WARN = 90           # flag run as warm
# Fallback-only: used when the child run's metadata.json (which now records real
# thermal_throttle_events) can't be read — e.g. GPU path, or a run that failed
# before writing metadata. 95°C sits just under typical consumer x86 Tjmax
# (~100-105°C), where sustained throttling is effectively guaranteed. The
# previous value (120°C) was above any real package temperature and never
# fired, silently letting every throttled run through as "not contaminated".
TEMP_CONTAMINATED_FALLBACK_C = 95
COOL_CHECK_INTERVAL = 60 # seconds between temperature checks while waiting

# Timing
MIN_COOL_SECONDS = 60          # mandatory cooldown between runs (5 min)
LARGE_MODEL_EXTRA_COOL = 60    # extra cooldown for 7B+ models (5 min)
MAX_RUN_SECONDS = 7200          # 2-hour hard timeout per run
PROGRESS_INTERVAL = 120         # re-print progress every 5 min

# Output tokens (fixed across all runs for comparability)
OUTPUT_TOKENS = 256

# ── Experiment Matrix ──────────────────────────────────────────────────────────

# CPU path: GGUF models via demo_perf_mem.py + perf mem
# Each entry: (model_id, model_file, params_b)
CPU_MODELS = [
    ("tinyllama-1.1b",  "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",       1.1),
    ("llama-3.2-1b",    "Llama-3.2-1B-Instruct-Q4_K_M.gguf",          1.0),
    ("llama-3.2-3b",    "Llama-3.2-3B-Instruct-Q4_K_M.gguf",          3.0),
    ("phi-3.5-mini",    "Phi-3.5-mini-instruct-Q4_K_M.gguf",           3.8),
    ("mistral-7b",      "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",        7.0),
    ("llama-3.1-8b",    "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",      8.0),
]

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
    run_validity: str = "unknown"   # "clean" | "marginal" | "contaminated" | "unknown"
    error: Optional[str] = None


# ── Temperature utilities ─────────────────────────────────────────────────────

def _get_cpu_package_temp() -> float:
    """Read CPU package temperature in Celsius from sysfs."""
    return read_package_temp_c()


def _is_throttled() -> bool:
    """Check whether any CPU core is running below 85% of its rated max frequency."""
    cpu_root = pathlib.Path("/sys/devices/system/cpu")
    for cpu_dir in sorted(cpu_root.glob("cpu[0-9]*"))[:8]:
        max_f = cpu_dir / "cpufreq" / "cpuinfo_max_freq"
        cur_f = cpu_dir / "cpufreq" / "scaling_cur_freq"
        try:
            if int(cur_f.read_text()) < int(max_f.read_text()) * 0.85:
                return True
        except OSError:
            continue
    return False


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

def _build_command(spec: RunSpec, models_dir: pathlib.Path) -> List[str]:
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
            "--out", str(RESULTS_DIR),
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
            "--collector", "auto",
            "--out", str(RESULTS_DIR),
        ]


def run_one(spec: RunSpec, models_dir: pathlib.Path, log_fn) -> RunResult:
    started_at = _now()
    start_time = time.time()
    start_temp = _get_cpu_package_temp()
    peak_temp = start_temp

    cmd = _build_command(spec, models_dir)
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
        )

        # Stream output while monitoring temperature and timeout
        output_lines = []
        while True:
            line = proc.stdout.readline()
            if line:
                line = line.rstrip()
                output_lines.append(line)
                log_fn(f"    | {line}")
                # Track peak temperature from our own readings
            elif proc.poll() is not None:
                break

            temp = _get_cpu_package_temp()
            peak_temp = max(peak_temp, temp)

            elapsed = time.time() - start_time
            if elapsed > spec.max_runtime:
                log_fn(f"  TIMEOUT after {elapsed:.0f}s — killing process")
                proc.kill()
                proc.wait()
                error = f"Timeout after {elapsed:.0f}s"
                break

        if proc.poll() is not None:
            returncode = proc.returncode

    except Exception as exc:
        error = str(exc)
        log_fn(f"  EXCEPTION: {exc}")
        if proc and proc.poll() is None:
            proc.kill()

    elapsed = time.time() - start_time
    finished_at = _now()

    throttled = _is_throttled()

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

    run_validity = _resolve_run_validity(output_dir, peak_temp)
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
        error=error,
    )


def _resolve_run_validity(output_dir: Optional[str], peak_temp: float) -> str:
    """
    Prefer the real thermal-throttle-based classification the child run wrote
    into its own metadata.json (see demo_perf_mem.py build_metadata()). Falls
    back to a peak-temperature heuristic for runs that didn't write it (GPU
    path, failed runs, older script versions).
    """
    if output_dir:
        meta_path = pathlib.Path(output_dir) / "metadata.json"
        try:
            meta = json.loads(meta_path.read_text())
            validity = meta.get("thermal", {}).get("run_validity")
            if validity:
                return validity
        except (OSError, ValueError):
            pass
    return "contaminated" if peak_temp > TEMP_CONTAMINATED_FALLBACK_C else "unknown"


# ── Matrix builder ─────────────────────────────────────────────────────────────

def build_matrix(fast: bool = False, cpu_only: bool = False, gpu_only: bool = False) -> List[RunSpec]:
    if fast:
        return [
            RunSpec(
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
        ]

    specs = []

    if not gpu_only:
        for model_id, model_file, params_b in CPU_MODELS:
            for ctx in CONTEXT_LENGTHS:
                for batch in BATCH_SIZES:
                    for rep in REPS:
                        key = f"{model_id}__ctx{ctx}__b{batch}__rep{rep}__cpu"
                        specs.append(RunSpec(
                            key=key,
                            path_type="cpu",
                            model_id=model_id,
                            model_file=model_file,
                            params_b=params_b,
                            ctx=ctx,
                            batch=batch,
                            rep=rep,
                            is_warmup=(rep == 0),
                            max_runtime=MAX_RUN_SECONDS,
                        ))

    if not cpu_only:
        for model_id, hf_repo, max_ctx in GPU_MODELS:
            for ctx in GPU_CONTEXT_LENGTHS:
                if ctx > max_ctx:
                    continue
                for batch in GPU_BATCH_SIZES:
                    for rep in GPU_REPS:
                        key = f"{model_id}__ctx{ctx}__b{batch}__rep{rep}__gpu"
                        specs.append(RunSpec(
                            key=key,
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
    print(f"  MRM-Trace Experiment Runner — Status")
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


# ── Main ──────────────────────────────────────────────────────────────────────

_shutdown_requested = False

def _handle_sigint(sig, frame):
    global _shutdown_requested
    if _shutdown_requested:
        print("\nForce exit.")
        sys.exit(1)
    _shutdown_requested = True
    print("\n[Ctrl+C] Shutdown after current run completes. Press again to force-exit.")


def main():
    global _shutdown_requested

    parser = argparse.ArgumentParser(description="MRM-trace experiment runner")
    parser.add_argument("--status",       action="store_true", help="Show status and exit")
    parser.add_argument("--restart",      action="store_true", help="Discard state, restart from scratch")
    parser.add_argument("--retry-failed", action="store_true", dest="retry_failed", help="Retry failed runs")
    parser.add_argument("--dry-run",      action="store_true", dest="dry_run", help="Print plan without running")
    parser.add_argument("--fast",         action="store_true", help="Single validation run (TinyLlama 1 prompt)")
    parser.add_argument("--cpu-only",     action="store_true", dest="cpu_only")
    parser.add_argument("--gpu-only",     action="store_true", dest="gpu_only")
    parser.add_argument("--models-dir",   default="models", dest="models_dir",
                        help="Directory containing GGUF model files (default: models/)")
    args = parser.parse_args()

    models_dir = pathlib.Path(args.models_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGINT, _handle_sigint)

    log = Logger(LOG_FILE)
    log(f"experiment_runner.py started — pid {os.getpid()}")

    # Build the full experiment matrix
    all_specs = build_matrix(fast=args.fast, cpu_only=args.cpu_only, gpu_only=args.gpu_only)
    log(f"Matrix: {len(all_specs)} runs planned")

    # Load or reset state
    if args.restart:
        log("--restart: discarding existing state")
        state = {"version": 1, "created_at": _now(), "last_updated": _now(), "runs": {}}
        _save_state(state)
    else:
        state = _load_state()

    # Reset failed runs if requested
    if args.retry_failed:
        for key, v in list(state["runs"].items()):
            if v.get("status") == "failed":
                del state["runs"][key]
                log(f"  Reset failed run: {key}")
        _save_state(state)

    # --status: just print and exit
    if args.status:
        print_status(state, all_specs)
        return

    # --dry-run: print the pending runs
    if args.dry_run:
        print_status(state, all_specs)
        print("Pending runs (would execute):")
        for s in all_specs:
            if state["runs"].get(s.key, {}).get("status") != "done":
                print(f"  {s.key}")
        return

    # Verify model files exist
    missing = []
    for s in all_specs:
        if s.path_type == "cpu":
            p = models_dir / s.model_file
            if not p.exists():
                missing.append(str(p))
    if missing:
        print("\nMissing model files:")
        for m in missing:
            print(f"  {m}")
        print("\nDownload them per experiment-workplan.md Section 6.1, then re-run.")
        sys.exit(1)

    # Print starting summary
    print_status(state, all_specs)

    # ── Main loop ──────────────────────────────────────────────────────────────
    run_count = 0
    last_progress_time = time.time()

    for spec in all_specs:
        if _shutdown_requested:
            log("Shutdown requested — stopping after saving state.")
            break

        existing = state["runs"].get(spec.key, {})
        if existing.get("status") == "done":
            continue  # already completed

        # Periodic progress re-print
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

        # Wait to cool if needed
        if temp > TEMP_START_MAX:
            log(f"   Too hot ({temp:.1f}°C > {TEMP_START_MAX}°C). Waiting to cool...")
            wait_for_cool(TEMP_START_MAX, log)
            log(f"   Cool enough ({_get_cpu_package_temp():.1f}°C). Starting run.")

        # Mark as running in state (allows detection of mid-run crashes)
        state["runs"][spec.key] = {"status": "running", "started_at": _now()}
        _save_state(state)

        # Execute
        result = run_one(spec, models_dir, log)
        _mark(state, spec.key, result)
        _save_state(state)
        run_count += 1

        if result.status == "done":
            flag = ""
            if result.contaminated:
                flag = " [CONTAMINATED — excluded from dataset]"
            elif result.throttled:
                flag = " [THROTTLED — check quality]"
            elif result.peak_temp_c > TEMP_WARN:
                flag = " [WARM — check quality]"
            log(f"   DONE in {result.duration_s:.0f}s, peak {result.peak_temp_c:.1f}°C{flag}")
        else:
            log(f"   FAILED: {result.error}")

        if _shutdown_requested:
            break

        # Mandatory cooldown between runs
        cool_time = MIN_COOL_SECONDS
        if spec.params_b >= 7.0:
            cool_time += LARGE_MODEL_EXTRA_COOL
        log(f"   Cooling {cool_time}s before next run...")
        time.sleep(cool_time)

    # ── Final summary ──────────────────────────────────────────────────────────
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

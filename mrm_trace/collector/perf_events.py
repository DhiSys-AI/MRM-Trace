"""
perf event configuration, shared by every collection script.

This lives in the package rather than in a demo script because the two scripts
drifted: demo_perf_mem.py gained load-dense sampling and recorded its
acquisition parameters, while demo_qwen_hf.py kept `perf mem record` with no
period and no events, and recorded none of them.  16 GPU runs in one sweep came
out with no sampling period, no event strings and no granularity, so their
sample counts could not be interpreted afterwards.

Why load-dense sampling: `perf mem record -c N` applies one period to both
events, and perf samples stores on a plain period while gating loads behind a
PEBS latency threshold.  The result is ~99.8% stores — and model weights are
mapped read-only, so no store can ever land in them.  Measured: 114 of
28,321,245 samples were inside the weight mapping.  With independent periods and
ldlat=3 the same workload gave 90% loads and 6,992 weight samples.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
from typing import List, Optional, Tuple

# perf mem's default ldlat is 30 cycles, which discards weight reads that hit in
# L1/L2 — of 114 weight samples observed under it, 87 were L1 or L2 hits.
DEFAULT_LDLAT = 3

# Stores are the uninformative majority; loads carry the weight and KV reads.
DEFAULT_STORE_MULTIPLIER = 500

_SAMPLE_RATE_PATH = pathlib.Path("/proc/sys/kernel/perf_event_max_sample_rate")
_PARANOID_PATH = pathlib.Path("/proc/sys/kernel/perf_event_paranoid")

# Below this the kernel is clamping hard enough that sample counts are not
# comparable with runs collected at the default.
SAMPLE_RATE_FLOOR = 50_000


def build_perf_events(load_period: int, ldlat: int = DEFAULT_LDLAT,
                      store_multiplier: int = DEFAULT_STORE_MULTIPLIER) -> List[str]:
    """Load and store event specs with independent periods."""
    return [
        f"cpu/mem-loads,ldlat={ldlat},period={load_period}/P",
        f"cpu/mem-stores,period={load_period * store_multiplier}/P",
    ]


def record_command(perf_bin: str, events: List[str], output: str) -> List[str]:
    """
    `perf record` invocation for the given event specs.

    `-d` is mandatory: it requests PERF_SAMPLE_ADDR.  Without it `perf script`
    reports "do not have ADDR attribute set" and emits nothing at all, which is
    an empty trace rather than an error.  `--weight` carries the load latency.
    """
    cmd = [perf_bin, "record", "-d", "--weight", "-o", output]
    for event in events:
        cmd += ["-e", event]
    return cmd


def probe_perf_events(events: List[str]) -> Tuple[bool, str]:
    """
    Check that this kernel accepts the event specs before a long run.

    The ldlat and period term syntax varies across kernel versions, so a failure
    is expected on some hosts and the caller should fall back to `perf mem
    record`.
    """
    perf_bin = shutil.which("perf")
    if not perf_bin:
        return False, "perf binary not found"
    cmd = record_command(perf_bin, events, "/dev/null") + ["--", "true"]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"probe failed to run: {exc}"
    if result.returncode != 0:
        return False, result.stderr.decode(errors="replace").strip()[:200]
    return True, "ok"


def max_sample_rate() -> Optional[int]:
    """
    Current kernel.perf_event_max_sample_rate, or None if unreadable.

    The kernel lowers this itself under dense sampling ("perf: interrupt took
    too long ... lowering") and the reduction persists for the boot, so a run
    can silently sample far below the period it asked for.  Read it before and
    after every collection.
    """
    try:
        return int(_SAMPLE_RATE_PATH.read_text().strip())
    except (OSError, ValueError):
        return None


def perf_event_paranoid() -> Optional[int]:
    """Current kernel.perf_event_paranoid, or None if unreadable."""
    try:
        return int(_PARANOID_PATH.read_text().strip())
    except (OSError, ValueError):
        return None


def achieved_rate_per_cpu(samples: Optional[int],
                          duration_s: Optional[float]) -> Optional[float]:
    """
    Samples per second per CPU actually achieved.

    Needed to tell a cap reduction that clipped the trace from one that changed
    nothing: measured, a cap fell 100,000 -> 76,000 while the run sampled at
    5,049/s/CPU — 6.6% of the cap.
    """
    cpus = os.cpu_count() or 1
    if not samples or not duration_s:
        return None
    return round(samples / duration_s / cpus, 1)


def collector_provenance(sample_period: int, events: List[str],
                         region_granularity: int, samples: Optional[int],
                         duration_s: Optional[float],
                         rate_before: Optional[int],
                         rate_after: Optional[int]) -> dict:
    """
    The acquisition parameters every run must record.

    A sample count means nothing without the period it was taken at, a
    retention figure means nothing without the granule, and neither is
    comparable across runs without the sampling cap that applied.
    """
    return {
        "sample_period": sample_period,
        "event_strings": list(events or []),
        "region_granularity_bytes": region_granularity,
        "samples_collected": samples,
        "collection_duration_s": duration_s,
        "cpu_count": os.cpu_count(),
        "achieved_sample_rate_per_cpu": achieved_rate_per_cpu(samples, duration_s),
        "perf_event_max_sample_rate_before": rate_before,
        "perf_event_max_sample_rate_after": rate_after,
    }

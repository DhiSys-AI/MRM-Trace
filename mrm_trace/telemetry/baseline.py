"""
Baseline capture — system CPU/memory snapshot taken before a run starts.

A high baseline CPU indicates the machine is not idle and that run measurements
may be contaminated by competing workloads.
"""

import time
from dataclasses import dataclass
from typing import Optional

import psutil


@dataclass
class BaselineSnapshot:
    """Snapshot of system-wide resource state at a point in time."""

    cpu_pct: float               # mean system-wide CPU utilisation over sample window
    cpu_pct_max: float           # peak during the sample window
    mem_total_mb: float          # total physical RAM (MB)
    mem_available_mb: float      # available RAM at snapshot time (MB)
    mem_used_pct: float          # used fraction (0–100)
    load_avg_1m: Optional[float] # 1-min load average (Linux/macOS only)
    timestamp_s: float           # monotonic time when snapshot was taken


def capture_baseline(
    duration_s: float = 2.0,
    poll_interval_s: float = 0.2,
) -> BaselineSnapshot:
    """
    Sample system CPU over *duration_s* seconds and return a baseline snapshot.

    Parameters
    ----------
    duration_s : float
        How long to measure baseline CPU (default 2 s).
    poll_interval_s : float
        Interval between CPU samples.
    """
    cpu_samples = []
    t_start = time.monotonic()
    t_end = t_start + duration_s

    while time.monotonic() < t_end:
        cpu_samples.append(psutil.cpu_percent(interval=None))
        time.sleep(poll_interval_s)

    if not cpu_samples:
        cpu_samples.append(psutil.cpu_percent(interval=0.1))

    mem = psutil.virtual_memory()
    load_avg: Optional[float]
    try:
        load_avg = psutil.getloadavg()[0]
    except AttributeError:
        load_avg = None

    return BaselineSnapshot(
        cpu_pct=sum(cpu_samples) / len(cpu_samples),
        cpu_pct_max=max(cpu_samples),
        mem_total_mb=mem.total / 1_048_576,
        mem_available_mb=mem.available / 1_048_576,
        mem_used_pct=mem.percent,
        load_avg_1m=load_avg,
        timestamp_s=time.monotonic(),
    )

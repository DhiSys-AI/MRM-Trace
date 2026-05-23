"""
Background-thread process monitor.

Polls psutil at a configurable interval to record RSS, CPU, I/O, and
context-switch metrics for one or more target PIDs throughout a run.
Always runs in parallel with whatever primary collector is active.
"""

import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import psutil


@dataclass
class ProcessSample:
    """One resource snapshot for a single PID at a point in time."""

    t_s: float
    pid: int
    cpu_pct: float
    rss_mb: float
    vms_mb: float
    shared_mb: float
    threads: int
    open_fds: int
    read_bytes: int
    write_bytes: int
    voluntary_ctx_switches: int
    involuntary_ctx_switches: int


class ProcessMonitor:
    """
    Collects per-process resource samples on a background daemon thread.

    Usage::

        monitor = ProcessMonitor([target_pid], poll_interval_ms=100)
        monitor.start()
        # ... inference runs ...
        samples = monitor.stop()
    """

    def __init__(self, pids: List[int], poll_interval_ms: int = 100) -> None:
        self._pids = list(pids)
        self._interval_s = poll_interval_ms / 1000.0
        self._samples: List[ProcessSample] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_mono: float = 0.0

    def start(self) -> None:
        """Start the background sampling thread."""
        self._start_mono = time.monotonic()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="process-monitor"
        )
        self._thread.start()

    def stop(self) -> List[ProcessSample]:
        """Signal the thread to stop, wait for it, and return all samples collected."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        return self.get_samples()

    def get_samples(self) -> List[ProcessSample]:
        """Return a copy of collected samples (safe to call while running)."""
        with self._lock:
            return list(self._samples)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            elapsed = time.monotonic() - self._start_mono
            for pid in self._pids:
                sample = self._sample(pid, elapsed)
                if sample is not None:
                    with self._lock:
                        self._samples.append(sample)
            self._stop_event.wait(self._interval_s)

    def _sample(self, pid: int, t_s: float) -> Optional[ProcessSample]:
        try:
            proc = psutil.Process(pid)
            mem = proc.memory_info()
            io = proc.io_counters() if hasattr(proc, "io_counters") else None
            ctx = proc.num_ctx_switches() if hasattr(proc, "num_ctx_switches") else None

            shared_mb = getattr(mem, "shared", 0) / 1_048_576
            open_fds = proc.num_fds() if hasattr(proc, "num_fds") else 0

            return ProcessSample(
                t_s=t_s,
                pid=pid,
                cpu_pct=proc.cpu_percent(),
                rss_mb=mem.rss / 1_048_576,
                vms_mb=mem.vms / 1_048_576,
                shared_mb=shared_mb,
                threads=proc.num_threads(),
                open_fds=open_fds,
                read_bytes=io.read_bytes if io else 0,
                write_bytes=io.write_bytes if io else 0,
                voluntary_ctx_switches=ctx.voluntary if ctx else 0,
                involuntary_ctx_switches=ctx.involuntary if ctx else 0,
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None

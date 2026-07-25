"""
Background monitors for CPU clock frequency and GPU memory during a run.

CpuFreqMonitor: polls psutil.cpu_freq() every 500 ms on a daemon thread.
                Gives start/min/mean/end MHz — detects frequency scaling under
                thermal throttle even when sysfs throttle counters don't fire.

GpuMonitor:     polls nvidia-smi every 2 s on a daemon thread.
                No-op (available=False, stop() returns None) if nvidia-smi is absent.
"""

import shutil
import subprocess
import threading
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CpuFreqStats:
    start_mhz: float
    min_mhz: float
    mean_mhz: float
    end_mhz: float
    n_samples: int


class CpuFreqMonitor:
    """Polls system-wide current CPU frequency throughout a run."""

    def __init__(self, poll_interval_s: float = 0.5) -> None:
        self._interval = poll_interval_s
        self._samples: List[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        freq = self._read()
        if freq is not None:
            self._samples.append(freq)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="cpu-freq-monitor"
        )
        self._thread.start()

    def stop(self) -> Optional[CpuFreqStats]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        freq = self._read()
        if freq is not None:
            self._samples.append(freq)
        if not self._samples:
            return None
        return CpuFreqStats(
            start_mhz=self._samples[0],
            min_mhz=min(self._samples),
            mean_mhz=sum(self._samples) / len(self._samples),
            end_mhz=self._samples[-1],
            n_samples=len(self._samples),
        )

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            freq = self._read()
            if freq is not None:
                self._samples.append(freq)

    @staticmethod
    def _read() -> Optional[float]:
        try:
            import psutil
            f = psutil.cpu_freq()
            return f.current if f else None
        except Exception:
            return None


@dataclass
class GpuStats:
    name: str
    vram_total_mb: int
    vram_start_used_mb: int
    vram_peak_used_mb: int
    vram_end_used_mb: int
    gpu_util_pct_mean: float
    n_samples: int


class GpuMonitor:
    """
    Polls GPU memory and utilisation via nvidia-smi throughout a run.

    No-op when nvidia-smi is not installed — stop() returns None.
    """

    def __init__(self, poll_interval_s: float = 2.0) -> None:
        self._interval = poll_interval_s
        self._vram_used: List[int] = []
        self._util: List[float] = []
        self._name = ""
        self._vram_total = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._available = bool(shutil.which("nvidia-smi"))

    @property
    def available(self) -> bool:
        return self._available

    def start(self) -> None:
        if not self._available:
            return
        sample = self._query()
        if sample:
            self._name, self._vram_total, used, util = sample
            self._vram_used.append(used)
            self._util.append(util)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="gpu-monitor"
        )
        self._thread.start()

    def stop(self) -> Optional[GpuStats]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10.0)
        if not self._vram_used:
            return None
        sample = self._query()
        if sample:
            self._vram_used.append(sample[2])
            self._util.append(sample[3])
        return GpuStats(
            name=self._name,
            vram_total_mb=self._vram_total,
            vram_start_used_mb=self._vram_used[0],
            vram_peak_used_mb=max(self._vram_used),
            vram_end_used_mb=self._vram_used[-1],
            gpu_util_pct_mean=sum(self._util) / len(self._util) if self._util else 0.0,
            n_samples=len(self._vram_used),
        )

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            sample = self._query()
            if sample:
                self._vram_used.append(sample[2])
                self._util.append(sample[3])

    @staticmethod
    def _query() -> "Optional[tuple]":
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                timeout=5,
                stderr=subprocess.DEVNULL,
            ).decode().strip().splitlines()[0]
            parts = [p.strip() for p in out.split(",")]
            if len(parts) < 4:
                return None
            return parts[0], int(parts[1]), int(parts[2]), float(parts[3])
        except Exception:
            return None

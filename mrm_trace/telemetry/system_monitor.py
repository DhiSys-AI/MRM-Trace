"""
Background monitors for CPU clock frequency and GPU memory during a run.

CpuFreqMonitor:  polls psutil.cpu_freq() every 500 ms on a daemon thread.
                 Gives start/min/mean/end MHz — detects frequency scaling under
                 thermal throttle even when sysfs throttle counters don't fire.

read_cpu_freq_limits: one-shot sysfs read of the governor-configured ceiling
                 (scaling_max_freq), hardware ceiling (cpuinfo_max_freq), floor,
                 and active governor.  Call once before the run; these are static
                 per-machine values needed to interpret the runtime CpuFreqStats.
                 E.g. if scaling_max_mhz=2300 and cpu_freq_min_mhz=2280, that is
                 not throttling — the machine was simply capped at 2.3 GHz.

GpuMonitor:      polls nvidia-smi every 2 s on a daemon thread.
                 No-op (available=False, stop() returns None) if nvidia-smi is absent.
"""

import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class CpuFreqLimits:
    """Static per-machine CPU frequency policy — read once before the run."""

    hw_max_mhz: Optional[float]       # cpuinfo_max_freq  (hardware ceiling)
    scaling_max_mhz: Optional[float]  # governor-set upper bound
    scaling_min_mhz: Optional[float]  # governor-set lower bound
    governor: Optional[str]           # e.g. "performance", "powersave"


def read_cpu_freq_limits(cpu: int = 0) -> CpuFreqLimits:
    """
    Read CPU frequency policy from sysfs for the given logical CPU.

    Returns None fields on non-Linux systems or when sysfs is unavailable
    (containers, WSL without cpufreq, etc.).
    """
    base = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq")

    def _read_khz(name: str) -> Optional[float]:
        try:
            return int((base / name).read_text().strip()) / 1_000.0
        except Exception:
            return None

    def _read_str(name: str) -> Optional[str]:
        try:
            return (base / name).read_text().strip()
        except Exception:
            return None

    return CpuFreqLimits(
        hw_max_mhz=_read_khz("cpuinfo_max_freq"),
        scaling_max_mhz=_read_khz("scaling_max_freq"),
        scaling_min_mhz=_read_khz("scaling_min_freq"),
        governor=_read_str("scaling_governor"),
    )


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


def gpu_section(gpu_stats, machine_ctx: dict | None = None) -> dict:
    """
    Build the metadata.json ``gpu`` section from a GpuStats and machine context.

    Shared by the CPU and GPU demo scripts so the two cannot drift.  Power draw
    is only published when ``validate_machine`` judged it plausible; when it did
    not, the reason is published instead so a reader can tell a missing sensor
    from a lying one.  The enforced limit is always recorded when known, because
    it is what makes the draw auditable.

    Returns {} when there is no GPU information at all.
    """
    section: dict = {}
    if gpu_stats is not None:
        section.update({
            "name": gpu_stats.name,
            "vram_total_mb": gpu_stats.vram_total_mb,
            "vram_start_used_mb": gpu_stats.vram_start_used_mb,
            "vram_peak_used_mb": gpu_stats.vram_peak_used_mb,
            "vram_end_used_mb": gpu_stats.vram_end_used_mb,
            "gpu_util_pct_mean": round(gpu_stats.gpu_util_pct_mean, 1),
        })
    if machine_ctx is not None:
        if machine_ctx.get("gpu_power_draw_w") is not None:
            section["power_draw_w"] = machine_ctx["gpu_power_draw_w"]
        if machine_ctx.get("gpu_power_limit_w") is not None:
            section["power_limit_w"] = machine_ctx["gpu_power_limit_w"]
        status = machine_ctx.get("gpu_power_draw_status")
        if status is not None and status != "ok":
            section["power_draw_status"] = status
    return section

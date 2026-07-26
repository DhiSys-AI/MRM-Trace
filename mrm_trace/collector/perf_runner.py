"""
perf-mem-based trace collector (primary collector).

Shells out to ``perf mem record`` to capture hardware-PMU-assisted memory
access traces, then converts the result to a scriptable text form via
``perf script``.  Commands are always built as lists — shell=True is never
used and user-supplied config values are never interpolated into shell strings.
"""

import shutil
import signal
import subprocess
from pathlib import Path
from typing import List, Optional

from .base import BaseCollector, CollectorMeta

# Pass no default events so perf mem auto-selects the right PMU events for the CPU.
# Explicit events like "cpu/mem-loads,ldlat=20/P" are Intel-specific and require ldlat
# support in the running kernel — not universally available even on Intel hardware.
# Leaving events empty lets perf mem pick cpu/mem-loads/ and cpu/mem-stores/ (or
# their architecture equivalents) automatically.
_DEFAULT_EVENTS: List[str] = []
_DEFAULT_SAMPLE_PERIOD = 4000
_PERF_SCRIPT_TIMEOUT_S = 120


def _find_perf() -> Optional[str]:
    """Return the absolute path to the ``perf`` binary, or None."""
    return shutil.which("perf")


class PerfMemRunner(BaseCollector):
    """
    Primary collector: wraps ``perf mem record`` + ``perf script``.

    Requires a PMU-capable Linux kernel (WSL2 included) and typically
    root or ``/proc/sys/kernel/perf_event_paranoid <= 1``.
    """

    def __init__(
        self,
        events: Optional[List[str]] = None,
        sample_period: int = _DEFAULT_SAMPLE_PERIOD,
    ) -> None:
        self._events = list(events or _DEFAULT_EVENTS)
        self._sample_period = sample_period
        self._proc: Optional[subprocess.Popen] = None
        self._output_dir: Optional[Path] = None
        self._warnings: List[str] = []

    @property
    def name(self) -> str:
        return "perf_mem"

    def is_available(self) -> bool:
        return _find_perf() is not None

    def start(self, target_pids: List[int], output_dir: Path) -> None:
        """
        Launch ``perf mem record`` targeting *target_pids*.

        Raises RuntimeError if perf is not found.  The process runs
        asynchronously; call :meth:`stop` to terminate it and collect metadata.
        """
        perf_bin = _find_perf()
        if not perf_bin:
            raise RuntimeError(
                "perf binary not found. Install linux-perf or run on a "
                "PMU-capable Linux system."
            )

        self._output_dir = output_dir
        self._warnings = []
        pid_str = ",".join(str(p) for p in target_pids)
        output_file = str(output_dir / "perf.data")

        cmd: List[str] = [perf_bin, "mem", "record", "-c", str(self._sample_period), "-p", pid_str]
        for ev in self._events:
            cmd += ["-e", ev]
        cmd += ["-o", output_file]

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def stop(self) -> CollectorMeta:
        """
        Send SIGINT to perf, wait for it to flush, run ``perf script``,
        and return a :class:`CollectorMeta` record.
        """
        warnings: List[str] = list(self._warnings)

        if self._proc is not None:
            try:
                self._proc.send_signal(signal.SIGINT)
                _, stderr_bytes = self._proc.communicate(timeout=30)
                if self._proc.returncode not in (0, -signal.SIGINT, 130):
                    stderr = (stderr_bytes or b"").decode(errors="replace")
                    warnings.append(
                        f"perf exited {self._proc.returncode}: {stderr[:200]}"
                    )
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
                warnings.append("perf did not exit cleanly; killed after timeout")
            except ProcessLookupError:
                pass
            self._proc = None

        output_file: Optional[str] = None
        output_size = 0
        perf_script_ok: Optional[bool] = None

        if self._output_dir is not None:
            perf_data = self._output_dir / "perf.data"
            if perf_data.exists():
                output_size = perf_data.stat().st_size
                output_file = str(perf_data)
                perf_script_ok = self._run_perf_script(perf_data, warnings)
            else:
                warnings.append(
                    "perf.data not produced — PMU unavailable or permission denied"
                )

        return CollectorMeta(
            collector_type="perf_mem",
            available=True,
            used=True,
            event_strings=list(self._events),
            sample_period=self._sample_period,
            output_file=output_file,
            output_file_size_bytes=output_size,
            parse_success=perf_script_ok,
            warnings=warnings,
        )

    def _run_perf_script(self, perf_data: Path, warnings: List[str]) -> bool:
        perf_bin = _find_perf()
        if not perf_bin:
            return False
        script_out = perf_data.parent / "perf_script.txt"
        try:
            with open(script_out, "w", encoding="utf-8") as fh:
                result = subprocess.run(
                    [
                        perf_bin, "script",
                        "-i", str(perf_data),
                        # Explicit field list keeps output single-line-per-sample.
                        # Default output includes multi-line callchains which the
                        # parser cannot handle, and some perf versions emit nothing
                        # without this flag for mem events.
                        # cpu attribute is not set for all perf mem event types;
                        # omitting it avoids "Cannot print 'cpu' field" exit 255.
                        # weight = PEBS load-latency cycles; data_src encodes
                        # cache hierarchy level and NUMA node.
                        "-F", "comm,pid,time,event,addr,sym,dso,weight,data_src",
                    ],
                    stdout=fh,
                    stderr=subprocess.PIPE,
                    timeout=_PERF_SCRIPT_TIMEOUT_S,
                )
            if result.returncode != 0:
                warnings.append(
                    f"perf script exited {result.returncode}: "
                    f"{result.stderr.decode(errors='replace')[:200]}"
                )
                return False
            return True
        except Exception as exc:
            warnings.append(f"perf script failed: {exc}")
            return False

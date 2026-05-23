"""
Memray-based allocation tracer (secondary/fallback collector).

Captures Python-level (and optionally native C/C++) memory allocations via
Bloomberg Memray.  Used when hardware PMU is unavailable or as a complement
to perf-mem for Python-level allocation visibility.
"""

import importlib
from pathlib import Path
from typing import Any, List, Optional

from .base import BaseCollector, CollectorMeta


def _try_import_memray() -> Optional[Any]:
    """Return the memray module, or None if not installed."""
    try:
        return importlib.import_module("memray")
    except ImportError:
        return None


class MemrayRunner(BaseCollector):
    """
    Secondary collector: wraps ``memray.Tracker``.

    Tracks allocations in the current Python process; suitable for
    environments without hardware PMU support.
    """

    def __init__(self, native_traces: bool = True) -> None:
        self._native_traces = native_traces
        self._output_file: Optional[Path] = None
        self._tracker: Optional[Any] = None

    @property
    def name(self) -> str:
        return "memray"

    def is_available(self) -> bool:
        return _try_import_memray() is not None

    def start(self, target_pids: List[int], output_dir: Path) -> None:
        """
        Begin memray tracking.

        *target_pids* is accepted for interface compatibility but memray
        always traces the current Python process.
        """
        memray = _try_import_memray()
        if memray is None:
            raise RuntimeError(
                "memray is not installed. Install with: pip install memray"
            )
        self._output_file = output_dir / "memray.bin"
        self._tracker = memray.Tracker(
            str(self._output_file),
            native_traces=self._native_traces,
        )
        self._tracker.__enter__()

    def stop(self) -> CollectorMeta:
        """Exit the memray Tracker context and return a metadata record."""
        if self._tracker is not None:
            self._tracker.__exit__(None, None, None)
            self._tracker = None

        output_size = 0
        output_file_str: Optional[str] = None
        if self._output_file is not None and self._output_file.exists():
            output_size = self._output_file.stat().st_size
            output_file_str = str(self._output_file)

        return CollectorMeta(
            collector_type="memray",
            available=True,
            used=True,
            output_file=output_file_str,
            output_file_size_bytes=output_size,
            parse_success=output_size > 0,
            extra={"native_traces": self._native_traces},
        )

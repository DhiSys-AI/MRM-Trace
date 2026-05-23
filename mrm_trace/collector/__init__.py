from .artifact_manager import ArtifactManager, DiskSpaceError
from .base import BaseCollector, CollectorMeta
from .memray_runner import MemrayRunner
from .perf_runner import PerfMemRunner
from .process_monitor import ProcessMonitor, ProcessSample

__all__ = [
    "ArtifactManager",
    "BaseCollector",
    "CollectorMeta",
    "DiskSpaceError",
    "MemrayRunner",
    "PerfMemRunner",
    "ProcessMonitor",
    "ProcessSample",
    "select_collector",
]


def select_collector(prefer: str = "perf_mem") -> BaseCollector:
    """
    Return the best available primary/secondary collector.

    Preference order: perf_mem → memray.  Pass ``prefer="memray"`` to
    skip perf even when it is available.

    Raises RuntimeError if no collector can be selected.
    """
    candidates: list
    if prefer == "memray":
        candidates = [MemrayRunner()]
    else:
        candidates = [PerfMemRunner(), MemrayRunner()]

    for collector in candidates:
        if collector.is_available():
            return collector

    raise RuntimeError(
        "No primary collector available. "
        "Install linux-perf (PMU) or 'pip install memray'."
    )

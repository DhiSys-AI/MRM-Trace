"""Abstract base class and shared data types for trace collectors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CollectorMeta:
    """Metadata recorded about a collector run — written to metadata.json."""

    collector_type: str
    available: bool
    used: bool
    fallback_reason: Optional[str] = None
    event_strings: List[str] = field(default_factory=list)
    sample_period: int = 0
    output_file: Optional[str] = None
    output_file_size_bytes: int = 0
    parse_success: Optional[bool] = None
    dropped_samples: Optional[int] = None
    coverage_estimate: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseCollector(ABC):
    """Interface that all primary/secondary trace collectors must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short collector identifier, e.g. 'perf_mem' or 'memray'."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this collector can run in the current environment."""

    @abstractmethod
    def start(self, target_pids: List[int], output_dir: Path) -> None:
        """Begin collection targeting the given PIDs, writing artifacts to output_dir."""

    @abstractmethod
    def stop(self) -> CollectorMeta:
        """Stop collection and return a metadata record describing the run."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional


class PhaseEvent(str, Enum):
    MODEL_LOAD_START = "model_load_start"
    MODEL_LOAD_END = "model_load_end"
    GENERATION_START = "generation_start"
    GENERATION_END = "generation_end"
    TEARDOWN_START = "teardown_start"
    TEARDOWN_END = "teardown_end"
    KV_CACHE_ALLOC = "kv_cache_alloc"
    TOKEN_EMITTED = "token_emitted"


@dataclass
class PhaseRecord:
    event: PhaseEvent
    timestamp_utc: str      # ISO 8601 wall-clock
    wall_ns: int            # time.time_ns() — wall clock for cross-run correlation
    mono_ns: int            # time.monotonic_ns() — monotonic for intra-run deltas
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    output_tokens: int
    duration_s: float
    tokens_per_second: float


@dataclass
class EngineRuntimeMetadata:
    engine_name: str
    engine_version: Optional[str]
    model_id: str
    model_path: Optional[str]
    context_length: int
    batch_size: int
    dtype: str
    quantisation: Optional[str]
    extra: Dict[str, Any] = field(default_factory=dict)


PhaseCallback = Callable[[PhaseRecord], None]


class BaseEngine(ABC):
    """
    Abstract base for all inference engine wrappers.

    Subclasses implement the inference lifecycle and emit phase events so that
    the collector can align memory traces to workload phases (weight_load,
    generation, teardown).  Engine wrappers must not embed analysis logic.
    """

    def __init__(self, model_config: Any, run_config: Any) -> None:
        self._model_config = model_config
        self._run_config = run_config
        self._phase_records: List[PhaseRecord] = []
        self._callbacks: List[PhaseCallback] = []

    # ── Abstract interface ─────────────────────────────────────────────────────

    @abstractmethod
    def prepare(self) -> None:
        """Check prerequisites (model file exists, binaries reachable)."""

    @abstractmethod
    def load_model(self) -> None:
        """Load model into memory. Must emit MODEL_LOAD_START and MODEL_LOAD_END."""

    @abstractmethod
    def warmup(self, prompt: str) -> None:
        """Run a discarded warmup inference pass to warm caches."""

    @abstractmethod
    def generate(self, prompt: str) -> GenerationResult:
        """Run one inference pass. Must emit GENERATION_START and GENERATION_END."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release model resources. Must emit TEARDOWN_START and TEARDOWN_END."""

    @abstractmethod
    def get_target_pids(self) -> List[int]:
        """Return PIDs of the engine process and any workers for collector attachment."""

    @abstractmethod
    def get_runtime_metadata(self) -> EngineRuntimeMetadata:
        """Return engine and model metadata for inclusion in metadata.json."""

    # ── Phase event machinery ──────────────────────────────────────────────────

    def register_phase_callback(self, callback: PhaseCallback) -> None:
        """Register a callback that fires synchronously on every phase event."""
        self._callbacks.append(callback)

    def emit_phase(
        self,
        event: PhaseEvent,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PhaseRecord:
        from datetime import datetime, timezone

        record = PhaseRecord(
            event=event,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            wall_ns=time.time_ns(),
            mono_ns=time.monotonic_ns(),
            metadata=metadata or {},
        )
        self._phase_records.append(record)
        for cb in self._callbacks:
            cb(record)
        return record

    def get_phase_records(self) -> List[PhaseRecord]:
        """Return all phase records emitted so far, in emission order."""
        return list(self._phase_records)

    # ── Context manager ────────────────────────────────────────────────────────

    @contextmanager
    def timed_session(self) -> Generator["BaseEngine", None, None]:
        """Run the full model lifecycle as a context manager."""
        self.prepare()
        self.load_model()
        try:
            yield self
        finally:
            self.shutdown()

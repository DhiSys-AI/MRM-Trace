"""
Tier-3 validity tests for Phase 2.

Phase records are the alignment primitive that links memory trace timestamps to
workload phases. If phase events are emitted out of order, or if timestamps are
non-monotonic, every downstream analysis that computes per-phase metrics will
produce incorrect results.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mrm_trace.engines.base import BaseEngine, GenerationResult, PhaseEvent, PhaseRecord
from mrm_trace.engines.llamacpp import LlamaCppEngine


# ── Minimal engine for validity testing ───────────────────────────────────────

class _StrictEngine(BaseEngine):
    """Engine that emits all required phase events faithfully."""

    def prepare(self): pass

    def load_model(self):
        self.emit_phase(PhaseEvent.MODEL_LOAD_START)
        self.emit_phase(PhaseEvent.MODEL_LOAD_END)

    def warmup(self, prompt=""): self.generate(prompt)

    def generate(self, prompt=""):
        self.emit_phase(PhaseEvent.GENERATION_START)
        self.emit_phase(PhaseEvent.GENERATION_END)
        return GenerationResult("out", 1, 2, 0.01, 200.0)

    def shutdown(self):
        self.emit_phase(PhaseEvent.TEARDOWN_START)
        self.emit_phase(PhaseEvent.TEARDOWN_END)

    def get_target_pids(self): return [1]
    def get_runtime_metadata(self): return None  # type: ignore[return-value]


REQUIRED_ORDER = [
    PhaseEvent.MODEL_LOAD_START,
    PhaseEvent.MODEL_LOAD_END,
    PhaseEvent.GENERATION_START,
    PhaseEvent.GENERATION_END,
    PhaseEvent.TEARDOWN_START,
    PhaseEvent.TEARDOWN_END,
]


def test_all_required_phase_events_present():
    """All six mandatory phase events must be present after a full run."""
    eng = _StrictEngine(None, None)
    with eng.timed_session() as s:
        s.generate()
    events = {r.event for r in eng.get_phase_records()}
    for ev in REQUIRED_ORDER:
        assert ev in events, (
            f"Required phase event '{ev}' was not emitted. "
            "The collector cannot align traces to this phase."
        )


def test_phase_events_strictly_ordered():
    """
    Phase events must appear in the canonical order.
    Out-of-order events produce incorrect phase labels on trace rows.
    """
    eng = _StrictEngine(None, None)
    with eng.timed_session() as s:
        s.generate()
    events = [r.event for r in eng.get_phase_records()]
    for i in range(len(REQUIRED_ORDER) - 1):
        a, b = REQUIRED_ORDER[i], REQUIRED_ORDER[i + 1]
        ia, ib = events.index(a), events.index(b)
        assert ia < ib, (
            f"Phase event '{a}' (index {ia}) must appear before '{b}' (index {ib}). "
            "This ordering invariant is required for correct trace phase labelling."
        )


def test_monotonic_wall_timestamps():
    """
    Wall-clock timestamps in phase records must be non-decreasing.
    Decreasing timestamps would corrupt time-based retention calculations.
    """
    eng = _StrictEngine(None, None)
    with eng.timed_session() as s:
        s.generate()
    records = eng.get_phase_records()
    for i in range(1, len(records)):
        assert records[i].wall_ns >= records[i - 1].wall_ns, (
            f"wall_ns decreased between records {i-1} and {i}: "
            f"{records[i-1].wall_ns} → {records[i].wall_ns}. "
            "Non-monotonic timestamps corrupt retention duration calculations."
        )


def test_monotonic_mono_timestamps():
    """Monotonic timestamps must strictly increase (no clock resets)."""
    eng = _StrictEngine(None, None)
    with eng.timed_session() as s:
        s.generate()
    records = eng.get_phase_records()
    for i in range(1, len(records)):
        assert records[i].mono_ns >= records[i - 1].mono_ns, (
            f"mono_ns decreased between records {i-1} and {i}"
        )


def test_model_load_pair_correct_ordering():
    """MODEL_LOAD_START must always precede MODEL_LOAD_END."""
    eng = _StrictEngine(None, None)
    eng.load_model()
    events = [r.event for r in eng.get_phase_records()]
    assert events.index(PhaseEvent.MODEL_LOAD_START) < events.index(PhaseEvent.MODEL_LOAD_END)


def test_generation_pair_correct_ordering():
    """GENERATION_START must always precede GENERATION_END."""
    eng = _StrictEngine(None, None)
    eng.load_model()
    eng.generate()
    events = [r.event for r in eng.get_phase_records()]
    assert events.index(PhaseEvent.GENERATION_START) < events.index(PhaseEvent.GENERATION_END)


def test_teardown_pair_correct_ordering():
    """TEARDOWN_START must always precede TEARDOWN_END."""
    eng = _StrictEngine(None, None)
    eng.shutdown()
    events = [r.event for r in eng.get_phase_records()]
    assert events.index(PhaseEvent.TEARDOWN_START) < events.index(PhaseEvent.TEARDOWN_END)


def test_phase_records_have_iso_timestamps():
    """All phase records must have ISO 8601 timestamp strings."""
    eng = _StrictEngine(None, None)
    with eng.timed_session() as s:
        s.generate()
    for rec in eng.get_phase_records():
        assert isinstance(rec.timestamp_utc, str)
        assert "T" in rec.timestamp_utc, (
            f"timestamp_utc '{rec.timestamp_utc}' is not ISO 8601"
        )


def test_multiple_generate_calls_each_emit_pair():
    """Each call to generate() must emit exactly one START and one END."""
    eng = _StrictEngine(None, None)
    eng.load_model()
    eng.generate()
    eng.generate()
    events = [r.event for r in eng.get_phase_records()]
    assert events.count(PhaseEvent.GENERATION_START) == 2
    assert events.count(PhaseEvent.GENERATION_END) == 2


def test_callbacks_see_same_records_as_get_phase_records():
    """Callbacks must receive exactly the same records stored internally."""
    received = []
    eng = _StrictEngine(None, None)
    eng.register_phase_callback(received.append)
    with eng.timed_session() as s:
        s.generate()
    stored = eng.get_phase_records()
    assert len(received) == len(stored)
    for cb_rec, stored_rec in zip(received, stored):
        assert cb_rec.event == stored_rec.event
        assert cb_rec.wall_ns == stored_rec.wall_ns


def test_llamacpp_all_phase_events_emitted(tmp_path):
    """LlamaCppEngine must emit all six required phase events during a full run."""
    model = tmp_path / "model.gguf"
    model.touch()
    mock_mod = MagicMock()
    mock_mod.__version__ = "0.2.90"
    mock_llm = MagicMock(return_value={
        "choices": [{"text": "out", "finish_reason": "length"}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 4},
    })
    mock_mod.Llama = MagicMock(return_value=mock_llm)
    eng = LlamaCppEngine(
        SimpleNamespace(id="m", engine="llamacpp", path=str(model), hf_repo=None,
                        family="", params_billions=None, quantisation=None, dtype="float16"),
        SimpleNamespace(run_id="v_001", context_length=512, batch_size=1, output_tokens=16),
    )
    eng._use_bindings = True
    with patch("mrm_trace.engines.llamacpp._try_import_llama_cpp", return_value=mock_mod):
        with eng.timed_session() as s:
            s.generate("test prompt")
    events = {r.event for r in eng.get_phase_records()}
    missing = [ev for ev in REQUIRED_ORDER if ev not in events]
    assert not missing, f"LlamaCppEngine did not emit: {missing}"

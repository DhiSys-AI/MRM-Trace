"""Unit tests for the BaseEngine abstract class and phase event machinery."""

import pytest

from mrm_trace.engines.base import (
    BaseEngine,
    EngineRuntimeMetadata,
    GenerationResult,
    PhaseEvent,
    PhaseRecord,
)


# ── Minimal concrete subclass for testing ─────────────────────────────────────

class _FakeEngine(BaseEngine):
    """Minimal concrete engine used to test the base class contract."""

    def __init__(self, model_config=None, run_config=None):
        super().__init__(model_config, run_config)
        self._prepared = False
        self._loaded = False
        self._shut = False

    def prepare(self) -> None:
        self._prepared = True

    def load_model(self) -> None:
        self.emit_phase(PhaseEvent.MODEL_LOAD_START)
        self._loaded = True
        self.emit_phase(PhaseEvent.MODEL_LOAD_END)

    def warmup(self, prompt: str = "Hello") -> None:
        self.generate(prompt)

    def generate(self, prompt: str) -> GenerationResult:
        self.emit_phase(PhaseEvent.GENERATION_START)
        result = GenerationResult(
            text="fake output",
            prompt_tokens=5,
            output_tokens=10,
            duration_s=0.1,
            tokens_per_second=100.0,
        )
        self.emit_phase(PhaseEvent.GENERATION_END)
        return result

    def shutdown(self) -> None:
        self.emit_phase(PhaseEvent.TEARDOWN_START)
        self._shut = True
        self.emit_phase(PhaseEvent.TEARDOWN_END)

    def get_target_pids(self):
        return [12345]

    def get_runtime_metadata(self) -> EngineRuntimeMetadata:
        return EngineRuntimeMetadata(
            engine_name="fake",
            engine_version="0.0.1",
            model_id="fake-model",
            model_path="/fake/path",
            context_length=2048,
            batch_size=1,
            dtype="float16",
            quantisation=None,
        )


# ── BaseEngine is abstract ─────────────────────────────────────────────────────

def test_base_engine_is_abstract():
    with pytest.raises(TypeError):
        BaseEngine(None, None)  # type: ignore[abstract]


def test_concrete_subclass_instantiates():
    eng = _FakeEngine()
    assert eng is not None


# ── Phase event recording ──────────────────────────────────────────────────────

def test_emit_phase_returns_record():
    eng = _FakeEngine()
    rec = eng.emit_phase(PhaseEvent.MODEL_LOAD_START)
    assert isinstance(rec, PhaseRecord)
    assert rec.event == PhaseEvent.MODEL_LOAD_START


def test_phase_record_has_timestamps():
    eng = _FakeEngine()
    rec = eng.emit_phase(PhaseEvent.GENERATION_START)
    assert isinstance(rec.wall_ns, int)
    assert rec.wall_ns > 0
    assert isinstance(rec.mono_ns, int)
    assert rec.mono_ns > 0


def test_phase_record_has_utc_string():
    eng = _FakeEngine()
    rec = eng.emit_phase(PhaseEvent.GENERATION_START)
    assert "T" in rec.timestamp_utc  # ISO 8601


def test_phase_record_metadata_stored():
    eng = _FakeEngine()
    rec = eng.emit_phase(PhaseEvent.GENERATION_START, {"prompt_len": 42})
    assert rec.metadata["prompt_len"] == 42


def test_get_phase_records_returns_all():
    eng = _FakeEngine()
    eng.load_model()
    records = eng.get_phase_records()
    events = [r.event for r in records]
    assert PhaseEvent.MODEL_LOAD_START in events
    assert PhaseEvent.MODEL_LOAD_END in events


def test_get_phase_records_in_order():
    eng = _FakeEngine()
    eng.load_model()
    eng.generate("hello")
    records = eng.get_phase_records()
    events = [r.event for r in records]
    assert events.index(PhaseEvent.MODEL_LOAD_START) < events.index(PhaseEvent.MODEL_LOAD_END)
    assert events.index(PhaseEvent.GENERATION_START) < events.index(PhaseEvent.GENERATION_END)
    assert events.index(PhaseEvent.MODEL_LOAD_END) < events.index(PhaseEvent.GENERATION_START)


def test_get_phase_records_returns_copy():
    eng = _FakeEngine()
    eng.emit_phase(PhaseEvent.MODEL_LOAD_START)
    records = eng.get_phase_records()
    records.clear()
    assert len(eng.get_phase_records()) == 1


def test_monotonic_timestamps_increase():
    eng = _FakeEngine()
    eng.load_model()
    records = eng.get_phase_records()
    assert records[1].mono_ns >= records[0].mono_ns


# ── Callbacks ─────────────────────────────────────────────────────────────────

def test_register_callback_fires_on_emit():
    received = []
    eng = _FakeEngine()
    eng.register_phase_callback(received.append)
    eng.emit_phase(PhaseEvent.GENERATION_START)
    assert len(received) == 1
    assert received[0].event == PhaseEvent.GENERATION_START


def test_multiple_callbacks_all_fire():
    a, b = [], []
    eng = _FakeEngine()
    eng.register_phase_callback(a.append)
    eng.register_phase_callback(b.append)
    eng.emit_phase(PhaseEvent.MODEL_LOAD_START)
    assert len(a) == 1
    assert len(b) == 1


def test_callback_receives_all_events_from_full_run():
    received = []
    eng = _FakeEngine()
    eng.register_phase_callback(received.append)
    eng.load_model()
    eng.generate("test")
    eng.shutdown()
    event_types = [r.event for r in received]
    assert PhaseEvent.MODEL_LOAD_START in event_types
    assert PhaseEvent.MODEL_LOAD_END in event_types
    assert PhaseEvent.GENERATION_START in event_types
    assert PhaseEvent.GENERATION_END in event_types
    assert PhaseEvent.TEARDOWN_START in event_types
    assert PhaseEvent.TEARDOWN_END in event_types


# ── Context manager ────────────────────────────────────────────────────────────

def test_timed_session_calls_prepare_load_shutdown():
    eng = _FakeEngine()
    with eng.timed_session() as session:
        assert eng._prepared
        assert eng._loaded
        session.generate("hello")
    assert eng._shut


def test_timed_session_shuts_down_on_exception():
    eng = _FakeEngine()
    with pytest.raises(RuntimeError):
        with eng.timed_session():
            raise RuntimeError("simulated error")
    assert eng._shut


def test_timed_session_yields_engine():
    eng = _FakeEngine()
    with eng.timed_session() as session:
        assert session is eng


# ── Interface contract ─────────────────────────────────────────────────────────

def test_get_target_pids_returns_list():
    eng = _FakeEngine()
    pids = eng.get_target_pids()
    assert isinstance(pids, list)
    assert all(isinstance(p, int) for p in pids)


def test_get_runtime_metadata_fields():
    eng = _FakeEngine()
    meta = eng.get_runtime_metadata()
    assert meta.engine_name == "fake"
    assert isinstance(meta.context_length, int)
    assert isinstance(meta.batch_size, int)
    assert isinstance(meta.extra, dict)


def test_generate_returns_generation_result():
    eng = _FakeEngine()
    result = eng.generate("hello")
    assert isinstance(result, GenerationResult)
    assert isinstance(result.text, str)
    assert isinstance(result.prompt_tokens, int)
    assert isinstance(result.output_tokens, int)
    assert result.duration_s >= 0
    assert result.tokens_per_second >= 0


# ── PhaseEvent enum ────────────────────────────────────────────────────────────

def test_phase_event_required_values():
    required = {
        "model_load_start", "model_load_end",
        "generation_start", "generation_end",
        "teardown_start", "teardown_end",
    }
    actual = {e.value for e in PhaseEvent}
    assert required.issubset(actual)


def test_phase_event_is_string_enum():
    assert isinstance(PhaseEvent.MODEL_LOAD_START, str)
    assert PhaseEvent.GENERATION_START == "generation_start"

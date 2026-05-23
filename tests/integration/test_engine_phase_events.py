"""
Integration tests for engine phase event contracts.

Verifies that:
- All six required phase events are emitted in the correct order.
- Callbacks receive every event.
- Phase records carry the fields needed by the collector.
- The engine factory routes to the correct class.
- Runtime metadata has the fields needed by metadata.json.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mrm_trace.engines import PhaseEvent, create_engine
from mrm_trace.engines.base import GenerationResult
from mrm_trace.engines.llamacpp import LlamaCppEngine
from mrm_trace.engines.vllm import VLLMEngine
from mrm_trace.config.schema import EngineType


REQUIRED_EVENTS_ORDERED = [
    PhaseEvent.MODEL_LOAD_START,
    PhaseEvent.MODEL_LOAD_END,
    PhaseEvent.GENERATION_START,
    PhaseEvent.GENERATION_END,
    PhaseEvent.TEARDOWN_START,
    PhaseEvent.TEARDOWN_END,
]


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _llama_model_config(path):
    return SimpleNamespace(
        id="llama3-q4", engine=EngineType.LLAMACPP,
        path=path, hf_repo=None, family="llama3",
        params_billions=8, quantisation="Q4_K_M", dtype="float16",
    )


def _vllm_model_config():
    return SimpleNamespace(
        id="llama3-vllm", engine=EngineType.VLLM,
        path=None, hf_repo="org/llama3", family="llama3",
        params_billions=8, quantisation=None, dtype="float16",
    )


def _run_config():
    return SimpleNamespace(
        run_id="integ_001", context_length=512, batch_size=1, output_tokens=32,
    )


def _mock_llama_output():
    return {
        "choices": [{"text": "Integration test output", "finish_reason": "length"}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 8},
    }


def _mock_vllm():
    token_ids = list(range(8))
    output_obj = MagicMock()
    output_obj.text = "vLLM integration output"
    output_obj.token_ids = token_ids
    request_output = MagicMock()
    request_output.prompt_token_ids = [0, 1, 2, 3]
    request_output.outputs = [output_obj]
    mock_llm_inst = MagicMock()
    mock_llm_inst.generate.return_value = [request_output]
    mock_vllm_mod = MagicMock()
    mock_vllm_mod.__version__ = "0.5.3"
    mock_vllm_mod.LLM = MagicMock(return_value=mock_llm_inst)
    mock_vllm_mod.SamplingParams = MagicMock(return_value=MagicMock())
    return mock_vllm_mod


# ── LlamaCpp phase event integration ─────────────────────────────────────────

class TestLlamaCppPhaseEvents:

    def _make_engine(self, model_path: str) -> LlamaCppEngine:
        mock_mod = MagicMock()
        mock_mod.__version__ = "0.2.90"
        mock_llm = MagicMock(return_value=_mock_llama_output())
        mock_mod.Llama = MagicMock(return_value=mock_llm)
        eng = LlamaCppEngine(_llama_model_config(model_path), _run_config())
        eng._use_bindings = True
        eng._llm_module = mock_mod
        return eng, mock_mod

    def test_all_six_events_emitted(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.touch()
        eng, mock_mod = self._make_engine(str(model))
        with patch("mrm_trace.engines.llamacpp._try_import_llama_cpp", return_value=mock_mod):
            with eng.timed_session() as s:
                s.generate("test")
        events = [r.event for r in eng.get_phase_records()]
        for ev in REQUIRED_EVENTS_ORDERED:
            assert ev in events, f"Missing phase event: {ev}"

    def test_events_in_correct_order(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.touch()
        eng, mock_mod = self._make_engine(str(model))
        with patch("mrm_trace.engines.llamacpp._try_import_llama_cpp", return_value=mock_mod):
            with eng.timed_session() as s:
                s.generate("test")
        events = [r.event for r in eng.get_phase_records()]
        for i in range(len(REQUIRED_EVENTS_ORDERED) - 1):
            a, b = REQUIRED_EVENTS_ORDERED[i], REQUIRED_EVENTS_ORDERED[i + 1]
            assert events.index(a) < events.index(b), f"{a} must precede {b}"

    def test_callbacks_receive_all_events(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.touch()
        eng, mock_mod = self._make_engine(str(model))
        received = []
        eng.register_phase_callback(received.append)
        with patch("mrm_trace.engines.llamacpp._try_import_llama_cpp", return_value=mock_mod):
            with eng.timed_session() as s:
                s.generate("test")
        cb_events = [r.event for r in received]
        for ev in REQUIRED_EVENTS_ORDERED:
            assert ev in cb_events

    def test_phase_records_have_timestamps(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.touch()
        eng, mock_mod = self._make_engine(str(model))
        with patch("mrm_trace.engines.llamacpp._try_import_llama_cpp", return_value=mock_mod):
            with eng.timed_session() as s:
                s.generate("test")
        for rec in eng.get_phase_records():
            assert rec.wall_ns > 0
            assert rec.mono_ns > 0
            assert "T" in rec.timestamp_utc

    def test_runtime_metadata_has_required_fields(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.touch()
        eng, mock_mod = self._make_engine(str(model))
        with patch("mrm_trace.engines.llamacpp._try_import_llama_cpp", return_value=mock_mod):
            eng.prepare()
            eng.load_model()
        meta = eng.get_runtime_metadata()
        assert meta.engine_name
        assert meta.model_id
        assert meta.context_length > 0
        assert meta.batch_size > 0
        assert meta.dtype
        assert isinstance(meta.extra, dict)


# ── vLLM phase event integration ──────────────────────────────────────────────

class TestVLLMPhaseEvents:

    def test_all_six_events_emitted(self):
        eng = VLLMEngine(_vllm_model_config(), _run_config())
        mock_vllm_mod = _mock_vllm()
        with patch.dict(sys.modules, {"vllm": mock_vllm_mod}):
            with patch("mrm_trace.engines.vllm._discover_worker_pids", return_value=[]):
                with eng.timed_session() as s:
                    s.generate("test")
        events = [r.event for r in eng.get_phase_records()]
        for ev in REQUIRED_EVENTS_ORDERED:
            assert ev in events, f"Missing event: {ev}"

    def test_events_in_correct_order(self):
        eng = VLLMEngine(_vllm_model_config(), _run_config())
        mock_vllm_mod = _mock_vllm()
        with patch.dict(sys.modules, {"vllm": mock_vllm_mod}):
            with patch("mrm_trace.engines.vllm._discover_worker_pids", return_value=[]):
                with eng.timed_session() as s:
                    s.generate("test")
        events = [r.event for r in eng.get_phase_records()]
        for i in range(len(REQUIRED_EVENTS_ORDERED) - 1):
            a, b = REQUIRED_EVENTS_ORDERED[i], REQUIRED_EVENTS_ORDERED[i + 1]
            assert events.index(a) < events.index(b), f"{a} must precede {b}"

    def test_generation_result_fields(self):
        eng = VLLMEngine(_vllm_model_config(), _run_config())
        mock_vllm_mod = _mock_vllm()
        with patch.dict(sys.modules, {"vllm": mock_vllm_mod}):
            with patch("mrm_trace.engines.vllm._discover_worker_pids", return_value=[]):
                eng.load_model()
                result = eng.generate("Hello")
        assert isinstance(result, GenerationResult)
        assert result.output_tokens > 0
        assert result.duration_s >= 0


# ── Engine factory ─────────────────────────────────────────────────────────────

def test_create_engine_returns_llamacpp(tmp_path):
    cfg = SimpleNamespace(engine=EngineType.LLAMACPP, path=str(tmp_path / "m.gguf"),
                          id="m", hf_repo=None, family="", params_billions=None,
                          quantisation=None, dtype="float16")
    run = _run_config()
    eng = create_engine(cfg, run)
    assert isinstance(eng, LlamaCppEngine)


def test_create_engine_returns_vllm():
    cfg = SimpleNamespace(engine=EngineType.VLLM, path=None, id="m",
                          hf_repo="org/model", family="", params_billions=None,
                          quantisation=None, dtype="float16")
    run = _run_config()
    eng = create_engine(cfg, run)
    assert isinstance(eng, VLLMEngine)


def test_create_engine_raises_for_unknown():
    cfg = SimpleNamespace(engine="unknown_engine")
    with pytest.raises((ValueError, AttributeError)):
        create_engine(cfg, _run_config())

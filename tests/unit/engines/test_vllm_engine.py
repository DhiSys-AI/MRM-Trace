"""
Unit tests for VLLMEngine.

vllm and torch are mocked throughout — no GPU or installed vllm required.
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mrm_trace.engines.base import GenerationResult, PhaseEvent
from mrm_trace.engines.vllm import VLLMEngine


# ── Helpers ────────────────────────────────────────────────────────────────────

def _model_config(hf_repo="org/llama3-8b", path=None, model_id="llama3-8b-vllm"):
    return SimpleNamespace(
        id=model_id,
        engine="vllm",
        path=path,
        hf_repo=hf_repo,
        family="llama3",
        params_billions=8,
        quantisation=None,
        dtype="float16",
    )


def _run_config(context_length=2048, batch_size=1, output_tokens=64):
    return SimpleNamespace(
        run_id="vllm_test_001",
        context_length=context_length,
        batch_size=batch_size,
        output_tokens=output_tokens,
    )


def _make_vllm_mocks(output_text="vLLM generated text", prompt_tokens=6, output_tokens=12):
    """Return a (vllm_module, SamplingParams_cls) mock pair."""
    token_ids = list(range(output_tokens))
    output_obj = MagicMock()
    output_obj.text = output_text
    output_obj.token_ids = token_ids

    request_output = MagicMock()
    request_output.prompt_token_ids = list(range(prompt_tokens))
    request_output.outputs = [output_obj]

    mock_llm_inst = MagicMock()
    mock_llm_inst.generate.return_value = [request_output]

    mock_sampling_params = MagicMock()

    mock_vllm = MagicMock()
    mock_vllm.__version__ = "0.5.3"
    mock_vllm.LLM = MagicMock(return_value=mock_llm_inst)
    mock_vllm.SamplingParams = MagicMock(return_value=mock_sampling_params)

    return mock_vllm, mock_llm_inst


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestVLLMEngine:

    def _make_engine(self, **kwargs) -> VLLMEngine:
        return VLLMEngine(_model_config(**kwargs), _run_config())

    def test_prepare_raises_when_vllm_not_installed(self):
        eng = self._make_engine()
        with patch.dict(sys.modules, {"vllm": None}):
            with pytest.raises(RuntimeError, match="vllm"):
                eng.prepare()

    def test_prepare_succeeds_when_vllm_available(self):
        eng = self._make_engine()
        mock_vllm, _ = _make_vllm_mocks()
        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            eng.prepare()  # should not raise

    def test_prepare_raises_when_no_model_ref(self):
        eng = VLLMEngine(_model_config(hf_repo=None, path=None), _run_config())
        mock_vllm, _ = _make_vllm_mocks()
        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            with pytest.raises(ValueError, match="hf_repo.*path"):
                eng.prepare()

    def test_model_ref_prefers_hf_repo(self):
        eng = self._make_engine(hf_repo="org/model", path="/local/path")
        assert eng._model_ref == "org/model"

    def test_model_ref_falls_back_to_path(self):
        eng = self._make_engine(hf_repo=None, path="/local/path")
        assert eng._model_ref == "/local/path"

    def test_load_model_emits_phase_events(self):
        eng = self._make_engine()
        mock_vllm, _ = _make_vllm_mocks()
        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            with patch("mrm_trace.engines.vllm._discover_worker_pids", return_value=[]):
                eng.load_model()
        events = [r.event for r in eng.get_phase_records()]
        assert PhaseEvent.MODEL_LOAD_START in events
        assert PhaseEvent.MODEL_LOAD_END in events

    def test_load_model_start_metadata(self):
        eng = self._make_engine()
        mock_vllm, _ = _make_vllm_mocks()
        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            with patch("mrm_trace.engines.vllm._discover_worker_pids", return_value=[]):
                eng.load_model()
        start_rec = next(r for r in eng.get_phase_records() if r.event == PhaseEvent.MODEL_LOAD_START)
        assert "model_ref" in start_rec.metadata

    def test_generate_returns_correct_result(self):
        eng = self._make_engine()
        mock_vllm, _ = _make_vllm_mocks(output_text="Hello from vLLM", prompt_tokens=4, output_tokens=8)
        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            with patch("mrm_trace.engines.vllm._discover_worker_pids", return_value=[]):
                eng.load_model()
                result = eng.generate("Test prompt")
        assert isinstance(result, GenerationResult)
        assert result.text == "Hello from vLLM"
        assert result.prompt_tokens == 4
        assert result.output_tokens == 8
        assert result.duration_s >= 0

    def test_generate_emits_phase_events(self):
        eng = self._make_engine()
        mock_vllm, _ = _make_vllm_mocks()
        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            with patch("mrm_trace.engines.vllm._discover_worker_pids", return_value=[]):
                eng.load_model()
                eng.generate("prompt")
        events = [r.event for r in eng.get_phase_records()]
        assert PhaseEvent.GENERATION_START in events
        assert PhaseEvent.GENERATION_END in events

    def test_generate_phase_order(self):
        eng = self._make_engine()
        mock_vllm, _ = _make_vllm_mocks()
        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            with patch("mrm_trace.engines.vllm._discover_worker_pids", return_value=[]):
                eng.load_model()
                eng.generate("test")
        events = [r.event for r in eng.get_phase_records()]
        assert events.index(PhaseEvent.MODEL_LOAD_END) < events.index(PhaseEvent.GENERATION_START)
        assert events.index(PhaseEvent.GENERATION_START) < events.index(PhaseEvent.GENERATION_END)

    def test_generate_end_metadata_has_output_tokens(self):
        eng = self._make_engine()
        mock_vllm, _ = _make_vllm_mocks(output_tokens=12)
        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            with patch("mrm_trace.engines.vllm._discover_worker_pids", return_value=[]):
                eng.load_model()
                eng.generate("test")
        end_rec = next(r for r in eng.get_phase_records() if r.event == PhaseEvent.GENERATION_END)
        assert end_rec.metadata["output_tokens"] == 12

    def test_shutdown_emits_teardown_events(self):
        eng = self._make_engine()
        mock_vllm, _ = _make_vllm_mocks()
        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            with patch("mrm_trace.engines.vllm._discover_worker_pids", return_value=[]):
                eng.load_model()
                eng.shutdown()
        events = [r.event for r in eng.get_phase_records()]
        assert PhaseEvent.TEARDOWN_START in events
        assert PhaseEvent.TEARDOWN_END in events

    def test_shutdown_clears_llm_reference(self):
        eng = self._make_engine()
        mock_vllm, _ = _make_vllm_mocks()
        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            with patch("mrm_trace.engines.vllm._discover_worker_pids", return_value=[]):
                eng.load_model()
                assert eng._llm is not None
                eng.shutdown()
        assert eng._llm is None

    def test_get_target_pids_includes_current_pid(self):
        eng = self._make_engine()
        mock_vllm, _ = _make_vllm_mocks()
        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            with patch("mrm_trace.engines.vllm._discover_worker_pids", return_value=[99999]):
                eng.load_model()
        pids = eng.get_target_pids()
        assert os.getpid() in pids

    def test_get_target_pids_includes_worker_pids(self):
        eng = self._make_engine()
        mock_vllm, _ = _make_vllm_mocks()
        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            with patch("mrm_trace.engines.vllm._discover_worker_pids", return_value=[11111, 22222]):
                eng.load_model()
        pids = eng.get_target_pids()
        assert 11111 in pids
        assert 22222 in pids

    def test_get_runtime_metadata_structure(self):
        eng = self._make_engine()
        mock_vllm, _ = _make_vllm_mocks()
        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            with patch("mrm_trace.engines.vllm._discover_worker_pids", return_value=[]):
                eng.load_model()
        meta = eng.get_runtime_metadata()
        assert meta.engine_name == "vllm"
        assert meta.model_id == "llama3-8b-vllm"
        assert meta.model_path == "org/llama3-8b"
        assert meta.dtype == "float16"
        assert "torch_version" in meta.extra
        assert "cuda_version" in meta.extra
        assert "worker_pids" in meta.extra

    def test_warmup_emits_generation_events(self):
        eng = self._make_engine()
        mock_vllm, _ = _make_vllm_mocks()
        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            with patch("mrm_trace.engines.vllm._discover_worker_pids", return_value=[]):
                eng.load_model()
                eng.warmup("warmup")
        events = [r.event for r in eng.get_phase_records()]
        assert PhaseEvent.GENERATION_START in events

    def test_full_lifecycle_via_timed_session(self):
        eng = self._make_engine()
        mock_vllm, _ = _make_vllm_mocks()
        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            with patch("mrm_trace.engines.vllm._discover_worker_pids", return_value=[]):
                with eng.timed_session() as session:
                    result = session.generate("Hello vLLM")
        assert isinstance(result, GenerationResult)
        events = [r.event for r in eng.get_phase_records()]
        assert PhaseEvent.TEARDOWN_END in events

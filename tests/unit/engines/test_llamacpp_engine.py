"""
Unit tests for LlamaCppEngine.

Both modes (Python bindings and CLI) are tested via mocks — no actual model
files or llama-cpp-python installation required.
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mrm_trace.engines.base import GenerationResult, PhaseEvent
from mrm_trace.engines.llamacpp import LlamaCppEngine


# ── Helpers ────────────────────────────────────────────────────────────────────

def _model_config(path="/fake/model.gguf", model_id="llama3-q4"):
    return SimpleNamespace(
        id=model_id,
        engine="llamacpp",
        path=path,
        hf_repo=None,
        family="llama3",
        params_billions=8,
        quantisation="Q4_K_M",
        dtype="float16",
    )


def _run_config(context_length=2048, batch_size=1, output_tokens=128):
    return SimpleNamespace(
        run_id="test_run_001",
        context_length=context_length,
        batch_size=batch_size,
        output_tokens=output_tokens,
    )


def _mock_llama_module(output_text="Hello world", prompt_tokens=5, completion_tokens=10):
    """Return a mock llama_cpp module whose Llama class produces canned output."""
    mock_output = {
        "choices": [{"text": output_text, "finish_reason": "length"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }
    mock_llm = MagicMock()
    mock_llm.return_value = mock_output
    mock_module = MagicMock()
    mock_module.__version__ = "0.2.90"
    mock_module.Llama = MagicMock(return_value=mock_llm)
    return mock_module, mock_llm


# ── Bindings mode tests ────────────────────────────────────────────────────────

class TestLlamaCppBindingsMode:
    """Tests where llama-cpp-python bindings are available (mocked)."""

    def _make_engine(self, path="/fake/model.gguf") -> LlamaCppEngine:
        eng = LlamaCppEngine(_model_config(path=path), _run_config())
        eng._use_bindings = True
        return eng

    def test_prepare_raises_file_not_found(self, tmp_path):
        eng = self._make_engine(path=str(tmp_path / "missing.gguf"))
        with pytest.raises(FileNotFoundError, match="not found"):
            eng.prepare()

    def test_prepare_succeeds_when_file_exists(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.touch()
        mock_mod, _ = _mock_llama_module()
        eng = self._make_engine(path=str(model))
        with patch("mrm_trace.engines.llamacpp._try_import_llama_cpp", return_value=mock_mod):
            eng.prepare()  # should not raise

    def test_load_model_emits_phase_events(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.touch()
        mock_mod, _ = _mock_llama_module()
        eng = self._make_engine(path=str(model))
        with patch("mrm_trace.engines.llamacpp._try_import_llama_cpp", return_value=mock_mod):
            eng.load_model()
        events = [r.event for r in eng.get_phase_records()]
        assert PhaseEvent.MODEL_LOAD_START in events
        assert PhaseEvent.MODEL_LOAD_END in events

    def test_load_model_sets_pid(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.touch()
        mock_mod, _ = _mock_llama_module()
        eng = self._make_engine(path=str(model))
        with patch("mrm_trace.engines.llamacpp._try_import_llama_cpp", return_value=mock_mod):
            eng.load_model()
        assert eng._pid == os.getpid()

    def test_generate_returns_correct_result(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.touch()
        mock_mod, mock_llm_inst = _mock_llama_module(
            output_text="Generated text", prompt_tokens=8, completion_tokens=15
        )
        eng = self._make_engine(path=str(model))
        with patch("mrm_trace.engines.llamacpp._try_import_llama_cpp", return_value=mock_mod):
            eng.load_model()
            result = eng.generate("Test prompt")

        assert isinstance(result, GenerationResult)
        assert result.text == "Generated text"
        assert result.prompt_tokens == 8
        assert result.output_tokens == 15
        assert result.duration_s >= 0
        assert result.tokens_per_second >= 0

    def test_generate_emits_phase_events(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.touch()
        mock_mod, _ = _mock_llama_module()
        eng = self._make_engine(path=str(model))
        with patch("mrm_trace.engines.llamacpp._try_import_llama_cpp", return_value=mock_mod):
            eng.load_model()
            eng.generate("prompt")
        events = [r.event for r in eng.get_phase_records()]
        assert PhaseEvent.GENERATION_START in events
        assert PhaseEvent.GENERATION_END in events

    def test_generate_phase_order(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.touch()
        mock_mod, _ = _mock_llama_module()
        eng = self._make_engine(path=str(model))
        with patch("mrm_trace.engines.llamacpp._try_import_llama_cpp", return_value=mock_mod):
            eng.load_model()
            eng.generate("test")
        records = eng.get_phase_records()
        events = [r.event for r in records]
        assert events.index(PhaseEvent.MODEL_LOAD_START) < events.index(PhaseEvent.MODEL_LOAD_END)
        assert events.index(PhaseEvent.MODEL_LOAD_END) < events.index(PhaseEvent.GENERATION_START)
        assert events.index(PhaseEvent.GENERATION_START) < events.index(PhaseEvent.GENERATION_END)

    def test_shutdown_emits_teardown_events(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.touch()
        mock_mod, _ = _mock_llama_module()
        eng = self._make_engine(path=str(model))
        with patch("mrm_trace.engines.llamacpp._try_import_llama_cpp", return_value=mock_mod):
            eng.load_model()
            eng.shutdown()
        events = [r.event for r in eng.get_phase_records()]
        assert PhaseEvent.TEARDOWN_START in events
        assert PhaseEvent.TEARDOWN_END in events

    def test_shutdown_clears_llm(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.touch()
        mock_mod, _ = _mock_llama_module()
        eng = self._make_engine(path=str(model))
        with patch("mrm_trace.engines.llamacpp._try_import_llama_cpp", return_value=mock_mod):
            eng.load_model()
            assert eng._llm is not None
            eng.shutdown()
        assert eng._llm is None

    def test_get_target_pids_after_load(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.touch()
        mock_mod, _ = _mock_llama_module()
        eng = self._make_engine(path=str(model))
        with patch("mrm_trace.engines.llamacpp._try_import_llama_cpp", return_value=mock_mod):
            eng.load_model()
        pids = eng.get_target_pids()
        assert isinstance(pids, list)
        assert len(pids) == 1
        assert pids[0] == os.getpid()

    def test_get_target_pids_empty_before_load(self):
        eng = self._make_engine()
        assert eng.get_target_pids() == []

    def test_get_runtime_metadata_structure(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.touch()
        mock_mod, _ = _mock_llama_module()
        eng = self._make_engine(path=str(model))
        with patch("mrm_trace.engines.llamacpp._try_import_llama_cpp", return_value=mock_mod):
            eng.load_model()
        meta = eng.get_runtime_metadata()
        assert meta.engine_name == "llamacpp"
        assert meta.model_id == "llama3-q4"
        assert meta.context_length == 2048
        assert meta.batch_size == 1
        assert meta.dtype == "float16"
        assert meta.quantisation == "Q4_K_M"
        assert meta.extra["use_python_bindings"] is True

    def test_warmup_calls_generate(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.touch()
        mock_mod, _ = _mock_llama_module()
        eng = self._make_engine(path=str(model))
        with patch("mrm_trace.engines.llamacpp._try_import_llama_cpp", return_value=mock_mod):
            eng.load_model()
            eng.warmup("warmup prompt")
        events = [r.event for r in eng.get_phase_records()]
        assert PhaseEvent.GENERATION_START in events

    def test_full_lifecycle_via_timed_session(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.touch()
        mock_mod, _ = _mock_llama_module()
        eng = self._make_engine(path=str(model))
        with patch("mrm_trace.engines.llamacpp._try_import_llama_cpp", return_value=mock_mod):
            with eng.timed_session() as session:
                result = session.generate("Hello")
        assert isinstance(result, GenerationResult)
        events = [r.event for r in eng.get_phase_records()]
        assert PhaseEvent.TEARDOWN_END in events


# ── CLI fallback mode tests ────────────────────────────────────────────────────

class TestLlamaCppCLIMode:
    """Tests where llama-cpp-python is absent and CLI fallback is used."""

    def _make_cli_engine(self, path="/fake/model.gguf") -> LlamaCppEngine:
        eng = LlamaCppEngine(_model_config(path=path), _run_config())
        eng._use_bindings = False
        eng._cli_path = "/usr/bin/llama-cli"
        return eng

    def test_prepare_raises_when_no_cli(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.touch()
        eng = LlamaCppEngine(_model_config(path=str(model)), _run_config())
        eng._use_bindings = False
        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="llama-cli"):
                eng.prepare()

    def test_load_model_emits_events_cli(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.touch()
        eng = self._make_cli_engine(path=str(model))
        eng.load_model()
        events = [r.event for r in eng.get_phase_records()]
        assert PhaseEvent.MODEL_LOAD_START in events
        assert PhaseEvent.MODEL_LOAD_END in events

    def test_generate_via_cli_success(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.touch()
        eng = self._make_cli_engine(path=str(model))
        eng.load_model()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "CLI generated text"
        with patch("subprocess.run", return_value=mock_result):
            result = eng.generate("a prompt")
        assert result.text == "CLI generated text"

    def test_generate_via_cli_failure_raises(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.touch()
        eng = self._make_cli_engine(path=str(model))
        eng.load_model()
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "out of memory"
        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="llama-cli"):
                eng.generate("prompt")

    def test_runtime_metadata_cli_mode(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.touch()
        eng = self._make_cli_engine(path=str(model))
        eng.load_model()
        meta = eng.get_runtime_metadata()
        assert meta.extra["use_python_bindings"] is False
        assert meta.extra["cli_path"] == "/usr/bin/llama-cli"

from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import Any, List, Optional, Tuple

from mrm_trace.engines.base import (
    BaseEngine,
    EngineRuntimeMetadata,
    GenerationResult,
    PhaseEvent,
)
from mrm_trace.utils.logging import get_logger

_LOG = get_logger(__name__)


def _try_import_llama_cpp() -> Optional[Any]:
    try:
        import llama_cpp
        return llama_cpp
    except ImportError:
        return None


def _llama_cpp_version(mod: Any) -> Optional[str]:
    return getattr(mod, "__version__", None)


class LlamaCppEngine(BaseEngine):
    """
    Inference engine wrapper for llama.cpp.

    Primary mode: llama-cpp-python Python bindings (in-process, PID = os.getpid()).
    Fallback mode: llama-cli subprocess (requires binary on PATH).

    The engine emits phase events before and after model load, generation,
    and teardown so the collector can align traces to workload phases.
    """

    def __init__(self, model_config: Any, run_config: Any) -> None:
        super().__init__(model_config, run_config)
        self._llm: Optional[Any] = None
        self._proc: Optional[subprocess.Popen] = None  # type: ignore[type-arg]
        self._pid: Optional[int] = None
        self._version: Optional[str] = None
        self._build_flags: List[str] = []
        self._cli_path: Optional[str] = None
        # Determine mode at construction so tests can override _use_bindings
        self._use_bindings: bool = _try_import_llama_cpp() is not None

    # ── BaseEngine ─────────────────────────────────────────────────────────────

    def prepare(self) -> None:
        """Verify model file exists and a llama.cpp backend is available."""
        from pathlib import Path

        path = Path(self._model_config.path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        if self._use_bindings:
            mod = _try_import_llama_cpp()
            self._version = _llama_cpp_version(mod)
            _LOG.info(
                "llama-cpp-python available (version=%s), using Python bindings",
                self._version,
            )
        else:
            cli = shutil.which("llama-cli") or shutil.which("llama")
            if cli is None:
                raise RuntimeError(
                    "Neither llama-cpp-python nor llama-cli found in PATH. "
                    "Install llama-cpp-python or put llama-cli on PATH."
                )
            self._cli_path = cli
            _LOG.info("Using CLI backend at: %s", cli)

    def load_model(self) -> None:
        self.emit_phase(
            PhaseEvent.MODEL_LOAD_START,
            {"model_path": self._model_config.path, "context_length": self._run_config.context_length},
        )
        if self._use_bindings:
            self._load_via_bindings()
        else:
            # CLI mode: model loads at generate() time; PID is current process.
            self._pid = os.getpid()
            _LOG.info("CLI mode: model will be loaded per generate() invocation")
        self.emit_phase(
            PhaseEvent.MODEL_LOAD_END,
            {"version": self._version, "mode": "bindings" if self._use_bindings else "cli"},
        )

    def _load_via_bindings(self) -> None:
        llama_cpp = _try_import_llama_cpp()
        _LOG.info(
            "Loading GGUF model: %s (n_ctx=%d)",
            self._model_config.path,
            self._run_config.context_length,
        )
        self._llm = llama_cpp.Llama(
            model_path=self._model_config.path,
            n_ctx=self._run_config.context_length,
            verbose=False,
        )
        self._pid = os.getpid()
        self._version = _llama_cpp_version(llama_cpp)

    def warmup(self, prompt: str = "Hello") -> None:
        _LOG.info("Warmup run (result discarded)")
        self.generate(prompt)

    def generate(self, prompt: str) -> GenerationResult:
        self.emit_phase(PhaseEvent.GENERATION_START, {"prompt_len": len(prompt)})
        start = time.perf_counter()

        if self._use_bindings:
            text, prompt_tokens, output_tokens = self._generate_via_bindings(prompt)
        else:
            text, prompt_tokens, output_tokens = self._generate_via_cli(prompt)

        elapsed = time.perf_counter() - start
        result = GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            duration_s=elapsed,
            tokens_per_second=output_tokens / elapsed if elapsed > 0 else 0.0,
        )
        self.emit_phase(
            PhaseEvent.GENERATION_END,
            {"output_tokens": result.output_tokens, "duration_s": result.duration_s,
             "tokens_per_second": result.tokens_per_second},
        )
        return result

    def _generate_via_bindings(self, prompt: str) -> Tuple[str, int, int]:
        # Reset KV cache so each generate() call uses a fresh context.
        # Without this, prompt tokens accumulate across calls and the context
        # window fills up, causing later prompts to produce 0 output tokens.
        if hasattr(self._llm, "reset"):
            self._llm.reset()

        output = self._llm(
            prompt,
            max_tokens=self._run_config.output_tokens,
            stop=None,
            echo=False,
        )
        choice = output["choices"][0]
        usage = output.get("usage", {})
        # completion_tokens can be 0 when llama-cpp-python omits usage; fall back
        # to counting the generated text's tokens via the tokeniser.
        text = choice["text"]
        completion_tokens = usage.get("completion_tokens") or 0
        if completion_tokens == 0 and text:
            try:
                completion_tokens = len(self._llm.tokenize(text.encode()))
            except Exception:
                completion_tokens = max(1, len(text.split()))
        return (
            text,
            usage.get("prompt_tokens", 0),
            completion_tokens,
        )

    def _generate_via_cli(self, prompt: str) -> Tuple[str, int, int]:
        cmd = [
            self._cli_path,
            "-m", self._model_config.path,
            "-p", prompt,
            "-n", str(self._run_config.output_tokens),
            "-c", str(self._run_config.context_length),
            "--no-display-prompt",
        ]
        _LOG.debug("CLI invocation: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            raise RuntimeError(
                f"llama-cli exited with code {proc.returncode}: {proc.stderr[:500]}"
            )
        return proc.stdout.strip(), 0, self._run_config.output_tokens

    def shutdown(self) -> None:
        self.emit_phase(PhaseEvent.TEARDOWN_START)
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._proc.wait(timeout=10)
        self._llm = None
        self.emit_phase(PhaseEvent.TEARDOWN_END)

    def get_target_pids(self) -> List[int]:
        if self._pid is not None:
            return [self._pid]
        return []

    def get_runtime_metadata(self) -> EngineRuntimeMetadata:
        return EngineRuntimeMetadata(
            engine_name="llamacpp",
            engine_version=self._version,
            model_id=self._model_config.id,
            model_path=self._model_config.path,
            context_length=self._run_config.context_length,
            batch_size=self._run_config.batch_size,
            dtype=self._model_config.dtype,
            quantisation=self._model_config.quantisation,
            extra={
                "use_python_bindings": self._use_bindings,
                "build_flags": self._build_flags,
                "cli_path": self._cli_path,
            },
        )

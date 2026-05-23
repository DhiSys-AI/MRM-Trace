from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from mrm_trace.engines.base import (
    BaseEngine,
    EngineRuntimeMetadata,
    GenerationResult,
    PhaseEvent,
)
from mrm_trace.utils.logging import get_logger

_LOG = get_logger(__name__)


def _collect_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {"vllm": None, "torch": None, "cuda": None}
    try:
        import vllm
        versions["vllm"] = getattr(vllm, "__version__", None)
    except ImportError:
        pass
    try:
        import torch
        versions["torch"] = torch.__version__
        versions["cuda"] = torch.version.cuda if torch.cuda.is_available() else None
    except ImportError:
        pass
    return versions


def _discover_worker_pids(parent_pid: int) -> List[int]:
    """Best-effort discovery of vLLM worker PIDs via psutil."""
    try:
        import psutil

        parent = psutil.Process(parent_pid)
        return [
            p.pid
            for p in parent.children(recursive=True)
            if any("vllm" in part.lower() for part in p.cmdline())
        ]
    except Exception:
        return []


class VLLMEngine(BaseEngine):
    """
    Inference engine wrapper for vLLM.

    Uses the vLLM Python API for local paths or HuggingFace model repositories.
    Handles CPU-only environments transparently (vLLM detects CUDA availability).
    Worker processes are discovered via psutil for collector PID attachment.
    """

    def __init__(self, model_config: Any, run_config: Any) -> None:
        super().__init__(model_config, run_config)
        self._llm: Optional[Any] = None
        self._versions: Dict[str, Optional[str]] = _collect_versions()
        self._worker_pids: List[int] = []

    @property
    def _model_ref(self) -> str:
        """HuggingFace repo takes precedence over local path."""
        return self._model_config.hf_repo or self._model_config.path

    # ── BaseEngine ─────────────────────────────────────────────────────────────

    def prepare(self) -> None:
        try:
            import vllm  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "vllm is not installed. Install with: pip install vllm"
            ) from exc

        if not self._model_ref:
            raise ValueError(
                f"vLLM engine requires 'hf_repo' or 'path' for model '{self._model_config.id}'"
            )
        _LOG.info(
            "vLLM %s | torch %s | CUDA %s",
            self._versions["vllm"],
            self._versions["torch"],
            self._versions["cuda"] or "n/a",
        )

    def load_model(self) -> None:
        self.emit_phase(
            PhaseEvent.MODEL_LOAD_START,
            {"model_ref": self._model_ref, "dtype": self._model_config.dtype},
        )
        from vllm import LLM

        _LOG.info("Loading model via vLLM: %s (dtype=%s)", self._model_ref, self._model_config.dtype)
        self._llm = LLM(
            model=self._model_ref,
            dtype=self._model_config.dtype,
            trust_remote_code=False,
        )
        self._worker_pids = _discover_worker_pids(os.getpid())
        self.emit_phase(
            PhaseEvent.MODEL_LOAD_END,
            {"worker_pids": self._worker_pids, "versions": self._versions},
        )

    def warmup(self, prompt: str = "Hello") -> None:
        _LOG.info("vLLM warmup run (result discarded)")
        self.generate(prompt)

    def generate(self, prompt: str) -> GenerationResult:
        from vllm import SamplingParams

        self.emit_phase(
            PhaseEvent.GENERATION_START,
            {"max_tokens": self._run_config.output_tokens, "n_prompts": 1},
        )
        start = time.perf_counter()

        params = SamplingParams(
            max_tokens=self._run_config.output_tokens,
            temperature=0.0,  # deterministic for reproducibility
        )
        outputs = self._llm.generate([prompt], params)
        elapsed = time.perf_counter() - start

        out = outputs[0].outputs[0]
        prompt_tokens = len(outputs[0].prompt_token_ids)
        output_tokens = len(out.token_ids)
        tps = output_tokens / elapsed if elapsed > 0 else 0.0

        self.emit_phase(
            PhaseEvent.GENERATION_END,
            {"output_tokens": output_tokens, "duration_s": elapsed, "tokens_per_second": tps},
        )
        return GenerationResult(
            text=out.text,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            duration_s=elapsed,
            tokens_per_second=tps,
        )

    def shutdown(self) -> None:
        self.emit_phase(PhaseEvent.TEARDOWN_START)
        try:
            if self._llm is not None:
                del self._llm
                self._llm = None
        except Exception as exc:
            _LOG.warning("Error during vLLM shutdown: %s", exc)
        self.emit_phase(PhaseEvent.TEARDOWN_END)

    def get_target_pids(self) -> List[int]:
        pids = [os.getpid()]
        pids.extend(self._worker_pids)
        return pids

    def get_runtime_metadata(self) -> EngineRuntimeMetadata:
        return EngineRuntimeMetadata(
            engine_name="vllm",
            engine_version=self._versions.get("vllm"),
            model_id=self._model_config.id,
            model_path=self._model_ref,
            context_length=self._run_config.context_length,
            batch_size=self._run_config.batch_size,
            dtype=self._model_config.dtype,
            quantisation=self._model_config.quantisation,
            extra={
                "torch_version": self._versions.get("torch"),
                "cuda_version": self._versions.get("cuda"),
                "worker_pids": self._worker_pids,
            },
        )

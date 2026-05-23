from mrm_trace.engines.base import (
    BaseEngine,
    EngineRuntimeMetadata,
    GenerationResult,
    PhaseEvent,
    PhaseRecord,
)
from mrm_trace.config.schema import EngineType


def create_engine(model_config: object, run_config: object) -> BaseEngine:
    """Factory: return the correct engine for the given model config."""
    if model_config.engine == EngineType.LLAMACPP:
        from mrm_trace.engines.llamacpp import LlamaCppEngine
        return LlamaCppEngine(model_config, run_config)
    if model_config.engine == EngineType.VLLM:
        from mrm_trace.engines.vllm import VLLMEngine
        return VLLMEngine(model_config, run_config)
    raise ValueError(f"Unsupported engine: {model_config.engine}")


__all__ = [
    "BaseEngine",
    "EngineRuntimeMetadata",
    "GenerationResult",
    "PhaseEvent",
    "PhaseRecord",
    "create_engine",
]

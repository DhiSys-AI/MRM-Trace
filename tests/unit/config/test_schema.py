import pytest
from pydantic import ValidationError

from mrm_trace.config.schema import (
    AnalyserConfig,
    CollectorConfig,
    CollectorType,
    Compression,
    EngineType,
    ExperimentConfig,
    LabellerConfig,
    MetricName,
    ModelConfig,
    OutputConfig,
    SweepConfig,
    UnknownPolicy,
)

MINIMAL_VALID = {
    "experiment": {"name": "test", "version": "1.0.0"},
    "models": [{"id": "llama3-q4", "engine": "llamacpp", "path": "models/test.gguf"}],
}


# ── ExperimentConfig ───────────────────────────────────────────────────────────

def test_minimal_config_valid():
    cfg = ExperimentConfig.model_validate(MINIMAL_VALID)
    assert cfg.experiment.name == "test"
    assert len(cfg.models) == 1
    assert cfg.models[0].engine == EngineType.LLAMACPP


def test_empty_models_rejected():
    data = {**MINIMAL_VALID, "models": []}
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(data)


def test_missing_experiment_name_rejected():
    data = {"experiment": {}, "models": MINIMAL_VALID["models"]}
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(data)


def test_missing_experiment_section_rejected():
    data = {"models": MINIMAL_VALID["models"]}
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(data)


def test_defaults_are_applied():
    cfg = ExperimentConfig.model_validate(MINIMAL_VALID)
    assert cfg.sweep.repetitions == 1
    assert cfg.collector.primary == CollectorType.PERF_MEM
    assert cfg.output.compression == Compression.ZSTD


def test_full_config_roundtrip():
    cfg = ExperimentConfig.model_validate(MINIMAL_VALID)
    dumped = cfg.model_dump()
    reloaded = ExperimentConfig.model_validate(dumped)
    assert reloaded.experiment.name == cfg.experiment.name
    assert reloaded.models[0].id == cfg.models[0].id


# ── ModelConfig ────────────────────────────────────────────────────────────────

def test_invalid_engine_rejected():
    with pytest.raises(ValidationError):
        ModelConfig.model_validate({"id": "m", "engine": "bad_engine", "path": "x.gguf"})


def test_llamacpp_requires_path():
    with pytest.raises(ValidationError):
        ModelConfig.model_validate({"id": "m", "engine": "llamacpp"})


def test_vllm_requires_path_or_repo():
    with pytest.raises(ValidationError):
        ModelConfig.model_validate({"id": "m", "engine": "vllm"})


def test_vllm_accepts_hf_repo():
    m = ModelConfig.model_validate({"id": "m", "engine": "vllm", "hf_repo": "org/model"})
    assert m.hf_repo == "org/model"
    assert m.path is None


def test_vllm_accepts_path():
    m = ModelConfig.model_validate({"id": "m", "engine": "vllm", "path": "/weights/model"})
    assert m.path == "/weights/model"


def test_llamacpp_accepts_path():
    m = ModelConfig.model_validate({"id": "m", "engine": "llamacpp", "path": "model.gguf"})
    assert m.path == "model.gguf"


# ── SweepConfig ────────────────────────────────────────────────────────────────

def test_sweep_defaults():
    s = SweepConfig()
    assert s.repetitions == 1
    assert s.warmup_runs == 0
    assert len(s.context_lengths) > 0
    assert len(s.batch_sizes) > 0
    assert len(s.output_tokens) > 0


def test_sweep_rejects_zero_context_length():
    with pytest.raises(ValidationError):
        SweepConfig.model_validate({"context_lengths": [0, 512]})


def test_sweep_rejects_negative_context_length():
    with pytest.raises(ValidationError):
        SweepConfig.model_validate({"context_lengths": [-1]})


def test_sweep_rejects_zero_batch_size():
    with pytest.raises(ValidationError):
        SweepConfig.model_validate({"batch_sizes": [0]})


def test_sweep_rejects_zero_repetitions():
    with pytest.raises(ValidationError):
        SweepConfig.model_validate({"repetitions": 0})


def test_sweep_rejects_negative_repetitions():
    with pytest.raises(ValidationError):
        SweepConfig.model_validate({"repetitions": -1})


def test_sweep_rejects_negative_warmup():
    with pytest.raises(ValidationError):
        SweepConfig.model_validate({"warmup_runs": -1})


def test_sweep_rejects_empty_context_lengths():
    with pytest.raises(ValidationError):
        SweepConfig.model_validate({"context_lengths": []})


def test_sweep_accepts_multiple_values():
    s = SweepConfig.model_validate({
        "context_lengths": [512, 2048, 8192],
        "batch_sizes": [1, 4, 8],
        "output_tokens": [128, 512],
        "repetitions": 3,
        "warmup_runs": 1,
    })
    assert len(s.context_lengths) == 3
    assert s.repetitions == 3


# ── CollectorConfig ────────────────────────────────────────────────────────────

def test_collector_defaults():
    c = CollectorConfig()
    assert c.primary == CollectorType.PERF_MEM
    assert c.fallback == CollectorType.MEMRAY
    assert c.fallback_2 == CollectorType.PROCESS_MONITOR


def test_collector_accepts_all_types():
    for ctype in CollectorType:
        c = CollectorConfig.model_validate({"primary": ctype.value})
        assert c.primary == ctype


# ── Enums ──────────────────────────────────────────────────────────────────────

def test_unknown_policy_all_values():
    for policy in UnknownPolicy:
        assert UnknownPolicy(policy.value) == policy


def test_metric_name_all_values():
    for m in MetricName:
        assert MetricName(m.value) == m


def test_compression_all_values():
    for c in Compression:
        assert Compression(c.value) == c


# ── AnalyserConfig ─────────────────────────────────────────────────────────────

def test_analyser_defaults():
    a = AnalyserConfig()
    assert MetricName.RETENTION_DURATION in a.metrics
    assert MetricName.WRITE_ONCE_RATIO in a.metrics
    assert a.histogram_bins == 100


# ── LabellerConfig ─────────────────────────────────────────────────────────────

def test_labeller_defaults():
    lab = LabellerConfig()
    assert lab.unknown_policy == UnknownPolicy.LOG_AND_SKIP
    assert lab.min_sample_count == 10


# ── OutputConfig ───────────────────────────────────────────────────────────────

def test_output_defaults():
    out = OutputConfig()
    assert out.base_dir == "results/"
    assert out.compression == Compression.ZSTD
    assert out.hf_dataset_push is False

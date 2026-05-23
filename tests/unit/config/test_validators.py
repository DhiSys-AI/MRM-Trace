import pytest
import yaml

from mrm_trace.config.schema import ExperimentConfig
from mrm_trace.config.validators import (
    ValidationResult,
    expand_sweep,
    validate_config_semantics,
    validate_model_paths,
)

BASE = {
    "experiment": {"name": "test"},
    "models": [{"id": "llama3-q4", "engine": "llamacpp", "path": "models/test.gguf"}],
    "sweep": {
        "context_lengths": [512, 2048],
        "batch_sizes": [1, 4],
        "output_tokens": [128],
        "repetitions": 2,
        "warmup_runs": 1,
    },
}


def make_config(**overrides) -> ExperimentConfig:
    data = {**BASE, **overrides}
    return ExperimentConfig.model_validate(data)


# ── ValidationResult ───────────────────────────────────────────────────────────

def test_validation_result_valid_when_no_errors():
    r = ValidationResult()
    assert r.is_valid

def test_validation_result_invalid_when_error_added():
    r = ValidationResult()
    r.error("something wrong")
    assert not r.is_valid
    assert "something wrong" in r.errors

def test_validation_result_warnings_dont_affect_validity():
    r = ValidationResult()
    r.warning("heads up")
    assert r.is_valid
    assert "heads up" in r.warnings


# ── expand_sweep ───────────────────────────────────────────────────────────────

def test_expand_sweep_correct_total():
    config = make_config()
    runs = expand_sweep(config)
    # 1 model * 2 ctx * 2 batch * 1 out_tokens * (1 warmup + 2 reps) = 12
    assert len(runs) == 12


def test_expand_sweep_warmup_count():
    config = make_config()
    runs = expand_sweep(config)
    warmups = [r for r in runs if r.is_warmup]
    # 1 model * 2 ctx * 2 batch * 1 out_tokens * 1 warmup = 4
    assert len(warmups) == 4  # 1 warmup run per combo


def test_expand_sweep_actual_count():
    config = make_config()
    runs = expand_sweep(config)
    actuals = [r for r in runs if not r.is_warmup]
    # 1 model * 2 ctx * 2 batch * 1 out_tokens * 2 reps = 8
    assert len(actuals) == 8  # 2 actual reps per combo


def test_expand_sweep_no_warmup():
    config = make_config(sweep={
        "context_lengths": [512], "batch_sizes": [1],
        "output_tokens": [128], "repetitions": 3, "warmup_runs": 0,
    })
    runs = expand_sweep(config)
    assert all(not r.is_warmup for r in runs)
    assert len(runs) == 3


def test_expand_sweep_unique_keys():
    config = make_config()
    runs = expand_sweep(config)
    keys = set()
    for r in runs:
        key = (r.model.id, r.context_length, r.batch_size, r.output_tokens, r.rep_index, r.is_warmup)
        assert key not in keys, f"Duplicate run key: {key}"
        keys.add(key)


def test_expand_sweep_run_ids_unique():
    config = make_config()
    runs = expand_sweep(config)
    ids = [r.run_id for r in runs]
    assert len(ids) == len(set(ids))


def test_expand_sweep_run_config_fields():
    config = make_config()
    runs = expand_sweep(config)
    for r in runs:
        assert r.run_id
        assert r.model is not None
        assert r.context_length > 0
        assert r.batch_size > 0
        assert r.output_tokens > 0
        assert r.output_dir is not None


def test_expand_sweep_output_dir_contains_model_id():
    config = make_config()
    runs = expand_sweep(config)
    for r in runs:
        assert "llama3-q4" in r.output_dir


def test_expand_sweep_multi_model():
    config = ExperimentConfig.model_validate({
        "experiment": {"name": "multi"},
        "models": [
            {"id": "model-a", "engine": "llamacpp", "path": "a.gguf"},
            {"id": "model-b", "engine": "llamacpp", "path": "b.gguf"},
        ],
        "sweep": {"context_lengths": [512], "batch_sizes": [1], "output_tokens": [128], "repetitions": 1},
    })
    runs = expand_sweep(config)
    model_ids = {r.model.id for r in runs}
    assert "model-a" in model_ids
    assert "model-b" in model_ids


# ── validate_model_paths ───────────────────────────────────────────────────────

def test_validate_missing_model_path_is_error():
    config = make_config()
    result = validate_model_paths(config)
    assert not result.is_valid
    assert any("not found" in e for e in result.errors)


def test_validate_existing_model_path_is_ok(tmp_path):
    model_file = tmp_path / "model.gguf"
    model_file.touch()
    config = ExperimentConfig.model_validate({
        "experiment": {"name": "t"},
        "models": [{"id": "m", "engine": "llamacpp", "path": str(model_file)}],
    })
    result = validate_model_paths(config)
    assert result.is_valid


def test_validate_hf_repo_no_path_check():
    config = ExperimentConfig.model_validate({
        "experiment": {"name": "t"},
        "models": [{"id": "m", "engine": "vllm", "hf_repo": "org/model"}],
    })
    result = validate_model_paths(config)
    assert result.is_valid  # no local path to check


# ── validate_config_semantics ──────────────────────────────────────────────────

def test_semantics_duplicate_model_ids():
    config = ExperimentConfig.model_validate({
        "experiment": {"name": "t"},
        "models": [
            {"id": "same", "engine": "llamacpp", "path": "a.gguf"},
            {"id": "same", "engine": "llamacpp", "path": "b.gguf"},
        ],
    })
    result = validate_config_semantics(config)
    assert not result.is_valid
    assert any("unique" in e.lower() for e in result.errors)


def test_semantics_unique_model_ids_valid():
    config = make_config()
    result = validate_config_semantics(config)
    assert result.is_valid


def test_semantics_large_sweep_warns():
    many_models = [
        {"id": f"m{i}", "engine": "llamacpp", "path": f"m{i}.gguf"}
        for i in range(5)
    ]
    config = ExperimentConfig.model_validate({
        "experiment": {"name": "big"},
        "models": many_models,
        "sweep": {
            "context_lengths": [512, 1024, 2048, 4096, 8192, 16384],
            "batch_sizes": [1, 4, 8, 16],
            "output_tokens": [128, 512, 1024],
            "repetitions": 5,
        },
    })
    result = validate_config_semantics(config)
    # 5 * 6 * 4 * 3 * 5 = 1800 > 1000 — should warn
    assert any("runs" in w for w in result.warnings)

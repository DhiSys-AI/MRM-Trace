import pytest
import yaml

from mrm_trace.config.loader import load_experiment
from mrm_trace.config.schema import EngineType

VALID_CONFIG = {
    "experiment": {"name": "test_exp", "version": "1.0.0"},
    "models": [
        {"id": "model1", "engine": "llamacpp", "path": "path/to/model.gguf"}
    ],
}


def test_load_valid_config(tmp_path):
    cfg_file = tmp_path / "exp.yaml"
    cfg_file.write_text(yaml.dump(VALID_CONFIG))
    exp = load_experiment(cfg_file)
    assert exp.experiment.name == "test_exp"
    assert exp.models[0].id == "model1"
    assert exp.models[0].engine == EngineType.LLAMACPP


def test_load_accepts_string_path(tmp_path):
    cfg_file = tmp_path / "exp.yaml"
    cfg_file.write_text(yaml.dump(VALID_CONFIG))
    exp = load_experiment(str(cfg_file))
    assert exp.experiment.name == "test_exp"


def test_load_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_experiment("/definitely/nonexistent/path.yaml")


def test_load_empty_file(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_experiment(empty)


def test_load_invalid_yaml_syntax(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(": : this is : invalid YAML :")
    with pytest.raises(ValueError):
        load_experiment(bad)


def test_load_missing_experiment_section(tmp_path):
    data = {"models": [{"id": "m", "engine": "llamacpp", "path": "x.gguf"}]}
    cfg_file = tmp_path / "missing.yaml"
    cfg_file.write_text(yaml.dump(data))
    with pytest.raises(ValueError, match="Invalid experiment config"):
        load_experiment(cfg_file)


def test_load_invalid_engine(tmp_path):
    data = {
        "experiment": {"name": "test"},
        "models": [{"id": "m", "engine": "unknown_engine", "path": "x.gguf"}],
    }
    cfg_file = tmp_path / "bad.yaml"
    cfg_file.write_text(yaml.dump(data))
    with pytest.raises(ValueError, match="Invalid experiment config"):
        load_experiment(cfg_file)


def test_load_missing_model_path_for_llamacpp(tmp_path):
    data = {
        "experiment": {"name": "test"},
        "models": [{"id": "m", "engine": "llamacpp"}],
    }
    cfg_file = tmp_path / "no_path.yaml"
    cfg_file.write_text(yaml.dump(data))
    with pytest.raises(ValueError):
        load_experiment(cfg_file)


def test_load_full_experiment_yaml(tmp_path):
    cfg_file = tmp_path / "full.yaml"
    cfg_file.write_text(yaml.dump({
        "experiment": {
            "name": "full_test",
            "description": "Full config test",
            "version": "1.0.0",
            "date_created": "2026-05-23",
        },
        "models": [
            {"id": "llama3-q4", "engine": "llamacpp", "path": "models/m.gguf",
             "family": "llama3", "params_billions": 8, "quantisation": "Q4_K_M"},
        ],
        "sweep": {
            "context_lengths": [512, 2048],
            "batch_sizes": [1, 4],
            "output_tokens": [128],
            "repetitions": 2,
            "warmup_runs": 1,
        },
        "collector": {"primary": "memray"},
        "output": {"base_dir": "results/", "compression": "zstd"},
    }))
    exp = load_experiment(cfg_file)
    assert exp.experiment.name == "full_test"
    assert exp.sweep.repetitions == 2
    assert len(exp.sweep.context_lengths) == 2

import yaml
from typer.testing import CliRunner

from mrm_trace.cli import app

runner = CliRunner()


def _write_config(tmp_path, data: dict) -> str:
    f = tmp_path / "exp.yaml"
    f.write_text(yaml.dump(data))
    return str(f)


# ── validate command ───────────────────────────────────────────────────────────

def test_validate_missing_config_file():
    result = runner.invoke(app, ["validate", "--config", "/nonexistent/path.yaml"])
    assert result.exit_code == 1


def test_validate_config_parse_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(": : not yaml :")
    result = runner.invoke(app, ["validate", "--config", str(bad)])
    assert result.exit_code == 1


def test_validate_missing_model_path_exits_1(tmp_path):
    cfg = _write_config(tmp_path, {
        "experiment": {"name": "test"},
        "models": [{"id": "m", "engine": "llamacpp", "path": "nonexistent_model.gguf"}],
    })
    result = runner.invoke(app, ["validate", "--config", cfg])
    assert result.exit_code == 1


def test_validate_error_message_shown(tmp_path):
    cfg = _write_config(tmp_path, {
        "experiment": {"name": "test"},
        "models": [{"id": "m", "engine": "llamacpp", "path": "no_such_model.gguf"}],
    })
    result = runner.invoke(app, ["validate", "--config", cfg])
    output = (result.output or "") + (result.stderr if hasattr(result, "stderr") else "")
    assert "ERROR" in output or "not found" in output.lower()


def test_validate_existing_model_path_exits_0(tmp_path):
    model = tmp_path / "model.gguf"
    model.touch()
    cfg = _write_config(tmp_path, {
        "experiment": {"name": "test"},
        "models": [{"id": "m", "engine": "llamacpp", "path": str(model)}],
    })
    result = runner.invoke(app, ["validate", "--config", cfg])
    assert result.exit_code == 0


def test_validate_success_message(tmp_path):
    model = tmp_path / "model.gguf"
    model.touch()
    cfg = _write_config(tmp_path, {
        "experiment": {"name": "test"},
        "models": [{"id": "m", "engine": "llamacpp", "path": str(model)}],
    })
    result = runner.invoke(app, ["validate", "--config", cfg])
    assert "valid" in result.output.lower()


def test_validate_duplicate_model_ids_exits_1(tmp_path):
    cfg = _write_config(tmp_path, {
        "experiment": {"name": "test"},
        "models": [
            {"id": "same", "engine": "llamacpp", "path": "a.gguf"},
            {"id": "same", "engine": "llamacpp", "path": "b.gguf"},
        ],
    })
    result = runner.invoke(app, ["validate", "--config", cfg])
    assert result.exit_code == 1


# ── plan command ───────────────────────────────────────────────────────────────

def test_plan_shows_experiment_name(tmp_path):
    model = tmp_path / "model.gguf"
    model.touch()
    cfg = _write_config(tmp_path, {
        "experiment": {"name": "my_experiment"},
        "models": [{"id": "m", "engine": "llamacpp", "path": str(model)}],
        "sweep": {"context_lengths": [512], "batch_sizes": [1], "output_tokens": [128],
                  "repetitions": 1, "warmup_runs": 0},
    })
    result = runner.invoke(app, ["plan", "--config", cfg])
    assert result.exit_code == 0
    assert "my_experiment" in result.output


def test_plan_shows_model_id(tmp_path):
    model = tmp_path / "model.gguf"
    model.touch()
    cfg = _write_config(tmp_path, {
        "experiment": {"name": "test"},
        "models": [{"id": "llama3-q4", "engine": "llamacpp", "path": str(model)}],
        "sweep": {"context_lengths": [512], "batch_sizes": [1], "output_tokens": [128],
                  "repetitions": 1, "warmup_runs": 0},
    })
    result = runner.invoke(app, ["plan", "--config", cfg])
    assert result.exit_code == 0
    assert "llama3-q4" in result.output


def test_plan_correct_run_count(tmp_path):
    model = tmp_path / "model.gguf"
    model.touch()
    cfg = _write_config(tmp_path, {
        "experiment": {"name": "sweep"},
        "models": [{"id": "m", "engine": "llamacpp", "path": str(model)}],
        "sweep": {"context_lengths": [512, 2048], "batch_sizes": [1, 4],
                  "output_tokens": [128], "repetitions": 3, "warmup_runs": 0},
    })
    result = runner.invoke(app, ["plan", "--config", cfg])
    assert result.exit_code == 0
    # 1 * 2 * 2 * 1 * 3 = 12
    assert "12" in result.output


def test_plan_shows_warmup_and_actual_counts(tmp_path):
    model = tmp_path / "model.gguf"
    model.touch()
    cfg = _write_config(tmp_path, {
        "experiment": {"name": "test"},
        "models": [{"id": "m", "engine": "llamacpp", "path": str(model)}],
        "sweep": {"context_lengths": [512], "batch_sizes": [1], "output_tokens": [128],
                  "repetitions": 2, "warmup_runs": 1},
    })
    result = runner.invoke(app, ["plan", "--config", cfg])
    assert result.exit_code == 0
    assert "warmup" in result.output.lower()


def test_plan_missing_config_exits_1():
    result = runner.invoke(app, ["plan", "--config", "/no/such/file.yaml"])
    assert result.exit_code == 1


# ── unimplemented commands ─────────────────────────────────────────────────────

def test_run_exits_1():
    result = runner.invoke(app, ["run", "--config", "any.yaml"])
    assert result.exit_code == 1
    assert "NOT IMPLEMENTED" in result.output


def test_analyse_exits_1():
    result = runner.invoke(app, ["analyse", "--input", "any/"])
    assert result.exit_code == 1
    assert "NOT IMPLEMENTED" in result.output


def test_report_exits_1():
    result = runner.invoke(app, ["report", "--input", "any/"])
    assert result.exit_code == 1
    assert "NOT IMPLEMENTED" in result.output


def test_export_exits_1():
    result = runner.invoke(app, ["export", "--input", "any/"])
    assert result.exit_code == 1
    assert "NOT IMPLEMENTED" in result.output

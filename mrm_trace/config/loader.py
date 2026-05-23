from pathlib import Path
from typing import Union

import yaml
from pydantic import ValidationError

from mrm_trace.config.schema import ExperimentConfig


def load_experiment(config_path: Union[str, Path]) -> ExperimentConfig:
    """Load and validate an experiment YAML config file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            raw = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML: {e}") from e

    if raw is None:
        raise ValueError("Config file is empty")

    try:
        return ExperimentConfig.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"Invalid experiment config:\n{e}") from e

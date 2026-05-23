from pathlib import Path
from typing import List, Union

from mrm_trace.config.loader import load_experiment
from mrm_trace.config.schema import ExperimentConfig, RunConfig
from mrm_trace.config.validators import (
    expand_sweep,
    validate_config_semantics,
    validate_environment,
    validate_model_paths,
)


class Experiment:
    """Python API for defining and executing mrm-trace experiments."""

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Experiment":
        """Load an experiment from a YAML config file."""
        return cls(load_experiment(path))

    def validate(self) -> bool:
        """Validate config and environment. Returns True if no errors."""
        all_ok = True
        for check_fn in (validate_config_semantics, validate_model_paths, validate_environment):
            result = check_fn(self._config)
            if not result.is_valid:
                all_ok = False
        return all_ok

    def plan(self) -> List[RunConfig]:
        """Materialise and return the full list of runs for this experiment."""
        return expand_sweep(self._config)

    def run(self) -> None:
        """Execute all planned runs. Implemented in Phase 3."""
        raise NotImplementedError("run() is available from Phase 3 onwards")

    def analyse(self) -> None:
        """Parse, label, and compute metrics. Implemented in Phase 4–6."""
        raise NotImplementedError("analyse() is available from Phase 4 onwards")

    def report(self, output_dir: Union[str, Path] = "reports/") -> None:
        """Generate aggregated outputs and figures. Implemented in Phase 8."""
        raise NotImplementedError("report() is available from Phase 8 onwards")

    @property
    def config(self) -> ExperimentConfig:
        return self._config

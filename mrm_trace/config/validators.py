import shutil
from pathlib import Path
from typing import List

from mrm_trace.config.schema import CollectorType, ExperimentConfig, RunConfig
from mrm_trace.utils.ids import generate_run_id


class ValidationResult:
    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warning(self, msg: str) -> None:
        self.warnings.append(msg)


def validate_config_semantics(config: ExperimentConfig) -> ValidationResult:
    result = ValidationResult()

    # Unique model IDs
    ids = [m.id for m in config.models]
    if len(ids) != len(set(ids)):
        result.error("Model IDs must be unique")

    # Output dir must be a directory if it already exists
    out_dir = Path(config.output.base_dir)
    if out_dir.exists() and not out_dir.is_dir():
        result.error(
            f"output.base_dir exists but is not a directory: {out_dir}"
        )

    # Warn about very large sweeps
    total = (
        len(config.models)
        * len(config.sweep.context_lengths)
        * len(config.sweep.batch_sizes)
        * len(config.sweep.output_tokens)
        * config.sweep.repetitions
    )
    if total > 1000:
        result.warning(
            f"Sweep will produce {total} actual runs — this may take a long time"
        )

    return result


def validate_model_paths(config: ExperimentConfig) -> ValidationResult:
    result = ValidationResult()
    for model in config.models:
        if model.path:
            p = Path(model.path)
            if not p.exists():
                result.error(
                    f"Model '{model.id}' path not found: {model.path}"
                )
    return result


def validate_environment(config: ExperimentConfig) -> ValidationResult:
    result = ValidationResult()

    primary = config.collector.primary
    if primary == CollectorType.PERF_MEM:
        if not shutil.which("perf"):
            result.warning(
                "'perf' not found in PATH; primary collector unavailable — "
                "will fall back to next available collector"
            )

    uses_memray = (
        primary == CollectorType.MEMRAY
        or config.collector.fallback == CollectorType.MEMRAY
    )
    if uses_memray:
        try:
            import memray  # noqa: F401
        except ImportError:
            result.warning(
                "memray not installed; memray collector unavailable "
                "(install with: pip install memray)"
            )

    return result


def expand_sweep(config: ExperimentConfig) -> List[RunConfig]:
    """Materialise the full set of runs from the sweep configuration."""
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runs: List[RunConfig] = []

    base_dir = Path(config.output.base_dir)

    for model in config.models:
        for ctx in config.sweep.context_lengths:
            for batch in config.sweep.batch_sizes:
                for out_tokens in config.sweep.output_tokens:
                    for warmup_idx in range(config.sweep.warmup_runs):
                        run_id = generate_run_id(
                            model.id, ctx, batch, out_tokens, warmup_idx,
                            warmup=True, ts=ts
                        )
                        runs.append(
                            RunConfig(
                                run_id=run_id,
                                model=model,
                                context_length=ctx,
                                batch_size=batch,
                                output_tokens=out_tokens,
                                rep_index=warmup_idx,
                                is_warmup=True,
                                output_dir=str(base_dir / model.id / run_id),
                            )
                        )
                    for rep in range(config.sweep.repetitions):
                        run_id = generate_run_id(
                            model.id, ctx, batch, out_tokens, rep,
                            warmup=False, ts=ts
                        )
                        runs.append(
                            RunConfig(
                                run_id=run_id,
                                model=model,
                                context_length=ctx,
                                batch_size=batch,
                                output_tokens=out_tokens,
                                rep_index=rep,
                                is_warmup=False,
                                output_dir=str(base_dir / model.id / run_id),
                            )
                        )

    return runs

import logging
from pathlib import Path
from typing import Optional

import typer

import mrm_trace
from mrm_trace.config.loader import load_experiment
from mrm_trace.config.validators import (
    expand_sweep,
    validate_config_semantics,
    validate_environment,
    validate_model_paths,
)
from mrm_trace.utils.logging import configure_root

app = typer.Typer(
    name="mrm-trace",
    help="MRM Trace — LLM inference memory trace platform for MRM research.",
    no_args_is_help=True,
)


@app.command()
def validate(
    config: Path = typer.Option(..., "--config", "-c", help="Path to experiment YAML"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Validate experiment YAML config and check the environment."""
    configure_root(level=logging.DEBUG if verbose else logging.INFO)
    typer.echo(f"Validating: {config}")

    try:
        exp = load_experiment(config)
    except FileNotFoundError as e:
        typer.echo(f"  [ERROR] {e}", err=True)
        raise typer.Exit(1)
    except ValueError as e:
        typer.echo(f"  [ERROR] {e}", err=True)
        raise typer.Exit(1)

    all_ok = True
    for check_fn in (validate_config_semantics, validate_model_paths, validate_environment):
        result = check_fn(exp)
        for err in result.errors:
            typer.echo(f"  [ERROR] {err}", err=True)
            all_ok = False
        for warn in result.warnings:
            typer.echo(f"  [WARN]  {warn}")

    if all_ok:
        typer.echo("Config is valid.")
    else:
        raise typer.Exit(1)


@app.command()
def plan(
    config: Path = typer.Option(..., "--config", "-c", help="Path to experiment YAML"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Expand sweep combinations and display the planned run list."""
    configure_root(level=logging.DEBUG if verbose else logging.INFO)

    try:
        exp = load_experiment(config)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"  [ERROR] {e}", err=True)
        raise typer.Exit(1)

    runs = expand_sweep(exp)
    warmups = [r for r in runs if r.is_warmup]
    actual = [r for r in runs if not r.is_warmup]

    typer.echo(f"\nExperiment : {exp.experiment.name}")
    typer.echo(f"Total runs : {len(runs)}  ({len(warmups)} warmup, {len(actual)} actual)\n")

    header = (
        f"{'#':<5} {'Model':<24} {'Engine':<10} "
        f"{'Ctx':>7} {'Batch':>5} {'Out':>5} {'Rep':>4}  {'Type':<8}"
    )
    typer.echo(header)
    typer.echo("-" * len(header))

    for i, run in enumerate(runs, 1):
        kind = "warmup" if run.is_warmup else f"rep{run.rep_index}"
        typer.echo(
            f"{i:<5} {run.model.id:<24} {run.model.engine.value:<10} "
            f"{run.context_length:>7} {run.batch_size:>5} {run.output_tokens:>5} "
            f"{run.rep_index:>4}  {kind:<8}"
        )


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="Path to experiment YAML"),
) -> None:
    """Execute experiment runs and collect raw artifacts. (Phase 3+)"""
    typer.echo("[NOT IMPLEMENTED] 'run' requires Phase 3 (collectors). Coming soon.", err=True)
    raise typer.Exit(1)


@app.command()
def analyse(
    input: Path = typer.Option(..., "--input", "-i", help="Results directory"),
    metrics: Optional[str] = typer.Option(
        None, "--metrics", "-m", help="Comma-separated metric names"
    ),
) -> None:
    """Parse, label, and compute metrics from raw artifacts. (Phase 4–6)"""
    typer.echo("[NOT IMPLEMENTED] 'analyse' requires Phase 4–6. Coming soon.", err=True)
    raise typer.Exit(1)


@app.command()
def report(
    input: Path = typer.Option(..., "--input", "-i", help="Results directory"),
    output: Path = typer.Option(Path("reports/"), "--output", "-o", help="Output directory"),
) -> None:
    """Generate aggregated CSVs and publication figures. (Phase 8)"""
    typer.echo("[NOT IMPLEMENTED] 'report' requires Phase 8. Coming soon.", err=True)
    raise typer.Exit(1)


@app.command()
def export(
    input: Path = typer.Option(..., "--input", "-i", help="Results directory"),
    format: str = typer.Option("hf", "--format", "-f", help="Export format: hf, zenodo"),
    output: Path = typer.Option(Path("dist/"), "--output", "-o", help="Output directory"),
) -> None:
    """Package outputs for dataset release. (Phase 8)"""
    typer.echo("[NOT IMPLEMENTED] 'export' requires Phase 8. Coming soon.", err=True)
    raise typer.Exit(1)


@app.command()
def version() -> None:
    """Print the mrm-trace version."""
    typer.echo(mrm_trace.__version__)


def main() -> None:
    app()


if __name__ == "__main__":
    main()

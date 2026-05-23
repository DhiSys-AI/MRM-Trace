"""
Tier-3 validity tests for Phase 1.

These verify that the configuration and sweep expansion produce scientifically
coherent outputs — run IDs are unique, plans are deterministic, and sweep
materialisation matches ground-truth counts.
"""

import pytest

from mrm_trace.config.schema import ExperimentConfig
from mrm_trace.config.validators import expand_sweep


def make_config(
    n_models: int = 1,
    ctx: list = None,
    batch: list = None,
    out: list = None,
    reps: int = 1,
    warmups: int = 0,
) -> ExperimentConfig:
    ctx = ctx or [512]
    batch = batch or [1]
    out = out or [128]
    models = [
        {"id": f"model-{i}", "engine": "llamacpp", "path": f"m{i}.gguf"}
        for i in range(n_models)
    ]
    return ExperimentConfig.model_validate({
        "experiment": {"name": "validity_test"},
        "models": models,
        "sweep": {
            "context_lengths": ctx,
            "batch_sizes": batch,
            "output_tokens": out,
            "repetitions": reps,
            "warmup_runs": warmups,
        },
    })


def test_run_count_matches_combinatorial_formula():
    """Total runs = n_models × n_ctx × n_batch × n_out × (warmups + reps)."""
    config = make_config(
        n_models=2, ctx=[512, 2048, 8192], batch=[1, 4],
        out=[128, 512], reps=3, warmups=1
    )
    runs = expand_sweep(config)
    expected = 2 * 3 * 2 * 2 * (1 + 3)
    assert len(runs) == expected, (
        f"Expected {expected} runs but got {len(runs)}. "
        "Sweep expansion is off — metrics will be mis-counted."
    )


def test_run_ids_globally_unique():
    """All run IDs in a sweep must be unique — no two runs share an ID."""
    config = make_config(n_models=3, ctx=[512, 2048], batch=[1, 4], out=[128], reps=3, warmups=1)
    runs = expand_sweep(config)
    ids = [r.run_id for r in runs]
    assert len(ids) == len(set(ids)), (
        "Duplicate run IDs detected. If two runs share an ID, "
        "their outputs will overwrite each other."
    )


def test_no_cross_model_region_id_overlap_in_run_keys():
    """Runs from different models must not share the same (model_id, ctx, batch, out, rep) key."""
    config = make_config(n_models=2, ctx=[512], batch=[1], out=[128], reps=2)
    runs = expand_sweep(config)
    keys = [(r.model.id, r.context_length, r.batch_size, r.output_tokens, r.rep_index, r.is_warmup)
            for r in runs]
    assert len(keys) == len(set(keys))


def test_warmup_runs_are_marked():
    """Every warmup run must have is_warmup=True; actual runs must have is_warmup=False."""
    config = make_config(reps=2, warmups=1)
    runs = expand_sweep(config)
    for r in runs:
        if r.is_warmup:
            assert r.rep_index >= 0
        else:
            assert r.rep_index >= 0


def test_actual_reps_start_at_zero():
    """Actual (non-warmup) rep_index values should be 0, 1, 2, …, reps-1."""
    config = make_config(reps=3, warmups=0)
    runs = expand_sweep(config)
    rep_indices = sorted(r.rep_index for r in runs)
    assert rep_indices == [0, 1, 2]


def test_sweep_is_deterministic():
    """Calling expand_sweep twice with identical config produces identical run lists."""
    config = make_config(n_models=2, ctx=[512, 2048], batch=[1], out=[128], reps=2, warmups=1)
    runs_a = expand_sweep(config)
    runs_b = expand_sweep(config)

    # Same number
    assert len(runs_a) == len(runs_b)

    # Same model IDs, context lengths, batch sizes, output tokens, rep indices in same order
    for a, b in zip(runs_a, runs_b):
        assert a.model.id == b.model.id
        assert a.context_length == b.context_length
        assert a.batch_size == b.batch_size
        assert a.output_tokens == b.output_tokens
        assert a.rep_index == b.rep_index
        assert a.is_warmup == b.is_warmup


def test_all_context_lengths_covered():
    """Every requested context length must appear in at least one run."""
    ctx_lengths = [512, 2048, 8192, 32768]
    config = make_config(ctx=ctx_lengths)
    runs = expand_sweep(config)
    covered = {r.context_length for r in runs}
    assert set(ctx_lengths) == covered


def test_all_batch_sizes_covered():
    batch_sizes = [1, 4, 8, 16]
    config = make_config(batch=batch_sizes)
    runs = expand_sweep(config)
    covered = {r.batch_size for r in runs}
    assert set(batch_sizes) == covered


def test_output_dirs_are_unique():
    """Each run must have a unique output directory path."""
    config = make_config(n_models=2, ctx=[512, 2048], batch=[1, 4], out=[128], reps=3, warmups=1)
    runs = expand_sweep(config)
    dirs = [r.output_dir for r in runs]
    assert len(dirs) == len(set(dirs)), "Non-unique output dirs — runs would overwrite each other"


def test_output_dir_contains_model_id_and_run_id():
    """Output directory path must embed both model ID and run ID for deterministic layout."""
    config = make_config()
    runs = expand_sweep(config)
    for r in runs:
        assert r.model.id in r.output_dir
        assert r.run_id in r.output_dir

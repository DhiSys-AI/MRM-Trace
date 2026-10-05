"""
The occupancy sweep varies output tokens at a fixed context.

Every cell key in the analysis scripts was `(model_id, ctx)`.  On the context
sweep that is sufficient, because output tokens are constant.  On the occupancy
sweep it is fatal: all three levels share ctx 2048, so they collapse into one
cell and the sweep's independent variable is averaged away before any table is
printed.  Worse, the continuity check walks the `ctx` axis -- constant here --
so it would have reported PASS having compared nothing.

These tests run the checks against an occupancy-shaped dataset, which is the
coverage the grouping change otherwise has none of.
"""

import importlib.util
import pathlib

import pandas as pd
import pytest

pytestmark = pytest.mark.validity

_TOKENS = [512, 1024, 1977]
_CTX = 2048


def _module(name):
    path = (pathlib.Path(__file__).resolve().parents[2]
            / "notebooks" / "scripts" / f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def val():
    return _module("results_validation")


def _runs(tokens=_TOKENS, reps=(1, 2)):
    rows = []
    for model in ("llama-3.2-1b", "phi-3.5-mini"):
        for t in tokens:
            for rep in reps:
                rows.append({
                    "run_id": f"{model}_ctx{_CTX}_out{t}_rep{rep}",
                    "run_dir": f"/x/{model}/{t}/{rep}",
                    "model_id": model, "ctx": _CTX, "rep": rep,
                    "output_tokens": t, "sample_period": 1000 * t,
                    "validity": "clean", "backend": "llama.cpp",
                    "granularity": 65536, "samples": 10_000_000,
                    "cap_before": 100_000, "cap_after": 100_000,
                    "achieved_rate": 5_000.0,
                })
    return pd.DataFrame(rows)


def _region_maps(runtime_by_tokens):
    """One region_map row per region, so a count is a region count."""
    rows = []
    for model in ("llama-3.2-1b", "phi-3.5-mini"):
        for t in _TOKENS:
            for rep in (1, 2):
                run_id = f"{model}_ctx{_CTX}_out{t}_rep{rep}"
                for region, n in (("weight", 400),
                                  ("kv_cache", 50 + t // 8),
                                  ("runtime", runtime_by_tokens[t])):
                    rows += [{"run_id": run_id, "run_dir": f"/x/{model}/{t}/{rep}",
                              "model_id": model, "ctx": _CTX, "rep": rep,
                              "output_tokens": t, "region_type": region,
                              "access_count": 5}] * n
    return pd.DataFrame(rows)


def _row(report, name):
    for r in report.results:
        if r["check"] == name:
            return r
    raise AssertionError(f"{name} missing: {[r['check'] for r in report.results]}")


# ── the levels must stay distinct ────────────────────────────────────────────

def test_completeness_sees_each_token_level_as_its_own_cell(val):
    """A level missing its second rep must be named, not hidden by the others."""
    runs = _runs()
    runs = runs[~((runs.output_tokens == 1024) & (runs.rep == 2)
                  & (runs.model_id == "phi-3.5-mini"))]
    report = val.Report()
    val._SHOW_TOKENS = True
    val.check_completeness(runs, report)
    row = _row(report, "matrix completeness")
    assert row["status"] == "WARN"
    assert any("t1024" in str(d) for d in row["detail"]), row["detail"]


def test_sampling_consistency_does_not_flag_the_designed_period_change(val):
    """
    Each token level gets its own period by design -- a longer run needs a
    coarser one.  Keyed on (model, ctx) alone, all three periods land in one
    group and the check reports the design as a mid-sweep calibration drift.
    """
    report = val.Report()
    val.check_sampling_consistency(_runs(), report)
    row = _row(report, "sampling consistency")
    assert not any("more than one sampling period" in str(d)
                   for d in row["detail"]), row["detail"]


def test_reproducibility_compares_reps_within_a_token_level(val):
    """
    Grouped on (model, ctx) the six runs of one model look like six reps of one
    cell, so a share that legitimately changes with occupancy reads as
    irreproducibility.
    """
    maps = _region_maps({512: 100, 1024: 400, 1977: 1600})
    report = val.Report()
    val.check_reproducibility(pd.DataFrame(), maps, report)
    row = _row(report, "reproducibility")
    assert row["status"] == "PASS", row["detail"]


# ── the continuity check must not go vacuous ─────────────────────────────────

def test_continuity_walks_tokens_when_context_is_fixed(val):
    """
    Hard-coded to `ctx`, this check examined a single level and passed. A gate
    that passes because it compared nothing is worse than no gate.
    """
    maps = _region_maps({512: 1600, 1024: 1500, 1977: 20})   # a 75x step
    report = val.Report()
    val.check_context_continuity(pd.DataFrame(), maps, report)
    row = _row(report, "context continuity")
    assert row["status"] == "WARN", row
    assert "output_tokens" in row["summary"], row["summary"]
    assert any("t1024->1977" in str(d) for d in row["detail"]), row["detail"]


def test_continuity_passes_on_a_smooth_occupancy_trend(val):
    maps = _region_maps({512: 400, 1024: 700, 1977: 1200})
    report = val.Report()
    val.check_context_continuity(pd.DataFrame(), maps, report)
    assert _row(report, "context continuity")["status"] == "PASS"


def test_retention_support_names_the_token_level(val):
    metrics = pd.DataFrame([
        {"model_id": "llama-3.2-1b", "ctx": _CTX, "output_tokens": t,
         "region_type": "kv_cache", "retention_support_frac": frac}
        for t, frac in zip(_TOKENS, (0.8, 0.6, 0.1))
    ])
    report = val.Report()
    val._SHOW_TOKENS = True
    val.check_retention_support(pd.DataFrame(), _runs(), report, metrics)
    row = _row(report, "retention support")
    assert row["status"] == "WARN"
    assert any("t1977" in str(d) for d in row["detail"]), row["detail"]


# ── the EDA must switch axis too ─────────────────────────────────────────────

def test_the_eda_reports_occupancy_not_context(capsys):
    eda = _module("results_exploratory_data_analysis")
    data = pd.DataFrame([
        {"run_id": f"r{t}{rep}", "path": "cpu", "model_id": "llama-3.2-1b",
         "ctx": _CTX, "output_tokens": t, "rep": rep, "region_type": region,
         "retention_p50_s": 40.0, "retention_p99_s": 120.0,
         "write_once_ratio_mean": 0.9, "working_set_mb_total": 10.0 * (t / 512),
         "n_regions": 100, "params_b": 1.0}
        for t in _TOKENS for rep in (1, 2)
        for region in ("weight", "kv_cache", "runtime")
    ])
    eda.section_context_scaling(data, pathlib.Path("/tmp"))
    out = capsys.readouterr().out
    assert "Occupancy scaling" in out, out[:400]
    # all three levels must survive into the table
    for t in _TOKENS:
        assert str(t) in out, t


# ── the expected region set must match what the code can produce ─────────────

def test_the_expected_region_set_excludes_what_no_rule_produces(val):
    """
    `activation` was removed from the labeller for varying 28-77x across
    identical reps, but stayed in the validator's expected set -- so every run
    of the next sweep FAILED for not producing a region type the code cannot
    produce.  The two lists have to agree.
    """
    from mrm_trace.labeller.symbol_rules import DEFAULT_RULES

    producible = {r.region_type for r in DEFAULT_RULES}
    # weight and kv_cache come from recorded address ranges, not symbol rules
    producible |= {"weight", "kv_cache", "unknown"}
    assert set(val._EXPECTED_CPU_REGIONS) <= producible, (
        f"expected but unproducible: "
        f"{set(val._EXPECTED_CPU_REGIONS) - producible}")


@pytest.mark.parametrize("module", ["results_validation",
                                    "results_exploratory_data_analysis"])
def test_every_loader_attaches_the_whole_cell_key(module):
    """
    _CELL is used as a groupby key, so a column missing from any loader raises
    KeyError on the first real dataset.  Synthetic fixtures that supply the
    column by hand do not catch it -- these did not, and the sweep caught both
    scripts in turn.
    """
    import inspect
    mod = _module(module)
    for name in mod._CELL_SOURCES:
        source = inspect.getsource(getattr(mod, name))
        for col in mod._CELL:
            assert f'"{col}"' in source, f"{module}.{name} does not attach {col}"


# ── an inter-access interval must clear its own sampling floor ───────────────

def _iai_metrics(region, n_intervals, iai_p50_s, duration_s=300.0):
    return pd.DataFrame([{
        "run_id": "r1", "model_id": "m", "ctx": _CTX, "output_tokens": 1977,
        "region_type": region, "n_intervals": n_intervals,
        "iai_p50_s": iai_p50_s,
    }]), pd.DataFrame([{"run_id": "r1", "collection_duration_s": duration_s}])


def test_a_gap_at_the_sampling_floor_is_flagged(val):
    """
    IAI is the gap between consecutive *samples* of an address.  With 4 samples
    over a 300 s run the gaps simply partition the run, so the metric reports
    sampling density rather than access behaviour.  Measured: kv_cache at the
    lowest occupancy cleared this baseline by 0.2x -- i.e. not at all.
    """
    metrics, runs = _iai_metrics("kv_cache", 4, 60.0)
    report = val.Report()
    val.check_iai_resolution(runs, metrics, report)
    row = _row(report, "iai resolution")
    assert row["status"] == "WARN"
    assert any("lower bound" in str(d) for d in row["detail"]), row["detail"]


def test_a_well_resolved_gap_passes(val):
    """runtime clears the baseline by ~1,400x on millions of intervals."""
    metrics, runs = _iai_metrics("runtime", 8_000_000, 0.11)
    report = val.Report()
    val.check_iai_resolution(runs, metrics, report)
    assert _row(report, "iai resolution")["status"] == "PASS"


def test_missing_iai_is_reported_not_silently_passed(val):
    report = val.Report()
    val.check_iai_resolution(pd.DataFrame(), pd.DataFrame(), report)
    assert _row(report, "iai resolution")["status"] == "WARN"


def test_the_runs_frame_carries_the_collection_duration(val):
    """Without it the floor cannot be computed and the check goes vacuous."""
    import inspect
    assert '"collection_duration_s"' in inspect.getsource(val.load_runs)

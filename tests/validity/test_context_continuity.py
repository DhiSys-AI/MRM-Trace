"""
Nothing validated the context dimension, which is the one the claims are read along.

Every other check in results_validation.py compares *repetitions within a cell*.
The v1.5 sweep contained two 36-44x steps in `runtime` region count between
adjacent context settings of one model — reproducible across both reps of each
cell, so the reproducibility check saw two stable cells and passed, while every
other step in the matrix was under 2.4x:

    llama-3.2-3b runtime  ctx8192 -> 32768 : 18,132 -> 416   (43.6x)
    phi-3.5-mini runtime  ctx2048 -> 8192  : 18,859 -> 520   (36.3x)

A step like that does not say which cell is wrong, only that the pair cannot both
be describing the same quantity. These tests pin the check that says so.
"""

import importlib.util
import pathlib

import pandas as pd
import pytest

pytestmark = pytest.mark.validity


def _validator():
    path = (pathlib.Path(__file__).resolve().parents[2]
            / "notebooks" / "scripts" / "results_validation.py")
    spec = importlib.util.spec_from_file_location("results_validation", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _maps(spec):
    """spec: {(model, ctx, region_type): regions_per_run} over two reps."""
    rows = []
    for (model, ctx, region), n in spec.items():
        for rep in (1, 2):
            run_id = f"{model}_ctx{ctx}_rep{rep}"
            rows += [{"run_id": run_id, "model_id": model, "ctx": ctx,
                      "region_type": region, "access_count": 5}] * n
    return pd.DataFrame(rows)


def _row(report, name):
    for r in report.results:
        if r["check"] == name:
            return r
    raise AssertionError(
        f"{name} not in report: {[r['check'] for r in report.results]}")


def _status(report, name):
    return _row(report, name)["status"]


def _text(report, name):
    r = _row(report, name)
    return r["summary"] + " " + " ".join(str(d) for d in r["detail"])


def test_a_smooth_trend_passes():
    v = _validator()
    maps = _maps({("m", 2048, "runtime"): 400,
                  ("m", 8192, "runtime"): 300,
                  ("m", 32768, "runtime"): 180})
    report = v.Report()
    v.check_context_continuity(pd.DataFrame(), maps, report)
    assert _status(report, "context continuity") == "PASS"


def test_the_shipped_discontinuity_is_caught():
    v = _validator()
    maps = _maps({("phi-3.5-mini", 2048, "runtime"): 18859,
                  ("phi-3.5-mini", 8192, "runtime"): 520,
                  ("phi-3.5-mini", 32768, "runtime"): 275})
    report = v.Report()
    v.check_context_continuity(pd.DataFrame(), maps, report)
    assert _status(report, "context continuity") == "WARN"
    assert "36.3x" in _text(report, "context continuity")


def test_counts_are_per_run_not_summed_over_reps():
    """
    Grouping straight to the cell sums the repetitions and reports a region
    count no run actually had — 37,718 for a cell whose runs held 18,859.
    """
    v = _validator()
    maps = _maps({("m", 2048, "runtime"): 18859,
                  ("m", 8192, "runtime"): 520})
    report = v.Report()
    v.check_context_continuity(pd.DataFrame(), maps, report)
    text = _text(report, "context continuity")
    assert "18,859" in text
    assert "37,718" not in text


def test_small_region_counts_are_not_reported_as_steps():
    """1 -> 25 regions is a 25x ratio and small-number noise, not a step."""
    v = _validator()
    maps = _maps({("m", 2048, "weight"): 1, ("m", 8192, "weight"): 19})
    report = v.Report()
    v.check_context_continuity(pd.DataFrame(), maps, report)
    assert _status(report, "context continuity") == "PASS"


def test_an_inferred_region_does_not_raise_the_status():
    """
    `activation` is already declared unstable; a step there is expected and must
    not be mixed in with a grounded region's discontinuity.
    """
    v = _validator()
    maps = _maps({("m", 2048, "activation"): 2000,
                  ("m", 8192, "activation"): 40})
    report = v.Report()
    v.check_context_continuity(pd.DataFrame(), maps, report)
    assert _status(report, "context continuity") == "PASS"

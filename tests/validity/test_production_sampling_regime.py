"""
One fixture at the regime the instrument actually runs in.

Three defects shipped in the v1.5 sweep despite 967 tests, and all three shared
one cause: every fixture in the suite was *densely* sampled while the sweep was
not.  The Tier-3 fixtures build five samples per region; production had **one**
sample for 70-79% of regions at ctx 32768, and 99.0% of KV blocks never had a
store sampled at all, because loads and stores run on independent periods with
the store period 500x coarser.

So this file builds a trace at the production regime and pushes it through the
whole analyser.  The rule it enforces is one rule: **every published number is
either supported by at least two observations or explicitly absent.**  A
derivation that imputes a value for an unobserved event fails here regardless of
which derivation it is, which is what the per-defect tests cannot do.

Regime, measured from the sweep:
  - load period : store period = 1 : 500
  - ~75% of regions sampled exactly once
  - runtime stores outnumber kv_cache stores ~63,000 : 1
  - monotonic timestamps in the 10^14 ns range (~46 h of uptime)
"""

import numpy as np
import pandas as pd
import pytest

from mrm_trace.analyser import compute_all
from mrm_trace.analyser.retention import compute_retention
from mrm_trace.labeller.kv_lifecycle import KVBlockTracker
from mrm_trace.reporter.metrics_csv import build_metrics_table

pytestmark = pytest.mark.validity

# The real sampler emits absolute monotonic timestamps, not small integers.  A
# fixture starting at 0 cannot reveal an imputed write time, because 0 is both
# "unknown" and a plausible first timestamp.
_CLOCK0 = 165_780_000_000_000          # ~46 h of uptime, as observed
_S = 1_000_000_000
_STORE_RATIO = 500                     # store period / load period
_SINGLE_SAMPLE_FRACTION = 0.75

_KV_BASE = 0x7F10_0000_0000
_RT_BASE = 0x7F20_0000_0000
_W_BASE = 0x7F00_0000_0000
_GRANULE = 65536


def _production_trace(n_regions_per_type=400, seed=7):
    """
    A trace with production sparsity: most regions sampled once, stores 500x
    rarer than loads, and KV stores rarer still.
    """
    rng = np.random.default_rng(seed)
    rows = []
    sample_id = 0
    ts = _CLOCK0

    def emit(region_id, address, op_type, when):
        nonlocal sample_id
        rows.append({
            "run_id": "regime", "sample_id": sample_id, "timestamp_ns": when,
            "phase": "generation", "address": address,
            "address_page": address >> 12, "op_type": op_type,
            "symbol": "", "dso": "", "region_type": None, "region_id": region_id,
        })
        sample_id += 1

    region_id = 1
    for region_type, base in (("weight", _W_BASE), ("kv_cache", _KV_BASE),
                              ("runtime", _RT_BASE)):
        for i in range(n_regions_per_type):
            address = base + i * _GRANULE
            single = rng.random() < _SINGLE_SAMPLE_FRACTION
            n_loads = 1 if single else int(rng.integers(2, 40))
            for k in range(n_loads):
                ts += 37_000_000
                emit(region_id, address + k * 64, "load", ts)
            # Stores are 500x rarer; weight is read-only (mapped r--s), and a
            # KV store is rarer again because runtime dominates the store stream.
            if region_type == "runtime" and not single and rng.random() < 1 / 4:
                ts += 11_000_000
                emit(region_id, address, "store", ts)
            elif region_type == "kv_cache" and rng.random() < 1 / _STORE_RATIO:
                ts += 11_000_000
                emit(region_id, address, "store", ts)
            region_id += 1

    frame = pd.DataFrame(rows)
    # region_type is assigned by address band, standing in for the labeller.
    frame["region_type"] = np.where(
        frame["address"] >= _RT_BASE, "runtime",
        np.where(frame["address"] >= _KV_BASE, "kv_cache", "weight"))
    return frame


@pytest.fixture(scope="module")
def trace():
    return _production_trace()


@pytest.fixture(scope="module")
def analysis(trace):
    return compute_all(trace)


# ── the fixture must actually be sparse, or it tests nothing ─────────────────

def test_the_fixture_reproduces_production_sparsity(trace):
    per_region = trace.groupby("region_id").size()
    single = (per_region == 1).mean()
    assert 0.6 <= single <= 0.9, f"single-sample fraction {single:.0%}"

    ops = trace.groupby("op_type").size()
    assert ops.get("store", 0) > 0
    assert ops["load"] / ops["store"] > 20, "stores are not rare enough"

    kv = trace[trace.region_type == "kv_cache"]
    assert (kv.op_type == "store").sum() < 10, "KV stores are not rare enough"
    assert (trace[trace.region_type == "weight"].op_type == "store").sum() == 0


# ── the one rule: supported, or absent ───────────────────────────────────────

def test_no_retention_is_reported_for_a_single_sample_region(trace):
    per_region, _ = compute_retention(trace)
    unsupported = per_region[per_region["n_samples"] < 2]
    assert len(unsupported) > 100, "fixture lost its sparsity"
    assert unsupported["retention_s"].isna().all(), (
        "a region sampled once was given a retention value")


def test_a_retention_percentile_never_exceeds_the_observed_window(trace, analysis):
    """
    The bound that catches an imputed timestamp: no derived duration may exceed
    the span of the trace it came from.  An absent write coerced to 0 produced
    165,781 s of "lifetime" inside a 73 s run.
    """
    window_s = (trace["timestamp_ns"].max() - trace["timestamp_ns"].min()) / 1e9
    summary = analysis["retention_summary"]
    for col in ("retention_p50_s", "retention_p90_s", "retention_p99_s",
                "retention_mean_s"):
        values = summary[col].dropna()
        assert (values <= window_s + 1e-6).all(), (
            f"{col} exceeds the {window_s:,.1f} s observed window: "
            f"{values.max():,.1f} s")


def test_every_published_percentile_declares_its_support(analysis):
    summary = analysis["retention_summary"]
    assert {"n_regions_observable", "retention_support_frac"} <= set(summary.columns)
    for _, row in summary.iterrows():
        if pd.notna(row["retention_p50_s"]):
            assert row["n_regions_observable"] >= 1, row["region_type"]
        else:
            assert row["n_regions_observable"] == 0, row["region_type"]


def test_kv_block_lifetime_is_absent_where_the_write_was_not_sampled(trace):
    tracker = KVBlockTracker()
    kv = trace[trace.region_type == "kv_cache"]
    for r in kv.itertuples():
        tracker.record(address=r.address, address_page=r.address_page,
                       region_id=r.region_id, op_type=r.op_type,
                       timestamp_ns=r.timestamp_ns,
                       block_key=r.address // _GRANULE)
    rows = tracker.lifecycle_rows()
    assert rows, "no KV blocks tracked"
    unobserved = [r for r in rows if not r["write_observed"]]
    assert len(unobserved) > len(rows) * 0.5, "fixture lost its KV sparsity"
    assert all(r["lifetime_ns"] is None for r in unobserved)
    # And no lifetime may be a clock reading.
    window = trace["timestamp_ns"].max() - trace["timestamp_ns"].min()
    for r in rows:
        if r["lifetime_ns"] is not None:
            assert r["lifetime_ns"] <= window, r


def test_a_read_only_region_is_unbounded_not_zero(analysis):
    """
    `weight` receives no stores because the GGUF is mapped r--s.  Zero writes
    with many reads is write-once in the limit -- the strongest read-many signal
    -- and scoring it 0.0 made it the weakest.
    """
    rf = analysis["read_freq"]
    weight = rf[rf["region_type"] == "weight"]
    assert len(weight) > 0
    assert (weight["total_writes"] == 0).all()
    assert np.isinf(weight["reads_per_write"]).all()


def test_an_unmeasurable_region_is_not_scored_low_mrm(analysis):
    suit = analysis["suitability"]
    for _, row in suit.iterrows():
        if pd.isna(row["retention_p50_s"]) and pd.isna(row["retention_p99_s"]):
            assert row["suitability"] == "unknown", row["region_type"]


def test_metrics_csv_carries_no_imputed_zero(analysis):
    """
    A published 0.0 must come from two observations at one instant, never from
    one observation.  This is the assertion the auditor's `np.isfinite` check
    could not make: 0.0 is finite.
    """
    table = build_metrics_table(analysis)
    ret = analysis["retention_per_region"]
    single_sample_ids = set(ret[ret["n_samples"] < 2]["region_id"])
    assert single_sample_ids, "fixture lost its sparsity"

    for _, row in table.iterrows():
        if row.get("retention_p50_s") == 0.0:
            observable = ret[(ret["region_type"] == row["region_type"])
                             & ret["retention_observable"]]
            assert len(observable) > 0, (
                f"{row['region_type']} reports p50 0.0 with nothing observable")

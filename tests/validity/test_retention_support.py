"""
A retention percentile must rest on regions that could be measured.

Retention is ``last access - first access`` of a region, so a region sampled
*once* has no interval: both ends are the same observation.  That was reported
as 0.0, and because samples-per-region falls as context grows, the imputed
zeros took over the median precisely where the context trend is read off.
Measured across the v1.5 sweep (CPU, analysis set, pooled):

    region      ctx   sampled once    p50 as reported    p50 over observable
    kv_cache   2048         42.8%            10.04 s             66.95 s
    kv_cache  32768         79.3%             0.00 s             44.91 s
    weight     2048         27.6%            49.46 s             80.45 s
    weight    32768         70.4%             0.00 s             37.30 s

Retention does fall with context -- by 1.5-2.2x over a 16x increase -- but not
to zero, and the absolute values were understated several-fold.  This is the
same defect class as an unsampled KV write becoming a 46-hour lifetime: an
absent measurement published as a number.
"""

import json

import numpy as np
import pandas as pd
import pytest

from mrm_trace.analyser.retention import compute_retention
from mrm_trace.analyser.rederive import rederive_metrics
from mrm_trace.analyser.suitability import classify_suitability
from mrm_trace.reporter.metrics_csv import build_metrics_table
from mrm_trace.analyser import compute_all

pytestmark = pytest.mark.validity

_S = 1_000_000_000


def _row(region_id, op_type, ts, region_type="weight", address=None):
    address = address if address is not None else 0x1000 * region_id
    return {"region_id": region_id, "region_type": region_type,
            "op_type": op_type, "timestamp_ns": ts,
            "address": address, "address_page": address >> 12}


def test_a_single_sample_region_reports_no_retention():
    per_region, _ = compute_retention(pd.DataFrame([_row(1, "load", 5 * _S)]))
    assert per_region.iloc[0]["n_samples"] == 1
    assert pd.isna(per_region.iloc[0]["retention_s"])


def test_imputed_zeros_no_longer_drag_the_median():
    """
    One region held 40 s; nineteen were sampled once.  The old rule returned a
    median of 0.0 s from a dataset whose only measurable region held 40.
    """
    rows = [_row(1, "store", 0), _row(1, "load", 40 * _S)]
    rows += [_row(rid, "load", 7 * _S) for rid in range(2, 21)]
    _, summary = compute_retention(pd.DataFrame(rows))
    r = summary.iloc[0]
    assert r["n_regions"] == 20
    assert r["n_regions_observable"] == 1
    assert r["retention_p50_s"] == pytest.approx(40.0)
    assert r["retention_support_frac"] == pytest.approx(0.05)


def test_metrics_csv_states_what_the_percentile_rests_on():
    """
    A p50 over 1 region and a p50 over 10,000 are different claims; the table
    has to be able to tell them apart.
    """
    rows = [_row(1, "store", 0), _row(1, "load", 40 * _S)]
    rows += [_row(rid, "load", 7 * _S) for rid in range(2, 21)]
    table = build_metrics_table(compute_all(pd.DataFrame(rows)))
    assert "n_regions_observable" in table.columns
    assert "retention_support_frac" in table.columns
    row = table[table["region_type"] == "weight"].iloc[0]
    assert row["n_regions_observable"] == 1
    assert row["retention_support_frac"] == pytest.approx(0.05)


def test_an_unmeasurable_region_type_is_unknown_not_low_mrm():
    """
    With retention null, every threshold comparison is False, which classified
    an unmeasured type as `low_mrm` -- the worst verdict, on no evidence.
    """
    ret = pd.DataFrame([{
        "region_type": "kv_cache", "n_regions": 10, "n_regions_observable": 0,
        "retention_support_frac": 0.0, "retention_p50_s": np.nan,
        "retention_p90_s": np.nan, "retention_p99_s": np.nan,
        "retention_mean_s": np.nan,
    }])
    wo = pd.DataFrame([{"region_id": 1, "region_type": "kv_cache",
                        "write_once_ratio": 1.0}])
    rf = pd.DataFrame([{"region_id": 1, "region_type": "kv_cache",
                        "total_reads": 10, "total_writes": 2}])
    out = classify_suitability(ret, wo, rf)
    assert out.iloc[0]["suitability"] == "unknown"


def test_rederived_metrics_equal_a_live_analysis(tmp_path):
    """
    The correction is applied by re-deriving from trace.parquet, so the
    re-derived file must equal what the pipeline computes from the same rows.
    Otherwise a re-derived dataset is a patched one.
    """
    rows = [_row(1, "store", 0), _row(1, "load", 40 * _S)]
    rows += [_row(rid, "load", 7 * _S) for rid in range(2, 21)]
    rows += [_row(rid, "store", 3 * _S, region_type="kv_cache",
                  address=0x7F00_0000 + 0x1000 * rid) for rid in range(21, 30)]
    rows += [_row(rid, "load", 9 * _S, region_type="kv_cache",
                  address=0x7F00_0000 + 0x1000 * rid) for rid in range(21, 30)]
    trace = pd.DataFrame(rows)

    d = tmp_path / "run"
    d.mkdir()
    trace.to_parquet(d / "trace.parquet")
    (d / "metadata.json").write_text(json.dumps({"collector": {}}))

    rederive_metrics(d)
    got = pd.read_csv(d / "metrics.csv")
    want = build_metrics_table(compute_all(trace))

    assert got["region_type"].tolist() == want["region_type"].tolist()
    for col in ("n_regions", "n_regions_observable", "retention_support_frac",
                "retention_p50_s", "retention_p99_s", "suitability"):
        a = got[col].tolist()
        b = want[col].tolist()
        if got[col].dtype.kind in "if":
            assert np.allclose(np.nan_to_num(np.array(a, dtype=float), nan=-1.0),
                               np.nan_to_num(np.array(b, dtype=float), nan=-1.0)), col
        else:
            assert a == b, col


def test_two_samples_at_one_instant_are_a_measured_zero():
    """A measured zero is still a measurement; only one sample is not."""
    per_region, _ = compute_retention(pd.DataFrame([
        _row(1, "store", 500), _row(1, "load", 500),
    ]))
    assert bool(per_region.iloc[0]["retention_observable"])
    assert per_region.iloc[0]["retention_s"] == 0.0

"""Unit tests for compute_retention."""

import pandas as pd
import pytest

from mrm_trace.analyser.retention import compute_retention


def _trace(*rows):
    return pd.DataFrame(rows)


def _row(region_id, region_type, op_type, timestamp_ns, address=0x1000):
    return {
        "region_id": region_id,
        "region_type": region_type,
        "op_type": op_type,
        "timestamp_ns": timestamp_ns,
        "address": address,
        "address_page": address >> 12,
    }


# ── basic retention ───────────────────────────────────────────────────────────

def test_retention_single_region():
    trace = _trace(
        _row(1, "weight", "store", 0),
        _row(1, "weight", "load",  1_000_000_000),
    )
    per_region, _ = compute_retention(trace)
    assert len(per_region) == 1
    assert per_region.iloc[0]["retention_s"] == pytest.approx(1.0)


def test_retention_ns_matches_s():
    trace = _trace(
        _row(1, "weight", "store", 0),
        _row(1, "weight", "load",  5_000_000_000),
    )
    per_region, _ = compute_retention(trace)
    r = per_region.iloc[0]
    assert r["retention_ns"] == 5_000_000_000
    assert r["retention_s"] == pytest.approx(5.0)


def test_single_sample_region_has_no_measurable_retention():
    """
    One sample is one end of the interval, not both.  Reporting 0.0 here
    fabricated a measurement, and since sampling density per region falls as
    context grows, those zeros took over the median precisely where the context
    trend is read off: kv_cache p50 came out 0.00 s at ctx 32768 against
    44.91 s over the regions that were actually sampled twice.
    """
    trace = _trace(_row(1, "weight", "store", 100))
    per_region, _ = compute_retention(trace)
    row = per_region.iloc[0]
    assert row["n_samples"] == 1
    assert row["retention_observable"] is False or not row["retention_observable"]
    assert pd.isna(row["retention_s"])
    assert pd.isna(row["retention_ns"])


def test_two_samples_at_the_same_instant_are_a_measured_zero():
    """Two observations *can* bound a zero-length interval; one cannot."""
    trace = _trace(
        _row(1, "weight", "store", 100),
        _row(1, "weight", "load", 100),
    )
    per_region, _ = compute_retention(trace)
    row = per_region.iloc[0]
    assert bool(row["retention_observable"])
    assert row["retention_s"] == 0.0


def test_percentiles_exclude_unobservable_regions():
    """
    A region type whose regions were mostly sampled once must not have its
    percentiles dragged to zero by them.
    """
    rows = [_row(1, "weight", "store", 0), _row(1, "weight", "load", 10_000_000_000)]
    for rid in range(2, 20):                      # 18 single-sample regions
        rows.append(_row(rid, "weight", "load", 500, address=0x1000 * rid))
    _, summary = compute_retention(_trace(*rows))
    r = summary.iloc[0]
    assert r["n_regions"] == 19
    assert r["n_regions_observable"] == 1
    assert r["retention_p50_s"] == pytest.approx(10.0)
    assert r["retention_support_frac"] == pytest.approx(1 / 19)


def test_retention_uses_last_read_not_first():
    trace = _trace(
        _row(1, "weight", "store", 0),
        _row(1, "weight", "load",  100),
        _row(1, "weight", "load",  5_000_000_000),
    )
    per_region, _ = compute_retention(trace)
    assert per_region.iloc[0]["retention_s"] == pytest.approx(5.0)


def test_retention_uses_first_write_not_last():
    trace = _trace(
        _row(1, "weight", "store", 0),
        _row(1, "weight", "store", 1_000_000_000),
        _row(1, "weight", "load",  5_000_000_000),
    )
    per_region, _ = compute_retention(trace)
    assert per_region.iloc[0]["first_write_ns"] == 0


# ── multiple regions ──────────────────────────────────────────────────────────

def test_two_regions_tracked_independently():
    trace = _trace(
        _row(1, "weight",   "store", 0,          address=0x1000),
        _row(1, "weight",   "load",  10_000_000_000, address=0x1000),
        _row(2, "kv_cache", "store", 0,          address=0x2000),
        _row(2, "kv_cache", "load",  500_000_000, address=0x2000),
    )
    per_region, _ = compute_retention(trace)
    by_id = per_region.set_index("region_id")["retention_s"]
    assert by_id[1] == pytest.approx(10.0)
    assert by_id[2] == pytest.approx(0.5)


# ── summary ───────────────────────────────────────────────────────────────────

def test_summary_has_one_row_per_type():
    trace = _trace(
        _row(1, "weight",   "store", 0,          address=0x1000),
        _row(1, "weight",   "load",  10_000_000_000, address=0x1000),
        _row(2, "kv_cache", "store", 0,          address=0x2000),
        _row(2, "kv_cache", "load",  500_000_000, address=0x2000),
    )
    _, summary = compute_retention(trace)
    assert set(summary["region_type"]) == {"weight", "kv_cache"}


def test_summary_columns_present():
    trace = _trace(
        _row(1, "weight", "store", 0),
        _row(1, "weight", "load",  1_000_000_000),
    )
    _, summary = compute_retention(trace)
    for col in ("region_type", "n_regions", "retention_p50_s",
                "retention_p90_s", "retention_p99_s", "retention_mean_s"):
        assert col in summary.columns


def test_summary_n_regions_correct():
    trace = _trace(
        _row(1, "weight", "store", 0,              address=0x1000),
        _row(1, "weight", "load",  1_000_000_000,  address=0x1000),
        _row(2, "weight", "store", 0,              address=0x2000),
        _row(2, "weight", "load",  2_000_000_000,  address=0x2000),
    )
    _, summary = compute_retention(trace)
    assert summary[summary["region_type"] == "weight"].iloc[0]["n_regions"] == 2

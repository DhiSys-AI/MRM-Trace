"""
Retention is only estimable when the same region is sampled more than once.

Both granularity extremes produce artifacts that look like data:
  too fine   → one sample per region → retention == 0 by construction
  too coarse → every granule spans the whole run → retention == run duration

These tests pin the detection of both, so a run cannot silently report either.
"""

import pandas as pd
import pytest

from mrm_trace.analyser.granularity import (
    recommend_granularity,
    retention_granularity_sweep,
)

pytestmark = pytest.mark.validity

_S = 1_000_000_000  # ns


def _rows(addresses, timestamps, region_type="weight"):
    return pd.DataFrame({
        "address": addresses,
        "timestamp_ns": timestamps,
        "region_type": [region_type] * len(addresses),
    })


def test_one_sample_per_region_is_flagged_undersampled():
    """4 KiB apart, one sample each — retention is 0 for every region."""
    trace = _rows([i * 4096 for i in range(200)], [i * _S // 10 for i in range(200)])
    sweep = retention_granularity_sweep(trace, granules_bytes=[4096])
    row = sweep.iloc[0]
    assert row["samples_per_region"] < 2.0
    assert row["p50_s"] == 0.0
    assert row["verdict"] == "undersampled"


def test_granule_spanning_whole_run_is_flagged_saturated():
    """One huge granule touched at both ends measures run length, not retention."""
    trace = _rows([i * 4096 for i in range(200)], [i * _S // 10 for i in range(200)])
    sweep = retention_granularity_sweep(trace, granules_bytes=[1 << 30])
    row = sweep.iloc[0]
    assert row["n_regions"] == 1
    assert row["verdict"] == "saturated"


def test_repeated_access_at_matching_granule_is_usable():
    """Five samples per 64 KiB granule, each spanning a fraction of the run."""
    addresses, timestamps = [], []
    for region in range(40):
        for k in range(5):
            addresses.append(region * 65536 + k * 64)
            timestamps.append(region * _S // 20 + k * _S // 100)
    sweep = retention_granularity_sweep(_rows(addresses, timestamps),
                                        granules_bytes=[65536])
    row = sweep.iloc[0]
    assert row["samples_per_region"] == 5.0
    assert row["verdict"] == "usable"
    assert row["p50_s"] > 0


def test_recommend_picks_smallest_usable_granule():
    addresses, timestamps = [], []
    for region in range(40):
        for k in range(5):
            addresses.append(region * 65536 + k * 64)
            timestamps.append(region * _S // 20 + k * _S // 100)
    sweep = retention_granularity_sweep(_rows(addresses, timestamps))
    usable = sweep[sweep["verdict"] == "usable"]["granule_bytes"]
    assert recommend_granularity(sweep) == usable.min()


def test_recommend_returns_none_when_every_granule_is_an_artifact():
    """
    None is a real answer — the trace cannot support a per-region retention
    distribution and only aggregate figures should be reported.
    """
    trace = _rows([i * 4096 for i in range(50)], [0] * 50)
    sweep = retention_granularity_sweep(trace)
    assert recommend_granularity(sweep) is None


def test_region_type_filter_excludes_unknown_by_default():
    trace = pd.concat([
        _rows([0, 64], [0, _S], region_type="weight"),
        _rows([1 << 40, (1 << 40) + 64], [0, _S], region_type="unknown"),
    ])
    sweep = retention_granularity_sweep(trace, granules_bytes=[4096])
    assert sweep.iloc[0]["n_regions"] == 1


def test_empty_trace_returns_empty_frame_with_columns():
    empty = _rows([], []).astype({"timestamp_ns": "int64"})
    sweep = retention_granularity_sweep(empty)
    assert sweep.empty
    assert "verdict" in sweep.columns

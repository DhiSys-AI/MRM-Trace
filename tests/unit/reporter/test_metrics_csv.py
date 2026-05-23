"""Unit tests for metrics_csv."""

import pandas as pd
import pytest

from mrm_trace.reporter.metrics_csv import build_metrics_table, write_metrics_csv


def _minimal_analysis():
    return {
        "retention_summary": pd.DataFrame([
            {"region_type": "weight",   "n_regions": 1, "retention_p50_s": 30.0,
             "retention_p90_s": 30.0, "retention_p99_s": 30.0, "retention_mean_s": 30.0},
            {"region_type": "kv_cache", "n_regions": 10, "retention_p50_s": 0.05,
             "retention_p90_s": 0.1, "retention_p99_s": 0.1, "retention_mean_s": 0.05},
        ]),
        "write_once": pd.DataFrame([
            {"region_id": 1, "region_type": "weight",   "write_once_ratio": 1.0,
             "n_written_addresses": 1, "n_write_once_addresses": 1},
            {"region_id": 2, "region_type": "kv_cache", "write_once_ratio": 1.0,
             "n_written_addresses": 1, "n_write_once_addresses": 1},
        ]),
        "read_freq": pd.DataFrame([
            {"region_id": 1, "region_type": "weight",   "reads_per_write": 1000.0,
             "total_reads": 1000, "total_writes": 1, "total_accesses": 1001,
             "read_fraction": 0.999},
            {"region_id": 2, "region_type": "kv_cache", "reads_per_write": 5.0,
             "total_reads": 5, "total_writes": 1, "total_accesses": 6,
             "read_fraction": 0.833},
        ]),
        "working_set_summary": pd.DataFrame([
            {"region_type": "weight",   "n_regions": 1, "total_pages": 10,
             "total_bytes": 40960, "mean_pages_per_region": 10.0, "max_pages_per_region": 10},
            {"region_type": "kv_cache", "n_regions": 10, "total_pages": 10,
             "total_bytes": 40960, "mean_pages_per_region": 1.0, "max_pages_per_region": 1},
        ]),
        "suitability": pd.DataFrame([
            {"region_type": "weight",   "suitability": "high_mrm",
             "write_once_ratio_mean": 1.0, "retention_p99_s": 30.0, "retention_p50_s": 30.0},
            {"region_type": "kv_cache", "suitability": "low_mrm",
             "write_once_ratio_mean": 1.0, "retention_p99_s": 0.1, "retention_p50_s": 0.05},
        ]),
    }


def test_build_metrics_table_has_required_columns():
    df = build_metrics_table(_minimal_analysis())
    for col in ("region_type", "n_regions", "retention_p99_s",
                "write_once_ratio_mean", "reads_per_write_mean",
                "working_set_mb_total", "suitability"):
        assert col in df.columns


def test_build_metrics_table_one_row_per_type():
    df = build_metrics_table(_minimal_analysis())
    assert len(df) == 2
    assert set(df["region_type"]) == {"weight", "kv_cache"}


def test_build_metrics_table_sorted_by_region_type():
    df = build_metrics_table(_minimal_analysis())
    types = list(df["region_type"])
    assert types == sorted(types)


def test_weight_row_has_high_mrm_suitability():
    df = build_metrics_table(_minimal_analysis())
    row = df[df["region_type"] == "weight"].iloc[0]
    assert row["suitability"] == "high_mrm"


def test_write_metrics_csv_creates_file(tmp_path):
    path = write_metrics_csv(_minimal_analysis(), tmp_path / "metrics.csv")
    assert path.exists()
    assert path.name == "metrics.csv"


def test_write_metrics_csv_is_readable_as_dataframe(tmp_path):
    import pandas as pd
    write_metrics_csv(_minimal_analysis(), tmp_path / "metrics.csv")
    df = pd.read_csv(tmp_path / "metrics.csv")
    assert "region_type" in df.columns
    assert len(df) >= 1


def test_working_set_mb_total_in_megabytes():
    df = build_metrics_table(_minimal_analysis())
    # 40960 bytes = 40960 / 1048576 MB ≈ 0.039
    assert df["working_set_mb_total"].iloc[0] == pytest.approx(40960 / 1_048_576, rel=1e-3)

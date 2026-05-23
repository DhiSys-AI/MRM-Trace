"""Unit tests for figures module — graceful on missing matplotlib."""

import pandas as pd
import pytest

from mrm_trace.reporter.figures import (
    _HAS_MPL,
    generate_all_figures,
    plot_retention_histogram,
    plot_working_set,
    plot_write_once,
)


def _retention_df():
    return pd.DataFrame([
        {"region_id": 1, "region_type": "weight",   "retention_s": 30.0,
         "retention_ns": 30_000_000_000, "first_write_ns": 0, "last_read_ns": 30_000_000_000},
        {"region_id": 2, "region_type": "kv_cache", "retention_s": 0.05,
         "retention_ns": 50_000_000,     "first_write_ns": 0, "last_read_ns": 50_000_000},
    ])


def _ws_summary():
    return pd.DataFrame([
        {"region_type": "weight",   "n_regions": 1, "total_pages": 10,
         "total_bytes": 40960, "mean_pages_per_region": 10.0, "max_pages_per_region": 10},
        {"region_type": "kv_cache", "n_regions": 5, "total_pages": 5,
         "total_bytes": 20480, "mean_pages_per_region": 1.0, "max_pages_per_region": 1},
    ])


def _wo_df():
    return pd.DataFrame([
        {"region_id": 1, "region_type": "weight",   "write_once_ratio": 1.0,
         "n_written_addresses": 1, "n_write_once_addresses": 1},
        {"region_id": 2, "region_type": "kv_cache", "write_once_ratio": 1.0,
         "n_written_addresses": 1, "n_write_once_addresses": 1},
    ])


@pytest.mark.skipif(not _HAS_MPL, reason="matplotlib not installed")
def test_plot_retention_histogram_creates_file(tmp_path):
    p = plot_retention_histogram(_retention_df(), tmp_path)
    assert p is not None
    assert p.exists()
    assert p.suffix == ".png"


@pytest.mark.skipif(not _HAS_MPL, reason="matplotlib not installed")
def test_plot_working_set_creates_file(tmp_path):
    p = plot_working_set(_ws_summary(), tmp_path)
    assert p is not None
    assert p.exists()


@pytest.mark.skipif(not _HAS_MPL, reason="matplotlib not installed")
def test_plot_write_once_creates_file(tmp_path):
    p = plot_write_once(_wo_df(), tmp_path)
    assert p is not None
    assert p.exists()


@pytest.mark.skipif(not _HAS_MPL, reason="matplotlib not installed")
def test_generate_all_figures_returns_list(tmp_path):
    analysis = {
        "retention_per_region": _retention_df(),
        "working_set_summary": _ws_summary(),
        "write_once": _wo_df(),
    }
    result = generate_all_figures(analysis, tmp_path)
    assert isinstance(result, list)
    assert len(result) == 3
    for p in result:
        assert p.exists()


def test_plot_functions_return_none_when_no_matplotlib(tmp_path, monkeypatch):
    monkeypatch.setattr("mrm_trace.reporter.figures._HAS_MPL", False)
    assert plot_retention_histogram(_retention_df(), tmp_path) is None
    assert plot_working_set(_ws_summary(), tmp_path) is None
    assert plot_write_once(_wo_df(), tmp_path) is None


def test_generate_all_figures_returns_empty_when_no_matplotlib(tmp_path, monkeypatch):
    monkeypatch.setattr("mrm_trace.reporter.figures._HAS_MPL", False)
    analysis = {
        "retention_per_region": _retention_df(),
        "working_set_summary": _ws_summary(),
        "write_once": _wo_df(),
    }
    result = generate_all_figures(analysis, tmp_path)
    assert result == []

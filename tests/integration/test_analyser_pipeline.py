"""
Integration tests for the full analyser pipeline.

Uses the synthetic_trace_factory conftest fixture (pre-labelled ground truth)
and verifies that compute_all() produces structurally valid, internally
consistent output.
"""

import pytest
import pandas as pd

from mrm_trace.analyser import compute_all


class TestComputeAll:

    @pytest.fixture(autouse=True)
    def trace(self, synthetic_trace_factory):
        self.trace = synthetic_trace_factory()
        self.results = compute_all(self.trace)

    def test_returns_all_expected_keys(self):
        expected = {
            "retention_per_region", "retention_summary",
            "write_once", "read_freq",
            "working_set_per_region", "working_set_summary",
            "locality_per_region", "locality_summary",
            "iai", "suitability",
        }
        assert expected == set(self.results.keys())

    def test_retention_per_region_has_weight_and_kv(self):
        types = set(self.results["retention_per_region"]["region_type"])
        assert "weight" in types
        assert "kv_cache" in types

    def test_write_once_has_weight_and_kv(self):
        types = set(self.results["write_once"]["region_type"])
        assert "weight" in types
        assert "kv_cache" in types

    def test_suitability_covers_both_types(self):
        types = set(self.results["suitability"]["region_type"])
        assert "weight" in types
        assert "kv_cache" in types

    def test_iai_has_kv_cache_row(self):
        types = set(self.results["iai"]["region_type"])
        assert "kv_cache" in types

    def test_all_region_ids_in_write_once_match_trace(self):
        trace_ids = set(self.trace["region_id"].unique())
        result_ids = set(self.results["write_once"]["region_id"].unique())
        # write_once only covers regions that have stores; all such ids should be a subset
        assert result_ids.issubset(trace_ids)

    def test_retention_ns_nonnegative(self):
        assert (self.results["retention_per_region"]["retention_ns"] >= 0).all()

    def test_write_once_ratio_between_0_and_1(self):
        ratios = self.results["write_once"]["write_once_ratio"]
        assert (ratios >= 0.0).all() and (ratios <= 1.0).all()

    def test_read_fraction_between_0_and_1(self):
        fracs = self.results["read_freq"]["read_fraction"]
        assert (fracs >= 0.0).all() and (fracs <= 1.0).all()

    def test_working_set_bytes_positive(self):
        assert (self.results["working_set_per_region"]["working_set_bytes"] > 0).all()

    def test_suitability_values_are_valid_labels(self):
        valid = {"high_mrm", "medium_mrm", "low_mrm"}
        assert set(self.results["suitability"]["suitability"]).issubset(valid)

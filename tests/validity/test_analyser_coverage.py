"""
Tier-3 validity tests for Phase 6 — Analysis.

These tests verify scientific correctness: known synthetic inputs must produce
known metric outputs. A violation means the metric is wrong, not just broken.
"""

import pytest

from mrm_trace.analyser import (
    compute_retention,
    compute_write_once,
    compute_read_freq,
    compute_iai,
    classify_suitability,
)


# ── weight region ground truth ────────────────────────────────────────────────

class TestWeightRetentionGroundTruth:

    @pytest.fixture(autouse=True)
    def setup(self, synthetic_trace_factory):
        self.trace = synthetic_trace_factory(
            weight_retention_s=30.0,
            n_weight_reads=1000,
        )
        self.per_region, self.summary = compute_retention(self.trace)
        self.weight_row = self.per_region[self.per_region["region_type"] == "weight"].iloc[0]

    def test_weight_retention_p99_approximately_30s(self):
        """
        Weight region must report retention_p99_s ≈ 30.0 s.

        Phase 6 acceptance criterion: synthetic traces yield expected metrics.
        """
        weight_summary = self.summary[self.summary["region_type"] == "weight"].iloc[0]
        assert abs(weight_summary["retention_p99_s"] - 30.0) < 1.0, (
            f"retention_p99_s={weight_summary['retention_p99_s']:.3f}, expected ≈30.0. "
            "Retention analysis must capture the full weight-load-to-last-read window."
        )

    def test_weight_first_write_ns_is_zero(self):
        assert self.weight_row["first_write_ns"] == 0, (
            "Weight first_write_ns must be 0 (initial store at t=0). "
            "Using the wrong timestamp corrupts retention windowing."
        )

    def test_weight_retention_s_approximately_30(self):
        assert abs(self.weight_row["retention_s"] - 30.0) < 1.0, (
            f"Weight retention_s={self.weight_row['retention_s']:.3f}, expected ≈30.0."
        )


class TestWeightWriteOnceGroundTruth:

    @pytest.fixture(autouse=True)
    def setup(self, synthetic_trace_factory):
        trace = synthetic_trace_factory()
        self.wo = compute_write_once(trace)

    def test_weight_write_once_ratio_is_1(self):
        """
        The weight region is written exactly once (one store at weight_addr_base).
        write_once_ratio must equal 1.0.
        """
        w = self.wo[self.wo["region_type"] == "weight"].iloc[0]
        assert w["write_once_ratio"] == pytest.approx(1.0), (
            f"Weight write_once_ratio={w['write_once_ratio']:.3f}, expected 1.0. "
            "A weight region written more than once would corrupt write-once classification."
        )

    def test_kv_write_once_ratio_is_1(self):
        """Each KV block is written exactly once per conftest fixture."""
        kv = self.wo[self.wo["region_type"] == "kv_cache"]
        assert (kv["write_once_ratio"] == 1.0).all(), (
            "KV cache blocks must each have write_once_ratio=1.0 in the synthetic fixture."
        )


class TestWeightReadFreqGroundTruth:

    @pytest.fixture(autouse=True)
    def setup(self, synthetic_trace_factory):
        trace = synthetic_trace_factory(n_weight_reads=1000, n_kv_writes=100, n_kv_reads=500)
        self.rf = compute_read_freq(trace)

    def test_weight_reads_per_write_approximately_1000(self):
        """
        Weight region: 1 store, 1000 reads → reads_per_write ≈ 1000.

        A large reads_per_write for weights is the primary signal that weight
        memory is worth persisting in MRM.
        """
        w = self.rf[self.rf["region_type"] == "weight"].iloc[0]
        assert abs(w["reads_per_write"] - 1000.0) < 5.0, (
            f"Weight reads_per_write={w['reads_per_write']:.1f}, expected ≈1000.0."
        )

    def test_kv_reads_per_write_approximately_5(self):
        """KV: reads_per_block=5 → per-region reads_per_write must be ≈5."""
        kv = self.rf[self.rf["region_type"] == "kv_cache"]
        # All KV regions should have reads_per_write ≈ 5
        assert (abs(kv["reads_per_write"] - 5.0) < 0.5).all(), (
            f"KV reads_per_write values: {kv['reads_per_write'].tolist()}. "
            "Expected ≈5.0 per block (n_kv_reads / n_kv_writes = 500/100 = 5)."
        )


class TestKVRetentionGroundTruth:

    @pytest.fixture(autouse=True)
    def setup(self, synthetic_trace_factory):
        trace = synthetic_trace_factory(
            n_kv_writes=100,
            n_kv_reads=500,
            kv_retention_s=5.0,
        )
        per_region, self.summary = compute_retention(trace)

    def test_kv_retention_p50_much_less_than_weight(self):
        """KV blocks have short lifetimes compared to the 30s weight window."""
        kv_p50 = self.summary[self.summary["region_type"] == "kv_cache"].iloc[0]["retention_p50_s"]
        assert kv_p50 < 1.0, (
            f"KV retention_p50_s={kv_p50:.4f}, expected < 1.0. "
            "KV blocks have short per-block read windows (reads_per_block * 0.01s = 0.05s)."
        )


class TestIAIGroundTruth:

    @pytest.fixture(autouse=True)
    def setup(self, synthetic_trace_factory):
        trace = synthetic_trace_factory(n_kv_writes=100, n_kv_reads=500)
        self.iai = compute_iai(trace)

    def test_kv_iai_p50_approximately_10ms(self):
        """
        KV accesses to the same address are spaced 0.01s apart in the fixture.
        iai_p50_ns for kv_cache must be ≈ 10,000,000 ns (10ms).
        """
        kv = self.iai[self.iai["region_type"] == "kv_cache"].iloc[0]
        expected_ns = 10_000_000
        assert abs(kv["iai_p50_ns"] - expected_ns) / expected_ns < 0.05, (
            f"KV iai_p50_ns={kv['iai_p50_ns']}, expected ≈{expected_ns}. "
            "IAI must group by exact address (not page) to correctly measure "
            "per-block access intervals for sub-page 256-byte KV blocks."
        )


class TestSuitabilityGroundTruth:

    @pytest.fixture(autouse=True)
    def setup(self, synthetic_trace_factory):
        trace = synthetic_trace_factory(
            weight_retention_s=30.0,
            n_kv_writes=100,
            n_kv_reads=500,
        )
        per_region, ret_summary = compute_retention(trace)
        wo = compute_write_once(trace)
        rf = compute_read_freq(trace)
        self.suitability = classify_suitability(ret_summary, wo, rf)

    def test_weight_classified_as_high_mrm(self):
        """
        Weight must be high_mrm: retention_p99≈30s (≥10) AND write_once_ratio=1.0 (≥0.8).

        Phase 6 acceptance criterion: synthetic traces yield expected suitability.
        """
        w = self.suitability[self.suitability["region_type"] == "weight"].iloc[0]
        assert w["suitability"] == "high_mrm", (
            f"Weight suitability={w['suitability']!r}, expected 'high_mrm'. "
            "A 30s write-once region must classify as high_mrm — "
            "wrong classification would exclude it from MRM candidate analysis."
        )

    def test_kv_classified_as_low_mrm(self):
        """
        KV blocks must be low_mrm: retention_p50 ≈ 0.05s (< 1.0s threshold).

        Short-lived KV blocks are not good MRM candidates in the default fixture.
        """
        kv = self.suitability[self.suitability["region_type"] == "kv_cache"].iloc[0]
        assert kv["suitability"] == "low_mrm", (
            f"KV suitability={kv['suitability']!r}, expected 'low_mrm'. "
            "Short-lived KV blocks (retention_p50≈0.05s) must not be classified "
            "as medium_mrm or high_mrm."
        )

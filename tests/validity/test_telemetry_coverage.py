"""
Tier-3 validity tests for Phase 7 — Telemetry + validity.

Acceptance criterion: contaminated/clean fixture runs classify correctly.
"""

import pytest

from mrm_trace.telemetry.validity import classify_from_summary, classify_run


class TestCleanFixtureClassifiesCorrectly:
    """
    The conftest clean_metadata fixture has well-known low-overhead values.
    The classifier must agree with the fixture's ground truth.
    """

    def test_clean_fixture_classified_as_clean(self, clean_metadata):
        """
        clean_metadata has cpu=4.2%, mem=312 MB, throttle=0, io_contention=False.
        classify_from_summary must return 'clean'.

        Acceptance criterion: 'Contaminated/clean fixture runs classify correctly.'
        """
        summary = clean_metadata["observer_effect"]["observer_effect_summary"]
        # Provide a large target RSS so 312 MB is well within 5% threshold
        result = classify_from_summary(summary, target_rss_mb=100_000.0)
        assert result.run_validity == "clean", (
            f"Expected 'clean', got {result.run_validity!r}. "
            f"Failures: {result.failure_reasons}. "
            "The clean fixture (cpu=4.2%, throttle=0) must classify as clean."
        )

    def test_clean_fixture_has_no_failure_reasons(self, clean_metadata):
        summary = clean_metadata["observer_effect"]["observer_effect_summary"]
        result = classify_from_summary(summary, target_rss_mb=100_000.0)
        assert result.failure_reasons == [], (
            f"Clean fixture should have no failure reasons, got: {result.failure_reasons}"
        )

    def test_clean_fixture_cpu_below_threshold(self, clean_metadata):
        """cpu=4.2% is well below the 10% clean threshold."""
        summary = clean_metadata["observer_effect"]["observer_effect_summary"]
        assert summary["observer_cpu_overhead_pct_max"] < 10.0


class TestContaminatedFixtureClassifiesCorrectly:
    """
    The conftest contaminated_metadata fixture has high-overhead values.
    The classifier must return 'contaminated', not 'clean' or 'marginal'.
    """

    def test_contaminated_fixture_classified_as_contaminated(
        self, contaminated_metadata
    ):
        """
        contaminated_metadata has cpu=28.5%, throttle=5, io_contention=True.
        classify_from_summary must return 'contaminated'.

        Acceptance criterion: 'Contaminated/clean fixture runs classify correctly.'
        """
        summary = contaminated_metadata["observer_effect"]["observer_effect_summary"]
        result = classify_from_summary(summary)
        assert result.run_validity == "contaminated", (
            f"Expected 'contaminated', got {result.run_validity!r}. "
            "The contaminated fixture (cpu=28.5%, throttle=5, io=True) must "
            "not be allowed into the clean or marginal dataset."
        )

    def test_contaminated_fixture_has_failure_reasons(self, contaminated_metadata):
        summary = contaminated_metadata["observer_effect"]["observer_effect_summary"]
        result = classify_from_summary(summary)
        assert len(result.failure_reasons) > 0, (
            "Contaminated fixture must produce at least one failure reason "
            "so analysts can understand why the run was rejected."
        )

    def test_contaminated_fixture_not_classified_as_clean(self, contaminated_metadata):
        summary = contaminated_metadata["observer_effect"]["observer_effect_summary"]
        result = classify_from_summary(summary)
        assert result.run_validity != "clean", (
            "A contaminated run must never be classified as clean."
        )

    def test_contaminated_fixture_not_classified_as_marginal(
        self, contaminated_metadata
    ):
        summary = contaminated_metadata["observer_effect"]["observer_effect_summary"]
        result = classify_from_summary(summary)
        assert result.run_validity != "marginal", (
            "cpu=28.5% (> 20% marginal threshold) must push to contaminated, "
            "not marginal."
        )


class TestClassificationBoundaries:
    """
    Exact boundary tests ensure no off-by-one in the thresholds defined
    in CLAUDE.md. These tests protect against threshold drift during refactors.
    """

    def test_cpu_9_99_is_clean(self):
        r = classify_run(9.99, 0.0, 0, 0.0)
        assert r.run_validity == "clean"

    def test_cpu_10_0_is_not_clean(self):
        r = classify_run(10.0, 0.0, 0, 0.0)
        assert r.run_validity != "clean"

    def test_cpu_19_99_is_marginal(self):
        r = classify_run(19.99, 0.0, 0, 0.0)
        assert r.run_validity == "marginal"

    def test_cpu_20_0_is_contaminated(self):
        r = classify_run(20.0, 0.0, 0, 0.0)
        assert r.run_validity == "contaminated"

    def test_mem_frac_0_049_is_clean(self):
        r = classify_run(0.0, 0.049, 0, 0.0)
        assert r.run_validity == "clean"

    def test_mem_frac_0_05_is_not_clean(self):
        r = classify_run(0.0, 0.05, 0, 0.0)
        assert r.run_validity != "clean"

    def test_thermal_2_is_marginal(self):
        r = classify_run(0.0, 0.0, 2, 0.0)
        assert r.run_validity == "marginal"

    def test_thermal_3_is_contaminated(self):
        r = classify_run(0.0, 0.0, 3, 0.0)
        assert r.run_validity == "contaminated"

    def test_baseline_cpu_14_9_is_clean(self):
        r = classify_run(0.0, 0.0, 0, 14.9)
        assert r.run_validity == "clean"

    def test_baseline_cpu_15_0_breaks_clean(self):
        r = classify_run(0.0, 0.0, 0, 15.0)
        assert r.run_validity != "clean"

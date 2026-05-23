"""
Property-based tests for the validity classifier.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mrm_trace.telemetry.validity import classify_run

_VALID_LABELS = {"clean", "marginal", "contaminated"}


@given(
    cpu=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
    mem_frac=st.floats(min_value=0.0, max_value=2.0, allow_nan=False),
    throttle=st.integers(min_value=0, max_value=100),
    baseline=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
    io_c=st.booleans(),
)
@settings(max_examples=500)
def test_classify_run_always_returns_valid_label(cpu, mem_frac, throttle, baseline, io_c):
    """classify_run must always return one of the three valid labels for any input."""
    result = classify_run(cpu, mem_frac, throttle, baseline, io_c)
    assert result.run_validity in _VALID_LABELS


@given(
    cpu=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
    mem_frac=st.floats(min_value=0.0, max_value=2.0, allow_nan=False),
    throttle=st.integers(min_value=0, max_value=100),
    baseline=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
)
@settings(max_examples=300)
def test_clean_implies_no_failure_reasons(cpu, mem_frac, throttle, baseline):
    """A 'clean' run must have an empty failure_reasons list."""
    result = classify_run(cpu, mem_frac, throttle, baseline)
    if result.run_validity == "clean":
        assert result.failure_reasons == []


@given(
    cpu=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
    mem_frac=st.floats(min_value=0.0, max_value=2.0, allow_nan=False),
    throttle=st.integers(min_value=0, max_value=100),
    baseline=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
    io_c=st.booleans(),
)
@settings(max_examples=300)
def test_contaminated_implies_failure_reasons(cpu, mem_frac, throttle, baseline, io_c):
    """A 'contaminated' run must have at least one failure reason recorded."""
    result = classify_run(cpu, mem_frac, throttle, baseline, io_c)
    if result.run_validity == "contaminated":
        assert len(result.failure_reasons) > 0


@given(
    cpu=st.floats(min_value=0.0, max_value=9.99, allow_nan=False),
    mem_frac=st.floats(min_value=0.0, max_value=0.049, allow_nan=False),
    baseline=st.floats(min_value=0.0, max_value=14.99, allow_nan=False),
)
@settings(max_examples=200)
def test_all_clean_inputs_produce_clean(cpu, mem_frac, baseline):
    """When all criteria are strictly within clean bounds, result is always 'clean'."""
    result = classify_run(cpu, mem_frac, thermal_throttle_events=0,
                          baseline_cpu_pct=baseline, io_contention_detected=False)
    assert result.run_validity == "clean", (
        f"Expected clean but got {result.run_validity}: "
        f"cpu={cpu:.2f}, mem={mem_frac:.3f}, baseline={baseline:.2f}"
    )


@given(
    cpu=st.floats(min_value=20.0, max_value=100.0, allow_nan=False),
    mem_frac=st.floats(min_value=0.0, max_value=2.0, allow_nan=False),
    throttle=st.integers(min_value=0, max_value=100),
)
@settings(max_examples=200)
def test_cpu_above_marginal_threshold_is_contaminated(cpu, mem_frac, throttle):
    """CPU ≥ 20% always pushes to contaminated regardless of other metrics."""
    result = classify_run(cpu, mem_frac, throttle)
    assert result.run_validity == "contaminated"

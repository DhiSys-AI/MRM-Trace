"""Telemetry subsystem for mrm-trace — Phase 7."""

from mrm_trace.telemetry.baseline import BaselineSnapshot, capture_baseline
from mrm_trace.telemetry.observer_effect import compute_observer_effect
from mrm_trace.telemetry.thermal import ThermalMonitor, ThermalSnapshot
from mrm_trace.telemetry.validity import (
    ValidityResult,
    classify_from_summary,
    classify_run,
)

__all__ = [
    "BaselineSnapshot",
    "capture_baseline",
    "ThermalSnapshot",
    "ThermalMonitor",
    "compute_observer_effect",
    "ValidityResult",
    "classify_run",
    "classify_from_summary",
]

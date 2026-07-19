"""Telemetry subsystem for mrm-trace — Phase 7."""

from mrm_trace.telemetry.baseline import BaselineSnapshot, capture_baseline
from mrm_trace.telemetry.observer_effect import compute_observer_effect
from mrm_trace.telemetry.thermal import ThermalMonitor, ThermalSnapshot, read_package_temp_c
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
    "read_package_temp_c",
    "compute_observer_effect",
    "ValidityResult",
    "classify_run",
    "classify_from_summary",
]

"""Analysis subsystem for mrm-trace — Phase 6."""

from mrm_trace.analyser.iai import compute_iai
from mrm_trace.analyser.locality import compute_locality
from mrm_trace.analyser.read_freq import compute_read_freq
from mrm_trace.analyser.retention import compute_retention
from mrm_trace.analyser.suitability import classify_suitability
from mrm_trace.analyser.working_set import compute_working_set
from mrm_trace.analyser.write_once import compute_write_once

__all__ = [
    "compute_retention",
    "compute_write_once",
    "compute_read_freq",
    "compute_working_set",
    "compute_locality",
    "compute_iai",
    "classify_suitability",
    "compute_all",
]


def compute_all(trace) -> dict:
    """
    Run all analysis modules on a trace DataFrame.

    Returns a dict with keys:
      retention_per_region, retention_summary,
      write_once, read_freq,
      working_set_per_region, working_set_summary,
      locality_per_region, locality_summary,
      iai, suitability
    """
    ret_per_region, ret_summary = compute_retention(trace)
    wo = compute_write_once(trace)
    rf = compute_read_freq(trace)
    ws_per_region, ws_summary = compute_working_set(trace)
    loc_per_region, loc_summary = compute_locality(trace)
    iai = compute_iai(trace)
    suit = classify_suitability(ret_summary, wo, rf)

    return {
        "retention_per_region": ret_per_region,
        "retention_summary": ret_summary,
        "write_once": wo,
        "read_freq": rf,
        "working_set_per_region": ws_per_region,
        "working_set_summary": ws_summary,
        "locality_per_region": loc_per_region,
        "locality_summary": loc_summary,
        "iai": iai,
        "suitability": suit,
    }

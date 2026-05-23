"""
Observer effect measurement — quantify how much the tracer perturbs the target.

Aggregates ProcessMonitor samples from the observer (tracer) and target (model)
processes into the observer_effect_summary dict that gets written to metadata.json.
"""

from typing import List, Optional

from mrm_trace.collector.process_monitor import ProcessSample


def compute_observer_effect(
    observer_samples: List[ProcessSample],
    target_samples: List[ProcessSample],
    thermal_throttle_events: int = 0,
    baseline_cpu_pct: float = 0.0,
) -> dict:
    """
    Compute observer overhead metrics and return an observer_effect_summary dict.

    Parameters
    ----------
    observer_samples : list of ProcessSample for the tracer/collector process
    target_samples   : list of ProcessSample for the model/inference process
    thermal_throttle_events : events counted by ThermalMonitor during the run
    baseline_cpu_pct : mean baseline CPU from BaselineSnapshot

    Returns
    -------
    dict with keys:
      observer_cpu_overhead_pct_max, observer_cpu_overhead_pct_mean,
      observer_memory_overhead_mb_peak, observer_memory_overhead_frac,
      target_rss_mb_peak, thermal_throttle_events,
      baseline_cpu_pct, io_contention_detected,
      n_observer_samples, n_target_samples
    """
    obs_cpu = [s.cpu_pct for s in observer_samples] or [0.0]
    obs_rss = [s.rss_mb for s in observer_samples] or [0.0]
    tgt_rss = [s.rss_mb for s in target_samples] or [0.0]
    tgt_cpu = [s.cpu_pct for s in target_samples] or [0.0]

    target_rss_peak = max(tgt_rss)
    observer_mem_peak = max(obs_rss)
    observer_mem_frac = (
        observer_mem_peak / target_rss_peak if target_rss_peak > 0 else 0.0
    )

    # I/O contention: flag if observer write_bytes > 10 MB during run
    observer_io_write = sum(
        max(0, observer_samples[i].write_bytes - observer_samples[i - 1].write_bytes)
        for i in range(1, len(observer_samples))
    )
    io_contention = observer_io_write > 10 * 1_048_576  # > 10 MB written

    return {
        "observer_cpu_overhead_pct_max": max(obs_cpu),
        "observer_cpu_overhead_pct_mean": sum(obs_cpu) / len(obs_cpu),
        "observer_memory_overhead_mb_peak": observer_mem_peak,
        "observer_memory_overhead_frac": observer_mem_frac,
        "target_rss_mb_peak": target_rss_peak,
        "thermal_throttle_events": thermal_throttle_events,
        "baseline_cpu_pct": baseline_cpu_pct,
        "io_contention_detected": io_contention,
        "n_observer_samples": len(observer_samples),
        "n_target_samples": len(target_samples),
    }

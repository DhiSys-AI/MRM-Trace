"""
Retention sensitivity to region granularity.

Retention is measured as (last access - first access) of a region, so it is only
estimable when the same region is sampled more than once.  perf mem samples
sparsely — and its load samples are PEBS cache-miss samples, so re-reads that hit
in cache are never recorded — which puts a hard floor on how fine a granule can
be before the estimate collapses.  Both extremes produce artifacts:

  too fine    most regions get a single sample, so retention == 0 by construction
  too coarse  every granule is touched near the start and near the end of the
              run, so retention converges to the run duration and is a property
              of how long you ran, not of the data

Measured on Llama-3.2-1B / ctx=512, 10.1M samples, 9,814 weight samples:

    granule   samples/region   retention>0   p50 (s)   p99 (s)
      4 KB          1.07            7%        0.000    13.587
     64 KB          2.27           66%        5.117    19.547
      1 MB         27.34           92%       19.769    20.868
     16 MB        213.35           80%       20.831    21.526

Only the 64 KB row shows real spread; 1 MB and above are saturated at the ~21 s
run length.  There is no universally correct granule — it depends on sample
density and working-set size — so the sweep belongs in the run output as
evidence for whichever value a study reports.
"""

from typing import Iterable, Optional

import pandas as pd

# Powers of two from one page to 16 MiB.
DEFAULT_GRANULES_BYTES = (4096, 16384, 65536, 262144, 1048576, 4194304, 16777216)

# An estimate needs repeat observations of the same region.
_MIN_SAMPLES_PER_REGION = 2.0
# Enough regions must show a non-zero span for the percentiles to mean anything.
# Sample count alone is not sufficient: repeated samples that all share one
# timestamp give retention 0 everywhere while looking well-sampled.
_MIN_NONZERO_FRACTION = 0.5
# Above this fraction of the trace duration, retention is measuring run length.
_SATURATION_FRACTION = 0.9


def retention_granularity_sweep(
    trace: pd.DataFrame,
    region_type: Optional[str] = None,
    granules_bytes: Iterable[int] = DEFAULT_GRANULES_BYTES,
) -> pd.DataFrame:
    """
    Recompute retention at several granularities and report the shape of each.

    Args:
        trace:          labelled trace with address, timestamp_ns, region_type
        region_type:    restrict to one region type (None = all classified rows)
        granules_bytes: granule sizes to evaluate

    Returns a DataFrame with one row per granule:
        granule_bytes, n_regions, samples_per_region, frac_retention_nonzero,
        p50_s, p90_s, p99_s, max_s, verdict
    """
    df = trace
    if region_type is not None:
        df = df[df["region_type"] == region_type]
    else:
        df = df[df["region_type"] != "unknown"]

    if df.empty:
        return pd.DataFrame(columns=[
            "granule_bytes", "n_regions", "samples_per_region",
            "frac_retention_nonzero", "p50_s", "p90_s", "p99_s", "max_s", "verdict",
        ])

    trace_span_s = float(df["timestamp_ns"].max() - df["timestamp_ns"].min()) / 1e9

    rows = []
    for granule in granules_bytes:
        key = df["address"] // granule
        bounds = df.groupby(key)["timestamp_ns"].agg(["min", "max"])
        retention_s = (bounds["max"] - bounds["min"]) / 1e9

        n_regions = len(bounds)
        per_region = len(df) / n_regions if n_regions else 0.0
        p50 = float(retention_s.quantile(0.50))

        nonzero_fraction = float((retention_s > 0).mean())

        if per_region < _MIN_SAMPLES_PER_REGION or nonzero_fraction < _MIN_NONZERO_FRACTION:
            verdict = "undersampled"
        elif trace_span_s > 0 and p50 >= _SATURATION_FRACTION * trace_span_s:
            verdict = "saturated"
        else:
            verdict = "usable"

        rows.append({
            "granule_bytes": granule,
            "n_regions": n_regions,
            "samples_per_region": round(per_region, 2),
            "frac_retention_nonzero": round(nonzero_fraction, 3),
            "p50_s": round(p50, 3),
            "p90_s": round(float(retention_s.quantile(0.90)), 3),
            "p99_s": round(float(retention_s.quantile(0.99)), 3),
            "max_s": round(float(retention_s.max()), 3),
            "verdict": verdict,
        })

    return pd.DataFrame(rows)


def recommend_granularity(sweep: pd.DataFrame) -> Optional[int]:
    """
    Smallest granule the sweep marks ``usable``, or None if every row is an
    artifact.  None is a real answer: it means this trace cannot support a
    per-region retention distribution and only aggregate figures should be
    reported.
    """
    usable = sweep[sweep["verdict"] == "usable"]
    return int(usable["granule_bytes"].iloc[0]) if len(usable) else None

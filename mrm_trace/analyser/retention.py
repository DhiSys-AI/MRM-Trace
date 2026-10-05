"""
Retention analysis — how long each region is live between first write and last read.

Per-region: first_write_ns, last_read_ns, n_samples, retention_observable,
            retention_ns, retention_s
Summary:    per region_type percentile stats (p50, p90, p99, mean) over the
            regions whose retention is observable, plus how many those are

A region's retention is ``last access - first access``, so a region that was
sampled **once** has no measurable retention: the two ends of the interval are
the same observation.  Reporting that as 0.0 does not understate the value, it
fabricates one, and because sampling density per region falls as context grows,
the fabricated zeros take over the median exactly where the context trend is
read off.  Measured across the v1.5 sweep (CPU, analysis set):

    region      ctx    regions sampled once    p50 reported    p50 observable
    kv_cache   2048               42.8%             10.04 s         66.95 s
    kv_cache   8192               69.1%              0.00 s         59.73 s
    kv_cache  32768               79.3%              0.00 s         44.91 s
    weight     2048               27.6%             49.46 s         80.45 s
    weight    32768               70.4%              0.00 s         37.30 s

So retention does fall with context, by about 1.5-2.2x over a 16x increase --
not to zero, which is what the imputed zeros reported.  Single-sample regions
now carry ``retention_s = NaN`` and are excluded from every percentile, and the
summary states what fraction of regions each percentile rests on.

Note that even an observable retention is a *lower bound*: sampling cannot see
accesses outside the observed window, and the value is a property of the region
granule as well as of the data.
"""

from typing import Tuple

import numpy as np
import pandas as pd

# A region needs at least this many samples for its first and last access to be
# distinct observations rather than the same one.
MIN_SAMPLES_FOR_RETENTION = 2


def compute_retention(trace: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute retention duration for each region.

    Returns
    -------
    per_region : DataFrame
        Columns: region_id, region_type, first_write_ns, last_read_ns,
                 n_samples, retention_observable, retention_ns, retention_s.
        ``retention_ns``/``retention_s`` are NaN where the region was sampled
        fewer than MIN_SAMPLES_FOR_RETENTION times.
    summary : DataFrame
        Columns: region_type, n_regions, n_regions_observable,
                 retention_support_frac, retention_p50_s, retention_p90_s,
                 retention_p99_s, retention_mean_s
    """
    stores = trace[trace["op_type"] == "store"]
    loads = trace[trace["op_type"] == "load"]

    first_write = (
        stores.groupby("region_id")["timestamp_ns"].min().rename("first_write_ns")
    )

    last_read = (
        loads.groupby("region_id")["timestamp_ns"].max().rename("last_read_ns")
    )

    region_types = trace.groupby("region_id")["region_type"].first()
    n_samples = trace.groupby("region_id").size().rename("n_samples")

    per_region = pd.concat([region_types, first_write, last_read], axis=1).reset_index()
    per_region.columns = ["region_id", "region_type", "first_write_ns", "last_read_ns"]
    per_region["n_samples"] = per_region["region_id"].map(n_samples).fillna(0).astype("int64")

    # Regions that were never written — use first access as write time
    no_write_mask = per_region["first_write_ns"].isna()
    if no_write_mask.any():
        fallback = (
            trace.groupby("region_id")["timestamp_ns"].min().rename("first_write_ns")
        )
        per_region.loc[no_write_mask, "first_write_ns"] = per_region.loc[
            no_write_mask, "region_id"
        ].map(fallback)

    # Regions with no sampled read — the interval ends at the write we did see.
    per_region["last_read_ns"] = per_region["last_read_ns"].fillna(
        per_region["first_write_ns"]
    )

    per_region = per_region.astype({"first_write_ns": "int64", "last_read_ns": "int64"})

    # A single sample cannot bound an interval: first and last are one
    # observation, so the span is 0 by construction rather than by measurement.
    per_region["retention_observable"] = (
        per_region["n_samples"] >= MIN_SAMPLES_FOR_RETENTION
    )
    span_ns = (
        per_region["last_read_ns"] - per_region["first_write_ns"]
    ).clip(lower=0)
    per_region["retention_ns"] = span_ns.where(per_region["retention_observable"])
    per_region["retention_s"] = per_region["retention_ns"] / 1e9

    # Summary per region_type, over observable regions only
    summary_rows = []
    for rtype, grp in per_region.groupby("region_type"):
        observable = grp[grp["retention_observable"]]
        s = observable["retention_s"]
        summary_rows.append(
            {
                "region_type": rtype,
                "n_regions": len(grp),
                "n_regions_observable": len(observable),
                "retention_support_frac": (
                    len(observable) / len(grp) if len(grp) else 0.0
                ),
                "retention_p50_s": float(s.quantile(0.50)) if len(s) else np.nan,
                "retention_p90_s": float(s.quantile(0.90)) if len(s) else np.nan,
                "retention_p99_s": float(s.quantile(0.99)) if len(s) else np.nan,
                "retention_mean_s": float(s.mean()) if len(s) else np.nan,
            }
        )
    summary = pd.DataFrame(summary_rows)

    return per_region, summary

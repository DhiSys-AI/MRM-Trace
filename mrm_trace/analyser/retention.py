"""
Retention analysis — how long each region is live between first write and last read.

Per-region: first_write_ns, last_read_ns, retention_ns, retention_s
Summary:    per region_type percentile stats (p50, p90, p99, mean)
"""

from typing import Tuple

import pandas as pd


def compute_retention(trace: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute retention duration for each region.

    Returns
    -------
    per_region : DataFrame
        Columns: region_id, region_type, first_write_ns, last_read_ns,
                 retention_ns, retention_s
    summary : DataFrame
        Columns: region_type, n_regions, retention_p50_s, retention_p90_s,
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

    per_region = pd.concat([region_types, first_write, last_read], axis=1).reset_index()
    per_region.columns = ["region_id", "region_type", "first_write_ns", "last_read_ns"]

    # Regions that were never written — use first access as write time
    no_write_mask = per_region["first_write_ns"].isna()
    if no_write_mask.any():
        fallback = (
            trace.groupby("region_id")["timestamp_ns"].min().rename("first_write_ns")
        )
        per_region.loc[no_write_mask, "first_write_ns"] = per_region.loc[
            no_write_mask, "region_id"
        ].map(fallback)

    # Regions with no reads — retention = 0
    per_region["last_read_ns"] = per_region["last_read_ns"].fillna(
        per_region["first_write_ns"]
    )

    per_region["retention_ns"] = (
        per_region["last_read_ns"] - per_region["first_write_ns"]
    ).clip(lower=0).astype("int64")
    per_region["retention_s"] = per_region["retention_ns"] / 1e9

    per_region = per_region.astype({"first_write_ns": "int64", "last_read_ns": "int64"})

    # Summary per region_type
    def _pct(g: pd.Series, q: float) -> float:
        return float(g.quantile(q))

    summary_rows = []
    for rtype, grp in per_region.groupby("region_type"):
        s = grp["retention_s"]
        summary_rows.append(
            {
                "region_type": rtype,
                "n_regions": len(grp),
                "retention_p50_s": _pct(s, 0.50),
                "retention_p90_s": _pct(s, 0.90),
                "retention_p99_s": _pct(s, 0.99),
                "retention_mean_s": float(s.mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)

    return per_region, summary

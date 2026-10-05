"""
metrics.csv export — one-row-per-region_type summary of all analysis results.

Columns: region_type, n_regions, n_regions_observable, retention_support_frac,
         retention_p50_s, retention_p90_s, retention_p99_s, retention_mean_s,
         write_once_ratio_mean, reads_per_write_mean, working_set_mb_total,
         unique_pages_total, suitability

`n_regions_observable` and `retention_support_frac` are not decoration: a
retention percentile over 1 region and one over 10,000 are different claims, and
without them the table cannot tell them apart.
"""

from pathlib import Path
from typing import Union

import pandas as pd


def build_metrics_table(analysis: dict) -> pd.DataFrame:
    """
    Merge analysis results into a single per-region_type summary DataFrame.

    Parameters
    ----------
    analysis : dict returned by compute_all()

    Returns
    -------
    DataFrame with one row per region_type.
    """
    ret = analysis["retention_summary"].copy()

    # write-once mean per type
    wo = (
        analysis["write_once"]
        .groupby("region_type")["write_once_ratio"]
        .mean()
        .rename("write_once_ratio_mean")
        .reset_index()
    )

    # Reads per write for the region type as a whole: total reads over total
    # writes, not the mean of the per-region ratios.
    #
    # Mean-of-ratios is not the ratio of the type.  A region type almost always
    # contains some read-only sub-regions, whose per-region ratio is infinite
    # (reads, no writes) — averaging that gives inf for the whole type even
    # when it has plenty of writes.  Observed on a real run: kv_cache with
    # 2,988 reads and 17 writes came out as inf rather than 176.
    #
    # The column keeps its name for schema compatibility.
    _rf = analysis["read_freq"].groupby("region_type")[
        ["total_reads", "total_writes"]
    ].sum()
    rf = (
        (_rf["total_reads"] / _rf["total_writes"].replace(0, float("nan")))
        .rename("reads_per_write_mean")
        .reset_index()
    )
    # A type with reads and no writes anywhere is unbounded, not missing.
    _read_only = (_rf["total_writes"] == 0) & (_rf["total_reads"] > 0)
    rf.loc[rf["region_type"].isin(_rf.index[_read_only]),
           "reads_per_write_mean"] = float("inf")

    # working set totals
    ws = analysis["working_set_summary"][
        ["region_type", "total_pages", "total_bytes"]
    ].rename(columns={
        "total_pages": "unique_pages_total",
        "total_bytes": "working_set_bytes_total",
    }).copy()
    ws["working_set_mb_total"] = ws["working_set_bytes_total"] / 1_048_576

    # suitability
    suit = analysis["suitability"][["region_type", "suitability"]]

    merged = (
        ret
        .merge(wo,   on="region_type", how="left")
        .merge(rf,   on="region_type", how="left")
        .merge(ws[["region_type", "unique_pages_total", "working_set_mb_total"]],
               on="region_type", how="left")
        .merge(suit, on="region_type", how="left")
    )

    col_order = [
        "region_type", "n_regions", "n_regions_observable",
        "retention_support_frac",
        "retention_p50_s", "retention_p90_s", "retention_p99_s", "retention_mean_s",
        "write_once_ratio_mean", "reads_per_write_mean",
        "working_set_mb_total", "unique_pages_total",
        "suitability",
    ]
    present = [c for c in col_order if c in merged.columns]
    return merged[present].sort_values("region_type").reset_index(drop=True)


def write_metrics_csv(
    analysis: dict,
    path: Union[str, Path],
) -> Path:
    """Write metrics.csv and return the path."""
    path = Path(path)
    df = build_metrics_table(analysis)
    df.to_csv(path, index=False)
    return path

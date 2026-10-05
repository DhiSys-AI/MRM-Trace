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

    # Inter-access interval.  This was computed by compute_all() and then
    # dropped on the floor, which mattered more than it sounds: retention
    # (last access - first access of a granule) is bounded above by the run and
    # for a region touched throughout simply reports it.  Measured across the
    # occupancy sweep, retention p99 was 89-95% of the collection duration for
    # *every* region type, so it separated them by 10-40%.  IAI -- the gap
    # between consecutive accesses to one address, i.e. how long the line could
    # sit in slow memory -- separates them by three orders of magnitude:
    #
    #   llama-3.2-1b, 98% occupancy, 258 s run, median gap:
    #     runtime   0.03 s   (re-touched constantly; poor MRM candidate)
    #     kv_cache  58.7 s
    #     weight    73.4 s   (untouched for a quarter of the run)
    #
    # n_intervals travels with it because the support is wildly uneven -- 146
    # intervals for weight against 10.2M for runtime -- and because the bias has
    # a direction: an address must be sampled twice to yield a gap, so the
    # addresses that qualify are the most frequently touched ones and the
    # measured gaps are biased *short*.  The weight and KV figures are therefore
    # conservative, which is the safe direction for an MRM claim.
    # .get rather than [] so a caller that assembles the analysis dict itself
    # -- fixtures, and any consumer predating IAI -- still gets a table.
    iai = analysis.get("iai")
    iai = iai.copy() if iai is not None else pd.DataFrame()
    if not iai.empty:
        for pct in ("p50", "p90", "p99"):
            col = f"iai_{pct}_ns"
            if col in iai.columns:
                iai[f"iai_{pct}_s"] = iai[col] / 1e9
        iai = iai[["region_type", "n_intervals"]
                  + [f"iai_{p}_s" for p in ("p50", "p90", "p99")
                     if f"iai_{p}_s" in iai.columns]]
    else:
        iai = pd.DataFrame(columns=["region_type", "n_intervals",
                                    "iai_p50_s", "iai_p90_s", "iai_p99_s"])

    # suitability
    suit = analysis["suitability"][["region_type", "suitability"]]

    merged = (
        ret
        .merge(wo,   on="region_type", how="left")
        .merge(rf,   on="region_type", how="left")
        .merge(ws[["region_type", "unique_pages_total", "working_set_mb_total"]],
               on="region_type", how="left")
        .merge(iai, on="region_type", how="left")
        .merge(suit, on="region_type", how="left")
    )

    col_order = [
        "region_type", "n_regions", "n_regions_observable",
        "retention_support_frac",
        "retention_p50_s", "retention_p90_s", "retention_p99_s", "retention_mean_s",
        "write_once_ratio_mean", "reads_per_write_mean",
        "working_set_mb_total", "unique_pages_total",
        "n_intervals", "iai_p50_s", "iai_p90_s", "iai_p99_s",
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

"""
MRM suitability classification.

Thresholds
----------
high_mrm   : write_once_ratio >= 0.8  AND  retention_p99_s >= 10.0
medium_mrm : write_once_ratio >= 0.5  AND  retention_p50_s >= 1.0
low_mrm    : everything else

Regions with no observed writes are judged on retention alone.  Write-once
ratio is writes-to-an-address-once over all writes, so with zero writes it is
0/0 — undefined, not zero.  Filling it with 0.0 classified the mmap'd weight
region (26,757 reads, 0 writes) as `low_mrm`, which inverts the central claim
this instrument exists to test: weights are the canonical write-once,
read-many, long-retention candidate.

The writes are unobservable rather than absent.  llama.cpp maps the GGUF file
read-only (r--s) and faults pages in from the page cache, so the inference
process never issues a store to the weight region — zero writes is the
limiting case of write-once.  `write_observability` records this, so a reader
can tell "no writes happened" from "writes were not measurable here".
"""

import pandas as pd


def classify_suitability(
    retention_summary: pd.DataFrame,
    write_once_per_region: pd.DataFrame,
    read_freq_per_region: pd.DataFrame,
    retention_threshold_high_s: float = 10.0,
    retention_threshold_medium_s: float = 1.0,
    write_once_threshold_high: float = 0.8,
    write_once_threshold_medium: float = 0.5,
) -> pd.DataFrame:
    """
    Classify each region_type as high_mrm, medium_mrm, or low_mrm.

    Parameters
    ----------
    retention_summary : output of compute_retention()[1]
    write_once_per_region : output of compute_write_once()
    read_freq_per_region : output of compute_read_freq()

    Returns
    -------
    DataFrame
        Columns: region_type, write_once_ratio_mean, retention_p99_s,
                 retention_p50_s, suitability
    """
    wo_by_type = (
        write_once_per_region.groupby("region_type")["write_once_ratio"]
        .mean()
        .rename("write_once_ratio_mean")
        .reset_index()
    )

    merged = retention_summary.merge(wo_by_type, on="region_type", how="left")

    # Whether writes were observable at all, per region type.
    ops = (
        read_freq_per_region.groupby("region_type")[["total_reads", "total_writes"]]
        .sum()
        .reset_index()
    )
    merged = merged.merge(ops, on="region_type", how="left")
    merged[["total_reads", "total_writes"]] = (
        merged[["total_reads", "total_writes"]].fillna(0)
    )
    read_only = (merged["total_writes"] == 0) & (merged["total_reads"] > 0)
    merged["write_observability"] = "observed"
    merged.loc[read_only, "write_observability"] = "read_only_no_writes"
    merged.loc[merged["total_reads"] + merged["total_writes"] == 0,
               "write_observability"] = "no_accesses"

    # Only fill where writes *were* seen: there a missing ratio really is 0.
    observed = merged["write_observability"] == "observed"
    merged.loc[observed, "write_once_ratio_mean"] = (
        merged.loc[observed, "write_once_ratio_mean"].fillna(0.0)
    )

    def _classify(row: pd.Series) -> str:
        wo = row["write_once_ratio_mean"]
        p99 = row["retention_p99_s"]
        p50 = row["retention_p50_s"]
        # Retention is null when no region of this type was sampled twice, so
        # there is no interval to measure.  Every comparison against a
        # threshold is then False, which would silently classify an
        # unmeasured type as `low_mrm` -- the worst verdict, on no evidence.
        if pd.isna(p50) and pd.isna(p99):
            return "unknown"
        if row["write_observability"] == "read_only_no_writes":
            # Zero writes with reads is write-once in the limit, so retention
            # is the only discriminator left.
            if p99 >= retention_threshold_high_s:
                return "high_mrm"
            if p50 >= retention_threshold_medium_s:
                return "medium_mrm"
            return "low_mrm"
        if pd.isna(wo):
            return "unknown"
        if wo >= write_once_threshold_high and p99 >= retention_threshold_high_s:
            return "high_mrm"
        if wo >= write_once_threshold_medium and p50 >= retention_threshold_medium_s:
            return "medium_mrm"
        return "low_mrm"

    merged["suitability"] = merged.apply(_classify, axis=1)

    return merged[
        ["region_type", "write_once_ratio_mean", "write_observability",
         "retention_p99_s", "retention_p50_s", "suitability"]
    ]

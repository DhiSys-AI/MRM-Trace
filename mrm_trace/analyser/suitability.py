"""
MRM suitability classification.

Thresholds
----------
high_mrm   : write_once_ratio >= 0.8  AND  retention_p99_s >= 10.0
medium_mrm : write_once_ratio >= 0.5  AND  retention_p50_s >= 1.0
low_mrm    : everything else
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
    merged["write_once_ratio_mean"] = merged["write_once_ratio_mean"].fillna(0.0)

    def _classify(row: pd.Series) -> str:
        wo = row["write_once_ratio_mean"]
        p99 = row["retention_p99_s"]
        p50 = row["retention_p50_s"]
        if wo >= write_once_threshold_high and p99 >= retention_threshold_high_s:
            return "high_mrm"
        if wo >= write_once_threshold_medium and p50 >= retention_threshold_medium_s:
            return "medium_mrm"
        return "low_mrm"

    merged["suitability"] = merged.apply(_classify, axis=1)

    return merged[
        ["region_type", "write_once_ratio_mean", "retention_p99_s",
         "retention_p50_s", "suitability"]
    ]

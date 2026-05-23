"""
Read frequency analysis — total accesses, read fraction, reads-per-write.
"""

import pandas as pd


def compute_read_freq(trace: pd.DataFrame) -> pd.DataFrame:
    """
    Compute read frequency statistics per region.

    Returns
    -------
    DataFrame
        Columns: region_id, region_type, total_reads, total_writes,
                 total_accesses, read_fraction, reads_per_write
    """
    region_types = trace.groupby("region_id")["region_type"].first()

    counts = trace.groupby(["region_id", "op_type"]).size().unstack(fill_value=0)

    # Ensure both columns exist even if all ops are one type
    for col in ("load", "store"):
        if col not in counts.columns:
            counts[col] = 0

    counts = counts.rename(columns={"load": "total_reads", "store": "total_writes"})
    counts["total_accesses"] = counts["total_reads"] + counts["total_writes"]
    counts["read_fraction"] = (counts["total_reads"] / counts["total_accesses"]).fillna(0.0)
    counts["reads_per_write"] = (
        counts["total_reads"] / counts["total_writes"].replace(0, float("nan"))
    ).fillna(0.0)

    result = counts.reset_index()
    result["region_type"] = result["region_id"].map(region_types)

    return result[
        ["region_id", "region_type", "total_reads", "total_writes",
         "total_accesses", "read_fraction", "reads_per_write"]
    ]

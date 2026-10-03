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
    # A region with reads and no writes has *unbounded* reads per write — the
    # strongest read-many signal there is.  fillna(0.0) scored it as the
    # weakest: the mmap'd weight region (26,757 reads, 0 writes, because the
    # GGUF file is mapped r--s and the process never writes it) came out at
    # 0.0, the minimum.  inf is the honest value; 0.0 is reserved for a region
    # with writes and no reads.
    counts["reads_per_write"] = (
        counts["total_reads"] / counts["total_writes"].replace(0, float("nan"))
    )
    read_only = (counts["total_writes"] == 0) & (counts["total_reads"] > 0)
    counts.loc[read_only, "reads_per_write"] = float("inf")
    counts["reads_per_write"] = counts["reads_per_write"].fillna(0.0)

    result = counts.reset_index()
    result["region_type"] = result["region_id"].map(region_types)

    return result[
        ["region_id", "region_type", "total_reads", "total_writes",
         "total_accesses", "read_fraction", "reads_per_write"]
    ]

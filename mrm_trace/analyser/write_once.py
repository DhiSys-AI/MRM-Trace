"""
Write-once analysis — what fraction of written addresses are written exactly once.

High write-once ratio indicates write-once memory behaviour ideal for MRM.
"""

import pandas as pd


def compute_write_once(trace: pd.DataFrame) -> pd.DataFrame:
    """
    Compute write-once ratio per region.

    An address is "write-once" if it appears in exactly one store access.

    Returns
    -------
    DataFrame
        Columns: region_id, region_type, n_written_addresses,
                 n_write_once_addresses, write_once_ratio
    """
    stores = trace[trace["op_type"] == "store"].copy()

    if stores.empty:
        return pd.DataFrame(
            columns=[
                "region_id", "region_type", "n_written_addresses",
                "n_write_once_addresses", "write_once_ratio",
            ]
        )

    write_counts = (
        stores.groupby(["region_id", "address"]).size().reset_index(name="write_count")
    )
    region_types = trace.groupby("region_id")["region_type"].first().reset_index()

    per_region = write_counts.groupby("region_id").agg(
        n_written_addresses=("address", "count"),
        n_write_once_addresses=("write_count", lambda x: (x == 1).sum()),
    ).reset_index()

    per_region = per_region.merge(region_types, on="region_id", how="left")
    per_region["write_once_ratio"] = (
        per_region["n_write_once_addresses"] / per_region["n_written_addresses"]
    ).clip(0.0, 1.0)

    return per_region[
        ["region_id", "region_type", "n_written_addresses",
         "n_write_once_addresses", "write_once_ratio"]
    ]

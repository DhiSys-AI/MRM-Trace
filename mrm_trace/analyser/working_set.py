"""
Working set size analysis — unique pages and bytes touched per region.
"""

from typing import Tuple

import pandas as pd

_PAGE_BYTES = 4096


def compute_working_set(trace: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute working set size per region and a summary per region_type.

    Returns
    -------
    per_region : DataFrame
        Columns: region_id, region_type, unique_pages, working_set_bytes
    type_summary : DataFrame
        Columns: region_type, n_regions, total_pages, total_bytes,
                 mean_pages_per_region, max_pages_per_region
    """
    region_types = trace.groupby("region_id")["region_type"].first()

    per_region = (
        trace.groupby("region_id")["address_page"]
        .nunique()
        .rename("unique_pages")
        .reset_index()
    )
    per_region["region_type"] = per_region["region_id"].map(region_types)
    per_region["working_set_bytes"] = per_region["unique_pages"] * _PAGE_BYTES

    type_summary_rows = []
    for rtype, grp in per_region.groupby("region_type"):
        type_summary_rows.append(
            {
                "region_type": rtype,
                "n_regions": len(grp),
                "total_pages": int(grp["unique_pages"].sum()),
                "total_bytes": int(grp["working_set_bytes"].sum()),
                "mean_pages_per_region": float(grp["unique_pages"].mean()),
                "max_pages_per_region": int(grp["unique_pages"].max()),
            }
        )
    type_summary = pd.DataFrame(type_summary_rows)

    return per_region, type_summary

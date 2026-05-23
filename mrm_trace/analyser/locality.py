"""
Access locality analysis — stride distribution and same-page access fraction.

Locality measures spatial regularity: small strides and high same-page fraction
indicate cache-friendly, predictable access patterns.
"""

from typing import Tuple

import numpy as np
import pandas as pd


def compute_locality(trace: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute locality statistics per region.

    Stride = |address[i+1] - address[i]| within consecutive accesses to same region.
    Same-page fraction = fraction of consecutive accesses that stay on the same page.

    Returns
    -------
    per_region : DataFrame
        Columns: region_id, region_type, n_accesses, mean_stride_bytes,
                 median_stride_bytes, same_page_fraction
    type_summary : DataFrame
        Columns: region_type, mean_stride_bytes, median_stride_bytes,
                 same_page_fraction
    """
    rows = []
    trace_sorted = trace.sort_values(["region_id", "timestamp_ns"])

    for region_id, grp in trace_sorted.groupby("region_id"):
        region_type = grp["region_type"].iloc[0]
        n = len(grp)

        if n < 2:
            rows.append(
                {
                    "region_id": region_id,
                    "region_type": region_type,
                    "n_accesses": n,
                    "mean_stride_bytes": 0.0,
                    "median_stride_bytes": 0.0,
                    "same_page_fraction": 0.0,
                }
            )
            continue

        addrs = grp["address"].to_numpy()
        pages = grp["address_page"].to_numpy()
        strides = np.abs(np.diff(addrs.astype(np.int64)))
        same_page = np.sum(pages[1:] == pages[:-1])

        rows.append(
            {
                "region_id": region_id,
                "region_type": region_type,
                "n_accesses": n,
                "mean_stride_bytes": float(strides.mean()),
                "median_stride_bytes": float(np.median(strides)),
                "same_page_fraction": float(same_page / len(strides)),
            }
        )

    per_region = pd.DataFrame(rows)
    if per_region.empty:
        return per_region, pd.DataFrame(
            columns=["region_type", "mean_stride_bytes", "median_stride_bytes",
                     "same_page_fraction"]
        )

    type_summary = (
        per_region.groupby("region_type")
        .agg(
            mean_stride_bytes=("mean_stride_bytes", "mean"),
            median_stride_bytes=("median_stride_bytes", "mean"),
            same_page_fraction=("same_page_fraction", "mean"),
        )
        .reset_index()
    )

    return per_region, type_summary

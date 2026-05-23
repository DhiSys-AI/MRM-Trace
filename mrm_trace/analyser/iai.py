"""
Inter-access interval (IAI) analysis — time between consecutive accesses.

Groups by exact address (not address_page) to correctly handle sub-page KV
blocks at 256-byte stride where multiple blocks share a 4K page.
"""

import numpy as np
import pandas as pd


def compute_iai(trace: pd.DataFrame) -> pd.DataFrame:
    """
    Compute inter-access interval distribution per region_type.

    IAI = time gap between consecutive accesses to the *same exact address*,
    sorted by timestamp_ns within each address group.

    Returns
    -------
    DataFrame
        Columns: region_type, n_intervals, iai_p50_ns, iai_p90_ns,
                 iai_p99_ns, iai_mean_ns
        One row per region_type (regions with < 2 accesses to any address
        contribute 0 intervals and are excluded from the summary if they
        have no intervals at all).
    """
    trace_sorted = trace.sort_values(["address", "timestamp_ns"])
    all_intervals = []

    for (address, region_type), grp in trace_sorted.groupby(
        ["address", "region_type"], sort=False
    ):
        ts = grp["timestamp_ns"].to_numpy(dtype=np.int64)
        if len(ts) < 2:
            continue
        intervals = np.diff(ts)
        intervals = intervals[intervals > 0]  # ignore same-timestamp duplicates
        if len(intervals) == 0:
            continue
        all_intervals.append(
            pd.DataFrame({"region_type": region_type, "iai_ns": intervals})
        )

    if not all_intervals:
        return pd.DataFrame(
            columns=["region_type", "n_intervals", "iai_p50_ns",
                     "iai_p90_ns", "iai_p99_ns", "iai_mean_ns"]
        )

    combined = pd.concat(all_intervals, ignore_index=True)

    rows = []
    for rtype, grp in combined.groupby("region_type"):
        s = grp["iai_ns"]
        rows.append(
            {
                "region_type": rtype,
                "n_intervals": len(s),
                "iai_p50_ns": int(s.quantile(0.50)),
                "iai_p90_ns": int(s.quantile(0.90)),
                "iai_p99_ns": int(s.quantile(0.99)),
                "iai_mean_ns": float(s.mean()),
            }
        )

    return pd.DataFrame(rows)

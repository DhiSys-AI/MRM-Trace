"""
MRM suitability classification.

Thresholds
----------
high_mrm   : write_once_ratio >= 0.8  AND  retention_p99_s >= 10.0
             AND  iai_p50_s >= 1.0        (when IAI is available)
medium_mrm : write_once_ratio >= 0.5  AND  retention_p50_s >= 1.0
             AND  iai_p50_s >= 0.1
low_mrm    : everything else

**The IAI gate is off by default, and must stay off until IAI is calibrated.**
A measured interval is the gap between consecutive *samples* of an address, not
consecutive accesses. Under 1-in-P sampling that inflates every gap by roughly
P, which cancels in a ratio between regions but makes the absolute value
meaningless as wall time. Checked on one run: dividing by P gives an estimated
15 ms between weight accesses where the workload implies ~130 ms per token --
a 9x discrepancy, because `ldlat` makes PEBS eligibility depend on whether the
access missed cache, so the sampled fraction is not uniform across regions.

So pass `iai_threshold_s` only with a figure derived from a calibrated device
*and* a validated correction. Until then the usable result is the **ratio**:
at 98% KV occupancy on llama-3.2-1B the median gap was 0.03 s for `runtime`
against 73.4 s for `weight`, a separation of three orders of magnitude, where
retention separated them by 10-40%. The ordering is the finding; the cut points
are not.

The IAI gate is not decoration.  Retention is `last access - first access` of a
granule, so it is bounded above by the run and, for any region touched
throughout, simply reports the run length: measured across the occupancy sweep,
retention p99 was 89-95% of the collection duration for *every* region type.
The retention conditions above were therefore satisfied by everything, and
`runtime` -- re-touched every 31 ms at the median -- came out `high_mrm`
alongside the weight region.  A verdict every region passes is not a
classification.

Inter-access interval is the quantity MRM actually turns on: how long the line
is left alone, and so whether it could sit in slow memory without costing a
stall.  At 98% KV occupancy on llama-3.2-1b the median gap was 0.03 s for
`runtime` against 73.4 s for `weight` -- a separation of three orders of
magnitude where retention gave 10-40%.

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
    iai_threshold_s: "float | None" = None,
    iai_threshold_medium_s: "float | None" = None,
    iai_summary: "pd.DataFrame | None" = None,
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

    # IAI is optional so existing callers keep working; when absent the gate is
    # skipped rather than failed, and `iai_p50_s` comes through as NaN.
    if iai_summary is not None and not iai_summary.empty:
        iai = iai_summary.copy()
        if "iai_p50_s" not in iai.columns and "iai_p50_ns" in iai.columns:
            iai["iai_p50_s"] = iai["iai_p50_ns"] / 1e9
        cols = [c for c in ("region_type", "iai_p50_s", "n_intervals")
                if c in iai.columns]
        merged = merged.merge(iai[cols], on="region_type", how="left")
    if "iai_p50_s" not in merged.columns:
        merged["iai_p50_s"] = float("nan")

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
        gap = row.get("iai_p50_s", float("nan"))
        # A region re-touched faster than the gate cannot be high_mrm whatever
        # its span.  NaN means IAI was not supplied, and the gate is skipped.
        # Off unless the caller supplies a calibrated threshold: a measured gap
        # is in units of sampled-access time, not wall time (see the module
        # docstring), so comparing it to an absolute number of seconds has no
        # defined meaning yet.
        gap_ok = (iai_threshold_s is None or pd.isna(gap)
                  or gap >= iai_threshold_s)
        gap_ok_medium = (iai_threshold_medium_s is None or pd.isna(gap)
                         or gap >= iai_threshold_medium_s)
        # Retention is null when no region of this type was sampled twice, so
        # there is no interval to measure.  Every comparison against a
        # threshold is then False, which would silently classify an
        # unmeasured type as `low_mrm` -- the worst verdict, on no evidence.
        if pd.isna(p50) and pd.isna(p99):
            return "unknown"
        if row["write_observability"] == "read_only_no_writes":
            # Zero writes with reads is write-once in the limit, so retention
            # is the only discriminator left.
            if p99 >= retention_threshold_high_s and gap_ok:
                return "high_mrm"
            if p50 >= retention_threshold_medium_s and gap_ok_medium:
                return "medium_mrm"
            return "low_mrm"
        if pd.isna(wo):
            return "unknown"
        if (wo >= write_once_threshold_high
                and p99 >= retention_threshold_high_s and gap_ok):
            return "high_mrm"
        if (wo >= write_once_threshold_medium
                and p50 >= retention_threshold_medium_s and gap_ok_medium):
            return "medium_mrm"
        return "low_mrm"

    merged["suitability"] = merged.apply(_classify, axis=1)

    out_cols = ["region_type", "write_once_ratio_mean", "write_observability",
                "retention_p99_s", "retention_p50_s", "iai_p50_s", "suitability"]
    return merged[[c for c in out_cols if c in merged.columns]]

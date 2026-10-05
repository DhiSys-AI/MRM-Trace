"""
Re-derive metrics.csv from an exported run's trace.parquet.

Like KV block lifecycle (``labeller.kv_rederive``) and the validity verdict
(``experiment_runner.py --reclassify``), every metric in metrics.csv is a
*derivation* over rows the run already recorded.  Correcting a derivation
therefore never requires re-collecting a trace, and the corrected file comes
from the same ``compute_all`` the pipeline runs, so a re-derived dataset is not
a patched one.

The trace is read in full here rather than streamed.  ADR-2 governs the parser,
where a run can produce 50+ GB before reduction; by export time a trace is the
sampled subset -- 3.5M rows, ~160 MB -- and every analyser already takes a
DataFrame.
"""

from __future__ import annotations

import pathlib
from typing import Dict, Union

import pandas as pd
import pyarrow.parquet as pq

from mrm_trace.analyser import compute_all
from mrm_trace.reporter.metrics_csv import write_metrics_csv

# The analysers need these; reading only them keeps a wide trace cheap.
_COLUMNS = ["region_id", "region_type", "op_type", "timestamp_ns",
            "address", "address_page"]


def rederive_metrics(
    run_dir: Union[str, pathlib.Path],
    dry_run: bool = False,
) -> Dict[str, object]:
    """
    Rebuild metrics.csv for one exported run.

    Returns the per-region_type retention support actually achieved, so a
    caller can report what the new percentiles rest on.
    """
    run_dir = pathlib.Path(run_dir)
    trace_path = run_dir / "trace.parquet"
    if not trace_path.exists():
        raise FileNotFoundError(trace_path)

    names = set(pq.ParquetFile(trace_path).schema_arrow.names)
    trace = pd.read_parquet(trace_path,
                            columns=[c for c in _COLUMNS if c in names])

    analysis = compute_all(trace)
    if not dry_run:
        write_metrics_csv(analysis, run_dir / "metrics.csv")

    ret = analysis["retention_summary"]
    support = {
        row["region_type"]: {
            "n_regions": int(row["n_regions"]),
            "observable": int(row["n_regions_observable"]),
            "frac": float(row["retention_support_frac"]),
            "p50_s": float(row["retention_p50_s"])
            if pd.notna(row["retention_p50_s"]) else None,
        }
        for _, row in ret.iterrows()
    }
    return {"rows": len(trace), "support": support}

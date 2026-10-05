"""
Re-derive kv_block_lifecycle.parquet from an exported run's trace.parquet.

KV block lifecycle is a *derivation* over rows the run already recorded, not an
independent measurement.  Correcting the derivation therefore never requires
re-collecting a trace — the same reasoning that lets
``experiment_runner.py --reclassify`` re-derive validity verdicts in place.

Replay is exact rather than approximate: blocks are keyed by
``address // region_granularity_bytes``, the same granule the labeller used
(see ``RegionLabeller._label_row``), and rows are replayed in file order, which
is the order in which they were originally labelled.  ``write_ns`` is the first
store seen and ``evict_ns`` the second distinct one, so order is what makes the
result reproducible.
"""

from __future__ import annotations

import json
import pathlib
from typing import Dict, Optional, Union

import pyarrow.compute as pc
import pyarrow.parquet as pq

from mrm_trace.labeller.kv_lifecycle import KVBlockTracker
from mrm_trace.reporter.parquet_export import export_kv_lifecycle

_COLUMNS = ["address", "address_page", "region_id", "op_type",
            "timestamp_ns", "region_type"]

DEFAULT_GRANULARITY = 65536


def granularity_for_run(run_dir: pathlib.Path) -> int:
    """Region granule in bytes, from the run's own metadata."""
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        return DEFAULT_GRANULARITY
    meta = json.loads(meta_path.read_text())
    granule = (meta.get("collector") or {}).get("region_granularity_bytes")
    return int(granule) if granule else DEFAULT_GRANULARITY


def rederive_kv_lifecycle(
    run_dir: Union[str, pathlib.Path],
    granularity: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, int]:
    """
    Rebuild kv_block_lifecycle.parquet for one exported run.

    Returns counts: blocks, write_observed, lifetime_null, kv_rows.
    Raises FileNotFoundError when the run has no trace.parquet to derive from.
    """
    run_dir = pathlib.Path(run_dir)
    trace_path = run_dir / "trace.parquet"
    if not trace_path.exists():
        raise FileNotFoundError(trace_path)

    if granularity is None:
        granularity = granularity_for_run(run_dir)

    tracker = KVBlockTracker()
    kv_rows = 0

    # Row-group streaming, per ADR-2: a trace may be far larger than RAM, and
    # the KV rows inside it are a small minority -- 2,936 of 3.5M in a measured
    # run.  The filter runs in Arrow so the Python loop sees only those rows;
    # iterating every row instead cost over 5 min of CPU for a single run,
    # which across the dataset is slower than re-collecting it.
    pf = pq.ParquetFile(trace_path)
    columns = [c for c in _COLUMNS if c in pf.schema_arrow.names]
    for batch in pf.iter_batches(batch_size=200_000, columns=columns):
        kv = batch.filter(pc.equal(batch.column("region_type"), "kv_cache"))
        if kv.num_rows == 0:
            continue
        kv_rows += kv.num_rows
        d = kv.to_pydict()
        pages = d["address_page"]
        region_ids = d["region_id"]
        ops = d["op_type"]
        stamps = d["timestamp_ns"]
        for i, address in enumerate(d["address"]):
            tracker.record(
                address=address,
                address_page=pages[i],
                region_id=region_ids[i],
                op_type=ops[i],
                timestamp_ns=stamps[i],
                block_key=address // granularity,
            )

    rows = tracker.lifecycle_rows()
    stats = {
        "kv_rows": kv_rows,
        "blocks": len(rows),
        "write_observed": sum(1 for r in rows if r["write_observed"]),
        "lifetime_null": sum(1 for r in rows if r["lifetime_ns"] is None),
    }
    if not dry_run:
        export_kv_lifecycle(rows, run_dir)
    return stats

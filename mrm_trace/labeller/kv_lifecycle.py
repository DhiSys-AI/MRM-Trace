"""
KV cache block lifecycle tracker.

For each unique KV cache address, records:
  write_ns   — timestamp of the first STORE, or None if no store was ever sampled
  last_read_ns — timestamp of the most recent LOAD
  evict_ns   — timestamp of a second STORE (block reuse / eviction)
  total_reads — LOAD count
  write_observed — whether a store for this block appeared in the sample
  lifetime_ns — (evict_ns or last_read_ns) - write_ns, or None when the write
                was never sampled

A block's lifetime is only defined when its write was *observed*.  Stores are
sampled on their own, much sparser period than loads, so the common case for a
KV block is many sampled loads and no sampled store at all.  Coercing that
absent write to 0 does not yield a short lifetime — it yields
``last_read_ns - 0``, the raw monotonic clock reading, which is the machine's
uptime (tens of hours) rather than a duration.  Such a block reports
``lifetime_ns = None``; the absence is recorded, never imputed.

This data is the primary input for KV cache retention analysis (Phase 6).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class KVBlockEntry:
    block_id: int             # = region_id for this block
    address: int              # exact address of first store
    address_page: int
    write_ns: Optional[int] = None   # None until first STORE seen
    last_read_ns: int = 0
    total_reads: int = 0
    evict_ns: Optional[int] = None   # set on second STORE


class KVBlockTracker:
    """
    Tracks write/read/eviction lifecycle of KV cache blocks.

    Blocks are keyed by the caller's region granule (``block_key``), not by page,
    so kv_block_lifecycle.parquet and region_map.parquet always agree on what one
    block is.  When they disagree, per-block retention in one file cannot be
    joined to region stats in the other — which is how a run once reported 48,000
    KV "regions" while the lifecycle file tracked only a handful.
    """

    def __init__(self) -> None:
        self._blocks: Dict[int, KVBlockEntry] = {}  # block_key → entry

    def record(
        self,
        address: int,
        address_page: int,
        region_id: int,
        op_type: str,
        timestamp_ns: int,
        block_key: Optional[int] = None,
    ) -> None:
        """Update block state for one KV cache access."""
        key = address_page if block_key is None else block_key
        if key not in self._blocks:
            self._blocks[key] = KVBlockEntry(
                block_id=region_id,
                address=address,
                address_page=address_page,
            )

        entry = self._blocks[key]

        if op_type == "store":
            if entry.write_ns is None:
                entry.write_ns = timestamp_ns
            elif entry.evict_ns is None and timestamp_ns != entry.write_ns:
                # Second distinct store timestamp → eviction / block reuse
                entry.evict_ns = timestamp_ns
        else:  # load
            entry.total_reads += 1
            if timestamp_ns > entry.last_read_ns:
                entry.last_read_ns = timestamp_ns

    def lifecycle_rows(self) -> List[dict]:
        """Return rows for kv_block_lifecycle.parquet."""
        rows = []
        for e in self._blocks.values():
            # An unsampled store is an unknown, not a write at t=0.  Subtracting
            # 0 from a monotonic timestamp yields the clock reading itself, so
            # the lifetime of a block whose write was never sampled is null.
            if e.write_ns is None:
                lifetime_ns = None
            else:
                end_ns = e.evict_ns or e.last_read_ns or e.write_ns
                lifetime_ns = max(0, end_ns - e.write_ns)
            rows.append({
                "block_id": e.block_id,
                "address": e.address,
                "address_page": e.address_page,
                "write_ns": e.write_ns,
                "last_read_ns": e.last_read_ns,
                "evict_ns": e.evict_ns,
                "total_reads": e.total_reads,
                "write_observed": e.write_ns is not None,
                "lifetime_ns": lifetime_ns,
            })
        return rows

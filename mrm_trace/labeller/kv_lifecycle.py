"""
KV cache block lifecycle tracker.

For each unique KV cache address, records:
  write_ns   — timestamp of the first STORE (block allocation)
  last_read_ns — timestamp of the most recent LOAD
  evict_ns   — timestamp of a second STORE (block eviction / reuse)
  total_reads — LOAD count
  lifetime_ns — (evict_ns or last_read_ns) - write_ns

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
            write_ns = e.write_ns or 0
            end_ns = e.evict_ns or e.last_read_ns or write_ns
            rows.append({
                "block_id": e.block_id,
                "address": e.address,
                "address_page": e.address_page,
                "write_ns": write_ns,
                "last_read_ns": e.last_read_ns,
                "evict_ns": e.evict_ns,
                "total_reads": e.total_reads,
                "lifetime_ns": max(0, end_ns - write_ns),
            })
        return rows

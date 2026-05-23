"""
Main labelling pipeline — classifies trace rows with region_type and region_id.

Labelling strategy (applied per row):
  1. Pass-through: rows with a pre-existing non-unknown region_type are not re-classified.
  2. Symbol rules: classify by symbol / DSO / phase using DEFAULT_RULES.
  3. Address propagation: once a page is classified, subsequent accesses inherit it.
  4. Unknown policy: "keep" (leave as unknown), "skip" (omit row), "error" (raise).

Region ID assignment:
  - weight    : all weight pages share region_id=1 (model weights are one logical region)
  - kv_cache  : each distinct write address gets a unique region_id (address granularity,
                so multiple KV blocks on the same 4K page each get their own id)
  - activation/runtime : each distinct address_page gets a unique region_id
"""

from dataclasses import dataclass, field
from typing import Dict, Generator, Iterable, List, Optional, Set

from mrm_trace.labeller.kv_lifecycle import KVBlockTracker
from mrm_trace.labeller.symbol_rules import DEFAULT_RULES, SymbolRule, classify_symbol

_UNKNOWN = "unknown"
_WEIGHT = "weight"
_KV = "kv_cache"
_WEIGHT_REGION_ID = 1
_VALID_POLICIES = frozenset({"keep", "skip", "error"})


@dataclass
class _RegionStats:
    region_id: int
    region_type: str
    page_min: int
    page_max: int
    first_seen_ns: int = 0
    last_seen_ns: int = 0
    access_count: int = 0

    def touch(self, page: int, ts_ns: int) -> None:
        self.access_count += 1
        if ts_ns > self.last_seen_ns:
            self.last_seen_ns = ts_ns
        if page < self.page_min:
            self.page_min = page
        if page > self.page_max:
            self.page_max = page


class TraceLabeller:
    """
    Streaming trace labeller.

    Usage::

        labeller = TraceLabeller()
        for row in labeller.label(raw_rows):
            writer.append(row)
        region_map   = labeller.region_map()   # call AFTER consuming label()
        kv_lifecycle = labeller.kv_lifecycle()
    """

    def __init__(
        self,
        rules: Optional[List[SymbolRule]] = None,
        unknown_policy: str = "keep",
    ) -> None:
        if unknown_policy not in _VALID_POLICIES:
            raise ValueError(
                f"unknown_policy must be one of {sorted(_VALID_POLICIES)}, "
                f"got {unknown_policy!r}"
            )
        self._rules = rules if rules is not None else DEFAULT_RULES
        self._unknown_policy = unknown_policy

        # Unified region ID counter (weight is always 1; all others start at 2)
        self._next_id: int = 2
        self._regions: Dict[int, _RegionStats] = {}  # region_id → stats

        # Page-level type propagation (address_page → region_type)
        self._page_types: Dict[int, str] = {}

        # Region ID lookup per type:
        #   weight:           always _WEIGHT_REGION_ID
        #   kv_cache:         address → region_id   (address-granularity)
        #   activation/runtime: address_page → region_id
        self._kv_addr_ids: Dict[int, int] = {}
        self._page_ids: Dict[int, int] = {}  # for activation / runtime

        self._kv_tracker = KVBlockTracker()

    # ── public API ───────────────────────────────────────────────────────────

    def label(self, rows: Iterable[dict]) -> Generator[dict, None, None]:
        """
        Stream trace rows, yielding each row with region_type and region_id set.
        """
        for row in rows:
            labelled = self._label_row(row)
            if labelled is None:
                continue
            yield labelled

    def region_map(self) -> List[dict]:
        """Return region_map rows (one per region).  Call after consuming label()."""
        return [
            {
                "region_id": s.region_id,
                "region_type": s.region_type,
                "page_min": s.page_min,
                "page_max": s.page_max,
                "first_seen_ns": s.first_seen_ns,
                "last_seen_ns": s.last_seen_ns,
                "access_count": s.access_count,
            }
            for s in self._regions.values()
        ]

    def kv_lifecycle(self) -> List[dict]:
        """Return KV block lifecycle rows.  Call after consuming label()."""
        return self._kv_tracker.lifecycle_rows()

    # ── internals ────────────────────────────────────────────────────────────

    def _label_row(self, row: dict) -> Optional[dict]:
        address = row["address"]
        address_page = row["address_page"]
        current_type = row.get("region_type", _UNKNOWN)
        ts_ns = row.get("timestamp_ns", 0)
        op_type = row.get("op_type", "load")

        # Step 1: pass-through for pre-labelled rows
        if current_type != _UNKNOWN:
            region_type = current_type
        else:
            # Step 2: page propagation (existing classification wins — first wins rule)
            page_type = self._page_types.get(address_page)
            if page_type is not None:
                region_type = page_type
            else:
                # Step 3: symbol rules (only for unclassified pages)
                region_type = classify_symbol(
                    symbol=row.get("symbol", ""),
                    dso=row.get("dso", ""),
                    phase=row.get("phase"),
                    op_type=op_type,
                    rules=self._rules,
                ) or _UNKNOWN

        # Apply unknown policy
        if region_type == _UNKNOWN:
            if self._unknown_policy == "skip":
                return None
            if self._unknown_policy == "error":
                raise ValueError(
                    f"Could not classify row: address_page={address_page:#x}, "
                    f"symbol={row.get('symbol')!r}, dso={row.get('dso')!r}. "
                    "Set unknown_policy='keep' to allow unknowns."
                )
            return {**row, "region_type": _UNKNOWN, "region_id": 0}

        # Step 4: assign region_id and update region stats
        region_id = self._assign_region_id(address, address_page, region_type, ts_ns)

        # Propagate page type for future unknown accesses to this page
        if address_page not in self._page_types:
            self._page_types[address_page] = region_type

        # Step 5: update KV lifecycle tracker
        if region_type == _KV:
            self._kv_tracker.record(
                address=address,
                address_page=address_page,
                region_id=region_id,
                op_type=op_type,
                timestamp_ns=ts_ns,
            )

        return {**row, "region_type": region_type, "region_id": region_id}

    def _assign_region_id(
        self,
        address: int,
        address_page: int,
        region_type: str,
        ts_ns: int,
    ) -> int:
        if region_type == _WEIGHT:
            rid = _WEIGHT_REGION_ID
            if rid not in self._regions:
                self._regions[rid] = _RegionStats(
                    region_id=rid, region_type=_WEIGHT,
                    page_min=address_page, page_max=address_page,
                    first_seen_ns=ts_ns, last_seen_ns=ts_ns, access_count=0,
                )
            self._regions[rid].touch(address_page, ts_ns)

        elif region_type == _KV:
            # Address-granularity: each distinct write address gets its own region
            if address not in self._kv_addr_ids:
                rid = self._next_id
                self._next_id += 1
                self._kv_addr_ids[address] = rid
                self._regions[rid] = _RegionStats(
                    region_id=rid, region_type=_KV,
                    page_min=address_page, page_max=address_page,
                    first_seen_ns=ts_ns, last_seen_ns=ts_ns, access_count=0,
                )
            else:
                rid = self._kv_addr_ids[address]
            self._regions[rid].touch(address_page, ts_ns)

        else:
            # activation / runtime — page-granularity
            if address_page not in self._page_ids:
                rid = self._next_id
                self._next_id += 1
                self._page_ids[address_page] = rid
                self._regions[rid] = _RegionStats(
                    region_id=rid, region_type=region_type,
                    page_min=address_page, page_max=address_page,
                    first_seen_ns=ts_ns, last_seen_ns=ts_ns, access_count=0,
                )
            else:
                rid = self._page_ids[address_page]
            self._regions[rid].touch(address_page, ts_ns)

        return rid

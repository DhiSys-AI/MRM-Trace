"""
Main labelling pipeline — classifies trace rows with region_type and region_id.

Labelling strategy (applied per row):
  1. Pass-through: rows with a pre-existing non-unknown region_type are not re-classified.
  2. Evidence-ranked classification: address-range and symbol evidence are both
     consulted and the STRONGER one wins (see EVIDENCE below).
  3. Unknown policy: "keep" (leave as unknown), "skip" (omit row), "error" (raise).

Why evidence ranking rather than "first source wins":
    Address ranges from /proc/self/maps cover the whole address space, so a
    first-wins rule means every page is already classified before the symbol
    rules are consulted — and symbol-based labelling never runs at all.  That
    silently discards the most specific evidence available (a `llama_kv_cache_*`
    symbol is far better evidence than "this page is inside an anonymous
    mapping").  Ranking the sources keeps both mechanisms live.

Region ID assignment:
    One region per ``region_granularity_bytes`` of address space (default 4 KiB,
    one page), for every region type.

    Retention is defined per region (spec §6.1) and must be reported as a
    distribution — p50/p90/p99/max/mean/std.  Two earlier schemes each destroyed
    that distribution in a different way:

      all weight pages share region_id=1
          → exactly one retention sample, so p50 == p99 == max.  A point, not a
            distribution.
      one region per distinct KV *address*
          → perf reports arbitrary byte addresses inside a block, so a store to
            X and a load to X+8 became different regions, each with a single
            access and therefore retention == 0.0 by construction.

    A granule coarse enough to contain multiple samples of the same logical
    object is what makes retention measurable at all.  4 KiB is the default
    because it is the unit the OS and the hardware actually manage, and because
    KVBlockTracker has always keyed blocks by address_page.  It is a parameter,
    not a constant, so the choice is auditable and can be revisited per study.
"""

from dataclasses import dataclass
from typing import Dict, Generator, Iterable, List, Optional, Tuple

from mrm_trace.labeller.kv_lifecycle import KVBlockTracker
from mrm_trace.labeller.symbol_rules import DEFAULT_RULES, SymbolRule, classify_symbol

_UNKNOWN = "unknown"
_WEIGHT = "weight"
_KV = "kv_cache"
_VALID_POLICIES = frozenset({"keep", "skip", "error"})

# ── Evidence strength ─────────────────────────────────────────────────────────
# Higher wins.  Ties resolve in favour of the already-recorded classification, so
# labelling stays stable and order-independent for equally-strong evidence.
EV_MAPS_COARSE = 1   # address is inside some mapping (runtime, heap, stack, lib)
EV_SYMBOL = 2        # a symbol rule matched
EV_MAPS_KV = 3       # anonymous mapping whose size matches the model's KV geometry
EV_SYMBOL_KV = 4     # symbol is engine KV-cache code — very specific
EV_MAPS_WEIGHT = 5   # address is inside the mmap'd GGUF file — unambiguous

_SOURCE_EVIDENCE = {
    "mmap_gguf": EV_MAPS_WEIGHT,
    "anon_kv_sized": EV_MAPS_KV,
}


def _range_evidence(region_type: str, source: Optional[str]) -> int:
    """Evidence strength for one pre-registered address range."""
    if source in _SOURCE_EVIDENCE:
        return _SOURCE_EVIDENCE[source]
    # Unsourced ranges (callers that pass bare region dicts) are trusted by type.
    if region_type == _WEIGHT:
        return EV_MAPS_WEIGHT
    if region_type == _KV:
        return EV_MAPS_KV
    return EV_MAPS_COARSE


def _symbol_evidence(region_type: str) -> int:
    return EV_SYMBOL_KV if region_type == _KV else EV_SYMBOL


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
        labeller.register_address_ranges(regions)   # optional, from /proc/self/maps
        labeller.prescan_symbols(rows)              # optional, needs materialised rows
        for row in labeller.label(rows):
            writer.append(row)
        region_map   = labeller.region_map()   # call AFTER consuming label()
        kv_lifecycle = labeller.kv_lifecycle()
    """

    def __init__(
        self,
        rules: Optional[List[SymbolRule]] = None,
        unknown_policy: str = "keep",
        region_granularity_bytes: int = 4096,
    ) -> None:
        if unknown_policy not in _VALID_POLICIES:
            raise ValueError(
                f"unknown_policy must be one of {sorted(_VALID_POLICIES)}, "
                f"got {unknown_policy}"
            )
        if region_granularity_bytes <= 0:
            raise ValueError(
                f"region_granularity_bytes must be positive, "
                f"got {region_granularity_bytes}"
            )
        self._rules = rules if rules is not None else DEFAULT_RULES
        self._unknown_policy = unknown_policy
        self._granularity = region_granularity_bytes

        self._next_id: int = 1
        self._regions: Dict[int, _RegionStats] = {}   # region_id → stats
        self._granule_ids: Dict[int, int] = {}        # granule → region_id

        # address_page → (region_type, evidence strength)
        self._page_types: Dict[int, Tuple[str, int]] = {}

        self._kv_tracker = KVBlockTracker()

    # ── public API ───────────────────────────────────────────────────────────

    def register_address_ranges(self, regions) -> int:
        """
        Pre-seed the page-type map with known address ranges before labelling.

        Accepts the output of ``extract_llamacpp_regions()`` /
        ``extract_pytorch_regions()`` — any iterable of objects with
        ``addr_start``, ``addr_end`` and ``region_type`` attributes or keys, and
        optionally ``source`` (which determines evidence strength).

        A range only overwrites an existing page classification when its evidence
        is strictly stronger, so overlapping or repeated snapshots are safe.

        Returns the number of pages whose classification was set or upgraded.
        """
        def _get(r, key):
            # source is optional — pytorch_regions.TensorRegion has no such field.
            return r.get(key) if isinstance(r, dict) else getattr(r, key, None)

        page_count = 0
        for r in regions:
            addr_start = _get(r, "addr_start")
            addr_end = _get(r, "addr_end")
            rtype = _get(r, "region_type")
            evidence = _range_evidence(rtype, _get(r, "source"))

            page_start = addr_start >> 12
            page_end = (max(addr_end, addr_start + 1) - 1) >> 12
            for page in range(page_start, page_end + 1):
                if self._record_page_type(page, rtype, evidence):
                    page_count += 1
        return page_count

    def prescan_symbols(self, rows: Iterable[dict]) -> int:
        """
        Consult symbol rules for every row and record the resulting page
        classifications before labelling begins.

        Needed because symbol evidence arrives per row while classification is
        per page: without a pre-pass, the first sample on a page would be
        labelled from address ranges alone and later samples carrying a decisive
        symbol would be labelled differently, making the output depend on row
        order.  Callers holding the rows in memory should run this; purely
        streaming callers can skip it and get address-range labelling only.

        Returns the number of pages set or upgraded by symbol evidence.
        """
        upgraded = 0
        for row in rows:
            if row.get("region_type", _UNKNOWN) != _UNKNOWN:
                continue
            symbol = row.get("symbol", "")
            if not symbol and not row.get("dso", ""):
                continue
            rtype = classify_symbol(
                symbol=symbol,
                dso=row.get("dso", ""),
                phase=row.get("phase"),
                op_type=row.get("op_type", "load"),
                rules=self._rules,
            )
            if rtype and self._record_page_type(
                row["address_page"], rtype, _symbol_evidence(rtype)
            ):
                upgraded += 1
        return upgraded

    def label(self, rows: Iterable[dict]) -> Generator[dict, None, None]:
        """Stream trace rows, yielding each row with region_type and region_id set."""
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

    def _record_page_type(self, page: int, region_type: str, evidence: int) -> bool:
        """Set or upgrade a page's classification.  True if it changed."""
        existing = self._page_types.get(page)
        if existing is not None and existing[1] >= evidence:
            return False
        self._page_types[page] = (region_type, evidence)
        return True

    def _label_row(self, row: dict) -> Optional[dict]:
        address = row["address"]
        address_page = row["address_page"]
        current_type = row.get("region_type", _UNKNOWN)
        ts_ns = row.get("timestamp_ns", 0)
        op_type = row.get("op_type", "load")

        if current_type != _UNKNOWN:
            # Step 1: pass-through for rows a caller already classified
            region_type = current_type
        else:
            # Step 2: strongest available evidence wins
            page_entry = self._page_types.get(address_page)
            region_type, evidence = page_entry if page_entry else (_UNKNOWN, 0)

            symbol_type = classify_symbol(
                symbol=row.get("symbol", ""),
                dso=row.get("dso", ""),
                phase=row.get("phase"),
                op_type=op_type,
                rules=self._rules,
            )
            if symbol_type and _symbol_evidence(symbol_type) > evidence:
                region_type = symbol_type
                evidence = _symbol_evidence(symbol_type)

            if region_type != _UNKNOWN:
                self._record_page_type(address_page, region_type, evidence)

        # Step 3: unknown policy
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

        # Step 4: region identity — one region per granule, every region type
        region_id = self._assign_region_id(address, address_page, region_type, ts_ns)

        # Step 5: KV lifecycle tracking — same granule as region identity, so
        # kv_block_lifecycle.parquet and region_map.parquet agree on "a block"
        if region_type == _KV:
            self._kv_tracker.record(
                address=address,
                address_page=address_page,
                region_id=region_id,
                op_type=op_type,
                timestamp_ns=ts_ns,
                block_key=address // self._granularity,
            )

        return {**row, "region_type": region_type, "region_id": region_id}

    def _assign_region_id(
        self, address: int, address_page: int, region_type: str, ts_ns: int
    ) -> int:
        granule = address // self._granularity
        rid = self._granule_ids.get(granule)
        if rid is None:
            rid = self._next_id
            self._next_id += 1
            self._granule_ids[granule] = rid
            self._regions[rid] = _RegionStats(
                region_id=rid, region_type=region_type,
                page_min=address_page, page_max=address_page,
                first_seen_ns=ts_ns, last_seen_ns=ts_ns, access_count=0,
            )
        self._regions[rid].touch(address_page, ts_ns)
        return rid

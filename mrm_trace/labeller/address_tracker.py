"""
Address-page to region mapping tracker.

Once a page is classified, that classification is stable — subsequent accesses
to the same page inherit the existing region_type and region_id (propagation rule).

Weight region: all weight pages share region_id=1.
All other types: each new page receives the next sequential region_id (≥ 2).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

_WEIGHT_REGION_ID = 1
_NON_WEIGHT_ID_START = 2


@dataclass
class RegionEntry:
    region_id: int
    region_type: str
    page_min: int
    page_max: int
    first_seen_ns: int = 0
    last_seen_ns: int = 0
    access_count: int = 0


class AddressTracker:
    """
    Maps address pages to RegionEntry objects.

    Invariant: once a page is registered, its region_type never changes.
    """

    def __init__(self) -> None:
        self._pages: Dict[int, int] = {}            # address_page → region_id
        self._regions: Dict[int, RegionEntry] = {}  # region_id → RegionEntry
        self._next_id: int = _NON_WEIGHT_ID_START

    def update(
        self,
        address_page: int,
        region_type: str,
        timestamp_ns: int,
    ) -> RegionEntry:
        """
        Register or update the region for address_page.

        If the page is already tracked, returns the existing entry unchanged
        (classification is stable). Otherwise creates a new entry.
        """
        if address_page in self._pages:
            rid = self._pages[address_page]
            entry = self._regions[rid]
            entry.access_count += 1
            if timestamp_ns > entry.last_seen_ns:
                entry.last_seen_ns = timestamp_ns
            if address_page < entry.page_min:
                entry.page_min = address_page
            if address_page > entry.page_max:
                entry.page_max = address_page
            return entry

        # New page — assign region_id
        rid = _WEIGHT_REGION_ID if region_type == "weight" else self._next_id
        if region_type != "weight":
            self._next_id += 1

        if rid not in self._regions:
            self._regions[rid] = RegionEntry(
                region_id=rid,
                region_type=region_type,
                page_min=address_page,
                page_max=address_page,
                first_seen_ns=timestamp_ns,
                last_seen_ns=timestamp_ns,
                access_count=1,
            )
        else:
            entry = self._regions[rid]
            entry.access_count += 1
            if timestamp_ns > entry.last_seen_ns:
                entry.last_seen_ns = timestamp_ns
            if address_page < entry.page_min:
                entry.page_min = address_page
            if address_page > entry.page_max:
                entry.page_max = address_page

        self._pages[address_page] = rid
        return self._regions[rid]

    def lookup(self, address_page: int) -> Optional[RegionEntry]:
        """Return the existing RegionEntry for a page, or None if unseen."""
        rid = self._pages.get(address_page)
        return self._regions.get(rid) if rid is not None else None

    def region_map_rows(self) -> List[dict]:
        """Return rows for region_map.parquet, one row per region."""
        return [
            {
                "region_id": e.region_id,
                "region_type": e.region_type,
                "page_min": e.page_min,
                "page_max": e.page_max,
                "first_seen_ns": e.first_seen_ns,
                "last_seen_ns": e.last_seen_ns,
                "access_count": e.access_count,
            }
            for e in self._regions.values()
        ]

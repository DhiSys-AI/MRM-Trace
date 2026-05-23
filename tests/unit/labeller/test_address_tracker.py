"""Unit tests for AddressTracker."""

import pytest

from mrm_trace.labeller.address_tracker import (
    AddressTracker,
    RegionEntry,
    _WEIGHT_REGION_ID,
    _NON_WEIGHT_ID_START,
)


# ── weight region grouping ────────────────────────────────────────────────────

def test_weight_pages_all_get_region_id_1():
    tracker = AddressTracker()
    e1 = tracker.update(0x7F000, "weight", 100)
    e2 = tracker.update(0x7F001, "weight", 200)
    assert e1.region_id == _WEIGHT_REGION_ID == 1
    assert e2.region_id == _WEIGHT_REGION_ID == 1


def test_weight_region_aggregates_multiple_pages():
    tracker = AddressTracker()
    tracker.update(0x7F000, "weight", 100)
    tracker.update(0x7F005, "weight", 200)
    rows = tracker.region_map_rows()
    weight_rows = [r for r in rows if r["region_type"] == "weight"]
    assert len(weight_rows) == 1  # single merged weight region
    assert weight_rows[0]["access_count"] == 2


# ── non-weight region IDs ─────────────────────────────────────────────────────

def test_kv_cache_pages_get_sequential_ids_from_2():
    tracker = AddressTracker()
    e1 = tracker.update(0xAA000, "kv_cache", 100)
    e2 = tracker.update(0xBB000, "kv_cache", 200)
    assert e1.region_id == _NON_WEIGHT_ID_START
    assert e2.region_id == _NON_WEIGHT_ID_START + 1


def test_weight_and_kv_regions_do_not_share_ids():
    tracker = AddressTracker()
    w = tracker.update(0x7F000, "weight", 100)
    k = tracker.update(0xAA000, "kv_cache", 200)
    assert w.region_id != k.region_id


def test_mixed_types_get_distinct_ids():
    tracker = AddressTracker()
    ids = set()
    for i, rtype in enumerate(["kv_cache", "activation", "runtime"]):
        e = tracker.update(0x10000 + i * 0x1000, rtype, i * 100)
        ids.add(e.region_id)
    assert len(ids) == 3


# ── classification stability ──────────────────────────────────────────────────

def test_same_page_returns_same_entry():
    tracker = AddressTracker()
    e1 = tracker.update(0x7F000, "weight", 100)
    e2 = tracker.update(0x7F000, "weight", 200)
    assert e1 is e2


def test_second_update_increments_access_count():
    tracker = AddressTracker()
    tracker.update(0x7F000, "kv_cache", 100)
    e = tracker.update(0x7F000, "kv_cache", 200)
    assert e.access_count == 2


def test_second_update_advances_last_seen_ns():
    tracker = AddressTracker()
    tracker.update(0x7F000, "kv_cache", 100)
    e = tracker.update(0x7F000, "kv_cache", 999)
    assert e.last_seen_ns == 999


def test_first_classification_is_stable():
    tracker = AddressTracker()
    tracker.update(0x7F000, "weight", 100)
    # Try to update same page with different type — stable, keeps original
    e = tracker.update(0x7F000, "kv_cache", 200)
    assert e.region_type == "weight"


# ── lookup ────────────────────────────────────────────────────────────────────

def test_lookup_returns_none_for_unseen_page():
    assert AddressTracker().lookup(0x12345) is None


def test_lookup_returns_entry_for_seen_page():
    tracker = AddressTracker()
    tracker.update(0x12345, "kv_cache", 100)
    entry = tracker.lookup(0x12345)
    assert entry is not None
    assert entry.region_type == "kv_cache"


# ── page_min / page_max ───────────────────────────────────────────────────────

def test_weight_region_page_min_max_span_all_pages():
    tracker = AddressTracker()
    pages = [0x7F000, 0x7F001, 0x7F010]
    for p in pages:
        tracker.update(p, "weight", 100)
    rows = tracker.region_map_rows()
    w = next(r for r in rows if r["region_type"] == "weight")
    assert w["page_min"] == min(pages)
    assert w["page_max"] == max(pages)


# ── region_map_rows() ─────────────────────────────────────────────────────────

def test_region_map_rows_empty_tracker():
    assert AddressTracker().region_map_rows() == []


def test_region_map_rows_has_required_keys():
    tracker = AddressTracker()
    tracker.update(0x1000, "kv_cache", 1000)
    rows = tracker.region_map_rows()
    required = {"region_id", "region_type", "page_min", "page_max",
                "first_seen_ns", "last_seen_ns", "access_count"}
    assert required.issubset(rows[0].keys())


def test_region_map_one_row_per_region():
    tracker = AddressTracker()
    tracker.update(0x1000, "weight", 100)
    tracker.update(0x2000, "weight", 200)    # same region
    tracker.update(0x3000, "kv_cache", 300)  # new region
    rows = tracker.region_map_rows()
    assert len(rows) == 2  # weight region + 1 kv region

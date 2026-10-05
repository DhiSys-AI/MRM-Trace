"""Unit tests for KVBlockTracker."""

import pytest

from mrm_trace.labeller.kv_lifecycle import KVBlockEntry, KVBlockTracker


# ── initial store ─────────────────────────────────────────────────────────────

def test_first_store_records_write_ns():
    tracker = KVBlockTracker()
    tracker.record(address=0x1000, address_page=0x1, region_id=2, op_type="store", timestamp_ns=100)
    rows = tracker.lifecycle_rows()
    assert rows[0]["write_ns"] == 100


def test_first_store_total_reads_zero():
    tracker = KVBlockTracker()
    tracker.record(address=0x1000, address_page=0x1, region_id=2, op_type="store", timestamp_ns=100)
    rows = tracker.lifecycle_rows()
    assert rows[0]["total_reads"] == 0


def test_first_store_evict_ns_is_none():
    tracker = KVBlockTracker()
    tracker.record(address=0x1000, address_page=0x1, region_id=2, op_type="store", timestamp_ns=100)
    rows = tracker.lifecycle_rows()
    assert rows[0]["evict_ns"] is None


# ── load tracking ─────────────────────────────────────────────────────────────

def test_load_increments_total_reads():
    tracker = KVBlockTracker()
    tracker.record(0x1000, 0x1, 2, "store", 100)
    tracker.record(0x1000, 0x1, 2, "load", 200)
    tracker.record(0x1000, 0x1, 2, "load", 300)
    rows = tracker.lifecycle_rows()
    assert rows[0]["total_reads"] == 2


def test_load_updates_last_read_ns():
    tracker = KVBlockTracker()
    tracker.record(0x1000, 0x1, 2, "store", 100)
    tracker.record(0x1000, 0x1, 2, "load", 200)
    tracker.record(0x1000, 0x1, 2, "load", 500)
    rows = tracker.lifecycle_rows()
    assert rows[0]["last_read_ns"] == 500


def test_last_read_ns_not_decremented():
    tracker = KVBlockTracker()
    tracker.record(0x1000, 0x1, 2, "store", 100)
    tracker.record(0x1000, 0x1, 2, "load", 500)
    tracker.record(0x1000, 0x1, 2, "load", 200)  # earlier timestamp
    rows = tracker.lifecycle_rows()
    assert rows[0]["last_read_ns"] == 500


# ── eviction ──────────────────────────────────────────────────────────────────

def test_second_store_records_evict_ns():
    tracker = KVBlockTracker()
    tracker.record(0x1000, 0x1, 2, "store", 100)
    tracker.record(0x1000, 0x1, 2, "store", 999)
    rows = tracker.lifecycle_rows()
    assert rows[0]["evict_ns"] == 999


def test_third_store_does_not_update_evict_ns():
    tracker = KVBlockTracker()
    tracker.record(0x1000, 0x1, 2, "store", 100)
    tracker.record(0x1000, 0x1, 2, "store", 999)  # eviction
    tracker.record(0x1000, 0x1, 2, "store", 1200)  # re-write; evict_ns stays at 999
    rows = tracker.lifecycle_rows()
    assert rows[0]["evict_ns"] == 999


# ── lifetime_ns ───────────────────────────────────────────────────────────────

def test_lifetime_ns_uses_last_read_when_no_eviction():
    tracker = KVBlockTracker()
    tracker.record(0x1000, 0x1, 2, "store", 100)
    tracker.record(0x1000, 0x1, 2, "load", 600)
    rows = tracker.lifecycle_rows()
    assert rows[0]["lifetime_ns"] == 500  # 600 - 100


def test_lifetime_ns_uses_evict_when_evicted():
    tracker = KVBlockTracker()
    tracker.record(0x1000, 0x1, 2, "store", 100)
    tracker.record(0x1000, 0x1, 2, "load", 300)
    tracker.record(0x1000, 0x1, 2, "store", 800)  # eviction
    rows = tracker.lifecycle_rows()
    assert rows[0]["lifetime_ns"] == 700  # evict_ns(800) - write_ns(100)


def test_lifetime_ns_zero_for_write_only_block():
    tracker = KVBlockTracker()
    tracker.record(0x1000, 0x1, 2, "store", 100)
    rows = tracker.lifecycle_rows()
    assert rows[0]["lifetime_ns"] == 0


# ── multiple blocks ───────────────────────────────────────────────────────────

def test_multiple_blocks_tracked_independently():
    tracker = KVBlockTracker()
    tracker.record(0x1000, 0x1, 2, "store", 100)
    tracker.record(0x2000, 0x2, 3, "store", 200)
    tracker.record(0x1000, 0x1, 2, "load", 300)
    rows = tracker.lifecycle_rows()
    assert len(rows) == 2
    by_page = {r["address_page"]: r for r in rows}
    assert by_page[0x1]["total_reads"] == 1
    assert by_page[0x2]["total_reads"] == 0


def test_lifecycle_rows_empty_when_no_records():
    assert KVBlockTracker().lifecycle_rows() == []


def test_lifecycle_rows_has_required_keys():
    tracker = KVBlockTracker()
    tracker.record(0x1000, 0x1, 2, "store", 100)
    rows = tracker.lifecycle_rows()
    required = {"block_id", "address", "address_page", "write_ns",
                "last_read_ns", "evict_ns", "total_reads", "write_observed",
                "lifetime_ns"}
    assert required.issubset(rows[0].keys())


# ── unobserved writes ─────────────────────────────────────────────────────────
#
# Stores are sampled on a far sparser period than loads, so a KV block with many
# sampled loads and no sampled store is the *common* case, not an edge case.
# These tests pin the rule that an absent write is reported as absent: coercing
# it to 0 turned lifetime_ns into the raw monotonic clock reading (machine
# uptime, tens of hours) on 99% of blocks in a real sweep.

def test_lifetime_is_none_when_write_never_sampled():
    tracker = KVBlockTracker()
    tracker.record(0x1000, 1, 7, "load", 165_780_902_560_000)
    rows = tracker.lifecycle_rows()
    assert rows[0]["lifetime_ns"] is None
    assert rows[0]["write_ns"] is None
    assert rows[0]["write_observed"] is False


def test_unsampled_write_does_not_become_a_clock_reading():
    """The regression itself: lifetime must never equal the absolute timestamp."""
    tracker = KVBlockTracker()
    clock = 165_780_902_560_000  # ~46 h of uptime, a plausible monotonic value
    tracker.record(0x2000, 2, 8, "load", clock)
    assert tracker.lifecycle_rows()[0]["lifetime_ns"] != clock


def test_write_observed_true_when_store_sampled():
    tracker = KVBlockTracker()
    tracker.record(0x3000, 3, 9, "store", 100)
    tracker.record(0x3000, 3, 9, "load", 600)
    row = tracker.lifecycle_rows()[0]
    assert row["write_observed"] is True
    assert row["lifetime_ns"] == 500


def test_read_only_and_written_blocks_are_distinguishable():
    tracker = KVBlockTracker()
    tracker.record(0x4000, 4, 10, "load", 500)           # no write sampled
    tracker.record(0x5000, 5, 11, "store", 100)          # write sampled
    tracker.record(0x5000, 5, 11, "load", 400)
    by_addr = {r["address"]: r for r in tracker.lifecycle_rows()}
    assert by_addr[0x4000]["lifetime_ns"] is None
    assert by_addr[0x5000]["lifetime_ns"] == 300

"""
A derived duration may never exceed the run that produced it.

A KV block's lifetime is ``end - write_ns``.  When no store for that block was
sampled -- the common case, because stores run on a far sparser period than
loads -- an absent ``write_ns`` was coerced to 0, so the subtraction returned
the raw monotonic timestamp: the machine's uptime.  A real sweep shipped 42,260
of 42,670 blocks (99.0%) carrying "lifetimes" of 165,781 s inside runs of 73 s,
and every other gate stayed green: per-run audit PASS, verdict `clean`,
metrics.csv complete.

Two mechanisms are pinned here: the tracker records the absence instead of
imputing a zero, and the auditor rejects any lifetime longer than its own run.
"""

import json

import pandas as pd
import pytest

from mrm_trace.labeller.kv_lifecycle import KVBlockTracker
from mrm_trace.labeller.kv_rederive import rederive_kv_lifecycle
from mrm_trace.reporter.run_audit import audit_run

pytestmark = pytest.mark.validity

_KV_BASE = 0x7F10_0000_0000
_GRANULE = 65536
_CLOCK = 165_780_902_560_000      # a plausible monotonic reading, ~46 h of uptime
_RUN_S = 73.0


def _run_with_lifecycle(root, lifecycle_rows, *, duration_s=_RUN_S):
    """A minimal run directory carrying only what the lifetime check reads."""
    d = root / "run-kv"
    d.mkdir(parents=True, exist_ok=True)
    (d / "generation_results.json").write_text(json.dumps({"address_regions": []}))
    (d / "manifest.json").write_text("{}")
    pd.DataFrame(lifecycle_rows).to_parquet(d / "kv_block_lifecycle.parquet")
    pd.DataFrame([{"address": _KV_BASE, "region_type": "kv_cache",
                   "op_type": "load"}]).to_parquet(d / "trace.parquet")
    pd.DataFrame([{"region_id": 1, "region_type": "kv_cache", "page_min": 0,
                   "page_max": 1, "first_seen_ns": 0, "last_seen_ns": 1,
                   "access_count": 1}]).to_parquet(d / "region_map.parquet")
    pd.DataFrame([{
        "region_type": "kv_cache", "n_regions": 1, "retention_p50_s": 1.0,
        "retention_p90_s": 2.0, "retention_p99_s": 3.0, "retention_mean_s": 1.5,
        "write_once_ratio_mean": 1.0, "reads_per_write_mean": 5.0,
        "working_set_mb_total": 1.0, "unique_pages_total": 1,
        "suitability": "high_mrm",
    }]).to_csv(d / "metrics.csv", index=False)
    (d / "metadata.json").write_text(json.dumps({
        "thermal": {"peak_temp_c": 65.0, "throttle_events": 0,
                    "run_validity": "clean", "validity_failure_reasons": []},
        "observer_effect": {"observer_memory_overhead_mb_peak": 418.0,
                            "io_contention_detected": False,
                            "observer_cpu_overhead_pct_mean": 4.4},
        "hardware": {"mem_total_mb": 31742.0},
        "generation": {"total_duration_s": duration_s},
        "collector": {"sample_period": 1270, "event_strings": ["e"],
                      "region_granularity_bytes": _GRANULE,
                      "collection_duration_s": duration_s,
                      "perf_event_max_sample_rate_before": 100_000,
                      "perf_event_max_sample_rate_after": 100_000,
                      "achieved_sample_rate_per_cpu": 5_000.0},
    }))
    return d


def _block(**kw):
    row = {"block_id": 1, "address": _KV_BASE, "address_page": 1,
           "write_ns": 1_000, "last_read_ns": 2_000, "evict_ns": None,
           "total_reads": 3, "write_observed": True, "lifetime_ns": 1_000}
    row.update(kw)
    return row


# ── the tracker records absence rather than imputing a zero ───────────────────

def test_unsampled_write_yields_no_lifetime():
    tracker = KVBlockTracker()
    tracker.record(_KV_BASE, 1, 7, "load", _CLOCK, block_key=_KV_BASE // _GRANULE)
    row = tracker.lifecycle_rows()[0]
    assert row["write_observed"] is False
    assert row["lifetime_ns"] is None


def test_sampled_write_yields_a_real_duration():
    tracker = KVBlockTracker()
    key = _KV_BASE // _GRANULE
    tracker.record(_KV_BASE, 1, 7, "store", _CLOCK, block_key=key)
    tracker.record(_KV_BASE, 1, 7, "load", _CLOCK + 5_000_000_000, block_key=key)
    row = tracker.lifecycle_rows()[0]
    assert row["write_observed"] is True
    assert row["lifetime_ns"] == 5_000_000_000


# ── the auditor rejects a lifetime longer than its own run ────────────────────

def test_lifetime_exceeding_the_run_fails_the_audit(tmp_path):
    """The shipped defect: 165,781 s of lifetime inside a 73 s run."""
    d = _run_with_lifecycle(tmp_path, [
        _block(write_ns=0, last_read_ns=_CLOCK, write_observed=False,
               lifetime_ns=_CLOCK),
    ])
    audit = audit_run(d)
    assert any("kv_lifetime" in f for f in audit.failures), audit.failures


def test_a_bounded_lifetime_passes_the_audit(tmp_path):
    d = _run_with_lifecycle(tmp_path, [_block(lifetime_ns=12_000_000_000)])
    audit = audit_run(d)
    assert not any("kv_lifetime" in f for f in audit.failures), audit.failures


def test_null_lifetime_with_unobserved_write_passes(tmp_path):
    """The corrected representation must itself be acceptable."""
    d = _run_with_lifecycle(tmp_path, [
        _block(write_ns=None, write_observed=False, lifetime_ns=None),
    ])
    audit = audit_run(d)
    assert not any("kv_lifetime" in f for f in audit.failures), audit.failures


def test_the_two_columns_must_agree(tmp_path):
    """An observed write with no lifetime, or a lifetime with no write."""
    d = _run_with_lifecycle(tmp_path, [
        _block(write_observed=True, lifetime_ns=None),
    ])
    assert any("kv_lifetime" in f for f in audit_run(d).failures)

    d2 = _run_with_lifecycle(tmp_path / "b", [
        _block(write_observed=False, lifetime_ns=1_000),
    ])
    assert any("kv_lifetime" in f for f in audit_run(d2).failures)


# ── re-derivation reproduces the pipeline, so no artifact needs patching ──────

def test_rederivation_equals_live_tracking(tmp_path):
    """
    The correction is applied by re-deriving from trace.parquet, so the
    re-derived file must equal what the pipeline would have produced from the
    same rows.  Without this, a re-derived dataset is a patched one.
    """
    rows, reference = [], KVBlockTracker()
    ts = _CLOCK
    for i in range(400):
        address = _KV_BASE + i * 4096
        op = "store" if i % 50 == 0 else "load"
        ts += 1_000_000
        rows.append({"address": address, "address_page": address >> 12,
                     "region_id": 1 + i // 16, "op_type": op,
                     "timestamp_ns": ts, "region_type": "kv_cache"})
        reference.record(address=address, address_page=address >> 12,
                         region_id=1 + i // 16, op_type=op, timestamp_ns=ts,
                         block_key=address // _GRANULE)
    # Non-KV rows must be ignored by the re-derivation.
    rows.append({"address": 0xDEAD0000, "address_page": 0xDEAD,
                 "region_id": 999, "op_type": "store", "timestamp_ns": ts + 1,
                 "region_type": "runtime"})

    d = tmp_path / "run-rd"
    d.mkdir()
    pd.DataFrame(rows).to_parquet(d / "trace.parquet")
    (d / "metadata.json").write_text(json.dumps(
        {"collector": {"region_granularity_bytes": _GRANULE}}))

    stats = rederive_kv_lifecycle(d)
    expected = reference.lifecycle_rows()
    assert stats["blocks"] == len(expected)

    got = pd.read_parquet(d / "kv_block_lifecycle.parquet")
    want = pd.DataFrame(expected)
    key = "address"
    got = got.sort_values(key).reset_index(drop=True)
    want = want.sort_values(key).reset_index(drop=True)
    for col in ("address", "address_page", "total_reads", "write_observed"):
        assert got[col].tolist() == want[col].tolist(), col
    assert (pd.to_numeric(got["lifetime_ns"], errors="coerce").fillna(-1).tolist()
            == pd.to_numeric(want["lifetime_ns"], errors="coerce").fillna(-1).tolist())


def test_rederivation_needs_a_trace(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        rederive_kv_lifecycle(d)

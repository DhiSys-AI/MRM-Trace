"""
The auditor is the mechanism that replaces noticing.

Every defect it checks for shipped a run with both quality gates PASS, a
`clean` verdict and a complete metrics.csv:

  - a symbol rule labelled 26,617,751 rows `weight`, 114 of which were inside
    the weight mapping;
  - a run reported `clean` while its own written metadata re-derived to
    `marginal`, because the live classifier was built from a partial dict;
  - `reads_per_write_mean` came out inf for a region type that had writes,
    because a mean of per-region ratios inherited inf from a read-only
    sub-region.

So the auditor itself needs tests: it is only worth having if it fails when it
should.
"""

import json

import pandas as pd
import pytest

from mrm_trace.reporter.run_audit import audit_run

pytestmark = pytest.mark.validity

_W_BASE = 0x7F00_0000_0000
_KV_BASE = 0x7F10_0000_0000


def _write_run(root, *, weight_rows_outside=0, verdict="clean",
               reads_per_write=5.0, write_once=0.9, weight_stores=0,
               achieved=5_000.0, cap_after=100_000, exports=True):
    d = root / "run-1"
    d.mkdir(parents=True, exist_ok=True)
    (d / "generation_results.json").write_text(json.dumps({"address_regions": [
        {"region_type": "weight", "addr_start": _W_BASE,
         "addr_end": _W_BASE + 0x10000, "source": "mmap_gguf"},
        {"region_type": "kv_cache", "addr_start": _KV_BASE,
         "addr_end": _KV_BASE + 0x10000, "source": "kv_allocation"},
    ]}))
    if not exports:
        return d

    rows = []
    for i in range(100):                                  # weight, in range
        rows.append({"address": _W_BASE + i * 64, "region_type": "weight",
                     "op_type": "load"})
    for i in range(weight_stores):                        # impossible stores
        rows.append({"address": _W_BASE + i * 64, "region_type": "weight",
                     "op_type": "store"})
    for i in range(weight_rows_outside):                  # mis-attributed
        rows.append({"address": 0xDEAD_0000_0000 + i * 64,
                     "region_type": "weight", "op_type": "store"})
    for i in range(50):                                   # kv, in range
        rows.append({"address": _KV_BASE + i * 256, "region_type": "kv_cache",
                     "op_type": "store" if i % 10 == 0 else "load"})
    pd.DataFrame(rows).to_parquet(d / "trace.parquet")

    pd.DataFrame([{"region_id": 1, "region_type": "weight", "page_min": 0,
                   "page_max": 1, "first_seen_ns": 0, "last_seen_ns": 1,
                   "access_count": 100}]).to_parquet(d / "region_map.parquet")
    pd.DataFrame([{
        "region_type": "weight", "n_regions": 1, "retention_p50_s": 40.0,
        "retention_p90_s": 60.0, "retention_p99_s": 62.0,
        "retention_mean_s": 38.0, "write_once_ratio_mean": write_once,
        "reads_per_write_mean": reads_per_write, "working_set_mb_total": 80.0,
        "unique_pages_total": 100, "suitability": "high_mrm",
    }]).to_csv(d / "metrics.csv", index=False)
    (d / "manifest.json").write_text("{}")
    (d / "metadata.json").write_text(json.dumps({
        "thermal": {"peak_temp_c": 65.0, "throttle_events": 0,
                    "run_validity": verdict, "validity_failure_reasons": [],
                    "collection_wall_duration_s": 73.0},
        "observer_effect": {"observer_memory_overhead_mb_peak": 418.0,
                            "io_contention_detected": False,
                            "observer_cpu_overhead_pct_mean": 4.4},
        "hardware": {"mem_total_mb": 31742.0},
        "collector": {"sample_period": 1270, "event_strings": ["e"],
                      "region_granularity_bytes": 65536,
                      "perf_event_max_sample_rate_before": 100_000,
                      "perf_event_max_sample_rate_after": cap_after,
                      "achieved_sample_rate_per_cpu": achieved},
    }))
    return d


def test_a_sound_run_passes(tmp_path):
    audit = audit_run(_write_run(tmp_path))
    assert audit.failures == [], audit.failures
    assert not audit.incomplete


def test_mis_attributed_rows_fail(tmp_path):
    """Rows labelled weight outside the weight mapping — the original defect."""
    audit = audit_run(_write_run(tmp_path, weight_rows_outside=500))
    assert any("attribution/weight" in f and "precision" in f
               for f in audit.failures), audit.failures


def test_a_store_in_the_read_only_weight_mapping_fails(tmp_path):
    """The GGUF file is mapped r--s, so a store there is impossible."""
    audit = audit_run(_write_run(tmp_path, weight_stores=5))
    assert any("read-only mapping" in f for f in audit.failures), audit.failures


def test_a_verdict_that_disagrees_with_its_metadata_fails(tmp_path):
    """
    The run reported `clean` while the written metadata re-derived to
    `marginal`; the live classifier had been handed a partial dict.
    """
    audit = audit_run(_write_run(tmp_path, verdict="clean",
                                 cap_after=50_000, achieved=40_000))
    assert any("verdict" in f for f in audit.failures), audit.failures


def test_an_unbounded_ratio_with_observed_writes_fails(tmp_path):
    """inf alongside a defined write-once ratio is the mean-of-ratios bug."""
    audit = audit_run(_write_run(tmp_path, reads_per_write=float("inf"),
                                 write_once=0.9))
    assert any("reads_per_write_mean is inf" in f
               for f in audit.failures), audit.failures


def test_an_unbounded_ratio_for_a_read_only_region_is_accepted(tmp_path):
    """Reads with no writes anywhere is genuinely unbounded; write_once is NaN."""
    audit = audit_run(_write_run(tmp_path, reads_per_write=float("inf"),
                                 write_once=float("nan")))
    assert not any("reads_per_write" in f for f in audit.failures), audit.failures


def test_a_non_finite_retention_fails(tmp_path):
    d = _write_run(tmp_path)
    metrics = pd.read_csv(d / "metrics.csv")
    metrics.loc[0, "retention_p99_s"] = float("nan")
    metrics.to_csv(d / "metrics.csv", index=False)
    audit = audit_run(d)
    assert any("retention_p99_s" in f for f in audit.failures), audit.failures


def test_an_interrupted_run_is_skipped_not_failed(tmp_path):
    """
    A directory with raw/ but no exports is debris to delete.  Calling it a
    failure buries the runs that really are wrong.
    """
    audit = audit_run(_write_run(tmp_path, exports=False))
    assert audit.incomplete
    assert audit.failures == []


def test_a_cap_reduction_well_below_the_achieved_rate_is_only_a_note(tmp_path):
    audit = audit_run(_write_run(tmp_path, cap_after=63_000, achieved=5_242))
    assert audit.failures == []
    assert any("lowered the cap" in n for n in audit.notes)


# ── a run that stopped early is not the run that was planned ─────────────────

def test_a_short_generation_fails_when_eos_was_meant_to_be_suppressed(tmp_path):
    """
    Asking Llama-3.2-1B for 1977 tokens produced 653: the KV cache reached 33%
    occupancy instead of 98%, and the run passed every gate it had.  With
    ignore_eos set, a shortfall means a stop token nobody named, which silently
    turns an occupancy sweep into repetitions of one experiment.
    """
    d = _write_run(tmp_path)
    meta = json.loads((d / "metadata.json").read_text())
    meta["generation"] = {"prompts": 1, "output_tokens": 1977,
                          "tokens_requested_per_prompt": 1977,
                          "total_output_tokens": 653, "ignore_eos": True}
    (d / "metadata.json").write_text(json.dumps(meta))
    audit = audit_run(d)
    assert any("tokens" in f and "short" in f for f in audit.failures), audit.failures


def test_a_short_generation_is_only_a_note_without_ignore_eos(tmp_path):
    """Then the ceiling is expected behaviour, and worth recording not failing."""
    d = _write_run(tmp_path)
    meta = json.loads((d / "metadata.json").read_text())
    meta["generation"] = {"prompts": 1, "output_tokens": 1977,
                          "tokens_requested_per_prompt": 1977,
                          "total_output_tokens": 653, "ignore_eos": False}
    (d / "metadata.json").write_text(json.dumps(meta))
    audit = audit_run(d)
    assert not any("tokens" in f for f in audit.failures), audit.failures
    assert any("ceiling" in n for n in audit.notes), audit.notes


def test_reaching_the_limit_is_silent(tmp_path):
    d = _write_run(tmp_path)
    meta = json.loads((d / "metadata.json").read_text())
    meta["generation"] = {"prompts": 1, "output_tokens": 1977,
                          "tokens_requested_per_prompt": 1977,
                          "total_output_tokens": 1977, "ignore_eos": True}
    (d / "metadata.json").write_text(json.dumps(meta))
    audit = audit_run(d)
    assert not any("tokens" in f for f in audit.failures), audit.failures
    assert not any("tokens:" in n for n in audit.notes), audit.notes

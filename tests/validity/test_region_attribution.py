"""
Region attribution must rest on address ranges, not on instruction symbols.

A PMU sample carries an instruction pointer and a data address.  The symbol
resolved from the instruction pointer says which code was running; it does not
say which region the data address belongs to.  Measured on a real run
(Llama-3.2-1B, ctx 2048, 28,321,245 samples):

  rows a ggml_vec_dot rule labelled `weight`      26,617,751
  of those, inside the mmap'd GGUF file                 114   (0.0004%)

The GGUF mapping is r--s, so a store can never be a weight address at all.
And because attribution is per granule, one mis-attributed sample captures the
whole granule: four granules tagged by rare llama_kv_cache_* samples — one of
them in the main stack — absorbed 279,186 accesses from nearest_int, memmove
and softmax.  See context-files/04 - Region Attribution Validity.md.
"""

import importlib.util
import pathlib

import numpy as np

import pytest

from mrm_trace.collector.kv_allocation import (
    build_interposer,
    interposer_env,
    kv_regions_from_alloc_log,
    parse_alloc_log,
)
from mrm_trace.labeller.labeller import TraceLabeller
from mrm_trace.labeller.symbol_rules import DEFAULT_RULES, classify_symbol

pytestmark = pytest.mark.validity

_DEMO = pathlib.Path(__file__).resolve().parents[2] / "notebooks" / "scripts" / "demo_perf_mem.py"


@pytest.fixture(scope="module")
def demo():
    spec = importlib.util.spec_from_file_location("demo_perf_mem", _DEMO)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── no symbol rule may establish weight or kv_cache ──────────────────────────

def test_no_rule_produces_weight_or_kv_cache():
    """Enforced on the rule table itself, so a new rule cannot reintroduce it."""
    claimed = {r.region_type for r in DEFAULT_RULES}
    assert "weight" not in claimed
    assert "kv_cache" not in claimed


@pytest.mark.parametrize("symbol", [
    "ggml_vec_dot_q6_K_q8_K", "ggml_gemv_q4_K_8x8_q8_K", "ggml_mul_mat",
    "ggml_init", "ggml_new_tensor", "model_load",
])
def test_weight_reading_kernels_do_not_claim_weight(symbol):
    for op in ("load", "store"):
        assert classify_symbol(symbol, "libggml-cpu.so", "generation", op) != "weight"


@pytest.mark.parametrize("symbol", [
    "llama_kv_cache_find_slot", "llama_kv_cache_update", "kv_write", "kv_read",
])
def test_kv_bookkeeping_symbols_do_not_claim_kv_cache(symbol):
    """
    llama_kv_cache_* manages slot metadata; the K/V tensor writes happen in
    generic ggml ops.  Such a sample is evidence against a KV address.
    """
    for op in ("load", "store"):
        assert classify_symbol(symbol, "libllama.so", "generation", op) != "kv_cache"


def test_a_symbol_cannot_override_an_address_range():
    """
    The inverted precedence was the amplifier: symbol-KV evidence outranked the
    address map, so one sample claimed a granule of whatever it landed in.
    """
    lab = TraceLabeller()
    lab.register_address_ranges([{
        "region_type": "runtime", "source": "thread_stack",
        "addr_start": 0x7FFD_0000_0000, "addr_end": 0x7FFD_0001_0000,
    }])
    row = {
        "run_id": "r", "sample_id": 0, "timestamp_ns": 1, "phase": "generation",
        "address": 0x7FFD_0000_0100, "address_page": 0x7FFD_0000_0100 >> 12,
        "op_type": "store", "symbol": "llama_kv_cache_find_slot",
        "dso": "libllama.so", "region_type": "unknown", "region_id": 0,
        "access_latency_ns": None, "cache_level": None, "numa_node": None,
        "cpu_id": 0, "thread_id": 1, "token_index": None, "context_tokens": None,
    }
    out = list(lab.label([row]))
    assert out[0]["region_type"] == "runtime"


# ── load-dense event configuration ───────────────────────────────────────────

def test_events_carry_independent_periods(demo):
    events = demo.build_perf_events(load_period=2003, ldlat=3, store_multiplier=500)
    assert events == [
        "cpu/mem-loads,ldlat=3,period=2003/P",
        "cpu/mem-stores,period=1001500/P",
    ]


def test_stores_are_sampled_far_more_coarsely_than_loads(demo):
    """
    Weights are read-only, so weight observation lives entirely in the load
    channel.  Under one shared period the trace was 99.77% stores and carried
    114 weight samples in 28.3M.
    """
    loads, stores = demo.build_perf_events(2003, 3)
    load_period = int(loads.split("period=")[1].split("/")[0])
    store_period = int(stores.split("period=")[1].split("/")[0])
    assert store_period >= 100 * load_period


def test_ldlat_default_is_low_enough_to_see_cached_reads(demo):
    """
    perf mem's default ldlat=30 discards weight reads that hit in L1/L2 — of
    the 114 weight samples observed, 87 were L1 or L2 hits.  ldlat=3 captured
    25x more eligible loads.
    """
    assert demo._DEFAULT_LDLAT <= 5


def test_the_probe_rejects_an_unsupported_event_spec(demo):
    ok, msg = demo.probe_perf_events(["cpu/definitely-not-an-event,period=1/P"])
    assert ok is False
    assert msg


# ── KV address capture ───────────────────────────────────────────────────────

def _log(tmp_path, lines):
    p = tmp_path / "alloc.log"
    p.write_text("".join(f"{l}\n" for l in lines))
    return p


def test_exact_allocation_match_yields_the_kv_range(tmp_path):
    path = _log(tmp_path, [
        "100 malloc 262668288 0x7344bcd7c010",
        "100 posix_memalign 67108864 0x7344dca7e040",
        "100 mmap 33554432 0x734553e00000",
    ])
    regions = kv_regions_from_alloc_log(path, 67_108_864, pid=100)
    assert len(regions) == 1
    assert regions[0] == {
        "region_type": "kv_cache",
        "addr_start": 0x7344dca7e040,
        "addr_end": 0x7344dca7e040 + 67_108_864,
        "size_bytes": 67_108_864,
        "source": "kv_allocation",
    }


def test_an_ambiguous_size_yields_nothing(tmp_path):
    """
    At ctx 1024 the implied 32 MiB collides with glibc's 32 MiB arena
    reservations.  Picking one would be the guessing this replaces.
    """
    path = _log(tmp_path, [
        "100 mmap 33554432 0x734553e00000",
        "100 mmap 33554432 0x734551e00000",
    ])
    assert kv_regions_from_alloc_log(path, 33_554_432, pid=100) == []


def test_no_match_yields_nothing(tmp_path):
    path = _log(tmp_path, ["100 malloc 262668288 0x7344bcd7c010"])
    assert kv_regions_from_alloc_log(path, 67_108_864, pid=100) == []


def test_unknown_geometry_yields_nothing(tmp_path):
    path = _log(tmp_path, ["100 posix_memalign 67108864 0x7344dca7e040"])
    assert kv_regions_from_alloc_log(path, None, pid=100) == []


def test_allocations_from_other_processes_are_ignored(tmp_path):
    """perf is itself under LD_PRELOAD; its allocations must not be mistaken."""
    path = _log(tmp_path, ["999 posix_memalign 67108864 0xdead0000"])
    assert kv_regions_from_alloc_log(path, 67_108_864, pid=100) == []
    assert len(kv_regions_from_alloc_log(path, 67_108_864, pid=999)) == 1


def test_a_missing_log_is_not_an_error(tmp_path):
    assert parse_alloc_log(tmp_path / "nope.log") == []
    assert kv_regions_from_alloc_log(tmp_path / "nope.log", 67_108_864) == []


def test_malformed_lines_are_skipped(tmp_path):
    path = _log(tmp_path, ["garbage", "1 2", "100 posix_memalign notanint 0x1",
                           "100 posix_memalign 67108864 0x7344dca7e040"])
    assert len(kv_regions_from_alloc_log(path, 67_108_864, pid=100)) == 1


def test_interposer_builds_and_produces_a_loadable_env(tmp_path):
    lib = build_interposer()
    if lib is None:
        pytest.skip("no C compiler available")
    assert lib.exists() and lib.suffix == ".so"
    env = interposer_env(lib, tmp_path / "alloc.log")
    assert str(lib) in env["LD_PRELOAD"]
    assert env["MRM_ALLOC_LOG"] == str(tmp_path / "alloc.log")


def test_env_is_empty_without_a_library():
    """A failed build must leave the run unaffected, with kv_cache absent."""
    assert interposer_env(None, pathlib.Path("/tmp/x")) == {}


# ── the sampling cap must be pinned, not inherited ───────────────────────────

@pytest.fixture(scope="module")
def runner():
    path = (pathlib.Path(__file__).resolve().parents[2]
            / "notebooks" / "scripts" / "experiment_runner.py")
    spec = importlib.util.spec_from_file_location("experiment_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_sweep_refuses_to_start_below_the_floor(runner, monkeypatch):
    """
    A single demo run can warn, but a 135-run sweep collected across a changing
    cap is not salvageable afterwards — half the dataset samples at a different
    density with nothing in the output to show it.
    """
    monkeypatch.setattr(runner, "read_sample_rate", lambda: 1000)
    monkeypatch.setattr(runner, "set_sample_rate", lambda v: False)
    assert runner.ensure_sample_rate(lambda *_: None) is False


def test_the_sweep_raises_the_cap_when_it_can(runner, monkeypatch):
    applied = []
    monkeypatch.setattr(runner, "read_sample_rate", lambda: 1000)
    monkeypatch.setattr(runner, "set_sample_rate",
                        lambda v: applied.append(v) or True)
    assert runner.ensure_sample_rate(lambda *_: None) is True
    assert applied == [runner.PERF_SAMPLE_RATE_TARGET]


def test_an_adequate_cap_is_left_alone(runner, monkeypatch):
    monkeypatch.setattr(runner, "read_sample_rate",
                        lambda: runner.PERF_SAMPLE_RATE_TARGET)
    monkeypatch.setattr(runner, "set_sample_rate",
                        lambda v: pytest.fail("must not rewrite an adequate cap"))
    assert runner.ensure_sample_rate(lambda *_: None) is True


def test_an_unreadable_cap_does_not_block_the_sweep(runner, monkeypatch):
    """Non-Linux or a restricted procfs should not be a hard stop."""
    monkeypatch.setattr(runner, "read_sample_rate", lambda: None)
    assert runner.ensure_sample_rate(lambda *_: None) is True


def test_a_mid_run_cap_drop_costs_the_run_its_clean_status():
    """
    Density higher at the start than at the end biases retention, which is
    last-access minus first-access of a region.
    """
    from mrm_trace.telemetry.validity import classify_from_metadata
    meta = {
        "thermal": {"peak_temp_c": 70.0, "throttle_events": 0},
        "observer_effect": {"observer_memory_overhead_mb_peak": 2512.0,
                            "io_contention_detected": False},
        "hardware": {"mem_total_mb": 31742.0},
        "collector": {"perf_event_max_sample_rate_before": 100_000,
                      "perf_event_max_sample_rate_after": 1_000},
    }
    result = classify_from_metadata(meta)
    assert result.run_validity == "marginal"
    assert any("perf_event_max_sample_rate" in r for r in result.failure_reasons)

    steady = {**meta, "collector": {"perf_event_max_sample_rate_before": 100_000,
                                    "perf_event_max_sample_rate_after": 100_000}}
    assert classify_from_metadata(steady).run_validity == "clean"


# ── a read-only region must not be scored as the worst MRM candidate ─────────

def _suitability(region_type, reads, writes, p99, p50, wo_ratio=None):
    import pandas as pd
    from mrm_trace.analyser.suitability import classify_suitability
    retention = pd.DataFrame([{
        "region_type": region_type, "retention_p99_s": p99,
        "retention_p50_s": p50, "retention_p90_s": p99, "retention_mean_s": p50,
    }])
    wo = pd.DataFrame(
        [] if wo_ratio is None
        else [{"region_id": 1, "region_type": region_type, "write_once_ratio": wo_ratio}],
        columns=["region_id", "region_type", "write_once_ratio"],
    )
    rf = pd.DataFrame([{"region_id": 1, "region_type": region_type,
                        "total_reads": reads, "total_writes": writes}])
    return classify_suitability(retention, wo, rf).iloc[0]


def test_a_read_only_long_retention_region_is_high_mrm():
    """
    The central claim this instrument tests: weights are write-once, read-many
    and long-lived.  With use_mmap llama.cpp maps the GGUF read-only, so the
    process never stores to them — write_once_ratio is 0/0, undefined.
    Filling that with 0.0 produced `low_mrm` for the weight region, inverting
    the result.  Measured on a real run: 26,757 reads, 0 writes, retention p99
    62.5 s -> low_mrm.
    """
    row = _suitability("weight", reads=26_757, writes=0, p99=62.5, p50=44.3)
    assert row["write_observability"] == "read_only_no_writes"
    assert row["suitability"] == "high_mrm"


def test_a_read_only_region_with_no_retention_is_not_promoted():
    """The exemption is only from the write-once gate, not from retention."""
    row = _suitability("activation", reads=1000, writes=0, p99=0.2, p50=0.05)
    assert row["write_observability"] == "read_only_no_writes"
    assert row["suitability"] == "low_mrm"


def test_a_region_with_observed_writes_still_uses_the_write_once_gate():
    low = _suitability("runtime", reads=100, writes=100, p99=60.0, p50=40.0,
                       wo_ratio=0.1)
    assert low["write_observability"] == "observed"
    assert low["suitability"] == "low_mrm"

    high = _suitability("kv_cache", reads=100, writes=100, p99=60.0, p50=40.0,
                        wo_ratio=1.0)
    assert high["suitability"] == "high_mrm"


def test_a_region_with_no_accesses_is_not_given_a_verdict():
    row = _suitability("unknown", reads=0, writes=0, p99=0.0, p50=0.0)
    assert row["write_observability"] == "no_accesses"
    assert row["suitability"] == "unknown"


def test_reads_per_write_is_a_ratio_of_totals_not_a_mean_of_ratios():
    """
    A region type almost always contains some read-only sub-regions, whose
    per-region ratio is infinite.  Averaging those gives inf for the whole
    type even when it has plenty of writes — observed on a real run: kv_cache
    with 2,988 reads and 17 writes reported inf instead of 176.
    """
    import pandas as pd
    import tempfile, pathlib
    from mrm_trace.reporter.metrics_csv import write_metrics_csv

    analysis = {
        "read_freq": pd.DataFrame([
            # two sub-regions with writes, one read-only
            {"region_id": 1, "region_type": "kv_cache", "total_reads": 1000,
             "total_writes": 10, "reads_per_write": 100.0},
            {"region_id": 2, "region_type": "kv_cache", "total_reads": 1988,
             "total_writes": 7, "reads_per_write": 284.0},
            {"region_id": 3, "region_type": "kv_cache", "total_reads": 500,
             "total_writes": 0, "reads_per_write": float("inf")},
        ]),
        "retention_summary": pd.DataFrame([{
            "region_type": "kv_cache", "retention_p50_s": 30.0,
            "retention_p90_s": 57.0, "retention_p99_s": 63.0,
            "retention_mean_s": 28.0,
        }]),
        "write_once": pd.DataFrame([{"region_id": 1, "region_type": "kv_cache",
                                     "write_once_ratio": 1.0}]),
        "working_set_summary": pd.DataFrame([{"region_type": "kv_cache",
                                              "total_pages": 2364,
                                              "total_bytes": 9_682_944}]),
        "suitability": pd.DataFrame([{"region_type": "kv_cache",
                                      "suitability": "high_mrm"}]),
    }
    out = pathlib.Path(tempfile.mkdtemp()) / "metrics.csv"
    write_metrics_csv(analysis, out)
    row = pd.read_csv(out).iloc[0]

    # (1000 + 1988 + 500) / (10 + 7) = 205.2, not the mean of 100/284/inf.
    assert row["reads_per_write_mean"] == pytest.approx(3488 / 17, rel=1e-6)
    assert np.isfinite(row["reads_per_write_mean"])

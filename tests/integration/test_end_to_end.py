"""
End-to-end integration test: raw perf-script rows → label → analyse → export.

Simulates the full pipeline that the CLI orchestrates, using only in-process
components (no shell subprocesses, no real perf data).
"""

import hashlib
import json

import pandas as pd
import pytest

from mrm_trace.analyser import compute_all
from mrm_trace.labeller.labeller import TraceLabeller
from mrm_trace.reporter.run_exporter import RunExporter
from mrm_trace.schema_version import check_parquet_schema


def _raw_row(address, symbol, dso, phase, op_type, ts_ns, sample_id):
    return {
        "run_id": "e2e_run", "sample_id": sample_id, "timestamp_ns": ts_ns,
        "phase": phase, "address": address, "address_page": address >> 12,
        "op_type": op_type, "symbol": symbol, "dso": dso,
        "region_type": "unknown", "region_id": 0,
        "access_latency_ns": None, "cache_level": None, "numa_node": None,
        "cpu_id": 0, "thread_id": 1, "token_index": None, "context_tokens": None,
    }


@pytest.fixture(scope="module")
def e2e_run(tmp_path_factory):
    """Build a complete run from raw rows → labelled trace → analysis → export."""
    raw_rows = []
    sid = 0

    # Weight: 1 store, 500 reads
    base_w = 0x7F00_0000_0000
    raw_rows.append(_raw_row(base_w, "ggml_init", "libggml.so",
                             "weight_load", "store", 0, sid))
    sid += 1
    for i in range(500):
        # spread reads over 30 s so weight retention > 10s → high_mrm threshold
        raw_rows.append(_raw_row(base_w + (i % 8) * 4096,
                                 "ggml_vec_dot_q4_K", "libggml.so",
                                 "generation", "load", int(i * 60_000_000), sid))
        sid += 1

    # KV: 50 blocks, 5 reads each
    base_kv = 0x7F10_0000_0000
    for i in range(50):
        addr = base_kv + i * 256
        raw_rows.append(_raw_row(addr, "llama_kv_cache_update", "libllama.so",
                                 "generation", "store", int(2e9) + i * 10_000_000, sid))
        sid += 1
        for j in range(5):
            raw_rows.append(_raw_row(addr, "llama_kv_cache_update", "libllama.so",
                                     "generation", "load",
                                     int(2e9) + i * 10_000_000 + (j + 1) * 1_000_000, sid))
            sid += 1

    # Label.  The fixture's ground truth is 50 KV blocks of 256 B, so the
    # labeller is run at that granularity — at the 4 KiB default those 50 blocks
    # share 4 pages and would correctly resolve to 4 regions.
    labeller = TraceLabeller(region_granularity_bytes=256)
    labelled = list(labeller.label(raw_rows))
    trace_df = pd.DataFrame(labelled)
    region_map = labeller.region_map()
    kv_lifecycle = labeller.kv_lifecycle()

    # Analyse
    analysis = compute_all(trace_df)

    # Export
    run_dir = tmp_path_factory.mktemp("e2e_run")
    exporter = RunExporter(run_dir)
    result = exporter.export(
        trace=trace_df,
        region_map=region_map,
        kv_lifecycle=kv_lifecycle,
        analysis=analysis,
        metadata={"run_id": "e2e_run", "model": "test", "context_tokens": 512},
        run_id="e2e_run",
        figures=False,
    )
    return result, run_dir, trace_df, analysis


class TestLabellingCorrectness:

    def test_all_rows_classified(self, e2e_run):
        _, _, trace_df, _ = e2e_run
        assert (trace_df["region_type"] != "unknown").all()

    def test_weight_regions_track_granules_not_a_single_id(self, e2e_run):
        """
        The fixture writes weights across 8 pages (base_w + (i % 8) * 4096) at
        256 B granularity, so weight resolves to 8 distinct regions — giving
        retention 8 samples to build a distribution from instead of 1.
        """
        _, _, trace_df, _ = e2e_run
        weight = trace_df[trace_df["region_type"] == "weight"]
        granules = weight["address"] // 256
        assert weight["region_id"].nunique() == granules.nunique()
        assert weight["region_id"].nunique() > 1

    def test_kv_blocks_have_distinct_region_ids(self, e2e_run):
        _, _, trace_df, _ = e2e_run
        kv_stores = trace_df[
            (trace_df["region_type"] == "kv_cache") & (trace_df["op_type"] == "store")
        ]
        assert kv_stores["region_id"].nunique() == 50


class TestAnalysisCorrectness:

    def test_weight_suitability_is_high_mrm(self, e2e_run):
        _, _, _, analysis = e2e_run
        suit = analysis["suitability"]
        w = suit[suit["region_type"] == "weight"].iloc[0]
        assert w["suitability"] == "high_mrm"

    def test_weight_write_once_ratio_is_1(self, e2e_run):
        _, _, _, analysis = e2e_run
        wo = analysis["write_once"]
        w = wo[wo["region_type"] == "weight"].iloc[0]
        assert w["write_once_ratio"] == 1.0

    def test_kv_lifecycle_has_50_blocks(self, e2e_run):
        _, _, trace_df, analysis = e2e_run
        kv_regions = analysis["retention_per_region"]
        kv = kv_regions[kv_regions["region_type"] == "kv_cache"]
        assert len(kv) == 50

    def test_kv_lifecycle_file_agrees_with_region_count(self, e2e_run):
        """
        kv_block_lifecycle.parquet and region_map.parquet must count blocks the
        same way.  They diverged before — region ids were per address while the
        lifecycle tracker keyed pages — so per-block retention in one file could
        not be joined to region stats in the other.
        """
        _, run_dir, trace_df, analysis = e2e_run
        lifecycle = pd.read_parquet(run_dir / "kv_block_lifecycle.parquet")
        kv = analysis["retention_per_region"]
        kv = kv[kv["region_type"] == "kv_cache"]
        assert len(lifecycle) == len(kv) == 50
        assert set(lifecycle["block_id"]) == set(kv["region_id"])


class TestExportCompleteness:

    def test_all_required_files_present(self, e2e_run):
        _, run_dir, _, _ = e2e_run
        for fname in ("trace.parquet", "region_map.parquet",
                      "kv_block_lifecycle.parquet", "metrics.csv",
                      "metadata.json", "manifest.json"):
            assert (run_dir / fname).exists(), f"Missing {fname}"

    def test_manifest_checksums_valid(self, e2e_run):
        result, run_dir, _, _ = e2e_run
        manifest = json.loads(result.manifest_path.read_text())
        for entry in manifest["files"]:
            fpath = run_dir / entry["path"]
            if fpath.exists():
                actual = hashlib.sha256(fpath.read_bytes()).hexdigest()
                assert actual == entry["sha256"]

    def test_trace_parquet_schema_valid(self, e2e_run):
        _, run_dir, _, _ = e2e_run
        check_parquet_schema(run_dir / "trace.parquet", "trace")

    def test_region_map_parquet_schema_valid(self, e2e_run):
        _, run_dir, _, _ = e2e_run
        check_parquet_schema(run_dir / "region_map.parquet", "region_map")

    def test_kv_lifecycle_parquet_schema_valid(self, e2e_run):
        _, run_dir, _, _ = e2e_run
        check_parquet_schema(run_dir / "kv_block_lifecycle.parquet", "kv_block_lifecycle")

    def test_metadata_json_round_trips(self, e2e_run):
        _, run_dir, _, _ = e2e_run
        data = json.loads((run_dir / "metadata.json").read_text())
        assert data["run_id"] == "e2e_run"
        assert data["context_tokens"] == 512

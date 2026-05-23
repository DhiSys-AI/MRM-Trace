"""Unit tests for RunExporter."""

import json

import pandas as pd
import pytest

from mrm_trace.reporter.run_exporter import ExportResult, RunExporter


def _trace():
    return pd.DataFrame([{
        "run_id": "r1", "sample_id": 0, "timestamp_ns": 0,
        "address": 0x1000, "address_page": 1, "op_type": "store",
        "region_type": "weight", "region_id": 1, "phase": "weight_load",
        "symbol": "ggml_init", "dso": "lib.so",
        "access_latency_ns": None, "cache_level": None, "numa_node": None,
        "cpu_id": 0, "thread_id": 1, "token_index": None, "context_tokens": None,
    }])


def _region_map():
    return [{"region_id": 1, "region_type": "weight", "page_min": 1, "page_max": 1,
             "first_seen_ns": 0, "last_seen_ns": 0, "access_count": 1}]


def _lifecycle():
    return [{"block_id": 2, "address": 0x2000, "address_page": 2, "write_ns": 0,
             "last_read_ns": 100, "evict_ns": None, "total_reads": 1, "lifetime_ns": 100}]


def _analysis():
    from tests.unit.reporter.test_metrics_csv import _minimal_analysis
    return _minimal_analysis()


def test_exporter_creates_run_dir(tmp_path):
    run_dir = tmp_path / "model" / "run_001"
    exporter = RunExporter(run_dir)
    result = exporter.export(_trace(), _region_map(), _lifecycle(), _analysis(),
                             run_id="run_001", figures=False)
    assert run_dir.is_dir()


def test_exporter_returns_export_result(tmp_path):
    exporter = RunExporter(tmp_path / "run")
    result = exporter.export(_trace(), _region_map(), _lifecycle(), _analysis(),
                             run_id="r1", figures=False)
    assert isinstance(result, ExportResult)


def test_exporter_writes_required_files(tmp_path):
    run_dir = tmp_path / "run"
    exporter = RunExporter(run_dir)
    result = exporter.export(_trace(), _region_map(), _lifecycle(), _analysis(),
                             run_id="r1", figures=False)
    required = {"trace.parquet", "region_map.parquet",
                "kv_block_lifecycle.parquet", "metrics.csv", "manifest.json"}
    assert required.issubset(set(result.file_names) | {f.name for f in run_dir.iterdir()})


def test_exporter_writes_metadata_json(tmp_path):
    run_dir = tmp_path / "run"
    exporter = RunExporter(run_dir)
    meta = {"run_id": "r1", "model": "llama-7b"}
    result = exporter.export(_trace(), _region_map(), _lifecycle(), _analysis(),
                             metadata=meta, run_id="r1", figures=False)
    meta_path = run_dir / "metadata.json"
    assert meta_path.exists()
    loaded = json.loads(meta_path.read_text())
    assert loaded["run_id"] == "r1"


def test_exporter_manifest_covers_written_files(tmp_path):
    run_dir = tmp_path / "run"
    exporter = RunExporter(run_dir)
    result = exporter.export(_trace(), _region_map(), _lifecycle(), _analysis(),
                             run_id="r1", figures=False)
    manifest = json.loads(result.manifest_path.read_text())
    manifest_names = {e["path"] for e in manifest["files"]}
    for f in result.written_files:
        assert f.name in manifest_names, f"manifest missing {f.name}"


def test_exporter_manifest_run_id_matches(tmp_path):
    run_dir = tmp_path / "run"
    exporter = RunExporter(run_dir)
    result = exporter.export(_trace(), _region_map(), _lifecycle(), _analysis(),
                             run_id="my_run_999", figures=False)
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["run_id"] == "my_run_999"


def test_exporter_all_parquet_readable(tmp_path):
    run_dir = tmp_path / "run"
    exporter = RunExporter(run_dir)
    exporter.export(_trace(), _region_map(), _lifecycle(), _analysis(),
                    run_id="r1", figures=False)
    for fname in ("trace.parquet", "region_map.parquet", "kv_block_lifecycle.parquet"):
        df = pd.read_parquet(run_dir / fname)
        assert len(df) >= 1

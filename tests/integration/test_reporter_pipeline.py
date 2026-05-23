"""
Integration tests for the reporter pipeline.

Uses the synthetic_trace_factory fixture → compute_all → RunExporter
and verifies that the output run dir is structurally complete.
"""

import hashlib
import json

import pandas as pd
import pytest

from mrm_trace.analyser import compute_all
from mrm_trace.labeller.labeller import TraceLabeller
from mrm_trace.reporter.run_exporter import RunExporter


@pytest.fixture
def run_dir_result(synthetic_trace_factory, tmp_path):
    """Full pipeline: synthetic trace → label → analyse → export."""
    trace = synthetic_trace_factory()
    labeller = TraceLabeller()
    _ = list(labeller.label(trace.to_dict("records")))
    region_map = labeller.region_map()
    kv_lifecycle = labeller.kv_lifecycle()

    analysis = compute_all(trace)

    run_dir = tmp_path / "results" / "llama-7b" / "run_001"
    exporter = RunExporter(run_dir)
    result = exporter.export(
        trace=trace,
        region_map=region_map,
        kv_lifecycle=kv_lifecycle,
        analysis=analysis,
        metadata={"run_id": "run_001", "model": "llama-7b"},
        run_id="run_001",
        figures=False,
    )
    return result, run_dir


class TestRunDirStructure:

    def test_run_dir_exists(self, run_dir_result):
        _, run_dir = run_dir_result
        assert run_dir.is_dir()

    def test_trace_parquet_present(self, run_dir_result):
        _, run_dir = run_dir_result
        assert (run_dir / "trace.parquet").exists()

    def test_region_map_parquet_present(self, run_dir_result):
        _, run_dir = run_dir_result
        assert (run_dir / "region_map.parquet").exists()

    def test_kv_lifecycle_parquet_present(self, run_dir_result):
        _, run_dir = run_dir_result
        assert (run_dir / "kv_block_lifecycle.parquet").exists()

    def test_metrics_csv_present(self, run_dir_result):
        _, run_dir = run_dir_result
        assert (run_dir / "metrics.csv").exists()

    def test_metadata_json_present(self, run_dir_result):
        _, run_dir = run_dir_result
        assert (run_dir / "metadata.json").exists()

    def test_manifest_json_present(self, run_dir_result):
        _, run_dir = run_dir_result
        assert (run_dir / "manifest.json").exists()


class TestManifestIntegrity:

    def test_manifest_valid_json(self, run_dir_result):
        _, run_dir = run_dir_result
        data = json.loads((run_dir / "manifest.json").read_text())
        assert "files" in data

    def test_manifest_checksums_match_files(self, run_dir_result):
        result, run_dir = run_dir_result
        manifest = json.loads(result.manifest_path.read_text())
        for entry in manifest["files"]:
            fpath = run_dir / entry["path"]
            if not fpath.exists():
                continue
            h = hashlib.sha256(fpath.read_bytes()).hexdigest()
            assert h == entry["sha256"], (
                f"Checksum mismatch for {entry['path']}: "
                f"manifest={entry['sha256'][:8]}, actual={h[:8]}"
            )

    def test_manifest_file_sizes_correct(self, run_dir_result):
        result, run_dir = run_dir_result
        manifest = json.loads(result.manifest_path.read_text())
        for entry in manifest["files"]:
            fpath = run_dir / entry["path"]
            if fpath.exists():
                assert fpath.stat().st_size == entry["size_bytes"]


class TestParquetReadability:

    def test_trace_parquet_has_expected_columns(self, run_dir_result):
        _, run_dir = run_dir_result
        df = pd.read_parquet(run_dir / "trace.parquet")
        for col in ("region_type", "region_id", "address", "op_type"):
            assert col in df.columns

    def test_region_map_parquet_has_expected_columns(self, run_dir_result):
        _, run_dir = run_dir_result
        df = pd.read_parquet(run_dir / "region_map.parquet")
        assert "region_id" in df.columns
        assert "region_type" in df.columns

    def test_kv_lifecycle_parquet_has_expected_columns(self, run_dir_result):
        _, run_dir = run_dir_result
        df = pd.read_parquet(run_dir / "kv_block_lifecycle.parquet")
        assert "block_id" in df.columns


class TestMetricsCSVContent:

    def test_metrics_csv_has_weight_row(self, run_dir_result):
        _, run_dir = run_dir_result
        df = pd.read_csv(run_dir / "metrics.csv")
        assert "weight" in df["region_type"].values

    def test_metrics_csv_suitability_column_present(self, run_dir_result):
        _, run_dir = run_dir_result
        df = pd.read_csv(run_dir / "metrics.csv")
        assert "suitability" in df.columns

    def test_metrics_csv_weight_classified_high_mrm(self, run_dir_result):
        _, run_dir = run_dir_result
        df = pd.read_csv(run_dir / "metrics.csv")
        w = df[df["region_type"] == "weight"].iloc[0]
        assert w["suitability"] == "high_mrm"

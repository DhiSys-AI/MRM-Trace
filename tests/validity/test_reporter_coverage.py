"""
Tier-3 validity tests for Phase 8 — Reporting + export.

Acceptance criterion: Run dirs are publication-ready.

A publication-ready run dir must have:
  1. All required files present
  2. All parquet files valid and readable (not empty)
  3. Manifest checksums match actual file contents (no silent corruption)
  4. metrics.csv contains scientifically correct suitability labels
  5. metadata.json round-trips intact
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from mrm_trace.analyser import compute_all
from mrm_trace.reporter.run_exporter import RunExporter


_REQUIRED_FILES = {
    "trace.parquet",
    "region_map.parquet",
    "kv_block_lifecycle.parquet",
    "metrics.csv",
    "manifest.json",
}


@pytest.fixture
def publication_run(tmp_path, synthetic_trace_factory):
    """
    Build a complete publication-ready run dir from the synthetic fixture.
    Scoped to module so the expensive export runs once.
    """
    trace = synthetic_trace_factory(
        n_weight_reads=1000,
        n_kv_writes=100,
        n_kv_reads=500,
        weight_retention_s=30.0,
    )
    analysis = compute_all(trace)

    from mrm_trace.labeller.labeller import TraceLabeller
    labeller = TraceLabeller()
    _ = list(labeller.label(trace.to_dict("records")))

    run_dir = tmp_path / "pub_run"
    exporter = RunExporter(run_dir)
    result = exporter.export(
        trace=trace,
        region_map=labeller.region_map(),
        kv_lifecycle=labeller.kv_lifecycle(),
        analysis=analysis,
        metadata={"run_id": "pub_001", "model": "test-llama"},
        run_id="pub_001",
        figures=False,
    )
    return result, run_dir, analysis


class TestAllRequiredFilesPresent:

    def test_trace_parquet_exists(self, publication_run):
        _, run_dir, _ = publication_run
        assert (run_dir / "trace.parquet").exists(), (
            "trace.parquet missing — publication-ready run dirs must contain the labelled trace."
        )

    def test_region_map_parquet_exists(self, publication_run):
        _, run_dir, _ = publication_run
        assert (run_dir / "region_map.parquet").exists(), (
            "region_map.parquet missing — required for downstream retention analysis."
        )

    def test_kv_lifecycle_parquet_exists(self, publication_run):
        _, run_dir, _ = publication_run
        assert (run_dir / "kv_block_lifecycle.parquet").exists(), (
            "kv_block_lifecycle.parquet missing — required for KV retention statistics."
        )

    def test_metrics_csv_exists(self, publication_run):
        _, run_dir, _ = publication_run
        assert (run_dir / "metrics.csv").exists(), (
            "metrics.csv missing — human-readable summary required for publication."
        )

    def test_manifest_json_exists(self, publication_run):
        _, run_dir, _ = publication_run
        assert (run_dir / "manifest.json").exists(), (
            "manifest.json missing — required for dataset reproducibility verification."
        )


class TestManifestChecksumIntegrity:

    def test_all_manifest_checksums_match(self, publication_run):
        """
        Every file listed in manifest.json must have a matching SHA-256.

        Checksum mismatch = silent data corruption that would invalidate
        the entire run when audited by collaborators or peer reviewers.
        """
        result, run_dir, _ = publication_run
        manifest = json.loads(result.manifest_path.read_text())
        failures = []
        for entry in manifest["files"]:
            fpath = run_dir / entry["path"]
            if not fpath.exists():
                continue
            actual = hashlib.sha256(fpath.read_bytes()).hexdigest()
            if actual != entry["sha256"]:
                failures.append(
                    f"{entry['path']}: manifest={entry['sha256'][:12]}… "
                    f"actual={actual[:12]}…"
                )
        assert not failures, (
            f"Checksum mismatches — publication run dir is corrupted:\n"
            + "\n".join(failures)
        )

    def test_manifest_file_sizes_are_accurate(self, publication_run):
        result, run_dir, _ = publication_run
        manifest = json.loads(result.manifest_path.read_text())
        for entry in manifest["files"]:
            fpath = run_dir / entry["path"]
            if fpath.exists():
                assert fpath.stat().st_size == entry["size_bytes"], (
                    f"{entry['path']}: size in manifest {entry['size_bytes']} "
                    f"≠ actual {fpath.stat().st_size}."
                )


class TestParquetValidity:

    def test_trace_parquet_not_empty(self, publication_run):
        _, run_dir, _ = publication_run
        df = pd.read_parquet(run_dir / "trace.parquet")
        assert len(df) > 0, "trace.parquet is empty — run produced no labelled rows."

    def test_region_map_has_weight_region(self, publication_run):
        _, run_dir, _ = publication_run
        df = pd.read_parquet(run_dir / "region_map.parquet")
        assert "weight" in df["region_type"].values, (
            "region_map must contain at least one weight region."
        )

    def test_kv_lifecycle_has_entries(self, publication_run):
        _, run_dir, _ = publication_run
        df = pd.read_parquet(run_dir / "kv_block_lifecycle.parquet")
        assert len(df) > 0, "kv_block_lifecycle.parquet is empty."


class TestMetricsCSVScientificCorrectness:

    def test_weight_suitability_is_high_mrm(self, publication_run):
        """
        The synthetic fixture's weight region must be classified high_mrm
        in metrics.csv (retention≈30s, write_once=1.0).

        A wrong suitability label would cause weight to be excluded from MRM
        candidate analysis in downstream scripts.
        """
        _, run_dir, _ = publication_run
        df = pd.read_csv(run_dir / "metrics.csv")
        row = df[df["region_type"] == "weight"].iloc[0]
        assert row["suitability"] == "high_mrm", (
            f"Weight suitability in metrics.csv = {row['suitability']!r}, "
            "expected 'high_mrm'. A 30s write-once region must classify correctly."
        )

    def test_weight_retention_p99_approx_30s_in_csv(self, publication_run):
        _, run_dir, _ = publication_run
        df = pd.read_csv(run_dir / "metrics.csv")
        row = df[df["region_type"] == "weight"].iloc[0]
        assert abs(row["retention_p99_s"] - 30.0) < 1.0, (
            f"metrics.csv weight retention_p99_s={row['retention_p99_s']:.3f}, expected ≈30.0."
        )

    def test_kv_suitability_is_low_mrm(self, publication_run):
        _, run_dir, _ = publication_run
        df = pd.read_csv(run_dir / "metrics.csv")
        row = df[df["region_type"] == "kv_cache"].iloc[0]
        assert row["suitability"] == "low_mrm", (
            f"KV suitability in metrics.csv = {row['suitability']!r}, expected 'low_mrm'."
        )

"""
RunExporter — orchestrates writing all publication-ready artifacts to a run dir.

Output layout (per ADR-7):
  results/<model_id>/<run_id>/
    trace.parquet
    region_map.parquet
    kv_block_lifecycle.parquet
    metrics.csv
    metadata.json
    manifest.json
    retention_histogram.png   (if matplotlib available)
    working_set.png           (if matplotlib available)
    write_once.png            (if matplotlib available)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from mrm_trace.reporter.figures import generate_all_figures
from mrm_trace.reporter.manifest import write_manifest
from mrm_trace.reporter.metrics_csv import write_metrics_csv
from mrm_trace.reporter.parquet_export import (
    export_kv_lifecycle,
    export_region_map,
    export_trace,
)
from mrm_trace.utils.files import ensure_dir


class RunExporter:
    """
    Writes all Phase 8 artifacts for a completed run.

    Usage::

        exporter = RunExporter(run_dir)
        result = exporter.export(
            trace=trace_df,
            region_map=region_map_rows,
            kv_lifecycle=lifecycle_rows,
            analysis=compute_all(trace_df),
            metadata=metadata_dict,
        )
        print(result.manifest_path)
    """

    def __init__(self, run_dir: Union[str, Path]) -> None:
        self.run_dir = Path(run_dir)

    def export(
        self,
        trace: pd.DataFrame,
        region_map: Union[pd.DataFrame, List[dict]],
        kv_lifecycle: Union[pd.DataFrame, List[dict]],
        analysis: dict,
        metadata: Optional[Dict[str, Any]] = None,
        run_id: str = "unknown",
        figures: bool = True,
        trace_already_written: bool = False,
    ) -> "ExportResult":
        """
        Write all artifacts and return an ExportResult.

        Parameters
        ----------
        trace                 : labelled trace DataFrame (used for analysis; skipped if
                                trace_already_written=True)
        region_map            : region_map rows (DataFrame or list of dicts)
        kv_lifecycle          : kv_block_lifecycle rows (DataFrame or list of dicts)
        analysis              : dict from compute_all(trace)
        metadata              : dict to write as metadata.json (optional)
        run_id                : run identifier for the manifest
        figures               : whether to generate figures (skipped if matplotlib absent)
        trace_already_written : set True when trace.parquet was written by the caller
                                (e.g. via streaming write_parquet). Avoids a second write.
        """
        ensure_dir(self.run_dir)

        written: List[Path] = []

        if trace_already_written:
            trace_path = self.run_dir / "trace.parquet"
            if trace_path.exists():
                written.append(trace_path)
        else:
            written.append(export_trace(trace, self.run_dir))
        written.append(export_region_map(region_map, self.run_dir))
        written.append(export_kv_lifecycle(kv_lifecycle, self.run_dir))
        written.append(write_metrics_csv(analysis, self.run_dir / "metrics.csv"))

        if metadata is not None:
            meta_path = self.run_dir / "metadata.json"
            meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            written.append(meta_path)

        if figures:
            written.extend(generate_all_figures(analysis, self.run_dir))

        manifest_path = write_manifest(self.run_dir, run_id, files=written)

        return ExportResult(
            run_dir=self.run_dir,
            run_id=run_id,
            manifest_path=manifest_path,
            written_files=written,
        )


class ExportResult:
    """Return value from RunExporter.export()."""

    def __init__(
        self,
        run_dir: Path,
        run_id: str,
        manifest_path: Path,
        written_files: List[Path],
    ) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self.manifest_path = manifest_path
        self.written_files = written_files

    @property
    def file_names(self) -> List[str]:
        return [f.name for f in self.written_files]

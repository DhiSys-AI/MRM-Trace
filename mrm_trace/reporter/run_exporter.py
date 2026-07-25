"""
RunExporter — orchestrates writing all publication-ready artifacts to a run dir.

Output layout (per ADR-7, updated for multi-machine):
  results/
    manifest.csv                        ← dataset-level, one row per run (all machines)
    <machine_id>/
      <model_id>/
        <run_id>/
          trace.parquet
          region_map.parquet
          kv_block_lifecycle.parquet
          metrics.csv                   ← per-run analysis (one row per region_type)
          metadata.json
          manifest.json
          retention_histogram.png       (if matplotlib available)
          working_set.png
          write_once.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from mrm_trace.reporter.dataset_manifest import append_manifest_row
from mrm_trace.reporter.figures import generate_all_figures
from mrm_trace.reporter.manifest import write_manifest
from mrm_trace.reporter.metrics_csv import write_metrics_csv
from mrm_trace.reporter.parquet_export import (
    export_kv_lifecycle,
    export_region_map,
    export_trace,
)
from mrm_trace.utils.files import ensure_dir


def _write_prompt_outputs(run_dir: Path, gen_results: list) -> Path:
    """Write prompt_outputs.json — one entry per prompt with text and token counts."""
    data = {
        "prompts": [
            {
                "index": i,
                "prompt": r.get("prompt", ""),
                "output": r.get("output_text", ""),
                "prompt_tokens": r.get("prompt_tokens"),
                "tokens_generated": r.get("output_tokens"),
                "duration_s": r.get("duration_s"),
            }
            for i, r in enumerate(gen_results)
        ]
    }
    path = Path(run_dir) / "prompt_outputs.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


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
        manifest_csv: Optional[Path] = None,
        prompt_outputs: Optional[List[dict]] = None,
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
        manifest_csv          : path to the dataset-level manifest.csv (results root).
                                When provided, one summary row is appended after export.
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

        if prompt_outputs is not None:
            written.append(_write_prompt_outputs(self.run_dir, prompt_outputs))

        manifest_path = write_manifest(self.run_dir, run_id, files=written)

        if manifest_csv is not None and metadata is not None:
            append_manifest_row(
                Path(manifest_csv), run_id, metadata, analysis,
                has_prompt_outputs=(prompt_outputs is not None),
            )

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

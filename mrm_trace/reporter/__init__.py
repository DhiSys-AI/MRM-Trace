"""Reporting and export subsystem for mrm-trace — Phase 8."""

from mrm_trace.reporter.figures import generate_all_figures
from mrm_trace.reporter.manifest import build_manifest, write_manifest
from mrm_trace.reporter.metrics_csv import build_metrics_table, write_metrics_csv
from mrm_trace.reporter.parquet_export import (
    export_kv_lifecycle,
    export_region_map,
    export_trace,
)
from mrm_trace.reporter.run_exporter import ExportResult, RunExporter

__all__ = [
    "build_manifest",
    "write_manifest",
    "build_metrics_table",
    "write_metrics_csv",
    "export_trace",
    "export_region_map",
    "export_kv_lifecycle",
    "generate_all_figures",
    "RunExporter",
    "ExportResult",
]

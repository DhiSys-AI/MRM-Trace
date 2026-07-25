"""
Dataset-level manifest CSV — one flat row per completed run.

Written to <results_root>/manifest.csv, which is above the
machine/model/run directory tree. This is the entry point for
Kaggle notebooks: load manifest.csv to filter runs, then drill
into individual parquet / metadata files as needed.

Columns (all nullable):
  run_id, machine_id, model_id, collector,
  context_length, batch_size, max_tokens_per_prompt, rep, run_validity,
  peak_temp_c, throttle_events, total_wall_s,
  avg_tok_per_s, total_output_tokens,
  retention_p99_s, write_once_ratio_mean, working_set_pages_total,
  cpu_model, cpu_freq_scaling_max_mhz, os, kernel, recorded_at

  max_tokens_per_prompt: the --tokens ceiling passed at run time (the experimental variable).
  total_output_tokens:   actual tokens generated across all prompts (the measurement).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd


def _df_agg(analysis: dict, key: str, col: str, agg: str) -> Optional[float]:
    df = analysis.get(key)
    if df is None or not hasattr(df, "empty") or df.empty or col not in df.columns:
        return None
    vals = df[col].dropna()
    if vals.empty:
        return None
    return round(float(vals.agg(agg)), 4)


def build_manifest_row(run_id: str, metadata: dict, analysis: dict) -> dict[str, Any]:
    """
    Build a single flat manifest row from a completed run's metadata + analysis.

    Extracts the fields most useful for cross-run filtering on Kaggle.
    All values are nullable — missing metadata/analysis fields become None.
    """
    gen = metadata.get("generation", {})
    hw = metadata.get("hardware", {})
    sw = metadata.get("software", {})
    thermal = metadata.get("thermal", {})
    timing = metadata.get("timing", {})
    model_info = metadata.get("model", {})
    collector_info = metadata.get("collector", {})

    retention_p99_s = _df_agg(analysis, "retention_summary", "retention_p99_s", "mean")
    write_once_ratio_mean = _df_agg(analysis, "write_once", "write_once_ratio", "mean")
    working_set_pages_total = _df_agg(analysis, "working_set_summary", "total_pages", "sum")
    if working_set_pages_total is not None:
        working_set_pages_total = int(working_set_pages_total)

    return {
        "run_id": run_id,
        "machine_id": metadata.get("machine_id", ""),
        "model_id": model_info.get("id") or model_info.get("hf_repo"),
        "collector": collector_info.get("type"),
        "context_length": gen.get("context_length"),
        "batch_size": gen.get("batch_size"),
        "max_tokens_per_prompt": gen.get("output_tokens"),
        "rep": gen.get("rep_index"),
        "run_validity": thermal.get("run_validity"),
        "peak_temp_c": thermal.get("peak_temp_c"),
        "throttle_events": thermal.get("throttle_events"),
        "total_wall_s": timing.get("total_wall_duration_s"),
        "avg_tok_per_s": gen.get("avg_tokens_per_second"),
        "total_output_tokens": gen.get("total_output_tokens"),
        "retention_p99_s": retention_p99_s,
        "write_once_ratio_mean": write_once_ratio_mean,
        "working_set_pages_total": working_set_pages_total,
        "cpu_model": hw.get("cpu_model"),
        "cpu_freq_scaling_max_mhz": hw.get("cpu_freq_scaling_max_mhz"),
        "os": sw.get("os"),
        "kernel": sw.get("kernel"),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


def append_manifest_row(manifest_path: Path, run_id: str, metadata: dict, analysis: dict) -> Path:
    """
    Append one row to the dataset manifest CSV.

    Creates the file with a header if it doesn't exist yet; appends
    without a header otherwise. Thread-safe for sequential runners
    (one run completes before the next starts).

    Returns the manifest path.
    """
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    row = build_manifest_row(run_id, metadata, analysis)
    df = pd.DataFrame([row])
    write_header = not manifest_path.exists()
    df.to_csv(manifest_path, mode="a", header=write_header, index=False)
    return manifest_path

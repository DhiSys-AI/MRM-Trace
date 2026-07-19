#!/usr/bin/env python3
"""
build_datasets.py - Package MRM-Trace results/ into a Kaggle dataset
(derived artifacts only) and a Zenodo archive (raw perf data with checksums).

Usage:
    python3 build_datasets.py \
        --results-dir /home/gaurav/MRM-Trace/results \
        --out-dir /home/gaurav/MRM-Trace/publish \
        --sample-frac 0.10

Requires: pandas, pyarrow  (pip install pandas pyarrow)
"""

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SKIP_DIRS = {"logs"}
RAW_DIRNAME = "raw"
RUN_FILES = ["metadata.json", "metrics.csv", "manifest.json",
             "generation_results.json", "trace.parquet"]


def discover_runs(results_dir: Path):
    """Yield (model_name, run_id, run_path) for every valid run directory."""
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name in SKIP_DIRS:
            continue
        for run_dir in sorted(model_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            if (run_dir / "metrics.csv").exists():
                yield model_dir.name, run_dir.name, run_dir


def sha256sum(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_index_row(model: str, run_id: str, run_key: str, run_path: Path) -> dict:
    """Build one run_index.csv row, including thermal/timing fields if metadata.json has them."""
    row = {"model": model, "run_id": run_id, "run_key": run_key,
           "peak_temp_c": None, "throttled": None, "run_validity": None,
           "total_wall_duration_s": None}
    meta_src = run_path / "metadata.json"
    if meta_src.exists():
        try:
            meta = json.loads(meta_src.read_text())
        except (OSError, ValueError):
            return row
        thermal = meta.get("thermal", {})
        row["peak_temp_c"] = thermal.get("peak_temp_c")
        row["throttled"] = thermal.get("throttled")
        row["run_validity"] = thermal.get("run_validity")
        row["total_wall_duration_s"] = meta.get("timing", {}).get("total_wall_duration_s")
    return row


def build_kaggle_dataset(results_dir: Path, out_dir: Path, sample_frac: float, metadata_only: bool = False):
    kdir = out_dir / "kaggle_dataset"
    (kdir / "region_map").mkdir(parents=True, exist_ok=True)
    (kdir / "kv_lifecycle").mkdir(parents=True, exist_ok=True)
    (kdir / "trace_samples").mkdir(parents=True, exist_ok=True)
    (kdir / "metadata").mkdir(parents=True, exist_ok=True)
    (kdir / "manifests").mkdir(parents=True, exist_ok=True)
    (kdir / "generation_results").mkdir(parents=True, exist_ok=True)

    all_metrics = []
    run_index = []

    for model, run_id, run_path in discover_runs(results_dir):
        run_key = f"{model}__{run_id}"

        # 1. metrics.csv -> tag with model/run_id and append to master table
        if not metadata_only:
            try:
                df = pd.read_csv(run_path / "metrics.csv")
            except Exception as e:
                print(f"[WARN] failed to read metrics.csv for {run_key}: {e}", file=sys.stderr)
                continue
            df.insert(0, "run_id", run_id)
            df.insert(0, "model", model)
            all_metrics.append(df)

        # 2. metadata.json -> metadata/<model>__<run_id>.json
        meta_src = run_path / "metadata.json"
        if meta_src.exists():
            shutil.copy2(meta_src, kdir / "metadata" / f"{run_key}.json")

        # 3. manifest.json -> manifests/<model>__<run_id>.json
        manifest_src = run_path / "manifest.json"
        if manifest_src.exists():
            shutil.copy2(manifest_src, kdir / "manifests" / f"{run_key}.json")

        # 4. generation_results.json -> generation_results/<model>__<run_id>.json
        gen_src = run_path / "generation_results.json"
        if gen_src.exists():
            shutil.copy2(gen_src, kdir / "generation_results" / f"{run_key}.json")

        if not metadata_only:
            # 5. optional derived parquet files (present in some pipeline versions)
            for fname, subdir in [("region_map.parquet", "region_map"),
                                  ("kv_block_lifecycle.parquet", "kv_lifecycle"),
                                  ("kv_lifecycle.parquet", "kv_lifecycle")]:
                src = run_path / fname
                if src.exists():
                    dst = kdir / subdir / f"{run_key}.parquet"
                    try:
                        tmp = pd.read_parquet(src)
                        tmp.to_parquet(dst, compression="zstd", index=False)
                    except Exception as e:
                        print(f"[WARN] {run_key}: could not recompress {fname}: {e}", file=sys.stderr)

            # 6. trace.parquet -> 10% random sample, zstd compressed
            trace_src = run_path / "trace.parquet"
            if trace_src.exists():
                try:
                    trace_df = pd.read_parquet(trace_src)
                    sample = trace_df.sample(frac=sample_frac, random_state=42) if len(trace_df) else trace_df
                    sample.to_parquet(kdir / "trace_samples" / f"{run_key}.parquet",
                                       compression="zstd", index=False)
                except Exception as e:
                    print(f"[WARN] {run_key}: could not sample trace.parquet: {e}", file=sys.stderr)

        run_index.append(_run_index_row(model, run_id, run_key, run_path))

    if metadata_only:
        print(f"[OK] metadata-only refresh: {len(run_index)} runs (metrics.csv / parquet / zenodo untouched)")
    elif all_metrics:
        combined = pd.concat(all_metrics, ignore_index=True)
        combined.to_csv(kdir / "metrics.csv", index=False)
        print(f"[OK] combined metrics.csv: {len(combined)} rows, {combined['model'].nunique()} models, "
              f"{combined[['model', 'run_id']].drop_duplicates().shape[0]} runs")
    else:
        print("[ERROR] no run metrics found - check --results-dir path", file=sys.stderr)

    pd.DataFrame(run_index).to_csv(kdir / "run_index.csv", index=False)

    # carry over machine-level provenance files if present at results root
    for extra in ["machine_validation.json", ".experiment_state.json"]:
        src = results_dir / extra
        if src.exists():
            shutil.copy2(src, kdir / extra.lstrip("."))

    write_datacard(kdir, run_index)
    return kdir


def write_datacard(kdir: Path, run_index):
    models = sorted(set(r["model"] for r in run_index))
    n_runs = len(run_index)
    n_contaminated = sum(1 for r in run_index if r.get("run_validity") == "contaminated")
    n_marginal = sum(1 for r in run_index if r.get("run_validity") == "marginal")
    n_clean = sum(1 for r in run_index if r.get("run_validity") == "clean")
    n_unknown = n_runs - n_contaminated - n_marginal - n_clean
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    content = f"""# MRM-Trace Dataset - Data Card

Generated: {today}

## Summary
LLM inference memory-access traces collected with `perf mem` (CPU path) and
PyTorch/transformers (GPU path), for {n_runs} runs across {len(models)} models:
{', '.join(models)}.

## Hardware
- CPU: Intel Core i9-11980HK (Tiger Lake, 8c/16t), PMU version 5 (full PEBS, load-latency sampling)
- RAM: 32 GB DDR4
- GPU: NVIDIA RTX 3080 Laptop, 16 GB VRAM
- OS / kernel: Ubuntu 22.04.4 LTS (fill in `uname -r` output for the exact kernel version used)

## Collector
- Primary collector: `perf mem record` with event `cpu/mem-loads,ldlat=20/P`
- Perf version: (fill in `perf --version` output)
- GPU cross-validation path: PyTorch + transformers, RAPL/nvidia-smi power sampling where applicable

## Thermal caveat — READ BEFORE USING TIMING/RETENTION VALUES
This collection ran on a laptop CPU without sustained-load cooling headroom.
Per-run peak package temperature and thermal-throttle status are recorded in
`run_index.csv` (`peak_temp_c`, `run_validity`) and in each run's
`metadata/<model>__<run_id>.json` (`thermal` section). Across the runs in this
release: **{n_clean} clean, {n_marginal} marginal, {n_contaminated} contaminated,
{n_unknown} unknown** by thermal-throttle status (see `run_validity`).
Retention/latency values from `contaminated` runs were collected while the CPU
was sustained at or near its thermal limit (peak temps of ~95-100°C) — treat
absolute timing values from those runs as lower-confidence; the access-pattern
structure (write-once ratio, read-frequency ordering) is less affected by
throttling than raw latency is. Runs whose `thermal` block has
`"source": "backfilled_from_experiment_state"` were classified from a
peak-temperature proxy (no per-run throttle-event count was captured for that
run) rather than the full throttle-event-based classifier.

## Proxy caveat
Retention durations are observed lifetimes under a batch schedule on this
CPU/GPU's cache hierarchy. The logical access-pattern structure (write-once
ratio, read-frequency ordering by region type) transfers across substrates
and constitutes first-order evidence for MRM silicon design questions.
Absolute timing values are machine-specific.

## Files
- `metrics.csv` - one row per run, tagged with `model` and `run_id` columns
- `run_index.csv` - lookup table of every included model/run_id pair, with
  `peak_temp_c`, `throttled`, `run_validity`, `total_wall_duration_s` joined in
  from each run's metadata.json (see Thermal caveat above)
- `metadata/<model>__<run_id>.json` - per-run metadata (config, timestamps,
  hardware state, and a `thermal` section: `peak_temp_c`, `throttled`,
  `throttle_events` (if measured directly), `run_validity`) and a `timing`
  section (`total_wall_duration_s`)
- `manifests/<model>__<run_id>.json` - per-run manifest (file listing / provenance)
- `generation_results/<model>__<run_id>.json` - generated text / token-level output for the run
- `trace_samples/<model>__<run_id>.parquet` - 10% random sample of the full memory access trace
- `region_map/<model>__<run_id>.parquet` - memory region classification (if available for the run)
- `kv_lifecycle/<model>__<run_id>.parquet` - KV-cache block lifecycle records (if available for the run)
- `experiment_state.json` - raw orchestrator run log (peak temps, durations, pass/fail) for every run

## Schema (fill in exact column names/types before publishing)
### metrics.csv
- model (str), run_id (str), retention (float), write_once (float), read_freq (float), ... (complete from actual columns)

### run_index.csv
- model (str), run_id (str), run_key (str), peak_temp_c (float), throttled (bool),
  run_validity (str: clean/marginal/contaminated/unknown), total_wall_duration_s (float)

### trace_samples/*.parquet
- (list columns from `trace.parquet`, e.g. timestamp, address, region_id, access_type, latency_cycles, ...)

## Citation
Legtchenko, S. et al. (2025). "Managed-Retention Memory."
"""
    (kdir / "datacard.md").write_text(content)
    print(f"[OK] wrote {kdir / 'datacard.md'}")


def build_zenodo_archive(results_dir: Path, out_dir: Path):
    zdir = out_dir / "zenodo_archive"
    zdir.mkdir(parents=True, exist_ok=True)
    checksums = {}

    for model, run_id, run_path in discover_runs(results_dir):
        raw_dir = run_path / RAW_DIRNAME
        if not raw_dir.exists():
            continue
        run_key = f"{model}__{run_id}"
        tar_path = zdir / f"{run_key}_raw.tar.xz"
        print(f"[..] archiving {run_key} raw/ -> {tar_path.name}")
        with tarfile.open(tar_path, "w:xz") as tar:
            tar.add(raw_dir, arcname=f"{run_key}/raw")
        checksums[tar_path.name] = {
            "sha256": sha256sum(tar_path),
            "size_bytes": tar_path.stat().st_size,
            "model": model,
            "run_id": run_id,
        }

    (zdir / "checksums.json").write_text(json.dumps(checksums, indent=2))

    readme = f"""# MRM-Trace Raw Data Archive (Zenodo)

Raw `perf.data` / `perf_script.txt` files for every run, tarred with xz
compression, one archive per run: `<model>__<run_id>_raw.tar.xz`.

These are the full-fidelity raw captures behind the derived Kaggle dataset.
See `checksums.json` for SHA-256 integrity hashes and per-file provenance.

## Thermal caveat — READ BEFORE USING TIMING/RETENTION VALUES
This collection ran on a laptop CPU without sustained-load cooling headroom.
Per-run peak package temperature and thermal-throttle status are NOT stored
in this archive (it contains only `raw/perf.data` and `raw/perf_script.txt`)
— they live in the companion Kaggle dataset's `run_index.csv` and
`metadata/<model>__<run_id>.json` (`thermal` section). Across the runs
released alongside this archive, the overwhelming majority were classified
`contaminated` by sustained thermal throttling (peak temps of ~95-100°C).
If you parse this raw data independently of the Kaggle release, cross-check
`run_validity` there before treating absolute timing/retention values as
clean; the access-pattern structure (write-once ratio, read-frequency
ordering) is less affected by throttling than raw latency is.

Upload this folder to Zenodo after the paper is accepted or a pre-print is
live, and cite the resulting DOI in the paper and in the Kaggle datacard.
"""
    (zdir / "README.md").write_text(readme)
    print(f"[OK] zenodo archive built: {zdir} ({len(checksums)} run archives)")
    return zdir


def main():
    ap = argparse.ArgumentParser(description="Package MRM-Trace results into Kaggle + Zenodo datasets")
    ap.add_argument("--results-dir", default="/home/gaurav/MRM-Trace/results", type=Path)
    ap.add_argument("--out-dir", default="/home/gaurav/MRM-Trace/publish", type=Path)
    ap.add_argument("--sample-frac", default=0.10, type=float)
    ap.add_argument("--skip-zenodo", action="store_true",
                     help="Skip raw archive step (e.g. to iterate on Kaggle packaging first)")
    ap.add_argument("--skip-kaggle", action="store_true")
    ap.add_argument("--metadata-only", action="store_true",
                     help="Only refresh metadata/manifests/generation_results + run_index.csv + "
                          "datacard.md. Skips metrics.csv, parquet resampling, and implies "
                          "--skip-zenodo. Cheap way to pick up metadata.json changes (e.g. thermal "
                          "backfill) without recomputing the whole dataset.")
    args = ap.parse_args()

    if not args.results_dir.exists():
        sys.exit(f"results dir not found: {args.results_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.metadata_only:
        args.skip_zenodo = True

    if not args.skip_kaggle:
        build_kaggle_dataset(args.results_dir, args.out_dir, args.sample_frac,
                              metadata_only=args.metadata_only)
    if not args.skip_zenodo:
        build_zenodo_archive(args.results_dir, args.out_dir)

    print("Done. Next: kaggle datasets create -p", args.out_dir / "kaggle_dataset",
          "--dir-mode zip   (after adding dataset-metadata.json)")


if __name__ == "__main__":
    main()

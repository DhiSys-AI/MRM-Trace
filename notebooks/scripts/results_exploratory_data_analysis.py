#!/usr/bin/env python3
"""
Exploratory analysis of a completed MRM-trace dataset.

Answers the questions the experiment was built for, and states plainly what the
data does and does not support:

  1. Inventory        — what was collected, and which runs can answer what
  2. Region contrast  — retention, write-once, reads-per-write, working set
  3. Model scaling    — how each metric moves with parameter count
  4. Context scaling  — how each moves with n_ctx (KV grows linearly with it)
  5. KV lifecycle     — write-to-last-read per KV block
  6. Memory hierarchy — cache-level mix per region (needs --with-traces)
  7. Caveats          — derived from the data, not asserted

Run results_validation.py first.  This script reports what it finds; it does
not decide whether the dataset is sound.

    python notebooks/scripts/results_exploratory_data_analysis.py
    python notebooks/scripts/results_exploratory_data_analysis.py --with-traces
    python notebooks/scripts/results_exploratory_data_analysis.py --out /tmp/eda
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from typing import Optional

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

# rep 0 is a warmup (experiment_runner.REPS); it is excluded from aggregates.
_WARMUP_REP = 0

# `unknown` is addresses in no known mapping.  Workplan Section 12.2 excludes it
# from region-level comparisons, so it is reported separately rather than mixed
# into the contrast table.
_COMPARABLE_REGIONS = ("weight", "kv_cache", "activation", "runtime")

# Regions whose addresses come from a recorded range rather than from a symbol.
# Only these support a cross-model claim; see
# context-files/04 - Region Attribution Validity.md.
_ADDRESS_GROUNDED = ("weight", "kv_cache")

# Parameter counts for the existing dataset.  demo_perf_mem.py recorded
# params_billions=1.1 for every model before that was fixed, so metadata cannot
# be trusted for it; a value is recovered from the model id instead.
_PARAMS_BY_MODEL_ID = (
    ("tinyllama", 1.1), ("0.5b", 0.5), ("1.5b", 1.5), ("1b", 1.0),
    ("3.5-mini", 3.8), ("3.8b", 3.8), ("3b", 3.0), ("7b", 7.0), ("8b", 8.0),
)

# The GPU path writes pytorch_regions.json; the CPU path does not.  Both record
# collector="perf_mem", so the collector field cannot distinguish them.
# One experimental cell.  The occupancy sweep varies output tokens at a fixed
# context, so a key of (model, ctx) alone collapses all three levels into one
# row and the sweep's independent variable disappears from the tables.
_CELL = ["model_id", "ctx", "output_tokens"]

_GPU_MARKER = "pytorch_regions.json"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


# ── loading ───────────────────────────────────────────────────────────────────

def params_billions(model_id: str, recorded) -> float:
    """
    Parameter count for a model, from its id.

    The recorded value is used only when it disagrees with nothing: every run
    in the first sweep carried params_billions=1.1 from a hardcoded constant, so
    an id-derived value is preferred and 0.0 means unknown.
    """
    lowered = model_id.lower()
    for needle, value in _PARAMS_BY_MODEL_ID:
        if needle in lowered:
            return value
    try:
        return float(recorded) if recorded else 0.0
    except (TypeError, ValueError):
        return 0.0


def load_dataset(results_dir: pathlib.Path) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """
    Returns (runs, metrics).

    runs:    one row per exported run, with provenance and validity.
    metrics: one row per (run, region_type), joined to the run's attributes.
    """
    run_rows, metric_frames = [], []
    for meta_path in sorted(results_dir.glob("*/*/*/metadata.json")):
        d = meta_path.parent
        metrics_path = d / "metrics.csv"
        if not metrics_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        gen, coll, model = meta["generation"], meta["collector"], meta["model"]
        run = {
            "run_dir": str(d),
            "run_id": meta.get("run_id", d.name),
            "model_id": model["id"],
            "params_b": params_billions(model["id"], model.get("params_billions")),
            "params_b_recorded": model.get("params_billions"),
            "path": "gpu" if (d / _GPU_MARKER).exists() else "cpu",
            "quantisation": model.get("quantisation"),
            "collector": coll["type"],
            "ctx": gen["context_length"],
            "rep": gen.get("rep_index"),
            "output_tokens": gen.get("output_tokens"),
            "tok_per_s": gen.get("avg_tokens_per_second"),
            "samples": coll.get("samples_collected"),
            "sample_period": coll.get("sample_period"),
            "granularity": coll.get("region_granularity_bytes"),
            "validity": meta.get("thermal", {}).get("run_validity"),
            "peak_temp_c": meta.get("thermal", {}).get("peak_temp_c"),
            "relabelled": "relabelled" in meta,
        }
        run_rows.append(run)
        m = pd.read_csv(metrics_path)
        for key in ("run_id", "model_id", "params_b", "collector", "path", "ctx",
                    "rep", "validity", "granularity"):
            m[key] = run[key]
        metric_frames.append(m)

    runs = pd.DataFrame(run_rows)
    metrics = (pd.concat(metric_frames, ignore_index=True)
               if metric_frames else pd.DataFrame())
    return runs, metrics


def analysis_set(runs: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Metrics from the runs that may enter an aggregate: data reps only (rep 0 is
    a warmup) and not contaminated.
    """
    keep = metrics[(metrics["rep"] != _WARMUP_REP)
                   & (metrics["validity"].isin(["clean", "marginal"]))]
    return keep.copy()


# ── sections ──────────────────────────────────────────────────────────────────

def section_inventory(runs: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    print_header("1. Inventory")
    by_model = (runs.groupby(["model_id", "path"])
                .agg(runs=("run_id", "count"),
                     contexts=("ctx", lambda s: sorted(s.unique())),
                     reps=("rep", lambda s: sorted(s.unique())),
                     clean=("validity", lambda s: (s == "clean").sum()),
                     samples_mean=("samples", "mean"))
                .reset_index())
    print(by_model.to_string(index=False))

    # Which regions each model actually produced decides what it can answer.
    coverage = (metrics.groupby(["model_id", "path"])["region_type"]
                .apply(lambda s: sorted(set(s))).reset_index(name="regions"))
    print()
    print("  regions produced per model:")
    for row in coverage.itertuples():
        missing = [r for r in _COMPARABLE_REGIONS if r not in row.regions]
        flag = f"   MISSING: {', '.join(missing)}" if missing else ""
        print(f"    {row.model_id:<26} {', '.join(row.regions)}{flag}")
    return by_model


def section_region_contrast(data: pd.DataFrame, out_dir: pathlib.Path) -> pd.DataFrame:
    """
    The headline comparison: what each region type looks like to an MRM
    designer.  Aggregated over every run in the analysis set, with the spread
    across runs reported so instability is visible rather than averaged away.
    """
    print_header("2. Region contrast (analysis set, all models and contexts)")
    cols = {
        "retention_p50_s": "retention p50 (s)",
        "retention_p99_s": "retention p99 (s)",
        "write_once_ratio_mean": "write-once ratio",
        "reads_per_write_mean": "reads per write",
        "working_set_mb_total": "working set (MB)",
    }
    present = [c for c in cols if c in data.columns]
    rows = []
    for region, g in data[data["region_type"].isin(_COMPARABLE_REGIONS)].groupby(
            "region_type"):
        row = {"region_type": region, "runs": g["run_id"].nunique(),
               "models": g["model_id"].nunique()}
        for col in present:
            vals = g[col].replace([np.inf, -np.inf], np.nan).dropna()
            row[cols[col]] = vals.mean() if len(vals) else np.nan
            row[f"{cols[col]} sd"] = vals.std() if len(vals) > 1 else np.nan
        # reads per write is often unbounded for a read-only region; say so
        # rather than dropping it silently.
        if "reads_per_write_mean" in g:
            row["unbounded r/w runs"] = int(np.isinf(g["reads_per_write_mean"]).sum())
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("region_type")
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print()
    print("  grounded in recorded address ranges: " + ", ".join(_ADDRESS_GROUNDED))
    print("  inferred from symbols (arena refinement only): activation")
    _write_csv(table, out_dir / "region_contrast.csv")
    return table


def section_model_scaling(data: pd.DataFrame, out_dir: pathlib.Path) -> pd.DataFrame:
    print_header("3. Model scaling (CPU path, per region type)")
    cpu = data[data["path"] == "cpu"]
    piv = (cpu.groupby(["region_type", "model_id", "params_b"])
           .agg(retention_p50_s=("retention_p50_s", "mean"),
                retention_p99_s=("retention_p99_s", "mean"),
                write_once=("write_once_ratio_mean", "mean"),
                working_set_mb=("working_set_mb_total", "mean"),
                runs=("run_id", "nunique"))
           .reset_index().sort_values(["region_type", "params_b"]))
    for region in _COMPARABLE_REGIONS:
        sub = piv[piv["region_type"] == region]
        if sub.empty:
            continue
        print(f"  {region}")
        print(sub.drop(columns="region_type").to_string(
            index=False, float_format=lambda v: f"{v:,.3f}"))
        if sub["model_id"].nunique() < 3:
            print(f"    only {sub['model_id'].nunique()} model(s) — not enough "
                  "for a scaling claim")
        print()
    _write_csv(piv, out_dir / "model_scaling.csv")
    return piv


def section_context_scaling(data: pd.DataFrame, out_dir: pathlib.Path) -> pd.DataFrame:
    """
    The lever that moves the KV cache, whichever one this dataset pulled.

    Context length sets how much KV cache is *allocated*; generated tokens set
    how much is *written*.  The context sweep varies the first at a fixed
    second, so occupancy falls as context grows and the cache is never filled.
    The occupancy sweep varies the second at a fixed first.  Grouping on `ctx`
    unconditionally made this table a single row on the occupancy sweep --
    the independent variable averaged away.
    """
    cpu = data[data["path"] == "cpu"]
    axis = "ctx" if cpu["ctx"].nunique() > 1 else "output_tokens"
    title = ("4. Context scaling (CPU path, per region type)" if axis == "ctx"
             else "4. Occupancy scaling (CPU path, per region type)")
    print_header(title)
    if axis == "output_tokens":
        print("  context fixed; generated tokens are the independent variable, "
              "so the KV cache is progressively filled")
        print()
    piv = (cpu.groupby(["region_type", axis])
           .agg(retention_p50_s=("retention_p50_s", "mean"),
                retention_p99_s=("retention_p99_s", "mean"),
                write_once=("write_once_ratio_mean", "mean"),
                working_set_mb=("working_set_mb_total", "mean"),
                n_regions=("n_regions", "mean"),
                runs=("run_id", "nunique"))
           .reset_index().sort_values(["region_type", axis]))
    for region in _COMPARABLE_REGIONS:
        sub = piv[piv["region_type"] == region]
        if sub.empty:
            continue
        print(f"  {region}")
        print(sub.drop(columns="region_type").to_string(
            index=False, float_format=lambda v: f"{v:,.3f}"))
        if region == "kv_cache" and len(sub) > 1 and axis == "output_tokens":
            lo, hi = sub.iloc[0], sub.iloc[-1]
            if lo["working_set_mb"] > 0:
                growth = hi["working_set_mb"] / lo["working_set_mb"]
                span = hi[axis] / lo[axis]
                print(f"    KV working set grows {growth:.1f}x as written "
                      f"positions grow {span:.1f}x — here occupancy *is* the "
                      f"variable, so this is a property of the cache filling "
                      f"rather than of the allocation.")
        elif region == "kv_cache" and len(sub) > 1:
            lo, hi = sub.iloc[0], sub.iloc[-1]
            if lo["working_set_mb"] > 0:
                growth = hi["working_set_mb"] / lo["working_set_mb"]
                span = hi["ctx"] / lo["ctx"]
                # Stating the growth without the confound invites reading it as
                # a property of KV caching.  It is not: output tokens are fixed
                # across the context sweep, so the number of KV entries written
                # is the same in every run and only the *allocated* buffer
                # scales.  Occupancy therefore falls from ~26% to ~1.6%, and a
                # flat touched footprint is the expected consequence of the
                # design, not a finding about KV behaviour.
                print(f"    KV working set changes {growth:.1f}x over a "
                      f"{span:.0f}x context increase — but output tokens are "
                      f"fixed, so the same ~535 KV positions are written in "
                      f"every run and occupancy falls from ~26% to ~1.6%.")
                print(f"    This sweep varies KV *allocation*, not KV "
                      f"*occupancy*; it cannot support a claim about how KV "
                      f"memory behaves as the cache fills.")
        print()
    _write_csv(piv, out_dir / "context_scaling.csv")
    return piv


def section_kv_lifecycle(runs: pd.DataFrame, out_dir: pathlib.Path) -> pd.DataFrame:
    """
    Per-KV-block lifetimes, straight from kv_block_lifecycle.parquet (1.9 MB
    across the dataset, so all of it is loaded).
    """
    print_header("5. KV block lifecycle")
    frames = []
    for row in runs.itertuples():
        path = pathlib.Path(row.run_dir) / "kv_block_lifecycle.parquet"
        if not path.exists():
            continue
        lc = pd.read_parquet(path)
        if lc.empty:
            continue
        lc["model_id"], lc["ctx"], lc["rep"] = row.model_id, row.ctx, row.rep
        lc["output_tokens"] = row.output_tokens
        frames.append(lc)
    if not frames:
        print("  no KV lifecycle rows in the dataset")
        return pd.DataFrame()
    lc = pd.concat(frames, ignore_index=True)
    if "lifetime_ns" in lc:
        lc["lifetime_s"] = lc["lifetime_ns"] / 1e9

    # A block's lifetime is defined only when its write was sampled.  Stores run
    # on a much sparser period than loads, so for most blocks no store was ever
    # captured and lifetime_ns is null -- an absence, not a zero.  Averaging
    # over those would report the monotonic clock as a duration.
    if "write_observed" in lc.columns:
        timed = lc[lc["write_observed"].fillna(False).astype(bool)]
    elif "lifetime_s" in lc.columns:
        timed = lc[lc["lifetime_s"].notna()]
    else:
        timed = lc

    agg_cols = {c: ["count", "mean", "median"] for c in
                ("lifetime_s", "total_reads") if c in timed.columns}
    if agg_cols and not timed.empty:
        summary = timed.groupby(_CELL).agg(agg_cols).reset_index()
        print(summary.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
        print()
        print("  lifetime columns cover only blocks whose write was sampled; "
              "total_reads below covers every block")
    else:
        print("  no KV block in the dataset had its write sampled, so no block "
              "lifetime is measurable; read counts are reported below")

    print()
    print(f"  {len(lc):,} KV blocks across {lc['model_id'].nunique()} model(s)")
    n_timed = len(timed)
    print(f"  blocks with a sampled write (lifetime measurable): {n_timed:,} "
          f"({n_timed / len(lc):.1%})")
    if n_timed / max(len(lc), 1) < 0.5:
        print("    -> block lifetime rests on this minority; it is not a "
              "dataset-wide measurement. Lower the store sampling period to "
              "raise it.")
    if "total_reads" in lc:
        reads = (lc.groupby(_CELL)["total_reads"]
                 .agg(["count", "mean", "median"]).reset_index())
        print()
        print("  read counts over all blocks:")
        print(reads.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
        never = (lc["total_reads"] == 0).sum()
        print()
        print(f"  blocks with no sampled read: {never:,} "
              f"({never / len(lc):.1%})")
    _write_csv(lc, out_dir / "kv_lifecycle_blocks.csv")
    return lc


def section_memory_hierarchy(runs: pd.DataFrame, out_dir: pathlib.Path,
                             limit: int) -> pd.DataFrame:
    """
    Cache-level mix per region type.  Needs the traces (2.3 GB across the
    dataset), so it is opt-in and reads only the two columns it uses.
    """
    print_header("6. Memory hierarchy (cache level by region)")
    frames = []
    for row in list(runs.itertuples())[:limit]:
        path = pathlib.Path(row.run_dir) / "trace.parquet"
        if not path.exists():
            continue
        t = pd.read_parquet(path, columns=["region_type", "cache_level", "op_type"])
        counts = (t.groupby(["region_type", "op_type", "cache_level"])
                  .size().reset_index(name="n"))
        counts["model_id"], counts["ctx"] = row.model_id, row.ctx
        frames.append(counts)
        print(f"  read {row.run_id}")
    if not frames:
        print("  no traces read")
        return pd.DataFrame()
    allc = pd.concat(frames, ignore_index=True)
    piv = (allc.groupby(["region_type", "cache_level"])["n"].sum()
           .groupby(level=0).apply(lambda s: s / s.sum()).unstack(fill_value=0))
    print()
    print((piv * 100).round(1).to_string())
    _write_csv(allc, out_dir / "cache_levels.csv")
    return allc


def section_caveats(runs: pd.DataFrame, metrics: pd.DataFrame,
                    data: pd.DataFrame,
                    kv_blocks: Optional[pd.DataFrame] = None) -> None:
    """
    Caveats derived from the data, so the report cannot silently outrun what
    was measured.
    """
    print_header("7. Caveats for the data card")
    notes = []

    cpu_models = set(metrics[metrics["path"] == "cpu"]["model_id"])
    for region in _ADDRESS_GROUNDED:
        have = set(metrics[metrics["region_type"] == region]["model_id"])
        missing = sorted(cpu_models - have)
        if missing:
            notes.append(f"`{region}` is absent for {len(missing)} CPU model(s): "
                         f"{', '.join(missing)}. Cross-model claims about it rest "
                         f"on {len(have & cpu_models)} model(s).")

    gpu = metrics[metrics["path"] == "gpu"]
    if not gpu.empty:
        notes.append(f"{gpu['run_id'].nunique()} run(s) on the GPU path carry only "
                     "host-side regions: perf mem samples CPU accesses, and those "
                     "models' weights and KV live in VRAM, so weight and kv_cache "
                     "are unobtainable there by construction.")

    if "reads_per_write_mean" in data.columns:
        unb = data[np.isinf(data["reads_per_write_mean"])]["region_type"].unique()
        if len(unb):
            notes.append(f"reads-per-write is unbounded for {', '.join(sorted(unb))}: "
                         "those regions received no store samples, which for a "
                         "read-only mapping is the limiting case of write-once "
                         "rather than missing data.")

    # Loads and stores are sampled on independent periods, the store period
    # being far sparser.  Any metric that divides by an observed write count --
    # write-once ratio, reads-per-write -- is therefore supported only where
    # stores were actually captured.  The KV lifecycle file measures that
    # support directly, so the caveat is derived rather than assumed.
    if kv_blocks is not None and len(kv_blocks) and "write_observed" in kv_blocks:
        frac = kv_blocks["write_observed"].fillna(False).astype(bool).mean()
        if frac < 0.5:
            notes.append(
                f"only {frac:.1%} of {len(kv_blocks):,} KV blocks had a store "
                "sampled, because stores are sampled on a much sparser period "
                "than loads. `write_once_ratio` and `reads_per_write` for "
                "kv_cache are computed over that minority, so a value of 1.000 "
                "reflects at-most-one sampled store per address rather than "
                "demonstrated write-once behaviour. Report them as unsupported "
                "for kv_cache, or re-collect with a lower store period.")

    granules = sorted(runs["granularity"].dropna().unique())
    notes.append("retention is reported at a granule of "
                 + ", ".join(f"{int(g) // 1024} KB" for g in granules)
                 + "; it is last-minus-first access *of a region*, so the value "
                   "is a property of that granule choice as well as of the data.")

    marginal = runs[runs["validity"] == "marginal"]
    if len(marginal):
        notes.append(f"{len(marginal)} run(s) are `marginal` and carry a stated "
                     "caveat in their metadata.")

    # A dataset meant to be reproducible contains no patched artifacts: every
    # run's labels must come from the collecting pipeline, which is verifiable
    # from the code alone.  Runs carrying a `relabelled` block were corrected
    # after the fact and should be re-collected before release.
    relabelled = runs[runs["relabelled"]]
    if len(relabelled):
        notes.append(f"{len(relabelled)} run(s) carry a `relabelled` block: their "
                     "region labels were corrected after collection, not produced "
                     "by the pipeline, so they cannot be reproduced from the code. "
                     "Re-collect them before release.")

    for i, note in enumerate(notes, 1):
        print(f"  {i}. {note}")


# ── helpers ───────────────────────────────────────────────────────────────────

def print_header(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def _write_csv(df: pd.DataFrame, path: pathlib.Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    except OSError as exc:
        print(f"  (could not write {path}: {exc})")


def make_figures(contrast: pd.DataFrame, ctx_scaling: pd.DataFrame,
                 out_dir: pathlib.Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  (matplotlib not installed — skipping figures; "
              "pip install matplotlib to enable)")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    sub = contrast.dropna(subset=["retention p50 (s)"])
    ax.barh(sub["region_type"], sub["retention p50 (s)"])
    ax.set_xlabel("retention p50 (s)")
    ax.set_title("Median retention by region type")
    fig.tight_layout()
    fig.savefig(out_dir / "retention_by_region.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for region, g in ctx_scaling.groupby("region_type"):
        ax.plot(g["ctx"], g["retention_p50_s"], marker="o", label=region)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("context length")
    ax.set_ylabel("retention p50 (s)")
    ax.set_title("Retention vs context length")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "retention_vs_context.png", dpi=150)
    plt.close(fig)
    print(f"\n  figures written to {out_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results", type=pathlib.Path)
    ap.add_argument("--out", default=None, type=pathlib.Path,
                    help="where to write CSVs and figures "
                         "(default: <results>/analysis)")
    ap.add_argument("--with-traces", action="store_true",
                    help="also read trace.parquet for the cache-level section "
                         "(2.3 GB across the dataset)")
    ap.add_argument("--trace-limit", type=int, default=6,
                    help="how many traces to read with --with-traces (default 6)")
    args = ap.parse_args()

    out_dir = args.out or (args.results / "analysis")
    runs, metrics = load_dataset(args.results)
    if runs.empty:
        print(f"No exported runs under {args.results}")
        return 1
    data = analysis_set(runs, metrics)

    print_header("MRM-trace exploratory analysis")
    print(f"  {len(runs)} runs, {len(metrics)} region rows, "
          f"{data['run_id'].nunique()} runs in the analysis set "
          f"(rep > {_WARMUP_REP}, not contaminated)")
    print(f"  output: {out_dir}")

    section_inventory(runs, metrics)
    contrast = section_region_contrast(data, out_dir)
    section_model_scaling(data, out_dir)
    ctx_scaling = section_context_scaling(data, out_dir)
    kv_blocks = section_kv_lifecycle(runs, out_dir)
    if args.with_traces:
        section_memory_hierarchy(runs, out_dir, args.trace_limit)
    section_caveats(runs, metrics, data, kv_blocks)
    make_figures(contrast, ctx_scaling, out_dir)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

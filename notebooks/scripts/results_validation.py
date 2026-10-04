#!/usr/bin/env python3
"""
Validate a completed dataset before release.

The per-run auditor (mrm_trace.reporter.run_audit) checks that each run is
internally sound.  This script checks the things only visible across runs:
whether the matrix is complete, whether every model actually produced the
regions the analysis needs, whether repetitions of one configuration agree,
and whether the acquisition parameters are recorded everywhere.

Each check reports FAIL (blocks release), WARN (publishable with a stated
caveat) or PASS.  Exit code is non-zero if anything FAILs.

    python notebooks/scripts/results_validation.py
    python notebooks/scripts/results_validation.py --results results --json report.json

Why these particular checks: every one of them corresponds to a defect that
previously shipped a dataset that looked healthy.  Region attribution that
labelled rows outside the range they claimed, a region type silently absent for
the largest models, metrics whose value depended on which repetition you looked
at — none showed up in a quality gate or a validity verdict.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from mrm_trace.labeller.proc_maps import MAX_REGION_PAGES  # noqa: E402
from mrm_trace.reporter.run_audit import audit_run  # noqa: E402

# rep 0 is a warmup by convention (experiment_runner.REPS); only later reps
# carry data, so completeness and reproducibility are judged on those.
_WARMUP_REP = 0

# Region types the MRM comparison needs from a CPU run.  activation and
# runtime are inferred, weight and kv_cache are address-grounded.
_EXPECTED_CPU_REGIONS = ("weight", "kv_cache", "activation", "runtime")

# Across repetitions of one configuration, a region's share of the trace should
# be stable.  A large spread is the signature of propagated mis-attribution
# rather than measurement noise: one earlier dataset had kv_cache swing between
# 0.001% and 13% of the trace across five identical runs.
_MAX_SHARE_RATIO = 5.0

# Severity depends on what the region claims to be.
#
# An address-grounded region (weight, kv_cache) says "this address is in this
# mapping".  If that varies across identical runs the attribution is broken and
# the dataset cannot be released.
#
# An inferred region says "a forward-pass kernel stored here", which is a
# refinement of an arena the mapping only identifies as a bag.  `activation` is
# measured at 0.01-1.3% of the trace and swings up to 132x, so it is real but
# not a quantity to compare across runs.  That is a caveat to declare, not a
# release blocker — provided the analysis declares it, which
# results_exploratory_data_analysis.py does.
_INFERRED_REGIONS = ("activation",)

# Retention is last-minus-first access *of a region*, so it needs the same
# region sampled at least twice.  Below this the distribution is 0 by
# construction at the granule used.
_MIN_SAMPLES_PER_REGION = 2.0


class Report:
    """Accumulates check results and prints them as one report."""

    def __init__(self) -> None:
        self.results: list = []

    def add(self, check: str, status: str, summary: str, detail=None) -> None:
        self.results.append({"check": check, "status": status,
                             "summary": summary, "detail": detail or []})

    def failed(self) -> bool:
        return any(r["status"] == "FAIL" for r in self.results)

    def print(self) -> None:
        width = max(len(r["check"]) for r in self.results)
        print()
        print("=" * 78)
        print("  Dataset validation")
        print("=" * 78)
        for r in self.results:
            print(f"  [{r['status']:<4}] {r['check']:<{width}}  {r['summary']}")
            for line in r["detail"]:
                print(f"           {line}")
        counts = defaultdict(int)
        for r in self.results:
            counts[r["status"]] += 1
        print("-" * 78)
        print(f"  {counts['PASS']} pass, {counts['WARN']} warn, {counts['FAIL']} fail")
        print("=" * 78)


# ── loading ───────────────────────────────────────────────────────────────────

def load_runs(results_dir: pathlib.Path) -> pd.DataFrame:
    """
    One row per run with its metadata, plus the per-region metrics attached
    separately.  Incomplete directories are reported, not silently skipped.
    """
    rows, incomplete = [], []
    for meta_path in sorted(results_dir.glob("*/*/*/metadata.json")):
        d = meta_path.parent
        if not (d / "metrics.csv").exists():
            incomplete.append(d.name)
            continue
        meta = json.loads(meta_path.read_text())
        gen, coll, model = meta["generation"], meta["collector"], meta["model"]
        rows.append({
            "run_dir": str(d),
            "run_id": meta.get("run_id", d.name),
            "model_id": model["id"],
            "params_b": model.get("params_billions"),
            "model_path": meta.get("engine", {}).get("model_path"),
            "collector": coll["type"],
            "ctx": gen["context_length"],
            "rep": gen.get("rep_index"),
            "output_tokens": gen.get("output_tokens"),
            "prompts": gen.get("prompts"),
            "sample_period": coll.get("sample_period"),
            "granularity": coll.get("region_granularity_bytes"),
            "events": ",".join(coll.get("event_strings") or []),
            "samples": coll.get("samples_collected"),
            "achieved_rate": coll.get("achieved_sample_rate_per_cpu"),
            "cap_before": coll.get("perf_event_max_sample_rate_before"),
            "cap_after": coll.get("perf_event_max_sample_rate_after"),
            "validity": meta.get("thermal", {}).get("run_validity"),
            "peak_temp_c": meta.get("thermal", {}).get("peak_temp_c"),
        })
    df = pd.DataFrame(rows)
    df.attrs["incomplete"] = incomplete
    return df


def load_metrics(runs: pd.DataFrame) -> pd.DataFrame:
    """Per-run, per-region-type metrics for every run."""
    frames = []
    for _, run in runs.iterrows():
        m = pd.read_csv(pathlib.Path(run["run_dir"]) / "metrics.csv")
        for col in ("run_id", "model_id", "collector", "ctx", "rep", "params_b"):
            m[col] = run[col]
        frames.append(m)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_region_maps(runs: pd.DataFrame) -> pd.DataFrame:
    """
    region_map rows for every run — 27 MB across 61 runs, so loading all of
    them is cheap, unlike the 2.3 GB of traces.
    """
    frames = []
    for _, run in runs.iterrows():
        path = pathlib.Path(run["run_dir"]) / "region_map.parquet"
        if not path.exists():
            continue
        rm = pd.read_parquet(path, columns=["region_type", "access_count"])
        rm["run_id"] = run["run_id"]
        rm["model_id"] = run["model_id"]
        rm["collector"] = run["collector"]
        rm["ctx"] = run["ctx"]
        frames.append(rm)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── checks ────────────────────────────────────────────────────────────────────

def check_per_run_audit(runs: pd.DataFrame, report: Report) -> None:
    """Run the per-run auditor over the whole dataset."""
    failures = {}
    for _, run in runs.iterrows():
        audit = audit_run(run["run_dir"])
        if audit.failures:
            failures[run["run_id"]] = audit.failures
    if failures:
        detail = []
        for run_id, items in list(failures.items())[:10]:
            detail.append(f"{run_id}:")
            detail += [f"  x {i}" for i in items]
        if len(failures) > 10:
            detail.append(f"... and {len(failures) - 10} more")
        report.add("per-run audit", "FAIL",
                   f"{len(failures)} of {len(runs)} runs violate an invariant",
                   detail)
    else:
        report.add("per-run audit", "PASS",
                   f"all {len(runs)} runs pass every per-run invariant")


def check_completeness(runs: pd.DataFrame, report: Report) -> None:
    """Every (model, ctx) should have the same set of repetitions."""
    detail = []
    incomplete = runs.attrs.get("incomplete", [])
    if incomplete:
        detail.append(f"incomplete directories (no exports): {len(incomplete)}")
        detail += [f"  - {name}" for name in incomplete[:5]]

    expected_reps = sorted(r for r in runs["rep"].dropna().unique())
    missing = []
    for (model, ctx), group in runs.groupby(["model_id", "ctx"]):
        have = sorted(group["rep"].dropna().unique())
        gaps = [r for r in expected_reps if r not in have]
        if gaps:
            missing.append(f"{model} ctx{ctx}: missing rep(s) {gaps}")

    status = "PASS"
    if missing:
        status = "WARN"
        detail += missing[:10]
    if incomplete:
        status = "WARN"
    report.add("matrix completeness", status,
               f"{len(runs)} runs over {runs['model_id'].nunique()} models, "
               f"reps {expected_reps}", detail)


def check_region_coverage(runs: pd.DataFrame, metrics: pd.DataFrame,
                          report: Report) -> None:
    """
    Every CPU model must have produced the address-grounded regions.

    A region type absent for some models and present for others is the most
    consequential defect in this dataset class: the analysis silently compares
    models over different region sets.  Where `weight` is missing this also
    diagnoses the cause, because there is a specific one — the GGUF mapping is
    skipped when it exceeds MAX_REGION_PAGES.
    """
    present = (metrics.groupby(["model_id", "collector"])["region_type"]
               .apply(set).to_dict())
    detail, missing_any = [], False

    for (model, collector), have in sorted(present.items()):
        if collector != "perf_mem":
            continue
        gone = [r for r in _EXPECTED_CPU_REGIONS if r not in have]
        if not gone:
            continue
        missing_any = True
        detail.append(f"{model}: missing {', '.join(gone)}")
        if "weight" in gone:
            detail.append(_diagnose_missing_weight(runs, model))

    # The GPU path registers its ranges from PyTorch tensors; if those files are
    # empty the run can only produce runtime/unknown.
    for (model, collector), have in sorted(present.items()):
        if collector == "perf_mem":
            continue
        gone = [r for r in ("weight", "kv_cache") if r not in have]
        if gone:
            missing_any = True
            detail.append(f"{model} ({collector}): missing {', '.join(gone)}"
                          + _diagnose_empty_pytorch_regions(runs, model))

    report.add("region coverage", "FAIL" if missing_any else "PASS",
               "every model produced all expected regions" if not missing_any
               else "some models are missing region types the analysis compares",
               detail)


def _diagnose_missing_weight(runs: pd.DataFrame, model: str) -> str:
    """Name the cause when the weight mapping was never registered."""
    row = runs[runs["model_id"] == model].head(1)
    model_path = row.iloc[0]["model_path"] if not row.empty else None
    # GPU runs record an HF repo id, or nothing, rather than a local file.
    if not isinstance(model_path, str) or not model_path:
        return "  (cause undetermined: no local model file recorded)"
    path = pathlib.Path(model_path)
    if not path.exists():
        return f"  (cause undetermined: {path} not found)"
    pages = path.stat().st_size // 4096
    if pages > MAX_REGION_PAGES:
        return (f"  cause: {path.name} is {pages:,} pages, over the "
                f"MAX_REGION_PAGES limit of {MAX_REGION_PAGES:,} "
                f"({MAX_REGION_PAGES * 4096 / 1e9:.2f} GB), so the GGUF mapping "
                f"was skipped and no weight range was registered")
    return f"  cause undetermined: {path.name} is {pages:,} pages, under the limit"


def _diagnose_empty_pytorch_regions(runs: pd.DataFrame, model: str) -> str:
    row = runs[runs["model_id"] == model].head(1)
    if row.empty:
        return ""
    d = pathlib.Path(row.iloc[0]["run_dir"])
    bits = []
    for name, label in (("pytorch_regions.json", "weight"),
                        ("kv_ranges.json", "kv_cache")):
        path = d / name
        if path.exists():
            try:
                n = len(json.loads(path.read_text()))
            except (OSError, ValueError):
                continue
            bits.append(f"{name} has {n} entries ({label})")
    return ("\n           cause: " + "; ".join(bits)) if bits else ""


def check_provenance(runs: pd.DataFrame, report: Report) -> None:
    """
    A sample count means nothing without the period; a retention figure means
    nothing without the granule.  Both must be on every run.
    """
    required = {"sample_period": "sampling period",
                "granularity": "region granularity",
                "events": "PMU event strings",
                "samples": "sample count",
                "achieved_rate": "achieved sampling rate",
                "cap_before": "sampling cap at start",
                "cap_after": "sampling cap at end"}
    detail, missing_any = [], False
    for col, label in required.items():
        blank = runs[runs[col].isna() | (runs[col] == "")]
        if len(blank):
            missing_any = True
            detail.append(f"{label} missing on {len(blank)}/{len(runs)} runs "
                          f"(e.g. {blank.iloc[0]['run_id']})")
    report.add("provenance", "FAIL" if missing_any else "PASS",
               "acquisition parameters recorded on every run" if not missing_any
               else "runs are missing parameters needed to interpret them",
               detail)


def check_sampling_consistency(runs: pd.DataFrame, report: Report) -> None:
    """
    Sample counts are comparable only at a common cap, and a cap that fell
    during a run while sampling near it means density was not uniform.
    """
    detail, status = [], "PASS"
    clipped = runs[(runs["cap_after"].notna()) & (runs["cap_before"].notna())
                   & (runs["cap_after"] < runs["cap_before"])
                   & (runs["achieved_rate"].notna())]
    near = clipped[clipped["achieved_rate"] >= 0.5 * clipped["cap_after"]]
    if len(near):
        status = "FAIL"
        detail.append(f"{len(near)} run(s) sampled near a cap that fell mid-run "
                      "— density was not uniform")
        detail += [f"  - {r.run_id}: {r.achieved_rate:,.0f}/s/CPU vs cap "
                   f"{r.cap_before:,}->{r.cap_after:,}" for r in near.itertuples()]
    elif len(clipped):
        detail.append(f"{len(clipped)} run(s) saw the cap fall, but all sampled "
                      "far below it — no clipping")

    periods = runs.groupby(["model_id", "ctx"])["sample_period"].nunique()
    varied = periods[periods > 1]
    if len(varied):
        if status == "PASS":
            status = "WARN"
        detail.append(f"{len(varied)} (model, ctx) group(s) used more than one "
                      "sampling period — the calibration adjusted mid-sweep, so "
                      "compare sample counts with care")
        detail += [f"  - {m} ctx{c}: {n} periods" for (m, c), n in varied.items()][:5]
    report.add("sampling consistency", status,
               "no run was clipped by a falling cap", detail)


def check_reproducibility(metrics: pd.DataFrame, region_maps: pd.DataFrame,
                          report: Report) -> None:
    """
    Repetitions of one configuration should agree.

    A region's share of the trace swinging by orders of magnitude across
    identical runs is the signature of mis-attribution: whether a rare sample
    landed on a hot granule, not a property of the workload.
    """
    if region_maps.empty:
        report.add("reproducibility", "WARN", "no region maps to compare")
        return

    shares = (region_maps.groupby(["model_id", "ctx", "run_id", "region_type"])
              ["access_count"].sum().reset_index())
    totals = shares.groupby("run_id")["access_count"].transform("sum")
    shares["share"] = shares["access_count"] / totals

    grounded_bad, inferred_bad, worst_grounded = [], [], 0.0
    for (model, ctx, region), g in shares.groupby(["model_id", "ctx", "region_type"]):
        if len(g) < 2:
            continue
        lo, hi = g["share"].min(), g["share"].max()
        bucket = inferred_bad if region in _INFERRED_REGIONS else grounded_bad
        if lo <= 0:
            bucket.append(f"{model} ctx{ctx} {region}: share is 0 in "
                          f"{(g['share'] == 0).sum()} of {len(g)} reps")
            continue
        ratio = hi / lo
        if region not in _INFERRED_REGIONS:
            worst_grounded = max(worst_grounded, ratio)
        if ratio > _MAX_SHARE_RATIO:
            bucket.append(f"{model} ctx{ctx} {region}: share varies {ratio:.1f}x "
                          f"across {len(g)} reps ({lo:.2%} to {hi:.2%})")

    detail = list(grounded_bad[:8])
    if inferred_bad:
        detail.append(f"inferred regions ({', '.join(_INFERRED_REGIONS)}) are "
                      f"unstable in {len(inferred_bad)} cell(s) — declare them as "
                      f"inferred, do not compare them across runs:")
        detail += [f"  {line}" for line in inferred_bad[:6]]

    status = "FAIL" if grounded_bad else ("WARN" if inferred_bad else "PASS")
    report.add("reproducibility", status,
               f"address-grounded regions vary at most {worst_grounded:.1f}x "
               f"across reps (threshold {_MAX_SHARE_RATIO:.0f}x)", detail)


def check_retention_support(region_maps: pd.DataFrame, runs: pd.DataFrame,
                            report: Report) -> None:
    """
    Retention needs a region sampled at least twice at the granule used.

    The end-of-run granularity sweep reports this per run but is not persisted,
    so it is recomputed here from region_map access counts — the same quantity
    the sweep's `samples/region` column reports.
    """
    if region_maps.empty:
        report.add("retention support", "WARN", "no region maps to check")
        return
    per = (region_maps.groupby(["model_id", "ctx", "region_type"])["access_count"]
           .median().reset_index(name="median_samples_per_region"))
    thin = per[per["median_samples_per_region"] < _MIN_SAMPLES_PER_REGION]
    detail = [f"{r.model_id} ctx{r.ctx} {r.region_type}: median "
              f"{r.median_samples_per_region:.1f} samples/region"
              for r in thin.itertuples()]
    granules = sorted(runs["granularity"].dropna().unique())
    report.add("retention support", "WARN" if detail else "PASS",
               f"granule(s) used: {', '.join(f'{int(g) // 1024} KB' for g in granules)}; "
               f"{len(thin)} (model, ctx, region) cell(s) below "
               f"{_MIN_SAMPLES_PER_REGION:.0f} samples/region",
               detail[:12])


def check_validity(runs: pd.DataFrame, report: Report) -> None:
    """Which runs may enter the aggregated metrics."""
    counts = runs["validity"].value_counts().to_dict()
    data_runs = runs[runs["rep"] != _WARMUP_REP]
    usable = data_runs[data_runs["validity"].isin(["clean", "marginal"])]
    detail = [f"{k}: {v}" for k, v in sorted(counts.items())]
    detail.append(f"analysis set (rep > {_WARMUP_REP}, clean or marginal): "
                  f"{len(usable)} of {len(data_runs)} data runs")
    bad = runs[runs["validity"] == "contaminated"]
    if len(bad):
        detail += [f"excluded (contaminated): {r}" for r in bad["run_id"].head(5)]
    status = "FAIL" if len(usable) == 0 else (
        "WARN" if len(bad) or "marginal" in counts else "PASS")
    report.add("run validity", status,
               f"{counts.get('clean', 0)} clean, {counts.get('marginal', 0)} "
               f"marginal, {counts.get('contaminated', 0)} contaminated", detail)


def check_metric_sanity(metrics: pd.DataFrame, report: Report) -> None:
    """
    Published metrics must be interpretable: retention finite, write-once in
    [0, 1], and an unbounded reads-per-write only where writes were genuinely
    unobservable (write_once is NaN exactly then, because the ratio is 0/0).
    """
    detail = []
    finite_cols = [c for c in ("retention_p50_s", "retention_p90_s",
                               "retention_p99_s", "retention_mean_s",
                               "working_set_mb_total") if c in metrics.columns]
    for col in finite_cols:
        bad = metrics[~metrics[col].apply(lambda v: pd.notna(v) and abs(v) != float("inf"))]
        if len(bad):
            detail.append(f"{col}: {len(bad)} non-finite value(s), e.g. "
                          f"{bad.iloc[0]['model_id']}/{bad.iloc[0]['region_type']}")
    if "write_once_ratio_mean" in metrics:
        wo = metrics["write_once_ratio_mean"]
        out = metrics[wo.notna() & ((wo < 0) | (wo > 1))]
        if len(out):
            detail.append(f"write_once_ratio_mean outside [0,1] on {len(out)} row(s)")
    if {"reads_per_write_mean", "write_once_ratio_mean"} <= set(metrics.columns):
        inf_rows = metrics[metrics["reads_per_write_mean"] == float("inf")]
        wrong = inf_rows[inf_rows["write_once_ratio_mean"].notna()]
        if len(wrong):
            detail.append(f"reads_per_write_mean is inf on {len(wrong)} row(s) that "
                          "also have a defined write_once ratio — writes were "
                          "observed, so it cannot be unbounded")
    report.add("metric sanity", "FAIL" if detail else "PASS",
               f"{len(metrics)} region rows across {metrics['run_id'].nunique()} runs",
               detail)


# ── main ──────────────────────────────────────────────────────────────────────

def validate(results_dir: pathlib.Path) -> "tuple[Report, dict]":
    runs = load_runs(results_dir)
    if runs.empty:
        report = Report()
        report.add("dataset", "FAIL", f"no exported runs under {results_dir}")
        return report, {}
    metrics = load_metrics(runs)
    region_maps = load_region_maps(runs)

    report = Report()
    check_per_run_audit(runs, report)
    check_completeness(runs, report)
    check_region_coverage(runs, metrics, report)
    check_provenance(runs, report)
    check_sampling_consistency(runs, report)
    check_reproducibility(metrics, region_maps, report)
    check_retention_support(region_maps, runs, report)
    check_validity(runs, report)
    check_metric_sanity(metrics, report)
    return report, {"runs": len(runs), "models": int(runs["model_id"].nunique())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results", type=pathlib.Path,
                    help="results root (default: results)")
    ap.add_argument("--json", type=pathlib.Path,
                    help="also write the report as JSON")
    args = ap.parse_args()

    report, summary = validate(args.results)
    report.print()
    if args.json:
        args.json.write_text(json.dumps(
            {"summary": summary, "checks": report.results}, indent=2, default=str))
        print(f"  report written to {args.json}")
    if report.failed():
        print("\n  Dataset is NOT ready for release — see FAIL rows above.\n")
        return 1
    print("\n  No blocking issues. Review any WARN rows and state them in the "
          "data card.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Package a completed sweep in results/ as a Kaggle dataset under datasets/.

What goes in, and why, is decided in
context-files/06 - Kaggle Dataset Selection.md.  In short:

    <out>/
    ├── README.md                    data card (generated; edit the prose before upload)
    ├── dataset-metadata.json        for `kaggle datasets create -p <out>`
    ├── SHA256SUMS
    ├── runs.csv                     one row per run: manifest + flattened metadata
    ├── metrics.csv                  every run's metrics.csv, stacked, keyed by run_id
    ├── region_map.parquet           every run's region_map, stacked
    ├── kv_block_lifecycle.parquet   every run's KV block table, stacked
    ├── sample_calibration.json      sampling period per (model, output tokens)
    ├── analysis/                    cross-run summaries, copied unchanged
    └── runs/<model_id>/<run_id>/
        ├── metadata.json
        ├── prompt_outputs.json
        └── trace.parquet            stratified subsample (see below)

raw/perf.data (14 GB of a 16 GB sweep) is not published: everything a user
can read out of it is already in trace.parquet.

Trace subsampling.  ~98% of trace rows are `runtime`; `kv_cache` and `weight`
are already sparse (a few thousand rows per run, many regions single-sample).
A uniform subsample would thin exactly the regions the dataset is about, so
by default every non-runtime row is kept and runtime rows are kept with
probability --runtime-fraction.  Each row carries `sample_weight` = 1/p so
counts can be re-weighted.  Metrics in metrics.csv were computed from the full
trace; recomputing runtime write-once or retention from the subsample will not
reproduce them, and the data card says so.  --trace-mode full ships traces
unthinned (~2.6 GB for the 30-run sweep).

Privacy.  machine_id, hostname, output paths and the dso column contain the
collecting user's name and home directory.  They are rewritten to a machine
alias and /home/user/, and the finished package is scanned for the username;
the script fails if it is still present anywhere in a text file or in a
string column of a parquet file.

    venv/bin/python notebooks/scripts/prepare_kaggle_dataset.py
    venv/bin/python notebooks/scripts/prepare_kaggle_dataset.py \\
        --runtime-fraction 0.05 --kaggle-id myname/mrm-trace-llm-inference --force
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import pathlib
import re
import shutil
import sys
import zlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPO = pathlib.Path(__file__).resolve().parents[2]

_PER_RUN_FILES = ("metadata.json", "prompt_outputs.json", "trace.parquet")
_STACKED_PARQUET = ("region_map.parquet", "kv_block_lifecycle.parquet")
_ANALYSIS_FILES = (
    "region_contrast.csv",
    "model_scaling.csv",
    "context_scaling.csv",
    "kv_lifecycle_blocks.csv",
)
# Flattened metadata fields that are local paths or identify the machine's
# owner rather than describe the measurement.
_DROP_METADATA = {
    "engine.model_path",
    "collector.output_file",
    "software.hostname",
    "software.perf_bin",
}
_HOME_RE = re.compile(r"/home/[^/]+/")

_TRACE_BATCH_ROWS = 1_000_000  # ADR-2: stream, never load a whole trace


# ---------------------------------------------------------------- privacy ---

class Anonymiser:
    def __init__(self, machine_id: str, alias: str, usernames: set[str]):
        self.machine_id = machine_id
        self.alias = alias
        self.usernames = {u for u in usernames if u and u != "root"}

    def text(self, s: str) -> str:
        s = s.replace(self.machine_id, self.alias)
        s = _HOME_RE.sub("/home/user/", s)
        for u in self.usernames:
            s = re.sub(re.escape(u), "user", s, flags=re.IGNORECASE)
        return s

    def obj(self, o):
        if isinstance(o, dict):
            return {k: self.obj(v) for k, v in o.items()}
        if isinstance(o, list):
            return [self.obj(v) for v in o]
        if isinstance(o, str):
            return self.text(o)
        return o

    def arrow_strings(self, col: pa.ChunkedArray | pa.Array) -> pa.Array:
        col = pc.replace_substring_regex(col, _HOME_RE.pattern, "/home/user/")
        for u in self.usernames:
            col = pc.replace_substring_regex(col, f"(?i){re.escape(u)}", "user")
        return col


def _usernames() -> set[str]:
    names = {getpass.getuser(), os.environ.get("SUDO_USER", "")}
    m = re.match(r"/home/([^/]+)/", str(REPO) + "/")
    if m:
        names.add(m.group(1))
    return names


def _default_alias(machine_id: str, usernames: set[str]) -> str:
    for u in sorted(usernames, key=len, reverse=True):
        if u and machine_id.lower().startswith(u.lower() + "-"):
            return machine_id[len(u) + 1:]
    return "machine-01"


# ------------------------------------------------------------- selection ---

def select_runs(results: pathlib.Path, include_contaminated: bool) -> pd.DataFrame:
    manifest = pd.read_csv(results / "manifest.csv")
    if not include_contaminated:
        manifest = manifest[manifest["run_validity"] != "contaminated"]
    dirs, missing = [], []
    for row in manifest.itertuples():
        d = results / row.machine_id / row.model_id / row.run_id
        absent = [f for f in (*_PER_RUN_FILES, *_STACKED_PARQUET, "metrics.csv")
                  if not (d / f).exists()]
        if absent:
            missing.append(f"{row.run_id}: {', '.join(absent)}")
        dirs.append(d)
    if missing:
        sys.exit("runs listed in manifest.csv are incomplete:\n  " + "\n  ".join(missing))
    manifest = manifest.assign(run_dir=dirs)
    return manifest.sort_values(["model_id", "max_tokens_per_prompt", "rep"]).reset_index(drop=True)


# ----------------------------------------------------------------- traces ---

def write_trace(src: pathlib.Path, dst: pathlib.Path, run_id: str, mode: str,
                runtime_fraction: float, seed: int, anon: Anonymiser) -> dict:
    rng = np.random.default_rng([seed, zlib.crc32(run_id.encode())])
    pf = pq.ParquetFile(src)
    writer = None
    rows_in = rows_out = 0
    by_region_in: dict[str, int] = {}
    by_region_out: dict[str, int] = {}
    try:
        for batch in pf.iter_batches(batch_size=_TRACE_BATCH_ROWS):
            tbl = pa.Table.from_batches([batch])
            rows_in += tbl.num_rows
            region = tbl.column("region_type").to_numpy(zero_copy_only=False)
            if mode == "stratified":
                is_runtime = region == "runtime"
                keep = ~is_runtime | (rng.random(tbl.num_rows) < runtime_fraction)
                weight = np.where(is_runtime, 1.0 / runtime_fraction, 1.0).astype("float32")
                tbl = tbl.filter(pa.array(keep))
                weight = weight[keep]
                kept_region = region[keep]
            else:
                weight = np.ones(tbl.num_rows, dtype="float32")
                kept_region = region
            for r, n in zip(*np.unique(region.astype(str), return_counts=True)):
                by_region_in[r] = by_region_in.get(r, 0) + int(n)
            for r, n in zip(*np.unique(kept_region.astype(str), return_counts=True)):
                by_region_out[r] = by_region_out.get(r, 0) + int(n)
            tbl = tbl.set_column(tbl.schema.get_field_index("dso"), "dso",
                                 anon.arrow_strings(tbl.column("dso")))
            tbl = tbl.append_column("sample_weight", pa.array(weight))
            # Drop the pandas index metadata inherited from the source file.
            tbl = tbl.replace_schema_metadata(None)
            if writer is None:
                writer = pq.ParquetWriter(dst, tbl.schema, compression="zstd")
            writer.write_table(tbl)
            rows_out += tbl.num_rows
    finally:
        if writer is not None:
            writer.close()
    return {"run_id": run_id, "rows_full": rows_in, "rows_published": rows_out,
            "rows_full_by_region": by_region_in, "rows_published_by_region": by_region_out}


# ---------------------------------------------------------- stacked tables ---

def _with_run_keys(df: pd.DataFrame, run) -> pd.DataFrame:
    keys = pd.DataFrame({
        "run_id": run.run_id,
        "model_id": run.model_id,
        "output_tokens": run.max_tokens_per_prompt,
        "rep": run.rep,
    }, index=df.index)
    return pd.concat([keys, df], axis=1)


def stack_tables(runs: pd.DataFrame, out: pathlib.Path) -> None:
    pd.concat([_with_run_keys(pd.read_csv(r.run_dir / "metrics.csv"), r)
               for r in runs.itertuples()], ignore_index=True) \
        .to_csv(out / "metrics.csv", index=False)
    for name in _STACKED_PARQUET:
        df = pd.concat([_with_run_keys(pd.read_parquet(r.run_dir / name), r)
                        for r in runs.itertuples()], ignore_index=True)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False).replace_schema_metadata(None),
                       out / name, compression="zstd")


def build_runs_table(runs: pd.DataFrame, anon: Anonymiser) -> pd.DataFrame:
    base = runs.drop(columns="run_dir").copy()
    base["machine_id"] = anon.alias
    flat = []
    for r in runs.itertuples():
        md = json.loads((r.run_dir / "metadata.json").read_text())
        row = pd.json_normalize(md, sep=".").iloc[0].to_dict()
        flat.append({k: v for k, v in row.items()
                     if k not in _DROP_METADATA and not isinstance(v, (list, dict))})
    flat = pd.DataFrame(flat)
    extra = [c for c in flat.columns if c not in base.columns and c != "run_id"]
    return base.merge(flat[["run_id", *extra]], on="run_id", how="left")


# --------------------------------------------------------------- data card ---

_COLUMN_DOCS = {
    "trace.parquet": {
        "run_id": "Run this sample belongs to.",
        "sample_id": "Sequence number of the sample within the full (unthinned) trace.",
        "timestamp_ns": "perf sample time, nanoseconds on the monotonic clock (µs resolution).",
        "phase": "Inference phase: weight_load or generation; null before the engine reports a phase.",
        "address": "Sampled virtual data address.",
        "address_page": "address >> 12 (4 KiB page).",
        "op_type": "load or store. Loads and stores are sampled on different periods (see Sampling).",
        "symbol": "Function executing the access (resolved by perf); not used to label regions.",
        "dso": "Shared object containing that function; home directories rewritten to /home/user/.",
        "region_type": "weight / kv_cache / runtime / unknown, from address ranges (see Labels).",
        "region_id": "Region within the run; joins to region_map.region_id.",
        "access_latency_ns": "Load latency as reported by PEBS (the perf 'weight' field).",
        "cache_level": "Where the access was satisfied (L1/L2/L3/LFB/RAM ...).",
        "numa_node": "NUMA node of the access.",
        "cpu_id": "CPU the sample was taken on, when recorded.",
        "thread_id": "OS thread id.",
        "token_index": "Output token being generated, when known.",
        "context_tokens": "Context length in tokens at the time of the sample, when known.",
        "sample_weight": "1 / keep-probability in this release (1.0 unless the row was subsampled).",
    },
    "region_map.parquet": {
        "region_id": "Region within the run.",
        "region_type": "weight / kv_cache / runtime / unknown.",
        "page_min": "First 4 KiB page of the region.",
        "page_max": "Last 4 KiB page of the region.",
        "first_seen_ns": "Timestamp of the region's first sampled access.",
        "last_seen_ns": "Timestamp of the region's last sampled access.",
        "access_count": "Sampled accesses in the full trace.",
    },
    "kv_block_lifecycle.parquet": {
        "block_id": "KV block within the run.",
        "address": "Block address.",
        "address_page": "address >> 12.",
        "write_ns": "First sampled store to the block; null if no store was sampled.",
        "last_read_ns": "Last sampled load from the block.",
        "evict_ns": "Eviction time when observed; usually null.",
        "total_reads": "Sampled loads.",
        "write_observed": "Whether any store to the block was sampled.",
        "lifetime_ns": "last_read_ns - write_ns; null when write_observed is false.",
    },
    "metrics.csv": {
        "region_type": "Region type the row summarises.",
        "n_regions": "Regions of this type in the run.",
        "n_regions_observable": "Regions with >= 2 samples, the only ones a retention interval exists for.",
        "retention_support_frac": "n_regions_observable / n_regions.",
        "retention_p50_s": "Median observed retention (last - first sampled access) over observable regions.",
        "retention_p90_s": "90th percentile of the same.",
        "retention_p99_s": "99th percentile of the same.",
        "retention_mean_s": "Mean of the same.",
        "write_once_ratio_mean": "Share of written addresses with exactly one sampled store; empty if none written.",
        "reads_per_write_mean": "Sampled loads per sampled store; inf when no store was sampled.",
        "working_set_mb_total": "Distinct pages touched x 4 KiB.",
        "unique_pages_total": "Distinct 4 KiB pages touched.",
        "suitability": "Heuristic MRM suitability label for the region type.",
    },
}

_RUN_KEY_DOCS = {
    "run_id": "Run identifier; joins to runs.csv.",
    "model_id": "Model.",
    "output_tokens": "Output tokens requested for the run.",
    "rep": "Repetition index.",
}


def _schema_table(path: pathlib.Path, docs: dict[str, str]) -> str:
    if path.suffix == ".parquet":
        names = pq.read_schema(path).names
        # Null share from the first row group only keeps this cheap on big files.
        sample = pq.ParquetFile(path).read_row_group(0).to_pandas()
    else:
        sample = pd.read_csv(path)
        names = list(sample.columns)
    lines = ["| column | null % | description |", "|---|---|---|"]
    for n in names:
        null = f"{100 * sample[n].isna().mean():.0f}" if n in sample else ""
        lines.append(f"| `{n}` | {null} | {docs.get(n, _RUN_KEY_DOCS.get(n, ''))} |")
    return "\n".join(lines)


def write_data_card(out: pathlib.Path, runs: pd.DataFrame, rt: pd.DataFrame,
                    trace_stats: list[dict], args, alias: str) -> None:
    first_trace = next((out / "runs").rglob("trace.parquet"))
    md = json.loads(next((out / "runs").rglob("metadata.json")).read_text())
    hw, sw = md["hardware"], md["software"]
    full = sum(s["rows_full"] for s in trace_stats)
    pub = sum(s["rows_published"] for s in trace_stats)
    by_region_full: dict[str, int] = {}
    by_region_pub: dict[str, int] = {}
    for s in trace_stats:
        for k, v in s["rows_full_by_region"].items():
            by_region_full[k] = by_region_full.get(k, 0) + v
        for k, v in s["rows_published_by_region"].items():
            by_region_pub[k] = by_region_pub.get(k, 0) + v
    region_rows = "\n".join(
        f"| {k} | {by_region_full[k]:,} | {by_region_pub.get(k, 0):,} |"
        for k in sorted(by_region_full, key=by_region_full.get, reverse=True))
    models = ", ".join(sorted(runs["model_id"].unique()))
    tokens = ", ".join(str(t) for t in sorted(runs["max_tokens_per_prompt"].unique()))
    validity = ", ".join(f"{n} {v}" for v, n in runs["run_validity"].value_counts().items())
    if args.trace_mode == "stratified":
        trace_note = (f"Every `weight`, `kv_cache` and `unknown` row is kept; `runtime` rows are kept "
                      f"with probability {args.runtime_fraction} (seed {args.seed}), and "
                      f"`sample_weight` = {1 / args.runtime_fraction:g} on those rows. "
                      "`metrics.csv` was computed from the full trace, so recomputing runtime "
                      "retention or write-once ratio from this subsample will not reproduce it.")
    else:
        trace_note = "Traces are published unthinned; `sample_weight` is 1.0 throughout."

    card = f"""# MRM-Trace: memory access traces of LLM inference on CPU

Sampled hardware memory-access traces of {len(runs)} LLM inference runs, labelled by memory
region (model weights, KV cache, runtime), for studying how long data lives in each region —
the question behind Managed-Retention Memory (MRM; Legtchenko et al., 2025).

> **Edit before upload:** this card is generated by `prepare_kaggle_dataset.py`.
> Check the prose, add the citation for the paper/preprint, and choose the licence.

## What was run

| | |
|---|---|
| Models | {models} (GGUF, Q4_K_M, llama.cpp on CPU) |
| Context length | 2048 |
| Output tokens | {tokens} (generated to the limit, EOS ignored) |
| Repetitions | {runs['rep'].nunique()} per configuration |
| Runs | {len(runs)} ({validity}) |
| Machine | `{alias}`: {hw['cpu_model']}, {hw['mem_total_mb'] / 1024:.0f} GB RAM, turbo {'on' if hw['turbo_enabled'] else 'off'}, governor `{hw['cpu_freq_governor']}` |
| OS / kernel | {sw['os']} |
| Collector | `perf mem` (Intel PEBS), {sw['perf_version']}, spawn mode |
| Tool | mrm-trace {sw['mrm_trace_version']} |

## Files

| file | rows | what it is |
|---|---|---|
| `runs.csv` | {len(rt)} | One row per run: configuration, validity, thermal, observer overhead, sampling parameters, hardware. Start here. |
| `metrics.csv` | {len(pd.read_csv(out / 'metrics.csv'))} | Per run x region type: retention percentiles, write-once ratio, reads per write, working set. |
| `region_map.parquet` | {pq.ParquetFile(out / 'region_map.parquet').metadata.num_rows:,} | Address ranges and their region type, per run. |
| `kv_block_lifecycle.parquet` | {pq.ParquetFile(out / 'kv_block_lifecycle.parquet').metadata.num_rows:,} | KV-cache blocks: first write, last read, lifetime, per run. |
| `analysis/region_contrast.csv` | | Headline: each region type summarised across all runs. |
| `analysis/model_scaling.csv` | | Region metrics by model size. |
| `analysis/context_scaling.csv` | | Region metrics by output length. |
| `analysis/kv_lifecycle_blocks.csv` | | All KV blocks pooled, with lifetime in seconds. |
| `sample_calibration.json` | | Load sampling period chosen per (model, output tokens). |
| `runs/<model>/<run_id>/trace.parquet` | {pub:,} total | Sampled memory accesses (see Trace subsampling). |
| `runs/<model>/<run_id>/metadata.json` | | Full run metadata: hardware, software, collector, thermal, observer effect. |
| `runs/<model>/<run_id>/prompt_outputs.json` | | The prompt and the text the model generated. |

`run_id`, `model_id`, `output_tokens` and `rep` join every table.

## Sampling — read this first

`perf mem` **samples** accesses; it does not record every one. Loads are sampled every
`collector.sample_period` loads (calibrated per configuration so each run collects a similar
number of samples), and stores every 500x that period. So:

- A region seen once has no retention interval. `retention_*` is computed over regions with
  >= 2 samples only; `n_regions_observable` and `retention_support_frac` say how many that was.
- A KV block's lifetime needs its write. `lifetime_ns` is **null** when no store to the block
  was sampled (`write_observed = false`) — the large majority of blocks. Absent is never 0.
- `kv_cache` write-once ratio = 1.000 with zero variance is the signature of at most one
  sampled store per address, **not** evidence of write-once behaviour. Treat `write_once_ratio`
  and `reads_per_write` as unsupported for `kv_cache`.
- `weight` has no stores: the GGUF file is mapped read-only. That is a genuine finding.

## Labels

Region types come from address ranges only, never from function names:

| region | source | status |
|---|---|---|
| `weight` | the mmap'd GGUF file, from `/proc/self/maps` | exact |
| `kv_cache` | engine allocation captured at allocation time | exact when unambiguous, else absent |
| `runtime` | thread stacks, heap, malloc arenas, shared libraries | exact, but a catch-all whose contents vary |
| `unknown` | in no known mapping | — |

## Trace subsampling

{trace_note}

| region | rows in full traces | rows published |
|---|---|---|
{region_rows}

## Validity

Each run records the tracer's own CPU and memory overhead, baseline CPU load, temperature and
throttling, and is classified `clean` / `marginal` / `contaminated` from them
(`runs.csv: run_validity`, reasons in `thermal.validity_failure_reasons`).
{'Contaminated runs are excluded from this release.' if not args.include_contaminated else 'Contaminated runs are included; filter on run_validity.'}

## Caveats

- Retention values are **observed lifetimes on this CPU's cache hierarchy under this schedule**,
  not hardware retention requirements. The logical structure (which regions are written once,
  how read-heavy they are, their relative ordering) is what transfers to other memory systems.
- One machine, one engine (llama.cpp, CPU), one context length.
- The trace `dso` column has home directories rewritten to `/home/user/`; the machine is
  identified by an alias.

## Schemas

### trace.parquet
{_schema_table(first_trace, _COLUMN_DOCS['trace.parquet'])}

### region_map.parquet
{_schema_table(out / 'region_map.parquet', _COLUMN_DOCS['region_map.parquet'])}

### kv_block_lifecycle.parquet
{_schema_table(out / 'kv_block_lifecycle.parquet', _COLUMN_DOCS['kv_block_lifecycle.parquet'])}

### metrics.csv
{_schema_table(out / 'metrics.csv', _COLUMN_DOCS['metrics.csv'])}

Null % is measured on the first row group of each file.

## Provenance

Packaged {datetime.now(timezone.utc).strftime('%Y-%m-%d')} from the mrm-trace `results/` directory
by `notebooks/scripts/prepare_kaggle_dataset.py`. File checksums are in `SHA256SUMS`.
"""
    (out / "README.md").write_text(card)


# -------------------------------------------------------------- finishing ---

def write_kaggle_metadata(out: pathlib.Path, kaggle_id: str, license_name: str) -> None:
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": "MRM-Trace: LLM Inference Memory Access Traces",
        "id": kaggle_id,
        "licenses": [{"name": license_name}],
        "subtitle": "perf mem traces of llama.cpp inference, labelled by memory region",
        "keywords": ["memory", "llm", "computer science", "hardware"],
    }, indent=2) + "\n")


def write_checksums(out: pathlib.Path) -> None:
    lines = []
    for f in sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        h = hashlib.sha256()
        with f.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {f.relative_to(out).as_posix()}")
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def privacy_scan(out: pathlib.Path, usernames: set[str]) -> list[str]:
    pattern = re.compile("|".join(re.escape(u) for u in usernames), re.IGNORECASE)
    hits = []
    for f in sorted(p for p in out.rglob("*") if p.is_file()):
        if f.suffix in (".csv", ".json", ".md", ".txt"):
            if pattern.search(f.read_text(errors="replace")):
                hits.append(str(f.relative_to(out)))
        elif f.suffix == ".parquet":
            pfile = pq.ParquetFile(f)
            str_cols = [fld.name for fld in pfile.schema_arrow
                        if pa.types.is_string(fld.type) or pa.types.is_large_string(fld.type)]
            for batch in pfile.iter_batches(columns=str_cols, batch_size=_TRACE_BATCH_ROWS):
                for name, col in zip(batch.schema.names, batch.columns):
                    distinct = pc.unique(col).drop_null().to_pylist()
                    if any(pattern.search(v) for v in distinct):
                        hits.append(f"{f.relative_to(out)}:{name}")
                        break
                else:
                    continue
                break
    return hits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", type=pathlib.Path, default=REPO / "results")
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "datasets" / "mrm-trace-kaggle")
    ap.add_argument("--trace-mode", choices=("stratified", "full"), default="stratified")
    ap.add_argument("--runtime-fraction", type=float, default=0.10,
                    help="keep probability for runtime trace rows in stratified mode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-contaminated", action="store_true")
    ap.add_argument("--machine-alias", help="replaces machine_id (default: machine_id minus the username prefix)")
    ap.add_argument("--kaggle-id", default="USERNAME/mrm-trace-llm-inference")
    ap.add_argument("--license", default="CC-BY-4.0")
    ap.add_argument("--force", action="store_true", help="replace an existing --out directory")
    args = ap.parse_args()

    if not 0 < args.runtime_fraction <= 1:
        ap.error("--runtime-fraction must be in (0, 1]")
    if args.out.exists():
        if not args.force:
            ap.error(f"{args.out} exists; pass --force to replace it")
        shutil.rmtree(args.out)

    runs = select_runs(args.results, args.include_contaminated)
    machines = runs["machine_id"].unique()
    if len(machines) != 1:
        sys.exit(f"expected one machine, found {list(machines)}")
    usernames = _usernames()
    alias = args.machine_alias or _default_alias(machines[0], usernames)
    anon = Anonymiser(machines[0], alias, usernames)
    print(f"{len(runs)} runs from {machines[0]} -> machine alias '{alias}'")

    out = args.out
    (out / "analysis").mkdir(parents=True)

    trace_stats = []
    for i, r in enumerate(runs.itertuples(), 1):
        dst = out / "runs" / r.model_id / r.run_id
        dst.mkdir(parents=True)
        md = anon.obj(json.loads((r.run_dir / "metadata.json").read_text()))
        (dst / "metadata.json").write_text(json.dumps(md, indent=2) + "\n")
        po = anon.obj(json.loads((r.run_dir / "prompt_outputs.json").read_text()))
        (dst / "prompt_outputs.json").write_text(json.dumps(po, indent=2, ensure_ascii=False) + "\n")
        s = write_trace(r.run_dir / "trace.parquet", dst / "trace.parquet", r.run_id,
                        args.trace_mode, args.runtime_fraction, args.seed, anon)
        trace_stats.append(s)
        print(f"[{i:>2}/{len(runs)}] {r.run_id}: {s['rows_full']:,} -> {s['rows_published']:,} trace rows")

    stack_tables(runs, out)
    rt = build_runs_table(runs, anon)
    rt.to_csv(out / "runs.csv", index=False)
    for name in _ANALYSIS_FILES:
        shutil.copy2(args.results / "analysis" / name, out / "analysis" / name)
    calib = args.results / ".sample_calibration.json"
    if calib.exists():
        shutil.copy2(calib, out / "sample_calibration.json")

    write_data_card(out, runs, rt, trace_stats, args, alias)
    write_kaggle_metadata(out, args.kaggle_id, args.license)

    hits = privacy_scan(out, anon.usernames | {machines[0]})
    if hits:
        sys.exit("username still present in:\n  " + "\n  ".join(hits))
    write_checksums(out)

    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"\nwrote {out} ({size / 2**20:,.0f} MB). Privacy scan clean.")
    if args.kaggle_id.startswith("USERNAME/"):
        print("Set --kaggle-id (or edit dataset-metadata.json) before `kaggle datasets create`.")


if __name__ == "__main__":
    main()

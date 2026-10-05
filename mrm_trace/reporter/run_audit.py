"""
Post-export audit of a run against the invariants its metrics depend on.

Every defect these checks look for shipped a run that looked healthy: both
quality gates PASS, a `clean` verdict and a complete metrics.csv.  None was
visible in any summary the pipeline printed, and each was found only because
someone thought to look.  This module exists so that stops being the mechanism.

Used by experiment_runner.py after every run and by
notebooks/scripts/audit_run.py on demand.
"""

from __future__ import annotations

import json
import pathlib
import numpy as np
import pandas as pd

from mrm_trace.telemetry.validity import classify_from_metadata

# Region types whose addresses are ground truth, and must therefore agree
# exactly with the ranges the run recorded.
_EXACT_REGIONS = ("weight", "kv_cache")

# Columns in metrics.csv that must be finite for a region that was accessed.
_FINITE_METRICS = ("retention_p50_s", "retention_p90_s", "retention_p99_s",
                   "retention_mean_s", "working_set_mb_total")


class Audit:
    def __init__(self, run_dir: pathlib.Path):
        self.dir = run_dir
        self.failures: list = []
        self.notes: list = []
        self.incomplete = False

    def fail(self, check: str, detail: str) -> None:
        self.failures.append(f"{check}: {detail}")

    def note(self, text: str) -> None:
        self.notes.append(text)

    # ── checks ───────────────────────────────────────────────────────────────

    def check_artifacts(self) -> bool:
        """
        Distinguish an aborted run from an invalid one.

        A directory with raw/ but no exports is a run that was interrupted
        before export — debris to delete, not a measurement to judge.  Calling
        it a failure buries the runs that really are wrong.
        """
        required = ["trace.parquet", "region_map.parquet", "metadata.json",
                    "metrics.csv", "manifest.json"]
        missing = [f for f in required if not (self.dir / f).exists()]
        if not missing:
            return True
        if len(missing) == len(required):
            self.incomplete = True
            self.note("incomplete: no exported artifacts — interrupted before "
                      "export; delete this directory")
        else:
            self.fail("artifacts", f"missing {', '.join(missing)}")
        return False

    def _absence_reason(self, region_type: str) -> str:
        """
        Explain an absent weight/kv_cache attribution from recorded evidence.

        "by design if it could not be isolated" was a hedge the reader had to
        take on trust.  parameter_devices.json records where the parameters
        actually lived, so a GPU-resident model turns that hedge into a fact:
        a CUDA data_ptr() is a device address no CPU PMU can sample.
        """
        census_file = self.dir / "parameter_devices.json"
        if not census_file.exists():
            return "by design if it could not be isolated (no device census recorded)"
        try:
            census = json.loads(census_file.read_text())
        except Exception:
            return "by design if it could not be isolated (device census unreadable)"
        if not census:
            return "by design if it could not be isolated (device census empty)"
        non_cpu = {d: i for d, i in census.items() if not d.startswith("cpu")}
        on_cpu = {d: i for d, i in census.items() if d.startswith("cpu")}
        if non_cpu and not on_cpu:
            where = ", ".join(f"{i['tensors']} tensors / {i['bytes'] / 1e6:.0f} MB on {d}"
                              for d, i in sorted(non_cpu.items()))
            return (f"expected: all parameters in device memory ({where}), "
                    "which a CPU PMU cannot address")
        if non_cpu and on_cpu:
            return ("partially expected: parameters split across host and device "
                    "memory; only the host share could ever be attributed")
        return ("UNEXPECTED: parameters were in host memory, so this region "
                "should have been isolable")

    def check_attribution(self, meta: dict) -> None:
        """
        weight and kv_cache come from address ranges, so every labelled row
        must be inside its range and every sample in the range must carry the
        label.  A symbol rule once labelled 26,617,751 rows `weight` of which
        114 were inside the weight mapping; nothing in the output showed it.
        """
        gen = self.dir / "generation_results.json"
        if not gen.exists():
            self.note("attribution: no generation_results.json, ranges unknown")
            return
        regions = json.loads(gen.read_text()).get("address_regions", [])
        trace = pd.read_parquet(self.dir / "trace.parquet",
                                columns=["address", "region_type", "op_type"])
        addr = trace["address"].to_numpy()

        for region_type in _EXACT_REGIONS:
            ranges = [(r["addr_start"], r["addr_end"]) for r in regions
                      if r["region_type"] == region_type]
            labelled = (trace["region_type"] == region_type).to_numpy()
            if not ranges:
                if labelled.any():
                    self.fail(f"attribution/{region_type}",
                              f"{labelled.sum():,} rows labelled but no range "
                              "was registered — the label cannot be grounded")
                else:
                    reason = self._absence_reason(region_type)
                    if reason.startswith("UNEXPECTED"):
                        # Host-resident parameters with no attribution is a
                        # pipeline failure, not a platform limit.
                        self.fail(f"attribution/{region_type}",
                                  f"absent (no range, no rows) — {reason}")
                    else:
                        self.note(f"attribution/{region_type}: absent (no range, "
                                  f"no rows) — {reason}")
                continue

            inside = np.zeros(len(addr), dtype=bool)
            for lo, hi in ranges:
                inside |= (addr >= lo) & (addr < hi)

            if labelled.sum():
                precision = (labelled & inside).sum() / labelled.sum()
                if precision < 1.0:
                    self.fail(f"attribution/{region_type}",
                              f"precision {precision:.4%} — "
                              f"{(labelled & ~inside).sum():,} rows labelled "
                              "outside the registered range")
            if inside.sum():
                recall = (labelled & inside).sum() / inside.sum()
                if recall < 1.0:
                    self.fail(f"attribution/{region_type}",
                              f"recall {recall:.4%} — "
                              f"{(inside & ~labelled).sum():,} samples in the "
                              "range carry another label")

        # Weights are mapped read-only, so a store there is impossible.
        w_ranges = [(r["addr_start"], r["addr_end"]) for r in regions
                    if r["region_type"] == "weight"]
        if w_ranges:
            inside = np.zeros(len(addr), dtype=bool)
            for lo, hi in w_ranges:
                inside |= (addr >= lo) & (addr < hi)
            stores = (trace["op_type"] == "store").to_numpy() & inside
            if stores.any():
                self.fail("attribution/weight",
                          f"{stores.sum():,} store samples inside a read-only "
                          "mapping — the range or the op_type is wrong")

    def check_verdict_agrees(self, meta: dict) -> None:
        """
        The verdict the pipeline wrote must equal the verdict its own classifier
        derives from the metadata.  A run once reported `clean` while the
        written metadata re-derived to `marginal`, because the live call was
        built from a partial dict.
        """
        written = (meta.get("thermal") or {}).get("run_validity")
        rederived = classify_from_metadata(meta).run_validity
        if written != rederived:
            self.fail("verdict", f"pipeline wrote {written!r} but metadata "
                                 f"re-derives to {rederived!r}")

    def check_metrics_finite(self) -> None:
        """
        A published metric must be a number.  NaN and inf have both appeared:
        write_once_ratio was 0/0 for the read-only weight region, and
        reads_per_write was inf for a type that did have writes because a mean
        of per-region ratios inherited inf from a read-only sub-region.
        """
        metrics = pd.read_csv(self.dir / "metrics.csv")
        for _, row in metrics.iterrows():
            rt = row.get("region_type")
            for col in _FINITE_METRICS:
                if col in row and not np.isfinite(row[col]):
                    self.fail("metrics", f"{rt}.{col} is {row[col]}")
            # inf is the correct value for a region type with reads and no
            # writes at all; write_once_ratio is NaN exactly in that case,
            # because the ratio is 0/0.  inf alongside a *defined* write-once
            # ratio means writes were seen, so the aggregate is wrong — that is
            # the mean-of-ratios bug, where one read-only sub-region made the
            # whole type unbounded.
            rpw = row.get("reads_per_write_mean")
            wo = row.get("write_once_ratio_mean")
            if rpw is not None and np.isinf(rpw) and wo is not None and not pd.isna(wo):
                self.fail("metrics",
                          f"{rt}.reads_per_write_mean is inf but write_once is "
                          f"{wo} — writes were observed, so it cannot be "
                          "unbounded (mean-of-ratios rather than ratio-of-totals?)")
            if row.get("suitability") == "unknown":
                self.note(f"metrics: {rt} has no suitability verdict")

    def check_region_expectations(self) -> None:
        """
        The claims the experiment exists to test, as sanity bounds rather than
        expected answers: weights read-only, KV written then read.
        """
        trace = pd.read_parquet(self.dir / "trace.parquet",
                                columns=["region_type", "op_type"])
        by = trace.groupby(["region_type", "op_type"]).size().unstack(fill_value=0)
        for col in ("load", "store"):
            if col not in by.columns:
                by[col] = 0
        if "weight" in by.index:
            if by.loc["weight", "load"] == 0:
                self.fail("regions", "weight has no load samples — the region "
                                     "the study is about was not observed")
            if by.loc["weight", "store"] > 0:
                self.fail("regions", f"weight has {by.loc['weight','store']:,} "
                                     "stores, impossible for a read-only mapping")
        if "kv_cache" in by.index and by.loc["kv_cache"].sum() == 0:
            self.fail("regions", "kv_cache region registered but never accessed")

    def check_sampling(self, meta: dict) -> None:
        c = meta.get("collector") or {}
        before = c.get("perf_event_max_sample_rate_before")
        after = c.get("perf_event_max_sample_rate_after")
        achieved = c.get("achieved_sample_rate_per_cpu")
        if not c.get("sample_period"):
            self.fail("sampling", "sample_period not recorded — the sample "
                                  "count is uninterpretable")
        if not c.get("event_strings"):
            self.fail("sampling", "event_strings not recorded")
        if achieved and after:
            frac = achieved / after
            if frac >= 0.5:
                self.note(f"sampling: achieved {achieved:,.0f}/s/CPU is "
                          f"{frac:.0%} of the {after:,} cap — close to clipping")
        if before and after and after < before:
            self.note(f"sampling: kernel lowered the cap {before:,} -> {after:,}"
                      + (f" (achieved {achieved:,.0f}/s/CPU)" if achieved else ""))

    def check_granularity(self, meta: dict) -> None:
        g = (meta.get("collector") or {}).get("region_granularity_bytes")
        if not g:
            self.fail("granularity", "region_granularity_bytes not recorded — "
                                     "retention cannot be traced to a granule")

    def check_kv_lifetime_bounded(self, meta: dict) -> None:
        """
        No derived duration may exceed the run that produced it.

        A block whose write was never sampled has no lifetime.  Subtracting an
        imputed write time of 0 from a monotonic timestamp yielded the machine's
        uptime instead -- 165,781 s of "lifetime" inside a 73 s run, on 99% of
        blocks across a whole sweep, with every other gate still green.  The
        bound is what makes that arithmetically impossible to ship again.
        """
        path = self.dir / "kv_block_lifecycle.parquet"
        if not path.exists():
            return
        lc = pd.read_parquet(path)
        if lc.empty or "lifetime_ns" not in lc.columns:
            return

        c = meta.get("collector") or {}
        g = meta.get("generation") or {}
        budget_s = c.get("collection_duration_s") or g.get("total_duration_s")
        if not budget_s:
            return
        # Generous: the collector brackets the whole process, so allow an order
        # of magnitude over it.  The defect overshot by a factor of 2,000.
        limit_ns = float(budget_s) * 1e9 * 10

        lifetimes = pd.to_numeric(lc["lifetime_ns"], errors="coerce")
        over = lifetimes > limit_ns
        if over.any():
            worst = lifetimes[over].max() / 1e9
            self.fail("kv_lifetime",
                      f"{int(over.sum()):,} of {len(lc):,} KV blocks report a "
                      f"lifetime longer than 10x the {float(budget_s):,.1f} s run "
                      f"(worst {worst:,.1f} s) — a lifetime that exceeds its run "
                      f"is a clock reading, not a duration")

        if "write_observed" in lc.columns:
            observed = lc["write_observed"].fillna(False).astype(bool)
            # A null lifetime must mean the write was not observed, and an
            # observed write must yield a lifetime; otherwise the two columns
            # disagree about what was measured.
            if (observed & lifetimes.isna()).any():
                self.fail("kv_lifetime",
                          "block(s) have write_observed=True but a null lifetime")
            if ((~observed) & lifetimes.notna()).any():
                self.fail("kv_lifetime",
                          "block(s) have a lifetime without an observed write")
            frac = observed.mean()
            if frac < 0.5:
                self.note(f"kv_lifetime: only {frac:.1%} of {len(lc):,} KV blocks "
                          f"had a store sampled, so block lifetime and "
                          f"kv_cache write-once rest on that minority")

    def check_token_count(self, meta: dict) -> None:
        """
        A run that stopped early is not the run that was planned.

        `output_tokens` is a *ceiling*: generation ends at the model's
        end-of-turn token, whichever comes first.  Asking for 1977 tokens on
        Llama-3.2-1B produced 653, so the KV cache reached 33% occupancy
        instead of 98% -- and every other field in the run said success.  When a
        run asked for the end-of-turn token to be suppressed, falling short
        means a stop token nobody named, and the shortfall has to be loud.
        """
        g = meta.get("generation") or {}
        requested = g.get("tokens_requested_per_prompt") or g.get("output_tokens")
        prompts = g.get("prompts") or 0
        achieved = g.get("total_output_tokens")
        if not requested or not prompts or achieved is None:
            return

        expected = requested * prompts
        if achieved >= expected:
            return
        shortfall = 1.0 - achieved / expected
        detail = (f"generated {achieved:,} of {expected:,} requested tokens "
                  f"({shortfall:.0%} short)")
        if g.get("ignore_eos"):
            self.fail("tokens", detail + " despite ignore_eos — a stop token "
                                         "was not suppressed, so the token "
                                         "count was not the variable it was "
                                         "meant to be")
        else:
            self.note("tokens: " + detail + " (end-of-turn token fired; "
                      "output_tokens is a ceiling without ignore_eos)")

    # ── driver ───────────────────────────────────────────────────────────────

    def run(self) -> bool:
        if not self.check_artifacts():
            return False
        meta = json.loads((self.dir / "metadata.json").read_text())
        self.check_attribution(meta)
        self.check_verdict_agrees(meta)
        self.check_metrics_finite()
        self.check_region_expectations()
        self.check_sampling(meta)
        self.check_granularity(meta)
        self.check_kv_lifetime_bounded(meta)
        self.check_token_count(meta)
        return not self.failures



def audit_run(run_dir) -> Audit:
    """Audit one exported run directory and return the completed Audit."""
    audit = Audit(pathlib.Path(run_dir))
    audit.run()
    return audit

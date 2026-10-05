# Imputed Values in Sampled Measurements

*Lessons and future work from the v1.5 sweep. Measured on 42 runs (5 GGUF models
under llama.cpp on CPU, 3 HF models on GPU), Intel Core i9-11980HK, Linux 7.0,
`perf mem` / PEBS, 2026-10-04. Written for whoever maintains this instrument next,
and for the Limitations section.*

---

## 1. Summary

The v1.5 sweep completed 53 runs with every gate green: per-run audit PASS on all
42 exported runs, `clean` validity on all 42, complete `metrics.csv`, no run
clipped by a falling sample-rate cap. Three defects were nonetheless present, and
all three are **the same defect**: an observation that did not occur was published
as a value rather than as an absence.

| Defect | Scope | Published value | Correct value |
|---|---|---|---|
| KV block lifetime from an unsampled write | 42,260 of 42,670 blocks (99.0%) | 162,196 s inside a 73 s run | null |
| Retention of a single-sample region | 70–79% of regions at ctx 32768 | 0.00 s | null |
| Suitability with unmeasurable retention | every such region type | `low_mrm` | `unknown` |

None was visible in any summary the pipeline printed. Each was found by looking at
a number and asking whether it was physically possible.

The common mechanism is that **`perf mem` does not observe accesses, it samples
them** — loads on one period, stores on another 500× sparser. Every derivation
written before that asymmetry existed treats the trace as a record of what
happened. It is a record of what was *caught*.

---

## 2. What the defects cost

### 2.1 The arithmetic failure

`lifetime_ns = end_ns - write_ns`. With no store sampled for a block, `write_ns`
was coerced to 0, and subtracting 0 from a **monotonic** timestamp returns the
clock reading — the machine's uptime, not a duration. The published mean lifetime
was 45 hours for runs lasting 73 seconds. Over the 410 blocks (1.0%) that did
have a sampled write, the measured lifetime is mean 15.7 s, median 0.0 s,
maximum 216.4 s across the dataset — every value inside the run that produced
it. The median of 0.0 s is itself informative: most blocks whose write *was*
caught were not read again within the sample.

### 2.2 The statistical failure

Retention is `last access - first access` of a region, so a region sampled *once*
has no interval — both ends are the same observation. That was reported as `0.0`.
Because samples-per-region falls as context grows (the budget is fixed at 10M
samples per run, while the number of regions grows), the imputed zeros took over
the median exactly where the context trend is read off:

| Region | ctx | sampled once | p50 as published | p50 over observable regions |
|---|---|---|---|---|
| `kv_cache` | 2048 | 42.8% | 10.04 s | **66.95 s** |
| `kv_cache` | 8192 | 69.1% | 0.00 s | **59.73 s** |
| `kv_cache` | 32768 | 79.3% | 0.00 s | **44.91 s** |
| `weight` | 2048 | 27.6% | 49.46 s | **80.45 s** |
| `weight` | 32768 | 70.4% | 0.00 s | **37.30 s** |
| `runtime` | 32768 | 74.9% | 0.00 s | **36.74 s** |

Retention *does* fall with context — by 1.5–2.2× over a 16× increase — but not to
zero, and every absolute value was understated several-fold. **The published
decline was mostly a sampling-density artifact, strongest precisely where the
scientific claim was being made.**

### 2.3 The direction of the error matters

Both defects pushed the same way: toward *shorter* retention and *lower*
suitability. A measurement instrument whose errors are unbiased is merely noisy;
one whose errors are systematic in the direction of the null hypothesis will
quietly argue against its own subject. MRM research exists to find data with long
retention and write-once behaviour. These defects hid exactly that.

---

## 3. Why several hours of pre-run testing did not catch it

The suite was 967 tests across three tiers at the time of the sweep. Four
independent reasons it passed.

### 3.1 Every fixture was dense; production was sparse

The Tier-3 validity fixtures — the tier CLAUDE.md calls the most important in the
suite — build regions with `for k in range(5)`: five samples per region, every
region, by construction. Production had **one** sample for 70–79% of regions at
ctx 32768. The suite never executed the regime the sweep ran in.

### 3.2 The tests covered the rare direction of an asymmetry

The two tests that did touch a single-sample region tested its mirror image:

```python
def test_write_only_region_has_zero_retention():      # a store, no load
def test_lifetime_ns_zero_for_write_only_block():     # a store, no load
```

Production is the opposite — **a load with no store** — because stores are sampled
500× sparser. "Write, no read" was 0.6% of blocks; "read, no write" was 99.0%.

The asymmetry is *newer than the tests*. `DEFAULT_STORE_MULTIPLIER` arrived in
commit `b22b5e4`, **58 commits after** the retention and KV-lifecycle tests were
written. Those tests encode an implicit model in which every access appears in the
trace. That model was true when they were written; nothing revisited the
derivations that depended on it.

### 3.3 A test asserted the defect as correct

`test_write_only_region_has_zero_retention` did not overlook the single-sample
case — it **required** `retention_s == 0.0`. The suite was actively defending the
defect, and any attempt to fix it would have surfaced as a regression.

### 3.4 The post-run auditor asked the wrong question

`run_audit.py` was built specifically so that finding these would not depend on
someone noticing. Its check was:

```python
def check_metrics_finite(self):
    """A published metric must be a number."""
    if not np.isfinite(row[col]): self.fail(...)
```

`np.isfinite(0.0)` is `True`. The gate asked *is this a number?* when the question
is *is this a measurement?* — so it inspected the fabricated zeros and certified
them.

### 3.5 One path had no coverage at all

All four pre-run tests of `experiment_runner._build_command` opened identically:

```python
spec = next(s for s in runner.build_matrix(cpu_only=True) if s.path_type == "cpu")
```

The GPU branch was never executed by any test. That is why `--warmup-only` was
missing there (so every GPU rep 0 collected a full run instead of warming the page
cache, leaving each GPU cell at one data rep), and why `--num-prompts` had been
missing there earlier. Both were found by *running* the path, not by testing it.

### 3.6 The evidence was in the staging output and was misread

The pre-sweep staging run printed, and it was quoted back as evidence of
soundness:

```
kv_cache  1 range | 3,223 rows | precision 100.00% | recall 100.00% | 3,206 loads + 17 stores
```

**17 stores.** The asymmetry was in the output being read. It was taken as "both
operation types present" rather than "17 stores across 638 blocks cannot support a
write-once ratio or a block lifetime". The staging checklist had been assembled
from the *previous* defect class — region attribution — so precision and recall
were verified exhaustively while no derived quantity was checked for physical
plausibility. A 46-hour lifetime inside a 73-second run was one glance away.

---

## 4. Lessons

1. **In a sampled measurement, absence is a value of its own.** `None` is not a
   missing `0`. Any `fillna`, `or 0`, or `.get(k, 0)` on a quantity derived from
   samples is a claim that an unobserved event did not occur.

2. **A fixture must match the production regime, not the author's mental model.**
   Sampling density is part of the regime. A test suite built entirely on dense
   fixtures validates an instrument that was never run.

3. **Test the common direction of an asymmetry.** Where two populations are
   sampled at different rates, the interesting case is the sparse one, and it is
   usually the one the author does not picture.

4. **Changing a sampling parameter invalidates downstream derivations.** When
   `--store-multiplier` introduced a 500× asymmetry, every statistic that divides
   loads by stores silently changed meaning. A sampling change needs an audit of
   its consumers, not just its own tests.

5. **Publish the denominator with the statistic.** `retention_p50_s = 44.91` over
   67 regions and over 10,000 regions are different claims. `metrics.csv` now
   carries `n_regions_observable` and `retention_support_frac`; the lifecycle file
   carries `write_observed`.

6. **Gates must check measurement validity, not type validity.** "Is it finite"
   is a type check. "Could this value have arisen from this run" is a measurement
   check. Prefer bounds derived from physical quantities the run already records:
   *no derived duration may exceed its own collection duration* is arithmetic, and
   holds at any sampling density.

7. **A test that encodes a defect is worse than a missing test.** It converts the
   fix into a regression. When behaviour is corrected, the test asserting the old
   behaviour is itself evidence and should be rewritten with the reason attached,
   not deleted silently.

8. **Every branch of a multi-path system needs coverage.** A helper with a
   `if cpu: ... else: gpu: ...` shape and tests that only ever construct CPU
   inputs has one tested branch and one hypothesis.

9. **A verification checklist assembled from the last defect finds the last
   defect.** After the attribution work, staging validated attribution. The useful
   question is not "is the thing I last fixed still fixed" but "for each published
   column, what would a wrong value look like, and would I see it".

10. **Validate along every dimension the claims are read along.** Every check
    compared repetitions within a cell, so nothing examined the *context*
    dimension — the one the scaling results are read along. A 36–44× step in
    `runtime` region count between adjacent contexts was reproducible across
    both reps of each cell, so the reproducibility check saw two stable cells
    and passed it. Stability within a cell is not comparability across cells.

11. **Verify that the independent variable is actually controlled.** The
    occupancy sweep's whole premise is that `--tokens` sets how much KV cache
    gets written. It does not: generation also stops at the model's end-of-turn
    token, so `--tokens` is a *ceiling*. Asked for 1977, Llama-3.2-1B produced
    653 and reached 33% occupancy instead of 98% — and the run passed its audit,
    reported `clean`, and wrote a complete `metrics.csv`. Thirty runs of that
    would have been three repetitions of one experiment. A sweep should assert
    that each level achieved the level it asked for, before the sweep rather
    than after.

12. **A flag the child cannot see was never applied.** Both collection scripts
    re-invoke themselves under `perf` with `--headless`, and the child re-parses
    argv from scratch. A flag missing from that command is ignored while the
    parent still records it in metadata as though it applied. This has now
    happened twice — `--num-prompts` on the GPU path (one prompt requested,
    seven generated) and `--ignore-eos` on the CPU path (1977 requested, 653
    generated, `ignore_eos: true` in the metadata of both). The fix is not to
    remember: it is a test over the declared set of workload flags, which found
    a third candidate the moment it was written.

13. **Separating collection from derivation paid for itself.** All three defects
    were in *derivations* over data the runs already held, so all three were
    corrected in place — `--rederive-kv`, `--rederive-metrics`, `--reclassify` —
    with no re-collection of 8.6 GB. Each re-derivation runs the same code as the
    pipeline, and equivalence tests pin them equal, so a corrected dataset is not
    a patched dataset. This is worth protecting as the architecture grows.

---

## 4a. A cross-context discontinuity in `runtime` — real, not an artifact

Found after the re-derivation; the sampling hypothesis has since been tested and
ruled out.

`runtime` region count steps discontinuously between adjacent context settings
of two models, while every other step in the matrix is under 2.4×:

| Model | step | regions per run | ratio |
|---|---|---|---|
| `llama-3.2-3b` | ctx 8192 → 32768 | 18,132 → 416 | **43.6×** |
| `phi-3.5-mini` | ctx 2048 → 8192 | 18,859 → 520 | **36.3×** |

What is established:

- **It is in the data, not the labeller.** The distinct 64 KB granules of the
  sampled *addresses* equal the assigned `region_id` count exactly (18,965 and
  529 for the phi pair), so region assignment is faithful.
- **It is reproducible.** Both reps of each cell agree (529 / 511 and 416 / 424),
  which is why the reproducibility check passed it — it compares reps, not cells.
- **The granule is uniform.** All 30 CPU runs recorded
  `region_granularity_bytes = 65536`.
- **The sample share is unchanged.** `runtime` is ~98% of rows in both members of
  each pair; only its *address footprint* collapses — 1,185 MB → 33 MB for phi.
- **The hot set is unchanged.** The top 100 addresses hold 16.4% of rows in
  both. What disappears is the cold tail: ~1.2 GB of anon memory touched at the
  lower context and not at the higher one.
- **It does not track the sampling period cleanly.** phi collapses at period
  8,660 while llama-3.2-3b is still smooth at 6,950 and collapses at 20,900.

### The control run

The sampling hypothesis predicts that re-running the collapsed cell at the
*neighbour's* sampling period restores the footprint. It does not:

| Run | period | samples | `runtime` regions | `runtime` MB |
|---|---|---|---|---|
| phi ctx 2048 | 4,330 | 7,221,727 | 18,964 | **1,185** |
| phi ctx 8192 | 8,660 | 4,058,263 | 529 | **33** |
| phi ctx 8192 (control) | **4,330** | **8,406,861** | **706** | **44** |

At matched sampling density — and with *more* samples than the ctx 2048 run — the
footprint stays collapsed. In the same control, `weight` and `kv_cache` region
counts came out **higher** than at ctx 2048 (15,671 and 4,442 against 15,249 and
3,715), exactly as more samples should produce. So the instrument was discovering
more of every other region while `runtime` stayed at 4% of its ctx-2048 size.

**Sampling aliasing and sampling density are both ruled out.** The step is in the
workload.

### The cause: `runtime` is a catch-all, and its contents are not stable

Diffing the mappings each run already records (`generation_results.json →
address_regions`) names it immediately. One `anon_post_load` mapping accounts for
the whole step:

| Model | ctx | `kv_allocation` | `anon_post_load` | `runtime` regions |
|---|---|---|---|---|
| phi-3.5-mini | 2048 | 768 MB | **2,269 MB** | 18,964 |
| phi-3.5-mini | 8192 | 3,072 MB | **69 MB** | **529** |
| llama-3.2-3b | 2048 | 224 MB | 1,832 MB | 20,446 |
| llama-3.2-3b | 8192 | 896 MB | 2,926 MB | 18,010 |
| llama-3.2-3b | 32768 | 3,584 MB | **252 MB** | **407** |
| llama-3.2-1b | 2048 | 64 MB | 1,070 MB | 8,280 |
| llama-3.2-1b | 8192 | 256 MB | 1,303 MB | 7,599 |
| llama-3.2-1b | 32768 | 1,024 MB | 3,906 MB | 6,423 |

`runtime` collapses exactly where `anon_post_load` collapses, and llama-3.2-1b —
whose `anon_post_load` never collapses — shows no discontinuity at all. That is
the whole effect.

So the measurement was never wrong. **`runtime` is not a region type, it is
"everything not otherwise identified"**, and its largest constituent is a big
anonymous allocation whose size moves non-monotonically with context as the
engine trades it off against the KV buffer. A working set computed over a
category whose membership changes is not a quantity that can be compared across
the contexts it changes between.

This is the imputation defect in a third guise: not a fabricated *number* this
time, but a fabricated *category*. `unknown` at least announces itself;
`runtime` looks like a finding.

**Consequence for the dataset:** `runtime` working set and region count must not
be compared across contexts, and the reason is now specific — the category's
membership changes, not the memory's behaviour. Report `runtime` per context or
not at all.
`weight` and `kv_cache` are unaffected — their steps are 1.1–2.3×.
`check_context_continuity` in `results_validation.py` reports it, with a 5×
threshold and a 20-region floor so small-number ratios are not mistaken for
steps.

**The fix is to stop lumping it in.** `anon_post_load` is already recorded as a
distinct source, so the labeller can give it its own region type (the engine's
compute/graph pool) instead of folding it into `runtime`. That turns a moving
catch-all into two stable categories and makes both comparable. It changes
labelling, so it needs a re-label of the existing runs — cheap, since
`region_type` is derived and `trace.parquet` holds the addresses — and it should
land before the next sweep rather than after.

---

## 4b. The occupancy sweep's independent variable was not controlled

Caught by a single smoke run before the sweep, which is the only reason it is a
paragraph here rather than 10 hours of wasted collection.

`--tokens` is a ceiling, not a count. Generation also ends at the model's
end-of-turn token, and for Llama-3.2-1B on the standard prompt that happens at
**653 tokens**:

| requested | achieved | KV occupancy | finish reason |
|---|---|---|---|
| 1977 | **653** | **33%** | `stop` |
| 1977 (EOS suppressed) | **1977** | **98%** | `length` |

Suppressing it needs a `logit_bias` of -100 on every end-of-generation token the
model exposes — `token_eos()` and `token_eot()`, the same id (128009) on Llama 3
but not necessarily elsewhere. `-100` rather than `-inf`: logits sit in roughly
[-20, 20], so -100 removes the token without risking a NaN.

Two mechanisms now stand behind it, because naming stop tokens correctly for
every model family is not something to assume:

- `Audit.check_token_count()` **fails** a run that falls short of its requested
  tokens when `ignore_eos` was set — a shortfall then means a stop token nobody
  named. Without `ignore_eos` it is a note, since the ceiling is expected.
- `metadata.json → generation` records `ignore_eos`,
  `tokens_requested_per_prompt` and `reached_token_limit` alongside
  `total_output_tokens`, so requested and achieved can never be confused again.

The first attempt at the fix *still* produced 653 tokens, because
`--ignore-eos` was not forwarded to the `--headless` child that does the
generating — see lesson 12. The audit caught that too, which is what it is for.

---

## 5. Future work

### 5.1 A sparse-regime fixture for the validity tier — *highest value, ~1 hour*

Build one Tier-3 fixture at the production regime — load/store period ratio 500:1,
~75% of regions with a single sample — and push it through the whole analyser.
Assert that every output is either a value supported by ≥2 observations or
explicitly null. **This single fixture would have caught all three defects before
the sweep.** It is the direct remedy for §3.1 and §3.2, and until it exists the
suite still validates a regime the instrument does not run in.

### 5.2 Audit the remaining imputation sites

A grep for `fillna`, `or 0` and `.get(..., 0)` across `analyser/`, `labeller/` and
`reporter/` returns 10 code sites. Two are now guarded and one is dead code (the
`fillna(0.0)` at `read_freq.py:41` is unreachable for any region that has an
access). The rest are unreviewed. Specifically:

- `labeller.py:291` — `ts_ns = row.get("timestamp_ns", 0)`. A row with no
  timestamp becomes an access at monotonic time 0, which would place it ~46 h
  before every other access in the run and inflate any span containing it. Benign
  for `perf mem` (always present) and for memray (all zero, so uniformly
  meaningless), but it is the same coercion that produced §2.1.
- `read_freq.py:29` — `read_fraction` imputed to 0.0 for a region with no accesses.
- `retention.py:88` — `last_read_ns` falls back to `first_write_ns` for a region
  with stores and no sampled loads. Defensible under the stated definition
  (retention runs to the last *read*), but it means a write-heavy region reports
  retention 0 when the accesses observed in fact span a real interval. Worth
  recording the observed access span separately from the retention definition.

### 5.3 Period-weighted estimators for ratio metrics — *largest scientific stakes*

`reads_per_write` divides a densely sampled numerator by a sparsely sampled
denominator, so the published values are inflated by the period ratio. Every run
records both periods in `collector.event_strings`, and the ratio is exactly 500 on
all 42 runs, so the correction is computable from data already on disk. Weighting
each operation by its own period over the CPU analysis set:

| Region | published reads/write | period-weighted estimate |
|---|---|---|
| `kv_cache` | 250–312 | **0.50** |
| `activation` | 13 | **0.03** |
| `runtime` | 8 | **0.02** |
| `weight` | ∞ | ∞ (genuinely read-only) |

**This inverts the headline conclusion for `kv_cache`.** At 312 reads per write it
looks close to ideal for MRM — write-once, read-many. At 0.50 it is written about
twice as often as it is read, which is the *least* MRM-suitable profile. The same
applies to `write_once_ratio`: `kv_cache` reported 1.000 with sd 0.000 across 30
runs, which is the signature of at-most-one sampled store per address rather than
of demonstrated write-once behaviour.

Two caveats before this estimate is published. First, PEBS load sampling is
latency-filtered (`ldlat=3`), so the load population is not a uniform random
sample of loads — it is biased toward slower ones. At `ldlat=3` nearly all loads
qualify, but "nearly" needs quantifying before the estimator is called unbiased.
Second, the estimator should be validated against a run collected with a store
multiplier of 1, where both populations are sampled alike. That validation run is
cheap and is the prerequisite for any write-once or reads-per-write claim in the
paper.

Until then: report `write_once_ratio` and `reads_per_write` as **unsupported for
`kv_cache`**, and do not draw an MRM-suitability conclusion for it from this
dataset.

### 5.4 Choose the region granule from target support, not convention

Retention is reported at a fixed 64 KB granule, which yields a median of 1 sample
per region for several (model, ctx) cells and leaves percentiles resting on 13% of
regions at ctx 32768. Granule size and sample budget trade off directly against
support. The granularity sweep already exists (`analyser/granularity.py`); it
should be used to *select* the granule per run such that a stated fraction of
regions reach ≥2 samples, with the chosen granule and achieved support recorded as
provenance. A metric whose spatial resolution is chosen by convention and whose
support then varies 6× across the matrix is not comparable across that matrix.

### 5.5 Raise store sampling for a KV-focused run

At the current multiplier, `kv_cache` receives ~18 store samples per run. A run
series with `--store-multiplier` at 1–10 would make KV write behaviour and block
lifetime directly measurable rather than an estimate over 1% of blocks. Cost is
trace volume, which is why the asymmetry exists; the point is that it is a
*choice*, and the current choice makes two primary metrics unmeasurable for the
region the project most cares about.

### 5.6 Ground or drop `activation`

`activation` remains symbol-inferred rather than address-grounded, and its share
varies 28–77× across identical repetitions. A region that varies across identical
reps is mis-attributed, not noisy. Either ground it in recorded address ranges (as
`weight` and `kv_cache` now are) or remove the rule; carrying an unstable region
type invites its accidental use in a cross-run comparison.

### 5.7 Give `anon_post_load` its own region type

See §4a, which is now closed: the `runtime` discontinuity is a single large
`anon_post_load` mapping that the catch-all absorbs, and whose size the engine
trades off against the KV buffer as context grows. Promoting it to its own
region type — the engine's compute/graph pool — splits a moving category into
two stable ones and makes `runtime` comparable for the first time. It is a
labelling change, so it needs a re-label of the existing runs and should land
before the next sweep.

### 5.8 Coverage gate on both execution paths

Add a check that every branch of `_build_command` is exercised, and more generally
that the CPU and GPU paths are both constructed in tests. §3.5 was a one-line
omission with a sweep-wide consequence, and it recurred twice.

---

## 6. What is now mechanically prevented

For the record, so future work is not duplicated:

- A derived duration exceeding 10× its own collection duration **fails** the
  per-run audit (`Audit.check_kv_lifetime_bounded`). Regime-independent.
- `write_observed` and `lifetime_ns` must agree; disagreement **fails**.
- `retention_s` is null below `MIN_SAMPLES_FOR_RETENTION` (2); percentiles are
  computed over observable regions only; support is published.
- `classify_suitability` returns `unknown`, not `low_mrm`, when retention is
  unmeasurable.
- The runner passes `--warmup-only` on **both** paths, and
  `test_every_cell_has_two_data_reps` fails any cell that cannot be checked for
  reproducibility.
- `results_validation.py` reads recorded support rather than re-estimating it, and
  names any cell whose percentiles rest on under half its regions.

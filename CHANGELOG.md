# Changelog

All notable changes to `mrm-trace` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased] — v1.5 (current development)

> Three defects in this release share one cause — an unobserved event published as a
> value — and passed 967 tests because every fixture was densely sampled while the sweep
> was not. The post-mortem, the lessons, and the remaining work (including a
> period-weighted `reads_per_write` that **inverts** the `kv_cache` conclusion) are in
> [05 - Imputed Values in Sampled Measurements.md](context-files/05%20-%20Imputed%20Values%20in%20Sampled%20Measurements.md).

### Fixed — a region sampled once reported retention 0.0, understating the primary metric

Retention is `last access - first access` of a region, so a region sampled **once** has
no interval: both ends are the same observation. That was published as `0.0`, and because
samples-per-region falls as context grows, the imputed zeros took over the median exactly
where the context trend is read off. Pooled over the CPU analysis set:

| Region | ctx | regions sampled once | p50 as published | p50 over observable regions |
|---|---|---|---|---|
| `kv_cache` | 2048 | 42.8% | 10.04 s | **66.95 s** |
| `kv_cache` | 8192 | 69.1% | 0.00 s | **59.73 s** |
| `kv_cache` | 32768 | 79.3% | 0.00 s | **44.91 s** |
| `weight` | 2048 | 27.6% | 49.46 s | **80.45 s** |
| `weight` | 32768 | 70.4% | 0.00 s | **37.30 s** |
| `runtime` | 32768 | 74.9% | 0.00 s | **36.74 s** |

On the worst-supported single run (`llama-3.1-8b ctx32768 rep1`), `kv_cache` p50 moved
from 0.00 s to 102.66 s and `weight` from 0.00 s to 79.66 s.

Retention **does** fall with context — by roughly 1.5–2.2× over a 16× increase — but not
to zero, and the absolute values were understated several-fold. The published decline was
largely a sampling-density artifact: the per-run budget is fixed at 10M samples, so a
longer high-context run spreads the same samples over more regions.

This is the same defect class as the KV lifetime above: an absent measurement published
as a number. Four changes:

- `compute_retention()` adds `n_samples` and `retention_observable`; `retention_ns` and
  `retention_s` are null below `MIN_SAMPLES_FOR_RETENTION` (2), and every percentile is
  computed over observable regions only.
- The summary and `metrics.csv` gain `n_regions_observable` and
  `retention_support_frac`, because a p50 over 1 region and a p50 over 10,000 are
  different claims and the table could not previously tell them apart.
- `classify_suitability()` returns `unknown` when retention is unmeasurable. Previously
  every threshold comparison against NaN was False, so an unmeasured region type was
  classified `low_mrm` — the worst verdict, on no evidence.
- `results_validation.py` reads the recorded `retention_support_frac` rather than
  re-estimating it from region-map access counts, and names any cell whose percentiles
  rest on under half its regions.

A measured zero is still a measurement: two samples at one instant give
`retention_s = 0.0` and `retention_observable = True`. One sample gives null.

`metrics` schema → **1.1** (additive support columns).

Metrics are a derivation over rows each run already holds, so no trace needed
re-collecting: `experiment_runner.py --rederive-metrics` recomputes every `metrics.csv`
from its own `trace.parquet` through the same `compute_all()` the pipeline runs.
`test_rederived_metrics_equal_a_live_analysis` pins the two equal.

### Fixed — an unsampled KV write became a 46-hour block lifetime

A KV block's lifetime is `end_ns - write_ns`. Stores are sampled on their own, far
sparser period than loads (`period=2,000,000` against `4,000`), so the common case for a
KV block is many sampled loads and **no sampled store at all**. `lifecycle_rows()`
coerced that absent `write_ns` to 0 — and subtracting 0 from a monotonic timestamp does
not give a short lifetime, it gives the raw clock reading:

| | measured |
|---|---|
| blocks with no sampled store | **42,260 of 42,670 (99.0%)** |
| reported mean lifetime | 162,196 s (≈ 45 h) |
| actual run length | 73 s |
| lifetime after the fix (the 18 blocks that have one) | mean 19.8 s, max 59.2 s |

The dataset-level EDA published those values as `lifetime_s`, and every other gate stayed
green: per-run audit PASS, verdict `clean`, `metrics.csv` complete. This is the failure
mode the project exists to prevent — an absent measurement silently becoming a wrong
number — and it violated the project's own rule that unknowns are never coerced.

Three mechanisms changed, not one:

- `KVBlockTracker.lifecycle_rows()` reports `write_ns = None`, `lifetime_ns = None` and a
  new `write_observed` column. The absence is recorded, never imputed.
- `Audit.check_kv_lifetime_bounded()` **fails** any run whose block lifetime exceeds 10×
  its own collection duration, and fails any disagreement between `write_observed` and
  `lifetime_ns`. A derived duration longer than the run that produced it is arithmetically
  a clock reading, so the bound makes this class of defect unshippable rather than
  dependent on someone noticing.
- `results_exploratory_data_analysis.py` aggregates lifetime only over blocks whose write
  was sampled, states what fraction that is, and reports read counts separately over all
  blocks.

`kv_block_lifecycle` schema → **1.1** (additive `write_observed`; `lifetime_ns` nullable).

Block lifetime is a *derivation* over rows each run already holds, so no trace needed
re-collecting: `experiment_runner.py --rederive-kv` rebuilds every
`kv_block_lifecycle.parquet` from its own `trace.parquet`, mirroring `--reclassify`.
`test_rederivation_equals_live_tracking` pins the re-derived file to what the live
pipeline produces from the same rows, so a re-derived dataset is not a patched one.

### Known limitation — kv_cache write-once and reads-per-write are unsupported

The same sampling asymmetry undermines two of the four primary metrics, for `kv_cache`
only. Measured on one run: `kv_cache` received **18 store samples** against 2,918 loads,
over 18 distinct addresses, each written exactly once — so `write_once_ratio = 1.000`
with **sd 0.000 across all 30 runs**. That constancy is the signature of at-most-one
sampled store per address, not of demonstrated write-once behaviour. `reads_per_write`
(312 ± 200) divides a densely sampled numerator by a sparsely sampled denominator.

Well-sampled region types are unaffected: `runtime` has 1,140,498 stores and a real
distribution (45.1% write-once, max 6,692 writes per address); `weight` has 0 stores
because the GGUF is mapped read-only and faulted in by the kernel, which is a genuine
finding rather than missing data.

Report `write_once_ratio` and `reads_per_write` as unsupported for `kv_cache`, or
re-collect with a lower `--store-multiplier`. The EDA now emits this caveat itself,
derived from the `write_observed` fraction rather than assumed.

### Fixed — GPU warmups collected a full run, leaving every GPU cell at n=1

The matrix flagged rep 0 as a warmup on both paths, but only the CPU branch passed
`--warmup-only`, and `demo_qwen_hf.py` did not define the flag. So every GPU rep 0
collected a complete run: each GPU cell held one cold-cache run and one warm run, the
analysis set (`rep > 0`) kept only the warm one, and the GPU path sat at **one data rep
per cell** — with no cross-rep reproducibility check possible on that path at all.

The two reps are close in trace terms (samples within 1.00–1.04×, tok/s within 0.5), but
wall-clock differed by up to 35% (46.7 s against 34.5 s on mistral-7b), which retention
inherits directly, being last-minus-first access. So rep 0 stays out of the analysis set
rather than being promoted to data.

`demo_qwen_hf.py` now has `run_warmup_only()`, the runner passes `--warmup-only` on both
branches, and `GPU_REPS` is `[0, 1, 2]` so the GPU path gets a bare warmup and two data
reps like the CPU path. `test_every_cell_has_two_data_reps` fails on any cell that cannot
be checked for reproducibility.

### Fixed — matrix completeness reported ten lines of expected absences

The check took the union of every rep seen anywhere as the expected set, so it reported
"missing rep 0" for all 15 CPU cells (rep 0 is a bare warmup that writes nothing by
design) and "missing rep 2" for all GPU cells (that path ran two reps). The real signal —
a cell sitting at one data rep — was buried in the noise. Expected reps are now derived
per path over data reps only, and a cell with fewer than two is named as uncheckable.

### Fixed — GPU power draw was a driver artifact, published as machine context

`nvidia-smi --query-gpu=power.draw` reports a physically impossible value on mobile
Ampere under hybrid graphics. Measured on an RTX 3080 Laptop:

| | |
|---|---|
| `power.draw` | **752.67 W** |
| `enforced.power.limit` | 115.00 W |

The reading was copied verbatim into `metadata.json → gpu.power_draw_w`. Layer 3 exists
so trace values can be *independently audited*; a 752 W figure on a 115 W part
discredits the section it appears in.

`collect_machine_context()` now queries the limit alongside the draw and refuses to
publish a draw more than `_POWER_DRAW_TOLERANCE` (10%) above it. Instead it records
`gpu_power_limit_w` and `gpu_power_draw_status`, so a reader can tell a *missing* sensor
from a *lying* one. The 10% headroom admits genuine transients above the sustained limit
without admitting the artifact.

The two demo scripts had duplicated this field assembly; it now lives in
`telemetry.system_monitor.gpu_section()`, the fourth place where script-level
duplication had drifted from the package.

### Fixed — an absent GPU weight attribution could not be distinguished from a failure

`extract_pytorch_regions()` correctly skips non-CPU tensors: a CUDA `data_ptr()` is a
device address that no CPU PMU can sample. So `pytorch_regions.json` with 0 entries is
the *right* answer for a GPU-resident model — and byte-identical to the extractor having
silently failed. The run output said only `Weight regions recorded: 0 tensors across 0 MB`.

Runs now write `parameter_devices.json`, a census of where the parameters actually lived:

```json
{"cuda:0": {"tensors": 290, "bytes": 988123136}}
```

`Audit._absence_reason()` reads it and replaces the old hedge ("by design if it could
not be isolated") with a checkable statement. Host-resident parameters with no
attribution now **fail** the audit rather than being noted, because that is a pipeline
defect rather than a platform limit.

### Fixed — the dataset validator's GPU branch was unreachable

Routing GPU runs through `--collector perf-mem` (so they could not silently fall back to
memray) made `collector` stop discriminating the two paths: both read `perf_mem`. In
`check_region_coverage()` this sent every GPU model into the CPU loop — failing them for
a missing `weight` region and offering the CPU diagnosis (*"the GGUF mapping exceeded
MAX_REGION_PAGES"*) for a model that has no GGUF file — while the GPU branch below,
keyed on `collector != "perf_mem"`, could never fire.

Coverage now splits on `model.backend` (with a structural fallback to the presence of
`pytorch_regions.json` for runs written before the field existed), and a GPU model whose
census explains the absence reports **WARN**, not FAIL. The verdict is taken from the
same `Audit._absence_reason()` the per-run audit uses, so the two reports cannot disagree.

### Fixed — region attribution was invalid (breaks compatibility with earlier traces)

**Problem:** both region-attribution mechanisms were unsound, and each produced a
complete, internally consistent dataset that was wrong. Measured on one real run
(Llama-3.2-1B, ctx 2048, 28,321,245 samples):

| | |
|---|---|
| Rows labelled `weight` by the `ggml_vec_dot\|ggml_mul_mat` rule | 26,617,751 |
| Of those, inside the mmap'd GGUF file | **114** (0.0004%) |
| Rows labelled `kv_cache` | 279,186, spread over **4** granules — one in the main stack |
| `kv_cache` share across 5 repetitions of one configuration | 13.18%, 0.0010%, 0.28%, 0.28%, 0.0011% |

A PMU sample carries an instruction pointer *and* a data address. The symbol resolved
from the instruction pointer identifies the running code, not the region the data
address belongs to — `ggml_vec_dot_q6_K_q8_K` reads weights and writes activations, so
its store samples carry activation addresses. The GGUF mapping is `r--s`, so a store can
never be a weight address at all. And `llama_kv_cache_*` manages slot metadata, not the
K/V tensors, so such a sample is evidence *against* a KV address.

Because attribution is per granule, one mis-attributed sample captured the whole granule
and everything that later touched it.

**Fixes:**
- `symbol_rules.DEFAULT_RULES` no longer claims `weight` or `kv_cache`. Compute-kernel
  *stores* claim `activation` — the one inference a symbol supports soundly.
- `labeller.EV_SYMBOL_KV` removed. It ranked above `EV_MAPS_KV`, letting one symbol
  sample override the address map. Symbol evidence is now uniformly the weakest.
- `weight` comes only from the GGUF mapping; `kv_cache` only from the engine's own
  allocation (below). Where KV cannot be isolated it is **absent by design**.

Verified on a real run: 100% of `weight` and `kv_cache` rows inside their true ranges,
and 100% of samples in those ranges labelled correctly (was 0.0004% for weight).

### Added — KV cache address capture (`collector/kvtrap.c`, `collector/kv_allocation.py`)

**Problem:** the KV cache is not separable from `/proc/self/maps`. llama.cpp carves it
out of a larger ggml backend arena, so no mapping has its size — at ctx 2048 the implied
64 MiB is nowhere among the new 768 MiB / 512 MiB / 250 MiB mappings.

**Fix:** an `LD_PRELOAD` interposer records allocations, and the KV buffer is allocated in
one call of exactly the size the GGUF geometry implies:

| n_ctx | implied | captured |
|---|---|---|
| 512 | 16,777,216 | `posix_memalign` |
| 1,024 | 33,554,432 | `mmap` |
| 2,048 | 67,108,864 | `posix_memalign` |
| 4,096 | 134,217,728 | `posix_memalign` |

Both `posix_memalign` and `mmap` are intercepted (glibc routes by size). Matches must be
exact and unique; an ambiguous size yields no region rather than a guess. Built on demand
to `~/.cache/mrm-trace/`, and absent cleanly when no compiler is available.

### Fixed — load-dense sampling (weights were unobservable)

**Problem:** `perf mem record -c N` applies one period to both events, and perf samples
stores on a plain period while gating loads behind a PEBS latency threshold. The trace was
99.77% stores, and since weights are read-only the weight region was unobservable **by
construction**: 114 of 28,321,245 samples.

**Fix:** independent per-event periods and a lower threshold —
`perf record -d --weight -e cpu/mem-loads,ldlat=3,period=P/P -e cpu/mem-stores,period=500P/P`.

| | before | after |
|---|---|---|
| loads | 0.23% | **90%** |
| samples in the weight mapping | 114 | **6,992** |

`-d` is mandatory: it requests `PERF_SAMPLE_ADDR`, and without it `perf script` reports
"do not have ADDR attribute set" and emits nothing at all. `ldlat=3` rather than perf's
default 30 accounts for a 25x difference, because weight reads that hit in L1/L2 do not
clear a 30-cycle threshold. New flags `--ldlat` and `--store-multiplier`; a kernel that
rejects the syntax falls back to `perf mem record` with a loud warning.

### Fixed — kernel sampling cap is self-lowering and was unrecorded

**Problem:** `kernel.perf_event_max_sample_rate` caps samples/s/CPU, and the kernel reduces
it itself under dense sampling (`perf: interrupt took too long ... lowering ... to 1000`).
The reduction persists for the boot. Observed: 100,000 → 1,000, after which a run requesting
~103,500 samples/s delivered 13,790 — 7.5x short, with no error anywhere. Over a 135-run
sweep that silently splits the dataset into incomparable halves.

**Fixes:**
- `demo_perf_mem.py` reads the cap before and after every collection, records both in
  `metadata.json` as `collector.perf_event_max_sample_rate_before` / `_after`, and warns
  below 50,000 at start-up.
- `validity.classify_run` treats a mid-run reduction as a failure reason — such a run is at
  most `marginal`, because density higher at the start than the end biases retention.
- `experiment_runner.py` pins the cap at startup, **aborts** the sweep if it cannot be
  raised above the floor, and restores it between runs if the kernel drops it.

### Fixed — observer-memory criterion mis-normalised for an out-of-process collector

**Problem:** `observer_memory_frac` was observer RSS / target RSS, a ratio written for
memray (in-process). perf mem is a separate process whose footprint comes from CPU count and
ring-buffer size, not model size. Measured: perf peaked at 2,512 MB against a 1,478 MB
target — a ratio of 1.70 on a 31 GB machine with ~27 GB free. Nothing was evicted or
swapped, yet the rule marked **8 of 8** runs `contaminated`, and would have marked all 151.

**Fixes:**
- Out-of-process collectors are bounded against total system RAM (< 10% clean, < 25%
  marginal); memray keeps the target-RSS ratio.
- `thermal.validity_failure_reasons` is now persisted — the verdict previously discarded the
  reasons `classify_run` had already computed, so "contaminated" came with no stated cause.
- `experiment_runner.py --reclassify` re-derives every verdict from stored metadata. Validity
  is a label over numbers each run already recorded, so correcting a rule never requires
  re-collecting a trace.

### Fixed — perf mem collection overhaul

**Problem:** `perf mem record -p PID` (attach mode) failed on Ubuntu 24.04 / kernel 6.8 with
`"event cpu/mem-loads,ldlat=30/P not found"` even as root on bare-metal Intel hardware. The
Intel PEBS load-latency events require the process to be started *under* perf; retroactive
attachment does not work on this kernel configuration.

**Fix:** Switched to **spawn mode** — the demo script re-invokes itself under
`perf mem record -- python this_script.py --headless`. The headless invocation loads the model
and runs generation inside the process that perf launched, so the full process lifetime
(model load → weight copy → generation) is captured from the first instruction.

- `demo_qwen_hf.py` — `--headless` / `--results-file` internal flags; `collect_with_perf_mem`
  now calls `subprocess.run([perf, "mem", "record", "--", python, this_script, "--headless", ...])`
- `PerfMemRunner._DEFAULT_EVENTS` changed from `["cpu/mem-loads,ldlat=20/P", ...]` to `[]`
  (let perf auto-select CPU-appropriate events — ldlat syntax is not universally available)

### Fixed — perf script format + parser robustness

**Problem:** `_run_perf_script` used `-F time,pid,addr,sym,dso` which produced a 3-field
compact format (`pid time: addr`), but the parser regex expected the full
`comm pid [cpu] time: event: addr sym dso` format. Result: 0 parsed rows.

**Fixes:**
- `PerfMemRunner._run_perf_script` now uses `-F comm,pid,time,event,addr,sym,dso`
  (explicit fields, single-line-per-sample, no multi-line callchains)
- `cpu` field removed from the `-F` list — `cpu/mem-loads,ldlat=30/P` events do not set the
  CPU attribute, causing `perf script` to exit 255 with "Cannot print 'cpu' field"
- `perf_script_parser.py` fully rewritten with two regex patterns:
  - **Full regex**: matches `comm pid [cpu]? time: [period]? event: addr sym? dso?`
    (cpu, sym, dso all optional)
  - **Compact fallback**: matches `pid time: addr` (no event, no symbol)
  - Both sym and dso are now optional — perf omits them when no debug symbols are available
- Parser logs parse statistics (total lines, parsed count, skipped count) at INFO level
- Parser no longer crashes on unrecognised lines — skips with counter

### Fixed — Parquet OverflowError on kernel addresses

**Problem:** PyArrow's `from_pylist` auto-inferred `int64` for address columns. Kernel-space
addresses on x86-64 (e.g. `0xffffd3b955...`) are valid `uint64` values but exceed signed int64
max, causing `OverflowError: Python int too large to convert to C long` during export.

**Fix:**
- `writer.py` — added `TRACE_SCHEMA` (`pa.schema`) with `uint64` for `address`,
  `address_page`, `timestamp_ns`, `sample_id`; added optional `schema` parameter to
  `write_parquet()`
- `parquet_export.py` — `export_trace` now passes `schema=TRACE_SCHEMA` to enforce types

### Fixed — PMU availability probe false positives

Multiple iterations of the `_check_pmu()` probe:

| Version | What it tested | Why it failed |
|---|---|---|
| v1 | `perf stat -e cpu/mem-loads,ldlat=20/P true` | exits 0 even when event unavailable |
| v2 | `perf mem record -e cpu/mem-loads,ldlat=20/P -- true` | spawn mode passes; attach mode fails |
| v3 | `perf mem record -- true` (no explicit events) | spawn mode passes; ldlat auto-selected by perf still fails in attach |
| **v4 (current)** | `perf mem record -- true` (no explicit events) | **correct** — now matches the actual spawn-mode invocation used by the runner |

### Added — auto-collector selection

`demo_qwen_hf.py` now auto-selects the best available collector at startup:

- **Bare-metal Linux + root + PMU available** → `perf mem` (spawn mode)
- **WSL2 / no root / PMU probe fails** → `memray`

Override with `--collector memray` or `--collector perf-mem`.

WSL2 detection: reads `/proc/sys/kernel/osrelease` for `microsoft` substring.

### Changed — sudo invocation

`sudo -E python` resolves to the **system Python**, not the venv Python.
All docs and `useful-commands/environment-setup.md` updated to use:
```bash
sudo -E venv/bin/python notebooks/scripts/demo_qwen_hf.py
# or
sudo -E /home/oj/projects/mrm-trace/venv/bin/python notebooks/scripts/demo_qwen_hf.py
```

### Known limitation — region_map empty for PyTorch/transformers backend

`demo_qwen_hf.py` uses the HuggingFace transformers (PyTorch) backend. The labeller cannot
classify addresses into weight/kv_cache/activation regions because:

1. `perf mem` captures load/store events only — no allocation events. The address-range tracker
   has no malloc/free pairs to anchor regions to names.
2. PyTorch symbols (`at::native::mm_cpu_kernel`, `torch::autograd::...`) do not match the
   `ggml_*` / `llama_kv_cache_*` regex rules written for llama.cpp.

**Impact:** `region_type = unknown` for all records. Retention and write-once metrics are
correct but undifferentiated — you cannot separate weight behaviour from KV cache behaviour.

**Roadmap fix:** Before starting collection, enumerate `model.named_parameters()` and record
`(param.data_ptr(), param.data_ptr() + param.nbytes, "weight")` tuples as pre-known address
ranges. Feed these to `TraceLabeller` so addresses that fall within any parameter's range are
immediately classified as `weight` without needing symbol names. This is the only viable path
for PyTorch models since llama.cpp-style debug symbols are not present.

**For publication data with classified regions:** use `demo_perf_mem.py` with a GGUF model
and a debug-symbol build of llama-cpp-python
(`CMAKE_BUILD_TYPE=Debug pip install llama-cpp-python`). The `ggml_*` symbols will then be
present in the perf script output and the labeller rules will classify regions correctly.

---

## System compatibility matrix

| Environment | perf mem | memray | timestamps | region_map | Notes |
|---|---|---|---|---|---|
| **Bare-metal Ubuntu 22.04+** | ✓ spawn mode | ✓ | ns (perf) / 0 (memray) | With debug llama.cpp | Verified on Ubuntu 24.04 / kernel 6.8 |
| **Bare-metal Debian 12+** | ✓ spawn mode | ✓ | ns (perf) / 0 (memray) | With debug llama.cpp | Install `linux-perf` package |
| **WSL2 (any distro)** | ✗ | ✓ | 0 | ✗ | Microsoft kernel blocks PMU; use memray only |
| **Ubuntu / Debian in Docker** | ✗ (no PMU in containers) | ✓ | 0 | ✗ | `--privileged` + host PMU passthrough theoretically possible but untested |
| **Google Colab** | ✗ | ✓ | 0 | ✗ | VM environment, no PMU; notebooks 001-003 use synthetic/memray data |
| **Bare-metal AMD Linux** | Untested | ✓ | — | — | AMD uses IBS events; `perf mem` auto-select may pick different events |
| **macOS** | ✗ (perf is Linux-only) | ✗ (Linux-only) | — | — | Not supported |
| **Windows (native)** | ✗ | ✗ | — | — | Not supported; use WSL2 |

### Notes per environment

**Bare-metal Ubuntu / Debian:**
- Install perf: `sudo apt install linux-perf`
- Confirm spawn mode works: `sudo perf mem record -o /dev/null -- true`
- Use `sudo -E venv/bin/python` (not `sudo -E python`) — sudo drops PATH
- `perf_event_paranoid = 4` (Ubuntu default) blocks non-root perf but root bypasses it
- `perf mem record -p PID` (attach mode) fails on kernel 6.8 for PEBS ldlat events — use spawn mode

**WSL2:**
- `_is_wsl2()` detects `microsoft` in `/proc/sys/kernel/osrelease` → auto-selects memray
- All timestamps will be 0; retention metrics will be 0
- Useful for development and correctness testing of the pipeline, not for publication data

**Google Colab:**
- `pip install mrm-trace memray` works
- Notebooks 001, 002, 003 run without root or model files (synthetic/simulated data)
- For real collection, mount a GGUF model via Google Drive and use memray
- perf is not available in Colab's VM environment

**Docker:**
- Default Docker containers block `perf_event_open` syscall
- `--privileged` + `--pid=host` may allow perf mem but is security-sensitive
- memray works in standard containers without any special flags

**AMD CPUs:**
- AMD uses IBS (Instruction-Based Sampling) instead of Intel PEBS
- `perf mem record` auto-select should handle this but is untested
- Event names differ: AMD uses `ibs_op` events rather than `cpu/mem-loads`
- Parser and schema are event-agnostic — data format is the same regardless

---

## [v1.3] — 2026-05-30

- Fixing slow parsing for millions of records (streaming line-by-line from file)
- Frame scan finds the first non-empty function name → symbol/dso columns populated
  with Python-level names instead of empty strings

---

## [v1.0] — initial release

- Phases 1–9 complete: foundations, engine adapters, collectors, parsing, labelling,
  analysis, telemetry, reporting, hardening

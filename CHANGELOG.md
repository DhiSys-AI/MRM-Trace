# Changelog

All notable changes to `mrm-trace` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased] — v1.5 (current development)

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

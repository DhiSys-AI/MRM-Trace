# Changelog

All notable changes to `mrm-trace` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased] — v1.5 (current development)

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

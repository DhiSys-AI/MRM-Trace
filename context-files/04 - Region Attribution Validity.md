# Region Attribution in PMU-Sampled Inference Traces

*A methodological finding, for the Methods and Limitations sections. Measured on
Llama-3.2-1B-Instruct-Q4_K_M under llama.cpp, Intel Core i9-11980HK (Tiger Lake,
PMU v5), Linux 7.0, `perf mem` / PEBS, 2026-10-03.*

---

## 1. Summary

Characterising LLM inference memory by region — weights, KV cache, activations
— requires attributing each sampled address to a region. We find that the two
attribution mechanisms in common reach are both unsound on this platform, in
ways that produce confident and wrong results rather than obvious failures:

1. **Symbol-based attribution is invalid for data addresses.** A PMU sample
   carries an instruction pointer and a data address. The symbol resolved from
   the instruction pointer identifies *the code that was running*, not *the
   region the data address belongs to*. For a store sample the data address is
   the kernel's output buffer, not its inputs.

2. **Store-dominated sampling cannot observe a read-only region.** `perf mem
   record` samples stores at a fixed period but samples loads only above a PEBS
   load-latency threshold. The resulting trace is ~99.8% stores. Model weights
   are mapped read-only, so no store can ever target them: the weight region is
   unobservable by construction, not merely undersampled.

Both are correctable. We report them because each one independently yields a
complete, internally consistent, publication-shaped dataset whose region
attribution is wrong, and neither is visible in any summary statistic we
would normally inspect.

---

## 2. Symbol-based attribution

### 2.1 The mechanism and why it is tempting

llama.cpp exposes informative symbol names. A debug build resolves
`ggml_vec_dot_q6_K_q8_K` (quantised dot product, which reads weights),
`llama_kv_cache_find_slot`, and so on. Mapping symbol regexes to region types
is therefore an obvious attribution strategy, and it appears to work: symbol
coverage reaches 100% and region coverage reaches 100%.

### 2.2 Measurement

One run: Llama-3.2-1B, ctx 2048, 4 prompts x 512 tokens, 28,321,245 samples.
The weight region's ground truth is the mmap'd GGUF file, whose address range
is known exactly from `/proc/self/maps`.

| | |
|---|---|
| Rows labelled `weight` by `ggml_vec_dot\|ggml_mul_mat` rules | 26,617,751 |
| Of those, inside the mmap'd GGUF file | **114** (0.0004%) |

The symbol rule was wrong on essentially every row it fired on. The reason is
mechanical: `ggml_vec_dot_q6_K_q8_K` *reads* weights and *writes* activations,
so its store samples carry activation addresses. The GGUF mapping is `r--s`
(read-only, shared), which makes the error structural rather than statistical —
a store address can never be a weight address.

### 2.3 Error amplification through granule propagation

Region attribution is necessarily coarser than a single address: retention is
`last access − first access` *of a region*, so addresses are grouped into
granules (64 KiB here). Once a granule is classified, subsequent accesses
inherit the label. A single mis-attributed sample therefore captures a whole
granule and everything that later touches it.

The `kv_cache` label in the same run:

| | |
|---|---|
| Rows labelled `kv_cache` | 279,186 |
| Distinct 64 KiB granules claimed | **4** |
| Address span of those granules | 0x709b… to 0x7ffd… (one in the main stack) |

Top symbols on those rows: `nearest_int` (35,474), `__memmove_evex_unaligned_erms`
(27,630), `quantize_row_q8_K_ref` (27,471), `ggml_compute_forward_soft_max_f32`
(26,556), `ggml_compute_forward_swiglu_f32` (19,418) — quantisation helpers,
`memmove`, softmax and SwiGLU. None is a KV cache operation. Four granules
tagged by rare `llama_kv_cache_*` samples absorbed a quarter of a million
unrelated accesses.

There is a second, subtler problem specific to KV: in llama.cpp the functions
named `llama_kv_cache_*` manage *slot bookkeeping* (cell tables, sequence
positions). The K and V tensor writes themselves occur in generic `ggml`
compute ops. A sample whose instruction pointer lies in `llama_kv_cache_*` is
therefore touching cache metadata, not cache contents — the symbol is not
merely weak evidence for a KV address, it is evidence against it.

### 2.4 Reproducibility consequence

Across five runs of one configuration — which were *identical* workloads, since
the batch-size parameter was recorded but never reached the engine — the
`kv_cache` share of the trace was:

| rep | kv_cache accesses | share of trace |
|---|---|---|
| 1 | 3,730,498 | 13.18% |
| 2 | 287 | 0.0010% |
| 3 | 79,614 | 0.28% |
| 4 | 79,876 | 0.28% |
| 5 | 304 | 0.0011% |

A spread of four orders of magnitude between repetitions. This is the
signature to look for: whether a rare symbol sample happened to land on a hot
granule is a coin flip, and the resulting label then governs millions of rows.

---

## 3. Store-dominated sampling and read-only regions

### 3.1 Measurement

Same run, by operation type and against the weight mapping:

| | |
|---|---|
| Total samples | 28,321,245 |
| Loads | 65,288 (0.23%) |
| Stores | 28,255,957 (99.77%) |
| Samples inside the weight mapping | 114 — **all loads** |
| Cache level of those 114 | L2 47, L1 40, DRAM 14, LFB 10, L3 3 |

The asymmetry is a property of `perf mem`: the store event is sampled on a
plain period, while the load event is a PEBS load-latency event gated by an
`ldlat` threshold (default 30 cycles). Weights are read-only, so weight
observation depends entirely on the load channel, which here carried 0.23% of
samples — and 114 of those 65,288 loads fell in the 800 MB weight mapping.

A 22-minute run therefore yielded 114 observations of the region the study is
about.

### 3.2 Correction

Loads and stores can be given independent periods, and the latency threshold
lowered:

```
perf record -d --weight \
  -e cpu/mem-loads,ldlat=3,period=2003/P \
  -e cpu/mem-stores,period=1000003/P
```

Measured effect on the same workload:

| | `perf mem record -c N` | independent periods, `ldlat=3` |
|---|---|---|
| loads | 0.23% | **90.0%** |
| samples in the weight mapping | 114 | **6,992** |
| distinct weight granules observed | — | 3,941 (64 KiB) / 6,723 (4 KiB) |

61x more weight observations, from a probe one twentieth the length.

Two implementation notes that cost us a false negative. `ldlat=3` rather than
the default 30 accounts for a 25x difference in eligible load samples, because
weight reads that hit in L1/L2 do not clear a 30-cycle threshold. And `-d` is
mandatory: it requests `PERF_SAMPLE_ADDR`, and without it `perf script` reports
*"do not have ADDR attribute set"* and emits nothing — an empty trace rather
than an error.

### 3.3 The kernel caps the sample rate, and lowers the cap itself

Requesting a dense period does not guarantee getting it.
`kernel.perf_event_max_sample_rate` caps samples/s/CPU, and the kernel reduces
that cap on its own when sampling overhead exceeds
`kernel.perf_cpu_time_max_percent`, logging:

```
perf: interrupt took too long (155256 > 154436), lowering kernel.perf_event_max_sample_rate to 1000
```

The reduction persists for the rest of the boot. On our host the default
100,000 fell to 1,000 during a dense-sampling probe. A subsequent run
requesting ~103,500 samples/s delivered 13,790 — 862/s/CPU across 16 CPUs,
sitting exactly on the lowered cap, 7.5x short of the request, with no error
reported by `perf` or anywhere in the output.

Two consequences:

- **Cross-run comparability.** Two runs with identical `--sample-period` can
  differ in sample density by an order of magnitude depending on when in the
  boot they were collected. Sample counts are not comparable unless the cap is
  known and equal.
- **Within-run uniformity.** If the cap is lowered *during* a collection,
  density is higher at the start than at the end. Retention is
  `last access − first access` of a region, so a non-uniform density biases it
  directly.

We therefore read the cap before and after every collection, record both in
`metadata.json` (`collector.perf_event_max_sample_rate_before` / `_after`),
warn when it is below 50,000 at start-up, and mark any run during which it
changed as not comparable. Anyone reproducing this work should pin it
explicitly rather than inherit whatever the kernel has settled on:

```bash
sudo sysctl -w kernel.perf_event_max_sample_rate=100000
```

---

## 4. Establishing KV cache addresses

`/proc/self/maps` cannot isolate the KV cache under llama.cpp. The geometry
implies `2 · n_layer · n_ctx · n_head_kv · head_dim · bytes_per_element`
(64 MiB at ctx 2048 for this model), but the engine carves the cache out of a
larger ggml backend arena. Live mappings after model load at ctx 2048:

```
805.3 MB  rw-p  anon      (ggml arena)
799.9 MB  r--s  file      (GGUF weights)
536.9 MB  rw-p  anon      (ggml arena)
262.7 MB  rw-p  anon      (ggml arena)
 74.4 MB  rw-p  [heap]
```

Nothing of the implied size exists, so any size-matching rule either matches
nothing or matches an arena one order of magnitude too large. Labelling an
arena `kv_cache` produces exactly the confident-and-wrong KV lifecycle data
described in §2.3.

Interposing the allocator does give an exact answer. The KV buffer is allocated
in one call of precisely the implied size:

| n_ctx | implied KV size | captured allocation |
|---|---|---|
| 512 | 16,777,216 | `posix_memalign` @ 0x719a9e6ff040 |
| 1,024 | 33,554,432 | `mmap` @ 0x7b9b0fe00000 |
| 2,048 | 67,108,864 | `posix_memalign` @ 0x7344dca7e040 |
| 4,096 | 134,217,728 | `posix_memalign` @ 0x73c45f07e040 |

Both `posix_memalign` and `mmap` must be intercepted: glibc routes the same
request differently by size. The match must be exact and unique — at ctx 1,024
the implied 32 MiB collides with glibc's 32 MiB arena reservations, so that
size is ambiguous and must be reported as such rather than resolved by
preference.

---

## 5. What we report, and what we do not

Our attribution now rests only on address ranges, which state where an address
*is*:

| region | source | status |
|---|---|---|
| `weight` | mmap'd GGUF file, from `/proc/self/maps` | exact |
| `kv_cache` | engine allocation, captured at allocation time | exact when unambiguous, otherwise **absent** |
| `activation` | store destination of a forward-pass kernel | inferred |
| `runtime` | thread stacks, heap, malloc arenas, shared libraries | exact |
| `unknown` | in no known mapping | — |

Symbols are retained in the trace and used only for the one inference they
support soundly: a forward-pass kernel's *store* destination is an activation
buffer. They can no longer establish `weight` or `kv_cache`.

Where the KV cache cannot be isolated, `kv_cache` and the KV lifecycle table
are empty **by design**. An absent measurement is publishable; a wrong one is
not, and the four-orders-of-magnitude spread in §2.4 is what the alternative
looks like.

---

## 6. Guidance for others

For anyone attributing PMU-sampled addresses to logical memory regions:

1. **Never attribute a data address from an instruction-pointer symbol.** The
   two refer to different things. Where a symbol seems to work, check it
   against an independently known address range before trusting it.
2. **Check the permission bits of the region you mean to study.** A read-only
   region cannot appear in store samples at all, so a store-dominated trace
   cannot measure it, at any sample count.
3. **Record `perf_event_max_sample_rate` with every trace, and pin it.** It
   caps the sampling you actually get, the kernel lowers it without warning
   under the dense sampling that §3.2 requires, and the reduction outlives the
   run that caused it.
4. **Report load and store derived metrics separately.** A blended figure is
   dominated by whichever event has the finer period, which is a property of
   the collector configuration rather than of the workload.
5. **Treat repeated runs of one configuration as an attribution test.** Region
   shares that vary by orders of magnitude across identical repetitions
   indicate propagated mis-attribution, not measurement noise.
6. **Prefer an absent region to an approximated one**, and state the absence.
   Granule propagation means a single mis-attributed sample does not produce a
   small error; it produces a large and confident one.

---

## 7. Reproducing these measurements

The checks that produced each number above:

- §2.2 — compare `region_type == "weight"` addresses against the GGUF mapping
  range recorded in the run's `address_regions`.
- §2.3 — group `region_type == "kv_cache"` rows by granule and tabulate their
  symbols.
- §2.4 — `kv_cache` access counts from `region_map.parquet` across repetitions
  of one configuration.
- §3.1 — sample counts by `op_type`, and membership of the weight mapping.
- §3.2 — the same run under both event configurations.
- §4 — allocation log from the interposer against `expected_kv_bytes()`.

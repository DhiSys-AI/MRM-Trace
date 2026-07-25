# Retention-Aware Memory Placement for Batch LLM Inference on Reused Hardware

*Revised research framing and Phase-1 measurement design.*
*Working title for the eventual paper: "Where Does the Data Live? Retention-Aware Memory Placement for Batch LLM Inference on End-of-Life Hardware."*

---

## 1. Revised Contribution Statement (one page)

**Context.** Inference, not training, is now the dominant and fastest-growing share of LLM operational cost and carbon, and demand is projected to keep climbing. In parallel, a large and growing population of hardware sits idle or is discarded: last-generation datacenter GPUs, decommissioned servers, and consumer machines, much of it exported as e-waste rather than reused. A significant class of LLM and SLM work — scheduled, offline, batch jobs — is latency-tolerant and does not require frontier silicon. This creates an opportunity that the current research conversation, focused on new memory hardware (HBM, MRM) and on frontier accelerators (Groq, datacenter GPUs), largely ignores: making cheap, slow, heterogeneous, already-manufactured hardware a viable substrate for batch inference.

**Prior work and the gap.**

- *Managed-Retention Memory (Legtchenko et al., 2025)* characterises LLM memory behaviour qualitatively (read:write >1000:1, sequential large-block reads, soft-state KV cache) and proposes a *new memory cell technology*. It reports no empirical trace data, and its destination is new silicon — which by definition cannot be retrofitted into reused hardware. Its software insight (retention-aware placement and scheduling) is, however, hardware-agnostic and directly applicable here.
- *GreenLLM (2024) and carbon-aware life-cycle work (HotCarbon '24)* establish that reusing older, low-performing GPUs lowers the carbon of LLM serving, via **compute** disaggregation (prefill/decode) and carbon-aware **scheduling**. They do not characterise per-region byte lifetime, write-once ratio, or read frequency, and they do not address **memory-tier placement**.
- *Offloading systems (FlexGen, DeepSpeed ZeRO-Inference, llama.cpp NVMe offload)* optimise fitting a single large model onto one memory-constrained device. They do not target a heterogeneous fleet of end-of-life hardware, and they do not use measured retention behaviour to drive placement.
- *Existing trace studies (Banasik 2025, CHEOPS '25, Fisher 2025)* each measure one adjacent axis (CPU cache sensitivity, block-layer SSD I/O, NUMA locality) and none report retention duration, write-once ratio, or read frequency by region type.

**The unmeasured quantities.** No published work measures, for batch LLM/SLM inference: (i) the per-region **retention-duration distribution** — how long weights, KV blocks, and activations must remain valid before they can be discarded under a batch schedule; (ii) the **write-once ratio** — the fraction of allocated bytes written exactly once and thereafter only read; or (iii) the **read frequency per region type** as a function of model size, context length, and batch size. These are precisely the quantities needed to decide which data structure can tolerate which (cheap, slow) memory tier.

**Contribution.** This work presents the first empirical characterisation of LLM/SLM batch-inference memory behaviour framed for **retention-aware placement across the heterogeneous memory and storage tiers of reused hardware** (VRAM, system DRAM, CXL where present, NVMe, SATA SSD). We release an open trace and metrics dataset across [N small/mid models, M context lengths, K engines] measured on salvage-class hardware, and we provide a **crossover model** that maps measured byte-lifetime and access-frequency distributions to the throughput-, energy-, and embodied-carbon-optimal tier for each region, identifying the regimes (model size, context length, batch size) in which reused hardware meets batch service targets at lower total carbon than buying new. We differentiate from GreenLLM by operating at the **memory-placement** layer rather than the compute-disaggregation layer, and we inherit the retention-aware framing that the MRM authors called for but did not measure.

---

## 2. The Crossover Model (the bridge that earns the paper)

The reviewer's "so what?" is answered by one artifact: a model that turns measured distributions into placement decisions and a carbon verdict.

**Inputs (measured in Phase 1):** per-region size, retention-duration distribution, write-once ratio, read frequency, sequential/random access ratio — for `weights`, `kv_cache`, `activation`.

**Inputs (hardware specs, parameterised so others plug in their fleet):** per-tier read/write latency and bandwidth, capacity, power (W), $ cost, and amortised **embodied carbon** per the device's life-cycle assessment.

**Output:** for each region, the tier that minimises a chosen objective (tokens per job-hour, joules per token, or grams CO2e per token including amortised embodied carbon) subject to a batch latency/SLA budget — plus the **crossover points** (e.g. "weights tolerate NVMe offload up to model size X before batch throughput drops below target"; "reused T4-class hardware beats new-silicon procurement on total carbon below utilisation U").

Even a crude spreadsheet version of this is what survives scrutiny. Build it early; let it tell you which measurements actually matter.

---

## 3. Slimmed Phase-1 Measurement Design

Goal of Phase 1: produce the three metrics plus per-tier energy on representative reused hardware, feed the crossover model, and deliver a publishable trace dataset. Strip the enterprise instrumentation; keep only what protects scientific validity.

### 3.1 Scope (deliberately small — the reuse framing requires it)

- **Models (small/mid only):** e.g. Llama-3.2 1B and 3B, Phi-3.5-mini, Mistral-7B, and one small MoE (e.g. an 8x-small Mixtral variant) to expose expert-routing access patterns. No 70B — it contradicts the reuse premise and inflates the matrix.
- **Context lengths:** 2K, 8K, 32K. (KV behaviour scales here; 128K is interactive-frontier territory, not batch-reuse.)
- **Batch:** offline, throughput-oriented, fixed prompt suites; sweep batch sizes 1, 4, 8.
- **Engines:** vLLM (GPU path, PagedAttention/continuous batching — the realistic batch stack) and llama.cpp (CPU and consumer-GPU path — the realistic salvage stack).
- **Repetitions:** 2–3 for variance; 1 warmup discarded.

### 3.2 Hardware (representative salvage classes, not a fleet)

Phase 1 needs only 1–2 machines, each labelled as a *class* so the crossover model generalises:

- one older datacenter-inference GPU if accessible (T4 / V100 / similar) — the "decommissioned server" class;
- one consumer machine (mid-range GPU or strong CPU + NVMe + lots of system DRAM) — the "system lying in someone's home" class.

If only one machine is available, run it and parameterise the model so other hardware can be plugged in later; do not block Phase 1 on fleet access.

### 3.3 Minimal instrument (cut ~80% of the original spec)

Keep:

- **Region access trace:** `perf mem` where PMU is available; `memray` / process monitor as fallback. Stream-parse to a canonical trace schema (run_id, timestamp, address/page, op_type, region_type, phase).
- **Region labelling:** symbol + phase + address-range, into `weights` / `kv_cache` / `activation` / `runtime` / `unknown`. This is the part that earns the dataset — invest here.
- **Energy, now first-class:** `nvidia-smi` power draw and/or RAPL sampled through each run, aligned to phases. (In the reuse framing energy is a headline metric, not optional telemetry.)
- **Per-run metadata:** hardware/software identity, model/engine/config, phase timestamps, observed-lifetime caveat.

Defer to v2 (do **not** build in Phase 1): the thermal/observer-effect validity classifier, property-based test harness, CI gates, ADR ceremony, HF/Zenodo export tooling, and SQLite manifest indexing. They are good engineering and bad use of Phase-1 time.

### 3.4 Metrics (the three that matter, plus carbon)

Per region type, grouped by model / context / batch / engine / hardware class:

- retention duration: p50/p90/p99/max/mean/std (reported as *observed lifetime under the batch schedule*, explicitly a proxy, not a hardware requirement);
- write-once ratio (region-level and byte-level);
- read frequency (per second and per output token);
- sequential vs random access ratio and dominant stride;
- **energy per token, per region where attributable, and per tier.**

### 3.5 Validity caveat to state up front

`perf mem` / power traces on a CPU or old GPU are shaped by that machine's cache hierarchy, page cache, and the engine's mmap/offload behaviour. Report retention as *observed lifetime under a given batch scheduling policy*, and validate the weight-vs-KV lifetime ordering on at least the vLLM GPU path so the numbers are not an artifact of one CPU setup.

### 3.6 Phase-1 deliverables

1. Open trace + metrics dataset on salvage-class hardware (the foundational, citable artifact).
2. The crossover / placement model (script or spreadsheet) with the hardware-spec table parameterised.
3. A short embodied-vs-operational carbon analysis for the batch-reuse case, with the crossover points called out.
4. A 6–8 page workload-characterisation paper. Targets: HotCarbon, MLSys, USENIX ATC, or a sustainable-computing / systems venue where the carbon framing lands.

### 3.7 Rough timeline

A focused single researcher: instrument + labelling (4–6 weeks), measurement sweep on 1–2 machines (3–4 weeks), crossover model + carbon analysis (3–4 weeks), writing (3–4 weeks). Roughly one academic term, not the multi-quarter build the original spec implied.

---

## 4. What Changed From the Original Plan, and Why

- **North Star:** from "inform the design of new MRM silicon" to "retention-aware placement across existing tiers in reused hardware." Achievable with no new fabrication; keeps the MRM citation via its software insight.
- **Competitive frame:** from "MRM vs HBM" to "memory-placement layer, differentiated from GreenLLM's compute-disaggregation layer." Groq becomes contrast, not competitor.
- **Headline metric gains a partner:** retention/write-once/read-frequency now feed a carbon crossover, which is the "so what" that survives review.
- **Scope shrinks naturally:** small/mid models, batch, moderate contexts — the over-scope risk dissolves.
- **Instrument shrinks ~80%:** validity/thermal/observer apparatus deferred; energy measurement promoted to first-class.
- **New required citation:** GreenLLM (2412.20322) and the HotCarbon '24 carbon-life-cycle work, as the prior art you differentiate from.

Short answer: no GPU, no HBM, nothing exotic — but there is exactly one hardware requirement that will quietly make or break the whole effort, and it's worth pinning down on day one.

## The one thing you actually need: a PMU-capable bare-metal Linux box with root

`perf mem` — your primary collector — isn't software magic. It rides on hardware performance-monitoring facilities that sample memory loads/stores with their addresses and latencies: PEBS on Intel, IBS on AMD, SPE on ARM. So the requirement isn't "fast" or "expensive," it's "has a CPU whose PMU exposes memory sampling, and an environment that lets you reach it." Concretely that means:

- **Bare metal, not a shared cloud VM.** This is the gotcha that catches people. Most standard cloud instances and essentially all containers do *not* expose PEBS/IBS/SPE — the hypervisor hides it. You'll run `perf mem record` and get zero samples or "not supported." You need a physical machine you own, or a *bare-metal* cloud SKU specifically, not a normal VM.
- **Root** (or `perf_event_paranoid` lowered), since memory sampling needs elevated privilege.
- **The right CPU.** Intel Skylake-or-newer Xeon/Core is the best-trodden path for `perf mem` with `ldlat` (load-latency) sampling and clean data-linear-address reporting. AMD Zen 2+ works via IBS; ARM Neoverse (e.g. Graviton-class) via SPE — both supported by modern `perf` but less documented for this exact use.
- A recent kernel with the matching `linux-tools-$(uname -r)` perf build.

A perfectly ordinary multi-core workstation or a decommissioned server qualifies. The constraint is access and PMU, not money.

**Validate this before writing collection code.** Run `perf mem record -- sleep 2` then `perf mem report` and confirm you get real samples with addresses. If that works on your target machine, the rest is software. If it doesn't, no amount of code fixes it — you need a different box. This is a five-minute, free de-risking step and it's the first thing your `mrm-trace validate` command should check.

## Most of the project needs no special hardware at all

Worth separating two things the spec conflates:

The tool itself — config loader, parser, labeller, analyser, exporter, and the entire test suite — is developed and tested against the **synthetic trace fixtures** with known ground truth (the spec already mandates these). You can build and CI-test all of that on any laptop. Only the *live collection* runs need the PMU box. So the dependency is narrow: develop anywhere, collect on one bare-metal machine.

For the models, since you're sticking to small/mid (1B–8B quantised), a box with 32–64 GB RAM and ~200 GB free NVMe for trace files is plenty. `llama.cpp` CPU is the reliable primary engine for a tool others will run; vLLM CPU-only mode works but is fussier (needs a CPU build), so treat it as the secondary path rather than the default.

## The honest bit about "contributing back to Microsoft"

You should go in clear-eyed that a CPU-side trace is a *proxy* for what MRM ultimately cares about, which is datacenter GPU+HBM behaviour. If you frame the dataset as "HBM-accurate numbers," a reviewer or the MRM authors will rightly push back.

But here's why the dataset is still a real contribution if you frame it correctly: the three metrics MRM needs — retention duration, write-once ratio, read frequency by region — are largely *logical* properties of the inference engine's algorithm, not the memory substrate. Weights are written once and read many times whether they sit in HBM or DRAM; a KV block's lifetime is bounded by sequence completion or eviction regardless of where it physically lives. So a CPU trace captures the *logical access-pattern structure* faithfully, even though absolute latencies and bandwidths won't match HBM. Frame the dataset as "logical memory-access characterisation of LLM inference, retention-framed" — first-order evidence for the MRM design questions, explicitly not a hardware-timing model — and it's both honest and genuinely useful. Put the hardware/collector provenance and that caveat directly in the data card.

## Kaggle practicalities

Publishing needs no special hardware — it's uploading your derived outputs plus a data card. The one thing to manage is size: raw `perf.data` and `perf script` dumps balloon fast, and you don't want to publish every multi-gigabyte raw file. Publish the **derived, schema'd artifacts** — `metrics.csv`, `region_map.parquet`, `kv_block_lifecycle.parquet`, and a *representative sample* of `trace.parquet` (zstd-compressed) — and keep the full raw traces archived separately (Zenodo is better than Kaggle for bulky raw data and gives you a DOI for citation). A clean data card with schema documentation, hardware/collector metadata, and the proxy caveat is what makes it reusable and citable.


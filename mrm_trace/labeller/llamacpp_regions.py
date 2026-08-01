"""
llama-cpp-python memory region extraction for perf-mem region pre-classification.

perf mem captures load/store events — there are no malloc/free pairs, and ggml_*
symbols are stripped in pip-installed llama-cpp-python.  The memory layout is
therefore recovered from /proc/self/maps.

Spec §5 requires five region classes (weight, kv_cache, activation, runtime,
unknown).  Every mapping in /proc/self/maps is classified, not just the model
buffers — otherwise thread stacks, the heap, and shared libraries fall through to
``unknown`` and dominate the trace (they account for ~86% of perf mem samples on
a llama.cpp run).  ``unknown`` should mean "we could not account for this
address", not "we only looked at two of the mappings".

  weight    : GGUF file is mmap'd read-only — entries with the model filename
  kv_cache  : llama.cpp allocates the KV cache as a large anonymous mmap *after*
              model creation; comparing maps before vs after Llama() isolates it
  runtime   : thread stacks, malloc arenas, the heap, shared libraries, and any
              other mapping that is not model data

Usage::

    pre = snapshot_proc_maps()       # BEFORE Llama() construction
    llm = Llama(model_path=...)      # Llama() allocates weights + KV cache
    regions = extract_llamacpp_regions(model_path, pre)
    llm("warm up")                   # threadpool + compute buffers appear here
    regions += extract_llamacpp_regions(model_path, pre)   # catch late mappings
    labeller.register_address_ranges(regions)

The second call is not optional in practice: llama.cpp creates its threadpool at
generation time, so the worker-thread stacks do not exist yet at the first call.
``register_address_ranges`` de-duplicates by page (first classification wins), so
overlapping snapshots are safe to concatenate.
"""

from __future__ import annotations

import pathlib
from typing import List, Optional

_MIN_KV_BYTES = 2 * 1024 * 1024  # 2 MB — ignore tiny anonymous regions

# A mapping counts as the KV cache only if its size matches the size implied by
# the model geometry.  Allocators round up and llama.cpp adds padding, so the
# window is loose — but it is still a check, which "large and new" was not.
_KV_SIZE_TOLERANCE = (0.75, 2.0)

# A pthread stack is one anonymous rw mapping immediately preceded by a PROT_NONE
# guard mapping.  glibc's default is 8 MiB + a 4 KiB guard; musl and explicit
# pthread_attr_setstacksize() use other sizes, so the range is deliberately wide.
_MIN_STACK_BYTES = 64 * 1024
_MAX_STACK_BYTES = 64 * 1024 * 1024

# Refuse to page-expand absurdly large reservations (some runtimes reserve TB of
# PROT_NONE address space).  4 GiB = 1M pages is far above any real model buffer.
_MAX_REGION_PAGES = 1 << 20

_KERNEL_PATHS = frozenset({"[vvar]", "[vdso]", "[vsyscall]", "[vvar_vclock]"})


def _parse_proc_maps() -> List[dict]:
    """Return all /proc/self/maps entries as dicts, in address order."""
    regions: List[dict] = []
    try:
        with open("/proc/self/maps") as fh:
            for line in fh:
                parts = line.split()
                if not parts:
                    continue
                try:
                    start_s, end_s = parts[0].split("-")
                    start = int(start_s, 16)
                    end = int(end_s, 16)
                except ValueError:
                    continue
                perms = parts[1] if len(parts) > 1 else ""
                path = parts[5] if len(parts) >= 6 else ""
                regions.append({
                    "addr_start": start,
                    "addr_end": end,
                    "size": end - start,
                    "perms": perms,
                    "path": path,
                })
    except OSError:
        pass
    return regions


def snapshot_proc_maps() -> List[dict]:
    """
    Snapshot /proc/self/maps.  Call this BEFORE constructing a Llama() object so
    that extract_llamacpp_regions() can diff before vs after to find new regions.
    """
    return _parse_proc_maps()


def _is_guard(entry: Optional[dict]) -> bool:
    """True if *entry* is an anonymous PROT_NONE mapping (a guard page)."""
    return (
        entry is not None
        and not entry["path"]
        and entry["perms"].startswith("---")
    )


def _classify_anonymous(
    entry: dict,
    prev: Optional[dict],
    nxt: Optional[dict],
    is_new: bool,
    min_kv_bytes: int,
    expected_kv_bytes: Optional[int],
) -> "tuple[str, str]":
    """
    Classify an anonymous mapping.  Returns (region_type, source).

    Discriminating KV cache from thread stacks and malloc arenas matters: all
    three are new anonymous rw mappings of several MB, and a size-only rule
    mislabels stacks as kv_cache, which then dominates the KV lifecycle output.

      thread stack : guard page immediately BELOW  (stack grows down into it)
      malloc arena : PROT_NONE tail immediately ABOVE (unused arena remainder)
      kv cache     : new mapping whose size matches the model geometry

    A new anonymous mapping that is merely *large* is NOT the KV cache.  On a
    Llama-3.2-1B / ctx=512 run the post-load mappings are 262 MB, 778 MB, 536 MB
    and 2 MB, while the geometry implies a 16.8 MB KV cache — llama.cpp carves it
    out of a larger backend arena, so it is not separable at page granularity.
    Labelling those arenas ``kv_cache`` produces confident, wrong KV lifecycle
    data; they are reported as ``runtime``/``anon_post_load`` instead so the gap
    stays visible rather than being silently coerced.
    """
    if not entry["perms"].startswith("rw"):
        return "runtime", "anon_noaccess" if entry["perms"].startswith("---") else "anon_ro"

    if _MIN_STACK_BYTES <= entry["size"] <= _MAX_STACK_BYTES and _is_guard(prev):
        return "runtime", "thread_stack"

    if _is_guard(nxt):
        return "runtime", "malloc_arena"

    if is_new and entry["size"] >= min_kv_bytes:
        if expected_kv_bytes:
            lo, hi = _KV_SIZE_TOLERANCE
            if lo * expected_kv_bytes <= entry["size"] <= hi * expected_kv_bytes:
                return "kv_cache", "anon_kv_sized"
        return "runtime", "anon_post_load"

    return "runtime", "anon"


def expected_kv_bytes(gguf_metadata: dict, n_ctx: int, bytes_per_element: int = 2) -> Optional[int]:
    """
    KV cache size implied by GGUF model geometry, or None if the keys are absent.

        2 (K and V) × n_layer × n_ctx × n_head_kv × head_dim × bytes_per_element

    ``gguf_metadata`` is ``Llama(...).metadata``.  Defaults to f16 KV (2 bytes),
    which is llama.cpp's default; pass 1 for a q8_0 KV cache.
    """
    def _find(*suffixes):
        for key, value in gguf_metadata.items():
            if any(key.endswith(s) for s in suffixes):
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
        return None

    n_layer = _find(".block_count")
    n_head_kv = _find(".attention.head_count_kv")
    head_dim = _find(".attention.key_length")
    if head_dim is None:
        embed, n_head = _find(".embedding_length"), _find(".attention.head_count")
        head_dim = embed // n_head if embed and n_head else None
    if not (n_layer and n_head_kv and head_dim and n_ctx):
        return None
    return 2 * n_layer * n_ctx * n_head_kv * head_dim * bytes_per_element


def _classify_entry(
    entry: dict,
    prev: Optional[dict],
    nxt: Optional[dict],
    model_filename: str,
    pre_addrs: set,
    min_kv_bytes: int,
    expected_kv: Optional[int],
) -> "tuple[str, str]":
    """Classify one /proc/self/maps entry.  Returns (region_type, source)."""
    path = entry["path"]

    # weight: GGUF file mmap'd (may appear as multiple segments)
    if model_filename and (
        path.endswith(model_filename) or path.endswith(model_filename + " (deleted)")
    ):
        return "weight", "mmap_gguf"

    if path == "[stack]":
        return "runtime", "main_stack"
    if path == "[heap]":
        return "runtime", "heap"
    if path in _KERNEL_PATHS:
        return "runtime", "kernel_map"
    if path.startswith("["):
        return "runtime", "special_map"
    if path:
        return "runtime", "file_mapping"

    is_new = (entry["addr_start"], entry["addr_end"]) not in pre_addrs
    return _classify_anonymous(entry, prev, nxt, is_new, min_kv_bytes, expected_kv)


def extract_llamacpp_regions(
    model_path: str,
    pre_load_maps: List[dict],
    min_kv_bytes: int = _MIN_KV_BYTES,
    max_region_pages: int = _MAX_REGION_PAGES,
    expected_kv: Optional[int] = None,
) -> List[dict]:
    """
    Return address regions for the current process, classified into the spec's
    region classes.  Every mapping is classified; nothing is left for the
    labeller to guess at.

    Parameters
    ----------
    model_path:
        Path to the GGUF file (absolute or relative).
    pre_load_maps:
        Snapshot from snapshot_proc_maps() taken BEFORE Llama() was constructed.
        Used to tell model buffers (new) from pre-existing runtime allocations.
    min_kv_bytes:
        Minimum size for a new anonymous region to be considered as kv_cache.
    max_region_pages:
        Mappings larger than this many pages are skipped rather than expanded
        page-by-page by the labeller.
    expected_kv:
        KV cache size implied by the model geometry, from expected_kv_bytes().
        Without it no mapping is claimed as kv_cache — size alone is not
        evidence, and a wrong KV label is worse than an absent one.

    Returns
    -------
    List of region dicts with keys: region_type, addr_start, addr_end,
    size_bytes, source.
    """
    filename = pathlib.Path(model_path).name
    post_maps = _parse_proc_maps()
    pre_addrs = {(r["addr_start"], r["addr_end"]) for r in pre_load_maps}

    regions: List[dict] = []
    for i, entry in enumerate(post_maps):
        if entry["size"] >> 12 > max_region_pages:
            continue
        prev = post_maps[i - 1] if i > 0 else None
        nxt = post_maps[i + 1] if i + 1 < len(post_maps) else None
        region_type, source = _classify_entry(
            entry, prev, nxt, filename, pre_addrs, min_kv_bytes, expected_kv
        )
        regions.append({
            "region_type": region_type,
            "addr_start": entry["addr_start"],
            "addr_end": entry["addr_end"],
            "size_bytes": entry["size"],
            "source": source,
        })
    return regions


def summarise_regions(regions: List[dict]) -> "dict[str, dict]":
    """
    Aggregate regions by source for logging: {source: {count, bytes, region_type}}.

    Printing this makes a misclassification visible at run time — e.g. a
    ``thread_stack`` count that does not match the engine's thread count, or a
    ``kv_cache`` byte total far from the configured context size.
    """
    summary: "dict[str, dict]" = {}
    for r in regions:
        row = summary.setdefault(
            r["source"], {"count": 0, "bytes": 0, "region_type": r["region_type"]}
        )
        row["count"] += 1
        row["bytes"] += r["size_bytes"]
    return summary

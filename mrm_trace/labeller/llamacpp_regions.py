"""
llama-cpp-python memory region extraction for perf-mem region pre-classification.

perf mem captures load/store events — there are no malloc/free pairs, and the
pip wheel of llama-cpp-python ships stripped libraries.  The memory layout is
therefore recovered from /proc/self/maps.

This module supplies the two model-data claims; everything else is classified as
``runtime`` by :mod:`mrm_trace.labeller.proc_maps`, so no address is left for the
labeller to guess at.

  weight    : GGUF file is mmap'd read-only — entries with the model filename
  kv_cache  : a new anonymous mapping whose size matches the KV cache implied by
              the GGUF geometry.  Size alone is NOT evidence — see below.

Usage::

    pre = snapshot_proc_maps()       # BEFORE Llama() construction
    llm = Llama(model_path=...)      # Llama() allocates weights + KV cache
    kv = expected_kv_bytes(llm.metadata, n_ctx)
    regions = extract_llamacpp_regions(model_path, pre, expected_kv=kv)
    llm("warm up")                   # threadpool + compute buffers appear here
    regions += extract_llamacpp_regions(model_path, pre, expected_kv=kv)
    labeller.register_address_ranges(regions)

The second call is not optional in practice: llama.cpp creates its threadpool at
generation time, so the worker-thread stacks do not exist yet at the first call.
``register_address_ranges`` resolves overlaps by evidence strength, so
overlapping snapshots are safe to concatenate.
"""

from __future__ import annotations

import pathlib
from typing import List, Optional

from mrm_trace.labeller.proc_maps import (
    MAX_REGION_PAGES,
    classify_mapping_source,
    is_guard,
    parse_proc_maps,
    snapshot_proc_maps,
    summarise_regions,
)

_MIN_KV_BYTES = 2 * 1024 * 1024  # 2 MB — ignore tiny anonymous regions

# A mapping counts as the KV cache only if its size matches the size implied by
# the model geometry.  Allocators round up and llama.cpp adds padding, so the
# window is loose — but it is still a check, which "large and new" was not.
_KV_SIZE_TOLERANCE = (0.75, 2.0)

__all__ = [
    "expected_kv_bytes",
    "extract_llamacpp_regions",
    "snapshot_proc_maps",
    "summarise_regions",
]


def expected_kv_bytes(
    gguf_metadata: dict, n_ctx: int, bytes_per_element: int = 2
) -> Optional[int]:
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


def _is_kv_sized(size: int, expected_kv: Optional[int]) -> bool:
    """
    True if *size* matches the KV cache implied by the model geometry.

    A new anonymous mapping that is merely *large* is NOT the KV cache.  On a
    Llama-3.2-1B / ctx=512 run the post-load mappings are 262 MB, 778 MB, 536 MB
    and 2 MB, while the geometry implies 16.8 MB — llama.cpp carves the KV cache
    out of a larger backend arena, so it is not separable at page granularity.
    Labelling those arenas ``kv_cache`` produces confident, wrong KV lifecycle
    data; without a size match they stay ``runtime`` so the gap remains visible.
    """
    if not expected_kv:
        return False
    lo, hi = _KV_SIZE_TOLERANCE
    return lo * expected_kv <= size <= hi * expected_kv


def extract_llamacpp_regions(
    model_path: str,
    pre_load_maps: List[dict],
    min_kv_bytes: int = _MIN_KV_BYTES,
    max_region_pages: int = MAX_REGION_PAGES,
    expected_kv: Optional[int] = None,
) -> List[dict]:
    """
    Return address regions for the current process, classified into the spec's
    region classes.  Every mapping is classified; nothing is left to guess at.

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
    post_maps = parse_proc_maps()
    pre_addrs = {(r["addr_start"], r["addr_end"]) for r in pre_load_maps}

    regions: List[dict] = []
    for i, entry in enumerate(post_maps):
        if entry["size"] >> 12 > max_region_pages:
            continue
        prev = post_maps[i - 1] if i > 0 else None
        nxt = post_maps[i + 1] if i + 1 < len(post_maps) else None

        path = entry["path"]
        if filename and (
            path.endswith(filename) or path.endswith(filename + " (deleted)")
        ):
            region_type, source = "weight", "mmap_gguf"
        else:
            source = classify_mapping_source(entry, prev, nxt, pre_addrs)
            region_type = "runtime"
            if (
                source == "anon_post_load"
                and entry["size"] >= min_kv_bytes
                and _is_kv_sized(entry["size"], expected_kv)
            ):
                region_type, source = "kv_cache", "anon_kv_sized"

        regions.append({
            "region_type": region_type,
            "addr_start": entry["addr_start"],
            "addr_end": entry["addr_end"],
            "size_bytes": entry["size"],
            "source": source,
        })
    return regions

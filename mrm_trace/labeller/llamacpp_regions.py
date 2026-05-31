"""
llama-cpp-python memory region extraction for perf-mem region pre-classification.

perf mem captures load/store events — there are no malloc/free pairs, and ggml_*
symbols are stripped in pip-installed llama-cpp-python.  But we can recover the
memory layout from /proc/self/maps:

  weight   : GGUF file is mmap'd read-only — entries with the model filename
  kv_cache : llama.cpp allocates the KV cache as one large anonymous mmap *after*
             model creation; comparing maps before vs after Llama() isolates it

Usage::

    pre = snapshot_proc_maps()       # BEFORE Llama() construction
    llm = Llama(model_path=...)      # Llama() allocates weights + KV cache
    regions = extract_llamacpp_regions(model_path, pre)
    labeller.register_address_ranges(regions)
"""

from __future__ import annotations

import pathlib
from typing import List

_MIN_KV_BYTES = 2 * 1024 * 1024  # 2 MB — ignore tiny anonymous regions


def _parse_proc_maps() -> List[dict]:
    """Return all /proc/self/maps entries as dicts."""
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


def extract_llamacpp_regions(
    model_path: str,
    pre_load_maps: List[dict],
    min_kv_bytes: int = _MIN_KV_BYTES,
) -> List[dict]:
    """
    Return weight and kv_cache address regions for a loaded llama-cpp-python model.

    Parameters
    ----------
    model_path:
        Path to the GGUF file (absolute or relative).
    pre_load_maps:
        Snapshot from snapshot_proc_maps() taken BEFORE Llama() was constructed.
    min_kv_bytes:
        Minimum size for an anonymous region to be classified as kv_cache.
        Smaller anonymous regions are skipped (Python allocator noise).

    Returns
    -------
    List of region dicts with keys: region_type, addr_start, addr_end, size_bytes, source.
    """
    filename = pathlib.Path(model_path).name
    post_maps = _parse_proc_maps()
    pre_addrs = {(r["addr_start"], r["addr_end"]) for r in pre_load_maps}

    regions: List[dict] = []

    for r in post_maps:
        path = r["path"]

        # weight: GGUF file mmap'd read-only (may appear as multiple segments)
        if filename and (path.endswith(filename) or path.endswith(filename + " (deleted)")):
            regions.append({
                "region_type": "weight",
                "addr_start": r["addr_start"],
                "addr_end": r["addr_end"],
                "size_bytes": r["size"],
                "source": "mmap_gguf",
            })
            continue

        # kv_cache: new anonymous RW region large enough to be a model buffer
        key = (r["addr_start"], r["addr_end"])
        if (
            key not in pre_addrs
            and r["perms"].startswith("rw")
            and not r["path"]
            and r["size"] >= min_kv_bytes
        ):
            regions.append({
                "region_type": "kv_cache",
                "addr_start": r["addr_start"],
                "addr_end": r["addr_end"],
                "size_bytes": r["size"],
                "source": "anon_post_load",
            })

    return regions

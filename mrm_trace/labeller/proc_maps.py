"""
Engine-agnostic /proc/self/maps classification.

Every mapping in the address space is accounted for as ``runtime`` — thread
stacks, the heap, malloc arenas, shared libraries, and anonymous allocations.
Model-specific extractors (llamacpp_regions, pytorch_regions) supply the
``weight`` and ``kv_cache`` ranges on top; the labeller's evidence ranking lets
those stronger claims win wherever they overlap.

Without this baseline, everything the model-specific extractors do not recognise
falls through to ``region_type=unknown`` — which on a perf mem trace is the large
majority of samples, since ~86% of store samples land on thread stacks.
``unknown`` should mean "this address is unaccounted for", not "nobody looked".
"""

from __future__ import annotations

from typing import Iterable, List, Optional

# A pthread stack is one anonymous rw mapping immediately preceded by a PROT_NONE
# guard mapping.  glibc's default is 8 MiB + a 4 KiB guard; musl and explicit
# pthread_attr_setstacksize() use other sizes, so the range is deliberately wide.
MIN_STACK_BYTES = 64 * 1024
MAX_STACK_BYTES = 64 * 1024 * 1024

# Refuse to page-expand absurdly large reservations (some runtimes reserve TB of
# PROT_NONE address space).
#
# 1 << 20 pages (4 GiB) was too low: a Q4_K_M GGUF for a 7B model is 1,067,581
# pages and an 8B is 1,201,352, so the weight mapping was silently skipped and
# those runs produced no `weight` region at all — 18 of 45 runs in one sweep,
# with nothing in the output to show it.  1 << 22 (16 GiB) covers any quantised
# model file this instrument targets while still refusing a TB-scale
# reservation.  The cost is the page-expansion dict: ~100 bytes per page, so
# 120 MB for an 8B model's mapping.
MAX_REGION_PAGES = 1 << 22

KERNEL_PATHS = frozenset({"[vvar]", "[vdso]", "[vsyscall]", "[vvar_vclock]"})


def parse_proc_maps() -> List[dict]:
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
    Snapshot /proc/self/maps.  Call this BEFORE loading a model so extractors can
    diff before vs after and tell model buffers from pre-existing allocations.
    """
    return parse_proc_maps()


def is_guard(entry: Optional[dict]) -> bool:
    """True if *entry* is an anonymous PROT_NONE mapping (a guard page)."""
    return (
        entry is not None
        and not entry["path"]
        and entry["perms"].startswith("---")
    )


def classify_anonymous_source(
    entry: dict,
    prev: Optional[dict],
    nxt: Optional[dict],
    is_new: bool,
) -> str:
    """
    Name the kind of anonymous mapping this is.  Everything here is ``runtime``;
    the source string is what makes a misclassification visible in the run log.

      thread stack : guard page immediately BELOW  (stack grows down into it)
      malloc arena : PROT_NONE tail immediately ABOVE (unused arena remainder)
      anon_post_load : appeared after the pre-load snapshot — a model buffer
                       candidate, but size alone is not evidence of what it holds
    """
    if not entry["perms"].startswith("rw"):
        return "anon_noaccess" if entry["perms"].startswith("---") else "anon_ro"
    if MIN_STACK_BYTES <= entry["size"] <= MAX_STACK_BYTES and is_guard(prev):
        return "thread_stack"
    if is_guard(nxt):
        return "malloc_arena"
    if is_new:
        return "anon_post_load"
    return "anon"


def classify_mapping_source(
    entry: dict,
    prev: Optional[dict],
    nxt: Optional[dict],
    pre_addrs: set,
) -> str:
    """Return the source label for one mapping; region_type is always runtime."""
    path = entry["path"]
    if path == "[stack]":
        return "main_stack"
    if path == "[heap]":
        return "heap"
    if path in KERNEL_PATHS:
        return "kernel_map"
    if path.startswith("["):
        return "special_map"
    if path:
        return "file_mapping"
    is_new = (entry["addr_start"], entry["addr_end"]) not in pre_addrs
    return classify_anonymous_source(entry, prev, nxt, is_new)


def classify_runtime_maps(
    pre_load_maps: Iterable[dict] = (),
    max_region_pages: int = MAX_REGION_PAGES,
) -> List[dict]:
    """
    Classify every current mapping as ``runtime``, with a descriptive source.

    Engine-agnostic: no model file, no KV geometry.  Register these alongside a
    model-specific extractor's output — the labeller ranks GGUF/tensor evidence
    above these coarse labels, so overlaps resolve in favour of the model data.

    Returns region dicts with keys: region_type, addr_start, addr_end,
    size_bytes, source.
    """
    maps = parse_proc_maps()
    pre_addrs = {(r["addr_start"], r["addr_end"]) for r in pre_load_maps}

    regions: List[dict] = []
    for i, entry in enumerate(maps):
        if entry["size"] >> 12 > max_region_pages:
            continue
        prev = maps[i - 1] if i > 0 else None
        nxt = maps[i + 1] if i + 1 < len(maps) else None
        regions.append({
            "region_type": "runtime",
            "addr_start": entry["addr_start"],
            "addr_end": entry["addr_end"],
            "size_bytes": entry["size"],
            "source": classify_mapping_source(entry, prev, nxt, pre_addrs),
        })
    return regions


def summarise_regions(regions: Iterable[dict]) -> "dict[str, dict]":
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

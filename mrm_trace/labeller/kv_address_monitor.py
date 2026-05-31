"""
KV cache address monitor for HuggingFace transformers (PyTorch backend).

perf mem provides only load/store events — there are no allocation events to
anchor KV cache addresses. This module patches ``DynamicCache.update`` so every
K/V tensor address (including intermediate tensors created by torch.cat at each
generation step) is recorded with a nanosecond timestamp.

The captured ranges are written to ``kv_ranges.json`` alongside the generation
results.  The outer process reads them and pre-registers the address pages with
``TraceLabeller.register_address_ranges(..., region_type="kv_cache")`` so perf
samples hitting those pages are classified as ``kv_cache`` instead of ``unknown``.

Why intermediate addresses matter:
    Each DynamicCache.update call does ``torch.cat([past_k, new_k], dim=-2)``,
    which allocates a NEW tensor at a new address and frees the old one.  The
    perf trace captures accesses to ALL those intermediate addresses during the
    generation pass — only the final tensor survives to the end.  Recording every
    address the cache ever holds ensures complete coverage.
"""

from __future__ import annotations

import time
from typing import List


# ── Module-level capture state ────────────────────────────────────────────────

_kv_ranges: List[dict] = []
_original_update = None
_installed = False


def install_transformers_kv_monitor() -> None:
    """
    Monkey-patch ``DynamicCache.update`` to record every K/V tensor address.

    Safe to call multiple times — installs only once per process.
    Call ``uninstall_transformers_kv_monitor()`` to restore the original.
    """
    global _installed, _original_update, _kv_ranges

    if _installed:
        return

    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        return  # transformers not installed or pre-4.36 (no DynamicCache)

    _kv_ranges = []
    _original_update = DynamicCache.update

    def _patched_update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        result = _original_update(self, key_states, value_states, layer_idx, cache_kwargs)
        ts = time.time_ns()
        k_out, v_out = result
        for tensor, kind in ((k_out, "key"), (v_out, "value")):
            try:
                if tensor.device.type != "cpu":
                    continue
                ptr = tensor.data_ptr()
                if ptr == 0:
                    continue
                _kv_ranges.append({
                    "layer": layer_idx,
                    "type": kind,
                    "region_type": "kv_cache",
                    "addr_start": ptr,
                    "addr_end": ptr + tensor.nbytes,
                    "size_bytes": tensor.nbytes,
                    "captured_ns": ts,
                })
            except Exception:
                pass
        return result

    DynamicCache.update = _patched_update
    _installed = True


def uninstall_transformers_kv_monitor() -> None:
    """Restore the original ``DynamicCache.update``."""
    global _installed, _original_update
    if not _installed or _original_update is None:
        return
    try:
        from transformers.cache_utils import DynamicCache
        DynamicCache.update = _original_update
    except ImportError:
        pass
    _installed = False


def collect_kv_ranges() -> List[dict]:
    """
    Return deduplicated KV cache address ranges captured since last install.

    Deduplication is by (addr_start, layer, type) — if torch.cat produced the
    same pointer multiple times (rare but possible with memory reuse), we keep
    only the first occurrence.
    """
    seen: set = set()
    out: List[dict] = []
    for r in _kv_ranges:
        key = (r["addr_start"], r["layer"], r["type"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out

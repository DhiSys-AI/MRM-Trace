"""
Unit tests for /proc/self/maps region classification.

The discrimination that matters here is thread stack vs KV cache: both are new
anonymous rw mappings of several MB, and a size-only rule labels the worker
stacks as kv_cache — which then dominates kv_block_lifecycle.parquet with data
that has nothing to do with the KV cache.
"""

import pytest

from mrm_trace.labeller.llamacpp_regions import (
    expected_kv_bytes,
    extract_llamacpp_regions,
    summarise_regions,
)

pytestmark = pytest.mark.unit

_PAGE = 4096
_MB = 1024 * 1024


def _entry(start, size, perms="rw-p", path=""):
    return {
        "addr_start": start,
        "addr_end": start + size,
        "size": size,
        "perms": perms,
        "path": path,
    }


def _extract(monkeypatch, post_maps, pre_maps=(), model_path="models/tiny.gguf", **kw):
    monkeypatch.setattr(
        "mrm_trace.labeller.llamacpp_regions._parse_proc_maps",
        lambda: list(post_maps),
    )
    return extract_llamacpp_regions(model_path, list(pre_maps), **kw)


def _by_start(regions):
    return {r["addr_start"]: r for r in regions}


# ── weight ────────────────────────────────────────────────────────────────────

def test_gguf_mmap_classified_as_weight(monkeypatch):
    maps = [_entry(0x1000_0000, 800 * _MB, "r--p", "/home/u/models/tiny.gguf")]
    regions = _extract(monkeypatch, maps)
    assert regions[0]["region_type"] == "weight"
    assert regions[0]["source"] == "mmap_gguf"


def test_deleted_gguf_mmap_still_weight(monkeypatch):
    maps = [_entry(0x1000_0000, 8 * _MB, "r--p", "/models/tiny.gguf (deleted)")]
    assert _extract(monkeypatch, maps)[0]["region_type"] == "weight"


# ── kv_cache vs the things that look like it ──────────────────────────────────

def test_new_anon_matching_model_geometry_is_kv_cache(monkeypatch):
    maps = [_entry(0x2000_0000, 16 * _MB)]
    regions = _extract(monkeypatch, maps, expected_kv=16 * _MB)
    assert regions[0]["region_type"] == "kv_cache"
    assert regions[0]["source"] == "anon_kv_sized"


def test_new_large_anon_is_not_kv_cache_without_geometry(monkeypatch):
    """Size alone is not evidence — a wrong KV label is worse than an absent one."""
    regions = _extract(monkeypatch, [_entry(0x2000_0000, 16 * _MB)])
    assert regions[0]["region_type"] == "runtime"
    assert regions[0]["source"] == "anon_post_load"


def test_backend_arena_far_larger_than_kv_geometry_is_not_kv_cache(monkeypatch):
    """The real case: ctx=512 on a 1B model implies 16.8 MB, arenas are 262-778 MB."""
    maps = [_entry(0x2000_0000, 778 * _MB), _entry(0x8000_0000, 262 * _MB)]
    regions = _extract(monkeypatch, maps, expected_kv=16_777_216)
    assert all(r["region_type"] == "runtime" for r in regions)
    assert all(r["source"] == "anon_post_load" for r in regions)


def test_thread_stack_is_runtime_not_kv_cache(monkeypatch):
    """A guard page immediately below an 8 MB rw mapping is a pthread stack."""
    guard = _entry(0x7B6B_5000_0000, _PAGE, "---p")
    stack = _entry(0x7B6B_5000_0000 + _PAGE, 8 * _MB)
    regions = _by_start(_extract(monkeypatch, [guard, stack]))
    assert regions[stack["addr_start"]]["region_type"] == "runtime"
    assert regions[stack["addr_start"]]["source"] == "thread_stack"


def test_malloc_arena_is_runtime_not_kv_cache(monkeypatch):
    """A PROT_NONE tail immediately above marks an unused glibc arena remainder."""
    arena = _entry(0x7B6B_6000_0000, 4 * _MB)
    tail = _entry(0x7B6B_6000_0000 + 4 * _MB, 60 * _MB, "---p")
    regions = _by_start(_extract(monkeypatch, [arena, tail]))
    assert regions[arena["addr_start"]]["source"] == "malloc_arena"


def test_preexisting_anon_is_not_kv_cache(monkeypatch):
    """Only mappings that appeared after the pre-load snapshot can be KV cache."""
    entry = _entry(0x2000_0000, 16 * _MB)
    regions = _extract(monkeypatch, [entry], pre_maps=[entry], expected_kv=16 * _MB)
    assert regions[0]["region_type"] == "runtime"
    assert regions[0]["source"] == "anon"


def test_small_new_anon_is_runtime(monkeypatch):
    regions = _extract(monkeypatch, [_entry(0x2000_0000, 64 * 1024)])
    assert regions[0]["region_type"] == "runtime"


def test_stack_sized_mapping_without_guard_is_not_a_stack(monkeypatch):
    """The guard page is the signal — size alone must not decide."""
    regions = _extract(monkeypatch, [_entry(0x7B6B_5000_0000, 8 * _MB)],
                       expected_kv=8 * _MB)
    assert regions[0]["region_type"] == "kv_cache"


def test_guarded_stack_is_never_kv_cache_even_at_matching_size(monkeypatch):
    """A stack that happens to match the KV geometry must still read as a stack."""
    guard = _entry(0x7B6B_5000_0000, _PAGE, "---p")
    stack = _entry(0x7B6B_5000_0000 + _PAGE, 8 * _MB)
    regions = _by_start(_extract(monkeypatch, [guard, stack], expected_kv=8 * _MB))
    assert regions[stack["addr_start"]]["source"] == "thread_stack"


# ── runtime: everything else is accounted for, nothing is left unknown ────────

@pytest.mark.parametrize("path,source", [
    ("[stack]", "main_stack"),
    ("[heap]", "heap"),
    ("[vdso]", "kernel_map"),
    ("[vvar]", "kernel_map"),
    ("[anon:something]", "special_map"),
    ("/usr/lib/x86_64-linux-gnu/libc.so.6", "file_mapping"),
])
def test_non_model_mappings_are_runtime(monkeypatch, path, source):
    regions = _extract(monkeypatch, [_entry(0x3000_0000, 2 * _MB, "r-xp", path)])
    assert regions[0]["region_type"] == "runtime"
    assert regions[0]["source"] == source


class TestExpectedKvBytes:
    """2 × n_layer × n_ctx × n_head_kv × head_dim × bytes_per_element."""

    LLAMA_3_2_1B = {
        "llama.block_count": "16",
        "llama.attention.head_count_kv": "8",
        "llama.attention.key_length": "64",
        "llama.embedding_length": "2048",
        "llama.attention.head_count": "32",
    }

    def test_matches_llama_3_2_1b_geometry(self):
        # 2 * 16 * 512 * 8 * 64 * 2 = 16,777,216
        assert expected_kv_bytes(self.LLAMA_3_2_1B, 512) == 16_777_216

    def test_scales_linearly_with_context(self):
        at_512 = expected_kv_bytes(self.LLAMA_3_2_1B, 512)
        assert expected_kv_bytes(self.LLAMA_3_2_1B, 2048) == 4 * at_512

    def test_derives_head_dim_when_key_length_absent(self):
        md = {k: v for k, v in self.LLAMA_3_2_1B.items() if "key_length" not in k}
        assert expected_kv_bytes(md, 512) == 16_777_216   # 2048 / 32 = 64

    def test_quantised_kv_halves_the_size(self):
        assert expected_kv_bytes(self.LLAMA_3_2_1B, 512, bytes_per_element=1) == 8_388_608

    def test_returns_none_when_geometry_is_unavailable(self):
        assert expected_kv_bytes({"general.name": "x"}, 512) is None

    def test_returns_none_for_zero_context(self):
        assert expected_kv_bytes(self.LLAMA_3_2_1B, 0) is None


def test_every_mapping_is_classified(monkeypatch):
    """No mapping may be dropped — an address we saw must never become 'unknown'."""
    maps = [
        _entry(0x1000_0000, 8 * _MB, "r--p", "/models/tiny.gguf"),
        _entry(0x2000_0000, 16 * _MB),
        _entry(0x3000_0000, _PAGE, "---p"),
        _entry(0x3000_1000, 8 * _MB),
        _entry(0x4000_0000, 128 * 1024, "r-xp", "/usr/lib/libfoo.so"),
        _entry(0x5000_0000, 132 * 1024, "rw-p", "[heap]"),
    ]
    regions = _extract(monkeypatch, maps)
    assert len(regions) == len(maps)
    assert all(r["region_type"] in {"weight", "kv_cache", "runtime"} for r in regions)


def test_oversized_reservation_is_skipped(monkeypatch):
    """A TB-scale PROT_NONE reservation must not be expanded page-by-page."""
    maps = [
        _entry(0x1000_0000_0000, 1 << 42, "---p"),   # 4 TB
        _entry(0x2000_0000, 16 * _MB),
    ]
    regions = _extract(monkeypatch, maps)
    assert len(regions) == 1
    assert regions[0]["addr_start"] == 0x2000_0000


# ── summary helper ────────────────────────────────────────────────────────────

def test_summarise_regions_aggregates_by_source(monkeypatch):
    guard1 = _entry(0x7B6B_0000_0000, _PAGE, "---p")
    stack1 = _entry(0x7B6B_0000_1000, 8 * _MB)
    guard2 = _entry(0x7B6B_0801_0000, _PAGE, "---p")
    stack2 = _entry(0x7B6B_0801_1000, 8 * _MB)
    summary = summarise_regions(_extract(monkeypatch, [guard1, stack1, guard2, stack2]))
    assert summary["thread_stack"]["count"] == 2
    assert summary["thread_stack"]["region_type"] == "runtime"
    assert summary["thread_stack"]["bytes"] == 16 * _MB

"""
Engine-agnostic /proc/self/maps classification.

The baseline exists so that region_type=unknown means "this address is in no
mapping we saw", not "no extractor recognised it". Everything here is runtime;
model-data claims come from the engine-specific extractors and outrank these.
"""

import pytest

from mrm_trace.labeller.proc_maps import (
    classify_runtime_maps,
    parse_proc_maps,
    snapshot_proc_maps,
    summarise_regions,
)

pytestmark = pytest.mark.unit

_PAGE = 4096
_MB = 1024 * 1024


def _entry(start, size, perms="rw-p", path=""):
    return {
        "addr_start": start, "addr_end": start + size,
        "size": size, "perms": perms, "path": path,
    }


def _classify(monkeypatch, maps, pre=()):
    monkeypatch.setattr("mrm_trace.labeller.proc_maps.parse_proc_maps",
                        lambda: list(maps))
    return classify_runtime_maps(list(pre))


def test_every_mapping_is_runtime(monkeypatch):
    maps = [
        _entry(0x1000_0000, 2 * _MB, "r-xp", "/usr/lib/libc.so.6"),
        _entry(0x2000_0000, 16 * _MB),
        _entry(0x3000_0000, 132 * 1024, "rw-p", "[heap]"),
        _entry(0x4000_0000, 132 * 1024, "rw-p", "[stack]"),
    ]
    regions = _classify(monkeypatch, maps)
    assert len(regions) == len(maps)
    assert {r["region_type"] for r in regions} == {"runtime"}


def test_sources_distinguish_mapping_kinds(monkeypatch):
    guard = _entry(0x7B00_0000_0000, _PAGE, "---p")
    stack = _entry(0x7B00_0000_0000 + _PAGE, 8 * _MB)
    maps = [
        guard, stack,
        _entry(0x3000_0000, 2 * _MB, "rw-p", "[heap]"),
        _entry(0x4000_0000, 2 * _MB, "r-xp", "/usr/lib/libfoo.so"),
        _entry(0x5000_0000, 2 * _MB, "rw-p", "[stack]"),
        _entry(0x6000_0000, 2 * _MB, "rw-p", "[vdso]"),
    ]
    by_source = {r["source"] for r in _classify(monkeypatch, maps)}
    assert {"thread_stack", "heap", "file_mapping", "main_stack",
            "kernel_map"} <= by_source


def test_new_anonymous_mapping_is_flagged_but_not_claimed(monkeypatch):
    """anon_post_load marks a model-buffer candidate; it stays runtime."""
    entry = _entry(0x2000_0000, 700 * _MB)
    region = _classify(monkeypatch, [entry])[0]
    assert region["region_type"] == "runtime"
    assert region["source"] == "anon_post_load"


def test_preexisting_anonymous_mapping_is_plain_anon(monkeypatch):
    entry = _entry(0x2000_0000, 700 * _MB)
    region = _classify(monkeypatch, [entry], pre=[entry])[0]
    assert region["source"] == "anon"


def test_oversized_reservation_is_skipped(monkeypatch):
    maps = [_entry(0x1000_0000_0000, 1 << 42, "---p"), _entry(0x2000_0000, 2 * _MB)]
    regions = _classify(monkeypatch, maps)
    assert [r["addr_start"] for r in regions] == [0x2000_0000]


def test_summarise_regions_groups_by_source(monkeypatch):
    maps = [_entry(0x3000_0000 + i * _MB, _MB, "rw-p", "[heap]") for i in range(3)]
    summary = summarise_regions(_classify(monkeypatch, maps))
    assert summary["heap"]["count"] == 3
    assert summary["heap"]["bytes"] == 3 * _MB


class TestAgainstTheRealProcess:
    """These run against this interpreter's actual address space."""

    def test_parse_returns_mappings(self):
        maps = parse_proc_maps()
        assert len(maps) > 5
        assert all(m["addr_end"] > m["addr_start"] for m in maps)

    def test_snapshot_matches_parse_shape(self):
        assert {k for k in snapshot_proc_maps()[0]} == {
            "addr_start", "addr_end", "size", "perms", "path"
        }

    def test_live_process_is_fully_classified(self):
        regions = classify_runtime_maps()
        assert regions
        assert all(r["region_type"] == "runtime" for r in regions)
        # A real process always has a main stack and file-backed libraries.
        sources = {r["source"] for r in regions}
        assert "main_stack" in sources
        assert "file_mapping" in sources

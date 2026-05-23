"""Unit tests for BaseCollector ABC and CollectorMeta dataclass."""

from pathlib import Path

import pytest

from mrm_trace.collector.base import BaseCollector, CollectorMeta


# ── CollectorMeta ──────────────────────────────────────────────────────────────

def test_collector_meta_defaults():
    meta = CollectorMeta(collector_type="test", available=True, used=True)
    assert meta.fallback_reason is None
    assert meta.event_strings == []
    assert meta.sample_period == 0
    assert meta.output_file is None
    assert meta.output_file_size_bytes == 0
    assert meta.parse_success is None
    assert meta.dropped_samples is None
    assert meta.coverage_estimate is None
    assert meta.warnings == []
    assert meta.extra == {}


def test_collector_meta_fields_stored():
    meta = CollectorMeta(
        collector_type="perf_mem",
        available=True,
        used=True,
        fallback_reason="test reason",
        event_strings=["cpu/mem-loads/P"],
        sample_period=4000,
        output_file="/tmp/perf.data",
        output_file_size_bytes=1024,
        parse_success=True,
        dropped_samples=0,
        coverage_estimate=0.99,
        warnings=["one warning"],
        extra={"key": "val"},
    )
    assert meta.collector_type == "perf_mem"
    assert meta.fallback_reason == "test reason"
    assert meta.event_strings == ["cpu/mem-loads/P"]
    assert meta.sample_period == 4000
    assert meta.output_file == "/tmp/perf.data"
    assert meta.output_file_size_bytes == 1024
    assert meta.parse_success is True
    assert meta.dropped_samples == 0
    assert meta.coverage_estimate == 0.99
    assert meta.warnings == ["one warning"]
    assert meta.extra == {"key": "val"}


def test_collector_meta_warnings_default_is_mutable_safe():
    a = CollectorMeta(collector_type="a", available=True, used=True)
    b = CollectorMeta(collector_type="b", available=True, used=True)
    a.warnings.append("x")
    assert b.warnings == []


def test_collector_meta_extra_default_is_mutable_safe():
    a = CollectorMeta(collector_type="a", available=True, used=True)
    b = CollectorMeta(collector_type="b", available=True, used=True)
    a.extra["k"] = "v"
    assert b.extra == {}


# ── BaseCollector ──────────────────────────────────────────────────────────────

def test_base_collector_is_abstract():
    with pytest.raises(TypeError):
        BaseCollector()  # type: ignore[abstract]


def test_missing_stop_prevents_instantiation():
    class _Incomplete(BaseCollector):
        @property
        def name(self):
            return "x"
        def is_available(self):
            return True
        def start(self, pids, output_dir):
            pass
        # stop() is missing

    with pytest.raises(TypeError):
        _Incomplete()


def test_minimal_concrete_collector_instantiates():
    class _Minimal(BaseCollector):
        @property
        def name(self):
            return "minimal"
        def is_available(self):
            return True
        def start(self, pids, output_dir):
            pass
        def stop(self):
            return CollectorMeta("minimal", True, True)

    c = _Minimal()
    assert c.name == "minimal"
    assert c.is_available() is True
    meta = c.stop()
    assert meta.collector_type == "minimal"

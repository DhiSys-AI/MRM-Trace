"""
Phase annotation must survive the clock domain the collector actually uses.

perf script emits CLOCK_MONOTONIC timestamps (seconds since boot, ~2e12 ns).
Phase records carry both wall_ns (time.time_ns(), the UNIX epoch, ~1.78e18 ns)
and mono_ns.  Building phase boundaries from wall_ns puts every window decades
beyond any sample, so _annotate_phase matches nothing and the whole trace is
written with phase=NULL — silently disabling the phase-aware labelling that
ADR-3 and spec §5 require.  Nothing else fails; the run looks healthy.

These tests pin the clock domain and the gate that catches a regression.
"""

import importlib.util
import io
import pathlib
import time

import pandas as pd
import pytest

from mrm_trace.parser.perf_script_parser import parse_perf_script
from mrm_trace.reporter.trace_qa import check_trace_quality

pytestmark = pytest.mark.validity

_DEMO = pathlib.Path(__file__).resolve().parents[2] / "notebooks" / "scripts" / "demo_perf_mem.py"


def _load_demo():
    """Import demo_perf_mem.py by path (notebooks/ is not an installed package)."""
    spec = importlib.util.spec_from_file_location("demo_perf_mem", _DEMO)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _perf_line(mono_s: float) -> str:
    return (f"          python    9608  {mono_s:.6f}:         cpu/mem-stores/P: "
            f"7b6c02391270      1e05080144 |OP STORE|LVL L1 hit|SNP N/A|TLB N/A|"
            f"LCK N/A|BLK  N/A               0\n")


def _phase_records(mono_base: int):
    """Phase records as demo_perf_mem serialises them, both clocks present."""
    wall_base = time.time_ns()
    events = [
        ("model_load_start", 0), ("model_load_end", 2_000_000_000),
        ("generation_start", 2_000_000_000), ("generation_end", 6_000_000_000),
    ]
    return [
        {"event": ev, "wall_ns": wall_base + off, "mono_ns": mono_base + off}
        for ev, off in events
    ]


def test_phase_boundaries_use_the_monotonic_clock():
    demo = _load_demo()
    mono_base = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    boundaries = demo.build_phase_boundaries(_phase_records(mono_base))

    assert {b["phase"] for b in boundaries} == {"weight_load", "generation"}
    # Boundaries must sit in the boot-relative domain, not the epoch domain.
    for b in boundaries:
        assert b["start_ns"] < time.time_ns() / 1000, (
            "phase boundary is in the epoch domain — perf samples can never match it"
        )


def test_samples_receive_a_phase_label():
    demo = _load_demo()
    mono_base = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    boundaries = demo.build_phase_boundaries(_phase_records(mono_base))

    text = "".join(
        _perf_line((mono_base + off) / 1e9)
        for off in (500_000_000, 3_000_000_000, 5_000_000_000)
    )
    rows = list(parse_perf_script(io.StringIO(text), run_id="r1",
                                  phase_boundaries=boundaries))

    assert [r["phase"] for r in rows] == ["weight_load", "generation", "generation"]


def test_stale_records_without_mono_ns_are_rejected():
    """Old results files can't be retro-phased — fail loudly instead of yielding NULLs."""
    demo = _load_demo()
    stale = [{"event": "model_load_start", "wall_ns": time.time_ns()}]
    with pytest.raises(ValueError, match="mono_ns"):
        demo.build_phase_boundaries(stale)


def test_quality_gate_fails_when_phase_is_entirely_null():
    trace = pd.DataFrame({
        "access_latency_ns": [10, 20, 30],
        "cache_level": ["L1", "L2", "L3"],
        "symbol": ["", "", ""],
        "region_type": ["weight", "weight", "kv_cache"],
        "phase": [None, None, None],
    })
    qa = check_trace_quality(trace, collector_type="perf_mem_labelled",
                             print_report=False)
    assert qa.status == "FAIL"
    assert qa.phase_coverage == 0.0
    assert any("phase" in issue for issue in qa.issues)


def test_quality_gate_passes_when_phase_is_populated():
    trace = pd.DataFrame({
        "access_latency_ns": [10, 20, 30],
        "cache_level": ["L1", "L2", "L3"],
        "symbol": ["", "", ""],
        "region_type": ["weight", "weight", "kv_cache"],
        "phase": ["weight_load", "generation", "generation"],
    })
    qa = check_trace_quality(trace, collector_type="perf_mem_labelled",
                             print_report=False)
    assert qa.status == "PASS"
    assert qa.phase_coverage == 1.0

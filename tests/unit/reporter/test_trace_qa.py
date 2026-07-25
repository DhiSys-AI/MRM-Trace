"""Unit tests for mrm_trace/reporter/trace_qa.py"""

import pytest
import pandas as pd

from mrm_trace.reporter.trace_qa import (
    TraceQA,
    _bar,
    _check_column,
    _merge_status,
    check_trace_quality,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_trace(n, *, lat=1.0, cl=1.0, sym=0.0, reg=0.0):
    n_lat = int(n * lat)
    n_cl = int(n * cl)
    n_sym = int(n * sym)
    n_reg = int(n * reg)
    return pd.DataFrame({
        "access_latency_ns": [100] * n_lat + [None] * (n - n_lat),
        "cache_level":       ["L1"] * n_cl  + [None] * (n - n_cl),
        "symbol":            ["ggml_f"] * n_sym + [""] * (n - n_sym),
        "region_type":       ["weight"] * n_reg + ["unknown"] * (n - n_reg),
    })


# ── TraceQA dataclass ─────────────────────────────────────────────────────────

class TestTraceQA:
    def test_ok_pass(self):
        assert TraceQA("PASS", 100, 1.0, 1.0, 0.0, 0.0, []).ok

    def test_ok_warn(self):
        assert TraceQA("WARN", 100, 0.7, 1.0, 0.0, 0.0, ["x"]).ok

    def test_ok_fail(self):
        assert not TraceQA("FAIL", 0, 0.0, 0.0, 0.0, 0.0, ["no rows"]).ok


# ── _check_column ─────────────────────────────────────────────────────────────

class TestCheckColumn:
    _thresholds = {"col": (0.95, 0.50)}

    def test_above_pass_min(self):
        issues = []
        assert _check_column("col", 1.0, self._thresholds, issues) == "PASS"
        assert issues == []

    def test_between_warn_and_pass(self):
        issues = []
        assert _check_column("col", 0.70, self._thresholds, issues) == "WARN"
        assert "col" in issues[0]

    def test_below_warn_min(self):
        issues = []
        assert _check_column("col", 0.10, self._thresholds, issues) == "FAIL"
        assert "col" in issues[0]

    def test_missing_key_returns_pass(self):
        issues = []
        assert _check_column("absent", 0.0, self._thresholds, issues) == "PASS"
        assert issues == []


# ── _merge_status ─────────────────────────────────────────────────────────────

class TestMergeStatus:
    def test_pass_pass(self):   assert _merge_status("PASS", "PASS") == "PASS"
    def test_pass_warn(self):   assert _merge_status("PASS", "WARN") == "WARN"
    def test_pass_fail(self):   assert _merge_status("PASS", "FAIL") == "FAIL"
    def test_warn_fail(self):   assert _merge_status("WARN", "FAIL") == "FAIL"
    def test_fail_pass(self):   assert _merge_status("FAIL", "PASS") == "FAIL"
    def test_fail_warn(self):   assert _merge_status("FAIL", "WARN") == "FAIL"


# ── _bar ──────────────────────────────────────────────────────────────────────

class TestBar:
    def test_full(self):
        b = _bar(1.0)
        assert "100.0%" in b
        assert "░" not in b

    def test_empty(self):
        b = _bar(0.0)
        assert "0.0%" in b
        assert "█" not in b

    def test_half(self):
        b = _bar(0.5)
        assert "50.0%" in b


# ── check_trace_quality ───────────────────────────────────────────────────────

class TestCheckTraceQuality:
    def test_empty_trace_is_fail(self):
        qa = check_trace_quality(pd.DataFrame(), print_report=False)
        assert qa.status == "FAIL"
        assert qa.row_count == 0
        assert not qa.ok

    def test_perfect_perf_mem_pass(self):
        qa = check_trace_quality(_make_trace(1000), collector_type="perf_mem", print_report=False)
        assert qa.status == "PASS"
        assert qa.row_count == 1000
        assert qa.latency_coverage == pytest.approx(1.0)
        assert qa.cache_level_coverage == pytest.approx(1.0)

    def test_low_latency_is_fail(self):
        qa = check_trace_quality(_make_trace(1000, lat=0.3), collector_type="perf_mem", print_report=False)
        assert qa.status == "FAIL"
        assert any("access_latency_ns" in i for i in qa.issues)

    def test_partial_latency_is_warn(self):
        qa = check_trace_quality(_make_trace(1000, lat=0.70), collector_type="perf_mem", print_report=False)
        assert qa.status == "WARN"

    def test_memray_passes_with_zero_latency(self):
        qa = check_trace_quality(_make_trace(500, lat=0.0, cl=0.0), collector_type="memray", print_report=False)
        assert qa.status == "PASS"

    def test_perf_mem_labelled_fails_without_region(self):
        qa = check_trace_quality(_make_trace(1000, reg=0.0), collector_type="perf_mem_labelled", print_report=False)
        assert qa.status == "FAIL"
        assert any("region_classified" in i for i in qa.issues)

    def test_perf_mem_labelled_passes_with_region(self):
        qa = check_trace_quality(_make_trace(1000, reg=0.10), collector_type="perf_mem_labelled", print_report=False)
        assert qa.status == "PASS"

    def test_symbol_coverage_computed(self):
        qa = check_trace_quality(_make_trace(100, sym=0.5), collector_type="perf_mem", print_report=False)
        assert qa.symbol_coverage == pytest.approx(0.5)

    def test_region_coverage_computed(self):
        qa = check_trace_quality(_make_trace(100, reg=0.3), collector_type="perf_mem", print_report=False)
        assert qa.region_classified == pytest.approx(0.3)

    def test_fail_fast_exits(self):
        df = _make_trace(100, lat=0.0, cl=0.0)
        with pytest.raises(SystemExit):
            check_trace_quality(df, collector_type="perf_mem", print_report=False, fail_fast=True)

    def test_fail_fast_does_not_exit_on_pass(self):
        qa = check_trace_quality(_make_trace(100), collector_type="perf_mem", print_report=False, fail_fast=True)
        assert qa.ok

    def test_print_report_pass(self, capsys):
        check_trace_quality(_make_trace(200), collector_type="perf_mem", print_report=True)
        out = capsys.readouterr().out
        assert "perf_mem" in out
        assert "PASS" in out

    def test_print_report_fail_shows_issues(self, capsys):
        check_trace_quality(_make_trace(200, lat=0.0, cl=0.0), collector_type="perf_mem", print_report=True)
        out = capsys.readouterr().out
        assert "FAIL" in out
        assert "access_latency_ns" in out

    def test_unknown_collector_uses_default(self):
        qa = check_trace_quality(_make_trace(100), collector_type="other", print_report=False)
        assert qa.status in ("PASS", "WARN", "FAIL")

    def test_trace_without_symbol_column(self):
        df = _make_trace(100).drop(columns=["symbol"])
        qa = check_trace_quality(df, collector_type="perf_mem", print_report=False)
        assert qa.symbol_coverage == 0.0

    def test_trace_without_region_column(self):
        df = _make_trace(100).drop(columns=["region_type"])
        qa = check_trace_quality(df, collector_type="perf_mem", print_report=False)
        assert qa.region_classified == 0.0

    def test_empty_trace_fail_fast_exits(self):
        with pytest.raises(SystemExit):
            check_trace_quality(pd.DataFrame(), print_report=False, fail_fast=True)

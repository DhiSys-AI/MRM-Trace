"""
Symbol resolution depends on ``ip`` being in the ``perf script -F`` list.

perf silently omits BOTH sym and dso when ip is absent — no error, no
placeholder — which is why symbol coverage read 0.0% on every run regardless of
whether llama.cpp carried debug symbols. Adding ip also appends ``ip sym (dso)``
after the weight, and the decoded data_src block ends with ``|BLK  N/A``, so its
final ``|`` sits *inside* the block. Splitting on the last ``|`` therefore puts
the weight in the middle of the remaining tokens and drops access_latency_ns for
every row.

The fixtures below are genuine `perf script` output captured on the partner
machine (perf 7.0.12), not hand-written approximations.
"""

import io

import pytest

from mrm_trace.collector.perf_runner import PerfMemRunner
from mrm_trace.parser.perf_script_parser import parse_perf_script

pytestmark = pytest.mark.validity

# Resolved user-space symbol in a shared library.
_RESOLVED = (
    "          python   18407  7030.495004: cpu/mem-loads,ldlat=30/P:           "
    "703d36     11a68201042 |OP LOAD|LVL RAM hit|SNP Hit|TLB L1 or L2 hit|LCK No|"
    "BLK  N/A             318     7039e379badc __strlen_evex "
    "(/usr/lib/x86_64-linux-gnu/libc.so.6)\n"
)

# Unresolved symbol but a known dso.
_UNRESOLVED_SYM = (
    "          python   18407  7030.496137: cpu/mem-loads,ldlat=30/P:     "
    "7039e354d1b8     11868100242 |OP LOAD|LVL LFB/MAB hit|SNP None|"
    "TLB L1 or L2 hit|LCK No|BLK  N/A             162           61c8fc "
    "[unknown] (/usr/bin/python3.12)\n"
)

# Store sample with a kernel ip — neither sym nor dso resolvable.
_UNRESOLVED_BOTH = (
    "          python   18407  7030.557339:         cpu/mem-stores/P:                "
    "0      1e05080144 |OP STORE|LVL L1 hit|SNP N/A|TLB N/A|LCK N/A|"
    "BLK  N/A                          158 ffffffffa2ef7b50 [unknown] ([unknown])\n"
)

# Store sample with a resolved user-space ip.
_RESOLVED_STORE = (
    "          python   18407  7030.506390:         cpu/mem-stores/P:     "
    "7ffec1d15380      1e05080184 |OP STORE|LVL L1 miss|SNP N/A|TLB N/A|"
    "LCK N/A|BLK  N/A                           0           5d7799 "
    "_PyEval_EvalFrameDefault (/usr/bin/python3.12)\n"
)

# Genuine kernel-space DATA address — dropped, user-space samples only.
_KERNEL_DATA_ADDR = (
    "       perf-exec   18407  7030.492199:         cpu/mem-stores/P: "
    "ffffd3b7429b7af8      1e05080144 |OP STORE|LVL L1 hit|SNP N/A|TLB N/A|"
    "LCK N/A|BLK  N/A               0 ffffffffa32a1865 [unknown] ([unknown])\n"
)

# Same collector, before ip was added to -F: no sym/dso fields at all.
_PRE_IP_FORMAT = (
    "          python   18156  6838.720099:         cpu/mem-stores/P: "
    "7b6c02391270      1e05080144 |OP STORE|LVL L1 hit|SNP N/A|TLB N/A|"
    "LCK N/A|BLK  N/A               0\n"
)


def _parse(text):
    return list(parse_perf_script(io.StringIO(text), run_id="r1"))


class TestCollectorRequestsIp:
    def test_field_list_includes_ip(self):
        """Without ip, perf emits neither sym nor dso — symbol rules go dead."""
        import inspect
        source = inspect.getsource(PerfMemRunner._run_perf_script)
        assert "ip," in source, "perf script -F list must request ip"
        assert "sym,dso" in source


class TestIpFormatParsing:
    def test_resolved_symbol_and_dso(self):
        row = _parse(_RESOLVED)[0]
        assert row["symbol"] == "__strlen_evex"
        assert row["dso"] == "/usr/lib/x86_64-linux-gnu/libc.so.6"

    def test_weight_survives_the_trailing_group(self):
        """access_latency_ns is neither the first nor the last token after BLK."""
        assert _parse(_RESOLVED)[0]["access_latency_ns"] == 318
        assert _parse(_UNRESOLVED_SYM)[0]["access_latency_ns"] == 162
        assert _parse(_UNRESOLVED_BOTH)[0]["access_latency_ns"] == 158
        assert _parse(_RESOLVED_STORE)[0]["access_latency_ns"] == 0

    def test_cache_level_still_parsed(self):
        assert _parse(_RESOLVED)[0]["cache_level"] == "DRAM"
        assert _parse(_UNRESOLVED_SYM)[0]["cache_level"] == "LFB"
        assert _parse(_UNRESOLVED_BOTH)[0]["cache_level"] == "L1"
        assert _parse(_RESOLVED_STORE)[0]["cache_level"] == "L1"

    def test_kernel_data_addresses_still_dropped(self):
        """User-space samples only — a kernel data address is not model memory."""
        assert _parse(_KERNEL_DATA_ADDR) == []

    def test_unknown_symbol_becomes_empty_not_literal(self):
        """'[unknown]' must not count towards symbol coverage."""
        row = _parse(_UNRESOLVED_SYM)[0]
        assert row["symbol"] == ""
        assert row["dso"] == "/usr/bin/python3.12"

    def test_unknown_dso_becomes_empty(self):
        row = _parse(_UNRESOLVED_BOTH)[0]
        assert row["symbol"] == ""
        assert row["dso"] == ""

    def test_data_address_not_confused_with_ip(self):
        assert _parse(_RESOLVED)[0]["address"] == 0x703D36
        assert _parse(_UNRESOLVED_SYM)[0]["address"] == 0x7039E354D1B8

    def test_op_type_preserved(self):
        assert _parse(_RESOLVED)[0]["op_type"] == "load"
        assert _parse(_UNRESOLVED_BOTH)[0]["op_type"] == "store"

    def test_resolved_symbol_on_a_store_sample(self):
        row = _parse(_RESOLVED_STORE)[0]
        assert row["symbol"] == "_PyEval_EvalFrameDefault"
        assert row["dso"] == "/usr/bin/python3.12"


class TestBackwardCompatibility:
    """Traces collected before ip was added must keep parsing identically."""

    def test_pre_ip_lines_still_yield_latency_and_cache_level(self):
        row = _parse(_PRE_IP_FORMAT)[0]
        assert row["access_latency_ns"] == 0
        assert row["cache_level"] == "L1"
        assert row["address"] == 0x7B6C02391270

    def test_pre_ip_lines_have_no_symbol(self):
        row = _parse(_PRE_IP_FORMAT)[0]
        assert row["symbol"] == ""
        assert row["dso"] == ""

    def test_both_formats_parse_in_one_stream(self):
        rows = _parse(_PRE_IP_FORMAT + _RESOLVED)
        assert len(rows) == 2
        assert [r["symbol"] for r in rows] == ["", "__strlen_evex"]


class TestSymbolEvidenceReachesLabeller:
    def test_resolved_engine_symbol_reaches_the_labeller(self):
        """
        The point of the ip change: a resolved symbol must survive parsing and
        arrive at the labeller.  It must NOT by itself establish kv_cache —
        llama_kv_cache_* manages slot metadata, not the K/V tensors, and on a
        real run that rule tagged four granules (one in the main stack) which
        absorbed 279,186 accesses from nearest_int, memmove and softmax.
        """
        from mrm_trace.labeller import TraceLabeller

        line = _RESOLVED.replace("__strlen_evex", "llama_kv_cache_find_slot").replace(
            "/usr/lib/x86_64-linux-gnu/libc.so.6", "/venv/lib/libllama.so"
        )
        rows = _parse(line)
        assert rows[0]["symbol"] == "llama_kv_cache_find_slot"

        labelled = list(TraceLabeller().label(rows))
        assert labelled[0]["region_type"] != "kv_cache"

    def test_a_compute_kernel_symbol_reaches_the_labeller(self):
        """
        What this check is for: the symbol parsed out of the IP field has to
        arrive at the labeller intact, whatever the rules then do with it.

        It used to observe that through the `activation` rule, which bound a
        parser test to a labelling policy -- so it failed when that rule was
        removed for varying 28-77x across identical repetitions.  A rule defined
        here keeps the plumbing under test and leaves the policy free.
        """
        from mrm_trace.labeller import TraceLabeller
        from mrm_trace.labeller.symbol_rules import SymbolRule

        line = _RESOLVED.replace("__strlen_evex", "ggml_vec_dot_q6_K_q8_K").replace(
            "/usr/lib/x86_64-linux-gnu/libc.so.6", "/venv/lib/libggml-cpu.so"
        )
        rows = _parse(line)
        for r in rows:
            r["op_type"] = "store"
        assert rows[0]["symbol"] == "ggml_vec_dot_q6_K_q8_K"
        assert "libggml-cpu.so" in rows[0]["dso"]

        probe = [SymbolRule(region_type="probe", op_type="store",
                            symbol_re=r"ggml_vec_dot")]
        labelled = list(TraceLabeller(rules=probe).label(rows))
        assert labelled[0]["region_type"] == "probe"

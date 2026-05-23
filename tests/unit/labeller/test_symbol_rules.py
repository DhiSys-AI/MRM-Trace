"""Unit tests for symbol classification rules."""

import pytest

from mrm_trace.labeller.symbol_rules import (
    DEFAULT_RULES,
    SymbolRule,
    classify_symbol,
)


# ── SymbolRule.matches() ──────────────────────────────────────────────────────

def test_rule_matches_when_all_none():
    rule = SymbolRule(region_type="weight")
    assert rule.matches("anything", "/lib/any.so", "weight_load", "store")


def test_rule_phase_filter_match():
    rule = SymbolRule(region_type="weight", phase="weight_load")
    assert rule.matches("sym", "dso", "weight_load", "store")


def test_rule_phase_filter_no_match():
    rule = SymbolRule(region_type="weight", phase="weight_load")
    assert not rule.matches("sym", "dso", "generation", "store")


def test_rule_op_type_filter_match():
    rule = SymbolRule(region_type="weight", op_type="store")
    assert rule.matches("sym", "dso", None, "store")


def test_rule_op_type_filter_no_match():
    rule = SymbolRule(region_type="weight", op_type="store")
    assert not rule.matches("sym", "dso", None, "load")


def test_rule_symbol_re_match():
    rule = SymbolRule(region_type="weight", symbol_re=r"ggml_init")
    assert rule.matches("ggml_init", "lib.so", None, "load")


def test_rule_symbol_re_no_match():
    rule = SymbolRule(region_type="weight", symbol_re=r"ggml_init")
    assert not rule.matches("malloc", "libc.so", None, "store")


def test_rule_dso_re_match():
    rule = SymbolRule(region_type="weight", dso_re=r"libggml")
    assert rule.matches("any_sym", "/usr/lib/libggml.so", None, "load")


def test_rule_dso_re_no_match():
    rule = SymbolRule(region_type="weight", dso_re=r"libggml")
    assert not rule.matches("any_sym", "/usr/lib/libc.so", None, "load")


def test_rule_symbol_re_case_insensitive():
    rule = SymbolRule(region_type="weight", symbol_re=r"GGML_INIT")
    assert rule.matches("ggml_init", "lib.so", None, "load")


def test_rule_combined_all_must_match():
    rule = SymbolRule(region_type="weight", phase="weight_load", symbol_re=r"ggml_init")
    assert rule.matches("ggml_init", "lib.so", "weight_load", "store")
    assert not rule.matches("ggml_init", "lib.so", "generation", "store")
    assert not rule.matches("malloc", "lib.so", "weight_load", "store")


# ── classify_symbol() ─────────────────────────────────────────────────────────

def test_weight_load_store_classifies_weight():
    assert classify_symbol("ggml_init", "libggml.so", "weight_load", "store") == "weight"


def test_weight_load_load_classifies_weight():
    assert classify_symbol("any_sym", "lib.so", "weight_load", "load") == "weight"


def test_kv_cache_symbol_classifies_kv_cache():
    assert classify_symbol("llama_kv_cache_update", "libllama.so", "generation", "store") == "kv_cache"


def test_kv_cache_symbol_alternative_name():
    assert classify_symbol("kv_write", "libmodel.so", "generation", "store") == "kv_cache"


def test_weight_compute_kernel_classifies_weight():
    assert classify_symbol("ggml_vec_dot_q4_K", "libggml.so", "generation", "load") == "weight"


def test_activation_symbol_classifies_activation():
    assert classify_symbol("ggml_silu", "libggml.so", "generation", "store") == "activation"


def test_runtime_malloc_classifies_runtime():
    assert classify_symbol("malloc", "libc.so", None, "store") == "runtime"


def test_unknown_symbol_returns_none():
    assert classify_symbol("_totally_unknown_sym", "unknown.so", None, "load") is None


def test_first_rule_wins():
    # weight_load+store is higher priority than generic kv_cache match
    # Verify that phase rule fires before symbol rule for edge cases
    rules = [
        SymbolRule(region_type="weight", phase="weight_load"),
        SymbolRule(region_type="kv_cache", symbol_re=r".*"),  # catch-all
    ]
    assert classify_symbol("any", "any", "weight_load", "load", rules=rules) == "weight"


def test_custom_rule_list():
    rules = [SymbolRule(region_type="kv_cache", symbol_re=r"my_kv_func")]
    assert classify_symbol("my_kv_func", "lib.so", None, "store", rules=rules) == "kv_cache"


def test_empty_rule_list_returns_none():
    assert classify_symbol("any", "any", None, "load", rules=[]) is None


def test_default_rules_list_is_non_empty():
    assert len(DEFAULT_RULES) > 0

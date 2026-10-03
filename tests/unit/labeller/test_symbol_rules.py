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

def test_no_symbol_rule_can_claim_weight():
    """
    A symbol says which code was running, not which region the sampled address
    belongs to.  For a store the address is the kernel's output buffer; the
    mmap'd GGUF file is read-only, so a store can never be a weight address at
    all.  Measured on a real run: of 26,617,751 rows a ggml_vec_dot rule
    labelled `weight`, 114 were inside the weight mapping — 0.0004%.

    weight now comes only from the GGUF mapping's address range.
    """
    for symbol, dso, phase, op in [
        ("ggml_init", "libggml.so", "weight_load", "store"),
        ("any_sym", "libggml.so", "weight_load", "load"),
        ("ggml_vec_dot_q4_K", "libggml.so", "generation", "load"),
        ("ggml_mul_mat", "libggml.so", "generation", "store"),
        ("ggml_new_tensor", "libggml.so", "weight_load", "store"),
        ("model_load", "libllama.so", "weight_load", "store"),
    ]:
        assert classify_symbol(symbol, dso, phase, op) != "weight", symbol


def test_no_symbol_rule_can_claim_kv_cache():
    """
    llama_kv_cache_* manages slot metadata, not the K/V tensors, so a sample
    taken inside it is not a KV address.  kv_cache now comes only from the
    engine's own KV allocation.
    """
    for symbol in ("llama_kv_cache_update", "kv_write", "kv_read",
                   "attn_cache_store", "llama_kv_cache_find_slot"):
        assert classify_symbol(symbol, "libllama.so", "generation",
                               "store") != "kv_cache", symbol


def test_weight_load_outside_engine_code_is_not_weight():
    """
    During weight_load the process also imports modules, grows the heap and
    spills to thread stacks.  An unconditional phase rule would sweep all of it
    into the weight region and inflate weight retention and working-set size.
    """
    assert classify_symbol("PyObject_Malloc", "libpython3.12.so",
                           "weight_load", "store") is None


def test_a_compute_kernel_store_classifies_activation():
    """
    What a forward-pass kernel's store *does* soundly imply: it writes to an
    activation buffer.  That is the one region a symbol can establish.
    """
    for symbol in ("ggml_vec_dot_q4_K", "ggml_mul_mat", "ggml_gemv_q4_K_8x8_q8_K",
                   "ggml_compute_forward_soft_max_f32", "ggml_rms_norm"):
        assert classify_symbol(symbol, "libggml.so", "generation",
                               "store") == "activation", symbol


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

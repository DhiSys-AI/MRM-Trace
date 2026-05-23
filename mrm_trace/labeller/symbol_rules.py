"""
Symbol-based region classification rules.

Rules are matched in list order; the first match wins.
classify_symbol() returns None when no rule matches — the caller
decides what to do with unknowns.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SymbolRule:
    """Classify a trace row's region_type from symbol, DSO, phase, and op_type."""

    region_type: str
    phase: Optional[str] = None      # None = match any phase
    op_type: Optional[str] = None    # None = match any op
    symbol_re: Optional[str] = None  # regex against symbol name
    dso_re: Optional[str] = None     # regex against DSO path
    _sym_pat: Optional[re.Pattern] = field(default=None, init=False, repr=False)
    _dso_pat: Optional[re.Pattern] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._sym_pat = re.compile(self.symbol_re, re.IGNORECASE) if self.symbol_re else None
        self._dso_pat = re.compile(self.dso_re, re.IGNORECASE) if self.dso_re else None

    def matches(
        self,
        symbol: str,
        dso: str,
        phase: Optional[str],
        op_type: str,
    ) -> bool:
        if self.phase is not None and phase != self.phase:
            return False
        if self.op_type is not None and op_type != self.op_type:
            return False
        if self._sym_pat is not None and not self._sym_pat.search(symbol or ""):
            return False
        if self._dso_pat is not None and not self._dso_pat.search(dso or ""):
            return False
        return True


# Default rules for llama.cpp / vLLM inference traces.
# Matched in order; first match wins.
DEFAULT_RULES: List[SymbolRule] = [
    # Phase weight_load + store → weight write (highest priority)
    SymbolRule(region_type="weight", phase="weight_load", op_type="store"),
    # Any access during weight_load → weight
    SymbolRule(region_type="weight", phase="weight_load"),
    # KV cache symbols (before generic weight kernels to avoid false positives)
    SymbolRule(
        region_type="kv_cache",
        symbol_re=r"kv_cache|kv_update|kv_read|kv_write|attn_cache|llama_kv",
    ),
    # Weight compute kernels (dot-product, quantised matmul)
    SymbolRule(
        region_type="weight",
        symbol_re=r"ggml_vec_dot|ggml_mul_mat|ggml_q[48]_k|ggml_q[48]\b",
    ),
    # GGML allocator / tensor init → weight
    SymbolRule(
        region_type="weight",
        symbol_re=r"ggml_init|ggml_new_tensor|ggml_set_|model_load|load_model|load_weight",
    ),
    # Activation compute (forward pass intermediates)
    SymbolRule(
        region_type="activation",
        symbol_re=r"ggml_compute|ggml_add|ggml_silu|ggml_norm|ggml_rms_norm|ggml_rope",
    ),
    # Runtime memory management
    SymbolRule(
        region_type="runtime",
        symbol_re=r"^(malloc|calloc|realloc|free|mmap|munmap|brk)$",
    ),
]


def classify_symbol(
    symbol: str,
    dso: str,
    phase: Optional[str],
    op_type: str,
    rules: Optional[List[SymbolRule]] = None,
) -> Optional[str]:
    """
    Return region_type for a trace row, or None if no rule matches.

    Args:
        symbol:  symbol name from the trace row
        dso:     DSO path from the trace row
        phase:   phase label or None
        op_type: 'load' or 'store'
        rules:   rule list; uses DEFAULT_RULES when None

    Returns:
        region_type string or None
    """
    for rule in (rules if rules is not None else DEFAULT_RULES):
        if rule.matches(symbol, dso, phase, op_type):
            return rule.region_type
    return None

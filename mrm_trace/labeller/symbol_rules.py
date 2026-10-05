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
_ENGINE_DSO_RE = r"libggml|libllama|libmtmd|ggml|llama"

DEFAULT_RULES: List[SymbolRule] = [
    # A symbol identifies the code that was running, NOT the region the sampled
    # address belongs to.  For a store sample the address is the kernel's
    # *output* buffer; for a load it may be any of its inputs.  Measured on a
    # real llama.cpp run (Llama-3.2-1B, ctx 2048, 28.3M samples):
    #
    #   rows labelled weight by ggml_vec_dot/ggml_mul_mat : 26,617,751
    #   of those actually inside the mmap'd GGUF file     :        114
    #
    # i.e. 100% of the weight label was wrong.  ggml_vec_dot_q6_K_q8_K reads
    # weights and writes activations, so its store samples are activation
    # addresses.  The GGUF mapping is r--s (read-only), so a store can never
    # land in the weight region at all.
    #
    # Symbol rules therefore no longer claim `weight` or `kv_cache`.  Those two
    # come only from address ranges, which are ground truth:
    #   weight   — the mmap'd GGUF file (llamacpp_regions: mmap_gguf)
    #   kv_cache — the engine's own KV allocation (kv_allocation), captured by
    #              the allocation interposer; absent if it cannot be captured.
    #
    # A compute kernel's store was taken to imply an activation buffer, since
    # that is where a forward-pass kernel writes.  The inference is sound in
    # principle and useless in practice: the resulting `activation` share varied
    # 28-77x between repetitions of the *same* configuration, and a region that
    # varies across identical reps is mis-attributed, not noisy.  Measured
    # across the v1.5 sweep (share of rows, two reps per cell):
    #
    #   llama-3.2-1b ctx8192   0.03% vs 0.93%   (28.9x)
    #   llama-3.2-3b ctx2048   0.01% vs 0.95%   (76.7x)
    #   phi-3.5-mini ctx8192   0.02% vs 1.18%   (52.2x)
    #
    # The cause is that the same kernels also write scratch, intermediate and
    # reduction buffers, and which of those a sparse sample lands in is
    # arbitrary.  Carrying the region type invited its accidental use in a
    # cross-run comparison it could never support, so no rule claims
    # `activation` either.  Those stores now fall through to the address map,
    # which places them in the arena they actually belong to (`runtime`).
    #
    # To restore it, ground it the way `weight` and `kv_cache` are grounded: in
    # a recorded address range for the activation buffer, not in a symbol.
    # Runtime memory management — the address is the allocator's own bookkeeping.
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

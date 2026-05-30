"""
Streaming parser for memray binary trace files.

Memray records Python-level memory allocations. MALLOC/NEW → STORE (allocation),
FREE/DELETE → LOAD (read-before-free). Requires memray to be installed.
"""

import importlib
from pathlib import Path
from typing import Generator, List, Optional, Union

from mrm_trace.parser.schema import OpType

_FREE_ALLOCATORS = frozenset({"free", "delete", "delete[]"})


def _try_import_memray():
    try:
        return importlib.import_module("memray")
    except ImportError:
        return None


def parse_memray(
    path: Union[str, Path],
    run_id: str,
    phase_boundaries: Optional[List[dict]] = None,
    native_traces: bool = False,
    skip_frees: bool = True,
    sample_every_n: int = 1,
) -> Generator[dict, None, None]:
    """
    Stream-parse a memray binary trace file, yielding one canonical row dict per record.

    Args:
        native_traces: Set True when the trace was captured with native_traces=True and
            a debug-symbol build of the backend. Enables stack_trace() per record to
            extract ggml_* symbol names. Leave False (default) for pip-installed backends
            or Python-level traces — calling stack_trace() on every record is very slow
            and the symbols will be Python internals, not useful for region classification.
        skip_frees: Skip FREE/DELETE deallocation records (default True). These typically
            account for ~50% of records and are not needed for MRM retention/write-once
            analysis. Set False to include them.
        sample_every_n: Keep only every Nth allocation record (default 1 = keep all).
            Use 2–10 to reduce record count for very large traces; the address distribution
            and region classification are preserved at low sample rates.

    Raises:
        ImportError: if memray is not installed
    """
    memray = _try_import_memray()
    if memray is None:
        raise ImportError(
            "memray is not installed. "
            "Install it with: pip install memray"
        )

    path = Path(path)
    sample_id = 0
    raw_index = 0

    def _annotate_phase(ts_ns: int) -> Optional[str]:
        if not phase_boundaries:
            return None
        for pb in phase_boundaries:
            if pb["start_ns"] <= ts_ns < pb["end_ns"]:
                return pb["phase"]
        return None

    reader = memray.FileReader(str(path))
    for record in reader.get_allocation_records():
        allocator = str(getattr(record, "allocator", "")).lower()
        is_free = any(tok in allocator for tok in _FREE_ALLOCATORS)

        if skip_frees and is_free:
            continue

        raw_index += 1
        if sample_every_n > 1 and (raw_index % sample_every_n) != 0:
            continue

        addr = record.address
        ts_ns = int(getattr(record, "timestamp", 0))
        op_type = OpType.LOAD.value if is_free else OpType.STORE.value

        symbol = ""
        dso = ""
        if native_traces:
            # stack_trace() is expensive — only call it when symbols are expected.
            # With pip-installed backends (stripped C symbols) this is always empty.
            try:
                frames = record.stack_trace()
            except NotImplementedError:
                frames = []
            # Scan from innermost outward for the first frame with a non-empty
            # function name. frames[0] is the C allocator itself and is often
            # unnamed in stripped native builds; the meaningful symbol is a few
            # frames up.
            for frame in (frames or []):
                fn = getattr(frame, "function_name", "") or ""
                if fn:
                    symbol = fn
                    dso = getattr(frame, "filename", "") or ""
                    break

        yield {
            "run_id": run_id,
            "sample_id": sample_id,
            "timestamp_ns": ts_ns,
            "phase": _annotate_phase(ts_ns),
            "address": addr,
            "address_page": (addr >> 12) if addr else 0,
            "op_type": op_type,
            "symbol": symbol,
            "dso": dso,
            "region_type": "unknown",
            "region_id": 0,
            "access_latency_ns": None,
            "cache_level": None,
            "numa_node": None,
            "cpu_id": None,
            "thread_id": getattr(record, "thread_id", None),
            "token_index": None,
            "context_tokens": None,
        }
        sample_id += 1

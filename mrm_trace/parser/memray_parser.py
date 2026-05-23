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
) -> Generator[dict, None, None]:
    """
    Stream-parse a memray binary trace file, yielding one canonical row dict per record.

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

    def _annotate_phase(ts_ns: int) -> Optional[str]:
        if not phase_boundaries:
            return None
        for pb in phase_boundaries:
            if pb["start_ns"] <= ts_ns < pb["end_ns"]:
                return pb["phase"]
        return None

    reader = memray.FileReader(str(path))
    for record in reader.get_allocation_records():
        addr = record.address
        ts_ns = int(getattr(record, "timestamp", 0))
        allocator = str(getattr(record, "allocator", "")).lower()
        op_type = (
            OpType.LOAD.value
            if any(tok in allocator for tok in _FREE_ALLOCATORS)
            else OpType.STORE.value
        )

        try:
            frames = record.stack_trace()
        except NotImplementedError:
            frames = []

        # Scan from innermost outward for the first frame with a non-empty
        # function name. frames[0] is the C allocator itself and is often
        # unnamed in stripped native builds; the meaningful symbol is a few
        # frames up.
        symbol = ""
        dso = ""
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

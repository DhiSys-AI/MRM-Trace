"""
KV cache address capture via allocator interposition.

The KV cache is not separable from ``/proc/self/maps``: llama.cpp carves it out
of a larger ggml backend arena, so no mapping has its size.  Measured on
Llama-3.2-1B at ctx 2048, where the geometry implies 64 MiB, the new anonymous
mappings were 768 MiB, 512 MiB and 250 MiB — nothing to match against.

The *allocation* is exactly the implied size, though, so ``kvtrap.c`` interposes
the allocator and records base addresses.  This module builds that library on
demand and turns its log into pre-classified address ranges.

Without it, ``kv_cache`` is absent — which is the correct outcome: a wrong KV
label is worse than a missing one (workplan Section 5.1.1).
"""

from __future__ import annotations

import logging
import os
import pathlib
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

_SOURCE = pathlib.Path(__file__).with_name("kvtrap.c")
_LIB_NAME = "libkvtrap.so"

# A match must be exact.  The allocation is the geometry size to the byte on
# every context length measured, and a tolerance would re-admit the guessing
# this exists to remove.
_ENV_LOG = "MRM_ALLOC_LOG"


def _cache_dir() -> pathlib.Path:
    base = os.environ.get("XDG_CACHE_HOME") or (pathlib.Path.home() / ".cache")
    return pathlib.Path(base) / "mrm-trace"


def build_interposer(force: bool = False) -> Optional[pathlib.Path]:
    """
    Compile kvtrap.c and return the library path, or None if it cannot be built.

    Returning None is a supported outcome: the caller runs without KV capture
    and records that KV was not isolated, rather than failing the run.
    """
    if not _SOURCE.exists():
        log.warning("kvtrap.c missing at %s — KV capture unavailable", _SOURCE)
        return None

    out = _cache_dir() / _LIB_NAME
    if out.exists() and not force and out.stat().st_mtime >= _SOURCE.stat().st_mtime:
        return out

    cc = os.environ.get("CC") or shutil.which("gcc") or shutil.which("cc")
    if not cc:
        log.warning("no C compiler found — KV capture unavailable")
        return None

    out.parent.mkdir(parents=True, exist_ok=True)
    # Compile to a temporary name and move into place, so a concurrent run
    # never sees a half-written library.
    with tempfile.NamedTemporaryFile(suffix=".so", dir=out.parent, delete=False) as tmp:
        tmp_path = pathlib.Path(tmp.name)
    cmd = [cc, "-shared", "-fPIC", "-O2", "-o", str(tmp_path), str(_SOURCE), "-ldl"]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        tmp_path.unlink(missing_ok=True)
        log.warning("kvtrap build failed to start (%s) — KV capture unavailable", exc)
        return None
    if result.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        log.warning("kvtrap build failed: %s",
                    result.stderr.decode(errors="replace")[:300])
        return None
    tmp_path.replace(out)
    return out


def interposer_env(lib: Optional[pathlib.Path],
                   alloc_log: pathlib.Path) -> Dict[str, str]:
    """
    Environment additions that load the interposer into a child process.

    Empty when *lib* is None, so a caller can apply it unconditionally.
    """
    if lib is None:
        return {}
    env = {_ENV_LOG: str(alloc_log)}
    existing = os.environ.get("LD_PRELOAD", "")
    env["LD_PRELOAD"] = f"{lib}:{existing}" if existing else str(lib)
    return env


def parse_alloc_log(path) -> List[dict]:
    """Return [{pid, fn, size, addr}] from an allocation log, newest last."""
    entries: List[dict] = []
    try:
        with open(path) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) != 4:
                    continue
                try:
                    entries.append({
                        "pid": int(parts[0]),
                        "fn": parts[1],
                        "size": int(parts[2]),
                        "addr": int(parts[3], 16),
                    })
                except ValueError:
                    continue
    except OSError:
        return []
    return entries


def kv_regions_from_alloc_log(path, expected_kv: Optional[int],
                              pid: Optional[int] = None) -> List[dict]:
    """
    Address ranges for the KV cache, from an exact size match.

    Returns [] when *expected_kv* is unknown, when nothing matches, or when
    several allocations share the size — an ambiguous match is not evidence,
    and claiming one of them would be the guessing this replaces.  (Observed:
    at ctx 1024 the implied 32 MiB collides with glibc's 32 MiB arena
    reservations, while 64 MiB / 256 MiB / 1 GiB are unambiguous.)
    """
    if not expected_kv:
        return []
    entries = parse_alloc_log(path)
    if pid is not None:
        entries = [e for e in entries if e["pid"] == pid]
    matches = [e for e in entries if e["size"] == expected_kv]
    if len(matches) != 1:
        if matches:
            log.warning("KV allocation ambiguous: %d allocations of %d bytes — "
                        "kv_cache left absent", len(matches), expected_kv)
        return []
    hit = matches[0]
    return [{
        "region_type": "kv_cache",
        "addr_start": hit["addr"],
        "addr_end": hit["addr"] + hit["size"],
        "size_bytes": hit["size"],
        "source": "kv_allocation",
    }]

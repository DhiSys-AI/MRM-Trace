import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from mrm_trace.utils.files import sha256_file

MANIFEST_SCHEMA_VERSION = "1.0"

_KNOWN_SCHEMA_VERSIONS: Dict[str, str] = {
    "trace.parquet": "1.0",
    "region_map.parquet": "1.0",
    "kv_block_lifecycle.parquet": "1.0",
    "metrics.csv": "1.0",
    "metadata.json": "1.0",
}


def build_manifest(
    run_dir: Path,
    run_id: str,
    files: Optional[List[Path]] = None,
) -> Dict[str, Any]:
    """Build a manifest dict for all files in a run directory."""
    if files is None:
        files = [
            f for f in run_dir.iterdir()
            if f.is_file() and f.name != "manifest.json"
        ]

    file_entries: List[Dict[str, Any]] = []
    for f in sorted(files):
        entry: Dict[str, Any] = {
            "path": f.name,
            "size_bytes": f.stat().st_size,
            "sha256": sha256_file(f),
        }
        if f.name in _KNOWN_SCHEMA_VERSIONS:
            entry["schema_version"] = _KNOWN_SCHEMA_VERSIONS[f.name]
        file_entries.append(entry)

    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": file_entries,
    }


def write_manifest(
    run_dir: Path,
    run_id: str,
    files: Optional[List[Path]] = None,
) -> Path:
    """Write manifest.json into run_dir and return the path."""
    manifest = build_manifest(run_dir, run_id, files)
    out = run_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out

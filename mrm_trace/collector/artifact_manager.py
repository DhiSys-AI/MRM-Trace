"""Manages the raw artifact directory for a single collector run."""

import shutil
from pathlib import Path

MIN_FREE_BYTES: int = 1_073_741_824  # 1 GiB


class DiskSpaceError(RuntimeError):
    """Raised when available disk space is below the required minimum."""


class ArtifactManager:
    """
    Owns the directory layout for one run's raw collector output and
    enforces disk-space safety before collection starts.
    """

    def __init__(self, run_dir: Path, min_free_bytes: int = MIN_FREE_BYTES) -> None:
        self._run_dir = Path(run_dir)
        self._min_free_bytes = min_free_bytes

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    @property
    def raw_dir(self) -> Path:
        return self._run_dir / "raw"

    def ensure_directories(self) -> None:
        """Create run_dir/raw/ (and all parents) if they do not already exist."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def check_disk_space(self) -> None:
        """
        Raise DiskSpaceError if the filesystem hosting run_dir has less than
        min_free_bytes available.  Walks up to an existing ancestor if run_dir
        itself does not yet exist.
        """
        check_path = self._run_dir
        while not check_path.exists():
            check_path = check_path.parent
        usage = shutil.disk_usage(check_path)
        if usage.free < self._min_free_bytes:
            free_gib = usage.free / 1_073_741_824
            need_gib = self._min_free_bytes / 1_073_741_824
            raise DiskSpaceError(
                f"Insufficient disk space: {free_gib:.2f} GiB free, "
                f"{need_gib:.2f} GiB required"
            )

    # ── Canonical artifact paths ──────────────────────────────────────────────

    def perf_data_path(self) -> Path:
        return self.raw_dir / "perf.data"

    def perf_script_path(self) -> Path:
        return self.raw_dir / "perf_script.txt"

    def memray_path(self) -> Path:
        return self.raw_dir / "memray.bin"

    def get_file_size(self, path: Path) -> int:
        """Return byte size of *path*, or 0 if it does not exist."""
        p = Path(path)
        return p.stat().st_size if p.exists() else 0

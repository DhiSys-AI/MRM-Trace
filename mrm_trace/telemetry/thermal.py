"""
Thermal throttle detection.

Reads per-CPU throttle event counters from the Linux sysfs interface
(/sys/devices/system/cpu/cpu*/thermal_throttle/core_throttle_count).
Two snapshots (before and after the run) are diffed to count events that
occurred *during* the run.

On non-Linux or unavailable sysfs paths, returns 0 events (graceful degradation).
"""

from __future__ import annotations

import glob as _glob
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

_SYSFS_PATTERN = (
    "/sys/devices/system/cpu/cpu*/thermal_throttle/core_throttle_count"
)

_THERMAL_ZONE_ROOT = Path("/sys/class/thermal")


def read_package_temp_c() -> float:
    """
    Read CPU package temperature in Celsius from sysfs thermal zones.

    Prefers the x86 package zone (type containing "pkg", or "x86" + "cpu");
    falls back to the highest reading across all zones. Returns 0.0 if
    /sys/class/thermal is unavailable (e.g. non-Linux, containers without
    the mount, WSL2 without exposed thermal zones).
    """
    if not _THERMAL_ZONE_ROOT.exists():
        return 0.0
    for zone in _THERMAL_ZONE_ROOT.glob("thermal_zone*"):
        try:
            zone_type = (zone / "type").read_text().strip().lower()
            if "pkg" in zone_type or ("x86" in zone_type and "cpu" in zone_type):
                return int((zone / "temp").read_text()) / 1000.0
        except OSError:
            continue
    temps = []
    for temp_file in _THERMAL_ZONE_ROOT.glob("thermal_zone*/temp"):
        try:
            temps.append(int(temp_file.read_text()) / 1000.0)
        except OSError:
            pass
    return max(temps) if temps else 0.0


@dataclass
class ThermalSnapshot:
    """Per-CPU throttle counts at a point in time."""

    counts: Dict[str, int] = field(default_factory=dict)  # cpu_path → count

    @classmethod
    def capture(cls) -> "ThermalSnapshot":
        """Read current throttle counters from sysfs. Returns empty on failure."""
        counts: Dict[str, int] = {}
        for path_str in _glob.glob(_SYSFS_PATTERN):
            try:
                counts[path_str] = int(Path(path_str).read_text().strip())
            except (OSError, ValueError):
                pass
        return cls(counts=counts)

    def events_since(self, earlier: "ThermalSnapshot") -> int:
        """
        Return the total number of new throttle events since *earlier* snapshot.

        Uses max(0, current - earlier) per CPU to guard against counter wrap.
        """
        total = 0
        for path, current_count in self.counts.items():
            prev = earlier.counts.get(path, current_count)
            total += max(0, current_count - prev)
        return total


class ThermalMonitor:
    """
    Measures thermal throttle events across a run using two snapshots.

    Usage::

        monitor = ThermalMonitor()
        monitor.begin()
        # ... run ...
        events = monitor.end()
    """

    def __init__(self) -> None:
        self._before: ThermalSnapshot = ThermalSnapshot()
        self._after: ThermalSnapshot = ThermalSnapshot()

    def begin(self) -> None:
        """Capture the pre-run throttle snapshot."""
        self._before = ThermalSnapshot.capture()

    def end(self) -> int:
        """Capture the post-run snapshot and return the event delta."""
        self._after = ThermalSnapshot.capture()
        return self._after.events_since(self._before)

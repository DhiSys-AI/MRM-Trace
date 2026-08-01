"""
mrm-trace machine validator  --  day-one "can this box actually collect?" check.

Primary entry points:
    mrm-trace validate --machine              # via CLI (preferred once installed)
    mrm-trace validate --machine --json       # machine-readable, for CI
    python -m mrm_trace.validate_machine      # direct module run

Standalone (before venv/package is installed):
    python validate_machine.py                # repo-root redirect → this module
    python mrm_trace/validate_machine.py      # direct, no package needed

Exit codes:
    0  READY — publication-grade perf-mem collection with region labelling available
    1  degraded only (memray fallback; no nanosecond timestamps, empty region_map)
    2  cannot collect at all

Stdlib only on purpose: this must run on a fresh machine before the package's
own dependencies (memray, pandas, ...) are present.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path


# --------------------------------------------------------------------------- #
# Result model
# --------------------------------------------------------------------------- #

class Status(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    INFO = "INFO"
    SKIP = "SKIP"


@dataclass
class Check:
    name: str
    status: Status
    detail: str = ""
    remediation: str = ""
    data: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _use_colour() -> bool:
    """True when it is safe to emit ANSI colour on stdout."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM", "") == "dumb":
        return False
    return sys.stdout.isatty()


def _green(text: str) -> str:
    return f"\033[32m{text}\033[0m" if _use_colour() else text


def _read(path: str) -> str:
    try:
        return Path(path).read_text(errors="replace").strip()
    except OSError:
        return ""


def _run(argv, timeout=30):
    """Run a command, never raise. Returns (rc, stdout, stderr)."""
    try:
        p = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return 127, "", "command not found"
    except subprocess.TimeoutExpired:
        return 124, "", "timed out"


def _kver_tuple(s: str):
    m = re.match(r"(\d+)\.(\d+)", s)
    return (int(m.group(1)), int(m.group(2))) if m else None


# --------------------------------------------------------------------------- #
# Platform / kernel
# --------------------------------------------------------------------------- #

def check_os() -> Check:
    sysname = platform.system()
    if sysname != "Linux":
        return Check(
            "os", Status.FAIL, f"{sysname} detected",
            "perf mem requires Linux. Dual-boot Ubuntu LTS; do not collect from Windows/macOS.",
        )
    return Check("os", Status.PASS, f"Linux {platform.release()}")


def check_wsl2() -> Check:
    blob = (_read("/proc/version") + " " + _read("/proc/sys/kernel/osrelease")).lower()
    if "microsoft" in blob or "wsl" in blob:
        return Check(
            "wsl2", Status.FAIL, "Running under WSL2",
            "The WSL2 kernel does not expose hardware PMU counters; perf mem yields zero "
            "samples. Dual-boot native Linux (Ubuntu LTS). Use memray only for development.",
        )
    return Check("wsl2", Status.PASS, "Native kernel (not WSL2)")


def check_kernel() -> Check:
    rel = platform.release()
    kt = _kver_tuple(rel)
    note = ""
    if kt and kt == (6, 8):
        note = (" | kernel 6.8: `perf mem record -p PID` (attach) fails for Intel PEBS "
                "load-latency events even as root -- use spawn mode "
                "(`perf mem record -- python ...`).")
    return Check("kernel", Status.PASS, f"{rel}{note}",
                 data={"release": rel, "version": list(kt) if kt else None})


def check_perf_installed() -> Check:
    path = shutil.which("perf")
    if not path:
        return Check(
            "perf_installed", Status.FAIL, "perf binary not found on PATH",
            "sudo apt install linux-tools-common linux-tools-generic linux-tools-$(uname -r)",
        )
    return Check("perf_installed", Status.PASS, path, data={"path": path})


def check_perf_version_match() -> Check:
    if not shutil.which("perf"):
        return Check("perf_version", Status.SKIP, "perf not installed")
    rc, out, err = _run(["perf", "--version"])
    text = (out + err).strip()
    pv = _kver_tuple(re.sub(r"^perf version\s*", "", text))
    kv = _kver_tuple(platform.release())
    # perf often prints a build-mismatch warning to stderr itself:
    if "not found for kernel" in (out + err).lower():
        return Check("perf_version", Status.WARN, text,
                     "Install the matching build: sudo apt install linux-tools-$(uname -r)")
    if pv and kv and pv != kv:
        return Check(
            "perf_version", Status.WARN,
            f"perf {pv[0]}.{pv[1]} vs kernel {kv[0]}.{kv[1]} (minor mismatch)",
            "Usually harmless, but install linux-tools-$(uname -r) to be safe.",
            data={"perf": list(pv), "kernel": list(kv)},
        )
    return Check("perf_version", Status.PASS, text or "version reported")


# --------------------------------------------------------------------------- #
# Privilege / paranoia
# --------------------------------------------------------------------------- #

def check_root() -> Check:
    if os.geteuid() == 0:
        return Check("root", Status.PASS, "running as root (perf mem + memray native_traces enabled)")
    return Check(
        "root", Status.INFO, "not running as root",
        "Collection runs as root: `sudo -E venv/bin/python notebooks/scripts/demo_perf_mem.py ...` "
        "(plain `sudo -E python` resolves to system Python -- use the full venv path).",
    )


def check_paranoid() -> Check:
    raw = _read("/proc/sys/kernel/perf_event_paranoid")
    if raw == "":
        return Check("paranoid", Status.WARN, "could not read perf_event_paranoid")
    val = int(raw)
    root = os.geteuid() == 0
    if root:
        return Check("paranoid", Status.PASS,
                     f"perf_event_paranoid={val} (root overrides this)", data={"value": val})
    if val <= 0:
        return Check("paranoid", Status.PASS,
                     f"perf_event_paranoid={val} (unprivileged CPU events allowed)",
                     data={"value": val})
    return Check(
        "paranoid", Status.WARN, f"perf_event_paranoid={val} blocks unprivileged CPU events",
        "Run collection as root, or: sudo sysctl kernel.perf_event_paranoid=1 "
        "(make persistent in /etc/sysctl.d/ if you collect rootless).",
        data={"value": val},
    )


# --------------------------------------------------------------------------- #
# CPU vendor + sampling facility
# --------------------------------------------------------------------------- #

def _cpuinfo_field(key: str) -> str:
    for line in _read("/proc/cpuinfo").splitlines():
        if line.lower().startswith(key.lower()):
            return line.split(":", 1)[1].strip()
    return ""


def check_cpu_vendor() -> Check:
    vendor = _cpuinfo_field("vendor_id") or platform.machine()
    model = _cpuinfo_field("model name") or platform.processor()
    facility = {
        "GenuineIntel": "PEBS (Precise Event Based Sampling)",
        "AuthenticAMD": "IBS (Instruction Based Sampling)",
    }.get(vendor)
    if facility is None and ("aarch64" in platform.machine() or "arm" in platform.machine().lower()):
        facility = "SPE (Statistical Profiling Extension)"
    detail = f"{model or vendor}  ->  {facility or 'unknown'}"
    if facility is None:
        return Check("cpu", Status.WARN, detail,
                     "Unrecognised CPU vendor for memory sampling — check_sampling_facility "
                     "below will likely FAIL; perf mem needs Intel PEBS, AMD IBS, or ARM SPE.",
                     data={"vendor": vendor, "model": model, "facility": "unknown"})
    return Check("cpu", Status.PASS, detail,
                 data={"vendor": vendor, "model": model, "facility": facility})


def _exists_any(paths) -> str | None:
    for p in paths:
        if Path(p).exists():
            return p
    return None


def check_sampling_facility() -> Check:
    vendor = _cpuinfo_field("vendor_id")
    machine = platform.machine()

    if vendor == "GenuineIntel":
        # Homogeneous (cpu) and hybrid (cpu_core) sysfs paths.
        ldlat = _exists_any([
            "/sys/bus/event_source/devices/cpu/format/ldlat",
            "/sys/bus/event_source/devices/cpu_core/format/ldlat",
        ])
        memloads = _exists_any([
            "/sys/bus/event_source/devices/cpu/events/mem-loads",
            "/sys/bus/event_source/devices/cpu_core/events/mem-loads",
        ])
        if ldlat and memloads:
            return Check("facility", Status.PASS,
                         "Intel PEBS load-latency exposed (ldlat + mem-loads present)",
                         data={"ldlat": ldlat, "mem_loads": memloads})
        return Check(
            "facility", Status.FAIL,
            "Intel CPU but PEBS load-latency sysfs markers absent "
            f"(ldlat={'ok' if ldlat else 'missing'}, mem-loads={'ok' if memloads else 'missing'})",
            "Almost always means a virtualised/hidden PMU (cloud VM or WSL2). Use bare metal.",
        )

    if vendor == "AuthenticAMD":
        ibs = _exists_any([
            "/sys/bus/event_source/devices/ibs_op",
            "/sys/bus/event_source/devices/ibs_fetch",
        ])
        status = Status.PASS if ibs else Status.FAIL
        return Check("facility", status,
                     "AMD IBS exposed" if ibs else "AMD IBS not exposed",
                     "" if ibs else "Bare metal + recent kernel needed for IBS memory sampling.")

    if "aarch64" in machine or "arm" in machine.lower():
        spe = _exists_any([f"/sys/bus/event_source/devices/arm_spe_{i}" for i in range(4)])
        status = Status.PASS if spe else Status.FAIL
        return Check("facility", status,
                     "ARM SPE exposed" if spe else "ARM SPE not exposed",
                     "" if spe else "Needs SPE-capable core (Neoverse-class) + matching kernel.")

    return Check("facility", Status.WARN, "Unknown CPU vendor; rely on the live smoke test below.")


def check_pmu_dmesg() -> Check:
    """Best-effort: dmesg often needs root / dmesg_restrict=0."""
    rc, out, err = _run(["dmesg"])
    if rc != 0:
        return Check("pmu_dmesg", Status.SKIP,
                     "dmesg not readable (kernel.dmesg_restrict=1 or non-root); not decisive")
    for line in out.splitlines():
        if "PMU driver" in line or ("PMU" in line and "events" in line):
            ok = "software events only" not in line.lower()
            return Check("pmu_dmesg", Status.PASS if ok else Status.FAIL, line.strip())
    return Check("pmu_dmesg", Status.INFO, "no PMU driver line found in dmesg")


# --------------------------------------------------------------------------- #
# Live smoke test  --  the decisive check
# --------------------------------------------------------------------------- #

# A short, strided, prefetcher-defeating memory churn that overflows L3 so PEBS
# load-latency actually fires. Stdlib only; no numpy.
_SMOKE_WORKLOAD = r"""
import array
n = 1 << 22                       # 4,194,304 q-words = 32 MiB (> typical L3)
buf = array.array('q', bytes(8 * n))
step = 4099                       # prime stride defeats the HW prefetcher
i = 0
s = 0
for _ in range(n * 3):
    s += buf[i]
    i += step
    if i >= n:
        i -= n
print(s)
"""

_SAMPLES_RE = re.compile(r"\(([\d,]+)\s+samples\)")


def check_perf_mem_smoke() -> Check:
    if platform.system() != "Linux" or not shutil.which("perf"):
        return Check("smoke", Status.SKIP, "needs Linux + perf")
    blob = (_read("/proc/version") + _read("/proc/sys/kernel/osrelease")).lower()
    if "microsoft" in blob or "wsl" in blob:
        return Check("smoke", Status.SKIP, "skipped under WSL2 (PMU hidden)")

    tmp = Path(tempfile.mkdtemp(prefix="mrm_smoke_"))
    data = tmp / "perf.data"
    try:
        rc, out, err = _run(
            ["perf", "mem", "record", "-o", str(data), "--",
             sys.executable, "-c", _SMOKE_WORKLOAD],
            timeout=60,
        )
        combined = (out + "\n" + err)

        if rc != 0 or not data.exists():
            low = combined.lower()
            if "invalid argument" in low or "not supported" in low or "no such" in low:
                hint = ("PMU not delivering memory samples -- typical of a VM/WSL2 or a CPU "
                        "without exposed PEBS/IBS/SPE. Confirm bare metal.")
            elif "permission" in low or "paranoid" in low:
                hint = ("Privilege problem: run as root or lower perf_event_paranoid "
                        "(sudo sysctl kernel.perf_event_paranoid=1).")
            else:
                hint = "Inspect the perf output below; the box may not be collectible."
            return Check("smoke", Status.FAIL,
                         f"perf mem record failed (rc={rc}): {combined.strip()[:300]}", hint)

        m = _SAMPLES_RE.search(combined)
        n_samples = int(m.group(1).replace(",", "")) if m else None

        if n_samples == 0:
            return Check(
                "smoke", Status.FAIL, "perf mem ran but captured 0 samples",
                "Hallmark of a hidden/virtual PMU. The CPU is not delivering memory "
                "samples -- you need a different (bare-metal) box.",
            )

        # A high sample count is self-confirming: the workload is a 32 MiB strided
        # read that only PEBS-capable hardware can sample at this density.
        _CONFIDENT_THRESHOLD = 1000
        if n_samples is not None and n_samples >= _CONFIDENT_THRESHOLD:
            return Check(
                "smoke", Status.PASS,
                f"captured {n_samples:,} samples (>{_CONFIDENT_THRESHOLD} threshold) "
                "-- full-quality perf mem collection works",
                data={"samples": n_samples},
            )

        # For unexpectedly low sample counts also try to confirm via perf script.
        addr_ok = False
        rc2, sout, serr = _run(["perf", "script", "-i", str(data), "-F", "event,addr"])
        if rc2 == 0 and sout.strip():
            for line in sout.splitlines():
                # perf may emit bare "0x..." or labelled "addr: 0x..."
                mh = re.search(r"0x[0-9a-fA-F]{4,}", line)
                if mh and int(mh.group(0), 16) != 0:
                    addr_ok = True
                    break

        if not addr_ok:
            # Fallback: any non-comment output from perf script at all.
            rc3, sout3, _ = _run(["perf", "script", "-i", str(data)])
            addr_ok = rc3 == 0 and any(
                line.strip() and not line.startswith("#") for line in sout3.splitlines()
            )

        if addr_ok:
            return Check(
                "smoke", Status.PASS,
                f"captured {n_samples} samples with non-zero data addresses -- "
                "full-quality perf mem collection works",
                data={"samples": n_samples},
            )
        return Check(
            "smoke", Status.WARN,
            f"captured {n_samples} samples but could not confirm data addresses via perf script",
            "Run manually: sudo perf mem record -o /tmp/t.data -- sleep 1 && "
            "sudo perf script -i /tmp/t.data -F event,addr | head -5",
            data={"samples": n_samples},
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Thermal / power  --  reproducibility controls for a throttle-prone laptop
# --------------------------------------------------------------------------- #

def check_ac_power() -> Check:
    for online in Path("/sys/class/power_supply").glob("A*/online") if Path("/sys/class/power_supply").exists() else []:
        if _read(str(online)) == "1":
            return Check("ac_power", Status.PASS, "on AC power")
    # No AC line found, or it reads 0.
    return Check(
        "ac_power", Status.WARN, "not on AC power (or AC status unreadable)",
        "Collect on AC with a fixed power profile. On battery this chassis down-clocks and "
        "throttles, contaminating run-to-run variance.",
    )


def check_governor_turbo() -> Check:
    gov = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    no_turbo = _read("/sys/devices/system/cpu/intel_pstate/no_turbo")
    bits = []
    if gov:
        bits.append(f"governor={gov}")
    if no_turbo != "":
        bits.append("turbo=" + ("off" if no_turbo == "1" else "on"))
    detail = ", ".join(bits) or "cpufreq info unavailable"

    if not gov and no_turbo == "":
        return Check("governor", Status.WARN, detail,
                     "cpufreq/intel_pstate sysfs not readable — cannot verify frequency is "
                     "pinned for reproducibility.")

    turbo_on = no_turbo == "0"
    governor_ok = gov in ("", "performance")
    if governor_ok and not turbo_on:
        return Check("governor", Status.PASS, detail)

    remediation_bits = []
    if not governor_ok:
        remediation_bits.append("sudo cpupower frequency-set -g performance")
    if turbo_on:
        remediation_bits.append("echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo")
    return Check(
        "governor", Status.WARN, detail,
        "Frequency is not pinned — reproducibility risk under sustained perf mem collection. "
        + " && ".join(remediation_bits),
    )


def check_thermal() -> Check:
    temps = []
    base = Path("/sys/class/thermal")
    if base.exists():
        for tz in base.glob("thermal_zone*"):
            t = _read(str(tz / "temp"))
            ty = _read(str(tz / "type"))
            if t.isdigit():
                temps.append((ty or tz.name, int(t) / 1000.0))
    throttle = list(Path("/sys/devices/system/cpu").glob("cpu*/thermal_throttle/core_throttle_count"))
    if not temps:
        return Check("thermal", Status.WARN,
                     "no thermal sensors readable (load `coretemp`?)",
                     "modprobe coretemp; install lm-sensors so the validity classifier can "
                     "log package temp and throttle events.")
    hottest = max(t for _, t in temps)
    tail = " | throttle counters readable" if throttle else " | throttle counters NOT exposed"
    status = Status.PASS if throttle else Status.WARN
    return Check("thermal", status,
                 f"idle baseline ~{hottest:.0f}C across {len(temps)} zones{tail}",
                 "" if throttle else "Without throttle counters the validity classifier can't "
                 "auto-discard contaminated runs on this chassis.",
                 data={"baseline_c": hottest, "throttle_counters": bool(throttle)})


# --------------------------------------------------------------------------- #
# Fallback collector / GPU energy / storage
# --------------------------------------------------------------------------- #

def check_memray() -> Check:
    try:
        import importlib.util
        spec = importlib.util.find_spec("memray")
        if spec is not None:
            return Check("memray", Status.PASS, "memray importable (degraded fallback available)")
    except Exception:
        pass
    return Check("memray", Status.INFO, "memray not installed",
                 "pip install memray -- the fallback when perf mem is unavailable.")


def check_gpu_power() -> Check:
    if not shutil.which("nvidia-smi"):
        return Check("gpu", Status.INFO, "nvidia-smi not found (CPU-only collection still fine)")
    rc, out, _ = _run(["nvidia-smi",
                       "--query-gpu=name,memory.total,power.draw",
                       "--format=csv,noheader"])
    if rc != 0 or not out.strip():
        return Check("gpu", Status.WARN, "nvidia-smi present but query failed",
                     "Needed for per-run GPU energy (a first-class metric). Check the driver.")
    line = out.strip().splitlines()[0]
    powered = bool(re.search(r"\d+(\.\d+)?\s*W", line))
    return Check("gpu", Status.PASS if powered else Status.WARN,
                 f"{line}" + ("" if powered else "  (power.draw not reported)"),
                 "" if powered else "Energy telemetry needs power.draw; some laptop dGPUs gate it.",
                 data={"power_draw_ok": powered})


def check_disk(out_dir: str = ".", min_gb: int = 50) -> Check:
    try:
        free_gb = shutil.disk_usage(out_dir).free / 1e9
    except OSError:
        return Check("disk", Status.WARN, f"could not stat {out_dir}")
    status = Status.PASS if free_gb >= min_gb else Status.WARN
    return Check("disk", status, f"{free_gb:.0f} GB free at {os.path.abspath(out_dir)}",
                 "" if free_gb >= min_gb else f"Raw perf.data balloons fast; keep >= {min_gb} GB free.",
                 data={"free_gb": round(free_gb, 1)})


# --------------------------------------------------------------------------- #
# Machine context snapshot — for embedding in run metadata.json
# --------------------------------------------------------------------------- #

def collect_machine_context(out_dir: str = ".") -> dict:
    """
    Fast, non-destructive machine snapshot for embedding in metadata.json.

    Skips the smoke test (expensive). Safe to call at the start of every run.
    Returns a dict with three top-level keys:
      - "software"         → OS, kernel, perf version, paranoid, memray, mrm_trace
      - "hardware_context" → CPU model/vendor/PMU, AC power, turbo, disk free
      - "gpu_power_draw_w" → current GPU power draw in watts, or None
    """
    import importlib.util as _ilu
    import mrm_trace as _mrm

    # ── software ──────────────────────────────────────────────────────────────
    blob = (_read("/proc/version") + " " + _read("/proc/sys/kernel/osrelease")).lower()
    is_wsl2 = "microsoft" in blob or "wsl" in blob

    perf_bin = shutil.which("perf")
    perf_version = None
    if perf_bin:
        _, out, err = _run(["perf", "--version"], timeout=5)
        perf_version = (out + err).strip() or None

    paranoid_raw = _read("/proc/sys/kernel/perf_event_paranoid")
    paranoid: int | None = None
    if paranoid_raw.lstrip("-").isdigit():
        paranoid = int(paranoid_raw)

    memray_installed = False
    try:
        memray_installed = _ilu.find_spec("memray") is not None
    except Exception:
        pass

    # ── hardware context ──────────────────────────────────────────────────────
    vendor = _cpuinfo_field("vendor_id")
    cpu_model = _cpuinfo_field("model name") or platform.processor()

    pmu_type: str | None = {
        "GenuineIntel": "PEBS",
        "AuthenticAMD": "IBS",
    }.get(vendor)
    if pmu_type is None and ("aarch64" in platform.machine() or "arm" in platform.machine().lower()):
        pmu_type = "SPE"

    ldlat = _exists_any([
        "/sys/bus/event_source/devices/cpu/format/ldlat",
        "/sys/bus/event_source/devices/cpu_core/format/ldlat",
    ])
    memloads = _exists_any([
        "/sys/bus/event_source/devices/cpu/events/mem-loads",
        "/sys/bus/event_source/devices/cpu_core/events/mem-loads",
    ])
    pmu_facility = bool(ldlat and memloads)

    ac_power = False
    ac_dir = Path("/sys/class/power_supply")
    if ac_dir.exists():
        for online in ac_dir.glob("A*/online"):
            if _read(str(online)) == "1":
                ac_power = True
                break

    no_turbo = _read("/sys/devices/system/cpu/intel_pstate/no_turbo")
    turbo_enabled: bool | None = (no_turbo == "0") if no_turbo != "" else None

    disk_free_gb: float | None = None
    try:
        disk_free_gb = round(shutil.disk_usage(out_dir).free / 1e9, 1)
    except OSError:
        pass

    # ── GPU power draw (one-time snapshot) ───────────────────────────────────
    gpu_power_draw_w: float | None = None
    if shutil.which("nvidia-smi"):
        try:
            rc, out, _ = _run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                timeout=5,
            )
            if rc == 0 and out.strip():
                gpu_power_draw_w = float(out.strip().splitlines()[0].strip())
        except Exception:
            pass

    return {
        "software": {
            "os": f"{platform.system()} {platform.release()}",
            "kernel": platform.release(),
            "hostname": socket.gethostname(),
            "is_wsl2": is_wsl2,
            "perf_bin": perf_bin,
            "perf_version": perf_version,
            "perf_event_paranoid": paranoid,
            "memray_installed": memray_installed,
            "mrm_trace_version": getattr(_mrm, "__version__", None),
        },
        "hardware_context": {
            "cpu_model": cpu_model,
            "cpu_vendor": vendor,
            "pmu_type": pmu_type,
            "pmu_facility_available": pmu_facility,
            "ac_power": ac_power,
            "turbo_enabled": turbo_enabled,
            "disk_free_gb": disk_free_gb,
        },
        "gpu_power_draw_w": gpu_power_draw_w,
    }


# --------------------------------------------------------------------------- #
# Orchestration + verdict
# --------------------------------------------------------------------------- #

def run_all(out_dir: str = ".") -> list[Check]:
    return [
        check_os(),
        check_wsl2(),
        check_kernel(),
        check_perf_installed(),
        check_perf_version_match(),
        check_root(),
        check_paranoid(),
        check_cpu_vendor(),
        check_sampling_facility(),
        check_pmu_dmesg(),
        check_perf_mem_smoke(),       # decisive
        check_ac_power(),
        check_governor_turbo(),
        check_thermal(),
        check_memray(),
        check_gpu_power(),
        check_disk(out_dir),
    ]


def verdict(checks: list[Check]) -> tuple[str, int]:
    by = {c.name: c for c in checks}

    def failed(name):  # FAIL only — WARN/SKIP do not block classification
        return by.get(name) is None or by[name].status == Status.FAIL

    def smoke_ok():
        s = by.get("smoke")
        if s is None:
            return False
        # PASS = confirmed; WARN = samples captured but addr extraction uncertain —
        # still collectible, just note the caveat.
        return s.status in (Status.PASS, Status.WARN) and s.data.get("samples", 0) > 0

    ready_blockers = ["os", "wsl2", "perf_installed", "facility"]
    is_ready = not any(failed(n) for n in ready_blockers) and smoke_ok()

    smoke = by.get("smoke")
    smoke_caveat = (
        " (address extraction unconfirmed — run `sudo perf script -F event,addr` "
        "to verify manually)"
        if smoke and smoke.status == Status.WARN else ""
    )

    if is_ready:
        return (
            "READY: publication-grade perf mem collection AVAILABLE "
            "(nanosecond timestamps + memory-region labelling)"
            + smoke_caveat
            + ". Note: region labelling (region_map) requires llama.cpp GGUF models; "
            "PyTorch/transformers runs report region_type=unknown.",
            0,
        )

    if by.get("memray") and by["memray"].status == Status.PASS:
        return ("DEGRADED: perf mem unavailable -> memray fallback only "
                "(no nanosecond timestamps, empty region_map, retention_p99_s=0). "
                "Fix the FAILs above before collecting paper data.", 1)

    return ("CANNOT COLLECT: perf mem unavailable and no memray fallback. "
            "Resolve the FAILs above (most likely: dual-boot native Linux).", 2)


_GLYPH = {Status.PASS: "✓", Status.WARN: "!", Status.FAIL: "✗",
          Status.INFO: "·", Status.SKIP: "–"}


def validate_machine(out_dir: str = ".", json_output: bool = False) -> int:
    """Run every check, print a report, and return a process exit code.

    This is the public entry point. The typer CLI imports *this* function;
    `main()` below is only the argv shim for standalone `python validate_machine.py`.

        0  READY — publication-grade perf-mem collection with region labelling
        1  degraded only (memray fallback)
        2  cannot collect at all
    """
    checks = run_all(out_dir)
    summary, code = verdict(checks)

    if json_output:
        print(json.dumps(
            {"checks": [asdict(c) for c in checks], "verdict": summary, "exit_code": code},
            indent=2, default=str,
        ))
        return code

    print("\n  mrm-trace — machine validation\n  " + "-" * 48)
    for c in checks:
        print(f"  [{_GLYPH[c.status]}] {c.status.value:<4} {c.name:<16} {c.detail}")
        if c.remediation and c.status in (Status.FAIL, Status.WARN):
            print(f"        -> {c.remediation}")
    print("  " + "-" * 48)
    print(f"  VERDICT: {_green(summary) if code == 0 else summary}\n")
    return code


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out_dir = "."
    for i, a in enumerate(argv):
        if a in ("--out", "-o") and i + 1 < len(argv):
            out_dir = argv[i + 1]
    return validate_machine(out_dir=out_dir, json_output="--json" in argv)


if __name__ == "__main__":
    sys.exit(main())

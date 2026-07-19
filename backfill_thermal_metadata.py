#!/usr/bin/env python3
"""
backfill_thermal_metadata.py — inject CPU thermal + wall-clock duration data
into existing runs' metadata.json, sourced from results/.experiment_state.json.

Why this exists: demo_perf_mem.py didn't write thermal/timing data into
metadata.json until this fix (see mrm_trace/telemetry/thermal.py and
build_metadata() in notebooks/scripts/demo_perf_mem.py). experiment_runner.py
already measured peak_temp_c / throttled / duration_s for every run it
launched and stored them in .experiment_state.json — this script recovers
that data for already-collected runs so they don't need to be re-run.

Caveat: .experiment_state.json has no per-run thermal *throttle-event count*
(only a coarse point-in-time frequency check + peak temperature), so the
backfilled run_validity is a peak-temperature proxy, not the throttle-event
based classification new runs get from mrm_trace.telemetry.validity.classify_run.

Usage:
    python3 backfill_thermal_metadata.py --dry-run
    python3 backfill_thermal_metadata.py
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Peak temp thresholds for the backfill proxy classification. Consumer x86
# Tjmax is ~100-105°C; sustained throttling is effectively guaranteed at/above
# ~95°C and likely above ~90°C. See TEMP_CONTAMINATED_FALLBACK_C in
# experiment_runner.py for the equivalent forward-looking fallback.
PROXY_CONTAMINATED_C = 95.0
PROXY_MARGINAL_C = 90.0

BACKFILL_NOTE = (
    "Backfilled from results/.experiment_state.json (experiment_runner.py's "
    "own peak-temperature tracking). No per-run throttle-event count was "
    "captured for this run, so run_validity here is a peak-temperature proxy "
    "(>= {:.0f}C contaminated, >= {:.0f}C marginal, else clean-ish), not the "
    "throttle-event-based CLAUDE.md classification new runs get."
).format(PROXY_CONTAMINATED_C, PROXY_MARGINAL_C)


def proxy_validity(peak_temp_c: float) -> str:
    if peak_temp_c >= PROXY_CONTAMINATED_C:
        return "contaminated"
    if peak_temp_c >= PROXY_MARGINAL_C:
        return "marginal"
    return "clean"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state-file", default="results/.experiment_state.json", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="Print what would change, write nothing")
    ap.add_argument("--force", action="store_true", help="Overwrite an existing 'thermal' key if present")
    args = ap.parse_args()

    if not args.state_file.exists():
        sys.exit(f"state file not found: {args.state_file}")

    state = json.loads(args.state_file.read_text())
    runs = state.get("runs", {})

    patched, skipped_no_dir, skipped_no_meta, skipped_existing, missing_temp = 0, 0, 0, 0, 0
    validity_counts = {"clean": 0, "marginal": 0, "contaminated": 0}

    for key, entry in runs.items():
        if entry.get("status") != "done":
            continue
        output_dir = entry.get("output_dir")
        if not output_dir:
            skipped_no_dir += 1
            continue

        meta_path = Path(output_dir) / "metadata.json"
        if not meta_path.exists():
            print(f"[WARN] {key}: no metadata.json at {meta_path}", file=sys.stderr)
            skipped_no_meta += 1
            continue

        peak_temp_c = entry.get("peak_temp_c")
        if peak_temp_c is None:
            missing_temp += 1
            continue

        metadata = json.loads(meta_path.read_text())
        if "thermal" in metadata and not args.force:
            skipped_existing += 1
            continue

        validity = proxy_validity(peak_temp_c)
        validity_counts[validity] += 1

        metadata["thermal"] = {
            "peak_temp_c": peak_temp_c,
            "throttled": entry.get("throttled", False),
            "run_validity": validity,
            "run_validity_note": BACKFILL_NOTE,
            "source": "backfilled_from_experiment_state",
            "backfilled_at": datetime.now(timezone.utc).isoformat(),
        }
        metadata.setdefault("timing", {})["total_wall_duration_s"] = entry.get("duration_s")
        metadata["timing"]["source"] = "backfilled_from_experiment_state"

        if args.dry_run:
            print(f"[DRY] {key}: peak={peak_temp_c}C -> {validity}  ({meta_path})")
        else:
            meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(f"[OK]  {key}: peak={peak_temp_c}C -> {validity}  ({meta_path})")
        patched += 1

    print()
    print(f"Patched          : {patched}  ({'dry-run, nothing written' if args.dry_run else 'written'})")
    print(f"  clean          : {validity_counts['clean']}")
    print(f"  marginal       : {validity_counts['marginal']}")
    print(f"  contaminated   : {validity_counts['contaminated']}")
    print(f"Skipped (no dir)      : {skipped_no_dir}")
    print(f"Skipped (no metadata) : {skipped_no_meta}")
    print(f"Skipped (already has 'thermal', use --force) : {skipped_existing}")
    print(f"Skipped (no peak_temp_c in state) : {missing_temp}")


if __name__ == "__main__":
    main()

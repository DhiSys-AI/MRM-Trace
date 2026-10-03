#!/usr/bin/env python3
"""
Audit exported runs against the invariants their metrics depend on.

    python notebooks/scripts/audit_run.py <run_dir> [more_run_dirs...]
    python notebooks/scripts/audit_run.py results/<machine>/<model>/*/

Exit 0 if every exported run passes, 1 otherwise.  Incomplete directories
(interrupted before export) are reported as SKIP, not failures.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from mrm_trace.reporter.run_audit import audit_run  # noqa: E402


def main(argv) -> int:
    dirs = [pathlib.Path(a) for a in argv]
    if not dirs:
        print(__doc__)
        return 1
    bad = skipped = 0
    for d in dirs:
        audit = audit_run(d)
        ok = not audit.failures
        if audit.incomplete:
            status, skipped = "SKIP", skipped + 1
        else:
            status = "PASS" if ok else "FAIL"
            bad += 0 if ok else 1
        print(f"[{status}] {d.name}")
        for f in audit.failures:
            print(f"    x {f}")
        for note in audit.notes:
            print(f"    - {note}")
    judged = len(dirs) - skipped
    print(f"\n{judged - bad}/{judged} exported runs pass the audit"
          + (f"  ({skipped} incomplete, skipped)" if skipped else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""
Standalone entry point — runs before the venv/package is installed.

If mrm-trace is already installed, prefer:
    mrm-trace validate --machine
    mrm-trace validate --machine --json

Once the package is installed you can also run:
    python -m mrm_trace.validate_machine [--json] [--out DIR]
"""
import sys

try:
    from mrm_trace.validate_machine import main
except ImportError:
    # Package not installed — execute the full module inline via its own path.
    import importlib.util, pathlib
    _spec = importlib.util.spec_from_file_location(
        "_validate_machine_standalone",
        pathlib.Path(__file__).parent / "mrm_trace" / "validate_machine.py",
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    main = _mod.main

if __name__ == "__main__":
    sys.exit(main())

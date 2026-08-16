#!/usr/bin/env python3
"""Verify the cascadia_obs_ensemble environment: import the key dependencies and
report their versions. Run via `pixi run verify` (default env) after install.

Exits non-zero if any *core* dependency is missing, so it doubles as a CI/install
smoke test. Optional deps (heavy ML stack, internal pnwstore, paper toolchain) are
reported but do not fail the check.
"""
from __future__ import annotations

import importlib
import sys

# (import_name, human_name, is_core)
CORE = [
    ("numpy", "numpy", True),
    ("scipy", "scipy", True),
    ("pandas", "pandas", True),
    ("matplotlib", "matplotlib", True),
    ("obspy", "obspy", True),
    ("h5py", "h5py", True),
]
NOTEBOOK = [
    ("IPython", "ipython", True),
    ("ipykernel", "ipykernel", True),
    ("notebook", "jupyter notebook", False),
]
PLOTTING = [
    ("mpl_toolkits.basemap", "basemap", False),
    ("pygmt", "pygmt", False),
]
ML = [
    ("torch", "torch", False),
    ("seisbench", "seisbench", False),
    ("dask", "dask", False),
    ("tqdm", "tqdm", False),
    ("openpyxl", "openpyxl", False),
    ("adjustText", "adjustText", False),
]
OPTIONAL = [
    ("ELEP", "ELEP (picking)", False),
    ("pnwstore", "pnwstore (internal env only)", False),
]


def check(group_name, items):
    print(f"\n{group_name}")
    failures = []
    for mod, name, core in items:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            print(f"  ok    {name:28s} {ver}")
        except Exception as e:  # noqa: BLE001
            tag = "FAIL " if core else "miss "
            print(f"  {tag} {name:28s} ({type(e).__name__}: {str(e)[:50]})")
            if core:
                failures.append(name)
    return failures


def main() -> int:
    print(f"python {sys.version.split()[0]}  ({sys.executable})")
    failures = []
    failures += check("core (required)", CORE)
    failures += check("notebook", NOTEBOOK)
    check("plotting (optional)", PLOTTING)
    check("ML / pipeline (optional)", ML)
    check("optional (may be absent)", OPTIONAL)

    print()
    if failures:
        print(f"FAILED: {len(failures)} core dependency import(s) failed: {', '.join(failures)}")
        return 1
    print("OK: all core dependencies import.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

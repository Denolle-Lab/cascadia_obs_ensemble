"""Which Route A magnitude files the analysis, figure and catalog scripts read (decision D-B091).

Default: the converged magnitudes, phase19 solved to its exact minimum (phase19_solver.py, PR #45). They are
on the data disk with their MD5s (<WD1>/magnitude_converged/MD5SUMS.txt; copy in
data/magnitude_converged/MD5SUMS.txt).

The stored files (phase19 LSQR stopped at 3,000 iterations, before D-B091) are not modified and stay
reachable with the choice 'stored':

    choice      v4.1 (catalog v4.1; the paper)
    converged   <WD1>/magnitude_converged/v4.1/cascadia_catalog_M_routeA_converged.csv
    stored      <main checkout>/data/magnitude_v4/cascadia_catalog_M_routeA.csv

    choice      v5.1
    converged   <WD1>/magnitude_converged/v5.1/cascadia_catalog_v5.1_magnitudes_converged.csv
    stored      <WD1>/catalog_v5/v5.1_analysis/mag/cascadia_catalog_v5.1_magnitudes.csv

A choice is 'converged', 'stored' or the path of a file with the same columns. Precedence: the script's
--mag option, then the environment variable (CASCADIA_MAG_V41 or CASCADIA_MAG_V51), then 'converged'.
<WD1> is /wd1/mdenolle_data (environment variable CASCADIA_WD1); the main checkout is found from git
(environment variable CASCADIA_MAIN), as in v51_paths.py.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import v51_paths as vp

CHOICES = ("converged", "stored")
CONV = vp.WD1 / "magnitude_converged"
# MD5 of the converged files (MD5SUMS.txt); the stored v4.1 file is listed for reference
MD5 = {
    "v4.1/cascadia_catalog_M_routeA_converged.csv": "fb49ecc28b75432f49cf1a6e399cd90f",
    "v5.1/cascadia_catalog_v5.1_magnitudes_converged.csv": "d8c68099cbb529d5b0315f18d3b1c2ec",
    "data/magnitude_v4/cascadia_catalog_M_routeA.csv (stored)": "78a445098dd76ccf8c08085cb86f67ca",
}
# v5.1 regional Mc / b table (v51_mc_b.py), as committed in the repository (path relative to the repo root)
MCB_V51 = {"converged": "data/magnitude_converged/mc_b/mc_b_regional.csv",
           "stored": "data/catalog_v5/v5.1_analysis/mag/mc_b/mc_b_regional.csv"}


def _choice(choice: str | os.PathLike | None, env: str) -> str:
    return str(choice or os.environ.get(env) or "converged")


def v41(choice: str | os.PathLike | None = None) -> Path:
    """Route A magnitudes of catalog v4.1 (one row per event_id: M, M_type, ML, ...)."""
    c = _choice(choice, "CASCADIA_MAG_V41")
    if c == "converged":
        return CONV / "v4.1" / "cascadia_catalog_M_routeA_converged.csv"
    if c == "stored":
        return vp.main_data("magnitude_v4", "cascadia_catalog_M_routeA.csv")
    return Path(c)


def v51(choice: str | os.PathLike | None = None) -> Path:
    """Magnitude table of catalog v5.1 (one row per ID: M, M_type, ML, is_earthquake, ...)."""
    c = _choice(choice, "CASCADIA_MAG_V51")
    if c == "converged":
        return CONV / "v5.1" / "cascadia_catalog_v5.1_magnitudes_converged.csv"
    if c == "stored":
        return vp.WD1 / "catalog_v5" / "v5.1_analysis" / "mag" / "cascadia_catalog_v5.1_magnitudes.csv"
    return Path(c)


def mcb_v51(choice: str | None = None) -> str:
    """The v5.1 regional Mc / b table that goes with v51(choice) (repo-relative path; a file path keeps 'converged')."""
    return MCB_V51.get(_choice(choice, "CASCADIA_MAG_V51"), MCB_V51["converged"])


def md5(path: str | os.PathLike) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def describe(path: str | os.PathLike) -> dict:
    """Path and MD5 of the magnitude file actually read (for run_params.json / provenance)."""
    p = Path(path)
    return {"magnitudes": str(p), "magnitudes_md5": md5(p) if p.exists() else None}


def add_argument(ap, which: str = "v4.1") -> None:
    """The --mag option of a script (default: environment variable, else 'converged')."""
    env = "CASCADIA_MAG_V41" if which == "v4.1" else "CASCADIA_MAG_V51"
    ap.add_argument("--mag", default=None,
                    help=f"{which} Route A magnitudes: 'converged' (default, D-B091), 'stored' (before D-B091) or a "
                         f"file path; default from ${env}, else converged (see workflow/05_magnitude/mag_paths.py)")

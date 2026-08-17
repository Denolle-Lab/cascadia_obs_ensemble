#!/usr/bin/env python3
"""Fetch the Delph et al. (2018) Cascadia forearc 3-D shear-velocity model (IRIS EMC
Cascadia_ANT+RF_Delph2018) used by 4_relocation/magnitude/phase13_tomography.py.

Downloads the netCDF (~0.4 MB) into data/tomography/Delph2018.nc (git-ignored).

    python utils/fetch_tomography.py
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    # Delph et al. 2018 forearc Vs model (IRIS EMC), -3..80 km
    "Delph2018.nc":
        "https://ds.iris.edu/files/products/emc/emc-files/Cascadia-ANT+RF-Delph2018.nc",
    # CRESCENT Gen0 Community Velocity Model (He et al. 2026), -4..100 km, whole margin
    # (Figshare doi:10.6084/m9.figshare.31902061, file mcmc_vs_model_masked.nc)
    "CRESCENT_Gen0.nc":
        "https://ndownloader.figshare.com/files/64783392",
}
OUTDIR = ROOT / "data" / "tomography"


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for name, url in MODELS.items():
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(url, OUTDIR / name)
        print(f"  wrote data/tomography/{name} "
              f"({(OUTDIR / name).stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

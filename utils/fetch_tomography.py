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
URL = ("https://ds.iris.edu/files/products/emc/emc-files/"
       "Cascadia-ANT+RF-Delph2018.nc")
OUT = ROOT / "data" / "tomography" / "Delph2018.nc"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("downloading Delph2018 Vs model ...")
    urllib.request.urlretrieve(URL, OUT)
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

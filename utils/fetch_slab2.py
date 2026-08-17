#!/usr/bin/env python3
"""Fetch the Slab2 (Hayes et al., 2018) Cascadia subduction-interface depth grid used
by the depth cross-section (4_relocation/magnitude/phase7_depth_analysis.py).

Downloads the Slab2 distribution from USGS ScienceBase and extracts just the Cascadia
depth model as plain-text xyz into data/slab2/cas_slab2_dep.xyz (git-ignored). ~140 MB
download, one file kept.

    python utils/fetch_slab2.py
"""
from __future__ import annotations

import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL = ("https://www.sciencebase.gov/catalog/file/get/5aa1b00ee4b0b1c392e86467"
       "?name=Slab2Distribute_Mar2018.tar.gz")
MEMBER = "Slab2Distribute_Mar2018/Slab2_TXT/cas_slab2_dep_02.24.18.xyz"
OUT = ROOT / "data" / "slab2" / "cas_slab2_dep.xyz"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        print(f"downloading Slab2 (~140 MB) ...")
        urllib.request.urlretrieve(URL, tmp.name)
        with tarfile.open(tmp.name) as tar:
            f = tar.extractfile(MEMBER)
            OUT.write_bytes(f.read())
    os.unlink(tmp.name)
    n = sum(1 for _ in open(OUT))
    print(f"wrote {OUT.relative_to(ROOT)}  ({n:,} rows: lon,lat,depth)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

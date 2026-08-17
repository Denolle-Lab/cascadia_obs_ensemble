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
# depth model + its vertical uncertainty grid (both used by the megathrust bucket)
MEMBERS = {
    "cas_slab2_dep.xyz": "Slab2Distribute_Mar2018/Slab2_TXT/cas_slab2_dep_02.24.18.xyz",
    "cas_slab2_unc.xyz": "Slab2Distribute_Mar2018/Slab2_TXT/cas_slab2_unc_02.24.18.xyz",
}
OUTDIR = ROOT / "data" / "slab2"


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        print("downloading Slab2 (~140 MB) ...")
        urllib.request.urlretrieve(URL, tmp.name)
        with tarfile.open(tmp.name) as tar:
            for out, member in MEMBERS.items():
                f = tar.extractfile(member)
                if f is None:
                    raise RuntimeError(f"member not found in Slab2 tarball: {member}")
                (OUTDIR / out).write_bytes(f.read())
                with open(OUTDIR / out) as fh:
                    n = sum(1 for _ in fh)
                print(f"wrote data/slab2/{out}  ({n:,} rows: lon,lat,value)")
    os.unlink(tmp.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

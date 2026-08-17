#!/usr/bin/env python3
"""Assemble the Zenodo submission package under data/zenodo/ from the keep-set.

Builds a single, self-describing record (the user's choice: one record, everything)
that mirrors the three nested datasets of the pipeline plus the final QC catalog,
amplitudes, and comparison catalogs. Source files are the paper keep-set (see
utils/organize_review_data.py and data/LINEAGE.md); this script copies them into the
published tree with clean names and writes an md5 checksum manifest. It copies (never
moves) and overwrites the target tree, so it is safe to re-run.

    python utils/assemble_zenodo.py            # dry-run: print the plan
    python utils/assemble_zenodo.py --apply    # copy files + write CHECKSUMS.md5

The 5.5 GB raw ELEP picks are included (single record). data/zenodo/ is git-ignored.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "datasets_all_regions"
ANSS = ROOT / "data" / "datasets_anss"
PKG = ROOT / "data" / "zenodo" / "cascadia-obs-ensemble-catalog-v3"

# (published relative path, source file)  -- pipeline order, three nested datasets first
MAP = [
    # (a) raw ELEP ensemble picks
    ("01_raw_elep_picks/elep_picks_all_regions_2010_2015.csv", SRC / "all_picks_all_regions_2010_2015_ver3.csv"),
    # (b) GENIE-associated picks + events
    ("02_associated_genie/events.csv", SRC / "all_events_2010_2015_ver3.csv"),
    ("02_associated_genie/pick_assignments.csv", SRC / "all_pick_assignments_all_regions_2010_2015_ver3.csv"),
    # (c) relocated origins + picks (GraphDD + cross-correlation)
    ("03_relocated/catalog.csv", SRC / "Cascadia_relocated_catalog_ver_3.csv"),
    ("03_relocated/picks.csv", SRC / "Cascadia_relocated_catalog_picks_ver_3.csv"),
    ("03_relocated/origins_reloc_cog.csv", SRC / "origin_2010_2015_reloc_cog_ver3.csv"),
    ("03_relocated/origins_reloc_cog_cc.csv", SRC / "origin_2010_2015_reloc_cog_ver3_cc.csv"),
    # FINAL QC catalog (the paper figures) + its arrivals/associations/stations
    ("04_final_catalog_qc/events_qc_p4_s4_rms2.5.csv", SRC / "origin_2010_2015_reloc_cog_ver3_cc_p_4_s_4_rms_2_5.csv"),
    ("04_final_catalog_qc/arrivals.csv", SRC / "arrival_2010_2015_reloc_cog_ver3.csv"),
    ("04_final_catalog_qc/associations.csv", SRC / "assoc_2010_2015_reloc_cog_ver3.csv"),
    ("04_final_catalog_qc/stations.csv", SRC / "all_stations_2010_2015_ver3.csv"),
    # amplitudes / magnitude input (latest w_amp table)
    ("05_amplitudes/picks_with_amplitudes.csv", SRC / "Cascadia_updated_catalog_picks_assignment_ver_3_w_amp.csv"),
    # comparison catalogs
    ("06_comparison/anss_2010-2015.csv", ANSS / "anss_2010-15.csv"),
    ("06_comparison/morton_reloc.csv", SRC / "origin_2010_2015_reloc_cog_morton_ver3.csv"),
]


def md5(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="copy files (default: dry-run)")
    args = ap.parse_args()

    missing = [str(s.relative_to(ROOT)) for _, s in MAP if not s.exists()]
    total = sum(s.stat().st_size for _, s in MAP if s.exists())
    print(f"package: {PKG.relative_to(ROOT)}")
    print(f"{len(MAP)} files, {total/1e9:.2f} GB{'  (dry-run)' if not args.apply else ''}\n")
    for dest, s in MAP:
        tag = "  " if s.exists() else "??"
        sz = f"{s.stat().st_size/1e6:8.1f}MB" if s.exists() else "  missing "
        print(f" {tag} {sz}  {dest:52s} <- {s.name}")
    if missing:
        print("\nMISSING sources (fix before publishing):")
        for m in missing:
            print("   ", m)

    if not args.apply:
        print("\nRe-run with --apply to copy + write CHECKSUMS.md5.")
        return 0

    PKG.mkdir(parents=True, exist_ok=True)
    lines = []
    for dest, s in MAP:
        if not s.exists():
            continue
        d = PKG / dest
        d.parent.mkdir(parents=True, exist_ok=True)
        print(f"  copy {dest} ...")
        shutil.copy2(s, d)
        lines.append(f"{md5(d)}  {dest}")
    (PKG / "CHECKSUMS.md5").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {len(lines)} files + CHECKSUMS.md5 to {PKG.relative_to(ROOT)}")
    print("Next: add README.md (provenance/columns/license) + create the Zenodo record.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

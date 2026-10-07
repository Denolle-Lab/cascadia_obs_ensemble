#!/usr/bin/env python3
"""Assemble the Zenodo data record under data/zenodo/ (dry-run by default).

Layout of the record "Cascadia OBS ensemble earthquake catalog 2010-2015, v3":

    README.md  DATA_DICTIONARY.md  LINEAGE.md  CHECKSUMS.md5
    catalog/        origins_v3.csv  arrivals_v3.csv  picks_v3.csv.gz   (ANSS/QuakeML split)
    magnitude/      station terms, matched ComCat moment tensors
    intermediate/   GENIE association output (events, pick assignments), for provenance
    comparison/     ANSS ComCat and Morton et al. (2023) catalogs used in the paper

The three catalog tables come from utils/build_anss_tables.py (run it first). This
script copies (never moves) and overwrites the target tree, so it is safe to re-run.

    python utils/build_anss_tables.py          # -> data/catalog_v3/
    python utils/assemble_zenodo.py            # dry-run: print the plan and sizes
    python utils/assemble_zenodo.py --apply    # copy files + write CHECKSUMS.md5

data/zenodo/ is git-ignored.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "datasets_all_regions"
TABLES = ROOT / "data" / "catalog_v3"
ANSS = ROOT / "data" / "datasets_anss"
MAG = ROOT / "data" / "magnitude"
PKG = ROOT / "data" / "zenodo" / "cascadia-obs-ensemble-catalog-v3"

# (published relative path, source file)
MAP = [
    ("README.md", ROOT / "data" / "zenodo_README.md"),
    ("DATA_DICTIONARY.md", ROOT / "data" / "DATA_DICTIONARY.md"),
    ("LINEAGE.md", ROOT / "data" / "LINEAGE.md"),
    # the catalog: three ANSS-style tables
    ("catalog/origins_v3.csv", TABLES / "origins_v3.csv"),
    ("catalog/arrivals_v3.csv", TABLES / "arrivals_v3.csv"),
    ("catalog/picks_v3.csv.gz", TABLES / "picks_v3.csv.gz"),
    # magnitude inputs not already in the arrivals (amplitudes are there)
    ("magnitude/station_terms_routeA.csv", MAG / "route_b_station_terms_routeA.csv"),
    ("magnitude/comcat_mt_matched.csv", ROOT / "data" / "focal" / "comcat_mt_matched.csv"),
    # GENIE output, as received
    ("intermediate/genie_events.csv", SRC / "all_events_2010_2015_ver3.csv"),
    ("intermediate/genie_pick_assignments.csv", SRC / "all_pick_assignments_all_regions_2010_2015_ver3.csv"),
    # comparison catalogs
    ("comparison/anss_2010-2015.csv", ANSS / "anss_2010-15.csv"),
    ("comparison/morton_reloc.csv", SRC / "origin_2010_2015_reloc_cog_morton_ver3.csv"),
]


def md5(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="copy files (default: dry-run)")
    ap.add_argument("--out", type=Path, default=PKG, help="package directory")
    args = ap.parse_args()

    missing = [str(s.relative_to(ROOT)) for _, s in MAP if not s.exists()]
    total = sum(s.stat().st_size for _, s in MAP if s.exists())
    print(f"package: {args.out}")
    print(f"{len(MAP)} files, {total/1e9:.2f} GB{'  (dry-run)' if not args.apply else ''}\n")
    for dest, s in MAP:
        tag = "  " if s.exists() else "??"
        sz = f"{s.stat().st_size/1e6:9.1f} MB" if s.exists() else "   missing  "
        print(f" {tag} {sz}  {dest:42s} <- {s.relative_to(ROOT)}")
    if missing:
        print("\nMISSING sources (run utils/build_anss_tables.py, or fix before publishing):")
        for m in missing:
            print("   ", m)

    if not args.apply:
        print("\nRe-run with --apply to copy + write CHECKSUMS.md5.")
        return 0
    if missing:
        raise SystemExit("not applying with missing sources")

    free = shutil.disk_usage(args.out.parent if args.out.parent.exists() else ROOT).free
    if free < 1.2 * total:
        raise SystemExit(f"only {free/1e9:.1f} GB free for a {total/1e9:.1f} GB package; use --out")

    args.out.mkdir(parents=True, exist_ok=True)
    lines = []
    for dest, s in MAP:
        d = args.out / dest
        d.parent.mkdir(parents=True, exist_ok=True)
        print(f"  copy {dest} ...")
        shutil.copy2(s, d)
        lines.append(f"{md5(d)}  {dest}")
    (args.out / "CHECKSUMS.md5").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {len(lines)} files + CHECKSUMS.md5 to {args.out}")
    print("Check: cd <package> && md5sum -c CHECKSUMS.md5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

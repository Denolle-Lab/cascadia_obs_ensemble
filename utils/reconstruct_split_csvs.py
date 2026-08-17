#!/usr/bin/env python3
"""Reconstruct the monolithic CSVs from the GitHub-friendly chunks in
``data/split_files/`` -- the inverse of ``utils/split_large_csvs.py``.

The repo ships its large pick/catalog CSVs split into <=50 MB parts to stay under
GitHub's file-size limits. This rebuilds the full files (e.g.
``Cascadia_relocated_catalog_picks_ver_3.csv``) so a fresh clone is analysis-ready
without the server. Reconstructed monoliths land in ``data/`` and are git-ignored
(the repo tracks only the chunks).

Usage:
  python utils/reconstruct_split_csvs.py            # rebuild all -> data/<name>.csv
  python utils/reconstruct_split_csvs.py --list     # show what would be rebuilt
  python utils/reconstruct_split_csvs.py --name Cascadia_relocated_catalog_picks_ver_3
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPLIT = ROOT / "data" / "split_files"


def groups() -> dict[str, list[Path]]:
    # Only the standard <name>_partNNN.csv chunks; skip the superseded, differently
    # named *_part001_old.csv provenance files.
    g: dict[str, list[Path]] = {}
    for p in sorted(SPLIT.glob("*_part*.csv")):
        m = re.match(r"^(.+)_part\d+\.csv$", p.name)
        if not m:
            continue
        g.setdefault(m.group(1), []).append(p)
    return {k: sorted(v) for k, v in g.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(ROOT / "data"))
    ap.add_argument("--name", default=None, help="reconstruct only this base name")
    ap.add_argument("--list", action="store_true", help="list groups, do not write")
    args = ap.parse_args()

    g = groups()
    if not g:
        print(f"no *_part*.csv found in {SPLIT}")
        return 1
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    for base, parts in g.items():
        if args.name and base != args.name:
            continue
        if args.list:
            mb = sum(p.stat().st_size for p in parts) / 1e6
            print(f"  {base}.csv  <- {len(parts)} parts (~{mb:.0f} MB)")
            continue
        # stream the parts (each a CSV with a header) into one file rather than
        # loading multi-GB reconstructions fully into memory via pd.concat.
        dest = out / f"{base}.csv"
        rows = 0
        with open(dest, "w") as w:
            for i, p in enumerate(parts):
                with open(p) as r:
                    header = r.readline()
                    if i == 0:
                        w.write(header)
                    for line in r:
                        w.write(line)
                        rows += 1
        print(f"  ok  {dest.name:56s} {rows:>10,} rows  <- {len(parts)} parts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Triage the downloaded server data (``data/datasets_{all_regions,anss}/``) into the
**Zenodo keep-set** (the best datasets used in the paper) versus **everything else**,
which is moved to a local review folder (``data/_review/``) with a per-file manifest
that records its lineage stage and a junk flag.

Keep-set files stay in place (the repointed figure notebooks read them from
``data/datasets_all_regions``); they are the proposed Zenodo package (see
``data/ZENODO.md`` / ``data/LINEAGE.md``). Nothing is deleted -- non-keep files are
*moved* (reversible) and the manifest records their original location.

Usage:
  python utils/organize_review_data.py            # dry-run: print the plan, write no files
  python utils/organize_review_data.py --apply    # move non-keep -> data/_review/, write MANIFEST.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIRS = [ROOT / "data" / "datasets_all_regions", ROOT / "data" / "datasets_anss"]
REVIEW = ROOT / "data" / "_review"

# The best datasets used in the paper -> the proposed Zenodo package. These stay in place.
KEEP = {
    # (a) raw ELEP picks
    "all_picks_all_regions_2010_2015_ver3.csv",
    # (b) GENIE associated picks + events (Ian McBrearty)
    "all_events_2010_2015_ver3.csv",
    "all_pick_assignments_all_regions_2010_2015_ver3.csv",
    # (c) relocated origins + picks (GraphDD + cross-correlation)
    "Cascadia_relocated_catalog_ver_3.csv",
    "Cascadia_relocated_catalog_picks_ver_3.csv",
    "Cascadia_relocated_catalog_picks_ver_3_with_amplitudes.csv",
    "origin_2010_2015_reloc_cog_ver3.csv",
    "origin_2010_2015_reloc_cog_ver3_cc.csv",
    "origin_2010_2015_reloc_cog_ver3_cc_p_4_s_4_rms_2_5.csv",   # FINAL QC catalog (figures)
    "arrival_2010_2015_reloc_cog_ver3.csv",
    "assoc_2010_2015_reloc_cog_ver3.csv",
    "all_stations_2010_2015_ver3.csv",
    # amplitude / magnitude input (latest)
    "Cascadia_updated_catalog_picks_assignment_ver_3_w_amp.csv",
    # comparison catalogs
    "origin_2010_2015_reloc_cog_morton_ver3.csv",
    "anss_2010-15.csv",
}


def classify(name: str) -> tuple[str, bool, str]:
    """Return (category, is_junk, note) for a non-keep file."""
    n = name.lower()
    if "temp" in n:
        return "junk", True, "pipeline/rsync scratch (*_temp) -- safe to delete"
    if "test" in n:
        return "junk", True, "test/debug output (*_test) -- safe to delete"
    if "_old" in n:
        return "junk", True, "superseded old computation -- safe to delete"
    if re.search(r"w_amp_\d", n):
        return "superseded", False, "earlier amplitude run (kept only w_amp.csv)"
    if re.search(r"ver_?1(?![0-9])", n):
        return "superseded", False, "version 1 (pre-final catalog)"
    if re.search(r"ver_?2(?![0-9])", n):
        return "superseded", False, "version 2 (pre-final catalog)"
    if re.search(r"all_picks_\d", n) or "for_picking" in n or "for_assoc" in n:
        return "intermediate", False, "regional/pre-merge raw picks (subsumed by all_picks_all_regions_ver3)"
    if n.startswith("all_pick_assignments_") and "all_regions" not in n:
        return "intermediate", False, "regional pick assignments (subsumed by the all_regions version)"
    return "alternative", False, "other product; kept for provenance"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="move non-keep files (default: dry-run)")
    args = ap.parse_args()

    rows, counts = [], {}
    for d in SRC_DIRS:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.csv")):
            keep = f.name in KEEP
            if keep:
                cat, junk, note = "KEEP", False, "Zenodo package (paper dataset); stays in place"
            else:
                cat, junk, note = classify(f.name)
            st = f.stat()
            rows.append(dict(file=f.name, dir=d.name, category=cat, junk=junk,
                             size_mb=round(st.st_size / 1e6, 1),
                             created=datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d"),
                             note=note, orig_path=str(f.relative_to(ROOT))))
            counts[cat] = counts.get(cat, 0) + 1

    print("category      count   total GB")
    for cat in ("KEEP", "alternative", "superseded", "intermediate", "junk"):
        sel = [r for r in rows if r["category"] == cat]
        print(f"  {cat:11s} {len(sel):5d}   {sum(r['size_mb'] for r in sel)/1e3:7.1f}")
    print(f"  {'TOTAL':11s} {len(rows):5d}   {sum(r['size_mb'] for r in rows)/1e3:7.1f}")

    if not args.apply:
        print(f"\n(dry-run) {sum(1 for r in rows if r['category'] != 'KEEP')} files would move to "
              f"{REVIEW.relative_to(ROOT)}/. Re-run with --apply.")
        return 0

    REVIEW.mkdir(parents=True, exist_ok=True)
    moved = 0
    for r in rows:
        if r["category"] == "KEEP":
            continue
        dst = REVIEW / r["category"] / r["file"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(ROOT / r["orig_path"]), str(dst))
        moved += 1
    with open(REVIEW / "MANIFEST.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nmoved {moved} files -> {REVIEW.relative_to(ROOT)}/<category>/; "
          f"manifest: {(REVIEW / 'MANIFEST.csv').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

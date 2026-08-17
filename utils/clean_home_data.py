#!/usr/bin/env python3
"""Reclaim space in a working directory (e.g. a cramped $HOME on psound) by removing
data files that are safely redundant, while NEVER touching the authoritative copy.

For every file under --dir (above --min-mb) it decides:

  dup-of-ref   byte-identical (size+md5) to a same-named file under --reference
               (e.g. /wd1/.../data) -> safe to delete here; the data lives on --reference
  dup-in-dir   byte-identical to another file already seen under --dir -> keep the first,
               the rest are redundant copies
  junk         name matches *_temp* / *_test* / *_old -> scratch output
  superseded   ver_1 / ver2 / dated *_w_amp_<date> -> earlier versions
  keep         everything else (unique here, not on --reference) -> left alone

Dry-run by default: prints the plan and how much space each bucket frees. With --apply it
deletes the dup/junk buckets (add --superseded to also drop superseded), writes a
manifest (deleted path, size, md5, reason) to --dir/CLEAN_MANIFEST.csv, and never deletes
anything under --reference.

  python utils/clean_home_data.py --dir ~/cascadia_obs_ensemble/data \
      --reference /wd1/hbito_data/data                       # dry-run
  python utils/clean_home_data.py --dir ... --reference ... --apply
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
from pathlib import Path


def md5(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def classify_name(name: str):
    n = name.lower()
    if "temp" in n or "test" in n or "_old" in n:
        return "junk"
    if re.search(r"ver_?1(?![0-9])", n) or re.search(r"ver_?2(?![0-9])", n):
        return "superseded"
    if re.search(r"w_amp_\d", n):
        return "superseded"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="directory to clean (e.g. home data)")
    ap.add_argument("--reference", help="authoritative copy that is never touched "
                    "(e.g. /wd1/.../data); files identical here are removable from --dir")
    ap.add_argument("--min-mb", type=float, default=1.0, help="ignore files smaller than this")
    ap.add_argument("--fast", action="store_true",
                    help="match by name+size only (skip md5; faster but less strict)")
    ap.add_argument("--superseded", action="store_true", help="also delete superseded versions")
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry-run)")
    args = ap.parse_args()

    d = Path(os.path.expanduser(args.dir)).resolve()
    ref = Path(os.path.expanduser(args.reference)).resolve() if args.reference else None

    # index the reference by basename -> list of (size, path) for quick candidate match
    ref_by_name: dict[str, list[tuple[int, Path]]] = {}
    if ref and ref.exists():
        for p in ref.rglob("*"):
            if p.is_file():
                ref_by_name.setdefault(p.name, []).append((p.stat().st_size, p))

    def same_as_ref(p: Path, sz: int, digest):
        for rsz, rp in ref_by_name.get(p.name, []):
            if rsz != sz:
                continue
            if args.fast:
                return rp
            if digest() == md5(rp):
                return rp
        return None

    seen_by_key: dict[tuple, Path] = {}       # (size, md5|None) -> first path in --dir
    rows, buckets = [], {}
    for p in sorted(d.rglob("*")):
        if not p.is_file() or p.name == "CLEAN_MANIFEST.csv":
            continue
        sz = p.stat().st_size
        if sz < args.min_mb * 1e6:
            continue
        _digest_cache = {}

        def digest():
            if "v" not in _digest_cache:
                _digest_cache["v"] = md5(p)
            return _digest_cache["v"]

        reason, note = "keep", ""
        rp = same_as_ref(p, sz, digest) if ref_by_name else None
        if rp is not None:
            reason, note = "dup-of-ref", str(rp)
        else:
            key = (sz, None if args.fast else digest())
            if key in seen_by_key:
                reason, note = "dup-in-dir", str(seen_by_key[key])
            else:
                seen_by_key[key] = p
                reason = classify_name(p.name) or "keep"
        rows.append(dict(path=str(p), rel=str(p.relative_to(d)), size_mb=round(sz / 1e6, 1),
                         reason=reason, note=note,
                         md5=("" if args.fast else _digest_cache.get("v", ""))))
        buckets[reason] = buckets.get(reason, 0) + sz

    delete = {"dup-of-ref", "dup-in-dir", "junk"} | ({"superseded"} if args.superseded else set())
    freeable = sum(sz for r, sz in buckets.items() if r in delete)
    print(f"scanned {d}  ({len(rows)} files >= {args.min_mb} MB)"
          + (f"  vs reference {ref}" if ref else "  (no reference given)"))
    for r in ["dup-of-ref", "dup-in-dir", "junk", "superseded", "keep"]:
        n = sum(1 for x in rows if x["reason"] == r)
        gb = buckets.get(r, 0) / 1e9
        mark = "DELETE" if r in delete else ("skip" if r != "keep" else "keep ")
        print(f"  [{mark}] {r:12s} {n:5d} files  {gb:7.2f} GB")
    print(f"  -> would free {freeable/1e9:.2f} GB"
          + ("" if args.superseded else "  (add --superseded to also drop superseded)"))

    if not args.apply:
        print("\n(dry-run) re-run with --apply to delete the DELETE buckets. Nothing removed.")
        return 0

    man = d / "CLEAN_MANIFEST.csv"
    with open(man, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "rel", "size_mb", "reason", "note", "md5"])
        w.writeheader()
        w.writerows(rows)
    removed = 0
    for x in rows:
        if x["reason"] in delete:
            os.remove(x["path"])
            removed += 1
    print(f"\ndeleted {removed} files, freed {freeable/1e9:.2f} GB. "
          f"manifest (incl. kept files): {man}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

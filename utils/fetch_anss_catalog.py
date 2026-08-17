#!/usr/bin/env python3
"""Regenerate the ANSS/ComCat reference catalog data/datasets_anss/anss_2010-15.csv.

fig1 and fig3 plot the ANSS (USGS ComCat) catalog as an independent comparison for
the offshore Cascadia region. The original file was built by hand
(4_relocation/concat_anss_catalogs_2010_2015.ipynb): twelve half-year CSVs downloaded
from the USGS ComCat web search (split by half-year to stay under the 20,000-events
per-query cap) and concatenated. This script reproduces that programmatically by
querying the USGS FDSN event service (= ANSS ComCat) in half-year chunks and, if a
chunk still exceeds the cap, recursively bisecting it in time. Output columns are the
native ComCat schema (time, latitude, longitude, depth, mag, ...), saved with a leading
index column so fig1's `pd.read_csv(..., index_col=0)` works unchanged.

Usage:
  python utils/fetch_anss_catalog.py                 # defaults below (offshore Cascadia box)
  python utils/fetch_anss_catalog.py --minlat 39 --maxlat 51.5 --minlon -131 --maxlon -119
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FDSN = "https://earthquake.usgs.gov/fdsnws/event/1/query"
CAP = 20000  # USGS per-query event cap


def fetch(start: str, end: str, box: dict) -> pd.DataFrame:
    """Fetch one time window as a ComCat CSV; bisect in time if it hits the 20k cap."""
    q = {"format": "csv", "starttime": start, "endtime": end, "orderby": "time-asc", **box}
    url = f"{FDSN}?{urllib.parse.urlencode(q)}"
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            txt = r.read().decode()
    except urllib.error.HTTPError as e:
        if e.code == 400 and start < end:  # too many events -> split in half
            mid = (pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2).strftime("%Y-%m-%dT%H:%M:%S")
            print(f"    {start[:10]}..{end[:10]} exceeded cap; bisecting at {mid[:10]}")
            return pd.concat([fetch(start, mid, box), fetch(mid, end, box)], ignore_index=True)
        raise
    df = pd.read_csv(io.StringIO(txt)) if txt.strip() else pd.DataFrame()
    if len(df) >= CAP and start < end:  # hit the cap exactly -> may be truncated; bisect
        mid = (pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2).strftime("%Y-%m-%dT%H:%M:%S")
        print(f"    {start[:10]}..{end[:10]} returned {len(df)} (>= cap); bisecting at {mid[:10]}")
        return pd.concat([fetch(start, mid, box), fetch(mid, end, box)], ignore_index=True)
    print(f"    {start[:10]}..{end[:10]}: {len(df)} events")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # defaults padded from the paper catalog bounds (lat 39.15..51.16, lon -130.23..-119.04)
    ap.add_argument("--minlat", type=float, default=39.0)
    ap.add_argument("--maxlat", type=float, default=51.5)
    ap.add_argument("--minlon", type=float, default=-131.0)
    ap.add_argument("--maxlon", type=float, default=-119.0)
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2016-01-01")   # exclusive upper bound (through 2015)
    ap.add_argument("--out", default=str(ROOT / "data" / "datasets_anss" / "anss_2010-15.csv"))
    args = ap.parse_args()

    box = {"minlatitude": args.minlat, "maxlatitude": args.maxlat,
           "minlongitude": args.minlon, "maxlongitude": args.maxlon}
    print(f"ANSS/ComCat  box lat[{args.minlat},{args.maxlat}] lon[{args.minlon},{args.maxlon}]  "
          f"{args.start}..{args.end}")

    # half-year windows (mirrors the original 12-file split)
    edges = pd.date_range(args.start, args.end, freq="6MS").strftime("%Y-%m-%dT00:00:00").tolist()
    parts = [fetch(edges[i], edges[i + 1], box) for i in range(len(edges) - 1)]

    cat = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    cat = cat.drop_duplicates(subset="id").sort_values("time").reset_index(drop=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cat.to_csv(out)  # default index=True -> leading index col (fig1 uses index_col=0)
    print(f"\nwrote {len(cat)} events -> {out.relative_to(ROOT)}  "
          f"({', '.join(cat.columns[:5])}, ...)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

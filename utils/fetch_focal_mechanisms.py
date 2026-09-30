#!/usr/bin/env python3
"""Fetch moment tensors for the Cascadia map region from USGS ComCat.

For every ComCat event with a ``moment-tensor`` product (M >= --minmag, 1970 to
now) inside the full-margin map box, keep the PREFERRED moment tensor (the one
ComCat ranks highest: usually the US W-phase / GCMT solution for M5+, the NC/UW
regional TMTS for smaller events) and write one row per event:

    data/focal/comcat_mt.csv
    id, time, lat, lon, depth, mag, magtype, mt_source, mt_type,
    mrr, mtt, mpp, mrt, mrp, mtp  (N m),  strike1, dip1, rake1

Re-running only fetches events missing from the CSV.

    pixi run python utils/fetch_focal_mechanisms.py
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "focal" / "comcat_mt.csv"
FDSN = "https://earthquake.usgs.gov/fdsnws/event/1/query"
BOX = dict(minlatitude=38.5, maxlatitude=51.5, minlongitude=-130.8, maxlongitude=-118.5)


def get_json(params, tries=8):
    """GET with backoff; ComCat answers 429 when polled faster than ~2 requests/s."""
    url = f"{FDSN}?{urllib.parse.urlencode(params)}"
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            if k == tries - 1:
                raise
            wait = 30 * (k + 1) if getattr(e, "code", None) == 429 else 3 * (k + 1)
            time.sleep(wait)


def preferred_mt(detail):
    mts = detail["properties"]["products"].get("moment-tensor", [])
    mts = [m for m in mts if "tensor-mrr" in m["properties"]]
    if not mts:
        return None
    m = max(mts, key=lambda x: x.get("preferredWeight", 0))
    p = m["properties"]
    f = lambda k: float(p[k]) if k in p else float("nan")
    return dict(mt_source=m["source"], mt_type=p.get("beachball-type", p.get("derived-magnitude-type", "")),
                mt_depth=f("derived-depth"),
                mrr=f("tensor-mrr"), mtt=f("tensor-mtt"), mpp=f("tensor-mpp"),
                mrt=f("tensor-mrt"), mrp=f("tensor-mrp"), mtp=f("tensor-mtp"),
                strike1=f("nodal-plane-1-strike"), dip1=f("nodal-plane-1-dip"),
                rake1=f("nodal-plane-1-rake"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--minmag", type=float, default=4.0)
    ap.add_argument("--start", default="1970-01-01")
    args = ap.parse_args()

    ev = get_json(dict(format="geojson", producttype="moment-tensor", minmagnitude=args.minmag,
                       starttime=args.start, orderby="time-asc", **BOX))["features"]
    print(f"{len(ev)} ComCat events with a moment tensor (M>={args.minmag})")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["id"])
    have = set(done["id"])
    rows = []
    for i, e in enumerate(ev):
        if e["id"] in have:
            continue
        time.sleep(0.6)
        d = get_json(dict(format="geojson", eventid=e["id"]))
        mt = preferred_mt(d)
        if mt is None:
            continue
        pr = e["properties"]
        lon, lat, dep = e["geometry"]["coordinates"]
        rows.append(dict(id=e["id"], time=pd.to_datetime(pr["time"], unit="ms", utc=True).isoformat(),
                         lat=lat, lon=lon, depth=dep, mag=pr["mag"], magtype=pr["magType"],
                         place=pr.get("place", ""), **mt))
        if (i + 1) % 100 == 0 or i == len(ev) - 1:      # checkpoint (re-runs resume)
            out = pd.concat([done, pd.DataFrame(rows)], ignore_index=True).sort_values("time")
            out.to_csv(OUT, index=False)
            print(f"  {i + 1}/{len(ev)}: {len(out)} moment tensors in {OUT}", flush=True)
    out = pd.concat([done, pd.DataFrame(rows)], ignore_index=True).sort_values("time")
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(out)} moment tensors)")


if __name__ == "__main__":
    main()

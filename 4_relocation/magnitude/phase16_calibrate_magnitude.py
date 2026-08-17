#!/usr/bin/env python3
"""Calibrate an (uncalibrated / relative) event magnitude to an absolute scale by
regression against co-located ANSS events, and propagate an uncertainty.

Two intended uses once the Route A amplitudes exist:
  --target ml : calibrate the Wood-Anderson-based magnitude to absolute local ML,
                anchored to ANSS ml-typed events.
  --target mw : calibrate the broadband-displacement magnitude (from disp_amp_um) to
                moment magnitude Mw, anchored to ANSS Mw-typed events.
Until then it runs on the current ML catalog as a stand-in (and reproduces the
approximate ML->Mw conversion we quoted).

The input catalog needs event location + origin time and one magnitude-like column
(--mag-col). We match each ANSS anchor event to the nearest catalog event in time and
space, fit target = a + b * mag (Theil-Sen, robust to outliers), and write the whole
catalog with the calibrated magnitude and an uncertainty
sqrt((b*mag_unc)^2 + rms^2).

    python phase16_calibrate_magnitude.py --target mw          # ML catalog stand-in
    python phase16_calibrate_magnitude.py --catalog <routeA_ml.csv> --mag-col M_rel \
        --target ml --out ../../data/magnitude/cascadia_catalog_ML_routeA.csv
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

ANSS = "../../data/datasets_anss/anss_2010-15.csv"
DEFAULT_CAT = "../../data/magnitude/cascadia_catalog_ML_kpos.csv"
MW_TYPES = {"mw", "mww", "mwb", "mwc", "mwr"}
ML_TYPES = {"ml"}


def load_catalog(path, magcol):
    c = pd.read_csv(os.path.expanduser(path))
    lon = next(x for x in ("evlo", "lon", "longitude") if x in c.columns)
    lat = next(x for x in ("evla", "lat", "latitude") if x in c.columns)
    tcol = next(x for x in ("otime", "time", "t") if x in c.columns)
    # origin time may be epoch seconds or an ISO string
    t = pd.to_numeric(c[tcol], errors="coerce")
    c["t"] = (pd.to_datetime(t, unit="s") if t.notna().all()
              else pd.to_datetime(c[tcol], errors="coerce", utc=True).dt.tz_localize(None))
    return c.rename(columns={lon: "lon", lat: "lat", magcol: "mag_in"})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default=DEFAULT_CAT)
    ap.add_argument("--mag-col", default="ML", help="magnitude column to calibrate")
    ap.add_argument("--unc-col", default="ML_unc", help="its uncertainty column, if any")
    ap.add_argument("--target", choices=["ml", "mw"], default="mw")
    ap.add_argument("--dt", type=float, default=45.0, help="match window (s)")
    ap.add_argument("--dd", type=float, default=0.6, help="match window (deg)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    c = load_catalog(args.catalog, args.mag_col)
    a = pd.read_csv(os.path.expanduser(ANSS), index_col=0)
    a["t"] = pd.to_datetime(a["time"], format="%Y-%m-%dT%H:%M:%S.%fZ", errors="coerce")
    types = MW_TYPES if args.target == "mw" else ML_TYPES
    anchors = a[a.magType.str.lower().isin(types) & a.mag.notna()]

    cc = c.dropna(subset=["mag_in", "t", "lat", "lon"])   # candidates with a usable mag
    pairs = []
    for _, e in anchors.iterrows():
        dt = (cc.t - e.t).abs().dt.total_seconds()
        m = cc[(dt < args.dt) & ((cc.lat - e.latitude).abs() < args.dd)
               & ((cc.lon - e.longitude).abs() < args.dd)]
        if len(m):
            # closest anchor->catalog match: smallest space-time separation, not the
            # largest nearby magnitude (which could be an unrelated bigger event)
            sep = ((m.t - e.t).dt.total_seconds() / args.dt) ** 2 \
                + ((m.lat - e.latitude) / args.dd) ** 2 + ((m.lon - e.longitude) / args.dd) ** 2
            pairs.append((m.loc[sep.idxmin(), "mag_in"], e.mag))
    p = pd.DataFrame(pairs, columns=["mag_in", "target"]).dropna()
    if len(p) < 8:
        raise SystemExit(f"only {len(p)} {args.target} anchor matches; cannot calibrate.")

    b, a0, _, _ = theilslopes(p.target, p.mag_in)          # robust target = a0 + b*mag
    rms = float(np.sqrt(np.mean((p.target - (a0 + b * p.mag_in)) ** 2)))
    print(f"{args.target.upper()} calibration: {len(p)} anchors, "
          f"{args.target} = {a0:.2f} + {b:.2f}*{args.mag_col}  (Theil-Sen, RMS {rms:.2f})")
    print(f"  input {args.mag_col} range {p.mag_in.min():.1f}..{p.mag_in.max():.1f} -> "
          f"{args.target} {(a0+b*p.mag_in.min()):.1f}..{(a0+b*p.mag_in.max()):.1f}")

    outcol = args.target.upper() if args.target == "mw" else "ML_cal"
    c[outcol] = a0 + b * c["mag_in"]
    uin = c[args.unc_col] if args.unc_col in c.columns else 0.0
    c[f"{outcol}_unc"] = np.sqrt((b * uin) ** 2 + rms ** 2)
    out = os.path.expanduser(args.out) if args.out else \
        os.path.expanduser(args.catalog).replace(".csv", f"_{outcol}.csv")
    keep = [x for x in (c.columns) if x not in ("t",)]
    c[keep].to_csv(out, index=False)
    print(f"  wrote {out}  (+ {outcol} [median {c[outcol].median():.1f}] and "
          f"{outcol}_unc [median {c[f'{outcol}_unc'].median():.2f}])")


if __name__ == "__main__":
    main()

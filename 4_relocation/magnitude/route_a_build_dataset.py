#!/usr/bin/env python3
"""
Route A, step 2: assemble the analysis-ready amplitude-distance dataset from the raw
Wood-Anderson measurements. Applies the SNR gate and (optionally) folds the response
epoch into the station identifier so the inversion solves an epoch-specific station
term for redeployed OBS. The output matches the schema consumed by
phase3_route_b_relative_magnitude.py, so the rest of the pipeline
(phase3 -> phase2 -> phase4/5/6) runs unchanged with --suffix _routeA.

Usage:
  python route_a_build_dataset.py --raw raw_wa_amplitudes.csv \
      --out ../../data/magnitude/amp_distance_dataset_routeA.csv \
      --min-snr 3 --epoch-station
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True, help="raw_wa_amplitudes.csv")
    ap.add_argument("--out", default="../../data/magnitude/amp_distance_dataset_routeA.csv")
    ap.add_argument("--min-snr", type=float, default=3.0)
    ap.add_argument("--epoch-station", action="store_true",
                    help="append @epoch to the station id (per-deployment station terms)")
    args = ap.parse_args(argv)

    df = pd.read_csv(os.path.expanduser(args.raw))
    n0 = len(df)
    if "reason" in df.columns:
        df = df[df["reason"] == "ok"]
    amp = pd.to_numeric(df["wa_amp_mm"], errors="coerce")
    snr = pd.to_numeric(df.get("snr"), errors="coerce")
    keep = amp.notna() & (amp > 0) & ((snr >= args.min_snr) | snr.isna())
    n_lowsnr = int(((snr < args.min_snr)).sum())
    df = df[keep].copy()
    df["amp"] = amp[keep].to_numpy()
    df["log10A"] = np.log10(df["amp"])

    if args.epoch_station:                      # per-deployment station terms
        ep = df["epoch"].fillna("").astype(str)
        df["station"] = np.where(ep.ne(""), df["station"].astype(str) + "@" + ep,
                                 df["station"].astype(str))

    cols = ["arid", "event_id", "station", "network", "phase", "evla", "evlo", "evdp",
            "stla", "stlo", "stel_m", "dist_hypo_km", "amp", "log10A", "snr", "epoch"]
    cols = [c for c in cols if c in df.columns]
    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df[cols].to_csv(out, index=False)

    print(f"raw rows            : {n0:,}")
    print(f"dropped low-SNR (<{args.min_snr}) : {n_lowsnr:,}")
    print(f"clean observations  : {len(df):,}  ->  {out}")
    print(f"events              : {df['event_id'].nunique():,}")
    print(f"station terms (ids) : {df['station'].nunique():,}"
          + ("  (epoch-keyed)" if args.epoch_station else ""))
    print("phase counts        :", df["phase"].value_counts().to_dict())


if __name__ == "__main__":
    main()

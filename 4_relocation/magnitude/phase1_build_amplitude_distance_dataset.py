#!/usr/bin/env python3
"""
Phase 1 of the magnitude plan (see 4_relocation/magnitude_estimation_plan.md):
build the clean amplitude-distance dataset from the per-pick counts amplitude file.

Input  : Cascadia_updated_catalog_picks_assignment_ver_3_w_amp.csv
         (calculate_amplitudes.py output; has event hypocenter, station coords,
          phase, counts Amplitude). Already contains slatitude/slongitude/selevation,
          so no FDSN station inventory is needed for distances.

Output : a tidy per-(event, station, phase) table with hypocentral distance and
         log10(amplitude), ready for the Route-A/Route-B inversions, plus a
         printed readiness summary and an optional log10(A)-distance diagnostic.

Usage:
    python phase1_build_amplitude_distance_dataset.py \
        --amp-csv "~/Downloads/Cascadia_updated_catalog_picks_assignment_ver_3_w_amp (1).csv" \
        --out ../../data/magnitude/amp_distance_dataset.csv [--plot]

Notes / limitations (carried from the plan):
 - Amplitudes are COUNTS (instrument-response NOT removed). They are used here for
   the response-free relative-cluster route (Route B) and as the raw input to the
   joint inversion where station terms are solved. They are NOT a valid absolute-ML
   input on their own.
 - No SNR is available in this file. SNR gating is deferred (would require the noise
   window from the waveforms); we instead drop non-positive amplitudes and rely on
   robust statistics + residual outlier rejection downstream.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

R_EARTH_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized great-circle (epicentral) distance in km."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--amp-csv", required=True, help="per-pick counts amplitude CSV")
    p.add_argument("--out", default="../../data/magnitude/amp_distance_dataset.csv")
    p.add_argument("--plot", action="store_true", help="save log10(A)-distance diagnostic")
    args = p.parse_args(argv)

    amp_csv = os.path.expanduser(args.amp_csv)
    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    df = pd.read_csv(amp_csv)
    n0 = len(df)

    # ---- required columns (event hypocenter, station coords, phase, amplitude) ----
    need = {
        "arid": "arid", "idx": "event_id", "station": "station", "phase": "phase",
        "latitude": "evla", "longitude": "evlo", "depth": "evdp",
        "slatitude": "stla", "slongitude": "stlo", "selevation": "stel_m",
        "Amplitude": "amp", "timeres": "timeres", "RMS Residual (s)": "rms",
        "Num. P": "num_p", "Num. S": "num_s",
    }
    missing_cols = [c for c in need if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"input is missing expected columns: {missing_cols}")
    df = df[list(need)].rename(columns=need)

    df["network"] = df["station"].astype(str).str.split(".").str[0]

    # ---- QC: keep positive, finite amplitudes and valid coordinates ----
    amp = pd.to_numeric(df["amp"], errors="coerce")
    ok = amp.notna() & (amp > 0)
    for c in ("evla", "evlo", "evdp", "stla", "stlo", "stel_m"):
        ok &= pd.to_numeric(df[c], errors="coerce").notna()
    n_amp_bad = int((~(amp.notna() & (amp > 0))).sum())
    df = df.loc[ok].copy()
    df["amp"] = amp.loc[ok].astype(float)

    # ---- distances (hypocentral; station elevation vs event depth) ----
    df["dist_epi_km"] = haversine_km(df["evla"], df["evlo"], df["stla"], df["stlo"])
    # event depth is km below sea level (positive down); station elevation is m above
    # sea level -> vertical separation = evdp + stel_km
    vert_km = df["evdp"].astype(float) + df["stel_m"].astype(float) / 1000.0
    df["dist_hypo_km"] = np.sqrt(df["dist_epi_km"] ** 2 + vert_km ** 2)
    df["log10A"] = np.log10(df["amp"])

    cols = ["arid", "event_id", "station", "network", "phase",
            "evla", "evlo", "evdp", "stla", "stlo", "stel_m",
            "dist_epi_km", "dist_hypo_km", "amp", "log10A",
            "timeres", "rms", "num_p", "num_s"]
    df[cols].to_csv(out, index=False)

    # ---- readiness summary ----
    print(f"input rows            : {n0:,}")
    print(f"dropped (amp<=0/NaN)  : {n_amp_bad:,}")
    print(f"clean observations    : {len(df):,}  ->  {out}")
    print(f"events                : {df['event_id'].nunique():,}")
    print(f"stations              : {df['station'].nunique():,}")
    print("phase counts          :", df["phase"].value_counts().to_dict())
    print(f"dist_hypo_km          : min={df.dist_hypo_km.min():.1f} "
          f"median={df.dist_hypo_km.median():.1f} max={df.dist_hypo_km.max():.1f}")
    print(f"log10A                : p1={df.log10A.quantile(.01):.2f} "
          f"median={df.log10A.median():.2f} p99={df.log10A.quantile(.99):.2f}")
    # obs-per-station distribution (station terms need enough observations)
    per_sta = df.groupby("station").size()
    print(f"stations with >=8 obs : {(per_sta >= 8).sum():,} / {len(per_sta):,}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
        for ax, ph in zip(axes, ["P", "S"]):
            d = df[df.phase == ph]
            ax.hexbin(d.dist_hypo_km, d.log10A, gridsize=60, bins="log", mincnt=1)
            ax.set_title(f"{ph}  (n={len(d):,})")
            ax.set_xlabel("hypocentral distance (km)")
        axes[0].set_ylabel("log10(amplitude, counts)")
        fig.suptitle("Amplitude vs distance (counts) — attenuation trend + outliers")
        fig.tight_layout()
        png = os.path.splitext(out)[0] + "_log10A_vs_dist.png"
        fig.savefig(png, dpi=130)
        print("diagnostic plot       :", png)


if __name__ == "__main__":
    main()

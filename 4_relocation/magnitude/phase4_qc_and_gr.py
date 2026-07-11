#!/usr/bin/env python3
"""
QC + Gutenberg-Richter analysis for a magnitude catalog produced by the Route B
pipeline (cascadia_catalog_ML.csv).

Produces:
  - <tag>_gutenberg_richter.png : frequency-magnitude distribution (incremental +
        cumulative), magnitude of completeness Mc (maximum-curvature), and the
        maximum-likelihood b-value (Aki-Utsu) with Shi & Bolt uncertainty.
  - <tag>_qc.png : catalog QC panels (ML vs #obs, ML uncertainty, ML vs depth,
        event rate in time, station-magnitude scatter vs ML, ML map scatter).
  - prints the GR/QC summary.

Usage:
    python phase4_qc_and_gr.py --catalog ../../data/magnitude/cascadia_catalog_ML.csv --tag base
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

DM = 0.1  # magnitude bin width


def gutenberg_richter(mags, dm=DM):
    """Max-curvature Mc + Aki-Utsu MLE b-value (Shi & Bolt sigma)."""
    m = np.sort(mags[np.isfinite(mags)])
    edges = np.arange(np.floor(m.min() / dm) * dm, m.max() + dm, dm)
    centers = edges[:-1] + dm / 2
    inc, _ = np.histogram(m, bins=edges)
    cum = np.array([(m >= c - dm / 2).sum() for c in centers])
    mc = centers[np.argmax(inc)]                       # maximum curvature
    sel = m[m >= mc - 1e-9]
    n = len(sel)
    mean_m = sel.mean()
    b = np.log10(np.e) / (mean_m - (mc - dm / 2))      # Aki-Utsu MLE
    sigma_b = 2.30 * b**2 * np.sqrt(((sel - mean_m)**2).sum() / (n * (n - 1)))
    a = np.log10(n) + b * mc
    return dict(centers=centers, inc=inc, cum=cum, mc=mc, b=b, sigma_b=sigma_b,
                a=a, n_above_mc=n, mean_m=mean_m)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", default="../../data/magnitude/cascadia_catalog_ML.csv")
    p.add_argument("--tag", default="base")
    p.add_argument("--outdir", default="../../data/magnitude")
    args = p.parse_args(argv)
    outdir = os.path.expanduser(args.outdir); os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(os.path.expanduser(args.catalog))
    df["otime"] = pd.to_datetime(df["otime"], utc=True, errors="coerce")
    ML = df["ML"].to_numpy()

    gr = gutenberg_richter(ML)
    print(f"=== {args.tag}: catalog QC + Gutenberg-Richter ===")
    print(f"events            : {len(df):,}")
    print(f"ML range / median : {np.nanmin(ML):.2f} .. {np.nanmax(ML):.2f} / {np.nanmedian(ML):.2f}")
    print(f"Mc (max curvature): {gr['mc']:.2f}   (events >= Mc: {gr['n_above_mc']:,})")
    print(f"b-value (MLE)     : {gr['b']:.3f} +/- {gr['sigma_b']:.3f}")
    print(f"a-value           : {gr['a']:.2f}")
    if "ML_unc" in df:
        print(f"median ML_unc     : {df['ML_unc'].median():.2f}")
    if "n_obs" in df:
        print(f"median #obs/event : {df['n_obs'].median():.0f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    # ---------- Gutenberg-Richter ----------
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.semilogy(gr["centers"], gr["cum"], "s", ms=4, color="tab:blue", label="cumulative N(>=M)")
    ax.semilogy(gr["centers"], np.where(gr["inc"] > 0, gr["inc"], np.nan), "o", ms=3,
                color="tab:gray", alpha=0.6, label="incremental")
    mline = np.array([gr["mc"], np.nanmax(ML)])
    ax.semilogy(mline, 10 ** (gr["a"] - gr["b"] * mline), "r-",
                label=f"GR fit: b={gr['b']:.2f}±{gr['sigma_b']:.2f}")
    ax.axvline(gr["mc"], color="k", ls="--", lw=1, label=f"Mc={gr['mc']:.2f}")
    ax.set_xlabel("ML"); ax.set_ylabel("number of events"); ax.set_title(f"Gutenberg-Richter ({args.tag})")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); f1 = os.path.join(outdir, f"{args.tag}_gutenberg_richter.png")
    fig.savefig(f1, dpi=130); print("wrote", f1)

    # ---------- QC panels ----------
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    if "n_obs" in df:
        ax[0, 0].hexbin(df["n_obs"], ML, gridsize=50, bins="log", mincnt=1)
        ax[0, 0].set_xlabel("# observations"); ax[0, 0].set_ylabel("ML")
        ax[0, 0].set_title("ML vs #obs (bias check)")
    if "ML_unc" in df:
        ax[0, 1].hexbin(ML, df["ML_unc"], gridsize=50, bins="log", mincnt=1)
        ax[0, 1].set_xlabel("ML"); ax[0, 1].set_ylabel("ML uncertainty"); ax[0, 1].set_title("ML uncertainty")
    if "evdp" in df:
        ax[0, 2].hexbin(df["evdp"], ML, gridsize=50, bins="log", mincnt=1)
        ax[0, 2].set_xlabel("depth (km)"); ax[0, 2].set_ylabel("ML"); ax[0, 2].set_title("ML vs depth")
    if df["otime"].notna().any():
        t = df.dropna(subset=["otime"]).set_index("otime")
        rate = t["ML"].resample("30D").count()
        ax[1, 0].plot(rate.index, rate.values); ax[1, 0].set_title("event count / 30 days")
        ax[1, 0].set_ylabel("N"); ax[1, 0].tick_params(axis="x", rotation=30)
    if "M_sta_std" in df:
        ax[1, 1].hexbin(ML, df["M_sta_std"], gridsize=50, bins="log", mincnt=1)
        ax[1, 1].set_xlabel("ML"); ax[1, 1].set_ylabel("station-mag scatter"); ax[1, 1].set_title("scatter vs ML")
    sc = ax[1, 2].scatter(df["evlo"], df["evla"], c=ML, s=2, cmap="viridis")
    ax[1, 2].set_xlabel("lon"); ax[1, 2].set_ylabel("lat"); ax[1, 2].set_title("ML map (scatter)")
    plt.colorbar(sc, ax=ax[1, 2], label="ML")
    fig.suptitle(f"Catalog QC ({args.tag})"); fig.tight_layout()
    f2 = os.path.join(outdir, f"{args.tag}_qc.png"); fig.savefig(f2, dpi=130); print("wrote", f2)


if __name__ == "__main__":
    main()

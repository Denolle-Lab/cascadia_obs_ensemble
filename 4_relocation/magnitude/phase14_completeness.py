#!/usr/bin/env python3
"""Per-region frequency-magnitude distributions (FMD), magnitude of completeness (Mc),
and Gutenberg-Richter b-values for the ensemble catalog, benchmarked against ANSS.

For each regional zoom we build the FMD of our local magnitudes (ML), estimate Mc by the
maximum-curvature method (with the Woessner-Wiemer +0.2 correction), fit b by Aki-Utsu
maximum likelihood, and overlay the ANSS/ComCat catalog for the same box. (Morton et
al. cannot be benchmarked on magnitude: our only Morton table is our own catalog with
match metadata, not their events/magnitudes.)

    python phase14_completeness.py
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLS = "../../data/magnitude/cascadia_catalog_classified.csv"
ANSS = "../../data/datasets_anss/anss_2010-15.csv"
MORTON = "../../data/datasets_all_regions/origin_2010_2015_reloc_cog_morton_ver3.csv"
OUT = "../../data/magnitude/completeness_by_region.png"

# [W, E, S, N] -- same boxes as the map zooms (phase5 REGIONS)
REGIONS = {
    "Mendocino": [-125.6, -123.2, 39.6, 41.4], "Gorda": [-128.8, -123.8, 40.0, 43.2],
    "Blanco": [-130.6, -126.6, 42.6, 45.3], "Endeavour": [-130.4, -127.2, 47.4, 49.8],
    "Offshore WA": [-127.6, -123.2, 46.2, 49.2], "Offshore OR": [-126.8, -123.2, 42.8, 46.4],
    "Puget Sound": [-123.6, -121.4, 46.6, 48.6],
}


def fmd(mags, dm=0.1):
    mags = mags[np.isfinite(mags)]
    if len(mags) < 20:
        return None
    edges = np.arange(np.floor(mags.min() * 10) / 10, mags.max() + dm, dm)
    centers = edges[:-1] + dm / 2
    inc, _ = np.histogram(mags, edges)
    cum = np.cumsum(inc[::-1])[::-1]                  # N(>= M)
    mc = centers[np.argmax(inc)] + 0.2               # max-curvature + WW correction
    above = mags[mags >= mc - dm / 2]
    if len(above) < 20:
        return dict(centers=centers, cum=cum, inc=inc, mc=mc, b=np.nan, n=len(above))
    b = np.log10(np.e) / (above.mean() - (mc - dm / 2))
    return dict(centers=centers, cum=cum, inc=inc, mc=mc, b=b, n=len(above))


def main():
    cls = pd.read_csv(os.path.expanduser(CLS))
    anss = pd.read_csv(os.path.expanduser(ANSS), index_col=0)
    # ML-only comparison: both catalogs are local magnitude, so filter ANSS to its
    # ml-typed events (drops Mw/Md) for a like-for-like completeness benchmark.
    if "magType" in anss.columns:
        anss = anss[anss.magType.str.lower() == "ml"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    print(f"{'region':13s} {'our N':>6} {'Mc':>5} {'b':>5} | {'ANSS N':>6} {'Mc':>5} "
          f"{'b':>5}   (Mc improvement offshore)")
    for ax, (name, box) in zip(axes.ravel(), REGIONS.items()):
        w, e, s, n = box
        ours = cls[(cls.lon.between(w, e)) & (cls.lat.between(s, n))]["ML"].to_numpy()
        an = anss[(anss.longitude.between(w, e)) & (anss.latitude.between(s, n))]["mag"].to_numpy()
        fo, fa = fmd(ours), fmd(an)

        if fo:
            ax.semilogy(fo["centers"], fo["cum"], "o", ms=3, color="firebrick",
                        label=f"ours (b={fo['b']:.2f})")
            ax.axvline(fo["mc"], color="firebrick", ls=":", lw=1)
        if fa:
            ax.semilogy(fa["centers"], fa["cum"], "s", ms=3, color="navy",
                        label=f"ANSS (b={fa['b']:.2f})")
            ax.axvline(fa["mc"], color="navy", ls=":", lw=1)
        ax.set_title(f"{name}")
        ax.set_xlabel("magnitude"); ax.set_ylabel("N ($\\geq$ M)")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim(-1, 6)

        print(f"{name:13s} {len(ours):6d} {fo['mc'] if fo else np.nan:5.1f} "
              f"{fo['b'] if fo else np.nan:5.2f} | {len(an):6d} "
              f"{fa['mc'] if fa else np.nan:5.1f} {fa['b'] if fa else np.nan:5.2f}")

    axes.ravel()[-1].axis("off")
    fig.suptitle("Frequency-magnitude distributions by region: ensemble ML vs ANSS "
                 "(ML-typed events only)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.expanduser(OUT), dpi=200, bbox_inches="tight")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

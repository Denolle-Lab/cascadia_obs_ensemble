#!/usr/bin/env python3
"""
Three-panel comparison of magnitude catalogs over the same Cascadia extent:
  (1) this study  -- OBS Route-B ML (cascadia_catalog_ML_kpos.csv)
  (2) Morton et al. 2023 -- Md      (data/ds01.csv)
  (3) USGS ComCat -- ML             (data/magnitude/comcat_ml_events.csv)
Marker size and color both scale with each catalog's own magnitude, on a shared
scale, so coverage + magnitude range are directly comparable.

Usage:
    python phase6_catalog_comparison.py --outdir ../../data/magnitude
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

REGION = dict(llcrnrlon=-130.8, urcrnrlon=-118.5, llcrnrlat=38.5, urcrnrlat=51.5)
VMIN, VMAX = 0.5, 4.5


def in_region(df, lon, lat):
    out = df[(df[lon].between(REGION["llcrnrlon"], REGION["urcrnrlon"])) &
             (df[lat].between(REGION["llcrnrlat"], REGION["urcrnrlat"]))]
    if "mag" in out.columns:                 # drop sentinel/invalid magnitudes (e.g. Morton -9)
        out = out[out["mag"] > -5]
    return out


def size(mag):
    return np.clip(1.2 * 2.6 ** (mag - 1.0), 0.3, 320)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="../../data/magnitude")
    ap.add_argument("--repo", default="../..")
    args = ap.parse_args(argv)
    outdir = os.path.expanduser(args.outdir); repo = os.path.expanduser(args.repo)

    ours = pd.read_csv(os.path.join(outdir, "cascadia_catalog_ML_kpos.csv"))
    ours = in_region(ours.rename(columns={"evlo": "lon", "evla": "lat", "ML": "mag"}), "lon", "lat")
    morton = pd.read_csv(os.path.join(repo, "data/ds01.csv"))
    morton = in_region(morton.rename(columns={"LON": "lon", "LAT": "lat", "Md": "mag"}), "lon", "lat")
    comcat = pd.read_csv(os.path.join(outdir, "comcat_ml_events.csv"))
    comcat = in_region(comcat.rename(columns={"ml": "mag"}), "lon", "lat")

    panels = [("This study\n(OBS, Route-B ML)", ours),
              ("Morton et al. 2023\n(Md)", morton),
              ("USGS ComCat\n(ML)", comcat)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 9))
    sc = None
    for ax, (name, df) in zip(axes, panels):
        m = Basemap(projection="merc", resolution="l", ax=ax, **REGION)
        m.drawmapboundary(fill_color="#f4f8fc")
        m.fillcontinents("0.85", lake_color="#f4f8fc")
        m.drawcoastlines(linewidth=0.3, color="0.4")
        m.drawparallels(range(40, 52, 2), labels=[1, 0, 0, 0], fontsize=8, linewidth=0.15)
        m.drawmeridians(range(-130, -118, 4), labels=[0, 0, 0, 1], fontsize=8, linewidth=0.15)
        d = df.dropna(subset=["lon", "lat", "mag"]).sort_values("mag")
        x, y = m(d.lon.values, d.lat.values)
        sc = ax.scatter(x, y, s=size(d.mag.values), c=d.mag.values, cmap="plasma",
                        vmin=VMIN, vmax=VMAX, alpha=0.55, linewidths=0.15, edgecolors="k")
        ax.set_title(f"{name}\nN={len(d):,}   M {d.mag.min():.1f}–{d.mag.max():.1f}", fontsize=11)

    # shared magnitude colorbar + size legend
    cb = fig.colorbar(sc, ax=axes, orientation="horizontal", fraction=0.04, pad=0.07, aspect=40)
    cb.set_label("magnitude (each catalog's own scale)")
    for mag in [1, 2, 3, 4]:
        axes[0].scatter([], [], s=size(mag), c="0.5", edgecolors="k", linewidths=0.3,
                        alpha=0.7, label=f"M {mag}")
    axes[0].legend(scatterpoints=1, labelspacing=1.2, title="size ~ magnitude",
                   loc="upper left", frameon=True, framealpha=0.9, fontsize=8)
    fig.suptitle("Cascadia magnitude catalogs — coverage and magnitude comparison", fontsize=13)
    out = os.path.join(outdir, "catalog_comparison_3panel.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print("wrote", out)
    for name, df in panels:
        d = df.dropna(subset=["mag"])
        print(f"  {name.splitlines()[0]:20s} N={len(d):6,d}  M {d.mag.min():.1f}..{d.mag.max():.1f}  median {d.mag.median():.2f}")


if __name__ == "__main__":
    main()

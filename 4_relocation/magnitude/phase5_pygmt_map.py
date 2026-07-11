#!/usr/bin/env python3
"""
PyGMT map of the Cascadia catalog: marker size scales with ML, color = depth,
over shaded bathymetry/topography.

Usage:
    python phase5_pygmt_map.py --catalog ../../data/magnitude/cascadia_catalog_ML_kpos.csv \
        --out ../../data/magnitude/cascadia_ML_map.png
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import pygmt


def ml_to_size_cm(ml):
    """Marker diameter (cm) growing with magnitude; tiny for small events."""
    return np.clip(0.015 * 2.0 ** ml, 0.01, 0.6)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", default="../../data/magnitude/cascadia_catalog_ML_kpos.csv")
    p.add_argument("--out", default="../../data/magnitude/cascadia_ML_map.png")
    args = p.parse_args(argv)

    df = pd.read_csv(os.path.expanduser(args.catalog)).dropna(subset=["evla", "evlo", "ML"])
    df = df.sort_values("ML")                       # small first -> large drawn on top
    region = [-130.8, -118.5, 38.5, 51.5]

    fig = pygmt.Figure()
    proj = "M16c"
    try:                                            # shaded relief background
        grid = pygmt.datasets.load_earth_relief(resolution="02m", region=region)
        pygmt.makecpt(cmap="geo", series=[-5000, 3000])
        fig.grdimage(grid, region=region, projection=proj, shading=True, cmap=True)
        fig.coast(region=region, projection=proj, shorelines="0.4p,black")
    except Exception as e:
        print("relief unavailable, plain basemap:", e)
        fig.coast(region=region, projection=proj, land="gray90", water="lightblue",
                  shorelines="0.4p,black")
    fig.basemap(region=region, projection=proj,
                frame=["af", "WSne+tCascadia catalog: marker size ~ ML, color = depth"])

    # events: size ~ ML, color = depth
    depth = df["evdp"].clip(0, 80)
    pygmt.makecpt(cmap="turbo", series=[0, 60], reverse=True)
    fig.plot(x=df["evlo"], y=df["evla"], size=ml_to_size_cm(df["ML"].to_numpy()),
             fill=depth, cmap=True, style="cc", pen="0.25p,gray20", transparency=25)
    fig.colorbar(position="JMR+o0.6c/0c+w8c", frame=["x+lhypocentral depth", "y+lkm"])

    # magnitude size legend (reference circles + labels, top-left offshore)
    lg_x, lg_y0 = -130.2, 51.05
    for i, m in enumerate([1, 2, 3, 4]):
        y = lg_y0 - i * 0.42
        fig.plot(x=[lg_x], y=[y], size=[ml_to_size_cm(m)], style="cc",
                 fill="white", pen="0.5p,black")
        fig.text(x=lg_x + 0.35, y=y, text=f"ML {m}", justify="LM", font="9p,black")
    fig.text(x=lg_x, y=lg_y0 + 0.4, text="magnitude", justify="LM", font="10p,Helvetica-Bold,black")

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}  ({len(df):,} events, ML {df.ML.min():.1f}..{df.ML.max():.1f})")


if __name__ == "__main__":
    main()

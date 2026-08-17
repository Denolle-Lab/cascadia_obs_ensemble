#!/usr/bin/env python3
"""
PyGMT map of the Cascadia catalog over shaded gray relief. Two encodings:

  --mode confidence  (default): single color, marker size ~ ML, and per-event
      opacity ~ number of picks (nass) -- well-constrained events (many picks)
      are solid, weakly constrained ones fade out. This foregrounds the paper's
      point: a self-consistent, scalable catalog whose confidence is legible.
  --mode depth: color = hypocentral depth (turbo), fixed opacity (the earlier map).

In confidence mode the catalog is joined to the QC origin table on orid==event_id
to attach `nass` (associated-phase count) and restrict to the final QC catalog.

Usage:
    python phase5_pygmt_map.py                         # confidence map -> default out
    python phase5_pygmt_map.py --mode depth
    python phase5_pygmt_map.py --color navy --out ../../data/magnitude/final_map.png
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import pygmt


def ml_to_size_cm(ml):
    """Marker diameter (cm) growing strongly with magnitude (exaggerated so large
    events stand out); tiny for small events."""
    return np.clip(0.018 * 3.0 ** (ml - 1.0), 0.006, 1.6)


def picks_to_transparency(n, t_opaque=2.0, t_faint=92.0):
    """Map pick count -> GMT transparency (%). Many picks -> near-opaque (confident);
    few picks -> nearly invisible. Log scale (nass is heavily right-skewed); the wide
    2..92% range gives strong confidence contrast."""
    lp = np.log10(np.maximum(np.asarray(n, float), 1.0))
    span = lp.max() - lp.min()
    norm = (lp - lp.min()) / span if span > 0 else np.ones_like(lp)
    return t_faint - (t_faint - t_opaque) * norm


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", default="../../data/magnitude/cascadia_catalog_ML_kpos.csv")
    p.add_argument("--qc-catalog",
                   default="../../data/datasets_all_regions/origin_2010_2015_reloc_cog_ver3_cc_p_4_s_4_rms_2_5.csv",
                   help="QC origin table joined on orid==event_id to attach nass")
    p.add_argument("--mode", choices=["confidence", "depth"], default="confidence")
    p.add_argument("--color", default="firebrick", help="single fill (confidence mode)")
    p.add_argument("--out", default="../../data/magnitude/cascadia_ML_map_confidence.png")
    args = p.parse_args(argv)

    df = pd.read_csv(os.path.expanduser(args.catalog)).dropna(subset=["evla", "evlo", "ML"])

    picks = None
    if args.mode == "confidence":
        qc = pd.read_csv(os.path.expanduser(args.qc_catalog))
        df = df.merge(qc[["orid", "nass"]], left_on="event_id", right_on="orid", how="inner")
        picks = df["nass"].to_numpy()
        print(f"joined QC catalog: {len(df):,} events, nass {int(picks.min())}..{int(picks.max())}")

    df = df.sort_values("ML")                       # small first -> large drawn on top
    if picks is not None:
        picks = df["nass"].to_numpy()               # keep aligned with the sort
    region = [-130.8, -118.5, 38.5, 51.5]

    fig = pygmt.Figure()
    proj = "M16c"
    try:                                            # gray, semi-transparent shaded relief
        grid = pygmt.datasets.load_earth_relief(resolution="02m", region=region)
        pygmt.makecpt(cmap="gray", series=[-6000, 4000])
        fig.grdimage(grid, region=region, projection=proj, cmap=True,
                     shading="+a315+nt1.2", transparency=45)
        fig.coast(region=region, projection=proj, shorelines="0.3p,gray40")
    except Exception as e:
        print("relief unavailable, plain basemap:", e)
        fig.coast(region=region, projection=proj, land="gray95", water="white",
                  shorelines="0.3p,gray40")

    size = ml_to_size_cm(df["ML"].to_numpy())
    # no in-figure title: the caption lives in the manuscript
    fig.basemap(region=region, projection=proj, frame=["af", "WSne"])

    if args.mode == "depth":
        depth = df["evdp"].clip(0, 80)
        pygmt.makecpt(cmap="turbo", series=[0, 60], reverse=True)
        fig.plot(x=df["evlo"], y=df["evla"], size=size, fill=depth, cmap=True,
                 style="cc", pen="0.25p,gray20", transparency=25)
        fig.colorbar(position="JMR+o0.6c/0c+w8c", frame=["x+lhypocentral depth", "y+lkm"])
    else:
        transp = picks_to_transparency(picks)
        fig.plot(x=df["evlo"], y=df["evla"], size=size, fill=args.color,
                 style="cc", pen="0.2p,gray20", transparency=transp)

    # magnitude size legend (reference circles + labels, top-left offshore)
    lg_x, lg_y0 = -130.2, 51.05
    for i, m in enumerate([1, 2, 3, 4]):
        y = lg_y0 - i * 0.42
        fig.plot(x=[lg_x], y=[y], size=[ml_to_size_cm(m)], style="cc",
                 fill="white", pen="0.5p,black")
        fig.text(x=lg_x + 0.35, y=y, text=f"ML {m}", justify="LM", font="9p,black")
    fig.text(x=lg_x, y=lg_y0 + 0.4, text="magnitude", justify="LM",
             font="10p,Helvetica-Bold,black")

    if args.mode == "confidence":
        # opacity legend: same-size markers, few vs many picks
        ox, oy0 = -127.9, 51.05
        fig.text(x=ox, y=oy0 + 0.4, text="no. picks", justify="LM",
                 font="10p,Helvetica-Bold,black")
        for i, (lab, t) in enumerate([("few", 88.0), ("many", 4.0)]):
            y = oy0 - i * 0.42
            fig.plot(x=[ox], y=[y], size=[0.32], style="cc", fill=args.color,
                     pen="0.3p,gray20", transparency=t)
            fig.text(x=ox + 0.35, y=y, text=lab, justify="LM", font="9p,black")

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}  ({len(df):,} events, ML {df.ML.min():.1f}..{df.ML.max():.1f}, mode={args.mode})")


if __name__ == "__main__":
    main()

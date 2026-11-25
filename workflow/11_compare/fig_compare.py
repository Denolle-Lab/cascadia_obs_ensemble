#!/usr/bin/env python3
"""Supplementary figure: how the catalogs differ, region by region (outputs of compare_catalogs.py).

  A  median depth (km below sea level)       B  depth minus ANSS, events with a station within max(z, 5 km):
                                               median and MAD
  C  share of events above the local surface D  depth change and E horizontal shift of the latest catalog
                                               against the reference (same v4 events)
Catalogs with locations identical to the reference (e.g. v4.0 vs v4.1) are left out.

  python workflow/11_compare/fig_compare.py [--latest ID]
-> paper_sciadv/figures/figS_catalog_compare.png/.pdf
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pygmt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "workflow" / "08_figures"))
import v4_figures as V  # noqa: E402

C = REPO / "data" / "catalog_compare"
BIG = Path("/wd1/mdenolle_data/catalog_compare")
ORDER = ["Explorer", "Endeavour", "JdF ridge/plate", "Axial", "Puget/Georgia", "Offshore forearc", "Cascades",
         "Other onshore", "Blanco", "Gorda", "Mendocino"]
STYLE = ["c", "s", "d", "t", "i", "h", "n"]
COL = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#D55E00", "#CC79A7", "#0072B2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latest", help="catalog shown in D-E (default: last registry entry)")
    a = ap.parse_args()
    reg = json.load(open(C / "registry.json"))
    ref = reg["reference"]
    S, A, P = (pd.read_csv(C / f) for f in ("summary.csv", "anss.csv", "pairs.csv"))
    same = P[(P.against == ref) & (P.dh_p90 == 0) & (P.dz_p10 == 0) & (P.dz_p90 == 0)].catalog.tolist()
    ids = [e["id"] for e in reg["catalogs"] if e["id"] not in same]
    latest = a.latest or ids[-1]
    regs = [r for r in ORDER if r in set(S.region)]
    ny = len(regs)
    y = {r: ny - i for i, r in enumerate(regs)}
    off = {c: (k - (len(ids) - 1) / 2) * 0.13 for k, c in enumerate(ids)}

    V.config()
    fig = pygmt.Figure()
    W, H = 4.6, 6.2
    yl = ";".join(f"{y[r]} {r}" for r in regs)
    ann = REPO / "data" / "catalog_compare" / "_regions.txt"
    ann.write_text("\n".join(f"{y[r]} a {r}" for r in regs) + "\n")

    def panel(tag, region, xlabel, left):
        fig.basemap(region=region, projection=f"X{W}c/{H}c",
                    frame=[("W" if left else "w") + "Sne", f"xaf+l{xlabel}", f"yc{ann}" if left else "y0"])
        fig.text(position="TL", text=tag, font="8p,Helvetica-Bold", offset="-0.35c/0.35c", no_clip=True, justify="TL")
        for r in regs:
            fig.plot(x=region[:2], y=[y[r], y[r]], pen="0.2p,gray88")

    # A: median depth with p10-p90 bars
    panel("A", [-5, 40, 0.4, ny + 0.6], "depth (km b.s.l.), p10-p50-p90", True)
    for k, c in enumerate(ids):
        s = S[(S.catalog == c) & S.region.isin(regs)]
        yy = s.region.map(y) + off[c]
        for r in s.itertuples():
            fig.plot(x=[r.depth_p10, r.depth_p90], y=[y[r.region] + off[c]] * 2, pen=f"0.5p,{COL[k]}")
        fig.plot(x=s.depth_p50, y=yy, style=f"{STYLE[k]}0.14c", fill=COL[k], pen="0.2p,white", label=f"{c}+N{len(ids)}" if k == 0 else c)
    fig.legend(position=f"x0c/{H + 0.45}c+w{3 * W + 0.7}c/0.35c+jBL", box=False)

    # B: ANSS depth difference, near events
    fig.shift_origin(xshift=f"{W + 0.35}c")
    panel("B", [-15, 15, 0.4, ny + 0.6], "depth - ANSS (km), near stations", False)
    fig.plot(x=[0, 0], y=[0.4, ny + 0.6], pen="0.4p,gray50,2_2")
    for k, c in enumerate(ids):
        g = A[(A.catalog == c) & (A.subset == "near") & A.region.isin(regs) & (A.n >= 10)]
        for r in g.itertuples():
            fig.plot(x=[r.dz_p50 - r.dz_mad, r.dz_p50 + r.dz_mad], y=[y[r.region] + off[c]] * 2, pen=f"0.5p,{COL[k]}")
        fig.plot(x=g.dz_p50.clip(-14.5, 14.5), y=g.region.map(y) + off[c], style=f"{STYLE[k]}0.14c", fill=COL[k], pen="0.2p,white")

    # C: above surface
    fig.shift_origin(xshift=f"{W + 0.35}c")
    panel("C", [0, 75, 0.4, ny + 0.6], "events above the surface (%)", False)
    for k, c in enumerate(ids):
        s = S[(S.catalog == c) & S.region.isin(regs)]
        fig.plot(x=s.pct_above_surface, y=s.region.map(y) + off[c], style=f"{STYLE[k]}0.14c", fill=COL[k], pen="0.2p,white")

    # D, E: latest vs reference
    t = pd.read_parquet(BIG / f"{latest}.parquet")
    r0 = pd.read_parquet(BIG / f"{ref}.parquet").drop_duplicates("v4_id").set_index("v4_id")
    m = t[t.located].dropna(subset=["v4_id"]).join(r0[["lat", "lon", "depth"]], on="v4_id", rsuffix="_r").dropna(subset=["lat_r"])
    m = m[np.isfinite(m.depth) & np.isfinite(m.depth_r)]
    if m.empty:
        raise SystemExit(f"fig_compare: no located events of {latest} linked to {ref}; panels D and E cannot be drawn")
    dz = (m.depth - m.depth_r).clip(-20, 20)
    from compare_catalogs import dist_km
    dh = dist_km(m.lat.to_numpy(), m.lon.to_numpy(), m.lat_r.to_numpy(), m.lon_r.to_numpy()).clip(0, 20)
    dh = dh[np.isfinite(dh)]
    hw = 2 * W + 0.35
    fig.shift_origin(xshift=f"{-2 * (W + 0.35)}c", yshift="-4.3c")
    ymz = max(1.0, 1.15 * 100 * np.histogram(dz, np.arange(-20, 20.5, 0.5))[0].max() / len(dz))
    fig.histogram(data=dz, region=[-20, 20, 0, ymz], projection=f"X{hw}c/3c", series=0.5, fill="#0072B2", pen="0.1p,white",
                  histtype=1, frame=["WSne", "xa5f1+l" + f"depth {latest} - {ref} (km; + = deeper)", "yaf+lshare (%)"])
    fig.text(position="TL", text="D", font="8p,Helvetica-Bold", offset="-0.35c/0.35c", no_clip=True, justify="TL")
    fig.text(position="TR", text=f"n = {len(m):,}; median {np.median(m.depth - m.depth_r):+.1f} km", font="6p", offset="-0.1c/-0.1c", justify="TR")
    fig.text(position="BL", text=f"<= -20: {100 * np.mean(m.depth - m.depth_r <= -20):.1f}%", font="5.5p", offset="0.05c/0.6c", justify="BL")
    fig.text(position="BR", text=f">= 20: {100 * np.mean(m.depth - m.depth_r >= 20):.1f}%", font="5.5p", offset="-0.05c/0.6c", justify="BR")
    fig.shift_origin(xshift=f"{hw + 0.9}c")
    ymh = max(1.0, 1.15 * 100 * np.histogram(dh, np.arange(0, 20.25, 0.25))[0].max() / max(len(dh), 1))
    fig.histogram(data=dh, region=[0, 20, 0, ymh], projection=f"X{W}c/3c", series=0.25, fill="#D55E00", pen="0.1p,white",
                  histtype=1, frame=["WSne", "xa5f1+lhorizontal shift (km)", "yaf"])
    fig.text(position="TL", text="E", font="8p,Helvetica-Bold", offset="-0.35c/0.35c", no_clip=True, justify="TL")
    fig.text(position="TR", text=f"median {np.median(dh):.1f} km; >5 km {100 * np.mean(dh > 5):.0f}%", font="6p", offset="-0.1c/-0.1c", justify="TR")
    out = REPO / "paper_sciadv" / "figures" / "figS_catalog_compare"
    fig.savefig(f"{out}.png", dpi=600); fig.savefig(f"{out}.pdf")
    ann.unlink()
    print("wrote", out, "catalogs:", ids, "latest:", latest)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()

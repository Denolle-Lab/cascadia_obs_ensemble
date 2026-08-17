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

import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import pygmt

SLAB = "../../data/slab2/cas_slab2_dep.xyz"
GMRT_DIR = "../../data/gmrt"     # cached GMRT grids (git-ignored)


def fetch_gmrt(region, res):
    """Download (and cache) a GMRT multi-resolution topography grid for `region`.
    res in {med, high, max} -> ~240 m / ~120 m / ~100 m per node, far finer than the
    global SRTM15/GEBCO grids, especially offshore (multibeam bathymetry)."""
    xmin, xmax, ymin, ymax = region
    d = os.path.expanduser(GMRT_DIR)
    os.makedirs(d, exist_ok=True)
    fn = f"gmrt_{xmin}_{xmax}_{ymin}_{ymax}_{res}.nc".replace(" ", "")
    p = os.path.join(d, fn)
    if not os.path.exists(p):
        q = urllib.parse.urlencode(dict(west=xmin, east=xmax, south=ymin, north=ymax,
                                        format="coards", resolution=res, layer="topo"))
        urllib.request.urlretrieve(f"https://www.gmrt.org/services/GridServer?{q}", p)
    return p


def ml_to_size_cm(ml, scale=1.0):
    """Marker diameter (cm) growing strongly with magnitude (exaggerated so large
    events stand out); tiny for small events. `scale` enlarges markers on zooms."""
    return np.clip(scale * 0.018 * 3.0 ** (ml - 1.0), 0.006 * scale, 1.8)


def load_relief(region, prefer):
    """Load gray shaded relief, degrading to 02m rather than a blank basemap. The finest
    (15s) tiles can fail to download for the larger offshore boxes, so we only prefer
    15s for tight zooms and always fall back to the reliable 02m grid."""
    for res in dict.fromkeys([prefer, "02m"]):      # single attempt each, 02m last
        try:
            return pygmt.datasets.load_earth_relief(resolution=res, region=region), res
        except Exception:
            continue
    return None, None


def picks_to_transparency(n, ref, t_opaque=2.0, t_faint=92.0):
    """Map pick count -> GMT transparency (%). Many picks -> near-opaque (confident);
    few picks -> nearly invisible. Log scale (nass is heavily right-skewed). `ref` is
    the pick array to normalize against (the FULL catalog), so the confidence scale is
    identical across the whole-margin map and the regional zoom-ins."""
    lp = np.log10(np.maximum(np.asarray(n, float), 1.0))
    lo, hi = np.log10(max(ref.min(), 1.0)), np.log10(ref.max())
    span = hi - lo
    norm = np.clip((lp - lo) / span, 0, 1) if span > 0 else np.ones_like(lp)
    return t_faint - (t_faint - t_opaque) * norm


# Regional zoom presets [W, E, S, N] -- chosen to line up with the recent (2025-2026)
# region-focused Cascadia studies for side-by-side comparison. Tune as needed.
REGIONS = {
    "full":      [-130.8, -118.5, 38.5, 51.5],   # whole margin
    "mendocino": [-125.6, -123.2, 39.6, 41.4],   # Mendocino triple junction / Gorda
    "blanco":    [-130.6, -126.6, 42.6, 45.3],   # Blanco transform + Gorda ridge
    "gorda":     [-128.8, -123.8, 40.0, 43.2],   # Gorda deformation zone
    "endeavour": [-130.4, -127.2, 47.4, 49.8],   # Endeavour/JdF ridge + Nootka fault
    "wa_margin": [-127.6, -123.2, 46.2, 49.2],   # offshore Washington forearc
    "or_margin": [-126.8, -123.2, 42.8, 46.4],   # offshore Oregon forearc
    "puget":     [-123.6, -121.4, 46.6, 48.6],   # Puget Sound (deep slab-arch seismicity)
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", default="../../data/magnitude/cascadia_catalog_ML_kpos.csv")
    p.add_argument("--qc-catalog",
                   default="../../data/datasets_all_regions/origin_2010_2015_reloc_cog_ver3_cc.csv",
                   help="origin table joined on orid==event_id to attach nass "
                        "(default: ALL relocated events, not just the QC subset)")
    p.add_argument("--mode", choices=["confidence", "depth"], default="confidence")
    p.add_argument("--color", default="firebrick", help="single fill (confidence mode)")
    p.add_argument("--region", default="full", choices=list(REGIONS),
                   help="map extent preset (full margin or a regional zoom)")
    p.add_argument("--relief-res", default="auto",
                   help="earth-relief resolution ('auto' = 15s for zooms, 02m full)")
    p.add_argument("--gmrt", default="off", choices=["off", "med", "high", "max"],
                   help="use GMRT multi-resolution topography for the relief (much finer "
                        "offshore than SRTM15/GEBCO); med~240 m, high~120 m, max~100 m")
    p.add_argument("--slab-contours", action="store_true",
                   help="overlay Slab2 interface depth contours (10 km interval)")
    p.add_argument("--size-scale", type=float, default=None,
                   help="marker size multiplier (default: auto, larger on zooms)")
    p.add_argument("--legend", dest="legend", action="store_true", default=None,
                   help="draw the size/opacity legends (default: only on the full map)")
    p.add_argument("--no-legend", dest="legend", action="store_false")
    p.add_argument("--out", default=None,
                   help="output PNG (default: cascadia_ML_map_<region>.png)")
    args = p.parse_args(argv)

    region = REGIONS[args.region]
    xmin, xmax, ymin, ymax = region
    draw_legend = (args.region == "full") if args.legend is None else args.legend
    out = args.out or f"../../data/magnitude/cascadia_ML_map_{args.region}.png"

    df = pd.read_csv(os.path.expanduser(args.catalog)).dropna(subset=["evla", "evlo", "ML"])

    picks_ref = None
    if args.mode == "confidence":
        qc = pd.read_csv(os.path.expanduser(args.qc_catalog))
        df = df.merge(qc[["orid", "nass"]], left_on="event_id", right_on="orid", how="inner")
        picks_ref = df["nass"].to_numpy().copy()     # full-catalog reference for opacity
        print(f"joined QC catalog: {len(df):,} events, nass {int(picks_ref.min())}"
              f"..{int(picks_ref.max())}")

    df = df.sort_values("ML")                        # small first -> large drawn on top
    n_in = int(((df.evlo.between(xmin, xmax)) & (df.evla.between(ymin, ymax))).sum())
    picks = df["nass"].to_numpy() if args.mode == "confidence" else None

    span = xmax - xmin
    # markers scale up on zooms so events read at the tighter extent
    sscale = args.size_scale if args.size_scale else float(np.clip(8.0 / span, 1.0, 3.0))

    fig = pygmt.Figure()
    proj = "M16c"
    # finer relief for the regional zooms (15s ~ 450 m), 02m for the full margin;
    # load_relief degrades 15s -> 30s -> 02m rather than falling back to a blank map.
    grid = None
    if args.gmrt != "off":                           # GMRT multibeam topography
        try:
            grid = fetch_gmrt(region, args.gmrt)     # cached netCDF path
            print(f"GMRT relief ({args.gmrt})")
        except Exception as e:
            print(f"GMRT failed ({str(e)[:60]}); falling back to SRTM")
    if grid is None:
        prefer = args.relief_res
        if prefer == "auto":
            # 15s where the tiles load reliably here (near-margin, west edge >= -127);
            # the far-offshore SRTM15 ocean tiles fail to download, so stay 02m there.
            prefer = "15s" if (span <= 5 and xmin >= -127) else "02m"
        grid, _res = load_relief(region, prefer)
    if grid is not None:
        pygmt.makecpt(cmap="gray", series=[-6000, 4000])
        fig.grdimage(grid, region=region, projection=proj, cmap=True,
                     shading="+a315+nt1.2", transparency=45)
        fig.coast(region=region, projection=proj, shorelines="0.3p,gray40")
    else:
        print("relief unavailable, plain basemap")
        fig.coast(region=region, projection=proj, land="gray95", water="white",
                  shorelines="0.3p,gray40")

    size = ml_to_size_cm(df["ML"].to_numpy(), sscale)
    # no in-figure title: the caption lives in the manuscript
    fig.basemap(region=region, projection=proj, frame=["af", "WSne"])

    if args.mode == "depth":
        depth = df["evdp"].clip(0, 80)
        pygmt.makecpt(cmap="turbo", series=[0, 60], reverse=True)
        fig.plot(x=df["evlo"], y=df["evla"], size=size, fill=depth, cmap=True,
                 style="cc", pen="0.25p,gray20", transparency=25)
        fig.colorbar(position="JMR+o0.6c/0c+w8c", frame=["x+lhypocentral depth", "y+lkm"])
    else:
        transp = picks_to_transparency(picks, picks_ref)
        fig.plot(x=df["evlo"], y=df["evla"], size=size, fill=args.color,
                 style="cc", pen="0.2p,gray20", transparency=transp)

    if args.slab_contours and os.path.exists(os.path.expanduser(SLAB)):
        s = pd.read_csv(os.path.expanduser(SLAB), names=["lon", "lat", "v"])
        s["lon"] = np.where(s.lon > 180, s.lon - 360, s.lon)
        s = s.dropna()
        s = s[(s.lon.between(xmin, xmax)) & (s.lat.between(ymin, ymax))]
        # Slab2 interface depth contours (km) reveal the slab curvature/strike
        fig.contour(x=s.lon, y=s.lat, z=np.abs(s.v), region=region, projection=proj,
                    levels=10, annotation="20+f7p", pen="0.6p,dodgerblue3")

    if draw_legend:
        sx, sy = xmax - xmin, ymax - ymin
        lx, ly0, step = xmin + 0.03 * sx, ymax - 0.05 * sy, 0.055 * sy
        fig.text(x=lx, y=ly0 + step, text="magnitude", justify="LM",
                 font="10p,Helvetica-Bold,black")
        for i, m in enumerate([1, 2, 3, 4]):
            y = ly0 - i * step
            fig.plot(x=[lx], y=[y], size=[ml_to_size_cm(m, sscale)], style="cc",
                     fill="white", pen="0.5p,black")
            fig.text(x=lx + 0.05 * sx, y=y, text=f"ML {m}", justify="LM", font="9p,black")
        if args.mode == "confidence":
            ox = xmin + 0.20 * sx
            fig.text(x=ox, y=ly0 + step, text="no. picks", justify="LM",
                     font="10p,Helvetica-Bold,black")
            for i, (lab, t) in enumerate([("few", 88.0), ("many", 4.0)]):
                y = ly0 - i * step
                fig.plot(x=[ox], y=[y], size=[0.32], style="cc", fill=args.color,
                         pen="0.3p,gray20", transparency=t)
                fig.text(x=ox + 0.05 * sx, y=y, text=lab, justify="LM", font="9p,black")

    outp = os.path.expanduser(out)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    fig.savefig(outp, dpi=200)
    print(f"wrote {outp}  (region={args.region}: {n_in:,} events in view, "
          f"{len(df):,} total, mode={args.mode})")


if __name__ == "__main__":
    main()

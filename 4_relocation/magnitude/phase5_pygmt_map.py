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

import map_context as mc

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

# Tectonic context per map: plate-motion arrows (moving plate, fixed plate, lat, lon, label
# justify), relative to NNR-MORVEL56; minimum Mw of the ComCat moment tensors drawn inside
# the catalog window and at other dates; number of in-window events labeled with a date;
# default slip-deficit model on the forearc zooms.
CONTEXT = {
    "full":      dict(arrows=[("JF", "NA", 48.2, -126.9), ("JF", "NA", 45.4, -126.0),
                              ("PA", "NA", 39.3, -128.3), ("PA", "NA", 45.7, -129.3)],
                      coupling="pollitz", cbar="BR", mw_in=5.0, mw_out=5.5, n_label=6,
                      km_per_mm=3.0,
                      # label anchor (lat, lon, justify) per date, placed in open space
                      labels={"2014-03-10": (41.75, -130.65, "LM"),
                              "2010-01-10": (41.95, -123.0, "LM"),
                              "2014-04-24": (51.05, -126.6, "LM"),
                              "2011-09-09": (50.6, -125.6, "LM"),
                              "2012-11-08": (47.55, -130.7, "LM"),
                              "2013-09-03": (51.3, -129.3, "LM"),
                              # largest events at other dates (gray labels)
                              "2005-06-15": (42.6, -130.65, "LM"),
                              "2024-12-05": (38.85, -127.2, "LM"),
                              "2001-02-28": (46.35, -121.15, "LM"),
                              "2018-10-22": (47.15, -130.7, "LM")}),
    "mendocino": dict(arrows=[("PA", "NA", 39.85, -125.1)], label_side="L", label_top=41.3,
                      mw_in=4.5, mw_out=5.5, n_label=3, n_label_other=2, km_per_mm=1.2),
    "blanco":    dict(arrows=[("PA", "NA", 43.0, -130.1), ("JF", "NA", 44.75, -128.3)],
                      label_side="R", label_top=44.4,
                      mw_in=4.5, mw_out=5.5, n_label=3, n_label_other=1, km_per_mm=1.5),
    "gorda":     dict(arrows=[("PA", "NA", 40.35, -128.3)], label_side="L", label_top=42.9,
                      mw_in=4.5, mw_out=5.5, n_label=3, n_label_other=2, km_per_mm=1.5),
    "endeavour": dict(arrows=[("PA", "NA", 47.8, -130.1), ("JF", "NA", 47.8, -128.3)],
                      label_side="L", label_top=49.7,
                      mw_in=4.5, mw_out=5.5, n_label=3, n_label_other=1, km_per_mm=1.5),
    "wa_margin": dict(arrows=[("JF", "NA", 46.75, -127.1)], coupling="pollitz", cbar="BL",
                      label_side="R", label_top=49.0,
                      mw_in=4.0, mw_out=4.5, n_label=3, km_per_mm=1.5),
    "or_margin": dict(arrows=[("JF", "NA", 44.85, -126.05)], coupling="pollitz", cbar="TL",
                      label_side="L", label_top=44.2,
                      mw_in=4.0, mw_out=4.5, n_label=3, km_per_mm=1.5),
    "puget":     dict(arrows=[], mw_in=4.0, mw_out=4.5, n_label=3, n_label_other=2,
                      label_side="L", label_top=46.82, km_per_mm=1.0),
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", default="../../data/magnitude/cascadia_catalog_ML_routeA.csv")
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
    p.add_argument("--coupling", default="auto",
                   choices=["auto", "none"] + list(mc.COUPLING),
                   help="slip-deficit model drawn under the events (auto: Pollitz 2025 "
                        "on the WA/OR forearc zooms, none elsewhere)")
    p.add_argument("--layer", choices=["catalog", "context"], default="catalog",
                   help="full margin only: 'catalog' = the catalog map (plate boundaries "
                        "only); 'context' = faint seismicity + slip deficit, moment tensors "
                        "and plate motions (the companion tectonic panel)")
    p.add_argument("--no-context", action="store_true",
                   help="omit plate boundaries, plate motions and moment tensors")
    p.add_argument("--legend", dest="legend", action="store_true", default=None,
                   help="draw the size/opacity legends (default: only on the full map)")
    p.add_argument("--no-legend", dest="legend", action="store_false")
    p.add_argument("--legend-pos", default="TR", help="legend corner (GMT justify: TL, TR, BL, BR)")
    p.add_argument("--width-cm", type=float, default=None,
                   help="map width AT PRINT SIZE (cm); fonts/markers are set for it. Default: "
                        "8.4 (full margin, one Seismica column), 5.7 (regional zooms, a third "
                        "of the text width), 7.2 (puget, 0.85 column)")
    p.add_argument("--out", default=None,
                   help="output PNG (default: cascadia_ML_map_<region>.png)")
    args = p.parse_args(argv)

    region = REGIONS[args.region]
    xmin, xmax, ymin, ymax = region
    draw_legend = (args.region == "full") if args.legend is None else args.legend
    ctx_panel = args.layer == "context"
    stem = "cascadia_context_map" if ctx_panel else "cascadia_ML_map"
    out = args.out or f"../../data/magnitude/{stem}_{args.region}.png"
    # on the full-margin catalog map the context layers would hide the seismicity; they
    # go on the companion panel (--layer context). The zooms carry both.
    full_catalog = args.region == "full" and not ctx_panel

    df = pd.read_csv(os.path.expanduser(args.catalog)).dropna(subset=["evla", "evlo", "ML"])

    picks_ref = None
    if args.mode == "confidence":
        qc = pd.read_csv(os.path.expanduser(args.qc_catalog))
        df = df.merge(qc[["orid", "nass"]], left_on="event_id", right_on="orid", how="inner")
        picks_ref = df["nass"].to_numpy().copy()     # full-catalog reference for opacity
        print(f"joined QC catalog: {len(df):,} events, nass {int(picks_ref.min())}"
              f"..{int(picks_ref.max())}")

    df = df.sort_values("ML", ascending=False)       # large first -> small drawn on top, not hidden
    n_in = int(((df.evlo.between(xmin, xmax)) & (df.evla.between(ymin, ymax))).sum())
    picks = df["nass"].to_numpy() if args.mode == "confidence" else None

    span = xmax - xmin
    # markers scale up on zooms so events read at the tighter extent
    width = args.width_cm or {"full": 8.4, "puget": 7.2}.get(args.region, 5.7)
    # marker sizes were tuned on a 16 cm map; scale them to the printed width. All
    # regional zooms share ONE size scale so their panels read on the same encoding.
    zoom = 1.0 if args.region == "full" else 2.2
    sscale = (args.size_scale or zoom) * width / 16.0

    fig = pygmt.Figure()
    # fonts at print size (the figure is placed at 100% in the manuscript)
    pygmt.config(FONT_ANNOT_PRIMARY="6.5p,Helvetica,black", FONT_LABEL="7p,Helvetica,black",
                 MAP_FRAME_TYPE="fancy", MAP_FRAME_WIDTH="1.5p", MAP_TICK_LENGTH_PRIMARY="2p",
                 MAP_ANNOT_OFFSET_PRIMARY="1.5p",
                 MAP_FRAME_PEN="0.5p,black", FONT_TITLE="7p,Helvetica,black")
    proj = f"M{width}c"
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

    ctx = CONTEXT[args.region]
    coupling = ctx.get("coupling") if args.coupling == "auto" else \
        (None if args.coupling == "none" else args.coupling)
    if full_catalog:
        coupling = None
    if coupling:                                     # interseismic slip-deficit rate
        sd, sd_cite = mc.load_slip_deficit(coupling, region)
        pygmt.makecpt(cmap="lapaz", series=[0, 40, 5], reverse=True)
        fig.grdimage(sd, region=region, projection=proj, cmap=True, nan_transparent=True,
                     transparency=30)
        fig.grdcontour(sd, levels=10, pen="0.3p,0/50/100", region=region, projection=proj)
        print(f"slip deficit: {sd_cite}")
    if not args.no_context:
        mc.draw_boundaries(fig, region, pen_scale=0.8 if args.region == "full" else 1.0)

    size = ml_to_size_cm(df["ML"].to_numpy(), sscale)
    # no in-figure title: the caption lives in the manuscript
    fig.basemap(region=region, projection=proj, frame=["af", "WSne"])

    if ctx_panel:                                    # seismicity as a faint backdrop
        fig.plot(x=df["evlo"], y=df["evla"], style="c0.03c", fill=args.color,
                 transparency=70)
    elif args.mode == "depth":
        depth = df["evdp"].clip(0, 80)
        pygmt.makecpt(cmap="turbo", series=[0, 60], reverse=True)
        fig.plot(x=df["evlo"], y=df["evla"], size=size, fill=depth, cmap=True,
                 style="cc", pen="0.25p,gray20", transparency=25)
        fig.colorbar(position="JMR+o0.6c/0c+w8c", frame=["x+lhypocentral depth", "y+lkm"])
    else:
        transp = picks_to_transparency(picks, picks_ref)
        fig.plot(x=df["evlo"], y=df["evla"], size=size, fill=args.color,
                 style="cc", pen="0.2p,gray20", transparency=transp)

    mt = None if (args.no_context or full_catalog) else mc.load_mt(region, df)
    top = pd.DataFrame()
    if mt is not None:
        # moment tensors at other dates or not in our catalog (context, gray) under those
        # matched to our events (phase18); balls scale with Mw (size at Mw 5 = bscale)
        bscale = 0.24 if args.region == "full" else 0.26
        other = mt[~mt.matched & (mt.Mw >= ctx["mw_out"])]
        inwin = mt[mt.matched & (mt.Mw >= ctx["mw_in"])]
        mc.draw_mechanisms(fig, other, bscale, fill="gray55", pen="0.2p,gray30")
        mc.draw_mechanisms(fig, inwin, bscale, fill=args.color)
        if "labels" in ctx:          # hand-placed labels: exactly the events listed there
            keys = set(ctx["labels"])
            def pick(d):                # largest event of each listed date (d is Mw-sorted)
                day = d.time.dt.strftime("%Y-%m-%d")
                return d[day.isin(keys) & ~day.duplicated()]
            top = pd.concat([pick(inwin), pick(other)])
        else:
            top = pd.concat([inwin.head(ctx["n_label"]),
                             other.head(ctx.get("n_label_other", 0))])
        for _, r in top.iterrows():
            print(f"  labeled {r.time:%Y-%m-%d} Mw {r.Mw:.1f} ({r.mt_source}) "
                  f"our ML {r.ML_ours:.2f}  {r.place}")
        print(f"moment tensors: {len(inwin)} matched to our events (Mw>={ctx['mw_in']}), "
              f"{len(other)} other (Mw>={ctx['mw_out']})")
    if not args.no_context and not full_catalog and ctx["arrows"]:
        mc.draw_plate_motion(fig, ctx["arrows"], km_per_mm=ctx["km_per_mm"])
    spots = dict(ctx.get("labels", {}))
    if not top.empty and not spots:
        # stack the labels in a column along one map edge (0.38 cm apart, from label_top
        # down, in event-latitude order), each tied to its event by a leader line
        deg_per_cm = (xmax - xmin) / width                      # longitude per cm
        side = ctx.get("label_side", "R")
        lo = xmin + 0.12 * deg_per_cm if side == "L" else xmax - 0.12 * deg_per_cm
        just = "LM" if side == "L" else "RM"
        la = ctx.get("label_top", ymax)
        for _, r in top.sort_values("lat", ascending=False).iterrows():
            spots[f"{r.time:%Y-%m-%d}"] = (la, lo, just)
            la -= 0.38 * deg_per_cm * np.cos(np.radians(la))   # Mercator: dlat ~ dlon cos
    for _, r in top.iterrows():                      # date labels on top of everything
        key = f"{r.time:%Y-%m-%d}"
        la, lo, just = spots[key]
        fig.plot(x=[r.lon, lo], y=[r.lat, la], pen="0.35p,gray15")
        ink = "black" if r.matched else "gray30"        # gray = not in our catalog
        fig.text(x=lo, y=la, text=f"{key} M@-w@-{r.Mw:.1f}", justify=just,
                 font=f"5.5p,Helvetica,{ink}", fill="white@15", clearance="0.03c/0.02c")
    if coupling:                                     # colorbar inside the map
        with pygmt.config(FONT_ANNOT_PRIMARY="6.5p,Helvetica,black",
                          FONT_LABEL="6.5p,Helvetica,black", MAP_FRAME_PEN="0.3p,black",
                          MAP_TICK_LENGTH_PRIMARY="1.5p"):
            fig.colorbar(position=f"j{ctx.get('cbar', 'BL')}+w{min(0.38 * width, 2.6):.2f}c"
                                  f"/0.13c+h+o0.25c/0.55c",
                         box="+gwhite@15+p0.3p,gray50+c0.12c/0.1c",
                         frame=["xa10f5+lslip-deficit rate (mm/yr)"])

    if args.slab_contours and os.path.exists(os.path.expanduser(SLAB)):
        s = pd.read_csv(os.path.expanduser(SLAB), names=["lon", "lat", "v"])
        s["lon"] = np.where(s.lon > 180, s.lon - 360, s.lon)
        s = s.dropna()
        s = s[(s.lon.between(xmin, xmax)) & (s.lat.between(ymin, ymax))]
        # Slab2 interface depth contours (km) reveal the slab curvature/strike
        fig.contour(x=s.lon, y=s.lat, z=np.abs(s.v), region=region, projection=proj,
                    levels=10, annotation="20+f5.5p", pen="0.4p,0/114/178")

    if draw_legend:
        # legend drawn in an inset with centimetre coordinates: one row of magnitude
        # circles at their plotted size, labels underneath; then pick-count opacity
        mags = [] if ctx_panel else [2, 3, 4] + ([5] if args.region == "full" else [])
        sz = [float(ml_to_size_cm(m, sscale)) for m in mags] or [-0.5]
        cw = max(max(sz) + 0.1, 0.62)                 # column width (cm)
        conf = args.mode == "confidence" and not ctx_panel
        W = max(cw * len(mags) + 0.2, 3.6 if mt is not None else 2.75)  # widest header
        mt_key = mt is not None
        H = (0.35 + max(sz) + 0.3 + (1.0 if conf else 0.0) if mags else 0.0) \
            + (0.75 if mt_key else 0.0)
        with fig.inset(position=f"j{args.legend_pos}+w{W:.2f}c/{H:.2f}c+o0.1c",
                       box="+gwhite@15+p0.3p,gray50+c0.08c"):
            fig.basemap(region=[0, W, 0, H], projection=f"X{W:.2f}c/{H:.2f}c", frame="+n")
            y = H - 0.2
            if mags:
                fig.text(x=0.1, y=y, text="Magnitude", justify="LM", font="7p,Helvetica-Bold")
            yc = y - 0.2 - max(sz) / 2
            for i, (m, d) in enumerate(zip(mags, sz)):
                xc = 0.1 + cw * (i + 0.5)
                fig.plot(x=[xc], y=[yc], style=f"c{d:.3f}c", fill="white", pen="0.4p,black")
                fig.text(x=xc, y=yc - max(sz) / 2 - 0.13, text=f"M@-L@- {m}",
                         justify="CM", font="6.5p,Helvetica")
            if conf:
                ref = picks_ref if picks_ref is not None else picks
                lo, hi = int(np.min(ref)), int(np.max(ref))
                y2 = yc - max(sz) / 2 - 0.55
                fig.text(x=0.1, y=y2, text="Associated picks", justify="LM",
                         font="7p,Helvetica-Bold")
                for i, n in enumerate([lo, 30, hi]):
                    t = float(picks_to_transparency(np.array([n]), ref)[0])
                    xc = 0.25 + i * (W - 0.3) / 3
                    fig.plot(x=[xc], y=[y2 - 0.3], style="c0.2c", fill=args.color,
                             pen="0.3p,gray20", transparency=t)
                    fig.text(x=xc + 0.17, y=y2 - 0.3, text=str(n), justify="LM",
                             font="6.5p,Helvetica")
            if mt_key:                                # ComCat moment tensors
                y3 = H - 0.2 if not mags else yc - max(sz) / 2 - 0.55 - (1.0 if conf else 0.0)
                fig.text(x=0.1, y=y3, text="Moment tensors (ComCat)", justify="LM",
                         font="7p,Helvetica-Bold")
                ss = dict(mrr=[0.0], mtt=[-1.0], mff=[1.0], mrt=[0.0], mrf=[0.0], mtf=[0.0],
                          exponent=[24])                          # a strike-slip ball
                for i, (fc, lab) in enumerate([(args.color, "in this catalog"),
                                               ("gray55", "other")]):
                    xc = 0.25 + i * 1.8
                    fig.meca(spec=ss, convention="mt", longitude=[xc], latitude=[y3 - 0.33],
                             depth=[10], scale="0.26c+m", compressionfill=fc,
                             extensionfill="white", pen="0.25p,gray10")
                    fig.text(x=xc + 0.2, y=y3 - 0.33, text=lab, justify="LM",
                             font="6.5p,Helvetica")

    outp = os.path.expanduser(out)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    fig.savefig(outp, dpi=600)
    print(f"wrote {outp}  (region={args.region}: {n_in:,} events in view, "
          f"{len(df):,} total, mode={args.mode})")


if __name__ == "__main__":
    main()

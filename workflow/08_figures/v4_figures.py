#!/usr/bin/env python3
"""Paper figures from the catalog v4 layers (the same inputs as the 3D atlas), drawn with PyGMT.

Vector PDF + 600 dpi PNG at print size (Science Advances: 18 cm full width, 12 cm two thirds).

    python workflow/08_figures/v4_figures.py overview              # Fig. 1: setting, catalog, stations, depth section, array in time
    python workflow/08_figures/v4_figures.py faults                # Fig. 3: fault surfaces v0, map + two 3D views
    python workflow/08_figures/v4_figures.py zoom blanco --color fault     # any region of REGIONS, colour by fault/depth/time
    python workflow/08_figures/v4_figures.py perspective mendocino --az 210 --el 28
    python workflow/08_figures/v4_figures.py all
    python workflow/08_figures/v4_figures.py --mag stored --clusters <dir> --out <dir> clusters   # options before the figure

Layers: catalog v4 (data/catalog_v4), Route A magnitudes (converged, D-B091; --mag stored for data/magnitude_v4;
workflow/05_magnitude/mag_paths.py), fault surfaces v0 (data/faults/v0),
CRESCENT CFM traces (data/faults/crescent_cfm), Slab2 (data/slab2), PB2002 plate boundaries and ComCat moment
tensors (map_context.py), GMRT grids (data/gmrt, cached) or GMT earth_relief, the ver5/ver6 station table.
Mc maps for Fig. 2 come from workflow/06_analysis/v4_mc_maps.py (same style).

Run from the repo root with the default env:
    LD_LIBRARY_PATH=.pixi/envs/default/lib .pixi/envs/default/bin/python workflow/08_figures/v4_figures.py all
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pygmt
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data"
OUT = ROOT / "paper_sciadv" / "figures"
STATIONS = Path("/wd1/mdenolle_data/picks_share/stations/stations_ver5_v6.csv")
sys.path.insert(0, str(ROOT / "workflow" / "06_analysis"))
sys.path.append(str(ROOT / "workflow" / "05_magnitude"))
import mag_paths  # noqa: E402  (converged Route A magnitudes by default, D-B091)
MAG = mag_paths.v41()        # --mag / $CASCADIA_MAG_V41: converged (default), stored or a path
_cwd = os.getcwd(); os.chdir(ROOT / "workflow" / "06_analysis")     # map_context uses paths relative to its folder
import map_context as mctx  # noqa: E402
mctx.PB2002, mctx.FOCAL, mctx.FOCAL_MATCHED = (os.path.abspath(p) for p in (mctx.PB2002, mctx.FOCAL, mctx.FOCAL_MATCHED))
os.chdir(_cwd)

FULL, TWO3 = 18.0, 12.0                      # cm
MARGIN = [-131.0, -119.5, 39.0, 51.0]
REGIONS = {
    "endeavour": [-130.4, -127.2, 47.4, 49.8], "wa_margin": [-127.6, -123.2, 46.2, 49.2],
    "puget": [-123.6, -121.4, 46.6, 48.6], "axial": [-130.35, -129.75, 45.75, 46.2],
    "or_margin": [-126.8, -123.2, 42.8, 46.4], "blanco": [-130.6, -126.6, 42.6, 45.3],
    "blanco_central": [-129.3, -127.3, 43.3, 44.6], "gorda": [-128.8, -123.8, 40.0, 43.2],
    "mendocino": [-125.6, -123.2, 39.6, 41.4], "blanco_strip": [-130.7, -126.6, 42.75, 44.75],
}
# default 3D views: azimuth of the viewpoint, elevation, depth range (km), vertical exaggeration
VIEWS = {"blanco": (200, 30, 30, 2.0), "blanco_central": (200, 30, 34, 2.0), "mendocino": (210, 26, 40, 1.5),
         "gorda": (200, 30, 35, 2.0), "puget": (200, 28, 70, 2.0), "axial": (200, 35, 5, 4.0),
         "endeavour": (200, 30, 20, 4.0), "wa_margin": (200, 28, 60, 2.0), "or_margin": (200, 28, 60, 2.0)}
# qualitative colours for faults (Okabe-Ito + Tol muted), cycled
FCOL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#882255", "#117733", "#AA4499", "#44AA99",
        "#999933", "#332288"]
CFM_PEN, PLATE_PEN = "0.35p,#b8860b", "0.6p,gray15"


def config():
    pygmt.config(FONT_ANNOT_PRIMARY="6p,Helvetica,black", FONT_ANNOT_SECONDARY="6.5p,Helvetica,black", FONT_LABEL="6.5p,Helvetica,black", FONT_TAG="8p,Helvetica-Bold,black",
                 FONT_TITLE="7p,Helvetica-Bold,black", MAP_FRAME_TYPE="plain", MAP_FRAME_PEN="0.5p,gray20",
                 MAP_TICK_LENGTH_PRIMARY="2p", MAP_TICK_PEN_PRIMARY="0.4p,gray20", MAP_ANNOT_OFFSET_PRIMARY="1.5p",
                 MAP_LABEL_OFFSET="3p", MAP_GRID_PEN_PRIMARY="0.2p,gray80", FORMAT_GEO_MAP="dddF", PS_CHAR_ENCODING="ISOLatin1+",
                 COLOR_NAN="white")


# ------------------------------------------------------------------ layers
def events():
    c = pd.read_csv(D / "catalog_v4" / "Cascadia_relocated_catalog_v4.csv", skipinitialspace=True)
    c.columns = [x.strip() for x in c.columns]
    c = c.rename(columns={"ID": "id", "longitude": "lon", "latitude": "lat", "depth(km)": "dep"})
    c["time"] = pd.to_datetime(c["datetime"].str.strip(), utc=True)
    m = pd.read_csv(MAG, usecols=["event_id", "M"]).set_index("event_id").M
    c["M"] = c.id.map(m)
    c["qc"] = (c.Num_p_picks >= 4) & (c.Num_s_picks >= 4)
    mem = pd.read_csv(D / "faults" / "v0" / "fault_surfaces_v0_members.csv").set_index("event_id").surface_id
    c["fault"] = c.id.map(mem)
    return c


def msize(M, s=1.0):
    """Symbol diameter (cm): small events stay dots, large ones stand out."""
    M = np.nan_to_num(np.asarray(M, float), nan=0.5)
    return np.clip(s * 0.012 * 2.2 ** (M - 1.0), 0.008 * s, 0.6 * s)


def fault_surfaces():
    """Fault surfaces v0: per surface the 1 km vertex grid (rows x cols x [lon, lat, depth]) and properties."""
    g = json.load(open(D / "faults" / "v0" / "fault_surfaces_v0.geojson"))["features"]
    vx = pd.read_csv(D / "faults" / "v0" / "fault_surfaces_v0_vertices.csv")
    out = []
    for i, f in enumerate(g):
        p = f["properties"]
        v = vx[vx.surface_id == p["surface_id"]].sort_values("vertex")
        x, y, z = f["geometry"]["coordinates"][0][0][2]           # third vertex of the first triangle = grid (1, 1)
        d2 = ((v.lon - x) * 111.195 * np.cos(np.radians(y))) ** 2 + ((v.lat - y) * 111.195) ** 2 + (v.depth_km + z / 1000) ** 2
        nu = int(np.argmin(d2.to_numpy())) - 1
        G = v[["lon", "lat", "depth_km"]].to_numpy().reshape(len(v) // nu, nu, 3)
        S = v["sigma_total_km"].to_numpy().reshape(len(v) // nu, nu)
        corners = np.array([G[0, 0], G[0, -1], G[-1, -1], G[-1, 0]])
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        top = min(edges, key=lambda e: corners[list(e), 2].mean())
        out.append(dict(id=p["surface_id"], k=i, color=FCOL[i % len(FCOL)], G=G, S=S, corners=corners,
                        top=corners[list(top)], dep=float(p["depth_km"]), sig=float(p["sigma_total_median_km"]),
                        n=int(p["n_events"]), dip=float(p["dip"]), lit=p.get("lit_class") or ""))
    return out


def relief(region, max_nodes=900):
    """Smallest cached GMRT grid that covers the region, else GMT earth_relief; subsampled to <= max_nodes per side."""
    W, E, S, N = region
    best = None
    for f in glob.glob(str(D / "gmrt" / "gmrt_*.nc")):
        w, e, s, n = map(float, Path(f).stem.split("_")[1:5])
        if w <= W + 0.02 and e >= E - 0.02 and s <= S + 0.02 and n >= N - 0.02 and (best is None or (e - w) * (n - s) < best[0]):
            best = ((e - w) * (n - s), f)
    if best:
        ds = xr.open_dataset(best[1])
        z = ds[[v for v in ds.data_vars][0]].sel(lon=slice(W, E))
        z = z.sel(lat=slice(S, N)) if z.lat[0] < z.lat[-1] else z.sel(lat=slice(N, S)).isel(lat=slice(None, None, -1))
    else:
        span = max(E - W, N - S)
        res = "15s" if span < 2.5 else "30s" if span < 6 else "01m"
        try:
            z = pygmt.datasets.load_earth_relief(resolution=res, region=region, registration="gridline")
        except Exception:
            z = pygmt.datasets.load_earth_relief(resolution="01m", region=region, registration="gridline")
    sx, sy = max(1, int(np.ceil(z.sizes["lon"] / max_nodes))), max(1, int(np.ceil(z.sizes["lat"] / max_nodes)))
    if sx > 1 or sy > 1:
        z = z.coarsen(lon=sx, lat=sy, boundary="trim").mean()
    return z.astype("float32").rename("z")


def relief_cpt(tmp, scale=1.0):
    """Muted relief: pale blue-grey sea (darker with depth), warm grey land. scale=0.001 for km grids."""
    stops = [(-7000, "137/160/176"), (-4000, "165/186/199"), (-2000, "193/208/217"), (-200, "219/228/233"), (0, "236/241/243")]
    land = [(0, "236/233/226"), (500, "222/218/208"), (1500, "200/195/184"), (4500, "172/166/154")]
    rows = [f"{a[0] * scale:g} {a[1]} {b[0] * scale:g} {b[1]}" for seq in (stops, land) for a, b in zip(seq[:-1], seq[1:])]
    p = Path(tmp) / f"relief_{scale:g}.cpt"
    p.write_text("\n".join(rows) + "\nB 137/160/176\nF 172/166/154\nN white\n")
    return str(p)


def hillshade(z, amp=0.22):
    """Illumination for grdimage/grdview: gradient toward 315 deg, centred and squashed to +/- amp
    (GMT's own -N normalisation came back unscaled with PyGMT 0.13, which made the relief far too dark)."""
    g = pygmt.grdgradient(z, azimuth=315)
    v = g.values
    med, sd = np.nanmedian(v), np.nanstd(v)
    g.values = (amp * np.tanh((v - med) / (2 * sd if sd > 0 else 1))).astype("float32")
    return g


def basemap(fig, region, proj, tmp, frame, z=None, transparency=0):
    z = relief(region) if z is None else z
    shade = hillshade(z)
    fig.grdimage(z, region=region, projection=proj, cmap=relief_cpt(tmp), shading=shade, frame=frame, transparency=transparency)
    fig.coast(region=region, projection=proj, shorelines="0.25p,gray35", borders="2/0.25p,gray55", resolution="h")
    return z


def cfm_traces(fig, region):
    f = D / "faults" / "crescent_cfm" / "crescent_cfm_crustal_traces.geojson"
    if not f.exists():
        return
    W, E, S, N = region
    segs = []
    for ft in json.load(open(f))["features"]:
        xy = np.array(ft["geometry"]["coordinates"])[:, :2]
        if ((xy[:, 0] >= W) & (xy[:, 0] <= E) & (xy[:, 1] >= S) & (xy[:, 1] <= N)).any():
            segs.append(xy)
    if segs:
        fig.plot(data=_multi(segs), pen=CFM_PEN)


def _multi(segs, close=False):
    """Multi-segment table for GMT (one file, '>' between segments)."""
    fd, p = tempfile.mkstemp(suffix=".txt"); os.close(fd)
    with open(p, "w") as fh:
        for s in segs:
            fh.write(">\n")
            np.savetxt(fh, np.vstack([s, s[:1]]) if close else s, fmt="%.5f")
    return p


def plate_boundaries(fig, region, scale=1.0):
    pb = mctx.plate_boundaries(region)
    for L in pb["OSR"]:
        fig.plot(x=L[:, 0], y=L[:, 1], pen=f"{1.0 * scale:.2f}p,gray15")
    for L in pb["TF"]:
        fig.plot(x=L[:, 0], y=L[:, 1], pen=f"{0.6 * scale:.2f}p,gray15")
    for L in pb["SUB"]:
        L = L[np.argsort(-L[:, 1])]
        fig.plot(x=L[:, 0], y=L[:, 1], pen=f"{0.6 * scale:.2f}p,gray15", style=f"f{0.28 * scale:.2f}c/{0.07 * scale:.2f}c+l+t",
                 fill="gray15")


def stations():
    s = pd.read_csv(STATIONS)
    for c in ("epoch_start", "epoch_end"):
        s[c] = pd.to_datetime(s[c], errors="coerce", utc=True).dt.tz_localize(None)
    s["epoch_end"] = s.epoch_end.fillna(pd.Timestamp("2030-01-01"))
    s["sea"] = s.seafloor.astype(str).str.lower().eq("true")
    return s


def tag(fig, letter, offset="0.1c/-0.1c"):
    fig.text(position="TL", text=letter, font="8p,Helvetica-Bold,black", justify="TL", offset=offset, no_clip=True,
             fill="white@20", clearance="0.04c/0.03c")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=600)
    print(f"wrote {OUT / name}.pdf/.png")


# ------------------------------------------------------------------ Fig. 1
def fig_overview(a):
    ev = events()
    st = stations()
    tmp = tempfile.mkdtemp()
    config()
    fig = pygmt.Figure()
    W, GAP = 8.3, 1.35
    proj = f"M{W}c"
    q = ev[ev.qc].sort_values("M", ascending=True, na_position="first")      # large last, on top
    # (a) catalog by depth
    basemap(fig, MARGIN, proj, tmp, frame=["WSne", "xa4f1", "ya2f1"])
    cfm_traces(fig, MARGIN)
    plate_boundaries(fig, MARGIN, 0.8)
    pygmt.makecpt(cmap="lajolla", series=[0, 50], output=f"{tmp}/dep.cpt", reverse=True, background=True)
    fig.plot(x=q.lon, y=q.lat, size=msize(q.M, 0.9), style="c", fill=q.dep.clip(0, 50), cmap=f"{tmp}/dep.cpt",
             pen="0.05p,gray30@60", transparency=10)
    for k, b in REGIONS.items():
        if k in ("blanco", "mendocino", "endeavour", "puget", "axial", "gorda"):
            fig.plot(x=[b[0], b[1], b[1], b[0], b[0]], y=[b[2], b[2], b[3], b[3], b[2]], pen="0.5p,black,4_2")
    fig.colorbar(cmap=f"{tmp}/dep.cpt", position="jBL+w2.6c/0.14c+h+o0.3c/0.35c", box="+gwhite@15+p0.3p,gray50+c0.1c/0.08c",
                 frame=["xa10f5+lhypocentre depth (km)"])
    _legend_mag(fig, MARGIN, proj, 0.9)
    fig.basemap(region=MARGIN, projection=proj, map_scale="jTR+w100k+o0.3c/0.3c+f+l", frame=["WSne", "xa4f1", "ya2f1"])
    tag(fig, "A")
    # (c) depth vs longitude, aligned under (a)
    fig.shift_origin(yshift="-3.4c")
    H = 2.9
    fig.basemap(region=[MARGIN[0], MARGIN[1], 0, 60], projection=f"X{W}c/-{H}c",
                frame=["WSne", "xa4f1+llongitude (@.E)", "ya20f10+ldepth (km)"])
    fig.plot(x=q.lon, y=q.dep.clip(-5, 60), style="c0.025c", fill=q.dep.clip(0, 50), cmap=f"{tmp}/dep.cpt", transparency=40)
    s = pd.read_csv(D / "slab2" / "cas_slab2_dep.xyz", header=None, names=["lon", "lat", "z"]).dropna()
    s["lon"] = np.where(s.lon > 180, s.lon - 360, s.lon)
    for lat0, pen in ((47.5, "0.8p,#0b4f8a"), (44.0, "0.8p,#0b4f8a,4_2"), (41.0, "0.8p,#0b4f8a,1_1")):
        r = s[np.isclose(s.lat, lat0, atol=0.026)].groupby("lon").z.mean()
        fig.plot(x=r.index, y=-r.values, pen=pen)
        i45 = int(np.argmin(np.abs(-r.values - 45)))
        fig.text(x=r.index[i45] + 0.12, y=45, text=f"{lat0:g}@.N", font="5p,Helvetica,#0b4f8a", justify="LM", fill="white@30")
    fig.text(x=-121.6, y=56, text="Slab2", font="5.5p,Helvetica-Bold,#0b4f8a", justify="LM")
    tag(fig, "C")
    # (b) stations by first year of deployment
    fig.shift_origin(xshift=f"{W + GAP}c", yshift="3.4c")
    basemap(fig, MARGIN, proj, tmp, frame=["wSnE", "xa4f1", "ya2f1"])
    plate_boundaries(fig, MARGIN, 0.8)
    first = st.groupby(["network", "station"]).agg(lon=("lon", "first"), lat=("lat", "first"), sea=("sea", "first"),
                                                  y0=("epoch_start", "min")).reset_index()
    first["y0"] = first.y0.dt.year.clip(2010, 2016)
    pygmt.makecpt(cmap="batlow", series=[2009.5, 2016.5, 1], output=f"{tmp}/yr.cpt", continuous=False)
    land = first[~first.sea]
    fig.plot(x=land.lon, y=land.lat, style="i0.14c", fill="white", pen="0.3p,gray20")
    sea = first[first.sea].sort_values("y0")
    fig.plot(x=sea.lon, y=sea.lat, style="i0.17c", fill=sea.y0, cmap=f"{tmp}/yr.cpt", pen="0.25p,gray10")
    fig.colorbar(cmap=f"{tmp}/yr.cpt", position="jBL+w3.2c/0.14c+h+o0.3c/0.35c", box="+gwhite@15+p0.3p,gray50+c0.1c/0.08c",
                 frame=["xa1+lfirst year of the ocean-bottom station"])
    fig.text(position="TR", offset="-0.25c/-0.25c", justify="TR", font="6p,Helvetica,black", fill="white@15", clearance="0.06c/0.05c",
             text=f"{first.sea.sum()} ocean-bottom, {(~first.sea).sum()} land stations")
    tag(fig, "B")
    # (d) array and catalog in time
    fig.shift_origin(yshift="-3.4c")
    days = pd.date_range("2010-01-01", "2017-01-01", freq="7D")
    def active(mask):
        sub = st[mask]
        return np.array([((sub.epoch_start <= d) & (sub.epoch_end >= d)).sum() for d in days])
    n_sea, n_land = active(st.sea), active(~st.sea)
    wk = ev.set_index("time").resample("7D").size()
    t0 = "2010-01-01T"; t1 = "2017-01-01T"
    fig.basemap(region=[t0, t1, 0, max(n_land.max(), n_sea.max()) * 1.15], projection=f"X{W}c/{H}c",
                frame=["wSnE", "xa1Yf3o", "ya100f50+lstations operating"])
    x = [d.strftime("%Y-%m-%dT") for d in days]
    fig.plot(x=x + x[::-1], y=list(n_sea) + [0] * len(x), fill="#0072B2@55", close=True)
    fig.plot(x=x, y=n_sea, pen="0.6p,#0072B2", label="ocean-bottom")
    fig.plot(x=x, y=n_land, pen="0.6p,gray25", label="land")
    fig.basemap(region=[t0, t1, 0, wk.max() * 1.1], projection=f"X{W}c/{H}c", frame=["W", "ya2000f1000+levents per week"])
    fig.plot(x=[d.strftime("%Y-%m-%dT") for d in wk.index], y=wk.values, pen="0.45p,#D55E00", label="events per week")
    fig.legend(position="jTL+o0.55c/0.1c", box="+gwhite@15+p0.3p,gray60")
    tag(fig, "D")
    save(fig, "fig1_overview")


def _legend_mag(fig, region, proj, s):
    mags = [1, 2, 3, 4, 5]
    with fig.inset(position="jTL+w1.9c/1.15c+o0.12c/0.12c", box="+gwhite@15+p0.3p,gray50+c0.06c"):
        fig.basemap(region=[0, 1.9, 0, 1.15], projection="X1.9c/1.15c", frame="+n")
        fig.text(x=0.95, y=1.0, text="magnitude", font="6p,Helvetica,black", justify="CM")
        for i, m in enumerate(mags):
            xc = 0.2 + i * 0.37
            fig.plot(x=[xc], y=[0.55], style=f"c{msize([m], s)[0]:.3f}c", fill="gray70", pen="0.3p,gray20")
            fig.text(x=xc, y=0.15, text=str(m), font="5.5p,Helvetica,black", justify="CM")


# ------------------------------------------------------------------ zoom map
def zoom_map(fig, region, width, ev, faults, tmp, color="fault", frame=("WSne",), tensors=True, legend=True, mw_min=4.5, size=1.0):
    proj = f"M{width}c"
    span = region[1] - region[0]
    step = 1 if span > 3 else 0.5 if span > 1.2 else 0.2
    fr = list(frame) + [f"xa{step}f{step / 2}", f"ya{step}f{step / 2}"]
    basemap(fig, region, proj, tmp, frame=fr)
    cfm_traces(fig, region)
    plate_boundaries(fig, region, 1.0)
    W, E, S, N = region
    e = ev[ev.qc & ev.lon.between(W, E) & ev.lat.between(S, N)].sort_values("M", na_position="first")
    s = size * 1.2 * min(1.6, 4.0 / span) ** 0.5
    inf = [f for f in faults if (f["corners"][:, 0].max() >= W) and (f["corners"][:, 0].min() <= E)
           and (f["corners"][:, 1].max() >= S) and (f["corners"][:, 1].min() <= N)]
    # footprints of the rectangles, then events, then top edges
    for f in inf:
        fig.plot(x=f["corners"][:, 0], y=f["corners"][:, 1], close=True, fill=f["color"] + "@82", pen=f"0.25p,{f['color']}@40")
    if color == "fault":
        off = e[e.fault.isna()]
        fig.plot(x=off.lon, y=off.lat, size=msize(off.M, s), style="c", fill="gray55", pen="0.05p,gray30@50", transparency=45)
        col = {f["id"]: f["color"] for f in faults}
        on = e[e.fault.notna()]
        for c_, g in on.groupby(on.fault.map(col)):
            fig.plot(x=g.lon, y=g.lat, size=msize(g.M, s), style="c", fill=c_, pen="0.05p,gray15@40", transparency=15)
    else:
        if color == "depth":
            pygmt.makecpt(cmap="lajolla", series=[0, max(10, float(e.dep.quantile(0.98)))], output=f"{tmp}/z.cpt", background=True, reverse=True)
            val, lab = e.dep, "depth (km)"
        else:
            pygmt.makecpt(cmap="batlow", series=[2010, 2017], output=f"{tmp}/z.cpt", background=True)
            val, lab = e.time.dt.year + e.time.dt.dayofyear / 366, "origin time (year)"
        fig.plot(x=e.lon, y=e.lat, size=msize(e.M, s), style="c", fill=val, cmap=f"{tmp}/z.cpt", pen="0.05p,gray20@50", transparency=15)
        fig.colorbar(cmap=f"{tmp}/z.cpt", position="jBL+w2.4c/0.13c+h+o0.25c/0.3c", box="+gwhite@15+p0.3p,gray50+c0.1c/0.08c",
                     frame=[f"x+l{lab}"])
    for f in inf:
        fig.plot(x=f["top"][:, 0], y=f["top"][:, 1], pen=f"1.1p,{f['color']}")
        fig.plot(x=f["top"][:, 0], y=f["top"][:, 1], pen="0.25p,white")
    if tensors:
        mt = mctx.load_mt(region, None)
        if mt is not None:
            mt = mt[mt.time.between(pd.Timestamp("2010-01-01", tz="UTC"), pd.Timestamp("2017-01-01", tz="UTC")) & (mt.Mw >= mw_min)]
            mctx.draw_mechanisms(fig, mt, 0.2, fill="gray25", pen="0.2p,gray10")
    fig.basemap(region=region, projection=proj, map_scale=f"jBR+w{_nice(span * 111 * 0.25)}k+o0.3c/0.3c+f+l")
    return len(e), len(inf)


def map_height(region, width):
    """Height (cm) of a Mercator map of `region` drawn `width` cm wide."""
    m = lambda la: np.log(np.tan(np.pi / 4 + np.radians(la) / 2))
    return width * (m(region[3]) - m(region[2])) / np.radians(region[1] - region[0])


def _nice(x):
    for v in (5, 10, 20, 25, 50, 100, 200):
        if v >= x * 0.6:
            return v
    return 200


def fig_zoom(a):
    ev, faults = events(), fault_surfaces()
    tmp = tempfile.mkdtemp(); config()
    fig = pygmt.Figure()
    n, nf = zoom_map(fig, REGIONS[a.region], a.width, ev, faults, tmp, color=a.color)
    save(fig, f"zoom_{a.region}_{a.color}")
    print(f"  {n:,} QC events, {nf} fault surfaces")


# ------------------------------------------------------------------ 3D perspective
def view_projector(r3, width, zsize, az, el):
    """lon, lat, z (km, up) -> page x, y (cm, from the lower-left of the projected box) and distance from the viewer."""
    W, E, S, N, z0, z1 = r3
    m = lambda la: np.log(np.tan(np.pi / 4 + np.radians(np.asarray(la, float)) / 2))
    sa, ca, se, ce = np.sin(np.radians(az)), np.cos(np.radians(az)), np.sin(np.radians(el)), np.cos(np.radians(el))
    def raw(lon, lat, z):
        x = width * (np.asarray(lon, float) - W) / (E - W)
        y = width * (m(lat) - m(S)) / np.radians(E - W)
        zc = zsize * (np.asarray(z, float) - z0) / (z1 - z0)
        return -x * ca + y * sa, -(x * sa + y * ca) * se + zc * ce, -(x * sa + y * ca) * ce - zc * se
    cx, cy, _ = raw(*np.array([[x, y, z] for x in (W, E) for y in (S, N) for z in (z0, z1)]).T)
    def page(lon, lat, z):
        px, py, far = raw(lon, lat, z)
        return px - cx.min(), py - cy.min(), far
    return page


def _shade(hexcol, f):
    """Darken (f < 1) or lighten towards white (f > 1) a #rrggbb colour."""
    c = np.array([int(hexcol[i:i + 2], 16) for i in (1, 3, 5)], float)
    c = c * f if f <= 1 else c + (255 - c) * min(1.0, f - 1)
    return "#%02x%02x%02x" % tuple(np.clip(c, 0, 255).astype(int))


def perspective(fig, region, width, ev, faults, tmp, az, el, zmax, exag, color="fault", frame=True):
    W, E, S, N = region
    lat0 = (S + N) / 2
    km_w = (E - W) * 111.195 * np.cos(np.radians(lat0))
    z = relief(region, max_nodes=500) / 1000.0                      # km
    ztop = max(0.5, float(z.max()))
    zsize = (zmax + ztop) * exag * width / km_w
    r3 = [W, E, S, N, -zmax, ztop]
    proj = f"M{width}c"
    fr = ["xaf", "yaf", f"za{10 if zmax > 20 else 5}f{5 if zmax > 20 else 1}+lelevation (km)", "wSEnZ+b"] if frame else None
    cpt = relief_cpt(tmp, 0.001)
    shf = f"{tmp}/shade3d.nc"
    hillshade(z * 1000).to_netcdf(shf)
    fig.grdview(z, region=r3, projection=proj, zsize=f"{zsize:.3f}c", perspective=[az, el], surftype="i", cmap=cpt,
                shading=shf, frame=fr, transparency=35)
    # Everything below the surface goes on a flat overlay in page coordinates, so hypocentres are discs facing the
    # viewer (shaded like spheres) instead of GMT's symbols lying in the horizontal plane. The projection is GMT's
    # parallel view (gmt_xyz_to_xy), offset so the projected box starts at the origin; checked against plot3d.
    page = view_projector(r3, width, zsize, az, el)
    inf = [f for f in faults if W <= f["corners"][:, 0].mean() <= E and S <= f["corners"][:, 1].mean() <= N]
    e = ev[ev.qc & ev.lon.between(W, E) & ev.lat.between(S, N) & ev.dep.between(-ztop, zmax)].copy()
    e["px"], e["py"], e["far"] = page(e.lon, e.lat, -e.dep)
    e["d"] = np.minimum(msize(e.M, 1.0), 0.45)
    if color == "fault":
        col = {f["id"]: f["color"] for f in faults}
        e["c"] = e.fault.map(col).fillna("#9a9a9a")
    else:
        cpt = pygmt.makecpt(cmap="lajolla", series=[0, zmax], output=f"{tmp}/z3.cpt", background=True, reverse=True)
    ov = dict(region=[0, 40, 0, 40], projection="x1c", no_clip=True)
    items = []
    for f in inf:
        cx, cy, cf = page(f["corners"][:, 0], f["corners"][:, 1], -f["corners"][:, 2])
        tx, ty, _ = page(f["top"][:, 0], f["top"][:, 1], -f["top"][:, 2])
        items.append((float(cf.mean()), "f", (cx, cy, tx, ty, f["color"])))
    e = e.sort_values("far", ascending=False)                 # farthest first
    nb = 16
    for i, g in enumerate(np.array_split(e, nb)):
        if len(g):
            items.append((float(g.far.min()), "e", g))
    for _, kind, obj in sorted(items, key=lambda t: -t[0]):
        if kind == "f":
            cx, cy, tx, ty, c_ = obj
            fig.plot(x=np.r_[cx, cx[:1]], y=np.r_[cy, cy[:1]], fill=c_ + "@82", pen=f"0.3p,{c_}@30", **ov)
            fig.plot(x=tx, y=ty, pen=f"1.0p,{c_}", **ov)
            continue
        g = obj
        if color == "fault":
            for c_, h in g.groupby("c"):
                grey = c_ == "#9a9a9a"
                fig.plot(x=h.px, y=h.py, size=h.d, style="c", fill=c_, pen=f"0.15p,{_shade(c_, 0.55)}", transparency=50 if grey else 0, **ov)
                fig.plot(x=h.px - 0.18 * h.d, y=h.py + 0.18 * h.d, size=0.38 * h.d, style="c", fill=_shade(c_, 1.6),
                         transparency=70 if grey else 35, **ov)
        else:
            fig.plot(x=g.px, y=g.py, size=g.d, style="c", fill=g.dep, cmap=f"{tmp}/z3.cpt", pen="0.15p,gray25", **ov)
            fig.plot(x=g.px - 0.18 * g.d, y=g.py + 0.18 * g.d, size=0.38 * g.d, style="c", fill="white", transparency=55, **ov)
    bx, by, _ = page(np.array([E, E]), np.array([S, S]), np.array([-zmax, -zmax]))
    fig.text(x=bx[0], y=by[0] - 0.35, text=f"vertical exaggeration {exag:g}:1", font="5.5p,Helvetica,gray30", justify="TR", **ov)
    return len(e), len(inf)


def fig_perspective(a):
    ev, faults = events(), fault_surfaces()
    tmp = tempfile.mkdtemp(); config()
    az0, el0, z0, x0 = VIEWS.get(a.region, (200, 30, 30, 3.0))
    fig = pygmt.Figure()
    n, nf = perspective(fig, REGIONS[a.region], a.width, ev, faults, tmp, a.az or az0, a.el or el0, a.zmax or z0, a.exag or x0,
                        color=a.color)
    save(fig, f"perspective_{a.region}_{a.color}")
    print(f"  {n:,} QC events, {nf} fault surfaces")


# ------------------------------------------------------------------ Fig. 3
def fig_faults(a):
    """A: the margin, surfaces and on-fault events in one colour per fault; B: the Blanco strands; C, D: 3D views."""
    ev, faults = events(), fault_surfaces()
    tmp = tempfile.mkdtemp(); config()
    fig = pygmt.Figure()
    cur = [-0.5, -2.0]                                  # start 2 cm up: PyGMT layouts begin at page y = 0
    def goto(x, y):                                     # absolute placement of the next panel's origin (cm)
        fig.shift_origin(xshift=f"{x - cur[0]:.3f}c", yshift=f"{y - cur[1]:.3f}c"); cur[:] = [x, y]
    X0, WA = 0.9, 6.0
    WB = FULL - X0 - WA - 0.5
    hA, hB = map_height(MARGIN, WA), map_height(REGIONS["blanco_strip"], WB)
    W3, Y0 = 7.0, 7.9
    goto(X0 + 0.2, 0)
    perspective(fig, REGIONS["blanco_central"], W3, ev, faults, tmp, *VIEWS["blanco_central"])
    goto(X0 + W3 + 2.9, 0)
    perspective(fig, REGIONS["mendocino"], W3, ev, faults, tmp, *VIEWS["mendocino"])
    goto(X0, Y0)
    proj = f"M{WA}c"
    fr = ["WSne", "xa4f1", "ya2f1"]
    basemap(fig, MARGIN, proj, tmp, frame=fr)
    cfm_traces(fig, MARGIN); plate_boundaries(fig, MARGIN, 0.7)
    q = ev[ev.qc]
    fig.plot(x=q[q.fault.isna()].lon, y=q[q.fault.isna()].lat, style="c0.02c", fill="gray50", transparency=60)
    col = {f["id"]: f["color"] for f in faults}
    on = q[q.fault.notna()]
    for c_, g in on.groupby(on.fault.map(col)):
        fig.plot(x=g.lon, y=g.lat, style="c0.025c", fill=c_, transparency=30)
    for f in faults:
        fig.plot(x=f["top"][:, 0], y=f["top"][:, 1], pen=f"0.9p,{f['color']}")
    for k in ("blanco_strip", "blanco_central", "mendocino"):
        b = REGIONS[k]
        fig.plot(x=[b[0], b[1], b[1], b[0], b[0]], y=[b[2], b[2], b[3], b[3], b[2]], pen="0.5p,black,4_2" if k != "blanco_strip" else "0.6p,black")
    for k, t in (("blanco_central", "C"), ("mendocino", "D")):
        b = REGIONS[k]
        fig.text(x=b[1], y=b[3], text=t, font="6.5p,Helvetica-Bold,black", justify="BL", offset="0.04c/0.04c", fill="white@20", clearance="0.03c/0.02c")
    fig.text(position="BL", offset="0.2c/0.2c", justify="BL", font="5.5p,Helvetica,black", fill="white@15", clearance="0.06c/0.05c",
             text=f"{len(faults)} surfaces, {on.fault.notna().sum():,} QC events on a fault")
    fig.basemap(region=MARGIN, projection=proj, frame=fr)
    goto(X0 + WA + 0.5, Y0 + hA - hB)
    zoom_map(fig, REGIONS["blanco_strip"], WB, ev, faults, tmp, color="fault", frame=("wsNE",), mw_min=5.0, size=0.8)
    # legend in the space under B
    spec = Path(tmp) / "leg.txt"
    spec.write_text("\n".join([
        "N 2",
        "S 0.25c - 0.4c - 1.1p,#0072B2 0.6c fault surface v0 (top edge)",
        "S 0.25c r 0.4c #0072B2@82 0.3p,#0072B2 0.6c rectangle in map view",
        "S 0.25c c 0.16c #0072B2 0.1p,gray15 0.6c event on that fault",
        "S 0.25c c 0.16c gray55 0.1p,gray30 0.6c QC event on no fault",
        "S 0.25c - 0.4c - 0.6p,#b8860b 0.6c CRESCENT CFM fault trace",
        "S 0.25c - 0.4c - 0.8p,gray15 0.6c plate boundary (PB2002)",
    ]))
    goto(X0 + WA + 0.5, Y0 + hA - hB - 1.75)
    with pygmt.config(FONT_ANNOT_PRIMARY="5.5p,Helvetica,black"):
        fig.legend(spec=str(spec), region=[0, 1, 0, 1], projection=f"X{WB}c/1.5c", position=f"jTL+w{WB * 0.8:.2f}c+o0c/0c",
                   box="+gwhite+p0.3p,gray60+c0.1c/0.08c")
    # panel letters on a page-sized overlay
    goto(0, 0)
    H = Y0 + hA
    fig.basemap(region=[0, FULL, 0, H], projection=f"X{FULL}c/{H}c", frame="+n")
    for x, y, t in ((0.0, H, "A"), (X0 + WA + 0.5, H, "B"), (0.0, Y0 - 0.5, "C"), (X0 + W3 + 2.3, Y0 - 0.5, "D")):
        fig.text(x=x, y=y, text=t, font="8p,Helvetica-Bold,black", justify="TL", no_clip=True)
    save(fig, "fig3_fault_surfaces")


# ------------------------------------------------------------------ Fig. 4: Blanco strands
BL_REF, BL_AZ = (-130.55, 44.52), 111.0          # NW end near the Juan de Fuca ridge-transform intersection; transform strike
BL_BASINS = [(12, "WBD"), (95, "SD"), (150, "CAS"), (215, "Blanco Ridge"), (300, "GD")]


_BL_TRACE = None


def _blanco_trace():
    """PB2002 Blanco transform as one polyline (km, local), NW to SE; ridge segments at its ends excluded."""
    global _BL_TRACE
    if _BL_TRACE is None:
        pb = mctx.plate_boundaries([-130.8, -126.4, 42.7, 44.8])
        segs = [L for L in pb["TF"] if L[:, 0].min() < -126.4 and L[:, 0].max() > -130.8]
        L = max(segs, key=len)
        L = L[np.argsort(L[:, 0])]
        L = L[(L[:, 0] >= -130.7) & (L[:, 0] <= -126.5)]
        x = (L[:, 0] - BL_REF[0]) * 111.2 * np.cos(np.radians(44.0)); y = (L[:, 1] - BL_REF[1]) * 111.2
        _BL_TRACE = np.c_[x, y]
    return _BL_TRACE


def _blanco_xy(lon, lat):
    """lon, lat -> km along the PB2002 Blanco trace (arc length from its NW end) and across it (positive to the NE side)."""
    T = _blanco_trace()
    P = np.c_[(np.asarray(lon, float) - BL_REF[0]) * 111.2 * np.cos(np.radians(44.0)), (np.asarray(lat, float) - BL_REF[1]) * 111.2]
    A, B = T[:-1], T[1:]; AB = B - A; L2 = (AB ** 2).sum(1); cum = np.r_[0, np.cumsum(np.sqrt(L2))]
    t = np.clip(((P[:, None, :] - A[None]) * AB[None]).sum(2) / L2[None], 0, 1)
    Q = A[None] + t[..., None] * AB[None]
    d2 = ((P[:, None, :] - Q) ** 2).sum(2); k = d2.argmin(1); i = np.arange(len(P))
    along = cum[k] + t[i, k] * np.sqrt(L2[k])
    cross = AB[k, 0] * (P[i, 1] - A[k, 1]) - AB[k, 1] * (P[i, 0] - A[k, 0])
    return along, np.sign(cross) * np.sqrt(d2[i, k])


def fig_blanco(a):
    """A: Blanco events in a transform-aligned frame, coloured by strand, with strand top edges and published basins;
    B: separation of overlapping parallel strands (Ashman's D, overall and per year); C: dip needed if a pair were one fault."""
    tmp = tempfile.mkdtemp(); config()
    ev, faults = events(), fault_surfaces()
    V0 = D / "faults" / "v0"
    sep = pd.read_csv(V0 / "fault_surfaces_v0_strand_separation.csv")
    lit = pd.read_csv(V0 / "fault_surfaces_v0_literature.csv").set_index("surface_id")
    bl = [f for f in faults if f["id"].startswith("BL")]
    fig = pygmt.Figure()
    cur = [-0.5, -2.0]
    def goto(x, y):
        fig.shift_origin(xshift=f"{x - cur[0]:.3f}c", yshift=f"{y - cur[1]:.3f}c"); cur[:] = [x, y]
    # ---- A: strip view
    W, H, Y1 = FULL - 1.6, 5.2, 6.6
    goto(1.2, Y1)
    q = ev[ev.qc & ev.lon.between(-130.8, -126.4) & ev.lat.between(42.7, 44.8)].copy()
    q["u"], q["v"] = _blanco_xy(q.lon, q.lat)
    umax = float(np.ceil(_blanco_xy(*_blanco_trace()[-1:].T * 0 + np.array([[-126.5], [43.0]]))[0][0] / 10) * 10) if False else 370
    q = q[q.u.between(-5, umax) & q.v.between(-15, 15)]
    reg = [-5, umax, -15, 15]
    fig.basemap(region=reg, projection=f"X{W}c/{H}c", frame=["WSne", "xa50f10+ldistance along the PB2002 Blanco trace from the Juan de Fuca ridge (km)",
                                                             "ya5f1+ldistance from the trace (km, NE positive)"])
    off = q[q.fault.isna() | ~q.fault.fillna("").str.startswith("BL")]
    fig.plot(x=off.u, y=off.v, style="c0.03c", fill="gray65", transparency=40)
    col = {f["id"]: f["color"] for f in bl}
    on = q[q.fault.fillna("").str.startswith("BL")]
    for c_, g in on.groupby(on.fault.map(col)):
        fig.plot(x=g.u, y=g.v, style="c0.035c", fill=c_, transparency=20)
    for f in bl:
        u, v = _blanco_xy(f["top"][:, 0], f["top"][:, 1])
        fig.plot(x=u, y=v, pen=f"1.6p,{f['color']}"); fig.plot(x=u, y=v, pen="0.3p,white")
        fig.text(x=u.mean(), y=v.mean(), text=f["id"], font=f"4.5p,Helvetica-Bold,{f['color']}", offset="0c/0.18c", fill="white@30",
                 clearance="0.02c/0.01c")
    for uu, name in BL_BASINS:
        fig.text(x=uu, y=14.3, text=name, font="6p,Helvetica-Bold,gray20", justify="TC")
    fig.text(position="BR", offset="-0.1c/0.1c", justify="BR", font="5.5p,Helvetica,black", fill="white@15",
             text=f"{len(on):,} QC events on {len(bl)} Blanco strands; vertical exaggeration {(H / 30) / (W / (umax + 5)):.1f}:1")
    tags = [(0.0, Y1 + H + 0.1, "A")]
    # ---- B: Ashman's D
    sep = sep.copy()
    sep["u"] = [np.mean(_blanco_xy(*np.r_[[next(f for f in bl if f["id"] == s_)["top"].mean(0)[:2] for s_ in (r.a, r.b)]].T)[0])
                if all(any(f["id"] == s_ for f in bl) for s_ in (r.a, r.b)) else np.nan for r in sep.itertuples()]
    sep = sep.dropna(subset=["u", "D"]).sort_values("u").reset_index(drop=True)
    WB, HB = 9.6, 4.4
    goto(1.2, 0)
    fig.basemap(region=[0.6, 20, -0.7, len(sep) - 0.3], projection=f"X{WB}cl/-{HB}c",
                frame=["WSne", "xa3f2+lAshman's D (separation of the two strands)", "y+0"])
    fig.plot(x=[2, 2], y=[-0.7, len(sep) - 0.3], pen="0.6p,#D55E00,4_2")
    for k, r in sep.iterrows():
        if np.isfinite(r.D_per_year_min):
            fig.plot(x=[r.D_per_year_min, max(r.D, r.D_per_year_median)], y=[k, k], pen="1.4p,gray70")
        fig.plot(x=[r.D], y=[k], style="c0.14c", fill="#0072B2" if r.separation_justified else "white", pen="0.4p,#0072B2")
        fig.text(x=0.63, y=k, text=f"{r.a}/{r.b}  {r.sep_km:.1f} km", font="4.5p,Helvetica,black", justify="LM")
    fig.text(x=2.05, y=-0.45, text="D = 2", font="5p,Helvetica,#D55E00", justify="LM")
    fig.text(position="BR", offset="-0.1c/0.1c", justify="BR", font="5p,Helvetica,gray25", fill="white@15",
             text="dot: all years; grey: smallest per-year D to the median")
    tags.append((0.0, HB + 0.1, "B"))
    # ---- C: dip needed if the pair were one fault
    goto(1.2 + WB + 1.6, 0)
    WC = FULL - cur[0] - 0.4
    dd = sep.dip_if_single_fault.clip(0, 90)
    fig.basemap(region=[0, 4, 0, 90], projection=f"X{WC}c/{HB}c",
                frame=["WSne", "xa1f0.5+lseparation of the strands (km)", "ya15f5+ldip needed for one fault (@.)"])
    fig.plot(x=[0, 4, 4, 0], y=[70, 70, 90, 90], fill="gray92", close=True)
    fig.text(x=0.1, y=88, text="transform-like dips (70-90@.)", font="5p,Helvetica,gray30", justify="TL")
    fig.plot(x=sep.sep_km, y=dd, style="c0.16c", fill=np.where(sep.separation_justified, "#0072B2", "white").tolist() if False else "#0072B2",
             pen="0.4p,gray15")
    fig.text(x=0.1, y=66, text="below: one fault would need a shallow dip", font="5p,Helvetica,gray30", justify="TL")
    tags.append((cur[0] - 1.1, HB + 0.1, "C"))
    goto(0, 0)
    Ht = Y1 + H + 0.4
    fig.basemap(region=[0, FULL, 0, Ht], projection=f"X{FULL}c/{Ht}c", frame="+n")
    for x, y, t in tags:
        fig.text(x=x, y=y, text=t, font="8p,Helvetica-Bold,black", justify="TL", no_clip=True)
    save(fig, "fig4_blanco_strands")


# ------------------------------------------------------------------ Fig. 5: clustering
CL = D / "catalog_v4" / "clusters_zbz"
RNAME = {"AX": "Axial", "EN": "Endeavour", "BL": "Blanco", "GO": "Gorda", "MT": "Mendocino TJ", "WA": "WA forearc",
         "OR": "OR forearc", "PU": "Puget", "ON": "onshore"}
SETTINGS = [("Ridges and transforms", ["AX", "EN", "BL", "GO"], "#D55E00"), ("Mendocino triple junction", ["MT"], "#0072B2"),
            ("Forearc and onshore", ["WA", "OR", "PU", "ON"], "#009E73")]
TYPES = [("pct_background", "background", "#d9d9d9"), ("pct_MA", "mainshock-aftershock", "#0072B2"),
         ("pct_swarm", "swarm", "#D55E00"), ("pct_multiplet", "multiplet", "#CC79A7")]


# label positions in panel E, set for the converged magnitudes (D-B091; Blanco, Gorda, WA and OR moved)
LABPOS = {"AX": ("TC", "0c/-0.45c"), "BL": ("BC", "0c/0.5c"), "GO": ("TL", "0.1c/-0.2c"), "ON": ("BC", "0c/0.32c"),
          "PU": ("BC", "0c/0.3c"), "EN": ("LM", "0.22c/0c"), "MT": ("RM", "-0.42c/0c"), "WA": ("RM", "-0.2c/0c"), "OR": ("BC", "0c/0.2c")}


def fig_clusters(a):
    """A-C: rescaled time T and distance R to the nearest-neighbour parent, by setting, with each region's threshold;
    D: share of events per class and region; E: swarm share of families (>= 5 events) against aftershock productivity."""
    tmp = tempfile.mkdtemp(); config()
    ev = pd.read_csv(CL / "events_zbz.csv.gz", usecols=["region", "in_analysis", "log_T", "log_R", "log_eta"])
    ev = ev[ev.in_analysis & ev.log_T.notna() & ev.log_R.notna()]
    reg = pd.read_csv(CL / "regions_zbz.csv").set_index("region")
    st = pd.read_csv(CL / "setting_stats.csv").set_index("region")
    fig = pygmt.Figure()
    cur = [-0.5, -2.0]                                  # start 2 cm up: PyGMT layouts begin at page y = 0
    def goto(x, y):
        fig.shift_origin(xshift=f"{x - cur[0]:.3f}c", yshift=f"{y - cur[1]:.3f}c"); cur[:] = [x, y]
    tx, rx = (-9.0, 0.5), (-4.5, 2.5)
    pw, ph, y0 = 5.2, 4.2, 6.4
    pygmt.makecpt(cmap="lapaz", series=[0, 1], reverse=True, output=f"{tmp}/dens.cpt", background=True)
    for i, (name, regs, col) in enumerate(SETTINGS):
        goto(1.0 + i * (pw + 0.7), y0)
        e = ev[ev.region.isin(regs)]
        H, xe, ye = np.histogram2d(e.log_T, e.log_R, bins=[np.arange(tx[0], tx[1] + 0.1, 0.1), np.arange(rx[0], rx[1] + 0.1, 0.1)])
        from scipy.ndimage import gaussian_filter
        H = gaussian_filter(H, 1.2)
        g = xr.DataArray((H / H.max()).T, coords={"y": (ye[:-1] + ye[1:]) / 2, "x": (xe[:-1] + xe[1:]) / 2}, dims=("y", "x"))
        fr = ["WSne" if i == 0 else "wSne", "xa2f1+lrescaled time, log@-10@- T", "ya1f0.5" + ("+lrescaled distance, log@-10@- R" if i == 0 else "")]
        fig.grdimage(g, region=[*tx, *rx], projection=f"X{pw}c/{ph}c", cmap=f"{tmp}/dens.cpt", nan_transparent=True, frame=fr)
        fig.grdcontour(g, levels=0.2, pen="0.25p,white@30")
        for j, r in enumerate(sorted(regs, key=lambda r: reg.loc[r, "log_eta0"])):
            e0 = reg.loc[r, "log_eta0"]
            xx = np.array(tx)
            fig.plot(x=xx, y=e0 - xx, pen="0.6p,black,4_2")
            yl = rx[0] + 0.35 + 0.55 * j                      # labels staggered down the lower-right ends of the lines
            fig.text(x=e0 - yl, y=yl, text=r, font="5p,Helvetica-Bold,black", justify="CM", fill="white@10", clearance="0.02c/0.01c")
        n_cl = (e.log_eta < e.region.map(reg.log_eta0)).mean() * 100
        fig.text(position="TR", offset="-0.1c/-0.1c", justify="TR", font="6p,Helvetica-Bold," + col, text=name, fill="white@15")
        fig.text(x=tx[0] + 0.3, y=rx[0] + 0.3, text="clustered", font="5p,Helvetica-Oblique,gray20", justify="BL")
        fig.text(x=tx[1] - 0.3, y=rx[1] - 1.3, text="background", font="5p,Helvetica-Oblique,gray20", justify="TR")
        fig.text(position="TR", offset="-0.1c/-0.45c", justify="TR", font="5.5p,Helvetica,black", fill="white@15",
                 text=f"{len(e):,} events, {n_cl:.0f}% clustered")
        fig.text(position="TL", text="ABC"[i], font="8p,Helvetica-Bold,black", justify="TR", offset="-0.25c/0c", no_clip=True)
    fig.colorbar(cmap=f"{tmp}/dens.cpt", position=f"JMR+o0.3c/0c+w{ph * 0.8:.1f}c/0.15c", frame=["xa0.5+lrelative density"])
    # D: class shares per region
    order = [r for _, regs, _ in SETTINGS for r in regs]
    goto(1.0 + 1.3, 0)
    bw, bh = 7.0, 5.0
    fig.basemap(region=[0, 100, -0.6, len(order) - 0.4], projection=f"X{bw}c/-{bh}c",
                frame=["WSne", "xa20f10+lshare of events above Mc (%)", "y+0"])
    base = np.zeros(len(order))
    for col, lab, colr in TYPES:
        v = reg.loc[order, col].to_numpy()
        fig.plot(data=np.c_[base + v, np.arange(len(order)), base], style="B0.42c+b", fill=colr, pen="0.2p,white")
        base += v
    for k, r in enumerate(order):
        sc = next(c for _, regs, c in SETTINGS if r in regs)
        fig.text(x=-1.5, y=k, text=RNAME[r], font=f"6p,Helvetica,{sc}", justify="RM", no_clip=True)
        fig.text(x=101.5, y=k, text=f"{int(reg.loc[r, 'n_analysis']):,}", font="5p,Helvetica,gray25", justify="LM", no_clip=True)
    leg = Path(tmp) / "cls.txt"
    leg.write_text("N 1 1.7 0.8 0.9\n" + "\n".join(f"S 0.2c s 0.28c {c} 0.2p,white 0.45c {l}" for _, l, c in TYPES) + "\n")
    with pygmt.config(FONT_ANNOT_PRIMARY="6p,Helvetica,black"):
        fig.legend(spec=str(leg), position=f"JBC+o0c/1.05c+w{bw}c", box=False)
    fig.text(position="TL", text="D", font="8p,Helvetica-Bold,black", justify="TR", offset="-2.0c/0c", no_clip=True)
    # E: swarms vs productivity
    goto(1.0 + 1.3 + bw + 2.2, 0)
    ew = FULL - cur[0] - 0.4
    fig.basemap(region=[45, 90, -6, 106], projection=f"X{ew}c/{bh}c",
                frame=["WSne", "xa10f5+levents 1-2 units above Mc with no aftershock (%)", "ya20f10+lswarms among families of @~\\263@~5 events (%)"])
    for name, regs, col in SETTINGS:
        for r in regs:
            x, y, n = st.loc[r, "pct_Mc1to2_no_offspring"], st.loc[r, "pct_swarm_of_ge5"], st.loc[r, "n_fam_ge5"]
            fig.plot(x=[x], y=[y], style=f"c{0.12 + 0.05 * np.sqrt(n):.2f}c", fill=col + "@15", pen="0.4p,gray15")
            j, o = LABPOS.get(r, ("LM", "0.28c/0c"))
            fig.text(x=x, y=y, text=f"{RNAME[r]} ({n})", font="5.5p,Helvetica,black", justify=j, offset=o)
    fig.text(position="BL", offset="0.1c/0.1c", justify="BL", font="5p,Helvetica,gray25", text="symbol area ~ number of families of @~\\263@~5 events")
    fig.text(position="TL", text="E", font="8p,Helvetica-Bold,black", justify="TR", offset="-1.0c/0c", no_clip=True)
    save(fig, "fig5_clustering")


# ------------------------------------------------------------------ Fig. 6: short-lived processes
def _obs_near(st, lon, lat, km, days):
    s = st[st.sea]
    d = np.hypot((s.lon - lon) * 111.2 * np.cos(np.radians(lat)), (s.lat - lat) * 111.2)
    s = s[d < km]
    return np.array([((s.epoch_start <= t) & (s.epoch_end >= t)).sum() for t in days])


def fig_short(a):
    """A: Axial before the April 2015 eruption; B: the May 2013 Escanaba swarm; C: the persistent Gorda source near
    41.85 N, 126.1 W; D: the western Mendocino transform, 2012-2016, with the ocean-bottom stations nearby."""
    tmp = tempfile.mkdtemp(); config()
    ez = pd.read_csv(CL / "events_zbz.csv.gz", parse_dates=["time"])
    fam = pd.read_csv(CL / "families_zbz.csv", parse_dates=["start", "end"])
    reg = pd.read_csv(CL / "regions_zbz.csv").set_index("region")
    ev, faults, st = events(), fault_surfaces(), stations()
    fig = pygmt.Figure()
    cur = [-0.5, -2.0]                                  # start 2 cm up: PyGMT layouts begin at page y = 0
    def goto(x, y):
        fig.shift_origin(xshift=f"{x - cur[0]:.3f}c", yshift=f"{y - cur[1]:.3f}c"); cur[:] = [x, y]
    W2, H2, Y1 = 7.6, 4.6, 6.6
    # ---- A: Axial
    goto(1.1, Y1)
    t0, t1 = pd.Timestamp("2014-10-01", tz="UTC"), pd.Timestamp("2015-07-01", tz="UTC")   # Axial is in the catalog from Oct 2014 (OBS)
    ax = ez[(ez.region == "AX") & ez.in_analysis & ez.time.between(t0, t1)]
    wk = ax.set_index("time").resample("7D").size()
    sw = fam[(fam.region == "AX") & (fam.type == "swarm")]
    sw = sw[sw.start.dt.tz_localize("UTC").between(t0, t1)] if sw.start.dt.tz is None else sw[sw.start.between(t0, t1)]
    mo = sw.groupby(sw.start.dt.to_period("M")).size()
    T = lambda t: pd.Timestamp(t).strftime("%Y-%m-%dT")
    with pygmt.config(FORMAT_DATE_MAP="o", FORMAT_TIME_PRIMARY_MAP="abbreviated"):
        fig.basemap(region=[T(t0), T(t1), 0, wk.max() * 1.15], projection=f"X{W2}c/{H2}c",
                frame=["WSn", "pxa1Of1o", "sxa1Y", f"ya{_tick(wk.max())}+levents @~\\263@~ M@-c@- ({reg.loc['AX', 'Mc']}) per week"])  # noqa
    xs = [T(t) for t in wk.index]
    fig.plot(x=xs + xs[::-1], y=list(wk.values) + [0] * len(xs), fill="gray80", close=True)
    fig.plot(x=xs, y=wk.values, pen="0.5p,gray35")
    erupt = "2015-04-24T08:00"
    fig.plot(x=[erupt, erupt], y=[0, wk.max() * 1.15], pen="0.8p,#D55E00,4_2")
    fig.text(x=erupt, y=wk.max() * 1.08, text="eruption 24 Apr 2015", font="5.5p,Helvetica,#D55E00", justify="RM", offset="-0.08c/0c")
    fig.basemap(region=[T(t0), T(t1), 0, max(18, mo.max() * 1.15)], projection=f"X{W2}c/{H2}c", frame=["E", "ya4f2+lswarm families per month"])
    mx = [T(p.to_timestamp() + pd.Timedelta(days=15)) for p in mo.index]
    fig.plot(x=mx, y=mo.values, style="c0.14c", fill="#D55E00", pen="0.3p,white")
    fig.plot(x=mx, y=mo.values, pen="0.6p,#D55E00")
    fig.text(position="TL", offset="0.1c/-0.1c", justify="TL", font="6p,Helvetica-Bold,black", text="Axial Seamount", fill="white@15")
    tagxy = [(cur[0] - 1.0, cur[1] + H2, "A")]
    # ---- B: West Blanco swarm of June 2015, distance along the transform against time
    goto(1.1 + W2 + 1.9, Y1)
    fr0 = fam[(fam.region == "BL") & (fam.start.dt.strftime("%Y-%m") == "2015-06")].sort_values("n_all").iloc[-1]
    e = ez[ez.family_id == int(fr0.family_id)].sort_values("time")
    az = 111.0                                                             # Blanco strike
    x = (e.lon - e.lon.mean()) * 111.2 * np.cos(np.radians(e.lat.mean())); y = (e.lat - e.lat.mean()) * 111.2
    dist = (x * np.sin(np.radians(az)) + y * np.cos(np.radians(az))).to_numpy()
    days = ((e.time - e.time.iloc[0]).dt.total_seconds() / 86400).to_numpy()
    dmax = 12.0
    keep = days <= dmax
    pygmt.makecpt(cmap="lajolla", series=[1.0, 5.0], reverse=True, output=f"{tmp}/m.cpt", background=True)
    fig.basemap(region=[-0.3, dmax, dist.min() - 2, dist.max() + 2], projection=f"X{W2}c/{H2}c",
                frame=["WSne", f"xa2f1+ldays since {e.time.iloc[0]:%d %b %Y %H:%M} UTC", "ya10f5+ldistance along 111@. (km), ESE positive"])
    ek = e[keep]; o = np.argsort(ek.M.fillna(0).to_numpy())
    fig.plot(x=days[keep][o], y=dist[keep][o], size=np.minimum(msize(ek.M.to_numpy()[o], 1.8), 0.4), style="c", fill=ek.M.fillna(1.0).to_numpy()[o],
             cmap=f"{tmp}/m.cpt", pen="0.15p,gray20")
    big = ek.M.to_numpy() >= 5.0
    fig.plot(x=days[keep][big], y=dist[keep][big], style="a0.32c", fill="white", pen="0.5p,black")
    for dd, yy, mm in zip(days[keep][big], dist[keep][big], ek.M.to_numpy()[big]):
        fig.text(x=dd, y=yy, text=f"M {mm:.1f}", font="5.5p,Helvetica-Bold,black", justify="LM", offset="0.22c/0c")
    fig.text(position="BR", offset="-0.1c/0.1c", justify="BR", font="5.5p,Helvetica,black", fill="white@15",
             text=f"{keep.sum()} events in {dmax:g} days ({len(e)} in the family, {fr0.duration_days:.0f} days)")
    fig.text(position="TL", offset="0.1c/-0.1c", justify="TL", font="6p,Helvetica-Bold,black", fill="white@15", text="West Blanco swarm, June 2015")
    fig.colorbar(cmap=f"{tmp}/m.cpt", position=f"JMR+o0.2c/0c+w{H2 * 0.7:.1f}c/0.12c", frame=["xa1+lM"])
    tagxy.append((cur[0] - 1.0, cur[1] + H2, "B"))
    # ---- C: Gorda source, map
    goto(1.1, 0)
    rc = [-126.45, -125.6, 41.68, 42.05]
    wc = W2 - 0.6
    proj = f"M{wc}c"
    with pygmt.config(FORMAT_GEO_MAP="ddd.xF"):
        basemap(fig, rc, proj, tmp, frame=["WSne", "xa0.2f0.1", "ya0.1f0.05"])
    q = ev[ev.qc & ev.lon.between(*rc[:2]) & ev.lat.between(*rc[2:])].sort_values("M", na_position="first")
    pygmt.makecpt(cmap="batlow", series=[2010, 2017], output=f"{tmp}/yr.cpt", background=True)
    fig.plot(x=q.lon, y=q.lat, size=msize(q.M, 2.0), style="c", fill=q.time.dt.year + q.time.dt.dayofyear / 366, cmap=f"{tmp}/yr.cpt",
             pen="0.15p,gray20", transparency=10)
    for f in faults:
        if f["id"] in ("GO010", "GO011", "GO014", "GO017"):
            fig.plot(x=f["top"][:, 0], y=f["top"][:, 1], pen=f"1.2p,{f['color']}")
            iN = int(np.argmax(f["top"][:, 1]))
            fig.text(x=f["top"][iN, 0], y=f["top"][iN, 1], text=f["id"], font=f"5p,Helvetica-Bold,{f['color']}",
                     justify="BR" if f["top"][iN, 0] > rc[1] - 0.12 else "BC", offset="0c/0.08c", fill="white@30")
    mt = mctx.load_mt(rc, None)
    if mt is not None and len(mt):
        inw = mt.time.between(pd.Timestamp("2010-01-01", tz="UTC"), pd.Timestamp("2017-01-01", tz="UTC"))
        mctx.draw_mechanisms(fig, mt[~inw], 0.32, fill="gray60", pen="0.2p,gray30")
        mctx.draw_mechanisms(fig, mt[inw], 0.32, fill="gray15", pen="0.2p,black")
        for _, r in mt.iterrows():
            fig.text(x=r.lon, y=r.lat, text=f"{r.time:%Y}", font="4.5p,Helvetica,black", offset="0c/-0.3c")
    fig.colorbar(cmap=f"{tmp}/yr.cpt", position=f"JMR+o0.25c/0c+w{map_height(rc, wc) * 0.7:.1f}c/0.12c", frame=["xa1+lyear"])
    fig.text(position="TL", offset="0.1c/-0.1c", justify="TL", font="6p,Helvetica-Bold,black", fill="white@15", text="Gorda plate, 41.85@.N 126.1@.W")
    tagxy.append((cur[0] - 1.0, cur[1] + map_height(rc, wc), "C"))
    # ---- D: western Mendocino transform, longitude against time
    goto(1.1 + W2 + 1.9, 0)
    t0, t1 = pd.Timestamp("2012-01-01", tz="UTC"), pd.Timestamp("2017-01-01", tz="UTC")
    lon0, lon1 = -127.6, -124.9
    m = ev[ev.qc & ev.lat.between(40.25, 40.62) & ev.lon.between(lon0, lon1) & ev.time.between(t0, t1)].sort_values("M", na_position="first")
    days = pd.date_range("2012-01-01", "2017-01-01", freq="7D")
    nobs = _obs_near(st, -126.2, 40.45, 120, days)
    fig.basemap(region=[T(t0), T(t1), lon0, lon1], projection=f"X{W2}c/{H2}c", frame=["WSne", "xa1Yf3o", "ya0.5f0.25+llongitude (@.E)"])
    on = nobs > 0
    for i0 in np.flatnonzero(on & ~np.r_[False, on[:-1]]):
        i1 = i0
        while i1 + 1 < len(on) and on[i1 + 1]:
            i1 += 1
        fig.plot(x=[T(days[i0]), T(days[i1]), T(days[i1]), T(days[i0])], y=[lon0, lon0, lon1, lon1], fill="#56B4E9@80", close=True)
    fig.plot(x=[T(t) for t in m.time], y=m.lon, size=np.minimum(msize(m.M, 1.6), 0.4), style="c", fill="gray30", pen="0.1p,white", transparency=20)
    big = mctx.load_mt([lon0, lon1, 40.25, 40.62], None)
    if big is not None:
        big = big[big.time.between(t0, t1) & (big.Mw >= 5.0)]
        fig.plot(x=[T(t) for t in big.time], y=big.lon, style="a0.22c", fill="#D55E00", pen="0.25p,black")
        for _, r in big.iterrows():
            late = r.time > pd.Timestamp("2016-07-01", tz="UTC")
            fig.text(x=T(r.time), y=r.lon, text=f"{r.Mw:.1f}", font="5p,Helvetica-Bold,#D55E00", justify="RM" if late else "LM",
                     offset="-0.15c/0c" if late else "0.15c/0c",
                     fill="white@30", clearance="0.02c/0.01c")
    fig.text(position="TL", offset="0.1c/-0.1c", justify="TL", font="6p,Helvetica-Bold,black", fill="white@15",
             text="Mendocino transform, 40.25-40.62@.N")
    fig.text(position="BL", offset="0.1c/0.1c", justify="BL", font="5p,Helvetica,#1f77b4", fill="white@15",
             text="blue: ocean-bottom stations within 120 km; stars: ComCat M@-w@- @~\\263@~ 5")
    tagxy.append((cur[0] - 1.0, cur[1] + H2, "D"))
    goto(0, 0)
    Htot = Y1 + H2
    fig.basemap(region=[0, FULL, 0, Htot], projection=f"X{FULL}c/{Htot}c", frame="+n")
    for xx, yy, t in tagxy:
        fig.text(x=xx, y=yy, text=t, font="8p,Helvetica-Bold,black", justify="TL", no_clip=True)
    save(fig, "fig6_short_lived")


def _tick(v):
    for t in (5, 10, 20, 50, 100, 200, 500, 1000):
        if v / t <= 6:
            return t
    return 2000


def main(argv=None):
    global MAG, OUT, CL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mag_paths.add_argument(ap, "v4.1")
    ap.add_argument("--out", type=Path, default=None, help="figure folder (default paper_sciadv/figures)")
    ap.add_argument("--clusters", type=Path, default=None,
                    help="clustering folder for Figs. 5 and 6 (v4_clusters_zbz.py --out; default data/catalog_v4/clusters_zbz)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("overview"); sub.add_parser("faults"); sub.add_parser("blanco"); sub.add_parser("clusters"); sub.add_parser("short"); sub.add_parser("all")
    z = sub.add_parser("zoom"); z.add_argument("region", choices=list(REGIONS)); z.add_argument("--width", type=float, default=TWO3)
    z.add_argument("--color", choices=["fault", "depth", "time"], default="fault")
    p = sub.add_parser("perspective"); p.add_argument("region", choices=list(REGIONS)); p.add_argument("--width", type=float, default=TWO3)
    p.add_argument("--color", choices=["fault", "depth"], default="fault")
    for k in ("az", "el", "zmax", "exag"):
        p.add_argument(f"--{k}", type=float, default=None)
    a = ap.parse_args(argv)
    MAG = mag_paths.v41(a.mag)
    OUT = a.out or OUT
    CL = a.clusters or CL
    print(f"magnitudes {MAG}; clusters {CL}; figures -> {OUT}")
    if a.cmd in ("overview", "all"):
        fig_overview(a)
    if a.cmd in ("faults", "all"):
        fig_faults(a)
    if a.cmd in ("blanco", "all"):
        fig_blanco(a)
    if a.cmd in ("clusters", "all"):
        fig_clusters(a)
    if a.cmd in ("short", "all"):
        fig_short(a)
    if a.cmd == "zoom":
        fig_zoom(a)
    if a.cmd == "perspective":
        fig_perspective(a)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Paper figures from the catalog v4 layers (the same inputs as the 3D atlas), drawn with PyGMT.

Vector PDF + 600 dpi PNG at print size (Science Advances: 18 cm full width, 12 cm two thirds).

    python workflow/08_figures/v4_figures.py overview              # Fig. 1: setting, catalog, stations, depth section, array in time
    python workflow/08_figures/v4_figures.py faults                # Fig. 3: fault surfaces v0, map + two 3D views
    python workflow/08_figures/v4_figures.py zoom blanco --color fault     # any region of REGIONS, colour by fault/depth/time
    python workflow/08_figures/v4_figures.py perspective mendocino --az 210 --el 28
    python workflow/08_figures/v4_figures.py all

Layers: catalog v4 (data/catalog_v4), Route A magnitudes (data/magnitude_v4), fault surfaces v0 (data/faults/v0),
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
    pygmt.config(FONT_ANNOT_PRIMARY="6p,Helvetica,black", FONT_LABEL="6.5p,Helvetica,black", FONT_TAG="8p,Helvetica-Bold,black",
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
    m = pd.read_csv(D / "magnitude_v4" / "cascadia_catalog_M_routeA.csv", usecols=["event_id", "M"]).set_index("event_id").M
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
    cur = [0.0, 0.0]
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("overview"); sub.add_parser("faults"); sub.add_parser("all")
    z = sub.add_parser("zoom"); z.add_argument("region", choices=list(REGIONS)); z.add_argument("--width", type=float, default=TWO3)
    z.add_argument("--color", choices=["fault", "depth", "time"], default="fault")
    p = sub.add_parser("perspective"); p.add_argument("region", choices=list(REGIONS)); p.add_argument("--width", type=float, default=TWO3)
    p.add_argument("--color", choices=["fault", "depth"], default="fault")
    for k in ("az", "el", "zmax", "exag"):
        p.add_argument(f"--{k}", type=float, default=None)
    a = ap.parse_args(argv)
    if a.cmd in ("overview", "all"):
        fig_overview(a)
    if a.cmd in ("faults", "all"):
        fig_faults(a)
    if a.cmd == "zoom":
        fig_zoom(a)
    if a.cmd == "perspective":
        fig_perspective(a)


if __name__ == "__main__":
    main()

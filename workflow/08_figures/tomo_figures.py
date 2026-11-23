#!/usr/bin/env python3
"""Fig. 7: tomoDD-SE velocity model of one sub-region (depth slices, cross-sections, resolution), drawn with PyGMT.

One composite at print size (18 cm wide), house style of v4_figures.py:
  rows    = 3 (or 4) node depths with the best ray coverage (or --depths)
  columns = dVp (%), dVs (%), Vp/Vs, resolution (spike/checkerboard recovery at the same depth, if given)
  bottom  = 2 vertical sections (dVp by default) with the events of a swath, Slab2 top and the seafloor.
Perturbations are relative to the starting model of the run (run_dir/MOD, the copy tomoDD read). Areas where
the derivative weight sum (DWS) of the last iteration is below --dws are greyed out.

    python workflow/08_figures/tomo_figures.py RUN_DIR [--region NAME] [--dws 50] [--depths 12 21 29]
        [--resolution DIR] [--section LON1 LAT1 LON2 LAT2 ...] [--final] [--out PATH_WITHOUT_EXT]

    # test on the preliminary Mendocino-Gorda L-curve run, with the smoothing spike test as resolution panel
    python workflow/08_figures/tomo_figures.py /wd1/mdenolle_data/tomo_v1/b2_tomodd/mendocino_gorda/d1000_s20 \
        --resolution /wd1/mdenolle_data/tomo_v1/b2_tomodd/mendocino_gorda/smooth_test

tomoDD-SE files read (formats checked in the source, tomoDD-SE.f ~l.2505-2540 and ~l.2905-2920):
  RUN_DIR/MOD                    starting model, simul2000 style: "bld nx ny nz", lon nodes, lat nodes, depth nodes
                                 (km, + down), then Vp and Vp/Vs, each nz blocks of ny rows of nx values. All nodes
                                 including the boundary nodes; interior = [1:-1] on every axis.
  RUN_DIR/Output_Files/Vp_model.dat, Vs_model.dat   final model, same node order (nz x ny rows of nx values).
  RUN_DIR/Output_Files/tomoDD.vel                   per-iteration log; the LAST "DWS for P-wave at iteration" and
                                 "DWS for S-wave at iteration" blocks (nz x ny rows of nx f10.1 values) give the DWS.
  RUN_DIR/Output_Files/tomoDD.reloc(_geo)           relocated events: id lat lon depth(km) ...
  RUN_DIR/Input_Files -> <region>/inputs           station_codes.csv (geographic station coordinates), events.csv,
                                 forward.json (D-B028 P-2: if coords = ellipsoid, the MOD and tomoDD outputs carry
                                 lat' = c + (M/R)(lat - c), lon' = c + (N/R)(lon - c); mapped back here, as in
                                 b2_tomodd_backtransform.py). The model grid is lon/lat/depth: no Cartesian rotation
                                 is involved (tomoDD's x/y/z columns are only used internally).
Resolution DIR (optional): sub-runs <tag>/inv (a tomoDD run on synthetics) with <tag>/true_dvp.npy (nz, ny, nx) and a
null run null/inv; recovered dVp = Vp(tag) / Vp(null) - 1 (b2_smooth_spike_test.py layout; a checkerboard run in the
same layout works unchanged).

Run from the repo root with the default env:
    LD_LIBRARY_PATH=.pixi/envs/default/lib .pixi/envs/default/bin/python workflow/08_figures/tomo_figures.py RUN_DIR
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pygmt
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v4_figures as v4f  # noqa: E402

FIGDIR = Path("/wd1/mdenolle_data/tomo_v1_figs")       # scratch default; the paper copy goes to v4f.OUT with --out
SLAB2 = [v4f.D / "slab2" / "cas_slab2_dep.xyz",
         Path("/home/mdenolle/cascadia_obs_ensemble/data/slab2/cas_slab2_dep.xyz")]

# Per sub-region (D-B025) defaults: map extent and the two sections (lon1, lat1, lon2, lat2, label).
# A region without an entry gets the event box of the run inputs and two sections through its centre.
TOMO_REGIONS = {
    "mendocino_gorda": dict(
        map=[-127.5, -122.5, 39.5, 43.1],
        sections=[(-124.9, 39.8, -124.9, 42.2, "P"),       # N-S across the Mendocino transform, Gorda plate near the trench
                  (-127.2, 41.3, -122.9, 41.3, "Q")]),     # W-E across the Gorda plate and the subduction zone
}
TOMO_REGIONS["mendocino_gorda_v2"] = TOMO_REGIONS["mendocino_gorda"]


# ------------------------------------------------------------------ readers
def _inputs_dir(run):
    """The region's inputs folder (Input_Files holds symlinks into it)."""
    f = run / "Input_Files" / "event.dat"
    if f.is_symlink() or f.exists():
        return f.resolve().parent
    return run.parent / "inputs"


def _forward(run):
    f = _inputs_dir(run) / "forward.json"
    return json.loads(f.read_text()) if f.exists() else {"coords": "sphere"}


def to_geo(lat, lon, fw):
    """Inverse of the D-B028 P-2 coordinate rescaling (identity for coords = sphere)."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    if fw.get("coords", "sphere") != "ellipsoid":
        return lat, lon
    clat, clon = fw["center"]
    return clat + (lat - clat) / fw["lat_scale_M_over_R"], clon + (lon - clon) / fw["lon_scale_N_over_R"]


def read_mod(path):
    tok = Path(path).read_text().split()
    nx, ny, nz = int(tok[1]), int(tok[2]), int(tok[3])
    i = 4
    lon = np.array(tok[i:i + nx], float); i += nx
    lat = np.array(tok[i:i + ny], float); i += ny
    z = np.array(tok[i:i + nz], float); i += nz
    n = nx * ny * nz
    vp = np.array(tok[i:i + n], float).reshape(nz, ny, nx); i += n
    r = np.array(tok[i:i + n], float).reshape(nz, ny, nx)
    return dict(lon=lon, lat=lat, z=z, vp=vp, vs=vp / r)


def read_dws(vel, shape):
    """Last DWS blocks (P, S) of tomoDD.vel, fixed-width f10.1 (overflow '**********' -> inf)."""
    nz, ny, nx = shape
    lines = Path(vel).read_text(errors="replace").split("\n")
    out = []
    for key in ("DWS for P-wave at iteration", "DWS for S-wave at iteration"):
        i0 = max(i for i, l in enumerate(lines) if key in l)
        rows = lines[i0 + 1: i0 + 1 + ny * nz]
        a = np.array([[np.inf if "*" in r[10 * k:10 * k + 10] else float(r[10 * k:10 * k + 10]) for k in range(nx)]
                      for r in rows])
        out.append(a.reshape(nz, ny, nx))
    return out


def load_run(run):
    """Model, start model, DWS, events and stations of one run, all in geographic coordinates."""
    run = Path(run)
    fw = _forward(run)
    m0 = read_mod(run / "MOD")
    shape = m0["vp"].shape
    o = run / "Output_Files"
    vp = np.loadtxt(o / "Vp_model.dat").reshape(shape)
    vs = np.loadtxt(o / "Vs_model.dat").reshape(shape)
    c = fw.get("center", [0.0, 0.0])
    lat = to_geo(m0["lat"], np.full_like(m0["lat"], c[1]), fw)[0]     # separable map
    lon = to_geo(np.full_like(m0["lon"], c[0]), m0["lon"], fw)[1]
    dws_p, dws_s = read_dws(o / "tomoDD.vel", shape)
    f = o / "tomoDD.reloc_geo"
    geo = f.exists()
    if not geo:
        f = o / "tomoDD.reloc"
    ev = pd.read_csv(f, sep=r"\s+", header=None, usecols=[0, 1, 2, 3], names=["id", "lat", "lon", "dep"])
    if not geo:
        ev["lat"], ev["lon"] = to_geo(ev.lat, ev.lon, fw)
    inp = _inputs_dir(run)
    st = pd.read_csv(inp / "station_codes.csv").rename(columns={"sta_lat": "lat", "sta_lon": "lon"}) \
        if (inp / "station_codes.csv").exists() else pd.DataFrame(columns=["lat", "lon", "is_obs"])
    txt = (run / "tomoDD-SE.inp").read_text() if (run / "tomoDD-SE.inp").exists() else ""
    damp = re.findall(r"^\s+\d+\s+-9.*?\s(\d+(?:\.\d*)?)\s+1\s+[\d.]+\s*$", txt, flags=re.M)
    return dict(lon=lon, lat=lat, z=m0["z"], vp=vp, vs=vs, vp0=m0["vp"], vs0=m0["vs"],
                dvp=100 * (vp / m0["vp"] - 1), dvs=100 * (vs / m0["vs"] - 1), vpvs=vp / vs,
                dws_p=dws_p, dws_s=dws_s, ev=ev, st=st, fw=fw, inputs=inp,
                damp=float(damp[-1]) if damp else np.nan)


def load_resolution(rdir, M):
    """Spike/checkerboard sub-runs of a resolution test folder -> list of (k_layers, recovered %, true %)."""
    rdir = Path(rdir)
    null = rdir / "null" / "inv"
    if not (null / "Output_Files" / "Vp_model.dat").exists():
        return []
    shape = M["vp"].shape
    vnull = np.loadtxt(null / "Output_Files" / "Vp_model.dat").reshape(shape)
    out = []
    for d in sorted(rdir.glob("*/true_dvp.npy")):
        f = d.parent / "inv" / "Output_Files" / "Vp_model.dat"
        if not f.exists():
            continue
        true = 100 * np.load(d)
        rec = 100 * (np.loadtxt(f).reshape(shape) / vnull - 1)
        ks = sorted(set(np.nonzero(np.abs(true) > 0)[0]))
        out.append(dict(ks=ks, rec=rec, true=true, tag=d.parent.name))
    txt = (null / "tomoDD-SE.inp").read_text() if (null / "tomoDD-SE.inp").exists() else ""
    dm = re.findall(r"^\s+\d+\s+-9.*?\s(\d+(?:\.\d*)?)\s+1\s+[\d.]+\s*$", txt, flags=re.M)
    for r in out:
        r["damp"] = float(dm[-1]) if dm else np.nan
    return out


def slab2_grid():
    for f in SLAB2:
        if f.exists():
            s = pd.read_csv(f, header=None, names=["lon", "lat", "z"])
            s["lon"] = np.where(s.lon > 180, s.lon - 360, s.lon)
            g = s.pivot_table(index="lat", columns="lon", values="z", dropna=False)
            return xr.DataArray(-g.to_numpy(), coords={"lat": g.index.to_numpy(), "lon": g.columns.to_numpy()},
                                dims=("lat", "lon"), name="z")         # depth, km, + down
    return None


# ------------------------------------------------------------------ helpers
def _interp(M, field, dws, region, dws_min, step=0.025):
    """Node layer -> regular map grid (bilinear, as tomoDD interpolates), NaN where the DWS is below dws_min."""
    W, E, S, N = region
    lon, lat = np.arange(W, E + step / 2, step), np.arange(S, N + step / 2, step)
    da = xr.DataArray(field, coords={"lat": M["lat"], "lon": M["lon"]}, dims=("lat", "lon"))
    v = da.interp(lat=lat, lon=lon).values
    if dws is not None:
        w = xr.DataArray(dws, coords=da.coords, dims=da.dims).interp(lat=lat, lon=lon).values
        v = np.where(w >= dws_min, v, np.nan)
    return xr.DataArray(v.astype("float32"), coords={"lat": lat, "lon": lon}, dims=("lat", "lon"), name="v")


def coverage(M, region, dws_min):
    """Fraction of nodes inside the map region with DWS_P >= dws_min, per depth node."""
    W, E, S, N = region
    ii = (M["lon"] >= W) & (M["lon"] <= E)
    jj = (M["lat"] >= S) & (M["lat"] <= N)
    return np.array([(M["dws_p"][k][np.ix_(jj, ii)] >= dws_min).mean() for k in range(len(M["z"]))])


def pick_depths(M, region, dws_min, n=3):
    """n node depths spread over the layers whose coverage is >= 40% of the best layer (interior nodes only)."""
    cov = coverage(M, region, dws_min)
    cov[0] = cov[-1] = 0
    cand = np.nonzero(cov >= 0.4 * cov.max())[0]
    idx = cand[np.round(np.linspace(0, len(cand) - 1, n)).astype(int)]
    return [float(M["z"][k]) for k in sorted(set(idx))], cov


def _nice_lim(v, lo=2.0, hi=10.0):
    v = np.clip(v, lo, hi)
    for s in (2, 3, 4, 5, 6, 8, 10):
        if s >= v:
            return float(s)
    return hi


def section_sample(M, keys, line, zmax, dws_min, dx=1.0, dz=0.5):
    """Sample fields along a straight lon/lat line (trilinear). Returns distance (km), depth, dict of grids, local frame."""
    lo1, la1, lo2, la2 = line[:4]
    lat0 = np.radians((la1 + la2) / 2)
    kx, ky = 111.195 * np.cos(lat0), 111.195
    L = np.hypot((lo2 - lo1) * kx, (la2 - la1) * ky)
    s = np.arange(0, L + dx / 2, dx)
    lon, lat = lo1 + (lo2 - lo1) * s / L, la1 + (la2 - la1) * s / L
    z = np.arange(-2.0, zmax + dz / 2, dz)
    Z, Sg = np.meshgrid(z, s, indexing="ij")
    pts = np.c_[Z.ravel(), np.interp(Sg.ravel(), s, lat), np.interp(Sg.ravel(), s, lon)]
    out = {}
    for k in keys + ["dws_p", "dws_s"]:
        f = RegularGridInterpolator((M["z"], M["lat"], M["lon"]), M[k], bounds_error=False, fill_value=np.nan)
        out[k] = f(pts).reshape(Z.shape)
    for k in keys:
        w = out["dws_s"] if k == "dvs" else np.minimum(out["dws_p"], out["dws_s"]) if k == "vpvs" else out["dws_p"]
        out[k] = np.where(w >= dws_min, out[k], np.nan)
    frame = dict(lo1=lo1, la1=la1, kx=kx, ky=ky, ux=(lo2 - lo1) * kx / L, uy=(la2 - la1) * ky / L, L=L, lon=lon, lat=lat)
    return s, z, out, frame


def project(frame, lon, lat):
    x, y = (np.asarray(lon) - frame["lo1"]) * frame["kx"], (np.asarray(lat) - frame["la1"]) * frame["ky"]
    return x * frame["ux"] + y * frame["uy"], -x * frame["uy"] + y * frame["ux"]   # along, across (km)


def _colorbar(fig, cmap, pos, length, frame, annot=5.5, label=6.0):
    """Horizontal colourbar with true font sizes. GMT 6.6 scales colourbar fonts, ticks and offsets by about
    sqrt(length / 15 cm) even when they are set explicitly (measured: 2.8 cm bar -> 0.45x), so pre-compensate."""
    f = float(np.sqrt(length / 15.0))
    with pygmt.config(FONT_ANNOT_PRIMARY=f"{annot / f:.2f}p,Helvetica,black", FONT_LABEL=f"{label / f:.2f}p,Helvetica,black",
                      MAP_ANNOT_OFFSET_PRIMARY=f"{1.5 / f:.2f}p", MAP_LABEL_OFFSET=f"{2.5 / f:.2f}p",
                      MAP_TICK_LENGTH_PRIMARY=f"{2.0 / f:.2f}p", MAP_FRAME_PEN=f"{0.4 / f:.2f}p,gray20",
                      MAP_TICK_PEN_PRIMARY=f"{0.4 / f:.2f}p,gray20"):
        fig.colorbar(cmap=cmap, position=f"{pos}+w{length:.3f}c/0.12c+h", frame=frame)


# ------------------------------------------------------------------ figure
def fig_tomography(run_dir, region_name=None, dws_min=50.0, depths=None, n_depths=3, slab_half_km=2.5,
                   swath_km=10.0, sections=None, section_field="dvp", zmax=50.0, resolution=None,
                   map_region=None, vlim=None, preliminary=True, out=None):
    """Draw Fig. 7 for one tomoDD-SE run. Returns the output path stem.

    run_dir      tomoDD run folder (MOD, tomoDD-SE.inp, Output_Files, Input_Files)
    region_name  key of TOMO_REGIONS (map extent, default sections); default: the run folder's parent name
    dws_min      DWS threshold below which the model is greyed out (P DWS for dVp, S for dVs, min of both for Vp/Vs)
    depths       node depths (km) for the slices; default: n_depths depths with the best coverage (pick_depths)
    sections     list of (lon1, lat1, lon2, lat2, label); section_field: 'dvp', 'dvs' or 'vpvs' (one per section or one for all)
    resolution   folder of a spike/checkerboard test (load_resolution); None draws a labelled placeholder column
    vlim         (dVp, dVs) colour limits in %; default from the 98th percentile of the resolved slice values
    preliminary  stamp "PRELIMINARY (run <region>/<tag>)" on the figure
    """
    run = Path(run_dir)
    region_name = region_name or run.parent.name
    tag = f"{run.parent.name}/{run.name}"
    M = load_run(run)
    R = TOMO_REGIONS.get(region_name, {})
    if map_region is None:
        map_region = R.get("map")
    if map_region is None:
        e = pd.read_csv(M["inputs"] / "events.csv")
        map_region = [np.floor(e.lon.min() * 10) / 10, np.ceil(e.lon.max() * 10) / 10,
                      np.floor(e.lat.min() * 10) / 10, np.ceil(e.lat.max() * 10) / 10]
    W, E, S, N = map_region
    if sections is None:
        sections = R.get("sections") or [(W + 0.3, (S + N) / 2, E - 0.3, (S + N) / 2, "P"),
                                         ((W + E) / 2, S + 0.2, (W + E) / 2, N - 0.2, "Q")]
    fields = [section_field] * len(sections) if isinstance(section_field, str) else list(section_field)
    auto, cov = pick_depths(M, map_region, dws_min, n_depths)
    depths = depths or auto
    ks = [int(np.argmin(np.abs(M["z"] - d))) for d in depths]
    res = load_resolution(resolution, M) if resolution else []
    slab = slab2_grid()
    tmp = tempfile.mkdtemp()
    v4f.config()

    # colour scales
    vals = {k: [] for k in ("dvp", "dvs", "vpvs")}
    for k in ks:
        for key, dws in (("dvp", M["dws_p"][k]), ("dvs", M["dws_s"][k]), ("vpvs", np.minimum(M["dws_p"][k], M["dws_s"][k]))):
            g = _interp(M, M[key][k], dws, map_region, dws_min).values
            vals[key].append(g[np.isfinite(g)])
    vals = {k: np.concatenate(v) if v else np.array([0.0]) for k, v in vals.items()}
    lp = vlim[0] if vlim else _nice_lim(np.percentile(np.abs(vals["dvp"]), 98) if vals["dvp"].size else 4)
    ls = vlim[1] if vlim else _nice_lim(np.percentile(np.abs(vals["dvs"]), 98) if vals["dvs"].size else 4)
    rlo, rhi = (np.percentile(vals["vpvs"], [2, 98]) if vals["vpvs"].size > 10 else (1.6, 2.0))
    rlo, rhi = np.floor(rlo * 20) / 20, max(np.ceil(rhi * 20) / 20, np.floor(rlo * 20) / 20 + 0.1)
    cpt = {"dvp": f"{tmp}/dvp.cpt", "dvs": f"{tmp}/dvs.cpt", "vpvs": f"{tmp}/vpvs.cpt", "res": f"{tmp}/res.cpt"}
    pygmt.makecpt(cmap="roma", series=[-lp, lp], output=cpt["dvp"], background=True)
    pygmt.makecpt(cmap="roma", series=[-ls, ls], output=cpt["dvs"], background=True)
    pygmt.makecpt(cmap="batlow", series=[rlo, rhi], output=cpt["vpvs"], background=True)
    amp = max((np.abs(r["true"]).max() for r in res), default=5.0)
    pygmt.makecpt(cmap="vik", series=[-amp, amp], output=cpt["res"], background=True, reverse=True)
    lab = {"dvp": "@~d@~V@-P@- (%)", "dvs": "@~d@~V@-S@- (%)", "vpvs": "V@-P@-/V@-S@-", "res": "recovered @~d@~V@-P@- (%)"}

    # layout (cm)
    ncol, x0, gap = 4, 0.95, 0.18
    mw = (v4f.FULL - x0 - 0.35 - (ncol - 1) * gap) / ncol
    mh = v4f.map_height(map_region, mw)
    proj = f"M{mw:.3f}c"
    sec_h, sec_y = 3.0, 1.15
    cb_y = sec_y + sec_h + 1.25                       # colourbars of the map columns
    top_y = cb_y + 0.85 + len(ks) * mh + (len(ks) - 1) * 0.12
    fig = pygmt.Figure()
    cur = [0.0, 0.0]

    def goto(x, y):
        fig.shift_origin(xshift=f"{x - cur[0]:.3f}c", yshift=f"{y - cur[1]:.3f}c"); cur[:] = [x, y]

    step = 1 if (E - W) > 3 else 0.5
    letters = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for r, k in enumerate(ks):
        y = top_y - (r + 1) * mh - r * 0.12
        z = float(M["z"][k])
        sl = M["ev"][(M["ev"].dep - z).abs() <= slab_half_km]
        sl = sl[sl.lon.between(W, E) & sl.lat.between(S, N)]
        for c in range(ncol):
            goto(x0 + c * (mw + gap), y)
            fr = [("W" if c == 0 else "w") + ("S" if r == len(ks) - 1 else "s") + "ne", f"xa{step}f{step / 4}", f"ya{step}f{step / 4}"]
            fig.basemap(region=map_region, projection=proj, frame="+n")
            fig.coast(region=map_region, projection=proj, land="gray88", water="gray78", resolution="h")
            hook = False
            if c < 3:
                key = ("dvp", "dvs", "vpvs")[c]
                dws = (M["dws_p"][k], M["dws_s"][k], np.minimum(M["dws_p"][k], M["dws_s"][k]))[c]
                g = _interp(M, M[key][k], dws, map_region, dws_min)
                fig.grdimage(g, region=map_region, projection=proj, cmap=cpt[key], nan_transparent=True)
            else:
                hit = [x for x in res if k in x["ks"]]
                if not hit and res:
                    near = min(res, key=lambda x: min(abs(M["z"][kk] - z) for kk in x["ks"]))
                    hit = [near] if min(abs(M["z"][kk] - z) for kk in near["ks"]) <= 3.5 else []
                if hit:
                    h = hit[0]
                    kk = min(h["ks"], key=lambda q: abs(M["z"][q] - z))
                    g = _interp(M, h["rec"][kk], None, map_region, dws_min)
                    g = g.where(np.abs(g) >= 0.05 * amp)
                    fig.grdimage(g, region=map_region, projection=proj, cmap=cpt["res"], nan_transparent=True)
                    jj, ii = np.nonzero(np.abs(h["true"][kk]) > 0)
                    fig.plot(x=M["lon"][ii], y=M["lat"][jj], style="c0.16c", pen="0.4p,black")
                    if abs(M["z"][kk] - z) > 0.01:
                        fig.text(position="BR", text=f"test at {M['z'][kk]:g} km", font="5p,Helvetica,black", justify="BR",
                                 offset="-0.08c/0.08c", fill="white@20", clearance="0.03c/0.02c")
                else:
                    hook = True
            fig.coast(region=map_region, projection=proj, shorelines="0.25p,gray20", resolution="h")
            if slab is not None:
                try:
                    fig.grdcontour(slab.sel(lon=slice(W - 1, E + 1), lat=slice(S - 1, N + 1)), levels=f"{z:g},", pen="0.6p,#0b4f8a,4_2",
                                   region=map_region, projection=proj)
                except Exception:
                    pass
            v4f.plate_boundaries(fig, map_region, 0.55)
            if c < 3:
                fig.plot(x=sl.lon, y=sl.lat, style="c0.028c", fill="gray10", transparency=40)
            st = M["st"]
            if len(st) and c == 3:                         # stations only in the resolution / coverage column
                st = st[st.lon.between(W, E) & st.lat.between(S, N)]
                ob = st[st.is_obs.astype(bool)]
                ld = st[~st.is_obs.astype(bool)]
                fig.plot(x=ob.lon, y=ob.lat, style="i0.09c", fill="white", pen="0.25p,gray10")
                fig.plot(x=ld.lon, y=ld.lat, style="t0.09c", fill="white", pen="0.25p,gray10")
            if hook:                                       # no spike/checkerboard output: labelled placeholder
                fig.text(x=(W + E) / 2, y=(S + N) / 2, justify="CM", font="5.5p,Helvetica-Oblique,gray20",
                         text="no resolution test at this depth" if res else "resolution test: hook",
                         fill="white@10", clearance="0.04c/0.03c")
                if not res:
                    fig.text(x=(W + E) / 2, y=(S + N) / 2 - 0.12 * (N - S), justify="CM", font="5p,Helvetica,gray30",
                             text="(spike/checkerboard: --resolution DIR)", fill="white@10", clearance="0.04c/0.03c")
            if c == 0:
                for sec in sections:
                    fig.plot(x=[sec[0], sec[2]], y=[sec[1], sec[3]], pen="0.8p,black")
                    fig.plot(x=[sec[0], sec[2]], y=[sec[1], sec[3]], pen="0.3p,white")
                    if r == 0:
                        for xx, yy, t in ((sec[0], sec[1], sec[4]), (sec[2], sec[3], sec[4] + "'")):
                            fig.text(x=xx, y=yy, text=t, font="6p,Helvetica-Bold,black", fill="white@10", clearance="0.03c/0.02c",
                                     justify="CM", no_clip=True)
            fig.basemap(region=map_region, projection=proj, frame=fr)
            fig.text(position="TR", text=f"{z:g} km", font="6p,Helvetica-Bold,black", justify="TR", offset="-0.08c/-0.08c",
                     fill="white@15", clearance="0.04c/0.03c")
            v4f.tag(fig, next(letters))
            if r == 0:
                ttl = [lab["dvp"], lab["dvs"], lab["vpvs"],
                       "resolution" + (f" (spike test, damp {res[0]['damp']:g})" if res else "")][c]
                fig.text(position="TC", text=ttl, font="6.5p,Helvetica-Bold,black", justify="BC", offset="0c/0.12c", no_clip=True)
    # colourbars under the map columns
    goto(x0, cb_y)
    for c, key in enumerate(("dvp", "dvs", "vpvs", "res")):
        if key == "res" and not res:
            continue
        a = {"dvp": f"a{lp / 2:g}f{lp / 4:g}", "dvs": f"a{ls / 2:g}f{ls / 4:g}", "vpvs": "a0.1f0.05", "res": f"a{amp / 2:g}f{amp / 4:g}"}[key]
        _colorbar(fig, cpt[key], f"x{c * (mw + gap) + 0.15 * mw:.3f}c/0c", 0.7 * mw, [f"x{a}+l{lab[key]}"])

    # sections
    sw = (v4f.FULL - x0 - 0.35 - 1.1) / 2
    for i, (sec, key) in enumerate(zip(sections[:2], fields[:2])):
        s, zz, G, frame = section_sample(M, [key], sec, zmax, dws_min)
        g = xr.DataArray(G[key].astype("float32"), coords={"z": zz, "x": s}, dims=("z", "x"))
        reg = [0, float(s[-1]), -2.0, zmax]
        pj = f"X{sw:.3f}c/-{sec_h:.3f}c"
        goto(x0 + i * (sw + 1.1), sec_y)
        fig.basemap(region=reg, projection=pj, frame="+n")
        fig.plot(x=[reg[0], reg[1], reg[1], reg[0]], y=[reg[2], reg[2], reg[3], reg[3]], close=True, fill="gray80")
        fig.grdimage(g, region=reg, projection=pj, cmap=cpt[key], nan_transparent=True)
        try:
            rel = pygmt.datasets.load_earth_relief(resolution="01m", region=[min(sec[0], sec[2]) - 0.1, max(sec[0], sec[2]) + 0.1,
                                                                            min(sec[1], sec[3]) - 0.1, max(sec[1], sec[3]) + 0.1],
                                                    registration="gridline")
            topo = -rel.interp(lon=xr.DataArray(frame["lon"]), lat=xr.DataArray(frame["lat"])).values / 1000.0
            fig.plot(x=np.r_[s, s[::-1]], y=np.r_[topo, np.full_like(topo, -2.0)], close=True, fill="white")
            fig.plot(x=s, y=topo, pen="0.4p,gray20")
        except Exception:
            pass
        e = M["ev"]
        al, ac = project(frame, e.lon.to_numpy(), e.lat.to_numpy())
        m = (np.abs(ac) <= swath_km) & (al >= 0) & (al <= s[-1])
        fig.plot(x=al[m], y=e.dep.to_numpy()[m], style="c0.03c", fill="gray10", transparency=40)
        if slab is not None:
            sz = slab.interp(lon=xr.DataArray(frame["lon"]), lat=xr.DataArray(frame["lat"])).values
            if np.isfinite(sz).any():
                fig.plot(x=s, y=sz, pen="0.8p,#0b4f8a,4_2")
                j = np.nanargmax(np.where(np.isfinite(sz), s, -1))
                fig.text(x=s[j], y=sz[j], text="Slab2", font="5p,Helvetica-Bold,#0b4f8a", justify="BR", offset="-0.05c/0.06c")
        ve = (sec_h / (zmax + 2)) / (sw / s[-1])
        fig.basemap(region=reg, projection=pj, frame=["WSne", "xa50f10+ldistance (km)", "ya10f5+ldepth (km)"])
        fig.text(position="TL", text=sec[4], font="6.5p,Helvetica-Bold,black", justify="BL", offset="0.35c/0.06c", no_clip=True)
        fig.text(position="TR", text=sec[4] + "'", font="6.5p,Helvetica-Bold,black", justify="BR", offset="0c/0.06c", no_clip=True)
        fig.text(position="BR", text=f"{lab[key]}, swath @~\\261@~{swath_km:g} km, VE {ve:.1f}", font="5p,Helvetica,black",
                 justify="BR", offset="-0.08c/0.08c", fill="white@20", clearance="0.03c/0.02c")
        fig.text(position="TL", text=next(letters), font="8p,Helvetica-Bold,black", justify="BL", offset="0c/0.04c", no_clip=True)
    # legend line + preliminary stamp
    goto(x0, 0)
    for yy, t in ((0.38, f"Grey: DWS < {dws_min:g} (P DWS for @~d@~V@-P@-, S for @~d@~V@-S@-, both for V@-P@-/V@-S@-). "
                        f"Dots: relocated events within @~\\261@~{slab_half_km:g} km of the slice or in the section swath. "
                        "Dashed blue: Slab2 top (contour at the slice depth on maps)."),
                  (0.08, "Perturbations relative to the starting model of the run. Triangles: land / ocean-bottom (inverted) "
                        "stations used." + (" Circles: spike positions of the resolution test." if res else ""))):
        fig.text(x=0, y=yy, region=[0, 1, 0, 1], projection="X1c", no_clip=True, justify="BL", font="5p,Helvetica,gray20", text=t)
    if preliminary:
        goto(v4f.FULL - 0.35, top_y + 0.75)
        fig.text(x=0, y=0, region=[0, 1, 0, 1], projection="X1c", no_clip=True, justify="TR", font="7p,Helvetica-Bold,red3",
                 text=f"PRELIMINARY (run {tag})")
    out = Path(out) if out else FIGDIR / f"fig7_tomography_{run.parent.name}_{run.name}"
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=600)
    print(f"wrote {out}.pdf/.png  (depths {depths} km, coverage {[round(float(cov[k]), 2) for k in ks]}, "
          f"dVp +/-{lp:g}%, dVs +/-{ls:g}%, Vp/Vs {rlo:.2f}-{rhi:.2f}, resolution runs {len(res)})")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--region", default=None, help="key of TOMO_REGIONS (default: parent folder name of RUN_DIR)")
    ap.add_argument("--dws", type=float, default=50.0, help="DWS threshold for the grey mask (default 50)")
    ap.add_argument("--depths", type=float, nargs="+", default=None, help="slice depths (node depths, km)")
    ap.add_argument("--n-depths", type=int, default=3)
    ap.add_argument("--slab-half", type=float, default=2.5, help="events within +/- this many km of a slice")
    ap.add_argument("--swath", type=float, default=10.0, help="section swath half-width (km)")
    ap.add_argument("--section", type=float, nargs=4, action="append", metavar=("LON1", "LAT1", "LON2", "LAT2"))
    ap.add_argument("--section-field", nargs="+", default=["dvp"], choices=["dvp", "dvs", "vpvs"])
    ap.add_argument("--zmax", type=float, default=50.0)
    ap.add_argument("--resolution", default=None, help="spike/checkerboard test folder (null/inv + <tag>/inv + true_dvp.npy)")
    ap.add_argument("--map", type=float, nargs=4, default=None, metavar=("W", "E", "S", "N"))
    ap.add_argument("--vlim", type=float, nargs=2, default=None, metavar=("DVP", "DVS"))
    ap.add_argument("--final", action="store_true", help="drop the PRELIMINARY stamp")
    ap.add_argument("--out", default=None, help="output path without extension")
    a = ap.parse_args(argv)
    secs = [(*s, l) for s, l in zip(a.section, "PQRS")] if a.section else None
    fig_tomography(a.run_dir, a.region, a.dws, a.depths, a.n_depths, a.slab_half, a.swath, secs,
                   a.section_field[0] if len(a.section_field) == 1 else a.section_field, a.zmax, a.resolution,
                   a.map, a.vlim, not a.final, a.out)


if __name__ == "__main__":
    main()

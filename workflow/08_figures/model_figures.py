#!/usr/bin/env python3
"""Figures of a cascadia_obs_tomo velocity model (Track B): exploration gallery and paper-figure candidates.

    python workflow/08_figures/model_figures.py gallery          # one self-contained HTML, maps / sections / resolution
    python workflow/08_figures/model_figures.py paper            # Fig M1, M2, S-M3, S-M4 candidates + review HTML
    python workflow/08_figures/model_figures.py all
    options: --model PATH (default v0.4.2.nc)  --catalog PATH (v5.1; v6.0 when registered)  --tiers ABC
             --mask either|checker|dws  --dws-min 100  --out DIR  --jobs 4

Inputs (all read-only): the model netCDF (registry store, EMC-style variables: medium, Vp, Vs, VpVs,
Vp_perturbation, Vs_perturbation, dws_p/s, resolution_p/s, tile_weight_p/s, vs_reverted, shallow_crust_node),
a hypocentre catalogue with columns lat, lon, depth (km, + down), tier; the ver5/ver6 station table; Slab2
(data/slab2/cas_slab2_dep.xyz, utils/fetch_slab2.py); PB2002 plate boundaries (data/tectonics); GMT coastlines
and earth_relief 01m; prior Vs models Delph et al. (2018) and CRESCENT CVM Gen0 (data/tomography,
utils/fetch_tomography.py). A missing optional input is reported and skipped.

What is drawn (shown cells):
  rock nodes only (medium = 0: nothing above the seafloor or the ground),
  and resolved or well sampled: checkerboard-resolved (resolution_p/s > 0, D-B018 masks) OR DWS >= --dws-min
  (--mask either, default; 'checker' = checkerboard only, 'dws' = DWS only), P for Vp, S for Vs, both for Vp/Vs,
  and data-derived: tile_weight > 0 (not the M0 reference), for Vs also vs_reverted = 0 (D-B051) and not a
  D-B073 shallow-crust node (Vs set by rule, not by data).
Values between nodes: multilinear interpolation over the shown nodes only (weights renormalised), drawn only
where the NEAREST node is shown, so the drawn footprint is exactly the set of shown node cells and no value is
made from a masked node. Checkerboard-resolved cells are outlined (black line); the rest pass on DWS only.

Colour maps: 'seismic' (matplotlib, PI's atlas choice) and Crameri 'vik' (perceptually uniform, GMT SCM CPT);
velocities reversed (slow = red, fast = blue), Vp/Vs as is (high = red).

Run from the repo root (or any worktree) with the default env, niced, at most --jobs processes:
    LD_LIBRARY_PATH=<repo>/.pixi/envs/default/lib nice -n 10 <repo>/.pixi/envs/default/bin/python \
        workflow/08_figures/model_figures.py gallery
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one core per process (GMT is OpenMP)

import argparse  # noqa: E402
import base64  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import itertools  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from datetime import datetime  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, Normalize  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MAIN = Path("/home/mdenolle/cascadia_obs_ensemble")           # git-ignored data live in the main tree (read only)
MODEL = Path("/wd1/mdenolle_data/tomo_v1/models/cascadia_obs_tomo/v0.4.2.nc")
CATALOG = Path("/wd1/mdenolle_data/catalog_v5/v5.1/cascadia_catalog_v5.1.csv")
STATIONS = Path("/wd1/mdenolle_data/picks_share/stations/stations_ver5_v6.csv")
SLAB2 = [ROOT / "data/slab2/cas_slab2_dep.xyz", MAIN / "data/slab2/cas_slab2_dep.xyz"]
PRIORS = {"Delph et al. (2018)": [ROOT / "data/tomography/Delph2018.nc", MAIN / "data/tomography/Delph2018.nc"],
          "CRESCENT CVM Gen0": [ROOT / "data/tomography/CRESCENT_Gen0.nc", MAIN / "data/tomography/CRESCENT_Gen0.nc"]}
PB2002 = ROOT / "data/tectonics/PB2002_steps.json"
GMT_SHARE = Path(sys.executable).resolve().parents[1] / "share" / "gmt"
KM = 111.195
REGION = [-131.75, -120.125, 38.9, 50.6]
COL1, COL2 = 8.6, 17.8                                         # Science Advances widths (cm)
CM = 1 / 2.54

# margin-normal profiles (lon1, lat1, lon2, lat2) west -> east, and margin-parallel polylines south -> north
PROFILES = {
    "A": ("Mendocino 40.4N", [(-127.6, 40.4), (-121.0, 40.4)]),
    "B": ("Gorda 41.4N", [(-127.9, 41.4), (-121.0, 41.4)]),
    "C": ("Blanco / S. Oregon 43.3N", [(-130.2, 43.3), (-121.0, 43.3)]),
    "D": ("Central Oregon 44.6N", [(-130.6, 44.6), (-120.6, 44.6)]),
    "E": ("N. Oregon 45.6N", [(-130.6, 45.6), (-120.6, 45.6)]),
    "F": ("S. Washington 46.4N", [(-129.6, 46.4), (-120.6, 46.4)]),
    "G": ("Washington 47.3N", [(-129.6, 47.3), (-120.6, 47.3)]),
    "H": ("Olympics / Juan de Fuca 48.2N", [(-129.0, 48.2), (-120.6, 48.2)]),
    "I": ("Vancouver Island (N55E)", [(-129.0, 48.1), (-123.6, 50.5)]),
    "J": ("Forearc (Coast Ranges, margin-parallel)", [(-124.0, 40.2), (-123.9, 42.0), (-123.7, 44.0), (-123.6, 46.0),
                                                      (-123.6, 47.5), (-124.3, 48.4), (-125.6, 49.4), (-126.6, 50.1)]),
    "K": ("Arc (volcanic front, margin-parallel)", [(-121.5, 40.5), (-122.2, 41.4), (-122.1, 42.9), (-121.8, 44.1),
                                                    (-121.7, 45.4), (-121.8, 46.5), (-121.6, 47.5), (-121.5, 48.4),
                                                    (-121.8, 48.8), (-122.9, 49.8)]),
}

QTY = {   # key: (variable, label, unit, colour-map direction: True = reversed (slow/low = red))
    "vp": ("Vp", "Vp", "km/s", True), "vs": ("Vs", "Vs", "km/s", True), "vpvs": ("VpVs", "Vp/Vs", "", False),
    "dvp": ("Vp_perturbation", "dVp", "% from start", True), "dvs": ("Vs_perturbation", "dVs", "% from start", True),
}
CONTOUR = {"vp": 0.25, "vs": 0.15, "vpvs": 0.05, "dvp": 2.0, "dvs": 2.0}


# ------------------------------------------------------------------ style
def style(fs=6.0):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": fs, "axes.titlesize": fs + 0.5, "axes.labelsize": fs, "xtick.labelsize": fs - 0.5,
        "ytick.labelsize": fs - 0.5, "legend.fontsize": fs - 0.5, "axes.linewidth": 0.5, "xtick.major.width": 0.4,
        "ytick.major.width": 0.4, "xtick.major.size": 2, "ytick.major.size": 2, "xtick.direction": "out",
        "ytick.direction": "out", "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
        "axes.titlepad": 3, "axes.labelpad": 2})


def gmt_cmap(name, reverse=False):
    """A GMT Scientific Colour Map (Crameri) CPT -> matplotlib colormap (RGB 0-255 per slice)."""
    rows = []
    for line in (GMT_SHARE / "cpt" / "SCM" / f"{name}.cpt").read_text().splitlines():
        t = line.split()
        if not t or t[0][0] in "#BFN":
            continue
        rgb = [list(map(float, x.split("/"))) if "/" in x else None for x in t]
        if "/" in t[1]:
            rows.append(np.array(rgb[1]) / 255)
        else:
            rows.append(np.array(list(map(float, t[1:4]))) / 255)
    c = np.array(rows)
    return ListedColormap(c[::-1] if reverse else c, name=f"{name}{'_r' if reverse else ''}")


def cmap_for(key, scheme):
    rev = QTY[key][3]
    if scheme == "seismic":
        return plt.get_cmap("seismic_r" if rev else "seismic")
    return gmt_cmap("vik", reverse=rev)


def save(fig, path, dpi, pdf=False, quantize=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if quantize:
        from PIL import Image
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        im = Image.open(buf).convert("RGB").quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        im.save(path, optimize=True)
    else:
        fig.savefig(path, dpi=dpi)
    if pdf:
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


# ------------------------------------------------------------------ model
class Model:
    """The model on its node grid, with the show masks of every quantity."""

    def __init__(self, path=MODEL, mask="either", dws_min=100.0):
        self.path = Path(path)
        ds = xr.open_dataset(self.path)
        self.ds, self.attrs = ds, dict(ds.attrs)
        self.version = ds.attrs.get("version", self.path.stem)
        self.z, self.lat, self.lon = (ds[c].values.astype(float) for c in ("depth", "latitude", "longitude"))
        g = {v: ds[v].values for v in ds.data_vars}
        self.g = g
        rock = g["medium"] == 0
        self.rock = rock
        chk_p, chk_s = g["resolution_p"] > 0, g["resolution_s"] > 0
        dw_p, dw_s = g["dws_p"] >= dws_min, g["dws_s"] >= dws_min
        sel = {"either": (chk_p | dw_p, chk_s | dw_s), "checker": (chk_p, chk_s), "dws": (dw_p, dw_s)}[mask]
        data_p = rock & (g["tile_weight_p"] > 0)
        data_s = rock & (g["tile_weight_s"] > 0) & (g["vs_reverted"] == 0) & (g["shallow_crust_node"] == 0)
        ok_p, ok_s = sel[0] & data_p, sel[1] & data_s
        self.f = {k: g[v].astype(float) for k, (v, *_) in QTY.items()}
        self.valid = {"vp": ok_p, "dvp": ok_p, "vs": ok_s, "dvs": ok_s, "vpvs": ok_p & ok_s}
        for k in self.valid:
            self.valid[k] &= np.isfinite(self.f[k])
        self.chk = {"vp": chk_p, "dvp": chk_p, "vs": chk_s, "dvs": chk_s, "vpvs": chk_p & chk_s}
        self.mask, self.dws_min = mask, dws_min
        self.surface = g["surface_elevation"].astype(float)           # km, + up
        self.counts = pd.DataFrame({
            "depth_km": self.z, "rock": rock.sum((1, 2)),
            "checker_P": (rock & chk_p).sum((1, 2)), "checker_S": (rock & chk_s).sum((1, 2)),
            f"DWS_P>={dws_min:g}": (rock & dw_p).sum((1, 2)), f"DWS_S>={dws_min:g}": (rock & dw_s).sum((1, 2)),
            "shown_Vp": ok_p.sum((1, 2)), "shown_Vs": ok_s.sum((1, 2)), "shown_VpVs": self.valid["vpvs"].sum((1, 2))})

    def sha256(self):
        h = hashlib.sha256()
        with open(self.path, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()

    def sample(self, key, z, lat, lon, field=None, valid=None):
        """Values at points (arrays of one shape): multilinear over shown nodes, NaN unless the nearest node is shown."""
        field = self.f[key] if field is None else field
        valid = self.valid[key] if valid is None else valid
        return sample_grid(field, valid, (self.z, self.lat, self.lon), (z, lat, lon))

    def nearest(self, arr, z, lat, lon):
        idx = [_axis(c, q)[2] for c, q in zip((self.z, self.lat, self.lon), (z, lat, lon))]
        out = np.any([_axis(c, q)[3] for c, q in zip((self.z, self.lat, self.lon), (z, lat, lon))], axis=0)
        return np.where(out, False if arr.dtype == bool else np.nan, arr[tuple(idx)])


def _axis(c, q):
    q = np.asarray(q, float)
    i = np.clip(np.searchsorted(c, q) - 1, 0, len(c) - 2)
    t = np.clip((q - c[i]) / (c[i + 1] - c[i]), 0.0, 1.0)
    near = np.where(t < 0.5, i, i + 1)
    out = (q < c[0] - 1e-9) | (q > c[-1] + 1e-9)
    return i, t, near, out


def sample_grid(field, valid, axes, pts):
    ax = [_axis(c, q) for c, q in zip(axes, pts)]
    num = np.zeros(np.shape(pts[0])); den = np.zeros(np.shape(pts[0]))
    fz = np.where(valid, field, 0.0)
    for corner in itertools.product((0, 1), repeat=len(axes)):
        ii = tuple(a[0] + c for a, c in zip(ax, corner))
        w = np.prod([a[1] if c else 1 - a[1] for a, c in zip(ax, corner)], axis=0)
        ok = valid[ii]
        num += np.where(ok, w * fz[ii], 0.0)
        den += np.where(ok, w, 0.0)
    foot = valid[tuple(a[2] for a in ax)] & ~np.any([a[3] for a in ax], axis=0) & (den > 1e-12)
    return np.where(foot, num / np.where(den > 1e-12, den, 1.0), np.nan)


# ------------------------------------------------------------------ context layers
class Context:
    def __init__(self, catalog=CATALOG, tiers="ABC", stations=STATIONS, cache=None, need_relief=True):
        self.missing = []
        self.catalog, self.tiers = Path(catalog), tiers
        c = pd.read_csv(self.catalog, usecols=["lat", "lon", "depth", "tier", "region"])
        self.ev = c[c.tier.isin(list(tiers)) & c.depth.notna()].reset_index(drop=True)
        self.n_ev_all = len(c)
        s = pd.read_csv(stations)
        self.st = s[s.in_ver5.astype(str).str.lower().eq("true")].copy()
        self.st["sea"] = self.st.seafloor.astype(str).str.lower().eq("true")
        self.st = self.st.drop_duplicates(["network", "station"])
        self.cache = Path(cache) if cache else None
        self.coast = self._gmt_lines("coast", "-Dh -W")
        self.borders = self._gmt_lines("borders", "-Dh -N1 -N2")
        self.plates = self._plates()
        self.slab = self._slab2()
        self.relief = self._relief() if need_relief else None

    def _gmt_lines(self, name, args):
        f = self.cache / f"{name}_h.txt" if self.cache else None
        if f is None or not f.exists():
            import pygmt
            W, E, S, N = REGION
            f.parent.mkdir(parents=True, exist_ok=True)
            with pygmt.clib.Session() as lib:
                lib.call_module("coast", f"-R{W - 1}/{E + 1}/{S - 1}/{N + 1} {args} -M ->{f}")
        segs, cur = [], []
        for line in f.read_text().splitlines():
            if line.startswith(">"):
                if len(cur) > 1:
                    segs.append(np.array(cur))
                cur = []
            elif line.strip():
                cur.append(list(map(float, line.split()[:2])))
        if len(cur) > 1:
            segs.append(np.array(cur))
        return segs

    def _plates(self):
        try:
            sys.path.insert(0, str(ROOT / "workflow" / "06_analysis"))
            import map_context as mctx
            mctx.PB2002 = str(PB2002)
            return mctx.plate_boundaries([REGION[0] - 1, REGION[1] + 1, REGION[2] - 1, REGION[3] + 1])
        except Exception as e:                                         # noqa: BLE001
            self.missing.append(f"PB2002 plate boundaries ({e})")
            return {"SUB": [], "OSR": [], "TF": []}

    def _slab2(self):
        for f in SLAB2:
            if f.exists():
                s = pd.read_csv(f, header=None, names=["lon", "lat", "z"])
                s["lon"] = np.where(s.lon > 180, s.lon - 360, s.lon)
                g = s.pivot_table(index="lat", columns="lon", values="z", dropna=False)
                self.slab_path = f
                return dict(lat=g.index.to_numpy(), lon=g.columns.to_numpy(), dep=-g.to_numpy())   # km, + down
        self.missing.append("Slab2 (data/slab2/cas_slab2_dep.xyz; utils/fetch_slab2.py)")
        self.slab_path = None
        return None

    def _relief(self):
        f = self.cache / "relief_01m.npz" if self.cache else None
        if f is not None and f.exists():
            d = np.load(f)
            return dict(lon=d["lon"], lat=d["lat"], z=d["z"])
        try:
            import pygmt
            W, E, S, N = REGION
            r = pygmt.datasets.load_earth_relief(resolution="01m", region=[W - 0.5, E + 0.5, S - 0.5, N + 0.5],
                                                 registration="gridline")
            out = dict(lon=r.lon.values, lat=r.lat.values, z=r.values.astype("float32") / 1000.0)   # km, + up
            if f is not None:
                np.savez_compressed(f, **out)
            return out
        except Exception as e:                                         # noqa: BLE001
            self.missing.append(f"earth_relief 01m ({e}); sections use the model surface")
            return None

    def relief_at(self, lon, lat):
        if self.relief is None:
            return None
        R = self.relief
        return sample_grid(R["z"].astype(float), np.isfinite(R["z"]), (R["lat"], R["lon"]), (lat, lon))

    def slab_at(self, lon, lat):
        if self.slab is None:
            return np.full(np.shape(lon), np.nan)
        s = self.slab
        return sample_grid(np.nan_to_num(s["dep"]), np.isfinite(s["dep"]), (s["lat"], s["lon"]), (lat, lon))


# ------------------------------------------------------------------ map drawing
def setup_map(ax, region, ticks=True, step=2.0, left=True, bottom=True):
    W, E, S, N = region
    ax.set_xlim(W, E); ax.set_ylim(S, N)
    ax.set_aspect(1 / np.cos(np.radians((S + N) / 2)))
    xt = np.arange(np.ceil(W / step) * step, E + 1e-6, step)
    yt = np.arange(np.ceil(S / step) * step, N + 1e-6, step)
    ax.set_xticks(xt); ax.set_yticks(yt)
    ax.set_xticklabels([f"{-x:g}°W" for x in xt] if (ticks and bottom) else [])
    ax.set_yticklabels([f"{y:g}°N" for y in yt] if (ticks and left) else [])
    ax.tick_params(length=1.5, pad=1)


def draw_context(ax, C, region, lw=0.35, stations=False, plates=True, st_size=4, teeth_km=45):
    W, E, S, N = region
    ax.add_collection(LineCollection(C.coast, colors="0.1", linewidths=lw, zorder=6))
    ax.add_collection(LineCollection(C.borders, colors="0.45", linewidths=lw * 0.6, linestyles=(0, (3, 2)), zorder=6))
    if plates:
        for key, lwk in (("OSR", 1.6), ("TF", 1.0), ("SUB", 1.0)):
            for L in C.plates[key]:
                ax.plot(L[:, 0], L[:, 1], color="0.1", lw=lw * lwk, zorder=7)
        for L in C.plates["SUB"]:                                       # trench teeth on the upper-plate (east) side
            L = L[np.argsort(-L[:, 1])]
            px, py, ss = profile_points(L, 1.0)
            for j in range(0, len(ss) - 1, max(1, int(teeth_km))):
                ax.plot(px[j] + 0.06, py[j], marker=">", ms=2.0 * lw / 0.35, color="0.1", mew=0, zorder=7)
    if stations:
        st = C.st[C.st.lon.between(W, E) & C.st.lat.between(S, N)]
        for sea, mk in ((False, "^"), (True, "v")):
            s = st[st.sea == sea]
            ax.scatter(s.lon, s.lat, s=st_size, marker=mk, facecolor="white", edgecolor="0.1", linewidths=0.25, zorder=8)


def layer_bounds(z, k):
    lo = -np.inf if k == 0 else 0.5 * (z[k - 1] + z[k])
    hi = np.inf if k == len(z) - 1 else 0.5 * (z[k] + z[k + 1])
    return lo, hi


def map_grid(region, step):
    W, E, S, N = region
    return np.arange(W, E + step / 2, step), np.arange(S, N + step / 2, step)


def map_layer(ax, M, C, key, k, region, cmap, norm, step=0.025, contours=True, events=True, ev_size=0.3,
              chk_outline=True, slab=True, stations=False, ctx_lw=0.35, field=None, valid=None, grey="0.85", st_size=4):
    """One depth slice of quantity `key` at node layer k. Returns the image handle (or None if nothing shown)."""
    lonq, latq = map_grid(region, step)
    LON, LAT = np.meshgrid(lonq, latq)
    Z = np.full(LON.shape, M.z[k])
    G = M.sample(key, Z, LAT, LON, field=field, valid=valid)
    rockg = M.nearest(M.rock, Z, LAT, LON).astype(bool)
    ext = [lonq[0] - step / 2, lonq[-1] + step / 2, latq[0] - step / 2, latq[-1] + step / 2]
    ax.set_facecolor("white")
    grey_rgba = np.zeros(LON.shape + (4,))
    grey_rgba[rockg & ~np.isfinite(G)] = matplotlib.colors.to_rgba(grey)
    ax.imshow(grey_rgba, extent=ext, origin="lower", interpolation="nearest", zorder=1, aspect=ax.get_aspect())
    im = ax.imshow(np.ma.masked_invalid(G), extent=ext, origin="lower", cmap=cmap, norm=norm, interpolation="nearest",
                   zorder=2, aspect=ax.get_aspect())
    if contours and np.isfinite(G).sum() > 4 and key in CONTOUR:
        dv = CONTOUR[key]
        lv = np.arange(np.floor(np.nanmin(G) / dv) * dv, np.nanmax(G) + dv, dv)
        if len(lv) > 1:
            ax.contour(LON, LAT, np.ma.masked_invalid(G), levels=lv, colors="0.15", linewidths=0.2, zorder=3)
    if chk_outline:
        chk = M.chk[key] & (M.valid[key] if valid is None else valid)
        cg = M.nearest(chk.astype(float), Z, LAT, LON)
        if np.nansum(cg) > 0:
            ax.contour(LON, LAT, np.nan_to_num(cg), levels=[0.5], colors="k", linewidths=0.5, zorder=4)
    if slab and C.slab is not None:
        s = C.slab
        try:
            cs = ax.contour(s["lon"], s["lat"], np.ma.masked_invalid(s["dep"]), levels=[M.z[k]], colors="#0b4f8a",
                            linewidths=0.6, linestyles=(0, (4, 2)), zorder=5)
        except Exception:                                              # noqa: BLE001
            pass
    if events:
        lo, hi = layer_bounds(M.z, k)
        e = C.ev[(C.ev.depth >= lo) & (C.ev.depth < hi)]
        ax.scatter(e.lon, e.lat, s=ev_size, c="k", alpha=0.35, linewidths=0, zorder=5, rasterized=True)
    draw_context(ax, C, region, lw=ctx_lw, stations=stations, st_size=st_size)
    return im


def vrange(M, key, ks, sym_about=None, q=98, rnd=0.1, minhalf=0.1):
    vals = np.concatenate([M.f[key][k][M.valid[key][k]] for k in ks]) if ks else np.array([])
    if vals.size < 5:
        return None
    c = np.nanmedian(vals) if sym_about is None else sym_about
    h = max(np.nanpercentile(np.abs(vals - c), q), minhalf)
    h = np.ceil(h / rnd) * rnd
    if sym_about is None:
        c = np.round(c / rnd) * rnd
    return c - h, c + h


QUANT = False
FIXED = {"vpvs": (1.60, 1.90), "dvp": (-8.0, 8.0), "dvs": (-8.0, 8.0)}


def layer_centre(M, key, k):
    """Median over the inverted rock nodes of the layer (tile_weight > 0), shown or not: a regional reference."""
    tw = M.g["tile_weight_p" if key in ("vp", "dvp") else "tile_weight_s"][k] > 0
    v = M.f[key][k][M.rock[k] & tw & np.isfinite(M.f[key][k])]
    return float(np.round(np.median(v), 1)) if v.size else None


def gallery_norm(M, key, k):
    if key in FIXED:
        return Normalize(*FIXED[key])
    c = layer_centre(M, key, k)
    r = vrange(M, key, [k], sym_about=c) if c is not None else None
    return Normalize(*r) if r else Normalize(0, 1)


def colorbar(fig, im, rect, label, ticks=None, orientation="horizontal", fs=None):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(im, cax=cax, orientation=orientation, ticks=ticks)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(length=1.5, pad=1, width=0.4, labelsize=fs)
    cb.set_label(label, labelpad=1, fontsize=fs)
    return cb


# ------------------------------------------------------------------ sections
def profile_points(verts, dx=1.0):
    """Densely sampled polyline (lon, lat) every dx km and the cumulative distance (km)."""
    verts = np.asarray(verts, float)
    pts = [verts[0]]; s = [0.0]
    for a, b in zip(verts[:-1], verts[1:]):
        kx = KM * np.cos(np.radians(0.5 * (a[1] + b[1])))
        L = np.hypot((b[0] - a[0]) * kx, (b[1] - a[1]) * KM)
        n = max(1, int(np.ceil(L / dx)))
        t = np.arange(1, n + 1) / n
        pts.extend(a + np.outer(t, b - a)); s.extend(s[-1] + t * L)
    p = np.array(pts)
    return p[:, 0], p[:, 1], np.array(s)


def _xy(lon, lat, lat0):
    return np.c_[np.asarray(lon) * KM * np.cos(np.radians(lat0)), np.asarray(lat) * KM]


def project_events(ev, plon, plat, s, swath):
    lat0 = float(np.mean(plat))
    tree = cKDTree(_xy(plon, plat, lat0))
    box = ev[ev.lon.between(plon.min() - 1, plon.max() + 1) & ev.lat.between(plat.min() - 0.5, plat.max() + 0.5)]
    d, i = tree.query(_xy(box.lon, box.lat, lat0))
    m = (d <= swath) & (i > 0) & (i < len(s) - 1)
    return s[i[m]], box.depth.to_numpy()[m]


def sample_section(M, C, verts, zmin=-4.0, zmax=65.0, dx=1.0, dz=0.25, keys=("vp", "vs", "vpvs"), swath=10.0):
    plon, plat, s = profile_points(verts, dx)
    zz = np.arange(zmin, zmax + dz / 2, dz)
    Z, I = np.meshgrid(zz, np.arange(len(s)), indexing="ij")
    LAT, LON = plat[I], plon[I]
    topo = C.relief_at(plon, plat)
    if topo is None:
        topo = sample_grid(M.surface, np.isfinite(M.surface), (M.lat, M.lon), (plat, plon))
    above = Z < -topo[None, :]
    out = {}
    for k in keys:
        v = M.sample(k, Z, LAT, LON)
        out[k] = np.where(above, np.nan, v)
    rock = M.nearest(M.rock, Z, LAT, LON).astype(bool) & ~above
    es, ez = project_events(C.ev, plon, plat, s, swath)
    return dict(s=s, z=zz, lon=plon, lat=plat, f=out, rock=rock, topo=topo, slab=C.slab_at(plon, plat),
                es=es, ez=ez, swath=swath, chk={k: M.nearest(M.chk[k] & M.valid[k], Z, LAT, LON).astype(bool) & ~above
                                                for k in keys})


def section_panel(ax, S, key, cmap, norm, zmax, contours=True, ev_size=0.6, chk_outline=True, slab=True,
                  label_slab=True, grey="0.85", water="#e8f1f8"):
    s, zz = S["s"], S["z"]
    ext = [s[0], s[-1], zz[-1], zz[0]]
    ax.set_facecolor("white")
    G = S["f"][key]
    grey_rgba = np.zeros(G.shape + (4,))
    grey_rgba[S["rock"] & ~np.isfinite(G)] = matplotlib.colors.to_rgba(grey)
    ax.imshow(grey_rgba, extent=ext, aspect="auto", interpolation="nearest", zorder=1)
    im = ax.imshow(np.ma.masked_invalid(G), extent=ext, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest",
                   zorder=2)
    topo = S["topo"]
    ax.fill_between(s, 0, np.maximum(-topo, 0), color=water, lw=0, zorder=1.5)
    ax.plot(s, -topo, color="0.1", lw=0.5, zorder=6)
    if contours and key in CONTOUR and np.isfinite(G).sum() > 4:
        dv = CONTOUR[key] * (2 if key in ("vp", "vs") else 1)
        lv = np.arange(np.floor(np.nanmin(G) / dv) * dv, np.nanmax(G) + dv, dv)
        if len(lv) > 1:
            cs = ax.contour(s, zz, np.ma.masked_invalid(G), levels=lv, colors="0.15", linewidths=0.25, zorder=3)
            try:
                ax.clabel(cs, fontsize=4, fmt="%.2f" if key == "vpvs" else "%.1f", inline_spacing=1)
            except Exception:                                          # noqa: BLE001
                pass
    if chk_outline and S["chk"][key].any():
        ax.contour(s, zz, S["chk"][key].astype(float), levels=[0.5], colors="k", linewidths=0.5, zorder=4)
    ax.scatter(S["es"], S["ez"], s=ev_size, c="k", alpha=0.45, linewidths=0, zorder=5, rasterized=True)
    if slab and np.isfinite(S["slab"]).any():
        ax.plot(s, S["slab"], color="#0b4f8a", lw=0.8, ls=(0, (4, 2)), zorder=6)
        if label_slab:
            j = np.nanargmax(np.where(np.isfinite(S["slab"]), S["slab"], -np.inf))
            ax.text(s[j], S["slab"][j], "Slab2", color="#0b4f8a", fontsize=5, ha="right", va="top", zorder=7)
    ax.set_xlim(s[0], s[-1]); ax.set_ylim(zmax, zz[0])
    return im


# ------------------------------------------------------------------ gallery
def _gal_depth(args):
    M, C, k, out, scheme = args
    style(7)
    keys = ["vp", "vs", "vpvs", "dvp", "dvs"]
    fig, axs = plt.subplots(1, 5, figsize=(17.5, 6.3))
    fig.subplots_adjust(left=0.035, right=0.995, top=0.92, bottom=0.13, wspace=0.12)
    for j, (ax, key) in enumerate(zip(axs, keys)):
        setup_map(ax, REGION, left=(j == 0))
        norm = gallery_norm(M, key, k)
        im = map_layer(ax, M, C, key, k, REGION, cmap_for(key, scheme), norm, stations=(key == "vp"), st_size=3)
        n = int(M.valid[key][k].sum())
        ax.set_title(f"{QTY[key][1]}  ({n} shown nodes)", fontsize=7.5)
        if n == 0:
            ax.text(0.5, 0.5, "no shown nodes", transform=ax.transAxes, ha="center", fontsize=8, color="0.3")
        ax.apply_aspect()
        p = ax.get_position()
        unit = f" ({QTY[key][2]})" if QTY[key][2] else ""
        colorbar(fig, im, [p.x0 + 0.1 * p.width, p.y0 - 0.075, 0.8 * p.width, 0.018], f"{QTY[key][1]}{unit}")
    lo, hi = layer_bounds(M.z, k)
    fig.suptitle(f"{M.version}   depth {M.z[k]:g} km below sea level   (events {max(lo, -5):g} to {min(hi, 200):g} km, "
                 f"tiers {C.tiers}; black outline: checkerboard-resolved; grey: rock, not shown; white: water/air; "
                 f"dashed blue: Slab2 at this depth)", fontsize=8.5, y=0.985)
    f = out / f"map_z{M.z[k]:05.1f}.png"
    save(fig, f, 150, quantize=QUANT)
    return str(f)


def _gal_res(args):
    M, C, k, out = args
    style(7)
    fig, axs = plt.subplots(1, 4, figsize=(14.2, 6.3))
    fig.subplots_adjust(left=0.045, right=0.995, top=0.92, bottom=0.13, wspace=0.12)
    cm_res = ListedColormap(["#1b7837", "#7fbf7b", "#d9f0d3"])
    nres = BoundaryNorm([30, 50, 80, 120], cm_res.N)
    for j, (ax, var) in enumerate(zip(axs, ["resolution_p", "resolution_s", "dws_p", "dws_s"])):
        setup_map(ax, REGION, left=(j == 0))
        a = M.g[var][k].astype(float)
        a = np.where(M.rock[k] & (a > 0), a, np.nan)
        lonq, latq = map_grid(REGION, 0.025)
        LON, LAT = np.meshgrid(lonq, latq)
        A = M.nearest(a[None].repeat(len(M.z), 0), np.full(LON.shape, M.z[k]), LAT, LON)
        rockg = M.nearest(M.rock, np.full(LON.shape, M.z[k]), LAT, LON).astype(bool)
        ext = [lonq[0] - 0.0125, lonq[-1] + 0.0125, latq[0] - 0.0125, latq[-1] + 0.0125]
        gr = np.zeros(LON.shape + (4,)); gr[rockg & ~np.isfinite(A)] = matplotlib.colors.to_rgba("0.85")
        ax.imshow(gr, extent=ext, origin="lower", interpolation="nearest", aspect=ax.get_aspect())
        if var.startswith("res"):
            im = ax.imshow(np.ma.masked_invalid(A), extent=ext, origin="lower", cmap=cm_res, norm=nres,
                           interpolation="nearest", aspect=ax.get_aspect())
            ttl = f"finest resolved checkerboard, {'P' if var.endswith('p') else 'S'}"
        else:
            im = ax.imshow(np.ma.masked_invalid(A), extent=ext, origin="lower", cmap="viridis", norm=LogNorm(1, 1e4),
                           interpolation="nearest", aspect=ax.get_aspect())
            if np.isfinite(A).any():
                ax.contour(LON, LAT, np.nan_to_num(A), levels=[M.dws_min], colors="w", linewidths=0.6)
            ttl = f"DWS {'P' if var.endswith('p') else 'S'} (white: {M.dws_min:g})"
        n = int((M.rock[k] & (M.g[var][k] > (0 if var.startswith('res') else M.dws_min - 1e-9))).sum())
        ax.set_title(f"{ttl}  [{n} nodes]", fontsize=7.5)
        draw_context(ax, C, REGION, stations=(j == 2), st_size=3)
        ax.apply_aspect()
        p = ax.get_position()
        if var.startswith("res"):
            cb = colorbar(fig, im, [p.x0 + 0.1 * p.width, p.y0 - 0.075, 0.8 * p.width, 0.018], "cell size (km)",
                          ticks=[40, 65, 100])
            cb.ax.set_xticklabels(["40", "60", "100"])
        else:
            colorbar(fig, im, [p.x0 + 0.1 * p.width, p.y0 - 0.075, 0.8 * p.width, 0.018], "DWS")
    fig.suptitle(f"{M.version}   depth {M.z[k]:g} km: checkerboard resolution (D-B018: local corr >= 0.7 and amplitude "
                 f">= 0.5; 0 = not resolved, blank) and derivative weight sum", fontsize=8.5, y=0.985)
    f = out / f"res_z{M.z[k]:05.1f}.png"
    save(fig, f, 150, quantize=QUANT)
    return str(f)


def _gal_section(args):
    M, C, key, out, scheme, zmax = args
    style(7)
    name, verts = PROFILES[key]
    S = sample_section(M, C, verts, zmax=zmax)
    L = S["s"][-1]
    fig = plt.figure(figsize=(14.5, 9.6))
    hs = [0.6, 2.4, 2.4, 2.4]
    gs = fig.add_gridspec(4, 2, height_ratios=hs, width_ratios=[5.2, 1.0], left=0.05, right=0.985, top=0.93,
                          bottom=0.06, hspace=0.12, wspace=0.08)
    axt = fig.add_subplot(gs[0, 0])
    axt.fill_between(S["s"], S["topo"], 0, where=S["topo"] < 0, color="#9ecae1", lw=0)
    axt.fill_between(S["s"], S["topo"], -10, color="0.55", lw=0)
    axt.plot(S["s"], S["topo"], color="0.1", lw=0.6)
    axt.set_xlim(0, L); axt.set_ylim(min(-4.5, np.nanmin(S["topo"]) - 0.3), max(2.0, np.nanmax(S["topo"]) + 0.3))
    axt.set_ylabel("elev. (km)"); plt.setp(axt.get_xticklabels(), visible=False)
    axt.set_title(f"Profile {key}: {name}   ({L:.0f} km; events tiers {C.tiers} within ±{S['swath']:g} km; "
                  f"black outline: checkerboard-resolved; dashed: Slab2; grey: rock not shown)", fontsize=8.5, loc="left")
    norms = {"vp": Normalize(4.0, 8.4), "vs": Normalize(2.2, 4.8), "vpvs": Normalize(*FIXED["vpvs"])}
    for i, k in enumerate(("vp", "vs", "vpvs")):
        ax = fig.add_subplot(gs[i + 1, 0], sharex=axt)
        im = section_panel(ax, S, k, cmap_for(k, scheme), norms[k], zmax)
        ax.set_ylabel("depth (km)")
        if i < 2:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel("distance along profile (km)")
        unit = f" ({QTY[k][2]})" if QTY[k][2] else ""
        p = ax.get_position()
        colorbar(fig, im, [p.x1 + 0.006, p.y0 + 0.1 * p.height, 0.007, 0.8 * p.height], f"{QTY[k][1]}{unit}",
                 orientation="vertical")
        ve = (p.height * 9.6 / (zmax + 4)) / (p.width * 14.5 / L)
        ax.text(0.995, 0.03, f"VE {ve:.1f}", transform=ax.transAxes, ha="right", fontsize=6, color="0.3")
    # locator
    axm = fig.add_subplot(gs[:, 1])
    setup_map(axm, REGION, step=4)
    k21 = int(np.argmin(np.abs(M.z - 21)))
    lonq, latq = map_grid(REGION, 0.05)
    LON, LAT = np.meshgrid(lonq, latq)
    shown = M.nearest(M.valid["vp"], np.full(LON.shape, M.z[k21]), LAT, LON).astype(bool)
    axm.imshow(np.ma.masked_where(~shown, shown), extent=[REGION[0], REGION[1], REGION[2], REGION[3]], origin="lower",
               cmap=ListedColormap(["#fdd49e"]), interpolation="nearest", aspect=axm.get_aspect())
    draw_context(axm, C, REGION, lw=0.3)
    for kk, (nm, vv) in PROFILES.items():
        vv = np.asarray(vv)
        axm.plot(vv[:, 0], vv[:, 1], color="0.6" if kk != key else "r", lw=0.6 if kk != key else 1.4)
    vv = np.asarray(verts)
    axm.text(vv[0, 0], vv[0, 1], key, color="r", fontsize=7, ha="right", va="center", fontweight="bold")
    axm.set_title(f"orange: shown Vp at {M.z[k21]:g} km", fontsize=6.5)
    f = out / f"section_{key}.png"
    save(fig, f, 150, quantize=QUANT)
    return str(f)


def _gal_overview(M, C, out):
    style(7)
    fig, ax = plt.subplots(figsize=(6.0, 7.6))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.05)
    setup_map(ax, REGION)
    k = int(np.argmin(np.abs(M.z - 21)))
    lonq, latq = map_grid(REGION, 0.025)
    LON, LAT = np.meshgrid(lonq, latq)
    any_shown = M.valid["vp"].any(0)
    sh = M.nearest(any_shown[None].repeat(len(M.z), 0), np.full(LON.shape, M.z[k]), LAT, LON).astype(bool)
    ax.imshow(np.ma.masked_where(~sh, sh), extent=[lonq[0], lonq[-1], latq[0], latq[-1]], origin="lower",
              cmap=ListedColormap(["#fdd49e"]), interpolation="nearest", aspect=ax.get_aspect())
    ax.scatter(C.ev.lon, C.ev.lat, s=0.05, c="k", alpha=0.25, linewidths=0, rasterized=True)
    draw_context(ax, C, REGION, stations=True, st_size=4)
    for key, (nm, vv) in PROFILES.items():
        vv = np.asarray(vv)
        ax.plot(vv[:, 0], vv[:, 1], color="r", lw=1.0)
        ax.text(vv[0, 0] - 0.08, vv[0, 1], key, color="r", fontsize=8, ha="right", va="center", fontweight="bold")
    ax.set_title(f"Profiles; orange: Vp shown at any depth; dots: catalogue tiers {C.tiers}", fontsize=7)
    f = out / "profiles_map.png"
    save(fig, f, 150, quantize=QUANT)
    return str(f)


def _img(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def _html_page(title, body):
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title>
<style>
:root{{--bg:#fff;--fg:#1d1d1f;--mut:#5f6368;--line:#d9d9d9;--acc:#0b4f8a}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#fff;--fg:#1d1d1f}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:0 auto;
max-width:1500px;padding:0 16px 60px}}
h1{{font-size:22px;margin:18px 0 4px}} h2{{font-size:18px;margin:28px 0 8px;border-bottom:1px solid var(--line)}}
h3{{font-size:15px;margin:18px 0 4px}} .mut{{color:var(--mut)}} img{{max-width:100%;height:auto;display:block;
border:1px solid var(--line);margin:4px 0 14px}} table{{border-collapse:collapse;font-size:12.5px;margin:6px 0}}
td,th{{border:1px solid var(--line);padding:2px 7px;text-align:right}} th{{background:#f4f4f4}}
nav{{position:sticky;top:0;background:var(--bg);padding:6px 0;border-bottom:1px solid var(--line);font-size:13px;z-index:2}}
nav a{{margin-right:10px;color:var(--acc);text-decoration:none}} code{{font-size:12.5px}} ul{{margin:4px 0}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:12px}} @media(max-width:900px){{.pair{{grid-template-columns:1fr}}}}
.cap{{font-size:13px;margin:2px 0 8px}} .tag{{display:inline-block;background:#f4f4f4;border-radius:3px;padding:0 5px;font-size:12px}}
</style></head><body>{body}</body></html>"""


def git_head():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:                                                  # noqa: BLE001
        return "?"


def header_html(M, C, extra=""):
    a = M.attrs
    cnt = M.counts[M.counts.depth_km <= 60].to_html(index=False, border=0)
    miss = "".join(f"<li>{m}</li>" for m in C.missing) or "<li>none</li>"
    return f"""
<p><b>Model</b> <code>{M.path}</code> &middot; {a.get('id', M.version)} &middot; data revision {a.get('data_revision', '?')}
&middot; sha256 <code>{M.sha256()[:16]}</code><br><span class="mut">{a.get('title', '')}</span></p>
<p><b>Grid</b> {len(M.lat)} latitudes {M.lat[0]:g}&ndash;{M.lat[-1]:g}&deg;N (0.1&deg;) &times; {len(M.lon)} longitudes
{M.lon[0]:g}&ndash;{M.lon[-1]:g}&deg; (0.125&deg;) &times; {len(M.z)} depths (km below sea level, uneven):
{', '.join(f'{z:g}' for z in M.z)}. Node spacing about 11 km N&ndash;S and 9&ndash;10 km E&ndash;W.</p>
<p><b>Shown cells</b> (mask <code>{M.mask}</code>): rock nodes only (<code>medium = 0</code>; nothing above the seafloor
or ground) that are <b>checkerboard-resolved</b> (<code>resolution_p/s &gt; 0</code>, D-B018: local correlation &ge; 0.7 and
amplitude &ge; 0.5, finest of 40/60/100 km cells) <b>or well sampled</b> (<code>DWS &ge; {M.dws_min:g}</code>); P for Vp
and dVp, S for Vs and dVs, both for Vp/Vs; and data-derived (<code>tile_weight &gt; 0</code>; for Vs also
<code>vs_reverted = 0</code> and not a D-B073 shallow-crust node). <b>Which rule passes</b>: cells inside the black
outline are checkerboard-resolved; the rest pass on DWS only. The checkerboard-resolved set is small (table) and
empty at 21, 25 and 34&ndash;60 km, so those depths are DWS-only. Between nodes: multilinear interpolation over shown
nodes only, drawn only where the nearest node is shown (no value is made from a masked node).</p>
<p><b>Colour</b>: matplotlib <code>seismic</code> (PI atlas choice): slow = red, fast = blue; Vp/Vs high = red.
Maps: Vp and Vs centred on the median of the inverted rock nodes of that depth (tile_weight &gt; 0, shown or not; range per depth from the 98th percentile of the shown nodes, on each colour bar);
Vp/Vs 1.60&ndash;1.90; dVp and dVs &plusmn;8 % (change from the 3-D starting model, not from a 1-D average).
Contours every {CONTOUR['vp']:g} km/s (Vp), {CONTOUR['vs']:g} km/s (Vs), {CONTOUR['vpvs']:g} (Vp/Vs), {CONTOUR['dvp']:g} % (dV);
sections every {2 * CONTOUR['vp']:g} / {2 * CONTOUR['vs']:g} km/s, Vp 4.0&ndash;8.4, Vs 2.2&ndash;4.8 km/s.</p>
<p><b>Overlays</b>: catalogue <code>{C.catalog}</code>, tiers {C.tiers} ({len(C.ev):,} of {C.n_ev_all:,} events), maps:
events between the mid-depths to the neighbouring nodes; sections: within &plusmn;10 km. Stations: ver5 table
(<code>in_ver5</code>, &#9650; land, &#9660; ocean bottom). Coast: GMT high resolution. Plate boundaries and trench:
PB2002 (Bird 2003). Slab2 top: <code>{C.slab_path}</code>. Topography/bathymetry: GMT earth_relief 01m.
Missing inputs: <ul>{miss}</ul></p>
{extra}
<details><summary><b>Node counts per depth</b> (rock nodes, checkerboard-resolved, DWS &ge; {M.dws_min:g}, shown)</summary>{cnt}</details>
"""


def gallery(M, C, out, jobs=4, scheme="seismic", zmax_map=60.0, zmax_sec=65.0):
    gdir = out / "gallery_png"
    gdir.mkdir(parents=True, exist_ok=True)
    ks = [k for k in range(len(M.z)) if M.z[k] <= zmax_map]
    tasks_d = [(M, C, k, gdir, scheme) for k in ks]
    tasks_r = [(M, C, k, gdir) for k in ks]
    tasks_s = [(M, C, key, gdir, scheme, zmax_sec) for key in PROFILES]
    with Pool(jobs) as pool:
        fd = pool.map(_gal_depth, tasks_d)
        fs = pool.map(_gal_section, tasks_s)
        fr = pool.map(_gal_res, tasks_r)
    fo = _gal_overview(M, C, gdir)
    nav = ("<nav><a href='#maps'>Maps</a>" + "".join(f"<a href='#z{M.z[k]:g}'>{M.z[k]:g}</a>" for k in ks)
           + "<a href='#sections'>Sections</a>" + "".join(f"<a href='#sec{k}'>{k}</a>" for k in PROFILES)
           + "<a href='#res'>Resolution</a></nav>")
    when = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    body = [f"<h1>{M.version} exploration gallery</h1><p class='mut'>Made {when} by "
            f"<code>workflow/08_figures/model_figures.py gallery</code> (branch paper/model-figures, {git_head()}). "
            f"Exploration only: not paper figures.</p>", nav, header_html(M, C)]
    body.append("<h2 id='maps'>Map slices (every node depth to 60 km)</h2><p class='cap'>Columns: Vp, Vs, Vp/Vs, dVp, dVs. "
                "Stations on the Vp panel. Dots: events in the depth layer of the node.</p>")
    for k, f in zip(ks, fd):
        body.append(f"<h3 id='z{M.z[k]:g}'>{M.z[k]:g} km</h3><img loading='lazy' src='{_img(f)}' alt='maps at {M.z[k]:g} km'>")
    body.append("<h2 id='sections'>Cross-sections</h2><p class='cap'>Top strip: topography/bathymetry (earth_relief 01m). "
                "Panels: Vp, Vs, Vp/Vs; water light blue; air white. West or south on the left.</p>")
    body.append(f"<img src='{_img(fo)}' style='max-width:620px' alt='profile map'>")
    for key, f in zip(PROFILES, fs):
        body.append(f"<h3 id='sec{key}'>{key}: {PROFILES[key][0]}</h3>"
                    f"<p class='cap mut'>vertices (lon, lat): {', '.join(f'({a:g}, {b:g})' for a, b in PROFILES[key][1])}</p>"
                    f"<img loading='lazy' src='{_img(f)}' alt='section {key}'>")
    body.append("<h2 id='res'>Resolution</h2><p class='cap'>Finest resolved checkerboard cell (P, S; D-B018 per-tile masks "
                "merged as the finest over the covering tiles; blank = not resolved) and the merge-weighted derivative weight "
                f"sum (P, S; white contour = {M.dws_min:g}). Grey: rock nodes with no value.</p>")
    for k, f in zip(ks, fr):
        body.append(f"<h3>{M.z[k]:g} km</h3><img loading='lazy' src='{_img(f)}' alt='resolution at {M.z[k]:g} km'>")
    html = out / f"gallery_{M.version}.html"
    html.write_text(_html_page(f"{M.version} model gallery", "\n".join(body)))
    print(f"wrote {html} ({html.stat().st_size / 1e6:.1f} MB)")
    return html


# ------------------------------------------------------------------ paper candidates
PAPER_REGION = [-131.0, -121.0, 39.9, 49.3]
PAPER_DEPTHS = [9.0, 18.0, 29.0]        # node depths near 8 / 20 / 35 km with checkerboard-resolved nodes (21, 34: none)
# M2 profiles: margin-normal, 4 deg of longitude each (same km scale), through the shown Vp/Vs
PAPER_PROFILES = [("Mendocino 40.4°N", [(-126.4, 40.4), (-122.4, 40.4)]),
                  ("Gorda 40.9°N", [(-126.4, 40.9), (-122.4, 40.9)]),
                  ("Blanco 43.5°N", [(-129.4, 43.5), (-125.4, 43.5)]),
                  ("S. Washington 46.4°N", [(-125.0, 46.4), (-121.0, 46.4)])]
SCHEMES = ("seismic", "vik")
LAB = {"vp": "Vp (km/s)", "vs": "Vs (km/s)", "vpvs": "Vp/Vs"}


def _cm_axes(fig, W, H, x, y, w, h):
    """Axes from a layout in cm (origin bottom left)."""
    return fig.add_axes([x / W, y / H, w / W, h / H])


def _tag(ax, letter, x=0.025, y=0.025, fs=8, ha="left", va="bottom"):
    ax.text(x, y, letter, transform=ax.transAxes, ha=ha, va=va, fontsize=fs, fontweight="bold", zorder=12,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8))


def _inset_cbar(fig, ax, im, label, rect=(0.05, 0.885, 0.40, 0.028), nticks=3, fs=5.0):
    bg = ax.inset_axes([rect[0] - 0.03, rect[1] - 0.075, rect[2] + 0.06, rect[3] + 0.155])
    bg.set_facecolor((1, 1, 1, 0.85)); bg.set_xticks([]); bg.set_yticks([])
    for sp in bg.spines.values():
        sp.set_visible(False)
    bg.set_zorder(9)
    cax = ax.inset_axes(list(rect)); cax.set_zorder(10)
    lo, hi = im.norm.vmin, im.norm.vmax
    ticks = np.round(np.linspace(lo, hi, nticks), 2)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=ticks)
    cb.outline.set_linewidth(0.3)
    cb.ax.tick_params(length=1.2, pad=0.8, width=0.3, labelsize=fs)
    cb.ax.set_xticklabels([f"{t:g}" for t in ticks])
    cb.ax.set_title(label, fontsize=fs, pad=1.5)
    return cb


def paper_norm(M, key, k):
    if key == "vpvs":
        return Normalize(1.60, 1.90)
    c = layer_centre(M, key, k)
    r = vrange(M, key, [k], sym_about=c, rnd=0.1)
    return Normalize(*r)


def fig_m1(M, C, out, scheme, depths=PAPER_DEPTHS, region=PAPER_REGION):
    style(6)
    W, E, S, N = region
    asp = (N - S) / ((E - W) * np.cos(np.radians((S + N) / 2)))
    L, R, T, B, gx, gy = 1.0, 0.15, 0.55, 0.55, 0.12, 0.15
    pw = (COL2 - L - R - 2 * gx) / 3
    ph = pw * asp
    H = T + B + 3 * ph + 2 * gy
    fig = plt.figure(figsize=(COL2 * CM, H * CM))
    ks = [int(np.argmin(np.abs(M.z - d))) for d in depths]
    letters = iter("ABCDEFGHI")
    for r, k in enumerate(ks):
        for c, key in enumerate(("vp", "vs", "vpvs")):
            ax = _cm_axes(fig, COL2, H, L + c * (pw + gx), H - T - (r + 1) * ph - r * gy, pw, ph)
            setup_map(ax, region, step=2, left=(c == 0), bottom=(r == 2))
            im = map_layer(ax, M, C, key, k, region, cmap_for(key, scheme), paper_norm(M, key, k), step=0.02,
                           ev_size=0.12, ctx_lw=0.3, contours=True)
            _inset_cbar(fig, ax, im, LAB[key])
            ax.text(0.97, 0.97, f"{M.z[k]:g} km", transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
                    fontweight="bold", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0), zorder=11)
            _tag(ax, next(letters))
            if r == 0:
                ax.set_title({"vp": "P-wave velocity", "vs": "S-wave velocity", "vpvs": "Vp/Vs"}[key], fontsize=7, pad=4)
    f = out / f"figM1_overview_{scheme}.png"
    save(fig, f, 300, pdf=True)
    return f


def fig_m2(M, C, out, scheme, profiles=PAPER_PROFILES, zmax=45.0, xmax=345.0):
    style(6)
    L, R, T, B = 4.9, 1.15, 0.35, 0.75
    gy, n = 0.22, len(profiles)
    sw = COL2 - L - R
    sh = 1.6 * sw * (zmax + 3) / xmax                  # vertical exaggeration 1.6
    H = T + B + n * sh + (n - 1) * gy
    fig = plt.figure(figsize=(COL2 * CM, H * CM))
    cmap, norm = cmap_for("vpvs", scheme), Normalize(1.60, 1.90)
    # locator
    lw = L - 1.75
    reg = [-130.0, -120.6, 39.6, 47.4]
    lh = lw * (reg[3] - reg[2]) / ((reg[1] - reg[0]) * np.cos(np.radians(43.5)))
    axm = _cm_axes(fig, COL2, H, 0.85, H - T - lh, lw, lh)
    setup_map(axm, reg, step=4)
    lonq, latq = map_grid(reg, 0.05)
    LON, LAT = np.meshgrid(lonq, latq)
    sh_any = M.valid["vpvs"].any(0)
    g = M.nearest(sh_any[None].repeat(len(M.z), 0), np.full(LON.shape, 10.0), LAT, LON).astype(bool)
    axm.imshow(np.ma.masked_where(~g, g), extent=[lonq[0], lonq[-1], latq[0], latq[-1]], origin="lower",
               cmap=ListedColormap(["#fdae6b"]), interpolation="nearest", aspect=axm.get_aspect())
    draw_context(axm, C, reg, lw=0.3, teeth_km=60)
    letters = "ABCDEFGH"
    for i, (nm, vv) in enumerate(profiles):
        vv = np.asarray(vv)
        axm.plot(vv[:, 0], vv[:, 1], color="k", lw=0.9)
        axm.text(vv[0, 0] - 0.1, vv[0, 1], f"{letters[i + 1]}", fontsize=6, ha="right", va="center", fontweight="bold",
                 bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.3))
    _tag(axm, "A", x=0.03, y=0.97, va="top")
    for i, (nm, vv) in enumerate(profiles):
        y0 = H - T - (i + 1) * sh - i * gy
        ax = _cm_axes(fig, COL2, H, L, y0, sw, sh)
        Sx = sample_section(M, C, vv, zmin=-3.0, zmax=zmax, dx=0.5, dz=0.2, keys=("vp", "vs", "vpvs"))
        im = section_panel(ax, Sx, "vpvs", cmap, norm, zmax, contours=False, ev_size=0.25, label_slab=(i == 0))
        Gp = Sx["f"]["vp"]
        if np.isfinite(Gp).sum() > 4:
            cs = ax.contour(Sx["s"], Sx["z"], np.ma.masked_invalid(Gp), levels=[6.0, 6.5, 7.0, 7.5, 8.0], colors="0.1",
                            linewidths=0.35, linestyles="-", zorder=4.5)
            ax.clabel(cs, fontsize=4.5, fmt="%.1f", inline_spacing=1)
        ax.set_xlim(0, xmax)
        ax.set_ylabel("depth (km)")
        if i < n - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("distance (km)")
        ax.text(0.005, 0.04, f"{letters[i + 1]}  {nm}", transform=ax.transAxes, fontsize=6, fontweight="bold",
                va="bottom", bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8), zorder=11)
        ve = (sh / (zmax + 3)) / (sw / xmax)
        if i == 0:
            ax.text(0.995, 0.04, f"VE {ve:.1f}", transform=ax.transAxes, ha="right", fontsize=5, color="0.25", zorder=11)
    cax = _cm_axes(fig, COL2, H, COL2 - R + 0.2, B + 0.25 * (H - T - B), 0.15, 0.5 * (H - T - B))
    cb = fig.colorbar(im, cax=cax, orientation="vertical", ticks=[1.6, 1.7, 1.8, 1.9])
    cb.outline.set_linewidth(0.3); cb.ax.tick_params(length=1.5, width=0.3, labelsize=5.5)
    cb.set_label("Vp/Vs", fontsize=6)
    f = out / f"figM2_sections_{scheme}.png"
    save(fig, f, 300, pdf=True)
    return f


def fig_sm3(M, C, out, depths=PAPER_DEPTHS, region=PAPER_REGION):
    style(6)
    W, E, S, N = region
    asp = (N - S) / ((E - W) * np.cos(np.radians((S + N) / 2)))
    L, R, T, B, gx, gy = 1.0, 0.15, 0.5, 1.5, 0.1, 0.12
    pw = (COL2 - L - R - 3 * gx) / 4
    ph = pw * asp
    H = T + B + 3 * ph + 2 * gy
    fig = plt.figure(figsize=(COL2 * CM, H * CM))
    ks = [int(np.argmin(np.abs(M.z - d))) for d in depths]
    cm_res = ListedColormap(["#1b7837", "#7fbf7b", "#d9f0d3"])
    nres = BoundaryNorm([30, 50, 80, 120], cm_res.N)
    lonq, latq = map_grid(region, 0.02)
    LON, LAT = np.meshgrid(lonq, latq)
    ext = [lonq[0] - 0.01, lonq[-1] + 0.01, latq[0] - 0.01, latq[-1] + 0.01]
    letters = iter("ABCDEFGHIJKL")
    ims = {}
    for r, k in enumerate(ks):
        Z = np.full(LON.shape, M.z[k])
        rockg = M.nearest(M.rock, Z, LAT, LON).astype(bool)
        for c, var in enumerate(("resolution_p", "resolution_s", "dws_p", "dws_s")):
            ax = _cm_axes(fig, COL2, H, L + c * (pw + gx), H - T - (r + 1) * ph - r * gy, pw, ph)
            setup_map(ax, region, step=2, left=(c == 0), bottom=(r == 2))
            a = np.where(M.rock & (M.g[var] > 0), M.g[var].astype(float), np.nan)
            A = M.nearest(a, Z, LAT, LON)
            gr = np.zeros(LON.shape + (4,)); gr[rockg & ~np.isfinite(A)] = matplotlib.colors.to_rgba("0.85")
            ax.imshow(gr, extent=ext, origin="lower", interpolation="nearest", aspect=ax.get_aspect())
            if var.startswith("res"):
                ims[c] = ax.imshow(np.ma.masked_invalid(A), extent=ext, origin="lower", cmap=cm_res, norm=nres,
                                   interpolation="nearest", aspect=ax.get_aspect())
            else:
                ims[c] = ax.imshow(np.ma.masked_invalid(A), extent=ext, origin="lower", cmap="viridis",
                                   norm=LogNorm(1, 1e4), interpolation="nearest", aspect=ax.get_aspect())
                if np.isfinite(A).any():
                    ax.contour(LON, LAT, np.nan_to_num(A), levels=[M.dws_min], colors="w", linewidths=0.5)
            draw_context(ax, C, region, lw=0.3, stations=(c == 2 and r == 0), st_size=1.2)
            n = int((M.rock[k] & (M.g[var][k] >= (1e-9 if var.startswith("res") else M.dws_min))).sum())
            ax.text(0.97, 0.97, f"{M.z[k]:g} km", transform=ax.transAxes, ha="right", va="top", fontsize=6,
                    fontweight="bold", bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8), zorder=11)
            ax.text(0.97, 0.89, f"{n} nodes" + ("" if var.startswith("res") else f" ≥ {M.dws_min:g}"), ha="right",
                    va="top", transform=ax.transAxes, fontsize=5, bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6),
                    zorder=11)
            _tag(ax, next(letters))
            if r == 0:
                ax.set_title({"resolution_p": "checkerboard, P", "resolution_s": "checkerboard, S",
                              "dws_p": "DWS, P", "dws_s": "DWS, S"}[var], fontsize=7, pad=4)
    for c in range(4):
        cax = _cm_axes(fig, COL2, H, L + c * (pw + gx) + 0.15 * pw, 0.75, 0.7 * pw, 0.12)
        if c < 2:
            cb = fig.colorbar(ims[c], cax=cax, orientation="horizontal", ticks=[40, 65, 100])
            cb.ax.set_xticklabels(["40", "60", "100"]); cb.set_label("finest resolved cell (km)", fontsize=6, labelpad=1)
        else:
            cb = fig.colorbar(ims[c], cax=cax, orientation="horizontal"); cb.set_label("derivative weight sum", fontsize=6,
                                                                                       labelpad=1)
        cb.outline.set_linewidth(0.3); cb.ax.tick_params(length=1.5, width=0.3, labelsize=5.5)
    f = out / "figSM3_resolution.png"
    save(fig, f, 300, pdf=True)
    return f


def load_prior(paths):
    for p in paths:
        if Path(p).exists():
            d = xr.open_dataset(p)
            v = next(x for x in d.data_vars if x.lower() == "vs")
            dep = next(c for c in d.coords if "dep" in c.lower())
            la = next(c for c in d.coords if "lat" in c.lower())
            lo = next(c for c in d.coords if "lon" in c.lower())
            a = d[v].transpose(dep, la, lo)
            lon = d[lo].values.astype(float)
            lon = np.where(lon > 180, lon - 360, lon)
            lat = d[la].values.astype(float)
            V = a.values.astype(float)
            if lat[0] > lat[-1]:
                lat, V = lat[::-1], V[:, ::-1]
            o = np.argsort(lon)
            return dict(path=Path(p), z=d[dep].values.astype(float), lat=lat, lon=lon[o], v=V[:, :, o])
    return None


def fig_sm4(M, C, out, scheme, depths=PAPER_DEPTHS, region=PAPER_REGION):
    priors = {n: load_prior(p) for n, p in PRIORS.items()}
    for n, p in priors.items():
        if p is None and n not in " ".join(C.missing):
            C.missing.append(f"{n} (data/tomography; utils/fetch_tomography.py)")
    names = [n for n, p in priors.items() if p is not None]
    style(6)
    W, E, S, N = region
    asp = (N - S) / ((E - W) * np.cos(np.radians((S + N) / 2)))
    ncol = 1 + 2 * len(names)
    L, R, T, B, gx, gy = 1.0, 0.15, 0.75, 1.3, 0.1, 0.12
    pw = (COL2 - L - R - (ncol - 1) * gx) / ncol
    ph = pw * asp
    H = T + B + 3 * ph + 2 * gy
    fig = plt.figure(figsize=(COL2 * CM, H * CM))
    ks = [int(np.argmin(np.abs(M.z - d))) for d in depths]
    lonq, latq = map_grid(region, 0.02)
    LON, LAT = np.meshgrid(lonq, latq)
    letters = iter("ABCDEFGHIJKLMNO")
    stats = []
    cmv, cmd = cmap_for("vs", scheme), cmap_for("dvs", scheme)
    nd = Normalize(-15, 15)
    for r, k in enumerate(ks):
        z = M.z[k]
        Z = np.full(LON.shape, z)
        rockg = M.nearest(M.rock, Z, LAT, LON).astype(bool)
        norm = paper_norm(M, "vs", k)
        col = 0
        panels = [("ours", None)] + [("prior", n) for n in names] + [("diff", n) for n in names]
        for kind, n in panels:
            ax = _cm_axes(fig, COL2, H, L + col * (pw + gx), H - T - (r + 1) * ph - r * gy, pw, ph)
            setup_map(ax, region, step=3, left=(col == 0), bottom=(r == 2))
            if kind == "ours":
                im = map_layer(ax, M, C, "vs", k, region, cmv, norm, step=0.02, ev_size=0.08, ctx_lw=0.25,
                               contours=False, slab=False)
                ttl = f"this study ({M.version})"
            elif kind == "prior":
                p = priors[n]
                G = sample_grid(np.nan_to_num(p["v"]), np.isfinite(p["v"]), (p["z"], p["lat"], p["lon"]), (Z, LAT, LON))
                G = np.where(rockg, G, np.nan)
                ax.set_facecolor("white")
                im2 = ax.imshow(np.ma.masked_invalid(G), extent=[lonq[0], lonq[-1], latq[0], latq[-1]], origin="lower",
                                cmap=cmv, norm=norm, interpolation="nearest", aspect=ax.get_aspect())
                draw_context(ax, C, region, lw=0.25)
                ttl = n
            else:
                p = priors[n]
                kk = np.full(M.f["vs"].shape, False); kk[k] = True
                zz, yy, xx = np.meshgrid(M.z, M.lat, M.lon, indexing="ij")
                pv = sample_grid(np.nan_to_num(p["v"]), np.isfinite(p["v"]), (p["z"], p["lat"], p["lon"]),
                                 (zz[k], yy[k], xx[k]))
                d = np.full(M.f["vs"].shape, np.nan)
                d[k] = 100 * (M.f["vs"][k] / pv - 1)
                vd = M.valid["vs"] & np.isfinite(d)
                im3 = map_layer(ax, M, C, "dvs", k, region, cmd, nd, step=0.02, contours=False, events=False,
                                chk_outline=False, slab=False, ctx_lw=0.25, field=np.nan_to_num(d), valid=vd)
                dv = d[k][vd[k]]
                if dv.size:
                    stats.append(dict(depth_km=z, prior=n, n_nodes=int(dv.size), median_pct=float(np.median(dv)),
                                      mad_pct=float(np.median(np.abs(dv - np.median(dv)))),
                                      rms_pct=float(np.sqrt(np.mean(dv ** 2)))))
                    ax.text(0.04, 0.96, f"n {dv.size}\nmedian {np.median(dv):+.1f}%", transform=ax.transAxes,
                            fontsize=4.8, va="top", bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6), zorder=11)
                else:
                    ax.text(0.5, 0.5, "no overlap", transform=ax.transAxes, ha="center", fontsize=5.5, color="0.3")
                ttl = f"this study − {n.split(' ')[0]}"
            ax.text(0.97, 0.97, f"{z:g} km", transform=ax.transAxes, ha="right", va="top", fontsize=5.5,
                    fontweight="bold", bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6), zorder=11)
            _tag(ax, next(letters), fs=7)
            if r == 0:
                ax.set_title(ttl, fontsize=6, pad=3)
            if kind == "ours":
                _inset_cbar(fig, ax, im, "Vs (km/s)", rect=(0.06, 0.86, 0.5, 0.032), fs=4.6)
            col += 1
    # shared colour bar for the differences
    x0 = L + (1 + len(names)) * (pw + gx)
    cax = _cm_axes(fig, COL2, H, x0 + 0.1 * pw, 0.65, len(names) * (pw + gx) - gx - 0.2 * pw, 0.12)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=nd, cmap=cmd), cax=cax, orientation="horizontal",
                      ticks=[-15, -10, -5, 0, 5, 10, 15])
    cb.set_label("Vs difference, this study − prior (%)", fontsize=6, labelpad=1)
    cb.outline.set_linewidth(0.3); cb.ax.tick_params(length=1.5, width=0.3, labelsize=5.5)
    f = out / f"figSM4_prior_models_{scheme}.png"
    save(fig, f, 300, pdf=True)
    pd.DataFrame(stats).to_csv(out / "figSM4_differences.csv", index=False, float_format="%.3f")
    return f, pd.DataFrame(stats), {n: str(p["path"]) for n, p in priors.items() if p is not None}


def _and(v):
    v = [f"{x}" for x in v]
    return ", ".join(v[:-1]) + f" and {v[-1]}" if len(v) > 1 else v[0]


def _caption_m1(M, C, ks):
    nv = [int(M.valid['vp'][k].sum()) for k in ks]
    ns = [int(M.valid['vs'][k].sum()) for k in ks]
    nc = [int((M.chk['vp'][k] & M.valid['vp'][k]).sum()) for k in ks]
    return (f"P-wave velocity (A, D, G), S-wave velocity (B, E, H) and Vp/Vs (C, F, I) of {M.version} at "
            f"{', '.join(f'{M.z[k]:g}' for k in ks)} km below sea level. Only rock nodes that are checkerboard-resolved "
            f"(black outline) or have DWS ≥ {M.dws_min:g} are coloured (Vp: {_and(nv)} nodes, of which "
            f"{_and(nc)} checkerboard-resolved; Vs: {_and(ns)} nodes); grey: rock not shown; white: water or air. "
            f"Dots: catalogue "
            f"{C.catalog.name} tiers {C.tiers} events in the depth layer of the node; dashed blue: Slab2 top at that "
            f"depth; black lines: PB2002 plate boundaries and coast.")


def paper(M, C, out, a=None, schemes=SCHEMES):
    pdir = out / "paper_candidates"
    pdir.mkdir(parents=True, exist_ok=True)
    ks = [int(np.argmin(np.abs(M.z - d))) for d in PAPER_DEPTHS]
    figs = {}
    for sc in schemes:
        figs[("M1", sc)] = fig_m1(M, C, pdir, sc)
        figs[("M2", sc)] = fig_m2(M, C, pdir, sc)
        f4, st4, used = fig_sm4(M, C, pdir, sc)
        figs[("SM4", sc)] = f4
    figs[("SM3", "-")] = fig_sm3(M, C, pdir)
    write_review(M, C, out, pdir, figs, ks, st4, used)


def write_review(M, C, out, pdir, figs, ks, st4, used):
    when = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    prof = "; ".join(f"{chr(66 + i)} {nm} ({-vv[0][0]:g}°W to {-vv[1][0]:g}°W)" for i, (nm, vv) in enumerate(PAPER_PROFILES))
    st = st4.round(2).to_html(index=False, border=0) if len(st4) else "<p>no overlap</p>"
    caps = {
        "M1": _caption_m1(M, C, ks),
        "M2": (f"Vp/Vs along four margin-normal profiles of {M.version} (B-E; located in A), with Vp contours (6.0 to 8.0 "
               f"km/s every 0.5), catalogue {C.catalog.name} tiers {C.tiers} events within ±10 km (dots), the Slab2 top "
               f"(dashed blue) and the seafloor or ground (black line; water light blue). Profiles: {prof}. Same mask as "
               f"Fig. M1 (Vp/Vs where both Vp and Vs are shown; black outline: checkerboard-resolved); vertical exaggeration "
               f"about 1.6."),
        "SM3": (f"Resolution of {M.version} at {', '.join(f'{M.z[k]:g}' for k in ks)} km. A-B, E-F, I-J: finest checkerboard "
                f"cell (40, 60 or 100 km, ±8 % patterns in depth bands -10/12/25/40/65 km) recovered with local correlation "
                f"≥ 0.7 and amplitude ≥ 0.5 (D-B018), finest over the covering tiles, P and S. C-D, G-H, K-L: merge-weighted "
                f"derivative weight sum, P and S (white contour: {M.dws_min:g}). Triangles in C: stations."),
        "SM4": (f"S-wave velocity of {M.version} (first column, mask of Fig. M1) and of "
                f"{' and '.join(used)} at {', '.join(f'{M.z[k]:g}' for k in ks)} km below sea level, and the difference "
                f"100 (Vs/Vs_prior − 1) at the shown nodes of this study. Prior models sampled at the same depth "
                f"(linear in depth) on their own grids, with the Vs colour range of the first column of each row; number of "
                f"compared nodes and median difference per panel in the upper left. Depths in km below sea level."),
    }
    titles = {"M1": "Fig. M1 — model overview (Vp, Vs, Vp/Vs at 3 depths)", "M2": "Fig. M2 — margin-normal cross-sections",
              "SM3": "Fig. S-M3 — resolution (checkerboard, DWS)", "SM4": "Fig. S-M4 — comparison with prior Vs models"}
    def size(f):
        h, w = plt.imread(f).shape[:2]
        return f"{w / 300 * 2.54:.1f} × {h / 300 * 2.54:.1f} cm"
    body = [f"<h1>{M.version} paper-figure candidates</h1><p class='mut'>Made {when} by "
            f"<code>workflow/08_figures/model_figures.py paper</code> (branch paper/model-figures, {git_head()}). "
            f"Candidates for the PI to choose from; nothing is in paper_sciadv/figures/. Files (PNG 300 dpi + PDF): "
            f"<code>{pdir}</code>. Science Advances widths: 1 column 8.6 cm, 2 columns 17.8 cm (all candidates 17.8 cm).</p>",
            "<p><b>Colour map choice</b>: left = matplotlib <code>seismic</code> (atlas, PI choice), right = Crameri "
            "<code>vik</code> (perceptually uniform, colour-blind safe; from GMT's SCM CPT). Both: slow = red, fast = blue; "
            "Vp/Vs high = red. <code>seismic</code> is not perceptually uniform: its lightness jumps near the centre and "
            "saturates at the ends, so equal steps in velocity do not look equal.</p>",
            "<details><summary>Model, mask and inputs</summary>" + header_html(M, C) + "</details>"]
    for key in ("M1", "M2", "SM3", "SM4"):
        body.append(f"<h2>{titles[key]}</h2><p class='cap'><b>Caption draft (facts only):</b> {caps[key]}</p>")
        if key == "SM3":
            f = figs[("SM3", "-")]
            body.append(f"<p class='mut'>{f.name} ({size(f)}); no velocity colour map</p><img src='{_img(f)}' alt='{key}'>")
        else:
            body.append("<div class='pair'>" + "".join(
                f"<div><span class='tag'>{sc}</span> <span class='mut'>{figs[(key, sc)].name} ({size(figs[(key, sc)])})"
                f"</span><img src='{_img(figs[(key, sc)])}' alt='{key} {sc}'></div>" for sc in SCHEMES) + "</div>")
        if key == "SM4":
            body.append(f"<details><summary>Differences per depth (this study − prior, % of prior)</summary>{st}</details>")
    html = out / "paper_candidates.html"
    html.write_text(_html_page(f"{M.version} paper candidates", "\n".join(body)))
    print(f"wrote {html} ({html.stat().st_size / 1e6:.1f} MB)")
    return html


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["gallery", "paper", "all"])
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--catalog", default=str(CATALOG), help="hypocentres (lat, lon, depth, tier); v6.0: "
                    "/wd1/mdenolle_data/reloc_v6/catalog/catalog_b4_v6.0.csv")
    ap.add_argument("--tiers", default="ABC")
    ap.add_argument("--stations", default=str(STATIONS))
    ap.add_argument("--mask", default="either", choices=["either", "checker", "dws"])
    ap.add_argument("--dws-min", type=float, default=100.0)
    ap.add_argument("--out", default=None, help="default /wd1/mdenolle_data/tomo_v1/models/figures_<version>")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    M = Model(a.model, a.mask, a.dws_min)
    out = Path(a.out) if a.out else Path(a.model).resolve().parent.parent / f"figures_{M.version}"
    out.mkdir(parents=True, exist_ok=True)
    C = Context(a.catalog, a.tiers, a.stations, cache=out / "cache")
    if a.what in ("gallery", "all"):
        gallery(M, C, out, jobs=min(a.jobs, 4))
    if a.what in ("paper", "all"):
        paper(M, C, out, a)


if __name__ == "__main__":
    main()

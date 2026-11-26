#!/usr/bin/env python3
"""Figures of a cascadia_obs_tomo velocity model (Track B): exploration gallery and paper-figure candidates.

    python workflow/08_figures/model_figures.py gallery          # one self-contained HTML, maps / sections / resolution
    python workflow/08_figures/model_figures.py paper            # Fig M1, M2, S-M3, S-M4 candidates + review HTML
    python workflow/08_figures/model_figures.py palette          # palette_sheet.html (colour maps under CVD) + atlas_ramps.js
    python workflow/08_figures/model_figures.py all
    options: --model PATH (default v0.4.2.nc)  --catalog PATH (default v6.0)  --tiers ABC
             --mask either|checker|dws  --dws-min 100  --out DIR  --jobs 4  --cmaps dv=vik_r,v=batlow_r,vpvs=roma_r

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

Colour maps (D-B104): Crameri Scientific Colour Maps read from GMT's SCM CPTs (scm_palettes.py; no cmcrameri):
Vp, Vs batlow reversed (sequential, slow warm, fast cool), Vp/Vs roma reversed (diverging about 1.75, high warm),
dVp, dVs and differences vik reversed (diverging about 0, slow warm). `palette` shows the candidates under normal
vision, deuteranopia, protanopia, tritanopia (Machado et al. 2009) and greyscale, and writes the atlas ramps.
Paper look (paper_candidates_v2): shaded relief base, colour only on shown cells, haloed event dots, cased Slab2,
one colour bar per variable shared by all depths, Science Advances widths (17.8 cm) and heights (<= 20 cm).

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scm_palettes import (VIEW_LABEL, VIEWS, js_ramp, js_ramp_error, palette_metrics, scm_cmap,  # noqa: E402
                          scm_colors, simulate)

ROOT = Path(__file__).resolve().parents[2]
MAIN = Path("/home/mdenolle/cascadia_obs_ensemble")           # git-ignored data live in the main tree (read only)
MODEL = Path("/wd1/mdenolle_data/tomo_v1/models/cascadia_obs_tomo/v0.4.2.nc")
CATALOG = Path("/wd1/mdenolle_data/catalog_v6/v6.0/cascadia_catalog_v6.0.csv")
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


# Colour maps (D-B104, PI 2026-11-26: colour-blind friendly and beautiful; latest model only). Variable groups:
# dv = dVp, dVs (diverging about 0, slow = warm); v = Vp, Vs (sequential, slow = warm, fast = cool);
# vpvs = Vp/Vs (diverging about 1.75, high = warm, distinct from dv). (SCM name, reversed). The palette sheet
# (`palette`) shows the candidates under normal vision, deuteranopia, protanopia, tritanopia and greyscale.
GROUP = {"vp": "v", "vs": "v", "dvp": "dv", "dvs": "dv", "vpvs": "vpvs"}
PAL = {"dv": ("vik", True), "v": ("batlow", True), "vpvs": ("roma", True)}
PAL_CANDIDATES = {"dv": [("vik", True), ("roma", False), ("broc", True)],
                  "v": [("batlow", True), ("lipari", True), ("hawaii", False)],
                  "vpvs": [("roma", True), ("bam", True), ("broc", False)]}
PAL_TEXT = {"dv": "dVp, dVs (diverging, centred on 0)", "v": "Vp, Vs (sequential)",
            "vpvs": "Vp/Vs (diverging, centred on 1.75)"}


def pal_name(nm_rev):
    nm, rev = nm_rev
    return f"{nm}{' reversed' if rev else ''}"


def cmap_for(key, pal=None):
    name, rev = (pal or PAL)[GROUP[key]]
    return scm_cmap(name, rev)


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


# ------------------------------------------------------------------ paper look (v2): relief base, haloed events, cased lines
EV_CORE, EV_HALO = "#151515", "white"
SLAB_PEN = dict(color="0.05", lw=0.55, ls=(0, (3.2, 1.6)))
LAND, SEA = np.array([0.890, 0.882, 0.866]), np.array([0.800, 0.822, 0.842])   # base tones, shaded by relief


def relief_rgba(C, region, ve=6.0, az=315.0, alt=40.0, strength=0.30):
    """Light neutral shaded relief (land slightly lighter than sea) on the map grid of C.relief, cropped to region."""
    from matplotlib.colors import LightSource
    R = C.relief
    W, E, S, N = region
    i = (R["lat"] >= S - 0.05) & (R["lat"] <= N + 0.05)
    j = (R["lon"] >= W - 0.05) & (R["lon"] <= E + 0.05)
    z = R["z"][np.ix_(i, j)].astype(float)
    lat, lon = R["lat"][i], R["lon"][j]
    dy = (lat[1] - lat[0]) * KM
    dx = (lon[1] - lon[0]) * KM * np.cos(np.radians(0.5 * (S + N)))
    hs = LightSource(azdeg=az, altdeg=alt).hillshade(z, vert_exag=ve, dx=dx, dy=dy, fraction=1.0)
    base = np.where((z >= 0)[..., None], LAND, SEA)
    shade = 1.0 - strength * (1.0 - hs)
    rgb = np.clip(base * shade[..., None] / np.percentile(shade, 70), 0, 1)       # typical slope keeps the base tone
    ext = [lon[0] - (lon[1] - lon[0]) / 2, lon[-1] + (lon[1] - lon[0]) / 2,
           lat[0] - (lat[1] - lat[0]) / 2, lat[-1] + (lat[1] - lat[0]) / 2]
    return np.dstack([rgb, np.ones(z.shape)]), ext


def draw_relief(ax, C, region, zorder=0.5):
    if C.relief is None:
        ax.set_facecolor("0.88")
        return
    img, ext = relief_rgba(C, region)
    ax.imshow(img, extent=ext, origin="lower", interpolation="bilinear", zorder=zorder, aspect=ax.get_aspect())


def draw_events(ax, x, y, size=0.14, halo=0.42, alpha=0.9, zorder=5.0):
    """Small dark dots with a light halo: readable on light and dark colours alike (rasterized in the PDF)."""
    if len(x) == 0:
        return
    ax.scatter(x, y, s=halo, c=EV_HALO, alpha=0.6, linewidths=0, zorder=zorder, rasterized=True)
    ax.scatter(x, y, s=size, c=EV_CORE, alpha=alpha, linewidths=0, zorder=zorder + 0.05, rasterized=True)


def cased(ax, x, y, color="0.05", lw=0.55, ls="-", casing=1.4, zorder=6.0, **kw):
    """A line with a thin white casing underneath, so it reads over any colour."""
    ax.plot(x, y, color="white", lw=lw + casing, solid_capstyle="round", alpha=0.85, zorder=zorder, **kw)
    ax.plot(x, y, color=color, lw=lw, ls=ls, zorder=zorder + 0.01, **kw)


def contour_lines(x, y, G, levels, min_len=None, metric=None):
    """Contour lines of G (NaN = not shown; no line enters a NaN cell) as {level: [N x 2 arrays]}.
    Pieces shorter than min_len (in `metric` units: a function of an N x 2 array) are dropped."""
    import contourpy
    gen = contourpy.contour_generator(x, y, np.ma.masked_invalid(G), name="serial", line_type="Separate")
    out = {}
    for lv in levels:
        segs = [s for s in gen.lines(lv) if len(s) > 1]
        if min_len is not None and metric is not None:
            segs = [s for s in segs if metric(s) >= min_len]
        if segs:
            out[lv] = segs
    return out


def draw_contours(ax, lines, color="0.08", lw=0.35, label_fmt=None, fs=4.5, zorder=4.5, label_min=0):
    import matplotlib.patheffects as pe
    for lv, segs in lines.items():
        ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, zorder=zorder))
        if label_fmt is None:
            continue
        s = max(segs, key=len)
        if len(s) < max(label_min, 3):
            continue
        m = len(s) // 2
        a, b = s[max(m - 2, 0)], s[min(m + 2, len(s) - 1)]
        p0 = ax.transData.transform(a); p1 = ax.transData.transform(b)
        ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
        ang = ang - 180 if ang > 90 else (ang + 180 if ang < -90 else ang)
        ax.text(s[m, 0], s[m, 1], label_fmt.format(lv), fontsize=fs, rotation=ang, rotation_mode="anchor",
                ha="center", va="center", color=color, zorder=zorder + 0.1,
                path_effects=[pe.withStroke(linewidth=1.2, foreground="white")])


def panel_letter(ax, letter, x=0.03, y=0.97, fs=8, ha="left", va="top"):
    import matplotlib.patheffects as pe
    ax.text(x, y, letter, transform=ax.transAxes, ha=ha, va=va, fontsize=fs, fontweight="bold", zorder=20,
            path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])


def halo_text(ax, x, y, s, transform=None, fs=6, color="0.05", weight="normal", lw=1.6, **kw):
    import matplotlib.patheffects as pe
    return ax.text(x, y, s, transform=transform or ax.transData, fontsize=fs, color=color, fontweight=weight,
                   path_effects=[pe.withStroke(linewidth=lw, foreground="white")], **kw)


def draw_context2(ax, C, region, lw=0.3, teeth_km=55, plates=True, borders=True, coast_z=1.9):
    """Coast, borders and PB2002 boundaries for the paper look (thin, dark grey; trench teeth small). The coast and
    borders sit under the colour fields (coast_z < 2), so they mark the relief but do not hatch the model."""
    ax.add_collection(LineCollection(C.coast, colors="0.32", linewidths=lw * 0.85, zorder=coast_z))
    if borders:
        ax.add_collection(LineCollection(C.borders, colors="0.5", linewidths=lw * 0.6, linestyles=(0, (2.5, 1.5)),
                                         zorder=coast_z))
    if plates:
        for key, lwk in (("OSR", 1.7), ("TF", 1.2), ("SUB", 1.2)):
            for L in C.plates[key]:
                ax.plot(L[:, 0], L[:, 1], color="0.15", lw=lw * lwk, zorder=7, solid_capstyle="round")
        for L in C.plates["SUB"]:
            L = L[np.argsort(-L[:, 1])]
            px, py, ss = profile_points(L, 1.0)
            for j in range(int(teeth_km) // 2, len(ss) - 1, max(1, int(teeth_km))):
                ax.plot(px[j] + 0.05, py[j], marker=">", ms=1.9, color="0.15", mew=0, zorder=7)


def map_layer2(ax, M, C, key, k, region, cmap, norm, step=0.02, events=True, ev=(0.14, 0.42), chk_outline=True,
               slab=True, ctx_lw=0.3, field=None, valid=None, relief=True, edge=True, teeth_km=55):
    """Paper look: relief base, shown cells coloured (opaque), a hairline at the edge of the shown cells, black
    outline around the checkerboard-resolved cells, haloed events of the node's depth layer, Slab2 at that depth."""
    W, E, S, N = region
    lonq, latq = map_grid(region, step)
    LON, LAT = np.meshgrid(lonq, latq)
    Z = np.full(LON.shape, M.z[k])
    G = M.sample(key, Z, LAT, LON, field=field, valid=valid)
    ext = [lonq[0] - step / 2, lonq[-1] + step / 2, latq[0] - step / 2, latq[-1] + step / 2]
    if relief:
        draw_relief(ax, C, region)
    else:
        ax.set_facecolor("0.88")
    im = ax.imshow(np.ma.masked_invalid(G), extent=ext, origin="lower", cmap=cmap, norm=norm, interpolation="nearest",
                   zorder=2, aspect=ax.get_aspect())
    vv = M.valid[key] if valid is None else valid
    if edge and np.isfinite(G).any():
        ax.contour(LON, LAT, np.isfinite(G).astype(float), levels=[0.5], colors="0.3", linewidths=0.25, zorder=3)
    if chk_outline:
        cg = M.nearest((M.chk[key] & vv).astype(float), Z, LAT, LON)
        if np.nansum(cg) > 0:
            ax.contour(LON, LAT, np.nan_to_num(cg), levels=[0.5], colors="k", linewidths=0.55, zorder=4)
    if slab and C.slab is not None:
        s = C.slab
        gen_lines = contour_lines(s["lon"], s["lat"], s["dep"], [M.z[k]])
        for seg in gen_lines.get(M.z[k], []):
            cased(ax, seg[:, 0], seg[:, 1], color=SLAB_PEN["color"], lw=SLAB_PEN["lw"], ls=SLAB_PEN["ls"], casing=1.1,
                  zorder=5.5)
    if events:
        lo, hi = layer_bounds(M.z, k)
        e = C.ev[(C.ev.depth >= lo) & (C.ev.depth < hi) & C.ev.lon.between(W, E) & C.ev.lat.between(S, N)]
        draw_events(ax, e.lon.to_numpy(), e.lat.to_numpy(), size=ev[0], halo=ev[1])
    draw_context2(ax, C, region, lw=ctx_lw, teeth_km=teeth_km)
    ax.set_xlim(W, E); ax.set_ylim(S, N)
    return im


def cbar2(fig, im, cax, label, ticks, orientation="vertical", fs=6, extend=None, fmt="{:g}"):
    cb = fig.colorbar(im, cax=cax, orientation=orientation, ticks=ticks, extend=extend or "neither",
                      extendfrac=0.045)
    cb.outline.set_linewidth(0.3)
    cb.ax.tick_params(length=1.6, width=0.35, pad=1.2, labelsize=fs - 0.5)
    lab = [fmt.format(t).replace("-", "−") for t in ticks]
    (cb.ax.set_yticklabels if orientation == "vertical" else cb.ax.set_xticklabels)(lab)
    cb.set_label(label, fontsize=fs, labelpad=2)
    cb.solids.set_rasterized(True)
    return cb


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
            cs = ax.contour(s["lon"], s["lat"], np.ma.masked_invalid(s["dep"]), levels=[M.z[k]], colors="k",
                            linewidths=0.7, linestyles=(0, (4, 2)), zorder=5)
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
    return dict(s=s, z=zz, lon=plon, lat=plat, f=out, rock=rock, below=~above, topo=topo, slab=C.slab_at(plon, plat),
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
        ax.plot(s, S["slab"], color="k", lw=0.8, ls=(0, (4, 2)), zorder=6)
        if label_slab:
            j = np.nanargmax(np.where(np.isfinite(S["slab"]) & (S["slab"] <= zmax - 3), S["slab"], -np.inf))
            ax.text(s[j], S["slab"][j], "Slab2", color="k", fontsize=5, ha="right", va="top", zorder=7)
    ax.set_xlim(s[0], s[-1]); ax.set_ylim(zmax, zz[0])
    return im


def _seg_km(seg):
    return float(np.hypot(*np.diff(seg, axis=0).T).sum())


VP_SEC_LEVELS = [6.0, 6.5, 7.0, 7.5, 8.0]


def section_panel2(ax, S, key, cmap, norm, zmax, ev=(0.10, 0.34), vp_contours=True, slab=True, grey="0.9",
                   water="#dbe6ef", edge=True, chk_outline=True, min_contour_km=10.0, label_fs=4.3):
    """Paper look of a section: rock not shown light grey, water pale blue, air white; colour only on shown cells;
    hairline at the edge of the shown cells, black outline around checkerboard-resolved cells; Vp contours
    (VP_SEC_LEVELS) drawn only through cells where Vp is shown, pieces shorter than min_contour_km dropped;
    haloed events; Slab2 top dashed and cased."""
    s, zz = S["s"], S["z"]
    ext = [s[0], s[-1], zz[-1], zz[0]]
    G = S["f"][key]
    ax.set_facecolor("white")
    gr = np.zeros(G.shape + (4,))
    gr[S["below"] & ~np.isfinite(G)] = matplotlib.colors.to_rgba(grey)        # below the seafloor or ground, not shown
    ax.imshow(gr, extent=ext, aspect="auto", interpolation="nearest", zorder=1)
    im = ax.imshow(np.ma.masked_invalid(G), extent=ext, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest",
                   zorder=2)
    topo = S["topo"]
    ax.fill_between(s, 0, np.maximum(-topo, 0), color=water, lw=0, zorder=1.5)
    ax.plot(s, -topo, color="0.15", lw=0.45, zorder=6)
    if edge and np.isfinite(G).any():
        ax.contour(s, zz, np.isfinite(G).astype(float), levels=[0.5], colors="0.3", linewidths=0.25, zorder=3)
    if chk_outline and S["chk"][key].any():
        ax.contour(s, zz, S["chk"][key].astype(float), levels=[0.5], colors="k", linewidths=0.55, zorder=4)
    if vp_contours and "vp" in S["f"] and np.isfinite(S["f"]["vp"]).sum() > 4:
        lines = contour_lines(s, zz, S["f"]["vp"], VP_SEC_LEVELS, min_len=min_contour_km, metric=_seg_km)
        draw_contours(ax, lines, lw=0.35, label_fmt="{:.1f}", fs=label_fs, label_min=12)
    draw_events(ax, S["es"], S["ez"], size=ev[0], halo=ev[1])
    if slab and np.isfinite(S["slab"]).any():
        cased(ax, s, S["slab"], color=SLAB_PEN["color"], lw=SLAB_PEN["lw"] + 0.1, ls=SLAB_PEN["ls"], casing=1.1,
              zorder=6.5)
    ax.set_xlim(s[0], s[-1]); ax.set_ylim(zmax, zz[0])
    return im


# ------------------------------------------------------------------ palette sheet (D-B104)
PAL_PANEL = {   # the real panel each variable group is judged on
    "dv": dict(kind="map", key="dvp", depth=18.0, region=[-124.9, -121.0, 45.2, 49.0], norm=(-8.0, 8.0),
               ticks=[-8, -4, 0, 4, 8], label="dVp (%)", what="dVp at 18 km, Washington"),
    "v": dict(kind="map", key="vp", depth=29.0, region=[-127.7, -122.7, 39.95, 42.25], norm=(5.0, 8.5),
              ticks=[5, 6, 7, 8], label="Vp (km/s)", what="Vp at 29 km, Mendocino (range of Fig. M1)"),
    "vpvs": dict(kind="section", key="vpvs", prof=0, norm=(1.60, 1.90), ticks=[1.6, 1.7, 1.8, 1.9], label="Vp/Vs",
                 what="Vp/Vs section B, Mendocino 40.4°N (Fig. M2)"),
}


def _fig_array(fig, dpi):
    fig.set_dpi(dpi)
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(float) / 255.0
    plt.close(fig)
    return a


def _pal_panel(M, C, grp, nm_rev, dpi=250, S=None):
    P = PAL_PANEL[grp]
    style(6)
    cm = scm_cmap(*nm_rev)
    norm = Normalize(*P["norm"])
    if P["kind"] == "map":
        W, E, Sx, N = P["region"]
        asp = (N - Sx) / ((E - W) * np.cos(np.radians((Sx + N) / 2)))
        w = 6.2
        aw = w - 1.25
        h = aw * asp + 1.95
        fig = plt.figure(figsize=(w * CM, h * CM), dpi=dpi)
        ax = _cm_axes(fig, w, h, 0.85, 1.55, aw, aw * asp)
        setup_map(ax, P["region"], step=1.0)
        k = int(np.argmin(np.abs(M.z - P["depth"])))
        im = map_layer2(ax, M, C, P["key"], k, P["region"], cm, norm, step=0.0125, teeth_km=35)
        cax = _cm_axes(fig, w, h, 0.85 + 0.15 * aw, 0.62, 0.7 * aw, 0.16)
    else:
        nm, vv = PAPER_PROFILES[P["prof"]]
        w, h = 8.0, 4.9
        aw = w - 1.2
        fig = plt.figure(figsize=(w * CM, h * CM), dpi=dpi)
        ax = _cm_axes(fig, w, h, 0.85, 1.65, aw, h - 1.8)
        S = S if S is not None else sample_section(M, C, vv, zmin=-3.0, zmax=45.0, dx=0.5, dz=0.2)
        im = section_panel2(ax, S, P["key"], cm, norm, 45.0)
        ax.set_xlim(0, float(S["s"][-1])); ax.set_ylabel("depth (km)", labelpad=1)
        ax.tick_params(length=1.5, pad=1)
        cax = _cm_axes(fig, w, h, 0.85 + 0.2 * aw, 0.55, 0.6 * aw, 0.16)
    cbar2(fig, im, cax, P["label"], P["ticks"], orientation="horizontal", fs=6)
    return _fig_array(fig, dpi)


def _pal_strips(cands, diverging):
    """Colour bars of every candidate under every view, and the lightness L* along each map."""
    from scm_palettes import rgb_to_lab
    style(6)
    n = len(cands)
    w, rh = 18.0, 1.05
    h = 0.7 + n * rh
    fig = plt.figure(figsize=(w * CM, h * CM))
    for i, nm_rev in enumerate(cands):
        c = scm_colors(*nm_rev)
        y = h - 0.55 - (i + 1) * rh + 0.25
        for j, v in enumerate(VIEWS):
            ax = _cm_axes(fig, w, h, 2.3 + j * 2.45, y + 0.15, 2.3, 0.42)
            ax.imshow(simulate(c, v)[None], aspect="auto", extent=[0, 1, 0, 1])
            ax.set_xticks([]); ax.set_yticks([])
            for sp_ in ax.spines.values():
                sp_.set_linewidth(0.3)
            if i == 0:
                ax.set_title(VIEW_LABEL[v], fontsize=6, pad=2)
        fig.text(2.2 / w, (y + 0.36) / h, pal_name(nm_rev), ha="right", va="center", fontsize=6.5, fontweight="bold")
        ax = _cm_axes(fig, w, h, 2.3 + 5 * 2.45 + 0.35, y + 0.02, 2.75, 0.72)
        L = rgb_to_lab(c)[:, 0]
        Lg = rgb_to_lab(simulate(c, "grey"))[:, 0]
        x = np.linspace(0, 1, len(L))
        ax.scatter(x, L, c=c, s=0.6, linewidths=0)
        ax.plot(x, Lg, color="0.4", lw=0.3, ls=":")
        ax.set_ylim(0, 100); ax.set_xlim(0, 1); ax.set_yticks([0, 50, 100]); ax.set_xticks([])
        ax.tick_params(length=1.2, pad=0.8, labelsize=4.5)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
        if i == 0:
            ax.set_title("lightness L* along the map", fontsize=6, pad=2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220)
    plt.close(fig)
    return buf.getvalue()


def _png_bytes(a):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray((np.clip(a, 0, 1) * 255).round().astype(np.uint8)).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _b64(b):
    return "data:image/png;base64," + base64.b64encode(b).decode()


PAL_CHOICE = {   # one line each, written after reading the sheet (numbers from its tables, v0.4.2 run 2026-11-26)
    "dv": "Zero change is a near-neutral light grey (L* 91), so 'no change' carries no hue; the slow red-brown and fast "
          "blue arms stay apart under deuteranopia, protanopia and tritanopia (arms ΔE 46, 36, 40) and the weakest 1/8 "
          "step stays ≥ 11.8 in every dichromacy. roma puts a pale green on zero; broc's arms merge under tritanopia "
          "(ΔE 9.6).",
    "v": "Lightness falls monotonically from slow (L* 87, pale pink) to fast (L* 12, dark blue), so the order holds in "
         "greyscale; best weakest 1/8 step under deuteranopia (8.4; lipari 6.3, hawaii 7.2) with protanopia 6.5 and "
         "tritanopia 6.2; event dots stay ≥ 19.7 ΔE from every colour (lipari 11.1: its fast end is near black). "
         "hawaii does better under protanopia (8.8) but drops to 4.7 under tritanopia and puts slow at the dark end.",
    "vpvs": "Strongest colour-blind numbers of the diverging candidates (weakest 1/8 step ≥ 11.5 and arms ≥ 45 in every "
            "dichromacy; bam 8.7 and 29.5 under deuteranopia, broc arms 10.0 under tritanopia) and the most even steps "
            "(uniformity 0.19); high Vp/Vs warm ochre-brown, low blue; its pale yellow-green centre and cyan-to-ochre "
            "arms keep it visibly apart from the grey-centred vik of the dVp, dVs maps.",
}


def palette_sheet(M, C, out, cands=None, dpi=230):
    cands = cands or PAL_CANDIDATES
    S0 = sample_section(M, C, PAPER_PROFILES[PAL_PANEL["vpvs"]["prof"]][1], zmin=-3.0, zmax=45.0, dx=0.5, dz=0.2)
    when = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    body = [f"<h1>Velocity-model colour maps: palette sheet ({M.version})</h1>"
            f"<p class='mut'>Made {when} by <code>workflow/08_figures/model_figures.py palette</code> (branch "
            f"paper/model-figures, {git_head()}). PI request D-B104: colour-blind friendly and beautiful, latest model "
            f"only. Candidates are Crameri Scientific Colour Maps v8 (Crameri 2018; Crameri, Shephard &amp; Heron 2020), "
            f"read from GMT's SCM CPTs.</p>",
            "<p><b>Convention kept</b>: slow and low velocity warm, fast cool; high Vp/Vs warm. <b>Views</b>: each real "
            "panel was drawn once and the whole image (colours, events, lines) was then transformed: deuteranopia, "
            "protanopia and tritanopia with the Machado, Oliveira &amp; Fernandes (2009) matrices at severity 1 "
            "(complete dichromacy, the worst case), applied in linear RGB; greyscale as the neutral grey of the same CIE "
            "luminance. Colour differences are CIEDE2000 (&Delta;E; about 1 is just noticeable, 10 is an obvious step).</p>",
            "<p><b>Numbers</b> (table under each variable): <i>uniformity</i> = coefficient of variation of the &Delta;E "
            "between neighbouring colours (0 = perfectly even steps); <i>length</i> = summed &Delta;E along the map, as a "
            "fraction of normal vision (how much of the map's contrast survives); <i>weakest 1/8</i> = smallest &Delta;E "
            "between two colours one eighth of the range apart (can 9 classes be told apart; lower = a flat stretch); "
            "<i>arms</i> (diverging) = smallest &Delta;E between colours equally far either side of the centre, 1/4 to "
            "1/2 of the range out (can slow and fast anomalies of the same size be told apart); <i>dot</i> = smallest "
            "&Delta;E between the event-dot core (#151515) and the map (dots also carry a white halo). In greyscale a "
            "diverging map shows the size of an anomaly, not its sign: its two arms have equal lightness by design, so "
            "<i>arms grey</i> and <i>weakest 1/8 grey</i> are near 0 for all three.</p>"]
    order = ["dv", "v", "vpvs"]
    nav = "<nav>" + "".join(f"<a href='#{g}'>{PAL_TEXT[g]}</a>" for g in order) + "<a href='#choice'>Choice</a>" \
          "<a href='#atlas'>Atlas ramps</a></nav>"
    body.insert(1, nav)
    for g in order:
        P = PAL_PANEL[g]
        div = g != "v"
        body.append(f"<h2 id='{g}'>{PAL_TEXT[g]}</h2><p class='cap'>Real panel: {P['what']}, events of catalogue "
                    f"{C.catalog.name} tiers {C.tiers}. Rows: candidates; columns: views.</p>")
        body.append(f"<img src='{_b64(_pal_strips(cands[g], div))}' alt='colour bars {g}'>")
        rows = []
        grid = ["<div class='pgrid'>", "<div></div>"] + [f"<div class='vh'>{VIEW_LABEL[v]}</div>" for v in VIEWS]
        for nm_rev in cands[g]:
            img = _pal_panel(M, C, g, nm_rev, dpi=dpi, S=S0)
            grid.append(f"<div class='rh'>{pal_name(nm_rev)}</div>")
            for v in VIEWS:
                grid.append(f"<div><img src='{_b64(_png_bytes(simulate(img, v)))}' alt='{nm_rev[0]} {v}'></div>")
            m = palette_metrics(scm_colors(*nm_rev), diverging=div, dot=matplotlib.colors.to_rgb(EV_CORE))
            r = {"map": pal_name(nm_rev), "L* low/mid/high": f"{m['L_lo']:.0f} / {m['L_mid']:.0f} / {m['L_hi']:.0f}",
                 "uniformity": f"{m['uniformity_cv']:.2f}"}
            for v in VIEWS[1:]:
                r[f"length {v}"] = f"{m[f'lenfrac_{v}']:.2f}"
            for v in VIEWS:
                r[f"weakest 1/8 {v}"] = f"{m[f'weakest_{v}']:.1f}"
            if div:
                for v in VIEWS:
                    r[f"arms {v}"] = f"{m[f'arms_{v}']:.1f}"
            r["dot"] = f"{m['dot_min']:.1f}"
            rows.append(r)
        grid.append("</div>")
        body.append("\n".join(grid))
        body.append("<div class='scroll'>" + pd.DataFrame(rows).to_html(index=False, border=0) + "</div>")
    body.append("<h2 id='choice'>Choice</h2><ul>" + "".join(
        f"<li><b>{PAL_TEXT[g]}: {pal_name(PAL[g])}</b>. {PAL_CHOICE[g]}</li>" for g in order) + "</ul>")
    ramps_js, ramp_rows = atlas_ramps(M)
    (out / "atlas_ramps.js").write_text(ramps_js)
    body.append("<h2 id='atlas'>Atlas ramps</h2><p class='cap'>Written to <code>"
                f"{out / 'atlas_ramps.js'}</code> for the 3-D atlas (applied by session -18; the atlas is not edited "
                "here). 17 stops each, t from &minus;1 (low end of the display range) to +1 (high end), linear in sRGB "
                "between stops as in <code>rampCol</code>. Largest &Delta;E between the full map and its 17-stop ramp:"
                "</p>" + pd.DataFrame(ramp_rows).to_html(index=False, border=0) + f"<pre>{ramps_js}</pre>")
    css = ("<style>.pgrid{display:grid;grid-template-columns:110px repeat(5,minmax(0,1fr));gap:6px 6px;"
           "align-items:center}.pgrid img{border:none;margin:0}.vh{font-size:12.5px;text-align:center;color:#444}"
           ".rh{font-size:12.5px;font-weight:600;text-align:right;padding-right:4px}.scroll{overflow-x:auto}"
           "pre{font-size:11px;white-space:pre-wrap;background:#f6f6f6;padding:8px}</style>")
    html = out / "palette_sheet.html"
    html.write_text(_html_page(f"{M.version} palette sheet", css + "\n".join(body)))
    print(f"wrote {html} ({html.stat().st_size / 1e6:.1f} MB)")
    return html


ATLAS_VARS = [  # atlas key, unit, note, ends; display range = SHARED (the paper figures and the gallery)
    ("dvp", "%", "change from the 3-D start model, centre 0 (gallery)", ["slow", "fast"]),
    ("dvs", "%", "change from the 3-D start model, centre 0 (gallery)", ["slow", "fast"]),
    ("vp", "km/s", "Fig. M1 range, all depths", ["slow", "fast"]),
    ("vs", "km/s", "Fig. M1 and S-M4 range, all depths", ["slow", "fast"]),
    ("vpvs", "", "centre 1.75, Figs. M1 and M2", ["low", "high"]),
]


def atlas_ramps(M, n=17):
    when = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    consts, rows, objs = [], [], []
    for g in ("dv", "v", "vpvs"):
        c = scm_colors(*PAL[g])
        r = js_ramp(c, n)
        nm = {"dv": "RAMP_DV", "v": "RAMP_V", "vpvs": "RAMP_VPVS"}[g]
        kind = "sequential" if g == "v" else "diverging"
        txt = ", ".join(f"[{t:g}, {a}, {b}, {d}]" for t, a, b, d in r)
        consts.append(f"// {PAL_TEXT[g]}: Crameri {pal_name(PAL[g])}; t = -1 low end, +1 high end; largest CIEDE2000 "
                      f"between the full map and these {n} stops: {js_ramp_error(c, r):.2f}\n"
                      f"const {nm} = [{txt}];")
        rows.append({"ramp": nm, "map": pal_name(PAL[g]), "kind": kind, "stops": n,
                     "max ΔE00 to full map": f"{js_ramp_error(c, r):.2f}"})
    for key, unit, note, ends in ATLAS_VARS:
        g, (lo, hi) = GROUP[key], SHARED[key]["norm"]
        kind = "sequential" if g == "v" else "diverging"
        nm = {"dv": "RAMP_DV", "v": "RAMP_V", "vpvs": "RAMP_VPVS"}[g]
        cen = f", centre: {(lo + hi) / 2:g}" if kind == "diverging" else ""
        objs.append(f'  {key}: {{ key: "{key}", kind: "{kind}", map: "{pal_name(PAL[g])}", lo: {lo:g}, hi: {hi:g}{cen}, '
                    f'unit: "{unit}", ends: ["{ends[0]}", "{ends[1]}"], stops: {nm} }},   // {note}')
    js = (f"// atlas_ramps.js: colour ramps for the tomography layers of the 3-D atlas, model {M.version}.\n"
          f"// Made {when} by workflow/08_figures/model_figures.py palette (branch paper/model-figures, {git_head()}).\n"
          "// Crameri Scientific Colour Maps v8 (perceptually uniform, colour-vision-deficiency friendly), read from GMT's\n"
          "// SCM CPTs. Format of RAMP_RED_LOW in atlas_template.html: [[t, r, g, b], ...], t in [-1, 1], -1 = lo, +1 = hi,\n"
          "// RGB 0-255, linear in sRGB between stops (rampCol). Convention: slow / low velocity warm, fast cool; Vp/Vs high warm.\n"
          "// diverging: t = 0 (the centre colour) sits at (lo + hi) / 2, so lo and hi must stay symmetric about the\n"
          "//   centre given (0 % for dVp / dVs, 1.75 for Vp/Vs); sequential: no centre, lo / hi may move freely.\n"
          "// lo / hi: the display ranges of the paper-figure candidates and the gallery (paper_candidates_v2.html).\n"
          "// Changes from the atlas TVARS now: Vp/Vs 1.65-1.95 (centre 1.80) -> 1.60-1.90 (centre 1.75); dVp/dVs -6..6 ->\n"
          "// -8..8 %; Vp 4.5-8.2 -> 5.0-8.5 km/s; Vs 2.5-4.7 -> 2.9-4.9 km/s (keep the atlas ranges if preferred: only\n"
          "// the diverging centres matter for the colour meaning).\n"
          + "\n".join(consts) + "\nconst ATLAS_RAMPS = {\n" + "\n".join(objs) + "\n};\n")
    return js, rows


# ------------------------------------------------------------------ gallery
def _gal_depth(args):
    M, C, k, out, pal = args
    style(7)
    keys = ["vp", "vs", "vpvs", "dvp", "dvs"]
    fig, axs = plt.subplots(1, 5, figsize=(17.5, 6.3))
    fig.subplots_adjust(left=0.035, right=0.995, top=0.92, bottom=0.13, wspace=0.12)
    for j, (ax, key) in enumerate(zip(axs, keys)):
        setup_map(ax, REGION, left=(j == 0))
        norm = gallery_norm(M, key, k)
        im = map_layer(ax, M, C, key, k, REGION, cmap_for(key, pal), norm, stations=(key == "vp"), st_size=3)
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
                 f"dashed black: Slab2 at this depth)", fontsize=8.5, y=0.985)
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
    M, C, key, out, pal, zmax = args
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
        im = section_panel(ax, S, k, cmap_for(k, pal), norms[k], zmax)
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
<p><b>Colour</b> (D-B104; palette_sheet.html): Crameri {pal_name(PAL['v'])} for Vp and Vs (slow warm, fast cool),
{pal_name(PAL['vpvs'])} for Vp/Vs (centred on 1.75, high warm), {pal_name(PAL['dv'])} for dVp and dVs (centred on 0,
slow warm). Maps: Vp and Vs centred on the median of the inverted rock nodes of that depth (tile_weight &gt; 0, shown or not; range per depth from the 98th percentile of the shown nodes, on each colour bar);
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


def gallery(M, C, out, jobs=4, zmax_map=60.0, zmax_sec=65.0):
    gdir = out / "gallery_png"
    gdir.mkdir(parents=True, exist_ok=True)
    ks = [k for k in range(len(M.z)) if M.z[k] <= zmax_map]
    tasks_d = [(M, C, k, gdir, dict(PAL)) for k in ks]
    tasks_r = [(M, C, k, gdir) for k in ks]
    tasks_s = [(M, C, key, gdir, dict(PAL), zmax_sec) for key in PROFILES]
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


# ------------------------------------------------------------------ paper candidates (v2: D-B104 colour maps, Science Advances)
PAPER_REGION = [-130.9, -121.2, 39.95, 49.1]
PAPER_DEPTHS = [9.0, 18.0, 29.0]        # node depths near 8 / 20 / 35 km with checkerboard-resolved nodes (21, 34: none)
M1B_DEPTHS = [18.0, 29.0]               # two-depth variant: the depths with most shown nodes
# M2 profiles: margin-normal, 4 deg of longitude each (same km scale), through the shown Vp/Vs
PAPER_PROFILES = [("Mendocino 40.4°N", [(-126.4, 40.4), (-122.4, 40.4)]),
                  ("Gorda 40.9°N", [(-126.4, 40.9), (-122.4, 40.9)]),
                  ("Blanco 43.5°N", [(-129.4, 43.5), (-125.4, 43.5)]),
                  ("S. Washington 46.4°N", [(-125.0, 46.4), (-121.0, 46.4)])]
PAPER_DIR = "paper_candidates_v2"
H_MAX = 19.9                            # cm, Science Advances page budget (at or under about 20 cm)
LAB = {"vp": "Vp (km/s)", "vs": "Vs (km/s)", "vpvs": "Vp/Vs", "dvp": "dVp (%)", "dvs": "dVs (%)"}
NAME = {"vp": "P-wave velocity", "vs": "S-wave velocity", "vpvs": "Vp/Vs"}
# one display range per variable for every depth (shared colour bars), also the atlas ranges (atlas_ramps.js)
SHARED = {"vp": dict(norm=(5.0, 8.5), ticks=[5, 6, 7, 8], fmt="{:g}"),
          "vs": dict(norm=(2.9, 4.9), ticks=[3.0, 3.5, 4.0, 4.5], fmt="{:.1f}"),
          "vpvs": dict(norm=(1.60, 1.90), ticks=[1.60, 1.75, 1.90], fmt="{:.2f}"),
          "dvp": dict(norm=(-8.0, 8.0), ticks=[-8, -4, 0, 4, 8], fmt="{:g}"),
          "dvs": dict(norm=(-8.0, 8.0), ticks=[-8, -4, 0, 4, 8], fmt="{:g}")}
DIFF_RANGE = (-15.0, 15.0)              # S-M4: this study - prior (%)


def _cm_axes(fig, W, H, x, y, w, h):
    """Axes from a layout in cm (origin bottom left)."""
    return fig.add_axes([x / W, y / H, w / W, h / H])


def _extend(M, key, ks, lo, hi):
    v = np.concatenate([M.f[key][k][M.valid[key][k]] for k in ks])
    a, b = (v.size and v.min() < lo - 1e-9), (v.size and v.max() > hi + 1e-9)
    return "both" if a and b else ("min" if a else ("max" if b else "neither"))


def _asp(region):
    W, E, S, N = region
    return (N - S) / ((E - W) * np.cos(np.radians((S + N) / 2)))


def _slab_label(ax, C, z, lat_at, dx=-0.12, fs=5.0):
    """Write 'Slab2' next to the Slab2 contour at depth z, near latitude lat_at."""
    if C.slab is None:
        return
    segs = contour_lines(C.slab["lon"], C.slab["lat"], C.slab["dep"], [z]).get(z, [])
    if not segs:
        return
    pts = np.concatenate(segs)
    j = int(np.argmin(np.abs(pts[:, 1] - lat_at)))
    halo_text(ax, pts[j, 0] + dx, pts[j, 1], f"Slab2\n{z:g} km", fs=fs, ha="right" if dx < 0 else "left", va="center",
              zorder=12, linespacing=0.95)


def fig_m1(M, C, out, depths=PAPER_DEPTHS, region=PAPER_REGION, name="figM1_overview"):
    """3 x 3: rows Vp, Vs, Vp/Vs; columns depths; one colour bar per variable (shared by the depths)."""
    style(6)
    asp = _asp(region)
    keys, nc = ("vp", "vs", "vpvs"), len(depths)
    Lmin, R, T, B, gx, gy = 0.95, 1.2, 0.42, 0.42, 0.12, 0.12
    ph = (H_MAX - T - B - 2 * gy) / 3
    pw = min(ph / asp, (COL2 - Lmin - R - (nc - 1) * gx) / nc)
    ph = pw * asp
    spare = COL2 - Lmin - R - nc * pw - (nc - 1) * gx
    L = Lmin + spare / 2
    H = T + B + 3 * ph + 2 * gy
    fig = plt.figure(figsize=(COL2 * CM, H * CM))
    ks = [int(np.argmin(np.abs(M.z - d))) for d in depths]
    letters = iter("ABCDEFGHIJKL")
    for r, key in enumerate(keys):
        y0 = H - T - (r + 1) * ph - r * gy
        norm = Normalize(*SHARED[key]["norm"])
        cm = cmap_for(key)
        for c, k in enumerate(ks):
            ax = _cm_axes(fig, COL2, H, L + c * (pw + gx), y0, pw, ph)
            setup_map(ax, region, step=2, left=(c == 0), bottom=(r == 2))
            im = map_layer2(ax, M, C, key, k, region, cm, norm)
            panel_letter(ax, next(letters))
            if r == 0:
                ax.set_title(f"{M.z[k]:g} km", fontsize=7, pad=3, fontweight="bold")
                if c == 0:
                    _slab_label(ax, C, M.z[k], 47.6, dx=-0.1)
        cax = _cm_axes(fig, COL2, H, L + nc * pw + (nc - 1) * gx + 0.2, y0 + 0.14 * ph, 0.17, 0.72 * ph)
        cbar2(fig, im, cax, LAB[key], SHARED[key]["ticks"], extend=_extend(M, key, ks, *SHARED[key]["norm"]),
              fmt=SHARED[key]["fmt"])
    f = out / f"{name}.png"
    save(fig, f, 300, pdf=True)
    return f


def fig_m1b(M, C, out, depths=M1B_DEPTHS, region=PAPER_REGION, name="figM1b_two_depths"):
    """2 x 3: rows depths; columns Vp, Vs, Vp/Vs; one colour bar per variable under its column."""
    style(6)
    asp = _asp(region)
    keys = ("vp", "vs", "vpvs")
    L, R, T, B, gx, gy = 0.95, 0.15, 0.42, 1.5, 0.12, 0.12
    pw = (COL2 - L - R - 2 * gx) / 3
    ph = pw * asp
    H = T + B + len(depths) * ph + (len(depths) - 1) * gy
    fig = plt.figure(figsize=(COL2 * CM, H * CM))
    ks = [int(np.argmin(np.abs(M.z - d))) for d in depths]
    letters = iter("ABCDEFGHI")
    ims = {}
    for r, k in enumerate(ks):
        for c, key in enumerate(keys):
            ax = _cm_axes(fig, COL2, H, L + c * (pw + gx), H - T - (r + 1) * ph - r * gy, pw, ph)
            setup_map(ax, region, step=2, left=(c == 0), bottom=(r == len(ks) - 1))
            ims[key] = map_layer2(ax, M, C, key, k, region, cmap_for(key), Normalize(*SHARED[key]["norm"]))
            panel_letter(ax, next(letters))
            halo_text(ax, 0.115, 0.965, f"{M.z[k]:g} km", transform=ax.transAxes, fs=7, weight="bold", ha="left",
                      va="top", zorder=12, lw=1.8)
            if r == 0:
                ax.set_title(NAME[key], fontsize=7, pad=3)
    for c, key in enumerate(keys):
        cax = _cm_axes(fig, COL2, H, L + c * (pw + gx) + 0.15 * pw, 0.62, 0.7 * pw, 0.15)
        cbar2(fig, ims[key], cax, LAB[key], SHARED[key]["ticks"], orientation="horizontal",
              extend=_extend(M, key, ks, *SHARED[key]["norm"]), fmt=SHARED[key]["fmt"])
    f = out / f"{name}.png"
    save(fig, f, 300, pdf=True)
    return f


def fig_m2(M, C, out, profiles=PAPER_PROFILES, zmax=45.0, xmax=345.0, name="figM2_sections"):
    style(6)
    L, R, T, B = 4.7, 0.15, 0.25, 0.75
    gy, n = 0.2, len(profiles)
    sw = COL2 - L - R
    sh = 1.6 * sw * (zmax + 3) / xmax                  # vertical exaggeration 1.6
    H = T + B + n * sh + (n - 1) * gy
    fig = plt.figure(figsize=(COL2 * CM, H * CM))
    cmap, norm = cmap_for("vpvs"), Normalize(*SHARED["vpvs"]["norm"])
    # locator: shaded relief, where Vp/Vs is shown at any depth (purple-grey), plate boundaries, profiles
    reg = [-130.0, -120.6, 39.6, 47.4]
    lw_ = L - 1.5
    lh = lw_ * _asp(reg)
    axm = _cm_axes(fig, COL2, H, 0.8, H - T - lh, lw_, lh)
    setup_map(axm, reg, step=4)
    draw_relief(axm, C, reg)
    lonq, latq = map_grid(reg, 0.025)
    LON, LAT = np.meshgrid(lonq, latq)
    sh_any = M.valid["vpvs"].any(0)
    g = M.nearest(sh_any[None].repeat(len(M.z), 0), np.full(LON.shape, 10.0), LAT, LON).astype(bool)
    axm.imshow(np.ma.masked_where(~g, g), extent=[lonq[0], lonq[-1], latq[0], latq[-1]], origin="lower",
               cmap=ListedColormap(["#7d6b9e"]), alpha=0.75, interpolation="nearest", aspect=axm.get_aspect(), zorder=2)
    draw_context2(axm, C, reg, lw=0.3, teeth_km=60)
    letters = "ABCDEFGH"
    for i, (nm, vv) in enumerate(profiles):
        vv = np.asarray(vv)
        cased(axm, vv[:, 0], vv[:, 1], color="0.05", lw=0.9, casing=1.2, zorder=9)
        halo_text(axm, vv[0, 0] - 0.12, vv[0, 1], letters[i + 1], fs=6.5, weight="bold", ha="right", va="center",
                  zorder=12)
    panel_letter(axm, "A")
    axm.set_xlim(reg[0], reg[1]); axm.set_ylim(reg[2], reg[3])
    for i, (nm, vv) in enumerate(profiles):
        y0 = H - T - (i + 1) * sh - i * gy
        Sx = sample_section(M, C, vv, zmin=-3.0, zmax=zmax, dx=0.5, dz=0.2, keys=("vp", "vs", "vpvs"))
        Lk = float(Sx["s"][-1])
        ax = _cm_axes(fig, COL2, H, L, y0, sw * min(Lk, xmax) / xmax, sh)
        im = section_panel2(ax, Sx, "vpvs", cmap, norm, zmax)
        ax.set_xlim(0, min(Lk, xmax))
        ax.set_xticks(np.arange(0, min(Lk, xmax) + 1e-6, 50))
        ax.set_yticks([0, 10, 20, 30, 40])
        ax.set_ylabel("depth (km)", labelpad=1.5)
        ax.tick_params(length=1.5, pad=1)
        if i < n - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("distance from the west end (km)", labelpad=1.5)
        panel_letter(ax, letters[i + 1], x=0.008, y=0.06, va="bottom")
        halo_text(ax, 0.032, 0.06, nm, transform=ax.transAxes, fs=6.5, va="bottom", zorder=12, lw=1.8)
        if i == 0:
            ve = (sh / (zmax + 3)) / (sw / xmax)
            halo_text(ax, 0.995, 0.05, f"V.E. {ve:.1f}", transform=ax.transAxes, fs=5, color="0.25", ha="right",
                      va="bottom", zorder=12)
            j = np.where(np.isfinite(Sx["slab"]) & (Sx["slab"] < zmax - 4))[0]
            if j.size:
                jj = j[-1]
                halo_text(ax, Sx["s"][jj] + 4, Sx["slab"][jj], "Slab2", fs=5, ha="left", va="center", zorder=12)
    # left column under the locator: colour bar and a compact legend
    ycb = H - T - lh - 1.05
    cax = _cm_axes(fig, COL2, H, 0.8 + 0.08 * lw_, ycb, 0.84 * lw_, 0.15)
    cbar2(fig, im, cax, "Vp/Vs", SHARED["vpvs"]["ticks"], orientation="horizontal", fmt=SHARED["vpvs"]["fmt"],
          extend=_extend(M, "vpvs", range(len(M.z)), *SHARED["vpvs"]["norm"]))
    _section_legend(fig, COL2, H, 0.8, ycb - 3.0, lw_, 2.35)
    f = out / f"{name}.png"
    save(fig, f, 300, pdf=True)
    return f


def _section_legend(fig, W, H, x, y, w, h, fs=5.5):
    """Legend of the section symbols, drawn with the same artists as the panels."""
    from matplotlib.patches import Rectangle
    ax = _cm_axes(fig, W, H, x, y, w, h)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    rows = ["events within ±10 km", "Slab2 top", "Vp contour (km/s)", "checkerboard-resolved", "not shown", "water"]
    ys = np.linspace(0.93, 0.07, len(rows))
    x0, x1 = 0.02, 0.17
    rng = np.random.default_rng(3)
    draw_events(ax, rng.uniform(x0 + 0.01, x1 - 0.01, 40), ys[0] + rng.normal(0, 0.015, 40), size=0.14, halo=0.42)
    cased(ax, [x0, x1], [ys[1], ys[1]], color=SLAB_PEN["color"], lw=SLAB_PEN["lw"] + 0.1, ls=SLAB_PEN["ls"], casing=1.1)
    ax.plot([x0, x1], [ys[2], ys[2]], color="0.08", lw=0.35)
    ax.add_patch(Rectangle((x0, ys[3] - 0.045), x1 - x0, 0.09, fc="none", ec="k", lw=0.55))
    ax.add_patch(Rectangle((x0, ys[4] - 0.045), x1 - x0, 0.09, fc="0.9", ec="none"))
    ax.add_patch(Rectangle((x0, ys[5] - 0.045), x1 - x0, 0.09, fc="#dbe6ef", ec="none"))
    for yy, t in zip(ys, rows):
        ax.text(x1 + 0.04, yy, t, fontsize=fs, va="center", ha="left")


RES_COLORS = scm_colors("bamako", False, n=256)[[40, 135, 225]]    # finest resolved cell 40 / 60 / 100 km: dark to light


def fig_sm3(M, C, out, depths=PAPER_DEPTHS, region=PAPER_REGION, name="figSM3_resolution"):
    style(6)
    asp = _asp(region)
    L, R, T, B, gx, gy = 0.95, 0.15, 0.42, 1.45, 0.1, 0.12
    pw = (COL2 - L - R - 3 * gx) / 4
    ph = pw * asp
    H = T + B + 3 * ph + 2 * gy
    fig = plt.figure(figsize=(COL2 * CM, H * CM))
    ks = [int(np.argmin(np.abs(M.z - d))) for d in depths]
    cm_res = ListedColormap(RES_COLORS)
    nres = BoundaryNorm([30, 50, 80, 120], cm_res.N)
    cm_dws = scm_cmap("devon", True)
    lonq, latq = map_grid(region, 0.02)
    LON, LAT = np.meshgrid(lonq, latq)
    ext = [lonq[0] - 0.01, lonq[-1] + 0.01, latq[0] - 0.01, latq[-1] + 0.01]
    letters = iter("ABCDEFGHIJKL")
    ims = {}
    for r, k in enumerate(ks):
        Z = np.full(LON.shape, M.z[k])
        for c, var in enumerate(("resolution_p", "resolution_s", "dws_p", "dws_s")):
            ax = _cm_axes(fig, COL2, H, L + c * (pw + gx), H - T - (r + 1) * ph - r * gy, pw, ph)
            setup_map(ax, region, step=2, left=(c == 0), bottom=(r == 2))
            draw_relief(ax, C, region)
            a = np.where(M.rock & (M.g[var] > 0), M.g[var].astype(float), np.nan)
            A = M.nearest(a, Z, LAT, LON)
            if var.startswith("res"):
                ims[c] = ax.imshow(np.ma.masked_invalid(A), extent=ext, origin="lower", cmap=cm_res, norm=nres,
                                   interpolation="nearest", aspect=ax.get_aspect(), zorder=2)
            else:
                ims[c] = ax.imshow(np.ma.masked_invalid(A), extent=ext, origin="lower", cmap=cm_dws,
                                   norm=LogNorm(1, 1e4), interpolation="nearest", aspect=ax.get_aspect(), zorder=2)
                if np.isfinite(A).any():
                    ax.contour(LON, LAT, np.nan_to_num(A), levels=[M.dws_min], colors="0.05", linewidths=0.5, zorder=4)
            draw_context2(ax, C, region, lw=0.28)
            n = int((M.rock[k] & (M.g[var][k] >= (1e-9 if var.startswith("res") else M.dws_min))).sum())
            halo_text(ax, 0.13, 0.965, f"{M.z[k]:g} km", transform=ax.transAxes, fs=6.5, weight="bold", ha="left",
                      va="top", zorder=12)
            halo_text(ax, 0.035, 0.895, f"{n} nodes" + ("" if var.startswith("res") else f" ≥ {M.dws_min:g}"),
                      transform=ax.transAxes, fs=5, ha="left", va="top", zorder=12)
            panel_letter(ax, next(letters))
            if r == 0:
                ax.set_title({"resolution_p": "checkerboard, P", "resolution_s": "checkerboard, S",
                              "dws_p": "DWS, P", "dws_s": "DWS, S"}[var], fontsize=7, pad=3)
    for c0, lab in ((0, "finest resolved checkerboard cell (km)"), (2, "derivative weight sum")):
        cax = _cm_axes(fig, COL2, H, L + c0 * (pw + gx) + 0.25 * pw, 0.62, 1.5 * pw + gx, 0.15)
        if c0 == 0:
            cb = fig.colorbar(ims[0], cax=cax, orientation="horizontal", ticks=[40, 65, 100])
            cb.ax.set_xticklabels(["40", "60", "100"])
        else:
            cb = fig.colorbar(ims[2], cax=cax, orientation="horizontal")
        cb.set_label(lab, fontsize=6, labelpad=2)
        cb.outline.set_linewidth(0.3); cb.ax.tick_params(length=1.6, width=0.35, pad=1.2, labelsize=5.5)
        cb.solids.set_rasterized(True)
    f = out / f"{name}.png"
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


def fig_sm4(M, C, out, depths=PAPER_DEPTHS, region=PAPER_REGION, name="figSM4_prior_models"):
    priors = {n: load_prior(p) for n, p in PRIORS.items()}
    for n, p in priors.items():
        if p is None and n not in " ".join(C.missing):
            C.missing.append(f"{n} (data/tomography; utils/fetch_tomography.py)")
    names = [n for n, p in priors.items() if p is not None]
    style(6)
    asp = _asp(region)
    ncol = 1 + 2 * len(names)
    L, R, T, B, gx, gy = 0.95, 0.15, 0.55, 1.45, 0.1, 0.12
    pw = (COL2 - L - R - (ncol - 1) * gx) / ncol
    ph = pw * asp
    H = T + B + 3 * ph + 2 * gy
    fig = plt.figure(figsize=(COL2 * CM, H * CM))
    ks = [int(np.argmin(np.abs(M.z - d))) for d in depths]
    lonq, latq = map_grid(region, 0.02)
    LON, LAT = np.meshgrid(lonq, latq)
    letters = iter("ABCDEFGHIJKLMNO")
    stats = []
    cmv, cmd = cmap_for("vs"), cmap_for("dvs")
    nv, nd = Normalize(*SHARED["vs"]["norm"]), Normalize(*DIFF_RANGE)
    lo_all, hi_all = [], []
    for r, k in enumerate(ks):
        z = M.z[k]
        Z = np.full(LON.shape, z)
        rockg = M.nearest(M.rock, Z, LAT, LON).astype(bool)
        col = 0
        panels = [("ours", None)] + [("prior", n) for n in names] + [("diff", n) for n in names]
        for kind, n in panels:
            ax = _cm_axes(fig, COL2, H, L + col * (pw + gx), H - T - (r + 1) * ph - r * gy, pw, ph)
            setup_map(ax, region, step=3, left=(col == 0), bottom=(r == 2))
            if kind == "ours":
                im = map_layer2(ax, M, C, "vs", k, region, cmv, nv, ev=(0.08, 0.26), ctx_lw=0.25, slab=False,
                                teeth_km=70)
                ttl = f"this study ({M.version})"
            elif kind == "prior":
                p = priors[n]
                G = sample_grid(np.nan_to_num(p["v"]), np.isfinite(p["v"]), (p["z"], p["lat"], p["lon"]), (Z, LAT, LON))
                G = np.where(rockg, G, np.nan)
                lo_all.append(np.nanmin(G)); hi_all.append(np.nanmax(G))
                draw_relief(ax, C, region)
                ax.imshow(np.ma.masked_invalid(G), extent=[lonq[0], lonq[-1], latq[0], latq[-1]], origin="lower",
                          cmap=cmv, norm=nv, interpolation="nearest", aspect=ax.get_aspect(), zorder=2)
                draw_context2(ax, C, region, lw=0.25, teeth_km=70)
                ttl = n
            else:
                p = priors[n]
                zz, yy, xx = np.meshgrid(M.z, M.lat, M.lon, indexing="ij")
                pv = sample_grid(np.nan_to_num(p["v"]), np.isfinite(p["v"]), (p["z"], p["lat"], p["lon"]),
                                 (zz[k], yy[k], xx[k]))
                d = np.full(M.f["vs"].shape, np.nan)
                d[k] = 100 * (M.f["vs"][k] / pv - 1)
                vd = M.valid["vs"] & np.isfinite(d)
                map_layer2(ax, M, C, "dvs", k, region, cmd, nd, events=False, chk_outline=False, slab=False,
                           ctx_lw=0.25, field=np.nan_to_num(d), valid=vd, teeth_km=70)
                dv = d[k][vd[k]]
                if dv.size:
                    stats.append(dict(depth_km=z, prior=n, n_nodes=int(dv.size), median_pct=float(np.median(dv)),
                                      mad_pct=float(np.median(np.abs(dv - np.median(dv)))),
                                      rms_pct=float(np.sqrt(np.mean(dv ** 2)))))
                    halo_text(ax, 0.04, 0.035, f"n = {dv.size}\nmedian {np.median(dv):+.1f} %".replace("-", "−"),
                              transform=ax.transAxes, fs=4.8, va="bottom", zorder=12, linespacing=1.0)
                else:
                    halo_text(ax, 0.5, 0.5, "no overlap", transform=ax.transAxes, ha="center", fs=5.5, color="0.3")
                ttl = f"this study − {n.split(' ')[0]}"
            halo_text(ax, 0.16, 0.965, f"{z:g} km", transform=ax.transAxes, fs=5.8, weight="bold", ha="left", va="top",
                      zorder=12)
            panel_letter(ax, next(letters), fs=7)
            if r == 0:
                ax.set_title(ttl, fontsize=6, pad=3)
            col += 1
    ours = np.concatenate([M.f["vs"][k][M.valid["vs"][k]] for k in ks])
    lo_x, hi_x = min([ours.min()] + lo_all), max([ours.max()] + hi_all)
    a_, b_ = lo_x < nv.vmin - 1e-9, hi_x > nv.vmax + 1e-9
    ext_v = "both" if a_ and b_ else ("min" if a_ else ("max" if b_ else "neither"))
    cax = _cm_axes(fig, COL2, H, L + 0.1 * pw, 0.62, (1 + len(names)) * (pw + gx) - gx - 0.2 * pw, 0.15)
    cbar2(fig, im, cax, "Vs (km/s)", SHARED["vs"]["ticks"], orientation="horizontal", extend=ext_v,
          fmt=SHARED["vs"]["fmt"])
    x0 = L + (1 + len(names)) * (pw + gx)
    cax = _cm_axes(fig, COL2, H, x0 + 0.1 * pw, 0.62, len(names) * (pw + gx) - gx - 0.2 * pw, 0.15)
    cbar2(fig, plt.cm.ScalarMappable(norm=nd, cmap=cmd), cax, "Vs difference, this study − prior (%)",
          [-15, -10, -5, 0, 5, 10, 15], orientation="horizontal", extend="both")
    f = out / f"{name}.png"
    save(fig, f, 300, pdf=True)
    pd.DataFrame(stats).to_csv(out / "figSM4_differences.csv", index=False, float_format="%.3f")
    return f, pd.DataFrame(stats), {n: str(p["path"]) for n, p in priors.items() if p is not None}


def _and(v):
    v = [f"{x}" for x in v]
    return ", ".join(v[:-1]) + f" and {v[-1]}" if len(v) > 1 else v[0]


def _ev_counts(M, C, ks, region):
    W, E, S, N = region
    out = []
    for k in ks:
        lo, hi = layer_bounds(M.z, k)
        e = C.ev[(C.ev.depth >= lo) & (C.ev.depth < hi) & C.ev.lon.between(W, E) & C.ev.lat.between(S, N)]
        out.append((lo, hi, len(e)))
    return out


def _caption_m1(M, C, ks, region, variant="3x3"):
    nv = [int(M.valid['vp'][k].sum()) for k in ks]
    ns = [int(M.valid['vs'][k].sum()) for k in ks]
    nr = [int(M.valid['vpvs'][k].sum()) for k in ks]
    ncp = [int((M.chk['vp'][k] & M.valid['vp'][k]).sum()) for k in ks]
    ncs = [int((M.chk['vs'][k] & M.valid['vs'][k]).sum()) for k in ks]
    evc = _ev_counts(M, C, ks, region)
    dz = ", ".join(f"{M.z[k]:g}" for k in ks)
    if variant == "3x3":
        lay = (f"P-wave velocity (A-C), S-wave velocity (D-F) and Vp/Vs (G-I) of model {M.version} at {dz} km below sea "
               f"level (columns).")
    else:
        lay = (f"P-wave velocity (A, D), S-wave velocity (B, E) and Vp/Vs (C, F) of model {M.version} at {dz} km below "
               f"sea level (rows).")
    return (f"{lay} Colour is drawn only on rock nodes that are checkerboard-resolved or have a derivative weight sum "
            f"(DWS) ≥ {M.dws_min:g} (P for Vp, S for Vs, both for Vp/Vs); values between nodes are interpolated from "
            f"shown nodes only. Shown nodes per depth: Vp {_and(nv)} ({_and(ncp)} checkerboard-resolved), Vs {_and(ns)} "
            f"({_and(ncs)}), Vp/Vs {_and(nr)}. Black outline: checkerboard-resolved cells; grey hairline: edge of the "
            f"shown cells; uncoloured: shaded relief (GMT earth_relief 01m). Dots: catalogue {C.catalog.name} tiers "
            f"{C.tiers}, events between the mid-depths to the neighbouring nodes "
            f"({'; '.join(f'{max(a, 0):g}-{b:g} km: {n:,}' for a, b, n in evc)} events). Dashed: Slab2 top at the depth "
            f"of the panel (Hayes et al. 2018). Lines with teeth: subduction front; solid: PB2002 plate boundaries (Bird "
            f"2003). One colour scale per variable for all depths: Vp {SHARED['vp']['norm'][0]:g}-"
            f"{SHARED['vp']['norm'][1]:g} km/s, Vs {SHARED['vs']['norm'][0]:g}-{SHARED['vs']['norm'][1]:g} km/s, Vp/Vs "
            f"{SHARED['vpvs']['norm'][0]:.2f}-{SHARED['vpvs']['norm'][1]:.2f}; triangles on the colour bars: values "
            f"beyond the range. Colour maps (Crameri 2018): {pal_name(PAL['v'])} (Vp, Vs) and {pal_name(PAL['vpvs'])} "
            f"(Vp/Vs, centred on 1.75).")


def paper(M, C, out, a=None):
    pdir = out / PAPER_DIR
    pdir.mkdir(parents=True, exist_ok=True)
    figs = {"M1": fig_m1(M, C, pdir), "M1b": fig_m1b(M, C, pdir), "M2": fig_m2(M, C, pdir),
            "SM3": fig_sm3(M, C, pdir)}
    f4, st4, used = fig_sm4(M, C, pdir)
    figs["SM4"] = f4
    write_review(M, C, out, pdir, figs, st4, used)


def write_review(M, C, out, pdir, figs, st4, used):
    when = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    ks = [int(np.argmin(np.abs(M.z - d))) for d in PAPER_DEPTHS]
    kb = [int(np.argmin(np.abs(M.z - d))) for d in M1B_DEPTHS]
    prof = "; ".join(f"{chr(66 + i)} {nm} ({-vv[0][0]:g}°W to {-vv[1][0]:g}°W)" for i, (nm, vv) in enumerate(PAPER_PROFILES))
    st = st4.round(2).to_html(index=False, border=0) if len(st4) else "<p>no overlap</p>"
    used_s = " and ".join(used)
    caps = {
        "M1": _caption_m1(M, C, ks, PAPER_REGION, "3x3"),
        "M1b": _caption_m1(M, C, kb, PAPER_REGION, "2x3"),
        "M2": (f"Vp/Vs of model {M.version} along four margin-normal profiles (B-E), located in A (purple: Vp/Vs shown "
               f"at one depth or more; lines with teeth: subduction front). Profiles: {prof}; 4° of longitude each, "
               f"vertical exaggeration about 1.6. Colour only where Vp and Vs are both shown (same rule as Fig. M1); "
               f"black outline: checkerboard-resolved; light grey: below the seafloor or ground, not shown; pale blue: "
               f"water; colour bar and symbol legend under A. Contours: Vp "
               f"{VP_SEC_LEVELS[0]:.1f} to {VP_SEC_LEVELS[-1]:.1f} km/s every 0.5 km/s, drawn only through cells where "
               f"Vp is shown (pieces shorter than 10 km omitted). Dots: catalogue {C.catalog.name} tiers {C.tiers} "
               f"events within ±10 km of the profile. Dashed: Slab2 top (Hayes et al. 2018). Black line: seafloor or "
               f"ground (GMT earth_relief 01m). Colour map: {pal_name(PAL['vpvs'])} (Crameri 2018), centred on 1.75."),
        "SM3": (f"Resolution of model {M.version} at {', '.join(f'{M.z[k]:g}' for k in ks)} km below sea level. A-B, "
                f"E-F, I-J: finest checkerboard cell (40, 60 or 100 km; ±8 % patterns in the depth bands -10, 12, 25, "
                f"40 and 65 km) recovered with local correlation ≥ 0.7 and amplitude ≥ 0.5 (D-B018), finest over the "
                f"covering tiles, P and S (dark: 40 km, light: 100 km). C-D, G-H, K-L: merge-weighted derivative weight sum (DWS), P and S; black "
                f"contour: DWS = {M.dws_min:g}. Node counts in the upper left. Uncoloured: shaded relief. Colour maps: "
                f"bamako (3 classes) and devon reversed (Crameri 2018)."),
        "SM4": (f"S-wave velocity of model {M.version} (first column; shown cells as in Fig. M1) and of {used_s} "
                f"(Delph, Levander &amp; Niu 2018, Geophys. Res. Lett. 45(20), doi:10.1029/2018GL079518; CRESCENT "
                f"Gen0 Community Velocity Model, He 2026, doi:10.6084/m9.figshare.31902061) at {', '.join(f'{M.z[k]:g}' for k in ks)} km below sea level, "
                f"and the difference 100 (Vs / Vs_prior − 1) at the shown nodes of this study (last two columns). Prior "
                f"models are sampled at the same depth (linear in depth) on their own grids and drawn over rock only. "
                f"One Vs scale ({SHARED['vs']['norm'][0]:g}-{SHARED['vs']['norm'][1]:g} km/s, as Fig. M1) for all "
                f"panels; number of compared nodes and median difference in the lower left of each difference panel. "
                f"Colour maps: {pal_name(PAL['v'])} and {pal_name(PAL['dv'])} (Crameri 2018)."),
    }
    titles = {"M1": "Fig. M1: model overview, Vp, Vs and Vp/Vs at 9, 18 and 29 km (3 × 3)",
              "M1b": "Fig. M1, alternative: two depths (18 and 29 km, 2 × 3)",
              "M2": "Fig. M2: margin-normal Vp/Vs sections",
              "M3": "Fig. M3 (slot): ensemble uncertainty of v0.4.2",
              "SM3": "Fig. S-M3: resolution (checkerboard, DWS)",
              "SM4": "Fig. S-M4: comparison with published Vs models"}

    def size(f):
        h, w = plt.imread(f).shape[:2]
        return f"{w / 300 * 2.54:.1f} × {h / 300 * 2.54:.1f} cm"

    def preview(f, dpi=150):
        from PIL import Image
        im = Image.open(f).convert("RGB")
        im = im.resize((int(im.width * dpi / 300), int(im.height * dpi / 300)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        return _b64(buf.getvalue())

    body = [f"<h1>{M.version} paper-figure candidates, v2 (Science Advances)</h1><p class='mut'>Made {when} by "
            f"<code>workflow/08_figures/model_figures.py paper</code> (branch paper/model-figures, {git_head()}). "
            f"Candidates for the PI to choose from; nothing is in paper_sciadv/figures/. Files (vector PDF with "
            f"rasterised colour fields and events, and PNG 300 dpi): <code>{pdir}</code>. Previews below are 150 dpi; "
            f"click a preview for the PDF. Width 17.8 cm (two columns); heights at or under 20 cm.</p>",
            f"<p><b>Model</b>: {M.version} only (latest). <b>Colour</b> (palette sheet: "
            f"<a href='palette_sheet.html'>palette_sheet.html</a>): Vp, Vs {pal_name(PAL['v'])} (slow warm, fast "
            f"cool); Vp/Vs {pal_name(PAL['vpvs'])} centred on 1.75 (high warm); differences {pal_name(PAL['dv'])} "
            f"centred on 0 (slow warm). <b>Style</b>: sans-serif (Nimbus Sans = Helvetica metrics, as the GMT figures "
            f"of v4_figures.py) 5-7 pt, frame 0.5 pt, panel letters 8 pt bold; shaded relief as the base map.</p>",
            "<details><summary>Model, mask and inputs</summary>" + header_html(M, C) + "</details>"]
    for key in ("M1", "M1b", "M2", "M3", "SM3", "SM4"):
        body.append(f"<h2>{titles[key]}</h2>")
        if key == "M3":
            body.append("<p class='cap'>Reserved for the node uncertainty of v0.4.2 from the 100-member "
                        "randomize-then-optimize ensemble (D-B030, D-B032), registered tonight; not made here. "
                        "Planned layout as Fig. M1 (rows Vp, Vs, Vp/Vs; columns 9, 18, 29 km; calibrated standard "
                        "deviation, one colour bar per variable, same mask), with a sequential map distinct from "
                        f"{pal_name(PAL['v'])}.</p>")
            continue
        f = figs[key]
        body.append(f"<p class='cap'><b>Caption draft (facts only):</b> {caps[key]}</p>")
        body.append(f"<p class='mut'><a href='{PAPER_DIR}/{f.with_suffix('.pdf').name}'>{f.with_suffix('.pdf').name}</a>"
                    f" &middot; <a href='{PAPER_DIR}/{f.name}'>{f.name}</a> ({size(f)})</p>"
                    f"<a href='{PAPER_DIR}/{f.with_suffix('.pdf').name}'><img src='{preview(f)}' alt='{key}' "
                    f"style='max-width:{1000 if key != 'M2' else 1200}px'></a>")
        if key == "SM4":
            body.append(f"<details><summary>Differences per depth (this study − prior, % of prior)</summary>{st}</details>"
                        "<p class='mut'>Citation: the Delph model is Delph, Levander &amp; Niu (2018), GRL 45(20), "
                        "doi:10.1029/2018GL079518. The EMC netCDF's attribute naming Ward &amp; Lin is wrong and is not "
                        "used.</p>")
    html = out / f"{PAPER_DIR}.html"
    html.write_text(_html_page(f"{M.version} paper candidates v2", "\n".join(body)))
    print(f"wrote {html} ({html.stat().st_size / 1e6:.1f} MB)")
    return html


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["gallery", "paper", "palette", "all"])
    ap.add_argument("--cmaps", default=None, help="override the chosen maps, e.g. 'dv=vik_r,v=batlow_r,vpvs=roma_r' "
                    "(SCM names; _r = reversed)")
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--catalog", default=str(CATALOG), help="hypocentres (lat, lon, depth, tier); default the "
                    "released v6.0 catalogue")
    ap.add_argument("--tiers", default="ABC")
    ap.add_argument("--stations", default=str(STATIONS))
    ap.add_argument("--mask", default="either", choices=["either", "checker", "dws"])
    ap.add_argument("--dws-min", type=float, default=100.0)
    ap.add_argument("--out", default=None, help="default /wd1/mdenolle_data/tomo_v1/models/figures_<version>")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    if a.cmaps:
        for item in a.cmaps.split(","):
            g, nm = item.split("=")
            PAL[g.strip()] = (nm.strip().removesuffix("_r"), nm.strip().endswith("_r"))
    M = Model(a.model, a.mask, a.dws_min)
    out = Path(a.out) if a.out else Path(a.model).resolve().parent.parent / f"figures_{M.version}"
    out.mkdir(parents=True, exist_ok=True)
    C = Context(a.catalog, a.tiers, a.stations, cache=out / "cache")
    if a.what in ("palette", "all"):
        palette_sheet(M, C, out)
    if a.what in ("gallery", "all"):
        gallery(M, C, out, jobs=min(a.jobs, 4))
    if a.what in ("paper", "all"):
        paper(M, C, out, a)


if __name__ == "__main__":
    main()

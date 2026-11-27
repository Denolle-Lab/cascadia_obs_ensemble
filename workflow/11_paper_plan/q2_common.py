"""Q2 (D-B115): what limits the depth of offshore earthquakes - temperature or composition? Shared inputs and models.

Pre-registration: data/paper_plan/q2_depth_limit/RESUME.md (2026-11-27 13:27:22 PST). Nothing here compares a depth
with a level; q2_analysis.py does, with the registered rules.

Conventions (as pb_common): depth km below sea level (bsl), positive down; bsf = depth + surface_km (below the solid
surface: seafloor offshore). Isotherms are computed below the seafloor and converted to bsl with the event's surface_km
(z_bsl = z_bsf - surface_km). Velocity levels are model depths bsl.

Environment: Q2_CATALOG (v6.1 or v6.0), Q2_SEGMENTS (segment table, default the PB table of that catalog), Q2_OUT
(small tables), Q2_BIG (report, figures). The segment join reads the segment table at run time (path, md5 and mtime
are recorded in the outputs), so a change of the plate-boundary axes only needs a rerun.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import erf, erfc

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb_common as P  # noqa: E402

WD1 = Path("/wd1/mdenolle_data")
ROOT = P.ROOT
_env = os.environ.get
CATS = {
    "v6.1": dict(csv=WD1 / "catalog_v6/v6.1/cascadia_catalog_v6.1.csv",
                 seg=WD1 / "paper_plan/v6.1/events_segments_v61.parquet", axes="D-B113 (plate_axes_v1)"),
    "v6.0": dict(csv=WD1 / "catalog_v6/v6.0/cascadia_catalog_v6.0.csv",
                 seg=WD1 / "paper_plan/events_segments_v60.parquet", axes="PB2002 (first v6.0 build)"),
}
OUT = Path(_env("Q2_OUT", ROOT / "data/paper_plan/q2_depth_limit"))
BIG = Path(_env("Q2_BIG", WD1 / "paper_plan/q2_depth_limit"))
EXT = WD1 / "external_geo"
AGE_GRID = EXT / "agegrid_seton2020/age.2020.1.GTS2012.1m.nc"
SED_GRID = EXT / "globsed_v3/extract/GlobSed-v3.nc"
HF_TABLE = EXT / "ghfdb_r2024/extract/IHFC_2024_GHFDB_v.2026.03.txt"
TOMODD = WD1 / "tomo_v1/b2_tomodd"
TOMODD_SETS = ["margin_m0_v4", "mendocino_gorda_c_m0g", "blanco_m0g", "oregon_m0f", "washington_m0g"]  # v0.4.2 config
SYNTH = WD1 / "tomo_v1/ensemble_v0.4.2/synth"
NULL = WD1 / "tomo_v1/ensemble_v0.4.2_null/synth"
SEED = 20261127
NDRAW = int(_env("Q2_NDRAW", 1000))   # registered 1000; the override is for code tests only
DELTA = 3.0                     # km, registered tolerance
VINFL = 1.50                    # Vp spread calibration (D-B099), applied once to raw member level deviations
MYR = 3.15576e13                # s

OFFSHORE = {"AX": "ridge", "EN": "ridge", "JR": "ridge", "EXR": "ridge", "GR": "ridge",
            "BL": "transform", "MFZ": "transform", "SOV": "transform",
            "GP": "intraplate", "JP": "intraplate", "EXP": "intraplate",
            "MTJ": "triple junction", "GS": "slab", "SIF": "slab", "IS": "slab"}
TRANSFORMS = ("BL", "MFZ", "SOV")
GROUPS = {   # registered groups (a): boundary types by segment, plus GS alone and ridge without AX
    "ridge": ["AX", "EN", "JR", "EXR", "GR"], "ridge (no AX)": ["EN", "JR", "EXR", "GR"],
    "transform": ["BL", "MFZ", "SOV"], "intraplate": ["GP", "JP", "EXP"], "triple junction": ["MTJ"],
    "slab": ["GS", "SIF", "IS"], "Gorda slab (GS)": ["GS"],
}
# thermal models (registered): T/Tm = F(z, a), a = 2 sqrt(kappa t)
THERMAL = {
    "HSC": dict(Tm=1350.0, kappa=1.0e-6, L=None, Tm_lo=1300.0, Tm_hi=1400.0,
                ref="half-space cooling, Tm 1350 C, kappa 1e-6 m2/s (Turcotte & Schubert 2002)"),
    "GDH1": dict(Tm=1450.0, kappa=8.04e-7, L=95.0, Tm_lo=1400.0, Tm_hi=1500.0,
                 ref="plate model GDH1, 95 km, Tm 1450 C, kappa 8.04e-7 m2/s (Stein & Stein 1992)"),
}
ISOTHERMS = (350.0, 600.0, 800.0)
LEVELS = (7.5, 8.0)
log = P.log


def md5(path, n=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(n), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ cooling models
def F(z, a, L=None):
    """Normalised temperature T/Tm at depth z (km) below the top for a = 2 sqrt(kappa t) (km). Half-space: erf(z/a).
    Plate of thickness L (T = 0 at the top, Tm at the base, initially Tm): image series (exact sum of the Fourier
    series solution; 4 image pairs are enough for a < 2L)."""
    a = np.maximum(np.asarray(a, float), 1e-6)
    z = np.asarray(z, float)
    if L is None:
        return erf(z / a)
    u = np.zeros(np.broadcast(z, a).shape)
    for n in range(4):
        u = u + erfc((2 * n * L + z) / a) - erfc((2 * (n + 1) * L - z) / a)
    return 1.0 - u


def a_of(age_myr, kappa):
    return 2.0 * np.sqrt(kappa * np.maximum(np.asarray(age_myr, float), 0.0) * MYR) / 1000.0   # km


def isotherm_depth(theta, a1, a2=None, L=None, zmax=300.0, it=48):
    """Depth (km) where T/Tm = theta; with a2, of the mean geotherm (F(z,a1) + F(z,a2)) / 2 (transforms). Bisection,
    vectorised over the broadcast of the inputs."""
    theta = np.asarray(theta, float)
    shape = np.broadcast(theta, a1, a1 if a2 is None else a2).shape
    lo = np.zeros(shape)
    hi = np.full(shape, (L - 1e-6) if L else zmax)
    f = (lambda z: F(z, a1, L)) if a2 is None else (lambda z: 0.5 * (F(z, a1, L) + F(z, a2, L)))
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        up = f(mid) < theta
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    return 0.5 * (lo + hi)


def temperature_at(z, model, age1, age2=None):
    """Temperature (C) at depth z (km below the top) for the nominal parameters of `model`."""
    p = THERMAL[model]
    a1 = a_of(age1, p["kappa"])
    if age2 is None:
        return p["Tm"] * F(z, a1, p["L"])
    return p["Tm"] * 0.5 * (F(z, a1, p["L"]) + F(z, a_of(age2, p["kappa"]), p["L"]))


# ------------------------------------------------------------------ grids
class Grid:
    """Regular lon/lat grid subset with bilinear sampling (NaN where any corner is NaN)."""

    def __init__(self, path, var="z", box=(-136.0, -116.0, 35.0, 54.0)):
        import xarray as xr
        ds = xr.open_dataset(path)
        lon, lat = ds.lon.values, ds.lat.values
        i0, i1 = np.searchsorted(lon, box[0]), np.searchsorted(lon, box[1])
        j0, j1 = np.searchsorted(lat, box[2]), np.searchsorted(lat, box[3])
        self.lon, self.lat = lon[i0:i1], lat[j0:j1]
        self.v = ds[var].values[j0:j1, i0:i1].astype(float)
        self.dlon, self.dlat = self.lon[1] - self.lon[0], self.lat[1] - self.lat[0]

    def __call__(self, lon, lat):
        lon, lat = np.asarray(lon, float), np.asarray(lat, float)
        x = (lon - self.lon[0]) / self.dlon
        y = (lat - self.lat[0]) / self.dlat
        i = np.clip(np.floor(x).astype(int), 0, len(self.lon) - 2)
        j = np.clip(np.floor(y).astype(int), 0, len(self.lat) - 2)
        fx, fy = x - i, y - j
        v = self.v
        out = (v[j, i] * (1 - fx) * (1 - fy) + v[j, i + 1] * fx * (1 - fy) + v[j + 1, i] * (1 - fx) * fy
               + v[j + 1, i + 1] * fx * fy)
        bad = (x < 0) | (y < 0) | (x > len(self.lon) - 1) | (y > len(self.lat) - 1)
        out[bad] = np.nan
        return out

    def disc_median(self, lon, lat, r_km):
        """Median of the grid nodes within r_km of each point (NaN-aware)."""
        lon, lat = np.atleast_1d(lon), np.atleast_1d(lat)
        out = np.full(len(lon), np.nan)
        for k, (lo, la) in enumerate(zip(lon, lat)):
            dl = r_km / P.KM
            dlo = dl / np.cos(np.radians(la))
            ii = (self.lon >= lo - dlo) & (self.lon <= lo + dlo)
            jj = (self.lat >= la - dl) & (self.lat <= la + dl)
            if not ii.any() or not jj.any():
                continue
            LO, LA = np.meshgrid(self.lon[ii], self.lat[jj])
            d = P.hav(la, lo, LA, LO)
            vv = self.v[np.ix_(jj, ii)][d <= r_km]
            vv = vv[np.isfinite(vv)]
            if len(vv):
                out[k] = np.median(vv)
        return out


def offset_point(lon, lat, az_deg, d_km):
    """Point d_km from (lon, lat) along azimuth az (deg from north), flat-earth (d << R)."""
    az = np.radians(az_deg)
    dlat = d_km * np.cos(az) / P.KM
    dlon = d_km * np.sin(az) / (P.KM * np.cos(np.radians(lat)))
    return lon + dlon, lat + dlat


def transform_side_ages(age: Grid, traces_otf, lon, lat, d_km=20.0, r_km=5.0):
    """Ages on both sides of the nearest transform trace point: median grid age within r_km of the points d_km on
    either side, perpendicular to the local trace (strike from the trace points within 5 km). Returns (young, old,
    side-A, side-B, distance to the trace)."""
    from scipy.spatial import cKDTree
    tr = traces_otf
    xyz = P.ecef(tr[:, 0], tr[:, 1])
    tree = cKDTree(xyz)
    dist, idx = tree.query(P.ecef(lon, lat))
    uniq = np.unique(idx)
    res = {}
    for k in uniq:
        nb = tree.query_ball_point(xyz[k], 5.0)
        p = tr[nb]
        if len(p) < 2:
            nb = tree.query(xyz[k], 5)[1]
            p = tr[nb]
        x = (p[:, 0] - tr[k, 0]) * P.KM * np.cos(np.radians(tr[k, 1]))
        y = (p[:, 1] - tr[k, 1]) * P.KM
        c = np.cov(np.vstack([x, y]))
        w, v = np.linalg.eigh(c)
        sx, sy = v[:, -1]
        strike = np.degrees(np.arctan2(sx, sy)) % 180.0
        ages = []
        for az in (strike + 90.0, strike - 90.0):
            plo, pla = offset_point(tr[k, 0], tr[k, 1], az, d_km)
            ages.append(age.disc_median(plo, pla, r_km)[0])
        res[k] = ages
    A = np.array([res[k][0] for k in idx])
    B = np.array([res[k][1] for k in idx])
    return np.fmin(A, B), np.fmax(A, B), A, B, dist


def slab_dip_fn():
    """Slab2 dip (deg) from the gradient of the depth grid."""
    from scipy.interpolate import RegularGridInterpolator
    s = pd.read_csv(P.SLAB, header=None, names=["lon", "lat", "z"])
    s["lon"] = np.where(s.lon > 180, s.lon - 360, s.lon)
    lo, la = np.sort(s.lon.unique()), np.sort(s.lat.unique())
    g = -s.pivot_table(index="lat", columns="lon", values="z", dropna=False).reindex(index=la, columns=lo).values
    dy = np.gradient(g, la * P.KM, axis=0)
    dx = np.gradient(g, axis=1) / (np.gradient(lo)[None, :] * P.KM * np.cos(np.radians(la))[:, None])
    dip = np.degrees(np.arctan(np.hypot(dx, dy)))
    f = RegularGridInterpolator((la, lo), dip, bounds_error=False, fill_value=np.nan)
    return lambda lon, lat: f(np.column_stack([np.asarray(lat, float), np.asarray(lon, float)]))


def heat_flow_table():
    """GHFDB R2024 rows in the study box: lon, lat, q (mW/m2), environment."""
    rows = []
    with open(HF_TABLE, encoding="latin-1") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    hdr = lines[5].rstrip("\n").split("\t")
    iq, ila, ilo, ien = hdr.index("q"), hdr.index("lat_NS"), hdr.index("long_EW"), hdr.index("environment")
    for ln in lines[6:]:
        p = ln.rstrip("\n").split("\t")
        try:
            la, lo, q = float(p[ila]), float(p[ilo]), float(p[iq])
        except (ValueError, IndexError):
            continue
        if 37 < la < 52 and -133 < lo < -120:
            rows.append((lo, la, q, p[ien]))
    return pd.DataFrame(rows, columns=["lon", "lat", "q", "environment"])


# ------------------------------------------------------------------ events
def tomodd_ids():
    ids = set()
    for s in TOMODD_SETS:
        ids |= set(pd.read_csv(TOMODD / s / "inputs/events.csv", usecols=["ID"]).ID.astype(int))
    return ids


def load_events(catalog="v6.1", seg_path=None):
    """Located events of the catalog with the D-B113 (v6.1) segment join, quality flags, dist_min and in_tomo."""
    cfg = CATS[catalog]
    seg_path = Path(seg_path or _env("Q2_SEGMENTS", cfg["seg"]))
    c = pd.read_parquet(seg_path)
    extra = pd.read_csv(cfg["csv"], usecols=["ID", "dist_min", "ell_len3"], low_memory=False)
    c = c.merge(extra, on="ID", how="left")
    c["in_tomo"] = c.ID.astype(int).isin(tomodd_ids())
    meta = dict(catalog=catalog, segments=str(seg_path), segments_md5=md5(seg_path),
                segments_mtime=pd.Timestamp(os.path.getmtime(seg_path), unit="s", tz="UTC").tz_convert(
                    "US/Pacific").strftime("%Y-%m-%d %H:%M %Z"), axes=cfg["axes"])
    return c, meta


def select(c, tiers=("A", "B"), sigz=3.0):
    """Registered depth-reliable set E (quality cuts only; the offshore / frame restriction is applied by the caller)."""
    m = (c.loc_source.ne("none") & c.tier.isin(tiers) & c.depth_quality.eq("ok") & ~c.floor_limited & ~c.far_only
         & ~c.deep_pass & ~c.five_phase & c.is_eq & ~c.review_P1)
    if sigz is not None:
        m &= c.sig_z <= sigz
    return c[m].copy()

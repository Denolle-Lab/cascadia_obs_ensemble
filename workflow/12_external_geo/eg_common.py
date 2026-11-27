"""Shared paths, readers, sampling and thermal models for the external geophysical data (D-B114).

D-B114 (PI 2026-11-27 13:13): pull published geophysical data (heat flow, bathymetry roughness, plate age, thermal
structure, sediment thickness) that could corroborate the plate-boundary seismicity paper.

Raw downloads live under /wd1/mdenolle_data/external_geo/<dataset>/ (fetched by fetch_all.sh; untrusted data, read
here as data only). Derived products go to /wd1/mdenolle_data/external_geo/derived/. Nothing large goes into git.

Conventions: longitude -180..180 (W negative); depth km below sea level, positive down; "bsf" = depth below the solid
surface (seafloor or ground); heat flow mW/m2; age Myr; temperature degC.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.special import erfinv

ROOT = Path(os.environ.get("EXTGEO_ROOT", "/wd1/mdenolle_data/external_geo"))
DER = ROOT / "derived"
CATALOG = Path(os.environ.get("EXTGEO_CATALOG", "/wd1/mdenolle_data/catalog_v6/v6.1/cascadia_catalog_v6.1.csv"))
CATTAG = os.environ.get("EXTGEO_TAG", "v61")
# per-event segment table (paper plan, D-B107); its definitions are being corrected, so the join is a separate step
SEGTABLE = Path(os.environ.get("EXTGEO_SEGTABLE", "/wd1/mdenolle_data/paper_plan/v6.1/events_segments_v61.parquet"))
GMRT_DIR = Path("/wd1/mdenolle_data/reloc_v5/topo/gmrt")

# study area and the common lattice (0.1 deg, node-registered)
LAT0, LAT1, LON0, LON1 = 39.0, 52.0, -132.0, -120.0
DLAT = DLON = 0.1
LAT_NODES = np.round(np.arange(LAT0, LAT1 + 1e-9, DLAT), 4)
LON_NODES = np.round(np.arange(LON0, LON1 + 1e-9, DLON), 4)

RE_KM = 6371.0
SEC_PER_MYR = 3.15576e13

# ------------------------------------------------------------------ raw file locations
F_GHFDB = ROOT / "ghfdb_r2024/extract/IHFC_2024_GHFDB_v.2026.03.txt"
F_AGE = ROOT / "agegrid_seton2020/age.2020.1.GTS2012.1m.nc"
F_AGE_MISFIT = ROOT / "agegrid_seton2020/age_misfit.2020.1.GeeK2007.6m.nc"
F_AGE_CONF = ROOT / "agegrid_seton2020/conf.2020.1.GeeK2007.6m.nc"
F_RATE = ROOT / "agegrid_seton2020/full_rate.2020.1.GTS2012.6m.nc"
F_SED = ROOT / "globsed_v3/extract/GlobSed-v3.nc"
F_CRUST1 = ROOT / "crust1/crsthk.xyz.zip"
F_GEBCO = DER / "gebco_subset/gebco_2026_cascadia.bil"          # gdal_translate subset of the GEBCO_2026 tile
F_GEBCO_TID = DER / "gebco_subset/gebco_2026_tid_cascadia.bil"
VKW = ROOT / "vankeken_wilson_2023/zip_members"


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ geometry
def unit_xyz(lon, lat):
    lo, la = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def chord_to_km(c):
    return 2 * RE_KM * np.arcsin(np.clip(c / 2, 0, 1))


def km_to_chord(d):
    return 2 * np.sin(np.asarray(d, float) / (2 * RE_KM))


def lattice():
    """(lon2d, lat2d) of the 0.1 deg lattice, shape (nlat, nlon)."""
    return np.meshgrid(LON_NODES, LAT_NODES)


# ------------------------------------------------------------------ grid readers and sampling
def _coords(ds, names):
    for n in names:
        if n in ds.coords or n in ds.variables:
            return n
    raise KeyError(names)


def read_grid(path, var=None, pad=1.0):
    """Subset a global lon/lat netCDF grid to the study area (+pad deg). Returns (lon, lat, z) with lon -180..180."""
    ds = xr.open_dataset(path)
    xn = _coords(ds, ["lon", "x", "longitude"])
    yn = _coords(ds, ["lat", "y", "latitude"])
    var = var or [v for v in ds.data_vars if ds[v].ndim == 2][0]
    lon = ds[xn].values.astype(float)
    lon = np.where(lon > 180, lon - 360, lon)
    lat = ds[yn].values.astype(float)
    ix = np.where((lon >= LON0 - pad) & (lon <= LON1 + pad))[0]
    iy = np.where((lat >= LAT0 - pad) & (lat <= LAT1 + pad))[0]
    z = ds[var].isel({xn: ix, yn: iy}).values.astype(float)
    lon, lat = lon[ix], lat[iy]
    if ds[var].dims[0] == xn:
        z = z.T
    o = np.argsort(lon)
    lon, z = lon[o], z[:, o]
    if lat[0] > lat[-1]:
        lat, z = lat[::-1], z[::-1]
    return lon, lat, z


def sample(lon_g, lat_g, z, lon, lat, method="linear"):
    """Sample a (lat, lon) grid at points. Linear: NaN if any of the 4 corners is NaN."""
    f = RegularGridInterpolator((lat_g, lon_g), z, method=method, bounds_error=False, fill_value=np.nan)
    return f(np.column_stack([np.asarray(lat, float).ravel(), np.asarray(lon, float).ravel()])).reshape(np.shape(lon))


def read_ehdr(path):
    """GDAL EHdr (.bil + .hdr) single band -> (lon, lat ascending, z float with NaN for NODATA)."""
    hdr = {}
    for line in open(str(path).replace(".bil", ".hdr")):
        k, v = line.split(None, 1)
        hdr[k.upper()] = v.strip()
    nr, nc, nb = int(hdr["NROWS"]), int(hdr["NCOLS"]), int(hdr["NBITS"])
    dt = {16: "<i2", 8: "i1", 32: "<f4"}[nb]
    z = np.fromfile(path, dtype=dt).reshape(nr, nc).astype(float)
    z[z == float(hdr["NODATA"])] = np.nan
    x0, y0, dx, dy = (float(hdr[k]) for k in ("ULXMAP", "ULYMAP", "XDIM", "YDIM"))
    lon = x0 + dx * np.arange(nc)
    lat = y0 - dy * np.arange(nr)
    return lon, lat[::-1], z[::-1]


# ------------------------------------------------------------------ datasets
def read_ghfdb(pad=1.0) -> pd.DataFrame:
    """IHFC Global Heat Flow Database R2024 (v2026.03): one row per parent site (ID_parent) in the area (+pad).

    The release is one row per child measurement; q, lat_NS, long_EW and Quality_Score_Parent are parent fields.
    Exact duplicates (same position to 1e-4 deg and same q, e.g. one value cited by two publications) are kept once.
    """
    df = pd.read_csv(F_GHFDB, sep="\t", skiprows=12, encoding="latin-1", low_memory=False)
    for c in ["q", "q_uncertainty", "qc", "lat_NS", "long_EW", "elevation", "Year", "probe_penetration"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df.lat_NS.between(LAT0 - pad, LAT1 + pad) & df.long_EW.between(LON0 - pad, LON1 + pad)]
    g = df.groupby("ID_parent", sort=False)
    s = g.agg(lat=("lat_NS", "first"), lon=("long_EW", "first"), q=("q", "first"), q_unc=("q_uncertainty", "first"),
              elevation=("elevation", "first"), environment=("environment", "first"),
              ref=("publication_reference", "first"), year=("Year", "first"), q_method=("q_method", "first"),
              quality_parent=("Quality_Score_Parent", "first"), expedition=("expedition", "first"),
              n_child=("q", "size")).reset_index().rename(columns={"ID_parent": "site_id"})
    s = s[np.isfinite(s.q) & np.isfinite(s.lat) & np.isfinite(s.lon)]
    key = s.lat.round(4).astype(str) + "_" + s.lon.round(4).astype(str) + "_" + s.q.round(2).astype(str)
    s = s.loc[~key.duplicated()].reset_index(drop=True)
    s["in_area"] = s.lat.between(LAT0, LAT1) & s.lon.between(LON0, LON1)
    return s


def read_crust1():
    """CRUST1.0 crustal thickness (km), 1x1 deg cell centres -> (lon, lat, z) grid."""
    import io
    import zipfile
    with zipfile.ZipFile(F_CRUST1) as zf:
        name = [n for n in zf.namelist() if n.endswith(".xyz") and not n.startswith("__")][0]
        a = np.loadtxt(io.TextIOWrapper(zf.open(name)))
    lon_u, lat_u = np.unique(a[:, 0]), np.unique(a[:, 1])
    z = np.full((lat_u.size, lon_u.size), np.nan)
    z[np.searchsorted(lat_u, a[:, 1]), np.searchsorted(lon_u, a[:, 0])] = a[:, 2]
    lon_u = np.where(lon_u > 180, lon_u - 360, lon_u)
    o = np.argsort(lon_u)
    return lon_u[o], lat_u, z[:, o]


# ------------------------------------------------------------------ oceanic lithosphere cooling models
# Half-space cooling (HSC): T(z,t) = Tm erf(z / (2 sqrt(kappa t))), surface at 0 degC; values of Turcotte & Schubert
# (Geodynamics, ch. 4, cooling of the oceanic lithosphere): Tm - T0 = 1300 K, kappa = 1e-6 m2/s, k = 3.3 W/m/K.
HSC = dict(Tm=1300.0, kappa=1.0e-6, k=3.3)
# Plate model GDH1 (Stein & Stein 1992, Nature 359, 123-129): a = 95 km, Tm = 1450 degC, k = 3.138 W/m/K,
# rho = 3330 kg/m3, Cp = 1171 J/kg/K -> kappa = 8.04e-7 m2/s. T(z,t) = Tm [z/a + sum 2/(n pi) exp(-n^2 pi^2 kappa t
# / a^2) sin(n pi z / a)] (McKenzie 1967; Parsons & Sclater 1977).
GDH1 = dict(Tm=1450.0, a=95.0e3, k=3.138, kappa=3.138 / (3330.0 * 1171.0))
AGE_FLOOR = 0.1    # Myr; ages below it are set to it for the isotherm depths (flagged in the tables)


def iso_hsc(age_myr, T):
    t = np.maximum(np.asarray(age_myr, float), AGE_FLOOR) * SEC_PER_MYR
    return 2 * np.sqrt(HSC["kappa"] * t) * erfinv(T / HSC["Tm"]) / 1e3          # km below the seafloor


def T_plate(z_m, t_s, nterm=200):
    a, Tm, kap = GDH1["a"], GDH1["Tm"], GDH1["kappa"]
    n = np.arange(1, nterm + 1)[:, None]
    s = (2 / (n * np.pi)) * np.exp(-(n ** 2) * np.pi ** 2 * kap * t_s[None] / a ** 2) * np.sin(n * np.pi * z_m[None] / a)
    return Tm * (z_m / a + s.sum(0))


def iso_plate(age_myr, T, iters=60):
    """Depth (km below the seafloor) of isotherm T in the GDH1 plate model; bisection on [0, a]."""
    age = np.asarray(age_myr, float)
    out = np.full(age.shape, np.nan)
    ok = np.isfinite(age)
    t = np.maximum(age[ok], AGE_FLOOR) * SEC_PER_MYR
    lo, hi = np.zeros_like(t), np.full_like(t, GDH1["a"])
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        above = T_plate(mid, t) < T                   # T increases with depth
        lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
    out[ok] = 0.5 * (lo + hi) / 1e3
    return out


def q_gdh1(age_myr):
    """GDH1 conductive surface heat flow (mW/m2): 510 t^-1/2 for t <= 55 Myr, 48 + 96 exp(-0.0278 t) beyond."""
    t = np.maximum(np.asarray(age_myr, float), AGE_FLOOR)
    return np.where(t <= 55, 510.0 / np.sqrt(t), 48.0 + 96.0 * np.exp(-0.0278 * t))


def q_hsc(age_myr):
    t = np.maximum(np.asarray(age_myr, float), AGE_FLOOR) * SEC_PER_MYR
    return 1e3 * HSC["k"] * HSC["Tm"] / np.sqrt(np.pi * HSC["kappa"] * t)


# ------------------------------------------------------------------ slab thermal models (van Keken & Wilson 2023)
# Syracuse et al. (2010) global suite, as rerun by van Keken & Wilson (2023, Zenodo 10.5281/zenodo.14854311 v0.9.6):
# D80 (Syracuse et al. 2010 reproduction) and D80new (lower mantle potential temperature, erf inflow). Two Cascadia
# cross-sections: directory 03_British_Columbia = "North Cascadia" (Table 2: 122 W, 47 N, arc-trench 349 km) and
# 04_Cascadia = "Central Cascadia" (121.5 W, 42 N, 295 km). slabpaths.040: x, y (km, y negative down), T (degC),
# p, id, s after 40 Myr; id 98 = slab surface, 100 = base of sediments, 104-112 = 2.5-10.5 km below it.
SLAB_MODELS = {"NCas47N": "03_British_Columbia", "CCas42N": "04_Cascadia"}
SLAB_MODEL_LAT = {"NCas47N": 47.0, "CCas42N": 42.0}


def slab_paths(suite="D80new", model="NCas47N"):
    f = VKW / f"vankeken_wilson_peps_2023__{suite}__sepran__{SLAB_MODELS[model]}__slabpaths.040"
    a = np.loadtxt(f)
    return pd.DataFrame(a, columns=["x", "y", "T", "p", "id", "s"])


def slab_T_vs_depth(suite="D80new", model="NCas47N", path_id=98):
    """(depth_km ascending, T degC) along one slab path; depth = -y (km below sea level in the model frame)."""
    d = slab_paths(suite, model)
    d = d[d.id == path_id].sort_values("s")
    dep, T = -d.y.values, d["T"].values
    keep = np.r_[True, np.diff(dep) > 0]          # monotone in depth
    return dep[keep], T[keep]

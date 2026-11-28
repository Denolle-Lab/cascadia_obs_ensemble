#!/usr/bin/env python3
"""Q1 (D-B115): interim per-event covariates sampled from the D-B114 downloads, until the external job's per-event
table exists (then q1_tests.py uses that table and reports both).

Per located event of the catalog (pre-registered definitions, RESUME.md):
  age_myr      Seton et al. 2020 age grid (age.2020.1.GTS2012.1m.nc), nearest 1' node; NaN off oceanic crust
  rate_mmyr    full spreading rate grid of the same release (full_rate.2020.1.GTS2012.6m.nc), nearest 6' node
  sed_m        GlobSed v3 total sediment thickness (GlobSed-v3.nc), nearest 5' node
  heatflow_mw  median of GHFDB R2024 heat-flow values (column q) within 25 km of the epicentre, >= 3 values
  z600_km      depth of the 600 C isotherm of a plate cooling model (GDH1, Stein & Stein 1992: a = 95 km,
               Tm = 1450 C, kappa = 0.804e-6 m2/s) at age_myr, below the seafloor; rank-identical to age_myr
  rough_m      RMS of GEBCO 2026 elevation after removing a best-fit plane in the 0.2 x 0.2 degree window around the
               nearest 0.025 degree node; NaN unless every 15" node of the window is below sea level
Provenance (URL, md5, date) of every input: /wd1/mdenolle_data/external_geo/fetch_log.tsv (D-B114 job). The GEBCO
window is cut once to covariates/gebco2026_cut.nc with
  gmt grdconvert gebco_2026_n90.0_s0.0_w-180.0_e-90.0_geotiff.tif=gd -R-133/-118/37/53 -G gebco2026_cut.nc

Writes BIG/covariates/covariates_interim_<tag>.parquet and its provenance json.
Run: python q1_covariates.py [--tag v61|v60]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import q1_common as Q  # noqa: E402

EXT = Q.WD1 / "external_geo"
FILES = dict(
    age=EXT / "agegrid_seton2020/age.2020.1.GTS2012.1m.nc",
    rate=EXT / "agegrid_seton2020/full_rate.2020.1.GTS2012.6m.nc",
    sed=EXT / "globsed_v3/extract/GlobSed-v3.nc",
    hf=EXT / "ghfdb_r2024/extract/IHFC_2024_GHFDB_v.2026.03.txt",
    gebco=Q.BIG / "covariates/gebco2026_cut.nc",
)
BOX = (-133.0, -118.0, 37.0, 53.0)


def nearest_grid(f, lon, lat, var="z"):
    d = xr.open_dataset(f)
    lo, la = d.lon.values, d.lat.values
    sub = d[var].sel(lon=slice(BOX[0] - 0.5, BOX[1] + 0.5), lat=slice(BOX[2] - 0.5, BOX[3] + 0.5)).load()
    lo, la = sub.lon.values, sub.lat.values
    ix = np.clip(np.round((lon - lo[0]) / (lo[1] - lo[0])).astype(int), 0, len(lo) - 1)
    iy = np.clip(np.round((lat - la[0]) / (la[1] - la[0])).astype(int), 0, len(la) - 1)
    return sub.values[iy, ix].astype(float)


def heatflow(lon, lat, r_km=25.0, nmin=3):
    h = pd.read_csv(FILES["hf"], sep="\t", skiprows=12, encoding="latin-1", low_memory=False,
                    usecols=["q", "lat_NS", "long_EW"])
    for c in h.columns:
        h[c] = pd.to_numeric(h[c], errors="coerce")
    h = h.dropna()
    h = h[(h.long_EW > BOX[0] - 1) & (h.long_EW < BOX[1] + 1) & (h.lat_NS > BOX[2] - 1) & (h.lat_NS < BOX[3] + 1)]
    t = cKDTree(Q.P.ecef(h.long_EW.values, h.lat_NS.values))
    pts = Q.P.ecef(lon, lat)
    nb = t.query_ball_point(pts, r=r_km)
    q = h.q.values
    out = np.array([np.median(q[i]) if len(i) >= nmin else np.nan for i in nb])
    n = np.array([len(i) for i in nb])
    return out, n, len(h)


def gdh1_isotherm(age_myr, T=600.0, a=95.0, Tm=1450.0, kappa=0.804e-6, nterm=200):
    """Depth (km below the seafloor) of isotherm T in the GDH1 plate model at age (Myr); 0 at age 0."""
    age = np.asarray(age_myr, float)
    n = np.arange(1, nterm + 1)
    z = np.linspace(0, a, 951)                                            # km, 0.1 km steps
    S = np.sin(n[:, None] * np.pi * z[None, :] / a)                      # [nterm, z]
    tab_age = np.arange(0, 200.01, 0.02)                                 # Myr lookup table
    tab = np.zeros(len(tab_age))
    for i, t in enumerate(tab_age[1:], start=1):
        e = np.exp(-(n ** 2) * np.pi ** 2 * kappa * t * 3.15576e13 / (a * 1e3) ** 2)
        Tz = Tm * (z / a + (2 / (n * np.pi) * e) @ S)
        k = int(np.argmax(Tz >= T))
        tab[i] = float(np.interp(T, Tz[k - 1:k + 1], z[k - 1:k + 1])) if k > 0 else 0.0
    out = np.full(age.shape, np.nan)
    ok = np.isfinite(age)
    out[ok] = np.interp(age[ok], tab_age, tab)
    return out


def roughness_lattice(step=0.025, win=0.2):
    d = xr.open_dataset(FILES["gebco"])
    z = d.z.values.astype(float)               # [lat, lon], 15"
    lo, la = d.lon.values, d.lat.values
    dx = lo[1] - lo[0]
    w = int(round(win / dx))                   # 48 nodes
    if w % 2 == 0:
        w += 1
    hw = w // 2
    # plane fit residual over a w x w window on a regular grid: centred x, y and 1 are orthogonal, so
    # RSS = sum z^2 - (sum z)^2 / n - (sum x z)^2 / sum x^2 - (sum y z)^2 / sum y^2
    n = w * w
    off = np.arange(-hw, hw + 1, dtype=float)
    sx2 = n / w * np.sum(off ** 2)
    from scipy.ndimage import correlate1d
    S = uniform_filter(z, size=w, mode="nearest") * n
    S2 = uniform_filter(z * z, size=w, mode="nearest") * n
    Sx = correlate1d(uniform_filter(z, size=(w, 1), mode="nearest") * w, off, axis=1, mode="nearest")
    Sy = correlate1d(uniform_filter(z, size=(1, w), mode="nearest") * w, off, axis=0, mode="nearest")
    rss = S2 - S ** 2 / n - Sx ** 2 / sx2 - Sy ** 2 / sx2
    rms = np.sqrt(np.maximum(rss, 0) / n)
    ocean = uniform_filter((z < 0).astype(float), size=w, mode="nearest") > 0.9999
    rms = np.where(ocean, rms, np.nan)
    # edges: windows that run off the cut are not valid
    rms[:hw, :] = np.nan
    rms[-hw:, :] = np.nan
    rms[:, :hw] = np.nan
    rms[:, -hw:] = np.nan
    k = max(1, int(round(step / dx)))
    return lo[::k], la[::k], rms[::k, ::k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v61")
    a = ap.parse_args()
    c = pd.read_csv(Q.CATS[a.tag]["csv"], usecols=["ID", "lat", "lon", "loc_source"], low_memory=False)
    c = c[c.loc_source.ne("none")].reset_index(drop=True)
    lon, lat = c.lon.values, c.lat.values
    out = pd.DataFrame(dict(ID=c.ID.values))
    out["age_myr"] = nearest_grid(FILES["age"], lon, lat)
    out["rate_mmyr"] = nearest_grid(FILES["rate"], lon, lat)
    out["sed_m"] = nearest_grid(FILES["sed"], lon, lat)
    hf, nhf, nhf_all = heatflow(lon, lat)
    out["heatflow_mw"], out["n_heatflow"] = hf, nhf
    out["z600_km"] = gdh1_isotherm(out.age_myr.values)
    rlo, rla, rms = roughness_lattice()
    ix = np.clip(np.round((lon - rlo[0]) / (rlo[1] - rlo[0])).astype(int), 0, len(rlo) - 1)
    iy = np.clip(np.round((lat - rla[0]) / (rla[1] - rla[0])).astype(int), 0, len(rla) - 1)
    out["rough_m"] = rms[iy, ix]
    f = Q.covariate_file(a.tag)
    f.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(f, index=False)
    prov = dict(tag=a.tag, catalog=str(Q.CATS[a.tag]["csv"]), n_events=len(out), heatflow_points_in_box=nhf_all,
                files={k: str(v) for k, v in FILES.items()},
                md5={k: Q.md5(v) for k, v in FILES.items() if k != "gebco"},
                gebco_source=str(EXT / "gebco_2026/gebco_2026_n90.0_s0.0_w-180.0_e-90.0_geotiff.tif"),
                coverage={k: float(out[k].notna().mean()) for k in Q.COVS if k in out.columns})
    json.dump(prov, open(f.with_suffix(".json"), "w"), indent=1)
    Q.log(json.dumps(prov["coverage"], indent=1))
    Q.log(out.describe().T.to_string())


if __name__ == "__main__":
    main()

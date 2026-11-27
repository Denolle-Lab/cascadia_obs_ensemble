"""Per-event and 0.1 deg lattice values of the external geophysical data (D-B114).

Reads the catalog (EXTGEO_CATALOG, default v6.1: ID, lat, lon, depth, surface_km) and the raw downloads; writes
  derived/events_external_geo_<tag>.parquet (+ .csv.gz)  one row per located event, keyed by ID
  derived/lattice_external_geo_0p1.nc                    the same fields on the 0.1 deg lattice, 39-52 N, 132-120 W
  derived/ghfdb_sites_area.csv                           heat-flow sites used (area + 1 deg pad)
The segment join is a separate, cheap step (eg_segments.py), because the segment definitions are still changing.

Field definitions
  age_myr        Seton et al. (2020) age grid, GTS2012, 1 arc-min, bilinear (NaN = no oceanic crust in the grid)
  age_min10/max10 min / max of the 1' age grid in a +-10 km box (age contrast across transforms and ridges)
  age_misfit_myr, age_conf   Seton et al. (2020) misfit and confidence grids (GeeK2007, 6'), nearest node
  full_rate_mmyr Seton et al. (2020) full spreading rate at which the crust was formed (GTS2012, 6'), bilinear
  sed_thk_m      GlobSed v3 (Straume et al. 2019), 5', bilinear
  crust1_thk_km  CRUST1.0 crustal thickness (Laske et al. 2013), 1 deg cell
  hf_near_*      nearest GHFDB R2024 site (parent heat-flow value) within 20 km: q, distance, site ID, reference
  hf_n20         number of GHFDB sites within 20 km
  hf_smooth      Gaussian-weighted median of GHFDB site values, sigma = 15 km, sites within 45 km; NaN unless the
                 effective number of sites (sum w)^2 / sum w^2 >= 3 (hf_neff)
  q_gdh1, q_hsc  conductive surface heat flow predicted from age_myr (GDH1 plate; half-space)
  iso{350,600,800}_{hsc,plate}_bsf  isotherm depth below the seafloor (km) from age_myr; _bsl adds the water depth
  water_depth_km merged GEBCO_2026/GMRT elevation at the epicentre (500 m grid, nearest node), positive down offshore
  rough5/10/20, landfrac20, directfrac10, fabric_az20, aniso20, bathy_src: see eg_bathy.py
  slab_km        Slab2 depth below the epicentre (Hayes et al. 2018), bilinear; rel_slab = depth - slab_km
  Tslab_{D80,D80new}  slab-surface temperature (degC) at slab_km on the nearer van Keken & Wilson (2023) Cascadia
                 cross-section (slab_model: NCas47N or CCas42N, split at 44.5 N); NaN outside the model depth range
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import xarray as xr
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))   # run with python -I (downloads are untrusted data)
import eg_bathy as B  # noqa: E402
import eg_common as C  # noqa: E402

SLAB2 = Path("/home/mdenolle/cascadia_obs_ensemble/data/slab2/cas_slab2_dep.xyz")
HF_NEAR_KM, HF_SIGMA_KM, HF_CUT_KM, HF_NEFF_MIN = 20.0, 15.0, 45.0, 3.0
ISOS = (350, 600, 800)
SLAB_SPLIT_LAT = 44.5


def load_points():
    c = pd.read_csv(C.CATALOG, usecols=["ID", "lat", "lon", "depth", "surface_km", "loc_source"], low_memory=False)
    c = c[c.loc_source.ne("none") & np.isfinite(c.lat) & np.isfinite(c.lon)].drop(columns="loc_source")
    return c.reset_index(drop=True)


def weighted_median(v, w):
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    return v[o][np.searchsorted(cw, 0.5 * cw[-1])]


def heat_flow(sites, lon, lat):
    tree = cKDTree(C.unit_xyz(sites.lon.values, sites.lat.values))
    P = C.unit_xyz(lon, lat)
    d, i = tree.query(P, k=1, distance_upper_bound=C.km_to_chord(HF_NEAR_KM))
    ok = np.isfinite(d)
    out = pd.DataFrame(index=range(len(lon)))
    ii = np.where(ok, i, 0)
    out["hf_near_q"] = np.where(ok, sites.q.values[ii], np.nan)
    out["hf_near_km"] = np.where(ok, C.chord_to_km(np.where(ok, d, 0)), np.nan)
    out["hf_near_site"] = np.where(ok, sites.site_id.values[ii], None)
    out["hf_near_ref"] = np.where(ok, sites.ref.values[ii], None)
    out["hf_n20"] = [len(x) for x in tree.query_ball_point(P, C.km_to_chord(HF_NEAR_KM))]
    q = sites.q.values
    sm, neff = np.full(len(lon), np.nan), np.zeros(len(lon))
    # unique query positions (many events share a cell): round to ~1 km
    key = np.round(np.c_[lon, lat] * 100).astype(np.int64)
    uk, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    Pu = C.unit_xyz(uk[:, 0] / 100.0, uk[:, 1] / 100.0)
    lists = tree.query_ball_point(Pu, C.km_to_chord(HF_CUT_KM))
    smu, neffu = np.full(len(uk), np.nan), np.zeros(len(uk))
    for j, L in enumerate(lists):
        if not L:
            continue
        L = np.asarray(L)
        dk = C.chord_to_km(np.linalg.norm(tree.data[L] - Pu[j], axis=1))
        w = np.exp(-0.5 * (dk / HF_SIGMA_KM) ** 2)
        neffu[j] = w.sum() ** 2 / (w ** 2).sum()
        if neffu[j] >= HF_NEFF_MIN:
            smu[j] = weighted_median(q[L], w)
    sm, neff = smu[inv], neffu[inv]
    out["hf_smooth"], out["hf_neff"] = sm, neff
    return out


def age_fields(lon, lat):
    glon, glat, age = C.read_grid(C.F_AGE, pad=1.5)
    out = {"age_myr": C.sample(glon, glat, age, lon, lat)}
    # +-10 km box min/max on the 1' grid (lat half-width 10 km; lon half-width at the box latitude 45.5 N)
    hy = int(np.ceil(10.0 / (111.195 / 60)))
    hx = int(np.ceil(10.0 / (111.195 * np.cos(np.radians(45.5)) / 60)))
    big = np.where(np.isnan(age), -1.0, age)
    small = np.where(np.isnan(age), 1e9, age)
    amax = maximum_filter(big, size=(2 * hy + 1, 2 * hx + 1), mode="nearest")
    amin = minimum_filter(small, size=(2 * hy + 1, 2 * hx + 1), mode="nearest")
    amax = C.sample(glon, glat, amax, lon, lat, "nearest")
    amin = C.sample(glon, glat, amin, lon, lat, "nearest")
    out["age_max10"] = np.where(amax < 0, np.nan, amax)
    out["age_min10"] = np.where(amin > 1e8, np.nan, amin)
    for k, f, m in (("age_misfit_myr", C.F_AGE_MISFIT, "nearest"), ("age_conf", C.F_AGE_CONF, "nearest"),
                    ("full_rate_mmyr", C.F_RATE, "linear")):
        a, b, z = C.read_grid(f, pad=1.5)
        out[k] = C.sample(a, b, z, lon, lat, m)
    a, b, z = C.read_grid(C.F_SED, pad=1.5)
    out["sed_thk_m"] = C.sample(a, b, z, lon, lat)
    a, b, z = C.read_crust1()
    out["crust1_thk_km"] = C.sample(a, b, z, lon, lat, "nearest")
    return out


def bathy_fields(lon, lat):
    ds = xr.open_dataset(C.DER / "bathy_roughness_500m.nc")
    x, y = ds.x.values, ds.y.values
    fx, fy = pyproj.Transformer.from_crs("EPSG:4326", B.PROJ, always_xy=True).transform(lon, lat)
    ix = np.clip(np.round((fx - x[0]) / B.DX).astype(int), 0, x.size - 1)
    iy = np.clip(np.round((fy - y[0]) / B.DX).astype(int), 0, y.size - 1)
    out = {}
    for k in ("rough5", "rough10", "rough20", "landfrac20", "directfrac10", "fabric_az20", "aniso20", "src", "tid"):
        out["bathy_src" if k == "src" else ("gebco_tid" if k == "tid" else k)] = ds[k].values[iy, ix]
    out["water_depth_km"] = -ds["z"].values[iy, ix].astype(float) / 1e3
    return out


def slab_fields(lon, lat, depth):
    s = pd.read_csv(SLAB2, header=None, names=["lon", "lat", "z"])
    s["lon"] = np.where(s.lon > 180, s.lon - 360, s.lon)
    lo, la = np.sort(s.lon.unique()), np.sort(s.lat.unique())
    grid = s.pivot_table(index="lat", columns="lon", values="z", dropna=False).reindex(index=la, columns=lo).values
    slab = C.sample(lo, la, -grid, lon, lat)
    out = {"slab_km": slab, "rel_slab": depth - slab if depth is not None else np.full(len(lon), np.nan)}
    model = np.where(np.asarray(lat) >= SLAB_SPLIT_LAT, "NCas47N", "CCas42N")
    out["slab_model"] = np.where(np.isfinite(slab), model, None)
    for suite in ("D80", "D80new"):
        T = np.full(len(lon), np.nan)
        for m in C.SLAB_MODELS:
            dep, Ts = C.slab_T_vs_depth(suite, m, 98)
            sel = (model == m) & np.isfinite(slab)
            T[sel] = np.interp(slab[sel], dep, Ts, left=np.nan, right=np.nan)
        out[f"Tslab_{suite}"] = T
    return out


def thermal_fields(age, water_km):
    out = {}
    for T in ISOS:
        out[f"iso{T}_hsc_bsf"] = C.iso_hsc(age, T)
        out[f"iso{T}_plate_bsf"] = C.iso_plate(age, T)
    for m in ("hsc", "plate"):
        out[f"iso600_{m}_bsl"] = out[f"iso600_{m}_bsf"] + np.where(water_km > 0, water_km, 0.0)
    out["q_gdh1"] = C.q_gdh1(age)
    out["q_hsc"] = C.q_hsc(age)
    out["age_floored"] = np.isfinite(age) & (age < C.AGE_FLOOR)
    for k in list(out):
        if k != "age_floored":
            out[k] = np.where(np.isfinite(age), out[k], np.nan)
    return out


def build(lon, lat, depth, sites):
    t0 = time.time()
    cols = {}
    cols.update(age_fields(lon, lat))
    C.log(f"  age/sediment/crust ({time.time() - t0:.0f} s)")
    cols.update(bathy_fields(lon, lat))
    C.log(f"  bathymetry ({time.time() - t0:.0f} s)")
    cols.update(thermal_fields(cols["age_myr"], cols["water_depth_km"]))
    cols.update(slab_fields(lon, lat, depth))
    C.log(f"  thermal/slab ({time.time() - t0:.0f} s)")
    df = pd.DataFrame(cols)
    hf = heat_flow(sites, lon, lat)
    C.log(f"  heat flow ({time.time() - t0:.0f} s)")
    return pd.concat([df, hf], axis=1)


def main():
    C.DER.mkdir(parents=True, exist_ok=True)
    sites = C.read_ghfdb(pad=1.0)
    sites.to_csv(C.DER / "ghfdb_sites_area.csv", index=False)
    C.log(f"GHFDB sites (area + 1 deg pad): {len(sites)}; in area {int(sites.in_area.sum())}")

    ev = load_points()
    C.log(f"events: {len(ev)} ({C.CATALOG.name})")
    out = build(ev.lon.values, ev.lat.values, ev.depth.values, sites)
    out.insert(0, "ID", ev.ID.values)
    out.insert(1, "lat", ev.lat.values)
    out.insert(2, "lon", ev.lon.values)
    out.insert(3, "depth", ev.depth.values)
    out.insert(4, "bsf", ev.depth.values + ev.surface_km.values)
    out["catalog"] = C.CATALOG.name
    f = C.DER / f"events_external_geo_{C.CATTAG}.parquet"
    out.to_parquet(f, index=False)
    out.to_csv(str(f).replace(".parquet", ".csv.gz"), index=False, float_format="%.5g")
    C.log(f"wrote {f} ({len(out)} rows, {out.shape[1]} columns)")

    lon2, lat2 = C.lattice()
    lat_df = build(lon2.ravel(), lat2.ravel(), None, sites)
    ds = xr.Dataset(coords=dict(lat=C.LAT_NODES, lon=C.LON_NODES))
    for k in lat_df.columns:
        if k in ("rel_slab",) or lat_df[k].dtype == object:
            continue
        ds[k] = (("lat", "lon"), lat_df[k].values.reshape(lon2.shape).astype(float))
    ds.attrs["method"] = __doc__
    ds.to_netcdf(C.DER / "lattice_external_geo_0p1.nc")
    C.log(f"wrote lattice ({ds.sizes})")


if __name__ == "__main__":
    main()

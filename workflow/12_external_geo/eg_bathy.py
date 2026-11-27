"""Merged bathymetry on a 500 m projected grid, seafloor roughness and abyssal-hill fabric (D-B114).

Inputs
  GEBCO_2026 (15 arc-sec, tile n90 s0 w-180 e-90, cut to 37.5-53.5 N, 134-118 W with gdal_translate) and its Type
  Identifier (TID) grid; the local GMRT v4 GridServer tiles in /wd1/mdenolle_data/reloc_v5/topo/gmrt (11 boxes).
Method
  1. Projected grid: transverse Mercator (WGS84, lon_0 = -126, lat_0 = 45.5, k0 = 1), 500 m spacing, covering the
     study area 39-52 N, 132-120 W plus 15 km. Scale error < 0.3 % in the area.
  2. Elevation = GEBCO_2026 (bilinear); replaced by GMRT where a GMRT tile covers the node (tiles block-averaged to
     >= 400 m first, finer tiles applied last). src = 1 where GMRT. The GMRT tiles carry GMRT's own background where
     there is no multibeam, so the GEBCO TID class (nearest cell) is the data-quality flag in both cases: TID 10-17 =
     direct measurement, 40-48 = interpolated/predicted, 70-72 = pre-gridded/unknown, 0 = land.
  3. Roughness R_L = RMS of the residual after a least-squares plane fit in a square L x L window centred on each
     node, for L = 5, 10, 20 km (11, 21, 41 nodes: 5.5, 10.5, 20.5 km). Box filters give the plane fit exactly,
     because the window coordinates are centred: var = <z2> - <z>2 - <x'z>2/<x'2> - <y'z>2/<y'2>.
  4. Fabric (20 km window): gradient structure tensor of the elevation with the window-mean gradient removed
     (= plane removed); fabric azimuth = strike of the dominant relief, i.e. 90 deg from the principal gradient
     direction, converted from grid north to true north (grid azimuth of the local meridian, by finite difference); anisotropy =
     (l1 - l2)/(l1 + l2), 0 = isotropic, 1 = perfectly lineated.
  5. Per window: fraction of land nodes (elevation > 0) and fraction of TID-direct nodes.
Output
  derived/bathy_roughness_500m.nc (projected grid, float32) and derived/bathy_roughness_lattice_0p1.nc (sampled on
  the 0.1 deg lattice, nearest projected node).
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pyproj
import xarray as xr
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))   # run with python -I (downloads are untrusted data)
import eg_common as C  # noqa: E402

DX = 500.0
PROJ = "+proj=tmerc +lat_0=45.5 +lon_0=-126 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
WINDOWS_KM = (5, 10, 20)
PAD_KM = 15.0
GMRT_MIN_DX_M = 400.0


def grid_nodes():
    tr = pyproj.Transformer.from_crs("EPSG:4326", PROJ, always_xy=True)
    blon = np.r_[np.linspace(C.LON0, C.LON1, 200), np.full(200, C.LON1), np.linspace(C.LON1, C.LON0, 200), np.full(200, C.LON0)]
    blat = np.r_[np.full(200, C.LAT0), np.linspace(C.LAT0, C.LAT1, 200), np.full(200, C.LAT1), np.linspace(C.LAT1, C.LAT0, 200)]
    bx, by = tr.transform(blon, blat)
    p = PAD_KM * 1e3
    x = np.arange(np.floor((bx.min() - p) / DX) * DX, bx.max() + p + DX, DX)
    y = np.arange(np.floor((by.min() - p) / DX) * DX, by.max() + p + DX, DX)
    return x, y


def block_mean(lon, lat, z, f):
    if f <= 1:
        return lon, lat, z
    ny, nx = (z.shape[0] // f) * f, (z.shape[1] // f) * f
    zz = z[:ny, :nx].reshape(ny // f, f, nx // f, f)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN blocks (tile edges) stay NaN
        zb = np.nanmean(zz, axis=(1, 3))
    return lon[:nx].reshape(-1, f).mean(1), lat[:ny].reshape(-1, f).mean(1), zb


def box(a, n):
    return uniform_filter(a, size=n, mode="nearest")


def main():
    t0 = time.time()
    C.DER.mkdir(parents=True, exist_ok=True)
    x, y = grid_nodes()
    X, Y = np.meshgrid(x, y)
    inv = pyproj.Transformer.from_crs(PROJ, "EPSG:4326", always_xy=True)
    LON, LAT = inv.transform(X, Y)
    C.log(f"grid {x.size} x {y.size} = {X.size / 1e6:.1f} M nodes at {DX:.0f} m")

    glon, glat, gz = C.read_ehdr(C.F_GEBCO)
    z = C.sample(glon, glat, gz, LON, LAT)
    tlon, tlat, tid = C.read_ehdr(C.F_GEBCO_TID)
    tid = C.sample(tlon, tlat, tid, LON, LAT, method="nearest")
    src = np.zeros(z.shape, np.int8)
    tiles = []
    for f in sorted(C.GMRT_DIR.glob("gmrt_*.nc")):
        d = xr.open_dataset(f)
        lo, la, zz = d.lon.values.astype(float), d.lat.values.astype(float), d.altitude.values.astype(float)
        if la[0] > la[-1]:
            la, zz = la[::-1], zz[::-1]
        dlat_m = abs(np.diff(la).mean()) * 111195.0
        fac = max(1, int(round(GMRT_MIN_DX_M / dlat_m)))
        tiles.append((dlat_m, f.name, *block_mean(lo, la, zz, fac), fac))
    for dlat_m, name, lo, la, zz, fac in sorted(tiles, key=lambda t: -t[0]):      # coarse first, finest last
        v = C.sample(lo, la, zz, LON, LAT)
        m = np.isfinite(v)
        z[m], src[m] = v[m], 1
        C.log(f"  GMRT {name}: {dlat_m:.0f} m x{fac} -> {m.sum()} nodes")
    C.log(f"GMRT nodes {src.mean():.3f} of grid; NaN {np.isnan(z).sum()}  ({time.time() - t0:.0f} s)")
    z = np.where(np.isnan(z), np.nanmean(z), z)       # none expected; keeps the box filters finite

    direct = ((tid >= 10) & (tid <= 17)).astype(float)
    land = (z > 0).astype(float)
    Xl = (np.arange(x.size) * DX)[None, :] * np.ones((y.size, 1))
    Yl = (np.arange(y.size) * DX)[:, None] * np.ones((1, x.size))
    out = dict(z=z.astype(np.float32), src=src, tid=np.nan_to_num(tid, nan=-1).astype(np.int8))
    for L in WINDOWS_KM:
        n = 2 * int(round(L * 1e3 / (2 * DX))) + 1
        m = box(z, n)
        sxx = DX ** 2 * (n ** 2 - 1) / 12.0
        cx = box(Xl * z, n) - Xl * m
        cy = box(Yl * z, n) - Yl * m
        var = box(z * z, n) - m * m - cx ** 2 / sxx - cy ** 2 / sxx
        out[f"rough{L}"] = np.sqrt(np.maximum(var, 0)).astype(np.float32)
        out[f"landfrac{L}"] = np.clip(box(land, n), 0, 1).astype(np.float32)
        out[f"directfrac{L}"] = np.clip(box(direct, n), 0, 1).astype(np.float32)
        C.log(f"  roughness L={L} km (n={n}, {n * DX / 1e3:.1f} km)  ({time.time() - t0:.0f} s)")
    # fabric, 20 km window
    n = 2 * int(round(20e3 / (2 * DX))) + 1
    gy, gx = np.gradient(z, DX)
    mx, my = box(gx, n), box(gy, n)
    j11 = box(gx * gx, n) - mx * mx
    j22 = box(gy * gy, n) - my * my
    j12 = box(gx * gy, n) - mx * my
    theta_g = 0.5 * np.degrees(np.arctan2(2 * j12, j11 - j22))     # principal gradient, deg ccw from grid +x (east)
    az_grid = (90.0 - (theta_g + 90.0)) % 180.0                     # fabric strike, deg cw from grid north
    fwd = pyproj.Transformer.from_crs("EPSG:4326", PROJ, always_xy=True)
    xn, yn = fwd.transform(LON, LAT + 0.01)
    conv = np.degrees(np.arctan2(xn - X, yn - Y))                   # grid azimuth of true north (deg)
    out["fabric_az20"] = ((az_grid - conv) % 180.0).astype(np.float32)
    out["aniso20"] = (np.sqrt((j11 - j22) ** 2 + 4 * j12 ** 2) / np.maximum(j11 + j22, 1e-12)).astype(np.float32)
    out["conv"] = conv.astype(np.float32)
    C.log(f"fabric done ({time.time() - t0:.0f} s)")

    ds = xr.Dataset({k: (("y", "x"), v) for k, v in out.items()}, coords=dict(x=x, y=y))
    ds["lon"] = (("y", "x"), LON.astype(np.float32))
    ds["lat"] = (("y", "x"), LAT.astype(np.float32))
    ds.attrs.update(proj=PROJ, dx_m=DX, method=__doc__, windows_km=str(WINDOWS_KM),
                    sources="GEBCO_2026 (doi:10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa); GMRT v4 GridServer tiles")
    enc = {k: dict(zlib=True, complevel=4) for k in ds.data_vars}
    f_out = C.DER / "bathy_roughness_500m.nc"
    ds.to_netcdf(f_out, encoding=enc)
    C.log(f"wrote {f_out}")

    # sample on the 0.1 deg lattice (nearest projected node)
    lon2, lat2 = C.lattice()
    fx, fy = pyproj.Transformer.from_crs("EPSG:4326", PROJ, always_xy=True).transform(lon2, lat2)
    ix = np.clip(np.round((fx - x[0]) / DX).astype(int), 0, x.size - 1)
    iy = np.clip(np.round((fy - y[0]) / DX).astype(int), 0, y.size - 1)
    lat_ds = xr.Dataset({k: (("lat", "lon"), np.asarray(v)[iy, ix]) for k, v in out.items() if k != "conv"},
                        coords=dict(lat=C.LAT_NODES, lon=C.LON_NODES))
    lat_ds.attrs.update(note="nearest node of bathy_roughness_500m.nc", method=__doc__)
    lat_ds.to_netcdf(C.DER / "bathy_roughness_lattice_0p1.nc")
    C.log(f"wrote lattice; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

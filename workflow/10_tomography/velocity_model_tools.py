"""Point queries of the Cascadia OBS tomography velocity models (cascadia_obs_tomo), with the solid surface and the
D-B073 shallow oceanic crust rule. Shipped with read_velocity_model.py in velocity_models.zip; numpy, xarray, pyproj.

    from velocity_model_tools import velocity_at
    r = velocity_at("velocity_models/cascadia_obs_tomo", "0.3.2", lat, lon, depth_km)   # mode="field" (default)
    r["Vp"], r["Vs"], r["VpVs"], r["medium"]        # medium 0 rock, 1 water, 2 air; Vs NaN in water, all NaN in air

Depth in km below sea level, positive down. Modes (v0.3.2 and later; earlier versions have only "nodes"):
* "field" (default): the rule of the v5.1 relocation's velocity field, evaluated at the point itself. Medium from the
  point's depth against the merged solid surface (D-B068; bilinear between the 1 km lattice cell centres of
  v<version>_surface_lattice.nc): fluid iff -depth > surface, water iff depth >= 0. Rock: Vp, Vs = trilinear
  interpolation of Vp_tomography, Vs_tomography (the inversion's node values, tomoDD's lookup); in offshore columns
  (lattice cell containing the point), below sea level and at most shallow_crust_thickness_km below the seafloor:
  Vs = min(Vs, Vp / shallow_crust_vpvs) (D-B073). Water: Vp 1.5 km/s, Vs NaN. Air: NaN. At the 1 km cell centres this
  equals the NonLinLoc grids (the production field) inside the model volume; outside the model volume: NaN.
* "nll": the value of the NonLinLoc grid cell holding the point (1 km cells, piecewise constant, unflattened to true
  velocity): exactly the velocities Grid2Time used for the v5.1 relocation, including water (Vp 1.5, Vs 0.4 km/s),
  air (0.3 / 0.3 km/s), the station rock patches and the D-B005 model outside the tomography volume.
* "nodes": trilinear interpolation of the node variables Vp and Vs as stored. Above the surface they are water (Vp 1.5,
  Vs NaN) or air (NaN), so within one node spacing of the surface the result mixes rock with water or is NaN, and the
  node values carry only a 2-km-node approximation of D-B073: use "field" for points near the surface.
"""
from pathlib import Path

import numpy as np
import xarray as xr

RE = 6371.0


def _axis(nodes, q):
    i = np.clip(np.searchsorted(nodes, q, side="right") - 1, 0, len(nodes) - 2)
    return i, np.clip((q - nodes[i]) / (nodes[i + 1] - nodes[i]), 0.0, 1.0)


def trilinear(ds, var, lat, lon, depth):
    """tomoDD's trilinear lookup of a node variable at points (NaN outside the node volume)."""
    lat, lon, depth = (np.atleast_1d(np.asarray(v, float)) for v in (lat, lon, depth))
    lo, la, z = ds.longitude.values, ds.latitude.values, ds.depth.values
    v = np.asarray(ds[var].values, float)                       # [depth, latitude, longitude]
    ix, wx = _axis(lo, lon)
    iy, wy = _axis(la, lat)
    iz, wz = _axis(z, depth)
    out = 0.0
    for dz, az in ((0, 1 - wz), (1, wz)):
        for dy, ay in ((0, 1 - wy), (1, wy)):
            for dx, ax in ((0, 1 - wx), (1, wx)):
                out = out + az * ay * ax * v[iz + dz, iy + dy, ix + dx]
    inside = ((lon >= lo[0]) & (lon <= lo[-1]) & (lat >= la[0]) & (lat <= la[-1]) & (depth >= z[0]) & (depth <= z[-1]))
    return np.where(inside, out, np.nan)


def _proj():
    from pyproj import Proj
    return Proj(proj="lcc", lat_0=45.0, lon_0=-125.8, lat_1=41.0, lat_2=49.0, ellps="WGS84")


def _open(root, version):
    root = Path(root)
    z = root / f"v{version}.zarr"
    return xr.open_zarr(z) if z.exists() else xr.open_dataset(root / f"v{version}.nc")


def velocity_at(root, version, lat, lon, depth, mode="field"):
    """Vp, Vs, VpVs (km/s) and medium (0 rock, 1 water, 2 air; -1 unknown) at points. See the module docstring."""
    root = Path(root)
    lat, lon, depth = (np.atleast_1d(np.asarray(v, float)) for v in (lat, lon, depth))
    ds = _open(root, version)
    if mode == "nodes":
        vp, vs = trilinear(ds, "Vp", lat, lon, depth), trilinear(ds, "Vs", lat, lon, depth)
        med = np.full(lat.shape, -1, "int8")
        return dict(Vp=vp, Vs=vs, VpVs=vp / vs, medium=med)
    if mode == "nll":
        return _nll(root / f"v{version}_nll", lat, lon, depth)
    if mode != "field":
        raise ValueError(mode)
    L = xr.open_dataset(root / f"v{version}_surface_lattice.nc")
    xs, ys = L.x.values, L.y.values
    x, y = (np.asarray(c) / 1e3 for c in _proj()(lon, lat))
    # surface: bilinear between cell centres (equal to the cell-centre value at a centre); offshore: containing cell
    fx, fy = (x - xs[0]) / (xs[1] - xs[0]), (y - ys[0]) / (ys[1] - ys[0])
    i0 = np.clip(np.floor(fx).astype(int), 0, len(xs) - 2)
    j0 = np.clip(np.floor(fy).astype(int), 0, len(ys) - 2)
    wx, wy = np.clip(fx - i0, 0, 1), np.clip(fy - j0, 0, 1)
    S = np.asarray(L.surface_km.values, float)
    surf = ((1 - wx) * (1 - wy) * S[i0, j0] + wx * (1 - wy) * S[i0 + 1, j0] + (1 - wx) * wy * S[i0, j0 + 1]
            + wx * wy * S[i0 + 1, j0 + 1])
    ic = np.clip(np.floor(fx + 0.5).astype(int), 0, len(xs) - 1)          # cell containing the point
    jc = np.clip(np.floor(fy + 0.5).astype(int), 0, len(ys) - 1)
    off = L.offshore.values[ic, jc].astype(bool)
    above = -depth > surf
    med = np.where(above, np.where(depth >= 0, 1, 2), 0).astype("int8")
    vp = trilinear(ds, "Vp_tomography", lat, lon, depth)
    vs = trilinear(ds, "Vs_tomography", lat, lon, depth)
    a = ds.attrs
    rule = off & (med == 0) & (depth > 0) & (depth + surf <= float(a["shallow_crust_thickness_km"]))
    vs = np.where(rule, np.minimum(vs, vp / float(a["shallow_crust_vpvs"])), vs)
    vp = np.where(med == 0, vp, np.where(med == 1, 1.5, np.nan))
    vs = np.where(med == 0, vs, np.nan)
    return dict(Vp=vp, Vs=vs, VpVs=vp / vs, medium=med, surface_km=surf, offshore=off, shallow_crust=rule)


def _nll(folder, lat, lon, depth):
    """Cell values of the Earth-flattened NonLinLoc SLOW_LEN grids, as true velocities."""
    out = {}
    x, y = (np.asarray(c) / 1e3 for c in _proj()(lon, lat))
    zf = RE * np.log(RE / (RE - depth))
    for ph in ("P", "S"):
        h = (folder / f"model.{ph}.mod.hdr").read_text().split()
        nx, ny, nz = map(int, h[:3])
        x0, y0, z0, dx = map(float, h[3:7])
        a = np.memmap(folder / f"model.{ph}.mod.buf", dtype="<f4", mode="r", shape=(nx, ny, nz))
        i, j, k = (np.floor((c - o) / dx).astype(int) for c, o in ((x, x0), (y, y0), (zf, z0)))
        ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny) & (k >= 0) & (k < nz)
        v = np.full(lat.shape, np.nan)
        zc = z0 + dx * (k[ok] + 0.5)                             # flattened cell-centre depth
        v[ok] = dx / (np.asarray(a[i[ok], j[ok], k[ok]], float) * np.exp(zc / RE))
        out[f"V{ph.lower()}"] = v
    vp = out["Vp"]
    med = np.where(np.isnan(vp), -1, np.where(vp >= 1.6, 0, np.where(depth >= 0, 1, 2))).astype("int8")
    return dict(Vp=vp, Vs=out["Vs"], VpVs=vp / out["Vs"], medium=med)

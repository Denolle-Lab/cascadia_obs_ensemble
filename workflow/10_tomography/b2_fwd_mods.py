"""Forward-accuracy test, step 3: alternative tomoDD MOD files built from the v4-NLL3D model.

Same lon/lat/edge node layout as inputs/MOD (b2_tomodd_inputs.py); what changes is how the NLL model is
sampled at the nodes:
  vel11   velocity (arithmetic) mean of rock cells over an 11 km box (= inputs/MOD, rebuilt here as a check)
  slow11  slowness mean (harmonic velocity mean) over 11 km
  slow5   slowness mean over 5 km
  point   the NLL rock column nearest to the node (box 1 cell)
  slow5z  slowness mean over 5 km with 1 km depth nodes to 9 km (-3, -1, 0, 1, ..., 9, 12, ...)
  pointz1 nearest column, 1 km depth nodes (-3..80; NLL-only diagnostic, exceeds tomoDD's 50 depth nodes)
  pointz2 nearest column, 1 km nodes to 9 km then 2 km to 49 km (42 depth nodes)
  slow5z2 slowness mean over 5 km, same depth nodes as pointz2
  oned    laterally homogeneous: median of inputs/MOD interior nodes per depth (1D sanity test)
All: rock cells only (Vp > 1.6), above the rock surface the shallowest rock level of the box, linear in depth
between the 1 km NLL levels, below 80 km and edges as in b2_tomodd_inputs.py.

Also: ellipsoid_coords(), the coordinate rescaling that makes tomoDD's spherical (R = 6371 km, geographic
latitude) distances equal WGS-84 distances around the region centre (used by b2_fwd_tomodd.py).

  python b2_fwd_mods.py [variant ...]     -> OUT/mods/<variant>.MOD
"""
import sys

import numpy as np
import pandas as pd

from b2_fwd_common import INPUTS, NLL_MODEL, OUT, PROJ, RE, read_mod, write_mod

DEEP_VP = {100: 8.05, 125: 8.10, 400: 8.90}
Z_FINE = [-3, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 18, 21, 25, 29, 34, 40, 47, 55, 65, 80, 100, 125]
Z_2KM = [-3, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9] + list(range(11, 50, 2)) + [52, 56, 60, 65, 70, 80, 100, 125]
Z_1KM = list(range(-3, 81)) + [100, 125]       # NLL-only diagnostic (tomoDD allows at most 50 depth nodes)
VARIANTS = {"pointz1": (1, "slow", Z_1KM), "slow5z2": (5, "slow", Z_2KM), "pointz2": (1, "slow", Z_2KM),
            "vel11": (11, "vel", None), "slow11": (11, "slow", None), "slow5": (5, "slow", None),
            "point": (1, "slow", None), "slow5z": (5, "slow", Z_FINE)}

WGS_A, WGS_F = 6378.137, 1 / 298.257223563


def radii(lat_deg):
    e2 = WGS_F * (2 - WGS_F)
    s2 = np.sin(np.radians(lat_deg)) ** 2
    M = WGS_A * (1 - e2) / (1 - e2 * s2) ** 1.5
    N = WGS_A / np.sqrt(1 - e2 * s2)
    return M, N


def ellipsoid_coords(lat, lon, lat_c=41.3, lon_c=-125.0):
    """Linear map (lat, lon) -> (lat', lon') so that great-circle distances on tomoDD's sphere (R = 6371 km)
    match WGS-84 distances near (lat_c, lon_c): lat' - lat_c = (M/R)(lat - lat_c), lon' - lon_c = (N/R)(lon - lon_c).
    Residual scale error < 2e-4 within 2 deg of the centre."""
    M, N = radii(lat_c)
    return lat_c + (np.asarray(lat) - lat_c) * M / RE, lon_c + (np.asarray(lon) - lon_c) * N / RE


def load_nll(phase):
    h = (NLL_MODEL / f"cascadia3d.{phase}.mod.hdr").read_text().split()
    nx, ny, nz = map(int, h[:3])
    x0, y0, z0, dx = map(float, h[3:7])
    s = np.fromfile(NLL_MODEL / f"cascadia3d.{phase}.mod.buf", dtype="<f4").reshape(nx, ny, nz)
    return dx / s, (x0, y0, z0, dx)


def sample(xn, yn, zq, box, mean):
    vp, g = load_nll("P")
    vs, _ = load_nll("S")
    x0, y0, z0, dx = g
    nx, ny, nz = vp.shape
    rock = vp > 1.6
    LON, LAT = np.meshgrid(xn, yn, indexing="ij")
    X, Y = PROJ(LON, LAT)
    ix = np.rint((X / 1e3 - x0) / dx).astype(int)
    iy = np.rint((Y / 1e3 - y0) / dx).astype(int)
    h = int(box // 2)
    zlev = z0 + dx * np.arange(nz)
    zq = np.clip(np.asarray(zq, float), zlev[0], zlev[-1])
    VP = np.full((len(xn), len(yn), len(zq)), np.nan)
    VS = np.full_like(VP, np.nan)
    for i in range(len(xn)):
        for j in range(len(yn)):
            a0, a1 = max(ix[i, j] - h, 0), min(ix[i, j] + h + 1, nx)
            b0, b1 = max(iy[i, j] - h, 0), min(iy[i, j] + h + 1, ny)
            if a0 >= a1 or b0 >= b1:
                continue
            r = rock[a0:a1, b0:b1, :]
            cnt = r.sum(axis=(0, 1))
            out = []
            for v in (vp[a0:a1, b0:b1, :], vs[a0:a1, b0:b1, :]):
                with np.errstate(invalid="ignore", divide="ignore"):
                    if mean == "vel":
                        m = np.where(r, v, 0).sum(axis=(0, 1)) / cnt
                    else:
                        m = cnt / np.where(r, 1.0 / v, 0).sum(axis=(0, 1))
                out.append(m)
            mp, ms = out
            ok = cnt > 0
            if not ok.any():
                continue
            first = np.argmax(ok)
            mp[:first], ms[:first] = mp[first], ms[first]
            mp = pd.Series(mp).ffill().bfill().values
            ms = pd.Series(ms).ffill().bfill().values
            if mean == "slow":   # linear in slowness between the 1 km levels
                VP[i, j, :] = 1.0 / np.interp(zq, zlev, 1.0 / mp)
                VS[i, j, :] = 1.0 / np.interp(zq, zlev, 1.0 / ms)
            else:
                VP[i, j, :] = np.interp(zq, zlev, mp)
                VS[i, j, :] = np.interp(zq, zlev, ms)
    return VP, VS


def build(variant):
    ref = read_mod(INPUTS / "MOD")
    xi, yi = ref["lon"][1:-1], ref["lat"][1:-1]
    zt, zb = ref["z"][0], ref["z"][-1]
    if variant == "oned":
        vp = np.median(ref["vp"][1:-1, 1:-1, :], axis=(0, 1))
        r = np.median((ref["vp"] / ref["vs"])[1:-1, 1:-1, :], axis=(0, 1))
        shape = ref["vp"].shape
        m = dict(lon=ref["lon"], lat=ref["lat"], z=ref["z"], vp=np.broadcast_to(vp, shape).copy())
        m["vs"] = m["vp"] / np.broadcast_to(r, shape)
        write_mod(OUT / "mods" / "oned.MOD", m)
        return
    box, mean, zfine = VARIANTS[variant]
    zi = np.array(zfine if zfine else ref["z"][1:-1], float)
    zm = zi[zi <= 80]
    VP, VS = sample(xi, yi, zm, box, mean)
    bad = np.isnan(VP[:, :, 0])
    if bad.any():
        from scipy import ndimage
        idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
        VP, VS = VP[idx[0], idx[1]], VS[idx[0], idx[1]]
    ratio = VP / VS
    deep = [z for z in zi if z > 80]
    vp = np.concatenate([VP] + [np.full(VP.shape[:2] + (1,), DEEP_VP[int(z)]) for z in deep], axis=2)
    r = np.concatenate([ratio] + [ratio[:, :, -1:]] * len(deep), axis=2)
    vp = np.pad(vp, ((1, 1), (1, 1), (0, 0)), mode="edge")
    r = np.pad(r, ((1, 1), (1, 1), (0, 0)), mode="edge")
    vp = np.concatenate([vp[:, :, :1], vp, np.full(vp.shape[:2] + (1,), DEEP_VP[int(zb)])], axis=2)
    r = np.concatenate([r[:, :, :1], r, r[:, :, -1:]], axis=2)
    m = dict(lon=ref["lon"], lat=ref["lat"], z=np.r_[zt, zi, zb], vp=vp, vs=vp / r)
    write_mod(OUT / "mods" / f"{variant}.MOD", m)
    if variant == "vel11":
        d = np.abs(vp - ref["vp"]).max()
        print(f"vel11 vs inputs/MOD: max |dVp| {d:.4f} km/s (rounding only expected)")


def main():
    (OUT / "mods").mkdir(parents=True, exist_ok=True)
    for v in sys.argv[1:] or ["oned"] + list(VARIANTS):
        build(v)
        print("built", v, flush=True)


if __name__ == "__main__":
    main()

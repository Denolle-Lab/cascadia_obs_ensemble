"""Forward-accuracy test, step 2: NonLinLoc Grid2Time (Podvin-Lecomte) times for the subset in several models.

  python b2_fwd_nll.py VARIANT [VARIANT ...]

Variants (model / cell size / cell convention / Earth flattening):
  orig_c1      v4-NLL3D model (node values of model_vpvs176), 1 km, node value at the cell centre
               (production grids put the node value on the cell [k, k+1]: half-cell shift down/east/north)
  orig_c05     same model (nearest node), 0.5 km cells
  orig_c05_eft same, Earth-flattening transform
  origlin_*    v4-NLL3D model as a continuous field: trilinear interpolation of the node velocities
  mod_c1       tomoDD MOD (inputs/MOD, tomoDD trilinear lookup, depth clamped at 0), 1 km, cell centred
  mod_c05      same, 0.5 km
  mod_c05_eft  same, Earth-flattening transform (reference for the tomoDD ray tracer)
  <name>_c05[_eft] with OUT/mods/<name>.MOD for any other MOD (built by b2_fwd_mods.py)
  *_c025*      0.25 km cells (FD convergence check; use --stations A,B to limit the station set)

Station source depth: true elevation (zSrce = -elev) for MOD variants (the MOD has rock above the surface);
the production GTSRCE depth (stations_GTSRCE.in) for the orig variants (station cell made solid by a1).
Grids: OUT/nll/<variant>/<station>/ (time grids only); times at the picks: OUT/nll/<variant>.csv (t_<variant>).
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from b2_fwd_common import (INPUTS, NLL_MODEL, OUT, eft_depth, grid2time, lambert_km, mod_velocity, nll_time,
                           read_mod, read_nll_hdr)

NPROC = int(__import__("os").environ.get("FWD_NPROC", "12"))
Z_RANGE = (-5.0, 70.0)
PAD = 4.0

_CACHE = {}


def model_fn(variant):
    base = variant.split("_c")[0]
    if base == "orig":
        g = read_nll_hdr(NLL_MODEL / "cascadia3d.P.mod")
        def f_phase(phase):
            a = np.memmap(NLL_MODEL / f"cascadia3d.{phase}.mod.buf", dtype="<f4", mode="r",
                          shape=(g["nx"], g["ny"], g["nz"]))
            def fn(lon, lat, x, y, z):
                i = np.clip(np.rint((x - g["x0"]) / g["dx"]).astype(int), 0, g["nx"] - 1)
                j = np.clip(np.rint((y - g["y0"]) / g["dx"]).astype(int), 0, g["ny"] - 1)
                k = np.clip(np.rint((z - g["z0"]) / g["dx"]).astype(int), 0, g["nz"] - 1)
                return g["dx"] / a[i, j, k]
            return fn
        return f_phase
    if base == "origlin":
        g = read_nll_hdr(NLL_MODEL / "cascadia3d.P.mod")
        def f_phase(phase):
            a = np.memmap(NLL_MODEL / f"cascadia3d.{phase}.mod.buf", dtype="<f4", mode="r",
                          shape=(g["nx"], g["ny"], g["nz"]))
            def fn(lon, lat, x, y, z):
                fx, fy, fz = (x - g["x0"]) / g["dx"], (y - g["y0"]) / g["dx"], (z - g["z0"]) / g["dx"]
                i = np.clip(np.floor(fx).astype(int), 0, g["nx"] - 2)
                j = np.clip(np.floor(fy).astype(int), 0, g["ny"] - 2)
                k = np.clip(np.floor(fz).astype(int), 0, g["nz"] - 2)
                wx, wy, wz = (np.clip(f - n, 0, 1) for f, n in ((fx, i), (fy, j), (fz, k)))
                v = 0.0
                for di, ax in ((0, 1 - wx), (1, wx)):
                    for dj, ay in ((0, 1 - wy), (1, wy)):
                        for dk, az in ((0, 1 - wz), (1, wz)):
                            v = v + ax * ay * az * g["dx"] / a[i + di, j + dj, k + dk]
                return v
            return fn
        return f_phase
    path = INPUTS / "MOD" if base == "mod" else OUT / "mods" / f"{base}.MOD"
    m = read_mod(path)
    return lambda phase: (lambda lon, lat, x, y, z: mod_velocity(m, phase, lon, lat, z))


def job(args):
    variant, sta, lat, lon, zsrc, phase, extent = args
    dx = 0.25 if "_c025" in variant else 0.5 if "_c05" in variant else 1.0
    eft = variant.endswith("_eft")
    fn = model_fn(variant)(phase)
    if variant.split("_c")[0] == "orig" and dx == 1.0:      # put the cell centres on the model nodes
        extent = tuple(v - 0.5 for v in extent)
    wd = OUT / "nll" / variant / sta
    zr = (Z_RANGE[0] - 0.5, Z_RANGE[1] - 0.5) if variant.split("_c")[0] == "orig" and dx == 1.0 else Z_RANGE
    root = grid2time(wd, sta, lat, lon, zsrc, phase, fn, extent, dx, zr, eft=eft, centred=True)
    return str(root)


def main():
    sub = OUT / "subset"
    p = pd.read_csv(sub / "picks.csv")
    st = pd.read_csv(sub / "stations.csv")
    gts = {l.split()[1]: float(l.split()[5]) for l in (NLL_MODEL / "stations_GTSRCE.in").read_text().splitlines()}
    args = sys.argv[1:]
    only = None
    if "--stations" in args:
        i = args.index("--stations")
        only = args[i + 1].split(",")
        args = args[:i] + args[i + 2:]
    for variant in args:
        jobs = []
        for r in st.itertuples():
            if only and r.station not in only:
                continue
            d = p[p.station == r.station]
            sx, sy = lambert_km(r.sta_lon, r.sta_lat)
            ext = (np.floor(min(d.x_km.min(), sx) - PAD), np.ceil(max(d.x_km.max(), sx) + PAD),
                   np.floor(min(d.y_km.min(), sy) - PAD), np.ceil(max(d.y_km.max(), sy) + PAD))
            zsrc = gts[r.station] if variant.startswith("orig") else -r.sta_elev_km
            for ph in ("P", "S"):
                jobs.append((variant, r.station, r.sta_lat, r.sta_lon, zsrc, ph, ext))
        with ProcessPoolExecutor(min(NPROC, len(jobs))) as ex:
            roots = list(ex.map(job, jobs))
        col = f"t_{variant}"
        p[col] = np.nan
        for (v, sta, _la, _lo, _z, ph, _e), root in zip(jobs, roots):
            m = (p.station == sta) & (p.phase == ph)
            z = p.loc[m, "depth_bsl_km"].values
            if variant.endswith("_eft"):
                z = eft_depth(z)
            p.loc[m, col] = nll_time(Path(root), p.loc[m, "x_km"].values, p.loc[m, "y_km"].values, z)
        p[["event_ID", "code", "phase", col]].to_csv(OUT / "nll" / f"{variant}.csv", index=False)
        print(variant, "done; NaN", int(p[col].isna().sum()), flush=True)


if __name__ == "__main__":
    main()

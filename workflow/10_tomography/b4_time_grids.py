"""Track B, step B4: NonLinLoc travel-time grids of every station in a registered velocity model (D-B047), built the
D-B028 P-1 / P-5 way, for the relocation of the whole catalog with Track A's NLLoc tools (workflow/08_relocation).

Velocity field: the model version's tomoDD field (trilinear in longitude, latitude, depth) inside its grid, the D-B005
model (continuous, trilinear between its NLL nodes) outside it and in D-B005's water and air cells (Vp < 1.6 km/s),
so bathymetry and topography stay those of Track A.
Grids: the D-B005 Lambert lattice (origin -513, -709 km, z -5 to 80 km); per station a box of +-250 km around it
(as Track A's GridCut), clipped to the lattice. Velocities at the cell centres, Earth-flattening transform of depth
and velocity, Podvin-Lecomte Grid2Time (no angles) at 1 km and 0.5 km cells, combined on the 1 km nodes as
T = T0.5 - k (T1 - T0.5) with k = tomodd.p1.richardson_k (P 1.33, S 1.28): +0.014 / +0.022 s median against the
TauP-checked reference (notes/forward_accuracy.md), against +0.21 / +0.37 s for the v4-NLL3D grids.
The velocity field is computed once per cell size on the whole lattice (memmaps in the work folder) and cropped for
each station.

Depth axis of the time grids = flattened depth z' = Re ln(Re / (Re - z)): NLLoc hypocentres from these grids are in
z' and are mapped back with z = Re (1 - exp(-z' / Re)) (b4_unflatten.py; 0.2 km at 50 km).

Output: /wd1/mdenolle_data/reloc_v5/<time_name>/cascadia3d.<P|S>.<STA>.time.{hdr,buf} (Track A names; use
a2_run_nlloc.py --time <time_name>) and a manifest.

  pixi run -e models python workflow/10_tomography/b4_time_grids.py VERSION TIME_NAME [--nproc 40] [--stations A,B]
"""
import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import load_params, write_manifest  # noqa: E402
from b2_fwd_common import NLL_BIN, NLL_MODEL, RE, eft_depth, mod_velocity, read_nll_hdr  # noqa: E402

P = load_params()
T = P["tomodd"]
X = T["models"]
RELOC = Path("/wd1/mdenolle_data/reloc_v5")
HALF = 250.0                                          # km, box half-width (Track A)
TRANS = ("TRANSFORM  LAMBERT RefEllipsoid WGS-84  LatOrig 45.000000  LongOrig -125.800000  FirstStdParal 41.000000"
         "  SecondStdParal 49.000000  RotCW 0.000000")
G = read_nll_hdr(NLL_MODEL / "cascadia3d.P.mod")      # the D-B005 lattice


def proj():
    from pyproj import Proj
    return Proj(proj="lcc", lat_0=45.0, lon_0=-125.8, lat_1=41.0, lat_2=49.0, ellps="WGS84")


def dB005(phase):
    a = np.memmap(NLL_MODEL / f"cascadia3d.{phase}.mod.buf", dtype="<f4", mode="r", shape=(G["nx"], G["ny"], G["nz"]))

    def fn(x, y, z):                                  # continuous D-B005 ('origlin' of b2_fwd_nll.model_fn)
        fx, fy, fz = (x - G["x0"]) / G["dx"], (y - G["y0"]) / G["dx"], (z - G["z0"]) / G["dx"]
        i = np.clip(np.floor(fx).astype(int), 0, G["nx"] - 2)
        j = np.clip(np.floor(fy).astype(int), 0, G["ny"] - 2)
        k = np.clip(np.floor(fz).astype(int), 0, G["nz"] - 2)
        wx, wy, wz = (np.clip(f - n, 0, 1) for f, n in ((fx, i), (fy, j), (fz, k)))
        v = 0.0
        for di, ax in ((0, 1 - wx), (1, wx)):
            for dj, ay in ((0, 1 - wy), (1, wy)):
                for dk, az in ((0, 1 - wz), (1, wz)):
                    v = v + ax * ay * az * G["dx"] / a[i + di, j + dj, k + dk]
        return v
    return fn


def build_field(ds, dx, work):
    """Cell-centred, Earth-flattened SLOW_LEN fields (P, S) of the whole lattice at cell size dx (memmaps)."""
    nx = int(round((G["nx"] - 1) * G["dx"] / dx)) + 1
    ny = int(round((G["ny"] - 1) * G["dx"] / dx)) + 1
    nz = int(round((G["nz"] - 1) * G["dx"] / dx)) + 1
    xs, ys, zs = (G[k0] + dx * np.arange(n) + dx / 2 for k0, n in (("x0", nx), ("y0", ny), ("z0", nz)))
    X2, Y2 = np.meshgrid(xs, ys, indexing="ij")
    lon, lat = proj()(X2 * 1e3, Y2 * 1e3, inverse=True)
    m = dict(lon=ds.longitude.values, lat=ds.latitude.values, z=ds.depth.values)
    inside = (lon >= m["lon"][0]) & (lon <= m["lon"][-1]) & (lat >= m["lat"][0]) & (lat <= m["lat"][-1])
    bgP, bgS = dB005("P"), dB005("S")
    out = {}
    for ph, var in (("P", "Vp"), ("S", "Vs")):
        f = work / f"field_{dx:g}.{ph}.buf"
        out[ph] = f
        if f.exists():                                # renamed only when complete
            continue
        mm = dict(m, vp=ds[var].values.transpose(2, 1, 0), vs=ds[var].values.transpose(2, 1, 0))
        tmp = work / f"field_{dx:g}.{ph}.partial"
        a = np.memmap(tmp, mode="w+", dtype="<f4", shape=(nx, ny, nz))
        for k, zf in enumerate(zs):                   # zf = flattened depth of the cell centre
            zt = RE * (1 - np.exp(-zf / RE))          # true depth
            vbgP = bgP(X2.ravel(), Y2.ravel(), np.full(X2.size, zt)).reshape(X2.shape)
            v = vbgP if ph == "P" else bgS(X2.ravel(), Y2.ravel(), np.full(X2.size, zt)).reshape(X2.shape)
            use = inside & (vbgP >= 1.6) & (zt >= m["z"][0]) & (zt <= m["z"][-1])
            if use.any():
                v = v.copy()
                v[use] = mod_velocity(mm, "P", lon[use], lat[use], np.full(use.sum(), zt), clamp_surface=False)
            a[:, :, k] = (dx / (v * np.exp(zf / RE))).astype("<f4")
        a.flush()
        del a
        tmp.rename(f)
        print(f"field {ph} {dx:g} km: {nx} x {ny} x {nz}", flush=True)
    return out, (nx, ny, nz)


def g2t(wd, tag, label, lat, lon, zsrc, phase, field, shape, dx, i0, i1, j0, j1):
    """Crop the field, run Grid2Time; returns the time-grid root."""
    nx, ny, nz = shape
    a = np.memmap(field, dtype="<f4", mode="r", shape=(nx, ny, nz))
    crop = np.ascontiguousarray(a[i0:i1 + 1, j0:j1 + 1, :])
    root = wd / f"mod_{tag}.{phase}.mod"
    crop.tofile(f"{root}.buf")
    x0, y0 = G["x0"] + dx * i0, G["y0"] + dx * j0
    Path(f"{root}.hdr").write_text(f"{crop.shape[0]} {crop.shape[1]} {crop.shape[2]}  {x0:f} {y0:f} {G['z0']:f}  "
                                   f"{dx:f} {dx:f} {dx:f} SLOW_LEN FLOAT\n{TRANS}\n")
    ctl = wd / f"g2t_{tag}_{phase}.in"
    ctl.write_text(f"CONTROL 1 54321\nTRANS LAMBERT WGS-84 45.0 -125.8 41.0 49.0 0.0\n"
                   f"GTFILES mod_{tag} t_{tag} {phase}\nGTMODE GRID3D ANGLES_NO\n"
                   f"GTSRCE {label} LATLON {lat:.6f} {lon:.6f} {float(eft_depth(zsrc)):.4f} 0.0\nGT_PLFD 1.0e-3 0\n")
    with open(wd / f"g2t_{tag}_{phase}.log", "w") as f:
        subprocess.run([str(NLL_BIN / "Grid2Time"), ctl.name], cwd=wd, stdout=f, stderr=subprocess.STDOUT, check=True)
    for c in wd.glob(f"t_{tag}.{phase}.mod.*"):
        c.unlink()
    return wd / f"t_{tag}.{phase}.{label}.time"


def job(a):
    label, lat, lon, zsrc, phase, fields, shapes, work, out = a
    dst = out / f"cascadia3d.{phase}.{label}.time"
    if Path(f"{dst}.hdr").exists():
        return label, phase, "exists"
    sx, sy = (v / 1e3 for v in proj()(lon, lat))
    i0 = max(0, int(np.floor(sx - HALF - G["x0"]))); i1 = min(G["nx"] - 1, int(np.ceil(sx + HALF - G["x0"])))
    j0 = max(0, int(np.floor(sy - HALF - G["y0"]))); j1 = min(G["ny"] - 1, int(np.ceil(sy + HALF - G["y0"])))
    wd = work / f"{label}.{phase}"
    wd.mkdir(parents=True, exist_ok=True)
    r1 = g2t(wd, "1", label, lat, lon, zsrc, phase, fields[1.0][phase], shapes[1.0], 1.0, i0, i1, j0, j1)
    r05 = g2t(wd, "05", label, lat, lon, zsrc, phase, fields[0.5][phase], shapes[0.5], 0.5, 2 * i0, 2 * i1, 2 * j0, 2 * j1)
    g1, g05 = read_nll_hdr(r1), read_nll_hdr(r05)
    t1 = np.fromfile(f"{r1}.buf", "<f4").reshape(g1["nx"], g1["ny"], g1["nz"])
    t05 = np.fromfile(f"{r05}.buf", "<f4").reshape(g05["nx"], g05["ny"], g05["nz"])[::2, ::2, ::2]
    assert t05.shape == t1.shape, (t05.shape, t1.shape)
    k = T["p1"]["richardson_k"][phase]
    (t05 - k * (t1 - t05)).astype("<f4").tofile(f"{dst}.buf")
    shutil.copy(f"{r1}.hdr", f"{dst}.hdr")
    shutil.rmtree(wd)
    return label, phase, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("version")
    ap.add_argument("time_name")
    ap.add_argument("--nproc", type=int, default=40)
    ap.add_argument("--stations", help="comma-separated labels (test)")
    a = ap.parse_args()
    ds = xr.open_dataset(Path(X["store"]) / X["name"] / f"v{a.version}.nc")
    out = RELOC / a.time_name
    work = RELOC / f"work_{a.time_name}"
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    fields, shapes = {}, {}
    for dx in (1.0, 0.5):
        fields[dx], shapes[dx] = build_field(ds, dx, work)
    st = [l.split() for l in (NLL_MODEL / "stations_GTSRCE.in").read_text().splitlines() if l.startswith("GTSRCE")]
    st = [(s[1], float(s[3]), float(s[4]), float(s[5])) for s in st]
    if a.stations:
        st = [s for s in st if s[0] in a.stations.split(",")]
    jobs = [(lab, la, lo, z, ph, fields, shapes, work, out) for lab, la, lo, z in st for ph in ("P", "S")]
    print(f"{len(jobs)} station-phases", flush=True)
    res = []
    with ProcessPoolExecutor(a.nproc) as ex:
        for i, r in enumerate(ex.map(job, jobs, chunksize=1)):
            res.append(r)
            if i % 50 == 0:
                print(f"{i + 1}/{len(jobs)} {r}", flush=True)
    write_manifest(f"b4_time_grids_{a.time_name}", out,
                   inputs=[Path(X["store"]) / X["name"] / f"v{a.version}.nc", NLL_MODEL / "cascadia3d.P.mod.buf",
                           NLL_MODEL / "stations_GTSRCE.in"],
                   params={"version": a.version, "half_km": HALF, "richardson_k": T["p1"]["richardson_k"],
                           "eft": True, "centred": True, "cells_km": [1.0, 0.5]},
                   counts={"station_phases": len(jobs), "ok": sum(r[2] == "ok" for r in res)})
    print(json.dumps({"done": len(res), "ok": sum(r[2] == "ok" for r in res)}))


if __name__ == "__main__":
    main()

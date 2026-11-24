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
Within 2 km of the station T0.5 alone, tapered linearly to the combination at 5 km (R_T05, R_COMB): the
extrapolation fails next to the source (negative times; PR #35 finding 14). This removes the negative and too-early
times within 5 km only. T1's start-up error at the source also travels along the rays, and beyond 5 km the taper
does not correct it: there T stays early by the excess of k (T1 - T0.5) over the network's (NED.CC about 0.04 s P /
0.06 s S earlier than SEP.CC and VALT.CC, 0.4 and 1.6 km away). The manifest records per grid the median of T0.5 - T
over rock nodes 5-15 km from the station (notes.t05_minus_t_s). In v0.3.0 it is 0.014 s P / 0.024 s S for the median
station, 0.036 / 0.064 s at the 90th percentile, 0.056 / 0.093 s for NED.CC and up to 0.078 / 0.166 s at seafloor
stations (J48A.7D). D-B052 (PI 2026-11-24): no whole-grid fallback. A T0.5-only grid for the outliers (median +
3 x 1.4826 MAD: P 0.044 s, S 0.075 s) fixed some stations against co-located neighbours (NED.CC, J17D.7D, BS58.XG)
and made others worse (FN08C.7D, G30B.7D, J25C.7D), and T0.5 is itself late with distance; the remaining offset is
close to a constant shift per station, which the B4 station terms absorb. The diagnostic stays in the manifest; the
fallback code is kept but off (params b4.grid_fallback.enabled false).
The velocity field is computed once per cell size on the whole lattice (memmaps in the work folder) and cropped for
each station. A rerun into the same TIME_NAME keeps the grids already there, so it is refused unless the parameters
are those of the earlier run (its manifest, or work_<TIME_NAME>/params.json of an interrupted run): time_b4_v0.3.0
was built without the taper, and a new build needs a new TIME_NAME.

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
# job() takes crop indices as km from x0/y0 and doubles them for the 0.5 km field: only right on a 1 km lattice
assert G["dx"] == 1.0, "b4_time_grids crop math assumes a 1 km D-B005 lattice"
# Near the station T1's error comes from Grid2Time's start-up of the source cell, not from h^p, and the Richardson
# step amplifies it (NED.CC P: -0.10 s at 0.4 km, 0.17 s for 0.38 s at 1.3 km; PR #35 finding 14). So T0.5 alone
# within R_T05 of the source, the combination beyond R_COMB, linear in distance between. Only the nodes within R_COMB
# are fixed: the start-up error carried along the rays beyond it is not (NED.CC about -0.04 s P / -0.06 s S against
# SEP.CC and VALT.CC); job() measures it over R_DIAG for the manifest (D-B052: no whole-grid fallback, see docstring).
R_T05, R_COMB = 2.0, 5.0                               # km, 3D distance from the GTSRCE position in the grid frame
R_DIAG = (5.0, 15.0)                                   # km, shell of that diagnostic, just beyond the taper
_GF = dict((P.get("b4") or {}).get("grid_fallback") or {})                 # D-B052: off unless enabled
FALLBACK = ({"P": float(_GF.get("P", 0.044)), "S": float(_GF.get("S", 0.075))} if _GF.get("enabled")
            else {"P": float("inf"), "S": float("inf")})                   # s; inf = never


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


def near(src, g, rad):
    """Block of nodes of grid g within rad km of src per axis (clipped to the grid), and their 3D distance to src."""
    blk, ax = [], []
    for s, o, n in zip(src, ("x0", "y0", "z0"), ("nx", "ny", "nz")):
        f = (s - g[o]) / g["dx"]
        b0, b1 = max(0, int(np.floor(f - rad / g["dx"]))), min(g[n], int(np.ceil(f + rad / g["dx"])) + 1)
        blk.append(slice(b0, b1))
        ax.append(g[o] + g["dx"] * np.arange(b0, b1) - s)
    return tuple(blk), np.sqrt(ax[0][:, None, None] ** 2 + ax[1][None, :, None] ** 2 + ax[2][None, None, :] ** 2)


def job(a):
    label, lat, lon, zsrc, phase, fields, shapes, work, out = a
    dst = out / f"cascadia3d.{phase}.{label}.time"
    diag = work / f"{label}.{phase}.t05off"           # the diagnostic and fallback below, for a resumed run's manifest
    if Path(f"{dst}.hdr").exists():
        d = diag.read_text().split() if diag.exists() else []
        return label, phase, "exists", float(d[0]) if d else None, bool(int(d[1])) if len(d) > 1 else None
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
    tt = (t05 - k * (t1 - t05)).astype("<f4")
    # taper to T0.5 near the source, only on the block of nodes within R_COMB (w = 1 leaves tt unchanged elsewhere)
    src = [float(v) for v in Path(f"{r1}.hdr").read_text().splitlines()[1].split()[1:4]]   # x, y, z' of the source
    blk, r = near(src, g1, R_COMB)
    w = np.clip((r - R_T05) / (R_COMB - R_T05), 0.0, 1.0)
    tt[blk] = t05[blk] + w * (tt[blk] - t05[blk])
    # what the taper leaves: beyond R_COMB, T0.5 - T = k (T1 - T0.5); well above the network's = start-up error of T1
    # carried along the rays (not corrected; see the docstring). Rock nodes only (1 km P cell from the node with
    # Vp >= 1.6 km/s, build_field's rule; flattening adds at most 0.3% this shallow): water and air nodes have a small
    # T0.5 - T and hide the offset (NED.CC P 0.028 s over all nodes, 0.056 s over rock)
    blk, r = near(src, g1, R_DIAG[1])
    vp = 1.0 / np.memmap(fields[1.0]["P"], dtype="<f4", mode="r", shape=shapes[1.0])[
        tuple(slice(o + b.start, o + b.stop) for o, b in zip((i0, j0, 0), blk))]
    sel = (r >= R_DIAG[0]) & (r <= R_DIAG[1]) & (vp >= 1.6)
    off = round(float(np.median((t05[blk] - tt[blk])[sel])), 4)
    # D-B052: start-up offset above the network's -> T0.5 for the whole grid (1 km nodes of the 0.5 km run, same hdr)
    fb = bool(off > FALLBACK[phase])
    if fb:
        tt = t05.astype("<f4")                         # contiguous copy of the strided view
    # NLLoc rejects or drops negative predicted times without a message: stop here instead
    assert np.isfinite(tt).all() and tt.min() >= 0, (label, phase, float(np.nanmin(tt)))
    tt.tofile(f"{dst}.buf")
    diag.write_text(f"{off} {int(fb)}\n")
    shutil.copy(f"{r1}.hdr", f"{dst}.hdr")           # last: marks the grid complete
    shutil.rmtree(wd)
    return label, phase, "ok", off, fb


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
    prm = {"version": a.version, "half_km": HALF, "richardson_k": T["p1"]["richardson_k"], "eft": True,
           "centred": True, "cells_km": [1.0, 0.5], "t05_taper_km": [R_T05, R_COMB], "grid_fallback_s": FALLBACK}
    # a rerun keeps the grids and fields already there: refuse them when built with other or unknown parameters
    # (time_b4_v0.3.0 has no taper), or the manifest would give them the new ones
    stamp = work / "params.json"                      # written now, so an interrupted run can be resumed
    old = [json.loads(p.read_text())["params"] for p in out.glob("b4_time_grids_*.manifest.json")]
    old += [json.loads(stamp.read_text())] if stamp.exists() else []
    if any(o != json.loads(json.dumps(prm)) for o in old) or (not old and any(out.glob("*.time.hdr"))):
        sys.exit(f"{out} holds grids built with other or unknown parameters {old}: use a new TIME_NAME")
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    stamp.write_text(json.dumps(prm))
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
            if i % 50 == 0 or r[4]:
                print(f"{i + 1}/{len(jobs)} {r}" + (" D-B052 fallback: T0.5 whole grid" if r[4] else ""), flush=True)
    # per grid: median of T0.5 - T over rock nodes R_DIAG km from the station, the offset the taper leaves (docstring)
    off = {f"{ph}.{lab}": o for lab, ph, _, o, _ in res}
    fbk = {f"{ph}.{lab}": f for lab, ph, _, _, f in res}
    for ph in ("P", "S"):
        v = sorted(k for k, f in fbk.items() if k[0] == ph and f)
        print(f"D-B052 fallback {ph} (> {FALLBACK[ph]} s): {len(v)}/{sum(k[0] == ph for k in fbk)} {v}", flush=True)
    summ = {}
    for ph in ("P", "S"):
        v = {k: o for k, o in off.items() if k[0] == ph and o is not None}
        if v:
            x = np.array(list(v.values()))
            summ[ph] = {"median": round(float(np.median(x)), 4), "p90": round(float(np.percentile(x, 90)), 4),
                        "top10": dict(sorted(v.items(), key=lambda kv: -kv[1])[:10])}
            print(f"T0.5 - T, {ph}, {R_DIAG[0]:g}-{R_DIAG[1]:g} km: {json.dumps(summ[ph])}", flush=True)
    write_manifest(f"b4_time_grids_{a.time_name}", out,
                   inputs=[Path(X["store"]) / X["name"] / f"v{a.version}.nc", NLL_MODEL / "cascadia3d.P.mod.buf",
                           NLL_MODEL / "stations_GTSRCE.in"],
                   params=prm,
                   counts={"station_phases": len(jobs), "ok": sum(r[2] == "ok" for r in res),
                           "fallback": sum(bool(f) for f in fbk.values())},
                   notes={"t05_minus_t_s": off, "t05_minus_t_summary": summ, "t05_minus_t_shell_km": list(R_DIAG),
                          "t05_minus_t_about": f"median of T0.5 - T over rock nodes {R_DIAG[0]:g}-{R_DIAG[1]:g} km "
                          "from the station, T = the tapered Richardson combination; well above the network median "
                          "(v0.3.0: 0.014 s P / 0.024 s S) = start-up error of T1 carried beyond the taper; None = "
                          "grid kept from a run without the diagnostic",
                          "grid_fallback": fbk, "grid_fallback_threshold_s": FALLBACK,
                          "grid_fallback_about": "D-B052: true = t05_minus_t_s above the phase threshold, so the grid "
                          "written is T0.5 (0.5 km run on the 1 km nodes) everywhere instead of the combination; "
                          "None = grid kept from a run without the record"})
    print(json.dumps({"done": len(res), "ok": sum(r[2] == "ok" for r in res)}))


if __name__ == "__main__":
    main()

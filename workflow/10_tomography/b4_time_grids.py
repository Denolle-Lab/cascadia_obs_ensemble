"""Track B, step B4: NonLinLoc travel-time grids of every station in a registered velocity model (D-B047), built the
D-B028 P-1 / P-5 way, for the relocation of the whole catalog with Track A's NLLoc tools (workflow/08_relocation).

Velocity field, water/rock boundary by params b4.surface_mask:
* 'd_b005' (default when the key is absent; every build up to time_b4_v0.3.1): the model version's tomoDD field
  (trilinear in longitude, latitude, depth) inside its grid, the D-B005 model (continuous, trilinear between its NLL
  nodes) outside it and in D-B005's water and air cells (Vp < 1.6 km/s), so bathymetry and topography stay those of
  Track A (earth_relief 01m, blurred over one 1 km node by the trilinear interpolation).
* 'merged' (D-B068, v5.1): the boundary is b4_surface.surface_m (GMRT patches, earth_relief 01m elsewhere; interface
  I1), sharp at the cell scale. A cell is fluid when its centre (true depth) is above surface_m at its centre: water
  if the centre is below sea level, air above, with D-B005's own water and air velocities (read from its nodes:
  fluid_velocities(); P 1.5 / 0.3, S 0.4 / 0.3 km/s). Every other cell is rock: the model version inside its grid
  (lon, lat and depth range, as above), outside it D-B005 continued upward (dB005(phase, rock=True): every D-B005 water
  or air node takes the shallowest rock node below it in its column, so where D-B005 is fluid but the merged surface
  is rock, and in rock cells next to D-B005's surface, no water or air velocity enters a rock cell, nor a blend).
  Station source cells (STATION_PATCH_KM): as D-B005's station patch (a1_build_model.py: the 4 x 4 nodes around the
  station's 1 km cell made solid from the node at/above the station down), the cells whose centre lies within
  [x0 + i0 - 1, x0 + i0 + 2] km (i0 = the station's 1 km lattice cell; same in y) are rock from the cell that holds the
  station (GTSRCE, z') down, whatever the surface: without it the source cell of 336 of 696 stations at 1 km and 278
  at 0.5 km (centre up to 0.5 km above the surface) would be water or air (v51_build/field/scripts/s01). These cells
  are the only rock cell centres above surface_m (counted in field_<dx>.<ph>.mask.json and the manifest).
  Time-grid nodes above the merged surface (FLUID_NODES = rock_continued_1km): NLLoc interpolates times trilinearly
  between the 1 km nodes, so an event within one node below the surface takes part of its time from the node above
  it. With the sharp boundary that node's time is a water or air path (S 0.4 / 0.3 km/s): at Axial events within
  0.5 km below the seafloor the merged grids without this step were late by a median 0.06 s (P) / 0.23 s (S), p95
  0.31 / 1.32 s (v51_build/field/times). So each node above the surface (true depth against b4_surface at the node;
  work/surface_nodes.npz) takes the time of the shallowest node at or below the surface in its column plus the
  difference of the 1 km times between the two nodes in the rock-everywhere field (work/field_1.<ph>.rock.buf: the
  rock rule at every cell, a third Grid2Time run at 1 km); a negative result (near the source) is set to 0 and
  counted. Nodes at or below the surface keep the times of the physical field. Per grid the .t05off file records
  the number of nodes replaced, the negatives, and the median / p99 / max change at the node just above the surface.
  Test (v51_build/field/nlloc, 26,721 events recorded only by 20 test stations, nearly all Axial; v0.3.1 iter1
  nlloc.in, merged floor): without this step 2,042 events moved from a median 2.4 km to 0.2 km below the seafloor
  (late S picks fitted by the artefact; Axial share within 2 km of the seafloor 37.8 -> 43.6 %); with it 24,283 are
  identical to the run in the v0.3.1 grids and the share is 37.7 %. The v0.3.1 grids carry the same kind of late
  times at the node just above the surface where its blurred boundary left that node in fluid (SEP.CC P: 66 % of
  columns, median +0.65 s; AXAS1.OO S: 12 %, p95 +0.57 s; v51_build/field/scripts/s04_first_node.py).
  params gets surface_mask, the surface sources (md5), the fluid velocities, the station patch and fluid_node_times,
  so the guard below never mixes a merged build with an older one.
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
a2_run_nlloc.py --time <time_name>) and a manifest; the fields in work_<time_name>/ (the wide grids read field_1).

  pixi run -e models python workflow/10_tomography/b4_time_grids.py VERSION TIME_NAME [--nproc 40] [--stations A,B]
                                                                    [--root DIR]
--root (tests): writes <DIR>/<time_name> and <DIR>/work_<time_name> instead of reloc_v5, and the manifest only next to
the grids (no append to the shared manifests.jsonl). The merged mask reads b4_surface's pinned sources (params
b4.surface.dir, or B4_SURFACE_DIR for tests).
"""
import argparse
import hashlib
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
FLUID_VP = 1.6                       # km/s: D-B005 nodes below it are water (1.5) or air (0.3); its rock is >= 2.07
MASKS = ("d_b005", "merged")
MASK = str((P.get("b4") or {}).get("surface_mask") or "d_b005")         # absent = the rule of every build to v0.3.1
assert MASK in MASKS, f"params b4.surface_mask {MASK!r} not in {MASKS}"
# Shallow oceanic crust rule (D-B070 test; adopted by D-B073 if params b4.shallow_crust is set): in offshore rock cells
# whose centre lies within thickness_km below the seafloor, Vs = min(Vs, Vp / vpvs). Offshore = the coast-scan rule of
# the D-B070 test (check_slow_s/scripts/s10_masks.py): below sea level, west of the first run of land columns at least
# coast_run_km long along each lattice row, and not in the Strait of Juan de Fuca or the Columbia River mouth (cuts).
# Needs the merged surface mask (seafloor and coast from b4_surface.surface_m). Absent = no rule (every build to v0.3.1).
SC = (P.get("b4") or {}).get("shallow_crust") or None
if SC:
    assert MASK == "merged", "params b4.shallow_crust needs b4.surface_mask merged"
    SC = {"vpvs": float(SC["vpvs"]), "thickness_km": float(SC["thickness_km"]),
          "coast_run_km": float(SC.get("coast_run_km", 5.0)),
          "cuts": [list(map(float, c)) for c in SC.get("cuts", [[47.9, 48.75, -124.55], [46.1, 46.4, -124.05]])]}


def offshore_mask(surf_km, lon, lat, dx):
    """Offshore columns of a lattice (x along axis 0, y along axis 1) by the coast-scan rule (SC; D-B070 test)."""
    land = surf_km >= 0
    nx, ny = land.shape
    n = int(round(SC["coast_run_km"] / dx))
    off = np.zeros_like(land)
    for j in range(ny):
        run = np.convolve(land[:, j].astype(int), np.ones(n, int), "valid") == n
        i = int(np.argmax(run)) if run.any() else nx
        off[:i, j] = ~land[:i, j]
    for la0, la1, lomax in SC["cuts"]:
        off &= ~((lat >= la0) & (lat <= la1) & (lon >= lomax))
    return off
STATION_PATCH_KM = (-1.0, 2.0)       # merged mask: station rock columns, km from the station's 1 km lattice cell origin
FLUID_NODES = "rock_continued_1km"   # merged mask: time-grid nodes above the surface (module docstring)


def proj():
    from pyproj import Proj
    return Proj(proj="lcc", lat_0=45.0, lon_0=-125.8, lat_1=41.0, lat_2=49.0, ellps="WGS84")


def _db005_nodes(phase):
    return np.memmap(NLL_MODEL / f"cascadia3d.{phase}.mod.buf", dtype="<f4", mode="r",
                     shape=(G["nx"], G["ny"], G["nz"]))


def db005_rock_nodes(phase):
    """D-B005 SLOW_LEN nodes with every water or air node (Vp < FLUID_VP, P's rule for both phases) replaced by the
    shallowest rock node below it in its column: D-B005 continued upward (merged mask). In RAM (479 MB)."""
    rock = (G["dx"] / np.asarray(_db005_nodes("P"))) >= FLUID_VP
    assert rock[:, :, -1].all(), "D-B005 has water or air in its deepest layer"
    a = np.array(_db005_nodes(phase))
    for k in range(G["nz"] - 2, -1, -1):               # from the bottom up: a fluid node takes the node below it
        f = ~rock[:, :, k]
        a[:, :, k][f] = a[:, :, k + 1][f]
    return a


def fluid_velocities():
    """D-B005's water and air velocities (km/s) read from its nodes: fluid (Vp < FLUID_VP) at z >= 0 is water, at
    z < 0 air (a1_build_model.py's rule); each must be one value per phase (to 1e-6 km/s, float32 storage)."""
    zn = G["z0"] + G["dx"] * np.arange(G["nz"])
    fluid = (G["dx"] / np.asarray(_db005_nodes("P"))) < FLUID_VP
    out = {}
    for ph in ("P", "S"):
        a = np.asarray(_db005_nodes(ph))
        for kind, sel in (("water", zn >= 0), ("air", zn < 0)):
            v = np.unique(np.round(G["dx"] / a[:, :, sel][fluid[:, :, sel]].astype("f8"), 6))
            assert v.size == 1, f"D-B005 {ph} {kind} velocities not unique: {v[:10]}"
            out.setdefault(ph, {})[kind] = float(v[0])
    return out


def gtsrce():
    """Stations of the build: (label, lat, lon, zSrce km) of stations_GTSRCE.in."""
    st = [l.split() for l in (NLL_MODEL / "stations_GTSRCE.in").read_text().splitlines() if l.startswith("GTSRCE")]
    return [(s[1], float(s[3]), float(s[4]), float(s[5])) for s in st]


def station_rock_top(dx, shape):
    """Merged mask: per column of the dx field, the index of the first cell kept rock for a station source (the cell
    holding the station's GTSRCE z'; shape[2] = none). Columns: cell centres within STATION_PATCH_KM of the station's
    1 km lattice cell origin (D-B005's 4 x 4 node patch). All stations of stations_GTSRCE.in, whatever --stations."""
    nx, ny, nz = shape
    top = np.full((nx, ny), nz, np.int32)
    xc = G["x0"] + dx * (np.arange(nx) + 0.5)
    yc = G["y0"] + dx * (np.arange(ny) + 0.5)
    for _, lat, lon, z in gtsrce():
        sx, sy = (float(v) / 1e3 for v in proj()(lon, lat))
        ox = G["x0"] + G["dx"] * np.floor((sx - G["x0"]) / G["dx"])
        oy = G["y0"] + G["dx"] * np.floor((sy - G["y0"]) / G["dx"])
        ii = np.flatnonzero((xc >= ox + STATION_PATCH_KM[0]) & (xc <= ox + STATION_PATCH_KM[1]))
        jj = np.flatnonzero((yc >= oy + STATION_PATCH_KM[0]) & (yc <= oy + STATION_PATCH_KM[1]))
        k = int(np.floor((float(eft_depth(z)) - G["z0"]) / dx))
        blk = np.ix_(ii, jj)
        top[blk] = np.minimum(top[blk], k)
    return top


def dB005(phase, rock=False):
    """Continuous D-B005 velocity fn(x, y, z) (km/s; Lambert km, true depth). rock=True: D-B005 continued upward
    (db005_rock_nodes), never a water or air velocity."""
    a = db005_rock_nodes(phase) if rock else _db005_nodes(phase)

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


def field_shape(dx):
    return tuple(int(round((G[n] - 1) * G["dx"] / dx)) + 1 for n in ("nx", "ny", "nz"))


def build_field(ds, dx, work, mask=None):
    """Cell-centred, Earth-flattened SLOW_LEN fields (P, S) of the whole lattice at cell size dx (memmaps), water/rock
    boundary by `mask` (default params b4.surface_mask; module docstring)."""
    mask = mask or MASK
    assert mask in MASKS, mask
    nx, ny, nz = field_shape(dx)
    xs, ys, zs = (G[k0] + dx * np.arange(n) + dx / 2 for k0, n in (("x0", nx), ("y0", ny), ("z0", nz)))
    X2, Y2 = np.meshgrid(xs, ys, indexing="ij")
    lon, lat = proj()(X2 * 1e3, Y2 * 1e3, inverse=True)
    m = dict(lon=ds.longitude.values, lat=ds.latitude.values, z=ds.depth.values)
    inside = (lon >= m["lon"][0]) & (lon <= m["lon"][-1]) & (lat >= m["lat"][0]) & (lat <= m["lat"][-1])
    out = {ph: work / f"field_{dx:g}.{ph}.buf" for ph in ("P", "S")}
    if mask == "merged" and dx == 1.0:                  # + the rock-everywhere 1 km field (FLUID_NODES)
        out.update({f"rock_{ph}": work / f"field_1.{ph}.rock.buf" for ph in ("P", "S")})
    if all(f.exists() for f in out.values()):           # renamed only when complete (each phase: rock first)
        return out, (nx, ny, nz)
    if mask == "merged":
        from b4_surface import surface_m
        surf = surface_m(lon, lat) / 1e3               # km, positive up, at the column centres
        assert np.isfinite(surf).all(), f"surface_m NaN at {int((~np.isfinite(surf)).sum())} cell centres"
        fv = fluid_velocities()
        ktop = station_rock_top(dx, (nx, ny, nz))
        off = offshore_mask(surf, lon, lat, dx) if SC else None
        sc_cells = {"field": 0, "rock_field": 0, "offshore_columns": int(off.sum()) if SC else 0}
    else:
        bgP = dB005("P")
    for ph, var in (("P", "Vp"), ("S", "Vs")):
        f = out[ph]
        if f.exists() and (f"rock_{ph}" not in out or out[f"rock_{ph}"].exists()):
            continue
        mm = dict(m, vp=ds[var].values.transpose(2, 1, 0), vs=ds[var].values.transpose(2, 1, 0))
        tmp = work / f"field_{dx:g}.{ph}.partial"
        a = np.memmap(tmp, mode="w+", dtype="<f4", shape=(nx, ny, nz))
        if mask == "merged":
            bg = dB005(ph, rock=True)
            cnt = {"rock": 0, "water": 0, "air": 0, "station_rock_above_surface": 0, "model": 0, "d_b005_rock": 0}
            rk = out.get(f"rock_{ph}")
            ar = np.memmap(work / f"{rk.name}.partial", mode="w+", dtype="<f4", shape=(nx, ny, nz)) if rk else None
        else:
            bgS = dB005("S") if ph == "S" else None
        for k, zf in enumerate(zs):                   # zf = flattened depth of the cell centre
            zt = RE * (1 - np.exp(-zf / RE))          # true depth
            if mask == "merged":
                # fluid: centre above the merged surface, except station source columns from the station's cell down
                above = -zt > surf
                rock = ~above | (k >= ktop)
                v = bg(X2.ravel(), Y2.ravel(), np.full(X2.size, zt)).reshape(X2.shape)
                use = inside & (zt >= m["z"][0]) & (zt <= m["z"][-1])  # rock rule at every cell (pointwise lookups)
                if use.any():
                    v[use] = mod_velocity(mm, "P", lon[use], lat[use], np.full(use.sum(), zt), clamp_surface=False)
                if SC and ph == "S" and zt > 0:       # shallow oceanic crust (SC): Vs = min(Vs, Vp / vpvs)
                    top = off & (zt + surf <= SC["thickness_km"])         # zt - seafloor depth (= -surf) <= thickness
                    if top.any():
                        pf = out.get("rock_P", out["P"])                   # P of the same cells (rock everywhere if built)
                        vp = dx / (np.asarray(np.memmap(pf, dtype="<f4", mode="r", shape=(nx, ny, nz))[:, :, k],
                                              dtype="f8") * np.exp(zf / RE))
                        top &= vp >= 1.6
                        new = np.minimum(v, vp / SC["vpvs"])
                        sc_cells["rock_field"] += int((top & (new < v)).sum())
                        sc_cells["field"] += int((top & rock & (new < v)).sum())
                        v = np.where(top, new, v)
                if ar is not None:                    # rock everywhere: the fluid-node continuation of job()
                    ar[:, :, k] = (dx / (v * np.exp(zf / RE))).astype("<f4")
                use = use & rock
                kind = "water" if zt >= 0 else "air"
                v[~rock] = fv[ph][kind]
                cnt["rock"] += int(rock.sum()); cnt[kind] += int((~rock).sum())
                cnt["station_rock_above_surface"] += int((rock & above).sum())
                cnt["model"] += int(use.sum()); cnt["d_b005_rock"] += int((rock & ~use).sum())
            else:
                vbgP = bgP(X2.ravel(), Y2.ravel(), np.full(X2.size, zt)).reshape(X2.shape)
                v = vbgP if ph == "P" else bgS(X2.ravel(), Y2.ravel(), np.full(X2.size, zt)).reshape(X2.shape)
                use = inside & (vbgP >= 1.6) & (zt >= m["z"][0]) & (zt <= m["z"][-1])
                if use.any():
                    v = v.copy()
                    v[use] = mod_velocity(mm, "P", lon[use], lat[use], np.full(use.sum(), zt), clamp_surface=False)
            a[:, :, k] = (dx / (v * np.exp(zf / RE))).astype("<f4")
        a.flush()
        del a
        if mask == "merged":
            if ar is not None:
                ar.flush()
                del ar
                (work / f"{rk.name}.partial").rename(rk)
            (work / f"field_{dx:g}.{ph}.mask.json").write_text(json.dumps(
                {"surface_mask": mask, "dx_km": dx, "shape": [nx, ny, nz], "cells": cnt,
                 "shallow_crust": SC, "shallow_crust_cells": sc_cells if ph == "S" else None,
                 "fluid_velocity_km_s": fv[ph]}, indent=1))
            del bg
        if f.exists():                                 # a direct call must not replace a field of another build
            raise SystemExit(f"{f} exists: refusing to replace it (new TIME_NAME, or remove the work folder)")
        tmp.rename(f)
        print(f"field {ph} {dx:g} km: {nx} x {ny} x {nz}" + (f" {cnt}" if mask == "merged" else ""), flush=True)
    return out, (nx, ny, nz)


def surface_nodes(work):
    """Merged mask: per node column of the 1 km lattice (x0 + i, y0 + j km) the merged surface (km, positive up) and
    the index of the shallowest node at or below it (true depth), saved once as work/surface_nodes.npz (FLUID_NODES)."""
    f = work / "surface_nodes.npz"
    if f.exists():
        return f
    from b4_surface import surface_m
    xs, ys = (G[o] + G["dx"] * np.arange(G[n]) for o, n in (("x0", "nx"), ("y0", "ny")))
    X2, Y2 = np.meshgrid(xs, ys, indexing="ij")
    lon, lat = proj()(X2 * 1e3, Y2 * 1e3, inverse=True)
    sk = surface_m(lon, lat) / 1e3
    assert np.isfinite(sk).all(), "surface_m NaN at lattice nodes"
    zt = RE * (1 - np.exp(-(G["z0"] + G["dx"] * np.arange(G["nz"])) / RE))
    below = zt[None, None, :] >= -sk[:, :, None]
    assert below[:, :, -1].all()
    kr = np.argmax(below, axis=2)
    tmp = work / "surface_nodes.partial.npz"
    np.savez(tmp, surface_km=sk.astype("f4"), k_rock=kr.astype("i2"))
    tmp.rename(f)
    return f


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
    # Vp >= 1.6 km/s, build_field's d_b005 rule; with the merged mask fluid cells are exactly D-B005's 1.5 / 0.3 and
    # rock >= 2.07, so the same test; flattening adds at most 0.3% this shallow): water and air nodes have a small
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
    flx = ""
    if f"rock_{phase}" in fields[1.0]:                 # merged mask: FLUID_NODES (module docstring)
        r1r = g2t(wd, "1r", label, lat, lon, zsrc, phase, fields[1.0][f"rock_{phase}"], shapes[1.0], 1.0, i0, i1, j0, j1)
        assert read_nll_hdr(r1r) == g1, (label, phase)
        t1r = np.fromfile(f"{r1r}.buf", "<f4").reshape(g1["nx"], g1["ny"], g1["nz"]).astype("f8")
        ks = np.load(work / "surface_nodes.npz")["k_rock"][i0:i1 + 1, j0:j1 + 1].astype(int)[:, :, None]
        kk = np.arange(g1["nz"])[None, None, :]
        fl = kk < ks                                   # nodes above the merged surface
        cont = np.take_along_axis(tt, ks, axis=2).astype("f8") + t1r - np.take_along_axis(t1r, ks, axis=2)
        first = fl & (kk == ks - 1)                    # the node just above the surface
        chg = np.abs(cont - tt)[first]
        neg = int((fl & (cont < 0)).sum())
        tt = np.where(fl, np.maximum(cont, 0.0), tt).astype("<f4")
        flx = f" {int(fl.sum())} {neg} {np.median(chg):.4f} {np.percentile(chg, 99):.4f} {chg.max():.4f}"
    # NLLoc rejects or drops negative predicted times without a message: stop here instead
    assert np.isfinite(tt).all() and tt.min() >= 0, (label, phase, float(np.nanmin(tt)))
    tt.tofile(f"{dst}.buf")
    diag.write_text(f"{off} {int(fb)}{flx}\n")
    shutil.copy(f"{r1}.hdr", f"{dst}.hdr")           # last: marks the grid complete
    shutil.rmtree(wd)
    return label, phase, "ok", off, fb


MASK_RULE = {
    "d_b005": "D-B005 rule (builds to time_b4_v0.3.1): model version inside its grid where the trilinear D-B005 P "
              "velocity at the cell centre is >= 1.6 km/s, D-B005 trilinear elsewhere (water/air by earth_relief 01m, "
              "blurred over one 1 km node)",
    "merged": "D-B068: cell fluid iff its centre (true depth) is above b4_surface.surface_m at its centre (water below "
              "sea level, air above, D-B005's own fluid velocities), except station source columns (station_patch_km "
              "around the station's 1 km lattice cell, from the cell holding the GTSRCE z' down: rock); rock = model "
              "version inside its grid (lon, lat, depth), D-B005 continued upward from the shallowest rock node "
              "elsewhere (no water or air velocity, no blend, in a rock cell); time-grid nodes above the surface: "
              "time at the shallowest node at or below it + the rock-everywhere 1 km time difference "
              "(fluid_node_times rock_continued_1km)"}


def local_manifest(name, outdir, inputs=(), outputs=(), params=None, counts=None, notes=None):
    """write_manifest's record next to the outputs only (tests: no append to the shared manifests.jsonl)."""
    import manifest as M
    rec = {"name": name, "script": str(Path(sys.argv[0]).resolve()), "argv": sys.argv[1:],
           "utc": M.datetime.now(M.timezone.utc).isoformat(), "git_commit": M._git("rev-parse", "HEAD"),
           "git_dirty": bool(M._git("status", "--porcelain", "--", str(M.HERE))), "params_md5": M._md5(M.PARAMS).get("md5"),
           "params": params, "inputs": [M._md5(p) for p in inputs], "outputs": [M._md5(p) for p in outputs],
           "counts": counts or {}, "notes": notes, "local": True}
    Path(outdir).mkdir(parents=True, exist_ok=True)
    (Path(outdir) / f"{name}.manifest.json").write_text(json.dumps(rec, indent=1, default=str))
    return rec


def run_params(version, mask=None):
    """params of a build (guarded, in the manifest and work/params.json), extra inputs and notes of its mask."""
    mask = mask or MASK
    prm = {"version": version, "half_km": HALF, "richardson_k": T["p1"]["richardson_k"], "eft": True,
           "centred": True, "cells_km": [1.0, 0.5], "t05_taper_km": [R_T05, R_COMB], "grid_fallback_s": FALLBACK}
    surf_inputs, surf_notes = [], {"surface_mask": mask, "surface_mask_rule": MASK_RULE[mask]}
    if mask != "d_b005":           # d_b005 keeps the params of the builds to v0.3.1 (absent key = d_b005)
        import b4_surface
        srcs = b4_surface.sources()
        code = [Path(__file__).resolve(), Path(b4_surface.__file__).resolve()]
        prm.update({"code_md5": {f.name: hashlib.md5(f.read_bytes()).hexdigest() for f in code},
                    "surface_mask": mask, "surface_sources_md5": {s.name: s.md5 for s in srcs},
                    "fluid_velocity_km_s": fluid_velocities(), "station_patch_km": list(STATION_PATCH_KM),
                    "fluid_node_times": FLUID_NODES})
        surf_inputs = ([b4_surface._dir() / "gmrt" / s.name for s in srcs[:-1]]
                       + [b4_surface._dir() / b4_surface.P["earth_relief"]["cache"]])
        surf_notes["surface_source"] = b4_surface.SOURCE
    if SC:
        prm["shallow_crust"] = SC
        surf_notes["shallow_crust_rule"] = ("offshore rock cells with centre within thickness_km below the seafloor: "
                                            "Vs = min(Vs, Vp / vpvs) (D-B070 test, D-B073)")
    return prm, surf_inputs, surf_notes


def check_guard(out, work, prm):
    """A rerun keeps the grids and fields already there: refuse them when built with other or unknown parameters
    (time_b4_v0.3.0 has no taper; a d_b005 build has no surface_mask key, a merged one has it), or the manifest would
    give them the new ones. Returns work/params.json (written by main before any field, so an interrupted run can be
    resumed)."""
    stamp = work / "params.json"
    old = [json.loads(p.read_text())["params"] for p in out.glob("b4_time_grids_*.manifest.json")]
    old += [json.loads(stamp.read_text())] if stamp.exists() else []
    if any(o != json.loads(json.dumps(prm)) for o in old) or (
            not old and (any(out.glob("*.time.hdr")) or any(work.glob("field_*")))):
        sys.exit(f"{out} holds grids or fields built with other or unknown parameters {old}: use a new TIME_NAME")
    return stamp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("version")
    ap.add_argument("time_name")
    ap.add_argument("--nproc", type=int, default=40)
    ap.add_argument("--stations", help="comma-separated labels (test)")
    ap.add_argument("--root", default=str(RELOC), help="output root (tests); manifest then local only")
    a = ap.parse_args()
    ds = xr.open_dataset(Path(X["store"]) / X["name"] / f"v{a.version}.nc")
    root = Path(a.root)
    out = root / a.time_name
    work = root / f"work_{a.time_name}"
    prm, surf_inputs, surf_notes = run_params(a.version)
    stamp = check_guard(out, work, prm)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    stamp.write_text(json.dumps(prm))
    fields, shapes = {}, {}
    for dx in (1.0, 0.5):
        fields[dx], shapes[dx] = build_field(ds, dx, work)
    if MASK == "merged":
        surface_nodes(work)
    st = gtsrce()
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
    masks = {p.name: json.loads(p.read_text())["cells"] for p in sorted(work.glob("field_*.mask.json"))}
    fln = {}                                           # FLUID_NODES record per grid (merged mask)
    for lab, ph, *_ in res:
        d = (work / f"{lab}.{ph}.t05off").read_text().split() if (work / f"{lab}.{ph}.t05off").exists() else []
        if len(d) >= 7:
            fln[f"{ph}.{lab}"] = {"nodes_replaced": int(d[2]), "negative_set_0": int(d[3]),
                                  "first_node_change_s": [float(d[4]), float(d[5]), float(d[6])]}
    if fln:
        surf_notes["fluid_node_times"] = fln
        surf_notes["fluid_node_times_about"] = ("FLUID_NODES: nodes above the merged surface = time at the shallowest "
                                                "node at or below it + the rock-everywhere 1 km time difference; "
                                                "first_node_change_s = median, p99, max |change| at the node just "
                                                "above the surface")
        print("fluid nodes: negative set to 0 in", sum(v["negative_set_0"] > 0 for v in fln.values()), "grids;",
              "max first-node change", max(v["first_node_change_s"][2] for v in fln.values()), "s", flush=True)
    (write_manifest if root == RELOC else local_manifest)(
                   f"b4_time_grids_{a.time_name}", out,
                   inputs=[Path(X["store"]) / X["name"] / f"v{a.version}.nc", NLL_MODEL / "cascadia3d.P.mod.buf",
                           NLL_MODEL / "cascadia3d.S.mod.buf", NLL_MODEL / "stations_GTSRCE.in"] + surf_inputs,
                   params=prm,
                   counts={"station_phases": len(jobs), "ok": sum(r[2] == "ok" for r in res),
                           "fallback": sum(bool(f) for f in fbk.values())},
                   notes={**surf_notes, "field_cells": masks, "t05_minus_t_s": off, "t05_minus_t_summary": summ, "t05_minus_t_shell_km": list(R_DIAG),
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

"""Tests of b4_time_grids.build_field's merged-surface water/rock boundary (D-B068, v5.1 piece 'field').

  guard  --merged-root DIR --out D       the params guard (d_b005 = v0.3.1 params; merged refused on older builds)
  rockfield --work DIR --out D           the rock-everywhere 1 km field = the field in every rock cell, no fluid velocity
  repro  --work DIR [--ref WORK]          d_b005 fields rebuilt into DIR: byte-identical to the reference work fields
                                          (default: work_time_b4_v0.3.1, the production v0.3.1 build)?
  cells  --work DIR [--ref WORK] --out D  merged fields against the reference: changed cells by class and by height of
                                          the cell centre above the merged and the earth_relief surface; mask checks
                                          (no rock centre above surface_m except station source columns, no fluid
                                          centre below it; fluid = D-B005's water below sea level, air above; no water,
                                          air or blend velocity in rock; S fluid where P fluid)
  times  --new DIR --ref DIR --stations A,B --out D [--events CSV]
                                          travel times of the new grids against the reference grids: at grid nodes by
                                          depth below the merged surface and distance, and at the located hypocentres
                                          of a catalog (default v0.3.1 iter1)
  nlloc-prep / nlloc-run / nlloc-cmp      relocation of events whose arrivals all have new grids (see the functions)

Run with the models env python (xarray) and B4_SURFACE_DIR set when the production sources are not installed:
  B4_SURFACE_DIR=/wd1/mdenolle_data/tomo_v1/v51_build/surface/src nice -n 10 <models python> b4_field_check.py ...
Writes only under --out / --work.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b4_time_grids as B  # noqa: E402
from b2_fwd_common import RE, eft_depth, read_nll_hdr  # noqa: E402

RELOC = Path("/wd1/mdenolle_data/reloc_v5")
REF_WORK = RELOC / "work_time_b4_v0.3.1"
REF_TIME = RELOC / "time_b4_v0.3.1"
ITER1 = RELOC / "runs" / "b4_iter1_v0.3.1"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def open_ds(version="0.3.1"):
    import xarray as xr
    return xr.open_dataset(Path(B.X["store"]) / B.X["name"] / f"v{version}.nc")


# ------------------------------------------------------------------------------------------------------------ repro
def cmd_repro(a):
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    ds = open_ds(a.version)
    res = {}
    for dx in (1.0, 0.5):
        t = time.time()
        f, _ = B.build_field(ds, dx, work, mask="d_b005")
        for ph in ("P", "S"):
            new, ref = md5(f[ph]), md5(Path(a.ref) / f[ph].name)
            res[f"{dx:g}.{ph}"] = {"new": new, "ref": ref, "identical": new == ref}
        print(dx, round(time.time() - t), res, flush=True)
    (work / "repro.json").write_text(json.dumps(res, indent=1))
    print("ALL IDENTICAL" if all(r["identical"] for r in res.values()) else "DIFFERENT")


# ------------------------------------------------------------------------------------------------------------ guard
def cmd_guard(a):
    """The params guard: d_b005 params = the v0.3.1 build's; merged params refused on the v0.3.1 build and on a
    folder holding fields without a stamp; a merged test build accepted by merged params, refused by d_b005 ones."""
    res = {}

    def refused(out, work, prm):
        try:
            B.check_guard(Path(out), Path(work), prm)
            return False
        except SystemExit:
            return True
    d, _, _ = B.run_params("0.3.1", "d_b005")
    m, _, _ = B.run_params("0.3.1", "merged")
    man = json.loads((REF_TIME / "b4_time_grids_time_b4_v0.3.1.manifest.json").read_text())["params"]
    res["d_b005 params == v0.3.1 manifest params"] = json.loads(json.dumps(d)) == man
    res["v0.3.1 build accepted with d_b005"] = not refused(REF_TIME, REF_WORK, d)
    res["v0.3.1 build refused with merged"] = refused(REF_TIME, REF_WORK, m)
    res["merged test build accepted with merged"] = not refused(Path(a.merged_root) / "time_b4_v0.3.1m",
                                                               Path(a.merged_root) / "work_time_b4_v0.3.1m", m)
    res["merged test build refused with d_b005"] = refused(Path(a.merged_root) / "time_b4_v0.3.1m",
                                                           Path(a.merged_root) / "work_time_b4_v0.3.1m", d)
    tmp = Path(a.out) / "guard_tmp"
    (tmp / "work_x").mkdir(parents=True, exist_ok=True)
    (tmp / "work_x" / "field_1.P.buf").write_bytes(b"")
    res["fields without a stamp refused"] = refused(tmp / "x", tmp / "work_x", m)
    (tmp / "work_x" / "field_1.P.buf").unlink()
    res["empty folder accepted"] = not refused(tmp / "x", tmp / "work_x", m)
    res["merged params keys"] = sorted(set(m) - set(d))
    print(json.dumps(res, indent=1))
    (Path(a.out) / "guard.json").write_text(json.dumps(res, indent=1))
    assert all(v for k, v in res.items() if k != "merged params keys"), res


# ------------------------------------------------------------------------------------------------------------ rockfield
def cmd_rockfield(a):
    """field_1.<ph>.rock.buf (the rock-everywhere field of the fluid-node step) equals field_1.<ph>.buf in every rock
    cell (bitwise) and holds no water or air velocity."""
    work = Path(a.work)
    nx, ny, nz = B.field_shape(1.0)
    zs = B.G["z0"] + np.arange(nz) + 0.5
    sc = np.exp(zs / RE)
    res = {}
    for ph in ("P", "S"):
        f = np.fromfile(work / f"field_1.{ph}.buf", "<f4").reshape(nx, ny, nz)
        r = np.fromfile(work / f"field_1.{ph}.rock.buf", "<f4").reshape(nx, ny, nz)
        fp = np.fromfile(work / "field_1.P.buf", "<f4").reshape(nx, ny, nz)
        rock = 1.0 / (fp * sc) >= B.FLUID_VP
        vr = 1.0 / (r.astype("f8") * sc)
        res[ph] = {"rock_cells": int(rock.sum()), "rock_cells_bitwise_equal": int((f[rock] == r[rock]).sum()),
                   "rock_field_min_v": float(vr.min()), "fluid_cells_in_true_field": int((~rock).sum())}
        if ph == "P":
            res[ph]["rock_field_cells_below_1.6"] = int((vr < B.FLUID_VP).sum())
    print(json.dumps(res, indent=1))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "rockfield.json").write_text(json.dumps(res, indent=1))


# ------------------------------------------------------------------------------------------------------------ cells
def er_m(lon, lat):
    """earth_relief 01m alone (bilinear, the last source of b4_surface): the D-B005 surface (a1_build_model.py)."""
    import b4_surface as S
    srcs = S.sources()
    x, y = np.broadcast_arrays(np.asarray(lon, "f8"), np.asarray(lat, "f8"))
    v, _ = S._eval(srcs[-1:], x.ravel(), y.ravel())
    return v.reshape(x.shape)


def station_columns(dx, shape):
    """Independent re-implementation of the station source columns: dict (i, j) -> first rock cell index."""
    from pyproj import Proj
    pr = Proj(proj="lcc", lat_0=45.0, lon_0=-125.8, lat_1=41.0, lat_2=49.0, ellps="WGS84")
    G = B.G
    top = np.full(shape[:2], shape[2], np.int32)
    for line in (B.NLL_MODEL / "stations_GTSRCE.in").read_text().splitlines():
        if not line.startswith("GTSRCE"):
            continue
        s = line.split()
        x, y = pr(float(s[4]), float(s[3]))
        x, y = x / 1e3, y / 1e3
        i0, j0 = int(np.floor(x - G["x0"])), int(np.floor(y - G["y0"]))     # 1 km lattice cell
        # cells whose centre x0 + dx (i + 1/2) lies in [x0 + i0 - 1, x0 + i0 + 2]
        ia, ib = int(np.ceil((i0 - 1) / dx - 0.5)), int(np.floor((i0 + 2) / dx - 0.5))
        ja, jb = int(np.ceil((j0 - 1) / dx - 0.5)), int(np.floor((j0 + 2) / dx - 0.5))
        k = int(np.floor((float(eft_depth(float(s[5]))) - G["z0"]) / dx))
        sl = (slice(max(ia, 0), min(ib, shape[0] - 1) + 1), slice(max(ja, 0), min(jb, shape[1] - 1) + 1))
        top[sl] = np.minimum(top[sl], k)
    return top


HB = np.arange(-90.0, 10.01, 0.25)     # km, bins of the height of a cell centre above a surface


def _cells_slab(args):
    dx, i0, i1, work, ref, surf, er, lo, hi, ktop, stnear, fv = args
    G = B.G
    nx, ny, nz = B.field_shape(dx)
    zs = G["z0"] + dx * np.arange(nz) + dx / 2
    zt = RE * (1 - np.exp(-zs / RE))
    sc = np.exp(zs / RE)
    out = {"n": 0}
    v = {}
    for ph in ("P", "S"):
        for tag, w in (("new", work), ("ref", ref)):
            a = np.memmap(Path(w) / f"field_{dx:g}.{ph}.buf", dtype="<f4", mode="r", shape=(nx, ny, nz))[i0:i1]
            v[tag, ph] = (dx / (np.asarray(a, "f8") * sc))                    # true velocity, km/s
            v[tag, ph + "s"] = np.asarray(a)
    hm = -zt[None, None, :] - surf[i0:i1, :, None]                                # km, centre above merged surface
    he = -zt[None, None, :] - er[i0:i1, :, None]                                  # km, centre above earth_relief
    kk = np.arange(nz)[None, None, :]
    patch = kk >= ktop[i0:i1, :, None]
    above = hm > 0
    fl_new = v["new", "P"] < B.FLUID_VP
    fl_ref = v["ref", "P"] < B.FLUID_VP
    water = (zt >= 0)[None, None, :]
    out["n"] = int(fl_new.size)
    out["fluid_new"] = int(fl_new.sum())
    out["patch_cells"] = int(patch.sum())
    # mask checks
    out["rock_above_surface"] = int((~fl_new & above).sum())
    out["rock_above_surface_in_patch"] = int((~fl_new & above & patch).sum())
    out["rock_above_surface_outside_patch"] = int((~fl_new & above & ~patch).sum())
    out["fluid_below_surface"] = int((fl_new & ~above).sum())
    out["fluid_in_patch"] = int((fl_new & patch).sum())
    out["max_patch_rock_height_km"] = float(hm[~fl_new & above].max()) if (~fl_new & above).any() else None
    for ph in ("P", "S"):
        vn = v["new", ph]
        wv, av = fv[ph]["water"], fv[ph]["air"]
        out[f"{ph}_fluid_wrong_velocity"] = int((fl_new & water & (np.abs(vn - wv) > 1e-5 * wv)).sum()
                                                + (fl_new & ~water & (np.abs(vn - av) > 1e-5 * av)).sum())
        rock_v = vn[~fl_new]
        out[f"{ph}_rock_min_v"] = float(rock_v.min())
        out[f"{ph}_rock_fluid_like"] = int((np.isclose(rock_v, wv, rtol=1e-5) | np.isclose(rock_v, av, rtol=1e-5)).sum())
    out["S_fluid_where_P_rock"] = int((~fl_new & ((np.abs(v["new", "S"] - fv["S"]["water"]) < 1e-6)
                                                  | (np.abs(v["new", "S"] - fv["S"]["air"]) < 1e-6))).sum())
    # changes against the reference
    for ph in ("P", "S"):
        d_any = v["new", ph + "s"] != v["ref", ph + "s"]
        rel = np.abs(v["new", ph] - v["ref", ph]) / v["ref", ph]
        d_mat = rel > 1e-5
        out[f"{ph}_changed_any"] = int(d_any.sum())
        out[f"{ph}_changed"] = int(d_mat.sum())
        for cls, sel in (("rock_rock", ~fl_ref & ~fl_new), ("rock_fluid", ~fl_ref & fl_new),
                         ("fluid_rock", fl_ref & ~fl_new), ("fluid_fluid", fl_ref & fl_new)):
            out[f"{ph}_changed_{cls}"] = int((d_mat & sel).sum())
        out[f"{ph}_changed_in_patch"] = int((d_mat & patch).sum())
        # unchanged where it must be: rock in both, all 8 D-B005 nodes rock -> identical
        dist = np.minimum(np.abs(hm), np.abs(he))
        out[f"{ph}_hist_hm"] = np.histogram(np.clip(hm[d_mat], HB[0], HB[-1]), HB)[0].tolist()
        out[f"{ph}_hist_he"] = np.histogram(np.clip(he[d_mat], HB[0], HB[-1]), HB)[0].tolist()
        out[f"{ph}_hist_dist"] = np.histogram(np.clip(dist[d_mat], 0, HB[-1]), np.arange(0, 10.01, 0.25))[0].tolist()
        # nearness: height of the centre outside the local surface band [lo, hi] (both surfaces over the columns
        # within 1 km, the reach of D-B005's trilinear blur); fluid -> fluid changes: distance to sea level
        elev = -zt[None, None, :]
        dband = np.maximum(0.0, np.maximum(elev - hi[i0:i1, :, None], lo[i0:i1, :, None] - elev))
        ff = d_mat & fl_ref & fl_new
        nf = d_mat & ~(fl_ref & fl_new)
        dsea = np.minimum(np.abs(zt)[None, None, :], dband)           # sea level or the land/sea-floor band
        out[f"{ph}_nonff_band_dist_hist"] = np.histogram(np.clip(dband[nf], 0, 10), np.arange(0, 10.01, 0.25))[0].tolist()
        out[f"{ph}_ff_sea_dist_hist"] = np.histogram(np.clip(dsea[ff], 0, 10), np.arange(0, 10.01, 0.25))[0].tolist()
        out[f"{ph}_max_nonff_band_dist_km"] = float(dband[nf].max()) if nf.any() else 0.0
        out[f"{ph}_max_ff_sea_dist_km"] = float(dsea[ff].max()) if ff.any() else 0.0
        far = (nf & (dband > 1.0)) | (ff & (dsea > 1.0))
        out[f"{ph}_changed_not_near"] = int(far.sum())
        sn = stnear[i0:i1, :, None]
        out[f"{ph}_changed_not_near_outside_station_columns"] = int((far & ~sn).sum())
        out[f"{ph}_changed_near_stations"] = int((d_mat & sn).sum())
        far = far & ~sn
        out[f"{ph}_not_near_examples"] = [[int(i + i0), int(j), int(k), float(hm[i, j, k]), float(he[i, j, k]),
                                          float(dband[i, j, k]), float(v["ref", ph][i, j, k]),
                                          float(v["new", ph][i, j, k])] for i, j, k in np.argwhere(far)[:20]]
        # boundary shift per column: rock cells per column, new minus reference (km), offshore / onshore columns
        if ph == "P":
            dr = (dx * ((~fl_new).sum(axis=2) - (~fl_ref).sum(axis=2)))
            off = surf[i0:i1] < 0
            for nm, sel in (("offshore", off), ("onshore", ~off)):
                u, c = np.unique(np.round(dr[sel], 3), return_counts=True)
                out[f"rock_column_shift_km_{nm}"] = {f"{x:g}": int(n) for x, n in zip(u, c)}
        # cells deeper than 2 km below both surfaces that changed
        deep = d_mat & (hm < -2.0) & (he < -2.0)
        out[f"{ph}_changed_deep2"] = int(deep.sum())
        idx = np.argwhere(deep)[:20]
        out[f"{ph}_deep2_examples"] = [[int(i + i0), int(j), int(k), float(hm[i, j, k]), float(he[i, j, k]),
                                       float(v["ref", ph][i, j, k]), float(v["new", ph][i, j, k]), bool(patch[i, j, k])]
                                      for i, j, k in idx]
        out[f"{ph}_max_rel"] = float(rel.max())
        dv = v["new", ph] - v["ref", ph]
        out[f"{ph}_dv_hist"] = np.histogram(np.clip(dv[d_mat], -8, 8), np.arange(-8, 8.01, 0.5))[0].tolist()
        # rock -> rock changes by depth below merged surface (velocity change magnitude)
    return out


def cmd_cells(a):
    import b4_surface as S
    work, ref, outd = Path(a.work), Path(a.ref), Path(a.out)
    outd.mkdir(parents=True, exist_ok=True)
    fv = B.fluid_velocities()
    G = B.G
    res = {"work": str(work), "ref": str(ref), "fluid_velocity_km_s": fv, "surface_source": S.SOURCE}
    for dx in a.dx:
        t = time.time()
        shape = B.field_shape(dx)
        nx, ny, nz = shape
        xs = G["x0"] + dx * (np.arange(nx) + 0.5)
        ys = G["y0"] + dx * (np.arange(ny) + 0.5)
        X2, Y2 = np.meshgrid(xs, ys, indexing="ij")
        lon, lat = B.proj()(X2 * 1e3, Y2 * 1e3, inverse=True)
        surf = S.surface_m(lon, lat) / 1e3
        er = er_m(lon, lat) / 1e3
        from scipy.ndimage import maximum_filter, minimum_filter
        w = 2 * int(np.ceil(1.0 / dx)) + 1                          # columns within 1 km (D-B005 node spacing)
        lo = np.minimum(minimum_filter(surf, w, mode="nearest"), minimum_filter(er, w, mode="nearest"))
        hi = np.maximum(maximum_filter(surf, w, mode="nearest"), maximum_filter(er, w, mode="nearest"))
        ktop = station_columns(dx, shape)
        ktop_b = B.station_rock_top(dx, shape)
        same_top = bool((ktop == ktop_b).all())
        step = max(1, int(round(60 / dx)))
        # station neighbourhood: the station source columns widened by 1 km (D-B005's node patch blurs that far)
        stnear = minimum_filter(ktop, 2 * int(np.ceil(1.0 / dx)) + 1, mode="nearest") < nz
        jobs = [(dx, i0, min(i0 + step, nx), work, ref, surf, er, lo, hi, ktop, stnear, fv)
                for i0 in range(0, nx, step)]
        with ProcessPoolExecutor(a.nproc) as ex:
            parts = list(ex.map(_cells_slab, jobs))
        tot = {}
        for p in parts:
            for k, v in p.items():
                if k.endswith("_examples"):
                    tot.setdefault(k, []).extend(v)
                elif isinstance(v, dict):
                    d = tot.setdefault(k, {})
                    for kk_, vv in v.items():
                        d[kk_] = d.get(kk_, 0) + vv
                elif k.startswith("max_") or k.endswith("_max_rel") or "_max_" in k:
                    tot[k] = max([x for x in (tot.get(k), v) if x is not None], default=None)
                elif k.endswith("_min_v"):
                    tot[k] = min(tot.get(k, np.inf), v)
                elif isinstance(v, list):
                    tot[k] = (np.array(tot[k]) + np.array(v)).tolist() if k in tot else v
                else:
                    tot[k] = tot.get(k, 0) + v
        tot["station_columns_equal_builder"] = same_top
        tot["seconds"] = round(time.time() - t, 1)
        for ph in ("P", "S"):
            for nm in ("hm", "he"):
                h = np.array(tot[f"{ph}_hist_{nm}"])
                nzb = np.flatnonzero(h)
                tot[f"{ph}_changed_height_range_{nm}_km"] = ([float(HB[nzb[0]]), float(HB[nzb[-1] + 1])]
                                                             if nzb.size else None)
            h = np.array(tot[f"{ph}_hist_dist"])
            nzb = np.flatnonzero(h)
            tot[f"{ph}_changed_max_dist_to_a_surface_km"] = float(0.25 * (nzb[-1] + 1)) if nzb.size else 0.0
            c = np.cumsum(h) / max(h.sum(), 1)
            tot[f"{ph}_changed_within_km"] = {f"{0.25 * (i + 1):g}": round(float(c[i]), 6) for i in (1, 3, 5, 7, 11)}
        res[f"{dx:g}"] = tot
        print(dx, json.dumps({k: v for k, v in tot.items() if "hist" not in k and "examples" not in k}), flush=True)
    res["height_bins_km"] = HB.tolist()
    (outd / f"cells_{'_'.join(f'{d:g}' for d in a.dx)}.json").write_text(json.dumps(res, indent=1))


# ------------------------------------------------------------------------------------------------------------ times
def _grid(root):
    g = read_nll_hdr(root)
    return g, np.fromfile(f"{root}.buf", "<f4").reshape(g["nx"], g["ny"], g["nz"])


def trilin(g, a, x, y, z):
    fx, fy, fz = (x - g["x0"]) / g["dx"], (y - g["y0"]) / g["dx"], (z - g["z0"]) / g["dx"]
    ok = (fx >= 0) & (fy >= 0) & (fz >= 0) & (fx <= g["nx"] - 1) & (fy <= g["ny"] - 1) & (fz <= g["nz"] - 1)
    ix = np.clip(np.floor(fx).astype(int), 0, g["nx"] - 2)
    iy = np.clip(np.floor(fy).astype(int), 0, g["ny"] - 2)
    iz = np.clip(np.floor(fz).astype(int), 0, g["nz"] - 2)
    wx, wy, wz = fx - ix, fy - iy, fz - iz
    out = np.zeros(len(x))
    for dx_, ax in ((0, 1 - wx), (1, wx)):
        for dy_, ay in ((0, 1 - wy), (1, wy)):
            for dz_, az in ((0, 1 - wz), (1, wz)):
                out += ax * ay * az * a[ix + dx_, iy + dy_, iz + dz_]
    out[~ok] = np.nan
    return out


def q(x, ps=(50, 95, 99)):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"n": 0}
    return {"n": int(x.size), **{f"p{p}": round(float(np.percentile(x, p)), 4) for p in ps},
            "max": round(float(x.max()), 4)}


def _times_one(args):
    import b4_surface as S
    lab, ph, new, ref, ev = args
    rn, rr = Path(new) / f"cascadia3d.{ph}.{lab}.time", Path(ref) / f"cascadia3d.{ph}.{lab}.time"
    gn, tn = _grid(rn)
    gr, tr = _grid(rr)
    assert {k: gn[k] for k in ("nx", "ny", "nz", "x0", "y0", "z0", "dx")} == \
        {k: gr[k] for k in ("nx", "ny", "nz", "x0", "y0", "z0", "dx")}, (lab, ph)
    src = [float(v) for v in Path(f"{rn}.hdr").read_text().splitlines()[1].split()[1:4]]
    xs = gn["x0"] + gn["dx"] * np.arange(gn["nx"])
    ys = gn["y0"] + gn["dx"] * np.arange(gn["ny"])
    zs = gn["z0"] + gn["dx"] * np.arange(gn["nz"])
    X2, Y2 = np.meshgrid(xs, ys, indexing="ij")
    lon, lat = B.proj()(X2 * 1e3, Y2 * 1e3, inverse=True)
    surf = S.surface_m(lon, lat) / 1e3
    zt = RE * (1 - np.exp(-zs / RE))
    below = zt[None, None, :] + surf[:, :, None]                     # km below the merged surface (node)
    hd = np.hypot(X2 - src[0], Y2 - src[1])[:, :, None] * np.ones((1, 1, gn["nz"]))
    d = (tn.astype("f8") - tr.astype("f8"))
    rock = below >= 0
    o = {"station": lab, "phase": ph, "src_xyz": src}
    o["all_rock_nodes"] = q(np.abs(d[rock]))
    o["mean_rock"] = round(float(d[rock].mean()), 5)
    for nm, sel in (("rock_0_1km_below_surface", rock & (below < 1)), ("rock_1_3km", (below >= 1) & (below < 3)),
                    ("rock_3_10km", (below >= 3) & (below < 10)), ("deep_10km_plus", below >= 10),
                    ("deep_10km_plus_dist_gt_50km", (below >= 10) & (hd > 50)),
                    ("near_station_5km_rock", rock & (hd <= 5) & (below < 5))):
        o[nm] = q(np.abs(d[sel]))
        o[nm + "_median_signed"] = round(float(np.median(d[sel])), 5) if sel.any() else None
    if ev is not None:
        x, y, zf, zb = ev                                             # zb: km below the merged surface (true)
        tnv, trv = trilin(gn, tn, x, y, zf), trilin(gr, tr, x, y, zf)
        dd = tnv - trv
        ok = np.isfinite(dd)
        dist = np.hypot(x - src[0], y - src[1])
        o["events"] = q(np.abs(dd[ok & (zb >= 0)]))
        o["events_median_signed"] = round(float(np.median(dd[ok & (zb >= 0)])), 5) if (ok & (zb >= 0)).any() else None
        o["events_above_merged_surface_n"] = int((ok & (zb < 0)).sum())
        o["events_classes"] = {}
        for nm, sel in (("below_0_0.5km", (zb >= 0) & (zb < 0.5)), ("below_0.5_2km", (zb >= 0.5) & (zb < 2)),
                        ("below_2_5km", (zb >= 2) & (zb < 5)), ("below_5km_plus", zb >= 5),
                        ("below_0_2km_dist_lt_10km", (zb >= 0) & (zb < 2) & (dist < 10)),
                        ("below_0_2km_dist_ge_10km", (zb >= 0) & (zb < 2) & (dist >= 10)),
                        ("below_2km_plus_dist_lt_10km", (zb >= 2) & (dist < 10))):
            sel = sel & ok
            o["events_classes"][nm] = {**q(np.abs(dd[sel])),
                                       "median_signed": round(float(np.median(dd[sel])), 5) if sel.any() else None}
    return o


def cmd_times(a):
    import pandas as pd
    outd = Path(a.out)
    outd.mkdir(parents=True, exist_ok=True)
    ev = None
    if a.events:
        c = pd.read_csv(a.events)
        c = c[c.status == "LOCATED"]
        import b4_surface as S
        x, y = B.proj()(c.lon.values, c.lat.values)
        zf = c.depth.values.astype("f8")                                   # NLLoc depth = z' (B4 frame)
        zb = RE * (1 - np.exp(-zf / RE)) + S.surface_m(c.lon.values, c.lat.values) / 1e3
        ev = (np.asarray(x) / 1e3, np.asarray(y) / 1e3, zf, zb)
    jobs = [(lab, ph, a.new, a.ref, ev) for lab in a.stations.split(",") for ph in ("P", "S")]
    with ProcessPoolExecutor(a.nproc) as ex:
        res = list(ex.map(_times_one, jobs))
    (outd / "times.json").write_text(json.dumps(res, indent=1))
    rows = []
    for r in res:
        rows.append({"station": r["station"], "phase": r["phase"],
                     **{f"{k}_{s}": r[k].get(s) for k in ("all_rock_nodes", "rock_0_1km_below_surface", "rock_1_3km",
                                                          "rock_3_10km", "deep_10km_plus",
                                                          "deep_10km_plus_dist_gt_50km", "near_station_5km_rock")
                        for s in ("p50", "p95", "max")},
                     "deep_median_signed": r["deep_10km_plus_median_signed"],
                     **({f"events_{s}": r["events"].get(s) for s in ("n", "p50", "p95", "max")} if "events" in r else {}),
                     "events_median_signed": r.get("events_median_signed"),
                     **{f"ev_{k}_{s}": v.get(s) for k, v in r.get("events_classes", {}).items()
                        for s in ("n", "p50", "p95", "max", "median_signed")}})
    df = pd.DataFrame(rows)
    df.to_csv(outd / "times_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(df.to_string(index=False))


# ------------------------------------------------------------------------------------------------------------ nlloc
def read_obs_blocks(path):
    """NLLOC_OBS file -> list of (header comment lines + phase lines) blocks separated by blank lines."""
    blocks, cur = [], []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            if cur:
                blocks.append(cur)
                cur = []
            continue
        cur.append(line)
    if cur:
        blocks.append(cur)
    return blocks


def block_stations(b):
    return [line.split()[0] for line in b if line.strip() and not line.startswith(("#", "PUBLIC_ID"))]


def cmd_nlloc_prep(a):
    """Events of the reference run whose every arrival is at one of --stations: per tile one obs file and three
    control files (ref grids + production floor, ref grids + merged floor, new grids + merged floor), each the tile's
    own nlloc.in with only LOCFILES (obs, grid root, output) and LOCTOPO_SURFACE changed."""
    import pandas as pd
    stations = set(a.stations.split(","))
    run = Path(a.run)
    outd = Path(a.out)
    loc = pd.read_csv(run / "locations.csv")
    keep_ids = []
    tiles = sorted(p for p in run.iterdir() if p.is_dir() and (p / "events.obs").exists())
    sel_tiles = []
    for t in tiles:
        blocks = read_obs_blocks(t / "events.obs")
        mine = []
        for b in blocks:
            sta = block_stations(b)
            if sta and set(sta) <= stations:
                mine.append(b)
        if a.max_per_tile:
            mine = mine[:a.max_per_tile]
        if mine:
            sel_tiles.append((t, mine))
    variants = {"ref_prodfloor": (a.ref_time, a.floor_prod), "ref_merged": (a.ref_time, a.floor_merged),
                "new_merged": (a.new_time, a.floor_merged)}
    if a.only:                                        # add or redo some variants: NAME=TIME_DIR (merged floor)
        variants = {k: (v, a.floor_merged) for k, v in (x.split("=") for x in a.only.split(","))}
    assert all(t for t, _ in variants.values()), "--new-time is required without --only"
    n = 0
    for t, mine in sel_tiles:
        for v, (troot, floor) in variants.items():
            wd = outd / v / t.name
            wd.mkdir(parents=True, exist_ok=True)
            (wd / "events.obs").write_text("\n\n".join("\n".join(b) for b in mine) + "\n\n")
            lines = []
            for line in (t / "nlloc.in").read_text().splitlines():
                if line.startswith("LOCFILES"):
                    line = f"LOCFILES {wd / 'events.obs'} NLLOC_OBS {Path(troot) / 'cascadia3d'} {wd / 'loc'}"
                elif line.startswith("LOCTOPO_SURFACE"):
                    tok = line.split()
                    line = f"LOCTOPO_SURFACE {floor} {tok[2] if len(tok) > 2 else 0}"
                lines.append(line)
            (wd / "nlloc.in").write_text("\n".join(lines) + "\n")
        n += len(mine)
    print(f"{len(sel_tiles)} tiles, {n} event blocks with arrivals only at {len(stations)} stations")
    (outd / f"prep{'_' + '_'.join(variants) if a.only else ''}.json").write_text(json.dumps(
        {"tiles": [t.name for t, _ in sel_tiles], "events": n, "variants": variants, "stations": sorted(stations)},
        indent=1))


def _run_one(wd):
    t = time.time()
    with open(Path(wd) / "nlloc.log", "w") as f:
        r = subprocess.run([str(B.NLL_BIN / "NLLoc"), "nlloc.in"], cwd=wd, stdout=f, stderr=subprocess.STDOUT)
    log = (Path(wd) / "nlloc.log").read_text(errors="replace")
    return str(wd), r.returncode, round(time.time() - t, 1), "ERROR: reading Topo" in log


def cmd_nlloc_run(a):
    outd = Path(a.out)
    wds = sorted(p.parent for p in outd.glob("*/*/nlloc.in")
                 if not a.variants or p.parent.parent.name in a.variants.split(","))
    with ProcessPoolExecutor(a.nproc) as ex:
        res = list(ex.map(_run_one, wds))
    bad = [r for r in res if r[1] != 0 or r[3]]
    print(f"{len(res)} runs, {len(bad)} with an error: {bad[:5]}")
    (outd / "run.json").write_text(json.dumps(res, indent=1))


def parse_hyp(path):
    """NLLoc .hyp (SAVE_NLLOC_ALL sum file) -> list of dicts: id (from the PUBLIC_ID / obs comment), status, lat, lon,
    z (km, frame of the grids), rms, nphs, sigma_z."""
    out, cur = [], None
    for line in Path(path).read_text(errors="replace").splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] == "NLLOC":
            cur = {"file": t[1].strip('"'), "status": t[2].strip('"') if len(t) > 2 else ""}
        elif t[0] == "SIGNATURE" and cur is not None:
            pass
        elif t[0] == "HYPOCENTER" and cur is not None:
            cur.update(x=float(t[2]), y=float(t[4]), z=float(t[6]))
        elif t[0] == "GEOGRAPHIC" and cur is not None:
            cur.update(lat=float(t[9]), lon=float(t[11]), z=float(t[13]))
        elif t[0] == "QUALITY" and cur is not None:
            cur.update(rms=float(t[8]), nphs=int(t[10]), gap=float(t[12]))
        elif t[0] == "STATISTICS" and cur is not None:
            cur.update(cov_zz=float(t[18]))
        elif t[0] == "PUBLIC_ID" and cur is not None:
            cur["public_id"] = t[1]
        elif t[0] == "END_NLLOC" and cur is not None:
            out.append(cur)
            cur = None
    return out


def cmd_nlloc_cmp(a):
    import pandas as pd
    import b4_surface as S
    outd = Path(a.out)
    frames = {}
    names = ["ref_prodfloor", "ref_merged", "new_merged"] + sorted(
        p.name for p in outd.iterdir() if p.is_dir() and p.name.startswith("fix") and list(p.glob("*/loc.sum.grid0.loc.hyp")))
    for v in names:
        rows = []
        for hyp in sorted(outd.glob(f"{v}/*/loc.sum.grid0.loc.hyp")):
            for r in parse_hyp(hyp):
                r["tile"] = hyp.parent.name
                rows.append(r)
        df = pd.DataFrame(rows)
        df["ID"] = df.public_id.str.replace("v4_", "").astype(int)
        assert df.ID.is_unique, v
        frames[v] = df.set_index("ID")
    ids = frames["ref_prodfloor"].index
    m = pd.DataFrame(index=ids)
    m["tile"] = frames["ref_prodfloor"].tile
    for v, df in frames.items():
        for c in ("status", "lat", "lon", "z", "rms", "nphs", "cov_zz"):
            m[f"{c}_{v}"] = df[c].reindex(ids)
        ok = m[f"status_{v}"] == "LOCATED"
        zt = RE * (1 - np.exp(-m[f"z_{v}"].astype(float) / RE))
        s = S.surface_m(m[f"lon_{v}"].astype(float).values, m[f"lat_{v}"].astype(float).values) / 1e3
        m[f"zbelow_{v}"] = zt + s                                  # km below the merged surface (true depth)
        m[f"above_{v}"] = ok & (m[f"zbelow_{v}"] < 0)
    prod = pd.read_csv(Path(a.run) / "locations.csv").set_index("ID")
    pr = prod.reindex(ids)
    rep = {"events": int(len(ids))}
    okp = (pr.status == "LOCATED") & (m.status_ref_prodfloor == "LOCATED")
    same = okp & (np.abs(pr.lat - m.lat_ref_prodfloor) < 1e-6) & (np.abs(pr.lon - m.lon_ref_prodfloor) < 1e-6) \
        & (np.abs(pr.depth - m.z_ref_prodfloor) < 1e-5)
    rep["ref_prodfloor reproduces production locations.csv"] = {
        "same_status": int((pr.status.values == m.status_ref_prodfloor.values).sum()),
        "located_both": int(okp.sum()), "identical_hypocentre": int(same.sum())}
    for v in frames:
        ok = m[f"status_{v}"] == "LOCATED"
        rep[v] = {"located": int(ok.sum()), "above_merged_surface": int(m.loc[ok, f"above_{v}"].sum()),
                  "km_below_merged_surface": q(m.loc[ok, f"zbelow_{v}"], (5, 50, 95)),
                  "within_0.1km_of_surface": int((ok & (m[f"zbelow_{v}"] < 0.1)).sum())}
    m.to_csv(outd / "compare.csv")

    def shift(v1, v2, sel=None):
        ok = (m[f"status_{v1}"] == "LOCATED") & (m[f"status_{v2}"] == "LOCATED")
        if sel is not None:
            ok &= sel
        mm = m[ok]
        dz = (mm[f"z_{v2}"] - mm[f"z_{v1}"]).astype(float)
        dy = (mm[f"lat_{v2}"] - mm[f"lat_{v1}"]).astype(float) * 111.19
        dx = (mm[f"lon_{v2}"] - mm[f"lon_{v1}"]).astype(float) * 111.19 * np.cos(np.radians(mm[f"lat_{v1}"].astype(float)))
        dh = np.hypot(dx, dy)
        drms = (mm[f"rms_{v2}"] - mm[f"rms_{v1}"]).astype(float)
        return {"n": int(ok.sum()), "identical": int(((dz.abs() < 1e-6) & (dh < 1e-6)).sum()),
                "status_changed": int(((m[f"status_{v1}"] == "LOCATED") != (m[f"status_{v2}"] == "LOCATED")).sum()),
                "dz_km": q(dz, (1, 5, 50, 95, 99)), "absdz_km": q(dz.abs()), "dh_km": q(dh),
                "drms_s": q(drms, (5, 50, 95)), "dz_gt_1km": int((dz.abs() > 1).sum())}
    rep["field effect: new_merged - ref_merged"] = shift("ref_merged", "new_merged")
    near = m.zbelow_ref_merged < 0.5
    rep["field effect, events < 0.5 km below the surface in ref_merged"] = shift("ref_merged", "new_merged", near)
    rep["field effect, events >= 0.5 km below the surface in ref_merged"] = shift("ref_merged", "new_merged", ~near)
    rep["floor effect: ref_merged - ref_prodfloor"] = shift("ref_prodfloor", "ref_merged")
    rep["total: new_merged - ref_prodfloor"] = shift("ref_prodfloor", "new_merged")
    for v in names[3:]:
        rep[f"field effect: {v} - ref_merged"] = shift("ref_merged", v)
        rep[f"field effect, events < 0.5 km below the surface in ref_merged: {v}"] = shift("ref_merged", v, near)
        rep[f"field effect, events >= 0.5 km below the surface in ref_merged: {v}"] = shift("ref_merged", v, ~near)
        rep[f"{v} - new_merged"] = shift("new_merged", v)
        rep[f"total: {v} - ref_prodfloor"] = shift("ref_prodfloor", v)
    (outd / "compare.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("repro")
    s.add_argument("--work", required=True)
    s.add_argument("--ref", default=str(REF_WORK))
    s.add_argument("--version", default="0.3.1")
    s = sub.add_parser("guard")
    s.add_argument("--merged-root", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("rockfield")
    s.add_argument("--work", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("cells")
    s.add_argument("--work", required=True)
    s.add_argument("--ref", default=str(REF_WORK))
    s.add_argument("--out", required=True)
    s.add_argument("--nproc", type=int, default=8)
    s.add_argument("--dx", type=float, nargs="+", default=[1.0, 0.5])
    s = sub.add_parser("times")
    s.add_argument("--new", required=True)
    s.add_argument("--ref", default=str(REF_TIME))
    s.add_argument("--stations", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--events", default=str(ITER1 / "locations.csv"))
    s.add_argument("--nproc", type=int, default=8)
    s = sub.add_parser("nlloc-prep")
    s.add_argument("--run", default=str(ITER1))
    s.add_argument("--stations", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--ref-time", default=str(REF_TIME))
    s.add_argument("--new-time", help="time-grid folder of the new_merged variant (required without --only)")
    s.add_argument("--floor-prod", default=str(RELOC / "topo" / "topo_15s_cons_eft.bin"))
    s.add_argument("--floor-merged", required=True)
    s.add_argument("--max-per-tile", type=int, default=0)
    s.add_argument("--only", help="NAME=TIME_DIR[,...]: prepare only these variants (merged floor)")
    s = sub.add_parser("nlloc-run")
    s.add_argument("--out", required=True)
    s.add_argument("--variants", help="comma-separated variant folders (default all)")
    s.add_argument("--nproc", type=int, default=12)
    s = sub.add_parser("nlloc-cmp")
    s.add_argument("--out", required=True)
    s.add_argument("--run", default=str(ITER1))
    a = ap.parse_args()
    {"repro": cmd_repro, "guard": cmd_guard, "rockfield": cmd_rockfield, "cells": cmd_cells, "times": cmd_times, "nlloc-prep": cmd_nlloc_prep,
     "nlloc-run": cmd_nlloc_run, "nlloc-cmp": cmd_nlloc_cmp}[a.cmd](a)


if __name__ == "__main__":
    main()

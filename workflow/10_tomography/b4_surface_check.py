"""Checks of b4_surface.py (D-B068 merged surface, conservative NLLoc floor; v5.1 build piece 'surface').

Every output goes under --out (the v5.1 test folder); nothing is written elsewhere. Production files are only read.
  lookup     test (a): NLLoc's LOCTOPO_SURFACE lookup reproduced (header parsed as read_grd does, float32 zdata,
             node = (int)(0.5 + (x - x_min)/x_inc), C truncation, rows north -> south). For >= 5 million points
             (uniform, inside GMRT patches, exactly on NLLoc node-cell boundaries +-1 ulp, on every source's grid
             lines and patch edges +-1 ulp) plus every located v0.3.1 iter1 epicentre, the highest elevation NLLoc
             accepts (h' = the float32 node value, back to the true frame) is compared with surface_m. Negative
             controls: the production D-B056 floor (earth_relief only) and a naive floor (surface_m at the nodes - 1 m).
  exact      brute force: for sampled node cells, surface_m at every grid-line intersection of every source inside
             the cell, and +-eps around each (one-sided limits), against the floor value + margin; and the floor
             against the production D-B056 file where no GMRT patch touches the cell (same earth_relief minimum).
  lowering   floor minus surface_m under every located v0.3.1 iter1 epicentre, for each floor file, by region.
  stats      test (c): GMRT minus earth_relief by region and patch (at the GMRT nodes the merged surface uses and
             under the located events), and the size of the jumps along patch edges.
  nlloc-prep / nlloc-run / nlloc-analyse   test (b): the v5.0 events that sit above GMRT but below earth_relief plus
             random controls rerun with the v0.3.1 iter1 production setup (each event's workdir nlloc.in, only the
             LOCFILES obs/output paths and LOCTOPO_SURFACE changed), one run per floor configuration.
Usage (default env; B4_SURFACE_DIR = the test copy of the sources, or the production dir once installed):
  python b4_surface_check.py lookup --out DIR --floor NAME=PATH_eft.bin ...
  python b4_surface_check.py exact --out DIR --floor NAME=PATH_eft.bin ...
  python b4_surface_check.py lowering --out DIR --floor NAME=PATH_eft.bin ...
  python b4_surface_check.py stats --out DIR
  python b4_surface_check.py nlloc-prep --out DIR --floor NAME=PATH_eft.bin ... [--n-control 200] [--sub S --extra]
  python b4_surface_check.py nlloc-run --out DIR [--nproc 12] [--sub S]
  python b4_surface_check.py nlloc-analyse --out DIR [--sub S]
  (--extra: instead of the atlas list + controls, every other located v0.3.1 iter1 event above surface_m)
"""
import argparse
import ast
import json
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b4_surface as bs  # noqa: E402
from manifest import load_params  # noqa: E402

RE_M = bs.RE_M
RUNS = Path("/wd1/mdenolle_data/reloc_v5/runs")
ITER1 = "b4_iter1_v0.3.1"
PROD_FLOOR = Path("/wd1/mdenolle_data/reloc_v5/topo/topo_15s_cons_eft.bin")
GMRT_AIR = Path("/wd1/mdenolle_data/tomo_v1/above_surface/v5.0_atlas_gmrt_air_rows.json")
NLL = Path("/wd1/mdenolle_data/reloc_v5/src/NonLinLoc/src/bin/NLLoc")
LARGE_FLOAT = float(np.float32(1.0e20))   # velmod.h


# ------------------------------------------------------------------------------------------------ helpers
def regions():
    """REGIONS of a2_parse_hyp.py (read as a literal, not imported: the Track A tree is shared)."""
    src = Path(load_params()["catalog_versions"]["track_a_code"], "a2_parse_hyp.py").read_text()
    m = re.search(r"^REGIONS = (\[.*?^\])", src, re.S | re.M)
    return ast.literal_eval(re.sub(r"#[^\n]*", "", m.group(1)))


def region_of(lat, lon):
    lat, lon = np.asarray(lat, "f8"), np.asarray(lon, "f8")
    out = np.full(lat.shape, "Other onshore", dtype=object)
    done = np.zeros(lat.shape, bool)
    for name, a, b, c, d in regions():
        m = ~done & (lat >= a) & (lat < b) & (lon >= c) & (lon < d)
        out[m] = name
        done |= m
    return out


class NllSurface:
    """LOCTOPO_SURFACE as NLLoc reads it (velmod.c read_grd for a .bin/.hdr pair, get_surface_z)."""

    def __init__(self, path):
        self.path = Path(path)
        h = Path(f"{path}.hdr").read_text().splitlines()
        self.pix_shift = 0.5 if h[3].split()[1] == "Normal" else 0.0
        mx = re.search(r"x_min: (\S+) x_max: (\S+) x_inc: (\S+) \S+ (\S+) \S+ (\S+)", h[5])
        my = re.search(r"y_min: (\S+) y_max: (\S+) y_inc: (\S+) \S+ (\S+) \S+ (\S+)", h[6])
        self.x_min, self.x_inc, self.nx = float(mx.group(1)), float(mx.group(3)), int(mx.group(5))
        self.y_min, self.y_inc, self.ny = float(my.group(1)), float(my.group(3)), int(my.group(5))
        sf = float(re.search(r"scale_factor: (\S+)", h[8]).group(1))
        z = np.fromfile(path, "<f8")
        assert z.size == self.nx * self.ny
        self.z = (z * sf).astype(np.float32)          # *(ps->zdata) = zval * scale_factor + zshift (0), float
        self.eft = "_eft" in self.path.name

    def node(self, x, y):
        ix = np.trunc(self.pix_shift + (np.asarray(x, "f8") - self.x_min) / self.x_inc).astype(np.int64)
        iy = np.trunc(self.pix_shift + (np.asarray(y, "f8") - self.y_min) / self.y_inc).astype(np.int64)
        iy = self.ny - 1 - iy
        return ix, iy

    def lookup(self, x, y):
        """float64 of the float32 node value NLLoc compares with (LARGE_FLOAT outside the grid)."""
        ix, iy = self.node(x, y)
        out = np.full(ix.shape, LARGE_FLOAT)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        out[ok] = self.z[iy[ok] * self.nx + ix[ok]].astype("f8")
        return out

    def max_accepted_true(self, x, y):
        """Highest TRUE elevation (m) of a point NLLoc accepts at (x, y): elev' <= value; elev' -> true frame."""
        v = self.lookup(x, y)
        return RE_M * np.expm1(v / RE_M) if self.eft else v


def floors(specs):
    out = {}
    for s in specs:
        name, path = s.split("=", 1)
        out[name] = NllSurface(path)
    return out


def iter1_epicentres():
    loc = pd.read_csv(RUNS / ITER1 / "locations.csv")
    return loc[loc.status == "LOCATED"].reset_index(drop=True)


def qstats(v):
    v = np.asarray(v, "f8")
    if v.size == 0:
        return {"n": 0}
    q = np.percentile(v, [0, 1, 5, 10, 50, 90, 95, 99, 100])
    return {"n": int(v.size), "min": q[0], "p01": q[1], "p05": q[2], "p10": q[3], "median": q[4], "p90": q[5],
            "p95": q[6], "p99": q[7], "max": q[8], "mean": float(v.mean())}


def save(out, name, obj):
    p = Path(out) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(p, index=False, float_format="%.6g")
    else:
        p.write_text(json.dumps(obj, indent=1, default=lambda o: float(o) if np.isscalar(o) else str(o)))
    print("wrote", p)


# ------------------------------------------------------------------------------------------------ (a) lookup
def test_points(n_uniform=2_000_000, n_patch=1_000_000, n_cellb=1_200_000, n_lines=1_000_000, seed=20261124,
                grid=None):
    """Points for test (a). grid: an NllSurface whose node-cell boundaries are targeted."""
    rng = np.random.default_rng(seed)
    W, E, S, N = load_params()["b4"]["surface"]["earth_relief"]["region"]
    srcs = bs.sources()
    pts = {}
    pts["uniform"] = (rng.uniform(W, E, n_uniform), rng.uniform(S, N, n_uniform))
    pat = srcs[:-1]
    k = rng.integers(0, len(pat), n_patch)
    bx = np.array([s.box for s in pat])
    x = bx[k, 0] + rng.random(n_patch) * (bx[k, 1] - bx[k, 0])
    y = bx[k, 2] + rng.random(n_patch) * (bx[k, 3] - bx[k, 2])
    pts["in_patches"] = (x, y)
    if grid is not None:   # exactly on NLLoc's node-cell boundaries (in its arithmetic) and +-1 ulp
        m = n_cellb // 6
        kx = rng.integers(0, grid.nx - 1, m)
        ky = rng.integers(0, grid.ny - 1, m)
        bx = grid.x_min + (kx + 0.5) * grid.x_inc
        by = grid.y_min + (ky + 0.5) * grid.y_inc
        ry, rx = rng.uniform(S, N, m), rng.uniform(W, E, m)
        xs, ys = [], []
        for d in (-np.inf, 0, np.inf):
            bxd = bx if d == 0 else np.nextafter(bx, d)
            byd = by if d == 0 else np.nextafter(by, d)
            xs += [bxd, rx]
            ys += [ry, byd]
        # cell corners (both coordinates on boundaries)
        xs.append(bx)
        ys.append(by)
        pts["nlloc_cell_boundaries"] = (np.concatenate(xs), np.concatenate(ys))
    # on source grid lines (incl. patch edges and the first/last finite rows) and +-1 ulp
    xs, ys = [], []
    m = n_lines // (2 * 3 * len(srcs))
    for s in srcs:
        rows_ok = np.flatnonzero(s.cellok.any(1))
        cols_ok = np.flatnonzero(s.cellok.any(0))
        special_y = np.unique(np.r_[s.lat[[0, -1]], s.lat[rows_ok[[0, -1]]], s.lat[rows_ok[[0, -1]] + 1]])
        special_x = np.unique(np.r_[s.lon[[0, -1]], s.lon[cols_ok[[0, -1]]], s.lon[cols_ok[[0, -1]] + 1]])
        ly = np.where(rng.random(m) < 0.5, rng.choice(special_y, m), rng.choice(s.lat, m))
        lx = np.where(rng.random(m) < 0.5, rng.choice(special_x, m), rng.choice(s.lon, m))
        for d in (-np.inf, 0, np.inf):
            ly_d = ly if d == 0 else np.nextafter(ly, d)
            lx_d = lx if d == 0 else np.nextafter(lx, d)
            xs += [rng.uniform(s.lon[0] - 0.01, s.lon[-1] + 0.01, m), lx_d]
            ys += [ly_d, rng.uniform(s.lat[0] - 0.01, s.lat[-1] + 0.01, m)]
    x, y = np.concatenate(xs), np.concatenate(ys)
    keep = (x >= W) & (x <= E) & (y >= S) & (y <= N)
    pts["source_lines"] = (x[keep], y[keep])
    loc = iter1_epicentres()
    pts["iter1_epicentres"] = (loc.lon.values, loc.lat.values)
    return pts


def naive_floor(out, spacing_s, margin=1.0):
    """Negative control: surface_m sampled at the nodes (no cell minimum) - margin, flattened."""
    W, E, S, N = load_params()["b4"]["surface"]["earth_relief"]["region"]
    xn, _ = bs.node_axis(W, E, spacing_s)
    yn, _ = bs.node_axis(S, N, spacing_s)
    z = np.empty((yn.size, xn.size))
    for j0 in range(0, yn.size, 256):
        z[j0:j0 + 256] = bs.surface_m(xn[None, :], yn[j0:j0 + 256, None])
    p = Path(out) / "controls" / f"topo_{spacing_s:g}s_naive_merged_eft.bin"
    bs.write_grid(p, bs.eft(z - margin), (W, E, S, N), spacing_s / 3600, "NEGATIVE CONTROL: merged surface at nodes",
                  "flattened")
    return p


def cmd_lookup(a):
    fl = floors(a.floor)
    if a.controls:
        fl["CONTROL_prod_D-B056_earth_relief_only"] = NllSurface(PROD_FLOOR)
        fl["CONTROL_naive_15s_node_values"] = NllSurface(naive_floor(a.out, 15.0))
    res, rows = {}, []
    for fname, g in fl.items():
        pts = test_points(grid=g)
        res[fname] = {}
        for pname, (x, y) in pts.items():
            s = bs.surface_m(x, y)
            assert np.isfinite(s).all(), f"{pname}: point outside the surface region"
            h = g.max_accepted_true(x, y)
            gap = s - h               # >= 0: no accepted point above the surface
            nbad = int((gap < 0).sum())
            res[fname][pname] = {"n": int(x.size), "above_surface": nbad, "min_gap_m": float(gap.min()),
                                 "median_gap_m": float(np.median(gap)),
                                 "below_margin_minus_1mm": int((gap < bs.P["floor"]["margin_m"] - 1e-3).sum())}
            rows.append(dict(floor=fname, points=pname, **res[fname][pname]))
            print(fname, pname, res[fname][pname])
        tot = sum(v["n"] for v in res[fname].values())
        res[fname]["total_points"] = tot
        res[fname]["total_above"] = sum(v["above_surface"] for k, v in res[fname].items() if isinstance(v, dict))
    save(a.out, "lookup/lookup_results.json", res)
    save(a.out, "lookup/lookup_results.csv", pd.DataFrame(rows))


# ------------------------------------------------------------------------------------------------ exact (brute)
def brute_cell(ex0, ex1, ey0, ey1, eps=1e-9):
    srcs = bs.sources()
    X, Y = [np.array([ex0, ex1])], [np.array([ey0, ey1])]
    for s in srcs:
        X.append(s.lon[(s.lon >= ex0) & (s.lon <= ex1)])
        Y.append(s.lat[(s.lat >= ey0) & (s.lat <= ey1)])
    X, Y = np.unique(np.concatenate(X)), np.unique(np.concatenate(Y))
    off = np.array([-eps, 0.0, eps])
    xx = np.clip((X[:, None] + off).ravel(), ex0, ex1)
    yy = np.clip((Y[:, None] + off).ravel(), ey0, ey1)
    v = bs.surface_m(xx[None, :], yy[:, None])
    return float(np.nanmin(v)), xx.size * yy.size


def _brute_job(args):
    return [brute_cell(*c) for c in args]


def cmd_exact(a):
    rng = np.random.default_rng(7)
    W, E, S, N = bs.P["earth_relief"]["region"]
    srcs = bs.sources()
    margin = bs.P["floor"]["margin_m"]
    out = {}
    prod = NllSurface(PROD_FLOOR.with_name(PROD_FLOOR.name.replace("_eft", "")))   # true frame, earth_relief only
    for fname, g in floors(a.floor).items():
        true = NllSurface(g.path.with_name(g.path.name.replace("_eft", "")))
        sp = round(g.x_inc * 3600, 6)
        xn, xe = bs.node_axis(W, E, sp)
        yn, ye = bs.node_axis(S, N, sp)
        # sampled node cells: uniform, inside patches, straddling patch edges / finite-row edges
        i = list(rng.integers(0, xn.size, a.n))
        j = list(rng.integers(0, yn.size, a.n))
        for s in srcs[:-1]:
            m = a.n // len(srcs)
            i += list(np.searchsorted(xn, rng.uniform(s.lon[0], s.lon[-1], m)))
            j += list(np.searchsorted(yn, rng.uniform(s.lat[0], s.lat[-1], m)))
            rows_ok = np.flatnonzero(s.cellok.any(1))
            ey = np.r_[s.lat[[0, -1]], s.lat[rows_ok[0]], s.lat[rows_ok[-1] + 1]]
            ex = np.r_[s.lon[[0, -1]]]
            m2 = a.n // len(srcs)
            for _ in range(m2):
                if rng.random() < 0.5:
                    i.append(np.searchsorted(xn, rng.uniform(s.lon[0], s.lon[-1])))
                    j.append(int(np.argmin(np.abs(yn - rng.choice(ey)))))
                else:
                    i.append(int(np.argmin(np.abs(xn - rng.choice(ex)))))
                    j.append(np.searchsorted(yn, rng.uniform(s.lat[0], s.lat[-1])))
        i = np.clip(np.array(i), 0, xn.size - 1)
        j = np.clip(np.array(j), 0, yn.size - 1)
        cells = [(xe[p], xe[p + 1], ye[q], ye[q + 1]) for p, q in zip(i, j)]
        chunks = [cells[k::a.nproc] for k in range(a.nproc)]
        with ProcessPoolExecutor(a.nproc) as ex:
            res = list(ex.map(_brute_job, chunks))
        brute = np.empty(len(cells))
        npts = 0
        for k in range(a.nproc):
            for r, (bmin, n) in enumerate(res[k]):
                brute[k + r * a.nproc] = bmin
                npts += n
        fv = true.z[(true.ny - 1 - j) * true.nx + i].astype("f8") + margin     # floor + margin = cell minimum
        d = brute - fv
        out[fname] = {"cells": len(cells), "points_evaluated": npts,
                      "brute_min_minus_floor_cell_min_m": qstats(d),
                      "cells_where_brute_below_floor_by_1mm": int((d < -1e-3).sum())}
        # earth_relief-only cells vs the production D-B056 file (a2_make_topo_surface_cons.py, 1/8 lattice)
        if abs(sp - 15) < 1e-9:
            far = np.ones((yn.size, xn.size), bool)
            for s in srcs[:-1]:
                ci = (xe[1:] >= s.lon[0] - 1e-9) & (xe[:-1] <= s.lon[-1] + 1e-9)
                cj = (ye[1:] >= s.lat[0] - 1e-9) & (ye[:-1] <= s.lat[-1] + 1e-9)
                far[np.ix_(cj, ci)] = False
            dz = (true.z.reshape(true.ny, true.nx)[::-1][far].astype("f8")
                  - prod.z.reshape(prod.ny, prod.nx)[::-1][far].astype("f8"))
            out[fname]["vs_production_D-B056_on_earth_relief_only_cells"] = {
                "nodes": int(far.sum()), "max_abs_diff_m": float(np.abs(dz).max())}
        print(fname, json.dumps(out[fname], default=float))
    save(a.out, "exact/exact_results.json", out)


# ------------------------------------------------------------------------------------------------ lowering
def cmd_lowering(a):
    loc = iter1_epicentres()
    x, y = loc.lon.values, loc.lat.values
    s = bs.surface_m(x, y)
    er = bs.sources()[-1]
    s_er, _ = bs._eval([er], x, y)
    src = bs.surface_source(x, y)
    loc["gmrt_used"] = src != er.name
    loc["surface_m"] = s
    depth_true = RE_M / 1e3 * (1 - np.exp(-loc.depth.values / (RE_M / 1e3)))
    fl = floors(a.floor)
    fl["prod_D-B056"] = NllSurface(PROD_FLOOR)
    rows, out = [], {}
    for fname, g in fl.items():
        h = g.max_accepted_true(x, y)
        loc[f"low_{fname}"] = h - s
        if fname == "prod_D-B056":
            loc["low_prod_vs_er"] = h - s_er
    # events at the production floor (ML depth within 0.1 km of it): their depth is set by the floor
    hprod = fl["prod_D-B056"].max_accepted_true(x, y)
    loc["at_prod_floor"] = (depth_true + hprod / 1e3) < 0.1
    for fname in list(fl) + ["prod_vs_er"]:
        col = f"low_{fname}"
        for grp, d in [("all", loc), ("gmrt_used", loc[loc.gmrt_used]), ("at_prod_floor", loc[loc.at_prod_floor])] + \
                [(f"region:{r}", d) for r, d in loc.groupby("region")]:
            q = qstats(d[col])
            rows.append(dict(floor=fname, group=grp, frac_above_surface=float((d[col] > 0).mean()),
                             frac_lower_10m=float((d[col] < -10).mean()), frac_lower_50m=float((d[col] < -50).mean()),
                             frac_lower_100m=float((d[col] < -100).mean()), **q))
    df = pd.DataFrame(rows)
    save(a.out, "lowering/lowering_by_region.csv", df)
    keep = ["ID", "lat", "lon", "depth", "region", "tile", "gmrt_used", "surface_m", "at_prod_floor"] + \
        [c for c in loc if c.startswith("low_")]
    save(a.out, "lowering/lowering_events.csv", loc[keep])
    print(df[df.group.isin(["all", "gmrt_used", "at_prod_floor"])][
        ["floor", "group", "n", "median", "p10", "p01", "min", "frac_above_surface", "frac_lower_10m",
         "frac_lower_50m"]].to_string())


# ------------------------------------------------------------------------------------------------ (c) stats
def cmd_stats(a):
    srcs = bs.sources()
    er = srcs[-1]
    rows, edge_rows = [], []
    allv = []
    for k, s in enumerate(srcs[:-1]):
        LO, LA = np.meshgrid(s.lon, s.lat)
        ok = np.isfinite(s.z)
        x, y, zg = LO[ok], LA[ok], s.z[ok]
        _, used = bs._eval(srcs, x, y)
        u = used == k                                   # GMRT nodes the merged surface takes from this patch
        x, y, zg = x[u], y[u], zg[u]
        zer, _ = bs._eval([er], x, y)
        d = zg - zer
        reg = region_of(y, x)
        df = pd.DataFrame({"patch": s.name, "region": reg, "d": d, "zer": zer})
        allv.append(df)
        # jumps along this patch's edges: just inside vs just outside of its used area (eps = 1e-7 deg ~ 1 cm)
        eps = 1e-7
        rows_ok = np.flatnonzero(s.cellok.any(1))
        cols_ok = np.flatnonzero(s.cellok.any(0))
        ylo, yhi = s.lat[rows_ok[0]], s.lat[rows_ok[-1] + 1]
        xlo, xhi = s.lon[cols_ok[0]], s.lon[cols_ok[-1] + 1]
        n = 4000
        tx, ty = np.linspace(xlo, xhi, n), np.linspace(ylo, yhi, n)
        for side, xi, yi, xo, yo in (("south", tx, np.full(n, ylo + eps), tx, np.full(n, ylo - eps)),
                                     ("north", tx, np.full(n, yhi - eps), tx, np.full(n, yhi + eps)),
                                     ("west", np.full(n, xlo + eps), ty, np.full(n, xlo - eps), ty),
                                     ("east", np.full(n, xhi - eps), ty, np.full(n, xhi + eps), ty)):
            vi, ki = bs._eval(srcs, xi, yi)
            vo, ko = bs._eval(srcs, xo, yo)
            m = (ki == k) & (ko != k) & np.isfinite(vo)
            if m.any():
                j = vo[m] - vi[m]
                edge_rows.append(dict(patch=s.name, edge=side, n=int(m.sum()), median_abs_jump_m=np.median(np.abs(j)),
                                      p95_abs_jump_m=np.percentile(np.abs(j), 95), max_abs_jump_m=np.abs(j).max(),
                                      outside_lower_frac=float((j < 0).mean())))
    df = pd.concat(allv, ignore_index=True)

    def agg(g):
        d = g.d.values
        return pd.Series({"n_nodes": d.size, "median_m": np.median(d), "mean_m": d.mean(),
                          "p05_m": np.percentile(d, 5), "p95_m": np.percentile(d, 95), "min_m": d.min(),
                          "max_m": d.max(), "frac_gmrt_lower": (d < 0).mean(), "frac_lower_50m": (d < -50).mean(),
                          "median_abs_m": np.median(np.abs(d)), "seafloor_frac": (g.zer < 0).mean()})
    by_region = df.groupby("region").apply(agg, include_groups=False).reset_index()
    by_patch = df.groupby("patch").apply(agg, include_groups=False).reset_index()
    save(a.out, "stats/gmrt_minus_er_by_region.csv", by_region)
    save(a.out, "stats/gmrt_minus_er_by_patch.csv", by_patch)
    save(a.out, "stats/patch_edge_jumps.csv", pd.DataFrame(edge_rows))
    # under the located v0.3.1 iter1 epicentres
    loc = iter1_epicentres()
    x, y = loc.lon.values, loc.lat.values
    s_m, used = bs._eval(srcs, x, y)
    zer, _ = bs._eval([er], x, y)
    loc["d"] = s_m - zer
    loc["gmrt"] = used < len(srcs) - 1
    ev = loc[loc.gmrt].groupby("region").d.agg(
        n="size", median="median", p05=lambda v: np.percentile(v, 5), p95=lambda v: np.percentile(v, 95),
        min="min", max="max", frac_gmrt_lower=lambda v: (v < 0).mean()).reset_index()
    dg = loc.d[loc.gmrt]
    ev = pd.concat([ev, pd.DataFrame([dict(region="ALL", n=int(dg.size), median=dg.median(), p05=dg.quantile(0.05),
                                           p95=dg.quantile(0.95), min=dg.min(), max=dg.max(),
                                           frac_gmrt_lower=float((dg < 0).mean()))])])
    save(a.out, "stats/gmrt_minus_er_under_iter1_events.csv", ev)
    print(by_region.round(3).to_string())
    print(by_patch.round(3).to_string())
    print(pd.DataFrame(edge_rows).round(2).to_string())
    print(ev.round(3).to_string())


# ------------------------------------------------------------------------------------------------ (b) NLLoc
def obs_blocks(path):
    blocks = [b for b in Path(path).read_text().split("\n\n") if b.strip()]
    out = {}
    for b in blocks:
        m = re.search(r"PUBLIC_ID (\S+)", b)
        out[int(m.group(1).split("_")[1])] = b
    return blocks, out


def cmd_nlloc_prep(a):
    out = Path(a.out) / a.sub
    d = json.loads(GMRT_AIR.read_text())
    rows = pd.DataFrame(d["rows"], columns=["ID", "tier_code", "above_gmrt_km", "gmrt_km"])
    sel = rows[rows.tier_code < 4]                      # located (tiers A-D); tier U rows are v4 positions
    loc = pd.read_csv(RUNS / ITER1 / "locations.csv")
    loc = loc[loc.status == "LOCATED"]
    assert len(sel) == 725 and sel.ID.isin(loc.ID).all()
    rng = np.random.default_rng(20261124)
    ctrl = rng.choice(loc.ID[~loc.ID.isin(sel.ID)].values, a.n_control, replace=False)
    ev = pd.concat([pd.DataFrame({"ID": sel.ID, "group": "gmrt_air"}), pd.DataFrame({"ID": ctrl, "group": "control"})])
    if a.extra:   # every other located iter1 event above surface_m (the atlas list used a downsampled GMRT)
        zt = RE_M / 1e3 * (1 - np.exp(-loc.depth.values / (RE_M / 1e3)))
        above = loc.ID[(zt < -bs.surface_m(loc.lon.values, loc.lat.values) / 1e3) & ~loc.ID.isin(ev.ID)]
        ev = pd.DataFrame({"ID": above.values, "group": "merged_air_extra"})
    ev = ev.merge(loc[["ID", "tile"]], on="ID")
    configs = dict(s.split("=", 1) for s in a.floor)
    configs = {"prod": str(PROD_FLOOR), **configs}
    for c, fpath in configs.items():
        assert Path(fpath).exists() and Path(f"{fpath}.hdr").exists(), fpath
    jobs = []
    for wd, g in ev.groupby("tile"):
        src = RUNS / ITER1 / wd
        blocks, by_id = obs_blocks(src / "events.obs")
        want = set(g.ID)
        keep = [b for b in blocks if int(re.search(r"PUBLIC_ID (\S+)", b).group(1).split("_")[1]) in want]
        assert len(keep) == len(want), wd
        ctl = (src / "nlloc.in").read_text()
        for c, fpath in configs.items():
            w = out / c / wd
            w.mkdir(parents=True, exist_ok=True)
            (w / "events.obs").write_text("\n\n".join(keep) + "\n\n")
            lines = []
            n_files = n_topo = 0
            for line in ctl.splitlines():
                t = line.split()
                if t and t[0] == "LOCFILES":
                    assert len(t) == 5, line
                    line = f"LOCFILES {w / 'events.obs'} {t[2]} {t[3]} {w / 'loc'}"
                    n_files += 1
                elif t and t[0] == "LOCTOPO_SURFACE":
                    line = f"LOCTOPO_SURFACE {fpath} {t[2]}"
                    n_topo += 1
                lines.append(line)
            assert n_files == 1 and n_topo == 1, wd
            (w / "nlloc.in").write_text("\n".join(lines) + "\n")
            jobs.append(str(w))
    ev.to_csv(out / "events.csv", index=False)
    (out / "configs.json").write_text(json.dumps(configs, indent=1))
    (out / "jobs.txt").write_text("\n".join(jobs) + "\n")
    print(len(ev), "events", ev.group.value_counts().to_dict(), ev.tile.nunique(), "workdirs x", len(configs),
          "configs =", len(jobs), "jobs")


def _run(w):
    w = Path(w)
    if (w / "loc.sum.grid0.loc.hyp").exists():
        return w.name, 0
    with open(w / "nlloc.log", "w") as log:
        r = subprocess.run(["nice", "-n", "10", "/usr/bin/time", "-f", "TIME %e s %M kB", "-o", "time.log", str(NLL),
                            "nlloc.in"], cwd=w, stdout=log, stderr=subprocess.STDOUT)
    return str(w), r.returncode


def cmd_nlloc_run(a):
    jobs = (Path(a.out) / a.sub / "jobs.txt").read_text().split()
    bad = []
    with ProcessPoolExecutor(a.nproc) as ex:
        for w, rc in ex.map(_run, jobs):
            if rc:
                bad.append(w)
    print(len(jobs), "jobs,", len(bad), "failed", bad[:5])


def parse_hyp(path):
    rows, cur = [], None
    for line in open(path):
        t = line.split()
        if not t:
            continue
        k = t[0]
        if k == "NLLOC":
            cur = {"status": t[2].strip('"')}
        elif cur is None:
            continue
        elif k == "PUBLIC_ID":
            cur["ID"] = int(t[1].split("_")[1])
        elif k == "SEARCH":
            cur["ns_z"] = float(t[t.index("smallestNodeSide") + 1].split("/")[2])
        elif k == "GEOGRAPHIC":
            d = dict(zip(t[8::2], t[9::2]))
            cur.update(lat=float(d["Lat"]), lon=float(d["Long"]), zf=float(d["Depth"]))
        elif k == "QUALITY":
            d = dict(zip(t[1::2], t[2::2]))
            cur.update(rms=float(d["RMS"]), nphs=int(d["Nphs"]), gap=float(d["Gap"]))
        elif k == "STATISTICS":
            d = dict(zip(t[1::2], t[2::2]))
            cur.update(sig_z=float(np.sqrt(float(d["ZZ"]))))
        elif k == "END_NLLOC":
            rows.append(cur)
            cur = None
    return rows


def run_table(root, wds=None):
    rows = []
    files = sorted(Path(root).glob("*/loc.sum.grid0.loc.hyp")) if wds is None else \
        [Path(root) / w / "loc.sum.grid0.loc.hyp" for w in sorted(set(wds))]
    for f in files:
        for r in parse_hyp(f):
            r["wd"] = f.parent.name
            rows.append(r)
    return pd.DataFrame(rows)


def km_h(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2 +
                                          np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))


def cmd_nlloc_analyse(a):
    out = Path(a.out) / a.sub
    ev = pd.read_csv(out / "events.csv")
    configs = json.loads((out / "configs.json").read_text())
    R = RE_M / 1e3
    tabs = {"production": run_table(RUNS / ITER1, ev.tile)}
    tabs["production"] = tabs["production"][tabs["production"].ID.isin(ev.ID)]
    flo = {"production": NllSurface(PROD_FLOOR), **{c: NllSurface(f) for c, f in configs.items()}}
    for c in configs:
        tabs[c] = run_table(out / c)
    allt = []
    for c, t in tabs.items():
        t = t.merge(ev, on="ID", how="right")
        t["config"] = c
        t["depth_true"] = R * (1 - np.exp(-t.zf / R))
        t["surface_m"] = bs.surface_m(t.lon.values, t.lat.values)
        t["above_surface_m"] = -t.depth_true * 1e3 - t.surface_m        # > 0: above the merged surface
        t["dist_to_surface_km"] = t.depth_true + t.surface_m / 1e3
        t["floor_true_m"] = flo[c].max_accepted_true(t.lon.values, t.lat.values)
        t["dist_to_floor_km"] = t.depth_true + t.floor_true_m / 1e3
        t["at_floor"] = t.dist_to_floor_km < t.ns_z                     # ML within one smallest oct-tree cell
        allt.append(t)
    T = pd.concat(allt, ignore_index=True)
    save(a.out, f"{a.sub}/results_long.csv", T)
    base = T[T.config == "prod"].set_index("ID")
    summ = []
    for c in ["production"] + list(configs):
        t = T[T.config == c].set_index("ID")
        for grp in sorted(ev.group.unique()):
            g = t[t.group == grp]
            located = g.status == "LOCATED"
            gl = g[located]
            b = base.loc[gl.index]
            dz = (gl.depth_true - b.depth_true)
            dh = km_h(gl.lat, gl.lon, b.lat, b.lon)
            summ.append(dict(config=c, group=grp, n=len(g), located=int(located.sum()),
                             above_merged_surface=int((gl.above_surface_m > 0).sum()),
                             max_above_m=float(gl.above_surface_m.max()),
                             at_floor_ns_z=int(gl.at_floor.sum()),
                             dz_vs_prod_median_km=float(dz.median()), absdz_vs_prod_median_km=float(dz.abs().median()),
                             absdz_vs_prod_p90_km=float(dz.abs().quantile(0.9)),
                             dz_vs_prod_max_km=float(dz.abs().max()),
                             n_dz_gt_0p1km=int((dz.abs() > 0.1).sum()), n_identical=int(((dz.abs() < 1e-6) & (dh < 1e-6)).sum()),
                             dh_vs_prod_median_km=float(dh.median()), dh_vs_prod_p90_km=float(dh.quantile(0.9)),
                             dh_vs_prod_max_km=float(dh.max()),
                             rms_median_s=float(gl.rms.median()), drms_vs_prod_median_s=float((gl.rms - b.rms).median()),
                             drms_vs_prod_p90_s=float((gl.rms - b.rms).quantile(0.9)),
                             depth_median_km=float(gl.depth_true.median()),
                             below_surface_median_km=float(gl.dist_to_surface_km.median())))
    S = pd.DataFrame(summ)
    save(a.out, f"{a.sub}/summary.csv", S)
    print(S.T.to_string())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("lookup", "exact", "lowering", "stats", "nlloc-prep", "nlloc-run", "nlloc-analyse"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True)
        s.add_argument("--floor", nargs="*", default=[])
        s.add_argument("--nproc", type=int, default=12)
        s.add_argument("--n", type=int, default=3000)
        s.add_argument("--n-control", type=int, default=200)
        s.add_argument("--controls", action="store_true")
        s.add_argument("--sub", default="nlloc", help="nlloc-*: subfolder of --out")
        s.add_argument("--extra", action="store_true", help="nlloc-prep: only the located iter1 events above "
                       "surface_m that are not in the atlas list or the controls")
    a = ap.parse_args()
    assert str(Path(a.out).resolve()).startswith("/wd1/mdenolle_data/tomo_v1/v51_build/"), "--out outside v51_build"
    {"lookup": cmd_lookup, "exact": cmd_exact, "lowering": cmd_lowering, "stats": cmd_stats,
     "nlloc-prep": cmd_nlloc_prep, "nlloc-run": cmd_nlloc_run, "nlloc-analyse": cmd_nlloc_analyse}[a.cmd](a)


if __name__ == "__main__":
    main()

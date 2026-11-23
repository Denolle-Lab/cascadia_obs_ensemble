"""Fault surfaces v0 from catalog v4: plane-based detection, rectangular faults.

A first v0 (fault_surfaces_v0.py, kept for its helpers) found clusters by density (DBSCAN)
and kept only the densest knots (20% of events). This version looks for planes instead: the criterion is that events gather around planes and that
most events can be assigned to a fault, whatever the local density. Still a method test on
the v4 locations (shallow-biased depths, unconfirmed datum, no relative relocation).

Detection (sequential local RANSAC with growth)
  * Events: QC (>= 4 P and >= 4 S), catalog depth >= 0. No spatial_sigma cut. Axial Seamount
    is left out (--exclude AX): published fault maps there are much better than v4 allows.
  * Planes flatter than --min-dip (30 deg) are rejected: with v4 depths a diffuse cloud passes
    for a flat plane, and the target is faults.
  * A plane's thickness tolerance follows the location error along its normal:
        tol = k * sqrt((s_h * |n_h|)^2 + (s_z * n_z)^2)
    with nominal relative errors s_h = --tol-h, s_z = --tol-z (depth worse than epicentre),
    so vertical faults are thin and flat ones thicker.
  * Seeds are taken from dense to sparse (random jitter). Around a seed, --trials random
    planes through the seed and two events within 10 km are scored by the number of free
    events within tol in a --ball km ball. The best plane then grows: free events within tol
    of the plane, kept only if connected to the seed in the plane (cells of --gap km,
    8-connectivity), refit, repeat. A plane is kept with >= --n-min events, planarity
    sqrt(l3/l2) <= --planarity and an in-plane width >= 1 km; its events are removed.
  * Reassignment (k-planes): every event goes to the plane with the smallest |d|/tol among
    planes whose rectangle (2-98% of u and v, plus --pad km) contains its projection;
    refit; three passes. Planes left with < n-min events are dropped.
Surfaces
  * Rectangles: each fault is the rectangle [u2, u98] x [v2, v98] (+ --pad km) in its own
    plane (u along strike, v down dip), whether or not it reaches the surface; w = f(u, v)
    plane or quadratic by BIC. "constrained" flags vertices with >= 5 events within 2.5 km.
  * Uncertainty as in v0 (bootstrap, location Monte Carlo, model choice, combined), but the
    location Monte Carlo uses the nominal relative errors (--tol-h, --tol-z), since
    spatial_sigma behaves like an absolute error. Stability: detection rerun with two other
    random seeds and k -/+ 0.25; median Jaccard overlap with the best-matching plane.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fault_surfaces_v0 as v0  # noqa: E402

D = v0.D


def tol_of(n, a, k=None):
    k = a.tol_k if k is None else k
    return k * np.sqrt((a.tol_h * np.hypot(n[0], n[1])) ** 2 + (a.tol_z * n[2]) ** 2)


def frame_of(n):
    n = v0.up_normal(n)
    strike, _ = v0.strike_dip(n)
    s = np.radians(strike)
    eu = np.array([np.sin(s), np.cos(s), 0.0])
    ev = np.cross(n, eu)
    return n, eu, (ev if ev[2] >= 0 else -ev)


def connected_to(P, c0, n, seed_pt, gap):
    """Boolean mask of points connected (in-plane cells of size gap, 8-connectivity) to the seed's cell."""
    n, eu, ev = frame_of(n)
    Q = P - c0
    u, v = Q @ eu, Q @ ev
    su, sv = (seed_pt - c0) @ eu, (seed_pt - c0) @ ev
    u0, v0_ = min(u.min(), su), min(v.min(), sv)
    iu, iv = ((u - u0) / gap).astype(int), ((v - v0_) / gap).astype(int)
    G = np.zeros((iu.max() + 1, iv.max() + 1), bool)
    G[iu, iv] = True
    siu, siv = int((su - u0) / gap), int((sv - v0_) / gap)
    siu, siv = min(siu, G.shape[0] - 1), min(siv, G.shape[1] - 1)
    lab, _ = ndimage.label(G, structure=np.ones((3, 3)))
    L = lab[siu, siv]
    if L == 0:                                        # seed cell empty: take the largest component
        L = np.bincount(lab[G]).argmax() if G.any() else 0
    return lab[iu, iv] == L


def detect(P, a, rng, k=None):
    tree = cKDTree(P)
    N = len(P)
    free = np.ones(N, bool)
    tried = np.zeros(N, bool)
    nb = tree.query_ball_point(P, 5.0, return_length=True)
    order = np.argsort(-(nb * rng.uniform(0.8, 1.25, N)))
    planes = []
    for s in order:
        if not free[s] or tried[s] or nb[s] < a.min_seed_nb:
            continue
        ball = np.array(tree.query_ball_point(P[s], a.ball))
        ball = ball[free[ball]]
        near = ball[np.linalg.norm(P[ball] - P[s], axis=1) <= 10.0]
        if len(ball) < a.n_min or len(near) < 3:
            tried[s] = True
            continue
        best = (0, None)
        Pb = P[ball] - P[s]
        for _ in range(a.trials):
            j, kk = rng.choice(near, 2, replace=False)
            nv = np.cross(P[j] - P[s], P[kk] - P[s])
            L = np.linalg.norm(nv)
            if L < 1.0:
                continue
            nv /= L
            cnt = int((np.abs(Pb @ nv) <= tol_of(nv, a, k)).sum())
            if cnt > best[0]:
                best = (cnt, nv)
        if best[0] < a.n_min:
            tried[s] = True
            tried[tree.query_ball_point(P[s], 1.0)] = True
            continue
        nv, c0, members, ext = best[1], P[s].copy(), None, a.ball
        for _ in range(8):
            R = min(a.max_reach, max(a.ball, 0.75 * ext + a.ball))
            cand = np.array(tree.query_ball_point(c0, R))
            cand = cand[free[cand]]
            cand = cand[np.abs((P[cand] - c0) @ nv) <= tol_of(nv, a, k)]
            if len(cand) < a.n_min:
                break
            cand = cand[connected_to(P[cand], c0, nv, P[s], a.gap)]
            if len(cand) < a.n_min:
                break
            c0, var, V = v0.pca(P[cand])
            nv = V[:, 2]
            ext = 2 * 1.96 * np.sqrt(var[0])
            if members is not None and len(cand) == len(members) and (np.sort(cand) == np.sort(members)).all():
                break
            members = cand
        if members is None or len(members) < a.n_min:
            tried[s] = True
            continue
        _, var, V = v0.pca(P[members])
        if (np.sqrt(var[2] / var[1]) > a.planarity or np.sqrt(var[1]) < 1.0
                or v0.strike_dip(V[:, 2])[1] < a.min_dip):
            tried[s] = True
            continue
        free[members] = False
        planes.append(members)
    return planes


def rect_of(P, members, a):
    c0, _, V = v0.pca(P[members])
    n, eu, ev = frame_of(V[:, 2])
    Q = P[members] - c0
    u, v = Q @ eu, Q @ ev
    return c0, n, eu, ev, (np.percentile(u, 2) - a.pad, np.percentile(u, 98) + a.pad,
                           np.percentile(v, 2) - a.pad, np.percentile(v, 98) + a.pad)


def reassign(P, planes, a, k=None):
    tree = cKDTree(P)
    for _ in range(3):
        best = np.full(len(P), np.inf)
        owner = -np.ones(len(P), int)
        for i, m in enumerate(planes):
            if len(m) < 3:
                continue
            c0, n, eu, ev, (u0, u1, v0_, v1) = rect_of(P, m, a)
            R = np.hypot(max(abs(u0), abs(u1)), max(abs(v0_), abs(v1))) + 1.0
            cand = np.array(tree.query_ball_point(c0, R))
            Q = P[cand] - c0
            u, v, w = Q @ eu, Q @ ev, Q @ n
            r = np.abs(w) / tol_of(n, a, k)
            ok = (r <= 1.0) & (u >= u0) & (u <= u1) & (v >= v0_) & (v <= v1)
            cand, r = cand[ok], r[ok]
            better = r < best[cand]
            best[cand[better]] = r[better]
            owner[cand[better]] = i
        planes = [np.flatnonzero(owner == i) for i in range(len(planes))]
        planes = [m for m in planes if len(m) >= a.n_min]
    return planes


def merge(P, planes, a, k=None):
    """Join planes with normals within --merge-angle, each within tol of the other, events touching
    (closest pair < --gap) and a joined set that is still planar with RMS normal residual <= tol."""
    planes = list(planes)
    changed = True
    while changed:
        changed = False
        info = []
        for m in planes:
            c0, var, V = v0.pca(P[m])
            n = v0.up_normal(V[:, 2])
            info.append((c0, n, 1.96 * np.sqrt(var[0]), tol_of(n, a, k)))
        order = np.argsort([-len(m) for m in planes])
        for ii in order:
            ci, ni, li, ti = info[ii]
            for jj in order:
                if jj == ii:
                    continue
                cj, nj, lj, tj = info[jj]
                if np.linalg.norm(ci - cj) > li + lj + a.gap:
                    continue
                if np.degrees(np.arccos(min(1.0, abs(ni @ nj)))) > a.merge_angle:
                    continue
                if abs((cj - ci) @ ni) > ti or abs((ci - cj) @ nj) > tj:
                    continue
                small, big = (planes[ii], planes[jj]) if len(planes[ii]) < len(planes[jj]) else (planes[jj], planes[ii])
                if cKDTree(P[big]).query(P[small], distance_upper_bound=a.gap)[0].min() > a.gap:
                    continue
                u = np.union1d(planes[ii], planes[jj])
                c0, var, V = v0.pca(P[u])
                n = v0.up_normal(V[:, 2])
                rms = np.sqrt(np.mean(((P[u] - c0) @ n) ** 2))
                if np.sqrt(var[2] / var[1]) > a.planarity or rms > tol_of(n, a, k) or v0.strike_dip(n)[1] < a.min_dip:
                    continue
                planes[ii] = u
                planes.pop(jj)
                changed = True
                break
            if changed:
                break
    return planes


def surface_rect(c, idx, a, cfmk, slabf, reliefk, rng):
    lon, lat, dep = (c[k].values[idx] for k in ("longitude", "latitude", "z"))
    lon0, lat0 = np.median(lon), np.median(lat)
    kx = v0.KM * np.cos(np.radians(lat0))
    P = np.c_[(lon - lon0) * kx, (lat - lat0) * v0.KM, dep]
    c0, var, V = v0.pca(P)
    planarity = float(np.sqrt(var[2] / var[1]))
    n, eu, ev = frame_of(V[:, 2])
    strike, dip = v0.strike_dip(n)
    Q = P - c0
    u, v, w = Q @ eu, Q @ ev, Q @ n
    quad = v0.bic(u, v, w, True) < v0.bic(u, v, w, False) - 10
    u0, u1 = np.percentile(u, 2) - a.pad, np.percentile(u, 98) + a.pad
    v0_, v1 = np.percentile(v, 2) - a.pad, np.percentile(v, 98) + a.pad
    nu, nv_ = max(2, int(np.ceil((u1 - u0) / a.grid)) + 1), max(2, int(np.ceil((v1 - v0_) / a.grid)) + 1)
    GU, GV = np.meshgrid(np.linspace(u0, u1, nu), np.linspace(v0_, v1, nv_))
    vu, vv = GU.ravel(), GV.ravel()
    coef = v0.fit_w(u, v, w, quad)
    vw = v0.design(vu, vv, quad) @ coef
    constrained = np.array([len(x) for x in cKDTree(np.c_[u, v]).query_ball_point(np.c_[vu, vv], 2.5)]) >= 5
    rms = float(np.sqrt(np.mean((w - v0.design(u, v, quad) @ coef) ** 2)))
    sig = np.array([a.tol_h, a.tol_h, a.tol_z])

    def draw(boot, perturb):
        W, S, Dp = [], [], []
        for _ in range(a.draws):
            j = rng.integers(0, len(idx), len(idx)) if boot else np.arange(len(idx))
            Pj = P[j] + (rng.normal(size=(len(j), 3)) * sig if perturb else 0.0)
            Qj = Pj - c0
            W.append(v0.design(vu, vv, quad) @ v0.fit_w(Qj @ eu, Qj @ ev, Qj @ n, quad) - vw)
            sj, dj = v0.strike_dip(v0.pca(Pj)[2][:, 2])
            S.append(sj); Dp.append(dj)
        return np.array(W), np.array(S), np.array(Dp)
    Wb, Sb, Db = draw(True, False)
    Wl, _, _ = draw(False, True)
    Wt, St, Dt = draw(True, True)
    variants = [v0.design(vu, vv, q) @ v0.fit_w(u, v, w, q, t) for q in (False, True) for t in (False, True)]
    sb, sl, sm = Wb.std(0), Wl.std(0), np.std(variants, axis=0)
    st = np.sqrt(sb ** 2 + sl ** 2 + sm ** 2)
    s_int, d_int = v0.interval_stats(Sb, Db)
    s_mc, d_mc = v0.interval_stats(St, Dt)
    X = c0 + np.outer(vu, eu) + np.outer(vv, ev) + np.outer(vw, n)
    vlon, vlat = X[:, 0] / kx + lon0, X[:, 1] / v0.KM + lat0
    tris = []
    for i in range(nv_ - 1):
        for k in range(nu - 1):
            q = i * nu + k, i * nu + k + 1, (i + 1) * nu + k + 1, (i + 1) * nu + k
            tris += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
    clon, clat = c0[0] / kx + lon0, c0[1] / v0.KM + lat0
    dist, j = cfmk[0].query(v0.ecef([clon], [clat], [c0[2]])[0])
    fp = cfmk[2][cfmk[1][j]]
    sd = float(slabf([clon], [clat])[0] - c0[2])
    r1 = lambda x: round(float(x), 1)
    reg = pd.Series(c.region.values[idx]).mode()[0]
    top, bot = float(X[:, 2].min()), float(X[:, 2].max())
    return dict(
        idx=idx, lon=vlon, lat=vlat, dep=X[:, 2], tris=np.array(tris), constrained=constrained,
        sb=sb, sl=sl, sm=sm, st=st, Wt=Wt, normal=n, surface_elev_km=reliefk(vlon, vlat),
        props=dict(
            region=reg, n_events=int(len(idx)), lon=round(float(clon), 4), lat=round(float(clat), 4),
            depth_km=round(float(c0[2]), 2), top_km=r1(top), bottom_km=r1(bot),
            strike=r1(strike), dip=r1(dip),
            strike_68=[r1(s_int[1]), r1(s_int[3])], strike_95=[r1(s_int[0]), r1(s_int[4])],
            dip_68=[r1(d_int[1]), r1(d_int[3])], dip_95=[r1(d_int[0]), r1(d_int[4])],
            strike_68_with_loc_mc=[r1(s_mc[1]), r1(s_mc[3])], dip_68_with_loc_mc=[r1(d_mc[1]), r1(d_mc[3])],
            model="quadratic" if quad else "plane", planarity=round(planarity, 3),
            length_km=r1(u1 - u0), width_km=r1(v1 - v0_),
            rms_normal_km=round(rms, 2), tolerance_km=round(float(tol_of(n, a)), 2),
            sigma_total_median_km=round(float(np.median(st)), 2), sigma_boot_median_km=round(float(np.median(sb)), 2),
            sigma_loc_median_km=round(float(np.median(sl)), 2), sigma_method_median_km=round(float(np.median(sm)), 2),
            constrained_fraction=round(float(constrained.mean()), 2), n_vertices=int(len(vu)), n_triangles=int(len(tris)),
            nearest_cfm=fp.get("name"), nearest_cfm_km=r1(dist), nearest_cfm_dip=fp.get("dip"),
            slab2_minus_depth_km=r1(sd) if np.isfinite(sd) else None,
        ))


def run_detection(c, a, seed, k=None):
    P = np.c_[c.x, c.y, c.z]
    rng = np.random.default_rng(seed)
    planes = reassign(P, detect(P, a, rng, k), a, k)
    return reassign(P, merge(P, planes, a, k), a, k)


def stability(c, surfaces, a):
    runs = [run_detection(c, a, 101), run_detection(c, a, 202),
            run_detection(c, a, 0, a.tol_k - 0.25), run_detection(c, a, 0, a.tol_k + 0.25)]
    labs = []
    for planes in runs:
        lab = -np.ones(len(c), int)
        for i, m in enumerate(planes):
            lab[m] = i
        labs.append(lab)
    P = np.c_[c.x, c.y, c.z]
    for s in surfaces:
        A = s["idx"]; J, S, Dp = [], [], []
        for lab in labs:
            la = lab[A]; la = la[la >= 0]
            if len(la) == 0:
                J.append(0.0); continue
            B = np.flatnonzero(lab == np.bincount(la).argmax())
            J.append(len(np.intersect1d(A, B)) / len(np.union1d(A, B)))
            st_, dp = v0.strike_dip(v0.pca(P[B])[2][:, 2]); S.append(st_); Dp.append(dp)
        p = s["props"]
        p["cluster_jaccard_median"] = round(float(np.median(J)), 2)
        si, di = v0.interval_stats(np.array(S), np.array(Dp)) if S else (None, None)
        if S:
            p["strike_settings_range"] = [round(float(si[0]), 1), round(float(si[4]), 1)]
            p["dip_settings_range"] = [round(float(di[0]), 1), round(float(di[4]), 1)]
        dz = p["slab2_minus_depth_km"]
        p["kind"] = ("steep zone (fault candidate)" if p["dip"] >= 45 else
                     "near-slab layer (not a single fault)" if dz is not None and abs(dz) <= 5 and p["dip"] <= 25
                     else "dipping zone")
        p["stable"] = bool(p["cluster_jaccard_median"] >= 0.5)
    return [len(r) for r in runs]


def write(c, surfaces, out, tag):
    out.mkdir(parents=True, exist_ok=True)
    feats, rows, vrows, erows, mrows, count = [], [], [], [], [], {}
    for s in surfaces:
        r = s["props"]["region"]; count[r] = count.get(r, 0) + 1
        sid = f"{r}{count[r]:03d}"
        s["props"]["surface_id"] = sid
        p = dict(version=tag, catalog="Cascadia catalog v4", **s["props"])
        p["normal_xyz"] = [round(float(x), 4) for x in s["normal"]]
        tri = [[[[round(float(s["lon"][i]), 5), round(float(s["lat"][i]), 5), round(float(-s["dep"][i] * 1000), 1)]
                 for i in (t0, t1, t2, t0)]] for t0, t1, t2 in s["tris"]]
        feats.append(dict(type="Feature", properties=p, geometry=dict(type="MultiPolygon", coordinates=tri)))
        rows.append(p)
        el = s["surface_elev_km"]
        for i in range(len(s["lon"])):
            vrows.append((sid, i, round(float(s["lon"][i]), 5), round(float(s["lat"][i]), 5), round(float(s["dep"][i]), 3),
                          round(float(s["sb"][i]), 3), round(float(s["sl"][i]), 3), round(float(s["sm"][i]), 3),
                          round(float(s["st"][i]), 3), int(s["constrained"][i]),
                          round(float(el[i]), 3) if np.isfinite(el[i]) else np.nan))
        W = np.round(s["Wt"], 3)
        erows.append(pd.DataFrame({"surface_id": sid, "vertex": np.tile(np.arange(W.shape[1]), W.shape[0]),
                                   "draw": np.repeat(np.arange(W.shape[0]), W.shape[1]), "normal_offset_km": W.ravel()}))
        mrows += [(int(e), sid) for e in c.event_id.values[s["idx"]]]
    stem = f"fault_surfaces_{tag.replace('.', '')}"
    (out / f"{stem}.geojson").write_text(json.dumps(dict(
        type="FeatureCollection", name=stem, crs=dict(type="name", properties=dict(name="urn:ogc:def:crs:OGC:1.3:CRS84")),
        features=feats)))
    summ = pd.DataFrame(rows)
    for col in summ.columns:
        if summ[col].map(lambda v: isinstance(v, list)).any():
            summ[col] = summ[col].map(lambda v: ";".join(map(str, v)) if isinstance(v, list) else v)
    summ.to_csv(out / f"{stem}_summary.csv", index=False)
    pd.DataFrame(vrows, columns=["surface_id", "vertex", "lon", "lat", "depth_km", "sigma_boot_km", "sigma_loc_km",
                                 "sigma_method_km", "sigma_total_km", "constrained", "surface_elev_km"]
                 ).to_csv(out / f"{stem}_vertices.csv", index=False)
    pd.concat(erows).to_csv(out / f"{stem}_ensemble.csv.gz", index=False, compression="gzip")
    pd.DataFrame(mrows, columns=["event_id", "surface_id"]).to_csv(out / f"{stem}_members.csv", index=False)
    return summ, stem


def figure(c, surfaces, out, stem, assigned):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    views = [("Whole margin", [-131, -119.5, 39, 51]), ("Blanco transform", [-130.6, -126.6, 42.6, 45.3]),
             ("Mendocino and Gorda", [-128.0, -123.2, 39.8, 42.5])]
    fig, axs = plt.subplots(2, 3, figsize=(17, 11), constrained_layout=True)
    rng = np.random.default_rng(3)
    cols = {}
    for s in surfaces:
        cols[s["props"]["surface_id"]] = plt.cm.tab20(rng.integers(0, 20))
    for col, (title, b) in enumerate(views):
        sel = c.longitude.between(b[0], b[1]) & c.latitude.between(b[2], b[3])
        ax = axs[0, col]
        ax.scatter(c.longitude[sel & ~assigned], c.latitude[sel & ~assigned], s=0.3, c="0.8", lw=0, rasterized=True)
        for s in surfaces:
            m = s["idx"]
            ax.scatter(c.longitude.values[m], c.latitude.values[m], s=0.4, color=cols[s["props"]["surface_id"]], lw=0,
                       rasterized=True)
        ax.set_title(f"{title}: events by fault (grey = unassigned)", fontsize=10)
        ax = axs[1, col]
        ax.scatter(c.longitude[sel], c.latitude[sel], s=0.3, c="0.85", lw=0, rasterized=True)
        polys, vals = [], []
        for s in surfaces:
            for t in s["tris"]:
                polys.append(np.c_[s["lon"][t], s["lat"][t]]); vals.append(s["dep"][t].mean())
        pc = PolyCollection(polys, array=np.array(vals), cmap="viridis_r", clim=(0, 30), edgecolors="none", alpha=0.8)
        ax.add_collection(pc)
        ax.set_title(f"{title}: rectangular surfaces (colour = depth)", fontsize=10)
        for ax in axs[:, col]:
            ax.set_xlim(b[0], b[1]); ax.set_ylim(b[2], b[3]); ax.set_aspect(1 / np.cos(np.radians((b[2] + b[3]) / 2)))
    fig.colorbar(pc, ax=axs[1, :], label="Surface depth (km, catalog datum)", shrink=0.6)
    fig.suptitle(f"Fault surfaces {stem.split('_')[-1]} from catalog v4 (plane-based detection; method test)", fontsize=12)
    fig.savefig(out / f"{stem}_map.png", dpi=170)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--box", type=float, nargs=4, default=None, help="lon_min lon_max lat_min lat_max (default: all)")
    ap.add_argument("--tol-h", type=float, default=1.5, help="nominal relative horizontal error (km)")
    ap.add_argument("--tol-z", type=float, default=3.0, help="nominal relative depth error (km)")
    ap.add_argument("--tol-k", type=float, default=1.5, help="tolerance in units of the error along the normal")
    ap.add_argument("--ball", type=float, default=20.0, help="radius for scoring a candidate plane (km)")
    ap.add_argument("--max-reach", type=float, default=150.0, help="largest search radius while growing (km)")
    ap.add_argument("--gap", type=float, default=4.0, help="in-plane connectivity cell (km)")
    ap.add_argument("--trials", type=int, default=150)
    ap.add_argument("--min-seed-nb", type=int, default=6, help="events within 5 km for a seed")
    ap.add_argument("--n-min", type=int, default=30)
    ap.add_argument("--merge-angle", type=float, default=15.0)
    ap.add_argument("--min-dip", type=float, default=30.0, help="reject flatter planes (v4 depths cannot resolve them)")
    ap.add_argument("--exclude", nargs="*", default=["AX"], help="region prefixes left out (Axial: published fault maps are better)")
    ap.add_argument("--planarity", type=float, default=0.5)
    ap.add_argument("--pad", type=float, default=0.5, help="rectangle padding (km)")
    ap.add_argument("--grid", type=float, default=1.0)
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--no-stability", action="store_true")
    ap.add_argument("--tag", default="v0")
    ap.add_argument("--out", default=str(D / "faults" / "v0"))
    a = ap.parse_args(argv)
    c, _ = v0.load(1.0, 1e9)                          # QC and depth >= 0, no sigma cut
    if a.exclude:
        c = c[~c.region.isin(a.exclude)].reset_index(drop=True)
    if a.box:
        c = c[c.longitude.between(a.box[0], a.box[1]) & c.latitude.between(a.box[2], a.box[3])].reset_index(drop=True)
    print(f"{len(c):,} events used")
    planes = run_detection(c, a, 0)
    assigned = np.zeros(len(c), bool)
    for m in planes:
        assigned[m] = True
    print(f"{len(planes)} planes; {assigned.sum():,} of {len(c):,} events assigned ({assigned.mean():.0%})")
    print(pd.DataFrame({"region": c.region, "assigned": assigned}).groupby("region")["assigned"]
          .agg(["size", "mean"]).round(2).to_string())
    rel, cf, sl = v0.relief(), v0.cfm(), v0.slab()
    rng = np.random.default_rng(0)
    surfaces = [surface_rect(c, m, a, cf, sl, rel, rng) for m in planes]
    surfaces.sort(key=lambda s: (s["props"]["region"], -s["props"]["n_events"]))
    if not a.no_stability:
        print("stability runs, planes:", stability(c, surfaces, a))
    else:
        for s in surfaces:
            s["props"].update(cluster_jaccard_median=None, kind="", stable=None)
    out = Path(a.out)
    summ, stem = write(c, surfaces, out, a.tag)
    figure(c, surfaces, out, stem, pd.Series(assigned, index=c.index))
    pd.DataFrame({"region": c.region, "assigned": assigned}).groupby("region")["assigned"].agg(
        events="size", assigned="sum", fraction="mean").round(3).to_csv(out / f"{stem}_assignment.csv")
    pd.set_option("display.width", 250)
    cols = ["surface_id", "kind", "stable", "n_events", "lon", "lat", "depth_km", "strike", "dip", "length_km",
            "width_km", "rms_normal_km", "sigma_total_median_km", "cluster_jaccard_median"]
    print(summ[[x for x in cols if x in summ]].to_string(index=False))


if __name__ == "__main__":
    main()

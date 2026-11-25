"""Fault surfaces v1 from catalog v5.1, step 2: strands, screening, surfaces, uncertainties, assignment, v0 comparison.

Input: the plane detection of v1_detect.py (v0 method and parameters on the v1 fit set, with the dip rule applied after
every reassignment pass; v1_common.py has the event selection). Steps:

1. Blanco strands (the rule of split_strands.py, now automatic). Parents are the Blanco 3D planes that are steep
   (dip >= 45 deg), within 35 deg of plate motion (111 deg) and with less than half of their events inside the Cascadia
   Depression box, in three pools along the transform as in v0 (west: Surveyor, East and West Blanco depressions,
   lon < -128.85; Blanco Ridge, -128.85 to -127.30; Gorda Depression, > -127.30). A pool is the parents' events plus
   the unassigned fit events inside their map footprint (2 km buffer), without events inside the Cascadia Depression
   box (v0 literature check: the plate-motion strand search makes NW strands there, where published faults strike
   N-S). Strands: map-view RANSAC lines of split_strands.py (azimuth 111 +/- 20 deg, half-width 1.5 km, gaps < 4 km,
   >= 30 events); two overlapping parallel strands are kept apart only if Ashman's D > 2 in the overlap
   (split_strands.merge_lines) and D > 2 again within every year where both have >= 15 events there; a pair whose
   depth offset would need a single fault dipping >= 60 deg (median depths of depth_quality ok events in the overlap)
   is merged: the offset follows depth, as for the two "deeper band" strands merged in v0.
2. Rectangles with uncertainties for every 3D plane (non-parent) and strand (fit_surface): as
   fault_surfaces_planes.surface_rect, but the location Monte Carlo moves each event by its own catalog error
   (sigma_h, sigma_h, sigma_z; v1_common) instead of the nominal 1.5/1.5/3 km. Extra fields: centroid uncertainty,
   the members' error along the normal (sigma_n), observed scatter / sigma_n, the zone width left once the errors
   are taken out, tier and depth_quality mix, a dip and depth range from the depth_quality ok members only, the
   v4 -> v5.1 shift of the members, the weekday-daytime share, flags.
3. Screening (removed surfaces listed with the reason and their members): final dip < 30 deg; >= 60 % of events on
   weekdays 08-17 local time (quarry blasts or other non-earthquake signals; the catalog average is 28 %); Blanco
   surfaces centred deeper than 25 km with < 30 depth_quality ok events (deep artefacts, as BL022/BL024/BL026 in v0).
4. Stability: Jaccard overlap with the best-matching plane of the four stability runs (3D planes) or of four strand
   reruns (seeds 101, 202; half-width 1.2, 1.8 km), as in v0. Detection-level recovery of each 3D plane in the
   verifier's three detection reruns (80 % subsample, one perturbation of every event by its own error, tiers A-B
   only; WORK/verifier_reruns), and the strike/dip range over the four stability runs. sigma_total and the strike/dip
   intervals of step 2 hold for a fixed member set; these columns give the spread of the detection itself.
   Flags carry the catalog problems: Mount St Helens deep population, 2012 Brookings swarm, members moved > 10 km in
   depth from v4 to v5.1, weekday-daytime excess (binomial, Bonferroni), strand dips set by the band selection.
5. Assignment of every located event (loc_source nll3d or comcat): distance d from the event to the surface (normal
   offset from the fitted surface at the nearest point of the rectangle, plus the in-plane distance outside it) against
   z = d / sqrt(s_e^2 + s_s^2 + W^2): s_e the event's catalog error along that direction, s_s the surface's median
   sigma_total, W the surface's zone width. Compatible: z <= 2 and d <= 10 km. Fit members keep their surface;
   others are assigned when exactly one surface is compatible (compatible, not located on it), ambiguous when two or
   more are. on_surface_strict: fit members and assigned tier A-C events within 2 km without floor_limited, far_only or
   deep_pass. Axial events are 'not fitted' (Axial was left out). Non-earthquakes (event_flag) and the weekday-daytime
   members of surfaces removed for their timing are never assigned; the other members of removed surfaces are
   assigned like any event (column removed_surface).
6. Comparison with v0 (data/faults/v0): v0 members (catalog v4 IDs = v5.1 IDs) against v1 fit members, removed v1
   surfaces included. A v0 and a v1 surface are linked when they share >= 15 events and these are >= 50 % of the v0
   surface's events that are v1 fit members, or >= 50 % of the v1 surface's members that were on a v0 surface, or
   >= 30 % of both. Connected groups give matched (1-1), split (1-n), merged (n-1), split and merged, new (v1 only),
   lost (v0 only; "dispersed" when most of its re-found events sit on several v1 surfaces). Literature fields of v0 are
   carried to v1: unchanged where matched with normals within 25 deg and centroids within 10 km, marked for re-check
   for every other link; new surfaces are listed for a literature check.

    python workflow/07_faults/v1_build.py [--procs 14]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from matplotlib.path import Path as MPath
from scipy.spatial import ConvexHull, cKDTree
from scipy.stats import binom

import v1_common as C
import fault_surfaces_v0 as v0
import fault_surfaces_planes as fp
import split_strands as ss
from v1_detect import RUNS, load_run

KM = v0.KM
POOLS = [("west", -131.5, -128.85), ("blanco_ridge", -128.85, -127.30), ("gorda_depression", -127.30, -126.0)]
CAS_BOX = (-128.82, -128.60, 43.74, 43.88)
PLATE_AZ = 111.0
Z_MAX = 2.0
D_CAP = 10.0
LINK_N = 15
LIT_DZ = 5.0                       # literature carried unchanged only if the centroid depth moved <= 5 km
STRICT_D = 2.0                     # on_surface_strict: assigned events within 2 km (tiers A-C, no floor/far/deep flags)
AXIAL_CLASS = "not fitted (Axial Seamount left out)"
BLAST_CLASS = "non-earthquake timing (weekday-daytime member of a surface removed for its timing)"
MSH_BOX = (-122.35, -122.10, 46.05, 46.30)   # Mount St Helens box of the v5.1 review (P1)
BROOK_BOX = (-125.0, -124.4, 42.0, 42.4)     # 2012 Brookings swarm (v5.1 review P2)
SHIFT_FLAG = C.SHIFT_FLAG          # flag surfaces whose members moved > 10 km in depth from v4 to v5.1
LIT =["lit_match", "lit_confidence", "lit_class", "lit_new_candidate", "lit_refs", "lit_dois", "lit_urls"]
G = {}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def in_cas(lon, lat):
    return (lon >= CAS_BOX[0]) & (lon <= CAS_BOX[1]) & (lat >= CAS_BOX[2]) & (lat <= CAS_BOX[3])


# ------------------------------------------------------------------ rectangle fit with per-event errors
def fit_surface(job):
    idx, seed = job
    f, a = G["f"], G["a"]
    cfmk, slabf, reliefk = G["aux"]
    rng = np.random.default_rng(seed)
    lon, lat, dep = f.longitude.values[idx], f.latitude.values[idx], f.z.values[idx]
    sh, sz = f.sh.values[idx], f.sz.values[idx]
    lon0, lat0 = float(np.median(lon)), float(np.median(lat))
    kx = KM * np.cos(np.radians(lat0))
    P = np.c_[(lon - lon0) * kx, (lat - lat0) * KM, dep]
    c0, var, V = v0.pca(P)
    planarity = float(np.sqrt(var[2] / var[1]))
    n, eu, ev = fp.frame_of(V[:, 2])
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
    resid = w - v0.design(u, v, quad) @ coef
    rms = float(np.sqrt(np.mean(resid ** 2)))
    sn = np.sqrt((sh * np.hypot(n[0], n[1])) ** 2 + (sz * n[2]) ** 2)       # event error along the normal
    sig = np.c_[sh, sh, sz]

    def draw(boot, perturb):
        W, S, Dp, Cc = [], [], [], []
        for _ in range(a.draws):
            j = rng.integers(0, len(idx), len(idx)) if boot else np.arange(len(idx))
            Pj = P[j] + (rng.normal(size=(len(j), 3)) * sig[j] if perturb else 0.0)
            Qj = Pj - c0
            W.append(v0.design(vu, vv, quad) @ v0.fit_w(Qj @ eu, Qj @ ev, Qj @ n, quad) - vw)
            cj, _, Vj = v0.pca(Pj)
            sj, dj = v0.strike_dip(Vj[:, 2])
            S.append(sj); Dp.append(dj); Cc.append(cj)
        return np.array(W), np.array(S), np.array(Dp), np.array(Cc)
    Wb, Sb, Db, Cb = draw(True, False)
    Wl, Sl, Dl, Cl = draw(False, True)
    Wt, St, Dt, Ct = draw(True, True)
    variants = [v0.design(vu, vv, q) @ v0.fit_w(u, v, w, q, t) for q in (False, True) for t in (False, True)]
    sb, sl, sm = Wb.std(0), Wl.std(0), np.std(variants, axis=0)
    st = np.sqrt(sb ** 2 + sl ** 2 + sm ** 2)
    s_int, d_int = v0.interval_stats(Sb, Db)
    s_loc, d_loc = v0.interval_stats(Sl, Dl)
    s_mc, d_mc = v0.interval_stats(St, Dt)
    X = c0 + np.outer(vu, eu) + np.outer(vv, ev) + np.outer(vw, n)
    vlon, vlat = X[:, 0] / kx + lon0, X[:, 1] / KM + lat0
    tris = []
    for i in range(nv_ - 1):
        for k in range(nu - 1):
            q = i * nu + k, i * nu + k + 1, (i + 1) * nu + k + 1, (i + 1) * nu + k
            tris += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
    clon, clat = c0[0] / kx + lon0, c0[1] / KM + lat0
    dist, jj = cfmk[0].query(v0.ecef([clon], [clat], [c0[2]])[0])
    fpp = cfmk[2][cfmk[1][jj]]
    sd = float(slabf([clon], [clat])[0] - c0[2])
    elev = reliefk(vlon, vlat)
    above = -X[:, 2] > elev
    dq = f.depth_quality.values[idx]
    ok = dq == "ok"
    dip_ok = strike_ok = np.nan
    top_ok = bot_ok = np.nan
    if ok.sum() >= 30:
        strike_ok, dip_ok = v0.strike_dip(v0.pca(P[ok])[2][:, 2])
        top_ok, bot_ok = np.percentile(dep[ok], [2, 98])
    tier = f.tier.values[idx]
    yrs = f.year.values[idx]
    r1 = lambda x: round(float(x), 1)
    r2 = lambda x: round(float(x), 2)
    top, bot = float(X[:, 2].min()), float(X[:, 2].max())
    W2 = max(0.0, rms ** 2 - float(np.mean(sn ** 2)))
    props = dict(
        region=pd.Series(f.region.values[idx]).mode()[0], n_events=int(len(idx)),
        lon=round(float(clon), 4), lat=round(float(clat), 4), depth_km=r2(c0[2]), top_km=r1(top), bottom_km=r1(bot),
        strike=r1(strike), dip=r1(dip),
        strike_68=[r1(s_int[1]), r1(s_int[3])], strike_95=[r1(s_int[0]), r1(s_int[4])],
        dip_68=[r1(d_int[1]), r1(d_int[3])], dip_95=[r1(d_int[0]), r1(d_int[4])],
        strike_68_loc_mc=[r1(s_loc[1]), r1(s_loc[3])], dip_68_loc_mc=[r1(d_loc[1]), r1(d_loc[3])],
        strike_68_with_loc_mc=[r1(s_mc[1]), r1(s_mc[3])], dip_68_with_loc_mc=[r1(d_mc[1]), r1(d_mc[3])],
        centroid_sigma_h_boot_km=r2(np.sqrt(Cb[:, 0].var() + Cb[:, 1].var())),
        centroid_sigma_z_boot_km=r2(Cb[:, 2].std()),
        centroid_sigma_h_loc_km=r2(np.sqrt(Cl[:, 0].var() + Cl[:, 1].var())),
        centroid_sigma_z_loc_km=r2(Cl[:, 2].std()),
        model="quadratic" if quad else "plane", planarity=round(planarity, 3),
        length_km=r1(u1 - u0), width_km=r1(v1 - v0_),
        rms_normal_km=r2(rms), tolerance_km=r2(fp.tol_of(n, a)),
        median_sigma_h_km=r2(np.median(sh)), median_sigma_z_km=r2(np.median(sz)),
        median_sigma_n_km=r2(np.median(sn)), scatter_to_sigma_n=r2(rms / np.sqrt(np.mean(sn ** 2))),
        median_abs_resid_over_sigma_n=r2(np.median(np.abs(resid) / sn)),
        zone_width_km=r2(np.sqrt(W2)),
        sigma_total_median_km=r2(np.median(st)), sigma_boot_median_km=r2(np.median(sb)),
        sigma_loc_median_km=r2(np.median(sl)), sigma_method_median_km=r2(np.median(sm)),
        constrained_fraction=r2(constrained.mean()), n_vertices=int(len(vu)), n_triangles=int(len(tris)),
        n_vertices_above_surface=int(above.sum()),
        n_tier_A=int((tier == "A").sum()), n_tier_B=int((tier == "B").sum()), n_tier_C=int((tier == "C").sum()),
        frac_depth_ok=r2(ok.mean()), frac_depth_unconstrained=r2((dq == "unconstrained").mean()),
        frac_depth_biased_deep=r2((dq == "biased_deep").mean()),
        strike_depth_ok=None if np.isnan(dip_ok) else r1(strike_ok), dip_depth_ok=None if np.isnan(dip_ok) else r1(dip_ok),
        events_depth_ok_2pct_km=None if np.isnan(top_ok) else r1(top_ok),
        events_depth_ok_98pct_km=None if np.isnan(bot_ok) else r1(bot_ok),
        median_shift_h_km=r2(np.nanmedian(f.shift_h_km.values[idx])),
        median_shift_z_km=r2(np.nanmedian(f.shift_z_km.values[idx])),
        median_shift_z_depth_ok_km=r2(np.nanmedian(f.shift_z_km.values[idx][ok])) if ok.sum() >= 10 else None,
        weekday_daytime_fraction=r2(f.weekday_daytime.values[idx].mean()),
        first_year=int(yrs.min()), last_year=int(yrs.max()),
        nearest_cfm=fpp.get("name"), nearest_cfm_km=r1(dist), nearest_cfm_dip=fpp.get("dip"),
        slab2_minus_depth_km=r1(sd) if np.isfinite(sd) else None,
    )
    geom = dict(lon0=lon0, lat0=lat0, kx=kx, c0=c0, n=n, eu=eu, ev=ev, box=(u0, u1, v0_, v1), quad=bool(quad),
                coef=coef, s_surf=float(np.median(st)), W=float(np.sqrt(W2)))
    return dict(idx=idx, lon=vlon, lat=vlat, dep=X[:, 2], tris=np.array(tris), constrained=constrained, sb=sb, sl=sl,
                sm=sm, st=st, Wt=Wt, normal=n, surface_elev_km=elev, props=props, geom=geom, resid=resid, sn=sn)


# ------------------------------------------------------------------ Blanco strands
def year_D(XY, A, B, years, min_n=15):
    ca, da, _, _ = ss.line_geom(XY, A)
    cb, db, _, _ = ss.line_geom(XY, B)
    d = da + np.sign(da @ db) * db
    d /= np.linalg.norm(d)
    nr = np.array([-d[1], d[0]])
    ua, ub = XY[A] @ d, XY[B] @ d
    lo, hi = max(ua.min(), ub.min()), min(ua.max(), ub.max())
    ia, ib = (ua >= lo) & (ua <= hi), (ub >= lo) & (ub <= hi)
    out = []
    for y in np.unique(np.r_[years[A], years[B]]):
        wa = (XY[A] @ nr)[ia & (years[A] == y)]
        wb = (XY[B] @ nr)[ib & (years[B] == y)]
        if len(wa) >= min_n and len(wb) >= min_n:
            out.append(float(np.sqrt(2) * abs(wa.mean() - wb.mean()) / np.sqrt(wa.var() + wb.var())))
    return out, ia, ib


def depth_test(XY, A, B, dep, ok, ia, ib, sep):
    za, zb = dep[A][ia & ok[A]], dep[B][ib & ok[B]]
    if len(za) < 10 or len(zb) < 10:
        return np.nan, np.nan, np.nan
    ma, mb = float(np.median(za)), float(np.median(zb))
    return ma, mb, float(np.degrees(np.arctan2(abs(ma - mb), sep)))


def strands_pool(name, XY, years, dep, ok, lat, sa, log_rows):
    lines = ss.run(XY, sa, 0, sa.tol)
    changed = True
    while changed:                                     # per-year and depth tests (merge on failure)
        changed = False
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                sp = ss.separation(XY, lines[i], lines[j])
                if sp is None:
                    continue
                yd, ia, ib = year_D(XY, lines[i], lines[j], years)
                za, zb, dsf = depth_test(XY, lines[i], lines[j], dep, ok, ia, ib, sp["sep_km"])
                why = None
                if yd and min(yd) <= sa.min_d:
                    why = f"per-year D {min(yd):.2f} <= {sa.min_d}"
                elif np.isfinite(dsf) and dsf >= 60:
                    why = f"depth offset needs a single fault dipping {dsf:.0f} deg (>= 60): offset follows depth"
                if why:
                    log_rows.append(dict(pool=name, n_a=len(lines[i]), n_b=len(lines[j]), D=round(sp["D"], 2),
                                         sep_km=round(sp["sep_km"], 2), reason=why))
                    lines[i] = np.union1d(lines[i], lines[j])
                    lines.pop(j)
                    changed = True
                    break
            if changed:
                break
    lines = ss.reassign_lines(XY, lines, sa, sa.tol)
    lines = sorted(lines, key=lambda m: -len(m))
    alt = [ss.run(XY, sa, 101, sa.tol), ss.run(XY, sa, 202, sa.tol), ss.run(XY, sa, 0, sa.tol - 0.3),
           ss.run(XY, sa, 0, sa.tol + 0.3)]
    stab = []
    for m in lines:
        J = []
        for L in alt:
            lab = -np.ones(len(XY), int)
            for k, mm in enumerate(L):
                lab[mm] = k
            la = lab[m]; la = la[la >= 0]
            if len(la) == 0:
                J.append(0.0); continue
            B = np.flatnonzero(lab == np.bincount(la).argmax())
            J.append(len(np.intersect1d(m, B)) / len(np.union1d(m, B)))
        stab.append(float(np.median(J)))
    pairs = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            sp = ss.separation(XY, lines[i], lines[j])
            if sp is None:
                continue
            yd, ia, ib = year_D(XY, lines[i], lines[j], years)
            za, zb, dsf = depth_test(XY, lines[i], lines[j], dep, ok, ia, ib, sp["sep_km"])
            pairs.append(dict(pool=name, a=i, b=j, **{k: round(v, 2) for k, v in sp.items()},
                              north_strand="a" if lat[lines[i]].mean() > lat[lines[j]].mean() else "b",
                              depth_med_a=round(za, 1), depth_med_b=round(zb, 1),
                              dip_if_single_fault=round(dsf, 0) if np.isfinite(dsf) else np.nan,
                              n_years_both=len(yd), D_per_year_min=round(min(yd), 2) if yd else np.nan,
                              D_per_year_median=round(float(np.median(yd)), 2) if yd else np.nan,
                              separation_justified=bool(sp["D"] > sa.min_d and (not yd or min(yd) > sa.min_d))))
    az = [float(ss.az_of(ss.line_geom(XY, m)[1])) for m in lines]
    return lines, stab, pairs, az


def blanco_strands(f, planes, props):
    """planes: list of f-index arrays; props: (lon, lat, strike, dip) per plane from a quick PCA."""
    sa = C.strand_params()
    assigned = np.zeros(len(f), bool)
    for m in planes:
        assigned[m] = True
    lon, lat = f.longitude.to_numpy(), f.latitude.to_numpy()
    out, parents_all, pair_rows, merge_log = [], set(), [], []
    years, dep, ok = f.year.to_numpy(), f.z.to_numpy(), (f.depth_quality == "ok").to_numpy()
    in_pool = np.zeros(len(f), bool)                   # an event enters one pool only (footprints overlap)
    for name, w, e in POOLS:
        par = [i for i, (plon, plat, pst, pdp) in enumerate(props)
               if f.region.values[planes[i]].tolist().count("BL") > 0.5 * len(planes[i]) and pdp >= 45
               and ss.az_diff(pst % 180, PLATE_AZ) <= 35 and w <= plon < e
               and in_cas(lon[planes[i]], lat[planes[i]]).mean() < 0.5]
        if not par:
            continue
        parents_all |= set(par)
        pm = np.unique(np.concatenate([planes[i] for i in par]))
        lon0, lat0 = np.median(lon[pm]), np.median(lat[pm])
        kx = KM * np.cos(np.radians(lat0))
        XYall = np.c_[(lon - lon0) * kx, (lat - lat0) * KM]
        hp = XYall[pm][ConvexHull(XYall[pm]).vertices]
        cen = hp.mean(0)
        hp = cen + (hp - cen) * (1 + 2.0 / np.linalg.norm(hp - cen, axis=1).mean())
        inside = MPath(hp).contains_points(XYall)
        pool = np.flatnonzero((np.isin(np.arange(len(f)), pm) | (inside & ~assigned)) & ~in_cas(lon, lat) & ~in_pool)
        in_pool[pool] = True
        log(f"strands {name}: {len(par)} parents, {len(pm):,} parent events, pool {len(pool):,}")
        lines, stab, pairs, az = strands_pool(name, XYall[pool], years[pool], dep[pool], ok[pool], lat[pool], sa, merge_log)
        log(f"strands {name}: {len(lines)} strands, {sum(len(m) for m in lines):,} events")
        for k, m in enumerate(lines):
            out.append(dict(idx=pool[m], pool=name, key=f"{name}:{k}", stab=stab[k], az=az[k],
                            parents=[int(i) for i in par if np.isin(pool[m], planes[i]).any()]))
        for p in pairs:
            p["a"], p["b"] = f"{name}:{p['a']}", f"{name}:{p['b']}"
            pair_rows.append(p)
    return out, sorted(parents_all), pair_rows, merge_log


# ------------------------------------------------------------------ stability of 3D planes
def plane_stability(f, surfaces):
    ev = f.event_id.to_numpy()
    pos = pd.Series(np.arange(len(f)), index=ev)
    P = np.c_[f.x, f.y, f.z]
    labs = []
    for name in RUNS:
        if name == "main":
            continue
        lab = -np.ones(len(f), int)
        for i, m in enumerate(load_run(name)):
            lab[pos.loc[m].to_numpy()] = i
        labs.append(lab)
    for s in surfaces:
        if s.get("strand"):
            continue
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
        if S:
            si, di = v0.interval_stats(np.array(S), np.array(Dp))
            p["strike_settings_range"] = [round(float(si[0]), 1), round(float(si[4]), 1)]
            p["dip_settings_range"] = [round(float(di[0]), 1), round(float(di[4]), 1)]


def range_width(x, circular):
    """Width of a [lo, hi] range (strike: modulo 360, the short way)."""
    if not isinstance(x, (list, tuple)) or len(x) != 2:
        return None
    d = float(x[1]) - float(x[0])
    if circular:
        d %= 360
        d = min(d, 360 - d)
    return round(d, 1)


# ------------------------------------------------------------------ detection-level recovery (verifier's reruns)
REC_DIR = C.WORK / "verifier_reruns"


def recovery_runs(f):
    """Plane detections rerun by the verifier of this piece (copied to WORK/verifier_reruns; scripts there):
    sub80 = random 80 % of the fit set (rng 7), tierAB = tiers A-B only, perturbed = every fit event moved once by its
    own catalog error (rng 11). Returns {name: (event -> plane label, universe or None)}."""
    out = {}
    for name in ("sub80", "perturbed", "tierAB"):
        p = REC_DIR / f"detect_{name}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        planes = [z[k] for k in sorted(z.files)]
        lab = pd.Series(np.concatenate([np.full(len(m), i) for i, m in enumerate(planes)]), index=np.concatenate(planes))
        if name == "sub80":
            rng = np.random.default_rng(7)
            uni = set(f.event_id.values[np.sort(rng.choice(len(f), int(0.8 * len(f)), replace=False))])
        elif name == "tierAB":
            uni = set(f.event_id.values[f.tier.isin(["A", "B"]).to_numpy()])
        else:
            uni = None
        out[name] = (lab, uni)
    return out


def recovery_J(ev, lab, uni):
    """Jaccard overlap of a surface's fit members with the best-matching rerun plane (both within the rerun's events)."""
    A = set(ev) if uni is None else set(ev) & uni
    if not A:
        return np.nan
    la = lab.reindex(list(A)).dropna().astype(int)
    if len(la) == 0:
        return 0.0
    k = la.value_counts().idxmax()
    B = set(lab.index[lab.to_numpy() == k])
    return len(A & B) / len(A | B)


# ------------------------------------------------------------------ catalog problems carried as surface flags
def wed_window(t):
    """Wednesday 19:14-19:21 UTC: the weekly artificial signal on analog UW channels (v5.1 review P1)."""
    return ((t.dt.dayofweek == 2) & (t.dt.hour == 19) & t.dt.minute.between(14, 21)).to_numpy()


def catalog_flags(f, idx, p):
    """Flags for the catalog problems found by this piece and reviewed in /wd1/.../catalog_v5/v5.1_review."""
    fl = []
    dep, v4 = f.z.values[idx], f.v4_depth.values[idx]
    ok = (f.depth_quality.values[idx] == "ok")
    lo, la = p["lon"], p["lat"]
    if MSH_BOX[0] <= lo <= MSH_BOX[1] and MSH_BOX[2] <= la <= MSH_BOX[3] and (p["depth_km"] > 20 or (dep > 30).mean() >= 0.25):
        nw = int(wed_window(f.t.iloc[idx]).sum())
        fl.append(f"Mount St Helens deep population: {int((dep > 30).sum())} of {len(idx)} members deeper than 30 km "
                  f"(members {np.percentile(dep, 2):.0f}-{np.percentile(dep, 98):.0f} km), {nw} in the Wednesday 19:14-19:21 UTC "
                  f"window, the same suspect population as removed v1x surface under the volcano (analog UW channels HSR, EDM, "
                  f"STD carry a weekly artificial signal 2011-05 to 2013-08, v5.1 review P1): members may be false events, "
                  f"surface not usable until reviewed")
    if BROOK_BOX[0] <= lo <= BROOK_BOX[1] and BROOK_BOX[2] <= la <= BROOK_BOX[3] and p["depth_km"] > 30:
        sd = p.get("slab2_minus_depth_km")
        fl.append(f"2012 Brookings swarm depth probably wrong: members at {dep.min():.0f}-{dep.max():.0f} km in v5.1 against a "
                  f"median {np.nanmedian(v4):.0f} km in v4, depth_quality ok for {ok.mean():.0%}"
                  + (f", Slab2 slab top {sd + p['depth_km']:.0f} km at the centroid" if sd is not None else "")
                  + ". Predicted S-P is not monotonic with depth in the v5.1 grids, so about 7-9 km and about 56 km both fit "
                  "the distant stations and NonLinLoc took the deep one (v5.1 review P2)")
    sz_all, sz_ok = p["median_shift_z_km"], p.get("median_shift_z_depth_ok_km")
    if any(x is not None and np.isfinite(x) and abs(x) > SHIFT_FLAG for x in (sz_all, sz_ok)):
        fl.append(f"members moved a median {sz_all:+.1f} km in depth from catalog v4 to v5.1"
                  + (f" ({sz_ok:+.1f} km for the {ok.mean():.0%} with depth_quality ok)" if sz_ok is not None else "")
                  + ": depth not confirmed (v5.1 events with ComCat moment tensors are a median 13.7 km deeper than ComCat, "
                  "v5.1 review P3)")
    return fl


def event_notes(c, asg, keep):
    """Per-event notes for the reviewed catalog problems (v5.1 review P1, P2)."""
    e = c.set_index("event_id").loc[asg.event_id]
    note = np.full(len(asg), "", object)
    msh = (e.longitude.between(MSH_BOX[0], MSH_BOX[1]) & e.latitude.between(MSH_BOX[2], MSH_BOX[3]) & (e.z > 30)).to_numpy()
    note[msh] = ("Mount St Helens deep population (box, deeper than 30 km; v5.1 review P1): likely false events from a "
                 "weekly artificial signal on analog UW channels, pending human review")
    brook = set()
    for s in keep:
        if "Brookings swarm" in s["props"]["flags"]:
            brook |= set(G["f"].event_id.values[s["idx"]])
    note[asg.event_id.isin(brook).to_numpy()] = ("2012 Brookings swarm (v5.1 review P2): v5.1 depth probably wrong "
                                                 "(two S-P depth minima in the v5.1 grids; NonLinLoc took the deep one)")
    return note


# ------------------------------------------------------------------ assignment of every located event
def assign(c, surfaces, member_of, blast_events):
    L = c[c.located].reset_index(drop=True)
    eq = ~L.nonquake.to_numpy()
    xy = np.c_[L.x, L.y]
    tree = cKDTree(xy)
    lon, lat, dep = L.longitude.to_numpy(), L.latitude.to_numpy(), L.z.to_numpy()
    sh, sz = L.sh.to_numpy(), L.sz.to_numpy()
    N = len(L)
    best_z, best_s, best_d = np.full(N, np.inf), -np.ones(N, int), np.full(N, np.nan)
    second_z, second_s, ncomp = np.full(N, np.inf), -np.ones(N, int), np.zeros(N, int)
    zmem, dmem = np.full(N, np.nan), np.full(N, np.nan)
    mem_s = L.event_id.map(member_of).fillna(-1).astype(int).to_numpy()
    for k, s in enumerate(surfaces):
        g = s["geom"]
        u0, u1, v0_, v1 = g["box"]
        clon, clat = s["props"]["lon"], s["props"]["lat"]
        cx, cy = v0.sinu([clon], [clat])
        R = np.hypot(max(abs(u0), abs(u1)), max(abs(v0_), abs(v1))) + 40.0
        cand = np.array(tree.query_ball_point([cx[0], cy[0]], R), int)
        if len(cand) == 0:
            continue
        P = np.c_[(lon[cand] - g["lon0"]) * g["kx"], (lat[cand] - g["lat0"]) * KM, dep[cand]] - g["c0"]
        u, v, w = P @ g["eu"], P @ g["ev"], P @ g["n"]
        uc, vc = np.clip(u, u0, u1), np.clip(v, v0_, v1)     # surface evaluated at the nearest point of the rectangle
        wf = w - v0.design(uc, vc, g["quad"]) @ g["coef"]
        du, dv = u - uc, v - vc
        dvec = du[:, None] * g["eu"] + dv[:, None] * g["ev"] + wf[:, None] * g["n"]
        d = np.linalg.norm(dvec, axis=1)
        dirn = np.where(d[:, None] > 1e-9, dvec / np.maximum(d, 1e-9)[:, None], g["n"])
        se2 = (sh[cand] ** 2) * (dirn[:, 0] ** 2 + dirn[:, 1] ** 2) + (sz[cand] ** 2) * dirn[:, 2] ** 2
        z = d / np.sqrt(se2 + g["s_surf"] ** 2 + g["W"] ** 2)
        mine = mem_s[cand] == k
        zmem[cand[mine]], dmem[cand[mine]] = z[mine], d[mine]
        z = np.where(d <= D_CAP, z, np.inf)                 # compatible: z <= Z_MAX and d <= D_CAP
        ncomp[cand] += (z <= Z_MAX)
        b = z < best_z[cand]
        cb = cand[b]
        second_z[cb], second_s[cb] = best_z[cb], best_s[cb]
        best_z[cb], best_s[cb], best_d[cb] = z[b], k, d[b]
        nb = ~b & (z < second_z[cand])
        second_z[cand[nb]], second_s[cand[nb]] = z[nb], k
    cls = np.full(N, "none", object)
    sid = -np.ones(N, int)
    a = eq & (best_z <= Z_MAX) & (ncomp == 1)
    cls[a], sid[a] = "assigned", best_s[a]
    cls[eq & (ncomp >= 2)] = "ambiguous"
    # Axial Seamount was never fitted (left out of the fit set, as in v0): its events are not "on no surface"
    ax = eq & (L.fregion.to_numpy() == "AX")
    cls[ax], sid[ax] = AXIAL_CLASS, -1
    # members of surfaces removed for their timing: only their weekday-daytime events are marked, one by one
    bl = L.event_id.isin(blast_events).to_numpy() & L.weekday_daytime.to_numpy() & eq
    cls[bl], sid[bl] = BLAST_CLASS, -1
    m = mem_s >= 0
    cls[m], sid[m] = "fit member", mem_s[m]
    cls[~eq] = "not an earthquake"
    sid[~eq] = -1
    zmem = np.where(np.isfinite(zmem), zmem, np.nan)
    zout = np.where(m, zmem, best_z)
    dout = np.where(m, dmem, best_d)
    tier = L.tier.to_numpy()
    strict = (cls == "fit member") | ((cls == "assigned") & np.isin(tier, ["A", "B", "C"]) & (dout <= STRICT_D)
                                      & ~L.floor_limited.to_numpy() & ~L.far_only.to_numpy() & ~L.deep_pass.to_numpy())
    out = pd.DataFrame(dict(event_id=L.event_id, assignment=cls, surface=sid,
                            distance_km=np.round(dout, 2), z=np.round(np.where(np.isfinite(zout), zout, np.nan), 2),
                            n_compatible=np.where(eq, ncomp, 0), on_surface_strict=strict,
                            second_surface=np.where(eq & np.isfinite(second_z), second_s, -1),
                            second_z=np.round(np.where(eq & np.isfinite(second_z), second_z, np.nan), 2),
                            nearest_surface=np.where(eq & np.isfinite(best_z), best_s, -1),
                            nearest_z=np.round(np.where(eq & np.isfinite(best_z), best_z, np.nan), 2),
                            tier=L.tier, fit_set=L.fit, fit_excluded_because=L.fit_excl, region=L.fregion,
                            depth_quality=L.depth_quality, floor_limited=L.floor_limited, far_only=L.far_only,
                            deep_pass=L.deep_pass, five_phase=L.five_phase, weekday_daytime=L.weekday_daytime,
                            sigma_h_km=np.round(sh, 2), sigma_z_km=np.round(sz, 2), sigma_fallback=L.sigma_fallback))
    return out


# ------------------------------------------------------------------ comparison with v0
def _uf_groups(v0_ids, v1_ids, links):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a in v0_ids:
        find(("v0", a))
    for b in v1_ids:
        find(("v1", b))
    for a, b, *_ in links:
        ra, rb = find(("v0", a)), find(("v1", b))
        if ra != rb:
            parent[ra] = rb
    groups = {}
    for x in list(parent):
        groups.setdefault(find(x), []).append(x)
    return list(groups.values())


def angle(n1, n2):
    return float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(n1, n2)))))))


def strike_diff(a, b, steep):
    d = (a - b) % (180 if steep else 360)
    m = 180 if steep else 360
    return float(min(d, m - d))


def comparison(c, f, nodes):
    """nodes: v1 surfaces, kept and removed: dicts with id, idx (fit-set rows of the fit members), props, normal,
    kept, reason. v0 members against v1 fit members; links, groups, literature carry-forward."""
    s0 = pd.read_csv(C.V0DIR / "fault_surfaces_v0_summary.csv").set_index("surface_id")
    m0 = pd.read_csv(C.V0DIR / "fault_surfaces_v0_members.csv")
    lit = pd.read_csv(C.V0DIR / "fault_surfaces_v0_literature.csv").set_index("surface_id")
    ids = [nd["id"] for nd in nodes]
    v1_of = pd.Series(-1, index=c.event_id.to_numpy())
    for k, nd in enumerate(nodes):
        v1_of.loc[f.event_id.values[nd["idx"]]] = k
    cat = c.set_index("event_id")
    m0 = m0.assign(v1=m0.event_id.map(v1_of), located=m0.event_id.map(cat.located),
                   nonquake=m0.event_id.map(cat.nonquake), fit=m0.event_id.map(cat.fit),
                   shift_h=m0.event_id.map(cat.shift_h_km), shift_z=m0.event_id.map(cat.shift_z_km))
    in_v0 = v1_of.index.isin(m0.event_id.to_numpy())
    n_from_v0 = np.array([int(((v1_of == k) & in_v0).sum()) for k in range(len(nodes))])
    links = []
    for sid0, g in m0.groupby("surface_id"):
        on = g.v1.fillna(-1) >= 0
        n_on = int(on.sum())
        for k, nab in g.v1[on].astype(int).value_counts().items():
            f0, f1 = nab / max(n_on, 1), nab / max(n_from_v0[k], 1)
            if nab >= LINK_N and (f0 >= 0.5 or f1 >= 0.5 or (f0 >= 0.3 and f1 >= 0.3)):
                links.append((sid0, int(k), int(nab), round(float(f0), 3), round(float(f1), 3)))
    groups = _uf_groups(list(s0.index), list(range(len(nodes))), links)
    cls0, cls1, grp0, grp1 = {}, {}, {}, {}
    for g in groups:
        a = sorted(x[1] for x in g if x[0] == "v0")
        b_all = sorted(x[1] for x in g if x[0] == "v1")
        b = [x for x in b_all if nodes[x]["kept"]]
        kind = ("matched" if len(a) == 1 and len(b) == 1 else "split" if len(a) == 1 and len(b) > 1 else
                "merged" if len(a) > 1 and len(b) == 1 else "split and merged" if a and b else
                "new" if b else "lost" if a else "removed in v1")
        for x in a:
            cls0[x], grp0[x] = kind, (a, b_all)
        for x in b_all:
            cls1[x], grp1[x] = (kind if nodes[x]["kept"] else "removed in v1"), (a, b_all)
    prow = []
    for sid0, k, nab, f0, f1 in links:
        r0, p1 = s0.loc[sid0], nodes[k]["props"]
        n0 = np.array([float(x) for x in str(r0.normal_xyz).split(";")])
        steep = min(r0.dip, p1["dip"]) > 75
        kx = KM * np.cos(np.radians((r0.lat + p1["lat"]) / 2))
        dh = float(np.hypot((p1["lon"] - r0.lon) * kx, (p1["lat"] - r0.lat) * KM))
        g = m0[m0.surface_id == sid0]
        prow.append(dict(v0_surface=sid0, v1_surface=ids[k], v1_kept=nodes[k]["kept"], change=cls0[sid0],
                         shared_events=nab, frac_of_v0_refound=f0, frac_of_v1_from_v0=f1,
                         v0_strike=r0.strike, v1_strike=p1["strike"],
                         strike_change=round(strike_diff(p1["strike"], r0.strike, steep), 1),
                         v0_dip=r0.dip, v1_dip=p1["dip"], dip_change=round(p1["dip"] - r0.dip, 1),
                         normal_angle=round(angle(n0, nodes[k]["normal"]), 1),
                         centroid_shift_h_km=round(dh, 1), v0_depth_km=r0.depth_km, v1_depth_km=p1["depth_km"],
                         centroid_shift_z_km=round(p1["depth_km"] - r0.depth_km, 1),
                         v0_members_median_shift_h_km=round(float(g.shift_h.median()), 1),
                         v0_members_median_shift_z_km=round(float(g.shift_z.median()), 1),
                         v0_length_km=r0.length_km, v1_length_km=p1["length_km"],
                         v0_width_km=r0.width_km, v1_width_km=p1["width_km"],
                         v0_n_events=int(r0.n_events), v1_n_events=p1["n_events"]))
    pairs = pd.DataFrame(prow)
    r0rows = []
    for sid0, r0 in s0.iterrows():
        g = m0[m0.surface_id == sid0]
        nloc = int(g.located.fillna(False).sum())
        k1 = g.v1.fillna(-1).astype(int)
        kept_on = k1.map(lambda k: k >= 0 and nodes[k]["kept"])
        rem_on = k1.map(lambda k: k >= 0 and not nodes[k]["kept"])
        vc = k1[k1 >= 0].value_counts()
        top = ", ".join(f"{ids[k]} {n / max(nloc, 1):.0%}" for k, n in vc.head(3).items())
        a, b = grp0.get(sid0, ([sid0], []))
        kind = cls0.get(sid0, "lost")
        note = ""
        if kind == "lost":
            rb = [x for x in b if not nodes[x]["kept"]]
            nfit = int(g.fit.fillna(False).sum())
            if rb:
                note = "events now form " + "; ".join(f"{ids[x]} (removed: {nodes[x]['reason']})" for x in rb)
            else:
                note = (f"{nfit / max(len(g), 1):.0%} of its events are in the v1 fit set; {kept_on.mean():.0%} are fit "
                        f"members of v1 surfaces, none holding enough of them for a link")
                if kept_on.sum() >= 0.5 * max(nfit, 1):
                    kind = "dispersed"
        r0rows.append(dict(v0_surface=sid0, change=kind, v1_surfaces=";".join(ids[k] for k in b if nodes[k]["kept"]),
                           v1_removed=";".join(ids[k] for k in b if not nodes[k]["kept"]), v0_group=";".join(a),
                           v0_n_events=int(r0.n_events), located_in_v51=nloc,
                           not_an_earthquake=int(g.nonquake.fillna(False).sum()),
                           in_v1_fit_set=int(g.fit.fillna(False).sum()), fit_members_of_kept_v1=int(kept_on.sum()),
                           fit_members_of_removed_v1=int(rem_on.sum()), top_v1_surfaces=top, note=note,
                           v0_strike=r0.strike, v0_dip=r0.dip, v0_depth_km=r0.depth_km, v0_lon=r0.lon, v0_lat=r0.lat,
                           v0_lit_class=lit.lit_class.get(sid0, ""), v0_lit_match=lit.lit_match.get(sid0, ""),
                           v0_lit_confidence=lit.lit_confidence.get(sid0, ""),
                           members_median_shift_h_km=round(float(g.shift_h.median()), 1),
                           members_median_shift_z_km=round(float(g.shift_z.median()), 1)))
    t0 = pd.DataFrame(r0rows)
    r1rows = []
    for k, nd in enumerate(nodes):
        if not nd["kept"]:
            continue
        p = nd["props"]
        a, b = grp1.get(k, ([], [k]))
        kind = cls1.get(k, "new")
        pl = pairs[pairs.v1_surface == ids[k]].sort_values("shared_events", ascending=False)
        row = dict(v1_surface=ids[k], change=kind, v0_surfaces=";".join(a),
                   v1_group=";".join(ids[x] for x in b if nodes[x]["kept"]), n_events=p["n_events"],
                   fit_members_from_v0_surfaces=int(n_from_v0[k]))
        kx = KM * np.cos(np.radians(p["lat"]))
        d = np.hypot((s0.lon - p["lon"]) * kx, (s0.lat - p["lat"]) * KM)
        j = d.idxmin()
        n0 = np.array([float(x) for x in str(s0.loc[j].normal_xyz).split(";")])
        row.update(nearest_v0=j, nearest_v0_km=round(float(d.min()), 1),
                   nearest_v0_normal_angle=round(angle(n0, nd["normal"]), 1))
        if kind == "new":
            lv = {x: "" for x in LIT}
            lv["lit_status"] = "new in v1: literature check needed"
        else:
            best = pl.iloc[0]
            src = best.v0_surface
            lv = {x: (lit[x].get(src, "") if x in lit else "") for x in LIT}
            clear = (kind == "matched" and best.normal_angle <= 25 and best.centroid_shift_h_km <= 10
                     and abs(best.centroid_shift_z_km) <= LIT_DZ)
            if kind in ("merged", "split and merged"):
                srcs = list(pl.v0_surface)
                lv["lit_match"] = " | ".join(f"{x}: {lit.lit_match.get(x, '')}" for x in srcs)
                lv["lit_refs"] = "; ".join(dict.fromkeys(str(lit.lit_refs.get(x, "")) for x in srcs
                                                        if str(lit.lit_refs.get(x, "")) not in ("", "nan")))
                lv["lit_dois"] = ";".join(dict.fromkeys(dd for x in srcs for dd in str(lit.lit_dois.get(x, "")).split(";")
                                                        if dd and dd != "nan"))
            if clear:
                lv["lit_status"] = (f"carried from v0 {src} (matched: normals {best.normal_angle:.0f} deg apart, "
                                    f"centroids {best.centroid_shift_h_km:.0f} km apart horizontally and "
                                    f"{best.centroid_shift_z_km:+.0f} km in depth)")
            elif kind == "matched":
                lv["lit_status"] = (f"carried from v0 {src}, re-check: same events, geometry changed "
                                    f"(normals {best.normal_angle:.0f} deg, centroids {best.centroid_shift_h_km:.0f} km apart "
                                    f"horizontally, {best.centroid_shift_z_km:+.0f} km in depth)")
            elif kind == "split":
                lv["lit_status"] = (f"carried from v0 {src}, re-check: {src} split into "
                                    f"{', '.join(ids[x] for x in b if nodes[x]['kept'])}")
            else:
                lv["lit_status"] = f"carried from v0 {', '.join(pl.v0_surface)} ({kind}), re-check"
        row.update(lv)
        r1rows.append(row)
    t1 = pd.DataFrame(r1rows)
    return pairs, t0, t1


# ------------------------------------------------------------------ writing
def write_all(c, f, surfaces, ids, asg, removed, pairs_str, merge_log, cmp_pairs, t0, t1, counts):
    C.OUT.mkdir(parents=True, exist_ok=True)
    C.BIG.mkdir(parents=True, exist_ok=True)
    stem = "fault_surfaces_v1"
    lit1 = t1.set_index("v1_surface")
    feats, light, rows, vrows, erows, mrows, trows = [], [], [], [], [], [], []
    for k, s in enumerate(surfaces):
        sid = ids[k]
        p = dict(surface_id=sid, version=C.TAG, catalog=C.CATLABEL, **s["props"])
        p["normal_xyz"] = [round(float(x), 4) for x in s["normal"]]
        p["change_from_v0"] = lit1.loc[sid, "change"]
        p["v0_surfaces"] = lit1.loc[sid, "v0_surfaces"]
        for x in LIT + ["lit_status"]:
            v = lit1.loc[sid, x]
            p[x] = "" if (v is None or (isinstance(v, float) and np.isnan(v))) else (bool(v) if x == "lit_new_candidate" and v != "" else v)
        p = {kk: (vv.item() if isinstance(vv, np.generic) else vv) for kk, vv in p.items()}
        p = {kk: (None if isinstance(vv, float) and not np.isfinite(vv) else vv) for kk, vv in p.items()}
        tri = [[[[round(float(s["lon"][i]), 5), round(float(s["lat"][i]), 5), round(float(-s["dep"][i] * 1000), 1)]
                 for i in (t0_, t1_, t2_, t0_)]] for t0_, t1_, t2_ in s["tris"]]
        feats.append(dict(type="Feature", properties=p, geometry=dict(type="MultiPolygon", coordinates=tri)))
        # light: the four corners of the rectangle and the centre (planes are exact; quadratics approximate)
        g = s["geom"]
        u0, u1, v0_, v1 = g["box"]
        cu, cv = np.array([u0, u1, u1, u0]), np.array([v0_, v0_, v1, v1])
        cw = v0.design(cu, cv, g["quad"]) @ g["coef"]
        X = g["c0"] + np.outer(cu, g["eu"]) + np.outer(cv, g["ev"]) + np.outer(cw, g["n"])
        lo_, la_ = X[:, 0] / g["kx"] + g["lon0"], X[:, 1] / KM + g["lat0"]
        ring = [[round(float(lo_[i]), 5), round(float(la_[i]), 5), round(float(-X[i, 2] * 1000), 1)] for i in (0, 1, 2, 3, 0)]
        light.append(dict(type="Feature", properties={kk: p[kk] for kk in ("surface_id", "region", "n_events", "strike", "dip",
                     "depth_km", "top_km", "bottom_km", "length_km", "width_km", "model", "kind", "frac_depth_ok",
                     "sigma_total_median_km", "stable", "strike_settings_range_width", "recovery_tests", "flags",
                     "change_from_v0", "v0_surfaces", "lit_match", "lit_status") if kk in p},
                          geometry=dict(type="Polygon", coordinates=[ring])))
        rows.append(p)
        el = s["surface_elev_km"]
        vdf = pd.DataFrame({"surface_id": sid, "vertex": np.arange(len(s["lon"])), "lon": np.round(s["lon"], 5),
                            "lat": np.round(s["lat"], 5), "depth_km": np.round(s["dep"], 3),
                            "sigma_boot_km": np.round(s["sb"], 3), "sigma_loc_km": np.round(s["sl"], 3),
                            "sigma_method_km": np.round(s["sm"], 3), "sigma_total_km": np.round(s["st"], 3),
                            "constrained": s["constrained"].astype(int), "surface_elev_km": np.round(el, 3)})
        vrows.append(vdf)
        W = np.round(s["Wt"], 3)
        erows.append(pd.DataFrame({"surface_id": sid, "vertex": np.tile(np.arange(W.shape[1]), W.shape[0]),
                                   "draw": np.repeat(np.arange(W.shape[0]), W.shape[1]), "normal_offset_km": W.ravel()}))
        mrows.append(pd.DataFrame({"event_id": f.event_id.values[s["idx"]], "surface_id": sid}))
        trows.append(dict(surface_id=sid, region=p["region"], n_events=p["n_events"], lon=p["lon"], lat=p["lat"],
                          depth_km=p["depth_km"], top_km=p["top_km"], bottom_km=p["bottom_km"], strike=p["strike"],
                          dip=p["dip"], length_km=p["length_km"], width_km=p["width_km"], kind=p["kind"],
                          **ss.trace_row(p, vdf)))
    crs = dict(type="name", properties=dict(name="urn:ogc:def:crs:OGC:1.3:CRS84"))
    (C.BIG / f"{stem}.geojson").write_text(json.dumps(dict(type="FeatureCollection", name=stem, crs=crs, features=feats)))
    (C.OUT / f"{stem}_rectangles.geojson").write_text(json.dumps(dict(type="FeatureCollection", name=f"{stem}_rectangles",
                                                                      crs=crs, features=light)))
    summ = pd.DataFrame(rows)
    for col in summ.columns:
        if summ[col].map(lambda v: isinstance(v, list)).any():
            summ[col] = summ[col].map(lambda v: ";".join(map(str, v)) if isinstance(v, list) else v)
    unc_cols = ["surface_id", "n_events", "strike", "strike_68", "strike_95", "strike_68_loc_mc", "strike_68_with_loc_mc",
                "strike_settings_range", "dip", "dip_68", "dip_95", "dip_68_loc_mc", "dip_68_with_loc_mc", "dip_settings_range",
                "dip_depth_ok", "strike_depth_ok", "frac_depth_ok", "centroid_sigma_h_boot_km", "centroid_sigma_z_boot_km",
                "centroid_sigma_h_loc_km", "centroid_sigma_z_loc_km", "sigma_boot_median_km", "sigma_loc_median_km",
                "sigma_method_median_km", "sigma_total_median_km", "rms_normal_km", "median_sigma_n_km",
                "scatter_to_sigma_n", "median_abs_resid_over_sigma_n", "zone_width_km", "cluster_jaccard_median", "stable",
                "strike_settings_range_width", "dip_settings_range_width", "recovery_J_sub80", "recovery_J_perturbed",
                "recovery_J_tierAB", "recovery_tests", "strand"]
    summ.to_csv(C.OUT / f"{stem}_summary.csv", index=False)
    summ[[x for x in unc_cols if x in summ]].to_csv(C.OUT / f"{stem}_uncertainty.csv", index=False)
    V = pd.concat(vrows)
    V.to_csv(C.BIG / f"{stem}_vertices.csv.gz", index=False, compression="gzip")
    pd.concat(erows).to_csv(C.BIG / f"{stem}_ensemble.csv.gz", index=False, compression="gzip")
    pd.concat(mrows).to_csv(C.OUT / f"{stem}_members.csv", index=False)
    pd.DataFrame(trows).to_csv(C.OUT / f"{stem}_traces.csv", index=False)
    a = asg.copy()
    a["surface_id"] = np.where(a.surface >= 0, np.array(ids + [""], object)[a.surface], "")
    for col in ("second_surface", "nearest_surface"):
        a[col] = np.where(a[col] >= 0, np.array(ids + [""], object)[a[col]], "")
    a = a.drop(columns="surface")
    first = ["event_id", "assignment", "surface_id", "distance_km", "z", "n_compatible"]
    a = a[first + [x for x in a.columns if x not in first]]
    a.to_csv(C.BIG / f"{stem}_assignment.csv.gz", index=False, compression="gzip")
    a.to_csv(C.OUT / f"{stem}_assignment.csv.gz", index=False, compression="gzip")
    # region table
    rr = []
    for reg, g in a.groupby("region"):
        nf = g.assignment == AXIAL_CLASS
        eq = (g.assignment != "not an earthquake") & ~nf           # fractions only where surfaces were fitted
        fs = g.fit_set
        rr.append(dict(region=reg, located=int(len(g)), earthquakes=int((g.assignment != "not an earthquake").sum()),
                       not_fitted=int(nf.sum()), fit_set=int(fs.sum()),
                       fit_members=int((g.assignment == "fit member").sum()),
                       frac_fit_set_on_a_fault=round(float((g.assignment[fs] == "fit member").mean()), 3) if fs.any() else None,
                       assigned_other=int((g.assignment == "assigned").sum()),
                       frac_earthquakes_on_a_fault=(round(float(g.assignment[eq].isin(["fit member", "assigned"]).mean()), 3)
                                                    if eq.any() else None),
                       on_surface_strict=int(g.on_surface_strict.sum()),
                       frac_earthquakes_on_surface_strict=round(float(g.on_surface_strict[eq].mean()), 3) if eq.any() else None,
                       ambiguous=int((g.assignment == "ambiguous").sum()),
                       none=int((g.assignment == "none").sum()),
                       non_earthquake_timing_screening=int(g.assignment.str.startswith("non-earthquake timing").sum())))
    pd.DataFrame(rr).to_csv(C.OUT / f"{stem}_assignment_by_region.csv", index=False)
    pd.DataFrame(counts, columns=["rule", "events"]).to_csv(C.OUT / f"{stem}_selection.csv", index=False)
    removed.to_csv(C.OUT / f"{stem}_removed.csv", index=False)
    ps = pd.DataFrame(pairs_str)
    if len(ps):
        ps["a"] = ps.a.map(G["strand_ids"]); ps["b"] = ps.b.map(G["strand_ids"])
        ps["north_strand"] = np.where(ps.north_strand == "a", ps.a, ps.b)
    ps.to_csv(C.OUT / f"{stem}_strand_separation.csv", index=False)
    pd.DataFrame(merge_log).to_csv(C.OUT / f"{stem}_strand_merges.csv", index=False)
    cmp_pairs.to_csv(C.OUT / f"{stem}_v0_pairs.csv", index=False)
    t0.to_csv(C.OUT / f"{stem}_v0_status.csv", index=False)
    t1.to_csv(C.OUT / f"{stem}_v0_comparison.csv", index=False)
    t1[t1.change == "new"].to_csv(C.OUT / f"{stem}_literature_todo.csv", index=False)
    return summ


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--procs", type=int, default=14)
    a_ = ap.parse_args()
    t_start = time.time()
    c, counts = C.load()
    f = C.fit_set(c)
    a = C.params()
    log(f"fit set {len(f):,} events; located {int(c.located.sum()):,}; no hypocentre {int((~c.located).sum()):,}")
    pos = pd.Series(np.arange(len(f)), index=f.event_id.to_numpy())
    planes = [pos.loc[m].to_numpy() for m in load_run("main")]
    log(f"main detection: {len(planes)} planes, {sum(len(m) for m in planes):,} events")
    P = np.c_[f.x, f.y, f.z]
    quick = []
    for m in planes:
        cc, _, V = v0.pca(P[m])
        st_, dp = v0.strike_dip(V[:, 2])
        quick.append((float(np.median(f.longitude.values[m])), float(np.median(f.latitude.values[m])), st_, dp))
    strands, parents, pairs_str, merge_log = blanco_strands(f, planes, quick)
    log(f"Blanco: {len(parents)} parent planes replaced by {len(strands)} strands")
    jobs, kinds = [], []
    for i, m in enumerate(planes):
        if i not in parents:
            jobs.append(m); kinds.append(dict(strand=False, detect_plane=i))
    for s in strands:
        jobs.append(s["idx"]); kinds.append(dict(strand=True, pool=s["pool"], key=s["key"], stab=s["stab"], az=s["az"],
                                                  parents=s["parents"]))
    G.update(f=f, a=a, aux=(v0.cfm(), v0.slab(), v0.relief()))
    log(f"fitting {len(jobs)} surfaces on {a_.procs} processes")
    with Pool(a_.procs) as pool:
        fits = pool.map(fit_surface, [(m, 1000 + i) for i, m in enumerate(jobs)], chunksize=1)
    for s, kd in zip(fits, kinds):
        s.update(kd)
        if kd["strand"]:
            s["props"].update(cluster_jaccard_median=round(kd["stab"], 2), strand_azimuth=round(kd["az"], 1),
                              strand_pool=kd["pool"])
    plane_stability(f, fits)
    # screening
    keep, rem = [], []
    for s in fits:
        p = s["props"]
        why = None
        if p["dip"] < a.min_dip:
            why = f"final dip {p['dip']} deg < {a.min_dip} (v0 rule, checked on the final fit)"
        elif p["weekday_daytime_fraction"] >= 0.6:
            why = (f"non-earthquake timing (quarry blasts or other signals): {p['weekday_daytime_fraction']:.0%} of events "
                   f"on weekdays 08-17 local time")
        elif p["region"] == "BL" and p["depth_km"] > 25 and round(p["frac_depth_ok"] * p["n_events"]) < 30:
            why = (f"deep Blanco artefact: centroid at {p['depth_km']} km with {round(p['frac_depth_ok'] * p['n_events'])} "
                   f"depth_quality ok events (published Blanco seismicity stops at 10-14 km; v0 removed BL022, BL024 and "
                   f"BL026 for the same reason)")
        if why:
            s["reason"] = why
            rem.append(s)
        else:
            keep.append(s)
    rec = recovery_runs(f)
    log("verifier detection reruns for the recovery test: " + (", ".join(rec) if rec else "none found"))
    p_wd, n_tests = float(f.weekday_daytime.mean()), len(fits)
    for s in keep:
        p = s["props"]
        idx = s["idx"]
        p["kind"] = "steep zone (fault candidate)" if p["dip"] >= 45 else "dipping zone"
        p["stable"] = bool(p.get("cluster_jaccard_median", 0) >= 0.5)
        p["strand"] = bool(s["strand"])
        # detection-level spread: builder's four reruns (strike/dip range) and the verifier's reruns (recovery)
        p["strike_settings_range_width"] = range_width(p.get("strike_settings_range"), circular=True)
        p["dip_settings_range_width"] = range_width(p.get("dip_settings_range"), circular=False)
        J = {name: (np.nan if s["strand"] else recovery_J(f.event_id.values[idx], *run)) for name, run in rec.items()}
        for name in ("sub80", "perturbed", "tierAB"):
            p[f"recovery_J_{name}"] = None if not np.isfinite(J.get(name, np.nan)) else round(float(J[name]), 2)
        nt = sum(np.isfinite(v) for v in J.values())
        nr = sum(v >= 0.5 for v in J.values() if np.isfinite(v))
        p["recovery_tests"] = f"{nr}/{nt}" if nt else ""
        k_wd = int(f.weekday_daytime.values[idx].sum())
        p["weekday_daytime_p"] = float(f"{binom.sf(k_wd - 1, p['n_events'], p_wd):.2g}")
        fl = []
        if p["frac_depth_ok"] < 0.5:
            fl.append(f"depth poorly constrained ({p['frac_depth_ok']:.0%} depth_quality ok): dip and depth extent not reliable")
        if p["region"] == "BL" and p["depth_km"] > 20:
            fl.append("centroid deeper than 20 km, below published Blanco seismicity (10-14 km)")
        if p["region"] in ("EN",) and p["depth_km"] > 25:
            fl.append("centroid deeper than 25 km on unconstrained ridge-region depths")
        fl += catalog_flags(f, idx, p)
        if p["n_vertices_above_surface"] > 0:
            fl.append(f"rectangle above the ground or seafloor at {p['n_vertices_above_surface']} mesh vertices (extrapolation)")
        if k_wd >= 2 * p_wd * p["n_events"]:          # at least twice the fit-set rate (an effect size, not only a p)
            fl.append(f"weekday-daytime excess: {k_wd} of {p['n_events']} events ({k_wd / p['n_events']:.0%}) on weekdays "
                      f"08-17 local time against {p_wd:.0%} in the fit set (binomial p = {p['weekday_daytime_p']:.2e}, "
                      f"Bonferroni level 0.05/{n_tests} = {0.05 / n_tests:.2e}), may include blasts")
        if s["strand"]:
            fl.append("Blanco strand: dip near 90 deg and its narrow interval are set by the map-view band selection "
                      "(half-width 1.5 km), not measured")
        if not p["stable"]:
            fl.append("unstable under detection reruns (Jaccard < 0.5)")
        if nt and nr < nt:
            fl.append(f"not recovered in {nt - nr} of {nt} verifier detection reruns (80 % subsample, error perturbation, "
                      f"tiers A-B only): sigma_total and the strike/dip intervals hold for this member set only")
        p["flags"] = "; ".join(fl)
    keep.sort(key=lambda s: (s["props"]["region"], -s["props"]["n_events"]))
    count, ids = {}, []
    for s in keep:
        r = s["props"]["region"]
        count[r] = count.get(r, 0) + 1
        ids.append(f"v1_{r}{count[r]:03d}")
    G["strand_ids"] = {s["key"]: sid for s, sid in zip(keep, ids) if s["strand"]}
    rem.sort(key=lambda s: (s["props"]["region"], -s["props"]["n_events"]))
    rids = [f"v1x_{i + 1:02d}" for i in range(len(rem))]
    G["strand_ids"].update({s["key"]: rid for s, rid in zip(rem, rids) if s["strand"]})
    removed = pd.DataFrame([dict(removed_id=rid, n_events=s["props"]["n_events"], region=s["props"]["region"],
                                 lon=s["props"]["lon"], lat=s["props"]["lat"], depth_km=s["props"]["depth_km"],
                                 strike=s["props"]["strike"], dip=s["props"]["dip"],
                                 weekday_daytime_fraction=s["props"]["weekday_daytime_fraction"],
                                 frac_depth_ok=s["props"]["frac_depth_ok"], strand=s["strand"], reason=s["reason"])
                            for s, rid in zip(rem, rids)])
    rem_members = pd.concat([pd.DataFrame(dict(event_id=f.event_id.values[s["idx"]], removed_id=rid))
                             for s, rid in zip(rem, rids)]) if rem else pd.DataFrame(columns=["event_id", "removed_id"])
    blast_events = set(f.event_id.values[np.concatenate([s["idx"] for s in rem if s["reason"].startswith("non-earthquake")]
                                                          or [np.array([], int)])])
    log(f"{len(keep)} surfaces kept, {len(rem)} removed by screening")
    member_of = {}
    for k, s in enumerate(keep):
        for e in f.event_id.values[s["idx"]]:
            member_of[e] = k
    msh_rem = [rid for s, rid in zip(rem, rids) if MSH_BOX[0] <= s["props"]["lon"] <= MSH_BOX[1]
               and MSH_BOX[2] <= s["props"]["lat"] <= MSH_BOX[3]]
    for s in keep:
        s["props"]["flags"] = s["props"]["flags"].replace("removed v1x surface under the volcano",
                                                          "removed " + (", ".join(msh_rem) or "surfaces"))
    asg = assign(c, keep, pd.Series(member_of), blast_events)
    asg["removed_surface"] = asg.event_id.map(rem_members.set_index("event_id").removed_id).fillna("")
    asg["catalog_note"] = event_notes(c, asg, keep)
    log("assignment: " + ", ".join(f"{k} {v:,}" for k, v in asg.assignment.value_counts().items()))
    log(f"on_surface_strict {int(asg.on_surface_strict.sum()):,}; catalog notes "
        + ", ".join(f"{k[:40]} {v}" for k, v in asg.catalog_note[asg.catalog_note != ''].value_counts().items()))
    nodes = [dict(id=i, idx=s["idx"], props=s["props"], normal=s["normal"], kept=True, reason="") for i, s in zip(ids, keep)]
    nodes += [dict(id=i, idx=s["idx"], props=s["props"], normal=s["normal"], kept=False, reason=s["reason"])
              for i, s in zip(rids, rem)]
    cmp_pairs, t0, t1 = comparison(c, f, nodes)
    log("v1 vs v0: " + ", ".join(f"{k} {v}" for k, v in t1.change.value_counts().items()) + " | v0: " +
        ", ".join(f"{k} {v}" for k, v in t0.change.value_counts().items()))
    summ = write_all(c, f, keep, ids, asg, removed, pairs_str, merge_log, cmp_pairs, t0, t1, counts)
    rem_members.to_csv(C.OUT / "fault_surfaces_v1_removed_members.csv", index=False)
    allm = pd.read_csv(C.OUT / "fault_surfaces_v1_members.csv")
    assert allm.event_id.is_unique, "an event is a fit member of two surfaces"
    (C.OUT / "fault_surfaces_v1_run.json").write_text(json.dumps(dict(
        fit_set=int(len(f)), detection_planes=len(planes), blanco_parents=len(parents), blanco_strands=len(strands),
        surfaces_fitted=len(fits), kept=len(keep), removed=len(rem), fit_members=int(len(allm)),
        assignment={k: int(v) for k, v in asg.assignment.value_counts().items()},
        on_surface_strict=int(asg.on_surface_strict.sum()),
        catalog_note={k: int(v) for k, v in asg.catalog_note[asg.catalog_note != ""].value_counts().items()},
        members_of_removed_surfaces=int((asg.removed_surface != "").sum()),
        notes=["'none' = compatible with no surface, in a fitted region; Axial Seamount events (never fitted) are "
               f"'{AXIAL_CLASS}'",
               "'assigned' = compatible (z <= 2, d <= 10 km) with exactly one surface, not located on it; "
               "on_surface_strict = fit members plus assigned tier A-C events within 2 km, without floor_limited, "
               "far_only or deep_pass",
               "sigma_total and the strike/dip intervals hold for a fixed member set; see recovery_* and "
               "strike_settings_range_width in the summary for the detection-level spread"]), indent=1))
    # residual calibration sample for the report
    cal = pd.concat([pd.DataFrame(dict(surface_id=sid, resid_km=np.round(s["resid"], 3), sigma_n_km=np.round(s["sn"], 3),
                                       tier=f.tier.values[s["idx"]], depth_quality=f.depth_quality.values[s["idx"]]))
                     for sid, s in zip(ids, keep)])
    cal.to_csv(C.BIG / "work" / "member_residuals.csv.gz", index=False, compression="gzip")
    pd.set_option("display.width", 250)
    cols = ["surface_id", "kind", "stable", "n_events", "lon", "lat", "depth_km", "strike", "dip", "length_km",
            "width_km", "rms_normal_km", "sigma_total_median_km", "cluster_jaccard_median", "change_from_v0", "v0_surfaces"]
    print(summ[[x for x in cols if x in summ]].to_string(index=False))
    log(f"done in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()

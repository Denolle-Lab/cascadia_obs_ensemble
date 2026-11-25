"""Nearest-neighbour clustering of catalog v5.1 with the v5.1 magnitudes (Zaliapin & Ben-Zion method).

The method, parameters and classification rules are those of v4_clusters_zbz.py (catalog v4; read its
docstring for the method); the functions are imported from it so the two runs stay comparable. What is
new for v5.1:

Events
* Catalog v5.1 (NonLinLoc in model v0.3.1) and the v5.1 Route A magnitudes (piece "mag",
  cascadia_catalog_v5.1_magnitudes.csv, column M).
* Left out: rows without a hypocentre (loc_source none, 623) and the 3,313 events flagged as not
  earthquakes (is_earthquake False: ComCat explosions / quarry blasts / other, Texada quarry_blast_likely).
  They are kept in the event table with role "excluded" and the reason.
* floor_limited, far_only and deep_pass events enter through their epicentres: the main run uses
  epicentral (2-D) distance, as in v4. Their depths are not used anywhere: depth ranges, depth
  migration and the 3-D variant d3q use only events with a well-constrained depth (depth_good: tier A-C,
  depth_quality ok, not floor_limited / far_only / deep_pass). Tier D events (epicentre errors of several
  km, one degree of freedom for five-phase events) are in the main analysis set; variant tierABC leaves
  them out of it (they are then attached afterwards, like events below Mc).
* Regions: the nine boxes of v4 applied to the v5.1 epicentres; Mc per box from the v5.1 magnitudes with the
  v4 rule (maximum curvature + 0.2, largest yearly value over years with >= 200 events).

Per family, in addition to the v4 fields: magnitude range of all members, duration of all members, depth
range of the well-constrained depths, migration from the tier A-C members (radial speed, Theil-Sen velocity
vector and azimuth, depth migration) and a sqrt(t) diffusion fit (Shapiro et al. 1997: r = sqrt(4 pi D t);
D90 is the diffusivity of the front that encloses 90 % of the events, and the exponent p of the 90 %
distance envelope against time in log bins tells sqrt(t)-like (p near 0.5) from linear (p near 1)
migration).

Variants (sensitivity): d3 (3-D distance, all depths), d3q (3-D only between two depth_good events), mc+0.3,
mc_best (Mc_best of the mag piece), df_est, df1.3, df2.0, noM_floor, mc_year, eta0_shuf5, tierABC, v41M
(v4.1 magnitudes on the v5.1 locations: separates the location change from the magnitude change).

Steps (each caches its output; rerun skips what exists unless --force):
    main      main run, completeness table, events and families
    shuffle   time-shuffled catalogs and eta0_shuf5 thresholds
    variants  sensitivity variants (one file each on /wd1)
    faults    join with the fault surfaces v1 (piece "faults")
    compare   comparison with the v4 results, known sequences, volcano zoom, setting statistics

Run (from this folder):
    python v51_clusters_zbz.py --steps main,shuffle,variants,faults,compare --workers 14
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr, theilslopes

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from v4_clusters_zbz import (REGIONS, REGION_NAMES, REG_ORDER, RE, Q, mc_maxc, b_aki, xyz,  # noqa: E402
                             gmm1d, tree_metrics, classify, rupture_cap, migration, completeness_table)

ROOT = HERE.parents[1]
CAT = Path("/wd1/mdenolle_data/catalog_v5/v5.1/cascadia_catalog_v5.1.csv")
MAG = Path("/wd1/mdenolle_data/catalog_v5/v5.1_analysis/mag/cascadia_catalog_v5.1_magnitudes.csv")
MCB = Path("/home/mdenolle/wt_a51_mag/data/catalog_v5/v5.1_analysis/mag/mc_b/mc_b_regional.csv")
FAULTS = Path("/home/mdenolle/wt_a51_faults/data/catalog_v5/v5.1_analysis/faults")
V4 = ROOT / "data/catalog_v4/clusters_zbz"
OUT = ROOT / "data/catalog_v5/v5.1_analysis/clusters"
BIG = Path("/wd1/mdenolle_data/catalog_v5/v5.1_analysis/clusters")
T0 = pd.Timestamp("2010-01-01", tz="UTC")
YR = 365.25
KM = 111.195
VARIANTS = ["d3", "d3q", "mc+0.3", "mc_best", "df_est", "df1.3", "df2.0", "noM_floor", "mc_year",
            "eta0_shuf5", "tierABC", "v41M"]
# known events for validation (catalog v4 event IDs = v5.1 IDs; ComCat magnitudes); list of v4_clusters_zbz_report.py
KNOWN = [
    (262, "2010-01-10 Mw 6.5 Gorda", "mainshock-aftershock"),
    (83695, "2014-03-10 Mw 6.8 Gorda", "mainshock-aftershock"),
    (19960, "2011-09-09 Mw 6.4 Vancouver Island", "mainshock-aftershock"),
    (89617, "2014-04-24 Mw 6.5 Vancouver Island", "mainshock-aftershock"),
    (40500, "2012-11-08 Mw 6.1 Explorer plate", "mainshock-aftershock"),
    (117222, "2015-01-28 Mw 5.7 Mendocino fault zone", "mainshock-aftershock"),
    (150165, "2015-06-01 Mw 5.8, 5.4, 5.9 Blanco triplet", "swarm or multiplet"),
    (27442, "2012-02/04 Mw 5.8 and 6.0 Blanco", "swarm or multiplet"),
    (900001, "2016-12-08 Mw 6.6 Mendocino fracture zone (v4.1 addition)", "mainshock-aftershock"),
]
VOLCANOES = {"Mount Baker": (-121.813, 48.777), "Glacier Peak": (-121.113, 48.112),
             "Mount Rainier": (-121.760, 46.853), "Mount St. Helens": (-122.180, 46.200),
             "Mount Adams": (-121.490, 46.206), "Mount Hood": (-121.695, 45.374),
             "Mount Jefferson": (-121.800, 44.674), "Three Sisters": (-121.768, 44.103),
             "Newberry": (-121.229, 43.722), "Crater Lake": (-122.110, 42.940),
             "Medicine Lake": (-121.550, 41.590), "Mount Shasta": (-122.193, 41.409),
             "Lassen Peak": (-121.505, 40.492)}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().eq("true")


# ---------------------------------------------------------------- loading
def load_all() -> pd.DataFrame:
    """All 172,342 catalog rows with the v5.1 magnitudes and the exclusion reason ('' = used)."""
    cols = ["ID", "time", "lat", "lon", "depth", "loc_source", "tier", "region", "h_min", "h_max", "sig_z",
            "depth_quality", "event_flag", "floor_limited", "far_only", "deep_pass", "five_phase", "M"]
    c = pd.read_csv(CAT, usecols=cols, low_memory=False).rename(
        columns={"ID": "event_id", "region": "region_v51", "M": "M_v41"})
    m = pd.read_csv(MAG, usecols=["ID", "M", "M_type", "is_earthquake"]).rename(columns={"ID": "event_id"})
    c = c.merge(m, on="event_id", how="left", validate="1:1")
    for f in ["floor_limited", "far_only", "deep_pass", "five_phase"]:
        c[f] = _bool(c[f])
    c["is_earthquake"] = _bool(c["is_earthquake"])
    c["excluded"] = np.where(c.loc_source.eq("none"), "no_hypocentre",
                             np.where(~c.is_earthquake, "not_earthquake", ""))
    return c


def box_of(lon, lat) -> np.ndarray:
    reg = np.full(len(lon), "ON", dtype=object)
    for key, _, (a, b, c, d) in REGIONS[::-1]:
        s = (lon >= a) & (lon <= b) & (lat >= c) & (lat <= d)
        reg[np.asarray(s)] = key
    return reg


def prepare(c: pd.DataFrame, mcol: str = "M") -> pd.DataFrame:
    df = c[c.excluded.eq("")].copy()
    df["M"] = df[mcol]
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    df["t_yr"] = (df["time"] - T0).dt.total_seconds() / (YR * 86400)
    df["year"] = df["time"].dt.year
    df["region"] = box_of(df.lon.values, df.lat.values)
    df["tierABC"] = df.tier.isin(["A", "B", "C"])
    df["depth_good"] = (df.tierABC & df.depth_quality.eq("ok") & ~df.floor_limited & ~df.far_only
                        & ~df.deep_pass)
    return df.sort_values("t_yr", kind="stable").reset_index(drop=True)


def completeness(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ctab, ytab = completeness_table(df)
    ctab["Mc_year"] = [ytab[ytab.region == r].set_index("year").Mc_maxc.to_dict() for r in ctab.region]
    # Mc_best of the mag piece (v5.1 magnitudes, v5.1 locations, whole period), for variant mc_best
    try:
        mb = pd.read_csv(MCB)
        mb = mb[(mb.version == "C_v51M_v51loc") & (mb.period == "all")].set_index("region")
        ctab["Mc_best"] = ctab.region.map(mb.mc_best)
    except FileNotFoundError:
        ctab["Mc_best"] = np.nan
    bb = [b_aki(df.M.values[df.region.values == r], m)[0] if np.isfinite(m) else np.nan
          for r, m in zip(ctab.region, ctab.Mc_best)]
    ctab["b_at_Mc_best"] = bb
    return ctab, ytab


# ---------------------------------------------------------------- nearest neighbours
_G: dict = {}  # arrays shared with forked workers


def _nn_chunk(rng_):
    """Nearest earlier parent for children rng_[0]:rng_[1] (v4 _nn_chunk plus the depth-quality mask)."""
    a, b_ = rng_
    G = _G
    tc, uc, zc, dfc = G["tc"][a:b_], G["uc"][a:b_], G["zc"][a:b_], G["dfc"][a:b_]
    qc = G["qc"][a:b_] if G["qc"] is not None else None
    tp, up, zp, bmp, qp = G["tp"], G["up"], G["zp"], G["bmp"], G["qp"]
    prefix, rmin, d3 = G["prefix"], G["rmin"], G["d3"]
    regc, regp, capp = G["regc"][a:b_], G["regp"], G["capp"]
    nc = b_ - a
    par = np.full(nc, -1)
    out = np.full((nc, 5), np.nan)
    sub = 256
    for s0 in range(0, nc, sub):
        sl = slice(s0, min(s0 + sub, nc))
        npar = (a + sl.stop) if prefix else np.searchsorted(tp, tc[sl].max(), side="left")
        if npar == 0:
            continue
        dt = tc[sl, None] - tp[None, :npar]
        if prefix:
            bad = np.arange(npar)[None, :] >= np.arange(a + sl.start, a + sl.stop)[:, None]
        else:
            bad = dt <= 0
        dt = np.where(bad, 1.0, np.maximum(dt, 1e-9))
        ch = np.sqrt(((uc[sl, None, :] - up[None, :npar, :]) ** 2).sum(-1))
        r = 2 * RE * np.arcsin(np.clip(ch / 2, 0, 1))
        if d3:
            dz2 = (zc[sl, None] - zp[None, :npar]) ** 2
            if qc is not None:  # 3-D only between two well-constrained depths
                dz2 = np.where(qc[sl, None] & qp[None, :npar], dz2, 0.0)
            r = np.sqrt(r ** 2 + dz2)
        r = np.maximum(r, rmin)
        bad = bad | ((regc[sl, None] != regp[None, :npar]) & (r > capp[None, :npar]))
        dt = np.where(bad, 1.0, dt)
        lr_all = np.log10(r)
        le = np.log10(dt) + dfc[sl, None] * lr_all - bmp[None, :npar]
        le[bad] = np.inf
        k = np.argmin(le, axis=1)
        rows = np.arange(le.shape[0])
        lev = le[rows, k]
        ok = np.isfinite(lev)
        idx = np.arange(sl.start, sl.stop)[ok]
        kk = k[ok]
        par[idx] = kk
        lt = np.log10(dt[rows, k])[ok]
        lr = lr_all[rows, k][ok]
        out[idx, 0] = lev[ok]
        out[idx, 1] = lt - Q * bmp[kk]
        out[idx, 2] = dfc[sl][ok] * lr - (1 - Q) * bmp[kk]
        out[idx, 3] = r[rows, k][ok]
        out[idx, 4] = dt[rows, k][ok]
    return par, out


def nn_search(tc, uc, zc, dfc, tp, up, zp, bmp, rmin, d3, prefix, workers, regc, regp, capp,
              qc=None, qp=None, chunk=2000):
    """As v4_clusters_zbz.nn_search; qc/qp (bool) restrict the depth term of d3 to well-constrained pairs."""
    _G.update(regc=regc, regp=regp, capp=capp, tc=tc, uc=uc, zc=zc, dfc=dfc, tp=tp, up=up, zp=zp, bmp=bmp,
              prefix=prefix, rmin=rmin, d3=d3, qc=qc if qc is None else qc[:len(tc)], qp=qp)
    n = len(tc)
    rngs = [(a, min(a + chunk, n)) for a in range(0, n, chunk)]
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as ex:
        res = list(ex.map(_nn_chunk, rngs))
    return np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res])


# ---------------------------------------------------------------- one run
def select(df, ctab, variant):
    c = ctab.set_index("region")
    mm = df.M.values.copy()
    mc = df.region.map(c.Mc).values.astype(float)
    if variant == "mc+0.3":
        mc = mc + 0.3
    if variant == "mc_best":
        mc = df.region.map(c.Mc_best).values.astype(float)
    if variant == "noM_floor":
        mm = np.where(np.isnan(mm), mc, mm)
    if variant == "mc_year":
        mcy = np.array([c.Mc_year[r].get(y, np.nan) for r, y in zip(df.region, df.year)])
        mc = np.where(np.isfinite(mcy), mcy, mc)
    sel = np.isfinite(mm) & (mm >= mc - 1e-9)
    if variant == "tierABC":
        sel &= df.tierABC.values
    if variant in ("df1.3", "df2.0"):
        dfe = np.full(len(df), float(variant[2:]))
    elif variant == "df_est":
        dfe = df.region.map(c.df_est_clip).values
    else:
        dfe = df.region.map(c.df).values
    be = df.region.map(c.b_at_Mc_best if variant == "mc_best" else c.b).values
    return sel, mm, mc, dfe, be


def build(df, ctab, variant, a, thr_override=None, shuffle_seed=None):
    """v4_clusters_zbz.build for v5.1: same links, families and classification; extra event columns."""
    sel, mm, mcv, dfe, be = select(df, ctab, variant)
    d3 = variant in ("d3", "d3q")
    A = df[sel].copy()
    A["m"] = mm[sel]
    A["dfe"], A["be"] = dfe[sel], be[sel]
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        for r, g in A.groupby("region"):
            A.loc[g.index, "t_yr"] = rng.permutation(g.t_yr.values)
        A = A.sort_values("t_yr", kind="stable")
    A = A.reset_index().rename(columns={"index": "row"})
    u = xyz(A.lat.values, A.lon.values)
    t, z, m = A.t_yr.values, A.depth.values, A.m.values
    q = A.depth_good.values if variant == "d3q" else None
    bm = A.be.values * m
    rc = A.region.map({r: k for k, r in enumerate(REG_ORDER)}).values
    capm = rupture_cap(m, a.rcap_min)
    par, o = nn_search(t, u, z, A.dfe.values, t, u, z, bm, a.rmin, d3, True, a.workers, rc, rc, capm, q, q)
    reg = A.region.values
    thr = {}
    for r in REG_ORDER:
        g = gmm1d(o[reg == r, 0])
        if thr_override is not None:
            g["log_eta0"] = thr_override[r]
        thr[r] = g
    le0 = np.array([thr[r]["log_eta0"] for r in reg])
    cap = rupture_cap(m[np.maximum(par, 0)], a.rcap_min)
    below = np.isfinite(o[:, 0]) & (o[:, 0] < le0)
    linked = below & (o[:, 3] <= cap)
    if shuffle_seed is not None:
        return dict(log_eta=o[:, 0], log_T=o[:, 1], log_R=o[:, 2], region=reg, below=below, linked=linked)
    n = len(A)
    ii = np.where(linked)[0]
    _, lab = connected_components(coo_matrix((np.ones(len(ii)), (ii, par[ii])), shape=(n, n)), directed=False)
    sizes = np.bincount(lab)
    fam_local = np.where(sizes[lab] >= 2, lab, -1)
    role = np.array(["single"] * n, dtype=object)
    fid = np.full(n, -1)
    fam_rows = []
    order = pd.Series(np.arange(n))[fam_local >= 0].groupby(fam_local[fam_local >= 0])
    starts = sorted(((g.values[0], c) for c, g in order), key=lambda x: x[0])
    groups = {c: g.values for c, g in order}
    for F, (_, c) in enumerate(starts, start=1):
        mem = groups[c]
        tm = tree_metrics(mem, par, linked, m)
        ms = np.sort(m[mem])[::-1]
        dM = float(ms[0] - ms[1])
        k = tm["imax_local"]
        frac_before = k / len(mem)
        typ, flags = classify(len(mem), dM, frac_before, tm)
        fid[mem] = F
        if typ in ("swarm", "multiplet"):
            role[mem] = typ
        else:
            role[mem[:k]] = "foreshock"
            role[mem[k + 1:]] = "aftershock"
            role[mem[k]] = "mainshock"
        la, lo = A.lat.values[mem], A.lon.values[mem]
        x = (lo - lo.mean()) * KM * np.cos(np.radians(la.mean()))
        y = (la - la.mean()) * KM
        dc = np.hypot(x, y)
        tday = (t[mem] - t[mem[0]]) * YR
        v, rho, p = migration(tday, x, y)
        regs = pd.Series(reg[mem]).value_counts()
        fam_rows.append(dict(
            family_id=F, region=reg[mem[k]], region_v51=A.region_v51.values[mem[k]],
            regions=";".join(f"{q_}:{w}" for q_, w in regs.items()),
            n=len(mem), Mmax=float(ms[0]), M2=float(ms[1]), dM=dM, Mmin=float(ms[-1]),
            mainshock_id=int(A.event_id.values[mem[k]]),
            start=A.time.values[mem[0]], end=A.time.values[mem[-1]], duration_days=float(tday[-1]),
            duration90_days=float(np.quantile(tday, 0.9)),
            lon=float(lo.mean()), lat=float(la.mean()), lon_main=float(lo[k]), lat_main=float(la[k]),
            depth_main=float(A.depth.values[mem[k]]), depth_main_good=bool(A.depth_good.values[mem[k]]),
            extent_km=float(2 * np.quantile(dc, 0.95)), type=typ,
            S1_no_dominant=flags[0], S2_late_peak=flags[1], S3_chain=flags[2],
            frac_before_largest=frac_before, t_largest_days=float(tday[k]),
            mean_leaf_depth=tm["mean_leaf_depth"], max_depth=tm["max_depth"],
            main_offspring_frac=tm["main_offspring_frac"], mean_branching=tm["mean_branching"],
            migration_km_day=v, migration_rho=rho, migration_p=p,
            migrating=bool(np.isfinite(rho) and rho >= 0.5 and p < 0.01 and v > 0)))
    fam = pd.DataFrame(fam_rows)
    ev = pd.DataFrame(dict(
        event_id=A.event_id.values, region=reg, in_analysis=True, M=m,
        parent_id=np.where(par >= 0, A.event_id.values[np.maximum(par, 0)], -1),
        log_eta=o[:, 0], log_T=o[:, 1], log_R=o[:, 2], nn_dist_km=o[:, 3], nn_dt_days=o[:, 4] * YR,
        linked=linked, family_id=fid, role=role))
    # ---- attach events below Mc / without M (and tier D in variant tierABC)
    O = df[~sel]
    if len(O):
        qo = O.depth_good.values if variant == "d3q" else None
        pp, oo = nn_search(O.t_yr.values, xyz(O.lat.values, O.lon.values), O.depth.values, dfe[~sel],
                           t, u, z, bm, a.rmin, d3, False, a.workers,
                           O.region.map({r: k for k, r in enumerate(REG_ORDER)}).values, rc, capm, qo, q)
        le0o = O.region.map({r: thr[r]["log_eta0"] for r in thr}).values
        capo = rupture_cap(m[np.maximum(pp, 0)], a.rcap_min)
        lk = np.isfinite(oo[:, 0]) & (oo[:, 0] < le0o) & (oo[:, 3] <= capo)
        pf = np.where(pp >= 0, fid[np.maximum(pp, 0)], -1)
        ftype = dict(zip(fam.family_id, fam.type)) if len(fam) else {}
        tmain = df.set_index("event_id").t_yr
        mt = dict(zip(fam.family_id, tmain.reindex(fam.mainshock_id).values)) if len(fam) else {}
        orole = np.where(~lk, "single_belowMc", "linked_to_single_belowMc").astype(object)
        to = O.t_yr.values
        for q_ in np.where(lk & (pf >= 0))[0]:
            ty = ftype[pf[q_]]
            if ty == "swarm":
                orole[q_] = "swarm_belowMc"
            elif ty == "multiplet":
                orole[q_] = "multiplet_belowMc"
            else:
                orole[q_] = "foreshock_belowMc" if to[q_] < mt[pf[q_]] else "aftershock_belowMc"
        ev2 = pd.DataFrame(dict(
            event_id=O.event_id.values, region=O.region.values, in_analysis=False, M=O.M.values,
            parent_id=np.where(pp >= 0, A.event_id.values[np.maximum(pp, 0)], -1),
            log_eta=oo[:, 0], log_T=oo[:, 1], log_R=oo[:, 2], nn_dist_km=oo[:, 3],
            nn_dt_days=oo[:, 4] * YR, linked=lk, family_id=np.where(lk, pf, -1), role=orole))
        ev = pd.concat([ev, ev2], ignore_index=True)
    ftype = dict(zip(fam.family_id, fam.type)) if len(fam) else {}
    ev["family_type"] = ev.family_id.map(ftype).fillna("")
    ev["background"] = (ev.family_id <= 0) & ~ev.linked
    ev["cls"] = np.where(ev.family_id > 0, ev.family_type,
                         np.where(ev.linked, "linked_to_single", "background"))
    regt = pd.DataFrame([dict(region=r, n_analysis=int((reg == r).sum()),
                              Mc=float(np.nanmax(mcv[sel & (df.region.values == r)]))
                              if variant != "mc_year" and (sel & (df.region.values == r)).any() else np.nan,
                              n_below_eta0=int(below[reg == r].sum()),
                              n_cut_by_cap=int((below & ~linked)[reg == r].sum()), **thr[r])
                         for r in REG_ORDER])
    regt = regt.merge(ctab[["region", "b", "df"]], on="region")
    return ev, fam, regt, thr


def summarize(ev, fam, reg):
    rows = []
    for r in REG_ORDER:
        e = ev[(ev.region == r) & ev.in_analysis]
        f = fam[fam.region == r] if len(fam) else fam
        rr = reg[reg.region == r].iloc[0]
        rows.append(dict(
            region=r, n_analysis=len(e), Mc=rr.Mc, b=round(rr.b, 2), df=round(rr.df, 2),
            log_eta0=round(rr.log_eta0, 2), ashman_D=round(rr.ashman_D, 2),
            pct_background=round(100 * (e.role == "single").mean(), 1),
            pct_MA=round(100 * e.role.isin(["mainshock", "foreshock", "aftershock"]).mean(), 1),
            pct_swarm=round(100 * (e.role == "swarm").mean(), 1),
            pct_multiplet=round(100 * (e.role == "multiplet").mean(), 1),
            n_families=len(f), n_MA=int((f.type == "mainshock-aftershock").sum()) if len(f) else 0,
            n_MA_ge5=int(((f.type == "mainshock-aftershock") & (f.n >= 5)).sum()) if len(f) else 0,
            n_swarm=int((f.type == "swarm").sum()) if len(f) else 0,
            n_multiplet=int((f.type == "multiplet").sum()) if len(f) else 0))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- per-family details
def migration_details(s: pd.DataFrame, nmin: int = 10, nmax: int = 3000) -> dict:
    """Migration and sqrt(t) diffusion fit on the members s (time-sorted; tier A-C epicentres)."""
    out = dict(mig_n=len(s))
    if len(s) < nmin:
        return out
    la, lo = s.lat.values, s.lon.values
    x = (lo - lo.mean()) * KM * np.cos(np.radians(la.mean()))
    y = (la - la.mean()) * KM
    t = (s.t_yr.values - s.t_yr.values[0]) * YR
    if np.ptp(t) <= 0:
        return out
    k0 = max(3, len(s) // 10)
    x0, y0 = x[:k0].mean(), y[:k0].mean()
    r = np.hypot(x - x0, y - y0)
    w, vec = np.linalg.eigh(np.cov(np.c_[x, y].T))
    out["axis_az"] = float(np.degrees(np.arctan2(vec[0, 1], vec[1, 1])) % 180)
    out["axis_aspect"] = float(np.sqrt(w[1] / max(w[0], 1e-6)))
    h = s.h_max.values
    out["h_err_med_km"] = float(np.nanmedian(h))
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(s), nmax, replace=False)) if len(s) > nmax else np.arange(len(s))
    rho, p = spearmanr(t, r)
    out.update(mig_rho=float(rho), mig_p=float(p), mig_v_radial_km_day=float(theilslopes(r[idx], t[idx])[0]))
    vx, vy = theilslopes(x[idx], t[idx])[0], theilslopes(y[idx], t[idx])[0]
    out.update(mig_vx_km_day=float(vx), mig_vy_km_day=float(vy), mig_v_dir_km_day=float(np.hypot(vx, vy)),
               mig_az=float(np.degrees(np.arctan2(vx, vy)) % 360))
    zg = s.depth_good.values
    if zg.sum() >= nmin and np.ptp(t[zg]) > 0:
        iz = np.where(zg)[0]
        iz = np.sort(rng.choice(iz, nmax, replace=False)) if len(iz) > nmax else iz
        out["mig_vz_km_day"] = float(theilslopes(s.depth.values[iz], t[iz])[0])
        out["mig_rho_z"] = float(spearmanr(t[iz], s.depth.values[iz])[0])
    r90 = float(np.quantile(r, 0.9))
    out["mig_r90_km"] = r90
    out["mig_resolvable"] = bool(r90 >= 2 * out["h_err_med_km"])
    tt, rr = t[k0:], r[k0:]
    ok = tt > 0
    if ok.sum() >= 5:
        Di = rr[ok] ** 2 / (4 * np.pi * tt[ok])          # km^2/day
        conv = 1e6 / 86400.0                              # km^2/day -> m^2/s
        out["D90_m2s"] = float(np.quantile(Di, 0.9) * conv)
        out["D50_m2s"] = float(np.quantile(Di, 0.5) * conv)
    if ok.sum() >= 12:
        lt = np.log10(tt[ok])
        edges = np.linspace(lt.min(), lt.max() + 1e-9, 9)
        bx, by = [], []
        for e0, e1 in zip(edges[:-1], edges[1:]):
            k = (lt >= e0) & (lt < e1)
            if k.sum() >= 3:
                bx.append(np.median(lt[k]))
                by.append(np.log10(max(np.quantile(rr[ok][k], 0.9), 0.05)))
        bx, by = np.array(bx), np.array(by)
        out["env_nbins"] = len(bx)
        if len(bx) >= 4 and np.ptp(bx) > 0.5:
            pfit, cfit = np.polyfit(bx, by, 1)
            out["env_p"] = float(pfit)
            out["env_rms_free"] = float(np.sqrt(np.mean((by - (pfit * bx + cfit)) ** 2)))
            for tag, sl in [("sqrt", 0.5), ("lin", 1.0)]:
                c0 = np.mean(by - sl * bx)
                out[f"env_rms_{tag}"] = float(np.sqrt(np.mean((by - (sl * bx + c0)) ** 2)))
    mig = bool(np.isfinite(out.get("mig_rho", np.nan)) and out["mig_rho"] >= 0.5 and out["mig_p"] < 0.01
               and out["mig_v_radial_km_day"] > 0 and out["mig_resolvable"])
    out["mig_migrating"] = mig
    # style of a migrating family from the exponent p of its 90 % distance envelope, r90 ~ t^p
    style = ""
    if mig:
        pe = out.get("env_p", np.nan)
        style = ("too few events for the envelope" if not np.isfinite(pe) else
                 "slower than sqrt(t)" if pe < 0.3 else "sqrt(t)-like" if pe < 0.75 else "linear-like")
    out["mig_style"] = style
    return out


def family_details(ev: pd.DataFrame, fam: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    cols = ["event_id", "t_yr", "time", "lat", "lon", "depth", "tier", "tierABC", "depth_good", "h_max"]
    e = ev[ev.family_id > 0][["event_id", "family_id", "M", "in_analysis"]].merge(df[cols], on="event_id")
    rows = []
    for F, g in e.groupby("family_id"):
        g = g.sort_values("t_yr")
        d = dict(family_id=F, n_all=len(g), n_tierABC=int(g.tierABC.sum()), n_tierD=int((g.tier == "D").sum()),
                 n_depth_good=int(g.depth_good.sum()), n_with_M=int(g.M.notna().sum()),
                 Mmin_all=float(g.M.min()) if g.M.notna().any() else np.nan,
                 start_all=g.time.iloc[0], end_all=g.time.iloc[-1],
                 duration_all_days=float((g.t_yr.iloc[-1] - g.t_yr.iloc[0]) * YR))
        z = g.depth[g.depth_good]
        if len(z):
            d.update(depth_good_min=float(z.min()), depth_good_q10=float(z.quantile(0.1)),
                     depth_good_med=float(z.median()), depth_good_q90=float(z.quantile(0.9)),
                     depth_good_max=float(z.max()))
        d.update(depth_all_min=float(g.depth.min()), depth_all_med=float(g.depth.median()),
                 depth_all_max=float(g.depth.max()))
        d.update(migration_details(g[g.tierABC]))
        rows.append(d)
    return fam.merge(pd.DataFrame(rows), on="family_id", how="left")


# ---------------------------------------------------------------- steps
def step_main(c, a):
    df = prepare(c)
    log(f"{len(c)} rows; {len(df)} located earthquakes, {df.M.notna().sum()} with M; "
        f"excluded {c.excluded.value_counts().to_dict()}")
    ctab, ytab = completeness(df)
    ctab.drop(columns="Mc_year").to_csv(OUT / "completeness_b_df.csv", index=False, float_format="%.3f")
    ytab.to_csv(OUT / "completeness_by_year.csv", index=False, float_format="%.2f")
    log("\n" + ctab.drop(columns="Mc_year").to_string())
    ev, fam, reg, thr = build(df, ctab, "main", a)
    log(f"main: {len(fam)} families")
    fam = family_details(ev, fam, df)
    summ = summarize(ev, fam, reg)
    log("\n" + summ.to_string())
    reg.to_csv(OUT / "thresholds_zbz.csv", index=False, float_format="%.4f")
    summ.to_csv(OUT / "regions_zbz.csv", index=False)
    # event table: every catalog row
    dd = df[["event_id", "time", "lat", "lon", "depth", "tier", "region_v51", "depth_good",
             "floor_limited", "far_only", "deep_pass", "five_phase"]].copy()
    dd["time"] = dd.time.dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    ev = ev.merge(dd, on="event_id", how="left")
    ev["excluded"] = ""
    x = c[c.excluded.ne("")]
    exc = pd.DataFrame(dict(event_id=x.event_id.values, region=box_of(x.lon.values, x.lat.values),
                            in_analysis=False, M=x.M.values, parent_id=-1, linked=False, family_id=-1,
                            role="excluded", family_type="", background=False, cls="excluded",
                            time=x.time.values, lat=x.lat.values,
                            lon=x.lon.values, depth=x.depth.values, tier=x.tier.values,
                            region_v51=x.region_v51.values, excluded=x.excluded.values))
    exc.loc[exc.excluded.eq("no_hypocentre"), "region"] = ""
    ev = pd.concat([ev, exc], ignore_index=True).sort_values("event_id")
    BIG.mkdir(parents=True, exist_ok=True)
    ev.to_csv(BIG / "events_zbz_v51.csv.gz", index=False, float_format="%.4f")
    slim = ["event_id", "region", "in_analysis", "M", "parent_id", "log_eta", "log_T", "log_R", "nn_dist_km",
            "nn_dt_days", "linked", "family_id", "family_type", "role", "background", "cls", "excluded"]
    ev[slim].to_csv(OUT / "events_zbz_v51.csv.gz", index=False, float_format="%.4f")
    fam.to_csv(OUT / "families_zbz_v51.csv", index=False, float_format="%.4f")
    json.dump(dict(rmin=a.rmin, rcap_min=a.rcap_min, q=Q, df_main=1.6, catalog=str(CAT), magnitudes=str(MAG),
                   n_rows=len(c), n_used=len(df), excluded=c.excluded.value_counts().to_dict()),
              open(OUT / "run_params.json", "w"), indent=1)
    return df, ctab, thr


def step_shuffle(df, ctab, thr, a):
    srows, shuf_all = [], {}
    for sd in range(a.nshuffle):
        s = build(df, ctab, "main", a, shuffle_seed=100 + sd, thr_override={r: thr[r]["log_eta0"] for r in thr})
        for r in REG_ORDER:
            k = s["region"] == r
            le = s["log_eta"][k]
            g = gmm1d(le)
            srows.append(dict(region=r, seed=sd, n=int(k.sum()),
                              frac_below_eta0=float(np.nanmean(le < thr[r]["log_eta0"])),
                              frac_linked=float(s["linked"][k].mean()),
                              ashman_D=g["ashman_D"], mu1=g["mu1"], mu2=g["mu2"], w1=g["w1"]))
            shuf_all.setdefault(r, []).append(le[np.isfinite(le)])
        if sd == 0:
            np.savez_compressed(OUT / "shuffle_eta_seed0.npz", log_eta=s["log_eta"], log_T=s["log_T"],
                                log_R=s["log_R"], region=s["region"].astype(str))
        log(f"shuffle {sd} done")
    sh = pd.DataFrame(srows)
    sh.to_csv(OUT / "shuffle_test.csv", index=False, float_format="%.4f")
    thr_shuf = {r: float(np.quantile(np.concatenate(v), 0.05)) for r, v in shuf_all.items()}
    reg = pd.read_csv(OUT / "thresholds_zbz.csv")
    reg["log_eta0_shuf5"] = reg.region.map(thr_shuf)
    reg.to_csv(OUT / "thresholds_zbz.csv", index=False, float_format="%.4f")
    log("\n" + sh.groupby("region").mean(numeric_only=True).to_string())


def step_variants(c, df, ctab, a):
    vdir = BIG / "variants"
    vdir.mkdir(parents=True, exist_ok=True)
    main_ev = pd.read_csv(OUT / "events_zbz_v51.csv.gz", usecols=["event_id", "in_analysis", "background"])
    main_bg = main_ev[main_ev.in_analysis].set_index("event_id").background
    regs = pd.read_csv(OUT / "thresholds_zbz.csv")
    thr_shuf = dict(zip(regs.region, regs.log_eta0_shuf5))
    rows = [pd.read_csv(OUT / "regions_zbz.csv").assign(variant="main")]
    for v in [x for x in a.variants.split(",") if x]:
        f = vdir / f"events_{v}.csv.gz"
        fs = vdir / f"summary_{v}.csv"
        if f.exists() and fs.exists() and not a.force:
            rows.append(pd.read_csv(fs))
            continue
        t0 = time.time()
        if v == "v41M":
            dfv = prepare(c, "M_v41")
            ctv, _ = completeness(dfv)
            ctv.drop(columns="Mc_year").to_csv(vdir / "completeness_v41M.csv", index=False, float_format="%.3f")
            ev_v, fam_v, reg_v, _ = build(dfv, ctv, "main", a)
        else:
            ev_v, fam_v, reg_v, _ = build(df, ctab, v, a, thr_override=thr_shuf if v == "eta0_shuf5" else None)
        sv = summarize(ev_v, fam_v, reg_v).assign(variant=v)
        e2 = ev_v[ev_v.in_analysis].set_index("event_id")
        com = main_bg.index.intersection(e2.index)
        same = pd.Series(main_bg[com].values == e2.background[com].values, index=com)
        sv["agree_bg_vs_clustered_with_main"] = sv.region.map(same.groupby(e2.region[com]).mean()).round(3)
        sv["n_common_with_main"] = sv.region.map(same.groupby(e2.region[com]).size())
        ev_v[["event_id", "region", "in_analysis", "M", "family_id", "family_type", "role", "background",
              "linked", "log_eta"]].to_csv(f, index=False, float_format="%.4f")
        fam_v.to_csv(vdir / f"families_{v}.csv", index=False, float_format="%.4f")
        sv.to_csv(fs, index=False)
        rows.append(sv)
        log(f"variant {v}: {time.time() - t0:.0f} s\n" + sv.to_string())
    pd.concat(rows).to_csv(OUT / "sensitivity_zbz.csv", index=False)


def step_faults(ev, fam):
    """Join families with the fault surfaces v1 (fit members and uniquely assigned events)."""
    asg = pd.read_csv(FAULTS / "fault_surfaces_v1_assignment.csv.gz",
                      usecols=["event_id", "assignment", "surface_id"])
    summ = pd.read_csv(FAULTS / "fault_surfaces_v1_summary.csv",
                       usecols=["surface_id", "region", "n_events", "strike", "dip", "kind", "change_from_v0",
                                "v0_surfaces", "lit_match"])
    on = asg[asg.assignment.isin(["fit member", "assigned"])][["event_id", "surface_id", "assignment"]]
    e = ev[ev.excluded.fillna("").eq("")].merge(on, on="event_id", how="left")
    rows = []
    for F, g in e[e.family_id > 0].groupby("family_id"):
        vc = g.surface_id.value_counts()
        fr = (vc / len(g)).round(3)
        rows.append(dict(family_id=F, frac_on_faults=round(g.surface_id.notna().mean(), 3),
                         main_fault=vc.index[0] if len(vc) else "",
                         main_fault_frac=float(fr.iloc[0]) if len(fr) else 0.0,
                         fault_ids=";".join(f"{k}:{v}" for k, v in fr.items())))
    fj = pd.DataFrame(rows)
    fam2 = fam.drop(columns=[c for c in ["frac_on_faults", "main_fault", "main_fault_frac", "fault_ids"]
                             if c in fam]).merge(fj, on="family_id", how="left")
    ea = e[e.surface_id.notna()]
    pf = []
    for sid, g in ea.groupby("surface_id"):
        an = g[g.in_analysis.astype(bool)]
        d = dict(surface_id=sid, n_events=len(g), n_analysis=len(an))
        for cc in ["background", "mainshock-aftershock", "swarm", "multiplet"]:
            d[f"n_{cc}"] = int((an.cls == cc).sum())
        d["pct_background"] = round(100 * d["n_background"] / len(an), 1) if len(an) else np.nan
        d["pct_swarm"] = round(100 * d["n_swarm"] / len(an), 1) if len(an) else np.nan
        d["n_families"] = int(an.family_id[an.family_id > 0].nunique())
        d["n_swarm_families_all"] = int(g[g.family_type.eq("swarm")].family_id.nunique())
        pf.append(d)
    pf = pd.DataFrame(pf).rename(columns={"n_events": "n_events_joined"}).merge(summ, on="surface_id", how="left")
    pf.to_csv(OUT / "faults_v1_cluster_counts.csv", index=False)
    # events of surfaces the faults piece removed for weekday-daytime timing (possible unflagged blasts)
    ndt = asg[asg.assignment.str.startswith("non-earthquake timing")][["event_id"]]
    x = ndt.merge(ev, on="event_id", how="left")
    pd.DataFrame([dict(n=len(x), in_analysis=int(x.in_analysis.astype(bool).sum()),
                       in_families=int((x.family_id > 0).sum()),
                       **{f"cls_{k}": int(v) for k, v in x.cls.value_counts().items()})]).to_csv(
        OUT / "faults_v1_daytime_surfaces_events.csv", index=False)
    return fam2, pf


# ---------------------------------------------------------------- comparison with v4
def cls_of(e):
    return np.where(e.family_id > 0, "family", np.where(e.linked.astype(bool), "linked_to_single", "background"))


def match_families(m4: pd.DataFrame, m5: pd.DataFrame, f4: pd.DataFrame, f5: pd.DataFrame,
                   min_share: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """m4, m5: event_id -> family id (members incl. attached). Status of each v4 and each v5.1 family."""
    j = m4.merge(m5, on="event_id")
    ov = j.groupby(["f4", "f5"]).size().rename("n_shared").reset_index()
    n4, n5 = m4.f4.value_counts(), m5.f5.value_counts()
    ov["n4"], ov["n5"] = ov.f4.map(n4), ov.f5.map(n5)
    ov["sig4"] = ov.n_shared >= np.maximum(2, min_share * ov.n4)     # significant part of the v4 family
    ov["sig5"] = ov.n_shared >= np.maximum(2, min_share * ov.n5)     # significant part of the v5.1 family
    s4 = ov[ov.sig4].groupby("f4").f5.nunique()
    s5 = ov[ov.sig5].groupby("f5").f4.nunique()
    best4 = ov.sort_values("n_shared", ascending=False).drop_duplicates("f4").set_index("f4")
    best5 = ov.sort_values("n_shared", ascending=False).drop_duplicates("f5").set_index("f5")
    tot4 = ov.groupby("f4").n_shared.sum()
    tot5 = ov.groupby("f5").n_shared.sum()
    r4 = []
    for F, nF in n4.items():
        d = dict(f4=F, n4_all=int(nF), frac_in_v51_families=float(tot4.get(F, 0) / nF),
                 n_v51_parts=int(s4.get(F, 0)))
        if F in best4.index:
            b = best4.loc[F]
            d.update(f5_main=int(b.f5), shared_main=int(b.n_shared), n5_main=int(b.n5))
        if d["frac_in_v51_families"] < 0.5:
            st = "dissolved"
        elif d["n_v51_parts"] >= 2:
            st = "split"
        elif s5.get(d["f5_main"], 0) >= 2:
            st = "merged"
        elif d["shared_main"] >= 0.5 * nF and d["shared_main"] >= 0.5 * d["n5_main"]:
            st = "kept"
        elif d["shared_main"] >= 0.5 * nF:
            st = "grown"
        else:
            st = "changed"
        d["status"] = st
        r4.append(d)
    r4 = pd.DataFrame(r4)
    st4 = r4.set_index("f4").status
    r5 = []
    for G, nG in n5.items():
        d = dict(f5=G, n5_all=int(nG), frac_from_v4_families=float(tot5.get(G, 0) / nG),
                 n_v4_parts=int(s5.get(G, 0)))
        if G in best5.index:
            b = best5.loc[G]
            d.update(f4_main=int(b.f4), shared_main=int(b.n_shared), n4_main=int(b.n4))
        if d["n_v4_parts"] == 0:
            st = "new"
        elif d["n_v4_parts"] >= 2:
            st = "merged"
        elif st4.get(d.get("f4_main", -1), "") == "split":
            st = "split_part"
        elif st4.get(d.get("f4_main", -1), "") == "kept":
            st = "kept"
        elif st4.get(d.get("f4_main", -1), "") == "grown":
            st = "grown"
        else:
            st = "changed"
        d["status"] = st
        r5.append(d)
    r5 = pd.DataFrame(r5)
    return ov, r4, r5


def step_compare(c, df, ctab, a):
    ev = pd.read_csv(BIG / "events_zbz_v51.csv.gz", low_memory=False)
    fam = pd.read_csv(OUT / "families_zbz_v51.csv")
    e4 = pd.read_csv(V4 / "events_zbz.csv.gz",
                     usecols=["event_id", "region", "in_analysis", "M", "family_id", "role", "linked", "lat", "lon",
                              "depth"])
    f4 = pd.read_csv(V4 / "families_zbz.csv")
    # ---- event-level transitions
    e4["cls4"] = cls_of(e4)
    e4["type4"] = e4.family_id.map(f4.set_index("family_id").type).fillna("")
    ev["cls5"] = np.where(ev.excluded.fillna("").ne(""), "excluded_" + ev.excluded.fillna(""), cls_of(ev))
    j = e4.merge(ev[["event_id", "cls5", "in_analysis", "family_type", "region", "M"]], on="event_id",
                 how="outer", suffixes=("_4", "_5"))
    j["cls4"] = j.cls4.fillna("not in v4")
    j["cls5"] = j.cls5.fillna("not in v5.1")
    pd.crosstab(j.cls4, j.cls5, margins=True).to_csv(OUT / "v4_vs_v51_event_classes.csv")
    both = j[j.in_analysis_4.eq(True) & j.in_analysis_5.eq(True)]
    tr = pd.crosstab([both.region_4], [both.cls4 + "->" + both.cls5])
    tr.to_csv(OUT / "v4_vs_v51_event_classes_analysis_by_region.csv")
    # analysis-set membership changes (Mc and magnitude changes)
    j["an4"] = j.in_analysis_4.eq(True)
    j["an5"] = j.in_analysis_5.eq(True)
    pd.crosstab(j.region_4.fillna(j.region_5), [j.an4, j.an5]).to_csv(OUT / "v4_vs_v51_analysis_membership.csv")
    # ---- family matching
    m4 = e4[e4.family_id > 0][["event_id", "family_id"]].rename(columns={"family_id": "f4"})
    m5 = ev[ev.family_id > 0][["event_id", "family_id"]].rename(columns={"family_id": "f5"})
    ov, r4, r5 = match_families(m4, m5, f4, fam)
    # where did the members of dissolved v4 families go
    dis = m4[m4.f4.isin(r4[r4.status == "dissolved"].f4)].merge(ev[["event_id", "cls5"]], on="event_id",
                                                                 how="left")
    r4 = r4.merge(f4[["family_id", "region", "type", "n", "Mmax", "start", "lon", "lat", "extent_km",
                      "duration90_days", "mean_leaf_depth"]].rename(columns={"family_id": "f4"}), on="f4", how="left")
    r5 = r5.merge(fam[["family_id", "region", "type", "n", "Mmax", "start", "lon", "lat", "extent_km",
                       "duration90_days", "mean_leaf_depth"]].rename(columns={"family_id": "f5"}), on="f5",
                  how="left")
    tmap = r5.set_index("f5").type
    r4["type_v51_main"] = r4.f5_main.map(tmap)
    r4.to_csv(OUT / "v4_families_status.csv", index=False, float_format="%.3f")
    r5["type_v4_main"] = r5.f4_main.map(f4.set_index("family_id").type)
    r5.to_csv(OUT / "v51_families_status.csv", index=False, float_format="%.3f")
    fam = fam.drop(columns=[c for c in ["v4_status", "v4_family_main", "v4_shared_main"] if c in fam]).merge(
        r5[["f5", "status", "f4_main", "shared_main"]].rename(columns={
            "f5": "family_id", "status": "v4_status", "f4_main": "v4_family_main", "shared_main": "v4_shared_main"}),
        on="family_id", how="left")
    fam.to_csv(OUT / "families_zbz_v51.csv", index=False, float_format="%.4f")
    ov.to_csv(OUT / "v4_v51_family_overlap.csv", index=False)
    dis_counts = dis.cls5.fillna("not in v5.1").value_counts()
    dis_counts.rename("n").to_csv(OUT / "v4_dissolved_members_fate.csv")
    # ---- location change of matched family members
    sh = df[["event_id", "lat", "lon", "depth", "depth_good"]].merge(
        e4[["event_id", "lat", "lon", "depth"]], on="event_id", suffixes=("", "_v4"))
    sh["dh_km"] = np.hypot((sh.lon - sh.lon_v4) * KM * np.cos(np.radians(sh.lat)), (sh.lat - sh.lat_v4) * KM)
    sh["dz_km"] = sh.depth - sh.depth_v4
    sh[["event_id", "dh_km", "dz_km", "depth_good"]].to_csv(BIG / "v4_v51_shift.csv.gz", index=False,
                                                            float_format="%.3f")
    # ---- validation, volcano zoom, setting stats
    validate(ev, fam)
    volcano_local(df, a)
    setting_stats(ev, fam, ctab)


def validate(ev, fam):
    rows = []
    fi = fam.set_index("family_id")
    ev = ev.copy()
    ev["time"] = pd.to_datetime(ev.time, utc=True, format="ISO8601")
    for eid, label, expect in KNOWN:
        e = ev[ev.event_id == eid]
        if not len(e):
            continue
        e = e.iloc[0]
        d = dict(event_id=eid, label=label, expected=expect, region=e.region, M=e.M, role=e.role,
                 family_id=int(e.family_id), excluded=e.excluded)
        if e.family_id > 0:
            f = fi.loc[e.family_id]
            d.update(type=f.type, n=int(f.n), n_all=int(f.n_all), Mmax=f.Mmax, M2=f.M2,
                     duration90_days=round(f.duration90_days, 1), extent_km=round(f.extent_km, 1),
                     mean_leaf_depth=round(f.mean_leaf_depth, 2), regions=f.regions)
        rows.append(d)
    ax = ev[(ev.region == "AX") & ev.in_analysis.astype(bool) & (ev.time >= "2015-04-23") & (ev.time < "2015-05-31")]
    big = ax[ax.family_id > 0].family_id.value_counts().head(1)
    for F, nn in big.items():
        f = fi.loc[F]
        rows.append(dict(event_id=int(f.mainshock_id), label="2015-04-24 Axial Seamount eruption onset",
                         expected="swarm", region="AX", M=f.Mmax, role="largest event", family_id=int(F),
                         type=f.type, n=int(f.n), n_all=int(f.n_all), Mmax=f.Mmax, M2=f.M2,
                         duration90_days=round(f.duration90_days, 1), extent_km=round(f.extent_km, 1),
                         mean_leaf_depth=round(f.mean_leaf_depth, 2), regions=f.regions,
                         note=f"start {f.start}; {100 * nn / len(ax):.0f}% of AX analysis events "
                              f"2015-04-23 to 05-31 in this family"))
    pd.DataFrame(rows).to_csv(OUT / "validation_sequences.csv", index=False)
    vrows = []
    for v, (lo, la) in VOLCANOES.items():
        f = fam[km(fam.lon, fam.lat, lo, la) <= 15]
        e2 = ev[(km(ev.lon, ev.lat, lo, la) <= 15) & ev.in_analysis.astype(bool)]
        if len(e2) == 0:
            continue
        vrows.append(dict(volcano=v, n_events_15km=len(e2), pct_background=round(100 * e2.background.mean(), 1),
                          n_families=len(f), n_swarm=int((f.type == "swarm").sum()),
                          n_MA=int((f.type == "mainshock-aftershock").sum()),
                          n_multiplet=int((f.type == "multiplet").sum()),
                          largest_family_n=int(f.n.max()) if len(f) else 0,
                          largest_family_type=f.sort_values("n").type.iloc[-1] if len(f) else ""))
    pd.DataFrame(vrows).to_csv(OUT / "volcano_families.csv", index=False)


def km(lon, lat, lon0, lat0):
    return np.hypot((lon - lon0) * KM * np.cos(np.radians(lat0)), (lat - lat0) * KM)


def volcano_local(df, a, radius=20.0, names=("Mount St. Helens", "Mount Hood", "Mount Rainier")):
    """Zoomed analysis around volcanoes with a local Mc (as in v4_clusters_zbz_report.volcano_local)."""
    rows, frows = [], []
    for v in names:
        lo, la = VOLCANOES[v]
        sub = df[km(df.lon.values, df.lat.values, lo, la) <= radius]
        mc = mc_maxc(sub.M.values)
        b, _, _ = b_aki(sub.M.values, mc)
        A = sub[sub.M >= mc].sort_values("t_yr").reset_index(drop=True)
        n = len(A)
        if n < 50:
            continue
        u = xyz(A.lat.values, A.lon.values)
        t, z, m = A.t_yr.values, A.depth.values, A.M.values
        zero = np.zeros(n, int)
        cap = rupture_cap(m)
        res = {}
        for tag, tt in [("real", t), ("shuffled", np.sort(np.random.default_rng(5).permutation(t)))]:
            uu = u if tag == "real" else u[np.random.default_rng(6).permutation(n)]
            res[tag] = nn_search(tt, uu, z, np.full(n, 1.6), tt, uu, z, b * m, 0.5, False, True, min(a.workers, 8),
                                 zero, zero, cap)
        par, o = res["real"]
        g = gmm1d(o[:, 0])
        gs = gmm1d(res["shuffled"][1][:, 0])
        linked = np.isfinite(o[:, 0]) & (o[:, 0] < g["log_eta0"]) & (o[:, 3] <= cap[np.maximum(par, 0)])
        ii = np.where(linked)[0]
        _, lab = connected_components(coo_matrix((np.ones(len(ii)), (ii, par[ii])), shape=(n, n)), directed=False)
        sz = np.bincount(lab)
        types = []
        for cc in np.unique(lab[sz[lab] >= 2]):
            mem = np.where(lab == cc)[0]
            tm = tree_metrics(mem, par, linked, m)
            ms = np.sort(m[mem])[::-1]
            typ, _ = classify(len(mem), ms[0] - ms[1], tm["imax_local"] / len(mem), tm)
            types.append(typ)
            zg = A.depth.values[mem][A.depth_good.values[mem]]
            frows.append(dict(volcano=v, n=len(mem), Mmax=ms[0], M2=ms[1], type=typ, start=A.time.values[mem[0]],
                              duration_days=(t[mem[-1]] - t[mem[0]]) * YR, mean_leaf_depth=tm["mean_leaf_depth"],
                              depth_good_med=float(np.median(zg)) if len(zg) else np.nan, n_depth_good=len(zg)))
        ty = pd.Series(types, dtype=object)
        rows.append(dict(volcano=v, radius_km=radius, n_all=len(sub), Mc_local=mc, b_local=round(b, 2), n_above=n,
                         log_eta0=round(g["log_eta0"], 2), ashman_D=round(g["ashman_D"], 2),
                         ashman_D_shuffled=round(gs["ashman_D"], 2),
                         pct_background=round(100 * (sz[lab] == 1).mean(), 1),
                         n_families=len(types), n_swarm=int((ty == "swarm").sum()),
                         n_MA=int((ty == "mainshock-aftershock").sum()), n_multiplet=int((ty == "multiplet").sum())))
    pd.DataFrame(rows).to_csv(OUT / "volcano_local_zbz.csv", index=False)
    pd.DataFrame(frows).to_csv(OUT / "volcano_local_families.csv", index=False, float_format="%.3f")


def setting_stats(ev, fam, ctab):
    rows = []
    mc = ctab.set_index("region").Mc
    ea = ev[ev.in_analysis.astype(bool)]
    off = ea[ea.linked.astype(bool)].groupby("parent_id").size()
    e_fs = ea[ea.role == "foreshock"].groupby("family_id").size()
    for r in REG_ORDER:
        f = fam[fam.region == r]
        f5 = f[f.n >= 5]
        ma = f[f.type == "mainshock-aftershock"]
        band = ea[(ea.region == r) & (ea.M - mc[r] >= 1.0) & (ea.M - mc[r] < 2.0)]
        e_as = band.event_id.map(off).fillna(0)
        e = ea[ea.region == r]
        big = e[e.M >= mc[r] + 1.0]
        f10 = f[f.n_all >= 10]
        rows.append(dict(
            region=r, n_fam_ge5=len(f5),
            pct_swarm_of_ge5=round(100 * (f5.type == "swarm").mean(), 1) if len(f5) else np.nan,
            median_leaf_depth_ge5=round(f5.mean_leaf_depth.median(), 2) if len(f5) else np.nan,
            n_events_Mc1to2=len(band),
            mean_offspring_Mc1to2=round(e_as.mean(), 2) if len(band) else np.nan,
            pct_Mc1to2_no_offspring=round(100 * (e_as == 0).mean(), 1) if len(band) else np.nan,
            n_events_ge_Mc1=len(big), pct_ge_Mc1_single=round(100 * (big.role == "single").mean(), 1) if len(big)
            else np.nan,
            pct_MA_with_foreshocks=round(100 * (ma.family_id.map(e_fs).fillna(0) > 0).mean(), 1) if len(ma)
            else np.nan,
            pct_fam_migrating_ge10=round(100 * f[f.n >= 10].migrating.mean(), 1) if (f.n >= 10).any() else np.nan,
            n_fam_ge10=int((f.n >= 10).sum()),
            n_fam_all_ge10=len(f10),
            pct_fam_all_ge10_migrating_v51=round(100 * f10.mig_migrating.fillna(False).astype(bool).mean(), 1)
            if len(f10) else np.nan))
    pd.DataFrame(rows).to_csv(OUT / "setting_stats.csv", index=False)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steps", default="main,shuffle,variants,faults,compare")
    ap.add_argument("--rmin", type=float, default=0.5, help="distance floor (km)")
    ap.add_argument("--rcap-min", type=float, default=10.0, help="minimum link distance cap (km)")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--nshuffle", type=int, default=3)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--force", action="store_true", help="recompute cached steps")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    BIG.mkdir(parents=True, exist_ok=True)
    steps = a.steps.split(",")
    c = load_all()
    if "main" in steps and (a.force or not (OUT / "families_zbz_v51.csv").exists()):
        step_main(c, a)
    df = prepare(c)
    ctab, _ = completeness(df)
    reg = pd.read_csv(OUT / "thresholds_zbz.csv")
    thr = {r: dict(log_eta0=v) for r, v in zip(reg.region, reg.log_eta0)}
    if "shuffle" in steps and (a.force or "log_eta0_shuf5" not in reg.columns):
        step_shuffle(df, ctab, thr, a)
    if "variants" in steps:
        step_variants(c, df, ctab, a)
    if "faults" in steps:
        ev = pd.read_csv(BIG / "events_zbz_v51.csv.gz", low_memory=False)
        fam = pd.read_csv(OUT / "families_zbz_v51.csv")
        fam, pf = step_faults(ev, fam)
        fam.to_csv(OUT / "families_zbz_v51.csv", index=False, float_format="%.4f")
        log(f"faults: {len(pf)} surfaces with events")
    if "compare" in steps:
        step_compare(c, df, ctab, a)
        log("compare done")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Converged Route A magnitudes (decision D-B091): v4.1 and v5.1 recomputed with the exact
phase19 solver (phase19_solver.py), the gauge of the offshore deployments, comparisons
with the stored (3000-iteration LSQR) values and with ComCat ML and tensor Mw.

Stages (scale/run_magnitude_converged.sh runs them in order; each reads the outputs of
the previous ones and writes under --out, small tables under --small):

  patch      v4.1: the phase19 output of the v4 inputs (exact solver) + the v4.1 patch of
             utils/fix_catalog_v4.py (event 172113 removed; 900001-900003 added as ComCat
             Mw, rows copied from the stored file) -> v4.1/cascadia_catalog_M_routeA_converged.csv
             and its .provenance.json. The stored file is only read.
  near       per-pick tables of the phase19 'near' inversion (both passes redone) for
             v4.1 with the exact and the legacy solver (v5.1: written by v51_magnitude_table.py
             for the exact solver; the stored legacy table is the v5.1 piece's).
  gauge      per ocean-bottom deployment: links to the land network, ComCat anchors, change
             of the mean station term, stiffness of a rigid shift, formal error of its level,
             the smallest eigen-directions of the reduced system, and the misfit change of
             the +delta / -delta test; v4.1 and v5.1.
  compare    converged vs stored by region, magnitude, distance, ML_method; ComCat ML anchors
             and tensor Mw; v5.1 - v4.1 with both converged.
  threads    phase19 'near' ML of v4.1 with 1 and 12 BLAS threads, exact and legacy solvers
             (separate processes), and LSQR (plain and column-scaled) stopped at a series of
             iteration counts against the exact solution.
  reconcile  on the stored pick sets: misfit of the stored, the D-B091 LSQR-to-istop-2 and the exact
             solutions, and their ML differences by region, plus the effect of the 4-MAD rejection
             (full exact chain vs exact on the stored pick set) and the picks it changes, by
             deployment (reconcile_db091.csv, reconcile_misfit.csv, reconcile_rejection_*.csv).
  mcb        Mc / b (v51_mc_b.py outputs on the converged table) against the stored tables.

Usage (from workflow/05_magnitude; Python of the pixi env):
  python mag_converged.py patch|near|gauge|compare|threads|mcb [--out DIR] [--small DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
V4 = "/home/mdenolle/cascadia_obs_ensemble/data/magnitude_v4"
V51D = "/wd1/mdenolle_data/catalog_v5/v5.1_analysis/mag"
V51CAT = "/wd1/mdenolle_data/catalog_v5/v5.1/cascadia_catalog_v5.1.csv"
OUT = "/wd1/mdenolle_data/magnitude_converged"
SMALL = os.path.normpath(os.path.join(HERE, "../../data/magnitude_converged"))
STORED_MCB = os.path.normpath(os.path.join(HERE, "../../data/catalog_v5/v5.1_analysis/mag/mc_b"))
PAPER_MC = "/home/mdenolle/cascadia_obs_ensemble/data/catalog_v4/mc_maps/mc_regional.csv"
REMOVE_V41 = {172113}
ADDED_V41 = {900001, 900002, 900003}
REG_ORDER = ["Axial", "Endeavour", "Explorer", "JdF ridge/plate", "Blanco", "Gorda", "Mendocino",
             "Offshore forearc", "Puget/Georgia", "Cascades", "Other onshore"]
OFF6 = ["Axial", "Endeavour", "Explorer", "JdF ridge/plate", "Blanco", "Gorda"]
COAST = ["Mendocino", "Offshore forearc"]
CI_YEAR = {2011: "Y1", 2012: "Y2", 2013: "Y3", 2014: "Y4"}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", HERE, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def med_iqr(s):
    s = pd.Series(s).dropna()
    if not len(s):
        return pd.Series(dict(n=0, median=np.nan, p25=np.nan, p75=np.nan, mean=np.nan))
    return pd.Series(dict(n=len(s), median=s.median(), p25=s.quantile(.25), p75=s.quantile(.75), mean=s.mean()))


def grp(df, key, col, order=None):
    if isinstance(df[key].dtype, pd.CategoricalDtype):
        df = df.assign(**{key: df[key].cat.rename_categories([str(c) for c in df[key].cat.categories])})
    t = df.groupby(key, observed=True)[col].apply(med_iqr).unstack()
    if order is not None:
        t = t.reindex([o for o in order if o in t.index])
    t["n"] = t["n"].astype(int)
    return t.reset_index()


# ------------------------------------------------------------------ patch (v4.1)
def patch(a):
    d = f"{a.out}/v4.1"
    raw = pd.read_csv(f"{d}/cascadia_catalog_M_routeA_converged_raw.csv")
    stored_f = f"{V4}/cascadia_catalog_M_routeA.csv"
    st = pd.read_csv(stored_f)
    add = st[st.event_id.isin(ADDED_V41)]
    assert len(add) == len(ADDED_V41), add
    out = pd.concat([raw[~raw.event_id.isin(REMOVE_V41 | ADDED_V41)], add[raw.columns.intersection(add.columns)]],
                    ignore_index=True)[raw.columns]
    assert set(out.event_id) == set(st.event_id), "event set differs from the stored v4.1 file"
    fo = f"{d}/cascadia_catalog_M_routeA_converged.csv"
    out.to_csv(fo, index=False)
    prov = json.load(open(f"{d}/cascadia_catalog_M_routeA_converged_raw.provenance.json"))
    prov.update(output=fo, catalog="v4.1", decision="D-B091", git_commit=git_commit(),
                v41_patch=dict(removed=sorted(REMOVE_V41), added_from_stored=sorted(ADDED_V41),
                               rule="utils/fix_catalog_v4.py patch_m66: 172113 removed, 900001-900003 = ComCat Mww"),
                stored_file=stored_f, stored_md5=md5(stored_f), output_md5=md5(fo), n_events=len(out),
                written=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(prov, open(fo.replace(".csv", ".provenance.json"), "w"), indent=1, default=str)
    print(f"wrote {fo}: {len(out):,} events; M_type {out.M_type.value_counts().to_dict()}")


# ------------------------------------------------------------------ near tables
def load_ds(path):
    from phase18_moment_tensor_match import hutton_boore
    ds = pd.read_csv(path, usecols=["arid", "event_id", "station", "phase", "dist_hypo_km", "log10A", "stel_m"])
    ds["m"] = hutton_boore(ds.log10A, ds.dist_hypo_km)
    ds["sp"] = ds.station + "|" + ds.phase
    return ds


def near_tables(ds, solver):
    import phase19_preferred_magnitude as p19
    from v51_magnitude_table import near_inversion
    p19.SOLVER = solver
    near, M, C = near_inversion(ds)
    return near, M, C


def anchor_offset(M, near, anchors):
    g = near.groupby("event_id").size()
    a_ = anchors.join(M[g.reindex(M.index).fillna(0) >= 3].rename("m"), on="event_id", how="inner")
    return float(np.median(a_.ml - a_.m))


def near(a):
    d = f"{a.out}/v4.1"
    ds = load_ds(f"{V4}/amp_distance_dataset_routeA.csv")
    an = pd.read_csv(f"{V4}/route_b_ml_anchors_routeA.csv")[["event_id", "ml"]]
    for solver, tag in (("exact", "converged"), ("legacy3000", "legacy3000")):
        nr, M, C = near_tables(ds, solver)
        off = anchor_offset(M, nr, an)
        nr = nr.assign(offset=off)
        nr.to_parquet(f"{d}/phase19_near_picks_routeA_{tag}.parquet", index=False)
        print(f"v4.1 {solver}: {len(nr):,} picks, offset {off:+.4f}", flush=True)


# ------------------------------------------------------------------ gauge
def deployment(station, stel):
    """Ocean-bottom deployment of a station-epoch id NET.STA@YYYY-MM-DD; 'land' on land."""
    if not (stel < 0):
        return "land"
    net, rest = station.split(".", 1)
    yr = int(rest.split("@")[1][:4])
    if net == "7D":
        return f"7D {CI_YEAR.get(yr, yr)} ({yr}-{(yr + 1) % 100:02d})"
    return f"{net} {yr}"


def subarray(station, stel):
    dep = deployment(station, stel)
    if not dep.startswith("7D"):
        return dep
    code = station.split(".", 1)[1].split("@")[0]
    pre = "".join(ch for ch in code if ch.isalpha())[:-1] or code[:1]   # J06A -> J, FN04C -> FN
    return f"{dep} {pre}"


def gauge_one(nc, nu, anchors, region, label, out_small):
    """nc / nu: converged / unconverged near-pick tables (event_id, sp, station, stel_m, m, C,
    M_ev, res, offset). Returns the per-deployment table and the eigen table."""
    nc = nc.copy()
    stel = nc.groupby("station").stel_m.first()
    nc["dep"] = [deployment(s, stel[s]) for s in nc.station]
    nc["sub"] = [subarray(s, stel[s]) for s in nc.station]
    # land networks (tested against the rest of the land network)
    nc["lnet"] = np.where(nc.dep == "land", nc.station.str.split(".").str[0], "land")
    big_nets = nc[nc.lnet != "land"].groupby("lnet").size()
    nc.loc[nc.lnet.isin(big_nets[big_nets < 1000].index), "lnet"] = "land"   # small networks stay in the reference
    n_pick, phi = len(nc), float((nc.res ** 2).sum())
    n_ev, n_sp = nc.event_id.nunique(), nc.sp.nunique()
    sigma2 = phi / (n_pick - n_ev - n_sp + 1)
    # reduced system S0 = diag(n_j) - B' diag(1/n_e) B (no gauge rows, no damping)
    e, ev_ids = pd.factorize(nc.event_id)
    j, sp_ids = pd.factorize(nc.sp)
    from scipy import sparse
    ne_cnt = np.bincount(e).astype(float)
    B = sparse.coo_matrix((np.ones(n_pick), (e, j)), shape=(len(ev_ids), len(sp_ids))).tocsr()
    S0 = np.diag(np.bincount(j).astype(float)) - (B.T @ sparse.diags(1.0 / ne_cnt) @ B).toarray()
    w, V = np.linalg.eigh(S0)
    keep = w > 1e-3                                   # drop the global shift and isolated terms
    import phase19_solver as ps
    isolated = set(ps.components(ps.build(nc))["terms_outside_main"])   # level set by the damping only
    sp_dep = nc.groupby("sp").dep.first().reindex(sp_ids).to_numpy()
    sp_sub = nc.groupby("sp")["sub"].first().reindex(sp_ids).to_numpy()
    land = sp_dep == "land"
    # unconverged terms and event ML
    Cu = nu.groupby("sp").C.first()
    Cc = nc.groupby("sp").C.first()
    dep_of_sp = nc.groupby("sp").dep.first()
    dCall = Cc - Cu.reindex(Cc.index)
    dCall = dCall[~dCall.index.isin(list(isolated))]
    dCl = float(dCall[(dep_of_sp.reindex(dCall.index) == "land").to_numpy()].mean())
    MLc = nc.groupby("event_id").M_ev.first() + nc.offset.iat[0]
    MLu = nu.groupby("event_id").M_ev.first() + nu.offset.iat[0]
    cnt_e = nc.groupby("event_id").size()
    land_ev = set(nc.event_id[nc.dep == "land"])
    an = anchors.set_index("event_id").ml
    rows = []
    for key, col in (("deployment", "dep"), ("sub-array", "sub"), ("land network", "lnet")):
        for G in sorted(nc[col].unique()):
            if G == "land":
                continue
            inG = nc[col] == G
            terms = np.isin(sp_ids, nc.sp[inG].unique()) & ~np.isin(sp_ids, list(isolated))
            ref = land & ~terms                       # land terms outside the group
            ref_sp = set(sp_ids[ref])
            ref_ev = set(nc.event_id[nc.sp.isin(ref_sp)])
            evG = nc.event_id[inG].unique()
            nG = nc[inG].groupby("event_id").size()
            frac = (nG / cnt_e.reindex(nG.index))
            # stiffness of a rigid shift of the terms of G with the events re-fitted
            v = terms.astype(float)
            k = float(v @ S0 @ v)
            # formal standard error of the mean level of G minus the mean level of land
            u = v / v.sum() - ref / ref.sum()
            c = V[:, keep].T @ u
            se = float(np.sqrt(sigma2 * np.sum(c ** 2 / w[keep])))
            # +delta on the terms of G, -delta on every event with a pick on G (data rows only)
            ev_in = nc.event_id.isin(evG).to_numpy()
            cross = ev_in & ~inG.to_numpy()           # picks of those events on other terms
            outside = (~ev_in) & inG.to_numpy()       # none by construction
            r = nc.res.to_numpy()
            dphi = {f"dphi_naive_{dl}": float(((r[cross] + dl) ** 2 - r[cross] ** 2).sum()) for dl in (0.1, 0.5)}
            dphi_opt = {f"dphi_opt_{dl}": k * dl ** 2 for dl in (0.1, 0.5)}
            ev_land = np.intersect1d(evG, list(ref_ev))
            ev_oth = nc.event_id[cross & (nc.dep != "land").to_numpy()].unique()
            dom = frac[(frac >= 0.5) & (cnt_e.reindex(frac.index) >= 3)].index
            an_near = an.reindex(np.intersect1d(evG, cnt_e[cnt_e >= 3].index)).dropna()
            an_dom = an.reindex(dom).dropna()
            dCG = Cc.reindex(sp_ids[terms]) - Cu.reindex(sp_ids[terms])
            dCr = float(dCall.reindex(list(ref_sp)).mean())       # land reference outside the group
            dC = float(dCG.mean()) - dCr
            rows.append(dict(
                catalog=label, level=key, group=G, n_terms=int(terms.sum()), n_picks=int(inG.sum()),
                n_events=len(evG), n_events_dominated=len(dom),
                n_events_with_land=len(ev_land),
                n_land_picks_of_linked=int((nc.event_id.isin(ev_land) & nc.sp.isin(ref_sp)).sum()),
                n_events_with_other_obs=len(ev_oth),
                stiffness=k, stiffness_per_term=k / max(terms.sum(), 1), se_level=se,
                n_anchors=len(an_near), n_anchors_dominated=len(an_dom),
                anchor_gap_unconv=float((MLu.reindex(an_dom.index) - an_dom).median()) if len(an_dom) else np.nan,
                anchor_gap_conv=float((MLc.reindex(an_dom.index) - an_dom).median()) if len(an_dom) else np.nan,
                dC_minus_land=dC, dC_median_minus_land=float(dCG.median()) - dCr,
                dML_dominated=float((MLc.reindex(dom) - MLu.reindex(dom)).median()) if len(dom) else np.nan,
                rigid_mode_eig=k / (terms.sum() + len(evG)),
                main_region=region.reindex(dom).mode().iat[0] if len(dom) and region.reindex(dom).notna().any() else "",
                **dphi_opt, **dphi, n_cross_picks=int(cross.sum()), outside_check=int(outside.sum())))
    t = pd.DataFrame(rows)
    t["phi"] = phi
    t["sigma"] = np.sqrt(sigma2)
    # eigen-directions: the 25 smallest kept eigenvalues and where their energy sits
    erows = []
    for i in np.flatnonzero(keep)[:25]:
        en = pd.Series(V[:, i] ** 2).groupby(sp_dep).sum().sort_values(ascending=False)
        es = pd.Series(V[:, i] ** 2).groupby(sp_sub).sum().sort_values(ascending=False)
        mean_dep = pd.Series(V[:, i]).groupby(sp_dep).mean()
        erows.append(dict(catalog=label, eigenvalue=w[i], top_deployment=en.index[0], energy_share=en.iat[0],
                          second=en.index[1], second_share=en.iat[1], top_subarray=es.index[0],
                          subarray_share=es.iat[0],
                          coherent=float(abs(mean_dep[en.index[0]]) * np.sqrt((sp_dep == en.index[0]).sum())
                                         / np.sqrt(en.iat[0]))))
    et = pd.DataFrame(erows)
    return t, et, dict(phi=phi, sigma=float(np.sqrt(sigma2)), n_pick=n_pick, n_events=n_ev, n_terms=n_sp,
                       eig_global=[float(x) for x in w[~keep]])


def gauge(a):
    reg = pd.read_csv(V51CAT, usecols=["ID", "region"]).set_index("ID").region
    out = []
    eig = []
    meta = {}
    # v4.1
    nc = pd.read_parquet(f"{a.out}/v4.1/phase19_near_picks_routeA_converged.parquet")
    nu = pd.read_parquet(f"{a.out}/v4.1/phase19_near_picks_routeA_legacy3000.parquet")
    an = pd.read_csv(f"{V4}/route_b_ml_anchors_routeA.csv")[["event_id", "ml"]]
    t, e, m = gauge_one(nc, nu, an, reg, "v4.1", a.small)
    out.append(t); eig.append(e); meta["v4.1"] = m
    # v5.1: converged from the table step, stored legacy from the v5.1 piece
    ds = pd.read_csv(f"{V51D}/amp_distance_dataset_v51.csv", usecols=["station", "stel_m"]).drop_duplicates("station")
    stel = ds.set_index("station").stel_m
    nc = pd.read_parquet(f"{a.out}/v5.1/phase19_near_picks_v51_converged.parquet")
    nu = pd.read_parquet(f"{V51D}/phase19_near_picks_v51.parquet")
    an = pd.read_csv(f"{V51D}/route_b_ml_anchors_v51.csv")[["event_id", "ml"]]
    for x in (nc, nu):
        x["stel_m"] = x.station.map(stel)
    nc["offset"] = anchor_offset(nc.groupby("event_id").M_ev.first(), nc, an)
    nu["offset"] = anchor_offset(nu.groupby("event_id").M_ev.first(), nu, an)
    t, e, m = gauge_one(nc, nu, an, reg, "v5.1", a.small)
    out.append(t); eig.append(e); meta["v5.1"] = m
    t = pd.concat(out)
    t.to_csv(f"{a.small}/gauge_deployments.csv", index=False, float_format="%.4f")
    pd.concat(eig).to_csv(f"{a.small}/gauge_eigen.csv", index=False, float_format="%.4f")
    json.dump(meta, open(f"{a.small}/gauge_meta.json", "w"), indent=1)
    cols = ["group", "n_terms", "n_picks", "n_events", "n_events_with_land", "n_events_with_other_obs", "n_anchors",
            "n_anchors_dominated", "stiffness", "rigid_mode_eig", "se_level", "dphi_opt_0.5", "dphi_naive_0.5",
            "dC_minus_land", "dML_dominated", "anchor_gap_unconv", "anchor_gap_conv", "main_region"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        for lab in ("v4.1", "v5.1"):
            print(lab, meta[lab])
            print(t[(t.catalog == lab) & (t.level.isin(["deployment", "land network"]))][cols].round(3).to_string(index=False))
        print(pd.concat(eig).round(3).to_string(index=False))


# ------------------------------------------------------------------ compare
def tensor_rows(x, label, region):
    """x: event_id, ML, ML_method, Mw_mt (matched tensors). Rows for every event with ML and
    for the 'near' events only (the mapped ML depends on the Theil-Sen mapping fit)."""
    rows = []
    for sub, xx in (("all", x), ("near", x[x.ML_method == "near"])):
        for lo, hi in ((3, 4.5), (4.5, 5.5), (5.5, 7.5)):
            s = xx[(xx.Mw_mt >= lo) & (xx.Mw_mt < hi)]
            rows.append(dict(solution=label, events=sub, Mw_class=f"{lo}-{hi}", n=len(s),
                             ML_minus_Mw=float(np.median(s.ML - s.Mw_mt)) if len(s) else np.nan))
        s = xx[(xx.Mw_mt >= 3) & (xx.Mw_mt < 4.5)]
        off = s.event_id.map(region).isin(OFF6 + COAST)
        for lab, o in (("offshore and coastal labels", off), ("onshore labels", ~off)):
            g = s[o]
            rows.append(dict(solution=label, events=sub, Mw_class=f"3-4.5, {lab}", n=len(g),
                             ML_minus_Mw=float(np.median(g.ML - g.Mw_mt)) if len(g) else np.nan))
    return rows


def anchor_rows(x, label, region):
    """x: event_id, ml (ComCat), ML (ours)."""
    x = x.assign(res=x.ML - x.ml, region=x.event_id.map(region))
    rows = [dict(solution=label, group="region", value=r, **med_iqr(g.res).to_dict(), std=g.res.std())
            for r, g in x.groupby("region")]
    off = x.region.isin(OFF6 + COAST)
    for lab, g in (("offshore and coastal labels", x[off]), ("onshore labels", x[~off]), ("all", x)):
        rows.append(dict(solution=label, group="split", value=lab, **med_iqr(g.res).to_dict(), std=g.res.std()))
    return rows


def compare(a):
    cat = pd.read_csv(V51CAT, usecols=["ID", "region", "dist_min"]).set_index("ID")
    region = cat.region
    res = {}
    # ---- v4.1: converged vs stored
    s4 = pd.read_csv(f"{V4}/cascadia_catalog_M_routeA.csv")
    c4 = pd.read_csv(f"{a.out}/v4.1/cascadia_catalog_M_routeA_converged.csv")
    l4 = pd.read_csv(f"{a.out}/v4.1/cascadia_catalog_M_routeA_legacy3000.csv")
    ds = pd.read_csv(f"{V4}/amp_distance_dataset_routeA.csv", usecols=["event_id", "dist_hypo_km"])
    dmin4 = ds.groupby("event_id").dist_hypo_km.min()
    d = s4.merge(c4, on="event_id", suffixes=("_s", "_c"))
    d["dML"] = d.ML_c - d.ML_s
    d["dM"] = d.M_c - d.M_s
    d["region"] = d.event_id.map(region)
    d["dist_min"] = d.event_id.map(dmin4)
    tabs = {}
    tabs["v41_by_region"] = grp(d[d.ML_s.notna()], "region", "dML", REG_ORDER)
    tabs["v41_by_method"] = grp(d, "ML_method_s", "dML")
    tabs["v41_by_mag"] = grp(d.assign(Mbin=pd.cut(d.ML_s, [-3, 0, 1, 2, 3, 4, 7])), "Mbin", "dML")
    tabs["v41_by_dist"] = grp(d.assign(dbin=pd.cut(d.dist_min, [0, 10, 25, 50, 100, 150, 1000])),
                              "dbin", "dML")
    tabs["v41_by_region_near"] = grp(d[d.ML_method_s == "near"], "region", "dML", REG_ORDER)
    tabs["v41_by_region_mapped"] = grp(d[d.ML_method_s == "mapped"], "region", "dML", REG_ORDER)
    tabs["v41_M_type"] = pd.crosstab(d.M_type_s.fillna("none"), d.M_type_c.fillna("none")).reset_index()
    # internal check: median ML of near events by region and year (deployments change by year)
    yr = pd.to_datetime(d.otime_s, utc=True, format="ISO8601").dt.year
    ry = (d[d.ML_method_s == "near"].assign(year=yr).groupby(["region", "year"])
          .agg(n=("dML", "size"), ML_stored=("ML_s", "median"), ML_converged=("ML_c", "median"),
               dML=("dML", "median")).reset_index())
    ry = ry[ry.n >= 50]
    tabs["v41_region_year"] = ry
    big = d[d.dML.abs() > 1.0]
    tabs["v41_large_changes"] = big[["event_id", "region", "ML_method_s", "n_ML_s", "ML_s", "ML_c", "dML",
                                     "M_type_s", "M_type_c", "dist_min"]].sort_values("dML")
    # legacy rerun (12 threads) vs stored: reproducibility of the 3000-iteration solver
    dl = s4.merge(l4, on="event_id", suffixes=("_s", "_l"))
    dl = dl[dl.ML_method_s == "near"]
    x = (dl.ML_l - dl.ML_s).abs()
    res["legacy_rerun_vs_stored"] = dict(n=len(dl), max=float(x.max()), p99=float(x.quantile(.99)),
                                         p95=float(x.quantile(.95)), median_signed=float((dl.ML_l - dl.ML_s).median()))
    # ComCat anchors and tensors
    an4 = pd.read_csv(f"{V4}/route_b_ml_anchors_routeA.csv")[["event_id", "ml"]]
    mt4 = pd.read_csv(f"{V4}/comcat_mt_matched_v4.csv")[["event_id", "Mw_mt"]].drop_duplicates("event_id")
    arows, trows = [], []
    for lab, f in (("v4.1 stored", s4), ("v4.1 converged", c4)):
        arows += anchor_rows(an4.merge(f[f.ML_method == "near"][["event_id", "ML"]], on="event_id"), lab, region)
        trows += tensor_rows(mt4.merge(f[["event_id", "ML", "ML_method"]], on="event_id"), lab, region)
    # ---- v5.1: converged vs stored (tables)
    s5 = pd.read_csv(f"{V51D}/cascadia_catalog_v5.1_magnitudes.csv", low_memory=False)
    c5 = pd.read_csv(f"{a.out}/v5.1/cascadia_catalog_v5.1_magnitudes_converged.csv", low_memory=False)
    e = s5[["ID", "ML", "M", "M_type", "ML_method", "ML_v4", "M_v4", "region", "offshore"]].merge(
        c5[["ID", "ML", "M", "M_type", "ML_method", "ML_v4", "M_v4"]], on="ID", suffixes=("_s", "_c"))
    e["dML"] = e.ML_c - e.ML_s
    e["dist_min"] = e.ID.map(cat.dist_min)
    tabs["v51_by_region"] = grp(e[e.ML_s.notna()], "region", "dML", REG_ORDER)
    tabs["v51_by_region_near"] = grp(e[e.ML_method_s == "near"], "region", "dML", REG_ORDER)
    tabs["v51_by_region_mapped"] = grp(e[e.ML_method_s == "mapped"], "region", "dML", REG_ORDER)
    tabs["v51_by_mag"] = grp(e.assign(Mbin=pd.cut(e.ML_s, [-3, 0, 1, 2, 3, 4, 7])), "Mbin", "dML")
    tabs["v51_by_dist"] = grp(e.assign(dbin=pd.cut(e.dist_min, [0, 10, 25, 50, 100, 150, 1000])),
                              "dbin", "dML")
    tabs["v51_M_type"] = pd.crosstab(e.M_type_s.fillna("none"), e.M_type_c.fillna("none")).reset_index()
    # v5.1 - v4.1: stored pair vs converged pair (same events, both with ML)
    e["d_stored"] = e.ML_s - e.ML_v4_s
    e["d_conv"] = e.ML_c - e.ML_v4_c
    both = e[e.ML_s.notna() & e.ML_v4_s.notna() & e.ML_c.notna() & e.ML_v4_c.notna()]
    t = both.groupby("region")[["d_stored", "d_conv"]].median().reindex(REG_ORDER)
    t.insert(0, "n", both.groupby("region").size().reindex(REG_ORDER))
    t.loc["all"] = [len(both), both.d_stored.median(), both.d_conv.median()]
    off = both.region.isin(OFF6 + COAST)
    t.loc["offshore and coastal labels"] = [int(off.sum()), both[off].d_stored.median(), both[off].d_conv.median()]
    t.loc["onshore labels"] = [int((~off).sum()), both[~off].d_stored.median(), both[~off].d_conv.median()]
    tabs["v51_minus_v41"] = t.rename_axis("region").reset_index()
    tabs["v51_minus_v41_by_mag"] = (both.assign(Mbin=pd.cut(both.ML_v4_c, [-3, 0, 1, 2, 3, 4, 7]))
                                    .groupby("Mbin", observed=True)[["d_stored", "d_conv"]].median().reset_index()
                                    .assign(Mbin=lambda z: z.Mbin.astype(str)))
    an5 = pd.read_csv(f"{V51D}/route_b_ml_anchors_v51.csv")[["event_id", "ml"]]
    mt5 = pd.read_csv(f"{V51D}/comcat_mt_matched_v51.csv")[["event_id", "Mw_mt"]].drop_duplicates("event_id")
    for lab, f in (("v5.1 stored", s5), ("v5.1 converged", c5)):
        f = f.rename(columns={"ID": "event_id"})
        arows += anchor_rows(an5.merge(f[f.ML_method == "near"][["event_id", "ML"]], on="event_id"), lab, region)
        trows += tensor_rows(mt5.merge(f[["event_id", "ML", "ML_method"]], on="event_id"), lab, region)
    tabs["comcat_anchors"] = pd.DataFrame(arows)
    tabs["tensors"] = pd.DataFrame(trows)
    # phase19 provenance numbers side by side
    pv = []
    for lab, f in (("v4.1 converged", f"{a.out}/v4.1/cascadia_catalog_M_routeA_converged_raw.provenance.json"),
                   ("v4.1 legacy rerun", f"{a.out}/v4.1/cascadia_catalog_M_routeA_legacy3000.provenance.json"),
                   ("v5.1 converged", f"{a.out}/v5.1/cascadia_catalog_M_v51_converged.provenance.json")):
        p = json.load(open(f))
        inv = p["inversions"][-1]
        pv.append(dict(run=lab, solver=p["solver"], istop=inv["istop"], itn=inv["itn"], r_norm=inv["r_norm"],
                       normal_res_rel=inv["normal_res_rel"], anchor_offset=p["anchor_offset"],
                       n_anchors=p["n_anchors"], anchor_std=p["anchor_std"], s_anchor=p["s_anchor"],
                       mapped_intercept=p["mapped_fit"]["intercept"], mapped_slope=p["mapped_fit"]["slope"],
                       mapped_std=p["mapped_fit"]["std"], mw_cal_near=p["mw_cal_offset"].get("near"),
                       mw_cal_mapped=p["mw_cal_offset"].get("mapped"), **{f"M_type_{k}": v for k, v in p["m_type"].items()}))
    tabs["phase19_runs"] = pd.DataFrame(pv)
    for k, v in tabs.items():
        v.to_csv(f"{a.small}/{k}.csv", index=False, float_format="%.4f")
    json.dump(res, open(f"{a.small}/compare_meta.json", "w"), indent=1)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        for k in ("v41_by_region", "v41_by_mag", "v41_by_dist", "v41_by_method", "v41_M_type", "v51_by_region",
                  "v51_minus_v41", "comcat_anchors", "tensors", "phase19_runs"):
            print(f"== {k}\n{tabs[k].round(3).to_string(index=False)}")
    print(res)
    # per-event file of the v4.1 change (small enough for the repo as gz)
    d[["event_id", "region", "ML_method_s", "ML_s", "ML_c", "dML", "M_s", "M_c", "M_type_s", "M_type_c"]].round(3) \
        .to_csv(f"{a.out}/v4.1/v41_converged_vs_stored_events.csv.gz", index=False)


# ------------------------------------------------------------------ threads + convergence curve
def threads_worker(a):
    """One process: phase19 near ML (both passes + anchor offset) of v4.1, unrounded."""
    ds = load_ds(f"{V4}/amp_distance_dataset_routeA.csv")
    an = pd.read_csv(f"{V4}/route_b_ml_anchors_routeA.csv")[["event_id", "ml"]]
    t0 = time.time()
    nr, M, C = near_tables(ds, a.solver)
    off = anchor_offset(M, nr, an)
    keep = nr.groupby("event_id").size()
    ml = (M + off)[keep[keep >= 3].index]
    pd.DataFrame({"event_id": ml.index, "ML": ml.to_numpy()}).to_parquet(a.dest, index=False)
    print(f"{a.solver} threads {os.environ.get('OPENBLAS_NUM_THREADS')}: {len(ml):,} events, offset {off:+.6f}, "
          f"{time.time() - t0:.0f} s", flush=True)


def threads(a):
    d = f"{a.out}/threads"
    os.makedirs(d, exist_ok=True)
    for solver in ("exact", "legacy3000"):
        for nt in (1, 12):
            dest = f"{d}/ml_{solver}_t{nt}.parquet"
            if os.path.exists(dest) and not a.force:
                continue
            env = dict(os.environ, OMP_NUM_THREADS=str(nt), OPENBLAS_NUM_THREADS=str(nt), MKL_NUM_THREADS=str(nt))
            subprocess.run([sys.executable, __file__, "threads-worker", "--solver", solver, "--dest", dest,
                            "--out", a.out], env=env, check=True)
    rows = []
    for solver in ("exact", "legacy3000"):
        x1 = pd.read_parquet(f"{d}/ml_{solver}_t1.parquet").set_index("event_id").ML
        x12 = pd.read_parquet(f"{d}/ml_{solver}_t12.parquet").set_index("event_id").ML
        i = x1.index.intersection(x12.index)
        dd = (x1[i] - x12[i]).abs()
        rows.append(dict(solver=solver, n_events_t1=len(x1), n_events_t12=len(x12), n_common=len(i),
                         max_abs=float(dd.max()), p99_abs=float(dd.quantile(.99)), median_abs=float(dd.median()),
                         share_gt_1e4=float((dd > 1e-4).mean()), share_gt_0p1=float((dd > 0.1).mean())))
    t = pd.DataFrame(rows)
    t.to_csv(f"{a.small}/threads_test.csv", index=False, float_format="%.3e")
    print(t.to_string(index=False))
    convergence_curve(a)


def convergence_curve(a):
    """LSQR (plain, phase19 settings, and column-scaled) stopped at a series of iteration
    counts, on the final (pass-2) v4.1 near pick set of the exact run, against the exact
    solution: max and p99 |ML - ML_exact| of the near events (same anchor offset rule)."""
    import phase19_solver as ps
    nr = pd.read_parquet(f"{a.out}/v4.1/phase19_near_picks_routeA_converged.parquet")
    an = pd.read_csv(f"{V4}/route_b_ml_anchors_routeA.csv")[["event_id", "ml"]]
    s = ps.build(nr)
    xe, _ = ps.solve_exact(s)
    ne = s["ne"]
    cnt = np.bincount(s["e"])
    ok = cnt >= 3

    def ml(x):
        M = pd.Series(x[:ne], index=s["ev_ids"])
        return (M + anchor_offset(M, nr, an)).to_numpy()

    mle = ml(xe)
    rows = []
    for scaled, its in ((False, (500, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 12000)),
                        (True, (50, 100, 200, 300, 400))):
        for it in its:
            x, info = ps.solve_lsqr(s, iter_lim=it, tol=1e-8 if not scaled else 1e-14, check=False, scaled=scaled)
            dd = np.abs(ml(x) - mle)[ok]
            rows.append(dict(solver="LSQR column-scaled" if scaled else "LSQR (phase19 settings)", iter_lim=it,
                             istop=info["istop"], itn=info["itn"], max_abs=float(dd.max()),
                             p99_abs=float(np.quantile(dd, .99)), median_abs=float(np.median(dd)),
                             normal_res_rel=info["normal_res_rel"]))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"{a.small}/convergence_curve.csv", index=False, float_format="%.4e")


# ------------------------------------------------------------------ reconcile with D-B091
def reconcile(a):
    """On the stored pick sets (v4.1 legacy rerun, v5.1 stored near picks): data misfit of the
    unconverged and exact solutions, and the D-B091 'converged' LSQR (plain, atol = btol = 1e-8,
    iter_lim 50000; istop 2) against the exact solution, by region."""
    import phase19_solver as ps
    reg = pd.read_csv(V51CAT, usecols=["ID", "region"]).set_index("ID").region
    rows, mrows = [], []
    for lab, f, anf in (("v4.1", f"{a.out}/v4.1/phase19_near_picks_routeA_legacy3000.parquet",
                         f"{V4}/route_b_ml_anchors_routeA.csv"),
                        ("v5.1", f"{V51D}/phase19_near_picks_v51.parquet", f"{V51D}/route_b_ml_anchors_v51.csv")):
        nu = pd.read_parquet(f)
        an = pd.read_csv(anf)[["event_id", "ml"]]
        s = ps.build(nu)
        ne, n, A, b = s["ne"], s["n"], s["A"], s["b"]
        xe, _ = ps.solve_exact(s)
        xu = np.concatenate([nu.groupby("event_id").M_ev.first().reindex(s["ev_ids"]).to_numpy(),
                             nu.groupby("sp").C.first().reindex(s["sp_ids"]).to_numpy()])
        xl, il = ps.solve_lsqr(s, iter_lim=50000, tol=1e-8, check=False, scaled=False)
        keep = np.bincount(s["e"]) >= 3

        def ml(x):
            M = pd.Series(x[:ne], index=s["ev_ids"])
            return (M + anchor_offset(M, nu, an))[keep]

        for name, x in (("3000 iterations" + (" (rerun, 12 threads)" if lab == "v4.1" else " (stored)"), xu),
                        ("D-B091 LSQR to istop 2", xl), ("exact", xe)):
            r = (b - A @ x)[:n]
            mrows.append(dict(catalog=lab, solution=name, misfit=float(r @ r), r_norm=float(np.sqrt(r @ r)),
                              itn=il["itn"] if name.startswith("D-B091") else None,
                              istop=il["istop"] if name.startswith("D-B091") else None))
        mle, mll, mlu = ml(xe), ml(xl), ml(xu)
        # the full exact chain (exact solver in both passes: its own 4-MAD rejection)
        fc = (f"{a.out}/v4.1/phase19_near_picks_routeA_converged.parquet" if lab == "v4.1"
              else f"{a.out}/v5.1/phase19_near_picks_v51_converged.parquet")
        nf = pd.read_parquet(fc)
        Mf = nf.groupby("event_id").M_ev.first()
        kf = nf.groupby("event_id").size()
        mlf = (Mf + anchor_offset(Mf, nf, an))[kf[kf >= 3].index]
        t = pd.DataFrame({"exact_minus_stored": mle - mlu, "db091_minus_stored": mll - mlu,
                          "exact_minus_db091": mle - mll,
                          "full_chain_minus_stored": mlf.reindex(mlu.index) - mlu,
                          "rejection_effect": mlf.reindex(mlu.index) - mle})
        # pick sets: kept by one chain and rejected by the other, by deployment
        stel = (pd.read_csv(f"{V4}/amp_distance_dataset_routeA.csv", usecols=["station", "stel_m"])
                .drop_duplicates("station").set_index("station").stel_m)
        ku, kf_ = set(nu.arid), set(nf.arid)
        allp = pd.concat([nu[["arid", "station"]], nf[["arid", "station"]]]).drop_duplicates("arid")
        allp["dep"] = [deployment(x, stel.get(x, 0.0)) for x in allp.station]
        allp["only_stored"] = allp.arid.isin(ku - kf_)
        allp["only_exact"] = allp.arid.isin(kf_ - ku)
        pk = allp.groupby("dep")[["only_stored", "only_exact"]].sum().astype(int)
        pk["n_exact"] = nf.assign(dep=[deployment(x, stel.get(x, 0.0)) for x in nf.station]).groupby("dep").size()
        pk.rename_axis("deployment").reset_index().assign(catalog=lab).to_csv(
            f"{a.small}/reconcile_rejection_{lab}.csv", index=False)
        print(lab, "picks kept only by the stored chain / only by the exact chain, by deployment:\n", pk.to_string())
        t["region"] = t.index.map(reg)
        g = t.groupby("region").median().reindex(REG_ORDER)
        g.insert(0, "n", t.groupby("region").size().reindex(REG_ORDER))
        g.loc["all"] = [len(t)] + list(t[["exact_minus_stored", "db091_minus_stored", "exact_minus_db091",
                                         "full_chain_minus_stored", "rejection_effect"]].median())
        g["max_abs_exact_minus_db091"] = t.groupby("region").exact_minus_db091.apply(lambda v: v.abs().max())
        g.loc["all", "max_abs_exact_minus_db091"] = t.exact_minus_db091.abs().max()
        rows.append(g.rename_axis("region").reset_index().assign(catalog=lab))
    r = pd.concat(rows)
    m = pd.DataFrame(mrows)
    r.to_csv(f"{a.small}/reconcile_db091.csv", index=False, float_format="%.4f")
    m.to_csv(f"{a.small}/reconcile_misfit.csv", index=False, float_format="%.3f")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(m.to_string(index=False))
        print(r.round(3).to_string(index=False))


# ------------------------------------------------------------------ Mc / b
def mcb(a):
    new = pd.read_csv(f"{a.out}/mc_b/mc_b_regional.csv")
    old = pd.read_csv(f"{STORED_MCB}/mc_b_regional.csv")
    k = ["version", "region", "period"]
    cols = ["n", "mc_maxc", "mc_mbs", "mc_best", "b", "b_sigma", "b_sigma_boot"]
    cols = [c for c in cols if c in new.columns]
    m = old[k + cols].merge(new[k + cols], on=k, suffixes=("_stored", "_conv"))
    for c in ("mc_best", "mc_maxc", "b"):
        m[f"d_{c}"] = m[f"{c}_conv"] - m[f"{c}_stored"]
    m.to_csv(f"{a.small}/mc_b_change.csv", index=False, float_format="%.3f")
    po = pd.read_csv(f"{a.out}/mc_b/paired_b.csv")
    po.to_csv(f"{a.small}/paired_b_converged.csv", index=False, float_format="%.4f")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        sel = m[(m.period == "all") & m.version.isin(["A_v4M_v4loc", "C_v51M_v51loc"])]
        print(sel[k + ["n_stored", "n_conv", "mc_best_stored", "mc_best_conv", "b_stored", "b_conv", "d_mc_best",
                       "d_b"]].round(2).to_string(index=False))
    if os.path.exists(PAPER_MC):
        p = pd.read_csv(PAPER_MC)
        print("paper mc_regional.csv columns:", list(p.columns)[:20])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["patch", "near", "gauge", "compare", "threads", "threads-worker", "curve",
                                      "reconcile", "mcb"])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--small", default=SMALL)
    ap.add_argument("--solver", default="exact")
    ap.add_argument("--dest", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.small, exist_ok=True)
    {"patch": patch, "near": near, "gauge": gauge, "compare": compare, "threads": threads,
     "threads-worker": threads_worker, "curve": convergence_curve, "reconcile": reconcile, "mcb": mcb}[a.stage](a)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Catalog v5.1: velocity structure (Vp, Vs, Vp/Vs of model v0.3.1, resolved nodes only) against seismicity.

Steps (each writes tables to data/catalog_v5/v5.1_analysis/struct/ and large files to
/wd1/mdenolle_data/catalog_v5/v5.1_analysis/struct/):

  prep       event table (catalog v5.1 + v5.1 magnitudes + clustering + fault surfaces v1 + Slab2) sampled at the
             nearest model node; node table with the resolution masks and strata
  coverage   fraction of events in resolved cells by region, class and depth; what cannot be tested
  t1         swarms vs mainshock-aftershock families: Vp/Vs, Vp, Vs and their changes at the families
  t2         seismicity density and b-value against Vp/Vs and Vp change at fixed depth and coverage
  t3         depth extent of seismicity: a Vp level or the slab?
  t4         fault surfaces v1 against velocity gradients (contrasts)
  t5         region-specific: Blanco segments, Mendocino triple junction, forearc vs slab, Mount St. Helens
  synth      circularity control: the same statistics on checkerboard inversion errors (rec - true) of each tile
  mc         location Monte Carlo (each event's catalog errors) of the headline statistics
  v4         the v4-era counterpart (phase13_tomography.py: dlnVs of an external model at the events), v4 vs v5.1
  figs       figures

Controls for circularity (events and rays share locations; resolution and DWS follow seismicity):
* every comparison is within strata zone x depth node x DWS tercile (coverage-matched anomaly V_res);
* randomized catalogs: event sets of nodes permuted among nodes of the same stratum, family types permuted among
  families of the same stratum;
* block bootstrap over 40 km blocks (the finest resolved checkerboard cell) for intervals;
* the checkerboard inversions of each tile give the size of a velocity error that follows the events (synth).

  python v51_struct.py [--steps prep,coverage,...] [--nboot 2000] [--nperm 2000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v51_struct_common import (BIG, BLANCO_SEGMENTS, CLU, COSREF, EXT_MODELS, FLT, FLT_BIG, KM, LAT0,  # noqa: E402
                               LON0, OUT, RNG_SEED, TOMO_RUNS, ZONES, Model, attach_nodes, b_aki, block_bootstrap,
                               block_of, ci, input_provenance, load_events, log, perm_p, permute_within,
                               slab_depth_fn)

EV = BIG / "events_struct_v51.parquet"
NODES = BIG / "nodes_v031.parquet"
PVARS = ["Vp", "dVp"]
SVARS = ["VpVs", "dVpVs", "Vs", "dVs"]
FAM_TYPES = ["swarm", "mainshock-aftershock", "multiplet"]


def save(df: pd.DataFrame, name: str, big=False, **kw):
    p = (BIG if big else OUT) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    if name.endswith(".parquet"):
        df.to_parquet(p, index=False)
    else:
        df.to_csv(p, index=False, float_format="%.6g", **kw)
    log("wrote", p, len(df))
    return p


def load(model=None):
    ev = pd.read_parquet(EV)
    nodes = pd.read_parquet(NODES)
    return ev, nodes


# ======================================================================== prep
def step_prep(model: Model, a):
    t0 = time.time()
    ev = load_events(model)
    nodes = model.node_table()
    ev.to_parquet(EV, index=False)
    nodes.to_parquet(NODES, index=False)
    log(f"prep: {len(ev)} rows, {ev.located.sum()} located, {int((ev.located & ev.is_earthquake).sum())} located "
        f"earthquakes; resolved nodes P {int(model.mP.sum())}, S {int(model.mS.sum())}, PS {int(model.mPS.sum())}; "
        f"resolved nodes not in rock: {model.n_res_not_rock}  ({time.time() - t0:.0f} s)")
    prov = input_provenance()
    (OUT / "input_commits.json").write_text(json.dumps(prov, indent=1))
    log("inputs:", json.dumps(prov, indent=1))
    keep = nodes[nodes.resP | nodes.resS]
    save(keep.drop(columns=["node"]), "nodes_resolved_v031.csv")
    cols = ["ID", "time", "lon", "lat", "depth", "tier", "depth_good", "depth_quality", "far_only", "review",
            "ml_caution", "region", "box", "zone", "node_zone", "M",
            "Mc_box", "cls", "family_id", "family_type", "assignment", "surface_id", "slab_km", "node_depth",
            "resP", "resPS", "resP8", "resPS8", "dws_p", "dws_s"] + PVARS + SVARS + \
           [v + "_res" for v in PVARS + SVARS]
    save(ev.loc[ev.resP | ev.resPS, cols], "events_in_resolved_cells.csv.gz")


# ======================================================================== coverage
def step_coverage(model: Model, a):
    ev, nodes = load()
    located = ev[ev.located]
    eq = located[located.is_earthquake]
    rows = []

    def add(name, d):
        rows.append(dict(group=name, n=len(d), n_resP=int(d.resP.sum()), n_resPS=int(d.resPS.sum()),
                         n_resP8=int(d.resP8.sum()), n_resPS8=int(d.resPS8.sum()),
                         pct_resP=100 * d.resP.mean() if len(d) else np.nan,
                         pct_resPS=100 * d.resPS.mean() if len(d) else np.nan))
    add("all rows (incl. no hypocentre)", ev)
    add("located (loc_source nll3d/comcat)", located)
    add("located earthquakes", eq)
    add("located earthquakes, depth_good", eq[eq.depth_good])
    add("located earthquakes, M >= Mc of box", eq[eq.M >= eq.Mc_box])
    for c in ["background", "mainshock-aftershock", "swarm", "multiplet", "linked_to_single"]:
        add(f"class: {c}", eq[eq.cls == c])
    for c in sorted(eq.assignment.dropna().unique()):
        add(f"fault assignment: {c}", eq[eq.assignment == c])
    for r, d in eq.groupby("region"):
        add(f"region: {r}", d)
    for t, d in eq.groupby("tier"):
        add(f"tier: {t}", d)
    cov = pd.DataFrame(rows)
    save(cov, "coverage_fraction.csv")
    # by zone and depth node
    z = eq[eq.resP].groupby(["node_zone", "node_depth"]).agg(
        n_resP=("ID", "size"), n_resPS=("resPS", "sum"), n_depth_good=("depth_good", "sum"),
        n_aboveMc=("M", lambda s: int((s >= eq.loc[s.index, "Mc_box"]).sum()))).reset_index()
    nz = nodes[nodes.resP].groupby(["zone", "depth"]).agg(nodes_P=("node", "size"),
                                                          nodes_PS=("resPS", "sum"),
                                                          dws_p_med=("dws_p", "median"),
                                                          res_km_med=("res_p_km", "median")).reset_index()
    nz = nz.rename(columns={"zone": "node_zone", "depth": "node_depth"}).merge(z, on=["node_zone", "node_depth"],
                                                                              how="outer")
    # area and number of independent resolution cells (area / res_km^2)
    cell_km2 = (0.125 * KM * np.cos(np.radians(nodes.lat))) * (0.1 * KM)
    nodes["area_km2"] = cell_km2
    ar = nodes[nodes.resP].groupby(["zone", "depth"]).agg(area_P_km2=("area_km2", "sum"),
                                                          ncell_P=("res_p_km", lambda s: float(
                                                              (cell_km2[s.index] / s ** 2).sum()))).reset_index()
    ars = nodes[nodes.resPS].groupby(["zone", "depth"]).agg(area_PS_km2=("area_km2", "sum"), ncell_PS=(
        "res_s_km", lambda s: float((cell_km2[s.index] / s ** 2).sum()))).reset_index()
    ar = ar.merge(ars, on=["zone", "depth"], how="left").rename(columns={"zone": "node_zone", "depth": "node_depth"})
    nz = nz.merge(ar, on=["node_zone", "node_depth"], how="left")
    save(nz, "coverage_by_zone_depth.csv")
    # where nothing is resolved: located earthquakes by region and depth band
    eq = eq.assign(dband=pd.cut(eq.depth, [-5, 5, 10, 15, 20, 30, 50, 200], right=False).astype(str))
    tab = eq.groupby(["region", "dband"]).agg(n=("ID", "size"), pct_resP=("resP", "mean"),
                                              pct_resPS=("resPS", "mean")).reset_index()
    tab["pct_resP"] *= 100
    tab["pct_resPS"] *= 100
    save(tab, "coverage_region_depthband.csv")
    log(cov.to_string())
    log(nz.to_string())
    # catalog flags of the events in resolved cells, by zone: how many depths are poorly constrained, and which
    # events belong to the waveform-review sets (P1 Mount St Helens deep, P2 Brookings, P3 tensors, P4 deep/far)
    rows = []
    for mask in ("resP", "resPS"):
        for zone, d in eq[eq[mask]].groupby("node_zone"):
            rv = d.review.fillna("")
            mlc = d.ml_caution.fillna("")
            rows.append(dict(mask=mask, zone=zone, n=len(d),
                             pct_tier_D=100 * d.tier.eq("D").mean(),
                             pct_depth_unconstrained=100 * d.depth_quality.eq("unconstrained").mean(),
                             pct_depth_biased_deep=100 * d.depth_quality.eq("biased_deep").mean(),
                             pct_depth_good=100 * d.depth_good.mean(), n_far_only=int(d.far_only.sum()),
                             n_deep_pass=int(d.deep_pass.sum()), n_floor_limited=int(d.floor_limited.sum()),
                             n_review_P1_deep=int(rv.str.contains("P1:deep").sum()),
                             n_review_P1_other=int((rv.str.contains("P1:") & ~rv.str.contains("P1:deep")).sum()),
                             n_review_P2=int(rv.str.contains("P2:").sum()),
                             n_review_P3=int(rv.str.contains("P3:").sum()),
                             n_review_P4_far_only=int(rv.str.contains("P4:far_only").sum()),
                             n_review_P4_deep_pass=int(rv.str.contains("P4:deep_pass").sum()),
                             n_ml_caution=int(mlc.ne("").sum()),
                             tierD_h_max_med_km=float(d.loc[d.tier.eq("D"), "h_max"].median()),
                             sig_z_med_km=float(d.sig_z.median())))
    fl = pd.DataFrame(rows)
    save(fl, "flags_resolved_by_zone.csv")
    log(fl.to_string())


# ======================================================================== t1
def family_table(ev: pd.DataFrame, mask_col: str, vars_: list[str], min_res: int = 3, min_frac: float = 0.5,
                 depth_good_only=False) -> pd.DataFrame:
    e = ev[ev.located & ev.is_earthquake & (ev.family_id >= 0)]
    if depth_good_only:
        e = e[e.depth_good]
    st_col = "stratum_S" if mask_col == "resPS" else "stratum_P"
    g = e.groupby("family_id")
    f = pd.DataFrame(dict(type=g.family_type.first(), n=g.size(), n_res=g[mask_col].sum()))
    r = e[e[mask_col]]
    gr = r.groupby("family_id")
    for v in vars_:
        f[v] = gr[v].median()
        f[v + "_res"] = gr[v + "_res"].median()
    f["stratum"] = gr[st_col].agg(lambda s: s.mode().iloc[0])
    f["block"] = gr.block.agg(lambda s: s.mode().iloc[0])
    f["zone"] = gr.node_zone.agg(lambda s: s.mode().iloc[0])
    f["depth_node"] = gr.node_depth.median()
    f["box"] = g.box.first()
    f["frac_res"] = f.n_res / f.n
    f = f[(f.n_res >= min_res) & (f.frac_res >= min_frac)].reset_index()
    return f


def diff_types(f: pd.DataFrame, v: str, t1="swarm", t2="mainshock-aftershock", how="median"):
    fn = np.median if how == "median" else np.mean
    a, b = f.loc[f.type == t1, v].values, f.loc[f.type == t2, v].values
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return float(fn(a) - fn(b))


def step_t1(model: Model, a):
    ev, nodes = load()
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    fams = {}
    for mask_col, vars_ in (("resPS", SVARS + PVARS), ("resP", PVARS)):
        for dg, mf in ((False, 0.0), (True, 0.0), (False, 0.5)):
            f = family_table(ev, mask_col, vars_, depth_good_only=dg, min_frac=mf)
            key = f"{mask_col}{'_dg' if dg else ''}{'_strict' if mf > 0 else ''}"
            fams[key] = f
            for zone in sorted(f.zone.unique()):
                fz = f[f.zone == zone]
                cnt = fz.type.value_counts().to_dict()
                for v in vars_:
                    for suffix in ("", "_res"):
                        col = v + suffix
                        for t2 in ("mainshock-aftershock", "multiplet"):
                            real = diff_types(fz, col, "swarm", t2)
                            if not np.isfinite(real):
                                continue
                            boot = block_bootstrap(fz.block.values, lambda i: diff_types(fz.iloc[i], col, "swarm", t2),
                                                   n=a.nboot)
                            null = np.array([diff_types(fz.assign(type=permute_within(fz.type.values, fz.stratum.values,
                                                                                      rng)), col, "swarm", t2)
                                             for _ in range(a.nperm)])
                            rows.append(dict(set=key, zone=zone, var=col, contrast=f"swarm - {t2}",
                                             n_swarm=cnt.get("swarm", 0), n_other=cnt.get(t2, 0),
                                             n_blocks=fz.block.nunique(),
                                             med_swarm=fz.loc[fz.type == "swarm", col].median(),
                                             med_other=fz.loc[fz.type == t2, col].median(),
                                             diff=real, ci_lo=ci(boot)[0], ci_hi=ci(boot)[1],
                                             null_lo=ci(null)[0], null_hi=ci(null)[1], p_perm=perm_p(real, null)))
    res = pd.DataFrame(rows)
    save(res, "t1_swarm_vs_ma.csv")
    for k, f in fams.items():
        save(f, f"t1_families_{k}.csv")
    # event-level: share of clustered events (above Mc) that are swarm events, low vs high Vp/Vs_res at Blanco
    e = ev[ev.located & ev.is_earthquake & ev.resPS & (ev.M >= ev.Mc_box)]
    out = []
    for zone, ez in e.groupby("node_zone"):
        for v in ("VpVs_res", "dVpVs_res", "dVp_res", "dVs_res"):
            hi = ez[v] >= 0
            def share(d):
                cl = d[d.cls.isin(FAM_TYPES)]
                return dict(n=len(d), n_clustered=len(cl), swarm_share_clustered=(cl.cls == "swarm").mean()
                            if len(cl) else np.nan, background_share=(d.cls == "background").mean())
            sl, sh = share(ez[~hi]), share(ez[hi])
            def dshare(i, ez=ez, v=v):
                d = ez.iloc[i]
                cl = d[d.cls.isin(FAM_TYPES)]
                h = cl[v] >= 0
                if h.sum() < 5 or (~h).sum() < 5:
                    return np.nan
                return (cl[h].cls == "swarm").mean() - (cl[~h].cls == "swarm").mean()
            real = dshare(np.arange(len(ez)))
            boot = block_bootstrap(ez.block.values, dshare, n=a.nboot)
            out.append(dict(zone=zone, var=v, n_low=sl["n"], n_high=sh["n"], clustered_low=sl["n_clustered"],
                            clustered_high=sh["n_clustered"], swarm_share_low=sl["swarm_share_clustered"],
                            swarm_share_high=sh["swarm_share_clustered"], bg_share_low=sl["background_share"],
                            bg_share_high=sh["background_share"], diff_swarm_share=real, ci_lo=ci(boot)[0],
                            ci_hi=ci(boot)[1]))
    save(pd.DataFrame(out), "t1_event_swarm_share.csv")
    log(res[res["var"].str.contains("VpVs")].to_string())


# ======================================================================== t2
def stratified_spearman(y, x, strata, min_n=6):
    """Fisher-z average (weights n - 3) of within-stratum Spearman correlations."""
    from scipy.stats import spearmanr
    zs, ws = [], []
    for s in np.unique(strata):
        i = strata == s
        if i.sum() < min_n or np.ptp(y[i]) == 0 or np.ptp(x[i]) == 0:
            continue
        r = spearmanr(y[i], x[i]).statistic
        if not np.isfinite(r):
            continue
        zs.append(np.arctanh(np.clip(r, -0.999, 0.999)))
        ws.append(i.sum() - 3)
    if not zs:
        return np.nan
    return float(np.tanh(np.average(zs, weights=ws)))


def node_counts(ev, nodes, mask_col, sel=None):
    e = ev[ev.located & ev.is_earthquake & ev[mask_col]]
    if sel is not None:
        e = e[sel(e)]
    cnt = e.groupby("node").size()
    n = nodes[nodes[mask_col]].copy()
    n["count"] = n.node.map(cnt).fillna(0).values
    return n


def weighted_anom(count, vres, strata=None):
    w = count.sum()
    return float((count * vres).sum() / w) if w > 0 else np.nan


def step_t2(model: Model, a):
    ev, nodes = load()
    rng = np.random.default_rng(RNG_SEED + 2)
    rows = []
    sets = {"all": None, "depth_good": lambda e: e.depth_good, "aboveMc": lambda e: e.M >= e.Mc_box,
            "background": lambda e: e.cls == "background"}
    for mask_col, vars_, st in (("resP", PVARS, "stratum_P"), ("resPS", SVARS, "stratum_S")):
        for sname, sel in sets.items():
            n = node_counts(ev, nodes, mask_col, sel)
            for zone in ZONES:
                nz_ = n[n.zone == zone]
                if len(nz_) < 6 or nz_["count"].sum() == 0:
                    continue
                y = nz_["count"].values.astype(float)
                strata = nz_[st].values
                depth_only = (nz_.zone + nz_.depth.astype(str)).values
                for v in vars_:
                    x = nz_[v + "_res"].values
                    xr_ = nz_[v].values
                    rho = stratified_spearman(y, x, strata)
                    rho_depth = stratified_spearman(y, xr_, depth_only)
                    anom = weighted_anom(y, x)
                    bl = nz_.block.values
                    boot_rho = block_bootstrap(bl, lambda i: stratified_spearman(y[i], x[i], strata[i]), n=a.nboot)
                    boot_an = block_bootstrap(bl, lambda i: weighted_anom(y[i], x[i]), n=a.nboot)
                    # occupancy: each node holding events counts once (how much of the event-weighted value comes
                    # from a few busy nodes)
                    occ = (y > 0).astype(float)
                    boot_occ = block_bootstrap(bl, lambda i: weighted_anom(occ[i], x[i]), n=a.nboot)
                    null_rho, null_an = [], []
                    for _ in range(a.nperm):
                        yp = permute_within(y, strata, rng)
                        null_rho.append(stratified_spearman(yp, x, strata))
                        null_an.append(weighted_anom(yp, x))
                    # sensitivity: DWS quintiles instead of terciles (coverage matched more finely)
                    dws = nz_["dws_p" if mask_col == "resP" else "dws_s"].values
                    q5 = nz_.groupby(["depth"])[["dws_p" if mask_col == "resP" else "dws_s"]].transform(
                        lambda s_: pd.qcut(np.log10(np.maximum(s_, 1e-3)), min(5, max(1, len(s_) // 6)),
                                           labels=False, duplicates="drop")).iloc[:, 0].values
                    st5 = nz_.depth.values * 100 + np.nan_to_num(q5, nan=0)
                    x5 = xr_ - pd.Series(xr_).groupby(st5).transform("mean").values
                    rho5 = stratified_spearman(y, x5, st5)
                    anom5 = weighted_anom(y, x5)
                    rows.append(dict(mask=mask_col, events=sname, zone=zone, var=v, n_nodes=len(nz_),
                                     rho_cov_q5=rho5, event_mean_res_q5=anom5,
                                     n_nodes_with_events=int((y > 0).sum()), n_events=int(y.sum()),
                                     n_blocks=int(pd.Series(bl).nunique()), n_strata=int(pd.Series(strata).nunique()),
                                     rho_depth_only=rho_depth, rho_cov=rho, rho_ci_lo=ci(boot_rho)[0],
                                     rho_ci_hi=ci(boot_rho)[1], rho_null_lo=ci(null_rho)[0],
                                     rho_null_hi=ci(null_rho)[1], rho_p=perm_p(rho, null_rho),
                                     event_mean_res=anom, res_ci_lo=ci(boot_an)[0], res_ci_hi=ci(boot_an)[1],
                                     res_null_lo=ci(null_an)[0], res_null_hi=ci(null_an)[1],
                                     res_p=perm_p(anom, null_an),
                                     occ_mean_res=weighted_anom(occ, x), occ_ci_lo=ci(boot_occ)[0],
                                     occ_ci_hi=ci(boot_occ)[1],
                                     node_sd=float(np.nanstd(x))))
    # variant: the field smoothed laterally (Gaussian sigma 20 km, i.e. the 40 km resolved cell) before the
    # coverage-matched anomaly, so only structure on the resolved scale is compared with the events
    e = ev[ev.located & ev.is_earthquake]
    for mask_col, vars_, ph in (("resP", PVARS, "P"), ("resPS", SVARS, "S")):
        for v in vars_:
            sres = model.smoothed_res(v, ph).ravel()
            n = node_counts(ev, nodes, mask_col)
            for zone in ZONES:
                nz_ = n[n.zone == zone]
                if len(nz_) < 6 or nz_["count"].sum() == 0:
                    continue
                y = nz_["count"].values.astype(float)
                x = sres[nz_.node.values]
                strata = nz_["stratum_P" if ph == "P" else "stratum_S"].values
                bl = nz_.block.values
                anom = weighted_anom(y, x)
                boot = block_bootstrap(bl, lambda i: weighted_anom(y[i], x[i]), n=a.nboot)
                null = [weighted_anom(permute_within(y, strata, rng), x) for _ in range(a.nperm)]
                rows.append(dict(mask=mask_col, events="all, field smoothed 20 km", zone=zone, var=v,
                                 n_nodes=len(nz_), n_nodes_with_events=int((y > 0).sum()), n_events=int(y.sum()),
                                 n_blocks=int(pd.Series(bl).nunique()), event_mean_res=anom,
                                 res_ci_lo=ci(boot)[0], res_ci_hi=ci(boot)[1], res_null_lo=ci(null)[0],
                                 res_null_hi=ci(null)[1], res_p=perm_p(anom, null),
                                 rho_cov=stratified_spearman(y, x, strata), node_sd=float(np.nanstd(x))))
    res = pd.DataFrame(rows)
    save(res, "t2_density.csv")
    log(res[res.events == "all"].to_string())
    # ---- b-value split by the coverage-matched anomaly
    brow = []
    e0 = ev[ev.located & ev.is_earthquake & ev.M.notna() & (ev.M >= ev.Mc_box)].copy()
    e0["excess"] = e0.M - e0.Mc_box
    for mask_col, vars_ in (("resP", PVARS), ("resPS", SVARS)):
        for bgonly in (False, True):
            e = e0[e0[mask_col]]
            if bgonly:
                e = e[e.cls == "background"]
            for zone, ez in e.groupby("node_zone"):
                for v in vars_:
                    x = ez[v + "_res"].values
                    ex = ez.excess.values
                    hi = x >= 0

                    def db(i, x=x, ex=ex):
                        h = x[i] >= 0
                        return b_aki(ex[i][h]) - b_aki(ex[i][~h])
                    real = db(np.arange(len(ez)))
                    if not np.isfinite(real):
                        continue
                    boot = block_bootstrap(ez.block.values, db, n=a.nboot)
                    # null: node values permuted among nodes of the same stratum, events keep their node
                    nd = nodes[nodes[mask_col] & (nodes.zone == zone)]
                    st = "stratum_P" if mask_col == "resP" else "stratum_S"
                    null = []
                    for _ in range(a.nperm):
                        perm = permute_within(nd[v + "_res"].values, nd[st].values, rng)
                        m = dict(zip(nd.node.values, perm))
                        xp = ez.node.map(m).values
                        h = xp >= 0
                        null.append(b_aki(ex[h]) - b_aki(ex[~h]))
                    brow.append(dict(mask=mask_col, events="background" if bgonly else "all above Mc", zone=zone,
                                     var=v, n_low=int((~hi).sum()), n_high=int(hi.sum()), b_low=b_aki(ex[~hi]),
                                     b_high=b_aki(ex[hi]), db_high_minus_low=real, ci_lo=ci(boot)[0],
                                     ci_hi=ci(boot)[1], null_lo=ci(null)[0], null_hi=ci(null)[1],
                                     p_perm=perm_p(real, null), n_blocks=ez.block.nunique(),
                                     mc_boxes=";".join(sorted(ez.box.dropna().unique()))))
    bres = pd.DataFrame(brow)
    save(bres, "t2_bvalue.csv")
    log(bres.to_string())
    # checks of each split (point values only): by Mc box, by depth group of the node, and with Mc + 0.2
    crow = []
    for mask_col, vars_ in (("resP", PVARS), ("resPS", SVARS)):
        for bgonly in (False, True):
            e = e0[e0[mask_col]]
            if bgonly:
                e = e[e.cls == "background"]
            e = e.assign(dgroup=pd.cut(e.node_depth, [-5, 10, 20, 40], labels=["1-9 km", "12-18 km", "29 km"]))
            for zone, ez in e.groupby("node_zone"):
                for v in vars_:
                    for kind, grp in (("all", None), ("box", "box"), ("node depth", "dgroup")):
                        parts = [("all", ez)] if grp is None else list(ez.groupby(grp, observed=True))
                        for key, d in parts:
                            for extra in (0.0, 0.2):
                                dd = d[d.excess >= extra]
                                hi = dd[v + "_res"].values >= 0
                                ex = dd.excess.values - extra
                                crow.append(dict(mask=mask_col, events="background" if bgonly else "all above Mc",
                                                 zone=zone, var=v, split_by=kind, group=str(key), mc_plus=extra,
                                                 n_low=int((~hi).sum()), n_high=int(hi.sum()),
                                                 b_low=b_aki(ex[~hi]), b_high=b_aki(ex[hi]),
                                                 db_high_minus_low=b_aki(ex[hi]) - b_aki(ex[~hi]),
                                                 n_ML=int(dd.M_type.eq("ML").sum()), n_review=int(dd.review.ne("")
                                                                                                    .sum())))
    save(pd.DataFrame(crow), "t2_bvalue_checks.csv")


# ======================================================================== t3
def step_t3(model: Model, a):
    ev, nodes = load()
    e = ev[ev.located & ev.is_earthquake & ev.depth_good].copy()
    e["rv_P1P2"] = e.review.fillna("").str.contains("P1:deep|P2:")
    sl = slab_depth_fn()

    def columns(e):
        g = e.groupby(["iy", "ix"])
        col = pd.DataFrame(dict(n=g.size(), d50=g.depth.median(), d90=g.depth.quantile(0.9),
                                d95=g.depth.quantile(0.95), d10=g.depth.quantile(0.1), dmax=g.depth.max(),
                                n_review_P1deep=g.review.agg(lambda s: int(s.fillna("").str.contains("P1:deep")
                                                                           .sum())),
                                n_review_P2=g.review.agg(lambda s: int(s.fillna("").str.contains("P2:").sum())),
                                box=g.box.agg(lambda s: s.mode().iloc[0] if s.notna().any() else ""))).reset_index()
        col = col[col.n >= a.min_col].copy()
        col["lon"] = model.lon[col.ix]
        col["lat"] = model.lat[col.iy]
        col["zone"] = model.zone2[col.iy, col.ix]
        col["block"] = block_of(col.lon, col.lat)
        col["slab_km"] = sl(col.lon.values, col.lat.values)
        col["d95_minus_slab"] = col.d95 - col.slab_km
        return col
    col = columns(e)

    # Vp at a depth z in a column, from the two resolved nodes that bracket it (adjacent depth nodes only)
    def vp_at(iy, ix, z, var="Vp"):
        k = np.searchsorted(model.dep, z) - 1
        if k < 0 or k + 1 >= len(model.dep):
            return np.nan
        if not (model.mP[k, iy, ix] and model.mP[k + 1, iy, ix]):
            return np.nan
        w = (z - model.dep[k]) / (model.dep[k + 1] - model.dep[k])
        arr = getattr(model, var)
        return float((1 - w) * arr[k, iy, ix] + w * arr[k + 1, iy, ix])
    for q in ("d90", "d95"):
        col[f"Vp_{q}"] = [vp_at(r.iy, r.ix, getattr(r, q)) for r in col.itertuples()]
        col[f"Vp0_{q}"] = [vp_at(r.iy, r.ix, getattr(r, q), "Vp0") for r in col.itertuples()]
    save(col, "t3_columns.csv")
    rng = np.random.default_rng(RNG_SEED + 3)
    rows = []
    # (a) Vp level: spread of Vp(D95) across columns against the same D95 values placed in other columns of the zone
    for q in ("d90", "d95"):
        for zone, cz in col.groupby("zone"):
            ok = cz[np.isfinite(cz[f"Vp_{q}"])]
            if len(ok) < 5:
                rows.append(dict(test=f"Vp level at {q}", zone=zone, n_columns=len(ok),
                                 note="fewer than 5 columns with resolved Vp bracketing the depth"))
                continue
            real_sd = float(np.std(ok[f"Vp_{q}"]))
            null = []
            for _ in range(a.nperm):
                zz = rng.permutation(cz[q].values)
                vals = np.array([vp_at(r.iy, r.ix, z) for r, z in zip(cz.itertuples(), zz)])
                vals = vals[np.isfinite(vals)]
                if len(vals) >= 5:
                    null.append(np.std(vals))
            null = np.array(null)
            boot = block_bootstrap(ok.block.values, lambda i: np.std(ok[f"Vp_{q}"].values[i]), n=a.nboot)
            rows.append(dict(test=f"Vp level at {q}", zone=zone, n_columns=len(ok), n_blocks=ok.block.nunique(),
                             median=float(np.median(ok[f"Vp_{q}"])), sd_real=real_sd, sd_ci_lo=ci(boot)[0],
                             sd_ci_hi=ci(boot)[1], sd_null_med=float(np.median(null)) if len(null) else np.nan,
                             sd_null_lo=ci(null)[0], sd_null_hi=ci(null)[1],
                             p_smaller=float((1 + np.sum(null <= real_sd)) / (1 + len(null))) if len(null) else np.nan,
                             depth_sd=float(np.std(ok[q]))))
    # (b) slab: spread of D95 - slab depth against D95 - slab depth of other columns (slab depths permuted)
    from scipy.stats import spearmanr

    def slab_rows(col, label=""):
        for zone, cz in col.groupby("box"):
            ok = cz[np.isfinite(cz.slab_km)]
            if len(ok) < 5:
                continue
            real = float(np.std(ok.d95 - ok.slab_km))
            null = np.array([np.std(ok.d95.values - rng.permutation(ok.slab_km.values)) for _ in range(a.nperm)])
            boot = block_bootstrap(ok.block.values, lambda i: np.std(ok.d95.values[i] - ok.slab_km.values[i]),
                                   n=a.nboot)
            rows.append(dict(test="slab: sd(D95 - slab)", zone=zone + label, n_columns=len(ok),
                             n_blocks=ok.block.nunique(),
                             median=float(np.median(ok.d95 - ok.slab_km)), sd_real=real, sd_ci_lo=ci(boot)[0],
                             sd_ci_hi=ci(boot)[1], sd_null_med=float(np.median(null)), sd_null_lo=ci(null)[0],
                             sd_null_hi=ci(null)[1], p_smaller=float((1 + np.sum(null <= real)) / (1 + len(null))),
                             depth_sd=float(np.std(ok.d95)),
                             rho_d95_slab=float(spearmanr(ok.d95, ok.slab_km).statistic),
                             frac_d95_below_slab=float((ok.d95 > ok.slab_km).mean())))
    slab_rows(col)
    # variant: the waveform-review sets P1 (Mount St Helens deep events, likely false) and P2 (Brookings 2012, two
    # depth solutions) left out, in the boxes that hold any of them (flagged, not removed from the main result)
    boxes_rv = set(e.loc[e.rv_P1P2, "box"].dropna())
    col_rv = columns(e[~e.rv_P1P2])
    slab_rows(col_rv[col_rv.box.isin(boxes_rv)], " (without review P1/P2)")
    res = pd.DataFrame(rows)
    save(res, "t3_depth_extent.csv")
    log(res.to_string())
    # event position relative to the slab, by box (depth_good events where Slab2 exists)
    e = e[np.isfinite(e.slab_km)]
    e["rel"] = pd.cut(e.depth - e.slab_km, [-200, -10, -3, 3, 10, 300],
                      labels=["upper plate (>10 km above)", "3-10 km above", "within 3 km", "3-10 km below",
                              ">10 km below"])
    t = e.groupby(["box", "rel"], observed=False).size().unstack(fill_value=0)
    t["n"] = t.sum(axis=1)
    save(t.reset_index(), "t3_slab_position_by_box.csv")


# ======================================================================== t4
def step_t4(model: Model, a):
    ev, nodes = load()
    summ = pd.read_csv(FLT / "fault_surfaces_v1_summary.csv")
    vert = pd.read_csv(FLT_BIG / "fault_surfaces_v1_vertices.csv.gz")
    vert = vert[vert.constrained == 1]
    ix, iy, iz, inside = model.node_index(vert.lon.values, vert.lat.values, vert.depth_km.values)
    vert = vert.assign(ix=ix, iy=iy, iz=iz, inside=inside)
    rng = np.random.default_rng(RNG_SEED + 4)
    rows, nulls = [], {}
    for var, mask in (("Vp", model.mP), ("dVp", model.mP), ("VpVs", model.mPS), ("dVpVs", model.mPS)):
        G = model.gradients(var, mask)
        gh = G[:2]
        ghmag = np.hypot(gh[0], gh[1])
        stratum = model.stratum_P if var in ("Vp", "dVp") else model.stratum_S
        for sid, vs in vert[vert.inside].groupby("surface_id"):
            gx = gh[0][vs.iz, vs.iy, vs.ix]
            gy = gh[1][vs.iz, vs.iy, vs.ix]
            ok = np.isfinite(gx) & np.isfinite(gy) & (np.hypot(gx, gy) > 0)
            if ok.sum() < a.min_vert:
                continue
            s = summ[summ.surface_id == sid].iloc[0]
            n = np.array([float(t) for t in s.normal_xyz.split(";")])
            nh = n[:2] / max(np.hypot(*n[:2]), 1e-9)
            gx, gy = gx[ok], gy[ok]
            mag = np.hypot(gx, gy)
            align = np.abs(nh[0] * gx + nh[1] * gy) / mag
            # null orientation: the surface rotated about the vertical by a random angle (same vertices, same gradients)
            th = rng.uniform(0, np.pi, a.nperm)
            null_al = np.array([np.median(np.abs(np.cos(t) * nh[0] * gx - np.sin(t) * nh[1] * gx
                                                 + np.sin(t) * nh[0] * gy + np.cos(t) * nh[1] * gy) / mag) for t in th])
            nulls[(var, sid)] = null_al
            # position: |grad| at the vertices against resolved nodes of the same strata
            sts = stratum[vs.iz.values[ok], vs.iy.values[ok], vs.ix.values[ok]]
            pool_mag = []
            for st in np.unique(sts):
                sel = (stratum == st) & np.isfinite(ghmag)
                pool_mag.append(ghmag[sel])
            pool = np.concatenate(pool_mag) if pool_mag else np.array([])
            nodes_v = np.unique(np.ravel_multi_index((vs.iz.values[ok], vs.iy.values[ok], vs.ix.values[ok]),
                                                     model.shape))
            rows.append(dict(var=var, surface_id=sid, region=s.region, stable=bool(s.stable), strike=s.strike,
                             dip=s.dip, flags=s["flags"] if isinstance(s["flags"], str) else "",
                             n_vertices=int(ok.sum()), n_nodes=len(nodes_v),
                             depth_med=float(np.median(vs.depth_km.values[ok])),
                             align_med=float(np.median(align)), align_null_med=float(np.median(null_al)),
                             align_p_greater=float((1 + np.sum(null_al >= np.median(align))) / (1 + len(null_al))),
                             grad_med=float(np.median(mag)),
                             grad_pool_med=float(np.median(pool)) if len(pool) else np.nan,
                             grad_ratio=float(np.median(mag) / np.median(pool)) if len(pool) else np.nan,
                             contrast_20km=float(np.median(np.abs(nh[0] * gx + nh[1] * gy)) * 20.0),
                             grad_az=float(np.degrees(np.arctan2(np.median(gx), np.median(gy))) % 360)))
    res = pd.DataFrame(rows)
    save(res, "t4_surfaces.csv")
    # pooled statistics over surfaces: joint random-rotation null (every surface rotated independently) and a
    # block bootstrap over surfaces (block of the surface centroid)
    out = []
    cen = summ.set_index("surface_id")
    for var, r in res.groupby("var"):
        for subset, rr in (("all", r), ("stable", r[r.stable])):
            if len(rr) < 3:
                out.append(dict(var=var, subset=subset, n_surfaces=len(rr)))
                continue
            real = rr.align_med.mean()
            null = np.mean(np.vstack([nulls[(var, sid)] for sid in rr.surface_id]), axis=0)
            bl = block_of(cen.loc[rr.surface_id, "lon"].values, cen.loc[rr.surface_id, "lat"].values)
            bt_al = block_bootstrap(bl, lambda i: np.mean(rr.align_med.values[i]), n=a.nboot)
            bt_gr = block_bootstrap(bl, lambda i: np.median(rr.grad_ratio.values[i]), n=a.nboot)
            out.append(dict(var=var, subset=subset, n_surfaces=len(rr), n_blocks=len(np.unique(bl)),
                            align_mean=real, align_ci_lo=ci(bt_al)[0], align_ci_hi=ci(bt_al)[1],
                            align_null_mean=float(np.mean(null)), align_null_lo=ci(null)[0],
                            align_null_hi=ci(null)[1],
                            align_p_greater=float((1 + np.sum(null >= real)) / (1 + len(null))),
                            frac_surfaces_p05=float((rr.align_p_greater < 0.05).mean()),
                            grad_ratio_med=float(rr.grad_ratio.median()), grad_ratio_ci_lo=ci(bt_gr)[0],
                            grad_ratio_ci_hi=ci(bt_gr)[1],
                            contrast_20km_med=float(rr.contrast_20km.median())))
    pooled = pd.DataFrame(out)
    save(pooled, "t4_pooled.csv")
    log(res.to_string())
    log(pooled.to_string())


# ======================================================================== t5
def step_t5(model: Model, a):
    ev, nodes = load()
    rng = np.random.default_rng(RNG_SEED + 5)
    e = ev[ev.located & ev.is_earthquake].copy()
    # ---- Blanco segments
    seg = np.full(len(e), "", dtype=object)
    inb = (e.lat >= 42.4) & (e.lat <= 45.0)
    for k, _, lo, hi in BLANCO_SEGMENTS:
        seg[(inb & (e.lon >= lo) & (e.lon < hi)).values] = k
    e["segment"] = seg
    fam = pd.read_csv(CLU / "families_zbz_v51.csv")
    b = e[e.segment != ""]
    rows = []
    for k, name, lo, hi in BLANCO_SEGMENTS:
        d = b[b.segment == k]
        r = d[d.resPS]
        rp = d[d.resP]
        above = d[d.M >= d.Mc_box]
        cl = above[above.cls.isin(FAM_TYPES)]
        famids = d.loc[d.family_id >= 0, "family_id"].unique()
        ff = fam[fam.family_id.isin(famids) & (fam.n >= 5)]
        # nodes of the segment box at 15 and 18 km (resolved PS), node-weighted
        nd = nodes[nodes.resPS & (nodes.lon >= lo) & (nodes.lon < hi) & (nodes.lat >= 42.4) & (nodes.lat <= 45.0)]
        rows.append(dict(segment=k, name=name, lon_lo=lo, lon_hi=hi, n_events=len(d), n_depth_good=int(d.depth_good.sum()),
                         n_resPS=len(r), pct_resPS=100 * len(r) / max(len(d), 1), n_resP=len(rp),
                         VpVs_events=r.VpVs.median(), VpVs_res_events=r.VpVs_res.median(),
                         dVpVs_events=r.dVpVs.median(), dVp_events=rp.dVp.median(), dVs_events=r.dVs.median(),
                         Vp_events=rp.Vp.median(),
                         VpVs_nodes=nd.VpVs.median(), dVp_nodes=nd.dVp.median(), n_nodes=len(nd),
                         depth_med=d.depth.median(), depth_good_med=d.loc[d.depth_good, "depth"].median(),
                         n_aboveMc=len(above), n_clustered=len(cl),
                         swarm_share=(cl.cls == "swarm").mean() if len(cl) else np.nan,
                         background_share=(above.cls == "background").mean() if len(above) else np.nan,
                         n_fam5=len(ff), swarm_frac_fam5=(ff.type == "swarm").mean() if len(ff) else np.nan,
                         b_value=b_aki((above.M - above.Mc_box).values)))
    segt = pd.DataFrame(rows)
    # bootstrap per segment (blocks within segment) for VpVs and swarm share
    for i, (k, *_r) in enumerate(BLANCO_SEGMENTS):
        d = b[(b.segment == k)]
        r = d[d.resPS]
        if len(r) >= 10:
            bt = block_bootstrap(r.block.values, lambda j: np.median(r.VpVs.values[j]), n=a.nboot)
            segt.loc[i, "VpVs_ci_lo"], segt.loc[i, "VpVs_ci_hi"] = ci(bt)
        above = d[(d.M >= d.Mc_box) & d.cls.isin(FAM_TYPES)]
        if len(above) >= 10:
            bt = block_bootstrap(above.block.values, lambda j: (above.cls.values[j] == "swarm").mean(), n=a.nboot)
            segt.loc[i, "swarm_ci_lo"], segt.loc[i, "swarm_ci_hi"] = ci(bt)
    from itertools import permutations
    from scipy.stats import spearmanr
    st = segt.dropna(subset=["VpVs_events", "swarm_share"])
    corr = []
    for xv in ("VpVs_events", "dVpVs_events", "dVp_events", "dVs_events"):
        for yv in ("swarm_share", "background_share", "b_value", "swarm_frac_fam5"):
            s2 = st.dropna(subset=[xv, yv])
            if len(s2) < 4:
                continue
            r = spearmanr(s2[xv], s2[yv]).statistic
            allp = [spearmanr(s2[xv].values, np.array(p)).statistic for p in permutations(s2[yv].values)]
            corr.append(dict(x=xv, y=yv, n_segments=len(s2), spearman=r,
                             p_exact=float(np.mean(np.abs(allp) >= abs(r) - 1e-12))))
    save(segt, "t5_blanco_segments.csv")
    save(pd.DataFrame(corr), "t5_blanco_segment_corr.csv")
    log(segt.to_string())
    log(pd.DataFrame(corr).to_string())
    # ---- position relative to the slab in resolved P cells: upper plate / interface / slab, coverage-matched dVp
    out = []
    ep = e[e.resP & np.isfinite(e.slab_km)].copy()
    ep["rel"] = np.select([ep.depth < ep.slab_km - 3, ep.depth > ep.slab_km + 3], ["upper plate", "slab"],
                          "within 3 km of slab top")
    for zone, ez in ep.groupby("node_zone"):
        for rel, d in ez.groupby("rel"):
            bt = block_bootstrap(d.block.values, lambda j: np.mean(d.dVp_res.values[j]), n=a.nboot) if len(d) > 5 \
                else np.array([np.nan])
            out.append(dict(zone=zone, rel=rel, n=len(d), n_depth_good=int(d.depth_good.sum()),
                            depth_med=d.depth.median(), slab_med=d.slab_km.median(),
                            dVp_res_mean=d.dVp_res.mean(), ci_lo=ci(bt)[0], ci_hi=ci(bt)[1],
                            dVp_mean=d.dVp.mean(), Vp_med=d.Vp.median(), n_blocks=d.block.nunique()))
    save(pd.DataFrame(out), "t5_slab_position_dvp.csv")
    log(pd.DataFrame(out).to_string())
    # ---- Mount St. Helens box and the Mendocino triple junction: coverage-matched dVp at events vs randomized
    boxes = {"MSH (46.0-46.45 N, 122.4-122.0 W)": (-122.45, -122.0, 46.0, 46.45),
             "SW Washington outside MSH box": None,
             "MTJ 29 km north of 40.45 N": (-126.4, -120, 40.45, 42.0),
             "MTJ 29 km south of 40.45 N": (-126.4, -120, 38.0, 40.45)}
    out = []
    for name, bx in boxes.items():
        if bx is None:
            d = e[e.resP & (e.node_zone == "WA") & ~((e.lon >= -122.45) & (e.lon <= -122.0) & (e.lat >= 46.0)
                                                        & (e.lat <= 46.45))]
        else:
            d = e[e.resP & (e.lon >= bx[0]) & (e.lon < bx[1]) & (e.lat >= bx[2]) & (e.lat < bx[3])]
        if len(d) < 10:
            out.append(dict(area=name, n=len(d)))
            continue
        for dep, dd in d.groupby("node_depth"):
            if len(dd) < 10:
                continue
            # randomized catalog: each occupied node's events moved to a random resolved node of the same stratum
            st = dd.stratum_P.values
            cnt = dd.groupby("node").size()
            nd = nodes[nodes.resP & nodes.stratum_P.isin(np.unique(st))]
            real = dd.dVp_res.mean()
            null = []
            for _ in range(a.nperm):
                tot, s = 0.0, 0
                for node_id, c in cnt.items():
                    stn = nodes.at[node_id, "stratum_P"]
                    pool = nd[nd.stratum_P == stn].dVp_res.values
                    tot += c * pool[rng.integers(len(pool))]
                    s += c
                null.append(tot / s)
            bt = block_bootstrap(dd.block.values, lambda j: np.mean(dd.dVp_res.values[j]), n=a.nboot)
            out.append(dict(area=name, node_depth=dep, n=len(dd), n_nodes=dd.node.nunique(),
                            n_blocks=dd.block.nunique(), dVp_mean=dd.dVp.mean(), dVp_res_mean=real,
                            ci_lo=ci(bt)[0], ci_hi=ci(bt)[1], null_lo=ci(null)[0], null_hi=ci(null)[1],
                            p_perm=perm_p(real, null)))
    save(pd.DataFrame(out), "t5_areas_dvp.csv")
    log(pd.DataFrame(out).to_string())


# ======================================================================== synth
SYNTH_TILES = {"BL": [("blanco_m0", "inv_d200_s20")],
               "WA": [("oregon_m0", "inv_d200_s480"), ("washington_m0", "inv_d200_s480")],
               "MTJ": [("mendocino_gorda_c_m0", "inv_d200_s480")]}


def read_mod_file(path: Path):
    lines = path.read_text().split("\n")
    _, nx, ny, nz = lines[0].split()
    nx, ny, nz = int(nx), int(ny), int(nz)
    rows = [np.array(l.split(), float) for l in lines[4:4 + 2 * ny * nz]]
    vp = np.array(rows[:ny * nz]).reshape(nz, ny, nx)
    vpvs = np.array(rows[ny * nz:]).reshape(nz, ny, nx)
    return vp, vpvs


def synth_fields(model: Model, tile: str, inv: str):
    """Checkerboard inversion errors of one tile on the merged grid: {case: (eP %, eS %, eVpVs %)} where
    e = recovered - true change (true Vp/Vs change is 0 in the checkerboards)."""
    base = TOMO_RUNS / tile
    vp0, vpvs0 = read_mod_file(base / "inputs" / "MOD")
    vs0 = vp0 / vpvs0
    out = {}
    for run in sorted((base / "res_tests").glob(f"chk_L*_p*_noise*/{inv}/Output_Files/model_geo.npz")):
        case = run.parents[2].name[4:]                       # L060_p0_noise_ploc
        pat = "_".join(case.split("_")[:2])
        tf = base / "res_tests" / f"chk_{pat}" / "true_dvp.npy"
        if not tf.exists():
            continue
        m = np.load(run)
        true = np.load(tf)                                   # [nz, ny, nx]
        vp = np.transpose(m["vp"], (2, 1, 0))
        vs = np.transpose(m["vs"], (2, 1, 0))
        rp, rs = vp / vp0 - 1, vs / vs0 - 1
        eP, eS = 100 * (rp - true), 100 * (rs - true)
        eR = 100 * ((1 + rp) / (1 + rs) - 1)
        # tile lattices are offset from the merged lattice (e.g. Blanco by 0.025 deg in longitude): trilinear
        # interpolation of the error on the tile's interior nodes (boundary nodes excluded) at the merged nodes
        from scipy.interpolate import RegularGridInterpolator
        zi, la, lo = m["z"][1:-1], m["lat"][1:-1], m["lon"][1:-1]
        Z, LA, LO = np.meshgrid(model.dep, model.lat, model.lon, indexing="ij")
        pts = np.column_stack([Z.ravel(), LA.ravel(), LO.ravel()])
        F = []
        for e in (eP, eS, eR):
            f = RegularGridInterpolator((zi, la, lo), e[1:-1, 1:-1, 1:-1], bounds_error=False, fill_value=np.nan)
            F.append(f(pts).reshape(model.shape))
        out[case] = F
    return out


def step_synth(model: Model, a):
    ev, nodes = load()
    e = ev[ev.located & ev.is_earthquake]
    fam_ps = pd.read_csv(OUT / "t1_families_resPS.csv")
    rows = []
    for zone, tiles in SYNTH_TILES.items():
        fields = {}
        for tile, inv in tiles:
            for case, F in synth_fields(model, tile, inv).items():
                if case in fields:                           # two tiles cover the zone: average where both exist
                    G = fields[case]
                    fields[case] = [np.where(np.isfinite(g) & np.isfinite(f), 0.5 * (g + f), np.where(np.isfinite(g),
                                                                                                         g, f))
                                    for g, f in zip(G, F)]
                else:
                    fields[case] = F
        for case, (eP, eS, eR) in fields.items():
            for name, arr, mask, st in (("dVp", eP, model.mP, model.stratum_P), ("dVs", eS, model.mPS, model.stratum_S),
                                        ("dVpVs", eR, model.mPS, model.stratum_S)):
                sel = mask & (model.zone3 == zone) & np.isfinite(arr)
                if sel.sum() < 6:
                    continue
                res = np.full(model.shape, np.nan)
                for s_ in np.unique(st[sel]):
                    m_ = sel & (st == s_)
                    res[m_] = arr[m_] - arr[m_].mean()
                flat = res.ravel()
                ez = e[e.node >= 0]
                v = flat[ez.node.values]
                ok = np.isfinite(v)
                cnt = ez[ok].groupby("node").size()
                nd = np.flatnonzero(sel.ravel())
                y = pd.Series(nd).map(cnt).fillna(0).values
                x = flat[nd]
                rho = stratified_spearman(y, x, st.ravel()[nd])
                row = dict(zone=zone, case=case, var=name, n_nodes=len(nd), coverage_frac=float(sel.sum() / (mask & (
                    model.zone3 == zone)).sum()), err_rms=float(np.sqrt(np.nanmean(arr[sel] ** 2))),
                           event_mean_res=float(np.mean(v[ok])) if ok.any() else np.nan, rho_cov=rho)
                if zone == "BL" and name in ("dVp", "dVs", "dVpVs"):
                    em = e[(e.family_id >= 0) & e.resPS]
                    vv = pd.Series(flat[em.node.values], index=em.index)
                    med = vv.groupby(em.family_id).median()
                    f = fam_ps.set_index("family_id").assign(err=med)
                    row["swarm_minus_ma"] = diff_types(f.reset_index(), "err")
                rows.append(row)
    res = pd.DataFrame(rows)
    save(res, "synth_artifact.csv")
    summ = res.groupby(["zone", "var"]).agg(n_cases=("case", "size"), err_rms_med=("err_rms", "median"),
                                           event_mean_res_min=("event_mean_res", "min"),
                                           event_mean_res_max=("event_mean_res", "max"),
                                           event_mean_res_absmax=("event_mean_res", lambda s: s.abs().max()),
                                           rho_min=("rho_cov", "min"), rho_max=("rho_cov", "max"),
                                           ploc_mean_res=("event_mean_res", lambda s: s[res.loc[s.index, "case"]
                                                                                         .str.contains("ploc")].mean()))
    if "swarm_minus_ma" in res:
        summ = summ.join(res.groupby(["zone", "var"]).swarm_minus_ma.agg(lambda s: s.abs().max()).rename(
            "swarm_minus_ma_absmax"))
    save(summ.reset_index(), "synth_artifact_summary.csv")
    log(res.to_string())
    log(summ.to_string())


# ======================================================================== mc
def headline(ev: pd.DataFrame, model: Model, nodecol="node"):
    """Headline statistics for a given node assignment (used by the location Monte Carlo)."""
    e = ev[ev.located & ev.is_earthquake & (ev[nodecol] >= 0)]
    node = e[nodecol].values
    out = {}
    fl = {v: getattr(model, v + "_res").ravel() for v in ("dVp", "Vp", "dVs", "VpVs", "dVpVs")}
    mP, mPS = model.mP.ravel()[node], model.mPS.ravel()[node]
    zone = model.zone3.ravel()[node]
    n_eq = int((ev.located & ev.is_earthquake).sum())
    out["pct_resP"] = 100 * mP.sum() / n_eq
    out["pct_resPS"] = 100 * mPS.sum() / n_eq
    for zn in ZONES:
        z = zone == zn
        for v in ("dVp", "Vp"):
            sel = z & mP
            out[f"{zn}_{v}_res_mean"] = float(np.mean(fl[v][node[sel]])) if sel.sum() else np.nan
        if zn == "BL":
            for v in ("dVs", "VpVs", "dVpVs"):
                sel = z & mPS
                out[f"{zn}_{v}_res_mean"] = float(np.mean(fl[v][node[sel]])) if sel.sum() else np.nan
    # swarm - mainshock-aftershock at the Blanco (family medians of members in Vp/Vs-resolved voxels, >= 3)
    f = e[(e.family_id >= 0) & mPS & (zone == "BL")]
    for v in ("VpVs", "dVp"):
        vals = pd.Series(fl[v][f[nodecol].values], index=f.index)
        g = vals.groupby(f.family_id)
        med = g.median()[g.size() >= 3]
        typ = f.groupby("family_id").family_type.first().reindex(med.index)
        a_, b_ = med[typ == "swarm"], med[typ == "mainshock-aftershock"]
        out[f"BL_swarm_minus_MA_{v}_res"] = float(a_.median() - b_.median()) if len(a_) > 1 and len(b_) > 1 else np.nan
        out["BL_n_swarm_fam"], out["BL_n_MA_fam"] = len(a_), len(b_)
    return out


def step_mc(model: Model, a):
    ev, nodes = load()
    rng = np.random.default_rng(RNG_SEED + 6)
    sh = ev.sigma_h.fillna(1.5).values
    sz = ev.sig_z.fillna(3.0).values
    rows = [dict(draw=-1, **headline(ev, model))]
    for k in range(a.nmc):
        lon = ev.lon.values + rng.normal(0, 1, len(ev)) * sh / (KM * np.cos(np.radians(ev.lat.fillna(45).values)))
        lat = ev.lat.values + rng.normal(0, 1, len(ev)) * sh / KM
        dep = ev.depth.values + rng.normal(0, 1, len(ev)) * sz
        attach_nodes(ev, model, lon, lat, dep, prefix="mc_")
        rows.append(dict(draw=k, **headline(ev, model, "mc_node")))
    res = pd.DataFrame(rows)
    save(res, "mc_location.csv")
    d = res[res.draw >= 0].drop(columns="draw")
    summ = pd.DataFrame(dict(catalog=res[res.draw == -1].drop(columns="draw").iloc[0], mc_median=d.median(),
                             mc_p05=d.quantile(0.05), mc_p95=d.quantile(0.95))).reset_index(names="statistic")
    save(summ, "mc_location_summary.csv")
    log(summ.to_string())


# ======================================================================== v4hypo
V4HYPO_POS = {"v5.1 (all located earthquakes)": ("lon", "lat", "depth", False),
              "v5.1 (events with a v4 hypocentre)": ("lon", "lat", "depth", True),
              "v4": ("v4_lon", "v4_lat", "v4_depth", True),
              "v4 epicentre, v5.1 depth": ("v4_lon", "v4_lat", "depth", True),
              "v5.1 epicentre, v4 depth": ("lon", "lat", "v4_depth", True)}


def step_v4hypo(model: Model, a):
    """Q2 control: the catalog v4 hypocentres of the same events (GraphDD, not located in v0.3.1) sampled in the
    same model, and the two mixed positions, to see which coordinate carries the event-velocity association."""
    ev, nodes = load()
    rng = np.random.default_rng(RNG_SEED + 7)
    e0 = ev[ev.located & ev.is_earthquake].copy()
    rows = []
    for name, (lo, la, z, need_v4) in V4HYPO_POS.items():
        e = e0[e0.v4_depth.notna()].copy() if need_v4 else e0.copy()
        attach_nodes(e, model, e[lo].values, e[la].values, e[z].values, prefix="h_")
        for mask_col, vars_, st in (("resP", PVARS, "stratum_P"), ("resPS", SVARS, "stratum_S")):
            cnt = e[e["h_" + mask_col]].groupby("h_node").size()
            n = nodes[nodes[mask_col]].copy()
            n["count"] = n.node.map(cnt).fillna(0).values
            for zone in ZONES:
                nz_ = n[n.zone == zone]
                if len(nz_) < 6 or nz_["count"].sum() == 0:
                    continue
                y = nz_["count"].values.astype(float)
                bl, strata = nz_.block.values, nz_[st].values
                occ = (y > 0).astype(float)
                for v in vars_:
                    x = nz_[v + "_res"].values
                    anom = weighted_anom(y, x)
                    boot = block_bootstrap(bl, lambda i: weighted_anom(y[i], x[i]), n=a.nboot)
                    null = [weighted_anom(permute_within(y, strata, rng), x) for _ in range(a.nperm)]
                    rows.append(dict(positions=name, mask=mask_col, zone=zone, var=v, n_events=int(y.sum()),
                                     n_nodes_with_events=int((y > 0).sum()), n_blocks=int(pd.Series(bl).nunique()),
                                     event_mean_res=anom, res_ci_lo=ci(boot)[0], res_ci_hi=ci(boot)[1],
                                     res_null_lo=ci(null)[0], res_null_hi=ci(null)[1], res_p=perm_p(anom, null),
                                     occ_mean_res=weighted_anom(occ, x),
                                     rho_cov=stratified_spearman(y, x, strata)))
    res = pd.DataFrame(rows)
    save(res, "t2_v4hypo.csv")
    log(res[res.zone == "BL"].to_string())


# ======================================================================== blocks
_BLK = {}


def _blocks_task(args):
    var, size, fx, fy, nboot = args
    y, x, st, xk, yk = _BLK[var]
    blk = np.floor((xk + fx * size) / size).astype(int) * 1000 + np.floor((yk + fy * size) / size).astype(int)
    seed = RNG_SEED + int(size * 100 + fx * 40 + fy * 4)
    bw = block_bootstrap(blk, lambda i: weighted_anom(y[i], x[i]), n=nboot, seed=seed)
    bs = block_bootstrap(blk, lambda i: stratified_spearman(y[i], x[i], st[i]), n=nboot, seed=seed)
    return dict(var=var, block_km=size, shift_x=fx, shift_y=fy, n_blocks=len(np.unique(blk)),
                w_ci_lo=ci(bw)[0], w_ci_hi=ci(bw)[1], s_ci_lo=ci(bs)[0], s_ci_hi=ci(bs)[1])


def step_blocks(model: Model, a):
    """Q2 Blanco: do the block-bootstrap intervals of the event-weighted anomaly and of the stratified Spearman
    correlation depend on where the 40 / 60 km block grid falls? 16 placements per block size (origin shifted by
    0, 1/4, 1/2, 3/4 of a block in x and y)."""
    import multiprocessing as mp
    ev, nodes = load()
    e = ev[ev.located & ev.is_earthquake]
    for mask_col, vars_, stc in (("resP", ["Vp", "dVp"], "stratum_P"), ("resPS", ["Vs", "dVs"], "stratum_S")):
        n = nodes[nodes[mask_col] & (nodes.zone == "BL")]
        y = n.node.map(e[e[mask_col]].groupby("node").size()).fillna(0).values.astype(float)
        xk = (n.lon.values - LON0) * KM * COSREF
        yk = (n.lat.values - LAT0) * KM
        for v in vars_:
            _BLK[v] = (y, n[v + "_res"].values, n[stc].values, xk, yk)
    tasks = [(v, s, fx, fy, a.nboot) for v in _BLK for s in (40, 60) for fx in (0, .25, .5, .75)
             for fy in (0, .25, .5, .75)]
    with mp.get_context("fork").Pool(min(a.nproc, len(tasks))) as pool:
        rows = pool.map(_blocks_task, tasks)
    res = pd.DataFrame(rows)
    save(res, "t2_block_offsets.csv")
    summ = res.groupby(["var", "block_km"]).agg(
        n_placements=("w_ci_hi", "size"),
        w_excludes_0=("w_ci_hi", lambda s: int(((s < 0) | (res.loc[s.index, "w_ci_lo"] > 0)).sum())),
        w_ci_hi_min=("w_ci_hi", "min"), w_ci_hi_max=("w_ci_hi", "max"),
        s_excludes_0=("s_ci_hi", lambda s: int(((s < 0) | (res.loc[s.index, "s_ci_lo"] > 0)).sum())),
        s_ci_hi_min=("s_ci_hi", "min"), s_ci_hi_max=("s_ci_hi", "max")).reset_index()
    save(summ, "t2_block_offsets_summary.csv")
    log(summ.to_string())


# ======================================================================== tiles
TILE_RUNS = [("blanco_m0", "d200_s20", "the run merged into v0.3.1"),
             ("blanco_m0", "d200_s60", "smoothing 60"),
             ("blanco_m0", "d200_s120", "smoothing 120"),
             ("blanco_m0t", "d200_s20", "re-timed picks; the Blanco tile of v0.4.0"),
             ("blanco_m0t", "d200_s60", "re-timed picks, smoothing 60"),
             ("blanco_m0tnf", "d200_s20", "re-timed picks, start from the margin_m0_v3nf reference")]


def step_tiles(model: Model, a):
    """Q2 Blanco: the same event statistic on Blanco tile inversions with stronger smoothing and with the re-timed
    picks (v0.4.0's Blanco tile), on the v0.3.1 resolved nodes and strata (fields interpolated trilinearly from the
    tile lattice, interior nodes only, as in the synth step)."""
    from scipy.interpolate import RegularGridInterpolator
    ev, nodes = load()
    e = ev[ev.located & ev.is_earthquake]
    rows = []
    Z, LA, LO = np.meshgrid(model.dep, model.lat, model.lon, indexing="ij")
    pts = np.column_stack([Z.ravel(), LA.ravel(), LO.ravel()])
    for tile, inv, note in TILE_RUNS:
        f = TOMO_RUNS / tile / inv / "Output_Files" / "model_geo.npz"
        if not f.exists():
            log("missing", f)
            continue
        vp0, vpvs0 = read_mod_file(TOMO_RUNS / tile / "inputs" / "MOD")
        m = np.load(f)
        vp = np.transpose(m["vp"], (2, 1, 0))
        vs = np.transpose(m["vs"], (2, 1, 0))
        zi, la, lo = m["z"][1:-1], m["lat"][1:-1], m["lon"][1:-1]
        for nm, arr, a0, mask_col, st in (("dVp", vp, vp0, "resP", "stratum_P"),
                                          ("dVs", vs, vp0 / vpvs0, "resPS", "stratum_S")):
            d = 100 * (arr / a0 - 1)
            F = RegularGridInterpolator((zi, la, lo), d[1:-1, 1:-1, 1:-1], bounds_error=False,
                                        fill_value=np.nan)(pts)
            n = nodes[nodes[mask_col] & (nodes.zone == "BL")].copy()
            n["d"] = F[n.node.values]
            n = n[np.isfinite(n.d)]
            n["d_res"] = n.d - n.groupby(st).d.transform("mean")
            y = n.node.map(e[e[mask_col]].groupby("node").size()).fillna(0).values.astype(float)
            x = n.d_res.values
            boot = block_bootstrap(n.block.values, lambda i: weighted_anom(y[i], x[i]), n=a.nboot)
            rows.append(dict(tile=tile, run=inv, note=note, var=nm, n_nodes=len(n), n_events=int(y.sum()),
                             event_mean_res=weighted_anom(y, x), res_ci_lo=ci(boot)[0], res_ci_hi=ci(boot)[1],
                             node_sd=float(n.d.std()), corr_with_v031=float(np.corrcoef(n.d, n[nm])[0, 1])))
    res = pd.DataFrame(rows)
    save(res, "t2_tile_runs.csv")
    log(res.to_string())


# ======================================================================== v4
def step_v4(model: Model, a):
    """The earlier counterpart (phase13_tomography.py: dlnVs from the depth-mean of an external Vs model at the
    events, median against the whole-model baseline) on the v4 and the v5.1 hypocentres of the same events."""
    import xarray as xr
    from scipy.interpolate import RegularGridInterpolator
    ev, nodes = load()
    e = ev[ev.located & ev.is_earthquake & ev.v4_depth.notna()].copy()
    rows = []
    for fname in ("CRESCENT_Gen0.nc", "Delph2018.nc"):
        d = xr.open_dataset(EXT_MODELS / fname)
        vname = "Vs" if "Vs" in d else "vs"
        V = d[vname].transpose("depth", "latitude", "longitude").values.astype(float)
        dep, lat, lon = d.depth.values, d.latitude.values, d.longitude.values
        dm = np.nanmean(V, axis=(1, 2))
        dln = 100 * (V - dm[:, None, None]) / dm[:, None, None]
        f = RegularGridInterpolator((dep, lat, lon), dln, bounds_error=False, fill_value=np.nan)
        base = dln[np.isfinite(dln)]
        for tag, (la, lo, z) in {"v4": ("v4_lat", "v4_lon", "v4_depth"), "v5.1": ("lat", "lon", "depth")}.items():
            e[f"dln_{tag}"] = f(np.column_stack([e[z], e[la], e[lo]]))
        both = e[np.isfinite(e.dln_v4) & np.isfinite(e["dln_v5.1"])]
        for grp, g in [("all located earthquakes", both)] + list(both.groupby("region")):
            if len(g) < 30:
                continue
            # the same group without the restriction to events inside the external model at both positions
            ga = e if grp == "all located earthquakes" else e[e.region == grp]
            rows.append(dict(model=fname, group=grp, n=len(g), baseline_median=float(np.median(base)),
                             median_v4=g.dln_v4.median(), median_v51=g["dln_v5.1"].median(),
                             change=g["dln_v5.1"].median() - g.dln_v4.median(),
                             depth_med_v4=g.v4_depth.median(), depth_med_v51=g.depth.median(),
                             depth_shift_med=(g.depth - g.v4_depth).median(),
                             frac_slow_v4=(g.dln_v4 < 0).mean(), frac_slow_v51=(g["dln_v5.1"] < 0).mean(),
                             n_group_all=len(ga), depth_med_v4_all=ga.v4_depth.median(),
                             depth_med_v51_all=ga.depth.median(),
                             depth_shift_med_all=(ga.depth - ga.v4_depth).median()))
        # on the resolved cells of v0.3.1: external dlnVs against our coverage-matched dVs (Blanco) and dVp
        r = both[both.resPS]
        if len(r) > 30:
            from scipy.stats import spearmanr
            rows.append(dict(model=fname, group="v5.1 events in Vp/Vs-resolved cells: Spearman(external dlnVs, "
                                                "v0.3.1 dVs_res)", n=len(r),
                             median_v51=r["dln_v5.1"].median(),
                             change=float(spearmanr(r["dln_v5.1"], r.dVs_res, nan_policy="omit").statistic)))
    res = pd.DataFrame(rows)
    save(res, "v4_counterpart_external_vs.csv")
    log(res.to_string())


# ======================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default="prep,coverage,t1,t2,t3,t4,t5")
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--nperm", type=int, default=2000)
    ap.add_argument("--min-col", dest="min_col", type=int, default=30)
    ap.add_argument("--min-vert", dest="min_vert", type=int, default=10)
    ap.add_argument("--nmc", type=int, default=50)
    ap.add_argument("--nproc", type=int, default=8, help="worker processes of the blocks step (at most 12)")
    a = ap.parse_args()
    a.nproc = max(1, min(a.nproc, 12))
    OUT.mkdir(parents=True, exist_ok=True)
    BIG.mkdir(parents=True, exist_ok=True)
    model = Model()
    for s in a.steps.split(","):
        t0 = time.time()
        log(f"==== step {s}")
        globals()[f"step_{s}"](model, a)
        log(f"==== step {s} done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

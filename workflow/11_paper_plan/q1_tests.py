#!/usr/bin/env python3
"""Q1 (D-B115): the pre-registered tests, controls, nulls and decision rules (RESUME.md, written before any
covariate correlation was computed). Reads the tables of q1_metrics.py, q1_recluster.py, q1_vpvs.py and the per-event
covariates (q1_covariates.py interim; the D-B114 table when it exists). Writes BIG/tables/tests_*.csv, decision.json
and copies the small tables to OUT.

Statistics:
* rank correlation: Spearman rho; permutation p from 10,000 permutations of the covariate (one-sided in the predicted
  direction, two-sided where no sign is predicted); 95 % interval from 2,000 bootstraps of 1 x 1 degree blocks of cells.
* type-controlled: weighted mean of within-type Spearman rho (weight = cells of the type; types with >= 5 cells with a
  value), covariate permuted within type (10,000); effect size from a quasi-binomial GLM logit(swarm share) ~
  z(covariate) + type (IRLS, small ridge on the type terms against separation).
* Holm over the five primary tests; Benjamini-Hochberg (q = 0.05) over the secondary family and, separately, the
  exploratory family.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import q1_common as Q  # noqa: E402

TAB = Q.BIG / "tables"
NPERM = 10000
NBOOT = 2000
ALPHA = 0.05
RUNS_CTRL = (["mcu_count", "mcu_recluster", "mc_local"] + [f"var_{v}" for v in Q.VARIANTS]
             + ["offshore", "partial", "ci_window", "cell04", "N30", "N100", "v60"])


# ------------------------------------------------------------------ statistics
def rho_ranks(rx, ry):
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else np.nan


def perm_rho(x, y, sign, rng, n=NPERM, strata=None):
    """Spearman rho of (x, y) and its permutation p (x permuted, within strata if given)."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 5:
        return np.nan, np.nan, len(x)
    rx, ry = rankdata(x), rankdata(y)
    r0 = rho_ranks(rx, ry)
    ry_c = ry - ry.mean()
    dn = np.sqrt(((rx - rx.mean()) ** 2).sum() * (ry_c ** 2).sum())
    if strata is None:
        P = np.argsort(rng.random((n, len(x))), axis=1)
        rp = ((rx[P] - rx.mean()) * ry_c).sum(axis=1) / dn
    else:
        st = np.asarray(strata)[ok]
        idx = np.tile(np.arange(len(x)), (n, 1))
        for s in np.unique(st):
            k = np.where(st == s)[0]
            idx[:, k] = k[np.argsort(rng.random((n, len(k))), axis=1)]
        rp = ((rx[idx] - rx.mean()) * ry_c).sum(axis=1) / dn
    return r0, pval(r0, rp, sign), len(x)


def pval(obs, null, sign):
    null = np.asarray(null)
    if sign > 0:
        return float((1 + np.sum(null >= obs - 1e-12)) / (1 + len(null)))
    if sign < 0:
        return float((1 + np.sum(null <= obs + 1e-12)) / (1 + len(null)))
    return float((1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (1 + len(null)))


def strat_rho(x, y, t, sign, rng, n=NPERM, nmin=5):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, t = x[ok], y[ok], np.asarray(t)[ok]
    types = [s for s in np.unique(t) if (t == s).sum() >= nmin]
    if not types:
        return np.nan, np.nan, 0, {}
    per, num, den = {}, 0.0, 0
    perms = []
    for s in types:
        k = t == s
        rx, ry = rankdata(x[k]), rankdata(y[k])
        r = rho_ranks(rx, ry)
        per[s] = (r, int(k.sum()))
        if np.isfinite(r):
            num += r * k.sum()
            den += k.sum()
        ryc = ry - ry.mean()
        dn = np.sqrt(((rx - rx.mean()) ** 2).sum() * (ryc ** 2).sum())
        P = np.argsort(rng.random((n, k.sum())), axis=1)
        rp = ((rx[P] - rx.mean()) * ryc).sum(axis=1) / dn if dn > 0 else np.full(n, np.nan)
        perms.append((np.nan_to_num(rp) * k.sum()) if np.isfinite(r) else np.zeros(n))
    if den == 0:
        return np.nan, np.nan, 0, per
    stat = num / den
    null = np.sum(perms, axis=0) / den
    return float(stat), pval(stat, null, sign), int(den), per


def block_boot(x, y, blocks, rng, n=NBOOT):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, b = x[ok], y[ok], np.asarray(blocks)[ok]
    ub = np.unique(b)
    if len(ub) < 4:
        return np.nan, np.nan
    groups = [np.where(b == u)[0] for u in ub]
    out = []
    for _ in range(n):
        k = np.concatenate([groups[i] for i in rng.integers(0, len(ub), len(ub))])
        if len(np.unique(x[k])) < 3 or len(np.unique(y[k])) < 3:
            continue
        out.append(rho_ranks(rankdata(x[k]), rankdata(y[k])))
    return Q.ci(out)


def partial_rho(x, y, Z, sign, rng, n=NPERM):
    ok = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    x, y, Z = x[ok], y[ok], Z[ok]
    if len(x) < 8:
        return np.nan, np.nan, len(x)
    R = np.column_stack([np.ones(len(x))] + [rankdata(z) for z in Z.T])
    res = lambda v: v - R @ np.linalg.lstsq(R, v, rcond=None)[0]  # noqa: E731
    ex, ey = res(rankdata(x)), res(rankdata(y))
    r0 = float(np.corrcoef(ex, ey)[0, 1])
    P = np.argsort(rng.random((n, len(x))), axis=1)
    rp = np.array([np.corrcoef(ex[p], ey)[0, 1] for p in P[:2000]])
    return r0, pval(r0, rp, sign), len(x)


def glm_qb(k, n, xcov, types, ridge=1e-3, it=100):
    """Quasi-binomial logit GLM: k/n ~ z(xcov) + type dummies. Returns coef of z(xcov), its quasi SE, dispersion."""
    ok = np.isfinite(xcov) & (n > 0)
    k, n, xcov, types = k[ok], n[ok], xcov[ok], np.asarray(types)[ok]
    if len(k) < 8:
        return np.nan, np.nan, np.nan
    z = (xcov - xcov.mean()) / xcov.std()
    ut = sorted(set(types))
    X = np.column_stack([np.ones(len(k)), z] + [(types == u).astype(float) for u in ut[1:]])
    pen = np.r_[0, 0, np.full(len(ut) - 1, ridge)]
    b = np.zeros(X.shape[1])
    b[0] = np.log((k.sum() + 0.5) / (n.sum() - k.sum() + 0.5))
    for _ in range(it):
        eta = X @ b
        mu = 1 / (1 + np.exp(-eta))
        W = n * mu * (1 - mu) + 1e-12
        zz = eta + (k - n * mu) / W
        A = X.T @ (W[:, None] * X) + np.diag(pen * W.sum())
        bn = np.linalg.solve(A, X.T @ (W * zz))
        if np.max(np.abs(bn - b)) < 1e-8:
            b = bn
            break
        b = bn
    mu = 1 / (1 + np.exp(-(X @ b)))
    W = n * mu * (1 - mu) + 1e-12
    phi = float(np.sum((k - n * mu) ** 2 / W) / max(len(k) - X.shape[1], 1))
    cov = np.linalg.inv(X.T @ (W[:, None] * X) + np.diag(pen * W.sum())) * phi
    return float(b[1]), float(np.sqrt(cov[1, 1])), phi


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for i, k in enumerate(o):
        run = max(run, min(1.0, (len(p) - i) * p[k]))
        adj[k] = run
    return adj


def bh(p):
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    adj = np.full(len(p), np.nan)
    q = p[ok]
    o = np.argsort(q)
    m = len(q)
    r = q[o] * m / (np.arange(m) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    a = np.empty(m)
    a[o] = np.minimum(r, 1)
    adj[ok] = a
    return adj


# ------------------------------------------------------------------ inputs
EXT_MAP = {"age_myr": "age_myr", "heatflow_mw": "hf_smooth", "z600_km": "iso600_plate_bsf",
           "rate_mmyr": "full_rate_mmyr", "rough_m": "rough20", "sed_m": "sed_thk_m", "Tslab_C": "Tslab_D80"}


def covariates(tag="v61", source="external"):
    """Per-event covariates. external: the D-B114 table (derived/events_external_geo_v61.parquet; v6.1 positions):
    age_myr; heat flow = hf_smooth (Gaussian-weighted median of GHFDB sites, sigma 15 km, >= 3 effective sites);
    z600 = iso600_plate_bsf (plate model, ages floored at 0.1 Myr); full_rate_mmyr; roughness = rough20 (20 km window
    plane-fit RMS) where the window has no land (landfrac20 == 0); sed_thk_m; Tslab_D80 (exploratory).
    interim: q1_covariates.py. v6.0 always uses the interim sampler on its own positions."""
    if source == "interim" or tag != "v61":
        return pd.read_parquet(Q.covariate_file(tag))
    f = Q.external_event_table()
    if f is None:
        return None
    d = pd.read_parquet(f)
    out = pd.DataFrame({"ID": d.ID.values})
    for k, v in EXT_MAP.items():
        out[k] = d[v].values.astype(float)
    out.loc[d.landfrac20.values > 0, "rough_m"] = np.nan
    return out


def unit_table(run, unit, nmin, events, cov, vp=None):
    m = pd.read_csv(TAB / f"metrics_{run}.csv")
    m = m[(m.unit == unit) & (m.n >= nmin)].copy()
    uc = Q.unit_covariates(events, cov, unit).rename(columns={unit: "unit_id"})
    m = m.merge(uc, on="unit_id", how="left")
    if vp is not None:
        key = "seg" if unit == "seg" else "cell"
        if key == unit:
            v = vp[vp.valid].groupby(key).agg(vpvs_fam=("vpvs", "median"), n_vpvs=("vpvs", "size"))
            v.loc[v.n_vpvs < 3, "vpvs_fam"] = np.nan
            m = m.merge(v, left_on="unit_id", right_index=True, how="left")
    if unit.startswith("cell"):
        m["block"] = (np.floor(m.lon).astype(int).astype(str) + "_" + np.floor(m.lat).astype(int).astype(str))
    else:
        m["block"] = m.unit_id
    return m


# ------------------------------------------------------------------ main
def main():
    """Both covariate sources; the decision (decision.json) uses the D-B114 table (pre-registered)."""
    for src in ("external", "interim"):
        if src == "external" and Q.external_event_table() is None:
            Q.log("no D-B114 table: external run skipped")
            continue
        run(src)
    dsrc = "external" if (TAB / "decision_external.json").exists() else "interim"
    (TAB / "decision.json").write_bytes((TAB / f"decision_{dsrc}.json").read_bytes())
    (Q.OUT / "decision.json").write_bytes((TAB / "decision.json").read_bytes())


def run(src):
    rng = np.random.default_rng(Q.SEED)
    ev61 = pd.read_parquet(TAB / "events_q1_v61.parquet")
    cov61 = covariates("v61", src)
    vp = pd.read_csv(Q.BIG / "vpvs" / "vpvs_families.csv") if (Q.BIG / "vpvs" / "vpvs_families.csv").exists() else None
    cells = unit_table("main", "cell", Q.NMIN_CELL, ev61, cov61)
    segs = unit_table("main", "seg", Q.NMIN_SEG, ev61, cov61, vp)
    segs_t = segs[segs.seg_type_major != "other"]
    cells.to_csv(TAB / f"cells_main_{src}.csv", index=False, float_format="%.5g")
    segs.to_csv(TAB / f"segments_main_{src}.csv", index=False, float_format="%.5g")
    rows = []

    def add(test, family, unit, metric, cov, sign, r, p, n, extra=None):
        d = dict(test=test, family=family, unit=unit, metric=metric, covariate=cov, pred_sign=sign, rho=r, p=p, n=n)
        d.update(extra or {})
        rows.append(d)

    # ---- primary
    for test, cov in (("T1", "age_myr"), ("T2", "heatflow_mw")):
        s = Q.PRED_SIGN[cov]
        r, p, n = perm_rho(cells[cov].values, cells.swarm_share.values, s, rng)
        lo, hi = block_boot(cells[cov].values, cells.swarm_share.values, cells.block.values, rng)
        rt, pt_two, _ = perm_rho(cells[cov].values, cells.swarm_share.values, 0, rng)
        add(test, "primary", "cell", "swarm_share", cov, s, r, p, n, dict(block_lo=lo, block_hi=hi, p_two=pt_two))
    for test, cov in (("T4a", "age_myr"), ("T4b", "heatflow_mw")):
        s = Q.PRED_SIGN[cov]
        st, p, n, per = strat_rho(cells[cov].values, cells.swarm_share.values, cells.seg_type_major.values, s, rng)
        k = np.round(cells.swarm_share.values * cells.n.values / 100)
        b, se, phi = glm_qb(k, cells.n.values.astype(float), cells[cov].values, cells.seg_type_major.values)
        add(test, "primary", "cell", "swarm_share", cov, s, st, p, n,
            dict(within_type=json.dumps({t: [round(v[0], 3) if np.isfinite(v[0]) else None, v[1]]
                                         for t, v in per.items()}),
                 glm_coef_per_sd=b, glm_se=se, glm_dispersion=phi))
    # T3 Vp/Vs swarm vs MA families
    t3 = dict(test="T3", family="primary", unit="family", metric="vpvs", covariate="family type", pred_sign=1)
    if vp is not None:
        f = vp[vp.valid & vp.carried & vp.type61.isin(["swarm", "mainshock-aftershock"])].copy()
        nsw, nma = int((f.type61 == "swarm").sum()), int((f.type61 == "mainshock-aftershock").sum())
        t3.update(n_swarm=nsw, n_ma=nma, n=nsw + nma)
        if nsw >= 20 and nma >= 20:
            lab = (f.type61 == "swarm").values
            v = f.vpvs.values
            st = f.seg_type.fillna("none").values
            d0 = np.median(v[lab]) - np.median(v[~lab])
            null = []
            for _ in range(NPERM):
                lp = lab.copy()
                for s_ in np.unique(st):
                    k_ = np.where(st == s_)[0]
                    lp[k_] = lab[k_][rng.permutation(len(k_))]
                null.append(np.median(v[lp]) - np.median(v[~lp]))
            t3.update(rho=d0, p=pval(d0, null, 1), p_two=pval(d0, null, 0),
                      med_swarm=float(np.median(v[lab])), med_ma=float(np.median(v[~lab])))
        else:
            t3.update(rho=np.nan, p=np.nan, note="not testable (< 20 swarm or < 20 MA families with a valid Vp/Vs)")
    else:
        t3.update(rho=np.nan, p=np.nan, note="no Vp/Vs table")
    rows.append(t3)

    # ---- secondary (cells, segments)
    sec = [("b", "age_myr", -1), ("b", "heatflow_mw", 1), ("productivity", "age_myr", 1),
           ("productivity", "heatflow_mw", -1), ("fam_mix", "age_myr", -1), ("fam_mix", "heatflow_mw", 1),
           ("bg_frac", "age_myr", 0), ("bg_frac", "heatflow_mw", 0), ("swarm_share", "rate_mmyr", 1)]
    for met, cov, s in sec:
        r, p, n = perm_rho(cells[cov].values, cells[met].values, s, rng)
        lo, hi = block_boot(cells[cov].values, cells[met].values, cells.block.values, rng)
        add("S", "secondary", "cell", met, cov, s, r, p, n, dict(block_lo=lo, block_hi=hi))
    for cov in ("age_myr", "heatflow_mw", "rate_mmyr", "vpvs_fam"):
        if cov not in segs_t.columns:
            continue
        s = Q.PRED_SIGN[cov]
        r, p, n = perm_rho(segs_t[cov].values, segs_t.swarm_share.values, s, rng)
        add("S", "secondary", "seg", "swarm_share", cov, s, r, p, n)
    # ---- exploratory
    for cov in ("rough_m", "sed_m", "z600_km", "Tslab_C"):
        if cov not in cells.columns:
            continue
        r, p, n = perm_rho(cells[cov].values, cells.swarm_share.values, Q.PRED_SIGN[cov], rng)
        lo, hi = block_boot(cells[cov].values, cells.swarm_share.values, cells.block.values, rng)
        add("E", "exploratory", "cell", "swarm_share", cov, Q.PRED_SIGN[cov], r, p, n, dict(block_lo=lo, block_hi=hi))
        r, p, n = perm_rho(segs_t[cov].values, segs_t.swarm_share.values, Q.PRED_SIGN[cov], rng)
        add("E", "exploratory", "seg", "swarm_share", cov, Q.PRED_SIGN[cov], r, p, n)
    for met in ("b", "productivity", "bg_frac"):
        for cov in ("rough_m", "sed_m"):
            r, p, n = perm_rho(cells[cov].values, cells[met].values, 0, rng)
            add("E", "exploratory", "cell", met, cov, 0, r, p, n)
    T = pd.DataFrame(rows)
    pr = T.family == "primary"
    T.loc[pr, "p_adj"] = holm(T.loc[pr, "p"].fillna(1).values)
    for fam_ in ("secondary", "exploratory"):
        k = T.family == fam_
        T.loc[k, "p_adj"] = bh(T.loc[k, "p"].values)
    T["significant"] = T.p_adj < ALPHA
    T["sign_ok"] = np.where(T.pred_sign == 0, True, np.sign(T.rho) == T.pred_sign)
    T.to_csv(TAB / f"tests_{src}.csv", index=False, float_format="%.4g")

    # ---- controls: sign of the cell rho for age and heat flow in every control run
    crow = []
    cov60 = covariates("v60", "interim")
    ev60 = pd.read_parquet(TAB / "events_q1_v60.parquet") if (TAB / "events_q1_v60.parquet").exists() else None
    for run in RUNS_CTRL:
        if run in ("offshore", "partial", "N30", "N100", "cell04"):
            base, unit, nmin = "main", ("cell04" if run == "cell04" else "cell"), {"N30": 30, "N100": 100}.get(run, 50)
        else:
            base, unit, nmin = run, "cell", Q.NMIN_CELL
        if not (TAB / f"metrics_{base}.csv").exists():
            continue
        if run == "v60":
            if ev60 is None:
                continue
            u = unit_table("v60", "cell", Q.NMIN_CELL, ev60, cov60)
        else:
            u = unit_table(base, unit, nmin, ev61, cov61)
        if run == "offshore":
            u = u[u.offshore_frac >= 0.5]
        for cov in ("age_myr", "heatflow_mw"):
            s = Q.PRED_SIGN[cov]
            if run == "partial":
                r, p, n = partial_rho(u[cov].values, u.swarm_share.values,
                                      u[["mc_map_med", "dist_min_med"]].values, s, rng)
            else:
                r, p, n = perm_rho(u[cov].values, u.swarm_share.values, s, rng, n=2000)
            crow.append(dict(run=run, covariate=cov, rho=r, p_one=p, n=n, sign_ok=bool(np.sign(r) == s)
                             if np.isfinite(r) else False))
    C = pd.DataFrame(crow)
    C.to_csv(TAB / f"controls_{src}.csv", index=False, float_format="%.4g")

    # ---- nulls per segment and cell
    nul = {}
    E = ev61[ev61.event_set][["ID", "seg", "cell", "cell04"]]
    for nm in ("tshuf", "mshuf"):
        f = Q.BIG / "nulls" / f"{nm}_swarm_ids.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f)
        seeds = np.sort(d.seed.unique())
        parts = []
        for key in ("seg", "cell", "cell04"):
            n_u = E.groupby(key).size()
            x = d.merge(E[["ID", key]], on="ID").groupby(["seed", key]).size().unstack(fill_value=0)
            x = x.reindex(index=seeds, columns=n_u.index, fill_value=0)
            sh = 100 * x / n_u.values[None, :]
            parts.append(pd.DataFrame(dict(unit=key, unit_id=sh.columns, null_mean=sh.mean().values,
                                           null_p95=sh.quantile(0.95).values, null_p05=sh.quantile(0.05).values,
                                           n_seeds=len(seeds))))
        S = pd.concat(parts, ignore_index=True)
        obs = pd.read_csv(TAB / "metrics_main.csv")[["unit", "unit_id", "n", "swarm_share"]]
        S = S.merge(obs, on=["unit", "unit_id"], how="inner")
        S["above_p95"] = S.swarm_share > S.null_p95
        S["below_p05"] = S.swarm_share < S.null_p05
        S["null"] = nm
        nul[nm] = S
    N = pd.concat(nul.values(), ignore_index=True) if nul else pd.DataFrame()
    N.to_csv(TAB / "nulls_units.csv", index=False, float_format="%.4g")

    # ---- uniform-Mc reclustering: hot vs cold segments
    hc = {}
    for run, nmin in (("mcu_recluster", Q.NMIN_SEG), ("mcu_recluster", 30), ("mcu_count", Q.NMIN_SEG),
                      ("mcu_count", 30), ("main", Q.NMIN_SEG)):
        f = TAB / f"metrics_{run}.csv"
        if not f.exists():
            continue
        m = pd.read_csv(f)
        m = m[(m.unit == "seg") & (m.n >= nmin)].set_index("unit_id")
        h, c = m.swarm_share.reindex(Q.HOT).dropna(), m.swarm_share.reindex(Q.COLD).dropna()
        hc[f"{run}_n{nmin}"] = dict(hot=h.round(2).to_dict(), cold=c.round(2).to_dict(),
                                    hot_median=float(h.median()) if len(h) else None,
                                    cold_median=float(c.median()) if len(c) else None,
                                    hot_above_cold=bool(len(h) and len(c) and h.median() > c.median()))

    # ---- decision rules
    t = T.set_index("test")
    prim = T[pr].set_index("test")

    def sig(k):
        return bool(k in prim.index and prim.loc[k, "significant"] and prim.loc[k, "sign_ok"])
    ctrl_ok = {cov: int(C[(C.covariate == cov)].sign_ok.sum()) if len(C) else 0 for cov in ("age_myr", "heatflow_mw")}
    ctrl_n = {cov: int((C.covariate == cov).sum()) if len(C) else 0 for cov in ("age_myr", "heatflow_mw")}
    null_rule = None
    if "tshuf" in nul:
        S = nul["tshuf"]
        s4 = S[(S.unit == "seg") & S.unit_id.isin(["AX", "BL", "GR", "MSH"])].set_index("unit_id")
        null_rule = dict(above=s4.above_p95.to_dict(), n_above=int(s4.above_p95.sum()), passed=bool(s4.above_p95.sum() >= 3))
    d1 = {}
    for test, cov in (("T1", "age_myr"), ("T2", "heatflow_mw")):
        r = prim.loc[test]
        block_ok = bool(np.isfinite(r.block_lo) and ((r.block_hi < 0) if r.pred_sign < 0 else (r.block_lo > 0)))
        d1[test] = dict(significant=sig(test), block_ci_excludes_0=block_ok, controls_sign_ok=ctrl_ok[cov],
                        controls_n=ctrl_n[cov], controls_pass=ctrl_ok[cov] >= 18,
                        null_rule=bool(null_rule and null_rule["passed"]))
        d1[test]["D1"] = all([d1[test]["significant"], block_ok, d1[test]["controls_pass"], d1[test]["null_rule"]])
    D1 = d1["T1"]["D1"] or d1["T2"]["D1"]
    D2 = (d1["T1"]["D1"] and sig("T4a")) or (d1["T2"]["D1"] and sig("T4b"))
    D3 = sig("T3")
    fals = dict(
        T1_T2_both_reversed=bool(all(np.sign(prim.loc[k, "rho"]) == -prim.loc[k, "pred_sign"]
                                     and prim.loc[k, "p_two"] < 0.05 for k in ("T1", "T2"))),
        uniform_mc_hot_not_above_cold=(not hc["mcu_recluster_n100"]["hot_above_cold"]) if "mcu_recluster_n100" in hc
        else None,
        null_rule_failed=(not null_rule["passed"]) if null_rule else None,
        vpvs_ma_above_swarm=bool("T3" in prim.index and np.isfinite(prim.loc["T3", "rho"]) and prim.loc["T3", "rho"] < 0
                                 and prim.loc["T3", "p_two"] < 0.05))
    verdict = ("supported beyond boundary type" if D2 else "supported as a boundary-type contrast" if D1 else
               "falsified" if (fals["T1_T2_both_reversed"] or fals["uniform_mc_hot_not_above_cold"]
                               or fals["null_rule_failed"]) else "inconclusive")
    dec = dict(D1=D1, D2=D2, D3=D3, d1_detail=d1, falsification=fals, null_rule=null_rule, hot_cold=hc,
               verdict_thermal=verdict,
               verdict_vpvs=("supported" if D3 else "falsified" if fals["vpvs_ma_above_swarm"] else
                             ("not testable" if "note" in t3 else "inconclusive")),
               covariate_source=src, external_table=str(Q.external_event_table()))
    json.dump(dec, open(TAB / f"decision_{src}.json", "w"), indent=1, default=str)
    for f in (f"tests_{src}.csv", f"controls_{src}.csv", "nulls_units.csv", f"decision_{src}.json",
              f"segments_main_{src}.csv", f"cells_main_{src}.csv"):
        (Q.OUT / f).write_bytes((TAB / f).read_bytes())
    Q.log(T[["test", "family", "unit", "metric", "covariate", "pred_sign", "rho", "p", "p_adj", "n"]].round(4).to_string())
    Q.log(json.dumps({k: dec[k] for k in ("D1", "D2", "D3", "verdict_thermal", "verdict_vpvs")}, indent=1))


if __name__ == "__main__":
    main()

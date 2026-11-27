#!/usr/bin/env python3
"""Q2 circularity controls C5b and C5c (registered in data/paper_plan/q2_depth_limit/RESUME.md, 13:27:22 PST).

C5b: synthetic calibration ensemble (ensemble_v0.4.2/synth: truth = v0.4.2 x (1 + f), members s000-s099). Levels
     (7.5, 8.0 km/s) in the truth and in each member on the v0.4.2 P mask; per group of the main configuration (U2
     units, central D90), apparent following A_i = rho(D90, z_rec,i) - rho(D90, z_true). Flag if median A > 0 and
     >= 90 % of A_i > 0; then the real rho is corrected by median A and a following claim needs the corrected rho > 0
     with the block-bootstrap (1 deg blocks) 90 % lower bound > 0.
C5c: D-B112 null truth (ensemble_v0.4.2_null/synth; structure within 45 km of the segments set to layer means): rho
     (D90, z_rec,null_i) over the merged null members; a following claim needs rho_real > their 95th percentile (>= 15
     members); fewer: provisional.
Writes <Q2_OUT>/circularity_levels.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb_common as P  # noqa: E402
import q2_analysis as A  # noqa: E402
import q2_common as Q  # noqa: E402

MIN_NULL = 15
NBOOT = 2000


def rho(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return np.nan
    a, b = rankdata(x[ok]), rankdata(y[ok])
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def unit_levels(E, units, Z):
    """Unit value = median of the column levels of the unit's events (>= 50 % valid), as for the real model."""
    return np.array([A.unit_predictor(Z[E.iy.values[u["idx"]], E.ix.values[u["idx"]]]) for u in units])


def member_vp(d):
    z = np.load(d / "merged.npz")
    return np.transpose(z["vp"], (2, 1, 0)).astype(float)


def main():
    m = P.Model()
    E = pd.read_parquet(Q.BIG / "events_q2_main.parquet")
    U = pd.read_csv(Q.OUT / "units_main.csv")
    off = E.seg.isin(list(Q.OFFSHORE)).values
    units = []
    for grouping in ("seg", "frame"):
        Eg = E[off] if grouping == "seg" else E[E.frame.isin(["oceanic", "slab"])]
        for u in A.unit_table(Eg, grouping, "bin"):
            u["idx"] = Eg.index.values[u["idx"]]
            units.append(u)
    assert len(units) == len(U), (len(units), len(U))
    for u, (_, r) in zip(units, U.iterrows()):
        assert u["key"] == r.key and u["bx"] == r.bx and u["by"] == r.by
    D = U.D.values
    gsets = A.group_sets(U)
    lev = lambda V: {L: P.level_depth(V, m.mP, m.rock, m.dep, L) for L in Q.LEVELS}     # noqa: E731
    # synthetic calibration
    tr = np.load(Q.SYNTH / "truth" / "truth_model.npz")
    z_true = lev(np.transpose(tr["vp"], (2, 1, 0)).astype(float))
    syn_names = sorted(p.name for p in Q.SYNTH.iterdir() if p.name.startswith("s") and (p / "merged.npz").exists())
    ut = {L: unit_levels(E, units, z_true[L]) for L in Q.LEVELS}
    urec = {L: [] for L in Q.LEVELS}
    for n in syn_names:
        zl = lev(member_vp(Q.SYNTH / n))
        for L in Q.LEVELS:
            urec[L].append(unit_levels(E, units, zl[L]))
    urec = {L: np.array(v) for L, v in urec.items()}
    # null truth
    null_names = sorted(p.name for p in Q.NULL.iterdir() if p.name.startswith("s") and (p / "MERGED").exists()
                        and (p / "merged.npz").exists())
    un_true, unull = None, {L: [] for L in Q.LEVELS}
    if (Q.NULL / "truth" / "truth_model.npz").exists():
        tn = np.load(Q.NULL / "truth" / "truth_model.npz")
        zn = lev(np.transpose(tn["vp"], (2, 1, 0)).astype(float))
        un_true = {L: unit_levels(E, units, zn[L]) for L in Q.LEVELS}
    for n in null_names:
        zl = lev(member_vp(Q.NULL / n))
        for L in Q.LEVELS:
            unull[L].append(unit_levels(E, units, zl[L]))
    unull = {L: np.array(v) if len(v) else np.zeros((0, len(units))) for L, v in unull.items()}
    rng = np.random.default_rng(Q.SEED + 99)
    rows = []
    for g, msk in gsets.items():
        for L, lk in zip(Q.LEVELS, A.LEV_KEYS):
            zr = U[lk].values
            ok = msk & np.isfinite(zr)
            r = dict(group=g, level=lk, n_units=int(ok.sum()), n_synth=len(syn_names), n_null=len(null_names))
            if ok.sum() < 5:
                r["status"] = "not testable (< 5 units)"
                rows.append(r)
                continue
            rr = rho(D[ok], zr[ok])
            r["rho_real"] = rr
            # block bootstrap of the real rho (1 deg blocks)
            blk = U.block.values[ok]
            ub = np.unique(blk)
            idx_b = [np.where(blk == b)[0] for b in ub]
            bs = []
            for _ in range(NBOOT):
                pick = rng.integers(0, len(ub), len(ub))
                ii = np.concatenate([idx_b[k] for k in pick])
                bs.append(rho(D[ok][ii], zr[ok][ii]))
            bs = np.array(bs, float)
            r["rho_real_lo90"] = float(np.nanpercentile(bs, 5))
            r["n_blocks"] = int(len(ub))
            # C5b
            mt = msk & np.isfinite(ut[L])
            rt = rho(D[mt], ut[L][mt]) if mt.sum() >= 4 else np.nan
            Ai = []
            for i in range(len(syn_names)):
                mm = mt & np.isfinite(urec[L][i])
                Ai.append(rho(D[mm], urec[L][i][mm]) - rho(D[mm], ut[L][mm]) if mm.sum() >= 4 else np.nan)
            Ai = np.array(Ai, float)
            r["rho_truth"] = rt
            r["A_median"] = float(np.nanmedian(Ai)) if np.isfinite(Ai).any() else np.nan
            r["A_share_pos"] = float(np.mean(Ai[np.isfinite(Ai)] > 0)) if np.isfinite(Ai).any() else np.nan
            r["A_p95"] = float(np.nanpercentile(Ai, 95)) if np.isfinite(Ai).any() else np.nan
            r["synth_rec_bias_km"] = float(np.nanmedian(urec[L][:, mt] - ut[L][mt][None])) if mt.any() else np.nan
            flag = bool(np.isfinite(r["A_median"]) and r["A_median"] > 0 and r["A_share_pos"] >= 0.9)
            r["C5b_flag"] = flag
            corr = r["A_median"] if flag else 0.0
            r["rho_corrected"] = rr - corr
            r["rho_corrected_lo90"] = r["rho_real_lo90"] - corr
            r["C5b_pass"] = bool(r["rho_corrected"] > 0 and r["rho_corrected_lo90"] > 0)
            # C5c
            if len(null_names):
                rn = []
                for i in range(len(null_names)):
                    mm = msk & np.isfinite(unull[L][i])
                    rn.append(rho(D[mm], unull[L][i][mm]) if mm.sum() >= 4 else np.nan)
                rn = np.array(rn, float)
                v = rn[np.isfinite(rn)]
                r["rho_null_median"] = float(np.median(v)) if len(v) else np.nan
                r["rho_null_p95"] = float(np.percentile(v, 95)) if len(v) else np.nan
                r["n_null_valid"] = int(len(v))
                if un_true is not None:
                    mm = msk & np.isfinite(un_true[L])
                    r["rho_null_truth"] = rho(D[mm], un_true[L][mm]) if mm.sum() >= 4 else np.nan
                r["C5c_pass"] = bool(len(v) and rr > r["rho_null_p95"])
                r["C5c_status"] = "final" if len(v) >= MIN_NULL else f"provisional ({len(v)} of >= {MIN_NULL} members)"
            else:
                r["C5c_status"] = "pending (no null member merged)"
            r["status"] = "ok"
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(Q.OUT / "circularity_levels.csv", index=False, float_format="%.4g")
    Q.log(out[out.status.eq("ok")].to_string())


if __name__ == "__main__":
    main()

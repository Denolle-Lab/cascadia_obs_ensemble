#!/usr/bin/env python3
"""Q2 Monte Carlo precision (coordinator request after the final report; no rule changes).

Reruns the main configuration with N independent seeds at the registered draw count (1000) and records every
registered probability of the decision rules:
  P_sit (sits if >= 0.5), P_far (falsified if >= 0.95), P_skill_pos (follows if >= 0.9, does not follow if <= 0.1),
  preference P_T (primary, best of family, cross-validated; H_T if >= 0.9, H_C if <= 0.1),
for the registered predictors (six isotherms, two levels) and every registered group, and the registered answer per
group. The registered run (seed SEED + 0) is the reference; probabilities within +-0.03 of a threshold there are the
ones reported. Writes <Q2_OUT>/mc_precision_runs.csv (all seeds, all probabilities), mc_precision.csv (summary of the
near-threshold probabilities) and mc_precision_answers.csv (answer frequencies per group).

usage: q2_seeds.py [N_SEEDS] [N_PROCESSES]     (defaults 20, 4; nice 10 from run_q2.sh or the caller)
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import q2_analysis as A  # noqa: E402
import q2_common as Q  # noqa: E402

SEED0 = Q.SEED + 5000          # study seeds: SEED0 + k, independent of the registered seeds (SEED + 0..14)
THR = {"P_sit": [0.5], "P_far": [0.95], "P_skill_pos": [0.9, 0.1], "primary_P_T": [0.9, 0.1],
       "best_of_family_P_T": [0.9, 0.1], "cv_P_T": [0.9, 0.1]}
_E = {}


def one(seed):
    if "E" not in _E:
        c, meta = Q.load_events("v6.1")
        _E["E"], _E["meta"] = Q.select(c, ("A", "B"), 3.0), meta
    cfg, E, U, D, Dsf, preds, meta = A.run_config("main", _E["E"], _E["meta"], seed=seed)
    G, PR = A.evaluate(U, D, preds)
    AN = A.answers(G, PR, U)
    rows = []
    g = G[G.level.isin(A.ISO_KEYS + A.LEV_KEYS)]
    for _, r in g.iterrows():
        for p in ("P_sit", "P_far", "P_skill_pos"):
            if p in r and pd.notna(r[p]):
                if p == "P_skill_pos" and not bool(r.get("shape_testable", False)):
                    continue
                rows.append(dict(seed=seed, group=r.group, item=r.level, prob=p, value=float(r[p])))
    for _, r in PR.iterrows():
        for p in ("primary_P_T", "best_of_family_P_T", "cv_P_T"):
            if p in r and pd.notna(r[p]):
                rows.append(dict(seed=seed, group=r.group, item="preference", prob=p, value=float(r[p])))
    ans = AN[["group", "answer"]].assign(seed=seed)
    return pd.DataFrame(rows), ans


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    npro = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    seeds = [Q.SEED + 0] + [SEED0 + k for k in range(n)]
    with Pool(npro) as pool:
        res = pool.map(one, seeds)
    runs = pd.concat([r[0] for r in res], ignore_index=True)
    ans = pd.concat([r[1] for r in res], ignore_index=True)
    runs.to_csv(Q.OUT / "mc_precision_runs.csv", index=False, float_format="%.4f")
    reg = runs[runs.seed.eq(Q.SEED)]
    study = runs[~runs.seed.eq(Q.SEED)]
    out = []
    for _, r in reg.iterrows():
        for t in THR[r.prob]:
            if abs(r.value - t) <= 0.03 + 1e-12:
                v = study[study.group.eq(r.group) & study.item.eq(r["item"]) & study.prob.eq(r.prob)].value
                if not len(v):
                    continue
                m, sd = float(v.mean()), float(v.std(ddof=1))
                need = np.nan
                if sd > 0 and abs(m - t) > 0:
                    need = Q.NDRAW * (2.0 * sd / abs(m - t)) ** 2          # draws for |mean - t| = 2 SD
                out.append(dict(group=r.group, item=r["item"], prob=r.prob, threshold=t, registered=float(r.value),
                                n_seeds=int(len(v)), mean=m, sd=sd, min=float(v.min()), max=float(v.max()),
                                share_at_or_above=float((v >= t).mean()),
                                straddles=bool(v.min() < t <= v.max()),
                                draws_for_2sd=need))
    S = pd.DataFrame(out).sort_values(["group", "item", "prob"])
    S.to_csv(Q.OUT / "mc_precision.csv", index=False, float_format="%.4f")
    fa = (ans[~ans.seed.eq(Q.SEED)].groupby(["group", "answer"]).size().rename("n_seeds").reset_index())
    fa = fa.merge(ans[ans.seed.eq(Q.SEED)][["group", "answer"]].rename(columns={"answer": "registered_answer"}),
                  on="group", how="left")
    fa.to_csv(Q.OUT / "mc_precision_answers.csv", index=False)
    Q.log(S.to_string())
    Q.log(fa[fa.group.isin(fa.groupby("group").answer.nunique().loc[lambda x: x > 1].index)].to_string())


if __name__ == "__main__":
    main()

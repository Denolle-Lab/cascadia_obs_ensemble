"""Track B, v6.0 (D-B095 follow-up): where to stop the SSST CharDist schedule. Scores of the stopping-distance test on
the group-2 clusters (georgia, gorda; b4_v6_termtest_sets.py --group 2, runs of b4_v6_termtest.py --arms
ssst,coh_stops) and the choice by the rule written in /wd1/mdenolle_data/reloc_v6/termtest/RESUME.md before the scores.

Arms (one SSST chain 9999 -> 16 -> 8 -> 4 -> 2 km per run): ssst16, ssst8, ssst4, ssst2 = the NLLoc pass after the
16, 8, 4, 2 km correction (it2, it3, it4, it5); coh16, coh8, coh4, coh2 = the coherence stage (b4_v6_termtest.
coherence_run, events with Len3 <= 5 km relocated, the others keep the SSST position) on each of them.

Criteria as b4_v6_termtest_score.py: 1. held-out stations (-18's fold_score, 3 folds, common rows of all 8 arms),
2. CC fit of the full run and repeater separation, 3. depths against PNSN (georgia) / NCSN (gorda) through ANSS ComCat,
above-surface and floor-limited counts.

Choice: S(d) = mean over clusters of [MAD_P + MAD_S] of arm coh<d>; d* = argmin; if the 95 % paired-bootstrap interval
of S(d+) - S(d*) for the next coarser stop d+ includes 0, d+ is taken (one step). Bootstrap: 2,000 replicates,
event pairs resampled within each cluster (both phases of a pair together), the same resample for every arm.

  python b4_v6_termtest_stop_score.py [--sets georgia,gorda] [--nboot 2000]
      -> /wd1/mdenolle_data/reloc_v6/termtest/score_stop/{scores.json, choice.json, heldout_rows_<set>.parquet}
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b4_v6_termtest as T  # noqa: E402
import b4_v6_termtest_score as SC  # noqa: E402
import b5_cmp_common as CM  # noqa: E402

KM = {2: 16, 3: 8, 4: 4, 5: 2}
STOPS_KM = [16, 8, 4, 2]                       # coarse -> fine
SC.OUT = T.TT / "score_stop"
SC.ARMS = ([(f"ssst{KM[k]}", f"ssst/it{k}", None) for k in sorted(KM)]
           + [(f"coh{KM[k]}", f"coherence_it{k}", f"ssst/it{k}") for k in sorted(KM)])
SC.MAIN = tuple(a for a, _, _ in SC.ARMS)
SC.ANSS_RULE = {**SC.ANSS_RULE, "georgia": ("uw", "PNSN (ANSS ComCat, net uw)", 2.0, 10.0),
                "gorda": ("nc", "NCSN (ANSS ComCat, net nc)", 3.0, 20.0)}
ARMS = list(SC.MAIN)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def mad(v):
    return float(np.median(np.abs(v - np.median(v)))) if len(v) else np.nan


def heldout_rows(s):
    """common held-out rows of all arms (score per arm), pooled over folds, + per-arm own-event scores, v5.1."""
    import b5_relocate as BR
    cc = CM.cc_rows(s)
    CM._G["G"] = BR.Grids(T.set_dir(s) / "grids_base")
    ev = pd.read_csv(T.SETDIR / f"events_{s}.csv").set_index("ID")
    parts, own, v51, nloc = [], {}, [], {}
    for run in T.run_names()[1:]:
        ho = T.held_out(s, run)
        tabs, located = {}, {}
        for arm in ARMS:
            e, _ = SC.positions(s, run, arm)
            pos = e.set_index("ID")[["lat", "lon", "depth_t"]].rename(columns={"depth_t": "depth"})
            located[arm] = set(pos.index)
            tabs[arm] = CM.fold_score(cc, pos, ho)
            own.setdefault(arm, []).append(tabs[arm])
            nloc.setdefault(arm, []).append(len(pos))
        common = set.intersection(*located.values())
        base = None
        for arm in ARMS:
            h = tabs[arm]
            h = h[h.ID1.isin(common) & h.ID2.isin(common)].set_index(["ID1", "ID2", "station", "phase"]).score
            base = h.rename(arm).to_frame() if base is None else base.join(h.rename(arm), how="inner")
        parts.append(base.reset_index().assign(run=run))
        v51.append(CM.fold_score(cc, ev[["lat", "lon", "depth"]], ho))
    C = pd.concat(parts, ignore_index=True)
    C["pair"] = pd.factorize(pd.Series(list(zip(C.ID1, C.ID2))))[0]
    V = pd.concat(v51)
    out = {"v5.1": {ph: dict(n=len(g), mad_ms=mad(g.score.values) * 1e3) for ph, g in V.groupby("phase")},
           "own": {a: {ph: dict(n=len(g), mad_ms=mad(g.score.values) * 1e3,
                                rms_ms=float(np.sqrt(np.mean(g.score.values ** 2))) * 1e3)
                       for ph, g in pd.concat(own[a]).groupby("phase")} for a in ARMS},
           "located_per_fold": nloc}
    return C, out


def resample_index(codes, rng):
    """row indices of one cluster bootstrap resample (rows grouped by codes, all rows of a drawn group kept)."""
    order = np.argsort(codes, kind="stable")
    cnt = np.bincount(codes)
    starts = np.r_[0, np.cumsum(cnt)[:-1]]
    pick = rng.integers(0, len(cnt), len(cnt))
    lens = cnt[pick]
    offs = np.repeat(starts[pick] - np.r_[0, np.cumsum(lens)[:-1]], lens) + np.arange(lens.sum())
    return order[offs]


def boot(tables, nboot, seed=11):
    """replicates of MAD (ms) per cluster, phase and arm: array (nboot + 1, clusters, 2, arms); row 0 = estimate."""
    rng = np.random.default_rng(seed)
    sets = list(tables)
    R = np.full((nboot + 1, len(sets), 2, len(ARMS)), np.nan)
    prep = {}
    for i, s in enumerate(sets):
        C = tables[s]
        prep[s] = (C.pair.to_numpy(), (C.phase == "P").to_numpy(), C[ARMS].to_numpy(float))
    for b in range(nboot + 1):
        for i, s in enumerate(sets):
            codes, isP, X = prep[s]
            idx = np.arange(len(codes)) if b == 0 else resample_index(codes, rng)
            for j, m in enumerate((isP[idx], ~isP[idx])):
                Y = X[idx][m]
                med = np.median(Y, axis=0)
                R[b, i, j] = np.median(np.abs(Y - med), axis=0) * 1e3
    return sets, R


def ci(x):
    x = np.asarray(x, float)
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="georgia,gorda")
    ap.add_argument("--nboot", type=int, default=2000)
    a = ap.parse_args()
    SC.OUT.mkdir(parents=True, exist_ok=True)
    sets = a.sets.split(",")
    tables, res = {}, {}
    for s in sets:
        t0 = time.time()
        C, ho = heldout_rows(s)
        C.to_parquet(SC.OUT / f"heldout_rows_{s}.parquet", index=False)
        tables[s] = C
        res[s] = dict(heldout=ho, common_rows={ph: int((C.phase == ph).sum()) for ph in ("P", "S")},
                      common_pairs=int(C.pair.nunique()),
                      ccfit=SC.ccfit(s, a.nboot), depth=SC.depth(s, a.nboot))
        log(f"{s} scored, {time.time() - t0:.0f} s")
    names, R = boot(tables, a.nboot)
    est, rep = R[0], R[1:]
    for i, s in enumerate(names):
        h = res[s]["heldout"]
        h["common"] = {ph: {arm: [float(est[i, j, k])] + ci(rep[:, i, j, k]) for k, arm in enumerate(ARMS)}
                       for j, ph in enumerate(("P", "S"))}
        # coherence minus no coherence, per stop and phase (negative = coherence helps)
        h["coh_minus_ssst"] = {ph: {str(km): [float(est[i, j, ARMS.index(f"coh{km}")] - est[i, j, ARMS.index(f"ssst{km}")])]
                                    + ci(rep[:, i, j, ARMS.index(f"coh{km}")] - rep[:, i, j, ARMS.index(f"ssst{km}")])
                                    for km in STOPS_KM} for j, ph in enumerate(("P", "S"))}
        # P + S per arm for this cluster
        h["sum_PS"] = {arm: [float(est[i, :, k].sum())] + ci(rep[:, i, :, k].sum(axis=1)) for k, arm in enumerate(ARMS)}
    # the rule: S(d) = mean over clusters of MAD_P + MAD_S of coh<d>
    Sx = {km: (est[:, :, ARMS.index(f"coh{km}")].sum(axis=1).mean(),
               rep[:, :, :, ARMS.index(f"coh{km}")].sum(axis=2).mean(axis=1)) for km in STOPS_KM}
    Sn = {km: (est[:, :, ARMS.index(f"ssst{km}")].sum(axis=1).mean(),
               rep[:, :, :, ARMS.index(f"ssst{km}")].sum(axis=2).mean(axis=1)) for km in STOPS_KM}
    best = min(STOPS_KM, key=lambda km: Sx[km][0])
    i = STOPS_KM.index(best)
    coarser = STOPS_KM[i - 1] if i > 0 else None
    choice = dict(score_coh={str(km): [float(Sx[km][0])] + ci(Sx[km][1]) for km in STOPS_KM},
                  score_no_coh={str(km): [float(Sn[km][0])] + ci(Sn[km][1]) for km in STOPS_KM},
                  diffs_coh={f"{km}-{best}": [float(Sx[km][0] - Sx[best][0])] + ci(Sx[km][1] - Sx[best][1])
                             for km in STOPS_KM if km != best},
                  best_km=best, next_coarser_km=coarser)
    if coarser is None:
        choice.update(chosen_km=best, reason="the best stop is already the coarsest tested")
    else:
        lo, hi = ci(Sx[coarser][1] - Sx[best][1])
        b_lo, b_hi = ci(Sx[best][1])
        c_lo, c_hi = ci(Sx[coarser][1])
        overlap = not (c_lo > b_hi or b_lo > c_hi)
        same = lo <= 0 <= hi
        choice.update(diff_coarser_minus_best=[float(Sx[coarser][0] - Sx[best][0]), lo, hi],
                      individual_intervals_overlap=bool(overlap), chosen_km=coarser if same else best,
                      reason=(f"{coarser} km not distinguishable from {best} km (interval of the difference includes 0)"
                              if same else f"{coarser} km worse than {best} km (interval of the difference excludes 0)"))
    (SC.OUT / "scores.json").write_text(json.dumps(res, indent=1, default=float))
    (SC.OUT / "choice.json").write_text(json.dumps(choice, indent=1, default=float))
    print(json.dumps(choice, indent=1, default=float))
    for s in names:
        print(s, {ph: {arm: round(v[0], 1) for arm, v in d.items()} for ph, d in res[s]["heldout"]["common"].items()})
        print("   coh - ssst", {ph: {km: [round(x, 1) for x in v] for km, v in d.items()}
                                for ph, d in res[s]["heldout"]["coh_minus_ssst"].items()})


if __name__ == "__main__":
    main()

"""Track B, v6.0 station-term test: scores of the arms of b4_v6_termtest.py on three independent criteria.

Arms (per set and run): v5.1 (catalog positions, reference), static (static_oct), ssst_it0..ssst_it5 (it5 = SSST to
2 km alone), A (static_coherence: the coherence stage on static; events it does not relocate keep their static
position), B (coherence: the coherence stage on ssst_it5; the others keep their it5 position). LOCATED events,
true depths (Earth-flattening undone), as b4_v6_ssst_metrics.located_frame.

1. Held-out stations (session -18's D-B078 scorer, b5_cmp_common.fold_score of tomo/b5 b0dffcdd, imported): for each
   fold, the accepted CC dt rows (b5_cmp_common.cc_rows cuts: status ok, above threshold, not a failed-triangle row,
   not straddling a clock epoch) at the fold's held-out stations are predicted from the fold run's positions; each
   row's residual minus the pair's median residual at the training stations; pooled over the 3 folds; MAD (median
   absolute deviation about the median) and rms per phase. One change: the time grid of each station-phase is the
   one NLLoc used (runs/g51/<set>/grids_base: fine 1 km grid, or the wide 2 km grid when the fine grid does not hold
   the search box), not always the fine grid (-18's GRIDS). The two are the same for the stations whose fine grid
   holds the box; for the others (91 of 143 station-phases in blancow) the fine-grid lookup would be clipped at the
   grid edge.
   Scored on each arm's own located events (as -18's legs) and, for the paired comparison, on the rows whose two
   events are located in every arm of the fold ("common").
2. CC fit of the full run (b4_v6_ssst_metrics.cc_score, same cuts and formula): r = dt_cc + T0v51_1 - T0v51_2 -
   (OT_1 - OT_2) - [(TTpred + Tcorr)_1 - (TTpred + Tcorr)_2], minus the pair's median r; median |.| (= the "median
   per-pair residual" of the v6-prep table) and rms; repeater pairs (qc/<set>_repeaters.csv): median 3-D separation.
3. Depths of the full run against published catalogs: blancow - Lange et al. 2026 (absolute OBS locations, pair
   table of b4_lit_paired.py, IDs); puget - PNSN, mtj - NCSN: ANSS ComCat events (net uw / nc) of
   data/magnitude_v4/anss_2010-16.csv paired one-to-one with the v5.1 events (greedy by |dt|, then distance; puget
   |dt| <= 2 s and <= 10 km, mtj <= 3 s and <= 20 km). Median (arm - reference) and median |arm - reference|, tiers
   A-D and A-B (v5.1 tier, fixed subset). Events above the merged surface (must be 0) and floor-limited events.

Bootstrap: 2,000 replicates, paired (same resample for every arm), resampling event pairs (criteria 1, 2: a pair's
rows in every fold go together) or events (criterion 3); 95 % percentile intervals of each arm's statistic and of
B - A. Decision rule: /wd1/mdenolle_data/reloc_v6/termtest/RESUME.md (written before the scores).

  python b4_v6_termtest_score.py [--sets puget,mtj,blancow] [--nboot 2000]
      -> /wd1/mdenolle_data/reloc_v6/termtest/score/{scores.json, heldout.csv, ccfit.csv, depth.csv, verdict.csv,
         pos/<set>_<run>_<arm>.parquet}
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
sys.path.insert(0, str(HERE.parent / "08_relocation"))
import a2_parse_hyp as PH  # noqa: E402
import b4_v6_ssst as S  # noqa: E402
import b4_v6_ssst_metrics as M  # noqa: E402
import b4_v6_termtest as T  # noqa: E402
import b5_cmp_common as CM  # noqa: E402

M.CC_DIR = T.CC_DIR          # cc rows and repeaters of the test sets
CM.CC = T.CC_DIR
OUT = T.TT / "score"
ANSS = Path("/home/mdenolle/cascadia_obs_ensemble/data/magnitude_v4/anss_2010-16.csv")
LANGE = Path("/wd1/mdenolle_data/tomo_v1/depth_literature/pairs/blanco_lange2026_abs_pairs.csv")
ANSS_RULE = {"puget": ("uw", "PNSN (ANSS ComCat, net uw)", 2.0, 10.0), "mtj": ("nc", "NCSN (ANSS ComCat, net nc)", 3.0, 20.0)}
NS = len(S.SCHEDULE)
ARMS = ([("static", "static_oct", None)] + [(f"ssst_it{k}", f"ssst/it{k}", None) for k in range(NS + 1)]
        + [("A", "static_coherence", "static_oct"), ("B", "coherence", f"ssst/it{NS}")])
MAIN = ("static", f"ssst_it{NS}", "A", "B")
REL = 0.05          # materiality: 5 % of A (held-out MAD, CC fit)
DZ_MAT = 0.2        # km (median |depth - reference|)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------------------------------------- positions
_V4 = {}


def positions(s, run, arm):
    """LOCATED events of an arm with true depth etc. (cached parquet). Coherence arms: events not relocated by the
    coherence stage (or not LOCATED there) keep the base arm's location."""
    f = OUT / "pos" / f"{s}_{run}_{arm}.parquet"
    fp = OUT / "pos" / f"{s}_{run}_{arm}_phases.parquet"
    if f.exists() and fp.exists():
        return pd.read_parquet(f), pd.read_parquet(fp)
    if "v4" not in _V4:
        _V4["v4"] = PH.read_v4()
    sub, base = dict((a, (b, c)) for a, b, c in ARMS)[arm]
    rd = T.run_dir(s, run) / sub
    if not (rd / "run.json").exists():
        return None, None
    ev, ph = M.read_arm(rd)
    n_coh_notloc = 0
    if base is not None:
        n_coh_notloc = int((ev.status != "LOCATED").sum()) if len(ev) else 0
        ev = ev[ev.status == "LOCATED"] if len(ev) else ev
        ph = ph[ph.ID.isin(ev.ID)] if len(ph) else ph
        e5, p5 = M.read_arm(T.run_dir(s, run) / base)
        keep = ~e5.ID.isin(ev.ID)
        ev = pd.concat([ev, e5[keep]], ignore_index=True)
        ph = pd.concat([ph, p5[p5.ID.isin(e5.ID[keep])]], ignore_index=True)
    e = M.located_frame(ev, _V4["v4"])
    e["coh_not_located"] = n_coh_notloc
    f.parent.mkdir(parents=True, exist_ok=True)
    keep_cols = ["ID", "time", "lat", "lon", "depth", "depth_t", "x", "y", "tier", "above_surface", "floor_limited",
                 "ell_len3", "rms", "coh_not_located"]
    e = e[[c for c in keep_cols if c in e]].copy()
    e["time"] = e["time"].astype(str)
    e.to_parquet(f, index=False)
    ph.to_parquet(fp, index=False)
    return e, ph


def v51_positions(s):
    ev = pd.read_csv(T.SETDIR / f"events_{s}.csv")
    c = M.v51_arm(set(ev.ID))
    return c


# ---------------------------------------------------------------------------------------------------------- bootstrap
def boot_groups(keys, nboot, seed):
    """index arrays of nboot cluster-bootstrap resamples over the unique keys (rows of a key go together)."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(pd.Series(list(keys)) if not isinstance(keys, pd.Series) else keys)
    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(len(uniq)))
    ends = np.r_[starts[1:], len(codes)]
    for _ in range(nboot):
        pick = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([order[starts[k]:ends[k]] for k in pick]) if len(pick) else np.array([], int)


def mad(v):
    v = np.asarray(v, float)
    return float(np.median(np.abs(v - np.median(v)))) if len(v) else np.nan


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) else (np.nan, np.nan)


def paired_boot(table, keycol, stat, arms, nboot, seed):
    """table: rows with keycol and one value column per arm; stat(values) -> float. Returns {arm: (value, lo, hi)},
    and the B - A difference (value, lo, hi)."""
    vals = {a: table[a].to_numpy(float) for a in arms}
    est = {a: stat(vals[a]) for a in arms}
    reps = {a: [] for a in arms}
    for idx in boot_groups(table[keycol], nboot, seed):
        for a in arms:
            reps[a].append(stat(vals[a][idx]))
    out = {a: (est[a], *ci(reps[a])) for a in arms}
    if "A" in arms and "B" in arms:
        d = np.array(reps["B"]) - np.array(reps["A"])
        out["B-A"] = (est["B"] - est["A"], *ci(d))
    return out


# -------------------------------------------------------------------------------------------- criterion 1: held out
def heldout(s, nboot):
    cc = CM.cc_rows(s)
    import b5_relocate as BR
    CM._G["G"] = BR.Grids(T.set_dir(s) / "grids_base")
    ev = pd.read_csv(T.SETDIR / f"events_{s}.csv").set_index("ID")
    rows = {}
    own = {}
    v51 = []
    for run in T.run_names()[1:]:
        ho = T.held_out(s, run)
        located = {}
        for arm, _, _ in ARMS:
            e, _ = positions(s, run, arm)
            if e is None:
                continue
            pos = e.set_index("ID")[["lat", "lon", "depth_t"]].rename(columns={"depth_t": "depth"})
            located[arm] = set(pos.index)
            h = CM.fold_score(cc, pos, ho)
            h["run"] = run
            own.setdefault(arm, {})[run] = h
        h = CM.fold_score(cc, ev[["lat", "lon", "depth"]], ho)
        h["run"] = run
        v51.append(h)
        rows[run] = located
    res = {"v5.1": {}, "own": {}, "common": {}}
    V = pd.concat(v51)
    for ph, g in V.groupby("phase"):
        res["v5.1"][ph] = dict(n=len(g), mad_ms=mad(g.score) * 1e3, rms_ms=float(np.sqrt(np.mean(g.score ** 2))) * 1e3)
    for arm, hs in own.items():
        H = pd.concat(list(hs.values()))
        res["own"][arm] = {ph: dict(n=len(g), mad_ms=mad(g.score) * 1e3,
                                    rms_ms=float(np.sqrt(np.mean(g.score ** 2))) * 1e3,
                                    p90_ms=float(np.percentile(np.abs(g.score), 90)) * 1e3)
                           for ph, g in H.groupby("phase")}
        res["own"][arm]["located_per_fold"] = [len(rows[r].get(arm, ())) for r in T.run_names()[1:]]
    # common rows: both events located in every arm of the fold
    arms = [a for a, _, _ in ARMS if a in own and all(r in own[a] for r in T.run_names()[1:])]
    parts = []
    for run in T.run_names()[1:]:
        common = set.intersection(*[rows[run][a] for a in arms])
        base = None
        for a in arms:
            h = own[a][run]
            h = h[h.ID1.isin(common) & h.ID2.isin(common)].set_index(["ID1", "ID2", "station", "phase"]).score
            base = h.rename(a).to_frame() if base is None else base.join(h.rename(a), how="inner")
        base = base.reset_index()
        base["run"] = run
        # events the coherence stage relocated (LOCATED there) in both A and B
        rel = None
        for sub in ("static_coherence", "coherence"):
            ev, _ = M.read_arm(T.run_dir(s, run) / sub)
            ids = set(ev.ID[ev.status == "LOCATED"]) if len(ev) else set()
            rel = ids if rel is None else rel & ids
        base["coh_both"] = base.ID1.isin(rel) & base.ID2.isin(rel)
        parts.append(base)
    C = pd.concat(parts, ignore_index=True)
    C["pair"] = list(zip(C.ID1, C.ID2))
    mainarms = [a for a in MAIN if a in C]
    for ph, g in C.groupby("phase"):
        g = g.reset_index(drop=True)
        r_mad = paired_boot(g, "pair", lambda v: mad(v) * 1e3, mainarms, nboot, 1)
        r_rms = paired_boot(g, "pair", lambda v: float(np.sqrt(np.mean(v ** 2))) * 1e3, mainarms, nboot, 2)
        res["common"][ph] = dict(n=len(g), pairs=int(g.pair.nunique()), mad=r_mad, rms=r_rms,
                                 all_mad={a: mad(g[a]) * 1e3 for a in arms},
                                 all_rms={a: float(np.sqrt(np.mean(g[a] ** 2))) * 1e3 for a in arms})
        # split: rows whose two events the coherence stage relocated in both A and B, and the other rows
        for lab, sub in (("coh_both", g[g.coh_both]), ("rest", g[~g.coh_both])):
            sub = sub.reset_index(drop=True)
            if sub.pair.nunique() >= 20:
                res["common"][ph][lab] = dict(n=len(sub), pairs=int(sub.pair.nunique()),
                                              mad=paired_boot(sub, "pair", lambda v: mad(v) * 1e3, mainarms, nboot, 8))
            else:
                res["common"][ph][lab] = dict(n=len(sub), pairs=int(sub.pair.nunique()))
    return res


# ------------------------------------------------------------------------------------------------ criterion 2: CC fit
def cc_fit_rows(e, ph, cc):
    """per-row pair-demeaned residual r of b4_v6_ssst_metrics.cc_score (label cc), with the pair key."""
    ot = pd.Series(M.ot_seconds(e.time).to_numpy(), index=e.ID.to_numpy())
    p = ph[ph.ID.isin(e.ID)].copy()
    p["pred"] = p.ttpred + p.tcorr
    p = p.drop_duplicates(["ID", "sta", "phase"]).set_index(["ID", "sta", "phase"]).pred
    c = cc[cc.ID1.isin(ot.index) & cc.ID2.isin(ot.index)].copy()
    k1 = list(zip(c.ID1, c.station, c.phase))
    k2 = list(zip(c.ID2, c.station, c.phase))
    have = np.array([(a in p.index) and (b in p.index) for a, b in zip(k1, k2)], bool)
    c = c[have]
    p1 = p.loc[[k for k, h in zip(k1, have) if h]].to_numpy()
    p2 = p.loc[[k for k, h in zip(k2, have) if h]].to_numpy()
    # T0 are POSIX seconds here (b4_v6_ssst_metrics parses them with pd.to_datetime, i.e. as ns: wrong by a
    # constant per pair, which the pair median below removes; same result)
    t0 = c.T0_1.to_numpy(float) - c.T0_2.to_numpy(float)
    dot = ot.loc[c.ID1].to_numpy() - ot.loc[c.ID2].to_numpy()
    r = c.dt_cc.to_numpy() + t0 - dot - (p1 - p2)
    d = pd.DataFrame({"ID1": c.ID1.to_numpy(), "ID2": c.ID2.to_numpy(), "station": c.station.to_numpy(),
                      "phase": c.phase.to_numpy(), "r": r})
    n = d.groupby(["ID1", "ID2"]).r.transform("size")
    d = d[n >= 3].copy()
    d["rr"] = d.r - d.groupby(["ID1", "ID2"]).r.transform("median")
    return d


def ccfit(s, nboot):
    cc = M.cc_rows(s)
    per = {}
    out = {"arms": {}}
    pos = {}
    for arm, _, _ in ARMS:
        e, ph = positions(s, "full", arm)
        if e is None:
            continue
        d = cc_fit_rows(e, ph, cc)
        per[arm] = d.set_index(["ID1", "ID2", "station", "phase"]).rr
        pos[arm] = e.set_index("ID")[["x", "y", "depth_t"]]
        out["arms"][arm] = dict(rows=len(d), mad_pair_ms=float(np.median(np.abs(d.rr))) * 1e3,
                                rms_pair_ms=float(np.sqrt(np.mean(d.rr ** 2))) * 1e3,
                                located=len(e), above_surface=int(e.above_surface.sum()),
                                floor_limited=int(e.floor_limited.sum()))
    mainarms = [a for a in MAIN if a in per]
    J = pd.concat([per[a].rename(a) for a in mainarms], axis=1, join="inner").reset_index()
    J["pair"] = list(zip(J.ID1, J.ID2))
    out["common_rows"] = len(J)
    out["mad"] = paired_boot(J, "pair", lambda v: float(np.median(np.abs(v))) * 1e3, mainarms, nboot, 3)
    out["rms"] = paired_boot(J, "pair", lambda v: float(np.sqrt(np.mean(v ** 2))) * 1e3, mainarms, nboot, 4)
    # repeaters
    rp = pd.read_csv(T.CC_DIR / "qc" / f"{s}_repeaters.csv")
    common = set.intersection(*[set(pos[a].index) for a in mainarms]) if mainarms else set()
    rp = rp[rp.ID1.isin(common) & rp.ID2.isin(common)]
    if len(rp):
        R = pd.DataFrame({a: np.linalg.norm(pos[a].loc[rp.ID1].to_numpy() - pos[a].loc[rp.ID2].to_numpy(), axis=1)
                          for a in mainarms})
        R["pair"] = list(zip(rp.ID1, rp.ID2))
        out["repeaters"] = dict(n=len(R), sep=paired_boot(R, "pair", lambda v: float(np.median(v)), mainarms, nboot, 5))
        allarms = [a for a in pos if set(rp.ID1) <= set(pos[a].index) and set(rp.ID2) <= set(pos[a].index)]
        out["repeaters"]["all_median_km"] = {a: float(np.median(np.linalg.norm(
            pos[a].loc[rp.ID1].to_numpy() - pos[a].loc[rp.ID2].to_numpy(), axis=1))) for a in allarms}
        c51 = v51_positions(s).set_index("ID")[["x", "y", "depth_t"]]
        ok = rp.ID1.isin(c51.index) & rp.ID2.isin(c51.index)
        out["repeaters"]["v5.1_median_km"] = float(np.median(np.linalg.norm(
            c51.loc[rp.ID1[ok]].to_numpy() - c51.loc[rp.ID2[ok]].to_numpy(), axis=1))) if ok.any() else np.nan
    else:
        out["repeaters"] = dict(n=0)
    return out


# ------------------------------------------------------------------------------------------------ criterion 3: depths
def anss_pairs(s):
    net, ref, dtmax, dmax = ANSS_RULE[s]
    ev = pd.read_csv(T.SETDIR / f"events_{s}.csv")
    a = pd.read_csv(ANSS, usecols=["time", "latitude", "longitude", "depth", "mag", "net", "id"])
    a = a[a.net == net].copy()
    a["t"] = pd.to_datetime(a.time, utc=True).astype("int64") / 1e9
    et = pd.to_datetime(ev.time, utc=True).astype("int64").to_numpy() / 1e9
    cand = []
    at = a.t.to_numpy()
    o = np.argsort(at)
    at_s = at[o]
    for i, (t, la, lo) in enumerate(zip(et, ev.lat, ev.lon)):
        j0, j1 = np.searchsorted(at_s, t - dtmax), np.searchsorted(at_s, t + dtmax, side="right")
        for j in o[j0:j1]:
            r = a.iloc[j]
            dx = np.hypot((r.latitude - la) * 111.195, (r.longitude - lo) * 111.195 * np.cos(np.radians(la)))
            if dx <= dmax:
                cand.append((abs(r.t - t), dx, int(ev.ID.iloc[i]), r.id, float(r.depth), float(r.mag)))
    cand.sort()
    used_e, used_a, out = set(), set(), []
    for dt, dx, eid, aid, z, m in cand:
        if eid in used_e or aid in used_a:
            continue
        used_e.add(eid)
        used_a.add(aid)
        out.append(dict(ID=eid, ref_id=aid, dt=dt, dx_km=dx, ref_depth=z, ref_mag=m))
    p = pd.DataFrame(out, columns=["ID", "ref_id", "dt", "dx_km", "ref_depth", "ref_mag"])
    p.to_csv(OUT / f"depth_pairs_{s}.csv", index=False)
    return p, ref


def depth(s, nboot):
    if s == "blancow":
        p = pd.read_csv(LANGE)[["ID", "ref_depth"]]
        ref = "Lange et al. 2026 (absolute)"
    else:
        p, ref = anss_pairs(s)
    ev = pd.read_csv(T.SETDIR / f"events_{s}.csv")[["ID", "tier"]].rename(columns={"tier": "tier_v51"})
    p = p.merge(ev, on="ID")
    out = {"reference": ref, "pairs": int(len(p))}
    cols = {}
    for arm, _, _ in ARMS:
        e, _ = positions(s, "full", arm)
        if e is not None:
            cols[arm] = e.set_index("ID").depth_t
    c51 = v51_positions(s).set_index("ID").depth_t
    cols["v5.1"] = c51
    mainarms = [a for a in MAIN if a in cols]
    for lab, tt in (("A-D", list("ABCD")), ("A-B", list("AB"))):
        q = p[p.tier_v51.isin(tt)].set_index("ID")
        D = pd.DataFrame({a: cols[a].reindex(q.index) - q.ref_depth for a in list(cols)}).dropna()
        if not len(D):
            out[lab] = dict(n=0)
            continue
        D["ev"] = D.index
        D = D.reset_index(drop=True)
        out[lab] = dict(n=len(D),
                        bias=paired_boot(D, "ev", lambda v: float(np.median(v)), mainarms, nboot, 6),
                        absdev=paired_boot(D, "ev", lambda v: float(np.median(np.abs(v))), mainarms, nboot, 7),
                        all_bias={a: float(np.median(D[a])) for a in cols},
                        all_absdev={a: float(np.median(np.abs(D[a]))) for a in cols},
                        ref_median=float(q.ref_depth.reindex(D.ev).median()))
    return out


# ------------------------------------------------------------------------------------------------------------ verdict
def judge(d, a_val, rel=REL, absmin=None):
    """d = (B-A, lo, hi): 'B' (B lower = better), 'A', or '-' (no material difference)."""
    v, lo, hi = d
    thr = rel * abs(a_val) if absmin is None else absmin
    if lo > 0 and v >= thr:
        return "A"
    if hi < 0 and -v >= thr:
        return "B"
    return "-"


def verdict(res):
    rows = []
    for s, r in res.items():
        h = r["heldout"]["common"]
        ph_out = {}
        for ph in ("P", "S"):
            if ph in h:
                m = h[ph]["mad"]
                ph_out[ph] = judge(m["B-A"], m["A"][0])
                rows.append(dict(set=s, criterion=f"1 held-out MAD {ph}", A=m["A"][0], B=m["B"][0], diff=m["B-A"][0],
                                 lo=m["B-A"][1], hi=m["B-A"][2], better=ph_out[ph]))
        vals = set(ph_out.values())
        c1 = "B" if "B" in vals and "A" not in vals else "A" if "A" in vals and "B" not in vals else "-"
        rows.append(dict(set=s, criterion="1 held-out (cluster)", better=c1))
        m = r["ccfit"]["mad"]
        rows.append(dict(set=s, criterion="2 CC fit (median per-pair residual)", A=m["A"][0], B=m["B"][0],
                         diff=m["B-A"][0], lo=m["B-A"][1], hi=m["B-A"][2], better=judge(m["B-A"], m["A"][0])))
        rp = r["ccfit"].get("repeaters", {})
        if rp.get("n"):
            m = rp["sep"]
            rows.append(dict(set=s, criterion="2 repeater separation (km)", A=m["A"][0], B=m["B"][0],
                             diff=m["B-A"][0], lo=m["B-A"][1], hi=m["B-A"][2], better=judge(m["B-A"], m["A"][0])))
        for lab in ("A-D", "A-B"):
            dd = r["depth"].get(lab, {})
            if dd.get("n"):
                m = dd["absdev"]
                rows.append(dict(set=s, criterion=f"3 depth |dz| {lab} (km)", A=m["A"][0], B=m["B"][0],
                                 diff=m["B-A"][0], lo=m["B-A"][1], hi=m["B-A"][2],
                                 better=judge(m["B-A"], m["A"][0], absmin=DZ_MAT)))
        a_s = r["ccfit"]["arms"]
        rows.append(dict(set=s, criterion="3 above surface (A / B)", A=a_s["A"]["above_surface"],
                         B=a_s["B"]["above_surface"], better="-" if a_s["A"]["above_surface"] == a_s["B"]["above_surface"]
                         else ("A" if a_s["B"]["above_surface"] > a_s["A"]["above_surface"] else "B")))
    V = pd.DataFrame(rows)
    c1 = V[V.criterion == "1 held-out (cluster)"].set_index("set").better
    nB, nA = int((c1 == "B").sum()), int((c1 == "A").sum())
    above_B = int(V[V.criterion == "3 above surface (A / B)"].B.sum())
    above_A = int(V[V.criterion == "3 above surface (A / B)"].A.sum())
    # veto: the winner of criterion 1 loses materially on both criterion 2 and criterion 3 in >= 2 clusters
    def loses_both(w):
        other = "A" if w == "B" else "B"
        n = 0
        for s in V.set.unique():
            v = V[V.set == s]
            c2 = (v[v.criterion.str.startswith("2")].better == other).any()
            c3 = (v[v.criterion.str.startswith("3 depth")].better == other).any()
            n += int(c2 and c3)
        return n >= 2
    if nB >= 2 and nA == 0 and above_B == 0 and not loses_both("B"):
        rec = "B"
    elif nA >= 2 and nB == 0 and above_A == 0 and not loses_both("A"):
        rec = "A"
    else:
        rec = "no material difference (prefer A, the simpler recipe)"
    return V, dict(heldout_wins_B=nB, heldout_wins_A=nA, above_A=above_A, above_B=above_B,
                   veto_B=loses_both("B"), veto_A=loses_both("A"), recommendation=rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default=",".join(T.SETS))
    ap.add_argument("--nboot", type=int, default=2000)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for s in a.sets.split(","):
        t0 = time.time()
        res[s] = dict(heldout=heldout(s, a.nboot), ccfit=ccfit(s, a.nboot), depth=depth(s, a.nboot))
        log(f"{s} scored, {time.time() - t0:.0f} s")
    (OUT / "scores.json").write_text(json.dumps(res, indent=1, default=float))
    V, rec = verdict(res)
    V.to_csv(OUT / "verdict.csv", index=False, float_format="%.4g")
    (OUT / "recommendation.json").write_text(json.dumps(rec, indent=1))
    print(V.to_string())
    print(rec)


if __name__ == "__main__":
    main()

"""Track B, catalog v6.1 (D-B100): scores and the pre-registered decision of the station-term depth test.

Implements the rule written in /wd1/mdenolle_data/reloc_v61/depth_test/RESUME.md (2026-11-26 13:09 PST, before any
run or score). Inputs: the runs of b4_v61_depth_test.py. Outputs (score/):
* heldout_rows_<set>.parquet  held-out CC rows with one score column per arm (rows located in every arm, 3 folds)
* heldout.csv                 MAD and rms per set, phase and arm; MAD(arm) - MAD(best) with its 95 % interval
* depth.csv                   event-paired offsets (median, p10, p90, n) per literature cluster and arm; Axial shares
* surface.csv                 events above the solid surface and floor-limited shares per set, run and arm
* decision.json               the rule applied step by step, and the winner

  python b4_v61_depth_test.py score [--nboot 2000]
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "08_relocation"))
import b4_v61_depth_test as T  # noqa: E402
import b4_v6_heldout as H  # noqa: E402
from b4_catalog import floor_at, read_floor, surface_km, unflatten  # noqa: E402
from b4_v6_ssst_metrics import read_arm  # noqa: E402

warnings.simplefilter("ignore", FutureWarning)
SC = T.OUT / "score"
CAT = Path("/wd1/mdenolle_data/reloc_v6/catalog/catalog_b4_v6.0.csv")
PAIRS = Path("/wd1/mdenolle_data/tomo_v1/depth_literature/pairs")
FLOOR = Path("/wd1/mdenolle_data/reloc_v5/topo/topo_15s_cons_merged_eft.bin")
LIT = {   # set: (reference, pair table, longitude range of the published epicentre or None)
    "msh": ("Hirao et al. 2025", "msh_hirao2025_pairs.csv", None),
    "blancow": ("Lange et al. 2026 (absolute)", "blanco_lange2026_abs_pairs.csv", None),
    "gordad": ("Lange et al. 2026 (absolute)", "blanco_lange2026_abs_pairs.csv", (-127.1, -126.0)),
}
LIT_REL = {"blancow": "blanco_lange2026_rel_pairs.csv", "gordad": "blanco_lange2026_rel_pairs.csv"}
WALDHAUSER = ((0.85, 1.6), (0.90, 2.0))       # share within depth below the seafloor (Waldhauser et al. 2020)
MIN_LIT_EVENTS = 100
FIT_TOL, TIE_KM = 0.05, 0.05


def log(*a):
    print(time.strftime("%F %T"), *a, flush=True)


_CACHE = {}


def read_pass(rd):
    """LOCATED events of an NLLoc run folder (per-event files), true depth; cached."""
    if rd not in _CACHE:
        ev, _ = read_arm(rd)
        ev = ev[ev.status == "LOCATED"].drop_duplicates("ID").set_index("ID") if len(ev) else ev
        if len(ev):
            ev["depth_zf"] = ev.depth
            ev["depth"] = unflatten(ev.depth_zf)
        _CACHE[rd] = ev
    return _CACHE[rd]


def positions(s, run, arm):
    """(positions lat, lon, depth [true], depth_zf, ns_z; relocated-by-coherence IDs) of an arm in a run."""
    d = H.OUT / s / run
    pas, coh = {**T.ARMS, **T.EXTRA}[arm]
    e = read_pass(d / pas)[["lat", "lon", "depth", "depth_zf", "ns_z"]].copy()
    rel = set()
    if coh:
        c = read_pass(d / coh)
        if len(c):
            idx = c.index.intersection(e.index)
            e.loc[idx, ["lat", "lon", "depth", "depth_zf", "ns_z"]] = c.loc[idx, ["lat", "lon", "depth", "depth_zf",
                                                                                     "ns_z"]].to_numpy()
            rel = set(idx)
    return e, rel


def complete(s, run, arm):
    d = H.OUT / s / run
    pas, coh = {**T.ARMS, **T.EXTRA}[arm]
    return (d / pas / "run.json").exists() and (coh is None or (d / coh / "run.json").exists())


# ------------------------------------------------------------------------------------------------------- held out
def heldout_rows(s, arms):
    import b5_cmp_common as CM
    import b5_relocate as BR
    CM.CC = Path(H.SETS[s][3]).parent
    CM._G["G"] = BR.Grids(H.OUT / s / "grids_base")
    cc = CM.cc_rows(Path(H.SETS[s][3]).stem.replace("cc_", ""))
    parts, nrel = [], {a: [] for a in arms}
    for run in T.RUNS[1:]:
        ho = H.held_out(s, run)
        cols = []
        for arm in arms:
            pos, rel = positions(s, run, arm)
            nrel[arm].append(len(rel))
            cols.append(CM.fold_score(cc, pos[["lat", "lon", "depth"]], ho)
                        .set_index(["ID1", "ID2", "station", "phase"]).score.rename(arm))
        j = pd.concat(cols, axis=1, join="inner").reset_index()
        j["run"] = run
        parts.append(j)
    C = pd.concat(parts, ignore_index=True)
    C["pair"] = C.ID1.astype(str) + "_" + C.ID2.astype(str)
    return C, nrel


def mad(v):
    return float(np.median(np.abs(v - np.median(v))))


def boot(g, arms, nboot, seed):
    """MAD of every arm on the same event-pair resamples: array (nboot, arms)."""
    codes, uniq = pd.factorize(g.pair)
    rng = np.random.default_rng(seed)
    V = g[arms].to_numpy()
    out = np.empty((nboot, len(arms)))
    for b in range(nboot):
        cnt = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        idx = np.repeat(np.arange(len(g)), cnt[codes])
        out[b] = [mad(V[idx, k]) for k in range(len(arms))]
    return out, len(uniq)


# ---------------------------------------------------------------------------------------------------------- depth
def lit_pairs(s, table, lon):
    p = pd.read_csv(PAIRS / table)
    if lon is not None:
        p = p[(p.ref_lon >= lon[0]) & (p.ref_lon < lon[1])]
    return p


def q(x):
    x = np.asarray(x, float)
    return dict(n=int(len(x)), med=float(np.median(x)), p10=float(np.percentile(x, 10)),
                p90=float(np.percentile(x, 90))) if len(x) else dict(n=0, med=np.nan, p10=np.nan, p90=np.nan)


def main(a):
    global SC
    sets = a.sets.split(",")
    if a.out:
        SC = Path(a.out)
    SC.mkdir(parents=True, exist_ok=True)
    arms = list(T.ARMS)
    status = {s: {arm: all(complete(s, r, arm) for r in T.RUNS) for arm in arms} for s in sets}
    eligible_complete = [arm for arm in arms if all(status[s][arm] for s in sets)]
    log("complete arms:", eligible_complete)
    cat = pd.read_csv(CAT, usecols=["ID", "tier"], low_memory=False).set_index("ID").tier
    fl = read_floor(FLOOR)

    # 1. surface / floor
    srows = []
    for s in sets:
        for run in T.RUNS:
            for arm in arms:
                if not complete(s, run, arm):
                    continue
                e, rel = positions(s, run, arm)
                sk = surface_km(e.lat.values, e.lon.values)
                hf = e.depth_zf.to_numpy(float) + floor_at(fl, e.lat.values, e.lon.values) / 1e3
                srows.append(dict(set=s, run=run, arm=arm, located=len(e), relocated_by_coherence=len(rel),
                                  above_surface=int((e.depth.to_numpy() + sk < 0).sum()),
                                  floor_limited_pct=100 * float(np.mean(hf < 1.5 * e.ns_z.to_numpy()))))
    Sf = pd.DataFrame(srows)
    Sf.to_csv(SC / "surface.csv", index=False)

    # 2. held-out relative fit
    hrows, best = [], {}
    for s in sets:
        use = [arm for arm in arms if status[s][arm]]
        C, nrel = heldout_rows(s, use)
        C.to_parquet(SC / f"heldout_rows_{s}.parquet", index=False)
        for ph, g in C.groupby("phase"):
            g = g.reset_index(drop=True)
            m = {arm: mad(g[arm].to_numpy()) for arm in use}
            b = min(m, key=m.get)
            best[(s, ph)] = b
            B, npair = boot(g, use, a.nboot, 100 + sets.index(s) * 2 + (ph == "S"))
            kb = use.index(b)
            for k, arm in enumerate(use):
                dd = B[:, k] - B[:, kb]
                lo, hi = np.percentile(dd, [2.5, 97.5])
                hrows.append(dict(set=s, phase=ph, arm=arm, rows=len(g), pairs=npair, mad_ms=m[arm] * 1e3,
                                  rms_ms=float(np.sqrt(np.mean(g[arm] ** 2))) * 1e3, best=b,
                                  diff_best_ms=(m[arm] - m[b]) * 1e3, lo_ms=lo * 1e3, hi_ms=hi * 1e3,
                                  tol_ms=FIT_TOL * m[b] * 1e3, out=bool(lo > FIT_TOL * m[b]),
                                  relocated_by_coherence_per_fold=json.dumps(nrel[arm])))
            log(s, ph, "best", b, {k: round(v * 1e3, 1) for k, v in m.items()})
    Hd = pd.DataFrame(hrows)
    Hd.round(3).to_csv(SC / "heldout.csv", index=False)

    # 3. depth against the literature (full runs)
    drows = []
    pos_full = {s: {arm: positions(s, "full", arm)[0] for arm in arms if complete(s, "full", arm)} for s in sets}
    for s in sets:
        P = pos_full[s]
        common = sorted(set.intersection(*[set(p.index) for p in P.values()]))
        if s == "axial":
            ids = [i for i in common if cat.get(i) in ("A", "B")]
            for arm, p in P.items():
                e = p.loc[ids]
                bsf = e.depth.to_numpy() + surface_km(e.lat.values, e.lon.values)
                off = float(np.mean([np.quantile(bsf, qq) - z for qq, z in WALDHAUSER]))
                drows.append(dict(set=s, reference="Waldhauser et al. 2020 (seafloor quantiles)", arm=arm, n=len(e),
                                  offset_km=off, within_1_6_pct=100 * float(np.mean(bsf <= 1.6)),
                                  within_2_0_pct=100 * float(np.mean(bsf <= 2.0)), q85=float(np.quantile(bsf, .85)),
                                  q90=float(np.quantile(bsf, .90)), median_bsf=float(np.median(bsf))))
            continue
        ref, table, lon = LIT[s]
        pr = lit_pairs(s, table, lon).set_index("ID")
        ids = [i for i in common if i in pr.index and cat.get(i) in ("A", "B", "C")]
        rel = lit_pairs(s, LIT_REL[s], lon).set_index("ID") if s in LIT_REL else None
        for arm, p in P.items():
            dz = p.loc[ids].depth.to_numpy() - pr.loc[ids].ref_depth.to_numpy()
            r = q(dz)
            row = dict(set=s, reference=ref, arm=arm, n=r["n"], offset_km=r["med"], p10_km=r["p10"],
                       p90_km=r["p90"], ours_median_km=float(np.median(p.loc[ids].depth)),
                       ref_median_km=float(np.median(pr.loc[ids].ref_depth)))
            if rel is not None:
                ir = [i for i in common if i in rel.index and cat.get(i) in ("A", "B", "C")]
                rr = q(p.loc[ir].depth.to_numpy() - rel.loc[ir].ref_depth.to_numpy())
                row.update(rel_n=rr["n"], rel_offset_km=rr["med"], rel_p10_km=rr["p10"], rel_p90_km=rr["p90"])
            drows.append(row)
    Dp = pd.DataFrame(drows)
    Dp.round(4).to_csv(SC / "depth.csv", index=False)

    # 4. the rule
    lit_sets = [s for s in sets if s == "axial" or
                Dp[(Dp.set == s)].n.min() >= MIN_LIT_EVENTS]
    out_a = sorted(set(Sf[Sf.above_surface > 0].arm))
    out_b = {}
    for r in Hd[Hd.out].itertuples():
        out_b.setdefault(r.arm, []).append(f"{r.set} {r.phase}: {r.diff_best_ms:+.1f} ms [{r.lo_ms:+.1f}, "
                                           f"{r.hi_ms:+.1f}] vs {r.best}, tolerance {r.tol_ms:.1f}")
    elig = [arm for arm in eligible_complete if arm not in out_a and arm not in out_b]
    D = {arm: float(np.mean([abs(Dp[(Dp.set == s) & (Dp.arm == arm)].offset_km.iloc[0]) for s in lit_sets]))
         for arm in arms if all(((Dp.set == s) & (Dp.arm == arm)).any() for s in lit_sets)}
    winner, tied = None, []
    if elig:
        dmin = min(D[arm] for arm in elig)
        tied = [arm for arm in elig if D[arm] <= dmin + TIE_KM]
        winner = min(tied, key=T.ORDER.index)
    dec = dict(rule_written="2026-11-26 13:09 PST (RESUME.md)", scored=time.strftime("%F %T %Z"), nboot=a.nboot,
               arms=arms, incomplete=[arm for arm in arms if arm not in eligible_complete],
               out_above_surface=out_a, out_relative_fit=out_b, eligible=elig, literature_sets=lit_sets,
               D_km=D, D_eligible={arm: D.get(arm) for arm in elig}, tied=tied, winner=winner,
               simplicity_order=T.ORDER, best_by_set_phase={f"{k[0]} {k[1]}": v for k, v in best.items()})
    (SC / "decision.json").write_text(json.dumps(dec, indent=1, default=float))
    post_hoc(a, sets, arms, best, cat, lit_sets, Dp)
    log("decision:", json.dumps({k: dec[k] for k in ("out_above_surface", "eligible", "D_eligible", "tied", "winner")},
                                default=float))


def post_hoc(a, sets, arms, best, cat, lit_sets, Dp):
    """the post-hoc arms (T.EXTRA, not part of the rule), scored the same way but on their own row sets (rows located
    in every registered arm AND the extra arm), against the registered best arm of each cluster and phase; depth on
    the registered events also located by the extra arm. -> extra_heldout.csv, extra_depth.csv, extra.json"""
    extra = [x for x in T.EXTRA if all(complete(s, r, x) for s in sets for r in T.RUNS)]
    if not extra:
        return
    rows = []
    for s in sets:
        C, _ = heldout_rows(s, arms + extra)
        for ph, g in C.groupby("phase"):
            g = g.reset_index(drop=True)
            b = best[(s, ph)]
            use = [b] + extra
            B, npair = boot(g, use, a.nboot, 500 + sets.index(s) * 2 + (ph == "S"))
            mb = mad(g[b].to_numpy())
            for k, x in enumerate(extra, start=1):
                dd = B[:, k] - B[:, 0]
                lo, hi = np.percentile(dd, [2.5, 97.5])
                rows.append(dict(set=s, phase=ph, arm=x, rows=len(g), pairs=npair, best=b, mad_best_ms=mb * 1e3,
                                 mad_ms=mad(g[x].to_numpy()) * 1e3, diff_best_ms=(mad(g[x].to_numpy()) - mb) * 1e3,
                                 lo_ms=lo * 1e3, hi_ms=hi * 1e3, tol_ms=FIT_TOL * mb * 1e3,
                                 would_be_out=bool(lo > FIT_TOL * mb)))
    He = pd.DataFrame(rows)
    He.round(3).to_csv(SC / "extra_heldout.csv", index=False)
    drows = []
    for s in sets:
        P = {arm: positions(s, "full", arm)[0] for arm in arms + extra}
        common = sorted(set.intersection(*[set(p.index) for p in P.values()]))
        for x in extra:
            p = P[x]
            if s == "axial":
                ids = [i for i in common if cat.get(i) in ("A", "B")]
                e = p.loc[ids]
                bsf = e.depth.to_numpy() + surface_km(e.lat.values, e.lon.values)
                drows.append(dict(set=s, arm=x, n=len(e), offset_km=float(np.mean([np.quantile(bsf, qq) - z
                                                                                   for qq, z in WALDHAUSER])),
                                  within_1_6_pct=100 * float(np.mean(bsf <= 1.6)),
                                  within_2_0_pct=100 * float(np.mean(bsf <= 2.0))))
                continue
            ref, table, lon = LIT[s]
            pr = lit_pairs(s, table, lon).set_index("ID")
            ids = [i for i in common if i in pr.index and cat.get(i) in ("A", "B", "C")]
            r = q(p.loc[ids].depth.to_numpy() - pr.loc[ids].ref_depth.to_numpy())
            drows.append(dict(set=s, arm=x, n=r["n"], offset_km=r["med"], p10_km=r["p10"], p90_km=r["p90"]))
    De = pd.DataFrame(drows)
    De.round(4).to_csv(SC / "extra_depth.csv", index=False)
    D = {x: float(np.mean([abs(De[(De.set == s) & (De.arm == x)].offset_km.iloc[0]) for s in lit_sets])) for x in extra}
    (SC / "extra.json").write_text(json.dumps(dict(extra=extra, D_km=D, would_be_out={
        x: He[(He.arm == x) & He.would_be_out][["set", "phase"]].astype(str).agg(" ".join, axis=1).tolist()
        for x in extra}, note="post-hoc arms: reported only, not part of the pre-registered decision"),
        indent=1, default=float))
    log("post-hoc:", json.dumps(D, default=float))

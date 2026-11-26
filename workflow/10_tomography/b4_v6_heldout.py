"""Track B, catalog v6.0 QC (D-B096): held-out effect of the NLL-coherence stage by region, with the v6.0 recipe.

D-B096: the coherence stage helped where it relocates many events (Puget, Georgia), was neutral where it relocates few
(Gorda, western Blanco) and hurt at the Mendocino 2015 sequence; v6.0's QC reports its held-out effect by region. Here the
v6.0 recipe (model v0.4.1 grids, NLL-SSST 9999 -> 16 -> 8 -> 4 km, then coherence) is rerun on the nine test clusters
that have waveform CC data (D-B078 sets axial, blanco, msh, jdf; termtest sets puget, mtj, blancow, georgia, gorda) with
the termtest protocol (b4_v6_termtest.py, session -18's held-out-station scorer b5_cmp_common.fold_score): the stations
with picks in a set are dealt into 3 folds (numpy default_rng(78 + set index), the indices of -18 and of the termtest);
each fold reruns the whole chain without the picks of its stations and without CC rows at them, and the held-out CC
differential times are predicted from each arm's positions in the grids NLLoc used (fine 1 km, or wide 2 km where the
fine grid does not hold the box), minus the pair's median training residual. Arms: ssst (pass it4: SSST stopped at 4 km)
and coherence (the coherence stage on it4; events it does not relocate keep their it4 position). Pair coherence: median
cc_max of the training-station rows (status ok, peak not at the lag edge, not straddling a confirmed clock epoch),
>= 3 rows. Scores on the rows whose two events are located in both arms of the fold, pooled over the folds: MAD and rms
per phase, coherence - ssst with a 95 % paired bootstrap interval over event pairs (2,000 resamples).

Each cluster box is the set's v5.1 epicentres +- 30 km (b4_v6_ssst.MARGIN), as in the tests (the catalog uses 25 km tiles
+- 20 km); NLLoc, Loc2ssst and the coherence stage are b4_v6_ssst.nll_run / loc2ssst / the termtest coherence_run.

  python b4_v6_heldout.py run [--sets axial,puget] [--nproc 8]     # relocation (resumable; one set-fold at a time)
  python b4_v6_heldout.py score                                      # -> reloc_v6/heldout/scores.json, heldout.csv
Output: /wd1/mdenolle_data/reloc_v6/heldout/<set>/{grids_base, fold<k>/{obs, picks.csv, ssst/it<k>, ssst/g<k>, coherence}}
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TA = HERE.parent / "08_relocation"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TA))
import a2_run_nlloc as RN  # noqa: E402
import b4_v6_ssst as S  # noqa: E402

R6 = Path("/wd1/mdenolle_data/reloc_v6")
OUT = Path(os.environ.get("V6_HELDOUT_OUT", str(R6 / "heldout")))
GR = dict(fine=Path(os.environ.get("V6_FINE", str(R6 / "time_v6_v0.4.1"))),
          wide=Path(os.environ.get("V6_WIDE", str(R6 / "wide_time_v6_v0.4.1"))))
SCHEDULE = (9999.0, 16.0, 8.0, 4.0)        # D-B096
FOLDS, SEED = 3, 78
TT = R6 / "termtest"
SETS = {   # set: (events (v5.1 positions), picks, v4 events, CC parquet, seed index, region)
    "axial": ("/wd1/mdenolle_data/tomo_v1/b5/cc_test/events_axial.csv", R6 / "ssst_test/inputs/picks_axial.csv",
              R6 / "ssst_test/inputs/events_axial.csv", "/wd1/mdenolle_data/tomo_v1/cc_dt/cc_axial.parquet", 0, "Axial"),
    "blanco": ("/wd1/mdenolle_data/tomo_v1/b5/cc_test/events_blanco.csv", R6 / "ssst_test/inputs/picks_blanco.csv",
               R6 / "ssst_test/inputs/events_blanco.csv", "/wd1/mdenolle_data/tomo_v1/cc_dt/cc_blanco.parquet", 1, "Blanco"),
    "msh": ("/wd1/mdenolle_data/tomo_v1/b5/cc_test/events_msh.csv", R6 / "ssst_test/inputs/picks_msh.csv",
            R6 / "ssst_test/inputs/events_msh.csv", "/wd1/mdenolle_data/tomo_v1/cc_dt/cc_msh.parquet", 2, "Cascades (Mount St Helens)"),
    "jdf": ("/wd1/mdenolle_data/tomo_v1/b5/cc_test/events_jdf.csv", R6 / "ssst_test/inputs/picks_jdf.csv",
            R6 / "ssst_test/inputs/events_jdf.csv", "/wd1/mdenolle_data/tomo_v1/cc_dt/cc_jdf.parquet", 3, "JdF ridge/plate"),
    "puget": (TT / "sets/events_puget.csv", TT / "inputs/picks_puget.csv", TT / "inputs/events_puget.csv",
              TT / "cc/cc_puget.parquet", 4, "Puget/Georgia"),
    "mtj": (TT / "sets/events_mtj.csv", TT / "inputs/picks_mtj.csv", TT / "inputs/events_mtj.csv",
            TT / "cc/cc_mtj.parquet", 5, "Mendocino"),
    "blancow": (TT / "sets/events_blancow.csv", TT / "inputs/picks_blancow.csv", TT / "inputs/events_blancow.csv",
                TT / "cc/cc_blancow.parquet", 6, "Blanco"),
    "georgia": (TT / "sets/events_georgia.csv", TT / "inputs/picks_georgia.csv", TT / "inputs/events_georgia.csv",
                TT / "cc/cc_georgia.parquet", 7, "Puget/Georgia"),
    "gorda": (TT / "sets/events_gorda.csv", TT / "inputs/picks_gorda.csv", TT / "inputs/events_gorda.csv",
              TT / "cc/cc_gorda.parquet", 8, "Gorda"),
}
RUNS = [f"fold{k}" for k in range(FOLDS)]
PY = S.PY


def log(*a):
    print(time.strftime("%F %T"), *a, flush=True)


def folds(s):
    sta = np.array(sorted(pd.read_csv(SETS[s][1], usecols=["station"]).station.unique()))
    order = np.random.default_rng(SEED + SETS[s][4]).permutation(len(sta))
    return {sta[i]: k % FOLDS for k, i in enumerate(order)}


def held_out(s, run):
    k = int(run.replace("fold", ""))
    return {st for st, f in folds(s).items() if f == k}


def set_box(s):
    ev = pd.read_csv(SETS[s][0])
    x, y = S.lambert()(ev.lon.values, ev.lat.values)
    x0, x1 = np.floor(x.min() / 1e3 - S.MARGIN), np.ceil(x.max() / 1e3 + S.MARGIN)
    y0, y1 = np.floor(y.min() / 1e3 - S.MARGIN), np.ceil(y.max() / 1e3 + S.MARGIN)
    x0, y0, x1, y1 = max(x0, -513.0), max(y0, -709.0), min(x1, 457.0), min(y1, 724.0)
    return (float(x0), float(y0), int(x1 - x0) + 1, int(y1 - y0) + 1)


def blocks_of(obs):
    out = []
    for f in sorted(obs.glob("*.obs")):
        out += [b.strip("\n") for b in f.read_text().split("\n\n") if b.strip()]
    return out


def prep(s, run):
    d = OUT / s / run
    obs = d / "obs"
    if not (obs / "obs_meta.json").exists():
        assert not obs.exists() or not any(obs.iterdir()), f"{obs} incomplete: remove it"
        d.mkdir(parents=True, exist_ok=True)
        p = pd.read_csv(SETS[s][1])
        ho = held_out(s, run)
        q = p[~p.station.isin(ho)]
        q.to_csv(d / "picks.csv", index=False)
        (d / "held_out.json").write_text(json.dumps({"held_out": sorted(ho), "picks_dropped": int(len(p) - len(q))}))
        env = dict(os.environ, LD_LIBRARY_PATH=str(PY.parents[1] / "lib"))
        subprocess.run([str(PY), str(TA / "a2_make_obs.py"), "--picks", str(d / "picks.csv"), "--events",
                        str(SETS[s][2]), "--out", str(obs), "--time", str(GR["fine"])], check=True, env=env, cwd=TA,
                       stdout=subprocess.DEVNULL)
    return blocks_of(obs), set_box(s)


def base_grids(s, box):
    """links per station-phase of the set's picks: fine grid if it holds the box, else wide (NLLoc's header rule)."""
    d = OUT / s / "grids_base"
    if (d / "base.json").exists():
        return d
    p = pd.read_csv(SETS[s][1], usecols=["station", "phase"]).drop_duplicates()
    roots = (f"{GR['fine']}/cascadia3d", f"{GR['wide']}/cascadia3d")
    ext = S.box_ext(box)
    used = {}
    d.mkdir(parents=True, exist_ok=True)
    for sta, ph in sorted(set(zip(p.station, p.phase))):
        i = RN.serving_root(ext, (ph, sta), roots)
        if i is None:
            continue          # station without a grid: a2_make_obs drops its picks as well
        for e in ("hdr", "buf"):
            ln = d / f"cascadia3d.{ph}.{sta}.time.{e}"
            if not ln.is_symlink():
                ln.symlink_to(Path(roots[i]).parent / f"cascadia3d.{ph}.{sta}.time.{e}")
        used[f"{ph}.{sta}"] = ("fine", "wide")[i]
    (d / "base.json").write_text(json.dumps(used, indent=0))
    return d


def cc_pairs(s, run):
    c = pd.read_parquet(SETS[s][3], columns=["ID1", "ID2", "station", "status", "edge", "cc_max",
                                             "straddles_clock_epoch"])
    c = c[(c.status == "ok") & ~c.edge.astype(bool) & ~c.straddles_clock_epoch.astype(bool)
          & ~c.station.isin(held_out(s, run))]
    g = c.groupby(["ID1", "ID2"]).cc_max.agg(["median", "size"]).reset_index()
    g.columns = ["ID1", "ID2", "coh", "n"]
    return g[g.n >= S.COH_NMIN]


def coherence_run(s, run, it_dir, groot, box, nproc):
    """b4_v6_ssst.coherence_run (termtest copy) with the fold's training-station CC rows."""
    import argparse as _ap
    from concurrent.futures import ProcessPoolExecutor
    import a2_parse_hyp as PH
    rd = OUT / s / run / "coherence"
    if (rd / "run.json").exists():
        return rd
    roots, dup = S.hyp_roots(it_dir)
    len3 = {eid: PH.parse(f"{r}.hyp")[0].get("ell_len3", np.inf) for eid, r in roots.items()}
    compact = {eid for eid, v in len3.items() if v <= S.COH_MAX_SE3}
    coh = cc_pairs(s, run)
    coh = coh[(coh.coh >= S.COH_MIN) & coh.ID1.isin(roots) & coh.ID2.isin(roots)]
    nb = {}
    for r in coh.itertuples():
        nb.setdefault(r.ID1, []).append((r.coh, r.ID2))
        nb.setdefault(r.ID2, []).append((r.coh, r.ID1))
    rd.mkdir(parents=True, exist_ok=True)
    opt = _ap.Namespace(time=str(groot), roots=(str(groot),), template_text=(TA / "nlloc_template.in").read_text(),
                        locgau=S.LOCGAU, locgau2=S.LOCGAU2, topo=S.FLOOR, max_dist=RN.MAX_DIST)
    jobs = []
    for eid, root in sorted(roots.items()):
        if eid not in compact:
            continue
        wd = rd / f"e{eid}"
        wd.mkdir(exist_ok=True)
        lines = ["STREAM_COHERENCES", f"  1.000000 {root}"] + [f"  {c:.6f} {roots[o]}" for c, o in sorted(nb.get(eid, []), reverse=True)]
        sc = wd / "event.stream_coherences"
        sc.write_text("\n".join(lines) + "\n")
        ctl = RN.control(box, f"{root}.hyp", wd / "loc", None, opt)
        ctl = RN.sub1(r"^LOCFILES .*$", f"LOCFILES {root}.hyp NLLOC_OBS {groot} {wd}/loc", ctl)
        ctl += f"LOCPOSTERIOR OCT_TREE {sc} 1e-30 {S.COH_MIN} -1.0 {S.COH_MAX_SE3} -1 -1.0 -999\n"
        jobs.append((wd, ctl))
    with ProcessPoolExecutor(nproc) as ex:
        bad = [n for n, rc in ex.map(RN.run_one, jobs, chunksize=8) if rc != 0]
    with open(rd / "all.sum.grid0.loc.hyp", "w") as f:
        for wd, _ in jobs:
            p = wd / "loc.sum.grid0.loc.hyp"
            if p.exists():
                f.write(p.read_text())
    (rd / "run.json").write_text(json.dumps({"events": len(jobs), "failed": bad, "compact": len(compact),
                                             "events_with_partners": len(nb), "pairs_used": len(coh),
                                             "held_out": sorted(held_out(s, run)), "grids": str(groot)}, indent=1))
    log(f"coherence {rd}: {len(jobs)} events, {len(nb)} with partners, {len(bad)} failed")
    return rd


def nproc_now(default):
    f = R6 / "control" / "nproc_heldout"
    while S_mem() < 35.0:
        time.sleep(30)
    try:
        return max(1, int(f.read_text().split()[0]))
    except (OSError, ValueError, IndexError):
        return default


def S_mem():
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576.0
    return 1e9


def run_set(s, nproc):
    box = set_box(s)
    base = base_grids(s, box)
    for run in RUNS:
        d = OUT / s / run
        if (d / "coherence" / "run.json").exists():
            continue
        blocks, _ = prep(s, run)
        sp = S.station_phases(blocks)
        log(f"{s}/{run}: {len(blocks)} events, {len(sp)} station-phases, {len(held_out(s, run))} stations held out")
        d_in = base
        for k, char in enumerate(SCHEDULE):
            it = S.nll_run(d / "ssst" / f"it{k}", blocks, box, (f"{d_in}/cascadia3d",), nproc=nproc_now(nproc))
            d_in = S.loc2ssst(it, d_in, d / "ssst" / f"g{k}", char, box, sp, nproc=nproc_now(nproc))
        last = S.nll_run(d / "ssst" / f"it{len(SCHEDULE)}", blocks, box, (f"{d_in}/cascadia3d",), octree=True,
                         nproc=nproc_now(nproc))
        coherence_run(s, run, last, f"{d_in}/cascadia3d", box, nproc_now(nproc))


def cmd_run(a):
    for s in a.sets.split(","):
        run_set(s, a.nproc)
        (OUT / s / "DONE").write_text(time.strftime("%FT%T%z"))
        log(f"{s} done")


# ------------------------------------------------------------------------------------------------------------- score
def mad(v):
    v = np.asarray(v, float)
    return float(np.median(np.abs(v - np.median(v)))) if len(v) else np.nan


def arm_positions(s, run):
    import b4_v6_ssst_metrics as M
    from b4_catalog import unflatten
    e4, _ = M.read_arm(OUT / s / run / "ssst" / f"it{len(SCHEDULE)}")
    ec, _ = M.read_arm(OUT / s / run / "coherence")
    e4 = e4[e4.status == "LOCATED"].set_index("ID")
    ec = ec[ec.status == "LOCATED"].set_index("ID") if len(ec) else ec
    pos4 = pd.DataFrame({"lat": e4.lat, "lon": e4.lon, "depth": unflatten(e4.depth)})
    posc = pos4.copy()
    if len(ec):
        idx = ec.index.intersection(posc.index)
        posc.loc[idx, "lat"], posc.loc[idx, "lon"] = ec.loc[idx, "lat"], ec.loc[idx, "lon"]
        posc.loc[idx, "depth"] = unflatten(ec.loc[idx, "depth"])
    return pos4, posc, (set(ec.index) if len(ec) else set())


def cmd_score(a):
    import b5_cmp_common as CM
    import b5_relocate as BR
    res, rows = {}, []
    sets = [s for s in a.sets.split(",") if (OUT / s / "DONE").exists()]
    for s in sets:
        CM.CC = Path(SETS[s][3]).parent
        CM._G["G"] = BR.Grids(OUT / s / "grids_base")
        cc = CM.cc_rows(Path(SETS[s][3]).stem.replace("cc_", ""))
        parts, nrel = [], []
        for run in RUNS:
            ho = held_out(s, run)
            p4, pc, rel = arm_positions(s, run)
            h4 = CM.fold_score(cc, p4, ho).set_index(["ID1", "ID2", "station", "phase"]).score.rename("ssst")
            hc = CM.fold_score(cc, pc, ho).set_index(["ID1", "ID2", "station", "phase"]).score.rename("coherence")
            j = pd.concat([h4, hc], axis=1, join="inner").reset_index()
            j["run"] = run
            j["rel_both"] = j.ID1.isin(rel) & j.ID2.isin(rel)
            parts.append(j)
            nrel.append(len(rel))
        C = pd.concat(parts, ignore_index=True)
        C["pair"] = C.ID1.astype(str) + "_" + C.ID2.astype(str)
        res[s] = {"region": SETS[s][5], "relocated_by_coherence_per_fold": nrel,
                  "events": int(len(pd.read_csv(SETS[s][0])))}
        for ph, g in C.groupby("phase"):
            g = g.reset_index(drop=True)
            codes, uniq = pd.factorize(g.pair)
            rng = np.random.default_rng(96)
            order = np.argsort(codes, kind="stable")
            starts = np.searchsorted(codes[order], np.arange(len(uniq)))
            ends = np.r_[starts[1:], len(codes)]
            a4, ac = g.ssst.to_numpy(), g.coherence.to_numpy()
            d = []
            for _ in range(a.nboot):
                pick = rng.integers(0, len(uniq), len(uniq))
                idx = np.concatenate([order[starts[k]:ends[k]] for k in pick])
                d.append(mad(ac[idx]) - mad(a4[idx]))
            lo, hi = np.percentile(d, [2.5, 97.5])
            r = dict(set=s, region=SETS[s][5], phase=ph, rows=len(g), pairs=len(uniq), mad_ssst_ms=mad(a4) * 1e3,
                     mad_coh_ms=mad(ac) * 1e3, diff_ms=(mad(ac) - mad(a4)) * 1e3, lo_ms=lo * 1e3, hi_ms=hi * 1e3,
                     rms_ssst_ms=float(np.sqrt(np.mean(a4 ** 2))) * 1e3, rms_coh_ms=float(np.sqrt(np.mean(ac ** 2))) * 1e3,
                     rows_both_relocated=int(g.rel_both.sum()))
            r["effect"] = ("helps" if hi < 0 else "hurts" if lo > 0 else "no clear effect")
            rows.append(r)
            res[s][ph] = r
        log(s, {ph: (round(res[s][ph]["mad_ssst_ms"], 1), round(res[s][ph]["mad_coh_ms"], 1), res[s][ph]["effect"])
                for ph in ("P", "S") if ph in res[s]})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(3).to_csv(OUT / "heldout.csv", index=False)
    (OUT / "scores.json").write_text(json.dumps(dict(sets=res, schedule=SCHEDULE, folds=FOLDS, seed=SEED,
                                                     nboot=a.nboot, grids={k: str(v) for k, v in GR.items()},
                                                     missing=[s for s in a.sets.split(",") if s not in sets],
                                                     utc=time.strftime("%FT%T%z")), indent=1, default=float))
    log(f"scores -> {OUT}/scores.json ({len(sets)} sets)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--sets", default=",".join(SETS))
    r.add_argument("--nproc", type=int, default=8)
    s = sub.add_parser("score")
    s.add_argument("--sets", default=",".join(SETS))
    s.add_argument("--nboot", type=int, default=2000)
    a = ap.parse_args()
    {"run": cmd_run, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()

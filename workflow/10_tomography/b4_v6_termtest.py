"""Track B, catalog v6.0: decisive test of the station-term recipe (PI, 2026-11-25 14:29).

Recipes (both followed by the NLL-coherence stage of D-B093 / b4_v6_ssst.py):
  A  static station terms (the v5.1 terms, b4_v51_iter0/station_terms.in) + coherence      (arm static_coherence)
  B  NLL-SSST, CharDist schedule 9999, 16, 8, 4, 2 km (stopped at 2 km, b4_v6_ssst.SCHEDULE) + coherence
                                                                                            (arm coherence)
References: static alone (arm static, = static_oct: the same NLLoc run saving the oct-trees; identical locations,
checked on the v6-prep axial run) and SSST-to-2-km alone (arm ssst_it5).

Everything is the v6-prep method test (b4_v6_ssst.py, branch reloc/v6-prep 2d8cf6ff) on three new clusters
(b4_v6_termtest_sets.py: puget, mtj, blancow): the D-B090 clock-corrected picks v4.1tc (md5 checked), a2_make_obs.py,
the v5.1 grids (g51: fine 1 km, wide where the fine grid does not hold the box), search box = set epicentres +- 30 km,
z' -5..80 km, NLLoc as v5.1 (LOCGAU 0.2 0.0, >= 5 phases, merged-surface floor), Loc2ssst with LSPHSTAT/WeightFloor
of b4_v6_ssst, the coherence stage of b4_v6_ssst.coherence_run (oct-tree stacks of the events with pair coherence
>= 0.5, coherence = median cc_max of the pair's measured CC rows, >= 3 rows; only events with Len3 <= 5 km are
relocated, the others keep their previous location). The functions nll_run and loc2ssst of b4_v6_ssst are used
unchanged; prep, base_grids and coherence_run are copies that take the run folder as an argument and, for the
held-out test, leave the held-out stations out of the obs (picks) and of the pair coherences (CC rows).

Held-out-station protocol (session -18's D-B078 frame, b5_cmp_common on tomo/b5 b0dffcdd): the stations with picks
in a set, sorted, permuted with numpy default_rng(78 + k) (k = 4, 5, 6 for puget, mtj, blancow; -18's sets used
0-3) and dealt into 3 folds. Run fold<k> relocates every arm from scratch without the picks and CC rows of fold k's
stations: NLLoc's starting positions never see the held-out stations (no leak); run full uses every station.

Output: /wd1/mdenolle_data/reloc_v6/termtest/runs/g51/<set>/<run>/{obs, picks.csv, static_oct, ssst/it<k>,
ssst/g<k>, static_coherence, coherence}, <set>/grids_base; inputs/ (picks and v4 events of the sets).
Processes per NLLoc pass: --nproc, or the number in termtest/control/nproc_<set> when that file exists (read
before every pass, so a running chain can be given more processes).

  python b4_v6_termtest.py picks
  python b4_v6_termtest.py prep
  python b4_v6_termtest.py run SET --runs full,fold0,fold1,fold2 --arms static,ssst[,static_coherence,coherence]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TA = HERE.parent / "08_relocation"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TA))
import a2_run_nlloc as RN  # noqa: E402
import b4_v6_ssst as S  # noqa: E402

TT = Path("/wd1/mdenolle_data/reloc_v6/termtest")
SETDIR = TT / "sets"
CC_DIR = TT / "cc"
INPUTS = TT / "inputs"
RUNS = TT / "runs"
SETS = ("puget", "mtj", "blancow")
G = "g51"
FOLDS, SEED = 3, 78
SEED_INDEX = {"puget": 4, "mtj": 5, "blancow": 6}
ARMS = ("static", "ssst", "static_coherence", "coherence")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def run_names():
    return ["full"] + [f"fold{k}" for k in range(FOLDS)]


def set_dir(s):
    return RUNS / G / s


def run_dir(s, run):
    return set_dir(s) / run


def nproc_for(s, default):
    f = TT / "control" / f"nproc_{s}"
    try:
        return max(1, int(f.read_text().split()[0]))
    except (OSError, ValueError, IndexError):
        return default


# ------------------------------------------------------------------------------------------------------------ inputs
def cmd_picks(a):
    INPUTS.mkdir(parents=True, exist_ok=True)
    h = md5(S.PICKS)
    assert h == S.PICKS_MD5, f"{S.PICKS} md5 {h}, expected {S.PICKS_MD5}"
    ids = {s: set(pd.read_csv(SETDIR / f"events_{s}.csv").ID) for s in SETS}
    allid = set().union(*ids.values())
    parts = [ch[ch.event_ID.isin(allid)] for ch in pd.read_csv(S.PICKS, skipinitialspace=True, chunksize=500_000)]
    p = pd.concat(parts)
    ev = pd.read_csv(S.EVENTS, skipinitialspace=True)
    out = {}
    for s in SETS:
        q = p[p.event_ID.isin(ids[s])]
        q.to_csv(INPUTS / f"picks_{s}.csv", index=False)
        e = ev[ev.ID.isin(ids[s])]
        assert len(e) == len(ids[s]), f"{s}: {len(ids[s]) - len(e)} events missing from {S.EVENTS}"
        e.to_csv(INPUTS / f"events_{s}.csv", index=False)
        out[s] = dict(events=len(e), picks=len(q), stations=int(q.station.nunique()))
        log(s, out[s])
    (INPUTS / "inputs.json").write_text(json.dumps({"picks": str(S.PICKS), "picks_md5": h, "events": str(S.EVENTS),
                                                    "events_md5": md5(S.EVENTS), "sets": str(SETDIR), "counts": out},
                                                   indent=1))


def folds(s):
    """{station: fold} of the stations with picks in the set (seeded as b5_cmp_common.folds)."""
    sta = np.array(sorted(pd.read_csv(INPUTS / f"picks_{s}.csv", usecols=["station"]).station.unique()))
    rng = np.random.default_rng(SEED + SEED_INDEX[s])
    order = rng.permutation(len(sta))
    return {sta[i]: k % FOLDS for k, i in enumerate(order)}


def held_out(s, run):
    if run == "full":
        return set()
    k = int(run.replace("fold", ""))
    return {st for st, f in folds(s).items() if f == k}


def set_box(s):
    """search box of the set (b4_v6_ssst.prep): v5.1 epicentres +- MARGIN km on the 1 km lattice."""
    ev = pd.read_csv(SETDIR / f"events_{s}.csv")
    x, y = S.lambert()(ev.lon.values, ev.lat.values)
    x0, x1 = np.floor(x.min() / 1e3 - S.MARGIN), np.ceil(x.max() / 1e3 + S.MARGIN)
    y0, y1 = np.floor(y.min() / 1e3 - S.MARGIN), np.ceil(y.max() / 1e3 + S.MARGIN)
    x0, y0 = max(x0, -513.0), max(y0, -709.0)
    x1, y1 = min(x1, 457.0), min(y1, 724.0)
    return (float(x0), float(y0), int(x1 - x0) + 1, int(y1 - y0) + 1)


def prep(s, run):
    """obs of a run (a2_make_obs.py on the set's picks without the held-out stations); returns (blocks, box)."""
    d = run_dir(s, run)
    obs = d / "obs"
    if not (obs / "obs_meta.json").exists():
        assert not obs.exists() or not any(obs.iterdir()), f"{obs} incomplete: remove it"
        d.mkdir(parents=True, exist_ok=True)
        p = pd.read_csv(INPUTS / f"picks_{s}.csv")
        ho = held_out(s, run)
        q = p[~p.station.isin(ho)]
        q.to_csv(d / "picks.csv", index=False)
        (d / "held_out.json").write_text(json.dumps({"held_out": sorted(ho), "picks_dropped": int(len(p) - len(q)),
                                                     "picks": int(len(q))}, indent=1))
        env = dict(os.environ, LD_LIBRARY_PATH=str(S.PY.parents[1] / "lib"))
        subprocess.run([str(S.PY), str(TA / "a2_make_obs.py"), "--picks", str(d / "picks.csv"),
                        "--events", str(INPUTS / f"events_{s}.csv"), "--out", str(obs),
                        "--time", str(S.GRIDS[G]["fine"])], check=True, env=env, cwd=TA)
    blocks = []
    for f in sorted(obs.glob("*.obs")):
        blocks += [b.strip("\n") for b in f.read_text().split("\n\n") if b.strip()]
    return blocks, set_box(s)


def base_grids(s, box):
    """grids_base of the set (b4_v6_ssst.base_grids): per station-phase of the full run's obs, a link to the fine
    grid if it holds the box, else to the wide grid. The fold runs use the same links (subset of station-phases)."""
    d = set_dir(s) / "grids_base"
    if (d / "base.json").exists():
        return d
    blocks, _ = prep(s, "full")
    roots = (f"{S.GRIDS[G]['fine']}/cascadia3d", f"{S.GRIDS[G]['wide']}/cascadia3d")
    ext = S.box_ext(box)
    used = {}
    d.mkdir(parents=True, exist_ok=True)
    for (ph, sta) in sorted(S.station_phases(blocks)):
        i = RN.serving_root(ext, (ph, sta), roots)
        assert i is not None, f"no grid contains the box for {ph}.{sta}"
        src = Path(roots[i])
        for e in ("hdr", "buf"):
            ln = d / f"cascadia3d.{ph}.{sta}.time.{e}"
            tgt = src.parent / f"cascadia3d.{ph}.{sta}.time.{e}"
            if not ln.exists():
                ln.symlink_to(tgt)
            assert os.path.realpath(ln) == str(tgt.resolve()), ln
        used[f"{ph}.{sta}"] = ("fine", "wide")[i]
    (d / "base.json").write_text(json.dumps(used, indent=1))
    return d


# --------------------------------------------------------------------------------------------------------- coherence
def coherences(s, run):
    """pair coherence (b4_v6_ssst.coherences) from the set's CC rows at the run's training stations only."""
    c = pd.read_parquet(CC_DIR / f"cc_{s}.parquet",
                        columns=["ID1", "ID2", "station", "status", "edge", "cc_max"])
    c = c[(c.status == "ok") & (~c.edge.astype(bool)) & ~c.station.isin(held_out(s, run))]
    g = c.groupby(["ID1", "ID2"]).cc_max.agg(["median", "size"]).reset_index()
    g.columns = ["ID1", "ID2", "coh", "n"]
    return g[g.n >= S.COH_NMIN]


def coherence_run(s, run, it_dir, roots_t, box, stat=None, name="coherence", nproc=4):
    """b4_v6_ssst.coherence_run with the run folder and the run's coherences (training stations only)."""
    rd = run_dir(s, run) / name
    if (rd / "run.json").exists():
        return rd
    import a2_parse_hyp as PH
    roots, dup = S.hyp_roots(it_dir)
    len3 = {eid: PH.parse(f"{r}.hyp")[0].get("ell_len3", np.inf) for eid, r in roots.items()}
    compact = {eid for eid, v in len3.items() if v <= S.COH_MAX_SE3}
    coh = coherences(s, run)
    coh = coh[(coh.coh >= S.COH_MIN) & coh.ID1.isin(roots) & coh.ID2.isin(roots)]
    nb = {}
    for r in coh.itertuples():
        nb.setdefault(r.ID1, []).append((r.coh, r.ID2))
        nb.setdefault(r.ID2, []).append((r.coh, r.ID1))
    rd.mkdir(parents=True, exist_ok=True)
    tmpl = (TA / "nlloc_template.in").read_text()
    opt = argparse.Namespace(time=str(roots_t[0]), roots=tuple(str(r) for r in roots_t), template_text=tmpl,
                             locgau=S.LOCGAU, locgau2=S.LOCGAU2, topo=S.FLOOR, max_dist=RN.MAX_DIST)
    jobs = []
    for eid, root in sorted(roots.items()):
        if eid not in compact:
            continue
        wd = rd / f"e{eid}"
        wd.mkdir(exist_ok=True)
        lines = ["STREAM_COHERENCES", f"  1.000000 {root}"]
        lines += [f"  {c:.6f} {roots[o]}" for c, o in sorted(nb.get(eid, []), reverse=True)]
        sc = wd / "event.stream_coherences"
        sc.write_text("\n".join(lines) + "\n")
        ctl = RN.control(box, f"{root}.hyp", wd / "loc", str(stat) if stat else None, opt)
        ctl = RN.sub1(r"^LOCFILES .*$", f"LOCFILES {root}.hyp NLLOC_OBS {','.join(opt.roots)} {wd}/loc", ctl)
        ctl += f"LOCPOSTERIOR OCT_TREE {sc} 1e-30 {S.COH_MIN} -1.0 {S.COH_MAX_SE3} -1 -1.0 -999\n"
        jobs.append((wd, ctl))
    t0 = time.time()
    with ProcessPoolExecutor(nproc) as ex:
        bad = [nm for nm, rc in ex.map(RN.run_one, jobs, chunksize=8) if rc != 0]
    with open(rd / "all.sum.grid0.loc.hyp", "w") as f:
        for wd, _ in jobs:
            p = wd / "loc.sum.grid0.loc.hyp"
            if p.exists():
                f.write(p.read_text())
    part = {e for e in compact if any(o in compact for _, o in nb.get(e, []))}
    (rd / "run.json").write_text(json.dumps({"events": len(jobs), "failed": bad, "duplicate_roots": len(dup),
                                             "events_final": len(roots), "events_compact": len(compact),
                                             "compact_with_compact_partner": len(part),
                                             "pairs_used": len(coh), "coh_min": S.COH_MIN, "coh_nmin": S.COH_NMIN,
                                             "max_se3": S.COH_MAX_SE3, "events_with_partners": len(nb),
                                             "held_out": sorted(held_out(s, run)),
                                             "grids": list(opt.roots), "stat": str(stat) if stat else None,
                                             "located_from": str(it_dir), "wall_s": round(time.time() - t0, 1)},
                                            indent=1))
    log(f"coherence {rd}: {len(jobs)} events, {len(nb)} with partners, {len(bad)} failed, {time.time() - t0:.0f} s")
    return rd


# ---------------------------------------------------------------------------------------------------------------- run
def cc_ready(s):
    f = CC_DIR / f"cc_{s}.parquet"
    if not f.exists():
        return False
    import pyarrow.parquet as pq
    return "straddles_clock_epoch" in pq.read_schema(f).names


def run_one_set_run(s, run, arms, nproc):
    blocks, box = prep(s, run)
    sp = S.station_phases(blocks)
    base = base_grids(s, box)
    d = run_dir(s, run)
    log(f"{s}/{run} ({G}): {len(blocks)} events, {len(sp)} station-phases, held out {len(held_out(s, run))} "
        f"stations, box x0 {box[0]} y0 {box[1]} {box[2]} x {box[3]} km")
    fw = (f"{S.GRIDS[G]['fine']}/cascadia3d", f"{S.GRIDS[G]['wide']}/cascadia3d")
    stat = S.GRIDS[G]["static"]
    so = None
    if "static" in arms or "static_coherence" in arms:
        so = S.nll_run(d / "static_oct", blocks, box, fw, stat=stat, octree=True, nproc=nproc_for(s, nproc))
    d_in, last = base, None
    if "ssst" in arms or "coherence" in arms:
        nfin = len(S.SCHEDULE)
        for k, char in enumerate(S.SCHEDULE):
            it = S.nll_run(d / "ssst" / f"it{k}", blocks, box, (f"{d_in}/cascadia3d",), nproc=nproc_for(s, nproc))
            d_in = S.loc2ssst(it, d_in, d / "ssst" / f"g{k}", char, box, sp, nproc=nproc_for(s, nproc))
        last = S.nll_run(d / "ssst" / f"it{nfin}", blocks, box, (f"{d_in}/cascadia3d",), octree=True,
                         nproc=nproc_for(s, nproc))
    if "static_coherence" in arms or "coherence" in arms:
        while not cc_ready(s):
            time.sleep(30)
        if "static_coherence" in arms:
            coherence_run(s, run, so, fw, box, stat=stat, name="static_coherence", nproc=nproc_for(s, nproc))
        if "coherence" in arms:
            coherence_run(s, run, last, (f"{d_in}/cascadia3d",), box, name="coherence", nproc=nproc_for(s, nproc))


def cmd_prep(a):
    """obs of every run and the grid links of every set, one process (the run chains then only read them)."""
    for s in SETS:
        box = set_box(s)
        base_grids(s, box)
        for run in run_names():
            blocks, _ = prep(s, run)
            ho = held_out(s, run)
            log(f"{s}/{run}: {len(blocks)} events in the obs, {len(ho)} stations held out")


def cmd_run(a):
    arms = a.arms.split(",")
    for run in a.runs.split(","):
        run_one_set_run(a.set, run, arms, a.nproc)
    log(f"{a.set} {a.runs} {a.arms} done")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("picks")
    sub.add_parser("prep")
    r = sub.add_parser("run")
    r.add_argument("set", choices=SETS)
    r.add_argument("--runs", default=",".join(run_names()))
    r.add_argument("--arms", default=",".join(ARMS))
    r.add_argument("--nproc", type=int, default=2)
    a = ap.parse_args()
    {"picks": cmd_picks, "prep": cmd_prep, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()

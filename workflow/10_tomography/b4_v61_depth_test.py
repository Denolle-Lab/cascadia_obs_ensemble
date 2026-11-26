"""Track B, catalog v6.1 (D-B100): station-term recipe test scored on relative fit AND absolute depth.

The pre-registered rule (written 2026-11-26 13:09 PST, before any run or score) is in
/wd1/mdenolle_data/reloc_v61/depth_test/RESUME.md; this script implements it.

Clusters: axial, msh (D-B078 sets), blancow (termtest set), gordad (b4_v61_depth_sets.py). Runs: full (every station)
and fold0-2 (held-out stations, b4_v6_heldout.folds). Everything uses b4_v6_heldout's prep / base grids / coherence
stage (partner rule R3) and b4_v6_ssst's nll_run / loc2ssst, unchanged.

Chains per run (run folder runs/<set>/<run>/):
* ssst/: it0 -> g0 (CharDist 9999) -> it1 -> g1 (16) -> it2 -> g2 (8) -> it3 -> g3 (4) -> it4; oct-trees saved on it0,
  it1, it2, it4 for the coherence stage (coh_it0, coh_it1, coh_it2, coh_it4);
* nf/: the same schedule, but every Loc2ssst output (nf/g<k>_raw; g0_raw = ssst/g0) is replaced by nf/g<k>: the
  corrected time grids minus a constant d_i per station-phase that removes the part of the correction a common shift
  of the cluster could absorb (nf_project); nf/it1..it4 read them; coherence on nf/it4 (coh_nf4).

Arms: it0, static (it1), ssst16 (it2), ssst4 (it4), nf4 (nf/it4), each without and with coherence (*_coh: events the
coherence stage relocates take its position, the others keep the pass's).

Processes: a shared pool of slots (control/slots, at most control/max_procs NLLoc/Loc2ssst processes, default 24);
a pass starts only while MemAvailable >= MEM_MIN_GB. Everything runs at the caller's nice (10).

  python b4_v61_depth_test.py run SET RUN          # one chain (resumable: run.json / loc2ssst.json / nf.json)
  python b4_v61_depth_test.py score [--nboot 2000] # -> score/
"""
import argparse
import fcntl
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("/wd1/mdenolle_data/reloc_v61/depth_test")
os.environ["V6_HELDOUT_OUT"] = str(OUT / "runs")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "08_relocation"))
import b4_v6_heldout as H  # noqa: E402
import b4_v6_ssst as S  # noqa: E402
from b4_v61_grids import Grid, parse_hyp  # noqa: E402

assert H.OUT == OUT / "runs", H.OUT
H.SETS["gordad"] = (OUT / "sets/events_gordad.csv", OUT / "inputs/picks_gordad.csv", OUT / "inputs/events_gordad.csv",
                    OUT / "cc/cc_gordad.parquet", 9, "Blanco (Gorda Depression)")
SETS = ("axial", "msh", "blancow", "gordad")
RUNS = ("full", "fold0", "fold1", "fold2")
SCHEDULE = (9999.0, 16.0, 8.0, 4.0)
OCT = {0, 1, 2, 4}
ARMS = {   # arm: (pass folder, coherence folder or None)
    "it0": ("ssst/it0", None), "it0_coh": ("ssst/it0", "coh_it0"),
    "static": ("ssst/it1", None), "static_coh": ("ssst/it1", "coh_it1"),
    "ssst16": ("ssst/it2", None), "ssst16_coh": ("ssst/it2", "coh_it2"),
    "ssst4": ("ssst/it4", None), "ssst4_coh": ("ssst/it4", "coh_it4"),
    "nf4": ("nf/it4", None), "nf4_coh": ("nf/it4", "coh_nf4"),
}
ORDER = list(ARMS)            # simplicity order of the rule (simplest first)
CTRL = OUT / "control"
MEM_MIN_GB = 45.0             # coordinator: keep MemAvailable above 40 GB
WANT = {"msh": 6, "blancow": 4, "gordad": 3, "axial": 2}
NEED = {"msh": 3, "blancow": 2, "gordad": 2, "axial": 1}


def log(*a):
    print(time.strftime("%F %T"), *a, flush=True)


def mem_gb():
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576.0
    return 1e9


class Slots:
    """at most control/max_procs processes over all chains: flock on slot files; a pass holds its slots. Every attempt
    locks all free slots, keeps what it may and releases the rest. msh (the longest chains) has priority: while an msh
    chain waits (fresh control/wait_* flag), the other sets leave RESERVE slots free."""
    RESERVE = 6

    def __init__(self, want, need=1, tag=None):
        self.want, self.need, self.fhs, self.tag = want, need, [], tag
        self.prio = bool(tag) and tag.startswith("msh")
        self.flag = CTRL / f"wait_{tag}" if tag else None

    @staticmethod
    def prio_waiting():
        now = time.time()
        return any(now - f.stat().st_mtime < 60 for f in CTRL.glob("wait_msh*"))

    def __enter__(self):
        (CTRL / "slots").mkdir(parents=True, exist_ok=True)
        while True:
            try:
                nmax = int((CTRL / "max_procs").read_text().split()[0])
            except (OSError, ValueError, IndexError):
                nmax = 24
            got = []
            if mem_gb() >= MEM_MIN_GB:
                for i in range(nmax):
                    fh = open(CTRL / "slots" / f"slot{i:02d}", "w")
                    try:
                        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        got.append(fh)
                    except BlockingIOError:
                        fh.close()
            reserve = 0 if self.prio or not self.prio_waiting() else self.RESERVE
            take = min(self.want, len(got) - reserve)
            if take >= self.need:
                for fh in got[take:]:
                    fcntl.flock(fh, fcntl.LOCK_UN)
                    fh.close()
                self.fhs = got[:take]
                if self.flag is not None and self.flag.exists():
                    self.flag.unlink()
                return take
            for fh in got:
                fcntl.flock(fh, fcntl.LOCK_UN)
                fh.close()
            if self.prio:
                self.flag.touch()
            time.sleep(3 + 4 * random.random())

    def __exit__(self, *a):
        for fh in self.fhs:
            fcntl.flock(fh, fcntl.LOCK_UN)
            fh.close()
        self.fhs = []


def lss_need(sp):
    """Loc2ssst instances needed so that every LSSTATIONS line stays below NonLinLoc's 1,000 characters."""
    stas = {sta for _, sta in sp}
    return max(1, math.ceil(sum(len(s) + 1 for s in stas) / 700))


# ------------------------------------------------------------------------------------------------- nf projection
def pass_arrivals(it_dir):
    """LOCATED events and their weighted arrivals of an NLLoc pass (per-event files of job*/)."""
    evs, phs = [], []
    for f in sorted(it_dir.glob("job*/loc.*.grid0.loc.hyp")):
        if ".sum." in f.name:
            continue
        e, p = parse_hyp(f)
        evs.append(e)
        phs.append(p)
    ev = pd.concat(evs, ignore_index=True)
    ph = pd.concat(phs, ignore_index=True)
    ev = ev[ev.status == "LOCATED"].drop_duplicates("ID").set_index("ID")
    ph = ph[ph.ID.isin(ev.index) & (ph.weight > 0)].reset_index(drop=True)
    ph["key"] = ph.phase + "." + ph.sta
    return ev, ph


def nf_project(it_dir, raw, d_in, d_out, base):
    """d_out = raw Loc2ssst output with the common-shift part removed (docstring of the module; RESUME.md rule).
    Corrected station-phases (regular files in raw): time grid = raw time grid - d_i (exact), ssst grid = raw - d_i;
    the others: links to d_in's grids (as Loc2ssst leaves them)."""
    done = d_out / "nf.json"
    if done.exists():
        return d_out
    t0 = time.time()
    ev, ph = pass_arrivals(it_dir)
    corr = sorted(f.name[len("cascadia3d."):-len(".ssst.buf")] for f in raw.glob("cascadia3d.*.ssst.buf")
                  if not f.is_symlink())
    corr = [k for k in corr if not (raw / f"cascadia3d.{k}.time.buf").is_symlink()]
    ph = ph[ph.key.isin(corr)].reset_index(drop=True)
    pos = ev.loc[ph.ID, ["x", "y", "z"]].to_numpy()
    G = np.zeros((len(ph), 4))
    G[:, 3] = 1.0
    c = np.zeros(len(ph))
    for key, r in ph.groupby("key"):
        ii = r.index.to_numpy()
        G[ii, :3] = Grid(base / f"cascadia3d.{key}.time.hdr").grad(pos[ii])
        c[ii] = Grid(raw / f"cascadia3d.{key}.ssst.hdr").at(pos[ii])
    w = ph.weight.to_numpy()
    # a_i: arrival-weighted mean partials of station-phase i; Abar: a_i on each arrival's row
    a = {key: np.average(G[r.index.to_numpy()], axis=0, weights=w[r.index.to_numpy()]) for key, r in ph.groupby("key")}
    Abar = np.vstack([a[k] for k in ph.key])
    ids = ph.ID.to_numpy()
    # sum_e G_e' W_e (c_e - Abar_e beta) = 0  ->  (sum G'W Abar) beta = sum G'W c   (rows of all events stacked)
    M = (G * w[:, None]).T @ Abar
    b = (G * w[:, None]).T @ c
    beta = np.linalg.solve(M, b)
    d = {k: float(a[k] @ beta) for k in a}
    # check: cluster-shift right-hand side before/after, and the per-event linearised shift (median dz) before/after
    rhs_after = (G * w[:, None]).T @ (c - Abar @ beta)
    order = np.argsort(ids, kind="stable")
    uniq, starts = np.unique(ids[order], return_index=True)
    ends = np.r_[starts[1:], len(order)]
    dz0, dz1 = [], []
    for s0, s1 in zip(starts, ends):
        ix = order[s0:s1]
        if len(ix) < 5:
            continue
        Gi, Wi = G[ix], w[ix]
        try:
            K = -np.linalg.solve(Gi.T @ (Wi[:, None] * Gi), (Gi * Wi[:, None]).T)
        except np.linalg.LinAlgError:
            continue
        dz0.append(K[2] @ c[ix])
        dz1.append(K[2] @ (c[ix] - Abar[ix] @ beta))
    d_out.mkdir(parents=True, exist_ok=True)
    for f in sorted(d_in.glob("cascadia3d.*.time.*")):
        ln = d_out / f.name
        if not ln.exists() and not ln.is_symlink():
            ln.symlink_to(os.path.realpath(f))
    nneg, tmin = 0, np.inf
    for key in corr:
        if key not in d:                       # corrected grid but no weighted arrival in the pass: keep it as is
            d[key] = 0.0
        for kind in ("time", "ssst"):
            src_h = raw / f"cascadia3d.{key}.{kind}.hdr"
            dst_h, dst_b = d_out / src_h.name, d_out / f"cascadia3d.{key}.{kind}.buf"
            for p in (dst_h, dst_b):
                if p.is_symlink() or p.exists():
                    p.unlink()
            v = np.fromfile(raw / f"cascadia3d.{key}.{kind}.buf", "<f4") - np.float32(d[key])
            v.astype("<f4").tofile(dst_b)
            dst_h.write_text(src_h.read_text())
            if kind == "time":
                nneg += int((v < 0).sum())
                tmin = min(tmin, float(v.min()))
    rec = dict(raw=str(raw), d_in=str(d_in), base=str(base), it=str(it_dir), events=int(len(ev)),
               arrivals=int(len(ph)), station_phases=len(corr), beta=beta.tolist(),
               beta_names=["x_s_per_km", "y_s_per_km", "z_s_per_km", "t0_s"], d=d,
               rhs_before=b.tolist(), rhs_after=rhs_after.tolist(),
               dz_lin_median_before=float(np.median(dz0)) if dz0 else None,
               dz_lin_median_after=float(np.median(dz1)) if dz1 else None,
               dz_lin_mean_before=float(np.mean(dz0)) if dz0 else None,
               dz_lin_mean_after=float(np.mean(dz1)) if dz1 else None,
               negative_nodes=nneg, t_min=tmin, wall_s=round(time.time() - t0, 1))
    done.write_text(json.dumps(rec, indent=1))
    log(f"nf {d_out}: beta {np.round(beta, 4).tolist()}, lin dz median {rec['dz_lin_median_before']:.3f} -> "
        f"{rec['dz_lin_median_after']:.3f} km, {nneg} negative nodes")
    return d_out


# ------------------------------------------------------------------------------------------------------------ chain
def nll(rd, blocks, box, groot, s, octree=False):
    if (rd / "run.json").exists():
        return rd
    with Slots(WANT[s], NEED[s], tag=f"{s}_{rd.parent.parent.name}") as n:
        return S.nll_run(rd, blocks, box, (f"{groot}/cascadia3d",), octree=octree, nproc=n)


def lss(it, d_in, d_out, char, box, sp, s):
    if (d_out / "loc2ssst.json").exists():
        return d_out
    need = lss_need(sp)
    with Slots(max(need, WANT[s]), need, tag=f"{s}_{d_out.parent.parent.name}") as n:
        return S.loc2ssst(it, d_in, d_out, char, box, sp, nproc=n)


def coh(s, run, it_dir, groot, box, name):
    if (H.OUT / s / run / name / "run.json").exists():
        return
    with Slots(WANT[s], NEED[s], tag=f"{s}_{run}") as n:
        H.coherence_run(s, run, it_dir, f"{groot}/cascadia3d", box, n, rule="r3", name=name)


def cmd_run(a):
    s, run = a.set, a.run
    blocks, box = H.prep(s, run)
    sp = S.station_phases(blocks)
    base = H.base_grids(s, box)
    d = H.OUT / s / run
    log(f"{s}/{run}: {len(blocks)} events, {len(sp)} station-phases, {len(H.held_out(s, run))} stations held out")
    # standard chain
    g = {-1: base}
    for k, char in enumerate(SCHEDULE):
        it = nll(d / "ssst" / f"it{k}", blocks, box, g[k - 1], s, octree=k in OCT)
        g[k] = lss(it, g[k - 1], d / "ssst" / f"g{k}", char, box, sp, s)
    nll(d / "ssst" / "it4", blocks, box, g[3], s, octree=True)
    # nf chain (it0 and its raw g0 are the standard chain's)
    gn = {-1: base}
    for k, char in enumerate(SCHEDULE):
        it = d / "ssst" / "it0" if k == 0 else nll(d / "nf" / f"it{k}", blocks, box, gn[k - 1], s)
        raw = g[0] if k == 0 else lss(it, gn[k - 1], d / "nf" / f"g{k}_raw", char, box, sp, s)
        gn[k] = nf_project(it, raw, gn[k - 1], d / "nf" / f"g{k}", base)
    nll(d / "nf" / "it4", blocks, box, gn[3], s, octree=True)
    # coherence stage on it0, it1, it2, it4 and nf/it4
    coh(s, run, d / "ssst" / "it0", base, box, "coh_it0")
    coh(s, run, d / "ssst" / "it1", g[0], box, "coh_it1")
    coh(s, run, d / "ssst" / "it2", g[1], box, "coh_it2")
    coh(s, run, d / "ssst" / "it4", g[3], box, "coh_it4")
    coh(s, run, d / "nf" / "it4", gn[3], box, "coh_nf4")
    (d / "DONE").write_text(time.strftime("%FT%T%z"))
    log(f"{s}/{run} done")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("set", choices=SETS)
    r.add_argument("run", choices=RUNS)
    sc = sub.add_parser("score")
    sc.add_argument("--nboot", type=int, default=2000)
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a)
    else:
        import b4_v61_depth_score as SC
        SC.main(a)


if __name__ == "__main__":
    main()

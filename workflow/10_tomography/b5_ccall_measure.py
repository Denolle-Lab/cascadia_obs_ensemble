"""Stage 3 of the all-family CC differential times (D-B093): cross-correlate every pair-station-phase.

The measurement is b5_cc_measure.measure, unchanged (D-B078 parameters, b5_cc_params.yaml; band 2-12 Hz for the
msh block, 3-15 Hz elsewhere). Only the event cache differs: a tile job (b5_ccall_tasks) loads every event it
needs once (pnwstore snippets of this run, the D-B078 cache read only, NCEDC snippets) and keeps each band-passed
trace as a cut [pick - 5 s, pick + 3 s] of the trace filtered over the full 16 s snippet (same samples as the
full trace inside the cut; the CC and noise windows are inside it). Windows are placed on the header-time picks
(t_hdr); dt_cat and dt_cc of the measurement are on the header base (dt_cc_hdr); b5_ccall_write.py moves both to
the v4.1tc base with the clock model.

Output per job and block: <OUT>/work/raw/<block>/<job>.parquet (pair, p1, p2 = rows of tasks/picks.parquet, the
measure() columns); <OUT>/work/done/<job> marks a finished job (resumable).

  python b5_ccall_measure.py [--workers 24 --max-workers 48] [--validate-set jdf]
"""
import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import obspy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b5_ccall_common as A  # noqa: E402
import b5_cc_common as C  # noqa: E402
import b5_cc_measure as M  # noqa: E402

CUT_PRE, CUT_POST = 5.0, 3.0
G = {}            # task tables, filled in the parent before the pool forks


class Cut:
    """Band-passed trace kept only on [pick - CUT_PRE, pick + CUT_POST]; len() and slicing behave as on the full
    filtered trace (same indices), as long as the slice lies inside the cut."""
    __slots__ = ("y", "off", "n")

    def __init__(self, y, off, n):
        self.y, self.off, self.n = y, off, n

    def __len__(self):
        return self.n

    def __getitem__(self, s):
        a, b = s.start - self.off, s.stop - self.off
        if a < 0 or b > len(self.y) or s.step is not None:
            raise IndexError(f"slice {s.start}:{s.stop} outside the cut [{self.off}, {self.off + len(self.y)})")
        return self.y[a:b]


class Store:
    """Event cache for M.measure: get(eid) -> ({(sta, phase): {channel: Trace}}, None); pre = prepped cuts."""

    def __init__(self):
        self.d = {}
        self.pre = {}
        self.pick = {}

    def load(self, eid, picks_of_event):
        ev = {}
        srcs = []
        old = G["old"].get(eid)
        bases = [Path(old)] if old else [A.snip_dir(A.SNIP, eid)]
        bases.append(A.snip_dir(A.SNIP_NC, eid))
        for b in bases:
            for ph in ("P", "S"):
                f = b / f"{eid}.{ph}.mseed"
                if f.exists():
                    for tr in obspy.read(str(f)):
                        ev.setdefault((f"{tr.stats.station}.{tr.stats.network}", ph), {})[tr.stats.channel] = tr
                    srcs.append(str(f))
        self.d[eid] = (ev, None)
        for sta, ph, t in picks_of_event:
            self.pick[(eid, sta, ph)] = t

    def get(self, eid):
        return self.d[eid]


def prepped_cut(cache, eid, sta, ph, cha, tr, sr_target, cfg):
    """M.prepped with the result kept as a Cut around the event's header-time pick (no size cap: one tile)."""
    key = (eid, sta, ph, cha, sr_target, tuple(cfg["band"]))
    v = cache.pre.get(key)
    if v is not None:
        return v
    sr = tr.stats.sampling_rate
    x = tr.data.astype(np.float64)
    if abs(sr - sr_target) <= 1e-3 * sr:
        sr_target = sr
    else:
        x = M._resample(x - x.mean(), sr, sr_target)
    y = C.prep(x, sr_target, cfg["band"], cfg["taper"])
    s0 = tr.stats.starttime.timestamp
    t = cache.pick[(eid, sta, ph)]
    a = max(0, int(np.floor((t - CUT_PRE - s0) * sr_target)))
    b = min(len(y), int(np.ceil((t + CUT_POST - s0) * sr_target)) + 1)
    out = (Cut(y[a:b].copy(), a, len(y)), s0)
    cache.pre[key] = out
    return out


M.prepped = prepped_cut          # measure() looks the name up in its module at call time


def load_tables(out=None):
    T = A.TASKS
    ev = pd.read_parquet(T / "events.parquet", columns=["ID", "T0", "block", "ev_i"])
    pk = pd.read_parquet(T / "picks.parquet", columns=["pk_i", "ev_i", "station", "phase", "spc", "t_hdr", "T0",
                                                      "sp_est"])
    pr = pd.read_parquet(T / "pairs.parquet", columns=["pair", "i1", "i2", "block"])
    jobs = pd.read_parquet(T / "jobs.parquet")
    assert (pk.pk_i.values == np.arange(len(pk))).all() and (pr.pair.values == np.arange(len(pr))).all()
    starts = np.searchsorted(pk.ev_i.values, np.arange(len(ev) + 1))
    G.update(ev_id=ev.ID.values, ev_T0=ev.T0.values, pk_ev=pk.ev_i.values, pk_spc=pk.spc.values,
             pk_sta=pk.station.values, pk_ph=pk.phase.values, pk_t=pk.t_hdr.values, pk_T0=pk.T0.values,
             pk_sp=pk.sp_est.values, starts=starts, pr_i1=pr.i1.values, pr_i2=pr.i2.values,
             pr_block=pr.block.values, jobs=jobs)
    import b5_ccall_snippets as S
    G["old"] = S.old_cache_index()
    G["cfg"] = {b: {**M.DEFAULT, **A.load_params(b)[0]} for b in A.BLOCKS}


def obs_rows(pair_ids):
    """(pair, p1, p2) for every common station-phase of the pairs (picks sorted by event, then spc)."""
    st, spc = G["starts"], G["pk_spc"]
    P, P1, P2 = [], [], []
    for p in pair_ids:
        i1, i2 = G["pr_i1"][p], G["pr_i2"][p]
        a = np.arange(st[i1], st[i1 + 1])
        b = np.arange(st[i2], st[i2 + 1])
        _, x, y = np.intersect1d(spc[a], spc[b], assume_unique=True, return_indices=True)
        if len(x):
            P.append(np.full(len(x), p))
            P1.append(a[x])
            P2.append(b[y])
    if not P:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(P), np.concatenate(P1), np.concatenate(P2)


def run_tile(task):
    """task = list of (job, pair_lo, pair_hi) of one tile; returns [(job, rows, status counts, seconds)]."""
    t0 = time.time()
    todo = [j for j in task if not (A.WORK / "done" / j[0]).exists()]
    if not todo:
        return []
    pairs = np.concatenate([np.arange(lo, hi + 1) for _, lo, hi in todo])
    evs = np.unique(np.r_[G["pr_i1"][pairs], G["pr_i2"][pairs]])
    store = Store()
    st = G["starts"]
    for e in evs:
        k = np.arange(st[e], st[e + 1])
        store.load(int(G["ev_id"][e]), zip(G["pk_sta"][k], G["pk_ph"][k], G["pk_t"][k]))
    t_load = time.time() - t0
    res = []
    for job, lo, hi in todo:
        t1 = time.time()
        pid, p1, p2 = obs_rows(np.arange(lo, hi + 1))
        out = []
        for p, a, b in zip(pid, p1, p2):
            e1, e2 = G["pr_i1"][p], G["pr_i2"][p]
            row = SimpleNamespace(pair=int(p), station=G["pk_sta"][a], phase=G["pk_ph"][a],
                                  ID1=int(G["ev_id"][e1]), ID2=int(G["ev_id"][e2]),
                                  sp_est1=G["pk_sp"][a], sp_est2=G["pk_sp"][b],
                                  t_pick1=G["pk_t"][a], t_pick2=G["pk_t"][b], T01=G["pk_T0"][a], T02=G["pk_T0"][b])
            row.dt_cat = (row.t_pick1 - row.T01) - (row.t_pick2 - row.T02)      # header base
            try:
                r = M.measure(row, store, G["cfg"][G["pr_block"][p]])
            except Exception as e:
                r = dict(pair=int(p), status=f"error:{type(e).__name__}:{str(e)[:60]}")
            r.pop("station", None)
            r.pop("phase", None)
            out.append(r)
        df = pd.DataFrame(out)
        df["p1"] = p1.astype(np.int32)
        df["p2"] = p2.astype(np.int32)
        if "dt_cc" in df:
            df = df.rename(columns={"dt_cc": "dt_cc_hdr"})
        df["block"] = G["pr_block"][pid] if len(pid) else []
        for blk, g in df.groupby("block"):
            d = A.WORK / "raw" / blk
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / f"{job}.tmp.parquet"
            g.drop(columns="block").to_parquet(tmp, index=False)
            os.replace(tmp, d / f"{job}.parquet")
        (A.WORK / "done").mkdir(parents=True, exist_ok=True)
        stat = df.status.value_counts().to_dict() if len(df) else {}
        (A.WORK / "done" / job).write_text(json.dumps(dict(rows=len(df), status=stat, s=round(time.time() - t1, 1),
                                                           load_s=round(t_load, 1), events=len(evs))))
        res.append((job, len(df), stat, time.time() - t1))
    return res


def safe_tile(task):
    try:
        return run_tile(task)
    except Exception:
        traceback.print_exc()
        return [(task[0][0], -1, {"error": 1}, 0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--max-workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0, help="only the first N tiles (test)")
    a = ap.parse_args()
    logf = A.OUT / "logs" / "measure.log"
    logf.parent.mkdir(parents=True, exist_ok=True)
    load_tables()
    jobs = G["jobs"]
    done = {p.name for p in (A.WORK / "done").glob("*")} if (A.WORK / "done").exists() else set()
    tiles = []
    for (ta, tb), g in jobs.groupby(["tile_a", "tile_b"]):
        tiles.append((int(g.rows.sum()), [(j, int(lo), int(hi)) for j, lo, hi in zip(g.job, g.pair_lo, g.pair_hi)]))
    tiles.sort(key=lambda x: -x[0])
    tiles = [t for r, t in tiles if any(j not in done for j, _, _ in t)]
    if a.limit:
        tiles = tiles[:a.limit]
    total_rows = int(jobs[~jobs.job.isin(done)].rows.sum())
    A.log(f"measure: {len(jobs)} jobs, {len(done)} done, {len(tiles)} tiles to run, {total_rows} rows; "
          f"workers {a.workers}..{a.max_workers}", logf)
    A.write_state("measure", jobs=len(jobs), done_at_start=len(done), tiles_todo=len(tiles), rows_todo=total_rows,
                  done=False)
    W = a.workers
    t0 = time.time()
    n_rows = n_jobs = n_err = 0
    last_adj = time.time()
    high_since = None
    last_log = 0
    with ProcessPoolExecutor(max_workers=a.max_workers) as ex:
        it = iter(tiles)
        running = set()
        exhausted = False
        while running or not exhausted:
            mem = A.mem_available_gb()
            now = time.time()
            high_since = (high_since or now) if mem > 60 else None
            if now - last_adj > 60:
                if high_since and now - high_since > 120 and W < a.max_workers:
                    W = min(a.max_workers, W + 4)
                    A.log(f"MemAvailable {mem:.0f} GB > 60 for 2 min: in-flight -> {W}", logf)
                    last_adj = now
                elif mem < 30 and W > 8:
                    W = max(8, W - 4)
                    A.log(f"MemAvailable {mem:.0f} GB < 30: in-flight -> {W}", logf)
                    last_adj = now
            while not exhausted and len(running) < W and A.mem_available_gb() > 22:
                t = next(it, None)
                if t is None:
                    exhausted = True
                    break
                running.add(ex.submit(safe_tile, t))
            if not running:
                time.sleep(5)
                continue
            fin, running = wait(running, timeout=30, return_when=FIRST_COMPLETED)
            running = set(running)
            for f in fin:
                for job, nr, stat, s in f.result():
                    n_jobs += 1
                    if nr < 0:
                        n_err += 1
                    else:
                        n_rows += nr
            if time.time() - last_log > 120 or (not running and exhausted):
                last_log = time.time()
                el = time.time() - t0
                rate = n_rows / el if el else 0
                A.log(f"{n_jobs} jobs, {n_rows}/{total_rows} rows, {n_err} failed tiles, {rate:.0f} rows/s, in-flight "
                      f"{W}, MemAvailable {mem:.0f} GB, ETA {(total_rows - n_rows) / max(rate, 1) / 3600:.2f} h", logf)
                A.write_state("measure", jobs_done=n_jobs, rows_done=n_rows, failed_tiles=n_err,
                              rate_rows_s=round(rate), in_flight=W)
    done = {p.name for p in (A.WORK / "done").glob("*")}
    missing = int((~jobs.job.isin(done)).sum())
    A.write_state("measure", done=missing == 0, missing_jobs=missing, rows_done=n_rows, failed_tiles=n_err,
                  wall_s=round(time.time() - t0), max_in_flight=W)
    A.log(f"measure finished: {n_jobs} jobs, {n_rows} rows, {n_err} failed tiles, {missing} jobs missing, "
          f"{time.time() - t0:.0f} s", logf)


if __name__ == "__main__":
    main()

"""Track B, v6.0 station-term test: waveform CC differential times of the three test sets (b4_v6_termtest_sets.py),
with the tested D-B040/D-B078 code (tomo/cc-dt 2d033ec6: b5_cc_tasks, b5_cc_snippets, b5_cc_measure, b5_cc_write;
only the output folder and set list come from the environment, B5_CC_OUT / B5_CC_SETDIR / B5_CC_SETS) and the
parameters b4_v6_termtest_cc_params.yaml (= b5_cc_params.yaml, band 3-15 Hz for the three sets).

Snippets. The D-B093 all-family job (tomo/cc-dt-all, /wd1/mdenolle_data/tomo_v1/cc_dt_all) reads the same snippets
with the same reader (b5_cc_snippets.event_snippets, b5_cc_common.read_channel) and the NC/BK snippets from NCEDC
FDSN (b5_ccall_ncedc.py). Its caches are read only here: an event whose pnwstore snippet is complete there
(meta file) is copied; the others are read from pnwstore by event_snippets into this folder, with the D-B093 guard
against a chunk that looks like a record header but is not (safe_parse_headers, 7D FS*B 2012-13). NC/BK snippets
(mtj) come from the NCEDC cache; they are merged into the event's P/S files and replace its no_pnwstore meta rows.

Station clocks: the confirmed D-B090 model (clock_model.json; MH07.XD and MG07.XD, spring 2016; puget has 24 such
picks) gives clock_epoch1/2, clock_c1/2, clock_dt_s and straddles_clock_epoch as in the D-B078 parquet (sub-epochs
of at most 20 ms of modelled drift).

  python b4_v6_termtest_cc.py [--stages tasks,snippets,measure,write,clock] [--sets puget,mtj,blancow] [--workers 8]
      -> /wd1/mdenolle_data/reloc_v6/termtest/cc/{cc_<set>.parquet, dt_<set>*.cc, qc/, snippets/<set>/, <set>/}
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TT = Path("/wd1/mdenolle_data/reloc_v6/termtest")
CCOUT = TT / "cc"
SETDIR = TT / "sets"
SETS = ("puget", "mtj", "blancow")
PARAMS = HERE / "b4_v6_termtest_cc_params.yaml"
ALL = Path("/wd1/mdenolle_data/tomo_v1/cc_dt_all")             # D-B093 job: read only
ALL_SNIP, ALL_NC = ALL / "snippets", ALL / "snippets_ncedc"
CLOCK_MODEL = Path("/wd1/mdenolle_data/tomo_v1/checks/pick_timing_v6/clock_model.json")
PY = "/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/bin/python"
ENV = dict(os.environ, B5_CC_OUT=str(CCOUT), B5_CC_SETDIR=str(SETDIR), B5_CC_SETS=",".join(SETS),
           LD_LIBRARY_PATH="/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/lib", OMP_NUM_THREADS="1")
os.environ.update({k: ENV[k] for k in ("B5_CC_OUT", "B5_CC_SETDIR", "B5_CC_SETS")})
sys.path.insert(0, str(HERE))
import b5_cc_common as C  # noqa: E402
import b5_cc_snippets as CS  # noqa: E402

assert C.OUT == CCOUT and C.SETDIR == SETDIR


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def shard(eid):
    return f"{int(eid) // 1000:03d}"


# ---------------------------------------------------------------------------------------------------------- snippets
_parse = C.parse_headers


def safe_parse_headers(buf):
    """as b5_ccall_snippets.safe_parse_headers (D-B093): a garbage date in a chunk that is not on a record boundary
    gives no record, so records_window reads the whole index block (its own rule) instead of raising."""
    try:
        return _parse(buf)
    except ValueError:
        return pd.DataFrame(columns=["offset", "reclen", "t", "npts", "rate", "cha"])


def init_worker():
    C.parse_headers = safe_parse_headers


def read_one(t):
    try:
        return CS.event_snippets(t)
    except Exception:
        traceback.print_exc()
        return t[1], "error", 0


def merge_nc(s, eid):
    """append the NCEDC snippets of eid (D-B093 cache) to its P/S files and meta (replacing no_pnwstore rows)."""
    import obspy
    d = C.SNIP / s
    src = ALL_NC / shard(eid)
    fm = d / f"{eid}.meta.csv"
    m = pd.read_csv(fm)
    if "source" in m and (m.source == "ncedc").any():
        return "already"
    mn = src / f"{eid}.meta.csv"
    if not mn.exists():
        return "missing"
    nc = pd.read_csv(mn)
    for ph in ("P", "S"):
        fn = src / f"{eid}.{ph}.mseed"
        if not fn.exists():
            continue
        new = obspy.read(str(fn))
        f = d / f"{eid}.{ph}.mseed"
        st = obspy.read(str(f)) if f.exists() else obspy.Stream()
        st += new
        for tr in st:
            tr.stats.pop("mseed", None)
        ints = obspy.Stream([tr for tr in st if tr.data.dtype.kind != "f"])
        flts = obspy.Stream([tr for tr in st if tr.data.dtype.kind == "f"])
        tmp = f.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            if len(ints):
                ints.write(fh, format="MSEED", encoding="STEIM2", reclen=4096)
            if len(flts):
                flts.write(fh, format="MSEED", encoding="FLOAT32", reclen=4096)
        os.replace(tmp, f)
    m = pd.concat([m[m.status != "no_pnwstore"], nc], ignore_index=True)
    tmp = fm.with_suffix(".tmp")
    m.to_csv(tmp, index=False)
    os.replace(tmp, fm)
    return "merged"


def stage_snippets(a):
    for s in a.sets:
        picks = pd.read_parquet(CCOUT / s / "picks.parquet")
        ids = sorted(picks.ID.unique())
        d = C.SNIP / s
        d.mkdir(parents=True, exist_ok=True)
        n_copy = 0
        for i in ids:
            if (d / f"{i}.meta.csv").exists():
                continue
            src = ALL_SNIP / shard(i)
            if (src / f"{i}.meta.csv").exists():
                for ph in ("P", "S"):
                    f = src / f"{i}.{ph}.mseed"
                    if f.exists():
                        shutil.copyfile(f, d / f"{i}.{ph}.mseed.tmp")
                        os.replace(d / f"{i}.{ph}.mseed.tmp", d / f"{i}.{ph}.mseed")
                shutil.copyfile(src / f"{i}.meta.csv", d / f"{i}.meta.tmp")
                (d / f"{i}.meta.csv.copied_from_cc_dt_all").write_text(str(src) + "\n")
                os.replace(d / f"{i}.meta.tmp", d / f"{i}.meta.csv")
                n_copy += 1
        todo = [i for i in ids if not (d / f"{i}.meta.csv").exists()]
        log(f"{s}: {len(ids)} events, {n_copy} copied from the D-B093 cache, {len(todo)} to read ({a.workers} workers)")
        tasks = [(s, int(i), picks[picks.ID == i], {}) for i in todo]
        n = 0
        t0 = time.time()
        if tasks:
            with Pool(a.workers, initializer=init_worker) as pool:
                for eid, st, k in pool.imap_unordered(read_one, tasks, chunksize=1):
                    n += 1
                    if n % 100 == 0 or n == len(tasks):
                        log(f"{s}: {n}/{len(tasks)} read, {time.time() - t0:.0f} s")
        miss = [i for i in ids if not (d / f"{i}.meta.csv").exists()]
        assert not miss, f"{s}: {len(miss)} events without snippets, e.g. {miss[:5]}"
        nc_ids = sorted(picks[picks.net.isin(["NC", "BK"])].ID.unique())
        res = {}
        for i in nc_ids:
            r = merge_nc(s, i)
            res[r] = res.get(r, 0) + 1
        log(f"{s}: NC/BK snippets {res} (events with NC/BK picks: {len(nc_ids)})")
        (d / "DONE").write_text(json.dumps(dict(events=len(ids), copied=n_copy, read=len(todo), nc=res,
                                                utc=time.strftime("%Y-%m-%dT%H:%M:%S%z")), indent=1))


# ------------------------------------------------------------------------------------------------------------- clock
def stage_clock(a):
    M = json.loads(CLOCK_MODEL.read_text())["stations"]
    step = 0.02
    edges = {}
    for st, eps in M.items():
        for e in eps:
            dur = (e["t_end_s"] - e["t_start_s"]) / 86400.0
            n = int(np.ceil(abs(e["size"]) * dur / step))
            edges[st] = np.array([e["t_start_s"] + (e["t_end_s"] - e["t_start_s"]) * k / n for k in range(n + 1)])

    def c_of(st, t):
        c = np.zeros(len(t))
        for e in M.get(st, []):
            m = (t >= e["t_start_s"]) & (t < e["t_end_s"])
            c[m] = e["size"] * (t[m] - e["t_start_s"]) / 86400.0
        return c

    def ep_of(st, t):
        if st not in edges:
            return np.zeros(len(t), int)
        k = np.searchsorted(edges[st], t, side="right")
        return np.where((k >= 1) & (k <= len(edges[st]) - 1), k, 0)

    for s in a.sets:
        f = CCOUT / f"cc_{s}.parquet"
        c = pd.read_parquet(f)
        for k in ("1", "2"):
            c[f"clock_epoch{k}"] = 0
            c[f"clock_c{k}"] = 0.0
        for st, g in c.groupby("station"):
            if st not in M:
                continue
            for k in ("1", "2"):
                t = g[f"t_pick{k}"].to_numpy(float)
                c.loc[g.index, f"clock_c{k}"] = c_of(st, t)
                c.loc[g.index, f"clock_epoch{k}"] = ep_of(st, t)
        c["clock_dt_s"] = c.clock_c1 - c.clock_c2
        c["straddles_clock_epoch"] = c.clock_epoch1 != c.clock_epoch2
        c.to_parquet(f, index=False)
        log(f"{s}: clock columns; rows at clock stations {int(c.station.isin(list(M)).sum())}, straddling "
            f"{int(c.straddles_clock_epoch.sum())} (accepted {int((c.straddles_clock_epoch & c.above).sum())})")


def sh(cmd):
    log("$", " ".join(cmd))
    subprocess.run(cmd, check=True, env=ENV, cwd=HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="tasks,snippets,measure,write,clock")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    a.sets = a.sets.split(",")
    st = a.stages.split(",")
    CCOUT.mkdir(parents=True, exist_ok=True)
    if "tasks" in st:
        sh([PY, "b5_cc_tasks.py", "--sets", ",".join(a.sets)])
    if "snippets" in st:
        stage_snippets(a)
    for s in a.sets:
        if "measure" in st:
            sh([PY, "b5_cc_measure.py", "--set", s, "--params", str(PARAMS), "--out",
                str(CCOUT / s / "cc_raw.parquet"), "--workers", str(a.workers)])
        if "write" in st:
            (CCOUT / "logs").mkdir(exist_ok=True)
            with open(CCOUT / "logs" / f"write_{s}.log", "w") as fh:
                log("$ b5_cc_write.py --set", s)
                subprocess.run([PY, "b5_cc_write.py", "--set", s, "--params", str(PARAMS)], check=True, env=ENV,
                               cwd=HERE, stdout=fh, stderr=subprocess.STDOUT)
    if "clock" in st:
        stage_clock(a)
    log("done", a.stages)


if __name__ == "__main__":
    main()

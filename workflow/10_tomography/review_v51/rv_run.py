"""Batch driver of the v5.1 location review (resumable: cached events and existing figures are skipped).

  python rv_run.py compute [--nproc 12] [--problems P1,P2,P3,P4]   # WORK/events/<ID>.json/.npz/.scat.npy
  python rv_run.py figs    [--nproc 12] [--force]                  # OUT/<folder>/fig/<ID>_{geom,rec,zoom}.webp
"""
import argparse
import os
import sys
import time
import traceback
from multiprocessing import Pool

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import pandas as pd  # noqa: E402

import rv_common as C  # noqa: E402

FOLDERS = {"P1": "P1_msh_deep", "P2": "P2_brookings", "P3": "P3_tensor", "P4": "P4_deep_far"}


def views(problems):
    out = []
    for p in problems:
        v = pd.read_csv(C.WORK / "sets" / f"{p}_views.csv")
        v["problem"] = p
        out.append(v[["ID", "problem"]])
    return pd.concat(out)


def _compute(eid):
    import rv_event as E
    t = time.time()
    try:
        d = E.compute(int(eid))
        return eid, "ok" if "error" not in d else d["error"], round(time.time() - t, 1)
    except Exception as e:
        return eid, f"FAILED {type(e).__name__}: {e} | {traceback.format_exc()[-400:]}", round(time.time() - t, 1)


def _figs(args):
    eid, problem, force = args
    import rv_figs as F
    t = time.time()
    try:
        r = F.make_all(int(eid), C.OUT / FOLDERS[problem] / "fig", force=force, raw=(problem == "P1"))
        return eid, str(r), round(time.time() - t, 1)
    except Exception as e:
        return eid, f"FAILED {type(e).__name__}: {e} | {traceback.format_exc()[-400:]}", round(time.time() - t, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["compute", "figs"])
    ap.add_argument("--nproc", type=int, default=12)
    ap.add_argument("--problems", default="P1,P2,P3,P4")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    V = views(a.problems.split(","))
    t0 = time.time()
    if a.stage == "compute":
        todo = [e for e in V.ID.unique() if not (C.WORK / "events" / f"{e}.json").exists()]
        print(f"compute: {len(todo)} of {V.ID.nunique()} events to do", flush=True)
        jobs, fn = todo, _compute
    else:
        jobs, fn = [(r.ID, r.problem, a.force) for r in V.itertuples()], _figs
        print(f"figs: {len(jobs)} event views", flush=True)
    nbad = 0
    with Pool(min(a.nproc, 12)) as pool:
        for i, (eid, st, dt) in enumerate(pool.imap_unordered(fn, jobs, chunksize=1)):
            bad = st.startswith("FAILED") or st.startswith("no NonLinLoc")
            nbad += bad
            if bad or i % 20 == 0:
                print(f"{time.strftime('%H:%M:%S')} {i + 1}/{len(jobs)} {eid} {dt}s {st[:300]}", flush=True)
    print(f"{a.stage} done in {time.time() - t0:.0f} s, {nbad} failures", flush=True)


if __name__ == "__main__":
    main()

"""Fault surfaces v1, step 1: plane detection on catalog v5.1 (the v0 method; dip rule also after reassignment).

Runs the detection of fault_surfaces_planes.py (sequential local RANSAC with growth, k-planes reassignment, merging,
reassignment) on the v1 fit set (v1_common.load) with the v0 parameters, with one change: the dip >= 30 deg rule is
applied after every reassignment pass too (reassign_dip). Runs: the main run (seed 0, tolerance factor
1.5) and the four stability runs of v0 (seeds 101 and 202; tolerance factor 1.25 and 1.75), in parallel.
Each run is cached in WORK/detect_<name>.npz (event_id lists per plane); finished runs are not redone.

    python workflow/07_faults/v1_detect.py [--procs 5]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

import v1_common as C
import fault_surfaces_planes as fp

RUNS = {"main": (0, None), "seed101": (101, None), "seed202": (202, None), "k1.25": (0, 1.25), "k1.75": (0, 1.75)}


def reassign_dip(P, planes, a, k=None):
    """fault_surfaces_planes.reassign, plus the dip rule after every pass: a plane whose members fit a plane flatter
    than --min-dip is dropped and its events are free for the other planes in the next pass. v0 checked the dip only
    at detection and merging, and 24 of its 141 surfaces ended flatter than 30 deg after reassignment."""
    from scipy.spatial import cKDTree
    import fault_surfaces_v0 as v0
    tree = cKDTree(P)
    for _ in range(3):
        best = np.full(len(P), np.inf)
        owner = -np.ones(len(P), int)
        for i, m in enumerate(planes):
            if len(m) < 3:
                continue
            c0, n, eu, ev, (u0, u1, v0_, v1) = fp.rect_of(P, m, a)
            R = np.hypot(max(abs(u0), abs(u1)), max(abs(v0_), abs(v1))) + 1.0
            cand = np.array(tree.query_ball_point(c0, R))
            Q = P[cand] - c0
            u, v, w = Q @ eu, Q @ ev, Q @ n
            r = np.abs(w) / fp.tol_of(n, a, k)
            ok = (r <= 1.0) & (u >= u0) & (u <= u1) & (v >= v0_) & (v <= v1)
            cand, r = cand[ok], r[ok]
            better = r < best[cand]
            best[cand[better]] = r[better]
            owner[cand[better]] = i
        planes = [np.flatnonzero(owner == i) for i in range(len(planes))]
        planes = [m for m in planes if len(m) >= a.n_min and v0.strike_dip(v0.pca(P[m])[2][:, 2])[1] >= a.min_dip]
    return planes


def run_detection(c, a, seed, k=None):
    P = np.c_[c.x, c.y, c.z]
    rng = np.random.default_rng(seed)
    planes = reassign_dip(P, fp.detect(P, a, rng, k), a, k)
    return reassign_dip(P, fp.merge(P, planes, a, k), a, k)


def one(name):
    out = C.WORK / f"detect_{name}.npz"
    if out.exists():
        return name, "cached"
    c, _ = C.load()
    f = C.fit_set(c)
    seed, k = RUNS[name]
    a = C.params()
    t0 = time.time()
    planes = run_detection(f, a, seed, k)
    ev = f.event_id.to_numpy()
    np.savez_compressed(out.with_suffix(".tmp.npz"), **{f"p{i:04d}": ev[m] for i, m in enumerate(planes)})
    os.replace(out.with_suffix(".tmp.npz"), out)
    msg = f"{len(planes)} planes, {sum(len(m) for m in planes):,} of {len(f):,} events, {time.time() - t0:.0f} s"
    (C.WORK / f"detect_{name}.json").write_text(json.dumps(dict(name=name, seed=seed, k=k, msg=msg)))
    return name, msg


def load_run(name):
    z = np.load(C.WORK / f"detect_{name}.npz")
    return [z[k] for k in sorted(z.files)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--only", nargs="*", default=list(RUNS))
    a = ap.parse_args()
    C.WORK.mkdir(parents=True, exist_ok=True)
    with Pool(min(a.procs, len(a.only))) as pool:
        for name, msg in pool.imap_unordered(one, a.only):
            print(f"{time.strftime('%H:%M:%S')} {name}: {msg}", flush=True)


if __name__ == "__main__":
    main()

"""Fault surfaces v1: the detection reruns that v1_build.py reads for its detection-level recovery columns.

The adversarial verifier of the faults piece reran the plane detection of v1_detect.py three times (scripts kept outside
git in /wd1/mdenolle_data/catalog_v5/v5.1_analysis/faults/work/verifier_reruns/: rerun_detect.py and
rerun_detect_perturbed.py, 2026-11-25). This file is the same code in the repository, so the pipeline can make the
reruns for a new catalog (D-B110):
  sub80      a random 80 % of the fit set (numpy default_rng(7)), seed 0
  tierAB     tiers A and B of the fit set only, seed 0
  perturbed  every fit event moved once by its own catalog error (sigma_h, sigma_h, sigma_z; default_rng(11)), seed 0
Each run writes WORK/verifier_reruns/detect_<name>.npz (one event_id array per plane) and detect_<name>.json; a run
whose npz exists is skipped. v1_build.recovery_runs reads them; without them the recovery_* columns stay empty.
Paths follow v1_common (FAULTS_V1_* environment variables for another catalog).

    python workflow/07_faults/v1_recovery.py [--only sub80 tierAB perturbed] [--procs 3]
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool

import numpy as np

import v1_common as C
import fault_surfaces_v0 as v0
from v1_detect import run_detection

OUT = C.WORK / "verifier_reruns"
RUNS = ("sub80", "tierAB", "perturbed")


def rerun(name: str) -> dict:
    out = OUT / f"detect_{name}.npz"
    if out.exists():
        return dict(name=name, cached=True)
    c, _ = C.load()
    f = C.fit_set(c)
    if name == "sub80":
        rng = np.random.default_rng(7)
        f = f.iloc[np.sort(rng.choice(len(f), int(0.8 * len(f)), replace=False))].reset_index(drop=True)
    elif name == "tierAB":
        f = f[f.tier.isin(["A", "B"])].reset_index(drop=True)
    elif name == "perturbed":
        rng = np.random.default_rng(11)
        lat0 = f.latitude.to_numpy()
        dx = rng.normal(size=len(f)) * f.sh.to_numpy()
        dy = rng.normal(size=len(f)) * f.sh.to_numpy()
        dz = rng.normal(size=len(f)) * f.sz.to_numpy()
        lon = f.longitude.to_numpy() + dx / (v0.KM * np.cos(np.radians(lat0)))
        lat = lat0 + dy / v0.KM
        f = f.copy()
        f["x"], f["y"] = v0.sinu(lon, lat)
        f["z"] = f.z.to_numpy() + dz
    else:
        raise SystemExit(f"unknown rerun {name}")
    t0 = time.time()
    planes = run_detection(f, C.params(), 0, None)
    ev = f.event_id.to_numpy()
    np.savez_compressed(out, **{f"p{i:04d}": ev[m] for i, m in enumerate(planes)})
    msg = dict(name=name, n_fit=len(f), planes=len(planes), members=int(sum(len(m) for m in planes)),
               sec=round(time.time() - t0))
    (OUT / f"detect_{name}.json").write_text(json.dumps(msg))
    return msg


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", default=list(RUNS), choices=RUNS)
    ap.add_argument("--procs", type=int, default=3)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(min(a.procs, len(a.only))) as pool:
        for msg in pool.imap_unordered(rerun, a.only):
            print(msg, flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Q1 (D-B115): clustering controls and nulls that need the clustering code (v51_clusters_zbz.py, catalog v6.1).

Steps (RESUME.md, pre-registered):
  mcu     the clustering redone with Mc = 2.15 in every region (thresholds refitted by the same Gaussian mixture);
          writes nulls/recluster_mcu/{events.csv.gz, families.csv, thresholds.csv}
  tshuf   time-shuffled catalogs: event times permuted within each clustering region over the whole period, the REAL
          v6.1 thresholds and caps, families formed and classified as in the real run; 100 catalogs (seeds 5000+k);
          stored per catalog as the IDs of the analysis-set events in swarm families -> nulls/tshuf_swarm_ids.parquet
          (seed, ID); per-unit shares are formed in q1_tests.py with the current segment join
  mshuf   magnitude shuffle: magnitudes permuted among the analysis-set events of each region (200 draws, seeds
          7000+k); family membership and trees of the real run kept; every family reclassified with the v4 rules
          (S1 no dominant event, S2 late peak, S3 chain-like tree) -> nulls/mshuf_swarm_ids.parquet (seed, ID)
Workers: --workers (default 5; the nearest-neighbour search forks them), nice 10 from run_q1.sh.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "06_analysis"))
import q1_common as Q  # noqa: E402

NUL = Q.BIG / "nulls"


def zbz_module(workers):
    import v51_clusters_zbz as Z
    Z.configure("v6.1")
    a = argparse.Namespace(rmin=0.5, rcap_min=10.0, workers=workers)
    c = Z.load_all()
    df = Z.prepare(c)
    ctab, _ = Z.completeness(df)
    return Z, a, c, df, ctab


def step_mcu(a_):
    Z, a, c, df, ctab = zbz_module(a_.workers)
    cu = ctab.copy()
    cu["Mc"] = Q.MC_U
    t0 = time.time()
    ev, fam, reg, thr = Z.build(df, cu, "main", a)
    out = NUL / "recluster_mcu"
    out.mkdir(parents=True, exist_ok=True)
    ev[["event_id", "region", "in_analysis", "M", "parent_id", "linked", "family_id", "family_type", "role",
        "cls"]].to_csv(out / "events.csv.gz", index=False, float_format="%.4f")
    fam.to_csv(out / "families.csv", index=False, float_format="%.4f")
    reg.to_csv(out / "thresholds.csv", index=False, float_format="%.4f")
    Q.log(f"mcu: {time.time() - t0:.0f} s, {len(fam)} families\n" + Z.summarize(ev, fam, reg).to_string())


def step_tshuf(a_):
    Z, a, c, df, ctab = zbz_module(a_.workers)
    reg = pd.read_csv(Q.CATS["v61"]["clu"] / "thresholds_zbz.csv")
    thr0 = dict(zip(reg.region, reg.log_eta0))
    f = NUL / "tshuf_swarm_ids.parquet"
    done = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["seed", "ID"])
    have = set(done.seed.unique())
    parts = [done]
    for k in range(a_.n_tshuf):
        if k in have:
            continue
        t0 = time.time()
        s = Z.build(df, ctab, "main", a, thr_override=thr0, shuffle_seed=5000 + k, null="region", null_families=True)
        fam = s["fam"]
        sw_f = fam.family_id[fam.type == "swarm"].values if len(fam) else np.array([], int)
        ids = s["event_id"][np.isin(s["family_id"], sw_f)]
        parts.append(pd.DataFrame(dict(seed=k, ID=ids.astype(np.int64))))
        if k % 10 == 9 or k == a_.n_tshuf - 1:
            pd.concat(parts, ignore_index=True).to_parquet(f, index=False)
        Q.log(f"tshuf {k}: {time.time() - t0:.0f} s, {len(fam)} families, {len(sw_f)} swarms, {len(ids)} swarm events")
    pd.concat(parts, ignore_index=True).to_parquet(f, index=False)


def step_mshuf(a_):
    from v4_clusters_zbz import classify
    z, _ = Q.load_zbz("v61")
    cat = pd.read_csv(Q.CATS["v61"]["csv"], usecols=["ID", "time"])
    t = pd.to_datetime(cat.set_index("ID").time, utc=True, format="ISO8601")
    A = z[z.in_analysis].copy()
    A["t"] = A.ID.map(t)
    A = A.sort_values("t", kind="stable").reset_index(drop=True)
    A["M_zbz"] = A.M_zbz.astype(float)
    # families of the real run: members in time order, local parent index, child counts, mean leaf depth
    fams = []
    for fid, g in A[A.family_id > 0].groupby("family_id"):
        ids = g.ID.values
        pos = {v: i for i, v in enumerate(ids)}
        n = len(ids)
        par = np.full(n, -1)
        depth = np.zeros(n, int)
        nchild = np.zeros(n, int)
        for i, (p, lk) in enumerate(zip(g.parent_id.values, g.linked.values)):
            if lk and p in pos:
                par[i] = pos[p]
                depth[i] = depth[pos[p]] + 1
                nchild[pos[p]] += 1
        leaves = nchild == 0
        fams.append((fid, g.index.values, nchild, float(depth[leaves].mean()) if leaves.any() else 0.0))
    reg_idx = {r: np.where(A.zbz_region.values == r)[0] for r in A.zbz_region.unique()}
    M0 = A.M_zbz.values
    rows = []
    for k in range(a_.n_mshuf):
        rng = np.random.default_rng(7000 + k)
        m = M0.copy()
        for r, ix in reg_idx.items():
            m[ix] = rng.permutation(M0[ix])
        sw = []
        for fid, ix, nchild, mld in fams:
            mm = m[ix]
            n = len(mm)
            ms = np.sort(mm)[::-1]
            imax = int(np.argmax(mm))
            tm = dict(mean_leaf_depth=mld, main_offspring_frac=nchild[imax] / max(n - 1, 1))
            typ, _ = classify(n, float(ms[0] - ms[1]), imax / n, tm)
            if typ == "swarm":
                sw.append(fid)
        ids = A.ID.values[np.isin(A.family_id.values, sw)]
        rows.append(pd.DataFrame(dict(seed=k, ID=ids.astype(np.int64))))
        if k % 20 == 0:
            Q.log(f"mshuf {k}: {len(sw)} swarm families of {len(fams)}")
    pd.concat(rows, ignore_index=True).to_parquet(NUL / "mshuf_swarm_ids.parquet", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="mcu,tshuf,mshuf")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--n-tshuf", type=int, default=100)
    ap.add_argument("--n-mshuf", type=int, default=200)
    a = ap.parse_args()
    NUL.mkdir(parents=True, exist_ok=True)
    for s in a.steps.split(","):
        {"mcu": step_mcu, "tshuf": step_tshuf, "mshuf": step_mshuf}[s](a)


if __name__ == "__main__":
    main()

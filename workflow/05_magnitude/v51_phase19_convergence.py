#!/usr/bin/env python3
"""
Convergence check of the phase19 'near' inversion (station ML = M_i + C_jp, LSQR with
damp 1e-3 and iter_lim 3000), on the exact pick set of the v5.1 run.

phase19 stops LSQR at 3000 iterations. This script re-solves the same system
  (a) with the same settings (3000 iterations): sensitivity to floating-point order
      (thread count), and
  (b) to convergence (iter_lim 50000),
and compares the final ML = M + anchor offset (median of ComCat ML - M over anchors with
>= 3 picks, as phase19) with the stored v5.1 values. Reads only; the production magnitudes
are not changed.

Writes <out>/phase19_convergence_summary.csv and <out>/phase19_convergence_by_region.csv.

Usage:
  OMP_NUM_THREADS=12 python v51_phase19_convergence.py --data /wd1/.../v5.1_analysis/mag \
      --out <worktree>/data/catalog_v5/v5.1_analysis/mag
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr


def solve(d, iter_lim):
    """Same system as phase19_preferred_magnitude.invert, returning the LSQR diagnostics."""
    e, ev_ids = pd.factorize(d.event_id)
    j, sp_ids = pd.factorize(d.sp)
    n, ne, ns = len(d), len(ev_ids), len(sp_ids)
    ph = np.array([k.split("|")[1] for k in sp_ids])
    rows, cols, vals, rhs = [np.arange(n)] * 2, [e, ne + j], [np.ones(n)] * 2, list(d.m)
    for i, p in enumerate(("P", "S")):
        sel = np.flatnonzero(ph == p)
        rows.append(np.full(len(sel), n + i))
        cols.append(ne + sel)
        vals.append(np.full(len(sel), 1000.0))
        rhs.append(0.0)
    A = sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n + 2, ne + ns)).tocsr()
    r = lsqr(A, np.asarray(rhs), damp=1e-3, atol=1e-8, btol=1e-8, iter_lim=iter_lim)
    x = r[0]
    return (pd.Series(x[:ne], index=ev_ids), pd.Series(x[ne:], index=sp_ids),
            dict(istop=int(r[1]), itn=int(r[2]), r1norm=float(r[3])))


def offset(M, near, anchors):
    g = near.groupby("event_id").size()
    a = anchors.join(M[g.reindex(M.index).fillna(0) >= 3].rename("m"), on="event_id", how="inner")
    return float(np.median(a.ml - a.m))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--v4", default=None,
                    help="v4 Route A run (default data/magnitude_v4 of the main checkout, see v51_paths.py)")
    a = ap.parse_args(argv)
    near = pd.read_parquet(f"{a.data}/phase19_near_picks_v51.parquet",
                           columns=["event_id", "sp", "m", "M_ev"])
    anchors = pd.read_csv(f"{a.data}/route_b_ml_anchors_v51.csv")[["event_id", "ml"]]
    mags = pd.read_csv(f"{a.data}/cascadia_catalog_v5.1_magnitudes.csv",
                       usecols=["ID", "region", "ML", "ML_method", "ml_caution"], low_memory=False)
    M0 = near.groupby("event_id").M_ev.first()
    off0 = offset(M0, near, anchors)
    n_ev = near.groupby("event_id").size()
    keep = n_ev[n_ev >= 3].index                  # events whose ML is 'near'
    rows, per = [], {}
    for name, lim in (("same settings (3000 iterations)", 3000), ("converged", 50000)):
        M, C, info = solve(near, lim)
        off = offset(M, near, anchors)
        d = (M + off - (M0 + off0)).reindex(keep)
        per[name] = d
        if name == "converged":
            Mconv, offconv = M, off
        rows.append(dict(run=name, **info, offset=off, offset_stored=off0, n_events=len(d),
                         median=d.median(), p05=d.quantile(.05), p95=d.quantile(.95),
                         share_abs_gt_0p1=float((d.abs() > 0.1).mean()),
                         share_abs_gt_0p3=float((d.abs() > 0.3).mean()), max_abs=float(d.abs().max()),
                         J35C_S=float(C.get("7D.J35C@2013-08-09|S", np.nan))))
        print(rows[-1], flush=True)
    s = pd.DataFrame(rows)
    s.to_csv(f"{a.out}/phase19_convergence_summary.csv", index=False, float_format="%.4f")
    dc = per["converged"].rename("dML").reset_index().rename(columns={"event_id": "ID"}).merge(mags, on="ID")
    dc["caution"] = dc.ml_caution.fillna("").ne("")
    t = dc.groupby("region").dML.agg(n="size", median="median", p05=lambda v: v.quantile(.05),
                                     p95=lambda v: v.quantile(.95),
                                     share_gt_0p1=lambda v: (v.abs() > 0.1).mean()).reset_index()
    t.to_csv(f"{a.out}/phase19_convergence_by_region.csv", index=False, float_format="%.3f")
    print(t.to_string(index=False))
    print("with ml_caution:", dc[dc.caution].dML.describe().round(3).to_dict())

    # outside references, stored vs converged: ComCat ML anchors and tensor Mw 3-4.5, by region
    reg = mags.set_index("ID").region
    ref = []
    an = anchors[anchors.event_id.isin(keep)]
    for lab, Mx, ox in (("stored", M0, off0), ("converged", Mconv, offconv)):
        r_ = (Mx.reindex(an.event_id).to_numpy() + ox) - an.ml.to_numpy()
        g = pd.Series(r_, index=an.event_id.map(reg).to_numpy())
        for k_, v in g.groupby(level=0):
            ref.append(dict(solution=lab, reference="ComCat ML anchors", region=k_, n=len(v), median=v.median()))
        ref.append(dict(solution=lab, reference="ComCat ML anchors", region="all", n=len(g), median=g.median(),
                        std=g.std()))
    mt = pd.read_csv(f"{a.data}/comcat_mt_matched_v51.csv")[["event_id", "Mw_mt"]]
    mt = mt[(mt.Mw_mt >= 3) & (mt.Mw_mt < 4.5) & mt.event_id.isin(keep)]
    for lab, Mx, ox in (("stored", M0, off0), ("converged", Mconv, offconv)):
        r_ = (Mx.reindex(mt.event_id).to_numpy() + ox) - mt.Mw_mt.to_numpy()
        lbl = mt.event_id.map(reg).isin(["Axial", "Endeavour", "Explorer", "JdF ridge/plate", "Blanco", "Gorda",
                                         "Mendocino", "Offshore forearc"]).to_numpy()
        for k_, sel in (("offshore and coastal labels", lbl), ("onshore labels", ~lbl), ("all", np.ones(len(mt), bool))):
            ref.append(dict(solution=lab, reference="tensor Mw 3-4.5 (ML - Mw)", region=k_, n=int(sel.sum()),
                            median=float(np.median(r_[sel])) if sel.any() else np.nan))
    ref = pd.DataFrame(ref)
    ref.to_csv(f"{a.out}/phase19_convergence_references.csv", index=False, float_format="%.3f")
    print(ref.pivot_table(index=["reference", "region"], columns="solution", values="median").round(3).to_string())

    # v5.1 - v4.1 with both inversions converged, against the stored (3000-iteration) change.
    # v4.1 pick set rebuilt as in v51_magnitude_report.decomposition (near_inversion on the v4 dataset).
    from phase18_moment_tensor_match import hutton_boore
    from v51_magnitude_table import near_inversion
    import v51_paths as vp
    V4 = a.v4 or vp.main_data("magnitude_v4")
    ds4 = pd.read_csv(f"{V4}/amp_distance_dataset_routeA.csv",
                      usecols=["arid", "event_id", "station", "phase", "dist_hypo_km", "log10A"])
    ds4["m"] = hutton_boore(ds4.log10A, ds4.dist_hypo_km)
    ds4["sp"] = ds4.station + "|" + ds4.phase
    near4, M4s, _ = near_inversion(ds4)
    an4 = pd.read_csv(f"{V4}/route_b_ml_anchors_routeA.csv")[["event_id", "ml"]]
    M4c, _, info4 = solve(near4, 50000)
    print("v4.1 converged:", info4, flush=True)
    M5c, _, _ = solve(near, 50000)
    n4 = near4.groupby("event_id").size()
    k = keep.intersection(n4[n4 >= 3].index)
    ml5s, ml4s = M0 + off0, M4s + offset(M4s, near4, an4)
    ml5c, ml4c = M5c + offset(M5c, near, anchors), M4c + offset(M4c, near4, an4)
    x = pd.DataFrame({"stored": (ml5s - ml4s).reindex(k), "converged": (ml5c - ml4c).reindex(k)})
    x = x.rename_axis("ID").reset_index().merge(mags[["ID", "region"]], on="ID")
    t2 = x.groupby("region")[["stored", "converged"]].median()
    t2.loc["all"] = x[["stored", "converged"]].median()
    t2.insert(0, "n", x.groupby("region").size().reindex(t2.index).fillna(len(x)).astype(int))
    t2.to_csv(f"{a.out}/phase19_convergence_v51_minus_v41.csv", float_format="%.3f")
    print("ML(v5.1) - ML(v4.1), stored (3000 iterations) vs both converged:")
    print(t2.round(3).to_string())


if __name__ == "__main__":
    main()

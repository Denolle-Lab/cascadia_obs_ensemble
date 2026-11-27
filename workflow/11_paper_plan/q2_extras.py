#!/usr/bin/env python3
"""Q2 extras on the main configuration:
* C3 (registered, second part): Spearman rho between the unit residual (D90 - primary level) and the unit median
  distance to the nearest station, block bootstrap (1 deg blocks, 2000 draws); flagged if |rho| >= 0.3 and the 90 %
  interval excludes 0.
* Reported covariates (registered, not in the decision): heat flow (GHFDB R2024, median of the measurements within
  25 km of the unit centre) and the linear-conductive 600 C depth k 600 / q (k 3.138 W/m/K); sediment thickness.
* Monte Carlo shift: median of the D90 draws minus the observed D90 per unit.
* Not registered (exploratory, labelled so in the report): rho within the Blanco cluster (north of 42.5 N) and the
  Mendocino-Gorda cluster (south of 42.5 N) of the oceanic frame.
Writes <Q2_OUT>/c3_geometry.csv, covariates_units.csv, mc_shift.csv, within_cluster.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb_common as P  # noqa: E402
import q2_analysis as A  # noqa: E402
import q2_common as Q  # noqa: E402

K_LITH = 3.138


def rho(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return np.nan
    a, b = rankdata(x[ok]), rankdata(y[ok])
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan


def block_boot(x, y, blocks, rng, n=2000):
    ub = np.unique(blocks)
    idx = [np.where(blocks == b)[0] for b in ub]
    out = []
    for _ in range(n):
        ii = np.concatenate([idx[k] for k in rng.integers(0, len(ub), len(ub))])
        out.append(rho(x[ii], y[ii]))
    out = np.array(out, float)
    return np.nanpercentile(out, [5, 95]) if np.isfinite(out).any() else (np.nan, np.nan)


def main():
    rng = np.random.default_rng(Q.SEED + 7)
    U = pd.read_csv(Q.OUT / "units_main.csv")
    gs = A.group_sets(U)
    rows = []
    for g, msk in gs.items():
        for L in (A.PRIMARY_T, A.PRIMARY_C, "GDH1_600", "V8.0"):
            u = U[msk]
            r = (u.D - u[L]).values
            ok = np.isfinite(r) & np.isfinite(u.dist_min_med.values)
            row = dict(group=g, level=L, n_units=int(ok.sum()))
            if ok.sum() >= 5:
                rr = rho(r[ok], u.dist_min_med.values[ok])
                lo, hi = block_boot(r[ok], u.dist_min_med.values[ok], u.block.values[ok], rng)
                row.update(rho=rr, lo90=lo, hi90=hi, flagged=bool(abs(rr) >= 0.3 and (lo > 0 or hi < 0)))
            rows.append(row)
    pd.DataFrame(rows).to_csv(Q.OUT / "c3_geometry.csv", index=False, float_format="%.4g")
    # covariates
    hf = Q.heat_flow_table()
    cov = U[["grouping", "key", "bx", "by", "lon", "lat", "n", "D", "D_bsf", "age", "sed_km", "ttype"]].copy()
    q_med, q_n = [], []
    for lo, la in zip(cov.lon, cov.lat):
        d = P.hav(la, lo, hf.lat.values, hf.lon.values)
        v = hf.q.values[(d <= 25.0) & (hf.q.values > 0)]
        q_med.append(float(np.median(v)) if len(v) else np.nan)
        q_n.append(int(len(v)))
    cov["q_mWm2"] = q_med
    cov["q_n"] = q_n
    cov["z600_linear_bsf"] = K_LITH * 600.0 / (cov.q_mWm2 / 1000.0) / 1000.0
    cov["D_bsf_minus_z600_linear"] = cov.D_bsf - cov.z600_linear_bsf
    cov.to_csv(Q.OUT / "covariates_units.csv", index=False, float_format="%.4g")
    # Monte Carlo shift
    dr = np.load(Q.BIG / "draws_main.npz")
    sh = np.median(dr["D"], axis=1) - U.D.values
    pd.DataFrame({"grouping": U.grouping, "key": U.key, "bx": U.bx, "by": U.by, "n": U.n, "sig_z_med": U.sig_z_med,
                  "D": U.D, "D_draw_median": np.median(dr["D"], axis=1), "shift": sh}).to_csv(
        Q.OUT / "mc_shift.csv", index=False, float_format="%.4g")
    # exploratory: within-cluster rho in the oceanic frame
    rows = []
    F = U[U.grouping.eq("frame") & U.key.eq("oceanic")]
    for name, m in (("Blanco cluster (oceanic, lat >= 42.5)", F.lat >= 42.5),
                    ("Mendocino-Gorda cluster (oceanic, lat < 42.5)", F.lat < 42.5),
                    ("oceanic frame, all", F.lat > 0)):
        f = F[m]
        for L in (A.PRIMARY_T, "GDH1_600", "HSC_800", A.PRIMARY_C):
            ok = np.isfinite(f[L].values)
            rows.append(dict(cluster=name, level=L, n_units=int(ok.sum()),
                             rho=rho(f.D.values[ok], f[L].values[ok]) if ok.sum() >= 4 else np.nan,
                             bias=float(np.median(f.D.values[ok] - f[L].values[ok])) if ok.any() else np.nan,
                             D_range=float(np.ptp(f.D.values[ok])) if ok.any() else np.nan,
                             level_range=float(np.ptp(f[L].values[ok])) if ok.any() else np.nan))
    pd.DataFrame(rows).to_csv(Q.OUT / "within_cluster.csv", index=False, float_format="%.4g")
    Q.log(pd.read_csv(Q.OUT / "c3_geometry.csv").dropna(subset=["rho"]).to_string())
    Q.log(pd.read_csv(Q.OUT / "within_cluster.csv").to_string())
    Q.log(f"MC shift median {np.median(sh):.2f} km, p90 {np.percentile(sh, 90):.2f} km")


if __name__ == "__main__":
    main()

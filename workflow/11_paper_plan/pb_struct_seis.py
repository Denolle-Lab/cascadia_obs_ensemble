#!/usr/bin/env python3
"""Two structure-versus-seismicity checks for the figure plan (D-B107), as feasibility numbers, not results.

(a) Depth extent against the 7.5 and 8 km/s levels, per model column: columns (0.125 x 0.1 deg) of a segment with
    >= 20 depth-good tier A-C events and a resolved level; D90 (90th percentile of event depth, km bsl) against the
    level depth. Per segment: number of columns, median D90 - level, Spearman rho, and the spread sd(D90 - level)
    against a null that permutes the level depths among the segment's columns (1000 draws; p = share of null spreads
    <= the real one: small p = the seismicity bottom follows the level more closely than chance).
(b) Swarm against mainshock-aftershock families: each v6.0 family (zbz) with >= 5 depth-good members is placed at
    the node nearest to the median position of those members; Vp/Vs (P and S mask) and Vp (P mask) there. Per
    segment with >= 5 families of each type on resolved nodes: median swarm - median MA, a null that permutes the type
    labels among families of the same depth layer (2000 draws), and the member spread of the same difference (100
    ensemble members, x calibration factor).

Writes data/paper_plan/plate_boundaries/depth_vs_level_columns.csv, depth_vs_level.csv, families_on_nodes.csv,
swarm_vs_ma.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb_common as P  # noqa: E402

RNG = np.random.default_rng(P.SEED + 7)


def main():
    m = P.Model()
    c = pd.read_parquet(P.BIG / "events_segments_v60.parquet")
    E = c[c.is_eq & c.abc & ~c.review_P1 & c.depth_good].copy()
    E["iy"], E["ix"] = m.column_index(E.lon.values, E.lat.values)
    # (a)
    rows, crow = [], []
    for seg, g in E.groupby("seg"):
        col = g.groupby(["iy", "ix"]).agg(n=("ID", "size"), d90=("depth", lambda s: np.percentile(s, 90)),
                                         d50=("depth", "median"), z75=("z75", "first"), z8=("z8", "first"))
        col = col[col.n >= 20].reset_index()
        col["seg"] = seg
        crow.append(col)
        for tag in ("z75", "z8"):
            ok = col[np.isfinite(col[tag])]
            r = dict(seg=seg, level=tag, n_columns=len(ok))
            if len(ok) >= 5:
                dd = ok.d90 - ok[tag]
                real = float(np.std(dd))
                null = np.array([np.std(ok.d90.values - RNG.permutation(ok[tag].values)) for _ in range(1000)])
                r.update(median_d90_minus_level=float(np.median(dd)), sd_real=real,
                         sd_null_med=float(np.median(null)), p_smaller=float((1 + (null <= real).sum()) / 1001),
                         rho=float(spearmanr(ok.d90, ok[tag]).statistic), sd_d90=float(ok.d90.std()),
                         sd_level=float(ok[tag].std()))
            rows.append(r)
    pd.concat(crow).to_csv(P.OUT / "depth_vs_level_columns.csv", index=False, float_format="%.4g")
    dl = pd.DataFrame(rows)
    dl.to_csv(P.OUT / "depth_vs_level.csv", index=False, float_format="%.4g")
    P.log(dl.dropna(subset=["sd_real"]).to_string())
    # (b)
    z = pd.read_csv(P.CLU / "events_zbz_v60.csv.gz", usecols=["event_id", "family_id"])
    fam = pd.read_csv(P.CLU / "families_zbz_v60.csv", usecols=["family_id", "type", "n"])
    g = E.merge(z.rename(columns={"event_id": "ID"}), on="ID")
    g = g[g.family_id >= 0]
    f = g.groupby("family_id").agg(nd=("ID", "size"), lon=("lon", "median"), lat=("lat", "median"),
                                   depth=("depth", "median"),
                                   seg=("seg", lambda s: s.value_counts().index[0])).reset_index()
    f = f[f.nd >= 5].merge(fam, on="family_id")
    iy, ix = m.column_index(f.lon.values, f.lat.values)
    iz = np.clip(np.searchsorted(0.5 * (m.dep[1:] + m.dep[:-1]), f.depth.values), 0, len(m.dep) - 1)
    f["iz"], f["iy"], f["ix"] = iz, iy, ix
    f["inP"], f["inPS"] = m.mP[iz, iy, ix], m.mPS[iz, iy, ix]
    f["Vp"], f["VpVs"] = m.Vp[iz, iy, ix], m.VpVs[iz, iy, ix]
    f.to_csv(P.OUT / "families_on_nodes.csv", index=False, float_format="%.5g")
    names, mv = P.load_members()
    out = []
    for seg, h in f.groupby("seg"):
        for var, mk in (("VpVs", "inPS"), ("Vp", "inP")):
            k = h[h[mk]]
            sw, ma = k[k.type == "swarm"], k[k.type == "mainshock-aftershock"]
            r = dict(seg=seg, var=var, n_swarm=len(sw), n_MA=len(ma), n_multiplet=int((k.type == "multiplet").sum()),
                     n_families_all=len(h))
            if len(sw) >= 5 and len(ma) >= 5:
                real = float(np.median(sw[var]) - np.median(ma[var]))
                kk = k[k.type.isin(["swarm", "mainshock-aftershock"])].reset_index(drop=True)
                lab = kk.type.values
                null = []
                for _ in range(2000):
                    perm = lab.copy()
                    for lay in np.unique(kk.iz):
                        s_ = np.where(kk.iz.values == lay)[0]
                        perm[s_] = RNG.permutation(perm[s_])
                    null.append(np.median(kk[var].values[perm == "swarm"]) -
                                np.median(kk[var].values[perm == "mainshock-aftershock"]))
                null = np.array(null)
                arr = mv[var][:, kk.iz.values, kk.iy.values, kk.ix.values]           # [member, family]
                dm = np.array([np.median(a[lab == "swarm"]) - np.median(a[lab == "mainshock-aftershock"])
                               for a in arr])
                r.update(diff=real, null_lo=float(np.percentile(null, 2.5)), null_hi=float(np.percentile(null, 97.5)),
                         p_two=float((1 + (np.abs(null) >= abs(real)).sum()) / 2001),
                         sigma_members=float(np.std(dm) * P.INFL[var]))
            out.append(r)
    sv = pd.DataFrame(out)
    sv.to_csv(P.OUT / "swarm_vs_ma.csv", index=False, float_format="%.4g")
    P.log(sv.to_string())


if __name__ == "__main__":
    main()

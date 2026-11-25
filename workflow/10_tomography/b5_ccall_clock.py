"""Stage 5 of the all-family CC differential times (D-B093): station clock screen on every accepted CC row.

Nothing here is applied to any output; candidates go to the PI (as D-B090).

Method. For an accepted row of pair (1, 2) at station X, dt_cc = (T1 - T2) + (moveout of X) + c_X(t1) - c_X(t2),
with T the origin times and c_X the clock error of X. In pairs whose accepted rows agree across stations (median
absolute deviation from the pair median below SPREAD_S, >= NMIN rows at >= NSTA stations: effectively
co-located events), the pair median removes T1 - T2 and the residual r = dt_cc - median is c_X(t1) - c_X(t2)
plus noise. Per station, a monthly clock term o_m is fitted to r = o(month of t1) - o(month of t2) by Huber IRLS
(sparse least squares, weak ridge; the station's row-weighted median month is the reference 0). A month is a
candidate when |o_m| >= 0.05 s with >= 10 informative rows (the other event in another month); the P and S terms
are fitted separately too (a clock moves both alike).

Run on the header-base CC dt (dt_cc_hdr: what the archive's record times give, the D-B090 MH07/MG07 drifts
included) and on the clock-corrected dt (dt_cc). The two known drifts must appear in the first and vanish in the
second. Candidates are compared with the pick-residual screen of tools/pick_timing (detect_full/candidates.csv).

  python b5_ccall_clock.py [--accepted <dir of accepted parquet files>]
Output: qc/clock_screen_cc_months.csv, qc/clock_candidates_cc.csv, qc/clock_screen_cc.json
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b5_ccall_common as A  # noqa: E402

SPREAD_S, NMIN, NSTA = 0.015, 6, 4
OFF_MIN, NINF_MIN = 0.05, 10
PT_CAND = Path("/wd1/mdenolle_data/tomo_v1/checks/pick_timing_v6/detect_full/candidates.csv")


def colocated(a, col):
    g = a.groupby("pair")[col]
    med = g.transform("median")
    dev = (a[col] - med).abs()
    spread = dev.groupby(a.pair).transform("median")
    n = g.transform("size")
    nsta = a.groupby("pair").station.transform("nunique")
    m = (n >= NMIN) & (nsta >= NSTA) & (spread < SPREAD_S)
    return a[m].assign(r=(a[col] - med)[m])


def fit_months(x, ridge=1e-3, iters=8, huber=0.03):
    """r = o[m1] - o[m2]; returns DataFrame month, o, n_inf (rows with the other event in another month)."""
    x = x[x.m1 != x.m2]
    if len(x) < NINF_MIN:
        return None
    months = np.unique(np.r_[x.m1.values, x.m2.values])
    k = {m: i for i, m in enumerate(months)}
    i1 = np.array([k[m] for m in x.m1.values])
    i2 = np.array([k[m] for m in x.m2.values])
    n = len(x)
    rows = np.r_[np.arange(n), np.arange(n)]
    G = sp.csr_matrix((np.r_[np.ones(n), -np.ones(n)], (rows, np.r_[i1, i2])), shape=(n, len(months)))
    y = x.r.values
    w = np.ones(n)
    for _ in range(iters):
        sw = np.sqrt(w)
        Gw = sp.diags(sw) @ G
        o = lsqr(Gw, y * sw, damp=np.sqrt(ridge), atol=1e-10, btol=1e-10)[0]
        res = y - G @ o
        w = 1.0 / np.maximum(1.0, np.abs(res) / huber)
    ninf = np.bincount(i1, minlength=len(months)) + np.bincount(i2, minlength=len(months))
    o = o - np.average(o, weights=ninf) if ninf.sum() else o
    ref = np.median(np.repeat(o, ninf))      # row-weighted median month = reference
    return pd.DataFrame({"month": months, "o": o - ref, "n_inf": ninf})


def screen(a, col):
    c = colocated(a, col)
    c = c.assign(m1=pd.to_datetime(c.t_pick1, unit="s").dt.strftime("%Y-%m"),
                 m2=pd.to_datetime(c.t_pick2, unit="s").dt.strftime("%Y-%m"))
    out = []
    for st, x in c.groupby("station"):
        f = fit_months(x)
        if f is None:
            continue
        f["station"] = st
        for ph in ("P", "S"):
            fp = fit_months(x[x.phase == ph])
            f[f"o_{ph}"] = f.month.map(fp.set_index("month").o) if fp is not None else np.nan
            f[f"n_inf_{ph}"] = f.month.map(fp.set_index("month").n_inf) if fp is not None else 0
        f["rows_station"] = len(x)
        out.append(f)
    months = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    return c, months


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accepted", default=str(A.WORK / "accepted"))
    ap.add_argument("--out", default=str(A.QC))
    a_ = ap.parse_args()
    t0 = time.time()
    out = Path(a_.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(Path(a_.accepted) / "*.parquet")))
    a = pd.concat([pd.read_parquet(f).assign(block=Path(f).stem) for f in files], ignore_index=True)
    a["pair"] = a.block + ":" + a.pair.astype(str)
    S = dict(accepted_rows=len(a), blocks=[Path(f).stem for f in files], spread_s=SPREAD_S, nmin=NMIN, nsta=NSTA,
             off_min_s=OFF_MIN, ninf_min=NINF_MIN)
    res = {}
    for col in ("dt_cc_hdr", "dt_cc"):
        c, mo = screen(a, col)
        mo["base"] = col
        res[col] = mo
        S[col] = dict(colocated_rows=int(len(c)), colocated_pairs=int(c.pair.nunique()),
                      stations_screened=int(mo.station.nunique()) if len(mo) else 0,
                      station_months=int(len(mo)))
    M = pd.concat(res.values(), ignore_index=True)
    M.to_csv(out / "clock_screen_cc_months.csv", index=False, float_format="%.4f")
    cand = M[(M.o.abs() >= OFF_MIN) & (M.n_inf >= NINF_MIN)].copy()
    same_sign = np.sign(cand.o_P) == np.sign(cand.o_S)
    cand["P_S_agree"] = np.where(cand.o_P.notna() & cand.o_S.notna(), same_sign, pd.NA)
    known = set(A.clock_model())
    cand["clock_like"] = (cand.o_P.notna() & cand.o_S.notna() & (np.sign(cand.o_P) == np.sign(cand.o_S))
                          & (np.minimum(cand.o_P.abs(), cand.o_S.abs()) >= 0.5 * OFF_MIN)
                          & (cand.n_inf_P >= 3) & (cand.n_inf_S >= 3))
    cand["known_D_B090"] = cand.station.isin(known)
    # pick-residual candidates (tools/pick_timing detect_full) at the same station overlapping the month
    if PT_CAND.exists():
        pt = pd.read_csv(PT_CAND)
        tcol = [c for c in pt.columns if "start" in c.lower()]
        ecol = [c for c in pt.columns if "end" in c.lower()]
        scol = "station" if "station" in pt.columns else pt.columns[0]
        hits = []
        for r in cand.itertuples():
            m0 = pd.Timestamp(r.month + "-01", tz="UTC")
            m1 = m0 + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1)
            x = pt[pt[scol] == r.station]
            h = []
            for q in x.itertuples():
                ts = pd.Timestamp(getattr(q, tcol[0])) if tcol else None
                te = pd.Timestamp(getattr(q, ecol[0])) if ecol and pd.notna(getattr(q, ecol[0])) else None
                ts = ts.tz_localize("UTC") if ts is not None and ts.tzinfo is None else ts
                te = te.tz_localize("UTC") if te is not None and te.tzinfo is None else te
                if ts is not None and ts < m1 and (te is None or te > m0):
                    h.append(str(getattr(q, tcol[0]))[:10])
            hits.append(";".join(h))
        cand["pick_timing_candidate_start"] = hits
    cand = cand.sort_values(["base", "station", "month"])
    cand.to_csv(out / "clock_candidates_cc.csv", index=False, float_format="%.4f")
    for col in ("dt_cc_hdr", "dt_cc"):
        cc = cand[cand.base == col]
        S[col]["candidate_months"] = int(len(cc))
        S[col]["candidate_stations"] = sorted(cc.station.unique().tolist())
        S[col]["new_candidate_stations"] = sorted(cc[~cc.known_D_B090].station.unique().tolist())
        S[col]["new_clock_like_stations"] = sorted(cc[~cc.known_D_B090 & cc.clock_like].station.unique().tolist())
    k = M[M.station.isin(known)]
    S["known_drifts"] = {f"{st}:{b}": g.loc[g.o.abs().idxmax(), ["month", "o", "n_inf"]].to_dict() if len(g) else {}
                         for (st, b), g in k.groupby(["station", "base"])}
    (out / "clock_screen_cc.json").write_text(json.dumps(S, indent=1, default=str))
    print(json.dumps(S, indent=1, default=str))
    A.write_state("clock", done=True, **{f"{c}_candidates": S[c].get("candidate_months", 0)
                                         for c in ("dt_cc_hdr", "dt_cc")}, wall_s=round(time.time() - t0))


if __name__ == "__main__":
    main()

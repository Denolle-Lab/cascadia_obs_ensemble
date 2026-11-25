"""Parameter study for the CC differential times (filter band, windows, max lag, CC threshold), per set.

Runs b5_cc_measure on all pairs of a set (given + added) for a list of configurations and scores each one with
data of the set itself:
* closure: for every triangle of pairs (A,B), (B,C), (A,C) measured at the same station-phase,
  e = dt_AB + dt_BC - dt_AC. Catalog dt close exactly (same picks), so e is the CC error alone (sum of three);
  reported against the smallest CC of the three (MAD and fraction |e| > 20 ms, cycle skips);
* agreement with the catalog: dt_cc - dt_cat (pick errors of both events + CC error): MAD and fraction > 0.2 s,
  by CC bin;
* lag at high CC (cc >= 0.9): quantiles of |lag_s| -> needed max lag;
* count of measurements and pairs with >= 4 / >= 8 CC dt above a threshold.

  python b5_cc_study.py --set jdf --grid bands [--workers 4]
  python b5_cc_study.py --set jdf --grid windows --band 3,15
  python b5_cc_study.py --set jdf --grid lags --band 3,15      (max lag x S-P fraction, chosen windows)
  python b5_cc_study.py --set jdf --report      (tables + figures from the saved runs)
Output: /wd1/mdenolle_data/tomo_v1/cc_dt/study/<set>/<tag>.parquet, study_<set>.csv, figures.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b5_cc_common as C  # noqa: E402
import b5_cc_measure as M  # noqa: E402

KEEP = ["pair", "ID1", "ID2", "source", "sep_km", "station", "phase", "timing1", "timing2", "dt_cat", "status",
        "comps", "rate", "win_pre", "win_post", "maxlag_s", "cc_max", "lag_s", "dt_cc", "cc_2nd", "edge", "snr1", "snr2"]
BANDS = [[1, 8], [2, 12], [3, 15], [4, 20], [6, 30], [10, 45]]
CC_BINS = [0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.01]


def tag_of(cfg):
    return (f"b{cfg['band'][0]:g}-{cfg['band'][1]:g}_p{cfg['p_pre']:g}-{cfg['p_post']:g}"
            f"_s{cfg['s_pre']:g}-{cfg['s_post']:g}_l{cfg['p_maxlag']:g}-{cfg['s_maxlag']:g}"
            + (f"_f{cfg['lag_sp_frac']:g}" if cfg.get("lag_sp_frac", 0.5) != 0.5 else ""))


def closure(d):
    """Triangle closure errors per station-phase among the rows of d (already filtered: status ok, accepted):
    e = dt_AB + dt_BC - dt_AC with A < B < C (dt of (Y, X) = -dt of (X, Y)). Catalog dt close exactly, so e is
    the CC error of the three measurements. Returns DataFrame A, B, Cc, station, phase, e, ccmin, maxlag."""
    a = d[["ID1", "ID2", "station", "phase", "dt_cc", "cc_max", "lag_s"]].copy()
    b = a.rename(columns={"ID1": "ID2", "ID2": "ID1"}).copy()
    b["dt_cc"] = -b.dt_cc
    u = pd.concat([a, b], ignore_index=True)
    ab = u[u.ID1 < u.ID2]
    t = ab.merge(ab, left_on=["ID2", "station", "phase"], right_on=["ID1", "station", "phase"], suffixes=("_ab", "_bc"))
    t = t.rename(columns={"ID1_ab": "A", "ID2_ab": "B", "ID2_bc": "Cc"})
    ac = ab.rename(columns={"ID1": "A", "ID2": "Cc", "dt_cc": "dt_ac", "cc_max": "cc_ac", "lag_s": "lag_ac"})
    t = t.merge(ac[["A", "Cc", "station", "phase", "dt_ac", "cc_ac", "lag_ac"]], on=["A", "Cc", "station", "phase"])
    t["e"] = t.dt_cc_ab + t.dt_cc_bc - t.dt_ac
    t["ccmin"] = np.minimum(np.minimum(t.cc_max_ab, t.cc_max_bc), t.cc_ac)
    t["maxlag"] = np.maximum(np.maximum(t.lag_s_ab.abs(), t.lag_s_bc.abs()), t.lag_ac.abs())
    return t[["A", "B", "Cc", "station", "phase", "e", "ccmin", "maxlag"]]


def mad(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return float(np.median(np.abs(x - np.median(x)))) if len(x) else np.nan


def accepted(df, cc_min, margin_min):
    ok = (df.status == "ok") & ~df.edge.astype("boolean").fillna(False).astype(bool)
    return ok & (df.cc_max >= cc_min) & ((df.cc_max - df.cc_2nd) >= margin_min)


def score(df, tag, thresholds=((0.6, 0.1), (0.7, 0.0), (0.7, 0.1), (0.8, 0.0), (0.8, 0.1), (0.9, 0.0))):
    rows = []
    okall = df[(df.status == "ok")]
    for lo, mg in thresholds:
        acc = df[accepted(df, lo, mg)]
        cl = closure(acc)
        for ph in ("P", "S", "all"):
            s = acc if ph == "all" else acc[acc.phase == ph]
            o = okall if ph == "all" else okall[okall.phase == ph]
            cs = cl if ph == "all" else cl[cl.phase == ph]
            r = s.dt_cc - s.dt_cat
            npair = s.groupby("pair").size()
            rows.append(dict(tag=tag, phase=ph, cc_min=lo, margin_min=mg, n=len(s), frac_of_ok=len(s) / max(len(o), 1),
                             pairs_ge4=int((npair >= 4).sum()), pairs_ge8=int((npair >= 8).sum()),
                             diff_med_ms=float(np.median(r) * 1e3) if len(r) else np.nan,
                             diff_mad_ms=mad(r) * 1e3, diff_gt200=float((r.abs() > 0.2).mean()) if len(r) else np.nan,
                             n_tri=len(cs), clos_mad_ms=mad(cs.e) * 1e3,
                             clos_gt10=float((cs.e.abs() > 0.01).mean()) if len(cs) else np.nan,
                             clos_gt20=float((cs.e.abs() > 0.02).mean()) if len(cs) else np.nan,
                             lag_q95_ms=float(np.quantile(s.lag_s.abs(), 0.95) * 1e3) if len(s) else np.nan,
                             lag_q99_ms=float(np.quantile(s.lag_s.abs(), 0.99) * 1e3) if len(s) else np.nan,
                             cc_median_ok=float(o.cc_max.median())))
    return pd.DataFrame(rows)


def run_grid(name, grid, band, workers):
    out = C.OUT / "study" / name
    out.mkdir(parents=True, exist_ok=True)
    obs = M.load_obs(name, "all")
    cfgs = []
    if grid == "bands":
        for b in BANDS:
            cfgs.append({**M.DEFAULT, "band": b, "p_maxlag": 1.0, "s_maxlag": 1.0})
    elif grid == "windows":
        for pp, sp_ in (((0.2, 0.6), (0.3, 1.0)), ((0.2, 1.0), (0.3, 1.5)), ((0.3, 1.5), (0.5, 2.5)),
                        ((0.1, 0.5), (0.2, 0.8))):
            cfgs.append({**M.DEFAULT, "band": band, "p_pre": pp[0], "p_post": pp[1], "s_pre": sp_[0],
                         "s_post": sp_[1], "p_maxlag": 1.0, "s_maxlag": 1.0})
    elif grid == "lags":
        for ml, fr in ((1.0, 0.5), (0.4, 0.5), (0.4, 0.3), (0.3, 0.3)):
            cfgs.append({**M.DEFAULT, "band": band, "p_pre": 0.2, "p_post": 0.6, "s_pre": 0.3, "s_post": 1.0,
                         "p_maxlag": ml, "s_maxlag": ml, "lag_sp_frac": fr})
    for cfg in cfgs:
        tag = tag_of(cfg)
        f = out / f"{tag}.parquet"
        if f.exists():
            continue
        t = time.time()
        r = M.run(name, obs, cfg, workers)
        r[KEEP].to_parquet(f, index=False)
        (out / f"{tag}.json").write_text(json.dumps(cfg))
        print(f"{time.strftime('%H:%M:%S')} {name} {tag}: {len(r)} rows {time.time() - t:.0f} s "
              f"{r.status.value_counts().to_dict()}", flush=True)


def report(name):
    out = C.OUT / "study" / name
    tabs = []
    for f in sorted(out.glob("b*.parquet")):
        df = pd.read_parquet(f)
        tabs.append(score(df, f.stem))
    tab = pd.concat(tabs, ignore_index=True)
    tab.to_csv(C.OUT / "study" / f"study_{name}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_columns", 30):
        print(tab[(tab.margin_min == 0.1) | (tab.cc_min == 0.9)].round(3).to_string(index=False))
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--grid", default="")
    ap.add_argument("--band", default="2,12")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.grid:
        run_grid(a.set, a.grid, [float(x) for x in a.band.split(",")], a.workers)
    if a.report:
        report(a.set)


if __name__ == "__main__":
    main()

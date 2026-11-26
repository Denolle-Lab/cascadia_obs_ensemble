"""Station clock epochs for the CC rows (follow-up of the MH07.XD / MG07.XD offsets found on repeating pairs).

Data: NLLoc phase residuals of the v5.1 relocation (b5_cc_clock_res.py -> clock/nll_residuals.parquet) and MiniSEED
record headers (b5_cc_clock_hdr.py -> clock/hdr_<STA>.csv, hdr_<STA>_tears.csv).
* Event-demeaned residual rd = Res - median Res of the event over all its stations (removes the event's common
  origin-time / location term). A station clock error c(t) adds c to that station's P and S picks.
* screen: monthly median rd per station and phase for the stations of the CC sets (events of the MSH tiles for msh
  stations, all parsed events otherwise); flags months with |median| > 0.1 s in P and S with the same sign.
* fit: for a station with a free-running clock (no timing flags in the XD headers), c(t) = b + r * (t - ts) for
  ts <= t < te, b outside (b = reference, set to 0): te = the data restart that ends the drift (headers), ts
  fitted on rd (grid over ts, Huber IRLS, S weight 0.5, 300 event bootstraps); r then refitted on the CC rows
  (cc_rate: station dt_cc minus the median of the other stations of the pair), because the v5.1 locations absorb
  part of the clock error and the NLLoc residuals give a smaller rate (kept as rate_nll_s_per_day).
* epochs: reference epoch (offset 0, id 0, before ts and after te), and the drift split into sub-epochs over which
  the model offset changes by <= 20 ms (ids 1..n) -> qc/clock_epochs.csv.
* apply: per CC row at a station with epochs: clock_epoch1/2, clock_c1/2 (model offset at each event, s),
  clock_dt_s = c1 - c2 (what the clock adds to dt_cat and dt_cc), straddles_clock_epoch = epoch ids differ.
  Rows at other stations: epoch 0, clock_dt_s 0, False.

  python b5_cc_clock.py --screen
  python b5_cc_clock.py --fit MH07.XD:2016-05-12T05:09:09,MG07.XD:2016-05-08T07:57:28 --apply
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b5_cc_common as C  # noqa: E402

CLK = C.OUT / "clock"
QC = C.OUT / "qc"
STEP = 0.02     # s of drift per sub-epoch
CLOCK_STATIONS = ["MH07.XD", "MG07.XD"]


def residuals():
    r = pd.read_parquet(CLK / "nll_residuals.parquet")
    r = r[r.weight > 0].copy()
    r["rd"] = r.res - r.groupby("hyp").res.transform("median")
    r["t"] = pd.to_datetime(r.t_pick, unit="s")
    return r


def screen():
    r = residuals()
    sts = set()
    for s in C.SETS:
        sts |= set(pd.read_parquet(C.OUT / f"cc_{s}.parquet", columns=["station"]).station.unique())
    msh = set(pd.read_parquet(C.OUT / "cc_msh.parquet", columns=["station"]).station.unique())
    out = []
    for st in sorted(sts):
        x = r[r.station == st]
        if st in msh:
            x = x[x.tile.str.startswith("t31_33")]
        if len(x) < 20:
            out.append(dict(station=st, n=len(x), months=0, flagged_months=0, flagged=""))
            continue
        x = x.assign(m=x.t.dt.to_period("M"))
        g = x.groupby(["m", "phase"]).rd.agg(["size", "median"]).unstack("phase")
        g.columns = [f"{a}_{b}" for a, b in g.columns]
        if "median_S" not in g:
            g["median_S"], g["size_S"] = np.nan, 0
        f = g[(g.get("size_P", 0) >= 8) & (g.get("size_S", 0) >= 8) &
              (((g.median_P > 0.1) & (g.median_S > 0.1)) | ((g.median_P < -0.1) & (g.median_S < -0.1)))]
        out.append(dict(station=st, n=len(x), months=len(g), flagged_months=len(f),
                        flagged=" ".join(f"{m}:{p:+.2f}/{s:+.2f}" for m, p, s in zip(f.index.astype(str), f.median_P, f.median_S))))
    o = pd.DataFrame(out).sort_values("flagged_months", ascending=False)
    o.to_csv(QC / "clock_screen.csv", index=False)
    print(o[o.flagged_months > 0].to_string(index=False))
    return o


def fit_station(r, st, te, t_lo, t_hi, nboot=300, seed=0):
    """Ramp fit: c = b + r (t - ts) on [ts, te), b elsewhere, data in [t_lo, t_hi)."""
    x = r[(r.station == st) & (r.t >= t_lo) & (r.t < t_hi)].copy()
    x["d"] = (x.t - pd.Timestamp(te)).dt.total_seconds() / 86400.0          # days relative to te (negative before)
    x["w0"] = np.where(x.phase == "P", 1.0, 0.5)
    ts_grid = np.arange(-160, -5, 0.25)

    def one(xx):
        best = None
        y, w0, d = xx.rd.values, xx.w0.values, xx.d.values
        for ts in ts_grid:
            ramp = np.where((d >= ts) & (d < 0), d - ts, 0.0)
            A = np.c_[np.ones_like(d), ramp]
            w = w0.copy()
            for _ in range(6):
                sw = np.sqrt(w)
                p, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
                res = y - A @ p
                w = w0 / np.maximum(1, np.abs(res) / 0.1)
            cost = np.sum(w0 * np.where(np.abs(res) <= 0.1, 0.5 * res ** 2, 0.1 * np.abs(res) - 0.005))
            if best is None or cost < best[0]:
                best = (cost, ts, p[0], p[1])
        return best
    cost, ts, b, rate = one(x)
    rng = np.random.default_rng(seed)
    ev = x.hyp.unique()
    boots = []
    for _ in range(nboot):
        pick = rng.choice(ev, len(ev), replace=True)
        xb = x.set_index("hyp").loc[pick].reset_index()
        boots.append(one(xb)[1:])
    boots = np.array(boots)
    return dict(station=st, te=te, ts_days=ts, b=b, rate=rate, boots=boots, n=len(x), x=x)


def cc_rate(f, nboot=300, seed=0):
    """Rate of the ramp from the CC rows (accepted, msh): r = dt_cc - consensus of the other stations of the pair
    (median dt_cc of the pair without the clock stations, >= 5 rows) = rate * (ramp(t1) - ramp(t2)), ramp(t) =
    days since the drift start inside [start, reset). Huber IRLS; pair bootstrap. The NLLoc residuals under-
    estimate the amplitude because the v5.1 locations, made with these picks, absorb part of the clock error."""
    d = pd.read_parquet(C.OUT / "cc_msh.parquet")
    a = d[d.above]
    cons = a[~a.station.isin(CLOCK_STATIONS)].groupby("pair").dt_cc.agg(["median", "size"])
    x = a[a.station == f["station"]].copy()
    x["cons"] = x.pair.map(cons["median"])
    x["ncons"] = x.pair.map(cons["size"])
    x = x[x.ncons >= 5]
    te = pd.Timestamp(f["te"]).timestamp()

    def ramp(t, ts):
        dd = (np.asarray(t) - te) / 86400.0
        return np.where((dd >= ts) & (dd < 0), dd - ts, 0.0)
    y = (x.dt_cc - x.cons).values

    def one(idx, ts):
        g = ramp(x.t_pick1.values[idx], ts) - ramp(x.t_pick2.values[idx], ts)
        yy = y[idx]
        m = g != 0
        if m.sum() < 5:
            return np.nan
        w = np.ones(m.sum())
        for _ in range(8):
            rr = np.sum(w * g[m] * yy[m]) / np.sum(w * g[m] ** 2)
            res = yy[m] - rr * g[m]
            w = 1 / np.maximum(1, np.abs(res) / 0.03)
        return rr
    rate = one(np.arange(len(x)), f["ts_days"])
    rng = np.random.default_rng(seed)
    bt = f["boots"]
    rb = np.array([one(rng.integers(0, len(x), len(x)), bt[k, 0]) for k in range(min(nboot, len(bt)))])
    g = ramp(x.t_pick1.values, f["ts_days"]) - ramp(x.t_pick2.values, f["ts_days"])
    return rate, rb, int((g != 0).sum()), x.assign(g=g, r=y)


def model(f, t_posix):
    te = pd.Timestamp(f["te"]).timestamp()
    d = (np.asarray(t_posix) - te) / 86400.0
    return np.where((d >= f["ts_days"]) & (d < 0), f["rate"] * (d - f["ts_days"]), 0.0)


def epochs_for(f):
    te = pd.Timestamp(f["te"])
    ts = te + pd.Timedelta(days=f["ts_days"])
    dur = -f["ts_days"]
    n = int(np.ceil(abs(f["rate"]) * dur / STEP))
    edges = [ts + pd.Timedelta(days=dur * k / n) for k in range(n + 1)]
    rows = [dict(station=f["station"], epoch=0, start_utc="", end_utc=ts.isoformat(), offset_s=0.0, offset_sd_s=np.nan,
                 kind="reference (before drift; also after reset)")]
    bt = f["boots"]
    for k in range(n):
        a, b = edges[k], edges[k + 1]
        mid = a + (b - a) / 2
        dm = (mid - te).total_seconds() / 86400.0
        off = f["rate"] * (dm - f["ts_days"])
        offb = np.where(dm >= bt[:, 0], bt[:, 2] * (dm - bt[:, 0]), 0.0)
        rows.append(dict(station=f["station"], epoch=k + 1, start_utc=a.isoformat(), end_utc=b.isoformat(),
                         offset_s=off, offset_sd_s=float(np.nanstd(offb)), kind="drift"))
    rows.append(dict(station=f["station"], epoch=0, start_utc=te.isoformat(), end_utc="", offset_s=0.0,
                     offset_sd_s=np.nan, kind="reference (after reset)"))
    return rows, edges


def epoch_id(t_posix, edges):
    e = np.array([x.timestamp() for x in edges])
    k = np.searchsorted(e, t_posix, side="right")
    return np.where((k >= 1) & (k <= len(e) - 1), k, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", action="store_true")
    ap.add_argument("--fit", default="", help="STA.NET:reset_time,...")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    QC.mkdir(parents=True, exist_ok=True)
    if a.screen:
        screen()
    if not a.fit:
        return
    r = residuals()
    xd = r[r.station.str.endswith(".XD")].copy()          # demean over the XD stations of the event (as for the fit)
    xd["rd"] = xd.res - xd.groupby("hyp").res.transform("median")
    fits, erows, edges = {}, [], {}
    fig, axs = plt.subplots(len(a.fit.split(",")), 1, figsize=(12, 3.2 * len(a.fit.split(","))), squeeze=False)
    for ax, item in zip(axs[:, 0], a.fit.split(",")):
        st, te = item.split(":", 1)
        f = fit_station(xd, st, te, "2015-10-09", "2016-06-21")
        f["rate_nll"] = f["rate"]
        rate, rb, ncc, xcc = cc_rate(f)
        f["rate"], f["n_cc"] = rate, ncc
        f["boots"] = np.c_[f["boots"][:len(rb), :2], rb]          # (ts, b, rate): rate from the CC bootstrap
        xcc.to_csv(CLK / f"cc_rows_{st}.csv", index=False)
        fits[st] = f
        rows, ed = epochs_for(f)
        erows += rows
        edges[st] = ed
        x = f["x"]
        for ph, c in (("P", "C0"), ("S", "C3")):
            y = x[x.phase == ph]
            ax.plot(y.t, y.rd.clip(-1, 1), ".", ms=2, color=c, alpha=0.5, label=ph)
        tt = pd.date_range("2015-10-09", "2016-06-21", freq="6h")
        ax.plot(tt, model(f, tt.astype("int64") / 1e9), "k-", lw=1.2, label="ramp model")
        ax.axvline(pd.Timestamp(te), color="g", lw=0.8)
        ts = pd.Timestamp(te) + pd.Timedelta(days=f["ts_days"])
        bt = f["boots"]
        ax.set_title(f"{st}: drift from {ts:%Y-%m-%d %H:%M} (NLLoc fit, bootstrap SD {np.std(bt[:, 0]):.1f} d), "
                     f"{f['rate'] * 1e3:+.2f} ms/day from {f['n_cc']} CC rows (SD {np.nanstd(bt[:, 2]) * 1e3:.2f}; NLLoc "
                     f"{f['rate_nll'] * 1e3:+.2f}), reset {te} (data restart); dots: {f['n']} NLLoc picks", fontsize=8)
        f2 = dict(f, rate=f["rate_nll"])
        ax.plot(tt, model(f2, tt.astype("int64") / 1e9), "k:", lw=0.8, label="NLLoc-only rate")
        ax.set_ylim(-1, 1)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(QC / "clock_fit.png", dpi=100)
    # epochs table with n_rows (CC rows of all sets at that station with at least one event in the epoch)
    E = pd.DataFrame(erows)
    cc = {s: pd.read_parquet(C.OUT / f"cc_{s}.parquet") for s in C.SETS}
    nrows = []
    for e in E.itertuples():
        n = 0
        for s, d in cc.items():
            x = d[d.station == e.station]
            if not len(x):
                continue
            ids = [epoch_id(x.t_pick1.values, edges[e.station]), epoch_id(x.t_pick2.values, edges[e.station])]
            m = (ids[0] == e.epoch) | (ids[1] == e.epoch)
            if e.epoch == 0:
                lo = pd.Timestamp(e.start_utc).timestamp() if e.start_utc else -np.inf
                hi = pd.Timestamp(e.end_utc).timestamp() if e.end_utc else np.inf
                m = ((x.t_pick1 >= lo) & (x.t_pick1 < hi)) | ((x.t_pick2 >= lo) & (x.t_pick2 < hi))
            n += int(m.sum())
        nrows.append(n)
    E["n_rows"] = nrows
    E["method"] = [("ramp: start from NLLoc v5.1 residuals (event bootstrap), rate from CC dt vs other stations "
                    "(pair bootstrap), end = MiniSEED data restart"
                    if k == "drift" else "reference: GPS-synchronized clock (offset defined 0)") for k in E.kind]
    for st, f in fits.items():
        bt = f["boots"]
        m = E.station == st
        E.loc[m, "drift_start_utc"] = (pd.Timestamp(f["te"]) + pd.Timedelta(days=f["ts_days"])).isoformat()
        E.loc[m, "drift_start_sd_days"] = float(np.std(bt[:, 0]))
        E.loc[m, "rate_s_per_day"] = f["rate"]
        E.loc[m, "rate_sd_s_per_day"] = float(np.nanstd(bt[:, 2]))
        E.loc[m, "rate_nll_s_per_day"] = f["rate_nll"]
        E.loc[m, "n_cc_rows_rate"] = f["n_cc"]
        E.loc[m, "reset_utc"] = pd.Timestamp(f["te"]).isoformat()
    cols = ["station", "start_utc", "end_utc", "offset_s", "offset_sd_s", "n_rows", "method", "epoch", "kind",
            "drift_start_utc", "drift_start_sd_days", "rate_s_per_day", "rate_sd_s_per_day", "rate_nll_s_per_day",
            "n_cc_rows_rate", "reset_utc"]
    E[cols].to_csv(QC / "clock_epochs.csv", index=False, float_format="%.5g")
    (CLK / "clock_fits.json").write_text(json.dumps({st: dict(te=f["te"], ts_days=f["ts_days"], rate=f["rate"], b=f["b"],
                                                              n=f["n"]) for st, f in fits.items()}, indent=1))
    print(E[cols].to_string(index=False))
    if a.apply:
        summ = {}
        for s, d in cc.items():
            d = d.drop(columns=[c for c in d.columns if c.startswith("clock_") or c == "straddles_clock_epoch"])
            e1 = np.zeros(len(d), int)
            e2 = np.zeros(len(d), int)
            c1 = np.zeros(len(d))
            c2 = np.zeros(len(d))
            for st, f in fits.items():
                m = (d.station == st).values
                if not m.any():
                    continue
                e1[m] = epoch_id(d.t_pick1.values[m], edges[st])
                e2[m] = epoch_id(d.t_pick2.values[m], edges[st])
                c1[m] = model(f, d.t_pick1.values[m])
                c2[m] = model(f, d.t_pick2.values[m])
            d["clock_epoch1"], d["clock_epoch2"] = e1, e2
            d["clock_c1"], d["clock_c2"] = c1, c2
            d["clock_dt_s"] = c1 - c2
            d["straddles_clock_epoch"] = e1 != e2
            d.to_parquet(C.OUT / f"cc_{s}.parquet", index=False)
            st_ = d[d.straddles_clock_epoch]
            summ[s] = dict(rows=int(len(st_)), accepted=int(st_.above.sum()),
                           accepted_in_dt_given=int((st_.above & (st_.pair_source == "given")).sum()),
                           accepted_in_dt_added=int((st_.above & (st_.pair_source == "added")).sum()),
                           by_station={k: int(v) for k, v in st_.groupby("station").size().items()},
                           abs_clock_dt_gt_0p02=int((d.clock_dt_s.abs() > 0.02).sum()))
        (QC / "clock_apply_summary.json").write_text(json.dumps(summ, indent=1))
        print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()

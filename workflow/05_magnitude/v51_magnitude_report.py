#!/usr/bin/env python3
"""
Catalog v5.1 magnitudes: comparisons, pick-timing check and the HTML report.

Inputs: the outputs of scale/run_magnitude_v51.sh in --data, the v4 Route A run in
/home/mdenolle/cascadia_obs_ensemble/data/magnitude_v4, the v3 run (data/magnitude) for
the v3 continuity line, and the Mc / b tables of workflow/06_analysis/v51_mc_b.py in --out.

Writes into --out (worktree, small files):
  v51_vs_v4_by_region.csv, v51_vs_v4_by_mag.csv, v51_vs_v4_by_dist.csv, v51_vs_v4_by_flag.csv,
  v51_vs_v4_decomposition.csv, v51_vs_comcat.csv, tensor_compare.csv, timing_check.csv,
  station_terms_v51_vs_v4.csv, nohypo_fallback.csv, explosions.csv,
  cascadia_catalog_v5.1_magnitudes_slim.csv.gz, fig_*.png and v51_magnitude_report.html.

Usage:
  python v51_magnitude_report.py --data /wd1/.../v5.1_analysis/mag --out <worktree>/data/catalog_v5/v5.1_analysis/mag
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import re

import numpy as np
import pandas as pd

import phase19_preferred_magnitude as p19
from phase18_moment_tensor_match import hutton_boore
from v51_magnitude_table import near_inversion

V4 = "/home/mdenolle/cascadia_obs_ensemble/data/magnitude_v4"
V3 = "/home/mdenolle/cascadia_obs_ensemble/data/magnitude"
V3V4 = "/home/mdenolle/cascadia_obs_ensemble/data/catalog_v4/report/v3_v4_match.csv"
V4MC = "/home/mdenolle/wt_a51_mag/data/catalog_v4/mc_maps/mc_regional.csv"
STA = "/wd1/mdenolle_data/picks_share/stations/stations_ver5_v6.csv"
PICKS41T = "/wd1/mdenolle_data/catalog_v5/v5.1/Cascadia_picks_catalog_v4.1t.csv.gz"
REG_ORDER = ["Axial", "Endeavour", "Explorer", "JdF ridge/plate", "Blanco", "Gorda", "Mendocino",
             "Offshore forearc", "Puget/Georgia", "Cascades", "Other onshore"]

FIGS = {}


def med_iqr(s):
    s = s.dropna()
    if not len(s):
        return pd.Series(dict(n=0, median=np.nan, p25=np.nan, p75=np.nan, mean=np.nan))
    return pd.Series(dict(n=len(s), median=s.median(), p25=s.quantile(.25), p75=s.quantile(.75),
                          mean=s.mean()))


def grp(df, key, col, order=None):
    t = df.groupby(key, observed=True)[col].apply(med_iqr).unstack()
    if order is not None:
        t = t.reindex([o for o in order if o in t.index])
    t["n"] = t["n"].astype(int)
    return t.reset_index()


def savefig(fig, name, out):
    import matplotlib.pyplot as plt
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=130, bbox_inches="tight")
    fig.savefig(os.path.join(out, f"{name}.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    FIGS[name] = base64.b64encode(b.getvalue()).decode()


def img_file(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ------------------------------------------------------------------ comparisons
def compare_v4(m, out):
    """v5.1 ML - v4.1 ML on events with both (located in v5.1)."""
    d = m[m.ML.notna() & m.ML_v4.notna()].copy()
    d["dML"] = d.ML - d.ML_v4
    d["dM"] = d.M - d.M_v4
    d["offshore_lbl"] = np.where(d.offshore.astype(str) == "True", "offshore", "onshore")
    by_reg = grp(d, "region", "dML", REG_ORDER)
    by_off = grp(d, "offshore_lbl", "dML")
    by_meth = grp(d.assign(meth=d.ML_method_v4 + "->" + d.ML_method), "meth", "dML")
    bins = [-3, 0, 1, 2, 3, 4, 7]
    by_mag = grp(d.assign(Mbin=pd.cut(d.ML_v4, bins)), "Mbin", "dML")
    by_mag["Mbin"] = by_mag["Mbin"].astype(str)
    dbins = [0, 10, 25, 50, 100, 200, 1000]
    dd = d.merge(CAT[["ID", "dist_min", "shift_z_km", "shift_h_km"]], on="ID", how="left")
    by_dist = grp(dd.assign(dbin=pd.cut(dd.dist_min, dbins)), "dbin", "dML")
    by_dist["dbin"] = by_dist["dbin"].astype(str)
    zb = [-200, -10, -5, -2, 2, 5, 10, 200]
    by_dz = grp(dd.assign(zbin=pd.cut(dd.shift_z_km, zb)), "zbin", "dML")
    by_dz["zbin"] = by_dz["zbin"].astype(str)
    flags = []
    for f in ("floor_limited", "far_only", "deep_pass", "five_phase"):
        s = dd[dd[f].astype(str) == "True"]
        flags.append(dict(flag=f, **med_iqr(s.dML).to_dict()))
    for t in "ABCD":
        flags.append(dict(flag=f"tier {t}", **med_iqr(dd[dd.tier == t].dML).to_dict()))
    flags.append(dict(flag="non-earthquake flag", **med_iqr(dd[~dd.is_earthquake.astype(bool)].dML).to_dict()))
    by_flag = pd.DataFrame(flags)
    by_reg.to_csv(f"{out}/v51_vs_v4_by_region.csv", index=False, float_format="%.3f")
    by_mag.to_csv(f"{out}/v51_vs_v4_by_mag.csv", index=False, float_format="%.3f")
    by_dist.to_csv(f"{out}/v51_vs_v4_by_dist.csv", index=False, float_format="%.3f")
    by_dz.to_csv(f"{out}/v51_vs_v4_by_dz.csv", index=False, float_format="%.3f")
    by_flag.to_csv(f"{out}/v51_vs_v4_by_flag.csv", index=False, float_format="%.3f")
    by_off.to_csv(f"{out}/v51_vs_v4_offshore.csv", index=False, float_format="%.3f")
    # M_type transitions
    tt = pd.crosstab(m.M_type_v4.fillna("none"), m.M_type.fillna("none"))
    return dict(d=dd, by_reg=by_reg, by_off=by_off, by_meth=by_meth, by_mag=by_mag,
                by_dist=by_dist, by_dz=by_dz, by_flag=by_flag, mtype=tt)


def decomposition(D, out):
    """Per pick (same pick_id in both runs): change of the Hutton-Boore distance term,
    of log10 A (window length follows the distance), of the station term, plus the
    anchor offsets; event-level means compared with the observed ML change."""
    S = "_v51"
    near51 = pd.read_parquet(f"{D}/phase19_near_picks{S}.parquet")
    # v4: rebuild the phase19 near inversion on the v4 dataset
    ds4 = pd.read_csv(f"{V4}/amp_distance_dataset_routeA.csv",
                      usecols=["arid", "event_id", "station", "phase", "dist_hypo_km", "log10A"])
    raw4 = pd.read_csv(f"{V4}/raw_wa_amplitudes_v4.csv", usecols=["arid", "pick_id", "window_s"])
    ds4["m"] = hutton_boore(ds4.log10A, ds4.dist_hypo_km)
    ds4["sp"] = ds4.station + "|" + ds4.phase
    near4, M4, C4 = near_inversion(ds4)
    near4 = near4.merge(raw4, on="arid", how="left")
    an4 = pd.read_csv(f"{V4}/route_b_ml_anchors_routeA.csv")[["event_id", "ml"]]
    g4 = near4.groupby("event_id").res.size()
    a4 = an4.join(M4[g4.reindex(M4.index).fillna(0) >= 3].rename("m"), on="event_id", how="inner")
    off4 = float(np.median(a4.ml - a4.m))
    an51 = pd.read_csv(f"{D}/route_b_ml_anchors{S}.csv")[["event_id", "ml"]]
    M51 = near51.groupby("event_id").M_ev.first()
    g51 = near51.groupby("event_id").res.size()
    a51 = an51.join(M51[g51 >= 3].rename("m"), on="event_id", how="inner")
    off51 = float(np.median(a51.ml - a51.m))
    raw51 = pd.read_csv(f"{D}/raw_wa_amplitudes_v51.csv", usecols=["pick_id", "window_s"])
    near51 = near51.merge(raw51, on="pick_id", how="left")
    k = ["pick_id"]
    j = near51[["event_id", "pick_id", "station", "phase", "dist_hypo_km", "log10A", "C", "res",
                "window_s", "timing_correction_s"]].merge(
        near4[["pick_id", "event_id", "dist_hypo_km", "log10A", "C", "res", "window_s", "station"]],
        on=k, suffixes=("", "_4"))
    j["d_dist"] = (hutton_boore(0, j.dist_hypo_km) - hutton_boore(0, j.dist_hypo_km_4))
    j["d_logA"] = j.log10A - j.log10A_4
    j["d_C"] = j.C - j.C_4
    ev = j.groupby("event_id")[["d_dist", "d_logA", "d_C"]].mean()
    ev["n_common"] = j.groupby("event_id").size()
    t51 = pd.read_csv(f"{D}/phase19_station_terms{S}.csv")
    t51["sp"] = t51.station + "|" + t51.phase
    st = t51.set_index("sp")[["C", "n_obs"]].rename(columns={"C": "C_v51", "n_obs": "n_v51"}).join(
        C4.rename("C_v4"), how="outer")
    st["n_v4"] = near4.groupby("sp").size()
    st = st.reset_index().rename(columns={"index": "sp"})
    st[["station", "phase"]] = st.sp.str.split("|", expand=True)
    st["dC"] = st.C_v51 - st.C_v4
    sta = pd.read_csv(STA).drop_duplicates(["network", "station"])
    sta["key"] = sta.network + "." + sta.station
    st["seafloor"] = st.station.str.split("@").str[0].map(sta.set_index("key").seafloor)
    st.to_csv(f"{out}/station_terms_v51_vs_v4.csv", index=False, float_format="%.4f")
    return dict(j=j, ev=ev, off4=off4, off51=off51, st=st, n4=len(near4), n51=len(near51),
                C4=C4, near4=near4)


def compare_comcat(m, out):
    rows = []
    a51 = pd.read_csv(f"{DATA}/route_b_ml_anchors_v51.csv")
    a4 = pd.read_csv(f"{V4}/route_b_ml_anchors_routeA.csv")
    v4m = pd.read_csv(f"{V4}/cascadia_catalog_M_routeA.csv")[["event_id", "ML", "ML_method"]]
    x51 = a51[["event_id", "ml"]].merge(m[["ID", "ML", "ML_method", "region", "offshore", "is_earthquake"]],
                                        left_on="event_id", right_on="ID")
    x51["res"] = x51.ML - x51.ml
    x4 = a4[["event_id", "ml"]].merge(v4m, on="event_id").merge(
        m[["ID", "region", "offshore"]], left_on="event_id", right_on="ID", how="left")
    x4["res"] = x4.ML - x4.ml
    for name, x in (("v5.1", x51), ("v4.1", x4)):
        x = x.assign(off=np.where(x.offshore.astype(str) == "True", "offshore", "onshore"))
        for key in ("region", "off"):
            for k_, g in x.groupby(key):
                rows.append(dict(catalog=name, group=key, value=k_, **med_iqr(g.res).to_dict(),
                                 std=g.res.std()))
        rows.append(dict(catalog=name, group="all", value="all", **med_iqr(x.res).to_dict(), std=x.res.std()))
        for lo, hi in ((1.5, 2.5), (2.5, 3.5), (3.5, 6)):
            g = x[(x.ml >= lo) & (x.ml < hi)]
            rows.append(dict(catalog=name, group="ComCat ML", value=f"{lo}-{hi}", **med_iqr(g.res).to_dict(),
                             std=g.res.std()))
    t = pd.DataFrame(rows)
    t.to_csv(f"{out}/v51_vs_comcat.csv", index=False, float_format="%.3f")
    # ANSS any type, by magType (v5.1 M vs ANSS mag)
    y = m[m.anss_mag.notna() & m.M.notna()].copy()
    y["res"] = y.M - y.anss_mag
    y4 = y[y.M_v4.notna()].assign(res4=lambda z: z.M_v4 - z.anss_mag)
    tm = y.groupby(y.anss_magtype.str.lower()).res.apply(med_iqr).unstack()
    tm["v4_median"] = y4.groupby(y4.anss_magtype.str.lower()).res4.median()
    tm = tm[tm.n >= 20].sort_values("n", ascending=False).reset_index()
    tm["n"] = tm.n.astype(int)
    return dict(x51=x51, x4=x4, t=t, tm=tm)


def tensors(D, out):
    t51 = pd.read_csv(f"{D}/comcat_mt_matched_v51.csv")
    t4 = pd.read_csv(f"{V4}/comcat_mt_matched_v4.csv")
    m51 = pd.read_csv(f"{D}/cascadia_catalog_M_v51.csv")[["event_id", "ML", "M", "M_type"]]
    m4 = pd.read_csv(f"{V4}/cascadia_catalog_M_routeA.csv")[["event_id", "ML", "M", "M_type"]]
    rows = []
    for name, t, mm_ in (("v5.1", t51, m51), ("v4.1", t4, m4)):
        x = t.merge(mm_, on="event_id", how="left")
        for lo, hi in ((3, 4.5), (4.5, 5.5), (5.5, 7.5)):
            s = x[(x.Mw_mt >= lo) & (x.Mw_mt < hi)]
            rows.append(dict(catalog=name, Mw_class=f"{lo}-{hi}", n=len(s),
                             ML_minus_Mw=np.median(s.ML - s.Mw_mt) if len(s) else np.nan,
                             dist_km_median=s.dist_km.median(), ddepth_km_median=s.ddepth_km.median()))
        rows.append(dict(catalog=name, Mw_class="all matched", n=len(x), ML_minus_Mw=np.nan,
                         dist_km_median=x.dist_km.median(), ddepth_km_median=x.ddepth_km.median()))
    r = pd.DataFrame(rows)
    r.to_csv(f"{out}/tensor_compare.csv", index=False, float_format="%.3f")
    lost = sorted(set(t4.comcat_id) - set(t51.comcat_id))
    gained = sorted(set(t51.comcat_id) - set(t4.comcat_id))
    return dict(t=r, n51=len(t51), n4=len(t4), lost=lost, gained=gained, t51=t51, t4=t4)


def timing_check(dec, out):
    """Station-magnitude residuals (phase19, v5.1) against the pick-timing correction."""
    n = pd.read_parquet(f"{DATA}/phase19_near_picks_v51.parquet")
    n["corr"] = n.timing_correction_s.fillna(0.0)
    n["absc"] = n["corr"].abs()
    sta_ret = set(n.loc[n.absc >= 0.005, "station"])
    same = n[n.station.isin(sta_ret)]
    bins = [-0.001, 0.005, 0.1, 0.2, 0.3, 0.5, 1.0]
    rows = []
    for ph in ("P", "S"):
        s = same[same.phase == ph]
        for b, g in s.groupby(pd.cut(s.absc, bins), observed=True):
            rows.append(dict(phase=ph, abs_correction_s=str(b), **med_iqr(g.res).to_dict()))
    t = pd.DataFrame(rows)
    t["n"] = t.n.astype(int)
    # LAD-like slope of residual on |correction| for picks with a correction >= 5 ms, per phase,
    # with a station-phase intercept (residual minus its station-phase median)
    sl = {}
    for ph in ("P", "S"):
        s = same[same.phase == ph].copy()
        s["r0"] = s.res - s.groupby("sp").res.transform("median")
        x = s[s.absc >= 0.005]
        if len(x) > 100:
            A = np.c_[x.absc.values, np.ones(len(x))]
            coef = np.linalg.lstsq(A, x.r0.values, rcond=None)[0]
            bs = []
            rng = np.random.default_rng(0)
            for _ in range(200):
                i = rng.integers(0, len(x), len(x))
                bs.append(np.linalg.lstsq(A[i], x.r0.values[i], rcond=None)[0][0])
            sl[ph] = (coef[0], np.std(bs), len(x))
    t.to_csv(f"{out}/timing_check.csv", index=False, float_format="%.4f")
    # Control for time of day: the listed drift grows through each UTC day (0 to ~0.53 s), so
    # |correction| is also a clock of the hour. On stations with no re-timed pick, regress the
    # residual on the same "pseudo correction" 0.53 s x (seconds of day / 86400).
    pt = pd.read_csv(PICKS41T, usecols=[2], skipinitialspace=True).iloc[:, 0].str.strip()
    sod = pd.to_datetime(pt, utc=True, format="ISO8601")
    sod = (sod.dt.hour * 3600 + sod.dt.minute * 60 + sod.dt.second).to_numpy() / 86400.0
    n["pseudo"] = 0.53 * sod[n.arid.to_numpy()]
    ctl = n[~n.station.isin(sta_ret)].copy()
    ctl["r0"] = ctl.res - ctl.groupby("sp").res.transform("median")
    sc = {}
    for ph in ("P", "S"):
        x = ctl[ctl.phase == ph]
        A = np.c_[x.pseudo.values, np.ones(len(x))]
        coef = np.linalg.lstsq(A, x.r0.values, rcond=None)[0]
        rng = np.random.default_rng(1)
        bs = []
        for _ in range(100):
            i = rng.integers(0, len(x), len(x))
            bs.append(np.linalg.lstsq(A[i], x.r0.values[i], rcond=None)[0][0])
        sc[ph] = (coef[0], np.std(bs), len(x))
    pd.DataFrame([dict(test=f"re-timed picks, residual vs |correction|, {ph}", slope_per_s=v[0],
                       slope_sigma=v[1], n=v[2]) for ph, v in sl.items()]
                 + [dict(test=f"control (stations never re-timed), residual vs 0.53 s x day fraction, {ph}",
                         slope_per_s=v[0], slope_sigma=v[1], n=v[2]) for ph, v in sc.items()]).to_csv(
        f"{out}/timing_slopes.csv", index=False, float_format="%.4f")
    return dict(t=t, slopes=sl, control=sc, same=same, n_ret=int((n.absc >= 0.005).sum()),
                n_ret01=int((n.absc >= 0.1).sum()), n_sta=len(sta_ret))


def whatif_window(D, out, nmax=60000):
    """For picks re-timed by >= 0.1 s: log10 of the amplitude in the selected window over
    the amplitude in the same window cut short at the end by the correction (rounded to the
    grid). The size of the error a misaligned window would have made."""
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    import pyarrow as pa
    from route_a_day_amplitudes import WINDOWS, WA_COLS
    n = pd.read_parquet(f"{D}/phase19_near_picks_v51.parquet",
                        columns=["pick_id", "phase", "dist_hypo_km", "timing_correction_s"])
    n = n[n.timing_correction_s.abs() >= 0.1]
    if len(n) > nmax:
        n = n.sample(nmax, random_state=0)
    raw = pd.read_csv(f"{D}/raw_wa_amplitudes_v51.csv", usecols=["pick_id", "station", "window_s"])
    n = n.merge(raw, on="pick_id")
    files = sorted({f"/wd1/mdenolle_data/amps_v6_all/{f}" for f in os.listdir("/wd1/mdenolle_data/amps_v6_all")
                    if f.endswith(".parquet") and any(f"_{s.split('.')[0]}.{s.split('.')[1]}." in f
                                                      for s in n.station.unique())})
    vs = pa.array(n.pick_id.to_numpy())
    parts = []
    for f in files:
        t = pq.read_table(f, columns=["pick_id"] + WA_COLS)
        t = t.filter(pc.is_in(t["pick_id"], value_set=vs))
        if t.num_rows:
            parts.append(t.to_pandas())
    a = pd.concat(parts).drop_duplicates("pick_id").set_index("pick_id")
    w = np.asarray(WINDOWS)
    n = n[n.pick_id.isin(a.index)].copy()
    iw = np.searchsorted(w, n.window_s.values - 1e-9)
    ic = np.searchsorted(w, n.window_s.values - n.timing_correction_s.abs().values + 1e-9) - 1
    ic = np.clip(ic, 0, None)
    A = a.loc[n.pick_id, WA_COLS].to_numpy()
    r = np.arange(len(n))
    with np.errstate(divide="ignore", invalid="ignore"):
        n["dlog"] = np.log10(A[r, iw]) - np.log10(A[r, ic])
    rows = []
    for ph in ("P", "S"):
        s = n[n.phase == ph]
        rows.append(dict(phase=ph, **med_iqr(s.dlog).to_dict(), p90=s.dlog.quantile(.9),
                         share_gt_0p1=(s.dlog > 0.1).mean()))
    t = pd.DataFrame(rows)
    t["n"] = t.n.astype(int)
    t.to_csv(f"{out}/timing_whatif.csv", index=False, float_format="%.4f")
    return t


# ------------------------------------------------------------------ figures
def figures(m, cv, dec, cc, tc, mc, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6})
    d = cv["d"]

    # 1) v5.1 - v4.1 by region, offshore/onshore, magnitude, distance, depth change
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    br = cv["by_reg"]
    y = np.arange(len(br))
    ax[0, 0].errorbar(br["median"], y, xerr=[br["median"] - br.p25, br.p75 - br["median"]], fmt="o",
                      color="k", ms=4, capsize=2)
    ax[0, 0].set_yticks(y, [f"{r} ({n:,})" for r, n in zip(br.region, br.n)])
    ax[0, 0].axvline(0, color="0.6", lw=0.6)
    ax[0, 0].invert_yaxis()
    ax[0, 0].set_xlabel("ML(v5.1) - ML(v4.1), median and IQR")
    ax[0, 0].set_title("a) by region (v4 region label)")
    bm = cv["by_mag"]
    ax[0, 1].errorbar(np.arange(len(bm)), bm["median"], yerr=[bm["median"] - bm.p25, bm.p75 - bm["median"]],
                      fmt="o", color="k", ms=4, capsize=2)
    ax[0, 1].set_xticks(np.arange(len(bm)), [f"{b}\n{n:,}" for b, n in zip(bm.Mbin, bm.n)])
    ax[0, 1].axhline(0, color="0.6", lw=0.6)
    ax[0, 1].set_xlabel("ML(v4.1) bin, events")
    ax[0, 1].set_title("b) by magnitude")
    bd = cv["by_dist"]
    ax[1, 0].errorbar(np.arange(len(bd)), bd["median"], yerr=[bd["median"] - bd.p25, bd.p75 - bd["median"]],
                      fmt="o", color="k", ms=4, capsize=2)
    ax[1, 0].set_xticks(np.arange(len(bd)), [f"{b}\n{n:,}" for b, n in zip(bd.dbin, bd.n)])
    ax[1, 0].axhline(0, color="0.6", lw=0.6)
    ax[1, 0].set_xlabel("v5.1 nearest-station epicentral distance (km), events")
    ax[1, 0].set_title("c) by nearest-station distance")
    bz = cv["by_dz"]
    ax[1, 1].errorbar(np.arange(len(bz)), bz["median"], yerr=[bz["median"] - bz.p25, bz.p75 - bz["median"]],
                      fmt="o", color="k", ms=4, capsize=2)
    ax[1, 1].set_xticks(np.arange(len(bz)), [f"{b}\n{n:,}" for b, n in zip(bz.zbin, bz.n)], fontsize=6.5)
    ax[1, 1].axhline(0, color="0.6", lw=0.6)
    ax[1, 1].set_xlabel("depth change v5.1 - v4 (km), events")
    ax[1, 1].set_title("d) by depth change")
    for a in ax.ravel():
        a.grid(alpha=0.25)
    fig.tight_layout()
    savefig(fig, "fig_v51_minus_v4", out)

    # 2) decomposition scatter: predicted (distance + window + station term + offset) vs observed
    ev = dec["ev"].join(d.set_index("ID")[["dML", "region", "ML_method", "ML_method_v4"]], how="inner")
    ev = ev[(ev.ML_method == "near") & (ev.ML_method_v4 == "near")]
    ev["pred"] = ev.d_dist + ev.d_logA - ev.d_C + (dec["off51"] - dec["off4"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    ax[0].hexbin(ev.pred, ev.dML, gridsize=80, bins="log", mincnt=1, extent=(-1, 1, -1, 1), cmap="viridis")
    ax[0].plot([-1, 1], [-1, 1], "w--", lw=0.8)
    ax[0].set_xlabel("predicted from common picks (distance + window + station term + offset)")
    ax[0].set_ylabel("observed ML(v5.1) - ML(v4.1)")
    ax[0].set_title(f"a) {len(ev):,} events 'near' in both runs")
    comp = ev.groupby("region")[["d_dist", "d_logA", "d_C", "dML"]].median().reindex(
        [r for r in REG_ORDER if r in ev.region.unique()])
    comp["neg_d_C"] = -comp.d_C
    x = np.arange(len(comp))
    w = 0.2
    ax[1].bar(x - 1.5 * w, comp.d_dist, w, label="distance term")
    ax[1].bar(x - 0.5 * w, comp.d_logA, w, label="log A (window)")
    ax[1].bar(x + 0.5 * w, comp.neg_d_C, w, label="- station term")
    ax[1].bar(x + 1.5 * w, comp.dML, w, label="observed dML", color="k")
    ax[1].axhline(dec["off51"] - dec["off4"], color="r", lw=0.8, ls="--",
                  label=f"anchor offset change {dec['off51'] - dec['off4']:+.3f}")
    ax[1].set_xticks(x, comp.index, rotation=45, ha="right")
    ax[1].legend(fontsize=6.5)
    ax[1].set_title("b) median contributions by region")
    ax[1].grid(alpha=0.25)
    fig.tight_layout()
    savefig(fig, "fig_decomposition", out)
    comp.to_csv(f"{out}/v51_vs_v4_decomposition.csv", float_format="%.4f")

    # 3) station terms
    st = dec["st"].dropna(subset=["C_v51", "C_v4"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for ph, mk in (("P", "o"), ("S", "s")):
        for sf, col in (("True", "tab:blue"), ("False", "tab:orange")):
            s = st[(st.phase == ph) & (st.seafloor.astype(str) == sf)]
            ax[0].scatter(s.C_v4, s.C_v51, s=6, marker=mk, color=col, alpha=0.6,
                          label=f"{ph} {'seafloor' if sf == 'True' else 'land'} ({len(s)})")
    lim = [st[["C_v4", "C_v51"]].min().min() - 0.1, st[["C_v4", "C_v51"]].max().max() + 0.1]
    ax[0].plot(lim, lim, "k--", lw=0.6)
    ax[0].set_xlabel("phase19 station term, v4.1 run")
    ax[0].set_ylabel("phase19 station term, v5.1 run")
    ax[0].legend(fontsize=6.5)
    ax[0].set_title("a) station-phase terms (log10 units)")
    for sf, col in (("True", "tab:blue"), ("False", "tab:orange")):
        s = st[st.seafloor.astype(str) == sf]
        ax[1].hist(s.dC, bins=np.arange(-0.6, 0.61, 0.025), alpha=0.6, color=col,
                   label=f"{'seafloor' if sf == 'True' else 'land'}: median {s.dC.median():+.3f}")
    ax[1].set_xlabel("C(v5.1) - C(v4.1)")
    ax[1].legend(fontsize=7)
    ax[1].set_title("b) change of the station terms")
    fig.tight_layout()
    savefig(fig, "fig_station_terms", out)

    # 4) ComCat anchors
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    for a_, (name, x) in zip(ax[:2], (("v4.1", cc["x4"]), ("v5.1", cc["x51"]))):
        a_.hexbin(x.ml, x.ML, gridsize=50, bins="log", mincnt=1, extent=(1, 5.5, 0, 5.5))
        a_.plot([1, 5.5], [1, 5.5], "k--", lw=0.6)
        a_.set_xlabel("ComCat ML")
        a_.set_ylabel(f"our ML ({name})")
        a_.set_title(f"{name}: n={len(x):,}, median {x.res.median():+.3f}, std {x.res.std():.3f}")
    t = cc["t"]
    t = t[(t.group == "region")]
    piv = t.pivot(index="value", columns="catalog", values="median").reindex(
        [r for r in REG_ORDER if r in t.value.unique()])
    y = np.arange(len(piv))
    ax[2].plot(piv["v4.1"], y, "o", mfc="white", mec="k", label="v4.1")
    ax[2].plot(piv["v5.1"], y, "o", color="k", label="v5.1")
    ax[2].set_yticks(y, piv.index)
    ax[2].invert_yaxis()
    ax[2].axvline(0, color="0.6", lw=0.6)
    ax[2].set_xlabel("our ML - ComCat ML (median over anchors)")
    ax[2].legend(fontsize=7)
    ax[2].set_title("anchor residual by region")
    fig.tight_layout()
    savefig(fig, "fig_comcat", out)

    # 5) timing check
    t = tc["t"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a_, ph in zip(ax, ("P", "S")):
        s = t[t.phase == ph]
        a_.errorbar(np.arange(len(s)), s["median"], yerr=[s["median"] - s.p25, s.p75 - s["median"]],
                    fmt="o", color="k", ms=4, capsize=2)
        a_.set_xticks(np.arange(len(s)), [f"{b}\n{n:,}" for b, n in zip(s.abs_correction_s, s.n)], fontsize=6.5)
        a_.axhline(0, color="0.6", lw=0.6)
        sl = tc["slopes"].get(ph)
        a_.set_title(f"{ph}: residual vs |timing correction| on the {tc['n_sta']} re-timed stations"
                     + (f"\nslope {sl[0]:+.3f} ± {sl[1]:.3f} per s (n={sl[2]:,})" if sl else ""), fontsize=7.5)
        a_.set_xlabel("|timing correction| (s), picks")
        a_.set_ylabel("station ML residual (phase19)")
        a_.grid(alpha=0.25)
    fig.tight_layout()
    savefig(fig, "fig_timing_check", out)

    # 6) regional FMDs, v5.1 vs v4.1 M on the same events
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "06_analysis"))
    import v4_mc_maps as mm
    sel = m[m.loc_source.isin(["nll3d", "comcat"]) & m.is_earthquake.astype(bool)].copy()
    reg = np.full(len(sel), "ON", dtype=object)
    for key, _, (a1, b1, c1, d1) in mm.REGIONS[::-1]:
        s = sel.lon.between(a1, b1) & sel.lat.between(c1, d1)
        reg[s.values] = key
    sel["box"] = reg
    rt = mc["reg"]
    rt = rt[(rt.period == "all")]
    fig, axs = plt.subplots(3, 3, figsize=(10, 8.5), sharex=True)
    for axx, r in zip(axs.ravel(), mm.REG_ORDER):
        s = sel[sel.box == r]
        for col, lab, sty in (("M_v4", "v4.1 M", dict(mfc="white", mec="0.4")), ("M", "v5.1 M", dict(color="k"))):
            v = np.round(s[col].dropna().values, 1)
            if not len(v):
                continue
            mb = np.arange(-2.5, 7.01, 0.1)
            cum = np.array([(v >= x - 1e-6).sum() for x in mb])
            axx.semilogy(mb[cum > 0], cum[cum > 0], "o", ms=2, label=lab, **sty)
        rC = rt[(rt.version == "C_v51M_v51loc") & (rt.region == r)]
        rB = rt[(rt.version == "B_v4M_v51loc") & (rt.region == r)]
        ttl = f"{r} {mm.REGION_NAMES[r]}"
        if len(rC) and np.isfinite(rC.mc_best.iat[0]):
            axx.axvline(rC.mc_best.iat[0], color="k", lw=0.8)
            ttl += f"\nv5.1 Mc {rC.mc_best.iat[0]:.1f} b {rC.b.iat[0]:.2f}"
        if len(rB) and np.isfinite(rB.mc_best.iat[0]):
            axx.axvline(rB.mc_best.iat[0], color="0.5", lw=0.8, ls="--")
            ttl += f" | v4.1 Mc {rB.mc_best.iat[0]:.1f} b {rB.b.iat[0]:.2f}"
        axx.set_title(ttl, fontsize=7)
        axx.set_xlim(-1.5, 6.5)
        axx.grid(alpha=0.2)
        axx.legend(fontsize=6, loc="lower left")
    for a_ in axs[-1]:
        a_.set_xlabel("M")
    fig.tight_layout()
    savefig(fig, "fig_regional_fmd", out)

    # 7) distributions
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].hist(d.dML, bins=np.arange(-1.5, 1.51, 0.02), color="k")
    ax[0].set_xlabel("ML(v5.1) - ML(v4.1)")
    ax[0].set_title(f"a) {len(d):,} events: median {d.dML.median():+.3f}, IQR "
                    f"{d.dML.quantile(.25):+.3f} to {d.dML.quantile(.75):+.3f}")
    ax[1].hist(m.M.dropna(), bins=np.arange(-2, 7, 0.1), histtype="step", color="k", label="v5.1 M")
    ax[1].hist(m.M_v4.dropna(), bins=np.arange(-2, 7, 0.1), histtype="step", color="0.6", label="v4.1 M")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("M")
    ax[1].legend()
    ax[1].set_title("b) magnitude histograms")
    fig.tight_layout()
    savefig(fig, "fig_distributions", out)


# ------------------------------------------------------------------ html
CSS = """
:root{--bg:#ffffff;--ink:#222;--muted:#666;--rule:#ddd;--head:#f4f4f4}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1b1b1b;--ink:#e6e6e6;--muted:#aaa;--rule:#444;--head:#2a2a2a}
 img{background:#fff}}
body{background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;
 max-width:1050px;margin:0 auto;padding:16px}
h1{font-size:1.6em;margin-bottom:.2em} h2{margin-top:1.8em;border-bottom:1px solid var(--rule)} h3{margin-top:1.3em}
table{border-collapse:collapse;font-size:12.5px;margin:.6em 0;display:block;overflow-x:auto}
th,td{border:1px solid var(--rule);padding:3px 6px;text-align:right;white-space:nowrap}
th{background:var(--head)} td:first-child,th:first-child{text-align:left}
img{max-width:100%;height:auto;border:1px solid var(--rule)}
pre{font-size:11.5px;overflow-x:auto;background:var(--head);padding:8px}
.muted{color:var(--muted);font-size:.92em} code{font-size:.92em}
"""


def tab(df, fmt=None):
    df = df.copy()
    for c in df.columns:
        if (str(c) == "n" or str(c).startswith("n_")) and pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype("Int64")
    return df.to_html(index=False, border=0, na_rep="", float_format=fmt or (lambda x: f"{x:.2f}"))


def fig(name, cap):
    return (f"<figure><img alt='{name}' src='data:image/png;base64,{FIGS[name]}'>"
            f"<figcaption class='muted'>{cap}</figcaption></figure>")


def main(argv=None):
    global CAT, CATF, DATA
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mcb", default=None, help="v51_mc_b.py output dir (default <out>/mc_b)")
    ap.add_argument("--skip-whatif", action="store_true")
    a = ap.parse_args(argv)
    DATA, out = a.data, a.out
    os.makedirs(out, exist_ok=True)
    mcb = a.mcb or os.path.join(out, "mc_b")
    CAT = pd.read_csv("/wd1/mdenolle_data/catalog_v5/v5.1/cascadia_catalog_v5.1.csv",
                      usecols=["ID", "dist_min", "shift_z_km", "shift_h_km", "u_reason"])
    CATF = pd.read_csv("/wd1/mdenolle_data/catalog_v5/v5.1/cascadia_catalog_v5.1.csv",
                       usecols=["ID", "far_only", "shift_h_km", "shift_z_km", "tier", "region"])
    m = pd.read_csv(f"{DATA}/cascadia_catalog_v5.1_magnitudes.csv", low_memory=False)
    m["offshore"] = m.offshore.map({1.0: "True", 0.0: "False", True: "True", False: "False",
                                    "True": "True", "False": "False"}).fillna("")
    located = m.loc_source.isin(["nll3d", "comcat"])

    cv = compare_v4(m, out)
    dec = decomposition(DATA, out)
    cc = compare_comcat(m, out)
    tz = tensors(DATA, out)
    tc = timing_check(dec, out)
    wi = None if a.skip_whatif else whatif_window(DATA, out)
    reg = pd.read_csv(f"{mcb}/mc_b_regional.csv")
    regc = pd.read_csv(f"{mcb}/mc_b_catalog_region.csv")
    v4mc = pd.read_csv(V4MC)
    mc = dict(reg=reg, regc=regc)
    figures(m, cv, dec, cc, tc, mc, out)

    # ---- v3 continuity (shared events, v3 -> v4 match)
    v3 = pd.read_csv(f"{V3}/cascadia_catalog_M_routeA.csv", usecols=["event_id", "ML"])
    mt = pd.read_csv(V3V4).dropna()
    mt["ID"] = mt.ID.astype(int)
    x3 = mt.merge(v3.rename(columns={"event_id": "orid", "ML": "ML_v3"}), on="orid").merge(
        m[["ID", "ML", "ML_v4", "offshore", "region"]], on="ID")
    x3 = x3.dropna(subset=["ML_v3", "ML", "ML_v4"])
    x3["off"] = np.where(x3.offshore.astype(str) == "True", "offshore", "onshore")
    v3rows = []
    for k_, g in [("all", x3)] + list(x3.groupby("off")):
        v3rows.append(dict(group=k_, n=len(g), v4_minus_v3=(g.ML_v4 - g.ML_v3).median(),
                           v51_minus_v3=(g.ML - g.ML_v3).median(), v51_minus_v4=(g.ML - g.ML_v4).median()))
    v3t = pd.DataFrame(v3rows)
    v3t.to_csv(f"{out}/v3_v4_v51_shared.csv", index=False, float_format="%.3f")

    # ---- rows without hypocentre / explosions
    nh = m[~located]
    nht = pd.DataFrame([dict(item="rows without a hypocentre (loc_source none)", n=len(nh)),
                        dict(item="  with a v4.1 magnitude", n=int(nh.M_v4.notna().sum())),
                        dict(item="  fallback ML_v4hyp (>= 3 picks within 150 km of the v4 hypocentre)",
                             n=int(nh.ML_v4hyp.notna().sum())),
                        dict(item="  with a v4.1 tensor Mw (kept in Mw_mt_v4)", n=int(nh.Mw_mt_v4.notna().sum()))])
    fb = nh[nh.ML_v4hyp.notna() & nh.ML_v4.notna()]
    fbd = (fb.ML_v4hyp - fb.ML_v4)
    nh.merge(CAT[["ID", "u_reason"]], on="ID", how="left")[
        ["ID", "region", "u_reason", "M_v4", "M_type_v4", "ML_v4",
         "ML_v4hyp", "ML_v4hyp_unc", "n_ML_v4hyp", "Mw_mt_v4"]].to_csv(
        f"{out}/nohypo_fallback.csv", index=False, float_format="%.3f")
    ex = m[~m.is_earthquake.astype(bool)]
    ext = ex.assign(flag=ex.event_flag).groupby("flag").agg(
        n=("ID", "size"), with_M=("M", lambda s: int(s.notna().sum())), M_median=("M", "median"),
        M_max=("M", "max"), located=("loc_source", lambda s: int(s.isin(["nll3d", "comcat"]).sum()))).reset_index()
    ext.to_csv(f"{out}/explosions.csv", index=False, float_format="%.2f")

    # ---- slim table into the worktree
    slim = m[["ID", "M", "M_unc", "M_type", "ML", "ML_unc", "ML_method", "n_ML", "n_sta_ML", "Mw_mt", "comcat_id",
              "mt_match",
              "anchor_ml", "is_earthquake", "ML_v4hyp", "ML_v4hyp_unc", "n_ML_v4hyp", "Mw_mt_v4"]]
    slim.to_csv(f"{out}/cascadia_catalog_v5.1_magnitudes_slim.csv.gz", index=False,
                float_format="%.3f", compression="gzip")

    # ---- logs
    log51 = open(f"{DATA}/driver.log").read()
    log4 = open(f"{V4}/driver.log").read()

    def grab(pat, s, default="?"):
        r = re.search(pat, s)
        return r.group(1) if r else default

    write_html(out, m, cv, dec, cc, tz, tc, wi, reg, regc, v4mc, v3t, nht, fbd, ext, log51, log4, grab)




def fmt_int(x):
    return f"{int(x):,}"


def write_html(out, m, cv, dec, cc, tz, tc, wi, reg, regc, v4mc, v3t, nht, fbd, ext, log51, log4, grab):
    located = m.loc_source.isin(["nll3d", "comcat"])
    d = cv["d"]
    nloc = int(located.sum())
    nM = int(m.M.notna().sum())
    mt = m.M_type.value_counts()
    mt4 = m.M_type_v4.value_counts()
    meth = m.ML_method.value_counts()
    nh = m[~located]

    # ---- chain table (v4.1 run vs v5.1 run), from the two driver logs
    def g2(pat, n=1):
        a = re.search(pat, log4)
        b = re.search(pat, log51)
        return [(a.group(n) if a else "?"), (b.group(n) if b else "?")]
    rows = [
        ("arrivals with a ver6 pick_id", *g2(r"([\d,]+) arrivals ->")),
        ("  with an amplitude (reason ok)", *g2(r"\nok\s+(\d+)")),
        ("observations after SNR >= 3", *g2(r"clean observations\s+: ([\d,]+)")),
        ("events with >= 1 observation", *g2(r"events\s+: ([\d,]+)")),
        ("phase3 events with ML_inv", *g2(r"events WITH magnitude\s+: ([\d,]+)")),
        ("phase3 anelastic k, P (1/km; n fixed at 1)", *g2(r"P: n=1.000 k=([\d.]+)")),
        ("phase3 anelastic k, S", *g2(r"S: n=1.000 k=([\d.]+)")),
        ("phase2 ComCat ML anchors", *g2(r"matched anchors: ([\d,]+)")),
        ("phase2 calibration slope", *g2(r"calibration  ML = ([\d.]+) \*")),
        ("phase19 picks within 150 km", *g2(r"inversion within 150 km: ([\d,]+) picks")),
        ("phase19 station-phase terms", *g2(r"picks, ([\d,]+) station-phase terms")),
        ("phase19 residual robust std", *g2(r"residual robust std ([\d.]+)")),
        ("phase19 anchors / offset", *[f"{x} / +{y}" for x, y in zip(g2(r"ML offset to ([\d,]+) ComCat"),
                                                                    g2(r"ComCat anchors \+([\d.]+)"))]),
        ("phase19 anchor residual std", *g2(r"anchor residual std ([\d.]+) \(s_anchor")),
        ("ML_method", *g2(r"ML_method: (\{[^}]*\})")),
        ("M_type (phase19)", *g2(r"M_type: (\{[^}]*\})")),
        ("'mapped' ML >= 4 (lower bounds)", *g2(r"uncalibrated 'mapped' ML >= 4: (\d+)")),
        ("tensors matched (|dt| < 5 s, < 50 km)", *g2(r"matched (\d+) \(\|dt\|")),
        ("tensor match: origin time dt median (s)", *g2(r"dt median ([+-][\d.]+) s")),
        ("tensor match: depth (ours - ComCat) IQR (km)", *[f"{a}..{b}" for a, b in zip(
            g2(r"depth diff IQR ([+-][\d.]+)\.\."), g2(r"depth diff IQR [+-][\d.]+\.\.([+-][\d.]+) km"))]),
        ("Mw_cal offset ('near')", *g2(r"Mw_cal \(near\) = ML ([+-][\d.]+)")),
        ("ML - tensor Mw, Mw 3-4.5", *g2(r"tensor Mw 3-4.5: n=\s*\d+\s+ML - Mw ([+-][\d.]+)")),
        ("ML - tensor Mw, Mw 4.5-5.5", *g2(r"tensor Mw 4.5-5.5: n=\s*\d+\s+ML - Mw ([+-][\d.]+)")),
        ("ML - tensor Mw, Mw 5.5-7.5", *g2(r"tensor Mw 5.5-7.5: n=\s*\d+\s+ML - Mw ([+-][\d.]+)")),
    ]
    chain = pd.DataFrame(rows, columns=["step", "v4.1 run (v4 hypocentres)", "v5.1 run (v5.1 hypocentres)"])

    # ---- numbers used in the text
    dml = d.dML
    off = cv["by_off"].set_index("offshore_lbl")
    br = cv["by_reg"].set_index("region")
    big = {t: int((dml.abs() > t).sum()) for t in (0.3, 0.5, 1.0)}
    fl = cv["by_flag"].set_index("flag")
    big_reg = d[dml.abs() > 0.5].region.value_counts()
    ev = dec["ev"].join(d.set_index("ID")[["dML", "ML_method", "ML_method_v4", "region"]], how="inner")
    ev = ev[(ev.ML_method == "near") & (ev.ML_method_v4 == "near")]
    ev["pred"] = ev.d_dist + ev.d_logA - ev.d_C + (dec["off51"] - dec["off4"])
    r_pred = np.corrcoef(ev.pred, ev.dML)[0, 1]
    sd_pred = (ev.dML - ev.pred).std()
    st = dec["st"]
    sl = tc["slopes"]
    sc = tc["control"]
    t51 = tz["t51"]
    t4 = tz["t4"]
    cc_all = cc["t"][(cc["t"].group == "all")].set_index("catalog")
    cc_reg = cc["t"][cc["t"].group == "region"].pivot(index="value", columns="catalog", values="median")
    near = pd.read_parquet(f"{DATA}/phase19_near_picks_v51.parquet", columns=["event_id", "sp"])
    nterm = near.groupby("sp").size()
    weak = near.assign(w=near.sp.map(nterm) < 20).groupby("event_id").w.all()
    n_weak = int(weak.sum())
    j35 = st[st.station.str.startswith("7D.J35C") & (st.phase == "S")]
    ex_with_M = int(m.loc[~m.is_earthquake.astype(bool), "M"].notna().sum())
    n_ex = int((~m.is_earthquake.astype(bool)).sum())
    regC = reg[(reg.version == "C_v51M_v51loc") & (reg.period == "all")].set_index("region")
    regB = reg[(reg.version == "B_v4M_v51loc") & (reg.period == "all")].set_index("region")
    regA = reg[(reg.version == "A_v4M_v4loc") & (reg.period == "all")].set_index("region")
    if wi is not None:
        w_ = wi.set_index("phase")
        wi_txt = (f"{w_.loc['P', 'mean']:+.3f} (P) and {w_.loc['S', 'mean']:+.3f} (S) log10 units on average "
                  f"(median 0)")
    else:
        wi_txt = "(not computed)"
    boxes = [r for r in regC.index if r not in ("OFF", "ALL")]
    dmc = {r: regC.loc[r, "mc_best"] - regB.loc[r, "mc_best"] for r in boxes}
    mc_txt = ", ".join(f"{r} {regB.loc[r, 'mc_best']:.1f} to {regC.loc[r, 'mc_best']:.1f}" for r in boxes
                       if abs(dmc[r]) > 0.05)
    mc_txt = (mc_txt + "; 0.0 elsewhere") if mc_txt else "0.0 everywhere"
    bsig = [r for r in boxes if abs(regC.loc[r, "b"] - regB.loc[r, "b"]) > 2 * regC.loc[r, "b_sigma_boot"]]
    b_txt = ", ".join(f"{r} ({regB.loc[r, 'b']:.2f} to {regC.loc[r, 'b']:.2f} &plusmn; "
                      f"{regC.loc[r, 'b_sigma_boot']:.2f})" for r in bsig) or "no box"
    nfar = int((t51.merge(CATF[["ID", "far_only"]], left_on="event_id", right_on="ID").far_only.astype(str) == "True").sum())
    mtm = m.mt_match.value_counts()
    lost_carried = m[(m.mt_match == "v4.1") & ~m.ID.isin([900001, 900002, 900003])]
    mw34 = grab(r"tensor Mw 3-4.5: n=\s*\d+\s+ML - Mw ([+-][\d.]+)", log51)
    nmap4 = grab(r"uncalibrated .mapped. ML >= 4: (\d+)", log51)

    h = ["<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
         f"content='width=device-width,initial-scale=1'><title>Catalog v5.1 magnitudes</title><style>{CSS}</style>"
         "</head><body>"]
    h.append("<h1>Magnitudes for catalog v5.1</h1>")
    h.append("<p class='muted'>Route A local magnitudes recomputed with the v5.1 hypocentres. Built 2026-11-25 by "
             "<code>scale/run_magnitude_v51.sh</code> (chain), <code>workflow/05_magnitude/v51_build_arrivals.py</code>, "
             "<code>v51_magnitude_table.py</code>, <code>workflow/06_analysis/v51_mc_b.py</code> (Mc, b) and "
             "<code>workflow/05_magnitude/v51_magnitude_report.py</code> (this page), branch <code>analysis/v51-mag</code>. "
             "Large files in <code>/wd1/mdenolle_data/catalog_v5/v5.1_analysis/mag/</code>; tables and figures in "
             "<code>data/catalog_v5/v5.1_analysis/mag/</code>. Private working material, not peer reviewed.</p>")

    # ---- summary
    h.append("<h2>Summary</h2><ul>")
    h.append(f"<li><b>Coverage.</b> {nM:,} of the {len(m):,} v5.1 rows have a magnitude M: "
             f"{mt.get('ML', 0):,} ML, {mt.get('Mw', 0)} Mw from ComCat moment tensors and {mt.get('Mw_cal', 0)} "
             f"Mw_cal (ML plus a constant, near-recorded events with ML + offset &ge; 4.5). In v4.1: "
             f"{mt4.get('ML', 0):,} ML, {mt4.get('Mw', 0)} Mw, {mt4.get('Mw_cal', 0)} Mw_cal. "
             f"{meth.get('near', 0):,} ML come from picks within 150 km ('near'), {meth.get('mapped', 0):,} are "
             f"mapped from the phase3 inversion ('mapped').</li>")
    h.append(f"<li><b>Rows without a hypocentre.</b> The {len(nh)} rows with loc_source none get no M "
             f"({int(nh.M_v4.notna().sum())} of them had one in v4.1). A fallback ML_v4hyp, computed at the v4 "
             f"hypocentre with the v5.1 station terms held fixed, is given in a separate column for "
             f"{int(nh.ML_v4hyp.notna().sum())} of them, and the v4.1 tensor Mw for {int(nh.Mw_mt_v4.notna().sum())}. "
             f"Neither is written to M.</li>")
    h.append(f"<li><b>Amplitudes were reused, not re-measured.</b> The amplitude windows are anchored at the pick "
             f"(0.3 s before it), but the amplitudes were measured at the uncorrected pick time on day traces "
             f"assembled the same way as the picker's, so they cover the samples the picker picked. On the "
             f"{tc['n_sta']} re-timed stations the station-magnitude residuals change by "
             f"{sl['P'][0]:+.3f} &plusmn; {sl['P'][1]:.3f} (P) and {sl['S'][0]:+.3f} &plusmn; {sl['S'][1]:.3f} (S) "
             f"log10 units per second of correction, i.e. at most 0.02 at the largest correction (0.77 s); on "
             f"stations never re-timed, the same regression against the time of day gives {sc['P'][0]:+.3f} (P) and "
             f"{sc['S'][0]:+.3f} (S). Cutting each window short by its correction would change the amplitude by "
             f"{wi_txt}.</li>")
    h.append(f"<li><b>Change from v4.1.</b> On the {len(d):,} events with ML in both versions, ML(v5.1) - ML(v4.1) "
             f"has median {dml.median():+.3f} (IQR {dml.quantile(.25):+.3f} to {dml.quantile(.75):+.3f}): "
             f"offshore {off.loc['offshore', 'median']:+.3f}, onshore {off.loc['onshore', 'median']:+.3f} "
             f"(offshore = epicentre below sea level). By region: Axial {br.loc['Axial', 'median']:+.2f}, "
             f"JdF ridge/plate {br.loc['JdF ridge/plate', 'median']:+.2f}, Endeavour {br.loc['Endeavour', 'median']:+.2f}, "
             f"Explorer {br.loc['Explorer', 'median']:+.2f}, Gorda {br.loc['Gorda', 'median']:+.2f}, "
             f"Mendocino {br.loc['Mendocino', 'median']:+.2f}, Blanco {br.loc['Blanco', 'median']:+.2f}, "
             f"offshore forearc {br.loc['Offshore forearc', 'median']:+.2f}, Puget/Georgia "
             f"{br.loc['Puget/Georgia', 'median']:+.2f}, Cascades {br.loc['Cascades', 'median']:+.2f}. "
             f"{big[0.3]:,} events change by more than 0.3 and {big[0.5]:,} by more than 0.5, "
             f"{int(big_reg.get('Cascades', 0) + big_reg.get('Other onshore', 0))} of the latter onshore "
             f"(mostly Mount St. Helens, where the stations are 1-10 km away).</li>")
    h.append(f"<li><b>Why.</b> For events 'near' in both runs, the change is reproduced from the common picks by "
             f"three terms: the Hutton-Boore distance term (new hypocentral distances), the station terms and the "
             f"anchor offset (correlation {r_pred:.3f}, residual std {sd_pred:.3f}). The window length almost never "
             f"changes (the distance-scaled window rounds to the same grid value). Offshore, the station terms of the "
             f"ocean-bottom stations dropped (median {st[st.seafloor.astype(str) == 'True'].dC.median():+.3f} vs "
             f"{st[st.seafloor.astype(str) == 'False'].dC.median():+.3f} on land), which raises offshore ML; at "
             f"Axial the hypocentres moved toward the caldera stations, so the distance term dropped by "
             f"{-ev[ev.region == 'Axial'].d_dist.median():.2f}, of which the station terms took back "
             f"{-ev[ev.region == 'Axial'].d_C.median():.2f}. Events whose depth is poorly constrained rose most: "
             f"deep-pass events by {fl.loc['deep_pass', 'median']:+.2f} (median depth 86 km in v5.1 against 31 km "
             f"in v4) and far_only events by {fl.loc['far_only', 'median']:+.2f} (52 km against 4 km); their ML "
             f"inherits the depth uncertainty.</li>")
    h.append(f"<li><b>ComCat.</b> Against the ComCat ML anchors the v5.1 ML has median residual "
             f"{cc_all.loc['v5.1', 'median']:+.3f} and std {cc_all.loc['v5.1', 'std']:.3f} "
             f"(v4.1: {cc_all.loc['v4.1', 'median']:+.3f}, {cc_all.loc['v4.1', 'std']:.3f}). "
             f"{tz['n51']} ComCat moment tensors match a v5.1 origin (v4.1: {tz['n4']}); below Mw 4.5 ML - Mw is "
             f"{mw34} (v4.1: -0.02), above it ML saturates as before and the tensor Mw is used. "
             f"{len(lost_carried)} located events lost their v4.1 tensor match because their v5.1 origin is now "
             f"31-101 km and 4.6-12 s from ComCat's; their v4.1 tensor is carried (the event is the same set of picks), "
             f"which keeps {int((lost_carried.M_type == 'Mw').sum())} of them at the tensor Mw instead of a "
             f"saturated ML. Offshore, our ML is above ComCat ML at Mendocino, Blanco and Gorda (anchor medians "
             f"{cc_reg.loc['Mendocino', 'v5.1']:+.2f}, {cc_reg.loc['Blanco', 'v5.1']:+.2f}, "
             f"{cc_reg.loc['Gorda', 'v5.1']:+.2f}; v4.1 {cc_reg.loc['Mendocino', 'v4.1']:+.2f}, "
             f"{cc_reg.loc['Blanco', 'v4.1']:+.2f}, {cc_reg.loc['Gorda', 'v4.1']:+.2f}).</li>")
    h.append(f"<li><b>A location result seen through the tensors.</b> The v5.1 hypocentres of the {tz['n51']} "
             f"matched tensor events lie deeper than the ComCat ones (median {t51.ddepth_km.median():+.1f} km, "
             f"IQR {t51.ddepth_km.quantile(.25):+.1f} to {t51.ddepth_km.quantile(.75):+.1f}; v4: "
             f"{t4.ddepth_km.median():+.1f}) and their origin times earlier (median {t51.dt_s.median():+.2f} s; "
             f"v4: {t4.dt_s.median():+.2f} s). Only {nfar} of them are far_only and none is from the deep pass. "
             f"ComCat depths offshore are often fixed (10 km), so this is a flag for the location work, not a "
             f"magnitude result.</li>")
    h.append(f"<li><b>Mc and b.</b> With the v5.1 magnitudes on the v5.1 epicentres (earthquakes only), "
             f"Mc_best is {regC.loc['ALL', 'mc_best']:.1f} for the whole catalog and b = {regC.loc['ALL', 'b']:.3f} "
             f"&plusmn; {regC.loc['ALL', 'b_sigma_boot']:.3f} (bootstrap; v4.1 magnitudes on the same events: "
             f"{regB.loc['ALL', 'mc_best']:.1f}, {regB.loc['ALL', 'b']:.3f}). Region by region (v4 cluster "
             f"boxes), Mc_best changes by {mc_txt}. b changes by more than twice its bootstrap error in "
             f"{b_txt}; elsewhere the change is within about two bootstrap errors.</li>")
    h.append("</ul>")

    # ---- decisions
    h.append("<h2>Decisions made here (proposed)</h2>")
    dec_rows = [
        ("Amplitudes", "Reuse the per-station-day Route A amplitudes (amps_v6_all, keyed by pick_id); do not "
         "re-measure, and do not shift the windows by the pick-timing correction.",
         "The windows sit on the samples the picker picked (same day-trace time base); shifting them by the "
         "correction would move them off the arrival. The residual check shows no dependence above 0.03 units."),
        ("Distance-scaled window", "Re-select the window length from the stored grid with the v5.1 hypocentral "
         "distance (as for v4).", "Same rule as v4; the stored running maxima cover 1-60 s."),
        ("Rows without a hypocentre", "No M (empty M, M_type). Fallback ML_v4hyp in its own column: median of the "
         "Hutton-Boore station magnitudes within 150 km of the v4 hypocentre minus the v5.1 phase19 station terms, "
         "plus the v5.1 anchor offset, >= 3 picks; not part of the joint inversion.",
         "A magnitude needs a distance; the v4 hypocentre is the one NonLinLoc could not reproduce, so it stays "
         "out of M but is kept for users who need a size."),
        ("v4.1 additions 900001-900003", "Keep the ComCat tensor Mw (M_type Mw) carried by v4.1.",
         "Their picks are repicks without a ver6 pick_id, so they have no Route A amplitude (as in v4.1)."),
        ("Tensor matches lost by the relocation", "Carry the v4.1 ComCat tensor of an event whose v5.1 origin "
         "fails phase18's 5 s / 50 km test (mt_match = v4.1); M = tensor Mw when Mw >= 4.5.",
         "The event is the same set of picks; the tensor Mw does not depend on our hypocentre, and without it "
         "these M 4.6-5.7 events would carry a saturated ML (0.4-1.3 low)."),
        ("Explosions and quarry blasts", "Keep their magnitude; is_earthquake = False; leave them out of Mc and b "
         "fits; keep them as ComCat ML anchors.", "Their ML is an amplitude measurement like any other; ComCat "
         "computes ML for them the same way."),
        ("ComCat inputs", "Reuse the v4 ComCat ML cache (4,458 events) and ANSS table.",
         "Keeps the anchor set identical to v4.1 so the comparison isolates the hypocentres. The v5.1 epicentres "
         "span a box larger by 0.2-0.3 deg to the N and E; ComCat events only there are not anchors."),
        ("Everything else", "Same chain and options as v4 (SNR >= 3, epoch-keyed station terms, n fixed at 1, "
         "150 km Hutton-Boore inversion, Mw switch at 4.5).", "Comparability with v4.1."),
        ("Mc / b", "Report Mc and b for the v4 cluster boxes (v4_mc_maps.py estimators) and for the v5.1 region "
         "labels; b uncertainty as the bootstrap std with Mc re-estimated per resample.",
         "Same boxes as the v4 Mc maps and the clustering analysis."),
    ]
    h.append(tab(pd.DataFrame(dec_rows, columns=["topic", "decision", "reason"])))

    # ---- chain
    h.append("<h2>The chain, v4.1 run against v5.1 run</h2>")
    h.append("<p>Same code and options; only the hypocentres and the event list differ. Numbers are read from the "
             "two driver logs.</p>")
    h.append(tab(chain))

    # ---- timing
    h.append("<h2>Pick timing and the amplitude windows</h2>")
    h.append("<p><b>How the windows depend on the picks.</b> For each pick, Route A takes the peak Wood-Anderson "
             "amplitude from 0.3 s before the pick to W s after it, with W = 1 + 0.03 r (P, at most 15 s and 0.8 of "
             "the S-P time) or 2 + 0.06 r (S, at most 60 s), r the hypocentral distance; the noise level comes from "
             "11 s to 1 s before the pick. So the window position depends on the pick time and its length on the "
             "hypocentre.</p>")
    h.append("<p><b>Why the existing amplitudes stay valid.</b> The re-timed picks (D-B072, D-B075) were wrong because "
             "the picker read a whole UTC day, ObsPy merged the MiniSEED records into one trace per contiguous "
             "segment and timed sample n as segment start + n/100 s, while the digitizer ran slightly off 100 Hz. "
             "The amplitude code (<code>route_a_day_amplitudes.py</code>) reads the same day from the same archive and "
             "merges it the same way (sample time = segment start + n / sampling rate), and it measured each pick at "
             "its uncorrected catalog time. The pick and the window therefore share one time base: the window starts "
             "0.3 s before the sample the picker picked. The correction changes the absolute time of the pick (which "
             "matters for travel times) but not which samples belong to the arrival; applying it to the amplitude "
             "window would move the window off the arrival by up to 0.77 s. The window length only depends on the "
             "hypocentre, and the stored running maxima (1-60 s grid) let the v5.1 distance pick a new length "
             "without reading waveforms again.</p>")
    h.append(f"<p><b>Check.</b> {tc['n_ret']:,} of the picks used by the phase19 inversion were re-timed by 5 ms "
             f"or more ({tc['n_ret01']:,} by 0.1 s or more) on {tc['n_sta']} station epochs. If the windows were "
             f"misplaced by the correction, their residuals would fall with the size of the correction. They do not, "
             f"beyond 0.02-0.03 units at the largest corrections, and the stations that were never re-timed show the "
             f"same small trend against the time of day (the drift of the listed stations grows through each UTC "
             f"day, so the correction is also a clock).</p>")
    h.append(fig("fig_timing_check", "Phase19 station-magnitude residuals on the station epochs that have re-timed "
                 "picks, binned by the absolute timing correction (median and IQR; numbers of picks under the bins)."))
    ts = pd.read_csv(f"{out}/timing_slopes.csv")
    h.append(tab(ts, fmt=lambda x: f"{x:.4f}"))
    if wi is not None:
        h.append("<p>For scale: had a window been misplaced, it would have ended early by the correction. Over "
                 "the picks re-timed by 0.1 s or more, the ratio of the amplitude in the selected window to the "
                 "amplitude in the same window cut short by the correction is (log10 units):</p>")
        h.append(tab(wi, fmt=lambda x: f"{x:.3f}"))
    h.append("<p class='muted'>Caveats: days whose amplitudes were read from another archive copy (EarthScope fill "
             "of incomplete pnwstore days) could have a different segment layout from the picker's; NC and BK were "
             "not checked for drift (D-B072). The check above covers the stations that were re-timed.</p>")

    # ---- rows without hypocentre, explosions
    h.append("<h2>Rows without a hypocentre</h2>")
    h.append(tab(nht, fmt=lambda x: f"{x:,.0f}"))
    h.append(f"<p>The {len(nh)} rows are events NonLinLoc rejected (maximum likelihood on a face of the search grid); "
             f"{int(nh.region.isin(['Cascades', 'Other onshore']).sum())} are onshore (Cascades and Other onshore "
             f"labels). Their picks are mostly P (5,360 P and 562 S observations above SNR 3). Where both exist, "
             f"ML_v4hyp - ML(v4.1) has median {fbd.median():+.3f} and IQR {fbd.quantile(.25):+.3f} to "
             f"{fbd.quantile(.75):+.3f} over {len(fbd)} events. Table: <code>nohypo_fallback.csv</code>.</p>")
    h.append("<h2>Explosions and quarry blasts</h2>")
    h.append(f"<p>{n_ex:,} rows carry a non-earthquake flag; {ex_with_M:,} of them have a magnitude, which is kept "
             f"(is_earthquake = False). They are left out of the Mc and b fits below.</p>")
    h.append(tab(ext))

    # ---- v4 comparison
    h.append("<h2>Comparison with the v4.1 Route A magnitudes</h2>")
    h.append(fig("fig_distributions", "Distribution of the ML change and the magnitude histograms."))
    h.append(fig("fig_v51_minus_v4", "ML(v5.1) - ML(v4.1), median and interquartile range: by region (the region label "
                 "of the v4 epicentre), by v4.1 ML, by the v5.1 distance to the nearest station and by the depth change."))
    h.append("<h3>By region</h3>" + tab(cv["by_reg"], fmt=lambda x: f"{x:+.3f}"))
    h.append("<h3>Offshore and onshore</h3>" + tab(cv["by_off"], fmt=lambda x: f"{x:+.3f}"))
    h.append("<h3>By magnitude (v4.1 ML)</h3>" + tab(cv["by_mag"], fmt=lambda x: f"{x:+.3f}"))
    h.append("<h3>By distance to the nearest station (v5.1)</h3>" + tab(cv["by_dist"], fmt=lambda x: f"{x:+.3f}"))
    h.append("<h3>By depth change (v5.1 - v4, km)</h3>" + tab(cv["by_dz"], fmt=lambda x: f"{x:+.3f}"))
    h.append("<h3>By location flag and tier</h3>" + tab(cv["by_flag"], fmt=lambda x: f"{x:+.3f}"))
    h.append("<p>Deep-pass events (median depth 86 km in v5.1, 31 km in v4) and far_only events (52 km against "
             "4 km) moved down by tens of km, which lengthens their distances and raises their ML. Their depths are "
             "flagged as poorly constrained, so their ML is too.</p>")
    h.append("<h3>ML_method, v4.1 to v5.1</h3>" + tab(cv["by_meth"], fmt=lambda x: f"{x:+.3f}"))
    mtt = cv["mtype"].reset_index().rename(columns={"M_type_v4": "M_type v4.1 \\ v5.1"})
    h.append("<h3>M_type, v4.1 (rows) to v5.1 (columns)</h3>" + tab(mtt, fmt=lambda x: f"{x:,.0f}"))
    h.append("<h3>What moves the magnitudes</h3>")
    h.append(f"<p>For each pick used by both phase19 inversions (same pick_id), the change of the event ML is the "
             f"mean over its picks of the change of the distance term, of log10 A (window length) and of the station "
             f"term, plus the change of the anchor offset ({dec['off4']:+.3f} to {dec['off51']:+.3f}). Over "
             f"{len(ev):,} events 'near' in both runs this reproduces the observed change with correlation "
             f"{r_pred:.3f}; the rest comes from picks that enter or leave the 150 km set or the robust rejection.</p>")
    h.append(fig("fig_decomposition", "a) observed against reconstructed change; b) median contributions per region."))
    comp = pd.read_csv(f"{out}/v51_vs_v4_decomposition.csv")
    h.append(tab(comp, fmt=lambda x: f"{x:+.3f}"))
    h.append(fig("fig_station_terms", "Phase19 station-phase terms of the two runs (the 7D.J35C S term, about "
                 "-234 in both runs, is off the plot; see open issues)."))
    h.append(f"<p>Large changes: {big[0.3]:,} events change by more than 0.3, {big[0.5]:,} by more than 0.5 and "
             f"{big[1.0]:,} by more than 1.0. Of those above 0.5, by region: "
             + ", ".join(f"{k} {v}" for k, v in big_reg.items()) + ". Most are within a few km of the Mount St. "
             "Helens stations, where a 1-2 km change of depth changes the hypocentral distance by a factor of two "
             "and the Hutton-Boore term (1.11 log10 r) by 0.3 or more. Hutton-Boore was calibrated at distances "
             "of tens of km; at 1-5 km the station terms carry much of the scale.</p>")
    h.append("<h3>Continuity with v3</h3>")
    h.append("<p>Events with an ML in v3, v4.1 and v5.1 (v3-v4 match table; offshore = v5.1 epicentre below sea "
             "level). The v4 science report gave v4 - v3 = +0.19 offshore on 41,111 events with its own split; on "
             "this set it is +0.17. v5.1 adds about +0.09 offshore.</p>")
    h.append(tab(v3t, fmt=lambda x: f"{x:+.3f}"))

    # ---- ComCat
    h.append("<h2>Comparison with ComCat</h2>")
    h.append(fig("fig_comcat", "Our ML against the ComCat ML of the phase2 anchors (same ComCat events in both runs)."))
    t = cc["t"].copy()
    h.append("<h3>Anchor residual (our ML - ComCat ML)</h3>")
    h.append(tab(t, fmt=lambda x: f"{x:+.3f}"))
    h.append("<h3>All ANSS magnitudes of matched events, by magnitude type (v5.1 M - ANSS mag)</h3>")
    h.append(tab(cc["tm"], fmt=lambda x: f"{x:+.3f}"))
    h.append("<h3>ComCat moment tensors</h3>")
    h.append(tab(tz["t"], fmt=lambda x: f"{x:+.2f}"))
    h.append(f"<p>Tensors matched by phase18 in v4.1 but not in v5.1: {len(tz['lost'])} "
             f"({', '.join(tz['lost'][:12])}); new matches: {len(tz['gained'])} ({', '.join(tz['gained'][:12])}). "
             f"A match needs |dt| &lt; 5 s and &lt; 50 km. Three of the lost ones are rows without a hypocentre "
             f"(their tensor Mw is in Mw_mt_v4). For the located ones the v4.1 tensor is carried "
             f"(mt_match = v4.1):</p>")
    lc = lost_carried.merge(CATF[["ID", "shift_h_km", "shift_z_km"]], on="ID", how="left")
    h.append(tab(lc[["ID", "region", "tier", "far_only", "depth", "shift_h_km", "shift_z_km", "comcat_id", "Mw_mt",
                     "ML", "M", "M_type", "M_v4", "M_type_v4"]], fmt=lambda x: f"{x:.2f}"))
    h.append(f"<p>Matched tensor events by v5.1 flag: far_only {nfar} of {tz['n51']}; tiers "
             f"{', '.join(f'{k} {v}' for k, v in t51.merge(CATF[['ID', 'tier']], left_on='event_id', right_on='ID').tier.value_counts().sort_index().items())}. "
             f"Median depth difference (ours - ComCat) by region: "
             + ", ".join(f"{k} {v:+.1f} km" for k, v in t51.merge(CATF[['ID', 'region']], left_on='event_id', right_on='ID')
                         .groupby('region').ddepth_km.median().items()) + ".</p>")

    # ---- Mc / b
    h.append("<h2>Magnitude of completeness and b-values</h2>")
    h.append("<p>Estimators of <code>v4_mc_maps.py</code>: MAXC + 0.2, MBS, GFT90/95 and the Woessner &amp; Wiemer "
             "(2005) 'best' rule; Aki-Utsu b with the Shi &amp; Bolt error; 200 bootstrap resamples. Maps: nearest "
             "250 events within 50 km of each 0.1 deg node (150 for the periods). Events: located in v5.1, with M, "
             "earthquakes only (C). B uses the v4.1 magnitudes on the same events and epicentres, A the v4.1 "
             "magnitudes on the v4 epicentres with every event (the v4 Mc maps).</p>")
    h.append(f"<p>{img_tag(f'{out}/mc_b/fig1_mc_three_methods_v51.png', 'Mc maps')}</p>"
             "<p class='muted'>Mc from the three estimators, v5.1 magnitudes. White inverted triangles: seafloor "
             "stations; dark triangles: land stations.</p>")
    h.append(f"<p>{img_tag(f'{out}/mc_b/fig2_b_and_change_v51.png', 'b and change')}</p>"
             "<p class='muted'>a) b above Mc_best; b) change of Mc_best and c) of b when the v4.1 magnitudes are "
             "replaced by the v5.1 ones on the same events.</p>")
    h.append(f"<p>{img_tag(f'{out}/mc_b/fig3_mc_periods_v51.png', 'Mc periods')}</p>"
             "<p class='muted'>Mc_best per deployment period, with the stations open in each period.</p>")
    cols = ["region", "n", "mc_maxc", "mc_mbs", "mc_gft90", "mc_gft95", "mc_best", "sigma_boot", "b", "b_sigma",
            "b_sigma_boot", "b_maxc", "b_maxc_sigma_boot"]
    h.append("<h3>Regional Mc and b (v4 cluster boxes), all years</h3>")
    for ver, lab in (("C_v51M_v51loc", "C: v5.1 M"), ("B_v4M_v51loc", "B: v4.1 M, same events"),
                     ("A_v4M_v4loc", "A: v4.1 M, v4 epicentres, all events")):
        h.append(f"<p><b>{lab}</b></p>")
        h.append(tab(reg[(reg.version == ver) & (reg.period == "all")][cols], fmt=lambda x: f"{x:.2f}"))
    cmp = regC[["n", "mc_best", "b", "b_sigma_boot"]].join(regB[["mc_best", "b"]], rsuffix="_B").join(
        regA[["mc_best", "b"]], rsuffix="_A").reset_index()
    cmp["d_mc_C_minus_B"] = cmp.mc_best - cmp.mc_best_B
    cmp["d_b_C_minus_B"] = cmp.b - cmp.b_B
    h.append("<h3>Change, region by region</h3>" + tab(cmp, fmt=lambda x: f"{x:.2f}"))
    h.append("<h3>Per period, v5.1 (C)</h3>")
    h.append(tab(reg[(reg.version == "C_v51M_v51loc") & (reg.period != "all")][["period"] + cols],
                 fmt=lambda x: f"{x:.2f}"))
    h.append("<h3>By v5.1 region label (all years)</h3>")
    rc = regc[["version", "region", "n", "mc_maxc", "mc_best", "sigma_boot", "b", "b_sigma", "b_sigma_boot",
               "b_maxc", "b_maxc_sigma_boot"]]
    h.append(tab(rc, fmt=lambda x: f"{x:.2f}"))
    h.append(fig("fig_regional_fmd", "Cumulative FMDs per box, v4.1 and v5.1 magnitudes on the same events; "
                 "vertical lines at Mc_best."))
    h.append(f"<p>Notes. The onshore box (ON) mixes Mount St. Helens (Mc near 0) with the rest; its MAXC and MBS "
             f"values are not meaningful (MAXC {regC.loc['ON', 'mc_maxc']:.1f}, MBS {regC.loc['ON', 'mc_mbs']:.1f}), "
             f"use GFT or the node maps. Axial's pooled GFT Mc was unstable in v4 (bootstrap sigma "
             f"{regA.loc['AX', 'sigma_boot']:.2f}); with v5.1 magnitudes it is {regC.loc['AX', 'mc_best']:.1f} "
             f"with bootstrap sigma {regC.loc['AX', 'sigma_boot']:.2f} (MAXC+0.2 {regC.loc['AX', 'mc_maxc']:.1f}). Offshore Oregon (OR) has only "
             f"{int(regC.loc['OR', 'n'])} events and the largest b change; its bootstrap error is "
             f"{regC.loc['OR', 'b_sigma_boot']:.2f}. Mc and b are in the preferred magnitude M (ML for 99.9% of "
             f"events).</p>")

    # ---- open issues
    h.append("<h2>Open issues</h2><ul>")
    h.append(f"<li><b>Unconstrained station term.</b> 7D.J35C S (2013-08-09 epoch) appears only in 9 events "
             f"that have no other pick within 150 km. Its phase19 term is undetermined and takes "
             f"{j35.C_v51.iat[0] if len(j35) else float('nan'):.0f} (v4.1 run: "
             f"{j35.C_v4.iat[0] if len(j35) else float('nan'):.0f}), which absorbs the S-phase gauge "
             f"(sum of S terms = 0). The 9 events are 'mapped' and unaffected; the absolute level is fixed by the "
             f"ComCat anchors. Same in v4.1. Proposed fix for the next version: drop station-phase terms not linked "
             f"to any event with a second station, and use one gauge.</li>")
    h.append(f"<li><b>Weakly tied ocean-bottom terms.</b> The 2013-14 7D FN stations and J49C have terms of +2 to +6 "
             f"log10 units that move by up to 2.8 between runs; several of their events have all their 'near' picks "
             f"on these stations, so their ML moves with them (e.g. event 74485: {m.set_index('ID').ML.get(74485, np.nan):+.2f} "
             f"in v5.1, {m.set_index('ID').ML_v4.get(74485, np.nan):+.2f} in v4.1). {n_weak:,} events have all their "
             f"'near' picks on station-phase terms with fewer than 20 picks. Terms this large point to a response "
             f"problem on those channels.</li>")
    h.append(f"<li><b>Short distances.</b> Hutton-Boore is used down to r of 1 km (Mount St. Helens, Axial). Small "
             f"depth changes move these ML by 0.3-1; Axial ML fell by {-br.loc['Axial', 'median']:.2f}.</li>")
    h.append(f"<li><b>Mapped events.</b> {nmap4} 'mapped' events "
             f"have ML &ge; 4 without a tensor; their ML is a lower bound (phase19 docstring).</li>")
    h.append(f"<li><b>Tensor events are deeper in v5.1.</b> Median {t51.ddepth_km.median():+.1f} km against ComCat "
             f"(v4: {t4.ddepth_km.median():+.1f}); {len(lost_carried)} events moved beyond the 5 s / 50 km match. "
             f"For the location work.</li>")
    h.append("<li><b>Offshore ML against ComCat ML.</b> Our ML sits 0.3-0.5 above ComCat ML at Mendocino, Blanco "
             "and Gorda and agrees onshore; the single anchor offset is set mostly by onshore anchors (2,724 of "
             "3,561). Against tensor Mw our ML is within 0.1 for Mw 3-4.5 (64 events, mostly offshore). Whether the "
             "gap to ComCat ML offshore comes from ComCat's offshore ML or from ours is not resolved here.</li>")
    h.append("<li><b>Poorly constrained depths.</b> floor_limited, far_only, deep-pass and five-phase events get a "
             "magnitude like any other, but at short distance their ML depends on the depth. The flags are in the "
             "table so that users can filter.</li>")
    h.append("</ul>")

    # ---- files
    h.append("<h2>Files</h2>")
    files = [
        ("/wd1/.../v5.1_analysis/mag/cascadia_catalog_v5.1_magnitudes.csv", "one row per v5.1 event (172,342): "
         "M, M_unc, M_type, ML, ML_unc, ML_method, n_ML (picks), n_sta_ML (stations), ML_inv, Mw_mt, comcat_id, "
         "anchor_ml, nearest ANSS event, v4.1 values, fallback ML_v4hyp, flags"),
        ("data/catalog_v5/v5.1_analysis/mag/cascadia_catalog_v5.1_magnitudes_slim.csv.gz", "the magnitude columns of "
         "the table above"),
        ("/wd1/.../mag/cascadia_catalog_M_v51.csv", "phase19 output"),
        ("/wd1/.../mag/phase19_near_picks_v51.parquet, phase19_station_terms_v51.csv", "per-pick station ML, "
         "residuals, station terms"),
        ("/wd1/.../mag/raw_wa_amplitudes_v51.csv, amp_distance_dataset_v51.csv", "selected amplitudes and the SNR-gated dataset"),
        ("data/catalog_v5/v5.1_analysis/mag/mc_b/", "Mc grids (CSV), regional Mc and b, maps"),
        ("data/catalog_v5/v5.1_analysis/mag/*.csv", "the comparison tables shown here"),
    ]
    h.append(tab(pd.DataFrame(files, columns=["file", "content"])))
    h.append("<h3>References</h3><ul class='muted'>"
             "<li>Hutton, L. K., &amp; Boore, D. M. (1987). The ML scale in southern California. BSSA 77(6), 2074-2094.</li>"
             "<li>Aki, K. (1965). Bull. Earthq. Res. Inst. 43, 237-239. Utsu, T. (1965). Geophys. Bull. Hokkaido Univ. 13, 99-103.</li>"
             "<li>Shi, Y., &amp; Bolt, B. A. (1982). BSSA 72(5), 1677-1687. doi:10.1785/BSSA0720051677</li>"
             "<li>Wiemer, S., &amp; Wyss, M. (2000). BSSA 90(4), 859-869. doi:10.1785/0119990114</li>"
             "<li>Cao, A., &amp; Gao, S. S. (2002). GRL 29(9), 1334. doi:10.1029/2001GL013775</li>"
             "<li>Woessner, J., &amp; Wiemer, S. (2005). BSSA 95(2), 684-698. doi:10.1785/0120040007</li></ul>")
    h.append("</body></html>")
    with open(f"{out}/v51_magnitude_report.html", "w") as f:
        f.write("\n".join(h))
    print("wrote", f"{out}/v51_magnitude_report.html")


def img_tag(path, alt):
    return f"<img alt='{alt}' src='data:image/png;base64,{img_file(path)}'>"


if __name__ == "__main__":
    main()

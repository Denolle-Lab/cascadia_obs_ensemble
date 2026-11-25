"""Step 5 of the CC differential times: final tables, hypoDD dt.cc files and quality summaries per set.

Input: <set>/cc_raw.parquet (b5_cc_measure.py --params b5_cc_params.yaml, all pairs given + added).
Output in /wd1/mdenolle_data/tomo_v1/cc_dt:
* cc_<set>.parquet: one row per pair-station-phase (also below the threshold and failed rows, status column),
  with the threshold flag `above` (status ok, peak not at the lag edge, cc_max >= cc_min, cc_max - cc_2nd >= margin_min);
* dt_<set>.cc (given pairs) and dt_<set>_added.cc (added pairs), hypoDD format:
      # ID1 ID2 0.0
      STA DT CC PHA
  DT = dt_cc = (t1 - T0_1) - (t2 - T0_2) with v5.1 origin times (OTC 0), STA = STA.NET of the pick file,
  CC = cc_max; only rows with above = True (status ok, peak not at the lag edge, cc_max >= cc_min,
  cc_max - cc_2nd >= margin_min, same sampling rate for both records). Rows with an un-re-timable pick (D-B072 off_grid / rate_not_100)
  are in these files and flagged in the parquet; dt_<set>_no_unretimable.cc (+ _added) leave them out.
* qc/<set>_*.csv, qc/<set>_qc.png, qc/summary_<set>.json.
Per-row triangle closure (accepted rows): n_tri = closed triangles (same station-phase, all three legs accepted)
the row is a leg of, n_tri_fail20 = of those, how many have |dt_AB + dt_BC - dt_AC| > 20 ms.

  python b5_cc_write.py --set msh [--params b5_cc_params.yaml]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b5_cc_common as C  # noqa: E402
from b5_cc_study import closure, mad  # noqa: E402
from manifest import write_manifest  # noqa: E402

OBS_NETS = {"OO", "NV", "Z5", "7D", "X9", "7A", "ZZ", "YS"}   # seafloor (OBS and cabled seafloor stations)


def iso(t):
    return [C.UTCDateTime(x).isoformat() + "Z" if np.isfinite(x) else "" for x in t]


def build(name, P):
    raw = pd.read_parquet(C.OUT / name / "cc_raw.parquet")
    th = P["threshold"]
    d = raw.copy()
    d.insert(0, "set", name)
    d["net"] = d.station.str.split(".").str[1]
    d["obs_class"] = np.where(d.net.isin(OBS_NETS), "seafloor", "land")
    d["edge"] = d.edge.fillna(False).astype(bool)
    d["margin"] = d.cc_max - d.cc_2nd
    d["rate_mismatch"] = d.rate_mismatch.astype("boolean").fillna(False).astype(bool)
    d["chan_mismatch"] = d.chan_mismatch.astype("boolean").fillna(False).astype(bool)
    # rate_mismatch: the two records have different sampling rates (100 vs 50 Hz: SEP/SUG EHZ digitizer change
    # 2014, XD days with HH 100 Hz only vs BH 50 Hz only): different acquisition filter chains, triangle
    # closure failures 4-7x higher -> not accepted (kept in the parquet)
    d["above"] = ((d.status == "ok") & ~d.edge & (d.cc_max >= th["cc_min"]) & (d.margin >= th["margin_min"])
                  & ~d.rate_mismatch)
    d["filter"] = np.where(d.status == "ok", "Butterworth band-pass 4 poles zero-phase "
                           + d.band_lo.round(2).astype(str) + "-" + d.band_hi.round(2).astype(str) + " Hz", "")
    d["window"] = np.where(d.win_pre.notna(), "[-" + d.win_pre.round(3).astype(str) + ", +"
                           + d.win_post.round(3).astype(str) + "] s around pick", "")
    d["pick_time1"] = iso(d.t_pick1)
    d["pick_time2"] = iso(d.t_pick2)
    d["retimed_any"] = (d.timing1 == "retimed") | (d.timing2 == "retimed")
    d["unretimable_any"] = (d.timing1 == "unretimable") | (d.timing2 == "unretimable")
    d["dt_diff"] = d.dt_cc - d.dt_cat
    # per-measurement triangle closure among the accepted rows (same station-phase): how many closed
    # triangles the row is a leg of, and how many of them miss closure by > 20 ms
    cl = closure(d[d.above])
    legs = []
    for x, y in (("A", "B"), ("B", "Cc"), ("A", "Cc")):
        legs.append(pd.DataFrame({"k1": np.minimum(cl[x], cl[y]), "k2": np.maximum(cl[x], cl[y]),
                                  "station": cl.station, "phase": cl.phase, "fail": cl.e.abs() > 0.02}))
    legs = pd.concat(legs, ignore_index=True)
    lg = legs.groupby(["k1", "k2", "station", "phase"]).fail.agg(n_tri="size", n_tri_fail20="sum").reset_index()
    d["k1"] = np.minimum(d.ID1, d.ID2)
    d["k2"] = np.maximum(d.ID1, d.ID2)
    d = d.merge(lg, on=["k1", "k2", "station", "phase"], how="left").drop(columns=["k1", "k2"])
    d["n_tri"] = d.n_tri.fillna(0).astype(int)
    d["n_tri_fail20"] = d.n_tri_fail20.fillna(0).astype(int)
    d = d.rename(columns={"source": "pair_source", "flag1": "retime_flag1", "flag2": "retime_flag2",
                          "T01": "T0_1", "T02": "T0_2"})
    cols = ["set", "pair", "pair_source", "ID1", "ID2", "sep_km", "station", "net", "obs_class", "phase", "status",
            "above", "cc_max", "cc_2nd", "margin", "edge", "lag_s", "dt_cat", "dt_cc", "dt_diff", "comps",
            "channel1", "channel2", "chan_mismatch", "rate_mismatch", "rate1", "rate2", "rate", "filter", "band_lo", "band_hi",
            "window", "win_pre", "win_post", "maxlag_s", "pick_time1", "pick_time2", "t_pick1", "t_pick2",
            "T0_1", "T0_2", "timing1", "timing2", "retime_flag1", "retime_flag2", "retimed_any", "unretimable_any",
            "snr1", "snr2", "n_tri", "n_tri_fail20", "pick_row1", "pick_row2", "pick_prob1", "pick_prob2"]
    d = d[cols]
    d.to_parquet(C.OUT / f"cc_{name}.parquet", index=False)
    return d


def write_dt(d, path):
    a = d[d.above].sort_values(["pair", "phase", "station"])
    n_pairs = 0
    with open(path, "w") as f:
        for (pr, i1, i2), g in a.groupby(["pair", "ID1", "ID2"], sort=False):
            f.write(f"# {i1} {i2} 0.0\n")
            for r in g.itertuples(index=False):
                f.write(f"{r.station} {r.dt_cc:.5f} {r.cc_max:.4f} {r.phase}\n")
            n_pairs += 1
    return n_pairs, len(a)


def qstats(s):
    r = s.dt_diff
    return dict(n=len(s), cc_median=float(s.cc_max.median()) if len(s) else np.nan,
                diff_median_ms=float(r.median() * 1e3) if len(s) else np.nan, diff_mad_ms=mad(r) * 1e3,
                diff_gt100_frac=float((r.abs() > 0.1).mean()) if len(s) else np.nan)


def repeaters(d, nmin=6, ccmin=0.95):
    """Pairs whose accepted CC dt have median cc >= ccmin over >= nmin station-phases: a repeating source.
    For co-located events the differential travel time is the same at every station (origin-time
    difference); spread = std of dt around the pair mean, CC vs catalog."""
    a = d[d.above]
    g = a.groupby("pair")
    s = g.agg(n=("cc_max", "size"), cc_med=("cc_max", "median"), sep_km=("sep_km", "first"),
              ID1=("ID1", "first"), ID2=("ID2", "first"), dt_cc_mean=("dt_cc", "mean"),
              dt_cc_std=("dt_cc", "std"), dt_cat_mean=("dt_cat", "mean"), dt_cat_std=("dt_cat", "std"))
    s = s[(s.n >= nmin) & (s.cc_med >= ccmin)].sort_values("cc_med", ascending=False)
    return s.reset_index()


def drift_slopes(a, nmax=1500, seed=0):
    """Time-base check of the picks with the CC dt (record-header time base): per station, phase and timing
    class pair, Theil-Sen slope of dt_cc - dt_cat against f1 - f2 (UTC fraction of day of the two picks).
    If the catalog picks of a channel are early by r * f (whole-day count drift at r s/day, as the UW analog
    channels after 2013-10-22: r ~ +0.5 s/day), dt_cat = dt_true - r (f1 - f2) and the slope is +r; picks on
    the header time base give ~0 (data restarts within a day reset the drift, so |slope| is a lower bound)."""
    from scipy.stats import theilslopes
    f1 = (a.t_pick1 % 86400) / 86400
    f2 = (a.t_pick2 % 86400) / 86400
    b = a.assign(df=f1 - f2, cls=np.where(a.timing1 == a.timing2, a.timing1, "mixed"))
    b = b[b.dt_diff.abs() < 0.5]
    rows = []
    rng = np.random.default_rng(seed)
    for (sta, ph, cls), g in b.groupby(["station", "phase", "cls"]):
        if len(g) < 10 or g["df"].std() < 0.05:
            continue
        if len(g) > nmax:
            g = g.iloc[rng.choice(len(g), nmax, replace=False)]
        s, i, lo, hi = theilslopes(g.dt_diff.values, g["df"].values)
        rows.append(dict(station=sta, phase=ph, timing=cls, n=len(g), slope_s_per_day=s, lo95=lo, hi95=hi,
                         intercept_ms=i * 1e3))
    return pd.DataFrame(rows, columns=["station", "phase", "timing", "n", "slope_s_per_day", "lo95", "hi95",
                                       "intercept_ms"])


def qc(name, d, P, nfiles):
    out = C.OUT / "qc"
    out.mkdir(parents=True, exist_ok=True)
    ok = d[d.status == "ok"]
    a = d[d.above]
    S = dict(set=name, rows=len(d), status=d.status.value_counts().to_dict(), threshold=P["threshold"],
             cfg={**P.get("common", {}), **P["sets"][name]}, files=nfiles)
    for src in ("given", "added"):
        dd = d[d.pair_source == src]
        npairs = dd.pair.nunique()
        na = dd[dd.above].groupby("pair").size()
        S[f"pairs_{src}"] = dict(pairs=int(npairs), obs=int(len(dd)), obs_ok=int((dd.status == "ok").sum()),
                                 obs_above=int(dd.above.sum()), ge1=int((na >= 1).sum()), ge4=int((na >= 4).sum()),
                                 ge8=int((na >= 8).sum()),
                                 frac_ge1=float((na >= 1).sum() / max(npairs, 1)),
                                 frac_ge8=float((na >= 8).sum() / max(npairs, 1)),
                                 cc_median_ok=float(dd[dd.status == "ok"].cc_max.median()),
                                 cc_median_above=float(dd[dd.above].cc_max.median()))
    S["above_by_phase"] = {ph: qstats(a[a.phase == ph]) for ph in ("P", "S")}
    S["above_by_class"] = {c: qstats(a[a.obs_class == c]) for c in sorted(a.obs_class.unique())}
    S["ok_by_class"] = {c: dict(n=int((ok.obs_class == c).sum()),
                                frac_above=float(a.obs_class.eq(c).sum() / max((ok.obs_class == c).sum(), 1)))
                        for c in sorted(ok.obs_class.unique())}
    # timing classes (D-B072)
    tc = []
    for cls in ("retimed", "checked", "not_checked", "unretimable"):
        m_all = (d.timing1 == cls) | (d.timing2 == cls)
        tc.append(dict(timing=cls, obs=int(m_all.sum()), ok=int((m_all & (d.status == "ok")).sum()),
                       above=int((m_all & d.above).sum()), **{f"above_{k}": v for k, v in qstats(a[(a.timing1 == cls) | (a.timing2 == cls)]).items()}))
    S["timing"] = tc
    ur = d[d.unretimable_any]
    S["unretimable"] = dict(obs=int(len(ur)), ok=int((ur.status == "ok").sum()), above=int(ur.above.sum()),
                            pairs_with_above=int(ur[ur.above].pair.nunique()),
                            by_station={f"{k[0]}:{k[1]}": int(v) for k, v in ur[ur.above].groupby(["station", "phase"]).size().items()},
                            diff_mad_ms=mad(ur[ur.above].dt_diff) * 1e3 if ur.above.any() else np.nan,
                            diff_median_ms=float(ur[ur.above].dt_diff.median() * 1e3) if ur.above.any() else np.nan,
                            abs_diff_gt100=float((ur[ur.above].dt_diff.abs() > 0.1).mean()) if ur.above.any() else np.nan)
    # closure among accepted
    cl = closure(a.rename(columns={}))
    S["closure"] = {ph: dict(n_tri=int((cl.phase == ph).sum()), mad_ms=mad(cl[cl.phase == ph].e) * 1e3,
                             gt10_frac=float((cl[cl.phase == ph].e.abs() > 0.01).mean()) if (cl.phase == ph).any() else np.nan,
                             gt20_frac=float((cl[cl.phase == ph].e.abs() > 0.02).mean()) if (cl.phase == ph).any() else np.nan)
                    for ph in ("P", "S")}
    # per station-phase table
    st = []
    for (sta, ph), g in ok.groupby(["station", "phase"]):
        ga = g[g.above]
        st.append(dict(station=sta, phase=ph, obs_class=g.obs_class.iloc[0], comps=",".join(sorted(g.comps.dropna().unique())),
                       channel=",".join(sorted(g.channel1.dropna().unique())), rate=g.rate.median(), n_ok=len(g),
                       n_above=len(ga), frac_above=len(ga) / len(g), cc_median=g.cc_max.median(),
                       diff_median_ms=ga.dt_diff.median() * 1e3 if len(ga) else np.nan,
                       diff_mad_ms=mad(ga.dt_diff) * 1e3 if len(ga) else np.nan,
                       timing=",".join(f"{k}:{v}" for k, v in pd.concat([g.timing1, g.timing2]).value_counts().items()),
                       snr_median=float(np.nanmedian(np.r_[g.snr1, g.snr2]))))
    st = pd.DataFrame(st).sort_values(["phase", "n_above"], ascending=[True, False])
    st.to_csv(out / f"{name}_by_station.csv", index=False, float_format="%.4g")
    dr = drift_slopes(a)
    dr.to_csv(out / f"{name}_drift_slopes.csv", index=False, float_format="%.4g")
    S["drift_slopes"] = dr[dr.n >= 30].round(4).to_dict("records")
    rp = repeaters(d)
    rp.to_csv(out / f"{name}_repeaters.csv", index=False, float_format="%.5g")
    S["repeaters"] = dict(n=len(rp), dt_cc_std_median_ms=float(rp.dt_cc_std.median() * 1e3) if len(rp) else np.nan,
                          dt_cat_std_median_ms=float(rp.dt_cat_std.median() * 1e3) if len(rp) else np.nan,
                          examples=rp.head(5)[["ID1", "ID2", "n", "cc_med", "sep_km", "dt_cc_mean", "dt_cc_std",
                                               "dt_cat_std"]].round(4).to_dict("records"))
    # lag distribution of accepted rows against max lag
    S["lag_abs_q"] = {ph: [float(x) for x in np.quantile(a[a.phase == ph].lag_s.abs(), [0.5, 0.9, 0.99])]
                      if (a.phase == ph).any() else [] for ph in ("P", "S")}
    S["edge_frac_ok"] = float(d[d.status == "ok"].edge.mean())
    (out / f"summary_{name}.json").write_text(json.dumps(S, indent=1, default=float))
    # figure
    fig, ax = plt.subplots(2, 2, figsize=(9, 7))
    bins = np.linspace(0, 1, 51)
    for ph, c in (("P", "C0"), ("S", "C3")):
        ax[0, 0].hist(ok[ok.phase == ph].cc_max, bins, histtype="step", color=c, label=f"{ph} all ok")
        ax[0, 0].hist(a[a.phase == ph].cc_max, bins, histtype="stepfilled", alpha=0.3, color=c, label=f"{ph} above")
    ax[0, 0].set_xlabel("cc_max")
    ax[0, 0].legend(fontsize=7)
    m = ok.dt_diff.abs() < 1
    ax[0, 1].hist2d(ok[m].cc_max, ok[m].dt_diff * 1e3, bins=[40, 80], cmap="Greys", cmin=1)
    ax[0, 1].axvline(P["threshold"]["cc_min"], color="r", lw=0.8)
    ax[0, 1].set_xlabel("cc_max")
    ax[0, 1].set_ylabel("dt_cc - dt_cat (ms)")
    ax[0, 1].set_ylim(-500, 500)
    for ph, c in (("P", "C0"), ("S", "C3")):
        e = cl[cl.phase == ph].e * 1e3
        ax[1, 0].hist(np.clip(e, -30, 30), np.linspace(-30, 30, 61), histtype="step", color=c,
                      label=f"{ph} n={len(e)} MAD={mad(e):.2f} ms")
    ax[1, 0].set_xlabel("triangle closure dt_AB + dt_BC - dt_AC (ms), accepted rows")
    ax[1, 0].legend(fontsize=7)
    for ph, c in (("P", "C0"), ("S", "C3")):
        x = a[a.phase == ph].dt_diff * 1e3
        ax[1, 1].hist(np.clip(x, -400, 400), np.linspace(-400, 400, 81), histtype="step", color=c,
                      label=f"{ph} med {np.median(x) if len(x) else np.nan:.1f} MAD {mad(x):.1f} ms")
    ax[1, 1].set_xlabel("dt_cc - dt_cat (ms), accepted rows")
    ax[1, 1].legend(fontsize=7)
    fig.suptitle(f"{name}: {P['sets'][name]['band']} Hz, threshold cc >= {P['threshold']['cc_min']}, "
                 f"margin >= {P['threshold']['margin_min']}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / f"{name}_qc.png", dpi=110)
    plt.close(fig)
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--params", default=str(HERE / "b5_cc_params.yaml"))
    a = ap.parse_args()
    P = yaml.safe_load(Path(a.params).read_text())
    d = build(a.set, P)
    files = {}
    for src, suf in (("given", ""), ("added", "_added")):
        dd = d[d.pair_source == src]
        files[f"dt_{a.set}{suf}.cc"] = write_dt(dd, C.OUT / f"dt_{a.set}{suf}.cc")
        if d.unretimable_any.any():
            files[f"dt_{a.set}{suf}_no_unretimable.cc"] = write_dt(dd[~dd.unretimable_any],
                                                                   C.OUT / f"dt_{a.set}{suf}_no_unretimable.cc")
    S = qc(a.set, d, P, {k: dict(pairs=v[0], lines=v[1]) for k, v in files.items()})
    print(json.dumps({k: S[k] for k in ("pairs_given", "pairs_added", "above_by_phase", "above_by_class", "closure",
                                        "unretimable", "repeaters", "files")}, indent=1, default=float))
    write_manifest(f"b5_cc_write_{a.set}", C.OUT / "qc", inputs=[C.OUT / a.set / "cc_raw.parquet", Path(a.params)],
                   outputs=[C.OUT / f"cc_{a.set}.parquet"] + [C.OUT / k for k in files],
                   params=P, counts={k: S[k] for k in ("pairs_given", "pairs_added", "files")})


if __name__ == "__main__":
    main()

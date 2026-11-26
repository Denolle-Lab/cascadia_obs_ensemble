"""P1: clock errors at MH07.XD and MG07.XD (CC-dt QC, /wd1/mdenolle_data/tomo_v1/cc_dt/README.md: about -0.45 s at
MH07.XD and +0.28 s at MG07.XD on some days of 2015-11-21 to 2016-05-02, in the record headers, so the re-timed picks
carry them). Do the v5.1 NonLinLoc residuals at these stations show the offsets, and which deep / v1_ON028 / v1x_06
events use them in that period?

Reads the per-event .hyp files of every Mount St Helens box event located in 2015-09-01 to 2016-08-31.
Outputs: WORK/p1_clock/xd_residuals.csv, clock_summary.json; OUT/P1_msh_deep/ctx/xd_clock.webp.
"""
import json
import os

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import rv_common as C  # noqa: E402

W = C.WORK / "p1_clock"
P0, P1 = pd.Timestamp("2015-11-21", tz="UTC"), pd.Timestamp("2016-05-03", tz="UTC")
BAD = {"MH07.XD": -0.45, "MG07.XD": +0.28}


def collect():
    f = W / "xd_residuals.csv"
    if f.exists():
        return pd.read_csv(f, parse_dates=["t"])
    c = C.catalog()
    m = (c.loc_source.eq("nll3d") & c.lat.between(*C.MSH_BOX["lat"]) & c.lon.between(*C.MSH_BOX["lon"])
         & (c.t >= pd.Timestamp("2015-09-01", tz="UTC")) & (c.t < pd.Timestamp("2016-09-01", tz="UTC")))
    rows = []
    for eid, r in c[m].iterrows():
        h, _, wd = C.find_hyp(eid, r)
        if h is None:
            h = C.sum_block(eid, wd)
        if h is None:
            continue
        P = C.parse_hyp(h)["phases"]
        X = P[P.sta.str.endswith(".XD")]
        for p in X.itertuples():
            rows.append(dict(ID=eid, t=r.t, depth=r.depth, tier=r.tier, sta=p.sta, phase=p.phase, res=p.res,
                             weight=p.weight, dist=p.sdist))
    df = pd.DataFrame(rows)
    W.mkdir(parents=True, exist_ok=True)
    df.to_csv(f, index=False)
    return pd.read_csv(f, parse_dates=["t"])


def main():
    W.mkdir(parents=True, exist_ok=True)
    df = collect()
    df["day"] = df.t.dt.floor("D")
    inper = (df.t >= P0) & (df.t < P1)
    # per station: median residual in / out of the period; per day medians at the two stations and at all others
    out = {"period": [str(P0.date()), str(P1.date())], "stations": {}}
    for sta in sorted(df.sta.unique()):
        x = df[df.sta == sta]
        out["stations"][sta] = dict(n_in=int((inper & (df.sta == sta)).sum()), n_out=int((~inper & (df.sta == sta)).sum()),
                                    med_in=round(float(x[inper[x.index]].res.median()), 3) if (inper & (df.sta == sta)).any() else None,
                                    med_out=round(float(x[~inper[x.index]].res.median()), 3) if (~inper & (df.sta == sta)).any() else None)
    daily = df.groupby(["sta", "day"]).agg(n=("res", "size"), med=("res", "median")).reset_index()
    ref = df[~df.sta.isin(BAD)].groupby("day").res.median().rename("ref_med")
    daily = daily.merge(ref, on="day", how="left")
    daily["rel"] = daily.med - daily.ref_med
    for sta, off in BAD.items():
        d = daily[(daily.sta == sta) & (daily.n >= 2)]
        dd = d[(d.day >= P0) & (d.day < P1)]
        big = dd[np.abs(dd.rel) > 0.25]           # sign-free: the CC-dt values are pair differences
        out["stations"].setdefault(sta, {}).update(
            days_with_2plus_picks_in_period=int(len(dd)), days_abs_offset_gt_0p25=int(len(big)),
            offset_day_range=[str(big.day.min().date()), str(big.day.max().date())] if len(big) else None,
            offset_median_on_those_days=round(float(big.rel.median()), 3) if len(big) else None,
            days_abs_offset_lt_0p1=int((np.abs(dd.rel) < 0.1).sum()),
            offset_days=[f"{x.day.date()} {x.rel:+.2f}" for x in big.itertuples()][:60])
    # populations
    c = C.catalog()
    mem = pd.read_csv(C.FAULTS / "fault_surfaces_v1_members.csv")
    rem = pd.read_csv(C.FAULTS / "fault_surfaces_v1_removed_members.csv")
    pops = {"deep (> 30 km) box events": set(c.index[c.loc_source.eq("nll3d") & c.lat.between(*C.MSH_BOX["lat"])
                                                       & c.lon.between(*C.MSH_BOX["lon"]) & (c.depth > 30)]),
            "v1_ON028 members": set(mem[mem.surface_id == "v1_ON028"].event_id),
            "v1x_06 members (removed surface)": set(rem[rem.removed_id == "v1x_06"].event_id)}
    out["populations"] = {}
    for lab, ids in pops.items():
        e = c.loc[sorted(ids & set(c.index))]
        ein = e[(e.t >= P0) & (e.t < P1)]
        use = df[df.ID.isin(ein.ID) & df.sta.isin(BAD)]
        out["populations"][lab] = dict(n=int(len(e)), n_in_period=int(len(ein)),
                                       n_using_MH07_or_MG07=int(use.ID.nunique()),
                                       years=e.t.dt.year.value_counts().sort_index().to_dict(),
                                       residuals=[dict(ID=int(r.ID), sta=r.sta, phase=r.phase, res=round(float(r.res), 2),
                                                       weight=round(float(r.weight), 2)) for r in use.itertuples()][:40])
    (W / "clock_summary.json").write_text(json.dumps(out, indent=1, default=str))
    # figure
    fig, axs = plt.subplots(2, 1, figsize=(15, 6.8), sharex=True)
    for ax, sta in zip(axs, BAD):
        x = df[df.sta == sta]
        o = df[(~df.sta.isin(BAD))]
        ax.plot(o.t, o.res, ".", ms=1.5, color="#c9c8c3", label="other XD stations (all picks)", zorder=1)
        for ph, col, mk in (("P", "#2a78d6", "o"), ("S", "#eb6834", "s")):
            y = x[x.phase == ph]
            ax.plot(y.t, y.res, mk, ms=3, color=col, alpha=0.8, label=f"{sta} {ph}", zorder=3)
        ax.axhline(BAD[sta], color="#0b0b0b", lw=0.8, ls="--", label=f"CC-dt pair value {BAD[sta]:+.2f} s (sign depends on pair order)")
        ax.axhline(-BAD[sta], color="#0b0b0b", lw=0.8, ls="--")
        ax.axhline(0, color="#6b6a66", lw=0.5)
        ax.axvspan(P0, P1, color="#eef3fb", zorder=0, label="2015-11-21 to 2016-05-02")
        ax.set_ylim(-1.2, 1.2)
        ax.set_ylabel("NonLinLoc residual (s)")
        s = out["stations"].get(sta, {})
        ax.set_title(f"{sta}: residuals of Mount St Helens box events (v5.1); in the period median {s.get('med_in')} s "
                     f"(n={s.get('n_in')}), outside {s.get('med_out')} s (n={s.get('n_out')})", loc="left", fontsize=9)
        ax.grid(color="#d9d8d4", lw=0.4)
        ax.legend(fontsize=7, loc="lower left", ncol=5)
    fig.tight_layout()
    d = C.OUT / "P1_msh_deep" / "ctx"
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / "xd_clock.webp", dpi=90, format="webp", pil_kwargs={"quality": 88})
    plt.close(fig)
    print(json.dumps({k: v for k, v in out.items() if k != "populations"}, indent=1)[:3000])
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "residuals"} for k, v in out["populations"].items()}, indent=1))


if __name__ == "__main__":
    main()

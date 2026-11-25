"""Problem-level context figures of the v5.1 location review (written to OUT/<folder>/ctx/).

P1 time-of-week and hour-of-day histograms of the Mount St Helens box events by depth class
P2 v4 vs v5.1 depths of the v1_GO022 members, map with stations, S-P against depth at the nearest stations
P3 v5.1 minus ComCat depth, epicentre and origin-time differences of the tensor events
P4 v4 vs v5.1 depth histograms by region for deep_pass and far_only events
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import rv_common as C  # noqa: E402

B1, B2, B3, B4 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, MUTED, GRID = "#0b0b0b", "#6b6a66", "#d9d8d4"
SAVE = dict(format="webp", pil_kwargs={"quality": 90})
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED,
                     "ytick.color": MUTED, "legend.frameon": False, "savefig.facecolor": "white"})


def _sets(name):
    return pd.read_csv(C.WORK / "sets" / f"{name}.csv")


# ------------------------------------------------------------------------------------------------ P1
def p1_week(out):
    c = C.catalog()
    m = c.loc_source.eq("nll3d") & c.lat.between(*C.MSH_BOX["lat"]) & c.lon.between(*C.MSH_BOX["lon"])
    e = c[m].copy()
    e["how"] = e.t.dt.dayofweek * 24 + e.t.dt.hour + e.t.dt.minute / 60
    e["hod"] = e.t.dt.hour + e.t.dt.minute / 60
    lt = e.t.dt.tz_convert("America/Los_Angeles")
    e["hod_local"] = lt.dt.hour + lt.dt.minute / 60
    classes = [("depth < 10 km", e.depth < 10), ("10-30 km", e.depth.between(10, 30)), ("> 30 km", e.depth > 30)]
    fig, axs = plt.subplots(3, 3, figsize=(16, 7.2), gridspec_kw=dict(width_ratios=[2.6, 1, 1]))
    for i, (lab, mm) in enumerate(classes):
        x = e[mm]
        ax = axs[i, 0]
        ax.hist(x.how, bins=np.arange(0, 169, 1), color=B1, edgecolor="white", linewidth=0.3)
        ax.axvspan(2 * 24 + 19, 2 * 24 + 20, color=B2, alpha=0.25, lw=0)
        for d in range(1, 7):
            ax.axvline(d * 24, color=MUTED, lw=0.5)
        ax.set_xlim(0, 168)
        ax.set_xticks(np.arange(12, 168, 24))
        ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        ax.set_ylabel(f"{lab}\n(n={len(x)}) events per hour")
        ax.grid(axis="y", color=GRID, lw=0.4)
        axs[i, 1].hist(x.hod, bins=np.arange(0, 25, 1), color=B1, edgecolor="white", linewidth=0.3)
        axs[i, 1].axvspan(19, 20, color=B2, alpha=0.25, lw=0)
        axs[i, 2].hist(x.hod_local, bins=np.arange(0, 25, 1), color=B3, edgecolor="white", linewidth=0.3)
        for ax in axs[i, 1:]:
            ax.set_xlim(0, 24)
            ax.set_xticks([0, 6, 12, 18, 24])
            ax.grid(axis="y", color=GRID, lw=0.4)
    axs[0, 0].set_title("time of week (UTC, 1-hour bins); shaded: Wednesday 19-20 UTC", loc="left")
    axs[0, 1].set_title("hour of day, UTC", loc="left")
    axs[0, 2].set_title("hour of day, local (PST/PDT)", loc="left")
    axs[2, 0].set_xlabel("day of week (UTC)")
    axs[2, 1].set_xlabel("hour (UTC)")
    axs[2, 2].set_xlabel("hour (local)")
    fig.suptitle(f"Mount St Helens box (46.05-46.30N, 122.35-122.10W), {len(e)} located events by v5.1 depth; "
                 "each row has its own count scale", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=90, **SAVE)
    plt.close(fig)
    deep = e[e.depth > 30]
    return dict(n=len(e), n_deep=len(deep),
                deep_dow=deep.t.dt.dayofweek.value_counts().sort_index().to_dict(),
                deep_hour=deep.t.dt.hour.value_counts().sort_index().to_dict(),
                deep_wed_19=int(((deep.t.dt.dayofweek == 2) & (deep.t.dt.hour == 19)).sum()),
                shallow_wed_19=int(((e.depth < 10) & (e.t.dt.dayofweek == 2) & (e.t.dt.hour == 19)).sum()))


def p1_pops():
    """Population contrast: deep box events, v1_ON028, v1x_06, shallow tier A-B box events."""
    import rv_select as S
    c = C.catalog()
    ps = S.pick_stats().set_index("ID")
    box = c.loc_source.eq("nll3d") & c.lat.between(*C.MSH_BOX["lat"]) & c.lon.between(*C.MSH_BOX["lon"])
    mem = pd.read_csv(C.FAULTS / "fault_surfaces_v1_members.csv")
    rem = pd.read_csv(C.FAULTS / "fault_surfaces_v1_removed_members.csv")
    pops = [("box events deeper than 30 km", c.index[box & (c.depth > 30)]),
            ("v1_ON028 members", mem[mem.surface_id == "v1_ON028"].event_id.values),
            ("v1x_06 members (removed)", rem[rem.removed_id == "v1x_06"].event_id.values),
            ("box events 10-30 km", c.index[box & c.depth.between(10, 30)]),
            ("box events shallower than 10 km, tier A-B", c.index[box & (c.depth < 10) & c.tier.isin(["A", "B"])])]
    rows = []
    for lab, ids in pops:
        e = c.loc[[i for i in ids if i in c.index]]
        x = ps.reindex(e.index)
        t = e.t
        wed = (t.dt.dayofweek == 2) & (t.dt.hour == 19) & t.dt.minute.between(14, 20)
        wd19 = (t.dt.dayofweek == 2) & (t.dt.hour == 19)
        span_ok = x.P_dmax_km > 8
        rows.append(dict(population=lab, n=int(len(e)), depth_med=round(float(e.depth.median()), 1),
                         wed_window=int(wed.sum()), wed_19h=int(wd19.sum()),
                         weekday_daytime_pct=round(100 * float(((t.dt.tz_convert("America/Los_Angeles").dt.dayofweek < 5)
                                                               & t.dt.tz_convert("America/Los_Angeles").dt.hour.between(8, 16)).mean()), 0),
                         median_P=float(x.n_P.median()), median_S=float(x.n_S.median()),
                         pct_no_S=round(100 * float((x.n_S == 0).mean()), 0),
                         P_span_med=round(float(x.P_span_s[span_ok].median()), 2),
                         pct_P_span_lt_0p5=round(100 * float((x.P_span_s[span_ok] < 0.5).mean()), 0),
                         pct_P_at_UW=round(100 * float(np.mean([st.endswith(".UW") for v in x.P_stations.dropna() for st in v.split()])), 0),
                         n_span=int(span_ok.sum())))
    return rows


# ------------------------------------------------------------------------------------------------ P2
def _event_json(eid):
    f = C.WORK / "events" / f"{eid}.json"
    return json.loads(f.read_text()) if f.exists() else None


def p2_figs(out1, out2):
    v = _sets("P2_views")
    c = C.catalog()
    S = C.stations()
    rows = []
    for eid in v.ID:
        J = _event_json(eid)
        if J is None:
            continue
        P = pd.DataFrame(J["phases"])
        P["ID"] = eid
        rows.append(P)
    A = pd.concat(rows)
    A["tobs_ts"] = pd.to_datetime(A.tobs, utc=True, format="ISO8601")
    # ---- figure 1: depths, map, time
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.0), gridspec_kw=dict(width_ratios=[1, 1.25, 1.3]))
    ax = axs[0]
    mk = {"A": "*", "B": "o", "C": "s", "D": "^"}
    for t_, g in v.groupby("tier"):
        ax.plot(g.v4_depth, g.depth, mk.get(t_, "o"), color=B1, ms=7 if t_ != "A" else 11, mec="white", mew=0.5,
                label=f"tier {t_} ({len(g)})")
    lim = [min(-2, v.v4_depth.min() - 2), max(70, v.depth.max() + 3)]
    ax.plot(lim, lim, color=MUTED, lw=0.7, ls="--", label="1:1")
    ax.set_xlim(lim); ax.set_ylim(lim[::-1][::-1])
    ax.invert_yaxis()
    ax.set_xlabel("catalog v4 depth (km)")
    ax.set_ylabel("catalog v5.1 depth (km)")
    sl = float(C.slab2_depth(np.array([v.lon.median()]), np.array([v.lat.median()]))[0])
    ax.axhline(sl, color=B4, lw=1.0, ls="-.")
    ax.axvline(sl, color=B4, lw=1.0, ls="-.", label=f"Slab2 slab top at the centroid ({sl:.1f} km)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(color=GRID, lw=0.4)
    ax.set_title(f"(a) {len(v)} members of v1_GO022: v4 vs v5.1 depth", loc="left")
    # map
    ax = axs[1]
    cnt = A.groupby(["sta", "phase"]).size().unstack(fill_value=0)
    cnt["lat"], cnt["lon"] = S.lat.reindex(cnt.index), S.lon.reindex(cnt.index)
    lo0, lo1, la0, la1 = -125.6, -122.3, 40.4, 44.6
    lo, la = np.meshgrid(np.linspace(lo0, lo1, 250), np.linspace(la0, la1, 250))
    h = C.surface_km(lo, la)
    ax.contour(lo, la, h, levels=[0], colors=[MUTED], linewidths=0.6)
    sz = 10 + 6 * cnt.get("P", 0)
    ax.scatter(cnt.lon, cnt.lat, s=np.clip(sz, 10, 260), marker="^", c="white", edgecolors=INK, linewidths=0.7, zorder=3,
               label="station (size: number of member P picks)")
    for st, r in cnt.iterrows():
        if lo0 < r.lon < lo1 and la0 < r.lat < la1 and (r.get("P", 0) + r.get("S", 0)) >= 10:
            ax.annotate(f"{st} {int(r.get('P', 0))}P {int(r.get('S', 0))}S", (r.lon, r.lat), xytext=(4, -3),
                        textcoords="offset points", fontsize=6.3, color=INK)
    ax.plot(v.v4_lon, v.v4_lat, "s", color=B2, ms=4, label="v4 epicentres", zorder=4)
    ax.plot(v.lon, v.lat, "o", color=B1, ms=4, label="v5.1 epicentres", zorder=5)
    ax.set_xlim(lo0, lo1); ax.set_ylim(la0, la1)
    ax.set_aspect(1 / np.cos(np.radians(42.3)))
    ax.grid(color=GRID, lw=0.4)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("(b) members and the stations that picked them", loc="left")
    # time
    ax = axs[2]
    tt = pd.to_datetime(v.time, format="ISO8601")
    ax.plot(tt, v.v4_depth, "s", color=B2, ms=5, label="v4")
    ax.plot(tt, v.depth, "o", color=B1, ms=5, label="v5.1")
    for a_, b_, d_ in zip(tt, v.v4_depth, v.depth):
        ax.plot([a_, a_], [b_, d_], color=GRID, lw=0.8, zorder=0)
    ax.axhline(sl, color=B4, lw=1.0, ls="-.", label="Slab2 slab top")
    ax.invert_yaxis()
    ax.set_ylabel("depth (km)")
    ax.legend(fontsize=7)
    ax.grid(color=GRID, lw=0.4)
    ax.set_title("(c) depth against time", loc="left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out1, dpi=90, **SAVE)
    plt.close(fig)
    # ---- figure 2: S-P vs depth at the nearest stations
    lat0, lon0 = float(v.lat.median()), float(v.lon.median())
    x0, y0 = C.lambert(lon0, lat0)
    both = []
    for (eid, st), g in A.groupby(["ID", "sta"]):
        if set(g.phase) >= {"P", "S"}:
            p_, s_ = g[g.phase == "P"].iloc[0], g[g.phase == "S"].iloc[0]
            both.append(dict(ID=eid, sta=st, dist=p_.dist, sp=(s_.tobs_ts - p_.tobs_ts).total_seconds(),
                             wP=p_.weight, wS=s_.weight, resP=p_.res, resS=s_.res))
    B = pd.DataFrame(both)
    near = B.groupby("sta").agg(n=("ID", "size"), dist=("dist", "median"), sp_med=("sp", "median"),
                                sp_q1=("sp", lambda s: s.quantile(0.25)), sp_q3=("sp", lambda s: s.quantile(0.75)))
    near = near[near.n >= 3].sort_values("dist").head(5)
    zps = np.arange(-2.0, 100.0, 1.0)
    tc = C.station_terms()
    fig, ax = plt.subplots(figsize=(11, 5.2))
    cols = [B1, B2, B3, B4, "#e34948"]
    res = []
    for k, (st, r) in enumerate(near.iterrows()):
        tp = C.tt_profile("P", st, x0, y0, zps) + tc.get((st, "P"), 0)
        ts = C.tt_profile("S", st, x0, y0, zps) + tc.get((st, "S"), 0)
        z = C.ieft(zps)
        ax.plot(z, ts - tp, color=cols[k], lw=1.6, label=f"{st} ({r.dist:.0f} km): predicted S-P below the median v5.1 epicentre")
        ax.axhspan(r.sp_q1, r.sp_q3, color=cols[k], alpha=0.18, lw=0)
        ax.axhline(r.sp_med, color=cols[k], lw=0.9, ls="--")
        ax.text(z.max(), r.sp_med, f" {st} observed median {r.sp_med:.2f} s (n={r.n})", color=cols[k], fontsize=7,
                va="center")
        sp = ts - tp
        ok = np.isfinite(sp)
        cross = [float(z[ok][i]) for i in range(1, ok.sum()) if (sp[ok][i - 1] - r.sp_med) * (sp[ok][i] - r.sp_med) <= 0]
        res.append(dict(sta=st, dist=round(float(r.dist), 1), n=int(r.n), sp_obs_med=round(float(r.sp_med), 2),
                        sp_obs_iqr=[round(float(r.sp_q1), 2), round(float(r.sp_q3), 2)],
                        sp_pred_at_v51_median_depth=round(float(np.interp(v.depth.median(), z[ok], sp[ok])), 2),
                        sp_pred_at_v4_median_depth=round(float(np.interp(v.v4_depth.median(), z[ok], sp[ok])), 2),
                        depth_where_pred_equals_obs=[round(x, 1) for x in cross],
                        share_weight_lt_0p1=round(float(((B.sta == st) & ((B.wP < 0.1) | (B.wS < 0.1))).sum() / max(1, (B.sta == st).sum())), 2)))
    ax.axvspan(v.depth.min(), v.depth.max(), color=B1, alpha=0.07, lw=0)
    ax.axvline(v.depth.median(), color=B1, lw=1.0, label=f"v5.1 member depths (median {v.depth.median():.1f} km, shaded: range)")
    ax.axvline(v.v4_depth.median(), color=B2, lw=1.0, ls="--", label=f"v4 median depth {v.v4_depth.median():.1f} km")
    ax.set_xlim(-2, 100)
    ax.set_xlabel("source depth below the median v5.1 epicentre (km)")
    ax.set_ylabel("S-P (s), station terms included")
    ax.grid(color=GRID, lw=0.4)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("S-P at the nearest stations with P and S picks of the members: v5.1 grids (lines) against observed "
                 "(dashed: median; band: inter-quartile range)", loc="left", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 0.83, 1))
    fig.savefig(out2, dpi=90, **SAVE)
    plt.close(fig)
    return dict(sp=res, n_pairs=len(B))


# ------------------------------------------------------------------------------------------------ P3
def p3_fig(out):
    v = _sets("P3_views")
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw=dict(width_ratios=[1.4, 1, 1]))
    regs = v.groupby("region").dz_cc.median().sort_values().index.tolist()
    ax = axs[0]
    for i, r in enumerate(regs):
        g = v[v.region == r]
        jit = (np.random.default_rng(i).random(len(g)) - 0.5) * 0.35
        ax.plot(g.dz_cc, i + jit, "o", color=B1, ms=4, alpha=0.75, mec="white", mew=0.3)
        mv = g[g.mover]
        ax.plot(mv.dz_cc, i + jit[g.mover.values], "o", ms=7, mfc="none", mec=B2, mew=1.4)
        ax.plot([g.dz_cc.median()] * 2, [i - 0.3, i + 0.3], color=INK, lw=1.6)
        ax.text(ax.get_xlim()[1] if False else 75, i, f" n={len(g)}, median {g.dz_cc.median():+.1f}", va="center", fontsize=7)
    ax.set_yticks(range(len(regs)))
    ax.set_yticklabels(regs)
    ax.axvline(0, color=MUTED, lw=0.7)
    ax.set_xlim(-30, 100)
    ax.set_xlabel("v5.1 minus ComCat depth (km)")
    ax.grid(axis="x", color=GRID, lw=0.4)
    ax.set_title("(a) depth difference by region (black bar: median; orange ring: large mover)", loc="left")
    ax = axs[1]
    ax.plot(v.dh_cc, v.dz_cc, "o", color=B1, ms=4, alpha=0.75)
    mv = v[v.mover]
    ax.plot(mv.dh_cc, mv.dz_cc, "o", ms=8, mfc="none", mec=B2, mew=1.4, label="large movers (v4.1 tensor carried)")
    for r in mv.itertuples():
        ax.annotate(str(r.ID), (r.dh_cc, r.dz_cc), xytext=(4, 2), textcoords="offset points", fontsize=6.5)
    ax.axhline(0, color=MUTED, lw=0.7)
    ax.set_xlabel("epicentre distance v5.1 - ComCat (km)")
    ax.set_ylabel("v5.1 minus ComCat depth (km)")
    ax.grid(color=GRID, lw=0.4)
    ax.legend(fontsize=7)
    ax.set_title("(b) epicentre vs depth difference", loc="left")
    ax = axs[2]
    ax.hist(v.dt_cc.clip(-13, 13), bins=np.arange(-13, 13.5, 0.5), color=B1, edgecolor="white", lw=0.3)
    ax.axvline(v.dt_cc.median(), color=INK, lw=1.2)
    ax.set_xlabel("v5.1 minus ComCat origin time (s), clipped at +-13 s")
    ax.set_ylabel("events")
    ax.grid(axis="y", color=GRID, lw=0.4)
    ax.set_title(f"(c) origin time difference (median {v.dt_cc.median():+.2f} s)", loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=90, **SAVE)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ P4
def p4_fig(out):
    t = _sets("P4_table")
    fig, axs = plt.subplots(2, 6, figsize=(17, 6.2), sharex=True)
    stats = {}
    for i, flag in enumerate(("deep_pass", "far_only")):
        x = t[t.flag == flag]
        vc = x.region.value_counts()
        regs = [r for r in vc.index if vc[r] >= 10][:5]
        groups = [(r, x[x.region == r]) for r in regs] + [("other regions", x[~x.region.isin(regs)])]
        stats[flag] = {}
        for j, (r, g) in enumerate(groups):
            ax = axs[i, j]
            bins = np.arange(-5, 125, 5)
            ax.hist(g.v4_depth.clip(-5, 124), bins=bins, histtype="step", color=B2, lw=1.6, ls="--", label="v4")
            ax.hist(g.depth.clip(-5, 124), bins=bins, histtype="step", color=B1, lw=1.6, label="v5.1")
            ax.set_title(f"{flag}: {r} (n={len(g)})\nmedian v4 {g.v4_depth.median():.0f}, v5.1 {g.depth.median():.0f} km",
                         loc="left", fontsize=8)
            ax.grid(color=GRID, lw=0.4)
            stats[flag][r] = dict(n=int(len(g)), v4_med=round(float(g.v4_depth.median()), 1), v51_med=round(float(g.depth.median()), 1))
            if i == 1:
                ax.set_xlabel("depth (km)")
    axs[0, 0].legend(fontsize=7)
    axs[0, 0].set_ylabel("events")
    axs[1, 0].set_ylabel("events")
    fig.suptitle("v4 (dashed) vs v5.1 (solid) depths of deep_pass and far_only events by region (5 km bins)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=90, **SAVE)
    plt.close(fig)
    return stats


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["P1", "P2", "P3", "P4"]
    res = {}
    if "P1" in what:
        d = C.OUT / "P1_msh_deep" / "ctx"; d.mkdir(parents=True, exist_ok=True)
        res["P1"] = p1_week(d / "week_hist.webp")
        res["P1"]["pops"] = p1_pops()
    if "P2" in what:
        d = C.OUT / "P2_brookings" / "ctx"; d.mkdir(parents=True, exist_ok=True)
        res["P2"] = p2_figs(d / "depths_map.webp", d / "sp_depth.webp")
    if "P3" in what:
        d = C.OUT / "P3_tensor" / "ctx"; d.mkdir(parents=True, exist_ok=True)
        p3_fig(d / "comcat_diff.webp")
    if "P4" in what:
        d = C.OUT / "P4_deep_far" / "ctx"; d.mkdir(parents=True, exist_ok=True)
        res["P4"] = p4_fig(d / "depth_hist.webp")
    f = C.WORK / "context_stats.json"
    old = json.loads(f.read_text()) if f.exists() else {}
    old.update(res)
    f.write_text(json.dumps(old, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))

"""Event sets of the four review reports (tables written to data/review_v51/sets/ in the repo and WORK/sets/).

P1 Mount St Helens deep events: box 46.05-46.30N, 122.35-122.10W, loc_source nll3d (faults piece, v1_report.py), deeper
   than 30 km. Views: every event of the box on a Wednesday 19:14-19:20 UTC (any depth), ~40 other deep events
   stratified by depth and tier (all tier A-B; tier C with sig_z < 14 km and nearest station < 4 km included), 10 shallow
   tier A-B reference events (depth < 10 km, M 1.0-2.7, 2011-2013).
P2 Brookings 2012 swarm: members of fault surface v1_GO022 (faults piece fault_surfaces_v1_members.csv); every member.
P3 ComCat moment tensors: the 139 v5.1 matches (comcat_mt_matched_v51.csv) and the located v4.1-carried matches
   (mt_match v4.1 in the magnitude table), the 8 large movers of the mag piece first.
P4 deep_pass and far_only events: ~40 of each, stratified by region (at least 2 per region where available) and depth
   (spread over the depth quantiles of the region), events already shown in P1-P3 avoided where possible.
"""
import json

import numpy as np
import pandas as pd

import rv_common as C

SETS_REPO = C.HERE.parent.parent.parent / "data" / "review_v51" / "sets"
SETS_WORK = C.WORK / "sets"
SEED = 51


def pick_stats():
    """Per event: n_P, n_S (pick file), P-pick time span (s) and station distance range of the P picks (v5.1 epicentre)."""
    f = C.WORK / "pick_stats.parquet"
    if f.exists():
        return pd.read_parquet(f)
    p = pd.read_csv(C.PICKS, skipinitialspace=True, usecols=["event_ID", "pick_datetime", "station", "phase"])
    c = C.catalog()
    S = C.stations()
    p = p[p.event_ID.isin(c.index[c.loc_source.eq("nll3d")])]
    p["t"] = pd.to_datetime(p.pick_datetime, utc=True, format="ISO8601")
    p["slat"] = p.station.map(S.lat)
    p["slon"] = p.station.map(S.lon)
    p["elat"] = p.event_ID.map(c.lat)
    p["elon"] = p.event_ID.map(c.lon)
    p["d"] = C.hav(p.elat.values, p.elon.values, p.slat.values, p.slon.values)
    g = p.groupby("event_ID")
    out = pd.DataFrame({"n_P": g.phase.apply(lambda s: int((s == "P").sum())),
                        "n_S": g.phase.apply(lambda s: int((s == "S").sum()))})
    pp = p[p.phase == "P"].groupby("event_ID")
    out["P_span_s"] = (pp.t.max() - pp.t.min()).dt.total_seconds()
    out["P_dmin_km"] = pp.d.min()
    out["P_dmax_km"] = pp.d.max()
    out["P_stations"] = pp.station.apply(lambda s: " ".join(sorted(set(s))))
    out.index.name = "ID"
    out = out.reset_index()
    out.to_parquet(f)
    return out


def strat(df, by_cols, n_total, rng, min_per=2):
    """Stratified sample: groups by by_cols, allocation proportional to sqrt(size) with a minimum, spread in depth."""
    g = df.groupby(by_cols)
    sizes = g.size()
    w = np.sqrt(sizes)
    alloc = np.maximum(np.minimum(sizes, min_per), np.round(n_total * w / w.sum()).astype(int))
    alloc = np.minimum(alloc, sizes)
    picks = []
    for k, n in alloc.items():
        x = g.get_group(k).sort_values("depth")
        if n >= len(x):
            picks.append(x)
            continue
        idx = np.unique(np.linspace(0, len(x) - 1, int(n)).round().astype(int))
        picks.append(x.iloc[idx])
    return pd.concat(picks)


def main():
    SETS_REPO.mkdir(parents=True, exist_ok=True)
    SETS_WORK.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    c = C.catalog().copy()
    c = c[c.loc_source.eq("nll3d")]
    ps = pick_stats().set_index("ID")
    c = c.join(ps, how="left")
    t = c.t
    c["dow"] = t.dt.dayofweek
    c["wed_window"] = (t.dt.dayofweek == 2) & (t.dt.hour == 19) & t.dt.minute.between(14, 20)
    c["tue_window"] = (t.dt.dayofweek == 1) & (t.dt.hour == 19) & t.dt.minute.between(14, 20)
    keep_cols = ["ID", "time", "lat", "lon", "depth", "exp_depth", "tier", "sig_z", "h_min", "h_max", "nphs", "n_sta",
                 "gap", "rms", "dist_min", "depth_quality", "event_flag", "floor_limited", "far_only", "deep_pass",
                 "five_phase", "region", "M", "M_type", "v4_lat", "v4_lon", "v4_depth", "v4_time", "shift_h_km",
                 "shift_z_km", "comcat_id", "n_P", "n_S", "P_span_s", "P_dmin_km", "P_dmax_km", "P_stations",
                 "wed_window", "tue_window"]
    summary = {}
    # ---------------------------------------------------------------- P1
    box = c.lat.between(*C.MSH_BOX["lat"]) & c.lon.between(*C.MSH_BOX["lon"])
    msh = c[box].copy()
    deep = msh[msh.depth > 30]
    v_wed = msh[msh.wed_window].assign(view_group="Wednesday 19:14-19:20 UTC")
    rest = deep[~deep.wed_window].copy()
    rest["zbin"] = pd.cut(rest.depth, [30, 45, 60, 80, 100, 200], labels=["30-45", "45-60", "60-80", "80-100", ">100"])
    ab = rest[rest.tier.isin(["A", "B"])]
    opt = rest[(rest.tier == "C") & (rest.sig_z < 14) & (rest.dist_min < 4)]
    opt = opt.sort_values("sig_z").iloc[np.unique(np.linspace(0, len(opt) - 1, min(6, len(opt))).round().astype(int))]
    pool = rest[~rest.ID.isin(ab.ID) & ~rest.ID.isin(opt.ID) & rest.tier.isin(["C", "D"])]
    smp = strat(pool, ["zbin", "tier"], 30, rng, min_per=2)
    v_deep = pd.concat([ab.assign(view_group="deep, tier A-B"),
                        opt.assign(view_group="deep, tier C with sig_z < 14 km and nearest station < 4 km"),
                        smp.assign(view_group="deep, stratified by depth and tier")])
    sh = msh[(msh.depth < 10) & msh.tier.isin(["A", "B"]) & msh.M.between(1.0, 2.7) & msh.t.dt.year.between(2011, 2013)]
    sh = sh.sort_values("time")
    v_sh = sh.iloc[np.unique(np.linspace(0, len(sh) - 1, 10).round().astype(int))].assign(view_group="shallow reference (tier A-B, < 10 km)")
    p1v = pd.concat([v_wed, v_deep, v_sh]).drop_duplicates("ID")
    p1v["problem"] = "P1"
    p1t = deep.copy()
    p1t["has_view"] = p1t.ID.isin(p1v.ID)
    summary["P1"] = dict(box_located=int(len(msh)), deep=int(len(deep)), deep_tiers=deep.tier.value_counts().to_dict(),
                         deep_sigz_med=float(deep.sig_z.median()), deep_depth_range=[float(deep.depth.min()), float(deep.depth.max())],
                         deep_distmin_med=float(deep.dist_min.median()),
                         wed_deep=int(deep.wed_window.sum()), tue_deep=int(deep.tue_window.sum()), wed_all=int(msh.wed_window.sum()),
                         views=p1v.view_group.value_counts().to_dict())
    # ---------------------------------------------------------------- P2
    mem = pd.read_csv(C.FAULTS / "fault_surfaces_v1_members.csv")
    gid = mem[mem.surface_id == "v1_GO022"].event_id
    p2 = c[c.ID.isin(gid)].copy().assign(view_group="v1_GO022 member", problem="P2")
    summary["P2"] = dict(n=int(len(p2)), depth_range=[float(p2.depth.min()), float(p2.depth.max())],
                         v4_depth_med=float(p2.v4_depth.median()), depth_med=float(p2.depth.median()),
                         sigz_med=float(p2.sig_z.median()), distmin_med=float(p2.dist_min.median()),
                         years=p2.t.dt.year.value_counts().sort_index().to_dict(),
                         depth_quality=p2.depth_quality.value_counts().to_dict())
    # ---------------------------------------------------------------- P3
    mt = C.mt_match()
    m = C.mags()
    ids41 = m.index[m.mt_match.eq("v4.1")]
    p3 = c[c.ID.isin(mt.event_id) | c.ID.isin(ids41)].copy()
    cc = {i: C.comcat_for(i) for i in p3.ID}
    p3["cc_id"] = [cc[i]["id"] for i in p3.ID]
    p3["cc_src"] = [cc[i]["src"] for i in p3.ID]
    p3["cc_depth"] = [cc[i]["depth"] for i in p3.ID]
    p3["cc_lat"] = [cc[i]["lat"] for i in p3.ID]
    p3["cc_lon"] = [cc[i]["lon"] for i in p3.ID]
    p3["cc_time"] = [str(cc[i]["t"]) for i in p3.ID]
    p3["cc_mag"] = [cc[i]["mag"] for i in p3.ID]
    p3["dz_cc"] = p3.depth - p3.cc_depth
    p3["dh_cc"] = C.hav(p3.lat.values, p3.lon.values, p3.cc_lat.values, p3.cc_lon.values)
    p3["dt_cc"] = [(r.t - cc[r.ID]["t"]).total_seconds() if cc[r.ID]["t"] is not None else np.nan for _, r in p3.iterrows()]
    p3["d3_cc"] = np.hypot(p3.dh_cc, p3.dz_cc)
    p3["mover"] = p3.ID.isin(ids41) & (p3.ID < 900000)
    p3["view_group"] = np.where(p3.mover, "large mover (v4.1 tensor carried; 31-101 km, 4.6-12 s from ComCat)",
                                np.where(p3.ID.isin(ids41), "v4.1 addition, tensor carried", "tensor matched in v5.1"))
    p3 = pd.concat([p3[p3.mover].sort_values("d3_cc", ascending=False),
                    p3[~p3.mover & p3.ID.isin(ids41)],
                    p3[~p3.ID.isin(ids41)].sort_values(["region", "dz_cc"], ascending=[True, False])])
    p3["problem"] = "P3"
    v51 = p3[~p3.ID.isin(ids41)]
    summary["P3"] = dict(n=int(len(p3)), n_v51_match=int(len(v51)), n_carried=int(p3.ID.isin(ids41).sum()),
                         dz_med_v51match=float(v51.dz_cc.median()), dt_med_v51match=float(v51.dt_cc.median()),
                         dz_med_by_region=v51.groupby("region").dz_cc.median().round(1).to_dict(),
                         movers=p3[p3.mover][["ID", "dh_cc", "dz_cc", "d3_cc", "dt_cc"]].round(1).to_dict("records"))
    # ---------------------------------------------------------------- P4
    shown = set(p1v.ID) | set(p2.ID) | set(p3.ID)
    p4v, p4t = [], []
    for flag, lab in (("deep_pass", "deep pass"), ("far_only", "far only")):
        s = c[c[flag]].copy()
        s["flag"] = flag
        p4t.append(s)
        pool = s[~s.ID.isin(shown)]
        smp = strat(pool, ["region"], 40, rng, min_per=2)
        p4v.append(smp.assign(view_group=f"{lab}: stratified by region and depth"))
        summary[f"P4_{flag}"] = dict(n=int(len(s)), depth_med=float(s.depth.median()), v4_depth_med=float(s.v4_depth.median()),
                                     by_region=s.region.value_counts().to_dict(), views=int(len(smp)))
    p4v = pd.concat(p4v).drop_duplicates("ID")
    p4v["problem"] = "P4"
    p4t = pd.concat(p4t)
    # ---------------------------------------------------------------- write
    for name, df, extra in (("P1_views", p1v, ["view_group"]), ("P1_table", p1t, ["has_view"]),
                            ("P2_views", p2, ["view_group"]),
                            ("P3_views", p3, ["view_group", "cc_id", "cc_src", "cc_lat", "cc_lon", "cc_depth", "cc_time",
                                              "cc_mag", "dz_cc", "dh_cc", "dt_cc", "d3_cc", "mover"]),
                            ("P4_views", p4v, ["view_group", "flag"]), ("P4_table", p4t, ["flag"])):
        cols = [k for k in keep_cols if k in df.columns] + extra
        df[cols].to_csv(SETS_WORK / f"{name}.csv", index=False, float_format="%.4f")
        df[cols].to_csv(SETS_REPO / f"{name}.csv", index=False, float_format="%.4f")
    (SETS_WORK / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (SETS_REPO / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))
    allv = pd.concat([p1v, p2, p3, p4v])
    print("views:", {k: int(v) for k, v in allv.problem.value_counts().items()}, "unique events", allv.ID.nunique())


if __name__ == "__main__":
    main()

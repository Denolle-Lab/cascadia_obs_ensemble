"""Track B, step B1: tomography event set, picks and stations (code-agnostic; TomoATT conversion comes later).

1. Events (D-B003, params.yaml `selection`): v4-NLL3D iter1 catalog, tiers A-C, nphs >= 12, gap <= 200 deg,
   RMS <= 0.5 s. Declustering in cubic cells of `decluster_cell_km` in the NLL Lambert frame (LCC WGS-84,
   lat_0 45, lon_0 -125.8, standard parallels 41 and 49; x, y in km) and depth below sea level (km); in each
   cell keep the `decluster_keep_per_cell` events with the most phases (NLLoc nphs), ties broken by lower
   RMS, then lower ID.
2. Picks of those events from catalog v4 (stations in the v4-NLL3D GTSRCE list only), sigma per D-B008.
   Pick QC (params.yaml `pick_qc`, proposed in notes/dataset.md), applied in this order, first reason wins:
     no_nll      pick not found among the iter1 NLLoc phases of the event (matched on event, station, phase,
                 pick time to 0.1 ms)
     w0          NLLoc EDT weight 0 in iter1 (station outside the tile's time grids, >~150-175 km)
     duplicate   same event/station/phase more than once (keep the highest pick_prob, then earliest)
     res         |iter1 residual (with v4-NLL3D station terms) - median| > k * 1.4826 MAD, where median and MAD
                 come from the iter0 residuals (no station terms) of the same phase, used picks only
     sp          S pick of an event-station pair whose S/P travel-time ratio is outside sp_ratio_range, or whose
                 S-P differential residual (iter1 location, no terms) minus the station median differs from
                 zero by more than sp_k_mad * 1.4826 MAD (MAD over all pairs). The S pick is dropped.
   Events then below nphs/gap of D-B003 are listed with a decision (pick_qc.post_qc_*).
3. Station analysis for D-B006 (OBS vs land residuals and v4-NLL3D station terms).
4. Outputs in <base>/b1_dataset: events.csv, picks.csv, stations.csv, picks_rejected.csv,
   events_below_threshold.csv, station_terms_analysis.csv; QC figures and tables in data/tomo_v1/b1_dataset.

Run: LD_LIBRARY_PATH=.pixi/envs/default/lib .pixi/envs/default/bin/python workflow/10_tomography/b1_select_events.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Proj

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import REPO, load_params, write_manifest  # noqa: E402
from b1_hyp_phases import load_phases  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
P = load_params()
BASE = Path(P["paths"]["base"])
OUT = BASE / "b1_dataset"
FIG = REPO / "data/tomo_v1/b1_dataset"
STATIONS_META = Path("/wd1/mdenolle_data/picks_share/stations/stations_ver5_v6.csv")  # same table as a1_build_model.py
PROJ = Proj(proj="lcc", lat_0=45.0, lon_0=-125.8, lat_1=41.0, lat_2=49.0, ellps="WGS84")
MADK = 1.4826
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def md_table(df, index=True):
    df = df.reset_index() if index else df
    rows = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    rows += ["| " + " | ".join("" if (not isinstance(v, str) and pd.isna(v)) else str(v) for v in r) + " |"
             for r in df.itertuples(index=False)]
    return "\n".join(rows)


def rounded(df):
    """Round floats for CSV output: coordinates 6 decimals, everything else 4 decimals."""
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].round(6 if c in ("lat", "lon", "sta_lat", "sta_lon") else 4)
    return df


def robust(x):
    x = np.asarray(x, float)
    m = np.median(x)
    return m, MADK * np.median(np.abs(x - m))


# ---------------------------------------------------------------- 1. events
def select_events():
    S = P["selection"]
    cat = pd.read_csv(P["paths"]["trackA_catalog"])
    # the catalog CSV is written with 5 significant digits (lon -121.71 = +-0.005 deg); take full-precision
    # maximum-likelihood hypocentres from the run's locations.csv (same NLLoc solutions)
    loc = pd.read_csv(Path(P["paths"]["reloc_v5"]) / "runs" / P["paths"]["trackA_run"] / "locations.csv",
                      usecols=["status", "ID", "time", "lat", "lon", "depth"])
    loc = loc[loc.status == "LOCATED"].drop_duplicates("ID").set_index("ID")
    m = cat.loc_source == "nll3d"
    for c in ("time", "lat", "lon", "depth"):
        cat.loc[m, c] = cat.loc[m, "ID"].map(loc[c]).values
    R = S["ridge"]                                       # D-B023: softer rule for the ridge regions
    is_r = cat.region.isin(R["regions"])
    s = cat[~is_r & cat.tier.isin(S["tiers"]) & (cat.nphs >= S["min_nphs"]) & (cat.gap <= S["max_gap_deg"])
            & (cat.rms <= S["max_rms_s"])].copy()
    s["cell_km"], s["keep_n"], s["rule"] = S["decluster_cell_km"], S["decluster_keep_per_cell"], "main"
    sr = cat[is_r & cat.tier.isin(R["tiers"]) & (cat.nphs >= R["min_nphs"]) & (cat.gap <= R["max_gap_deg"])
             & (cat.rms <= R["max_rms_s"])].copy()
    sr["cell_km"], sr["keep_n"], sr["rule"] = R["decluster_cell_km"], R["decluster_keep_per_cell"], "ridge"
    log(f"catalog {len(cat)}; passing tier/nphs/gap/rms: main {len(s)} (outside {R['regions']}), ridge {len(sr)}")
    s = pd.concat([s, sr])
    x, y = PROJ(s.lon.values, s.lat.values)
    s["x_km"], s["y_km"] = x / 1e3, y / 1e3
    c = s.cell_km
    s["cell"] = list(zip(s.rule, np.floor(s.x_km / c).astype(int), np.floor(s.y_km / c).astype(int),
                         np.floor(s.depth / c).astype(int)))
    s = s.sort_values(["nphs", "rms", "ID"], ascending=[False, True, True])
    s["cell_rank"] = s.groupby("cell").cumcount()
    s["cell_n"] = s.groupby("cell").ID.transform("size")
    k = s[s.cell_rank < s.keep_n].sort_values("ID").copy()
    c = S["decluster_cell_km"]
    log(f"decluster main {c} km x {S['decluster_keep_per_cell']}, ridge {R['decluster_cell_km']} km x "
        f"{R['decluster_keep_per_cell']}: {s.cell.nunique()} occupied cells, kept {len(k)} "
        f"(ridge {int((k.rule == 'ridge').sum())})")
    cm = cat[~is_r]
    t = cm.tier.isin(S["tiers"])
    funnel = pd.DataFrame({
        "step": ["catalog v4-NLL3D (all)", f"outside the ridge regions {R['regions']}", "tier A-C",
                 "+ nphs >= %d" % S["min_nphs"], "+ gap <= %d" % S["max_gap_deg"], "+ rms <= %.1f" % S["max_rms_s"],
                 "decluster (main rule)", f"ridge regions: {len(cat[is_r])} events, ridge rule (D-B023)",
                 "ridge, decluster"],
        "events": [len(cat), len(cm), int(t.sum()), int((t & (cm.nphs >= S["min_nphs"])).sum()),
                   int((t & (cm.nphs >= S["min_nphs"]) & (cm.gap <= S["max_gap_deg"])).sum()),
                   int((s.rule == "main").sum()), int((k.rule == "main").sum()), int((s.rule == "ridge").sum()),
                   int((k.rule == "ridge").sum())]})
    return cat, s, k, funnel


# ---------------------------------------------------------------- stations
def load_stations():
    g = pd.read_csv(P["paths"]["stations"], sep=r"\s+", header=None,
                    names=["kw", "station", "ll", "lat", "lon", "zsrce_km", "elev_col"])
    meta = pd.read_csv(STATIONS_META)
    meta["station_code"] = meta.station + "." + meta.network
    meta = meta.drop_duplicates("station_code").set_index("station_code")
    st = g[["station", "lat", "lon", "zsrce_km"]].copy()
    st["network"] = st.station.str.split(".").str[1]
    st["elev_km"] = st.station.map(meta.elev_m) / 1e3
    st["seafloor_meta"] = st.station.map(meta.seafloor).astype("boolean")
    # OBS: seafloor flag of the station table; GTSRCE zSrce (depth +down) > 0.1 km is the same set except
    # coastal land stations a few m below sea level (checked below)
    st["is_obs"] = st.seafloor_meta.fillna(st.zsrce_km > 0.1).astype(bool)
    n_disagree = (st.is_obs != (st.zsrce_km > 0.1)).sum()
    log(f"stations: {len(st)} in GTSRCE, OBS {st.is_obs.sum()} (seafloor flag); "
        f"flag vs zSrce>0.1 km disagree for {n_disagree}")
    return st


# ---------------------------------------------------------------- 2. picks
def load_picks(k, st):
    pk = pd.read_csv(REPO / P["paths"]["picks_v4"], skipinitialspace=True)
    pk = pk.rename(columns={"event_ID": "ID", "travel_time(s)": "v4_tt", "amplitude(mm)": "amp_mm"})
    pk = pk[pk.ID.isin(k.ID)].copy()
    n0 = len(pk)
    pk = pk[pk.station.isin(st.station)]
    log(f"v4 picks of selected events: {n0}; dropped (station not in v4-NLL3D list): {n0 - len(pk)}")
    pk["pick_time"] = pd.to_datetime(pk.pick_datetime.str.rstrip("Z"), format="ISO8601")
    pk["key"] = np.round(pk.pick_time.astype("int64") / 1e5).astype("int64")
    E = P["pick_errors"]
    cls = np.select([pk.pick_prob >= 0.6, pk.pick_prob >= 0.3, pk.pick_prob.isna()], [2, 1, 1], 0)
    pk["sigma_s"] = pd.Series(cls, index=pk.index).map({int(a): b for a, b in E["P_sigma_s"].items()}) \
        * np.where(pk.phase == "S", E["S_over_P"], 1.0)
    for run, sfx in ((P["paths"]["trackA_run"], "i1"), (P["pick_qc"]["res_ref_run"], "i0")):
        q = load_phases(run)
        q = q[q.ID.isin(k.ID)].copy()
        q["key"] = np.round(q.pick_time.astype("int64") / 1e5).astype("int64")
        q = q.rename(columns={"sta": "station"})
        q["station"] = q.station.astype(str)
        q["phase"] = q.phase.astype(str)
        q = q.drop_duplicates(["ID", "station", "phase", "key"])
        q = q[["ID", "station", "phase", "key", "res", "w", "tcorr", "ttpred", "dist"]]
        q = q.rename(columns={c: f"{c}_{sfx}" for c in ("res", "w", "tcorr", "ttpred", "dist")})
        pk = pk.merge(q, on=["ID", "station", "phase", "key"], how="left")
    ot = pd.to_datetime(k.set_index("ID").time)
    pk["tt"] = (pk.pick_time - pk.ID.map(ot)).dt.total_seconds()
    pk["res_noterm_i1"] = pk.res_i1 + pk.tcorr_i1        # residual without station term, iter1 location
    pk["rule"] = pk.ID.map(k.set_index("ID").rule)
    pk = pk.merge(st[["station", "network", "lat", "lon", "elev_km", "is_obs"]].rename(
        columns={"lat": "sta_lat", "lon": "sta_lon", "elev_km": "sta_elev_km"}), on="station", how="left")
    return pk


def pick_qc(pk):
    Q = P["pick_qc"]
    pk["qc"] = ""
    rows = []

    def flag(mask, reason):
        m = mask & (pk.qc == "")
        pk.loc[m, "qc"] = reason
        ev_before = pk.loc[(pk.qc == "") | m, "ID"].nunique()
        ev_after = pk.loc[pk.qc == "", "ID"].nunique()
        rows.append(dict(filter=reason, P=int((m & (pk.phase == "P")).sum()), S=int((m & (pk.phase == "S")).sum()),
                         OBS_S=int((m & (pk.phase == "S") & pk.is_obs).sum()),
                         events_lost=ev_before - ev_after))

    flag(pk.res_i1.isna(), "no_nll")
    if Q["require_iter1_weight"]:
        flag(pk.w_i1 <= 0, "w0")
    # duplicates: keep highest pick_prob, then earliest
    o = pk.assign(_pp=pk.pick_prob.fillna(-1)).sort_values(["_pp", "pick_time"], ascending=[False, True])
    dup = o.duplicated(["ID", "station", "phase"], keep="first")
    flag(pk.index.isin(o.index[dup.values]), "duplicate")
    # residual outliers: reference distribution = iter0 residuals of used picks (w>0) of these events
    thr = {}
    for ph in ("P", "S"):
        ref = pk.loc[(pk.phase == ph) & (pk.w_i0 > 0) & (pk.rule == "main"), "res_i0"]  # D-B023: main set sets the spread
        m, sd = robust(ref)
        thr[ph] = dict(median=m, sigma_mad=sd, threshold=Q["res_k_mad"] * sd)
        col = {"iter1": "res_i1", "iter0": "res_i0", "iter1_noterm": "res_noterm_i1"}[Q["res_applied_to"]]
        flag((pk.phase == ph) & ((pk[col] - m).abs() > Q["res_k_mad"] * sd), "res")
    # S-P consistency on pairs still kept
    ok = pk[pk.qc == ""]
    pv = ok.pivot_table(index=["ID", "station"], columns="phase", values=["tt", "res_noterm_i1"],
                        aggfunc="first").dropna()
    pv.columns = [f"{a}_{b}" for a, b in pv.columns]
    pv = pv.reset_index()
    pv["ratio"] = pv.tt_S / pv.tt_P
    pv["dres"] = pv.res_noterm_i1_S - pv.res_noterm_i1_P
    pv["dres_c"] = pv.dres - pv.groupby("station").dres.transform("median")
    m, sd = robust(pv.loc[pv.ID.map(pk.drop_duplicates("ID").set_index("ID").rule) == "main", "dres_c"])
    lo, hi = Q["sp_ratio_range"]
    bad = pv[(pv.ratio < lo) | (pv.ratio > hi) | ((pv.dres_c - m).abs() > Q["sp_k_mad"] * sd)]
    thr["SP"] = dict(median=m, sigma_mad=sd, threshold=Q["sp_k_mad"] * sd, pairs=len(pv),
                     ratio_out=int(((pv.ratio < lo) | (pv.ratio > hi)).sum()), bad_pairs=len(bad))
    key = set(zip(bad.ID, bad.station))
    flag((pk.phase == "S") & pd.Series([(a, b) in key for a, b in zip(pk.ID, pk.station)], index=pk.index), "sp")
    tab = pd.DataFrame(rows)
    tab["picks"] = tab.P + tab.S
    return pk, tab, thr, pv


def residual_alternatives(pk):
    """Fraction of used picks (w>0) dropped by alternative residual thresholds (for notes/dataset.md)."""
    u = pk[(pk.w_i1 > 0) & pk.res_i0.notna()]
    out = []
    for ph in ("P", "S"):
        g = u[u.phase == ph]
        for ref in ("res_i0", "res_i1"):
            m, sd = robust(g[ref])
            for col in ("res_i0", "res_i1"):
                for kk in (4, 5, 6):
                    d = (g[col] - m).abs() > kk * sd
                    out.append(dict(phase=ph, sigma_from=ref.replace("res_i", "iter"),
                                    applied_to=col.replace("res_i", "iter"), k=kk,
                                    threshold_s=round(kk * sd, 2), drop_pct=round(100 * d.mean(), 2),
                                    drop_obs_pct=round(100 * d[g.is_obs].mean(), 2),
                                    drop_land_pct=round(100 * d[~g.is_obs].mean(), 2)))
    return pd.DataFrame(out)


def gap_deg(elat, elon, slat, slon):
    az = np.degrees(np.arctan2(np.sin(np.radians(slon - elon)) * np.cos(np.radians(slat)),
                               np.cos(np.radians(elat)) * np.sin(np.radians(slat)) -
                               np.sin(np.radians(elat)) * np.cos(np.radians(slat)) * np.cos(np.radians(slon - elon))))
    az = np.sort(np.mod(az, 360))
    if len(az) < 2:
        return 360.0
    return float(np.max(np.diff(np.r_[az, az[0] + 360])))


def event_check(k, kept):
    Q, S = P["pick_qc"], P["selection"]
    n = kept.groupby("ID").phase.value_counts().unstack(fill_value=0).reindex(k.ID, fill_value=0)
    ev = k.set_index("ID")
    ev["nP"], ev["nS"] = n.get("P", 0), n.get("S", 0)
    ev["npick_qc"] = ev.nP + ev.nS
    g = kept.groupby("ID")
    gaps = {i: gap_deg(ev.at[i, "lat"], ev.at[i, "lon"], d.sta_lat.values, d.sta_lon.values) for i, d in g}
    ev["gap_qc"] = pd.Series(gaps).reindex(ev.index).fillna(360.0)
    R = S["ridge"]
    r = ev.rule == "ridge"
    mn = np.where(r, R["min_nphs"], S["min_nphs"])
    mg = np.where(r, R["max_gap_deg"], S["max_gap_deg"])
    below = ev[(ev.npick_qc < mn) | (ev.gap_qc > mg)].copy()
    rb = below.rule == "ridge"
    keep = (below.npick_qc >= np.where(rb, R["post_qc_min_picks"], Q["post_qc_min_picks"])) & \
        (below.gap_qc <= np.where(rb, R["post_qc_max_gap_deg"], Q["post_qc_max_gap_deg"]))
    below["decision"] = np.where(keep, "keep", "drop")
    mnb, mgb = mn[(ev.npick_qc < mn) | (ev.gap_qc > mg)], mg[(ev.npick_qc < mn) | (ev.gap_qc > mg)]
    below["reason"] = (pd.Series(np.where(below.npick_qc < mnb, "npick<min", ""), index=below.index)
                       + np.where(below.gap_qc > mgb, " gap>max", "")).str.strip()
    return ev.reset_index(), below.reset_index()


# ---------------------------------------------------------------- 3. stations / OBS
def station_analysis(st):
    """Per-station mean/median iter0 residuals (no terms; used picks of well-located events, as in
    a2_station_terms.py) and the cumulative v4-NLL3D terms after iter1 (runs/iter1/station_terms.csv)."""
    rd = Path(P["paths"]["reloc_v5"]) / "runs"
    run0 = P["pick_qc"]["res_ref_run"]
    loc = pd.read_csv(rd / run0 / "locations.csv")
    good = set(loc.query("status == 'LOCATED' and nphs >= 8 and gap <= 270 and rms <= 0.6").ID)
    ph = load_phases(run0)
    ph = ph[(ph.w > 0) & ph.ID.isin(good)]
    ph = ph.assign(sta=ph.sta.astype(str), phase=ph.phase.astype(str))
    a = ph.groupby(["sta", "phase"]).res.agg(n="size", mean="mean", median="median").unstack()
    a.columns = [f"i0_{s}_{p}" for s, p in a.columns]
    t1 = pd.read_csv(rd / P["paths"]["trackA_run"] / "station_terms.csv")
    t1 = t1.pivot(index="sta", columns="phase", values="delay").add_prefix("term_")
    s = st.set_index("station").join(a).join(t1)
    s["S_over_P_term"] = s.term_S / s.term_P.where(s.term_P.abs() > 0.05)
    return s.reset_index().rename(columns={"index": "station"})


def obs_summary(sa, kept):
    rows = []
    for name, g in sa.groupby("is_obs"):
        g2 = g.dropna(subset=["term_P", "term_S"])
        r = np.corrcoef(g2.term_P, g2.term_S)[0, 1] if len(g2) > 2 else np.nan
        slope = np.polyfit(g2.term_P, g2.term_S, 1)[0] if len(g2) > 2 else np.nan
        rows.append({"group": "OBS" if name else "land", "stations": len(g),
                     "with_P_term": int(g.term_P.notna().sum()), "with_S_term": int(g.term_S.notna().sum()),
                     "i0_meanres_P_median": round(g.i0_mean_P.median(), 3),
                     "i0_meanres_S_median": round(g.i0_mean_S.median(), 3),
                     "term_P_median": round(g.term_P.median(), 3), "term_S_median": round(g.term_S.median(), 3),
                     "term_P_p5_p95": f"{g.term_P.quantile(.05):.2f} / {g.term_P.quantile(.95):.2f}",
                     "term_S_p5_p95": f"{g.term_S.quantile(.05):.2f} / {g.term_S.quantile(.95):.2f}",
                     "abs_S_term_gt_0.5": int((g.term_S.abs() > 0.5).sum()),
                     "abs_S_term_gt_1.0": int((g.term_S.abs() > 1.0).sum()),
                     "corr_P_S_terms": round(r, 2), "slope_S_on_P": round(slope, 2),
                     "S_picks_in_dataset": int(((kept.phase == "S") & (kept.is_obs == name)).sum()),
                     "P_picks_in_dataset": int(((kept.phase == "P") & (kept.is_obs == name)).sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures
def figures(ev, kept, stations, sa):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pygmt
    FIG.mkdir(parents=True, exist_ok=True)
    rel = pygmt.datasets.load_earth_relief("05m", region=[-133, -119, 38.5, 52], registration="gridline")
    ext = [-132, -119.5, 39.3, 51.5]

    def base(ax):
        ax.contour(rel.lon, rel.lat, rel.values, levels=[0], colors="0.4", linewidths=0.5)
        ax.set_xlim(ext[:2]); ax.set_ylim(ext[2:])
        ax.set_aspect(1 / np.cos(np.radians(45.5)))
        ax.tick_params(labelsize=7)

    fig, ax = plt.subplots(figsize=(6.5, 7.5))
    base(ax)
    o = ev.sort_values("depth", ascending=False)
    sc = ax.scatter(o.lon, o.lat, c=o.depth, s=1.5, cmap="viridis_r", vmin=0, vmax=50, lw=0, rasterized=True)
    s = stations
    ax.scatter(s[~s.is_obs].lon, s[~s.is_obs].lat, marker="^", s=10, fc="none", ec="k", lw=0.5, label="land")
    ax.scatter(s[s.is_obs].lon, s[s.is_obs].lat, marker="v", s=10, fc="none", ec="tab:red", lw=0.5, label="OBS")
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, label="depth below sea level (km)")
    cb.ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="lower left", title=f"{len(s)} stations", title_fontsize=7)
    ax.set_title(f"B1 tomography events: {len(ev)}", fontsize=9)
    fig.savefig(FIG / "b1_map_events.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    regs = ev.region.value_counts().index.tolist()
    nc = 4
    nr = int(np.ceil(len(regs) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(8, 1.8 * nr), sharex=True)
    for axx, r in zip(axs.flat, regs):
        d = ev[ev.region == r]
        axx.hist(d.depth, bins=np.arange(-3, 61, 1.5), orientation="horizontal", color="0.3")
        axx.set_title(f"{r} ({len(d)})", fontsize=7)
        axx.set_ylim(60, -3)
        axx.tick_params(labelsize=6)
    for axx in list(axs.flat)[len(regs):]:
        axx.axis("off")
    fig.supylabel("depth bsl (km)", fontsize=8)
    fig.supxlabel("events", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "b1_depth_by_region.png", dpi=150)
    plt.close(fig)

    # ray-coverage proxy: straight event-station paths sampled every ~2 km, unique paths per 0.25 deg cell
    fig, axs = plt.subplots(1, 2, figsize=(10, 6.2))
    for axx, ph in zip(axs, ("P", "S")):
        d = kept[kept.phase == ph].merge(ev[["ID", "lat", "lon"]], on="ID")
        la0, lo0, la1, lo1 = d.lat.values, d.lon.values, d.sta_lat.values, d.sta_lon.values
        L = np.hypot((la1 - la0) * 111.2, (lo1 - lo0) * 111.2 * np.cos(np.radians(la0)))
        ns = np.maximum(2, np.ceil(L / 2).astype(int))
        idx = np.repeat(np.arange(len(d)), ns)
        f = np.concatenate([np.linspace(0, 1, n) for n in ns])
        la = la0[idx] + f * (la1 - la0)[idx]
        lo = lo0[idx] + f * (lo1 - lo0)[idx]
        cx = np.floor((lo - ext[0]) / 0.25).astype(int)
        cy = np.floor((la - ext[2]) / 0.25).astype(int)
        u = pd.DataFrame({"i": idx, "x": cx, "y": cy}).drop_duplicates()
        cnt = u.groupby(["x", "y"]).size()
        nx, ny = int(np.ceil((ext[1] - ext[0]) / 0.25)), int(np.ceil((ext[3] - ext[2]) / 0.25))
        G = np.zeros((ny, nx))
        cnt = cnt[(cnt.index.get_level_values(0) >= 0) & (cnt.index.get_level_values(0) < nx) &
                  (cnt.index.get_level_values(1) >= 0) & (cnt.index.get_level_values(1) < ny)]
        G[cnt.index.get_level_values(1), cnt.index.get_level_values(0)] = cnt.values
        G[G == 0] = np.nan
        im = axx.pcolormesh(ext[0] + 0.25 * np.arange(nx + 1), ext[2] + 0.25 * np.arange(ny + 1), np.log10(G),
                            cmap="magma_r", vmin=0, vmax=5, rasterized=True)
        base(axx)
        axx.set_title(f"{ph}: {len(d)} paths", fontsize=9)
        cb = fig.colorbar(im, ax=axx, shrink=0.6, label="log10 paths per 0.25 deg cell")
        cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "b1_ray_coverage.png", dpi=150)
    plt.close(fig)

    # pick counts per station
    s = stations.sort_values("n_P", ascending=False)
    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    for axx, ob in zip(axs, (False, True)):
        d = s[s.is_obs == ob].sort_values(["n_P"], ascending=False)
        x = np.arange(len(d))
        axx.bar(x - 0.2, d.n_P, width=0.4, color="tab:blue", label="P")
        axx.bar(x + 0.2, d.n_S, width=0.4, color="tab:orange", label="S")
        axx.set_yscale("log")
        axx.set_xlim(-1, len(d))
        axx.set_xticks([])
        axx.set_ylabel("picks", fontsize=8)
        axx.set_title(f"{'OBS' if ob else 'land'} stations ({len(d)}, sorted by P count)", fontsize=8)
        axx.legend(fontsize=7)
        axx.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "b1_picks_per_station.png", dpi=150)
    plt.close(fig)

    # OBS / land station terms
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    for ob, c in ((False, "k"), (True, "tab:red")):
        d = sa[sa.is_obs == ob]
        axs[0].scatter(d.term_P, d.term_S, s=6, c=c, label="OBS" if ob else "land")
        axs[1].hist(d.term_S.dropna(), bins=np.arange(-2.5, 2.6, 0.1), histtype="step", color=c,
                    label="OBS" if ob else "land")
    axs[0].plot([-1, 1], [-1.73, 1.73], "b--", lw=0.7, label="S = 1.73 P")
    axs[0].set_xlabel("v4-NLL3D P term (s)"); axs[0].set_ylabel("v4-NLL3D S term (s)")
    axs[0].axhline(0, c="0.7", lw=0.5); axs[0].axvline(0, c="0.7", lw=0.5)
    axs[0].legend(fontsize=7)
    axs[1].set_xlabel("v4-NLL3D S term (s)"); axs[1].set_ylabel("stations"); axs[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "b1_station_terms_obs_land.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    cat, cand, k, funnel = select_events()
    st = load_stations()
    pk = load_picks(k, st)
    alt = residual_alternatives(pk)
    pk, qtab, thr, pairs = pick_qc(pk)
    kept = pk[pk.qc == ""].copy()
    ev, below = event_check(k, kept)
    drop_ids = set(below.loc[below.decision == "drop", "ID"])
    ev = ev[~ev.ID.isin(drop_ids)].copy()
    ev["below_DB003_after_qc"] = ev.ID.isin(below.ID)   # kept under pick_qc.post_qc_* (listed in events_below_threshold.csv)
    rows_ev = dict(filter="event below D-B003 after QC (dropped)",
                   P=int(((kept.phase == "P") & kept.ID.isin(drop_ids)).sum()),
                   S=int(((kept.phase == "S") & kept.ID.isin(drop_ids)).sum()),
                   OBS_S=int(((kept.phase == "S") & kept.is_obs & kept.ID.isin(drop_ids)).sum()),
                   events_lost=len(drop_ids))
    pk.loc[pk.ID.isin(drop_ids) & (pk.qc == ""), "qc"] = "event_dropped"
    kept = kept[~kept.ID.isin(drop_ids)]
    qtab = pd.concat([qtab, pd.DataFrame([rows_ev])], ignore_index=True)
    qtab["picks"] = qtab.P + qtab.S

    # S-count distribution per event (after QC)
    s_hist = ev.nS.clip(upper=10).value_counts().sort_index()
    s_hist.index = [str(i) if i < 10 else ">=10" for i in s_hist.index]

    # stations
    n = kept.groupby(["station", "phase"]).size().unstack(fill_value=0)
    stations = st.copy()
    stations["n_P"] = stations.station.map(n.get("P")).fillna(0).astype(int)
    stations["n_S"] = stations.station.map(n.get("S")).fillna(0).astype(int)
    stations_out = stations[(stations.n_P + stations.n_S) > 0]
    sa = station_analysis(st)
    osum = obs_summary(sa, kept)

    # ---- write data
    evo = ev.rename(columns={"time": "origin_time", "depth": "depth_bsl_km"})
    evo = evo[["ID", "origin_time", "lat", "lon", "depth_bsl_km", "tier", "nphs", "gap", "rms", "h_max", "sig_z",
               "region", "rule", "nP", "nS", "gap_qc", "below_DB003_after_qc", "dist_min", "x_km", "y_km", "cell_n"]]
    evo["nphs"] = evo.nphs.astype(int)
    rounded(evo).to_csv(OUT / "events.csv", index=False)
    cols = ["ID", "station", "network", "phase", "pick_time", "tt", "sigma_s", "pick_prob", "res_i1", "w_i1",
            "tcorr_i1", "res_i0", "dist_i1", "sta_lat", "sta_lon", "sta_elev_km", "is_obs"]
    ren = {"ID": "event_ID", "tt": "travel_time_s", "res_i1": "iter1_res_s", "w_i1": "iter1_weight",
           "tcorr_i1": "iter1_station_term_s", "res_i0": "iter0_res_s", "dist_i1": "epi_dist_km"}
    kept = kept.sort_values(["ID", "pick_time"])
    ko = kept[cols].rename(columns=ren)
    ko["pick_time"] = ko.pick_time.dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    rounded(ko).to_csv(OUT / "picks.csv", index=False)
    rej = pk[pk.qc != ""][cols + ["qc"]].rename(columns=ren)
    rounded(rej).to_csv(OUT / "picks_rejected.csv", index=False)
    so = stations_out.rename(columns={"elev_km": "elevation_km"})[
        ["station", "network", "lat", "lon", "elevation_km", "zsrce_km", "is_obs", "n_P", "n_S"]]
    rounded(so).to_csv(OUT / "stations.csv", index=False)
    rounded(below).to_csv(OUT / "events_below_threshold.csv", index=False)
    rounded(sa).to_csv(OUT / "station_terms_analysis.csv", index=False)
    alt.to_csv(OUT / "residual_threshold_alternatives.csv", index=False)

    # ---- tables for notes/dataset.md
    big = sa[sa.term_S.abs() > 0.5].sort_values("term_S", key=np.abs, ascending=False)
    big = big[["station", "is_obs", "zsrce_km", "term_P", "term_S", "i0_n_S", "i0_mean_P", "i0_mean_S",
               "S_over_P_term"]].round(3)
    tier_region = pd.crosstab(ev.region, ev.tier, margins=True)
    picks_region = kept.merge(ev[["ID", "region"]], on="ID").groupby(["region", "phase"]).size().unstack()
    thr_df = pd.DataFrame(thr).T.round(3)
    lines = ["# B1 tables (written by b1_select_events.py)", "", "## Event selection funnel", "",
             md_table(funnel, index=False), "", "## Pick QC (applied in order; first reason wins)", "",
             md_table(qtab[["filter", "picks", "P", "S", "OBS_S", "events_lost"]], index=False), "",
             "## Thresholds", "", md_table(thr_df), "",
             "## Residual threshold alternatives (used picks, w>0)", "", md_table(alt, index=False), "",
             "## Events by S-pick count after QC", "", md_table(s_hist.rename("events").to_frame()), "",
             "## Events below D-B003 after pick QC", "",
             md_table(below.groupby(["reason", "decision"]).size().rename("events").to_frame()), "",
             "## Events by region and tier", "", md_table(tier_region), "",
             "## Kept picks by region", "", md_table(picks_region), "",
             "## OBS vs land: v4-NLL3D station terms (iter1 cumulative) and iter0 mean residuals", "",
             md_table(osum, index=False), "",
             f"## Stations with |S term| > 0.5 s ({len(big)})", "", md_table(big, index=False), "",
             "## Log", "", "```", *LOG, "```"]
    (FIG / "b1_tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    figures(ev, kept, so, sa)

    outs = [OUT / f for f in ("events.csv", "picks.csv", "stations.csv", "picks_rejected.csv",
                              "events_below_threshold.csv", "station_terms_analysis.csv",
                              "residual_threshold_alternatives.csv")]
    outs += sorted(FIG.glob("*.png")) + [FIG / "b1_tables.md"]
    write_manifest("b1_select_events", OUT,
                   inputs=[P["paths"]["trackA_catalog"], REPO / P["paths"]["picks_v4"], P["paths"]["stations"],
                           STATIONS_META, BASE / "b1_dataset/cache/phases_iter1_vpvs176.parquet",
                           BASE / "b1_dataset/cache/phases_iter0_vpvs176.parquet"],
                   outputs=outs, params={k_: P[k_] for k_ in ("selection", "pick_errors", "pick_qc", "obs")},
                   counts=dict(events=len(ev), picks=len(kept), P=int((kept.phase == "P").sum()),
                               S=int((kept.phase == "S").sum()), stations=len(so), obs_stations=int(so.is_obs.sum()),
                               rejected=int((pk.qc != "").sum())))


if __name__ == "__main__":
    main()

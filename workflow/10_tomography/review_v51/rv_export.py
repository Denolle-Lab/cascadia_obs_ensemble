"""review_v51_events.csv: one row per event and problem of the four review sets (all events, not only those with an
event view), with location metrics, problem metrics, an automatic indicator (rules below, NOT a verdict) and empty
reviewer columns. PI request 2026-11-25: CSV report + HTML viewer + CSV returned with corrected events.

Rows
  P1  the 331 box events deeper than 30 km (role deep), the 10 shallow references (role reference) and the 2 events of
      the Wednesday window shallower than 30 km that have an event view (role wednesday_shallow)
  P2  the 39 members of v1_GO022
  P3  the 148 located tensor events and 900002, 900003 (role no_hypocentre: no v5.1 hypocentre)
  P4  the 356 deep_pass and 624 far_only events
Metrics for every located row are recomputed from the v5.1 NonLinLoc .hyp (phases, NLLoc weights) and scatter, in the
v5.1 grids (rv_event.predict: OT refitted per hypothesis). P1 calibration signal: a short read of HSR.UW, EDM.UW and
STD.UW (EHZ) around the origin time (rv_p1_cont quiet-stretch and step metrics).

  python rv_export.py [--nproc 12]
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import rv_common as C  # noqa: E402

OUT_CSV = C.OUT / "review_v51_events.csv"
REPO = C.HERE.parent.parent.parent / "data" / "review_v51"
FOLDERS = {"P1": "P1_msh_deep", "P2": "P2_brookings", "P3": "P3_tensor", "P4": "P4_deep_far"}
SP_STA = ["KBO.NC", "KSXB.NC", "L02D.TA", "L02E.TA"]
CALIB_STA = ["HSR.UW", "EDM.UW", "STD.UW"]
REVIEW_COLS = ["reviewer", "is_earthquake", "depth_verdict", "corrected_depth_km", "action", "comment", "reviewed_utc"]
RULES = {
    "P1_common_mode": "P1: share of P picks at UW stations >= 0.9 AND P-pick time spread <= 0.5 s (farthest P pick > 8 km "
                      "from the v5.1 epicentre) AND no S pick",
    "P1_calibration": "P1: the weekly calibration-type signal at HSR.UW, EDM.UW or STD.UW within 30 s before to 90 s after "
                      "the v5.1 origin time (a quiet stretch >= 10 s of the 2-15 Hz trace AND a step of the raw trace >= 30 "
                      "times the median 1-s standard deviation)",
    "P1_wed_window": "P1: origin time on a Wednesday 19:14:00-19:20:59 UTC",
    "P2_nearest_downweighted": "P2: NonLinLoc weight < 0.1 for an arrival at the nearest station",
    "P2_sp_mismatch": "P2: |observed - predicted (v5.1) S-P| > 0.5 s at one of KBO.NC, KSXB.NC, L02D.TA, L02E.TA",
    "P2_v4_fits_better": "P2: residual rms (OT refit) of the v4 hypocentre < that of v5.1 - 0.05 s",
    "P3_large_move": "P3: v5.1 more than 30 km (3-D) or more than 4 s from the ComCat origin",
    "P3_deeper_20km": "P3: v5.1 more than 20 km deeper than ComCat",
    "P3_comcat_fits_better": "P3: residual rms (OT refit) of the ComCat hypocentre < that of v5.1 - 0.05 s",
    "P4_two_cluster": "P4: the NonLinLoc scatter depths form two clusters (two peaks of a 2-km Gaussian kernel density, at "
                      "least 10 km apart, the smaller at least 25 % of the larger, the density between them below 50 % of "
                      "the smaller peak)",
    "P4_depth_change_30km": "P4: |v5.1 depth - v4 depth| > 30 km",
    "P4_v4_fits_better": "P4: residual rms (OT refit) of the v4 hypocentre < that of v5.1 - 0.05 s",
    "P4_sigz_gt_10km": "P4: catalog sig_z > 10 km",
    "no_hypocentre": "no v5.1 hypocentre (NonLinLoc did not locate the event)",
}


def two_cluster(z):
    z = np.asarray(z, float)
    z = z[np.isfinite(z)]
    if len(z) < 200:
        return False, None, None
    g = np.arange(z.min() - 6, z.max() + 6, 0.5)
    d = np.zeros_like(g)
    for k in range(0, len(z), 500):
        d += np.exp(-0.5 * ((g[:, None] - z[None, k:k + 500]) / 2.0) ** 2).sum(axis=1)
    pk = [i for i in range(1, len(d) - 1) if d[i] >= d[i - 1] and d[i] > d[i + 1] and d[i] >= 0.25 * d.max()]
    best = None
    for a in range(len(pk)):
        for b in range(a + 1, len(pk)):
            i, j = pk[a], pk[b]
            if g[j] - g[i] < 10:
                continue
            lo = min(d[i], d[j])
            if d[i:j + 1].min() < 0.5 * lo:
                best = (float(g[i]), float(g[j]))
    return best is not None, (best[0] if best else None), (best[1] if best else None)


def light(eid):
    """Metrics of one located event from its .hyp and scatter (no waveforms)."""
    import rv_event as E
    c = C.catalog()
    row = c.loc[eid]
    out = dict(ID=int(eid))
    hf, sf, wd = C.find_hyp(eid, row)
    if hf is None:
        hf = C.sum_block(eid, wd)
    if hf is None:
        out["err"] = "no NonLinLoc output"
        return out
    H = C.parse_hyp(hf)
    P = H["phases"]
    box = C.control_box(wd)
    hy = E.hypotheses(row, H)
    P, otfit = E.predict(P, box, hy, row)
    S = C.stations()
    P["dist"] = C.hav(H["lat"], H["lon"], [S.lat.get(s, np.nan) for s in P.sta], [S.lon.get(s, np.nan) for s in P.sta])
    out["nP"] = int((P.phase == "P").sum())
    out["nS"] = int((P.phase == "S").sum())
    for h in E.HYPS:
        k = {"v5.1": "v51", "v4": "v4", "ComCat": "comcat"}[h]
        out[f"rms_{k}"] = round(E.rms(P[f"resfit_{h}"]), 3) if h in hy else np.nan
    if "ComCat" in hy:
        cc = hy["ComCat"]
        out["comcat_id"] = cc["info"]["id"]
        out["comcat_depth"] = cc["depth"]
        out["comcat_time"] = str(cc["OT"])
        out["dz_comcat"] = round(hy["v5.1"]["depth"] - cc["depth"], 2)
        out["dh_comcat"] = round(float(C.hav(hy["v5.1"]["lat"], hy["v5.1"]["lon"], cc["lat"], cc["lon"])), 2)
        out["dt_comcat"] = round((hy["v5.1"]["OT"] - cc["OT"]).total_seconds(), 2) if cc["OT"] is not None else np.nan
    if len(P):
        near = P.sort_values("dist").sta.iloc[0]
        out["nearest_sta"] = near
        out["nearest_min_weight"] = round(float(P[P.sta == near].weight.min()), 2)
    out["n_weight_lt_0p1"] = int((P.weight < 0.1).sum())
    tc = C.station_terms()
    for st in SP_STA:
        x = P[P.sta == st]
        key = st.split(".")[0]
        if set(x.phase) >= {"P", "S"}:
            p_, s_ = x[x.phase == "P"].iloc[0], x[x.phase == "S"].iloc[0]
            out[f"sp_obs_{key}"] = round((s_.tobs - p_.tobs).total_seconds(), 2)
            for h, k in (("v5.1", "v51"), ("v4", "v4")):
                if h in hy:
                    out[f"sp_pred_{k}_{key}"] = round(float((s_[f"tt_{h}"] + s_.tcorr) - (p_[f"tt_{h}"] + p_.tcorr)), 2)
    if sf is not None:
        a = C.read_scatter(sf, max_n=4000)
        z = C.ieft(a[:, 2])
        q = np.percentile(z, [16, 50, 84])
        out["scatter_z16"], out["scatter_z50"], out["scatter_z84"] = (round(float(v), 1) for v in q)
        tc_, z1, z2 = two_cluster(z)
        out["two_cluster_scatter"] = "yes" if tc_ else "no"
        if tc_:
            out["two_cluster_depths"] = f"{z1:.0f};{z2:.0f}"
    else:
        out["two_cluster_scatter"] = "no scatter"
    return out


def calib(eid):
    """Calibration-type signal at the analog MSH stations around the v5.1 origin time."""
    import rv_p1_cont as R
    c = C.catalog()
    t = c.loc[eid].t
    t0, t1 = t - pd.Timedelta(seconds=30), t + pd.Timedelta(seconds=90)
    out = dict(ID=int(eid))
    for sta in CALIB_STA:
        key = sta.split(".")[0]
        try:
            r, st = R.read(sta, t0, t1)
            rr, _ = R.read(sta, t0, t1, raw=True)
        except Exception as e:
            out[f"calib_{key}"] = f"error {type(e).__name__}"
            continue
        if r is None or rr is None:
            out[f"calib_{key}"] = "no data"
            continue
        tt, y = r
        ok = np.isfinite(y)
        y = y[ok]
        yr = rr[1][np.isfinite(rr[1])]
        sr = 1.0 / np.median(np.diff(tt[ok]))
        m = int(round(sr))
        nb = len(y) // m
        if nb < 30:
            out[f"calib_{key}"] = "short"
            continue
        sd = y[:nb * m].reshape(nb, m).std(axis=1)
        q = sd < 0.1 * np.median(sd)
        best = cur = 0
        for f in q:
            cur = cur + 1 if f else 0
            best = max(best, cur)
        nbr = len(yr) // m
        mu = yr[:nbr * m].reshape(nbr, m).mean(axis=1)
        sdr = yr[:nbr * m].reshape(nbr, m).std(axis=1)
        step = float(np.max(np.abs(mu - np.median(mu))) / max(np.median(sdr), 1e-9))
        out[f"quiet_s_{key}"] = float(best)
        out[f"step_{key}"] = round(step, 1)
        out[f"calib_{key}"] = "yes" if (best >= 10 and step >= 30) else "no"
    return out


def rows():
    """(ID, problem, role) of every row."""
    W = C.WORK / "sets"
    out = []
    p1t = pd.read_csv(W / "P1_table.csv")
    p1v = pd.read_csv(W / "P1_views.csv")
    out += [(i, "P1", "deep") for i in p1t.ID]
    out += [(i, "P1", "reference") for i in p1v[p1v.view_group.str.startswith("shallow")].ID]
    out += [(i, "P1", "wednesday_shallow") for i in p1v[p1v.view_group.str.startswith("Wednesday") & (p1v.depth <= 30)].ID]
    out += [(i, "P2", "v1_GO022") for i in pd.read_csv(W / "P2_views.csv").ID]
    p3 = pd.read_csv(W / "P3_views.csv")
    out += [(r.ID, "P3", "large_mover" if r.mover else ("v4.1_addition" if r.ID >= 900000 else "tensor_v51_match"))
            for r in p3.itertuples()]
    out += [(900002, "P3", "no_hypocentre"), (900003, "P3", "no_hypocentre")]
    p4 = pd.read_csv(W / "P4_table.csv")
    out += [(r.ID, "P4", r.flag) for r in p4.itertuples()]
    return pd.DataFrame(out, columns=["ID", "problem", "role"])


def indicator(r):
    f = []
    if r.role == "no_hypocentre":
        return "no_hypocentre"
    g = lambda k: r.get(k) if k in r and pd.notna(r.get(k)) else None  # noqa: E731
    if r.problem == "P1":
        if (g("pct_P_at_UW") is not None and g("pct_P_at_UW") >= 90 and g("P_span_s") is not None and g("P_span_s") <= 0.5
                and (g("P_dmax_km") or 0) > 8 and (g("n_S_picks") or 0) == 0):
            f.append("P1_common_mode")
        if r.get("calib_any") == "yes":
            f.append("P1_calibration")
        if r.get("wed_1914_1921") == "yes":
            f.append("P1_wed_window")
    elif r.problem == "P2":
        if g("nearest_min_weight") is not None and g("nearest_min_weight") < 0.1:
            f.append("P2_nearest_downweighted")
        if any(g(f"sp_obs_{s.split('.')[0]}") is not None and g(f"sp_pred_v51_{s.split('.')[0]}") is not None and
               abs(g(f"sp_obs_{s.split('.')[0]}") - g(f"sp_pred_v51_{s.split('.')[0]}")) > 0.5 for s in SP_STA):
            f.append("P2_sp_mismatch")
        if g("rms_v4") is not None and g("rms_v51") is not None and g("rms_v4") < g("rms_v51") - 0.05:
            f.append("P2_v4_fits_better")
    elif r.problem == "P3":
        if g("dh_comcat") is not None and (np.hypot(g("dh_comcat"), g("dz_comcat")) > 30 or abs(g("dt_comcat") or 0) > 4):
            f.append("P3_large_move")
        if g("dz_comcat") is not None and g("dz_comcat") > 20:
            f.append("P3_deeper_20km")
        if g("rms_comcat") is not None and g("rms_v51") is not None and g("rms_comcat") < g("rms_v51") - 0.05:
            f.append("P3_comcat_fits_better")
    elif r.problem == "P4":
        if r.get("two_cluster_scatter") == "yes":
            f.append("P4_two_cluster")
        if g("dz_v4_to_v51") is not None and abs(g("dz_v4_to_v51")) > 30:
            f.append("P4_depth_change_30km")
        if g("rms_v4") is not None and g("rms_v51") is not None and g("rms_v4") < g("rms_v51") - 0.05:
            f.append("P4_v4_fits_better")
        if g("sig_z") is not None and g("sig_z") > 10:
            f.append("P4_sigz_gt_10km")
    return ";".join(f) if f else "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=12)
    a = ap.parse_args()
    R = rows()
    c = C.catalog()
    loc_ids = sorted({i for i in R.ID if i in c.index and c.loc[i].loc_source == "nll3d"})
    fl = C.WORK / "export_light.parquet"
    if fl.exists():
        L = pd.read_parquet(fl)
    else:
        with mp.get_context("spawn").Pool(a.nproc) as p:
            L = pd.DataFrame(p.map(light, loc_ids, chunksize=8))
        L.to_parquet(fl)
    print("light rows", len(L), "errors", int(L.get("err", pd.Series(dtype=str)).notna().sum()), flush=True)
    p1ids = sorted(set(R[R.problem == "P1"].ID))
    fc = C.WORK / "export_calib.parquet"
    if fc.exists():
        K = pd.read_parquet(fc)
    else:
        with mp.get_context("spawn").Pool(a.nproc) as p:
            K = pd.DataFrame(p.map(calib, p1ids, chunksize=4))
        K.to_parquet(fc)
    K["calib_any"] = np.where((K[[f"calib_{s.split('.')[0]}" for s in CALIB_STA]] == "yes").any(axis=1), "yes", "no")
    print("calib rows", len(K), K.calib_any.value_counts().to_dict(), flush=True)
    import rv_select
    ps = rv_select.pick_stats().set_index("ID")
    ps["pct_P_at_UW"] = [round(100 * np.mean([s.endswith(".UW") for s in v.split()]), 1) if isinstance(v, str) else np.nan
                         for v in ps.P_stations]
    # assemble
    cat = c.reindex(R.ID)
    D = R.copy()
    D["time_utc"] = cat.time.values
    D["lat"] = cat.lat.values
    D["lon"] = cat.lon.values
    D["depth_v51"] = cat.depth.values
    D["depth_v4"] = cat.v4_depth.values
    D = D.merge(L.drop(columns=[k for k in ("err",) if k in L]), on="ID", how="left")
    for k in ("tier", "sig_z", "h_max", "gap", "dist_min", "region", "M", "M_type", "depth_quality", "event_flag", "loc_source"):
        D[k] = cat[k].values
    D["flags"] = [";".join(f for f in ("floor_limited", "far_only", "deep_pass", "five_phase") if bool(c.loc[i, f]) is True)
                  if i in c.index and c.loc[i].loc_source == "nll3d" else "" for i in R.ID]
    D = D.rename(columns={"dist_min": "nearest_station_km", "M": "magnitude", "M_type": "magnitude_type"})
    # ComCat for the no-hypocentre rows
    for i in (900002, 900003):
        cc = C.comcat_for(i)
        if cc:
            m = D.ID == i
            D.loc[m, "comcat_id"], D.loc[m, "comcat_depth"], D.loc[m, "comcat_time"] = cc["id"], cc["depth"], str(cc["t"])
    # P1
    t = pd.to_datetime(D.time_utc, utc=True, format="ISO8601")
    D["wed_1914_1921"] = np.where((t.dt.dayofweek == 2) & (t.dt.hour == 19) & t.dt.minute.between(14, 20), "yes", "no")
    D.loc[D.time_utc.isna(), "wed_1914_1921"] = ""
    psr = ps.reindex(D.ID)
    D["pct_P_at_UW"] = psr.pct_P_at_UW.values
    D["P_span_s"] = psr.P_span_s.round(3).values
    D["P_dmax_km"] = psr.P_dmax_km.round(1).values
    D["n_S_picks"] = psr.n_S.values
    D["has_S"] = np.where(psr.n_S.values > 0, "yes", "no")
    D = D.merge(K, on="ID", how="left")
    for k in [x for x in K.columns if x != "ID"]:
        D.loc[D.problem != "P1", k] = np.nan if D[k].dtype.kind == "f" else ""
    for k in ("pct_P_at_UW", "P_span_s", "P_dmax_km", "n_S_picks", "has_S", "wed_1914_1921"):
        D.loc[D.problem != "P1", k] = np.nan if D[k].dtype.kind == "f" else ""
    # P2 / P3 / P4
    spc = [k for k in D.columns if k.startswith("sp_")]
    D.loc[D.problem != "P2", spc] = np.nan
    for k in ("dz_comcat", "dh_comcat", "dt_comcat"):
        D.loc[D.problem != "P3", k] = np.nan
    D["dz_v4_to_v51"] = (D.depth_v51 - D.depth_v4).round(2)
    D.loc[D.problem != "P4", "dz_v4_to_v51"] = np.nan
    D.loc[D.problem != "P4", ["two_cluster_scatter", "two_cluster_depths"]] = ""
    # views
    views = set()
    for p in FOLDERS:
        views |= {(i, p) for i in pd.read_csv(C.WORK / "sets" / f"{p}_views.csv").ID}
    D["has_view"] = ["yes" if (i, p) in views else "no" for i, p in zip(D.ID, D.problem)]
    D["view_path"] = [f"{FOLDERS[p]}/index.html#ev-{i}" if (i, p) in views else "" for i, p in zip(D.ID, D.problem)]
    D["auto_indicator_rule_not_verdict"] = [indicator(r) for _, r in D.iterrows()]
    for k in REVIEW_COLS:
        D[k] = ""
    cols = (["ID", "problem", "role", "time_utc", "lat", "lon", "depth_v51", "depth_v4", "comcat_id", "comcat_depth",
             "tier", "sig_z", "h_max", "nP", "nS", "gap", "nearest_station_km", "flags", "depth_quality", "event_flag",
             "region", "magnitude", "magnitude_type", "rms_v51", "rms_v4", "rms_comcat", "nearest_sta",
             "nearest_min_weight", "n_weight_lt_0p1", "scatter_z16", "scatter_z50", "scatter_z84",
             "pct_P_at_UW", "P_span_s", "P_dmax_km", "n_S_picks", "has_S", "wed_1914_1921"]
            + [f"{p}_{s.split('.')[0]}" for s in CALIB_STA for p in ("calib", "quiet_s", "step")] + ["calib_any"]
            + [f"{k}_{s.split('.')[0]}" for s in SP_STA for k in ("sp_obs", "sp_pred_v51", "sp_pred_v4")]
            + ["dz_comcat", "dh_comcat", "dt_comcat", "comcat_time", "dz_v4_to_v51", "two_cluster_scatter",
               "two_cluster_depths", "has_view", "view_path", "auto_indicator_rule_not_verdict"] + REVIEW_COLS)
    for k in cols:
        if k not in D:
            D[k] = np.nan
    D = D[cols]
    for k in ("lat", "lon"):
        D[k] = D[k].round(5)
    for k in ("depth_v51", "depth_v4", "comcat_depth", "sig_z", "h_max", "gap", "nearest_station_km", "magnitude"):
        D[k] = pd.to_numeric(D[k], errors="coerce").round(3)
    D.to_csv(OUT_CSV, index=False, lineterminator="\n")
    REPO.mkdir(parents=True, exist_ok=True)
    D.to_csv(REPO / "review_v51_events.csv", index=False, lineterminator="\n")
    cnt = {}
    for p, g in D.groupby("problem"):
        toks = g.auto_indicator_rule_not_verdict.str.split(";").explode()
        cnt[p] = dict(rows=int(len(g)), any_rule=int((g.auto_indicator_rule_not_verdict != "none").sum()),
                      by_rule=toks.value_counts().to_dict(), by_role=g.role.value_counts().to_dict(),
                      with_view=int((g.has_view == "yes").sum()))
    (C.WORK / "export_counts.json").write_text(json.dumps(dict(counts=cnt, rules=RULES), indent=1))
    (REPO / "summary" / "export_counts.json").write_text(json.dumps(dict(counts=cnt, rules=RULES), indent=1))
    print(json.dumps(cnt, indent=1))
    print(OUT_CSV, len(D), "rows", OUT_CSV.stat().st_size // 1024, "kB")


if __name__ == "__main__":
    main()

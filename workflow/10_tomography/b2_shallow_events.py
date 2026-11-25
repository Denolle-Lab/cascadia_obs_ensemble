"""Track B, B2: shallow events added to a region's tomoDD set (upper-crust coverage test; PI 2026-11-23).

The B1/B1b selection (tiers A-C, nphs >= 12, gap <= 200, core-station balance) keeps few events above 12 km
(Mendocino-Gorda: 8% of the inversion set vs 18% of the v4-NLL3D catalogue in the box). Depth uncertainty is what
the joint inversion is meant to reduce, so this selection does not use tier, sig_z or gap:

* Events: v4-NLL3D (full-precision hypocentres, as b1_select_events.select_events) in the region's event box,
  depth < max_depth_km, nphs >= min_nphs, not already in B1b, not above the region's Air_dep (tomoDD air quakes);
  declustered as D-B003 (cubic cells, keep the most-picked per cell).
* Picks: v4 picks with B1 sigma classes (b1_select_events.load_picks); B1 pick QC with the B1 thresholds frozen
  (from the B1 run, params `tomodd.shallow_add.qc`): no NLL residual, iter1 weight 0, duplicates, residual
  outliers, S-P ratio and S-P differential outliers (station medians from these picks); then only stations already
  in the base region's inputs (the balanced core set); events with >= min_picks picks left.
* w_bal: median w_bal of the base region's picks at the same station and phase (fallback: phase median).
Writes <base>/<out_dir>/{events.csv, picks.csv} in B1b format, read by b2_tomodd_inputs.py through the region's
`extra_b1b`.

  python b2_shallow_events.py      (params tomodd.shallow_add)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import load_params, write_manifest  # noqa: E402
import b1_select_events as b1  # noqa: E402

P = load_params()
T = P["tomodd"]
A = T["shallow_add"]
BASE = Path(P["paths"]["base"])


def catalog():
    cat = pd.read_csv(P["paths"]["trackA_catalog"])
    loc = pd.read_csv(Path(P["paths"]["reloc_v5"]) / "runs" / P["paths"]["trackA_run"] / "locations.csv",
                      usecols=["status", "ID", "time", "lat", "lon", "depth"])
    loc = loc[loc.status == "LOCATED"].drop_duplicates("ID").set_index("ID")
    m = cat.loc_source == "nll3d"
    for c in ("time", "lat", "lon", "depth"):
        cat.loc[m, c] = cat.loc[m, "ID"].map(loc[c]).values
    return cat


def main():
    R = T["regions"][A["base_region"]]
    base_inp = Path(T["runs"]) / A["base_region"] / "inputs"
    air = json.loads((base_inp / "summary.json").read_text())["air_dep_km"]
    b1b = pd.read_csv(BASE / "b1b_balanced" / "events.csv", usecols=["ID"])
    cat = catalog()
    (la0, la1), (lo0, lo1) = R["events_lat"], R["events_lon"]
    box = cat[cat.lat.between(la0, la1) & cat.lon.between(lo0, lo1)]
    c = box[(box.depth < A["max_depth_km"]) & (box.nphs >= A["min_nphs"])]
    n = {"box": len(box), "shallow_nphs": len(c), "in_b1b": int(c.ID.isin(b1b.ID).sum())}
    c = c[~c.ID.isin(b1b.ID)]
    n["above_air_dep"] = int((c.depth < air).sum())
    c = c[c.depth >= air].copy()
    n["candidates"] = len(c)
    x, y = b1.PROJ(c.lon.values, c.lat.values)
    cs = A["decluster_cell_km"]
    c["cell"] = list(zip(np.floor(x / 1e3 / cs).astype(int), np.floor(y / 1e3 / cs).astype(int),
                         np.floor(c.depth / cs).astype(int)))
    c = c.sort_values(["nphs", "rms", "ID"], ascending=[False, True, True])
    k = c[c.groupby("cell").cumcount() < A["decluster_keep_per_cell"]].sort_values("ID").copy()
    k["rule"] = "shallow"
    n["declustered"] = len(k)
    print(n, flush=True)

    st = b1.load_stations()
    pk = b1.load_picks(k, st)
    n["picks_loaded"] = len(pk)
    Q = A["qc"]
    pk["qc"] = ""

    def flag(mask, reason):
        m = mask & (pk.qc == "")
        pk.loc[m, "qc"] = reason
        n[f"qc_{reason}"] = int(m.sum())

    flag(pk.res_i1.isna(), "no_nll")
    flag(pk.w_i1 <= 0, "w0")
    o = pk.assign(_pp=pk.pick_prob.fillna(-1)).sort_values(["_pp", "pick_time"], ascending=[False, True])
    dup = o.duplicated(["ID", "station", "phase"], keep="first")
    flag(pk.index.isin(o.index[dup.values]), "duplicate")
    for ph in ("P", "S"):
        flag((pk.phase == ph) & ((pk.res_i1 - Q[ph]["median"]).abs() > Q[ph]["threshold"]), "res")
    ok = pk[pk.qc == ""]
    pv = ok.pivot_table(index=["ID", "station"], columns="phase", values=["tt", "res_noterm_i1"], aggfunc="first").dropna()
    pv.columns = [f"{a}_{b}" for a, b in pv.columns]
    pv = pv.reset_index()
    pv["ratio"] = pv.tt_S / pv.tt_P
    pv["dres_c"] = (pv.res_noterm_i1_S - pv.res_noterm_i1_P)
    pv["dres_c"] -= pv.groupby("station").dres_c.transform("median")
    lo, hi = P["pick_qc"]["sp_ratio_range"]
    bad = pv[(pv.ratio < lo) | (pv.ratio > hi) | ((pv.dres_c - Q["SP"]["median"]).abs() > Q["SP"]["threshold"])]
    key = set(zip(bad.ID, bad.station))
    flag((pk.phase == "S") & pd.Series([(a, b) in key for a, b in zip(pk.ID, pk.station)], index=pk.index), "sp")
    core = pd.read_csv(base_inp / "station_codes.csv").station
    flag(~pk.station.isin(core), "not_core_station")
    kept = pk[pk.qc == ""].copy()
    npk = kept.groupby("ID").size()
    good = npk[npk >= A["min_picks"]].index
    n["events_lt_min_picks"] = int(len(k) - len(good))
    kept = kept[kept.ID.isin(good)]
    k = k[k.ID.isin(good)]

    # balance weight: base region's median at the same station and phase
    bp = pd.read_csv(base_inp / "picks_used.csv", usecols=["station", "phase", "w_bal"])
    wsp = bp.groupby(["station", "phase"]).w_bal.median()
    wp = bp.groupby("phase").w_bal.median()
    kept["w_bal"] = pd.MultiIndex.from_frame(kept[["station", "phase"]]).map(wsp)
    kept["w_bal"] = kept.w_bal.fillna(kept.phase.map(wp))

    out = BASE / A["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    ev = k.rename(columns={"time": "origin_time", "depth": "depth_bsl_km"})
    ev = ev[["ID", "origin_time", "lat", "lon", "depth_bsl_km", "tier", "nphs", "gap", "rms", "sig_z", "region", "rule"]]
    ev = ev.assign(nP=ev.ID.map(kept[kept.phase == "P"].groupby("ID").size()).fillna(0).astype(int),
                   nS=ev.ID.map(kept[kept.phase == "S"].groupby("ID").size()).fillna(0).astype(int))
    ev.to_csv(out / "events.csv", index=False)
    ren = {"ID": "event_ID", "tt": "travel_time_s", "res_i1": "iter1_res_s", "w_i1": "iter1_weight",
           "tcorr_i1": "iter1_station_term_s", "res_i0": "iter0_res_s", "dist_i1": "epi_dist_km"}
    cols = ["ID", "station", "network", "phase", "pick_time", "tt", "sigma_s", "pick_prob", "res_i1", "w_i1",
            "tcorr_i1", "res_i0", "dist_i1", "sta_lat", "sta_lon", "sta_elev_km", "is_obs", "rule", "w_bal"]
    po = kept.sort_values(["ID", "pick_time"])[cols].rename(columns=ren)
    po["pick_time"] = po.pick_time.dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    po.to_csv(out / "picks.csv", index=False)
    n.update(events=len(ev), picks=len(po), P=int((po.phase == "P").sum()), S=int((po.phase == "S").sum()),
             obs_picks=int(po.is_obs.sum()), depth_median=float(ev.depth_bsl_km.median()),
             tiers=ev.tier.value_counts().sort_index().to_dict(), w_bal_from_phase_median=int(
                 (~pd.MultiIndex.from_frame(po[["station", "phase"]]).isin(wsp.index)).sum()))
    print(n)
    write_manifest("b2_shallow_events", out, inputs=[P["paths"]["trackA_catalog"], BASE / "b1b_balanced" / "events.csv",
                                                     base_inp / "station_codes.csv", base_inp / "picks_used.csv"],
                   outputs=[out / "events.csv", out / "picks.csv"], params=A, counts=n)


if __name__ == "__main__":
    main()

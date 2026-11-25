#!/usr/bin/env python3
"""
Catalog v5.1 -> the arrivals table that `route_a_day_amplitudes.py select` reads.

Same role as v4_build_arrivals.py, for catalog v5.1 (catalog v4.1 events relocated
with NonLinLoc in model v0.3.1; picks re-timed on drifting channels, D-B072/D-B075).

Picks: Cascadia_picks_catalog_v4.1t.csv.gz. The pick_id of each arrival comes from
the v4 cache (data/catalog_v4/ridge_report/v4_pick_ids.parquet of the main checkout,
gitignored; input paths and their defaults in v51_paths.py), joined on event,
station and phase; v4.1 keeps every v4 pick of the events it keeps, so the join is
exact. The picks v4.1 added (events 900001-900003 and the repicks of 172114) are not
ver6 picks: no pick_id, so no Route A amplitude (FIXES.md, v4.1).

Pick times are not used here. The amplitude tables (amps_v6_all) are keyed by
pick_id and were measured at the catalog (uncorrected) pick time on day traces read
and merged the way the picker read them, so the window sits on the samples the
picker picked; the re-timing changes the time label of a pick, not which samples
the window covers (see the v5.1 magnitude report).

Hypocentres:
  * loc_source nll3d (located in v5.1) and comcat (two v4.1 additions): the v5.1
    lat, lon, depth (km below sea level) -> arrivals_v51.csv.
  * loc_source none (623 events NonLinLoc rejected): no hypocentre in v5.1. They
    are left out of arrivals_v51.csv (no magnitude in the main run). A separate
    file arrivals_v51_nohypo_v4hyp.csv carries their arrivals with the v4
    hypocentre, for the documented fallback magnitude (column ML_v4hyp, computed
    with the v5.1 station terms held fixed, never mixed into M).

Hypocentral distance (select): hypot(epicentral, depth + station elevation / 1000);
v5.1 depth is below sea level and the station elevation is above it.

Also writes catalog_v51_for_phase2.csv: the v5.1 origins in the column names that
phase2_anchor_comcat_ml.py expects ("Event ID", "Origin Time (UTC)", ...), origin
times written as ISO UTC with a trailing Z (v5.1 mixes both forms).

Usage:
  python v51_build_arrivals.py --outdir /wd1/mdenolle_data/catalog_v5/v5.1_analysis/mag
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import v51_paths as vp


def iso_z(s: pd.Series) -> pd.Series:
    t = pd.to_datetime(s.str.strip(), utc=True, format="ISO8601")
    return t.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ").where(t.notna(), "")


def arrivals(picks, ev, sta):
    s = picks["station"].str.split(".", expand=True)          # STA.NET
    arr = pd.DataFrame({
        "arid": picks["row"].to_numpy(),                        # 0-based row of the v4.1t file
        "event_id": picks["event_ID"].astype(int).to_numpy(),
        "network": s[1].to_numpy(),
        "station": (s[1] + "." + s[0]).to_numpy(),
        "phase": picks["phase"].str.upper().to_numpy(),
        "pick_id": picks["pick_id"].astype("int64").to_numpy(),
        "timing_correction_s": picks["timing_correction_s"].to_numpy(),
    })
    arr = arr.join(ev, on="event_id", how="inner")
    st = sta.reindex(arr["station"]).to_numpy()
    arr["stla"], arr["stlo"], arr["stel_m"] = st[:, 0], st[:, 1], st[:, 2]
    bad = arr[["evla", "evlo", "evdp", "stla"]].isna().any(axis=1)
    assert not bad.any(), f"{bad.sum()} arrivals without event or station coordinates"
    return arr


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--catalog", default=str(vp.V51_CATALOG), help="catalog v5.1 csv")
    ap.add_argument("--picks", default=str(vp.V41T_PICKS), help="v4.1t pick file")
    ap.add_argument("--pick-ids", default=None,
                    help="v4 pick_id cache (default data/catalog_v4/ridge_report/v4_pick_ids.parquet "
                         "of the main checkout, see v51_paths.py)")
    ap.add_argument("--stations", default=str(vp.STATIONS))
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)
    pick_ids = a.pick_ids or vp.main_data("catalog_v4/ridge_report/v4_pick_ids.parquet")

    cat = pd.read_csv(a.catalog)
    picks = pd.read_csv(a.picks, skipinitialspace=True)
    picks.columns = [c.strip() for c in picks.columns]
    picks["row"] = np.arange(len(picks))
    for c in ("station", "phase"):
        picks[c] = picks[c].str.strip()
    ids = pd.read_parquet(pick_ids)
    k = ["event_ID", "station", "phase"]
    assert not picks.duplicated(k).any(), "event/station/phase not unique in v4.1t"
    picks = picks.merge(ids[k + ["pick_id"]], on=k, how="left", validate="one_to_one")
    n_all = len(picks)
    no_id = picks["pick_id"].isna()
    print(f"v4.1t picks {n_all:,}; without a ver6 pick_id (v4.1 additions): {no_id.sum():,} "
          f"in events {sorted(picks.loc[no_id, 'event_ID'].unique().tolist())}")
    picks = picks[~no_id].copy()
    assert not picks["pick_id"].duplicated().any()

    sta = pd.read_csv(a.stations)
    sta["key"] = sta["network"] + "." + sta["station"]
    sta = sta.drop_duplicates("key").set_index("key")[["lat", "lon", "elev_m"]]

    has = cat["loc_source"].isin(["nll3d", "comcat"])
    ev = cat.loc[has].set_index("ID")[["lat", "lon", "depth"]]
    ev.columns = ["evla", "evlo", "evdp"]
    arr = arrivals(picks, ev, sta)
    out = os.path.join(a.outdir, "arrivals_v51.csv")
    arr.to_csv(out, index=False)
    print(f"{len(arr):,} arrivals, {arr.event_id.nunique():,} events, "
          f"{arr.station.nunique()} stations -> {out}")

    none = cat.loc[~has].set_index("ID")[["v4_lat", "v4_lon", "v4_depth"]]
    none.columns = ["evla", "evlo", "evdp"]
    arr0 = arrivals(picks, none, sta)
    out0 = os.path.join(a.outdir, "arrivals_v51_nohypo_v4hyp.csv")
    arr0.to_csv(out0, index=False)
    print(f"{len(arr0):,} arrivals of {arr0.event_id.nunique():,} of the {len(none):,} "
          f"events without a v5.1 hypocentre (v4 hypocentre) -> {out0}")

    c = cat.loc[has]
    p2 = pd.DataFrame({"Event ID": c["ID"], "Origin Time (UTC)": iso_z(c["time"]),
                       "Latitude": c["lat"], "Longitude": c["lon"], "Depth (km)": c["depth"],
                       "Num. P": np.nan, "Num. S": np.nan})
    out2 = os.path.join(a.outdir, "catalog_v51_for_phase2.csv")
    p2.to_csv(out2, index=False)
    print(f"{len(p2):,} origins -> {out2}")


if __name__ == "__main__":
    main()

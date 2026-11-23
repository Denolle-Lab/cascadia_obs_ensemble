#!/usr/bin/env python3
"""
Catalog v4 -> the arrivals table that `route_a_day_amplitudes.py select` reads.

Catalog v4 (data/catalog_v4/) has no pick_id; the ver6 pick_id of each arrival was
matched by station and time in ridge_transform_v4_report.py and cached in
data/catalog_v4/ridge_report/v4_pick_ids.parquet (sorted by pick time; joined here on event, station and phase).
Station coordinates come from the ver5/ver6 station table (one row per station).

Also writes catalog_v4_for_phase2.csv: the v4 origins in the column names that
phase2_anchor_comcat_ml.py expects ("Event ID", "Origin Time (UTC)", ...).

Usage:
  python v4_build_arrivals.py --outdir ../../data/magnitude_v4
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
CAT = os.path.join(ROOT, "data/catalog_v4/Cascadia_relocated_catalog_v4.csv")
PICKS = os.path.join(ROOT, "data/catalog_v4/Cascadia_picks_catalog_v4.csv")
PICK_IDS = os.path.join(ROOT, "data/catalog_v4/ridge_report/v4_pick_ids.parquet")
STATIONS = "/wd1/mdenolle_data/picks_share/stations/stations_ver5_v6.csv"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=os.path.join(ROOT, "data/magnitude_v4"))
    ap.add_argument("--stations", default=STATIONS)
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)

    cat = pd.read_csv(CAT, skipinitialspace=True)
    cat.columns = [c.strip() for c in cat.columns]
    picks = pd.read_csv(PICKS, skipinitialspace=True)
    picks.columns = [c.strip() for c in picks.columns]
    ids = pd.read_parquet(PICK_IDS)
    # the cache is sorted by pick time; (event, station, phase) is unique in v4
    k = ["event_ID", "station", "phase"]
    for c in ("station", "phase"):
        picks[c] = picks[c].str.strip()
    picks = picks.merge(ids[k + ["pick_id"]], on=k, how="left", validate="one_to_one")
    assert picks["pick_id"].notna().all() and not picks["pick_id"].duplicated().any()

    sta = pd.read_csv(a.stations)
    sta["key"] = sta["network"] + "." + sta["station"]
    sta = sta.set_index("key")[["lat", "lon", "elev_m"]]

    s = picks["station"].str.split(".", expand=True)     # STA.NET
    arr = pd.DataFrame({
        "arid": np.arange(len(picks)),
        "event_id": picks["event_ID"].astype(int),
        "network": s[1],
        "station": s[1] + "." + s[0],
        "phase": picks["phase"].str.upper(),
        "pick_id": picks["pick_id"].astype("int64"),
    })
    ev = cat.set_index("ID")[["latitude", "longitude", "depth(km)"]]
    ev.columns = ["evla", "evlo", "evdp"]
    arr = arr.join(ev, on="event_id")
    st = sta.reindex(arr["station"]).to_numpy()
    arr["stla"], arr["stlo"], arr["stel_m"] = st[:, 0], st[:, 1], st[:, 2]
    bad = arr[["evla", "stla"]].isna().any(axis=1)
    assert not bad.any(), f"{bad.sum()} arrivals without event or station coordinates"
    out = os.path.join(a.outdir, "arrivals_v4.csv")
    arr.to_csv(out, index=False)
    print(f"{len(arr):,} arrivals, {arr.event_id.nunique():,} events, "
          f"{arr.station.nunique()} stations -> {out}")

    # phase2 reads the origin times from the catalog in the v3 column names
    p2 = pd.DataFrame({"Event ID": cat["ID"], "Origin Time (UTC)": cat["datetime"].str.strip(),
                       "Latitude": cat["latitude"], "Longitude": cat["longitude"],
                       "Depth (km)": cat["depth(km)"], "Num. P": cat["Num_p_picks"],
                       "Num. S": cat["Num_s_picks"]})
    out2 = os.path.join(a.outdir, "catalog_v4_for_phase2.csv")
    p2.to_csv(out2, index=False)
    print(f"{len(p2):,} origins -> {out2}")


if __name__ == "__main__":
    main()

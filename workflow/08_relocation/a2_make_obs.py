"""Track A / B4, step A2 input: catalog v4 picks -> NonLinLoc NLLOC_OBS files, one per 25 km tile.

Each event block starts with `PUBLIC_ID v4_<event_ID>` and is followed by a blank line.
Pick error (Gaussian, s) by class of the ensemble pick probability p (v4 `pick_prob`, 0.05-0.90,
median 0.38; missing for 19% of picks, 2011-2015): p >= 0.6 -> P 0.05 / S 0.10; 0.3 <= p < 0.6 or
missing -> 0.10 / 0.20; p < 0.3 -> 0.15 / 0.30.
Stations: picks of a station-phase without a fine time grid (with --time: no grid header in that folder; without:
station missing from --stations, the GTSRCE list) are dropped, listed in picks_dropped.csv and counted. Catalog v4.0:
none. Catalog v4.1: 246 picks at 105 stations (NC, BK, NP...) of the four events it re-picked or added, 172114 (94 of
160 picks), 900001 (136 of 200), 900002 (15 of 71), 900003 (1 of 12); the B4 grids cover the 696 stations of v4.0.
A grid header without its complete buffer stops the script (a2_run_nlloc.py checks the grids again before running).

The output folder must be new or empty (never overwrites an obs set, e.g. reloc_v5/obs_tiles of v5.0); it gets
<tile>.obs, tiles.csv (tile, ix, iy, n_events, search box), picks_dropped.csv and obs_meta.json (inputs with md5,
counts).

Run (v5.1: catalog v4.1 picks and events):
  LD_LIBRARY_PATH=.pixi/envs/default/lib .pixi/envs/default/bin/python workflow/08_relocation/a2_make_obs.py \
      --picks data/catalog_v4/Cascadia_picks_catalog_v4.csv --events data/catalog_v4/Cascadia_relocated_catalog_v4.csv \
      --out /wd1/mdenolle_data/reloc_v5/obs_v51 [--time time_b4_v0.3.1m]
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Proj

B = Path("/wd1/mdenolle_data/reloc_v5")
# NLL grid (must match a1_build_model.py)
PROJ = Proj(proj="lcc", lat_0=45.0, lon_0=-125.8, lat_1=41.0, lat_2=49.0, ellps="WGS84")
X0, Y0, X1, Y1 = -513.0, -709.0, 457.0, 724.0
TILE, MARGIN = 25.0, 75.0
ERR_P = {0: 0.15, 1: 0.10, 2: 0.05}   # class -> sigma; S is twice P


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--picks", required=True, help="catalog v4 picks CSV")
    ap.add_argument("--events", required=True, help="catalog v4 events CSV (v4 epicentres place events in tiles)")
    ap.add_argument("--out", required=True, help="new (or empty) output folder")
    ap.add_argument("--time", help="fine time-grid folder (under reloc_v5 or absolute): every station-phase must have "
                                   "a complete grid there")
    ap.add_argument("--stations", default=str(B / "model_vpvs176/stations_GTSRCE.in"),
                    help="GTSRCE station list (used without --time)")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"REFUSED: {out} is not empty; obs sets are never overwritten, use a new folder")
    p = pd.read_csv(a.picks, skipinitialspace=True)
    p = p.rename(columns={"travel_time(s)": "tt"})
    n0 = len(p)
    if a.time:
        tdir = Path(a.time) if a.time.startswith("/") else B / a.time
        have, broken = set(), []
        for ph, sta in p[["phase", "station"]].drop_duplicates().itertuples(index=False):
            h, buf = tdir / f"cascadia3d.{ph}.{sta}.time.hdr", tdir / f"cascadia3d.{ph}.{sta}.time.buf"
            if not h.exists():
                continue
            n = [int(v) for v in h.read_text().split()[:3]]
            if not buf.exists() or buf.stat().st_size != 4 * n[0] * n[1] * n[2]:
                broken.append((ph, sta))
            have.add((ph, sta))
        if broken:
            sys.exit(f"REFUSED: {len(broken)} time grids in {tdir} have no complete buffer: {broken[:10]}")
        keep = pd.Series([k in have for k in zip(p.phase, p.station)], index=p.index)
    else:
        sta = {l.split()[1] for l in Path(a.stations).read_text().splitlines() if l.startswith("GTSRCE")}
        keep = p.station.isin(sta)
    drop = p[~keep]
    p = p[keep]
    print(f"{len(drop)} picks dropped (no time grid for the station-phase), {drop.station.nunique()} stations, "
          f"events {drop.groupby('event_ID').size().to_dict() if len(drop) < 5000 else drop.event_ID.nunique()}")
    t = pd.to_datetime(p.pick_datetime.str.rstrip("Z"), format="ISO8601")
    p["date"] = t.dt.strftime("%Y%m%d")
    p["hhmm"] = t.dt.strftime("%H%M")
    p["sec"] = t.dt.second + t.dt.microsecond / 1e6
    cls = np.select([p.pick_prob >= 0.6, p.pick_prob >= 0.3, p.pick_prob.isna()], [2, 1, 1], 0)
    p["err"] = pd.Series(cls, index=p.index).map(ERR_P) * np.where(p.phase == "S", 2.0, 1.0)
    # group events in TILE x TILE km tiles of their v4 epicentre (NLL grid coordinates); each tile is located in its own
    # box (tile +- MARGIN km, clipped to the model), which must lie inside the time grids NLLoc uses (a2_run_nlloc.py)
    cat = pd.read_csv(a.events, skipinitialspace=True, usecols=["ID", "latitude", "longitude"])
    x, y = PROJ(cat.longitude.values, cat.latitude.values)
    if ((x / 1e3 < X0) | (x / 1e3 > X1) | (y / 1e3 < Y0) | (y / 1e3 > Y1)).any():
        sys.exit("REFUSED: events outside the model lattice")
    cat["ix"] = np.floor((x / 1e3 - X0) / TILE).astype(int)
    cat["iy"] = np.floor((y / 1e3 - Y0) / TILE).astype(int)
    cat["tile"] = "t" + cat.ix.astype(str).str.zfill(2) + "_" + cat.iy.astype(str).str.zfill(2)
    n_ev = p.event_ID.nunique()
    p = p.merge(cat[["ID", "tile"]], left_on="event_ID", right_on="ID")
    if p.event_ID.nunique() != n_ev:
        sys.exit(f"REFUSED: {n_ev - p.event_ID.nunique()} picked events are not in {a.events}")
    p = p.assign(t=pd.to_datetime(p.pick_datetime.str.rstrip("Z"), format="ISO8601"))
    p = p.sort_values(["event_ID", "t"])
    out.mkdir(parents=True, exist_ok=True)
    drop.to_csv(out / "picks_dropped.csv", index=False)
    for tile, g in p.groupby("tile"):
        lines = []
        for eid, e in g.groupby("event_ID", sort=False):
            lines.append(f"PUBLIC_ID v4_{eid}")
            for r in e.itertuples():
                lines.append(f"{r.station:<10s} ?    ?    ? {r.phase:<6s} ? {r.date} {r.hhmm} {r.sec:7.4f} "
                             f"GAU {r.err:9.2e} -1.00e+00 -1.00e+00 -1.00e+00 1.0")
            lines.append("")
        (out / f"{tile}.obs").write_text("\n".join(lines) + "\n")
    tiles = cat.groupby(["tile", "ix", "iy"]).size().rename("n_events").reset_index()
    tiles["box_x0"] = np.maximum(X0 + tiles.ix * TILE - MARGIN, X0)
    tiles["box_y0"] = np.maximum(Y0 + tiles.iy * TILE - MARGIN, Y0)
    tiles["box_x1"] = np.minimum(X0 + (tiles.ix + 1) * TILE + MARGIN, X1)
    tiles["box_y1"] = np.minimum(Y0 + (tiles.iy + 1) * TILE + MARGIN, Y1)
    tiles = tiles[tiles.tile.isin(p.tile.unique())]      # tiles with picked events (an obs file)
    tiles.to_csv(out / "tiles.csv", index=False)
    meta = {"script": str(Path(__file__).resolve()), "argv": sys.argv[1:],
            "utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {k: {"path": str(Path(v).resolve()), "md5": md5(v)} for k, v in
                       (("picks", a.picks), ("events", a.events))},
            "time": a.time, "counts": {"picks_read": n0, "picks_dropped": n0 - len(p), "picks": len(p),
                                       "events": int(p.event_ID.nunique()), "tiles": len(tiles),
                                       "events_in_catalog": len(cat)}}
    (out / "obs_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(len(tiles), "tiles;", tiles.n_events.describe().round(0).to_dict())
    print(p.groupby("phase").err.describe())
    print(f"{len(p)} picks of {p.event_ID.nunique()} events -> {out}")


if __name__ == "__main__":
    main()

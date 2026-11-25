"""Track A, step A2 input: catalog v4 picks -> NonLinLoc NLLOC_OBS files, one per 25 km tile.

Each event block starts with `PUBLIC_ID v4_<event_ID>` and is followed by a blank line.
Pick error (Gaussian, s) by class of the ensemble pick probability p (v4 `pick_prob`, 0.05-0.90,
median 0.38; missing for 19% of picks, 2011-2015): p >= 0.6 -> P 0.05 / S 0.10; 0.3 <= p < 0.6 or
missing -> 0.10 / 0.20; p < 0.3 -> 0.15 / 0.30. Stations without a time grid are dropped (none expected).

Run: LD_LIBRARY_PATH=.pixi/envs/default/lib .pixi/envs/default/bin/python workflow/08_relocation/a2_make_obs.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import Proj

REPO = Path(__file__).resolve().parents[2]
B = Path("/wd1/mdenolle_data/reloc_v5")
PICKS = REPO / "data/catalog_v4/Cascadia_picks_catalog_v4.csv"
CATALOG = REPO / "data/catalog_v4/Cascadia_relocated_catalog_v4.csv"
# NLL grid (must match a1_build_model.py)
PROJ = Proj(proj="lcc", lat_0=45.0, lon_0=-125.8, lat_1=41.0, lat_2=49.0, ellps="WGS84")
X0, Y0, X1, Y1 = -513.0, -709.0, 457.0, 724.0
TILE, MARGIN = 25.0, 75.0
ERR_P = {0: 0.15, 1: 0.10, 2: 0.05}   # class -> sigma; S is twice P


def main():
    p = pd.read_csv(PICKS, skipinitialspace=True)
    p = p.rename(columns={"travel_time(s)": "tt"})
    sta = {l.split()[1] for l in (B / "model_vpvs176/stations_GTSRCE.in").read_text().splitlines()}
    n0 = len(p)
    p = p[p.station.isin(sta)]
    print(f"{n0 - len(p)} picks dropped (no station grid)")
    t = pd.to_datetime(p.pick_datetime.str.rstrip("Z"), format="ISO8601")
    p["date"] = t.dt.strftime("%Y%m%d")
    p["hhmm"] = t.dt.strftime("%H%M")
    p["sec"] = t.dt.second + t.dt.microsecond / 1e6
    cls = np.select([p.pick_prob >= 0.6, p.pick_prob >= 0.3, p.pick_prob.isna()], [2, 1, 1], 0)
    p["err"] = pd.Series(cls, index=p.index).map(ERR_P) * np.where(p.phase == "S", 2.0, 1.0)
    # group events in TILE x TILE km tiles of their v4 epicentre (NLL grid coordinates). NLLoc needs the
    # search grid inside every station's +-250 km time grid, so each tile is located in its own box
    # (tile +- MARGIN km, clipped to the model); stations farther than ~150-175 km drop out.
    cat = pd.read_csv(CATALOG, skipinitialspace=True, usecols=["ID", "latitude", "longitude"])
    x, y = PROJ(cat.longitude.values, cat.latitude.values)
    cat["ix"] = np.floor((x / 1e3 - X0) / TILE).astype(int)
    cat["iy"] = np.floor((y / 1e3 - Y0) / TILE).astype(int)
    cat["tile"] = "t" + cat.ix.astype(str).str.zfill(2) + "_" + cat.iy.astype(str).str.zfill(2)
    p = p.merge(cat[["ID", "tile"]], left_on="event_ID", right_on="ID")
    p = p.assign(t=pd.to_datetime(p.pick_datetime.str.rstrip("Z"), format="ISO8601"))
    p = p.sort_values(["event_ID", "t"])
    (B / "obs_tiles").mkdir(parents=True, exist_ok=True)
    for tile, g in p.groupby("tile"):
        out = []
        for eid, e in g.groupby("event_ID", sort=False):
            out.append(f"PUBLIC_ID v4_{eid}")
            for r in e.itertuples():
                out.append(f"{r.station:<10s} ?    ?    ? {r.phase:<6s} ? {r.date} {r.hhmm} {r.sec:7.4f} "
                           f"GAU {r.err:9.2e} -1.00e+00 -1.00e+00 -1.00e+00 1.0")
            out.append("")
        (B / f"obs_tiles/{tile}.obs").write_text("\n".join(out) + "\n")
    tiles = cat.groupby(["tile", "ix", "iy"]).size().rename("n_events").reset_index()
    tiles["box_x0"] = np.maximum(X0 + tiles.ix * TILE - MARGIN, X0)
    tiles["box_y0"] = np.maximum(Y0 + tiles.iy * TILE - MARGIN, Y0)
    tiles["box_x1"] = np.minimum(X0 + (tiles.ix + 1) * TILE + MARGIN, X1)
    tiles["box_y1"] = np.minimum(Y0 + (tiles.iy + 1) * TILE + MARGIN, Y1)
    tiles.to_csv(B / "obs_tiles/tiles.csv", index=False)
    print(len(tiles), "tiles;", tiles.n_events.describe().round(0).to_dict())
    print(p.groupby("phase").err.describe())

if __name__ == "__main__":
    main()

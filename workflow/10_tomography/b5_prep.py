"""D-B048: data of one Blanco swarm case for both relocation codes (b5_common.py describes the format).

Events: members of the case's zbz families (events_zbz.csv.gz family_id; event_id = catalog ID, checked against the
v4 origin time) located in 3D by B4 (loc_source nll3d); start = B4 hypocentre and origin time (model v0.3.0), read
at full precision from the run's NLLoc solutions (b5_test.catalog_full, status LOCATED; EFT undone here as in
b4_catalog.py) because the catalog CSVs round longitude to 0.01 deg; the two agree within that rounding (checked).
Picks: catalog v4.0 picks of those events (the B4 input), travel time = pick - B4 origin time, kept where the event
lies inside the station's B4 time grid with grid_margin_km to spare (GrowClust3D can only use those; B4 dropped the
same far stations). Weight (0.05 / sigma)^2 with the D-B008 sigma of the pick_prob class.
Differential times: b5_common.diff_times with b5_test.dt (= tomodd.dt). One file, fed to both codes.

  python b5_prep.py CASE
"""
import sys

import numpy as np
import pandas as pd

from b5_common import B5, Grids, diff_times, eft_depth, grid_stations, lambert_km, pick_class, pick_sigma, unflatten, \
    weight, write_case


def main():
    case = sys.argv[1]
    fams = B5["cases"][case]
    mem = pd.read_csv(B5["members"])
    fam = pd.read_csv(B5["families"]).set_index("family_id").loc[fams]
    m = mem[mem.family_id.isin(fams)]
    cat = pd.read_csv(B5["catalog"]).set_index("ID")
    c = cat.loc[m.event_id]
    dt_check = (pd.to_datetime(c.v4_time).dt.tz_localize(None).values
                - pd.to_datetime(m.time).dt.tz_localize(None).values) / np.timedelta64(1, "s")
    assert np.abs(dt_check).max() < 1e-3, "zbz event_id does not match the catalog ID"
    n_all = len(c)
    c = c[c.loc_source == B5["loc_source"]].copy()
    full = pd.read_csv(B5["catalog_full"])
    full = full[full.status == "LOCATED"].set_index("ID").loc[c.index]
    full["depth"] = unflatten(full.depth)
    for col, tol in (("lat", 6e-4), ("lon", 6e-3), ("depth", 6e-3)):
        assert (full[col] - c[col]).abs().max() < tol, f"{col}: full-precision B4 differs from catalog_b4"
        c[col] = full[col]
    assert (full.time == c.time).all()
    print(f"{case}: families {fams} ({fam.n_all.sum()} events), {len(c)} located by B4 in 3D, "
          f"{n_all - len(c)} kept their v4 hypocentre (tier U) and are left out")
    x, y = lambert_km(c.lon.values, c.lat.values)
    ev = pd.DataFrame({"ID": c.index.values, "time": pd.to_datetime(c.time).dt.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                       "lat": c.lat.values, "lon": c.lon.values, "depth": c.depth.values, "x_km": x, "y_km": y,
                       "zf_km": eft_depth(c.depth.values), "h_min": c.h_min.values, "h_max": c.h_max.values,
                       "sig_z": c.sig_z.values, "tier": c.tier.values, "mag": c.v4_magnitude.values})
    pk = pd.read_csv(B5["picks"], skipinitialspace=True)
    pk.columns = [s.strip() for s in pk.columns]
    pk = pk[pk.event_ID.isin(ev.ID)].copy()
    n0 = len(pk)
    gs = grid_stations()
    pk = pk[pk.station.isin(gs.station)]
    n_nogrid = n0 - len(pk)
    e = ev.set_index("ID")
    t0 = pd.to_datetime(e.time.reindex(pk.event_ID).values)
    pk["tt"] = (pd.to_datetime(pk.pick_datetime.str.rstrip("Z"), format="ISO8601").values - t0.values) \
        / np.timedelta64(1, "s")
    G = Grids()
    ok = np.zeros(len(pk), bool)
    for (sta, ph), g in pk.groupby(["station", "phase"]):
        ee = e.loc[g.event_ID]
        ok[pk.index.get_indexer(g.index)] = G.inside(ph, sta, ee.x_km.values, ee.y_km.values, ee.zf_km.values,
                                                     B5["grid_margin_km"])
    n_out = int((~ok).sum())
    pk = pk[ok].copy()
    pk["cls"] = pick_class(pk.pick_prob).values
    pk["sigma"] = pick_sigma(pk.cls.values, pk.phase.values)
    pk["w"] = weight(pk.sigma)
    pk = pk[["event_ID", "station", "phase", "tt", "cls", "sigma", "w"]].sort_values(["event_ID", "phase", "station"])
    st = gs[gs.station.isin(pk.station)].sort_values("station").reset_index(drop=True)
    st["code"] = [f"S{i:04d}" for i in range(len(st))]
    ev = ev[ev.ID.isin(pk.event_ID)].reset_index(drop=True)
    dt, n_cand = diff_times(ev, pk, B5["dt"])
    print(f"picks {n0}: {n_nogrid} at stations without a grid, {n_out} outside the station grid; kept {len(pk)} "
          f"at {len(st)} stations; dt pairs {dt[['id1', 'id2']].drop_duplicates().shape[0]} of {n_cand} candidates")
    summ = write_case(case, ev, pk, st, dt,
                      extra=dict(families=fams, family_events=int(fam.n_all.sum()), b4_not_3d=n_all - len(c),
                                 picks_v4=n0, picks_no_grid=n_nogrid, picks_outside_grid=n_out, dt_candidates=n_cand),
                      inputs=[B5["catalog"], B5["catalog_full"], B5["picks"], B5["members"], B5["families"]], params={"b5_test": B5})
    print(summ)


if __name__ == "__main__":
    main()

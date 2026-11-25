"""Forward-accuracy test, step 1: station/event subset and the reference numbers of the check01 finding.

Subset: 12 Mendocino-Gorda stations (4 land near sea level, 3 land at 1.5-1.9 km, 5 OBS from shelf to 3 km
depth); for each station up to N_PER_BIN events per distance bin (0-25, 25-50, 50-100, 100-150, 150-200 km)
drawn at random (seed 0) from the events with a pick at that station in the B2a inputs; all picks of the
chosen events at the 12 stations are kept.

For every pick: observed travel time (B2a absolute.dat value, OBS S statics applied), v4-NLL3D hypocentre,
the NLL predicted time read from the production time grids (time_vpvs176, trilinear like NLLoc) and the
NLL / tomoDD predictions implied by check01 (raytracer_vs_nll.csv), so the subset reproduces the finding.

Output: OUT/subset/{picks,events,stations}.csv
"""
import numpy as np
import pandas as pd

from b2_fwd_common import BINS, CHECK01, INPUTS, NLL_TIME, OUT, lambert_km, nll_time, bias_table

STATIONS = ["B932.PB", "JCC.BK", "B045.PB", "B049.PB",          # land, < 0.2 km
            "KHBB.NC", "KHMB.NC", "KCS.NC",                      # land, 1.5-1.9 km
            "FS04D.7D", "FS07D.7D", "FS13D.7D", "G10D.7D", "GB281.Z5"]   # OBS 0.16-3.1 km bsl
N_PER_BIN = 40


def main():
    sub = OUT / "subset"
    sub.mkdir(parents=True, exist_ok=True)
    pk = pd.read_csv(INPUTS / "picks_used.csv")
    ev = pd.read_csv(INPUTS / "events.csv")
    st = pd.read_csv(INPUTS / "station_codes.csv")
    st = st[st.station.isin(STATIONS)].copy()
    assert len(st) == len(STATIONS)
    p = pk[pk.station.isin(STATIONS)].copy()
    rng = np.random.default_rng(0)
    chosen = set()
    for s, d in p.groupby("station"):
        d = d.drop_duplicates("event_ID")
        for a, b in zip(BINS[:-2], BINS[1:-1]):
            e = d[(d.epi_dist_km > a) & (d.epi_dist_km <= b)].event_ID.values
            chosen |= set(rng.choice(e, min(N_PER_BIN, len(e)), replace=False).tolist())
    p = p[p.event_ID.isin(chosen)].copy()
    e = ev[ev.ID.isin(chosen)].copy()
    p = p.merge(e[["ID", "lat", "lon", "depth_bsl_km"]].rename(columns={"ID": "event_ID"}), on="event_ID")
    p = p.merge(st[["station", "sta_lat", "sta_lon", "sta_elev_km"]], on="station")
    p["stype"] = np.where(p.is_obs, "obs", "land")
    # NLL production grids
    x, y = lambert_km(p.lon.values, p.lat.values)
    p["x_km"], p["y_km"] = x, y
    p["t_nll_prod"] = np.nan
    for (s, ph), d in p.groupby(["station", "phase"]):
        p.loc[d.index, "t_nll_prod"] = nll_time(NLL_TIME / f"cascadia3d.{ph}.{s}.time", d.x_km.values,
                                                d.y_km.values, d.depth_bsl_km.values)
    # check01 numbers: t_tomodd = tt - res; t_nll = travel_time_s - nll_noterm (NLL used the uncorrected pick)
    c = pd.read_csv(CHECK01 / "raytracer_vs_nll.csv")
    p = p.merge(c[["event_ID", "code", "phase", "dpred", "res_ms", "nll_noterm"]], on=["event_ID", "code", "phase"],
                how="left")
    p["t_tomodd_check01"] = p.tt - p.res_ms / 1000.0
    p["t_nll_check01"] = p.travel_time_s - p.nll_noterm
    p["d_check01"] = p.t_tomodd_check01 - p.t_nll_check01
    p["d_prod_vs_check01_nll"] = p.t_nll_prod - p.t_nll_check01
    p.to_csv(sub / "picks.csv", index=False)
    e.to_csv(sub / "events.csv", index=False)
    st.to_csv(sub / "stations.csv", index=False)
    print(f"{len(st)} stations, {len(e)} events, {len(p)} picks "
          f"(P {int((p.phase == 'P').sum())}, S {int((p.phase == 'S').sum())})")
    print("NLL production grid lookup minus check01 NLL prediction (s):",
          p.d_prod_vs_check01_nll.describe().round(4).to_dict())
    print("dpred(check01 csv) - d_check01:", (p.dpred - p.d_check01).abs().max())
    print(bias_table(p, "d_check01").round(3).to_string())


if __name__ == "__main__":
    main()

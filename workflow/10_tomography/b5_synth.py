"""D-B048 synthetic case: a planar swarm with known geometry through both relocation codes.

Truth: n_events uniformly on a plane length_km (along strike) x width_km (down dip), strike_deg, dip_deg, centred
on the B4 median epicentre and median depth of the case in b5_test.synth.centre.
Picks: event j takes the station-phases and pick classes of a random real event of that case (so the station
geometry and the number of picks per event are the swarm's), kept where the true hypocentre is inside the station's
time grid. True travel times from the B4 grids (the grids GrowClust3D reads; tomoDD traces its own rays in the
v0.3.0 nodes), plus Gaussian noise with the D-B008 sigma of the pick class.
Start: true hypocentre perturbed by the B4 uncertainty of a random real event (68 % horizontal semi-axes h_min,
h_max -> sigma = h / 1.515 along a random azimuth; depth sigma sig_z), depth clipped to start_depth_min_km;
start origin time = true + median residual at the start position (as a locator would put it).
Differential times from the start hypocentres with b5_test.dt. Writes case `synth` (data/ + data/truth.csv).
--source tomodd (case synth_tdd): the same truth, picks and noise, but the true times are tomoDD-SE's own forward
times in the v0.3.0 MOD (initial residuals of a 1-iteration relocation-only run) minus the P-4 correction, so that
tomoDD has no forward error and GrowClust3D (B4 grids) has tomoDD's; the swap shows how much each code gains from
fitting data made with its own forward model.

--start-scale f (case synth_s025: f = 0.25): the start perturbation sigmas times f (well-recorded events), the
truth, picks and noise unchanged; it separates what the codes do from what the B4-sized start errors allow.

  python b5_synth.py [--case synth] [--source grids|tomodd] [--start-scale 1.0]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import shutil
import subprocess

from b5_common import B5, P, Grids, KM_DEG, case_dir, data_dir, diff_times, eft_depth, lambert_km, load_case, \
    pick_sigma, weight, write_case

S = B5["synth"]


def tomodd_times(case, truth, pk, st):
    """tomoDD-SE forward times of the picks at the true hypocentres (relocation-only run, 1 iteration; predicted =
    observed - initial residual of tomoDD-SE.res.ini), in the case's v0.3.0 MOD and P-2 frame."""
    from b2_fwd_mods import ellipsoid_coords
    from b2_tomodd_run import INP
    from b5_tomodd import write_mod
    T, BT = P["tomodd"], B5["tomodd"]
    wd = case_dir(case) / "tomodd_forward"
    if wd.exists():
        shutil.rmtree(wd)
    (wd / "Input_Files").mkdir(parents=True)
    (wd / "Output_Files").mkdir()
    center = (round(float(truth.lat.median()), 3), round(float(truth.lon.median()), 3))
    code = {s_: f"S{i:04d}" for i, s_ in enumerate(st.station)}
    la, lo = ellipsoid_coords(st.lat.values, st.lon.values, *center)
    (wd / "Input_Files/station.dat").write_text("".join(f"{code[s_]} {a:.6f} {b:.6f} {-z * 1000.0:.1f}\n" for s_, a, b, z
                                                       in zip(st.station, la, lo, st.zsrc_km)))
    la, lo = ellipsoid_coords(truth.lat.values, truth.lon.values, *center)
    (wd / "Input_Files/event.dat").write_text("".join(f"20150601  00000000 {a:10.6f} {b:11.6f} {z:8.3f} 1.0 0.0 0.0 0.0 "
                                                     f"{i:10d} 0\n" for a, b, z, i in
                                                     zip(la, lo, truth.depth, truth.ID)))
    with open(wd / "Input_Files/absolute.dat", "w") as f:
        for eid, g in pk.groupby("event_ID"):
            f.write(f"#  {eid}\n" + "".join(f"{code[s_]:<7s} {t:9.4f} {1.0:6.3f} {ph}\n" for s_, t, ph in
                                             zip(g.station, g.t_true, g.phase)))
    a_, b_ = pk.event_ID.unique()[:2]
    A = pk[pk.event_ID == a_].set_index(["station", "phase"]).t_true
    B = pk[pk.event_ID == b_].set_index(["station", "phase"]).t_true
    com = A.index.intersection(B.index)
    (wd / "Input_Files/dt.ct").write_text(f"#  {a_}  {b_}\n" + "".join(
        f"{code[k[0]]:<7s} {A[k]:9.4f} {B[k]:9.4f} 1.000 {k[1]}\n" for k in com))
    write_mod(wd / "MOD", truth, st, center)
    for f in ("ak135.15.SKS", "layer-16.dat"):
        shutil.copy(Path(T["bin"]).parent / f, wd / f)
    row = f"  1    -9    -9    -9   -9   1.00  1.00   -9   -9      1 {BT['damp']:6g}   0  {T['inp']['thres']:g}"
    (wd / "tomoDD-SE.inp").write_text(INP.format(region=case, tag="forward", air=-99.0, nset=1, stepl=1,
                                                 clat=center[0], clon=center[1], w1=20, w2=20, w3=10, sets=row))
    with open(wd / "stdout.txt", "w") as so:
        subprocess.run([BT["bin"], "tomoDD-SE.inp"], cwd=wd, stdout=so, stderr=subprocess.STDOUT)
    r = pd.read_csv(wd / "tomoDD-SE.res.ini", sep=r"\s+", skiprows=1, header=None, dtype=str,
                    names=["STA", "DT", "C1", "C2", "IDX", "QUAL", "RES", "WT", "OFFS"])
    for c in ("DT", "C1", "C2", "IDX", "RES"):
        r[c] = pd.to_numeric(r[c], errors="coerce")
    r = r[r.C1 == r.C2]
    inv = {v: k for k, v in code.items()}
    r = pd.DataFrame({"event_ID": r.C1.astype(int), "station": r.STA.map(inv), "phase": np.where(r.IDX == 3, "P", "S"),
                      "t_tomodd": r.DT - r.RES / 1000.0})
    out = pk.merge(r, on=["event_ID", "station", "phase"], how="left")
    print(f"tomoDD forward: {out.t_tomodd.notna().sum()} of {len(out)} picks; tomoDD - grids median "
          f"{(out.t_tomodd - out.t_true).median():+.3f} s")
    return out.t_tomodd.values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="synth")
    ap.add_argument("--source", default="grids", choices=["grids", "tomodd"])
    ap.add_argument("--start-scale", type=float, default=1.0)
    a = ap.parse_args()
    rng = np.random.default_rng(S["seed"])
    ev0, pk0, st0, _ = load_case(S["centre"])
    lat0, lon0, z0 = float(ev0.lat.median()), float(ev0.lon.median()), float(ev0.depth.median())
    n = int(S["n_events"])
    s = rng.uniform(-S["length_km"] / 2, S["length_km"] / 2, n)       # along strike
    d = rng.uniform(-S["width_km"] / 2, S["width_km"] / 2, n)         # down dip
    st_, dp = np.radians(S["strike_deg"]), np.radians(S["dip_deg"])
    dipdir = st_ + np.pi / 2
    east = s * np.sin(st_) + d * np.cos(dp) * np.sin(dipdir)
    north = s * np.cos(st_) + d * np.cos(dp) * np.cos(dipdir)
    depth = z0 + d * np.sin(dp)
    lat = lat0 + north / KM_DEG
    lon = lon0 + east / (KM_DEG * np.cos(np.radians(lat0)))
    ids = 900000 + np.arange(n)
    x, y = lambert_km(lon, lat)
    zf = eft_depth(depth)
    truth = pd.DataFrame({"ID": ids, "lat": lat, "lon": lon, "depth": depth, "along_km": s, "down_km": d,
                          "x_km": x, "y_km": y, "zf_km": zf})
    G = Grids()
    donor = rng.choice(ev0.ID.values, n)
    t_origin = pd.Timestamp("2015-06-01T00:00:00") + pd.to_timedelta(np.arange(n) * 3600.0, unit="s")
    rows = []
    for j in range(n):
        p = pk0[pk0.event_ID == donor[j]]
        for r in p.itertuples():
            if not G.inside(r.phase, r.station, x[j], y[j], zf[j], B5["grid_margin_km"]):
                continue
            t = float(G.time(r.phase, r.station, x[j], y[j], zf[j])[0])
            rows.append((ids[j], r.station, r.phase, t, r.cls))
    pk = pd.DataFrame(rows, columns=["event_ID", "station", "phase", "t_true", "cls"])
    if a.source == "tomodd":                                    # tomoDD forward minus P-4 (see docstring)
        from b5_tomodd import p4_corr
        pk["t_grid"] = pk.t_true
        st_all = st0[st0.station.isin(pk.station)].reset_index(drop=True)
        tdd = tomodd_times(a.case, truth, pk, st_all)
        c = pd.MultiIndex.from_frame(pk[["station", "phase"]]).map(p4_corr()).fillna(0.0).values
        pk["t_true"] = tdd - c
        pk = pk[np.isfinite(pk.t_true) & (np.abs(pk.t_true - pk.t_grid) < 5.0)].copy()
    pk["sigma"] = pick_sigma(pk.cls.values, pk.phase.values)
    pk["w"] = weight(pk.sigma)
    pk["noise"] = rng.normal(0.0, pk.sigma.values)
    pk["t_obs"] = pk.t_true + pk.noise                          # relative to the true origin time
    # start hypocentres: B4 uncertainty of a random real event
    e0 = ev0.set_index("ID").loc[rng.choice(ev0.ID.values, n)]
    az = rng.uniform(0, 2 * np.pi, n)
    a1 = rng.normal(0, a.start_scale * e0.h_max.values / 1.515)
    a2 = rng.normal(0, a.start_scale * e0.h_min.values / 1.515)
    de = a1 * np.sin(az) + a2 * np.cos(az)
    dn = a1 * np.cos(az) - a2 * np.sin(az)
    dz = rng.normal(0, a.start_scale * e0.sig_z.values)
    s_lat = lat + dn / KM_DEG
    s_lon = lon + de / (KM_DEG * np.cos(np.radians(lat0)))
    s_dep = np.clip(depth + dz, S["start_depth_min_km"], 60.0)
    sx, sy = lambert_km(s_lon, s_lat)
    szf = eft_depth(s_dep)
    shift = np.zeros(n)
    for j in range(n):
        m = pk.event_ID.values == ids[j]
        if not m.any():
            continue
        pred = np.array([G.time(ph, sta, sx[j], sy[j], szf[j])[0] for sta, ph in
                         zip(pk.station.values[m], pk.phase.values[m])])
        shift[j] = np.median(pk.t_obs.values[m] - pred)
    pk["tt"] = pk.t_obs - pk.event_ID.map(dict(zip(ids, shift)))
    ev = pd.DataFrame({"ID": ids, "time": (t_origin + pd.to_timedelta(shift, unit="s")).strftime("%Y-%m-%dT%H:%M:%S.%f"),
                       "lat": s_lat, "lon": s_lon, "depth": s_dep, "x_km": sx, "y_km": sy, "zf_km": szf,
                       "h_min": e0.h_min.values, "h_max": e0.h_max.values, "sig_z": e0.sig_z.values, "tier": "S",
                       "mag": 1.0})
    ev = ev[ev.ID.isin(pk.event_ID)].reset_index(drop=True)
    st = st0[st0.station.isin(pk.station)].reset_index(drop=True)
    dt, n_cand = diff_times(ev, pk, B5["dt"])
    data_dir(a.case).mkdir(parents=True, exist_ok=True)
    truth.to_csv(data_dir(a.case) / "truth.csv", index=False, float_format="%.6f")
    pk[["event_ID", "station", "phase", "t_true", "noise"]].to_csv(data_dir(a.case) / "picks_truth.csv", index=False,
                                                                 float_format="%.5f")
    pert = pd.DataFrame({"ID": ids, "dh": np.hypot(de, dn), "dz": s_dep - depth})
    summ = write_case(a.case, ev, pk[["event_ID", "station", "phase", "tt", "cls", "sigma", "w"]], st, dt,
                      extra=dict(synthetic=True, source={"grids": "b4_grids", "tomodd": "tomodd_forward"}[a.source], centre=[lat0, lon0, z0], donor_case=S["centre"], start_scale=a.start_scale,
                                 start_dh_med_km=float(pert.dh.median()), start_dz_absmed_km=float(pert.dz.abs().median()),
                                 start_depth_clipped=int((depth + dz < S["start_depth_min_km"]).sum()),
                                 dt_candidates=n_cand, synth=S),
                      inputs=[data_dir(S["centre"]) / "events.csv", data_dir(S["centre"]) / "picks.csv"],
                      params={"synth": S, "dt": B5["dt"]})
    print(summ)


if __name__ == "__main__":
    main()

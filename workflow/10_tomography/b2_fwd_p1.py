"""D-B028 P-1 reference travel times for one tomoDD region (Mendocino-Gorda).

For every station of the region (inputs/station_codes.csv) and phase P, S: Grid2Time (Podvin-Lecomte) in the
v4-NLL3D model taken as a continuous field (trilinear between the NLL nodes, 'origlin'), velocity at the cell
centres, Earth-flattening transform, cells of 1 km and 0.5 km; times read at the points (trilinear, as NLLoc)
in each grid and combined: t_p1 = T0.5 - k (T1 - T0.5), k from params `tomodd.p1.richardson_k` (calibrated in
notes/forward_accuracy.md against TauP). Station source depth: production GTSRCE (a1_build_model.py).

Points per station:
* the region's picks (inputs/picks_used.csv) at the B1b (v4-NLL3D iter1) hypocentres (inputs/events.csv);
* for OBS stations, S phases of the v4-NLL3D iter1 run (weight > 0) within p1.statics_max_dist_km at the iter1
  locations (all catalog events, needed for the D-B020 statics, P-6).
Grid per station: box around its points + 4 km, z -5 km to max(70, deepest point + 10) km (flattened depth).

Output (<runs>/<out-region>/p1_reference/):
  region_picks_p1.parquet   event_ID, station, phase, t1, t05, t_p1 (s)
  obs_S_iter1_p1.parquet    ID, sta, dist, obs_tt (pick - iter1 origin), t1, t05, t_p1
  grids/<sta>/p1.<ph>.<sta>.time.{hdr,buf}  combined 1 km grid (kept only with --keep-grids; flattened depth)
The 0.5 km and 1 km Grid2Time grids are deleted right after sampling.

  python b2_fwd_p1.py REGION [--out-region R2] [--nproc 16] [--keep-grids]
  (Mendocino-Gorda: b2_fwd_p1.py mendocino_gorda --out-region mendocino_gorda_v2 --keep-grids; same picks)
"""
import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from b2_fwd_common import NLL_MODEL, eft_depth, grid2time, lambert_km, nll_time, read_nll_hdr
from b2_fwd_nll import model_fn
from manifest import load_params, write_manifest

P = load_params()
T = P["tomodd"]
PAD = 4.0


def job(a):
    sta, lat, lon, zsrc, phase, ext, z1, pts, wd, keep = a
    fn = model_fn("origlin_c05_eft")(phase)
    out = {}
    roots = {}
    for dx, tag in ((1.0, "t1"), (0.5, "t05")):
        roots[tag] = grid2time(wd / tag, sta, lat, lon, zsrc, phase, fn, ext, dx, (-5.0, z1), eft=True, centred=True)
        out[tag] = nll_time(roots[tag], pts[:, 0], pts[:, 1], pts[:, 2])
    k = T["p1"]["richardson_k"][phase]
    out["t_p1"] = out["t05"] - k * (out["t1"] - out["t05"])
    if keep:
        g1, g05 = (read_nll_hdr(roots[t]) for t in ("t1", "t05"))
        a1 = np.fromfile(f"{roots['t1']}.buf", "<f4").reshape(g1["nx"], g1["ny"], g1["nz"])
        a05 = np.fromfile(f"{roots['t05']}.buf", "<f4").reshape(g05["nx"], g05["ny"], g05["nz"])[::2, ::2, ::2]
        assert a05.shape == a1.shape
        comb = (a05 - k * (a1 - a05)).astype("<f4")
        gd = wd.parent.parent / "grids" / sta
        gd.mkdir(parents=True, exist_ok=True)
        r = gd / f"p1.{phase}.{sta}.time"
        comb.tofile(f"{r}.buf")
        hdr = Path(f"{roots['t1']}.hdr").read_text().splitlines()
        Path(f"{r}.hdr").write_text("\n".join(hdr) + "\n")
    shutil.rmtree(wd)
    return sta, phase, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("region")
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--keep-grids", action="store_true")
    ap.add_argument("--out-region", help="write to <runs>/<out-region>/p1_reference (default: REGION)")
    a = ap.parse_args()
    inp = Path(T["runs"]) / a.region / "inputs"
    out = Path(T["runs"]) / (a.out_region or a.region) / "p1_reference"
    (out / "work").mkdir(parents=True, exist_ok=True)
    st = pd.read_csv(inp / "station_codes.csv")
    pk = pd.read_csv(inp / "picks_used.csv", usecols=["event_ID", "station", "phase"])
    ev = pd.read_csv(inp / "events.csv", usecols=["ID", "lat", "lon", "depth_bsl_km"])
    pk = pk.merge(ev.rename(columns={"ID": "event_ID"}), on="event_ID")
    pk["x"], pk["y"] = lambert_km(pk.lon.values, pk.lat.values)
    pk["zf"] = eft_depth(pk.depth_bsl_km.values)
    # iter1 OBS S phases near the region's OBS (statics, P-6)
    run = Path(P["paths"]["reloc_v5"]) / "runs" / P["paths"]["trackA_run"]
    ph = pd.read_parquet(Path(P["paths"]["base"]) / "b1_dataset" / "cache" / f"phases_{P['paths']['trackA_run']}.parquet",
                         columns=["ID", "sta", "phase", "pick_time", "w", "dist", "res", "tcorr", "ttpred"])
    ph = ph[(ph.phase.astype(str) == "S") & (ph.w > 0) & (ph.dist <= T["p1"]["statics_max_dist_km"])]
    ph["sta"] = ph.sta.astype(str)
    ph = ph[ph.sta.isin(st.station[st.is_obs])]
    loc = pd.read_csv(run / "locations.csv", usecols=["ID", "time", "lat", "lon", "depth"])
    ph = ph.merge(loc, on="ID")
    ph["obs_tt"] = (pd.to_datetime(ph.pick_time) - pd.to_datetime(ph.time)).dt.total_seconds()
    ph["x"], ph["y"] = lambert_km(ph.lon.values, ph.lat.values)
    ph["zf"] = eft_depth(ph.depth.values)
    gts = {l.split()[1]: float(l.split()[5]) for l in (NLL_MODEL / "stations_GTSRCE.in").read_text().splitlines()}

    jobs, keys = [], []
    for r in st.itertuples():
        sx, sy = lambert_km(r.sta_lon, r.sta_lat)
        d1 = pk[pk.station == r.station]
        d2 = ph[ph.sta == r.station]
        xs = np.r_[d1.x, d2.x, sx]
        ys = np.r_[d1.y, d2.y, sy]
        zs = np.r_[d1.zf, d2.zf, 0.0]
        ext = (np.floor(xs.min() - PAD), np.ceil(xs.max() + PAD), np.floor(ys.min() - PAD), np.ceil(ys.max() + PAD))
        z1 = float(max(70.0, np.ceil(zs.max() + 10)))
        for phase in ("P", "S"):
            a1 = d1[d1.phase == phase]
            a2 = d2 if phase == "S" else d2.iloc[:0]
            if len(a1) + len(a2) == 0:
                continue
            pts = np.c_[np.r_[a1.x, a2.x], np.r_[a1.y, a2.y], np.r_[a1.zf, a2.zf]]
            keys.append((r.station, phase, a1.index.values, a2.index.values))
            jobs.append((r.station, r.sta_lat, r.sta_lon, gts[r.station], phase, ext, z1, pts,
                         out / "work" / f"{r.station}_{phase}", a.keep_grids))
    print(f"{len(jobs)} station-phase jobs; {len(pk)} region picks; {len(ph)} iter1 OBS S phases <= "
          f"{T['p1']['statics_max_dist_km']} km", flush=True)
    for c in ("t1", "t05", "t_p1"):
        pk[c] = np.nan
        ph[c] = np.nan
    with ProcessPoolExecutor(a.nproc) as ex:
        for n, (job_res, key) in enumerate(zip(ex.map(job, jobs), keys)):
            sta, phase, o = job_res
            i1, i2 = key[2], key[3]
            for c in ("t1", "t05", "t_p1"):
                pk.loc[i1, c] = o[c][:len(i1)]
                ph.loc[i2, c] = o[c][len(i1):]
            if n % 20 == 0:
                print(f"  {n + 1}/{len(jobs)} {sta} {phase}", flush=True)
    shutil.rmtree(out / "work", ignore_errors=True)
    pk[["event_ID", "station", "phase", "t1", "t05", "t_p1"]].to_parquet(out / "region_picks_p1.parquet")
    ph[["ID", "sta", "dist", "obs_tt", "res", "tcorr", "ttpred", "t1", "t05", "t_p1"]].to_parquet(out / "obs_S_iter1_p1.parquet")
    counts = dict(jobs=len(jobs), region_picks=len(pk), region_nan=int(pk.t_p1.isna().sum()),
                  obs_S=len(ph), obs_S_nan=int(ph.t_p1.isna().sum()))
    if a.keep_grids:
        counts["grids_GB"] = round(sum(f.stat().st_size for f in (out / "grids").rglob("*.buf")) / 1e9, 2)
    print(counts)
    write_manifest(f"b2_fwd_p1_{a.region}", out, inputs=[inp / "picks_used.csv", inp / "events.csv",
                   inp / "station_codes.csv", NLL_MODEL / "cascadia3d.P.mod.buf", NLL_MODEL / "cascadia3d.S.mod.buf"],
                   outputs=[out / "region_picks_p1.parquet", out / "obs_S_iter1_p1.parquet"],
                   params={"p1": T["p1"], "region": a.region, "keep_grids": a.keep_grids}, counts=counts)


if __name__ == "__main__":
    main()

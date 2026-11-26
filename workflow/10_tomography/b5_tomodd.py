"""D-B048: relocate one case with tomoDD-SE in relocation-only mode (JOINT = 0, velocity frozen) in model v0.3.0.

Inputs from <out>/<case>/data (b5_prep.py or b5_synth.py), written to <out>/<case>/tomodd/inputs:
* MOD: model v0.3.0 (netCDF, D-B047) at its own nodes over the events and stations + node_margin_deg, depth nodes
  of the model (-3 ... 125 km) + edge nodes (tomodd.edge_z, tomodd.edge_deg; 400 km Vp from tomodd.deep_vp);
  latitude and longitude rescaled about the case centre (D-B028 P-2, b2_fwd_mods.ellipsoid_coords).
* dt.ct: the case's differential times (dt.csv), weight = pair weight; station codes S0000...
* absolute.dat: the picks, tt - B4 station term + P-4 tomoDD forward correction (D-B028 P-4, reused from
  b5_test.tomodd_p4) so that tomoDD's absolute residuals are referred to the B4 frame; synthetic cases (no station
  terms in the data): tt + P-4 correction.
* event.dat: start hypocentres (true depth), station.dat: elevation = -zsrc.
Run: <out>/<case>/tomodd/<variant>[_bNNN]/; tomoDD-SE.inp as b2_tomodd_run.py with JOINT = 0 and the variant's sets.
Bootstrap member k (--boot k, k >= 1): dt rows resampled with replacement (numpy seed 48000 + k), absolute data kept.
Output: reloc.csv (ID, lat, lon, depth, ex, ey, ez km (LSQR), nct_p, nct_s, rct, cid, relocated) with geographic
coordinates (P-2 map inverted), and a manifest.

  python b5_tomodd.py CASE VARIANT [--boot K] [--damp D]
"""
import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from b5_common import B5, P, case_dir, frame_centre, load_case, write_manifest
from b2_fwd_mods import ellipsoid_coords
from b2_tomodd_backtransform import inverse
from b2_tomodd_run import INP

T = P["tomodd"]
BT = B5["tomodd"]


def write_mod(path, ev, st, center):
    ds = xr.open_dataset(Path(T["models"]["store"]) / T["models"]["name"] / f"v{B5['model_version']}.nc")
    m = BT["node_margin_deg"]
    lo = np.r_[ev.lon.values, st.lon.values]
    la = np.r_[ev.lat.values, st.lat.values]
    s = ds.sel(longitude=slice(lo.min() - m, lo.max() + m), latitude=slice(la.min() - m, la.max() + m))
    xi, yi, zi = s.longitude.values, s.latitude.values, s.depth.values
    vp = s.Vp.values.transpose(2, 1, 0).astype(float)        # (lon, lat, depth)
    r = (s.Vp / s.Vs).values.transpose(2, 1, 0).astype(float)
    E = T["edge_deg"]
    xn, yn = np.r_[xi[0] - E, xi, xi[-1] + E], np.r_[yi[0] - E, yi, yi[-1] + E]
    zt, zb = T["edge_z"]
    zn = np.r_[zt, zi, zb]
    vp = np.pad(vp, ((1, 1), (1, 1), (0, 0)), mode="edge")
    r = np.pad(r, ((1, 1), (1, 1), (0, 0)), mode="edge")
    vp = np.concatenate([vp[:, :, :1], vp, np.full(vp.shape[:2] + (1,), T["deep_vp"][int(zb)])], axis=2)
    r = np.concatenate([r[:, :, :1], r, r[:, :, -1:]], axis=2)
    yw, xw = ellipsoid_coords(yn, xn, *center)
    lines = [f"0.01 {len(xw)} {len(yw)} {len(zn)}", " ".join(f"{v:.4f}" for v in xw),
             " ".join(f"{v:.4f}" for v in yw), " ".join(f"{v:.3f}" for v in zn)]
    for arr, fmt in ((vp, "{:.4f}"), (r, "{:.5f}")):
        for k in range(len(zn)):
            for j in range(len(yn)):
                lines.append(" ".join(fmt.format(v) for v in arr[:, j, k]))
    Path(path).write_text("\n".join(lines) + "\n")
    return dict(nx=len(xn), ny=len(yn), nz=len(zn), lon=[float(xi[0]), float(xi[-1])], lat=[float(yi[0]), float(yi[-1])])


def station_terms():
    rows = [l.split() for l in Path(B5["station_terms"]).read_text().splitlines() if l.startswith("LOCDELAY")]
    return pd.DataFrame({"station": [r[1] for r in rows], "phase": [r[2] for r in rows],
                         "term": [float(r[4]) for r in rows]}).set_index(["station", "phase"]).term


def p4_corr():
    c = pd.read_csv(Path(T["runs"]) / B5["tomodd_p4"] / "inputs" / "station_corr.csv")
    return c.set_index(["station", "phase"]).corr_s


def build_inputs(case):
    ev, pk, st, dt = load_case(case)
    summ = json.loads((case_dir(case) / "data" / "summary.json").read_text())
    synthetic = bool(summ.get("synthetic", False))
    out = case_dir(case) / "tomodd" / "inputs"
    if (out / "done.json").exists():
        return out, json.loads((out / "done.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    center = frame_centre(ev.lat, ev.lon)                 # no event at tomoDD's Cartesian origin (NaN rays)
    code = dict(zip(st.station, st.code))
    la, lo = ellipsoid_coords(st.lat.values, st.lon.values, *center)
    with open(out / "station.dat", "w") as f:
        for c_, a, b, z in zip(st.code, la, lo, st.zsrc_km):
            f.write(f"{c_} {a:.6f} {b:.6f} {-z * 1000.0:.1f}\n")
    la, lo = ellipsoid_coords(ev.lat.values, ev.lon.values, *center)
    with open(out / "event.dat", "w") as f:
        for r, a, b in zip(ev.itertuples(), la, lo):
            t = pd.Timestamp(r.time)
            cs = int(round((t.second + t.microsecond / 1e6) * 100))
            f.write(f"{t.strftime('%Y%m%d')}  {t.hour:02d}{t.minute:02d}{cs:04d} {a:10.6f} {b:11.6f} "
                    f"{r.depth:8.3f} 1.0 0.0 0.0 0.0 {r.ID:10d} 0\n")
    # tomoDD takes the travel times as given (event.dat origin time is only carried to the output), so the
    # dt rows are written exactly as in dt.csv: the same numbers GrowClust3D reads
    k = pd.MultiIndex.from_frame(pk[["station", "phase"]])
    pk = pk.copy()
    pk["term"] = 0.0 if synthetic else k.map(station_terms()).fillna(0.0).values
    pk["p4"] = k.map(p4_corr()).fillna(0.0).values
    pk["tt_abs"] = pk.tt - pk.term + pk.p4
    with open(out / "absolute.dat", "w") as f:
        for eid, d in pk.groupby("event_ID"):
            f.write(f"#  {eid}\n")
            for r in d.itertuples():
                f.write(f"{code[r.station]:<7s} {r.tt_abs:9.4f} {r.w:6.3f} {r.phase}\n")
    write_dt(out / "dt.ct", dt, code)
    mod = write_mod(out / "MOD", ev, st, center)
    air = round(float(st.zsrc_km.min()) - BT["air_dep_margin_km"], 3)     # -(highest elevation + margin)
    info = dict(center=center, air_dep_km=air, synthetic=synthetic, mod=mod, n_events=len(ev), n_sta=len(st),
                picks_missing_term=int((pk.term == 0).sum()) if not synthetic else None,
                picks_missing_p4=int((pk.p4 == 0).sum()))
    pk[["event_ID", "station", "phase", "tt", "term", "p4", "tt_abs", "w"]].to_csv(out / "absolute_used.csv",
                                                                                   index=False, float_format="%.5f")
    (out / "done.json").write_text(json.dumps(info, indent=1))
    return out, info


def write_dt(path, dt, code):
    lines = []
    for (a, b), d in dt.groupby(["id1", "id2"], sort=True):
        lines.append(f"#  {a}  {b}")
        lines += [f"{code[s]:<7s} {t1:9.4f} {t2:9.4f} {w:6.3f} {ph}" for s, t1, t2, w, ph in
                  zip(d.station, d.tt1, d.tt2, d.w, d.phase)]
    Path(path).write_text("\n".join(lines) + "\n")


def read_reloc(path, center):
    cols = ["ID", "lat", "lon", "depth", "x", "y", "z", "ex", "ey", "ez", "yr", "mo", "dy", "hr", "mi", "sc", "mag",
            "nccp", "nccs", "nctp", "ncts", "rcc", "rct", "cid"]
    r = pd.read_csv(path, sep=r"\s+", header=None, names=cols)
    fw = {"coords": "ellipsoid", "center": list(center)}
    from b2_fwd_mods import radii, RE
    M, N = radii(center[0])
    fw.update(lat_scale_M_over_R=M / RE, lon_scale_N_over_R=N / RE)
    r["lat"], r["lon"] = inverse(r.lat.values, r.lon.values, fw)
    for c in ("ex", "ey", "ez"):
        r[c] = r[c] / 1000.0                                  # m -> km
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("variant")
    ap.add_argument("--boot", type=int, default=0)
    ap.add_argument("--damp", type=float, default=BT["damp"])
    a = ap.parse_args()
    inp, info = build_inputs(a.case)
    tag = a.variant + (f"_b{a.boot:03d}" if a.boot else "") + ("" if a.damp == BT["damp"] else f"_d{a.damp:g}")
    run = case_dir(a.case) / "tomodd" / tag
    if (run / "reloc.csv").exists():
        print(f"{run}: done")
        return
    if run.exists():
        shutil.rmtree(run)
    (run / "Output_Files").mkdir(parents=True)
    (run / "Input_Files").mkdir()
    for f in ("event.dat", "station.dat", "absolute.dat"):
        (run / "Input_Files" / f).symlink_to(inp / f)
    if a.boot:
        st = load_case(a.case)[2]
        dt = load_case(a.case)[3]
        rng = np.random.default_rng(48000 + a.boot)
        dt = dt.iloc[np.sort(rng.integers(0, len(dt), len(dt)))]
        write_dt(run / "Input_Files" / "dt.ct", dt, dict(zip(st.station, st.code)))
    else:
        (run / "Input_Files" / "dt.ct").symlink_to(inp / "dt.ct")
    shutil.copy(inp / "MOD", run / "MOD")
    bindir = Path(T["bin"]).parent
    for f in ("ak135.15.SKS", "layer-16.dat"):
        shutil.copy(bindir / f, run / f)
    sets = BT["variants"][a.variant]
    rows = [f"  {s[0]:<3d}  -9    -9    -9   -9  {s[1]:5.2f} {s[2]:5.2f} {s[3]:4g} {s[4]:4g} {s[5]:6g} {a.damp:6g}"
            f"   0  {T['inp']['thres']:g}" for s in sets]
    w1, w2, w3 = (f"{v:g}" for v in T["inp"]["smooth"])
    (run / "tomoDD-SE.inp").write_text(INP.format(
        region=a.case, tag=tag, air=info["air_dep_km"], nset=len(rows), stepl=T["inp"]["stepl_km"],
        clat=info["center"][0], clon=info["center"][1], w1=w1, w2=w2, w3=w3, sets="\n".join(rows)))
    t0 = time.time()
    with open(run / "stdout.txt", "w") as so:
        rc = subprocess.run([BT["bin"], "tomoDD-SE.inp"], cwd=run, stdout=so, stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0
    rel = run / "Output_Files" / "tomoDD.reloc"
    log = (run / "tomoDD-SE.log").read_text(errors="replace") if (run / "tomoDD-SE.log").exists() else ""
    cnd = [float(l.split("CND)=")[1].split(";")[0]) for l in log.splitlines() if "CND)=" in l]
    n_aq = sum("negative depth" in l for l in log.splitlines())
    if rc != 0 or not rel.exists() or rel.stat().st_size == 0:
        raise SystemExit(f"{run}: tomoDD failed (exit {rc}); see stdout.txt")
    r = read_reloc(rel, info["center"])
    ev, _, _, _ = load_case(a.case)
    r = ev[["ID"]].merge(r, on="ID", how="left")
    r["relocated"] = r.lat.notna()
    r[["ID", "lat", "lon", "depth", "ex", "ey", "ez", "nctp", "ncts", "rct", "cid", "relocated"]].to_csv(
        run / "reloc.csv", index=False, float_format="%.6f")
    counts = dict(exit=rc, wall_s=round(wall, 1), relocated=int(r.relocated.sum()), events=len(r),
                  cnd_first=cnd[0] if cnd else None, cnd_last=cnd[-1] if cnd else None, airquake_lines=n_aq)
    (run / "run.json").write_text(json.dumps(counts, indent=1))
    write_manifest(f"b5_tomodd_{a.case}_{tag}", run, inputs=[inp / "dt.ct", inp / "absolute.dat", inp / "MOD",
                                                               inp / "event.dat", BT["bin"]],
                   outputs=[run / "reloc.csv", rel, run / "tomoDD-SE.inp", run / "tomoDD-SE.log"],
                   params={"variant": a.variant, "sets": sets, "damp": a.damp, "boot": a.boot, "joint": 0,
                           "info": info}, counts=counts)
    print(f"{a.case}/{tag}: {counts}")


if __name__ == "__main__":
    main()

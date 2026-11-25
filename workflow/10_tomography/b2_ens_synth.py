"""Track B Bayesian ensemble (D-B030 gate): synthetic truth and synthetic data for the calibration
(notes/bayes_design.md section 5, with P-M5: one truth for the whole margin, shared by M0 and every tile).

  python b2_ens_synth.py truth        -> ROOT/synth/truth/ and ROOT/synth/{abs,dt}_synth_<stage>.parquet

Truth (seed prior.truth_seed, the member draw functions of b2_ens_member.py):
* model on the merge grid: the tomography nodes of configs.<c>.truth_model (v0.3.1: Vp, Vs as inverted, no surface,
  no D-B073 rule, like the inversion's own model space) times (1 + f) with one draw of the 3 % Vp and Vs random fields
  (30 km / 8 km correlation, tapered below 50-60 km), then the members' start floor Vs = Vp / 1.65 where Vp/Vs < 1.65
  (P-E5; review 2026-11-25: without it 10.6 % of the resolved S nodes had truth Vp/Vs < 1.6, which every merged member
  caps at 1.6 by D-B051, so the gate would miss them by construction); per stage the truth MOD = this model sampled at the stage's
  nodes with tomoDD's trilinear lookup (clamped at the grid edges), boundary nodes padded as the builder does. M0's
  30 km nodes see a node-sampled truth: its synthetic data are computed in that coarser truth (a simplification; M0
  only provides the tiles' starting model).
* hypocentres: v4-NLL3D + one draw of the starting-hypocentre prior (truncated at the merged solid surface), by event
  ID for every stage; members start from their own draws around v4-NLL3D, not around the truth.
* data: tomoDD-SE_syn (frozen syn binary of the stage's class) on the real event/station geometry, the real pick list
  and pair list, in the truth MOD at the truth hypocentres; + one draw of statics, forward offsets and pick noise
  (lambda_c at its point estimate). No P-4 correction: the synthetic forward is tomoDD itself.
"""
import hashlib
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b2_ens_common as C  # noqa: E402
import b2_ens_member as EM  # noqa: E402
from b2_ens_common import CFG, PR, PRIOR, STAGES, T  # noqa: E402
from b2_fwd_common import mod_velocity, read_mod, write_mod  # noqa: E402
from b2_fwd_mods import ellipsoid_coords  # noqa: E402
from b2_smooth_spike_test import SYN_INP  # noqa: E402

SD = C.ROOT / "synth"
TD = SD / "truth"
FLOOR = float(T.get("start_vpvs_min", 1.65))        # D-B046 start floor, applied to every member start (P-E5)


def truth_model():
    nd = xr.open_dataset(Path(T["models"]["store"]) / T["models"]["name"] / f"v{CFG['truth_model']}.nc")
    vpn = "Vp_tomography" if "Vp_tomography" in nd else "Vp"
    vsn = "Vs_tomography" if "Vs_tomography" in nd else "Vs"
    lon, lat, z = nd.longitude.values, nd.latitude.values, nd.depth.values
    vp, vs = (nd[v].values.transpose(2, 1, 0).astype(float) for v in (vpn, vsn))
    seed = int(PR["truth_seed"])
    fp, fs = C.random_field(C.rng(seed, "rf_p")), C.random_field(C.rng(seed, "rf_s"))
    vp = vp * (1 + EM.field_at(fp, lon, lat, z))
    vs = vs * (1 + EM.field_at(fs, lon, lat, z))
    vs = np.where(vp / vs < FLOOR, vp / FLOOR, vs)   # the members' start floor (P-E5): the truth is a draw of their prior
    return dict(lon=lon, lat=lat, z=z, vp=vp, vs=vs), fp, fs


def stage_truth_mod(stage, tm):
    m = read_mod(C.det_inputs(stage) / "MOD")
    R = EM.region_cfg(stage)
    import b2_tomodd_inputs as TI
    lon, lat, zi = TI.nodes(R["node_lon"]), TI.nodes(R["node_lat"]), np.array(R.get("node_z", T["node_z"]), float)
    LO, LA, Z = np.meshgrid(lon, lat, zi, indexing="ij")
    vp = mod_velocity(tm, "P", LO.ravel(), LA.ravel(), Z.ravel(), clamp_surface=False).reshape(LO.shape)
    vs = mod_velocity(tm, "S", LO.ravel(), LA.ravel(), Z.ravel(), clamp_surface=False).reshape(LO.shape)
    vs = np.where(vp / vs < FLOOR, vp / FLOOR, vs)   # floor again at the stage nodes (trilinear Vp and Vs separately)
    VP, R_ = m["vp"].copy(), (m["vp"] / m["vs"]).copy()
    VP[1:-1, 1:-1, 1:-1], R_[1:-1, 1:-1, 1:-1] = vp, vp / vs
    for a in (VP, R_):
        a[0], a[-1] = a[1], a[-2]
        a[:, 0], a[:, -1] = a[:, 1], a[:, -2]
        a[:, :, 0] = a[:, :, 1]
    R_[:, :, -1] = R_[:, :, -2]
    return {**m, "vp": VP, "vs": VP / R_}


def syn_stage(stage, tm, hyp):
    """tomoDD-SE_syn for one stage. Reuses an earlier run only if its MOD and event.dat are byte-identical to the ones
    this truth gives (md5 of both stored in DONE); anything else is rerun from scratch."""
    wd = TD / f"syn_{stage}"
    tmp = TD / f"_check_{stage}"
    tmp.mkdir(parents=True, exist_ok=True)
    det = C.det_inputs(stage)
    write_mod(tmp / "MOD", stage_truth_mod(stage, tm))
    fw = json.loads((det / "forward.json").read_text())
    h = hyp.set_index("ID")
    la, lo = ellipsoid_coords(h.lat.values, h.lon.values, *fw["center"])
    C.write_event_dat(tmp / "event.dat", det / "event.dat",
                      pd.DataFrame({"lat": la, "lon": lo, "depth": h.depth.values}, index=h.index))
    key = dict(mod_md5=hashlib.md5((tmp / "MOD").read_bytes()).hexdigest(),
               event_md5=hashlib.md5((tmp / "event.dat").read_bytes()).hexdigest())
    if (wd / "absolute.syn").exists() and (wd / "DONE").exists():
        old = json.loads((wd / "DONE").read_text())
        if all(old.get(k) == v for k, v in key.items()):
            shutil.rmtree(tmp)
            return stage, "exists (same MOD and event.dat)"
    if wd.exists():
        shutil.rmtree(wd)
    (wd / "Input_Files").mkdir(parents=True)
    shutil.move(str(tmp / "MOD"), wd / "MOD")
    shutil.move(str(tmp / "event.dat"), wd / "Input_Files" / "event.dat")
    shutil.rmtree(tmp)
    for f in ("station.dat", "absolute.dat", "dt.ct"):
        (wd / "Input_Files" / f).symlink_to(det / f)
    for f in ("ak135.15.SKS", "layer-16.dat"):
        shutil.copy(Path(C.E["runtime_files"]) / f, wd / f)
    summ = json.loads((det / "summary.json").read_text())
    (wd / "tomoDD-SE_syn.inp").write_text(SYN_INP.format(air=summ["air_dep_km"], stepl=T["inp"]["stepl_km"],
                                                         clat=summ["center_lat"], clon=summ["center_lon"]))
    t0 = time.time()
    env = {"PATH": "/usr/bin:/bin", "HOME": str(Path.home()), "OMP_NUM_THREADS": "1"}
    with open(wd / "stdout.txt", "w") as so, open(wd / "time.txt", "w") as se:
        rc = subprocess.run(["nice", "-n", str(C.E["nice"]), "/usr/bin/time", "-f", "%e s %U user %M KB",
                             C.binary(stage, "syn"), "tomoDD-SE_syn.inp"], cwd=wd, stdout=so, stderr=se,
                            env=env).returncode
    if rc or not (wd / "absolute.syn").exists():
        raise SystemExit(f"tomoDD-SE_syn {stage}: exit {rc}")
    C.jdump(wd / "DONE", dict(wall_min=round((time.time() - t0) / 60, 2), time=(wd / "time.txt").read_text()[-120:],
                              finished=time.strftime("%F %T"), **key))
    return stage, "done"


def truth():
    TD.mkdir(parents=True, exist_ok=True)
    seed = int(PR["truth_seed"])
    tm, fp, fs = truth_model()
    np.savez_compressed(TD / "truth_model.npz", lon=tm["lon"], lat=tm["lat"], z=tm["z"], vp=tm["vp"].astype("f4"),
                        vs=tm["vs"].astype("f4"), rf_p=fp, rf_s=fs)
    ev = pd.read_parquet(PRIOR / "events.parquet")
    hyp, hinfo = EM.draw_hypo(seed, ev)
    hyp.to_parquet(TD / "truth_hypo.parquet")
    lam = pd.read_parquet(PRIOR / "class_factors.parquet").set_index("cls").lam_mad.to_dict()
    delta, stat, dinfo = EM.draw_delta(seed, lam_scale=lam)
    np.save(TD / "delta_true.npy", delta)
    with ThreadPoolExecutor(len(STAGES)) as ex:
        res = list(ex.map(lambda s: syn_stage(s, tm, hyp), STAGES))
    print(res, flush=True)
    chk = {}
    for s in STAGES:
        wd = TD / f"syn_{s}"
        a = pd.read_csv(wd / "absolute.syn", sep=r"\s+", header=None, names=["event_ID", "code", "t_syn", "w", "phase"])
        syn = a.set_index(["event_ID", "code", "phase"]).t_syn
        syn = syn[~syn.index.duplicated()]
        ab, dt = pd.read_parquet(PRIOR / f"abs_real_{s}.parquet"), pd.read_parquet(PRIOR / f"dt_real_{s}.parquet")
        t = pd.MultiIndex.from_frame(ab[["event_ID", "code", "phase"]]).map(syn).values.astype(float)
        miss = int(np.isnan(t).sum())
        if miss:
            raise SystemExit(f"{s}: {miss} picks without a synthetic time")
        ab["t"] = t + delta[ab.pidx.values]
        t1 = pd.MultiIndex.from_frame(dt[["ev1", "code", "phase"]]).map(syn).values.astype(float)
        t2 = pd.MultiIndex.from_frame(dt[["ev2", "code", "phase"]]).map(syn).values.astype(float)
        if np.isnan(t1).any() or np.isnan(t2).any():
            raise SystemExit(f"{s}: dt picks without a synthetic time")
        dt["t1"], dt["t2"] = t1 + delta[dt.pidx1.values], t2 + delta[dt.pidx2.values]
        ab.to_parquet(SD / f"abs_synth_{s}.parquet")
        dt.to_parquet(SD / f"dt_synth_{s}.parquet")
        real = pd.read_parquet(PRIOR / f"abs_real_{s}.parquet").t.values
        d = t - real
        # dt.syn consistency on its first 100k lines
        ds = pd.read_csv(wd / "dt.syn", sep=r"\s+", header=None, nrows=100_000,
                         names=["e1", "e2", "code", "t1", "t2", "w", "phase"])
        k = ds.merge(a, left_on=["e1", "code", "phase"], right_on=["event_ID", "code", "phase"])
        chk[s] = dict(picks=len(ab), dt=len(dt), syn_minus_real_median_s=round(float(np.median(d)), 4),
                      syn_minus_real_mad_s=round(float(1.4826 * np.median(np.abs(d - np.median(d)))), 4),
                      dt_syn_vs_abs_syn_maxdiff_s=float(np.abs(k.t1 - k.t_syn).max()),
                      run=json.loads((wd / "DONE").read_text()))
    info = dict(seed=seed, config=C.CONFIG, truth_model=f"v{CFG['truth_model']} tomography nodes x (1 + f)",
                rf_p_std_pct=float(100 * fp[:, :, :12].std()), rf_s_std_pct=float(100 * fs[:, :, :12].std()),
                hypo=hinfo, data=dinfo, stages=chk)
    C.jdump(TD / "truth.json", info)
    print(json.dumps({k: v for k, v in info.items() if k != "data"}, indent=1, default=str), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "truth":
        truth()

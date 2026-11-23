"""Track B Bayesian ensemble (D-B030): synthetic truth for the calibration (notes/bayes_design.md section 5).

  python b2_ens_synth.py truth [ENS]        (default ENS = synth -> <runs>/mendocino_gorda_v2/ensemble_synth/)

Truth
* model: the v4-NLL3D MOD of mendocino_gorda_v2 (Vp/Vs 1.76, nearest column) times (1 + f) with one draw of the
  3% Vp and Vs random fields of item 1 (seed TRUTH_SEED); boundary nodes re-padded as the builder does
* hypocentres: the baseline relocations d1000_s20 (tomoDD.reloc, tomoDD's rescaled frame; truth in geographic
  coordinates = tomoDD.reloc_geo); the 2 events tomoDD dropped keep their v4-NLL3D position
* times: tomoDD-SE_syn (params tomodd.bin_syn) on the real event/station geometry and the real pick list (v2
  conventions: ellipsoid-rescaled coordinates, same MOD node layout, same air depth), as b2_smooth_spike_test.py
* "known" statics, forward-error offsets and pick noise: one draw of items 2-5 (seed TRUTH_SEED) with lambda_c at
  its point estimate; added to the synthetic times (no P-4 correction: the synthetic forward is tomoDD itself)
Writes <ens>/truth/{MOD_true, truth_model.npz, truth_hypo.csv, truth.json, syn/} and <ens>/base/picks.parquet
(t_abs = t_dt = synthetic time + truth perturbation; weights, sigma and pick set as the real data).
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from b2_ens_common import (BASELINE, INPUTS, PRIOR, PRIOR_CFG as C, T, ens_dir, random_field, read_reloc,  # noqa: E402
                           repad, rng)
from b2_ens_member import draw_data  # noqa: E402
from b2_fwd_common import read_mod, write_mod  # noqa: E402
from b2_smooth_spike_test import SYN_INP  # noqa: E402
from manifest import write_manifest  # noqa: E402

TRUTH_SEED = 100_000


def truth(ens="synth"):
    E = ens_dir(ens)
    td = E / "truth"
    sd = td / "syn"
    (sd / "Input_Files").mkdir(parents=True, exist_ok=True)
    (E / "base").mkdir(parents=True, exist_ok=True)
    summ = json.loads((INPUTS / "summary.json").read_text())
    # model
    m = read_mod(INPUTS / "MOD")
    s = np.load(INPUTS / "start_model.npz")
    args = (s["lon"], s["lat"], s["z"], C["rf_std"], C["rf_len_lat_km"], C["rf_len_z_km"], C["rf_taper_km"])
    fp = random_field(rng(TRUTH_SEED, "rf_p"), *args)
    fs = random_field(rng(TRUTH_SEED, "rf_s"), *args)
    m["vp"][1:-1, 1:-1, 1:-1] *= 1 + fp
    m["vs"][1:-1, 1:-1, 1:-1] *= 1 + fs
    m = repad(m)
    write_mod(td / "MOD_true", m)
    mt = read_mod(td / "MOD_true")                  # as tomoDD reads it (4/5 decimals)
    np.savez_compressed(td / "truth_model.npz", vp=mt["vp"], vs=mt["vs"], fp=fp, fs=fs)
    # hypocentres: baseline relocations (rescaled frame for the syn run, geographic for the truth table)
    rel = read_reloc(BASELINE / "Output_Files" / "tomoDD.reloc")
    relg = read_reloc(BASELINE / "Output_Files" / "tomoDD.reloc_geo")
    lines = (INPUTS / "event.dat").read_text().splitlines()
    out, n_keep = [], 0
    for l in lines:
        t = l.split()
        eid = int(t[9])
        if eid in rel.index:
            r = rel.loc[eid]
            out.append(f"{t[0]}  {t[1]} {r.lat:10.6f} {r.lon:11.6f} {r.depth:8.3f} {' '.join(t[5:])}")
        else:
            out.append(l)
            n_keep += 1
    (sd / "Input_Files" / "event.dat").write_text("\n".join(out) + "\n")
    ev = pd.read_csv(INPUTS / "events.csv").set_index("ID")
    th = ev[["lat", "lon", "depth_bsl_km"]].rename(columns={"depth_bsl_km": "depth"}).copy()
    th.loc[relg.index.intersection(th.index), ["lat", "lon", "depth"]] = relg.loc[relg.index.intersection(th.index)].values
    th["from_baseline"] = th.index.isin(relg.index)
    th.reset_index().to_csv(td / "truth_hypo.csv", index=False)
    # synthetic times
    for f in ("station.dat", "absolute.dat", "dt.ct"):
        dst = sd / "Input_Files" / f
        if not dst.exists():
            dst.symlink_to(INPUTS / f)
    shutil.copy(td / "MOD_true", sd / "MOD")
    for f in ("ak135.15.SKS", "layer-16.dat"):
        shutil.copy(Path(T["bin"]).parent / f, sd / f)
    (sd / "tomoDD-SE_syn.inp").write_text(SYN_INP.format(air=summ["air_dep_km"], stepl=T["inp"]["stepl_km"],
                                                         clat=summ["center_lat"], clon=summ["center_lon"]))
    if not (sd / "absolute.syn").exists():
        with open(sd / "stdout.txt", "w") as so:
            rc = subprocess.run(["/usr/bin/time", "-f", "%e s %M KB", T["bin_syn"], "tomoDD-SE_syn.inp"], cwd=sd,
                                stdout=so, stderr=subprocess.STDOUT).returncode
        if rc:
            raise SystemExit(f"tomoDD-SE_syn exit {rc}")
    a = pd.read_csv(sd / "absolute.syn", sep=r"\s+", header=None, names=["event_ID", "code", "t_syn", "wsyn", "phase"])
    pk = pd.read_parquet(PRIOR / "picks_real.parquet")
    n0 = len(pk)
    pk = pk.merge(a[["event_ID", "code", "phase", "t_syn"]], on=["event_ID", "code", "phase"], how="left")
    miss = int(pk.t_syn.isna().sum())
    if miss:
        raise SystemExit(f"{miss} picks without a synthetic time")
    # check against dt.syn (first 200k lines): same times per pick?
    d = pd.read_csv(sd / "dt.syn", sep=r"\s+", header=None, nrows=200_000,
                    names=["e1", "e2", "code", "t1", "t2", "w", "phase"])
    chk = d.merge(a, left_on=["e1", "code", "phase"], right_on=["event_ID", "code", "phase"])
    dmax = float(np.abs(chk.t1 - chk.t_syn).max())
    lam = pd.read_csv(PRIOR / "class_factors.csv").set_index("cls").lam_mad.to_dict()
    pk = pk.sort_values(["event_ID", "phase", "code"], kind="stable").reset_index(drop=True)
    delta, info = draw_data(TRUTH_SEED, pk, lam_scale=lam)
    pk["delta_true"] = delta
    pk["t_abs"] = pk.t_syn + delta
    pk["t_dt"] = pk.t_abs
    pk.to_parquet(E / "base" / "picks.parquet")
    # residual of the real data vs the synthetic (sanity: same order as the real misfit)
    real = pd.read_parquet(PRIOR / "picks_real.parquet").set_index(["event_ID", "code", "phase"]).t_abs
    dr = pk.set_index(["event_ID", "code", "phase"]).t_syn - real
    tinfo = dict(seed=TRUTH_SEED, rf_p_std_pct=float(100 * fp[:, :, s["z"] <= 50].std()),
                 rf_s_std_pct=float(100 * fs[:, :, s["z"] <= 50].std()), events_kept_v4=n_keep, picks=n0,
                 dt_syn_vs_abs_syn_maxdiff_s=dmax, syn_minus_real_median_s=float(dr.median()),
                 syn_minus_real_mad_s=float(1.4826 * (dr - dr.median()).abs().median()),
                 truth_noise=info)
    (td / "truth.json").write_text(json.dumps(tinfo, indent=1))
    print(json.dumps({k: v for k, v in tinfo.items() if k != "truth_noise"}, indent=1))
    print("truth lambda", info["lambda_c"], "delta sd", round(info["delta_sd"], 4))
    write_manifest("b2_ens_synth_truth", td, inputs=[INPUTS / "MOD", BASELINE / "Output_Files" / "tomoDD.reloc",
                                                     T["bin_syn"], PRIOR / "picks_real.parquet"],
                   outputs=[td / "MOD_true", td / "truth_model.npz", td / "truth_hypo.csv", E / "base" / "picks.parquet"],
                   params={"truth_seed": TRUTH_SEED, "cfg": C}, counts={k: v for k, v in tinfo.items() if k != "truth_noise"})


if __name__ == "__main__":
    if sys.argv[1] == "truth":
        truth(sys.argv[2] if len(sys.argv) > 2 else "synth")

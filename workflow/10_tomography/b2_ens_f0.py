"""Track B ensemble, calibration diagnostic (review 2026-11-25): the background bias of the synthetic truth.

The calibration truth is v0.3.1 x (1 + f), floored at Vp/Vs 1.65. Its part v0.3.1 / start - 1 is not a draw of the
members' starting-model prior, so what the inversions do not recover of it is a bias common to every member, which
narrows the coverage. This maps that bias directly: a second synthetic dataset from v0.3.1 ALONE (f = 0, the same
1.65 floor, the same truth hypocentres and the same delta_true as synth/truth), inverted once with null draws (start
model = the deterministic rule, data = synthetic times + delta_true, start hypocentres = v4-NLL3D, damping factor 1,
the configured smoothing) through M0, the four tiles and the merge.

  python b2_ens_f0.py run          resumable chain: truth_f0 -> d000 (M0, tiles, merge) -> report
  python b2_ens_f0.py stage NAME   one tomoDD stage of d000 (called by run)
  python b2_ens_f0.py report       ROOT/synth/f0_bias.{md,json}

Folders: ROOT/synth/truth_f0/ (truth, syn_<stage>, abs_f0_<stage>.parquet, dt_f0_<stage>.parquet), ROOT/synth/d000/.
Nothing in synth/truth or the s*/nf* members is read for writing; the shared modules are used unchanged (their
base_tables and draws are replaced in this process only). At most 5 tomoDD processes (M0, then the four tiles),
each started only while MemAvailable stays above tomodd.ensemble.min_avail_gb + mem_per_run_gb.
Log: ROOT/f0.log (launch detached; rerun the same command to resume).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b2_ens_common as C  # noqa: E402
import b2_ens_member as EM  # noqa: E402
import b2_ens_synth as S  # noqa: E402
from b2_ens_common import CFG, E, PR, PRIOR, ROOT, STAGES, T  # noqa: E402

SD = ROOT / "synth"
TF = SD / "truth_f0"
KIND, NAME = "synth", "d000"
MD = C.member_dir(KIND, NAME)


def log(*a):
    print(time.strftime("%F %T"), "[f0]", *a, flush=True)


# ---------------------------------------------------------------- truth f0
def truth_model_f0():
    import xarray as xr
    nd = xr.open_dataset(Path(T["models"]["store"]) / T["models"]["name"] / f"v{CFG['truth_model']}.nc")
    vpn = "Vp_tomography" if "Vp_tomography" in nd else "Vp"
    vsn = "Vs_tomography" if "Vs_tomography" in nd else "Vs"
    vp, vs = (nd[v].values.transpose(2, 1, 0).astype(float) for v in (vpn, vsn))
    vs = np.where(vp / vs < S.FLOOR, vp / S.FLOOR, vs)          # the same floor as the main truth (P-E5)
    return dict(lon=nd.longitude.values, lat=nd.latitude.values, z=nd.depth.values, vp=vp, vs=vs)


def truth_f0():
    if (TF / "DONE").exists():
        return
    TF.mkdir(parents=True, exist_ok=True)
    tm = truth_model_f0()
    np.savez_compressed(TF / "truth_model.npz", lon=tm["lon"], lat=tm["lat"], z=tm["z"], vp=tm["vp"].astype("f4"),
                        vs=tm["vs"].astype("f4"))
    hyp = pd.read_parquet(S.TD / "truth_hypo.parquet")              # the main truth's hypocentres and delta_true
    delta = np.load(S.TD / "delta_true.npy")
    hyp.to_parquet(TF / "truth_hypo.parquet")
    np.save(TF / "delta_true.npy", delta)
    S.TD = TF                                                        # syn_stage writes into TF in this process only
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(len(STAGES)) as ex:
        res = list(ex.map(lambda s: S.syn_stage(s, tm, hyp), STAGES))
    log("syn", res)
    info = {}
    for s in STAGES:
        a = pd.read_csv(TF / f"syn_{s}" / "absolute.syn", sep=r"\s+", header=None,
                        names=["event_ID", "code", "t_syn", "w", "phase"])
        syn = a.set_index(["event_ID", "code", "phase"]).t_syn
        syn = syn[~syn.index.duplicated()]
        ab, dt = pd.read_parquet(PRIOR / f"abs_real_{s}.parquet"), pd.read_parquet(PRIOR / f"dt_real_{s}.parquet")
        t = pd.MultiIndex.from_frame(ab[["event_ID", "code", "phase"]]).map(syn).values.astype(float)
        t1 = pd.MultiIndex.from_frame(dt[["ev1", "code", "phase"]]).map(syn).values.astype(float)
        t2 = pd.MultiIndex.from_frame(dt[["ev2", "code", "phase"]]).map(syn).values.astype(float)
        if np.isnan(t).any() or np.isnan(t1).any() or np.isnan(t2).any():
            raise SystemExit(f"{s}: picks without a synthetic time")
        ab["t"] = t + delta[ab.pidx.values]
        dt["t1"], dt["t2"] = t1 + delta[dt.pidx1.values], t2 + delta[dt.pidx2.values]
        ab.to_parquet(TF / f"abs_f0_{s}.parquet")
        dt.to_parquet(TF / f"dt_f0_{s}.parquet")
        main = pd.read_parquet(SD / f"abs_synth_{s}.parquet").t.values
        info[s] = dict(picks=len(ab), f0_minus_main_median_s=round(float(np.median(ab.t.values - main)), 4),
                       f0_minus_main_mad_s=round(float(1.4826 * np.median(np.abs(ab.t.values - main))), 4))
    C.jdump(TF / "DONE", dict(finished=time.strftime("%F %T"), stages=info))
    log("truth_f0 done", info)


# ---------------------------------------------------------------- d000 with null draws
def null_draws():
    if (MD / "draws.json").exists():
        return
    MD.mkdir(parents=True, exist_ok=True)
    x, y, z = C.rf_lattice()
    u = pd.read_parquet(PRIOR / "picks.parquet")
    sp = pd.read_parquet(PRIOR / "static_sd.parquet")
    zero = np.zeros((len(x), len(y), len(z)), "float32")
    np.savez(MD / "draws.npz", rf_p=zero, rf_s=zero, delta=np.zeros(len(u)), static=np.zeros(len(sp)))
    ev = pd.read_parquet(PRIOR / "events.parquet")
    ev[["ID", "lat", "lon"]].assign(depth=ev.depth_bsl_km.values).to_parquet(MD / "start_hypo.parquet")
    C.jdump(MD / "draws.json", dict(kind=KIND, member=NAME, seed=None, eps_clone=None, config=C.CONFIG,
                                    component="v4_176", rf_p_std_pct=0.0, rf_s_std_pct=0.0, damp_factor=1.0,
                                    damp={s: float(C.stage_cfg(s)["damp"]) for s in STAGES},
                                    smooth={s: C.stage_cfg(s)["smooth"] for s in STAGES},
                                    shallow_crust=dict(vpvs=2.1, thickness_km=2.5),
                                    data=dict(note="null draws: data = synthetic (f = 0) + delta_true only"),
                                    hypo=dict(note="v4-NLL3D starting hypocentres")))


def _patch():
    EM.base_tables = lambda kind, stage: (pd.read_parquet(TF / f"abs_f0_{stage}.parquet"),
                                          pd.read_parquet(TF / f"dt_f0_{stage}.parquet"))
    EM.draws = lambda kind, name: json.loads((C.member_dir(kind, name) / "draws.json").read_text())


def stage(name):
    _patch()
    EM.run_stage(KIND, NAME, name)


def mem_ok():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1e6 >= float(E["min_avail_gb"]) + float(E["mem_per_run_gb"])
    return False


def run_stages():
    env = {**os.environ, "LD_LIBRARY_PATH": f"{C.PY_DEFAULT}/lib", "OMP_NUM_THREADS": "1"}
    py = f"{C.PY_DEFAULT}/bin/python"
    for wave in (["m0"], STAGES[1:]):
        todo = [s for s in wave if not (MD / s / "DONE").exists()]
        procs = {}
        for s in todo:
            while not mem_ok():
                time.sleep(30)
            fh = open(MD / f"{s}.log", "a")
            procs[s] = (subprocess.Popen([py, str(Path(__file__).resolve()), "stage", s], stdout=fh,
                                         stderr=subprocess.STDOUT, env=env, cwd=HERE), fh)
            log("started", s)
            time.sleep(60)                                      # let it allocate before the next memory check
        for s, (p, fh) in procs.items():
            rc = p.wait()
            fh.close()
            log(s, "exit", rc)
        bad = [s for s in wave if not (MD / s / "DONE").exists()]
        if bad:
            raise SystemExit(f"d000 stages not done: {bad} (rerun to resume)")
    if not (MD / "MERGED").exists():
        EM.merge(KIND, NAME)


# ---------------------------------------------------------------- report
def report():
    import pickle
    import xarray as xr
    tm = np.load(TF / "truth_model.npz")
    M = np.load(MD / "merged.npz")
    nd = xr.open_dataset(Path(T["models"]["store"]) / T["models"]["name"] / f"v{CFG['truth_model']}.nc")
    tr = lambda a: np.asarray(a).transpose(2, 1, 0)
    start31 = {"P": tr(nd.Vp_start.values), "S": tr(nd.Vs_start.values)}
    with open(PRIOR / "merge_geometry.pkl", "rb") as f:
        GEO = pickle.load(f)
    shp = GEO["shape"]
    rows = []
    for ph, key, res in (("P", "vp", "resolution_p"), ("S", "vs", "resolution_s")):
        gate = (tr(nd[res].values) > 0) & (M[f"r_{ph.lower()}"] > 0)         # the D-B030 gate mask (one member)
        truth, rec = tm[key].astype(float), M[key].astype(float)
        bg = truth / start31[ph] - 1                                       # v0.3.1 / start - 1 (floored truth)
        got = rec / start31[ph] - 1
        bg_own = truth / M[f"{key}_start"] - 1                             # against d000's own start
        got_own = rec / M[f"{key}_start"] - 1
        for g in GEO["geo"] + [None]:
            if g is None:
                m, tile = gate, "all gate nodes"
            else:
                m, tile = gate & g["sel"].reshape(shp), g["tile"]["stage"]
            n = int(m.sum())
            if not n:
                rows.append(dict(phase=ph, tile=tile, n=0))
                continue
            err = 100 * (rec[m] - truth[m]) / truth[m]
            share = float((got[m] * bg[m]).sum() / max((bg[m] ** 2).sum(), 1e-12))
            share_own = float((got_own[m] * bg_own[m]).sum() / max((bg_own[m] ** 2).sum(), 1e-12))
            rows.append(dict(phase=ph, tile=tile, n=n, err_rms_pct=round(float(np.sqrt(np.mean(err ** 2))), 3),
                             err_median_pct=round(float(np.median(err)), 3),
                             err_mad_pct=round(float(1.4826 * np.median(np.abs(err - np.median(err)))), 3),
                             background_rms_pct=round(float(100 * np.sqrt(np.mean(bg[m] ** 2))), 3),
                             recovered_share=round(share, 3),
                             corr=round(float(np.corrcoef(got[m], bg[m])[0, 1]), 3) if n > 2 else None,
                             background_own_start_rms_pct=round(float(100 * np.sqrt(np.mean(bg_own[m] ** 2))), 3),
                             recovered_share_own_start=round(share_own, 3)))
    runs = {s: json.loads((MD / s / "DONE").read_text()) for s in STAGES}
    out = dict(created=time.strftime("%F %T"), member=str(MD), rows=rows,
               runs={s: {k: r.get(k) for k in ("wall_min", "max_rss_gb", "rms_abs_final", "events_final", "events_start")}
                     for s, r in runs.items()})
    C.jdump(SD / "f0_bias.json", out)
    md = ["# Background bias of the calibration truth (f = 0 dataset, one inversion with null draws)", "",
          f"Created {out['created']}. Truth = v{CFG['truth_model']} tomography nodes alone (no random field), Vp/Vs "
          "floor 1.65, the main truth's hypocentres and delta_true; one inversion with null draws (deterministic start "
          "rule, damping factor 1, configured smoothing) through M0, the four tiles and the merge (synth/d000).",
          "Gate nodes = the D-B030 mask (checkerboard-resolved, tile share R > 0). Per tile: gate nodes inside its "
          "events box (overlaps counted in both). err = recovered minus truth in % of the truth. background = "
          "v0.3.1 truth / v0.3.1 start - 1; recovered share = sum(rec * bg) / sum(bg^2) with rec = recovered / v0.3.1 "
          "start - 1 (1 = fully recovered, 0 = none); 'own start' uses d000's merged start instead.", "",
          "| phase | tile | gate nodes | err rms % | err median % | err MAD % | background rms % | recovered share | "
          "corr(rec, bg) | background rms % (own start) | recovered share (own start) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["n"]:
            md.append(f"| {r['phase']} | {r['tile']} | {r['n']} | {r['err_rms_pct']} | {r['err_median_pct']} | "
                      f"{r['err_mad_pct']} | {r['background_rms_pct']} | {r['recovered_share']} | {r['corr']} | "
                      f"{r['background_own_start_rms_pct']} | {r['recovered_share_own_start']} |")
        else:
            md.append(f"| {r['phase']} | {r['tile']} | 0 | | | | | | | | |")
    md += ["", "Runs: " + json.dumps(out["runs"])]
    (SD / "f0_bias.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)


def run():
    log("start", ROOT)
    truth_f0()
    null_draws()
    run_stages()
    report()
    log("finished: report", SD / "f0_bias.md")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        run()
    elif cmd == "stage":
        stage(sys.argv[2])
    else:
        report()

"""Track B Bayesian ensemble (D-B030/D-B032): run members in parallel, restartable.

  python b2_ens_run.py launch ENS --members 0-19 [--clones 5 --clone-of 0] [-j 30]
  python b2_ens_run.py one ENS m007            (one member: build, tomoDD, back-transform, trim)
  python b2_ens_run.py one ENS nf2 --clone-of 0

One member = b2_ens_member.py build -> b2_tomodd_run.py <region>/ensemble_<ENS>/<member> run --damp --smooth
(same binary, flags and host for every member, D-B026) -> b2_tomodd_backtransform.py -> trim.
Members with <member>/DONE are skipped, so `launch` can be rerun after a failure or interruption; a failed member
gets <member>/FAILED (reason) and is retried on the next launch. Log: <ens>/run_log.jsonl (one line per member).

Trim keeps (~1 MB per member): run/Output_Files/{model_geo.npz, tomoDD.reloc_geo, tomoDD.sta_geo}, dws.npz (final
P and S DWS on all nodes), resid.json (final absolute residuals per class, log summary), run/{tomoDD-SE.inp,
tomoDD-SE.log, stdout.txt, time.txt, manifest}, inputs/{draws.json, start_hypo.csv, start_model.npz}. Everything
else (copied inputs, tomoDD.res/.vel/.src, per-iteration relocations, res.ini) is deleted.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from b2_ens_common import INPUTS, PY, REGION, REPO, ens_dir, read_dws, read_res_abs  # noqa: E402
from b2_fwd_common import read_mod  # noqa: E402
from b2_tomodd_summary import parse_log  # noqa: E402

ENV = {"LD_LIBRARY_PATH": str(REPO / ".pixi/envs/default/lib"), "PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "1",
       "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "HOME": str(Path.home())}


# tomoDD must load the system libgfortran (gcc 8.5, the build compiler): the pixi libgfortran on LD_LIBRARY_PATH is
# stricter and stops at tomoDD's malformed FORMAT in the "Location summary" write (tomoDD-SE.f line 2324) after the
# first LSQR. b2_tomodd_run.py needs only yaml, so it runs without LD_LIBRARY_PATH.
ENV_TOMODD = {k: v for k, v in ENV.items() if k != "LD_LIBRARY_PATH"}


def sh(cmd, log, env=None):
    with open(log, "a") as f:
        f.write(f"$ {' '.join(map(str, cmd))}\n")
        f.flush()
        return subprocess.run(list(map(str, cmd)), cwd=REPO, stdout=f, stderr=subprocess.STDOUT,
                              env=env or ENV).returncode


def trim(md):
    run = md / "run"
    o = run / "Output_Files"
    g = np.load(o / "model_geo.npz")
    dp, ds = read_dws(o / "tomoDD.vel", g["vp"].shape)
    np.savez_compressed(md / "dws.npz", p=dp.astype("f4"), s=ds.astype("f4"))
    r = read_res_abs(o / "tomoDD.res")
    codes = pd.read_csv(INPUTS / "station_codes.csv").set_index("code").is_obs
    r["cls"] = np.where(r.code.map(codes).values, "obs_", "land_") + r.phase.values
    res = {c: dict(n=int(len(x)), rms=float(np.sqrt(np.mean(x.res ** 2))),
                   mad=float(1.4826 * np.median(np.abs(x.res - x.res.median()))), median=float(x.res.median()))
           for c, x in r.groupby("cls")}
    info = dict(resid_abs=res, log=parse_log(run / "tomoDD-SE.log"), time=(run / "time.txt").read_text().strip())
    (md / "resid.json").write_text(json.dumps(info, indent=1, default=float))
    m = read_mod(md / "inputs" / "MOD")
    np.savez_compressed(md / "inputs" / "start_model.npz", vp=m["vp"].astype("f4"), vs=m["vs"].astype("f4"))
    keep_out = {"model_geo.npz", "tomoDD.reloc_geo", "tomoDD.sta_geo"}
    for f in o.iterdir():
        if f.name not in keep_out:
            f.unlink()
    shutil.rmtree(run / "Input_Files", ignore_errors=True)
    for f in ("MOD", "ak135.15.SKS", "layer-16.dat", "grid.dat", "tomoDD-SE.res.ini"):
        (run / f).unlink(missing_ok=True)
    for f in ("absolute.dat", "dt.ct", "event.dat", "MOD", "station.dat", "summary.json"):
        (md / "inputs" / f).unlink(missing_ok=True)
    return info


def one(ens, mname, clone_of=None):
    E = ens_dir(ens)
    md = E / mname
    if (md / "DONE").exists():
        return "skip"
    md.mkdir(parents=True, exist_ok=True)
    for f in ("FAILED", "member.log"):
        (md / f).unlink(missing_ok=True)
    shutil.rmtree(md / "run", ignore_errors=True)
    log = md / "member.log"
    t0 = time.time()
    rec = dict(member=mname, clone_of=clone_of, start=time.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        cmd = PY + [HERE / "b2_ens_member.py", "build", ens, mname] + (["--clone-of", clone_of] if clone_of is not None else [])
        if sh(cmd, log):
            raise RuntimeError("build failed")
        reg = json.loads((md / "inputs" / "draws.json").read_text())["reg"]
        rel = f"{REGION}/ensemble_{ens}/{mname}"
        rc = sh(PY + [HERE / "b2_tomodd_run.py", rel, "run", "--damp", reg["damp"], "--smooth", *reg["smooth"]], log,
                env=ENV_TOMODD)
        if rc or not (md / "run" / "Output_Files" / "tomoDD.reloc").exists():
            raise RuntimeError(f"tomoDD exit {rc}")
        if sh(PY + [HERE / "b2_tomodd_backtransform.py", rel, "run"], log):
            raise RuntimeError("backtransform failed")
        info = trim(md)
        rec.update(status="done", wall_min=round((time.time() - t0) / 60, 2), time=info["time"],
                   rms_abs_final=info["log"]["rms_abs_final"], cnd_last=info["log"]["cnd_last"], **reg)
        (md / "DONE").write_text(json.dumps(rec))
    except Exception as ex:                                             # noqa: BLE001
        rec.update(status="failed", reason=str(ex), wall_min=round((time.time() - t0) / 60, 2))
        (md / "FAILED").write_text(json.dumps(rec))
    with open(E / "run_log.jsonl", "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    du = sum(p.stat().st_size for p in md.rglob("*") if p.is_file() and not p.is_symlink()) / 1e6
    print(f"{ens}/{mname}: {rec['status']} {rec.get('reason', '')} {rec['wall_min']} min, {du:.1f} MB kept", flush=True)
    return rec["status"]


def parse_range(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def launch(a):
    jobs = [(f"m{i:03d}", None) for i in parse_range(a.members)] if a.members else []
    jobs += [(f"nf{i}", a.clone_of) for i in range(a.clones)]
    print(f"{a.ens}: {len(jobs)} members, {a.j} at once", flush=True)
    E = ens_dir(a.ens)
    E.mkdir(parents=True, exist_ok=True)

    def work(j):
        cmd = PY + [HERE / "b2_ens_run.py", "one", a.ens, j[0]] + (["--clone-of", j[1]] if j[1] is not None else [])
        return subprocess.run(list(map(str, cmd)), cwd=REPO, env=ENV).returncode

    with ThreadPoolExecutor(max_workers=a.j) as ex:
        list(ex.map(work, jobs))
    st = [json.loads(l) for l in open(E / "run_log.jsonl")] if (E / "run_log.jsonl").exists() else []
    print(f"{a.ens}: done {sum((E / m / 'DONE').exists() for m, _ in jobs)} / {len(jobs)}; log lines {len(st)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["launch", "one"])
    ap.add_argument("ens")
    ap.add_argument("member", nargs="?")
    ap.add_argument("--members")
    ap.add_argument("--clones", type=int, default=0)
    ap.add_argument("--clone-of", type=int, default=0)
    ap.add_argument("-j", type=int, default=30)
    a = ap.parse_args()
    if a.cmd == "one":
        one(a.ens, a.member, a.clone_of if a.member.startswith("nf") else None)
    else:
        launch(a)


if __name__ == "__main__":
    main()

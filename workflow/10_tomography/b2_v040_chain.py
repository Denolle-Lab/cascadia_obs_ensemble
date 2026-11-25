"""v0.4.0 chain (PI 2026-11-25: "re-run v0.3.1 with the correct re-timed picks"): the v0.3.1 chain with only the pick
times changed, plus the rule checks and the numerical noise floor. Detached, resumable, at most MAX_TOMODD tomoDD
processes at once (every job that runs tomoDD holds one slot), nice 10.

  cd ~/wt_retime
  setsid nohup nice -n 10 /home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/bin/python \
      workflow/10_tomography/b2_v040_chain.py >> /wd1/mdenolle_data/tomo_v1/v040_retimed/chain.log 2>&1 < /dev/null &
  (LD_LIBRARY_PATH=/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/lib; see RESUME.md)

Jobs (dependency graph; a job is skipped when its done-check passes, so a rerun resumes; a failed job stops its
dependants, the others go on; the chain exits 1 if any job failed):
  dataset          b1t_retime_dataset.py (b1_dataset_ct, b1b_balanced_ct)
  qc_check         b1t_qc_check.py (would the B1 MAD cuts choose differently?)
  inp_m0           b2_tomodd_inputs.py margin_m0_v3 (P-4 recomputed by its 1-iteration forward run: 1 slot)
  ver_m0           b2_v040_verify_inputs.py margin_m0_v3 margin_m0_v2 (only the times may differ)
  chk_m0           bounds-checked run check01 (D-B026), as the v0.3.1 build stage
  m0_s480          M0 = margin_m0_v3 d200_s480 (D-B046 choice kept), then the back-transform
  m0_s<W>          M0 smoothing scan 20-1920 (rule check of D-B046 only)
  inp_m0nf, m0nf_s480        noise-floor M0 (b2_v040_noise.py: N(0, 1e-6 s) on every time)
  inp_<t>, ver_<t>, chk_<t>  tiles <t>_m0t from the new M0 (P-4 reused from <t>_v2 as v0.3.1), verified against <t>_m0
  run_<t>          the v0.3.1 smoothing of the tile (data/tomo_v1/b2_merge/tiles_m0t/choice.csv), damping 200
  scan_<t>_s<W>    the other smoothings of the D-B046 tile scan (rule check only)
  inp_<t>nf, run_<t>nf       noise-floor tiles from the noise-floor M0
  merge            b2_merge.py --product product_v040 0.4.0 (D-B049 gate with the v0.3.1 d200 masks, D-B050, D-B051),
                   status intermediate, no NonLinLoc grids; exported and registered by b2_model_export.py
  merge_nf         b2_merge.py --product product_v040nf --dry-run (noise floor of the merged model, not registered)
  summaries        b2_tomodd_summary.py for every new region (scan tables for the rule checks)
  analyse          b2_v040_analyse.py (report)
State: <out>/chain.state (one line per finished job with start, end, seconds), job logs <out>/logs/<job>.log.
"""
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import REPO, load_params  # noqa: E402

P = load_params()
T, RT = P["tomodd"], P["retime"]
RUNS = Path(T["runs"])
OUT = Path(RT["out"])
LOGS = OUT / "logs"
STATE = OUT / "chain.state"
MAX_TOMODD = int(os.environ.get("MAX_TOMODD", 10))
MAX_OTHER = int(os.environ.get("MAX_OTHER", 4))
ENV = "/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default"
MENV = "/home/mdenolle/wt_tomo_res/.pixi/envs/models"
PY = f"LD_LIBRARY_PATH={ENV}/lib {ENV}/bin/python"
MPY = f"LD_LIBRARY_PATH={MENV}/lib {MENV}/bin/python"
W = "workflow/10_tomography"
TILES = {"mendocino_gorda_c": 480, "blanco": 20, "oregon": 480, "washington": 480}   # v0.3.1 choice.csv
TILE_SCAN = [s[0] for s in T["merge"]["tile_smooth_scan"]]
M0_SCAN = [20] + [s[0] for s in T["merge"]["m0_smooth_scan"]]


def log(*a):
    print(datetime.now().strftime("%F %T"), "[v040]", *a, flush=True)


def finished(region, tag):
    lg = RUNS / region / f"{tag}.log"
    v = RUNS / region / tag / "Output_Files" / "Vp_model.dat"
    g = RUNS / region / tag / "Output_Files" / "model_geo.npz"
    return (lg.exists() and "exit 0" in lg.read_text(errors="replace") and v.exists() and v.stat().st_size > 0
            and g.exists())


def run_cmd(region, tag, smooth):
    w1 = smooth
    rd = RUNS / region / tag
    return (f"if [ -d {rd} ]; then mv {rd} {rd}.partial.$(date +%s); fi; "
            f"{PY} {W}/b2_tomodd_run.py {region} {tag} --damp 200 --smooth {w1} {w1} {w1 // 2} "
            f"> {RUNS / region / tag}.log 2>&1 && {PY} {W}/b2_tomodd_backtransform.py {region} {tag} "
            f">> {RUNS / region / tag}.log 2>&1")


def check01_cmd(region):
    rd = RUNS / region / "check01"
    return (f"rm -rf {rd}; {PY} {W}/b2_tomodd_run.py {region} check01 --check > {RUNS / region}/check01.log 2>&1; "
            f"grep -q 'exit 0' {RUNS / region}/check01.log")


def check01_ok(region):
    lg = RUNS / region / "check01.log"
    s = RUNS / region / "inputs" / "summary.json"
    return lg.exists() and s.exists() and lg.stat().st_mtime > s.stat().st_mtime and "exit 0" in lg.read_text()


def verified(new, old):
    f = RUNS / new / "inputs" / f"verify_vs_{old}.json"
    return f.exists() and '"problems": []' in f.read_text()


def jobs():
    J = {}

    def add(name, cmd, deps=(), slot=False, done=None, prio=50):
        J[name] = dict(cmd=cmd, deps=list(deps), slot=slot, done=done, prio=prio)

    base = Path(P["paths"]["base"])
    add("dataset", f"{PY} {W}/b1t_retime_dataset.py",
        done=lambda: (base / RT["datasets"]["b1b"][1] / "picks.csv").exists()
        and (OUT / "dataset" / "b1t_retime_dataset.manifest.json").exists(), prio=0)
    add("qc_check", f"{PY} {W}/b1t_qc_check.py", deps=["dataset"],
        done=lambda: (OUT / "qc_check" / "qc_summary.json").exists(), prio=90)
    # M0
    add("inp_m0", f"{PY} {W}/b2_tomodd_inputs.py margin_m0_v3", deps=["dataset"], slot=True,
        done=lambda: (RUNS / "margin_m0_v3/inputs/summary.json").exists(), prio=1)
    add("ver_m0", f"{PY} {W}/b2_v040_verify_inputs.py margin_m0_v3 margin_m0_v2", deps=["inp_m0"],
        done=lambda: verified("margin_m0_v3", "margin_m0_v2"), prio=1)
    add("chk_m0", check01_cmd("margin_m0_v3"), deps=["ver_m0"], slot=True,
        done=lambda: check01_ok("margin_m0_v3"), prio=2)
    add("m0_s480", run_cmd("margin_m0_v3", "d200_s480", 480), deps=["ver_m0", "chk_m0"], slot=True,
        done=lambda: finished("margin_m0_v3", "d200_s480"), prio=3)
    for s in M0_SCAN:
        if s != 480:
            add(f"m0_s{s}", run_cmd("margin_m0_v3", f"d200_s{s}", s), deps=["ver_m0", "chk_m0"], slot=True,
                done=lambda s=s: finished("margin_m0_v3", f"d200_s{s}"), prio=60)
    add("inp_m0nf", f"{PY} {W}/b2_tomodd_inputs.py margin_m0_v3nf && {PY} {W}/b2_v040_noise.py margin_m0_v3nf",
        deps=["dataset"], slot=True, done=lambda: (RUNS / "margin_m0_v3nf/inputs/noise.json").exists(), prio=4)
    add("m0nf_s480", run_cmd("margin_m0_v3nf", "d200_s480", 480), deps=["inp_m0nf", "chk_m0"], slot=True,
        done=lambda: finished("margin_m0_v3nf", "d200_s480"), prio=5)
    # tiles (longest first)
    for i, (t, sc) in enumerate(sorted(TILES.items(), key=lambda kv: ["washington", "mendocino_gorda_c", "oregon",
                                                                        "blanco"].index(kv[0]))):
        r, rnf = f"{t}_m0t", f"{t}_m0tnf"
        add(f"inp_{t}", f"{PY} {W}/b2_tomodd_inputs.py {r}", deps=["m0_s480"],
            done=lambda r=r: (RUNS / r / "inputs/summary.json").exists(), prio=10 + i)
        add(f"ver_{t}", f"{PY} {W}/b2_v040_verify_inputs.py {r} {t}_m0", deps=[f"inp_{t}"],
            done=lambda r=r, t=t: verified(r, f"{t}_m0"), prio=10 + i)
        add(f"chk_{t}", check01_cmd(r), deps=[f"ver_{t}"], slot=True, done=lambda r=r: check01_ok(r), prio=10 + i)
        add(f"run_{t}", run_cmd(r, f"d200_s{sc}", sc), deps=[f"ver_{t}", f"chk_{t}"], slot=True,
            done=lambda r=r, sc=sc: finished(r, f"d200_s{sc}"), prio=20 + i)
        add(f"inp_{t}nf", f"{PY} {W}/b2_tomodd_inputs.py {rnf} && {PY} {W}/b2_v040_noise.py {rnf}",
            deps=["m0nf_s480"], done=lambda rnf=rnf: (RUNS / rnf / "inputs/noise.json").exists(), prio=30 + i)
        add(f"run_{t}nf", run_cmd(rnf, f"d200_s{sc}", sc), deps=[f"inp_{t}nf", f"chk_{t}"], slot=True,
            done=lambda rnf=rnf, sc=sc: finished(rnf, f"d200_s{sc}"), prio=30 + i)
        for s in TILE_SCAN:
            if s != sc:
                add(f"scan_{t}_s{s}", run_cmd(r, f"d200_s{s}", s), deps=[f"ver_{t}", f"chk_{t}"], slot=True,
                    done=lambda r=r, s=s: finished(r, f"d200_s{s}"), prio=70 + i)
    store = Path(T["models"]["store"]) / T["models"]["name"]
    ver = RT["version"]
    add("merge", f"{MPY} {W}/b2_merge.py --product product_v040 {ver} \"{RT['description']}\" --status intermediate",
        deps=["m0_s480"] + [f"run_{t}" for t in TILES], done=lambda: (store / f"v{ver}.nc").exists(), prio=40)
    add("merge_nf", f"mkdir -p {OUT}/noise && {MPY} {W}/b2_merge.py --product product_v040nf --dry-run "
        f"{OUT}/noise/merged_v040nf.nc", deps=["m0nf_s480"] + [f"run_{t}nf" for t in TILES],
        done=lambda: (OUT / "noise" / "merged_v040nf.nc").exists(), prio=45)
    regs = ["margin_m0_v3", "margin_m0_v3nf"] + [f"{t}_m0t" for t in TILES] + [f"{t}_m0tnf" for t in TILES]
    add("summaries", " && ".join(f"{PY} {W}/b2_tomodd_summary.py {r} 'd200_s*'" for r in regs),
        deps=[j for j in J if j.startswith(("m0", "run_", "scan_"))],
        done=lambda: all((RUNS / r / "summary_runs.csv").exists() and
                         (RUNS / r / "summary_runs.csv").stat().st_mtime > max(
                             (p.stat().st_mtime for p in (RUNS / r).glob("d200_s*.log")), default=0) for r in regs),
        prio=95)
    if (HERE / "b2_v040_analyse.py").exists():
        add("analyse", f"{PY} {W}/b2_v040_analyse.py", deps=["merge", "merge_nf", "summaries", "qc_check"],
            done=lambda: (OUT / "report" / "ANALYSE.DONE").exists(), prio=99)
    return J


def main():
    os.chdir(REPO)
    LOGS.mkdir(parents=True, exist_ok=True)
    J = jobs()
    for n, j in J.items():
        bad = [d for d in j["deps"] if d not in J]
        if bad:
            raise SystemExit(f"{n}: unknown deps {bad}")
    done = {n for n, j in J.items() if j["done"] and j["done"]()}
    log(f"start: {len(J)} jobs, {len(done)} already done; max {MAX_TOMODD} tomoDD jobs, {MAX_OTHER} others; pid "
        f"{os.getpid()}")
    running, failed, t0 = {}, set(), {}
    while True:
        for n, (p, slot) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            del running[n]
            dt = time.time() - t0[n]
            ok = rc == 0 and (J[n]["done"] is None or J[n]["done"]())
            if ok:
                done.add(n)
                with open(STATE, "a") as f:
                    f.write(f"{n}\t{datetime.fromtimestamp(t0[n]).isoformat(timespec='seconds')}\t"
                            f"{datetime.now().isoformat(timespec='seconds')}\t{dt:.0f}\n")
                log(f"done {n} ({dt / 60:.1f} min)")
            else:
                failed.add(n)
                log(f"FAILED {n} (exit {rc}, done-check {'passed' if J[n]['done'] and J[n]['done']() else 'failed'}; "
                    f"log {LOGS / n}.log)")
        blocked = {n for n in J if n not in done and any(d in failed for d in J[n]["deps"])}
        # propagate: a job blocked by a failure blocks its dependants too
        changed = True
        while changed:
            new = {n for n in J if n not in done and n not in blocked and any(d in blocked for d in J[n]["deps"])}
            changed = bool(new)
            blocked |= new
        ready = sorted((n for n in J if n not in done and n not in failed and n not in blocked and n not in running
                        and all(d in done for d in J[n]["deps"])), key=lambda n: J[n]["prio"])
        ns = sum(1 for _, s in running.values() if s)
        no = sum(1 for _, s in running.values() if not s)
        for n in ready:
            j = J[n]
            if j["slot"] and ns >= MAX_TOMODD or not j["slot"] and no >= MAX_OTHER:
                continue
            lf = open(LOGS / f"{n}.log", "a")
            lf.write(f"\n=== {datetime.now().isoformat(timespec='seconds')} {j['cmd']}\n")
            lf.flush()
            p = subprocess.Popen(["bash", "-c", j["cmd"]], stdout=lf, stderr=subprocess.STDOUT, cwd=REPO,
                                 stdin=subprocess.DEVNULL)
            running[n], t0[n] = (p, j["slot"]), time.time()
            ns += j["slot"]
            no += not j["slot"]
            log(f"start {n} (prio {j['prio']}, tomoDD slots {ns}/{MAX_TOMODD})")
        if not running:
            left = [n for n in J if n not in done]
            if failed or blocked:
                log(f"stopped: failed {sorted(failed)}; blocked {sorted(blocked)}")
                sys.exit(1)
            if left:
                log(f"nothing runnable but not done: {left}")
                sys.exit(1)
            log("chain finished")
            return
        time.sleep(15)


if __name__ == "__main__":
    main()

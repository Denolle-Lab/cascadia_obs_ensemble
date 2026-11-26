"""v0.4.1 chain (D-B083, PI 2026-11-25 07:59): the v0.4.0 chain re-inverted with tomoDD's event updates floored at the
local solid surface (D-B081 floor in the inversion), with its numerical noise floor, the merge and the product.
Detached, resumable (a job is skipped when its done-check passes), nice 10, at most MAX_TOMODD tomoDD processes of the
2026-11-25 inversion binaries at once (counted over the whole machine, so the step-1 identity runs and the step-3 tests
count too), and a new tomoDD job starts only while MemAvailable stays above MIN_AVAIL_GB + 4 GB.

  cd ~/wt_v041
  LD_LIBRARY_PATH=/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/lib setsid nohup nice -n 10 \
      /home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/bin/python workflow/10_tomography/b2_v041_chain.py \
      >> /wd1/mdenolle_data/tomo_v1/v041_floor/chain.log 2>&1 < /dev/null &

Jobs (no smoothing scans: v0.4.1 keeps v0.4.0's choices):
  ident_<cls>      gate: the step-1 identity run of size class cls (v041_floor/identity/<run>/compare.txt: every
                   Output_Files file byte-identical to v0.4.0) has passed; production of that class waits for it
  test_floor       gate: the step-3 floored blanco test has finished (exit 0) and the no-hit test is byte-identical
  inp_m0           b2_tomodd_inputs.py margin_m0_v4 (P-4 recomputed by its 1-iteration forward run, new binary)
  ver_m0           b2_v041_verify_inputs.py margin_m0_v4 margin_m0_v3 (byte-identical inputs)
  flr_m0           b2_tomodd_floor.py write + check margin_m0_v4 (45 s floor)
  chk_m0           bounds-checked check01 run WITH the floor (D-B026)
  m0               margin_m0_v4 d200_s480 (+ back-transform)
  inp_m0nf ... m0nf    noise floor: margin_m0_v4nf (b2_v040_noise.py: N(0, 1e-6 s) on every time), floored
  inp_<t>, ver_<t>, flr_<t>, chk_<t>, run_<t>      tiles <t>_m0f from the new M0 (30 s floor), v0.4.0 smoothing
  inp_<t>nf, ver_<t>nf, flr_<t>nf, run_<t>nf      noise-floor tiles from margin_m0_v4nf
  merge            b2_merge.py --product product_v041 --dry-run <out>/product/merged_v041.nc (D-B049 gate with the v0.3.1
                   d200 masks, D-B050, D-B051)
  merge_nf         b2_merge.py --product product_v041nf --dry-run <out>/noise/merged_v041nf.nc
  prod_nodes       b2_model_v041.py nodes (D-B076 product on the merged nodes; written, not registered)
  prod_nll         b2_model_v041.py nll (NonLinLoc grids by the field rule, b4_time_grids.build_field)
  prod_register    b2_model_v041.py register (candidate; registry lock, read-back checked)
  summaries        b2_tomodd_summary.py for every new region
  analyse          b2_v041_analyse.py (report)
State: <out>/chain.state, job logs <out>/logs/<job>.log.
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
T, V = P["tomodd"], P["v041"]
RUNS = Path(T["runs"])
OUT = Path(V["out"])
LOGS = OUT / "logs"
STATE = OUT / "chain.state"
MAX_TOMODD = int(os.environ.get("MAX_TOMODD", 12))
MAX_OTHER = int(os.environ.get("MAX_OTHER", 3))
MIN_AVAIL_GB = float(os.environ.get("MIN_AVAIL_GB", 40))
BIN_TAG = "frozen_2026-11-25_inv/bin/tomoDD-SE"
ENV = "/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default"
MENV = "/home/mdenolle/wt_tomo_res/.pixi/envs/models"
PY = f"LD_LIBRARY_PATH={ENV}/lib {ENV}/bin/python"
MPY = f"LD_LIBRARY_PATH={MENV}/lib {MENV}/bin/python"
W = "workflow/10_tomography"
TILES = {"mendocino_gorda_c": 480, "blanco": 20, "oregon": 480, "washington": 480}   # v0.3.1 / v0.4.0 choice
CLASS = {"margin_m0_v4": "m", "mendocino_gorda_c": "e13k", "blanco": "e10k", "oregon": "e10k", "washington": "w"}
IDENT = {"m": "margin_m0_v3_d200_s480", "e10k": "blanco_m0t_d200_s20", "e13k": "mendocino_gorda_c_m0t_d200_s480",
         "w": "washington_m0t_d200_s480"}


def log(*a):
    print(datetime.now().strftime("%F %T"), "[v041]", *a, flush=True)


def finished(region, tag):
    lg = RUNS / region / f"{tag}.log"
    v = RUNS / region / tag / "Output_Files" / "Vp_model.dat"
    g = RUNS / region / tag / "Output_Files" / "model_geo.npz"
    return (lg.exists() and "exit 0" in lg.read_text(errors="replace") and v.exists() and v.stat().st_size > 0
            and g.exists())


def run_cmd(region, tag, smooth):
    rd = RUNS / region / tag
    return (f"if [ -d {rd} ]; then mv {rd} {rd}.partial.$(date +%s); fi; "
            f"{PY} {W}/b2_tomodd_run.py {region} {tag} --damp 200 --smooth {smooth} {smooth} {smooth // 2} "
            f"> {rd}.log 2>&1 && {PY} {W}/b2_tomodd_backtransform.py {region} {tag} >> {rd}.log 2>&1")


def check01_cmd(region):
    rd = RUNS / region / "check01"
    return (f"rm -rf {rd}; {PY} {W}/b2_tomodd_run.py {region} check01 --check > {RUNS / region}/check01.log 2>&1; "
            f"grep -q 'exit 0' {RUNS / region}/check01.log")


def check01_ok(region):
    lg = RUNS / region / "check01.log"
    s = RUNS / region / "inputs" / "surface.dat"
    return lg.exists() and s.exists() and lg.stat().st_mtime > s.stat().st_mtime and "exit 0" in lg.read_text()


def verified(new, old):
    f = RUNS / new / "inputs" / f"verify_vs_{old}.json"
    return f.exists() and '"problems": []' in f.read_text()


def floored(region):
    i = RUNS / region / "inputs"
    return (i / "surface.dat").exists() and (i / "floor_check.json").exists()


def ident_state(cls):
    """None while the identity run of class cls runs; True if every Output_Files file is byte-identical."""
    d = OUT / "identity" / IDENT[cls]
    if not (d / "DONE").exists():
        return None
    c = d / "compare.txt"
    return ((d / "exit.txt").read_text().strip() == "exit 0" and c.exists()
            and c.read_text().strip().endswith("different 0"))


def test_state():
    """None while the step-3 tests run; True if both exited 0 and the no-hit run is byte-identical to v0.4.0."""
    t = OUT / "test"
    if not all((t / n / "DONE").exists() for n in ("blanco_floor", "blanco_nohit")):
        return None
    return (all((t / n / "exit.txt").read_text().strip() == "exit 0" for n in ("blanco_floor", "blanco_nohit"))
            and "DIFFERENT" not in (t / "blanco_nohit" / "compare.txt").read_text())


def gate_state(name):
    return test_state() if name == "test" else ident_state(name)


def wait_cmd(name):
    """Gate job: waits for the step-1 / step-3 run and fails if it did not pass."""
    return f"{PY} {W}/b2_v041_chain.py --wait {name}"


def jobs():
    J = {}

    def add(name, cmd, deps=(), slot=False, done=None, prio=50):
        J[name] = dict(cmd=cmd, deps=list(deps), slot=slot, done=done, prio=prio)

    for cls in IDENT:
        add(f"ident_{cls}", wait_cmd(cls), done=lambda cls=cls: ident_state(cls) is True, prio=0)
    add("test_floor", wait_cmd("test"), done=lambda: test_state() is True, prio=0)
    gate = ["ident_m", "test_floor"]
    # M0 and its noise-floor copy
    for r, old, nf in (("margin_m0_v4", "margin_m0_v3", False), ("margin_m0_v4nf", "margin_m0_v3nf", True)):
        k = "m0nf" if nf else "m0"
        noise = f" && {PY} {W}/b2_v040_noise.py {r}" if nf else ""
        add(f"inp_{k}", f"{PY} {W}/b2_tomodd_inputs.py {r}{noise}", deps=gate, slot=True,
            done=lambda r=r, nf=nf: (RUNS / r / "inputs" / ("noise.json" if nf else "summary.json")).exists(), prio=1)
        add(f"ver_{k}", f"{PY} {W}/b2_v041_verify_inputs.py {r} {old}", deps=[f"inp_{k}"],
            done=lambda r=r, old=old: verified(r, old), prio=1)
        add(f"flr_{k}", f"{PY} {W}/b2_tomodd_floor.py write {r} && {PY} {W}/b2_tomodd_floor.py check {r}",
            deps=[f"inp_{k}"], done=lambda r=r: floored(r), prio=1)
    add("chk_m0", check01_cmd("margin_m0_v4"), deps=["ver_m0", "flr_m0"], slot=True,
        done=lambda: check01_ok("margin_m0_v4"), prio=2)
    add("m0", run_cmd("margin_m0_v4", "d200_s480", 480), deps=["ver_m0", "flr_m0", "chk_m0"], slot=True,
        done=lambda: finished("margin_m0_v4", "d200_s480"), prio=3)
    add("m0nf", run_cmd("margin_m0_v4nf", "d200_s480", 480), deps=["ver_m0nf", "flr_m0nf", "chk_m0"], slot=True,
        done=lambda: finished("margin_m0_v4nf", "d200_s480"), prio=4)
    # tiles, longest first
    order = ["washington", "mendocino_gorda_c", "oregon", "blanco"]
    for i, t in enumerate(sorted(TILES, key=order.index)):
        sc = TILES[t]
        ig = [f"ident_{CLASS[t]}"]
        for r, old, nf in ((f"{t}_m0f", f"{t}_m0t", False), (f"{t}_m0fnf", f"{t}_m0tnf", True)):
            k = f"{t}nf" if nf else t
            noise = f" && {PY} {W}/b2_v040_noise.py {r}" if nf else ""
            add(f"inp_{k}", f"{PY} {W}/b2_tomodd_inputs.py {r}{noise}", deps=["m0nf" if nf else "m0"],
                done=lambda r=r, nf=nf: (RUNS / r / "inputs" / ("noise.json" if nf else "summary.json")).exists(),
                prio=10 + i)
            add(f"ver_{k}", f"{PY} {W}/b2_v041_verify_inputs.py {r} {old}", deps=[f"inp_{k}"],
                done=lambda r=r, old=old: verified(r, old), prio=10 + i)
            add(f"flr_{k}", f"{PY} {W}/b2_tomodd_floor.py write {r} && {PY} {W}/b2_tomodd_floor.py check {r}",
                deps=[f"inp_{k}"], done=lambda r=r: floored(r), prio=10 + i)
        r, rnf = f"{t}_m0f", f"{t}_m0fnf"
        add(f"chk_{t}", check01_cmd(r), deps=[f"ver_{t}", f"flr_{t}"] + ig, slot=True, done=lambda r=r: check01_ok(r),
            prio=10 + i)
        add(f"run_{t}", run_cmd(r, f"d200_s{sc}", sc), deps=[f"ver_{t}", f"flr_{t}", f"chk_{t}"] + ig, slot=True,
            done=lambda r=r, sc=sc: finished(r, f"d200_s{sc}"), prio=20 + i)
        add(f"run_{t}nf", run_cmd(rnf, f"d200_s{sc}", sc), deps=[f"ver_{t}nf", f"flr_{t}nf", f"chk_{t}"] + ig,
            slot=True, done=lambda rnf=rnf, sc=sc: finished(rnf, f"d200_s{sc}"), prio=30 + i)
    prod = OUT / "product"
    add("merge", f"mkdir -p {prod} && {MPY} {W}/b2_merge.py --product product_v041 --dry-run {prod}/merged_v041.nc",
        deps=["m0"] + [f"run_{t}" for t in TILES], done=lambda: (prod / "merged_v041.nc").exists(), prio=40)
    add("merge_nf", f"mkdir -p {OUT}/noise && {MPY} {W}/b2_merge.py --product product_v041nf --dry-run "
        f"{OUT}/noise/merged_v041nf.nc", deps=["m0nf"] + [f"run_{t}nf" for t in TILES],
        done=lambda: (OUT / "noise" / "merged_v041nf.nc").exists(), prio=45)
    store = Path(T["models"]["store"]) / T["models"]["name"]
    ver = V["version"]
    add("prod_nodes", f"{MPY} {W}/b2_model_v041.py nodes", deps=["merge"],
        done=lambda: (prod / "nodes_written.json").exists(), prio=41)
    add("prod_nll", f"{MPY} {W}/b2_model_v041.py nll", deps=["prod_nodes"],
        done=lambda: (prod / "nll_written.json").exists(), prio=42)
    add("prod_register", f"{MPY} {W}/b2_model_v041.py register", deps=["prod_nll"],
        done=lambda: (store / f"v{ver}.provenance.json").exists(), prio=43)
    regs = ["margin_m0_v4", "margin_m0_v4nf"] + [f"{t}_m0f" for t in TILES] + [f"{t}_m0fnf" for t in TILES]
    add("summaries", " && ".join(f"{PY} {W}/b2_tomodd_summary.py {r} 'd200_s*'" for r in regs),
        deps=[j for j in J if j.startswith(("m0", "run_"))],
        done=lambda: all((RUNS / r / "summary_runs.csv").exists() and
                         (RUNS / r / "summary_runs.csv").stat().st_mtime > max(
                             (p.stat().st_mtime for p in (RUNS / r).glob("d200_s*.log")), default=0) for r in regs),
        prio=95)
    if (HERE / "b2_v041_analyse.py").exists():
        add("analyse", f"{PY} {W}/b2_v041_analyse.py", deps=["merge", "merge_nf", "summaries", "prod_register"],
            done=lambda: (OUT / "report" / "ANALYSE.DONE").exists(), prio=99)
    return J


def n_tomodd():
    r = subprocess.run(["pgrep", "-c", "-f", f"^[^ ]*{BIN_TAG}"], capture_output=True, text=True)
    return int(r.stdout.strip() or 0)


def mem_avail_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 ** 2
    return 0.0


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--wait":
        while True:
            st = gate_state(sys.argv[2])
            if st is not None:
                log(f"gate {sys.argv[2]}: {'passed' if st else 'FAILED'}")
                sys.exit(0 if st else 1)
            time.sleep(60)
    os.chdir(REPO)
    LOGS.mkdir(parents=True, exist_ok=True)
    J = jobs()
    for n, j in J.items():
        bad = [d for d in j["deps"] if d not in J]
        if bad:
            raise SystemExit(f"{n}: unknown deps {bad}")
    done = {n for n, j in J.items() if j["done"] and j["done"]()}
    log(f"start: {len(J)} jobs, {len(done)} already done; max {MAX_TOMODD} tomoDD processes (machine-wide, "
        f"{BIN_TAG}*), {MAX_OTHER} others, MemAvailable > {MIN_AVAIL_GB} GB; pid {os.getpid()}")
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
        changed = True
        while changed:
            new = {n for n in J if n not in done and n not in blocked and any(d in blocked for d in J[n]["deps"])}
            changed = bool(new)
            blocked |= new
        ready = sorted((n for n in J if n not in done and n not in failed and n not in blocked and n not in running
                        and all(d in done for d in J[n]["deps"])), key=lambda n: J[n]["prio"])
        no = sum(1 for _, s in running.values() if not s)
        for n in ready:
            j = J[n]
            if j["slot"]:
                # machine-wide count of this freeze's tomoDD processes (+ slot jobs just started whose binary may not
                # be up yet), and memory
                starting = sum(1 for m, (p, s) in running.items() if s and time.time() - t0[m] < 60)
                if n_tomodd() + starting >= MAX_TOMODD or mem_avail_gb() < MIN_AVAIL_GB + 4:
                    continue
            elif no >= MAX_OTHER and not n.startswith(("ident_", "test_")):
                continue
            lf = open(LOGS / f"{n}.log", "a")
            lf.write(f"\n=== {datetime.now().isoformat(timespec='seconds')} {j['cmd']}\n")
            lf.flush()
            p = subprocess.Popen(["bash", "-c", j["cmd"]], stdout=lf, stderr=subprocess.STDOUT, cwd=REPO,
                                 stdin=subprocess.DEVNULL)
            running[n], t0[n] = (p, j["slot"]), time.time()
            if not j["slot"] and not n.startswith(("ident_", "test_")):
                no += 1
            log(f"start {n} (prio {j['prio']}; tomoDD processes {n_tomodd()}, MemAvailable {mem_avail_gb():.0f} GB)")
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
        time.sleep(20)


if __name__ == "__main__":
    main()

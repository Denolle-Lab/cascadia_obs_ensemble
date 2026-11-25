"""Track B Bayesian ensemble (D-B030, D-B032): the detached, resumable chain and its member scheduler.

  python b2_ens_run.py chain                       full chain (launch detached, see RESUME.md in ROOT)
  python b2_ens_run.py members KIND NAME [NAME ..]  run these members end to end (tests), same scheduler
  python b2_ens_run.py status                      counts of done / failed stages per kind

Chain stages (ROOT/chain.state records the finished ones; rerunning the same command resumes):
  prep       b2_ens_member.py prep (prior ingredients)
  truth      b2_ens_synth.py truth (synthetic truth, tomoDD-SE_syn per stage)
  synth      synthetic members s000..s(N-1) and the noise-floor clones nf0..nf(K-1) of s000
  calibrate  b2_ens_coverage.py: the D-B030 gate. FAIL -> the chain stops with exit 2 (real members not started)
  real       real-data members r000..r(N-1)
  stats      b2_ens_stats.py stats real (per-node statistics)
  products   b2_ens_stats.py products real (versioned ensemble product, HTML report)
Scheduler: per member M0 first, then its tiles (all four at once), then the merge; at most tomodd.ensemble.max_procs
tomoDD processes of this chain, each started only while MemAvailable, minus mem_per_run_gb for every process started in
the last 5 minutes (still allocating), stays above min_avail_gb + mem_per_run_gb; tile
stages of started members go before new M0 stages (members finish early), longest tiles first. A failed stage is
retried once; a second failure marks the member failed (FAILED_MEMBER) and the chain carries on. Every tomoDD run is
niced (tomodd.ensemble.nice) and uses the frozen binaries (D-B054).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b2_ens_common as C  # noqa: E402
from b2_ens_common import E, ROOT, STAGES  # noqa: E402

PYD = [f"{C.PY_DEFAULT}/bin/python"]
PYM = [f"{C.PY_MODELS}/bin/python"]
ENVD = {**os.environ, "LD_LIBRARY_PATH": f"{C.PY_DEFAULT}/lib", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1"}
ENVM = {**ENVD, "LD_LIBRARY_PATH": f"{C.PY_MODELS}/lib"}
TILE_ORDER = ["washington", "mendocino_gorda_c", "oregon", "blanco"]      # longest first (observed run times)
STATE = ROOT / "chain.state"


def log(*a):
    print(time.strftime("%F %T"), "[ens]", *a, flush=True)


def mem_avail_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1e6
    return 0.0


def done(kind, name, stage):
    md = C.member_dir(kind, name)
    return (md / "MERGED").exists() if stage == "merge" else (md / stage / "DONE").exists()


def run_members(kind, names, max_procs=None):
    """Run members to their merge. Returns (merged, failed) name lists."""
    max_procs = int(max_procs or E["max_procs"])
    need_gb = float(E["min_avail_gb"]) + float(E["mem_per_run_gb"])
    order = ["m0"] + [t for t in TILE_ORDER if t in STAGES] + [t for t in STAGES[1:] if t not in TILE_ORDER]
    tries, running, failed = {}, {}, set()
    for n in names:
        if (C.member_dir(kind, n) / "FAILED_MEMBER").exists():
            failed.add(n)
    t0, last = time.time(), 0
    rl = ROOT / kind / "run_log.jsonl"
    rl.parent.mkdir(parents=True, exist_ok=True)
    while True:
        for key, (p, t_s, lf, fh) in list(running.items()):      # reap finished jobs
            rc = p.poll()
            if rc is None:
                continue
            fh.close()                                           # one log handle per job: keep under the fd limit
            del running[key]
            n, st = key
            ok = rc == 0 and done(kind, n, st)
            with open(rl, "a") as f:
                f.write(json.dumps(dict(member=n, stage=st, rc=rc, ok=ok, wall_min=round((time.time() - t_s) / 60, 2),
                                        end=time.strftime("%F %T"))) + "\n")
            if not ok:
                tries[key] = tries.get(key, 0) + 1
                log(f"{kind}/{n}/{st} failed (rc {rc}, try {tries[key]}); log {lf}")
                if tries[key] >= 2:
                    failed.add(n)
                    C.jdump(C.member_dir(kind, n) / "FAILED_MEMBER", dict(stage=st, rc=rc, log=str(lf)))
        pending = []                                             # (priority, member, stage)
        for i, n in enumerate(names):
            if n in failed or done(kind, n, "merge"):
                continue
            if not done(kind, n, "m0"):
                if (n, "m0") not in running:
                    pending.append((1, i, n, "m0"))
                continue
            tiles_left = [s for s in order[1:] if not done(kind, n, s)]
            for s in tiles_left:
                if (n, s) not in running:
                    pending.append((0, i * 10 + order.index(s), n, s))
            if not tiles_left and (n, "merge") not in running:
                pending.append((-1, i, n, "merge"))
        if not pending and not running:
            break
        pending.sort()
        n_tdd = sum(1 for (n, s) in running if s != "merge")
        # processes started in the last 5 minutes may not have reached their full memory yet: reserve it for them
        young = sum(1 for (n, s), v in running.items() if s != "merge" and time.time() - v[1] < 300)
        for pr, _, n, s in pending:
            if s == "merge":
                if sum(1 for (_, s2) in running if s2 == "merge") >= 4:
                    continue
                cmd = PYD + [str(HERE / "b2_ens_member.py"), "merge", kind, n]
            else:
                if n_tdd >= max_procs or mem_avail_gb() - young * float(E["mem_per_run_gb"]) < need_gb:
                    continue
                cmd = PYD + [str(HERE / "b2_ens_member.py"), "stage", kind, n, s]
                n_tdd += 1
                young += 1
            md = C.member_dir(kind, n)
            md.mkdir(parents=True, exist_ok=True)
            lf = md / f"{s}.log"
            fh = open(lf, "a")
            fh.write(f"\n{time.strftime('%F %T')} $ {' '.join(cmd)}\n")
            fh.flush()
            running[(n, s)] = (subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=ENVD, cwd=HERE,
                                                start_new_session=False), time.time(), lf, fh)
        if time.time() - last > 600:
            nd = sum(done(kind, n, "merge") for n in names)
            log(f"{kind}: {nd}/{len(names)} merged, {len(failed)} failed, {len(running)} running "
                f"({sum(1 for (_, s) in running if s != 'merge')} tomoDD), MemAvailable {mem_avail_gb():.0f} GB, "
                f"{(time.time() - t0) / 3600:.2f} h")
            last = time.time()
        time.sleep(15)
    merged = [n for n in names if done(kind, n, "merge")]
    log(f"{kind}: finished, {len(merged)} merged, {len(failed)} failed ({sorted(failed)}), "
        f"{(time.time() - t0) / 3600:.2f} h")
    return merged, sorted(failed)


def sh(cmd, env, name):
    lf = ROOT / "logs" / f"{name}.log"
    lf.parent.mkdir(parents=True, exist_ok=True)
    with open(lf, "a") as f:
        f.write(f"\n{time.strftime('%F %T')} $ {' '.join(map(str, cmd))}\n")
        f.flush()
        rc = subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT, env=env, cwd=HERE).returncode
    log(f"{name}: exit {rc} (log {lf})")
    return rc


def stage_done(s):
    return STATE.exists() and s in STATE.read_text().split()


def mark(s):
    with open(STATE, "a") as f:
        f.write(s + "\n")
    log(f"stage {s} done")


def resume_md():
    N = E["members"]
    txt = f"""# Ensemble {C.CONFIG} (Track B, D-B030/D-B032): how to resume

Written by b2_ens_run.py at the start of every chain run ({time.strftime('%F %T')}).

* Configuration: tomodd.ensemble.configs.{C.CONFIG} in params.yaml (M0 {E['configs'][C.CONFIG]['m0']['region']}/{E['configs'][C.CONFIG]['m0']['tag']}, tiles {', '.join(f"{t['region']}/{t['tag']}" for t in E['configs'][C.CONFIG]['tiles'].values())}).
* Chain state: {STATE} (one finished stage per line: prep truth synth calibrate real stats products).
* Log: {ROOT}/chain.log; per member {ROOT}/<synth|real>/<member>/<stage>.log; run log <kind>/run_log.jsonl.
* Members: synth {N['synth']} + {N['floor']} noise-floor clones, real {N['real']}; at most {E['max_procs']} tomoDD processes,
  nice {E['nice']}, MemAvailable kept above {E['min_avail_gb']} GB.
* Resume after a crash or reboot: run the launch command again (same command). Finished stages and finished member
  stages are skipped; an interrupted member stage is rerun from its inputs.
* Calibration gate: if the stage calibrate fails, the chain stops with exit 2 and the real members are not started;
  see {ROOT}/synth/coverage.md. Do not remove 'calibrate' from the state by hand without the PI's decision.
* Stop: kill the process group of the chain (pgid in {ROOT}/chain.pid).

Launch (from the code worktree):

    cd {HERE.parents[1]}
    LD_LIBRARY_PATH={C.PY_DEFAULT}/lib setsid nohup {PYD[0]} workflow/10_tomography/b2_ens_run.py chain >> {ROOT}/chain.log 2>&1 < /dev/null &
"""
    (ROOT / "RESUME.md").write_text(txt)


def chain():
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "chain.pid").write_text(f"pid {os.getpid()} pgid {os.getpgid(0)}\n")
    resume_md()
    N = E["members"]
    log(f"chain start, config {C.CONFIG}, root {ROOT}")
    if not stage_done("prep"):
        if sh(PYD + [HERE / "b2_ens_member.py", "prep"], ENVD, "prep"):
            sys.exit(1)
        mark("prep")
    if not stage_done("truth"):
        if sh(PYD + [HERE / "b2_ens_synth.py", "truth"], ENVD, "truth"):
            sys.exit(1)
        mark("truth")
    if not stage_done("synth"):
        names = [f"s{i:03d}" for i in range(N["synth"])] + [f"nf{i}" for i in range(N["floor"])]
        merged, failed = run_members("synth", names)
        if len(failed) > 0.1 * len(names):
            log(f"too many failed synthetic members ({len(failed)}): stop")
            sys.exit(3)
        mark("synth")
    if not stage_done("calibrate"):
        rc = sh(PYD + [HERE / "b2_ens_coverage.py"], ENVD, "calibrate")
        if rc == 2:
            log("CALIBRATION FAILED: real-data ensemble not started (see synth/coverage.md)")
            sys.exit(2)
        if rc:
            sys.exit(1)
        mark("calibrate")
    if not stage_done("real"):
        names = [f"r{i:03d}" for i in range(N["real"])]
        merged, failed = run_members("real", names)
        if len(failed) > 0.1 * len(names):
            log(f"too many failed real members ({len(failed)}): stop")
            sys.exit(3)
        mark("real")
    if not stage_done("stats"):
        if sh(PYD + [HERE / "b2_ens_stats.py", "stats", "real"], ENVD, "stats"):
            sys.exit(1)
        mark("stats")
    if not stage_done("products"):
        if sh(PYM + [HERE / "b2_ens_stats.py", "products", "real"], ENVM, "products"):
            sys.exit(1)
        mark("products")
    log("chain finished")


def status():
    for kind in ("synth", "real"):
        d = ROOT / kind
        if not d.exists():
            continue
        ms = sorted(p.name for p in d.iterdir() if p.is_dir() and p.name[0] in "srn")
        cnt = {s: sum(done(kind, m, s) for m in ms) for s in STAGES + ["merge"]}
        print(kind, len(ms), "members:", cnt, "failed:", sum((d / m / "FAILED_MEMBER").exists() for m in ms))


def main():
    cmd = sys.argv[1]
    if cmd == "chain":
        chain()
    elif cmd == "members":
        run_members(sys.argv[2], sys.argv[3:])
    else:
        status()


if __name__ == "__main__":
    main()

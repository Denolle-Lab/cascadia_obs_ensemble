"""Track B Bayesian ensemble (D-B030, D-B032): the detached, resumable chain and its member scheduler.

  python b2_ens_run.py chain                       full chain (launch detached, see RESUME.md in ROOT)
  python b2_ens_run.py members KIND NAME [NAME ..] [--max-procs N]   run these members end to end (tests), same
                                                   scheduler (N: at most N tomoDD processes, default max_procs)
  python b2_ens_run.py status                      counts of done / failed stages per kind
  python b2_ens_run.py clear-failed KIND (NAME .. | --all)   clear FAILED_MEMBER markers explicitly (renamed to
                                                   FAILED_MEMBER.cleared.<time>, stage FAILED markers removed), so the
                                                   next chain run retries those members
Environment: B2_ENS_CONFIG (configuration, default tomodd.ensemble.config), B2_ENS_ROOT (root, default
configs.<c>.root or tomodd.ensemble.root), B2_ENS_STOP_AFTER (stage), B2_ENS_MAX_PROCS (max_procs).

Chain stages (ROOT/chain.state records the finished ones; rerunning the same command resumes):
  prep       b2_ens_member.py prep (prior ingredients; pinned external inputs; the floor's surface.dat per stage)
  truth      b2_ens_synth.py truth (synthetic truth, tomoDD-SE_syn per stage)
  synth      synthetic members s000..s(N-1) and the noise-floor clones nf0..nf(K-1) of s000
  calibrate  b2_ens_coverage.py: the D-B030 gate. FAIL -> the chain stops with exit 2 (real members not started)
             With configs.<c>.calibration_root set (v0.4.1: the v0.3.2 synthetic calibration), truth, synth and
             calibrate are not run here: the gate is that root's passed calibration (chain.state has calibrate and
             synth/coverage.json result PASS); otherwise exit 2 before any real member.
  real       real-data members r000..r(N-1)
  stats      b2_ens_stats.py stats real (per-node statistics)
  products   b2_ens_stats.py products real (dry-run netCDF + HTML report; with B2_ENS_REGISTER=1, after the PI's go,
             the immutable version <model>+ens.<members> is registered, read-back checked first)
Scheduler: per member M0 first, then its tiles (all four at once), then the merge; at most tomodd.ensemble.max_procs
tomoDD processes of this chain, each started only while MemAvailable, minus mem_per_run_gb for every process started in
the last 5 minutes (still allocating), stays above min_avail_gb + mem_per_run_gb; tile
stages of started members go before new M0 stages (members finish early), longest tiles first. A failed stage is
retried once; a second failure marks the member failed (FAILED_MEMBER) and the chain carries on. Every tomoDD run is
niced (tomodd.ensemble.nice) and uses the frozen binaries (D-B054).
Markers and locks: members with a FAILED_MEMBER marker from an earlier run are skipped and listed at the start (clear
them with clear-failed); a marker of a member that has merged since is stale and is cleared (renamed, logged). Each
member stage runs under its own lock (<member>/<stage>/LOCK, flock + pid, pgid, host, start; b2_ens_member.py), and
the scheduler does not start a stage another process holds; one chain per root (ROOT/chain.lock).
Process group: the chain (and the members command) makes itself a process-group leader; every member stage and its
tomoDD run in that group. On an exception, SIGTERM or SIGINT the chain kills the whole group (no orphan tomoDD keeps
running) and exits non-zero.
"""
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b2_ens_common as C  # noqa: E402
from b2_ens_common import CFG, E, ROOT, STAGES, T  # noqa: E402

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


def stage_locked(kind, n, s):
    md = C.member_dir(kind, n)
    return C.is_locked(md / "merge.LOCK" if s == "merge" else md / s / "LOCK")


def clear_marker(md, why):
    f = md / "FAILED_MEMBER"
    dst = md / f"FAILED_MEMBER.cleared.{time.strftime('%Y%m%dT%H%M%S')}"
    f.rename(dst)
    for st in STAGES:
        (md / st / "FAILED").unlink(missing_ok=True)
    log(f"cleared FAILED_MEMBER of {md.parent.name}/{md.name} ({why}); kept as {dst.name}")


def failed_markers(kind, names):
    """Members with a FAILED_MEMBER marker: a stale one (the member merged since) is cleared; the others are skipped
    and listed (clear them explicitly with clear-failed)."""
    out = []
    for n in names:
        md = C.member_dir(kind, n)
        if not (md / "FAILED_MEMBER").exists():
            continue
        if (md / "MERGED").exists():
            clear_marker(md, "stale: the member has merged since")
            continue
        out.append(n)
    if out:
        log(f"{kind}: {len(out)} members skipped, FAILED_MEMBER from an earlier run: {out}; to retry them: "
            f"b2_ens_run.py clear-failed {kind} {' '.join(out)}")
    return out


def run_members(kind, names, max_procs=None):
    """Run members to their merge. Returns (merged, failed) name lists."""
    max_procs = int(max_procs or os.environ.get("B2_ENS_MAX_PROCS") or E["max_procs"])
    need_gb = float(E["min_avail_gb"]) + float(E["mem_per_run_gb"])
    order = ["m0"] + [t for t in TILE_ORDER if t in STAGES] + [t for t in STAGES[1:] if t not in TILE_ORDER]
    tries, running, failed, held = {}, {}, set(), set()
    failed.update(failed_markers(kind, names))
    log(f"{kind}: {len(names)} members, at most {max_procs} tomoDD processes, config {C.CONFIG}, root {ROOT}"
        f"{', floor on' if C.FLOOR else ''}")
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
            if stage_locked(kind, n, s):                         # another process runs this member stage
                if (n, s) not in held:
                    log(f"{kind}/{n}/{s}: locked by another process, not started")
                    held.add((n, s))
                continue
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
    cr = CFG.get("calibration_root")
    gate = (f"the synthetic calibration of {cr} (its chain.state has calibrate, synth/coverage.json PASS); this root "
            f"runs no synthetic members. If it has not passed, the chain stops with exit 2 before any real member"
            if cr else f"if the stage calibrate fails, the chain stops with exit 2 and the real members are not "
                       f"started; see {ROOT}/synth/coverage.md. Do not remove 'calibrate' from the state by hand "
                       f"without the PI's decision")
    stages = "prep calibrate(gate) real stats products" if cr else "prep truth synth calibrate real stats products"
    txt = f"""# Ensemble of model v{C.product_version()} (configuration {C.CONFIG}; products {C.product_version()}+ens.<members>): how to resume

Track B, D-B030/D-B032. Written by b2_ens_run.py at the start of every chain run ({time.strftime('%F %T')}).

* Configuration: tomodd.ensemble.configs.{C.CONFIG} in params.yaml (M0 {CFG['m0']['region']}/{CFG['m0']['tag']}, tiles {', '.join(f"{t['region']}/{t['tag']}" for t in CFG['tiles'].values())};
  dataset {CFG.get('dataset', 'as the regions')}; binaries {CFG.get('bin_dir', E['bin_dir'])}; floor {'on (D-B083: surface.dat per member stage, Air_dep ' + str(T['floor']['air_dep_km']) + ' km)' if C.FLOOR else 'off'}).
* Chain state: {STATE} (one finished stage per line: {stages}).
* Log: {ROOT}/chain.log; per member {ROOT}/<synth|real>/<member>/<stage>.log; run log <kind>/run_log.jsonl.
* Members: {'real ' + str(N['real']) if cr else f"synth {N['synth']} + {N['floor']} noise-floor clones, real {N['real']}"}; at most {os.environ.get('B2_ENS_MAX_PROCS') or E['max_procs']} tomoDD processes,
  nice {E['nice']}, MemAvailable kept above {E['min_avail_gb']} GB.
* Resume after a crash or reboot: run the launch command again (same command). Finished stages and finished member
  stages are skipped; an interrupted member stage is rerun from its inputs. Members marked FAILED_MEMBER are skipped
  until cleared: b2_ens_run.py clear-failed <kind> <names> (same environment).
* Calibration gate: {gate}.
* Stop: kill the process group of the chain (pgid in {ROOT}/chain.pid); the chain also kills its group itself on an
  exception, SIGTERM or SIGINT.

Launch (from the code worktree):

    cd {HERE.parents[1]}
    B2_ENS_CONFIG={C.CONFIG} LD_LIBRARY_PATH={C.PY_DEFAULT}/lib setsid nohup {PYD[0]} workflow/10_tomography/b2_ens_run.py chain >> {ROOT}/chain.log 2>&1 < /dev/null &
"""
    (ROOT / "RESUME.md").write_text(txt)


def maybe_stop(stage):
    """B2_ENS_STOP_AFTER=<stage> ends the chain after that stage (done now or before), e.g. calibrate: synthetic gate only."""
    if os.environ.get("B2_ENS_STOP_AFTER") == stage:
        log(f"stopping after {stage} (B2_ENS_STOP_AFTER); rerun without it to continue")
        sys.exit(0)


MEMBER_KEYS = ("m0", "tiles", "dataset", "bin_dir", "runtime_files", "floor", "merge_product", "node_model", "mask_damp")


def check_prepared():
    """The root's members share one prior and one configuration: the effective prior and the member-relevant
    configuration keys now must be those prep recorded in prior/prior.json (else: a new root)."""
    pj = C.PRIOR / "prior.json"
    pri = json.loads(pj.read_text())
    bad = []
    if json.loads(json.dumps(C.PR, default=str)) != pri.get("prior"):
        bad.append("prior (tomodd.ensemble.prior / configs.<c>.prior)")
    rec = pri.get("cfg") or {}
    bad += [f"configs.{C.CONFIG}.{k}" for k in MEMBER_KEYS
            if json.loads(json.dumps(CFG.get(k), default=str)) != rec.get(k)]
    if bad:
        log(f"{pj}: changed since prep: {', '.join(bad)}; members of one root must share them: use a new root")
        sys.exit(1)


def calibration_gate():
    """Real phase gated on another root's synthetic calibration (configs.<c>.calibration_root): exit 2 unless it passed."""
    cr = Path(CFG["calibration_root"])
    st, cj = cr / "chain.state", cr / "synth" / "coverage.json"
    done_ = st.exists() and "calibrate" in st.read_text().split()
    res = json.loads(cj.read_text()).get("result") if cj.exists() else None
    if not (done_ and res == "PASS"):
        log(f"CALIBRATION GATE NOT PASSED: {cr} (chain.state calibrate: {done_}, coverage.json result: {res}); "
            "real-data ensemble not started")
        sys.exit(2)
    log(f"calibration gate: {cj} result PASS")


def chain():
    ROOT.mkdir(parents=True, exist_ok=True)
    lk = C.Lock(ROOT / "chain.lock")
    if not lk.acquire(f"chain {C.CONFIG}"):
        raise SystemExit(f"another chain runs on {ROOT}: {lk.owner()}")
    C.check_config()
    (ROOT / "chain.pid").write_text(f"pid {os.getpid()} pgid {os.getpgid(0)}\n")
    resume_md()
    N = E["members"]
    cr = CFG.get("calibration_root")
    log(f"chain start, config {C.CONFIG}, root {ROOT}, floor {'on' if C.FLOOR else 'off'}"
        f"{', calibration from ' + cr if cr else ''}; pid {os.getpid()} pgid {os.getpgid(0)}")
    if not stage_done("prep"):
        if sh(PYD + [HERE / "b2_ens_member.py", "prep"], ENVD, "prep"):
            sys.exit(1)
        mark("prep")
    check_prepared()
    maybe_stop("prep")
    if cr:
        calibration_gate()                     # every run: the gate is the other root's current verdict
        maybe_stop("calibrate")
    if not cr and not stage_done("truth"):
        if sh(PYD + [HERE / "b2_ens_synth.py", "truth"], ENVD, "truth"):
            sys.exit(1)
        mark("truth")
    maybe_stop("truth")
    if not cr and not stage_done("synth"):
        names = [f"s{i:03d}" for i in range(N["synth"])] + [f"nf{i}" for i in range(N["floor"])]
        merged, failed = run_members("synth", names)
        if len(failed) > 0.1 * len(names):
            log(f"too many failed synthetic members ({len(failed)}): stop")
            sys.exit(3)
        mark("synth")
    maybe_stop("synth")
    if not cr and not stage_done("calibrate"):
        rc = sh(PYD + [HERE / "b2_ens_coverage.py"], ENVD, "calibrate")
        if rc == 2:
            log("CALIBRATION FAILED: real-data ensemble not started (see synth/coverage.md)")
            sys.exit(2)
        if rc:
            sys.exit(1)
        mark("calibrate")
    if not cr:
        maybe_stop("calibrate")
    if not stage_done("real"):
        names = [f"r{i:03d}" for i in range(N["real"])]
        merged, failed = run_members("real", names)
        if len(failed) > 0.1 * len(names):
            log(f"too many failed real members ({len(failed)}): stop")
            sys.exit(3)
        mark("real")
    maybe_stop("real")
    if not stage_done("stats"):
        if sh(PYD + [HERE / "b2_ens_stats.py", "stats", "real"], ENVD, "stats"):
            sys.exit(1)
        mark("stats")
    maybe_stop("stats")
    if not stage_done("products"):
        if sh(PYM + [HERE / "b2_ens_stats.py", "products", "real"], ENVM, "products"):
            sys.exit(1)
        mark("products")
    maybe_stop("products")
    log("chain finished")


def status():
    for kind in ("synth", "real"):
        d = ROOT / kind
        if not d.exists():
            continue
        ms = sorted(p.name for p in d.iterdir() if p.is_dir() and p.name[0] in "srn")
        cnt = {s: sum(done(kind, m, s) for m in ms) for s in STAGES + ["merge"]}
        print(kind, len(ms), "members:", cnt, "failed:", sum((d / m / "FAILED_MEMBER").exists() for m in ms),
              "locked stages:", sum(stage_locked(kind, m, s) for m in ms for s in STAGES + ["merge"]))


def clear_failed(kind, names):
    d = ROOT / kind
    if names == ["--all"]:
        names = sorted(p.name for p in d.iterdir() if (p / "FAILED_MEMBER").exists()) if d.exists() else []
    for n in names:
        md = C.member_dir(kind, n)
        if (md / "FAILED_MEMBER").exists():
            clear_marker(md, "cleared by hand (clear-failed)")
        else:
            log(f"{kind}/{n}: no FAILED_MEMBER marker")


# ---------------------------------------------------------------- process group
def own_group():
    """Make this process a process-group leader (setsid launches already are), so killpg reaches every member stage
    and tomoDD run it starts and nothing else."""
    if os.getpgid(0) != os.getpid():
        try:
            os.setpgid(0, 0)
        except OSError as ex:                 # e.g. a session leader of another group: cannot happen with setsid
            log(f"cannot become a process-group leader ({ex}); children will be killed one by one on errors")


def _on_signal(signum, frame):
    raise SystemExit(128 + signum)


def group_members(pg):
    """pids of the other live processes of process group pg (/proc)."""
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) == os.getpid():
            continue
        try:
            st = (d / "stat").read_text()
        except OSError:
            continue
        f = st[st.rindex(")") + 2:].split()            # fields after the command name: state ppid pgrp ...
        if int(f[2]) == pg and f[0] != "Z":
            out.append(int(d.name))
    return out


def kill_group(why):
    """Kill every other process of this process group (member stages and their tomoDD runs): SIGTERM, then SIGKILL
    after 10 s for any left. This process exits by itself afterwards (its exit code is kept)."""
    pg = os.getpgid(0)
    if pg != os.getpid():
        log(f"not a group leader: not killing group {pg} ({why})")
        return
    left = group_members(pg)
    if not left:
        return
    log(f"killing process group {pg} ({len(left)} processes): {why}")
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    os.killpg(pg, signal.SIGTERM)
    for _ in range(20):
        time.sleep(0.5)
        left = group_members(pg)
        if not left:
            break
    for q in left:
        try:
            os.kill(q, signal.SIGKILL)
        except ProcessLookupError:
            pass
    log(f"process group {pg}: {len(group_members(pg))} processes left")


def guarded(fn, *args):
    own_group()
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    try:
        return fn(*args)
    except SystemExit as ex:
        if ex.code not in (None, 0):
            kill_group(f"exit {ex.code}")
        raise
    except BaseException:                     # noqa: BLE001  any error: no orphan member stage or tomoDD
        log("chain exception:\n" + traceback.format_exc())
        kill_group("exception")
        raise


def main():
    cmd = sys.argv[1]
    if cmd == "chain":
        guarded(chain)
    elif cmd == "members":
        args = sys.argv[2:]
        mp = None
        if "--max-procs" in args:
            i = args.index("--max-procs")
            mp = int(args[i + 1])
            del args[i:i + 2]
        guarded(run_members, args[0], args[1:], mp)
    elif cmd == "clear-failed":
        clear_failed(sys.argv[2], sys.argv[3:])
    else:
        status()


if __name__ == "__main__":
    main()

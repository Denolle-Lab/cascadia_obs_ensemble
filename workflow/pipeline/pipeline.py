#!/usr/bin/env python3
"""End-to-end driver of the Cascadia OBS analysis (D-B110): one documented chain, rerunnable for a new version.

The driver does not hold science code. Each stage calls the existing scripts with the arguments of one version
configuration (workflow/pipeline/configs/<version>.yaml), in the dependency order below, and keeps a provenance record.

    python workflow/pipeline/pipeline.py CONFIG --list                 # stages, kinds, status
    python workflow/pipeline/pipeline.py CONFIG --dry-run              # every command, every input checked
    python workflow/pipeline/pipeline.py CONFIG --from mc_b --to plate_boundaries --out-root DIR
    python workflow/pipeline/pipeline.py CONFIG --only clustering --force clustering

Stage order (D-B110; clustering, swarm statistics and b need the magnitudes and Mc, so they follow the magnitudes):
  picks           pick timing: v4.1 picks, re-timed v4.1t (D-B072, D-B075), clock-corrected v4.1tc (D-B090)    check
  reloc_start     relocation in the start model (Track A, NonLinLoc, static terms)                              run
  tomo_dataset    tomography data set (B1 selection, B1b balance, re-timed copy)                                run
  tomo_inversion  near-source tomography: M0, tiles, merge, product nodes (tomoDD-SE)                            run
  tomo_register   model registration (append-only registry; the PI's session)                                  gated
  tomo_ensemble   Bayesian (RTO) ensemble of the model and its synthetic calibration                            run
  ens_register    ensemble statistics registration                                                              gated
  cc_dt           waveform cross-correlation differential times (coherence partners)                            run
  event_flags     Mount St Helens analog test-signal flags (event_flag, D-B108)                                 run
  relocation      relocation in the new model: grids, NLL-SSST, coherence, deep and edge passes, catalog        run
  magnitudes      Route A magnitudes on the catalog's own hypocentres (D-B105)                                  run
  reloc_acceptance acceptance: literature depths, held-out coherence scores, above-surface check               run
  reloc_publish   commit the relocation reports to the relocation branch                                       gated
  catalog_bundle  catalog version bundle, registry row, zip (catalog_version.py)                               gated
  mc_b            Mc maps and b per region on the catalog's own hypocentres and magnitudes                       run
  clustering      nearest-neighbour clustering (Zaliapin & Ben-Zion)                                           run
  swarms          swarm scan and swarm statistics                                                               run
  faults          fault surfaces v1 refit on the catalog, joined to the clusters                                run
  structure       velocity structure against seismicity near the faults (model + ensemble)                      run
  validation      validation against existing networks (ComCat) and published local catalogs                    run
  plate_boundaries plate-boundary science: seismicity-style metrics per segment, figures, page                 run
  atlas           3D atlas build and headless check                                                             gated
  atlas_deploy    atlas to gh-pages                                                                             manual
  zenodo_upload   upload to the private Zenodo draft                                                            manual

Kinds. run: the driver runs the commands when the stage is not done. check: never runs anything; the inputs (with
pinned md5) must exist. gated: runs only with --run-gated <stage> after the PI's go; otherwise the driver stops there
if the stage is not done. manual: never run by the driver (printed only).

Done and resume. A stage is done when all its `outputs` exist and every `done` check passes (lines in a chain's own
state file, a key of a JSON state file). Done stages are skipped, so the same command resumes the chain; the
scripts resume inside a stage through their own state (b4_v6_relocate.state, ensemble chain.state, cached steps).
--force <stage> runs a done stage again (its scripts may still skip their own cached steps).

Provenance. After each stage (also for a done stage seen for the first time) the driver writes
<state_dir>/provenance/<stage>.json and appends it to <state_dir>/provenance.jsonl: driver git commit and dirty flag,
the git blob of every code file (and whether it equals HEAD), md5 / size / mtime of every input and output file,
the commands, environment, exit codes, log file and start / end time. Logs: <state_dir>/logs/<stage>.log.

Branches. A stage may list `requires` (branches that must be merged into the checkout). --dry-run reports them as
PENDING MERGE; a run refuses such a stage unless --allow-unmerged.

Resources: the driver runs at nice 10 (os.nice); per-stage process counts come from the config variables (nproc).
Run long stages detached:  setsid nohup python workflow/pipeline/pipeline.py ... > LOG 2>&1 < /dev/null &
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

STAGES = [
    ("picks", "Pick timing (v4.1, v4.1t, v4.1tc)"),
    ("reloc_start", "Relocation in the start model (Track A)"),
    ("tomo_dataset", "Tomography data set (B1, B1b, re-timed copy)"),
    ("tomo_inversion", "Near-source tomography (M0, tiles, merge, product)"),
    ("tomo_register", "Model registration"),
    ("tomo_ensemble", "Bayesian ensemble and calibration"),
    ("ens_register", "Ensemble registration"),
    ("cc_dt", "Waveform CC differential times"),
    ("event_flags", "Mount St Helens analog-signal flags"),
    ("relocation", "Relocation in the new model (NLL-SSST + coherence)"),
    ("magnitudes", "Magnitudes on the catalog's own hypocentres"),
    ("reloc_acceptance", "Relocation acceptance"),
    ("reloc_publish", "Relocation reports to git"),
    ("catalog_bundle", "Catalog version bundle"),
    ("mc_b", "Mc and b per region"),
    ("clustering", "Clustering"),
    ("swarms", "Swarm statistics"),
    ("faults", "Fault fitting"),
    ("structure", "Structure versus seismicity"),
    ("validation", "Validation against existing networks"),
    ("plate_boundaries", "Plate-boundary science"),
    ("atlas", "3D atlas build and check"),
    ("atlas_deploy", "Atlas deploy (gh-pages)"),
    ("zenodo_upload", "Zenodo upload"),
]
ORDER = [s for s, _ in STAGES]
KINDS = ("run", "check", "gated", "manual")
VAR = re.compile(r"(?<!\$)\{([A-Za-z_][A-Za-z0-9_]*)\}")


# ------------------------------------------------------------------ helpers
def now() -> dict:
    t = datetime.now(timezone.utc)
    return dict(utc=t.isoformat(timespec="seconds"), local=t.astimezone().isoformat(timespec="seconds"))


def git(*a, cwd=REPO) -> str:
    r = subprocess.run(["git", "-C", str(cwd), *a], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def md5(p: Path, cache: dict | None = None) -> str:
    st = p.stat()
    key = f"{p}|{st.st_size}|{st.st_mtime_ns}"
    if cache is not None and key in cache:
        return cache[key]
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 24)
            if not b:
                break
            h.update(b)
    if cache is not None:
        cache[key] = h.hexdigest()
    return h.hexdigest()


def present(p: Path) -> bool:
    """A file with content, or a folder with at least one entry."""
    if p.is_dir():
        return any(p.iterdir())
    return p.is_file() and p.stat().st_size > 0


def describe(p: Path, cache: dict | None, max_mb: float) -> dict:
    if not p.exists():
        return dict(path=str(p), exists=False)
    if p.is_dir():
        return dict(path=str(p), dir=True, entries=sum(1 for _ in p.iterdir()))
    st = p.stat()
    d = dict(path=str(p), size=st.st_size, mtime=datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"))
    if st.st_size <= max_mb * 1e6:
        d["md5"] = md5(p, cache)
    return d


# ------------------------------------------------------------------ configuration
class Config:
    def __init__(self, path: Path, overrides: dict, scratch: bool = False):
        self.path = path.resolve()
        raw = yaml.safe_load(self.path.read_text())
        self.raw = raw
        self.version = str(raw["version"])
        v = {"repo": str(REPO), "version": self.version, "config": str(self.path)}
        v.update({k: str(x) for k, x in (raw.get("vars") or {}).items()})
        if scratch:   # --out-root: products that production keeps outside out_root move under it (scratch_vars)
            v.update({k: str(x) for k, x in (raw.get("scratch_vars") or {}).items()})
        v.update(overrides)
        v.setdefault("state_dir", "{out_root}/_pipeline")
        self.vars = self._resolve(v)
        # gitignored data a fresh worktree lacks (data/slab2, ...): repo path -> source; linked before a run
        self.links = {k: self.fill(v) for k, v in (raw.get("repo_links") or {}).items()}
        self.env = {k: str(self.fill(str(v))) for k, v in (raw.get("env") or {}).items()}
        self.stages = []
        seen = []
        for s in raw["stages"]:
            sid = s["id"]
            if sid not in ORDER:
                raise SystemExit(f"{self.path}: unknown stage {sid} (known: {', '.join(ORDER)})")
            if s.get("kind", "run") not in KINDS:
                raise SystemExit(f"{self.path}: stage {sid}: kind {s.get('kind')} not in {KINDS}")
            seen.append(sid)
            self.stages.append(Stage(s, self))
        if seen != [x for x in ORDER if x in seen]:
            raise SystemExit(f"{self.path}: stages out of the D-B110 order: {seen}")
        missing = [x for x in ORDER if x not in seen]
        if missing:
            raise SystemExit(f"{self.path}: stages missing (give each one, with skip: <reason> if it does not apply): "
                             f"{missing}")

    @staticmethod
    def _resolve(v: dict) -> dict:
        for _ in range(20):
            changed = False
            for k, x in v.items():
                y = VAR.sub(lambda m: v[m.group(1)] if m.group(1) in v else m.group(0), x)
                if y != x:
                    v[k], changed = y, True
            if not changed:
                break
        for k, x in v.items():
            bad = [m.group(1) for m in VAR.finditer(x) if m.group(1) not in v]
            if bad:
                raise SystemExit(f"variable {k}: unknown {bad}")
        return v

    def fill(self, x):
        if isinstance(x, str):
            def sub(m):
                if m.group(1) not in self.vars:
                    raise SystemExit(f"{self.path}: unknown variable {{{m.group(1)}}} in: {x}")
                return self.vars[m.group(1)]
            return VAR.sub(sub, x)
        if isinstance(x, list):
            return [self.fill(y) for y in x]
        if isinstance(x, dict):
            return {k: self.fill(y) for k, y in x.items()}
        return x


class Stage:
    def __init__(self, s: dict, cfg: Config):
        f = cfg.fill
        self.id = s["id"]
        self.title = dict(STAGES)[self.id]
        self.kind = s.get("kind", "run")
        self.skip = s.get("skip")
        self.decisions = s.get("decisions", [])
        self.code = [Path(c) for c in f(s.get("code", []))]
        self.requires = s.get("requires", [])
        self.inputs = []
        for x in f(s.get("inputs", [])):
            self.inputs.append(dict(path=Path(x["path"]), md5=x.get("md5")) if isinstance(x, dict)
                               else dict(path=Path(x), md5=None))
        self.outputs = [Path(o) for o in f(s.get("outputs", []))]
        self.done = f(s.get("done", []))
        self.cwd = Path(f(s.get("cwd", "{repo}")))
        self.env = {**cfg.env, **{k: str(v) for k, v in f(s.get("env", {})).items()}}
        self.env_own = {k: str(v) for k, v in f(s.get("env", {})).items()}
        self.commands = f(s.get("commands", []))
        self.runtime = s.get("runtime", "")
        self.resources = s.get("resources", "")
        self.notes = f(s.get("notes", ""))

    # done checks -----------------------------------------------------------
    def status(self) -> tuple[bool | None, list[str]]:
        """(done, reasons). None = the driver cannot tell (no outputs, no done checks)."""
        why, ok = [], True
        if self.skip:
            return True, [f"skipped by the config: {self.skip}"]
        for o in self.outputs:
            if not present(o):
                ok = False
                why.append(f"missing output {o}")
        for d in self.done:
            if "state" in d:
                p = Path(d["state"])
                have = set(p.read_text().split()) if p.exists() else set()
                lack = [x for x in d["lines"] if x not in have]
                if lack:
                    ok = False
                    why.append(f"{p}: not yet {' '.join(lack)}")
            elif "json" in d:
                p = Path(d["json"])
                try:
                    val = json.loads(p.read_text()).get(d["key"])
                except (OSError, json.JSONDecodeError):
                    val = None
                if val != d.get("value", True):
                    ok = False
                    why.append(f"{p}: {d['key']} is {val!r}, not {d.get('value', True)!r}")
            elif "md5" in d:
                p = Path(d["path"])
                if not p.exists() or md5(p) != d["md5"]:
                    ok = False
                    why.append(f"{p}: md5 is not {d['md5']}")
        if not self.outputs and not self.done:
            return None, ["no outputs or done checks"]
        return ok, why

    def code_state(self, head_tree: set) -> list[tuple[str, str]]:
        out = []
        for c in self.code:
            rel = c.as_posix()
            if rel in head_tree:
                out.append((rel, "ok"))
            elif (REPO / rel).exists():
                out.append((rel, "uncommitted (present in the working tree only)"))
            else:
                where = [b["branch"] for b in self.requires
                         if subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"origin/{b['branch']}:{rel}"],
                                           capture_output=True).returncode == 0]
                out.append((rel, f"PENDING MERGE (on origin/{where[0]})" if where else "MISSING"))
        return out

    def merges(self) -> list[tuple[dict, bool]]:
        res = []
        for b in self.requires:
            r = subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", f"origin/{b['branch']}", "HEAD"],
                               capture_output=True)
            res.append((b, r.returncode == 0))
        return res


# ------------------------------------------------------------------ driver
class Driver:
    def __init__(self, cfg: Config, a):
        self.cfg, self.a = cfg, a
        self.state = Path(cfg.vars["state_dir"])
        self.head = git("rev-parse", "HEAD")
        self.tree = set(git("ls-tree", "-r", "--name-only", "HEAD").splitlines())
        self.cache: dict = {}
        self.cache_file = self.state / "md5_cache.json"

    def links(self, make: bool) -> list[str]:
        """Gitignored data inputs of the checkout (config repo_links): link them from their source when absent."""
        bad = []
        for rel, src in self.cfg.links.items():
            dst = REPO / rel
            if dst.exists():
                continue
            if not Path(src).exists():
                bad.append(f"repo_links {rel}: source {src} missing")
            elif make:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.symlink_to(src)
                print(f"linked {dst} -> {src}")
            else:
                print(f"    repo link {rel} -> {src}: absent, made by the run")
        return bad

    def select(self) -> list[Stage]:
        ids = ORDER
        if self.a.only:
            want = self.a.only.split(",")
            bad = [x for x in want if x not in ORDER]
            if bad:
                raise SystemExit(f"--only: unknown stages {bad}")
            ids = [x for x in ORDER if x in want]
        else:
            i0 = ORDER.index(self.a.from_) if self.a.from_ else 0
            i1 = ORDER.index(self.a.to) if self.a.to else len(ORDER) - 1
            ids = ORDER[i0:i1 + 1]
        return [s for s in self.cfg.stages if s.id in ids]

    # listing ---------------------------------------------------------------
    def listing(self):
        print(f"configuration {self.cfg.path} (version {self.cfg.version}); state {self.state}")
        print(f"{'#':>2} {'stage':17} {'kind':7} {'status':9} {'runtime':28} decisions")
        for i, s in enumerate(self.cfg.stages):
            done, _ = s.status()
            st = "skip" if s.skip else {True: "done", False: "pending", None: "?"}[done]
            print(f"{i:>2} {s.id:17} {s.kind:7} {st:9} {s.runtime[:28]:28} {', '.join(s.decisions)}")

    # dry run ---------------------------------------------------------------
    def dry_run(self, stages: list[Stage]) -> int:
        produced: dict[str, str] = {}
        problems, pending = [], []
        print(f"DRY RUN  version {self.cfg.version}  config {self.cfg.path}")
        print(f"repo {REPO} at {self.head[:10]}{' (dirty)' if git('status', '--porcelain') else ''}; "
              f"state {self.state}; out_root {self.cfg.vars.get('out_root')}; {now()['local']}")
        print("environment of every command: " + " ".join(f"{k}={v}" for k, v in self.cfg.env.items()))
        problems += self.links(make=False)
        for s in stages:
            done, why = s.status()
            st = "SKIPPED (config)" if s.skip else {True: "DONE", False: "TO RUN" if s.kind == "run" else
                                                    f"NOT DONE ({s.kind})", None: "UNKNOWN"}[done]
            print(f"\n=== [{ORDER.index(s.id):>2}] {s.id}: {s.title}  [{s.kind}]  {st}")
            if s.decisions:
                print(f"    decisions: {', '.join(s.decisions)}")
            if s.runtime or s.resources:
                print(f"    runtime: {s.runtime}; resources: {s.resources}")
            if s.skip:
                continue
            for w in why[:6]:
                print(f"    status: {w}")
            for b, ok in s.merges():
                tag = "merged" if ok else "PENDING MERGE"
                print(f"    requires {b['branch']} ({b.get('pr', 'no PR')}): {tag}")
                if not ok:
                    pending.append(f"{s.id}: {b['branch']} ({b.get('pr', 'no PR')})")
            for rel, state in s.code_state(self.tree):
                if state != "ok":
                    print(f"    code {rel}: {state}")
                    if state.startswith("PENDING"):
                        pending.append(f"{s.id}: code {rel} {state}")
                    elif state == "MISSING":
                        problems.append(f"{s.id}: code {rel} missing")
            nok = 0
            for x in s.inputs:
                p = x["path"]
                if present(p):
                    if x["md5"]:
                        h = md5(p, self.cache)
                        if h != x["md5"]:
                            print(f"    input {p}: md5 {h} is NOT the pinned {x['md5']}")
                            problems.append(f"{s.id}: input {p} md5 {h} != {x['md5']}")
                            continue
                        print(f"    input {p}: md5 {h} ok")
                    nok += 1
                elif str(p) in produced:
                    print(f"    input {p}: made by stage {produced[str(p)]}")
                    nok += 1
                else:
                    print(f"    input {p}: MISSING")
                    problems.append(f"{s.id}: input {p} missing")
            print(f"    inputs: {nok}/{len(s.inputs)} resolve; outputs: "
                  f"{sum(present(o) for o in s.outputs)}/{len(s.outputs)} present")
            if s.kind == "run" and not done:
                for o in s.outputs:
                    produced[str(o)] = s.id
                for o in s.outputs:          # files inside an output folder count as produced too
                    for x in stages:
                        for i in x.inputs:
                            if str(i["path"]).startswith(str(o) + "/"):
                                produced[str(i["path"])] = s.id
            if s.env_own:
                print("    env: " + " ".join(f"{k}={v}" for k, v in s.env_own.items()))
            print(f"    cd {s.cwd}")
            for c in s.commands:
                print("    $ " + c.strip().replace("\n", "\n      "))
            if s.notes:
                print("    note: " + " ".join(s.notes.split()))
        print("\n=== summary")
        for s in stages:
            done, _ = s.status()
            st = "skip" if s.skip else {True: "done", False: "pending", None: "?"}[done]
            print(f"  {s.id:17} {s.kind:7} {st}")
        if pending:
            print("pending merges (main is not self-sufficient until these merge):")
            for p in dict.fromkeys(pending):
                print(f"  {p}")
        if problems:
            print("PROBLEMS:")
            for p in problems:
                print(f"  {p}")
            return 1
        print("every input and code path resolves" + (" (some only on pending branches)" if pending else ""))
        return 0

    # provenance ------------------------------------------------------------
    def provenance(self, s: Stage, rec: dict):
        code = {}
        for c in s.code:
            p = REPO / c
            blob = git("hash-object", str(p)) if p.exists() else None
            code[c.as_posix()] = dict(blob=blob, head_blob=git("rev-parse", f"HEAD:{c.as_posix()}") or None)
            code[c.as_posix()]["same_as_head"] = blob is not None and blob == code[c.as_posix()]["head_blob"]
        mx = float(self.cfg.vars.get("md5_max_mb", 4000))
        rec.update(stage=s.id, kind=s.kind, version=self.cfg.version, config=str(self.cfg.path),
                   config_md5=md5(self.cfg.path), driver_commit=self.head, dirty=bool(git("status", "--porcelain")),
                   host=socket.gethostname(), decisions=s.decisions, code=code,
                   inputs=[describe(x["path"], self.cache, mx) for x in s.inputs],
                   outputs=[describe(o, self.cache, mx) for o in s.outputs],
                   cwd=str(s.cwd), env=s.env, commands=s.commands)
        d = self.state / "provenance"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{s.id}.json").write_text(json.dumps(rec, indent=1, default=str))
        with open(self.state / "provenance.jsonl", "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        self.cache_file.write_text(json.dumps(self.cache))

    # run -------------------------------------------------------------------
    def run(self, stages: list[Stage]) -> int:
        self.state.mkdir(parents=True, exist_ok=True)
        (self.state / "logs").mkdir(exist_ok=True)
        try:
            self.cache = json.loads(self.cache_file.read_text())
        except (OSError, json.JSONDecodeError):
            self.cache = {}
        lock = open(self.state / "pipeline.lock", "w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise SystemExit(f"another driver holds {self.state / 'pipeline.lock'}")
        try:
            os.nice(10)
        except OSError:
            pass
        force = set((self.a.force or "").split(",")) - {""}
        gated = set((self.a.run_gated or "").split(",")) - {""}
        say = lambda m: print(f"{datetime.now().strftime('%F %T')} [pipeline {self.cfg.version}] {m}", flush=True)
        say(f"start: stages {[s.id for s in stages]}, repo {self.head[:10]}, state {self.state}")
        bad = self.links(make=True)
        if bad:
            say(f"missing data: {bad}")
            return 5
        for s in stages:
            done, why = s.status()
            if s.skip:
                say(f"{s.id}: skipped ({s.skip})")
                continue
            if done and s.id not in force:
                say(f"{s.id}: done, skipped")
                if not (self.state / "provenance" / f"{s.id}.json").exists():
                    self.provenance(s, dict(event="adopted (done before this driver ran it)", **{"time": now()}))
                continue
            if s.kind == "manual":
                say(f"{s.id}: manual step, not run by the driver:")
                for c in s.commands:
                    print("    " + c.strip().replace("\n", "\n    "))
                continue
            if s.kind == "check":
                say(f"{s.id}: NOT READY: {'; '.join(why)} (check stage: the driver does not build it)")
                return 2
            if s.kind == "gated" and s.id not in gated:
                say(f"{s.id}: gated stage not done ({'; '.join(why[:3])}). After the PI's go: --run-gated {s.id}, "
                    "or by hand:")
                for c in s.commands:
                    print("    " + c.strip().replace("\n", "\n    "))
                return 3
            unmerged = [b for b, ok in s.merges() if not ok]
            if unmerged and not self.a.allow_unmerged:
                say(f"{s.id}: needs unmerged branches {[b['branch'] for b in unmerged]} (--allow-unmerged to run anyway)")
                return 4
            bad = [str(x["path"]) for x in s.inputs if not present(x["path"])]
            bad += [f"{x['path']} (md5 != {x['md5']})" for x in s.inputs
                    if x["md5"] and present(x["path"]) and md5(x["path"], self.cache) != x["md5"]]
            bad += [f"code {c}" for c in s.code if not (REPO / c).exists()]
            if bad:
                say(f"{s.id}: missing inputs: {bad}")
                return 5
            log = self.state / "logs" / f"{s.id}.log"
            env = {**os.environ, **s.env}
            if self.cfg.vars.get("manifest_log"):
                env["CASCADIA_MANIFEST_LOG"] = self.cfg.vars["manifest_log"]
            rec = dict(event="run", start=now(), log=str(log), exits=[])
            say(f"{s.id}: running {len(s.commands)} command(s), log {log}")
            t0 = time.time()
            with open(log, "a") as L:
                L.write(f"\n#### {now()['local']} stage {s.id} (driver {self.head[:10]})\n")
                for c in s.commands:
                    L.write(f"$ {c}\n")
                    L.flush()
                    r = subprocess.run(["bash", "-c", "set -o pipefail; " + c], cwd=s.cwd, env=env, stdout=L,
                                       stderr=subprocess.STDOUT)
                    rec["exits"].append(r.returncode)
                    if r.returncode != 0:
                        rec.update(end=now(), seconds=round(time.time() - t0))
                        self.provenance(s, rec)
                        say(f"{s.id}: FAILED (exit {r.returncode}) in: {c.strip()[:160]}; log {log}")
                        return 1
            rec.update(end=now(), seconds=round(time.time() - t0))
            done, why = s.status()
            if done is False:
                self.provenance(s, rec)
                say(f"{s.id}: commands finished but the stage is not done: {'; '.join(why[:5])}")
                return 1
            self.provenance(s, rec)
            say(f"{s.id}: done in {rec['seconds'] / 60:.1f} min")
        say("finished")
        return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--list", action="store_true", help="stage table with status")
    ap.add_argument("--dry-run", action="store_true", help="print every command and check its inputs; run nothing")
    ap.add_argument("--from", dest="from_", choices=ORDER, help="first stage")
    ap.add_argument("--to", choices=ORDER, help="last stage")
    ap.add_argument("--only", help="comma-separated stages")
    ap.add_argument("--force", help="comma-separated stages to run again although done")
    ap.add_argument("--run-gated", help="comma-separated gated stages to run (after the PI's go)")
    ap.add_argument("--allow-unmerged", action="store_true", help="run stages whose required branches are not merged")
    ap.add_argument("--out-root", help="analysis output root (overrides the config's out_root; e.g. a scratch root)")
    ap.add_argument("--var", action="append", default=[], help="override a config variable: NAME=VALUE")
    a = ap.parse_args()
    ov = dict(v.split("=", 1) for v in a.var)
    if a.out_root:
        ov["out_root"] = str(Path(a.out_root).resolve())
        # scratch runs keep the Track B manifest records (manifest.write_manifest) in their own log, not in the shared
        # /wd1/mdenolle_data/tomo_v1/manifests.jsonl
        ov.setdefault("manifest_log", "{state_dir}/manifests.jsonl")
    cfg = Config(a.config, ov, scratch=bool(a.out_root))
    d = Driver(cfg, a)
    if a.list:
        d.listing()
        return 0
    stages = d.select()
    if a.dry_run:
        return d.dry_run(stages)
    return d.run(stages)


if __name__ == "__main__":
    sys.exit(main())

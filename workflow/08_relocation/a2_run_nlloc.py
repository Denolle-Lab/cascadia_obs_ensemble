"""Track A / B4, step A2: run NLLoc tile by tile (obs from a2_make_obs.py), in parallel (v5.1 runner).

Boxes. Each tile's search box (LOCGRID) is the 25 km tile +- 75 km (tiles.csv of the obs folder), z' -5..80 km, 1 km;
the oct-tree starts with ~10 km cells. Tiles with more than CHUNK events are split so they run in parallel.

Station reach (round 2, investigation T, 2026-11-24). NLLoc uses an arrival only if the search box lies inside the time
grid it opens for that station-phase (NLLocLib.c GetObservations, IsGridInside). The fine grids are +-250 km around
the station, so with the tile +- 75 km box a station counted only within ~150-175 km of the event (B4 iter1 v0.3.0
dropped 169,011 of 2,503,189 arrivals). --time-wide NAME lists whole-lattice 2 km grids (b4_wide_grids.py) as a
second root in LOCFILES, after the fine root: per arrival NLLoc takes the first root whose grid file opens and contains
the box, so the fine grids serve every arrival they can and the wide grids the others. The box is unchanged.

Arrival accounting (interface I2). After the jobs, every event of every workdir of the run is checked: the arrival
lines of its events.obs block (n_obs) against NLLoc's "N observations read, M will be used" line and its logged box
rejections (nlloc.log, CONTROL 1), and each arrival is given the root that served it by the header rule (the first
LOCFILES root whose grid contains the box; what NLLoc does once every grid file it needs exists, which the guards
below enforce). The one documented drop is the box rejection: no listed grid contains the box (the old reach without
--time-wide; none expected with it), logged by NLLoc and predicted identically from the grid headers. Any other drop
(a missing grid file, the LOCMETH distance cut, a duplicate or unreadable pick, a log that does not match the obs
file, a job that did not finish) is undocumented: it is listed in arrival_problems.csv and the runner exits with
status 3. Outputs in the run folder: arrivals_used.csv (ID, n_obs, n_used, n_fine, n_wide, pass = 'main' or 'deep'),
arrival_drops.csv (one row per documented box rejection).

Guards (the runner refuses to start, exit status 2):
  (a) a fine grid (header and complete buffer) is missing for a station-phase that an obs block references: NLLoc
      would silently fall back to the wide root, or drop the arrival without a message;
  (b) with --time-wide: a referenced station-phase whose fine grid does not contain its job's box has no complete wide
      grid containing that box, or the wide folder has no b4_wide_grids manifest naming the fine folder;
  (c) resume: the run folder's run_config.json differs from this invocation (grid roots, obs folder, template, LOCGAU,
      LOCGAU2, topography file and contents, station-terms file and contents, NLLoc binary, chunking, pass), or the
      folder holds workdirs but no run_config.json, or a workdir's nlloc.in or events.obs differs from what this
      invocation would write. A workdir with loc.sum.grid0.loc.hyp and the same nlloc.in is skipped (resume).

Deep pass (D-B067). --deep-from MAIN_RUN reruns, into this run, the events of MAIN_RUN that were REJECTED at the bottom
face (NLLoc boundary code 31) or LOCATED with maximum-likelihood z' >= DEEP_Z, unless far_only (no used arrival served
by a fine grid, from MAIN_RUN's arrivals_used.csv: such events are flagged, not pushed deeper). Same x/y box as in
MAIN_RUN (its nlloc.in), LOCGRID z' -5..120 km (NZ_DEEP nodes, LOCSEARCH initial nz NZ_OCT_DEEP); no fine grid contains
that box, so every arrival is served by the wide grid, which must reach z' >= 120 km (n_fine = 0 in the deep run's
arrivals_used.csv). The options must equal MAIN_RUN's (checked against its run_config.json). Selection:
deep_selection.csv; arrivals_used.csv has pass = 'deep'.

Usage: python a2_run_nlloc.py RUN [--time NAME] [--time-wide NAME] [--obs-dir DIR] [--stat FILE] [--tiles a,b]
                              [--max-events N] [--nproc 100] [--template FILE] [--locgau "0.1 0.0"]
                              [--locgau2 "0.02 0.05 2.0" | off] [--topo FILE] [--max-dist KM] [--deep-from MAIN_RUN]
  RUN, MAIN_RUN: run name under reloc_v5/runs, or an absolute run folder. --time, --time-wide: grid folder under
  reloc_v5, or absolute. --obs-dir: folder of a2_make_obs.py (<tile>.obs, tiles.csv; default reloc_v5/obs_tiles).
  --stat: NLL LOCDELAY station-delay file (a2_station_terms.py) to include. --template: control template (default
  nlloc_template.in next to this script). Without --time-wide the old reach applies (a warning is printed).
Exit status: 0 ok, 1 an NLLoc job failed, 2 refused to start, 3 undocumented arrival drops.
"""
import argparse
import functools
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

B = Path("/wd1/mdenolle_data/reloc_v5")
NLL = B / "src/NonLinLoc/src/bin/NLLoc"
TEMPLATE = Path(__file__).with_name("nlloc_template.in")
CHUNK = 500
Z0, NZ, NZ_DEEP = -5.0, 86, 126          # LOCGRID z' -5..80 km (main pass), -5..120 km (deep pass), 1 km nodes
NZ_OCT, NZ_OCT_DEEP = 9, 13              # LOCSEARCH initial oct-tree cells in z (~9.4 km)
MAX_DIST = 9999.0                        # km, LOCMETH maxDistStaGrid (no cut)
LOCGAU = "0.1 0.0"                       # likelihood and floor of the v0.3.1 production runs (I4 defaults)
LOCGAU2 = "0.02 0.05 2.0"
TOPO = "/wd1/mdenolle_data/reloc_v5/topo/topo_15s_cons_eft.bin"
DEEP_Z, BOTTOM = 78.0, 31                # deep-pass trigger: ML z' (km); NLLoc isOnGridBoundary code of the bottom face
BOX_REJECT = re.compile(r"not contained inside arrival time grid, rejecting grid for: (\S+) (\S+)")
READ_USED = re.compile(r"\.\.\. (\d+) observations read, (\d+) will be used")
NEXT_EVENT = "Reading next set of observations"


def refuse(msg):
    print(f"REFUSED: {msg}", flush=True)
    sys.exit(2)


def grid_dir(name):
    return Path(name) if str(name).startswith("/") else B / name


def run_dir(name):
    return Path(name) if str(name).startswith("/") else B / "runs" / name


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def split_obs(text, max_events=None):
    blocks = [b for b in text.split("\n\n") if b.strip()]
    if max_events:
        blocks = blocks[:max_events]
    return [blocks[i:i + CHUNK] for i in range(0, len(blocks), CHUNK)]


def block_arrivals(block):
    """(event ID, [(phase, station), ...]) of one NLLOC_OBS event block written by a2_make_obs.py."""
    lines = [l for l in block.strip("\n").split("\n") if l.strip()]
    if not lines[0].startswith("PUBLIC_ID v4_"):
        raise ValueError(f"obs block does not start with PUBLIC_ID v4_<ID>: {lines[0]!r}")
    return int(lines[0].split()[1].split("_")[1]), [(l.split()[4], l.split()[0]) for l in lines[1:]]


@functools.lru_cache(maxsize=None)
def headers(root):
    """{(phase, station): (x0, x1, y0, y1, z0, z1, complete)} of the time grids <root>.<ph>.<sta>.time.hdr;
    complete = the .buf exists with nx ny nz floats."""
    root = Path(root)
    out = {}
    for h in root.parent.glob(f"{root.name}.*.time.hdr"):
        ph, sta = h.name[len(root.name) + 1:-len(".time.hdr")].split(".", 1)
        t = h.read_text().split()
        nx, ny, nz = (int(v) for v in t[:3])
        x0, y0, z0, dx, dy, dz = (float(v) for v in t[3:9])
        buf = h.with_name(h.name[:-3] + "buf")
        ok = buf.exists() and buf.stat().st_size == 4 * nx * ny * nz
        out[(ph, sta)] = (x0, x0 + (nx - 1) * dx, y0, y0 + (ny - 1) * dy, z0, z0 + (nz - 1) * dz, ok)
    return out


def inside(box, g):
    """NLLoc IsGridInside: box (x0, x1, y0, y1, z0, z1) lies inside grid extent g."""
    return not (box[0] < g[0] or box[1] > g[1] or box[2] < g[2] or box[3] > g[3] or box[4] < g[4] or box[5] > g[5])


def serving_root(box, key, roots):
    """index of the LOCFILES root whose grid serves arrival key = (phase, station) in this box (NLLoc's order: the
    first whose grid exists and contains the box), or None (box rejection)."""
    for i, r in enumerate(roots):
        g = headers(r).get(key)
        if g is not None and inside(box, g):
            return i
    return None


def read_control(text):
    """(box, roots) of an nlloc.in: LOCGRID extent (x0, x1, y0, y1, z0, z1) and the LOCFILES time-grid roots."""
    g = re.search(r"^LOCGRID +(\d+) +(\d+) +(\d+) +(\S+) +(\S+) +(\S+) +(\S+) +(\S+) +(\S+)", text, re.M)
    nx, ny, nz = (int(g.group(i)) for i in (1, 2, 3))
    x0, y0, z0, dx, dy, dz = (float(g.group(i)) for i in range(4, 10))
    roots = re.search(r"^LOCFILES +\S+ +\S+ +(\S+) +\S+", text, re.M).group(1).split(",")
    return (x0, x0 + (nx - 1) * dx, y0, y0 + (ny - 1) * dy, z0, z0 + (nz - 1) * dz), tuple(roots)


def sub1(pattern, repl, text):
    """replace the one line matching pattern (re.M) with repl (taken literally)."""
    out, n = re.subn(pattern, lambda m: repl, text, flags=re.M)
    if n != 1:
        raise SystemExit(f"template: {n} lines match {pattern!r}, expected 1")
    return out


def control(box, obs, out, stat, opt, deep=False):
    """NLLoc control file of one job; box = (x0, y0, nx, ny): the tile's search box (km, 1 km nodes)."""
    x0, y0, nx, ny = box
    locgrid = (f"LOCGRID {nx} {ny} {NZ_DEEP if deep else NZ} {x0:.1f} {y0:.1f} {Z0:.1f} 1.0 1.0 1.0 "
               "PROB_DENSITY SAVE")
    search = (f"LOCSEARCH OCT {max(nx // 10, 2)} {max(ny // 10, 2)} {NZ_OCT_DEEP if deep else NZ_OCT} "
              "0.02 40000 2000 0 1")
    t = opt.template_text
    t = t.replace("__TIME__", opt.time).replace("__OBS__", str(obs)).replace("__OUT__", str(out))
    t = t.replace("__STATERMS__", f"INCLUDE {stat}" if stat else "")
    t = sub1(r"^LOCFILES .*$", f"LOCFILES {obs} NLLOC_OBS {','.join(opt.roots)} {out}", t)
    m = re.search(r"^(LOCMETH +\S+) +\S+", t, re.M)
    t = sub1(r"^LOCMETH +\S+ +\S+", f"{m.group(1)} {opt.max_dist:.1f}", t)   # 2nd field: maxDistStaGrid, km
    t = sub1(r"^LOCGAU .*$", f"LOCGAU {opt.locgau}", t)
    t = sub1(r"^LOCGAU2 .*$", "LOCGAU2 __OFF__" if opt.locgau2 == "off" else f"LOCGAU2 {opt.locgau2}", t)
    m = re.search(r"^LOCTOPO_SURFACE +\S+( .*)?$", t, re.M)
    t = sub1(r"^LOCTOPO_SURFACE .*$", f"LOCTOPO_SURFACE {opt.topo}{m.group(1) or ''}", t)
    lines = [l for l in t.splitlines() if not l.startswith(("LOCGRID", "LOCSEARCH", "LOCGAU2 __OFF__"))]
    return "\n".join(lines + [locgrid, search]) + "\n"


def run_one(job):
    wd, ctl = job
    (wd / "nlloc.in").write_text(ctl)
    with open(wd / "nlloc.log", "w") as log:
        r = subprocess.run([str(NLL), "nlloc.in"], cwd=wd, stdout=log, stderr=subprocess.STDOUT)
    return wd.name, r.returncode


# ---------------------------------------------------------------------------------------------------------- planning
def options(a):
    template = Path(a.template) if a.template else TEMPLATE
    roots = [f"{grid_dir(a.time)}/cascadia3d"] + ([f"{grid_dir(a.time_wide)}/cascadia3d"] if a.time_wide else [])
    for f in (a.topo, a.topo + ".hdr"):
        if not Path(f).is_file():
            refuse(f"topography file {f} does not exist")
    if a.stat and not Path(a.stat).is_file():
        refuse(f"station-terms file {a.stat} does not exist")
    return argparse.Namespace(time=a.time, roots=tuple(roots), template=template, template_text=template.read_text(),
                              locgau=a.locgau, locgau2=a.locgau2, topo=a.topo, max_dist=a.max_dist)


def run_config(a, opt, obs_dir):
    """what defines a run (guard c); the template enters by its contents, not its path."""
    return {"pass": "deep" if a.deep_from else "main", "time": opt.roots[0],
            "time_wide": opt.roots[1] if len(opt.roots) > 1 else None,
            "obs": str(run_dir(a.deep_from)) if a.deep_from else str(obs_dir.resolve()),
            "template_md5": md5(opt.template), "locgau": opt.locgau, "locgau2": opt.locgau2,
            "topo": opt.topo, "topo_md5": md5(opt.topo), "topo_hdr_md5": md5(opt.topo + ".hdr"),
            "stat": a.stat, "stat_md5": md5(a.stat) if a.stat else None, "max_dist": opt.max_dist,
            "nll": str(NLL), "nll_md5": md5(NLL), "chunk": CHUNK, "max_events": a.max_events,
            "nz": NZ_DEEP if a.deep_from else NZ, "deep_z": DEEP_Z if a.deep_from else None}


def plan_main(a, opt, rd, obs_dir):
    """[(workdir name, nlloc.in text, events.obs text)] of every job of the requested tiles."""
    tiles = pd.read_csv(obs_dir / "tiles.csv")
    if a.tiles:
        want = a.tiles.split(",")
        miss = sorted(set(want) - set(tiles.tile))
        if miss:
            refuse(f"tiles not in {obs_dir}/tiles.csv: {miss}")
        tiles = tiles[tiles.tile.isin(want)]
    specs = []
    for tile in tiles.itertuples():
        nx = int(round(tile.box_x1 - tile.box_x0)) + 1
        ny = int(round(tile.box_y1 - tile.box_y0)) + 1
        chunks = split_obs((obs_dir / f"{tile.tile}.obs").read_text(), a.max_events)
        for c, blocks in enumerate(chunks):
            name = tile.tile if len(chunks) == 1 else f"{tile.tile}_c{c:02d}"
            wd = rd / name
            specs.append((name, control((tile.box_x0, tile.box_y0, nx, ny), wd / "events.obs", wd / "loc", a.stat,
                                        opt), "\n\n".join(blocks) + "\n\n"))
    return specs


def hyp_status(path):
    """{ID: (status, NLLoc boundary code or None, ML z')} of a loc.sum.grid0.loc.hyp file."""
    out, cur = {}, None
    for line in open(path):
        if line.startswith("NLLOC "):
            q = re.findall(r'"([^"]*)"', line)
            b = re.search(r"grid boundary (\d+)", q[2] if len(q) > 2 else "")
            cur = {"status": q[1], "boundary": int(b.group(1)) if b else None, "z": None, "ID": None}
        elif cur is None:
            continue
        elif line.startswith("PUBLIC_ID"):
            cur["ID"] = int(line.split()[1].split("_")[1])
        elif line.startswith("HYPOCENTER"):
            t = line.split()
            cur["z"] = float(t[t.index("z") + 1])
        elif line.startswith("END_NLLOC"):
            out[cur["ID"]] = (cur["status"], cur["boundary"], cur["z"])
            cur = None
    return out


def plan_deep(a, opt, rd, cfg):
    """deep pass (D-B067): ([(workdir name, nlloc.in text, events.obs text)], selection table) of MAIN_RUN."""
    md = run_dir(a.deep_from)
    try:
        mcfg = json.loads((md / "run_config.json").read_text())
        used = pd.read_csv(md / "arrivals_used.csv")
    except FileNotFoundError as e:
        refuse(f"--deep-from {md}: {e.filename} missing (main run not finished by this runner)")
    if mcfg.get("pass") != "main" or set(used["pass"]) != {"main"}:
        refuse(f"--deep-from {md} is not a main-pass run")
    same = ("time", "time_wide", "template_md5", "locgau", "locgau2", "topo", "topo_md5", "topo_hdr_md5", "stat",
            "stat_md5", "max_dist", "nll_md5", "chunk", "max_events")
    diff = {k: (mcfg.get(k), cfg[k]) for k in same if mcfg.get(k) != cfg[k]}
    if diff:
        refuse(f"deep pass options differ from the main run {md} (main, this): {diff}")
    if (md / "arrival_problems.csv").exists():
        refuse(f"main run {md} has undocumented arrival drops (arrival_problems.csv)")
    n_fine = used.set_index("ID").n_fine
    specs, sel = [], []
    for wd in sorted(p for p in md.iterdir() if (p / "events.obs").exists()):
        st = hyp_status(wd / "loc.sum.grid0.loc.hyp")
        box, _ = read_control((wd / "nlloc.in").read_text())
        keep = []
        for b in (b for b in (wd / "events.obs").read_text().split("\n\n") if b.strip()):
            eid = block_arrivals(b)[0]
            status, bnd, z = st.get(eid, ("none", None, None))
            trig = ("rejected_bottom" if status == "REJECTED" and bnd == BOTTOM else
                    "located_deep" if status == "LOCATED" and z >= DEEP_Z else None)
            if trig is None:
                continue
            far_only = bool(n_fine[eid] == 0)
            sel.append(dict(ID=eid, workdir=wd.name, status=status, boundary=bnd, z_ml=z, trigger=trig,
                            n_fine=int(n_fine[eid]), far_only=far_only, selected=not far_only))
            if not far_only:
                keep.append(b)
        if keep:
            d = rd / wd.name
            ctl = control((box[0], box[2], int(round(box[1] - box[0])) + 1, int(round(box[3] - box[2])) + 1),
                          d / "events.obs", d / "loc", a.stat, opt, deep=True)
            specs.append((wd.name, ctl, "\n\n".join(keep) + "\n\n"))
    sel = pd.DataFrame(sel, columns=["ID", "workdir", "status", "boundary", "z_ml", "trigger", "n_fine", "far_only",
                                     "selected"])
    print(f"deep pass from {md}: {len(sel)} triggered ({(sel.trigger == 'rejected_bottom').sum()} rejected at the "
          f"bottom face, {(sel.trigger == 'located_deep').sum()} located at z' >= {DEEP_Z:g} km), "
          f"{int(sel.far_only.sum())} far_only (not rerun), {int(sel.selected.sum())} rerun in {len(specs)} jobs",
          flush=True)
    return specs, sel


# ------------------------------------------------------------------------------------------------------------ guards
def check_wide_manifest(time, time_wide):
    man = sorted(grid_dir(time_wide).glob("b4_wide_grids*.manifest.json"))
    if not man:
        refuse(f"{grid_dir(time_wide)} has no b4_wide_grids manifest: cannot check it was built for {time}")
    built = {Path(str(json.loads(m.read_text()).get("params", {}).get("time_name"))).name for m in man}
    if built != {grid_dir(time).name}:
        refuse(f"{grid_dir(time_wide)} was built from {sorted(built)}, not from {grid_dir(time).name}")


def check_grids(specs, roots):
    """guards (a) and (b); returns the number of arrivals the header rule gives to each root (0 fine, 1 wide) and to
    none (box rejection)."""
    fine = headers(roots[0])
    missing, bad_wide, count = set(), set(), {0: 0, 1: 0, None: 0}
    for name, ctl, text in specs:
        box, _ = read_control(ctl)
        for b in text.split("\n\n"):
            if not b.strip():
                continue
            for k in block_arrivals(b)[1]:
                g = fine.get(k)
                if g is None or not g[6]:
                    missing.add(k)
                    continue
                s = serving_root(box, k, roots)
                if (s is None and len(roots) > 1) or (s == 1 and not headers(roots[1])[k][6]):
                    bad_wide.add((k, name))
                count[s] += 1
    if missing:
        refuse(f"{len(missing)} station-phases referenced by the obs have no complete fine grid in {roots[0]}: "
               f"{sorted(missing)[:10]}")
    if bad_wide:
        refuse(f"{len(bad_wide)} (station-phase, job) pairs need the wide grid but {roots[1]} has none containing the "
               f"box (or its buffer is incomplete): {sorted(bad_wide)[:10]}")
    return count


def check_resume(rd, cfg, specs, deep):
    """guard (c); returns the jobs to run [(workdir, nlloc.in text, events.obs text)]."""
    cp = rd / "run_config.json"
    wds = [p for p in rd.iterdir() if p.is_dir()] if rd.exists() else []
    if cp.exists():
        old = json.loads(cp.read_text())
        diff = {k: (old.get(k), cfg.get(k)) for k in sorted(set(old) | set(cfg)) if old.get(k) != cfg.get(k)}
        if diff:
            refuse(f"{rd} was run with other settings (existing, this invocation): {diff}")
    elif any((p / "nlloc.in").exists() or (p / "events.obs").exists() for p in wds):
        refuse(f"{rd} holds NLLoc workdirs but no run_config.json: cannot check how they were run; use a new run name")
    if deep:
        extra = sorted(p.name for p in wds if p.name not in {s[0] for s in specs})
        if extra:
            refuse(f"{rd} holds workdirs this deep pass does not select: {extra[:10]}")
    conflicts, jobs = [], []
    for name, ctl, text in specs:
        wd = rd / name
        if (wd / "nlloc.in").exists() and (wd / "nlloc.in").read_text() != ctl:
            conflicts.append(f"{name}/nlloc.in")
        if (wd / "events.obs").exists() and (wd / "events.obs").read_text() != text:
            conflicts.append(f"{name}/events.obs")
        if not ((wd / "loc.sum.grid0.loc.hyp").exists() and (wd / "nlloc.in").exists()):
            jobs.append((wd, ctl, text))
    if conflicts:
        refuse(f"{len(conflicts)} files in {rd} differ from what this invocation would write: {conflicts[:10]}")
    return jobs


# -------------------------------------------------------------------------------------------------------- accounting
def account_wd(wd):
    """per-event arrival accounting of one workdir: (rows, drops, problems)."""
    rows, drops = [], []
    if not ((wd / "loc.sum.grid0.loc.hyp").exists() and (wd / "nlloc.log").exists()):
        return rows, drops, [dict(workdir=wd.name, ID=None, problem="NLLoc did not finish (no loc.sum.grid0.loc.hyp)")]
    box, roots = read_control((wd / "nlloc.in").read_text())
    blocks = [b for b in (wd / "events.obs").read_text().split("\n\n") if b.strip()]
    chunks = (wd / "nlloc.log").read_text(errors="replace").split(NEXT_EVENT)[1:]
    if len(chunks) != len(blocks) + 1:
        return rows, drops, [dict(workdir=wd.name, ID=None, problem=f"nlloc.log has {len(chunks)} observation sets "
                                                                     f"for {len(blocks)} events")]
    probs = []
    for b, c in zip(blocks, chunks):
        eid, arrs = block_arrivals(b)
        served = [serving_root(box, k, roots) for k in arrs]
        none = sorted(k for k, s in zip(arrs, served) if s is None)
        logged = sorted((ph, sta) for sta, ph in BOX_REJECT.findall(c))
        m = READ_USED.search(c)
        n_read, n_used = (int(m.group(1)), int(m.group(2))) if m else (None, 0)
        rows.append(dict(ID=eid, n_obs=len(arrs), n_used=n_used, n_fine=sum(s == 0 for s in served),
                         n_wide=sum(s == 1 for s in served)))
        left = Counter(logged)
        for ph, sta in none:          # documented only if NLLoc logged it
            drops.append(dict(ID=eid, workdir=wd.name, station=sta, phase=ph,
                              reason="box" if left[(ph, sta)] > 0 else "no grid file (not logged)"))
            left[(ph, sta)] -= 1
        why = []
        if logged != none:
            why.append(f"logged box rejections ({len(logged)}) differ from the header rule ({len(none)})")
        if m is None and len(logged) != len(arrs):
            why.append(f"no 'observations read' line, {len(logged)} logged rejections of {len(arrs)} arrivals")
        if m is not None and n_read != len(arrs):
            why.append(f"NLLoc read {n_read} of {len(arrs)} arrivals (duplicate or unreadable pick)")
        if len(arrs) - n_used != len(none):
            why.append(f"{len(arrs) - n_used - len(none)} arrivals not used without a box rejection "
                       "(missing grid file, LOCMETH distance cut or ignored pick)")
        if why:
            probs.append(dict(workdir=wd.name, ID=eid, problem="; ".join(why)))
    return rows, drops, probs


def account(rd, kind, nproc):
    """arrivals_used.csv (I2), arrival_drops.csv, arrival_problems.csv of a run folder; returns the problem count."""
    wds = sorted(p for p in rd.iterdir() if (p / "events.obs").exists())
    rows, drops, probs = [], [], []
    with ProcessPoolExecutor(max(1, min(nproc, 32))) as ex:
        for r, d, p in ex.map(account_wd, wds, chunksize=8):
            rows += r
            drops += d
            probs += p
    used = pd.DataFrame(rows, columns=["ID", "n_obs", "n_used", "n_fine", "n_wide"]).assign(**{"pass": kind})
    probs += [dict(workdir=None, ID=i, problem="event in more than one workdir")
              for i in used.ID[used.ID.duplicated()].unique()]
    used.sort_values("ID").to_csv(rd / "arrivals_used.csv", index=False)
    drops = pd.DataFrame(drops, columns=["ID", "workdir", "station", "phase", "reason"])
    drops.to_csv(rd / "arrival_drops.csv", index=False)
    pf = rd / "arrival_problems.csv"
    if probs:
        pd.DataFrame(probs, columns=["workdir", "ID", "problem"]).to_csv(pf, index=False)
    elif pf.exists():
        pf.unlink()
    print(f"arrivals ({kind} pass, {len(wds)} workdirs, {len(used)} events): {used.n_obs.sum()} in the obs, "
          f"{used.n_used.sum()} used by NLLoc ({used.n_fine.sum()} fine, {used.n_wide.sum()} wide by the header rule), "
          f"{int((drops.reason == 'box').sum())} box rejections (documented); events without a fine-grid arrival: "
          f"{int((used.n_fine == 0).sum())}", flush=True)
    if probs:
        print(f"ERROR: undocumented arrival drops or log mismatches in {len(probs)} cases: {pf}", flush=True)
    return len(probs)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("--tiles")
    ap.add_argument("--max-events", type=int)
    ap.add_argument("--nproc", type=int, default=100)
    ap.add_argument("--stat")
    ap.add_argument("--time", default="time_vpvs176", help="fine travel-time grid folder under reloc_v5, or absolute")
    ap.add_argument("--time-wide", help="wide grid folder (b4_wide_grids.py) under reloc_v5, or absolute")
    ap.add_argument("--obs-dir", default=str(B / "obs_tiles"), help="a2_make_obs.py output folder")
    ap.add_argument("--template", help="NLLoc control template (default nlloc_template.in)")
    ap.add_argument("--locgau", default=LOCGAU, help=f"LOCGAU values (default '{LOCGAU}')")
    ap.add_argument("--locgau2", default=LOCGAU2, help=f"LOCGAU2 values, or 'off' (default '{LOCGAU2}')")
    ap.add_argument("--topo", default=TOPO, help=f"LOCTOPO_SURFACE file (default {TOPO})")
    ap.add_argument("--max-dist", type=float, default=MAX_DIST, help="LOCMETH maximum station distance, km")
    ap.add_argument("--deep-from", help="deep pass (D-B067) of the events of this main run")
    a = ap.parse_args()
    rd = run_dir(a.run)
    obs_dir = Path(a.obs_dir)
    opt = options(a)
    cfg = run_config(a, opt, obs_dir)
    if a.time_wide:
        check_wide_manifest(a.time, a.time_wide)
    elif a.deep_from:
        refuse("the deep pass needs --time-wide (no fine grid contains the z' -5..120 km box)")
    else:
        print("WARNING: no --time-wide: arrivals whose fine grid does not contain the box are dropped "
              "(reach ~150-175 km)", flush=True)
    if a.deep_from:
        if run_dir(a.deep_from).resolve() == rd.resolve():
            refuse("the deep pass needs its own run name")
        specs, sel = plan_deep(a, opt, rd, cfg)
    else:
        specs = plan_main(a, opt, rd, obs_dir)
    count = check_grids(specs, opt.roots)
    print(f"grids: {len(specs)} jobs; arrivals by the header rule: fine {count[0]}, wide {count[1]}, "
          f"box rejection {count[None]}", flush=True)
    jobs = check_resume(rd, cfg, specs, bool(a.deep_from))
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "run_config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    if a.deep_from:
        sel.to_csv(rd / "deep_selection.csv", index=False)
    for wd, ctl, text in jobs:
        wd.mkdir(exist_ok=True)
        (wd / "events.obs").write_text(text)
    print(f"{len(jobs)} NLLoc jobs ({len(specs) - len(jobs)} done before)", flush=True)
    bad = []
    with ProcessPoolExecutor(max(1, a.nproc)) as ex:
        for name, rc in ex.map(run_one, [(wd, ctl) for wd, ctl, _ in jobs]):
            if rc != 0:
                bad.append(name)
    print("failed:", bad if bad else "none", flush=True)
    nprob = account(rd, "deep" if a.deep_from else "main", a.nproc)
    if bad:
        sys.exit(1)
    if nprob:
        sys.exit(3)


if __name__ == "__main__":
    main()

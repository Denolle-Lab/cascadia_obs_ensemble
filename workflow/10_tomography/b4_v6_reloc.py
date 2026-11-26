"""Track B, catalog v6.0 (D-B079, D-B092, D-B095, D-B096, D-B097): NonLinLoc relocation of every catalog v4.1 event in
model v0.4.1 with NLL-SSST (Loc2ssst, CharDist 9999 -> 16 -> 8 -> 4 km) and the NLL-coherence stage, at catalog scale.
Called stage by stage by b4_v6_relocate.sh (one detached, resumable chain); every output goes under ROOT (reloc_v6/v60 for
the catalog, reloc_v6/smoke for the smoke test).

The method is the tested one (b4_v6_ssst.py, the cluster test of D-B092/D-B093, and the termtest harness of D-B095/D-B096):
NLLoc as v5.1 (EDT_OT_WT, LOCGAU 0.2 0.0 without LOCGAU2: D-B069; >= 5 phases: D-B066; the D-B056 conservative floor on the
D-B068 merged surface), Loc2ssst with b4_v6_ssst's LSPHSTAT, WeightFloor and LSGRID spacing, accumulated corrections
(each pass reads the previous pass's corrected grids), coherence = LOCPOSTERIOR OCT_TREE stacking of the final SSST
oct-trees for events with a compact pdf (Len3 <= 5 km), partners with pair coherence >= 0.5 (median cc_max of >= 3 CC rows,
rows straddling a confirmed clock epoch left out). What changes at catalog scale (notes/v6_plan.md):

* SSST tiles. The 25 km tiles of a2_make_obs.py, the events placed by their catalog v5.1 epicentre (v4 when v5.1 has no
  hypocentre; obs stage of the chain), as the cluster tests placed their search box around the v5.1 epicentres. Each tile
  is located in its own box, tile +- MARGIN km, z' -5..80 km (1 km nodes); that box is also Loc2ssst's LSOUTGRID, so the
  corrected grids of a pass are ~1.5 MB per station-phase (~150 GB per pass for the catalog).
* Base grids (pass it0 and the input of the first Loc2ssst): per tile and station-phase the fine 1 km grid if it holds the
  tile box, else the wide 2 km grid (NLLoc's header rule; links in ROOT/base/<tile>, base.json), so it0's residuals are
  relative to the grid Loc2ssst corrects.
* Loc2ssst per tile: the tile's events plus a halo (tiles within min(3 CharDist, 50 km) of the tile, nearest first);
  Loc2ssst reads at most 20,000 arrivals per station-phase (MAX_NUM_ARRIVALS_LOC2SSST), so a station-phase that would
  exceed it gets the nearest events only (a seeded random subset inside a tile that exceeds it alone: Axial); the
  LSSTATIONS line (< 1,000 characters) is split over several Loc2ssst instances (b4_v6_ssst.station_sets). Output grids
  start as links to the input grids, so a station-phase without accepted arrivals keeps its previous grid.
* Deep pass (D-B067) and edge pass: events of the final SSST pass rejected at the bottom face or located at z' >= 78 km
  (not far_only), and events rejected at a lateral face of the tile box, are located again as v5.1 did: the obs tile
  +- 75 km box, fine then wide grids (deep: z' -5..120 km, wide grids only), with static station terms from it0 (median
  residual of fine-grid arrivals, a2_station_terms.py rule); the SSST corrections stop at z' = 80 km and at the tile box.
  Flags deep_pass, edge_pass; station_terms = 'static' for these rows.

Checks (stop the chain): every NLLoc job returns 0 and passes the arrival accounting of a2_run_nlloc.account (no
undocumented arrival drop); every Loc2ssst instance returns 0 without an ERROR line (Loc2ssst's makecpt calls are
skipped: no GMT on its PATH); the deep and edge passes refuse missing grids (a2_run_nlloc.check_grids, check_wide_manifest)
and must cover every selected event; every NLLoc run records the grid manifest md5 (grids_<TIME>.md5) and the md5 of the
floor, template and NLLoc binary.

Processes: at most the number in reloc_v6/control/nproc (read before each start; default --nproc), a new process only
while MemAvailable >= V6_MEM_MIN_GB (35; PI: keep > 30 GB), nice inherited from the chain (10).

  python b4_v6_reloc.py plan ROOT [--tiles a,b]   # tiles, boxes, base grid links, jobs
  python b4_v6_reloc.py nll ROOT K                # NLLoc pass it<K> (K = 0..4; it4 keeps oct-trees)
  python b4_v6_reloc.py lss ROOT K                # Loc2ssst with CharDist SCHEDULE[K] on it<K> -> g<K>
  python b4_v6_reloc.py coh ROOT [--cc DIR]       # NLL-coherence on it4 (CC of DIR)
  python b4_v6_reloc.py terms ROOT                # static terms from it0 (deep and edge passes)
  python b4_v6_reloc.py fallback ROOT             # deep and edge passes
  python b4_v6_reloc.py catalog ROOT --out CSV    # catalog v6.0 (b4_v6_catalog.py)
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TA = HERE.parent / "08_relocation"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TA))
import a2_run_nlloc as RN  # noqa: E402
import b4_v6_ssst as S  # noqa: E402

R6 = Path("/wd1/mdenolle_data/reloc_v6")
VERSION = os.environ.get("VERSION", "0.4.1")
TIME = os.environ.get("TIME", f"time_v6_v{VERSION}")
FINE = Path(os.environ.get("V6_FINE", str(R6 / TIME)))
WIDE = Path(os.environ.get("V6_WIDE", str(R6 / f"wide_{TIME}")))
GRID_MD5 = Path(os.environ.get("V6_GRID_MD5", str(R6 / f"grids_{TIME}.md5")))
OBS = Path(os.environ.get("V6_OBS", str(R6 / "obs_tiles_v6")))
CC_DIR = Path(os.environ.get("V6_CC", "/wd1/mdenolle_data/tomo_v1/cc_dt_all"))
FLOOR = S.FLOOR
NLL = RN.NLL
LOC2SSST = S.BIN / "Loc2ssst"
TEMPLATE = TA / "nlloc_template.in"
SCHEDULE = (9999.0, 16.0, 8.0, 4.0)       # D-B096
V51_CATALOG = Path(os.environ.get("V6_V51_CATALOG", "/wd1/mdenolle_data/catalog_v5/v5.1/cascadia_catalog_v5.1.csv"))
NPASS = len(SCHEDULE)                       # NLLoc passes it0..it4; it4 is the final SSST pass
MARGIN = 20.0                               # km: SSST tile box = 25 km tile +- MARGIN
TILE = 25.0
X0, Y0, X1, Y1 = -513.0, -709.0, 457.0, 724.0     # model lattice (a2_make_obs.py)
CHUNK = int(os.environ.get("V6_CHUNK", RN.CHUNK))      # events per NLLoc job (results do not depend on it)
HALO_MAX = 50.0                             # km: Loc2ssst halo min(3 CharDist, HALO_MAX)
MAX_ARR = 20000                             # Loc2ssst MAX_NUM_ARRIVALS_LOC2SSST
N_MAX = int(os.environ.get("V6_LSS_NMAX", "4000"))   # input events per Loc2ssst run (its cost grows as N^2 per station-phase)
LSS_MAXLEN = 990                            # LSSTATIONS line length (NonLinLoc line buffer 1,000)
MEM_MIN_GB = float(os.environ.get("V6_MEM_MIN_GB", "35"))
CTRL = R6 / "control" / "nproc"
CTRL_LSS = R6 / "control" / "nproc_lss"    # Loc2ssst: 'auto 1 32' = one instance while the ensemble runs (coordinator 20:00)
LOCGAU, LOCGAU2 = S.LOCGAU, S.LOCGAU2       # D-B069
LATERAL, BOTTOM, DEEP_Z = {10, 11, 20, 21}, 31, 78.0
ENV = dict(os.environ)


def log(*a):
    print(time.strftime("%F %T"), *a, flush=True)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def md5_text(t):
    return hashlib.md5(t.encode()).hexdigest()


# --------------------------------------------------------------------------------------------------------------- pool
def mem_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576.0
    return 1e9


_BUSY = {"t": 0.0, "busy": True}


def machine_busy():
    """True while the tomography ensemble (tomoDD-SE processes) runs; checked at most once a minute."""
    if time.time() - _BUSY["t"] > 60:
        n = 0
        for p in Path("/proc").iterdir():
            if p.name.isdigit():
                try:
                    if (p / "comm").read_text().startswith("tomoDD-SE"):
                        n += 1
                except OSError:
                    pass
        _BUSY.update(t=time.time(), busy=n > 0)
    return _BUSY["busy"]


def nproc_limit(default, ctrl=None):
    """process limit: the control file (control/nproc, or control/nproc_lss for Loc2ssst) holds a number, or
    'auto BUSY FREE' (BUSY while tomoDD-SE processes run, else FREE)."""
    try:
        t = (ctrl or CTRL).read_text().split()
        if t[0] == "auto":
            return max(1, int(t[1] if machine_busy() else t[2]))
        return max(1, int(t[0]))
    except (OSError, ValueError, IndexError):
        return default


def run_pool(tasks, label, nproc, ramp=0.05, env=None, on_done=None, est_default=0.5, ctrl=None):
    """tasks: [(name, argv, cwd, logfile[, est_gb])]; at most nproc_limit(nproc) at a time, at most one start per `ramp`
    seconds; a task starts only while MemAvailable - (estimates of the running tasks started in the last 20 s, whose
    memory MemAvailable may not show yet) - its own estimate >= MEM_MIN_GB; the first queued task that fits starts (smaller
    ones skip ahead of a large one that does not fit). on_done(name, rc) is called as each finishes. Returns {name: rc}."""
    queue = [tuple(x) + ((est_default,) if len(x) == 4 else ()) for x in tasks]
    running, rc, recent = {}, {}, []
    t0, last_log, last_start = time.time(), 0.0, 0.0
    while queue or running:
        for name in list(running):
            p, fh = running[name]
            r = p.poll()
            if r is not None:
                fh.close()
                rc[name] = r
                del running[name]
                if on_done:
                    on_done(name, r)
        lim = nproc_limit(nproc, ctrl)
        while queue and len(running) < lim and time.time() - last_start >= ramp:
            now = time.time()
            recent = [(ts, e, nm) for ts, e, nm in recent if now - ts < 20 and nm in running]
            free = mem_gb() - sum(e for _, e, _ in recent) - MEM_MIN_GB
            i = next((j for j, x in enumerate(queue) if x[4] <= free), None)
            if i is None and not running:      # idle: a large task may use memory down to the PI's 30 GB floor
                i = next((j for j, x in enumerate(queue) if x[4] <= free + MEM_MIN_GB - 30.0), None)
                if i is None and now - last_start > 600:
                    log(f"{label}: waiting for memory: {len(queue)} queued, smallest needs "
                        f"{min(x[4] for x in queue):.1f} GB, MemAvailable {mem_gb():.0f} GB")
                    last_start = now
            if i is None:
                break
            name, argv, cwd, logf, est = queue.pop(i)
            fh = open(logf, "w")
            running[name] = (subprocess.Popen([str(x) for x in argv], cwd=cwd, stdout=fh, stderr=subprocess.STDOUT,
                                              env=env or ENV), fh)
            recent.append((now, est, name))
            last_start = now
            if ramp >= 0.2:
                break
        if time.time() - last_log > 300:
            log(f"{label}: {len(rc)}/{len(tasks)} done, {len(running)} running (limit {lim}), "
                f"MemAvailable {mem_gb():.0f} GB, {(time.time() - t0) / 60:.0f} min")
            last_log = time.time()
        time.sleep(0.1)
    log(f"{label}: {len(rc)} tasks in {(time.time() - t0) / 60:.1f} min, {sum(r != 0 for r in rc.values())} failed")
    return rc


# --------------------------------------------------------------------------------------------------------------- plan
def tile_box(ix, iy, margin=MARGIN):
    x0 = max(X0 + ix * TILE - margin, X0)
    x1 = min(X0 + (ix + 1) * TILE + margin, X1)
    y0 = max(Y0 + iy * TILE - margin, Y0)
    y1 = min(Y0 + (iy + 1) * TILE + margin, Y1)
    return (float(x0), float(y0), int(round(x1 - x0)) + 1, int(round(y1 - y0)) + 1)


def read_blocks(path):
    return [b.strip("\n") for b in Path(path).read_text().split("\n\n") if b.strip()]


def plan_dir(root):
    return Path(root) / "plan"


def load_plan(root):
    P = plan_dir(root)
    if not (P / "plan.json").exists():
        raise SystemExit(f"{P}/plan.json missing: run the plan stage")
    tiles = pd.read_csv(P / "tiles.csv").set_index("tile")
    jobs = json.loads((P / "jobs.json").read_text())
    return tiles, jobs


def grid_manifest_md5():
    return md5(GRID_MD5) if GRID_MD5.exists() else None


def cut_fine(args):
    """write the tile-box cut of a fine grid: the fine grids are on the 1 km lattice with integer origins and z' -5..80 km,
    as the tile boxes, so the cut holds exactly the grid's node values (1.5 MB instead of ~80 MB per NLLoc grid in
    memory); header lines after the first are copied."""
    src, dst, x0, y0, nx_b, ny_b = args
    h = Path(f"{src}.hdr").read_text().splitlines()
    t = h[0].split()
    nx, ny, nz = (int(v) for v in t[:3])
    ox, oy, oz, dx, dy, dz = (float(v) for v in t[3:9])
    if not (dx == dy == dz == 1.0 and oz == RN.Z0 and nz == RN.NZ and ox == round(ox) and oy == round(oy)):
        raise ValueError(f"{src}: grid not on the 1 km lattice of the tile boxes")
    i0, j0 = int(round(x0 - ox)), int(round(y0 - oy))
    if i0 < 0 or j0 < 0 or i0 + nx_b > nx or j0 + ny_b > ny:
        raise ValueError(f"{src}: tile box outside the grid")
    a = np.memmap(f"{src}.buf", dtype="<f4", mode="r", shape=(nx, ny, nz))
    sub = np.ascontiguousarray(a[i0:i0 + nx_b, j0:j0 + ny_b, :])
    sub.tofile(f"{dst}.buf.tmp")
    os.replace(f"{dst}.buf.tmp", f"{dst}.buf")
    t[0], t[1], t[3], t[4] = str(nx_b), str(ny_b), f"{x0:.6f}", f"{y0:.6f}"
    Path(f"{dst}.hdr").write_text(" ".join(t) + "\n" + "\n".join(h[1:]) + "\n")
    return dst


def cmd_plan(a):
    root = Path(a.root)
    P = plan_dir(root)
    if (P / "plan.json").exists():
        log(f"plan {P} exists")
        return
    if P.exists():
        shutil.rmtree(P)
    tiles = pd.read_csv(OBS / "tiles.csv")
    if a.tiles:
        want = a.tiles.split(",")
        miss = sorted(set(want) - set(tiles.tile))
        if miss:
            raise SystemExit(f"tiles not in {OBS}/tiles.csv: {miss}")
        tiles = tiles[tiles.tile.isin(want)]
    roots = (f"{FINE}/cascadia3d", f"{WIDE}/cascadia3d")
    (P / "obs").mkdir(parents=True)
    rows, jobs, counts, events, cuts = [], {}, [], [], []
    for t in tiles.itertuples():
        blocks = read_blocks(OBS / f"{t.tile}.obs")
        box = tile_box(t.ix, t.iy)
        ext = S.box_ext(box)
        sp = S.station_phases(blocks)
        bd = root / "base" / t.tile
        bd.mkdir(parents=True, exist_ok=True)
        used = {}
        # both phases of every station of the tile: Loc2ssst selects stations (LSSTATIONS), not phases, so it corrects
        # the other phase too when halo events have it, and needs its input grid
        for (ph, sta) in sorted({(p_, s_) for _, s_ in sp for p_ in ("P", "S")}):
            n = sp.get((ph, sta), 0)
            i = RN.serving_root(ext, (ph, sta), roots)
            if i is None:
                raise SystemExit(f"{t.tile}: no grid holds the box for {ph}.{sta}")
            if not RN.headers(roots[i])[(ph, sta)][6]:
                raise SystemExit(f"{t.tile}: incomplete grid buffer {roots[i]}.{ph}.{sta}")
            if i == 0:          # fine grid: exact tile-box cut (cut_fine), written below in parallel
                cuts.append((f"{roots[0]}.{ph}.{sta}.time", str(bd / f"cascadia3d.{ph}.{sta}.time"), *box))
            else:               # wide grid (2 km): a link, NLLoc and Loc2ssst interpolate it as in the tests
                for e in ("hdr", "buf"):
                    ln = bd / f"cascadia3d.{ph}.{sta}.time.{e}"
                    tgt = Path(roots[i]).parent / f"cascadia3d.{ph}.{sta}.time.{e}"
                    if not ln.is_symlink():
                        ln.symlink_to(tgt)
                    if os.readlink(ln) != str(tgt):
                        raise SystemExit(f"{ln} points to {os.readlink(ln)}, not {tgt}")
            used[f"{ph}.{sta}"] = ("fine", "wide")[i]
            if n:
                counts.append((t.tile, ph, sta, n))
        (bd / "base.json").write_text(json.dumps(used, indent=0))
        chunks = [blocks[i:i + CHUNK] for i in range(0, len(blocks), CHUNK)]
        for c, bl in enumerate(chunks):
            name = t.tile if len(chunks) == 1 else f"{t.tile}_c{c:02d}"
            (P / "obs" / f"{name}.obs").write_text("\n\n".join(bl) + "\n\n")
            jobs[name] = dict(tile=t.tile, n=len(bl))
            for b in bl:
                eid, arr = RN.block_arrivals(b)
                nf = sum(used[f"{ph}.{sta}"] == "fine" for ph, sta in arr)
                events.append((eid, t.tile, name, len(arr), nf, len(arr) - nf))
        rows.append(dict(tile=t.tile, ix=t.ix, iy=t.iy, x0=box[0], y0=box[1], nx=box[2], ny=box[3],
                         box_x0=t.box_x0, box_y0=t.box_y0, box_x1=t.box_x1, box_y1=t.box_y1, n_events=len(blocks),
                         n_stph=len(sp), n_fine=sum(used[f"{p_}.{s_}"] == "fine" for p_, s_ in sp),
                         n_wide=sum(used[f"{p_}.{s_}"] == "wide" for p_, s_ in sp), n_base_links=len(used)))
    t0 = time.time()
    with ProcessPoolExecutor(16) as ex:
        n_cut = len(list(ex.map(cut_fine, cuts, chunksize=64)))
    log(f"plan: {n_cut} fine base grids cut to their tile box in {(time.time() - t0) / 60:.1f} min")
    tl = pd.DataFrame(rows)
    tl.to_csv(P / "tiles.csv", index=False)
    (P / "jobs.json").write_text(json.dumps(jobs, indent=0))
    pd.DataFrame(counts, columns=["tile", "phase", "sta", "n"]).to_parquet(P / "stph_counts.parquet", index=False)
    ev = pd.DataFrame(events, columns=["ID", "tile", "job", "n_obs", "n_fine_base", "n_wide_base"])
    if ev.ID.duplicated().any():
        raise SystemExit(f"{int(ev.ID.duplicated().sum())} events in more than one tile")
    ev.to_parquet(P / "events.parquet", index=False)
    meta = dict(obs=str(OBS), obs_meta_md5=md5(OBS / "obs_meta.json"), fine=str(FINE), wide=str(WIDE),
                base="fine grids cut to the tile box (exact node values), wide grids linked", fine_cuts=n_cut,
                grid_manifest=str(GRID_MD5), grid_manifest_md5=grid_manifest_md5(), margin_km=MARGIN, tile_km=TILE,
                schedule=SCHEDULE, tiles=len(tl), jobs=len(jobs), events=len(ev), station_phases=int(tl.n_stph.sum()),
                station_phases_fine=int(tl.n_fine.sum()), station_phases_wide=int(tl.n_wide.sum()),
                corrected_grid_bytes_per_pass=int((tl.nx * tl.ny * RN.NZ * 4 * tl.n_stph).sum()),
                utc=time.strftime("%FT%T%z"))
    (P / "plan.json").write_text(json.dumps(meta, indent=1))
    log(f"plan: {len(tl)} tiles, {len(jobs)} jobs, {len(ev)} events, {int(tl.n_stph.sum())} station-phases "
        f"({int(tl.n_fine.sum())} fine, {int(tl.n_wide.sum())} wide base grids); corrected grids "
        f"{meta['corrected_grid_bytes_per_pass'] / 1e9:.0f} GB per pass")


# ---------------------------------------------------------------------------------------------------------- NLLoc pass
def template_text(octree=False, public_id=True):
    t = TEMPLATE.read_text()
    hyp = "LOCHYPOUT SAVE_NLLOC_ALL NLL_FORMAT_VER_2" + (" FILENAME_PUBLIC_ID" if public_id else "") + \
          (" SAVE_NLLOC_OCTREE" if octree else "")
    return RN.sub1(r"^LOCHYPOUT .*$", hyp, t)


def make_opt(roots, tmpl):
    return argparse.Namespace(time=str(roots[0]), roots=tuple(str(r) for r in roots), template_text=tmpl,
                              locgau=LOCGAU, locgau2=LOCGAU2, topo=FLOOR, max_dist=RN.MAX_DIST)


def grid_root(root, k, tile):
    return f"{root}/base/{tile}/cascadia3d" if k == 0 else f"{root}/g{k - 1}/{tile}/cascadia3d"


def sum_rows(path):
    """(ID, status, comment, boundary code, x, y, z) of every event of an NLLoc summary file."""
    out, cur = [], None
    for line in open(path, errors="replace"):
        if line.startswith("NLLOC "):
            q = re.findall(r'"([^"]*)"', line)
            b = re.search(r"grid boundary (\d+)", q[2] if len(q) > 2 else "")
            cur = dict(root=q[0], status=q[1], comment=q[2] if len(q) > 2 else "", boundary=int(b.group(1)) if b else None)
        elif cur is None:
            continue
        elif line.startswith("PUBLIC_ID"):
            cur["ID"] = int(line.split()[1].split("_")[1])
        elif line.startswith("HYPOCENTER"):
            t = line.split()
            cur.update(x=float(t[t.index("x") + 1]), y=float(t[t.index("y") + 1]), z=float(t[t.index("z") + 1]))
        elif line.startswith("STATISTICS"):
            t = line.split()
            cur["len3"] = float(t[t.index("Len3") + 1])
        elif line.startswith("END_NLLOC"):
            out.append(cur)
            cur = None
    return out


def pass_summary(rd):
    rows = []
    for f in sorted(rd.glob("*/loc.sum.grid0.loc.hyp")):
        for r in sum_rows(f):
            r["job"] = f.parent.name
            rows.append(r)
    return pd.DataFrame(rows)


def concat_tile_hyps(args):
    rd, tile, jobs = args
    out = rd / "hyp" / f"{tile}.hyp"
    n = 0
    with open(out, "w") as fo:
        for j in jobs:
            for f in sorted((rd / j).glob("loc.*_v4_*.grid0.loc.hyp")):
                fo.write(f.read_text(errors="replace").rstrip("\n") + "\n\n")
                n += 1
    return tile, n


def cmd_nll(a):
    root, k = Path(a.root), int(a.k)
    rd = root / f"it{k}"
    if (rd / "run.json").exists():
        log(f"{rd} done")
        return
    tiles, jobs = load_plan(root)
    P = plan_dir(root)
    if k > 0:
        miss = [t for t in tiles.index if not (root / f"g{k - 1}" / t / "loc2ssst.json").exists()]
        if miss:
            raise SystemExit(f"g{k - 1} incomplete for {len(miss)} tiles, e.g. {miss[:5]}")
    tmpl = template_text(octree=(k == NPASS))
    rd.mkdir(parents=True, exist_ok=True)
    tasks, ctl_md5 = [], {}
    for name, j in jobs.items():
        t = tiles.loc[j["tile"]]
        box = (float(t.x0), float(t.y0), int(t.nx), int(t.ny))
        wd = rd / name
        ctl = RN.control(box, wd / "events.obs", wd / "loc", None, make_opt((grid_root(root, k, j["tile"]),), tmpl))
        if (wd / "done").exists() and (wd / "nlloc.in").exists() and (wd / "nlloc.in").read_text() == ctl:
            continue
        if wd.exists():
            shutil.rmtree(wd)
        wd.mkdir(parents=True)
        shutil.copyfile(P / "obs" / f"{name}.obs", wd / "events.obs")
        (wd / "nlloc.in").write_text(ctl)
        tasks.append((name, [NLL, "nlloc.in"], wd, wd / "nlloc.log"))

    def done(name, r):
        if r == 0:
            (rd / name / "done").write_text("")
    # largest jobs first (arrivals in the job's obs): only the order of the queue, not the results, so no long tail
    cost = pd.read_parquet(P / "events.parquet").groupby("job").n_obs.sum()
    tasks.sort(key=lambda x: -cost.get(x[0], 0))
    t0 = time.time()
    log(f"it{k}: {len(tasks)} NLLoc jobs to run ({len(jobs) - len(tasks)} done before)")
    rc = run_pool(tasks, f"it{k}", a.nproc, ramp=(0.5 if k == 0 else 0.05), on_done=done)   # ~0.35 GB each (box grids)
    bad = sorted(n for n, r in rc.items() if r != 0)
    if bad:
        raise SystemExit(f"it{k}: {len(bad)} NLLoc jobs failed, e.g. {bad[:5]} (see their nlloc.log)")
    nprob = RN.account(rd, "main", 32)
    if nprob:
        raise SystemExit(f"it{k}: {nprob} undocumented arrival drops ({rd}/arrival_problems.csv)")
    (rd / "hyp").mkdir(exist_ok=True)
    bytile = {}
    for name, j in jobs.items():
        bytile.setdefault(j["tile"], []).append(name)
    with ProcessPoolExecutor(16) as ex:
        nfiles = dict(ex.map(concat_tile_hyps, [(rd, t, js) for t, js in bytile.items()]))
    s = pass_summary(rd)
    nev = sum(j["n"] for j in jobs.values())
    if s.ID.duplicated().any():
        raise SystemExit(f"it{k}: {int(s.ID.duplicated().sum())} events with more than one summary row")
    nfile = sum(nfiles.values())
    st = s.status.value_counts().to_dict()
    lat = int((s.status.eq("REJECTED") & s.boundary.isin(LATERAL)).sum())
    bot = int((s.status.eq("REJECTED") & s.boundary.eq(BOTTOM)).sum())
    no_row = nev - len(s)          # NLLoc writes no row for an event with fewer usable phases than the minimum
    info = dict(pass_=k, events_without_row=no_row, grids=("base (fine/wide links)" if k == 0 else f"g{k - 1} (Loc2ssst CharDist {SCHEDULE[k - 1]:g} km)"),
                octree=(k == NPASS), events=nev, per_event_files=nfile, status=st, rejected_lateral=lat,
                rejected_bottom=bot, located_deep=int((s.status.eq("LOCATED") & (s.z >= DEEP_Z)).sum()),
                jobs=len(jobs), jobs_run=len(tasks), wall_min=round((time.time() - t0) / 60, 1),
                grid_manifest_md5=grid_manifest_md5(), floor=FLOOR, floor_md5=md5(FLOOR), nll_md5=md5(NLL),
                template_md5=md5_text(tmpl), locgau=LOCGAU, locgau2=LOCGAU2, utc=time.strftime("%FT%T%z"))
    if nfile != nev:
        log(f"WARNING it{k}: {nfile} per-event files for {nev} events")
    (rd / "run.json").write_text(json.dumps(info, indent=1))
    log(f"it{k}: {nev} events, status {st}, rejected at a lateral face {lat}, at the bottom {bot}")


# ----------------------------------------------------------------------------------------------------------- Loc2ssst
def tile_dist(a, b):
    """km between the extents of two tiles (ix, iy)."""
    dx = max(0, abs(a[0] - b[0]) - 1) * TILE
    dy = max(0, abs(a[1] - b[1]) - 1) * TILE
    return float(np.hypot(dx, dy))


_CACHE = {}


def hyp_blocks(path):
    """{ID: block text} of a concatenated per-event hyp file (cached: dense tiles are partial inputs of many tiles)."""
    if str(path) in _CACHE:
        return _CACHE[str(path)]
    out = {}
    for b in Path(path).read_text(errors="replace").split("END_NLLOC"):
        m = re.search(r"^PUBLIC_ID v4_(\d+)", b, re.M)
        if m:
            out[int(m.group(1))] = b.strip("\n") + "\nEND_NLLOC\n\n"
    _CACHE[str(path)] = out
    return out


def obs_arrivals(root, tile, jobs_by_tile):
    """{ID: [(phase, station)]} of a tile (plan obs; cached)."""
    key = ("obs", str(root), tile)
    if key in _CACHE:
        return _CACHE[key]
    out = {}
    for j in jobs_by_tile[tile]:
        for b in read_blocks(plan_dir(root) / "obs" / f"{j}.obs"):
            eid, arr = RN.block_arrivals(b)
            out[eid] = arr
    _CACHE[key] = out
    return out


def lss_inputs(root, k, tile, tiles, counts, jobs_by_tile, pos, char, located):
    """Loc2ssst input files of a tile in g<k>/<tile>/lsin: the LOCATED events of the tile and of the tiles within the
    halo min(3 CharDist, HALO_MAX), at most N_MAX events and at most MAX_ARR arrivals per station-phase. Loc2ssst's cost
    grows as (input events)^2 per station-phase (NonLinLoc's linked-list lookups), so dense regions get a subset: the
    tile's own events in a seeded random order first, then the halo events nearest to the tile. When everything fits,
    the whole tile files are linked. located = {tile: [IDs LOCATED in it<k>]}. Returns a record of what went in."""
    rd = root / f"it{k}"
    lsin = root / f"g{k}" / tile / "lsin"
    lsin.mkdir(parents=True, exist_ok=True)
    H = min(3.0 * char, HALO_MAX)
    me = (int(tiles.loc[tile, "ix"]), int(tiles.loc[tile, "iy"]))
    cand = sorted(((tile_dist(me, (int(r.ix), int(r.iy))), t) for t, r in tiles.iterrows()
                   if tile_dist(me, (int(r.ix), int(r.iy))) <= H), key=lambda v: (v[0], v[1] != tile, v[1]))
    rec = dict(halo_km=H, n_max=N_MAX, tiles_full=[], tiles_partial={}, events_dropped=0)
    n_all = sum(len(located.get(t, ())) for _, t in cand)
    tot = {}
    for _, t in cand:
        for kk, v in counts.get(t, {}).items():
            tot[kk] = tot.get(kk, 0) + v
    if n_all <= N_MAX and all(v <= MAX_ARR for v in tot.values()):
        for _, t in cand:
            (lsin / f"{t}.hyp").symlink_to(rd / "hyp" / f"{t}.hyp")
            rec["tiles_full"].append(t)
    else:
        rng = np.random.default_rng(int(md5_text(f"{tile}:{k}")[:8], 16))      # stable seed per tile and pass
        own = list(located.get(tile, ()))
        rng.shuffle(own)
        x0, y0 = X0 + me[0] * TILE, Y0 + me[1] * TILE
        def dd(i):
            p = pos[i]
            return np.hypot(max(0, x0 - p[0], p[0] - x0 - TILE), max(0, y0 - p[1], p[1] - y0 - TILE))
        halo = sorted((dd(i), i, t) for _, t in cand if t != tile for i in located.get(t, ()))
        order = [(i, tile) for i in own] + [(i, t) for _, i, t in halo]
        keep, tot = {}, {}
        n = 0
        for i, t in order:
            if n >= N_MAX:
                break
            need = {}
            for kk in obs_arrivals(root, t, jobs_by_tile)[i]:
                need[kk] = need.get(kk, 0) + 1
            if all(tot.get(kk, 0) + v <= MAX_ARR for kk, v in need.items()):
                keep.setdefault(t, []).append(i)
                for kk, v in need.items():
                    tot[kk] = tot.get(kk, 0) + v
                n += 1
        for t, ids in keep.items():
            blocks = hyp_blocks(rd / "hyp" / f"{t}.hyp")
            with open(lsin / f"{t}.hyp", "w") as fo:
                for i in ids:
                    fo.write(blocks[i])
            rec["tiles_partial"][t] = [len(ids), len(located.get(t, ()))]
        rec["events_dropped"] = n_all - n
    rec["events_in"] = int(n_all - rec["events_dropped"])
    rec["max_arrivals_station_phase"] = max(tot.values()) if tot else 0
    rec["arrivals_in"] = int(sum(tot.values()))
    return rec


def lss_tile_setup(root, k, tile, tiles, counts, jobs_by_tile, pos, located):
    """g<k>/<tile>: links to the previous grids, the inputs and the control files of the Loc2ssst instances; returns the
    tasks [(name, argv, cwd, log)]."""
    char = SCHEDULE[k]
    d_in = root / "base" / tile if k == 0 else root / f"g{k - 1}" / tile
    d_out = root / f"g{k}" / tile
    if d_out.exists():
        shutil.rmtree(d_out)
    (d_out / "work").mkdir(parents=True)
    for f in sorted(d_in.glob("cascadia3d.*.time.*")):
        (d_out / f.name).symlink_to(os.path.realpath(f))
    rec = lss_inputs(root, k, tile, tiles, counts, jobs_by_tile, pos, char, located)
    t = tiles.loc[tile]
    x0, y0, nx, ny = float(t.x0), float(t.y0), int(t.nx), int(t.ny)
    s = S.lsgrid_km(char)
    gx, gy, gz = (int(np.ceil((n - 1) / s)) + 1 for n in (nx, ny, RN.NZ))
    sp = counts[tile]
    n = 1
    while True:
        sets = S.station_sets(sp, n)
        if all(len("LSSTATIONS =" + "=".join(x) + "=") < LSS_MAXLEN for x in sets):
            break
        n += 1
    obs = plan_dir(root) / "obs" / f"{jobs_by_tile[tile][0]}.obs"
    base = (f"CONTROL 1 54321\nTRANS LAMBERT WGS-84 45.0 -125.8 41.0 49.0 0.0\n"
            f"LSGRID {gx} {gy} {gz} {x0:.1f} {y0:.1f} {RN.Z0:.1f} {s:.1f} {s:.1f} {s:.1f} SSST_TIMECORR FLOAT\n"
            f"LSOUTGRID {nx} {ny} {RN.NZ} {x0:.1f} {y0:.1f} {RN.Z0:.1f} 1.0 1.0 1.0 TIME FLOAT\n"
            f"LSPHSTAT {S.LSPHSTAT}\nLOCPHASEID P P p Pg Pn\nLOCPHASEID S S s Sg Sn\n"
            f"LSPARAMS {char:g} {S.WEIGHT_FLOOR:g} 0\nLSMODE ANGLES_NO\nLSOUT {d_out}/cascadia3d\n"
            f"LSLOCFILES {d_out}/lsin/*.hyp\n"
            f"LOCFILES {obs} NLLOC_OBS {d_in}/cascadia3d {d_out}/unused 0\n"
            f"LOCMETH EDT_OT_WT 9999.0 5 -1 -1 -1 20 -1.0 1\n")
    tasks = []
    for i, st in enumerate(sets):
        (d_out / "work" / f"ssst_{i}.in").write_text(base + "LSSTATIONS =" + "=".join(st) + "=\n")
        # memory: ~1.55 GB per 100,000 input arrivals per instance (measured on g0, 2026-11-25)
        tasks.append((f"{tile}#{i}", ["/usr/bin/time", "-f", "maxrss_kb %M", "-o", f"ssst_{i}.time", LOC2SSST,
                                      f"ssst_{i}.in"], d_out / "work", d_out / "work" / f"ssst_{i}.log",
                      0.2 + 1.6e-5 * rec["arrivals_in"]))
    rec.update(char_dist_km=char, lsgrid_km=s, lsgrid=[gx, gy, gz], box=[x0, y0, nx, ny], station_sets=sets,
               weight_floor=S.WEIGHT_FLOOR, lsphstat=S.LSPHSTAT, input=str(d_in))
    (d_out / "setup.json").write_text(json.dumps(rec, indent=0))
    return tasks


def lss_tile_check(args):
    """after the instances of a tile: return codes, error lines, accepted events, corrected grids, negative nodes."""
    d_out, rcs = args
    d_out = Path(d_out)
    rec = json.loads((d_out / "setup.json").read_text())
    logs = "\n".join((d_out / "work" / f"ssst_{i}.log").read_text(errors="replace") for i in range(len(rec["station_sets"])))
    errs = [ln for ln in logs.splitlines() if "ERROR" in ln and "makecpt" not in ln]
    m = re.findall(r"(\d+) location files read, (\d+) accepted", logs)
    acc = int(m[0][1]) if m else 0
    new = sorted(f for f in d_out.glob("cascadia3d.*.time.buf") if not f.is_symlink())
    n_neg, tmin, grids = 0, None, {}
    for f in new:
        t = np.fromfile(f, "<f4")
        neg = int((t < 0).sum())
        n_neg += neg
        tm = float(t.min())
        tmin = tm if tmin is None else min(tmin, tm)
        c = np.fromfile(f.with_name(f.name.replace(".time.buf", ".ssst.buf")), "<f4")
        grids[f.name[len("cascadia3d."):-len(".time.buf")]] = dict(neg_nodes=neg, corr_mean=round(float(c.mean()), 4),
                                                                   corr_min=round(float(c.min()), 4),
                                                                   corr_max=round(float(c.max()), 4))
    links = len([f for f in d_out.glob("cascadia3d.*.time.buf") if f.is_symlink()])
    peak = []
    for i in range(len(rec["station_sets"])):
        m = re.search(r"maxrss_kb (\d+)", (d_out / "work" / f"ssst_{i}.time").read_text()) \
            if (d_out / "work" / f"ssst_{i}.time").exists() else None
        peak.append(round(int(m.group(1)) / 1048576, 3) if m else None)
    rec["peak_rss_gb"] = peak
    rec.update(return_codes=rcs, error_lines=len(errs), error_examples=errs[:3], events_accepted=acc,
               corrected_grids=len(new), kept_previous=links, negative_nodes=n_neg, t_min=tmin, grids=grids,
               loc2ssst_md5=md5(LOC2SSST))
    ok = not any(rcs) and not errs
    if ok:
        (d_out / "loc2ssst.json").write_text(json.dumps(rec, indent=0))
    return d_out.name, ok, rec


def prune(root, k):
    """delete the corrected grids of g<k-1> that no grid of g<k> links to (those of the station-phases g<k> corrected);
    g<k-1>'s correction grids (.ssst.*), records and inputs stay."""
    old, new = root / f"g{k - 1}", root / f"g{k}"
    keep = set()
    for f in new.glob("*/cascadia3d.*.time.*"):
        if f.is_symlink():
            keep.add(os.path.realpath(f))
    n, b = 0, 0
    for f in old.glob("*/cascadia3d.*.time.*"):
        if not f.is_symlink() and str(f) not in keep:
            b += f.stat().st_size
            f.unlink()
            n += 1
    log(f"pruned g{k - 1}: {n} corrected grids ({b / 1e9:.1f} GB) no longer referenced; {len(keep)} links of g{k} kept")


DISK_MIN_GB = float(os.environ.get("V6_DISK_MIN_GB", "500"))     # keep at least this much free on /wd1 (PI)


def disk_free_gb(path=R6):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def wait_disk(need_gb, label):
    """wait (log every 10 min) until the disk keeps DISK_MIN_GB free after writing need_gb."""
    n = 0
    while disk_free_gb() - need_gb < DISK_MIN_GB:
        if n % 10 == 0:
            log(f"{label}: {disk_free_gb():.0f} GB free, {need_gb:.0f} GB needed, {DISK_MIN_GB:.0f} GB must stay free: waiting")
        n += 1
        time.sleep(60)


def cmd_lss(a):
    root, k = Path(a.root), int(a.k)
    gd = root / f"g{k}"
    if (gd / "summary.json").exists():
        log(f"{gd} done")
        return
    plan = json.loads((plan_dir(root) / "plan.json").read_text())
    wait_disk(1.3 * plan["corrected_grid_bytes_per_pass"] / 1e9, f"g{k}")
    rd = root / f"it{k}"
    if not (rd / "run.json").exists():
        raise SystemExit(f"{rd} not finished")
    tiles, jobs = load_plan(root)
    jobs_by_tile = {}
    for name, j in jobs.items():
        jobs_by_tile.setdefault(j["tile"], []).append(name)
    cdf = pd.read_parquet(plan_dir(root) / "stph_counts.parquet")
    counts = {t: {(r.phase, r.sta): int(r.n) for r in g.itertuples()} for t, g in cdf.groupby("tile")}
    s = pass_summary(rd)
    pos = {int(r.ID): (r.x, r.y) for r in s.itertuples() if r.status == "LOCATED"}
    ev_tile = pd.read_parquet(plan_dir(root) / "events.parquet").set_index("ID").tile
    located = {}
    for i in pos:
        located.setdefault(ev_tile[i], []).append(i)
    t0 = time.time()
    tasks, todo = [], []
    for tile in tiles.index:
        if (gd / tile / "loc2ssst.json").exists():
            continue
        todo.append(tile)
        tasks += lss_tile_setup(root, k, tile, tiles, counts, jobs_by_tile, pos, located)
    log(f"g{k} (CharDist {SCHEDULE[k]:g} km): {len(todo)} tiles, {len(tasks)} Loc2ssst instances")
    env = dict(ENV, PATH="/usr/bin:/bin")             # no GMT: Loc2ssst's makecpt call fails at once (cpt not needed)
    tasks.sort(key=lambda x: -x[4])                    # largest first; smaller ones fill in while memory allows
    log(f"g{k}: estimated memory per instance {min(x[4] for x in tasks):.1f}-{max(x[4] for x in tasks):.1f} GB")
    rc = run_pool(tasks, f"g{k}", a.nproc, ramp=0.05, env=env, ctrl=CTRL_LSS)
    bytile = {}
    for name, r in rc.items():
        bytile.setdefault(name.split("#")[0], []).append(r)
    with ProcessPoolExecutor(16) as ex:
        res = list(ex.map(lss_tile_check, [(str(gd / t), bytile.get(t, [])) for t in todo], chunksize=4))
    bad = [(t, r["return_codes"], r["error_examples"]) for t, ok, r in res if not ok]
    if bad:
        raise SystemExit(f"g{k}: Loc2ssst failed in {len(bad)} tiles, e.g. {bad[:3]}")
    recs = [json.loads((gd / t / "loc2ssst.json").read_text()) for t in tiles.index]
    summ = dict(char_dist_km=SCHEDULE[k], tiles=len(recs), instances=sum(len(r["station_sets"]) for r in recs),
                events_accepted_sum=sum(r["events_accepted"] for r in recs),
                tiles_no_accepted=sum(r["events_accepted"] == 0 for r in recs),
                corrected_grids=sum(r["corrected_grids"] for r in recs),
                kept_previous=sum(r["kept_previous"] for r in recs),
                negative_nodes=sum(r["negative_nodes"] for r in recs),
                tiles_with_negative=sum(r["negative_nodes"] > 0 for r in recs),
                t_min=min((r["t_min"] for r in recs if r["t_min"] is not None), default=None),
                tiles_partial_input=sum(bool(r["tiles_partial"]) for r in recs),
                events_dropped_arrival_limit=sum(r["events_dropped"] for r in recs),
                n_max=N_MAX, tiles_capped=sum(bool(r["tiles_partial"]) for r in recs),
                peak_rss_gb_max=max((x for r in recs for x in (r.get("peak_rss_gb") or []) if x is not None), default=None),
                events_in_max=max(r.get("events_in", 0) for r in recs),
                max_arrivals_station_phase=max(r["max_arrivals_station_phase"] for r in recs),
                wall_min=round((time.time() - t0) / 60, 1), loc2ssst_md5=md5(LOC2SSST), utc=time.strftime("%FT%T%z"))
    (gd / "summary.json").write_text(json.dumps(summ, indent=1))
    log(f"g{k}: {json.dumps({k_: v for k_, v in summ.items() if k_ not in ('utc',)})}")
    if k >= 1:
        prune(root, k)


# ---------------------------------------------------------------------------------------------------------- coherence
def event_files(rd, pattern="loc.*_v4_*.grid0.loc.hyp"):
    out = {}
    for f in rd.glob(f"*/{pattern}"):
        m = re.search(r"_v4_(\d+)\.grid0\.loc\.hyp$", f.name)
        if m:
            out[int(m.group(1))] = f
    return out


SEP_MAX, NACC_MIN = 3.0, 3          # coherence partner rule R3 (coordinator 20:00; notes/v6_plan.md)
RULECHECK = R6 / "heldout" / "coh_rule_check.json"


def pair_coherence(cc_dir, check=True):
    """coherence partners by rule R3 from cc_dir/pair_coherence.parquet (coh = median cc_max over the measured rows,
    n_meas, n_accepted, sep_km): sep_km <= SEP_MAX, n_accepted >= NACC_MIN, n_meas >= COH_NMIN (and coh >= COH_MIN,
    applied by the caller). Pairs with rows straddling a confirmed clock epoch are recomputed from their cc_<block>.parquet
    rows without those rows (streamed in batches: the full loader needs 25 GB for gorda); with check, the recomputation
    including the straddling rows must reproduce pair_coherence.parquet for those pairs."""
    import pyarrow.parquet as pq
    pcf = Path(cc_dir) / "pair_coherence.parquet"
    pc = pd.read_parquet(pcf, columns=["ID1", "ID2", "sep_km", "n_meas", "n_accepted", "coh"])
    affected, blocks = set(), []
    for f in sorted(Path(cc_dir).glob("cc_*.parquet")):
        hit = False
        for b in pq.ParquetFile(f).iter_batches(batch_size=2_000_000, columns=["ID1", "ID2", "straddles_clock_epoch"]):
            m = b.column("straddles_clock_epoch").to_numpy(zero_copy_only=False).astype(bool)
            if m.any():
                affected.update(zip(b.column("ID1").to_numpy()[m].tolist(), b.column("ID2").to_numpy()[m].tolist()))
                hit = True
        if hit:
            blocks.append(f)
    rows = []
    keys = pd.MultiIndex.from_tuples(sorted(affected), names=["ID1", "ID2"]) if affected else None
    for f in blocks:
        for b in pq.ParquetFile(f).iter_batches(batch_size=2_000_000, columns=["ID1", "ID2", "status", "edge", "above",
                                                                               "cc_max", "straddles_clock_epoch"]):
            d = b.to_pandas()
            k = pd.MultiIndex.from_arrays([d.ID1, d.ID2]).isin(keys)
            if k.any():
                rows.append(d[k])
    info = dict(pairs=len(pc), pairs_with_straddling_rows=len(affected), straddling_rows=0, straddling_accepted=0)
    if rows:
        r = pd.concat(rows, ignore_index=True)
        r["meas"] = r.status.eq("ok") & ~r.edge.astype(bool)
        r["acc"] = r.above.astype(bool)
        r["str"] = r.straddles_clock_epoch.astype(bool)
        info.update(straddling_rows=int(r.str.sum()), straddling_accepted=int((r.str & r.acc).sum()))

        def agg(x):
            m = x[x.meas]
            return pd.DataFrame({"n_meas": m.groupby(["ID1", "ID2"]).size(), "coh": m.groupby(["ID1", "ID2"]).cc_max.median(),
                                 "n_accepted": x[x.acc].groupby(["ID1", "ID2"]).size()}).reindex(keys)
        if check:                   # the definition reproduces pair_coherence.parquet on these pairs (all rows)
            full = agg(r).fillna({"n_meas": 0, "n_accepted": 0})
            ref = pc.set_index(["ID1", "ID2"]).reindex(keys)
            bad = int(((full.n_meas != ref.n_meas) | (full.n_accepted != ref.n_accepted) |
                       ((full.coh - ref.coh).abs() > 1e-9)).sum())
            info["check_mismatch"] = bad
            if bad:
                raise SystemExit(f"pair coherence: the recomputation differs from {pcf} on {bad} pairs")
        new = agg(r[~r.str]).fillna({"n_meas": 0, "n_accepted": 0})
        pc = pc.set_index(["ID1", "ID2"])
        pc.loc[keys, ["n_meas", "n_accepted", "coh"]] = new[["n_meas", "n_accepted", "coh"]].to_numpy()
        pc = pc.reset_index()
    test = (pc.n_meas >= S.COH_NMIN) & (pc.coh >= S.COH_MIN)
    r3 = test & (pc.sep_km <= SEP_MAX) & (pc.n_accepted >= NACC_MIN)
    info.update(pairs_test_rule=int(test.sum()), pairs_r3=int(r3.sum()), sep_max_km=SEP_MAX, n_accepted_min=NACC_MIN,
                pair_coherence_md5=md5(pcf))
    g = pc[r3][["ID1", "ID2", "coh", "n_meas", "n_accepted", "sep_km"]].rename(columns={"n_meas": "n"})
    return g.reset_index(drop=True), info


def len3_of(f):
    st, l3 = None, np.inf
    with open(f, errors="replace") as fh:
        for line in fh:
            if line.startswith("NLLOC "):
                st = re.findall(r'"([^"]*)"', line)[1]
            elif line.startswith("STATISTICS"):
                t = line.split()
                l3 = float(t[t.index("Len3") + 1])
                break
    return st, l3


def _len3_many(fs):
    return [len3_of(f) for f in fs]


def cc_done(cc_dir):
    f = Path(cc_dir) / "state" / "final.json"
    try:
        return bool(json.loads(f.read_text()).get("done"))
    except (OSError, ValueError):
        return False


def cmd_coh(a):
    root = Path(a.root)
    cc_dir = Path(a.cc) if a.cc else CC_DIR
    rd = root / "coh"
    if (rd / "run.json").exists():
        log(f"{rd} done")
        return
    if not cc_done(cc_dir):
        raise SystemExit(f"CC job not finished ({cc_dir}/state/final.json)")
    if a.no_rulecheck:
        check = dict(use="r3", note="rule check skipped (smoke test)")
    else:
        if not RULECHECK.exists():          # R3 against the test rule on puget and mtj (b4_v6_heldout.py rulecheck)
            log("coherence: running the partner-rule check (b4_v6_heldout.py rulecheck)")
            r = subprocess.run([str(S.PY), str(HERE / "b4_v6_heldout.py"), "rulecheck"], env=ENV)
            if r.returncode or not RULECHECK.exists():
                raise SystemExit("coherence: the partner-rule check failed")
        check = json.loads(RULECHECK.read_text())
        if check.get("use") != "r3":
            raise SystemExit(f"coherence: rule R3 is worse than the test rule on held-out MAD ({RULECHECK}): the stage "
                             "waits for a decision (coordinator 20:00: report and wait)")
    it = root / f"it{NPASS}"
    if not (it / "run.json").exists():
        raise SystemExit(f"{it} not finished")
    tiles, jobs = load_plan(root)
    files = event_files(it)
    ids = sorted(files)
    chunks = [[files[i] for i in ids[j:j + 2000]] for j in range(0, len(ids), 2000)]
    with ProcessPoolExecutor(16) as ex:
        res = [r for part in ex.map(_len3_many, chunks) for r in part]
    info = {i: r for i, r in zip(ids, res)}
    located = {i for i, (st, _) in info.items() if st == "LOCATED" and Path(str(files[i])[:-4] + ".octree").exists()}
    compact = {i for i in located if info[i][1] <= S.COH_MAX_SE3}
    # PI 2026-11-25 (D-B098): no coherence relocation in the Mendocino region, where the held-out test found it
    # worse (2015 sequence: P +8, S +20 ms); those events keep their final SSST (it4) location. They may still serve
    # as stacking partners of neighbouring events outside the region. Region = catalog v5.1 label of the v4 epicentre.
    skip_regions = set(os.environ.get("V6_COH_SKIP_REGIONS", "Mendocino").split(",")) - {""}
    reg = pd.read_csv(V51_CATALOG, usecols=["ID", "region"]).set_index("ID").region
    skipped = {i for i in compact if reg.get(i) in skip_regions}
    compact_all = compact
    compact = compact - skipped
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "coherence_skipped_region.csv").write_text(
        "ID,region\n" + "".join(f"{i},{reg.get(i)}\n" for i in sorted(skipped)))
    log(f"coherence: {len(skipped)} compact events skipped in regions {sorted(skip_regions)} (D-B098)")
    coh, cinfo = pair_coherence(cc_dir)
    coh = coh[(coh.coh >= S.COH_MIN) & coh.ID1.isin(located) & coh.ID2.isin(located)]
    nb = {}
    for r in coh.itertuples():
        nb.setdefault(int(r.ID1), []).append((r.coh, int(r.ID2)))
        nb.setdefault(int(r.ID2), []).append((r.coh, int(r.ID1)))
    ev = pd.read_parquet(plan_dir(root) / "events.parquet").set_index("ID")
    tmpl = TEMPLATE.read_text()
    tasks, rows = [], []
    for eid in sorted(compact):
        tile = ev.loc[eid, "tile"]
        t = tiles.loc[tile]
        box = (float(t.x0), float(t.y0), int(t.nx), int(t.ny))
        froot = str(files[eid])[:-4]
        wd = rd / tile / f"e{eid}"
        partners = sorted(nb.get(eid, []), reverse=True)
        n_stack = sum(1 for _, o in partners if o in compact_all)
        rows.append((eid, tile, len(partners), n_stack))
        if (wd / "done").exists():
            continue
        if wd.exists():
            shutil.rmtree(wd)
        wd.mkdir(parents=True)
        lines = ["STREAM_COHERENCES", f"  1.000000 {froot}"] + [f"  {c:.6f} {str(files[o])[:-4]}" for c, o in partners]
        sc = wd / "event.stream_coherences"
        sc.write_text("\n".join(lines) + "\n")
        groot = f"{root}/g{NPASS - 1}/{tile}/cascadia3d"
        opt = make_opt((groot,), tmpl)
        ctl = RN.control(box, f"{froot}.hyp", wd / "loc", None, opt)
        ctl = RN.sub1(r"^LOCFILES .*$", f"LOCFILES {froot}.hyp NLLOC_OBS {groot} {wd}/loc", ctl)
        ctl += f"LOCPOSTERIOR OCT_TREE {sc} 1e-30 {S.COH_MIN} -1.0 {S.COH_MAX_SE3} -1 -1.0 -999\n"
        (wd / "nlloc.in").write_text(ctl)
        # memory: ~0.23 GB plus ~6 MB per listed partner oct-tree (measured on the held-out Axial runs)
        tasks.append((f"{tile}/e{eid}", [NLL, "nlloc.in"], wd, wd / "nlloc.log", 0.3 + 0.0065 * len(partners)))

    def done(name, r):
        if r == 0:
            (rd / name / "done").write_text("")
    tasks.sort(key=lambda x: -x[4])                    # largest stacks first; small ones fill in while memory allows
    t0 = time.time()
    log(f"coherence: {len(compact)} compact events of {len(located)} located (Len3 <= {S.COH_MAX_SE3} km), "
        f"{len(coh)} pairs with coherence >= {S.COH_MIN}, {len(tasks)} NLLoc runs to do")
    rc = run_pool(tasks, "coherence", a.nproc, ramp=0.02, on_done=done)
    bad = sorted(n for n, r in rc.items() if r != 0)
    tab = pd.DataFrame(rows, columns=["ID", "tile", "partners", "partners_compact"])
    tab.to_csv(rd / "coherence_events.csv", index=False)
    summ = dict(events_final=len(files), located_with_octree=len(located), compact=len(compact),
                skipped_region=len(skipped), skip_regions=sorted(skip_regions), decision_skip="D-B098",
                compact_with_compact_partner=int((tab.partners_compact > 0).sum()), pairs_used=len(coh),
                coh_min=S.COH_MIN, coh_nmin=S.COH_NMIN, max_se3=S.COH_MAX_SE3, cc_dir=str(cc_dir), **cinfo,
                partner_rule="R3", rule_check={k: check.get(k) for k in ("use", "decision_rule", "utc", "note")},
                failed=bad[:50], n_failed=len(bad), wall_min=round((time.time() - t0) / 60, 1),
                grids=f"{root}/g{NPASS - 1}", utc=time.strftime("%FT%T%z"))
    if bad:
        raise SystemExit(f"coherence: {len(bad)} NLLoc runs failed, e.g. {bad[:5]}")
    (rd / "run.json").write_text(json.dumps(summ, indent=1))
    log(f"coherence: {json.dumps({k: v for k, v in summ.items() if k != 'failed'})}")


# ------------------------------------------------------------------------------------------------- static terms (it0)
def _phases_terms(args):
    """phase rows (ID, sta, phase, res, weight, tcorr) of LOCATED it0 events passing the a2_station_terms.py rule."""
    fs = args
    out = []
    for f in fs:
        ok, eid, q, inph = False, None, {}, False
        rows = []
        for line in open(f, errors="replace"):
            if line.startswith("NLLOC "):
                ok = '"LOCATED"' in line
            elif line.startswith("PUBLIC_ID"):
                eid = int(line.split()[1].split("_")[1])
            elif line.startswith("QUALITY"):
                t = line.split()
                q = dict(rms=float(t[t.index("RMS") + 1]), nphs=int(t[t.index("Nphs") + 1]), gap=float(t[t.index("Gap") + 1]))
            elif line.startswith("PHASE ID"):
                inph = True
            elif line.startswith("END_PHASE"):
                inph = False
            elif inph and ok and ">" in line:
                a_, b_ = line.split(">")
                t, u = a_.split(), b_.split()
                rows.append((eid, t[0], t[4], float(u[1]), float(u[2]), float(u[11])))
        if ok and q and q["nphs"] >= 8 and q["gap"] <= 270 and q["rms"] <= 0.6:
            out += rows
    return out


def cmd_terms(a):
    root = Path(a.root)
    od = root / "terms"
    if (od / "station_terms.in").exists():
        log(f"{od} done")
        return
    it0 = root / "it0"
    files = event_files(it0)
    ev = pd.read_parquet(plan_dir(root) / "events.parquet").set_index("ID")
    fl = sorted(files.items())
    chunks = [[f for _, f in fl[j:j + 2000]] for j in range(0, len(fl), 2000)]
    with ProcessPoolExecutor(16) as ex:
        rows = [r for part in ex.map(_phases_terms, chunks) for r in part]
    ph = pd.DataFrame(rows, columns=["ID", "sta", "phase", "res", "w", "tcorr"])
    ph = ph[(ph.w > 0) & ph.phase.isin(["P", "S"])]
    base = {}
    for tile in ev.tile.unique():
        base[tile] = json.loads((root / "base" / tile / "base.json").read_text())
    tl = ev.tile.reindex(ph.ID).to_numpy()
    kind = np.array([base[t].get(f"{p}.{s}") for t, p, s in zip(tl, ph.phase, ph.sta)])
    n_wide = int((kind != "fine").sum())
    ph = ph[kind == "fine"]
    st = ph.groupby(["sta", "phase"]).agg(n=("res", "size"), med_res=("res", "median"),
                                          mad=("res", lambda s: np.median(np.abs(s - s.median()))),
                                          tcorr_old=("tcorr", "median")).reset_index()
    st = st[st.n >= 10].copy()
    st["delay"] = (st.tcorr_old + st.med_res).round(4)
    od.mkdir(parents=True, exist_ok=True)
    st.to_csv(od / "station_terms.csv", index=False)
    (od / "station_terms.in").write_text(
        "\n".join(f"LOCDELAY {r.sta} {r.phase} {r.n} {r.delay:.4f}" for r in st.itertuples()) + "\n")
    log(f"static terms from it0: {len(ph)} fine-grid residuals ({n_wide} wide left out), {len(st)} station-phase terms")


# ---------------------------------------------------------------------------------------------- deep and edge passes
def cmd_fallback(a):
    root = Path(a.root)
    it = root / f"it{NPASS}"
    stat = root / "terms" / "station_terms.in"
    if not stat.exists():
        raise SystemExit(f"{stat} missing (terms stage)")
    s = pass_summary(it).set_index("ID")
    ev = pd.read_parquet(plan_dir(root) / "events.parquet").set_index("ID")
    far = ev.n_fine_base.reindex(s.index).fillna(0).eq(0)
    deep = ((s.status.eq("REJECTED") & s.boundary.eq(BOTTOM)) | (s.status.eq("LOCATED") & (s.z >= DEEP_Z)))
    edge = s.status.eq("REJECTED") & s.boundary.isin(LATERAL)
    sel = pd.DataFrame({"status": s.status, "boundary": s.boundary, "z_ml": s.z, "far_only": far,
                        "trigger": np.select([deep, edge], ["deep", "edge"], "")})
    sel = sel[sel.trigger != ""].copy()
    sel["selected"] = ~(sel.far_only & (sel.trigger == "deep"))
    obs_tiles = pd.read_csv(OBS / "tiles.csv").set_index("tile")
    tiles, jobs = load_plan(root)
    roots = (f"{FINE}/cascadia3d", f"{WIDE}/cascadia3d")
    RN.check_wide_manifest(str(FINE), str(WIDE))
    for kind in ("deep", "edge"):
        rd = root / kind
        if (rd / "run.json").exists():
            log(f"{rd} done")
            continue
        want = sel[(sel.trigger == kind) & sel.selected]
        tmpl = template_text(octree=False)
        opt = make_opt(roots, tmpl)
        specs = []
        for tile, g in want.groupby(ev.tile.reindex(want.index)):
            ids = set(g.index)
            blocks = [b for j in [n for n, jj in jobs.items() if jj["tile"] == tile]
                      for b in read_blocks(plan_dir(root) / "obs" / f"{j}.obs") if RN.block_arrivals(b)[0] in ids]
            ot = obs_tiles.loc[tile]
            box = (float(ot.box_x0), float(ot.box_y0), int(round(ot.box_x1 - ot.box_x0)) + 1,
                   int(round(ot.box_y1 - ot.box_y0)) + 1)
            for c in range(0, len(blocks), CHUNK):
                name = f"{tile}_c{c // CHUNK:02d}"
                wd = rd / name
                specs.append((name, RN.control(box, wd / "events.obs", wd / "loc", str(stat), opt, deep=(kind == "deep")),
                              "\n\n".join(blocks[c:c + CHUNK]) + "\n\n"))
        rd.mkdir(parents=True, exist_ok=True)
        sel[sel.trigger == kind].to_csv(rd / f"{kind}_selection.csv")
        if not specs:
            (rd / "run.json").write_text(json.dumps(dict(kind=kind, selected=0, events=0), indent=1))
            log(f"{kind} pass: no event selected")
            continue
        count = RN.check_grids(specs, roots)
        cfg = dict(kind=kind, time=roots[0], time_wide=roots[1],
                   fine_manifest_md5=md5(RN.fine_manifest(str(FINE))), wide_manifest_md5=md5(RN.wide_manifest(str(WIDE))),
                   grid_manifest_md5=grid_manifest_md5(), template_md5=md5_text(tmpl), locgau=LOCGAU, locgau2=LOCGAU2,
                   topo=FLOOR, topo_md5=md5(FLOOR), stat=str(stat), stat_md5=md5(stat), nll_md5=md5(NLL),
                   nz=RN.NZ_DEEP if kind == "deep" else RN.NZ, from_pass=str(it), arrivals_by_header_rule=
                   {str(k_): v for k_, v in count.items()})
        cp = rd / "run_config.json"
        if cp.exists() and json.loads(cp.read_text()) != cfg:
            shutil.rmtree(rd)
            rd.mkdir(parents=True)
            sel[sel.trigger == kind].to_csv(rd / f"{kind}_selection.csv")
        cp.write_text(json.dumps(cfg, indent=1))
        tasks = []
        for name, ctl, text in specs:
            wd = rd / name
            if (wd / "done").exists() and (wd / "nlloc.in").read_text() == ctl:
                continue
            if wd.exists():
                shutil.rmtree(wd)
            wd.mkdir(parents=True)
            (wd / "events.obs").write_text(text)
            (wd / "nlloc.in").write_text(ctl)
            tasks.append((name, [NLL, "nlloc.in"], wd, wd / "nlloc.log"))

        def done(name, r, rd=rd):
            if r == 0:
                (rd / name / "done").write_text("")
        rc = run_pool(tasks, f"{kind} pass", a.nproc, ramp=0.5, on_done=done, est_default=1.6)   # full fine/wide grids
        bad = sorted(n for n, r in rc.items() if r != 0)
        if bad:
            raise SystemExit(f"{kind} pass: {len(bad)} NLLoc jobs failed: {bad[:5]}")
        nprob = RN.account(rd, kind, 16)
        if nprob:
            raise SystemExit(f"{kind} pass: {nprob} undocumented arrival drops ({rd}/arrival_problems.csv)")
        got = pass_summary(rd)
        miss = sorted(set(want.index) - set(got.ID))
        if miss:
            raise SystemExit(f"{kind} pass incomplete: {len(miss)} selected events without a row, e.g. {miss[:5]}")
        if kind == "deep":
            au = pd.read_csv(rd / "arrivals_used.csv")
            if (au.n_fine > 0).any():
                raise SystemExit("deep pass: arrivals served by a fine grid (the deep box must use the wide grids only)")
        st = got.status.value_counts().to_dict()
        (rd / "run.json").write_text(json.dumps(dict(kind=kind, triggered=int((sel.trigger == kind).sum()),
                                                     selected=len(want), far_only_not_rerun=int(((sel.trigger == kind) & ~sel.selected).sum()),
                                                     events=len(got), status=st, utc=time.strftime("%FT%T%z")), indent=1))
        log(f"{kind} pass: {len(want)} events, status {st}")


# --------------------------------------------------------------------------------------------------------- gridcheck
def _grid_stats(path):
    h = Path(str(path)[:-3] + "hdr").read_text().split()
    nx, ny, nz = (int(v) for v in h[:3])
    t = np.fromfile(path, "<f4")
    if t.size != nx * ny * nz:
        return str(path), "size", None, None
    fin = np.isfinite(t)
    return str(path), "ok", float(t[fin].min()) if fin.any() else None, int((~fin).sum())


def cmd_gridcheck(a):
    """before any relocation: the grid manifest (stage manifest) lists the fine and wide P and S grid (hdr + buf) of every
    station, every buffer is complete, and no grid holds a negative or non-finite travel time."""
    gt = Path("/wd1/mdenolle_data/reloc_v5/model_vpvs176/stations_GTSRCE.in")
    st = [l.split()[1] for l in gt.read_text().splitlines() if l.startswith("GTSRCE")]
    want = {f"{d}/cascadia3d.{ph}.{s}.time.{e}" for d in (FINE.name, WIDE.name) for ph in ("P", "S") for s in st
            for e in ("hdr", "buf")}
    have = {}
    for line in GRID_MD5.read_text().splitlines():
        h, p = line.split(None, 1)
        have[p.strip()] = h
    miss, extra = sorted(want - set(have)), sorted(set(have) - want)
    if miss or extra:
        raise SystemExit(f"grid manifest {GRID_MD5}: {len(miss)} grids missing (e.g. {miss[:3]}), {len(extra)} extra "
                         f"(e.g. {extra[:3]})")
    bufs = sorted(R6 / p for p in want if p.endswith(".buf"))
    with ProcessPoolExecutor(16) as ex:
        res = list(ex.map(_grid_stats, bufs, chunksize=8))
    bad = [r for r in res if r[1] != "ok"]
    neg = [r for r in res if r[2] is not None and r[2] < 0]
    nonfin = [r for r in res if r[3]]
    out = dict(manifest=str(GRID_MD5), manifest_md5=md5(GRID_MD5), grids=len(bufs), stations=len(st),
               incomplete=[r[0] for r in bad], negative=[(r[0], r[2]) for r in neg],
               non_finite=[(r[0], r[3]) for r in nonfin], t_min=min(r[2] for r in res if r[2] is not None),
               fine_manifest_md5=md5(RN.fine_manifest(str(FINE))), wide_manifest_md5=md5(RN.wide_manifest(str(WIDE))),
               utc=time.strftime("%FT%T%z"))
    (R6 / f"grids_{TIME}.check.json").write_text(json.dumps(out, indent=1))
    log(f"gridcheck: {len(bufs)} grids, incomplete {len(bad)}, negative {len(neg)}, non-finite {len(nonfin)}, "
        f"t_min {out['t_min']:.4f} s, manifest md5 {out['manifest_md5']}")
    if bad or neg or nonfin:
        raise SystemExit("gridcheck failed")


# --------------------------------------------------------------------------------------------------- tile positions
V51_CATALOG = Path("/wd1/mdenolle_data/catalog_v5/v5.1/cascadia_catalog_v5.1.csv")
V41_EVENTS = Path("/home/mdenolle/cascadia_obs_ensemble/data/catalog_v4/Cascadia_relocated_catalog_v4.csv")


def cmd_tilepos(a):
    """events file for a2_make_obs.py --events: the catalog v4.1 event list with latitude/longitude replaced by the
    catalog v5.1 epicentre where v5.1 has one (v4 otherwise): the tile (and SSST box) of each event is placed where v5.1
    located it, as the cluster tests placed their boxes around the v5.1 epicentres."""
    v4 = pd.read_csv(V41_EVENTS, skipinitialspace=True)
    v4.columns = [c.strip() for c in v4.columns]
    c = pd.read_csv(V51_CATALOG, usecols=["ID", "lat", "lon", "loc_source"]).set_index("ID")
    ok = c.loc_source.isin(["nll3d", "comcat"]) & c.lat.notna() & c.lon.notna()
    lat = c.lat[ok].reindex(v4.ID)
    lon = c.lon[ok].reindex(v4.ID)
    use = lat.notna().to_numpy()
    out = pd.DataFrame({"ID": v4.ID, "latitude": np.where(use, lat.to_numpy(), v4.latitude),
                        "longitude": np.where(use, lon.to_numpy(), v4.longitude),
                        "pos_source": np.where(use, "v5.1", "v4")})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False, float_format="%.6f")
    (Path(a.out).with_suffix(".json")).write_text(json.dumps(dict(
        events=str(V41_EVENTS), events_md5=md5(V41_EVENTS), v51=str(V51_CATALOG), v51_md5=md5(V51_CATALOG),
        n=len(out), from_v51=int(use.sum()), from_v4=int((~use).sum())), indent=1))
    log(f"tile positions: {len(out)} events, {int(use.sum())} at the v5.1 epicentre, {int((~use).sum())} at v4 -> {a.out}")


# ------------------------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    x = sub.add_parser("tilepos")
    x.add_argument("out")
    sub.add_parser("gridcheck")
    for c in ("plan", "nll", "lss", "coh", "terms", "fallback"):
        x = sub.add_parser(c)
        x.add_argument("root")
        if c in ("nll", "lss"):
            x.add_argument("k", type=int)
        if c == "plan":
            x.add_argument("--tiles")
        if c == "coh":
            x.add_argument("--cc")
            x.add_argument("--no-rulecheck", action="store_true", help="smoke test only")
        x.add_argument("--nproc", type=int, default=int(os.environ.get("NPROC_NLL", "40")))
    a = ap.parse_args()
    {"tilepos": cmd_tilepos, "gridcheck": cmd_gridcheck, "plan": cmd_plan, "nll": cmd_nll, "lss": cmd_lss, "coh": cmd_coh, "terms": cmd_terms,
     "fallback": cmd_fallback}[a.cmd](a)


if __name__ == "__main__":
    main()

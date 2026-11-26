"""Track B, catalog v6.0 preparation (D-B079, D-B092): NLL-SSST against static station terms on the four D-B078 test
clusters (session -18's /wd1/mdenolle_data/tomo_v1/b5/cc_test/events_<set>.csv: axial, blanco, msh, jdf).

NLL-SSST (Lomax & Savvaidis 2022, JGR 127, e2021JB023190) with the NonLinLoc build the relocations use
(/wd1/mdenolle_data/reloc_v5/src/NonLinLoc, ut-beg-texnet/NonLinLoc 3eb0c1e5, PVER 7.1.07): its Loc2ssst program
computes, per station-phase, a source-specific correction on a 3D grid, the Gaussian-weighted mean of the residuals
(+ any delay) of the located events, weight exp(-d^2 / CharDist^2) + WeightFloor with d the event-node distance, and
writes travel-time grids = input time + correction, which the next NLLoc iteration reads as its time grids. The
iterations follow the published example (NLL-SSST-coherence Parkfield example v9.0, run_ssst_relocations.bash):
iteration 0 without corrections, Loc2ssst with CharDist 9999 km (one static per station-phase), then CharDist halving
from a start of about the size of the seismicity (SCHEDULE), WeightFloor 1e-7, each iteration reading the previous
iteration's corrected grids (corrections accumulate). Nothing in NonLinLoc is changed.

Set-up per cluster (both arms identical except the station corrections):
* picks: the D-B090 clock-corrected pick file (PICKS, md5 checked), the cluster's events, a2_make_obs.py (pick errors
  as production);
* search box (LOCGRID, also Loc2ssst's LSOUTGRID): the cluster's v5.1 epicentres +- MARGIN km on the 1 km lattice,
  z' -5..80 km (the production depth range); production uses the 25 km tile +- 75 km, so the static arm is also
  compared with the v5.1 catalog to show the box effect;
* NLLoc as v5.1 (a2_run_nlloc.control and nlloc_template.in: EDT_OT_WT, LOCGAU 0.2 0.0 without LOCGAU2 (D-B069),
  minimum 5 phases (D-B066), merged-surface floor (D-B056/D-B068)); the arrival accounting of a2_run_nlloc.account;
* grids: GRIDS[label]: the fine grids, and the wide grids for station-phases whose fine grid does not contain the box.

Arms:
* static: the production static terms (D-B006/D-B014 statics with the D-B069 model error; for the v5.1 grids the
  v5.1 file b4_v51_iter0/station_terms.in, computed on the whole catalog), grids as production (fine, then wide);
* ssst: it0 (no terms) -> Loc2ssst(9999) -> it1 -> Loc2ssst(16) -> ... -> it<n> (final, with SAVE_NLLOC_OCTREE for the
  coherence stage). Loc2ssst runs NPROC instances on disjoint station sets (LSSTATIONS, as the example's run_ssst.bash);
  every Loc2ssst output folder first gets links to the previous grids, so a station-phase without accepted arrivals
  keeps its previous grid (as the example). LSGRID covers LSOUTGRID with nodes every LSGRID_KM(CharDist) km.
  Checks per Loc2ssst run: no ERROR in the logs, no corrected node outside an input grid (Loc2ssst writes 0 there and
  logs "DEBUG ERROR"), the negative corrected times counted (NLLoc rejects a negative predicted time silently).
* coherence (NLL-coherence, Lomax & Savvaidis 2022): per event NLLoc with LOCPOSTERIOR OCT_TREE stacking the final
  SSST pdfs (octree files) of the events whose waveform coherence with it is >= COH_MIN, coherence from the
  D-B040/D-B078 CC files (cc_<set>.parquet: median cc_max over the pair's measured station-phases, >= COH_NMIN rows),
  in place of the example's multi-second stream coherency (labelled as such in the report).

Outputs: /wd1/mdenolle_data/reloc_v6/ssst_test/<grids label>/<set>/{obs, grids_base, static, ssst/it<k>, ssst/g<k>,
coherence}; each NLLoc run folder has all.sum.grid0.loc.hyp, arrivals_used.csv, run.json.

  python b4_v6_ssst.py picks                        # picks of the four sets from PICKS (once)
  python b4_v6_ssst.py run SET [--grids g51] [--arms static,ssst,coherence] [--nproc 4]
"""
import argparse
import hashlib
import json
import os
import re
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
from manifest import load_params  # noqa: E402

R5 = Path("/wd1/mdenolle_data/reloc_v5")
R6 = Path("/wd1/mdenolle_data/reloc_v6")
OUT = R6 / "ssst_test"
BIN = R5 / "src/NonLinLoc/src/bin"
SETS_DIR = Path("/wd1/mdenolle_data/tomo_v1/b5/cc_test")
CC_DIR = Path("/wd1/mdenolle_data/tomo_v1/cc_dt")
PICKS = Path("/wd1/mdenolle_data/tomo_v1/checks/pick_timing_v6/Cascadia_picks_catalog_v4.1tc.csv")
PICKS_MD5 = "1b21ab1c2cf5aba8b48c64a58fc37356"
EVENTS = Path("/home/mdenolle/cascadia_obs_ensemble/data/catalog_v4/Cascadia_relocated_catalog_v4.csv")
PY = Path("/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/bin/python")
FLOOR = load_params()["b4"]["surface"]["floor"]["file"]
GRIDS = {
    "g51": dict(fine=R5 / "time_b4_v0.3.1m_sc21", wide=R5 / "wide_time_b4_v0.3.1m_sc21",
                static=R5 / "runs/b4_v51_iter0/station_terms.in",
                about="v5.1 production grids (model v0.3.1 + D-B073 field rule): method test, not the v6.0 model"),
    "g041": dict(fine=R6 / "time_v6_v0.4.1", wide=R6 / "wide_time_v6_v0.4.1", static=None,
                 about="v6.0 grids (model v0.4.1)"),
}
SETS = ("axial", "blanco", "msh", "jdf")
MARGIN = 30.0                       # km around the cluster's v5.1 epicentres (test box, both arms)
NZ, Z0 = RN.NZ, RN.Z0               # z' -5..80 km, 1 km nodes (production)
SCHEDULE = (9999.0, 16.0, 8.0, 4.0, 2.0)    # Loc2ssst CharDist per SSST iteration (example: 9999, start, /2, min 1)
WEIGHT_FLOOR = 1e-7                 # example's LSPARAMS WeightFloor
# LSPHSTAT: RMSMax NRdgsMin GapMax PResidualMax SResidualMax EllLen3Max (events and residuals that enter the
# corrections). Example (Parkfield, 0.05 s model error): 0.35 12 135 1.0 2.0 5.0; here offshore networks (gap often
# > 180) and the 0.2 s model error of D-B069: RMS <= 0.6 s, >= 6 readings, gap <= 300, |res| P <= 1, S <= 2 s, len3
# <= 30 km.
# EllLen3Max 30 km, not 10: the Blanco and JdF clusters are tier C (v5.1 len3 median 12.9 and 24.7 km; Axial 2.9, MSH
# 1.8), so 10 km would leave JdF without any event and Blanco with a third of its events.
LSPHSTAT = "0.6 6 300.0 1.0 2.0 30.0"
COH_MIN, COH_NMIN = 0.5, 3          # coherence stage: event coherence threshold (example 0.5), CC rows per pair
COH_MAX_SE3 = 5.0                   # example's MaxSE3 (km)
LOCGAU, LOCGAU2 = "0.2 0.0", "off"  # D-B069


def lsgrid_km(char):
    """Loc2ssst LSGRID node spacing for a CharDist (the correction is trilinear between its nodes)."""
    return 10.0 if char >= 1000 else max(1.0, min(4.0, char / 2.0))


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------------------------------------------------ inputs
def cmd_picks(a):
    out = OUT / "inputs"
    out.mkdir(parents=True, exist_ok=True)
    h = md5(PICKS)
    assert h == PICKS_MD5, f"{PICKS} md5 {h}, expected {PICKS_MD5}"
    ids = {s: set(pd.read_csv(SETS_DIR / f"events_{s}.csv").ID) for s in SETS}
    allid = set().union(*ids.values())
    parts = []
    for ch in pd.read_csv(PICKS, skipinitialspace=True, chunksize=500_000):
        parts.append(ch[ch.event_ID.isin(allid)])
    p = pd.concat(parts)
    ev = pd.read_csv(EVENTS, skipinitialspace=True)
    for s in SETS:
        p[p.event_ID.isin(ids[s])].to_csv(out / f"picks_{s}.csv", index=False)
        ev[ev.ID.isin(ids[s])].to_csv(out / f"events_{s}.csv", index=False)
        log(s, len(ids[s]), "events,", int(p.event_ID.isin(ids[s]).sum()), "picks")
    (out / "inputs.json").write_text(json.dumps({"picks": str(PICKS), "picks_md5": h, "events": str(EVENTS),
                                                 "events_md5": md5(EVENTS), "sets": str(SETS_DIR)}, indent=1))


def lambert():
    from pyproj import Proj
    return Proj(proj="lcc", lat_0=45.0, lon_0=-125.8, lat_1=41.0, lat_2=49.0, ellps="WGS84")


def prep(s, g):
    """obs folder (a2_make_obs.py) and the search box of set s with grids g; returns (blocks, box)."""
    d = OUT / g / s
    obs = d / "obs"
    if not (obs / "obs_meta.json").exists():
        assert not obs.exists() or not any(obs.iterdir()), f"{obs} incomplete: remove it"
        env = dict(os.environ, LD_LIBRARY_PATH=str(PY.parents[1] / "lib"))
        subprocess.run([str(PY), str(TA / "a2_make_obs.py"), "--picks", str(OUT / "inputs" / f"picks_{s}.csv"),
                        "--events", str(OUT / "inputs" / f"events_{s}.csv"), "--out", str(obs),
                        "--time", str(GRIDS[g]["fine"])], check=True, env=env, cwd=TA)
    blocks = []
    for f in sorted(obs.glob("*.obs")):
        blocks += [b.strip("\n") for b in f.read_text().split("\n\n") if b.strip()]
    ev = pd.read_csv(SETS_DIR / f"events_{s}.csv")
    x, y = lambert()(ev.lon.values, ev.lat.values)
    x0, x1 = np.floor(x.min() / 1e3 - MARGIN), np.ceil(x.max() / 1e3 + MARGIN)
    y0, y1 = np.floor(y.min() / 1e3 - MARGIN), np.ceil(y.max() / 1e3 + MARGIN)
    x0, y0 = max(x0, -513.0), max(y0, -709.0)
    x1, y1 = min(x1, 457.0), min(y1, 724.0)
    box = (float(x0), float(y0), int(x1 - x0) + 1, int(y1 - y0) + 1)
    return blocks, box


def station_phases(blocks):
    sp = {}
    for b in blocks:
        for ph, sta in RN.block_arrivals(b)[1]:
            sp[(ph, sta)] = sp.get((ph, sta), 0) + 1
    return sp


def box_ext(box):
    x0, y0, nx, ny = box
    return (x0, x0 + nx - 1, y0, y0 + ny - 1, Z0, Z0 + NZ - 1)


def base_grids(s, g, blocks, box):
    """grids_base: per station-phase of the obs, a link to the fine grid if it contains the box, else to the wide grid
    (the grid NLLoc's header rule would take with LOCFILES fine,wide)."""
    d = OUT / g / s / "grids_base"
    roots = (f"{GRIDS[g]['fine']}/cascadia3d", f"{GRIDS[g]['wide']}/cascadia3d")
    ext = box_ext(box)
    used = {}
    d.mkdir(parents=True, exist_ok=True)
    for (ph, sta) in sorted(station_phases(blocks)):
        i = RN.serving_root(ext, (ph, sta), roots)
        assert i is not None, f"no grid contains the box for {ph}.{sta}"
        src = Path(roots[i])
        for e in ("hdr", "buf"):
            ln = d / f"cascadia3d.{ph}.{sta}.time.{e}"
            tgt = src.parent / f"cascadia3d.{ph}.{sta}.time.{e}"
            if not ln.exists():
                ln.symlink_to(tgt)
            assert os.path.realpath(ln) == str(tgt.resolve()), ln
        used[f"{ph}.{sta}"] = ("fine", "wide")[i]
    (d / "base.json").write_text(json.dumps(used, indent=1))
    return d


# ------------------------------------------------------------------------------------------------------------- NLLoc
def nll_run(rd, blocks, box, roots, stat=None, octree=False, nproc=4):
    """NLLoc on the blocks in nproc jobs (box, roots, station terms); a2_run_nlloc's control file and accounting."""
    done = rd / "run.json"
    if done.exists():
        return rd
    rd.mkdir(parents=True, exist_ok=True)
    tmpl = (TA / "nlloc_template.in").read_text()
    if octree:
        tmpl = RN.sub1(r"^LOCHYPOUT .*$", "LOCHYPOUT SAVE_NLLOC_ALL NLL_FORMAT_VER_2 SAVE_NLLOC_OCTREE", tmpl)
    opt = argparse.Namespace(time=str(roots[0]), roots=tuple(str(r) for r in roots), template_text=tmpl,
                             locgau=LOCGAU, locgau2=LOCGAU2, topo=FLOOR, max_dist=RN.MAX_DIST)
    n = -(-len(blocks) // nproc)
    jobs = []
    for i in range(nproc):
        bl = blocks[i * n:(i + 1) * n]
        if not bl:
            continue
        wd = rd / f"job{i:02d}"
        wd.mkdir(exist_ok=True)
        (wd / "events.obs").write_text("\n\n".join(bl) + "\n\n")
        jobs.append((wd, RN.control(box, wd / "events.obs", wd / "loc", str(stat) if stat else None, opt)))
    t0 = time.time()
    with ProcessPoolExecutor(len(jobs)) as ex:
        bad = [name for name, rc in ex.map(RN.run_one, jobs) if rc != 0]
    if bad:
        raise SystemExit(f"NLLoc failed in {rd}: {bad}")
    nprob = RN.account(rd, "main", nproc)
    if nprob:
        raise SystemExit(f"{rd}: {nprob} undocumented arrival drops (arrival_problems.csv)")
    with open(rd / "all.sum.grid0.loc.hyp", "w") as f:
        for wd, _ in jobs:
            f.write((wd / "loc.sum.grid0.loc.hyp").read_text())
    done.write_text(json.dumps({"box": box, "roots": [str(r) for r in roots], "stat": str(stat) if stat else None,
                                "stat_md5": md5(stat) if stat else None, "octree": octree, "nproc": nproc,
                                "events": len(blocks), "wall_s": round(time.time() - t0, 1), "floor": FLOOR,
                                "locgau": LOCGAU, "locgau2": LOCGAU2, "nll_md5": md5(RN.NLL)}, indent=1))
    log(f"NLLoc {rd}: {len(blocks)} events, {time.time() - t0:.0f} s")
    return rd


# ---------------------------------------------------------------------------------------------------------- Loc2ssst
def station_sets(sp, n):
    """n disjoint station sets for LSSTATIONS, balanced by arrivals. Loc2ssst selects a station when its code is a
    substring of the set's string, so a station whose code is inside another set's string is moved into that set."""
    tot = {}
    for (ph, sta), k in sp.items():
        tot[sta] = tot.get(sta, 0) + k
    sets = [[] for _ in range(n)]
    load = [0] * n
    for sta, k in sorted(tot.items(), key=lambda kv: -kv[1]):
        i = int(np.argmin(load))
        sets[i].append(sta)
        load[i] += k
    for _ in range(10):
        moved = False
        strs = ["=" + "=".join(x) + "=" for x in sets]
        for i, x in enumerate(sets):
            for sta in list(x):
                for j, sj in enumerate(strs):
                    if j != i and sta in sj:
                        x.remove(sta)
                        sets[j].append(sta)
                        moved = True
                        break
        if not moved:
            break
    assert not moved, "station sets: substring collisions not resolved"
    return [x for x in sets if x]


def link_hyps(it_dir):
    """it_dir/hyp/: links to the per-event .hyp files of every job (Loc2ssst reads phase lines, which the summary file
    does not have, and expands wildcards only in the file name). Returns (folder, per-event files, LOCATED events in
    the summaries): fewer files than LOCATED events = NLLoc wrote two events of one job to one file name (same
    origin second)."""
    h = it_dir / "hyp"
    h.mkdir(exist_ok=True)
    n = 0
    for f in sorted(it_dir.glob("job*/loc.*.grid0.loc.hyp")):
        if ".sum." in f.name:
            continue
        ln = h / f"{f.parent.name}_{f.name}"
        if not ln.is_symlink():
            ln.symlink_to(f)
        n += 1
    located = (it_dir / "all.sum.grid0.loc.hyp").read_text().count('"LOCATED"')
    return h, n, located


def loc2ssst(it_dir, d_in, d_out, char, box, sp, nproc=4):
    """Loc2ssst on the LOCATED events of it_dir (all.sum.grid0.loc.hyp): corrections with CharDist char, corrected time
    grids into d_out (links to d_in first). Returns d_out."""
    done = d_out / "loc2ssst.json"
    if done.exists():
        return d_out
    d_out.mkdir(parents=True, exist_ok=True)
    for f in sorted(d_in.glob("cascadia3d.*.time.*")):
        ln = d_out / f.name
        if not ln.exists() and not ln.is_symlink():
            ln.symlink_to(os.path.realpath(f))
    hyp, n_files, n_located = link_hyps(it_dir)
    x0, y0, nx, ny = box
    s = lsgrid_km(char)
    gx, gy, gz = (int(np.ceil((n - 1) / s)) + 1 for n in (nx, ny, NZ))
    sets = station_sets(sp, nproc)
    base = (f"CONTROL 1 54321\nTRANS LAMBERT WGS-84 45.0 -125.8 41.0 49.0 0.0\n"
            f"LSGRID {gx} {gy} {gz} {x0:.1f} {y0:.1f} {Z0:.1f} {s:.1f} {s:.1f} {s:.1f} SSST_TIMECORR FLOAT\n"
            f"LSOUTGRID {nx} {ny} {NZ} {x0:.1f} {y0:.1f} {Z0:.1f} 1.0 1.0 1.0 TIME FLOAT\n"
            f"LSPHSTAT {LSPHSTAT}\nLOCPHASEID P P p Pg Pn\nLOCPHASEID S S s Sg Sn\n"
            f"LSPARAMS {char:g} {WEIGHT_FLOOR:g} 0\nLSMODE ANGLES_NO\nLSOUT {d_out}/cascadia3d\n"
            f"LSLOCFILES {hyp}/job*.grid0.loc.hyp\n"
            f"LOCFILES {it_dir}/job00/events.obs NLLOC_OBS {d_in}/cascadia3d {d_out}/unused 0\n"
            f"LOCMETH EDT_OT_WT 9999.0 5 -1 -1 -1 20 -1.0 1\n")
    wd = d_out / "work"
    wd.mkdir(exist_ok=True)
    procs = []
    t0 = time.time()
    for i, st in enumerate(sets):
        line = "LSSTATIONS =" + "=".join(st) + "="
        assert len(line) < 1000, "LSSTATIONS line too long for NonLinLoc's line buffer"
        (wd / f"ssst_{i}.in").write_text(base + line + "\n")
        procs.append(subprocess.Popen([str(BIN / "Loc2ssst"), f"ssst_{i}.in"], cwd=wd,
                                      stdout=open(wd / f"ssst_{i}.log", "w"), stderr=subprocess.STDOUT))
    rcs = [p.wait() for p in procs]
    logs = "\n".join((wd / f"ssst_{i}.log").read_text(errors="replace") for i in range(len(sets)))
    errs = [ln for ln in logs.splitlines() if "ERROR" in ln and "makecpt" not in ln]
    if any(rcs) or errs:
        raise SystemExit(f"Loc2ssst {d_out}: return codes {rcs}, {len(errs)} error lines, e.g. {errs[:3]}")
    # what was written: corrected grids are regular files now, the rest stay links
    new = sorted(f for f in d_out.glob("cascadia3d.*.time.buf") if not f.is_symlink())
    stats, n_neg = {}, 0
    for f in new:
        key = f.name[len("cascadia3d."):-len(".time.buf")]
        t = np.fromfile(f, "<f4")
        c = np.fromfile(f.with_name(f.name.replace(".time.buf", ".ssst.buf")), "<f4")
        neg = int((t < 0).sum())
        n_neg += neg
        stats[key] = dict(t_min=round(float(t.min()), 4), neg_nodes=neg, corr_mean=round(float(c.mean()), 4),
                          corr_min=round(float(c.min()), 4), corr_max=round(float(c.max()), 4))
        if neg:                                    # where: distance of the negative nodes from the station (km)
            hd = f.with_name(f.name[:-3] + "hdr").read_text().splitlines()
            gx_, gy_, gz_ = (int(v) for v in hd[0].split()[:3])
            ox, oy, oz = (float(v) for v in hd[0].split()[3:6])
            sx, sy, sz = (float(v) for v in hd[1].split()[1:4])
            i, j, k = np.unravel_index(np.flatnonzero(t < 0), (gx_, gy_, gz_))
            dist = np.sqrt((ox + i - sx) ** 2 + (oy + j - sy) ** 2 + (oz + k - sz) ** 2)
            stats[key]["neg_max_dist_km"] = round(float(dist.max()), 2)
    m = re.findall(r"(\d+) location files read, (\d+) accepted", logs)
    acc = int(m[0][1]) if m else None
    if not new or not acc:
        raise SystemExit(f"Loc2ssst {d_out}: {acc} events accepted, {len(new)} corrected grids: see {wd}")
    done.write_text(json.dumps({"char_dist_km": char, "lsgrid_km": s, "lsgrid": [gx, gy, gz], "box": box,
                                "weight_floor": WEIGHT_FLOOR, "lsphstat": LSPHSTAT, "station_sets": sets,
                                "events_accepted": acc, "event_files": n_files, "located": n_located,
                                "corrected_grids": len(new), "negative_nodes": n_neg,
                                "wall_s": round(time.time() - t0, 1), "grids": stats,
                                "loc2ssst_md5": md5(BIN / "Loc2ssst"), "input": str(d_in)}, indent=1))
    log(f"Loc2ssst {d_out}: CharDist {char:g} km, LSGRID {s:g} km, {acc} events accepted, {len(new)} corrected grids,"
        f" {n_neg} negative nodes, {time.time() - t0:.0f} s")
    return d_out


# --------------------------------------------------------------------------------------------------------- coherence
def hyp_roots(it_dir):
    """{ID: per-event file root (without .hyp)} of an NLLoc run with SAVE_NLLOC_ALL (LOCATED events)."""
    out = {}
    for f in it_dir.glob("job*/loc.*.grid0.loc.hyp"):
        if ".sum." in f.name:
            continue
        txt = f.read_text(errors="replace")
        m = re.search(r"^PUBLIC_ID v4_(\d+)", txt, re.M)
        st = re.search(r'^NLLOC "[^"]*" "([A-Z]+)"', txt, re.M)
        if m and st and st.group(1) == "LOCATED":
            out.setdefault(int(m.group(1)), []).append(str(f)[:-4])
    dup = {k: v for k, v in out.items() if len(v) > 1}
    return {k: v[0] for k, v in out.items() if len(v) == 1}, dup


def coherences(s):
    """event-pair coherence = median cc_max over the pair's measured station-phases (status ok, not at the lag edge)."""
    c = pd.read_parquet(CC_DIR / f"cc_{s}.parquet")
    id1, id2 = [k for k in c.columns if k.lower() in ("id1", "ev1", "event1")], \
        [k for k in c.columns if k.lower() in ("id2", "ev2", "event2")]
    assert id1 and id2, f"cc_{s}.parquet: event id columns not found in {list(c.columns)[:20]}"
    c = c[(c.status == "ok") & (~c.edge.astype(bool))]
    g = c.groupby([id1[0], id2[0]]).cc_max.agg(["median", "size"]).reset_index()
    g.columns = ["ID1", "ID2", "coh", "n"]
    return g[g.n >= COH_NMIN]


def coherence_run(s, g, it_dir, roots_t, box, stat=None, name="coherence", nproc=4):
    """NLL-coherence on the final SSST iteration. Only events whose own pdf is compact (ellipsoid len3 <= MaxSE3, the
    filter NLLoc applies to the other events of a stack) are relocated: in the JdF test (tier C, len3 ~25 km) a
    target whose stack held only itself moved from 3.1 to 33.8 km (the oct-tree posterior search of a broad pdf does
    not return its maximum-likelihood point); the other events keep their final SSST location (metrics)."""
    rd = OUT / g / s / name
    if (rd / "run.json").exists():
        return rd
    import a2_parse_hyp as PH
    roots, dup = hyp_roots(it_dir)
    len3 = {eid: PH.parse(f"{r}.hyp")[0].get("ell_len3", np.inf) for eid, r in roots.items()}
    compact = {eid for eid, v in len3.items() if v <= COH_MAX_SE3}
    coh = coherences(s)
    coh = coh[(coh.coh >= COH_MIN) & coh.ID1.isin(roots) & coh.ID2.isin(roots)]
    nb = {}
    for r in coh.itertuples():
        nb.setdefault(r.ID1, []).append((r.coh, r.ID2))
        nb.setdefault(r.ID2, []).append((r.coh, r.ID1))
    rd.mkdir(parents=True, exist_ok=True)
    tmpl = (TA / "nlloc_template.in").read_text()
    opt = argparse.Namespace(time=str(roots_t[0]), roots=tuple(str(r) for r in roots_t), template_text=tmpl,
                             locgau=LOCGAU, locgau2=LOCGAU2, topo=FLOOR, max_dist=RN.MAX_DIST)
    jobs = []
    for eid, root in sorted(roots.items()):
        if eid not in compact:
            continue
        wd = rd / f"e{eid}"
        wd.mkdir(exist_ok=True)
        lines = ["STREAM_COHERENCES", f"  1.000000 {root}"]
        lines += [f"  {c:.6f} {roots[o]}" for c, o in sorted(nb.get(eid, []), reverse=True)]
        sc = wd / "event.stream_coherences"
        sc.write_text("\n".join(lines) + "\n")
        ctl = RN.control(box, f"{root}.hyp", wd / "loc", str(stat) if stat else None, opt)
        ctl = RN.sub1(r"^LOCFILES .*$", f"LOCFILES {root}.hyp NLLOC_OBS {','.join(opt.roots)} {wd}/loc", ctl)
        ctl += f"LOCPOSTERIOR OCT_TREE {sc} 1e-30 {COH_MIN} -1.0 {COH_MAX_SE3} -1 -1.0 -999\n"
        jobs.append((wd, ctl))
    t0 = time.time()
    with ProcessPoolExecutor(nproc) as ex:
        bad = [name for name, rc in ex.map(RN.run_one, jobs, chunksize=8) if rc != 0]
    with open(rd / "all.sum.grid0.loc.hyp", "w") as f:
        for wd, _ in jobs:
            p = wd / "loc.sum.grid0.loc.hyp"
            if p.exists():
                f.write(p.read_text())
    part = {e for e in compact if any(o in compact for _, o in nb.get(e, []))}
    (rd / "run.json").write_text(json.dumps({"events": len(jobs), "failed": bad, "duplicate_roots": len(dup),
                                             "events_final_ssst": len(roots), "events_compact": len(compact),
                                             "compact_with_compact_partner": len(part),
                                             "pairs_used": len(coh), "coh_min": COH_MIN, "coh_nmin": COH_NMIN,
                                             "max_se3": COH_MAX_SE3, "events_with_partners": len(nb),
                                             "grids": list(opt.roots), "stat": str(stat) if stat else None,
                                             "located_from": str(it_dir), "wall_s": round(time.time() - t0, 1)},
                                            indent=1))
    log(f"coherence {rd}: {len(jobs)} events, {len(nb)} with partners, {len(bad)} failed, {time.time() - t0:.0f} s")
    return rd


# ---------------------------------------------------------------------------------------------------------------- run
def cmd_run(a):
    s, g = a.set, a.grids
    blocks, box = prep(s, g)
    sp = station_phases(blocks)
    base = base_grids(s, g, blocks, box)
    d = OUT / g / s
    log(f"{s} ({g}): {len(blocks)} events, {len(sp)} station-phases, box x0 {box[0]} y0 {box[1]} {box[2]} x {box[3]} km")
    arms = a.arms.split(",")
    fw = (f"{GRIDS[g]['fine']}/cascadia3d", f"{GRIDS[g]['wide']}/cascadia3d")
    if "static" in arms and GRIDS[g]["static"]:
        nll_run(d / "static", blocks, box, fw, stat=GRIDS[g]["static"], nproc=a.nproc)
    if "static_coherence" in arms and GRIDS[g]["static"]:      # coherence on the static arm (octree files kept)
        so = nll_run(d / "static_oct", blocks, box, fw, stat=GRIDS[g]["static"], octree=True, nproc=a.nproc)
        coherence_run(s, g, so, fw, box, stat=GRIDS[g]["static"], name="static_coherence", nproc=a.nproc)
    if "ssst" in arms or "coherence" in arms:
        d_in = base
        nfin = len(SCHEDULE)
        for k, char in enumerate(SCHEDULE):
            it = nll_run(d / "ssst" / f"it{k}", blocks, box, (f"{d_in}/cascadia3d",), nproc=a.nproc)
            d_in = loc2ssst(it, d_in, d / "ssst" / f"g{k}", char, box, sp, nproc=a.nproc)
        last = nll_run(d / "ssst" / f"it{nfin}", blocks, box, (f"{d_in}/cascadia3d",), octree=True, nproc=a.nproc)
        if "coherence" in arms:
            coherence_run(s, g, last, (f"{d_in}/cascadia3d",), box, nproc=a.nproc)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("picks")
    r = sub.add_parser("run")
    r.add_argument("set", choices=SETS)
    r.add_argument("--grids", default="g51", choices=list(GRIDS))
    r.add_argument("--arms", default="static,ssst,coherence,static_coherence")
    r.add_argument("--nproc", type=int, default=4)
    a = ap.parse_args()
    {"picks": cmd_picks, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()

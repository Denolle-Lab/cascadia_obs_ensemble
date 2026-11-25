"""D-B078 comparison, NLL-SSST-coherence leg (Lomax & Savvaidis 2022, JGR 127, e2021JB023190): NonLinLoc with
source-specific station terms (Loc2ssst, shrinking smoothing distance) and then waveform-coherence relocation (each
event's location PDF stacked with the PDFs of its coherent neighbours, NLLoc LOCPOSTERIOR + STREAM_COHERENCES).

Common frame: b5_cmp_common (events, picks, CC rows, folds, held-out stations, output layout, scorer). For each set s
and run in runs() (full, fold0..fold2; a fold run drops the picks and CC rows at its held-out stations):

  it0   baseline: NLLoc of the set's events with picks(s), the v5.1 control settings (b4_v51_iter1/<tile>/nlloc.in:
        LOCMETH EDT_OT_WT 9999 5 -1 -1 -1 20 -1 1 but with every grid in memory (-1 for 20: no effect on the result,
        the cropped / SSST grids are small), LOCGAU 0.2 0.0, no LOCGAU2, LOCPHASEID, LOCQUAL2ERR, LOCPHSTAT,
        LOCANGLES, LOCTOPO_SURFACE merged floor, oct-tree 0.02 km / 40000 nodes / 2000 scatter / stop at min node)
        and the static terms of b4_v51_iter0 (INCLUDE station_terms.in); pick error = picks(s).sigma (D-B008 classes,
        the same classes as a2_make_obs.py). Search box: the set's epicentres +- MARGIN km, z' ZTOP..ZBOT (not the
        production tile +- 75 km box); oct-tree start cells ~2 km (about the production count of start cells).
  SSST  for k = 0..len(SCHEDULE)-1: Loc2ssst on the hyps of pass it<k> with LSPARAMS CharDist = SCHEDULE[k]
        (WeightFloor 1e-7, Lomax's run_ssst_relocations.bash), input time root T<k> (T0 = the v5.1 grids), output
        T<k+1> = T<k> + smoothed (residual + delay) on LSOUTGRID; then NLLoc pass it<k+1> with T<k+1> and no static
        terms (the delay of it0 enters the first correction, Lomax's cumulative scheme). The last pass saves the
        oct-tree PDFs (SAVE_NLLOC_OCTREE). Only training stations have picks in a fold run, so only they get terms.
  coh   one NLLoc run per event with LOCPOSTERIOR OCT_TREE <ev>.stream_coherences 1e-30 0.5 -1 -1 -1 -1 -999
        (CoherenceMin of run_coherence.bash; MaxSE3 off, see COH_MIN below), the time root of the last SSST pass, the
        event's own picks (they set the origin time and residuals; the PDF is the coherence stack only), finer
        oct-tree start cells (>= ~10^4, as run_coherence.bash advises) and no stop at the minimum node. Event-pair
        coherence = mean cc_max over the pair's accepted CC rows (cc_rows(s)) at the run's training stations; pairs
        without rows get no coherence. Self first with coherence 1; neighbours that were not LOCATED in the last
        SSST pass are not listed (no oct-tree).

Time grids: per set, the v5.1 grid of each station-phase (fine time_b4_v0.3.1m_sc21 if its extent contains the box,
else the wide 2 km grid: NLLoc's rule for the two LOCFILES roots of production) cropped to the box + 1 node (values
at the nodes unchanged, so NLLoc's trilinear times inside the box are those of the v5.1 grids); SSST grids on
LSGRID = LSOUTGRID = the box at DXS km (nodes aligned with the 1 and 2 km nodes).

Outputs: out_dir(leg, s, run)/reloc.csv (ID, lat, lon, depth: true km bsl = Re (1 - exp(-z'/Re))) and run.json, for
legs nllbase (it0), nllssst (last SSST pass) and nllcoh; LOCATED events only (others listed in run.json).
Scratch: WORK/<set>/<run>/. CPU: at most NPROC processes.

Usage (from workflow/10_tomography, pixi python; steps whose outputs exist are skipped, so a rerun resumes):
  python b5_cmp_nll.py SET [SET ...] [--runs full,fold0,fold1,fold2] [--nproc 16]
  python b5_cmp_nll.py --score                  # score_set of nllbase / nllssst / nllcoh -> CMP/nllcoh/scores.json
  python b5_cmp_nll.py SET --pass-scores        # diagnostic: held-out scores of every pass -> WORK/<set>/
  python b5_cmp_nll.py SET --variant K          # diagnostic: coherence on top of SSST pass it<K> (var_coh<K>)
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b5_cmp_common as C  # noqa: E402
from b2_fwd_common import eft_depth, lambert_km  # noqa: E402

BIN = Path("/wd1/mdenolle_data/reloc_v5/src/NonLinLoc/src/bin")
FINE = Path(C.GRIDS)
WIDE = Path("/wd1/mdenolle_data/reloc_v5/wide_time_b4_v0.3.1m_sc21")
STAT = Path("/wd1/mdenolle_data/reloc_v5/runs/b4_v51_iter0/station_terms.in")
TOPO = "/wd1/mdenolle_data/reloc_v5/topo/topo_15s_cons_merged_eft.bin"
WORK = Path("/wd1/mdenolle_data/tomo_v1/b5/cmp/nll_work")
PRODUCTION = (Path("/wd1/mdenolle_data/reloc_v5/runs"), Path("/wd1/mdenolle_data/reloc_v5/catalog"))
RE = 6371.0

MARGIN = 10.0            # km around the set's epicentres (search box, SSST grids)
ZTOP, ZBOT = -5.0, 25.0  # z' (km) of the box; -5 = top of the v5.1 grids
DXS = 0.25               # km, LSGRID / LSOUTGRID spacing (<= D_min / 2)
SCHEDULE = [9999.0, 2.0, 1.0, 0.5]   # Loc2ssst CharDist (km) per SSST pass: static, then 2 km halving to 0.5 km
WEIGHT_FLOOR = 1e-7
# Loc2ssst event / arrival filter: RMS max 1 s, no Nrdgs or gap cut, |residual| max 1 s (P) / 2 s (S) (Lomax's
# Parkfield example), no Len3 cut (Loc2ssst default; his 5 km cut is tuned to LOCGAU 0.05 s, with LOCGAU 0.2 s the
# 68% ellipsoids of the set events are 2-15 km, so it would drop 70% of jdf and blanco)
LSPHSTAT = "1.0 0 360.0 1.0 2.0 9999.0"
# LOCPOSTERIOR: CoherenceMin 0.5 (run_coherence.bash); MaxSE3 off (NLLoc default; same reason as the Len3 cut above),
# MaxOtherWeight, MaxNother, MaxMagDiff, MinMag off (run_coherence.bash and NLLoc defaults)
COH_MIN, MAX_SE3, COH_TOTAL, COH_NMAX, COH_DMAG, COH_MINMAG = 0.5, -1.0, -1.0, -1, -1.0, -999
OCT_CELL = 2.0           # km, start cells of the baseline / SSST oct-tree
COH_CELLS = 12000        # about this many start cells for the coherence oct-tree
NPROC = 16
QS = [0, 10, 50, 90, 100]
LOG = None

CONTROL = """CONTROL 1 54321
TRANS LAMBERT WGS-84 45.0 -125.8 41.0 49.0 0.0
LOCSIG Cascadia OBS ensemble, D-B078 NLL-SSST-coherence leg (b5_cmp_nll.py)
LOCCOM {com}
LOCFILES {obs} NLLOC_OBS {time} {out}
LOCHYPOUT SAVE_NLLOC_ALL NLL_FORMAT_VER_2 FILENAME_PUBLIC_ID{octree}
LOCMETH EDT_OT_WT 9999.0 5 -1 -1 -1 -1 -1.0 1
LOCGAU 0.2 0.0
LOCPHASEID P P p Pg Pn
LOCPHASEID S S s Sg Sn
LOCQUAL2ERR 0.05 0.1 0.15 0.3 99999.9
LOCPHSTAT 9999.0 -1 9999.0 1.0 1.0 9999.9 -9999.9 9999.9
LOCANGLES ANGLES_YES 5
LOCTOPO_SURFACE {topo} 0
{grid}
{search}
{extra}
"""


def log(msg):
    line = f"{datetime.now():%H:%M:%S} {msg}"
    print(line, flush=True)
    if LOG is not None:
        with open(LOG, "a") as f:
            f.write(line + "\n")


def safe(path):
    p = Path(path).resolve()
    assert str(p).startswith(str(WORK)) or str(p).startswith(str(C.CMP)), f"refusing to write outside scratch: {p}"
    for q in PRODUCTION:
        assert not str(p).startswith(str(q)), f"refusing to write into production: {p}"
    return p


def true_depth(zf):
    return RE * (1.0 - np.exp(-np.asarray(zf, float) / RE))


# ------------------------------------------------------------------------------------------------------ geometry
def box(s):
    """search box of a set: dict x0, y0, z0, nx, ny, nz (1 km nodes, integer km origin)."""
    e = C.events(s)
    x, y = lambert_km(e.lon.values, e.lat.values)
    x0, x1 = np.floor(x.min() - MARGIN), np.ceil(x.max() + MARGIN)
    y0, y1 = np.floor(y.min() - MARGIN), np.ceil(y.max() + MARGIN)
    assert eft_depth(e.depth.values).max() < ZBOT - 5, f"{s}: events deeper than the box allows"
    return dict(x0=float(x0), y0=float(y0), z0=ZTOP, nx=int(x1 - x0) + 1, ny=int(y1 - y0) + 1,
                nz=int(ZBOT - ZTOP) + 1)


def extent(b, d=1.0):
    return (b["x0"], b["x0"] + (b["nx"] - 1) * d, b["y0"], b["y0"] + (b["ny"] - 1) * d, b["z0"],
            b["z0"] + (b["nz"] - 1) * d)


def read_hdr(path):
    lines = Path(path).read_text().splitlines()
    t = lines[0].split()
    return dict(n=[int(v) for v in t[:3]], o=[float(v) for v in t[3:6]], d=[float(v) for v in t[6:9]],
                rest=t[9:], lines=lines)


def crop_grid(src_root, dst_root, ext):
    """crop a NLL time grid (hdr + buf) to the nodes covering ext (+1 node where available); values unchanged."""
    h = read_hdr(f"{src_root}.time.hdr")
    n, o, d = h["n"], h["o"], h["d"]
    lo, hi = [], []
    for k in range(3):
        a = int(np.floor((ext[2 * k] - o[k]) / d[k] + 1e-9)) - 1
        b = int(np.ceil((ext[2 * k + 1] - o[k]) / d[k] - 1e-9)) + 1
        a, b = max(a, 0), min(b, n[k] - 1)
        assert o[k] + a * d[k] <= ext[2 * k] + 1e-9 and o[k] + b * d[k] >= ext[2 * k + 1] - 1e-9, \
            f"{src_root} does not contain the box"
        lo.append(a)
        hi.append(b)
    buf = np.fromfile(f"{src_root}.time.buf", dtype="<f4")
    assert buf.size == n[0] * n[1] * n[2], f"{src_root}: incomplete buffer"
    g = buf.reshape(n)[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]
    nn = g.shape
    oo = [o[k] + lo[k] * d[k] for k in range(3)]
    first = (f"{nn[0]} {nn[1]} {nn[2]}  {oo[0]:.6f} {oo[1]:.6f} {oo[2]:.6f}  {d[0]:.6f} {d[1]:.6f} {d[2]:.6f} "
             + " ".join(h["rest"]))
    Path(f"{dst_root}.time.hdr").write_text("\n".join([first] + h["lines"][1:]) + "\n")
    np.ascontiguousarray(g, dtype="<f4").tofile(f"{dst_root}.time.buf")


def inside(ext, root):
    h = read_hdr(f"{root}.time.hdr")
    for k in range(3):
        lo, hi = h["o"][k], h["o"][k] + (h["n"][k] - 1) * h["d"][k]
        if ext[2 * k] < lo - 1e-9 or ext[2 * k + 1] > hi + 1e-9:
            return False
    return True


def time0(s):
    """per-set folder of the v5.1 grids cropped to the box; returns (root, {(phase, station): 'fine'|'wide'})."""
    d = safe(WORK / s / "time0")
    b = box(s)
    ext = extent(b)
    keys = sorted(set(zip(C.picks(s).phase, C.picks(s).station)))
    done = d / "source.json"
    if done.exists():
        src = {tuple(k.split("|")): v for k, v in json.loads(done.read_text()).items()}
        if set(src) == set(keys):
            return d / "cascadia3d", src
    d.mkdir(parents=True, exist_ok=True)
    src = {}
    for ph, st in keys:
        f, w = FINE / f"cascadia3d.{ph}.{st}", WIDE / f"cascadia3d.{ph}.{st}"
        if Path(f"{f}.time.hdr").exists() and inside(ext, f):
            root, src[(ph, st)] = f, "fine"
        elif Path(f"{w}.time.hdr").exists() and inside(ext, w):
            root, src[(ph, st)] = w, "wide"
        else:
            raise SystemExit(f"{s}: no v5.1 grid of {ph} {st} contains the box {ext}")
        crop_grid(root, d / f"cascadia3d.{ph}.{st}", ext)
    done.write_text(json.dumps({f"{k[0]}|{k[1]}": v for k, v in src.items()}, indent=0))
    log(f"{s}: time0 {len(src)} station-phases ({sum(v == 'wide' for v in src.values())} from the wide grids), "
        f"box x {ext[0]:g}..{ext[1]:g} y {ext[2]:g}..{ext[3]:g} z' {ext[4]:g}..{ext[5]:g}")
    return d / "cascadia3d", src


# ------------------------------------------------------------------------------------------------------ obs
def run_picks(s, run):
    p = C.picks(s)
    ho = C.held_out(s, run)
    return p[~p.station.isin(ho)].reset_index(drop=True)


def obs_blocks(s, run):
    """{ID: NLLOC_OBS block} of the run's picks; absolute times = v5.1 origin + tt; GAU error = sigma."""
    p = run_picks(s, run)
    t0 = pd.to_datetime(C.events(s).set_index("ID").time)
    p["t"] = t0.reindex(p.ID).values + pd.to_timedelta(p.tt.values, unit="s")
    p = p.sort_values(["ID", "t"])
    out = {}
    for eid, g in p.groupby("ID", sort=True):
        lines = [f"PUBLIC_ID ev{eid}"]
        for r in g.itertuples():
            t = pd.Timestamp(r.t)
            sec = t.second + t.microsecond / 1e6 + t.nanosecond / 1e9
            lines.append(f"{r.station:<10s} ?    ?    ? {r.phase:<6s} ? {t:%Y%m%d} {t:%H%M} {sec:9.6f} GAU "
                         f"{r.sigma:9.2e} -1.00e+00 -1.00e+00 -1.00e+00 1.0")
        out[int(eid)] = "\n".join(lines) + "\n"
    return out


def chunks(ids, n):
    ids = sorted(ids)
    return [ids[i::n] for i in range(n) if ids[i::n]]


# ------------------------------------------------------------------------------------------------------ NLLoc
def search_line(b, cell=OCT_CELL, coh=False):
    ex = [(b["nx"] - 1), (b["ny"] - 1), (b["nz"] - 1)]
    if coh:
        cell = (ex[0] * ex[1] * ex[2] / COH_CELLS) ** (1 / 3)
        return (f"LOCSEARCH OCT {max(2, round(ex[0] / cell))} {max(2, round(ex[1] / cell))} "
                f"{max(2, round(ex[2] / cell))} 0.02 50000 2000 0 0")
    return (f"LOCSEARCH OCT {max(2, round(ex[0] / cell))} {max(2, round(ex[1] / cell))} {max(2, round(ex[2] / cell))} "
            "0.02 40000 2000 0 1")


def grid_line(b):
    return (f"LOCGRID {b['nx']} {b['ny']} {b['nz']} {b['x0']:.1f} {b['y0']:.1f} {b['z0']:.1f} 1.0 1.0 1.0 "
            "PROB_DENSITY SAVE")


def control(com, obs, troot, out, b, stat=False, octree=False, posterior=None):
    extra = []
    if stat:
        extra.append(f"INCLUDE {STAT}")
    if posterior:
        extra.append(f"LOCPOSTERIOR OCT_TREE {posterior} 1e-30 {COH_MIN} {COH_TOTAL} {MAX_SE3} {COH_NMAX} {COH_DMAG} "
                     f"{COH_MINMAG}")
    return CONTROL.format(com=com, obs=obs, time=troot, out=out, octree=" SAVE_NLLOC_OCTREE" if octree else "",
                          topo=TOPO, grid=grid_line(b), search=search_line(b, coh=bool(posterior)),
                          extra="\n".join(extra))


def run_bin(prog, ctl, cwd):
    with open(cwd / f"{prog.lower()}.log", "w") as lf:
        r = subprocess.run([str(BIN / prog), str(ctl)], cwd=cwd, stdout=lf, stderr=subprocess.STDOUT)
    return r.returncode


def parse_sum(path):
    """rows of a NLLoc .hyp file (sum or event): ID, status, comment, x, y, z' (ML), lat, lon, rms, nphs, gap,
    len3, root."""
    rows, cur = [], None
    for line in open(path):
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "NLLOC":
            q = re.findall(r'"([^"]*)"', line)
            cur = dict(root=q[0], status=q[1], comment=q[2] if len(q) > 2 else "")
        elif cur is None:
            continue
        elif tok[0] == "PUBLIC_ID":
            cur["ID"] = int(tok[1].replace("ev", ""))
        elif tok[0] == "HYPOCENTER":
            cur.update(x=float(tok[2]), y=float(tok[4]), zf=float(tok[6]))
        elif tok[0] == "GEOGRAPHIC":
            d = dict(zip(tok[8::2], tok[9::2]))
            cur.update(lat=float(d["Lat"]), lon=float(d["Long"]), depth_f=float(d["Depth"]))
        elif tok[0] == "QUALITY":
            d = dict(zip(tok[1::2], tok[2::2]))
            cur.update(rms=float(d["RMS"]), nphs=int(d["Nphs"]), gap=float(d["Gap"]))
        elif tok[0] == "STATISTICS":
            d = dict(zip(tok[1::2], tok[2::2]))
            cur.update(len3=float(d["Len3"]))
        elif tok[0] == "END_NLLOC":
            rows.append(cur)
            cur = None
    return rows


def nlloc_pass(s, run, k, troot, blocks, b, stat, octree, nproc, name=None):
    """NLLoc pass it<k>: chunks in parallel; returns DataFrame of the sum rows (one per event) and writes all.hyp
    (per-event hyps concatenated, the Loc2ssst input)."""
    name = name or f"it{k}"
    d = safe(WORK / s / run / name)
    if (d / "locs.csv").exists():
        return pd.read_csv(d / "locs.csv")
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    jobs = []
    for i, ids in enumerate(chunks(blocks, nproc)):
        cd = d / f"c{i:02d}"
        cd.mkdir()
        (cd / "events.obs").write_text("\n".join(blocks[e] for e in ids) + "\n")
        (cd / "nlloc.in").write_text(control(f"{s} {run} it{k}", cd / "events.obs", troot, cd / f"c{i:02d}", b,
                                             stat=stat, octree=octree))
        jobs.append(cd)
    t = time.time()
    with ThreadPoolExecutor(nproc) as ex:
        rc = list(ex.map(lambda cd: run_bin("NLLoc", cd / "nlloc.in", cd), jobs))
    if any(rc):
        raise SystemExit(f"{s} {run} it{k}: NLLoc failed in {[j.name for j, r in zip(jobs, rc) if r]}")
    rows = []
    with open(d / "all.hyp", "w") as fa:
        for cd in jobs:
            rows += parse_sum(cd / f"{cd.name}.sum.grid0.loc.hyp")
            for f in sorted(cd.glob(f"{cd.name}.*_ev*.grid0.loc.hyp")):
                fa.write(f.read_text())
    loc = pd.DataFrame(rows)
    assert loc.ID.is_unique and set(loc.ID) <= set(blocks), f"{s} {run} it{k}: IDs differ from the obs"
    loc["depth"] = true_depth(loc.zf)
    loc.to_csv(d / "locs.csv", index=False)
    log(f"{s} {run} {name}: {int((loc.status == 'LOCATED').sum())}/{len(blocks)} LOCATED "
        f"({time.time() - t:.0f} s){' + oct-trees' if octree else ''}")
    for cd in jobs:            # scatter files are not used here
        for f in cd.glob("*.scat"):
            f.unlink()
    return loc


# ------------------------------------------------------------------------------------------------------ Loc2ssst
def station_sets(stations, n):
    """n groups of stations for parallel Loc2ssst instances; LSSTATIONS matches by substring (strstr), so a station
    whose code is contained in another's goes with it."""
    stations = sorted(stations)
    parent = {s: s for s in stations}

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a
    for a in stations:
        for c in stations:
            if a != c and a in c:
                parent[find(a)] = find(c)
    groups = {}
    for a in stations:
        groups.setdefault(find(a), []).append(a)
    groups = sorted(groups.values(), key=len, reverse=True)
    rng = np.random.default_rng(78)
    order = rng.permutation(len(groups))
    sets = [[] for _ in range(min(n, len(groups)))]
    for j, i in enumerate(order):
        sets[j % len(sets)] += groups[i]
    return sets


def loc2ssst(s, run, k, D, tin, tout_dir, hyp, b, nproc):
    """Loc2ssst pass k: T<k+1> = T<k> + SSST(CharDist D); tout_dir gets links to every grid of T<k> first."""
    tout_dir = safe(tout_dir)
    if (tout_dir / "done.json").exists():
        return json.loads((tout_dir / "done.json").read_text())
    if tout_dir.exists():
        shutil.rmtree(tout_dir)
    tout_dir.mkdir(parents=True)
    for f in Path(tin).parent.glob(f"{Path(tin).name}.*.time.*"):
        os.symlink(os.path.realpath(f), tout_dir / f.name)
    stations = sorted({l.split()[0] for l in open(hyp) if re.match(r"^\S+\s+\?\s+\?\s+\?\s+[PS]\s", l)})
    ext = extent(b, 1.0)
    n = [int(round((ext[1] - ext[0]) / DXS)) + 1, int(round((ext[3] - ext[2]) / DXS)) + 1,
         int(round((ext[5] - ext[4]) / DXS)) + 1]
    g = f"{n[0]} {n[1]} {n[2]}  {ext[0]:.2f} {ext[2]:.2f} {ext[4]:.2f}  {DXS} {DXS} {DXS}"
    base = (f"CONTROL 1 54321\nTRANS LAMBERT WGS-84 45.0 -125.8 41.0 49.0 0.0\n"
            f"LSGRID {g} SSST_TIMECORR FLOAT\nLSOUTGRID {g} TIME FLOAT\nLSPHSTAT {LSPHSTAT}\n"
            "LOCPHASEID P P p Pg Pn\nLOCPHASEID S S s Sg Sn\n"
            f"LSPARAMS {D} {WEIGHT_FLOOR} 0\nLSMODE ANGLES_NO\nLSOUT {tout_dir}/cascadia3d\nLSLOCFILES {hyp}\n"
            f"LOCFILES {Path(hyp).parent}/none.obs NLLOC_OBS {tin} {Path(hyp).parent}/none 0\n"
            "LOCMETH EDT_OT_WT 9999.0 5 -1 -1 -1 20 -1.0 1\n")
    work = tout_dir / "l2s"
    work.mkdir()
    jobs = []
    for i, st in enumerate(station_sets(stations, nproc)):
        cd = work / f"s{i:02d}"
        cd.mkdir()
        (cd / "loc2ssst.in").write_text(base + f"LSSTATIONS {','.join(st)}\n")
        jobs.append(cd)
    t = time.time()
    with ThreadPoolExecutor(nproc) as ex:
        rc = list(ex.map(lambda cd: run_bin("Loc2ssst", cd / "loc2ssst.in", cd), jobs))
    if any(rc):
        raise SystemExit(f"{s} {run} Loc2ssst pass {k}: failed in {[j.name for j, r in zip(jobs, rc) if r]}")
    # every station-phase of the hyps must now have a new (regular-file) grid
    new = sorted(f.name for f in tout_dir.glob("cascadia3d.*.time.buf") if not f.is_symlink())
    stat = {}
    for f in tout_dir.glob("cascadia3d.*.ssst.buf"):
        h = read_hdr(str(f)[:-4] + ".hdr")
        v = np.fromfile(f, dtype="<f4")
        stat[f.name[len("cascadia3d."):-len(".ssst.buf")]] = [round(float(np.median(v)), 4),
                                                              round(float(v.min()), 4), round(float(v.max()), 4)]
        assert v.size == np.prod(h["n"])
        f.unlink()
        Path(str(f)[:-4] + ".hdr").unlink()
    lines = sum(1 for _ in open(jobs[0] / "loc2ssst.log"))
    accepted = [re.search(r"(\d+) location files read, (\d+) accepted", (cd / "loc2ssst.log").read_text())
                for cd in jobs]
    info = dict(k=k, char_dist=D, station_phases=len(new), stations=len(stations), wall_s=round(time.time() - t, 1),
                events_accepted=int(accepted[0].group(2)) if accepted[0] else None, ssst_median_min_max=stat)
    (tout_dir / "done.json").write_text(json.dumps(info, indent=1))
    log(f"{s} {run} Loc2ssst pass {k} D={D:g} km: {len(new)} station-phases of {len(stations)} stations, "
        f"{info['events_accepted']} events accepted ({info['wall_s']:.0f} s; log {lines} lines)")
    return info


# ------------------------------------------------------------------------------------------------------ coherence
def coherences(s, run):
    """event-pair coherence: mean cc_max over the pair's accepted CC rows at the run's training stations."""
    c = C.cc_rows(s)
    c = c[~c.station.isin(C.held_out(s, run))]
    a = np.minimum(c.ID1.values, c.ID2.values)
    b2 = np.maximum(c.ID1.values, c.ID2.values)
    g = pd.DataFrame(dict(a=a, b=b2, cc=c.cc_max.values)).groupby(["a", "b"]).cc.agg(["mean", "size"])
    return g.reset_index()


def coherence_pass(s, run, troot, last, blocks, b, nproc, name="coh"):
    d = safe(WORK / s / run / name)
    if (d / "locs.csv").exists():
        return pd.read_csv(d / "locs.csv"), json.loads((d / "coh_info.json").read_text())
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    ok = last[last.status == "LOCATED"].set_index("ID")
    roots = {}
    for i, r in ok.iterrows():
        rt = r["root"] + ".loc"
        if Path(rt + ".octree").exists() and Path(rt + ".hyp").exists():
            roots[i] = rt
    coh = coherences(s, run)
    nb = {}
    for a_, b_, c_ in zip(coh.a.values, coh.b.values, coh["mean"].values):
        if a_ in roots and b_ in roots:
            nb.setdefault(a_, []).append((b_, float(c_)))
            nb.setdefault(b_, []).append((a_, float(c_)))
    jobs = []
    (d / "sc").mkdir()
    (d / "obs").mkdir()
    (d / "ev").mkdir()
    for eid in sorted(roots):
        sc = d / "sc" / f"ev{eid}.stream_coherences"
        lines = ["STREAM_COHERENCES", f"  1.000000 {roots[eid]}"]
        lines += [f"  {c:.6f} {roots[j]}" for j, c in sorted(nb.get(eid, []), key=lambda t: -t[1])]
        sc.write_text("\n".join(lines) + "\n")
        (d / "obs" / f"ev{eid}.obs").write_text(blocks[eid] + "\n")
        cd = d / "ev" / f"ev{eid}"
        cd.mkdir()
        (cd / "nlloc.in").write_text(control(f"{s} {run} coherence ev{eid}", d / "obs" / f"ev{eid}.obs", troot,
                                             cd / "loc", b, posterior=sc))
        jobs.append(cd)
    t = time.time()
    with ThreadPoolExecutor(nproc) as ex:
        rc = list(ex.map(lambda cd: run_bin("NLLoc", cd / "nlloc.in", cd), jobs))
    bad = [j.name for j, r in zip(jobs, rc) if r]
    rows = []
    for cd in jobs:
        f = cd / "loc.sum.grid0.loc.hyp"
        if f.exists():
            rows += parse_sum(f)
        for g in cd.glob("*"):
            if g.suffix in (".scat",) or g.name.startswith("last."):
                g.unlink()
    loc = pd.DataFrame(rows)
    loc["depth"] = true_depth(loc.zf)
    n_nb = np.array([len(nb.get(e, [])) for e in sorted(roots)])
    w = lambda c: np.where(c >= 0.9, 1.0, np.where(c <= COH_MIN, 0.0,
                                                   0.5 * np.cos(np.pi * (1 - (c - COH_MIN) / (0.9 - COH_MIN))) + 0.5))
    se3 = ok.len3
    n_nb_se3 = np.array([sum(1 for j, _ in nb.get(e, []) if MAX_SE3 < 0 or se3[j] <= MAX_SE3)
                         for e in sorted(roots)])
    info = dict(events_with_octree=len(roots), pairs_cc=int(len(coh)), pairs_used=int(sum(n_nb) // 2),
                neighbours_median=float(np.median(n_nb)) if len(n_nb) else 0,
                events_without_neighbours=int((n_nb == 0).sum()),
                neighbours_after_se3_median=float(np.median(n_nb_se3)) if len(n_nb_se3) else 0,
                events_without_neighbours_after_se3=int((n_nb_se3 == 0).sum()),
                coherence_quantiles=[round(float(q), 3) for q in np.percentile(coh["mean"], QS)],
                weight_quantiles=[round(float(q), 3) for q in np.percentile(w(coh["mean"].values), QS)],
                rows_per_pair_median=float(coh["size"].median()), nlloc_nonzero_exit=bad,
                wall_s=round(time.time() - t, 1))
    (d / "coh_info.json").write_text(json.dumps(info, indent=1))
    loc.to_csv(d / "locs.csv", index=False)
    tag = "coherence" if name == "coh" else name
    log(f"{s} {run} {tag}: {int((loc.status == 'LOCATED').sum())}/{len(jobs)} LOCATED, "
        f"median {info['neighbours_median']:g} neighbours ({info['wall_s']:.0f} s)")
    return loc, info


# ------------------------------------------------------------------------------------------------------ outputs
def write_leg(leg, s, run, loc, ids_all, params, wall, notes):
    od = safe(C.out_dir(leg, s, run))
    ok = loc[loc.status == "LOCATED"]
    ok[["ID", "lat", "lon", "depth"]].sort_values("ID").to_csv(od / "reloc.csv", index=False, float_format="%.6f")
    npk = run_picks(s, run).groupby("ID").size()
    st = loc.set_index("ID").status if len(loc) else pd.Series(dtype=str)
    failed = {int(i): (st[i] if i in st.index else
                       f"skipped by NLLoc: {int(npk.get(i, 0))} picks < 5" if npk.get(i, 0) < 5 else
                       "not located (no oct-tree in the last SSST pass)" if leg == "nllcoh" else "no output")
              for i in sorted(set(ids_all) - set(ok.ID))}
    reasons = pd.Series(list(failed.values()), dtype=str).str.replace(r": \d+ picks", ": n picks", regex=True)
    meta = dict(code="NonLinLoc v7.1.07 (NLLoc, Loc2ssst)", leg=leg, set=s, run=run,
                held_out=sorted(C.held_out(s, run)), relocated=int(len(ok)), of=len(C.events(s)),
                events_with_picks=len(ids_all), n_failed=len(failed),
                failed_reasons={k: int(v) for k, v in reasons.value_counts().items()}, failed=failed,
                parameters=params, wall_s=round(wall, 1), notes=notes,
                script="workflow/10_tomography/b5_cmp_nll.py", utc=datetime.now(timezone.utc).isoformat())
    (od / "run.json").write_text(json.dumps(meta, indent=1) + "\n")


def params_common(s, b, src):
    return dict(box=dict(LOCGRID=grid_line(b), extent_km=extent(b), margin_km=MARGIN),
                time0=dict(fine=str(FINE), wide=str(WIDE), n_fine=sum(v == "fine" for v in src.values()),
                           n_wide=sum(v == "wide" for v in src.values()), cropped="box + 1 node, values unchanged"),
                nlloc="LOCMETH EDT_OT_WT 9999.0 5 -1 -1 -1 -1 -1.0 1 (production: 20 grids in memory); LOCGAU 0.2 0.0; "
                      f"no LOCGAU2; LOCTOPO_SURFACE {TOPO} 0; {search_line(b)}",
                pick_error="GAU sigma of picks(s) (D-B008 classes)")


def pass_times(s, run):
    """wall time (s) of each pass of a run from its run.log (last occurrence; survives resumed runs)."""
    out = {}
    for line in open(WORK / s / run / "run.log"):
        m = re.search(rf"^\S+ {s} {run} (it\d+|Loc2ssst pass \d+|coherence)\b.*\((\d+) s", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    base = out.get("it0", np.nan)
    ssst = base + sum(out.get(f"Loc2ssst pass {k}", np.nan) + out.get(f"it{k + 1}", np.nan)
                      for k in range(len(SCHEDULE)))
    return dict(nllbase=base, nllssst=ssst, nllcoh=ssst + out.get("coherence", np.nan), passes=out)


def do_run(s, run, nproc):
    global LOG
    rd = safe(WORK / s / run)
    rd.mkdir(parents=True, exist_ok=True)
    LOG = rd / "run.log"
    b = box(s)
    troot, src = time0(s)
    blocks = obs_blocks(s, run)
    ids_all = sorted(blocks)
    t_all = time.time()
    log(f"{s} {run}: {len(blocks)} events with picks, {sum(b_.count(chr(10)) - 1 for b_ in blocks.values())} picks, "
        f"held out {sorted(C.held_out(s, run))}")
    # baseline = it0
    loc0 = nlloc_pass(s, run, 0, troot, blocks, b, stat=True, octree=False, nproc=nproc)
    pc = params_common(s, b, src)
    write_leg("nllbase", s, run, loc0, ids_all, dict(pc, static_terms=str(STAT)), pass_times(s, run)["nllbase"],
              "NLLoc baseline: v5.1 control settings and b4_v51_iter0 static terms, picks(s) of the run")
    # SSST
    tin, l2s = troot, []
    loc = loc0
    for k, D in enumerate(SCHEDULE):
        hyp = WORK / s / run / f"it{k}" / "all.hyp"
        tdir = WORK / s / run / f"T{k + 1}"
        l2s.append(loc2ssst(s, run, k, D, tin, tdir, hyp, b, nproc))
        tin = tdir / "cascadia3d"
        loc = nlloc_pass(s, run, k + 1, tin, blocks, b, stat=False, octree=(k + 1 == len(SCHEDULE)), nproc=nproc)
    ps = dict(pc, schedule_km=SCHEDULE, weight_floor=WEIGHT_FLOOR, lsphstat=LSPHSTAT, lsgrid_dx_km=DXS,
              loc2ssst=[{k: v for k, v in x.items() if k != "ssst_median_min_max"} for x in l2s],
              passes=len(SCHEDULE) + 1, start="it0 = nllbase (static terms of b4_v51_iter0 enter the first "
                                                "correction as the arrival delay)")
    write_leg("nllssst", s, run, loc, ids_all, ps, pass_times(s, run)["nllssst"],
              f"SSST only: NLLoc pass it{len(SCHEDULE)} with the time grids of Loc2ssst pass {len(SCHEDULE) - 1}")
    # coherence
    lc, info = coherence_pass(s, run, tin, loc, blocks, b, nproc)
    pcoh = dict(ps, coherence=dict(rule="mean cc_max over the pair's accepted CC rows (b5_cmp_common.cc_rows) at "
                                        "training stations; pairs without rows: no coherence",
                                   locposterior=f"LOCPOSTERIOR OCT_TREE <ev>.stream_coherences 1e-30 {COH_MIN} "
                                                f"{COH_TOTAL} {MAX_SE3} {COH_NMAX} {COH_DMAG} {COH_MINMAG}",
                                   weight="NLLoc: 0 at CoherenceMin, 1 at >= 0.9, cosine taper; self weight 1",
                                   search=search_line(b, coh=True), **info))
    pt = pass_times(s, run)
    write_leg("nllcoh", s, run, lc, ids_all, dict(pcoh, pass_wall_s=pt["passes"]), pt["nllcoh"],
              "SSST + coherence: one NLLoc run per event with its coherent neighbours' oct-tree PDFs stacked "
              "(LOCPOSTERIOR), time grids of the last SSST pass")
    log(f"{s} {run}: done in {time.time() - t_all:.0f} s (this invocation; passes {pt['passes']})")


def variant(s, run, k, nproc):
    """diagnostic (not a leg output): the coherence pass on top of SSST pass it<k> instead of the last one. Reruns
    it<k> with oct-trees as var_it<k> (time grids T<k>; time0 + static terms for k = 0) and then var_coh<k>.
    Scored by --pass-scores."""
    global LOG
    rd = safe(WORK / s / run)
    LOG = rd / "run.log"
    b = box(s)
    troot, _ = time0(s)
    blocks = obs_blocks(s, run)
    tk = troot if k == 0 else WORK / s / run / f"T{k}" / "cascadia3d"
    loc = nlloc_pass(s, run, k, tk, blocks, b, stat=(k == 0), octree=True, nproc=nproc, name=f"var_it{k}")
    coherence_pass(s, run, tk, loc, blocks, b, nproc, name=f"var_coh{k}")


def score():
    """CMP/nllcoh/scores.json: per set, score_set("nllcoh") as the other legs write it ("v5.1", "nllcoh", "full"),
    plus "nllssst" / "full_nllssst" and "nllbase" / "full_nllbase" from score_set of those legs."""
    out = {}
    for s in C.SETS:
        out[s] = {}
        for leg in ("nllcoh", "nllssst", "nllbase"):
            if C.read_reloc(leg, s, "full") is None:
                continue
            r = C.score_set(leg, s)
            print(s, leg, json.dumps(r), flush=True)
            if leg == "nllcoh":
                out[s].update(r)
            else:
                out[s].setdefault("v5.1", r.get("v5.1"))
                out[s][leg] = r.get(leg)
                out[s][f"full_{leg}"] = r.get("full")
    f = safe(C.CMP / "nllcoh" / "scores.json")
    f.write_text(json.dumps(out, indent=1) + "\n")
    print("->", f)


def pass_scores(s):
    """diagnostic: held-out CC scores (as score_set, pooled over the folds) of every NLLoc pass it0..it<N>, of the
    coherence pass and of the variants, from the scratch folders -> WORK/<set>/pass_scores.json. "v5.1_same" scores
    the v5.1 positions on the events the coherence pass located in each fold (the same CC rows as "coh")."""
    cc = C.cc_rows(s)
    ev = C.events(s).set_index("ID")
    tags = [f"it{k}" for k in range(len(SCHEDULE) + 1)] + ["coh"]
    tags += [f"var_coh{k}" for k in range(len(SCHEDULE))]
    out = {}
    for tag in ["v5.1", "v5.1_same"] + tags:
        rows, n = [], []
        for run in C.runs()[1:]:
            if tag == "v5.1":
                pos = ev
            elif tag == "v5.1_same":
                c = pd.read_csv(WORK / s / run / "coh" / "locs.csv")
                pos = ev.loc[c[c.status == "LOCATED"].ID]
            else:
                f = WORK / s / run / tag / "locs.csv"
                if not f.exists():
                    continue
                pos = pd.read_csv(f)
                pos = pos[pos.status == "LOCATED"].set_index("ID")[["lat", "lon", "depth"]]
            n.append(len(pos))
            rows.append(C.fold_score(cc, pos, C.held_out(s, run)))
        if not rows:
            continue
        H = pd.concat(rows)
        d = dict(folds=len(rows), located=n)
        for ph, g in H.groupby("phase"):
            v = g.score.values
            d[ph] = dict(n=len(v), rms_ms=round(float(np.sqrt(np.mean(v ** 2))) * 1e3, 2),
                         mad_ms=round(float(np.median(np.abs(v - np.median(v)))) * 1e3, 2))
        out[tag] = d
        print(s, tag, json.dumps(d), flush=True)
    (WORK / s / "pass_scores.json").write_text(json.dumps(out, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sets", nargs="*")
    ap.add_argument("--runs", default=",".join(C.runs()))
    ap.add_argument("--nproc", type=int, default=NPROC)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--pass-scores", action="store_true", help="diagnostic scores of every pass of SETS")
    ap.add_argument("--variant", type=int, help="diagnostic: coherence on top of SSST pass it<K> (see variant())")
    a = ap.parse_args()
    assert a.nproc <= 16
    if a.score:
        score()
        return
    if a.pass_scores:
        for s in a.sets:
            pass_scores(s)
        return
    if a.variant is not None:
        for s in a.sets:
            for run in a.runs.split(","):
                variant(s, run, a.variant, a.nproc)
        return
    for s in a.sets:
        for run in a.runs.split(","):
            do_run(s, run, a.nproc)


if __name__ == "__main__":
    main()

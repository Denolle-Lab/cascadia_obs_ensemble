"""D-B078 comparison, tomoDD leg: tomoDD-SE relocation-only with the D-B081 floor, in model v0.3.2 (B5 MOD), on the
common test sets, folds and data of b5_cmp_common.py; catalog dt (D-B061..D-B064 weights) + waveform CC dt with
hierarchical weighting (catalog first, CC dominant in the later sets, as hypoDD practice).

Inputs of each set come from its source cluster run in v0.3.2 with the floor (run b5_v5.1_s4: MOD, station.dat, frame
centre, binary), restricted to the set's events (dt.ct, absolute.dat, event.dat), plus dt.cc from cc_rows(s) (station
names mapped to the run's station codes; QUAL = cc_max). In run foldK every dt.ct, dt.cc and absolute row at the held-out
stations is removed. The floor (surface.dat, b5_relocate.write_floor) is written for the final frame: the set's mean
(tomoDD - v5.1) shift is removed afterwards (D-B053), so runs repeat with the floor offset by that shift until it holds
(at most 3), then the vertical shift is limited by the floor if needed (as b5_relocate._do_cluster).

  python b5_cmp_tomodd.py [--damp D] [--obscc N] [SET ...]
      -> <cmp>/<leg>/<set>/<run>/{reloc.csv, run.json, work/}, scores.json; leg = tomodd (damp 100, OBSCC 0) or
         tomodd_d<D>_o<N> for other settings (pre-declared variants: damping 100 / 300, OBSCC 0 / 4)
"""
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b5_cmp_common as C  # noqa: E402
import b5_relocate as BR  # noqa: E402
from b5_common import S_OVER_P  # noqa: E402
from b5_wtest import read_dt, read_events, write_dt, write_events  # noqa: E402

LEG = "tomodd"
SRC = C.B5 / "b5_v5.1_s4" / "clusters"
SOURCE = {"axial": "c000001_p001", "blanco": "c000010", "msh": "c000007", "jdf": "c000032"}
DAMP = 100
SW = BR.B["tomodd"]["s_weight"] * S_OVER_P ** 2      # catalog S weight factor (D-B061)
# NITER WTCCP WTCCS WRCC WDCC WTCTP WTCTS WRCT WDCT WTCD: catalog first, then CC dominant (catalog weight / 100)
SETS = [[3, 0.01, 0.01, -9, -9, 1.0, 1.0 * SW, -9, -9, 0.0],
        [3, 1.0, 1.0, -9, -9, 0.01, 0.01 * SW, 8, -9, 0.0],
        [3, 1.0, 1.0, 6, -9, 0.01, 0.01 * SW, 6, 10, 0.0],
        [3, 1.0, 1.0, 6, 2, 0.01, 0.01 * SW, 6, 10, 0.0]]
MAX_PASSES = 3


def base(s):
    """Source cluster inputs restricted to the set's events; station code map; frame centre; binary."""
    src = SRC / SOURCE[s]
    rj = json.loads((src / "run.json").read_text())
    ids = set(C.events(s).ID)
    ev = read_events(src / "Input_Files" / "event.dat")
    ev = ev[ev.ID.isin(ids)]
    assert len(ev) == len(ids), (s, len(ev), len(ids))
    dt = read_dt(src / "Input_Files" / "dt.ct")
    dt = dt[dt.e1.isin(ids) & dt.e2.isin(ids)]
    ab, cur = [], None
    for l in (src / "Input_Files" / "absolute.dat").read_text().splitlines():
        if l.startswith("#"):
            cur = int(l.split()[1])
        if cur in ids:
            ab.append(l)
    st = pd.read_csv(C.PREP / "stations.csv")
    codes = set(pd.read_csv(src / "Input_Files" / "station.dat", sep=r"\s+", header=None)[0])
    name2code = {r.station: r.code for r in st.itertuples() if r.code in codes}
    return dict(src=src, rj=rj, ev=ev, dt=dt, ab=ab, name2code=name2code)


def write_cc(path, cc, name2code):
    cc = cc[cc.station.isin(name2code)].copy()
    cc["code"] = cc.station.map(name2code)
    out = []
    for (a, b), g in cc.groupby(["ID1", "ID2"], sort=True):
        out.append(f"# {a} {b} 0.0")
        out += [f"{r.code:<7s} {r.dt_cc:10.5f} {r.cc_max:6.4f} {r.phase}" for r in g.itertuples()]
    Path(path).write_text("\n".join(out) + "\n")
    return len(cc), cc.groupby(["ID1", "ID2"]).ngroups


def write_inp(d, src, center):
    L = (src / "tomoDD-SE.inp").read_text().splitlines()
    i = next(k for k, l in enumerate(L) if l.startswith("* cross correlation diff times"))
    L[i + 1] = "Input_Files/dt.cc"
    j = next(k for k, l in enumerate(L) if l.startswith("* IDAT"))
    L[j + 1] = re.sub(r"^\s*\d+", "   3", L[j + 1])                       # catalog + CC
    k = next(k for k, l in enumerate(L) if l.startswith("* OBSCC"))
    L[k + 1] = f"    {OBSCC}      0     {BR.B['tomodd']['floor']['air_dep_km']:.3f}"
    m = next(k for k, l in enumerate(L) if l.startswith("*  ISTART"))
    L[m + 1] = re.sub(r"(\d+)\s*$", str(len(SETS)), L[m + 1])
    n = next(k for k, l in enumerate(L) if l.startswith("* NITER"))
    old = [k for k in range(n + 1, len(L)) if re.match(r"^\s+\d+\s+[-\d.]+\s+[-\d.]+", L[k])]
    thres = BR.T["inp"]["thres"]
    rows = [f"  {s[0]:<3d} {s[1]:5.2f} {s[2]:5.2f} {s[3]:4g} {s[4]:4g} {s[5]:5.2f} {s[6]:5.2f} {s[7]:4g} {s[8]:4g} "
            f"{s[9]:6g} {DAMP:6g}   0  {thres:g}" for s in SETS]
    L[old[0]:old[-1] + 1] = rows
    (d / "tomoDD-SE.inp").write_text("\n".join(L) + "\n")


START = "v5.1"     # or "nllbase": every run starts from NonLinLoc relocated without its held-out stations


def start_positions(s, run):
    """Start positions (index ID; lat, lon, depth): the v5.1 hypocentres, or (START = nllbase) the D-B078 NonLinLoc
    baseline of the same run (training stations only), so that no held-out station enters the start (pipeline test)."""
    ev = C.events(s).set_index("ID")
    if START == "v5.1":
        return ev[["lat", "lon", "depth"]]
    r = C.read_reloc("nllbase", s, run)
    assert r is not None, (s, run)
    return r[["lat", "lon", "depth"]]


def run_once(d, b, cc, out_sta, shift):
    shutil.rmtree(d, ignore_errors=True)
    (d / "Input_Files").mkdir(parents=True)
    (d / "Output_Files").mkdir()
    src = b["src"]
    for f in ("MOD", "ak135.15.SKS", "layer-16.dat"):
        shutil.copyfile(src / f, d / f)
    shutil.copyfile(src / "Input_Files" / "station.dat", d / "Input_Files" / "station.dat")
    out_codes = {b["name2code"][s] for s in out_sta if s in b["name2code"]}
    ev = b["ev"]
    st0 = b["start"]
    if START != "v5.1":                       # start = NonLinLoc baseline of the run, in tomoDD's rescaled frame
        from b2_fwd_mods import ellipsoid_coords
        ev = ev[ev.ID.isin(st0.index)].copy()
        la, lo = ellipsoid_coords(st0.loc[ev.ID, "lat"].values, st0.loc[ev.ID, "lon"].values, *b["rj"]["center"])
        ev["lat"], ev["lon"], ev["depth"] = la, lo, st0.loc[ev.ID, "depth"].values
    write_events(d / "Input_Files" / "event.dat", ev)
    dt = b["dt"][~b["dt"].sta.isin(out_codes)]
    write_dt(d / "Input_Files" / "dt.ct", dt)
    ab = [l for l in b["ab"] if l.startswith("#") or l.split()[0] not in out_codes]
    (d / "Input_Files" / "absolute.dat").write_text("\n".join(ab) + "\n")
    n_cc, n_ccp = write_cc(d / "Input_Files" / "dt.cc", cc[~cc.station.isin(out_sta)], b["name2code"])
    center = b["rj"]["center"]
    BR.write_floor(d / "surface.dat", st0.lat.values, st0.lon.values, center, shift)
    write_inp(d, src, center)
    t0 = time.time()
    with open(d / "stdout.txt", "w") as so:
        rc = subprocess.run([b["rj"]["bin"], "tomoDD-SE.inp"], cwd=d, stdout=so, stderr=subprocess.STDOUT,
                            timeout=7200).returncode
    lg = (d / "tomoDD-SE.log").read_text(errors="replace") if (d / "tomoDD-SE.log").exists() else ""
    info = dict(exit=rc, wall_s=round(time.time() - t0, 1), n_dt_ct=len(dt), n_cc=n_cc, n_cc_pairs=n_ccp,
                floor_hits=len({l.split()[1] for l in lg.splitlines() if l.startswith(">>>Floor: ")}),
                airquakes=len({l.split("-")[-1] for l in lg.splitlines() if "negative depth" in l}),
                cnd=[float(x) for x in re.findall(r"CND\s*=?\s*([\d.]+)", lg)][-1:] or None)
    rel = d / "Output_Files" / "tomoDD.reloc"
    r = BR.read_reloc(rel, center, ev.ID.values) if rc == 0 and rel.exists() and rel.stat().st_size else None
    return r, info


def relocate(s, run, b, cc):
    out = C.out_dir(LEG, s, run)
    ev0 = start_positions(s, run)
    b["start"] = ev0
    shift, passes, r, info = (0.0, 0.0, 0.0), [], None, {}
    for p in range(MAX_PASSES):
        r, info = run_once(out / "work", b, cc, C.held_out(s, run), shift)
        if r is None:
            break
        st = ev0.loc[r.ID]
        sh = (float((r.lat.values - st.lat.values).mean()), float((r.lon.values - st.lon.values).mean()),
              float((r.depth.values - st.depth.values).mean()))
        on, above = BR.floor_check(r, sh)
        passes.append(dict(given=list(shift), shift=list(sh), on_floor=int(on.sum()), above=int(above.sum()), **info))
        if not above.any():
            break
        shift = sh
    anchor = None
    if r is not None and passes and passes[-1]["above"] > 0:
        from b4_catalog import floor_at
        sh = passes[-1]["shift"]
        fdep = -floor_at(BR.floor_grid(), r.lat.values - sh[0], r.lon.values - sh[1]) / 1000.0
        sz = min(sh[2], float(np.min(r.depth.values - fdep)) - 1e-4)
        anchor = [sh[0], sh[1], sz]
        passes[-1]["anchor_z_limited_km"] = sh[2] - sz
        sh = anchor
    else:
        sh = passes[-1]["shift"] if passes else None
    rec = dict(leg=LEG, set=s, run=run, source_cluster=SOURCE[s], bin=b["rj"]["bin"], center=b["rj"]["center"],
               model="v0.3.2 (B5 MOD, water/air nodes rock-continued)", floor="D-B081", damp=DAMP, sets=SETS,
               held_out=sorted(C.held_out(s, run)), passes=passes, anchor=anchor, start=START)
    if r is not None:
        f = pd.DataFrame({"ID": r.ID.values, "lat": r.lat.values - sh[0], "lon": r.lon.values - sh[1],
                          "depth": r.depth.values - sh[2]})
        f.to_csv(out / "reloc.csv", index=False, float_format="%.8g")
        rec["relocated"] = len(f)
        rec["not_relocated"] = sorted(set(ev0.index) - set(f.ID))
    (out / "run.json").write_text(json.dumps(rec, indent=1, default=float))
    return rec


OBSCC = 0


def main(sets):
    scores = {}
    for s in sets:
        b = base(s)
        b["set"] = s
        cc = C.cc_rows(s)
        for run in C.runs():
            rec = relocate(s, run, b, cc)
            print(s, run, "relocated", rec.get("relocated"), "passes", [(p["on_floor"], p["above"]) for p in rec["passes"]],
                  "cc rows", rec["passes"][-1]["n_cc"] if rec["passes"] else None, flush=True)
        scores[s] = C.score_set(LEG, s)
        print(json.dumps({s: scores[s]}), flush=True)
    f = C.CMP / LEG / "scores.json"
    old = json.loads(f.read_text()) if f.exists() else {}
    old.update(scores)
    f.write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="*")
    ap.add_argument("--damp", type=float, default=100)
    ap.add_argument("--obscc", type=int, default=0)
    ap.add_argument("--start", default="v5.1", choices=["v5.1", "nllbase"])
    a = ap.parse_args()
    DAMP, OBSCC, START = a.damp, a.obscc, a.start
    if (DAMP, OBSCC) != (100, 0):
        LEG = f"tomodd_d{DAMP:g}_o{OBSCC}"
    if START == "nllbase":
        LEG += "_snll"
    main(a.sets or list(C.SETS))

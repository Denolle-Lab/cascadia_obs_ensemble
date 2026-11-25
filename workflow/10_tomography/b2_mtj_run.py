"""MTJ shallow-anomaly tests (notes/mtj_shallow.md): one tomoDD-SE run on a variant of the Mendocino-Gorda v2 inputs.

Run directory <runs>/mendocino_gorda_v2/mtj_tests/<tag>/ (Input_Files -> mtj_tests/inputs_<variant>/, MOD copy),
tomoDD-SE.inp from b2_tomodd_run.INP with the baseline settings (damp 1000, smooth 20/20/10, THRES 0.2, 4 sets of
3 iterations, WTCD 10/10/1/0.1) unless overridden. Back-transform as b2_tomodd_backtransform.py.

Input variants (built once, reused):
  base            the v2 inputs as they are (symlinks)
  no_shelf5       absolute and differential data of G22D, FS07D, FS02B, BB660, BB200 removed (largest P-4)
  no_mtj_shallow  events in the MTJ box (40.0-40.65 N, 125.3-123.9 W) shallower than 10 km (B1b depth) removed
  stastat         absolute times corrected by a per-station-phase static = median absolute residual after a
                  location-only tomoDD run in the starting model (tag loconly; JOINT 0) minus the median of these
                  medians over all stations of that phase (network-wide offset left to the model). Needs loconly.
  stastat_raw     same without removing the network-wide median
  dropbad         base data minus all absolute and differential data of station-phases whose stastat static
                  exceeds DROP_S = 0.5 s in magnitude
  mod_<name>      base data with MOD replaced by mtj_tests/mods/MOD_<name> (b2_mtj_mods.py)
Variants combine with '+', e.g. stastat+mod_sed.

  python b2_mtj_run.py TAG --variant V [--damp D] [--smooth WX WY WZ] [--thres T [T T T]] [--wtcd W W W W]
                       [--joint J] [--niter N N N N]
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import write_manifest  # noqa: E402
from b2_tomodd_run import INP  # noqa: E402
from b2_tomodd_backtransform import table, read_model_dat  # noqa: E402
from b2_mtj_common import BASE, INPUTS, TESTS, T, read_res, stations, inverse_coords  # noqa: E402
from b2_fwd_common import read_mod  # noqa: E402

SHELF5 = ["G22D.7D", "FS07D.7D", "FS02B.7D", "BB660.Z5", "BB200.X9"]
DROP_S = 0.5
MTJ_BOX = dict(lat=(40.0, 40.65), lon=(-125.3, -123.9), zmax=10.0)


def read_blocks(path):
    """absolute.dat / dt.ct -> list of (header line, [data lines])."""
    blocks = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            blocks.append([line, []])
        elif line.strip():
            blocks[-1][1].append(line)
    return blocks


def write_blocks(path, blocks, min_lines=1):
    with open(path, "w") as f:
        for h, ls in blocks:
            if len(ls) >= min_lines:
                f.write(h + "\n" + "".join(l + "\n" for l in ls))


def build_variant(name):
    out = TESTS / f"inputs_{name}"
    if (out / "done").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    parts = name.split("+")
    ev = (INPUTS / "event.dat").read_text().splitlines()
    ab = read_blocks(INPUTS / "absolute.dat")
    dt = read_blocks(INPUTS / "dt.ct")
    mod = INPUTS / "MOD"
    notes = []
    for p in parts:
        if p == "base":
            continue
        if p == "no_shelf5":
            st = stations()
            codes = set(st[st.station.isin(SHELF5)].code)
            assert len(codes) == 5, codes
            n0 = sum(len(b[1]) for b in ab) + sum(len(b[1]) for b in dt)
            ab = [[h, [l for l in ls if l.split()[0] not in codes]] for h, ls in ab]
            dt = [[h, [l for l in ls if l.split()[0] not in codes]] for h, ls in dt]
            n1 = sum(len(b[1]) for b in ab) + sum(len(b[1]) for b in dt)
            notes.append(f"no_shelf5: removed {n0 - n1} data of {sorted(codes)}")
        elif p == "no_mtj_shallow":
            e = pd.read_csv(INPUTS / "events.csv")
            m = (e.lat.between(*MTJ_BOX["lat"]) & e.lon.between(*MTJ_BOX["lon"])
                 & (e.depth_bsl_km < MTJ_BOX["zmax"]))
            drop = set(e.ID[m].astype(int))
            ev = [l for l in ev if int(l.split()[9]) not in drop]
            ab = [b for b in ab if int(b[0].split()[1]) not in drop]
            dt = [b for b in dt if int(b[0].split()[1]) not in drop and int(b[0].split()[2]) not in drop]
            notes.append(f"no_mtj_shallow: removed {len(drop)} events {MTJ_BOX}")
        elif p in ("stastat", "stastat_raw", "dropbad"):
            r = read_res(TESTS / "loconly")
            a = r[r.absolute & (r.wt > 0)]
            c = (a.groupby(["code", "phase"]).res_ms.median() / 1000.0).rename("static_s")
            if p == "stastat" or p == "dropbad":     # remove the network-wide median per phase (left to the model)
                c = c - c.groupby(level=1).transform("median")
            c.reset_index().round(4).to_csv(out / f"{p}.csv", index=False)
            if p == "dropbad":
                bad = set(c[c.abs() > DROP_S].index)
                n0 = sum(len(b[1]) for b in ab) + sum(len(b[1]) for b in dt)
                ab = [[h, [l for l in ls if (l.split()[0], l.split()[3]) not in bad]] for h, ls in ab]
                dt = [[h, [l for l in ls if (l.split()[0], l.split()[4]) not in bad]] for h, ls in dt]
                n1 = sum(len(b[1]) for b in ab) + sum(len(b[1]) for b in dt)
                notes.append(f"dropbad: {len(bad)} station-phases with |static - phase median| > {DROP_S} s removed "
                             f"({n0 - n1} data): {sorted(bad)}")
                continue
            new = []
            for h, ls in ab:
                o = []
                for l in ls:
                    t = l.split()
                    s = c.get((t[0], t[3]), 0.0)
                    o.append(f"{t[0]:<7s} {float(t[1]) - s:9.4f} {t[2]:>6s} {t[3]}")
                new.append([h, o])
            ab = new
            notes.append(f"stastat: {len(c)} station-phase statics subtracted from absolute times, "
                         f"median P {c.xs('P', level=1).median():+.3f} S {c.xs('S', level=1).median():+.3f} s")
        elif p.startswith("mod_"):
            mod = TESTS / "mods" / f"MOD_{p[4:]}"
            assert mod.exists(), mod
            notes.append(f"MOD from {mod}")
        else:
            raise SystemExit(f"unknown variant part {p}")
    (out / "event.dat").write_text("\n".join(ev) + "\n")
    write_blocks(out / "absolute.dat", ab)
    write_blocks(out / "dt.ct", dt)
    for f in ("station.dat",):
        if not (out / f).exists():
            (out / f).symlink_to(INPUTS / f)
    shutil.copy(mod, out / "MOD")
    shutil.copy(INPUTS / "summary.json", out / "summary.json")
    (out / "variant.txt").write_text("\n".join(notes) + "\n")
    (out / "done").write_text("ok\n")
    print("\n".join(notes))
    return out


def backtransform(run):
    fw = json.loads((INPUTS / "forward.json").read_text())
    o = run / "Output_Files"
    for name in ("tomoDD.reloc", "tomoDD.loc"):
        if (o / name).exists():
            table(o / name, o / f"{name}_geo", 1, 2, fw)
    if (o / "tomoDD.sta").exists():
        table(o / "tomoDD.sta", o / "tomoDD.sta_geo", 1, 2, fw, "{:.5f}")
    m = read_mod(run / "MOD")
    lat = inverse_coords(m["lat"], np.full_like(m["lat"], -125.0))[0]
    lon = inverse_coords(np.full_like(m["lon"], 41.3), m["lon"])[1]
    out = dict(lon=lon, lat=lat, z=m["z"])
    for ph in ("Vp", "Vs"):
        f = o / f"{ph}_model.dat"
        if f.exists():
            out[ph.lower()] = read_model_dat(f, len(m["lon"]), len(m["lat"]), len(m["z"]))
    np.savez_compressed(o / "model_geo.npz", **out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--damp", type=float, default=1000.0)
    ap.add_argument("--smooth", type=float, nargs=3, default=[20, 20, 10])
    ap.add_argument("--thres", type=float, nargs="+", default=[0.2])
    ap.add_argument("--wtcd", type=float, nargs="+", default=None)
    ap.add_argument("--niter", type=int, nargs="+", default=None)
    ap.add_argument("--joint", type=int, default=1)
    a = ap.parse_args()

    sets = [list(s) for s in T["inp"]["sets"]]           # NITER WTCTP WTCTS WRCT WDCT WTCD
    if a.niter:
        sets = [s for s, n in zip(sets, a.niter + [0] * 4) if n > 0]
        for s, n in zip(sets, [n for n in a.niter if n > 0]):
            s[0] = n
    if a.wtcd:
        assert len(a.wtcd) == len(sets)
        for s, w in zip(sets, a.wtcd):
            s[5] = w
    thres = a.thres * len(sets) if len(a.thres) == 1 else a.thres
    assert len(thres) == len(sets)

    inp_dir = build_variant(a.variant)
    summ = json.loads((inp_dir / "summary.json").read_text())
    run = TESTS / a.tag
    if (run / "Output_Files" / "tomoDD.reloc").exists():
        raise SystemExit(f"{run} already has results; use a new tag")
    (run / "Output_Files").mkdir(parents=True, exist_ok=True)
    (run / "Input_Files").mkdir(exist_ok=True)
    for f in ("event.dat", "station.dat", "absolute.dat", "dt.ct"):
        dst = run / "Input_Files" / f
        if not dst.exists():
            dst.symlink_to(inp_dir / f)
    shutil.copy(inp_dir / "MOD", run / "MOD")
    bindir = Path(T["bin"]).parent
    for f in ("ak135.15.SKS", "layer-16.dat"):
        shutil.copy(bindir / f, run / f)
    rows = [f"  {s[0]:<3d}  -9    -9    -9   -9  {s[1]:5.2f} {s[2]:5.2f} {s[3]:4g} {s[4]:4g} "
            f"{s[5]:6g} {a.damp:6g}   {a.joint}  {th:g}" for s, th in zip(sets, thres)]
    w1, w2, w3 = (f"{v:g}" for v in a.smooth)
    (run / "tomoDD-SE.inp").write_text(INP.format(
        region="mendocino_gorda_v2 mtj_tests", tag=a.tag, air=summ["air_dep_km"], nset=len(rows),
        stepl=T["inp"]["stepl_km"], clat=summ["center_lat"], clon=summ["center_lon"], w1=w1, w2=w2, w3=w3,
        sets="\n".join(rows)))
    t0 = time.time()
    with open(run / "stdout.txt", "w") as so, open(run / "time.txt", "w") as se:
        rc = subprocess.run(["/usr/bin/time", "-f", "%e s %U user %M KB", T["bin"], "tomoDD-SE.inp"],
                            cwd=run, stdout=so, stderr=se).returncode
    wall = time.time() - t0
    if (run / "Output_Files" / "tomoDD.reloc").exists():
        backtransform(run)
    print(f"{a.tag}: exit {rc}, {wall / 60:.1f} min, {(run / 'time.txt').read_text().strip()[-60:]}")
    outs = sorted((run / "Output_Files").glob("*")) + [run / "tomoDD-SE.inp", run / "tomoDD-SE.log"]
    write_manifest(f"b2_mtj_run_{a.tag}", run, inputs=[inp_dir / "MOD", inp_dir / "absolute.dat", inp_dir / "dt.ct",
                                                       inp_dir / "event.dat", T["bin"]],
                   outputs=[o for o in outs if o.is_file() and o.stat().st_size < 3e8],
                   params={"tag": a.tag, "variant": a.variant, "damp": a.damp, "smooth": a.smooth, "thres": thres,
                           "sets": sets, "joint": a.joint, "binary": T["bin"]},
                   counts={"exit": rc, "wall_min": round(wall / 60, 2)})
    sys.exit(rc)


if __name__ == "__main__":
    main()

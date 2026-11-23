"""Track B, step B2: set up and run one tomoDD-SE inversion (D-B026, D-B027).

Creates <runs>/<region>/<tag>/ with Input_Files -> ../inputs (symlinks), MOD (copy), the global 1D files,
and tomoDD-SE.inp written from params.yaml `tomodd.inp` (overrides on the command line), runs the binary,
and writes a manifest with the run's settings and output checksums.

  python b2_tomodd_run.py REGION TAG [--damp D] [--smooth WX WY WZ] [--thres T] [--check] [--niter-first N]
    --check: bounds-checked binary, first iteration set only, 1 iteration (array-overflow test, D-B026)

Run from the repo root with the default env (LD_LIBRARY_PATH=.pixi/envs/default/lib).
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import load_params, write_manifest  # noqa: E402

P = load_params()
T = P["tomodd"]

INP = """* tomoDD-SE.inp written by b2_tomodd_run.py ({region} / {tag})
*--- input file selection
* cross correlation diff times:

*
*catalog diff times:
Input_Files/dt.ct
*
* event file:
Input_Files/event.dat
*
* station file:
Input_Files/station.dat
*
*--- output file selection
* original locations:
Output_Files/tomoDD.loc
* relocations:
Output_Files/tomoDD.reloc
* station information:
Output_Files/tomoDD.sta
* residual information:
Output_Files/tomoDD.res
* source paramater information:
Output_Files/tomoDD.src
* Velocity file
Output_Files/tomoDD.vel
* Vp Model
Output_Files/Vp_model.dat
* Vs Model
Output_Files/Vs_model.dat
* Absolute file
Input_Files/absolute.dat
*
*--- data type selection:
* IDAT   IPHA   DIST
   2      3     8000
*
*--- event clustering:
* OBSCC  OBSCT   Air_dep
    0      0     {air:.3f}
*
*--- solution control:
*  ISTART  ISOLV  NSET
     2       2     {nset}
* iuses iuseq invdel  stepl
    2     0     0       {stepl}
* wlat   wlon   rota
  {clat}  {clon}   0
* weight1 weight2  weight3  CC_format
    {w1}    {w2}    {w3}       1
*
*--- data weighting and re-weighting:
* NITER WTCCP WTCCS WRCC WDCC WTCTP WTCTS WRCT WDCT WTCD DAMP JOINT THRES
{sets}
*--- event selection:
* CID
    0
* ID
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("region")
    ap.add_argument("tag")
    ap.add_argument("--damp", type=float, default=T["inp"]["damp"])
    ap.add_argument("--smooth", type=float, nargs=3, default=T["inp"]["smooth"])
    ap.add_argument("--thres", type=float, default=T["inp"]["thres"])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    base = Path(T["runs"]) / a.region
    inp_dir = base / "inputs"
    summ = json.loads((inp_dir / "summary.json").read_text())
    run = base / a.tag
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

    sets = T["inp"]["sets"][:1] if a.check else T["inp"]["sets"]
    rows = []
    for s in sets:
        niter = 1 if a.check else s[0]
        rows.append(f"  {niter:<3d}  -9    -9    -9   -9  {s[1]:5.2f} {s[2]:5.2f} {s[3]:4g} {s[4]:4g} "
                    f"{s[5]:6g} {a.damp:6g}   1  {a.thres:g}")
    w1, w2, w3 = (f"{v:g}" for v in a.smooth)
    (run / "tomoDD-SE.inp").write_text(INP.format(
        region=a.region, tag=a.tag, air=summ["air_dep_km"], nset=len(rows), stepl=T["inp"]["stepl_km"],
        clat=summ["center_lat"], clon=summ["center_lon"], w1=w1, w2=w2, w3=w3, sets="\n".join(rows)))

    binary = T["bin_check"] if a.check else T["bin"]
    t0 = time.time()
    with open(run / "stdout.txt", "w") as so, open(run / "time.txt", "w") as se:
        rc = subprocess.run(["/usr/bin/time", "-f", "%e s %U user %M KB", binary, "tomoDD-SE.inp"],
                            cwd=run, stdout=so, stderr=se).returncode
    wall = time.time() - t0
    print(f"{a.region}/{a.tag}: exit {rc}, {wall / 60:.1f} min, {(run / 'time.txt').read_text().strip()[-60:]}")
    outs = sorted((run / "Output_Files").glob("*")) + [run / "tomoDD-SE.inp", run / "tomoDD-SE.log", run / "stdout.txt"]
    write_manifest(f"b2_tomodd_run_{a.region}_{a.tag}".replace("/", "_"), run, inputs=[inp_dir / "summary.json", inp_dir / "MOD", binary],
                   outputs=[o for o in outs if o.is_file() and o.stat().st_size < 3e9],
                   params={"region": a.region, "tag": a.tag, "damp": a.damp, "smooth": a.smooth, "thres": a.thres,
                           "check": a.check, "sets": sets, "binary": binary},
                   counts={"exit": rc, "wall_min": round(wall / 60, 2)})
    sys.exit(rc)


if __name__ == "__main__":
    main()

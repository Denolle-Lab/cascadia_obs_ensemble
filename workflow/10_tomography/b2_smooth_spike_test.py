"""Track B, B2: spike test of tomoDD smoothing against uneven node spacing (Mendocino-Gorda grid).

tomoDD-SE builds smoothing rows as weight * (m_i - m_j) between neighbouring nodes, without dividing by the
node spacing (dx, dy, dz are computed and unused), so the effective smoothing changes with the spacing of
`node_z`. This test measures it: one run per target depth, each with +A% Vp spikes (Vp/Vs unchanged, so Vs
also +A%) on a lateral lattice of nodes; noise-free synthetics with tomoDD-SE_syn on the real event/station
geometry, the real pick list, pairs and weights; inversion with the production settings of params.yaml
`tomodd.inp` from the background model and the true locations (b2_tomodd_run.py). Analysis:
b2_smooth_spike_analyse.py.

  python b2_smooth_spike_test.py prep            # build <runs>/<region>/smooth_test/z<depth>/{syn,inputs}
  python b2_smooth_spike_test.py syn  ZTAG       # run the forward code for one depth and write inputs
Inversions: python b2_tomodd_run.py <region>/smooth_test/<ZTAG> inv
Run from the repo root with LD_LIBRARY_PATH=.pixi/envs/default/lib.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import load_params, write_manifest  # noqa: E402

P = load_params()
T = P["tomodd"]
S = T["smooth_test"]
REG = S["region"]
BASE = Path(T["runs"]) / REG
INP = BASE / "inputs"
TEST = BASE / "smooth_test"

SYN_INP = """* tomoDD-SE_syn.inp written by b2_smooth_spike_test.py
*catalog diff times:

*
Input_Files/dt.ct
*
Input_Files/event.dat
*
Input_Files/station.dat
*
tomoDD.loc
*
tomoDD.reloc
*
tomoDD.sta
*
tomoDD.res
*

*
tomoDD.vel
*
Vp_model.dat
*
Vs_model.dat
*
Input_Files/absolute.dat
*
* IDAT   IPHA   DIST
   2      3     8000
* OBSCC  OBSCT   Air_dep
    0      0     {air:.3f}
*  ISTART  ISOLV  NSET
     0       2     1
* iuses iuseq invdel  stepl
    2     0     0       {stepl}
* wlat   wlon   rota
  {clat}  {clon}   0
* weight1 weight2  weight3  CC_format
    1    1    1       1
* NITER WTCCP WTCCS WRCC WDCC WTCTP WTCTS WRCT WDCT WTCD DAMP JOINT THRES
  1  -9  -9  -9  -9  1  1  -9  -9  1  100  1  0.2
* CID
    0
* ID
"""


def read_mod(path):
    lines = Path(path).read_text().split("\n")
    bld, nx, ny, nz = lines[0].split()
    nx, ny, nz = int(nx), int(ny), int(nz)
    x = np.array(lines[1].split(), float)
    y = np.array(lines[2].split(), float)
    z = np.array(lines[3].split(), float)
    rows = [np.array(l.split(), float) for l in lines[4:4 + 2 * ny * nz]]
    vp = np.array(rows[:ny * nz]).reshape(nz, ny, nx)
    vpvs = np.array(rows[ny * nz:]).reshape(nz, ny, nx)
    return dict(head=lines[0], x=x, y=y, z=z, vp=vp, vpvs=vpvs, xyz_lines=lines[1:4])


def write_mod(path, m, vp):
    nz, ny, nx = vp.shape
    out = [m["head"], *m["xyz_lines"]]
    out += [" ".join(f"{v:.3f}" for v in vp[k, j]) for k in range(nz) for j in range(ny)]
    out += [" ".join(f"{v:.3f}" for v in m["vpvs"][k, j]) for k in range(nz) for j in range(ny)]
    Path(path).write_text("\n".join(out) + "\n")


def spikes():
    """(ztag, k, [(j, i), ...]) for each target depth."""
    m = read_mod(INP / "MOD")
    out = []
    for zd in S["depths_km"]:
        k = int(np.argmin(np.abs(m["z"] - zd)))
        assert abs(m["z"][k] - zd) < 1e-6, f"{zd} km is not a node"
        ii = [int(np.argmin(np.abs(m["x"] - v))) for v in S["lon"]]
        jj = [int(np.argmin(np.abs(m["y"] - v))) for v in S["lat"]]
        out.append((f"z{zd:+04.0f}".replace("+", "p").replace("-", "m"), k, [(j, i) for j in jj for i in ii]))
    return m, out


def prep():
    m, sp = spikes()
    summ = json.loads((INP / "summary.json").read_text())
    for ztag, k, nodes in sp:
        d = TEST / ztag
        (d / "syn" / "Input_Files").mkdir(parents=True, exist_ok=True)
        (d / "inputs").mkdir(parents=True, exist_ok=True)
        vp = m["vp"].copy()
        for j, i in nodes:
            vp[k, j, i] *= 1 + S["amp"]
        write_mod(d / "syn" / "MOD", m, vp)
        np.save(d / "true_dvp.npy", vp / m["vp"] - 1)
        for f in ("event.dat", "station.dat", "absolute.dat", "dt.ct"):
            dst = d / "syn" / "Input_Files" / f
            if not dst.exists():
                dst.symlink_to(INP / f)
        bindir = Path(T["bin"]).parent
        for f in ("ak135.15.SKS", "layer-16.dat"):
            shutil.copy(bindir / f, d / "syn" / f)
        (d / "syn" / "tomoDD-SE_syn.inp").write_text(SYN_INP.format(
            air=summ["air_dep_km"], stepl=T["inp"]["stepl_km"], clat=summ["center_lat"], clon=summ["center_lon"]))
        # inversion inputs: background model, real geometry; synthetic times are added by `syn`
        shutil.copy(INP / "MOD", d / "inputs" / "MOD")
        shutil.copy(INP / "summary.json", d / "inputs" / "summary.json")
        for f in ("event.dat", "station.dat"):
            dst = d / "inputs" / f
            if not dst.exists():
                dst.symlink_to(INP / f)
        print(ztag, f"z={m['z'][k]:g} km (k={k})", len(nodes), "spikes")
    write_manifest("b2_smooth_spike_prep", TEST, inputs=[INP / "MOD", INP / "summary.json"],
                   outputs=[TEST / t / "syn" / "MOD" for t, _, _ in sp], params=S,
                   counts={"runs": len(sp), "spikes_per_run": len(sp[0][2])})


def read_weights_abs(path):
    w = {}
    ev = None
    for line in open(path):
        if line.startswith("#"):
            ev = int(line.split()[1])
            continue
        sta, t, wt, ph = line.split()
        w[(ev, sta, ph)] = wt
    return w


def syn(ztag):
    d = TEST / ztag
    sd = d / "syn"
    with open(sd / "stdout.txt", "w") as so:
        rc = subprocess.run([T["bin_syn"], "tomoDD-SE_syn.inp"], cwd=sd, stdout=so, stderr=subprocess.STDOUT).returncode
    if rc:
        raise SystemExit(f"{ztag}: tomoDD-SE_syn exit {rc}")
    # absolute: keep real weights, synthetic times (absolute.syn: ev sta t wt phase)
    wabs = read_weights_abs(INP / "absolute.dat")
    n_abs = miss = 0
    with open(d / "inputs" / "absolute.dat", "w") as f:
        last = None
        for line in open(sd / "absolute.syn"):
            ev, sta, t, _, ph = line.split()
            ev = int(ev)
            wt = wabs.get((ev, sta, ph))
            if wt is None:
                miss += 1
                continue
            if ev != last:
                f.write(f"#  {ev}\n")
                last = ev
            f.write(f"{sta} {float(t):11.4f} {wt} {ph}\n")
            n_abs += 1
    # differential: real pair list and weights from dt.ct, synthetic times (dt.syn: ev1 ev2 sta t1 t2 wt phase)
    wdt = {}
    pair = None
    for line in open(INP / "dt.ct"):
        if line.startswith("#"):
            pair = tuple(int(v) for v in line.split()[1:3])
            continue
        sta, _, _, wt, ph = line.split()
        wdt[(*pair, sta, ph)] = wt
    n_dt = miss_dt = 0
    with open(d / "inputs" / "dt.ct", "w") as f:
        last = None
        for line in open(sd / "dt.syn"):
            e1, e2, sta, t1, t2, _, ph = line.split()
            key = (int(e1), int(e2), sta, ph)
            wt = wdt.get(key)
            if wt is None:
                miss_dt += 1
                continue
            if key[:2] != last:
                f.write(f"#  {key[0]}  {key[1]}\n")
                last = key[:2]
            f.write(f"{sta} {float(t1):11.4f} {float(t2):11.4f} {wt} {ph}\n")
            n_dt += 1
    counts = {"abs": n_abs, "abs_unmatched": miss, "dt": n_dt, "dt_unmatched": miss_dt,
              "abs_real": len(wabs), "dt_real": len(wdt)}
    print(ztag, counts)
    write_manifest(f"b2_smooth_spike_syn_{ztag}", d, inputs=[sd / "MOD", T["bin_syn"], INP / "absolute.dat", INP / "dt.ct"],
                   outputs=[d / "inputs" / "absolute.dat", d / "inputs" / "dt.ct", sd / "absolute.syn", sd / "dt.syn"],
                   params={"ztag": ztag, **S}, counts=counts)


if __name__ == "__main__":
    if sys.argv[1] == "prep":
        prep()
    elif sys.argv[1] == "syn":
        syn(sys.argv[2])

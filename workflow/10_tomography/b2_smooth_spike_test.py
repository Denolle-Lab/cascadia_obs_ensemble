"""Track B, B2: spike test of tomoDD smoothing against uneven node spacing (Mendocino-Gorda grid).

tomoDD-SE builds smoothing rows as weight * (m_i - m_j) between neighbouring nodes, without dividing by the
node spacing (dx, dy, dz are computed and unused), so the effective smoothing changes with the spacing of
`node_z`. This test measures it: one run per target depth, each with +A% Vp spikes (Vp/Vs unchanged, so Vs
also +A%) on a lateral lattice of nodes; noise-free synthetics with tomoDD-SE_syn on the real event/station
geometry, the real pick list, pairs and weights; inversion with the production settings of params.yaml
`tomodd.inp` from the background model and the true locations (b2_tomodd_run.py). Analysis:
b2_smooth_spike_analyse.py.

  python b2_smooth_spike_test.py [--cfg CFG] prep        # build the per-depth {syn,inputs} dirs (+ null)
  python b2_smooth_spike_test.py [--cfg CFG] syn  ZTAG   # run the forward code for one depth and write inputs
CFG = smooth_test (default; params tomodd.smooth_test, first test: <runs>/<region>/smooth_test/<ZTAG>, inversion
tag `inv`) or res_tests (params tomodd.res_tests: region, damping and smoothing candidates of the resolution tests;
dirs <runs>/<region>/res_tests/spike_<ZTAG>, inversion tags inv_d<damp>_s<w1>, notes/resolution_tests.md).
Inversions: python b2_tomodd_run.py <region>/smooth_test/<ZTAG> inv                       (smooth_test)
            python b2_tomodd_run.py <region>/res_tests/spike_<ZTAG> inv_d1000_s20 --damp 1000 --smooth 20 20 10
Run from the repo root with LD_LIBRARY_PATH=.pixi/envs/default/lib.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import load_params, tomodd_bin, write_manifest  # noqa: E402

P = load_params()
T = P["tomodd"]


def res_cfg():
    """tomodd.res_tests for the selected tile: env TOMO_RES_REGION (default res_tests.region), with
    res_tests.regions.<tile> over the defaults (spike and checker merged key by key; other keys replaced)."""
    base = {k: v for k, v in T["res_tests"].items() if k != "regions"}
    region = os.environ.get("TOMO_RES_REGION") or base["region"]
    for k, v in (T["res_tests"].get("regions") or {}).get(region, {}).items():
        base[k] = {**base[k], **v} if k in ("spike", "checker") else v
    base["region"] = region
    return base


def config(name="smooth_test"):
    """Spike-test configuration. smooth_test: first test (unchanged behaviour); res_tests: resolution tests."""
    if name == "smooth_test":
        s = dict(T["smooth_test"])
        test = Path(T["runs"]) / s["region"] / "smooth_test"
        return dict(name=name, S=s, region=s["region"], test=test, prefix="", exact_mod=False,
                    inv_tags=["inv"], damp=T["inp"]["damp"], smooth=[T["inp"]["smooth"]])
    if name == "res_tests":
        r = res_cfg()
        s = dict(r["spike"], region=r["region"])
        test = Path(T["runs"]) / r["region"] / r["subdir"]
        return dict(name=name, S=s, region=r["region"], test=test, prefix="spike_", exact_mod=True,
                    inv_tags=[inv_tag(r["damp"], w) for w in r["smooth"]], damp=r["damp"], smooth=r["smooth"])
    raise SystemExit(f"unknown cfg {name}")


def inv_tag(damp, smooth):
    return f"inv_d{damp:g}_s{smooth[0]:g}"


def use(name):
    """Set the module globals (S, REG, BASE, INP, TEST, CFG) for one configuration."""
    global S, REG, BASE, INP, TEST, CFG
    CFG = config(name)
    S, REG = CFG["S"], CFG["region"]
    BASE = Path(T["runs"]) / REG
    INP = BASE / "inputs"
    TEST = CFG["test"]


def zdir(ztag):
    return TEST / f"{CFG['prefix']}{ztag}"


use("smooth_test")

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
    return dict(head=lines[0], x=x, y=y, z=z, vp=vp, vpvs=vpvs, xyz_lines=lines[1:4],
                vpvs_lines=lines[4 + ny * nz:4 + 2 * ny * nz])


def write_mod(path, m, vp, exact=False):
    """exact=False: first-test format (Vp/Vs rounded to 3 decimals, as in smooth_test); exact=True: Vp/Vs lines
    copied verbatim from the source MOD, so only Vp (3 decimals, as in the source) differs from the background."""
    nz, ny, nx = vp.shape
    out = [m["head"], *m["xyz_lines"]]
    out += [" ".join(f"{v:.3f}" for v in vp[k, j]) for k in range(nz) for j in range(ny)]
    if exact:
        out += m["vpvs_lines"]
    else:
        out += [" ".join(f"{v:.3f}" for v in m["vpvs"][k, j]) for k in range(nz) for j in range(ny)]
    Path(path).write_text("\n".join(out) + "\n")


def geo_to_mod(lat, lon):
    """Geographic -> MOD coordinates (D-B028 P-2 rescaling if the region inputs have forward.json)."""
    f = INP / "forward.json"
    if not f.exists():
        return lat, lon
    from b2_tomodd_backtransform import forward
    return forward(lat, lon, json.loads(f.read_text()))


def spikes():
    """(ztag, k, [(j, i), ...]) for each target depth."""
    m = read_mod(INP / "MOD")
    out = []
    for zd in S["depths_km"]:
        k = int(np.argmin(np.abs(m["z"] - zd)))
        assert abs(m["z"][k] - zd) < 1e-6, f"{zd} km is not a node"
        lat_m, lon_m = geo_to_mod(np.array(S["lat"], float), np.array(S["lon"], float))
        ii = [int(np.argmin(np.abs(m["x"] - v))) for v in lon_m]
        jj = [int(np.argmin(np.abs(m["y"] - v))) for v in lat_m]
        out.append((f"z{zd:+04.0f}".replace("+", "p").replace("-", "m"), k, [(j, i) for j in jj for i in ii]))
    return m, out


def prep_syn_dir(d, mod_writer, inp=None):
    """Forward-run directory d/syn (MOD written by mod_writer(path), real geometry) and d/inputs (background MOD)."""
    inp = INP if inp is None else inp
    summ = json.loads((inp / "summary.json").read_text())
    (d / "syn" / "Input_Files").mkdir(parents=True, exist_ok=True)
    (d / "inputs").mkdir(parents=True, exist_ok=True)
    mod_writer(d / "syn" / "MOD")
    for f in ("event.dat", "station.dat", "absolute.dat", "dt.ct"):
        dst = d / "syn" / "Input_Files" / f
        if not dst.exists():
            dst.symlink_to(inp / f)
    bindir = Path(T["bin"]).parent
    for f in ("ak135.15.SKS", "layer-16.dat"):
        shutil.copy(bindir / f, d / "syn" / f)
    (d / "syn" / "tomoDD-SE_syn.inp").write_text(SYN_INP.format(
        air=summ["air_dep_km"], stepl=T["inp"]["stepl_km"], clat=summ["center_lat"], clon=summ["center_lon"]))
    # inversion inputs: background model, real geometry; synthetic times are added by `syn`
    shutil.copy(inp / "MOD", d / "inputs" / "MOD")
    for f in ("summary.json", "forward.json"):
        if (inp / f).exists():
            shutil.copy(inp / f, d / "inputs" / f)
    for f in ("event.dat", "station.dat"):
        dst = d / "inputs" / f
        if not dst.exists():
            dst.symlink_to(inp / f)


def prep():
    m, sp = spikes()
    ex = CFG["exact_mod"]
    for ztag, k, nodes in sp + [("null", None, [])]:
        d = zdir(ztag)
        vp = m["vp"].copy()
        for j, i in nodes:
            vp[k, j, i] *= 1 + S["amp"]
        if ex:
            vp = np.round(vp, 3)          # the true model is what the MOD file holds
        if ztag == "null" and (d / "syn" / "MOD").exists():
            continue                      # first test: null was set up by hand with the background MOD
        prep_syn_dir(d, lambda path: write_mod(path, m, vp, exact=ex))
        np.save(d / "true_dvp.npy", vp / m["vp"] - 1)
        print(ztag, "" if k is None else f"z={m['z'][k]:g} km (k={k})", len(nodes), "spikes")
    write_manifest(f"b2_smooth_spike_prep{'' if CFG['name'] == 'smooth_test' else '_' + CFG['name']}", TEST,
                   inputs=[INP / "MOD", INP / "summary.json"],
                   outputs=[zdir(t) / "syn" / "MOD" for t, _, _ in sp], params=S,
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


def run_syn(sd, label):
    with open(sd / "stdout.txt", "w") as so:
        rc = subprocess.run([tomodd_bin("bin_syn", sd), "tomoDD-SE_syn.inp"], cwd=sd, stdout=so, stderr=subprocess.STDOUT).returncode
    if rc:
        raise SystemExit(f"{label}: tomoDD-SE_syn exit {rc}")


def syn(ztag):
    d = zdir(ztag)
    sd = d / "syn"
    run_syn(sd, ztag)
    counts = syn_to_inputs(sd, d / "inputs")
    print(ztag, counts)
    write_manifest(f"b2_smooth_spike_syn_{ztag}", d, inputs=[sd / "MOD", tomodd_bin("bin_syn", sd), INP / "absolute.dat", INP / "dt.ct"],
                   outputs=[d / "inputs" / "absolute.dat", d / "inputs" / "dt.ct", sd / "absolute.syn", sd / "dt.syn"],
                   params={"ztag": ztag, "cfg": CFG["name"], **S}, counts=counts)


def syn_to_inputs(sd, out, inp=None, noise=None):
    """Synthetic times of forward dir sd -> out/absolute.dat, out/dt.ct with the real pick list, pairs and weights.
    noise: optional dict (event, station code, phase) -> added time error (s); the same draw enters the absolute
    pick and every differential time built from it, so absolute and differential data stay consistent."""
    inp = INP if inp is None else inp
    nz = (lambda key: 0.0) if noise is None else (lambda key: noise[key])
    # absolute: keep real weights, synthetic times (absolute.syn: ev sta t wt phase)
    wabs = read_weights_abs(inp / "absolute.dat")
    n_abs = miss = 0
    with open(out / "absolute.dat", "w") as f:
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
            f.write(f"{sta} {float(t) + nz((ev, sta, ph)):11.4f} {wt} {ph}\n")
            n_abs += 1
    # differential: real pair list and weights from dt.ct, synthetic times (dt.syn: ev1 ev2 sta t1 t2 wt phase)
    wdt = {}
    pair = None
    for line in open(inp / "dt.ct"):
        if line.startswith("#"):
            pair = tuple(int(v) for v in line.split()[1:3])
            continue
        sta, _, _, wt, ph = line.split()
        wdt[(*pair, sta, ph)] = wt
    n_dt = miss_dt = 0
    with open(out / "dt.ct", "w") as f:
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
            t1 = float(t1) + nz((key[0], sta, ph))
            t2 = float(t2) + nz((key[1], sta, ph))
            f.write(f"{sta} {t1:11.4f} {t2:11.4f} {wt} {ph}\n")
            n_dt += 1
    return {"abs": n_abs, "abs_unmatched": miss, "dt": n_dt, "dt_unmatched": miss_dt,
            "abs_real": len(wabs), "dt_real": len(wdt)}


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[0] == "--cfg":
        use(args[1])
        args = args[2:]
    if args[0] == "prep":
        prep()
    elif args[0] == "syn":
        syn(args[1])

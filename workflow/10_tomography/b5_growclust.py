"""D-B048: relocate one case with GrowClust3D.jl in model v0.3.0 (the B4 NonLinLoc time grids, EFT frame).

Inputs (<out>/<case>/growclust/<variant>/in/), from <out>/<case>/data:
* evlist (format 1): start hypocentres with depth z' = EFT depth (the grids' depth axis); magnitude v4 (0 if none).
* stlist (format 2): stations, elevation = -zsrc (not used with nllgrid tables).
* xcordata: the case's dt.csv; tdif = tt1 - tt2 (tdif_fmt 12); rxcor = pair weight (0-1). GrowClust3D's misfit is
  an unweighted L1 norm: rxcor only orders the cluster merges (mean of the best nbeststa values) and filters
  (rmin = rmincut = 0 here, so every dt is used).
* control.inp: nllgrid tables = b5_test.time_grids (cascadia3d.<P|S>.<STA>.time), LAMBERT = lcc WGS84 lon0 -125.8
  lat0 45 parallels 41/49 (checked against every grid header by GrowClust3D), table box = event box + box_km.
* aux.txt: the algorithm constants of the variant (b5_test.growclust.variants), read by b5_growclust_run.jl.
Run: julia -t threads b5_growclust_run.jl control.inp aux.txt (JULIA_DEPOT_PATH = b5_test.growclust.depot).
Output: reloc.csv (ID, lat, lon, depth = true depth (EFT undone), boot_mad_h, boot_mad_z, boot_std_h, boot_std_z km,
cid, nb, relocated = in a cluster of >= nbranch_min) and boot.npz (bootstrap positions, true depth).

  python b5_growclust.py CASE VARIANT [--nboot N]
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd

from b5_common import B5, GRIDS, HERE, PREFIX, case_dir, eft_depth, lambert_km, load_case, unflatten, write_manifest

G = B5["growclust"]

CTL = """* control file written by b5_growclust.py ({case} / {variant})
* evlist_fmt
1
* fin_evlist
{d}/evlist.txt
* stlist_fmt
2
* fin_stlist
{d}/stlist.txt
* xcordat_fmt tdif_fmt (12 = tt1 - tt2)
1  12
* fin_xcordat
{d}/xcordata.txt
* ttabsrc
nllgrid
* fin_vzmdl
{prefix}
* fdir_ttab
{grids}/
* projection (proj, ellps, lon0, lat0, rotANG, latP1, latP2)
lcc WGS84 -125.8 45.0 0.0 41.0 49.0
* vpvs_factor (S grids exist for every station: not used)
1.76
* tt_zmin tt_zmax
{zmin} {zmax}
* tt_xmin tt_xmax tt_ymin tt_ymax
{x0:.1f} {x1:.1f} {y0:.1f} {y1:.1f}
* rmin delmax rmsmax
{rmin} {delmax} {rmsmax}
* rpsavgmin rmincut ngoodmin iponly
{rpsavgmin} {rmincut} {ngoodmin} {iponly}
* nboot nbranch_min
{nboot} {nbranch_min}
* fout_cat
{o}/out.cat
* fout_clust
{o}/out.clust
* fout_log
{o}/out.log
* fout_boot
{o}/out.boot
"""


def write_inputs(case, variant, run, nboot):
    ev, pk, st, dt = load_case(case)
    d = run / "in"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "evlist.txt", "w") as f:
        for r in ev.itertuples():
            t = pd.Timestamp(r.time)
            mag = 0.0 if pd.isna(r.mag) else r.mag
            f.write(f"{t.year} {t.month} {t.day} {t.hour} {t.minute} {t.second + t.microsecond / 1e6:.6f} "
                    f"{r.lat:.6f} {r.lon:.6f} {r.zf_km:.4f} {mag:.2f} 0.0 0.0 0.0 {r.ID}\n")
    with open(d / "stlist.txt", "w") as f:
        for r in st.itertuples():
            f.write(f"{r.station} {r.lat:.6f} {r.lon:.6f} {-r.zsrc_km * 1000.0:.1f}\n")
    lines = []
    for (a, b), g in dt.groupby(["id1", "id2"], sort=True):
        lines.append(f"# {a} {b} 0.0")
        lines += [f"{s} {t1 - t2:.5f} {w:.4f} {ph}" for s, t1, t2, w, ph in zip(g.station, g.tt1, g.tt2, g.w, g.phase)]
    (d / "xcordata.txt").write_text("\n".join(lines) + "\n")
    V, I = G["variants"][variant], G["inp"]
    x, y = lambert_km(ev.lon.values, ev.lat.values)
    h = I["box_km"] / 2
    o = run / "out"
    o.mkdir(exist_ok=True)
    (run / "control.inp").write_text(CTL.format(
        case=case, variant=variant, d=d, prefix=PREFIX, grids=GRIDS, zmin=I["tt_zmin"], zmax=I["tt_zmax"],
        x0=x.min() - h, x1=x.max() + h, y0=y.min() - h, y1=y.max() + h, rmin=I["rmin"], delmax=I["delmax"],
        rmsmax=V["rmsmax"], rpsavgmin=I["rpsavgmin"], rmincut=I["rmincut"], ngoodmin=I["ngoodmin"],
        iponly=I["iponly"], nboot=nboot, nbranch_min=I["nbranch_min"], o=o))
    (run / "aux.txt").write_text("\n".join(f"{k} {v}" for k, v in V.items()) + "\n")
    return ev


def read_cat(path):
    cols = ["yr", "mo", "dy", "hr", "mi", "sc", "ID", "lat", "lon", "zf", "mag", "enum", "cid", "nb", "npair", "ndiffp",
            "ndiffs", "rmsp", "rmss", "mad_h", "mad_z", "mad_t", "qlat", "qlon", "qzf"]
    return pd.read_csv(path, sep=r"\s+", header=None, names=cols)


def read_boot(path):
    with open(path) as f:
        nq, nb = map(int, f.readline().split())
        rows, boot = [], np.zeros((nq, nb, 3))
        for i in range(nq):
            t = f.readline().split()
            rows.append((int(t[1]), float(t[16]), float(t[18])))   # ID, std_h, std_z (madH madZ madT stdH stdT stdZ)
            for k in range(nb):
                b = f.readline().split()
                boot[i, k] = float(b[0]), float(b[1]), float(b[2])
    return pd.DataFrame(rows, columns=["ID", "std_h", "std_zf"]), boot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("variant")
    ap.add_argument("--nboot", type=int, default=G["nboot"])
    a = ap.parse_args()
    run = case_dir(a.case) / "growclust" / a.variant
    if (run / "reloc.csv").exists():
        print(f"{run}: done")
        return
    run.mkdir(parents=True, exist_ok=True)
    ev = write_inputs(a.case, a.variant, run, a.nboot)
    env = dict(os.environ, JULIA_DEPOT_PATH=G["depot"], JULIA_PROJECT=G["project"])
    t0 = time.time()
    with open(run / "stdout.txt", "w") as so:
        rc = subprocess.run([G["julia"], "-t", str(G["threads"]), str(HERE / "b5_growclust_run.jl"),
                             str(run / "control.inp"), str(run / "aux.txt")], cwd=run, env=env, stdout=so,
                            stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0
    if rc != 0 or not (run / "out" / "out.cat").exists():
        raise SystemExit(f"{run}: GrowClust3D failed (exit {rc}); see stdout.txt")
    c = read_cat(run / "out" / "out.cat")
    r = pd.DataFrame({"ID": c.ID, "lat": c.lat, "lon": c.lon, "depth": unflatten(c.zf), "cid": c.cid, "nb": c.nb,
                      "npair": c.npair, "rmsp": c.rmsp, "rmss": c.rmss, "boot_mad_h": c.mad_h,
                      "boot_mad_z": c.mad_z})
    if a.nboot > 1:
        sd, boot = read_boot(run / "out" / "out.boot")
        boot[:, :, 2] = unflatten(boot[:, :, 2])
        r = r.merge(sd.rename(columns={"std_h": "boot_std_h", "std_zf": "boot_std_z"}), on="ID", how="left")
        np.savez_compressed(run / "boot.npz", ID=sd.ID.values, lat=boot[:, :, 0], lon=boot[:, :, 1],
                            depth=boot[:, :, 2])
    r["relocated"] = r.nb >= G["inp"]["nbranch_min"]
    r = ev[["ID"]].merge(r, on="ID", how="left")
    r.to_csv(run / "reloc.csv", index=False, float_format="%.6f")
    counts = dict(exit=rc, wall_s=round(wall, 1), events=len(r), relocated=int(r.relocated.sum()),
                  clusters=int(r[r.relocated].cid.nunique()), largest_cluster=int(r.nb.max()), nboot=a.nboot)
    (run / "run.json").write_text(json.dumps(counts, indent=1))
    write_manifest(f"b5_growclust_{a.case}_{a.variant}", run,
                   inputs=[run / "in" / "evlist.txt", run / "in" / "xcordata.txt", run / "control.inp",
                           run / "aux.txt", HERE / "b5_growclust_run.jl"],
                   outputs=[run / "reloc.csv", run / "out" / "out.cat", run / "out" / "out.log"],
                   params={"variant": a.variant, "aux": G["variants"][a.variant], "inp": G["inp"], "nboot": a.nboot,
                           "growclust3d": "0.1.0 git 592f6179", "julia": "1.13.4"}, counts=counts)
    print(f"{a.case}/growclust/{a.variant}: {counts}")


if __name__ == "__main__":
    main()

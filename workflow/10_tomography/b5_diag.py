"""B5 diagnostics for events the relocation pulls above the surface (D-B055: find the cause, never move events).

Reruns one cluster of a B5 run with tomoDD settings changed, in <run>/diag/<tag>/<cid>/ (the run's own results are
not touched), and compares with the run's solution (same centroid restoration as collect):
* --air-dep KM: Air_dep in km (depth, negative = above sea level). The run uses -(highest station elevation +
  0.05 km), so tomoDD removes ("air quakes") every event whose update goes above the highest station; a value far
  above (e.g. -10) keeps them in and shows where the double-difference solution puts them.
* --damp D: LSQR damping (the run: b5.tomodd.damp = 100); hypoDD's guide is a condition number of 40-80.
Output: <run>/diag/<tag>/<cid>/ (run files, reloc.csv, diag.json) and the summary line in
data/tomo_v1/b5/<run>/diag.md (appended), manifest.

  python b5_diag.py --version V --cid CID --tag TAG [--run RUN] [--air-dep KM] [--damp D]
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

import b5_relocate as R
from b5_relocate import B, KM_DEG, _W, config, load_shared, read_reloc, run_tomodd, write_inputs, write_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--run")
    ap.add_argument("--cid", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--air-dep", type=float)
    ap.add_argument("--damp", type=float)
    a = ap.parse_args()
    ns = argparse.Namespace(version=a.version, run=a.run, b4_run=None, iter0=None, time=None)
    C = config(ns)
    load_shared(C)
    rdir = C["dir"] / "diag" / a.tag / a.cid
    if rdir.exists():
        shutil.rmtree(rdir)
    rdir.mkdir(parents=True)
    info, tcs = write_inputs(rdir, a.cid)
    if a.air_dep is not None:
        info["air_dep_km"] = a.air_dep
    res, it = run_tomodd(rdir, a.cid, info, f" diag {a.tag}", damp=a.damp)
    ev = _W["ev"].set_index("ID")
    g = _W["mem"].get_group(a.cid)
    core = set(ev.index[g.ev.values[g.core.values]])
    out = dict(cid=a.cid, tag=a.tag, air_dep_km=info["air_dep_km"], damp=a.damp or B["tomodd"]["damp"],
               **{k: res[k] for k in ("exit", "wall_s", "airquakes", "cnd_first", "cnd_last", "rmsct_first_ms",
                                      "rmsct_last_ms", "iterations")})
    rel = rdir / "Output_Files" / "tomoDD.reloc"
    if res["exit"] == 0 and rel.exists() and rel.stat().st_size:
        from b4_catalog import surface_km
        r = read_reloc(rel, info["center"], ev.index.values).set_index("ID")
        rc = r[r.index.isin(core)]
        b = ev.loc[rc.index]
        for c in ("lat", "lon", "depth"):
            r[c] -= (rc[c] - b[c]).mean()
        r.to_csv(rdir / "reloc.csv", float_format="%.8g")
        rc = r[r.index.isin(core)]
        b = ev.loc[rc.index]
        s = surface_km(rc.lat.values, rc.lon.values)
        above = rc.depth.values < -s
        hgt = -s - rc.depth.values
        base = C["dir"] / "clusters" / a.cid / "reloc.csv"
        out.update(relocated_core=len(rc), above_surface=int(above.sum()),
                   above_height_med_km=float(np.median(hgt[above])) if above.any() else None,
                   above_height_max_km=float(hgt[above].max()) if above.any() else None,
                   dz_med=float((rc.depth - b.depth).median()), dz_p10=float((rc.depth - b.depth).quantile(0.1)),
                   dz_p90=float((rc.depth - b.depth).quantile(0.9)),
                   dh_med=float(np.median(np.hypot((rc.lat - b.lat) * KM_DEG, (rc.lon - b.lon) * KM_DEG
                                                   * np.cos(np.radians(b.lat))))))
        if base.exists():                         # the run's own solution, same centroid rule
            r0 = pd.read_csv(base).set_index("ID")
            r0c = r0[r0.index.isin(core)]
            b0 = ev.loc[r0c.index]
            for c in ("lat", "lon", "depth"):
                r0[c] -= (r0c[c] - b0[c]).mean()
            com = rc.index.intersection(r0.index)
            dz = rc.depth.loc[com] - r0.depth.loc[com]
            out.update(run_relocated_core=len(r0c), common=len(com), dz_vs_run_med=float(dz.abs().median()))
            aq = set(json.loads((C["dir"] / "clusters" / a.cid / "run.json").read_text()).get("airquake_ids", []))
            ia = rc.index.intersection(list(aq))
            if len(ia):
                sa = surface_km(rc.lat.loc[ia].values, rc.lon.loc[ia].values)
                out.update(run_airquakes_relocated_here=len(ia),
                           run_airquakes_above_surface_here=int((rc.depth.loc[ia].values < -sa).sum()),
                           run_airquakes_height_med_km=float(np.median(-sa - rc.depth.loc[ia].values)))
    (rdir / "diag.json").write_text(json.dumps(out, indent=1, default=float))
    od = R.REPO / B["summary"] / C["run"]
    od.mkdir(parents=True, exist_ok=True)
    with open(od / "diag.md", "a") as f:
        f.write(f"- {time.strftime('%F %H:%M')} {json.dumps(out, default=float)}\n")
    write_manifest(f"b5_diag_{a.cid}_{a.tag}", rdir, inputs=[rdir / "MOD", rdir / "Input_Files" / "dt.ct", res["bin"]],
                   outputs=[rdir / "diag.json", rdir / "reloc.csv", od / "diag.md"], params=vars(a), counts=out)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()

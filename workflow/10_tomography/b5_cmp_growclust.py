"""D-B078 comparison, GrowClust3D leg: relocate each test set with GrowClust3D.jl on the waveform cross-correlation
differential times of the common frame (b5_cmp_common), in the NonLinLoc grids of catalog v5.1.

Per set s and run in b5_cmp_common.runs() (full, fold0..fold2), under out_dir("growclust", s, run):
* in/evlist.txt (format 1): every event of events(s), start = the v5.1 hypocentre, depth z' = Earth-flattened depth
  (the grids' depth axis, z' = Re ln(Re / (Re - z))), magnitude 0 (not used), origin time = v5.1.
* in/stlist.txt (format 2): the stations used, lat / lon from the grids' GTSRCE file (only used for delmax).
* in/xcordata.txt: cc_rows(s) minus the rows at held_out(s, run), minus rows at a station-phase without a v5.1 grid;
  "# ID1 ID2 0.0" (DT already uses the v5.1 origin times: origin-time correction 0), then "station dt_cc cc_max
  phase" (tdif_fmt 12: dt_cc = tt1 - tt2; rxcor = cc_max).
* control.inp, aux.txt: D-B048 (params.yaml b5_test.growclust, variant `default`) with the D-B078 changes in CHANGES.
* out/out.cat, out.clust, out.log (GrowClust3D), stdout.txt.
* reloc.csv: ID, lat, lon, depth (km bsl, true depth z = Re (1 - exp(-z'/Re))) of the events in a cluster of at least
  nbranch_min events; singletons (GrowClust3D leaves them at their start) are left out and listed in run.json.
* detail.csv: every event, with GrowClust3D's cluster id, cluster size, pair and dt counts, rms, bootstrap MAD.
* run.json: code and version, parameters, input rows / pairs, events relocated / in clusters / singletons, wall time.

Scores (--score, --score-only): <CMP>/growclust/scores.json (score_set per set) and scores_same_events.json (v5.1
scored on the events GrowClust3D relocated). --diag: diag.json (held-out misfit by shift, training fit, above-surface
events, events of the ngoodmin 8 run). --sens-ngoodmin N: the same runs with ngoodmin N under growclust/_sens_ngoodmin<N>/
(a sensitivity, not the leg's result). Notes: notes/b5_cmp_growclust.md.

  python b5_cmp_growclust.py [--sets axial ...] [--runs full fold0 ...] [--force] [--score | --score-only | --diag]
                             [--sens-ngoodmin N]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b5_cmp_common as C  # noqa: E402
from b2_fwd_common import RE, eft_depth, lambert_km  # noqa: E402
from manifest import load_params  # noqa: E402

LEG = "growclust"
GC = load_params()["b5_test"]["growclust"]       # D-B048: julia, depot, project, threads, nboot, inp, variants
VARIANT = "default"                               # D-B048 cross-correlation variant (run_growclust3D.jl example)
PREFIX = "cascadia3d"
VERSION = "GrowClust3D.jl 0.1.0 (git 592f6179, git-tree-sha1 2f35f37e), Julia 1.13.4"
# D-B078 changes to the D-B048 setup, each demanded by the CC data (see notes/b5_cmp_growclust.md)
CHANGES = {
    "ngoodmin": (8, 0, "D-B048 took 8 from its catalog-dt rule (8 common station-phases per pair). The CC pairs have "
                       "a median of 2 dt (p90 4-9): with 8, only 41/162 (jdf) to 578/1,436 (msh) events can be linked. "
                       "0 is the value of the package's own cross-correlation example (example.*.inp)."),
}
SENS = {}          # --sens-ngoodmin N: sensitivity run (not the leg's result), under <CMP>/growclust/_sens_ngoodmin<N>/


def params():
    inp = dict(GC["inp"])
    for k, (old, new, _) in CHANGES.items():
        assert inp[k] == old, (k, inp[k])
        inp[k] = new
    inp.update(SENS)
    return inp, dict(GC["variants"][VARIANT])


CTL = """* control file written by b5_cmp_growclust.py (D-B078, {s} / {run}; D-B048 variant {variant})
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
* vpvs_factor (S grids exist for every station used: not used)
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
NONE
"""


def grid_stations():
    """GTSRCE stations of the v5.1 grids (label, lat, lon), from the grids' manifest."""
    man = json.loads((Path(C.GRIDS) / "b4_time_grids_time_b4_v0.3.1m_sc21.manifest.json").read_text())
    src = [i["path"] for i in man["inputs"] if i["path"].endswith("stations_GTSRCE.in")][0]
    rows = [l.split() for l in Path(src).read_text().splitlines() if l.startswith("GTSRCE")]
    return pd.DataFrame({"station": [r[1] for r in rows], "lat": [float(r[3]) for r in rows],
                         "lon": [float(r[4]) for r in rows]}).drop_duplicates("station").set_index("station")


def has_grid(station, phase):
    return (Path(C.GRIDS) / f"{PREFIX}.{phase}.{station}.time.hdr").exists()


def data(s, run):
    """Events and the CC rows of one run (held-out stations and station-phases without a grid removed)."""
    ev = C.events(s)
    cc = C.cc_rows(s)
    n_all = len(cc)
    ho = C.held_out(s, run)
    cc = cc[~cc.station.isin(ho)]
    n_fold = len(cc)
    ok = {(st, ph): has_grid(st, ph) for st, ph in cc[["station", "phase"]].drop_duplicates().itertuples(index=False)}
    cc = cc[[ok[k] for k in zip(cc.station, cc.phase)]].sort_values(["ID1", "ID2", "station", "phase"])
    md5 = {str(f): hashlib.md5(f.read_bytes()).hexdigest() for f in (C.TESTS / f"events_{s}.csv",
                                                                       C.CC / f"cc_{s}.parquet")}
    info = dict(files_md5=md5, cc_rows_set=n_all, cc_rows_after_fold=n_fold, cc_rows_input=len(cc),
                rows_dropped_no_grid=n_fold - len(cc), held_out_stations=sorted(ho),
                station_phases_without_grid=sorted(f"{a}.{b}" for (a, b), v in ok.items() if not v))
    return ev, cc.reset_index(drop=True), info


def write_inputs(s, run, d, nboot):
    ev, cc, info = data(s, run)
    i = d / "in"
    i.mkdir(parents=True, exist_ok=True)
    zf = eft_depth(ev.depth.values)
    with open(i / "evlist.txt", "w") as f:
        for r, z in zip(ev.itertuples(), zf):
            t = pd.Timestamp(r.time)
            f.write(f"{t.year} {t.month} {t.day} {t.hour} {t.minute} {t.second + t.microsecond / 1e6:.6f} "
                    f"{r.lat:.7f} {r.lon:.7f} {z:.5f} 0.00 0.0 0.0 0.0 {r.ID}\n")
    gs = grid_stations()
    sta = sorted(cc.station.unique())
    miss = [x for x in sta if x not in gs.index]
    if miss:
        raise SystemExit(f"stations with grids but not in GTSRCE: {miss}")
    with open(i / "stlist.txt", "w") as f:
        for x in sta:
            f.write(f"{x} {gs.lat[x]:.6f} {gs.lon[x]:.6f} 0.0\n")
    lines = []
    for (a, b), g in cc.groupby(["ID1", "ID2"], sort=True):
        lines.append(f"# {a} {b} 0.0")
        lines += [f"{x} {t:.6f} {r:.4f} {p}" for x, t, r, p in zip(g.station, g.dt_cc, g.cc_max, g.phase)]
    (i / "xcordata.txt").write_text("\n".join(lines) + "\n")
    inp, aux = params()
    x, y = lambert_km(ev.lon.values, ev.lat.values)
    h = inp["box_km"] / 2
    o = d / "out"
    o.mkdir(exist_ok=True)
    (d / "control.inp").write_text(CTL.format(
        s=s, run=run, variant=VARIANT, d=i, prefix=PREFIX, grids=C.GRIDS, zmin=inp["tt_zmin"], zmax=inp["tt_zmax"],
        x0=x.min() - h, x1=x.max() + h, y0=y.min() - h, y1=y.max() + h, rmin=inp["rmin"], delmax=inp["delmax"],
        rmsmax=aux["rmsmax"], rpsavgmin=inp["rpsavgmin"], rmincut=inp["rmincut"], ngoodmin=inp["ngoodmin"],
        iponly=inp["iponly"], nboot=nboot, nbranch_min=inp["nbranch_min"], o=o))
    (d / "aux.txt").write_text("\n".join(f"{k} {v}" for k, v in aux.items()) + "\n")
    info.update(pairs_input=int(cc.groupby(["ID1", "ID2"]).ngroups), stations_input=len(sta),
                rows_input_by_phase={k: int(v) for k, v in cc.phase.value_counts().items()},
                tt_box_km=[round(x.min() - h, 1), round(x.max() + h, 1), round(y.min() - h, 1), round(y.max() + h, 1)])
    return ev, cc, info


def read_cat(path):
    cols = ["yr", "mo", "dy", "hr", "mi", "sc", "ID", "lat", "lon", "zf", "mag", "enum", "cid", "nb", "npair", "ndiffp",
            "ndiffs", "rmsp", "rmss", "mad_h", "mad_z", "mad_t", "qlat", "qlon", "qzf"]
    return pd.read_csv(path, sep=r"\s+", header=None, names=cols)


def unflatten(zf):
    return RE * (1 - np.exp(-np.asarray(zf, float) / RE))


def relocate(s, run, nboot, threads, force=False):
    d = C.out_dir(LEG, s, run)
    if (d / "reloc.csv").exists() and (d / "run.json").exists() and not force:
        print(f"{s}/{run}: done")
        return json.loads((d / "run.json").read_text())
    ev, cc, info = write_inputs(s, run, d, nboot)
    if not gc_used_ids(cc):          # no pair passes the pair filter: GrowClust3D stops with "0 unique stations"
        pd.DataFrame(columns=["ID", "lat", "lon", "depth"]).to_csv(d / "reloc.csv", index=False)
        rec = dict(leg=LEG, set=s, run=run, code=VERSION, sensitivity_override=dict(SENS) or None,
                   params=dict(inp=params()[0], aux=params()[1]), input=info, events=len(ev), relocated=0,
                   clusters=0, singletons=len(ev), unrelocated_ids=[int(v) for v in sorted(ev.ID)],
                   notes="no event pair passes the pair filter (ngoodmin); GrowClust3D not run")
        (d / "run.json").write_text(json.dumps(rec, indent=1))
        print(f"{s}/{run}: no usable pair, relocated 0/{len(ev)}")
        return rec
    env = dict(os.environ, JULIA_DEPOT_PATH=GC["depot"], JULIA_PROJECT=GC["project"])
    t0 = time.time()
    with open(d / "stdout.txt", "w") as so:
        rc = subprocess.run([GC["julia"], "-t", str(threads), str(HERE / "b5_growclust_run.jl"),
                             str(d / "control.inp"), str(d / "aux.txt")], cwd=d, env=env, stdout=so,
                            stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0
    if rc != 0 or not (d / "out" / "out.cat").exists():
        raise SystemExit(f"{d}: GrowClust3D failed (exit {rc}); see stdout.txt")
    c = read_cat(d / "out" / "out.cat")
    assert len(c) == len(ev) and set(c.ID) == set(ev.ID)
    inp, aux = params()
    c["depth"] = unflatten(c.zf)
    c["relocated"] = c.nb >= inp["nbranch_min"]
    linked = set(cc.ID1) | set(cc.ID2)
    used = gc_used_ids(cc)
    c["linked_input"] = c.ID.isin(linked)
    c["linked_used"] = c.ID.isin(used)
    r = c[c.relocated]
    r[["ID", "lat", "lon", "depth"]].to_csv(d / "reloc.csv", index=False, float_format="%.6f")
    c[["ID", "lat", "lon", "depth", "cid", "nb", "npair", "ndiffp", "ndiffs", "rmsp", "rmss", "mad_h", "mad_z",
       "mad_t", "relocated", "linked_input", "linked_used"]].to_csv(d / "detail.csv", index=False,
                                                                     float_format="%.6f")
    single = c[~c.relocated]
    cl = r.groupby("cid").size()
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    rec = dict(
        leg=LEG, set=s, run=run, code=VERSION, driver="b5_cmp_growclust.py + b5_growclust_run.jl (D-B048)",
        git_commit=git, decision="D-B078", variant=VARIANT, sensitivity_override=dict(SENS) or None,
        params=dict(inp=inp, aux=aux, nboot=nboot, threads=threads, tdifmax=30.0, nupdate=10000, iseed=0,
                    vzmodel_type=1, shallowmode="flat", ttabsrc="nllgrid", grids=str(C.GRIDS), prefix=PREFIX,
                    projection="lcc WGS84 lon0 -125.8 lat0 45 parallels 41/49 (checked by GrowClust3D on every grid)",
                    rxcor="cc_max", origin_time_correction=0.0,
                    changes_from_D_B048={k: dict(D_B048=a, D_B078=b, why=w) for k, (a, b, w) in CHANGES.items()}),
        input=info,
        events=len(ev), events_linked_input=len(linked & set(ev.ID)),
        events_linked_after_pair_filter=len(used & set(ev.ID)), growclust_summary=gc_summary(d),
        relocated=int(len(r)), clusters=int(len(cl)), largest_cluster=int(cl.max()) if len(cl) else 0,
        cluster_sizes_top10=[int(v) for v in cl.sort_values(ascending=False).values[:10]],
        singletons=int(len(single)), singletons_linked=int(single.linked_used.sum()),
        singletons_unlinked=int((~single.linked_used).sum()),
        unrelocated_ids=[int(v) for v in single.ID.sort_values()],
        wall_s=round(wall, 1), exit=rc,
        notes=("reloc.csv: events in a cluster of >= nbranch_min events (relocated); depth = true depth (EFT undone). "
               "Singletons (no usable CC pair, or every merge rejected by the rms / shift / distance limits) stay at "
               "the v5.1 start in GrowClust3D's output and are left out here. Each cluster keeps the centroid of its "
               "start positions (GrowClust3D design). No surface constraint (GrowClust3D only refuses a merge whose "
               "relocated cluster centroid is shallower than min(0, shallowest start z')). out.cat precision: "
               "1e-5 deg, 1 m depth."))
    (d / "run.json").write_text(json.dumps(rec, indent=1))
    print(f"{s}/{run}: relocated {rec['relocated']}/{rec['events']} in {rec['clusters']} clusters "
          f"(largest {rec['largest_cluster']}), singletons {rec['singletons']} ({rec['singletons_linked']} linked), "
          f"{rec['wall_s']} s")
    return rec


def gc_used_ids(cc):
    """IDs of the events in at least one pair that passes GrowClust3D's pair filter (read_xcordata_proj: rows with
    rxcor >= rmincut and pair-mean rxcor >= rpsavgmin; pair kept if ngood >= ngoodmin, good = rxcor >= rmin and
    station within delmax of the pair; delmax 400 km takes every station of these sets)."""
    inp, _ = params()
    x = cc[cc.cc_max >= inp["rmincut"]]
    x = x[x.groupby(["ID1", "ID2"]).cc_max.transform("mean") >= inp["rpsavgmin"]]
    n = (x.cc_max >= inp["rmin"]).groupby([x.ID1, x.ID2]).sum()
    p = n[n >= inp["ngoodmin"]].index
    return set(p.get_level_values(0)) | set(p.get_level_values(1))


def gc_summary(d):
    """GrowClust3D's run summary (out.log): label -> number."""
    out, on = {}, False
    for line in (d / "out" / "out.log").read_text().splitlines():
        if "GROWCLUST Run Summary" in line:
            on = True
            continue
        if on and ":" in line:
            k, v = line.rsplit(":", 1)
            try:
                out[" ".join(k.split())] = float(v) if "." in v else int(v)
            except ValueError:
                pass
    return out


def _stats(H):
    return {ph: dict(n=len(g), rms_ms=round(float(np.sqrt(np.mean(g.score ** 2))) * 1e3, 2),
                     mad_ms=round(float(np.median(np.abs(g.score - np.median(g.score)))) * 1e3, 2),
                     p90_ms=round(float(np.percentile(np.abs(g.score), 90)) * 1e3, 2)) for ph, g in H.groupby("phase")}


def score(sets):
    """score_set of each set -> <CMP>/<LEG>/scores.json. Also scores_same_events.json: the v5.1 reference scored on
    the events GrowClust3D relocated in each fold (score_set's v5.1 reference uses every event of the set), with the
    leg's own score next to it (same numbers as score_set)."""
    f, f2 = C.CMP / LEG / "scores.json", C.CMP / LEG / "scores_same_events.json"
    out = json.loads(f.read_text()) if f.exists() else {}
    same = json.loads(f2.read_text()) if f2.exists() else {}
    for s in sets:
        out[s] = C.score_set(LEG, s)
        print(s, json.dumps(out[s]))
        cc, ev, rows = C.cc_rows(s), C.events(s).set_index("ID"), {"v5.1_same_events": [], LEG: []}
        for run in C.runs()[1:]:
            r = C.read_reloc(LEG, s, run)
            if r is not None:
                rows["v5.1_same_events"].append(C.fold_score(cc, ev.loc[r.index], C.held_out(s, run)))
                rows[LEG].append(C.fold_score(cc, r, C.held_out(s, run)))
        same[s] = {tag: _stats(pd.concat(R)) for tag, R in rows.items() if R}
        print(s, "same events:", json.dumps(same[s]))
    f.write_text(json.dumps(out, indent=1))
    f2.write_text(json.dumps(same, indent=1))
    return out


def diag(sets):
    """Diagnostics for the note -> <CMP>/<LEG>/diag.json: held-out scores binned by the larger 3D shift (from v5.1)
    of the pair's two events (pooled over folds), and the v5.1 positions on the same rows; training-station fit (residual minus the pair's median) of v5.1
    and the leg; events above the surface in the full run (height above it, v5.1 depth below it)."""
    from b4_catalog import surface_km
    bins = [0, 0.5, 1, 2, 3, 5, 50]
    out = {}
    for s in sets:
        cc, ev = C.cc_rows(s), C.events(s).set_index("ID")
        H, T = [], {"v5.1_same_events": [], LEG: []}
        for run in C.runs()[1:]:
            r = C.read_reloc(LEG, s, run)
            ho = C.held_out(s, run)
            j = r.join(ev[["lat", "lon", "depth"]], rsuffix="0")
            sh = np.sqrt(((j.lat - j.lat0) * C.KM) ** 2 + (j.depth - j.depth0) ** 2
                         + ((j.lon - j.lon0) * C.KM * np.cos(np.radians(j.lat0))) ** 2)
            h = C.fold_score(cc, r, ho)
            h["shift"] = np.maximum(sh.loc[h.ID1].values, sh.loc[h.ID2].values)
            key = ["ID1", "ID2", "station", "phase"]
            h0 = C.fold_score(cc, ev.loc[r.index], ho)[key + ["score"]].rename(columns={"score": "score_v51"})
            H.append(h.merge(h0, on=key, how="left"))
            for tag, pos in (("v5.1_same_events", ev.loc[r.index]), (LEG, r)):
                x = C.residuals(cc, pos).dropna(subset=["res"])
                x = x[~x.station.isin(ho)].copy()
                x["score"] = x.res - x.groupby(["ID1", "ID2"]).res.transform("median")
                T[tag].append(x)
        H = pd.concat(H)
        H["bin"] = pd.cut(H["shift"], bins).astype(str)
        sens = "growclust/_sens_ngoodmin8"      # main leg and v5.1 on the events the ngoodmin 8 run relocated
        R8 = {run: C.read_reloc(sens, s, run) for run in C.runs()[1:]}
        if all(v is not None for v in R8.values()):
            Q = {"v5.1": [], LEG: [], sens: []}
            for run, r8 in R8.items():
                ids = r8.index.intersection(C.read_reloc(LEG, s, run).index)
                ho = C.held_out(s, run)
                Q["v5.1"].append(C.fold_score(cc, ev.loc[ids], ho))
                Q[LEG].append(C.fold_score(cc, C.read_reloc(LEG, s, run).loc[ids], ho))
                Q[sens].append(C.fold_score(cc, r8.loc[ids], ho))
            out.setdefault("_events_of_ngoodmin8_run", {})[s] = {k: _stats(pd.concat(v)) for k, v in Q.items()}
        H0 = H.dropna(subset=["score_v51"]).drop(columns="score").rename(columns={"score_v51": "score"})
        d = {"heldout_by_shift_km": {b: _stats(g) for b, g in H.groupby("bin", sort=False)},
             "heldout_by_shift_km_v51_same_rows": {b: _stats(g) for b, g in H0.groupby("bin", sort=False)},
             "training_fit": {k: _stats(pd.concat(v)) for k, v in T.items()}}
        f = C.read_reloc(LEG, s, "full").join(ev[["depth"]], rsuffix="0")
        sk = surface_km(f.lat.values, f.lon.values)
        up = f.depth.values < -sk
        d["above_surface_full"] = dict(
            n=int(up.sum()), v51_n=int((ev.depth.values < -surface_km(ev.lat.values, ev.lon.values)).sum()),
            height_med_km=round(float(np.median(-sk[up] - f.depth.values[up])), 3) if up.any() else None,
            height_max_km=round(float(np.max(-sk[up] - f.depth.values[up])), 3) if up.any() else None,
            v51_depth_below_surface_med_km=round(float(np.median(f.depth0.values[up] + sk[up])), 3) if up.any() else None)
        out[s] = d
        print(s, json.dumps(d))
    (C.CMP / LEG / "diag.json").write_text(json.dumps(out, indent=1))


def main():
    global LEG
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=list(C.SETS))
    ap.add_argument("--runs", nargs="+", default=C.runs())
    ap.add_argument("--nboot", type=int, default=GC["nboot"])
    ap.add_argument("--threads", type=int, default=min(16, GC["threads"]))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--score", action="store_true", help="score the sets after relocating (scores.json)")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--diag", action="store_true", help="diagnostics of existing runs (diag.json)")
    ap.add_argument("--sens-ngoodmin", type=int, help="sensitivity run with this ngoodmin (own folder, not the leg)")
    a = ap.parse_args()
    if a.sens_ngoodmin is not None:
        SENS["ngoodmin"] = a.sens_ngoodmin
        LEG = f"growclust/_sens_ngoodmin{a.sens_ngoodmin}"
    if a.score_only:
        score(a.sets)
        return
    if a.diag:
        diag(a.sets)
        return
    for s in a.sets:
        for run in a.runs:
            relocate(s, run, a.nboot, a.threads, a.force)
    if a.score:
        score(a.sets)


if __name__ == "__main__":
    main()

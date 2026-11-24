"""B5 weighting test (PI 2026-11-24): which S weight and re-weighting give the best relative relocation?

Reuses the production cluster inputs of run b5.<run> (MOD, dt.ct, event.dat, station.dat, frozen binary, Air_dep,
frame centre); changes only WTCTS, WRCT/WDCT, the dt rows (cross-validation) or the data and start (synthetic).
Variants = s_weight x reweight (params b5_wtest). Per cluster:
* base: one-iteration tomoDD run at the B4 start (initial residuals), gives the measured S/P ratio and the CV baseline.
* full: relocation with all dt rows, then a one-iteration evaluation at the final positions (events tomoDD did not
  relocate stay at their start): in-sample rms P/S, air quakes, events above the surface (earth_relief 01m, bilinear).
* fold k: relocation without the dt rows of the stations in fold k; evaluation with all rows; held-out rows scored
  after removing, per event pair, the median residual of that pair's training rows.
* synth: truth = B4 positions; times from the B4 grids + noise resampled per phase from the cluster's B4 dt
  residuals / sqrt(2) + origin-time error; start perturbed (params synth); error = centroid-removed position error.
Outputs <out>/<cluster>/<variant>/<kind>/result.json, <out>/summary/*.csv; resumable (a task with result.json is skipped).

  python b5_wtest.py setup | base | run | collect
"""
import json, os, re, shutil, subprocess, sys, zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import load_params, write_manifest  # noqa: E402
from b2_fwd_common import eft_depth, lambert_km  # noqa: E402
from b2_fwd_mods import ellipsoid_coords  # noqa: E402

PR = load_params()
P = PR["b5_wtest"]
RUN = Path(PR["b5"]["out"]) / P["run"]
OUT = Path(P["out"])
CL = RUN / "clusters"
RE_ = 6371.0


# ------------------------------------------------------------------ helpers
def read_dt(path):
    """dt.ct -> (header lines with their row ranges, rows DataFrame)."""
    pairs, rows = [], []
    for l in Path(path).read_text().splitlines():
        if l.startswith("#"):
            a, b = map(int, l.split()[1:3])
            pairs.append((a, b))
        elif l.strip():
            s, t1, t2, w, ph = l.split()
            rows.append((len(pairs) - 1, s, float(t1), float(t2), float(w), ph))
    df = pd.DataFrame(rows, columns=["pair", "sta", "t1", "t2", "w", "ph"])
    pr = np.array(pairs)
    df["e1"], df["e2"] = pr[df.pair, 0], pr[df.pair, 1]
    return df


def write_dt(path, df):
    out = []
    for (a, b), g in df.groupby(["e1", "e2"], sort=False):
        out.append(f"#  {a}  {b}")
        out += [f"{r.sta:<7s} {r.t1:10.4f} {r.t2:10.4f} {r.w:6.3f} {r.ph}" for r in g.itertuples()]
    Path(path).write_text("\n".join(out) + "\n")


def read_events(path):
    rows = [l.split() for l in Path(path).read_text().splitlines() if l.strip()]
    return pd.DataFrame({"date": [r[0] for r in rows], "time": [r[1] for r in rows],
                         "lat": [float(r[2]) for r in rows], "lon": [float(r[3]) for r in rows],
                         "depth": [float(r[4]) for r in rows], "rest": [" ".join(r[5:9]) for r in rows],
                         "ID": [int(r[9]) for r in rows]})


def write_events(path, ev):
    Path(path).write_text("".join(f"{r.date}  {r.time} {r.lat:10.6f} {r.lon:11.6f} {r.depth:8.3f} {r.rest} {r.ID:10d} 0\n"
                                  for r in ev.itertuples()))


def read_reloc(path):
    r = pd.read_csv(path, sep=r"\s+", header=None)
    return pd.DataFrame({"ID": r[0].astype(int), "lat": r[1], "lon": r[2], "depth": r[3]})


def read_res(path):
    r = pd.read_csv(path, sep=r"\s+", skiprows=1, header=None,
                    names=["sta", "DT", "e1", "e2", "IDX", "QUAL", "RES", "WT", "OFFS"])
    for c in ("e1", "e2", "IDX", "RES"):
        r[c] = pd.to_numeric(r[c], errors="coerce")      # Fortran writes '****' when a residual overflows its field
    bad = int(r.RES.isna().sum())
    if bad:
        Path(path).with_name("nonnumeric_rows.txt").write_text(f"{bad}\n")
    r = r[r.RES.notna() & (r.e1 != r.e2) & r.IDX.isin([3, 4])].copy()
    r["e1"], r["e2"] = r.e1.astype(np.int64), r.e2.astype(np.int64)
    r["ph"] = np.where(r.IDX == 3, "P", "S")
    r["res"] = r.RES / 1000.0
    return r[["sta", "e1", "e2", "ph", "res"]]


def inp_lines(src):
    return (src / "tomoDD-SE.inp").read_text().splitlines()


def write_inp(d, src, sets):
    L = inp_lines(src)
    i = next(k for k, l in enumerate(L) if l.startswith("*  ISTART"))
    L[i + 1] = re.sub(r"(\d+)\s*$", str(len(sets)), L[i + 1])
    j = next(k for k, l in enumerate(L) if l.startswith("* NITER"))
    n_old = sum(1 for l in L[j + 1:j + 6] if re.match(r"^\s+\d+\s+-9", l))
    L[j + 1:j + 1 + n_old] = ["  " + "    ".join(map(str, s)) for s in sets]
    (d / "tomoDD-SE.inp").write_text("\n".join(L) + "\n")


def prod_sets(src):
    L = inp_lines(src)
    j = next(k for k, l in enumerate(L) if l.startswith("* NITER"))
    return [l.split() for l in L[j + 1:j + 6] if re.match(r"^\s+\d+\s+-9", l)]


def setup_dir(d, src, dt_df=None, events=None):
    shutil.rmtree(d, ignore_errors=True)
    (d / "Input_Files").mkdir(parents=True)
    (d / "Output_Files").mkdir()
    for f in ("MOD", "ak135.15.SKS", "layer-16.dat"):
        os.symlink(src / f, d / f)
    for f in ("station.dat", "absolute.dat"):
        os.symlink(src / "Input_Files" / f, d / "Input_Files" / f)
    if dt_df is None:
        os.symlink(src / "Input_Files" / "dt.ct", d / "Input_Files" / "dt.ct")
    else:
        write_dt(d / "Input_Files" / "dt.ct", dt_df)
    if events is None:
        os.symlink(src / "Input_Files" / "event.dat", d / "Input_Files" / "event.dat")
    else:
        write_events(d / "Input_Files" / "event.dat", events)


def run_tomodd(d, src):
    exe = json.loads((src / "run.json").read_text())["bin"]
    with open(d / "stdout.txt", "w") as so:
        p = subprocess.run([exe, "tomoDD-SE.inp"], cwd=d, stdout=so, stderr=subprocess.STDOUT, timeout=7200)
    txt = (d / "stdout.txt").read_text()
    aq = sum(int(m) for m in re.findall(r"^\s*\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+[-\d.]+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+(\d+)\s+\d+\s*$", txt, re.M))
    return p.returncode, aq


def eval_at(d, src, sets0, start_ev, reloc=None):
    """One-iteration run at the given positions (start events overridden by reloc rows) -> residual table."""
    ev = start_ev.copy()
    if reloc is not None and len(reloc):
        m = ev.ID.isin(reloc.ID)
        r = reloc.set_index("ID").loc[ev.ID[m]]
        ev.loc[m, "lat"], ev.loc[m, "lon"], ev.loc[m, "depth"] = r.lat.values, r.lon.values, r.depth.values
    setup_dir(d, src, events=ev)
    s = list(sets0[0])
    s[0] = "1"
    write_inp(d, src, [s])
    run_tomodd(d, src)
    return read_res(d / "tomoDD-SE.res.ini")


def pair_offset_score(res, train_mask):
    """rms and MAD per phase of held-out rows (~train_mask) after removing each pair's training-row median."""
    key = res.e1.astype(np.int64) * 10_000_000 + res.e2.astype(np.int64)
    off = res.res[train_mask].groupby(key[train_mask]).median()
    h = res[~train_mask].copy()
    h["r"] = h.res - key[~train_mask].map(off)
    h = h[np.isfinite(h.r)]
    out = {}
    for ph in ("P", "S"):
        v = h.r[h.ph == ph].values
        out[ph] = dict(n=int(len(v)), rms=float(np.sqrt(np.mean(v ** 2))) if len(v) else None,
                       mad=float(np.median(np.abs(v))) if len(v) else None)
    return out


def insample_score(res):
    key = res.e1.astype(np.int64) * 10_000_000 + res.e2.astype(np.int64)
    r = res.res - res.res.groupby(key).transform("median")
    return {ph: dict(n=int((res.ph == ph).sum()), rms=float(np.sqrt(np.mean(r[res.ph == ph] ** 2))),
                     mad=float(np.median(np.abs(r[res.ph == ph])))) for ph in ("P", "S")}


def geo(reloc, centre):
    """tomoDD-frame positions back to geographic (inverse of ellipsoid_coords about the frame centre)."""
    from b2_fwd_mods import radii
    M, N = radii(centre[0])
    lat = centre[0] + (reloc.lat.values - centre[0]) * RE_ / M
    lon = centre[1] + (reloc.lon.values - centre[1]) * RE_ / N
    return lat, lon


# ------------------------------------------------------------------ setup
def stage_setup():
    s = pd.read_csv(RUN / "clusters_summary.csv")
    lo, hi = P["size_range"]
    s = s[(s.status == "ok") & (s.part == -1) & s.n_events.between(lo, hi)]
    pick = s.sort_values("n_events", ascending=False).groupby("region").head(P["clusters_per_region"])
    rows = [dict(name=r.cid, src=str(CL / r.cid), region=r.region, n_events=int(r.n_events)) for r in pick.itertuples()]
    # Axial subsets: the n nearest core events to each chunk's median epicentre, rows restricted to them
    for ch in P["axial_subsets"]["chunks"]:
        src = CL / ch
        d = OUT / "src" / f"axial_{ch}"
        if not (d / "run.json").exists():
            ev = read_events(src / "Input_Files" / "event.dat")
            core = set(pd.read_csv(src / "reloc.csv").ID) if (src / "reloc.csv").exists() else set(ev.ID)
            evc = ev[ev.ID.isin(core)]
            la0, lo0 = evc.lat.median(), evc.lon.median()
            dist = np.hypot((evc.lat - la0) * 111.2, (evc.lon - lo0) * 111.2 * np.cos(np.radians(la0)))
            keep = set(evc.ID.values[np.argsort(dist.values)[:P["axial_subsets"]["n_events"]]])
            (d / "Input_Files").mkdir(parents=True, exist_ok=True)
            write_events(d / "Input_Files" / "event.dat", ev[ev.ID.isin(keep)])
            dt = read_dt(src / "Input_Files" / "dt.ct")
            write_dt(d / "Input_Files" / "dt.ct", dt[dt.e1.isin(keep) & dt.e2.isin(keep)])
            ab, cur = [], None
            for l in (src / "Input_Files" / "absolute.dat").read_text().splitlines():
                if l.startswith("#"):
                    cur = int(l.split()[1])
                if cur in keep:
                    ab.append(l)
            (d / "Input_Files" / "absolute.dat").write_text("\n".join(ab) + "\n")
            shutil.copy(src / "Input_Files" / "station.dat", d / "Input_Files" / "station.dat")
            for f in ("MOD", "ak135.15.SKS", "layer-16.dat", "tomoDD-SE.inp"):
                shutil.copy(src / f, d / f)
            rj = json.loads((src / "run.json").read_text())
            (d / "run.json").write_text(json.dumps({"bin": rj["bin"], "center": rj["center"], "from": ch}))
        rows.append(dict(name=f"axial_{ch}", src=str(d), region="Axial", n_events=P["axial_subsets"]["n_events"]))
    sel = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    sel.to_csv(OUT / "clusters.csv", index=False)
    print(sel.groupby("region").size().to_string(), f"\n{len(sel)} clusters")


# ------------------------------------------------------------------ tasks
def task_base(row):
    src, d = Path(row["src"]), OUT / row["name"] / "base"
    f = d / "result.json"
    if f.exists():
        return row["name"]
    ev = read_events(src / "Input_Files" / "event.dat")
    res = eval_at(d / "eval", src, prod_sets(src), ev)
    res.to_parquet(d / "res_b4.parquet")
    sc = insample_score(res)
    f.write_text(json.dumps(dict(insample=sc, ratio=sc["S"]["mad"] / sc["P"]["mad"])))
    return row["name"]


def variant_sets(src, sw, rw, ratio):
    w = 1.0 / ratio ** 2 if sw == "meas" else float(sw)
    sets = []
    for s, (wrct, wdct) in zip(prod_sets(src), P["reweight"][rw]):
        s = list(s)
        s[6] = f"{4.0 * w:.3f}"            # WTCTS: dt.ct S weights already carry the D-B008 factor 1/4
        s[7], s[8] = str(wrct), str(wdct)
        sets.append(s)
    return sets, w


def folds_of(src, k):
    st = [l.split()[0] for l in (src / "Input_Files" / "station.dat").read_text().splitlines() if l.strip()]
    rng = np.random.default_rng(P["seed"])
    f = rng.permutation(len(st)) % k
    return dict(zip(st, f))


def task_variant(job):
    row, sw, rw, kind, ratio = job
    src = Path(row["src"])
    vname = f"s{sw}_{rw}"
    d = OUT / row["name"] / vname / kind
    f = d / "result.json"
    if f.exists():
        return str(f)
    sets, w = variant_sets(src, sw, rw, ratio)
    ev = read_events(src / "Input_Files" / "event.dat")
    centre = json.loads((src / "run.json").read_text())["center"]
    out = dict(cluster=row["name"], region=row["region"], s_weight=sw, s_weight_value=w, reweight=rw, kind=kind)
    if kind == "full" or kind.startswith("fold"):
        dt_df = None
        if kind.startswith("fold"):
            k = int(kind[4:])
            fo = folds_of(src, P["folds"])
            dt = read_dt(src / "Input_Files" / "dt.ct")
            dt_df = dt[dt.sta.map(fo) != k]
        setup_dir(d / "run", src, dt_df=dt_df)
        write_inp(d / "run", src, sets)
        rc, aq = run_tomodd(d / "run", src)
        rp = d / "run" / "Output_Files" / "tomoDD.reloc"
        rel = read_reloc(rp) if rp.exists() and rp.stat().st_size else pd.DataFrame(columns=["ID", "lat", "lon", "depth"])
        out.update(exit=rc, airquakes=aq, relocated=int(len(rel)), events=int(len(ev)))
        res = eval_at(d / "eval", src, prod_sets(src), ev, rel)
        if kind == "full":
            out["insample"] = insample_score(res)
            if len(rel):
                from b4_catalog import surface_km
                la, lo = geo(rel, centre)
                surf = surface_km(la, lo)
                out["above_surface"] = int((rel.depth.values < -surf).sum())
                st0 = ev.set_index("ID").loc[rel.ID]
                la0, lo0 = geo(st0.reset_index(), centre)
                dz = rel.depth.values - st0.depth.values
                dh = np.hypot((la - la0) * 111.195, (lo - lo0) * 111.195 * np.cos(np.radians(la0)))
                out["shift_h_med"], out["shift_z_med"] = float(np.median(dh)), float(np.median(np.abs(dz)))
        else:
            fo = folds_of(src, P["folds"])
            train = (res.sta.map(fo) != int(kind[4:])).values
            out["heldout"] = pair_offset_score(res, train)
            b4 = pd.read_parquet(OUT / row["name"] / "base" / "res_b4.parquet")
            out["heldout_b4"] = pair_offset_score(b4, (b4.sta.map(fo) != int(kind[4:])).values)
        for sub in ("run", "eval"):     # keep results, drop the copied dt rows
            for g in ("Input_Files/dt.ct", "tomoDD-SE.res.ini") if sub == "run" else ("Input_Files/event.dat",):
                p = d / sub / g
                if p.exists() and not p.is_symlink() and kind != "full":
                    p.unlink()
    elif kind == "synth":
        out.update(synth(row, src, ev, centre, sets, d))
    d.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out))
    return str(f)


def synth(row, src, ev, centre, sets, d):
    import b5_relocate as BR
    S = P["synth"]
    rng = np.random.default_rng(P["seed"] + zlib.crc32(row["name"].encode()) % 10_000)   # stable across processes
    st = pd.read_csv(RUN / "prep" / "stations.csv").set_index("code")
    pe = pd.read_parquet(RUN / "prep" / "events.parquet").set_index("ID")
    cat = pd.read_csv(Path(PR["b5"]["reloc_v5"]) / "catalog" / "catalog_b4_b4_iter1_v0.3.1.csv",
                      usecols=["ID", "h_max", "sig_z", "surface_km"]).set_index("ID")
    ev = ev[ev.ID.isin(pe.index)].reset_index(drop=True)
    dt = read_dt(src / "Input_Files" / "dt.ct")
    dt = dt[dt.e1.isin(ev.ID) & dt.e2.isin(ev.ID)]
    # true times from the B4 grids at the B4 positions
    G = BR.Grids(Path(PR["b5"]["reloc_v5"]) / "time_b4_v0.3.1")
    picks = pd.concat([dt[["e1", "sta", "ph"]].rename(columns={"e1": "ID"}), dt[["e2", "sta", "ph"]].rename(columns={"e2": "ID"})]).drop_duplicates().reset_index(drop=True)
    picks["T"] = np.nan
    for (s_, ph), g in picks.groupby(["sta", "ph"]):
        e = g.ID.values
        picks.loc[g.index, "T"] = G.time(ph, st.loc[s_, "station"], pe.x_km.loc[e].values, pe.y_km.loc[e].values, pe.zf_km.loc[e].values)
    # noise: per pick, resampled from the cluster's B4 dt residuals / sqrt(2), per phase
    b4 = pd.read_parquet(OUT / row["name"] / "base" / "res_b4.parquet")
    key = b4.e1.astype(np.int64) * 10_000_000 + b4.e2.astype(np.int64)
    r = (b4.res - b4.res.groupby(key).transform("median")) / np.sqrt(2)
    picks["n"] = 0.0
    for ph in ("P", "S"):
        m = (picks.ph == ph).values
        pool = r[b4.ph == ph].values
        picks.loc[m, "n"] = rng.choice(pool, m.sum()) * rng.choice([-1, 1], m.sum())
    tau = pd.Series(rng.normal(0, S["origin_sigma_s"], len(ev)), index=ev.ID.values)
    picks["tt"] = picks["T"] + picks["n"] + picks.ID.map(tau)
    tt = picks.set_index(["ID", "sta", "ph"]).tt
    dt = dt.copy()
    dt["t1"] = tt.reindex(pd.MultiIndex.from_arrays([dt.e1, dt.sta, dt.ph])).values
    dt["t2"] = tt.reindex(pd.MultiIndex.from_arrays([dt.e2, dt.sta, dt.ph])).values
    dt = dt[np.isfinite(dt.t1) & np.isfinite(dt.t2)]
    # start: truth (event.dat = B4 in the tomoDD frame) + perturbation
    c = cat.reindex(ev.ID)
    sh = np.clip(S["start_scale"] * c.h_max.fillna(1.0).values / 1.515, *S["sigma_h_clip"])
    sz = np.clip(S["start_scale"] * c.sig_z.fillna(1.0).values, *S["sigma_z_clip"])
    de, dn, dz = rng.normal(0, 1, len(ev)) * sh, rng.normal(0, 1, len(ev)) * sh, rng.normal(0, 1, len(ev)) * sz
    start = ev.copy()
    start["lat"] = ev.lat + dn / 111.195
    start["lon"] = ev.lon + de / (111.195 * np.cos(np.radians(ev.lat)))
    zs = ev.depth + dz
    surf = c.surface_km.fillna(0).values
    zs = np.where(zs < -surf, ev.depth + np.abs(dz), zs)      # starts never above the surface (synthetic set-up only)
    start["depth"] = zs
    setup_dir(d / "run", src, dt_df=dt, events=start)
    write_inp(d / "run", src, sets)
    rc, aq = run_tomodd(d / "run", src)
    rp = d / "run" / "Output_Files" / "tomoDD.reloc"
    rel = read_reloc(rp) if rp.exists() and rp.stat().st_size else pd.DataFrame(columns=["ID", "lat", "lon", "depth"])
    (d / "run" / "Input_Files" / "dt.ct").unlink()

    def rel_err(pos):
        t = ev.set_index("ID").loc[pos.ID]
        e = (pos.lon.values - t.lon.values) * 111.195 * np.cos(np.radians(t.lat.values))
        n = (pos.lat.values - t.lat.values) * 111.195
        z = pos.depth.values - t.depth.values
        e, n, z = e - e.mean(), n - n.mean(), z - z.mean()
        return float(np.median(np.hypot(e, n))), float(np.median(np.abs(z)))
    o = dict(exit=rc, airquakes=aq, relocated=int(len(rel)), events=int(len(ev)), dt_rows=int(len(dt)))
    if len(rel) >= 5:
        o["err_start"] = rel_err(start[start.ID.isin(rel.ID)].set_index("ID").loc[rel.ID].reset_index())
        o["err_final"] = rel_err(rel)
    return o


# ------------------------------------------------------------------ stages
def stage_base():
    sel = pd.read_csv(OUT / "clusters.csv").to_dict("records")
    with ProcessPoolExecutor(P["nproc"]) as ex:
        for f in as_completed([ex.submit(task_base, r) for r in sel]):
            print("base", f.result(), flush=True)
    rat = {r["name"]: json.loads((OUT / r["name"] / "base" / "result.json").read_text())["ratio"] for r in sel}
    s = pd.DataFrame(sel)
    s["ratio"] = s.name.map(rat)
    reg = s.groupby("region").ratio.median()
    s["ratio_region"] = s.region.map(reg)
    s.to_csv(OUT / "clusters.csv", index=False)
    print(s[["name", "region", "ratio", "ratio_region"]].to_string(index=False))


def stage_run():
    sel = pd.read_csv(OUT / "clusters.csv").to_dict("records")
    kinds = ["full", "synth"] + [f"fold{k}" for k in range(P["folds"])]
    jobs = [(r, sw, rw, kind, r["ratio_region"]) for r in sorted(sel, key=lambda r: -r["n_events"])
            for kind in kinds for sw in P["s_weight"] for rw in P["reweight"]]
    n = 0
    with ProcessPoolExecutor(P["nproc"]) as ex:
        futs = [ex.submit(task_variant, j) for j in jobs]
        for f in as_completed(futs):
            n += 1
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                print("FAILED", repr(e)[:300], flush=True)
            if n % 50 == 0:
                print(f"{n}/{len(jobs)}", flush=True)
    print(f"run done: {n}/{len(jobs)}")


def stage_collect():
    rows = []
    for f in OUT.glob("*/*/*/result.json"):
        if f.parts[-3] == "base":
            continue
        rows.append(json.loads(f.read_text()))
    R = pd.json_normalize(rows)
    od = OUT / "summary"
    od.mkdir(exist_ok=True)
    R.to_csv(od / "results.csv", index=False)
    write_manifest("b5_wtest", od, outputs=[od / "results.csv"], params=P, counts={"results": len(R)})
    print(len(R), "results ->", od / "results.csv")


if __name__ == "__main__":
    {"setup": stage_setup, "base": stage_base, "run": stage_run, "collect": stage_collect}[sys.argv[1]]()

"""Parse NLLoc *.hyp summary files of a run into one CSV and compare with catalog v4.

Usage: python a2_parse_hyp.py RUN_NAME [--deep DEEP_RUN]  ->  /wd1/mdenolle_data/reloc_v5/runs/RUN_NAME/locations.csv
Columns: v4 ID, status, NLLoc comment (why REJECTED), maximum-likelihood origin (depth in the run's frame: z' for the
Earth-flattened B4 grids), expectation depth, RMS, phases used, gap, distances, 68% horizontal ellipse (min/max), 3D
ellipsoid semi-axes, depth sigma (sqrt CovZZ), ns_z (smallestNodeSide z of the SEARCH OCTREE line: the final oct-tree
cell height, km), run and tile (workdir) of the row, pass ('main' or 'deep'), v4 values, surface check.
One row per event. --deep DEEP_RUN (D-B067, v5.1: b4_v51_deep) also reads the second pass of the events rejected at
the bottom face or located at z' >= 78 km; an event's row is its deep-pass row when that is LOCATED, else its main-pass
row (LOCATED preferred, else the last one read; files in name order), else its deep-pass row.
v4 values come from the catalog v4.1 event list (V4).
"""
import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
B = Path("/wd1/mdenolle_data/reloc_v5")
# catalog v4.1 event list (v4.0 minus 172113, plus 900001-900003 with ComCat hypocentres; utils/fix_catalog_v4.py).
# data/catalog_v4 is gitignored and ~/wt_reloc/data/catalog_v4 links to the v4.0 backup, so the main tree's file is named
V4 = Path("/home/mdenolle/cascadia_obs_ensemble/data/catalog_v4/Cascadia_relocated_catalog_v4.csv")
V41_REMOVED, V41_ADDED = {172113}, {900001, 900002, 900003}
ANSS = REPO / "data/magnitude_v4/anss_2010-16.csv"


def kv(tokens):
    return {tokens[i]: tokens[i + 1] for i in range(len(tokens) - 1)}


def parse(path):
    rows, cur = [], None
    for line in open(path):
        tok = line.split()
        if not tok:
            continue
        k = tok[0]
        if k == "NLLOC":
            q = re.findall(r'"([^"]*)"', line)     # file root, status, comment (e.g. why REJECTED)
            cur = {"status": q[1], "comment": q[2] if len(q) > 2 else ""}
        elif cur is None:
            continue
        elif k == "SEARCH" and tok[1] == "OCTREE":
            cur["ns_z"] = float(kv(tok[2:])["smallestNodeSide"].split("/")[2])
        elif k == "PUBLIC_ID":
            cur["ID"] = int(tok[1].split("_")[1])
        elif k == "GEOGRAPHIC":
            cur["time"] = f"{tok[2]}-{tok[3]}-{tok[4]}T{tok[5]}:{tok[6]}:{float(tok[7]):09.6f}"
            d = kv(tok[8:])
            cur.update(lat=float(d["Lat"]), lon=float(d["Long"]), depth=float(d["Depth"]))
        elif k == "QUALITY":
            d = kv(tok[1:])
            cur.update(rms=float(d["RMS"]), nphs=int(d["Nphs"]), gap=float(d["Gap"]), dist_min=float(d["Dist"]))
        elif k == "STATISTICS":
            d = kv(tok[1:])
            cur.update(sig_z=np.sqrt(float(d["ZZ"])), ell_len3=float(d["Len3"]))
        elif k == "STAT_GEOG":
            d = kv(tok[1:])
            cur.update(exp_lat=float(d["ExpectLat"]), exp_lon=float(d["Long"]), exp_depth=float(d["Depth"]))
        elif k == "QML_OriginQuality":
            d = kv(tok[1:])
            cur.update(n_sta=int(d["usedStaCt"]), dist_max=float(d["maxDist"]), dist_med=float(d["medDist"]))
        elif k == "QML_OriginUncertainty":
            d = kv(tok[1:])
            cur.update(h_min=float(d["minHorUnc"]), h_max=float(d["maxHorUnc"]))
        elif k == "END_NLLOC":
            rows.append(cur)
            cur = None
    return rows


REGIONS = [  # name, lat0, lat1, lon0, lon1 (first match wins; v4 epicentre)
    ("Axial", 45.7, 46.2, -130.3, -129.7), ("Endeavour", 47.7, 48.3, -129.5, -128.7),
    ("Explorer", 48.3, 51.5, -131.5, -127.5), ("Blanco", 42.8, 44.7, -130.8, -126.4),
    ("Gorda", 40.5, 42.8, -128.0, -124.9), ("Mendocino", 39.5, 40.5, -128.5, -123.5),
    ("JdF ridge/plate", 44.7, 48.3, -131.5, -127.3), ("Offshore forearc", 40.5, 51.5, -127.3, -124.0),
    ("Puget/Georgia", 46.8, 49.5, -123.5, -121.8), ("Cascades", 40.5, 49.5, -122.4, -120.5),
]


def region(lat, lon):
    for name, a, b, c, d in REGIONS:
        if a <= lat < b and c <= lon < d:
            return name
    return "Other onshore"


def read_v4():
    """Catalog v4.1 event list with v4_* column names (stops if V4 is not the v4.1 list)."""
    v4 = pd.read_csv(V4, skipinitialspace=True)
    v4.columns = [c.strip() for c in v4.columns]
    ids = set(v4.ID)
    assert not ids & V41_REMOVED and V41_ADDED <= ids and v4.ID.is_unique, f"{V4} is not the catalog v4.1 event list"
    v4 = v4.rename(columns={"datetime": "v4_time", "latitude": "v4_lat", "longitude": "v4_lon", "depth(km)": "v4_depth",
                            "spatial_sigma(m)": "v4_sigma_m"})
    v4["v4_npick"] = v4.Num_p_picks + v4.Num_s_picks
    return v4


def read_run(run, pas):
    rows = []
    for f in sorted((B / "runs" / run).glob("*/loc.sum.grid0.loc.hyp")):
        for r in parse(f):
            r.update(run=run, tile=f.parent.name, **{"pass": pas})
            rows.append(r)
    assert rows, f"no loc.sum.grid0.loc.hyp under {B / 'runs' / run}"
    return rows


def one_row(loc):
    """One row per event (docstring): deep LOCATED > main LOCATED > main other > deep other; ties: last read."""
    ok = loc.status == "LOCATED"
    deep = loc["pass"] == "deep"
    k = np.select([deep & ok, ok, ~deep], [3, 2, 1], 0)
    n_dup = int(loc[~deep].ID.duplicated().sum())
    loc = loc.assign(_k=k, _i=np.arange(len(loc))).sort_values(["ID", "_k", "_i"])
    return loc.drop_duplicates("ID", keep="last").drop(columns=["_k", "_i"]).reset_index(drop=True), n_dup


DEEP_Z, BOTTOM = 78.0, 31          # deep-pass trigger, as a2_run_nlloc.py: ML z' (km); NLLoc bottom-face boundary code
SAME = ("time", "time_wide", "fine_manifest_md5", "wide_manifest_md5", "template_md5", "locgau", "locgau2", "topo",
        "topo_md5", "topo_hdr_md5", "stat", "stat_md5", "max_dist", "nll_md5")   # a deep run must equal its main run


def run_cfg(run):
    p = B / "runs" / run / "run_config.json"
    return json.loads(p.read_text()) if p.exists() else None


def deep_triggers(rows, run):
    """IDs of the main run that the deep pass (D-B067) must cover: REJECTED at the bottom face or LOCATED at
    ML z' >= DEEP_Z, and not far_only (n_fine > 0 in arrivals_used.csv). None for runs made before the v5.1 runner
    (no arrivals_used.csv: no deep pass exists for them)."""
    used = B / "runs" / run / "arrivals_used.csv"
    if not used.exists():
        return None
    a = pd.read_csv(used)
    n_fine = a[a["pass"] == "main"].set_index("ID").n_fine
    out = set()
    for r in rows:
        b = re.search(r"grid boundary (\d+)", r.get("comment", ""))
        trig = ((r["status"] == "REJECTED" and b is not None and int(b.group(1)) == BOTTOM) or
                (r["status"] == "LOCATED" and r.get("depth") is not None and r["depth"] >= DEEP_Z))
        if trig and n_fine.get(r["ID"], 0) > 0:
            out.add(r["ID"])
    return out


def check_deep(run, deep, deep_rows, trig):
    """the deep run belongs to this main run (same grids, template, likelihood, floor, station terms, NLLoc) and covers
    every event its selection and the trigger rule name; stops otherwise. Returns the IDs it should have and lacks."""
    mc, dc = run_cfg(run), run_cfg(deep)
    if mc is None or dc is None:
        raise SystemExit(f"--deep needs run_config.json in both runs (a2_run_nlloc.py v5.1): {run}, {deep}")
    if dc.get("pass") != "deep":
        raise SystemExit(f"{deep} is not a deep-pass run (run_config pass = {dc.get('pass')!r})")
    if Path(str(dc.get("obs", ""))).resolve() != (B / "runs" / run).resolve():
        raise SystemExit(f"deep run {deep} was made from {dc.get('obs')}, not from {B / 'runs' / run}")
    diff = {k: (mc.get(k), dc.get(k)) for k in SAME if mc.get(k) != dc.get(k)}
    if diff:
        raise SystemExit(f"deep run {deep} differs from the main run {run} (main, deep): {diff}")
    sel = pd.read_csv(B / "runs" / deep / "deep_selection.csv")
    want = set(sel.ID[sel.selected.astype(bool)])
    if trig is not None and not trig <= want:
        raise SystemExit(f"{len(trig - want)} triggered events of {run} are not selected in {deep}/deep_selection.csv "
                         f"(e.g. {sorted(trig - want)[:5]}): the deep run was made from another state of the main run")
    missing = sorted(want - {r["ID"] for r in deep_rows})
    if missing:
        raise SystemExit(f"deep pass {deep} incomplete: {len(missing)} selected events have no row (e.g. {missing[:5]})")
    return missing


def main(run, deep=None, no_deep=False):
    rd = B / "runs" / run
    main_rows = read_run(run, "main")
    trig = deep_triggers(main_rows, run)
    deep_rows = read_run(deep, "deep") if deep else []
    if deep:
        check_deep(run, deep, deep_rows, trig)
    elif trig and not no_deep:
        raise SystemExit(f"{len(trig)} events of {run} need the deep pass (D-B067: REJECTED at the bottom face or "
                         f"located at z' >= {DEEP_Z:g} km, not far_only): give --deep DEEP_RUN, or --no-deep to keep "
                         "their main-pass rows on purpose")
    rows = main_rows + deep_rows
    loc, n_dup = one_row(pd.DataFrame(rows))
    if deep:
        d = loc["pass"] == "deep"
        print(f"deep pass {deep}: {sum(r['pass'] == 'deep' for r in rows)} rows, {int(d.sum())} events take the deep "
              f"row ({int((d & (loc.status == 'LOCATED')).sum())} LOCATED)")
    if n_dup:
        print(f"{n_dup} duplicate main-pass rows dropped (one row per event)")
    v4 = read_v4()
    miss = sorted(set(loc.ID) - set(v4.ID))
    if miss:
        print(f"{len(miss)} events of the run are not in the v4.1 event list (kept here, no v4 values): {miss[:10]}")
    loc = loc.merge(v4[["ID", "v4_lat", "v4_lon", "v4_depth", "v4_npick", "v4_sigma_m"]], on="ID", how="left")
    # surface elevation under the new and v4 epicentres
    import pygmt
    rel = pygmt.datasets.load_earth_relief("01m", region=[-134, -118, 37.5, 52.5], registration="gridline")
    for pre, la, lo in (("", "lat", "lon"), ("v4_", "v4_lat", "v4_lon")):
        e = rel.interp(lat=("p", loc[la].values), lon=("p", loc[lo].values)).values / 1e3
        loc[f"{pre}surface_km"] = e
    loc["above_surface"] = loc.depth < -loc.surface_km - 1e-3
    loc["v4_above_surface"] = loc.v4_depth < -loc.v4_surface_km - 1e-3
    R = 6371.0
    la1, lo1, la2, lo2 = map(np.radians, (loc.lat, loc.lon, loc.v4_lat, loc.v4_lon))
    loc["shift_h_km"] = 2 * R * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2)**2 +
                                                 np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2)**2))
    loc["shift_z_km"] = loc.depth - loc.v4_depth
    loc["region"] = [region(a, b) for a, b in zip(loc.v4_lat, loc.v4_lon)]
    loc.to_csv(rd / "locations.csv", index=False)
    # read by b4_catalog.py: the deep pass was checked here (written after locations.csv, same invocation)
    (rd / "deep_check.json").write_text(json.dumps({
        "run": run, "deep_run": deep, "v51_runner": trig is not None, "triggers": len(trig or ()),
        "deep_omitted": bool(trig) and not deep, "deep_rows": len(deep_rows),
        "locations_md5": __import__("hashlib").md5((rd / "locations.csv").read_bytes()).hexdigest()}, indent=1) + "\n")
    ok = loc[loc.status == "LOCATED"]
    print(f"{len(loc)} events, {len(ok)} located; above surface: new {ok.above_surface.mean():.1%}, "
          f"v4 (same events) {ok.v4_above_surface.mean():.1%}")
    key = "tile" if loc.tile.nunique() <= 12 else "region"
    print(ok.groupby(key).agg(n=("ID", "size"), rms=("rms", "median"), nphs=("nphs", "median"),
                                 v4_npick=("v4_npick", "median"), h_max=("h_max", "median"),
                                 sig_z=("sig_z", "median"), dz=("shift_z_km", "median"),
                                 dh=("shift_h_km", "median"), depth=("depth", "median"),
                                 v4_depth=("v4_depth", "median"), above=("above_surface", "mean"),
                                 v4_above=("v4_above_surface", "mean")).round(2).to_string())
    anss_check(ok, rd)


def anss_check(loc, rd):
    """Depth vs ANSS/ComCat (PNSN, NCSN) for events matched within 4 s and 30 km."""
    a = pd.read_csv(ANSS)
    a["t"] = pd.to_datetime(a.time.str.rstrip("Z"), format="ISO8601", utc=True)
    a = a.rename(columns={"latitude": "a_lat", "longitude": "a_lon", "depth": "a_depth"}).sort_values("t")
    l = loc.assign(t=pd.to_datetime(loc.time, utc=True)).sort_values("t")
    m = pd.merge_asof(l, a[["t", "a_lat", "a_lon", "a_depth", "net"]], on="t",
                      tolerance=pd.Timedelta("4s"), direction="nearest").dropna(subset=["a_lat"])
    d = np.hypot((m.lat - m.a_lat) * 111.2, (m.lon - m.a_lon) * 111.2 * np.cos(np.radians(m.lat)))
    m = m[d < 30].copy()
    m["dz_new"] = m.depth - m.a_depth
    m["dz_v4"] = m.v4_depth - m.a_depth
    m.to_csv(rd / "anss_matched.csv", index=False)
    print(f"\nANSS depth check: {len(m)} matched (median depth; median |dz|)")
    key = "tile" if m.tile.nunique() <= 12 else "region"
    print(m.groupby(key).agg(n=("ID", "size"), anss=("a_depth", "median"), v4=("v4_depth", "median"),
                                new=("depth", "median"), v4_absdz=("dz_v4", lambda s: s.abs().median()),
                                new_absdz=("dz_new", lambda s: s.abs().median())).round(1).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--deep", help="deep-pass run (D-B067) whose LOCATED rows replace the main-pass rows")
    ap.add_argument("--no-deep", action="store_true",
                    help="keep the main-pass rows of events that need the deep pass (b4_catalog.py then refuses)")
    a = ap.parse_args()
    main(a.run, a.deep, a.no_deep)

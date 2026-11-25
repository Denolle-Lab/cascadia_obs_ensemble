"""Track B, step B4: catalog from the NLLoc relocation in a registered model, depths back from the Earth-flattened
frame, the surface check, the v5.1 flags, and a comparison with v4-NLL3D (Track A iter1).

Input: Track A's catalog of the run (a2_build_catalog.py RUN -> /wd1/mdenolle_data/reloc_v5/catalog/
catalog_v5_trackA_<RUN>.csv: every event of the catalog v4.1 event list, NLLoc location or none (tier U, loc_source
'none': no time, lat, lon or depth, the v4 values only in v4_*, u_reason says why; runs built before v5.1: loc_source
'v4', v4 hypocentre kept), tiers A-D/U). That file is written with '%.5g' (lon to 0.01 deg, up to 0.4 km;
a2_build_catalog.py of the reloc/track-a worktree), so lat, lon, depth and exp_depth of the located events are taken
again from runs/<RUN>/locations.csv (a2_parse_hyp.py: one row per event, the deep-pass row when it is LOCATED), and the
v4 coordinates from the catalog v4.1 event list (params catalog_versions.v4_1_events), both at full precision; the
output is written with '%.8g' (D-B013). The time grids of b4_time_grids.py use flattened depth z'; here depth,
exp_depth and shift_z_km of the located events (loc_source nll3d) are mapped back with z = Re (1 - exp(-z' / Re))
(0.07 km at 30 km, 0.2 km at 50 km). Tiers are kept (their depth conditions move by less than the change).
Events of the v4.1 additions (params catalog_versions.v4_1_added) that NLLoc did not locate are left out of this
catalog: catalog_version.py adds them with their ComCat hypocentre. Rows without a hypocentre keep region (box label of
the v4 epicentre, a2_build_catalog.py) and tier U; every other derived column of theirs is empty.
Flags (rule 4 of D-B049..D-B051, PI 2026-11-24; /wd1/mdenolle_data/tomo_v1/depth_literature/depth_literature.html),
parameters in params b4.depth_flags and b4.quarry (empty on rows without a hypocentre):
* depth_quality: "unconstrained" for every event of the Endeavour, Explorer and JdF ridge/plate regions and for Blanco
  events outside the 2012-09-20 to 2013-10-05 OBS deployment (UTC dates, inclusive); "biased_deep" for Gorda events
  with dist_min > 15 km; "ok" otherwise (rows without dist_min, e.g. ComCat rows, are not tested by the Gorda rule);
* event_flag: "quarry_blast_likely" for events in the Texada Island box (lat 49.40-49.85, lon -124.75 to -124.25)
  on a weekday between 07:00 and 19:00 America/Vancouver time, "" otherwise.
  D-B071 (PI 2026-11-24): plus "anss_explosion", "anss_quarry_blast", "anss_other_event" or "anss_sonic_boom" for
  located events within 3 s and 20 km of a ComCat event of that type (data/tomo_v1/ref_models/comcat_nonquake_2010-2016.csv),
  joined with ";" to the Texada flag; a surface depth is right for these events (19 % of onshore floor-held events).
Surface check (D-B055, PI 2026-11-24; D-B068): an event above the solid surface (land or seafloor) is a computation
error, never corrected by moving it, and a catalog with any such event is not publishable. surface_km (km, positive up)
is the solid surface at the final epicentre of every event with a hypocentre (empty otherwise): params
b4.surface.source "merged" -> b4_surface.surface_m (D-B068: GMRT patches where available, earth_relief 01m elsewhere,
bilinear on the source grid of each point), any other setting (default) -> GMT earth_relief 01m (gridline), bilinear;
"merged" never falls back to earth_relief. The build stops unless every located row (main and deep runs) used one and
the same NLLoc floor (LOCTOPO_SURFACE of its nlloc.in), and with "merged" that floor must be params
b4.surface.floor.file; with another surface it must not be a merged floor. above_surface is
depth < -surface_km on the true (unflattened) depth. Depths are never changed here; counts go to compare.md and the
manifest ("publishable").
v5.1 flags (booleans on NLLoc-located rows, empty on every other row):
* floor_limited: the maximum likelihood point lies within FLOOR_CELLS = 1.5 final oct-tree cells (ns_z, the
  smallestNodeSide z of the SEARCH OCTREE line, a2_parse_hyp.py) of the floor NLLoc used for it: the LOCTOPO_SURFACE
  file of the row's own nlloc.in (main or deep run), read as NLLoc reads it (velmod.c read_grd / get_surface_z: float32,
  nearest node with pix_shift 0.5 for "Normal" registration) and compared, as NLLoc does, with the NLLoc depth in the
  run's frame (Earth-flattened z' for B4, against the flattened floor file). Such depths are set by the floor, not the
  data (round 2, 2026-11-24: the pile-up on the floor extends to about 1.5 cells; 9,481 of 157,257 b4_iter0_v0.3.1
  events within 1 cell).
* far_only: none of the event's used arrivals in the main pass came from a fine time grid (main-pass row of
  runs/<run>/arrivals_used.csv, n_fine == 0): seen only by distant stations through the wide grids, so its depth is poor
  (round 2: ANSS depth differences 21-24 km). The main pass, because the deep pass reads every arrival from the wide
  deep grids, and far-only events are not sent to it (D-B067).
* deep_pass: the hypocentre comes from the deep pass (D-B067; the locations.csv row of pass 'deep', which needs its
  deep row in arrivals_used.csv): wide grids to z' ~121 km, ak135 below 80 km and outside the tomography box.
* five_phase: located with exactly 5 phases (D-B066; tier D, a2_build_catalog.py).
arrivals_used.csv (interface I2 of a2_run_nlloc.py: ID, n_obs, n_used, n_fine, n_wide, pass) is read from every run
named in locations.csv; every located event needs its main-pass row, and a deep-pass hypocentre its deep row too.
Deep pass (D-B067): for a run made by the v5.1 runner (it has arrivals_used.csv) a2_parse_hyp.py must have written
deep_check.json for this locations.csv (md5): the build stops if the deep pass was omitted (--no-deep) while events
needed it, or if the deep rows come from another run than the one a2_parse_hyp.py checked (it checks the deep run's
run_config.json against the main run's: grids by manifest md5, template, likelihood, floor, station terms, NLLoc).
Outputs: /wd1/mdenolle_data/reloc_v5/catalog/catalog_b4_<RUN>.csv, and data/tomo_v1/b4/<RUN>/ (compare.md,
compare_by_region.csv, depth_hist.png).

  python b4_catalog.py RUN [--ref iter1_vpvs176]
"""
import argparse
import json
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import REPO, load_params, write_manifest  # noqa: E402

_P = load_params()
B4 = _P.get("b4") or {}
CV = _P["catalog_versions"]
# defaults = the values agreed with the PI (2026-11-24); params b4.depth_flags / b4.quarry override them key by key
DF = {"unconstrained_regions": ["Endeavour", "Explorer", "JdF ridge/plate"],
      "blanco_window": ["2012-09-20", "2013-10-05"], "gorda_biased_dist_km": 15.0}
QB = {"name": "Texada Island", "lat": [49.40, 49.85], "lon": [-124.75, -124.25], "weekday_hours_local": [7, 19],
      "tz": "America/Vancouver"}
for _d, _k in ((DF, "depth_flags"), (QB, "quarry")):
    assert set(B4.get(_k) or {}) <= set(_d), f"params b4.{_k}: unknown keys {set(B4[_k]) - set(_d)}"   # else ignored
    _d.update(B4.get(_k) or {})
# solid surface (docstring): params b4.surface.source, "merged" (D-B068) or earth_relief 01m (any other value)
SURFACE = (B4.get("surface") or {}).get("source", "earth_relief_01m")
RE = 6371.0
FLOOR_CELLS = 1.5          # floor_limited: within this many final oct-tree cells (ns_z) of the NLLoc floor (docstring)
CAT = Path("/wd1/mdenolle_data/reloc_v5/catalog")
RUNS = Path("/wd1/mdenolle_data/reloc_v5/runs")
# catalog v4.1 event list, the one a2_parse_hyp.py / a2_build_catalog.py (reloc/track-a, v5.1) read; the check in
# main() stops the run when the coordinates differ beyond the 5th digit or an event is missing
V4 = Path(CV["v4_1_events"])


def unflatten(z):
    return RE * (1 - np.exp(-np.asarray(z, float) / RE))


def surface_source():
    """Text naming the solid surface of surface_km (manifests, compare.md)."""
    if SURFACE == "merged":
        import b4_surface                         # D-B068 (interface I1); no fallback
        return str(b4_surface.SOURCE)
    return "GMT earth_relief 01m (gridline), bilinear"


def surface_km(lat, lon):
    """Solid-surface elevation (km, positive up) at (lat, lon): params b4.surface.source "merged" -> b4_surface.surface_m
    (D-B068), else GMT earth_relief 01m, gridline, bilinear (as a2_parse_hyp.py). No fallback between the two."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    if SURFACE == "merged":
        from b4_surface import surface_m          # an ImportError stops the build (no silent earth_relief)
        e = np.asarray(surface_m(lon, lat), float) / 1e3
    else:
        import pygmt
        rel = pygmt.datasets.load_earth_relief("01m", region=[-134, -118, 37.5, 52.5], registration="gridline")
        e = rel.interp(lat=("p", lat), lon=("p", lon)).values / 1e3
    assert np.isfinite(e).all(), f"{int((~np.isfinite(e)).sum())} epicentres outside the surface ({SURFACE})"
    return e


def nlloc_value(run, tile, key):
    """Value (rest of the line) of the control statement `key` in runs/<run>/<tile>/nlloc.in (exactly one)."""
    f = RUNS / run / tile / "nlloc.in"
    v = [l.split(None, 1)[1].strip() for l in f.read_text().splitlines() if l.split(None, 1)[:1] == [key]]
    assert len(v) == 1, f"{f}: {len(v)} {key} lines"
    return v[0]


def read_floor(path):
    """LOCTOPO_SURFACE grid as NLLoc holds it (velmod.c read_grd, .bin + .hdr pair): float32 elevation (m) of
    zval * scale_factor, rows north -> south, and the header numbers get_surface_z uses."""
    assert str(path).endswith(".bin"), f"{path}: only the binary .bin/.hdr surface format is read here"
    txt = Path(f"{path}.hdr").read_text()
    h = txt.splitlines()
    g = {k: float(re.search(rf"\b{k}: (\S+)", txt).group(1))
         for k in ("x_min", "x_inc", "nx", "y_min", "y_inc", "ny", "scale_factor")}
    shift = 0.5 if h[3].split()[1] == "Normal" else 0.0
    assert h[5].split("name:")[1].split()[0] != "km", f"{path}: x/y in km (only lat/lon floors are read here)"
    nx, ny = int(g["nx"]), int(g["ny"])
    z = (np.fromfile(path, "<f8") * g["scale_factor"]).astype(np.float32)
    assert z.size == nx * ny, f"{path}: {z.size} values, header {nx} x {ny}"
    return dict(z=z.reshape(ny, nx), nx=nx, ny=ny, x0=g["x_min"], y0=g["y_min"], dx=g["x_inc"], dy=g["y_inc"],
                shift=shift)


def floor_at(fl, lat, lon):
    """velmod.c get_surface_z: value of the nearest node (int truncation of pix_shift + offset / inc); NaN outside."""
    ix = np.floor(fl["shift"] + (np.asarray(lon) - fl["x0"]) / fl["dx"]).astype(int)
    iy = fl["ny"] - 1 - np.floor(fl["shift"] + (np.asarray(lat) - fl["y0"]) / fl["dy"]).astype(int)
    ok = (ix >= 0) & (ix < fl["nx"]) & (iy >= 0) & (iy < fl["ny"])
    v = np.full(len(ix), np.nan)
    v[ok] = fl["z"][iy[ok], ix[ok]].astype(float)
    return v


def floor_height(loc):
    """Height (km) of each LOCATED row of locations.csv above the NLLoc floor of its own run and workdir, in the run's
    frame (NLLoc: rejected when -z * 1000 > floor), and the floor file of each row."""
    key = loc[["run", "tile"]].drop_duplicates()
    paths = {(r, t): nlloc_value(r, t, "LOCTOPO_SURFACE").split()[0] for r, t in key.itertuples(index=False)}
    f = pd.Series([paths[(r, t)] for r, t in zip(loc.run, loc.tile)], index=loc.index)
    h = pd.Series(np.nan, index=loc.index)
    for p, ix in f.groupby(f).groups.items():
        fl = read_floor(p)
        h.loc[ix] = loc.loc[ix, "depth"].to_numpy(float) + floor_at(fl, loc.loc[ix, "lat"], loc.loc[ix, "lon"]) / 1e3
    return h, f


def check_deep_pass(run):
    """D-B067: a run made by the v5.1 runner (it has arrivals_used.csv) needs the deep-pass check that a2_parse_hyp.py
    writes next to locations.csv in the same invocation (deep_check.json): same locations.csv (md5), deep pass not
    omitted, and the deep rows of locations.csv from the deep run it checked (provenance: same grids, template,
    likelihood, floor, station terms, NLLoc). Returns the deep run name or None."""
    rd = RUNS / run
    if not (rd / "arrivals_used.csv").exists():
        return None                                   # a run made before the v5.1 runner: no deep pass exists
    f = rd / "deep_check.json"
    if not f.exists():
        raise SystemExit(f"{f} missing: rerun a2_parse_hyp.py (v5.1) on {run}, with --deep for its deep pass")
    dc = json.loads(f.read_text())
    md5 = hashlib.md5((rd / "locations.csv").read_bytes()).hexdigest()
    if dc.get("locations_md5") != md5:
        raise SystemExit(f"{f} belongs to another locations.csv: rerun a2_parse_hyp.py on {run}")
    if dc.get("deep_omitted"):
        raise SystemExit(f"{run}: {dc.get('triggers')} events needed the deep pass (D-B067) and a2_parse_hyp.py was run "
                         "with --no-deep: run the deep pass and parse with --deep")
    return dc.get("deep_run")


def arrivals_used(loc):
    """Main-pass rows of runs/<run>/arrivals_used.csv (interface I2) for the LOCATED rows of locations.csv, read from
    every run named there; checks that each deep-pass hypocentre has its deep row. Stops when a file or row is missing.
    far_only comes from the main pass: in the deep pass every arrival is read from the wide deep grids (the fine grids
    stop at z' = 80 km), and far-only events are not sent to it (D-B067)."""
    a = []
    for r in sorted(set(loc.run)):
        f = RUNS / r / "arrivals_used.csv"
        if f.exists():
            a.append(pd.read_csv(f))
    assert a, f"no arrivals_used.csv in {sorted(set(loc.run))} (interface I2): far_only and deep_pass need it"
    a = pd.concat(a, ignore_index=True).drop_duplicates()
    assert {"ID", "n_obs", "n_used", "n_fine", "n_wide", "pass"} <= set(a.columns), list(a.columns)
    assert set(a["pass"]) <= {"main", "deep"}, set(a["pass"])
    assert not a.duplicated(["ID", "pass"]).any(), "arrivals_used.csv: several different rows for one event and pass"
    if (a["pass"] == "deep").any() and not (loc["pass"] == "deep").any():
        raise SystemExit("arrivals_used.csv has deep-pass rows but locations.csv none: run a2_parse_hyp.py with --deep")
    deep = loc.loc[loc["pass"] == "deep", ["ID"]].assign(**{"pass": "deep"}).merge(a, on=["ID", "pass"], how="left")
    assert deep.n_fine.notna().all(), f"{int(deep.n_fine.isna().sum())} deep-pass hypocentres without a deep row"
    m = loc[["ID"]].assign(**{"pass": "main"}).merge(a, on=["ID", "pass"], how="left").set_index(loc.index)
    assert m.n_fine.notna().all(), (f"{int(m.n_fine.isna().sum())} located events without a main-pass row in "
                                    f"arrivals_used.csv, e.g. {m[m.n_fine.isna()].ID.head(5).tolist()}")
    return m


def km_h(a, b):
    y = (a.lat.values - b.lat.values) * 111.195
    x = (a.lon.values - b.lon.values) * 111.195 * np.cos(np.radians(a.lat.values))
    return np.hypot(x, y)


def flags(c):
    """depth_quality and event_flag of catalog c (docstring; final lat, lon, time in UTC, region, dist_min)."""
    t = pd.to_datetime(c.time.str.rstrip("Z"), format="ISO8601").dt.tz_localize("UTC")
    w0, w1 = (pd.Timestamp(d, tz="UTC") for d in DF["blanco_window"])
    q = np.full(len(c), "ok", dtype=object)
    q[(c.region == "Gorda").to_numpy() & (c.dist_min > DF["gorda_biased_dist_km"]).to_numpy()] = "biased_deep"
    q[c.region.isin(DF["unconstrained_regions"]).to_numpy()] = "unconstrained"
    q[((c.region == "Blanco") & ((t < w0) | (t >= w1 + pd.Timedelta(days=1)))).to_numpy()] = "unconstrained"
    tl = t.dt.tz_convert(QB["tz"])
    qb = (c.lat.between(*QB["lat"]) & c.lon.between(*QB["lon"]) & (tl.dt.weekday < 5)
          & (tl.dt.hour >= QB["weekday_hours_local"][0]) & (tl.dt.hour < QB["weekday_hours_local"][1]))
    flag = np.where(qb, "quarry_blast_likely", "").astype(object)
    an = anss_nonquake(c, t)                       # D-B071
    both = (flag != "") & (an != "")
    flag[both] = flag[both] + ";" + an[both]
    flag[(flag == "") & (an != "")] = an[(flag == "") & (an != "")]
    return q, flag


NONQUAKE = REPO / "data" / "tomo_v1" / "ref_models" / "comcat_nonquake_2010-2016.csv"   # ComCat non-earthquakes (README)
ANSS_DT_S, ANSS_DX_KM = 3.0, 20.0


def anss_nonquake(c, t):
    """D-B071: 'anss_<type>' (explosion, quarry_blast, other_event, sonic_boom) for rows within ANSS_DT_S seconds and
    ANSS_DX_KM km of an ANSS/ComCat event that is not an earthquake (nearest in time among those within the distance);
    '' otherwise and for rows without a time or epicentre."""
    a = pd.read_csv(NONQUAKE)
    out = np.full(len(c), "", dtype=object)
    if a.empty:
        return out
    ta = a["t"].to_numpy(float)
    o = np.argsort(ta)
    ta, la, lo = ta[o], a["lat"].to_numpy(float)[o], a["lon"].to_numpy(float)[o]
    ty = a["type"].str.replace(" ", "_").to_numpy()[o]
    ok = t.notna().to_numpy() & c.lat.notna().to_numpy() & c.lon.notna().to_numpy()
    tc = (t - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()      # NaN where the row has no time
    lat, lon = c.lat.to_numpy(float), c.lon.to_numpy(float)
    for i in np.flatnonzero(ok):
        j0, j1 = np.searchsorted(ta, tc[i] - ANSS_DT_S), np.searchsorted(ta, tc[i] + ANSS_DT_S, side="right")
        best, bdt = "", np.inf
        for j in range(j0, j1):
            p1, p2 = np.radians(lat[i]), np.radians(la[j])
            h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lo[j] - lon[i]) / 2) ** 2
            if 2 * 6371.0 * np.arcsin(np.sqrt(h)) <= ANSS_DX_KM and abs(ta[j] - tc[i]) < bdt:
                best, bdt = f"anss_{ty[j]}", abs(ta[j] - tc[i])
        out[i] = best
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ref", default="iter1_vpvs176")
    a = ap.parse_args()
    c = pd.read_csv(CAT / f"catalog_v5_trackA_{a.run}.csv", dtype={"tier": str})
    nll = c.loc_source == "nll3d"
    # full-precision coordinates in place of the '%.5g' ones (one row per event: a2_parse_hyp.py)
    loc = pd.read_csv(RUNS / a.run / "locations.csv")
    assert loc.ID.is_unique, "locations.csv: several rows per event (rerun a2_parse_hyp.py)"
    deep_run = check_deep_pass(a.run)
    if "pass" in loc and (loc["pass"] == "deep").any():
        runs_deep = set(loc.loc[loc["pass"] == "deep", "run"])
        if runs_deep != {deep_run}:
            raise SystemExit(f"deep-pass rows of locations.csv come from {sorted(runs_deep)}, but a2_parse_hyp.py "
                             f"checked {deep_run!r}")
    loc = loc[loc.status == "LOCATED"].set_index("ID")
    v4 = pd.read_csv(V4, skipinitialspace=True)
    v4.columns = [x.strip() for x in v4.columns]
    v4 = v4.set_index("ID")
    out_list = sorted(set(loc.index) - set(v4.index))          # located, but not in the v4.1 event list
    assert set(out_list) <= set(CV["v4_1_removed"]), f"located events missing from the v4.1 list: {out_list[:10]}"
    assert set(c.ID) == set(v4.index), "the Track A catalog is not the v4.1 event list (rerun a2_build_catalog.py)"
    loc = loc.drop(index=out_list)
    assert set(loc.index) == set(c.ID[nll]), "locations.csv and the Track A catalog disagree on the located events"
    for col in ("ns_z", "run", "tile", "pass"):
        assert col in loc and loc[col].notna().all(), f"locations.csv: {col} missing (rerun a2_parse_hyp.py, v5.1)"
    V = (("lat", "latitude"), ("lon", "longitude"), ("depth", "depth(km)"))
    for m, col, src in ([(nll, k, loc[k]) for k in ("lat", "lon", "depth", "exp_depth")] +
                        [(np.ones(len(c), bool), f"v4_{k}", v4[s]) for k, s in V] +
                        [(c.loc_source == "v4", k, v4[s]) for k, s in V]):
        new, old = src.reindex(c.ID[m]).to_numpy(float), c.loc[m, col].to_numpy(float)
        # more than the 5th digit apart (or missing) = run and catalog do not belong together, not rounding
        assert np.all(np.abs(new - old) <= 1e-4 * np.abs(old) + 1e-6), (col, np.nanmax(np.abs(new - old)))
        c.loc[m, col] = new
    # v4.1 additions NLLoc did not locate: left out (catalog_version.py adds them with the ComCat hypocentre)
    drop_add = c.ID.isin(CV["v4_1_added"]) & ~nll
    c = c[~drop_add].reset_index(drop=True)
    nll = c.loc_source == "nll3d"
    pos = c[["lat", "lon", "depth"]].notna().all(axis=1)           # rows with a hypocentre
    assert (pos == (c.loc_source != "none")).all(), "hypocentre present/absent does not match loc_source none"
    assert c.loc[~pos, "time"].isna().all(), "rows without a hypocentre carry an origin time"
    li = loc.reindex(c.ID[nll])                                    # locations.csv rows of the located events
    for col in ("depth", "exp_depth"):
        c.loc[nll, col] = unflatten(c.loc[nll, col])
    # D-B055 / D-B068: report events above the solid surface (docstring); depths are not changed
    src_surf = surface_source()
    h_fl, floor_file = floor_height(li)                            # km above the NLLoc floor, run frame
    floors = sorted(set(floor_file))
    # one floor for every row (main and deep runs), and with the merged surface exactly the configured floor file
    if len(floors) != 1:
        raise SystemExit(f"the located events used {len(floors)} NLLoc floor files {floors}: one build, one floor")
    want = ((B4.get("surface") or {}).get("floor") or {}).get("file")
    if SURFACE == "merged" and (not want or Path(floors[0]).resolve() != Path(want).resolve()):
        raise SystemExit(f"surface 'merged' needs the run's NLLoc floor to be params b4.surface.floor.file ({want}); "
                         f"the run used {floors[0]}")
    if SURFACE != "merged" and "merged" in Path(floors[0]).name:
        raise SystemExit(f"surface {SURFACE!r} ({src_surf}) does not match the run's NLLoc floor {floors} (D-B068: one "
                         "surface for both); set params b4.surface.source")
    c["surface_km"] = np.nan
    c.loc[pos, "surface_km"] = surface_km(c.lat[pos], c.lon[pos])
    c["above_surface"] = pd.array(np.where(pos, c.depth < -c.surface_km, False), dtype="boolean")
    c.loc[~pos, "above_surface"] = pd.NA
    c.loc[nll, "shift_z_km"] = c.loc[nll, "depth"] - c.loc[nll, "v4_depth"]
    c["depth_frame"] = np.where(nll, "true depth (EFT undone, b4_catalog.py)",
                                np.where(c.loc_source == "v4", "v4", ""))
    c["depth_quality"], c["event_flag"] = flags(c)
    c.loc[~pos, "depth_quality"] = ""
    # v5.1 flags (docstring): NLLoc-located rows only
    au = arrivals_used(li.reset_index())
    flag = {"floor_limited": (h_fl < FLOOR_CELLS * li.ns_z).to_numpy(),
            "far_only": (au.n_fine == 0).to_numpy(),
            "deep_pass": (li["pass"] == "deep").to_numpy(),
            "five_phase": (li.nphs == 5).to_numpy()}
    assert h_fl.notna().all(), f"{int(h_fl.isna().sum())} located events outside their floor grid"
    # above the floor file: 0 when the run used that file (157,257 + 157,269 v0.3.1 events: 0); NLLoc tests the exact
    # x, y, the lat/lon here are printed to 1e-6 deg, so a node-boundary case could give a few
    n_above_fl = int((h_fl < 0).sum())
    if n_above_fl:
        print(f"WARNING: {n_above_fl} located events above the NLLoc floor file of their run (expected 0)")
    for k, v in flag.items():
        c[k] = pd.array([pd.NA] * len(c), dtype="boolean")
        c.loc[nll, k] = v
    out = CAT / f"catalog_b4_{a.run}.csv"
    c.to_csv(out, index=False, float_format="%.8g")   # D-B013: %.5g keeps lon to 0.01 deg (0.4 km)
    fl1 = int((h_fl < li.ns_z).sum())
    r = pd.read_csv(CAT / f"catalog_v5_trackA_{a.ref}.csv").set_index("ID")
    c = c.set_index("ID")
    both = c.index[(c.loc_source == "nll3d")].intersection(r.index[r.loc_source == "nll3d"])
    cb, rb = c.loc[both], r.loc[both]
    d = pd.DataFrame({"region": cb.region, "tier_b4": cb.tier, "tier_ref": rb.tier,
                      "dz": cb.depth - rb.depth, "dh": km_h(cb, rb),
                      "dt": (pd.to_datetime(cb.time) - pd.to_datetime(rb.time)).dt.total_seconds(),
                      "rms_b4": cb.rms, "rms_ref": rb.rms, "above_b4": cb.above_surface, "above_ref": rb.above_surface})
    od = REPO / "data" / "tomo_v1" / "b4" / a.run
    od.mkdir(parents=True, exist_ok=True)
    g = d.groupby("region").agg(n=("dz", "size"), dz_med=("dz", "median"), absdz_med=("dz", lambda x: x.abs().median()),
                                absdz_p90=("dz", lambda x: x.abs().quantile(0.9)), dh_med=("dh", "median"),
                                dh_p90=("dh", lambda x: x.quantile(0.9)), dt_med=("dt", "median"),
                                rms_b4=("rms_b4", "median"), rms_ref=("rms_ref", "median"))
    g.loc["all"] = [len(d), d.dz.median(), d.dz.abs().median(), d.dz.abs().quantile(0.9), d.dh.median(),
                    d.dh.quantile(0.9), d.dt.median(), d.rms_b4.median(), d.rms_ref.median()]
    g.round(3).to_csv(od / "compare_by_region.csv")
    tiers = pd.DataFrame({"b4": c.tier.value_counts(), "ref": r.tier.value_counts()}).fillna(0).astype(int)
    L = [f"# B4 relocation {a.run} against {a.ref}", "",
         f"Catalog: {out} ({len(c)} events, {int(nll.sum())} located in 3D). Depths in km below sea level, "
         "EFT undone. Differences B4 minus reference, events located in both.", "",
         "## Tiers", "", "| tier | B4 | reference |", "|---|---|---|"]
    L += [f"| {t} | {row.b4} | {row.ref} |" for t, row in tiers.sort_index().iterrows()]
    L += ["", "## By region", "", "| region | n | dz med | abs dz med / p90 | dh med / p90 | dt med s | rms B4 / ref |",
          "|---|---|---|---|---|---|---|"]
    L += [f"| {k} | {int(v.n)} | {v.dz_med:+.2f} | {v.absdz_med:.2f} / {v.absdz_p90:.2f} | {v.dh_med:.2f} / "
          f"{v.dh_p90:.2f} | {v.dt_med:+.3f} | {v.rms_b4:.3f} / {v.rms_ref:.3f} |" for k, v in g.iterrows()]
    ab = c.above_surface.fillna(False).astype(bool)
    na, nu = int(ab.sum()), int((ab & (c.loc_source != "nll3d")).sum())
    pub = na == 0
    L += ["", f"Above the surface (located in both): B4 {100 * d.above_b4.astype(float).mean():.1f} %, "
          f"reference {100 * d.above_ref.astype(float).mean():.1f} % (reference: a2_parse_hyp.py check, no EFT).",
          f"D-B055: {na} events above the surface ({na - nu} located in 3D, {nu} with a hypocentre from elsewhere, "
          f"loc_source v4). Surface: {src_surf}. "
          + ("Publishable." if pub else "NOT PUBLISHABLE: these are computation errors to fix, not to move."),
          "The ANSS check per tier is in the tier summary of a2_build_catalog.py (copied next to this file)."]
    fc = pd.DataFrame({k: c[k].fillna(False).astype(bool) for k in flag}).assign(above_surface=ab)
    ft = fc.groupby(c.region).sum().assign(located=(c.loc_source == "nll3d").groupby(c.region).sum(),
                                           events=c.region.value_counts())
    ft.loc["all"] = ft.sum()
    ft = ft[["events", "located"] + list(fc.columns)]
    L += ["", "## v5.1 flags by region (NLLoc-located events)", "",
          f"floor_limited: maximum likelihood within {FLOOR_CELLS} final oct-tree cells (ns_z) of the NLLoc floor of "
          f"the run ({', '.join(floors)}; nearest node as NLLoc reads it, run frame); {fl1} within 1 cell; "
          f"{n_above_fl} located events above that floor file (0 expected: NLLoc rejects them). far_only: "
          "n_fine == 0 in the main-pass row of arrivals_used.csv; deep_pass: hypocentre from the deep pass; "
          "five_phase: located with 5 phases.", "",
          "| region | " + " | ".join(ft.columns) + " |", "|---" * (len(ft.columns) + 1) + "|"]
    L += [f"| {k} | " + " | ".join(str(int(v)) for v in row) + " |" for k, row in ft.iterrows()]
    ur = c.loc[c.loc_source == "none", "u_reason"] if "u_reason" in c else pd.Series(dtype=str)
    ur = ur.fillna("").str.replace(r"rejected: .*boundary (\d+).*", r"rejected: max likelihood on box face \1",
                                   regex=True).value_counts()
    L += ["", f"Not located (loc_source none, tier U; {int(drop_add.sum())} v4.1 additions left out for "
          "catalog_version.py): u_reason", "", "| u_reason | n |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in ur.items()]
    fq = pd.crosstab(c.region, c.depth_quality.replace("", "(no hypocentre)"), margins=True, margins_name="all")
    nq = int(c.event_flag.str.contains("quarry_blast_likely", regex=False).sum())
    na_ = {k: int(c.event_flag.str.contains(k, regex=False).sum()) for k in
           ("anss_explosion", "anss_quarry_blast", "anss_other_event", "anss_sonic_boom")}
    L += ["", "## Flags (rule 4 of D-B049..D-B051)", "", "| region | " + " | ".join(fq.columns) + " |",
          "|---" * (len(fq.columns) + 1) + "|"]
    L += [f"| {k} | " + " | ".join(str(v) for v in row) + " |" for k, row in fq.iterrows()]
    L += ["", f"event_flag quarry_blast_likely: {nq} events ({QB['name']} box, weekdays "
          f"{QB['weekday_hours_local'][0]:02d}:00-{QB['weekday_hours_local'][1]:02d}:00 {QB['tz']}).",
          "event_flag ANSS non-earthquakes (D-B071, within 3 s and 20 km): " + ", ".join(f"{k} {v}" for k, v in na_.items()) + "."]
    (od / "compare.md").write_text("\n".join(L) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    regs = list(g.index[:-1])
    fig, ax = plt.subplots(1, len(regs), figsize=(2.6 * len(regs), 3.4), sharey=True, squeeze=False)
    for i, rg in enumerate(regs):
        s = d.region == rg
        bins = np.arange(-5, 61, 2)
        ax[0, i].hist(rb.depth[s.values], bins=bins, orientation="horizontal", histtype="step", color="0.5", label="ref")
        ax[0, i].hist(cb.depth[s.values], bins=bins, orientation="horizontal", histtype="step", color="C3", label="B4")
        ax[0, i].set_title(f"{rg} ({int(s.sum())})", fontsize=8)
        ax[0, i].tick_params(labelsize=7)
    ax[0, 0].invert_yaxis()
    ax[0, 0].set_ylabel("depth (km bsl)")
    ax[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(od / "depth_hist.png", dpi=130)
    write_manifest(f"b4_catalog_{a.run}", od, inputs=[CAT / f"catalog_v5_trackA_{a.run}.csv",
                                                      CAT / f"catalog_v5_trackA_{a.ref}.csv",
                                                      RUNS / a.run / "locations.csv", V4, *floors,
                                                      *[RUNS / r / "arrivals_used.csv" for r in sorted(set(li.run))]],
                   outputs=[out, od / "compare.md", od / "compare_by_region.csv", od / "depth_hist.png"],
                   params={"depth_flags": DF, "quarry": QB, "surface": SURFACE, "surface_source": src_surf,
                           "floor_files": floors, "floor_cells": FLOOR_CELLS, "runs": sorted(set(li.run))},
                   counts={"events": len(c), "nll3d": int(nll.sum()), "both": len(both),
                           "none": int((c.loc_source == "none").sum()), "v4_1_added_left_out": int(drop_add.sum()),
                           "depth_quality": {k: int(v) for k, v in c.depth_quality.value_counts().items()},
                           "quarry_blast_likely": nq, "above_surface": na, "above_surface_not_nll3d": nu,
                           "flags": {k: int(fc[k].sum()) for k in flag}, "floor_within_1_cell": fl1,
                           "above_floor_file": n_above_fl,
                           "u_reason": {k: int(v) for k, v in ur.items()}, "publishable": pub})
    print((od / "compare.md").read_text())


if __name__ == "__main__":
    main()

"""Catalog versions v5.x (D-B058): the catalog v4 events and picks (v4.1 event list) with the hypocentres of a
B4 NonLinLoc relocation, Route A magnitudes, flags and a README, bundled for the private Zenodo draft 23084158.

  python catalog_version.py 5.0               # build /wd1/mdenolle_data/catalog_v5/v5.0 and catalog_v5.zip
  python catalog_version.py 5.0 --test DIR    # same build under DIR (bundle/, repo/): nothing else is written

Inputs (params catalog_versions): the B4 catalog of the version's run (b4_catalog.py: catalog_b4_<run>.csv, every
event of its event list, full-precision coordinates, true depths, surface check, flags), the catalog v4.1 event list
(v4.0 minus 172113, plus 900001-900003 with ComCat hypocentres), and the v4.1 Route A magnitudes.
Rules:
* event list = v4.1. Events of v4_1_removed still in the B4 catalog are dropped; events of v4_1_added missing from
  it are added with their ComCat hypocentre (loc_source comcat, tier U), surface check and flags as b4_catalog.py.
* tier U (params versions.<v>.tier_u): "v4" keeps the v4 hypocentre, flagged (D-B059, v5.0 only); "none" (default)
  means no hypocentre (loc_source none, empty time/lat/lon/depth; the B4 catalog's u_reason says why). Rows with a
  hypocentre must carry the surface check; nothing is moved (D-B055): above_surface is reported and the version is
  "publishable: false" while any event is above the surface.
* columns: every column of the B4 catalog is carried; the build stops on a column the README does not describe
  (COLUMNS below, plus the version's params `columns`, which add or override descriptions). Boolean flag columns
  listed in the version's params `flags` are counted in the README.
* picks (params versions.<v>.picks_file, picks_md5, picks_note): when a version was located with a pick file that is
  not in the Zenodo record (v5.1: the re-timed v4.1t picks, D-B072/D-B075), the file is checked against its md5 and
  bundled gzipped (header time 0, so the same bytes at each build); picks_note replaces the README's picks sentence.
* surface (params versions.<v>.surface_source, b4_catalog.py's b4.surface.source values): the solid surface of the
  version's B4 run, also used for the ComCat rows (v5.0 earth_relief_01m, v5.1 merged, D-B068).
* extra_files: repository paths or absolute paths, or {path, name} to copy under another name.
* magnitudes = the v4.1 Route A table (M, M_type, ML, ...), computed with the v4 hypocentres (STATUS step 9 redoes
  them with the new hypocentres). Ian's v4 magnitude column is not carried (runs +0.3 high offshore).
Outputs: <out>/v<version>/ (catalog CSV '%.8g', README, station terms of the run, B4 comparison, MD5SUMS),
<out>/catalog_v5.zip (every version built so far, replaced at each version), registry
data/catalog_v5/registry.csv and data/catalog_v5/v<version>/ (README copy, summary, manifest) in the repository.
"""
import argparse
import gzip
import hashlib
import shutil
import sys
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import REPO, load_params, write_manifest  # noqa: E402
import b4_catalog  # noqa: E402
from b4_catalog import CAT, RUNS, flags, surface_km  # noqa: E402

P = load_params()["catalog_versions"]
OUT = Path(P["out"])
TA = Path(P["track_a_code"])
sys.path.insert(0, str(TA))
from a2_parse_hyp import region  # noqa: E402

MAG_COLS = ["M", "M_unc", "M_type", "ML", "ML_unc", "n_ML", "ML_method", "Mw_mt", "comcat_id"]
# README column table (a key may name several columns, ", "-separated); loc_source is written from LOC_SOURCE
COLUMNS = {
    "ID": "catalog v4 event ID (900001-900003: v4.1 additions)",
    "raw_ID": "ID in the GENIE association output",
    "time": "origin time (UTC) of this version's hypocentre",
    "lat, lon": "epicentre (deg, WGS84)",
    "depth": "km below sea level (negative = above sea level)",
    "exp_depth": "NonLinLoc expectation depth (km bsl)",
    "loc_source": None,
    "tier": "location quality A (best) to D, U = not located here (rules in b4_catalog_v5_trackA_*_tiers.md)",
    "nphs, n_sta, gap, rms": "phases and stations used, largest azimuthal gap (deg), rms residual (s)",
    "h_min, h_max, sig_z, ell_len3": "68 % horizontal semi-axes, depth standard error, longest ellipsoid axis (km)",
    "dist_min, dist_med, dist_max, dist_over_depth": "epicentral distance to the stations (km); nearest / depth",
    "surface_km": "ground or seafloor elevation at the epicentre (km, positive up; GMT earth_relief 01m, bilinear)",
    "above_surface": "depth above the solid surface: a location error, kept visible, not moved (D-B055)",
    "depth_quality": "ok, unconstrained (ridge regions, Blanco outside the 2012-13 OBS deployment), biased_deep "
                     "(Gorda, nearest station > 15 km)",
    "event_flag": "quarry_blast_likely (Texada Island box, weekday daytime) or empty",
    "shift_h_km, shift_z_km": "change from the v4 hypocentre",
    "tile, region": "NonLinLoc tile; region label (box rules of a2_parse_hyp.py)",
    "depth_frame": "how depth was obtained",
    "v4_time, v4_lat, v4_lon, v4_depth, v4_npick, v4_sigma_m": "catalog v4 origin, hypocentre, picks, GraphDD "
                                                                "spatial sigma (m)",
    ", ".join(MAG_COLS): "Route A magnitudes of catalog v4.1",
}
# loc_source values: README column text, counts-table label
LOC_SOURCE = {"nll3d": ("located here", "located here (loc_source nll3d)"),
              "v4": ("v4 hypocentre kept, tier U", "v4 hypocentre kept (loc_source v4, tier U)"),
              "comcat": ("ComCat hypocentre", "ComCat hypocentre (loc_source comcat, v4.1 additions)"),
              "none": ("not located: no hypocentre, see u_reason", "not located (loc_source none, tier U)")}
TIER_U_TEXT = {
    "v4": """* Events that NonLinLoc could not locate (tier U: too few picks or no solution inside the grids) keep the v4
  hypocentre (loc_source v4). The v4 depths come from a location without topography or water and are often above the
  surface; they are kept, with the above_surface flag, for this version only (D-B059).""",
    "none": """* Events that NonLinLoc could not locate (tier U) have no hypocentre in this version (loc_source none: empty
  time, lat, lon and depth); u_reason says why. Their picks are in the picks file.""",
}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build(version, out=OUT, rd=REPO / "data" / "catalog_v5"):
    V = P["versions"][version]
    run, model = V["b4_run"], V["model"]
    tier_u = V.get("tier_u", "none")
    flag_cols = V.get("flags", [])
    b4_catalog.SURFACE = V["surface_source"]       # surface_km reads it at each call
    src = CAT / f"catalog_b4_{run}.csv"
    c = pd.read_csv(src, dtype={"tier": str})
    # rows without a hypocentre are exactly the loc_source 'none' rows; every other row carries the surface check
    pos = c[["lat", "lon", "depth"]].notna().all(axis=1)
    nopos = c[["time", "lat", "lon", "depth"]].isna().all(axis=1)
    none = c.loc_source == "none"
    assert (pos == ~none).all() and nopos[none].all(), f"{src}: hypocentre present/absent does not match loc_source none"
    assert c.loc[pos, ["surface_km", "above_surface"]].notna().all().all(), f"{src}: surface check missing on some rows"
    assert {"v4": "none", "none": "v4"}[tier_u] not in set(c.loc_source), f"{src}: tier U rows do not follow tier_u {tier_u}"
    c["above_surface"] = c.above_surface.astype("boolean")
    v41 = pd.read_csv(P["v4_1_events"], skipinitialspace=True)
    v41.columns = [x.strip() for x in v41.columns]
    drop = sorted(set(c.ID) - set(v41.ID))
    add = sorted(set(v41.ID) - set(c.ID))
    assert set(drop) <= set(P["v4_1_removed"]) and set(add) <= set(P["v4_1_added"]), (drop, add)
    c = c[~c.ID.isin(drop)].copy()
    if add:  # v4.1 additions: ComCat hypocentre (utils/fix_catalog_v4.py); same columns as the B4 rows
        a = v41.set_index("ID").loc[add]
        t = pd.to_datetime(a["datetime"].str.strip().str.rstrip("Z")).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        n = pd.DataFrame({"ID": add, "raw_ID": np.nan, "v4_time": t.values, "v4_lat": a["latitude"].values,
                          "v4_lon": a["longitude"].values, "v4_depth": a["depth(km)"].values,
                          "v4_npick": (a["Num_p_picks"] + a["Num_s_picks"]).values,
                          "v4_sigma_m": a["spatial_sigma(m)"].values, "time": t.values, "lat": a["latitude"].values,
                          "lon": a["longitude"].values, "depth": a["depth(km)"].values, "loc_source": "comcat",
                          "tier": "U", "depth_frame": "ComCat"})
        n["region"] = [region(x, y) for x, y in zip(n.lat, n.lon)]
        n["surface_km"] = surface_km(n.lat, n.lon)
        n["above_surface"] = (n.depth < -n.surface_km).astype("boolean")
        n["dist_min"] = np.nan
        n["depth_quality"], n["event_flag"] = flags(n)
        c = pd.concat([c, n], ignore_index=True)
    # magnitudes: v4.1 Route A table (converged, D-B091; its md5 is checked); Ian's v4 column is dropped
    if P.get("v4_1_magnitudes_md5") and md5(P["v4_1_magnitudes"]) != P["v4_1_magnitudes_md5"]:
        raise SystemExit(f"{P['v4_1_magnitudes']}: md5 differs from params v4_1_magnitudes_md5; refusing to build the bundle")
    m = pd.read_csv(P["v4_1_magnitudes"]).set_index("event_id")
    for k in MAG_COLS:
        c[k] = c.ID.map(m[k])
    c = c.drop(columns=["v4_magnitude"], errors="ignore").sort_values("ID").reset_index(drop=True)
    assert c.ID.is_unique and len(c) == len(v41), (len(c), len(v41))
    # every column described in the README
    cols = {**COLUMNS, **V.get("columns", {})}
    ls = c.loc_source.value_counts().to_dict()
    cols["loc_source"] = ", ".join(f"{k} ({LOC_SOURCE[k][0]})" for k in LOC_SOURCE if k in ls)
    assert set(ls) <= set(LOC_SOURCE), set(ls) - set(LOC_SOURCE)
    described = {x for k in cols for x in k.split(", ")}
    undoc = [x for x in c.columns if x not in described]
    assert not undoc, f"columns without a README description (params catalog_versions.versions.{version}.columns): {undoc}"
    assert set(flag_cols) <= set(c.columns), set(flag_cols) - set(c.columns)

    vd = out / f"v{version}"
    vd.mkdir(parents=True, exist_ok=True)
    csv = vd / f"cascadia_catalog_v{version}.csv"
    c.to_csv(csv, index=False, float_format="%.8g")
    terms = RUNS / V["terms_run"] / "station_terms.in"
    shutil.copy(terms, vd / f"station_terms_v{version}.in")
    b4d = REPO / "data" / "tomo_v1" / "b4" / run
    for f in ("compare.md", "acceptance.md", f"catalog_v5_trackA_{run}_tiers.md"):
        if (b4d / f).exists():
            shutil.copy(b4d / f, vd / f"b4_{f}")
    extra = []
    for f in V.get("extra_files", []):
        f = f if isinstance(f, dict) else {"path": f}
        extra.append(REPO / f["path"])                # an absolute path stays absolute
        shutil.copy(extra[-1], vd / f.get("name", Path(f["path"]).name))
    picks = Path(V["picks_file"]) if V.get("picks_file") else None
    if picks:
        assert md5(picks) == V["picks_md5"], f"{picks}: md5 differs from params picks_md5"
        with open(picks, "rb") as fi, open(vd / f"{picks.name}.gz", "wb") as raw, \
                gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as fo:
            shutil.copyfileobj(fi, fo, 1 << 24)

    above = c.above_surface.fillna(False).astype(bool)
    ab = c[above]
    pub = len(ab) == 0
    tiers = c.tier.value_counts().sort_index().to_dict()
    dq = c.depth_quality.value_counts().to_dict()
    ab_src = ab.loc_source.value_counts().to_dict()
    nflag = {f: int(c[f].astype("boolean").fillna(False).sum()) for f in flag_cols}
    nm = int(c.M.notna().sum())
    ax = c[(c.loc_source == "nll3d") & (c.region == "Axial") & c.tier.isin(["A", "B"])]
    axz = ax.depth + ax.surface_km
    axial = (f"{100 * (axz <= 2.0).mean():.0f} % of the {len(ax):,} tier A-B Axial events are within 2 km of the seafloor "
             f"(median {axz.median():.1f} km below it), against 90 % in Waldhauser et al. (2020)")
    dec = ", ".join(V.get("decisions", ["D-B058"]))
    picks_txt = V.get("picks_note", "Picks are not repeated\n  here: they are Cascadia_picks_catalog_v4.csv.gz in this "
                                    "record, with the v4.1 patch.")
    ls_rows = "".join(f"| {LOC_SOURCE[k][1]} | {ls.get(k, 0):,} |\n" for k in LOC_SOURCE if k == "nll3d" or ls.get(k, 0))
    flag_rows = "".join(f"| {f} | {v:,} |\n" for f, v in nflag.items())
    col_rows = "".join(f"| {k} | {v} |\n" for k, v in cols.items() if any(x in c.columns for x in k.split(", ")))
    readme = f"""# Cascadia OBS ensemble earthquake catalog v{version}

{V['note']}

Built {date.today()} by workflow/10_tomography/catalog_version.py (decisions {dec} in
workflow/10_tomography/DECISIONS.md). Private working copy; not peer reviewed; not for redistribution.

## What it is

* Events and picks: catalog v4.1 (association catalog of I. McBrearty with the v4.1 patch: event 172113 removed as
  mis-associated, events 900001-900003 added with ComCat hypocentres and repicked phases). {picks_txt}
* Hypocentres: NonLinLoc (EDT likelihood, oct-tree) in the 3D merged Vp/Vs model v{model}
  (velocity_models.zip in this record), with static station terms from a first pass (station_terms_v{version}.in),
  run {run}. Depth is km below sea level (true depth; the Earth-flattening of the travel-time grids is undone).
""" + "".join(f"* {x}\n" for x in V.get("method_notes", [])) + TIER_U_TEXT[tier_u] + f"""
* Magnitudes: Route A preferred magnitude of catalog v4.1 (M, M_type ML / Mw / Mw_cal), computed with the v4
  hypocentres{f" ({P['v4_1_magnitudes_note']})" if P.get("v4_1_magnitudes_note") else ""};
  they will be recomputed with the new hypocentres in a later version.

## Counts

| item | value |
|---|---|
| events | {len(c):,} |
{ls_rows}| tiers | {', '.join(f'{k} {v:,}' for k, v in tiers.items())} |
| depth_quality | {', '.join(f'{k} {v:,}' for k, v in dq.items())} |
{flag_rows}| with a magnitude | {nm:,} |
| above the ground or seafloor | {len(ab):,} ({', '.join(f'{k} {v:,}' for k, v in ab_src.items()) or 'none'}) |
| publishable (D-B055) | {'yes' if pub else 'no: events above the surface are computation errors still to fix'} |

## Columns

| column | meaning |
|---|---|
{col_rows}
## Known limits of this version

""" + "".join(f"* {x.format(axial=axial)}\n" for x in V.get("known_limits", []))
    (vd / f"README_v{version}.md").write_text(readme)
    files = sorted(p for p in vd.iterdir() if p.is_file() and p.name != "MD5SUMS")
    (vd / "MD5SUMS").write_text("".join(f"{md5(p)}  {p.name}\n" for p in files))

    # registry (repository) and the bundle with every version built so far
    reg_f = rd / "registry.csv"
    row = {"version": version, "date": str(date.today()), "b4_run": run, "model": model, "events": len(c),
           **{k: ls.get(k, 0) for k in LOC_SOURCE}, "above_surface": len(ab), "publishable": pub,
           "csv_md5": md5(csv), "note": V["note"]}
    reg = pd.read_csv(reg_f, dtype=str) if reg_f.exists() else pd.DataFrame(columns=list(row))
    reg = pd.concat([reg[reg.version != version], pd.DataFrame([row]).astype(str)], ignore_index=True)
    rd.mkdir(parents=True, exist_ok=True)
    reg.sort_values("version").to_csv(reg_f, index=False)
    zp = out / "catalog_v5.zip"
    tmp = zp.with_suffix(".zip.part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(reg_f, "catalog_v5/registry.csv")
        for v in sorted(reg.version):
            for p in sorted((out / f"v{v}").iterdir()):
                z.write(p, f"catalog_v5/v{v}/{p.name}")
    tmp.replace(zp)

    od = rd / f"v{version}"
    od.mkdir(parents=True, exist_ok=True)
    shutil.copy(vd / f"README_v{version}.md", od)
    shutil.copy(vd / "MD5SUMS", od)
    reg_lab = c.region.fillna("(no position)")
    s = pd.crosstab(reg_lab, c.loc_source, margins=True, margins_name="all")
    s["above_surface"] = above.groupby(reg_lab).sum().reindex(s.index).fillna(len(ab)).astype(int)
    tab = ["| region | " + " | ".join(map(str, s.columns)) + " |", "|---" * (len(s.columns) + 1) + "|"]
    tab += [f"| {k} | " + " | ".join(f"{v:,}" for v in row) + " |" for k, row in s.iterrows()]
    (od / "summary.md").write_text(f"# Catalog v{version}: events by region and hypocentre source\n\n" + "\n".join(tab)
                                   + f"\n\nBundle {zp} md5 {md5(zp)} ({zp.stat().st_size:,} bytes)\n")
    write_manifest(f"catalog_v{version}", od,
                   inputs=[src, P["v4_1_events"], P["v4_1_magnitudes"], terms, *extra, *([picks] if picks else [])],
                   outputs=[csv, vd / f"README_v{version}.md", vd / "MD5SUMS", zp, reg_f],
                   params={"version": version, **V, "v4_1_removed": drop, "v4_1_added": add},
                   counts={"events": len(c), "loc_source": ls, "tiers": tiers, "above_surface": len(ab),
                           "above_surface_by_source": ab_src, "flags": nflag, "with_magnitude": nm,
                           "publishable": pub})
    print(readme.split("## Columns")[0])
    print(f"bundle {zp} md5 {md5(zp)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("version")
    ap.add_argument("--test", type=Path, help="build under this directory only (bundle/, repo/); for checks")
    a = ap.parse_args()
    if a.test:
        build(a.version, out=a.test / "bundle", rd=a.test / "repo")
    else:
        build(a.version)

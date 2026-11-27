"""Catalog versions v5.x, v6.x (D-B058, D-B079): the catalog v4 events and picks (v4.1 event list) with the hypocentres
of a NonLinLoc relocation, Route A magnitudes, flags and a README, bundled for the private Zenodo draft 23084158.

  python catalog_version.py 5.0 5.1           # build the versions' folders, then catalog_v5.zip once (all versions of
                                              # the series in the registry) and each version's summary and manifest
  python catalog_version.py 6.0               # /wd1/mdenolle_data/catalog_v6/v6.0 and catalog_v6.zip (series v6)
  python catalog_version.py 5.1 --test DIR    # same build under DIR (bundle/, repo/): nothing else is written

Inputs (params catalog_versions): the B4 catalog of the version's run (b4_catalog.py: catalog_b4_<run>.csv, every
event of its event list, full-precision coordinates, true depths, surface check, flags), the catalog v4.1 event list
(v4.0 minus 172113, plus 900001-900003 with ComCat hypocentres), and the converged Route A magnitude tables (D-B091):
v4.1 (v4 hypocentres) and v5.1 (v5.1 hypocentres), each md5-checked. Per version (params versions.<v>): `series`
(bundle catalog_<series>.zip in `out`, registry data/catalog_<series>/), `b4_catalog` and `terms_file` (input paths,
default the B4 run's), `columns_from` (inherit another version's column descriptions), README paragraphs
`hypocentres_text`, `magnitudes_text`.
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
* magnitudes (params versions.<v>.magnitude_tables, default [v4.1]): per event the first table with a magnitude, all
  magnitude columns from it; with more than one table the column magnitude_source names it (v5.1: [v5.1, v4.1],
  D-B102). `magnitudes: catalog` (v6.0): the catalog brings its own magnitude columns and magnitude_source (checked
  present); only the v4.1 additions it leaves out take the tables. Ian's v4 magnitude column is not carried.
Outputs: <out>/v<version>/ (catalog CSV '%.8g', README, station terms of the run, B4 comparison, MD5SUMS),
<out>/catalog_<series>.zip (every version of the series in its registry), registry data/catalog_<series>/registry.csv
and data/catalog_<series>/v<version>/ (README copy, summary, manifest, both with the final bundle md5).
"""
import argparse
import functools
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
# converged Route A magnitude tables (D-B091): path, md5, key column, magnitude_source label (short and machine-readable;
# the README column description says what each label means)
MAG_TABLES = {"v4.1": dict(path=P["v4_1_magnitudes"], md5=P.get("v4_1_magnitudes_md5"), key="event_id",
                           label="v4.1_converged"),
              "v5.1": dict(path=P.get("v5_1_magnitudes"), md5=P.get("v5_1_magnitudes_md5"), key="ID",
                           label="v5.1_converged"),
              "v6.1": dict(path=P.get("v6_1_magnitudes"), md5=P.get("v6_1_magnitudes_md5"), key="ID",
                           label="v6.1_converged")}
# long labels written by b4_v6_catalog.py (catalog v6.0 input) -> the short labels above
MAG_LABEL_LONG = {"v4.1_converged (D-B091, v4 hypocentres)": "v4.1_converged",
                  "v5.1_converged (D-B091, v5.1 hypocentres)": "v5.1_converged"}


def version_columns(version, seen=()):
    """README column descriptions of a version: those of its columns_from chain (oldest first), then its own."""
    V = P["versions"][version]
    need(version not in seen, f"columns_from loop: {seen + (version,)}")
    base = version_columns(V["columns_from"], seen + (version,)) if V.get("columns_from") else {}
    return {**base, **V.get("columns", {})}


def need(ok, msg):
    """Integrity check that also runs under python -O (unlike assert)."""
    if not ok:
        raise SystemExit(msg)


@functools.lru_cache(maxsize=None)
def mag_table(name):
    """The md5-checked table, read once per process (callers must not modify it)."""
    t = MAG_TABLES[name]
    if not t["md5"] or md5(t["path"]) != t["md5"]:
        raise SystemExit(f"magnitude table {name} ({t['path']}): md5 missing or different from params; refusing to build")
    return pd.read_csv(t["path"], low_memory=False).set_index(t["key"])


def assign_magnitudes(c, rows, names, source):
    """Magnitude columns of the rows (bool array) from the first table of `names` that has a magnitude (M) for the
    event; with `source`, the column magnitude_source names that table (empty without a magnitude)."""
    got = np.zeros(len(c), bool)
    lab = np.full(len(c), np.nan, dtype=object)
    for name in names:
        m = mag_table(name)
        take = rows & ~got & c.ID.map(m["M"]).notna().values
        for k in MAG_COLS:
            c.loc[take, k] = c.ID[take].map(m[k]).values
        lab[take] = MAG_TABLES[name]["label"]
        got |= take
    for k in MAG_COLS:                              # no magnitude in any table: every magnitude column empty
        c.loc[rows & ~got, k] = np.nan
    if source:
        c.loc[rows, "magnitude_source"] = lab[rows]
    return [MAG_TABLES[n]["path"] for n in names]
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


def build(version, out=None, rd=None):
    V = P["versions"][version]
    run, model = V["b4_run"], V["model"]
    series = V.get("series", "v5")                 # bundle series: catalog_v5.zip (v5.x), catalog_v6.zip (v6.x), ...
    out = out or Path(V.get("out", P["out"]))
    rd = rd or REPO / "data" / f"catalog_{series}"
    tier_u = V.get("tier_u", "none")
    flag_cols = V.get("flags", [])
    b4_catalog.SURFACE = V["surface_source"]       # surface_km reads it at each call
    src = Path(V["b4_catalog"]) if V.get("b4_catalog") else CAT / f"catalog_b4_{run}.csv"
    c = pd.read_csv(src, dtype={"tier": str})
    # rows without a hypocentre are exactly the loc_source 'none' rows; every other row carries the surface check
    pos = c[["lat", "lon", "depth"]].notna().all(axis=1)
    nopos = c[["time", "lat", "lon", "depth"]].isna().all(axis=1)
    none = c.loc_source == "none"
    need((pos == ~none).all() and nopos[none].all(), f"{src}: hypocentre present/absent does not match loc_source none")
    need(c.loc[pos, ["surface_km", "above_surface"]].notna().all().all(), f"{src}: surface check missing on some rows")
    need({"v4": "none", "none": "v4"}[tier_u] not in set(c.loc_source), f"{src}: tier U rows do not follow tier_u {tier_u}")
    c["above_surface"] = c.above_surface.astype("boolean")
    v41 = pd.read_csv(P["v4_1_events"], skipinitialspace=True)
    v41.columns = [x.strip() for x in v41.columns]
    drop = sorted(set(c.ID) - set(v41.ID))
    add = sorted(set(v41.ID) - set(c.ID))
    need(set(drop) <= set(P["v4_1_removed"]) and set(add) <= set(P["v4_1_added"]), f"removed {drop} / added {add} not in params v4_1_removed / v4_1_added")
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
    # magnitudes: converged Route A tables (D-B091, md5-checked); Ian's v4 column is dropped
    names = V.get("magnitude_tables", ["v4.1"])
    if V.get("magnitudes") == "catalog":           # the catalog brings its own magnitudes and their source; only the
        missing = [k for k in MAG_COLS + ["magnitude_source"] if k not in c]       # v4.1 additions it leaves out take
        if missing:                                                                  # the tables
            raise SystemExit(f"{src}: magnitudes: catalog, but columns missing: {missing}")
        ms = c["magnitude_source"].replace(MAG_LABEL_LONG)                           # short labels
        bad = sorted(set(ms.dropna()) - {t["label"] for t in MAG_TABLES.values()})
        if bad:
            raise SystemExit(f"{src}: unknown magnitude_source labels {bad}")
        c["magnitude_source"] = ms
        mag_inputs = assign_magnitudes(c, c.ID.isin(add).values, names, source=True)
    else:
        mag_inputs = assign_magnitudes(c, np.ones(len(c), bool), names, source=len(names) > 1)
    c = c.drop(columns=["v4_magnitude"], errors="ignore").sort_values("ID").reset_index(drop=True)
    need(c.ID.is_unique and len(c) == len(v41), f"IDs not unique or {len(c)} rows against {len(v41)} catalog v4.1 events")
    # every column described in the README
    cols = {**COLUMNS, **version_columns(version)}
    ls = c.loc_source.value_counts().to_dict()
    cols["loc_source"] = ", ".join(f"{k} ({LOC_SOURCE[k][0]})" for k in LOC_SOURCE if k in ls)
    need(set(ls) <= set(LOC_SOURCE), f"unknown loc_source {set(ls) - set(LOC_SOURCE)}")
    described = {x for k in cols for x in k.split(", ")}
    undoc = [x for x in c.columns if x not in described]
    need(not undoc, f"columns without a README description (params catalog_versions.versions.{version}.columns): {undoc}")
    need(set(flag_cols) <= set(c.columns), f"flag columns missing: {set(flag_cols) - set(c.columns)}")

    vd = out / f"v{version}"
    vd.mkdir(parents=True, exist_ok=True)
    csv = vd / f"cascadia_catalog_v{version}.csv"
    c.to_csv(csv, index=False, float_format="%.8g")
    terms = Path(V["terms_file"]) if V.get("terms_file") else RUNS / V["terms_run"] / "station_terms.in"
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
        need(md5(picks) == V["picks_md5"], f"{picks}: md5 differs from params picks_md5")
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
    ax = c[(c.loc_source == "nll3d") & (c.region == "Axial") & c.tier.isin(["A", "B"])].dropna(subset=["depth", "surface_km"])
    axz = ax.depth + ax.surface_km
    axial = (f"{100 * (axz <= 2.0).mean():.0f} % of the {len(ax):,} tier A-B Axial events are within 2 km of the seafloor "
             f"(median {axz.median():.1f} km below it), against 90 % in Waldhauser et al. (2020)")
    dec = ", ".join(V.get("decisions", ["D-B058"]))
    hypo = V.get("hypocentres_text", "Hypocentres: NonLinLoc (EDT likelihood, oct-tree) in the 3D merged Vp/Vs model "
                 "v{model}\n  (velocity_models.zip in this record), with static station terms from a first pass "
                 "(station_terms_v{version}.in),\n  run {run}. Depth is km below sea level (true depth; the "
                 "Earth-flattening of the travel-time grids is undone).").format(model=model, version=version, run=run)
    mnote = f" ({P['v4_1_magnitudes_note']})" if P.get("v4_1_magnitudes_note") else ""
    mags = V.get("magnitudes_text", "Magnitudes: Route A preferred magnitude of catalog v4.1 (M, M_type ML / Mw / "
                 "Mw_cal), computed with the v4\n  hypocentres{mnote};\n  catalog v5.1 carries magnitudes "
                 "computed with its own hypocentres (D-B102).").format(mnote=mnote)
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
* {hypo}
""" + "".join(f"* {x}\n" for x in V.get("method_notes", [])) + TIER_U_TEXT[tier_u] + f"""
* {mags}

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
    reg_lab = c.region.fillna("(no position)")
    stab = pd.crosstab(reg_lab, c.loc_source, margins=True, margins_name="all")
    stab["above_surface"] = above.groupby(reg_lab).sum().reindex(stab.index).fillna(len(ab)).astype(int)
    print(readme.split("## Columns")[0])
    return dict(version=version, series=series, out=out, rd=rd, reg_f=reg_f, vd=vd, csv=csv, stab=stab,
                inputs=[src, P["v4_1_events"], *mag_inputs, terms, *extra, *([picks] if picks else [])],
                params={"version": version, **V, "v4_1_removed": drop, "v4_1_added": add},
                counts={"events": len(c), "loc_source": ls, "tiers": tiers, "above_surface": len(ab),
                        "above_surface_by_source": ab_src, "flags": nflag, "with_magnitude": nm, "publishable": pub})


def finalize(built):
    """catalog_<series>.zip with every version of the series' registry (built once, after the versions of this run),
    then each built version's repository folder: README and MD5SUMS copies, summary.md and manifest with the final
    bundle md5."""
    b0 = built[0]
    series, out, reg_f = b0["series"], b0["out"], b0["reg_f"]
    reg = pd.read_csv(reg_f, dtype=str)
    zp = out / f"catalog_{series}.zip"
    tmp = zp.with_suffix(".zip.part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(reg_f, f"catalog_{series}/registry.csv")
        for v in sorted(reg.version):
            for p in sorted((out / f"v{v}").iterdir()):
                z.write(p, f"catalog_{series}/v{v}/{p.name}")
    tmp.replace(zp)
    zmd5 = md5(zp)
    for b in built:
        version, vd, s = b["version"], b["vd"], b["stab"]
        od = b["rd"] / f"v{version}"
        od.mkdir(parents=True, exist_ok=True)
        shutil.copy(vd / f"README_v{version}.md", od)
        shutil.copy(vd / "MD5SUMS", od)
        tab = ["| region | " + " | ".join(map(str, s.columns)) + " |", "|---" * (len(s.columns) + 1) + "|"]
        tab += [f"| {k} | " + " | ".join(f"{v:,}" for v in row) + " |" for k, row in s.iterrows()]
        (od / "summary.md").write_text(f"# Catalog v{version}: events by region and hypocentre source\n\n"
                                       + "\n".join(tab) + f"\n\nBundle {zp} md5 {zmd5} ({zp.stat().st_size:,} bytes)\n")
        write_manifest(f"catalog_v{version}", od, inputs=b["inputs"],
                       outputs=[b["csv"], vd / f"README_v{version}.md", vd / "MD5SUMS", zp, reg_f],
                       params=b["params"], counts=b["counts"])
    print(f"bundle {zp} md5 {zmd5}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("versions", nargs="+")
    ap.add_argument("--test", type=Path, help="build under this directory only (bundle/, repo/); for checks")
    a = ap.parse_args()
    kw = dict(out=a.test / "bundle", rd=a.test / "repo") if a.test else {}
    built = [build(v, **kw) for v in a.versions]
    for series in dict.fromkeys(b["series"] for b in built):      # one bundle per series, after its versions
        finalize([b for b in built if b["series"] == series])

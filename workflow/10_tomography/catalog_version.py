"""Catalog versions v5.x (D-B058): the catalog v4 events and picks (v4.1 event list) with the hypocentres of a
B4 NonLinLoc relocation, Route A magnitudes, flags and a README, bundled for the private Zenodo draft 23084158.

  python catalog_version.py 5.0            # build /wd1/mdenolle_data/catalog_v5/v5.0 and catalog_v5.zip

Inputs (params catalog_versions): the B4 catalog of the version's run (b4_catalog.py: catalog_b4_<run>.csv, every
v4.0 event, full-precision coordinates, true depths, surface check, flags), the catalog v4.1 event list (v4.0 minus
172113, plus 900001-900003 with ComCat hypocentres), and the v4.1 Route A magnitudes.
Rules:
* event list = v4.1. 172113 (mis-associated in v4.0) is dropped; 900001-900003 are added with their ComCat
  hypocentre (loc_source comcat, tier U), surface check and flags as b4_catalog.py. 172114 keeps its B4 location,
  which used the v4.0 picks (v4.1 added 160 picks to it; noted in the README).
* tier U events keep their v4 hypocentre, flagged (D-B059, v5.0 only); nothing is moved (D-B055): above_surface is
  reported and the version is "publishable: false" while any event is above the surface.
* magnitudes = the v4.1 Route A table (M, M_type, ML, ...), computed with the v4 hypocentres (STATUS step 9 redoes
  them with the new hypocentres). Ian's v4 magnitude column is not carried (runs +0.3 high offshore).
Outputs: <out>/v<version>/ (catalog CSV '%.8g', README, station terms of the run, B4 comparison, MD5SUMS),
<out>/catalog_v5.zip (every version built so far, replaced at each version), registry
data/catalog_v5/registry.csv and data/catalog_v5/v<version>/ (README copy, summary, manifest) in the repository.
"""
import argparse
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
from b4_catalog import CAT, RUNS, flags, surface_km  # noqa: E402

P = load_params()["catalog_versions"]
OUT = Path(P["out"])
TA = Path(P["track_a_code"])
sys.path.insert(0, str(TA))
from a2_parse_hyp import region  # noqa: E402

MAG_COLS = ["M", "M_unc", "M_type", "ML", "ML_unc", "n_ML", "ML_method", "Mw_mt", "comcat_id"]


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build(version):
    V = P["versions"][version]
    run, model = V["b4_run"], V["model"]
    src = CAT / f"catalog_b4_{run}.csv"
    c = pd.read_csv(src, dtype={"tier": str})
    # the surface check must cover every row (b4_catalog.py since 11-24; older B4 catalogs skipped tier U)
    assert c.surface_km.notna().all() and c.above_surface.notna().all(), f"{src}: surface check missing on some rows"
    c["above_surface"] = c.above_surface.astype(bool)
    v41 =pd.read_csv(P["v4_1_events"], skipinitialspace=True)
    v41.columns = [x.strip() for x in v41.columns]
    drop = sorted(set(c.ID) - set(v41.ID))
    add = sorted(set(v41.ID) - set(c.ID))
    assert drop == P["v4_1_removed"] and add == P["v4_1_added"], (drop, add)
    c = c[~c.ID.isin(drop)].copy()
    # v4.1 additions: ComCat hypocentre (utils/fix_catalog_v4.py); same columns as the B4 rows, located by nobody here
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
    n["above_surface"] = n.depth < -n.surface_km
    n["dist_min"] = np.nan
    n["depth_quality"], n["event_flag"] = flags(n)
    c = pd.concat([c, n], ignore_index=True)
    # magnitudes: v4.1 Route A table; Ian's v4 column is dropped
    m = pd.read_csv(P["v4_1_magnitudes"]).set_index("event_id")
    for k in MAG_COLS:
        c[k] = c.ID.map(m[k])
    c = c.drop(columns=["v4_magnitude"]).sort_values("ID").reset_index(drop=True)
    assert c.ID.is_unique and len(c) == len(v41), (len(c), len(v41))

    vd = OUT / f"v{version}"
    vd.mkdir(parents=True, exist_ok=True)
    csv = vd / f"cascadia_catalog_v{version}.csv"
    c.to_csv(csv, index=False, float_format="%.8g")
    terms = RUNS / V["terms_run"] / "station_terms.in"
    shutil.copy(terms, vd / f"station_terms_v{version}.in")
    b4d = REPO / "data" / "tomo_v1" / "b4" / run
    for f in ("compare.md", "acceptance.md", f"catalog_v5_trackA_{run}_tiers.md"):
        if (b4d / f).exists():
            shutil.copy(b4d / f, vd / f"b4_{f}")

    ls = c.loc_source.value_counts().to_dict()
    ab = c[c.above_surface]
    pub = len(ab) == 0
    tiers = c.tier.value_counts().sort_index().to_dict()
    dq = c.depth_quality.value_counts().to_dict()
    ab_src = ab.loc_source.value_counts().to_dict()
    nm = int(c.M.notna().sum())
    ax = c[(c.loc_source == "nll3d") & (c.region == "Axial") & c.tier.isin(["A", "B"])]
    axz = ax.depth + ax.surface_km
    axial = (f"{100 * (axz <= 2.0).mean():.0f} % of the {len(ax):,} tier A-B Axial events are within 2 km of the seafloor "
             f"(median {axz.median():.1f} km below it), against 90 % in Waldhauser et al. (2020)")
    readme = f"""# Cascadia OBS ensemble earthquake catalog v{version}

{V['note']}

Built {date.today()} by workflow/10_tomography/catalog_version.py (decisions D-B058, D-B059 in
workflow/10_tomography/DECISIONS.md). Private working copy; not peer reviewed; not for redistribution.

## What it is

* Events and picks: catalog v4.1 (association catalog of I. McBrearty with the v4.1 patch: event 172113 removed as
  mis-associated, events 900001-900003 added with ComCat hypocentres and repicked phases). Picks are not repeated
  here: they are Cascadia_picks_catalog_v4.csv.gz in this record, with the v4.1 patch.
* Hypocentres: NonLinLoc (EDT likelihood, oct-tree) in the 3D merged Vp/Vs model v{model}
  (velocity_models.zip in this record), with static station terms from a first pass (station_terms_v{version}.in),
  run {run}. Depth is km below sea level (true depth; the Earth-flattening of the travel-time grids is undone).
* Events that NonLinLoc could not locate (tier U: too few picks or no solution inside the grids) keep the v4
  hypocentre (loc_source v4). The v4 depths come from a location without topography or water and are often above the
  surface; they are kept, with the above_surface flag, for this version only (D-B059).
* Magnitudes: Route A preferred magnitude of catalog v4.1 (M, M_type ML / Mw / Mw_cal), computed with the v4
  hypocentres; they will be recomputed with the new hypocentres in a later version.

## Counts

| item | value |
|---|---|
| events | {len(c):,} |
| located here (loc_source nll3d) | {ls.get('nll3d', 0):,} |
| v4 hypocentre kept (loc_source v4, tier U) | {ls.get('v4', 0):,} |
| ComCat hypocentre (loc_source comcat, v4.1 additions) | {ls.get('comcat', 0):,} |
| tiers | {', '.join(f'{k} {v:,}' for k, v in tiers.items())} |
| depth_quality | {', '.join(f'{k} {v:,}' for k, v in dq.items())} |
| with a magnitude | {nm:,} |
| above the ground or seafloor | {len(ab):,} ({', '.join(f'{k} {v:,}' for k, v in ab_src.items()) or 'none'}) |
| publishable (D-B055) | {'yes' if pub else 'no: events above the surface are computation errors still to fix'} |

## Columns

| column | meaning |
|---|---|
| ID | catalog v4 event ID (900001-900003: v4.1 additions) |
| raw_ID | ID in the GENIE association output |
| time | origin time (UTC) of this version's hypocentre |
| lat, lon | epicentre (deg, WGS84) |
| depth | km below sea level (negative = above sea level) |
| exp_depth | NonLinLoc expectation depth (km bsl) |
| loc_source | nll3d (located here), v4 (v4 hypocentre kept, tier U), comcat (ComCat hypocentre) |
| tier | location quality A (best) to D, U = not located here (rules in b4_catalog_v5_trackA_*_tiers.md) |
| nphs, n_sta, gap, rms | phases and stations used, largest azimuthal gap (deg), rms residual (s) |
| h_min, h_max, sig_z, ell_len3 | 68 % horizontal semi-axes, depth standard error, longest ellipsoid axis (km) |
| dist_min, dist_med, dist_max, dist_over_depth | epicentral distance to the stations (km); nearest / depth |
| surface_km | ground or seafloor elevation at the epicentre (km, positive up; GMT earth_relief 01m, bilinear) |
| above_surface | depth above the solid surface: a location error, kept visible, not moved (D-B055) |
| depth_quality | ok, unconstrained (ridge regions, Blanco outside the 2012-13 OBS deployment), biased_deep (Gorda, nearest station > 15 km) |
| event_flag | quarry_blast_likely (Texada Island box, weekday daytime) or empty |
| shift_h_km, shift_z_km | change from the v4 hypocentre |
| tile, region | NonLinLoc tile; region label (box rules of a2_parse_hyp.py) |
| depth_frame | how depth was obtained |
| v4_time, v4_lat, v4_lon, v4_depth, v4_npick, v4_sigma_m | catalog v4 origin, hypocentre, picks, GraphDD spatial sigma (m) |
| M, M_unc, M_type, ML, ML_unc, n_ML, ML_method, Mw_mt, comcat_id | Route A magnitudes of catalog v4.1 |

## Known limits of this version

* Interim (D-B058): the NonLinLoc workflow fixes of round 2 (station reach, upward pull towards the surface, the 80 km
  grid floor, tier U) are not in it; v5.1 will be the rerun with them, v5.2 the tomoDD relative relocation.
* About 4 % of the located events have a likelihood that peaks above the surface; NonLinLoc's topography floor holds
  them at the surface (D-B056), so their depth is set by the floor, not by the data.
* Event 172114: located with its v4.0 picks; v4.1 added 160 picks to it that this run did not use.
* Depths are poorly constrained offshore without nearby OBS (depth_quality) and are deeper than published local
  studies at Axial, Blanco, Gorda and Mendocino (data/tomo_v1/depth_literature/depth_literature.html). Axial: {axial}.
  Region medians against the previous relocations: b4_acceptance.md.
"""
    (vd / f"README_v{version}.md").write_text(readme)
    files = sorted(p for p in vd.iterdir() if p.is_file() and p.name != "MD5SUMS")
    (vd / "MD5SUMS").write_text("".join(f"{md5(p)}  {p.name}\n" for p in files))

    # registry (repository) and the bundle with every version built so far
    rd = REPO / "data" / "catalog_v5"
    reg_f = rd / "registry.csv"
    row = {"version": version, "date": str(date.today()), "b4_run": run, "model": model, "events": len(c),
           "nll3d": ls.get("nll3d", 0), "v4": ls.get("v4", 0), "comcat": ls.get("comcat", 0),
           "above_surface": len(ab), "publishable": pub, "csv_md5": md5(csv), "note": V["note"]}
    reg = pd.read_csv(reg_f, dtype=str) if reg_f.exists() else pd.DataFrame(columns=list(row))
    reg = pd.concat([reg[reg.version != version], pd.DataFrame([row]).astype(str)], ignore_index=True)
    rd.mkdir(parents=True, exist_ok=True)
    reg.sort_values("version").to_csv(reg_f, index=False)
    zp = OUT / "catalog_v5.zip"
    tmp = zp.with_suffix(".zip.part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(reg_f, "catalog_v5/registry.csv")
        for v in sorted(reg.version):
            for p in sorted((OUT / f"v{v}").iterdir()):
                z.write(p, f"catalog_v5/v{v}/{p.name}")
    tmp.replace(zp)

    od = rd / f"v{version}"
    od.mkdir(parents=True, exist_ok=True)
    shutil.copy(vd / f"README_v{version}.md", od)
    shutil.copy(vd / "MD5SUMS", od)
    s = pd.crosstab(c.region, c.loc_source, margins=True, margins_name="all")
    s["above_surface"] = c.groupby("region").above_surface.sum().reindex(s.index).fillna(len(ab)).astype(int)
    tab = ["| region | " + " | ".join(map(str, s.columns)) + " |", "|---" * (len(s.columns) + 1) + "|"]
    tab += [f"| {k} | " + " | ".join(f"{v:,}" for v in row) + " |" for k, row in s.iterrows()]
    (od / "summary.md").write_text(f"# Catalog v{version}: events by region and hypocentre source\n\n" + "\n".join(tab)
                                   + f"\n\nBundle {zp} md5 {md5(zp)} ({zp.stat().st_size:,} bytes)\n")
    write_manifest(f"catalog_v{version}", od,
                   inputs=[src, P["v4_1_events"], P["v4_1_magnitudes"], terms],
                   outputs=[csv, vd / f"README_v{version}.md", vd / "MD5SUMS", zp, reg_f],
                   params={"version": version, **V, "v4_1_removed": drop, "v4_1_added": add},
                   counts={"events": len(c), "loc_source": ls, "tiers": tiers, "above_surface": len(ab),
                           "above_surface_by_source": ab_src, "with_magnitude": nm, "publishable": pub})
    print(readme.split("## Columns")[0])
    print(f"bundle {zp} md5 {md5(zp)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("version")
    build(ap.parse_args().version)

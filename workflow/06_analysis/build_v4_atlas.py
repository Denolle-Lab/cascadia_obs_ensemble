#!/usr/bin/env python3
"""
Build the Cascadia catalog v4 3D atlas: one self-contained HTML file (three.js, data embedded),
in the visual language of the Cascadia Offshore Sensor Atlas (yaoderek/cascadia-sensor-atlas-preview).

Layers packed here (gzip + base64 blobs, decoded in the browser with DecompressionStream):
  events     catalog v4 (data/catalog_v4/), magnitudes from data/magnitude_v4/ when present
             (our Route A ML / preferred M), else Ian's magnitude column (flagged), the share of
             each event's picks added after ver3 (via ver6 pick_id), and Ian's spatial_sigma
             + every other catalog of the registry workflow/11_compare/catalogs.yaml with atlas: true
             (relocations, velocity-model iterations): lon/lat/depth offsets from the reference catalog on the
             same event order, location tier A-D/U, nphs, gap, rms, read from the common tables built by
             workflow/11_compare/compare_catalogs.py (CMP below); whole-catalog numbers from its summary CSVs
             + the earthquake families of the clustering of one catalog (--clusters, --clusters-catalog; default the
             catalog v5.1 nearest-neighbour clustering of workflow/06_analysis/v51_clusters_zbz.py): class, family
             and role of each event, one row per family, a colour per family (CLU_PAL, neighbours differ)
  terrain    GMT earth_relief 01m for the whole margin + the GMRT patches in data/gmrt/
  slab       Slab2 (data/slab2/cas_slab2_dep.xyz) with the coupling models in data/coupling/
             sampled onto its nodes
  stations   ver5/ver6 station table; plates: PB2002; tensors: ComCat moment tensors 2010-2016
  tomo       (--model) a Track B velocity model netCDF: 3D textures for a depth slice, a vertical cut and the
             cross-section; nodes without a value (NaN; water and air nodes of models with a medium variable) not drawn.
             The page draws isocontours of the variable shown (slice, cut, section panel) and an optional isosurface
             from these codes; nothing more is packed for them

Usage (default env, from the repo root):
  LD_LIBRARY_PATH=.pixi/envs/default/lib .pixi/envs/default/bin/python \
      workflow/06_analysis/build_v4_atlas.py [--model MODEL.nc] [--out FILE]
Output: data/catalog_v4/atlas/cascadia_v4_atlas.html (template: workflow/06_analysis/atlas_template.html)

Catalogs come from the registry only. Its top-level keys atlas_relocated (the relocated catalog: default B of Compare,
tier colours of the reference view, end of the Moves layer) and atlas_moves_from (start of the Moves layer) choose the
two catalogs the page singles out; --catalog2 and --catalog-before override them for one build with a registry id or
the registered file path, and --catalog2-label, --catalog2-note, --catalog-before-label override the registry label and
note. A catalog that is not in the registry has to be added to it (and to compare_catalogs.py's tables) first.
Example, the live configuration of 2026-11-25 with model v0.4.2 in place of v0.3.2 (2026-11-26; relocated catalog v5.1
and Moves from NLL3D-i1 come from the registry; --catalog2 v5.1 --catalog-before NLL3D-i1, or the two csv paths, give
the same page; catalogs, clusters and Moves as the v0.3.2 page):
  ATLAS_DATA_DIRS=/home/mdenolle/wt_atlas/data LD_LIBRARY_PATH=.pixi/envs/default/lib .pixi/envs/default/bin/python \\
      workflow/06_analysis/build_v4_atlas.py --model /wd1/mdenolle_data/tomo_v1/models/cascadia_obs_tomo/v0.4.2.nc \\
      --catalog-before-label "v4-NLL3D (start model)" --out /tmp/cascadia_atlas.html

Clusters (colour mode Cluster, in the view of --clusters-catalog only): events_zbz_v51.csv.gz and families_zbz_v51.csv
(and sensitivity_zbz.csv for the Axial caveat) are read from the --clusters folder (default CLU_DIR), else from
data/catalog_v5/v5.1_analysis/clusters/ in the data folders, else from their committed copies on origin/main (git show);
--clusters none leaves the mode out.

A new location run appears on the atlas by adding it to catalogs.yaml with atlas: true, running
compare_catalogs.py (writes CMP/<id>.parquet and the summary tables), then this script.
Data files are read from <repo>/data; files missing there (gitignored, in a git worktree) are read from the
main working tree's data/ (found through git), or from the folders in $ATLAS_DATA_DIRS (':'-separated).
"""
from __future__ import annotations

import argparse
import base64
import glob
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data"


def _data_dirs() -> list[Path]:
    dirs = [D] + [Path(p) for p in os.environ.get("ATLAS_DATA_DIRS", "").split(":") if p]
    try:                                            # a git worktree: fall back to the main tree's data/
        import subprocess
        common = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                capture_output=True, text=True, check=True).stdout.strip()
        dirs.append(Path(common).parent / "data")
    except Exception:
        pass
    out = []
    for d in dirs:
        if d.exists() and d.resolve() not in [o.resolve() for o in out]:
            out.append(d)
    return out


DATA_DIRS = _data_dirs()


def dd(*parts) -> Path:
    """data/<parts>: the first data folder that has it (the repo's own first); the repo path if none has it."""
    for d in DATA_DIRS:
        if (d.joinpath(*parts)).exists():
            return d.joinpath(*parts)
    return D.joinpath(*parts)


def v0_dir() -> Path:
    """Fault surfaces v0: the geojson, members and vertices files must come from the same folder (same version)."""
    names = ("fault_surfaces_v0.geojson", "fault_surfaces_v0_members.csv", "fault_surfaces_v0_vertices.csv")
    for d in DATA_DIRS:
        if all((d / "faults" / "v0" / n).exists() for n in names):
            return d / "faults" / "v0"
    return D / "faults" / "v0"


def dglob(pattern: str) -> list[str]:
    """glob under data/, the union over the data folders (the repo's own copy wins for a path in several)."""
    seen = {}
    for d in DATA_DIRS:
        for f in glob.glob(str(d / pattern)):
            seen.setdefault(os.path.relpath(f, d), f)
    return [seen[k] for k in sorted(seen)]


REGION = [-131.0, -119.5, 39.0, 51.0]           # lon0 lon1 lat0 lat1 of the atlas
LON0, LAT0 = -125.25, 45.0                      # projection centre (1 unit = 1 km)
T0 = pd.Timestamp("2010-01-01", tz="UTC")
VER3_N = 39_597_551                              # pick_id < VER3_N: pick already in ver3
STATIONS = Path("/wd1/mdenolle_data/picks_share/stations/stations_ver5_v6.csv")
# Catalog registry and the common per-catalog tables of workflow/11_compare/compare_catalogs.py
REGISTRY = ROOT / "workflow" / "11_compare" / "catalogs.yaml"
CMP = Path(os.environ.get("CATALOG_COMPARE_DIR", "/wd1/mdenolle_data/catalog_compare"))   # <id>.parquet
TIERS = "ABCDU"                                  # tier codes 0-4; 5 = located, catalog without tiers; 6 = no hypocentre

# Named views (same boxes as phase5_pygmt_map.py REGIONS)
REGIONS = [
    ("full", "Whole margin", [-130.8, -118.5, 38.5, 51.5]),
    ("endeavour", "Endeavour – Explorer", [-130.4, -127.2, 47.4, 49.8]),
    ("wa_margin", "Offshore Washington", [-127.6, -123.2, 46.2, 49.2]),
    ("puget", "Puget Lowland", [-123.6, -121.4, 46.6, 48.6]),
    ("axial", "Axial Seamount", [-130.35, -129.75, 45.75, 46.2]),
    ("or_margin", "Offshore Oregon", [-126.8, -123.2, 42.8, 46.4]),
    ("blanco", "Blanco transform", [-130.6, -126.6, 42.6, 45.3]),
    ("gorda", "Gorda plate", [-128.8, -123.8, 40.0, 43.2]),
    ("mendocino", "Mendocino triple junction", [-125.6, -123.2, 39.6, 41.4]),
]


def blob(arrays: dict[str, np.ndarray]) -> dict:
    """Concatenate typed arrays into one gzip+base64 blob with an offset table."""
    parts, table, off = [], {}, 0
    for k, a in arrays.items():
        a = np.ascontiguousarray(a)
        pad = (-off) % 4                          # keep 4-byte alignment for typed-array views
        if pad:
            parts.append(b"\0" * pad); off += pad
        b = a.tobytes()
        table[k] = {"off": off, "n": int(a.size), "dtype": a.dtype.name}
        parts.append(b); off += len(b)
    raw = b"".join(parts)
    return {"b64": base64.b64encode(gzip.compress(raw, 9)).decode(), "table": table, "bytes": off}


def grid_i16(z: np.ndarray, scale: float = 1.0) -> np.ndarray:
    out = np.round(np.nan_to_num(z * scale, nan=-32768)).clip(-32768, 32767).astype("<i2")
    return out


def downsample(lon, lat, z, maxn):
    s = max(1, int(np.ceil(max(len(lon), len(lat)) / maxn)))
    if s > 1:
        ny, nx = (len(lat) // s) * s, (len(lon) // s) * s
        z = np.nanmean(z[:ny, :nx].reshape(ny // s, s, nx // s, s), axis=(1, 3))
        lon = lon[:nx].reshape(-1, s).mean(1); lat = lat[:ny].reshape(-1, s).mean(1)
    return lon, lat, z


def terrain():
    import pygmt
    import xarray as xr
    g = pygmt.datasets.load_earth_relief(resolution="01m", region=REGION, registration="gridline")
    lon, lat, z = g.lon.values, g.lat.values, g.values.astype(float)
    grids = {"base": dict(lon=lon, lat=lat, z=z, interval=200)}
    for f in dglob("gmrt/gmrt_*.nc"):
        ds = xr.open_dataset(f)
        lo, la, zz = ds.lon.values, ds.lat.values, ds.altitude.values.astype(float)
        hi = f.endswith("_high.nc")
        lo, la, zz = downsample(lo, la, zz, 640 if hi else 400)
        if la[0] > la[-1]:
            la, zz = la[::-1], zz[::-1]
        grids[Path(f).stem] = dict(lon=lo, lat=la, z=zz, interval=50 if hi else 100)
    out, meta = {}, {}
    for k, v in grids.items():
        out[k] = grid_i16(v["z"])
        meta[k] = dict(nx=len(v["lon"]), ny=len(v["lat"]), west=float(v["lon"][0]), east=float(v["lon"][-1]),
                       south=float(v["lat"][0]), north=float(v["lat"][-1]), interval=v["interval"])
        print(f"  terrain {k}: {meta[k]['nx']}x{meta[k]['ny']}")
    return out, meta


def slab_and_coupling():
    import xarray as xr
    s = pd.read_csv(dd("slab2", "cas_slab2_dep.xyz"), header=None, names=["lon", "lat", "z"])
    s["lon"] = np.where(s.lon > 180, s.lon - 360, s.lon)
    lons, lats = np.sort(s.lon.unique()), np.sort(s.lat.unique())
    Z = s.pivot_table(index="lat", columns="lon", values="z", dropna=False).reindex(index=lats, columns=lons).values
    keep_c = (lons >= REGION[0]) & (lons <= REGION[1]); keep_r = (lats >= REGION[2]) & (lats <= REGION[3])
    lons, lats, Z = lons[keep_c], lats[keep_r], Z[np.ix_(keep_r, keep_c)]
    dep = -Z                                       # km, positive down
    out = {"slab": grid_i16(dep, 10)}
    LO, LA = np.meshgrid(lons, lats)
    models = []
    for f in dglob("coupling/Cascadia/*/*.nc"):
        ds = xr.open_dataset(f)
        var = next((v for v in ("coupling", "gamma", "interSSE") if v in ds), None)
        if var is None:
            var = next(v for v in ds.data_vars if v != "depth")
        da = ds[var]
        if da.dims != ("lat", "lon"):
            da = da.transpose("lat", "lon")
        c = da.interp(lat=xr.DataArray(LA.ravel(), dims="p"), lon=xr.DataArray(LO.ravel(), dims="p")).values
        c = c.reshape(LA.shape)
        c = np.where(np.isfinite(dep), c, np.nan)
        name = Path(f).stem
        out[f"cpl_{name}"] = np.where(np.isfinite(c), np.round(np.clip(c, 0, 1) * 254), 255).astype("u1")
        models.append(name)
        print(f"  coupling {name} ({var}): {np.isfinite(c).sum():,} slab nodes")
    meta = dict(nx=len(lons), ny=len(lats), west=float(lons[0]), east=float(lons[-1]),
                south=float(lats[0]), north=float(lats[-1]), models=models)
    return out, meta


# Build options that single out registry catalogs (main(): --catalog2*, --catalog-before*); None = the registry's choice
SEL = {"rel": None, "rel_label": None, "rel_note": None, "before": None, "before_label": None}

# Earthquake families (colour mode Cluster): nearest-neighbour clustering of one registry catalog (main(): --clusters,
# --clusters-catalog). Files are looked up in CLU["dir"], then data/<CLU_REPO>/ in the data folders, then on CLU_GIT_REF.
CLU_DIR = Path("/wd1/mdenolle_data/catalog_v5/v5.1_analysis/clusters")
CLU_REPO = ("catalog_v5", "v5.1_analysis", "clusters")
CLU_GIT_REF = "origin/main"
CLU = {"dir": CLU_DIR, "cat": "v5.1", "events": "events_zbz_v51.csv.gz", "families": "families_zbz_v51.csv",
       "sensitivity": "sensitivity_zbz.csv"}
CLU_CLS = ["background", "mainshock-aftershock", "swarm", "multiplet", "linked_to_single", "excluded"]   # event class codes 0-5
# 12 colours for the families (sRGB on the dark page): picked to maximise the smallest OKLab distance between any two,
# under normal vision and simulated protan, deutan and tritan vision, and from the background grey (#6b6b6b)
CLU_PAL = ["#c59fe1", "#69f991", "#2ca470", "#e65719", "#8e71f5", "#f09c17", "#e0e111", "#c958c2", "#8fc184", "#eb6f9a",
           "#6bd8de", "#fda9b5"]


def clu_file(name: str):
    """A clustering output file: (path or bytes buffer, source), or (None, None) when none of the places has it."""
    if CLU["dir"] is not None and (Path(CLU["dir"]) / name).exists():
        return Path(CLU["dir"]) / name, str(Path(CLU["dir"]) / name)
    f = dd(*CLU_REPO, name)
    if f.exists():
        return f, str(f)
    import io
    import subprocess
    spec = f"{CLU_GIT_REF}:data/{'/'.join(CLU_REPO)}/{name}"
    r = subprocess.run(["git", "-C", str(ROOT), "show", spec], capture_output=True)
    return (io.BytesIO(r.stdout), f"git {spec}") if r.returncode == 0 and r.stdout else (None, None)


def family_colours(lon, lat, ext, size, k: int) -> np.ndarray:
    """Colour index 0..k-1 for each family. Largest families (size) first; each takes the colour least used by the
    families already coloured around it, weighted 1 / (1 + d / 2 km)^2 with d the centroid distance less both half
    extents (0 when they overlap), within 60 km; ties go to the colour used least so far. Depth is ignored (families
    above one another share the map)."""
    la, lo = np.radians(lat), np.radians(lon)
    dx = (lo[:, None] - lo[None, :]) * np.cos((la[:, None] + la[None, :]) / 2) * 6371.0
    d = np.hypot(dx, (la[:, None] - la[None, :]) * 6371.0) - 0.5 * (ext[:, None] + ext[None, :])
    w = np.where(d < 60, 1 / (1 + np.maximum(d, 0) / 2) ** 2, 0.0)
    col, used = np.full(len(lon), -1), np.zeros(k)
    for f in np.argsort(-size, kind="stable"):
        done = col >= 0
        cost = np.bincount(col[done], weights=w[f, done], minlength=k)
        best = np.flatnonzero(cost <= cost.min() + 1e-12)
        col[f] = best[np.argmin(used[best])]
        used[col[f]] += 1
    np.fill_diagonal(d, np.inf)
    nn = np.argmin(d, axis=1)
    same = np.where(col[:, None] == col[None, :], d, np.inf).min(axis=1)
    print(f"  family colours: {k} colours over {len(lon):,} families; nearest family shares the colour for "
          f"{(col[nn] == col).sum()} families; median distance to the nearest family of the same colour "
          f"{np.median(np.maximum(same, 0)):.0f} km")
    return col


def clusters(ids: pd.Series, cats: list[dict]) -> tuple[dict, dict | None]:
    """Earthquake families of the clustering of catalog CLU["cat"] on the reference event order: cl_cls (CLU_CLS code,
    255 = not in the table), cl_fam (row of the family table, 65535 = none), cl_role (index into roles). Meta: the
    catalog's position in the atlas list, the class and role names, one column list per family field, the family
    colours (index into CLU_PAL) and the numbers of the legend caveats."""
    if CLU["dir"] is None:
        return {}, None
    pos = next((k for k, c in enumerate(cats) if c["id"] == CLU["cat"]), None)
    fe, se = clu_file(CLU["events"])
    ff, sf = clu_file(CLU["families"])
    if pos is None or fe is None or ff is None:
        print(f"  clusters: skipped ({'catalog ' + CLU['cat'] + ' not in the atlas' if pos is None else 'files not found'})")
        return {}, None
    need = ["event_id", "family_id", "cls", "role", "in_analysis", "excluded"]
    e = pd.read_csv(fe, usecols=lambda x: x in need, compression="gzip" if CLU["events"].endswith(".gz") else "infer")
    miss = [x for x in need[:4] if x not in e]
    if miss:
        raise SystemExit(f"{se}: no column {miss}")
    f = pd.read_csv(ff).sort_values("family_id").reset_index(drop=True)
    e = e.drop_duplicates("event_id")
    unknown = sorted(set(e["cls"].dropna()) - set(CLU_CLS))
    if unknown:
        raise SystemExit(f"{se}: event classes {unknown} are not in CLU_CLS {CLU_CLS}")
    # checks: every family member's family is in the family table; n = members above Mc (in_analysis)
    fid = e.family_id[e.family_id >= 0]
    if not fid.isin(f.family_id).all():
        raise SystemExit(f"{se} and {sf}: {(~fid.isin(f.family_id)).sum()} events in families missing from the family table")
    if "in_analysis" in e:
        k = e[(e.family_id >= 0) & e.in_analysis.astype(bool)].groupby("family_id").size().reindex(f.family_id).fillna(0)
        if (k.to_numpy() != f["n"].to_numpy()).any():
            raise SystemExit(f"{se} and {sf}: family sizes n differ from the members above Mc (two different runs?); "
                             "point --clusters at a folder holding both files of one run, or --clusters none")
    role = e["role"].fillna("").astype(str)
    if "excluded" in e:                              # why an event is left out: not_earthquake / no_hypocentre
        role = np.where(e["cls"] == "excluded", "excluded_" + e["excluded"].fillna("").astype(str), role)
    e = e.assign(role=role).set_index("event_id")
    r = e.reindex(ids.to_numpy())
    roles = sorted(e["role"].unique())
    fidx = pd.Series(np.arange(len(f)), index=f.family_id.to_numpy())
    fk = r["family_id"].map(fidx)
    arr = {"cl_cls": r["cls"].map({c: j for j, c in enumerate(CLU_CLS)}).fillna(255).to_numpy().astype("u1"),
           "cl_fam": np.where(fk.notna(), fk.fillna(0), 65535).astype("<u2"),
           "cl_role": r["role"].map({x: j for j, x in enumerate(roles)}).fillna(255).to_numpy().astype("u1")}
    col = family_colours(f["lon"].to_numpy(float), f["lat"].to_numpy(float), f["extent_km"].fillna(0).to_numpy(float),
                         f.get("n_all", f["n"]).to_numpy(float), len(CLU_PAL))

    def L(name, fn=lambda v: v, default=None):      # one family column as a JSON list (NaN -> null)
        if name not in f:
            return [default] * len(f)
        return [None if (isinstance(v, float) and not np.isfinite(v)) or v is None or v is pd.NA else fn(v) for v in f[name]]
    tb = lambda v: {True: 1, False: 0, "True": 1, "False": 0}.get(v)
    fam = dict(id=L("family_id", int), type=L("type", CLU_CLS.index), n=L("n", int), n_all=L("n_all", int),
               M=L("Mmax", lambda v: round(float(v), 2)), Mmin=L("Mmin_all", lambda v: round(float(v), 2)),
               t0=L("start", lambda v: str(v)[:16]), t1=L("end", lambda v: str(v)[:16]),
               dur=L("duration_days", lambda v: round(float(v), 2)), ext=L("extent_km", lambda v: round(float(v), 1)),
               reg=L("region_v51", str), main=L("mainshock_id", int),
               dq=[None if not np.isfinite(a) else [round(float(a), 1), round(float(b), 1), round(float(c), 1)]
                   for a, b, c in zip(*(f[x] if x in f else pd.Series(np.nan, index=f.index)
                                        for x in ("depth_good_q10", "depth_good_med", "depth_good_q90")))],
               res=L("mig_resolvable", tb), mig=L("mig_migrating", tb),
               p1=L("n_review_P1_msh_deep", int, 0), day=L("n_daytime_screened", int, 0),
               p2=L("n_review_P2_brookings", int, 0), v4=L("v4_status", str), col=[int(x) for x in col])
    # legend caveats: Axial background share (main run, without tier D) and swarms whose migration is not resolvable
    cav = {}
    fs, _ = clu_file(CLU["sensitivity"])
    if fs is not None:
        s = pd.read_csv(fs)
        ax = s[s.region == "AX"].set_index("variant")["pct_background"]
        if {"main", "tierABC"} <= set(ax.index):
            cav["ax_bg"] = [float(ax["main"]), float(ax["tierABC"])]
    sw = f[f.type == "swarm"]
    if "mig_resolvable" in sw:
        mr = sw["mig_resolvable"].map(tb)
        cav["mig"] = dict(n=int(mr.notna().sum()), unres=int((mr == 0).sum()),
                          herr=_num(sw.loc[mr.notna(), "h_err_med_km"].median()) if "h_err_med_km" in sw else None)
    below = (e.family_id >= 0) & e.role.str.endswith("_belowMc")
    cnt = {c: int((r["cls"] == c).sum()) for c in CLU_CLS}
    print(f"  clusters ({CLU['cat']}): {len(f):,} families ({', '.join(f'{v} {k}' for k, v in f.type.value_counts().items())}); "
          f"atlas events by class: " + ", ".join(f"{k} {v:,}" for k, v in cnt.items())
          + f"; {int(r['cls'].isna().sum()):,} atlas events not in the table; from {se} and {sf}")
    meta = dict(cat=pos, cat_id=CLU["cat"], cls=CLU_CLS, roles=roles, pal=CLU_PAL, fam=fam, counts=cnt, cav=cav,
                n_fam_below=int(below.sum()), n_fam_events=int((e.family_id >= 0).sum()),
                src=f"{Path(se).name if not se.startswith('git') else se}, {Path(sf).name if not sf.startswith('git') else sf}")
    return arr, meta


def events():
    c = pd.read_csv(dd("catalog_v4", "Cascadia_relocated_catalog_v4.csv"), skipinitialspace=True)
    c.columns = [x.strip() for x in c.columns]
    t = pd.to_datetime(c["datetime"].str.strip(), utc=True)
    days = ((t - T0) / pd.Timedelta(days=1)).to_numpy()
    M = c["magnitude"].to_numpy(float); Mtype = np.full(len(c), 3, "u1")      # 3 = Ian's column
    msrc = "Ian's catalog magnitude (type unknown; runs high offshore)"
    mfile = dd("magnitude_v4", "cascadia_catalog_M_routeA.csv")
    if mfile.exists():
        m = pd.read_csv(mfile).set_index("event_id")
        mm = m.reindex(c["ID"])
        ok = mm["M"].notna().to_numpy()
        M = np.where(ok, mm["M"].to_numpy(float), np.nan)
        code = {"ML": 0, "Mw": 1, "Mw_cal": 2}
        Mtype = np.where(ok, mm["M_type"].map(code).fillna(0).to_numpy(), 255).astype("u1")
        msrc = f"Route A preferred magnitude, catalog v4 ({ok.sum():,} of {len(c):,} events)"
    pid = pd.read_parquet(dd("catalog_v4", "ridge_report", "v4_pick_ids.parquet"))
    newfrac = (pid["pick_id"] >= VER3_N).groupby(pid["event_ID"]).mean().reindex(c["ID"]).fillna(0).to_numpy()
    sig = pd.to_numeric(c["spatial_sigma(m)"], errors="coerce").to_numpy()
    arr = {
        "lon": c["longitude"].to_numpy("<f4"), "lat": c["latitude"].to_numpy("<f4"),
        "dep": c["depth(km)"].to_numpy("<f4"), "day": days.astype("<f4"),
        "id": c["ID"].to_numpy("<i4"),
        "mag": np.where(np.isfinite(M), np.round(M * 100), -32768).astype("<i2"),
        "sig": np.where(np.isfinite(sig), np.clip(np.round(sig / 10), 0, 65534), 65535).astype("<u2"),
        "np": c["Num_p_picks"].clip(0, 255).to_numpy("u1"), "ns": c["Num_s_picks"].clip(0, 255).to_numpy("u1"),
        "new": np.round(newfrac * 100).astype("u1"), "mtype": Mtype,
    }
    # fault of each event: index into the surfaces in GeoJSON feature order (the order the atlas draws them); 65535 = none
    flt = np.full(len(c), 65535, "<u2")
    sv = v0_dir() / "fault_surfaces_v0.geojson"
    if sv.exists():
        ids = [f["properties"]["surface_id"] for f in json.load(open(sv))["features"]]
        mem = pd.read_csv(v0_dir() / "fault_surfaces_v0_members.csv").set_index("event_id").surface_id
        k = c["ID"].map(mem).map({s_: i for i, s_ in enumerate(ids)})
        flt = np.where(k.notna(), k.fillna(0), 65535).astype("<u2")
        print(f"  events on a fault: {(flt < 65535).sum():,}")
    arr["flt"] = flt
    cat_arr, cats = catalogs(c["ID"], c[["latitude", "longitude", "depth(km)"]].to_numpy(float))
    arr.update(cat_arr)
    clu_arr, clu = clusters(c["ID"], cats["list"])
    arr.update(clu_arr)
    order = np.argsort(arr["day"], kind="stable")   # time order: the time slider is a range in index
    arr = {k: v[order] for k, v in arr.items()}
    print(f"  events: {len(c):,}; magnitudes: {msrc}")
    return arr, dict(n=len(c), msrc=msrc, t0="2010-01-01", days_max=float(np.nanmax(days)), cats=cats["list"],
                     rel=cats["rel"], moves=cats["moves"], clu=clu)


def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else round(v, 3)


def catalog_numbers(cid: str, ref: str) -> dict:
    """Whole-catalog numbers of compare_catalogs.py (data/catalog_compare/*.csv) for one catalog."""
    out = {}
    f = dd("catalog_compare", "summary.csv")
    if f.exists():
        t = pd.read_csv(f)
        r = t[(t.catalog == cid) & (t.region == "All")]
        if len(r):
            r = r.iloc[0]
            out.update({k: _num(r[k]) for k in ("n_events", "n_located", "depth_p10", "depth_p50", "depth_p90",
                                                 "pct_above_surface", "dh_ref_p50", "dh_ref_p90", "dz_ref_p50", "rms_p50")})
    f = dd("catalog_compare", "anss.csv")
    if f.exists():
        t = pd.read_csv(f)
        for sub in ("near", "all"):
            r = t[(t.catalog == cid) & (t.region == "All") & (t.subset == sub)]
            if len(r):
                r = r.iloc[0]
                out[f"anss_{sub}"] = {k: _num(r[k]) for k in ("n", "dz_p50", "dz_mad", "dh_p50", "abs_dz_p50")}
    f = dd("catalog_compare", "faults_summary.csv")
    if f.exists():
        t = pd.read_csv(f).set_index("catalog")
        if cid in t.index:
            out["fault_rms"] = _num(t.loc[cid, "rms_plane_p50"]); out["fault_n"] = int(t.loc[cid, "n_surfaces"])
    return out


def _registry_id(x: str, ents: dict) -> str:
    """A registry id, or the id of the entry whose file is the path x (absolute, or relative to the repo root)."""
    if x in ents:
        return x
    px = Path(x).expanduser()
    px = (px if px.is_absolute() else ROOT / px).resolve()
    for cid, e in ents.items():
        f = Path(e["file"])
        if (f if f.is_absolute() else ROOT / f).resolve() == px:
            return cid
    raise SystemExit(f"{x}: not a catalog of {REGISTRY}. Add it there (id, adapter, file, ...), run "
                     "workflow/11_compare/compare_catalogs.py --only <id>, then build the atlas.")


def catalogs(ids: pd.Series, ref_xyz: np.ndarray) -> tuple[dict, dict]:
    """Registry catalogs with atlas: true (plus the catalogs chosen with --catalog2 / --catalog-before). The reference
    catalog (registry 'reference') gives the event set and the base positions (the v4 csv read in events()); every other
    catalog is stored on the same event order: lon/lat as int16 offsets from the reference in 1e-4 deg (-32768 = not
    located there: tier U, no hypocentre, event missing, or an offset out of int16 range), the exact depth as float32
    (NaN = not located; 10 m steps could put an event just under the surface above it), tier 0-4 = A-D, U,
    5 = located without a tier class, 6 = no hypocentre in this catalog (loc_source none: not drawn in its view nor in
    Compare); nphs; gap in 2 deg and rms in 10 ms steps (uint8, 255 = none); with own_surface: true in the registry, the
    catalog's own solid surface (float32 km, up; NaN = none) for its above-surface test. Arrays are named c<k>_* with k
    the position in the list. Returns the arrays and dict(list=<catalog metadata>, rel=<index>, moves=dict(from, to) | None)."""
    import yaml
    if not REGISTRY.exists():
        print(f"  catalogs: {REGISTRY} missing, reference only")
        return {}, dict(list=[dict(id="v4", label="Catalog v4", key=None, status="final", ref=True, has_tiers=False,
                                   n_loc=len(ids))], rel=0, moves=None)
    reg = yaml.safe_load(open(REGISTRY))
    ref = reg["reference"]
    ents = {e["id"]: e for e in reg["catalogs"]}
    order = [ref] + [e["id"] for e in reg["catalogs"] if e.get("atlas") and e["id"] != ref]
    rel_id = _registry_id(SEL["rel"], ents) if SEL["rel"] else reg.get("atlas_relocated")
    mv_id = _registry_id(SEL["before"], ents) if SEL["before"] else reg.get("atlas_moves_from")
    for x in (rel_id, mv_id):                       # a catalog chosen for this build is shown even without atlas: true
        if x and x not in order:
            if x not in ents:
                raise SystemExit(f"{x}: atlas_relocated / atlas_moves_from names no catalog of {REGISTRY}")
            order.append(x)
    pref = pd.read_parquet(CMP / f"{ref}.parquet").set_index("v4_id") if (CMP / f"{ref}.parquet").exists() else None
    if pref is not None:                            # the reference table must hold the locations the atlas shows
        dmax = np.nanmax(np.abs(pref.reindex(ids.to_numpy())[["lat", "lon", "depth"]].to_numpy() - ref_xyz))
        if not dmax < 1e-3:
            raise SystemExit(f"{CMP / ref}.parquet differs from the v4 csv by {dmax}: rerun compare_catalogs.py")
    arr, cats = {}, []
    for k, cid in enumerate(order):
        e = dict(ents[cid])
        if cid == rel_id:
            e.update({k_: v for k_, v in (("label", SEL["rel_label"]), ("note", SEL["rel_note"])) if v})
        if cid == mv_id and SEL["before_label"]:
            e["label"] = SEL["before_label"]
        meta = dict(id=cid, label=e.get("label", cid), velocity=e.get("velocity"), method=e.get("method"),
                    status=e.get("status"), date=str(e.get("date") or ""), parent=e.get("parent"), note=e.get("note"),
                    ref=(k == 0), key=None, has_tiers=False, n_loc=len(ids), n_nohypo=0, num=catalog_numbers(cid, ref))
        if k == 0:
            cats.append(meta)
            continue
        f = CMP / f"{cid}.parquet"
        if not f.exists() or pref is None:
            print(f"  catalog {cid}: {f} missing (run workflow/11_compare/compare_catalogs.py), skipped")
            continue
        t = pd.read_parquet(f)
        t = t[t.v4_id.notna()].drop_duplicates("v4_id")
        t.index = t.v4_id.astype("int64")
        n = t.reindex(ids.to_numpy())
        r = pref.reindex(ids.to_numpy())
        loc = n["located"].astype("boolean").fillna(False).to_numpy(bool) & n.lat.notna().to_numpy() & n.depth.notna().to_numpy()
        # no hypocentre in this catalog (common-table column of compare_catalogs.py; absent in tables built before it)
        nohypo = n["no_hypo"].astype("boolean").fillna(False).to_numpy(bool) & ~loc if "no_hypo" in n else np.zeros(len(n), bool)
        off = {}
        for col, s_ in (("lon", 1e4), ("lat", 1e4)):
            v = np.round((n[col] - r[col]).to_numpy(float) * s_)
            loc &= np.isfinite(v) & (np.abs(v) <= 32767)
            off[col] = v
        key = f"c{len(cats)}"
        i16 = lambda v: np.where(loc, np.nan_to_num(v), -32768).astype("<i2")
        u1 = lambda v, s_: np.where(np.isfinite(v) & loc, np.clip(np.round(v * s_), 0, 254), 255).astype("u1")
        tl = n["tier"].fillna("").astype(str).str.strip()
        has_t = bool(tl.isin(list("ABCD")).any())
        tier = tl.map({t_: j for j, t_ in enumerate(TIERS)}).fillna(5).to_numpy()
        tier = np.where(loc, np.where(tier == 4, 5, tier), np.where(nohypo, 6, 4)).astype("u1")   # not located -> U
        arr.update({f"{key}_dlon": i16(off["lon"]), f"{key}_dlat": i16(off["lat"]),
                    f"{key}_dep": np.where(loc, n["depth"].to_numpy(float), np.nan).astype("<f4"),
                    f"{key}_tier": tier, f"{key}_nph": n["nphs"].fillna(0).clip(0, 255).to_numpy().astype("u1"),
                    f"{key}_gap": u1(n["gap"].to_numpy(float), 0.5), f"{key}_rms": u1(n["rms"].to_numpy(float), 100)})
        own = bool(e.get("own_surface")) and "surface_cat_km" in n
        if e.get("own_surface") and not own:
            print(f"  catalog {cid}: own_surface set but {f.name} has no surface_cat_km column: rerun compare_catalogs.py --only {cid}")
        if own:                                     # the catalog's own solid surface: the atlas tests "above surface" against it
            arr[f"{key}_surf"] = np.where(loc, n["surface_cat_km"].to_numpy(float), np.nan).astype("<f4")
        cnt = {TIERS[j] if j < 5 else ("-" if j == 5 else "none"): int((tier == j).sum()) for j in range(7)}
        meta.update(key=key, has_tiers=has_t, n_loc=int(loc.sum()), n_nohypo=int(nohypo.sum()), tiers=cnt, own_surface=own)
        cats.append(meta)
        print(f"  catalog {cid} ({e.get('status')}): {loc.sum():,} of {len(ids):,} reference events located, "
              f"{nohypo.sum():,} without a hypocentre; tiers " + " ".join(f"{t_} {v:,}" for t_, v in cnt.items() if v)
              + ("; above-surface test against its own surface" if own else ""))
    pos = {c["id"]: j for j, c in enumerate(cats)}
    rel = pos.get(rel_id, len(cats) - 1)
    moves = dict(**{"from": pos[mv_id]}, to=rel) if mv_id in pos and pos[mv_id] != rel else None
    print(f"  relocated catalog: {cats[rel]['id']}" + (f"; moves {mv_id} -> {cats[rel]['id']}" if moves else "; no moves layer"))
    return arr, dict(list=cats, rel=rel, moves=moves)


def stations():
    s = pd.read_csv(STATIONS)
    s = s[(s.lon.between(REGION[0], REGION[1])) & (s.lat.between(REGION[2], REGION[3]))]
    return [dict(net=r.network, sta=r.station, lon=round(r.lon, 4), lat=round(r.lat, 4), elev=round(float(r.elev_m), 1),
                 sea=bool(r.seafloor), start=str(r.epoch_start), end=str(r.epoch_end), v6=bool(r.v6_new_station),
                 site=str(r.site_name) if isinstance(r.site_name, str) else "")
            for r in s.itertuples()]


def plates():
    d = json.load(open(dd("tectonics", "PB2002_boundaries.json")))
    out = []
    for f in d["features"]:
        xy = np.array(f["geometry"]["coordinates"])
        xy[:, 0] = np.where(xy[:, 0] > 180, xy[:, 0] - 360, xy[:, 0])
        inb = (xy[:, 0] >= REGION[0] - 1) & (xy[:, 0] <= REGION[1] + 1) & (xy[:, 1] >= REGION[2] - 1) & (xy[:, 1] <= REGION[3] + 1)
        if inb.sum() >= 2:
            out.append(dict(name=f["properties"]["Name"], xy=np.round(xy[inb], 4).tolist()))
    return out


def tensors():
    m = pd.read_csv(dd("focal", "comcat_mt.csv"))
    t = pd.to_datetime(m["time"], utc=True, format="ISO8601")
    k = (t >= T0) & (t < pd.Timestamp("2017-01-01", tz="UTC")) & m.lon.between(*REGION[:2]) & m.lat.between(*REGION[2:])
    m = m[k]
    cols = ["mrr", "mtt", "mpp", "mrt", "mrp", "mtp"]
    out = []
    for r in m.itertuples():
        mt = np.array([getattr(r, c) for c in cols], float)
        mt = (mt / np.abs(mt).max()).round(4).tolist()
        out.append(dict(id=r.id, t=str(r.time)[:19], lon=r.lon, lat=r.lat, dep=float(r.mt_depth if np.isfinite(r.mt_depth) else r.depth),
                        mw=float(r.mag), src=f"{r.mt_source} {r.mt_type}", mt=mt,
                        sdr=[None if not np.isfinite(x) else float(x) for x in (r.strike1, r.dip1, r.rake1)],
                        place=str(r.place)))
    print(f"  tensors: {len(out)}")
    return out


def split_dois(v):
    """Split a ';'-separated DOI list; old GSA DOIs end in 'CO;2', so a lone '2' rejoins the previous DOI."""
    out = []
    for p in (x.strip() for x in str(v or "").split(";")):
        if out and p == "2" and out[-1].upper().endswith(".CO"):
            out[-1] += ";2"
        elif p.startswith("10."):
            out.append(p)
    return out


def surfaces_display(path, max_side=12):
    """Fault surfaces on a coarser grid for display: each rectangle's 1 km vertex grid (row-major in
    _vertices.csv) is subsampled to at most max_side points per side, corners kept."""
    feats = json.load(open(path))["features"]
    vx = pd.read_csv(v0_dir() / "fault_surfaces_v0_vertices.csv")
    lon, lat, dep, surf = [], [], [], []
    for f in feats:
        p = f["properties"]
        v = vx[vx.surface_id == p["surface_id"]].sort_values("vertex")
        x, y, z = f["geometry"]["coordinates"][0][0][2]          # third vertex of the first triangle = (row 1, col 1)
        km = 111.195
        d2 = ((v.lon - x) * km * np.cos(np.radians(y))) ** 2 + ((v.lat - y) * km) ** 2 + (v.depth_km + z / 1000.0) ** 2
        nu = int(np.argmin(d2.to_numpy())) - 1                   # match in 3D: steep faults repeat lon/lat down dip
        nv = len(v) // nu
        G = v[["lon", "lat", "depth_km"]].to_numpy().reshape(nv, nu, 3)
        ri = np.unique(np.r_[np.linspace(0, nv - 1, min(nv, max_side)).round().astype(int)])
        ci = np.unique(np.r_[np.linspace(0, nu - 1, min(nu, max_side)).round().astype(int)])
        G = G[np.ix_(ri, ci)]
        a0 = len(lon) // 3
        for i in range(G.shape[0] - 1):
            for k in range(G.shape[1] - 1):
                q = G[i, k], G[i, k + 1], G[i + 1, k + 1], G[i + 1, k]
                for t in ((q[0], q[1], q[2]), (q[0], q[2], q[3])):
                    for pt in t:
                        lon.append(pt[0]); lat.append(pt[1]); dep.append(pt[2])
        surf.append(dict(id=p["surface_id"], kind=p["kind"], stable=p["stable"], n=p["n_events"], strike=p["strike"],
                         dip=p["dip"], dip68=p["dip_68"], sig=p["sigma_total_median_km"],
                         L=p["length_km"], W=p["width_km"], a=a0, b=len(lon) // 3,
                         lm=(p.get("lit_match") or "")[:140], lc=p.get("lit_confidence"), lk=p.get("lit_class"),
                         ln=bool(p.get("lit_new_candidate")), refs=p.get("lit_refs") or "",
                         dois=split_dois(p.get("lit_dois")),
                         urls=[u.strip() for u in str(p.get("lit_urls") or "").split(";") if u.strip().startswith("http")],
                         md=p.get("min_D_to_parallel"), top=p.get("top_km"), bot=p.get("bottom_km"),
                         jac=p.get("cluster_jaccard_median")))
    return (np.array(lon, "<f4"), np.array(lat, "<f4"), np.array(dep, "<f4")), surf


def tri_mesh(features, keep):
    """MultiPolygon triangle features -> flat lon/lat/depth(km, down) per vertex + per-feature ranges."""
    lon, lat, dep, ranges = [], [], [], []
    for i, f in enumerate(features):
        if not keep(f):
            continue
        n0 = len(lon)
        for poly in f["geometry"]["coordinates"]:
            ring = poly[0][:3]                       # triangle (closing vertex dropped)
            if len(ring) < 3:
                continue
            for x, y, z in ring:
                lon.append(x); lat.append(y); dep.append(-z / 1000.0)
        if len(lon) > n0:
            ranges.append((i, n0, len(lon)))
    return (np.array(lon, "<f4"), np.array(lat, "<f4"), np.array(dep, "<f4")), ranges


def faults():
    """CRESCENT CFM v1.0 (data/faults/crescent_cfm/): crustal 3D meshes (5 km), traces, CASIE21."""
    F = dd("faults", "crescent_cfm")
    if not F.exists():
        print("  faults: data/faults/crescent_cfm missing, skipped")
        return {}, {}
    inreg = lambda f: any(REGION[0] <= p[0][0][0] <= REGION[1] and REGION[2] <= p[0][0][1] <= REGION[3]
                          for p in f["geometry"]["coordinates"][:50])
    mesh = json.load(open(F / "crescent_cfm_crustal_3d_5km.geojson"))["features"]
    (lo, la, de), rng = tri_mesh(mesh, inreg)
    props = []
    for i, a, b in rng:
        p = mesh[i]["properties"]
        props.append(dict(name=p.get("name"), id=p.get("CFM_ID"), dip=p.get("dip"), dipdir=p.get("dip_dir"),
                          rake=p.get("rake"), top=p.get("upper_depth"), bot=p.get("lower_depth"),
                          src=p.get("source"), a=a // 3, b=b // 3))   # triangle index range
    cas = json.load(open(F / "casie21_plate_boundary_multipart.geojson"))["features"]
    (clo, cla, cde), _ = tri_mesh(cas, lambda f: True)
    tr = json.load(open(F / "crescent_cfm_crustal_traces.geojson"))["features"]
    traces = []
    for f in tr:
        xy = np.array(f["geometry"]["coordinates"])[:, :2]
        if not ((xy[:, 0] >= REGION[0]) & (xy[:, 0] <= REGION[1]) & (xy[:, 1] >= REGION[2]) & (xy[:, 1] <= REGION[3])).any():
            continue
        if len(xy) > 60:
            xy = xy[np.unique(np.r_[np.linspace(0, len(xy) - 1, 60).astype(int)])]
        p = f["properties"]
        traces.append(dict(name=p.get("name"), sense=p.get("slip_sense"), dip=p.get("avg_dip"), dipdir=p.get("dip_dir"),
                           level=p.get("fault_level"), rate=p.get("NSHM23_sliprates"), age=p.get("QFFDAgeCatBin"),
                           region=p.get("region"), xy=np.round(xy, 4).tolist()))
    print(f"  faults: {len(props)} CFM meshes ({len(lo) // 3:,} triangles), {len(traces)} traces, "
          f"CASIE21 {len(clo) // 3:,} triangles")
    arr = {"f_lon": lo, "f_lat": la, "f_dep": de, "c_lon": clo, "c_lat": cla, "c_dep": cde}
    surf = []
    sv = v0_dir() / "fault_surfaces_v0.geojson"
    if sv.exists():                         # event-derived surfaces v0 (workflow/07_faults/fault_surfaces_planes.py)
        (slo, sla, sde), surf = surfaces_display(sv)
        arr.update({"s_lon": slo, "s_lat": sla, "s_dep": sde})
        print(f"  surfaces v0: {len(surf)} ({len(slo) // 3:,} display triangles)")
    return arr, dict(meshes=props, traces=traces, surf=surf)


MC_LAYERS = [  # key, label, file, column
    ("best", "All years · best (GFT95, GFT90, MAXC+0.2)", "mc_grid.csv", "mc_best"),
    ("maxc", "All years · MAXC+0.2", "mc_grid.csv", "mc_maxc"),
    ("mbs", "All years · b-value stability", "mc_grid.csv", "mc_mbs"),
    ("gft95", "All years · goodness of fit 95%", "mc_grid.csv", "mc_gft95"),
    ("p2010", "2010 (no OBS) · best", "mc_grid_2010.csv", "mc_best"),
    ("p2011", "2011–2012 (OBS) · best", "mc_grid_2011-2012.csv", "mc_best"),
    ("p2013", "2013–2015 (OBS) · best", "mc_grid_2013-2015.csv", "mc_best"),
    ("p2016", "2016 (no OBS) · best", "mc_grid_2016.csv", "mc_best"),
]
MC_SIG = {"mc_best": "sigma_boot", "mc_maxc": "sigma_maxc", "mc_mbs": "sigma_mbs", "mc_gft95": "sigma_gft"}


def mc_grids():
    """Mc on the 0.1 deg grid (v4_mc_maps.py): uint8 Mc*20+40 and sigma*50, 255 = no value."""
    d = dd("catalog_v4", "mc_maps", "mc_grid.csv").parent
    if not (d / "mc_grid.csv").exists():
        return None, None
    base = pd.read_csv(d / "mc_grid.csv")
    lons, lats = np.round(np.sort(base.lon.unique()), 4), np.round(np.sort(base.lat.unique()), 4)
    out, layers = {}, []
    u8 = lambda v, s, o: np.where(np.isfinite(v), np.clip(np.round(v * s + o), 0, 254), 255).astype("u1")
    for key, label, f, col in MC_LAYERS:
        g = pd.read_csv(d / f)
        g = g.assign(lon=g.lon.round(4), lat=g.lat.round(4)).set_index(["lat", "lon"])
        idx = pd.MultiIndex.from_product([lats, lons], names=["lat", "lon"])
        mc = g[col].reindex(idx).to_numpy(float); sg = g[MC_SIG[col]].reindex(idx).to_numpy(float)
        nn = g["n"].reindex(idx).to_numpy(float)
        out[f"mc_{key}"], out[f"sg_{key}"] = u8(mc, 20, 40), u8(sg, 50, 0)
        if key == "best" or key.startswith("p"):
            out[f"n_{key}"] = np.nan_to_num(nn).astype("<u2")
        layers.append(dict(key=key, label=label, nodes=int(np.isfinite(mc).sum())))
        print(f"  Mc {key}: {np.isfinite(mc).sum():,} nodes")
    # colour table of the paper figures (GMT cpt: z0 r/g/b z1 r/g/b)
    cpt = [ln.split() for ln in (d / "mc.cpt").read_text().splitlines() if ln[:1].replace("-", "").replace(".", "").isdigit()]
    stops = [[float(r[0])] + [int(round(float(x))) for x in r[1].split("/")] for r in cpt]
    stops.append([float(cpt[-1][2])] + [int(round(float(x))) for x in cpt[-1][3].split("/")])
    meta = dict(nx=len(lons), ny=len(lats), west=float(lons[0]), south=float(lats[0]), step=float(np.round(lons[1] - lons[0], 4)),
                layers=layers, cpt=stops)
    return out, meta


def tomo_model(path):
    """Velocity model (Track B, netCDF of b2_model_export.py) for the atlas: one uint8 per node and variable,
    255 = no value. dVp, dVs in % from the starting model (code = %*4 + 127, 0.25 % steps, +-31.75 %), Vp/Vs
    (1.4 + code/200), Vp (1.5 + code/32 km/s), Vs (0.5 + code/50 km/s); DWS of P and S rays as round(50 log10(1 + DWS))
    so the page can mask sparsely sampled cells at any threshold."""
    import xarray as xr
    d = xr.open_dataset(path)
    u8 = lambda v, scale, off: np.where(np.isfinite(v), np.clip(np.round((v - off) * scale), 0, 254), 255).astype("u1")
    a = lambda k: d[k].transpose("depth", "latitude", "longitude").to_numpy().astype(float)
    out = {"dvp": np.where(np.isfinite(a("Vp_perturbation")), np.clip(np.round(a("Vp_perturbation") * 4) + 127, 0, 254), 255).astype("u1"),
           "dvs": np.where(np.isfinite(a("Vs_perturbation")), np.clip(np.round(a("Vs_perturbation") * 4) + 127, 0, 254), 255).astype("u1"),
           "vpvs": u8(a("VpVs"), 200, 1.4), "vp": u8(a("Vp"), 32, 1.5), "vs": u8(a("Vs"), 50, 0.5),
           "dwsp": np.clip(np.round(50 * np.log10(1 + np.nan_to_num(a("dws_p")))), 0, 254).astype("u1"),
           "dwss": np.clip(np.round(50 * np.log10(1 + np.nan_to_num(a("dws_s")))), 0, 254).astype("u1")}
    if "resolution_p" in d:                                # merged models: finest resolved checkerboard cell (km), 0 = none
        out["resp"] = np.clip(np.nan_to_num(a("resolution_p")), 0, 254).astype("u1")
        out["ress"] = np.clip(np.nan_to_num(a("resolution_s")), 0, 254).astype("u1")
    masked = {}
    if "medium" in d:     # v0.3.2+: 0 rock, 1 water, 2 air (flag_meanings): no rock velocity there, every variable empty
        med = d["medium"].transpose("depth", "latitude", "longitude").to_numpy()
        for k in ("dvp", "dvs", "vpvs", "vp", "vs"):
            out[k][med != 0] = 255
        masked = {"water": int((med == 1).sum()), "air": int((med == 2).sum())}
    for k in ("dvp", "dvs", "vpvs", "vp", "vs"):           # NaN in rock would be a model error: say so, draw nothing there
        bad = int((out[k] == 255).sum()) - sum(masked.values())
        if bad > 0:
            print(f"  velocity model: {k} has no value at {bad:,} nodes besides water and air (not drawn)")
    lon, lat = d.longitude.to_numpy().astype(float), d.latitude.to_numpy().astype(float)
    ver = d.attrs.get("version") or d.attrs.get("model_version") or Path(path).stem
    status = d.attrs.get("status", "")
    reg = Path(path).parent / "registry.csv"                # status lives in the model registry (b2_model_export.py)
    if not status and reg.exists():
        r = pd.read_csv(reg, dtype=str)
        r = r[r.version.str.lstrip("v") == str(ver).lstrip("v")]
        status = r.status.iloc[-1] if len(r) else ""
    meta = dict(nx=len(lon), ny=len(lat), nz=d.sizes["depth"], west=float(lon[0]), south=float(lat[0]),
                dx=float(np.round(np.diff(lon).mean(), 5)), dy=float(np.round(np.diff(lat).mean(), 5)),
                depths=[float(z) for z in d.depth.values], version=str(ver), status=str(status),
                title=str(d.attrs.get("title", "")), masked=masked)
    print(f"  velocity model {ver}: {meta['nx']} x {meta['ny']} x {meta['nz']} nodes"
          + (f"; no value at {masked['water']:,} water and {masked['air']:,} air nodes" if masked else ""))
    return out, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", default=str(ROOT / "workflow" / "06_analysis" / "atlas_template.html"))
    ap.add_argument("--out", default=str(D / "catalog_v4" / "atlas" / "cascadia_v4_atlas.html"))
    ap.add_argument("--model", help="Track B velocity model netCDF to add as a layer")
    ap.add_argument("--catalog2", help="relocated catalog for this build: a registry id or the registered file path "
                    "(default: registry atlas_relocated)")
    ap.add_argument("--catalog2-label", help="its name on the page (default: its registry label)")
    ap.add_argument("--catalog2-note", help="one or two sentences for the legend (default: its registry note)")
    ap.add_argument("--catalog-before", help="catalog the Moves layer starts from: a registry id or the registered file path "
                    "(default: registry atlas_moves_from)")
    ap.add_argument("--catalog-before-label", help="its name on the page (default: its registry label)")
    ap.add_argument("--clusters", default=str(CLU_DIR), help="folder of the clustering outputs (events_zbz_v51.csv.gz, "
                    f"families_zbz_v51.csv, sensitivity_zbz.csv; default {CLU_DIR}; files missing there are read from "
                    f"data/{'/'.join(CLU_REPO)}/ or {CLU_GIT_REF}); 'none': no Cluster colour mode")
    ap.add_argument("--clusters-catalog", default=CLU["cat"], help="registry id of the catalog whose locations the "
                    f"families were computed on; the Cluster mode is offered in its view only (default {CLU['cat']})")
    a = ap.parse_args(argv)
    print("building atlas data")
    SEL.update(rel=a.catalog2, rel_label=a.catalog2_label, rel_note=a.catalog2_note, before=a.catalog_before,
               before_label=a.catalog_before_label)
    CLU.update(dir=None if a.clusters.lower() == "none" else Path(a.clusters), cat=a.clusters_catalog)
    ev, ev_meta = events()
    ter, ter_meta = terrain()
    sl, sl_meta = slab_and_coupling()
    fa, fa_meta = faults()
    mc, mc_meta = mc_grids()
    tm, tm_meta = tomo_model(a.model) if a.model else (None, None)
    payload = {
        "meta": dict(lon0=LON0, lat0=LAT0, region=REGION, regions=[dict(key=k, label=l, box=b) for k, l, b in REGIONS],
                     events=ev_meta, terrain=ter_meta, slab=sl_meta, faults=fa_meta, mc=mc_meta, tomo=tm_meta,
                     built=pd.Timestamp.now().strftime("%Y-%m-%d")),
        "events": blob(ev), "terrain": blob(ter), "slab": blob(sl), "faults": blob(fa) if fa else None,
        "mc": blob(mc) if mc else None, "tomo": blob(tm) if tm else None,
        "stations": stations(), "plates": plates(), "tensors": tensors(),
    }
    html = Path(a.template).read_text().replace("/*__ATLAS_DATA__*/null", json.dumps(payload, separators=(",", ":")))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

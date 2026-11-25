#!/usr/bin/env python3
"""
Build the Cascadia catalog v4 3D atlas: one self-contained HTML file (three.js, data embedded),
in the visual language of the Cascadia Offshore Sensor Atlas (yaoderek/cascadia-sensor-atlas-preview).

Layers packed here (gzip + base64 blobs, decoded in the browser with DecompressionStream):
  events     catalog v4 (data/catalog_v4/), Route A magnitudes when present (default the converged
             file of D-B091; --mag stored = data/magnitude_v4/, see workflow/05_magnitude/mag_paths.py)
             (our Route A ML / preferred M), else Ian's magnitude column (flagged), the share of
             each event's picks added after ver3 (via ver6 pick_id), and Ian's spatial_sigma
             + every other catalog of the registry workflow/11_compare/catalogs.yaml with atlas: true
             (relocations, velocity-model iterations): lon/lat/depth offsets from the reference catalog on the
             same event order, location tier A-D/U, nphs, gap, rms, read from the common tables built by
             workflow/11_compare/compare_catalogs.py (CMP below); whole-catalog numbers from its summary CSVs
  terrain    GMT earth_relief 01m for the whole margin + the GMRT patches in data/gmrt/
  slab       Slab2 (data/slab2/cas_slab2_dep.xyz) with the coupling models in data/coupling/
             sampled onto its nodes
  stations   ver5/ver6 station table; plates: PB2002; tensors: ComCat moment tensors 2010-2016

Usage (default env, from the repo root):
  LD_LIBRARY_PATH=.pixi/envs/default/lib .pixi/envs/default/bin/python \
      workflow/06_analysis/build_v4_atlas.py
Output: data/catalog_v4/atlas/cascadia_v4_atlas.html (template: workflow/06_analysis/atlas_template.html)

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
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data"
sys.path.append(str(ROOT / "workflow" / "05_magnitude"))
import mag_paths  # noqa: E402  (converged Route A magnitudes by default, D-B091)

MAG = mag_paths.v41()        # --mag / $CASCADIA_MAG_V41: converged (default), stored or a path


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
TIERS = "ABCDU"                                  # tier codes 0-4; 5 = located, catalog without tiers

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


def events():
    c = pd.read_csv(dd("catalog_v4", "Cascadia_relocated_catalog_v4.csv"), skipinitialspace=True)
    c.columns = [x.strip() for x in c.columns]
    t = pd.to_datetime(c["datetime"].str.strip(), utc=True)
    days = ((t - T0) / pd.Timedelta(days=1)).to_numpy()
    M = c["magnitude"].to_numpy(float); Mtype = np.full(len(c), 3, "u1")      # 3 = Ian's column
    msrc = "Ian's catalog magnitude (type unknown; runs high offshore)"
    mfile = MAG
    if mfile.exists():
        m = pd.read_csv(mfile).set_index("event_id")
        mm = m.reindex(c["ID"])
        ok = mm["M"].notna().to_numpy()
        M = np.where(ok, mm["M"].to_numpy(float), np.nan)
        code = {"ML": 0, "Mw": 1, "Mw_cal": 2}
        Mtype = np.where(ok, mm["M_type"].map(code).fillna(0).to_numpy(), 255).astype("u1")
        kind = "converged, D-B091" if "magnitude_converged" in str(mfile) else mfile.name
        msrc = f"Route A preferred magnitude ({kind}), catalog v4 ({ok.sum():,} of {len(c):,} events)"
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
    order = np.argsort(arr["day"], kind="stable")   # time order: the time slider is a range in index
    arr = {k: v[order] for k, v in arr.items()}
    print(f"  events: {len(c):,}; magnitudes: {msrc}")
    return arr, dict(n=len(c), msrc=msrc, t0="2010-01-01", days_max=float(np.nanmax(days)), cats=cats)


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


def catalogs(ids: pd.Series, ref_xyz: np.ndarray) -> tuple[dict, list]:
    """Registry catalogs with atlas: true. The reference catalog (registry 'reference') gives the event set and the
    base positions (the v4 csv read in events()); every other catalog is stored on the same event order as offsets
    from the reference: int16 lon/lat in 1e-4 deg and depth in 10 m (-32768 = not located there: tier U, event
    missing, or an offset out of int16 range); tier 0-4 = A-D, U, 5 = located without a tier class; nphs; gap in
    2 deg and rms in 10 ms steps (uint8, 255 = none). Arrays are named c<k>_* with k the position in the list."""
    import yaml
    if not REGISTRY.exists():
        print(f"  catalogs: {REGISTRY} missing, reference only")
        return {}, [dict(id="v4", label="Catalog v4", key=None, status="final", ref=True, has_tiers=False, n_loc=len(ids))]
    reg = yaml.safe_load(open(REGISTRY))
    ref = reg["reference"]
    ents = {e["id"]: e for e in reg["catalogs"]}
    order = [ref] + [e["id"] for e in reg["catalogs"] if e.get("atlas") and e["id"] != ref]
    pref = pd.read_parquet(CMP / f"{ref}.parquet").set_index("v4_id") if (CMP / f"{ref}.parquet").exists() else None
    if pref is not None:                            # the reference table must hold the locations the atlas shows
        dmax = np.nanmax(np.abs(pref.reindex(ids.to_numpy())[["lat", "lon", "depth"]].to_numpy() - ref_xyz))
        if not dmax < 1e-3:
            raise SystemExit(f"{CMP / ref}.parquet differs from the v4 csv by {dmax}: rerun compare_catalogs.py")
    arr, cats = {}, []
    for k, cid in enumerate(order):
        e = ents[cid]
        meta = dict(id=cid, label=e.get("label", cid), velocity=e.get("velocity"), method=e.get("method"),
                    status=e.get("status"), date=str(e.get("date") or ""), parent=e.get("parent"),
                    ref=(k == 0), key=None, has_tiers=False, n_loc=len(ids), num=catalog_numbers(cid, ref))
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
        off = {}
        for col, s_ in (("lon", 1e4), ("lat", 1e4), ("depth", 100)):
            v = np.round((n[col] - r[col]).to_numpy(float) * s_)
            loc &= np.isfinite(v) & (np.abs(v) <= 32767)
            off[col] = v
        key = f"c{len(cats)}"
        i16 = lambda v: np.where(loc, np.nan_to_num(v), -32768).astype("<i2")
        u1 = lambda v, s_: np.where(np.isfinite(v) & loc, np.clip(np.round(v * s_), 0, 254), 255).astype("u1")
        tl = n["tier"].fillna("").astype(str).str.strip()
        has_t = bool(tl.isin(list("ABCD")).any())
        tier = tl.map({t_: j for j, t_ in enumerate(TIERS)}).fillna(5).to_numpy()
        tier = np.where(loc, np.where(tier == 4, 5, tier), 4).astype("u1")   # not located -> U
        arr.update({f"{key}_dlon": i16(off["lon"]), f"{key}_dlat": i16(off["lat"]), f"{key}_ddep": i16(off["depth"]),
                    f"{key}_tier": tier, f"{key}_nph": n["nphs"].fillna(0).clip(0, 255).to_numpy().astype("u1"),
                    f"{key}_gap": u1(n["gap"].to_numpy(float), 0.5), f"{key}_rms": u1(n["rms"].to_numpy(float), 100)})
        cnt = {TIERS[j] if j < 5 else "-": int((tier == j).sum()) for j in range(6)}
        meta.update(key=key, has_tiers=has_t, n_loc=int(loc.sum()), tiers=cnt)
        cats.append(meta)
        print(f"  catalog {cid} ({e.get('status')}): {loc.sum():,} of {len(ids):,} reference events located; tiers "
              + " ".join(f"{t_} {v:,}" for t_, v in cnt.items() if v))
    return arr, cats


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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", default=str(ROOT / "workflow" / "06_analysis" / "atlas_template.html"))
    ap.add_argument("--out", default=str(D / "catalog_v4" / "atlas" / "cascadia_v4_atlas.html"))
    mag_paths.add_argument(ap, "v4.1")
    a = ap.parse_args(argv)
    global MAG
    MAG = mag_paths.v41(a.mag)
    print(f"magnitudes: {MAG}")
    print("building atlas data")
    ev, ev_meta = events()
    ter, ter_meta = terrain()
    sl, sl_meta = slab_and_coupling()
    fa, fa_meta = faults()
    mc, mc_meta = mc_grids()
    payload = {
        "meta": dict(lon0=LON0, lat0=LAT0, region=REGION, regions=[dict(key=k, label=l, box=b) for k, l, b in REGIONS],
                     events=ev_meta, terrain=ter_meta, slab=sl_meta, faults=fa_meta, mc=mc_meta,
                     built=pd.Timestamp.now().strftime("%Y-%m-%d")),
        "events": blob(ev), "terrain": blob(ter), "slab": blob(sl), "faults": blob(fa) if fa else None,
        "mc": blob(mc) if mc else None,
        "stations": stations(), "plates": plates(), "tensors": tensors(),
    }
    html = Path(a.template).read_text().replace("/*__ATLAS_DATA__*/null", json.dumps(payload, separators=(",", ":")))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

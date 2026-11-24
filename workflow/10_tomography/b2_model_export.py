"""Track B (D-B047, proposed): versioned, cloud-native export of the velocity models.

One xarray Dataset per model version, built from a finished tomoDD run (geographic nodes after the D-B028
back-transform, interior nodes only), written as
  * Zarr v3 (cloud-native, consolidated metadata): the canonical store, <store>/<name>/v<version>.zarr
  * netCDF-4 Classic (compressed): the EMC / CRESCENT CVM container, <store>/<name>/v<version>.nc, and a copy in git
    (data/tomo_v1/models/<name>/) when it is smaller than 5 MB
  * optionally NonLinLoc SLOW_LEN grids (P and S) on the D-B005 Lambert grid for relocation (--nll): the model where it
    is defined, D-B005 elsewhere, and D-B005's water and air cells kept (bathymetry, topography)
and registered in data/tomo_v1/models/registry.csv (append-only; one row per version, sha256 of every file) with a
provenance JSON next to the netCDF. Versions are immutable: the exporter refuses to overwrite an existing version.

Variables (CRESCENT CVM names, EMC/CF attributes): Vp, Vs (km/s), VpVs, Vp_start, Vs_start (the run's starting
model), Vp_perturbation, Vs_perturbation (% from the start), dws_p, dws_s (last-iteration derivative weight sum;
0 = no ray, value = start model). Coordinates longitude, latitude, depth (km below sea level, positive down;
negative = above sea level). data_layout vertex, grid_ref latitude_longitude.

  pixi run -e models python workflow/10_tomography/b2_model_export.py RUN VERSION "DESCRIPTION" [--nll] [--status S]
      RUN      <region>/<tag> under tomodd.runs, e.g. margin_m0_v2/d200_s480
      VERSION  semantic version, e.g. 0.1.0
"""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import REPO, load_params  # noqa: E402
from b2_fwd_common import mod_velocity, read_mod  # noqa: E402

P = load_params()
T = P["tomodd"]
X = T["models"]
RUNS = Path(T["runs"])
STORE = Path(X["store"])
GIT_DIR = REPO / "data" / "tomo_v1" / "models"
REGISTRY = GIT_DIR / "registry.csv"


def sha256(path):
    h = hashlib.sha256()
    p = Path(path)
    files = sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else [p]
    for f in files:                                   # a Zarr store: hash of its files in path order
        if p.is_dir():
            h.update(str(f.relative_to(p)).encode())
        with open(f, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
    return h.hexdigest()


def git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def dws(run, shape):
    lines = (run / "Output_Files" / "tomoDD.vel").read_text().split("\n")
    out = {}
    for ph, key in (("p", "DWS for P-wave"), ("s", "DWS for S-wave")):
        starts = [i for i, l in enumerate(lines) if key in l]
        if starts:
            nx, ny, nz = shape
            a = np.array([lines[starts[-1] + 1 + r].split() for r in range(ny * nz)], float).reshape(nz, ny, nx)
            out[ph] = a.transpose(2, 1, 0)
    return out


def build(run_spec, version, description):
    run = RUNS / run_spec
    region, tag = run_spec.split("/")
    R = T["regions"][region]
    g = np.load(run / "Output_Files" / "model_geo.npz")
    s = read_mod(run / "MOD")
    sl = (slice(1, -1),) * 3                         # interior nodes (boundary nodes are tomoDD padding)
    lon, lat, z = g["lon"][1:-1], g["lat"][1:-1], g["z"][1:-1]
    vp, vs, vp0, vs0 = g["vp"][sl], g["vs"][sl], s["vp"][sl], s["vs"][sl]
    D = dws(run, g["vp"].shape)
    dims = ("depth", "latitude", "longitude")        # EMC sort order: depth, latitude, longitude
    tr = lambda a: np.ascontiguousarray(a.transpose(2, 1, 0)).astype("float32")   # [x,y,z] -> [z,y,x]
    src = f"data-derived: tomoDD-SE inversion {run_spec}"
    v = {
        "Vp": (tr(vp), dict(long_name="P-wave velocity", display_name="Vp (km/s)", units="km/s", source=src)),
        "Vs": (tr(vs), dict(long_name="S-wave velocity", display_name="Vs (km/s)", units="km/s", source=src)),
        "VpVs": (tr(vp / vs), dict(long_name="Vp/Vs ratio", display_name="Vp/Vs", units="1", source="Vp / Vs")),
        "Vp_start": (tr(vp0), dict(long_name="P-wave velocity of the starting model", display_name="Vp start (km/s)",
                                   units="km/s", source=f"starting model of {run_spec} (see start_model)")),
        "Vs_start": (tr(vs0), dict(long_name="S-wave velocity of the starting model", display_name="Vs start (km/s)",
                                   units="km/s", source=f"starting model of {run_spec} (see start_model)")),
        "Vp_perturbation": (tr(100 * (vp / vp0 - 1)), dict(long_name="Vp change from the starting model",
                                                           display_name="dVp/Vp (%)", units="percent",
                                                           source="100 (Vp / Vp_start - 1)")),
        "Vs_perturbation": (tr(100 * (vs / vs0 - 1)), dict(long_name="Vs change from the starting model",
                                                           display_name="dVs/Vs (%)", units="percent",
                                                           source="100 (Vs / Vs_start - 1)")),
    }
    for ph in ("p", "s"):
        if ph in D:
            v[f"dws_{ph}"] = (tr(D[ph][sl]), dict(
                long_name=f"derivative weight sum of {ph.upper()} rays, last iteration",
                display_name=f"DWS {ph.upper()}", units="1",
                source="tomoDD.vel; 0 = no ray: the node keeps the starting model"))
    ds = xr.Dataset(
        {k: (dims, a, at) for k, (a, at) in v.items()},
        coords={"depth": ("depth", z.astype("float64"), dict(long_name="depth below sea level", units="km",
                                                             positive="down", standard_name="depth")),
                "latitude": ("latitude", lat.astype("float64"), dict(long_name="latitude", units="degrees_north",
                                                                     standard_name="latitude")),
                "longitude": ("longitude", lon.astype("float64"), dict(long_name="longitude", units="degrees_east",
                                                                       standard_name="longitude"))})
    commit = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--", "workflow/10_tomography"))
    inp = (run / "tomoDD-SE.inp").read_text() if (run / "tomoDD-SE.inp").exists() else ""
    A = dict(X["attributes"])                        # EMC global attributes (params tomodd.models.attributes)
    A.update(
        title=f"{X['name']} v{version}: {description}",
        id=f"{X['name']}.v{version}",
        model=X["name"], version=f"v{version}", data_revision=f"r{version}", grid_dim="3D",
        summary=f"{description}. {X['summary']}",
        year=datetime.now(timezone.utc).year,
        history=f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} b2_model_export.py from {run_spec}",
        grid_ref="latitude_longitude", data_layout="vertex",
        geospatial_lat_units="degrees_north", geospatial_lon_units="degrees_east",
        geospatial_lat_min=float(lat.min()), geospatial_lat_max=float(lat.max()),
        geospatial_lon_min=float(lon.min()), geospatial_lon_max=float(lon.max()),
        geospatial_lat_resolution=float(np.round(np.median(np.diff(lat)), 6)),
        geospatial_lon_resolution=float(np.round(np.median(np.diff(lon)), 6)),
        geospatial_vertical_units="km", geospatial_vertical_positive="down",
        geospatial_vertical_min=float(z.min()), geospatial_vertical_max=float(z.max()),
        provenance_run=str(run), provenance_region=region, provenance_tag=tag,
        provenance_git_commit=commit + ("-dirty" if dirty else ""),
        provenance_start_model=str(R.get("start_model", P["paths"]["start_model"])),
        provenance_region_json=json.dumps(R, sort_keys=True),
        provenance_tomodd_inp=inp,
        provenance_model_geo_sha256=sha256(run / "Output_Files" / "model_geo.npz"),
    )
    ds.attrs = {k: (json.dumps(val) if isinstance(val, (list, dict)) else val) for k, val in A.items()}
    return ds


# EMC Metadata Inspector rules (EarthScope emc-tools src/emc_inspector.py, read 2026-11-23): required global
# attributes and the string they must start with (None = any non-empty value); required variable attributes.
EMC_REQUIRED = {"author_institution": None, "author_name": None, "Conventions": "CF-1.0", "data_revision": "r",
                "id": None, "Metadata_Conventions": "Unidata Dataset Discovery v1.0", "model": None,
                "model_type": None, "model_subtype": None, "reference_pid": "doi:", "reference": None,
                "repository_institution": "EarthScope DS", "repository_name": "EMC", "repository_pid": "doi:",
                "summary": None, "title": None, "year": None, "grid_dim": ("2D", "3D"),
                "grid_ref": "latitude_longitude", "version": "v"}
EMC_GEOSPATIAL = ["geospatial_lat_min", "geospatial_lat_max", "geospatial_lon_min", "geospatial_lon_max",
                  "geospatial_vertical_min", "geospatial_vertical_max", "geospatial_vertical_units",
                  "geospatial_vertical_positive"]


def emc_check(ds):
    """Our own check against the EMC inspector's rules (the inspector itself is not run here). Returns a list of
    problems; 'TO FILL' placeholders are reported as pending (they must be filled before an EMC submission)."""
    out = []
    for k, start in EMC_REQUIRED.items():
        v = str(ds.attrs.get(k, "")).strip()
        if not v:
            out.append(f"missing global attribute {k}")
        elif v.startswith("TO FILL"):
            out.append(f"pending (placeholder) {k}")
        elif isinstance(start, tuple) and v not in start:
            out.append(f"{k} must be one of {start}")
        elif isinstance(start, str) and not v.startswith(start):
            out.append(f"{k} must start with '{start}'")
    out += [f"missing geospatial attribute {k}" for k in EMC_GEOSPATIAL if k not in ds.attrs]
    for c in ("latitude", "longitude"):
        out += [f"{c}: missing {a}" for a in ("long_name", "units", "standard_name") if a not in ds[c].attrs]
    out += [f"depth: missing {a}" for a in ("long_name", "units", "positive") if a not in ds["depth"].attrs]
    for v in ds.data_vars:
        out += [f"{v}: missing {a}" for a in ("long_name", "units", "display_name", "source") if a not in ds[v].attrs]
    for k, c in (("lat", "latitude"), ("lon", "longitude")):
        if abs(ds.attrs[f"geospatial_{k}_min"] - float(ds[c].min())) > 0.1 or \
                abs(ds.attrs[f"geospatial_{k}_max"] - float(ds[c].max())) > 0.1:
            out.append(f"geospatial_{k} bounds differ from the {c} values")
    return out


def write(ds, version, nll):
    out = STORE / X["name"]
    out.mkdir(parents=True, exist_ok=True)
    zp, ncp = out / f"v{version}.zarr", out / f"v{version}.nc"
    if zp.exists() or ncp.exists():
        raise SystemExit(f"v{version} exists in {out}: versions are immutable, use a new version number")
    ds.to_zarr(zp, mode="w-", zarr_format=3, consolidated=True)
    enc = {k: dict(zlib=True, complevel=4) for k in ds.data_vars}
    ds.to_netcdf(ncp, format="NETCDF4_CLASSIC", encoding=enc)
    files = {"zarr": zp, "netcdf": ncp}
    if nll:
        files.update(nll_grids(ds, out / f"v{version}_nll"))
    return files


def nll_grids(ds, out):
    """NonLinLoc SLOW_LEN grids on the D-B005 grid (header copied): the model (tomoDD trilinear lookup) inside its
    node box, D-B005 outside and in D-B005's water (Vp 1.5) and air (0.3) cells."""
    from pyproj import Proj
    out.mkdir(parents=True, exist_ok=False)
    model = Path(P["paths"]["start_model"])
    files = {}
    h = (model / "cascadia3d.P.mod.hdr").read_text().split()
    nx, ny, nz = map(int, h[:3])
    x0, y0, z0, dx = map(float, h[3:7])
    keep = dx / np.fromfile(model / "cascadia3d.P.mod.buf", dtype="<f4").reshape(nx, ny, nz) < 1.6   # water, air
    proj = Proj(proj="lcc", lat_0=45.0, lon_0=-125.8, lat_1=41.0, lat_2=49.0, ellps="WGS84")   # = the NLL grid
    X_, Y_ = np.meshgrid(1e3 * (x0 + dx * np.arange(nx)), 1e3 * (y0 + dx * np.arange(ny)), indexing="ij")
    lon, lat = proj(X_, Y_, inverse=True)
    for ph, var in (("P", "Vp"), ("S", "Vs")):
        hdr = (model / f"cascadia3d.{ph}.mod.hdr").read_text()
        vbg = dx / np.fromfile(model / f"cascadia3d.{ph}.mod.buf", dtype="<f4").reshape(nx, ny, nz)  # SLOW_LEN -> v
        m = dict(lon=ds.longitude.values, lat=ds.latitude.values, z=ds.depth.values,
                 vp=ds[var].values.transpose(2, 1, 0), vs=ds[var].values.transpose(2, 1, 0))
        inside = ((lon >= m["lon"][0]) & (lon <= m["lon"][-1]) & (lat >= m["lat"][0]) & (lat <= m["lat"][-1]))
        v = vbg.copy()
        for k in range(nz):
            zk = z0 + dx * k
            if zk < m["z"][0] or zk > m["z"][-1]:
                continue
            sel = inside & ~keep[:, :, k]
            if sel.any():
                v[:, :, k][sel] = mod_velocity(m, "P", lon[sel], lat[sel], np.full(sel.sum(), zk), clamp_surface=False)
        f = out / f"model.{ph}.mod.buf"
        (dx / v).astype("<f4").tofile(f)
        (out / f"model.{ph}.mod.hdr").write_text(hdr)
        files[f"nll_{ph}"] = f
    return files


def register(ds, version, description, files, status):
    GIT_DIR.mkdir(parents=True, exist_ok=True)
    name = X["name"]
    gd = GIT_DIR / name
    gd.mkdir(exist_ok=True)
    nc_git = None
    if files["netcdf"].stat().st_size < 5 * 1024 ** 2:
        nc_git = gd / files["netcdf"].name
        nc_git.write_bytes(files["netcdf"].read_bytes())
    prov = dict(model=name, version=version, description=description, status=status,
                created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                files={k: dict(path=str(p), sha256=sha256(p)) for k, p in files.items()},
                netcdf_in_git=str(nc_git.relative_to(REPO)) if nc_git else None,
                attributes={k: v for k, v in ds.attrs.items() if not k.startswith("provenance_tomodd_inp")},
                variables=list(ds.data_vars), shape=dict(ds.sizes))
    pj = gd / f"v{version}.provenance.json"
    pj.write_text(json.dumps(prov, indent=1, default=str))
    row = dict(model=name, version=version, status=status, created=prov["created"], description=description,
               run=ds.attrs["provenance_run"], git_commit=ds.attrs["provenance_git_commit"],
               zarr=str(files["zarr"]), zarr_sha256=prov["files"]["zarr"]["sha256"],
               netcdf=str(files["netcdf"]), netcdf_sha256=prov["files"]["netcdf"]["sha256"],
               netcdf_git=prov["netcdf_in_git"] or "", nll=str(files.get("nll_P", "")).rsplit("/", 1)[0] if "nll_P" in files else "")
    reg = pd.read_csv(REGISTRY) if REGISTRY.exists() else pd.DataFrame(columns=list(row))
    if ((reg.model == name) & (reg.version.astype(str) == version)).any():
        raise SystemExit(f"{name} v{version} already registered")
    reg = pd.concat([reg, pd.DataFrame([row])], ignore_index=True)
    reg.to_csv(REGISTRY, index=False)
    reg.to_csv(STORE / name / "registry.csv", index=False)          # a copy next to the stores (tutorial, bundle)
    (STORE / name / f"v{version}.provenance.json").write_text(pj.read_text())
    return pj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("version")
    ap.add_argument("description")
    ap.add_argument("--nll", action="store_true", help="also write NonLinLoc grids on the D-B005 grid")
    ap.add_argument("--status", default="intermediate", help="intermediate | candidate | release")
    a = ap.parse_args()
    ds = build(a.run, a.version, a.description)
    issues = emc_check(ds)
    ds.attrs["emc_check"] = "; ".join(issues) if issues else "passes the EMC inspector's required-attribute rules"
    print("EMC check:", ds.attrs["emc_check"])
    hard = [i for i in issues if not i.startswith("pending")]
    if hard:
        raise SystemExit(f"EMC metadata problems: {hard}")
    files = write(ds, a.version, a.nll)
    pj = register(ds, a.version, a.description, files, a.status)
    back = xr.open_zarr(files["zarr"], consolidated=True)          # read-back check
    assert np.allclose(back.Vp.values, ds.Vp.values) and back.attrs["id"] == ds.attrs["id"]
    print(json.dumps({k: str(p) for k, p in files.items()}, indent=1))
    print(f"registered {X['name']} v{a.version} -> {REGISTRY}; provenance {pj}")


if __name__ == "__main__":
    main()

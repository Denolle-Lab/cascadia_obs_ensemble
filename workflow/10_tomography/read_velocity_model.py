# %% [markdown]
# # Reading the Cascadia OBS tomography velocity models
#
# Each model version comes as a **Zarr v3 store** (cloud-native, the main copy), a **netCDF-4 Classic** file (the
# format EarthScope EMC and the CRESCENT Cascadia CVM ask for) and, for versions meant for relocation, **NonLinLoc
# grids**. All three hold the same numbers. `registry.csv` lists every version with its source run, git commit
# and sha256 checksums; `v<version>.provenance.json` has the full provenance.
#
# Requirements: `pip install xarray zarr netCDF4 scipy matplotlib` (Python >= 3.11 for zarr 3), or in this repository
# `pixi run -e models python workflow/10_tomography/read_velocity_model.py [path to the model folder]`.
#
# Run as a script or open as a notebook (VS Code and JupyterLab with jupytext read the `# %%` cells).

# %%
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import xarray as xr

if not hasattr(sys, "ps1") and "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Folder with the v<version>.zarr / .nc files (registry.csv there or one level up): the unzipped Zenodo bundle's
# velocity_models/ (or its cascadia_obs_tomo/), the model store, or data/tomo_v1/models/cascadia_obs_tomo in git.
# Command-line arguments are read only in a script: a Jupyter kernel's argv is "-f <connection file>".
# In a notebook, set ROOT and FIGS here.
args = [] if "ipykernel" in sys.modules else [a for a in sys.argv[1:] if not a.startswith("-")]
ROOT = Path(args[0]) if args else Path("/wd1/mdenolle_data/tomo_v1/models/cascadia_obs_tomo")
FIGS = Path(args[1]) if len(args) > 1 else Path(".")
if not any(ROOT.glob("v*.zarr")) and not any(ROOT.glob("v*.nc")) and (ROOT / "cascadia_obs_tomo").is_dir():
    ROOT = ROOT / "cascadia_obs_tomo"                # the bundle's top folder

# %% [markdown]
# ## 1. Which versions exist

# %%
reg_file = ROOT / "registry.csv" if (ROOT / "registry.csv").exists() else ROOT.parent / "registry.csv"
if reg_file.exists():
    reg = pd.read_csv(reg_file, dtype={"version": str})
    print(reg[["model", "version", "status", "created", "description"]].to_string(index=False))
versions = {m.group(1) for p in ROOT.glob("v*") if (m := re.fullmatch(r"v(\d+(?:\.\d+)*)\.(?:zarr|nc)", p.name))}
if not versions:
    sys.exit(f"no v<version>.zarr or v<version>.nc in {ROOT}: pass the folder that holds them "
             "(e.g. velocity_models/cascadia_obs_tomo)")
version = max(versions, key=lambda v: tuple(map(int, v.split("."))))   # newest, by number; or set e.g. "0.1.0"
print("using version", version)

# %% [markdown]
# ## 2. Open a version
#
# Zarr (lazy, chunked; the same call works on a remote store, e.g. `xr.open_zarr("s3://bucket/v0.1.0.zarr")` or a
# zipped store over HTTP with fsspec: `xr.open_zarr("zip::https://.../v0.1.0.zarr.zip")`).
# The netCDF file opens the same way with `xr.open_dataset("v0.1.0.nc")`.

# %%
zp = ROOT / f"v{version}.zarr"
ds = xr.open_zarr(zp) if zp.exists() else xr.open_dataset(ROOT / f"v{version}.nc")   # git keeps only the netCDF
print(ds)
print("\n", ds.attrs["title"])
print("source run:", ds.attrs["provenance_run"], "| git commit:", ds.attrs["provenance_git_commit"])

# %% [markdown]
# ## 3. Conventions
#
# * `depth` is in km **below sea level**, positive down; negative depths are above sea level (land nodes).
# * The grid is the tomoDD node grid (interior nodes): regular in longitude and latitude, uneven in depth.
#   Values between nodes: trilinear interpolation, which is what tomoDD itself uses.
# * `Vp`, `Vs` in km/s; `VpVs`; `Vp_start`, `Vs_start` = the model the inversion started from;
#   `Vp_perturbation`, `Vs_perturbation` = change from that start in %.
# * `dws_p`, `dws_s` = derivative weight sum (ray density) of the last iteration. **0 means no ray reached the
#   node: its value is the starting model, not data.** Where resolution tests exist, later versions add a mask.
# * Merged versions (0.3.x) add `resolution_p`, `resolution_s` (finest resolved checkerboard cell in km; 0 = not
#   resolved, do not interpret) and `n_tiles`. From the first version after 0.3.0 (D-B049, D-B051) they also add
#   * `tile_weight_p`, `tile_weight_s` = R, the share of the tiles in Vp and Vs (1 on nodes the tiles' checkerboard
#     tests resolve, tapering to 0 around them). **0 means the node holds the whole-margin reference model M0, not
#     tile data: do not interpret it as a tile result.**
#   * `vs_reverted` (int8 flag): 0 = Vs unchanged, 1 = Vs set back to the reference Vs because Vp/Vs < `vpvs_min` or
#     `dws_s` < `dws_s_min`, 2 = set back and then lowered to Vp / `vpvs_min` (Vp/Vs was still too low).
#   * global attributes `vpvs_min`, `dws_s_min`, the node counts `vs_check_nodes`, `vs_reverted_nodes`,
#     `vs_reverted_vpvs_nodes`, `vs_reverted_dws_s_nodes`, `vs_capped_nodes`, and `excluded_tiles`.
# * From v0.3.2 on: the solid surface and the shallow oceanic crust rule (D-B068, D-B073).
#   * `medium` (int8): 0 rock, 1 water, 2 air, from the merged solid surface (GMRT multibeam patches, GMT earth_relief
#     01m elsewhere) sampled at the node: a node above the surface is water if it is at or below sea level
#     (depth >= 0), air otherwise. **`Vp` is 1.5 km/s in water and NaN in air; `Vs` and `VpVs` are NaN in water and
#     air.** `surface_elevation` (km, positive up) is that surface at the nodes; `offshore` the rule's offshore mask.
#   * `Vs` at rock nodes in offshore columns, below sea level and at most `shallow_crust_thickness_km` (2.5 km) below
#     the seafloor: min(Vs, Vp / `shallow_crust_vpvs`) (2.1); `shallow_crust_node` = 1 where this lowered Vs.
#   * `Vp_tomography`, `Vs_tomography`: the inversion's values at every node, unchanged. The inversion had no surface,
#     so above the solid surface these are rock velocities that describe no real medium: use `Vp`/`Vs` instead.
#   * The node values carry a 2-km-node approximation of the rule. The NonLinLoc grids `v<version>_nll` carry the
#     exact rule on 1 km cells: they are the velocity fields the v5.1 relocation used. `velocity_model_tools.py`
#     evaluates the same rule at any point (section 6b).

# %%
for v in ds.data_vars:
    print(f"{v:16s} {ds[v].attrs.get('units', ''):8s} {ds[v].attrs.get('long_name', '')}")

# %% [markdown]
# ## 4. A depth slice, with the unsampled nodes greyed out
#
# In merged versions with `tile_weight_p`, nodes where it is 0 hold the reference model M0, not tile data: they are
# greyed out too.

# %%
z = 15.0
sl = ds.sel(depth=z, method="nearest")
ok = sl.dws_p > 0
if "medium" in sl:                                   # v0.3.2 on: water and air nodes are not rock
    ok = ok & (sl.medium == 0)
if "tile_weight_p" in sl:
    ok = ok & (sl.tile_weight_p > 0)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
for a, var, cmap in ((ax[0], "Vp", "RdBu"), (ax[1], "VpVs", "RdBu_r")):
    da = sl[var].where(ok)
    da.plot(ax=a, cmap=cmap, robust=True)
    a.set_aspect(1 / np.cos(np.radians(45)))
    a.set_title(f"{var} at {float(sl.depth):g} km (grey: no ray" + (" or M0)" if "tile_weight_p" in sl else ")"))
    a.set_facecolor("0.85")
fig.tight_layout()
fig.savefig(FIGS / f"tutorial_slice_{version}.png", dpi=120)

# %% [markdown]
# ## 5. An east-west section

# %%
sec = ds.sel(latitude=47.5, method="nearest").sel(depth=slice(0, 60))
fig, ax = plt.subplots(figsize=(10, 3.5))
sec.Vp.where(sec.dws_p > 0).plot(ax=ax, cmap="RdBu", vmin=4.5, vmax=8.2, yincrease=False)
ax.set_title(f"Vp, E-W at {float(sec.latitude):.1f} N")
fig.tight_layout()
fig.savefig(FIGS / f"tutorial_section_{version}.png", dpi=120)

# %% [markdown]
# ## 6. Velocities at your own points (stations, hypocentres)
#
# `interp` with `method="linear"` on all three axes is trilinear interpolation.

# %%
pts = pd.DataFrame({"lon": [-128.0, -124.5, -122.3], "lat": [44.5, 41.0, 47.6], "depth": [8.0, 20.0, 45.0]})
q = ds[["Vp", "Vs", "dws_p"]].interp(longitude=xr.DataArray(pts.lon, dims="pt"),
                                     latitude=xr.DataArray(pts.lat, dims="pt"),
                                     depth=xr.DataArray(pts.depth, dims="pt"), method="linear")
pts["Vp"], pts["Vs"], pts["sampled"] = q.Vp.values, q.Vs.values, q.dws_p.values > 0
print(pts.round(3).to_string(index=False))

# %% [markdown]
# ## 6b. Velocities with the solid surface and the shallow-crust rule (v0.3.2 and later)
#
# `velocity_model_tools.velocity_at` (shipped next to this tutorial) returns Vp, Vs, Vp/Vs and the medium at any
# points. `mode="field"` evaluates the rule of the v5.1 relocation's velocity field at the point: water (Vp 1.5,
# Vs NaN) or air (NaN) above the merged solid surface; below it the inversion's values (trilinear, as tomoDD) with
# Vs <= Vp / 2.1 within 2.5 km below the seafloor offshore. At the centres of the 1 km NonLinLoc cells it equals the
# NonLinLoc grids (checked to 1e-7 relative at 50,000 cells). `mode="nll"` returns the NonLinLoc cell values
# themselves (piecewise constant, including the D-B005 model outside the tomography volume and water Vs 0.4 km/s);
# `mode="nodes"` interpolates the stored node values (only the 2-km-node approximation; NaN or water mixed in within
# one node spacing of the surface).

# %%
sys.path.insert(0, str(Path(__file__).resolve().parent) if "__file__" in globals() else ".")
if "medium" in ds:
    from velocity_model_tools import velocity_at
    prof = pd.DataFrame({"depth": np.arange(0.0, 12.01, 0.5)})
    for mode in ("field", "nll", "nodes"):
        r = velocity_at(ROOT, version, np.full(len(prof), 45.95), np.full(len(prof), -130.0), prof.depth.values, mode=mode)
        prof[f"Vs_{mode}"] = r["Vs"]
        if mode == "field":
            prof["medium"] = r["medium"]
    print("Axial Seamount, 45.95 N 130.0 W (medium 0 rock, 1 water, 2 air):")
    print(prof.round(3).to_string(index=False))

# %% [markdown]
# ## 7. A 1D profile, for codes that need a layered model (hypoDD, GrowClust)

# %%
box = ds.sel(longitude=slice(-125.0, -123.0), latitude=slice(44.0, 46.0))
use = (box.dws_p > 0) & (box.medium == 0) if "medium" in box else box.dws_p > 0   # rock nodes with rays
prof = box[["Vp", "Vs"]].where(use).median(["latitude", "longitude"]).to_dataframe()
print(prof.round(3))

# %% [markdown]
# ## 8. Relocation with NonLinLoc
#
# Versions exported with `--nll` have `v<version>_nll/model.P.mod.buf|hdr` and `model.S.mod.buf|hdr`: NonLinLoc
# 3D grids (SLOW_LEN = slowness x node spacing, float32, Lambert conformal projection given in the header, 1 km
# nodes, depth -5 to 80 km). Inside the tomography volume they hold this model; outside it, and in the water and
# air cells, the 3D starting model of the v4-NLL3D catalog (D-B005). In a NonLinLoc control file, point
# Grid2Time at them, for example
#
#     GTFILES  <path>/v0.1.0_nll/model  <path>/time/layer  P
#     GTFILES  <path>/v0.1.0_nll/model  <path>/time/layer  S
#     TRANS  LAMBERT  WGS-84  45.0  -125.8  41.0  49.0  0.0
#
# Note: the grids are true velocities on a flat (Lambert) grid. For long paths, apply the same Earth-flattening and
# grid-refinement corrections as the v4-NLL3D processing (see notes/forward_accuracy.md, D-B028) before Grid2Time.
#
# From v0.3.2 on the grids are the exact fields of the v5.1 relocation: cell-centred and **Earth-flattened** (depth
# axis z' = Re ln(Re / (Re - z)), velocity v' = v Re / (Re - z), Re = 6371 km), with water and air cells from the
# merged solid surface and the D-B073 rule (see `v<version>_nll/model.P.mod.README`). Run Grid2Time on them with the
# station depths flattened the same way, and map located depths back with z = Re (1 - exp(-z' / Re)).

# %%
nll = ROOT / f"v{version}_nll"
if (nll / "model.P.mod.hdr").exists():
    h = (nll / "model.P.mod.hdr").read_text().split()
    nx, ny, nz = map(int, h[:3])
    x0, y0, z0, dx = map(float, h[3:7])
    vp = dx / np.fromfile(nll / "model.P.mod.buf", dtype="<f4").reshape(nx, ny, nz)
    print(f"NonLinLoc P grid {nx} x {ny} x {nz}, origin ({x0}, {y0}, {z0}) km, spacing {dx} km;"
          f" Vp {vp.min():.2f}-{vp.max():.2f} km/s")
else:
    print("no NonLinLoc grids for this version (regenerate with b2_model_export.py --nll)")

# %% [markdown]
# ## 9. Check a file against the registry

# %%
import hashlib  # noqa: E402

row = reg[reg.version == version] if reg_file.exists() else []
if len(row) and (ROOT / f"v{version}.nc").exists():
    h = hashlib.sha256((ROOT / f"v{version}.nc").read_bytes()).hexdigest()
    print("netCDF sha256 matches the registry:", h == row.netcdf_sha256.iloc[0])
else:
    print(f"v{version}: no registry row or no netCDF file to check")

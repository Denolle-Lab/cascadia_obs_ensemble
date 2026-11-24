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
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import xarray as xr

if not hasattr(sys, "ps1") and "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Folder with registry.csv and the v<version>.zarr / .nc files: the unzipped Zenodo bundle, or the model store.
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else \
    Path("/wd1/mdenolle_data/tomo_v1/models/cascadia_obs_tomo")
FIGS = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")

# %% [markdown]
# ## 1. Which versions exist

# %%
reg_file = ROOT / "registry.csv" if (ROOT / "registry.csv").exists() else ROOT.parent / "registry.csv"
if reg_file.exists():
    reg = pd.read_csv(reg_file)
    print(reg[["model", "version", "status", "created", "description"]].to_string(index=False))
version = sorted(p.name[1:-5] for p in ROOT.glob("v*.zarr"))[-1]      # newest version; or set e.g. "0.1.0"
print("using version", version)

# %% [markdown]
# ## 2. Open a version
#
# Zarr (lazy, chunked; the same call works on a remote store, e.g. `xr.open_zarr("s3://bucket/v0.1.0.zarr")` or a
# zipped store over HTTP with fsspec: `xr.open_zarr("zip::https://.../v0.1.0.zarr.zip")`).
# The netCDF file opens the same way with `xr.open_dataset("v0.1.0.nc")`.

# %%
ds = xr.open_zarr(ROOT / f"v{version}.zarr")
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

# %%
for v in ds.data_vars:
    print(f"{v:16s} {ds[v].attrs.get('units', ''):8s} {ds[v].attrs.get('long_name', '')}")

# %% [markdown]
# ## 4. A depth slice, with the unsampled nodes greyed out

# %%
z = 15.0
sl = ds.sel(depth=z, method="nearest")
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
for a, var, cmap in ((ax[0], "Vp", "RdBu"), (ax[1], "VpVs", "RdBu_r")):
    da = sl[var].where(sl.dws_p > 0)
    da.plot(ax=a, cmap=cmap, robust=True)
    a.set_aspect(1 / np.cos(np.radians(45)))
    a.set_title(f"{var} at {float(sl.depth):g} km (grey: no ray)")
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
# ## 7. A 1D profile, for codes that need a layered model (hypoDD, GrowClust)

# %%
box = ds.sel(longitude=slice(-125.0, -123.0), latitude=slice(44.0, 46.0))
prof = box[["Vp", "Vs"]].where(box.dws_p > 0).median(["latitude", "longitude"]).to_dataframe()
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

if reg_file.exists():
    row = reg[reg.version.astype(str) == version].iloc[0]
    h = hashlib.sha256((ROOT / f"v{version}.nc").read_bytes()).hexdigest()
    print("netCDF sha256 matches the registry:", h == row.netcdf_sha256)

# Cascadia OBS tomography: velocity models (versioned)

Three-dimensional Vp and Vs models of the Cascadia margin and the Juan de Fuca and Gorda plates from
double-difference local-earthquake tomography (tomoDD-SE) of the Cascadia OBS ensemble catalog (ocean-bottom and
land picks). Built in `workflow/10_tomography` (Track B); decisions in `workflow/10_tomography/DECISIONS.md`.

**Status: working copy, not for redistribution.** Versions below 1.0 are intermediate steps of the inversion. The
public release (version 1.0 or later) will follow the peer-reviewed paper; EarthScope EMC and the CRESCENT
Cascadia Community Velocity Model both require one.

## Versions

`registry.csv` lists every version: status, source run, git commit, and sha256 of each file. A version is never
overwritten; a change gives a new version number.

| version | what | status |
|---|---|---|
| 0.1.0 | M0: coarse whole-margin model (nodes ~0.375 x 0.3 deg), damping 200, smoothing 480 (D-B044, D-B046); the common starting model of the tiles | intermediate |

Planned: 0.2.x the tiles restarted from M0 (D-B046), 0.3.x the merged model (notes/merge_design.md), 1.0 the merged
ensemble model with uncertainties and resolution masks.

## Files of a version

* `v<version>.zarr`: Zarr v3 store with consolidated metadata, the main copy (cloud-native, lazy reads).
* `v<version>.nc`: the same Dataset as netCDF-4 Classic, compressed, with the global attributes of the EarthScope EMC
  model header (CF-1.0, Unidata Dataset Discovery) and the CRESCENT CVM variable names (`Vp`, `Vs`,
  `Vp_perturbation`, ...). Attributes marked `TO FILL` (authors, paper reference and DOI) are filled at release.
* `v<version>.provenance.json`: source run, git commit, region definition, tomoDD input file, checksums.
* `v<version>_nll/` (versions exported for relocation): NonLinLoc SLOW_LEN grids `model.P|S.mod.buf|hdr` on the
  1 km Lambert grid of the v4-NLL3D starting model.

## Reading a model

See `read_velocity_model.py` (runs as a script or as a notebook): open the Zarr or netCDF with xarray, depth slices,
sections, values at your own points, a 1D profile, the NonLinLoc grids, and a checksum test. In short:

```python
import xarray as xr
ds = xr.open_zarr("cascadia_obs_tomo/v0.1.0.zarr")          # or xr.open_dataset("cascadia_obs_tomo/v0.1.0.nc")
vp = ds.Vp.interp(longitude=-124.5, latitude=44.5, depth=20.0)   # trilinear, as tomoDD
sampled = ds.dws_p > 0                                             # 0 = no ray: starting model, not data
```

Depth is in km below sea level, positive down (negative = above sea level).

## Making a new version

```
pixi run -e models python workflow/10_tomography/b2_model_export.py <region>/<tag> <version> "<description>" [--nll]
```

It writes the Zarr store, the netCDF and (with `--nll`) the NonLinLoc grids to the model store
(`params.yaml` tomodd.models.store), checks the netCDF header against the EMC inspector's required attributes, copies
the netCDF into this folder when it is under 5 MB, and appends `registry.csv`.

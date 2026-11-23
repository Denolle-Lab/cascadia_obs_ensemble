# Forward accuracy: tomoDD-SE vs NonLinLoc (Mendocino-Gorda)

Date: 2026-11-22. Scripts: `b2_fwd_*.py` (this folder). Data and grids:
`/wd1/mdenolle_data/tomo_v1/b2_tomodd/fwd_test/` (all subset time grids were deleted after use; the sampled
times are kept in `nll/*.csv`, 735 MB in all). Tables in full: `fwd_test/compare/{decomposition,options}_tables.md`.

## 1. The finding and the short answer

check01 (`mendocino_gorda/check01/raytracer_vs_nll.csv`): tomoDD predicted time minus NLL predicted time,
same picks, same v4-NLL3D hypocentres, same nominal model: P median -0.16 s, S -0.26 s, growing with
distance. tomoDD looks faster.

**Most of it is NLL being too slow, not tomoDD being too fast.** The production NLL time grids
(`reloc_v5/time_vpvs176`) are slower than the exact times in their own model by a median 0.21 s (P) and
0.37 s (S) in this subset: about 1.5 to 2 % of the travel time. tomoDD is within +0.02 s (P) / +0.03 s (S)
of the exact times in its own MOD, and its MOD is 0.03 s (P) / 0.05 s (S) slower than the NLL model.

Three things make NLL slow (medians, P / S, all picks of the subset):

| NLL cause | P | S | distance trend |
|---|---|---|---|
| half-cell shift: a1_build_model.py evaluates the model at the grid nodes, but Podvin-Lecomte applies the value stored at node (i,j,k) to the whole cell [i,i+1] x [j,j+1] x [k,k+1], so the model is moved 0.5 km down (and east and north); velocity increases with depth, so it is slower | +0.08 | +0.15 | flat |
| Podvin-Lecomte FD error at 1 km cells (gradient media: rays bend, the scheme is first order) | +0.10 | +0.16 | grows (0.08 to 0.15 s P) |
| flat Earth (no Earth-flattening transform in the LAMBERT grid) | +0.03 | +0.05 | 0 near, +0.09 / +0.15 at 150-200 km |

## 2. Method

**Subset** (`b2_fwd_subset.py`): 12 stations (land at sea level B932.PB, JCC.BK, B045.PB, B049.PB; land at
1.5-1.9 km KHBB.NC, KHMB.NC, KCS.NC; OBS FS04D (0.16 km bsl), FS07D (1.27), FS13D (2.29), G10D (3.02),
GB281 (3.12)); up to 40 events per station and distance bin (0-25, 25-50, 50-100, 100-150, 150-200 km),
random (seed 0): 1,961 events, 17,348 picks (8,591 P, 8,757 S). Observed times and hypocentres from the B2a
inputs. The production NLL time grids read at the hypocentres (trilinear, as NLLoc) reproduce the check01
NLL predictions to 0.1 ms, and the subset reproduces the finding (P -0.18 s, S -0.30 s; a little larger than
the whole region because the subset is spread evenly over distance).

**tomoDD forward times** (`b2_fwd_tomodd.py`): one-iteration tomoDD-SE runs (production binary), predicted
time = observed - initial residual from `tomoDD-SE.res.ini`. Same numbers as check01 (max 5 ms; median 0).

**NLL times** (`b2_fwd_nll.py`): Grid2Time (Podvin-Lecomte, `GT_PLFD 1e-3`) per station and phase on
velocity fields written by Python, LAMBERT frame of the v4-NLL3D grid, z -5 to 70 km, cell velocity taken
at the cell centre (fixes the half-cell shift), optionally with the Earth-flattening transform (EFT:
z_f = R ln(R/(R - z)), v_f = v R/(R - z), R = 6371 km; source and event depths transformed). Models:
`orig` (production node values, nearest node = blocky), `origlin` (trilinear between the NLL nodes: the
NLL model as a continuous field), `mod` (inputs/MOD through a re-implementation of tomoDD's `cellvel`/`vel3`
lookup: trilinear in lon, lat, depth, Vs = Vp/(Vp/Vs) at nodes, depth clamped at 0), and MOD variants
(`b2_fwd_mods.py`). Cells 1, 0.5 and 0.25 km.

**References ("exact" times).**
* Analytic (`b2_fwd_pltest.py`): linear gradient 5 + 0.03 z km/s and a layer over a half-space. Grid2Time
  error, cell-centred, gradient model: +0.33 % of the travel time at 1 km cells, +0.21 % at 0.5 km, +0.12 %
  at 0.25 km (0.03-0.06 s at 1 km); with the production node convention +0.59 % at 1 km. The layered model
  (no gradient) is exact to <= 0.01 s. So the PL error is a gradient effect that converges slowly (order ~0.8).
* 1D, spherical, exact (`b2_fwd_taup.py`): ObsPy TauP in the laterally homogeneous MOD `oned` (median of
  the MOD per depth, tomoDD's linear-in-depth velocity, constant above sea level), receivers at their true
  depth/elevation; 98.4 % of picks (TauP fails for 276 source-receiver depth pairs).
* 3D: Richardson extrapolation of the cell-centred EFT Grid2Time times at 0.5 and 0.25 km,
  T0 = T0.25 - k (T0.5 - T0.25), k calibrated on the 1D model against TauP (k = 1.33 P, 1.28 S, i.e. order
  0.81 / 0.84). Check in 1D: extrapolated minus TauP median 0.000 s, p95 |d| 0.031 s (P) / 0.050 s (S).
  `ref_orig` = continuous NLL model, `ref_mod` = check01 MOD; other MODs: their 0.5 km times minus the
  FD error of `mod` (T0.5 - T0 of the MOD).

**WGS-84 vs sphere in tomoDD.** tomoDD-SE puts geographic latitudes on a 6371 km sphere (the
ellipticity factor is commented out). At 41 N this makes N-S distances 0.13 % too long and E-W distances
0.26 % too short. Test: `ellipsoid_coords()` in `b2_fwd_mods.py` rescales event, station and MOD-node
coordinates about the region centre, lat' - 41.3 = (M/R)(lat - 41.3), lon' + 125 = (N/R)(lon + 125),
so sphere distances equal WGS-84 distances (residual scale error < 2e-4 within 2 deg); runs `*_ellcoord`.

## 3. Results

Medians in s, a - b; columns: phase, station type, epicentral distance bin (km). Full tables (with n and
p95 per bin) in `fwd_test/compare/decomposition_tables.md`.

### 3.1 Split of the finding

finding = (b) - (c) + (a), each term per pick (medians are not exactly additive).

| | P land 0-25 | P land 50-100 | P land 150-200 | P OBS 0-25 | P OBS 50-100 | P OBS 150-200 | P all | P p95 \|d\| |
|---|---|---|---|---|---|---|---|---|
| finding: tomoDD(MOD) - NLL production | -0.126 | -0.205 | -0.273 | +0.012 | -0.127 | -0.217 | -0.180 | 0.284 |
| (c) NLL production - ref_orig | +0.147 | +0.241 | +0.299 | +0.143 | +0.173 | +0.288 | +0.213 | 0.320 |
| (c1) half-cell shift | +0.065 | +0.100 | +0.080 | +0.063 | +0.065 | +0.092 | +0.082 | 0.136 |
| (c2) flat Earth (1 km, flat - EFT) | +0.005 | +0.034 | +0.086 | +0.004 | +0.030 | +0.080 | +0.028 | 0.081 |
| (c3) PL FD error, 1 km | +0.077 | +0.115 | +0.151 | +0.056 | +0.076 | +0.098 | +0.096 | 0.174 |
| (a) MOD sampling: ref_mod - ref_orig | -0.007 | +0.029 | +0.053 | +0.046 | +0.033 | +0.048 | +0.027 | 0.124 |
| (b) tomoDD(MOD) - ref_mod | +0.020 | +0.012 | -0.021 | +0.070 | +0.014 | -0.008 | +0.016 | 0.080 |
| (b1) sphere coordinates | -0.001 | -0.014 | -0.044 | -0.001 | -0.013 | -0.047 | -0.010 | 0.043 |
| (b2) ray tracer | +0.021 | +0.021 | +0.020 | +0.071 | +0.021 | +0.026 | +0.025 | 0.092 |

| | S land 0-25 | S land 50-100 | S land 150-200 | S OBS 0-25 | S OBS 50-100 | S OBS 150-200 | S all | S p95 \|d\| |
|---|---|---|---|---|---|---|---|---|
| finding | -0.223 | -0.347 | -0.464 | -0.012 | -0.213 | -0.400 | -0.296 | 0.493 |
| (c) NLL production - ref_orig | +0.259 | +0.419 | +0.526 | +0.258 | +0.336 | +0.501 | +0.369 | 0.551 |
| (c1) half-cell shift | +0.118 | +0.177 | +0.170 | +0.123 | +0.139 | +0.180 | +0.154 | 0.248 |
| (c2) flat Earth | +0.009 | +0.060 | +0.149 | +0.008 | +0.054 | +0.135 | +0.047 | 0.139 |
| (c3) PL FD error, 1 km | +0.133 | +0.194 | +0.244 | +0.103 | +0.136 | +0.172 | +0.160 | 0.297 |
| (a) MOD sampling | -0.013 | +0.039 | +0.066 | +0.077 | +0.063 | +0.089 | +0.046 | 0.386 |
| (b) tomoDD(MOD) - ref_mod | +0.041 | +0.026 | -0.024 | +0.136 | +0.032 | -0.028 | +0.033 | 0.210 |
| (b1) sphere coordinates | -0.002 | -0.022 | -0.072 | -0.002 | -0.023 | -0.067 | -0.017 | 0.074 |
| (b2) ray tracer | +0.040 | +0.044 | +0.039 | +0.135 | +0.042 | +0.030 | +0.050 | 0.225 |

Reading: the check01 "OBS half the bias near the station" is tomoDD being slow near OBS (b2 +0.07 P,
+0.14 S at 0-25 km) plus the MOD being slow near OBS (a +0.05 / +0.08), partly cancelling the NLL error.

### 3.2 The tomoDD ray tracer

* 1D against TauP (sphere distances, tracer only): land stations 0.000 to +0.004 s (P), +0.007 (S) at all
  distances: pseudo-bending is exact for smooth paths. Departures are near-station and depend on the
  station: deep OBS +0.04-0.05 s P / +0.07-0.09 S (GB281, G10D, FS13D), high land stations -0.01 to -0.04 P.
* Cause (from the code, `RaySPDR2.f`): travel time is the trapezoid sum of endpoint slownesses over
  segments that stop being halved below 10 km (`mins = 10`, so final segments are 5-10 km). Where the
  velocity curves strongly over one segment (top 5-7 km, seafloor and coast) the trapezoid is biased: slow
  if slowness drops fast then flattens (below a seafloor station), fast if it is flat then drops (above
  sea level, where `cellvel` uses the z = 0 velocity). The bending minimises that biased sum.
* 3D (proposed MOD, scaled coordinates, tracer only) per station, median s:

| station | B045 | B049 | B932 | JCC | KCS | KHBB | KHMB | FS04D | FS07D | FS13D | G10D | GB281 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P | +0.044 | +0.058 | +0.063 | +0.054 | -0.006 | -0.024 | -0.013 | +0.012 | +0.096 | +0.119 | +0.016 | +0.007 |
| S | +0.081 | +0.098 | +0.114 | +0.108 | -0.014 | -0.042 | -0.024 | +0.023 | +0.238 | +0.283 | +0.029 | +0.010 |

  The error is nearly constant with distance for each station (it sits on the last segment). After removing a
  per station-phase median: P median 0.000, p95 |d| 0.049 s (all bins within -0.007..+0.010); S median 0.000,
  p95 0.105 s. That meets the D-B015-like target for P; S is 2x P as expected from Vp/Vs.
* Sphere coordinates (b1): -0.010 s P overall, -0.045 s at 150-200 km (E-W paths, the region's main
  direction). Removed exactly by the coordinate rescaling.

### 3.3 MOD sampling (against the continuous NLL model; `options_tables.md`)

| MOD | P all | P land 150-200 | P OBS 0-25 | S all | S OBS 0-25 |
|---|---|---|---|---|---|
| as built (velocity mean, 11 km box, depth nodes 2-9 km apart) | +0.027 | +0.053 | +0.046 | +0.046 | +0.077 |
| slowness mean 11 km | +0.028 | +0.054 | +0.054 | +0.049 | +0.087 |
| slowness mean 5 km | +0.028 | +0.053 | +0.045 | +0.047 | +0.072 |
| nearest column (no box) | +0.018 | +0.048 | +0.018 | +0.031 | +0.034 |
| slowness 5 km + depth nodes 1 km to 9 km, 2 km to 49 km | +0.015 | +0.027 | +0.048 | +0.030 | +0.074 |
| **nearest column + depth nodes 1 km to 9 km, 2 km to 49 km (`pointz2`)** | **+0.012** | **+0.028** | **+0.017** | **+0.017** | **+0.033** |
| nearest column, 1 km depth nodes everywhere (NLL only; > 50 depth nodes) | +0.008 | +0.022 | +0.016 | +0.011 | +0.030 |

* Velocity vs slowness averaging: no effect (< 0.003 s). The lateral model is smooth over 11 km.
* Box averaging hurts only near OBS: offshore columns that are water at a node depth are filled with the
  shallowest rock of the box, and on the slope that rock is the slow young-ocean top layer of deeper columns
  (FS07D: MOD Vs 1.88 km/s at 1 km where the NLL model has 2.0; S +0.5 s at FS07D, which is most of the
  S p95 of 0.39-0.46 s).
* The distance-growing part (+0.05 s P at 150-200 km) is the depth interpolation between nodes 4-15 km
  apart (Moho and lower crust): 2 km nodes to 49 km halve it. The rest (+0.02-0.03 s) is the 0.125 x 0.1 deg
  lateral node spacing.

### 3.4 NLL set-ups (against the continuous NLL model)

| NLL time grids | P all | P 150-200 (land / OBS) | P p95 | S all | S p95 |
|---|---|---|---|---|---|
| production (node value on cell [k,k+1], 1 km, flat) | +0.213 | +0.299 / +0.288 | 0.320 | +0.369 | 0.551 |
| cell-centred 1 km, flat | +0.131 | +0.239 / +0.187 | 0.242 | +0.222 | 0.413 |
| cell-centred 1 km, EFT | +0.104 | +0.158 / +0.114 | 0.181 | +0.174 | 0.309 |
| cell-centred 0.5 km, EFT | +0.062 | +0.088 / +0.070 | 0.127 | +0.104 | 0.215 |
| cell-centred 0.25 km, EFT | +0.035 | +0.050 / +0.040 | 0.072 | +0.058 | 0.121 |
| Richardson 1 km + 0.5 km, EFT: T0.5 - 1.33 (T1 - T0.5) | +0.014 | +0.001 / +0.015 | 0.076 | +0.022 | 0.139 |

### 3.5 End to end

| | P all | P p95 | S all | S p95 |
|---|---|---|---|---|
| tomoDD as check01 - NLL production (the finding) | -0.180 | 0.284 | -0.296 | 0.493 |
| tomoDD proposed (pointz2, scaled coords) - NLL production | -0.163 | 0.256 | -0.266 | 0.439 |
| tomoDD proposed - NLL cell-centred 0.5 km EFT | -0.020 | 0.125 | -0.022 | 0.512 |
| tomoDD proposed + station-phase tracer correction - ref_orig | +0.019 | 0.117 | +0.031 | 0.392 |
| tomoDD proposed + station-phase tracer correction - NLL Richardson 1 km/0.5 km EFT | -0.003 | 0.155 | +0.001 | 0.334 |

The large S p95 values come from FS07D (MOD near the shelf edge, section 3.3) and are station-constant.

## 4. Consequences

* **v4-NLL3D (Track A) was computed with the slow grids.** Its predicted times are ~1.5-2 % too long
  (P +0.15 s at 0-25 km, +0.30 s at 150-200 km). Station terms, origin times and probably depths absorbed
  part of this; the D-B020 near-event OBS S statics (from v4-NLL3D residuals at short distance) carry the
  NLL error at 0-25 km (S +0.26 s, here), so they are too small by about that much. The v4 depth bias against
  ANSS noted in track-a-relocation may be partly this (not tested here).
* tomoDD itself is close to exact; its model will fit the data with tomoDD's forward. Taking that model back
  to NLL with the production Grid2Time settings would reintroduce a 0.2-0.4 s slow bias, which B4 station
  terms would absorb as fake statics, growing with distance.

## 5. Proposed decisions

* **P-1 NLL grids for B4 (and any v4-NLL3D rerun): cell-centred velocities + Earth-flattening + Richardson.**
  Build the slowness of each Podvin-Lecomte cell at the cell centre (or shift the grid origin by -dx/2 in x,
  y, z relative to the node definition); apply the EFT to depths and velocities; compute each station's
  time grid at 1 km and 0.5 km and store T = T0.5 - 1.33 (T1 - T0.5) on the 1 km nodes (storage as now;
  0.5 km compute ~8x per grid). Median error +0.014 s P / +0.022 s S (from +0.21 / +0.37). If the 0.5 km
  grids are too costly, cell-centred 1 km EFT alone gives +0.10 / +0.17 s; it is not enough on its own.
  Source depth of the station: keep zSrce = -elevation with the a1 station-cell patching.
* **P-2 tomoDD coordinates:** rescale latitudes and longitudes (stations, events, MOD nodes, `wlat/wlon`)
  with `ellipsoid_coords` before writing the tomoDD inputs and invert the mapping on all outputs
  (`tomoDD.reloc`, `Vp_model.dat`/`Vs_model.dat` node coordinates). Input-side only, no source change.
  Removes -0.045 s (P) / -0.07 s (S) at 150-200 km.
* **P-3 MOD build:** nearest NLL rock column at each node (no box average), depth nodes -3, -1, 0, 1, ..., 9,
  then every 2 km to 49 km, 52, 56, 60, 65, 70, 80, 100, 125 (40 interior depth nodes, 42 with the edge nodes; tomoDD limit 50) [**not adopted**, PI 2026-11-22: section 7].
  Sampling error +0.012 s P / +0.017 S (from +0.027 / +0.046). Cost: about twice the inversion parameters
  and more non-zero derivatives per ray: re-run the bounds-checked binary (MAXNODE 400, MAXND) on this grid
  before use, and redo the D-B026 memory check. If that is too expensive keep the current depth nodes but
  still drop the box average (nearest column: +0.018 / +0.031).
* **P-4 tomoDD forward correction per station and phase:** c(sta, phase) = median over picks of
  [tomoDD time - reference time] in the starting model (reference = P-1 NLL grids, which agree with the
  exact times to ~0.01-0.02 s), **added** to the observed times when writing `absolute.dat` (a slow tomoDD
  needs a later observation for the same residual; equivalently, (reference - tomoDD) is subtracted) (like the
  D-B020 statics; differential times unchanged). Corrects the 10 km minimum-segment error at stations:
  -0.04 to +0.12 s P, -0.04 to +0.28 s S here. Residual after correction: P median 0.000, p95 0.049 s.
  Frozen at the starting model (the error depends on the near-station gradient, which the inversion changes
  little at our node spacing). It must be removed again when tomoDD residuals are compared with NLL.
* **P-5 converting the final tomoDD model to NLL:** sample tomoDD's own field (trilinear in lon, lat,
  depth on the nodes, depth clamped at 0, Vs = Vp/(Vp/Vs) at nodes: `mod_velocity` in `b2_fwd_common.py`)
  at the NLL cell centres, in the back-transformed coordinates of P-2, then restore water/air and the a1
  station-cell patching; grids per P-1. No residual offset should then remain to be absorbed by statics; any
  offset left (expected <= 0.02 s P median) goes into the B4 station terms, which are re-estimated anyway.
* **P-6 recompute the D-B020 OBS S statics** with the P-1 forward (or take them from tomoDD residuals after
  P-4) before production runs; the current ones carry the NLL production error (~0.26 s S near stations).
* **P-7 (follow-up) quantify the effect on v4-NLL3D:** relocate one or two tiles with P-1 grids and compare
  depths and origin times with iter1; decide whether a v4-NLL3D rerun is needed before B4.

Not tested: the NLL alternatives in the build (`fmm2grid`/`interface2fmm`, `sphfd_SWR_NLL`), which might
avoid the PL gradient error; tomoDD with a smaller `mins` (source change, not allowed under the licence).

## 6. Files

| file | role |
|---|---|
| b2_fwd_common.py | paths, MOD reader/writer, tomoDD lookup re-implementation, NLL grid lookup, Grid2Time and tomoDD drivers, tables |
| b2_fwd_subset.py | station/event subset, production NLL lookup, check01 reproduction |
| b2_fwd_nll.py | Grid2Time variants (model, cell size, centring, EFT) |
| b2_fwd_mods.py | MOD variants; `ellipsoid_coords` |
| b2_fwd_tomodd.py | tomoDD 1-iteration forward runs |
| b2_fwd_taup.py | TauP 1D spherical reference |
| b2_fwd_pltest.py | Grid2Time vs analytic times |
| b2_fwd_compare.py | references (Richardson), decomposition and option tables |
| b2_fwd_p1.py | P-1 reference times for a region (section 7.1) |
| b2_fwd_statics.py | P-6 OBS S statics from the P-1 times (7.2) |
| b2_tomodd_backtransform.py | P-2 outputs back to geographic coordinates (7.3) |
| b2_fwd_v2check.py | end-to-end residual check of a build (7.5) |

## 7. Implementation (Mendocino-Gorda v2)

Built 2026-11-22 after D-B028. Region `mendocino_gorda_v2` in params (same event box, picks and nodes in lon/lat as
`mendocino_gorda`; block `forward`), runs in `/wd1/mdenolle_data/tomo_v1/b2_tomodd/mendocino_gorda_v2/`.
**PI change:** the finer depth nodes of P-3 are not used (coverage-limited upper crust, notes/smoothing_test.md);
v2 keeps `tomodd.node_z` (21 interior depth nodes) with nearest-column sampling. The 40-level build is kept aside in
`inputs_z42/` (+ `check01_z42`, `check01_prod_z42`); its absolute.dat has the P-4 correction with the wrong sign
(see below) and must not be used as is.

### 7.1 P-1 reference times (`b2_fwd_p1.py mendocino_gorda --out-region mendocino_gorda_v2 --keep-grids`)

All 153 stations, P and S (303 station-phases): continuous v4-NLL3D model (trilinear between NLL nodes), cell-centred
velocities, EFT, Grid2Time at 1 km and 0.5 km on a box around each station's points (+4 km), z -5 to >= 70 km,
T = T0.5 - k (T1 - T0.5), k = 1.33 (P), 1.28 (S) (params `tomodd.p1`). Sampled at the 228,454 region picks (B1b
hypocentres) and at 124,973 iter1 OBS S phases within 100 km (for P-6); no NaN. 0.5 km grids deleted after sampling;
the combined 1 km grids are kept: `p1_reference/grids/<sta>/p1.<ph>.<sta>.time.*`, **4.8 GB** (17 MB per grid; z is the
flattened depth, region-limited boxes, so B4 will still need full ±250 km grids built the same way). Wall time ~70 min
on 16 processes. Check: P-1 agrees with the subset study's Richardson times to <= 0.011 s; production NLL minus P-1
over all region picks: P +0.19 s, S +0.34 s median (subset: +0.21 / +0.37).
Outputs: `p1_reference/region_picks_p1.parquet`, `p1_reference/obs_S_iter1_p1.parquet`.

### 7.2 P-6 OBS S statics (`b2_fwd_statics.py mendocino_gorda_v2`)

D-B020 rule unchanged (median no-term S residual within 60 km, n >= 10; else 100 km; else none) with residual = (iter1
pick - iter1 origin time) - P-1 time. New file `data/tomo_v1/obs_check/obs_near_station_S_terms_p1.csv` (the 110 region
OBS with S picks; old file untouched). Checks: the old statics are reproduced from the same picks to 0.0005 s.

| old static class | n | old median | new median | change median (range) |
|---|---|---|---|---|
| < -0.5 s | 8 | -0.641 | -0.417 | +0.229 (+0.15..+0.34) |
| -0.5..0 s | 37 | -0.150 | +0.043 | +0.255 (+0.07..+0.46) |
| 0..0.3 s | 33 | +0.098 | +0.410 | +0.292 (+0.14..+0.67) |
| > 0.3 s | 30 | +0.597 | +0.885 | +0.283 (+0.19..+0.60) |

All 108 statics increase (median +0.274 s, 5-95 % +0.15..+0.46); the change follows the P-1 minus production
prediction of the picks used (r = 0.94; change + (P-1 - production) median 0.012 s). Same 108 stations have a static
(101 near60, 7 near100, as before: no source changes), FS12B.7D and J10B.7D none. Statics < -0.5 s: 1 (was 8).
Reading: with a correct forward, near-station OBS S is late by +0.3 s on average (sediment, as D-B019 expected);
the old near-zero statics were the NLL slowness cancelling it.

### 7.3 Input builder (`b2_tomodd_inputs.py`, params `tomodd.regions.<region>.forward`)

Options (defaults = first build; re-running `mendocino_gorda` reproduces event.dat, station.dat, absolute.dat,
dt.ct and MOD byte for byte): `coords` ellipsoid|sphere (P-2), `mod` nearest|box (P-3), `station_corr` (P-4),
`statics_file` (P-6), `p1_reference`. v2 writes `inputs/forward.json` (mapping constants: centre 41.3, -125.0;
M/R 0.998784, N/R 1.002583), `station_corr.csv`, `forward_start_model.parquet` (tomoDD and P-1 time per pick).

* P-2: coordinates rescaled; round trip <= 5e-7 deg (stations), 5e-5 deg on MOD nodes (4-decimal MOD format, ~5 m);
  a 318 km station pair: WGS-84 317.852 km, sphere with geographic coordinates 317.070, with scaled ones 317.828.
* P-4 sign: c = median(tomoDD - P-1) is **added** to the absolute times. (D-B028 and my first note said
  "subtracted"; that is the sign of reference - tomoDD. A first v2 build with the minus sign doubled the bias, which
  the end-to-end check caught: P -0.095 s.) c, from a 1-iteration production run on the uncorrected inputs:

| | n | 5 % | median | 95 % | min | max |
|---|---|---|---|---|---|---|
| P land | 42 | -0.008 | +0.044 | +0.105 | -0.026 | +0.134 |
| P OBS | 110 | -0.025 | +0.062 | +0.184 | -0.128 | +0.336 |
| S land | 41 | -0.046 | +0.072 | +0.170 | -0.115 | +0.176 |
| S OBS | 110 | +0.013 | +0.145 | +0.467 | -0.131 | +0.819 |

  Median within-station MAD 0.020 s (P), 0.042 s (S). The largest (S at G22D +0.82, FS07D +0.63, FS02B +0.60,
  BB660 +0.59, BB200 +0.54) are shelf/slope OBS where the 10 km MOD cannot hold the seafloor-referenced near-station
  structure: c holds MOD sampling as well as ray-tracer error there.
* Back-transform: `b2_tomodd_backtransform.py REGION TAG` writes `Output_Files/tomoDD.reloc_geo`, `tomoDD.loc_geo`,
  `tomoDD.sta_geo` (lat/lon mapped back, other columns unchanged) and `model_geo.npz` (geographic node lon/lat/z with
  Vp, Vs from Vp_model.dat/Vs_model.dat). Tested on `fwd1`. `b2_tomodd_run.py mendocino_gorda_v2 TAG` works unchanged
  (the centre is a fixed point of the map).

### 7.4 Bounds-checked run (`b2_tomodd_run.py mendocino_gorda_v2 check01 --check`)

Ray tracing of all 228,454 rays, G assembly (36.2 M non-zero model derivatives after smoothing, 44 per datum, MAXND
100) and LSQR (CND ~1200) completed with **no bounds error** (MAXNODE, MAXOBS 57 of 100, MAXDATA 827,894 of 3 M fine).
The bounds-checked binary then stops (exit 2) at a malformed FORMAT in the "Location summary" write after LSQR
(tomoDD-SE.f line 2324, `f7.3")"`, gfortran "Missing comma between descriptors"); the production binary does not stop
there. tomoDD-SE.res.ini was written before. 178 s, peak RSS 2.66 GB. Production binary, same inputs, one
iteration (`fwd1`): exit 0, 284 s, 2.65 GB. For reference the 40-level grid: 52.2 M derivatives (63 per datum),
production 341 s, 3.73 GB, also no bounds error.

### 7.5 End to end (`b2_fwd_v2check.py mendocino_gorda_v2 fwd1`)

tomoDD initial residual minus P-1 reference residual (same 228,454 picks); medians, s
(tables per bin in `fwd1/v2check_tables.md`):

| | P land 0-25 | P land 50-100 | P land 150-200 | P OBS 0-25 | P OBS 50-100 | P OBS 150-200 | P all | P p95 |
|---|---|---|---|---|---|---|---|---|
| v2 | +0.006 | -0.001 | -0.002 | -0.004 | -0.000 | -0.002 | 0.000 | 0.085 |
| first build | -0.015 | -0.024 | -0.036 | -0.085 | -0.085 | -0.030 | -0.032 | 0.187 |

| | S land 0-25 | S land 50-100 | S land 150-200 | S OBS 0-25 | S OBS 50-100 | S OBS 150-200 | S all | S p95 |
|---|---|---|---|---|---|---|---|---|
| v2 | +0.009 | -0.004 | -0.003 | -0.010 | +0.001 | +0.002 | -0.000 | 0.187 |
| first build | -0.021 | -0.036 | -0.071 | -0.200 | -0.187 | -0.108 | -0.076 | 0.461 |

The overall medians are zero by construction (c is a per-station median over the same picks); what P-4 cannot remove
shows in the bins: P within ±0.02 s in every bin (worst 100-150 km land -0.019), S within ±0.035 s; p95 |d| P 0.067
land / 0.119 OBS, S 0.111 / 0.234. So the median target (<= 0.02 s P) is met; the D-B015-like p95 <= 0.05 s is not
(path-dependent MOD sampling with 10 km nodes and the nearest-column choice, largest near shelf OBS). These are
within the pick sigma (P 0.05-0.15 s) but are not random noise; the inversion will see them as structure near the
stations.

### 7.6 Points to decide

* Wording of D-B028 P-4: "added" (c = tomoDD - reference), not "subtracted".
* P-4 corrections up to +0.6-0.8 s S at five shelf/slope OBS mix MOD-sampling and tracer error with what the
  inversion could fit; keep as is, cap (e.g. |c| <= 0.3 s), or limit P-4 to the tracer part (would need a MOD-specific
  reference per station; ~1 h compute).
* The new statics are larger by +0.27 s; they apply to all OBS. Only the 110 Mendocino-Gorda OBS are recomputed;
  other regions need `b2_fwd_p1.py` + `b2_fwd_statics.py` for their stations.
* Keep the 4.8 GB P-1 1 km grids (region boxes, flattened z) or delete; they are not the B4 grids.

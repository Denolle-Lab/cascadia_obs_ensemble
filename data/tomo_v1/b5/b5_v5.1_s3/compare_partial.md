# B5 relocation b5_v5.1_s3 (tomoDD-SE relocation only, variant dd, model v0.3.1)

Start: B4 run b4_v51_iter1 (NLLoc, full precision, EFT undone). Catalog: /wd1/mdenolle_data/tomo_v1/b5/b5_v5.1_s3/catalog_b5_b5_v5.1_s3_partial.csv (172340 events). Depths in km below sea level. B5 = tomoDD solution with each cluster's centroid put back on its B4 centroid (D-B053).

PARTIAL: 2332 of 2336 clusters not run; their events keep B4.

## Events by location source

| region | events | b5_tomodd | nll3d (B4 kept) | v4 (tier U) | relocated % of B4 3D |
|---|---|---|---|---|---|
| Axial | 26731 | 10520 | 16206 | 0 | 39.4 |
| Blanco | 42385 | 1044 | 41305 | 0 | 2.5 |
| Cascades | 17280 | 1436 | 15657 | 0 | 8.4 |
| Endeavour | 179 | 0 | 178 | 0 | 0.0 |
| Explorer | 1523 | 0 | 1514 | 0 | 0.0 |
| Gorda | 11378 | 0 | 11350 | 0 | 0.0 |
| JdF ridge/plate | 2851 | 202 | 2641 | 0 | 7.1 |
| Mendocino | 24644 | 0 | 24608 | 0 | 0.0 |
| Offshore forearc | 15422 | 0 | 15398 | 0 | 0.0 |
| Other onshore | 15093 | 0 | 14825 | 0 | 0.0 |
| Puget/Georgia | 14854 | 0 | 14833 | 0 | 0.0 |
| all | 172340 | 13202 | 158515 | 0 | 7.7 |

## Why events were not relocated (b5_status)

| region | cluster_not_run | dropped_by_tomodd | relocated | tier_U | unlinked | all |
|---|---|---|---|---|---|---|
| Axial | 10548 | 12 | 10520 | 5 | 5646 | 26731 |
| Blanco | 20238 | 0 | 1044 | 36 | 21067 | 42385 |
| Cascades | 8151 | 0 | 1436 | 187 | 7506 | 17280 |
| Endeavour | 9 | 0 | 0 | 1 | 169 | 179 |
| Explorer | 474 | 0 | 0 | 9 | 1040 | 1523 |
| Gorda | 5589 | 0 | 0 | 28 | 5761 | 11378 |
| JdF ridge/plate | 466 | 9 | 202 | 8 | 2166 | 2851 |
| Mendocino | 15596 | 0 | 0 | 36 | 9012 | 24644 |
| Offshore forearc | 8961 | 0 | 0 | 24 | 6437 | 15422 |
| Other onshore | 9465 | 0 | 0 | 268 | 5360 | 15093 |
| Puget/Georgia | 9651 | 0 | 0 | 21 | 5182 | 14854 |
| all | 89148 | 21 | 13202 | 623 | 69346 | 172340 |

## Shift from B4 (relocated events, km)

| region | n | dh med / p90 | dz med | abs dz med / p90 | LSQR ex / ez med |
|---|---|---|---|---|---|
| Axial | 10520 | 0.334 / 0.967 | -0.054 | 0.276 / 0.920 | 0.024 / 0.033 |
| Blanco | 1044 | 0.268 / 0.630 | -0.013 | 0.163 / 0.485 | 0.059 / 0.073 |
| Cascades | 1436 | 0.279 / 0.770 | +0.006 | 0.332 / 1.080 | 0.074 / 0.096 |
| JdF ridge/plate | 202 | 0.285 / 3.189 | +0.024 | 0.046 / 0.912 | 0.145 / 0.131 |
| all | 13202 | 0.321 / 0.920 | -0.039 | 0.266 / 0.912 | 0.029 / 0.040 |

## dt residual rms in the B4 time grids (s), B4 -> B5

Relocated core-core pairs; residual = observed minus B4-grid differential time, per-pair median removed (b5_compare.py metric of the D-B048 test).

| region | n P | P B4 -> B5 | n S | S B4 -> B5 |
|---|---|---|---|---|
| Axial | 238029 | 0.086 -> 0.093 | 225184 | 0.105 -> 0.136 |
| Blanco | 10071 | 0.126 -> 0.130 | 10094 | 0.135 -> 0.144 |
| Cascades | 40715 | 0.128 -> 0.133 | 23990 | 0.178 -> 0.198 |
| JdF ridge/plate | 2336 | 0.153 -> 0.157 | 2154 | 0.167 -> 0.176 |
| all | 291151 | 0.095 -> 0.102 | 261422 | 0.116 -> 0.144 |

## tomoDD dt rms (its own forward model, first -> last iteration, ms) and condition number

| region | clusters | rmsct first med | rmsct last med | CND first med / max | CND last med / max |
|---|---|---|---|---|---|
| Axial | 1 | 64 | 40 | 268 / 268 | 237 / 237 |
| Blanco | 1 | 81 | 45 | 31 / 31 | 27 / 27 |
| Cascades | 1 | 87 | 59 | 62 / 62 | 51 / 51 |
| JdF ridge/plate | 1 | 139 | 87 | 10 / 10 | 10 / 10 |

## Cluster shape (PCA of relocated core events, clusters with >= 10 events, km)

sigma3 = thickness across the best-fitting plane (smallest axis).

| region | clusters | sigma3 B4 med | sigma3 B5 med | B5/B4 med | thinner % | sigma1 B4 / B5 med |
|---|---|---|---|---|---|---|
| Axial | 1 | 0.927 | 1.117 | 1.20 | 0 | 2.37 / 2.34 |
| Blanco | 1 | 0.617 | 0.642 | 1.04 | 0 | 3.89 / 3.89 |
| Cascades | 1 | 0.319 | 0.391 | 1.22 | 0 | 1.83 / 1.95 |
| JdF ridge/plate | 1 | 0.078 | 0.099 | 1.27 | 0 | 1.86 / 1.86 |
| all | 4 | 0.468 | 0.516 | 1.21 | 0 | 2.12 / 2.15 |

## Centroid shift removed (tomoDD drift of each cluster's relocated core events, D-B053)

Horizontal median 0.041 km (p90 0.065, max 0.071); vertical median +0.063 km (abs p90 0.129, max abs 0.135); origin time median -0.0036 s. Weighted by events: horizontal 0.063 km, vertical 0.112 km.
Frame centre: every start position >= 50 m from it (min 56 m); final relocated positions min 23 m.

## Above the solid surface (D-B055)

surface_km = D-B068 merged solid surface (b4_surface.py, /wd1/mdenolle_data/reloc_v5/topo): first covering source of [gmrt_-129.55_-128.75_47.65_48.35_high.nc (41 m, md5 7bcc9e645a3cb1b276fbdea079460c2b); gmrt_-130.35_-129.75_45.75_46.2_high.nc (42 m, md5 d2653715b20e7bc172bf88f2bdf48b54); gmrt_-129.7_-127.9_48.55_49.4_med.nc (80 m, md5 077056a5725bd1f92a6ee6d8c38bf7cc); gmrt_-129.3_-127.3_43.3_44.6_med.nc (88 m, md5 0dd4eda9525b07f4323d8255a4f33b94); gmrt_-130.4_-127.2_47.4_49.8_med.nc (162 m, md5 67f460ee394d9b575bc624fd0e8edd93); gmrt_-123.6_-121.4_46.6_48.6_med.nc (165 m, md5 1419d851c3112ef96228a4a3df54ec25); gmrt_-130.6_-126.6_42.6_45.3_med.nc (176 m, md5 9145fd1d13d7d2c0301596b44207fbb1); gmrt_-125.6_-123.2_39.6_41.4_med.nc (186 m, md5 8b0aef3b02017a3eb57f422637b7dac3); gmrt_-127.6_-123.2_46.2_49.2_med.nc (329 m, md5 96b54b1805c70ae9f22223384d558baa); gmrt_-126.8_-123.2_42.8_46.4_med.nc (348 m, md5 9b27b0947fd9bd495e7a1abecbbcd4aa); gmrt_-128.8_-123.8_40.0_43.2_med.nc (366 m, md5 98c18c8677c4bb705eaea0a0afb575d0); earth_relief_01m_g (1558 m, md5 fba0fa3df82fd1b39069850540f55a8f)] (GMRT finest first, then earth_relief 01m gridline; covering = inside the source's grid lines with 4 finite bilinear nodes), bilinear on that source's grid, no blending at the final epicentre, above_surface = depth < -surface_km on the true depth (b4_catalog.py). Nothing is moved or dropped.

| region | B5 relocated above | B4 kept above | tier U above | B5 events above at their B4 position |
|---|---|---|---|---|
| Axial | 0 | 0 | 0 | 0 |
| Blanco | 0 | 0 | 0 | 0 |
| Cascades | 0 | 0 | 0 | 0 |
| Endeavour | 0 | 0 | 0 | 0 |
| Explorer | 0 | 0 | 0 | 0 |
| Gorda | 0 | 0 | 0 | 0 |
| JdF ridge/plate | 0 | 0 | 0 | 0 |
| Mendocino | 0 | 0 | 0 | 0 |
| Offshore forearc | 0 | 0 | 0 | 0 |
| Other onshore | 0 | 0 | 0 | 0 |
| Puget/Georgia | 0 | 0 | 0 | 0 |
| all | 0 | 0 | 0 | 0 |

Total above the surface: 0 (0 B5-relocated, 0 B4 kept, 0 tier U). Publishable under D-B055.

## Run

4 clusters (1 chunks of split components), binaries {'e10k': 3, 'e13k': 1}; status {'ok': 4}; tomoDD wall time total 0.0 h (largest cluster 1.3 min, peak RSS 2.07 GB).

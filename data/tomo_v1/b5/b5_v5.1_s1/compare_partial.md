# B5 relocation b5_v5.1_s1 (tomoDD-SE relocation only, variant dd, model v0.3.1)

Start: B4 run b4_v51_iter1 (NLLoc, full precision, EFT undone). Catalog: /wd1/mdenolle_data/tomo_v1/b5/b5_v5.1_s1/catalog_b5_b5_v5.1_s1_partial.csv (172340 events). Depths in km below sea level. B5 = tomoDD solution with each cluster's centroid put back on its B4 centroid (D-B053).

PARTIAL: 2332 of 2336 clusters not run; their events keep B4.

## Events by location source

| region | events | b5_tomodd | nll3d (B4 kept) | v4 (tier U) | relocated % of B4 3D |
|---|---|---|---|---|---|
| Axial | 26731 | 9636 | 17090 | 0 | 36.1 |
| Blanco | 42385 | 1044 | 41305 | 0 | 2.5 |
| Cascades | 17280 | 1424 | 15669 | 0 | 8.3 |
| Endeavour | 179 | 0 | 178 | 0 | 0.0 |
| Explorer | 1523 | 0 | 1514 | 0 | 0.0 |
| Gorda | 11378 | 0 | 11350 | 0 | 0.0 |
| JdF ridge/plate | 2851 | 187 | 2656 | 0 | 6.6 |
| Mendocino | 24644 | 0 | 24608 | 0 | 0.0 |
| Offshore forearc | 15422 | 0 | 15398 | 0 | 0.0 |
| Other onshore | 15093 | 0 | 14825 | 0 | 0.0 |
| Puget/Georgia | 14854 | 0 | 14833 | 0 | 0.0 |
| all | 172340 | 12291 | 159426 | 0 | 7.2 |

## Why events were not relocated (b5_status)

| region | airquake_by_tomodd | cluster_not_run | dropped_by_tomodd | relocated | tier_U | unlinked | all |
|---|---|---|---|---|---|---|---|
| Axial | 891 | 10432 | 121 | 9636 | 5 | 5646 | 26731 |
| Blanco | 0 | 20238 | 0 | 1044 | 36 | 21067 | 42385 |
| Cascades | 12 | 8151 | 0 | 1424 | 187 | 7506 | 17280 |
| Endeavour | 0 | 9 | 0 | 0 | 1 | 169 | 179 |
| Explorer | 0 | 474 | 0 | 0 | 9 | 1040 | 1523 |
| Gorda | 0 | 5589 | 0 | 0 | 28 | 5761 | 11378 |
| JdF ridge/plate | 19 | 462 | 9 | 187 | 8 | 2166 | 2851 |
| Mendocino | 0 | 15596 | 0 | 0 | 36 | 9012 | 24644 |
| Offshore forearc | 0 | 8961 | 0 | 0 | 24 | 6437 | 15422 |
| Other onshore | 0 | 9465 | 0 | 0 | 268 | 5360 | 15093 |
| Puget/Georgia | 0 | 9651 | 0 | 0 | 21 | 5182 | 14854 |
| all | 922 | 89028 | 130 | 12291 | 623 | 69346 | 172340 |

## Shift from B4 (relocated events, km)

| region | n | dh med / p90 | dz med | abs dz med / p90 | LSQR ex / ez med |
|---|---|---|---|---|---|
| Axial | 9636 | 0.325 / 0.916 | -0.049 | 0.253 / 0.792 | 0.026 / 0.033 |
| Blanco | 1044 | 0.268 / 0.630 | -0.013 | 0.163 / 0.485 | 0.059 / 0.073 |
| Cascades | 1424 | 0.276 / 0.759 | -0.001 | 0.330 / 1.047 | 0.071 / 0.095 |
| JdF ridge/plate | 187 | 0.265 / 1.865 | +0.008 | 0.103 / 0.356 | 0.143 / 0.121 |
| all | 12291 | 0.312 / 0.871 | -0.040 | 0.247 / 0.798 | 0.030 / 0.040 |

## dt residual rms in the B4 time grids (s), B4 -> B5

Relocated core-core pairs; residual = observed minus B4-grid differential time, per-pair median removed (b5_compare.py metric of the D-B048 test).

| region | n P | P B4 -> B5 | n S | S B4 -> B5 |
|---|---|---|---|---|
| Axial | 222865 | 0.083 -> 0.090 | 209878 | 0.102 -> 0.132 |
| Blanco | 10071 | 0.126 -> 0.130 | 10094 | 0.135 -> 0.144 |
| Cascades | 40566 | 0.128 -> 0.132 | 23950 | 0.177 -> 0.196 |
| JdF ridge/plate | 2220 | 0.150 -> 0.151 | 2019 | 0.169 -> 0.174 |
| all | 275722 | 0.093 -> 0.099 | 245941 | 0.114 -> 0.141 |

## tomoDD dt rms (its own forward model, first -> last iteration, ms) and condition number

| region | clusters | rmsct first med | rmsct last med | CND first med / max | CND last med / max |
|---|---|---|---|---|---|
| Axial | 1 | 64 | 39 | 268 / 268 | 221 / 221 |
| Blanco | 1 | 81 | 45 | 31 / 31 | 27 / 27 |
| Cascades | 1 | 87 | 59 | 62 / 62 | 53 / 53 |
| JdF ridge/plate | 1 | 139 | 85 | 10 / 10 | 10 / 10 |

## Cluster shape (PCA of relocated core events, clusters with >= 10 events, km)

sigma3 = thickness across the best-fitting plane (smallest axis).

| region | clusters | sigma3 B4 med | sigma3 B5 med | B5/B4 med | thinner % | sigma1 B4 / B5 med |
|---|---|---|---|---|---|---|
| Axial | 1 | 0.860 | 1.052 | 1.22 | 0 | 2.25 / 2.22 |
| Blanco | 1 | 0.617 | 0.642 | 1.04 | 0 | 3.89 / 3.89 |
| Cascades | 1 | 0.318 | 0.386 | 1.22 | 0 | 1.83 / 1.94 |
| JdF ridge/plate | 1 | 0.078 | 0.164 | 2.11 | 0 | 1.86 / 1.86 |
| all | 4 | 0.467 | 0.514 | 1.22 | 0 | 2.06 / 2.08 |

## Centroid shift removed (tomoDD drift of each cluster's relocated core events, D-B053)

Horizontal median 0.039 km (p90 0.059, max 0.065); vertical median +0.008 km (abs p90 0.087, max abs 0.119); origin time median -0.0043 s. Weighted by events: horizontal 0.056 km, vertical 0.095 km.
Frame centre: every start position >= 50 m from it (min 56 m); final relocated positions min 25 m.

## Above the solid surface (D-B055)

surface_km = D-B068 merged solid surface (b4_surface.py, /wd1/mdenolle_data/reloc_v5/topo): first covering source of [gmrt_-129.55_-128.75_47.65_48.35_high.nc (41 m, md5 7bcc9e645a3cb1b276fbdea079460c2b); gmrt_-130.35_-129.75_45.75_46.2_high.nc (42 m, md5 d2653715b20e7bc172bf88f2bdf48b54); gmrt_-129.7_-127.9_48.55_49.4_med.nc (80 m, md5 077056a5725bd1f92a6ee6d8c38bf7cc); gmrt_-129.3_-127.3_43.3_44.6_med.nc (88 m, md5 0dd4eda9525b07f4323d8255a4f33b94); gmrt_-130.4_-127.2_47.4_49.8_med.nc (162 m, md5 67f460ee394d9b575bc624fd0e8edd93); gmrt_-123.6_-121.4_46.6_48.6_med.nc (165 m, md5 1419d851c3112ef96228a4a3df54ec25); gmrt_-130.6_-126.6_42.6_45.3_med.nc (176 m, md5 9145fd1d13d7d2c0301596b44207fbb1); gmrt_-125.6_-123.2_39.6_41.4_med.nc (186 m, md5 8b0aef3b02017a3eb57f422637b7dac3); gmrt_-127.6_-123.2_46.2_49.2_med.nc (329 m, md5 96b54b1805c70ae9f22223384d558baa); gmrt_-126.8_-123.2_42.8_46.4_med.nc (348 m, md5 9b27b0947fd9bd495e7a1abecbbcd4aa); gmrt_-128.8_-123.8_40.0_43.2_med.nc (366 m, md5 98c18c8677c4bb705eaea0a0afb575d0); earth_relief_01m_g (1558 m, md5 fba0fa3df82fd1b39069850540f55a8f)] (GMRT finest first, then earth_relief 01m gridline; covering = inside the source's grid lines with 4 finite bilinear nodes), bilinear on that source's grid, no blending at the final epicentre, above_surface = depth < -surface_km on the true depth (b4_catalog.py). Nothing is moved or dropped.

| region | B5 relocated above | B4 kept above | tier U above | B5 events above at their B4 position |
|---|---|---|---|---|
| Axial | 9 | 0 | 0 | 0 |
| Blanco | 0 | 0 | 0 | 0 |
| Cascades | 1 | 0 | 0 | 0 |
| Endeavour | 0 | 0 | 0 | 0 |
| Explorer | 0 | 0 | 0 | 0 |
| Gorda | 0 | 0 | 0 | 0 |
| JdF ridge/plate | 66 | 0 | 0 | 0 |
| Mendocino | 0 | 0 | 0 | 0 |
| Offshore forearc | 0 | 0 | 0 | 0 |
| Other onshore | 0 | 0 | 0 | 0 |
| Puget/Georgia | 0 | 0 | 0 | 0 |
| all | 76 | 0 | 0 | 0 |

Total above the surface: 76 (76 B5-relocated, 0 B4 kept, 0 tier U). NOT PUBLISHABLE under D-B055 (computation errors to find, not to move).
B5 events above: height above the surface median 93 m, max 395 m; 76 of them were below the surface at their B4 position; B4 depth below the surface median 20 m (all relocated events 1694 m); B5 - B4 depth median -0.140 km.
Above-surface B5 events by start depth below the surface (B4): < 0.1 km: 66 of 174, 0.1-0.5 km: 1 of 153, 0.5-1 km: 3 of 844, 1-3 km: 5 of 9873, >= 3 km: 1 of 1247.
Tier U above the surface: 0 (v4 hypocentres, D-B057; not relocated here).

## Run

4 clusters (1 chunks of split components), binaries {'e10k': 3, 'e13k': 1}; status {'ok': 4}; tomoDD wall time total 0.1 h (largest cluster 3.0 min, peak RSS 2.25 GB).

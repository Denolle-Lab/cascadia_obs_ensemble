# B5 relocation b5_v0.3.0 (tomoDD-SE relocation only, variant dd, model v0.3.0)

Start: B4 run b4_iter1_v0.3.0 (NLLoc, full precision, EFT undone). Catalog: /wd1/mdenolle_data/tomo_v1/b5/b5_v0.3.0/catalog_b5_b5_v0.3.0.csv (172339 events). Depths in km below sea level. B5 = tomoDD solution with each cluster's centroid put back on its B4 centroid (D-B053).

## Events by location source

| region | events | b5_tomodd | nll3d (B4 kept) | v4 (tier U) | relocated % of B4 3D |
|---|---|---|---|---|---|
| Axial | 26731 | 17577 | 8794 | 360 | 66.7 |
| Blanco | 42385 | 21393 | 14893 | 6099 | 59.0 |
| Cascades | 17280 | 9517 | 6536 | 1227 | 59.3 |
| Endeavour | 179 | 15 | 80 | 84 | 15.8 |
| Explorer | 1523 | 448 | 397 | 678 | 53.0 |
| Gorda | 11378 | 5352 | 4315 | 1711 | 55.4 |
| JdF ridge/plate | 2851 | 514 | 1653 | 684 | 23.7 |
| Mendocino | 24643 | 15332 | 7900 | 1411 | 66.0 |
| Offshore forearc | 15422 | 8783 | 5412 | 1227 | 61.9 |
| Other onshore | 15093 | 9261 | 4933 | 899 | 65.2 |
| Puget/Georgia | 14854 | 9455 | 4710 | 689 | 66.7 |
| all | 172339 | 97647 | 59623 | 15069 | 62.1 |

## Why events were not relocated (b5_status)

| region | airquake_by_tomodd | cluster_skipped | dropped_by_tomodd | relocated | tier_U | unlinked | all |
|---|---|---|---|---|---|---|---|
| Axial | 3162 | 4 | 325 | 17577 | 360 | 5303 | 26731 |
| Blanco | 28 | 3 | 94 | 21393 | 6099 | 14768 | 42385 |
| Cascades | 114 | 0 | 99 | 9517 | 1227 | 6323 | 17280 |
| Endeavour | 0 | 0 | 0 | 15 | 84 | 80 | 179 |
| Explorer | 0 | 2 | 1 | 448 | 678 | 394 | 1523 |
| Gorda | 28 | 7 | 147 | 5352 | 1711 | 4133 | 11378 |
| JdF ridge/plate | 148 | 7 | 40 | 514 | 684 | 1458 | 2851 |
| Mendocino | 22 | 0 | 215 | 15332 | 1411 | 7663 | 24643 |
| Offshore forearc | 27 | 2 | 169 | 8783 | 1227 | 5214 | 15422 |
| Other onshore | 144 | 5 | 215 | 9261 | 899 | 4569 | 15093 |
| Puget/Georgia | 42 | 2 | 156 | 9455 | 689 | 4510 | 14854 |
| all | 3715 | 32 | 1461 | 97647 | 15069 | 54415 | 172339 |

## Shift from B4 (relocated events, km)

| region | n | dh med / p90 | dz med | abs dz med / p90 | LSQR ex / ez med |
|---|---|---|---|---|---|
| Axial | 17577 | 0.571 / 1.606 | -0.030 | 0.344 / 1.585 | 0.024 / 0.023 |
| Blanco | 21393 | 0.388 / 1.410 | -0.009 | 0.381 / 2.500 | 0.066 / 0.106 |
| Cascades | 9517 | 0.511 / 1.968 | -0.002 | 0.449 / 2.348 | 0.118 / 0.130 |
| Endeavour | 15 | 0.019 / 0.055 | +0.001 | 0.019 / 0.070 | 0.397 / 0.238 |
| Explorer | 448 | 0.215 / 0.699 | +0.007 | 0.136 / 0.691 | 0.119 / 0.157 |
| Gorda | 5352 | 0.664 / 2.497 | -0.015 | 0.803 / 4.781 | 0.144 / 0.243 |
| JdF ridge/plate | 514 | 0.335 / 4.003 | +0.006 | 0.178 / 0.973 | 0.102 / 0.104 |
| Mendocino | 15332 | 0.932 / 2.912 | +0.156 | 0.749 / 3.267 | 0.082 / 0.103 |
| Offshore forearc | 8783 | 0.727 / 2.379 | -0.040 | 0.860 / 4.106 | 0.105 / 0.160 |
| Other onshore | 9261 | 0.708 / 2.608 | +0.009 | 0.667 / 3.679 | 0.120 / 0.159 |
| Puget/Georgia | 9455 | 0.826 / 2.443 | -0.037 | 0.906 / 4.068 | 0.122 / 0.161 |
| all | 97647 | 0.610 / 2.140 | -0.002 | 0.533 / 3.007 | 0.083 / 0.113 |

## dt residual rms in the B4 time grids (s), B4 -> B5

Relocated core-core pairs; residual = observed minus B4-grid differential time, per-pair median removed (b5_compare.py metric of the D-B048 test).

| region | n P | P B4 -> B5 | n S | S B4 -> B5 |
|---|---|---|---|---|
| Axial | 377999 | 0.077 -> 0.081 | 324556 | 0.126 -> 0.151 |
| Blanco | 304938 | 0.161 -> 0.164 | 375276 | 0.190 -> 0.269 |
| Cascades | 316004 | 0.154 -> 0.158 | 338505 | 0.167 -> 0.201 |
| Endeavour | 52 | 0.228 -> 0.227 | 39 | 0.287 -> 0.286 |
| Explorer | 6620 | 0.161 -> 0.154 | 6439 | 0.255 -> 0.266 |
| Gorda | 72561 | 0.198 -> 0.179 | 97310 | 0.223 -> 0.309 |
| JdF ridge/plate | 5300 | 0.139 -> 0.140 | 4238 | 0.184 -> 0.217 |
| Mendocino | 276973 | 0.149 -> 0.142 | 278310 | 0.182 -> 0.229 |
| Offshore forearc | 168737 | 0.146 -> 0.141 | 155892 | 0.201 -> 0.248 |
| Other onshore | 244650 | 0.157 -> 0.160 | 233497 | 0.185 -> 0.232 |
| Puget/Georgia | 254423 | 0.158 -> 0.164 | 267989 | 0.182 -> 0.233 |
| all | 2028257 | 0.145 -> 0.146 | 2082051 | 0.178 -> 0.230 |

## tomoDD dt rms (its own forward model, first -> last iteration, ms) and condition number

| region | clusters | rmsct first med | rmsct last med | CND first med / max | CND last med / max |
|---|---|---|---|---|---|
| Axial | 52 | 120 | 100 | 2 / 282 | 2 / 212 |
| Blanco | 855 | 130 | 101 | 2 / 135 | 2 / 117 |
| Cascades | 212 | 195 | 149 | 2 / 64 | 2 / 56 |
| Endeavour | 6 | 204 | 140 | 2 / 3 | 2 / 3 |
| Explorer | 38 | 163 | 133 | 2 / 8 | 2 / 7 |
| Gorda | 216 | 228 | 190 | 2 / 18 | 2 / 16 |
| JdF ridge/plate | 39 | 144 | 115 | 2 / 11 | 2 / 10 |
| Mendocino | 180 | 162 | 131 | 2 / 272 | 2 / 219 |
| Offshore forearc | 222 | 174 | 138 | 2 / 64 | 2 / 53 |
| Other onshore | 234 | 192 | 158 | 2 / 23 | 2 / 21 |
| Puget/Georgia | 121 | 141 | 109 | 2 / 335 | 2 / 280 |

## Cluster shape (PCA of relocated core events, clusters with >= 10 events, km)

sigma3 = thickness across the best-fitting plane (smallest axis).

| region | clusters | sigma3 B4 med | sigma3 B5 med | B5/B4 med | thinner % | sigma1 B4 / B5 med |
|---|---|---|---|---|---|---|
| Axial | 4 | 0.791 | 0.832 | 1.02 | 50 | 2.15 / 2.13 |
| Blanco | 151 | 0.458 | 0.491 | 1.02 | 42 | 1.51 / 1.55 |
| Cascades | 42 | 1.159 | 1.168 | 1.00 | 57 | 5.57 / 5.48 |
| Explorer | 9 | 1.669 | 1.778 | 1.00 | 44 | 7.49 / 7.45 |
| Gorda | 43 | 1.572 | 1.542 | 1.02 | 37 | 4.96 / 5.09 |
| JdF ridge/plate | 4 | 0.807 | 0.819 | 1.04 | 25 | 3.78 / 3.74 |
| Mendocino | 19 | 0.872 | 0.896 | 1.03 | 26 | 3.33 / 3.33 |
| Offshore forearc | 38 | 1.488 | 1.457 | 0.99 | 58 | 5.06 / 5.03 |
| Other onshore | 51 | 1.406 | 1.381 | 1.00 | 55 | 4.94 / 4.95 |
| Puget/Georgia | 26 | 0.977 | 0.949 | 0.99 | 62 | 2.82 / 2.80 |
| all | 387 | 0.828 | 0.820 | 1.00 | 47 | 2.98 / 2.97 |

## Centroid shift removed (tomoDD drift of each cluster's relocated core events, D-B053)

Horizontal median 0.002 km (p90 0.034, max 0.362); vertical median -0.000 km (abs p90 0.048, max abs 0.700); origin time median +0.0000 s. Weighted by events: horizontal 0.076 km, vertical 0.189 km.
Frame centre: every start position >= 50 m from it (min 50 m); final relocated positions min 2 m.

## Split effect (events relocated as core in one chunk and as halo in another)

2627 duplicates: horizontal difference median 0.398 km (p90 1.140), vertical 0.151 km (p90 0.941). By hop: hop 1: n 1242, dh 0.357, dz 0.126; hop 2: n 1385, dh 0.434, dz 0.172

## Above the solid surface (D-B055)

surface_km = GMT earth_relief 01m (gridline, bilinear) at the final epicentre, above_surface = depth < -surface_km on the true depth (b4_catalog.py). Nothing is moved or dropped.

| region | B5 relocated above | B4 kept above | tier U above | B5 events above at their B4 position |
|---|---|---|---|---|
| Axial | 66 | 170 | 259 | 21 |
| Blanco | 118 | 88 | 1658 | 38 |
| Cascades | 113 | 234 | 411 | 27 |
| Endeavour | 0 | 5 | 38 | 0 |
| Explorer | 0 | 2 | 211 | 0 |
| Gorda | 43 | 67 | 451 | 15 |
| JdF ridge/plate | 23 | 67 | 253 | 6 |
| Mendocino | 42 | 45 | 237 | 11 |
| Offshore forearc | 22 | 24 | 242 | 4 |
| Other onshore | 292 | 71 | 160 | 90 |
| Puget/Georgia | 110 | 62 | 159 | 32 |
| all | 829 | 835 | 4079 | 244 |

Total above the surface: 5743 (829 B5-relocated, 835 B4 kept, 4079 tier U). NOT PUBLISHABLE under D-B055 (computation errors to find, not to move).
B5 events above: height above the surface median 264 m, max 3675 m; their B4 depth was below the surface for 626 of them.

## Run

2209 clusters (6 chunks of split components), binaries {'e10k': 2196, 'o140': 7, 'e13k': 4, 'w': 1, 'm': 1}; status {'ok': 2175, 'skipped': 34}; tomoDD wall time total 2.1 h (largest cluster 22.2 min, peak RSS 3.38 GB).

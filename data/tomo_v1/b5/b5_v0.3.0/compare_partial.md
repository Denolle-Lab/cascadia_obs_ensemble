# B5 relocation b5_v0.3.0 (tomoDD-SE relocation only, variant dd, model v0.3.0)

Start: B4 run b4_iter1_v0.3.0 (NLLoc, full precision, EFT undone). Catalog: /wd1/mdenolle_data/tomo_v1/b5/b5_v0.3.0/catalog_b5_b5_v0.3.0_partial.csv (172339 events). Depths in km below sea level. B5 = tomoDD solution with each cluster's centroid put back on its B4 centroid (D-B053).

PARTIAL: 2206 of 2209 clusters not run; their events keep B4.

## Events by location source

| region | events | b5_tomodd | nll3d (B4 kept) | v4 (tier U) | relocated % of B4 3D |
|---|---|---|---|---|---|
| Axial | 26731 | 0 | 26371 | 360 | 0.0 |
| Blanco | 42385 | 4518 | 31768 | 6099 | 12.5 |
| Cascades | 17280 | 1443 | 14610 | 1227 | 9.0 |
| Endeavour | 179 | 0 | 95 | 84 | 0.0 |
| Explorer | 1523 | 0 | 845 | 678 | 0.0 |
| Gorda | 11378 | 0 | 9667 | 1711 | 0.0 |
| JdF ridge/plate | 2851 | 0 | 2167 | 684 | 0.0 |
| Mendocino | 24643 | 0 | 23232 | 1411 | 0.0 |
| Offshore forearc | 15422 | 0 | 14195 | 1227 | 0.0 |
| Other onshore | 15093 | 0 | 14194 | 899 | 0.0 |
| Puget/Georgia | 14854 | 0 | 14165 | 689 | 0.0 |
| all | 172339 | 5961 | 151309 | 15069 | 3.8 |

## Why events were not relocated (b5_status)

| region | airquake_by_tomodd | cluster_not_run | dropped_by_tomodd | relocated | tier_U | unlinked | all |
|---|---|---|---|---|---|---|---|
| Axial | 0 | 21068 | 0 | 0 | 360 | 5303 | 26731 |
| Blanco | 9 | 16945 | 46 | 4518 | 6099 | 14768 | 42385 |
| Cascades | 17 | 8270 | 0 | 1443 | 1227 | 6323 | 17280 |
| Endeavour | 0 | 15 | 0 | 0 | 84 | 80 | 179 |
| Explorer | 0 | 451 | 0 | 0 | 678 | 394 | 1523 |
| Gorda | 0 | 5534 | 0 | 0 | 1711 | 4133 | 11378 |
| JdF ridge/plate | 0 | 709 | 0 | 0 | 684 | 1458 | 2851 |
| Mendocino | 0 | 15569 | 0 | 0 | 1411 | 7663 | 24643 |
| Offshore forearc | 0 | 8981 | 0 | 0 | 1227 | 5214 | 15422 |
| Other onshore | 0 | 9625 | 0 | 0 | 899 | 4569 | 15093 |
| Puget/Georgia | 0 | 9655 | 0 | 0 | 689 | 4510 | 14854 |
| all | 26 | 96822 | 46 | 5961 | 15069 | 54415 | 172339 |

## Shift from B4 (relocated events, km)

| region | n | dh med / p90 | dz med | abs dz med / p90 | LSQR ex / ez med |
|---|---|---|---|---|---|
| Blanco | 4518 | 0.694 / 1.943 | -0.295 | 1.078 / 4.083 | 0.059 / 0.103 |
| Cascades | 1443 | 0.336 / 0.942 | -0.001 | 0.443 / 1.402 | 0.074 / 0.107 |
| all | 5961 | 0.593 / 1.768 | -0.181 | 0.856 / 3.406 | 0.062 / 0.105 |

## dt residual rms in the B4 time grids (s), B4 -> B5

Relocated core-core pairs; residual = observed minus B4-grid differential time, per-pair median removed (b5_compare.py metric of the D-B048 test).

| region | n P | P B4 -> B5 | n S | S B4 -> B5 |
|---|---|---|---|---|
| Blanco | 84104 | 0.166 -> 0.163 | 105612 | 0.191 -> 0.304 |
| Cascades | 40074 | 0.158 -> 0.159 | 23618 | 0.201 -> 0.252 |
| all | 124178 | 0.163 -> 0.162 | 129230 | 0.193 -> 0.295 |

## tomoDD dt rms (its own forward model, first -> last iteration, ms) and condition number

| region | clusters | rmsct first med | rmsct last med | CND first med / max | CND last med / max |
|---|---|---|---|---|---|
| Blanco | 2 | 124 | 84 | 71 / 135 | 62 / 117 |
| Cascades | 1 | 110 | 76 | 64 / 64 | 56 / 56 |

## Cluster shape (PCA of relocated core events, clusters with >= 10 events, km)

sigma3 = thickness across the best-fitting plane (smallest axis).

| region | clusters | sigma3 B4 med | sigma3 B5 med | B5/B4 med | thinner % | sigma1 B4 / B5 med |
|---|---|---|---|---|---|---|
| Blanco | 2 | 1.426 | 1.522 | 1.09 | 0 | 9.12 / 9.18 |
| Cascades | 1 | 0.338 | 0.528 | 1.56 | 0 | 1.79 / 1.91 |
| all | 3 | 0.372 | 0.528 | 1.12 | 0 | 1.79 / 1.91 |

## Centroid shift removed (tomoDD drift of each cluster's relocated core events, D-B053)

Horizontal median 0.074 km (p90 0.128, max 0.141); vertical median +0.067 km (abs p90 0.379, max abs 0.457); origin time median -0.0098 s. Weighted by events: horizontal 0.124 km, vertical 0.358 km.
Frame centre: every start position >= 50 m from it (min 86 m); final relocated positions min 56 m.

## Above the solid surface (D-B055)

surface_km = GMT earth_relief 01m (gridline, bilinear) at the final epicentre, above_surface = depth < -surface_km on the true depth (b4_catalog.py). Nothing is moved or dropped.

| region | B5 relocated above | B4 kept above | tier U above | B5 events above at their B4 position |
|---|---|---|---|---|
| Axial | 0 | 191 | 259 | 0 |
| Blanco | 7 | 125 | 1658 | 1 |
| Cascades | 2 | 261 | 411 | 0 |
| Endeavour | 0 | 5 | 38 | 0 |
| Explorer | 0 | 2 | 211 | 0 |
| Gorda | 0 | 82 | 451 | 0 |
| JdF ridge/plate | 0 | 73 | 253 | 0 |
| Mendocino | 0 | 56 | 237 | 0 |
| Offshore forearc | 0 | 28 | 242 | 0 |
| Other onshore | 0 | 161 | 160 | 0 |
| Puget/Georgia | 0 | 94 | 159 | 0 |
| all | 9 | 1078 | 4079 | 1 |

Total above the surface: 5166 (9 B5-relocated, 1078 B4 kept, 4079 tier U). NOT PUBLISHABLE under D-B055 (computation errors to find, not to move).
B5 events above: height above the surface median 116 m, max 1080 m; their B4 depth was below the surface for 9 of them.

## Run

3 clusters (0 chunks of split components), binaries {'e10k': 3}; status {'ok': 3}; tomoDD wall time total 0.1 h (largest cluster 7.0 min, peak RSS 0.96 GB).

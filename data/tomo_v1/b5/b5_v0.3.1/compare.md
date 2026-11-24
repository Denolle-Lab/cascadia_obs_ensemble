# B5 relocation b5_v0.3.1 (tomoDD-SE relocation only, variant dd, model v0.3.1)

Start: B4 run b4_iter1_v0.3.1 (NLLoc, full precision, EFT undone). Catalog: /wd1/mdenolle_data/tomo_v1/b5/b5_v0.3.1/catalog_b5_b5_v0.3.1.csv (172339 events). Depths in km below sea level. B5 = tomoDD solution with each cluster's centroid put back on its B4 centroid (D-B053).

## Events by location source

| region | events | b5_tomodd | nll3d (B4 kept) | v4 (tier U) | relocated % of B4 3D |
|---|---|---|---|---|---|
| Axial | 26731 | 17683 | 8688 | 360 | 67.1 |
| Blanco | 42385 | 21281 | 15010 | 6094 | 58.6 |
| Cascades | 17280 | 9091 | 6962 | 1227 | 56.6 |
| Endeavour | 179 | 10 | 86 | 83 | 10.4 |
| Explorer | 1523 | 446 | 398 | 679 | 52.8 |
| Gorda | 11378 | 5342 | 4326 | 1710 | 55.3 |
| JdF ridge/plate | 2851 | 525 | 1637 | 689 | 24.3 |
| Mendocino | 24643 | 15331 | 7899 | 1413 | 66.0 |
| Offshore forearc | 15422 | 8685 | 5510 | 1227 | 61.2 |
| Other onshore | 15093 | 8809 | 5383 | 901 | 62.1 |
| Puget/Georgia | 14854 | 1226 | 12941 | 687 | 8.7 |
| all | 172339 | 88429 | 68840 | 15070 | 56.2 |

## Why events were not relocated (b5_status)

| region | airquake_by_tomodd | cluster_failed | cluster_skipped | dropped_by_tomodd | relocated | tier_U | unlinked | all |
|---|---|---|---|---|---|---|---|---|
| Axial | 3039 | 0 | 0 | 328 | 17683 | 360 | 5321 | 26731 |
| Blanco | 30 | 0 | 1 | 118 | 21281 | 6094 | 14861 | 42385 |
| Cascades | 128 | 430 | 0 | 82 | 9091 | 1227 | 6322 | 17280 |
| Endeavour | 0 | 0 | 0 | 0 | 10 | 83 | 86 | 179 |
| Explorer | 0 | 0 | 0 | 2 | 446 | 679 | 396 | 1523 |
| Gorda | 27 | 0 | 2 | 147 | 5342 | 1710 | 4150 | 11378 |
| JdF ridge/plate | 124 | 0 | 5 | 30 | 525 | 689 | 1478 | 2851 |
| Mendocino | 26 | 0 | 2 | 224 | 15331 | 1413 | 7647 | 24643 |
| Offshore forearc | 23 | 121 | 2 | 173 | 8685 | 1227 | 5191 | 15422 |
| Other onshore | 140 | 473 | 2 | 201 | 8809 | 901 | 4567 | 15093 |
| Puget/Georgia | 14 | 8396 | 1 | 1 | 1226 | 687 | 4529 | 14854 |
| all | 3551 | 9420 | 15 | 1306 | 88429 | 15070 | 54548 | 172339 |

## Shift from B4 (relocated events, km)

| region | n | dh med / p90 | dz med | abs dz med / p90 | LSQR ex / ez med |
|---|---|---|---|---|---|
| Axial | 17683 | 0.447 / 1.283 | -0.045 | 0.337 / 1.293 | 0.024 / 0.025 |
| Blanco | 21281 | 0.450 / 1.635 | -0.007 | 0.359 / 2.421 | 0.068 / 0.106 |
| Cascades | 9091 | 0.512 / 1.946 | -0.011 | 0.434 / 2.251 | 0.119 / 0.125 |
| Endeavour | 10 | 0.013 / 0.021 | +0.002 | 0.007 / 0.017 | 0.246 / 0.307 |
| Explorer | 446 | 0.234 / 0.855 | +0.005 | 0.203 / 1.123 | 0.128 / 0.173 |
| Gorda | 5342 | 0.656 / 2.512 | -0.013 | 0.829 / 4.732 | 0.147 / 0.248 |
| JdF ridge/plate | 525 | 0.192 / 3.486 | +0.002 | 0.092 / 0.671 | 0.118 / 0.100 |
| Mendocino | 15331 | 0.915 / 2.840 | +0.134 | 0.738 / 3.229 | 0.081 / 0.098 |
| Offshore forearc | 8685 | 0.722 / 2.332 | -0.031 | 0.875 / 4.049 | 0.106 / 0.162 |
| Other onshore | 8809 | 0.718 / 2.627 | +0.007 | 0.633 / 3.819 | 0.120 / 0.159 |
| Puget/Georgia | 1226 | 0.162 / 0.743 | +0.002 | 0.099 / 0.530 | 0.159 / 0.180 |
| all | 88429 | 0.561 / 2.074 | -0.002 | 0.474 / 2.741 | 0.078 / 0.104 |

## dt residual rms in the B4 time grids (s), B4 -> B5

Relocated core-core pairs; residual = observed minus B4-grid differential time, per-pair median removed (b5_compare.py metric of the D-B048 test).

| region | n P | P B4 -> B5 | n S | S B4 -> B5 |
|---|---|---|---|---|
| Axial | 377807 | 0.079 -> 0.081 | 327339 | 0.123 -> 0.158 |
| Blanco | 316791 | 0.167 -> 0.174 | 386199 | 0.207 -> 0.274 |
| Cascades | 300906 | 0.152 -> 0.156 | 323054 | 0.166 -> 0.199 |
| Endeavour | 32 | 0.493 -> 0.493 | 27 | 0.534 -> 0.533 |
| Explorer | 6627 | 0.175 -> 0.162 | 6611 | 0.274 -> 0.279 |
| Gorda | 72419 | 0.197 -> 0.178 | 97559 | 0.223 -> 0.313 |
| JdF ridge/plate | 5025 | 0.159 -> 0.162 | 4206 | 0.193 -> 0.221 |
| Mendocino | 276797 | 0.149 -> 0.142 | 277909 | 0.182 -> 0.228 |
| Offshore forearc | 166926 | 0.146 -> 0.140 | 154438 | 0.202 -> 0.249 |
| Other onshore | 233600 | 0.158 -> 0.161 | 223681 | 0.185 -> 0.235 |
| Puget/Georgia | 25631 | 0.179 -> 0.176 | 19468 | 0.203 -> 0.225 |
| all | 1782561 | 0.145 -> 0.146 | 1820491 | 0.181 -> 0.231 |

## tomoDD dt rms (its own forward model, first -> last iteration, ms) and condition number

| region | clusters | rmsct first med | rmsct last med | CND first med / max | CND last med / max |
|---|---|---|---|---|---|
| Axial | 44 | 162 | 112 | 2 / 279 | 2 / 208 |
| Blanco | 798 | 143 | 111 | 2 / 178 | 2 / 153 |
| Cascades | 209 | 188 | 147 | 2 / 61 | 2 / 54 |
| Endeavour | 3 | 262 | 193 | 2 / 2 | 2 / 2 |
| Explorer | 28 | 200 | 134 | 2 / 9 | 2 / 8 |
| Gorda | 211 | 226 | 188 | 2 / 18 | 2 / 16 |
| JdF ridge/plate | 57 | 147 | 116 | 2 / 10 | 2 / 9 |
| Mendocino | 180 | 169 | 127 | 2 / 268 | 2 / 215 |
| Offshore forearc | 218 | 174 | 137 | 2 / 64 | 2 / 53 |
| Other onshore | 235 | 194 | 162 | 2 / 23 | 2 / 21 |
| Puget/Georgia | 121 | 136 | 108 | 2 / 339 | 2 / 284 |

## Cluster shape (PCA of relocated core events, clusters with >= 10 events, km)

sigma3 = thickness across the best-fitting plane (smallest axis).

| region | clusters | sigma3 B4 med | sigma3 B5 med | B5/B4 med | thinner % | sigma1 B4 / B5 med |
|---|---|---|---|---|---|---|
| Axial | 5 | 0.220 | 0.216 | 0.98 | 60 | 0.69 / 0.68 |
| Blanco | 135 | 0.503 | 0.518 | 1.01 | 41 | 1.77 / 1.81 |
| Cascades | 40 | 1.166 | 1.168 | 1.00 | 57 | 5.62 / 5.51 |
| Explorer | 8 | 1.809 | 1.872 | 1.01 | 50 | 9.36 / 9.18 |
| Gorda | 38 | 1.396 | 1.389 | 1.04 | 24 | 5.53 / 5.58 |
| JdF ridge/plate | 6 | 0.688 | 0.644 | 0.99 | 50 | 3.45 / 3.49 |
| Mendocino | 20 | 1.167 | 1.172 | 1.02 | 40 | 4.57 / 4.54 |
| Offshore forearc | 35 | 1.541 | 1.426 | 0.99 | 60 | 4.34 / 4.28 |
| Other onshore | 49 | 1.564 | 1.661 | 1.00 | 51 | 5.42 / 5.43 |
| Puget/Georgia | 28 | 0.936 | 0.871 | 1.00 | 54 | 2.91 / 2.90 |
| all | 364 | 0.883 | 0.905 | 1.01 | 46 | 3.33 / 3.33 |

## Centroid shift removed (tomoDD drift of each cluster's relocated core events, D-B053)

Horizontal median 0.002 km (p90 0.037, max 0.380); vertical median -0.000 km (abs p90 0.047, max abs 0.589); origin time median +0.0000 s. Weighted by events: horizontal 0.087 km, vertical 0.178 km.
Frame centre: every start position >= 50 m from it (min 50 m); final relocated positions min 0 m.

## Split effect (events relocated as core in one chunk and as halo in another)

2514 duplicates: horizontal difference median 0.290 km (p90 0.996), vertical 0.245 km (p90 0.838). By hop: hop 1: n 1209, dh 0.259, dz 0.241; hop 2: n 1305, dh 0.326, dz 0.252

## Above the solid surface (D-B055)

surface_km = GMT earth_relief 01m (gridline, bilinear) at the final epicentre, above_surface = depth < -surface_km on the true depth (b4_catalog.py). Nothing is moved or dropped.

| region | B5 relocated above | B4 kept above | tier U above | B5 events above at their B4 position |
|---|---|---|---|---|
| Axial | 37 | 0 | 259 | 0 |
| Blanco | 159 | 0 | 1655 | 0 |
| Cascades | 101 | 0 | 411 | 0 |
| Endeavour | 0 | 0 | 37 | 0 |
| Explorer | 0 | 0 | 212 | 0 |
| Gorda | 36 | 0 | 451 | 0 |
| JdF ridge/plate | 88 | 0 | 253 | 0 |
| Mendocino | 40 | 0 | 237 | 0 |
| Offshore forearc | 23 | 0 | 243 | 0 |
| Other onshore | 286 | 0 | 160 | 0 |
| Puget/Georgia | 33 | 0 | 159 | 0 |
| all | 803 | 0 | 4077 | 0 |

Total above the surface: 4880 (803 B5-relocated, 0 B4 kept, 4077 tier U). NOT PUBLISHABLE under D-B055 (computation errors to find, not to move).
B5 events above: height above the surface median 224 m, max 5237 m; their B4 depth was below the surface for 747 of them.

## Run

2128 clusters (6 chunks of split components), binaries {'e10k': 2115, 'o140': 7, 'e13k': 4, 'w': 1, 'm': 1}; status {'ok': 2104, 'skipped': 23, 'failed': 1}; tomoDD wall time total 1.9 h (largest cluster 19.7 min, peak RSS 3.23 GB).

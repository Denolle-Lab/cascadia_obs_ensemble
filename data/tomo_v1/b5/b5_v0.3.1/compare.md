# B5 relocation b5_v0.3.1 (tomoDD-SE relocation only, variant dd, model v0.3.1)

Start: B4 run b4_iter1_v0.3.1 (NLLoc, full precision, EFT undone). Catalog: /wd1/mdenolle_data/tomo_v1/b5/b5_v0.3.1/catalog_b5_b5_v0.3.1.csv (172339 events). Depths in km below sea level. B5 = tomoDD solution with each cluster's centroid put back on its B4 centroid (D-B053).

## Events by location source

| region | events | b5_tomodd | nll3d (B4 kept) | v4 (tier U) | relocated % of B4 3D |
|---|---|---|---|---|---|
| Axial | 26731 | 17683 | 8688 | 360 | 67.1 |
| Blanco | 42385 | 21281 | 15010 | 6094 | 58.6 |
| Cascades | 17280 | 9490 | 6563 | 1227 | 59.1 |
| Endeavour | 179 | 10 | 86 | 83 | 10.4 |
| Explorer | 1523 | 446 | 398 | 679 | 52.8 |
| Gorda | 11378 | 5342 | 4326 | 1710 | 55.3 |
| JdF ridge/plate | 2851 | 525 | 1637 | 689 | 24.3 |
| Mendocino | 24643 | 15331 | 7899 | 1413 | 66.0 |
| Offshore forearc | 15422 | 8804 | 5391 | 1227 | 62.0 |
| Other onshore | 15093 | 9260 | 4932 | 901 | 65.2 |
| Puget/Georgia | 14854 | 9447 | 4720 | 687 | 66.7 |
| all | 172339 | 97619 | 59650 | 15070 | 62.1 |

## Why events were not relocated (b5_status)

| region | airquake_by_tomodd | cluster_skipped | dropped_by_tomodd | relocated | tier_U | unlinked | all |
|---|---|---|---|---|---|---|---|
| Axial | 3039 | 0 | 328 | 17683 | 360 | 5321 | 26731 |
| Blanco | 30 | 1 | 118 | 21281 | 6094 | 14861 | 42385 |
| Cascades | 145 | 0 | 96 | 9490 | 1227 | 6322 | 17280 |
| Endeavour | 0 | 0 | 0 | 10 | 83 | 86 | 179 |
| Explorer | 0 | 0 | 2 | 446 | 679 | 396 | 1523 |
| Gorda | 27 | 2 | 147 | 5342 | 1710 | 4150 | 11378 |
| JdF ridge/plate | 124 | 5 | 30 | 525 | 689 | 1478 | 2851 |
| Mendocino | 26 | 2 | 224 | 15331 | 1413 | 7647 | 24643 |
| Offshore forearc | 23 | 2 | 175 | 8804 | 1227 | 5191 | 15422 |
| Other onshore | 143 | 2 | 220 | 9260 | 901 | 4567 | 15093 |
| Puget/Georgia | 50 | 1 | 140 | 9447 | 687 | 4529 | 14854 |
| all | 3607 | 15 | 1480 | 97619 | 15070 | 54548 | 172339 |

## Shift from B4 (relocated events, km)

| region | n | dh med / p90 | dz med | abs dz med / p90 | LSQR ex / ez med |
|---|---|---|---|---|---|
| Axial | 17683 | 0.447 / 1.283 | -0.045 | 0.337 / 1.293 | 0.024 / 0.025 |
| Blanco | 21281 | 0.450 / 1.635 | -0.007 | 0.359 / 2.421 | 0.068 / 0.106 |
| Cascades | 9490 | 0.532 / 1.988 | -0.009 | 0.457 / 2.420 | 0.122 / 0.128 |
| Endeavour | 10 | 0.013 / 0.021 | +0.002 | 0.007 / 0.017 | 0.246 / 0.307 |
| Explorer | 446 | 0.234 / 0.855 | +0.005 | 0.203 / 1.123 | 0.128 / 0.173 |
| Gorda | 5342 | 0.656 / 2.512 | -0.013 | 0.829 / 4.732 | 0.147 / 0.248 |
| JdF ridge/plate | 525 | 0.192 / 3.486 | +0.002 | 0.092 / 0.671 | 0.118 / 0.100 |
| Mendocino | 15331 | 0.915 / 2.840 | +0.134 | 0.738 / 3.229 | 0.081 / 0.098 |
| Offshore forearc | 8804 | 0.729 / 2.355 | -0.035 | 0.875 / 4.022 | 0.107 / 0.162 |
| Other onshore | 9260 | 0.727 / 2.644 | +0.003 | 0.657 / 3.825 | 0.120 / 0.158 |
| Puget/Georgia | 9447 | 0.829 / 2.498 | -0.026 | 0.919 / 4.060 | 0.121 / 0.163 |
| all | 97619 | 0.594 / 2.145 | -0.004 | 0.521 / 2.955 | 0.084 / 0.111 |

## dt residual rms in the B4 time grids (s), B4 -> B5

Relocated core-core pairs; residual = observed minus B4-grid differential time, per-pair median removed (b5_compare.py metric of the D-B048 test).

| region | n P | P B4 -> B5 | n S | S B4 -> B5 |
|---|---|---|---|---|
| Axial | 377807 | 0.079 -> 0.081 | 327339 | 0.123 -> 0.158 |
| Blanco | 316791 | 0.167 -> 0.174 | 386199 | 0.207 -> 0.274 |
| Cascades | 316116 | 0.154 -> 0.158 | 338001 | 0.167 -> 0.202 |
| Endeavour | 32 | 0.493 -> 0.493 | 27 | 0.534 -> 0.533 |
| Explorer | 6627 | 0.175 -> 0.162 | 6611 | 0.274 -> 0.279 |
| Gorda | 72419 | 0.197 -> 0.178 | 97559 | 0.223 -> 0.313 |
| JdF ridge/plate | 5025 | 0.159 -> 0.162 | 4206 | 0.193 -> 0.221 |
| Mendocino | 276797 | 0.149 -> 0.142 | 277909 | 0.182 -> 0.228 |
| Offshore forearc | 168931 | 0.146 -> 0.141 | 156173 | 0.202 -> 0.249 |
| Other onshore | 244471 | 0.158 -> 0.161 | 233843 | 0.186 -> 0.236 |
| Puget/Georgia | 255375 | 0.159 -> 0.164 | 268848 | 0.182 -> 0.234 |
| all | 2040391 | 0.147 -> 0.148 | 2096715 | 0.181 -> 0.232 |

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
| Puget/Georgia | 122 | 134 | 108 | 2 / 339 | 2 / 284 |

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
| Puget/Georgia | 29 | 0.958 | 0.929 | 1.00 | 52 | 2.91 / 2.93 |
| all | 365 | 0.886 | 0.910 | 1.01 | 45 | 3.34 / 3.33 |

## Centroid shift removed (tomoDD drift of each cluster's relocated core events, D-B053)

Horizontal median 0.002 km (p90 0.037, max 0.380); vertical median -0.000 km (abs p90 0.048, max abs 0.589); origin time median +0.0000 s. Weighted by events: horizontal 0.085 km, vertical 0.188 km.
Frame centre: every start position >= 50 m from it (min 50 m); final relocated positions min 0 m.

## Split effect (events relocated as core in one chunk and as halo in another)

2673 duplicates: horizontal difference median 0.305 km (p90 1.059), vertical 0.249 km (p90 0.942). By hop: hop 1: n 1279, dh 0.269, dz 0.243; hop 2: n 1394, dh 0.340, dz 0.260

## Above the solid surface (D-B055)

surface_km = GMT earth_relief 01m (gridline, bilinear) at the final epicentre, above_surface = depth < -surface_km on the true depth (b4_catalog.py). Nothing is moved or dropped.

| region | B5 relocated above | B4 kept above | tier U above | B5 events above at their B4 position |
|---|---|---|---|---|
| Axial | 37 | 0 | 259 | 0 |
| Blanco | 159 | 0 | 1655 | 0 |
| Cascades | 102 | 0 | 411 | 0 |
| Endeavour | 0 | 0 | 37 | 0 |
| Explorer | 0 | 0 | 212 | 0 |
| Gorda | 36 | 0 | 451 | 0 |
| JdF ridge/plate | 88 | 0 | 253 | 0 |
| Mendocino | 40 | 0 | 237 | 0 |
| Offshore forearc | 23 | 0 | 243 | 0 |
| Other onshore | 286 | 0 | 160 | 0 |
| Puget/Georgia | 98 | 0 | 159 | 0 |
| all | 869 | 0 | 4077 | 0 |

Total above the surface: 4946 (869 B5-relocated, 0 B4 kept, 4077 tier U). NOT PUBLISHABLE under D-B055 (computation errors to find, not to move).
B5 events above: height above the surface median 249 m, max 5237 m; 869 of them were below the surface at their B4 position; B4 depth below the surface median 76 m (all relocated events 12478 m); B5 - B4 depth median -0.560 km.
Above-surface B5 events by start depth below the surface (B4): < 0.1 km: 479 of 1334, 0.1-0.5 km: 152 of 1194, 0.5-1 km: 63 of 1602, 1-3 km: 118 of 18185, >= 3 km: 57 of 75304.
Tier U above the surface: 4077 (v4 hypocentres, D-B057; not relocated here).

## Run

2128 clusters (6 chunks of split components), binaries {'e10k': 2115, 'o140': 7, 'e13k': 4, 'w': 1, 'm': 1}; status {'ok': 2105, 'skipped': 23}; tomoDD wall time total 2.1 h (largest cluster 20.2 min, peak RSS 3.23 GB).

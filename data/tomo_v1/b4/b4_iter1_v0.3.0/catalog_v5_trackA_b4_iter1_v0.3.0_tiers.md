# Catalog v5 Track A (b4_iter1_v0.3.0): location-quality tiers

172340 events (all of catalog v4). File: `/wd1/mdenolle_data/reloc_v5/catalog/catalog_v5_trackA_b4_iter1_v0.3.0.csv`

Tier definitions: see `workflow/08_relocation/a2_build_catalog.py`.

## Tiers

| tier | n | rms | nphs | h_max | sig_z | depth | share |
|---|---|---|---|---|---|---|---|
| A | 17892 | 0.13 | 23.0 | 1.31 | 1.11 | 19.18 | 10.4 |
| B | 25427 | 0.13 | 13.0 | 2.43 | 2.25 | 17.32 | 14.8 |
| C | 72263 | 0.12 | 9.0 | 3.71 | 5.41 | 11.68 | 41.9 |
| D | 41688 | 0.08 | 8.0 | 7.07 | 4.22 | 5.7 | 24.2 |
| U | 15070 |  |  |  |  | 8.26 | 8.7 |

## Tiers by region (counts)

| region | A | B | C | D | U | All |
|---|---|---|---|---|---|---|
| Axial | 237 | 1049 | 5343 | 19742 | 360 | 26731 |
| Blanco | 829 | 3313 | 29761 | 2383 | 6099 | 42385 |
| Cascades | 4488 | 2855 | 5572 | 3138 | 1227 | 17280 |
| Endeavour | 0 | 2 | 39 | 54 | 84 | 179 |
| Explorer | 0 | 4 | 630 | 211 | 678 | 1523 |
| Gorda | 373 | 2303 | 5722 | 1269 | 1711 | 11378 |
| JdF ridge/plate | 0 | 21 | 1057 | 1089 | 684 | 2851 |
| Mendocino | 5430 | 4846 | 7695 | 5261 | 1411 | 24643 |
| Offshore forearc | 1617 | 3938 | 5534 | 3106 | 1227 | 15422 |
| Other onshore | 2632 | 3143 | 5379 | 3040 | 900 | 15094 |
| Puget/Georgia | 2286 | 3953 | 5531 | 2395 | 689 | 14854 |
| All | 17892 | 25427 | 72263 | 41688 | 15070 | 172340 |

## Check against ANSS/ComCat (onshore, Mendocino, offshore forearc)

dz = this catalog minus ANSS depth (km); abs_dz_v4 = same for catalog v4.

| tier | n | dz_median | abs_dz | abs_dz_v4 | dh_median |
|---|---|---|---|---|---|
| A | 5705 | 0.86 | 1.49 | 1.69 | 1.29 |
| B | 3813 | 0.87 | 2.29 | 2.48 | 1.75 |
| C | 5048 | 2.4 | 4.56 | 4.22 | 2.55 |
| D | 1007 | 1.54 | 5.95 | 5.79 | 5.75 |


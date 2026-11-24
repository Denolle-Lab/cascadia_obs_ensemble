# Catalog v5 Track A (b4_iter1_v0.3.1): location-quality tiers

172340 events (all of catalog v4). File: `/wd1/mdenolle_data/reloc_v5/catalog/catalog_v5_trackA_b4_iter1_v0.3.1.csv`

Tier definitions: see `workflow/08_relocation/a2_build_catalog.py`.

## Tiers

| tier | n | rms | nphs | h_max | sig_z | depth | share |
|---|---|---|---|---|---|---|---|
| A | 18274 | 0.13 | 23.0 | 1.33 | 1.11 | 18.99 | 10.6 |
| B | 27011 | 0.13 | 13.0 | 2.37 | 2.17 | 16.51 | 15.7 |
| C | 76687 | 0.12 | 9.0 | 3.59 | 5.25 | 9.83 | 44.5 |
| D | 35297 | 0.08 | 8.0 | 14.03 | 6.62 | 5.59 | 20.5 |
| U | 15071 |  |  |  |  | 8.26 | 8.7 |

## Tiers by region (counts)

| region | A | B | C | D | U | All |
|---|---|---|---|---|---|---|
| Axial | 738 | 2736 | 10013 | 12884 | 360 | 26731 |
| Blanco | 712 | 3242 | 29476 | 2861 | 6094 | 42385 |
| Cascades | 4493 | 2849 | 5573 | 3138 | 1227 | 17280 |
| Endeavour | 0 | 2 | 42 | 52 | 83 | 179 |
| Explorer | 0 | 7 | 624 | 213 | 679 | 1523 |
| Gorda | 366 | 2290 | 5725 | 1287 | 1710 | 11378 |
| JdF ridge/plate | 0 | 2 | 1081 | 1079 | 689 | 2851 |
| Mendocino | 5447 | 4834 | 7704 | 5245 | 1413 | 24643 |
| Offshore forearc | 1605 | 3952 | 5531 | 3107 | 1227 | 15422 |
| Other onshore | 2629 | 3145 | 5381 | 3037 | 902 | 15094 |
| Puget/Georgia | 2284 | 3952 | 5537 | 2394 | 687 | 14854 |
| All | 18274 | 27011 | 76687 | 35297 | 15071 | 172340 |

## Check against ANSS/ComCat (onshore, Mendocino, offshore forearc)

dz = this catalog minus ANSS depth (km); abs_dz_v4 = same for catalog v4.

| tier | n | dz_median | abs_dz | abs_dz_v4 | dh_median |
|---|---|---|---|---|---|
| A | 5700 | 0.85 | 1.49 | 1.7 | 1.29 |
| B | 3820 | 0.85 | 2.28 | 2.47 | 1.75 |
| C | 5055 | 2.34 | 4.52 | 4.22 | 2.56 |
| D | 997 | 1.53 | 5.97 | 5.78 | 5.89 |


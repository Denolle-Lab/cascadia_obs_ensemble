# Catalog v5 Track A (iter1_vpvs176): location-quality tiers

172340 events (all of catalog v4). File: `/wd1/mdenolle_data/reloc_v5/catalog/catalog_v5_trackA_iter1_vpvs176.csv`

Tier definitions: see `workflow/08_relocation/a2_build_catalog.py`.

## Tiers

| tier | n | rms | nphs | h_max | sig_z | depth | share |
|---|---|---|---|---|---|---|---|
| A | 20704 | 0.13 | 18.0 | 1.34 | 1.07 | 14.72 | 12.0 |
| B | 33506 | 0.13 | 12.0 | 2.14 | 1.89 | 13.96 | 19.4 |
| C | 75379 | 0.13 | 9.0 | 3.68 | 5.25 | 11.01 | 43.7 |
| D | 27678 | 0.1 | 7.0 | 17.9 | 8.35 | 10.09 | 16.1 |
| U | 15073 |  |  |  |  | 8.25 | 8.7 |

## Tiers by region (counts)

| region | A | B | C | D | U | All |
|---|---|---|---|---|---|---|
| Axial | 4549 | 8595 | 8145 | 5082 | 360 | 26731 |
| Blanco | 789 | 3484 | 29316 | 2698 | 6098 | 42385 |
| Cascades | 4463 | 2820 | 5565 | 3204 | 1228 | 17280 |
| Endeavour | 0 | 2 | 45 | 49 | 83 | 179 |
| Explorer | 0 | 3 | 605 | 234 | 681 | 1523 |
| Gorda | 112 | 2057 | 6039 | 1456 | 1714 | 11378 |
| JdF ridge/plate | 0 | 1 | 1032 | 1133 | 685 | 2851 |
| Mendocino | 4922 | 4999 | 7996 | 5310 | 1416 | 24643 |
| Offshore forearc | 1218 | 4240 | 5623 | 3116 | 1225 | 15422 |
| Other onshore | 2499 | 3228 | 5439 | 3027 | 901 | 15094 |
| Puget/Georgia | 2152 | 4077 | 5574 | 2369 | 682 | 14854 |
| All | 20704 | 33506 | 75379 | 27678 | 15073 | 172340 |

## Check against ANSS/ComCat (onshore, Mendocino, offshore forearc)

dz = this catalog minus ANSS depth (km); abs_dz_v4 = same for catalog v4.

| tier | n | dz_median | abs_dz | abs_dz_v4 | dh_median |
|---|---|---|---|---|---|
| A | 5408 | 0.59 | 1.49 | 1.64 | 1.57 |
| B | 4067 | 0.98 | 2.74 | 2.55 | 2.38 |
| C | 5098 | 2.36 | 5.08 | 4.21 | 3.28 |
| D | 998 | 2.83 | 6.31 | 5.55 | 6.39 |


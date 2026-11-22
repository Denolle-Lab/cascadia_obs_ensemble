# Catalog v5 Track A (iter1_vpvs176): location-quality tiers

172340 events (all of catalog v4). File: `/wd1/mdenolle_data/reloc_v5/catalog/catalog_v5_trackA_iter1_vpvs176.csv`

Tier definitions: see `workflow/08_relocation/a2_build_catalog.py`.

## Tiers

| tier | n | rms | nphs | h_max | sig_z | depth | share |
|---|---|---|---|---|---|---|---|
| A | 22499 | 0.13 | 18.0 | 1.35 | 1.11 | 15.03 | 13.1 |
| B | 37783 | 0.13 | 13.0 | 2.15 | 2.22 | 15.33 | 21.9 |
| C | 69307 | 0.13 | 9.0 | 3.96 | 5.54 | 10.05 | 40.2 |
| D | 27678 | 0.1 | 7.0 | 17.9 | 8.35 | 10.09 | 16.1 |
| U | 15073 |  |  |  |  | 8.25 | 8.7 |

## Tiers by region (counts)

| region | A | B | C | D | U | All |
|---|---|---|---|---|---|---|
| Axial | 4549 | 8595 | 8145 | 5082 | 360 | 26731 |
| Blanco | 1519 | 6428 | 25642 | 2698 | 6098 | 42385 |
| Cascades | 4569 | 2970 | 5309 | 3204 | 1228 | 17280 |
| Endeavour | 0 | 2 | 45 | 49 | 83 | 179 |
| Explorer | 0 | 42 | 566 | 234 | 681 | 1523 |
| Gorda | 198 | 2474 | 5536 | 1456 | 1714 | 11378 |
| JdF ridge/plate | 0 | 2 | 1031 | 1133 | 685 | 2851 |
| Mendocino | 5006 | 5126 | 7785 | 5310 | 1416 | 24643 |
| Offshore forearc | 1533 | 4329 | 5219 | 3116 | 1225 | 15422 |
| Other onshore | 2735 | 3328 | 5103 | 3027 | 901 | 15094 |
| Puget/Georgia | 2390 | 4487 | 4926 | 2369 | 682 | 14854 |
| All | 22499 | 37783 | 69307 | 27678 | 15073 | 172340 |

## Check against ANSS/ComCat (onshore, Mendocino, offshore forearc)

dz = this catalog minus ANSS depth (km); abs_dz_v4 = same for catalog v4.

| tier | n | dz_median | abs_dz | abs_dz_v4 | dh_median |
|---|---|---|---|---|---|
| A | 5735 | 0.67 | 1.55 | 1.69 | 1.59 |
| B | 4462 | 1.24 | 2.99 | 2.75 | 2.45 |
| C | 4376 | 1.9 | 4.93 | 4.18 | 3.45 |
| D | 998 | 2.83 | 6.31 | 5.55 | 6.39 |


# Catalog v5 Track A (b4_v51_iter1): location-quality tiers

172342 events (catalog v4.1 event list). File: `/wd1/mdenolle_data/reloc_v5/catalog/catalog_v5_trackA_b4_v51_iter1.csv`

Tier definitions: see `workflow/08_relocation/a2_build_catalog.py`.

## Tiers

| tier | n | rms | nphs | h_max | sig_z | depth | share |
|---|---|---|---|---|---|---|---|
| A | 21325 | 0.13 | 19.0 | 1.42 | 1.23 | 17.02 | 12.4 |
| B | 33896 | 0.12 | 12.0 | 2.24 | 2.0 | 13.63 | 19.7 |
| C | 73464 | 0.14 | 10.0 | 3.48 | 4.98 | 9.98 | 42.6 |
| D | 43032 | 0.09 | 6.0 | 22.96 | 11.09 | 12.04 | 25.0 |
| U | 625 |  |  |  |  |  | 0.4 |

## Tiers by region (counts)

| region | A | B | C | D | U | All |
|---|---|---|---|---|---|---|
| Axial | 3909 | 9066 | 8350 | 5401 | 5 | 26731 |
| Blanco | 700 | 3565 | 28767 | 9317 | 36 | 42385 |
| Cascades | 4338 | 2838 | 5246 | 4671 | 187 | 17280 |
| Endeavour | 0 | 4 | 42 | 132 | 1 | 179 |
| Explorer | 0 | 19 | 972 | 523 | 9 | 1523 |
| Gorda | 437 | 2536 | 5573 | 2804 | 30 | 11380 |
| JdF ridge/plate | 0 | 7 | 975 | 1861 | 8 | 2851 |
| Mendocino | 5399 | 4907 | 7531 | 6771 | 36 | 24644 |
| Offshore forearc | 1613 | 3968 | 5473 | 4344 | 24 | 15422 |
| Other onshore | 2652 | 3110 | 5132 | 3931 | 268 | 15093 |
| Puget/Georgia | 2277 | 3876 | 5403 | 3277 | 21 | 14854 |
| All | 21325 | 33896 | 73464 | 43032 | 625 | 172342 |

## Not located (tier U): why (LOCMETH minimum 5 phases, from the run's nlloc.in)

| u_reason | n |
|---|---|
| rejected: max likelihood on box face 31 | 307 |
| rejected: max likelihood on box face 10 | 185 |
| rejected: max likelihood on box face 20 | 74 |
| rejected: max likelihood on box face 21 | 32 |
| rejected: max likelihood on box face 11 | 27 |

## Check against ANSS/ComCat (onshore, Mendocino, offshore forearc)

dz = this catalog minus ANSS depth (km); abs_dz_v4 = same for catalog v4.

| tier | n | dz_median | abs_dz | abs_dz_v4 | dh_median |
|---|---|---|---|---|---|
| A | 5627 | 0.97 | 1.61 | 1.71 | 1.33 |
| B | 3875 | 1.1 | 2.42 | 2.45 | 1.86 |
| C | 5189 | 1.54 | 4.41 | 4.32 | 2.69 |
| D | 1148 | 1.05 | 5.18 | 5.4 | 5.74 |


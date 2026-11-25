# B4 relocation b4_iter1_v0.3.1 against iter1_vpvs176

Catalog: /wd1/mdenolle_data/reloc_v5/catalog/catalog_b4_b4_iter1_v0.3.1.csv (172340 events, 157269 located in 3D). Depths in km below sea level, EFT undone. Differences B4 minus reference, events located in both.

## Tiers

| tier | B4 | reference |
|---|---|---|
| A | 18274 | 20704 |
| B | 27011 | 33506 |
| C | 76687 | 75379 |
| D | 35297 | 27678 |
| U | 15071 | 15073 |

## By region

| region | n | dz med | abs dz med / p90 | dh med / p90 | dt med s | rms B4 / ref |
|---|---|---|---|---|---|---|
| Axial | 26371 | +0.74 | 0.96 / 1.79 | 1.31 / 2.56 | -0.105 | 0.060 / 0.054 |
| Blanco | 36285 | -0.60 | 1.66 / 10.68 | 0.87 / 2.28 | -0.016 | 0.142 / 0.142 |
| Cascades | 16052 | -0.01 | 0.45 / 3.08 | 0.49 / 2.45 | +0.062 | 0.109 / 0.114 |
| Endeavour | 96 | -0.00 | 0.30 / 28.99 | 1.42 / 10.20 | -0.145 | 0.137 / 0.122 |
| Explorer | 840 | +2.04 | 4.50 / 15.90 | 2.06 / 5.45 | -0.103 | 0.142 / 0.194 |
| Gorda | 9659 | -1.30 | 3.04 / 14.43 | 1.61 / 5.33 | +0.160 | 0.161 / 0.209 |
| JdF ridge/plate | 2161 | -0.11 | 0.52 / 32.44 | 1.88 / 11.01 | -0.105 | 0.129 / 0.121 |
| Mendocino | 23215 | -0.18 | 0.90 / 6.54 | 1.37 / 4.38 | +0.063 | 0.116 / 0.133 |
| Offshore forearc | 14193 | -0.30 | 1.33 / 6.43 | 1.18 / 4.12 | +0.109 | 0.125 / 0.143 |
| Other onshore | 14190 | -0.11 | 0.97 / 4.92 | 0.87 / 3.21 | +0.093 | 0.127 / 0.141 |
| Puget/Georgia | 14164 | -0.30 | 1.20 / 5.16 | 1.49 / 4.20 | +0.220 | 0.119 / 0.141 |
| all | 157226 | -0.08 | 1.11 / 6.42 | 1.12 / 3.36 | +0.027 | 0.116 / 0.124 |

Above the surface (located in both): B4 0.0 %, reference 0.4 % (reference: a2_parse_hyp.py check, no EFT).
D-B055: 4078 events above the surface (0 located in 3D, 4078 tier U at the v4 position). NOT PUBLISHABLE: these are computation errors to fix, not to move.
The ANSS check per tier is in the tier summary of a2_build_catalog.py (copied next to this file).

## Flags (rule 4 of D-B049..D-B051)

| region | biased_deep | ok | unconstrained | all |
|---|---|---|---|---|
| Axial | 0 | 26731 | 0 | 26731 |
| Blanco | 0 | 21949 | 20436 | 42385 |
| Cascades | 0 | 17280 | 0 | 17280 |
| Endeavour | 0 | 0 | 179 | 179 |
| Explorer | 0 | 0 | 1523 | 1523 |
| Gorda | 7848 | 3530 | 0 | 11378 |
| JdF ridge/plate | 0 | 0 | 2851 | 2851 |
| Mendocino | 0 | 24643 | 0 | 24643 |
| Offshore forearc | 0 | 15422 | 0 | 15422 |
| Other onshore | 0 | 15094 | 0 | 15094 |
| Puget/Georgia | 0 | 14854 | 0 | 14854 |
| all | 7848 | 139503 | 24989 | 172340 |

event_flag quarry_blast_likely: 1573 events (Texada Island box, weekdays 07:00-19:00 America/Vancouver).

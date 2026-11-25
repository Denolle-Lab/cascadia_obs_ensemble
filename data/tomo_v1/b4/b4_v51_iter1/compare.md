# B4 relocation b4_v51_iter1 against iter1_vpvs176

Catalog: /wd1/mdenolle_data/reloc_v5/catalog/catalog_b4_b4_v51_iter1.csv (172340 events, 171717 located in 3D). Depths in km below sea level, EFT undone. Differences B4 minus reference, events located in both.

## Tiers

| tier | B4 | reference |
|---|---|---|
| A | 21325 | 20704 |
| B | 33896 | 33506 |
| C | 73464 | 75379 |
| D | 43032 | 27678 |
| U | 623 | 15073 |

## By region

| region | n | dz med | abs dz med / p90 | dh med / p90 | dt med s | rms B4 / ref |
|---|---|---|---|---|---|---|
| Axial | 26366 | +0.13 | 0.44 / 1.44 | 0.40 / 1.49 | +0.109 | 0.070 / 0.054 |
| Blanco | 36286 | -0.56 | 1.69 / 11.45 | 0.96 / 2.76 | -0.022 | 0.153 / 0.142 |
| Cascades | 15932 | +0.14 | 0.65 / 4.59 | 0.67 / 3.23 | +0.043 | 0.119 / 0.114 |
| Endeavour | 95 | +0.29 | 0.44 / 26.48 | 1.45 / 11.31 | +0.038 | 0.138 / 0.122 |
| Explorer | 842 | +4.94 | 6.35 / 28.97 | 2.59 / 8.74 | -0.123 | 0.189 / 0.193 |
| Gorda | 9663 | -1.17 | 3.45 / 15.36 | 1.73 / 6.39 | +0.150 | 0.163 / 0.209 |
| JdF ridge/plate | 2164 | -0.07 | 0.52 / 31.93 | 2.02 / 13.52 | -0.043 | 0.119 / 0.121 |
| Mendocino | 23216 | -0.01 | 1.04 / 6.92 | 1.59 / 5.14 | +0.010 | 0.127 / 0.133 |
| Offshore forearc | 14194 | -0.28 | 1.45 / 7.69 | 1.51 / 5.06 | +0.062 | 0.131 / 0.143 |
| Other onshore | 13956 | -0.07 | 1.15 / 7.18 | 1.02 / 4.51 | +0.065 | 0.127 / 0.140 |
| Puget/Georgia | 14158 | -0.39 | 1.40 / 6.12 | 1.62 / 4.78 | +0.176 | 0.126 / 0.141 |
| all | 156872 | -0.07 | 1.06 / 7.64 | 1.02 / 3.93 | +0.066 | 0.125 / 0.124 |

Above the surface (located in both): B4 0.0 %, reference 0.4 % (reference: a2_parse_hyp.py check, no EFT).
D-B055: 0 events above the surface (0 located in 3D, 0 with a hypocentre from elsewhere, loc_source v4). Surface: D-B068 merged solid surface (b4_surface.py, /wd1/mdenolle_data/reloc_v5/topo): first covering source of [gmrt_-129.55_-128.75_47.65_48.35_high.nc (41 m, md5 7bcc9e645a3cb1b276fbdea079460c2b); gmrt_-130.35_-129.75_45.75_46.2_high.nc (42 m, md5 d2653715b20e7bc172bf88f2bdf48b54); gmrt_-129.7_-127.9_48.55_49.4_med.nc (80 m, md5 077056a5725bd1f92a6ee6d8c38bf7cc); gmrt_-129.3_-127.3_43.3_44.6_med.nc (88 m, md5 0dd4eda9525b07f4323d8255a4f33b94); gmrt_-130.4_-127.2_47.4_49.8_med.nc (162 m, md5 67f460ee394d9b575bc624fd0e8edd93); gmrt_-123.6_-121.4_46.6_48.6_med.nc (165 m, md5 1419d851c3112ef96228a4a3df54ec25); gmrt_-130.6_-126.6_42.6_45.3_med.nc (176 m, md5 9145fd1d13d7d2c0301596b44207fbb1); gmrt_-125.6_-123.2_39.6_41.4_med.nc (186 m, md5 8b0aef3b02017a3eb57f422637b7dac3); gmrt_-127.6_-123.2_46.2_49.2_med.nc (329 m, md5 96b54b1805c70ae9f22223384d558baa); gmrt_-126.8_-123.2_42.8_46.4_med.nc (348 m, md5 9b27b0947fd9bd495e7a1abecbbcd4aa); gmrt_-128.8_-123.8_40.0_43.2_med.nc (366 m, md5 98c18c8677c4bb705eaea0a0afb575d0); earth_relief_01m_g (1558 m, md5 fba0fa3df82fd1b39069850540f55a8f)] (GMRT finest first, then earth_relief 01m gridline; covering = inside the source's grid lines with 4 finite bilinear nodes), bilinear on that source's grid, no blending. Publishable.
The ANSS check per tier is in the tier summary of a2_build_catalog.py (copied next to this file).

## v5.1 flags by region (NLLoc-located events)

floor_limited: maximum likelihood within 1.5 final oct-tree cells (ns_z) of the NLLoc floor of the run (/wd1/mdenolle_data/reloc_v5/topo/topo_15s_cons_merged_eft.bin; nearest node as NLLoc reads it, run frame); 7917 within 1 cell; 0 located events above that floor file (0 expected: NLLoc rejects them). far_only: n_fine == 0 in the main-pass row of arrivals_used.csv; deep_pass: hypocentre from the deep pass; five_phase: located with 5 phases.

| region | events | located | floor_limited | far_only | deep_pass | five_phase | above_surface |
|---|---|---|---|---|---|---|---|
| Axial | 26731 | 26726 | 1562 | 0 | 3 | 359 | 0 |
| Blanco | 42385 | 42349 | 1046 | 286 | 13 | 5070 | 0 |
| Cascades | 17280 | 17093 | 1358 | 0 | 178 | 1115 | 0 |
| Endeavour | 179 | 178 | 47 | 0 | 1 | 59 | 0 |
| Explorer | 1523 | 1514 | 50 | 24 | 27 | 192 | 0 |
| Gorda | 11378 | 11350 | 345 | 214 | 15 | 1166 | 0 |
| JdF ridge/plate | 2851 | 2843 | 1173 | 11 | 15 | 517 | 0 |
| Mendocino | 24644 | 24608 | 448 | 50 | 25 | 1187 | 0 |
| Offshore forearc | 15422 | 15398 | 442 | 30 | 21 | 960 | 0 |
| Other onshore | 15093 | 14825 | 1277 | 9 | 41 | 729 | 0 |
| Puget/Georgia | 14854 | 14833 | 934 | 0 | 17 | 624 | 0 |
| all | 172340 | 171717 | 8682 | 624 | 356 | 11978 | 0 |

Not located (loc_source none, tier U; 2 v4.1 additions left out for catalog_version.py): u_reason

| u_reason | n |
|---|---|
| rejected: max likelihood on box face 31 | 305 |
| rejected: max likelihood on box face 10 | 185 |
| rejected: max likelihood on box face 20 | 74 |
| rejected: max likelihood on box face 21 | 32 |
| rejected: max likelihood on box face 11 | 27 |

## Flags (rule 4 of D-B049..D-B051)

| region | (no hypocentre) | biased_deep | ok | unconstrained | all |
|---|---|---|---|---|---|
| Axial | 5 | 0 | 26726 | 0 | 26731 |
| Blanco | 36 | 0 | 21945 | 20404 | 42385 |
| Cascades | 187 | 0 | 17093 | 0 | 17280 |
| Endeavour | 1 | 0 | 0 | 178 | 179 |
| Explorer | 9 | 0 | 0 | 1514 | 1523 |
| Gorda | 28 | 9188 | 2162 | 0 | 11378 |
| JdF ridge/plate | 8 | 0 | 0 | 2843 | 2851 |
| Mendocino | 36 | 0 | 24608 | 0 | 24644 |
| Offshore forearc | 24 | 0 | 15398 | 0 | 15422 |
| Other onshore | 268 | 0 | 14825 | 0 | 15093 |
| Puget/Georgia | 21 | 0 | 14833 | 0 | 14854 |
| all | 623 | 9188 | 137590 | 24939 | 172340 |

event_flag quarry_blast_likely: 1573 events (Texada Island box, weekdays 07:00-19:00 America/Vancouver).
event_flag ANSS non-earthquakes (D-B071, within 3 s and 20 km): anss_explosion 1666, anss_quarry_blast 74, anss_other_event 1, anss_sonic_boom 0.

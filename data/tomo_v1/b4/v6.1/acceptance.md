# Catalog v6.1 acceptance

Catalog /wd1/mdenolle_data/reloc_v61/run/catalog/catalog_b4_v6.1.csv (md5 a7f1a54f79d2f3c17dca2d2b744874d1), model v0.4.2, grid manifest md5 e989c12e08cabf81ad4eaf6474c94518, floor /wd1/mdenolle_data/reloc_v5/topo/topo_15s_cons_merged_eft.bin (md5 aaf46fe4321b016c9f993c404d12ce22), picks /wd1/mdenolle_data/tomo_v1/checks/pick_timing_v6/Cascadia_picks_catalog_v4.1tc.csv. Built 2026-11-27 01:42:22.

## Located events

| catalog | events | located | not located |
|---|---|---|---|
| v6.1 | 172,341 | 171,996 | 345 |
| v5.1 | 172,342 | 171,717 | 625 |

| loc_pass | events |
|---|---|
| ssst_it2 | 119,955 |
| coherence | 51,415 |
| edge | 342 |
| deep | 284 |

| u_reason | events |
|---|---|
| rejected: max likelihood on box face 31 | 133 |
| rejected: max likelihood on box face 10 | 101 |
| rejected: max likelihood on box face 20 | 65 |
| rejected: max likelihood on box face 21 | 26 |
| rejected: max likelihood on box face 11 | 20 |


## Events above the solid surface (D-B055)

Hard requirement: 0. An event above the ground or seafloor is a computation error to fix at its root; no depth is clamped or moved.

Above the surface: 0 of 171,996 located events (PASS). Located events above the NLLoc floor file of their run: 0 (expected 0). Surface: D-B068 merged solid surface (b4_surface.py, /wd1/mdenolle_data/reloc_v5/topo): first covering source of [gmrt_-129.55_-128.75_47.65_48.35_high.nc (41 m, md5 7bcc9e645a3cb1b276fbdea079460c2b); gmrt_-130.35_-129.75_45.75_46.2_high.nc (42 m, md5 d2653715b20e7bc172bf88f2bdf48b54); gmrt_-129.7_-127.9_48.55_49.4_med.nc (80 m, md5 077056a5725bd1f92a6ee6d8c38bf7cc); gmrt_-129.3_-127.3_43.3_44.6_med.nc (88 m, md5 0dd4eda9525b07f4323d8255a4f33b94); gmrt_-130.4_-127.2_47.4_49.8_med.nc (162 m, md5 67f460ee394d9b575bc624fd0e8edd93); gmrt_-123.6_-121.4_46.6_48.6_med.nc (165 m, md5 1419d851c3112ef96228a4a3df54ec25); gmrt_-130.6_-126.6_42.6_45.3_med.nc (176 m, md5 9145fd1d13d7d2c0301596b44207fbb1); gmrt_-125.6_-123.2_39.6_41.4_med.nc (186 m, md5 8b0aef3b02017a3eb57f422637b7dac3); gmrt_-127.6_-123.2_46.2_49.2_med.nc (329 m, md5 96b54b1805c70ae9f22223384d558baa); gmrt_-126.8_-123.2_42.8_46.4_med.nc (348 m, md5 9b27b0947fd9bd495e7a1abecbbcd4aa); gmrt_-128.8_-123.8_40.0_43.2_med.nc (366 m, md5 98c18c8677c4bb705eaea0a0afb575d0); earth_relief_01m_g (1558 m, md5 fba0fa3df82fd1b39069850540f55a8f)] (GMRT finest first, then earth_relief 01m gridline; covering = inside the source's grid lines with 4 finite bilinear nodes), bilinear on that source's grid, no blending.


## Floor-limited events by region

floor_limited: maximum likelihood within 1.5 final oct-tree cells of the NLLoc floor (depth set by the floor, not the data).

| region | located | floor_limited | located_v51 | floor_limited_v51 | share_v6.1_% | share_v51_% |
|---|---|---|---|---|---|---|
| Axial | 26,725 | 1,481 | 26,726 | 1,562 | 5.5 | 5.8 |
| Endeavour | 178 | 43 | 178 | 47 | 24.2 | 26.4 |
| Explorer | 1,516 | 30 | 1,514 | 50 | 2.0 | 3.3 |
| JdF ridge/plate | 2,846 | 1,002 | 2,843 | 1,173 | 35.2 | 41.3 |
| Blanco | 42,376 | 571 | 42,349 | 1,046 | 1.3 | 2.5 |
| Gorda | 11,371 | 255 | 11,350 | 345 | 2.2 | 3.0 |
| Mendocino | 24,625 | 351 | 24,608 | 448 | 1.4 | 1.8 |
| Offshore forearc | 15,408 | 340 | 15,398 | 442 | 2.2 | 2.9 |
| Puget/Georgia | 14,848 | 840 | 14,833 | 934 | 5.7 | 6.3 |
| Cascades | 17,174 | 1,010 | 17,093 | 1,358 | 5.9 | 7.9 |
| Other onshore | 14,929 | 1,153 | 14,825 | 1,277 | 7.7 | 8.6 |
| all | 171,996 | 7,076 | 171,717 | 8,682 | 4.1 | 5.1 |


## Location-quality tiers

Tier rules of a2_build_catalog.py (D tier also for every 5-phase event, D-B066).

| tier | v6.1 | v5.1 |
|---|---|---|
| A | 22,516 | 21,325 |
| B | 33,719 | 33,896 |
| C | 82,349 | 73,464 |
| D | 33,412 | 43,032 |
| U | 345 | 625 |

| region (v6.1) | A | B | C | D | U |
|---|---|---|---|---|---|
| Axial | 4,195 | 8,389 | 8,587 | 5,554 | 6 |
| Endeavour | 0 | 10 | 71 | 97 | 1 |
| Explorer | 0 | 30 | 1,132 | 354 | 7 |
| JdF ridge/plate | 1 | 25 | 1,578 | 1,242 | 5 |
| Blanco | 788 | 3,689 | 32,046 | 5,853 | 9 |
| Gorda | 561 | 2,654 | 6,297 | 1,859 | 8 |
| Mendocino | 5,681 | 4,993 | 8,315 | 5,636 | 19 |
| Offshore forearc | 1,751 | 4,064 | 6,335 | 3,258 | 14 |
| Puget/Georgia | 2,404 | 3,904 | 5,879 | 2,661 | 6 |
| Cascades | 4,393 | 2,818 | 6,101 | 3,862 | 106 |
| Other onshore | 2,742 | 3,143 | 6,008 | 3,036 | 164 |


## Change from catalog v5.1 by region

Events located in both; dz positive = v6.1 deeper.

| region | n | dh_med | dh_p90 | dz_med | absdz_med | absdz_p90 |
|---|---|---|---|---|---|---|
| Axial | 26,723 | 0.14 | 0.53 | 0.09 | 0.13 | 0.59 |
| Endeavour | 177 | 1.53 | 10.46 | -0.00 | 1.03 | 18.44 |
| Explorer | 1,512 | 2.14 | 10.69 | 0.15 | 2.93 | 23.77 |
| JdF ridge/plate | 2,842 | 0.89 | 9.03 | 0.00 | 0.52 | 29.13 |
| Blanco | 42,347 | 0.74 | 3.29 | 0.00 | 0.92 | 11.12 |
| Gorda | 11,347 | 0.82 | 3.97 | -0.07 | 1.29 | 9.43 |
| Mendocino | 24,605 | 0.81 | 2.87 | 0.22 | 0.61 | 3.66 |
| Offshore forearc | 15,395 | 0.85 | 3.08 | 0.00 | 0.82 | 4.42 |
| Puget/Georgia | 14,833 | 0.57 | 2.49 | 0.22 | 0.59 | 3.24 |
| Cascades | 17,088 | 0.33 | 1.57 | 0.15 | 0.37 | 2.21 |
| Other onshore | 14,816 | 0.52 | 3.21 | 0.00 | 0.52 | 3.61 |
| all | 171,685 | 0.57 | 2.76 | 0.09 | 0.55 | 4.99 |


## Depths against published local catalogs (event-paired)

b4_lit_paired.py: catalog minus published depth (km, positive = ours deeper), median [p10, p90] and n; v5.1 in the same table (v5.1 report: /wd1/mdenolle_data/catalog_v5/v5.1/paired_v5.1.md).


Catalog minus published depth, km (positive = our event deeper); median [p10, p90] and n. Events located in 3D only. Mount St Helens: both in km bsl. Blanco: Lange datum read as km bsl. Script: workflow/10_tomography/b4_lit_paired.py.

## Mount St Helens, Hirao et al. 2025

| tiers | group | v6.1 | v6.0 | v5.1 |
|---|---|---|---|---|
| A-D | all | +0.85 [-0.9, +2.3] n=2369 | +0.75 [-0.9, +2.1] n=2369 | +0.54 [-1.5, +2.2] n=2369 |
| A-C | all | +0.88 [-0.8, +2.3] n=2148 | +0.77 [-0.8, +2.1] n=2144 | +0.58 [-1.4, +2.2] n=2039 |
| A-B | all | +1.04 [-0.5, +2.3] n=1691 | +0.91 [-0.5, +2.2] n=1683 | +0.76 [-1.1, +2.2] n=1671 |

## Blanco, Lange et al. 2026 (absolute)

| tiers | group | v6.1 | v6.0 | v5.1 |
|---|---|---|---|---|
| A-D | all | +1.01 [-2.6, +11.2] n=11298 | +1.38 [-2.5, +12.2] n=11298 | +1.43 [-2.8, +11.8] n=11298 |
| A-D | West (WBD/EBD) | +1.92 [-2.0, +11.9] n=4785 | +2.72 [-1.9, +12.7] n=4785 | +2.61 [-2.1, +12.9] n=4785 |
| A-D | Surveyor | -0.38 [-2.6, +3.5] n=2075 | -0.28 [-2.4, +3.9] n=2075 | +0.74 [-2.3, +6.6] n=2075 |
| A-D | Cascadia Depression | -1.21 [-3.5, +5.6] n=1900 | -1.09 [-3.4, +7.7] n=1900 | -1.02 [-3.7, +7.0] n=1900 |
| A-D | Blanco Ridge | +3.94 [-2.1, +13.9] n=2035 | +3.69 [-2.0, +14.3] n=2035 | +2.82 [-3.2, +13.3] n=2035 |
| A-D | Gorda Depression | +4.85 [-5.2, +17.0] n=503 | +7.90 [-4.0, +18.9] n=503 | +5.43 [-5.9, +16.6] n=503 |
| A-C | all | +1.01 [-2.6, +11.2] n=11229 | +1.37 [-2.5, +12.2] n=11247 | +1.40 [-2.8, +11.6] n=10887 |
| A-C | West (WBD/EBD) | +1.92 [-2.0, +11.9] n=4764 | +2.72 [-1.9, +12.7] n=4766 | +2.51 [-2.0, +12.6] n=4578 |
| A-C | Surveyor | -0.38 [-2.6, +3.5] n=2069 | -0.28 [-2.4, +3.9] n=2072 | +0.73 [-2.3, +6.4] n=2024 |
| A-C | Cascadia Depression | -1.22 [-3.5, +5.5] n=1889 | -1.09 [-3.4, +7.6] n=1895 | -1.04 [-3.6, +6.7] n=1841 |
| A-C | Blanco Ridge | +3.95 [-2.1, +13.9] n=2012 | +3.69 [-1.9, +14.4] n=2017 | +2.85 [-3.0, +13.4] n=1963 |
| A-C | Gorda Depression | +4.86 [-5.1, +17.1] n=495 | +7.94 [-4.0, +19.0] n=497 | +5.64 [-5.7, +16.6] n=481 |
| A-B | all | +3.17 [-1.3, +9.9] n=2389 | +3.15 [-1.3, +10.1] n=2436 | +2.63 [-0.3, +10.2] n=2395 |
| A-B | West (WBD/EBD) | +5.31 [+3.2, +11.5] n=674 | +5.30 [+3.2, +11.8] n=690 | +5.69 [+2.5, +12.2] n=657 |
| A-B | Surveyor | -0.24 [-2.1, +3.1] n=800 | -0.24 [-2.0, +3.2] n=817 | +0.92 [-1.1, +4.2] n=858 |
| A-B | Cascadia Depression | +1.63 [+0.1, +11.8] n=117 | +2.11 [+0.3, +15.0] n=130 | +2.38 [+0.0, +12.6] n=116 |
| A-B | Blanco Ridge | +3.84 [+0.2, +10.5] n=775 | +3.61 [+0.3, +10.1] n=778 | +2.68 [-0.1, +10.4] n=740 |
| A-B | Gorda Depression | +9.58 [+5.6, +24.3] n=23 | +14.25 [+7.7, +29.4] n=21 | +10.86 [+2.3, +22.9] n=24 |

## Blanco, Lange et al. 2026 (relative)

| tiers | group | v6.1 | v6.0 | v5.1 |
|---|---|---|---|---|
| A-D | all | +0.98 [-2.3, +11.3] n=10706 | +1.26 [-2.1, +12.3] n=10706 | +1.43 [-2.4, +11.8] n=10706 |
| A-D | West (WBD/EBD) | +2.00 [-1.6, +12.0] n=4530 | +2.93 [-1.5, +12.7] n=4530 | +2.97 [-1.7, +12.9] n=4530 |
| A-D | Surveyor | -0.51 [-2.7, +2.2] n=2007 | -0.44 [-2.5, +2.4] n=2007 | +0.47 [-2.4, +5.9] n=2007 |
| A-D | Cascadia Depression | -1.13 [-2.8, +5.8] n=1793 | -1.01 [-2.7, +7.7] n=1793 | -0.91 [-2.9, +7.5] n=1793 |
| A-D | Blanco Ridge | +3.87 [-0.9, +13.3] n=1897 | +3.64 [-0.8, +14.2] n=1897 | +2.96 [-2.3, +13.0] n=1897 |
| A-D | Gorda Depression | +6.39 [-3.3, +18.0] n=479 | +9.16 [-2.6, +19.4] n=479 | +7.16 [-3.8, +17.4] n=479 |
| A-C | all | +0.98 [-2.2, +11.3] n=10643 | +1.26 [-2.1, +12.3] n=10659 | +1.40 [-2.4, +11.5] n=10314 |
| A-C | West (WBD/EBD) | +2.00 [-1.6, +12.0] n=4509 | +2.93 [-1.5, +12.7] n=4511 | +2.88 [-1.6, +12.7] n=4329 |
| A-C | Surveyor | -0.51 [-2.7, +2.2] n=2003 | -0.44 [-2.5, +2.4] n=2005 | +0.45 [-2.4, +5.5] n=1958 |
| A-C | Cascadia Depression | -1.13 [-2.8, +5.6] n=1783 | -1.01 [-2.7, +7.6] n=1789 | -0.94 [-2.9, +7.1] n=1738 |
| A-C | Blanco Ridge | +3.90 [-0.8, +13.3] n=1877 | +3.65 [-0.7, +14.3] n=1881 | +3.00 [-2.2, +13.0] n=1831 |
| A-C | Gorda Depression | +6.57 [-3.2, +18.0] n=471 | +9.21 [-2.6, +19.4] n=473 | +7.22 [-3.8, +17.4] n=458 |
| A-B | all | +3.19 [-1.4, +9.6] n=2307 | +3.19 [-1.4, +9.9] n=2354 | +2.70 [-0.2, +9.7] n=2320 |
| A-B | West (WBD/EBD) | +6.55 [+3.8, +11.5] n=653 | +6.48 [+3.6, +12.2] n=668 | +6.51 [+3.6, +12.2] n=642 |
| A-B | Surveyor | -0.23 [-2.3, +2.1] n=787 | -0.23 [-2.2, +2.1] n=805 | +0.76 [-1.3, +2.9] n=846 |
| A-B | Cascadia Depression | +3.12 [+2.2, +13.2] n=109 | +3.12 [+2.5, +14.7] n=121 | +3.30 [+1.8, +13.6] n=107 |
| A-B | Blanco Ridge | +3.66 [+1.2, +9.3] n=736 | +3.59 [+1.1, +8.6] n=740 | +2.85 [+0.8, +9.0] n=702 |
| A-B | Gorda Depression | +14.67 [+7.8, +23.0] n=22 | +16.82 [+10.5, +25.3] n=20 | +12.18 [+8.1, +21.7] n=23 |


## Axial: share within 2 km of the seafloor

Tier A-B Axial events, depth below the seafloor = depth + surface_km. v5.1: 81 %; target 85 %; Waldhauser et al. (2020): 85 % within 1.6 km, 90 % within 2.0 km.

| catalog | tiers | n | median_bsf_km | within_1_6_pct | within_2_0_pct |
|---|---|---|---|---|---|
| v6.1 | A-B | 12,584 | 1.79 | 32.8 | 72.7 |
| v6.1 | A-D | 26,725 | 1.86 | 31.5 | 60.0 |
| v5.1 | A-B | 12,975 | 1.67 | 43.2 | 81.0 |
| v5.1 | A-D | 26,726 | 1.78 | 38.5 | 66.3 |

v6.1 tier A-B within 2 km: 72.7 % (v5.1 81.0 %: below; target 85 %: not met).


## Residual rms by region

rms_unweighted: unweighted rms of the residuals of the used arrivals (s); rms: NLLoc's EDT-weighted RMS (D-B069), also for v5.1.

| region | n | rms_unweighted_med | rms_unweighted_p90 | rms_med | rms_v51_med |
|---|---|---|---|---|---|
| Axial | 26,725 | 0.073 | 0.159 | 0.068 | 0.070 |
| Endeavour | 178 | 0.110 | 0.352 | 0.096 | 0.123 |
| Explorer | 1,516 | 0.136 | 0.291 | 0.118 | 0.165 |
| JdF ridge/plate | 2,846 | 0.101 | 0.244 | 0.092 | 0.113 |
| Blanco | 42,376 | 0.126 | 0.222 | 0.113 | 0.147 |
| Gorda | 11,371 | 0.144 | 0.246 | 0.126 | 0.158 |
| Mendocino | 24,625 | 0.110 | 0.203 | 0.098 | 0.125 |
| Offshore forearc | 15,408 | 0.114 | 0.213 | 0.102 | 0.128 |
| Puget/Georgia | 14,848 | 0.110 | 0.187 | 0.099 | 0.124 |
| Cascades | 17,174 | 0.120 | 0.227 | 0.104 | 0.117 |
| Other onshore | 14,929 | 0.116 | 0.211 | 0.102 | 0.125 |
| all | 171,996 | 0.111 | 0.211 | 0.099 | 0.122 |


## Held-out effect of the coherence stage by region

b4_v6_heldout.py: the v6.1 recipe (v0.4.2 grids, NLL-SSST CharDist 9999 -> 16 km, then coherence) on nine clusters with waveform CC (termtest protocol: 3 station folds, held-out CC differential times, pair-demeaned); MAD per phase pooled over the folds on the rows located in both arms; coherence - ssst with a 95 % paired bootstrap interval over event pairs.

| region | located | coherence_relocated | with_partner | skipped_region |
|---|---|---|---|---|
| Axial | 26,725 | 21,338 | 17,461 | 0 |
| Endeavour | 178 | 3 | 0 | 0 |
| Explorer | 1,516 | 39 | 10 | 0 |
| JdF ridge/plate | 2,846 | 23 | 6 | 0 |
| Blanco | 42,376 | 4,236 | 3,649 | 0 |
| Gorda | 11,371 | 1,324 | 582 | 0 |
| Mendocino | 24,625 | 0 | 0 | 10,500 |
| Offshore forearc | 15,408 | 5,252 | 2,700 | 0 |
| Puget/Georgia | 14,848 | 5,515 | 2,333 | 0 |
| Cascades | 17,174 | 8,092 | 3,878 | 0 |
| Other onshore | 14,929 | 5,593 | 1,206 | 0 |

D-B098 (PI 2026-11-25 20:35): no coherence relocation in the Mendocino region (v5.1 region label), where the held-out test on the 2015 sequence found the stage worse (P +8 ms, S +20 ms); those compact events keep their final SSST (16 km) location (skipped_region) and can still be stacking partners elsewhere. Provisional: one Mendocino test cluster.

Partner rule of v6.0 (R3, coordinator 2026-11-25 20:00, notes/v6_plan.md): separation <= 3 km, >= 3 accepted CC rows, median cc_max >= 0.5 over the measured rows, rows straddling a confirmed clock epoch left out; the test sets' rule (median cc_max >= 0.5 over >= 3 measured rows) passes 55 % of all-family pairs 10-20 km apart.

Held-out MAD (ms): ssst = SSST to 16 km alone; coh = coherence with the test rule; coh_r3 = coherence with R3 (the v6.0 rule); diff = coherence - ssst with its 95 % interval (effect: helps / hurts / no clear effect).

| set | region | phase | pairs | mad_ssst_ms | mad_coh_ms | diff_ms | effect | mad_coh_r3_ms | diff_r3_ms | lo_r3_ms | hi_r3_ms | effect_r3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| axial | Axial | P | 4,343 | 49.7 | 35.7 | -13.9 | helps | 39.6 | -10.1 | -11.5 | -8.8 | helps |
| axial | Axial | S | 2,939 | 76.7 | 45.1 | -31.7 | helps | 51.3 | -25.4 | -28.7 | -22.7 | helps |
| blanco | Blanco | P | 2,932 | 87.5 | 87.5 | +0.0 | no clear effect | 87.4 | -0.1 | -0.3 | +0.4 | no clear effect |
| blanco | Blanco | S | 2,387 | 146.8 | 146.7 | -0.1 | no clear effect | 146.7 | -0.1 | -0.3 | +0.5 | no clear effect |
| msh | Cascades (Mount St Helens) | P | 6,621 | 44.2 | 21.7 | -22.5 | helps | 23.0 | -21.2 | -22.3 | -20.1 | helps |
| msh | Cascades (Mount St Helens) | S | 3,412 | 48.9 | 19.9 | -29.0 | helps | 20.1 | -28.8 | -30.6 | -26.8 | helps |
| jdf | JdF ridge/plate | P | 1,317 | 111.2 | 111.2 | +0.0 | no clear effect | 111.2 | +0.0 | +0.0 | +0.0 | no clear effect |
| jdf | JdF ridge/plate | S | 1,180 | 303.5 | 303.5 | +0.0 | no clear effect | 303.5 | +0.0 | +0.0 | +0.0 | no clear effect |
| puget | Puget/Georgia | P | 3,943 | 74.8 | 67.6 | -7.2 | helps | 67.8 | -7.0 | -10.2 | -4.1 | helps |
| puget | Puget/Georgia | S | 3,846 | 112.4 | 105.0 | -7.4 | helps | 105.2 | -7.2 | -11.5 | -4.1 | helps |
| mtj | Mendocino | P | 2,062 | 52.4 | 53.3 | +0.9 | no clear effect | 53.3 | +0.8 | -2.3 | +4.2 | no clear effect |
| mtj | Mendocino | S | 2,003 | 75.6 | 88.6 | +13.1 | hurts | 88.7 | +13.1 | +7.4 | +18.5 | hurts |
| blancow | Blanco | P | 5,612 | 94.5 | 94.6 | +0.1 | no clear effect | 94.6 | +0.1 | -0.3 | +0.4 | no clear effect |
| blancow | Blanco | S | 6,450 | 160.6 | 160.6 | +0.0 | no clear effect | 160.6 | +0.0 | -0.2 | +0.3 | no clear effect |
| georgia | Puget/Georgia | P | 4,087 | 35.6 | 20.6 | -15.0 | helps | 20.5 | -15.1 | -16.4 | -13.7 | helps |
| georgia | Puget/Georgia | S | 4,056 | 53.4 | 31.3 | -22.1 | helps | 31.4 | -22.0 | -23.6 | -20.0 | helps |
| gorda | Gorda | P | 2,970 | 47.9 | 48.1 | +0.2 | no clear effect | 48.1 | +0.3 | -0.2 | +0.9 | no clear effect |
| gorda | Gorda | S | 3,275 | 60.5 | 61.6 | +1.1 | hurts | 61.6 | +1.1 | +0.0 | +2.0 | hurts |

Against /wd1/mdenolle_data/reloc_v6/heldout (the earlier version's held-out scores, same clusters, folds and partner rule R3): MAD of the final arm (coherence R3), ms; change > 5 % = worse by the plan's margin (4 of 18 cluster-phases).

| set | region | phase | mad_v6.1_ms | mad_ref_ms | change_pct | worse_than_5pct |
|---|---|---|---|---|---|---|
| axial | Axial | P | 39.6 | 40.5 | -2.3 | 0 |
| axial | Axial | S | 51.3 | 51.8 | -0.9 | 0 |
| blanco | Blanco | P | 87.4 | 88.1 | -0.8 | 0 |
| blanco | Blanco | S | 146.7 | 143.0 | +2.6 | 0 |
| msh | Cascades (Mount St Helens) | P | 23.0 | 22.8 | +0.9 | 0 |
| msh | Cascades (Mount St Helens) | S | 20.1 | 20.3 | -0.8 | 0 |
| jdf | JdF ridge/plate | P | 111.2 | 163.7 | -32.1 | 0 |
| jdf | JdF ridge/plate | S | 303.5 | 293.0 | +3.6 | 0 |
| puget | Puget/Georgia | P | 67.8 | 57.5 | +17.9 | 1 |
| puget | Puget/Georgia | S | 105.2 | 91.3 | +15.2 | 1 |
| mtj | Mendocino | P | 53.3 | 51.8 | +2.8 | 0 |
| mtj | Mendocino | S | 88.7 | 82.7 | +7.2 | 1 |
| blancow | Blanco | P | 94.6 | 99.4 | -4.9 | 0 |
| blancow | Blanco | S | 160.6 | 155.1 | +3.6 | 0 |
| georgia | Puget/Georgia | P | 20.5 | 19.7 | +3.8 | 0 |
| georgia | Puget/Georgia | S | 31.4 | 30.3 | +3.7 | 0 |
| gorda | Gorda | P | 48.1 | 46.8 | +3.0 | 0 |
| gorda | Gorda | S | 61.6 | 57.2 | +7.7 | 1 |


## Passes

| run | events | status | rejected_lateral | rejected_bottom | located_deep | wall_min | compact | compact_with_compact_partner | pairs_used | n_failed | selected |
|---|---|---|---|---|---|---|---|---|---|---|---|
| it0 | 172,342.00 | {"LOCATED": 171217, "REJECTED": 1125} | 550.00 | 575.00 | 71.00 | 30.70 |  |  |  |  |  |
| it1 | 172,342.00 | {"LOCATED": 171373, "REJECTED": 969} | 475.00 | 494.00 | 46.00 | 27.60 |  |  |  |  |  |
| it2 | 172,342.00 | {"LOCATED": 171425, "REJECTED": 917} | 554.00 | 363.00 | 61.00 | 26.50 |  |  |  |  |  |
| coh |  |  |  |  |  | 86.90 | 51,423.00 | 31,825.00 | 594,612.00 | 0.00 |  |
| deep | 415.00 | {"LOCATED": 284, "REJECTED": 131} |  |  |  |  |  |  |  |  | 415.00 |
| edge | 554.00 | {"LOCATED": 342, "REJECTED": 212} |  |  |  |  |  |  |  |  | 554.00 |

Loc2ssst: corrected grids per pass; kept_previous = station-phases without accepted arrivals (previous grid kept); negative_nodes = corrected time < 0 near a station (NLLoc does not use a negative predicted time); events_dropped_arrival_limit = events left out of a tile's input by the 20,000 arrivals per station-phase limit (nearest kept).

| pass | char_dist_km | tiles | instances | events_accepted_sum | corrected_grids | kept_previous | negative_nodes | tiles_with_negative | t_min | tiles_partial_input | events_dropped_arrival_limit | wall_min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| g0 | 9,999.00 | 1,227 | 1,397 | 1,861,568 | 115,644 | 12,812 | 1,529 | 273 | -1.66 | 351 | 3,709,913 | 9.50 |
| g1 | 16.00 | 1,227 | 1,397 | 1,481,396 | 114,481 | 13,975 | 5,186 | 386 | -1.65 | 244 | 2,298,050 | 9.30 |

Flags: {"floor_limited": 7076, "far_only": 347, "deep_pass": 284, "edge_pass": 342, "five_phase": 12036, "coherence_relocated": 51415}. Review indicators carried on 1,340 events (reviewer columns empty, D-B097). Magnitudes: 130,558 v6.1_converged, 537 v4.1_converged (magnitude_source; D-B091, D-B102, D-B105).


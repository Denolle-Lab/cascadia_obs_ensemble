# Catalog v6.0 acceptance

Catalog /wd1/mdenolle_data/reloc_v6/catalog/catalog_b4_v6.0.csv (md5 ae77ad3c66aa6fbaa48528660a1c09ec), model v0.4.1, grid manifest md5 dbe2f8a1c3a559f8166a4b599fd75898, floor /wd1/mdenolle_data/reloc_v5/topo/topo_15s_cons_merged_eft.bin (md5 aaf46fe4321b016c9f993c404d12ce22), picks /wd1/mdenolle_data/tomo_v1/checks/pick_timing_v6/Cascadia_picks_catalog_v4.1tc.csv. Built 2026-11-26 08:05:33.

## Located events

| catalog | events | located | not located |
|---|---|---|---|
| v6.0 | 172,341 | 172,182 | 159 |
| v5.1 | 172,342 | 171,717 | 625 |

| loc_pass | events |
|---|---|
| ssst_it4 | 118,002 |
| coherence | 53,746 |
| edge | 283 |
| deep | 151 |

| u_reason | events |
|---|---|
| rejected: max likelihood on box face 10 | 63 |
| rejected: max likelihood on box face 31 | 39 |
| rejected: max likelihood on box face 20 | 28 |
| rejected: max likelihood on box face 21 | 15 |
| rejected: max likelihood on box face 11 | 14 |


## Events above the solid surface (D-B055)

Hard requirement: 0. An event above the ground or seafloor is a computation error to fix at its root; no depth is clamped or moved.

Above the surface: 0 of 172,182 located events (PASS). Located events above the NLLoc floor file of their run: 0 (expected 0). Surface: D-B068 merged solid surface (b4_surface.py, /wd1/mdenolle_data/reloc_v5/topo): first covering source of [gmrt_-129.55_-128.75_47.65_48.35_high.nc (41 m, md5 7bcc9e645a3cb1b276fbdea079460c2b); gmrt_-130.35_-129.75_45.75_46.2_high.nc (42 m, md5 d2653715b20e7bc172bf88f2bdf48b54); gmrt_-129.7_-127.9_48.55_49.4_med.nc (80 m, md5 077056a5725bd1f92a6ee6d8c38bf7cc); gmrt_-129.3_-127.3_43.3_44.6_med.nc (88 m, md5 0dd4eda9525b07f4323d8255a4f33b94); gmrt_-130.4_-127.2_47.4_49.8_med.nc (162 m, md5 67f460ee394d9b575bc624fd0e8edd93); gmrt_-123.6_-121.4_46.6_48.6_med.nc (165 m, md5 1419d851c3112ef96228a4a3df54ec25); gmrt_-130.6_-126.6_42.6_45.3_med.nc (176 m, md5 9145fd1d13d7d2c0301596b44207fbb1); gmrt_-125.6_-123.2_39.6_41.4_med.nc (186 m, md5 8b0aef3b02017a3eb57f422637b7dac3); gmrt_-127.6_-123.2_46.2_49.2_med.nc (329 m, md5 96b54b1805c70ae9f22223384d558baa); gmrt_-126.8_-123.2_42.8_46.4_med.nc (348 m, md5 9b27b0947fd9bd495e7a1abecbbcd4aa); gmrt_-128.8_-123.8_40.0_43.2_med.nc (366 m, md5 98c18c8677c4bb705eaea0a0afb575d0); earth_relief_01m_g (1558 m, md5 fba0fa3df82fd1b39069850540f55a8f)] (GMRT finest first, then earth_relief 01m gridline; covering = inside the source's grid lines with 4 finite bilinear nodes), bilinear on that source's grid, no blending.


## Floor-limited events by region

floor_limited: maximum likelihood within 1.5 final oct-tree cells of the NLLoc floor (depth set by the floor, not the data).

| region | located | floor_limited | located_v51 | floor_limited_v51 | share_v6_% | share_v51_% |
|---|---|---|---|---|---|---|
| Axial | 26,729 | 1,321 | 26,726 | 1,562 | 4.9 | 5.8 |
| Endeavour | 179 | 30 | 178 | 47 | 16.8 | 26.4 |
| Explorer | 1,517 | 32 | 1,514 | 50 | 2.1 | 3.3 |
| JdF ridge/plate | 2,846 | 627 | 2,843 | 1,173 | 22.0 | 41.3 |
| Blanco | 42,378 | 495 | 42,349 | 1,046 | 1.2 | 2.5 |
| Gorda | 11,374 | 183 | 11,350 | 345 | 1.6 | 3.0 |
| Mendocino | 24,637 | 269 | 24,608 | 448 | 1.1 | 1.8 |
| Offshore forearc | 15,412 | 323 | 15,398 | 442 | 2.1 | 2.9 |
| Puget/Georgia | 14,850 | 653 | 14,833 | 934 | 4.4 | 6.3 |
| Cascades | 17,243 | 917 | 17,093 | 1,358 | 5.3 | 7.9 |
| Other onshore | 15,017 | 888 | 14,825 | 1,277 | 5.9 | 8.6 |
| all | 172,182 | 5,738 | 171,717 | 8,682 | 3.3 | 5.1 |


## Location-quality tiers

Tier rules of a2_build_catalog.py (D tier also for every 5-phase event, D-B066).

| tier | v6.0 | v5.1 |
|---|---|---|
| A | 22,907 | 21,325 |
| B | 33,191 | 33,896 |
| C | 83,149 | 73,464 |
| D | 32,935 | 43,032 |
| U | 159 | 625 |

| region (v6.0) | A | B | C | D | U |
|---|---|---|---|---|---|
| Axial | 4,114 | 7,825 | 9,057 | 5,733 | 2 |
| Endeavour | 2 | 10 | 77 | 90 | 0 |
| Explorer | 2 | 34 | 1,138 | 343 | 6 |
| JdF ridge/plate | 1 | 39 | 1,624 | 1,182 | 5 |
| Blanco | 826 | 3,703 | 32,136 | 5,713 | 7 |
| Gorda | 654 | 2,621 | 6,318 | 1,781 | 5 |
| Mendocino | 5,800 | 5,026 | 8,326 | 5,485 | 7 |
| Offshore forearc | 1,810 | 4,048 | 6,353 | 3,201 | 10 |
| Puget/Georgia | 2,490 | 3,874 | 5,872 | 2,614 | 4 |
| Cascades | 4,401 | 2,859 | 6,192 | 3,791 | 37 |
| Other onshore | 2,807 | 3,152 | 6,056 | 3,002 | 76 |


## Change from catalog v5.1 by region

Events located in both; dz positive = v6.0 deeper.

| region | n | dh_med | dh_p90 | dz_med | absdz_med | absdz_p90 |
|---|---|---|---|---|---|---|
| Axial | 26,724 | 0.20 | 0.76 | 0.15 | 0.18 | 0.89 |
| Endeavour | 178 | 2.48 | 13.77 | -0.00 | 2.02 | 29.08 |
| Explorer | 1,513 | 2.67 | 12.04 | 0.07 | 3.82 | 25.67 |
| JdF ridge/plate | 2,842 | 1.45 | 13.13 | 0.22 | 1.03 | 32.07 |
| Blanco | 42,348 | 0.94 | 3.96 | 0.11 | 1.33 | 14.58 |
| Gorda | 11,347 | 1.11 | 5.15 | 0.00 | 1.91 | 14.24 |
| Mendocino | 24,605 | 1.09 | 4.14 | 0.29 | 0.88 | 5.58 |
| Offshore forearc | 15,397 | 1.15 | 4.37 | 0.00 | 1.10 | 6.76 |
| Puget/Georgia | 14,833 | 0.77 | 4.11 | 0.29 | 0.88 | 5.30 |
| Cascades | 17,089 | 0.44 | 2.50 | 0.15 | 0.48 | 3.83 |
| Other onshore | 14,819 | 0.71 | 4.59 | 0.07 | 0.74 | 5.75 |
| all | 171,695 | 0.75 | 3.85 | 0.15 | 0.76 | 8.26 |


## Depths against published local catalogs (event-paired)

b4_lit_paired.py: catalog minus published depth (km, positive = ours deeper), median [p10, p90] and n; v5.1 in the same table (v5.1 report: /wd1/mdenolle_data/catalog_v5/v5.1/paired_v5.1.md).


Catalog minus published depth, km (positive = our event deeper); median [p10, p90] and n. Events located in 3D only. Mount St Helens: both in km bsl. Blanco: Lange datum read as km bsl. Script: workflow/10_tomography/b4_lit_paired.py.

## Mount St Helens, Hirao et al. 2025

| tiers | group | v6.0 | v5.1 |
|---|---|---|---|
| A-D | all | +0.75 [-0.9, +2.1] n=2369 | +0.54 [-1.5, +2.2] n=2369 |
| A-C | all | +0.77 [-0.8, +2.1] n=2144 | +0.58 [-1.4, +2.2] n=2039 |
| A-B | all | +0.91 [-0.5, +2.2] n=1683 | +0.76 [-1.1, +2.2] n=1671 |

## Blanco, Lange et al. 2026 (absolute)

| tiers | group | v6.0 | v5.1 |
|---|---|---|---|
| A-D | all | +1.38 [-2.5, +12.2] n=11298 | +1.43 [-2.8, +11.8] n=11298 |
| A-D | West (WBD/EBD) | +2.72 [-1.9, +12.7] n=4785 | +2.61 [-2.1, +12.9] n=4785 |
| A-D | Surveyor | -0.28 [-2.4, +3.9] n=2075 | +0.74 [-2.3, +6.6] n=2075 |
| A-D | Cascadia Depression | -1.09 [-3.4, +7.7] n=1900 | -1.02 [-3.7, +7.0] n=1900 |
| A-D | Blanco Ridge | +3.69 [-2.0, +14.3] n=2035 | +2.82 [-3.2, +13.3] n=2035 |
| A-D | Gorda Depression | +7.90 [-4.0, +18.9] n=503 | +5.43 [-5.9, +16.6] n=503 |
| A-C | all | +1.37 [-2.5, +12.2] n=11247 | +1.40 [-2.8, +11.6] n=10887 |
| A-C | West (WBD/EBD) | +2.72 [-1.9, +12.7] n=4766 | +2.51 [-2.0, +12.6] n=4578 |
| A-C | Surveyor | -0.28 [-2.4, +3.9] n=2072 | +0.73 [-2.3, +6.4] n=2024 |
| A-C | Cascadia Depression | -1.09 [-3.4, +7.6] n=1895 | -1.04 [-3.6, +6.7] n=1841 |
| A-C | Blanco Ridge | +3.69 [-1.9, +14.4] n=2017 | +2.85 [-3.0, +13.4] n=1963 |
| A-C | Gorda Depression | +7.94 [-4.0, +19.0] n=497 | +5.64 [-5.7, +16.6] n=481 |
| A-B | all | +3.15 [-1.3, +10.1] n=2436 | +2.63 [-0.3, +10.2] n=2395 |
| A-B | West (WBD/EBD) | +5.30 [+3.2, +11.8] n=690 | +5.69 [+2.5, +12.2] n=657 |
| A-B | Surveyor | -0.24 [-2.0, +3.2] n=817 | +0.92 [-1.1, +4.2] n=858 |
| A-B | Cascadia Depression | +2.11 [+0.3, +15.0] n=130 | +2.38 [+0.0, +12.6] n=116 |
| A-B | Blanco Ridge | +3.61 [+0.3, +10.1] n=778 | +2.68 [-0.1, +10.4] n=740 |
| A-B | Gorda Depression | +14.25 [+7.7, +29.4] n=21 | +10.86 [+2.3, +22.9] n=24 |

## Blanco, Lange et al. 2026 (relative)

| tiers | group | v6.0 | v5.1 |
|---|---|---|---|
| A-D | all | +1.26 [-2.1, +12.3] n=10706 | +1.43 [-2.4, +11.8] n=10706 |
| A-D | West (WBD/EBD) | +2.93 [-1.5, +12.7] n=4530 | +2.97 [-1.7, +12.9] n=4530 |
| A-D | Surveyor | -0.44 [-2.5, +2.4] n=2007 | +0.47 [-2.4, +5.9] n=2007 |
| A-D | Cascadia Depression | -1.01 [-2.7, +7.7] n=1793 | -0.91 [-2.9, +7.5] n=1793 |
| A-D | Blanco Ridge | +3.64 [-0.8, +14.2] n=1897 | +2.96 [-2.3, +13.0] n=1897 |
| A-D | Gorda Depression | +9.16 [-2.6, +19.4] n=479 | +7.16 [-3.8, +17.4] n=479 |
| A-C | all | +1.26 [-2.1, +12.3] n=10659 | +1.40 [-2.4, +11.5] n=10314 |
| A-C | West (WBD/EBD) | +2.93 [-1.5, +12.7] n=4511 | +2.88 [-1.6, +12.7] n=4329 |
| A-C | Surveyor | -0.44 [-2.5, +2.4] n=2005 | +0.45 [-2.4, +5.5] n=1958 |
| A-C | Cascadia Depression | -1.01 [-2.7, +7.6] n=1789 | -0.94 [-2.9, +7.1] n=1738 |
| A-C | Blanco Ridge | +3.65 [-0.7, +14.3] n=1881 | +3.00 [-2.2, +13.0] n=1831 |
| A-C | Gorda Depression | +9.21 [-2.6, +19.4] n=473 | +7.22 [-3.8, +17.4] n=458 |
| A-B | all | +3.19 [-1.4, +9.9] n=2354 | +2.70 [-0.2, +9.7] n=2320 |
| A-B | West (WBD/EBD) | +6.48 [+3.6, +12.2] n=668 | +6.51 [+3.6, +12.2] n=642 |
| A-B | Surveyor | -0.23 [-2.2, +2.1] n=805 | +0.76 [-1.3, +2.9] n=846 |
| A-B | Cascadia Depression | +3.12 [+2.5, +14.7] n=121 | +3.30 [+1.8, +13.6] n=107 |
| A-B | Blanco Ridge | +3.59 [+1.1, +8.6] n=740 | +2.85 [+0.8, +9.0] n=702 |
| A-B | Gorda Depression | +16.82 [+10.5, +25.3] n=20 | +12.18 [+8.1, +21.7] n=23 |


## Axial: share within 2 km of the seafloor

Tier A-B Axial events, depth below the seafloor = depth + surface_km. v5.1: 81 %; target 85 %; Waldhauser et al. (2020): 85 % within 1.6 km, 90 % within 2.0 km.

| catalog | tiers | n | median_bsf_km | within_1_6_pct | within_2_0_pct |
|---|---|---|---|---|---|
| v6.0 | A-B | 11,939 | 1.86 | 28.2 | 66.1 |
| v6.0 | A-D | 26,729 | 1.93 | 27.9 | 55.4 |
| v5.1 | A-B | 12,975 | 1.67 | 43.2 | 81.0 |
| v5.1 | A-D | 26,726 | 1.78 | 38.5 | 66.3 |

v6.0 tier A-B within 2 km: 66.1 % (target 85 %: not met).


## Residual rms by region

rms_unweighted: unweighted rms of the residuals of the used arrivals (s); rms: NLLoc's EDT-weighted RMS (D-B069), also for v5.1.

| region | n | rms_unweighted_med | rms_unweighted_p90 | rms_med | rms_v51_med |
|---|---|---|---|---|---|
| Axial | 26,729 | 0.071 | 0.158 | 0.066 | 0.070 |
| Endeavour | 179 | 0.051 | 0.175 | 0.049 | 0.123 |
| Explorer | 1,517 | 0.062 | 0.170 | 0.056 | 0.165 |
| JdF ridge/plate | 2,846 | 0.072 | 0.180 | 0.067 | 0.113 |
| Blanco | 42,378 | 0.098 | 0.185 | 0.090 | 0.147 |
| Gorda | 11,374 | 0.089 | 0.179 | 0.083 | 0.158 |
| Mendocino | 24,637 | 0.082 | 0.154 | 0.076 | 0.125 |
| Offshore forearc | 15,412 | 0.080 | 0.167 | 0.074 | 0.128 |
| Puget/Georgia | 14,850 | 0.071 | 0.141 | 0.066 | 0.124 |
| Cascades | 17,243 | 0.095 | 0.209 | 0.084 | 0.117 |
| Other onshore | 15,017 | 0.072 | 0.151 | 0.066 | 0.125 |
| all | 172,182 | 0.082 | 0.172 | 0.076 | 0.122 |


## Held-out effect of the coherence stage by region

b4_v6_heldout.py: the v6.0 recipe (v0.4.1 grids, SSST to 4 km, then coherence) on nine clusters with waveform CC (termtest protocol: 3 station folds, held-out CC differential times, pair-demeaned); MAD per phase pooled over the folds on the rows located in both arms; coherence - ssst with a 95 % paired bootstrap interval over event pairs.

| region | located | coherence_relocated | with_partner | skipped_region |
|---|---|---|---|---|
| Axial | 26,729 | 20,637 | 17,010 | 0 |
| Endeavour | 179 | 8 | 0 | 0 |
| Explorer | 1,517 | 138 | 23 | 0 |
| JdF ridge/plate | 2,846 | 53 | 18 | 0 |
| Blanco | 42,378 | 4,450 | 3,551 | 0 |
| Gorda | 11,374 | 2,202 | 917 | 0 |
| Mendocino | 24,637 | 0 | 0 | 11,148 |
| Offshore forearc | 15,412 | 5,660 | 2,790 | 0 |
| Puget/Georgia | 14,850 | 6,048 | 2,444 | 0 |
| Cascades | 17,243 | 8,285 | 3,909 | 0 |
| Other onshore | 15,017 | 6,265 | 1,230 | 0 |

D-B098 (PI 2026-11-25 20:35): no coherence relocation in the Mendocino region (v5.1 region label), where the held-out test on the 2015 sequence found the stage worse (P +8 ms, S +20 ms); those compact events keep their SSST 4 km location (skipped_region) and can still be stacking partners elsewhere. Provisional: one Mendocino test cluster.

Partner rule of v6.0 (R3, coordinator 2026-11-25 20:00, notes/v6_plan.md): separation <= 3 km, >= 3 accepted CC rows, median cc_max >= 0.5 over the measured rows, rows straddling a confirmed clock epoch left out; the test sets' rule (median cc_max >= 0.5 over >= 3 measured rows) passes 55 % of all-family pairs 10-20 km apart.

Rule check before the coherence stage (puget, mtj; decision rule: R3 worse if MAD(R3)-MAD(test) has a 95% bootstrap interval above 0 and >= 5% of MAD(test) in puget or mtj, P or S): use r3.

| set | phase | pairs | mad_ssst_ms | mad_test_ms | mad_r3_ms | r3_minus_test_ms | lo_ms | hi_ms | r3_worse |
|---|---|---|---|---|---|---|---|---|---|
| puget | P | 3,943 | 64.7 | 57.2 | 57.5 | +0.3 | -0.8 | +1.6 | 0 |
| puget | S | 3,846 | 100.4 | 90.7 | 91.3 | +0.6 | -1.4 | +2.3 | 0 |
| mtj | P | 2,062 | 43.5 | 51.9 | 51.8 | -0.0 | -0.3 | +0.4 | 0 |
| mtj | S | 2,003 | 62.4 | 83.1 | 82.7 | -0.4 | -1.1 | +0.1 | 0 |

Held-out MAD (ms): ssst = SSST to 4 km alone; coh = coherence with the test rule; coh_r3 = coherence with R3 (the v6.0 rule); diff = coherence - ssst with its 95 % interval (effect: helps / hurts / no clear effect).

| set | region | phase | pairs | mad_ssst_ms | mad_coh_ms | diff_ms | effect | mad_coh_r3_ms | diff_r3_ms | lo_r3_ms | hi_r3_ms | effect_r3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| axial | Axial | P | 4,343 | 50.3 | 36.4 | -13.9 | helps | 40.5 | -9.8 | -11.1 | -8.4 | helps |
| axial | Axial | S | 2,939 | 77.1 | 45.2 | -31.9 | helps | 51.8 | -25.3 | -28.5 | -22.5 | helps |
| blanco | Blanco | P | 2,934 | 87.8 | 88.1 | +0.3 | no clear effect | 88.1 | +0.3 | -0.2 | +0.8 | no clear effect |
| blanco | Blanco | S | 2,389 | 143.0 | 143.0 | +0.0 | no clear effect | 143.0 | +0.0 | -0.6 | +0.7 | no clear effect |
| msh | Cascades (Mount St Helens) | P | 6,621 | 44.0 | 21.4 | -22.6 | helps | 22.8 | -21.2 | -22.4 | -20.1 | helps |
| msh | Cascades (Mount St Helens) | S | 3,412 | 48.8 | 19.9 | -28.9 | helps | 20.3 | -28.5 | -30.4 | -26.6 | helps |
| jdf | JdF ridge/plate | P | 1,297 | 163.7 | 163.7 | +0.0 | no clear effect | 163.7 | +0.0 | +0.0 | +0.0 | no clear effect |
| jdf | JdF ridge/plate | S | 1,216 | 293.0 | 293.0 | +0.0 | no clear effect | 293.0 | +0.0 | +0.0 | +0.0 | no clear effect |
| puget | Puget/Georgia | P | 3,943 | 64.7 | 57.2 | -7.5 | helps | 57.5 | -7.2 | -9.7 | -4.8 | helps |
| puget | Puget/Georgia | S | 3,846 | 100.4 | 90.7 | -9.7 | helps | 91.3 | -9.1 | -13.3 | -6.3 | helps |
| mtj | Mendocino | P | 2,062 | 43.5 | 51.9 | +8.3 | hurts | 51.8 | +8.3 | +5.3 | +11.7 | hurts |
| mtj | Mendocino | S | 2,003 | 62.4 | 83.1 | +20.6 | hurts | 82.7 | +20.3 | +15.8 | +25.8 | hurts |
| blancow | Blanco | P | 5,612 | 99.3 | 99.4 | +0.1 | no clear effect | 99.4 | +0.1 | -0.1 | +0.4 | no clear effect |
| blancow | Blanco | S | 6,450 | 155.1 | 155.1 | -0.1 | no clear effect | 155.1 | -0.1 | -0.3 | +0.3 | no clear effect |
| georgia | Puget/Georgia | P | 4,087 | 34.5 | 19.8 | -14.7 | helps | 19.7 | -14.7 | -16.0 | -13.4 | helps |
| georgia | Puget/Georgia | S | 4,056 | 50.6 | 30.2 | -20.4 | helps | 30.3 | -20.3 | -22.1 | -18.4 | helps |
| gorda | Gorda | P | 2,970 | 46.5 | 46.8 | +0.2 | no clear effect | 46.8 | +0.2 | -0.3 | +0.8 | no clear effect |
| gorda | Gorda | S | 3,275 | 57.0 | 57.3 | +0.3 | no clear effect | 57.2 | +0.2 | -0.9 | +1.2 | no clear effect |


## Passes

| run | events | status | rejected_lateral | rejected_bottom | located_deep | wall_min | compact | compact_with_compact_partner | pairs_used | n_failed | selected |
|---|---|---|---|---|---|---|---|---|---|---|---|
| it0 | 172,342.00 | {"LOCATED": 171216, "REJECTED": 1126} | 550.00 | 576.00 | 69.00 | 63.70 |  |  |  |  |  |
| it1 | 172,342.00 | {"LOCATED": 171371, "REJECTED": 971} | 478.00 | 493.00 | 49.00 | 27.70 |  |  |  |  |  |
| it2 | 172,342.00 | {"LOCATED": 171446, "REJECTED": 896} | 552.00 | 344.00 | 58.00 | 25.50 |  |  |  |  |  |
| it3 | 172,342.00 | {"LOCATED": 171634, "REJECTED": 708} | 482.00 | 226.00 | 44.00 | 24.60 |  |  |  |  |  |
| it4 | 172,342.00 | {"LOCATED": 171789, "REJECTED": 553} | 404.00 | 149.00 | 44.00 | 26.30 |  |  |  |  |  |
| coh |  |  |  |  |  | 90.70 | 53,754.00 | 31,892.00 | 594,607.00 | 0.00 |  |
| deep | 187.00 | {"LOCATED": 151, "REJECTED": 36} |  |  |  |  |  |  |  |  | 187.00 |
| edge | 404.00 | {"LOCATED": 283, "REJECTED": 121} |  |  |  |  |  |  |  |  | 404.00 |

Loc2ssst: corrected grids per pass; kept_previous = station-phases without accepted arrivals (previous grid kept); negative_nodes = corrected time < 0 near a station (NLLoc does not use a negative predicted time); events_dropped_arrival_limit = events left out of a tile's input by the 20,000 arrivals per station-phase limit (nearest kept).

| pass | char_dist_km | tiles | instances | events_accepted_sum | corrected_grids | kept_previous | negative_nodes | tiles_with_negative | t_min | tiles_partial_input | events_dropped_arrival_limit | wall_min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| g0 | 9,999.00 | 1,227 | 1,397 | 1,861,510 | 115,640 | 12,816 | 1,543 | 276 | -1.66 | 351 | 3,709,881 | 51.70 |
| g1 | 16.00 | 1,227 | 1,397 | 1,481,250 | 114,448 | 14,008 | 5,173 | 387 | -1.65 | 244 | 2,297,937 | 25.60 |
| g2 | 8.00 | 1,227 | 1,397 | 714,681 | 109,272 | 19,184 | 7,165 | 427 | -1.65 | 88 | 643,480 | 9.10 |
| g3 | 4.00 | 1,227 | 1,397 | 716,714 | 109,326 | 19,130 | 9,358 | 451 | -1.65 | 88 | 644,790 | 12.80 |

Flags: {"floor_limited": 5738, "far_only": 353, "deep_pass": 151, "edge_pass": 283, "five_phase": 12082, "coherence_relocated": 53746}. Review indicators carried on 1,340 events (reviewer columns empty, D-B097). Magnitudes: 130349 from the v5.1 converged table, 763 from the v4.1 converged table (D-B091).


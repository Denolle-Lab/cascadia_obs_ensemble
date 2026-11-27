# Catalog comparison

Reference: **v4.1**. Built by `workflow/11_compare/compare_catalogs.py` from `catalogs.yaml`. Depths in km below sea level. ANSS events matched to v4.1: 16,584.

| catalog | velocity model | method | status |
|---|---|---|---|
| v4.1 | PNSN 1D regional models (I. McBrearty); ComCat for 3 added events | GENIE + GraphDD; targeted ELEP repick of four 2016 offshore events | final |
| v3 | PNSN 1D regional models | GENIE association, GraphDD + cross-correlation relocation (ver3 picks) | superseded |
| v4.0 | PNSN 1D regional models (I. McBrearty) | GENIE association, GraphDD relocation (ver6 picks) | superseded |
| NLL3D-i0 | CRESCENT Gen0 Vs + Brocher Vp, Vp/Vs >= 1.76, literature 1D at the ridges | NonLinLoc EDT, 3D grids, no station terms | superseded |
| NLL3D-i1 | CRESCENT Gen0 Vs + Brocher Vp, Vp/Vs >= 1.76, literature 1D at the ridges | NonLinLoc EDT, 3D grids, static station terms from iteration 0; tiers A-D, U = v4 kept | preliminary |
| v5.0 | merged tomoDD model v0.3.1 (tomo_v1/models/cascadia_obs_tomo) | NonLinLoc EDT, 3D grids, static station terms from a first pass (run b4_iter1_v0.3.1); tier U keeps the v4 hypocentre (D-B058, D-B059) | superseded |
| v5.1 | merged tomoDD model v0.3.1; offshore rock within 2.5 km of the seafloor with Vp/Vs >= 2.1 (D-B073) | NonLinLoc EDT, 3D grids, static station terms from a first pass, round-2 fixes (every arrival, five-pick events, deep pass, D-B068 solid surface), re-timed picks (run b4_v51_iter1); tier U without a hypocentre | preliminary |
| v6.0 | merged tomoDD model v0.4.1 (re-inverted with the re-timed picks and the tomoDD surface floor, D-B083) | NonLinLoc EDT, 3D grids, source-specific station terms (NLL-SSST, 4 passes, D-B095/D-B096) and NLL-coherence relocation (not in Mendocino, D-B098), re-timed and clock-corrected picks v4.1tc (D-B090), run v6.0; tier U without a hypocentre (D-B100) | preliminary |

## Whole catalog

| catalog | events | located | depth p10/p50/p90 | above surface | shift from ref. h p50/p90, dz p50 | ANSS dz p50 (MAD), all / near | ANSS dh p50 | fault-plane RMS (km) |
|---|---:|---:|---|---:|---|---|---:|---:|
| v4.1 | 172,342 | 172,342 | -1.4 / 10.1 / 25.2 | 23.5% | 0.0 / 0.0, +0.0 | -0.9 (3.8) / -1.0 (2.4) | 2.2 | 1.14 |
| v3 | 63,887 | 63,887 | 3.2 / 15.6 / 28.6 | 6.2% | 3.6 / 14.5, -1.9 | +0.1 (5.3) / -0.4 (4.1) | 3.5 | 3.23 |
| v4.0 | 172,340 | 172,340 | -1.4 / 10.1 / 25.2 | 23.5% | 0.0 / 0.0, +0.0 | -0.9 (3.8) / -1.0 (2.4) | 2.2 | 1.14 |
| NLL3D-i0 | 157,249 | 157,249 | 2.6 / 11.3 / 30.0 | 0.0% | 2.9 / 7.5, +2.5 | +1.1 (4.7) / +0.2 (2.8) | 2.8 | 2.56 |
| NLL3D-i1 | 172,340 | 157,267 | 2.6 / 11.8 / 30.1 | 0.0% | 2.4 / 6.3, +2.7 | +1.2 (4.2) / +0.4 (2.4) | 2.5 | 2.24 |
| v5.0 | 172,342 | 157,269 | 2.6 / 12.0 / 28.8 | 0.5% | 1.8 / 5.4, +2.4 | +1.2 (3.7) / +0.5 (2.2) | 1.9 | 1.79 |
| v5.1 | 172,342 | 171,717 | 2.7 / 12.0 / 30.4 | 0.4% | 2.2 / 6.5, +2.6 | +1.2 (3.8) / +0.6 (2.3) | 2.1 | 1.83 |
| v6.0 | 172,342 | 172,182 | 2.9 / 12.8 / 31.0 | 0.3% | 2.1 / 6.3, +2.9 | +1.5 (3.7) / +0.8 (2.2) | 1.9 | 1.92 |

Fault-plane RMS: median over the 119 fault surfaces with >= 20 located members in every catalog. The surfaces were fitted to v4 locations and their members chosen on v4, so this number favours v4; double-difference catalogs (v3, v4, tomoDD) are also sharper than single-event locations (NonLinLoc) by construction. Compare like with like.

## By region: median depth / above surface / ANSS dz (near)

| region | v4.1 | v3 | v4.0 | NLL3D-i0 | NLL3D-i1 | v5.0 | v5.1 | v6.0 |
|---|---|---|---|---|---|---|---|---|
| Axial | 0.3 / 71% / -12.5 |  | 0.3 / 71% / -12.5 | 3.0 / 0% / -5.4 | 3.0 / 0% / -5.4 | 3.9 / 1% / -4.7 | 3.3 / 0% / -4.9 | 3.4 / 0% / -4.3 |
| Blanco | 9.2 / 21% / -9.2 | 9.3 / 22% / -1.6 | 9.2 / 21% / -9.2 | 11.0 / 0% / -1.6 | 10.1 / 0% / -2.7 | 9.0 / 0% / -2.6 | 9.3 / 0% / -2.9 | 9.9 / 0% / -4.3 |
| Cascades | 4.7 / 17% / -1.0 | 11.2 / 6% / +1.4 | 4.7 / 17% / -1.0 | 6.8 / 0% / +0.3 | 7.0 / 0% / +0.3 | 7.3 / 1% / +0.6 | 7.0 / 1% / +0.7 | 7.2 / 1% / +0.9 |
| Endeavour | 1.4 / 55% / +nan | 17.1 / 29% / +nan | 1.4 / 55% / +nan | 2.8 / 0% / +nan | 2.8 / 0% / +nan | 2.9 / 4% / +nan | 10.8 / 4% / +nan | 19.8 / 1% / +nan |
| Explorer | 8.9 / 26% / +nan | 18.7 / 20% / +nan | 8.9 / 26% / +nan | 33.1 / 0% / +nan | 32.9 / 0% / +nan | 35.2 / 0% / +nan | 40.5 / 0% / +nan | 41.9 / 0% / +nan |
| Gorda | 21.2 / 22% / -0.1 | 13.3 / 15% / -10.2 | 21.2 / 22% / -0.1 | 25.8 / 0% / +3.9 | 27.3 / 0% / +3.7 | 24.9 / 1% / +2.2 | 24.7 / 0% / +2.0 | 24.2 / 0% / +3.2 |
| JdF ridge/plate | 5.1 / 43% / +nan | 7.4 / 29% / +nan | 5.1 / 43% / +nan | 3.0 / 0% / +nan | 2.9 / 0% / +nan | 2.8 / 2% / +nan | 3.3 / 2% / +nan | 4.1 / 1% / +nan |
| Mendocino | 21.1 / 5% / +0.8 | 17.0 / 2% / -1.9 | 21.1 / 5% / +0.8 | 23.0 / 0% / +2.7 | 23.6 / 0% / +2.8 | 23.2 / 0% / +2.9 | 23.4 / 0% / +3.2 | 23.7 / 0% / +3.3 |
| Offshore forearc | 18.1 / 8% / -1.3 | 16.5 / 3% / -2.5 | 18.1 / 8% / -1.3 | 18.9 / 0% / +0.8 | 19.0 / 0% / +0.9 | 18.8 / 0% / +0.7 | 19.0 / 0% / +1.0 | 18.7 / 0% / +0.7 |
| Other onshore | 14.2 / 9% / -1.4 | 19.0 / 3% / +0.6 | 14.2 / 9% / -1.4 | 15.3 / 0% / -0.4 | 15.7 / 0% / -0.2 | 15.8 / 1% / +0.2 | 15.8 / 1% / +0.3 | 16.0 / 1% / +0.6 |
| Puget/Georgia | 14.3 / 12% / -1.3 | 16.7 / 3% / -0.9 | 14.3 / 12% / -1.3 | 14.4 / 0% / -1.7 | 15.4 / 0% / -0.6 | 15.0 / 0% / -0.8 | 14.7 / 0% / -0.8 | 15.3 / 0% / -0.3 |

## Each catalog against its parent

| catalog | against | n | dh p50 / p90 | dz p10 / p50 / p90 | dh > 5 km |
|---|---|---:|---|---|---:|
| v4.1 | v4.0 (parent) | 172,339 | 0.0 / 0.0 | +0.0 / +0.0 / +0.0 | 0% |
| v3 | v4.1 (reference) | 46,451 | 3.6 / 14.5 | -11.4 / -1.9 / +9.8 | 36% |
| v4.0 | v4.1 (reference) | 172,339 | 0.0 / 0.0 | +0.0 / +0.0 / +0.0 | 0% |
| NLL3D-i0 | v4.1 (parent) | 157,249 | 2.9 / 7.5 | -4.6 / +2.5 / +11.5 | 21% |
| NLL3D-i0 | v4.1 (reference) | 157,249 | 2.9 / 7.5 | -4.6 / +2.5 / +11.5 | 21% |
| NLL3D-i1 | NLL3D-i0 (parent) | 157,226 | 0.8 / 3.2 | -2.5 / +0.1 / +2.6 | 5% |
| NLL3D-i1 | v4.1 (reference) | 157,267 | 2.4 / 6.3 | -3.7 / +2.7 / +10.6 | 15% |
| v5.0 | NLL3D-i1 (parent) | 157,226 | 1.1 / 3.4 | -3.6 / -0.1 / +2.6 | 6% |
| v5.0 | v4.1 (reference) | 157,269 | 1.8 / 5.4 | -4.0 / +2.4 / +9.7 | 11% |
| v5.1 | v5.0 (parent) | 156,879 | 0.6 / 2.2 | -1.3 / +0.0 / +1.3 | 4% |
| v5.1 | v4.1 (reference) | 171,717 | 2.2 / 6.5 | -4.1 / +2.6 / +11.3 | 15% |
| v6.0 | v5.1 (parent) | 171,695 | 0.8 / 3.9 | -2.8 / +0.1 / +3.5 | 7% |
| v6.0 | v4.1 (reference) | 172,182 | 2.1 / 6.3 | -3.7 / +2.9 / +11.7 | 14% |

# Catalog comparison

Reference: **v4.1**. Built by `workflow/11_compare/compare_catalogs.py` from `catalogs.yaml`. Depths in km below sea level. ANSS events matched to v4.1: 16,584.

| catalog | velocity model | method | status |
|---|---|---|---|
| v4.1 | PNSN 1D regional models (I. McBrearty); ComCat for 3 added events | GENIE + GraphDD; targeted ELEP repick of four 2016 offshore events | final |
| v3 | PNSN 1D regional models | GENIE association, GraphDD + cross-correlation relocation (ver3 picks) | superseded |
| v4.0 | PNSN 1D regional models (I. McBrearty) | GENIE association, GraphDD relocation (ver6 picks) | superseded |
| NLL3D-i0 | CRESCENT Gen0 Vs + Brocher Vp, Vp/Vs >= 1.76, literature 1D at the ridges | NonLinLoc EDT, 3D grids, no station terms | superseded |
| NLL3D-i1 | CRESCENT Gen0 Vs + Brocher Vp, Vp/Vs >= 1.76, literature 1D at the ridges | NonLinLoc EDT, 3D grids, static station terms from iteration 0; tiers A-D, U = v4 kept | preliminary |

## Whole catalog

| catalog | events | located | depth p10/p50/p90 | above surface | shift from ref. h p50/p90, dz p50 | ANSS dz p50 (MAD), all / near | ANSS dh p50 | fault-plane RMS (km) |
|---|---:|---:|---|---:|---|---|---:|---:|
| v4.1 | 172,342 | 172,342 | -1.4 / 10.1 / 25.2 | 23.5% | 0.0 / 0.0, +0.0 | -0.9 (3.8) / -1.0 (2.4) | 2.2 | 1.14 |
| v3 | 63,887 | 63,887 | 3.2 / 15.6 / 28.6 | 6.2% | 3.6 / 14.5, -1.9 | +0.1 (5.3) / -0.4 (4.1) | 3.5 | 3.23 |
| v4.0 | 172,340 | 172,340 | -1.4 / 10.1 / 25.2 | 23.5% | 0.0 / 0.0, +0.0 | -0.9 (3.8) / -1.0 (2.4) | 2.2 | 1.14 |
| NLL3D-i0 | 157,249 | 157,249 | 2.6 / 11.3 / 30.0 | 0.0% | 2.9 / 7.5, +2.5 | +1.1 (4.7) / +0.2 (2.8) | 2.8 | 2.56 |
| NLL3D-i1 | 172,340 | 157,267 | 2.6 / 11.8 / 30.1 | 0.0% | 2.4 / 6.3, +2.7 | +1.2 (4.2) / +0.4 (2.4) | 2.5 | 2.24 |

Fault-plane RMS: median over the 119 fault surfaces with >= 20 located members in every catalog.

## By region: median depth / above surface / ANSS dz (near)

| region | v4.1 | v3 | v4.0 | NLL3D-i0 | NLL3D-i1 |
|---|---|---|---|---|---|
| Axial | 0.3 / 71% / -12.5 |  | 0.3 / 71% / -12.5 | 3.0 / 0% / -5.4 | 3.0 / 0% / -5.4 |
| Blanco | 9.2 / 21% / -9.2 | 9.3 / 22% / -1.6 | 9.2 / 21% / -9.2 | 11.0 / 0% / -1.6 | 10.1 / 0% / -2.7 |
| Cascades | 4.7 / 17% / -1.0 | 11.2 / 6% / +1.4 | 4.7 / 17% / -1.0 | 6.8 / 0% / +0.3 | 7.0 / 0% / +0.3 |
| Endeavour | 1.4 / 55% / +nan | 17.1 / 29% / +nan | 1.4 / 55% / +nan | 2.8 / 0% / +nan | 2.8 / 0% / +nan |
| Explorer | 8.9 / 26% / +nan | 18.7 / 20% / +nan | 8.9 / 26% / +nan | 33.1 / 0% / +nan | 32.9 / 0% / +nan |
| Gorda | 21.2 / 22% / -0.1 | 13.3 / 15% / -10.2 | 21.2 / 22% / -0.1 | 25.8 / 0% / +3.9 | 27.3 / 0% / +3.7 |
| JdF ridge/plate | 5.1 / 43% / +nan | 7.4 / 29% / +nan | 5.1 / 43% / +nan | 3.0 / 0% / +nan | 2.9 / 0% / +nan |
| Mendocino | 21.1 / 5% / +0.8 | 17.0 / 2% / -1.9 | 21.1 / 5% / +0.8 | 23.0 / 0% / +2.7 | 23.6 / 0% / +2.8 |
| Offshore forearc | 18.1 / 8% / -1.3 | 16.5 / 3% / -2.5 | 18.1 / 8% / -1.3 | 18.9 / 0% / +0.8 | 19.0 / 0% / +0.9 |
| Other onshore | 14.2 / 9% / -1.4 | 19.0 / 3% / +0.6 | 14.2 / 9% / -1.4 | 15.3 / 0% / -0.4 | 15.7 / 0% / -0.2 |
| Puget/Georgia | 14.3 / 12% / -1.3 | 16.7 / 3% / -0.9 | 14.3 / 12% / -1.3 | 14.4 / 0% / -1.7 | 15.4 / 0% / -0.6 |

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

# Track A: relocation of catalog v4 in a 3D model

Goal: catalog v5 locations with real depths (no events above the surface), full location uncertainty
(NonLinLoc PDFs), then relative relocation (GrowClust). These feed the focal mechanisms and fault surfaces.
Plan: `data/catalog_v4/next_steps/catalog_v4_next_steps.html`, section 4. We own the relocation (decided 2026-11-22).

Large files go to `/wd1/mdenolle_data/reloc_v5/`. QC outputs that fit in git go to `data/reloc_v5/`.

## Conventions

* Depth is km below sea level everywhere (grid z, NLL output, catalog v5).
* Stations sit at their true elevation or seafloor depth (`GTSRCE zSrce = -elev`). When the 1-arcmin surface
  under a station is lower than the station, the source goes 50 m below the grid surface instead
  (368 of 696 stations, at most 0.5 km). Station terms absorb the time difference.
* Picks: `data/catalog_v4/Cascadia_picks_catalog_v4.csv` (2.50 M picks, 696 stations). No repicking.
  Distance cap for location: 250 km (95% of picks are within 200 km).

## A1. Velocity model and travel-time grids

`a1_build_model.py` writes NLL `SLOW_LEN` grids `cascadia3d.{P,S}.mod.{hdr,buf}` (LAMBERT WGS-84, origin
45.0 N 125.8 W, standard parallels 41 and 49; 971 x 1434 x 86 nodes at 1 km; z from -5 to 80 km) and
`stations_GTSRCE.in`. `a1_qc_model.py` draws `data/reloc_v5/a1_model/a1_model_qc.png`.

Model choices (each one is a setting to vary later):

| Item | Choice |
|---|---|
| Vs | CRESCENT CVM Gen0, offshore water-column fill removed, columns extended upward from their shallowest solid value |
| Vp | Brocher (2005) eq. 9 from Vs, Vp/Vs at least 1.73 |
| Outside Gen0 (west of ~128.5 W, south of ~40 N offshore) | 1D young oceanic crust below the seafloor (`YOUNG_OCEAN`), 80 km taper into Gen0 |
| Axial, Endeavour, Blanco (outside Gen0) | published 1D location models below the seafloor: Wilcock et al. 2016 (Axial, MGDS 23844), Weekly et al. 2013 (Endeavour, after Cudrak & Clowes 1993), Kuna et al. 2019 Supp. Table 1 (Blanco); full weight within 25 km (20 km of the Blanco trace), tapered to `YOUNG_OCEAN` by 50 km (45 km). Alternatives not used: Baillard et al. 2019 3D Axial grids (datum unverified), Ren et al. 2023 Blanco VELEST model (only in a figure) |
| Outside Gen0 on land (north of ~49.6 N) | nearest Gen0 column; coordinates clamped to the Gen0 box (no extrapolation). Leaves vertical stripes; few events there |
| Water / air | Vp 1.5, Vs 0.4 / 0.3 km/s |
| Surface | GMT earth_relief 01m |
| Sphericity | no Earth-flattening transform; error grows with distance, to be checked against TauP at 250 km |

Known limits: Gen0 has no sediment layer offshore (OBS S delays go into station terms); Vp is not
measured, only scaled from Vs.

Travel times (needs NonLinLoc v7 binaries, built from github.com/ut-beg-texnet/NonLinLoc into
`/wd1/mdenolle_data/reloc_v5/src`): for each station, `GridCut` a ±250 km sub-volume of the P and S
model centred on the station, then `Grid2Time` (Podvin-Lecomte, `GTMODE GRID3D ANGLES_YES`; angles kept
for focal-mechanism takeoffs). About 86 MB per grid, ~240 GB with angles.

## A2, A3 (next)

A2: NLLoc (EDT, oct-tree) on all 172k events, station terms by iteration (`LOCSTAWT`/SSST), outlier picks.
A3: GrowClust on catalog differential times, checked against v4 and ComCat.

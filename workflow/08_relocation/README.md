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

## A2 status (2026-11-22)

Runs in `/wd1/mdenolle_data/reloc_v5/runs/`: `iter0_vpvs176` (no station terms), `iter1_vpvs176` (static
terms from iter0, `a2_station_terms.py`). NLLoc needs the search grid inside every station's ±250 km time
grid, so events are located in 25 km tiles with a tile ±75 km search box (`a2_make_obs.py`, `a2_run_nlloc.py`).

Catalog with all v4 events and location-quality tiers A-D, U: `a2_build_catalog.py`; summary
`data/reloc_v5/catalog_v5_trackA_iter1_vpvs176_tiers.md`. Use tiers A-B (`depth_constrained`) for depth work.

Catalog v5.1 rules (2026-11-24, D-B055/D-B057/D-B066/D-B067): both scripts read the catalog v4.1 event list (main
tree `data/catalog_v4`, 172,342 events). `a2_parse_hyp.py RUN [--deep DEEP_RUN]` keeps one row per event (the
deep-pass row when it is LOCATED) and adds the NLLoc comment, `ns_z` (final oct-tree cell height), run, tile and pass.
`a2_build_catalog.py` gives unlocated events no hypocentre (loc_source `none`, tier U, v4 values only in `v4_*`) and a
`u_reason` (LOCMETH minimum read from the run's own `nlloc.in`, picks counted in its `events.obs`); events located
with 5 phases are tier D.

Mount Hood test (`a2_model_1d_test.py`, tile t33_30, 269 events, nearest station 29 km): Ian's 1D
`pnsn_wa` gives median depth -1.0 km, `pnsn_nor` 8.1 km, the 3D model 16.7 km, all at RMS 0.26-0.29 s;
ANSS 4.3 km. Depths without a station within ~2 focal depths follow the velocity model, not the data.
The 3D model is the deepest of the three, which matches the deep bias of tiers B-D against ANSS.

## A3a. Frame transfer: v4 (GraphDD) geometry in the Track A frame (2026-11-23)

Question: the 3D relocation fixes depths but loses the double-difference tightness of v4 where single-event
NLL errors are large. `a3_frame_transfer.py` keeps the v4 relative positions and moves them by the
**smoothed** NLL-minus-v4 shift (median of up to 300 NLL events within R km, at least 10). The event-by-event
remainder of the shift is treated as NLL error and dropped. Events without enough neighbours keep the NLL
location (tier U: v4). This is an interim product until B5 (double-difference relocation in the final
3D model, with cross-correlation times); the same script can be rerun on the B4 catalog.

First check (R = 5 km, iter1, NLL-located events): local thickness 0.53 (v4) / 0.49 (NLL) / 0.46 km
(transfer); depth IQR in 1 km columns 3.87 / 2.67 / 1.95 km; above surface 23.0% / 0.4% / 1.1%. The 3D
catalog spreads in depth only at Gorda, Mendocino and the offshore forearc. NLL depths pile up on grid
nodes at Axial (10% at 3.01 km) and Blanco (7.03, 9.98, 11.01 km): `pinning.csv` tracks this.

Radius test (R = 1, 2, 3, 5, 7.5, 10, 15 km; NLL `depth` and `exp_depth`), run detached:

    cd ~/wt_reloc   # worktree of reloc/track-a
    setsid nohup ~/cascadia_obs_ensemble/.pixi/envs/amplitude/bin/python \
        workflow/08_relocation/a3_frame_transfer.py \
        > /wd1/mdenolle_data/reloc_v5/frame_transfer/run.log 2>&1 < /dev/null &
    pgrep -af a3_frame_transfer        # running?  DONE marker: data/reloc_v5/a3_frame_transfer/DONE

(The `default` pixi env fails to import pandas on this host, GLIBCXX; the `amplitude` env works.)
Outputs: catalogs `/wd1/mdenolle_data/reloc_v5/frame_transfer/catalog_xfer_{depth,exp_depth}_R*.csv.gz`
(all 172,340 events; `src` = xfer / nll3d / v4; field `fE fN fZ` in km, `n_field`, `r_eff_km`); QC tables in
`data/reloc_v5/a3_frame_transfer/` (`metrics.csv`, `consistency.csv`, `field.csv`, `pinning.csv`, `run_meta.json`).
How to choose R: the largest radius that keeps most tier A-B transfers within the NLL uncertainty
(`consistency.csv`, `in_1sig_z`, `in_hmax`) while thickness and depth IQR still drop (`metrics.csv`).
Tighter is not automatically better: the final test is the fit to cross-correlation differential times.

Radius test results (2026-11-23, iter1; ALL = 157,267 NLL-located events):

| catalog | thickness km | depth IQR 1 km cols | above surface | tier A / B within 1 sig_z | transferred |
|---|---|---|---|---|---|
| v4 (GraphDD) | 0.53 | 3.85 | 23.0% | - | - |
| NLL 3D `depth` | 0.50 | 2.58 | 0.4% | - | - |
| xfer `depth` R=2 | 0.40 | 1.46 | 0.3% | 0.86 / 0.88 | 58% |
| xfer `depth` R=3 | 0.43 | 1.59 | 0.4% | 0.85 / 0.86 | 70% |
| xfer `depth` R=5 | 0.46 | 1.83 | 0.7% | 0.83 / 0.84 | 83% |
| xfer `depth` R=10 | 0.48 | 1.87 | 1.0% | 0.82 / 0.83 | 94% |
| xfer `exp_depth` R=3 | 0.43 | 1.78 | 0.0% | 0.84 / 0.88 | 70% |

* Tightest at R = 2-3 km; larger radii smooth out real changes of the shift and drift away from NLL. R = 1 km
  covers only a third of the events. In every case more than 68% of tier A-B transfers fall within 1 sig_z
  of NLL, so the transfer is consistent with the NLL data.
* Grid-node pile-up is a property of the NLL maximum-likelihood depth (`depth`: Axial 10%, JdF 13%, Blanco 4%
  of events in one 0.01 km bin); the expectation depth (`exp_depth`) has none (<1%) but is ~4 km deeper
  (median 16.1 vs 11.8 km), pulled by deep PDF tails of poorly constrained events. The `depth` transfer keeps
  the JdF pile-up (9-11%).
* By region (R = 3, `depth`): Gorda depth IQR 4.86 (NLL) -> 1.87 (v4 3.62); offshore forearc 3.60 -> 2.16
  (v4 1.86); Blanco 5.42 -> 2.44 (v4 7.64); Mendocino 5.40 -> 4.36 (v4 3.54), only partly recovered.
* Next: adaptive radius (smallest R in 2-10 km with >= 10 neighbours) so coverage is ~95% at the R = 2-3 km
  tightness; decide `depth` vs `exp_depth` (or ML depth with pinned events flagged); then a catalog for the atlas.

**What A3a is and is not (PI discussion 2026-11-23).** It is an empirical shift of the v4 locations by the
local median of the NLL-minus-v4 differences, not a relocation: no travel times are recomputed and no picks
are re-fit. It assumes (1) the GraphDD relative positions, solved in 1D, are right, and (2) the 1D -> 3D
effect varies slowly over R. Its tightness numbers partly improve by construction (smoothing removes NLL
scatter whether real or not; shallow v4 events pushed down more than deep ones compress the depth range).
The radius test shows self-consistency only (tighter, and inside the NLL error bars), not accuracy.
**Required validation before any use:** double-difference residuals of v4, NLL 3D and the transfer computed
with the NLL 3D travel-time grids (v4 pick-time differences). If the transfer does not fit the
differential times in 3D as well as NLL, it is cosmetic and we wait for B5 (GrowClust3D or GraphDD in the 3D
model). Optional: cross-section figure of one cluster (Gorda or Blanco), v4 / NLL / transfer side by side.

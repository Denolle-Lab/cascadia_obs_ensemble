# Track B design memo: joint Vp/Vs local-earthquake tomography with TomoATT

Status: proposal for PI review (2026-11-22). Agent: Claude (Track B design subagent).
Context: `../DECISIONS.md` (D-B001..008), `../params.yaml`, Track A model in `../../08_relocation/README.md`.
Every citation below was opened or resolved through doi.org on 2026-11-22. Where I could not read
the methods (paywall, supplement only), the text says so. Numbers that come from our own files are
marked "(ours)".

Our problem in one paragraph. Domain 39.5-51.5 N, 121-131 W (1,334 km N-S, 692-858 km E-W),
-3 to 80 km below sea level. 696 stations in the Track A station file (ours): 337 sit below sea
level, 270 deeper than 0.5 km, 204 deeper than 2 km (median OBS depth 2.65 km, max 4.79 km); the
highest land station is at 2.19 km elevation. About 23k declustered events and 0.70 M P and S picks
(D-B003). Pick sigma 0.05/0.10/0.15 s for P, twice that for S (D-B008). Track A iter0 static terms
(ours, stations with >= 20 picks): OBS S terms span -0.76 to +0.59 s (5th-95th percentile, max 2.15 s),
land S terms -0.58 to +0.49 s; OBS P terms -0.16 to +0.33 s. The OBS S terms are centred near zero
because the locations absorb the common part of the sediment delay, which is itself a warning (section 6).

---

## 1. What published TomoATT / adjoint-state traveltime tomography (ATT) studies did

Two facts about the code shape everything else, both checked in the source
(github.com/TomoATT/TomoATT, commit 8b72f69, 2026-09-04):

* One run inverts one velocity field (`vel`, plus `xi`, `eta` for azimuthal anisotropy). The phase
  label in the source-receiver file is carried through but not used to pick a model. "Joint Vp/Vs" in
  every ATT paper below means a P run and a separate S run; Vp/Vs is the ratio of the two final models.
  Relocation inside a run therefore uses one phase only.
* `use_sta_correction` updates station terms only from common-source differential data
  (`InputParams::station_correction_update`, `src/input_params.cpp`); for absolute times the code
  prints "not supported now". It was built for teleseismic data. We cannot invert OBS static terms for
  local absolute times inside TomoATT.

Other code facts used below: spherical coordinates (Earth curvature built in), fast sweeping eikonal
solver with multiplicative factorisation at the source, first- or third-order stencil
(`stencil_order` 1 or 3), stations treated as sources by reciprocity (`swap_src_rec: true`), a
domain top above sea level so stations sit at their true elevation, multiple-grid parametrisation
(`n_inversion_grid`, typically 5 staggered grids), a "trapezoid" option that widens the horizontal
inversion spacing with depth, kernel-density normalisation (`Kdensity_coe`), step-size-controlled
gradient descent (`step_length` = maximum relative update per iteration), a cosine `depth_taper`
that zeroes model updates above a given depth, and data-type weights for absolute (ABS),
common-source differential (CS) and common-receiver differential (CR) times, each with
residual-, distance- or azimuth-dependent down-weighting.

### Table 1. Published ATT applications (body-wave traveltimes)

| Study | Region, data | Forward grid | Inversion grid | Iterations, step | Smoothing / regularisation | Data types and weights | Relocation | Phases | Anisotropy | Topography |
|---|---|---|---|---|---|---|---|---|---|---|
| Tong 2021a, JGR 126, e2021JB021818, [doi:10.1029/2021JB021818](https://doi.org/10.1029/2021JB021818) | Anza, S. California, 190x190 km; 629k P (31,815 ev, 315 sta), 629k S (22,409 ev); 1 event per 0.4 km cube | not stated in the text I read | 5 staggered grids, 15 km lateral; vertical 3 km near surface to 7.5 km at 25 km | 6 outer iterations, each 4 relocation + 4 model steps; stop when objective reduction < 5 % of the first-iteration reduction; max update 1.5 % slowness, 1 % of 20 km in location | multiple grids only | ABS P, ABS S | yes, eikonal-based location alternating with model updates; checkerboard starts with 2 km / 4 km / 0.4 s location errors | separate P and S runs; S start = final Vp/1.73 | no | not stated |
| Tong 2021b, JGR 126, e2021JB022365, [doi:10.1029/2021JB022365](https://doi.org/10.1029/2021JB022365) | Parkfield, P (full text not read) | Cartesian | multiple grids | not read | multiple grids | ABS P | not read | P | yes (azimuthal) | not read |
| Chen et al. 2023, GJI 234, 712, [doi:10.1093/gji/ggad093](https://doi.org/10.1093/gji/ggad093) | Parkfield, 4.0 x 1.4 deg rotated box; 1.22 M P from 33,051 ev, 555 sta; depth 1-20 km; 1 event per 0.01 deg x 0.01 deg x 0.5 km | 2 km horizontal, 0.5 km vertical, spherical | 5 staggered grids; "one wavelength sampled by about five nodes of each grid" | 40; first update capped at 1 %, cap cut by 10 % whenever the objective rises | multiple grids | ABS P | no (catalogue locations fixed) | P | yes | layered start with CRUST1.0 Moho |
| Chen et al. 2025, Comput. Geosci. 204, 105995, [doi:10.1016/j.cageo.2025.105995](https://doi.org/10.1016/j.cageo.2025.105995) (TomoATT paper) | Parkfield benchmark 160 x 440 x 50 km; 1.22 M P, 32,721 ev, 607 sta | ~5 x 5 x 1 km (149,787 nodes) | multiple grids, `Kdensity_coe` 0.3 in the shipped example | 80 iterations, 64 cores, 57 min | multiple grids + kernel-density normalisation | ABS | synthetic: 50 iterations location only, 40 joint, 100 relocation in final model | P | yes | example domain starts 5 km above sea level |
| Tong et al. 2024, GJI 236, 139, [doi:10.1093/gji/ggad416](https://doi.org/10.1093/gji/ggad416) | method (synthetics) | 0.2 km | 5 grids, 16.13 km lateral | 40; update capped at +-2 % | multiple grids | CS (station pairs < 15 km) and CR (event pairs 2-15 km); CR "less sensitive to velocity errors and suitable for earthquake location" | ABS, CS or CR | P | no | n/a |
| Wang et al. 2024, BSSA 115, 228, [doi:10.1785/0120240035](https://doi.org/10.1785/0120240035) | Los Angeles basin; 151k P (24,918 ev, 218 sta), 150k S (22,701 ev); depth < 20 km, distance < 120 km; 1 event per 0.5 km cube | 0.25 x 0.25 x 0.4 km; domain 4 km above to 35 km below sea level "incorporating the elevation of all stations" | 5 grids shifted by 0.2 x spacing; 10 km lateral chosen from checkerboards at 12.5/10/7.5 km | stable after ~20 | multiple grids | ABS P, ABS S | yes, alternated every iteration; mean shifts 0.55 km horizontal, 0.54 km vertical | separate P and S; three different starting models compared | no | stations at elevation |
| Zhang et al. 2025, Sci. China Earth Sci. 68, 702, [doi:10.1007/s11430-024-1504-x](https://doi.org/10.1007/s11430-024-1504-x) | SE Tibet; 499k ABS P (38,661 ev), 5.6 M CR (event pairs < 10 km), 650k CS (ray azimuth difference < 30 deg) | in supplement (not read) | coarse grid first, then finer (supplement) | not read | multiple grids + kernel-density normalisation | ABS + CS for the model; ABS + CR for relocation | yes | P | isotropic run and anisotropic run in parallel | "incorporation of surface topography" listed as a code feature |
| Wu, Chen & Tong 2025, EPSL 660, 119362, [doi:10.1016/j.epsl.2025.119362](https://doi.org/10.1016/j.epsl.2025.119362) | New Zealand; 1.06 M P, 74,458 ev + shots, 591 sta; depth < 40 km, distance < 130 km; 1 event per 1.5 x 1.5 x 1.0 km; >= 8 picks/event | in supplement (not read) | in supplement | RMS 0.78 -> 0.37 s | multiple grids | ABS P | yes, three-step scheme | P | yes; checkerboards 100 km (velocity) and 150 km (anisotropy); leakage < 1 % where coverage is dense | not read |
| Bai et al. 2026, EPSL 679, 119879, [doi:10.1016/j.epsl.2026.119879](https://doi.org/10.1016/j.epsl.2026.119879) | Myanmar; PhaseNet picks; 46,751 P + 46,751 S from 4,651 ev; 1 event per 0.08 deg x 0.08 deg x 0.5 km; box weighting of events and stations | 0.04 deg lateral, 0.5 km vertical | 6 staggered grids; 0.5 deg (velocity), 0.7 deg (anisotropy); 12 layers thickening with depth | 60 per phase | multiple grids | ABS; S objective uses T_S,obs + (T_P,pred - T_P,obs), i.e. an S-P constraint | yes, in the P run after each model update | separate Vp and Vs; start Vs from ambient-noise model, Vp from Brocher (2005) | P only | not read |
| Chen et al. 2026, Nat. Geosci. 19, 223, [doi:10.1038/s41561-025-01893-z](https://doi.org/10.1038/s41561-025-01893-z) (TomoATT example `realcase3_regional_tomography_Turkey`) | E. Anatolian fault | 61 x 121 x 61 nodes over 60 km x 6 deg x 3 deg (1 km vertical, ~0.05 deg lateral); domain top 10 km above sea level | 5 grids, 0.5 deg lateral; depth nodes -20,-10,-2,0,2,6,12,20,30,42,54,68 km | 3 steps: 1D inversion, relocation (50 iterations), joint (40 iterations, 1 model + 1 relocation step per loop); `step_length` 0.01, decay 0.95, `Kdensity_coe` 0.3 | multiple grids + kernel density | ABS + CS (azimuth weight [30, 60, 1, 0.1]); CR off; residual weight [1, 3, 1, 0.1], distance weight [50, 150, 1, 0.1] | yes; max change 5 km and 0.8 s | P | yes | stations at elevation |

Methods paper for the teleseismic variant (not used here): Chen et al. 2023, JGR 128,
[doi:10.1029/2023JB027348](https://doi.org/10.1029/2023JB027348) (Thailand, CS teleseismic, 10 x 10 x 5 km).

What I take from the table:

* The forward grid is set by accuracy, not by resolution: 2 x 0.5 km (Chen 2023), 0.25 x 0.4 km (LA basin),
  ~5 x 1 km (benchmark), 0.04 deg x 0.5 km (Myanmar). The TomoATT paper reports first-order errors at
  surface stations below 0.1 s out to 800 km when the spacing is <= 5 km, and L1 errors of 0.052, ~0.025,
  ~0.017, 0.0096 s for 20, 10, 5, 2.5 km spacing in a smooth model.
* The inversion grid is always 5-6 staggered regular grids, with the lateral spacing picked so that
  one checkerboard wavelength is sampled by ~5 nodes of one grid, and set from checkerboard tests.
  No explicit damping or Laplacian smoothing in any published case; the regularisation is the grid,
  the step-size cap (1-2 % per iteration) and the iteration count (20-80).
* Relocation is alternated with model updates in every regional study that relocates.
* Vp and Vs are always separate runs. Only Bai et al. 2026 couple them, through the S-P term.
* Anisotropy: inverted in the method papers and NZ/Tibet/Myanmar, never for S, and always with a
  coarser grid than velocity.
* Every study declusters to one (or a few) events per small cell and restricts depth/distance to
  avoid Moho-refracted paths (LA < 20 km depth and < 120 km; NZ < 40 km and < 130 km; Parkfield 1-20 km).

None of these studies is offshore or uses OBS. We would be the first ATT application with seafloor
stations, which is a reason to do the forward-accuracy and OBS tests in section 2 before the real run.

---

## 2. Recommended setup

### 2.1 Forward grid: 0.02 deg (lat) x 0.025 deg (lon) x 1.0 km, spherical, z from -3 to 80 km

Nodes: 601 (lat) x 401 (lon) x 84 (depth) = 20.2 M. Lateral spacing 2.2 km N-S, 1.7-2.1 km E-W.

Memory. TomoATT Table 1 (Chen et al. 2025) gives 0.21 GB per process for a 101^3 (1.03 M node) grid,
~200 B per node. 20.2 M nodes -> ~4.1 GB per simultaneous source. With `n_sims` 32 and
`nproc_sub` 4 (128 ranks; hyperplane sweeping adds no memory) we need ~130 GB of the 250 GB.
A 1 km x 0.5 km grid (160 M nodes, ~33 GB per source) would allow only ~6 simultaneous sources
and would run ~40 times slower; it is not worth it given picks with sigma >= 0.05 s.

Run time. From the TomoATT Parkfield benchmark (57 min x 64 cores for 80 iterations x 607 sources on
0.15 M nodes) the cost is ~3e-5 core-s per node per source per iteration; the older Chen et al. 2023
run (1,378 CPU-h for 40 iterations, 555 sources, ~1.8 M nodes) gives ~1.2e-4. For 20.2 M nodes and
~700 stations as sources: 1 to 4 h per iteration per phase on 128 cores, so 40 iterations of P plus
40 of S take 3-13 days. Acceptable; a coarse 3 km x 1 km pilot (8.5 M nodes) cuts this by 2.4.

Accuracy test before adoption (proposed acceptance criterion). For 20 stations (10 OBS on the
slope and shelf, 10 land incl. Cascades), compute times on the 0.02/0.025/1 km grid and on a
0.01/0.0125/0.5 km reference grid in the Track A model; also run `stencil_order` 3. Accept the grid
if median |dt| <= 0.02 s and 95th percentile <= 0.05 s for P within 250 km (and 0.04/0.10 s for S).
If OBS on steep bathymetry fail, use stencil order 3 or drop to 0.5 km vertical in the top 10 km
(not possible on a uniform TomoATT grid; would mean 0.5 km throughout, 40 M nodes, still fits).
Also compare against the Track A NonLinLoc grids to catch datum or sign errors.

Vertical extent. Top at -3 km covers all stations (highest 2.19 km). Water nodes: Vp 1.5 km/s;
S model water and air at the shallowest solid Vs capped at 0.4 km/s (slower than any rock below, so
no first arrival uses it; Track A used the same idea). Bottom at 80 km covers slab events beneath
Puget Sound (to ~60-70 km).

### 2.2 Inversion grid: 5 staggered grids, two stages, depth-dependent lateral spacing

Stage 1 (coarse, all regions): lateral 0.30 deg lat x 0.40 deg lon (~33 x 31 km).
Stage 2 (fine): 0.20 deg x 0.25 deg (~22 x 20 km), started from the stage 1 model.
Depth nodes (km below sea level, both stages):
-3, 0, 2, 4, 6, 8, 11, 14, 18, 22, 27, 33, 40, 48, 57, 67, 80.
Use the `trapezoid` option to widen the lateral spacing by 1.5x between 30 and 60 km depth, where only
slab events and long paths sample the model. Grids offset by 0.2 x spacing in all three directions
(the LA-basin choice; TomoATT default).

Justification. Station spacing is ~10-30 km in Puget Sound and the Willamette valley, ~50 km
elsewhere on land, ~70 km on the main Cascadia Initiative grid and a few km at Axial/Endeavour.
With five nodes per wavelength (Chen et al. 2023; Wang et al. 2024) a 0.2 deg grid targets
anomalies of ~40-50 km half-wavelength, which is the smallest checker we can expect to recover
offshore. The Parkfield example uses 0.3 deg; Myanmar and Turkey 0.5 deg. Going finer than
0.2 deg only where checkerboards say so (Puget Sound, Axial) is a later step. Nodes at -3 and 0 km
exist so that land topography is parametrised; offshore they lie in water and get ~zero kernel.

### 2.3 Regularisation and optimisation

* `optim_method` 0 (step-size-controlled gradient descent), `step_length` 0.01 (max 1 % update per
  iteration; Chen et al. 2023, Turkey example), `step_method` 1, decay 0.9, gradient-angle rule 120 deg,
  change factors [0.5, 1.2] (code defaults).
* `Kdensity_coe` 0.3 (the value in both shipped real-data examples). Raise to 0.5 only if
  checkerboards show offshore recovery lagging; it amplifies noise in sparse areas (Chen et al. 2025).
* No Laplacian smoothing (`smooth_method` 1 is marked experimental).
* `depth_taper` not used (the seafloor is not flat); check after iteration 1 that water and air nodes
  did not move by more than 1 %, and reset them between stages if they did.
* The iteration count is part of the regularisation: synthetics must use the same settings and
  the same number of iterations as the real run.

### 2.4 Iterations and coupling with relocation

Because TomoATT relocates with one phase per run, and offshore depths need S, I recommend
alternating runs instead of TomoATT's internal joint mode for the main model:

```
outer loop k = 1..3
  a. P model update, locations fixed (run_mode 1), ABS-P (+ CS-P in stage 2), 15-20 iterations
  b. S model update, same fixed locations, ABS-S (+ CS-S), 15-20 iterations
     option: S data corrected by the P residual of the same event-station pair (Bai et al. 2026),
     which makes the S run fit S-P and keeps origin-time and location errors out of Vs
  c. relocate all ~23k events with P and S in the new pair of models
     (export station traveltime fields with output_source_field, locate with NonLinLoc/our own
     grid-search locator on those fields), keep Track A conventions; update origin times
```

Stop the outer loop when the median hypocentre change is < 0.5 km horizontal and < 1 km vertical,
or after 3 loops. Total model iterations per phase: 45-60, inside the published 40-80 range.

Alternative kept as a check: TomoATT run_mode 3 (`model_update_N_iter` 1, `relocation_N_iter` 1)
on the P data only, `max_change_dep_lat_lon_ortime` [5, 5, 5, 0.8] as in the Turkey example. Useful
onshore, where P alone fixes depths; offshore it will trade depth against velocity (section 6).

Cost of step c: 696 stations x 2 phases x 20.2 M nodes x 4 B = ~113 GB of fields per loop; write
in single precision to /wd1 and delete after each loop.

### 2.5 Data types and weights

| Data | Use in model update | Use in relocation | Weight | Why |
|---|---|---|---|---|
| ABS P | yes, all stages | yes | pick weight w = (0.05 s / sigma)^2 -> 1.0 / 0.25 / 0.11 for classes 2/1/0 | standard; carries absolute velocity |
| ABS S | yes (separate S run) | yes | same formula with sigma_S = 2 sigma_P; inside the S run only the relative weights matter | Vs, Vp/Vs, offshore depth |
| CS differential (same event, two stations) | stage 2 only | no | global weight 0.5 after `balance_data_weight`; azimuth weight [30, 60, 1, 0.1] (Zhang et al. 2025; Turkey example); station separation <= 60 km | cancels origin time and most of the location error, so offshore events with poor depth still constrain receiver-side structure; emphasises shallow structure under stations (Tong et al. 2024). Down-weighted because our CS times are differences of the same catalogue picks (no cross-correlation), so they repeat ABS information with sqrt(2) sigma |
| CR differential (same station, two events) | no | not in B2 | n/a | best for relative location (Tong et al. 2024) but GrowClust in B5 does this with waveform-free catalogue differences anyway (D-B002); adding 10^6-10^7 CR data to B2 buys little for the velocity model |

Residual weights (TomoATT piecewise-linear, full weight below the first value, 0.1 at and above the
second): P [0.5, 1.5, 1, 0.1] s, S [1.0, 3.0, 1, 0.1] s, i.e. about 3 sigma and 10 sigma of the worst
class. Distance weight [150, 250, 1, 0.3] km: full weight to 150 km, tapering to 0.3 at 250 km,
because first arrivals beyond ~150 km are mostly Moho- or slab-refracted, and the starting model
places those interfaces only approximately. Published studies cut harder (120-130 km);
we keep the long paths because offshore events have no closer stations.

Event weights: on top of the 3 km x 3 declustering, multiply each event weight by
1/sqrt(n_events in its 10 km cell) (a box weighting in the spirit of Bai et al. 2026) so that
Axial, Mendocino and the Blanco ridge do not set the step size for the whole model. Station
weights: 1/sqrt(n_stations within 10 km) for the Axial/Endeavour cabled arrays.

### 2.6 Anisotropy: no

Not for the first model. Reasons: our event-station azimuthal coverage offshore is one-sided
(OBS west, land east); published ATT anisotropy used 1.4-1.5x coarser grids than velocity (Wu et al.
2025: 100 km vs 150 km checkers; Bai et al. 2026: 0.5 vs 0.7 deg) and showed velocity-anisotropy
leakage outside dense coverage; S anisotropy is never inverted. If a reviewer asks, run a P-only
anisotropic test with `update_azi_ani` on and leakage checkerboards (velocity-only and
anisotropy-only inputs) as in Wu et al. 2025.

### 2.7 Stopping criterion

Within a run, stop at the first of: (1) objective decrease over the last 5 iterations < 1 % of the
current value; (2) normalised misfit sum(w r^2)/sum(w) / sigma_ref^2 <= 1.1 (target chi^2 ~ 1;
fitting below the pick noise is overfitting); (3) 20 iterations per run. The Anza criterion
(reduction < 5 % of the first-iteration reduction; Tong 2021a) is the fallback. Record the objective,
RMS by region (onshore, shelf, deep ocean, Axial, Mendocino) and by phase at every iteration in
the run manifest.

---

## 3. Ocean-bottom stations: sediment effects on S (and P)

What the literature measured in Cascadia:

* Ruan, Forsyth & Bell 2014, JGR 119, 6357, [doi:10.1002/2014JB011162](https://doi.org/10.1002/2014JB011162):
  sediment shear velocity and thickness beneath deep-water Cascadia Initiative OBS from the
  displacement/pressure ratio of 0.1-0.2 Hz Rayleigh waves; gives depth-dependent Vs and the
  sediment S delay per station. A later paper summarising Ruan et al. 2014 and Bell et al. reports
  abyssal-plain sediment Vs of 0.1-1.0 km/s, thickness 0.1-1.4 km, S delays 0.5-2.0 s (secondary
  citation; I could not open the Ruan PDF, the mirror was down).
* Bell, Ruan & Forsyth 2016, JGR 121, 7298, [doi:10.1002/2016JB012990](https://doi.org/10.1002/2016JB012990):
  Rayleigh-wave tomography of the Juan de Fuca plate with sediment structure in the model; notes the
  thick sediment of the Astoria and Nitinat fans.
* Rychert, Harmon & Tharimena 2018, Sci. Adv. 4, eaao1908, [doi:10.1126/sciadv.aao1908](https://doi.org/10.1126/sciadv.aao1908):
  P-to-S delays from the sediment base at each Cascadia Initiative OBS, used to set per-station sediment
  Vs (fixed velocity above 1.5 s delay). The per-station values are in Supplementary Table S1
  ("P-to-S delays measured from Cascadia Initiative Array stations"),
  http://advances.sciencemag.org/cgi/content/full/4/2/eaao1908/DC1 . The download was blocked from
  here (403); fetch by hand.
* Audet 2016, GJI 205, 1740, [doi:10.1093/gji/ggw111](https://doi.org/10.1093/gji/ggw111) and
  Janiszewski & Abers 2015, SRL 86, 1261, [doi:10.1785/0220150104](https://doi.org/10.1785/0220150104):
  receiver functions at CI OBS; sediment reverberations dominate deep-water sites, shallow forearc
  sites are usable. Source of Ps-delay estimates on the shelf, where the compliance method is noisy.
* Janiszewski et al. 2019, GJI 217, 1929, [doi:10.1093/gji/ggz051](https://doi.org/10.1093/gji/ggz051):
  amphibious phase velocities with tilt and compliance corrections; no per-station sediment table.
* GlobSed (Straume et al. 2019, G3 20, 1756, [doi:10.1029/2018GC008115](https://doi.org/10.1029/2018GC008115)):
  5 arc-minute total sediment thickness; NOAA page https://ngdc.noaa.gov/mgg/sedthick was retired
  in 2025; archived copies at PANGAEA (doi:10.1594/PANGAEA.982339) and Caltech Data
  (https://data.caltech.edu/records/k4070-ngc79). Total thickness, not the low-Vs top layer; poor on
  the accretionary prism, so use only where no station measurement exists.

What local-earthquake studies with OBS did:

* Lange, Ren & Grevemeyer (Blanco, arXiv 2503.17148, submitted to JGR): NonLinLoc with P and S station
  corrections iterated 4 times on 1,380 high-quality events, then SIMUL2000 2D Vp and Vp/Vs; no explicit
  sediment layer.
* Kuna et al. 2019, Nat. Geosci. 12, 138, [doi:10.1038/s41561-018-0287-1](https://doi.org/10.1038/s41561-018-0287-1):
  Blanco OBS; P-only locations in a 1D model (Supp. Table 1), Vp/Vs 1.85 crust / 1.73 mantle from S.
* Baillard et al. 2019, JGR 124, 12997, [doi:10.1029/2019JB017970](https://doi.org/10.1029/2019JB017970):
  Axial, joint 3D P and S with relocation from ~3,900 events, starting from an active-source 3D Vp.
* Guo, McGuire & Zhang 2021, Nat. Geosci. 14, 341, [doi:10.1038/s41561-021-00740-1](https://doi.org/10.1038/s41561-021-00740-1):
  onshore-offshore Vp/Vs of southern Cascadia with tomoTD; I could not read the methods (paywall),
  so how they treated OBS sediment is unverified.

Recommendation for us (to become D-B009 or a D-B006 amendment):

1. A priori OBS corrections, applied to the observed times before TomoATT, not inverted. The forward
   grid (1 km vertical) cannot represent a 0.1-1.4 km sediment layer and TomoATT cannot invert
   static terms for absolute local times. For each OBS: one-way vertical delay
   dt_S = integral over the sediment column of (1/Vs_sed(z) - 1/Vs_model(z)) dz, and the same for P with
   sediment Vp. Rays in sediment with Vs <= 1 km/s are within a few degrees of vertical, so the
   vertical delay is a good approximation for all local arrivals. Sources, in order of preference:
   Rychert et al. 2018 Table S1 Ps delays (deep CI sites); Ruan et al. 2014 / Bell et al. 2016 per-station
   profiles; our own Ps-P delays from teleseismic receiver functions on shelf OBS (a one-day job with
   our data); GlobSed thickness with the Ruan et al. Vs-depth law where nothing else exists. Expected
   size: S 0.5-2 s, P <= 0.2 s.
2. Check the corrections against the Track A terms: after correction, the OBS S term distribution
   should match the land one (5-95 % range ~ +-0.5 s). Large remaining terms flag bad sediment
   estimates or polarity/horizontal-orientation issues in S picks.
3. Do not add a sediment layer to the 3D starting model in this pass; it would be smeared over 1-2
   grid cells and bias Vs in the top 2-3 km of crust. Revisit if we move to a finer near-surface grid.
4. Sensitivity test: run the S stage with (a) a priori corrections, (b) no corrections, (c) Track A
   iter1 S terms. Report the Vs and Vp/Vs differences in the top 10 km under the shelf.

---

## 4. Resolution assessment

Synthetic setup (both P and S, same event-station-phase geometry and weights as the real data):

* Checkerboards, sinusoidal, +-8 % (the amplitude used by Chen et al. 2023, Wang et al. 2024, Wu et al.
  2025), on the starting model:
  - depth 0-12 km: half-wavelength 40 km (and 25 km over Puget Sound, Axial only)
  - depth 12-30 km: 60 km
  - depth 30-80 km: 100 km
  - vertical half-wavelength 4 km (0-12 km), 8 km (12-30 km), 15 km (30-80 km)
  Two phase-shifted versions so that no conclusion rests on nodes coinciding with checker peaks
  (Tong 2021a makes the same point).
* Spike tests: 30 km wide, 6 km thick, +-10 % blobs at targets we will interpret: Mendocino/Gorda slab at
  20 km; offshore forearc at 10 km under the mid-shelf; Puget Sound at 15 km; Blanco at 6 km below
  the seafloor; Axial at 3 km; the slab at 45 km under the Olympics. Report smearing length in x, y, z.
* Restoring test: final model as input, recover from the starting model (shows which features the
  data can rebuild at full amplitude).
* Noise: Gaussian with the pick's own sigma (0.05/0.10/0.15 s P, double for S), not a single 0.05 s
  value as in the published tests; plus a 5 % outlier population at +-1 s to exercise the residual weights.
* Location errors: perturb starting hypocentres with N(0, 3 km) horizontal, N(0, 5 km) vertical,
  N(0, 0.5 s) origin time (TomoATT paper used 0.1 deg / 10 km / 0.5 s; Anza 2 / 4 km / 0.4 s), and run
  the full outer loop of 2.4 including relocation, so the test includes the depth-velocity trade-off.
* OBS sediment: add a synthetic sediment S delay (drawn from the a priori corrections) and remove
  it with a 20 % random error, to show what an imperfect correction maps into the model.

Recovery metrics and what goes in the paper:

* Per node: local Pearson correlation between input and output in a moving window one checker wide,
  and amplitude ratio. Define "resolved" as correlation >= 0.7 and amplitude ratio >= 0.5; draw that
  contour on every model figure, P and S separately; Vp/Vs only where both are resolved.
* Hypocentre recovery: median and 90th percentile of horizontal/vertical/origin-time error by
  region and Track A tier.
* Paper: one checkerboard figure (map slices at 6, 15, 30 km, one cross-section per region, P and S),
  the spike figure with smearing lengths, a table of recovery statistics by region, and the noise/
  location-error settings in the methods. Supplement: all depths, restoring test, OBS test.

Model-independence checks:

1. 1D start: invert from a 1D Vp/Vs model hung below the solid surface (seafloor offshore), obtained
   by a 1D inversion of our data first (the Zhang et al. 2025 and Turkey-example procedure). In the
   resolved region, require |Vp(3D start) - Vp(1D start)| < 2 % (3 % for Vs) for features we interpret;
   map the difference.
2. Alternative 3D start: Delph et al. 2018 Vs with Brocher Vp onshore (independent of Gen0 data).
   Wang et al. 2024 compared three starting models in the same way.
3. Jackknife: 5 runs with a random 70 % of picks (Wu et al. 2025), and one run without OBS data, to show
   what the OBS add.
4. Relocation choice: final model with TomoATT P-only internal relocation versus the external P+S loop.

---

## 5. Existing models to compare against

| Model | Data / method | Parameter | Overlap with us | Note |
|---|---|---|---|---|
| CRESCENT CVM Gen0 (He et al. 2026; figshare [doi:10.6084/m9.figshare.31902061](https://doi.org/10.6084/m9.figshare.31902061); viewer cvm.cascadiaquakes.org) | Bayesian joint ambient noise + receiver functions | Vs, 0.2 deg, 0.5 km, -4 to 100 km | most of the domain east of ~128.5 W | our starting model; comparison measures how far the data moved us, not independent validation |
| Delph, Levander & Niu 2018, GRL 45, [doi:10.1029/2018GL079518](https://doi.org/10.1029/2018GL079518); EarthScope EMC Cascadia_ANT+RF_Delph2018 | ambient noise + RF | Vs, -3 to 80 km | onshore forearc and arc | already in `data/tomography/Delph2018.nc` |
| Delph, Thomas & Levander 2021, EPSL 556, 116724, [doi:10.1016/j.epsl.2020.116724](https://doi.org/10.1016/j.epsl.2020.116724) | RF with 3D correction, Vs + variable Vp/Vs | Vs, Vp/Vs | onshore forearc | compare forearc Vp/Vs |
| USGS Cascadia CVM: Stephenson, Reitman & Angster 2017, v1.6, [doi:10.3133/ofr20171152](https://doi.org/10.3133/ofr20171152); Wirth et al. 2025, v1.7, [doi:10.3133/ofr20251045](https://doi.org/10.3133/ofr20251045) | compiled geological/geophysical model with basins, slab | Vp, Vs | whole margin incl. offshore | the community ground-motion model; compare basins (Seattle, Tacoma, Willamette) |
| Gao 2018, Nat. Commun. 9, [doi:10.1038/s41467-018-03655-5](https://doi.org/10.1038/s41467-018-03655-5) | 3D Vs, slab geometry vs seismicity (methods not read) | Vs | margin | check extent before use |
| Bell, Ruan & Forsyth 2016 (above) | Rayleigh-wave tomography with CI OBS | Vs of Juan de Fuca plate, 25-65 km focus | offshore plate | only long wavelengths overlap |
| Merrill et al. 2020, JGR 125, [doi:10.1029/2020JB019750](https://doi.org/10.1029/2020JB019750) | tomoDD, Washington forearc | Vp, Vs, Poisson's ratio | Puget Sound, Olympics | closest analogue to our onshore Washington result (Siletzia) |
| Merrill et al. 2022, G3 23, [doi:10.1029/2021GC010205](https://doi.org/10.1029/2021GC010205) | tomoDD, Nootka fault zone | Vp, Vp/Vs | N Vancouver Island, offshore north edge | edge of our domain (north of ~49.5 N) |
| Savard, Bostock & Christensen 2018, G3 19, 1881, [doi:10.1029/2017GC007417](https://doi.org/10.1029/2017GC007417) | tomoDD, northern Cascadia forearc | Vp, Vp/Vs | S Vancouver Island, Strait of Georgia | |
| Ramachandran et al. 2005, JGR 110, [doi:10.1029/2004JB003258](https://doi.org/10.1029/2004JB003258) | ~50k earthquake + active-source first arrivals | Vp | SW British Columbia, Juan de Fuca Strait | |
| Van Wagoner et al. 2002, JGR 107, 2381, [doi:10.1029/2001JB000710](https://doi.org/10.1029/2001JB000710) | PNSN earthquakes + SHIPS, joint nonlinear | Vp to ~30 km | central Puget Lowland | basin and Seattle fault geometry |
| Guo, McGuire & Zhang 2021 (above) | tomoTD, onshore-offshore | Vp, Vp/Vs on the megathrust | Mendocino / southern Cascadia offshore | the only LET with OBS in our southern region |
| Baillard et al. 2019 (above); models MGDS [doi:10.1594/IEDA/324821](https://doi.org/10.1594/IEDA/324821) | joint P, S, relocation | Vp, Vs | Axial, top ~3 km | far finer than our grid; compare only the average Vp/Vs |
| Lange, Ren & Grevemeyer, arXiv [2503.17148](https://arxiv.org/abs/2503.17148) (preprint) | SIMUL2000 2D | Vp, Vp/Vs along Blanco | Blanco | Vp/Vs > 2 in most of the fault zone |
| Bodmer et al. 2018, GRL 45, 6954, [doi:10.1029/2018GL078700](https://doi.org/10.1029/2018GL078700); Hawley, Allen & Richards 2016, Science 353, 1406, [doi:10.1126/science.aad8104](https://doi.org/10.1126/science.aad8104) | teleseismic P with CI OBS | dVp, upper mantle | overlap only 50-80 km | compare slab position, not amplitudes |

Comparison method: interpolate each model to our grid, compute dlnV relative to each model's own
depth mean (absolute values differ by method), and report correlation in the resolved region, per depth.

---

## 6. Risks and mitigations

1. Depth-velocity trade-off for offshore events with no station within 1-2 focal depths
   (Blanco, Gorda, offshore forearc after 2015). In a joint inversion these events can move in depth
   and push velocity anomalies along their rays. Mitigation: relocation step c uses P and S; for
   events whose closest station is farther than 1.5 x depth, bound depth changes to +-3 km per loop; CS
   differential data in stage 2 (no origin-time sensitivity); the location-perturbed synthetic of
   section 4 measures the bias; report depth shifts by tier.
2. The common sediment S delay at OBS. If not removed, offshore events recorded mainly by OBS absorb it
   in origin time and depth (S late everywhere -> event deeper or later), and the residual goes into
   low Vs at 0-5 km under every OBS. Track A OBS S terms centred near zero (section intro) show this
   absorption already happens. Mitigation: section 3.
3. One-sided geometry: OBS only 2011-2015 (Cascadia Initiative) plus permanent sites at Axial,
   Endeavour and NEPTUNE; most offshore events outside 2011-2015 have only land stations to the east.
   Expect E-W smearing offshore. Mitigation: gap <= 200 deg already (D-B003); resolution masks;
   jackknife without OBS.
4. Event clustering (Axial, Mendocino, Blanco ridge, Puget Sound swarms): declustering (D-B003),
   event box weights (2.5), `Kdensity_coe` 0.3.
5. Separate P and S runs give two location sets. Mitigation: locations fixed inside TomoATT runs;
   one external P+S relocation per outer loop (2.4).
6. Vp/Vs at shallow depth: sediments and the accretionary prism have Vp/Vs well above 2 and are thinner
   than the grid; land Vp/Vs below ~2 km is constrained only near dense stations. Report Vp/Vs only where
   P and S checkerboards both pass, and not above 2 km below the solid surface offshore.
7. Moho- and slab-refracted arrivals at 150-250 km: distance taper (2.5); events deeper than the
   Moho in the Gen0 model kept only with >= 12 picks (already in D-B003).
8. Forward error near steep bathymetry and at the water-rock contrast (stations as eikonal sources
   in swap mode): accuracy test of 2.1 on slope and shelf OBS before the real run.
9. Water and air nodes updated through the multigrid basis functions: check and reset (2.3).
10. Over-fitting with no explicit damping: chi^2 stopping rule (2.7) and identical iteration counts in
    synthetics.
11. Starting model gaps: Gen0 has no coverage west of ~128.5 W; Track A uses 1D regional models there.
    The 1D-start inversion (section 4) shows how much those choices matter.
12. DL S picks at OBS (horizontals with tilt noise) have heavier tails than land picks: residual weights
    (2.5); compare residual distributions OBS vs land after iteration 1, and if OBS S tails are wider,
    raise their sigma class by one.

---

## Proposed decisions

**Tomography forward grid.** Recommendation: spherical 0.02 deg x 0.025 deg x 1.0 km, z -3 to 80 km
(20.2 M nodes, ~130 GB with 32 simultaneous sources on 128 cores), adopted after the accuracy test
in 2.1 (median <= 0.02 s, 95 % <= 0.05 s for P). Alternatives: 3 x 3 x 1 km pilot (8.5 M nodes);
1 x 1 x 0.5 km (160 M nodes, too slow). Evidence: TomoATT error vs spacing and memory table (Chen et al.
2025, doi:10.1016/j.cageo.2025.105995); 2 x 0.5 km in Chen et al. 2023 (doi:10.1093/gji/ggad093).

**Inversion grid.** Recommendation: 5 staggered grids, 0.30 x 0.40 deg then 0.20 x 0.25 deg, depth nodes
-3 ... 80 km as in 2.2, trapezoid widening x1.5 below 30 km. Alternatives: single 0.2 deg grid from the
start; 0.5 deg (Turkey/Myanmar). Evidence: five nodes per wavelength rule (Chen et al. 2023; Wang et al.
2024, doi:10.1785/0120240035); station spacing.

**Optimisation and regularisation.** Recommendation: gradient descent, 1 % max update, `Kdensity_coe` 0.3,
no Laplacian, stop on chi^2 <= 1.1 or < 1 % gain over 5 iterations, <= 20 iterations per run.
Alternatives: L-BFGS (experimental in the code); `Kdensity_coe` 0.5. Evidence: Tong 2021a
(doi:10.1029/2021JB021818); TomoATT examples; Chen et al. 2025.

**Vp/Vs coupling and relocation.** Recommendation: separate P and S runs with fixed locations,
external P+S relocation in each of up to 3 outer loops; test the S-P data correction of Bai et al. 2026.
Alternatives: TomoATT internal joint mode on P only; independent P and S runs with their own
relocation (Wang et al. 2024). Evidence: code inspection (one velocity field per run); Bai et al. 2026
(doi:10.1016/j.epsl.2026.119879).

**Data types.** Recommendation: ABS P and S throughout; CS differential (weight 0.5, azimuth window
30-60 deg, separation <= 60 km) added in stage 2; no CR in B2 (left to GrowClust in B5). Pick weights
(0.05/sigma)^2; residual weights P [0.5, 1.5], S [1, 3] s; distance taper 150-250 km; event box weights.
Alternatives: ABS only; ABS + CS + CR as in Zhang et al. 2025. Evidence: Tong et al. 2024
(doi:10.1093/gji/ggad416); Zhang et al. 2025 (doi:10.1007/s11430-024-1504-x).

**Anisotropy.** Recommendation: isotropic. Alternative: P azimuthal anisotropy test with leakage
checkerboards. Evidence: Wu et al. 2025 (doi:10.1016/j.epsl.2025.119362); Bai et al. 2026.

**OBS sediment (amends D-B006).** Recommendation: a priori per-OBS vertical sediment corrections for S
(and P) subtracted from the observed times; sources Rychert et al. 2018 Table S1, Ruan et al. 2014,
Bell et al. 2016, our own RF Ps delays, GlobSed as fallback; no sediment layer in the 1 km grid; three-way
sensitivity test. Alternatives: Track A iter1 S terms; no correction; sediment layer in the model.
Evidence: TomoATT cannot invert static terms for local absolute times (code); S delays 0.5-2 s
(Ruan et al. 2014, doi:10.1002/2014JB011162); Track A OBS S terms (ours).

**Resolution tests.** Recommendation: depth-dependent checkerboards (40/60/100 km, +-8 %, two phases),
6 spike targets, restoring test, per-pick noise, perturbed hypocentres with the full outer loop;
"resolved" = local correlation >= 0.7 and amplitude >= 0.5. Alternatives: single-size checkerboard with
0.05 s noise as in most ATT papers. Evidence: Chen et al. 2023; Wang et al. 2024; Wu et al. 2025.

**Model independence.** Recommendation: 1D-start inversion (1D first, as in Zhang et al. 2025), Delph 2018
start onshore, 5 x 70 % jackknife, no-OBS run; interpret only features stable to 2 % (Vp) / 3 % (Vs).
Alternative: Track A start only. Evidence: Wang et al. 2024 (three starts); Wu et al. 2025 (70 % subsets).

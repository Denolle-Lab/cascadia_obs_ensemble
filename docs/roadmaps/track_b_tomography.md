# Track B roadmap: joint Vp and Vs tomography with relocation

For a Claude Code session working on its own. Written 2026-11-22. The PI is Marine Denolle;
data choices need no co-author sign-off. Track A (relocation in an existing 3D model) runs in
parallel in another session and gives Track B its starting model and locations.

## Goal

A body-wave Vp, Vs and Vp/Vs model of the Cascadia margin, from the catalog v4 picks, with
resolution shown, plus the events relocated in it (catalog v6). The new part is offshore: the
Gorda plate, the Mendocino region, the Blanco transform and the offshore forearc, from the
2011–2015 ocean-bottom (OBS) data. There, surface-wave models are smooth and give Vs only. This is
probably a separate paper. Nothing else waits on it; when it is done, the focal mechanisms and
fault surfaces get recomputed in the new model.

Background: `data/catalog_v4/next_steps/catalog_v4_next_steps.html`, section 4 (on branch
`paper/v4-methods`).

## Data

- **Picks:** `data/catalog_v4/Cascadia_picks_catalog_v4.csv`, 1,276,597 P and 1,226,592 S on
  696 stations. 322 stations are on the seafloor and carry 0.99 M of the picks.
- **Events:** `data/catalog_v4/Cascadia_relocated_catalog_v4.csv`, 172,340 events: 122,306 with
  at least 8 picks and 44,034 with at least 15.
- **Stations:** `/wd1/mdenolle_data/picks_share/stations/stations_ver5_v6.csv`.
- **Starting model:** the Track A A1 model (CRESCENT CVM Gen0 Vs, Brocher Vp, young-ocean 1D
  offshore, earth_relief surface). See `workflow/08_relocation/README.md` on branch
  `reloc/track-a`; the files are under `/wd1/mdenolle_data/reloc_v5/model/`.
- **Comparison models (Vs only):** `data/tomography/CRESCENT_Gen0.nc` and
  `data/tomography/Delph2018.nc`, compared in `workflow/06_analysis/phase13_tomography.py`.
- **Slab and interfaces:** `data/slab2/`, and CASIE21 in `data/faults/crescent_cfm/`.

## Rules

- **Work in your own git worktree,** not in `/home/mdenolle/cascadia_obs_ensemble` itself, which
  another session shares. Create your branch there:
  `git worktree add -b tomo/track-b /home/mdenolle/wt_track_b origin/paper/v4-methods`
- **Push as you go:** commit and push your branch as steps finish. Open a PR only when the PI asks.
- **Where files go:** large files in `/wd1/mdenolle_data/tomo_v1/`; QC plots and small tables in
  `data/tomo_v1/`.
- **Python:** `ENV=/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default`, run with
  `LD_LIBRARY_PATH=$ENV/lib`. Compiled codes go in `/wd1/mdenolle_data/tomo_v1/src`; conda-forge
  compilers and MPI are available through `pixi exec --spec ...` if the system ones are too old.
- **No repicking.**
- **ComCat:** no ComCat focal mechanisms or locations as constraints; compare only.
- **Reports:** HTML files, not Artifacts. Prose follows the project's `plain-voice` skill.
- **Compute:** 128 cores, 250 GB RAM, no GPU, about 3.3 TB free on /wd1, shared with Track A,
  whose grids take about 240 GB. Keep track of disk use.

## Method choice (B0, 2–3 days, decide and write down)

- **First choice: TomoATT** (adjoint-state eikonal tomography; Chen et al. 2023, and the TomoATT
  papers and GitHub). It works in spherical coordinates, handles topography, bathymetry and
  seafloor stations, scales with MPI to millions of travel times, and inverts Vp, Vs and
  locations jointly.
- **Alternatives:** tomoDD (double-difference tomography; Zhang & Thurber), or FMTOMO / LOTOS if
  TomoATT will not build. Write in `data/tomo_v1/B0_choice.md` why the chosen code fits:
  topography, OBS, Vp/Vs, scaling, and whether it relocates.
- **Build** the code, then run its example to check it reproduces the published result.

## Steps

### B1. Data set (1 week)

1. **Select events:** at least 8 P+S picks, and decluster to keep about 30–40k events evenly in
   space (for example the best-recorded event per 2 × 2 × 2 km cell), so that Axial, Blanco and
   Mendocino don't dominate.
2. **Locations:** start from the v4 locations. When Track A A2 locations exist, switch to them,
   and use the A2 station terms as a starting point.
3. **Outliers:** remove picks with residuals beyond 4 MAD in the starting model.
4. **Weights** by pick probability (`pick_prob`) and phase.
5. **Report:**
   - ray coverage maps by depth slice;
   - picks per station;
   - share from OBS;
   - coverage by year (the OBS period 2011–2015 dominates offshore).

### B2. Resolution before inversion (1 week)

1. **Checkerboard tests** at 15, 25 and 40 km, with ±5% anomalies in Vp and Vs separately.
2. **Spike and restoring-resolution tests** for the key targets:
   - Gorda plate interior;
   - Mendocino slab edge;
   - Blanco segments;
   - forearc wedge under the shelf;
   - Axial.
3. **Trade-off tests** between Vp/Vs and source depth offshore: a synthetic sediment layer of
   0.5 km at Vs 0.3–0.6 km/s under the OBS, inverted without it, to see where it leaks.
4. Decide the grid, smoothing and damping from these tests, not from the real-data fit.

### B3. Inversion (3–4 weeks)

1. **Joint Vp, Vs and location inversion** from the A1 model. Run several smoothing and damping
   values and choose by L-curve and synthetic recovery.
2. **Station terms or a shallow layer** for the OBS sediment delays. Decide which and say why.
3. **Track per iteration:** residual RMS by phase and station type, and event shifts.
4. **Stability:**
   - run from a second starting model (Gen0 Vs with a constant Vp/Vs of 1.75, or the Track A
     model with a ±3% shift) and show the differences;
   - bootstrap of the events (5–10 runs) for a model standard deviation.

### B4. Products (1–2 weeks)

1. **Model** as NetCDF on a regular lon/lat/depth grid: Vp, Vs, Vp/Vs, dlnVp, dlnVs, ray density
   or DWS, checkerboard recovery, bootstrap standard deviation, and a mask of resolved nodes.
2. **Catalog v6:** the selected events relocated jointly, then all events relocated in the final
   model with NonLinLoc (reuse the Track A setup and grids code). Same columns as catalog v5.
3. **Figures:**
   - depth slices;
   - cross-sections through Mendocino, Gorda, Blanco and the Washington and Oregon forearc, with
     seismicity, Slab2 and CASIE21;
   - comparison with CRESCENT Gen0 and Delph et al. 2018, using `phase13_tomography.py`.
4. **Report:** HTML in `data/tomo_v1/`. A Zenodo draft is prepared, and published only when the
   PI says so.

### B5. Feedback (after B4)

- Tell the Track A session (or the PI) that the model is ready.
- Fault surfaces and focal mechanisms then get recomputed in it. The shift of the fault surfaces
  between the Track A and Track B locations is their velocity-model uncertainty, which v0 and v1
  do not include.

## Science questions to keep in view

- Vp/Vs of the Gorda plate interior: hydration of the deforming plate before subduction.
- The slab edge and the slab window south of the Mendocino triple junction.
- Low velocities along the Blanco transform and its step-overs, compared with the event-derived
  fault surfaces (Zenodo 10.5281/zenodo.24443891, v0).
- Forearc Vp/Vs along strike, against the locked/creeping coupling models (in `data/coupling/`).

## Limits to report honestly

- **Ray coverage:** event–station geometry offshore leaves the region between the ridge and the
  trench poorly sampled.
- **OBS sediment:** S delays at the OBS can map into shallow Vp/Vs and source depth.
- **Time span:** OBS coverage is 2011–2015 only.

## Timeline

B0 0.5 week, B1 1 week, B2 1 week, B3 3–4 weeks, B4 1–2 weeks: about 8–12 weeks. B1 and B2 can
start now on the v4 locations; switch to the Track A A2 locations when they arrive.

# Track A roadmap: relocation of catalog v4 in a 3D model

For a Claude Code session working on its own. Written 2026-11-22. The PI is Marine Denolle;
data choices need no co-author sign-off. Track B (tomography) runs in parallel in another session.

## Goal

Make catalog v5: every v4 event (172,340, 2010–2016) relocated in a 3D model, with real depths
(no event above the ground or seafloor), a full location uncertainty for each event, and a relative
relocation. Four things wait on it:
- the focal mechanisms, which need takeoff angles;
- the fault surfaces v1, which need locations and uncertainties;
- Track B, which needs its starting locations;
- the paper.

Why: in v4, 23.5% of events sit above the surface when depths are read below sea level, and
16.2% when read below the local surface. At Axial the median depth is −0.8 km, above the caldera
floor at about 1.5 km below sea level. And `spatial_sigma` (3–20 km) is not a usable relative
error: events scatter 0.6–1.8 km about planar structures. Background:
`data/catalog_v4/next_steps/catalog_v4_next_steps.html` (sections 1 and 4).

## Where things stand

A Track A session has already started on branch `reloc/track-a`. Read
`workflow/08_relocation/README.md` on that branch first: it sets the conventions (depth in km
below sea level, stations at true elevation, 250 km distance cap) and documents A1.

- **A1, done (commit 965d31b0):** 3D Vp/Vs model for NonLinLoc. Vs from CRESCENT CVM Gen0, Vp from
  Brocher (2005), 1D young oceanic crust where Gen0 is missing, GMT earth_relief 01m surface, and a
  station GTSRCE file. The NonLinLoc build and travel-time grids are under
  `/wd1/mdenolle_data/reloc_v5/`.
- **A2, A3, A4:** below.

## Rules

- **Work in your own git worktree,** not in `/home/mdenolle/cascadia_obs_ensemble` itself. Another
  session shares that directory, and a branch switch there moves everyone's work (on 2026-11-22 a
  fault-surface commit landed on `reloc/track-a` that way). For example:
  `git worktree add /home/mdenolle/wt_track_a reloc/track-a`
- **Push as you go:** commit and push to `reloc/track-a` as steps finish. Open a PR to `main` only
  when the PI asks.
- **Where files go:** large files in `/wd1/mdenolle_data/reloc_v5/`; QC plots and small tables in
  `data/reloc_v5/`.
- **Python:** `ENV=/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default`, run as
  `LD_LIBRARY_PATH=$ENV/lib PYTHONPATH=$PWD $ENV/bin/python ...`. The system libstdc++ is too old,
  so `LD_LIBRARY_PATH` is needed.
- **No repicking.** Use the existing picks in `data/catalog_v4/Cascadia_picks_catalog_v4.csv`
  (2,503,189 picks, columns `event_ID, raw_event_ID, pick_datetime, travel_time(s), station, phase,
  amplitude(mm), pick_prob`; the station is `STA.NET`). Stations are in
  `/wd1/mdenolle_data/picks_share/stations/stations_ver5_v6.csv`.
- **ComCat:** use it only to compare, never as ground truth, and use no ComCat focal mechanisms
  (they come from other catalogs' locations and velocity models).
- **Reports:** HTML files in the repo, not Artifacts. Prose follows the project's `plain-voice`
  skill.
- **Compute:** 128 cores, 250 GB RAM, no GPU, 3.3 TB free on /wd1. The travel-time grids with
  angles take about 240 GB; check free space before writing them.

## Steps

### A2. Absolute location with NonLinLoc (1–2 weeks)

1. **Travel-time grids** for all 696 stations, P and S, with angles (`GTMODE GRID3D ANGLES_YES`).
   The focal mechanisms will read takeoff angles from them. Run in parallel, about 100 jobs.
   Check a few grids against TauP or a 1D reference at 50, 150 and 250 km.
2. **NLLoc** on all events: EDT likelihood, oct-tree search. Write the full PDF summary for each
   event: expectation, covariance, 68% ellipsoid, and the scatter file for a subset.
3. **Station terms:** iterate static terms per station and phase (3–5 passes) until the median
   change is under 0.02 s. Then try source-specific terms (SSST) if residuals still depend on
   location. OBS S terms will be large (up to a few seconds, from sediment); that's expected.
   Report them by station type.
4. **Outlier picks:** drop picks with |residual| > max(1.0 s, 4 × station MAD) after the first
   pass, relocate, and record how many were dropped per station and phase.
5. **Checks** (HTML report `data/reloc_v5/a2_report.html`):
   - share of events above the surface (target: 0 by construction, and few events within 0.5 km
     of it);
   - depth distribution per region against v4;
   - Axial: seismicity should sit below the caldera floor, 0.5–2 km;
   - shift from v4 per region, horizontal and vertical;
   - RMS residuals before and after station terms;
   - location against ComCat for common events, as a comparison only;
   - how the 68% ellipsoid size depends on station coverage (OBS years 2011–2015 against other
     years);
   - the NC/BK coverage gap south of 42°N in 2010–2011 and 2016 (known; say how it affects the
     errors).

### A3. Relative relocation (1–2 weeks)

1. **Differential times:** catalog differential times from the picks (event pairs within 10 km,
   at least 8 shared phases). Waveform cross-correlation is optional; only add it if the PI asks,
   since the waveforms are on /wd1 but it adds weeks.
2. **GrowClust** (preferred, robust to outliers and gives bootstrap errors) on the A2 locations
   and the A1 model, or its 1D average if GrowClust needs 1D. Bootstrap with at least 100
   resamples for relative errors.
3. **Checks** (`data/reloc_v5/a3_report.html`):
   - fraction of events relocated;
   - relative errors by region;
   - sharpening of known structures: Blanco segments, Axial ring faults, Mendocino transform at
     depth;
   - no event above the surface;
   - comparison with published relocations where they exist (Blanco: Kuna 2019 and Ren et al.
     2023; Mendocino: Yoon & Shelly 2024; Axial: Waldhauser 2020), as a comparison only.

### A4. Catalog v5 release (3–5 days)

1. Write `Cascadia_relocated_catalog_v5.csv` with one row per event. Columns:
   - `event_id` (the v4 `ID`), origin time, lon, lat, depth_km (below sea level);
   - `x_err_km`, `y_err_km`, `z_err_km` and the 68% ellipsoid from NLLoc (absolute);
   - `rel_x_err_km`, `rel_y_err_km`, `rel_z_err_km` from the GrowClust bootstrap (blank when not
     relocated);
   - `n_p`, `n_s`, `rms_s`, `gap_deg`, `dmin_km`;
   - `reloc_type` (`nll` or `growclust`);
   - `surface_elev_km` under the epicentre.
2. Write a pick file with residuals and the takeoff angle and azimuth of every used pick. The
   focal mechanisms use this directly.
3. Write a README in the style of `data/faults/v0/README.md`: method, every column, settings,
   checks, limits.
4. **Zenodo:** keep everything private. Add the files to the existing private working-copy
   draft (deposition 23084158) with `utils/zenodo_picks_draft.py upload` and update its
   description (`describe`). Never publish, and never make a new record, unless the PI asks.
5. **Atlas:** in `workflow/06_analysis/build_v4_atlas.py`, `events()` reads the v4 file. Add a
   v5 option (keep v4 selectable) and say in the report how to rebuild. The atlas is published
   from the `gh-pages` branch; don't push there yourself, tell the PI.

## Handoffs (write them as soon as they exist)

| To | What | When |
|---|---|---|
| Track B | A1 model files and the A2 locations with station terms | after A2 |
| Focal mechanisms | pick file with takeoff angles, azimuths and residuals | after A2 (update after A3) |
| Fault surfaces v1 | catalog v5 with absolute and relative errors | after A3 |
| Paper | numbers for the methods (model, terms, errors, shifts) | after A4 |

Fault surfaces v1 are made by rerunning `workflow/07_faults/fault_surfaces_v0.py` (on branch
`paper/v4-methods`) on catalog v5. The script reads `depth`, `longitude`, `latitude`,
`spatial_sigma(m)`, `Num_p_picks` and `Num_s_picks`, so give v5 a reader or a column mapping.
Relative errors should replace `spatial_sigma` in the location Monte Carlo. That swap is the main
change between v0 and v1.

## Settings to vary later (record them, don't tune silently)

- Vp from Brocher against a constant Vp/Vs of 1.75;
- the young-ocean 1D model;
- the station elevation rule;
- the outlier threshold;
- the distance cap (150 against 250 km).

The relocation run that uses the Track B model (catalog v6) will show the velocity-model part of
the uncertainty.

## Timeline

A2 1–2 weeks, A3 1–2 weeks, A4 3–5 days: about 4–5 weeks in all.

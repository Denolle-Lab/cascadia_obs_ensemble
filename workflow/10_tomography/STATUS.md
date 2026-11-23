# Track B status: tomography (tomoDD) and relocation

One page that says where every step stands. Decisions are in `DECISIONS.md` (D-Bxxx), parameters in
`params.yaml`, details in `notes/`. Update this file whenever a step changes state.
Last update: 2026-11-22.

Naming: **v4-NLL3D** = catalog v4 relocated with NonLinLoc in the 3D starting model (formerly
"Track A", D-B024). Starting model and locations: D-B004, D-B005.

## Done

| Step | What | Where | Decisions | Commit |
|---|---|---|---|---|
| A | v4-NLL3D relocation of all 172,340 events, tiers A-D/U | workflow/08_relocation (branch reloc/track-a) | D-B004, D-B005 | reloc/track-a |
| B1 | Event and pick set: 24,621 events (23,319 main + 1,302 ridge), 703,559 picks, pick QC, OBS S statics | b1_select_events.py, notes/dataset.md | D-B003, D-B008..013, D-B020, D-B023 | 07a2525e |
| B1 OBS check | OBS S picks are true S; negative S terms are path misfit | b1_obs_waveform_check.py, notes/obs_check.md | D-B020 | 003339bc |
| B1b | Balanced dataset of well-recorded stations: 24,249 events, 689,602 picks, 475 stations; station, path and region weights; ridge regions with softer rules | b1b_balance.py, notes/balanced.md | D-B022, D-B023 | 07a2525e |
| Code | tomoDD-SE (USTC 2020 package) built, example reproduced, efficiency audit, Mendocino benchmark (9.9 min, 2.5 GB) | /wd1/mdenolle_data/tomo_v1/src (not in git: licence), notes/tomodd_build.md | D-B021, D-B026 (proposed) | 6cd89dd4 |
| Atlas | v4-NLL3D (preliminary) catalog switch and tier colours, public | branch atlas/v4-nll3d merged into paper/v4-methods (57447788); gh-pages 2818a585 | D-B024 | 57447788 |

## Waiting

| Item | Waiting on | Blocks |
|---|---|---|
| tomoDD version and permission to modify (OpenMP, sparse matrix build, units bug in the event-separation weighting) | PI email to Haijiang Zhang (zhang11@ustc.edu.cn) | Source changes only; runs with the unmodified code can go ahead |
| D-B026 adoption (copy, compiler, flags, array sizes) | PI | Production runs |

## Next, in order

1. **Input builder (B2a).** From B1b to tomoDD files for one region: absolute times with weight
   w_bal x pick-quality weight rescaled into (0, 3]; catalog differential times; numeric station codes;
   OBS S statics applied to the picks (D-B020); events and stations inside the region. Script
   `b2_tomodd_inputs.py`; first region Mendocino-Gorda (D-B025).
2. **Regions (D-B025).** Boundaries and overlaps of ~4 sub-regions (Mendocino-Gorda, Blanco, Oregon
   offshore forearc, Washington-Puget; ridges with Blanco or alone). Size set by memory and run time, not
   curvature (D-B026).
3. **Grid and ray-tracer accuracy test.** Velocity grid with top edge at or above -4 km; travel-time
   accuracy of the pseudo-bending tracer (10 km minimum segment, hard-coded) against a fine reference
   (D-B015 analogue for tomoDD); one bounds-checked run per new grid.
4. **Bayesian ensemble design (D-B027, note notes/bayes_design.md).** Randomize-then-optimize over the
   starting model (published models + smooth random perturbations), station statics (priors per station
   class from the v4-NLL3D terms), pick noise per class and damping/smoothing; ~100 members per region
   as parallel serial jobs; compiler, flags and machine pinned (D-B026: last-bit differences move
   hypocentres ~0.2 km).
5. **Synthetic calibration.** Known model with station delays through the full ensemble; credible
   intervals must cover the truth at the stated rate; checkerboard and spike tests as figures (D-B018).
   Compare balanced vs unweighted B1b here (notes/balanced.md).
6. **Real-data ensemble, region by region,** then overlap comparison and merge.
7. **B4.** Relocate all 172,340 events in the final model (NonLinLoc, new station terms, tiers),
   with location uncertainty from the ensemble.
8. **B5.** GrowClust on the B4 locations (D-B002).
9. **Products.** Model NetCDF with mean, std and resolution mask; catalog; report; private Zenodo draft
   23084158 only (never publish without the PI).

## Open questions for the PI

* Ridges weight held at the clip of 3 (7.5% of the misfit weight instead of 12.5%): keep? (notes/balanced.md)
* Far-station picks (D-B012) re-admitted after a forward check in the starting model: still wanted with tomoDD?

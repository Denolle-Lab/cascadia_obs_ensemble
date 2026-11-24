# Track B status: tomography (tomoDD) and relocation

One page that says where every step stands. Decisions are in `DECISIONS.md` (D-Bxxx), parameters in
`params.yaml`, details in `notes/`. Update this file whenever a step changes state.
Last update: 2026-11-23 16:15 (handoff section below).

Naming: **v4-NLL3D** = catalog v4 relocated with NonLinLoc in the 3D starting model (formerly
"Track A", D-B024). Starting model and locations: D-B004, D-B005.

## Handoff 2026-11-23 08:10 (read first)

**Where the work is.** Worktree `~/wt_tomo_res`, branch `tomo/tiles` = PR #33, stacked on `tomo/res-tests-mg_v2` = PR #32
(Mendocino-Gorda v2 resolution tests), stacked on `tomo/track-b`. PR #32 also holds a snapshot of the code that was
uncommitted in the main checkout (the main checkout has not changed since; it still sits on `tomo/track-b` with that
work uncommitted). The main checkout is not used for this work.

**What was decided today.** D-B033 tiles (blanco, oregon, washington, ridges_north beside Mendocino-Gorda); D-B034
FORMAT-fixed tomoDD builds (missing comma at tomoDD-SE.f line 2324 stopped some runs); D-B035 offshore forearc kept
(soft selection rule, soft-bar OBS keep all picks; datasets `b1_dataset_c`, `b1b_balanced_c`; 28,986 events).
mendocino_gorda_v2 / v2s and the ensemble work stay on the old dataset; `mendocino_gorda_c` is Mendocino-Gorda on
the new one.

**Running (detached, survive logout; not a reboot).** `b2_res_pipeline.sh <tile>_v2` for mendocino_gorda_c, blanco,
oregon, washington at damping 1000 (53 inversions each: 5 real-data scan runs + checkerboards + lattice spikes).
At 08:10: blanco 40/53, oregon 32/53, mendocino_gorda_c 8/53, washington 3/53. Each tile commits its results to
`tomo/tiles` and comments on PR #33 when done. Progress: `res_tests/queue/runner.log` and `pipeline.log` under
`/wd1/mdenolle_data/tomo_v1/b2_tomodd/<tile>_v2/`. If a run stopped (reboot, kill), resume with the same command
from `~/wt_tomo_res`; finished pieces are skipped:

    setsid nohup bash workflow/10_tomography/b2_res_pipeline.sh blanco_v2 \
        >> /wd1/mdenolle_data/tomo_v1/b2_tomodd/blanco_v2/pipeline.log 2>&1 < /dev/null &

**Update 09:05.** blanco and oregon finished at damping 1000 (on PR #33): over-damped like ridges (real-data condition
number at s20: 11 and 14, against ~60 for Mendocino-Gorda v2; no lattice spike recovered >= 50 %). PI asked for
damping 200 and 300 on both: `DAMP=200` and `DAMP=300` pipelines launched 09:05 (51 runs each, results in
`data/tomo_v1/b2_res_tests/<tile>/d200|d300/`). washington and mendocino_gorda_c still running at damping 1000.
*Correction (12:30):* the 11 and 14 are last-iteration condition numbers; the Mendocino-Gorda ~60 is first-iteration
(its last-iteration value is 19). Like for like, Blanco and Oregon at damping 1000 have 35 and 39, and damping 300
gives 170 and 200 (notes/merge_design.md section 2).

**Update 12:30.** Done and on PR #33: blanco and oregon at damping 200 and 300, mendocino_gorda_c at 1000.
washington at 1000 still running (real-data scan done: first-iteration CND 68 at damping 1000, like Mendocino-Gorda).
PI chose damping 200 for ridges_north (D-B043). Merge design and tile-vs-tile tests proposed
(notes/merge_design.md, P-M1..P-M8; `b2_tile_overlap.py`, results in data/tomo_v1/b2_merge/overlap/): no overlap
is resolved by both tiles; mixing dampings does not make a worse seam than either damping alone; the same events
relocated in two tiles differ by 0.2-1.0 km horizontally (median).

**Update 14:15.** washington_v2 at damping 1000 finished 12:05 (on PR #33): the best-resolved tile so far (18 P
spikes recover >= 50 %, 14 of them at DWS > 1,000; checkerboards resolve about 30 % of the box at 15-18 km for
40-60 km cells; Vs as well as Vp). PI decisions: M0 coarse whole-margin start (D-B044) and one small damping for
all tiles, 200 (D-B045). Implementing D-B045 showed that pick weights were scaled per tile (Oregon and Washington
picks weighed 3 times Blanco's, Mendocino-Gorda 1.8 times), which changes what one damping number does in tomoDD;
from now on all builds use one weight scale for the whole dataset. Launched 14:03, detached:
`b2_t6_m0.sh` = T6 joint blanco + oregon against the stitched pair at damping 200, and M0 (margin_m0_v2, 6,943
declustered events, 518 stations, new 560-station tomoDD build) at damping 200 and 1000. Log
`/wd1/mdenolle_data/tomo_v1/b2_tomodd/t6_m0.log`; it commits to `tomo/tiles` and comments on PR #33 when done.
Resume after a stop with the same command (finished stages are skipped):

    setsid nohup bash workflow/10_tomography/b2_t6_m0.sh \
        >> /wd1/mdenolle_data/tomo_v1/b2_tomodd/t6_m0.log 2>&1 < /dev/null &

Next after review of M0: rebuild every tile from the M0 model (builder option start_model, written) and rerun the
damping-200 run and resolution tests per tile.

**Update 16:15.** T6 and M0 done (data/tomo_v1/b2_merge/t6/, m0/; notes/merge_design.md section 6 item 7 and the
M0 paragraph after section 7). T6 passes in the tile interiors: joint and stitched differ by 1.7-2.0 % rms in Vp
(correlation 0.86-0.94), the same as one tile moved from damping 200 to 300; the joint model is about 15 % stronger
(more data rows, so damping weighs less); in the overlap, which neither tile resolves, the stitched model has half
the joint amplitude; overlap events differ by about 1 km between joint and tile runs. M0 at smoothing 20 was
node-scale noise (7 % rms Vp); a smoothing scan (20-1920, damping 200) has a flat weighted fit up to 480 and a
coherent model there (slow ridge crust at 7-15 km, fast 40 km under Washington). Proposed P-M11: M0 = d200_s480,
then tiles restart from it with a smoothing scan each (builder option `start_model` written, bc73862d). Waiting
for the PI before the tile reruns (about one day of the machine).

Options: `DAMP=<d>` (tests and real-data runs at damping d; results in `<tile>/d<d>/`), `SPIKES_ONLY=1` (spikes at
the event clusters with the DWS mask, `spikec_*`), `MAXJ` (parallel inversions, default 10).

**ridges_north result (done, all on PR #33).** Lattice checkerboards and spikes: nothing resolved at any damping
(1000, 300, 200, 100). Spikes at the event clusters with DWS >= 100 (52 of 105 P spikes sampled): median recovery
22 % (damping 100), 21 % (200), 19 % (300), 10 % (1000); only 3 spikes >= 50 %, all at 12 km between northern Blanco
and Axial (44.4-45.4 N, 130.0-130.3 W, DWS 1,165-7,203, smearing about one node). Real-data condition number at
s20, last iteration: 87 (d100), 33 (d200), 21 (d300), 6 (d1000); first iteration: 290, 119, 69, 15 (Mendocino-Gorda
v2 chose 1000 at first-iteration CND ~60, last-iteration 19).

**Open, for the PI.**
1. Resolved 2026-11-23 (D-B043): ridges_north damping 200. The D-B038 exception volume is not recovered at 200 and
   stays on the damping-100 model unless dropped.
2. Resolved 2026-11-23 (D-B045): damping 200 and smoothing 20 in every tile and in M0, pick weights on one scale
   for the whole dataset.
3. Resolved 2026-11-23 (D-B038): ridges_north for relocation only, except the 12 km volume between northern Blanco
   and Axial where cluster spikes recover 50-64 % at damping 100.
4. CASIE21 as a constraint on the shallow offshore Washington/Oregon starting model or prior (D-B035, not started).
5. Merge order: PR #32, then PR #33, into `tomo/track-b`.
6. Science framing (notes/science_framing.md, 2026-11-23): novelty wording (plate-wide local-earthquake Pn
   tomography already exists, VanderBeek & Toomey 2019), questions that join sources and structure, steps after
   the tomography. Adopted 2026-11-23 as D-B036..D-B042; references merged to main (PR #34, cbaf5efc).
7. Merge design and tile-vs-tile tests (notes/merge_design.md, 2026-11-23, revised after a literature review):
   P-M1..P-M10 proposed. Main choices: coarse whole-margin inversion as common start (P-M2, as Lin et al. 2010),
   resolution-window weights without inverse variance (P-M4), joint-vs-stitched test on blanco + oregon (T6).
   P-M1 and P-M2 adopted (D-B045, D-B044); T6 and M0 done (update 16:15). Open: P-M3..P-M11 (P-M11 = M0
   smoothing 480 and a smoothing scan per tile).

## Done

| Step | What | Where | Decisions | Commit |
|---|---|---|---|---|
| A | v4-NLL3D relocation of all 172,340 events, tiers A-D/U | workflow/08_relocation (branch reloc/track-a) | D-B004, D-B005 | reloc/track-a |
| B1 | Event and pick set: 24,621 events (23,319 main + 1,302 ridge), 703,559 picks, pick QC, OBS S statics | b1_select_events.py, notes/dataset.md | D-B003, D-B008..013, D-B020, D-B023 | 07a2525e |
| B1 OBS check | OBS S picks are true S; negative S terms are path misfit | b1_obs_waveform_check.py, notes/obs_check.md | D-B020 | 003339bc |
| B1b | Balanced dataset of well-recorded stations: 24,249 events, 689,602 picks, 475 stations; station, path and region weights; ridge regions with softer rules | b1b_balance.py, notes/balanced.md | D-B022, D-B023 | 07a2525e |
| Code | tomoDD-SE (USTC 2020 package) built, example reproduced, efficiency audit, Mendocino benchmark (9.9 min, 2.5 GB) | /wd1/mdenolle_data/tomo_v1/src (not in git: licence), notes/tomodd_build.md | D-B021, D-B026 | 6cd89dd4 |
| B2 smoothing check | Spike test of smoothing vs uneven node spacing, Mendocino-Gorda grid, production settings: broadening <= 1.1 at 12-34 km where resolved; upper crust (< 12 km) mostly unresolved at node scale (coverage-limited); syn binary rebuilt (first copy had MAXEVE 1650) | b2_smooth_spike_test.py, b2_smooth_spike_analyse.py, notes/smoothing_test.md | D-B028 (proposed) | uncommitted |
| Atlas | v4-NLL3D (preliminary) catalog switch and tier colours, public | branch atlas/v4-nll3d merged into paper/v4-methods (57447788); gh-pages 2818a585 | D-B024 | 57447788 |

## Waiting

| Item | Waiting on | Blocks |
|---|---|---|
| tomoDD version and permission to modify (OpenMP, sparse matrix build, units bug in the event-separation weighting) | PI email to Haijiang Zhang (zhang11@ustc.edu.cn) | Source changes only; runs with the unmodified code can go ahead |

## Next, in order

0. **Forward fixes per region (D-B028, D-B031).** For each region: b2_fwd_p1.py (P-1 reference times),
   b2_fwd_statics.py (P-6 OBS S statics), then the builder with the `forward` block (P-2, P-4, nearest-column MOD).
   Done: Mendocino-Gorda (mendocino_gorda_v2).
1. **Input builder (B2a).** From B1b to tomoDD files for one region: absolute times with weight
   w_bal x pick-quality weight rescaled into (0, 3]; catalog differential times; numeric station codes;
   OBS S statics applied to the picks (D-B020); events and stations inside the region. Script
   `b2_tomodd_inputs.py`; first region Mendocino-Gorda (D-B025).
2. **Regions (D-B025).** Boundaries and overlaps of ~4 sub-regions (Mendocino-Gorda, Blanco, Oregon
   offshore forearc, Washington-Puget; ridges with Blanco or alone). Size set by memory and run time, not
   curvature (D-B026).
3. **Grid and ray-tracer accuracy test.** Velocity grid with top edge at or above -4 km (bounds check check01 on Mendocino-Gorda 55x49x23: clean, 4.9 min, 2.6 GB, 2026-11-22); travel-time
   accuracy of the pseudo-bending tracer (10 km minimum segment, hard-coded) against a fine reference
   (D-B015 analogue for tomoDD); one bounds-checked run per new grid.
4. **Bayesian ensemble design (D-B030, note notes/bayes_design.md).** Randomize-then-optimize over the
   starting model (published models + smooth random perturbations), station statics (priors per station
   class from the v4-NLL3D terms), pick noise per class and damping/smoothing; ~100 members per region
   as parallel serial jobs; compiler, flags and machine pinned (D-B026: last-bit differences move
   hypocentres ~0.2 km).
5. **Synthetic calibration.** Known model with station delays through the full ensemble; credible
   intervals must cover the truth at the stated rate; checkerboard and spike tests as figures (D-B018).
   Compare balanced vs unweighted B1b here (notes/balanced.md).
6. **Real-data ensemble, region by region,** then overlap comparison and merge (notes/merge_design.md: one
   reference model, common grid, taper x resolution x 1/variance weights, member-by-member merge, tests T1-T5).
7. **B4.** Relocate all 172,340 events in the final model (NonLinLoc, new station terms, tiers),
   with location uncertainty from the ensemble.
8. **B5.** GrowClust on the B4 locations (D-B002).
9. **Magnitudes (PI request 2026-11-22).** Recompute the Route A local magnitudes (ML; workflow/05_magnitude,
   scale/run_magnitude_v4.sh, data/magnitude_v4) with the B4/B5 hypocentres: hypocentral distances change, so
   ML changes. Compare with the v4 magnitudes (offset by region, ComCat anchors) and update the catalog columns,
   Mc maps and atlas. If v4-NLL3D is rerun after D-B028 P-7, decide whether an interim recomputation is needed.
10. **Products.** Model NetCDF with mean, std and resolution mask; catalog; report; private Zenodo draft
   23084158 only (never publish without the PI).

## Open questions for the PI

* Resolved: depth nodes kept, smoothing spike test adopted as D-B029 (PI, 2026-11-22; amends D-B028 P-3).

* Ridges weight held at the clip of 3 (7.5% of the misfit weight instead of 12.5%): keep? (notes/balanced.md)
* Far-station picks (D-B012) re-admitted after a forward check in the starting model: still wanted with tomoDD?

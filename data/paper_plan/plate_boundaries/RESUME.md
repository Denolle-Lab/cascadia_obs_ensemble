# Plate-boundary paper plan (D-B107, D-B109, D-B111): resume notes

Worktree /home/mdenolle/wt_pbplan, branch analysis/plate-boundaries. Code workflow/11_paper_plan/, small tables here,
page and large files /wd1/mdenolle_data/paper_plan/. Python: LD_LIBRARY_PATH=/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/lib
/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/bin/python, nice 10, at most 6 processes.

## Status log
- 2026-11-26 16:32 PST: first build on catalog v6.0 (cecab0a5); page plate_boundaries_plan.html.
- Fri Nov 27 09:38:33 PST 2026: coordinator (D-B111): Vp/Vs spread raw (factor 1.0, not 1.5); circularity control on the
  v0.4.2 synthetic ensemble required before any contrast is interpreted. Rule below written and committed before any
  synthetic contrast is computed.

- Fri Nov 27 09:42:27 PST 2026: scored (pb_circularity.py, rule unchanged): 105 contrasts testable; 39 pass 2 sigma_ens
  (Vp/Vs raw); 36 survive the rule; dropped by S95: GS Vp/Vs near-far, MSH Vp along, MSH Vp/Vs across. Spurious sd ~
  raw member sigma (median ratio 0.97); S95 median 0.90 of 2 sigma_ens; noise-floor clone sd <= 0.0024. Null truth not
  possible with the existing synthetic set (one truth carrying the v0.4.2 contrasts); cost estimate ~2.1
  process-hours per member. Table circularity_contrasts.csv; page section 3.
- Fri Nov 27 10:11:44 PST 2026: task A (D-B112) done. Merged origin/pipeline/end-to-end (f21e87c6, PB_* variables). Ran every pb_*
  step on catalog v6.1 with workflow/11_paper_plan/run_v61.sh -> /wd1/mdenolle_data/paper_plan/v6.1/ (page
  plate_boundaries_plan.html, figures, tables/; small tables copied to data/paper_plan/plate_boundaries_v6.1/). All 7
  result tables byte-identical to the pipeline scratch run (/wd1/mdenolle_data/pipeline_smoke/v6.1_pb). 105 contrasts,
  32 >= 2 sigma_ens, 31 survive the D-B111 rule. Mount St Helens swarm share 38 -> 28 %: not the analog flags
  (v6.0 without them 38.4 %); family 144 retyped mainshock-aftershock, 13 swarm events to tier D, two 2014-15
  families split (pb_swarm_change.py, swarm_change_MSH_v60_v61.csv). pb_page.py made catalog-aware (v6.0 defaults
  unchanged in substance).
- 2026-11-27 10:13:48 PST: null-truth rule pre-registered (commit 6d45ad98, section below).
- 2026-11-27 10:14-10:16: pb_structure.py saves segment_node_dist.npz (tables unchanged); pb_null_truth.py built the
  null truth in /wd1/mdenolle_data/tomo_v1/ensemble_v0.4.2_null (prior/ copied from ensemble_v0.4.2, read only there;
  109,661 rock nodes set to zone layer means; median |dVp| 0.23 km/s, p95 0.85) and its synthetic data (5 x
  tomoDD-SE_syn, frozen binaries; synthetic minus real MAD 0.44-0.65 s, calibration truth 0.35-0.52 s).
- 2026-11-27 10:17:03: null members s000-s019 launched detached (pb_null_run.py: b2_ens_run scheduler, 6 tomoDD
  processes, nice 10, MemAvailable throttle 40 GB; draws identical to the calibration members of the same name,
  checked on s000). Expected about 7 h (2.1 process-hours per member). When all have merged, pb_null_run.py runs
  pb_null_score.py on v6.1 (writes /wd1/mdenolle_data/paper_plan/v6.1/tables/null_truth_contrasts.csv and
  figures/fig_null.png; no commit).
- Fri Nov 27 10:19:31 PST 2026: code committed; v6.1 page shows "null truth not yet scored".
- 2026-11-27 10:31:52: first null M0 stages done (s000, s002, s005: ok, 14.8 min; M0 residuals close to the
  calibration member s000); tiles started. Expected end about 17:00-17:30 PST.
- Fri Nov 27 13:26:13 PST 2026: D-B113 (PI 13:13): southern Gorda Ridge seismicity was in GP because PB2002 lies 10-20 km west of
  the axial valley. pb_axes.py re-picks the spreading axes and the Mendocino transform from the D-B068 merged surface
  (GMRT where available); a section's PB2002 trace is replaced only where its 15 km buffer misses >= 10 % of the
  nearby tier A-C earthquakes and a bathymetric candidate keeps >= 10 points more within 15 km (median distance not
  larger). Chosen: axial valley for the southern Gorda Ridge (OSR00, OSR01: 40.45-41.6 N), axial high for JdF 46.4-47.7 N
  (OSR05); PB2002 elsewhere, Mendocino transform included (base-of-slope pick 71 % vs 66 % within 15 km). Buffer 15 km.
  plate_axes_v1.csv, plate_axes_v1_evaluation.csv here; pb_common PB_AXES (gmrt default, pb2002 = old). v6.1 rerun
  (run_v61.sh, now with pb_axes and pb_axes_compare): 1,280 tier A-C earthquakes move (GP->GR 1,046, JP->JR 228,
  MFZ->GR 3, GR->GP 2, EXR->SOV 1); 33 contrasts >= 2 sigma, 32 survive D-B111. Old v6.1 tables kept in
  /wd1/mdenolle_data/paper_plan/v6.1/tables_pb2002_axes; changes in axes_changed_metrics.csv, axes_changed_contrasts.csv.
  The v6.0 tables in data/paper_plan/plate_boundaries/ were built with PB2002: rebuild them with PB_AXES=pb2002.
- Null truth and D-B113: the null zones were built on the PB2002 segments; reconstructed exactly (null_zones.npz in the
  null root's synth/truth/: 270 zone-layers constant, 109,661 nodes). The null members are scored on the NEW node sets
  (pb_null_score.py lists the zones of each contrast). Check (null_zone_check.csv): with old or new node sets only about
  half of a contrast's nodes lie in its own null zone (median 0.53 / 0.52) and the null truth keeps zone-boundary
  contrasts (median |D0_truth| 0.034 / 0.037 km/s); the rule (recovered - truth) still measures recovery error, but only
  single-zone contrasts are true null tests. D-B113 is material for GR only: its node sets reach into the old GP zone
  (own-zone share 0.59-0.71 -> 0.45-0.56; truth contrast Vp along -0.07 -> -0.29 km/s); GP, MFZ, JR barely change.
- 2026-11-27 15:24-15:27: null members s000-s019 all merged (0 failed); pb_null_score.py auto-scored them on the
  D-B113 (gmrt) node sets: 105 testable, 32 D-B111 survivors, 31 claimed. Kept as
  /wd1/mdenolle_data/paper_plan/v6.1/tables/null_truth_contrasts_gmrt.csv (and tables_gmrt_axes/, figures_gmrt_axes/,
  contrast_nodes_gmrt.npz, events_segments_v61_gmrt.parquet, plate_boundaries_plan_gmrt_axes.html).
- Fri Nov 27 19:24:42 PST 2026: D-B117 (PI 19:09): fast-forwarded to origin/analysis/plate-boundaries-v2 (ece2a586; PB_AXES default v2 =
  adopted boundaries v2). Full v6.1 chain rerun on v2 (run_v61.sh, now also pb_null_score, pb_null_compare,
  pb_null_zones): 727 tier A-C earthquakes change segment against gmrt (GR->MFZ 344, GP->MFZ 186, JP->JR 74, GP->GR 67,
  JR->JP 35, ...); 30 contrasts >= 2 sigma_ens, 29 survive D-B111; null truth (20 members, v2 node sets): 28 claimed
  (gmrt 31; lost GP Vs across, GR Vp across, GR Vp along, each now failing the D-B111 rule; none gained). Headline
  metrics outside the previous 95 % interval: GP bsf_p90 only (27.18 -> 25.43 km). Tables copied to
  data/paper_plan/plate_boundaries_v6.1/.
- To resume: log /wd1/mdenolle_data/tomo_v1/ensemble_v0.4.2_null/run.log; status: B2_ENS_CONFIG=v0.4.2
  B2_ENS_ROOT=/wd1/mdenolle_data/tomo_v1/ensemble_v0.4.2_null python workflow/10_tomography/b2_ens_run.py status;
  relaunch (resumes) with the pb_null_run.py command in its docstring. After scoring: bash workflow/11_paper_plan/run_v61.sh
  is NOT needed again; rebuild the page only (pb_page.py with the run_v61.sh environment), copy tables/null_truth_contrasts.csv
  to data/paper_plan/plate_boundaries_v6.1/, commit, push.

## Pre-registered rule: circularity control of the structure contrasts (written Fri Nov 27 09:38:33 PST 2026, before scoring)

Contrasts: every contrast of pb_structure.py, i.e. segment x variable (Vp, Vs, Vp/Vs) x type (near-far, across,
along), on the same node sets (v0.4.2 masks: rock and (checkerboard or DWS >= 100), P / S without Vs-reverted nodes /
both; distances to the segment's fault surfaces and v6.0 seismicity), depth-matched as in pb_structure.depth_matched.

For each contrast k:
1. D_real = the contrast in the deterministic model v0.4.2 (product values, as in structure_contrasts.csv).
2. sigma_ens = std of the contrast over the 100 real ensemble members (v0.4.2+ens.100) x factor: Vp 1.50, Vs 1.35,
   Vp/Vs 1.00 (D-B099, D-B111).
3. Synthetic (ensemble_v0.4.2/synth): D_truth = the contrast in the synthetic truth (synth/truth/truth_model.npz,
   v0.4.2 tomography x (1 + f)); D_rec,i = the contrast in synthetic member i (s000-s099, merged.npz vp and vs before
   the D-B073 node rule, like the truth); spurious s_i = D_rec,i - D_truth; S95 = 95th percentile of |s_i| over the
   members.
4. **Rule: contrast k counts (survives) only if |D_real| >= 2 sigma_ens AND |D_real| >= S95.** If fewer than 50
   synthetic members or the truth give no value on those nodes, k is "not testable" and does not survive.
5. Reported, not part of the rule: mean spurious contrast (bias) and whether it has the sign of D_real; recovery ratio
   mean(D_rec) / D_truth; spread of the noise-floor clones nf0-nf4 with s000; the rule with sigma_ens alone (old) for
   comparison.
6. Null truth: the synthetic set has one truth that carries the v0.4.2 contrasts. A truth without across-fault or
   along-strike contrast needs new synthetic inversions; if the existing members do not allow it, this is stated, not
   approximated.
7. The rule is not changed after the synthetic results are seen; any other threshold is reported as a variant only.

## Pre-registered rule: null-truth control of the structure contrasts (D-B112; written Fri Nov 27 10:13:48 PST 2026, before the null truth is built or scored)

Null truth T0 (root /wd1/mdenolle_data/tomo_v1/ensemble_v0.4.2_null; never ensemble_v0.4.2): the calibration truth
(v0.4.2 tomography x (1 + f), the same random-field draw, truth_seed 100000) with, for each of the 15 segments that have
contrasts on catalog v6.1 (resolved or marginal: JR, GR, BL, MFZ, GP, MTJ, GS, FA, SIF, IS, PU, MSH, CAS, OC, NAPA),
every rock node (v0.4.2 medium = 0) within 45 km (3-D, the same reference sets as the contrasts: the segment's fault
surfaces and seismicity) set to the mean of those nodes in its depth layer, Vp and Vs separately; a node within 45 km
of several segments belongs to the nearest one; then the start floor Vs >= Vp / 1.65 as in the calibration truth.
Truth hypocentres, statics, forward offsets and pick noise: the same seed as the calibration truth. Synthetic data with
the frozen tomoDD-SE_syn binaries through b2_ens_synth.py; members s000-s019 (the same member seeds as the calibration
members s000-s019) through M0, the four tiles and the merge (b2_ens_run.py scheduler, at most 6 tomoDD processes, nice
10, MemAvailable kept above 40 GB).

For each contrast k (v6.1 node sets, contrast_nodes.npz of /wd1/mdenolle_data/paper_plan/v6.1; depth-matched as before):
1. D0_truth = the contrast in T0 (zero where both node sets lie in one zone; reported for every k);
   D0_rec,i = the contrast in null member i; null spurious n_i = D0_rec,i - D0_truth; N95 = 95th percentile
   (numpy, linear) of |n_i| over the null members.
2. **Rule: a real v6.1 contrast is claimed only if |D_real| >= 2 sigma_ens AND |D_real| >= S95 (D-B111 rule) AND
   |D_real| >= N95.** Fewer than 15 null members with a value, or no D0_truth: not testable, not claimed.
3. Reported, not part of the rule: mean null spurious contrast (bias) and whether it has the sign of D_real; the variant
   with the 95th percentile of |D0_rec,i| itself; the paired difference null minus calibration member for the same
   member seed (s000-s019).
4. The rule is not changed after the null results are seen; any other threshold is a reported variant only.
- Fri Nov 27 14:32:23 PST 2026: D-B116 plate boundaries v2 (branch analysis/plate-boundaries-v2, worktree /home/mdenolle/wt_pbv2): proposal
  from the v6.1 seismicity (tier A-B, h_max <= 3 km) and the bathymetry, straight segments by BIC (rules pre-registered in
  data/paper_plan/plate_boundaries_v2/RULES.md, post-hoc A1-A5 flagged). Run: bash workflow/11_paper_plan/run_pbv2.sh.
  Outputs /wd1/mdenolle_data/paper_plan/plate_boundaries_v2/ (proposed_boundaries_v1.geojson, report_boundaries.html,
  figures/). Not adopted: PB_AXES=v2 switches pb_common to it (default stays gmrt); awaiting PI review.
- Fri Nov 27 19:19:24 PST 2026: D-B117 (PI 19:09): v2 ADOPTED with two changes: JdF Axial-Endeavour on its bathymetric axis (A6: Endeavour
  from MAPRIDGES), PB2002 triple junction kept (A7: Mendocino trace joined by an 8.4 km connector). pbv2_adopt.py ->
  /wd1/mdenolle_data/paper_plan/plate_boundaries_v2/adopted_boundaries_v2.geojson, plate_axes_v2_adopted.csv;
  pb_common PB_AXES default is now v2 (adopted); v2proposal, gmrt, pb2002 selectable. 727 of 135,433 v6.1 tier A-C
  earthquakes change segment against axes v1. Downstream (run_v61.sh, null scoring, Q1, Q2, figures) NOT rerun: owners.

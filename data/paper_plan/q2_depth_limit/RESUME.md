# Q2 (D-B115): what limits the depth of earthquakes across the offshore plate system - temperature or composition?

Worktree /home/mdenolle/wt_q2, branch science/q2-depth-limit (origin/main + origin/analysis/plate-boundaries through
ded23ee2, which carries the D-B113 axes). Code workflow/11_paper_plan/q2_*.py; small tables here; report, figures and
large tables /wd1/mdenolle_data/paper_plan/q2_depth_limit/. Python: LD_LIBRARY_PATH=/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/lib
/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/bin/python; nice 10; at most 6 processes; MemAvailable > 40 GB
(the D-B112 null-truth tomoDD run uses 6 processes).

## Status log
- 2026-11-27 13:27:22 PST: pre-registration below written, before any seismicity depth is compared with any isotherm
  or velocity level. Seen before writing it: counts of events per segment, tier and depth_quality, the sig_z
  distribution of tier A-B events, and the plate-boundary results already in the brief (Gorda slab D90 vs 7.5 km/s
  rho 0.53, p 0.001, 67 columns; Blanco rho 0.12; none at MFZ; Gorda plate p90 about 26 km bsf).
  Segment table at that time: /wd1/mdenolle_data/paper_plan/v6.1/events_segments_v61.parquet (md5 76542c2e,
  written 13:23 PST, D-B113 axes: GR 5,028, GP 5,832 events).
- 2026-11-27 13:28:16 PST: coordinator: the D-B113 segments are final (analysis/plate-boundaries ded23ee2, merged here; tables rebuilt in /wd1/mdenolle_data/paper_plan/v6.1/tables/, old ones in tables_pb2002_axes). The segment join uses them; the axis change is an input, noted in the report, not a test choice.
- 2026-11-27 13:30:11 PST: clarification (no rule change), after a coordinator check: the x1.50 in 'Predictors / velocity levels' is applied once, to the RAW member level depths (level_depth_members.npz is computed by pb_structure.py from the raw member grids ensemble_v0.4.2/real/*/merged.npz, not from the product). It is never applied to the stored, already inflated Vp_std / p05 / p95 of v0.4.2+ens.100.nc. Equivalent to Vp_std_raw x 1.50 = stored Vp_std. Vp/Vs is not used.
- 2026-11-27 13:34-13:36 PST: code test (q2_analysis.py main, 50 draws, scratch output, not kept). Two input-sampling details fixed after inspecting the sampled ages (not a rule change): (1) slab events south of the triple junction (40.31 N): the incoming-plate age was sampled at the junction latitude + 0.1 deg, where the 1' age grid mixes the Gorda and Pacific sides of the Mendocino age step (12.8-13.0 Myr); now + 0.2 deg (Gorda plate, 4.9 Myr, as at 40.5-41 N). (2) Transform side ages: where one probe point falls on land (no age, east end of the MFZ), the other side is used for both (stated in the report). The 50-draw test answers were seen; the rules are unchanged.
- 2026-11-27 13:37-13:48 PST: all ten configurations run with 1000 draws (q2_analysis.py; log /wd1/mdenolle_data/paper_plan/q2_depth_limit/q2_analysis.log), circularity (C5b 100 synthetic members; C5c 11 of 20 null members merged, provisional), extras (C3 geometry rho, heat flow, sediment, Monte Carlo shift, within-cluster rho - the last not registered), figures, report /wd1/mdenolle_data/paper_plan/q2_depth_limit/report_q2.html.
- received between 13:40 and 13:44 PST (exact time not logged): coordinator: D-B114 external data ready (events_external_geo_v61.parquet). Added as labelled variants only (EXT_HSC1300_600, HSC_600_warmtop with the van Keken & Wilson 2023 D80 slab-surface temperature, CRUST1_moho); registered outputs of main checked identical before/after. The external job's exploratory numbers (95-96 % of deep GP/MFZ events below the half-space 600 C isotherm) were seen; stated in the report.
- 2026-11-27 13:49:45 PST: results (main, registered rules): oceanic frame (26 bins): D90 +8.1 to +13.9 km below all six isotherms (H_T falsified under every control); +4.9 km below 7.5 km/s, follows (skill 0.65) but C3-flagged (residual vs station distance), C4 mask-sensitive (no checkerboard-resolved 7.5/8.0 crossing offshore), carried by the Blanco vs Mendocino-Gorda contrast; C5c passes by 0.02 (provisional). Gorda slab: -2.5 km (coldest-slab 600 C), +2.8 km (warm-top variant), 7.5 km/s +5.0 km; GS rho 0.66 fails C5b (4 blocks) and C5c. No composition-limited answer is robust; the rejection of a 350-800 C conductive isotherm in the oceanic plates is.
- To resume: when >= 15 null members have merged (log /wd1/mdenolle_data/tomo_v1/ensemble_v0.4.2_null/run.log), run `bash workflow/11_paper_plan/run_q2.sh circ` (rescoring C5c, figures, report), re-read section 5-6 of the report, commit the tables.
- 2026-11-27 13:50:43 PST: C5c rescored with 13 of 20 null members (run_q2.sh circ): unchanged categories (oceanic frame 0.85 vs null p95 0.83 pass; Gorda slab 0.66 vs 0.68 fail; still provisional). Report rebuilt.

- 2026-11-27 13:57:34 PST: control L computed (configs lit_med, lit_p90, lit_gd, lange, lange_ours). Headline does not survive as stated: lit_med keeps H_T falsified in the oceanic frame (800 C +4.9 km), Gorda plate interior (+6.7) and MFZ (+8.6), not the transforms (+1.0); Lange depths put Blanco's base at the 600-800 C isotherms (+1.1/-1.6 km); lit_p90 and lit_gd fit 800 C in the oceanic frame (-1.8/+0.8). Surviving claim: Gorda plate interior and MFZ need a 7-9 km depth bias for a <= 800 C conductive limit; no paired catalog there. With lit_med the oceanic base sits at 7.5 km/s (+1.2 km). Report sections Summary, 5b, 6, 7, 8 updated.
- 2026-11-27 14:16:07 PST: 15 of 20 null members merged at 14:14:59 PST; C5c rescored (run_q2.sh circ), now final except MFZ (8 valid members). Oceanic frame 7.5 km/s: real 0.85 > null p95 0.83 (pass). Gorda slab: real 0.66 > null p95 0.65 (pass by 0.01; was a fail at 11 and 13 members) but still fails C5b (block-bootstrap low -0.35, 4 blocks), so the Gorda-slab following is still not claimed. Blanco fails C5c (0.41 < 0.47). Report sections 5-6 (and the summary sentence on the Gorda slab) rebuilt from the tables; answers and claims unchanged.
- 2026-11-27 19:26:29 PST: coordinator: D-B117 (PI 19:09) adopts plate boundaries v2; merged origin/analysis/plate-boundaries 0904802e (pb_common PB_AXES default v2, plate_axes_v2_adopted.csv). Segment table /wd1/mdenolle_data/paper_plan/v6.1/events_segments_v61.parquet (md5 cc5e6918, written 19:22) checked: recomputing assign_segments with PB_AXES=v2 gives identical segments; 727 tier A-C events change vs the D-B113 table (GR->MFZ 344, GP->MFZ 186, JP->JR 74, GP->GR 67, JR->JP 35, ...). D-B113 run outputs kept in /wd1/mdenolle_data/paper_plan/q2_depth_limit/run_DB113_axes_v1/. Full rerun (all configurations incl. control L) and C5c on 20/20 null members, same registered rules; D-B117 is a post-registration input change, not a test choice.
- 2026-11-27 19:37:46 PST: v2 rerun done (all 15 configurations, circularity on 20/20 null members, extras, figures, report). Changes vs the D-B113 run: answers unchanged except frame:slab (both consistent -> H_T supported, not preferred: V7.5 shape probability 0.900 -> 0.885 at the 0.9 threshold, Monte Carlo draw sequence; slab-frame data unchanged). Oceanic frame 600/800 C bias +10.8/+8.1 -> +9.5/+7.4 km (v2 transform traces change the frame's transform typing); its H_T falsification now fails under C1 sig_z <= 2 km (800 C +4.6 km, marginal), so it is no longer robust; it stays robust for the MFZ (800 C +12.2 km, every control) and the Gorda plate interior (+9.0 km, now 2 units). Only 23 of the 649 events moved into MFZ enter the depth-reliable set (MFZ units unchanged). C5c final (20 members; MFZ 9 valid, provisional): oceanic 0.85 > 0.84; Gorda slab 0.66 > 0.58 but C5b fails; Blanco fails. Control L: lit_med keeps the rejection in the oceanic frame (+4.2), GP (+5.9) and MFZ (+9.1); transforms +1.4 sit; Lange depths at Blanco +1.2/-1.2 km. Report finalised (sections 2, 5, 5b, 5c, 6, 7, 8 data-driven).
- 2026-11-27 19:43:27 PST: Monte Carlo precision (coordinator check; no rule change): q2_seeds.py reran main with 20 independent seeds (SEED+5000+k) at 1000 draws. frame:slab V7.5 P_skill_pos mean 0.905, SD 0.007, range 0.892-0.915 (70 % of seeds >= 0.9): the slab-frame answer is 'both consistent' in 14 of 20 seeds and 'H_T supported, not preferred' in 6 (registered seed); both readings reported, about 9,800 draws would resolve it at 2 SD. Five other probabilities straddle their thresholds (GS V7.5 P_far, oceanic+slab V8.0 P_skill_pos, slab GDH1_350 P_far, IS HSC_600 P_sit) without changing any answer. Tables mc_precision*.csv; report section 5a, summary, 4.2 note, 6, 7.

## Post-registration control L (coordinator request, rule written 2026-11-27 13:52:46 PST, before computing; decision rules unchanged)
Our depths run deep against local OBS catalogs (v6.1 minus Lange et al. 2026, event-paired, b4_lit_paired.py: /wd1/mdenolle_data/catalog_v6/v6.1_analysis/validation/depth_literature/paired_v6.1.csv). Question: does the rejection of the 350-800 C isotherms (H_T falsified, main configuration, per frame and group) survive if our depths are too deep?
- L1 (shift): each event's depth minus the paired offset of its region, Lange absolute catalog, tiers A-B rows (our set is A-B): events of segment BL by event longitude in the Blanco segments of b4_lit_paired.py (West -131/-129.3, Surveyor -129.3/-128.75, Cascadia Depression -128.75/-128.45, Blanco Ridge -128.45/-127.1, Gorda Depression -127.1/-126.0; outside them: Blanco all); every other event (no paired reference): the Blanco-all value. Two configurations: lit_med (median offsets: Blanco all +3.17 km) and lit_p90 (p90 offsets, worst case: Blanco all +9.87 km). Sensitivity, same control: lit_gd = lit_med but the Mendocino-Gorda system (all events south of 42.5 N outside segment BL) takes the Gorda Depression median (+9.58 km, n 23), the paired segment nearest to it. Segments, frames and predictors stay those of the unshifted events; only the depths entering D90 move; nothing is written back.
- L2 (Lange depths): main-set events of segment BL paired with the Lange absolute catalog; D90 per 0.5 deg bin (>= 20 paired events) from Lange's depths (datum read as km bsl, as in b4_lit_paired.py; Monte Carlo with Lange's ez), configuration lange; the same events with our depths, configuration lange_ours. Compared with the same isotherms (transform two-sided geotherm of the event positions).
- Reported per frame and group with the registered rules: H_T falsified or not, bias against each isotherm. The headline survives if the oceanic frame keeps 'H_T falsified' under lit_med; lit_p90 and lit_gd are stated as worst cases; L2 says whether Blanco's base is still below the isotherms with an independent OBS depth set.

## Pre-registration (written 2026-11-27 13:27:22 PST; not changed after the numbers are seen; any other choice is a reported variant)

### Hypotheses
- **H_T (temperature):** the base of seismicity follows an isotherm of a conductive cooling model: 600 C (primary), also
  350 C and 800 C; two cooling models: half-space (HSC: Tm 1350 C, kappa 1.0e-6 m2/s; Turcotte & Schubert 2002) and
  plate GDH1 (Stein & Stein 1992: plate 95 km, Tm 1450 C, kappa 8.04e-7 m2/s, series solution). Six isotherm variants.
- **H_C (composition / velocity):** the base of seismicity follows a velocity level of model v0.4.2: Vp = 7.5 km/s
  (primary) or 8.0 km/s (Moho proxy).

### Events (depth-reliable set E)
Catalog v6.1, located (loc_source nll3d), tier A or B, depth_quality == ok, sig_z <= 3.0 km, not floor_limited, not
far_only, not deep_pass, not five_phase, event_flag empty, not review P1. Variants: sig_z <= 2 km; tier A-B without a
sig_z cut; tiers A-C (the plate-boundary set).
Offshore plate system only: segments AX, EN, JR, EXR, GR (ridge), BL, MFZ, SOV (transform), GP, JP, EXP (intraplate),
MTJ (triple junction), GS, SIF, IS (subduction); FA, NAPA and onshore segments are not tested.

### Depth frames
Every depth is reported below sea level (bsl, catalog depth) and below the seafloor (bsf = depth + surface_km of the
event). Isotherms are computed below the seafloor (variant: below the basement = seafloor + GlobSed v3 sediment
thickness); velocity levels are model depths bsl, converted to bsf with the same surface_km. Residuals do not depend on
the frame.

### Units
- U1: model columns (0.125 x 0.1 deg node columns of v0.4.2), >= 20 events of E (continuity with the plate-boundary
  analysis; columns are not independent at the 40 km model resolution).
- U2: 0.5 x 0.5 deg bins, >= 20 events of E. **Decisions use U2**; U1 is a reported variant.
- Groups: (a) boundary type by the D-B113 segments (ridge, ridge without AX, transform, intraplate, triple junction,
  slab = GS+SIF+IS, and the Gorda slab GS alone) and per segment with >= 5 units; (b) segment-independent frames from
  the event position only: oceanic (west of the PB2002 trench trace) and slab (east of it, Slab2 defined, depth >=
  Slab2 - 5 km); upper-plate events are not tested. The segment join is re-run from the segment table at run time
  (path, md5 and mtime recorded) so a later axis change only needs a rerun.

### Seismicity base
D90 (primary) and D95 (variant) of the unit's event depths. 1000 draws (seed 20261127): events resampled with
replacement within the unit, each depth + sig_z x N(0,1) (location Monte Carlo). Central value: D90 of the observed
depths; intervals: 68 % / 95 % of the draws.

### Predictors (per event, unit value = median over the unit's events)
- Plate age: Seton et al. (2020) 1 arc-minute grid (age.2020.1.GTS2012.1m.nc), bilinear at the epicentre.
- Oceanic frame: z_T(age) below the seafloor.
- Transforms (BL, MFZ, SOV, and in the frame analysis any oceanic event within 15 km of a transform trace of
  pb_common.pb2002()): ages on both sides = median grid age within 5 km of the points 20 km on either side of the
  local trace, perpendicular to it; geotherm = mean of the two geotherms, as for transform faults in Boettcher & Jordan
  (2004); variants young side only and old side only.
- Slab frame (east of the trench): incoming-plate age = grid age 10-30 km west of the trench trace at the event's
  latitude (median); isotherm hung below the Slab2 top, measured normal to the slab (vertical offset z_T / cos(dip)),
  slab top at 0 C, no heating from the upper plate: an upper bound on isotherm depth (coldest slab), not a slab
  thermal model. If a published slab thermal model becomes available under /wd1/mdenolle_data/external_geo, it is
  added as a variant.
- Ridges: age at the epicentre (near zero; the cooling model does not describe the axial magma system - stated).
- Velocity levels: z_7.5 and z_8.0 per column = pb_common.level_depth (shallowest crossing between two rock nodes both
  in the P mask; mask = rock and (checkerboard-resolved or DWS >= 100)). A unit has a level when >= 50 % of its events
  lie in columns with a resolved level. Uncertainty: the 100 members (level_depth_members.npz, same mask), deviation
  from the member median x 1.50 (Vp calibration, D-B099), added to the v0.4.2 level; a draw uses one random member.
- Thermal uncertainty in the draws: kappa x U(0.8, 1.2) and Tm (HSC U(1300, 1400) C, GDH1 U(1400, 1500) C) per draw
  (systematic), age x exp(0.15 N(0,1)) per unit.
- Reported covariates, not in the decision: heat flow (GHFDB R2024, median within 25 km) and the linear-conductive
  600 C depth k 600 / q (k 3.138 W/m/K); sediment thickness; crustal thickness if available.

### Metrics (per group, per predictor level L, per draw; units that have the predictor)
- Residual r_u = D90_u - z_L,u (km). Bias = median r_u. Misfit M = RMS of r_u.
- Shape (does the base follow L?): leave-one-block-out cross-validation with 1 deg blocks (U2) / 0.5 deg blocks (U1):
  prediction D90 = z_L + median(r on the training units) against the null prediction median(D90 on the training
  units); skill = 1 - RMSE_L / RMSE_null. Also Spearman rho(D90, z_L).
- Temperature at the base T* (each cooling model) and Vp at the base V* (model column, interpolated, mask) per unit:
  descriptive.

### Decision rules (delta = 3 km, about the node spacing at 15-35 km and the D90 uncertainty)
1. Absolute fit of L: "sits at L" if |bias| <= 3 km in >= 50 % of draws; "does not sit at L" (falsified) if |bias| >
   3 km in >= 95 % of draws; else marginal.
2. Shape: testable when the predictor spans >= 3 km (p90 - p10 of the unit values) over >= 5 units; "follows L" if
   skill > 0 in >= 90 % of draws; "does not follow L" if skill <= 0 in >= 90 % of draws; else undetermined.
3. Preference on units that have both predictors: primary pair 600 C HSC vs 7.5 km/s; secondary pair best-of-family
   (lowest M among the six isotherms vs among the two levels, chosen inside each draw). P_T = share of draws with
   M_T < M_C. H_T preferred if P_T >= 0.9; H_C preferred if P_T <= 0.1; else not distinguished. Same rule on the
   cross-validated RMSE for shape.
4. Falsification: H_T is falsified in a group if all six isotherm variants fail rule 1 (base > 3 km from each in >= 95
   % of draws) or, where the shape is testable, the primary isotherm "does not follow" (rule 2); it is also reported
   when the 95 % interval of T* lies outside 350-800 C. H_C is falsified if both levels fail rule 1 or the primary
   level "does not follow".
5. Answer per group: temperature-limited = H_T sits or follows, H_C falsified or H_T preferred; composition-limited =
   the reverse; both consistent / neither / not testable otherwise.

### Controls (a result that changes category under a control is labelled with it and not claimed as robust)
- C1 depth uncertainty: the draws above; variants sig_z <= 2 km and tier A-B without sig_z cut.
- C2 catalog version: the whole analysis on v6.0 (its own segment table events_segments_v60.parquet, PB2002 axes).
- C3 station geometry: events whose nearest station is within one focal depth (dist_min <= bsf depth); plus Spearman
  rho between unit residual (primary levels) and unit median dist_min, flagged if |rho| >= 0.3 with a block-bootstrap
  90 % interval excluding 0.
- C4 resolution: levels only on the mask; variant checkerboard-only (resolution_p > 0).
- C5 circularity (velocity levels come from an inversion of the same events):
  (a) D90 from events not in any v0.4.2 tomoDD dataset (union of inputs/events.csv of margin_m0_v4,
      mendocino_gorda_c_m0g, blanco_m0g, oregon_m0f, washington_m0g). An H_C "sits/follows" result is claimed only
      if it holds here too (same category; skill > 0 in >= 80 % of draws); < 5 units: not testable, not claimed.
  (b) synthetic calibration members s000-s099 (truth v0.4.2 x (1 + f)): levels in the truth and in each member (same
      mask); apparent following A_i = rho(D90, z_rec,i) - rho(D90, z_true) per group. If median A > 0 with >= 90 % of
      A_i > 0, the real rho is corrected by median A; a following claim needs the corrected rho > 0 with the
      block-bootstrap 90 % lower bound > 0.
  (c) D-B112 null-truth members (ensemble_v0.4.2_null, structure within 45 km of the segments set to layer means):
      a following claim needs rho_real > the 95th percentile of rho(D90, z_rec,null_i) (>= 15 null members). If the
      null run is not finished when the report is written, the claim is marked provisional and the command to score
      it is given.
  (d) spatial cross-validation: rule 2 is already leave-one-block-out.

### Outputs
Tables data/paper_plan/q2_depth_limit/*.csv; report /wd1/mdenolle_data/paper_plan/q2_depth_limit/report_q2.html
(question, pre-registered tests and time, data, results with figures, controls, claims and non-claims, manuscript
bullets, limitations). Stated limitations: Axial SSST depth bias (D-B100/D-B103), ridge depths other than Axial poorly
constrained (depth_quality unconstrained), 40 km model resolution, cooling-model assumptions (conductive, no
hydrothermal cooling, no shear heating, no sediment blanketing, slab bound only).

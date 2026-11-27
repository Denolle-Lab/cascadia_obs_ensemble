# Q1 slip style (D-B115): resume notes and pre-registration

Question Q1 (PI, D-B115, 2026-11-27 13:19): what sets how a plate boundary releases slip, in swarms or in
mainshock-aftershock sequences?

Worktree /home/mdenolle/wt_q1, branch science/q1-slip-style (from origin/main a6fb0db2, origin/analysis/plate-boundaries
7fcf85d7 merged for the pb_* helpers). Code workflow/11_paper_plan/q1_*.py and run_q1.sh; small tables in this folder;
report, figures, large tables /wd1/mdenolle_data/paper_plan/q1_slip_style/. Python: the pixi env of the main tree with
LD_LIBRARY_PATH=/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/lib; nice 10; at most 5 worker processes (a
tomoDD null-truth run holds 6); MemAvailable kept above 40 GB.

## Status log
- Fri Nov 27 13:26:14 PST 2026: worktree made, inputs inventoried (no metric or correlation computed in this session
  yet). Pre-registration below written before any covariate is sampled or any correlation is computed.

## Pre-registration (written Fri Nov 27 13:26:14 PST 2026, before any covariate correlation is computed)

What was already known when this was written (the numbers that motivated the hypothesis, so they cannot confirm it):
the v6.1 plate-boundary table (seismicity_metrics.csv, 10:07): swarm share AX 29 %, MSH 28 %, BL 25 %, GR 19 % against
<= 2.8 % at MFZ, GP, MTJ, GS; b 0.74 (IS) to 1.45 (PU); the region-level time-shuffle null of the v6.1 clustering
(null_test_summary.csv). No covariate (age, heat flow, isotherm, roughness, sediment, spreading rate, Vp/Vs) has been
looked at against any seismicity metric.

### Hypothesis H1
Swarm-dominated release where the lithosphere is young, hot, magmatic or fluid-rich (ridges, Blanco pull-aparts,
Mount St Helens); mainshock-aftershock-dominated release where it is colder and stronger (Gorda plate, Mendocino fault
zone, Gorda slab). Testable form: the swarm share rises with thermal state (younger plate, higher heat flow, shallower
isotherms) and with fluid content (higher in-situ Vp/Vs), and this holds across spatial bins that ignore the segment
boundaries, not only as a contrast between boundary types.

Pre-specified groups (from the hypothesis text, used only for descriptive contrasts): hot = AX, JR, EN, EXR, GR, BL,
SOV, MSH; cold = GP, MFZ, MTJ, GS. Every other segment is in neither group.

### Event set
Catalog v6.1 (cascadia_catalog_v6.1.csv, md5 1cd909b5): located earthquakes (loc_source not none; event_flag empty, so
quarry blasts, explosions, other events and analog_test_signal are out), tiers A-C, not in the waveform-review set P1
(Mount St Helens weekly signal), in the analysis set of the v6.1 nearest-neighbour clustering (in_analysis: M >= the
regional Mc of that run; M is the catalog M, magnitude_source v6.1_converged or v4.1_converged). This is the event set
of pb_seismicity.py. Clustering classes (cls) from /wd1/mdenolle_data/catalog_v6/v6.1_analysis/clusters/events_zbz_v61.csv.gz.
Depth metrics, where used, take depth_quality ok events only (depth_good).

### Units
- U1 segments: the plate-boundary segments of pb_common (D-B113 ridge axis when that table is written; the join is
  one step and is rerun). A segment enters a test when it has >= 100 events in the event set and the covariate is
  defined for it. Segment type "other" (OC, NAPA) is reported, not tested.
- U2 spatial bins (segment-independent): 0.2 x 0.2 degree cells (lon, lat) on a grid anchored at -132 W, 38 N, events
  by epicentre; a cell is a sample when it holds >= 50 events of the event set. Cell type = the segment type of most
  of its events (only for the type-controlled test). Cell covariate = median of the per-event covariate over its
  events (cells with fewer than 50 % of events carrying a value: no value).
- U3 families (Vp/Vs only): v5.1 families with an in-situ Vp/Vs (below), typed by their matched v6.1 family.

### Metrics (per unit)
- M1 swarm share (primary): percentage of the unit's event set in families of type swarm (cls == swarm).
- M2 background fraction: percentage with cls background.
- M3 family-type mix: swarm families / (swarm + mainshock-aftershock families), families with n >= 5 analysis-set
  events, a family in the unit of most of its members; also the swarm share of clustered events.
- M4 b: Aki-Utsu (continuous M) above the unit Mc (maximum curvature + 0.2, as pb_seismicity), >= 50 events above
  Mc; 95 % interval from 500 event bootstraps re-estimating Mc each draw.
- M5 aftershock productivity: mean number of direct offspring (linked analysis-set children, parent_id) of analysis-set
  events with M - Mc_region in [1.0, 2.0); >= 20 such parents; 95 % interval from 500 parent bootstraps.
- M6 swarm duration: median duration (duration_all_days) of swarm families with n >= 5 in the unit.
- M7 swarm migration: share of swarm families with n >= 10 flagged migrating (migrating or mig_migrating), median D90.
  M6 and M7 are descriptive (few families per unit); no test.

### Covariates (per event, aggregated per unit as the median over the unit's events)
Primary source: the D-B114 per-event table under /wd1/mdenolle_data/external_geo/ (PROVENANCE.csv) when it exists at
report time. Until then an interim sampler of the downloaded grids (q1_covariates.py); both are reported when both
exist, and the decision uses the D-B114 table.
- C1 plate age (Myr), Seton et al. 2020 age grid (1 arc-min) at the epicentre; NaN where the grid has no oceanic age.
- C2 heat flow (mW/m2): D-B114 value; interim: median of GHFDB R2024 measurements within 25 km of the epicentre
  (NaN with fewer than 3 measurements).
- C3 isotherm depth (600 C) from plate cooling at C1: a monotonic function of C1, so it has the same ranks; it is
  reported, not tested separately. A slab-model isotherm (D-B114) is a secondary covariate if provided.
- C4 spreading rate (full rate, mm/yr, Seton et al. 2020 grid) at the epicentre.
- C5 seafloor roughness: RMS of GEBCO 2026 bathymetry after removing a best-fit plane, in a 0.2 x 0.2 degree window
  centred on the epicentre (offshore only).
- C6 sediment thickness (GlobSed v3, m) at the epicentre.
- C7 in-situ Vp/Vs of families (U3; per unit: median over the unit's families).
Predicted signs (H1): swarm share vs C1 negative, vs C2 positive, vs C3 negative, vs C4 positive, vs C7 positive.
C5 and C6: no predicted sign (exploratory, two-sided).
Nuisance covariates (controls only): Mc map value (mc_best of the 0.1 degree cell, v6.1 map without non-earthquakes),
dist_min (nearest station, km), offshore flag (surface_km < 0).

### In-situ Vp/Vs (Lin & Shearer 2007 style)
- Data: accepted waveform CC differential times dt_<block>.cc of /wd1/mdenolle_data/tomo_v1/cc_dt_all (P window
  [-0.2, 0.6] s, S [-0.3, 1.0] s, cc >= 0.7, cc - cc2 >= 0.1, clock-corrected); pairs whose two events are in the same
  v5.1 family (events_zbz_v51 family_id) and lie within 5 km of each other (v5.1 hypocentres, pair_coherence sep_km).
- Per pair: stations with both an accepted P and an accepted S row; >= 3 such stations; P and S times demeaned over
  those stations (removes the origin-time difference). Points (dtP', dtS').
- Per family: >= 5 events in contributing pairs and >= 30 points; Vp/Vs = slope through the origin of dtS' on dtP' by
  least absolute deviations (L1, as Lin & Shearer 2007); 95 % interval from 200 bootstraps over pairs. Kept when the
  interval half-width is <= 0.10. Values outside 1.4-2.5 are kept in the table, flagged, and left out of the tests.
- Variants (reported): orthogonal (total least squares) slope; catalog picks (Cascadia_picks_catalog_v4.1tc) instead of
  CC on the same families.
- Family type: the v6.1 family holding most of the v5.1 family's members (carried over when it holds >= 50 % of them);
  the count carried over is reported. v5.1 type as a variant.

### Tests
- T1 (primary): 0.2 degree cells, Spearman rho(M1, C1), one-sided permutation p (10,000 permutations of the covariate),
  predicted rho < 0.
- T2 (primary): the same for C2, predicted rho > 0.
- T3 (primary): families, Vp/Vs of swarm families minus mainshock-aftershock families (difference of medians), labels
  permuted within segment type (10,000), one-sided, predicted > 0. Needs >= 20 swarm and >= 20 MA families with a valid
  Vp/Vs, else "not testable".
- T4a, T4b (primary, type-controlled): cells, the weighted mean of within-type Spearman rho (weight = number of cells of
  the type; types with >= 5 cells with a value) for C1 and C2; covariate permuted within type (10,000); one-sided in the
  predicted direction. Effect size reported alongside: quasi-binomial GLM logit(M1) ~ z(covariate) + type.
- Multiple testing: Holm at alpha 0.05 over the five primary tests (T1, T2, T3, T4a, T4b). Secondary tests (BH FDR,
  q = 0.05, one family): cells, M2-M5 against C1 and C2 and M1 against C4 (predictions: b falls with C1 and rises with
  C2; productivity rises with C1 and falls with C2; M3 as M1; M2 two-sided), segment-level Spearman of M1 against
  C1, C2, C4, C7 (n small). Exploratory (BH within, labelled exploratory): C5, C6, everything else.
- Spatial dependence: every cell-level rho also gets a 95 % interval from 2,000 bootstraps of 1 x 1 degree blocks.

### Decision rules
- D1 "thermal control supported": T1 or T2 significant after Holm with the predicted sign, its block-bootstrap 95 %
  interval excludes 0, the sign holds in >= 18 of the 22 control runs (list below; 80 %), and the time-shuffle null rule
  (below) passes.
- D2 "beyond boundary type": D1 and T4 of the same covariate significant after Holm. D1 without D2: the paper can claim
  a contrast between boundary types that tracks thermal state, not that thermal state sets the style within a type.
- D3 "fluid signal": T3 significant after Holm (swarm > MA).
- Falsified (on these data): T1 and T2 both with the opposite sign and two-sided p < 0.05; or under the uniform-Mc
  reclustering the median swarm share of the hot group is not above that of the cold group; or the time-shuffle null
  rule fails. For Vp/Vs: MA > swarm with two-sided p < 0.05.
- Otherwise inconclusive. Thresholds are not changed after the results are seen; other thresholds are variants only.
- Time-shuffle null rule: in >= 3 of the 4 hot segments with the largest event sets (AX, BL, GR, MSH) the observed M1 is
  above the 95th percentile of M1 in the time-shuffled catalogs.

### Controls (the 22 control runs of D1 are marked *)
- Completeness: * uniform Mc 2.15 (the largest regional Mc of the v6.1 clustering, GO) counting only M >= 2.15 events
  with the main families; * the clustering redone with Mc = 2.15 in every region (thresholds refitted); * local Mc:
  only events at or above the mc_best of their 0.1 degree Mc map cell.
- Clustering: * each of the 12 v6.1 variants (d3, d3q, df1.3, df2.0, df_est, eta0_shuf5, mc+0.3, mc_best, mc_year,
  noM_floor, storedM, tierABC).
- Network and detection: * offshore cells only; * partial rank correlation controlling for the Mc map value and
  dist_min; * the Cascadia Initiative window (2011-10-01 to 2015-09-30) only.
- Bins: * 0.4 degree cells; * N >= 30; * N >= 100.
- Catalog: * v6.0 (its clustering and segments) instead of v6.1.
- Nulls: (a) time-shuffled catalogs: event times permuted within each clustering region over the whole period, the
  real thresholds, 100 catalogs, M1 per segment and cell; (b) magnitude shuffle: magnitudes permuted among the
  analysis-set events of each region, family membership and trees kept, families reclassified (200 draws); reported,
  not part of a rule.

### Limitations stated up front
Families shift between catalog versions (D-B111): claims on individual families only for large sequences; the metrics
here are shares over many families. Mc differs between regions (0.65 at Axial to 2.15 at Gorda): handled by the
above-Mc event set and the uniform-Mc controls. Tier D and floor-limited depths are not used for depth metrics. No
waveform double-difference catalog: migration is provisional.

# Model v0.4.0: v0.3.1 re-inverted with the re-timed picks

PI, 2026-11-25: "we should probably re-run v0.3.1 with the correct re-timed picks". Goal: the v0.3.1 configuration with
only the pick times changed, so that v0.4.0 - v0.3.1 is the effect of the re-timing (D-B072, D-B075). No surface
treatment, no D-B073 rule (those are designed separately).

Branch tomo/v040-retimed (worktree ~/wt_retime, from tomo/tiles 4c201288). Outputs under
/wd1/mdenolle_data/tomo_v1/v040_retimed/ (RESUME.md, chain.log, chain.state, logs/, report/), runs under
/wd1/mdenolle_data/tomo_v1/b2_tomodd/{margin_m0_v3, <tile>_m0t, margin_m0_v3nf, <tile>_m0tnf}.

## 1. How pick times enter the tomography, and the join

* B1 (b1_select_events.py) reads catalog **v4.0** picks (params paths.picks_v4 = data/catalog_v4/Cascadia_picks_catalog_v4.csv,
  a link to /wd1/mdenolle_data/catalog_v4_0_backup/, md5 17b7d0bc, as D-B035 states), matches them to the iter0/iter1
  NLLoc phases on (event, station, phase, time to 0.1 ms), and writes pick_time (v4.0 time, 1 us) and
  travel_time_s = pick_time - v4-NLL3D origin time. B1b (b1b_balance.py) keeps a subset of the rows and adds weights.
  b2_tomodd_inputs.py uses only travel_time_s (absolute.dat = tt - OBS S static + P-4; dt.ct = tt - static).
* The pick files have no ID column. Pick ID = (event_ID, station, phase, catalog v4.0 pick time): unique in v4.0 and
  in v4.1 (0 duplicate keys). The re-timed file is v4.1t (D-B075), whose corrected rows carry the new time; the
  original time of each v4.1 row was rebuilt from corrections.csv (old_pick_datetime, applied rows; checked against
  timing_correction_s to 1 us). v4.0 vs v4.1: 2,503,184 rows in both, v4.1 adds 443 rows (events 900001-3, 172114),
  v4.0 has 5 rows (event 172113) v4.1 dropped; none of these is a tomography event.
* Every tomography pick (b1_dataset_c picks 765,486 + rejected 88,979; b1b_balanced_c picks 754,710 + dropped
  10,776) joins to exactly one v4.0 row and one v4.1 row: one-to-one, nothing unmatched (b1t_retime_dataset.py
  stops otherwise). travel_time_s = round(new time - origin, 4) reproduces the v0.3.1 value from the v4.0 time at
  every row before the shift. B1b picks map onto B1 picks with the same correction.
* Datasets b1_dataset_ct, b1b_balanced_ct: same rows, same order, all other columns identical; pick_time and
  travel_time_s re-timed; added pick_time_v40, timing_corrected, timing_correction_s, retime_epoch_class,
  pick_row_v40, pick_row_v41. The NLLoc diagnostic columns (iter1_res_s ...) are left as they were.

## 2. What was kept from v0.3.1

* Events, picks, weights (w_bal, sigma), OBS S statics (the same files: obs_near_station_S_terms_p1_<tile>.csv and
  _margin_m0.csv, which v0.3.1's M0 used but which had never been committed; added unchanged, md5 c16e040c as its
  manifest), start hypocentres (event.dat byte-identical), P-1 reference (same event-station-phase set, reused).
* P-4 recomputed as before: M0 from its own 1-iteration forward run (station_corr.csv byte-identical to v0.3.1's:
  P-4 does not depend on the observed times); tiles reuse <tile>_v2's, as v0.3.1 did (D-B046).
* M0: margin_m0_v3 = margin_m0_v2 with b1b_balanced_ct, damping 200, smoothing 480/480/240, same start (MOD
  byte-identical). Tiles <tile>_m0t restart from the new M0 (start_model_mode change, Vp/Vs floor 1.65) at the
  v0.3.1 smoothing (mendocino_gorda_c s480, blanco s20, oregon s480, washington s480), damping 200. ridges_north is
  not rerun: it is out of the merge (D-B050).
* Merge: b2_merge.py with the same rules (D-B049 gate for tiles at smoothing <= 100, D-B050, D-B051) and the v0.3.1
  d200 resolution masks (choice.csv column mask_region). The masks are valid for the new tiles: event.dat,
  station.dat, picks_used rows, dt.ct pairs and weights are identical (b2_v040_verify_inputs.py), so the synthetic
  tests would see the same rays up to the start-model change. Code check: b2_merge.py at this branch, run as a dry
  run on the v0.3.1 tile runs, reproduces v0.3.1.nc bit for bit (all 15 variables).
* Frozen tomoDD binaries (D-B054) of the same size classes (m, e13k, e10k, w); the D-B054 freeze audit found them
  byte-identical in results to the builds v0.3.1 ran with. Same host, flags, nice 10, at most 10 tomoDD at once.

## 3. Rule checks (v0.3.1's choices kept; reported only)

* B1 pick QC (4 sigma_MAD residual cut, 5 sigma_MAD S-P cut, post-QC event rule): b1t_qc_check.py rebuilds the B1
  candidates exactly (reproduces b1_dataset_c, 765,486 picks), shifts every candidate pick, its travel time and its
  NLLoc residuals by its correction and reruns the QC.
* M0 smoothing scan 20-1920 and tile scans 20-480 at damping 200 rerun with the re-timed picks; the D-B046 rule
  (largest smoothing within 1.5 % of the best weighted rms) applied to them.

## 4. Numerical noise floor

No ensemble noise level exists for this configuration, so the whole chain was rerun with N(0, 1e-6 s) added to
every observed time (b2_v040_noise.py; absolute.dat and dt.ct with 6 decimals; tomoDD reads them into REAL*4, so this
is about one float32 ulp of a travel time): margin_m0_v3nf, <tile>_m0tnf from it, merged as a dry run. Tiles restart
from the perturbed M0, so the floor includes the M0 noise every tile inherits.

## 5. Results

See section "Results" appended at the end of the run (report/report.html; tables in data/tomo_v1/v040_retimed/).

## 6. Proposed decisions (not in DECISIONS.md; for the PI)

See the end of this note.

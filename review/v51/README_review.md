# Catalog v5.1 location review: CSV, viewer and returned answers

PI request 2026-11-25: "Send a CSV report but also an HTML file to visualize it and then we can report back with a CSV
file for corrected events". Built 2026-11-25 by workflow/10_tomography/review_v51/ (branch review/v51-locations).
Nothing here changes the catalog: the files document four location problems of catalog v5.1 and collect reviewers'
answers for the next catalog version.

## Files

| file | what |
|---|---|
| review_v51_events.csv | 1,512 rows: one per event and problem of the four review sets (all events, not only those with an event page) |
| review_v51_viewer.html | one self-contained file (5.8 MB): summaries, sortable/filterable table, record-section thumbnails, answer fields, CSV load and download |
| index.html, P1_msh_deep/, P2_brookings/, P3_tensor/, P4_deep_far/ | the four HTML reports with the event pages (waveform moveouts, maps, residual tables) |
| README_review.md | this file |
| review_v51_package.zip | all of the above; unzip it and open review_v51_viewer.html (the report folders must stay next to it) |

## How to review

1. Unzip review_v51_package.zip and open review_v51_viewer.html in a browser (Chrome, Firefox, Safari; local file).
2. Type your name in *Reviewer*. Pick a problem tab, filter the table (problem, automatic rule, tier, status, only
   events with an event page, text search).
3. For each event open its event page (link or thumbnail) and fill: *earthquake?* (yes / no / unsure), *v5.1 depth*
   (plausible / implausible / cannot_tell), optional *corrected depth km*, *action* (keep / flag_not_earthquake /
   flag_depth_unreliable / relocate / other) and a comment. Answers are saved in the browser as you type.
4. Click *Download CSV* and send the file back. It has all 1,512 rows and the same columns as review_v51_events.csv.
5. To continue later, or to merge several reviewers, use *Load CSV* (the viewer's CSV or the CSV downloaded from a report
   page). An event answered differently by two reviewers is kept as it is in the viewer and listed as a conflict.

The answer boxes of the report pages also work (they download a smaller CSV per report); the viewer and
ingest_review.py read both formats.

## Rows

| problem | rows | roles | with an event page |
|---|---|---|---|
| P1 Mount St Helens deep events | 343 | 331 deep (> 30 km, box 46.05-46.30N 122.35-122.10W), 10 shallow references (tier A-B, < 10 km), 2 Wednesday-window events shallower than 30 km | 74 |
| P2 Brookings swarm (v1_GO022) | 39 | members | 39 |
| P3 ComCat moment tensors | 150 | 139 v5.1 tensor matches, 8 large movers (v4.1 tensor carried), 900001 (v4.1 addition), 900002-900003 (no v5.1 hypocentre) | 148 |
| P4 deep_pass and far_only | 980 | 356 deep_pass, 624 far_only | 81 |

An event in two sets has one row per set: 171 IDs appear twice: 140 in P1 and P4 (Mount St Helens deep events that are deep_pass), 31 in P3 and P4 (tensor events that are deep_pass or far_only).

## Columns

- ID, problem, role; time_utc, lat, lon, depth_v51 (km below sea level, true depth), depth_v4, comcat_id, comcat_depth
  (ComCat hypocentre: tensor match, else the mag piece's ANSS match); tier, sig_z, h_max, gap, nearest_station_km,
  flags (floor_limited, far_only, deep_pass, five_phase), depth_quality, event_flag, region, magnitude, magnitude_type
  (catalog v5.1).
- nP, nS: phases in the v5.1 NonLinLoc .hyp. rms_v51, rms_v4, rms_comcat: residual rms of the re-timed picks for each
  hypocentre in the SAME v5.1 travel-time grids and station terms, origin time refitted per hypothesis (weighted mean
  of observed minus predicted). The v5.1 point is NonLinLoc's optimum in these grids, so rms_v51 is expected to be the
  smallest. nearest_sta, nearest_min_weight (NonLinLoc weight of the nearest station's arrivals), n_weight_lt_0p1.
  scatter_z16 / z50 / z84: percentiles of the NonLinLoc scatter depths.
- P1 only: pct_P_at_UW (share of P picks at UW stations; UW stations at Mount St Helens are analog short-period
  channels digitized centrally), P_span_s (time between first and last P pick), P_dmax_km (farthest P pick from the v5.1
  epicentre), n_S_picks, has_S (pick file v4.1t); wed_1914_1921 (Wednesday 19:14:00-19:20:59 UTC); calib_HSR / EDM /
  STD (yes/no/no data), quiet_s_* (longest stretch of 1-s windows of the 2-15 Hz trace with a standard deviation below
  10 % of the median, s) and step_* (largest |1-s mean - median| of the raw trace over the median 1-s standard
  deviation) from a short read of HSR.UW, EDM.UW and STD.UW from 30 s before to 90 s after the v5.1 origin time;
  calib_any (yes at any of the three).
- P2 only: sp_obs_<STA>, sp_pred_v51_<STA>, sp_pred_v4_<STA> for KBO.NC, KSXB.NC, L02D.TA, L02E.TA (S-P in s; predicted
  in the v5.1 grids with station terms; empty when the event has no P and S pick there).
- P3 only: dz_comcat, dh_comcat, dt_comcat (v5.1 minus ComCat: depth km, epicentre distance km, origin time s), comcat_time.
- P4 only: dz_v4_to_v51 (v5.1 depth minus v4 depth), two_cluster_scatter (yes/no), two_cluster_depths (the two peaks, km).
- has_view, view_path (relative path of the event page, e.g. P1_msh_deep/index.html#ev-29839).
- auto_indicator_rule_not_verdict: the automatic rules that fired, joined with ';', or none. These are pointers for the
  reviewer, not verdicts.
- Reviewer columns (empty): reviewer, is_earthquake (yes/no/unsure), depth_verdict (plausible/implausible/cannot_tell),
  corrected_depth_km (optional), action (keep / flag_not_earthquake / flag_depth_unreliable / relocate / other),
  comment, reviewed_utc.

## Automatic rules (auto_indicator_rule_not_verdict)

| rule | definition | rows |
|---|---|---|
| P1_common_mode | share of P picks at UW stations >= 0.9 AND P-pick time spread <= 0.5 s (farthest P pick > 8 km from the v5.1 epicentre) AND no S pick | 194 of 343 |
| P1_calibration | calibration-type signal at HSR.UW, EDM.UW or STD.UW from 30 s before to 90 s after the origin time: quiet stretch >= 10 s AND step >= 30 | 74 |
| P1_wed_window | origin on a Wednesday 19:14:00-19:20:59 UTC | 24 |
| P2_nearest_downweighted | NonLinLoc weight < 0.1 for an arrival at the nearest station | 14 of 39 |
| P2_sp_mismatch | \|observed - predicted (v5.1) S-P\| > 0.5 s at KBO.NC, KSXB.NC, L02D.TA or L02E.TA | 9 |
| P2_v4_fits_better | rms_v4 < rms_v51 - 0.05 s | 16 |
| P3_large_move | v5.1 more than 30 km (3-D) or more than 4 s from the ComCat origin | 56 of 150 |
| P3_deeper_20km | v5.1 more than 20 km deeper than ComCat | 60 |
| P3_comcat_fits_better | rms_comcat < rms_v51 - 0.05 s | 2 |
| P4_two_cluster | scatter depths form two clusters: two peaks of a 2-km Gaussian kernel density at least 10 km apart, the smaller at least 25 % of the larger, density between them below 50 % of the smaller peak | 77 of 980 |
| P4_depth_change_30km | \|v5.1 depth - v4 depth\| > 30 km | 634 |
| P4_v4_fits_better | rms_v4 < rms_v51 - 0.05 s | 89 |
| P4_sigz_gt_10km | catalog sig_z > 10 km | 877 |
| no_hypocentre | no v5.1 hypocentre (900002, 900003) | 2 |

Rows with at least one rule: P1 216 of 343 (none of the 10 references), P2 18 of 39, P3 75 of 150 (73 plus the two
without a hypocentre), P4 927 of 980.

## Returned CSVs

    python workflow/10_tomography/review_v51/ingest_review.py returned1.csv [returned2.csv ...] [--out DIR]

Validates (known ID and problem, allowed values, corrected depth a number in -5..700) and lists conflicts (the same ID
answered differently by two reviewers). Writes to /wd1/mdenolle_data/catalog_v5/v5.1_review/returned/:
review_summary.md, review_answers_merged.csv, review_flag_not_earthquake.csv (rule: a reviewer's action =
flag_not_earthquake), review_depth_unreliable.csv (rule: action = flag_depth_unreliable or depth_verdict = implausible),
review_relocate.csv (action = relocate, with the corrected depths), review_conflicts.csv, review_invalid_rows.csv.
`--selftest` runs it on synthetic filled CSVs.

## Limits

- The viewer and the report pages were tested without a browser (none on the build machine): script syntax, a stub-DOM
  run of the page script (render, edit, autosave, download), CSV round trip byte-equal. Check the first real use.
- NC and BK waveforms come from NCEDC FDSN (the project's route for these networks), on NCEDC's times; every other
  network from pnwstore short reads on the record-header time base.
- Filters on the event pages are zero-phase (acausal ringing can precede a sharp onset); traces are scaled per row.

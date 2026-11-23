# B1b balanced travel-time dataset (well-recorded stations)

Agent: Claude (coordinator session), 2026-11-22. PI asked for a balanced dataset of travel times at
well-recorded stations, for the tomoDD tomography (D-B021) and the B4 relocation. Status: built;
D-B022 adopted on PI approval of the plan; ridge regions brought back with softer criteria (D-B023,
PI request, section "Ridge regions" below). Starting locations are the v4-NLL3D catalog (catalog v4
relocated with NonLinLoc in the 3D starting model; formerly "Track A", D-B024).

Script `workflow/10_tomography/b1b_balance.py`, parameters `params.yaml` block `balance`, input the B1
dataset (`/wd1/mdenolle_data/tomo_v1/b1_dataset`). Outputs in `/wd1/mdenolle_data/tomo_v1/b1b_balanced`
(picks.csv with weights, events.csv, stations_core.csv, picks_dropped.csv, balance_metrics.csv); tables and
figures in `data/tomo_v1/b1b_balanced/` (b1b_tables.md, b1b_stations.png, b1b_station_balance.png,
b1b_region_shares.png).

## Why balance the stations

B1 declustered the events (3 km cells, 3 events each) but not the stations. In B1 the top 50 of 646
stations hold 37% of the picks (PB boreholes and NC stations near Mendocino, 6-10k picks each); the three
onshore regions hold 55%; the per-station Gini index is 0.65. A misfit summed over picks is then dominated
by onshore paths and by many near-identical paths at a few stations.

## Method (three steps)

1. **Core stations.** Per phase: at least 100 picks (land) or 30 (OBS, ~1-year deployments), events in
   at least 3 back-azimuth sectors of 45 deg (land) or 2 (OBS), and in at least 2 of the distance bins
   0-50 / 50-100 / 100-200 / >200 km, with >= 5 picks per sector or bin. A station passes if its P picks
   or its S picks pass; both phases are then kept.
2. **Path-redundancy weight** w_bin = min(1, 40 / n), n = picks of the same station and phase in the same
   25 km x 30 deg (distance, back-azimuth) bin. No pick is dropped: events keep all their picks for
   relocation, but a dense bin counts as 40 picks in the misfit.
3. **Region weight**: equal target share (1/7) for Puget/Georgia, Other onshore, Cascades, Mendocino,
   Gorda, Blanco, Offshore forearc; w_region = target / share of sum(w_bin), clipped to [0.33, 3].
   w_bal = w_bin * w_region, rescaled to mean 1. The pick-quality weight (D-B008) multiplies w_bal when
   the tomoDD files are written.
4. Events keep >= 10 picks at core stations and gap <= 200 deg.

## Choices tested (scratch scan, then the script)

* Hard cap per bin (keep the K best picks) instead of weights: K = 40 kept 67% of picks and cut the
  events with >= 10 picks from 23k to 15.4k; K = 10 kept 31% and 6.9k events. Region shares barely moved
  (onshore dominance is the number of stations, not a few busy ones). Rejected: weights give the same
  station balance without losing events for relocation.
* Station bar land/OBS P picks 100/30 vs 100/50, 200/50, 50/20: 393 / 385 / 353 / 416 core stations,
  picks 672k / 671k / 661k / 675k. Pick totals hardly change; the bar mainly decides how many sparse OBS
  stay. 100/30 kept.
* P-only coverage test dropped 11 Mendocino and Gorda OBS with many more S than P picks (about 5,400 S);
  the P-or-S test brings them back (15 stations pass on S only).
* 3 azimuth sectors for OBS dropped the Blanco arrays (X9.BB330 586 P / 610 S, X9.BB070, Z5.BB870,
  7D.G05D, ...): each sees the linear transform from 2 sectors only. That is fault geometry, not poor
  recording; OBS bar set to 2 sectors.

## Result (main rule, before the ridge supplement)

| set | picks | Kish n_eff | stations | events | Gini (station) | top-50 share | OBS share | S share |
|---|---|---|---|---|---|---|---|---|
| B1 | 692,246 | 692,246 | 646 | 23,428 | 0.65 | 0.37 | 0.27 | 0.50 |
| B1b core, unweighted | 677,235 | 677,235 | 424 | 22,976 | 0.50 | 0.38 | 0.27 | 0.50 |
| B1b w_bin | 677,235 | 546,111 | 424 | 22,976 | 0.42 | 0.30 | 0.25 | 0.50 |
| B1b w_bal | 677,235 | 461,804 | 424 | 22,976 | 0.39 | 0.29 | 0.39 | 0.51 |

Region shares with w_bal: 14.3-14.4% for six regions, 13.8% Offshore forearc (B1: 7.8-24.2%). Region
weights: Puget/Georgia 0.56, Other onshore 0.62, Cascades 1.00, Mendocino 1.23, Blanco 1.62, Offshore
forearc 1.68, Gorda 1.80 (none clipped). The Kish effective sample size, (sum w)^2 / sum w^2, drops by a
third: the price of balance, to be weighed in the resolution tests.

Dropped: 221 stations (15,011 picks: 9,029 at dropped stations, 5,982 of 452 events left below 10 picks or
gap > 200). Most dropped stations are sparse mid-margin OBS of the offshore forearc (64 stations, 1,462
picks); 2,179 of 2,313 forearc events stay. In this first version Axial (29 events), Explorer (71) and
JdF ridge/plate (9) lost almost all events; the PI asked to bring them back (next section).

## Ridge regions: Axial, Explorer, JdF ridge/plate (D-B023)

Why they were lost. Two separate filters. (i) B1 event selection: Axial passes the main rule (3,264
events) but the caldera is a few km across, so 3 km cells x 3 events left 29; Explorer and JdF events
are recorded from 30-55 km away (median nphs 7, sig_z 12 km), so nphs >= 12 and gap <= 200 left 71 and 9.
(ii) B1b station test: the OOI Axial stations and the Explorer/JdF OBS have few picks after declustering
and see the events from one or two directions.

Softer rule, ridge regions only (params `selection.ridge`, `balance.ridge`, `balance.region_groups`):

| step | main rule | ridge rule |
|---|---|---|
| tiers | A-C | A-D |
| nphs | >= 12 | >= 8 |
| gap | <= 200 deg | <= 270 deg |
| RMS | <= 0.5 s | <= 0.5 s |
| decluster | 3 km cells x 3 | 1 km cells x 3 |
| after pick QC | >= 10 picks, gap <= 200 | >= 6 picks, gap <= 300 |
| station | coverage test (above), main-rule picks only | >= 10 picks of ridge events, no coverage test; such a station keeps only its ridge picks |
| region weight | one group per region | the three regions are one group, "Ridges" |

Pick QC thresholds (residual and S-P spreads) are computed from main-rule events only, so the main set
does not change: B1 main events identical (23,319), main picks 690,808 vs 690,809 before (2 picks differ
because the 109 old ridge-region events no longer enter the spread). B1 copy before the change:
`/wd1/mdenolle_data/tomo_v1/b1_dataset_pre_DB023/`. Declustering options tested (events Axial / Explorer /
JdF): 1 km x 3: 435 / 407 / 472; 1.5 km x 3: 236 / 406 / 437; 3 km x 3: 121 / 398 / 362.

Ridge events in B1 by tier: Axial A 89, B 194, C 133, D 12; Explorer B 3, C 377, D 23; JdF B 1, C 393,
D 77. Median 5 P and 4 S picks, gap median 177 deg.

## Result with the ridge supplement (current dataset)

| set | picks | Kish n_eff | stations | events | Gini (station) | top-50 share | OBS share | S share |
|---|---|---|---|---|---|---|---|---|
| B1 | 703,559 | 703,559 | 658 | 24,621 | 0.65 | 0.36 | 0.28 | 0.50 |
| B1b core, unweighted | 689,602 | 689,602 | 475 | 24,249 | 0.53 | 0.37 | 0.28 | 0.50 |
| B1b w_bin | 689,602 | 557,761 | 475 | 24,249 | 0.45 | 0.29 | 0.27 | 0.50 |
| B1b w_bal | 689,602 | 431,050 | 475 | 24,249 | 0.40 | 0.27 | 0.43 | 0.51 |

| rule | events | picks | stations | OBS share of picks |
|---|---|---|---|---|
| main | 22,955 | 676,939 | 418 core | 0.27 |
| ridge | 1,294 (Axial 428, Explorer 403, JdF 463) | 12,663 | 80 (57 ridge-only: Explorer 26, JdF 19, Axial 8, other 4) | 0.90 |

Main part vs the first version: 418 core stations (was 424; the 6 others had passed only thanks to the
109 old ridge-region events and are now ridge-only), 22,955 events (22,973).

Region weights (w_bal shares): seven main regions 12.7-13.4%, Ridges 7.5%. The Ridges weight hits the
clip of 3 (mean ridge pick weight 4.1 vs 0.94 for main picks after rescaling); an equal share (12.5%)
would need ~5x and let ~12k picks of 10-pick events with depth uncertainties of ~12 km steer the misfit.
Kept at the clip. In the regional inversions (D-B025) the ridge sub-region is weighted on its own anyway.

## Open for B2

* Re-admitted far-station picks (D-B012) are not in B1b yet: they need a forward residual test in the
  starting model first; when added they get w_bin from their own distance bins.
* Compare balanced vs unweighted in the synthetic recovery test (balanced kept only if offshore recovery
  improves without losing onshore recovery).
* In the Bayesian ensemble (D-B023, to be written) w_bal enters the likelihood as a fixed per-pick weight
  (a tempered likelihood); pick noise per station class is a hyperparameter on top.

## Proposed decisions

* D-B021 tomoDD replaces TomoATT (PI). Logged.
* D-B022 B1b balanced dataset as above. Logged.
* D-B023 ridge supplement. Logged.

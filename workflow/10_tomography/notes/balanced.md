# B1b balanced travel-time dataset (well-recorded stations)

Agent: Claude (coordinator session), 2026-11-22. PI asked for a balanced dataset of travel times at
well-recorded stations, for the tomoDD tomography (D-B021) and the B4 relocation. Status: built;
D-B022 adopted on PI approval of the plan.

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

## Result

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
picks); 2,179 of 2,313 forearc events stay. Axial (29 events), Explorer (71) and JdF ridge/plate (9) lose
almost all events: out of the tomography target (Axial has its own literature models, D-B005).

## Open for B2

* Re-admitted far-station picks (D-B012) are not in B1b yet: they need a forward residual test in the
  starting model first; when added they get w_bin from their own distance bins.
* Compare balanced vs unweighted in the synthetic recovery test (balanced kept only if offshore recovery
  improves without losing onshore recovery).
* In the Bayesian ensemble (D-B023, to be written) w_bal enters the likelihood as a fixed per-pick weight
  (a tempered likelihood); pick noise per station class is a hyperparameter on top.

## Proposed decisions

* D-B021 tomoDD replaces TomoATT (PI).
* D-B022 B1b balanced dataset as above.

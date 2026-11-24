# B5 relocation test on the Blanco swarms: tomoDD-SE with frozen velocity against GrowClust3D (D-B048)

2026-11-24. Branch `tomo/b5-blanco-test` (worktree `~/wt_b5`, from `tomo/tiles` aa7002b2). Code
`workflow/10_tomography/b5_*.py`, `b5_growclust_run.jl`, `b5_run.sh`; parameters `params.yaml` block `b5_test`;
results `data/tomo_v1/b5_test/<case>/` and `data/tomo_v1/b5_test/summary.md`; large files under
`/wd1/mdenolle_data/b5_test/`.

## 1. Short answer

Neither code improves the B4 locations of the West Blanco swarms with catalog differential times, and neither is
clearly more accurate than the other. GrowClust3D is not the safer choice: its result changes with resampling and
with small changes of the input, and with good starting locations it makes them worse.

* Real swarm, June 2015 (f1803, 641 events, 12 stations): tomoDD (differential times only) and GrowClust3D put the
  same events 0.9-1.1 km apart horizontally and 0.8-1.0 km apart vertically (median, centroids removed). Both keep the B4
  cloud: 14 km along strike 110 deg, depths 6 to 32 km (10th to 90th percentile).
* Synthetic plane with B4-sized start errors (2.9 km horizontal, 6.1 km depth, median): both codes thin the cloud
  across strike, but no code recovers depth (relative depth error 6.8 to 7.5 km, against 6.3 km at the start).
  GrowClust3D with catalog limits gets the best horizontal relative error (1.4 km, against 2.0 to 2.3 km for tomoDD
  and 2.9 km at the start).
* Synthetic plane with small start errors (0.7 km, 1.6 km): no code improves the start. tomoDD keeps it (0.75 km,
  1.7 km). GrowClust3D with default limits is 30 % worse (1.0 km, 1.9 km); with catalog limits it doubles the error
  (1.4 km, 2.8 km) and fits the S data better than the truth does.
* Uncertainty: GrowClust3D's bootstrap spread is 1.0 to 2.5 km horizontally and 1.1 to 3.3 km vertically; tomoDD's
  is 0.1 to 0.7 km, 2 to 11 times smaller than its true error in the synthetics. Under the D-B048 rule, GrowClust3D
  matches tomoDD within GrowClust3D's spread, not within tomoDD's. From a B4-sized start neither bootstrap sees the
  depth error (true 7.4 km; bootstrap 0.7 km for tomoDD, 2.9 km for GrowClust3D).

Recommendation: do not adopt GrowClust3D for B5 on this test. The Blanco swarms are a worst case (4 to 6 OBS per
event, median 8 picks), and catalog times carry little that B4 did not already use. Before choosing, run the same
test on a well-recorded onshore cluster, with D-B040 waveform cross-correlation times; that is the data GrowClust3D
was designed for. tomoDD in relocation-only mode with differential times only (variant `dd`) is a workable B5 code:
it ran 1,455 events in 84 s on one core and is stable. Its centroid is free (as hypoDD with LSQR): it stayed within
0.1 km of B4 in f1803 but moved 0.6 km down in f0399, so a B5 run with it should restore the B4 centroid per cluster,
as GrowClust3D does by design.

## 2. What was run

**Same model, same data.** Both codes relocate in model v0.3.0 and read the same differential times.

* Start: the B4 hypocentres (NonLinLoc in v0.3.0, events with loc_source nll3d), read at full precision from
  `runs/b4_iter1_v0.3.0/locations.csv` and mapped back from the Earth-flattened depth z' with
  z = Re (1 - exp(-z'/Re)), as `b4_catalog.py` does. Tier U events (kept their v4 hypocentre) are left out.
* Picks: the catalog v4.0 picks of the swarm events, kept where the event lies at least 5 km inside the station's
  B4 time grid (GrowClust3D can only use those; B4 drops the same far stations). Travel time = pick minus B4
  origin time. Weight (0.05/sigma)^2 with the D-B008 sigma of the pick_prob class (P 0.05/0.10/0.15 s, S twice).
* Differential times: built once (`b5_common.diff_times`, settings `tomodd.dt`): each event paired with up to 10
  nearest events within 10 km (3D, B4 positions), pair kept with at least 8 common station-phases, weight = the
  smaller of the two pick weights. The same rows go to both codes (`data/dt.csv`). No waveform cross-correlation
  times exist yet (D-B040); none were made and no waveforms were downloaded.
* tomoDD-SE: relocation-only mode, selected by the JOINT column of each weighting set (`JOINT = 0`:
  `lsfitH_FDD_shot`, "only relocate events"; velocity update skipped). MOD = the v0.3.0 nodes (0.125 x 0.1 deg,
  21 depths) over the events and stations plus 0.375 deg, edge nodes as the tiles; D-B028 P-2 coordinate rescaling;
  spherical pseudo-bending rays. LSQR damping 100. Two variants: `hier` = the project's four weighting sets
  (absolute times first, WTCD 10, 10, 1, 0.1; re-weighting 8 and 6 MAD; 10 km pair cut later), absolute times
  corrected with the B4 station terms and the P-4 tomoDD forward corrections of blanco_v2; `dd` = the same sets
  with WTCD 0, differential times only, as hypoDD.
* GrowClust3D.jl 0.1.0 (git 592f6179) on the B4 NonLinLoc grids (`time_b4_v0.3.0`, LAMBERT frame checked against
  every grid header, depths in z'). Its misfit is an unweighted L1 norm; the pair weight goes in the rxcor field,
  which only orders the cluster merges (rmin = rmincut = 0, so every dt is used; delmax 400 km). Two variants:
  `default` = the values of the package's example driver (made for cross-correlation data: distmax 8 km, distmax2
  6 km, shifts <= 2 km, rmsmax 0.2 s, rmedmax 0.1 s, box 3 km) and `catalog` = limits opened for catalog picks
  (distmax and distmax2 10 km, shifts <= 5 km, rmsmax 0.5 s, rmedmax 0.3 s, box 6 km). 100 bootstrap resamples.
* tomoDD bootstrap: 30 runs per variant with the dt rows resampled with replacement (as GrowClust3D does);
  absolute data kept as they are.
* Synthetic planar swarm (`b5_synth.py`): 200 events uniform on a vertical plane 10 km along strike 111 deg x 5 km
  down dip, centred on the f1803 B4 median epicentre and median depth (18.95 km). Each event takes the
  station-phases and pick classes of a random real f1803 event. True times from the B4 grids, Gaussian noise at the
  D-B008 sigma. Start = truth moved by the B4 uncertainty of a random real event (68 % semi-axes h_min, h_max
  -> sigma = h/1.515 along a random azimuth; depth sigma sig_z), clipped at 3 km depth; start origin time = true +
  median residual at the start position. Three cases: `synth` (B4-sized start errors), `synth_s025` (start errors
  x 0.25, a well-recorded event), `synth_tdd` (as synth, true times from tomoDD's own ray tracer in the v0.3.0
  MOD minus P-4, so that GrowClust3D carries the forward error instead of tomoDD).
* Metrics (`b5_compare.py`): centroid shift from B4; PCA of the cloud (sigma1-3, strike and dip of the plane normal
  to the smallest axis); depth percentiles; event-pair distances and centroid-removed position differences on the
  events every run relocated (common set); bootstrap spread; dt residuals of each solution in the B4 grids with the
  per-pair median removed; for synthetics, errors against the truth.

## 3. What had to be fixed first

* **GrowClust3D is not in the Julia General registry** (checked in the registry copy on 2026-11-24). It was added
  from the author's repository, as its README says (`add https://github.com/dttrugman/GrowClust3D.jl`), into a
  separate pixi project with Julia 1.13.4 and `JULIA_DEPOT_PATH` under `/wd1/mdenolle_data/b5_test/`. No install
  or download was refused. The package's own test suite passed.
* **tomoDD relocation-only mode stopped after the first LSQR** (exit 2). `lsfitHFDD_lsqr_lm5.f` line 504 has the
  same missing comma as D-B034 (`f7.1")"` in the "Location summary" FORMAT); joint runs never call this routine, so
  D-B034 did not fix it. Fixed the same way (one comma), built with gfortran 8.5 and the O3nofs flags:
  `/wd1/mdenolle_data/b5_test/bin/tomoDD-SE_O3nofs_e10k_fmt2` (md5 d59d8cdc938660732c6ba28f1c34aec1, source
  `b5_test/src/O3nofs_e10k_fmt2`). This is a source edit of the same kind as D-B034; it needs the PI's agreement
  before B5 uses it.
* **tomoDD segfaults when an event lies at its Cartesian origin.** The frame centre (wlat, wlon) was the median
  epicentre rounded to 0.001 deg; in f0399 one event sat exactly there and in one synthetic another sat 3 m away,
  and the ray tracer returned NaN. `b5_common.frame_centre` now keeps every event at least 50 m from the centre.
  Any B5 driver needs the same guard.
* **The B4 catalog files are rounded.** `catalog_b4_b4_iter1_v0.3.0.csv` (b4_catalog.py line 51) and
  `catalog_v5_trackA_b4_iter1_v0.3.0.csv` (a2_build_catalog.py in `~/wt_reloc`, line 89) are written with
  `float_format="%.5g"`: longitude to 0.01 deg (0.8 km at 44 N), latitude to 0.001 deg (0.11 km). The f1803 start
  had 27 distinct longitudes for 641 events before the fix. The test now reads the run's NonLinLoc solutions
  (`runs/b4_iter1_v0.3.0/locations.csv`, status LOCATED = the 157,270 3D events, 6 decimals), checked to agree with
  the catalog within the rounding. The atlas's B4 layer was built from this catalog (STATUS 2026-11-24 05:10), so it
  probably carries the same rounding; check it. The catalog should be rewritten with `%.8g` before B5 or any
  further use. Runs made with the rounded start are kept in
  `/wd1/mdenolle_data/b5_test/_superseded_rounded_start/` and are not used here.

## 4. Real swarms

Each code on the events it relocated ("own"); differences on the events every run relocated ("common").
Depth in km below sea level.

| case | events (B4 3D) / stations / picks | dt pairs / dt / events linked | relocated: tomoDD hier, dd; GrowClust3D catalog, default | common |
|---|---|---|---|---|
| f1803 West Blanco 2015-06 (swarm, M 5.9) | 641 / 12 / 5,467 | 1,093 / 9,458 / 399 | 636, 397; 379, 336 | 325 |
| f0399 Blanco 2012-10 (swarm, M 5.0) | 1,455 / 33 / 19,659 | 4,446 / 50,927 / 1,103 | 1,447, 1,088; 1,077, 930 | 915 |
| f1051 Blanco 2014-03 (M 5.3 sequence) | 1,132 / 25 / 11,661 | 2,492 / 26,134 / 685 | 1,120, 685; 681, 639 | 627 |

In f1803, 219 of 641 events have fewer than 8 picks and can never meet the 8 common station-phase rule; 114 more
events of the family (tier U) have no B4 location and are not in the test. tomoDD `hier` keeps unlinked events
through their absolute times; `dd` and GrowClust3D relocate only linked events (GrowClust3D drops more, because a
cluster merge must pass its rms, shift and distance limits).

**f1803, shape and position.**

| run | n | centroid shift from B4 h / z (km) | PCA sigma1 / sigma2 / sigma3 (km) | strike / dip (deg) | depth p10 / median / p90 |
|---|---|---|---|---|---|
| B4 start | 641 | - | 15.0 / 7.3 / 1.9 | 112 / 84 | 5.7 / 19.0 / 32.6 |
| tomoDD hier | 636 | 0.44 / +0.56 | 15.3 / 7.7 / 2.0 | 111 / 84 | 6.0 / 20.2 / 33.4 |
| tomoDD dd | 397 | 0.02 / +0.10 | 14.3 / 6.7 / 1.7 | 111 / 84 | 6.2 / 19.6 / 31.8 |
| GrowClust3D catalog | 379 | 0 (fixed) | 13.9 / 6.4 / 1.4 | 110 / 84 | 5.9 / 19.6 / 31.8 |
| GrowClust3D default | 336 | 0 (fixed) | 13.6 / 5.9 / 1.1 | 110 / 84 | 6.3 / 19.2 / 30.9 |

**f1803, common set (325 events).** Median (90th percentile) position difference with centroids removed,
horizontal / vertical, and median |difference| of the 52,650 event-pair distances:

| pair | horizontal | vertical | pair distance |
|---|---|---|---|
| B4 vs tomoDD dd | 0.42 (0.96) | 0.28 (1.50) | 0.48 |
| B4 vs GrowClust3D catalog | 0.88 (2.21) | 0.93 (2.58) | 1.17 |
| B4 vs GrowClust3D default | 0.69 (1.49) | 0.70 (1.71) | 0.87 |
| tomoDD dd vs GrowClust3D catalog | 1.07 (2.37) | 0.99 (2.85) | 1.31 |
| tomoDD dd vs GrowClust3D default | 0.88 (1.69) | 0.77 (2.48) | 1.03 |
| tomoDD hier vs tomoDD dd | 0.45 (1.06) | 0.68 (1.92) | 0.84 |

Bootstrap spread (median per event, horizontal / vertical, km): tomoDD hier 0.14 / 0.07, tomoDD dd 0.30 / 0.25,
GrowClust3D catalog 1.98 / 2.54 (2.07 / 2.65 counting only resamples in which the event was relocated), GrowClust3D
default 1.03 / 1.14. LSQR errors of tomoDD: 0.11-0.14 km. dt residuals at each solution in the B4 grids (per-pair
median removed, common set, rms P / S in s): B4 0.124 / 0.179, tomoDD hier 0.134 / 0.256, tomoDD dd 0.122 / 0.234,
GrowClust3D catalog 0.138 / 0.172, default 0.136 / 0.171. The B4 start already fits the differential times as well
as either code: catalog differential times are differences of the picks B4 located with, so they add little.

Both codes keep the B4 depth structure: a group at 5-10 km under the east half, one at 20-26 km near the centre, a
tail to 40 km in the west (`f1803/sections.png`). The synthetic test (section 5) shows that a 5 km tall plane
relocated with these stations spreads over the same 6-29 km range, so the depth extent of the real swarm is not
resolved by either code.

**f0399 (2012-10, the Blanco OBS array, 33 stations, median 12 picks).** Better geometry, same picture. tomoDD dd
and GrowClust3D catalog: 0.90 km horizontal and 1.63 km vertical apart (median, common 915 events). Strike 131 deg
(tomoDD dd), 140 deg (GrowClust3D catalog), 132 deg (default), 130 deg (B4 on the common set). Depth p10-p90 within
5.6-23.4 km for every run (B4 7.0-21.6 km). Bootstrap spread: tomoDD dd 0.48 / 0.92 km, GrowClust3D catalog
1.56 / 2.38 km, default 1.23 / 1.42 km. dt rms P / S: B4 0.155 / 0.202 s, tomoDD dd 0.155 / 0.289 s, GrowClust3D catalog 0.164 / 0.194 s.

**f1051 (2014-03, 25 stations).** The linked events form a shallow cluster: depth p10-p90 3.7-7.7 km for B4 and every
differential run on the common set (627 events). tomoDD dd and GrowClust3D catalog: 0.96 km horizontal and 0.63 km
vertical apart (median); tomoDD dd and GrowClust3D default: 0.71 km and 0.65 km. Strike 126 deg (tomoDD dd), 122 deg
(GrowClust3D catalog), 125 deg (default), 125 deg (B4). Bootstrap spread: tomoDD dd 0.29 / 0.14 km, GrowClust3D
catalog 1.68 / 1.77 km, default 1.19 / 1.04 km. dt rms P / S: B4 0.122 / 0.165 s, tomoDD dd 0.121 / 0.205 s,
GrowClust3D catalog 0.131 / 0.154 s. tomoDD hier moves the common-set centroid 1.0 km east and 0.9 km up.

Across the three swarms the two codes differ by 0.7-1.1 km horizontally and 0.6-1.6 km vertically (median), and each
moves events 0.4-1.0 km from B4. Those differences are below GrowClust3D's bootstrap spread and above tomoDD's.

tomoDD `hier` moves the centroid by 0.4-1.0 km horizontally and up to 0.6 km vertically: its absolute times are
fitted in tomoDD's own forward model (pseudo-bending in the node model, water replaced by rock), which differs from the B4
grids even after the P-4 corrections; its S residuals in the B4 grids are the largest of all runs (0.26 s in f1803,
0.41 s in f0399). `dd` is the hypoDD-like variant; its centroid stays within 0.1 km of B4 in f1803 and moves
0.6 km down in f0399 (LSQR leaves the mean position free).

## 5. Synthetic planar swarm

Truth: 200 events on a vertical plane, 10 km x 5 km, strike 111 deg, depth 16.5-21.5 km (sigma 2.9 / 1.5 / 0 km).
Same 12 stations and pick pattern as f1803. Relative errors are against the truth with each solution's centroid
removed (median, own relocated set); "pair" is the median |error| of event-pair distances; "plane" is the rms
distance from the true plane (across strike).

| case | run | n | relative h | relative z | pair | plane | depth p10-p90 |
|---|---|---|---|---|---|---|---|
| synth (B4-sized start) | start | 200 | 2.87 | 6.32 | 8.09 | 4.04 | 6.4-29.1 |
| | tomoDD hier | 194 | 2.02 | 6.81 | 7.18 | 1.70 | 6.8-29.6 |
| | tomoDD dd | 118 | 2.32 | 7.42 | 7.85 | 2.24 | 6.7-28.8 |
| | GrowClust3D catalog | 103 | 1.40 | 7.49 | 5.39 | 1.11 | 8.7-28.6 |
| | GrowClust3D default | 56 | 2.23 | 6.93 | 6.52 | 1.89 | 5.4-25.1 |
| synth_tdd (tomoDD forward) | tomoDD hier | 195 | 2.02 | 6.85 | 7.27 | 1.70 | 6.7-29.5 |
| | tomoDD dd | 118 | 2.49 | 7.33 | 7.83 | 2.20 | 6.7-28.8 |
| | GrowClust3D catalog | 107 | 1.76 | 7.31 | 6.69 | 1.23 | 7.6-29.4 |
| | GrowClust3D default | 50 | 1.54 | 7.78 | 5.58 | 1.47 | 6.0-26.0 |
| synth_s025 (start x 0.25) | start | 200 | 0.72 | 1.60 | 1.24 | 1.01 | 15.3-22.1 |
| | tomoDD hier | 200 | 0.78 | 1.68 | 1.63 | 0.60 | 12.8-23.1 |
| | tomoDD dd | 124 | 0.75 | 1.69 | 1.43 | 0.66 | 14.2-22.6 |
| | GrowClust3D catalog | 112 | 1.39 | 2.75 | 2.48 | 0.95 | 13.7-24.2 |
| | GrowClust3D default | 80 | 1.00 | 1.91 | 1.53 | 0.54 | 15.7-22.2 |

The true depth range is 17.1-20.9 km (p10-p90). What the table says:

1. Depth is not resolved. From a B4-sized start the relative depth error does not drop in any run (6.8-7.8 km
   against 6.3 km). With 4-6 OBS at 27-250 km and S picks of 0.1-0.3 s error, the differential times do not
   constrain relative depth. This holds for both codes and both forward models.
2. Both codes thin the cloud across strike (plane rms 4.0 km -> 1.1-2.2 km) and recover the strike within 6 deg
   from the poor start (GrowClust3D catalog is 9 deg off from the good start). GrowClust3D catalog does best
   horizontally from a poor start (1.4 km), but relocates only half of the events.
3. From a good start no code helps. tomoDD changes little (damped least squares). GrowClust3D moves whole clusters by
   grid search with an L1 misfit; with catalog-quality times its merges fit pick noise (S rms at its solution
   0.206 s, below the 0.256 s of the truth) and move events away from the truth.
4. The forward model matters little at this level. Swapping it (synth_tdd: data from tomoDD's ray tracer) changes the
   relative errors by 0.0-0.7 km, less than the differences between runs.
5. GrowClust3D is unstable. Between an earlier run and this one the synthetic plane moved by about 90 m (its
   centre is the f1803 median, which changed with the unrounded start); tomoDD hier gave 2.02 km both times,
   GrowClust3D catalog 1.78 km then 1.40 km. In f0399 the rounded and the unrounded start gave a GrowClust3D
   catalog strike of 129 deg then 140 deg (tomoDD dd 129 then 131 deg).

**Bootstrap against true error** (synth_s025, median, horizontal / vertical, km): tomoDD dd 0.38 / 0.63 against
0.75 / 1.69; GrowClust3D catalog 2.23 / 3.33 against 1.39 / 2.75; GrowClust3D default 1.07 / 1.23 against 1.00 / 1.91.
In synth: tomoDD dd 0.73 / 0.68 against 2.32 / 7.42; GrowClust3D catalog 2.38 / 2.85 against 1.40 / 7.49.
tomoDD's bootstrap underestimates its error 2-11 times. GrowClust3D's bootstrap covers its horizontal error but
not its depth error. Neither is usable as a B5 location uncertainty without calibration.

## 6. Which code is more accurate, and why

On these data neither. tomoDD (relocation only, damped LSQR on all pairs at once) is stable and neutral: it moves
events little from B4 and does not damage a good start. GrowClust3D (hierarchical cluster merging, grid search, L1)
moves events more, which helps from a poor start horizontally and hurts from a good start, and its answer depends on
the merge order, so it changes with resampling and with small input changes. The real-swarm differences between the
two (about 1 km) are the size of the synthetic errors, so the real swarms cannot say which is right. The PI's doubt is
borne out for catalog times: GrowClust3D is not more accurate than the hypoDD-type solution.

What would change this: waveform cross-correlation times (D-B040) with 0.01-0.05 s errors are what GrowClust3D's
limits (rmsmax 0.2 s, rmedmax 0.1 s, shifts 2 km) were set for; and a denser network constrains depth. Neither is in
this test.

## 7. What did not work, and limits of the test

* No waveform cross-correlation times; catalog times only (no waveforms downloaded, D-B040 not yet run).
* tomoDD relocation-only mode needed a source fix (section 3); the unfixed `_fmt` builds cannot run it.
* Only 61 % (f1051) to 76 % (f0399) of the B4 events are linked with 8 common station-phases; the rest cannot be
  relocated by either code with these settings. Tier U events (114 in f1803) have no B4 start and were left out.
* Bootstrap of tomoDD resamples dt rows only; the absolute times of `hier` are fixed, so its spread is a lower bound.
* GrowClust3D's misfit has no data weights: the pick weights only order the merges. tomoDD weights every datum.
  "Same weights" is therefore only true for the ordering, not the misfit.
* The two codes use different forward models of the same velocity model (B4 Podvin-Lecomte grids in the EFT frame;
  tomoDD pseudo-bending on the sphere in the node model, rock in the water column). synth_tdd bounds the effect.
* The Blanco swarms are offshore with 4-33 OBS. Most of the B5 catalog is onshore with denser networks; this test
  does not cover that case.
* GrowClust3D parameters were set by reasoning, not tuned. tomoDD damping 100 gives CND 7-10 (synthetics), 15-22
  (f1803), 29-37 (f1051), 42-59 (f0399): below the hypoDD guide of 40-80 for the small cases, so more damping than
  needed. The last of the 12 iterations still moves events by 14-56 m on average in f1803 and up to 148 m in depth in
  f0399: close to, not fully at, convergence.

## 8. Files

* Code: `workflow/10_tomography/b5_common.py`, `b5_prep.py`, `b5_tomodd.py`, `b5_growclust.py`,
  `b5_growclust_run.jl`, `b5_synth.py`, `b5_compare.py`, `b5_summary.py`, `b5_run.sh`; parameters `params.yaml`
  `b5_test`.
* Results (git): `data/tomo_v1/b5_test/summary.md` (all tables), `<case>/metrics.json`, `shape.csv`,
  `pair_distances.csv`, `relative_positions.csv`, `uncertainty.csv`, `dt_misfit.csv`, `synthetic_errors.csv`,
  `boot_*.csv`, `sections.png` (map and two sections per run), `depth_pairs.png`, manifests.
* Runs (not in git): `/wd1/mdenolle_data/b5_test/<case>/` (data, tomodd/<variant>[_bNNN], growclust/<variant>),
  logs `run.log`, `run_other.log`; Julia environment `julia_env/`, depot `julia_depot/`; fixed tomoDD build `bin/`,
  `src/`.
* Rerun (resumable, detached): `setsid nohup bash workflow/10_tomography/b5_run.sh f1803 synth synth_tdd synth_s025
  f0399 f1051 >> /wd1/mdenolle_data/b5_test/run.log 2>&1 < /dev/null &`, then `python b5_summary.py`.


# B5 weighting test: which S weight and re-weighting (PI 2026-11-24)

Follows notes/b5_s_mismatch.md (S misfit growth = S weight 1/4 of P + MAD re-weighting). PI: "test both options to
identify the optimal weighting mechanism". Code b5_wtest.py (setup, base, run, collect), b5_wtest_report.py,
b5_wtest_paired.py; params b5_wtest. Results /wd1/mdenolle_data/tomo_v1/b5/b5_v0.3.1/diag/weights (2,340 task results,
0 failures); summaries in data/tomo_v1/b5/b5_v0.3.1/weights/ (report.md, variant_scores.csv, per_cluster.csv,
paired_s1.0_prod.md, clusters.csv).

## Design

* 26 clusters of run b5_v0.3.1: the three largest unsplit clusters with 80-1,500 events in every region (two in
  Explorer, one in JdF ridge/plate) and two 1,200-event subsets of the Axial chunks (nearest core events to each
  chunk's median epicentre). Production inputs reused; only weights, dt rows or data/start change.
* Variants: S weight relative to P (same pick class) 0.25 (production: D-B008 sigma S = 2 x P), 0.5, meas (1/r^2, r =
  region median of the S/P MAD of B4 dt residuals: 1.02 Cascades to 1.69 Explorer), 1.0, 1.5, 2.0; times
  re-weighting prod (WRCT 8 then 6 x MAD, WDCT 10 km), loose (12 then 10), none. 18 variants.
* Criteria, per cluster:
  1. station cross-validation (3 folds): relocate without one third of the stations, score the held-out dt rows
     after removing each pair's training-row median; relative to production in the same cluster.
  2. synthetic accuracy: truth = B4 positions, times from the B4 grids, noise resampled per phase from the cluster's
     B4 dt residuals / sqrt(2) (heavy tails kept), origin-time error 0.05 s, start = truth + 0.5 x B4 sigma;
     centroid-removed relative error after / before.
  3. in-sample dt rms at the final positions / at B4; events above the surface; air quakes.
  Paired differences against s1.0_prod with 90 % bootstrap intervals over clusters (b5_wtest_paired.py).

## Results (medians over 26 clusters)

| variant | held-out P / S vs production | both phases better | synthetic h / z (after/before) | in-sample P / S vs B4 |
|---|---|---|---|---|
| s0.25_prod (production) | 1.000 / 1.000 | - | 0.769 / 0.839 | 0.955 / 1.181 |
| s0.5_prod | 0.968 / 0.947 | 25/26 | 0.714 / 0.807 | 0.972 / 1.060 |
| smeas_prod | 0.961 / 0.938 | 25/26 | 0.708 / 0.805 | 0.967 / 1.030 |
| **s1.0_prod** | **0.943 / 0.889** | **24/26** | **0.668 / 0.803** | **1.015 / 0.974** |
| s1.5_prod | 0.945 / 0.888 | 25/26 | 0.681 / 0.774 | 1.043 / 0.935 |
| s2.0_prod | 0.944 / 0.886 | 24/26 | 0.685 / 0.778 | 1.052 / 0.923 |
| s1.0_loose | 0.951 / 0.900 | 24/26 | 0.663 / 0.808 | 1.001 / 0.947 |
| s1.0_none | 0.963 / 0.926 | 21/26 | 0.654 / 0.775 | 0.995 / 0.906 |

Paired against s1.0_prod (median difference [90 % interval], clusters better):
* production s0.25_prod: held-out +0.080 [+0.063, +0.102] (1/26), synthetic h +0.097 [+0.079, +0.120], z +0.093
  [+0.012, +0.113]: clearly worse on every criterion except in-sample P.
* smeas_prod: held-out +0.025 [+0.016, +0.029] (3/26), synthetic h +0.018: inverse-variance weights from the measured
  ratio are worse than equal weights. The measured ratio is inflated by B4 location error, which moves S times
  ~Vp/Vs times more than P.
* s1.5_prod: held-out -0.005 [-0.011, -0.001] (17/26), synthetic z -0.011 [-0.027, -0.000], synthetic h +0.006
  (n.s.), in-sample P +0.024 in 26/26 clusters. s2.0_prod: held-out -0.008, synthetic h +0.012 [+0.001, +0.030]
  (worse), in-sample P +0.041. The optimum is flat between 1.0 and 2.0 (differences <= 1 %).
* re-weighting: none is worse on held-out data at every S weight (+0.017 to +0.020), loose slightly worse (+0.006).
  The production MAD cuts are kept.

By region (s1.0_prod against production): held-out P 0.86-0.99 and S 0.85-0.96 in all ten regions; synthetic better
everywhere except Explorer (flat). Cascades is the exception in absolute terms: synthetic relative error grows with
every variant (s1.0_prod h 1.06, z 1.28; production 1.38, 1.62) because B4's onshore start errors are already small.

Not changed by the weights: events above the surface (166-178 of the ~19,000 test events in every variant: JdF 55,
other onshore 15-17, Puget 9-10, Cascades 3-5): the upward-pull problem (D-B056, round 2) is independent of the
weighting. Air quakes (events tomoDD removes above its Air_dep level, counted from tomoDD-SE.log as
b5_relocate.py does; a first version of this note said 0 because the test parsed the wrong stdout column): 61 in the
full runs at production weights, 35 at S weight 1.0 (Cascades 21 -> 14, offshore forearc 19 -> 9, Mendocino 13 -> 9,
Axial 4 -> 0), 47 at 0.5 and 46 at the measured ratio: weighting S like P also removes fewer events.

Caveats: the held-out score against B4 stays above 1 (held-out P 1.11, S 1.23 for s1.0_prod) because B4 located
every event with all stations' absolute picks, including the held-out ones; the test ranks B5 variants, it does not
show that B5 beats B4 on real data. The synthetic test has no velocity-model error.

## Proposed decisions (for the PI; not in DECISIONS.md)

* P-W1: B5 weights S differential times like P differential times of the same pick class (S weight 1.0: drop the
  factor 1/4 of D-B008 in dt.ct, or WTCTS = 4 with the present dt.ct). 1.5 is equally good within the test's
  resolution (slightly better held-out, worse in-sample P); 1.0 is chosen as the simplest rule with the best
  synthetic horizontal accuracy.
* P-W2: keep the production MAD re-weighting (WRCT 8 then 6, WDCT 10 km).
* P-W3: the "measured S/P ratio" weighting is rejected (worse on every criterion than equal weights).
* P-W4: B5 acceptance (replaces P-S2): held-out dt rms by station cross-validation must not exceed production's in
  either phase, by region; above-surface events reported, never moved (D-B055). In-sample misfit vs B4 is not a
  criterion (B4 fitted the same picks).
* Rerun B5 with P-W1/P-W2 after v5.1 (round-2 NLLoc fixes), not on v5.0.

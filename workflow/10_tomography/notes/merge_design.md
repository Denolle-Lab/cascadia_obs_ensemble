# Merging the tiles into one volume, and testing the tiles against each other

Proposal, 2026-11-23 (PI question: "how do we tile them together if they have different damping parameters, how
will we merge the models into a consistent volume?"). Implements the part of D-B025 left open ("how the overlap
zones are compared or merged"). Nothing here is adopted; decisions to take are P-M1..P-M8 in section 6.
Code: `b2_tile_overlap.py` (tests T1, T2, T5 on the deterministic runs), params `tomodd.merge`. Results:
`data/tomo_v1/b2_merge/overlap/` (summary.md, one folder per pair with depth_stats.csv, events.csv,
summary.json, overlap.png).

## 1. Short answer

The tiles do not need the same damping number. They need the same rule for choosing it, one reference model, one
output grid, weights that follow resolution and uncertainty, and an ensemble merged member by member so that the
posterior spread is honest across the seams. Tests of the tiles against each other (section 4) check that the
merged volume has no seam that the data do not support.

## 2. Damping: what has to be the same

LSQR damping in tomoDD is weighed against the size and weights of each tile's own system, so the same number
regularises a small tile much more than a large one. The condition number (CND) of the damped system is the usual
tomoDD/hypoDD guide (aim for roughly 40-80). Compared at the same point of the inversion, real data, smoothing 20:

| tile | first-iteration CND at damping 100 / 200 / 300 / 500 / 1000 / 2000 | last-iteration CND at 1000 | damping for first-iteration CND ~60 |
|---|---|---|---|
| mendocino_gorda_v2 (old dataset) | - / - / - / 153 / 60 / 25 | 19 | 1000 (the benchmark) |
| mendocino_gorda_c_v2 | - / - / - / 161 / 65 / 26 | 19 | ~1050 |
| washington_v2 | - / - / - / 168 / 68 / 27 | 25 | ~1100 |
| oregon_v2 | - / 347 / 200 / 102 / 39 / 17 | 14 | ~730 |
| blanco_v2 | - / 284 / 170 / 85 / 35 / 15 | 11 | ~660 |
| ridges_north_v2 | 290 / 119 / 69 / 37 / 15 / 8 | 6 | ~340 (PI chose 200, D-B043) |

(Interpolated linearly in log CND vs log damping between scanned values.)

**Correction.** The Mendocino-Gorda benchmark "CND ~60" (D-B028, D-B030) is the *first*-iteration value. The
09:05 STATUS update, science_framing.md section 4 and the 10:00 status reply compared the *last*-iteration CND of
Blanco and Oregon (11, 14 at damping 1000; 52, 74 at 300) with that 60, and so called damping 300 "about right".
Like for like, damping 300 gives Blanco and Oregon about three times the benchmark CND (170, 200), and damping 1000
gives them about 0.6 of it. Damping 1000 is less over-damped for them than stated, and 300 is weaker than
Mendocino-Gorda's regularisation.

The CND rule is a numerical guide, not a resolution criterion. Lower damping does enlarge the checkerboard masks
(Blanco Vp at 15 km: 9,500 km2 at 1000, 19,800 km2 at 300; Vs 1,550 and 13,500 km2) but also moves unresolved
nodes and hypocentres more, and makes tiles disagree more in their overlaps (section 5). P-M1 asks for one rule.

## 3. How the tiles are merged

M1. **One reference model.** Each tile's starting model is the D-B005 3D model sampled at that tile's nodes
(nearest column, D-B029). Where nodes of two tiles fall on different columns the starting models differ: median
below 0.4 % at every depth in every overlap, but up to 12-30 % at single nodes in the top 5 km (steep bathymetry,
water against crust) and up to 3 % below 9 km (T5, section 5). Merging absolute velocities would carry those steps
into the merged model. Instead the merge combines each tile's change from its own start,
dlnV_k = V_k / V0_k - 1, and applies the weighted change to one reference: the D-B005 model sampled once on the
common grid, V(x) = V_ref(x) (1 + sum_k w_k(x) dlnV_k(x)). Vp and Vs separately.

M2. **One output grid.** The tile node grids do not line up: each tile's longitude nodes start at its own box edge
(offsets up to 0.05 deg, ~4 km), each tile has its own D-B028 coordinate rescale, and only the depth nodes are
shared. The merge grid is a geographic lattice of 0.125 deg x 0.1 deg (multiples of the step) on the common depth
nodes. Each tile is evaluated there with tomoDD's own trilinear lookup on its back-transformed nodes (the
back-transform is affine per axis, so this is the interpolation the tile's travel times used).

M3. **Weights.** w_k(x) = t_k(x) r_k(x) / s_k(x)^2, normalised over the tiles covering x.
* t_k, taper: sin^2(pi/2 min(1, d_k/W)), d_k = distance from x to the nearest edge of tile k's events box that
  faces another tile, W = width of that overlap (48-89 km, about 5-8 nodes). Across an overlap of two tiles the two
  tapers sum to 1. Nodes outside a tile's events box (its coarse 0.75 / 0.5 deg nodes) get t_k = 0.
* r_k, resolution: 1 inside tile k's resolved mask, a small value outside it (to be set from T3; 0.1 as a
  placeholder). A tile that resolves a node then outweighs one that does not, and unresolved nodes are still
  blended smoothly for relocation.
* s_k, posterior standard deviation from tile k's ensemble (before the ensemble exists: s_k = 1).
* No tile covers x (gaps between boxes, e.g. ~15 km between washington and ridges_north; outside all boxes):
  V = V_ref.
Two products come from the same weights: the interpretation volume (posterior statistics shown only inside the
union of the tile masks, D-B037) and the relocation model for B4 (every node).

M4. **Merge member by member.** The random inputs of the D-B030 ensemble are drawn once for the whole region, not
per tile: the starting-model random field on one regional lattice (sampled at each tile's nodes), station statics
by station code (a station in two tiles gets the same static), pick noise by pick ID (a pick in two tiles gets
the same noise), starting hypocentres by event ID. Member m of every tile then shares the same prior draw, member m
of the merged volume is one coherent model, and mean, std, percentiles and Vp/Vs (computed per member, then
summarised) are taken over the merged members. Damping and smoothing are drawn per tile (M5).

M5. **Damping prior per tile.** D-B030 item 7 draws damping log-uniform on [700, 1500], set from Mendocino-Gorda.
Per tile it becomes log-uniform on [0.7, 1.5] x that tile's chosen damping, so the regularisation uncertainty
ends up in each tile's spread instead of hidden in a seam.

M6. **Events.** The tomoDD relocations of each tile are intermediate. B4 relocates all 172,342 events with
NonLinLoc in 10-20 merged member models (D-B039), which gives one catalog and location uncertainties that include
the choice of tile.

## 4. Tests of the tiles against each other

T1. **Model agreement in the overlaps** (`b2_tile_overlap.py`, runs now on deterministic runs). On the common grid
inside the intersection of two events boxes, per depth, in three classes of the tiles' checkerboard masks:
resolved in both, in one, in neither. Statistics: rms of each tile's change, mean and rms of the difference,
correlation, amplitude ratio. Pass rule at the ensemble stage: on nodes resolved in both,
|mean_A - mean_B| / sqrt(s_A^2 + s_B^2) <= 2 for at least 90 % of nodes and no mean offset above 0.5 of the
combined std. Deterministic stage: report only.

T2. **Same events in two tiles.** Events relocated in both tiles (1,980 B1b events fall in two tiles): horizontal
and depth difference of the two final locations, against each tile's shift from the common start. Pass rule at
the ensemble stage: at least 90 % of events within 2 combined standard deviations of the member locations.

T3. **Synthetic across the seams.** The D-B030 synthetic calibration uses one truth for the whole region (one
random field and one set of statics, M4), run through every tile's ensemble and merged. Interval coverage
(68 / 95 %) is scored separately in the overlaps and in the tile interiors. This is the only test of the merge
where the truth is known, and the only one that works where no overlap node is resolved by both tiles (section 5).
It also sets r_k (M3).

T4. **Seams after B4.** Event residuals and NonLinLoc station terms against distance to the nearest seam: no step
at the seams beyond the scatter elsewhere.

T5. **Starting models.** Difference of the two tiles' starting models on the common grid (`b2_tile_overlap.py`,
column "start"). Explains why M1 merges changes, not absolute velocities.

## 5. First results (deterministic real-data runs, smoothing 20, 2026-11-23)

Pairs and dampings run (params `tomodd.merge.pairs`): blanco-oregon, mendocino_gorda_c-blanco,
mendocino_gorda_c-oregon, blanco-ridges_north, oregon-washington, at matched and mixed dampings. Full tables in
`data/tomo_v1/b2_merge/overlap/summary.md`. Vp, 1-34 km:

| pair (A / B) | dampings | nodes | rms change A / B (%) | rms difference (%) | correlation | amplitude B/A |
|---|---|---|---|---|---|---|
| blanco / oregon | 1000 / 1000 | 1,620 neither | 1.4 / 2.2 | 2.0 | 0.40 | 0.74 |
| blanco / oregon | 500 / 500 | 1,620 (no mask) | 1.7 / 2.7 | 2.6 | 0.33 | 0.59 |
| blanco / oregon | 300 / 300 | 1,620 neither | 2.0 / 2.9 | 3.0 | 0.29 | 0.49 |
| blanco / oregon | 300 / 1000 | 1,620 neither | 2.0 / 2.2 | 2.4 | 0.32 | 0.42 |
| blanco / oregon | 1000 / 300 | 1,620 neither | 1.4 / 2.9 | 2.7 | 0.34 | 0.84 |
| mendocino_gorda_c / blanco | 1000 / 1000 | 1,107 neither | 5.1 / 2.5 | 4.5 | 0.42 | 0.23 |
| mendocino_gorda_c / blanco | 1000 / 1000 | 44 one | 8.3 / 4.0 | 6.7 | 0.48 | 0.29 |
| mendocino_gorda_c / blanco | 500 / 500 | 1,152 (no mask) | 6.2 / 3.4 | 5.6 | 0.42 | 0.25 |
| mendocino_gorda_c / blanco | 1000 / 300 | 1,084 neither | 5.0 / 3.8 | 4.9 | 0.39 | 0.32 |
| mendocino_gorda_c / oregon | 1000 / 1000 | 2,154 neither | 2.9 / 2.4 | 3.1 | 0.29 | 0.27 |
| mendocino_gorda_c / oregon | 500 / 500 | 2,160 (no mask) | 3.6 / 3.1 | 3.9 | 0.31 | 0.29 |
| mendocino_gorda_c / oregon | 1000 / 300 | 2,154 neither | 2.9 / 3.6 | 3.8 | 0.29 | 0.39 |
| blanco / ridges_north | 1000 / 1000 | 3,442 neither | 2.8 / 2.3 | 2.4 | 0.52 | 0.45 |
| blanco / ridges_north | 300 / 200 | 3,435 neither | 3.7 / 3.2 | 3.5 | 0.48 | 0.44 |
| blanco / ridges_north | 1000 / 200 | 3,442 neither | 2.8 / 3.2 | 3.0 | 0.49 | 0.59 |
| oregon / washington | 1000 / 1000 | 3,528 (no mask yet) | 3.6 / 5.0 | 4.3 | 0.53 | 0.75 |
| oregon / washington | 300 / 1000 | 3,528 (no mask yet) | 5.0 / 5.0 | 5.2 | 0.45 | 0.45 |

Same events relocated in two tiles (events in the overlap box):

| pair | dampings | events | horizontal A-B median / p90 (km) | depth A-B abs median / p90 (km) | each tile's shift from start, horizontal (km) |
|---|---|---|---|---|---|
| blanco / oregon | 1000 / 1000 | 77 | 0.63 / 1.84 | 0.36 / 2.45 | 0.73 / 0.51 |
| blanco / oregon | 300 / 300 | 77 | 1.00 / 2.61 | 1.37 / 3.99 | 1.26 / 1.01 |
| mendocino_gorda_c / blanco | 1000 / 1000 | 103 | 0.36 / 0.91 | 0.38 / 1.62 | 0.64 / 0.56 |
| mendocino_gorda_c / blanco | 1000 / 300 | 103 | 0.66 / 1.10 | 0.96 / 3.33 | 0.64 / 0.94 |
| mendocino_gorda_c / oregon | 1000 / 1000 | 32 | 0.37 / 0.83 | 0.58 / 2.00 | 0.47 / 0.34 |
| blanco / ridges_north | 1000 / 200 | 281 | 0.33 / 1.06 | 0.33 / 1.17 | 0.49 / 0.63 |
| oregon / washington | 1000 / 1000 | 1,568 | 0.21 / 0.67 | 0.20 / 0.69 | 0.43 / 0.54 |
| oregon / washington | 300 / 1000 | 1,562 | 0.33 / 1.06 | 0.40 / 1.33 | 0.70 / 0.54 |

What this says:

1. **No overlap is resolved by both tiles.** At most 2 nodes (mendocino_gorda_c / blanco) are inside both masks
   in any pair; oregon / washington waits for the washington mask (pipeline still running). The boxes were cut
   where events are few, so the overlaps sit where coverage is poor. The merge is therefore in practice a choice
   of the resolving tile with a taper, not an average of two resolved estimates, and T1's pass rule cannot be
   applied yet. T3 carries the test of the merge.
2. **Mixing dampings does not make a worse seam.** In every pair the mixed-damping difference lies between the two
   matched cases (blanco / oregon: 2.0 % at 1000/1000, 3.0 % at 300/300, 2.4-2.7 % mixed). What sets the
   disagreement is how far each tile moves unresolved nodes, which grows as damping weakens.
3. **Unresolved nodes disagree at the level of the signal.** In the overlaps the two tiles change the same nodes by
   1.4-6 % rms with correlation 0.3-0.5. These nodes must not be interpreted (D-B037) and get low weight (M3).
4. **Where one tile resolves, the other sees a fraction of it.** mendocino_gorda_c resolves 44-66 overlap nodes
   that blanco does not; there blanco's change is 0.29-0.38 of mendocino_gorda_c's amplitude. r_k must let the
   resolving tile dominate.
5. **Location choice of tile matters.** The same events relocated in two tiles differ by 0.2-1.0 km horizontally
   and 0.2-1.4 km in depth (median; p90 up to 4 km in depth), as much as each tile moves them from the start, and
   more at weaker damping. Location uncertainty has to include it (M6).
6. **Starting models differ** by up to 30 % at single shallow nodes (T5; blanco / ridges_north at 5 km, 11.9 % at
   -3 to 3 km; oregon / washington 17 % at -3 / -1 km; mendocino_gorda_c / oregon 20 %), median < 0.4 %; below
   9 km at most 3 %. Handled by M1.

## 6. Proposed decisions (for the PI)

* P-M1 One rule for damping in every tile, stated once. Options: (a) first-iteration CND ~60 as Mendocino-Gorda
  (blanco ~660, oregon ~730, washington and mendocino_gorda_c ~1000-1100); (b) the weakest damping at which the
  overlap difference (T1) and the location difference (T2) stay below a set level; (c) per-tile PI choice with the
  numbers of section 2 recorded (as D-B043 for ridges_north).
* P-M2 Merge changes from each tile's own start onto one reference model (M1).
* P-M3 Common geographic grid 0.125 x 0.1 deg on the shared depth nodes, tomoDD trilinear lookup (M2).
* P-M4 Weights = taper x resolution x 1 / posterior variance, normalised (M3); r_k set from T3.
* P-M5 Ensemble draws shared across tiles (regional random field, statics by station, noise by pick, start
  hypocentres by event), merge member by member (M4). Amends notes/bayes_design.md section 2.
* P-M6 Damping prior per tile, log-uniform on [0.7, 1.5] x the chosen damping (M5). Amends D-B030 item 7.
* P-M7 Tests T1-T5 with the pass rules of section 4; T3 becomes part of the D-B030 calibration gate.
* P-M8 Overlaps: keep the D-B033 boxes, or widen mendocino_gorda_c / blanco and oregon / washington so that each
  seam crosses a volume both tiles resolve (more events per tile; check MAXEVE). Decide after the washington mask.

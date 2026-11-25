# Merging the tiles into one volume, and testing the tiles against each other

Proposal, 2026-11-23 (PI questions: "how do we tile them together if they have different damping parameters, how
will we merge the models into a consistent volume?"; "revisit the merging strategies, there's literature on fusing
velocity models of various resolutions"). Implements the part of D-B025 left open ("how the overlap zones are
compared or merged"). Decisions to take are P-M1..P-M10 in section 7; P-M1 and P-M2 were adopted the same day
(D-B045, D-B044). Revised the same day after a literature review (section 3), which changed the reference model
(M0), the weights (M3), the uncertainty (M4) and added two tests (T6, T7).
Code: `b2_tile_overlap.py` (tests T1, T2, T5 on the deterministic runs), params `tomodd.merge`. Results:
`data/tomo_v1/b2_merge/overlap/` (summary.md, one folder per pair with depth_stats.csv, events.csv,
summary.json, overlap.png).

## 1. Short answer

The tiles do not need the same damping number; they need the same rule for choosing it (section 2). The closest
precedent is the California statewide model built with tomoDD in five overlapping subregions (Lin et al. 2010): a
coarse inversion of the whole region first, used as the common starting model; finer subregion inversions with the
outside held at that model; averaging in the overlaps; and a joint inversion of two subregions to check the
stitching. Velocity-model hybridisation (Ajala & Persaud 2021, 2022) adds two lessons: blend only over each model's
well-resolved volume with a smooth window, and check with data that each piece is compatible with its neighbours.
The plan below takes both, keeps our member-by-member ensemble merge, and drops inverse-variance weighting, which
would count the data shared by two tiles twice.

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
nodes and hypocentres more, and makes tiles disagree more in their overlaps (section 6). P-M1 asks for one rule.

## 3. What others do

| approach | examples | how | fits our problem? |
|---|---|---|---|
| Coarse whole-region inversion, then subregions stitched | Lin et al. 2010 (California, tomoDD) | 30 km nodes from absolute times over the whole state as starting model; 5 subregions of 10-20 km nodes overlapping ~30 km; nodes outside a subregion fixed at the coarse model; absolute times of outside events recorded inside added; overlap = average of the two subregions; regularisation chosen per subregion from trade-off curves; check: subregions 1 and 2 inverted jointly, "quite similar" to the stitched model, differences put down to the different regularisation | Yes: same code family, same reason for tiling (memory), same per-tile regularisation. Fixing outside nodes needs the simul2000 fixed-node input, switched off in tomoDD-SE (RaySPDR2.f: `inf=0`, "removed by HZ") |
| Window-function embedding (hybrid models) | Ajala & Persaud 2021, 2022 (SCEC CVMs, UCVM); EPcrust (Molinari & Morelli 2011); Yeh & Ben-Zion windowing (SCEC abstracts) | H = w L + (1 - w) R, w a cosine-tapered window whose support is the well-resolved part of the local model, not its full box; any window shape on any grid | Yes for the weights. Their findings: choosing the well-resolved volume matters as much as tapering; strong tapers usually help; a local model can be "incompatible" with a background model, so misfit has to be checked by subdomain, including outside the embedded volume |
| Learned fusion | sparse dictionary learning (Zhang & Ben-Zion 2024); graph / Markov random field (Zhou, Gerstoft & Olsen 2024) | transfer fine-scale patterns from a high-resolution local model into a coarse regional one; both models change; validated with waveforms or with travel times across the boundaries (Zhou et al.: 38 % less boundary travel-time misfit than Gaussian interpolation) | No for the merge: our tiles have the same node spacing and data type; they differ in coverage, not resolution, and these methods give no uncertainty. Their validation (data across boundaries) is taken into T7 |
| Probabilistic fusion | Gaussian process (Scivier et al. 2024, arXiv; 3D follow-up EGU 2026) | input models as observations of one latent field; disagreement between models becomes predictive variance; samples drawn for simulations | Partly: inputs are treated as exact (no per-model uncertainty, stated by the authors). We keep the idea that disagreement between tiles is variance (M4) |
| Multiscale collaborative model | CSEM (Afanasiev et al. 2015; Fichtner et al. 2018) | regional refinements embedded in a long-wavelength global model; later whole-domain inversions keep regional updates consistent with all data and remove assembly artefacts (from the abstracts) | Yes for the idea: a common long-wavelength model from our own data (M0) and an optional consistency pass (M7) |
| Combining estimates that share data | covariance intersection (Julier & Uhlmann 1997); consensus Monte Carlo (Scott et al. 2016) | consensus Monte Carlo multiplies sub-posteriors from disjoint data subsets; covariance intersection combines estimates whose correlation is unknown, and for one scalar it never returns less than the smaller of the two variances | Our tiles share events (1,980 in two tiles), picks and stations, so they are not disjoint and a product (inverse-variance weights, the first version of M3) would count the shared data twice and make the merged spread too narrow |

## 4. How the tiles are merged (revised)

M0. **Common long-wavelength reference from our data (new; Lin et al. 2010, CSEM).** One coarse tomoDD inversion
of the whole margin from absolute times: nodes ~0.375 x 0.3 deg (~30 km), the shared depth nodes, events
declustered to the compiled limits (MAXEVE 10,000 or the 13k build; one or two best-recorded events per ~10 km
cell), started from the D-B005 model, damping and smoothing from its own trade-off curve. Every tile then restarts
from this model, so the tiles share their long wavelengths and their starting values in the overlaps (removes
the T5 steps, section 6 item 6), and the tile changes carry only finer structure. Cost: one coarse run (to be
measured; hours) and a rerun of each tile's damping scan and resolution tests on the new start (about one day of
the machine at today's load). Fallback without M0: M1.

M1. **One reference model.** If M0 is not run: the D-B005 3D model sampled once on the common grid. Either way the
merge combines each tile's change from its own start, dlnV_k = V_k / V0_k - 1, and applies the blended change to
the reference: V(x) = V_ref(x) (1 + sum_k w_k(x) dlnV_k(x)), Vp and Vs separately. With M0 and identical starts the
two forms coincide.

M2. **One output grid.** Geographic lattice 0.125 x 0.1 deg (multiples of the step) on the shared depth nodes;
each tile evaluated there with tomoDD's trilinear lookup on its back-transformed nodes (unchanged).

M3. **Weights: windows on the resolved volume (revised; Ajala & Persaud 2021).** w_k(x) = t_k(x) u_k(x),
normalised over the tiles covering x.
* t_k, box taper: sin^2(pi/2 min(1, d_k/W)) from the edges of tile k's events box that face another tile, W the
  overlap width (48-89 km); 0 outside the events box. Two tapers sum to 1 across an overlap.
* u_k, resolution window: a cosine-tapered window whose support is tile k's resolved mask (finest checkerboard
  cell > 0 for that phase), taper width one to two node spacings (10-20 km, about one to two spike half-widths),
  floor r (placeholder 0.1, set from T3) outside the mask so unresolved nodes still blend smoothly for relocation.
* No 1/variance factor (it would count shared data twice, section 3). Where several tiles resolve a node the merge
  is their taper-weighted average, as Lin et al. averaged overlaps; where one resolves, it dominates through u_k;
  where none covers x, V = V_ref.
Two products from the same weights: the interpretation volume, shown only inside the union of the tile masks
(D-B037), and the relocation model for B4 (all nodes).

M4. **Ensemble: member-by-member blend plus a disagreement term (revised).** Random inputs are drawn once for the
whole region (random field on one regional lattice, statics by station code, noise by pick ID, start hypocentres by
event ID), so member m of every tile shares its draw; merged member m = the M1/M3 blend of the tiles' members m.
Statistics over the merged members, Vp/Vs per member. In overlaps the merged variance adds the spread between the
tiles' ensemble means, sum_k w_k (mu_k - mu)^2 (variance of a mixture), so that disagreement between tiles shows
as uncertainty (the idea of Gaussian-process fusion) instead of being averaged away.

M5. **Damping prior per tile.** Log-uniform on [0.7, 1.5] x the tile's chosen damping (amends D-B030 item 7).
Lin et al. 2010 also chose regularisation per subregion; T6 measures what this does to the merged model.

M6. **Events.** Tile tomoDD relocations are intermediate; B4 relocates all events with NonLinLoc in 10-20 merged
member models (D-B039). One catalog; location spread includes the choice of tile.

M7. **Consistency pass (optional; CSEM, overlapping domain decomposition).** Each tile restarts from the merged
model and is inverted once more; then merge again. Only if T6 or T7 fail; stop when the overlap differences (T1)
and data misfits (T7) stop changing.

M8. **Outside nodes held at the reference (optional; Lin et al. 2010).** Restore the simul2000 fixed-node input in
tomoDD-SE (nfix set from a node list; RaySPDR2.f has the code, the list is switched off) and hold each tile's
nodes outside its events box at the M0 model, so the tiles cannot put structure into their coarse outer nodes.
This is a source change: waits for the answer on permission to modify tomoDD (STATUS, Waiting). Lin et al. also
added absolute times of events outside a subregion recorded at stations inside it; our builder could do this
without a source change.

## 5. Tests of the tiles against each other

T1. **Model agreement in the overlaps** (`b2_tile_overlap.py`, runs now). On the common grid inside the
intersection of two events boxes, per depth, in three classes of the tiles' masks (resolved in both, in one, in
neither): rms of each tile's change, mean and rms of the difference, correlation, amplitude ratio. Pass rule at the
ensemble stage: on nodes resolved in both, |mean_A - mean_B| / sqrt(s_A^2 + s_B^2) <= 2 for at least 90 % of
nodes and no mean offset above 0.5 of the combined std. Deterministic stage: report only.

T2. **Same events in two tiles** (runs now). Horizontal and depth difference of the two final locations against
each tile's shift from the start. Pass rule at the ensemble stage: at least 90 % of events within 2 combined
standard deviations of the member locations.

T3. **Synthetic across the seams.** The D-B030 calibration with one truth for the whole region (one random field,
one set of statics), through every tile's ensemble and the merge; 68 / 95 % interval coverage scored separately in
overlaps and interiors. Sets the floor r in M3. The only test with a known answer where no overlap node is resolved
by two tiles.

T4. **Seams after B4.** Event residuals and NonLinLoc station terms against distance to the nearest seam: no step
at the seams beyond the scatter elsewhere.

T5. **Starting models** (runs now). Difference of the two tiles' starting models on the common grid; vanishes
with M0.

T6. **Joint against stitched (new; Lin et al. 2010).** Invert two neighbouring tiles as one tile and compare with
the merged model of the two, in the overlap and up to one overlap width on each side. blanco + oregon fits the
compiled limits (7,691 events, 242 stations: the 260-station build; at most 88 stations per event; corrected from 7,355 on 2026-11-23); later mendocino_gorda_c + blanco or oregon +
washington with declustering or a larger build. Pass: differences inside the T3-calibrated uncertainty on resolved
nodes, and the same event locations within T2's tolerance. This is the direct check that tiling and different
damping do not change the answer.

T7. **Data compatibility across seams (new; Ajala & Persaud 2022, Zhou et al. 2024, CSEM).** Forward-compute each
tile's absolute and differential times through the merged model (tomoDD one-iteration forward, as in the D-B028
forward study) and compare the weighted RMS with the tile's own final model, by subdomain: interior, overlap, and
rays that cross a seam. Pass: no subdomain worse by more than a set tolerance (proposed 5 %). A tile whose data fit
the merged model worse than its own model is "incompatible" with its neighbour there.

## 6. First results (deterministic real-data runs, smoothing 20, 2026-11-23)

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

1. **Only the Oregon / Washington overlap is resolved by both tiles.** In the other pairs at most 2 nodes
   (mendocino_gorda_c / blanco) are inside both masks: the boxes were cut where events are few, so those overlaps
   sit where coverage is poor, the merge there is a choice of the resolving tile with a taper, and T3 carries the
   test. *Added 14:15, with the washington mask:* oregon / washington at damping 1000 share 220 nodes (Vp, 1-34 km)
   resolved in both. There the two tiles agree: correlation 0.74, amplitude ratio 0.97, mean difference -0.17 %,
   rms difference 4.0 % against rms changes of 4.8 % and 6.2 %. On the 3,078 nodes neither resolves, correlation
   0.48 and amplitude 0.70. The 1,568 events relocated in both differ by 0.21 km horizontally (median).
2. **Mixing dampings does not make a worse seam.** In every pair the mixed-damping difference lies between the two
   matched cases (blanco / oregon: 2.0 % at 1000/1000, 3.0 % at 300/300, 2.4-2.7 % mixed). What sets the
   disagreement is how far each tile moves unresolved nodes, which grows as damping weakens.
3. **Unresolved nodes disagree at the level of the signal.** In the overlaps the two tiles change the same nodes by
   1.4-6 % rms with correlation 0.3-0.5. These nodes must not be interpreted (D-B037) and get low weight (M3).
4. **Where one tile resolves, the other sees a fraction of it.** mendocino_gorda_c resolves 44-66 overlap nodes
   that blanco does not; there blanco's change is 0.29-0.38 of mendocino_gorda_c's amplitude. The resolution
   window u_k (M3) must let the resolving tile dominate.
5. **Location choice of tile matters.** The same events relocated in two tiles differ by 0.2-1.0 km horizontally
   and 0.2-1.4 km in depth (median; p90 up to 4 km in depth), as much as each tile moves them from the start, and
   more at weaker damping. Location uncertainty has to include it (M6).
6. **Starting models differ** by up to 30 % at single shallow nodes (T5; blanco / ridges_north at 5 km, 11.9 % at
   -3 to 3 km; oregon / washington 17 % at -3 / -1 km; mendocino_gorda_c / oregon 20 %), median < 0.4 %; below
   9 km at most 3 %. Removed by M0 (common start), or handled by M1.
7. **T6, joint against stitched (2026-11-23 15:26; data/tomo_v1/b2_merge/t6/blanco_oregon_v2_d200_s20/).**
   blanco_oregon_v2 (7,649 of 7,691 events relocated) against blanco_v2 + oregon_v2g, damping 200, smoothing 20,
   one weight scale, the D-B005 start. Vp, 1-34 km:
   * Tile interiors (25,000 grid points): correlation 0.86-0.94 (Vs 0.95-0.98), mean difference <= 0.1 %, rms
     difference 1.7-2.0 % (Vs 0.8 %). That is the size of moving one tile from damping 200 to 300 (1.7 % Blanco,
     2.3 % Oregon) and smaller than changing Oregon's weight scale (3.0 %). Tiling does not change the interiors
     by more than one damping step.
   * The joint model's anomalies are about 15 % stronger (amplitude ratio stitched/joint 0.77-0.89). Expected:
     tomoDD scales each column to the number of data rows, so damping weighs less in the joint system (0.81 M
     data against 0.39 M and 0.42 M). One damping number still regularises a larger tile less (D-B045).
   * Overlap (no node resolved by either tile at damping 200): correlation 0.76, stitched amplitude 0.56 of the
     joint. Averaging two tiles that partly disagree halves the anomalies there; rms difference 1.9 %, above the
     damping yardstick (0.7-1.1 %).
   * Events: in the interiors the same event differs by 0.24-0.27 km horizontally and 0.34-0.46 km in depth
     (median), a quarter of each tile's own shift from the start. Events in the overlap differ by 0.8-1.1 km and
     1.1-1.7 km, as much as the tiles move them: a tile sees an overlap event from one side. This supports M6
     (relocate every event in the merged model with all stations; tile relocations are intermediate).
   * Seen on the maps (maps.png): at 12 and 21 km both the joint and the tile models alternate in sign from node
     to node; damping 200 lets node-scale noise through.
   Verdict at the deterministic stage: pass in the interiors; in the overlap the stitched model is weaker than the
   joint one, which matters only for relocation (no resolved nodes there).

## 7. Proposed decisions (for the PI)

* P-M1 One rule for damping in every tile, stated once: (a) first-iteration CND ~60 as Mendocino-Gorda (blanco
  ~660, oregon ~730, washington and mendocino_gorda_c ~1000-1100); (b) the weakest damping at which T1, T2 and T7
  stay within tolerance; (c) per-tile choice with the section 2 numbers recorded (as D-B043).
  **Adopted 2026-11-23 as D-B045** (PI: one rule, small damping for all): damping 200 and smoothing 20 in every
  tile and in M0, with pick weights on one scale for the whole dataset. Found while implementing it: the weights
  were scaled per tile (largest pick weight 3 in each), and tomoDD-SE adds the smoothing rows before it scales
  the G columns, so the per-tile scale changed the data/smoothing/damping balance by up to a factor 3 between
  tiles (Oregon and Washington picks weighed 3 times Blanco's). Section 2's condition numbers were computed with
  those per-tile scales.
* P-M2 Reference: (a) coarse whole-margin inversion as common start for all tiles (M0, recommended; reruns the tile
  scans) or (b) the D-B005 model sampled once (M1). **Adopted 2026-11-23: (a), D-B044** (region margin_m0_v2).
* P-M3 Common geographic grid 0.125 x 0.1 deg on the shared depth nodes, tomoDD trilinear lookup (M2).
* P-M4 Weights = box taper x resolution window, normalised; no inverse-variance factor (M3).
* P-M5 Ensemble draws shared across tiles, member-by-member blend, disagreement added to the variance in overlaps
  (M4). Amends notes/bayes_design.md section 2.
* P-M6 Damping prior per tile, log-uniform on [0.7, 1.5] x the chosen damping (M5). Amends D-B030 item 7.
* P-M7 Tests T1-T7 with the pass rules of section 5; T3, T6 and T7 gate the merged product.
* P-M8 Overlaps: keep the D-B033 boxes, or widen mendocino_gorda_c / blanco and oregon / washington so that each
  seam crosses a volume both tiles resolve. Decide after the washington mask.
* P-M9 Restore fixed nodes in tomoDD-SE and hold outside nodes at the reference (M8): only with permission to
  modify the code.
* P-M10 Consistency pass (M7) only if T6 or T7 fail.
* P-M11 (2026-11-23 16:10) M0 = margin_m0_v2 d200_s480 (smoothing 480/480/240): the smoothest model whose weighted
  RMS stays within 1.5 % of the best of the scan. Tiles then restart from it (`start_model: margin_m0_v2/d200_s480`)
  and each tile's rerun scans smoothing 20-480 at damping 200 and takes its smoothing by the same rule, before
  its resolution tests. Alternative: 240 (best fit, rougher Vp) or 960 (smoother, fit 3.9 % worse).

First steps that need no decision on P-M2: run T6 on blanco + oregon at the damping rule chosen in P-M1, and write
the T7 forward check on the current tiles.

**Ran 2026-11-23 14:03-16:06** (`b2_t6_m0.sh`, log `<runs>/t6_m0.log`; smoothing scan `<runs>/m0_smooth.log`):
T6 = joint tile blanco_oregon_v2 (7,691 events, 242 stations, interior nodes 88 x 49 on the Blanco lattice) against
blanco_v2 + oregon_v2g (Oregon rebuilt on the dataset weight scale), all at damping 200, smoothing 20, on the D-B005
start; comparison `b2_t6_joint.py`; result in section 6 item 7. M0 = margin_m0_v2; maps `b2_m0_report.py`. Results
in `data/tomo_v1/b2_merge/t6/` and `m0/`.

**M0 result and its smoothing (2026-11-23 16:10; data/tomo_v1/b2_merge/m0/).** At damping 200 and smoothing 20,
M0 changes the start by 7.0 % rms in Vp (max 79 %) and alternates in sign from one 30 km node to the next: on the
coarse grid each node is crossed by many rays, so the data dominate each scaled column and the same smoothing
weight does much less than on the tiles. M0's smoothing was then taken from its own trade-off, as D-B044 says
(damping 200, weights 20 to 1920 laterally, half vertically):

| smoothing | weighted RMS | rms dlnVp / dlnVs % | max abs dlnVp % | first-iteration CND | events kept |
|---|---|---|---|---|---|
| 20 | 0.2010 | 7.0 / 4.0 | 79 | 338 | 6,899 |
| 60 | 0.1997 | 4.6 / 2.3 | 65 | 299 | 6,897 |
| 120 | 0.1997 | 3.2 / 1.6 | 59 | 264 | 6,878 |
| 240 | 0.1984 | 2.2 / 1.1 | 28 | 212 | 6,869 |
| 480 | 0.2012 | 1.5 / 0.7 | 26 | 164 | 6,856 |
| 960 | 0.2062 | 0.9 / 0.4 | 18 | 120 | 6,848 |
| 1920 | 0.2114 | 0.5 / 0.2 | 8 | 93 | 6,840 |
| (damping 1000, smoothing 20) | 0.2230 | 4.2 / 2.2 | 67 | 39 | 6,925 |

The weighted fit is flat from 20 to 480 (within 1.5 % of its best) and worsens beyond (+3.9 % at 960, +6.6 % at
1920). At 480 the change is coherent: Vp 3-6 % slower than the start along the Juan de Fuca and Endeavour ridges at
7-15 km, Vp and Vs about 4 % faster at 40 km under Washington and Puget Sound, a few isolated nodes left in
Washington and Mendocino-Gorda. Unweighted absolute RMS rises with smoothing (0.246 at 20, 0.288 at 480): the
low-weight picks fit less well. Events still move 1.5-1.8 km horizontally and about 2 km in depth (median) at
damping 200, against 0.9 and 0.8 km at damping 1000, whatever the smoothing. Proposed: P-M11 below.
The same effect is visible in the tiles at damping 200 and smoothing 20 (section 6 item 7, node-to-node sign
changes at 12 and 21 km): the tile reruns from M0 should scan smoothing at damping 200 with the same rule.

## 8. References (DOIs checked against Crossref / DataCite on 2026-11-23)

* Afanasiev, M., et al. (2015). Foundations for a multiscale collaborative Earth model. Geophys. J. Int.
  doi:10.1093/gji/ggv439
* Ajala, R., & Persaud, P. (2021). Effect of merging multiscale models on seismic wavefield predictions near the
  southern San Andreas fault. J. Geophys. Res. Solid Earth 126, e2021JB021915. doi:10.1029/2021JB021915
* Ajala, R., & Persaud, P. (2022). Ground-motion evaluation of hybrid seismic velocity models. The Seismic Record
  2(3), 186-196. doi:10.1785/0320220022
* Fichtner, A., et al. (2018). The Collaborative Seismic Earth Model: Generation 1. Geophys. Res. Lett. 45,
  4007-4016. doi:10.1029/2018GL077338
* Julier, S. J., & Uhlmann, J. K. (1997). A non-divergent estimation algorithm in the presence of unknown
  correlations. Proc. American Control Conference. doi:10.1109/ACC.1997.609105
* Lin, G., Thurber, C. H., Zhang, H., Hauksson, E., Shearer, P. M., Waldhauser, F., Brocher, T. M., & Hardebeck, J.
  (2010). A California statewide three-dimensional seismic velocity model from both absolute and differential
  times. Bull. Seismol. Soc. Am. 100(1), 225-240. doi:10.1785/0120090028
* Molinari, I., & Morelli, A. (2011). EPcrust: a reference crustal model for the European Plate. Geophys. J. Int.
  doi:10.1111/j.1365-246x.2011.04940.x
* Scivier, S. A., Nissen-Meyer, T., Koelemeijer, P., & Baydin, A. G. (2024). Gaussian processes for probabilistic
  estimates of earthquake ground shaking: a 1-D proof-of-concept. arXiv:2412.03299. doi:10.48550/arXiv.2412.03299
* Scott, S. L., et al. (2016). Bayes and big data: the consensus Monte Carlo algorithm. Int. J. Manag. Sci. Eng.
  Manag. doi:10.1080/17509653.2016.1142191
* Thurber, C., & Eberhart-Phillips, D. (1999). Local earthquake tomography with flexible gridding. Comput.
  Geosci. doi:10.1016/S0098-3004(99)00007-2
* Zhang, H., & Ben-Zion, Y. (2024). Enhancing regional seismic velocity models with higher-resolution local results
  using sparse dictionary learning. J. Geophys. Res. Solid Earth 129, e2023JB027016. doi:10.1029/2023JB027016
* Zhou, Z., Gerstoft, P., & Olsen, K. (2024). Graph-learning approach to combine multiresolution seismic velocity
  models. Geophys. J. Int. 238, 1353-1365. doi:10.1093/gji/ggae212

Read in full for this note: Lin et al. 2010, Ajala & Persaud 2021 and 2022, Zhou et al. 2024, Scivier et al. 2024.
From abstracts only: Zhang & Ben-Zion 2024 and the SCEC windowing work, CSEM (Afanasiev et al., Fichtner et al.).

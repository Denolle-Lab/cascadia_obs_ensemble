# Model v0.4.1: v0.4.0 re-inverted with the tomoDD floor (D-B083)

PI 2026-11-25 07:59 (D-B082, D-B083): v0.4.0 is the base model; v0.4.1 = the v0.4.0 inputs re-inverted with tomoDD's
event updates floored at the local solid surface (D-B081), velocities above the surface rock-continued, product as
D-B076. Branch tomo/v041-floor (worktree ~/wt_v041, from tomo/v040-retimed + merge of tomo/ensemble-v032), outputs
/wd1/mdenolle_data/tomo_v1/v041_floor/ (RESUME.md, chain.log, identity/, test/, floor/, product/, noise/, report/).

## Method

* **Binaries** (`/wd1/mdenolle_data/tomo_v1/src/frozen_2026-11-25_inv/`, CHANGES.md there): the D-B081 freeze
  rebuilt with its own `build_variants.sh` (read-only source, new build root) for the four classes of v0.4.0 (m: M0;
  e13k: mendocino_gorda_c; e10k: blanco, oregon; w: washington), targets main and chk, GCC 8.5.0. No source change.
  Main binaries md5-identical to session -18's `bin/frozen_2026-11-25`; chk binaries differ only by the build path
  compiled into them (strings identical after path normalisation). -18's folders were not touched.
* **Step 1, identity** (`identity/run_identity.sh`): one v0.4.0 joint run per class rerun with the new binary, same
  inputs, tomoDD-SE.inp and MOD, no surface.dat; every Output_Files file compared with cmp.
* **Step 2, writer** (`b2_tomodd_floor.py`): surface.dat per stage from a conservative floor of the D-B068 merged
  surface (`b4_surface.py floor --spacing S`: exact minimum of surface_m over each node's cell - 1 m, true frame;
  `floor/topo_{30,45}s_cons_merged.bin`), tiles 30 s, M0 45 s (tomoDD's 1001 x 1001 limit), covering the stage's
  interior node box + 0.05 deg; node coordinates rescaled about the stage centre with b2_fwd_mods.ellipsoid_coords
  (D-B028 P-2, centre from inputs/forward.json) exactly as station.dat / event.dat / MOD; depths in km with 4
  decimals rounded down (never above cell minimum - 1 m); depth not rescaled. `check` compares every start
  hypocentre (event.dat) with the exact surface at its true position and with the floor tomoDD will read
  (floordep reproduced in float32), and tomoDD's node with NonLinLoc's nearest-node rule.
* **Runs** (`b2_tomodd_run.py`): a region with `floor_spacing_s` copies inputs/surface.dat into the run folder and
  writes Air_dep = tomodd.floor.air_dep_km = -1000 km (tomoDD tests Air_dep on the raw LSQR update before the floor
  and then skips the whole iteration's update and drops the events: it must not trigger).
* **Step 3, tests** (`test/run_test.sh`, v0.4.0 blanco_m0t d200_s20 inputs/MOD/settings, new e10k binary):
  blanco_floor (30 s floor, Air_dep -1000) and blanco_nohit (a floor no event can reach, every node -50 km, Air_dep as
  v0.4.0: the floor code reads the file and looks up every event every iteration but never acts; must equal v0.4.0
  byte for byte).
* **Step 4, production** (`b2_v041_chain.py`, gates on steps 1 and 3): margin_m0_v4 (inputs rebuilt and checked
  byte-identical to margin_m0_v3, P-4 recomputed with the new binary), check01 bounds-checked run with the floor,
  d200_s480; tiles <t>_m0f from margin_m0_v4/d200_s480 (inputs identical to <t>_m0t apart from MOD), v0.4.0 smoothing
  (blanco 20, others 480), damping 200; merge `product_v041` (D-B049 with the v0.3.1 d200 masks, D-B050, D-B051) as a
  dry run, then `b2_model_v041.py` (D-B076 product: medium, D-B073 node rule, Vp/Vs_tomography, NLL grids by
  b4_time_grids.build_field, registered candidate through b2_model_export.register with the registry flock).
* **Step 5, noise floor**: margin_m0_v4nf and <t>_m0fnf (b2_v040_noise.py: N(0, 1e-6 s) on every time, seed by region
  name), floored the same way, merged as `product_v041nf` (dry run).
* No smoothing scans (v0.4.1 keeps v0.4.0's choices, as D-B083 asks).

## Known limits of the floor in the inversion

* tomoDD applies the floor to src_dz after the joint LSQR solve: the velocity update of an iteration in which an event
  was stopped was solved for the unstopped event move (no re-solve). The next iteration's rays and residuals use the
  floored position.
* The floor is the cell minimum at 30 s (tiles) / 45 s (M0), coarser than NonLinLoc's 15 s: where the surface is steep
  it lies tens to hundreds of metres below the local surface (numbers per stage in the report, section 2). An event
  stopped there is below the surface by at least that gap.
* The floor stops upward moves only; it never moves an event that is already below it.

## Results (chain 2026-11-25 08:05-10:19; report /wd1/mdenolle_data/tomo_v1/v041_floor/report/report.html, tables also in data/tomo_v1/v041_floor/)

**Step 1, identity.** All four classes byte-identical to v0.4.0 without surface.dat: e10k (blanco_m0t d200_s20), m
(margin_m0_v3 d200_s480), e13k (mendocino_gorda_c_m0t d200_s480), w (washington_m0t d200_s480): 20 of 20 Output_Files
each (loc, reloc + 12 iteration files, res, src, sta, vel, Vp_model, Vs_model). tomoDD-SE.log / stdout.txt differ in
text only (timestamps; the "weighted mean ( x %)" change printed against an uninitialised value: "Inf %" in v0.4.0,
"******* %" or a number in the reruns; one garbage byte after a date in stdout).

**Step 2, writer.** surface.dat 945 x 939 nodes (M0, 45 s), 565-613 x 793-841 (tiles, 30 s). Node choice: 0
mismatches with NonLinLoc's rule; 1-19 float32 edge ties per stage (position within 0.0006 cell of a cell edge; the
floor there is still >= 1 m below the exact surface). Floor below the exact surface at the start hypocentres: median
21-62 m, p90 73-215 m, p99 154-463 m, max 342-990 m (minimum 1.0 m). Start hypocentres (v0.4.0 = v4-NLL3D):

| stage | events | above the true surface at start | below the surface but above the floor |
|---|---|---|---|
| M0 | 6,943 | 13 (51-316 m above; Blanco 4, Other onshore 3, JdF 2, Gorda 2, Cascades 1, Mendocino 1) | 18 (27-440 m below the surface, 60-248 m above the floor) |
| mendocino_gorda_c | 12,191 | 4 | 7 |
| blanco | 4,378 | 6 (175-316 m above, all Blanco) | 2 (28-51 m below the surface, 125-155 m above the floor) |
| oregon | 3,390 | 4 | 4 |
| washington | 9,332 | 3 | 18 |

These are not moved by the writer; their first update that ends above the floor stops at it (the floor rule).

**Step 3, test (v0.4.0 blanco tile, inputs and settings of blanco_m0t d200_s20).** Floored: 97 floor stops on 20
events (10, 7, 7, 7, 6, 6, 6, 8, 10, 8, 11, 11 per iteration), none "not met"; air quakes 4 -> 0; iterations 15 (12 +
3 air-quake repeats) -> 12; events kept 4,374 -> 4,378 of 4,378; final hypocentres above the true surface 14 -> 0;
12 end on the floor. The 19 floored events also kept by v0.4.0 moved median 0.68 km deeper (p90 |dz| 1.29 km); the
other 4,355 moved median dz 6 m, p90 |dz| 0.53 km (blanco's own noise floor: p90 |dz| 0.53-0.61 km, D-B082). Vp
rms change 0.53 % (max 14.6 %, 7,318 of 49,335 node values changed), Vs 0.11 %; final abs rms 0.2405 -> 0.2407 s,
weighted 0.1284 -> 0.1277 s. Unreachable floor (all nodes 50 km above sea level, Air_dep 0.727 as v0.4.0): 20 of 20
output files byte-identical to v0.4.0, 0 floor lines: the floor code changes nothing when it does not act.

**Step 4, production (floored chain, same inputs: M0 inputs byte-identical to margin_m0_v3 incl. MOD and P-4; tile
inputs identical to <t>_m0t apart from MOD, whose start from the new M0 differs by 0.04-0.06 % rms, max 1.0-2.2 %
Vp).** Bounds-checked check01 runs with the floor: exit 0 for M0 and all tiles.

| stage | floor stops (lines) | events hit | end on the floor | air quakes v0.4.0 -> v0.4.1 | events kept v0.4.0 -> v0.4.1 | above true surface at the end v0.4.0 -> v0.4.1 | abs rms (s) v0.4.0 -> v0.4.1 |
|---|---|---|---|---|---|---|---|
| M0 | 2,090 | 266 | 156 (2.2 %) | 88 -> 0 | 6,855 -> 6,943 | 132 -> 0 | 0.2812 -> 0.2819 |
| mendocino_gorda_c | 503 | 70 | 48 (0.4 %) | 20 -> 0 | 12,170 -> 12,190 | 42 -> 0 | 0.2413 -> 0.2413 |
| blanco | 100 | 21 | 11 (0.3 %) | 4 -> 0 | 4,374 -> 4,378 | 14 -> 0 | 0.2405 -> 0.2404 |
| oregon | 1,724 | 218 | 125 (3.7 %) | 66 -> 0 | 3,324 -> 3,390 | 114 -> 0 | 0.1910 -> 0.1921 |
| washington | 2,008 | 278 | 154 (1.7 %) | 141 -> 0 | 9,190 -> 9,329 | 84 -> 0 | 0.2146 -> 0.2155 |

No "floor not met". Iterations 21-24 -> 12 (no air-quake repeats). Attempted updates stopped by the floor reached
median 0.2-0.9 km and at most 1.4-3.4 km above it. 3 final positions read 0.4 m above the floor (M0 2, washington 1):
REAL*4 positions and 3-decimal depths; they are 0.06-0.17 km below the true surface (the floor is 1 m under the cell
minimum). Events ending on the floor by region: M0 JdF ridge/plate 26 of 122 (21 %), Axial 3 of 24, Other onshore 70
of 1,454; oregon Other onshore 87, Cascades 21, Offshore forearc 17; washington Offshore forearc 70 of 1,921, Other
onshore 37, Puget/Georgia 29, Cascades 18. On the map they cluster at Mount St Helens, the Texada Island quarry box
(D-B071 quarry_blast_likely), the JdF ridge and scattered onshore Oregon events.

**Model v0.4.1 - v0.4.0 (merged inversion values).** Resolved nodes (D-B049 masks, identical): Vp 888 nodes rms 0.79 %
(noise floors 0.84 % v0.4.1, 0.74 % v0.4.0), 1.6 % of nodes above noise; Vs 320 nodes rms 0.70 % (noise 0.87 / 0.82 %),
0 % above noise. By depth: only 1 km stands out (20 resolved Vp nodes, rms 0.087 % against noise 0.012-0.019 %,
max 0.16 %); 15-29 km (Blanco s20) 0.76-1.2 % inside its noise floor (0.73-1.49 %). Tile-covered nodes: Vp rms
0.14 % (noise 0.10-0.14 %), shallow (-3 to 3 km) rms 0.10-0.18 % with 15-19 % of nodes above noise, max 8.8 % at
3 km; Vs rms 0.06 % (noise 0.06 %), -1 to 5 km 0.03-0.04 % rms with 27-36 % of nodes above noise. The floor changes
the shallow model a little, where the floored events' rays are; it does not change the resolved model beyond the
numerical noise.

**Hypocentres (v0.4.1 - v0.4.0, events kept by both).** Events hit by the floor: median dz +0.53 to +0.87 km (deeper),
p90 |dz| 1.3-3.4 km, 30-43 % moved > 1 km. Events never hit: median dh 10-65 m, p90 |dz| 0.14-0.61 km, close to the
noise floors (p90 |dz| 0.06-0.61 km; M0 0.16 km against 0.06-0.08 km and Oregon 0.14 against 0.08 are above them).
Largest region changes: M0 JdF ridge/plate p90 |dz| 1.9 km (20 % > 1 km).

**Residuals, same event set (events kept by both).** P and S rms change by -0.3 to +0.4 % in every stage and station
class except Oregon OBS P +1.5 % (819 picks) and Washington OBS S +1.15 % (121 picks); catalog dt 0.00 to +0.31 %.
With all events (v0.4.1 keeps the former air quakes): -0.3 to +0.6 %, Oregon OBS P +2.8 %.

**Run times (loaded machine, load 80-230 on 128 cores).** M0 21.2 min (v0.4.0 18.1), mendocino_gorda_c 35.7 (38.1),
blanco 14.7 (12.5), oregon 14.6 (17.8), washington 46.3 (47.0); noise-floor copies 14.0-45.9 min. Whole chain 08:15 ->
10:19 including the identity gates (76 min for washington's).

**Product and registration.** v0.4.1 registered as candidate 2026-11-25 18:18 UTC (git 94b084dc): v0.4.1.nc (sha256
88370da4...), .zarr, v0.4.1_surface_lattice.nc, v0.4.1_nll (P md5 52fff8c4..., S md5 1105aadd...; build_field
parameters equal to the v5.1 production build). medium: 203,988 rock, 8,538 water, 20,406 air nodes (405 water and
1,332 air nodes have P rays); D-B073 node rule lowered Vs at 8,739 nodes (median 14.5 %, max 23.8 %); read-back checks
all true (Vp/Vs rock = inversion, water 1.5, NaN above, nc = zarr, merge variables unchanged).

## Proposed decisions

* **P-F1 (inversion binaries).** v0.4.1 and the D-B083 configuration use `/wd1/mdenolle_data/tomo_v1/src/
  frozen_2026-11-25_inv/bin` (the D-B081 freeze rebuilt for classes m, e13k, e10k, w; main md5 = bin/frozen_2026-11-25;
  byte-identical to the D-B054 binaries without surface.dat in all four classes; archive
  archive/frozen_2026-11-25_inv.tar.gz sha256 83305275...).
* **P-F2 (floor of an inversion stage).** surface.dat = conservative floor of the D-B068 surface (exact cell minimum -
  1 m) at 30 s for tiles and 45 s for M0 (tomoDD's 1001-node limit), over the stage's node box + 0.05 deg, P-2 frame;
  Air_dep -1000 km. Accept the coarser floor's gap (median 21-62 m, max 0.34-0.99 km below the surface at start
  hypocentres) or ask for a finer floor (needs a larger sfz array: a source change).
* **P-F3 (ensemble on v0.4.1).** The D-B030/D-B032 real phase on the v0.4.1 configuration: margin_m0_v4 / <t>_m0f,
  bin_dir frozen_2026-11-25_inv, floor per stage as P-F2. The ensemble driver (tomo/ensemble-v032) reserves a `floor`
  option and refuses it: it must write surface.dat per member stage (each member's own frame is the stage's frame)
  and set Air_dep before the launch.
* **P-F4 (floor-held events).** The events that end on tomoDD's floor (494 stage-events; table floor_by_region.csv)
  have depths set by the floor, as NonLinLoc's floor_limited; keep them flagged in any tomoDD product; no event is
  removed or moved by hand.

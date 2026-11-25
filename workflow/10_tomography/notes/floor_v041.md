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

## Results

(filled from the report, /wd1/mdenolle_data/tomo_v1/v041_floor/report/report.html)

## Proposed decisions

(below)

# Bayesian ensemble (RTO) of a merged-model configuration: implementation, calibration gate, cost

Agent note, 2026-11-25 (worktree ~/wt_ens, branch tomo/ensemble-v032). PI: "start the ensembling for this
configuration so that we get uncertainties for this velocity model" (v0.3.2 = v0.3.1 + D-B073). Design:
notes/bayes_design.md (D-B030, D-B032) with notes/merge_design.md P-M5 (shared draws) and P-M6 (damping prior).
Replaces the single-region code and placeholder of 2026-11-22 (mendocino_gorda_v2, old dataset, never run at scale).
Status: built and tested on 2 synthetic + 1 real member; the full chain is NOT launched (main session review first).
The PI expects the full ensemble to run on a v0.4.x configuration (re-timed picks; another agent builds it), so the
configuration is a parameter.

## 1. Code and layout

* params `tomodd.ensemble`: root, frozen-binary folder, process and memory limits, member counts, the prior, the
  calibration criteria, and `configs.<name>` (M0 run, tile runs, smoothing, damping, binary class, node model, truth
  model, product label). `config: v0.3.1` now; a v0.4.x entry only needs its M0 and tile regions/tags.
  Environment overrides B2_ENS_CONFIG, B2_ENS_ROOT.
* b2_ens_common.py (config, paths, Lambert random field, tomoDD file readers/writers), b2_ens_member.py (prep, draws,
  one tomoDD stage, merge), b2_ens_synth.py (synthetic truth and data), b2_ens_run.py (chain and scheduler),
  b2_ens_coverage.py (calibration gate), b2_ens_stats.py (statistics, product, HTML report).
* b2_merge.py refactored into geometry() (tile boxes, resolution windows: the same for every member, cached) and
  blend() (per member). `b2_merge.py --dry-run` reproduces v0.3.1 exactly (all 15 variables equal).
* Outputs: /wd1/mdenolle_data/tomo_v1/ensemble_v0.3.2/ (prior/, synth/, real/, stats/, products/, logs/, chain.state,
  chain.log, RESUME.md).

One member = M0 (margin_m0_v2 inputs, smoothing 480/480/240) -> the four merged tiles (mendocino_gorda_c s480,
blanco s20, oregon s480, washington s480) restarted from that member's M0, all four at once -> b2_merge rules (D-B049
windows from the deterministic resolution tests at damping 200, fixed; D-B050: ridges_north out of the merge, so it
is not run) -> v0.3.2 medium and the D-B073 node rule with the member's drawn parameters. Kept per member: draws,
start hypocentres, per stage the start MOD, model_geo.npz, DWS, relocations, residual summary, compressed logs;
merged.npz. tomoDD scratch (Input_Files, tomoDD.vel/.res/.src, per-iteration relocations) is deleted.

## 2. Prior ingredients (prep, 30 s; ROOT/prior/prior.json)

* Unions over M0 and the four tiles: 742,576 picks (1,039,605 rows over the five inputs; sigma and distance identical
  between regions), 27,902 events, 519 stations, 1,027 station-phases, 4,666 station-phase-distance bins.
* Pick-noise class factors (final residual / sigma_pick of the five deterministic runs, 1.4826 MAD): land P 1.79
  (21,471 events), land S 1.93, OBS P 1.35, OBS S 1.58. Residual MAD 0.145 / 0.351 / 0.128 / 0.334 s. Drawn noise sd
  per pick class (member r000): 0.18 / 0.41 / 0.14 / 0.35 s.
* OBS S static sd (bootstrap sd of the near-event median + 0.05 s, max over regions): median 0.069 s, 75 % 0.080 s,
  max 0.557 s (240 OBS); land P 0.05, land S 0.08, OBS P 0.08 s.
* Forward-error sd (tomoDD - P-1 reference - P-4 correction, robust): P 0.025-0.062 s, S 0.037-0.12 s by class and
  distance bin.
* Delph et al. (2018) Vs (data/tomography/Delph2018.nc, untracked in git, read from the main checkout; md5
  05418226...; 40-49 N, 124.8-120 W): 61 % of the deterministic P DWS lies where its weight > 0, so it enters the
  mixture: v4-NLL3D Vp/Vs 1.76 40 %, 1.73 20 %, 1.80 20 %, Delph 20 %.
* Checks: the member start code with no draws reproduces the deterministic M0 start exactly and the tile starts to
  the MOD's printed precision (Vp 5e-4 km/s, Vp/Vs 5e-5); the member reference reproduces b2_merge.reference exactly.

## 3. Synthetic truth (b2_ens_synth.py; ROOT/synth/truth/truth.json)

Truth = v0.3.1 tomography nodes x (1 + one 3 % random-field draw; seed 100000; rms 2.9 % Vp, 3.0 % Vs above 50 km),
on the merge grid; hypocentres = v4-NLL3D + one prior draw (truncated at the merged surface; median 1.5 km lateral,
1.7 km vertical); data = tomoDD-SE_syn (frozen syn binaries) per stage in the truth sampled at the stage's nodes, on
the real pick and pair lists, + one draw of statics, forward offsets and pick noise (class factors at their point
values). Runs: M0 1.1 min, tiles 0.5-1.5 min, 2.4 GB max. Synthetic minus real absolute times: median -0.02 to -0.24 s,
MAD 0.36-0.52 s (the synthetic data are self-consistent with the truth; this is only a sanity check).
Simplification: M0's synthetic data come from the truth sampled on M0's 30 km nodes (M0 only gives the tiles' start).

## 4. Proposed decisions (for the main session to record; none adopted by me)

* P-E1 Configuration as parameters (dataset, M0 run, tile runs, binaries, product label); v0.3.1 now, v0.4.x later.
* P-E2 Draws once per member for the whole margin, shared by M0 and every tile (P-M5): random field on one Lambert
  10 km lattice (bilinear to any node), statics by station-phase, forward offsets by station-phase-distance bin, noise
  by pick (event, station, phase), start hypocentres by event ID.
* P-E3 Damping: ONE log-uniform factor in [0.7, 1.5] per member, applied to the damping 200 of M0 and of all four tiles,
  instead of an independent factor per tile (P-M6 as written). Reason: D-B045 asks for one damping on one weight scale
  in every tile; independent factors would break that inside a member and add seam variance that is not data
  uncertainty.
* P-E4 Smoothing fixed at the chosen values (M0 480, tiles 480 / 20), not drawn. Reasons: (1) D-B049 gates tiles by
  their smoothing (blanco s20 gated, s480 not) and uses resolution windows computed at that smoothing; a drawn
  smoothing would change the merge rule member by member; (2) the tile scans were flat in weighted RMS from 20 to 480
  (rule tolerance 1.5 %), so a smoothing prior would span a factor 24 and turn the ensemble into a regularisation
  scan; (3) the damping draw already samples regularisation strength.
* P-E5 The D-B046 start floor Vp/Vs >= 1.65 on every member start, M0 included: independent 3 % Vp and Vs fields gave
  Vp/Vs down to 1.49 at 3.4 % of M0 nodes (916 of 26,880) in a test draw.
* P-E6 Starting hypocentres from the truncated normal at the merged solid surface of the drawn epicentre (exact
  sampling), not clipped: no draw sits above the surface; 17 of 27,902 v4-NLL3D starting depths are above it.
* P-E7 D-B073 parameters drawn per member, Vp/Vs U[1.95, 2.25] and thickness U[2, 3] km (span of the D-B070 variants
  V2 to V4), applied to the merged member's rock nodes as in v0.3.2; stored in draws.json for the later relocation
  members (D-B039). The inversions themselves have no rule.
* P-E8 Calibration gate on nodes resolved by the deterministic checkerboards (resolution > 0) and covered by a tile
  in every member; at least 300 nodes per phase; Vp and Vs gated, Vp/Vs reported; hypocentres per tile and coordinate
  gated. Nodes with DWS >= 100 in 90 % of members reported as a second mask.
* P-E9 Binaries: the 2026-11-24 freeze (D-B054) for the v0.3.1 configuration, the build its deterministic runs used;
  the freeze folder is a parameter (section 5).
* P-E10 Ensemble product label `<model version>+ens.<merged members>` (e.g. `0.3.2+ens.100`), status intermediate,
  D-B047 exporter; the chain writes a dry run and the report by default and registers only with B2_ENS_REGISTER=1
  (PI's go; the exporter read-back checks Vp_mean before the registry is touched, under a registry flock); statistics on the v0.3.2
  convention (medium, node rule) and on the inversion values (Vp_tomography, Vs_tomography), with the tile
  disagreement term of P-M5 as separate variables.

## 5. Above-surface nodes in the ensemble's inversions (main session item 5)

Every M0 and tile inversion has rock velocities above the solid surface (D-B068 surface at the node; deterministic
runs; member grids are the same):

| run | interior nodes | above surface | water | air | above with dws_p > 0 (water / air) | Vp above, median [p5, p95] |
|---|---|---|---|---|---|---|
| M0 margin_m0_v2 | 26,880 | 3,339 | 984 | 2,355 | 396 (135 / 261) | 4.34 [2.94, 5.74] |
| mendocino_gorda_c_m0 | 52,311 | 6,774 | 2,135 | 4,639 | 307 (163 / 144) | 4.93 [3.40, 5.82] |
| blanco_m0 | 41,181 | 6,574 | 2,652 | 3,922 | 122 (122 / 0) | 4.88 [3.48, 5.76] |
| oregon_m0 | 55,566 | 5,648 | 932 | 4,716 | 343 (17 / 326) | 4.46 [3.76, 5.74] |
| washington_m0 | 59,976 | 5,688 | 539 | 5,149 | 456 (15 / 441) | 4.95 [3.79, 6.10] |

Assessment of a surface inside the inversion (no build, no source change made here):
* Water velocities above the seafloor in the MOD: rejected by the numbers. On the v0.3.2 nodes they bias OBS paths by
  +0.04 s P / +0.06 s S median and up to +0.30 / +0.77 s at p95 (model_v032.md section 3); -18's S1 test on graded
  0.5-1 km nodes found the same (JdF S rms 88 ms with water and air, 8.6 ms rock-continued). Rock-continued values
  above the surface are what NLLoc's own grids use at the fluid nodes (FLUID_NODES = rock_continued_1km).
* Finer near-surface depth nodes: every frozen build allows maxnx 90, maxny 80, maxnz 50 (RaySPDR.inc), so up to 48
  interior depth nodes (now 21); e.g. 0.5 km from -3 to 10 km (27 nodes) plus the 14 nodes below gives 41. Costs:
  derivatives per datum grow (the 42-node test of D-B029 doubled them; MAXND = 100, 44 used in Mendocino v2), so a
  bounds-checked run per tile is needed; D-B029 found the upper crust coverage-limited at node scale, so finer nodes
  add unresolved parameters. A depth mesh cannot follow a seafloor that varies from 0 to 5 km laterally, so it only
  narrows the blur.
* Holding nodes fixed: tomoDD-SE has the simul2000 fixed-node and linked-node machinery (nfix, imerge/jequal,
  RaySPDR2.f) but the input was removed: `inf=0` is hard-coded after the grid is read ("cz this set is removed by HZ",
  RaySPDR2.f ~1493) and the link arrays are zeroed (~1459). Without a source change every node with rays is free. The
  exact change would be: replace `inf=0` by a read of (ixf, iyf, izf) triples from the MOD file after the Vp/Vs block,
  ending at ixf = 0 (izf > nz for S nodes, as the existing loop at ~1596 expects); everything downstream (nfix,
  ndexfx/mdexfx, the "skip fixed nodes" DWS loop) is still in place. Linked nodes (an above-surface node following the
  rock node below it, "constant perturbation link") would need the link read restored as well. Not done (licence;
  PI decision).
* Without source changes, an outer loop can impose the surface: run each weighting set as its own tomoDD run and
  reset the above-surface nodes to the rock-continued values of the updated rock node below between runs (4 restarts
  per inversion instead of 1; hypocentres carried through event.dat). This keeps rays and interpolation in rock and
  stops the above-surface nodes from carrying independent structure.
* Revised item 5 (after -18's S1): rock-continued velocities above the surface during the inversion, the D-B081 floor
  on event updates, and `medium` masking only in the published product.
  (a) The D-B081 block (tomoDD-SE.f of the 2026-11-25 freeze, lines ~2465-2481) sits in the common source-update loop
  `do i= 1,nsrc` that follows both solver branches: the joint branch (`if(joint .eq. 1)`, line ~1361) and the
  relocation-only branch (`elseif (joint .eq.0)`, ~2350, lsfitH_FDD_shot) both end before the air-quake check (~2400)
  and that loop; no jump from the joint branch goes past it (its only `goto 778` is the "fewer than 2 non-air-quakes"
  exit). So it acts in joint mode whenever surface.dat is in the run folder. Caveats: the air-quake check runs first,
  on the unfloored update, and rejects the whole iteration (`goto 500`, no update, the air quakes dropped) if any event
  goes above Air_dep, so Air_dep must be set far up (-18 uses -10 km); the floor changes src_dz after the joint solve,
  so the velocity update of that iteration was solved for the unfloored event move (no re-solve); surface.dat holds
  at most 1001 x 1001 nodes (4.2 deg at 15 s): tiles span up to 6 x 5.5 deg with margin and M0 11.6 x 11.7 deg, so
  tiles need a 30 s and M0 a 45 s conservative floor (b4_surface.floor_min at that spacing, exact cell minimum), in the
  stage's rescaled frame (P-2), covering every event of the stage + margin.
  (b) No new build is needed for the tile sizes: the 2026-11-25 freeze (bin/frozen_2026-11-25) already has main and
  check binaries for e10k, e13k, w, m and o140 from the patched source (FORMAT fixes 1-6 included; syn binaries
  md5-identical to 2026-11-24, main and chk differ). Needed before use: one joint run per class without surface.dat
  compared byte for byte with the 2026-11-24 binary (e.g. blanco_m0 d200_s20, about 12 min, one process), one joint
  run with surface.dat (floor lines in the log, no event above the floor), and a surface.dat writer for tiles and M0
  (as -18's, without B5's centroid re-anchoring).
  (c) The driver already takes the freeze as a parameter: configs.<c>.bin_dir overrides tomodd.ensemble.bin_dir
  (b2_ens_common.binary). A `floor` option (surface.dat per stage, Air_dep) is reserved in the config and refused
  until implemented.

## 6. Test members and cost

Run 2026-11-25 06:06-07:15 (`b2_ens_run.py members synth s000 s001` and `members real r000`; 3 members at once next to
about 25 tomoDD processes of other sessions; 2026-11-24 frozen binaries, nice 10). All stages finished, no failure.

| stage | wall (min), s000 / s001 / r000 | max RSS (GB) | final abs RMS (s), synth / real | events kept / start (air quakes dropped) |
|---|---|---|---|---|
| M0 | 18.3 / 18.4 / 18.9 | 1.6-2.5 | 0.50-0.51 / 0.46 | 6,943 start |
| blanco | 15.5 / 15.5 / 13.9 | 1.2-2.0 | 0.53 / 0.44 | 4,313-4,325 / 4,378 synth, 4,358 real (det. 4,374) |
| oregon | 18.2 / 17.3 / 18.1 | 1.0-1.2 | 0.52-0.53 / 0.43 | 3,325-3,335 / 3,390 synth, 3,295 real |
| mendocino_gorda_c | 38.0 / 40.0 / 40.8 | 2.9-3.1 | 0.51 / 0.45 | 12,121-12,129 / 12,191 synth, 12,127 real |
| washington | 47.8 / 49.1 / 49.9 | 3.4-3.7 | 0.54 / 0.45 | 9,157-9,169 / 9,332 synth, 9,075 real (det. 9,184) |
| merge + node rule | < 1 s | small | | rule lowered Vs at 6,886-9,468 nodes (drawn Vp/Vs 2.10-2.24, 2.1-2.6 km) |

* One member: 140 process-minutes, about 70 min wall (M0, then washington). The final absolute RMS (0.45 s real,
  0.51-0.54 s synthetic) is above the deterministic runs' (washington 0.23 s) because every member fits data with
  the drawn noise added (drawn delta sd 0.31 s); synthetic members also start from hypocentres 1.5-1.7 km from the
  truth. Every member run drops 1-3 % of its events as tomoDD air quakes (6-15 repeated passes per run), as the
  deterministic runs do (washington 148): the D-B081 floor question (section 5).
* Disk: 29 MB kept per member (draws 9.3 MB, start hypocentres 1 MB, 2-3 MB per stage, merged.npz); prior 86 MB;
  synthetic truth 294 MB. A running stage needs up to 250 MB before trimming.
* Code paths tested end to end: coverage (2 members: FAIL as it must with 2 members; 41 s), statistics, product
  (dry-run netCDF 11 MB) and HTML report on r000 (/wd1/mdenolle_data/tomo_v1/ensemble_v0.3.2/test_r000/products/).
  The member merge with null draws reproduces v0.3.1 exactly (Vp, Vs) and v0.3.2's Vp and Vs (2.4e-7).
* Early indication from 2 synthetic members (not a verdict): on the 888 checkerboard-resolved Vp nodes the 2-member
  mean misses the truth by a median 1.45 % while the two members differ by a median 1.25 % (so a single-member sd of
  ~1.3 %); the recovered change correlates 0.62 with the truth's 2.9 % rms perturbation. A calibrated ensemble would
  miss by ~1.1 % (0.6745 x 1.3 % x sqrt(1.5)), so the intervals may come out ~1.3 times too narrow (rough, 2 members):
  the smoothing (480 in three tiles) removes part of a 30 km truth field in every member alike, and RTO's spread does
  not contain that regularisation bias. If the
  gate fails for this reason, the choices are a truth drawn at the scale the regularisation can carry, a bias term
  from the resolution tests, or a wider damping/smoothing prior (PI decision).

Cost of the full chain at max_procs 56 (100 synthetic + 5 noise-floor + 100 real members = 205 members x 140
process-minutes = 28,800 process-minutes): 8.6 h with perfect packing; with the slowdown of a full machine
(D-B026: 1.2-1.65x at 32-64 concurrent; 56 here + up to 40 of -18 + 16 others on 128 cores) and the two tails
(the last members' washington stage, ~70 min each phase), about 12-14 h: synthetic phase ~6.5 h, calibration
< 5 min, real phase ~6 h, statistics and products < 30 min. Memory: 56 runs x 2.3 GB mean (3.7 GB max) ~ 130 GB;
the scheduler keeps MemAvailable above 40 GB (mem_per_run_gb 3.5 reserved for each process younger than 5 min),
so it runs fewer than 56 when the other sessions are busy and the wall time grows accordingly. Disk: ~6 GB kept,
~15 GB transient.

Launch (not done; for the main session after review):

    cd /home/mdenolle/wt_ens
    LD_LIBRARY_PATH=/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/lib setsid nohup \
      /home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/bin/python workflow/10_tomography/b2_ens_run.py chain \
      >> /wd1/mdenolle_data/tomo_v1/ensemble_v0.3.2/chain.log 2>&1 < /dev/null &

The three test members are reused by the chain (their stages are DONE). For a v0.4.x configuration: add
configs.v0.4.x, set `config` (or B2_ENS_CONFIG) and a new root (B2_ENS_ROOT), then the same command.

## 7. Review items after the first launch (2026-11-25)

* Synthetic truth fixed (commit 4e7abb94): the truth takes the members' start floor Vp/Vs >= 1.65 (P-E5); before, 34
  of the 320 resolved S nodes had truth Vp/Vs < 1.6, which D-B051 caps in every member (missed by construction).
  New truth: minimum Vp/Vs on the resolved S nodes 1.65 (float32), 75 of 320 at the floor. tomoDD-SE_syn reruns are
  hash-checked (MOD and event.dat md5). The coverage report adds an N_eff-aware consistency table (N_eff P 25.7 /
  50.9, S 11.2 / 21.4 on the gate masks) next to the unchanged D-B030 verdict.
* Background bias of the truth (b2_ens_f0.py; result in ROOT/synth/f0_bias.md): the truth's v0.3.1 / start - 1 part
  is not a draw of the members' prior, so what the inversions miss of it is common to all members. A second
  synthetic dataset from v0.3.1 alone (f = 0, same floor, hypocentres and delta_true), inverted once with null draws,
  maps that bias per tile on the gate nodes.
* Open item for the real ensemble (no action yet): the pick-noise class factors are pooled over M0 and the four tiles
  but differ strongly between stages (reviewer: OBS P 0.90 in Oregon against 3.04 in Washington; OBS S 1.13 in Blanco
  against 3.36 in Washington). A pooled factor under-perturbs the noisier stages and over-perturbs the quieter ones.
  Options: class factors per stage and class (each stage's picks drawn with its own factor; a pick shared by two
  stages would then need one value, e.g. the factor of the stage that holds most of its picks), or per station
  network. Decide before the real-data members.

## 8. Real phase on model v0.4.1 (D-B083, P-F3): driver changes, test member, proposed decisions

Agent note, 2026-11-25 (worktree ~/wt_ens41, branch tomo/ensemble-v041 from tomo/v041-floor). The running v0.3.2
synthetic calibration (~/wt_ens, root ensemble_v0.3.2) was not touched. Outputs of this work:
/wd1/mdenolle_data/tomo_v1/ensemble_v0.4.1/ (prior/, real/r000, verify/nullmerge.json, test_r000/products/,
RESUME.md, logs/).

### 8.1 What changed in the driver

* **Floor (configs.<c>.floor, D-B083).** Per member and stage, after the member's inputs are built (its drawn start
  hypocentres rescaled into event.dat), surface.dat is written in the stage's own rescaled frame
  (b2_tomodd_floor.write_surface, refactored out of `write` without the manifest: centre from the member's
  forward.json, floor grid read from the pinned copy in prior/floor/) and copied into the run; its md5 must equal the
  prior's surface.dat of that stage, which prep checks byte-identical to the surface.dat of the deterministic v0.4.1
  run (all five stages identical). Air_dep = tomodd.floor.air_dep_km (-1000 km). Binaries: configs.<c>.bin_dir
  (frozen_2026-11-25_inv/bin, P-F1), runtime files from the same folder (md5 = the 2026-11-24 runtime_files).
  Recorded per stage: run/floor_hits.csv.gz (every >>>Floor line: iteration, ID, attempted depth, floor),
  run/floor_end.csv (events whose final tomoDD-frame depth lies within [-2 m, 5 m) of the floor, with region and
  number of stops), and DONE["floor"] (start hypocentres above tomoDD's floor, stops, events hit, events ending on
  the floor by region, not-met lines, air-quake lines, final positions above the floor by > 2 m = errors). MERGED
  carries the per-stage summary; stats.json and the report a min / median / max table over the members. The
  parser reproduces the deterministic v0.4.1 counts exactly (stops / events hit / end on the floor: M0
  2,090 / 266 / 156, mendocino_gorda_c 503 / 70 / 48, blanco 100 / 21 / 11, oregon 1,724 / 218 / 125, washington
  2,008 / 278 / 154).
* **Configuration v0.4.1** (params tomodd.ensemble.configs.v0.4.1; default config now v0.4.1, each config has its
  own root): dataset b1b_balanced_ct (checked against every stage's region), M0 margin_m0_v4/d200_s480, tiles
  <tile>_m0f (mendocino_gorda_c and oregon, washington d200_s480, blanco d200_s20: v0.4.0 smoothing), damping 200,
  masks of the v0.3.1 tiles (mask_region, as product_v041's choice.csv), merge settings tomodd.merge.product_v041,
  node model 0.4.1, product_version 0.4.1 -> products 0.4.1+ens.<merged members>, floor on,
  calibration_root = ensemble_v0.3.2 (8.4, P-E11).
* **Pick-noise class factors per tile** (prior.noise_groups tile): one factor per tile and class from that tile's
  deterministic run; M0's residuals are left out (reported only); a pick belongs to the tile holding most picks of
  its event among the tiles that hold it (4,361 M0-only picks use the factor pooled over the four tiles); a
  tile-class with < 30 events (prior.class_factor_min_events) uses the pooled factor (only blanco land, which has no
  picks). Each used group-class gets its own inverse-gamma draw per member. Configuration v0.3.1 keeps
  `pooled_with_m0` (the rule its running calibration drew with; its draws are unchanged). Check of the code on the
  v0.3.1 runs: OBS P 0.900 Oregon / 3.035 Washington, OBS S 1.129 Blanco / 3.359 Washington (the reviewer's
  numbers). On the v0.4.1 runs (1.4826 MAD of residual / sigma):

  | group | land P | land S | OBS P | OBS S |
  |---|---|---|---|---|
  | mendocino_gorda_c | 1.60 | 1.96 | 1.60 | 1.80 |
  | blanco | (pooled) | (pooled) | 1.05 | 1.13 |
  | oregon | 1.45 | 1.60 | 0.86 (310 events) | 1.38 (305) |
  | washington | 1.69 | 1.82 | 2.37 (61 events) | 2.67 (84) |
  | pooled over the tiles | 1.62 | 1.81 | 1.27 | 1.49 |
  | M0 (not used) | 2.02 | 2.19 | 1.82 | 2.06 |

  Picks by group: washington 284,282, mendocino_gorda_c 266,502, oregon 102,261, blanco 85,170, pooled 4,361.
* **Pinned prep inputs.** prep copies the five OBS S statics files, Delph2018.nc and (with the floor) the 30 s / 45 s
  floor grids (.bin + .hdr) into prior/ and records source and md5 in prior.json ("pins"); statics and Delph are
  read from the copies. A re-prep is refused if any source md5 changed, or if prior.json has no pins (a root made by
  an older driver, e.g. ensemble_v0.3.2). Members refuse to draw, and the chain refuses to run, when the effective
  prior or a member-relevant configuration key (m0, tiles, dataset, bin_dir, runtime_files, floor, merge_product,
  node_model, mask_damp) differs from what prep recorded: members of one root share one prior. Prep is
  deterministic (a second root reproduced every prior file byte for byte).
* **Locks and process group.** Each member stage runs under <member>/<stage>/LOCK (flock held for the process
  lifetime + pid, pgid, host, start; the merge under merge.LOCK); a second process on the same stage is refused (tested
  on the running M0 stage), the scheduler does not start a locked stage, and one chain runs per root (chain.lock). The
  chain and the members command make themselves process-group leaders; on an exception, SIGTERM or SIGINT they
  SIGTERM the group (SIGKILL after 10 s), so no member stage or tomoDD is left running (tested: exception and
  SIGTERM, child and grandchild processes all gone), and exit non-zero.
* **FAILED_MEMBER.** At the start of a run, a marker of a member that has merged since is stale: renamed to
  FAILED_MEMBER.cleared.<time> and logged. Other markers are listed with the command to clear them; `b2_ens_run.py
  clear-failed KIND NAME..|--all` clears them explicitly (renamed, stage FAILED markers removed).
* **Prior width as parameters (pending PI decision, unchanged values).** prior.rf: std 0.03, len_lat_km 30, len_z_km 8,
  taper, lattice, and rf.long, an optional independent long-wavelength component (std 0 = off; own std, lengths,
  optional taper) drawn from its own random stream, so the rf draws are unchanged whatever rf.long is. Any key can
  also be overridden per configuration (configs.<c>.prior, merged key by key). Changing it needs no code change but
  a new root (the members of a root share one prior; the test member r000 was drawn with 3 %, 30 km / 8 km).
* Smaller: RESUME.md title "Ensemble of model v<version> (configuration <c>; products <version>+ens.<members>)" (was
  "Ensemble v0.3.1"); its launch line carries B2_ENS_CONFIG; tomoDD niceness is absolute (tomodd.ensemble.nice) also
  when the driver itself is niced; `b2_ens_member.py nullmerge`; notes/bayes_design.md corrected: dt.ct is not
  rebuilt from the perturbed absolute times, the code adds each pick's shift to t1 and t2.

### 8.2 Tests (2026-11-25)

* Null draws: `b2_ens_member.py nullmerge` (deterministic v0.4.1 runs, start and reference, D-B073 at 2.1 / 2.5 km)
  against the registered v0.4.1.nc: Vp_tomography, Vs_tomography, Vp, Vp_start, Vs_start, dws_p, dws_s,
  tile_weight_p/s and vs_reverted identical; Vs (rule) differs by 1 float32 ulp at 1,347 nodes (max 2.4e-7 km/s);
  rule lowered Vs at 8,739 nodes as v0.4.1 (ROOT/verify/nullmerge.json).
* Chain path on a scratch root: prep (55 s), prior files byte-identical to the real root's, then exit 2 at the
  calibration gate (the v0.3.2 calibration has not run yet), no member started.
* Test member r000 (real, v0.4.1, floor, at most 4 tomoDD, nice 10): M0 22.7 min, 2.45 GB, abs rms 0.446 s,
  floor 1,577 stops on 212 events, 148 end on the floor, 9 start hypocentres above the floor, 0 above it by > 2 m at
  the end, 0 not met, 0 air quakes, 6,942 of 6,943 events kept. Tiles were still running at hand-back (started 11:38).

### 8.3 Proposed decisions (for the main session; none adopted here)

* P-E11 The v0.4.1 real phase is gated on the v0.3.2 synthetic calibration (calibration_root); no synthetic phase for
  v0.4.1 (its freeze has no syn build). That calibration drew pooled class factors and ran without the floor.
* P-E12 Floor per member stage as in 8.1 (surface.dat md5-locked to the deterministic run's, Air_dep -1000 km).
* P-E13 Start hypocentres stay truncated at the exact merged surface (P-E6), not at tomoDD's floor; starts between the
  surface and the floor are stopped by their first upward update, as in the deterministic runs.
* P-E14 Pick-noise class factors per tile, M0 out of the pooling, 30-event minimum, picks by home tile.
* P-E15 Prior width as parameters (values unchanged pending the PI); a changed prior needs a new root.

### 8.4 Calibration by inflation (D-B094, PI 2026-11-25 15:37) and the real-phase launch

The v0.3.2 synthetic calibration (ensemble_v0.3.2/synth/coverage.json, md5 30d52400..., 97 synthetic members of 102
merged, FAIL by the D-B030 rule) has the hypocentres (68 % 0.70-0.73, 95 % 0.95-0.96 in every tile and coordinate) and
Vp/Vs (0.75 / 0.96) calibrated, and Vp and Vs too narrow on the resolved nodes (Vp 0.565 / 0.849, rms z 1.28; Vs 0.581 /
0.863, rms z 1.22). D-B094: run the real phase on v0.4.1 and multiply the Vp and Vs spreads by factors from the
synthetic members, rounded up to the next 0.05; hypocentres and Vp/Vs not inflated.

**Derivation** (b2_ens_inflate.py, B2_ENS_CONFIG=v0.4.1; output ensemble_v0.4.1/calibration/inflation.{json,md}).
Inflated members x' = median + f (x - median) node by node, so every percentile interval widens by f about the
ensemble median and the std by f. Per node the smallest covering f is (truth - median) / (q_hi - median) above the
median, (median - truth) / (median - q_lo) below it; f68 = the smallest f with 16-84 % coverage >= 0.68 on the resolved
nodes, f95 = the smallest f with 2.5-97.5 % coverage >= 0.95; factor = max(f68, f95) rounded up to the next 0.05. Same
members, truth and masks as b2_ens_coverage.py; at f = 1 the script reproduces coverage.json exactly (it refuses
otherwise). Quantities: the merged member models before the D-B073 node rule against the truth.

| item | nodes | 68 % / 95 % raw | f68 | f95 | factor | 68 % / 95 % after (used factor) | rms z raw / after |
|---|---|---|---|---|---|---|---|
| **resolved Vp** | 888 | 0.565 / 0.849 | 1.301 | 1.366 | **1.40** | 0.707 / 0.956 | 1.28 / 0.92 |
| **resolved Vs** | 320 | 0.581 / 0.863 | 1.245 | 1.335 | **1.35** | 0.722 / 0.953 | 1.22 / 0.90 |
| sampled Vp (check) | 3,521 | 0.569 / 0.849 | 1.280 | 1.437 | 1.45 | 0.716 / 0.945 | 1.33 / 0.95 |
| sampled Vs (check) | 736 | 0.598 / 0.871 | 1.228 | 1.235 | 1.25 | 0.719 / 0.965 | 1.18 / 0.88 |

Per tile (node -> tile of largest merge weight; check only):

| tile | resolved Vp: nodes, factor, after 68 / 95 | sampled Vp: nodes, factor, after | sampled Vs: nodes, factor, after |
|---|---|---|---|
| mendocino_gorda_c | 83, 1.25, 0.759 / 0.988 | 1,096, 1.55, 0.669 / 0.925 | 274, 1.25, 0.719 / 0.971 |
| blanco | 346, 1.35, 0.720 / 0.960 | 294, 1.40, 0.694 / 0.952 | 54, 1.30, 0.722 / 0.982 |
| oregon | 29, 1.80, 0.586 / 0.897 | 470, 1.30, 0.785 / 0.970 | 78, 1.35, 0.795 / 0.962 |
| washington | 430, 1.40, 0.695 / 0.951 | 1,661, 1.45, 0.732 / 0.949 | 330, 1.35, 0.700 / 0.958 |

All 320 resolved Vs nodes are in Blanco (no other tile has resolved S nodes), so the Vs factor is Blanco's; the sampled
Vs nodes of the other tiles need 1.25-1.35, not more. The resolved-node factors are taken as the rule says: the
sampled-node Vp factor is 0.05 higher (1.45; with 1.40 its coverage is 0.716 / 0.945), Mendocino-Gorda's sampled Vp
needs 1.55 (0.669 / 0.925 with 1.40), Oregon's 29 resolved Vp nodes 1.80 (N_eff of a tile is a few: noisy). Gaussian
version (mean +- f std): resolved Vp f68 / f95 1.267 / 1.275, Vs 1.199 / 1.232; with the used factors 0.733 / 0.965
(Vp) and 0.719 / 0.963 (Vs). After inflation the four pooled items (resolved and sampled, Vp and Vs) are inside the
D-B030 bands (68 % 0.60-0.76, 95 % 0.90-0.98). Per tile, 8 of 13 items are inside; outside: Oregon resolved Vp 0.586 /
0.897 (29 nodes, still narrow), Mendocino-Gorda resolved Vp 95 % 0.988 (83 nodes, wide), Oregon sampled Vp 68 % 0.785
and Vs 68 % 0.795 (wide), Blanco sampled Vs 95 % 0.982 (54 nodes, wide). One factor per quantity for the whole margin,
as D-B094 says; the tile differences are reported, not used.

**Products** (b2_ens_stats.py products; stats.npz stays raw): with configs.<c>.calibration.inflation set, Vp_std,
Vs_std, Vp_tomography_std, Vs_tomography_std -> f std; their p05 / p95 -> median + f (p - median); the tomography
std_total from the inflated std and the (not inflated) tile disagreement. Raw values kept as <name>_raw (14 variables).
Mean, median, n, VpVs not changed. Attributes: the title / summary name the factors; provenance_uncertainty_calibration
(decision, factors, rule, source coverage.json and md5, derivation file md5 and whether its factors equal the
configuration's, resolved coverages before / after); each inflated variable carries calibration and inflation_factor.
The report has an "Uncertainty calibration" section (factors, rule, source, derivation table, all items) and labels the
std maps "calibrated x f"; its synthetic coverage table now comes from the calibration root. provenance_prior is the
effective prior (C.PR). Tested: inflate() equals inflating random members about their median (std, p05, p95); raw kept;
VpVs, mean, median, n untouched; dry run on the r000 statistics (15.4 MB with the raw variables, 11.0 MB before).

**Gate.** params tomodd.ensemble.configs.v0.4.1.calibration = {accepted: D-B094, inflation: {Vp: 1.40, Vs: 1.35},
coverage_md5: 30d524003e05f2c02f5a0afa9aedfdf5}. After a failed calibration the chain goes on only when this entry
has a decision id, factors for Vp and/or Vs only (finite, >= 1), covers every failed verdict item, and its md5 equals
the calibration root's coverage.json (b2_ens_common.calibration_acceptance); the calibration root is read only. Scratch
root test (prep, then B2_ENS_STOP_AFTER=calibrate): with the entry, "ACCEPTED ... (D-B094)", exit 0; without it exit 2;
wrong md5, Vp only (Vs failed and not inflated), a hypocentre key, no decision id, a factor 0.9: exit 2 each; no
member started. The scratch prep reproduced the real root's prior files byte for byte (only root paths in floor.json
and prior.json differ), so the chain's prep on the real root (its chain.state had no prep) leaves r000 valid.

**Resources** (PI: the whole machine today): max_procs 64 (was 56), min_avail_gb 30 (was 40), mem_per_run_gb 3.5:
a tomoDD stage starts only while MemAvailable - 3.5 GB x (stages started in the last 5 min) >= 33.5 GB. Beside it:
the v6 grid chain (36 x ~1.6 GB), NLLoc runs and the cross-correlation job (up to 48 I/O workers).

**Launch** (real members r000..r099, r000 reused; then stats and products as a dry run + report; registration only
with B2_ENS_REGISTER=1 after the PI's go):

    cd /home/mdenolle/wt_ens41
    B2_ENS_CONFIG=v0.4.1 LD_LIBRARY_PATH=/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/lib setsid nohup /home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/bin/python workflow/10_tomography/b2_ens_run.py chain >> /wd1/mdenolle_data/tomo_v1/ensemble_v0.4.1/chain.log 2>&1 < /dev/null &

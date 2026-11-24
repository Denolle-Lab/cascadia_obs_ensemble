# Track B status: tomography (tomoDD) and relocation

One page that says where every step stands. Decisions are in `DECISIONS.md` (D-Bxxx), parameters in
`params.yaml`, details in `notes/`. Update this file whenever a step changes state.
Last update: 2026-11-24 11:35 (handoff 11-24 07:55 below, read first).

Naming: **v4-NLL3D** = catalog v4 relocated with NonLinLoc in the 3D starting model (formerly
"Track A", D-B024). Starting model and locations: D-B004, D-B005.

## Update 2026-11-24 11:35: B5 whole-catalog tomoDD run b5_v0.3.1 done, NOT a catalog version (session -18)

Branch tomo/b5 (pushed, 4d894f31; worktree ~/wt_b5_full): b5_relocate.py/.sh (prep, split, run, collect; resumable;
crash retry with frame centre moved 0.5 km for NaN rays in RaySPDR2.f), params b5, notes/b5_relocation.md (proposed
decisions, not in DECISIONS.md). Catalog /wd1/mdenolle_data/tomo_v1/b5/b5_v0.3.1/catalog_b5_b5_v0.3.1.csv;
summaries data/tomo_v1/b5/b5_v0.3.1/compare.md (on tomo/b5). 2,125 dt components (3 split in 2 with a 2-hop halo,
0 pairs lost), 2,128 frozen-binary runs, 20 min wall at 40 processes.
* Relocated 97,619 of 157,269; 54,548 unlinked, 3,607 air quakes removed by tomoDD (3,039 Axial), 1,480 dropped, 15
  skipped: all keep B4 positions. Shift from B4 median 0.59 km h / 0.52 km |z|; centroid shift removed (D-B053).
* **Does not improve the catalog:** dt residual rms in the B4 grids P 0.147 -> 0.148 s, S 0.181 -> 0.232 s (worse in
  every region); cluster thickness B5/B4 median 1.00. As in D-B048: catalog dt add little to B4.
* **D-B055: 869 relocated events above the surface** (median 0.25 km, max 5.2 km; all below it at B4). Cause: the
  same floor-held events as the upward pull (36 % of events < 0.1 km below the surface at B4 end above it) + tomoDD has
  no topography (Air_dep = one flat level at the highest station). Damping 400 (CND ~40) halves Axial's count.
* **S worsening, root cause found (11:55, tomo/b5 99c488ed, notes/b5_s_mismatch.md):** NOT two forward models (tomoDD and the B4
  grids agree on dt to MAD 2-5 ms; S misfit grows in tomoDD's own model too). Cause: S dt weighted 1/4 of P (D-B008 sigma
  S = 2 x P) + MAD re-weighting, so B5 fits P at S's cost. Equal weights without MAD cuts: S improves (Blanco 0.269 -> 0.234 s,
  onshore 0.166 -> 0.154 s), P unchanged or better. Measured S/P noise ratio 1.16-1.51 (catalog 1.23), not 2. Proposed P-S1..P-S3.
* **Weighting test (12:45, tomo/b5 e1f14107, notes/b5_weighting_test.md):** 26 clusters (all regions), 18 variants (S weight 0.25/0.5/meas/1/1.5/2 x MAD cuts prod/loose/none), station cross-validation + synthetics. S weight = P (1.0) with production MAD cuts: held-out P -6 %, S -11 % vs production (24/26 clusters both), synthetic error -10 %; optimum flat 1.0-2.0; measured-ratio weights worse than equal; no MAD cuts worse. Above-surface count unchanged by weights. Proposed P-W1..P-W4.
* **PI adopted D-B061..D-B064 (13:00):** S weight 1.0, production MAD cuts, no measured-ratio weights, CV acceptance. In the B5 driver (tomo/b5 ea682f49: params b5.tomodd.s_weight -> WTCTS 4.00; chain stage accept = b5_wtest.py accept RUN, exit 2 on FAIL). Smoke test on b5_v0.3.1: PASS every region (held-out P 0.86-0.99, S 0.85-0.96 of the old weighting). Air quakes in the test: 61 -> 35 with S weight 1.0 (a first count of 0 was a parsing error).
* **Next for B5:** run B5 on v5.1 (after the round-2 NLLoc rerun)
  after v5.1 (round-2 upward-pull fix); per-cluster damping to CND 40-80; whether waveform CC dt (D-B060 second pass)
  comes first, since catalog dt do not sharpen clusters; air-quake and two-event-component handling; bootstrap not run
  (2 CPU-h per member; tomoDD bootstrap 2-11x too small in D-B048).

## Update 2026-11-24 11:00: catalog v5.0, Zenodo, atlas, B5 (session cascadia-obs-ensemble-18)

Split with the round-2 session (agreed by message): this session owns catalog versions, Zenodo, atlas and B5; the
round-2 session owns the NLLoc fixes and the iter0/iter1 rerun for v5.1. Decision numbers: D-B058..D-B064 here,
D-B065..D-B079 the round-2 session, D-B080+ here (agreed 15:00).

* **catalog_version.py ready for v5.1 (15:05):** per-version tier U mode (params `tier_u`: v4 for 5.0, D-B059; none
  afterwards: no hypocentre, u_reason), surface check only on rows with a hypocentre, every B4 catalog column carried but
  the build stops on a column without a README description (params `columns`, filled by the round-2 session for
  floor_limited, far_only, deep_pass, u_reason), flag counts (params `flags`), `--test DIR` dry build. v5.0 known limits
  moved to params, with the round-2 numbers (floor-limited ~6 %, arrivals beyond ~160 km); they enter the v5.0 README at
  its next rebuild. Checks: v5.0 rebuilt in scratch gives the published CSV (md5 23781b52) and files; README differs only in
  the known limits; a mock v5.1 catalog builds and stops on undocumented columns and on a tier U mode mismatch.

* **PI (10:10):** D-B058 catalog versions v5.0 (now) / v5.1 (after round 2) / v5.2 (B5 tomoDD); D-B059 tier U keeps
  its v4 hypocentre in v5.0, flagged (exception to D-B055); D-B060 B5 with catalog dt now, waveform CC dt later.
* **v0.3.1 chain finished 10:44** (1464b4f8): 157,269 located, 0 located events above the earth_relief 01m surface.
  Acceptance (b4_acceptance.py -> data/tomo_v1/b4/b4_iter1_v0.3.1/acceptance.md): Axial A-B 28 % within 2 km of the
  seafloor (target 85-90 %; median 2.4 km bsf; 1.2 km shallower than v0.3.0, 0.9 km deeper than v4-NLL3D) FAILS;
  Puget/onshore/forearc/Mendocino/Gorda unchanged from v0.3.0. Event-paired (round-2 session, 1a7a6a3b): Mt St Helens
  +0.49 km vs Hirao 2025 (passes); Blanco +1.99 km vs Lange 2026 (fails; West Blanco +4.64, worse than v0.3.0).
* **Catalog v5.0 built** (catalog_version.py, params catalog_versions): /wd1/mdenolle_data/catalog_v5/v5.0/
  (cascadia_catalog_v5.0.csv md5 23781b52..., README with known limits, station terms, B4 reports), registry
  data/catalog_v5/registry.csv. 172,342 events (v4.1 list: 172113 out, 900001-3 in with ComCat hypocentres),
  157,269 nll3d, 15,070 v4 (tier U), 3 comcat; 4,077 above the surface, all tier U at v4 positions -> publishable false.
  Magnitudes = v4.1 Route A (v4 hypocentres; STATUS step 9 still to do).
* **Zenodo:** catalog_v5.zip (all v5.x versions, replaced at each version) in the private draft 23084158, remote md5
  257a425b... OK (data/catalog_v5/zenodo_uploads.csv). The draft is now at Zenodo's 100-file limit: any new product
  must replace an existing file or go into an existing zip.
* **Atlas (public, gh-pages 2a6ac1c5):** catalog v5.0 is the relocated catalog, moves from v4-NLL3D, model v0.3.1.
  Built from atlas/model-layer a69a22ab with arguments only (--catalog2 <v5.0 csv> --catalog2-label "Catalog v5.0
  (interim)" --catalog-before catalog_v5_trackA_iter1_vpvs176.csv). The atlas flags 4,787 events above its GMRT
  terrain: the 4,077 catalog flags + 725 located events 0-354 m (median 17 m) above GMRT where GMRT is deeper than
  earth_relief 01m (floor-held events; list /wd1/mdenolle_data/tomo_v1/above_surface/v5.0_atlas_gmrt_air_rows.json);
  stated in the legend, flag not changed.
* **B5 (D-B053, D-B060):** whole-catalog tomoDD relocation-only driver being built by an agent on branch tomo/b5
  (worktree ~/wt_b5_full, from tomo/tiles + tomo/b5-blanco-test), outputs /wd1/mdenolle_data/tomo_v1/b5/<run>/,
  notes/b5_relocation.md; production on b4_iter1_v0.3.1 / model 0.3.1 with frozen binaries.

## Handoff 2026-11-24 07:55 (read first)

**Running (detached, survives logout).** `b4_relocate.sh` VERSION=0.3.1, process group 811130, private copy
`/wd1/mdenolle_data/tomo_v1/.b4_relocate_running_v0.3.1.sh`, log `/wd1/mdenolle_data/tomo_v1/b4_relocate_v0.3.1.log`,
state `b4_relocate_v0.3.1.state`. Done: merge (v0.3.1 = D-B049 gate on tiles smoothed <= 100, D-B050 ridges_north
out, D-B051 Vs revert), bundle (velocity_models.zip on the private Zenodo draft 23084158, remote md5 OK), commit_model,
gate (model v0.3.1 on the public atlas first: gh-pages c4d55588). Now: grids (started 07:29, ~2 h), then iter0,
terms, iter1, catalog, publish (git + PR comment only; nothing goes to Zenodo). PI chose (07:50) to let it run with the
current NLLoc topography mask; iter0/iter1 rerun once the mask fix is tested (~1.5 h). If it stopped, resume from
`~/wt_tomo_res` with `VERSION=0.3.1 setsid nohup bash workflow/10_tomography/b4_relocate.sh 9>&- >> <log>`.

**PI rule D-B055 (11-24).** No event above the ground or seafloor in a published catalog; such events are computation
errors to fix at the root, never moved (a clamp was tried and reverted: bf69f95c, atlas a69a22ab). b4_catalog.py now
writes surface_km / above_surface and "publishable" (false while any event is above). Counts: B4 v0.3.0 1,079 located
events above (median 30 m, max 0.41 km), tier U 4,080 of 15,070 (v4 positions), catalog v4 40,553 of 172,340 (23.5 %,
median 2.1 km above; 71 % of Axial).

**Root-cause workflow `above-surface-root-cause` (run wf_5c33d6dc-211, task wv14l12u2).** Four investigators, each
followed by an adversarial verifier: A GENIE/GraphDD conventions and Ian's inputs (data/vel_*.csv start at depth 0, no
water layer), B v4 depth forensics (datum hypotheses vs ANSS, NLL, Axial literature), C NLLoc LOCTOPO mask (suspected:
velmod.c get_surface_z reads the 01m grid by node lookup, the check is bilinear; fix + scratch test on ~400 events),
D tier U (why 15,070 events are not located; relaxed-settings test). Scratch outputs:
`/tmp/claude-39101/-home-mdenolle-cascadia-obs-ensemble/06a5cbfb-0ae0-4565-b809-72812014f122/scratchpad/above_surface/`.
If the session died, resume (finished agents come from cache): Workflow scriptPath
`~/.claude/projects/-home-mdenolle-cascadia-obs-ensemble/06a5cbfb-0ae0-4565-b809-72812014f122/workflows/scripts/above-surface-root-cause-wf_5c33d6dc-211.js`,
resumeFromRunId wf_5c33d6dc-211, same args {"scratch": "<scratchpad>/above_surface"}.

**Round 1 results (08:15).** Report `data/tomo_v1/above_surface/above_surface_round1.html`; evidence archived in
`/wd1/mdenolle_data/tomo_v1/above_surface/round1_2026-11-24/`. v4: no surface in GENIE/GraphDD, search domain up to +5 km,
rock in water/air, Axial model 2x too fast + mirror ambiguity -> v4 depths not publishable (paper uses v4.1: PI decision).
NLLoc: nearest-node surface lookup -> FIXED (D-B056, reloc/track-a 85cc3dae, topo_15s_cons_eft.bin, used from v0.3.1
iter0). Open: upward pull (~4 % of B4 events pressed on the floor; stations 50 m below surface in a1_build_model.py:253),
station reach (tile box drops 6.1 % of arrivals), 80 km floor, tier U (5-pick events; v4 hypocentres copied).
**Round 2 running** (workflow nlloc-upward-pull-and-reach, run wf_2c163670-389, task wqzs0ijq7; scratch
`<scratchpad>/above_surface2`; resume with resumeFromRunId and the same args). T aims for a tested station-reach fix
before v0.3.1 iter0 (~11:00): if ready, apply to a2_run_nlloc.py before iter0 starts, else rerun iter0/iter1 after.

**Surface definition (from session -18, 10:50).** In v5.0, 725 located events are below the earth_relief 01m surface
(D-B055 test, D-B056 floor) but above the atlas GMRT multibeam terrain, which is lower there (median -41 m, down to
-0.5 km): median 17 m, max 354 m above it (Axial 267, Blanco 191, JdF 65, ...). They are the floor-held (upward-pull)
events. List: /wd1/mdenolle_data/tomo_v1/above_surface/v5.0_atlas_gmrt_air_rows.json. For the round-2 synthesis: build
the floor and the D-B055 test from the best bathymetry (GMRT where available, earth_relief elsewhere) and check that the
3D model's seafloor/water boundary uses the same surface.

**Next, in order.** (1) Read the workflow results; implement the tested NLLoc mask fix (Track A template in
~/wt_reloc, used by b4 iter0/iter1); rerun iter0/iter1/catalog of v0.3.1 with new run names. (2) Acceptance tests of
v0.3.1 (Axial >= 85 % of A-B events above 2 km bsf vs Waldhauser 2020; Mt St Helens within ~0.5 km of Hirao 2025;
Blanco within 1 km of Lange 2026; Puget unchanged). (3) Tier U and catalog v4 plan for the PI (v4 depths are not
publishable as they are). (4) Atlas with the v0.3.1 catalog. (5) PR #35: Copilot's last issue fixed (9936f8bd, reply
posted); merge into tomo/track-b after a final look, then the part-2 results PR, close #32/#33.

**PI answers (09:15).** Fix 6 kept; no email to Haijiang Zhang (do not contact him); paper hypocentres = final NLLoc
relocation (D-B057). Round 2's first run died with the earlier Claude Code process (~09:30, no results); relaunched 09:40 as run
wf_991a6f87-ddc (script ..._v2.js; reuses the partial files in the scratch folder; large files in
/wd1/mdenolle_data/tomo_v1/above_surface/round2_scratch/). v0.3.1 iter0 started 09:26 with the D-B056 floor: first
24,000 located events, 0 above the surface. PR #35 merged into tomo/track-b (bda52686).

## Handoff 2026-11-23 08:10

**Where the work is.** Worktree `~/wt_tomo_res`, branch `tomo/tiles` = PR #33, stacked on `tomo/res-tests-mg_v2` = PR #32
(Mendocino-Gorda v2 resolution tests), stacked on `tomo/track-b`. PR #32 also holds a snapshot of the code that was
uncommitted in the main checkout (the main checkout has not changed since; it still sits on `tomo/track-b` with that
work uncommitted). The main checkout is not used for this work.

**What was decided today.** D-B033 tiles (blanco, oregon, washington, ridges_north beside Mendocino-Gorda); D-B034
FORMAT-fixed tomoDD builds (missing comma at tomoDD-SE.f line 2324 stopped some runs); D-B035 offshore forearc kept
(soft selection rule, soft-bar OBS keep all picks; datasets `b1_dataset_c`, `b1b_balanced_c`; 28,986 events).
mendocino_gorda_v2 / v2s and the ensemble work stay on the old dataset; `mendocino_gorda_c` is Mendocino-Gorda on
the new one.

**Running (detached, survive logout; not a reboot).** `b2_res_pipeline.sh <tile>_v2` for mendocino_gorda_c, blanco,
oregon, washington at damping 1000 (53 inversions each: 5 real-data scan runs + checkerboards + lattice spikes).
At 08:10: blanco 40/53, oregon 32/53, mendocino_gorda_c 8/53, washington 3/53. Each tile commits its results to
`tomo/tiles` and comments on PR #33 when done. Progress: `res_tests/queue/runner.log` and `pipeline.log` under
`/wd1/mdenolle_data/tomo_v1/b2_tomodd/<tile>_v2/`. If a run stopped (reboot, kill), resume with the same command
from `~/wt_tomo_res`; finished pieces are skipped:

    setsid nohup bash workflow/10_tomography/b2_res_pipeline.sh blanco_v2 \
        >> /wd1/mdenolle_data/tomo_v1/b2_tomodd/blanco_v2/pipeline.log 2>&1 < /dev/null &

**Update 09:05.** blanco and oregon finished at damping 1000 (on PR #33): over-damped like ridges (real-data condition
number at s20: 11 and 14, against ~60 for Mendocino-Gorda v2; no lattice spike recovered >= 50 %). PI asked for
damping 200 and 300 on both: `DAMP=200` and `DAMP=300` pipelines launched 09:05 (51 runs each, results in
`data/tomo_v1/b2_res_tests/<tile>/d200|d300/`). washington and mendocino_gorda_c still running at damping 1000.
*Correction (12:30):* the 11 and 14 are last-iteration condition numbers; the Mendocino-Gorda ~60 is first-iteration
(its last-iteration value is 19). Like for like, Blanco and Oregon at damping 1000 have 35 and 39, and damping 300
gives 170 and 200 (notes/merge_design.md section 2).

**Update 12:30.** Done and on PR #33: blanco and oregon at damping 200 and 300, mendocino_gorda_c at 1000.
washington at 1000 still running (real-data scan done: first-iteration CND 68 at damping 1000, like Mendocino-Gorda).
PI chose damping 200 for ridges_north (D-B043). Merge design and tile-vs-tile tests proposed
(notes/merge_design.md, P-M1..P-M8; `b2_tile_overlap.py`, results in data/tomo_v1/b2_merge/overlap/): no overlap
is resolved by both tiles; mixing dampings does not make a worse seam than either damping alone; the same events
relocated in two tiles differ by 0.2-1.0 km horizontally (median).

**Update 14:15.** washington_v2 at damping 1000 finished 12:05 (on PR #33): the best-resolved tile so far (18 P
spikes recover >= 50 %, 14 of them at DWS > 1,000; checkerboards resolve about 30 % of the box at 15-18 km for
40-60 km cells; Vs as well as Vp). PI decisions: M0 coarse whole-margin start (D-B044) and one small damping for
all tiles, 200 (D-B045). Implementing D-B045 showed that pick weights were scaled per tile (Oregon and Washington
picks weighed 3 times Blanco's, Mendocino-Gorda 1.8 times), which changes what one damping number does in tomoDD;
from now on all builds use one weight scale for the whole dataset. Launched 14:03, detached:
`b2_t6_m0.sh` = T6 joint blanco + oregon against the stitched pair at damping 200, and M0 (margin_m0_v2, 6,943
declustered events, 518 stations, new 560-station tomoDD build) at damping 200 and 1000. Log
`/wd1/mdenolle_data/tomo_v1/b2_tomodd/t6_m0.log`; it commits to `tomo/tiles` and comments on PR #33 when done.
Resume after a stop with the same command (finished stages are skipped):

    setsid nohup bash workflow/10_tomography/b2_t6_m0.sh \
        >> /wd1/mdenolle_data/tomo_v1/b2_tomodd/t6_m0.log 2>&1 < /dev/null &

Next after review of M0: rebuild every tile from the M0 model (builder option start_model, written) and rerun the
damping-200 run and resolution tests per tile.

**Update 16:15.** T6 and M0 done (data/tomo_v1/b2_merge/t6/, m0/; notes/merge_design.md section 6 item 7 and the
M0 paragraph after section 7). T6 passes in the tile interiors: joint and stitched differ by 1.7-2.0 % rms in Vp
(correlation 0.86-0.94), the same as one tile moved from damping 200 to 300; the joint model is about 15 % stronger
(more data rows, so damping weighs less); in the overlap, which neither tile resolves, the stitched model has half
the joint amplitude; overlap events differ by about 1 km between joint and tile runs. M0 at smoothing 20 was
node-scale noise (7 % rms Vp); a smoothing scan (20-1920, damping 200) has a flat weighted fit up to 480 and a
coherent model there (slow ridge crust at 7-15 km, fast 40 km under Washington). Proposed P-M11: M0 = d200_s480,
then tiles restart from it with a smoothing scan each (builder option `start_model` written, bc73862d). Waiting
for the PI before the tile reruns (about one day of the machine).

**Update 16:25.** PI adopted M0 = margin_m0_v2 d200_s480 (D-B046). Figures of the M0 model itself:
data/tomo_v1/b2_merge/m0/model_maps_d200_s480.png and model_sections_d200_s480.png (b2_m0_model_figs.py). Launched
16:24, detached: `b2_tiles_m0.sh` = the five tiles restarted from M0 (regions <tile>_m0: start = D-B005 x M0 change,
P-4 from <tile>_v2, Vp/Vs floor 1.65), smoothing scan 20-480 at damping 200, smoothing chosen by rule, published to
PR #33. Log `/wd1/mdenolle_data/tomo_v1/b2_tomodd/tiles_m0.log`; resume with
`setsid nohup bash workflow/10_tomography/b2_tiles_m0.sh >> /wd1/mdenolle_data/tomo_v1/b2_tomodd/tiles_m0.log 2>&1 < /dev/null &`.
Next (phase 2): resolution tests of each <tile>_m0 at damping 200 and its chosen smoothing (b2_res_pipeline.sh
needs a smoothing option and res_tests.regions entries for the _m0 names).

**Update 17:10.** PI: version and export the models (D-B047: Zarr + netCDF-4 Classic with EMC/CVM attributes +
NonLinLoc grids, registry data/tomo_v1/models/registry.csv, Zenodo bundle velocity_models.zip in draft 23084158;
tutorial read_velocity_model.py); the merged model is the Zenodo product; go on to relocation and stop before
GrowClust. v0.1.0 = M0 registered and uploaded. Three detached jobs now chain to the end (each resumable with the
same command from ~/wt_tomo_res; logs under /wd1/mdenolle_data/tomo_v1/):
1. `b2_tiles_m0.sh` (b2_tomodd/tiles_m0.log): tiles from M0, smoothing scan, choice by rule.
2. `b2_tiles_m0_phase2.sh` (b2_tomodd/tiles_m0_phase2.log): waits for 1, then the resolution tests of the five
   tiles at damping 200 and the chosen smoothing (b2_res_pipeline.sh SMOOTH= DAMP=200).
3. `b4_relocate.sh` (b4_relocate.log, state b4_relocate.state): waits for 2, then b2_merge.py -> v0.3.0 (registered,
   NonLinLoc grids at cell centres), b4_time_grids.py (696 stations, D-B028 P-1/P-5: cell-centred, EFT, Richardson
   1/0.5 km) -> reloc_v5/time_b4_v0.3.0, Track A NLLoc iter0 -> station terms -> iter1 (runs b4_iter0/iter1_v0.3.0),
   catalog (a2_build_catalog.py, b4_catalog.py: EFT depths back, comparison with v4-NLL3D iter1), Zenodo bundle
   with the merged model as product, commit and PR comment. Stops before GrowClust.
Resume any of them with `setsid nohup bash workflow/10_tomography/<script> 9>&- >> <its log> 2>&1 < /dev/null &`
(close fd 9: a launch from a shell that holds the publish lock passes the lock to every child).

**Update 19:15.** PI: no smoothing cap; the chain runs on the adopted rule (P-M11 / D-B046): Mendocino-Gorda,
Oregon and Washington at smoothing 480 (scan edge), Blanco 20, ridges 60. Whether the smoothing costs location quality
is judged from the B4 relocation (misfits, depths against v4-NLL3D). Session handover: the session that ran the chain
since 09:00 lost its sign-in at 18:55; this one took over, no job affected (all detached). At 19:09 Mendocino-Gorda
and Washington phase-2 tests 16/20, the others done; b4_relocate.sh waiting on them.

**Update 19:25.** PI: once v0.3.0 is built, upload it to Zenodo and put it on the atlas/dashboard *before* the
relocation. `b4_relocate.sh` relaunched 19:19 with the order wait -> merge -> bundle (Zenodo upload) -> commit_model
-> **gate** -> grids -> iter0 -> terms -> iter1 -> catalog -> publish. The gate waits for
`/wd1/mdenolle_data/tomo_v1/b4_gate_v0.3.0.release`; create it (one line saying why) once the model is on the atlas.
PR #33: description rewritten as a reading guide; the 16 automatic comments of the superseded `_v2` runs hidden as
outdated. To do, in order:
1. v0.3.0 built and uploaded (automatic, after washington_m0 phase 2, ~20:00).
2. Model layer on the atlas/dashboard (workflow/06_analysis/build_v4_atlas.py, branch atlas/v4-nll3d), then release
   the gate.
3. B4 relocation (automatic after the gate; grids ~2 h, NLLoc two passes), stop before GrowClust.
4. 2026-11-24: split PR #33 into code + decisions and generated results.

**Update 19:45.** PI chose a *private* preview for the model layer (gh-pages atlas stays unchanged). Atlas code:
branch `atlas/model-layer` (worktree ~/wt_atlas_model, commit 49c64544), `build_v4_atlas.py --model <netCDF>` adds a
"Velocity model" layer (one depth slice of dVp, dVs, Vp/Vs, Vp, Vs; DWS mask 0/10/100/1000). Private page:
https://claude.ai/artifact/9tH3U7Se35ds5h6yoBo8dj (now M0 v0.1.0; republish the same page with v0.3.0, from another
session pass that URL). Steps once `[b4] stage gate` appears in b4_relocate.log:
  1. `cd ~/wt_atlas_model && LD_LIBRARY_PATH=$ENV/lib $ENV/bin/python workflow/06_analysis/build_v4_atlas.py --model
     /wd1/mdenolle_data/tomo_v1/models/cascadia_obs_tomo/v0.3.0.nc --out <scratch>/atlas_preview.html`
     ($ENV = ~/cascadia_obs_ensemble/.pixi/envs/default; missing untracked inputs are symlinked from ~/wt_atlas/data)
  2. strip the document wrapper tags, title "Cascadia Atlas Model Preview", publish to the URL above
  3. `echo "model v0.3.0 on the private atlas preview <date>" > /wd1/mdenolle_data/tomo_v1/b4_gate_v0.3.0.release`

**Update 20:20.** v0.3.0 built 19:45, uploaded to the private Zenodo draft (velocity_models.zip 0.64 GB, remote md5
ac93e555fe2affa8796c38c8840c9305 OK), committed, on the private atlas preview (default ray mask DWS >= 10, option
"resolved in checkerboard tests"); gate released 19:57; B4 travel-time grids running (all 172,340 v4.0 events are
relocated, not only the tomography subset; v4.1's three added events keep their ComCat hypocentres and 172113 is
spurious, to handle when the B4 catalog is assembled). Added (D-B048): after B4, test tomoDD (fixed velocity) against
GrowClust3D on the Blanco swarms before B5. Proposed, not yet agreed: compare B4 NonLinLoc locations of the ~29k
tomography events with their tomoDD final locations (decides whether one more tomography-location iteration is needed:
median difference < 0.5 km and no depth offset -> none).

**Update 23:10. B4 done** (22:57, PR #33 comment; data/tomo_v1/b4/b4_iter1_v0.3.0/compare.md; catalog
/wd1/mdenolle_data/reloc_v5/catalog/catalog_b4_b4_iter1_v0.3.0.csv). Driver fixes: a2_parse_hyp.py before the station
terms and the catalog (5e6166fe); wt_reloc/data links to the v4.0 catalog backup and anss_2010-16.csv (gitignored).
157,270 of 172,340 events located in 3D (15,070 tier U as before). Median rms 0.110 s (v4-NLL3D iter1 0.124); median
shift 1.4 km depth, 1.4 km horizontal. Smoothing 480 costs nothing measurable: in the s480 tiles' regions rms drops
6-16 %, tiers unchanged, depths within 0.5 km. Blanco and Gorda improve. **Axial worse**: median depth 3.0 -> 5.0 km
bsl (below the magma chamber), tier A-B 50 % -> 5 %, h_max 1.8 -> 3.1 km, rms 0.054 -> 0.060 s; this is the whole
drop in A-B tiers. Cause: v0.3.0 at Axial comes from ridges_north (relocation only, D-B038, unresolved) and M0's slow
ridge crust. **Open for the PI:** (1) keep v4-NLL3D locations at Axial, flagged (recommended now); (2) leave
ridges_north out of the merge near Axial and relocate Axial again (~1 h); (3) accept B4. Also open: B4 vs tomoDD final
locations check; v4.1 additions and spurious 172113 in the B4 catalog; D-B048 test (after the Axial choice).

**Update 2026-11-24 05:10.** PI: the velocity model and the catalog can be public. Public atlas
https://denolle-lab.github.io/cascadia_obs_ensemble/ (gh-pages 2b24559a, from atlas/model-layer b34a9abf) now has
model v0.3.0 as 3D textures, as in the mt-rainier-digital-model viewer (horizontal slice at any depth, the model on
the cross-section plane and behind the events in the 2D section, nothing above the ground or seafloor; DWS >= 10 mask
by default, checkerboard-resolved option) and the B4 relocation as the second catalog ("Relocated in model v0.3.0
(preliminary)", Axial caveat in the legend). Checked headless (Chromium + SwiftShader): no console or shader errors.
The Zenodo draft stays private (publishing the record is a separate PI decision). Axial choice still open.

**Update 2026-11-24 06:00.** Public atlas (gh-pages 7df06a89, atlas/model-layer 7855966c): vertical model cut
(east-west or north-south, slider; ground removed in front; section panel and camera follow), "Moves" layer (line from
the v4-NLL3D location, start model, to the B4 location; red deeper, blue shallower; in 3D and in the section panel:
157,225 events located in both, median depth change +0.00 km, median |change| 1.40 km, 50 % deeper), section panel
fits by default, seismic colormap for the model (PI). PI suspects events went too deep: a literature comparison of
depths by region is needed (not started). PR split: part 1 = PR #35 (tomo/b2-code -> tomo/track-b, code, decisions,
pixi models env, OBS statics; Copilot review requested); part 2 (results) after #35 merges. D-B048 test running in a
background agent on branch tomo/b5-blanco-test (worktree ~/wt_b5; Julia + GrowClust3D install under
/wd1/mdenolle_data/b5_test).

**Update 2026-11-24 07:00.** (1) Literature comparison of depths by region done:
data/tomo_v1/depth_literature/depth_literature.html (95 verified studies, 7 unverified listed apart; AI-checked, not yet
re-read by a person). B4 consistent within about 1 km in Puget/Georgia, Oregon/SW Washington, the offshore forearc and at
Mount St Helens (Hirao et al. 2025, +0.5 km); too deep at Axial (+2 to +2.5 km, against Waldhauser et al. 2020; v4-NLL3D
agreed), Endeavour/Explorer/JdF ridge (depth unconstrained), Blanco (+2.2 to +2.4 km against Lange et al. 2026), Gorda
interior (+8 to +12 km) and Mendocino (+2 to +4 km against NCSN, already in v4-NLL3D). (2) D-B048 done (branch
tomo/b5-blanco-test, notes/b5_blanco_test.md): with catalog differential times GrowClust3D is not more accurate than
tomoDD; neither resolves swarm depth; GrowClust3D unstable from good starts; tomoDD relocation-only needs a second FORMAT
fix (lsfitHFDD_lsqr_lm5.f line 504, build tomoDD-SE_O3nofs_e10k_fmt2, PI approval pending). (3) PR #35 review: 24 of 26
findings confirmed, fixes made and verified (uncommitted in ~/wt_split); among them B4 near-station travel times (negative
within 2 km of 101 grids) and 0.01 deg rounding of the B4 catalog. (4) PI approved for v0.3.1: resolution-gated merge
(D-B049), ridges_north out of the merge (D-B050), Vs reverted where Vp/Vs < 1.6 or dws_s < 10 (D-B051), 0.5 km grid
for start-up-offset outlier grids (D-B052), catalog depth-quality and quarry flags; implementation running.
(5) Incident: a fix agent's test rewrote blanco_m0/inputs and 71 P-1 grids on /wd1 at 06:01 (contents identical to the
record, 11/11 md5 match); timestamps and manifest restored from manifests.jsonl line 1425, the 3.5 GB work folder deleted
(PI approved).

Options: `DAMP=<d>` (tests and real-data runs at damping d; results in `<tile>/d<d>/`), `SPIKES_ONLY=1` (spikes at
the event clusters with the DWS mask, `spikec_*`), `MAXJ` (parallel inversions, default 10).

**ridges_north result (done, all on PR #33).** Lattice checkerboards and spikes: nothing resolved at any damping
(1000, 300, 200, 100). Spikes at the event clusters with DWS >= 100 (52 of 105 P spikes sampled): median recovery
22 % (damping 100), 21 % (200), 19 % (300), 10 % (1000); only 3 spikes >= 50 %, all at 12 km between northern Blanco
and Axial (44.4-45.4 N, 130.0-130.3 W, DWS 1,165-7,203, smearing about one node). Real-data condition number at
s20, last iteration: 87 (d100), 33 (d200), 21 (d300), 6 (d1000); first iteration: 290, 119, 69, 15 (Mendocino-Gorda
v2 chose 1000 at first-iteration CND ~60, last-iteration 19).

**Open, for the PI.**
1. Resolved 2026-11-23 (D-B043): ridges_north damping 200. The D-B038 exception volume is not recovered at 200 and
   stays on the damping-100 model unless dropped.
2. Resolved 2026-11-23 (D-B045): damping 200 and smoothing 20 in every tile and in M0, pick weights on one scale
   for the whole dataset.
3. Resolved 2026-11-23 (D-B038): ridges_north for relocation only, except the 12 km volume between northern Blanco
   and Axial where cluster spikes recover 50-64 % at damping 100.
4. CASIE21 as a constraint on the shallow offshore Washington/Oregon starting model or prior (D-B035, not started).
5. Merge order: PR #32, then PR #33, into `tomo/track-b`.
6. Science framing (notes/science_framing.md, 2026-11-23): novelty wording (plate-wide local-earthquake Pn
   tomography already exists, VanderBeek & Toomey 2019), questions that join sources and structure, steps after
   the tomography. Adopted 2026-11-23 as D-B036..D-B042; references merged to main (PR #34, cbaf5efc).
7. Merge design and tile-vs-tile tests (notes/merge_design.md, 2026-11-23, revised after a literature review):
   P-M1..P-M10 proposed. Main choices: coarse whole-margin inversion as common start (P-M2, as Lin et al. 2010),
   resolution-window weights without inverse variance (P-M4), joint-vs-stitched test on blanco + oregon (T6).
   P-M1 and P-M2 adopted (D-B045, D-B044); T6 and M0 done (update 16:15). Open: P-M3..P-M11 (P-M11 = M0
   smoothing 480 and a smoothing scan per tile).

## Done

| Step | What | Where | Decisions | Commit |
|---|---|---|---|---|
| A | v4-NLL3D relocation of all 172,340 events, tiers A-D/U | workflow/08_relocation (branch reloc/track-a) | D-B004, D-B005 | reloc/track-a |
| B1 | Event and pick set: 24,621 events (23,319 main + 1,302 ridge), 703,559 picks, pick QC, OBS S statics | b1_select_events.py, notes/dataset.md | D-B003, D-B008..013, D-B020, D-B023 | 07a2525e |
| B1 OBS check | OBS S picks are true S; negative S terms are path misfit | b1_obs_waveform_check.py, notes/obs_check.md | D-B020 | 003339bc |
| B1b | Balanced dataset of well-recorded stations: 24,249 events, 689,602 picks, 475 stations; station, path and region weights; ridge regions with softer rules | b1b_balance.py, notes/balanced.md | D-B022, D-B023 | 07a2525e |
| Code | tomoDD-SE (USTC 2020 package) built, example reproduced, efficiency audit, Mendocino benchmark (9.9 min, 2.5 GB) | /wd1/mdenolle_data/tomo_v1/src (not in git: licence), notes/tomodd_build.md | D-B021, D-B026 | 6cd89dd4 |
| B2 smoothing check | Spike test of smoothing vs uneven node spacing, Mendocino-Gorda grid, production settings: broadening <= 1.1 at 12-34 km where resolved; upper crust (< 12 km) mostly unresolved at node scale (coverage-limited); syn binary rebuilt (first copy had MAXEVE 1650) | b2_smooth_spike_test.py, b2_smooth_spike_analyse.py, notes/smoothing_test.md | D-B028 (proposed) | uncommitted |
| Atlas | v4-NLL3D (preliminary) catalog switch and tier colours, public | branch atlas/v4-nll3d merged into paper/v4-methods (57447788); gh-pages 2818a585 | D-B024 | 57447788 |

## Waiting

| Item | Waiting on | Blocks |
|---|---|---|
| tomoDD version and permission to modify (OpenMP, sparse matrix build, units bug in the event-separation weighting) | PI email to Haijiang Zhang (zhang11@ustc.edu.cn) | Source changes only; runs with the unmodified code can go ahead |

## Next, in order

0. **Forward fixes per region (D-B028, D-B031).** For each region: b2_fwd_p1.py (P-1 reference times),
   b2_fwd_statics.py (P-6 OBS S statics), then the builder with the `forward` block (P-2, P-4, nearest-column MOD).
   Done: Mendocino-Gorda (mendocino_gorda_v2).
1. **Input builder (B2a).** From B1b to tomoDD files for one region: absolute times with weight
   w_bal x pick-quality weight rescaled into (0, 3]; catalog differential times; numeric station codes;
   OBS S statics applied to the picks (D-B020); events and stations inside the region. Script
   `b2_tomodd_inputs.py`; first region Mendocino-Gorda (D-B025).
2. **Regions (D-B025).** Boundaries and overlaps of ~4 sub-regions (Mendocino-Gorda, Blanco, Oregon
   offshore forearc, Washington-Puget; ridges with Blanco or alone). Size set by memory and run time, not
   curvature (D-B026).
3. **Grid and ray-tracer accuracy test.** Velocity grid with top edge at or above -4 km (bounds check check01 on Mendocino-Gorda 55x49x23: clean, 4.9 min, 2.6 GB, 2026-11-22); travel-time
   accuracy of the pseudo-bending tracer (10 km minimum segment, hard-coded) against a fine reference
   (D-B015 analogue for tomoDD); one bounds-checked run per new grid.
4. **Bayesian ensemble design (D-B030, note notes/bayes_design.md).** Randomize-then-optimize over the
   starting model (published models + smooth random perturbations), station statics (priors per station
   class from the v4-NLL3D terms), pick noise per class and damping/smoothing; ~100 members per region
   as parallel serial jobs; compiler, flags and machine pinned (D-B026: last-bit differences move
   hypocentres ~0.2 km).
5. **Synthetic calibration.** Known model with station delays through the full ensemble; credible
   intervals must cover the truth at the stated rate; checkerboard and spike tests as figures (D-B018).
   Compare balanced vs unweighted B1b here (notes/balanced.md).
6. **Real-data ensemble, region by region,** then overlap comparison and merge (notes/merge_design.md: one
   reference model, common grid, taper x resolution x 1/variance weights, member-by-member merge, tests T1-T5).
7. **B4.** Relocate all 172,340 events in the final model (NonLinLoc, new station terms, tiers),
   with location uncertainty from the ensemble.
8. **B5.** GrowClust on the B4 locations (D-B002).
9. **Magnitudes (PI request 2026-11-22).** Recompute the Route A local magnitudes (ML; workflow/05_magnitude,
   scale/run_magnitude_v4.sh, data/magnitude_v4) with the B4/B5 hypocentres: hypocentral distances change, so
   ML changes. Compare with the v4 magnitudes (offset by region, ComCat anchors) and update the catalog columns,
   Mc maps and atlas. If v4-NLL3D is rerun after D-B028 P-7, decide whether an interim recomputation is needed.
10. **Products.** Model NetCDF with mean, std and resolution mask; catalog; report; private Zenodo draft
   23084158 only (never publish without the PI).

## Open questions for the PI

* Resolved: depth nodes kept, smoothing spike test adopted as D-B029 (PI, 2026-11-22; amends D-B028 P-3).

* Ridges weight held at the clip of 3 (7.5% of the misfit weight instead of 12.5%): keep? (notes/balanced.md)
* Far-station picks (D-B012) re-admitted after a forward check in the starting model: still wanted with tomoDD?

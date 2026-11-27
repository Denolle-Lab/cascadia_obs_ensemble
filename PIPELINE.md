# End-to-end pipeline (D-B110)

One driver reruns the Cascadia OBS analysis in dependency order, from the picks to the plate-boundary science:

    workflow/pipeline/pipeline.py              the driver (stage order, resume, dry run, provenance)
    workflow/pipeline/configs/v6.1.yaml        one configuration per catalog version (v6.1 is the worked example)
    workflow/pipeline/configs/catalogs_v6.1.yaml   catalog registry of the validation stage for v6.1

The driver holds no science code. Every stage calls the scripts that produced the registered products, with the
arguments written in the version's configuration. Choices are set by the decisions in
`workflow/10_tomography/DECISIONS.md` (IDs listed per stage below and in the configuration).

## Quick start

    PY=/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/bin/python
    export LD_LIBRARY_PATH=/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/lib
    C=workflow/pipeline/configs/v6.1.yaml

    $PY workflow/pipeline/pipeline.py $C --list                    # stages, kinds, done or pending
    $PY workflow/pipeline/pipeline.py $C --dry-run                 # every command; every input and code path checked
    $PY workflow/pipeline/pipeline.py $C                           # run what is not done, in order (stops at a gate)
    $PY workflow/pipeline/pipeline.py $C --from mc_b --to plate_boundaries --out-root /wd1/mdenolle_data/pipeline_smoke/x
    $PY workflow/pipeline/pipeline.py $C --only clustering --force clustering

Options: `--from/--to/--only` select stages; `--force S` reruns a done stage; `--run-gated S` runs a gated stage
after the PI's go; `--allow-unmerged` runs a stage whose required branch is not merged; `--out-root DIR` moves the
analysis outputs (stages `mc_b` onward) to DIR, including the products that production keeps elsewhere
(`scratch_vars` of the configuration, e.g. the Mc maps); `--var NAME=VALUE` overrides a configuration variable. Long stages:
run the driver detached (`setsid nohup ... > LOG 2>&1 < /dev/null &`); it runs at nice 10 and holds a lock in its
state folder, so a second driver on the same version refuses to start.

Production folders of the v6.1 analysis products (D-B111): Mc maps `/wd1/mdenolle_data/magnitude_converged/mc_maps_v61`;
clustering and swarms `/wd1/mdenolle_data/catalog_v6/v6.1_analysis/clusters` (`events_zbz_v61.csv.gz`,
`families_zbz_v61.csv`, `dbscan_*_v61.csv`); fault surfaces `.../v6.1_analysis/faults`; validation
`.../v6.1_analysis/validation`; plate-boundary tables `.../v6.1_analysis/plate_boundaries`; driver state, logs and
provenance `.../v6.1_analysis/_pipeline`.

## How the driver works

* **Order.** The stage list below is fixed in the driver; a configuration must give every stage, in this order (a
  stage that does not apply says `skip: <reason>`). Clustering, the swarm statistics and b need the magnitudes and
  Mc, so they follow the magnitudes: relocation, near-source tomography (+ ensemble), relocation in the new model
  (+ magnitudes on its own hypocentres), Mc and b per region, clustering, swarms, fault fitting, structure versus
  seismicity, validation, plate-boundary science.
* **Kinds.** `run`: the driver runs the commands when the stage is not done. `check`: never runs anything; its inputs
  (pinned md5) must exist. `gated`: runs only with `--run-gated <stage>`; otherwise the driver stops there when the
  stage is not done and prints the commands. `manual`: never run by the driver (atlas deploy, Zenodo upload).
* **Done and resume.** A stage is done when all its outputs exist and its `done` checks pass (lines of a chain's own
  state file, e.g. `b4_v6_relocate.state` or the ensemble's `chain.state`; a key of a JSON state file; a pinned
  md5). Done stages are skipped, so the same command resumes. Inside a stage the scripts resume through their own
  state and caches (the relocation chain skips its done stages and existing grids; the clustering skips cached steps;
  the fault detection reuses `work/detect_*.npz`). `--force` reruns a stage, but those internal caches still apply:
  delete them (or use a new `--out-root`) for a clean rerun.
* **Dry run.** Prints every command fully expanded, checks every input (and its md5 when pinned) and every code path
  (in HEAD, uncommitted, or only on a pending branch), tracks inputs that an earlier pending stage will make, and ends
  with a summary; exit 1 when anything does not resolve.
* **Provenance.** After each stage, `<state_dir>/provenance/<stage>.json` (also appended to `provenance.jsonl`): driver
  commit and dirty flag; git blob of every code file and whether it equals HEAD; md5, size and mtime of every input
  and output file (md5 up to `md5_max_mb`); commands, environment, exit codes, log, start and end time. A stage found
  done the first time is recorded as "adopted". Logs: `<state_dir>/logs/<stage>.log`. Scratch runs (`--out-root`)
  keep the Track B manifest records (`manifest.write_manifest`) in `<state_dir>/manifests.jsonl` instead of the shared
  `/wd1/mdenolle_data/tomo_v1/manifests.jsonl` (variable `CASCADIA_MANIFEST_LOG`).
* **Checkout data.** A new worktree has none of the gitignored data some scripts read inside the repository
  (`data/slab2`, `data/tomography`, the v4.1 catalog and picks, the ANSS file). `repo_links` in the configuration
  lists them; the driver links them from the main checkout before a run.

## Stages

Times are wall-clock times of the v6.0 / v6.1 runs on this machine (the analysis stages at 8 processes).

| # | stage | kind | what | runtime | decisions |
|---|---|---|---|---|---|
| 0 | picks | check | v4.1 picks, re-timed v4.1t, clock-corrected v4.1tc | hours (rebuild) | D-B072, D-B075, D-B090, D-B097, D-B108 |
| 1 | reloc_start | run | NonLinLoc in the start model (Track A) | ~8 h | D-B004, D-B005, D-B024, D-B055, D-B066, D-B067 |
| 2 | tomo_dataset | run | B1 selection, B1b balance, re-timed copy | ~15 min | D-B003, D-B006, D-B008, D-B020, D-B022, D-B023, D-B035, D-B072, D-B075 |
| 3 | tomo_inversion | run | tomoDD-SE: M0, tiles, merge, product nodes | ~5 h | D-B026-D-B028, D-B031, D-B034, D-B035, D-B044-D-B051, D-B068, D-B073, D-B076, D-B081-D-B083, D-B099 |
| 4 | tomo_register | gated | model into the append-only registry | minutes | D-B047, D-B099, D-B104 |
| 5 | tomo_ensemble | run | RTO ensemble (100 real members) and synthetic calibration | ~8 h | D-B030, D-B032, D-B054, D-B094, D-B099 |
| 6 | ens_register | gated | ensemble statistics into the registry | minutes | D-B047, D-B094, D-B099 |
| 7 | cc_dt | run | waveform cross-correlation differential times (coherence partners) | ~7 h | D-B040, D-B078, D-B084, D-B093 |
| 8 | event_flags | run | Mount St Helens analog test-signal flags | ~1.5 h | D-B108 |
| 9 | relocation | run | grids, NLL-SSST, coherence, deep and edge passes, catalog | ~6.5 h | D-B052, D-B055, D-B067, D-B068, D-B073, D-B079, D-B084, D-B092, D-B095, D-B098, D-B103, D-B104, D-B108 |
| 10 | magnitudes | run | Route A on the catalog's own hypocentres, converged solver | ~25 min | D-B091, D-B102, D-B105 |
| 11 | reloc_acceptance | run | literature depths, held-out scores, above-surface check | ~1 h | D-B055, D-B096, D-B103, D-B085 |
| 12 | reloc_publish | gated | relocation reports committed to the relocation branch | 1 min | D-B085 |
| 13 | catalog_bundle | gated | version folder, README, registry row, zip | ~5 min | D-B058, D-B080, D-B100, D-B085, D-B101 |
| 14 | mc_b | run | Mc maps and b per region | ~10 min | D-B091, D-B104, D-B105 |
| 15 | clustering | run | nearest-neighbour clustering | ~15 min | D-B091, D-B104, D-B106, D-B109 |
| 16 | swarms | run | swarm scan and statistics | ~3 min | D-B106, D-B109 |
| 17 | faults | run | fault surfaces v1 refit, joined to the clusters | ~15 min | D-B109 |
| 18 | structure | run | structure versus seismicity near the faults | ~10 min | D-B018, D-B049, D-B051, D-B099, D-B107, D-B109 |
| 19 | validation | run | ComCat, earlier catalogs, published local catalogs | ~10 min | D-B055, D-B071, D-B103, D-B085 |
| 20 | plate_boundaries | run | seismicity style per plate-boundary segment | ~10 min | D-B107, D-B109 |
| 21 | atlas | gated | atlas build and headless check | ~10 min | D-B080, D-B104, D-B105 |
| 22 | atlas_deploy | manual | gh-pages | - | D-B104 |
| 23 | zenodo_upload | manual | private draft 23084158 | - | D-B085, D-B101 |

**0 picks (check).** The pick files the relocation reads, pinned by md5: the v4.1 picks (catalog v4.1 of Ian's
final catalog plus the 2016 repick), the re-timed v4.1t (drifting analog channels timed from their record headers,
D-B072, D-B075) and the clock-corrected v4.1tc (MH07.XD and MG07.XD, D-B090). Rebuilt with the pick-timing toolkit
(`tools/pick_timing`, D-B090; `run_v6.sh` in `/wd1/mdenolle_data/tomo_v1/checks/pick_timing_v6`). A person reviews
the clock candidates between `detect-clock` and `apply`, so the driver never builds this stage; the 129 other
candidate epochs wait for v6.2 (D-B108).

**1 reloc_start (run).** Track A: catalog v4.1 relocated with NonLinLoc (EDT, oct-tree) in the start model (CRESCENT
Gen0 Vs, Brocher Vp, Vp/Vs >= 1.76, published 1D models at the ridges; `a1_build_model.py`), two passes with static
station terms (`a2_run_nlloc.py`, `a2_station_terms.py`), catalog `catalog_v5_trackA_iter1_vpvs176.csv` (= v4-NLL3D,
D-B024), the starting hypocentres of the tomography (D-B004, D-B005). The run folders were cleared; the catalog and
the phase caches remain, so the stage is done. A rerun uses today's (v5.1, round-2) runner.

**2 tomo_dataset (run).** B1: tomography event set (tiers A-C, phases, gap, RMS, declustering; D-B003, D-B035) and
pick QC; B1b: balanced data set of well-recorded stations with path-redundancy and region weights (D-B022, D-B023);
then the re-timed copy (`b1t_retime_dataset.py`, D-B072/D-B075) that every v0.4.x model uses.

**3 tomo_inversion (run).** tomoDD-SE joint Vp/Vs inversion: the whole-margin M0 and the tiles restarted from it
(damping 200, smoothing by tile, D-B044-D-B046), with tomoDD's surface floor in the inversion (D-B081, D-B083) and,
for v0.4.2, the tiles re-inverted without self-paired differential times with the patched freeze (D-B099);
resolution-gated merge (`b2_merge.py`, D-B049-D-B051) and the product nodes (medium, D-B073 shallow-crust rule;
`b2_model_v042.py`). M0 and oregon come from the v0.4.1 chain (`b2_v041_chain.py`), reproduced byte for byte.

**4 tomo_register (gated).** The PI's session registers the staged model (append-only `registry.csv`, sha256 of
every file, D-B047).

**5 tomo_ensemble (run).** Randomize-then-optimize ensemble of the whole configuration (100 real members, D-B030,
D-B032), statistics, then this root's synthetic calibration (`calibration_order: after_real`, D-B099); the products
need a PASS or an accepted calibration entry (v0.4.2: Vp x1.50, Vs x1.35, D-B099). **6 ens_register (gated)** writes
`v0.4.2+ens.100` with `B2_ENS_REGISTER=1`.

**7 cc_dt (run).** Waveform cross-correlation differential times of every v5.1 family and its neighbours (D-B093),
which give the NLL-coherence stage its partners (pair coherence). Computed once and reused by every v6.x relocation.

**8 event_flags (run).** Likely false events from the weekly artificial test signal on the UW analog channels near
Mount St Helens, with waveform evidence; a flag only (event_flag `analog_test_signal`, D-B108). Read by the catalog
stage of the relocation.

**9 relocation (run).** `b4_v6_relocate.sh` with the version's environment (VERSION, CATVER, V6_ROOT, V6_SCHEDULE,
V6_EVENT_FLAGS; RESUME.md of the run root): travel-time grids of the registered model (identity check of the 1 km
fields), wide and deep grids, NLLoc without terms, NLL-SSST passes (9999 -> 16 km for v6.1, D-B103), NLL-coherence
(not at Mendocino, D-B098), static terms for the deep and edge passes, catalog (`b4_v6_catalog.py`, flags, surface
check). The chain's own state file is the done check; `STOP_AFTER` splits it into the driver's stages 9-12.

**10 magnitudes (run).** Stage `magnitudes` of the same chain (D-B105): `scale/run_magnitude_v51.sh` on the catalog's
own hypocentres (arrivals, the stored Route A amplitudes, ML anchoring to ComCat, Mw calibration, ComCat tensors),
`phase19_preferred_magnitude.py --solver exact` (converged station terms, D-B091), `v51_magnitude_table.py`; the
catalog is rebuilt with `magnitude_source` and must equal the hypocentre catalog outside the magnitude columns.

**11 reloc_acceptance (run).** Stage `acceptance`: literature depth pairs (`b4_lit_paired.py`: Hirao et al. 2025,
Lange et al. 2026), held-out coherence scores, the acceptance report; any event above the surface stops the chain
(D-B055). **12 reloc_publish (gated)** commits the reports to the relocation branch and writes the DONE file.

**13 catalog_bundle (gated).** `catalog_version.py <v> --test DIR` (compare its CSV with the previous version for the
columns that should not change), then `catalog_version.py <v>` (version folder, README with method notes and known
limits, registry row, zip of the series) and `md5sum -c MD5SUMS`. Built-in gates: integrity checks, unknown
`magnitude_source` labels stop the build, `above_surface > 0` means not publishable. The bundle CSV (md5 pinned in
the configuration) is the input of every analysis stage.

**14 mc_b (run).** `v4_mc_maps.py --catalog <bundle> --exclude-flagged`: Mc by MAXC+0.2, MBS, GFT90/95 and the
Woessner-Wiemer best rule on 0.1 deg nodes (250 nearest events within 50 km, 200 bootstrap resamples), b (Aki-Utsu,
Shi-Bolt) above Mc, per node and per region box, on the catalog's own hypocentres and magnitudes (D-B104), events
flagged as not earthquakes left out (D-B105).

**15 clustering (run).** `v51_clusters_zbz.py --catalog <v>`: Zaliapin-Ben-Zion nearest-neighbour distances with the
regional Mc, background / mainshock-aftershock / swarm / multiplet families, the regional eta0 thresholds,
shuffled-catalog test, sensitivity variants (d3, Mc+0.3, Mc_best from stage mc_b, fractal dimension, Mc per year,
tier A-C only ...), null tests, the waveform-review checks and the comparison with the previous version and v4.1.
Non-earthquakes are left out (D-B106). The Mc figures are drawn here, with the clustering Mc in the regional FMD.

**16 swarms (run).** `v51_swarms.py`: space-time DBSCAN scan (3 km, 2 km/day, >= 15 events, <= 30 days), rate
against the 90-day background, depth range from well-constrained depths, migration and sqrt(t) diffusion fits,
overlap with the families, match with the previous version's clusters.

**17 faults (run).** Fault surfaces v1 refit on the catalog (D-B109): plane detection (sequential RANSAC, k-planes
reassignment with the dip rule; five runs for stability), the detection reruns for recovery (`v1_recovery.py`: 80 %
subsample, tiers A-B, perturbed locations), Blanco strands, rectangles with location-error Monte Carlo, screening,
assignment of every event, comparison with v0 and its literature matches; then the clustering's `faults` step joins
the families to these surfaces.

**18 structure (run).** `pb_structure.py`, `pb_struct_seis.py`: per plate-boundary segment, resolved model nodes near
the faults, depth-matched contrasts (near/far, across, along strike) against the calibrated ensemble 1 sigma, the
seismicity bottom against the 7.5 and 8 km/s levels, swarm against mainshock-aftershock families on resolved nodes.

**19 validation (run).** `compare_catalogs.py` with the version's registry: ComCat (ANSS 2010-2016) offsets and depth
differences per region (all and near-station events), above-surface share, shifts from v4.1 and the previous version,
plane-fit thickness of the v0 fault surfaces in each catalog; `b4_lit_paired.py`: depths paired with Hirao et al.
(2025) at Mount St Helens and Lange et al. (2026) at Blanco. Magnitudes against ComCat are checked in stage
magnitudes; fault surfaces against published faults in stage faults (literature matches, new surfaces listed).

**20 plate_boundaries (run).** `pb_seismicity.py`, `pb_sensitivity.py`, `pb_figures.py`: 19 segments by boundary type
from ridges to transforms to the subduction zone: counts, Mmax, Mc and b with bootstrap intervals, clustering shares,
migrating swarms, depth extent, fault-surface geometry; the sensitivity of the segment rule; the draft figures.

**21 atlas (gated), 22 atlas_deploy (manual), 23 zenodo_upload (manual).** Atlas: an entry in
`workflow/11_compare/catalogs.yaml` (by hand), `compare_catalogs.py --only <id>` (shared common table),
`build_v4_atlas.py --model --mc-cat --clusters-catalog`, `atlas_check.js` (0 page errors; counts at M >= 2.0 equal the
CSV counts); deploy to gh-pages by hand. Zenodo: `utils/zenodo_picks_draft.py upload 23084158 <zip>` only with the
PI's go (the draft stays private), then a row in `data/catalog_v<major>/zenodo_uploads.csv`.

## Inventory (2026-11-27)

Where each stage's code lives, what fixes its version, and whether a new catalog or model version is a
configuration change. "main" = origin/main 9a18536f (PRs #60-#63 merged); "this branch" = pipeline/end-to-end.

| stage | scripts | git | version inputs | output root | new version |
|---|---|---|---|---|---|
| picks | tools/pick_timing (D-B090), b1t_retime_dataset.py | main (#54) | pick file + md5 | tomo_v1/checks/pick_timing_v6 | config (+ PI review) |
| reloc_start | 08_relocation/a1_*, a2_* | main (#38) | start model, picks | reloc_v5 | config; grid loop not in git |
| tomo_dataset | b1_select_events, b1b_balance, b1t_retime_dataset | main | params paths.b1_dir, retime | tomo_v1/b1*_c(t) | params |
| tomo_inversion | b2_v041_chain, b2_tomodd_inputs, b2_v042_tile.sh, b2_merge, b2_model_v042 | main (#50, #51) | params v041/v042, tile regions | tomo_v1/b2_tomodd, v042 | new params block + tile list (per model) |
| tomo_register | b2_model_export (registry) | main | version | models/cascadia_obs_tomo | manual (PI session) |
| tomo_ensemble | b2_ens_run chain | main (#50, #51) | tomodd.ensemble.configs.<v> | tomo_v1/ensemble_v<model> | params (B2_ENS_CONFIG) |
| cc_dt | b5_ccall_* | main (#53) | v5.1 families, picks v4.1tc | tomo_v1/cc_dt_all | CCALL_OUT (reused) |
| event_flags | msh_flags/maf_*.py, run.sh | main (#61) | catalog v6.0, picks | reloc_v61/flags | gap: paths in maf_common.py |
| relocation, magnitudes, acceptance, publish | b4_v6_relocate.sh (+ b4_v6_*.py, scale/run_magnitude_v51.sh, phase19 --solver exact, v51_magnitude_table.py) | main (#52, #60) | VERSION, CATVER, V6_ROOT, V6_SCHEDULE, V6_EVENT_FLAGS, PICKS | reloc_v61/run, magnitude_converged/v6.1 | environment (config) |
| catalog_bundle | catalog_version.py | main (#62); MAG_TABLES in params: #64 | params catalog_versions.versions.<v> | catalog_v6/v6.1 | params (+ MAG_TABLES line until #64) |
| mc_b | 06_analysis/v4_mc_maps.py --catalog | main (#59) | bundle CSV + picks | out_root/mc_maps | config |
| clustering | v51_clusters_zbz.py | main (#56) + this branch (v6.1, --mcb, --faults) | CATALOGS entry | out_root/clusters | CATALOGS entry (3 lines) |
| swarms | v51_swarms.py | main (#42, #56) + this branch | CATALOGS entry | out_root/clusters | as clustering |
| faults | 07_faults/v1_*.py, v1_recovery.py | main (#41) + this branch (FAULTS_V1_*) | catalog CSV | out_root/faults | config (environment) |
| structure | 11_paper_plan/pb_structure, pb_struct_seis | analysis/plate-boundaries (no PR), merged here + PB_* | PB_* environment | out_root/plate_boundaries | config |
| validation | 11_compare/compare_catalogs.py, b4_lit_paired.py | main (#31, #63) + this branch (--out, --big) | registry YAML | out_root/validation | config + registry file |
| plate_boundaries | pb_seismicity, pb_sensitivity, pb_figures (pb_page: v6.0 text) | as structure | PB_* environment | out_root/plate_boundaries | config |
| atlas | build_v4_atlas.py, atlas_template.html, atlas_check.js | main (#63) | catalogs.yaml entry, --mc-cat, --clusters-catalog | out_root/atlas | catalogs.yaml entry (by hand) |
| paper model figures | 08_figures/model_figures.py | paper/model-figures (no PR) | model version | - | not in the chain (D-B107) |

## Rerunning for a new version (e.g. v6.2)

1. Copy `configs/v6.1.yaml` to `configs/v6.2.yaml`; change `catver`, `tag`, `prev_catalog`, the run roots
   (`reloc_root`, `event_flags`, `mag_dir`), `catalog_md5` (after the bundle), and for a new model `model`,
   `tomo_out`, `ens_root`.
2. Code-side entries a new version needs (small, explicit):
   * picks: a new pick file (confirmed clock epochs) -> `picks_v41tc` and its md5; relocation env PICKS / PICKS_MD5
     (the obs stage is redone);
   * model: a params block (`v043:`), an ensemble configuration (`tomodd.ensemble.configs`), the tile sequence in
     the tomo_inversion commands; registration by the PI's session;
   * relocation: the b4 chain's environment (schedule, flags file);
   * bundle: `catalog_versions.versions.<v>` in params.yaml; a new magnitude table is one entry of
     `catalog_versions.magnitude_table_files` (path, md5, key, label) listed in the version's `magnitude_tables`
     (params only once PR #64 is merged);
   * clustering: one entry in `CATALOGS` of `v51_clusters_zbz.py` (tag, csv, big, mcb, refs), which also serves
     `v51_swarms.py`;
   * validation: a registry file like `configs/catalogs_v6.1.yaml`;
   * Mount St Helens flags: `maf_common.py` paths (gap below).
3. `pipeline.py configs/v6.2.yaml --dry-run`; fix what does not resolve; run (detached); pass the gates with the PI.

## External inputs (not in git)

* Waveforms: pnwstore (continuous data and headers: pick timing, CC snippets, Mount St Helens scan), NCEDC (CC).
* Route A amplitudes: `/wd1/mdenolle_data/amps_v6_all` (per station-day tables; not re-measured by the pipeline).
* Data roots: `/wd1/mdenolle_data/` (`tomo_v1/` tomography runs, models, ensembles, CC, checks; `reloc_v5/` Track A
  model, grids, topography floor; `reloc_v61/` v6.1 run; `magnitude_converged/`; `catalog_v5/`, `catalog_v6/`
  bundles; `picks_share/stations/`; `tomo_v1/depth_literature/pairs/`).
* Binaries: tomoDD-SE freezes `/wd1/mdenolle_data/tomo_v1/src/frozen_2026-11-25_inv/bin` and
  `frozen_2026-11-26_inv/bin` (SHA256SUMS there; D-B054, D-B099); NonLinLoc v7 in `/wd1/mdenolle_data/reloc_v5/src`.
* Environments: pixi `default` of the main checkout (`/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default`),
  pixi `models` in `/home/mdenolle/wt_tomo_res/.pixi/envs/models` (a worktree; see gaps), the Playwright env of the
  atlas check (`~/.cache/rattler/cache/cached-envs-v0/55304966f11fb1c1`).
* Gitignored data of the main checkout: catalog v4.1 and its picks (`data/catalog_v4/*.csv`), Slab2
  (`data/slab2`), CRESCENT Gen0 (`data/tomography`), ANSS 2010-2016 (`data/magnitude_v4/anss_2010-16.csv`).

## Gaps

Fixed on this branch (each default reproduces the committed results; checks in the results section):
* fault surfaces v1 ran on v5.1 only: `FAULTS_V1_CATALOG / _OUT / _BIG / _LABEL` in `v1_common.py`;
  `analog_test_signal` added to its non-earthquake flags (no v5.1 event carries it);
* the verifier's detection reruns (recovery columns) lived as scripts outside git: `workflow/07_faults/v1_recovery.py`;
* clustering and swarms knew v5.1 and v6.0 only: `--catalog v6.1`, `--mcb`, `--faults` (`v51_clusters_zbz.py`), the
  v6.x comparison runs in `v51_swarms.py`;
* plate-boundary scripts read v6.0 paths: `PB_*` environment variables in `pb_common.py`, file tags from `PB_TAG`;
* `compare_catalogs.py --out --big` and `b4_lit_paired.py --out` (scratch runs); `CASCADIA_MANIFEST_LOG` in
  `manifest.py`;
* the CC job ran from the removed worktree `~/wt_ccall` and read its families from the removed `~/wt_a51_clusters`:
  `b5_ccall_run.sh` / `b5_ccall_chain.sh` run from their own folder, `b5_ccall_common.FAMILIES` is the committed file.

Left (stated, not fixed here):
* Track A start-model travel-time grids (`time_vpvs176`): the GridCut / Grid2Time loop is not in git.
* v0.4.2 had no chain script: its sequence (build_inputs.sh outside git, `b2_v042_tile.sh` per tile, merge, product)
  is written out in the configuration; model registration is a manual step of the PI's session.
* `msh_flags/maf_common.py` hard-codes the v6.1 output folder and the v6.0 catalog (a v6.2 run needs parameters).
* The pixi env `models` lives in the worktree `wt_tomo_res` (`b2_ens_common.PY_MODELS`, `b2_ens_member.TOMO_RES`,
  `b4_v6_relocate.sh MENV`, `b2_v040_chain.MENV`); it should move to the main checkout's pixi environments.
* Report texts written for one version are not run for v6.1: `v1_report.py` (v5.1), `v60_clusters_report.py` (v6.0),
  `pb_page.py` (v6.0 feasibility study); their tables and figures are made by the stages above.
* The legacy v5.1 structure and interpretation analyses (`v51_struct*.py`, `v51_interp*.py`) read model v0.3.1 and
  removed worktrees (`wt_a51_clusters`, `wt_a51_faults`); they are superseded by `pb_*.py` and not in the chain.
* A new catalog version still needs a `CATALOGS` entry in `v51_clusters_zbz.py` (code, three lines).

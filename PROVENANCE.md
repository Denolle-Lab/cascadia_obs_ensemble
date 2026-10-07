# Provenance: which code, tool and version made each product

The catalog was built over two years (2024-11 to 2026-09), by several people, with external
tools and with versions of this repository that changed along the way. This file records,
for each stage: its product, when it was made, which code and version made it, and how to get
that code back. Git tags mark the repository state for each stage (`git tag -n1`). Items
marked **TO CONFIRM** must be filled in by the person who ran that stage before the Zenodo
release.

Product dates are file modification times on the lab server
(`/wd1/hbito_data/data/datasets_all_regions/`, preserved by `rsync -a`). Row counts and the
full creation chain are in [`data/LINEAGE.md`](data/LINEAGE.md).

## Summary

| # | Stage | Product | Made | Code | Tool / version |
|---|---|---|---|---|---|
| 1 | Phase picking | `all_picks_all_regions_2010_2015_ver3.csv` (39,597,551 picks) | 2025-03-05 | `1_picking/`, tag `stage-picking-ver3` (`eb85d194`) | ELEP `2f3f22a9d9aaccd7b84e4473b752737148fb1056` (github.com/congcy/ELEP); SeisBench models (version **TO CONFIRM**) |
| 2 | Association | `all_events_2010_2015_ver3.csv` (116,591), `all_pick_assignments_all_regions_2010_2015_ver3.csv` (1,086,007) | 2025-03-20 / 03-25 | external | GENIE, run by I. McBrearty (repository, commit, trained model, config: **TO CONFIRM**) |
| 3 | Relocation | `Cascadia_relocated_catalog_ver_3.csv` (63,887), `..._picks_ver_3.csv` (1,004,335) | 2025-03-30 / 03-28 | external | GraphDD, run by I. McBrearty and Y. Yu (repository, commit, config: **TO CONFIRM**); velocity and region inputs `data/vel_*.csv`, `data/nodes_*.csv` |
| 4 | Tables | `origin_`, `arrival_`, `assoc_2010_2015_reloc_cog_ver3.csv` | 2025-05-22 | `4_relocation/merge_events_*` (tag `stage-reloc-cc-qc-ver3`) | — |
| 5 | CC refinement | `origin_2010_2015_reloc_cog_ver3_cc.csv` (63,887) | 2025-10-14 | `4_relocation/cross_correlation/` (tag `stage-reloc-cc-qc-ver3`, `cd14edf5`) | waveform sets from this repo, GraphDD re-run with differential times (**TO CONFIRM** who ran it and with what) |
| 6 | QC subset | `origin_..._cc_p_4_s_4_rms_2_5.csv` (31,020) | 2025-10-14 | `4_relocation/quality_control/4_quality_control_*` | see the QC note below |
| 7 | Counts amplitudes (baseline) | `Cascadia_updated_catalog_picks_assignment_ver_3_w_amp.csv` | 2026-06-29 | `4_relocation/calculate_amplitudes.py`, tag `stage-amplitudes-counts` (`7a0cdd97`) | — |
| 8 | Route A amplitudes | `rerun_v2/raw_wa_amplitudes_v2.csv` (1,004,335 picks; 1,002,707 measured) | 2026-09-23 → 09-25 | `route_a_wa_amplitudes.py`, tag `stage-routeA-v2-amplitudes` (`8eba14b7`) | `amplitude` env (below) |
| 9 | Inversion + ComCat anchors | `amp_distance_dataset_routeA.csv`, `cascadia_catalog_ML_routeA.csv`, station terms | 2026-09-25 | stage 3 of `run_route_a_rerun.sh` at `d6f2c642` | `amplitude` env |
| 10 | Paper magnitudes | `data/focal/comcat_mt_matched.csv`, `cascadia_catalog_M_routeA.csv`, `cascadia_catalog_classified.csv` | 2026-09-29 | phase18 → phase19 → phase10, tag `magnitudes-v1` (`f46941f0`) | `default` env |
| 11 | Figures, paper | `paper/figures/*`, `paper/main.pdf` | 2026-09-30 | `paper/export_figures.py`, `make paper` at the release tag | `default` and `paper` envs |

## Stage notes

**1. Picking.**
- The ver3 picks were written on 2025-03-05. The last commit of `1_picking/` before that date
  is `eb85d194` (2025-01-22), tagged `stage-picking-ver3`. The scripts may have run from a
  working copy with edits made after that commit. Later commits to `1_picking/` (through
  2026-06-29) changed paths and tutorials. They also flip the channel rule back and forth;
  the one that ran is "HH, else BH, else EH" (the tag's rule would have skipped OBS
  station-days without EH, whose BH picks are in ver3).
- ver3 joins two picker runs: **v1** (2011-2015, `parallel_pick_{year}_HH_BH.py` with
  `picking_utils_2012.py`; HH/BH at the native sampling rate; 21.9 M picks, the rows with an
  empty `band_inst`) and **v2** (2010-2015, four region runs per year; 17.7 M picks).
- Rerunning: `1_picking/run_picking.py` with `picking_config.csv` (29 rows: 5 v1 runs, 24 v2
  runs) replaces the per-year scripts, now in `1_picking/legacy/`. It reproduces the stored
  per-day picks: v2 on 14 station-days (same picks and times, probabilities within 3e-7), v1
  on 11 station-days (10 byte-identical, 1 with one pick off by one sample). See
  `1_picking/legacy/README.md`.
- The merge of the per-day files into ver3 is not in the repository, and it dropped whole
  v2 station-days of the two 46-50°N edge regions (about 17% and 34% of their 2011 files); the
  rule was not recorded; ver4 adds these station-days back (PI decision, 2026-10-05). The 2013 files of `122-123_46-50` (1.78 M picks, in a
  nested folder) were not merged at all, and the EH runs of 2010, 2014 and 2015 were still
  writing when ver3 was merged (37,158 station-days, 5.1 M picks, files dated 2025-03 to 04).
  `utils/build_picks_v4.py` builds ver4 = ver3 + these picks (`data/PICKS_VER4.md`). All three gaps reached GENIE and GraphDD and are stated
  in the paper (Current limitations); see `1_picking/legacy/README.md`.
- The ELEP commit is pinned in `pixi.lock`.
- The Python environment used in 2024–25 was not locked (pixi arrived in 2026-03,
  `b07de7a0`). The current `default` env has seisbench 0.11.7, obspy 1.5.0 and numpy 1.26.4.
  Exact rerunning of the picker therefore depends on the SeisBench model weights, which should
  be recorded. The ensemble members are EQTransformer `original` (v3), `ethz`, `instance`,
  `scedc`, `stead` (v2); the weights used for the 2026-09 check are in
  `~hbito/.seisbench/models/v3/eqtransformer` (files dated 2025-06-25, after the ver3 run, so
  the versions at the time are **TO CONFIRM**; they reproduce the ver3 picks).
- Waveforms came from pnwstore (UW) with an FDSN fallback (`run_detection(source=...)`).

**2–3, 5. GENIE and GraphDD.** No GENIE or GraphDD code is in this repository. Their
outputs are the inputs of stage 4. To make these stages reproducible, the people who ran
them should record:
- the repository URL and commit of each code;
- the trained GENIE model and its training data;
- the configuration and velocity model used;
- the station file;
- the run date and host.

For the CC re-run, record the differential-time files and the GraphDD settings.

**6. QC: the thresholds that were actually applied.**
- The final QC file holds 31,020 events. Applied to `origin_..._cc.csv`, the rule
  **P picks > 4, S picks > 4, RMS < 2.5 s** (at least 5 of each) reproduces exactly that set.
- The rule written in the paper and in the notebook since `f12b7af8` (2026-07-11), "≥ 4 P and
  ≥ 4 S", gives 40,065 events.
- **Decided 2026-09-30:** keep the file the figures were made from; the paper now states
  "at least five P and five S picks" (≥ 5).

**8. Route A amplitudes.**
- Command:
  `setsid nohup bash run_route_a_rerun.sh`, run from `4_relocation/magnitude`.
  - 20 shards, one pass, started 2026-09-23 12:57:07; stage 1 finished 2026-09-25 05:22:53
    (log: `rerun_v2/STATUS`).
  - Every shard loaded the code of tag `stage-routeA-v2-amplitudes` at start. The review
    commit `d6f2c642` (14:59 the same day) did not reach the running shards.
- Inputs:
  - pick table `Cascadia_updated_catalog_picks_assignment_ver_3.csv` (md5 `6e26c7741aec…`,
    2025-10-14);
  - `station_inventory_v2_slim.xml` (md5 `042b3776b9a0…`), sliced from
    `station_inventory_v2.xml` (md5 `fc4f2c76954d…`, 2026-09-23), which was built by
    `route_a_build_station_inventory.py` from FDSN (IRIS, NCEDC for NC/BK);
  - waveforms from pnwstore 0.2.1, and NCEDC for NC/BK.
- Settings: Wood–Anderson (T0 0.8 s, h 0.7, gain 2080), **1 Hz high-pass after the
  Wood–Anderson filter**, 0.05–2 Hz displacement band, pad ≥ 60 s, one sensor per pick.
- `amplitude` env (pixi.lock at the tag): python 3.10.20, obspy 1.5.0, numpy 1.26.4,
  scipy 1.15.2, pandas 2.3.3, pnwstore 0.2.1.
- Archive of the raw outputs: `/auto/c-wsd01/PNSN_exotic/cascadia_magnitude_routeA_v2`
  (with `MANIFEST.sha256`).

**9. Inversion and anchors.**
- Run as stage 3 of the same driver on 2026-09-25: route_a_build_dataset (SNR ≥ 3, epoch
  station ids), phase3 (`--fix-n 1.0`), phase2 (ComCat ML anchors, 15 s / 50 km), phase4,
  phase16.
- ComCat was queried on 2026-09-25. The cache `data/magnitude/comcat_ml_events.csv` must be
  archived, because ComCat changes over time.

**10. Paper magnitudes.**
- `phase18_moment_tensor_match.py` matches the ComCat moment tensors in
  `data/focal/comcat_mt.csv` (fetched 2026-09-29 by `utils/fetch_focal_mechanisms.py`; the
  archived copy is authoritative).
- `phase19_preferred_magnitude.py` builds the preferred magnitude: Hutton–Boore ML with
  station terms from picks within 150 km, one offset to the ComCat anchors, the fitted-decay
  ML mapped for far-only events, and Mw at 4.5 and above.
- `phase10_event_classification.py` carries M, M_type and ML_method into the classified
  catalog. The method is described in `paper/main.qmd` (Magnitude estimation).

**Revision test (not in the paper).** A 0.5 Hz high-pass re-measurement started on 2026-09-30
with the same driver: `OUT=rerun_hp05`, `DATA=data/magnitude_hp05`, `SUFFIX=_routeA_hp05`,
`HIGHPASS=0.5`, at `magnitudes-v1`.

## Checksums of the magnitude products (md5, first 12 hex)

| File | md5 |
|---|---|
| `4_relocation/magnitude/rerun_v2/raw_wa_amplitudes_v2.csv` | `5e2ab30efda2` |
| `data/magnitude/amp_distance_dataset_routeA.csv` | `57272ae6664a` |
| `data/magnitude/cascadia_catalog_ML_routeA.csv` | `39ff0d95c3c5` |
| `data/magnitude/route_b_station_terms_routeA.csv` | `6746a9beed60` |
| `data/magnitude/cascadia_catalog_M_routeA.csv` (55,707 events) | `d3432bad39b4` |
| `data/magnitude/cascadia_catalog_classified.csv` | `f5a2d2477062` |
| `data/focal/comcat_mt.csv` / `comcat_mt_matched.csv` | `a244bb6faf1f` / `5e31dd368bfb` |

The earlier stages' checksums are in `data/LINEAGE.md`.

## Getting the code of one stage back

Each stage's code can be checked out next to the current tree, without switching branches:

```sh
git fetch --tags
git worktree add ../cascadia-picking   stage-picking-ver3
git worktree add ../cascadia-routeA-v2 stage-routeA-v2-amplitudes
cd ../cascadia-routeA-v2 && pixi install -e amplitude     # env from that tag's pixi.lock
```

Tags earlier than 2026-03-27 have no `pixi.lock`. Use the current `default` env for them
(versions above) and expect small numerical differences.

Files removed in the 2026-09 cleanup (`old/`, `3_post_processing/`, `0_data_availability/`,
the `_old` CSVs) are at tag `pre-cleanup-2026-09`:

```sh
git show pre-cleanup-2026-09:path/to/file > file    # or: git worktree add ../pre pre-cleanup-2026-09
```

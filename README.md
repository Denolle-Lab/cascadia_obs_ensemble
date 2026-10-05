# Ensemble Deep Learning to Mine Cascadia Offshore Seismicity

This project builds a high-resolution earthquake catalog for coastal and offshore
Cascadia (2010–2015) from the Cascadia Initiative ocean-bottom seismometers (OBS)
and regional land networks, using an ensemble deep-learning pipeline:

1. **Detection & phase picking** with an ensemble picker (ELEP; Yuan et al., 2023),
2. **Phase association** with the GENIE graph neural network (PyOcto was also tested),
3. **Relocation** with GraphDD double-difference relocation, refined by
   waveform **cross-correlation** differential times (HypoDD was also tested),
4. **Quality control, magnitude estimation, and comparison** against established
   catalogs — [USGS ComCat](https://earthquake.usgs.gov/earthquakes/search/) and
   [Morton et al., 2023](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023JB026607).

The manuscript describing this work is authored in [`paper/`](paper/) (Quarto → LaTeX,
synced to Overleaf; see [`paper/README.md`](paper/README.md)).

## Authors

**Repository / software** — code and analysis in this repository:

| Author | Affiliation | Contribution |
|--------|-------------|--------------|
| Marine Denolle (mdenolle@uw.edu) | UW Earth & Space Sciences | project lead, magnitudes, QC/post-processing |
| Hiroto Bito (hbito@uw.edu) | UW Earth & Space Sciences | picking, amplitude, QC & post-processing pipeline |
| Qibin Shi (qibins@uw.edu) | UW Earth & Space Sciences | code |
| Yiyu Ni (niyiyu@uw.edu) | UW Earth & Space Sciences | code |
| Nathan T. Stevens (ntstevens@uw.edu) | Pacific Northwest Seismic Network | code |
| Ian W. McBrearty | Stanford Geophysics | graph networks — GENIE association & GraphDD relocation |
| Yifan Yu | Stanford Geophysics | graph networks — GENIE association & GraphDD relocation |
| Zoe Krauss (zkrauss@uw.edu) | UW Oceanography | results cross-checking |

**Manuscript** — the working paper draft in [`paper/`](paper/) has a longer author
list: Marine Denolle, Hiroto Bito, Qibin Shi, Yiyu Ni, Ian W. McBrearty, Zoe Krauss,
Nathan T. Stevens, Yifan Yu, and Gregory C. Beroza (Stanford Geophysics).

## Pipeline and provenance

| Stage | Where | Product | Version used |
|---|---|---|---|
| 1. Picking (ELEP ensemble) | [`workflow/01_picking/`](workflow/01_picking/) | 39.6 M P/S picks | tag `stage-picking-ver3`, ELEP `2f3f22a9` |
| 2. Association (GENIE) | external (I. McBrearty); [`workflow/02_association/`](workflow/02_association/) | 116,591 events | see PROVENANCE |
| 3. Relocation (GraphDD) + CC refinement | external; CC datasets in [`workflow/03_relocation/`](workflow/03_relocation/) | 63,887 events, 1.00 M picks | tag `stage-reloc-cc-qc-ver3` |
| 4. Merge + QC | [`workflow/04_merge_qc/`](workflow/04_merge_qc/) | origin/arrival/assoc tables; 31,020-event QC subset | tag `stage-reloc-cc-qc-ver3` |
| 5. Amplitudes (Route A) | [`workflow/05_magnitude/`](workflow/05_magnitude/) (`scale/run_route_a.sh`) | Wood–Anderson amplitude per pick | tag `stage-routeA-v2-amplitudes` |
| 6. Magnitudes | `workflow/05_magnitude/` phase18 → phase19, then `workflow/06_analysis/` phase10 | preferred M (ML < 4.5, Mw above) for 55,707 events | tag `magnitudes-v1` |
| 7. Analysis, figures, paper | [`workflow/06_analysis/`](workflow/06_analysis/), [`figures/`](figures/), [`paper/`](paper/) | manuscript | release tag |

**[PROVENANCE.md](PROVENANCE.md)** records, for every product, the date, the code version
(git tag), the external tool and its version, the environment and checksums, and how to
check out the code of any single stage. GENIE and GraphDD are not part of this repository;
only their outputs are used. [`data/LINEAGE.md`](data/LINEAGE.md) gives the file-by-file
creation chain and row counts. The planned reorganization and Zenodo release are in
[CLEANUP_PLAN.md](CLEANUP_PLAN.md).

## Repository structure

```
📜 README.md · PROVENANCE.md · INSTALL.md · LICENSE · CITATIONS.cff · CLEANUP_PLAN.md
📜 pixi.toml / pixi.lock  # environments: default, internal (pnwstore), amplitude, paper
📜 Makefile           # `make paper` (manuscript), `make figs` (collect paper figures)
📜 download_data.sh   # rsync pipeline input catalogs from the lab server
📦 workflow           # the pipeline, in run order
 ┣ 📦 01_picking        # ELEP picking: run_picking.py + picking_config.csv; v5 picker dryrun_models.py; legacy/
 ┣ 📦 02_association    # GENIE (external): version and outputs
 ┣ 📦 03_relocation     # GraphDD (external); CC differential-time waveform dataset builders
 ┣ 📦 04_merge_qc       # merge GraphDD + CC output; QC notebooks that write the final catalog
 ┣ 📦 05_magnitude      # Route A amplitudes and magnitudes (phase1-4, 16-19), counts baseline
 ┗ 📦 06_analysis       # maps and supplement analyses (phase5, 7-14), classification (phase10)
📦 scale              # at-scale drivers: run_route_a.sh, run_fill.sh, inventory builders, RUNBOOK.md
📦 notebooks/diagnostics  # notebooks behind QC choices; no paper figure
📦 figures            # notebooks for Figs 1, 2, 3 (QC histograms), 4 (picks), 8
📦 data               # small inputs and config; large CSVs as chunks in data/split_files/
📦 utils              # data_client, paper_style, plot_utils, qc_utils, fetch_*, Zenodo helpers
📦 paper              # Quarto → Seismica manuscript, auto-synced to Overleaf
```

Removed in the 2026-09 cleanup and kept at tag `pre-cleanup-2026-09`: `old/` (abandoned
PyOcto association and HypoInverse location), `3_post_processing/` (a duplicate of
the old `4_relocation/`), `0_data_availability/`, and the superseded `*_old.csv` amplitude files.

## Installation

**Recommended: [pixi](https://pixi.sh)** — one command, reproducible.

```sh
# 1. Install pixi (once)
curl -fsSL https://pixi.sh/install.sh | bash

# 2. Clone (lightweight: see "Cloning" below)
git clone --filter=blob:none https://github.com/Denolle-Lab/cascadia_obs_ensemble.git
cd cascadia_obs_ensemble

# 3a. Public / EarthScope FDSN (default — works anywhere)
pixi install
# 3b. UW internal — also installs pnwstore waveform-archive access
pixi install --environment internal
# 3c. Manuscript toolchain (Quarto + LaTeX) — only if building the paper
pixi install --environment paper

# 4. Verify the install (imports the core deps + notebook stack)
pixi run verify

# 5. Use it
pixi run notebook        # Jupyter (notebook workflow)
pixi run pick            # example CLI entry point (workflow/01_picking)
```

### Cloning without the full history

The git history is about 1.4 GB, mostly large CSVs and notebooks that were deleted long ago.
It is kept on purpose, so that every stage in [PROVENANCE.md](PROVENANCE.md) can be checked
out. The files at the tip are about 0.3 GB. You rarely need the history, so pick the lightest
clone that works for you:

```sh
# files at the tip only, no history (smallest; cannot check out older tags)
git clone --depth 1 https://github.com/Denolle-Lab/cascadia_obs_ensemble.git

# full commit history, but file contents fetched only when needed (recommended):
# fast to clone, and `git checkout <tag>` still works (it downloads what that tag needs)
git clone --filter=blob:none https://github.com/Denolle-Lab/cascadia_obs_ensemble.git

# later, get one stage's code without the rest of the history
git fetch --depth 1 origin tag stage-routeA-v2-amplitudes
git worktree add ../cascadia-routeA-v2 stage-routeA-v2-amplitudes

# turn a shallow clone into a full one, if ever needed
git fetch --unshallow
```

**Conda fallback:** `conda env create -f environment.yml && conda activate seismo_cobs`
**pip fallback:** `pip install -r requirements.txt` (note: `basemap` and `pygmt` need
conda-forge / a system GMT; `obspy` is easiest via conda-forge).

## Data

The pipeline produces three **nested** datasets (each a smaller, higher-quality subset:
~39.6 M ELEP picks → 1.09 M associated picks / 116,591 events → 1.00 M relocated picks /
63,887 events → 31,020 events after QC). See [`data/LINEAGE.md`](data/LINEAGE.md) for the
full creation-time lineage, row counts, and which files are final vs alternative.

Three distinct data products span the pipeline; they are **not** the same artifact:

1. **Raw ELEP picks** (per-station P/S picks, 2010–2015) — the picker output, *upstream*
   of association. Currently on Google Drive (private until archived on Zenodo):
   [Picks folder](https://drive.google.com/drive/folders/1ACsaRj3GY-kBwPoXGb-RCDAlEiM3ArJP).
2. **Relocated catalog + picks** (shipped in this repo, `data/`) — the GraphDD-relocated
   event catalog (`Cascadia_relocated_catalog_ver_3.csv`) and its phase picks
   (`Cascadia_relocated_catalog_picks_ver_3*`, with optional Wood-Anderson amplitudes).
   These exceed GitHub's file-size limit, so they are committed as ≤50 MB chunks in
   `data/split_files/` (via `utils/split_large_csvs.py`). Rebuild the monoliths with:
   ```sh
   pixi run python utils/reconstruct_split_csvs.py        # --list to preview
   ```
3. **Final cross-correlated + QC-filtered catalog** used in the paper figures
   (`origin_2010_2015_reloc_cog_ver3_cc_p_4_s_4_rms_2_5.csv`, plus `arrival_*`,
   `assoc_*`, `all_stations_*`, comparison catalogs) — these live on the lab server,
   not in the repo. Fetch them with:
   ```sh
   ./download_data.sh mdenolle@<host>            # small catalogs (fig1/4/5/6)
   ./download_data.sh mdenolle@<host> --with-picks   # + arrival/assoc (fig3, large)
   ```

**Filename conventions:** `reloc` = GraphDD-relocated; `cog` = center-of-gravity cluster
step; `cc` = cross-correlation-refined; `p_4_s_4_rms_2_5` = QC filter (more than 4 P and
more than 4 S picks, RMS < 2.5 s; this reproduces the 31,020-event file, see
PROVENANCE.md, stage 6). `data/ds01.csv` is Morton et al. (2023); `nodes_*`/`vel_*.csv` are
GraphDD velocity/region config; `jgrb52524-*` are external published supplements.

### Zenodo archive (planned)

For publication, the code will be released through the GitHub–Zenodo integration, and the
data products (ANSS-style pick, association and origin tables with the magnitudes) as a
separate Zenodo data record. The plan is in [CLEANUP_PLAN.md](CLEANUP_PLAN.md) §6;
[`data/ZENODO.md`](data/ZENODO.md) is the older layout proposal.

## Building the manuscript

```sh
make paper       # paper/main.qmd -> paper/main.tex + main.pdf (Quarto -> Seismica -> tectonic)
make figs        # collect figure PNGs into paper/figures/ (needs the data above)
```

To regenerate the magnitude catalog and the figures that depend on it (from
`workflow/05_magnitude/`, `default` env, after the Route A products exist; see the
runbook):

```sh
python phase18_moment_tensor_match.py   # ComCat moment tensors -> data/focal/comcat_mt_matched.csv
python phase19_preferred_magnitude.py   # -> data/magnitude/cascadia_catalog_M_routeA.csv
cd ../06_analysis
python phase10_event_classification.py  # -> cascadia_catalog_classified.csv (+ Fig. S4)
```
See [`paper/README.md`](paper/README.md) for the authoring + Overleaf-sync workflow.

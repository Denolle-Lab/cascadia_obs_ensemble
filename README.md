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

## Repository structure

```
📜 README.md · INSTALL.md · LICENSE · CITATIONS.cff
📜 pixi.toml          # primary environment (pixi); environment.yml / requirements.txt are fallbacks
📜 Makefile           # `make paper` (manuscript), `make figs` (paper figures)
📜 verify_environment.py   # `pixi run verify` — dependency smoke test
📜 download_data.sh   # rsync figure/pipeline input catalogs from the lab server
📦 0_data_availability
📦 1_picking          # ensemble ELEP picking: parallel_pick_20{10-15}*.py, picking_utils*.py
📦 2_association      # (association is run with GENIE; see the graph-network tooling)
📦 3_post_processing  # amplitudes, catalog merge, ANSS concat, QC metrics, cross_correlation/, quality_control/
📦 4_relocation       # relocation post-processing + QC + magnitudes
 ┣ 📜 calculate_amplitudes.py            # per-pick peak amplitudes
 ┣ 📜 qc_metrics_all_regions_*.ipynb     # QC thresholds (P>=4, S>=4, RMS<2.5)
 ┣ 📦 cross_correlation                  # CC differential-time dataset builders
 ┣ 📦 quality_control                    # QC/examine notebooks
 ┗ 📦 magnitude                          # ML magnitude pipeline (phase1..6) + methods docs
📦 data               # catalogs + config (see "Data"); large CSVs live in data/split_files/
📦 utils              # data_client.py, plot_utils.py, qc_utils.py, split_large_csvs.py, reconstruct_split_csvs.py
📦 figures            # fig1..fig6 notebooks for the manuscript
📦 paper              # Quarto → Seismica manuscript, auto-synced to Overleaf
📦 .claude/skills     # pre-submission-reviewer + plain-voice (paper-iteration agents)
```

## Installation

**Recommended: [pixi](https://pixi.sh)** — one command, reproducible.

```sh
# 1. Install pixi (once)
curl -fsSL https://pixi.sh/install.sh | bash

# 2. Clone
git clone https://github.com/Denolle-Lab/cascadia_obs_ensemble.git
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
pixi run pick            # example CLI entry point (1_picking)
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
step; `cc` = cross-correlation-refined; `p_4_s_4_rms_2_5` = QC filter (≥4 P picks, ≥4 S
picks, RMS < 2.5 s). `data/ds01.csv` is Morton et al. (2023); `nodes_*`/`vel_*.csv` are
GraphDD velocity/region config; `jgrb52524-*` are external published supplements.

### Zenodo archive (planned)

For publication, archive the analysis-ready **monolithic** CSVs (not the GitHub
chunks) as a single Zenodo record — see [`data/ZENODO.md`](data/ZENODO.md) for the
proposed package layout and the download recipe. Superseded `*_old` amplitude files
are kept for provenance but excluded from the archive.

## Building the manuscript

```sh
make paper       # paper/main.qmd -> paper/main.tex + main.pdf (Quarto -> Seismica -> tectonic)
make figs        # collect figure PNGs into paper/figures/ (needs the data above)
```
See [`paper/README.md`](paper/README.md) for the authoring + Overleaf-sync workflow.

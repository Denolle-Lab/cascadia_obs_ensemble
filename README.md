# Ensemble Deep Learning to further mine Cascadia offshore data


This project uses an ensemble deep learning algorithm, ELEP (Yuan et al, 2023) to detect and pick P and S waves from continuous data in coastal and offshore Cascadia using the 2011-2015 Cascadia Initiative Experiment Ocean Bottom Seismometers.

The workflow included 1) ELEP detection, 2) testing PyOcto as asssociation, 3) testing GENIE for association, testing HypoDD and GraphDD for relocation.

The repository also provides scripts to compare the new catalog with established catalog in the region ([USGS ComCat](https://earthquake.usgs.gov/earthquakes/search/) and [Morton et al, 2023](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023JB026607)).

Project rteam:
- Hiroto Bito: hbito@uw.edu 
- Zoe Krauss: zkrauss@uw.edu
- Qibin Shi: qibins@uw.edu
- Yiyu Ni: niyiyu@uw.edu
- Marine Denolle: mdenolle@uw.edu
- Nate Stevens: ntstevens@uw.edu

Project team at Stanford:
- Ian MacBreatry
- Yifan Yu

## Repository Structure
```
📜README.md
📜INSTALL.md
📜LICENSE
📜CITATIONS.cff
📜pixi.toml          # primary environment (pixi)
📜environment.yml    # conda fallback
📜requirements.txt   # pip fallback
📦0_data_availability
 ┗ 📜0_data_availability_2011.ipynb
📦1_picking
 ┣ 📜parallel_pick_20{10-15}.py          # main entry points per year
 ┣ 📜parallel_pick_20{10-15}_*.py        # region/channel variants
 ┣ 📜picking_utils.py                    # core ELEP picking logic
 ┣ 📜picking_utils_prio.py               # HH/BH priority variant
 ┣ 📜picking_utils_prio_EH.py            # EH channel variant
 ┣ 📜picking_utils_123_127_HH_BH.py      # lon-band variant
 ┗ 📜tutorial_picking_workflow.ipynb
📦2_association                          # (empty — see PyOcto docs)
📦3_post_processing
 ┣ 📜event_waveform_processing.py        # batch amplitude computation
 ┣ 📜event_waveform_processing.ipynb
 ┣ 📜get_waveform_amplitude.py
 ┣ 📜concat_anss_catalogs_2010_2015.ipynb
 ┣ 📜merge_events_reloc_cog_ver3_cc.ipynb
 ┣ 📜qc_metrics_all_regions_reloc_cog_ver3_cc.ipynb
 ┣ 📦cross_correlation
 ┗ 📦quality_control
📦data
 ┣ 📜Cascadia_relocated_catalog_ver_3.csv
 ┣ 📜Cascadia_relocated_catalog_picks_ver_3.csv
 ┣ 📜Cascadia_relocated_catalog_picks_ver_3_with_amplitudes.csv
 ┣ 📜vel_*.csv        # velocity profiles
 ┣ 📜nodes_*.csv      # region boundaries
 ┗ 📜jgrb52524-*.csv  # Morton et al, 2023 comparison data
📦utils
 ┣ 📜data_client.py   # waveform backend abstraction (FDSN / pnwstore)
 ┣ 📜plot_utils.py
 ┣ 📜qc_utils.py
 ┗ 📜split_large_csvs.py
📦figures
```

# Installation Guide

**Recommended: [pixi](https://pixi.sh)** — single command, reproducible, no Conda setup required.

## Quick start (pixi)

```sh
# 1. Install pixi (once)
curl -fsSL https://pixi.sh/install.sh | bash

# 2. Clone the repository
git clone https://github.com/Denolle-lab/cascadia_obs_ensemble.git
cd cascadia_obs_ensemble

# 3a. Public / EarthScope FDSN (default — works anywhere)
pixi install

# 3b. UW internal — also installs pnwstore waveform archive access
pixi install --environment internal

# 4. Launch a notebook or run a script
pixi run notebook
pixi run verify
```

## Conda fallback

```sh
conda env create -f environment.yml
conda activate seismo_cobs
```

## pip fallback (no Conda)

> Note: `basemap` and `obspy` are easiest via conda-forge. pip-only installs
> may require manual steps for those packages.

```sh
pip install -r requirements.txt
```

## Running the Notebooks

1. **Start Jupyter Notebook:**

    ```sh
    jupyter notebook
    ```

2. **Open the notebooks:**

    In the Jupyter Notebook interface, navigate to the following notebooks and open them:
    <!-- ```
    - ./workflow_simplified/3_associate.ipynb
    - ./workflow_simplified/4_quality_control.ipynb
    ``` -->

## Pick data

After running ELEP, the picks are now available on Gdrive (private access until we upload to zenodo)

* [Picks](https://drive.google.com/drive/folders/1ACsaRj3GY-kBwPoXGb-RCDAlEiM3ArJP)


<!-- * [Picks 2011](https://drive.google.com/file/d/1D2hXbtvMPTiktmcg_ugrvIz8W_82x5Ht/view?usp=drive_link)
* [Picks 2012](https://drive.google.com/file/d/1gq2isg0dOuaRmorQRAc5PJKstBEjFd5L/view?usp=drive_link)
* [Picks 2013](https://drive.google.com/file/d/1M8UNhKxewNG48Rsjnk_DbWl2NpsgXXp8/view?usp=drive_link)
* [Picks 2014](https://drive.google.com/file/d/1sV7yTBDfVhBUixA0NvCmZKyf1L1SJh-C/view?usp=drive_link)
* [Picks 2015](https://drive.google.com/file/d/15Ok11F3r2Ia-5KanmlMGCDOhL0Usckpr/view?usp=drive_link)
 -->
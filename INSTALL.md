# Installation Guide

## Overview

This project supports three install paths:

| Method | When to use |
|--------|-------------|
| **pixi** (recommended) | Any user — reproducible, single command |
| **conda** (`environment.yml`) | If you already have Anaconda/Miniconda |
| **pip** (`requirements.txt`) | Lightweight / CI; note basemap requires conda |

---

## 1. pixi (recommended)

[pixi](https://pixi.sh) handles both conda and PyPI packages from a single `pixi.toml`.

### Install pixi

```sh
curl -fsSL https://pixi.sh/install.sh | bash
# restart your shell or: source ~/.bashrc
```

### Clone and install

```sh
git clone https://github.com/Denolle-lab/cascadia_obs_ensemble.git
cd cascadia_obs_ensemble
```

#### Public / EarthScope FDSN (default)

Works anywhere. Waveform downloads use the EarthScope FDSN client.

```sh
pixi install
```

#### UW internal (adds pnwstore)

Adds `pnwstore` for direct access to the UW waveform archive.
NC/BK networks are still fetched via NCEDC FDSN.

```sh
pixi install --environment internal
```

### Useful tasks

```sh
pixi run verify     # run verify_environment.py
pixi run notebook   # launch Jupyter
```

### GPU (PyTorch CUDA)

The default `pixi.toml` installs CPU PyTorch. To enable GPU:

```sh
pixi run python -c "import torch; print(torch.cuda.is_available())"
# If False, install pytorch via the pytorch channel manually:
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## 2. Conda fallback (`environment.yml`)

```sh
conda env create -f environment.yml
conda activate seismo_cobs
```

To also install pnwstore (UW internal):

```sh
conda activate seismo_cobs
git clone https://github.com/niyiyu/pnwstore.git
cd pnwstore
pip install .
```

---

## 3. pip fallback (`requirements.txt`)

> **Note:** `obspy` and `basemap` are easiest via conda-forge.
> A pip-only install may require extra steps for those packages.

```sh
pip install obspy           # or: conda install -c conda-forge obspy
pip install -r requirements.txt
# basemap: conda install -c conda-forge basemap
```

---

## Waveform data access

All waveform download logic is centralized in `utils/data_client.py`.

| `--source` flag | Backend | Env required |
|-----------------|---------|--------------|
| `fdsn` (default) | EarthScope FDSN | any |
| `pnwstore` | UW waveform archive + NCEDC for NC/BK | internal |

### Picking scripts (`1_picking/`)

```sh
# default (pnwstore — internal env):
python parallel_pick_2011.py

# public FDSN:
python parallel_pick_2011.py  # edit source='fdsn' in run_detection() call
```

`run_detection()` accepts a `source` keyword argument defaulting to `'pnwstore'`.

### Event waveform processing (`3_post_processing/`)

```sh
# public (EarthScope FDSN)
python event_waveform_processing.py \
    --events data/Cascadia_relocated_catalog_ver_3.csv \
    --picks  data/Cascadia_relocated_catalog_picks_ver_3.csv \
    --source fdsn

# UW internal (pnwstore)
python event_waveform_processing.py \
    --events data/Cascadia_relocated_catalog_ver_3.csv \
    --picks  data/Cascadia_relocated_catalog_picks_ver_3.csv \
    --source pnwstore
```

---

## Verify your install

```sh
pixi run verify          # pixi
# or:
python verify_environment.py
```

Expected output: all critical packages pass.
`basemap` is optional (only needed for map plots in `utils/plot_utils.py`).
`pnwstore` is optional (only needed for the `internal` environment).

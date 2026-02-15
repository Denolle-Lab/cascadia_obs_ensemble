# Installation Guide - Cascadia OBS Ensemble

This guide provides detailed instructions for setting up the Python environment needed to run the Cascadia OBS Ensemble picking and processing workflows.

## Quick Start (Recommended: Conda)

```bash
# Clone the repository
git clone https://github.com/Denolle-lab/cascadia_obs_ensemble.git
cd cascadia_obs_ensemble

# Create and activate conda environment
conda env create -f environment.yml
conda activate seismo_cobs

# Verify installation
python verify_environment.py
```

If the verification passes, you're ready to go! If not, see the troubleshooting section below.

## Detailed Installation Options

### Option 1: Conda (Recommended)

Conda provides better dependency resolution for scientific packages, especially for complex dependencies like ObsPy and PyTorch.

**Step 1: Install Conda**
- Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)

**Step 2: Create Environment**
```bash
cd cascadia_obs_ensemble
conda env create -f environment.yml
```

**Step 3: Activate Environment**
```bash
conda activate seismo_cobs
```

**Step 4: Verify Installation**
```bash
python verify_environment.py
```

### Option 2: Pip Only (Without Conda)

If you prefer not to use conda or need a pure pip installation:

**Step 1: Create Virtual Environment**
```bash
cd cascadia_obs_ensemble
python -m venv venv
```

**Step 2: Activate Virtual Environment**
```bash
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

**Step 3: Upgrade pip**
```bash
pip install --upgrade pip
```

**Step 4: Install Packages**
```bash
# Install core dependencies
pip install -r requirements.txt

# Install GitHub packages
pip install git+https://github.com/congcy/ELEP.git
pip install git+https://github.com/niyiyu/pnwstore.git
```

**Step 5: Verify Installation**
```bash
python verify_environment.py
```

## GPU Support (Optional)

For faster deep learning inference with GPU acceleration:

### Check GPU Availability
```bash
# Check if NVIDIA GPU is available
nvidia-smi
```

### Install PyTorch with CUDA Support

Replace the CPU-only PyTorch with GPU-enabled version:

**For CUDA 11.8:**
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) for other CUDA versions.

### Verify GPU
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Package Details

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.22 | Numerical computing |
| pandas | ≥1.5 | Data manipulation |
| scipy | ≥1.9 | Scientific computing |
| matplotlib | ≥3.5 | Plotting |
| obspy | ≥1.4 | Seismology data processing |
| seisbench | ≥0.4 | Deep learning models for seismology |
| torch | ≥1.13 | Deep learning framework |
| dask | ≥2022.10 | Parallel computing |
| h5py | ≥3.7 | HDF5 file I/O |
| basemap | ≥1.3.6 | Geographic plotting |
| tqdm | ≥4.64 | Progress bars |

### GitHub Packages (Required)

These packages are not available on PyPI and must be installed from GitHub:

**ELEP** - Ensemble Learning for Earthquake Picking
- Repository: https://github.com/congcy/ELEP
- Installation: `pip install git+https://github.com/congcy/ELEP.git`
- Used for: Ensemble deep learning phase picking

**pnwstore** - Pacific Northwest Seismic Network Data Store
- Repository: https://github.com/niyiyu/pnwstore
- Installation: `pip install git+https://github.com/niyiyu/pnwstore.git`
- Used for: Accessing waveform data from PNW seismic networks

### Optional Packages

**pyocto** - Phase Association
- Repository: https://github.com/seisbench/pyocto
- Installation: `pip install git+https://github.com/seisbench/pyocto.git`
- Used for: Phase association (if using association workflow)

## Troubleshooting

### Common Issues

#### 1. ELEP or pnwstore Import Fails

**Problem:**
```
ImportError: No module named 'ELEP'
```

**Solution:**
```bash
pip install git+https://github.com/congcy/ELEP.git
pip install git+https://github.com/niyiyu/pnwstore.git
```

#### 2. ObsPy Installation Issues

**Problem:** ObsPy fails to install with pip

**Solution:** Use conda instead
```bash
conda install -c conda-forge obspy
```

#### 3. Basemap Not Found

**Problem:**
```
ImportError: No module named 'mpl_toolkits.basemap'
```

**Solution:**
```bash
# With conda:
conda install -c conda-forge basemap

# With pip (may require additional system dependencies):
pip install basemap
```

#### 4. PyTorch CUDA Mismatch

**Problem:** PyTorch can't find CUDA or version mismatch

**Solution:**
```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch version
# See https://pytorch.org/get-started/locally/
```

#### 5. Dask Import Warnings

**Problem:** Warnings about missing optional dask dependencies

**Solution:**
```bash
# Install complete dask with all extras
pip install "dask[complete]"
```

### Verify Specific Package

Test if a specific package works:

```bash
# Test ELEP
python -c "from ELEP.elep import ensemble_statistics; print('ELEP OK')"

# Test pnwstore
python -c "from pnwstore.mseed import WaveformClient; print('pnwstore OK')"

# Test seisbench
python -c "import seisbench.models as sbm; print('seisbench OK')"

# Test ObsPy
python -c "from obspy.clients.fdsn import Client; print('ObsPy OK')"
```

## Testing Your Installation

### Run Verification Script
```bash
python verify_environment.py
```

This checks all required packages and reports any issues.

### Test with Jupyter Notebook

```bash
jupyter notebook 4_relocation/test_event_waveform_processing.ipynb
```

Run the first few cells to verify data loading and waveform processing works.

### Test Picking Script (Dry Run)

To verify the picking workflow without running the full parallel processing:

```bash
cd 1_picking
python -c "from picking_utils import *; print('Picking utilities loaded successfully')"
```

## Updating Packages

To update packages to their latest versions:

```bash
# Update all conda packages
conda update --all

# Update specific pip package
pip install --upgrade seisbench

# Update GitHub packages
pip install --upgrade git+https://github.com/congcy/ELEP.git
pip install --upgrade git+https://github.com/niyiyu/pnwstore.git
```

## Environment Management

### List Installed Packages
```bash
conda list  # If using conda
pip list    # If using pip
```

### Export Environment
```bash
# Export current environment
conda env export > environment_snapshot.yml

# Or for pip
pip freeze > requirements_snapshot.txt
```

### Remove Environment
```bash
# Deactivate first
conda deactivate

# Remove environment
conda env remove -n seismo_cobs
```

## Alternative Installation: Mamba

For faster package resolution, you can use [Mamba](https://mamba.readthedocs.io/) instead of conda:

```bash
# Install mamba
conda install -c conda-forge mamba

# Create environment with mamba
mamba env create -f environment.yml
```

## Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/Denolle-lab/cascadia_obs_ensemble/issues)
2. Run `python verify_environment.py` and report the output
3. Contact the project maintainers (see README.md)

## System Requirements

- **OS:** Linux, macOS, or Windows (Linux/macOS recommended)
- **RAM:** Minimum 8GB, 16GB+ recommended for parallel processing
- **Disk:** ~10GB for environment + space for data
- **GPU:** Optional, but recommended for faster processing (NVIDIA CUDA-compatible)

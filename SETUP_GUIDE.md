# Quick Reference - Environment Setup

## Files Created/Updated

### New Files
- **verify_environment.py** - Script to verify all dependencies are installed correctly
- **INSTALL.md** - Comprehensive installation guide with troubleshooting

### Updated Files
- **environment.yml** - Clean conda environment specification with all required packages
- **requirements.txt** - Clean pip requirements (alternative to conda)
- **README.md** - Updated installation section to reference INSTALL.md

## Installation Commands

### One-Line Setup (Recommended)
```bash
conda env create -f environment.yml && conda activate seismo_cobs && python verify_environment.py
```

### Step-by-Step
```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate seismo_cobs

# Verify installation
python verify_environment.py
```

## Common Commands

### Activate Environment
```bash
conda activate seismo_cobs
```

### Check What's Installed
```bash
python verify_environment.py
```

### Update Environment
```bash
conda env update -f environment.yml --prune
```

### Reinstall GitHub Packages
```bash
pip install --upgrade --force-reinstall git+https://github.com/congcy/ELEP.git
pip install --upgrade --force-reinstall git+https://github.com/niyiyu/pnwstore.git
```

### Export Current Environment
```bash
conda env export > my_environment.yml
```

## Core Package List

### From PyPI/Conda
- numpy (≥1.22)
- pandas (≥1.5)
- scipy (≥1.9)
- matplotlib (≥3.5)
- obspy (≥1.4)
- seisbench (≥0.4)
- torch (≥1.13)
- dask[complete] (≥2022.10)
- h5py (≥3.7)
- basemap (≥1.3.6)
- tqdm (≥4.64)
- openpyxl (≥3.0)
- adjustText (≥0.8)

### From GitHub
- ELEP: `pip install git+https://github.com/congcy/ELEP.git`
- pnwstore: `pip install git+https://github.com/niyiyu/pnwstore.git`

## Testing

### Verify Environment
```bash
python verify_environment.py
```

### Test Core Scripts
```bash
# Test picking utilities
cd 1_picking
python -c "from picking_utils import *; print('Success')"

# Test event processing
cd 4_relocation
python -c "from event_waveform_processing import find_column, process_event; print('Success')"
```

### Run Test Notebook
```bash
jupyter notebook 4_relocation/test_event_waveform_processing.ipynb
```

## Troubleshooting Quick Fixes

### ELEP Not Found
```bash
pip install git+https://github.com/congcy/ELEP.git
```

### pnwstore Not Found
```bash
pip install git+https://github.com/niyiyu/pnwstore.git
```

### ObsPy Issues
```bash
conda install -c conda-forge obspy
```

### Basemap Issues
```bash
conda install -c conda-forge basemap
```

### Full Reinstall
```bash
conda deactivate
conda env remove -n seismo_cobs
conda env create -f environment.yml
conda activate seismo_cobs
```

## What Changed

### Before (Old requirements.txt)
- 160+ system packages (blivet, jupyterhub, chrome-gnome-shell, etc.)
- Outdated versions (numpy 1.14.3 from 2018)
- No GitHub packages specified
- No clear installation path

### After (New Setup)
- ~15 core scientific packages with modern versions
- Clear separation of PyPI vs GitHub packages
- Comprehensive documentation (INSTALL.md)
- Automated verification (verify_environment.py)
- Both conda and pip installation paths documented

## Core Workflows Supported

### Picking Workflow
**Scripts:** `1_picking/parallel_pick_*.py`
**Key dependencies:** obspy, seisbench, torch, ELEP, pnwstore, dask

### Event Processing Workflow
**Scripts:** `4_relocation/event_waveform_processing.py`
**Key dependencies:** obspy, pnwstore, numpy, pandas

## Next Steps After Installation

1. **Verify environment:**
   ```bash
   python verify_environment.py
   ```

2. **Test with sample data:**
   ```bash
   jupyter notebook 4_relocation/test_event_waveform_processing.ipynb
   ```

3. **Run picking on small dataset:**
   - Edit `1_picking/parallel_pick_*.py` to use smaller date range
   - Test with 1-2 days of data before full run

4. **Check GPU availability (if using):**
   ```bash
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
   ```

For detailed documentation, see **INSTALL.md**.

# Picking Workflow Tutorial

This directory contains scripts for automated earthquake phase picking using ensemble machine learning models on Cascadia seismic data.

## Tutorial Notebook

**`tutorial_picking_workflow.ipynb`** - A comprehensive, interactive tutorial that demonstrates the complete picking workflow on a small subset of data.

### What the Tutorial Covers

1. **Setup and Configuration**: Initialize clients, models, and parameters
2. **Data Retrieval**: Download seismic waveforms from IRIS/PNWStore
3. **Channel Selection**: Priority-based selection (HH > BH > EH)
4. **Preprocessing**: Filtering, resampling, and windowing
5. **Ensemble ML Predictions**: Apply 5 pre-trained EQTransformer models
6. **Semblance Calculation**: Combine predictions using ELEP ensemble statistics
7. **Phase Detection**: Identify P and S wave arrivals
8. **Visualization**: Plot waveforms with picks and probability distributions
9. **Results Analysis**: Statistics and quality assessment

### Tutorial Parameters

The tutorial processes:
- **2 stations** (or fewer if limited availability)
- **1 day** of data (March 15, 2011)
- **Limited geographic region**: 44-46°N, 124-126°W
- **Networks**: UW, UO, 7D, C8

This small scope allows you to:
- Understand the complete workflow (~10-20 minutes runtime)
- Inspect intermediate results
- Validate the method before scaling up

### Running the Tutorial

```bash
# Activate your conda environment
conda activate cascadia_obs

# Launch Jupyter notebook
jupyter notebook tutorial_picking_workflow.ipynb
```

Or open directly in VS Code with the Jupyter extension.

### Expected Output

The tutorial will generate:
- **CSV files**: Pick times and probabilities in `tutorial_picks/`
- **Plots**: Waveforms with picks and probability functions
- **Statistics**: Pick counts, probability distributions, timing analysis

## Production Scripts

Once comfortable with the tutorial, the production scripts process full datasets:

### Main Parallel Processing Scripts

```
parallel_pick_20XX.py              # Full year, all stations
parallel_pick_20XX_HH_BH.py        # High-rate channels (HH/BH priority)
parallel_pick_20XX_123_127_EH.py   # EH channels (analog stations)
parallel_pick_20XX_122-123_46-50.py # Specific station subsets
```

### Utility Modules

- **`picking_utils.py`**: Core processing functions for HH/BH channels
- **`picking_utils_prio_EH.py`**: Specialized functions for EH priority
- **`picking_utils_prio.py`**: Alternative prioritization schemes
- **`picking_utils_123_127_HH_BH.py`**: Region-specific utilities

## Workflow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Station Inventory Query                   │
│      (IRIS FDSN - all networks in Cascadia region)          │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Create Task List (Station × Day)                │
│         One task per station per day combination             │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                 Dask Parallel Scheduler                      │
│        Distribute tasks across multiple workers              │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
┌─────────────────┐           ┌─────────────────┐
│  Worker Task 1  │    ...    │  Worker Task N  │
└────────┬────────┘           └────────┬────────┘
         │                              │
         ▼                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Per-Task Processing:                      │
│                                                              │
│  1. Download waveforms (PNWStore or NCEDC)                  │
│  2. Select channels (HH > BH > EH priority)                 │
│  3. Filter (4-15 Hz) & resample (100 Hz)                    │
│  4. Create overlapping windows (60s, 30s step)              │
│  5. Normalize (std for 'original', max for others)          │
│  6. Run 5 EQTransformer models                              │
│  7. Calculate ensemble semblance (ELEP)                     │
│  8. Stack predictions across windows                        │
│  9. Detect picks (threshold-based triggering)               │
│ 10. Save to CSV                                             │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Individual Pick CSV Files                       │
│        {network}_{station}_{date1}_{date2}.csv              │
└─────────────────────────────────────────────────────────────┘
```

## Key Parameters

### Window Parameters
```python
twin = 6000      # 60 seconds at 100 Hz
step = 3000      # 30-second overlap (50%)
l_blnd = 500     # 5-second blind zone at start
r_blnd = 500     # 5-second blind zone at end
```

### Detection Thresholds
```python
p_thrd = 0.05    # P-wave probability threshold
s_thrd = 0.05    # S-wave probability threshold
```

### Preprocessing
```python
freqmin = 4.0    # High-pass filter (Hz)
freqmax = 15.0   # Low-pass filter (Hz)
target_fs = 100  # Resampling rate (Hz)
```

## Pre-trained Models

The workflow uses 5 EQTransformer models from different regions:

| Model      | Training Region          | Characteristics                    |
|------------|--------------------------|-----------------------------------|
| original   | Pacific Northwest        | Best for local Cascadia events    |
| ethz       | Switzerland/Europe       | Alpine geology, good generalization|
| instance   | Global compilation       | Diverse event types              |
| scedc      | Southern California      | Similar tectonic setting         |
| stead      | Global synthetic         | Robust to noise                  |

### Why Ensemble?

Combining multiple models:
- Reduces false positives
- Improves robustness to noise and site effects
- Captures diverse waveform characteristics
- Provides confidence estimates via semblance

## Parallel Processing with Dask

Production scripts use Dask for parallelization:

```python
# Create delayed tasks
lazy_results = [
    dask.delayed(loop_days)(task, filepath, twin, step, l_blnd, r_blnd) 
    for task in task_list
]

# Execute with progress bar
with ProgressBar():
    compute(lazy_results, scheduler='processes', num_workers=8)
```

### Adjusting Worker Count

```python
num_workers=2   # Conservative for testing
num_workers=8   # Typical for workstation
num_workers=32  # HPC cluster
```

**Memory consideration**: Each worker loads all 5 models (~2-3 GB per worker)

## Channel Priority Logic

### Standard Channels (HH/BH priority)
```python
if has_HH:
    use HH channels      # 100 Hz broadband
elif has_BH:
    use BH channels      # 20-40 Hz broadband
elif has_EH:
    use EH channels      # High-frequency short-period
```

### EH Priority (Analog Stations)
```python
if has_EH:
    use EH channels      # Prioritize for PNSN analog
elif has_HH:
    use HH channels
elif has_BH:
    use BH channels
```

Different scripts handle different priorities based on data availability patterns.

## Output Format

CSV files contain:

| Column           | Description                              |
|------------------|------------------------------------------|
| network          | Network code (UW, UO, etc.)             |
| station          | Station code                            |
| location         | Location code                           |
| band_inst        | Band and instrument codes               |
| label            | Phase type (P or S)                     |
| trace_starttime  | Trace start time                        |
| trigger_onset    | Detection window start                  |
| pick_time        | Peak probability time (best estimate)   |
| trigger_offset   | Detection window end                    |
| max_prob         | Maximum probability in window           |
| thresh_prob      | Threshold used for detection            |

## Common Issues and Solutions

### 1. No Data Available
**Symptom**: `FDSNNoDataException` or empty streams

**Solutions**:
- Check network/station codes are correct
- Verify time range matches data availability
- Some networks require different clients (NC/BK use NCEDC)

### 2. Memory Issues
**Symptom**: Out of memory errors

**Solutions**:
- Reduce `num_workers`
- Process smaller time ranges
- Use CPU instead of GPU if limited VRAM

### 3. Too Many/Few Picks
**Symptom**: Pick counts seem unrealistic

**Solutions**:
- Adjust `p_thrd` and `s_thrd` (higher = fewer picks)
- Check filter frequencies for site characteristics
- Verify data quality (gaps, spikes)

### 4. Slow Processing
**Symptom**: Very slow progress

**Solutions**:
- Increase `num_workers` (if memory allows)
- Use GPU if available (`device = torch.device("cuda")`)
- Process limited geographic region first

## Testing Before Production

1. **Run Tutorial**: Execute the notebook on 1-2 stations, 1 day
2. **Inspect Outputs**: Check CSV format and pick quality
3. **Verify Waveforms**: Examine plots to validate picks
4. **Small Batch Test**: Run production script on a few days
5. **Scale Up**: Process full year with appropriate parallelization

## Example Production Usage

```bash
# Process 2011 with HH/BH priority
python parallel_pick_2011_HH_BH.py

# Process 2012 EH channels (analog stations)
python parallel_pick_2012_123_127_EH.py

# Process specific geographic subset
python parallel_pick_2013_127-129_46-50.py
```

## Next Steps After Picking

1. **Quality Control**: Analyze pick distributions and probabilities
2. **Association**: Group picks into events (see `../association/`)
3. **Location**: Determine earthquake locations
4. **Relocation**: Refine locations (see `../4_relocation/`)
5. **Catalog Assembly**: Combine all years into final catalog

## References

- **EQTransformer**: Mousavi et al. (2020), Nature Communications
- **ELEP**: Ensemble Learning for Earthquake Phase picking
- **SeisBench**: Woollam et al. (2022), Seismological Research Letters
- **Cascadia OBS**: Ocean bottom seismometer deployments

## Support

For questions about:
- **Tutorial**: Review cells and markdown explanations
- **Production Scripts**: Check inline comments and docstrings
- **Parameters**: See this README or notebook summary section
- **Errors**: Enable verbose logging and check data availability

---

**Happy Picking!** 🌊🏔️

# Polarity extraction for catalog v4 — status and handoff

**Status (2026-11-22): PAUSED.** We wait for corrected deep-learning polarity models before running the extraction.
Everything needed to resume is listed here. Branch `polarity/axialpolcap-test` (from `main`), worktree
`/home/mdenolle/cascadia_obs_ensemble_polarity`.

## 1. Goal

Add a P first-motion polarity (with confidence and QC fields, keyed by `pick_id`) to every catalog v4 P pick, then invert
for focal mechanisms with SKHASH using polarities plus the S/P amplitude ratios we already have.

- AxialPolCap (FM_ML, Zhang, Denolle, Wilcock, Waldhauser, Wang, Tolstoy, Tan; in prep. SRL) for OBS picks.
- The parent model (PolarCAP) or another model for onshore picks, to be decided after a bake-off.
- Mechanisms need takeoff angles from the Track A relocation (branch `reloc/track-a`). Polarity extraction itself does not.

## 2. Inputs (unchanged, ready)

| item | where | notes |
|---|---|---|
| v4 picks | `data/catalog_v4/Cascadia_picks_catalog_v4.csv` (local, gitignored) | 1,276,597 P, 1,226,592 S; separator `", "`; station as `STA.NET` |
| v4 events | `data/catalog_v4/Cascadia_relocated_catalog_v4.csv` | 172,340 events 2010–2016 |
| pick_id map | `data/catalog_v4/ridge_report/v4_pick_ids.parquet` | time-sorted, not row-aligned; join on event+station+phase as in `workflow/05_magnitude/v4_build_arrivals.py` |
| S/P amplitudes | `/wd1/mdenolle_data/amps_v6_all` (parquet keyed by pick_id) | Route A WA + displacement, 70 windows, noise; plain floats (do not use `utils/read_amps_v5.py`, which is the Zenodo decoder) |
| waveform reader | `workflow/05_magnitude/route_a_day_amplitudes.py` (`DayReader`) | pnwstore, NC/BK from s3://ncedc-pds, EarthScope fallback; one read per station-day |
| station metadata | `workflow/05_magnitude/station_inventory_v2.xml`, `station_inventory_v5_slim.xml` (local) | both miss 118 v4 stations (148,872 P picks: XD, XG, YH, CC, YW, XU, X4...) — fetch before the full run |
| validation | `data/focal/comcat_mt_matched.csv` (112 ComCat MTs; strike/dip/rake for 60; only 56 within 15 km / 2 s), `data/ds01.csv` (251 Morton et al. 2023 mechanisms) | |

P picks by class: land 823,288 on 362 stations; seafloor 453,309 on 328 stations (OO 143,529 — of which 139,826 at the 7 Axial
training stations; X9 116,202; 7D 101,997; Z5 89,239; NV 2,172; 7A 170). Events with ≥10 P and gap ≤180°: 22,261 (457k P picks);
Axial has 0 such events, Blanco 4,143.

## 3. What was done

1. **Planning workflow** (9 agents): `PLAN.md` (full plan: models per station class, pipeline, output schema, station-polarity QC,
   validation, SKHASH, milestones, PI decisions, risks, references) and `PLAN_CRITIQUE.md` (18 issues, 8 high). The plan pre-dates
   the filter-phase finding below; update section 3 (models/preprocessing) when resuming.
2. **PyTorch port of AxialPolCap**: `axialpolcap_torch.py`. Source: FM_ML commit `09633ad3`,
   `06-models/PolarPicker_h5_20260308_154910.keras` (Keras 3.11.3). Matches Keras 3.11 (JAX backend) to max |Δp| = 3e-6, 512/512 same class.
3. **Test on real Axial data**: 12,071 v4 P windows, 24 days (2014-11 to 2016, before/during/after the April 2015 eruption),
   8 Axial stations (AXBA1 not in training). Extraction `axial_test_extract.py`, helpers `polarity_eval.py`.
4. **Notebook documenting the issues**: `notebooks/polarity/axialpolcap_axial_v4_test.ipynb`, built by
   `build_axial_test_notebook.py`, executed with outputs. Shareable link:
   https://github.com/Denolle-Lab/cascadia_obs_ensemble/blob/polarity/axialpolcap-test/notebooks/polarity/axialpolcap_axial_v4_test.ipynb

## 4. Findings (why we are waiting)

The reference in these tests is a model-free automatic first motion (first excursion > 4× noise std on a causal 3–20 Hz trace),
not analyst labels: agreement values are a consistency check and are meaningless below ~15 dB.

1. **Output sign depends on the band-pass phase.** Causal (one-pass) 3–20 Hz input → class 1 = up (agreement 0.87–0.88 above
   20 dB). Zero-phase (two-pass: `filtfilt`, MATLAB `bandpass()`, ObsPy `zerophase=True`) or unfiltered input → inverted with the
   same confidence; the two settings agree on only 32 % of picks. Cause: a zero-phase filter puts an opposite-sign precursor before
   the onset. The FM_ML manuscript text says only "bandpass filtered between 3 and 20 Hz".
   *Correction of an earlier claim:* the "model is inverted" result from unfiltered synthetic pulses was wrong; with causal
   filtering synthetic up-onsets give class 1 with p = 1.00.
2. **Confident on noise.** Pre-event noise windows: mean confidence 0.95, 79 % pass conf ≥ 0.8 and entropy ≤ 0.2. P windows below
   5 dB: 76 % pass. The threshold is not a quality filter.
3. **No skill at 5–15 dB** (agreement 0.41–0.46), where many picks sit (median SNR 17.7 dB). Unclear how much is the reference.
4. **Sensitive to the exact filter.** Relative to causal 3–20 Hz order 4: order 2 changes 31 % of classes, 1–20 Hz 42 %, 2–30 Hz 28 %.
5. **Sampling rate ambiguous in the FM_ML repo.** The h5 pipeline is 100 Hz (±0.32 s); `CLAUDE.md` and the real-time plan describe
   200 Hz (±0.16 s). Reading our windows as 200 Hz changes 35 % of classes.
6. **Pick time: moderate sensitivity.** Automatic onset a median 10 ms after the v4 pick (±40 ms). Class changes ~3 % at ±10–20 ms,
   8–12 % at ±50 ms, 20–42 % at ±0.1 s; late windows are worse. Training windows carry an undocumented per-station `DDt` offset.
7. **Per station** (SNR > 15, kept): AXCC1 0.91, AXAS1 0.87, AXEC1/EC2 0.84, AXID1 0.78, AXEC3 0.76, AXAS2 0.73; AXBA1 only 10 picks.
8. **Station metadata is fine**: all Axial Z channels dip −90°, positive sensitivity, no negative stage gains.
9. **Other model facts.** Trained only on the 7 OOI Axial stations (EHZ; HHZ at AXCC1/AXEC2), Z only, no hydrophone, no Cascadia
   Initiative OBS, no land data; 2,009 manually labelled events augmented to ~150k (sign flip, real noise, time shift σ 0.02 s).
   Labels −1→class 0, +1→class 1. Input 64 samples, P at index 32, max-normalised. No "unknown" class. No LICENSE file.
   PolarCAP parent weights (`06-models/PolarCAP.h5`) are referenced but not in the repo; no official PolarCAP weights online.
   The only public pretrained onshore polarity model found: Ross et al. 2018 (SCSN), via seispolarity / Hugging Face.

## 5. Waiting on (questions sent with the notebook, section 9)

1. Exact training filter: function, order, corners, one- or two-pass, before or after 200→100 Hz resampling.
2. Label convention: `Po_{STA}` = +1 means upward on raw Z counts? Labels read on raw or filtered traces?
3. Templates centred on the analyst pick or on pick + `DDt_{STA}`?
4. Confirm 100 Hz input for `PolarPicker_h5_20260308_154910`.
5. 2015 Axial manual polarities (`Felix` struct) as a benchmark — they overlap our v4 Axial picks in time.
6. PolarCAP.h5 parent weights; license for the FM_ML weights.
7. **Fixed models**: retrained / documented AxialPolCap (filter-phase robust, ideally with an "unknown"/noise class and calibrated
   confidence), and an onshore model.

## 6. How to resume

1. Get the new weights. If Keras: port with `python workflow/09_polarity/axialpolcap_torch.py convert MODEL.keras OUT.pt`
   (update `_MAP` / the class if the architecture changed), and re-check against Keras with
   `/wd1/mdenolle_data/polarity/models/keras_ref.py` (run in a throwaway env:
   `pixi exec --spec python=3.11 --spec keras=3.11 --spec jax --spec h5py --spec numpy -- python -I keras_ref.py MODEL x.npy out.npy`).
2. Set the preprocessing in `polarity_eval.py` / the notebook to the confirmed training filter.
3. Re-run the notebook on the same windows (no new data fetch needed):
   ```
   cd <worktree>; export LD_LIBRARY_PATH=$PWD/.pixi/envs/default/lib PATH=$PWD/.pixi/envs/default/bin:$PATH
   python workflow/09_polarity/build_axial_test_notebook.py
   jupyter nbconvert --to notebook --execute --inplace notebooks/polarity/axialpolcap_axial_v4_test.ipynb
   ```
   (`DATA`/`MODELS` paths are at the top of the notebook; the worktree has no `.pixi` or `data/`, so run from the main checkout
   or symlink them.)
4. If the manual labels arrive: match them to v4 picks (station + time within ~0.1 s), report accuracy vs SNR, pick_prob, station.
5. Gates before the full run (see `PLAN.md` §8): sign check, accuracy ≥ target on labels, noise rejection, then a pilot on
   one region, then all 1.28 M P picks via the `DayReader` station-day loop (CPU only: no GPU; 128 threads; model is tiny).

## 7. File and data locations

| what | where |
|---|---|
| code, notebook, plan | this branch: `workflow/09_polarity/`, `notebooks/polarity/` |
| FM_ML model (original) + converted weights + Keras reference outputs + FM_ML commit | `/wd1/mdenolle_data/polarity/models/` (`PolarPicker_h5_20260308_154910.keras`, `axialpolcap_20260308_154910.pt`, `x.npy`, `keras_prob.npy`, `keras_ref.py`, `FM_ML_COMMIT`) |
| Axial test windows | `/wd1/mdenolle_data/polarity/axial_test/axial_windows_24d.npz` (12,071 × 500 samples, 100 Hz, P at 300, + metadata JSON); `smoke.npz`; `extract_24d.log` |
| notebook as HTML | `/wd1/mdenolle_data/polarity/axial_test/axialpolcap_axial_v4_test.html` |
| FM_ML clone | session scratchpad only (not durable); re-clone https://github.com/maochuan-zhang-UW/FM_ML at `09633ad3` |

Environment gotchas: the default pixi env needs `LD_LIBRARY_PATH=.pixi/envs/default/lib` (pandas/libstdc++); pnwstore needs
`PYTHONPATH=/wd1/mdenolle_data/pyext`; obspy `Trace.decimate` uses a causal IIR, so the extractor decimates with a zero-phase FIR
(flat in 3–20 Hz) to keep the raw onset shape.

# P-polarity extraction for catalog v4 — plan

Status: plan only. No jobs have run and nothing in the repo was changed. Facts come from four investigator reports plus one check made while writing this plan. Unverified items are marked "UNKNOWN — check".

**Found while writing this plan.** The untracked directory `workflow/09_polarity/` on `reloc/track-a` already holds two scripts, dated 2026-11-22 07:46–07:57:

- `axialpolcap_torch.py` is a PyTorch port of the AxialPolCap `.keras` weights. It reads `model.weights.h5` from the zip with h5py, so TensorFlow is not needed. It has `convert` and `selftest` (random input) subcommands.
- `axial_test_extract.py` cuts raw-count Z windows from −3 s to +2 s around v4 P picks at the Axial OO stations. It decimates to 100 Hz with a zero-phase anti-alias FIR and writes the windows to npz.

Whether the port gives the same outputs as Keras on real windows: UNKNOWN — check. Both scripts sit in the shared Track A tree. Move them to a worktree branch off `main` before building on them.

## 1. Goal and scope

For every one of the 1,276,597 v4 P picks, produce a signed first-motion polarity with a calibrated confidence and QC fields, keyed by `pick_id`. Join these to the Route A S/P amplitudes and invert for focal mechanisms with SKHASH once Track A locations exist.

The scientific product is the mechanism catalog. It feeds three uses:

- regional tectonics: Blanco, Gorda/Mendocino, the forearc and slab, and the eastern slice;
- stress inversion;
- nodal-plane checks of the 74 v0 fault surfaces.

Polarity extraction (M0–M4) does not depend on Track A. Mechanisms (M5) do.

## 2. What we have

**Picks**
- `data/catalog_v4/Cascadia_picks_catalog_v4.csv` holds 1,276,597 P and 1,226,592 S picks. The separator is `, `.
- `pick_id` comes from `data/catalog_v4/ridge_report/v4_pick_ids.parquet`. That file is time-sorted, not row-aligned, so join on event + station + phase as `workflow/05_magnitude/v4_build_arrivals.py` does.
- `data/magnitude_v4/arrivals_v4.csv` already has arid, pick_id and event/station coordinates.

**P picks by station type**
- Land: 823,288 picks on 362 stations.
- Seafloor: 453,309 picks on 328 stations (OO 143,529; X9 116,202; 7D 101,997; Z5 89,239; NV 2,172; 7A 170).
- The seafloor flag comes from `/wd1/mdenolle_data/picks_share/stations/stations_ver5_v6.csv`.
- Largest land networks: PB 251,025 (all EH borehole), UW 172,790, XD 108,395, NC 98,467, CN 65,042, TA 64,855.

**Pick probability and station-days**
- 464,885 P picks have pick_prob > 0.5, and 294,383 have pick_prob > 0.7.
- pick_prob is missing for 222,662 P picks from 2011–2015 (7D 61,775; NC 57,415; CN 41,351).
- The P picks fall on 300,084 station-days (243,846 land, 56,238 seafloor). The median is 2 P picks per station-day.

**Events**
- 30,834 events have ≥10 P picks and 43,095 have ≥8.
- 22,261 events have ≥10 P picks and azimuthal gap ≤180°, carrying 457k P picks. Blanco has 4,143 of these and Axial has 0.

**Amplitudes**
- `/wd1/mdenolle_data/amps_v6_all` holds per-pick Route A amplitudes: WA and displacement over 70 windows, plus noise, `n_comp`, `seg_margin_s`, `sensor`, `epoch` and `status`.
- These are plain floats. Do not use `utils/read_amps_v5.py`, which decodes only the Zenodo format.
- 859,303 event-station pairs (163,826 events) have both P and S with status ok, and P and S always use the same sensor.

**Sensors chosen by the amps run (P picks)**
- Route A took the best channel that has a response, which was often HN: NC 57,399; CN 28,377; TA 26,354; XD 18,567; BK 16,506; YH 7,643.
- 7D shows an "EH" sensor on 16,469 picks. UNKNOWN — check.

**Metadata**
- `workflow/05_magnitude/station_inventory_v2.xml` and `station_inventory_v5_slim.xml` both miss 118 v4 stations, which carry 148,872 P picks (XD 47 stations, XG 20, YH 12, CC 12, YW 9, XU 4, X4 4, and others).
- Neither inventory has ?DH hydrophone channels, although pnwstore holds HDH data.

**Validation material**
- `data/focal/comcat_mt_matched.csv` has 112 ComCat moment tensors (M4.0–6.8). Strike/dip/rake is present for only 60; full tensors are in `data/focal/comcat_mt.csv`. These events carry 866 v4 P picks on 221 stations.
- `data/ds01.csv` has 251 Morton et al. (2023) mechanisms.

**Compute**
- No GPU. 128 threads, 250 GB RAM, 3.0 TB free on `/wd1`.
- The default pixi env has torch 2.13 (CPU), seisbench 0.11.7 (no polarity class) and obspy 1.5. It needs `LD_LIBRARY_PATH=.pixi/envs/default/lib`, and pnwstore through `PYTHONPATH=/wd1/mdenolle_data/pyext`.
- The machine is busy with Track A Grid2Time `time_vpvs176` (about 65 processes) and a fault-surface job.

## 3. Models

### 3.1 FM_ML / AxialPolCap

The repo is https://github.com/maochuan-zhang-UW/FM_ML (HEAD 09633ad3), cloned at `/tmp/claude-39101/-home-mdenolle-cascadia-obs-ensemble/f3df6e58-5dbd-408f-a474-1341eca5ea78/scratchpad/polarity/FM_ML`. It is the code for the AxialPolCap manuscript (Zhang, Denolle, Wilcock, Waldhauser, Wang, Tolstoy, Tan; in prep. for SRL). It has no LICENSE file.

**Architecture and training**
- PolarCAP architecture retrained from scratch. Transfer learning from PolarCAP was tested and rejected.
- Autoencoder plus a Dense(2, softmax) head. Defined in `build_polarPicker()` in `01-scripts/training/train_axialpolcap_h5.py`.
- Training data: 7 OOI cabled Axial stations (AXAS1/AS2/EC1/EC3/ID1 on EHZ, AXCC1/EC2 on HHZ). 2,009 manually verified events, augmented to about 150k waveforms by sign flip, real-noise injection and time shift (σ 0.02 s, clipped at ±0.08 s).
- No Cascadia Initiative OBS, no hydrophone and no land data were used.

**Input**
- (64,1) Z at 100 Hz, P at index 32 (P−0.32 s to P+0.31 s), max-abs normalized.
- Manuscript text gives the preprocessing as demean, detrend, 3–20 Hz bandpass and resample. It is not in code.
- Filter order, causality and response handling: UNKNOWN — check with Zhang.
- Repo docs mention both 100 Hz and 200 Hz. The final H5 pipeline and the weights use 100 Hz.

**Output**
- [p_down, p_up]. No unknown class and no impulsivity output.
- Production keeps conf ≥0.80. The realtime path also requires entropy ≤0.2, which is about conf ≥0.95.

**Weights**
- `06-models/PolarPicker_h5_20260308_154910.keras` (294 KB), saved with Keras 3.11.3.
- The README's TF/Keras 2.13 cannot load it.
- **Reconciled:** use the existing PyTorch port in `workflow/09_polarity/axialpolcap_torch.py` in the default env. Build a Keras 3 / TF ≥2.16 env only to confirm the port matches (max |Δp| < 1e-5 on 1,000 real windows). This avoids a permanent TF env.

**Accuracy**
- About 98.5% held-out, but augmented copies of the same templates appear in train and test, so the number is optimistic.
- LOSO 96.5–99.6%.
- 82.3% agreement with cross-correlation (CC) polarities over 230,565 picks from 2015–2021.
- Median Kagan angle about 13° between CC and ML composite mechanisms.
- Published models on the Axial benchmark: CFM 88.5%, PolarCAP 87.9%, EQPolarity 87.8%, DiTingMotion 82.4%; CC 91.5%.
- There is no evaluation off Axial.

### 3.2 The parent model (PolarCAP) for onshore

PolarCAP (Chakraborty et al. 2022) was trained on INSTANCE (Italy, >130k traces) and reports about 98% test accuracy.

- No official code or weights were found.
- FM_ML refers to `06-models/PolarCAP.h5`, but that file is not in the repo. Zhang may have it.
- PolarCAP has no unknown class, and its ±0.32 s window is sensitive to pick timing. Our ML picks have errors of tens of ms.

**Reconciled choice:** PolarCAP enters the onshore bake-off only if Zhang supplies `PolarCAP.h5`. It is not the default.

### 3.3 Per-class assignment

| Class | P picks | Primary | Challengers | Reason |
|---|---|---|---|---|
| OO Axial (7 AX stations) | part of 143,529 | AxialPolCap | none | Training domain |
| OO non-Axial (HYS*, slope, shelf), 7D, X9, Z5, NV, 7A | about 310k | AxialPolCap, pending gate G2 | RPNet/EquiPol `tepicc_Z`; Ross-SCSN | The only OBS-trained model and the PI's request. Short window avoids most of the water multiple. Transfer untested. |
| Land EH/HH/BH (UW, PB, CN, NC, TA, CC, UO, BK, XD, YH, XU, X4, …) | about 823k | Bake-off winner, decided at G2 | AxialPolCap as a second opinion; PolarCAP if obtained | Candidates: Ross et al. 2018 3-class via SeisPolarity **SCSN** weights (causal 1–20 Hz, ±0.5 s jitter, EH/HH/HN training, has an unknown class); SeisBench 0.13 RPNet / EquiPol `tepicc_Z` (±200 samples, confidence output) |
| HN only, where no velocity Z exists | subset | Same as land, scored as its own stratum | | Accelerometer behavior differs |
| Hydrophone HDH | — | Excluded | | No metadata, and SIO and WHOI DPG signs are opposite |

**Constraints on the candidates**
- Do not use SeisPolarity GLOBAL weights in the benchmark. They include the PNW dataset and would leak into PNSN validation labels.
- SeisBench 0.13.0 was released 2026-11-17 and EquiPol is unpublished. Pin exact versions in a separate env, and treat EquiPol as a challenger only.
- AxialPolCap runs on every pick because inference is cheap.

## 4. Pipeline design

New code goes in `workflow/09_polarity/` on a worktree branch off `main`, starting with the two existing scripts. Outputs go to `/wd1/mdenolle_data/polarity_v4/`.

| File | Reuses | Role |
|---|---|---|
| `p01_build_pick_table.py` | `v4_pick_ids.parquet`; join logic from `v4_build_arrivals.py`; amps `sensor`/`epoch`/`status` from `amps_v6_all`; scratch `v4_P.parquet` | Plan input: `pick_id, network, station, label=P, pick_time`, plus the amps sensor as a hint |
| `p02_day_windows.py` | `plan`/`run`/`consolidate`/`replan`, `DayReader`, the s3 `ncedc-pds` and FDSN helpers, and `with_retries` from `workflow/05_magnitude/route_a_day_amplitudes.py`; window code from `axial_test_extract.py` | New `run_task` (below). Writes `chunks/sNNN_XXXXX.npz` and a parquet of metadata, atomically |
| `p03_infer.py` | `axialpolcap_torch.py`; SeisBench/SeisPolarity adapters | One model per call. Applies each model's filter and cut, runs the shift ensemble, resumes by chunk |
| `p04_station_sign.py` | scratch `inv_scan.py`; FDSN StationXML | Dip and sensitivity per channel-epoch, `sign_meta`, reconciliation of the two inventories |
| `p05_benchmarks.py` | scratch `match_anss.py` | Harvests PNSN QuakeML first motions and NCSN Hypoinverse first motions; builds the hand-label sampler |
| `p06_merge_sp.py` | Route A `select`/`window_for` | P/S amplitude pivot and SKHASH ampfile |
| `p07_skhash_inputs.py` | FM_ML `build_skhash_input_ML_2022_2026.py`, read for its format only | SKHASH control, polarity, amp, station and plfile inputs |
| `scale/run_polarity_windows.sh` | `scale/run_day_amplitudes.sh` | setsid nohup, N shards, 4 retries, STATUS/DONE/FAILED |

### 4.1 `run_task` contract

1. **Sensor choice.** Prefer a velocity Z: HH > BH > EH > SH, then HNZ as a flagged fallback. Polarity needs only the dip and sensitivity sign, not a full response. This moves most NC/BK/TA/XD picks off HN.
2. **Read.** Raw counts, Z only, no `remove_response`.
3. **Window.** **Reconciled:** P−5 s to P+3 s, demeaned, resampled to 100 Hz, giving 800 samples stored as float32. Record `fs_native`. The 5 s before P leaves room for filter edge effects and a 3 s noise segment. 3 s after P covers Ross's ±2 s and AxialPolCap's ±0.32 s windows. Total size is about 4 GB.
4. **Decimation filter.** `axial_test_extract.py` uses a zero-phase FIR. That can put ringing ahead of the onset. Test it against a causal (minimum-phase) decimation in M1 and keep whichever matches the labels better.
5. **Stored fields.** `snr_p` (RMS over P to P+0.5 s divided by RMS over P−3 s to P−0.1 s, on 3–20 Hz), a clip flag, and `seg_margin_s`.

### 4.2 Inference contract

**AxialPolCap**
- Detrend, then a 3–20 Hz 4th-order Butterworth.
- **Reconciled:** run both causal and zero-phase variants until Zhang answers, and keep the one that matches the Axial labels.
- Cut 64 samples with P at sample 32, then max-abs normalize.

**Ross**
- Detrend, causal 1–20 Hz Butterworth, 400 samples centered on P.

**SeisBench models**
- Use their own `classify` preprocessing.

**Shift ensemble**
- **Reconciled:** 7 shifts of −3 to +3 samples (±0.03 s). Store the median probability and the fraction of shifts that agree on sign.

### 4.3 Output schema

`polarity_v4_P.parquet` has one row per (`pick_id`, `model`, `filter_variant`). It is partitioned by network-year.

**Identity**
- `pick_id` (int64), `event_ID`, `network`, `station`, `channel`, `location`
- `sensor_class` (HH/BH/EH/SH/HN), `seafloor`, `fs_native`
- `pick_time`, `pick_prob` (nullable)

**Window quality**
- `snr_p`, `seg_margin_s`, `clip_flag`
- `window_status`: ok, gap, clipped, not_in_segment or no_data

**Metadata sign**
- `dip`, `sens_sign`, `sign_meta` (±1, or 0 if unusable)
- `inv_source` (v2, v5_slim or FDSN), `epoch_id`

**Model output**
- `model`, `model_version` (weights hash), `filter_variant`, `contract_hash`
- `p_up`, `p_down`, `p_unknown` (NaN for binary models), `conf`, `entropy`
- `shift_agree_frac`

**Polarity**
- `pol_raw` (−1/0/+1, in trace sign)
- `station_flip` (default +1; set from the plfile)
- `pol = pol_raw × sign_meta × station_flip`
- `pol_weight = (2·p_up − 1) × sign_meta × station_flip`, set to 0 when the pick fails the retention rule. This is SKHASH's signed `p_polarity`.
- `qc_flag` bitmask: low SNR, clipped, unstable shift, flagged epoch, no metadata

**Companion files**
- `polarity_consensus_v4_P.parquet`: the chosen model per pick.
- `station_polarity_epochs.csv`: the plfile.
- `README` and `MD5SUMS`.

**Retention rule (initial)**
- Start from FM_ML's conf ≥0.80, then calibrate τ per class at G2.
- `shift_agree_frac = 1`.
- `snr_p ≥ s_min`, with s_min set per class at G2.
- Ross: the predicted class is not unknown.
- Shelf OBS (water depth < about 200 m, a proposal): stricter τ. The water multiple arrives about 0.13 s after P at 100 m depth with opposite pressure sign, which is inside every model window.

**Pick selection**
- Extract all P picks; extraction is per station-day, so dropping picks saves little I/O.
- Do not filter on pick_prob.

## 5. Station polarity QC

**Sign rule**

`sign_meta = sign(sensitivity) × (−dip/|dip|)` for |dip| ≥ 45°.

- |dip| < 45° → `sign_meta = 0` and the pick is excluded. This covers KRMB.NC HNZ (dip 0) and X9 AFDAD/EULF1/OBC01 BHZ (dip 0).
- KEMF.NV has dip −84.8 and is treated as −90.
- ObsPy `remove_response` ignores dip, so the sign is applied explicitly.
- `pol_raw` and `pol` are stored separately.

**Conflict to resolve in M0.** The two inventories disagree on Z dip = +90 for 2010–2016:

| Inventory | UW EHZ stations | Z5 GB HHZ (2013-10 to 2014-09) | Other |
|---|---|---|---|
| v2 | 33 | 15, including GB130 and GB360 | PB B024/B031; UO RSBO |
| v5_slim | 21 | 13 | PB B024/B031 |

The v5_slim set carries 59,467 P picks. Diff both inventories against current FDSN StationXML and record the source of each value. Fetch StationXML for the 118 missing stations.

B031 is commented "EHZ and LHZ channels are bad". Exclude it unless tests show it is usable.

**Empirical tests, per channel-epoch, run in order**

1. **PNSN manual labels at the flagged UW stations.** Shows whether the +90 dips are real and whether analysts recorded raw or corrected sign.
2. **Our own teleseismic/regional P check.**
   - Events: M ≥ 6.5 at 30–90°, plus large regional events.
   - Data: instrument-corrected velocity, 0.5–2 Hz.
   - Correlate against a reference stack from land broadbands.
   - Targets: the Z5 GB Year-1 epochs (critical for the Blanco events); the 25 7D Year-4 stations hit by the LDEO error, to confirm the pnwstore copies dated 2021 postdate the 2016-10-07 re-archive (UNKNOWN — check); the 7D "EH" picks; Axial pairs with negative MUSTANG values (AXCC1/AXEC2 vs AXEC1/AXEC3); CN.HOPB; C8.BACQ.
   - MUSTANG `polarity_check` is a pointer only. It is never copied into the plfile: it uses a 0.01 Hz low-pass with no sensor matching, and 7D/Z5 values are noise-dominated.
3. **Neighbor P-waveform similarity.** For co-located small events at stations within 5–10 km. Used as a tie-breaker.
4. **SKHASH `outfile_pol_agree`.** Run after A2/A3 on A/B mechanisms only.
   - Flag a station-epoch when agreement is < 50% with N ≥ 20 (land) or ≥ 30 (OBS).
   - False-alarm probability at per-pick accuracy p = 0.9: 3.4e-5 at N = 15. At p = 0.8: 2.3e-4 at N = 30.
   - Iterate the plfile at most twice.

**Circularity guard.** Validation sets (§6) are never used to set flags, and flagged stations are not scored with the mechanisms that flagged them.

**Gate.** Every station-epoch with more than 500 P picks gets one of: `meta_confirmed`, `empirically_flipped` or `unresolved`. Unresolved epochs get weight 0.

## 6. Validation and benchmarks

**Split.** Split labels by event before choosing a model, then freeze the split: 30% for model selection and calibration, 70% as a locked test set.

**Onshore labels**
- **PNSN.** ComCat QuakeML manual first motions (`<polarity>`, evaluationMode manual).
  - A 20-event sample gave about 4.8 labels per event, mostly UW and PB EH.
  - 9,972 UW events match v4, so the estimate is 40–50k labels. This is extrapolated; harvest 1,000 events first.
  - Coverage is 2010–2015, because the ANSS table ends in 2015.
- **NCSN.** `s3://ncedc-pds/event_phases/YYYY/YYYY.MM.phase.Z`, first motion at column 16.
  - 3,649 matched events lie north of 40°N.
  - Keep analyst-reviewed picks only. The weight/remark decoding is UNKNOWN — check.
- **Sample size.** At least 864 labels per sensor class (EH, HH/BH, HN), which gives ±2% at 90% accuracy.

**OBS labels**
- None exist from the networks.
- Hand-label 450 picks in each of 4 strata: 7D, X9, Z5 and non-Axial OO. That is 1,800 total.
  - Stratify by SNR tercile.
  - Allow "undecidable".
  - Two labellers overlap on 10% to measure inter-rater agreement.
  - n = 456 gives ±2% at 95% accuracy; n = 384 gives ±3% at 90%.
  - Effort: about 3–5 person-days (estimate).
- **Axial.** Ask Zhang for `A_wave_val.h5`, which is split by event and so held out at the template level. Use it as the M0 pipeline test.

**Event-level checks**
- 112 ComCat MTs.
  - Drop matches with dist_km > 15 or dt_s > 2 (a PI choice).
  - Predict P from the full tensor and drop picks with |A_P| < 0.2.
  - Watch for EH clipping at M5+.
  - These give 866 picks: a sign sanity check, not per-station evidence.
- About 575 NCSN FPFIT mechanisms west of 124°W (`s3://ncedc-pds/mechanism/`).
- 251 Morton et al. mechanisms, compared by Kagan angle.

**Metrics**, per network × sensor class × seafloor:
- accuracy and balanced accuracy
- confusion matrix including "undecidable"
- reliability diagram and ECE
- accuracy vs coverage
- accuracy binned by SNR, pick_prob, magnitude, distance and year
- McNemar's test between models
- end to end: SKHASH A–D fractions and median Kagan angle

**Proposed targets (PI to confirm)**
- Land: ≥95% accuracy at ≥50% coverage.
- OBS: ≥90% accuracy at ≥30% coverage, with the 95% CI lower bound ≥85%.
- A/B mechanisms vs MTs: median Kagan angle ≤35°.

A stratum that misses its target gets `pol_weight = 0` in the release. Fallbacks: fine-tune AxialPolCap on CI OBS labels together with Zhang, use CC-template polarities (Shelly et al. 2016), or drop the stratum.

## 7. S/P ratios, focal mechanisms, Track A

**S/P ratios**
1. Map event + station + phase to `pick_id`.
2. Read P and S rows from `amps_v6_all`.
3. Pivot on (event_ID, station).
4. Choose windows with `window_for`: P = min(1 + 0.03r, 15, 0.8·tSP); S = min(2 + 0.06r, 60).
5. Write the SKHASH ampfile with `amp_p`, `amp_s`, `noise_p`, `noise_s`, using `ratmin` 3 and `qbadfrac` 0.3.

Caveats:
- Amplitudes are the maximum over components, not Z for P and horizontal for S.
- The P window may include coda.
- Whether `wa_noise_mm` / `disp_noise_um` is pre-P only or shared by P and S: UNKNOWN — check in M1. If it is shared, take P noise from the polarity windows.
- Whether to use `disp_um` or `wa_mm` is decision D5; the proposal is `disp_um`.
- No station S/P corrections exist yet (FM_ML hardcodes its own). Fit them from residuals later.
- Run SKHASH with and without S/P and compare.

**SKHASH**
- SKHASH 1.1.5 is public domain (Skoumal et al. 2024).
- Starting parameters: `npolmin` 8, `nmc` 30, `min_polarity_weight` 0.1, FM_ML's `badfrac` 0.1, `max_agap` 90, `max_pgap` 60, `delmax` 120 km.
- Target set: the 22,261 well-recorded events. Post-QC count: UNKNOWN until M4.
- Axial cannot get single-event mechanisms. Use composites over CC clusters, as FM_ML does (decision D6).

**Track A dependency**
- 23% of v4 events locate above the surface, so takeoff angles from current locations are unreliable.
- Pass per-pick `takeoff`, `takeoff_uncertainty`, `azimuth` and `azimuth_uncertainty` from the NonLinLoc `ANGLES_YES` grids at A2 (NLLoc) or A3 (GrowClust) locations.
- Grid status: `time_vpvs173` is done (5,568 files) and `time_vpvs176` is running. A2 inputs are drafted but uncommitted, and A3 has not started.
- Optional provisional run on v4 locations: only events below the surface, ±20° takeoff uncertainty, QC only. Not released and not used for reversal QC.

## 8. Milestones

| M | Work | Compute / time (estimates) | Go/no-go gate |
|---|---|---|---|
| M0 | Move the 09_polarity scripts to a worktree. Check the torch port against Keras 3. Email Zhang (filter, response handling, license, `A_wave_val.h5`, `PolarCAP.h5`). Inventory reconciliation and FDSN fetch for the 118 stations. Env for SeisBench 0.13 / SeisPolarity. Contract YAML | CPU < 1 day; 2–3 days of work plus waiting on Zhang | Port matches Keras. On Zhang's validation windows, our pipeline is within ±2% of his reported accuracy. `sign_meta` covers ≥99% of P picks. License OK from Zhang, or the PI accepts proceeding without it |
| M1 | Pilot: one month in 2012 and one in 2014 (7D Year 4, Z5 GB), all networks, about 8k station-days; all models and filter variants | 8 shards, a few hours; inference minutes | ≥97% of picks ok. Alignment correct on 100 random plots. The sign rule reproduces the PB B024 comment. Noise-field question answered. Decimation-filter A/B done |
| M2 | Harvest PNSN and NCSN labels; OBS hand labels; bake-off; calibration | Harvest hours; labelling 3–5 person-days | Onshore model chosen by McNemar; per-stratum targets in §6 met or the fallback chosen |
| M3 | Full extraction, 300,084 station-days | Per-station-day throughput not measured; at 5–15 s that is roughly 9–26 h on 48 shards. Upper bound: the ver5 amps pass covered 60.9M picks in about 1.5 days. Run after Grid2Time finishes. About 4 GB of windows | ≥98% of picks with `window_status = ok`; `replan` the gaps (e.g. 7D J73A 2013-100 and X9 BB070 2014-100 returned no index rows) |
| M4 | Full inference; teleseismic station checks; plfile v0; push to the private Zenodo draft 23084158 | Inference 1–3 CPU-hours per model (not measured); station checks 1–2 days | Station-QC gate in §5; MT sign check ≥80% agreement |
| M5 | After A2/A3: SKHASH with NLL takeoffs and S/P; `pol_agree` iteration; plfile v1 | Hours | Median Kagan angle vs MTs and FPFIT within target; number of A/B Blanco mechanisms reported |
| M6 | Release the polarity parquet, plfile, mechanism catalog, README and MD5SUMS to draft 23084158 | < 1 day | PI sign-off; the draft stays private |

## 9. Decisions needed from the PI

1. **D1:** Onshore primary model: bake-off winner, or a fixed choice? Should we pursue the original PolarCAP weights?
2. **D2:** Accuracy and coverage targets per class (proposed in §6), and the MT match thresholds.
3. **D3:** HN fallback: keep accelerometer polarities as a separate stratum, or drop them?
4. **D4:** Hydrophones: excluded for good, or a later per-instrument check?
5. **D5:** S/P amplitude type (`disp_um` vs `wa_mm`). Is the maximum-over-components amplitude acceptable?
6. **D6:** Axial: composites, Zhang's catalog, or skip?
7. **D7:** Who hand-labels about 1,800 OBS picks? Fine-tune AxialPolCap with Zhang if it fails on CI OBS?
8. **D8:** Wait for A3, or release first mechanisms on A2? Run the provisional pass on v4 locations?
9. **D9:** Add a polarity column to the picks file (v4.1), or keep it a separate product?
10. **D10:** FM_ML license and authorship terms.

## 10. Risks

- **Silent sign errors** from the dip convention, the conflicting inventories, unflagged OBS flips and the 7D Year-4 archive. A whole flipped station biases mechanisms more than random error. Mitigation: §5, with raw and corrected signs stored separately.
- **Domain shift.** AxialPolCap is untested off Axial. Its reported test accuracy is inflated by augmented-copy leakage, and its agreement with CC on real data is 82%. CI OBS on sediment and on the shelf have emergent onsets, 5–12 Hz noise and water multiples.
- **Binary models have no unknown class,** so noise is forced to up or down. Mitigation: SNR, entropy and shift-agreement cuts, plus calibration.
- **Filter causality is unknown**, including in our own zero-phase decimation. Precursory ringing can flip the apparent first motion. Mitigation: the A/B tests in M0 and M1.
- **Pick-timing sensitivity** of the ±0.32 s window. Mitigation: the shift ensemble.
- **Label problems.** Leakage if SeisPolarity GLOBAL weights are used; automatic NCSN picks treated as truth; circularity in residual-based station QC.
- **Thin validation:** 866 MT picks and no OBS analyst labels.
- **Software maturity:** SeisBench 0.13.0 is 5 days old, EquiPol is unpublished, and the torch port is unverified.
- **Shared machine** and `/wd1` 80% full.
- **Takeoff dependence:** mechanisms made before Track A would need to be redone.

## 11. References

- Zhang et al., FM_ML / AxialPolCap (in prep., SRL): https://github.com/maochuan-zhang-UW/FM_ML. AGU 2025 DI43A-0017: https://ooifb.org/wp-content/uploads/2026/01/Zhang_AGU2025_lightning.pdf
- Chakraborty et al. 2022, PolarCAP, AI in Geosciences 3:46–52, https://doi.org/10.1016/j.aiig.2022.08.001
- Ross, Meier & Hauksson 2018, JGR 123:5120, https://doi.org/10.1029/2017JB015251
- Han, Kim & Sheen 2025, RPNet, SRL 96(4):2405, https://doi.org/10.1785/0220240384
- SeisBench 0.13 polarity models: https://github.com/seisbench/seisbench
- SeisPolarity (BSD-3): https://github.com/Chuan1937/SeisPolarity
- Zhao et al. 2023, DiTingMotion, https://doi.org/10.3389/feart.2023.1103914
- Messuti et al. 2023, CFM, https://doi.org/10.3389/feart.2023.1223686
- Chen et al. 2024, EQPolarity, https://doi.org/10.1109/TGRS.2024.3407060
- Hara et al. 2019, https://doi.org/10.1186/s40623-019-1111-x
- Uchide 2020, https://doi.org/10.1093/gji/ggaa401
- Skoumal, Hardebeck & Shearer 2024, SKHASH, https://doi.org/10.1785/0220230329; https://code.usgs.gov/esc/SKHASH
- Hardebeck & Shearer 2002, https://doi.org/10.1785/0120010200; 2003 (S/P), https://doi.org/10.1785/0120020236
- Hilmo & Wilcock 2020, https://doi.org/10.1029/2020GC009085
- Doran & Laske 2017, BSSA (CI OBS orientations and vertical reversals)
- LDEO 7D Year-4 polarity notice: http://mailman.whoi.edu/pipermail/obsictec/2016-October/000102.html
- MUSTANG polarity_check: https://service.iris.edu/mustang/metrics/docs/1/desc/polarity_check/
- WHOI OBSIP orientation cases: https://obsic.whoi.edu/wp-content/uploads/sites/6/2019/05/Appendix_F_Possible_Orientation_Cases_Yr2.pdf

---
Relevant paths:
- `/home/mdenolle/cascadia_obs_ensemble/workflow/09_polarity/axialpolcap_torch.py`
- `/home/mdenolle/cascadia_obs_ensemble/workflow/09_polarity/axial_test_extract.py`
- `/home/mdenolle/cascadia_obs_ensemble/workflow/05_magnitude/route_a_day_amplitudes.py`
- `/tmp/claude-39101/-home-mdenolle-cascadia-obs-ensemble/f3df6e58-5dbd-408f-a474-1341eca5ea78/scratchpad/polarity/FM_ML`
- `/tmp/claude-39101/-home-mdenolle-cascadia-obs-ensemble/f3df6e58-5dbd-408f-a474-1341eca5ea78/scratchpad/polarity/SKHASH`
- `/tmp/claude-39101/-home-mdenolle-cascadia-obs-ensemble/f3df6e58-5dbd-408f-a474-1341eca5ea78/scratchpad/polarity/v4_P.parquet`
- `/tmp/claude-39101/-home-mdenolle-cascadia-obs-ensemble/f3df6e58-5dbd-408f-a474-1341eca5ea78/scratchpad/polarity/v4_pick_sensor.parquet`
- `/tmp/claude-39101/-home-mdenolle-cascadia-obs-ensemble/f3df6e58-5dbd-408f-a474-1341eca5ea78/scratchpad/polarity/inv_v2_full.parquet`
- `/tmp/claude-39101/-home-mdenolle-cascadia-obs-ensemble/f3df6e58-5dbd-408f-a474-1341eca5ea78/scratchpad/polarity/inv_v5_slim.parquet`
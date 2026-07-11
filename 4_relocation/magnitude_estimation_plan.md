# Magnitude estimation plan (ML) — Cascadia OBS catalog

Tracking issue: [#10](https://github.com/Denolle-Lab/cascadia_obs_ensemble/issues/10).
This document is the authoritative, revised plan after review (see §"Review response").

## Goal

Produce a calibrated **local magnitude (ML)** for the relocated Cascadia catalog from the
per-pick amplitudes + relocated hypocenters, with **station corrections solved inside the
inversion** and absolute scale tied to **ComCat calibration events**.

Target scale is **ML** (per PI). We do **not** compute Md, and we do **not** map Md→ML
(see review response for why that mapping is unreliable here).

## Review response (what changed and why)

A reviewer flagged the one genuinely unsafe step in the first draft: using **raw-count**
amplitudes for **absolute ML**.

- **Correct critique.** The counts→ground-motion mapping is *frequency dependent* — the
  instrument response is a filter, not a scalar gain. Event corner frequency scales with
  magnitude, and site/path reshape the spectrum differently at each station (soft marine
  sediment "mud piles" jiggle more, prolong codas, and amplify amplitudes nonlinearly vs
  hard rock). A single **scalar** per-station term therefore cannot represent a
  frequency-dependent operator acting on an event-dependent spectrum. Raw counts are **not**
  a valid input for *absolute* ML across a heterogeneous network.
- **Where counts remain valid.** Within a **spatially clustered** set of events, the
  instrument response + site + path are common-mode at a given station, so log-count-amplitude
  *differences* equal **relative** magnitude differences directly (response-free). This is the
  reviewer's recommended route and we adopt it as a complement/cross-check.
- **Consequences for the plan:**
  1. **Drop** raw-counts→absolute-ML.
  2. **Route A (primary):** remove instrument response → Wood-Anderson (mm) → ML. Station
     terms now capture **site/path only** (response already removed).
  3. **Route B (complement, response-free):** counts → **relative** magnitudes within
     spatial clusters, **anchored** to absolute ML by co-located ComCat events; also the
     fallback where OBS responses are missing/unreliable, and an independent check on Route A.
  4. **Drop Md** and any Md↔ML mapping.
  5. Station corrections stay in the inversion and may need to be **magnitude/frequency
     dependent** (not scalar) for soft-sediment OBS sites — test and upgrade if residuals demand.

## What we have

| Item | Location | Notes |
|---|---|---|
| Relocated events | `data/Cascadia_relocated_catalog_ver_3.csv` | lat/lon/depth/origin/RMS/gap/Num P/Num S/**Event ID**; no magnitude. |
| Amplitudes **A** — peak *counts*, per pick, phase-specific | `4_relocation/calculate_amplitudes.py` → `…_w_amp.csv` (remote `/wd1`) | max\|data\| in `[t_pick−0.5, t_pick+2]`, HP 2 Hz, 100 Hz, max over available comps. Used for **Route B** only. |
| Amplitudes **B** — Wood-Anderson *displacement mm*, per station | `3_post_processing/event_waveform_processing.py`, `get_waveform_amplitude.py` → `data/split_files/…_with_amplitudes_part*.csv` | remove_response→WA simulate, `[origin, origin+120]`, Z/N/E max/min/duration. **Route A input — must be QC'd** (older run gave nonphysical ~1e9; split-file run ~0.2 mm). |
| Calibration mags | ANSS/ComCat via `concat_anss_catalogs_2010_2015.ipynb`; Morton 2023 `data/ds01.csv` (Md, reference only) | Anchor with **ComCat ML** where available. |
| Catalog matching | `utils/qc_utils.py` `match_events` / `filter_and_match_events` | repair Morton matcher (no acceptance threshold — see audit). |
| SNR | `utils/qc_utils.py` `calc_snr` | vertical-only, gap-interpolates, div-by-zero guarded in PR #8; adapt to the amplitude band. |
| Skipped picks | `calculate_amplitudes_skipped_picks.csv` + `examine_calculate_amplitudes_skipped_picks*.ipynb` | **15.5% of picks (155k/1.00M) have no amplitude.** |

Networks by pick count: PB 269k, **7D 242k (CI OBS)**, UW 154k, NC 133k, CN 81k, Z5 52k,
TA 35k, BK 22k, X9 7.7k, UO 2.7k, C8, NV, OO, 7A. 441 stations. Phase Type: 0 = P, 1 = S.

## Method

### Route A (primary) — response-removed Wood-Anderson ML

For event *i*, station *j* (phase-specific where used):

```
ML_ij = log10(A_WA,ij) + D(r_ij) + S_j + C
```

- `A_WA` — Wood-Anderson displacement (mm) after **response removal + WA simulation**.
- `D(r)` — distance/attenuation correction (Hutton–Boore form
  `D(r) = n·log10(r/r_ref) + k·(r − r_ref) + const`), adopted from a PNW prior then re-fit.
- `S_j` — station **site/path** correction, solved in the inversion (Σ S_j = 0 gauge).
- `C` — constant fixed by **ComCat ML** calibration events.
- Solve `{ML_i, S_j, n, k, C}` jointly by robust (IRLS/Huber) weighted least squares;
  bootstrap for per-event and per-station uncertainties.
- Standard ML convention: horizontal components. Handle **Z-only EH** stations separately.

### Route B (complement, response-free) — relative-cluster counts, anchored

- Define spatial clusters (reuse GraphDD / cross-correlation cluster structure).
- Within a cluster, for events a,b at station j: `ΔML_ab = Δlog10(A_counts) − ΔD(r)` ≈
  `Δlog10(A_counts)` for small clusters (path/site/response cancel). Solve **relative** event
  magnitudes per cluster from count amplitudes across shared stations.
- **Anchor** each cluster's absolute level with co-located **ComCat ML** events; stitch clusters
  via stations shared across clusters.
- Use where OBS responses are unavailable/unreliable and as an **independent cross-check** on
  Route A (disagreement diagnoses response errors).

### Station corrections (both routes)

Solved in the inversion. Baseline is a scalar `S_j`; **test residuals vs magnitude and frequency**
and, for soft-sediment / OBS sites, upgrade to a magnitude- and/or frequency-dependent term if
the data require it (the reviewer's nonlinear-site point).

## Plan (phased)

### Phase 0 — Data readiness & amplitude QC
- [ ] Assemble station **inventory** (coords + response + **operational epochs**) for all 14
      networks from FDSN/pnwstore (`get_stations(level="response")`).
- [ ] **Validate responses**, especially OBS (7D/X9/Z5/OO): which stations have a usable response
      for Route A vs must fall back to Route B. Re-QC product B (fix the ~1e9 run).
- [ ] **SNR gate**: noise amplitude in a pre-P window (`[P−10, P−1] s`) with the *same* band as the
      amplitude; require `A_signal/A_noise ≥ ~3–5` to enter the fit; below → drop / upper bound.
- [ ] **Response-stability / redeployment**: key station terms by **(station, deployment epoch)**
      for CI OBS; per-station for stable land nets.
- [ ] Quantify missing amplitudes (15.5%) and **clipping/saturation** at near-source stations.

### Phase 1 — Amplitude–distance dataset
- [ ] Per QC-passing (event, station, phase): hypocentral distance, `log10(A_WA)` (Route A) and
      `log10(A_counts)` (Route B), SNR, component, epoch.
- [ ] Plot `log10(A)` vs distance (per phase/route; color by event) — attenuation & outliers.

### Phase 2 — Calibration & clusters
- [ ] Match relocated events to **ComCat ML** (repair `match_events`); report N, mag range/type,
      spatial/depth coverage. Morton Md for reference only.
- [ ] Build spatial clusters for Route B (from CC/GraphDD).

### Phase 3 — Inversion
- [ ] Route A: joint robust LSQ for `{ML_i, S_j, n, k, C}` with gauge + ComCat anchors.
- [ ] Route B: per-cluster relative magnitudes anchored to ComCat; stitch via shared stations.
- [ ] Bootstrap uncertainties.

### Phase 4 — Validation
- [ ] Route A vs Route B agreement; both vs ComCat ML (1:1, residuals vs mag/distance/depth/network/epoch).
- [ ] Station corrections vs site type (OBS/sediment vs land/rock).
- [ ] Gutenberg–Richter b-value & Mc sanity; leave-out cross-validation on calibration events.

### Phase 5 — Productionize
- [ ] Write `ML` (+uncertainty) to the catalog; publish station-correction table (value, ±, N_obs,
      epoch); document scale definition, calibration basis, and limitations.

## Risks

- **Route A depends on response availability/quality** (OBS the hardest) → Route B is the fallback.
- **CI OBS yearly redeployment** breaks single-station response/site assumptions → epoch-keyed terms.
- **SNR / noise-dominated windows** at OBS (no SNR attached today).
- **2.5 s amplitude window anchored at the pick, not the peak S** → distance-dependent bias; `D(r)`
  absorbs some but not all → documented limitation (re-measuring amplitudes is out of scope).
- **Component inconsistency** / Z-only EH stations.
- **Soft-sediment nonlinear site response** may require magnitude/frequency-dependent station terms.
- **Limited ComCat calibration range** → weak absolute scale / attenuation → extrapolation risk.

## References
- Richter (1935); Hutton & Boore (1987) ML distance correction; Uhrhammer & Collins (1990)
  Wood-Anderson constants; Morton et al. (2023, JGR) Cascadia offshore catalog.

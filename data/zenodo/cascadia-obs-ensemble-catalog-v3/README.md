# An ensemble deep-learning earthquake catalog for offshore Cascadia (2010–2015)

This record contains the earthquake catalogs, phase picks, and amplitude/magnitude
tables produced by an ensemble deep-learning cataloging pipeline applied to the
Cascadia Initiative ocean-bottom-seismometer (OBS) deployment and adjacent land
stations, 2010–2015. It accompanies the manuscript *(Offshore Cascadia Ensemble
Catalog; in preparation)* and the code at
<https://github.com/Denolle-Lab/cascadia_obs_ensemble>.

License: **CC-BY-4.0**. Please cite both this dataset (DOI, once minted) and the
paper.

## Pipeline and provenance

Continuous waveforms → **ELEP** ensemble phase picking (multiple deep pickers combined)
→ **GENIE** graph-neural-network phase association → **GraphDD** double-difference
relocation refined with waveform **cross-correlation** differential times → quality
control. Amplitudes were subsequently remeasured from the picks and used to estimate
local magnitudes. Naming conventions in the original files: `reloc` = GraphDD-relocated,
`cog` = center-of-gravity cluster step, `cc` = cross-correlation-refined, `nass` =
number of associated phases, `p_4_s_4_rms_2_5` = the QC filter (≥4 P picks, ≥4 S picks,
RMS < 2.5 s).

The data form **three nested datasets**, each a strict subset of the previous stage
(fewer samples as evidence requirements tighten):

| Stage | Product | Rows |
|------|---------|------|
| (a) raw picks | ELEP ensemble picks | 39,597,551 |
| (b) associated | GENIE picks assigned to events / events | 1,086,007 / 116,591 |
| (c) relocated | relocated origins / picks | 63,887 / 1,004,335 |
| final | QC catalog (paper figures) | 31,020 |

## Contents

```
01_raw_elep_picks/
  elep_picks_all_regions_2010_2015.csv    (a) 39.6M ensemble picks, all regions
02_associated_genie/
  events.csv                              (b) 116,591 associated events
  pick_assignments.csv                    (b) 1,086,007 picks assigned to events
03_relocated/
  catalog.csv                             (c) 63,887 relocated origins
  picks.csv                               (c) 1,004,335 picks used in relocation
  origins_reloc_cog.csv                   relocated origins (cog step)
  origins_reloc_cog_cc.csv                + cross-correlation refinement
04_final_catalog_qc/
  events_qc_p4_s4_rms2.5.csv              FINAL catalog: 31,020 events (paper figures)
  arrivals.csv                            phase arrivals for the relocated set
  associations.csv                        pick-origin associations (+ station geometry)
  stations.csv                            454 stations (OBS + land)
05_amplitudes/
  picks_with_amplitudes.csv               picks with remeasured amplitudes (magnitudes)
06_comparison/
  anss_2010-2015.csv                      ANSS/USGS ComCat reference (58,307 events)
  morton_reloc.csv                        Morton et al. comparison catalog (63,887)
CHECKSUMS.md5                             md5 of every file above
```

## File / column notes

- **elep_picks_all_regions_2010_2015.csv** — one row per pick: `network, station,
  location, band_inst, label` (P/S), `pick_time, trigger_onset/offset, max_prob,
  thresh_prob, pick_id, station_id`.
- **events.csv** (GENIE) — `time, latitude, longitude, depth, x/y/z` (local
  coordinates), `picks` (count).
- **pick_assignments.csv** — pick↔event links: `event_idx, pick_idx, arid, station,
  phase, time_pick, residual, max_prob, vmodel, delta`.
- **catalog.csv** / **picks.csv** (relocated) — human-readable headers: `Latitude,
  Longitude, Depth (km), Origin Time (UTC)`, uncertainties, `Num. P, Num. S, RMS`;
  picks as `Pick Time (UTC), Station Name, Phase Type, Residual (s), Event ID`.
- **events_qc_p4_s4_rms2.5.csv** — the catalog shown in the paper figures: `lat, lon,
  depth, time, orid, nass, p_picks, s_picks, rms, nsphz, gap, algorithm`. QC =
  ≥4 P **and** ≥4 S picks **and** RMS < 2.5 s.
- **arrivals.csv** / **associations.csv** — `arid, orid, sta, phase, prob, timeres`
  and station geometry (`slatitude, slongitude, selevation, delta, esaz, seaz`).
- **picks_with_amplitudes.csv** — the relocation picks with remeasured amplitude
  columns, used for local-magnitude estimation.
- **anss_2010-2015.csv** — native USGS ComCat schema (`time, latitude, longitude,
  depth, mag, magType, …`); regenerable via `utils/fetch_anss_catalog.py` in the repo.
- **morton_reloc.csv** — our catalog matched to Morton et al., with match distance/time
  (`dist, dt, NonDimDist, id_Morton`).

## Reproducing / verifying

- Verify integrity: `md5sum -c CHECKSUMS.md5`.
- The repo can rebuild the relocated catalog + picks from chunked CSVs
  (`utils/reconstruct_split_csvs.py`) and regenerate the ANSS comparison
  (`utils/fetch_anss_catalog.py`); see the repo README.

## Authors

See the manuscript author list and the repository `README.md` for software vs.
manuscript authorship and individual contributions.

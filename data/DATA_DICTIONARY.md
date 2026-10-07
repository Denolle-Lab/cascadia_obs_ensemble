# Data dictionary: Cascadia OBS ensemble catalog v3

Three tables follow the ANSS / QuakeML split of a catalog into origins, arrivals
and picks. `origins_v3.csv` starts with the 22 columns of the ANSS ComCat CSV
format, in order, so it loads in any tool that reads ComCat CSV. Project columns
come after them. `arrivals_v3.csv` links an origin (`origin_id` = `origins.id`) to
a pick (`pick_id` = `picks.pick_id`).

All three are written by `utils/build_anss_tables.py`. Times are UTC, ISO 8601,
ending in `Z`. An empty field means no value.

| File | Rows | QuakeML element |
|---|---|---|
| `origins_v3.csv` | 63,887 | Event / Origin / Magnitude |
| `arrivals_v3.csv` | 1,004,335 | Arrival (+ Amplitude) |
| `picks_v3.csv.gz` | 39,597,551 | Pick |

## origins_v3.csv

One row per relocated event (GraphDD with waveform cross-correlation). Use
`qc_pass` to select the 31,020-event quality-controlled subset used in the paper
figures.

| Column | Unit | Definition | Source |
|---|---|---|---|
| `time` | UTC | origin time (ms) | GraphDD + CC (`origin_..._ver3_cc.csv`) |
| `latitude`, `longitude` | deg | epicenter (WGS84) | same |
| `depth` | km | hypocenter depth (GraphDD output; datum **TO CONFIRM**) | same |
| `mag` | | preferred magnitude `M` (see `M_type`); empty for the 8,180 events without one | phase19 |
| `magType` | | `ml`, or `mw` when `M_type` is `Mw` or `Mw_cal` | phase19 |
| `nst` | | number of distinct stations with an arrival | arrivals |
| `gap` | deg | largest azimuthal gap between stations | origin table |
| `dmin` | deg | epicentral distance to the nearest station with an arrival | arrivals (`delta`/111.19) |
| `rms` | s | RMS travel-time residual | origin table |
| `net` | | catalog code `cobs` (a project code, not an ANSS network code) | |
| `id` | | event id, `cobs` + zero-padded `orid` (e.g. `cobs000053`) | |
| `updated` | UTC | time the table was built | |
| `place` | | empty | |
| `type` | | `earthquake` (see `event_class` for the tectonic class) | |
| `horizontalError` | km | horizontal location uncertainty | phase10 (`h_unc_km`) |
| `depthError` | km | empty (not estimated) | |
| `magError` | | uncertainty of `mag` (`M_unc`) | phase19 |
| `magNst` | | number of station magnitudes averaged into ML (`n_ML`); empty for `Mw` and for `ML_method` `mapped` | phase19 |
| `status` | | `automatic` | |
| `locationSource` | | `cobs` | |
| `magSource` | | `cobs` for ML and Mw_cal, `comcat` for a ComCat moment-tensor Mw | phase19 |
| `orid` | | origin id in the pipeline tables (the key used in the repository scripts) | |
| `qc_pass` | bool | at least 5 P picks, at least 5 S picks, RMS < 2.5 s (31,020 events) | QC notebook |
| `M_type` | | `ML` (Hutton-Boore ML with station terms), `Mw` (matched ComCat moment tensor), `Mw_cal` (ML calibrated to Mw, above M 4.5) | phase19 |
| `ML`, `ML_unc` | | local magnitude and its uncertainty | phase16 / phase19 |
| `ML_method` | | `near`: at least 3 retained station magnitudes within 150 km (Hutton-Boore ML with station terms); `mapped`: fewer, so the phase3 inversion ML is mapped onto the near scale by a Theil-Sen fit (not calibrated to Mw) | phase19 |
| `Mw_mt` | | moment magnitude of the matched ComCat moment tensor | phase18 |
| `comcat_id` | | ComCat id of the matched moment-tensor event | phase18 |
| `n_arrivals` | | number of arrivals (P + S) | origin table (`nass`) |
| `n_p`, `n_s` | | number of P and of S arrivals | origin table |
| `event_class` | | `oceanic`, `megathrust?`, `intraslab`, `crustal-fault`, `volcanic` | phase10 |

## arrivals_v3.csv

One row per pick associated with a relocated event.

| Column | Unit | Definition | Source |
|---|---|---|---|
| `origin_id` | | `origins.id` | |
| `arid` | | arrival id in the pipeline tables | arrival / assoc tables |
| `pick_id` | | `picks.pick_id`; the nearest ELEP pick at the same station within 5 ms (same phase preferred). Empty for 79 arrivals with no pick within 5 ms | `build_anss_tables.py` |
| `network`, `station` | | SEED codes | |
| `location` | | SEED location code of the pick | picks |
| `channel` | | band and instrument code of the picked channels (`HH`, `BH`, `EH`); empty for picks of the first picker run (`picks.picker_run` = `v1`) | picks |
| `phase` | | phase assigned by the association (GENIE). It differs from `phase_hint` for 29,757 arrivals (about 3%) | arrival table |
| `phase_hint` | | phase given by the picker | picks |
| `time` | UTC | arrival time (0.1 ms) | arrival table |
| `time_residual` | s | observed minus predicted travel time | assoc table |
| `distance` | deg | epicentral distance | assoc (`delta`/111.19) |
| `distance_km` | km | epicentral distance | assoc (`delta`) |
| `azimuth` | deg | event-to-station azimuth | assoc (`esaz`) |
| `backazimuth` | deg | station-to-event azimuth | assoc (`seaz`) |
| `station_latitude`, `station_longitude` | deg | station location | assoc |
| `station_elevation` | m | station elevation (negative below sea level) | assoc |
| `assoc_prob` | | `prob` column of the arrival table (association weight; **TO CONFIRM** its definition with the GENIE/GraphDD authors) | arrival table |
| `wa_amp_mm` | mm | peak Wood-Anderson amplitude (response removed, WA filter) | Route A (`raw_wa_amplitudes_v2.csv`) |
| `disp_amp_um` | µm | peak ground displacement, 0.05-2 Hz | Route A |
| `amp_snr` | | peak over RMS of a 10 s pre-signal window | Route A |
| `amp_n_comp` | | number of components in the amplitude | Route A |
| `amp_station_epoch` | date | start of the station epoch (instrument response) used | Route A |
| `amp_sensor` | | location and band of the sensor measured (`.EH` = empty location, EH) | Route A |
| `amp_status` | | `ok`, or why no amplitude was measured | Route A |
| `amp_in_ml` | bool | the amplitude entered the ML inversion (649,427 picks) | `amp_distance_dataset_routeA.csv` |

## picks_v3.csv.gz

Every ELEP pick, associated or not.

| Column | Unit | Definition |
|---|---|---|
| `pick_id` | | pick id (0-based, as in `all_picks_all_regions_2010_2015_ver3.csv`) |
| `network`, `station`, `location` | | SEED codes |
| `channel` | | band and instrument code (`HH`, `BH`, `EH`); empty for `v1` picks |
| `phase_hint` | | `P` or `S` |
| `time` | UTC | pick time: maximum of the ensemble semblance within the trigger |
| `onset_time`, `offset_time` | UTC | trigger on and off times (`v2` only) |
| `probability` | | maximum ensemble semblance within the trigger (`v2` only) |
| `threshold` | | trigger threshold (0.05; `v2` only) |
| `method` | | `ELEP`: semblance of 5 EQTransformer models (`original`, `ethz`, `instance`, `scedc`, `stead`) |
| `picker_run` | | `v1`: first run, 2011-2015, HH or BH at the native sampling rate, trigger off at half the threshold, channel and probability not kept (21.9 M picks). `v2`: the region runs, 2010-2015, resampled to 100 Hz (17.7 M picks). See `workflow/01_picking/legacy/README.md` |
| `evaluation_mode` | | `automatic` |

## Other files in the record

| File | Definition | Source |
|---|---|---|
| `magnitude/station_terms_routeA.csv` | station-phase terms `C` (log10 units) per station epoch (`NET.STA@epoch`) and phase, with the number of observations. It holds 1,014 terms; the paper cites 809 (**TO CONFIRM** which selection) | phase3 (`route_b_station_terms_routeA.csv`) |
| `magnitude/comcat_mt_matched.csv` | ComCat moment tensors matched to catalog events (time, distance and depth offsets, strike/dip/rake) | phase18 |
| `comparison/anss_2010-2015.csv` | ANSS ComCat events in the study region, 2010-2015 | `utils/fetch_anss_catalog.py` |
| `comparison/morton_reloc.csv` | our catalog as relocated before the cross-correlation step, matched to Morton et al. (2023), with the match offsets `dist`, `dt`, `NonDimDist` and `id_Morton` | `origin_2010_2015_reloc_cog_morton_ver3.csv` |

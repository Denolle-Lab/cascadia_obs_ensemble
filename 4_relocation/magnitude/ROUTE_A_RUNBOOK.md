# Route A runbook — Wood-Anderson magnitudes (response-removed)

Route A re-measures amplitudes as **Wood-Anderson displacement** (instrument response
removed) to address the main seismological caveats of the counts-based Method B:

| Method-B caveat | Route-A fix |
|---|---|
| counts + scalar station term can't represent a frequency-dependent response | remove response → WA displacement; station term becomes ~pure site |
| one station term per reused OBS code across yearly redeployments | per-**epoch** station terms (inventory epochs → station id `NET.STA@epoch`) |
| fixed 2.5 s window misses the delayed S/Lg peak | **distance-scaled** window per phase |
| no SNR gate (noisy OBS) | pre-signal **SNR** measured and gated |
| Wood-Anderson product missing NC/BK (old IRIS-only run) | NC/BK fetched via **NCEDC** (`utils/data_client.py`) |

Decisions in force: **Wood-Anderson** amplitude; measured for **both P and S**;
**distance-scaled** window; **all components, vertical as fallback**.

## Where to run
A UW-internal host with pnwstore + FDSN/NCEDC access:
```
pixi install --environment internal      # adds pnwstore
git pull                                 # get this branch
```
Step 1 is one network request per pick (~10^6 picks) — slow. It appends on the fly,
supports `--start-index` (resume / shard), and should be validated on a small
`--limit` first. Inputs are the pick-assignment CSV used by Method B
(`Cascadia_updated_catalog_picks_assignment_ver_3.csv`, which already carries event
and station coordinates).

## Steps

**0. Station inventory** (coords + response + epochs; any machine with internet):
```
python route_a_build_station_inventory.py \
    --picks /wd1/.../Cascadia_updated_catalog_picks_assignment_ver_3.csv \
    --out-xml station_inventory.xml --out-csv station_epochs.csv
```
Check the reported count of "stations with multiple epochs" — those are the OBS
redeployments the epoch-keyed station terms will separate.

**1. Wood-Anderson amplitudes** (pnwstore host; validate small first):
```
# smoke test on 500 picks — inspect wa_amp_mm (expect ~1e-4..10 mm) and snr
python route_a_wa_amplitudes.py --picks <picks.csv> --inventory station_inventory.xml \
    --out raw_wa_amplitudes.csv --source pnwstore --limit 500
# full run (optionally shard by --start-index across processes/hosts)
python route_a_wa_amplitudes.py --picks <picks.csv> --inventory station_inventory.xml \
    --out raw_wa_amplitudes.csv --source pnwstore
```
Output: one row per pick with `wa_amp_mm`, `snr`, `n_comp`, `epoch`, `dist_hypo_km`,
event/station coordinates, and a `reason` field.

**For the full ELEP detection set (~40M picks)** a single CSV is multi-GB — pass
`--chunk-rows` to rotate the output into numbered parts (`raw_wa_amplitudes_partNNN.csv`,
~a few 100k–1M rows each). Parts are numbered by absolute row index, so `--start-index`
shards/resumes land in stable, non-colliding parts, and step 2 reads them via a glob:
```
python route_a_wa_amplitudes.py --picks <picks.csv> --inventory station_inventory.xml \
    --out raw_wa_amplitudes.csv --source pnwstore --chunk-rows 1000000
```

**2. Build the analysis dataset** (SNR gate + epoch-keyed station ids; any machine):
```
python route_a_build_dataset.py --raw raw_wa_amplitudes.csv \
    --out ../../data/magnitude/amp_distance_dataset_routeA.csv --min-snr 3 --epoch-station
# chunked run: pass a glob (quote it) — parts are filtered individually then combined
python route_a_build_dataset.py --raw 'raw_wa_amplitudes_part*.csv' \
    --out ../../data/magnitude/amp_distance_dataset_routeA.csv --min-snr 3 --epoch-station
```

**3. Inversion → absolute ML → QC** (reuse the existing pipeline, `--suffix _routeA`):
```
python phase3_route_b_relative_magnitude.py \
    --dataset ../../data/magnitude/amp_distance_dataset_routeA.csv \
    --outdir  ../../data/magnitude --fix-n 1.0 --suffix _routeA
python phase2_anchor_comcat_ml.py \
    --events  ../../data/magnitude/route_b_event_relative_mag_routeA.csv \
    --catalog ../../data/Cascadia_relocated_catalog_ver_3.csv \
    --outdir  ../../data/magnitude --suffix _routeA
python phase4_qc_and_gr.py --catalog ../../data/magnitude/cascadia_catalog_ML_routeA.csv --tag routeA
python phase5_pygmt_map.py --catalog ../../data/magnitude/cascadia_catalog_ML_routeA.csv \
    --out ../../data/magnitude/cascadia_ML_map_routeA.png
```
Because the epoch is folded into the station id, phase3's per-`(station, phase)`
terms are automatically per-deployment — no change to the inversion code.

## Validation (Route A vs Method B)
Join `cascadia_catalog_ML_routeA.csv` and `cascadia_catalog_ML_kpos.csv` on
`event_id` and compare ML. They should agree within the ~0.3-mag station scatter;
systematic offsets flag response/gain problems (e.g. a station whose counts-based
term was wrong because of a redeployment). Expect Route A to (a) fill NC/BK, (b)
tighten the station scatter, and (c) reduce the low-magnitude / offshore calibration
bias, since the amplitudes are now physical.

## Method / parameter notes
- **Wood-Anderson**: IASPEI constants (T0=0.8 s, h=0.7, static gain 2080), applied to
  ground velocity after `remove_response(output="VEL")`. The absolute gain is absorbed
  by the ComCat-ML calibration, so it does not affect final ML.
- **Window**: P `[t_P−0.3, t_P + min(1+0.03r, 15, 0.8·(t_S−t_P))]`;
  S `[t_S−0.3, t_S + min(2+0.06r, 60)]` s (r = hypocentral distance, km).
- **Components**: peak over all available components; vertical used when only a
  vertical channel exists.
- **SNR**: peak signal / RMS of a pre-signal noise window (default 10 s); gate at 3.
- **Epochs**: the response epoch containing the pick time tags each measurement, so a
  reused OBS station code deployed with different instruments gets separate station
  terms.

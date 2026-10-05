# ELEP picks v5: scope, cost and optimization plan

Drafted 2026-09-30. Goal: an offshore-complete pick table for 2010-2016, west of 122°W,
from the pnwstore archive, published as `all_picks_all_regions_2010_2016_ver5`.

## 1. Scope

Region 40-50°N, 131-122°W (nothing east of 122°W); 1 January 2010 to 31 December 2016.
Stations: every station of the pnwstore index in that box with a vertical `HH`, `BH` or
`EH` channel (`Z` or `3`), including the networks not picked so far (XD, CC, IU, US, XU,
XG, XN, ZH, YG, ZZ, YW, GS, 3F). Station-days in the pnwstore index
(`/wd1/PNWstore_sqlite/{year}.sqlite`, table `tsindex`):

| | station-days | in ver4 | not picked |
|---|---:|---:|---:|
| all stations | 586,034 | 283,192 | 302,842 |
| seafloor stations (elevation < 0) | 120,350 | 63,019 | **57,331** |

"Not picked" is an upper bound: ver4 records only station-days that produced picks.
Unpicked seafloor station-days by strip: 127-129°W 21,198 (7D G lines, X9, Z5, NV);
west of 129°W 18,568 (7D J lines, X9, Z5, OO Axial cabled array, NV); 123-127°W 17,565.
By year: 2010 1,419; 2011 3,332; 2012 8,172; 2013 16,349; 2014 11,176; 2015 10,649;
2016 6,234.

## 2. Where the time goes (measured, one station-day, 7D.J38A 2011-12-31)

| stage | current code, 1 thread | current, 4 threads | optimized, 1 thread |
|---|---:|---:|---:|
| pnwstore read | 0.2 s | 0.2 s | 0.2 s |
| preprocessing | 5 s | 4 s | 5 s |
| inference, 5 EQTransformer models | 135 s | 43 s | 53 s (batches of 16) |
| semblance | 192 s | 190 s | < 1 s (vectorized) |
| **total, core-seconds** | **~330** | **~360** | **~60** |

The pnwstore read is not a bottleneck. The semblance is: ELEP's `ensemble_semblance`
calls `scipy.ndimage.generic_filter` with a Python callback, 6000 samples × 2879 windows
× 2 phases × 2 sums. A moving sum with `uniform_filter1d` over all windows at once gives
the same values to float32 rounding (max relative difference 5e-7) and is 255 times
faster. Inference on the whole day in one batch (2879 windows) is 2.5 times slower than
in batches of 16 (cache use). bfloat16 on the AMX units is another 1.6 times faster
but moves the probabilities by up to 0.028 against a 0.05 threshold; not used.

## 3. Run time (64 physical cores, no GPU; 60 single-thread workers)

| scope | station-days | optimized code | current code |
|---|---:|---:|---:|
| A1. offshore gaps only | 57,331 | **~16 h** | ~3.7 days |
| A. all gaps (ver4 + new) | 302,842 | ~3.5 days | ~19 days |
| B. full uniform rerun | 586,034 | **~7 days** | ~37 days |

Add ~30% for failed reads, restarts and sharing the machine. Memory with chunked
windows: < 1 GB per worker (the current code holds 1.4 GB of predictions per
station-day).

**Recommendation: B.** With the optimized code a full rerun costs about a week, and it
gives one pick set made by one method at 100 Hz for all stations and years, instead of
ver4's mix of the first run (native sampling rate, different trigger, 55% of ver3) and
the region runs. GENIE and GraphDD have to be rerun on the whole table in either case.
If the machine time is short, run A1 first (offshore, one day) and the rest after.

## 4. Code changes, each checked against the current picker

1. **Vectorized semblance** in `elep_picker.py` (no ELEP change needed): all windows at
   once with `uniform_filter1d`. Check: same picks on the 14 v2 test station-days.
2. **Mini-batched inference** (16 windows) under `torch.inference_mode()`, models loaded
   once per worker (`load_models()`), 1 torch thread per worker. Check: same picks.
3. **Chunked windows**: build, predict and stack 256 windows at a time, so the day is never
   held as a (2, 5, 2879, 6000) array.
4. **Channel request**: ask pnwstore for `?H?` (plus `?DH`, the hydrophone, if the
   PickBlue model of §7 is kept), not `*` (J38A returns 48 traces, 3 used). pnwstore
   only knows the wildcards `?` and `*` (SQL LIKE), not `[HBE]`, so the band is chosen
   after reading.
5. **Driver** (`scale/run_picking_v5.sh` + `run_picking.py --tasks`): the task list comes
   from the pnwstore index (station-days that have data), not from an IRIS inventory per
   day; shards by station; resumable (skips files already written); a STATUS file and a
   DONE marker, as `run_route_a_rerun.sh` does; one output per station-year instead of
   per day, to avoid ~600,000 small files.
6. **v5 settings**: one `channel_mode` for all stations: HH, else BH, else EH; vertical
   `Z` or `3`; Z copied to the three inputs when a horizontal is missing (so EH-only
   stations are picked). This merges the four region modes; it is a choice, recorded
   in `picking_config.csv` as a `v5` row.

7. **Amplitudes at picking time** (§8): response removal and Wood-Anderson amplitudes on
   the day trace already in memory, written next to the picks.

Validation before the run: steps 1-4 on the 14 v2 test station-days (same picks);
step 6 on ~20 station-days per network, compared with ver4 where it exists.

## 5. Assembly and release

`utils/build_picks_v4.py` becomes `build_picks.py --version 5`: concatenate the
station-year files into `all_picks_all_regions_2010_2016_ver5.csv` (same columns as
ver3/ver4, `picker_run = v5`), with a manifest (station, days, picks, source files) and
MD5 sums. Compressed size estimate 1.5-2.5 GB.

## 6. Open items

- **Disk**: outputs go to `/wd1/mdenolle_data/picks_v5/` (3.5 TB free, writable since
  2026-10-01).
- Start after `rerun_hp05/DONE`, so the two runs do not share the machine.
- 2016: the CI deployment ended in 2015; 2016 adds land stations, OO and NV mostly.

## 7. Base models: what SeisBench offers (checked 2026-10-01, SeisBench 0.11.7)

v3/v4 used five EQTransformer weights: `original`, `ethz`, `instance`, `scedc`, `stead`.
All EQTransformer weights take 6000 samples at 100 Hz, so any of them fits the ELEP
ensemble without changing windows or the semblance.

| weights | trained on | input | norm | in v3 | fit to Cascadia OBS + coastal land |
|---|---|---|---|---|---|
| `original` | STEAD, global, mostly land, < 300 km (Mousavi et al. 2020), conservative | ZNE | std | yes | keep: reference model, continuity with v3 |
| `original_nonconservative` | same, tuned for recall | ZNE | std | no | optional: more picks, more false ones; the semblance may filter them |
| `stead` | STEAD (SeisBench retraining) | ZNE | peak | yes | keep |
| `instance` | Italy, dense land network | ZNE | peak | yes | keep |
| `scedc` | southern California | ZNE | peak | yes | keep |
| `ethz` | Switzerland, Alpine | ZNE | peak | yes | keep for continuity; least like the target |
| `iquique` | 2014 Iquique sequence, northern Chile (subduction, land stations) | ZNE | peak | no | possible: subduction setting, but a small training set |
| `lendb` | LenDB, local events, global stations | ZNE | peak | no | low priority |
| `geofon` | GEOFON, mostly regional and teleseismic | ZNE | peak | no | drop: aimed at distant events |
| `neic` | NEIC, mostly teleseismic | ZNE | peak | no | drop: aimed at distant events |
| `volpick` | volcano-tectonic and long-period events (Zhong & Tan 2024) | ZNE | peak | no | drop, except as a separate test on tremor or LP signals |
| **`obs`** (PickBlue) | OBS deployments, 3 seismometer components + hydrophone (Bornstein et al. 2024) | Z12H, 4 channels | std | no | **add for seafloor stations**: needs the hydrophone; BDH/HDH are in pnwstore for 7D, X9, Z5, OO, NV |
| **`obst2024`** (OBSTransformer) | OBS data, EQTransformer transfer-learned with automatic labels (Niksejel & Zhang 2024) | ZNE (Z12 on OBS) | std | no | **add for seafloor stations**: drop-in, 3 components |

`obst2024` is a separate SeisBench class (`OBSTransformer`), but it uses the same
EQTransformer architecture and input. PhaseNet has its own `obs` weights; mixing
architectures in one semblance would be a change to ELEP, so it is not proposed.

**TO VERIFY before choosing:** whether the PickBlue and OBSTransformer training sets
include Cascadia Initiative (7D) data. If they do, the models are in domain for us, but
we cannot use 7D picks to evaluate them.

**Options:**
- (a) the v3 set, unchanged: v5 is then directly comparable with v3/v4.
- (b) v3 set + `obst2024` on all stations (6 models).
- (c) v3 set + `obst2024` + `obs` on seafloor stations, v3 set on land.
  The ensemble then differs between station types, so the 0.05 threshold means
  different things on the two.

**Test before the run** (about one day on the machine). Take ~300 seafloor and ~100
land station-days, with as reference the PNSN/ANSS analyst picks and the GraphDD-assigned
v3 picks. Measure recall, precision and the P/S time residuals for (a), (b), (c) and each
model alone. Each extra model adds ~10 core-seconds per station-day (~17% of the
optimized cost).

## 8. Amplitudes at picking time

The picker already has the station-day in memory, so Route A can be measured there
instead of fetching ~10 times more data per pick afterwards. Removing the response once
per day trace costs a few seconds per station-day.

- Per station-day: `remove_response` (VEL, water level 60, pre-filter from 0.05 Hz)
  on the day trace, then WA simulation, with **both** 1.0 and 0.5 Hz high-pass (this
  settles the hp05 question in the same pass), and the 0.05-2 Hz displacement. The
  inventory epoch is taken from `route_a_build_station_inventory.py`, rebuilt for the
  v5 station list.
- **Window problem**: Route A scales the measurement window with source distance
  (`phase_window(phase, r_km)`), but the distance is not known before association. So
  for each pick and each amplitude type we store the running maximum at 1, 2, ..., 15 s
  after the pick (pre-window 0.3 s), plus the pre-pick noise RMS (10 s). After GraphDD,
  the Route A amplitude is the stored maximum at the window length given by the
  event-station distance, rounded up to the next second. Check this rounding against
  `route_a_wa_amplitudes.py` on the v3 picks.
- Output: one `amps_{net}.{sta}.{year}.parquet` per station-year, keyed by the pick, with
  the channel, the epoch id and a response-failure flag. Size ≈ 50 floats × ~80 M picks
  in float32 ≈ 15 GB uncompressed.
- What this does not replace: amplitudes on channels the picker did not use (e.g. an
  accelerometer at a station with a broadband sensor).

## 9. Dry run of options (a) and (b), 2026-10-01

`dryrun_tasks.py` → `dryrun_models.py` → `dryrun_score.py`; outputs in
`/wd1/mdenolle_data/v5_dryrun/`. 400 station-days: 250 seafloor with v3 arrivals (the 3,293
7D station-days in the OBSTransformer training set excluded), 50 seafloor never picked in v3,
100 land. 399 picked, 1 constant. Vectorized semblance: same picks as ELEP's on 10 of 10.

**7D is in the OBSTransformer training set** (SeisBench `OBST2024`): 5,603 earthquake traces
on 4,179 7D station-days of 2011-2015 from 204 stations, plus 6,670 noise windows from 2011.
X9, Z5, NV, OO, 7A and C8 are not.

| group | phase | picks/day a → b | recall of v3 arrivals a → b | recall of ANSS M≥2 a → b | ver4 picks reproduced by a |
|---|---|---|---|---|---|
| seafloor (249 sd) | P | 174 → 174 | 0.81 → 0.80 | 0.36 → 0.36 (n=69, all M) | 0.54 |
| | S | 117 → 227 | 0.85 → 0.90 | 0.53 → 0.56 | 0.52 |
| land (100 sd) | P | 71 → 84 | 0.98 → 0.96 | 0.91 → 0.91 (n=203, all M) | 0.93 |
| | S | 62 → 152 | 0.96 → 0.97 | 0.96 → 0.96 | 0.92 |
| never-picked seafloor (48 sd) | P | 140 → 142 | — | (3 events) | 1.00 |
| | S | 100 → 191 | — | | 1.00 |

- (b) roughly doubles the S picks and keeps the P count; it recovers slightly more of the
  associated v3 S arrivals (0.85 → 0.90 offshore) and slightly fewer P (0.81 → 0.80).
  Whether the extra S picks are earthquakes, the GENIE dry run has to tell.
- About a quarter of option (a)'s picks change in (b) (ver4 reproduced 0.93 → 0.68 on land P):
  a sixth model that disagrees lowers the semblance below 0.05. The associated v3 arrivals
  are barely affected, so these are mostly picks that were never associated.
- `obst2024` alone at 0.05 gives 3-7 times more picks than the ensembles: too permissive
  at this threshold on its own.
- Option (a) with the v5 channel rule reproduces 93% of ver4 on land and 100% on the
  "never picked" seafloor days (whose ver4 picks are from the region runs). Offshore it
  reproduces only ~53%, because most ver4 seafloor picks are from the first run (native
  sampling rate, different trigger).
- ANSS has few events offshore (69 on 249 seafloor station-days), so it constrains little
  there.

**Throughput, measured:** 30 workers, 6 models, 125 s per station-day per worker
(74 s with 2 workers: the workers compete for memory bandwidth). That is 0.24 station-days
per second, so the full rerun (586,034) would take **~29 days at 30 workers**, not the
7 days of §3. Before the run: a scaling test with 30/60/90 workers once `rerun_hp05` is
done, and the cost of a sixth model (+20% inference) weighed against it. bfloat16
(1.6× faster, §2) needs a check of the pick changes it causes against the 0.05 threshold.

Two station-days failed on traces of unequal length after the trim (fixed: cut to the
shortest).

## 10. Decision of 2026-10-01: fill the gaps with option (a), offshore first

A full rerun (~25-29 days at the measured rate) is postponed. The gaps are filled with
option (a), the method of the region runs, so the filled station-days match ver4 (93-100%
of its region-run picks reproduced, §9). Option (b) on the gaps only would give the new
station-days twice the S density of the old ones.

| gap | station-days | option (a), 30 workers |
|---|---:|---:|
| offshore 2010-2015 | 51,097 | ~2.1 days |
| offshore 2016 | 6,234 | ~0.3 days |
| land 2010-2015 | 178,100 (122,416 on networks already picked) | ~7.5 days |
| land 2016 | 67,411 | ~2.8 days |

Offshore run: `1_picking/run_fill.sh` → `/wd1/mdenolle_data/picks_v5_fill/` (57,331
station-days, `tasks_offshore.csv`, random order). It waits for `rerun_hp05/DONE`, runs
the scaling test (30/60/90 workers, 20 minutes each) as the start of the fill, then
finishes with the fastest count. Picks only: their Route A amplitudes come after association,
with `route_a_wa_amplitudes.py`, as for v3. The land gaps and 2016 are still to be decided.

## 11. Offshore fill and ver5 (2026-10-01 to 10-05)

- Scaling test, station-days per minute: 30 workers 19.6, 60 workers 23.1, 90 workers 9.1
  (90 ran out of memory). The fill ran at 60 workers from a frozen worktree at `40c30245`.
- At 60 workers the run sat at the memory limit (5-7 GB per worker). Out-of-memory kills on
  2026-10-03 lost 18 station-days, and `multiprocessing.Pool` then waited forever. They and
  the 22 station-days that had failed with memory errors were rerun at 10 workers on 10-05.
  For the land gaps, use at most 40 workers, or give the pool `maxtasksperchild` and a
  per-task timeout.
- Outcome of the 57,331 station-days: 51,764 processed (48,638 with picks), 5,559 flat
  data, 8 data errors (short or empty traces).
- `utils/build_picks_v5.py` → `/wd1/mdenolle_data/picks_v5/`:
  `all_picks_all_regions_2010_2016_ver5.csv.gz` = ver4 unchanged + 11,377,129 picks
  (7.01 M P, 4.37 M S) on 48,638 station-days at 212 stations (7A, 7D, NV, OO, X9, Z5),
  60,918,847 picks in all. Checked: one header, 15 columns on every row, pick_id equal to
  the row index. `ver5_added_picks.csv.gz` holds the new rows alone.
- Land gaps (2010-2015 and 2016): postponed (decision of 2026-10-05). Amplitudes for the new
  picks come after association, as for v3.

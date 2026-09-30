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
4. **Channel request**: ask pnwstore for `[HBE]H?` only, not `*` (J38A returns 48 traces,
   3 used).
5. **Driver** (`scale/run_picking_v5.sh` + `run_picking.py --tasks`): the task list comes
   from the pnwstore index (station-days that have data), not from an IRIS inventory per
   day; shards by station; resumable (skips files already written); a STATUS file and a
   DONE marker, as `run_route_a_rerun.sh` does; one output per station-year instead of
   per day, to avoid ~600,000 small files.
6. **v5 settings**: one `channel_mode` for all stations: HH, else BH, else EH; vertical
   `Z` or `3`; Z copied to the three inputs when a horizontal is missing (so EH-only
   stations are picked). This merges the four region modes; it is a choice, recorded
   in `picking_config.csv` as a `v5` row.

Validation before the run: steps 1-4 on the 14 v2 test station-days (same picks);
step 6 on ~20 station-days per network, compared with ver4 where it exists.

## 5. Assembly and release

`utils/build_picks_v4.py` becomes `build_picks.py --version 5`: concatenate the
station-year files into `all_picks_all_regions_2010_2016_ver5.csv` (same columns as
ver3/ver4, `picker_run = v5`), with a manifest (station, days, picks, source files) and
MD5 sums. Compressed size estimate 1.5-2.5 GB.

## 6. Open items

- **Disk**: the root disk has ~16 GB free and `/wd1` is not writable for this account.
  Outputs need a directory with ~20 GB (per-station-year files plus the table).
- Start after `rerun_hp05/DONE`, so the two runs do not share the machine.
- 2016: the CI deployment ended in 2015; 2016 adds land stations, OO and NV mostly.

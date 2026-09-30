# Legacy picking scripts (the ver3 run)

These are the scripts H. Bito ran to write the per-station-day ELEP picks that were
merged into `all_picks_all_regions_2010_2015_ver3.csv` (39,597,551 picks, 2025-03-05).
They are kept as the record of that run. Use `../run_picking.py` with
`../picking_config.csv` to rerun; it reproduces these outputs (see below).

They are also in git at tag `pre-cleanup-2026-09` (and the picking code as of the
ver3 run at `stage-picking-ver3`). As committed here they will not run from this
directory: the utils add `..` to `sys.path` to import `utils.data_client`, which now
resolves to `1_picking/`, not the repository root.

## Which script wrote which picks

The ver3 table joins two picker runs.

**v1, the first run (2011-2015; 21.9 M picks, 55% of ver3).**
`parallel_pick_{year}_HH_BH.py`, box 40-50°N, 127-123°W, 4 workers, output
`picks_{year}/STA_YYYYMMDD.csv` in an older, SeisBench-like format (no channel, no
probability). The code that ran is `picking_utils_2012.py` (`picking_utils_2015.py`
is identical; `picking_utils_123_127_HH_BH.py` is the same code with the later
`data_client` change). As committed, the `_HH_BH` scripts import `picking_utils`,
whose `run_detection` no longer takes `lat, lon, elev`, and
`parallel_pick_2013_HH_BH.py` writes to `picks_2012/` while the 2013 files are in
`picks_2013/`. In ver3 these picks are the rows with an empty `band_inst`.

**v2, the region runs (2010-2015; 17.7 M picks).** Four runs per year, output
`picks_{year}_{region}/NET_STA_YYYYMMDD_YYYYMMDD.csv`:

| Region | Box (lat, lon) | Script | Utils | Channel choice |
|---|---|---|---|---|
| `122-129` | 40-50, -129 to -122 | `parallel_pick_{year}.py` | `picking_utils.py` | HH, else BH, else EH; vertical `Z` |
| `123-127_EH` | 40-50, -127 to -123 | `parallel_pick_{year}_123_127_EH.py` | `picking_utils_prio_EH.py` | EH only; vertical `3` or `Z`; Z copied to all 3 inputs if a horizontal is missing |
| `122-123_46-50` | 45.8-50.2, -123.2 to -121.8 | `parallel_pick_{year}_122-123_46-50.py` | `picking_utils_prio.py` | as `122-129`, vertical `3` or `Z`; skips station-days already in `picks_{year}_122-129/` |
| `127-129_46-50` | 45.8-50.2, -129.2 to -126.8 | `parallel_pick_{year}_127-129_46-50.py` | `picking_utils_prio.py` | as `122-129`, vertical `3` or `Z` |

Both runs share the ELEP core: 5 EQTransformer models (`original`, `ethz`,
`instance`, `scedc`, `stead`), 4-15 Hz band-pass, 6000-sample windows with a
3000-sample step, 500-sample blinding, ensemble semblance, threshold 0.05. The
scripts also load `pnw` and `geofon` models, but `run_detection` does not use them.
v1 differs from v2 in ways that change the picks:

| | v1 | v2 |
|---|---|---|
| channels requested | `?H?` | all |
| channel choice | HH, else BH (EH never) | per region, above |
| vertical and constant-data checks | no | yes |
| sampling rate seen by the models | native (e.g. 40 or 50 Hz) | resampled to 100 Hz |
| component order | as returned | Z, E/1, N/2 |
| trigger | ELEP `picks_summary_simple` (off at half the threshold) | on and off at the threshold |
| pick time | start of the first trace before the trim + index × delta | from the trimmed trace |
| kept per pick | time only | band code, trigger on/off, probability |

In `../elep_picker.py`, v1 is `run_detection_v1` and v2 is `run_detection`;
`../picking_config.csv` has one row per run (29 rows).

## Check of the consolidated picker (2026-09-30)

`../run_picking.py` was run on station-days whose stored per-day files it should
reproduce:

- **v2**, 14 station-days of 2011 from the four region directories (HH, BH and EH
  stations; UW, PB, NV, 7D): all 14 give the same picks (count, times, labels); the
  probabilities differ by at most 3e-7 (float32 rounding across library versions).
- **v1**, 11 station-days of 2011-2015 from `picks_{year}/` (UW, CN, NC, 7D, TA, Z5):
  10 are byte-identical to the stored files; in the eleventh (Z5.GB101 2015-09-07), one
  of 98 picks moves by one sample (0.02 s at 50 Hz), a near-tie in the semblance
  maximum.

Environment: pixi `internal` (SeisBench 0.11.7, torch 2.13, ELEP as pinned in
`pixi.lock`) with pnwstore. Weights: the SeisBench EQTransformer cache of H. Bito
(`~hbito/.seisbench/models/v3/eqtransformer`).

The channel rule "HH, else BH, else EH" is the one that ran in v2: at the
`stage-picking-ver3` tag the rule read "HH only if BH and EH are also present",
which would have skipped OBS station-days that have no EH channel (e.g.
7D.J38A 2011-12-31), but ver3 holds their BH picks.

## Merge into ver3 (not in the repository)

The step that concatenated the per-day files into the ver3 table is not in the
repository. It converted the v1 files to the v2 columns (empty `band_inst`, trigger
times and probability). The ver3 table does not hold every v2 per-day file: in 2011, all files of
`122-129` and `123-127_EH` are in it, but about 17% of the `122-123_46-50` files
and 34% of the `127-129_46-50` files are not, as whole station-days (for example,
UW EH-only stations LO2, JCW, RER and MBW are absent from ver3 in 2011). The rule
used to drop them is **TO CONFIRM** with H. Bito.

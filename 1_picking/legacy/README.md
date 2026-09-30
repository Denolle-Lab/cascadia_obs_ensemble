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

The ver3 picks come from four region runs per year, 2010-2015 (24 runs). The
per-day outputs are in `/wd1/hbito_data/data/picks_{year}_{region}/`.

| Region | Box (lat, lon) | Script | Utils | Channel choice |
|---|---|---|---|---|
| `122-129` | 40-50, -129 to -122 | `parallel_pick_{year}.py` | `picking_utils.py` | HH, else BH, else EH; vertical `Z` |
| `123-127_EH` | 40-50, -127 to -123 | `parallel_pick_{year}_123_127_EH.py` | `picking_utils_prio_EH.py` | EH only; vertical `3` or `Z`; Z copied to all 3 inputs if a horizontal is missing |
| `122-123_46-50` | 45.8-50.2, -123.2 to -121.8 | `parallel_pick_{year}_122-123_46-50.py` | `picking_utils_prio.py` | as `122-129`, vertical `3` or `Z`; skips station-days already in `picks_{year}_122-129/` |
| `127-129_46-50` | 45.8-50.2, -129.2 to -126.8 | `parallel_pick_{year}_127-129_46-50.py` | `picking_utils_prio.py` | as `122-129`, vertical `3` or `Z` |

All four use the same ELEP core: 5 EQTransformer models (`original`, `ethz`,
`instance`, `scedc`, `stead`), 4-15 Hz band-pass, 100 Hz, 60 s windows with a 30 s
step, 5 s blinding, ensemble semblance, trigger threshold 0.05. The scripts also
load `pnw` and `geofon` models, but `run_detection` does not use them.

Not part of ver3:

- `parallel_pick_{year}_HH_BH.py` with `picking_utils_123_127_HH_BH.py`,
  `picking_utils_2012.py`, `picking_utils_2015.py`: an older run whose files have a
  different column layout, none of which appears in ver3. As committed, the
  `_HH_BH` scripts import `picking_utils` but pass `lat, lon, elev`, which only the
  older utils accept, and `parallel_pick_2013_HH_BH.py` writes to `picks_2012/`.

## Check of the consolidated picker (2026-09-30)

`run_picking.py` / `elep_picker.py` were run on 14 station-days of 2011 drawn from
the four region directories (HH, BH and EH stations; UW, PB, NV, 7D). All 14 give
the same picks as the stored files: same count, same pick times and labels; the
maximum probabilities differ by at most 3e-7 (float32 rounding across library
versions). Weights: SeisBench EQTransformer cache of H. Bito
(`~hbito/.seisbench/models/v3/eqtransformer`), SeisBench 0.11.7, torch 2.13.

The channel choice "HH, else BH, else EH" is the one that ran: at the
`stage-picking-ver3` tag the rule read "HH only if BH and EH are also present",
which would have skipped OBS station-days that have no EH channel (e.g.
7D.J38A 2011-12-31), but ver3 holds their BH picks.

## Merge into ver3 (not in the repository)

The step that concatenated the per-day files into the ver3 table is not in the
repository. The ver3 table does not hold every per-day file: in 2011, all files of
`122-129` and `123-127_EH` are in it, but about 17% of the `122-123_46-50` files
and 34% of the `127-129_46-50` files are not, as whole station-days (for example,
UW EH-only stations LO2, JCW, RER and MBW are absent from ver3 in 2011). The rule
used to drop them is **TO CONFIRM** with H. Bito.

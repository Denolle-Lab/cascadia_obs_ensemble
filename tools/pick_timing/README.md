# pick_timing

Find and correct timing errors in seismic phase picks. Two failure modes, two fixes:

| failure | headers | fix | find it with |
|---|---|---|---|
| **day-merge drift**: a picker merged a day of records and counted samples at the nominal rate on a channel whose real rate differs | right | `retime` (each pick timed from its own record header) | `scan-headers` |
| **station clock error**: free-running or wrongly set clock (drift until a restart, or an offset) | wrong | `detect-clock` -> person reviews -> `model-clock` -> `apply` | `detect-clock` |

How to tell them apart: [docs/failure_modes.md](docs/failure_modes.md). Method: [docs/method.md](docs/method.md).
Checks: [docs/validation.md](docs/validation.md). Agent instructions: `.claude/skills/pick-timing/SKILL.md` at the
repository root.

Sign convention: **c(t) = header time - true time**; positive = picks late. The correction added to a pick is
**-c(t)**.

## Install

Dependencies: Python >= 3.9, numpy, scipy, pandas >= 2.0, obspy >= 1.4, pyyaml (tested with Python 3.10,
numpy 1.26.4, scipy 1.15.2, pandas 2.3.3, obspy 1.5.0, pyyaml 6.0.3). Parquet tables need pyarrow (optional).

```bash
pip install -e tools/pick_timing            # from a checkout of this repository
# or, without installing (this repository's pixi default environment has every dependency):
export PYTHONPATH=$PWD/tools/pick_timing
python -m pick_timing --help
```

## Quick start on the synthetic example

```bash
cd /a/scratch/folder
python -m pick_timing make-example --out example_data         # 12-station network + a drifting MiniSEED day
CFG=<repo>/tools/pick_timing/configs/example_generic.yaml
python -m pick_timing scan-headers -c $CFG --stations DRFT.XX --start 2020-01-01 --end 2020-01-02 --tag drft
python -m pick_timing retime       -c $CFG --scan example_out/drft_channel_days.csv
python -m pick_timing detect-clock -c $CFG                     # screen + false-alarm runs
python -m pick_timing report       -c $CFG                     # example_out/report.html + review_candidates.csv
# a person fills decision/reviewer in example_out/review_candidates.csv (saved as review_filled.csv), then:
python -m pick_timing ingest-review -c $CFG --review example_out/review_filled.csv
python -m pick_timing model-clock   -c $CFG --epochs example_out/confirmed_epochs.csv
python -m pick_timing apply         -c $CFG --clock-model example_out/clock_model.json \
       --retime-corrections example_out/retime/corrections.csv.gz --output example_out/picks_corrected.csv
python -m pick_timing validate      -c $CFG --picks-in example_data/picks.csv \
       --picks-out example_out/picks_corrected.csv --clock-model example_out/clock_model.json
```
`example_data/truth_clocks.csv` and `truth_retime.csv` hold the injected errors (S03.XX drift 6 ms/day from day 150
to 240; S07.XX offset -0.25 s on days 300-340; DRFT.XX re-timing corrections up to 1.18 s).

## Commands

| command | reads | writes (in `output_dir`) |
|---|---|---|
| `scan-headers` | waveforms (headers only) | `<tag>_channel_days.csv` (drift metrics, timing quality), `<tag>_events.csv` (tears, gaps, overlaps = restarts) |
| `retime` | picks, waveforms, scan output or task list | `<tag>/chunks/` (resumable), `<tag>/corrections.csv.gz` |
| `extract-nlloc` | NonLinLoc `*.grid0.loc.hyp` | residual table + `stations.csv` |
| `detect-clock` | residuals (+ stations, origins, CC table, restarts) | `rd.parquet`, `screen_stations.csv` (every station), `candidates.csv`, `series/`, `false_alarm.json`, `false_alarm_runs.csv` |
| `report` | detect-clock output | `report.html` (self-contained), `review_candidates.csv` |
| `ingest-review` | filled review CSV | `confirmed_epochs.csv` (accepted rows only; needs a reviewer name) |
| `model-clock` | confirmed epochs | `clock_model.json` |
| `apply` | picks, clock model, re-timing corrections | corrected pick file + `.provenance.json` |
| `validate` | input and corrected picks, clock model (+ re-located residuals, CC) | `validate/` (byte identity, residual re-fit, co-located slopes, CC) |
| `relocate-nlloc` | NonLinLoc run folder, clock model | `relocate/` (re-located residuals of the affected events) |
| `make-example` | - | the synthetic example |

Every command takes `--config`, `--out` (overrides `output_dir`) and `--workers`.

## Inputs: the config

One YAML file maps your tables to the package's names; no code change for a new dataset. Start from
[configs/example_generic.yaml](configs/example_generic.yaml). Canonical columns:

| table | required | optional |
|---|---|---|
| picks | event_id, station (or network + station_code), phase, time | travel_time, timing_correction |
| residuals | event_id, station, phase, time, residual | weight, x_km/y_km (station), ev_x_km/ev_y_km/ev_z_km or ev_lat/ev_lon/ev_depth_km |
| stations | station | lat/lon or x_km/y_km |
| origins | event_id, time | lat, lon, depth |
| cc | event1, event2, station, phase, dt_cc, time1, time2 | pair, accepted, cc |

Waveform readers (`waveforms.reader`): `sds` (SDS tree), `fdsn` (ObsPy FDSN, raw MiniSEED kept), `sqlite_index`
(an mseedindex/tsindex sqlite archive with a path rewrite; pnwstore = per-year databases + year-to-disk table),
`routed` (per-network readers + fallback). [configs/cascadia_v6.yaml](configs/cascadia_v6.yaml) is this project's
config (UW server paths; not portable).

## Decisions that need a person

1. Which candidates are clock errors (review CSV `decision`); nothing is applied without `accept` and a reviewer.
2. Which size to use for an accepted epoch: CC (`size_source: cc`) when CC rows exist (residuals understate the
   size because locations absorb part of it), else residuals.
3. The end of a drift: the header restart (`t_end_override_utc`) when there is one.
4. Whether to download waveforms from a data centre (scan-headers / retime with an FDSN reader read whole days).

## Tests

```bash
cd tools/pick_timing
python -m unittest discover -s tests -v      # or: pytest tests
```
17 tests, about 15 s: drift / offset / step / reset recovered within tolerance; 40 clean synthetic stations not
flagged; re-timing of a synthetic day-merged trace to 0.1 ms; header scan of a restart; column-mapping round
trip; clock model; apply never drops a pick and keeps untouched rows byte-identical; review ingest.

## Provenance

Written in the Cascadia OBS ensemble project (catalog v6.0 pick file, decision D-B090), generalising the D-B072/
D-B075 pick re-timing and the CC-dt clock-epoch work (MH07.XD / MG07.XD, spring 2016).

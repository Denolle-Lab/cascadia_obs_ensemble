---
name: pick-timing
description: >-
  Finds and corrects timing errors in seismic phase picks with the pick_timing toolkit (ObsPy, MiniSEED record
  headers, NonLinLoc or other travel-time residuals, optional cross-correlation differential times). Covers two
  failure modes: day-merge sample-count drift of a picker that merged whole days (re-time each pick from its own
  record header) and station clock errors (free-running clock drift until a restart, or offset epochs) found in
  network-relative residuals, reviewed by a person and corrected in a new pick file with provenance. Use when the
  user asks to "check pick timing", "find station clock drift", "correct clock errors in picks", "re-time picks",
  "screen stations for timing problems", or mentions GPS outages, OBS clock skew, MiniSEED time tears or residual
  steps at one station.
---

# Correcting pick timing

The toolkit is a Python package with a command line: `python -m pick_timing <command> -c CONFIG.yaml`.
Every command reads a YAML config that maps the user's columns to the package's names, so pick files of other
projects work without code changes. Read the references before deciding anything:
`tools/pick_timing/docs/failure_modes.md` (the two failure modes and how to recognise each), `tools/pick_timing/docs/method.md`
(algorithms, defaults), `tools/pick_timing/docs/validation.md` (which check proves what), `tools/pick_timing/docs/config.md` (config keys,
input columns, outputs).

Sign convention, everywhere: **c(t) = header time - true time**; positive c = the station's picks are late and its
residuals positive. The correction added to a pick is **-c(t)** (`clock_correction_s`).

## 0. Get the code

In the cascadia_obs_ensemble repository the package is `tools/pick_timing`:

```bash
export PYTHONPATH=$PWD/tools/pick_timing      # from the repository root
python -m pick_timing --help
python -m unittest discover -s tools/pick_timing/tests    # 17 tests, ~15 s; all must pass
```
Dependencies: Python >= 3.9, numpy, scipy, pandas >= 2.0, obspy >= 1.4, pyyaml (the pixi default environment has
them; pytest is optional, the tests are unittest classes).

## Ask a person first

Stop and ask before:
- **downloading waveforms** from a data centre (`scan-headers` and `retime` with an `fdsn` reader read whole days:
  about 1-10 MB per channel-day). Say how many station-days and which data centre.
- **writing outside the project folder** or the configured `output_dir`.
- **applying any clock correction that a person has not confirmed.** `detect-clock` produces candidates, not
  corrections. Only rows a person marked `accept` (with their name in `reviewer`) in the review CSV become
  confirmed epochs. Never fill `decision` or `reviewer` yourself unless the person tells you which candidates to
  accept and under what name or decision reference; then record that reference in `decision_ref`.
- **overwriting an existing pick file, catalog or anything the user did not create in this session.** Write new
  files only; `apply` never edits its input.

## 1. Set up the config

Copy `tools/pick_timing/configs/example_generic.yaml` and edit it. Check with the user:
- the pick file and its columns (`event_id`, `station` or `network` + `station_code`, `phase`, `time`; optional
  `travel_time`, and `timing_correction` if the file was already re-timed); the station id pattern
  (`station_id: "{sta}.{net}"` or `"{net}.{sta}"`); time format (ISO UTC strings or POSIX seconds);
- a residual table: one row per pick with event_id, station, phase, time, residual (and weight, station
  coordinates). From NonLinLoc: `python -m pick_timing extract-nlloc -c CFG --hyp-glob 'RUN/*/loc.*.grid0.loc.hyp'
  --stations-glob 'RUN/*/last.stations' --output OUT/residuals.parquet` (later globs win per event, e.g. a deep
  pass). Use the residuals of the location run that used these picks;
- the waveform source, only if headers or re-timing are needed (reader `sds`, `fdsn`, `sqlite_index`, `routed`);
- a CC differential-time table if one exists (refines the size of a clock error).
Check: `python -m pick_timing detect-clock -c CFG --stations-only ONE_STATION --no-false-alarm` runs and
`screen_stations.csv` shows `n_picks` close to what the user expects.

## 2. Day-merge drift (only if the picker read whole days)

Ask how the picker read data. If it merged whole days of MiniSEED:
```bash
python -m pick_timing scan-headers -c CFG --from-picks --sample weekly --tag scan     # headers only
python -m pick_timing retime -c CFG --scan OUT/scan_channel_days.csv                 # resumable chunks
```
Check: `scan_channel_days.csv` lists the drifting channel-days (`drifting`, `absmax_resid_s`, `apparent_rate`);
`retime/corrections.csv.gz` flags: most picks `ok`; report counts of `off_grid` / `rate_not_nominal` / `error`
(those keep their time). A sampled scan can miss drifting days: re-time all days of the stations the scan flags.
If the residuals were computed before re-timing, re-locate before screening for clock errors.

## 3. Screen every station for clock errors

```bash
python -m pick_timing detect-clock -c CFG            # every station; minutes for ~700 stations, 12 workers
```
Outputs: `screen_stations.csv` (every station: screened or why not), `candidates.csv`, `false_alarm.json`.
Checks:
- `false_alarm.json`: `synthetic` and `permuted` flag rates near 0 (a 700-station test gave 0 of 1,164 and 1 of 582). If not, raise
  `dbic_min` or `amp_min_s` and say so;
- the injection recovery rates tell how large a drift the data can reveal; report them;
- if a known problem exists (the user's sanity case), it must be among the candidates.

## 4. Headers for the candidates

For candidates that end inside the data, read the record headers around the epoch to find the restart that ended it:
```bash
python -m pick_timing scan-headers -c CFG --stations STA.NET --start YYYY-MM-DD --end YYYY-MM-DD --tag cand
python -m pick_timing detect-clock -c CFG --out OUT/refit --stations-only STA.NET --restarts OUT/cand_events.csv --no-false-alarm
```
Look in `cand_channel_days.csv` for timing quality (`tq_*`), clock-locked flags (`locked_frac`) and time
corrections; many archives carry none (then say so: the headers cannot show when the clock started drifting).

## 5. Review (a person decides)

```bash
python -m pick_timing report -c CFG [--restarts OUT/cand_events.csv] [--confirmed OUT/confirmed_epochs.csv]
```
Give the user `report.html` and `review_candidates.csv`. Explain the evidence column: **clock-like** needs
co-located event pairs that show the same jump; **event-mix-like** means pairs of nearby events inside and outside
the epoch do not differ (a swarm or aftershock sequence changed the residuals, not a clock); P = S; reset at a
header restart; CC confirms the size. Decision points for the person: accept or reject each candidate; which size
(`size_source`: `cc` when CC rows exist, because locations absorb part of a clock error and residuals understate
it); the end time (`t_end_override_utc`, normally the header restart). Then:
```bash
python -m pick_timing ingest-review -c CFG --review FILLED_REVIEW.csv --output OUT/confirmed_epochs.csv
python -m pick_timing model-clock -c CFG --epochs OUT/confirmed_epochs.csv --output OUT/clock_model.json
```

## 6. Apply

```bash
python -m pick_timing apply -c CFG --clock-model OUT/clock_model.json [--retime-corrections OUT/retime/corrections.csv.gz] --output OUT/NEW_PICKS.csv
md5sum OUT/NEW_PICKS.csv
```
Never drops a pick. Corrected rows: time (and travel time) rewritten with the original decimals; every row gets
`clock_correction_s, clock_correction_sd_s, clock_epoch, clock_flag, total_correction_s`. Provenance JSON next to
the output (input md5s, model epochs, code commit).

## 7. Validate

```bash
python -m pick_timing validate -c CFG --picks-in PICKS --picks-out OUT/NEW_PICKS.csv --clock-model OUT/clock_model.json
# NonLinLoc users: re-locate the affected events, then validate with their residuals
python -m pick_timing relocate-nlloc -c CFG --run-dir RUN --clock-model OUT/clock_model.json --nlloc /path/to/NLLoc --id-prefix PREFIX
python -m pick_timing validate -c CFG --picks-in PICKS --picks-out OUT/NEW_PICKS.csv --clock-model OUT/clock_model.json --residuals-after OUT/relocate/residuals_relocated.parquet
```
Pass criteria: `rows_changed_outside_rules` 0, `rows_dropped` 0, `max_shift_error_s` 0; after correction the
station is no longer flagged and the fixed-epoch size is near 0 in re-located residuals (residuals of the old
locations keep an opposite-sign remainder of the absorbed part: report it, do not tune it away); CC misfit drops for
rows straddling the epoch; co-located stations: slope -1 before, 0 after.

## Report back

Numbers, not adjectives: stations screened / not screened and why; candidates per network; the false-alarm and
recovery rates; per confirmed epoch: start (+- sd), end, size (+- sd) and source; picks corrected; validation
numbers; md5 of the new pick file; anything that failed or was skipped.

## Common mistakes

- Treating every candidate as a clock error. Most residual steps in real catalogs are changes in the event mix
  (swarms, aftershocks). The co-located-event control separates them; a person decides.
- Taking the size of a clock drift from residuals of locations made with the drifting picks (too small by the
  absorbed fraction, about 20 % in a worked example). Use CC or re-located residuals.
- Re-timing picks that were not made on day-merged traces, or with a channel rule different from the picker's.
- Mixing time bases: re-time first (day-merge drift), then screen clocks on residuals of the re-timed picks.
- Units: `size` is s/day for drifts and s for offsets; times are UTC; c(t) sign as above.

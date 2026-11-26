# Config keys, input columns, outputs

## Tables

Each table section: `path` (csv, csv.gz, parquet; a list of paths is concatenated, with `pair` ids made unique per
file), `read_kwargs` (passed to pandas.read_csv, e.g. `{skipinitialspace: true}` for ", "-separated files),
`time_format` (`iso` or `epoch`), `columns` ({canonical: your column}).

| table | required | optional | used by |
|---|---|---|---|
| picks | event_id, station (or network + station_code), phase, time | travel_time, timing_correction | retime, apply, validate, scan-headers --from-picks |
| residuals | event_id, station, phase, time, residual | weight (rows <= `detect.min_weight` dropped), x_km/y_km, ev_x_km/ev_y_km/ev_z_km, ev_lat/ev_lon/ev_depth_km, hyp_file | detect-clock, validate, relocate-nlloc |
| stations | station | lat/lon or x_km/y_km | detect-clock (if residuals lack x_km/y_km), validate (co-located test) |
| origins | event_id, time | lat, lon, depth | detect-clock (co-located-event control if residuals lack event coordinates) |
| cc | event1, event2, station, phase, dt_cc, time1, time2 | pair, accepted, cc | detect-clock (CC refinement), report, validate |

`station_id` gives the id pattern ("{sta}.{net}" or "{net}.{sta}"). CC sign: dt_cc = (pick1 - origin1) -
(pick2 - origin2) on the record-header time base, so a station clock adds c(t1) - c(t2).

## Waveforms

| reader | keys |
|---|---|
| sds | root |
| fdsn | client (ObsPy name or URL, e.g. EARTHSCOPE, NCEDC), channels, timeout |
| sqlite_index | index (path, may contain {year}), day_mode (`filename`: YYYY and DOY in the file path, as pnwstore; `time`), path_map ({type: replace, from, to} or {type: wd_by_year, root: ".../{wd}", strip, year_part, wd_by_year}) |
| routed | routes: [{networks: [...], reader: {...}}], default: {...}, fallback: {...} |

## Parameters (defaults)

`scan`: channels [HH?, BH?, EH?], components "Z3", drift_threshold_s 0.005, tear_samples 0.5.
`retime`: channel_priority `[HH, BH, EH]`, alternate_rules `[[EH]]`, components "Z3", day_seconds 86400,
nominal_rate 100, grid_tolerance 0.001 (sample), apply_threshold_s 0.005, value_window 50.
`detect`: neighbor_km 100, min_neighbors 3, fallback event, same_phase true, min_weight 0, max_abs_residual_s 3,
bin_picks 20, bin_max_days 7, bin_min_picks 3, min_picks 60, min_bins 12, grid_max 160, min_epoch_days 2,
min_bins_in_epoch 3, models [step, offset_epoch, drift_open, drift_reset], dbic_min 30, amp_min_s 0.05, p_max 0.01,
n_perm 200, n_boot 200, max_epochs 2, colocated_event_km 5, reset_tolerance_days 1, false_alarm {n_synthetic 2,
n_permuted 1, inject [[0.004, 60], [0.008, 90]]}, seed 0.
`cc_refine`: min_rows 10, min_consensus 5, huber_s 0.03, accepted_only true.
`model`: near_start_sigmas 2. `validate`: colocated_km 3, slope_min_rows 20.

## Outputs

| file | content |
|---|---|
| `<tag>_channel_days.csv` | per channel-day: n_records, n_segments, rate, apparent_rate, absmax_resid_s, slope_s_per_day, drifting, tq_min/median/max, locked_frac, tcorr_nonzero, n_tears/n_gaps/n_overlaps |
| `<tag>_events.csv` | per tear: station, loc, cha, time, gap_s, kind (tear/gap/overlap), locked and tq before/after |
| `retime/corrections.csv.gz` | per pick: pick_row (0-based data row of the pick file), flag, why, correction_s, applied, timing_correction_s, record_start, record_pos, day_absmax_s, channel, file |
| `rd.parquet` | network-relative residuals: event_id, station, phase, time_s, residual, rd, n_ref, ref_kind |
| `screen_stations.csv` | every station: status (screened / insufficient_picks / insufficient_bins), n_picks, n_bins, sigma, tau, T, T_model, dbic_* and amp_* per model, p_perm, candidate |
| `candidates.csv` | per epoch: candidate_id, model, t_start/t_end (and their bounds), a (s/day or s), amp_s, sd_*, dbic, p_perm, a_P, a_S, coloc_*, reset_at_restart, restart_utc, cc_a, cc_a_sd, cc_n, evidence |
| `false_alarm.json` | flag rates (synthetic, permuted) with 95 % upper limits, T percentiles, injection recovery |
| `review_candidates.csv` | candidates + decision, size_source, t_end_override_utc, reviewer, decision_ref, note |
| `confirmed_epochs.csv` | epoch_id, station, model, t_start_utc, t_start_sd_days, t_end_utc, size, size_sd, size_source, confirmed_by, decision_ref, evidence |
| `clock_model.json` | format pick_timing.clock_model v1: per station epochs with times (POSIX and ISO), size, size_sd, units, source md5, code version |
| corrected pick file | input columns (times rewritten in corrected rows) + clock_correction_s, clock_correction_sd_s, clock_epoch, clock_flag (none, drift, offset, near_start, near_start_outside), total_correction_s |
| `*.provenance.json` | inputs with md5 and size, code version and git commit, parameters, counts |

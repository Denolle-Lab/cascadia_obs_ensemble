# Method

Sign convention used everywhere: **c(t) = station header time - true time** (s). Positive c: the station's picks
are late and its residuals positive. The correction added to a pick is **-c(t)** (`clock_correction_s`).

## scan-headers

Reads only the fixed header (64 bytes) of every MiniSEED record of each channel-day (header-only reads where the
reader allows it). Per channel-day:
- records are joined into segments as libmseed joins them (start within `tear_samples` = 0.5 sample of the
  previous record's end, same rate);
- `resid = header start - (segment start + samples so far / nominal rate)`: the error a day-merged trace carries.
  `absmax_resid_s >= drift_threshold_s` (default 5 ms, half a sample at 100 Hz) marks a drifting day;
- tears (restarts) between records: `tear` (< 60 s), `gap` (>= 60 s), `overlap` (< 0). Consecutive days of a
  station are scanned in order, so a tear at midnight is seen;
- timing information if present: blockette-1001 timing quality, I/O 'clock locked' flag, time-correction field
  and its 'applied' bit.
Outputs `<tag>_channel_days.csv`, `<tag>_events.csv`.

## retime (day-merge drift)

For each pick of a station-day to re-time: rebuild the picker's merged day trace from the same records (reader
`day_blocks`, ObsPy read per block, trim to the day, the picker's channel rule, `Stream.merge`) on index traces, so
the merged index of each original sample is known; check that the pick lies on the merged sample grid
(`grid_tolerance`); map it to its record and position; check the raw counts around the sample against a short read
of the neighbouring records; true time = the record's own header start + position / rate. A station-day is
corrected when its largest header-vs-count difference reaches `apply_threshold_s`. No rate model is used.
Flags: ok, ok_nonunique, in_fill, off_grid, value_mismatch, map_fail, no_data, rate_not_nominal, error.

## detect-clock (station clock errors)

1. **Network-relative residual** per pick: residual minus the median residual of the other stations of the same
   event within `neighbor_km` (same phase), at least `min_neighbors`; else the median of all other stations of the
   event (`fallback: event`). This removes origin-time and most location terms.
2. Per station, P and S pooled after removing each phase's median. Picks are grouped in time into bins of at most
   `bin_picks` picks and `bin_max_days` days (bins never straddle a header restart). Bin value = median; sampling
   variance = (1.2533 sigma)^2 / n with sigma the station's MAD scale.
3. **Change models** of c(t): constant; step; offset_epoch; drift_open; drift_reset. Weighted least squares
   with weights 1 / (s2_i + tau^2), profiled over the change times on a grid of bin boundaries (at most
   `grid_max`) plus the header restarts as reset candidates, refined at full bin resolution (and a fine time grid
   for drift starts). tau^2 (bin-to-bin scatter beyond sampling noise, mostly event mix) is estimated by
   Paule-Mandel on the residuals of the best model, started from first differences.
4. **Model selection by BIC**: dBIC = BIC(constant) - BIC(model). The statistic T is the largest dBIC among the
   non-constant models whose clock term reaches `amp_min_s`.
5. **Pre-screen** T >= `dbic_min`; **permutation test**: the station's residuals are shuffled in time
   `n_perm` times (sequential stopping), refitted, p = (1 + #{T_perm >= T}) / (1 + n_perm). Candidate if
   p <= `p_max`. Up to `max_epochs` epochs per station (the fitted term is removed and the search repeated).
6. **Uncertainties**: wild (Rademacher) bootstrap of the bin residuals, `n_boot` times.
7. **Evidence** per candidate (for the reviewer, not a decision):
   - `a_P`, `a_S`: the same model fitted to P and to S separately (a clock gives the same size);
   - co-located-event control: pairs of events within `colocated_event_km`, one inside and one outside the
     epoch, same phase: slope of the pair difference of residuals against the pair difference of the model shape.
     A clock gives `coloc_ratio` ~ 1, a change in the event mix ~ 0;
   - `reset_at_restart`: the epoch ends at a MiniSEED restart (with `--restarts` from scan-headers);
   - CC refinement (with a CC table): station dt_cc minus the median dt_cc of the other stations of the same pair
     = a (g(t1) - g(t2)); Huber fit through the origin, pair bootstrap combined with the bootstrap of the change
     times. Gives `cc_a` and the median |misfit| before and after.
   - `evidence` summarises: clock-like / unclear / event-mix-like.
8. **False alarms** (`false_alarm.json`): the whole detector, with its permutation test, is run on (a) synthetic
   clean stations with each real station's pick times and noise level (Student-t(3) picks plus bin-level scatter
   tau) and (b) each real station with its residuals shuffled in time. Flag rate and its binomial 95 % upper
   limit. **Detection power**: drifts (rate, duration) injected into (b); recovered = flagged as a drift, start
   within 25 % of the duration, rate within 50 %.

## model-clock and apply

Confirmed epochs (accepted by a person) become c(t): drift `size * (t - t_start)` on [t_start, t_end), offset
`size`; zero elsewhere. Uncertainty `sqrt((t - t_start)^2 size_sd^2 + size^2 t_start_sd^2)` for drifts. Picks within
`near_start_sigmas` x t_start_sd of a start are flagged `near_start` (inside) or `near_start_outside`.
`apply` rewrites only the time (and travel-time) fields of the rows it corrects, with the original number of
decimals; integer-nanosecond arithmetic; appends `clock_correction_s, clock_correction_sd_s, clock_epoch,
clock_flag, total_correction_s` (and `timing_corrected, timing_correction_s` when re-timing is applied here) to
every row; writes `<output>.provenance.json` (inputs with md5, model epochs, code commit, counts).

## Defaults and why

| parameter | default | reason |
|---|---|---|
| neighbor_km | 100 | close enough for similar paths, far enough to have >= 3 stations in sparse networks |
| bin_picks / bin_max_days | 20 / 7 | medians of 20 picks are near-Gaussian; a week resolves epoch ends to days |
| dbic_min | 30 | a pre-screen: in a 700-station calibration 1 of 1,164 synthetic clean stations reached it (T 31.1) and the permutation test rejected it |
| amp_min_s | 0.05 | about the pick uncertainty; smaller errors are not worth a correction |
| p_max / n_perm | 0.01 / 200 | per-station false-alarm rate of order 0.2 % on shuffled real data |
| drift_threshold_s, apply_threshold_s | 0.005 | half a sample at 100 Hz |

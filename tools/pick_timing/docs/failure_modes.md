# The two timing failure modes and how to recognise each

Pick times can be wrong in two different ways. They need different fixes, so tell them apart first.

## 1. Day-merge drift (the picker's time base is wrong, the record headers are right)

**Cause.** A picker that reads a whole day of MiniSEED and merges the records into one trace (ObsPy
`Stream.merge`, or libmseed joining records whose start is within half a sample of the previous record's end)
times every sample as `segment start + n / nominal rate`. When the digitizer's real rate differs slightly from the
nominal rate (old analog short-period channels, some telemetry chains), each record header still carries the right
time, but the sample count runs ahead of or behind it. The error grows through each merged segment and resets at
every data gap or restart.

**Signature.**
- In the record headers: within a merged segment, `header start - (segment start + samples so far / nominal rate)`
  grows steadily (`scan-headers`: `absmax_resid_s`, `slope_s_per_day`, `apparent_rate` != nominal). Typical size:
  a channel running at 99.99939 Hz instead of 100 Hz is 0.5 s off by the end of a UTC day; worse channels reach
  about 1 s/day.
- In residuals: a saw-tooth with the period of the merged segments (often one UTC day), the same for P and S,
  largest late in the day. Picks made from short windows (not day-merged) at the same station do not show it.
- Day boundaries reset it, which a clock error does not do.

**Fix.** `retime`: each pick is re-timed from the header of the record that holds its sample. Only possible when
the pick sits exactly on the picker's sample grid and the archive returns the same records the picker read. Picks
that cannot be re-timed (`off_grid`, `rate_not_nominal`, `error`) keep their time and are flagged.

**Check.** A co-located station with a reliable (GPS-locked) clock: the pick time difference between the two
stations, regressed on the correction, has slope -1 before and 0 after re-timing.

## 2. Station clock error (the record headers themselves are wrong)

**Cause.** The station's clock lost its time reference (GPS outage, failed receiver, OBS without GPS) and ran
free, or was set wrong. The header times are then wrong by c(t). Typical shapes:
- `drift_reset`: linear drift from the moment the reference was lost until a restart re-synchronises the clock
  (worked example: a land broadband station drifting +8.5 ms/day for 95 days, about +0.8 s, reset at a data
  restart; a neighbour drifting -4.1 ms/day at the same time).
- `drift_open`: drift until the end of the data (e.g. an ocean-bottom seismometer whose clock skew was never
  corrected, or corrected with the wrong value).
- `offset_epoch` / `step`: a constant offset over an interval, or from some time on (wrong time set after a
  restart, leap-second or time-zone mistakes, a wrong time correction in the headers).

**Signature.**
- In network-relative residuals (the station's residual minus the median of nearby stations for the same event):
  P and S move by the same amount; the change does not depend on where the events are; it ends at a header
  restart (a resynchronisation shows as a time tear in the records); CC differential times of event pairs that
  straddle the epoch show the same jump.
- Re-timing does not change it (the headers are wrong).
- Many archives carry no timing information at all (no blockette-1001 timing quality, no clock-locked flag, no
  time-correction field). Then only the data can show the epoch, and only the headers' restarts can show when it
  ended.

**What looks like a clock error but is not.**
- A change in the mix of events (a swarm, aftershocks of a large earthquake, a new deployment nearby) changes the
  residuals of the stations that see it. Test: pairs of nearby events, one inside the epoch and one outside, do
  not show the jump (the `colocated` evidence column; ratio near 0 means event mix).
- Path effects scale differently for P and S (S residuals are larger); a clock moves P and S equally.
- A station term that changed (sensor swap, relocation of the station) gives a permanent step; check the station
  metadata before calling it a clock.
- Location absorption: the event locations were computed with the wrong picks and absorb part of a clock error,
  so residuals understate its size (worked example: 6.9 ms/day in residuals against 8.5 ms/day from CC times).
  Take the size from CC differential times or from a re-location, not from the residuals alone.

**Fix.** `detect-clock` -> person reviews (`report`, review CSV) -> `ingest-review` -> `model-clock` -> `apply`.
Nothing is applied without a person's accept.

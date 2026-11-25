# Validation: which check proves what

| check | command | passes when | what it proves |
|---|---|---|---|
| unit tests | `python -m unittest discover -s tests` | 17 tests OK | drift/offset/step/reset recovered on synthetic stations (start and end within 3 days, size within 10 % or 0.03 s); 40 clean synthetic stations not flagged; a synthetic day-merged trace re-timed to 0.1 ms; column map round trip; apply never drops a row |
| false alarms | `detect-clock` -> `false_alarm.json` | synthetic and permuted flag rates near 0 | the thresholds do not flag stations without time structure |
| detection power | `false_alarm.json` inject rows | recovery rate reported | how large a drift the data can reveal (depends on pick density) |
| byte identity | `validate` part 1 | `rows_changed_outside_rules` = 0, `rows_dropped` = 0, `max_shift_error_s` = 0 | only the intended rows changed, only in time fields, by exactly their written correction |
| residual re-fit | `validate` part 2 | after correction the detector no longer flags the station and the fixed-epoch size is ~0 | the model removes the signal it was fitted to |
| re-location | `relocate-nlloc` then `validate --residuals-after` | the drift is gone in residuals of events re-located with the corrected picks | the correction is right, not just self-consistent (locations made with wrong picks absorb part of a clock error, so step 2 alone leaves an opposite-sign remainder) |
| co-located stations | `validate` part 3 | slope -1 before, 0 after | the corrected station agrees with a neighbour on an independent clock |
| CC | `validate` part 4 | station-vs-pair-consensus misfit drops for rows straddling the epoch; triangle closure unchanged | independent waveform timing agrees with the model |

What none of these checks can do: decide whether a candidate is a clock error or a change in the event mix. That
is the reviewer's decision, helped by the evidence columns (P = S, co-located pairs, reset at a restart, CC).

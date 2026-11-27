# Pre-registered rule: Mount St Helens analog test-signal events (catalog v6.1 event_flag)

Registered: 2026-11-26 16:27:27 PST (from `date` on the build machine), before any event was scored.
Decision: D-B108 (PI, 2026-11-26 16:09; workflow/10_tomography/DECISIONS.md on origin/tomo/tiles 7b6b9173).
Code: workflow/10_tomography/msh_flags/ (branch analysis/msh-analog-flags). Flag only: no event is deleted or moved.

## What was looked at before this rule was written (window and channel definition only)

1. Pick timing (maf_hist.py): picks of HSR, EDM, STD.UW per weekday and local hour against all other stations; Wednesday
   11:00-13:30 local in 1-min bins per year; picks per station in the Wednesday noon window against the same clock
   window on the other weekdays. Result: two Wednesday clusters near 11:56-12:00 and 12:16-12:20 local, 2011-2013 only;
   the excess is on about 13 UW analog stations around the volcano, not only HSR, EDM, STD.
2. Continuous data (maf_template.py, maf_scan.py): template of the pulse from 16 traces (2011-11-23 and 2012-05-02, two
   pulses each, HSR/EDM/STD/SHW). Scan of every day 2010-2016 at EDM.UW and every Wednesday at HSR, STD, SHW, ELK.UW,
   11:00-13:30 local. Result: two pulses every Wednesday from 2011-05-18 to 2013-08-28, onsets identical to 0.1 s on all
   channels (common mode), drifting -3.1 s per week; the generator keeps the pre-2007 US DST rule (nine Wednesdays
   with pulses one hour late in local time); on four Wednesdays the plateau lasts from the first to the second pulse;
   13 irregular common-mode pulses on other days or times (2010-2016).
3. Station screen (maf_screen.py): 24 reference pulses at every station with picks within 300 km of the summit.
4. The edge detector (maf_wave.py) was run on 11 hand-chosen times of the continuous data (pulse edges, a long plateau,
   a Tuesday control). No catalog event was classified, and no class counts were computed before this rule.

## Definitions

- **Signal channels S** (maf_screen.py; WORK/signal_stations.json): stations whose vertical channel carries the pulse
  (|cc| >= 0.9 and amplitude/noise >= 5 within 2 s of the reference onset at half or more of the 24 reference pulses
  with data). 25 UW analog stations: HSR SHW EDM STD JUN SOS FL2 CDF ELK TDL MTM LVP LCW BOW NLO VG2 KMO HBO RNO
  (pulse of 700-950 counts) and KOS RVW VLM SSO STW RMW (inverted copy of about 3 counts; RMW from one pulse only).
  No digital station (CC, PB, XD, ...) carries it. HSR, EDM, STD are the three channels named by the v5.1 review.
- **Device time**: UTC - 8 h, plus 1 h from the first Sunday of April 02:00 PST to the last Sunday of October 02:00 PDT
  (pre-2007 US DST rule of the pulse generator). Local time is America/Los_Angeles (PST/PDT, 2007+ rule).
- **Pulse windows W** (maf_calendar.py; WORK/windows.csv), in UTC:
  - weekly pulses, era 2011-05-18 .. 2013-08-28, every Wednesday, pulse 1 (device 11:50-12:05) and pulse 2 (device
    12:10-12:25): detected common-mode onset (two or more scanned stations within 2 s) -40 s .. +90 s; where a pulse was
    not detected, the onset predicted by linear interpolation in device time between the nearest detected Wednesdays,
    -120 s .. +150 s;
  - irregular pulses: common-mode detections outside those windows, onset -40 s .. +90 s.
- **Pick on the signal**: a pick (P or S) on a station of S whose time lies in W.

## Event rule

Scope: every catalog v6.0 event in the Mount St Helens box (46.05-46.30 N, 122.35-122.10 W; v6.0 epicentre, the v4
epicentre for events without a v6.0 hypocentre) and every event with at least one pick on a station of S.
Pick file: Cascadia_picks_catalog_v4.1tc.csv (pick_datetime).

For each event: n_tot = all picks, n_in = picks on the signal, share = n_in / n_tot, n_nonS = picks on stations not in S.

1. **Timing (T)**: n_in >= 3 and share >= 0.5.
2. **Waveform (WF)**: for up to 8 picks on the signal (P first, then by time), read pick - 45 s .. pick + 45 s of the
   station's vertical (pnwstore, record-header time base; raw counts, 10-Hz block means) and correlate the two edge
   templates (onset edge, the negative box masked; plateau end), either polarity. A pick is matched when an edge with
   |cc| >= 0.9 and |amplitude| / noise >= 5 lies at most 3 s after or 10 s before the pick
   (edge - 3 s <= pick <= edge + 10 s). n_wf = picks read with data, n_match = matched picks.
   WF holds when n_match >= 2 and n_match >= 0.5 n_wf.
3. **Real-earthquake guard (G)**: n_nonS <= 3 (an event with four or more picks on stations without the signal may be a
   real earthquake that happened during a pulse; it is not flagged).

Classes:
- **certain** = T and WF and G -> event_flag "analog_test_signal" in event_flags_v61.csv;
- **probable** = T and G, not WF (timing only; WF failed or no waveform) -> listed, not flagged;
- **mixed** = T, not G -> listed for review, not flagged;
- **none** = not T.

The flag file carries "certain" only. Every non-empty v6.0 event_flag is carried over unchanged; an event that already
has a flag and is "certain" gets the two joined with ';' (existing value first).

## Checks reported (no threshold is changed after scoring; any departure is reported as such)

- counts by class, by year, by window kind (weekly / irregular) and by box membership;
- depth of flagged events against unflagged events in the box;
- false-positive checks: (a) events with picks on the signal and 4 or more non-S picks (mixed) are listed and inspected;
  (b) WF run on control events without picks in W (a random sample of 400 events with >= 3 picks on S, and every box
  event on a Wednesday 11:00-13:30 device time outside W): the share of controls with WF true is the false-match rate;
  (c) flagged events with a ComCat match (comcat_id) or a magnitude above 2.5 are listed;
  (d) waveform examples of flagged events and of a real event in or next to a window.

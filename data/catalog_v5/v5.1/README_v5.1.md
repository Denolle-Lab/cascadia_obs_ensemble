# Cascadia OBS ensemble earthquake catalog v5.1

Catalog v5.1 (D-B058): catalog v4.1 events relocated with NonLinLoc in the merged 3D model v0.3.1 after the round-2 workflow fixes (every arrival used through wide travel-time grids, five-pick events located, a deep pass below 80 km, one GMRT/earth_relief solid surface); events NonLinLoc does not locate have no hypocentre.

Built 2026-11-26 by workflow/10_tomography/catalog_version.py (decisions D-B055, D-B056, D-B057, D-B058, D-B065, D-B066, D-B067, D-B068, D-B069, D-B071, D-B072, D-B073, D-B074, D-B075, D-B091, D-B102 in
workflow/10_tomography/DECISIONS.md). Private working copy; not peer reviewed; not for redistribution.

## What it is

* Events and picks: catalog v4.1 (association catalog of I. McBrearty with the v4.1 patch: event 172113 removed as
  mis-associated, events 900001-900003 added with ComCat hypocentres and repicked phases). Picks: Cascadia_picks_catalog_v4.1t.csv.gz in this folder, the v4.1 picks re-timed on drifting channels (Pick timing below; columns timing_corrected, timing_correction_s = new minus old time, s). The 45,879 picks that could not be re-timed keep their v4.1 times: the rows of pick_retiming_corrections.csv.gz with applied False and flag off_grid, rate_not_100 or error. Method and checks: pick_retiming_README.md, pick_retiming_report.html. Cascadia_picks_catalog_v4.csv.gz elsewhere in this record has the original pick times.
* Hypocentres: NonLinLoc (EDT likelihood, oct-tree) in the 3D merged Vp/Vs model v0.3.1
  (velocity_models.zip in this record), with static station terms from a first pass (station_terms_v5.1.in),
  run b4_v51_iter1. Depth is km below sea level (true depth; the Earth-flattening of the travel-time grids is undone).
* Every arrival is used: each station and phase has a fine travel-time grid (1 km, within about 250 km of the station) and a coarse wide grid over the whole model, which NonLinLoc uses when the search box is not inside the fine grid (v5.0 dropped the arrivals beyond about 160 km).
* Deep pass (D-B067): events rejected at the bottom face of the 80 km grids or located at 78 km or deeper (flattened depth) are located again in wide grids extended to about 121 km; below 80 km and outside the tomography box the velocities are ak135, blended under the deepest model layer (deep_pass). Events seen only by distant stations (far_only) are not sent to the deep pass.
* Solid surface (D-B068): one merged surface, GMRT multibeam where available and GMT earth_relief 01m elsewhere, defines NonLinLoc's topography floor (each 15 s node holds the minimum of the bilinear surface over its cell, minus 1 m, Earth-flattened; D-B056), the above-surface test (surface_km, above_surface; D-B055) and the water/rock boundary of the velocity model.
* Events with 5 picks are located with a 5-phase minimum (D-B066): one degree of freedom, tier D (five_phase).
* Model error (D-B069): a constant 0.2 s for every phase (no travel-time-dependent term, which acted as a pull toward short travel times in NonLinLoc's EDT likelihood); station terms recomputed with it.
* Pick timing (D-B072, D-B075): the picker read whole days, so on channels whose sample count drifts from the record header times (analog short-period channels, worst at 34 UW stations from 2013-10-22: about 0.5 s by the end of each UTC day; UO HUO and FRIS about 1 s/day) pick times were off by up to about 0.9 s. 218,809 picks on 75 stations were re-timed from a short read of the waveform around each pick (each pick's own MiniSEED record header time; pick file Cascadia_picks_catalog_v4.1t.csv, columns timing_corrected, timing_correction_s); a test against co-located GPS-timed stations confirms the header times.
* event_flag also marks events that ComCat lists as explosions, quarry blasts or other non-earthquakes (D-B071).
* Shallow oceanic crust (D-B073): in the travel-time grids, offshore rock within 2.5 km below the seafloor has Vp/Vs of at least 2.1 (Vs = min(Vs, Vp / 2.1)); the waveform-checked S picks of floor-held events showed S slower than the model there. The velocities are model v0.3.1 plus this rule.
* The v4.1 additions 900001-900003 carry their ComCat hypocentre (loc_source comcat, tier U) when NonLinLoc does not locate them.
* Events that NonLinLoc could not locate (tier U) have no hypocentre in this version (loc_source none: empty
  time, lat, lon and depth); u_reason says why. Their picks are in the picks file.
* Magnitudes: converged Route A preferred magnitudes (D-B091, D-B102; M, M_type ML / Mw / Mw_cal), computed with the v5.1 hypocentres (catalog v5.1 magnitude table) where that table has a magnitude for the event, else the v4.1 table (v4 hypocentres); magnitude_source names which.

## Counts

| item | value |
|---|---|
| events | 172,342 |
| located here (loc_source nll3d) | 171,717 |
| ComCat hypocentre (loc_source comcat, v4.1 additions) | 2 |
| not located (loc_source none, tier U) | 623 |
| tiers | A 21,325, B 33,896, C 73,464, D 43,032, U 625 |
| depth_quality | ok 137,592, unconstrained 24,939, biased_deep 9,188 |
| floor_limited | 8,682 |
| far_only | 624 |
| deep_pass | 356 |
| five_phase | 11,978 |
| with a magnitude | 131,112 |
| above the ground or seafloor | 0 (none) |
| publishable (D-B055) | yes |

## Columns

| column | meaning |
|---|---|
| ID | catalog v4 event ID (900001-900003: v4.1 additions) |
| raw_ID | ID in the GENIE association output |
| time | origin time (UTC) of this version's hypocentre |
| lat, lon | epicentre (deg, WGS84) |
| depth | km below sea level (negative = above sea level) |
| exp_depth | NonLinLoc expectation depth (km bsl) |
| loc_source | nll3d (located here), comcat (ComCat hypocentre), none (not located: no hypocentre, see u_reason) |
| tier | location quality A (best) to D (D includes every event located with 5 phases, D-B066), U = not located here: no hypocentre, see u_reason (rules in a2_build_catalog.py; counts in b4_catalog_v5_trackA_*_tiers.md) |
| nphs, n_sta, gap, rms | phases and stations used, largest azimuthal gap (deg), rms residual (s) |
| h_min, h_max, sig_z, ell_len3 | 68 % horizontal semi-axes, depth standard error, longest ellipsoid axis (km) |
| dist_min, dist_med, dist_max, dist_over_depth | epicentral distance to the stations (km); nearest / depth |
| surface_km | ground or seafloor elevation at the epicentre (km, positive up): the merged solid surface of D-B068 (GMRT multibeam where available, GMT earth_relief 01m elsewhere, bilinear), the surface of NonLinLoc's floor; empty without a hypocentre |
| above_surface | depth above the solid surface (surface_km): a location error, kept visible, not moved (D-B055); empty without a hypocentre |
| depth_quality | ok, unconstrained (ridge regions, Blanco outside the 2012-13 OBS deployment), biased_deep (Gorda, nearest station > 15 km); empty without a hypocentre |
| event_flag | quarry_blast_likely (Texada Island box, weekday working hours) and/or anss_explosion, anss_quarry_blast, anss_other_event, anss_sonic_boom (within 3 s and 20 km of a ComCat event of that type, D-B071; a surface depth is right for these), joined with ';'; empty otherwise |
| shift_h_km, shift_z_km | change from the v4 hypocentre |
| tile, region | NonLinLoc tile (workdir of the main or deep pass; empty when not located here); region label of the v4 epicentre (box rules of a2_parse_hyp.py), also for rows without a hypocentre |
| depth_frame | how depth was obtained (true depth: the Earth-flattening of the NonLinLoc grids undone; ComCat for v4.1 additions); empty without a hypocentre |
| v4_time, v4_lat, v4_lon, v4_depth, v4_npick, v4_sigma_m | catalog v4 origin, hypocentre, picks, GraphDD spatial sigma (m) |
| M, M_unc, M_type, ML, ML_unc, n_ML, ML_method, Mw_mt, comcat_id | converged Route A magnitudes (D-B091, D-B102); magnitude_source names the table |
| magnitude_source | v5.1_converged (computed with the v5.1 hypocentres) or v4.1_converged (v4 hypocentres), D-B091/D-B102; empty without a magnitude |
| u_reason | why a tier U event has no hypocentre: 'rejected: <NonLinLoc message>' (e.g. maximum likelihood on a face of the search box), 'not in the run's observation files', 'fewer than N picks' (picks in the run's observation file; N = LOCMETH minimum of the run's nlloc.in, 5) or 'too few usable arrivals' (N or more picks, fewer usable); empty for located events |
| floor_limited | True when the maximum-likelihood point lies within 1.5 final oct-tree cells of NonLinLoc's topography floor (the floor file of the run, read as NonLinLoc reads it; D-B056, D-B068): the depth is set by the floor, not by the data; empty when not located here |
| far_only | True when none of the arrivals used came from a fine (near-station) travel-time grid: seen only by distant stations through the coarse wide grids; epicentre usable, depth poor; empty when not located here |
| deep_pass | True when the hypocentre comes from the deep pass (D-B067: grids to about 121 km, ak135 below 80 km and outside the tomography box); empty when not located here |
| five_phase | True when located with exactly 5 phases (D-B066: one degree of freedom, always tier D); empty when not located here |

## Known limits of this version

* floor_limited events (count above) have depths set by NonLinLoc's topography floor, not by the data (in round 2 such events had sigma_z about 9 km and expectation depths about 13 km below the best point).
* far_only events (count above) are located with distant stations only: epicentres usable, depths poor (round 2: median difference from ANSS 21-24 km, against 11 km for catalog v4; Blanco M5-6 events at 59-77 km).
* five_phase events have one degree of freedom (round 2: sigma_z about 13 km, horizontal error about 27 km).
* Deep-pass hypocentres below 80 km depend on ak135 outside the tomography box (D-B067); check sigma_z (e.g. one deep-pass event at 103.6 km under the Juan de Fuca plate has sigma_z 12.6 km).
* S arrivals beyond about 300 km have residuals of -0.8 to -1.1 s (velocity model or picks, not the travel-time grids; D-B074); they matter most for far_only events.
* Pick timing (D-B072, D-B075): 45,879 picks on drifting channels could not be re-timed and keep their catalog times (mostly ocean-bottom channels with drift below 0.02 s; CC SEP and SUG EHZ from 2014-01-18 on, whose re-timed picks before that date needed a median 0.01 s and up to about 0.08 s; channels the picker resampled from other rates); NC and BK picks were not checked (not in the waveform archive used).
* Axial: 81 % of the 12,975 tier A-B Axial events are within 2 km of the seafloor (median 1.7 km below it), against 90 % in Waldhauser et al. (2020).
* Blanco: with the shallow-crust rule (D-B073) depths are about 1 km deeper relative to the Lange et al. (2026) OBS catalog than without it if that catalog's depths are below sea level; its datum is not stated (below the seafloor would reverse the comparison).

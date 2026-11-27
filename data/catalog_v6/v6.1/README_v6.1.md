# Cascadia OBS ensemble earthquake catalog v6.1

Catalog v6.1 (D-B103, D-B104, D-B108): catalog v4.1 events relocated with NonLinLoc in the merged model v0.4.2, with source-specific station terms (NLL-SSST, characteristic distances 9999 and 16 km) and waveform-coherence relocation (NLL-coherence), on the re-timed and clock-corrected picks (v4.1tc); magnitudes recomputed with these hypocentres; events NonLinLoc does not locate have no hypocentre.

Built 2026-11-27 by workflow/10_tomography/catalog_version.py (decisions D-B055, D-B068, D-B084, D-B090, D-B091, D-B098, D-B099, D-B103, D-B104, D-B105, D-B108 in
workflow/10_tomography/DECISIONS.md). Private working copy; not peer reviewed; not for redistribution.

## What it is

* Events and picks: catalog v4.1 (association catalog of I. McBrearty with the v4.1 patch: event 172113 removed as
  mis-associated, events 900001-900003 added with ComCat hypocentres and repicked phases). Picks: Cascadia_picks_catalog_v4.1tc.csv.gz in this folder, as in v6.0: the v4.1 picks re-timed on drifting channels (v4.1t, D-B072/D-B075) plus the clock-drift corrections of MH07.XD and MG07.XD (spring 2016, D-B090; columns clock_correction_s, clock_correction_sd_s, clock_epoch, clock_flag, total_correction_s). The other candidate clock epochs wait for the PI's review (v6.2, D-B108). Method: pick_timing_README.md.
* Hypocentres: NonLinLoc (EDT likelihood, oct-tree) in the 3D merged Vp/Vs model v0.4.2 (velocity_models.zip in this record), with source-specific station terms and coherence relocation (below), run v6.1. Depth is km below sea level (true depth; the Earth-flattening of the travel-time grids is undone). station_terms_v6.1.in: the static terms of the first pass, used only by the deep and edge passes.
* Model v0.4.2 (D-B099, D-B104): model v0.4.1 (re-inverted with the re-timed picks and the tomoDD surface floor, D-B082, D-B083) with event self-pairs removed from the inversion data and a tomoDD numerical fix. Travel-time grids as for v5.1 and v6.0: merged solid surface (D-B068), fine and wide grids, deep pass below 80 km (D-B067).
* Source-specific station terms (NLL-SSST, Loc2ssst; Lomax & Savvaidis 2022): two passes, characteristic distances 9999 and 16 km (D-B103). A pre-registered depth test (Axial, Mount St Helens, western Blanco, Gorda Depression) kept only this schedule with coherence within the held-out-fit gate; v6.0's further passes at 8 and 4 km deepened events through the trade-off between station terms and depth. The deep and edge passes use the static terms of the first pass (column station_terms).
* Coherence relocation (NLL-coherence, Lomax & Savvaidis 2022), as in v6.0: an event with a compact location density is relocated on its density stacked with those of its waveform-coherent neighbours (pair coherence from waveform cross-correlation, D-B040; D-B084). Not applied in the Mendocino region (D-B098; column coherence_skipped_region).
* Edge pass: events rejected at a lateral face of their tile box are located again in a box centred on them (column edge_pass).
* event_flag marks events that ComCat lists as explosions, quarry blasts or other non-earthquakes (D-B071) and, new in v6.1, likely false events from the weekly artificial test signal on the UW analog channels near Mount St Helens (analog_test_signal, D-B108). Flags only: no event is removed.
* review_* columns: the automatic indicators of the v5.1 location review (D-B097); no reviewer verdict is applied in this version.
* Events that NonLinLoc could not locate (tier U) have no hypocentre in this version (loc_source none: empty
  time, lat, lon and depth); u_reason says why. Their picks are in the picks file.
* Magnitudes: converged Route A preferred magnitudes (exact station-term solver, D-B091; M, M_type ML / Mw / Mw_cal) computed with the v6.1 hypocentres (D-B105; magnitude_source v6.1_converged) where that table has a magnitude for the event, else the v4.1 table (v4 hypocentres; v4.1_converged).

## Counts

| item | value |
|---|---|
| events | 172,342 |
| located here (loc_source nll3d) | 171,996 |
| ComCat hypocentre (loc_source comcat, v4.1 additions) | 1 |
| not located (loc_source none, tier U) | 345 |
| tiers | A 22,516, B 33,719, C 82,349, D 33,412, U 346 |
| depth_quality | ok 137,799, unconstrained 24,967, biased_deep 9,231 |
| floor_limited | 7,076 |
| far_only | 347 |
| deep_pass | 284 |
| edge_pass | 342 |
| five_phase | 12,036 |
| coherence_relocated | 51,415 |
| coherence_skipped_region | 10,500 |
| with a magnitude | 131,095 |
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
| tier | location quality A (best) to D (D includes every event located with 5 phases, D-B066), U = not located here: no hypocentre, see u_reason (rules: a2_build_catalog.tier; counts in the table above) |
| nphs, n_sta, gap, rms | phases and stations used, largest azimuthal gap (deg), rms residual (s) |
| h_min, h_max, sig_z, ell_len3 | 68 % horizontal semi-axes, depth standard error, longest ellipsoid axis (km) |
| dist_min, dist_med, dist_max, dist_over_depth | epicentral distance to the stations (km); nearest / depth |
| surface_km | ground or seafloor elevation at the epicentre (km, positive up): the merged solid surface of D-B068 (GMRT multibeam where available, GMT earth_relief 01m elsewhere, bilinear), the surface of NonLinLoc's floor; empty without a hypocentre |
| above_surface | depth above the solid surface (surface_km): a location error, kept visible, not moved (D-B055); empty without a hypocentre |
| depth_quality | ok, unconstrained (ridge regions, Blanco outside the 2012-13 OBS deployment), biased_deep (Gorda, nearest station > 15 km); empty without a hypocentre |
| event_flag | quarry_blast_likely (Texada Island box, weekday working hours) and/or anss_explosion, anss_quarry_blast, anss_other_event, anss_sonic_boom (within 3 s and 20 km of a ComCat event of that type, D-B071; a surface depth is right for these), joined with ';'; analog_test_signal: likely false event from the weekly artificial test signal on the UW analog channels HSR, EDM and STD near Mount St Helens (D-B108; evidence msh_analog_flags.html and msh_analog_candidates.csv in this folder); empty otherwise. A flag only: no event is removed |
| shift_h_km, shift_z_km | change from the v4 hypocentre |
| tile, region | NonLinLoc tile (workdir of the main or deep pass; empty when not located here); region label of the v4 epicentre (box rules of a2_parse_hyp.py), also for rows without a hypocentre |
| depth_frame | how depth was obtained (true depth: the Earth-flattening of the NonLinLoc grids undone; ComCat for v4.1 additions); empty without a hypocentre |
| v4_time, v4_lat, v4_lon, v4_depth, v4_npick, v4_sigma_m | catalog v4 origin, hypocentre, picks, GraphDD spatial sigma (m) |
| M, M_unc, M_type, ML, ML_unc, n_ML, ML_method, Mw_mt, comcat_id | converged Route A magnitudes (D-B091) computed with the v6.1 hypocentres (D-B105); magnitude_source names the table |
| magnitude_source | table of the magnitude columns: v6.1_converged = converged Route A table computed with the v6.1 hypocentres (D-B091, D-B105); v4.1_converged = the same with the v4 hypocentres, for events the v6.1 table has no magnitude for; empty without a magnitude |
| u_reason | why a tier U event has no hypocentre: 'rejected: <NonLinLoc message>' (e.g. maximum likelihood on a face of the search box), 'not in the run's observation files', 'fewer than N picks' (picks in the run's observation file; N = LOCMETH minimum of the run's nlloc.in, 5) or 'too few usable arrivals' (N or more picks, fewer usable); empty for located events |
| floor_limited | True when the maximum-likelihood point lies within 1.5 final oct-tree cells of NonLinLoc's topography floor (the floor file of the run, read as NonLinLoc reads it; D-B056, D-B068): the depth is set by the floor, not by the data; empty when not located here |
| far_only | True when none of the arrivals used came from a fine (near-station) travel-time grid: seen only by distant stations through the coarse wide grids; epicentre usable, depth poor; empty when not located here |
| deep_pass | True when the hypocentre comes from the deep pass (D-B067: grids to about 121 km, ak135 below 80 km and outside the tomography box); empty when not located here |
| five_phase | True when located with exactly 5 phases (D-B066: one degree of freedom, always tier D); empty when not located here |
| rms_unweighted | unweighted rms of the residuals of the arrivals used (s); rms is NonLinLoc's EDT-weighted value (D-B069) |
| shift_h_v51_km, shift_z_v51_km | change from catalog v5.1 (horizontal; depth, positive = deeper), when both are located |
| model_version | velocity model of the travel-time grids (v0.4.2) |
| loc_pass | NonLinLoc run of the hypocentre: ssst_it2 (final SSST pass), coherence, deep or edge |
| station_terms | ssst (NLL-SSST, characteristic distances 9999-16 km, D-B103) or static (deep and edge passes) |
| ssst_char_km | final Loc2ssst characteristic distance of the event's station terms (16 km; empty for static rows) |
| coherence_relocated | relocated by the NLL-coherence stage (compact location density, Len3 <= 5 km) |
| coherence_partners | neighbours with pair coherence >= 0.5 and a compact density, stacked for the event |
| coherence_skipped_region | compact event not coherence-relocated because it lies in the Mendocino region (D-B098); SSST location kept |
| edge_pass | hypocentre from the edge pass (event rejected at a lateral face of its SSST tile box) |
| n_picks_retimed | picks of the event re-timed from a short waveform read (D-B072/D-B075) |
| n_picks_clock_corrected | picks of the event corrected for a confirmed station clock drift (D-B090) |
| review_problems, review_auto_indicators | v5.1 location review: problem groups and automatic indicators, ';'-joined (indicators, not verdicts; D-B097) |
| review_not_earthquake, review_depth_unreliable, review_relocate, review_depth_km | reviewer verdict columns, empty in this version (D-B097) |

## Known limits of this version

* Axial depths: 73 % of the 12,584 tier A-B Axial events are within 2 km of the seafloor (median 1.8 km below it), against 90 % in Waldhauser et al. (2020). The target (85 % of tier A-B events within 2 km of the seafloor) is still missed (v6.0 66 %, v5.1 81 %). Cause (D-B103): the station-term/depth trade-off and a real near-surface Vp/Vs error of the model (common S terms about -60 ms relative to P at Axial, about -150 ms at Puget). Fix path: correct the near-surface Vp/Vs of the model, then relocate. Acceptance report: b4_acceptance_v6.1.html.
* Flagged events (event_flag) stay in the catalog and also fed the station terms of this run; station terms without them are a candidate for v6.2.
* Deferred to v6.2 (D-B108): the other candidate station clock epochs (129, including the Axial cabled array, Dec 2014 - Jan 2015) and the location-review verdicts (1,512 events; review_not_earthquake, review_depth_unreliable, review_relocate, review_depth_km empty).
* No coherence relocation in the Mendocino region (D-B098; 10,500 compact events keep their final SSST location). Provisional.
* floor_limited, far_only and five_phase events have poorly constrained depths, as in v6.0 (counts above).

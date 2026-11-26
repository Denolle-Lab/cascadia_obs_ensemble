# Cascadia OBS ensemble earthquake catalog v6.0

Catalog v6.0 (D-B079, D-B100): catalog v4.1 events relocated with NonLinLoc in the re-inverted merged model v0.4.1, with source-specific station terms (NLL-SSST) and waveform-coherence relocation (NLL-coherence), on the re-timed and clock-corrected picks (v4.1tc); events NonLinLoc does not locate have no hypocentre.

Built 2026-11-26 by workflow/10_tomography/catalog_version.py (decisions D-B055, D-B068, D-B079, D-B082, D-B083, D-B084, D-B090, D-B091, D-B095, D-B096, D-B097, D-B098, D-B099, D-B100 in
workflow/10_tomography/DECISIONS.md). Private working copy; not peer reviewed; not for redistribution.

## What it is

* Events and picks: catalog v4.1 (association catalog of I. McBrearty with the v4.1 patch: event 172113 removed as
  mis-associated, events 900001-900003 added with ComCat hypocentres and repicked phases). Picks: Cascadia_picks_catalog_v4.1tc.csv.gz in this folder: the v4.1 picks re-timed on drifting channels (v4.1t, D-B072/D-B075) plus the clock-drift corrections of MH07.XD and MG07.XD (spring 2016, D-B090; columns clock_correction_s, clock_correction_sd_s, clock_epoch, clock_flag, total_correction_s). The other candidate clock epochs of the station screen are not applied (PI review, v6.1). Method: pick_timing_README.md.
* Hypocentres: NonLinLoc (EDT likelihood, oct-tree) in the 3D merged Vp/Vs model v0.4.1 (velocity_models.zip in this record), with source-specific station terms and coherence relocation (below), run v6.0. Depth is km below sea level (true depth; the Earth-flattening of the travel-time grids is undone). station_terms_v6.0.in: the static terms of the first pass, used only by the deep and edge passes.
* Model v0.4.1 (D-B083): the merged tomography model re-inverted with the re-timed picks (v0.4.0, D-B082) and the tomoDD surface floor (D-B081) in the inversion. Travel-time grids as for v5.1: merged solid surface (D-B068), fine and wide grids, deep pass below 80 km (D-B067).
* Source-specific station terms (NLL-SSST, Loc2ssst; Lomax & Savvaidis 2022): four passes with characteristic distances 9999, 16, 8 and 4 km (D-B095, D-B096) replace v5.1's static terms; the deep and edge passes use the static terms of the first pass (column station_terms).
* Coherence relocation (NLL-coherence, Lomax & Savvaidis 2022): an event with a compact location density is relocated on its density stacked with those of its waveform-coherent neighbours (pair coherence from waveform cross-correlation, D-B040), the relative-relocation step chosen for B5 (D-B078, D-B084). Not applied in the Mendocino region (D-B098; column coherence_skipped_region).
* Edge pass: events rejected at a lateral face of their tile box are located again in a box centred on them (column edge_pass).
* event_flag also marks events that ComCat lists as explosions, quarry blasts or other non-earthquakes (D-B071).
* review_* columns: the automatic indicators of the v5.1 location review (D-B097); no reviewer verdict is applied in this version.
* Events that NonLinLoc could not locate (tier U) have no hypocentre in this version (loc_source none: empty
  time, lat, lon and depth); u_reason says why. Their picks are in the picks file.
* Magnitudes: converged Route A preferred magnitudes (D-B091; M, M_type ML / Mw / Mw_cal): the catalog v5.1 table (computed with v5.1 hypocentres) where it has the event, else the v4.1 table (v4 hypocentres); magnitude_source names which. They are not recomputed with the v6.0 hypocentres.

## Counts

| item | value |
|---|---|
| events | 172,342 |
| located here (loc_source nll3d) | 172,182 |
| ComCat hypocentre (loc_source comcat, v4.1 additions) | 1 |
| not located (loc_source none, tier U) | 159 |
| tiers | A 22,907, B 33,191, C 83,149, D 32,935, U 160 |
| depth_quality | ok 137,970, unconstrained 24,971, biased_deep 9,242 |
| floor_limited | 5,738 |
| far_only | 353 |
| deep_pass | 151 |
| edge_pass | 283 |
| five_phase | 12,082 |
| coherence_relocated | 53,746 |
| coherence_skipped_region | 11,148 |
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
| M, M_unc, M_type, ML, ML_unc, n_ML, ML_method, Mw_mt, comcat_id | converged Route A magnitudes (D-B091); magnitude_source names the table |
| u_reason | why a tier U event has no hypocentre: 'rejected: <NonLinLoc message>' (e.g. maximum likelihood on a face of the search box), 'not in the run's observation files', 'fewer than N picks' (picks in the run's observation file; N = LOCMETH minimum of the run's nlloc.in, 5) or 'too few usable arrivals' (N or more picks, fewer usable); empty for located events |
| floor_limited | True when the maximum-likelihood point lies within 1.5 final oct-tree cells of NonLinLoc's topography floor (the floor file of the run, read as NonLinLoc reads it; D-B056, D-B068): the depth is set by the floor, not by the data; empty when not located here |
| far_only | True when none of the arrivals used came from a fine (near-station) travel-time grid: seen only by distant stations through the coarse wide grids; epicentre usable, depth poor; empty when not located here |
| deep_pass | True when the hypocentre comes from the deep pass (D-B067: grids to about 121 km, ak135 below 80 km and outside the tomography box); empty when not located here |
| five_phase | True when located with exactly 5 phases (D-B066: one degree of freedom, always tier D); empty when not located here |
| magnitude_source | v5.1_converged (computed with v5.1 hypocentres) or v4.1_converged (v4 hypocentres), D-B091; empty without a magnitude |
| rms_unweighted | unweighted rms of the residuals of the arrivals used (s); rms is NonLinLoc's EDT-weighted value (D-B069) |
| shift_h_v51_km, shift_z_v51_km | change from catalog v5.1 (horizontal; depth, positive = deeper), when both are located |
| model_version | velocity model of the travel-time grids (v0.4.1) |
| loc_pass | NonLinLoc run of the hypocentre: ssst_it4 (final SSST pass), coherence, deep or edge |
| station_terms | ssst (NLL-SSST, characteristic distances 9999-16-8-4 km) or static (deep and edge passes) |
| ssst_char_km | final Loc2ssst characteristic distance of the event's station terms (4 km; empty for static rows) |
| coherence_relocated | relocated by the NLL-coherence stage (compact location density, Len3 <= 5 km) |
| coherence_partners | neighbours with pair coherence >= 0.5 and a compact density, stacked for the event |
| coherence_skipped_region | compact event not coherence-relocated because it lies in the Mendocino region (D-B098); SSST location kept |
| edge_pass | hypocentre from the edge pass (event rejected at a lateral face of its SSST tile box) |
| n_picks_retimed | picks of the event re-timed from a short waveform read (D-B072/D-B075) |
| n_picks_clock_corrected | picks of the event corrected for a confirmed station clock drift (D-B090) |
| review_problems, review_auto_indicators | v5.1 location review: problem groups and automatic indicators, ';'-joined (indicators, not verdicts; D-B097) |
| review_not_earthquake, review_depth_unreliable, review_relocate, review_depth_km | reviewer verdict columns, empty in this version (D-B097) |

## Known limits of this version

* Axial depths: 66 % of the 11,939 tier A-B Axial events are within 2 km of the seafloor (median 1.9 km below it), against 90 % in Waldhauser et al. (2020). The SSST passes deepen Axial by 0.22 km (median; before them 82.7 % of tier A-B events were within 2 km of the seafloor, v5.1 81 %, target 85 %). Median depth changes from the SSST passes elsewhere: Puget +0.29, Mendocino +0.17, Blanco +0.15, Gorda -0.22, offshore forearc -0.15 km. Blanco Ridge and the Gorda Depression come out deeper against Lange et al. (2026). To be fixed in v6.1 (D-B100); acceptance report: b4_acceptance_v6.0.html.
* Model v0.4.1, not v0.4.2 (D-B099): v0.4.2 removes event self-pairs from the inversion data and adds a tomoDD numerical fix; the difference is local (Vs near 29 km depth under Vancouver Island).
* No coherence relocation in the Mendocino region (D-B098; 11,148 events keep their final SSST location): on the one Mendocino test cluster coherence made the held-out fit worse. Provisional.
* Deferred to v6.1 (D-B097): the weekly artificial signal on UW analog stations near Mount St Helens (HSR, EDM, STD.UW; candidate false deep events), the other candidate station clock epochs (including the Axial cabled array, Dec 2014 - Jan 2015), and the reviewer verdicts (review_not_earthquake, review_depth_unreliable, review_relocate, review_depth_km empty).
* Magnitudes are not recomputed with the v6.0 hypocentres (magnitude_source).
* floor_limited, far_only and five_phase events have poorly constrained depths, as in v5.1 (counts above).

# Cascadia OBS ensemble earthquake catalog v5.0

Catalog v5.0 (interim, D-B058): catalog v4.1 events relocated with NonLinLoc in the merged 3D model v0.3.1, before the round-2 NonLinLoc workflow fixes; tier U keeps v4 hypocentres, flagged (D-B059).

Built 2026-11-24 by workflow/10_tomography/catalog_version.py (decisions D-B058, D-B059 in
workflow/10_tomography/DECISIONS.md). Private working copy; not peer reviewed; not for redistribution.

## What it is

* Events and picks: catalog v4.1 (association catalog of I. McBrearty with the v4.1 patch: event 172113 removed as
  mis-associated, events 900001-900003 added with ComCat hypocentres and repicked phases). Picks are not repeated
  here: they are Cascadia_picks_catalog_v4.csv.gz in this record, with the v4.1 patch.
* Hypocentres: NonLinLoc (EDT likelihood, oct-tree) in the 3D merged Vp/Vs model v0.3.1
  (velocity_models.zip in this record), with static station terms from a first pass (station_terms_v5.0.in),
  run b4_iter1_v0.3.1. Depth is km below sea level (true depth; the Earth-flattening of the travel-time grids is undone).
* Events that NonLinLoc could not locate (tier U: too few picks or no solution inside the grids) keep the v4
  hypocentre (loc_source v4). The v4 depths come from a location without topography or water and are often above the
  surface; they are kept, with the above_surface flag, for this version only (D-B059).
* Magnitudes: Route A preferred magnitude of catalog v4.1 (M, M_type ML / Mw / Mw_cal), computed with the v4
  hypocentres; they will be recomputed with the new hypocentres in a later version.

## Counts

| item | value |
|---|---|
| events | 172,342 |
| located here (loc_source nll3d) | 157,269 |
| v4 hypocentre kept (loc_source v4, tier U) | 15,070 |
| ComCat hypocentre (loc_source comcat, v4.1 additions) | 3 |
| tiers | A 18,274, B 27,011, C 76,687, D 35,297, U 15,073 |
| depth_quality | ok 139,505, unconstrained 24,989, biased_deep 7,848 |
| with a magnitude | 130,769 |
| above the ground or seafloor | 4,077 (v4 4,077) |
| publishable (D-B055) | no: events above the surface are computation errors still to fix |

## Columns

| column | meaning |
|---|---|
| ID | catalog v4 event ID (900001-900003: v4.1 additions) |
| raw_ID | ID in the GENIE association output |
| time | origin time (UTC) of this version's hypocentre |
| lat, lon | epicentre (deg, WGS84) |
| depth | km below sea level (negative = above sea level) |
| exp_depth | NonLinLoc expectation depth (km bsl) |
| loc_source | nll3d (located here), v4 (v4 hypocentre kept, tier U), comcat (ComCat hypocentre) |
| tier | location quality A (best) to D, U = not located here (rules in b4_catalog_v5_trackA_*_tiers.md) |
| nphs, n_sta, gap, rms | phases and stations used, largest azimuthal gap (deg), rms residual (s) |
| h_min, h_max, sig_z, ell_len3 | 68 % horizontal semi-axes, depth standard error, longest ellipsoid axis (km) |
| dist_min, dist_med, dist_max, dist_over_depth | epicentral distance to the stations (km); nearest / depth |
| surface_km | ground or seafloor elevation at the epicentre (km, positive up; GMT earth_relief 01m, bilinear) |
| above_surface | depth above the solid surface: a location error, kept visible, not moved (D-B055) |
| depth_quality | ok, unconstrained (ridge regions, Blanco outside the 2012-13 OBS deployment), biased_deep (Gorda, nearest station > 15 km) |
| event_flag | quarry_blast_likely (Texada Island box, weekday daytime) or empty |
| shift_h_km, shift_z_km | change from the v4 hypocentre |
| tile, region | NonLinLoc tile; region label (box rules of a2_parse_hyp.py) |
| depth_frame | how depth was obtained |
| v4_time, v4_lat, v4_lon, v4_depth, v4_npick, v4_sigma_m | catalog v4 origin, hypocentre, picks, GraphDD spatial sigma (m) |
| M, M_unc, M_type, ML, ML_unc, n_ML, ML_method, Mw_mt, comcat_id | Route A magnitudes of catalog v4.1 |

## Known limits of this version

* Interim (D-B058): the NonLinLoc workflow fixes of round 2 (station reach, upward pull towards the surface, the 80 km
  grid floor, tier U) are not in it; v5.1 will be the rerun with them, v5.2 the tomoDD relative relocation.
* About 4 % of the located events have a likelihood that peaks above the surface; NonLinLoc's topography floor holds
  them at the surface (D-B056), so their depth is set by the floor, not by the data.
* Event 172114: located with its v4.0 picks; v4.1 added 160 picks to it that this run did not use.
* Depths are poorly constrained offshore without nearby OBS (depth_quality) and are deeper than published local
  studies at Axial, Blanco, Gorda and Mendocino (data/tomo_v1/depth_literature/depth_literature.html). Axial: 28 % of the 3,474 tier A-B Axial events are within 2 km of the seafloor (median 2.4 km below it), against 90 % in Waldhauser et al. (2020).
  Region medians against the previous relocations: b4_acceptance.md.
* Mount St Helens: median +0.49 km deeper than Hirao et al. (2025) on 2,369 paired events (tiers A-D; A-B +0.67 km).
* Blanco: median +1.99 km deeper than Lange et al. (2026) on 11,298 paired events (tiers A-D), above the 1 km target; West Blanco +4.64 km, worse than in the v0.3.0 relocation (+2.80 km), Blanco Ridge +2.89 km, Gorda Depression +6.72 km, Cascadia Depression -1.26 km (paired_v0.3.1.md).

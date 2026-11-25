# Cascadia fault surfaces v1 (from catalog v5.1)

**Version v1, 2026-11-25 (revised after review the same day). Private working product; not peer reviewed.** Report
with figures, findings and proposed decisions: `fault_surfaces_v1_report.html`.

**Status.** Catalog v5.1 will be superseded: the PI adopted velocity model v0.4.0 (D-B082), v0.4.1 (surface floor in the
inversion, D-B083) is being built, and catalog v6.0 (D-B079, NonLinLoc with SSST station terms) will replace v5.1. These
surfaces will be redone on v6.0. Waveform review reports for four v5.1 location problems:
`/wd1/mdenolle_data/catalog_v5/v5.1_review/index.html` (P1 Mount St Helens deep events, P2 Brookings 2012, P3 events with
ComCat moment tensors, P4 deep-pass and far-only events). The surfaces and events they touch are flagged here.

v1 refits the plane-based fault surfaces of v0 (`data/faults/v0/`, catalog v4) on catalog v5.1: catalog v4.1 events
relocated with NonLinLoc in the 3D model v0.3.1, re-timed picks, a deep pass, one merged solid surface
(`/wd1/mdenolle_data/catalog_v5/v5.1/README_v5.1.md`). Depths are km below sea level.

## Content

| | |
|---|---|
| Fit set | 101,296 events: tiers A-C located by NonLinLoc, without floor_limited, far_only, deep_pass, ComCat non-earthquakes and Texada quarry_blast_likely events; Axial left out as in v0 (`fault_surfaces_v1_selection.csv`) |
| Surfaces | 142 (99 with dip >= 45 deg; 12 Blanco strands); 10 removed by screening (`fault_surfaces_v1_removed.csv`) |
| Fit members | 64,272 (63 % of the fit set; v0: 53,291, 70 %) |
| Every located event (171,719) | 64,272 fit members; 12,651 assigned (compatible with exactly one surface, not located on it); 26,257 ambiguous (two or more compatible surfaces); 37,468 none (no compatible surface, in a fitted region); 26,874 at Axial Seamount, not fitted (left out, so neither on nor off a surface); 3,313 non-earthquakes; 884 weekday-daytime members of surfaces removed for their timing (likely blasts). The 623 rows without a hypocentre are left out. `on_surface_strict`: 65,709 (fit members plus assigned tier A-C events within 2 km, without floor_limited, far_only or deep_pass). |
| Against v0 | of the 141 v0 surfaces: 50 matched, 25 merged, 10 split, 20 split and merged, 13 dispersed, 23 lost; 39 v1 surfaces are new |

## Method (code in `workflow/07_faults/`)

- `v1_common.py`: event selection and per-event errors (sigma_h = sqrt((h_min^2 + h_max^2)/2) / 1.515, sigma_z = sig_z).
- `v1_detect.py`: the v0 detection (`fault_surfaces_planes.py`: RANSAC + growth, k-planes reassignment, merging,
  v0 parameters) with the dip >= 30 deg rule also applied after every reassignment pass; main run and four stability
  runs (seeds 101, 202; tolerance factor 1.25, 1.75).
- `v1_build.py`: Blanco strands (rule of `split_strands.py`, automatic: per-year Ashman D, depth test), rectangles with
  uncertainties (bootstrap, location Monte Carlo with each event's own error, model choice), screening (timing, deep
  Blanco artefacts, final dip), assignment of every located event (z = d / sqrt(s_e^2 + s_s^2 + W^2) <= 2 and
  d <= 10 km, unique), comparison with v0 by shared events and literature carry-forward.
- `v1_report.py`, `v1_report_text.py`: the HTML report.
- Detection-level recovery: the verifier's three detection reruns (80 % subsample, one perturbation of every event by its
  own error, tiers A-B only) are copied to `/wd1/.../faults/work/verifier_reruns/` (with their scripts) and read by
  `v1_build.py` for the `recovery_*` columns.

Needs `data/slab2/cas_slab2_dep.xyz` (gitignored; a symlink to the main tree in the worktree used here).
Run: `v1_detect.py` (about 9 min on 5 processes), `v1_build.py` (2 min), `v1_report.py`. Python and steps:
`/wd1/mdenolle_data/catalog_v5/v5.1_analysis/faults/RESUME.md`.

## Files here

- `fault_surfaces_v1_summary.csv`: one row per surface: position, strike, dip, extent, intervals, sigmas, stability,
  tier and depth_quality mix, dip and depth range from depth_quality ok events only, v4 -> v5.1 shift of the members,
  weekday-daytime share and binomial p, flags, detection-level spread (`strike_settings_range_width`,
  `dip_settings_range_width`, `recovery_J_sub80`, `recovery_J_perturbed`, `recovery_J_tierAB`, `recovery_tests`; empty
  for the 12 strands, which the reruns do not repeat), change from v0, v0 surfaces, carried literature fields and
  `lit_status`.
- `fault_surfaces_v1_uncertainty.csv`: the uncertainty columns only (bootstrap, location Monte Carlo, combined,
  centroid, scatter against catalog error, stability, detection-level spread). The member-set intervals and the
  detection-level spread answer different questions; see "Read before use".
- `fault_surfaces_v1_rectangles.geojson`: each surface as its four corners (lon, lat, elevation m, negative down).
  Exact for planes, approximate for the 31 quadratic surfaces; the 1 km mesh is on /wd1.
- `fault_surfaces_v1_members.csv`: fit members (event_id = catalog ID, surface_id).
- `fault_surfaces_v1_assignment.csv.gz`: every located event: assignment (fit member, assigned, ambiguous, none, not
  fitted (Axial), not an earthquake, non-earthquake timing), surface_id, distance_km, z, n_compatible,
  on_surface_strict, nearest and second surface, tier, catalog flags, weekday_daytime, removed_surface, catalog_note,
  errors.
- `fault_surfaces_v1_assignment_by_region.csv`, `fault_surfaces_v1_selection.csv`, `fault_surfaces_v1_run.json`.
- `fault_surfaces_v1_removed.csv`, `fault_surfaces_v1_removed_members.csv`: screened-out surfaces (v1x_nn).
- `fault_surfaces_v1_strand_separation.csv` (overlapping parallel strands kept apart, with per-year D and depth test),
  `fault_surfaces_v1_strand_merges.csv` (pairs merged by the tests).
- `fault_surfaces_v1_traces.csv`: top edge and sea-level trace of each rectangle.
- `fault_surfaces_v1_v0_pairs.csv` (every linked v0-v1 pair with strike, dip, position changes),
  `fault_surfaces_v1_v0_status.csv` (every v0 surface: what became of it), `fault_surfaces_v1_v0_comparison.csv`
  (every v1 surface: change class, v0 sources, nearest v0, literature fields), `fault_surfaces_v1_literature_todo.csv`
  (the 39 new surfaces).
- Figures `fault_surfaces_v1_*.png` and the report.

## Files on /wd1/mdenolle_data/catalog_v5/v5.1_analysis/faults/

- `fault_surfaces_v1.geojson`: 1 km triangle mesh per surface (CRESCENT CFM layout), all summary fields as properties.
- `fault_surfaces_v1_vertices.csv.gz`: per vertex: lon, lat, depth_km, sigma_boot_km, sigma_loc_km, sigma_method_km,
  sigma_total_km, constrained, surface_elev_km (GMT earth_relief 01m).
- `fault_surfaces_v1_ensemble.csv.gz`: 100 draws (bootstrap + location Monte Carlo) of each vertex's offset along the
  surface's upward normal (`normal_xyz`).
- `work/`: detection runs (`detect_*.npz`), the first run without the dip rule (`nodipfix/`), logs, member residuals.

## Read before use

- **Uncertainty scope.** sigma_total (median 0.59 km) and the strike/dip intervals hold for a fixed set of members. The
  detection itself is far less certain: over the four detection reruns the strike of the best-matching plane ranges over
  a median 103 deg for the 93 unstable surfaces (6.5 deg for the 49 stable ones). Of the 130 non-strand surfaces, 38 are
  recovered (Jaccard >= 0.5) on a random 80 % of the fit set (27 of the 42 'stable'), 12 after moving every event once by
  its own catalog error (9 of 42 'stable'; pessimistic, since tier B-C errors exceed the scatter), 10 of 116 on tiers A-B
  only; 9 pass both the subsample and the perturbation tests. Columns `stable`, `strike_settings_range_width`,
  `recovery_J_*`, `recovery_tests` and the flags give this per surface. Do not use 0.59 km as the position uncertainty of
  a surface that fails these tests.
- **'assigned' is loose.** It means compatible with exactly one surface (z <= 2 and d <= 10 km), not located on it: 45 %
  of these events are tier D, 4,899 lie more than 5 km from the surface (median 4.0 km against 0.71 km for fit members),
  831 are floor_limited and 37 far_only. Count events on faults with `on_surface_strict`, and leave Axial out of any
  on/off-fault fraction.
- **Flags** (summary `flags`, rectangles and mesh geojson): depth poorly constrained; deep Blanco and Endeavour
  centroids; Mount St Helens deep population (v1_ON028); 2012 Brookings swarm (v1_GO022); members moved > 10 km in depth
  from v4 to v5.1 (17 surfaces); weekday-daytime excess (v1_MT031, v1_ON030); strand dips set by the band selection (12
  Blanco strands); rectangle above the surface; unstable; not recovered in the verifier's reruns. Flags are separated by
  "; ". Assignment column `catalog_note` marks the 331 Mount St Helens deep events and the 39 Brookings swarm events.
- **Literature** is carried unchanged to 15 surfaces (matched, normals within 25 deg, centroids within 10 km
  horizontally and 5 km in depth); 88 carry a match marked for re-check; 39 are new.
- **Dip rule cost.** Faults flatter than 30 deg cannot be represented: v0 MT010 (1992 Cape Mendocino / Honeydew
  low-angle thrust zone, 774 events, medium-high literature confidence) is dispersed over v1_MT007, MT001 and MT003;
  PU016 (Olympia structure) is lost.

## Known limits

- Blanco depths: the v5.1 deep tail (90th percentile 13-17 km deeper than the Lange et al. 2026 OBS catalog) sets the
  bottom of the Blanco rectangles (30-50 km); use `events_depth_ok_*` and `dip_depth_ok`.
- Cascadia Depression: the N-S normal faults of v0 are not recovered as N-S planes (detection order); a restricted fit
  is needed.
- 37 surfaces rest mostly on depth_quality unconstrained or biased_deep events; Endeavour-Explorer surfaces are centred
  at 30-60 km on unconstrained depths.
- Catalog problems found here, now in the v5.1 review: deep events under Mount St. Helens on a weekly Wednesday
  19:14-19:21 UTC repeat (analog UW channels HSR, EDM, STD carry an artificial signal 2011-05 to 2013-08; likely false
  events, pending human review; P1); the 2012 Brookings swarm at 42-65 km in v5.1 (about 20 km in v4; S-P not monotonic
  with depth in the v5.1 grids, NonLinLoc took the deep minimum; P2).
- Tier calibration: member residuals are truncated by the selection tolerance; expected median |residual|/error 0.67,
  0.60, 0.42 for tiers A, B, C against 0.77, 0.49, 0.30 observed (tier A errors about 15 % too small, or zones have
  width).
- No null test of the detection yet (detection on a spatially randomised catalog of the same density).
- No relative relocation; ComCat focal mechanisms not used (PI rule).

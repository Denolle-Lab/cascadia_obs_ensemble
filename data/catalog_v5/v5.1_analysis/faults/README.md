# Cascadia fault surfaces v1 (from catalog v5.1)

**Version v1, 2026-11-25. Private working product; not peer reviewed.** Report with figures, findings and proposed
decisions: `fault_surfaces_v1_report.html`.

v1 refits the plane-based fault surfaces of v0 (`data/faults/v0/`, catalog v4) on catalog v5.1: catalog v4.1 events
relocated with NonLinLoc in the 3D model v0.3.1, re-timed picks, a deep pass, one merged solid surface
(`/wd1/mdenolle_data/catalog_v5/v5.1/README_v5.1.md`). Depths are km below sea level.

## Content

| | |
|---|---|
| Fit set | 101,296 events: tiers A-C located by NonLinLoc, without floor_limited, far_only, deep_pass, ComCat non-earthquakes and Texada quarry_blast_likely events; Axial left out as in v0 (`fault_surfaces_v1_selection.csv`) |
| Surfaces | 142 (99 with dip >= 45 deg; 12 Blanco strands); 10 removed by screening (`fault_surfaces_v1_removed.csv`) |
| Fit members | 64,272 (63 % of the fit set; v0: 53,291, 70 %) |
| Every located event (171,719) | 64,272 fit members, 12,638 assigned to one surface, 26,266 ambiguous (two or more compatible surfaces), 64,093 on none, 3,313 non-earthquakes, 1,137 events of surfaces removed for their weekday-daytime timing. The 623 rows without a hypocentre are left out. |
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

Needs `data/slab2/cas_slab2_dep.xyz` (gitignored; a symlink to the main tree in the worktree used here).
Run: `v1_detect.py` (about 9 min on 5 processes), `v1_build.py` (2 min), `v1_report.py`. Python and steps:
`/wd1/mdenolle_data/catalog_v5/v5.1_analysis/faults/RESUME.md`.

## Files here

- `fault_surfaces_v1_summary.csv`: one row per surface: position, strike, dip, extent, intervals, sigmas, stability,
  tier and depth_quality mix, dip and depth range from depth_quality ok events only, v4 -> v5.1 shift of the members,
  weekday-daytime share, flags, change from v0, v0 surfaces, carried literature fields and `lit_status`.
- `fault_surfaces_v1_uncertainty.csv`: the uncertainty columns only (bootstrap, location Monte Carlo, combined,
  centroid, scatter against catalog error, stability).
- `fault_surfaces_v1_rectangles.geojson`: each surface as its four corners (lon, lat, elevation m, negative down).
  Exact for planes, approximate for the 31 quadratic surfaces; the 1 km mesh is on /wd1.
- `fault_surfaces_v1_members.csv`: fit members (event_id = catalog ID, surface_id).
- `fault_surfaces_v1_assignment.csv.gz`: every located event: assignment (fit member, assigned, ambiguous, none, not an
  earthquake, non-earthquake timing), surface_id, distance_km, z, n_compatible, nearest and second surface, tier,
  flags, errors.
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

## Known limits

- Blanco depths: the v5.1 deep tail (90th percentile 13-17 km deeper than the Lange et al. 2026 OBS catalog) sets the
  bottom of the Blanco rectangles (30-50 km); use `events_depth_ok_*` and `dip_depth_ok`.
- Cascadia Depression: the N-S normal faults of v0 are not recovered as N-S planes (detection order); a restricted fit
  is needed.
- 37 surfaces rest mostly on depth_quality unconstrained or biased_deep events; Endeavour-Explorer surfaces are centred
  at 30-60 km on unconstrained depths.
- Catalog problems found here and passed on: deep events under Mount St. Helens with a weekly Wednesday 19:15-19:18 UTC
  repeat; the 2012 Brookings swarm at 42-65 km in v5.1 (about 20 km in v4).
- No relative relocation; ComCat focal mechanisms not used (PI rule).

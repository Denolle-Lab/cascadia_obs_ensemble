# Cascadia fault surfaces v0 (from catalog v4 seismicity)

**Version v0, 2026-11-22.** Shared privately in the catalog v4 working-copy Zenodo draft
(files `fault_surfaces_v0*`). A first product on the catalog v4 locations; it will be redone
on the relocated catalog.

v0 fits rectangular fault surfaces to planar alignments of earthquakes in the Cascadia catalog
v4 (172,340 events, 2010–2016; ELEP ver6 picks, GENIE association and GraphDD location by
I. McBrearty). The criterion is that events gather around planes and that most events can be
assigned to a fault, whatever the local density.

Three limits of v4 carry into v0:
- **Depths** are biased shallow. 23.5% of v4 events plot above the ground or seafloor when
  depths are read below sea level, and 16.2% when they're read below the local surface.
- **Datum:** the depth datum is not confirmed. `surface_elev_km` is given at every vertex:
  depth below sea level = `depth_km - surface_elev_km` if the catalog depth is below the
  local surface.
- **Relocation:** there is no relative relocation, so dips are less reliable than strikes.

**Not included:**
- **Axial Seamount:** published fault maps from the cabled-array catalogs are much better than
  v4 allows (for example Wilcock et al. 2016; Waldhauser et al. 2020).
- **Flat structures** (dip < 30°): with v4 depths, a diffuse cloud of events passes for a flat
  plane.
- **Published focal mechanisms** (ComCat): they belong to other catalogs' locations and velocity
  models, so they are not used.

Next: v1 on the catalog relocated in a 3D velocity model, with an uncertainty for each event;
v2 adds focal mechanisms we compute ourselves on our own locations.

Ten surfaces found by the method were removed after the literature check (2026-11-22): six
quarry-blast clusters (ON012, ON020, ON023, ON029, ON037, ON040) and four likely artefacts (BL022,
BL024, BL026, PU010). They are listed with the reason in `fault_surfaces_v0_removed.csv`
(`workflow/07_faults/drop_surfaces.py`); their events count as unassigned.

A first v0, built the same day by density clustering (DBSCAN), kept only the densest knots
(20% of events) and has been replaced by this one. Its code is still in `fault_surfaces_v0.py`,
which also holds the helpers.

## Content

| | |
|---|---|
| Events used | 76,281: QC (≥ 4 P and ≥ 4 S), catalog depth ≥ 0, Axial left out |
| Events on a fault | 52,647 (69%) |
| Surfaces | 145: Blanco 24, Mendocino triple junction 32, Gorda 20, Puget 24, Washington offshore 5, Oregon offshore 2, Endeavour 4, onshore and other 34 |
| Steep (dip ≥ 45°) | 94 |
| Size (10th–90th percentiles) | length 13–54 km, down-dip width 10–27 km |
| Mesh | 1 km grid; 98,853 vertices, 182,880 triangles |

Share of events assigned to a fault, by region:

| Region | Events | On a fault |
|---|---|---|
| Blanco | 20,331 | 87% |
| Mendocino triple junction | 24,996 | 72% |
| Puget | 8,491 | 61% |
| Onshore and other | 14,913 | 60% |
| Gorda | 4,060 | 50% |
| Endeavour | 845 | 33% |
| Washington offshore | 1,906 | 21% |
| Oregon offshore | 739 | 19% |

Sparse offshore areas stay low: with few events, few planes reach the 30-event minimum.

The surfaces outline the Blanco segments and their parallel strands, the N–S structures in the
Cascadia Depression, the Gorda Ridge segments, the Mendocino transform, and steep faults in the
Puget Lowland and around Mount St. Helens.

## Method (`fault_surfaces_planes.py`)

1. **Tolerance.** A plane with upward normal n has thickness tolerance
   tol = 1.5 · sqrt((1.5 km · |n_h|)² + (3 km · n_z)²). The nominal relative errors are
   1.5 km horizontal and 3 km in depth, so vertical faults are about 2.3 km thick and
   flatter ones thicker. `spatial_sigma` is not used: it behaves like an absolute error
   (3–20 km, while events scatter 0.7–1.7 km about the fitted surfaces).
2. **Detection.** Sequential local RANSAC with growth:
   - Seeds are taken from dense to sparse.
   - Around a seed, 150 random planes through the seed and two events within 10 km are scored
     by the number of free events within tol in a 20 km ball.
   - The best plane grows: free events within tol are kept only if connected to the seed in
     the plane (4 km cells), then the plane is refit, and this repeats.
   - A plane is kept with ≥ 30 events, planarity sqrt(l3/l2) ≤ 0.5, in-plane width ≥ 1 km and
     dip ≥ 30°. Its events are then removed from the pool.
3. **Reassignment** (k-planes, three passes). Each event goes to the plane with the smallest
   |distance|/tol among planes whose rectangle contains its projection. Planes with fewer than
   30 events are dropped.
4. **Merging.** Two planes join when all of these hold:
   - normals within 15°;
   - each within tol of the other;
   - events closer than 4 km;
   - the joined set is still planar, with RMS ≤ tol and dip ≥ 30°.

   Then reassignment runs again.
5. **Rectangles.** Each fault is the rectangle [u2%, u98%] × [v2%, v98%] (+ 0.5 km) in its own
   plane (u along strike, v down dip), whether or not it reaches the surface. On it,
   w = f(u, v) is a plane or a gentle quadratic chosen by BIC. `constrained` marks vertices
   with ≥ 5 events within 2.5 km in the plane.
6. **Uncertainty, per vertex** (1σ normal to the surface, km):
   - `sigma_boot`: bootstrap of the fault's events (100 draws).
   - `sigma_loc`: Monte Carlo with each event moved by N(0, 1.5, 1.5, 3 km) in x, y, z
     (100 draws).
   - `sigma_method`: spread over the surface-model choices (plane or quadratic, with or without
     trimming beyond 2.5 RMS).
   - `sigma_total`: the three combined in quadrature. Median 0.44 km (10th–90th: 0.18–0.77 km).

   Strike and dip intervals (`*_68`, `*_95`) come from the bootstrap; `*_with_loc_mc` add the
   location Monte Carlo. For dips above 75°, strike intervals are modulo 180°.
7. **Stability.** Detection is rerun four times: two other random seeds, and tolerance factor
   1.25 and 1.75.
   - `cluster_jaccard_median` is the median overlap of a fault's events with its best match in
     the reruns; `stable` means ≥ 0.5 (39 of 145).
   - `strike_settings_range` and `dip_settings_range` give the orientation spread of the
     matches.
   - Overlap is low where parallel or adjacent planes trade events between runs (most of
     Blanco), even when the geometry barely moves. Compare the orientation ranges before
     discarding a surface.
8. **Checks.**
   - `rms_normal_km` against `tolerance_km`;
   - nearest CRESCENT CFM v1.0 fault (3D distance, its dip);
   - `slab2_minus_depth_km`;
   - `kind`: "steep zone (fault candidate)" for dip ≥ 45°, otherwise "dipping zone".

## Files

- `fault_surfaces_v0.geojson`: one Feature per fault. The geometry is a MultiPolygon of
  triangles (lon, lat, elevation in m, negative down), as in the CRESCENT Community Fault
  Model; the properties are the summary fields.
- `fault_surfaces_v0_summary.csv`: one row per fault, with list fields separated by ";".
  Includes top_km, bottom_km, length_km, width_km, strike, dip, their intervals, the sigmas,
  stability and checks.
- `fault_surfaces_v0_vertices.csv`: surface_id, vertex, lon, lat, depth_km, sigma_boot_km,
  sigma_loc_km, sigma_method_km, sigma_total_km, constrained, surface_elev_km (GMT
  earth_relief 01m, km, positive up). Vertices are row-major on the rectangle grid.
- `fault_surfaces_v0_ensemble.csv.gz`: surface_id, vertex, draw, normal_offset_km. These are
  100 draws (bootstrap + location Monte Carlo) of each vertex's offset along the fault's upward
  normal (`normal_xyz` in the summary; x east, y north, z down).
- `fault_surfaces_v0_members.csv`: event_id (catalog v4 `ID`) and surface_id.
- `fault_surfaces_v0_assignment.csv`: events and share on a fault per region.
- `fault_surfaces_v0_map.png`: events coloured by fault, and the rectangles, for the whole
  margin, Blanco, and Mendocino–Gorda.
- `fault_surfaces_v0_traces.csv`: per fault, the top edge of the rectangle and the line where the
  plane would reach sea level if extended up dip, for comparison with mapped traces.
- `fault_surfaces_v0_literature.csv`: the literature match of every fault: `lit_match`,
  `lit_confidence` (high, medium-high, medium, low-medium, low, none), `lit_class` (known structure,
  duplicate, seismicity layer or volume, no known counterpart, likely artefact, quarry blasts),
  `lit_new_candidate`, evidence and references. The same fields are in the summary and the GeoJSON.
- `fault_surfaces_v0_literature.html`: report of the matches (summary by region, strong matches,
  candidate new structures, blasts and artefacts), with the five regional notes
  (`literature/matches_*.md`: known structures, evidence per fault, references with DOI checks).
- On Zenodo, the code, the assignment and trace tables and the literature files are bundled in
  `fault_surfaces_v0_extras.zip` (the record is at Zenodo's 100-file limit).
- `fault_surfaces_planes.py` (the method) and `fault_surfaces_v0.py` (helpers, and the first
  DBSCAN version). They use NumPy, SciPy, pandas, matplotlib and PyGMT.

Region prefixes in `surface_id`:
- BL: Blanco
- MT: Mendocino triple junction
- GO: Gorda
- PU: Puget
- WA: Washington offshore
- OR: Oregon offshore
- EN: Endeavour–Explorer
- ON: onshore and other

## Licence

Working copy shared with collaborators; not for redistribution. Code:
https://github.com/Denolle-Lab/cascadia_obs_ensemble (`workflow/07_faults/`).

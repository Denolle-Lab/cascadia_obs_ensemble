# Cascadia fault surfaces v0 (from catalog v4 seismicity)

**Version v0, 2026-11-22. A method test, not for interpretation.** Zenodo: https://doi.org/10.5281/zenodo.24443891

v0 fits 3D surfaces to clusters of earthquakes in the Cascadia catalog v4 (172,340 events,
2010–2016, GENIE association and GraphDD location of the ELEP ver6 picks; catalog by
I. McBrearty). The aim is to build and test the fitting and uncertainty steps before the
catalog is relocated. The surfaces use the v4 locations as they are, and three things limit them:

- **Depths:** absolute depths are biased shallow. 23.5% of v4 events plot above the ground or
  seafloor when depths are read below sea level; 16.2% do when depths are read below the
  local surface.
- **Datum:** the depth datum of v4 is not confirmed. `surface_elev_km` is given at every vertex
  so the surfaces can be moved: depth below sea level = `depth_km - surface_elev_km` if the
  catalog depth turns out to be below the local surface.
- **Relocation:** v4 has no relative relocation, and its `spatial_sigma` is 3–20 km.

Later versions: v1 reruns the same code on the catalog relocated in a 3D velocity model,
with an uncertainty for each event; v2 adds focal mechanisms that we compute ourselves on our
own locations. Published mechanisms (ComCat) are not used, because they belong to other
catalogs' locations and velocity models.

## Content

| | |
|---|---|
| Surfaces | 74 (Blanco 23, Mendocino triple junction 20, Gorda 6, Axial 2, Puget 11, onshore/other 12) |
| Steep surfaces (dip ≥ 45°) | 43, of which 39 are stable under changes of the clustering settings |
| Events used | 35,279 (members of a surface) of 51,559 selected |
| Mesh | 27,480 vertices, 48,552 triangles, 1 km grid |

The Blanco transform comes out as 23 near-vertical surfaces: segments along strike, parallel
overlapping strands and a N–S pair in the Cascadia Depression step-over.

Near the triple junction, several gently dipping surfaces within 5 km of Slab2 are labelled
"near-slab layer (not a single fault)". They are pieces of one continuous seismicity layer,
and the clustering splits it differently from one setting to the next (`stable` = False).

## Method (`fault_surfaces_v0.py`)

1. **Events:** at least 4 P and 4 S picks, catalog depth ≥ 0, and `spatial_sigma` below the
   60th percentile of its region, capped at 20 km. The regional cuts are in
   `fault_surfaces_v0_sigma_cuts.csv`. A relative cut is used because `spatial_sigma` behaves
   like an absolute error: events scatter 0.6–1.8 km (RMS normal to the surface, 10th–90th
   percentile) about the surfaces, while the median `spatial_sigma` of their members is
   2.6–15 km.
2. **Clusters:** DBSCAN in 3D, with a 2.5 km link distance, 8 links for a core event, and depth
   weighted by 0.5.
3. **Recursive splitting:** each cluster is split until every piece is planar
   (sqrt(l3/l2) ≤ 0.5, where l1 ≥ l2 ≥ l3 are the variances along the principal axes) and
   shorter than 40 km. Each split tries two options and keeps the one with the smaller summed
   normal variance:
   - two planes (k-planes with k = 2), which separates parallel strands and step-overs;
   - a cut across strike at the median, which separates segments.
4. **Surface:** each piece is fitted in a local projection centred on it, in the frame of its
   best plane (u along strike, v down dip, w normal). The fit is w = f(u, v), a plane or a
   quadratic chosen by BIC, on a 1 km (u, v) grid inside the convex hull of the events with a
   1 km buffer.
5. **Uncertainty, per vertex** (normal to the surface, 1σ, km):
   - `sigma_boot`: bootstrap of the events (100 draws).
   - `sigma_loc`: Monte Carlo over location errors, with each event moved by N(0, spatial_sigma)
     on x, y and z (100 draws). Since `spatial_sigma` is larger than the observed scatter, this
     is a worst case.
   - `sigma_method`: spread over the surface-model choices (plane or quadratic, with or without
     trimming events beyond 2.5 RMS).
   - `sigma_total`: the three combined in quadrature.

   Velocity-model and datum effects are not included in v0. They move all surfaces together
   and will be measured as the shift between v0 and v1.
6. **Orientation:** `strike_68`/`_95` and `dip_68`/`_95` come from refitting the plane in every
   bootstrap draw. `*_with_loc_mc` adds the location Monte Carlo (worst case). For dips above
   75°, strike intervals are given modulo 180°.
7. **Clustering stability:** the whole pipeline is rerun over a 3 × 3 grid of DBSCAN settings
   (link distance 2, 2.5, 3 km; core 6, 8, 10). `cluster_jaccard_median` is the median overlap
   of the surface's events with the best-matching piece; `stable` means it is ≥ 0.5.
   `strike_settings_range` and `dip_settings_range` give the spread of the matched pieces'
   orientation.
8. **Checks:**
   - `rms_normal_km` (zone width) against `median_sigma_km`, as `width_to_error`;
   - nearest CRESCENT CFM v1.0 fault (`nearest_cfm`, 3D distance, its dip);
   - `slab2_minus_depth_km`, Slab2 depth minus the surface's centroid depth.
9. **Kind:**
   - "steep zone (fault candidate)": dip ≥ 45°;
   - "near-slab layer (not a single fault)": dip ≤ 25° and within 5 km of Slab2;
   - otherwise "dipping zone".

## Files

- `fault_surfaces_v0.geojson`: one Feature per surface. The geometry is a MultiPolygon of
  triangles (lon, lat, elevation in m, negative down), in the layout of the CRESCENT
  Community Fault Model. The properties are the summary fields.
- `fault_surfaces_v0_summary.csv`: one row per surface. List fields are separated by ";".
- `fault_surfaces_v0_vertices.csv`: surface_id, vertex, lon, lat, depth_km, sigma_boot_km,
  sigma_loc_km, sigma_method_km, sigma_total_km, constrained (1 if at least 5 events lie
  within 2.5 km in the surface plane), surface_elev_km (GMT earth_relief 01m, km, positive up).
- `fault_surfaces_v0_ensemble.csv.gz`: the full ensemble (bootstrap plus location Monte Carlo,
  100 draws). Columns: surface_id, vertex, draw, normal_offset_km, the offset of the vertex
  along the surface's upward normal (`normal_xyz` in the summary; x east, y north, z down).
- `fault_surfaces_v0_members.csv`: event_id (catalog v4 `ID`) and surface_id.
- `fault_surfaces_v0_sigma_cuts.csv`: the `spatial_sigma` cut of each region.
- `fault_surfaces_v0_map.png`: map of the surfaces, for the whole margin, Blanco, and
  Mendocino–Gorda.
- `fault_surfaces_v0.py`: the code that made all of the above (uses NumPy, SciPy, pandas,
  matplotlib and PyGMT).

Region prefixes in `surface_id`:
- AX: Axial Seamount
- BL: Blanco transform
- GO: Gorda plate
- MT: Mendocino triple junction
- PU: Puget Lowland
- ON: onshore and other

## Licence and citation

CC BY 4.0. Please cite this record and the catalog v4 record. Code:
https://github.com/Denolle-Lab/cascadia_obs_ensemble (`workflow/07_faults/`).

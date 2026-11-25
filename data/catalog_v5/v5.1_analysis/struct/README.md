# Velocity structure and seismicity on catalog v5.1 (piece "struct")

2026-11-25. Private working product; not peer reviewed. Report with figures, numbers and proposed decisions:
`v51_structure_seismicity_report.html`.

Model v0.3.1 (`/wd1/mdenolle_data/tomo_v1/models/cascadia_obs_tomo/v0.3.1.nc`), resolved nodes only: Vp where
resolution_p > 0, Vs where resolution_s > 0 and vs_reverted == 0, Vp/Vs where both (D-B018, D-B049, D-B051). The
D-B073 shallow-crust rule is not used (it is not in v0.3.1); the v0.3.2 `medium` flag is read only to check that no
resolved node is in water or air (none is). Catalog v5.1 located earthquakes (623 rows without a hypocentre and 3,313
non-earthquakes left out), v5.1 magnitudes (mag piece), v5.1 clustering (clusters piece, 526d52bb), fault surfaces v1
(faults piece, 6be76cfd).

Code: `workflow/06_analysis/v51_struct_common.py`, `v51_struct.py` (steps prep, coverage, t1-t5, synth, mc, v4),
`v51_struct_report.py` and `v51_struct_text.py` (figures, HTML). Resume notes and the run script:
`/wd1/mdenolle_data/catalog_v5/v5.1_analysis/struct/RESUME.md`.

## Files

| file | content |
|---|---|
| coverage_fraction.csv, coverage_by_zone_depth.csv, coverage_region_depthband.csv | events in resolved cells by group, zone x depth node (with resolved area and number of independent checkerboard cells), region x depth band |
| nodes_resolved_v031.csv | every node resolved for Vp or Vs: values, changes, DWS, strata, coverage-matched anomalies (`*_res`) |
| events_in_resolved_cells.csv.gz | events in resolved cells with the sampled values |
| t1_swarm_vs_ma.csv, t1_families_*.csv, t1_event_swarm_share.csv | Q1: family types against Vp/Vs, Vp and Vs (sets resPS, resPS_dg, resPS_strict, resP, ...) |
| t2_density.csv, t2_bvalue.csv | Q2: count-velocity correlations within strata, event-weighted anomalies, randomized-catalog nulls, b-value splits |
| t3_columns.csv, t3_depth_extent.csv, t3_slab_position_by_box.csv | Q3: depth extent per model column against Vp and Slab2 |
| t4_surfaces.csv, t4_pooled.csv | Q4: fault surfaces v1 against horizontal velocity gradients (alignment, gradient ratio) |
| t5_blanco_segments.csv, t5_blanco_segment_corr.csv, t5_slab_position_dvp.csv, t5_areas_dvp.csv | Q5: Blanco segments, slab position, Mount St. Helens and Mendocino |
| synth_artifact.csv, synth_artifact_summary.csv | the same statistics on the tiles' checkerboard inversion errors |
| mc_location.csv, mc_location_summary.csv | location Monte Carlo of the headline statistics |
| v4_counterpart_external_vs.csv | the earlier phase13 statistic (external Vs models) on v4 and v5.1 hypocentres |
| tests_summary.csv | every test with effect, 95% interval, null range, p and Benjamini-Hochberg q |
| fig_*.png | report figures |

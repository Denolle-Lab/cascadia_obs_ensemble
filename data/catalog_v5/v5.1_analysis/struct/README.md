# Velocity structure and seismicity on catalog v5.1 (piece "struct")

2026-11-25, revised the same day after an adversarial review. Private working product; not peer reviewed. Report
with figures, numbers and proposed decisions: `v51_structure_seismicity_report.html`.

Model v0.3.1 (`/wd1/mdenolle_data/tomo_v1/models/cascadia_obs_tomo/v0.3.1.nc`), resolved nodes only (D-B018,
D-B049, D-B051): Vp where resolution_p > 0; Vs and Vp/Vs both on the nodes resolved for both phases (resolution_p >
0, resolution_s > 0 and vs_reverted == 0), so the Vs and Vp/Vs tests use the same nodes and strata. The 55 nodes
resolved for S but not P (591 located earthquakes) are not used. The D-B073 shallow-crust rule is not used (it is not in v0.3.1); the v0.3.2 `medium` flag is read only to check that no
resolved node is in water or air (none is). Catalog v5.1 located earthquakes (623 rows without a hypocentre and 3,313
non-earthquakes left out), v5.1 magnitudes (mag piece), v5.1 clustering (clusters piece) and fault surfaces v1
(faults piece), read from the sibling worktrees after their review fixes; the commits and checksums of every input
are in `input_commits.json`. The waveform-review list of four v5.1 location problems
(`/wd1/mdenolle_data/catalog_v5/v5.1_review/review_v51_events.csv`) is read only to flag events.

Superseded later: model v0.4.0 was adopted (D-B082), v0.4.1 is being built (D-B083) and catalog v6.0 (D-B079)
will replace v5.1; these products are to be redone on v6.0.

What the review changed (details in the report): the Washington background-event b-value split by Vp passes the
pass rule (the first version said no b-value difference anywhere); the Q1 null result is stated from the bootstrap
interval (half-width 0.09), not a 0.06 "detection limit"; the Blanco node-scale Q2 result has four new controls (v4
hypocentres, occupancy, block-grid placement, tile runs with other smoothing and re-timed picks); tests_summary.csv
has a primary and a variant family and a `signal` column.

Code: `workflow/06_analysis/v51_struct_common.py`, `v51_struct.py` (steps prep, coverage, t1-t5, synth, mc, v4hypo,
blocks, tiles, v4), `v51_struct_report.py` and `v51_struct_text.py` (figures, HTML). Resume notes and the run script:
`/wd1/mdenolle_data/catalog_v5/v5.1_analysis/struct/RESUME.md`, `run_all.sh`.

## Files

| file | content |
|---|---|
| coverage_fraction.csv, coverage_by_zone_depth.csv, coverage_region_depthband.csv | events in resolved cells by group, zone x depth node (with resolved area and number of independent checkerboard cells), region x depth band |
| flags_resolved_by_zone.csv | catalog flags (tier D, depth_quality, far_only, ...) and waveform-review sets of the events in resolved cells |
| nodes_resolved_v031.csv | every node resolved for P or S (masks resP, resS, resPS; Vs statistics use resPS): values, changes, DWS, strata, coverage-matched anomalies (`*_res`) |
| events_in_resolved_cells.csv.gz | events in resolved cells with the sampled values, flags and review labels |
| t1_swarm_vs_ma.csv, t1_families_*.csv, t1_event_swarm_share.csv | Q1: family types against Vp/Vs, Vp and Vs (sets resPS, resPS_dg, resPS_strict, resP, ...) |
| t2_density.csv, t2_bvalue.csv, t2_bvalue_checks.csv | Q2: count-velocity correlations within strata, event-weighted and occupancy anomalies, randomized-catalog nulls; b-value splits and their checks by box, node depth and Mc + 0.2 |
| t2_v4hypo.csv | Q2 control: catalog v4 hypocentres (and mixed positions) of the same events in the same model |
| t2_block_offsets.csv, t2_block_offsets_summary.csv | Q2 control: Blanco bootstrap intervals for 16 placements of 40 and 60 km block grids |
| t2_tile_runs.csv | Q2 control: the Blanco statistic on tile inversions with smoothing 20/60/120 and with re-timed picks (v0.4.0's Blanco tile) |
| t3_columns.csv, t3_depth_extent.csv, t3_slab_position_by_box.csv | Q3: depth extent per model column against Vp and Slab2 (with review P1/P2 counts and a variant without them) |
| t4_surfaces.csv, t4_pooled.csv | Q4: fault surfaces v1 against horizontal velocity gradients (alignment, gradient ratio), with the surfaces' flags |
| t5_blanco_segments.csv, t5_blanco_segment_corr.csv, t5_slab_position_dvp.csv, t5_areas_dvp.csv | Q5: Blanco segments, slab position, Mount St. Helens and Mendocino |
| synth_artifact.csv, synth_artifact_summary.csv | the same statistics on the tiles' checkerboard inversion errors |
| mc_location.csv, mc_location_summary.csv | location Monte Carlo of the headline statistics |
| v4_counterpart_external_vs.csv | the earlier phase13 statistic (external Vs models) on v4 and v5.1 hypocentres, with depth medians of all events of each region |
| tests_summary.csv | every test with effect, 95% interval, null range, p, family (primary / variant), signal, q within the primary family (q_BH) and over all (q_BH_all), passed / passed_all |
| input_commits.json | sibling worktree commits and size / mtime / md5 of every input file |
| fig_*.png | report figures |

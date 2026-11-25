# Catalog v5.1 clusters (nearest-neighbour method and swarm scan)

Private working product, 2026-11-25; not peer reviewed. Report with figures, method, comparison with v4 and
limits: `catalog_v51_clusters_report.html` (self-contained).

Catalog v5.1 (`/wd1/mdenolle_data/catalog_v5/v5.1/cascadia_catalog_v5.1.csv`) with the v5.1 Route A magnitudes
(`/wd1/mdenolle_data/catalog_v5/v5.1_analysis/mag/cascadia_catalog_v5.1_magnitudes.csv`, column M). Same method,
parameters and classification as the catalog v4 analysis (`data/catalog_v4/clusters_zbz/`,
`workflow/06_analysis/v4_clusters_zbz.py`), plus per-family magnitude range, depth range from well-constrained depths,
migration (radial speed, velocity vector and azimuth, depth migration) and a sqrt(t) diffusion fit.

Events used: the 168,406 located earthquakes (rows with loc_source none, 623, and the 3,313 events with
is_earthquake False are left out and listed with role `excluded`). floor_limited / far_only / deep_pass events enter
through their epicentres (epicentral distance); their depths are not used.

Code (`workflow/06_analysis/`): `v51_clusters_zbz.py` (steps main, shuffle, variants, faults, compare),
`v51_swarms.py` (DBSCAN swarm scan of `v4_swarms.py` and its v4 comparison), `v51_clusters_report.py` (figures and
report). Python, steps and logs: `/wd1/mdenolle_data/catalog_v5/v5.1_analysis/clusters/RESUME.md`.

| file | content |
|---|---|
| `events_zbz_v51.csv.gz` | every catalog row: region box, in_analysis, M, parent_id, log_eta, log_T, log_R, nn_dist_km, nn_dt_days, linked, family_id, family_type, role, background, cls, excluded. The full table (plus time, lat, lon, depth, tier, flags) is on /wd1 (`.../v5.1_analysis/clusters/events_zbz_v51.csv.gz`) |
| `families_zbz_v51.csv` | every family: type, n (above Mc), n_all, Mmin_all-Mmax, start/end, durations, extent, depth_good ranges, swarm indicators S1-S3, tree topology, v4-style migration, v5.1 migration (mig_*), D90/D50 (m2/s), envelope exponent env_p, mig_resolvable, mig_migrating, mig_style, fault fractions (surfaces v1), v4_status |
| `regions_zbz.csv`, `thresholds_zbz.csv`, `completeness_b_df.csv`, `completeness_by_year.csv` | per-region summary, mixture fits and thresholds, Mc and b |
| `sensitivity_zbz.csv` | per-region summary for each variant (per-variant tables on /wd1 `variants/`) |
| `shuffle_test.csv`, `shuffle_eta_seed0.npz` | time-shuffled catalogs |
| `v4_families_status.csv`, `v51_families_status.csv`, `v4_v51_family_overlap.csv` | family matching with v4 (kept, split, merged, grown, dissolved, new) |
| `v4_vs_v51_event_classes.csv`, `v4_vs_v51_event_classes_analysis_by_region.csv`, `v4_vs_v51_analysis_membership.csv`, `v4_dissolved_members_fate.csv` | event-level comparison with v4 |
| `dbscan_clusters_v51.csv`, `dbscan_members_v51.csv`, `dbscan_v4_status.csv`, `dbscan_v51_status.csv` | DBSCAN swarm scan and its matching with the v4 scan |
| `validation_sequences.csv`, `volcano_families.csv`, `volcano_local_zbz.csv`, `volcano_local_families.csv`, `setting_stats.csv` | validation and setting tables |
| `faults_v1_cluster_counts.csv`, `faults_v1_daytime_surfaces_events.csv` | join with the fault surfaces v1 (piece faults; rerun `--steps faults,compare` if they change) |
| `fig_*.png` | report figures |

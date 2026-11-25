# Catalog v5.1 clusters (nearest-neighbour method and swarm scan)

Private working product, 2026-11-25; not peer reviewed; revised the same day after an independent review. Report
with figures, method, comparison with v4, null test and limits: `catalog_v51_clusters_report.html` (self-contained).
Catalog v6.0 (D-B079) will supersede v5.1; these products will be redone on it.

Catalog v5.1 (`/wd1/mdenolle_data/catalog_v5/v5.1/cascadia_catalog_v5.1.csv`) with the v5.1 Route A magnitudes
(`/wd1/mdenolle_data/catalog_v5/v5.1_analysis/mag/cascadia_catalog_v5.1_magnitudes.csv`, column M). Same method,
parameters and classification as the catalog v4 analysis (`data/catalog_v4/clusters_zbz/`,
`workflow/06_analysis/v4_clusters_zbz.py`), plus per-family magnitude range, depth range from well-constrained depths,
migration (radial speed, velocity vector and azimuth, depth migration; permutation p-value) and a sqrt(t) diffusion
fit. Fault surfaces v1 and the mag piece's Mc_best are read from the pushed branches analysis/v51-faults and
analysis/v51-mag (commits in `input_commits.json`).

Events used: the 168,406 located earthquakes (rows with loc_source none, 623, and the 3,313 events with
is_earthquake False are left out and listed with role `excluded`). floor_limited / far_only / deep_pass events enter
through their epicentres (epicentral distance); their depths are not used in the main run or in any depth statistic.
Columns that use every depth are labelled: `depth_main_any_quality` (families; see `depth_main_good`) and
`depth_any_quality_med` / `_iqr` (DBSCAN clusters); variant d3 uses every depth as a sensitivity test.

Read before use:
- Time-shuffled catalogs at the real thresholds reproduce 21-75% of the links (75% at Axial, 48% offshore Oregon,
  42% at the Mendocino triple junction, 39% on the Blanco), and within-month shuffles 62-92%. Ashman D is not a test
  of clustering (`null_test_summary.csv`).
- The Axial background share ranges from 38% to 82% across the variants: give the clustered share as a range.
- Tier D epicentres are in the analysis set; their median h_max is 4.8 km at Axial but 9-45 km elsewhere, and 16% of
  links have a tier D end (`tierD_by_region.csv`). The 2014 Vancouver Island family absorbs a tier D M 5.5 from 2013.
- Families with Mount St Helens deep events (review P1, probably false events) carry `n_review_P1_msh_deep` > 0;
  10 of those 32 families start on a Wednesday at local noon.
- On-fault fractions use the faults piece's `on_surface_strict` and leave Axial (not fitted) out.

Code (`workflow/06_analysis/`): `v51_clusters_zbz.py` (steps main, shuffle, variants, faults, nulls, checks,
compare), `v51_swarms.py` (DBSCAN swarm scan of `v4_swarms.py` and its v4 comparison), `v51_clusters_report.py`
(figures and report). Python, steps and logs: `/wd1/mdenolle_data/catalog_v5/v5.1_analysis/clusters/RESUME.md`.

| file | content |
|---|---|
| `events_zbz_v51.csv.gz` | every catalog row: region box, in_analysis, M, parent_id, log_eta, log_T, log_R, nn_dist_km, nn_dt_days, linked, family_id, family_type, role, background, cls, excluded. The full table (plus time, lat, lon, depth, tier, flags) is on /wd1 (`.../v5.1_analysis/clusters/events_zbz_v51.csv.gz`) |
| `families_zbz_v51.csv` | every family: type, n (above Mc), n_all, Mmin_all-Mmax, start/end, durations, extent, depth_good ranges, depth_main_any_quality and depth_main_good, swarm indicators S1-S3, tree topology, v4-style migration, v5.1 migration (mig_*; mig_p Spearman, mig_p_perm permutation; mig_migrating uses mig_p_perm, mig_migrating_spearman_p the first build's test), D90/D50 (m2/s), envelope exponent env_p, mig_resolvable, mig_style, fault fractions (frac_on_faults strict, frac_on_faults_loose, frac_not_fitted), flags n_daytime_screened, n_review_P1_msh_deep, n_review_P2_brookings, v4_status |
| `regions_zbz.csv`, `thresholds_zbz.csv`, `completeness_b_df.csv`, `completeness_by_year.csv` | per-region summary, mixture fits and thresholds, Mc and b. b and n_above of `completeness_b_df.csv` come from the v4 function, which compares M with the unrounded Mc, so they leave out the 61 events with M exactly at Mc that the analysis set (n_analysis) includes; b would be at most 0.005 higher, less than b_err; kept as in v4 |
| `sensitivity_zbz.csv` | per-region summary for each variant (per-variant tables on /wd1 `variants/`) |
| `shuffle_test.csv`, `shuffle_eta_seed0.npz` | three time-shuffled catalogs (conservative threshold eta0_shuf5, figures) |
| `null_test.csv`, `null_test_summary.csv` | null test: shuffles within region (10), region-year (5), region-month (5) at the real thresholds: linked fraction, D, KDE valley, families, swarms, ratios to the real catalog |
| `tierD_by_region.csv` | tier D share, h_max of tier D, tier A-C and five-phase events, background shares, links with a tier D end |
| `mc_edge_check.csv` | Mc with the v4 floating-point bin edges and with exact decimal edges (BL and OR differ) |
| `depth_swarm_vs_ma_families.csv` | swarm against mainshock-aftershock depth, families as the unit (Mann-Whitney, Bonferroni), with and without review sets P1 and P3 |
| `review_sets_in_clusters.csv`, `review_P1_families.csv` | waveform-review sets P1-P4 in the clustering; families with Mount St Helens deep events (weekday, local time) |
| `v4_families_status.csv`, `v51_families_status.csv`, `v4_v51_family_overlap.csv` | family matching with v4 (kept, split, merged, grown, dissolved, new) |
| `v4_vs_v51_event_classes.csv`, `v4_vs_v51_event_classes_analysis_by_region.csv`, `v4_vs_v51_analysis_membership.csv`, `v4_dissolved_members_fate.csv` | event-level comparison with v4 |
| `dbscan_clusters_v51.csv`, `dbscan_members_v51.csv`, `dbscan_v4_status.csv`, `dbscan_v51_status.csv` | DBSCAN swarm scan and its matching with the v4 scan |
| `validation_sequences.csv`, `volcano_families.csv`, `volcano_local_zbz.csv`, `volcano_local_families.csv`, `setting_stats.csv` | validation (with the largest other member of each known family and the tierABC family) and setting tables; volcano zoom with 10 shuffles. The zoom's local Mc is rounded to 0.01 and applied as in the main run (M >= Mc); the v4 zoom compared M with the unrounded value and left out the events with M exactly at Mc (3 at Mount St Helens, 1 at Hood, 3 at Rainier) |
| `faults_v1_cluster_counts.csv`, `faults_v1_daytime_surfaces_events.csv` | join with the fault surfaces v1 (strict membership; swarm concentration per surface; rerun `--steps faults` and the report if they change) |
| `input_commits.json` | commits of the other pieces' products read by the run |
| `fig_*.png` | report figures |

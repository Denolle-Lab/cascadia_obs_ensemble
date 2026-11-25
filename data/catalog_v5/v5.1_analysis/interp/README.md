# Catalog v5.1 re-interpretation (piece "interp")

Private working product, 2026-11-25; not peer reviewed; revised the same day after an independent review. Region-by-region
re-interpretation of catalog v5.1 seismicity, fault surfaces v1 and swarm/sequence behaviour, compared with the v4-era interpretations.

Catalog v6.0 (D-B079, NonLinLoc with SSST station terms) will supersede v5.1 and these products will be redone on it; model v0.4.0
is adopted (D-B082) and v0.4.1 is being built (D-B083). The waveform review of four v5.1 location problems
(`/wd1/mdenolle_data/catalog_v5/v5.1_review/index.html`: P1 Mount St. Helens deep events, P2 Brookings 2012, P3 tensor events,
P4 deep-pass and far_only events) is cited where it applies.

Revision after review (same day):
- The Axial "3 km node pile-up" is withdrawn (the first metric measured the peak of the depth mode; a local spike test finds no
  spike at 3.0 km). Depth spikes occur at integer km whether or not the depth is a model node (`t8_depth_spikes*.csv`,
  `figs/fig_depth_spikes.png`); the likely cause is the 1 km lattice of the NonLinLoc grids. `t8_node_pileups.csv` is removed.
- Mount St. Helens deep events are binned in local time: the Wednesday cluster follows Pacific time in PST and PDT, and the review's
  19:14-19:21 UTC detections are all on daylight-time Wednesdays (`msh_weekly.json`).
- The blast-like onshore population is sized by its weekday-daytime excess by depth (`region_extras.json`, `daytime_by_depth`).
- Brookings 2012 (v1_GO022) against Slab2 at the members' epicentres (about 10 km; `region_extras.json`, `brookings`).
- Migration: tests counted and Bonferroni-corrected (`region_extras.json`, `migration`). Gorda against ANSS quoted as +7 to +9 km
  (second fixed-depth rule in `t6_anss_depth.csv`).
- Inputs are the revised mag, faults and clusters products; every number taken from them is updated (null test, variants,
  on_surface_strict, faults flags, phase19 convergence, paired b).

Report: `catalog_v51_interpretation.html` (self-contained, figures embedded).

Inputs (read only): catalog v5.1 (`/wd1/mdenolle_data/catalog_v5/v5.1/`), pieces mag (analysis/v51-mag cd0334d6),
faults (analysis/v51-faults 6ebe0230) and clusters (analysis/v51-clusters f904659b), all three revised after review, the
verifiers' tables on `/wd1/mdenolle_data/catalog_v5/v5.1_analysis/verify_*`, the waveform review's weekly scan
(`/wd1/mdenolle_data/catalog_v5/v5.1_review/work/p1_cont/weekly_scan_rows.csv`), the v4 science report and v4 clusters
(`data/catalog_v4/`), fault surfaces v0 and their literature matches (`data/faults/v0/`), the depth-literature report and
`paired_v5.1.md`.

Code (`workflow/06_analysis/`): `v51_interp_common.py` (loaders), `v51_interp_tables.py` (tables, joined event table),
`v51_interp_figs.py` (figures), `v51_interp_report.py` (HTML). Run in that order (about 1 minute in all).

| file | content |
|---|---|
| `t1_locations.csv` | per catalog region: counts, tiers, flags, v4 and v5.1 depths, shifts, errors |
| `t2_sharpness.csv` | nearest-neighbour distances on the same events (v4 vs v5.1), shared-epicentre share (oct-tree lattice) |
| `t3_clusters.csv`, `t3b_boxes.csv` | clustering classes by region (v5.1, v4); box-level variants (with the range over all variants), the clustering piece's shuffled/real link ratios (whole period, year, month), shuffled swarm counts and tier D h_max |
| `t4_faults.csv`, `surfaces_v1_interp.csv` | v1 surfaces by region (on_surface_strict and loose shares) with the recovery Jaccard, the faults piece's flags (`flt_flags`, `flt_flags_short`) and notes added here (`interp_flags`) |
| `families_v51_setting.csv` | clustering families with nearest PB2002 step and volcano, v4 family type |
| `t6_anss_depth.csv` | depth minus ANSS/ComCat (fixed depths removed; `_strict` columns drop every round ANSS depth), v4 and v5.1, by region |
| `t7_benchmarks.csv` | named depth benchmarks (boxes defined in the tables script) against literature values |
| `t8_depth_spikes.csv`, `t8_depth_spikes_summary.csv` | sharp-spike ratio (events within 0.1 km against the bins 0.3 km either side) at integer km (model nodes and other integers) and half-integers, by region |
| `blanco_segments.csv`, `region_extras.json`, `msh_weekly.json` | segment, region, Brookings, blast-excess, migration and Mount St. Helens deep-event (UTC and local time) numbers used in the text |
| `figs/` | report figures |

Large files on `/wd1/mdenolle_data/catalog_v5/v5.1_analysis/interp/`: `events_joined.pkl` (catalog + v5.1 M + cluster class +
fault assignment + nearest PB2002 step; Axial 'none' relabelled 'region not fitted (Axial)'), `anss_pairs.csv.gz`, `coast_i.txt`.

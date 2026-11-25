# Catalog v5.1 re-interpretation (piece "interp")

Private working product, 2026-11-25; not peer reviewed. Region-by-region re-interpretation of catalog v5.1 seismicity,
fault surfaces v1 and swarm/sequence behaviour, compared with the v4-era interpretations.

Report: `catalog_v51_interpretation.html` (self-contained, figures embedded).

Inputs (read only): catalog v5.1 (`/wd1/mdenolle_data/catalog_v5/v5.1/`), pieces mag (analysis/v51-mag aebcbf9a),
faults (analysis/v51-faults 6be76cfd) and clusters (analysis/v51-clusters 526d52bb), the three verifiers' tables on
`/wd1/mdenolle_data/catalog_v5/v5.1_analysis/verify_*`, the v4 science report and v4 clusters (`data/catalog_v4/`),
fault surfaces v0 and their literature matches (`data/faults/v0/`), the depth-literature report and `paired_v5.1.md`.

Code (`workflow/06_analysis/`): `v51_interp_common.py` (loaders), `v51_interp_tables.py` (tables, joined event table),
`v51_interp_figs.py` (figures), `v51_interp_report.py` (HTML). Run in that order (about 1 minute in all).

| file | content |
|---|---|
| `t1_locations.csv` | per catalog region: counts, tiers, flags, v4 and v5.1 depths, shifts, errors |
| `t2_sharpness.csv` | nearest-neighbour distances on the same events (v4 vs v5.1), shared-epicentre share (oct-tree lattice) |
| `t3_clusters.csv`, `t3b_boxes.csv` | clustering classes by region (v5.1, v4); box-level variants and the verifier's shuffled/real link ratio |
| `t4_faults.csv`, `surfaces_v1_interp.csv` | v1 surfaces by region with the verifier's recovery Jaccard and flags added here (`interp_flags`) |
| `families_v51_setting.csv` | clustering families with nearest PB2002 step and volcano, v4 family type |
| `t6_anss_depth.csv` | depth minus ANSS/ComCat (fixed depths removed), v4 and v5.1, by region |
| `t7_benchmarks.csv` | named depth benchmarks (boxes defined in the tables script) against literature values |
| `t8_node_pileups.csv` | events piled on depth nodes of model v0.3.1 |
| `blanco_segments.csv`, `region_extras.json`, `msh_weekly.json` | segment, region and Mount St. Helens deep-event numbers used in the text |
| `figs/` | report figures |

Large files on `/wd1/mdenolle_data/catalog_v5/v5.1_analysis/interp/`: `events_joined.pkl` (catalog + v5.1 M + cluster class +
fault assignment + nearest PB2002 step; Axial 'none' relabelled 'region not fitted (Axial)'), `anss_pairs.csv.gz`, `coast_i.txt`.

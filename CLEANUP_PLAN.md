# Repository cleanup and Zenodo release plan

Status (2026-09-30, evening): on branch `cleanup/safe-steps`.
- Done: safe steps of §8.1 (tags, PR #22, obsolete trees removed, README, PROVENANCE).
- Done: picking consolidated (§4): `1_picking/elep_picker.py` + `run_picking.py` +
  `picking_config.csv` (29 runs) replace the 29 per-year scripts and 6 utils, now in
  `1_picking/legacy/`; reproduces stored per-day picks of both picker runs (v1, v2).
- Done: QC threshold decided: the paper now says at least 5 P and 5 S picks, the rule of
  the 31,020-event file behind every figure.
- Done: data products (§6) are three ANSS-style tables, `utils/build_anss_tables.py`
  (origins / arrivals / picks, QuakeML split), `data/DATA_DICTIONARY.md`, and the Zenodo
  layout in `utils/assemble_zenodo.py`.
- Waiting for `rerun_hp05/DONE`: the layout move of §2-3.

Goal: the repo holds only the workflow behind the paper, in run order, with one script per
paper figure, the at-scale launchers kept apart, a README that reproduces the paper, and a
Zenodo release of the code and the data products.

**Hard constraint while the 0.5 Hz amplitude test runs** (`4_relocation/magnitude/rerun_hp05/`,
started 2026-09-30): do not move, rename or delete anything in `4_relocation/magnitude/`.
The driver calls the stage-3 scripts by relative path when stage 1 finishes. Every step below
that touches that directory waits for `rerun_hp05/DONE`.

## 1. What stays, what goes

GENIE (association) and GraphDD (relocation) were run outside this repo. No code of theirs is
here; the repo only reads their output files. That stays as it is. The README will link their
repositories and name the versions and commits used, and the data record will label their
outputs.

| Keep (the workflow) | Remove from the main tree |
|---|---|
| `1_picking/` ELEP ensemble picking | `old/`: 279 MB of abandoned PyOcto, HypoInverse and old figures, all tracked |
| `4_relocation/cross_correlation/` CC waveform sets (GraphDD input) | `3_post_processing/`, a duplicate of `4_relocation/` (identical or diverged copies) |
| `4_relocation/merge_events_*`, `quality_control/4_quality_control_*` (the final QC catalog) | `0_data_availability/` (no paper figure) |
| `4_relocation/magnitude/` Route A stages, phase18/19, analysis phases | `phase6_catalog_comparison.py`, `phase15_magnitude_diagnostic.py`, `METHODS_route_b.md` (superseded) |
| `4_relocation/calculate_amplitudes.py` + `phase1` (counts baseline for Fig. S8a–d) | `data/*_old.csv`, `data/split_files/*_old.csv` (117 MB tracked) |
| `figures/` notebooks for Figs 1, 2, 3, 4, 8 | `figures/fig4_match_events.ipynb` and `paper/figures/fig4_cc_no_relief_tremor_contours.png` (no longer used) |
| `utils/` (`data_client`, `paper_style`, `plot_utils`, `qc_utils`, `fetch_*`, `assemble_zenodo`) | `paper/template.tex`, `amplitude_run.log`, `3_post_processing/pnwstore/` (empty) |
| `paper/`, `data/` small inputs, `LINEAGE.md` | `data/coupling/{__MACOSX,Humboldt_*,materna2023.zip}` (untracked leftovers) |

Before deleting anything, tag the current `main` as `pre-cleanup-2026-09` and push the tag, so
every removed file stays recoverable from GitHub. Diagnostic notebooks (`examine_*`,
`verify_*`, `qc_metrics_*`, `plot_picks_*`) go to `notebooks/diagnostics/` with their outputs
stripped (`nbstripout`). They explain choices but produce no paper figure.

## 2. Target layout

```
README.md  LICENSE  CITATION.cff  pixi.toml  pixi.lock  Makefile
workflow/                      # run order = directory order
  01_picking/                  # one parameterized ELEP script (replaces 25 per-year copies) + picking_utils
  02_association/README.md     # GENIE: external repo, version, the command used, output files
  03_relocation/               # GraphDD: external (README); our CC dataset builders live here
  04_merge_qc/                 # merge GraphDD+CC output; QC thresholds -> final catalog
  05_magnitude/                # Route A: inventory, amplitudes, dataset, inversion, phase18, phase19
  06_analysis/                 # classification (phase10) and the analysis behind the supplement
figures/                       # ONE entry point per paper figure, named by figure number
  fig01_station_map.ipynb ... figS09_completeness.py
  make_figures.py              # replaces paper/export_figures.py: runs each, copies to paper/figures
scale/                         # at-scale launchers (not needed to redraw figures)
  run_picking.sh               # sharded ELEP picking per year/region
  run_route_a.sh               # = run_route_a_rerun.sh (OUT/DATA/SUFFIX/HIGHPASS)
  slim_inventory.py  build_station_inventory.py  RUNBOOK.md
paper/                         # Quarto source, bib, Seismica kit, paper/figures (outputs)
data/                          # small tracked inputs + fetch scripts; big products come from Zenodo
notebooks/diagnostics/
```

Renumbering the pipeline directories breaks the relative paths in the scripts (`../../data`).
Do it in one commit, together with a `utils/paths.py` that resolves `REPO/data` from any
script. Then run `make figures && make paper` and check that every figure is byte-identical or
visually unchanged.

## 3. Figure names

Name each file by the number it prints with. The current files don't match: `fig3.png` is
Figure 4 and `fig6.png` is Figure 8. The entry point and the output share the name.

| Printed | Label | Current file | New name | Source |
|---|---|---|---|---|
| Fig. 1 | `fig:fig1` | `fig1.png` | `fig01_stations.png` | `figures/fig1_map_stas.ipynb` |
| Fig. 2 | `fig:fig2` | `fig2.png` (hand) | `fig02_catalog_tables.png` | `figures/fig2.ipynb` |
| Fig. 3 | `fig:qc` | `fig5_histograms/*` (4 files) | `fig03_qc_histograms.png` | `figures/fig5_qc_histograms.py` (already a 2×2 version) |
| Fig. 4 | `fig:fig3` | `fig3.png` (hand) | `fig04_picks.png` | `figures/fig3_plot_picks.ipynb` |
| Fig. 5a/b | `fig:fig4` | `fig4_catalog_map.png`, `fig4_context_map.png` | `fig05a_catalog_map.png`, `fig05b_context_map.png` | `phase5 --region full [--layer context]` |
| Fig. 6a–f | `fig:regions` | `fig_regions/*.png` | `fig06a_endeavour.png` … `fig06f_mendocino.png` | `phase5 --region …` |
| Fig. 7 | `fig:puget` | `fig_slab_puget.png` | `fig07_puget_slab.png` | `phase5 --region puget` |
| Fig. 8 | `fig:fig6` | `fig6.png` | `fig08_subregions.png` | `figures/fig6_subregions_cc.ipynb` |
| Fig. S1–S9 | `fig:supp_*` | `supp_*.png` | `figS01_depth.png` … `figS09_completeness.png` | phase7, 9, 8, 10, 11, 12, 13, 17, 14 |

Also rename the LaTeX labels to match (`fig:fig3` → `fig:picks`, and so on). Labels that
disagree with the printed numbers are a trap during revision.

Figs 2 and 4 are assembled by hand today. Either script the assembly, or keep them as
committed source images with a note in the README saying so.

## 4. Scripts: figures vs at scale

- **Figures** (`figures/`, `make figures`): need only the data products and the small inputs.
  Each takes the Zenodo data paths through `utils/paths.py`. None fetches waveforms. This
  path lets a reader regenerate the paper from the release.
- **At scale** (`scale/`, run on the lab server): picking (days), Route A amplitudes
  (≈40 h on 20 shards), inventory building. Each is a detached, resumable driver with a
  STATUS file, as `run_route_a_rerun.sh` is now. It is documented in `scale/RUNBOOK.md` with
  host, env, runtime and memory.
- **Stages between them** (`workflow/05_magnitude`, `06_analysis`): minutes each, taking
  products to products. `make catalog` runs phase18 → phase19 → phase10 in order.
- Replace the 25 `parallel_pick_{year}[_region]` scripts and 6 `picking_utils*` variants with
  one script and a config table (year, region, channels, priority). Check that it reproduces a
  day of 2011 picks exactly before deleting the copies.

## 5. README (rewrite)

Sections:
1. What this is, with the paper citation.
2. The pipeline in one diagram: ELEP → GENIE (external) → GraphDD + CC (external) → QC →
   Route A magnitudes → analysis. Show product counts: 39.6 M picks, then 116,591 events,
   then 63,887 relocated, then 31,020 QC; 55,707 events have a magnitude.
3. Install (`pixi install`; the four envs and which step needs which).
4. Reproduce the paper figures from Zenodo: `download` → `make catalog` → `make figures` →
   `make paper`.
5. Rerun at scale (pointer to `scale/RUNBOOK.md`).
6. Data products table (file, rows, columns → data dictionary, DOI).
7. Figure map (the table in §3).
8. External codes (GENIE, GraphDD, ELEP, PhaseNet/SeisBench): links, versions, citations.
9. How to cite, license, authors.

Drop the out-of-date tree (`2_association/`, "phase1..6") and the Google Drive link once
Zenodo holds the raw picks.

## 6. Zenodo recommendations

**Two records**, linked to each other and to the paper.

1. **Software** (MIT; the license is already in the repo). Turn on the GitHub–Zenodo
   integration and publish a GitHub release `v1.0.0` when the paper is submitted. Zenodo
   archives the tree at that tag and mints a concept DOI (every version) and a version DOI.
   Cite the version DOI in the paper. Update `CITATION.cff` (version 1.0.0, all authors with
   ORCIDs; Zenodo reads it). Future tags (revision, v1.1.0) become new versions under the
   same concept DOI.
2. **Data: "Cascadia OBS ensemble earthquake catalog 2010–2015, v3"** (CC-BY-4.0). Upload it
   by hand or with `utils/assemble_zenodo.py --apply` plus the Zenodo API. **Reserve its DOI
   now** (Zenodo "reserve DOI" on the draft) so the manuscript can cite it before
   publication; publish it when the paper is accepted, or at submission if the journal
   requires access for review.

Proposed data record contents. `assemble_zenodo.py` must be extended: today it lacks every
magnitude product, and `data/ZENODO.md` still names the superseded `_kpos` magnitudes.

| Folder | File | Notes |
|---|---|---|
| `01_raw_picks` | ELEP ensemble picks (5.5 GB) | or a separate record if the size slows review |
| `02_associated` | GENIE events + pick assignments | label as GENIE output; cite GENIE |
| `03_relocated` | GraphDD + CC origins, arrivals, assoc | cite GraphDD |
| `04_catalog` | **final catalog**: all 63,887 relocated events with QC flag, pick counts, RMS, gap, h_unc, **M, M_type, M_unc, ML, ML_unc, ML_method**, Mw_mt, comcat_id, event_class | the main product; CSV plus QuakeML (ObsPy) |
| `05_magnitude` | Route A amplitudes per pick (`raw_wa_amplitudes_v2`: WA mm, 0.05–2 Hz displacement, SNR, sensor, epoch), station–phase terms, station epochs, `comcat_mt_matched.csv` | lets anyone recompute the magnitudes |
| `06_comparison` | Morton/ANSS matches used in the paper | |
| root | `README.md`, `DATA_DICTIONARY.md` (every column: unit, definition, script that wrote it), `MD5SUMS`, `LINEAGE.md` | |

Metadata:
- Creators with ORCIDs.
- `related_identifiers`: *isSupplementTo* the paper DOI, *isDerivedFrom* the FDSN network DOIs
  already cited (7D, CN, UW, …), *isCompiledBy* the software DOI.
- Keywords, the Cascadia bounding box, and the time span.
- A version string matching the file names (`v3`).

Paper text to fix (Data and Code Availability, `main.qmd` §Data): replace "zenodo (REF)"
with the two DOIs, list the data products, cite GENIE, GraphDD and ELEP, give PB its FDSN DOI,
and remove the double comma.

**Superseded (2026-09-30):** the data record now holds three ANSS-style tables
(origins, arrivals, picks) instead of the folders above; see `data/ZENODO.md` and
`utils/assemble_zenodo.py`.

**Not in v1:** the 0.5 Hz high-pass test (`data/magnitude_hp05/`). If it changes the
magnitudes, release it as data v3.1 during revision.

## 7. Git history (decided: keep it, no rewrite)

`.git` is 1.4 GB: 1.24 GB of packs, mostly 35–83 MB CSVs and notebooks deleted long ago.
Zenodo archives only the tree, so the release doesn't need a rewrite. Clones stay slow,
though. A `git filter-repo --strip-blobs-bigger-than 10M` after the cleanup would shrink it to
roughly 100–200 MB. It rewrites every commit hash, so do it only if all coauthors agree to
re-clone, and after merging or closing the open branches (`feat/*`, `marine-*`).

## 8. Order of work

1. **Done 2026-09-30:** tagged `pre-cleanup-2026-09` and stage tags; merged PR #22; deleted
   `old/`, `3_post_processing/`, `0_data_availability/` and the `_old` CSVs; stripped the two
   >1 MB diagnostic notebooks (`figures/fig2.ipynb` keeps its outputs, since Fig. 2 is
   assembled by hand); README and PROVENANCE. **Still to do by hand:** reserve the data DOI on
   Zenodo.
   **Found while writing PROVENANCE:** the 31,020-event QC file applies > 4 P and > 4 S picks,
   not the ≥ 4 stated in the paper (≥ 4 gives 40,065). Decided: keep the file; the paper
   now says at least five.
2. After `rerun_hp05/DONE`: move to the `workflow/`, `figures/` and `scale/` layout, rename the
   figures and labels, add `utils/paths.py`, `make catalog` and `make figures`, and verify that
   the figures and the PDF are unchanged.
3. Done (2026-09-30): `build_anss_tables.py`, data dictionary, `assemble_zenodo.py`. Package assembled
   in `data/zenodo/` (1.13 GB, 12 files, `md5sum -c` passes). To do: the TO CONFIRM items of
   the dictionary, the paper and software DOIs in `data/zenodo_README.md`, the upload. Rebuild after the 0.5 Hz test if magnitudes change.
4. At submission: GitHub release v1.0.0 (software DOI), fill in the DOIs in the paper, and
   publish or keep in review the data record.
5. No history rewrite (§7).

# Zenodo data package (proposed)

The picks and catalogs that back this study live in three places today — raw ELEP
picks on Google Drive, the relocated catalog + picks in this repo (`data/split_files/`,
chunked), and the final cross-correlated + QC-filtered catalog on the lab server
(`/wd1/hbito_data/data/...`). This note proposes a single, citable Zenodo record that
consolidates the *published* products in analysis-ready form.

## What to archive

Archive the **reconstructed monolithic** CSVs (not the 50 MB GitHub chunks), organized as:

```
cascadia-obs-ensemble-catalog-v3/
  README.md                        # provenance, columns, this recipe, license (CC-BY-4.0)
  catalog/
    events_reloc_cc_qc.csv         # <- origin_2010_2015_reloc_cog_ver3_cc_p_4_s_4_rms_2_5.csv
                                    #    (FINAL catalog shown in the paper figures)
    events_reloc_cc_all.csv        # <- origin_2010_2015_reloc_cog_ver3_cc.csv (pre-QC)
    arrivals.csv                   # <- arrival_2010_2015_reloc_cog_ver3.csv
    associations.csv               # <- assoc_2010_2015_reloc_cog_ver3.csv
    stations.csv                   # <- all_stations_2010_2015_ver3.csv
  amplitudes/
    picks_with_amplitudes.csv      # <- Cascadia_relocated_catalog_picks_ver_3_with_amplitudes.csv
    magnitudes.csv                 # <- 4_relocation/magnitude output (cascadia_catalog_ML_kpos.csv)
  raw_picks/
    elep_picks_2010.csv ... 2015   # <- the Google Drive "Picks" folder (raw per-station ELEP picks)
  comparison/
    anss_2010-2015.csv             # <- anss_2010-15.csv
    morton2023_reloc.csv           # <- origin_2010_2015_reloc_cog_morton_ver3.csv
```

**Exclude** the superseded `*_old` amplitude files (unreferenced provenance only) and
the duplicated monolith-plus-chunk copies.

### Keep-set vs staging (already applied)

`utils/organize_review_data.py` encodes the **keep-set** (the 13 files above + the ANSS
comparison catalog = the proposed Zenodo package) and, on `--apply`, moves *everything
else* out of `data/datasets_all_regions/` into `data/_review/<category>/` (git-ignored,
reversible), writing `data/_review/MANIFEST.csv` — one row per file with its lineage
`category` (`junk` / `superseded` / `intermediate` / `alternative`), a `junk` flag,
size, creation date, `note`, and original path. As applied here: 13 kept in place
(8.7 GB), 195 staged (junk 10 GB, intermediate 13 GB, superseded 0.7 GB, alternative
2.8 GB). Map overlays (`PB2002_boundaries.dig`, `Cascadia_momenttensors_M5_9.xml`,
`pnsn_tremor.json`) are figure inputs, not data products — they stay but are not
archived. Re-run the download to fetch the `.dig`/`.xml` overlays (the filter now
includes them).

To build the actual Zenodo upload, copy the keep-set (+ reconstructed repo monoliths)
into a `data/zenodo/` tree matching the layout above.

## Provenance / naming

`reloc` = GraphDD-relocated · `cog` = center-of-gravity cluster step · `cc` =
cross-correlation-refined differential times · `p_4_s_4_rms_2_5` = QC filter (≥4 P, ≥4 S
picks, RMS < 2.5 s). The QC threshold is defined in
`4_relocation/quality_control/4_quality_control_reloc_cog_ver3_cc_p_4_s_4_rms_2_5.ipynb`.

**Open item to resolve before publishing:** confirm the lineage relationship between the
repo's `Cascadia_relocated_catalog_ver_3.csv` (a merged all-regions catalog) and the
server's `origin_2010_2015_reloc_cog_ver3*` files that the figure notebooks read — i.e.
whether they are the same events pre/post the QC + cross-correlation step, or divergent
products. The figure notebooks use the server files; the repo ships the merged one.

## Reconstructing the repo chunks (already testable)

The repo tracks the large CSVs as chunks; rebuild the monoliths from a fresh clone:

```sh
pixi run python utils/reconstruct_split_csvs.py --list   # preview
pixi run python utils/reconstruct_split_csvs.py          # -> data/<name>.csv (git-ignored)
```

## Downloading the published record (after upload)

Once the record has a DOI, fetch it with the Zenodo API (no auth for public records):

```sh
DOI_RECORD=RECORD_ID          # e.g. from https://zenodo.org/records/RECORD_ID
mkdir -p data/zenodo && cd data/zenodo
curl -s "https://zenodo.org/api/records/${DOI_RECORD}" \
  | python -c "import sys,json;[print(f['links']['self']) for f in json.load(sys.stdin)['files']]" \
  | xargs -n1 -P4 curl -sO
```

Until then, use `./download_data.sh` (lab server) for the figure/pipeline inputs and
`utils/reconstruct_split_csvs.py` for the repo-shipped catalog + picks.

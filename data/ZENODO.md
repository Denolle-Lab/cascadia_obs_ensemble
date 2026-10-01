# Zenodo data record: how it is built

The data record "Cascadia OBS ensemble earthquake catalog 2010-2015, v3" (CC-BY-4.0)
holds three ANSS-style tables (origins, arrivals, picks) plus the magnitude inputs,
the GENIE output and the comparison catalogs. Its layout and file list are in
`utils/assemble_zenodo.py`; its README is `data/zenodo_README.md`; its columns are in
`data/DATA_DICTIONARY.md`.

```sh
pixi run -e amplitude python utils/build_anss_tables.py      # -> data/catalog_v3/ (~15 min, ~40 GB RAM)
pixi run -e amplitude python utils/assemble_zenodo.py        # dry-run: file list and sizes
pixi run -e amplitude python utils/assemble_zenodo.py --apply --out /path/with/space
```

The build reads the lab-server products (`data/datasets_all_regions/`, symlinks to
`/wd1/hbito_data/data/`), the magnitude outputs in `data/magnitude/` and the Route A
amplitudes in `4_relocation/magnitude/rerun_v2/`.

Before publishing:
- reserve the DOI on the Zenodo draft and cite it in the paper;
- fill in the paper citation and the software DOI in `data/zenodo_README.md`;
- resolve the **TO CONFIRM** items in `DATA_DICTIONARY.md`;
- metadata: creators with ORCIDs; related identifiers *isSupplementTo* (paper),
  *isDerivedFrom* (FDSN network DOIs), *isCompiledBy* (software DOI); keywords,
  bounding box 40-50°N, 122-129°W, time span 2010-2015; version `v3`.

Not in v3: the 0.5 Hz high-pass magnitude test (`data/magnitude_hp05/`). If it changes
the magnitudes, release it as v3.1.

## Downloading the published record

```sh
DOI_RECORD=RECORD_ID          # from https://zenodo.org/records/RECORD_ID
mkdir -p data/zenodo && cd data/zenodo
curl -s "https://zenodo.org/api/records/${DOI_RECORD}" \
  | python -c "import sys,json;[print(f['links']['self']) for f in json.load(sys.stdin)['files']]" \
  | xargs -n1 -P4 curl -sO
```

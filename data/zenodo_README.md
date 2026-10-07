# Cascadia OBS ensemble earthquake catalog 2010-2015, v3

Earthquake catalog of offshore Cascadia from the Cascadia Initiative ocean-bottom
seismometers and the onshore networks, 2010-2015, built with an ensemble machine-learning
picker (ELEP), GENIE association and GraphDD relocation with waveform cross-correlation.
Magnitudes are Hutton-Boore local magnitudes with station terms (Route A), replaced
by moment magnitudes above M 4.5.

Paper: **CITATION TO ADD**. Code: **SOFTWARE DOI TO ADD**
(github.com/Denolle-Lab/cascadia_obs_ensemble, release v1.0.0).

## Pipeline

Each stage keeps a subset of the previous one:

| Stage | Product | Rows |
|---|---|---|
| ELEP picking | picks | 39,597,551 |
| GENIE association | events / assigned picks | 116,591 / 1,086,007 |
| GraphDD + cross-correlation relocation | origins / arrivals | 63,887 / 1,004,335 |
| quality control (`qc_pass`) | origins used in the paper figures | 31,020 |
| Route A magnitudes | origins with a magnitude | 55,707 |

## Contents

| File | Rows | What it is |
|---|---|---|
| `catalog/origins_v3.csv` | 63,887 | relocated events: ANSS ComCat CSV columns, then magnitudes, QC flag, event class |
| `catalog/arrivals_v3.csv` | 1,004,335 | event-pick associations with residuals, distances and the Wood-Anderson amplitudes |
| `catalog/picks_v3.csv.gz` | 39,597,551 | every ELEP pick |
| `magnitude/station_terms_routeA.csv` | 1,014 | station-phase magnitude terms |
| `magnitude/comcat_mt_matched.csv` | 112 | ComCat moment tensors matched to catalog events |
| `intermediate/genie_*.csv` | | GENIE association output, as received |
| `comparison/anss_2010-2015.csv` | | ANSS ComCat events of the region (ComCat CSV format) |
| `comparison/morton_reloc.csv` | 63,887 | our catalog (relocation before the cross-correlation step) matched to Morton et al. (2023), with the match offsets (`dist`, `dt`, `NonDimDist`, `id_Morton`) |

`DATA_DICTIONARY.md` defines every column. `LINEAGE.md` gives the processing chain
from picks to catalog. `CHECKSUMS.md5` holds the MD5 sums.

The paper's figures use the quality-controlled subset: `qc_pass == True` in
`origins_v3.csv` (31,020 events: at least 5 P and 5 S picks, RMS < 2.5 s).

The tables follow the QuakeML split into picks, arrivals and origins:
`arrivals.origin_id` = `origins.id` and `arrivals.pick_id` = `picks.pick_id`.

## Codes used

- ELEP (github.com/congcy/ELEP) with SeisBench EQTransformer models; please cite ELEP
  and SeisBench.
- GENIE association (I. McBrearty); please cite GENIE.
- GraphDD relocation (I. McBrearty, Y. Yu); please cite GraphDD.

## Waveform data

FDSN networks 7D, 7A, C8, CN, NV, UW, UO, NC, BK, TA, OO, PB, X6, Z5 and X9 (network
DOIs in the paper), from the UW pnwstore archive, EarthScope and NCEDC.

## Authors

See the paper's author list and the repository `README.md` for contributions.

## License

CC-BY-4.0. Verify the files with `md5sum -c CHECKSUMS.md5`.

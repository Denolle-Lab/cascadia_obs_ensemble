# Data lineage (by creation time)

Reconstructed from the server files' modification times (preserved by `rsync -a` from
`/wd1/hbito_data/data/datasets_all_regions/`) and git history. The pipeline is a chain of
**decreasing samples** — each stage selects a smaller, higher-quality subset of the one
before. The **server files are the authoritative "final" products** (generated last, used
for the paper figures); the repo-tracked copies are byte-identical where they overlap.

## The three datasets (each a smaller subsample)

| # | Dataset | Final file (server) | Created | Count |
|---|---------|---------------------|---------|-------|
| **a** | **ELEP picks** (raw per-station P/S) | `all_picks_all_regions_2010_2015_ver3.csv` | 2025-03-05 | **39,597,551 picks** |
| **b** | **GENIE associated picks-events** (Ian McBrearty) | `all_events_2010_2015_ver3.csv` (events) + `all_pick_assignments_all_regions_2010_2015_ver3.csv` (picks) | 2025-03-20 / 03-25 | **116,591 events / 1,086,007 picks** |
| **c** | **Relocated origins/events + picks** (GraphDD + cross-correlation) | `Cascadia_relocated_catalog_ver_3.csv` (events) + `Cascadia_relocated_catalog_picks_ver_3.csv` (picks) | 2025-03-30 / 03-28 | **63,887 events / 1,004,335 picks** |

Sample funnel: **39.6 M** raw picks → **1.086 M** associated picks (116,591 events) →
**1.004 M** relocated picks (**63,887** events) → **31,020** events after QC (below).

## Full chain (creation-time ordered)

| File | Created | Rows | Stage |
|------|---------|------|-------|
| `all_picks_all_years_for_picking.csv` | 2024-11-29 | 23,206,169 | initial ELEP picks |
| `all_picks_all_years_for_assoc.csv` | 2024-11-30 | 23,206,169 | reformatted for association |
| `all_picks_all_regions_2010_2015_ver3.csv` | 2025-03-05 | 39,597,551 | **(a) ELEP picks, ver3 (final raw)** |
| `all_events_2010_2015_ver3.csv` | 2025-03-20 | 116,591 | **(b) GENIE events** |
| `all_pick_assignments_all_regions_2010_2015_ver3.csv` | 2025-03-25 | 1,086,007 | **(b) GENIE associated picks** |
| `Cascadia_relocated_catalog_picks_ver_3.csv` | 2025-03-28 | 1,004,335 | **(c) relocated picks** |
| `Cascadia_relocated_catalog_ver_3.csv` | 2025-03-30 | 63,887 | **(c) relocated events** (== repo, byte-identical) |
| `origin_2010_2015_reloc_cog_ver3.csv` | 2025-05-22 | 63,887 | relocated origins (QC-pipeline format) |
| `arrival_2010_2015_reloc_cog_ver3.csv` / `assoc_..._ver3.csv` | 2025-05-22 | 1,004,335 | arrivals / associations |
| `origin_2010_2015_reloc_cog_ver3_cc.csv` | 2025-10-14 | 63,887 | cross-correlation-refined origins |
| `origin_2010_2015_reloc_cog_ver3_cc_p_4_s_4_rms_2_5.csv` | 2025-10-14 | **31,020** | **FINAL QC catalog shown in the paper figures** (≥4 P, ≥4 S, RMS < 2.5 s) |

### Amplitude / magnitude side-run (from the relocated pick table)

A separate, later pass re-collected **amplitudes** for the relocated pick table (joined to
each pick's event/origin location + station coordinates) and computed **magnitudes**:

| File | Created | Rows | Stage |
|------|---------|------|-------|
| `Cascadia_updated_catalog_picks_assignment_ver_3.csv` | 2025-10-14 | 1,004,335 | pick table + event origins + station coords |
| `Cascadia_updated_catalog_picks_assignment_ver_3_w_amp_0616/0622/0626_2026.csv` | 2026-06 | 1,004,335 | amplitude runs (iterations) |
| `Cascadia_updated_catalog_picks_assignment_ver_3_w_amp.csv` | **2026-06-29** | 1,004,335 | **amplitudes used for magnitude** (byte-identical to the copy in `~/Downloads`) → `4_relocation/magnitude/` ML |

## Version history

`ver_1` (2025-01/02: `Cascadia_catalog_ver_1`, 74,662 events; `..._picks_ver_1`, 814,603) →
`ver2` (2025-03-22, 70,280) → **`ver_3`** (2025-03 onward, 63,887 events / 1,004,335 picks;
the current lineage) → `updated_ver_3` (2025-10, amplitude-ready pick table) →
`_w_amp` (2026-06, magnitudes). `ver_1`/`ver_2` exist only on the server, not in the repo.

## Identity checks (md5)

- `data/Cascadia_relocated_catalog_ver_3.csv` (repo) **==** server `.../Cascadia_relocated_catalog_ver_3.csv` (`247081ec…`) — the repo ships the true final relocated catalog.
- `~/Downloads/…_w_amp (1).csv` **==** server `…_w_amp.csv` (`607f4477…`) — the magnitude pipeline used the latest amplitude file.

## What to keep

- **Final (keep as canonical):** the server `origin_…_cc_p_4_s_4_rms_2_5.csv` (31,020 QC events) + `origin_…_cc.csv` (63,887) + `arrival/assoc` + `all_stations` + the `_w_amp` amplitudes.
- **Alternative (keep for provenance):** the repo's `Cascadia_relocated_catalog_ver_3*` lineage (identical to the pre-QC relocated catalog) and the raw ELEP picks (dataset a).
- **Drop from archives:** `*_temp` intermediates, `*_test`/`w_amp_test` files, and the superseded `*_old` amplitudes — provenance only, not published.

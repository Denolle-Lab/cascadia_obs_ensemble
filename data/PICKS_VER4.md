# ELEP pick table ver4

`all_picks_all_regions_2010_2015_ver4.csv.gz` (1.16 GB; 49,541,718 picks) is the ver3
pick table with the per-day picks that the ver3 merge left out. Built by
`utils/build_picks_v4.py` on 2026-09-30; files on the lab server in `data/picks_v4/`
(md5 in `MD5SUMS`).

**Format.** Same columns and conventions as ver3. The first 39,597,551 rows are ver3
byte for byte, so every ver3 `pick_id` is unchanged; new picks continue the ids from
39,597,551. A station-day (network, station, UTC day) already in ver3 was never added
again, so no station-day mixes two picker runs.

**What was added** (one row per station-day in `ver4_added_station_days.csv`, with its
`category`, pick-id range and source file):

| category | station-days | picks | stations | years | recommendation |
|---|---:|---:|---:|---|---|
| `eh_run_incomplete_at_merge` | 37,158 | 5,095,669 | 73 | 2010, 2014, 2015 | add: the EH runs were still writing when ver3 was merged (files dated 2025-03 to 04) |
| `2013_nested_folder` | 14,015 | 1,727,705 | 47 | 2013 | add: the 2013 files of the 122-123°W, 46-50°N run sit in a nested folder and were not read |
| `new_run_2010_40-46N` | 5,692 | 580,676 | 38 | 2010 | add: a run of 2025-09 at 122-123°W, 40-46°N (2010 only) |
| `edge_run_dropped` | 16,272 | 2,275,110 | 28 | 2010-2015 | add (PI decision, 2026-10-05): station-days of the two 46-50°N edge runs written before the merge but left out of ver3, for a reason not recorded |
| `v1_not_in_ver3` | 2,572 | 265,007 | 3 | 2011-2015 | add (PI decision, 2026-10-05): first-run picks of CN.MGB, CN.YOUB, 7D.FN05A, absent from ver3 |
| **total** | **75,709** | **9,944,167** | | | |

To keep only some categories, drop the rows whose `pick_id` falls in the ranges of the
excluded categories in `ver4_added_station_days.csv`.

**For association and relocation.** Rerun GENIE on the whole table (or at least every
day that gained picks), then GraphDD. The station file must include the stations that
are new in ver4. The CC differential times, Route A amplitudes, magnitudes and QC then
follow (see `CLEANUP_PLAN.md` and `PROVENANCE.md`).

**Still not picked** (data in pnwstore, see the station review of 2026-09-30):
the OBS at 127-129°W, 40-46°N (63 of 65 stations have no pick: 7D G lines, X9 BB1xx-3xx,
Z5 BB6xx-8xx), the stations west of 129°W (7D J23-J48, X9, Z5, OO Axial cabled array), and 122-123°W, 40-46°N in 2011-2015. Picking them needs new runs.

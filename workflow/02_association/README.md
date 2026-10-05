# 02 Association (GENIE, external)

Phase association was run outside this repository with GENIE (graph neural network
associator; McBrearty & Beroza), by I. McBrearty. No GENIE code is kept here; this stage
only reads and writes files.

| | v3 (the paper) | v5 rerun |
|---|---|---|
| input picks | `all_picks_all_regions_2010_2015_ver3.csv` (39,597,551) | `all_picks_all_regions_2010_2016_ver5.csv` (60,918,847; rebuild recipe in `ver5_HOW_TO_BUILD.txt`) |
| outputs | `all_events_2010_2015_ver3.csv` (116,591 events), `all_pick_assignments_all_regions_2010_2015_ver3.csv` (1,086,007 picks) | to come |
| run | 2025-03-20 / 03-25 | to come |

GENIE repository, commit, trained model and configuration: **TO CONFIRM** (see
[PROVENANCE.md](../../PROVENANCE.md), stage 2). The v5 table keeps every ver3 and ver4
`pick_id`, so assignments of the old and the new runs can be compared pick by pick; see
[`../01_picking/V5_PLAN.md`](../01_picking/V5_PLAN.md) §11.

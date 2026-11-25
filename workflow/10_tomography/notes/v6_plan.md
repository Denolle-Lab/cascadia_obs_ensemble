# Plan for the catalog v6.0 relocation (D-B079, D-B092)

Draft for the PI, 2026-11-25. Catalog v6.0 is the NonLinLoc relocation of all 172,342 catalog v4.1 events in model
v0.4.1, with the clock-corrected picks and the station-term method the PI chooses from the test in this report. It
runs after the PI's review of the remaining clock-candidate epochs (D-B090). Every output goes to
`/wd1/mdenolle_data/reloc_v6`; the driver is `workflow/10_tomography/b4_v6_relocate.sh` (branch `reloc/v6-prep`).

## Inputs

| input | file | state |
|---|---|---|
| fine grids | `reloc_v6/time_v6_v0.4.1` (1,392 station-phases) | built by the chain (stage grids) |
| wide and deep grids | `reloc_v6/wide_time_v6_v0.4.1` | built by the chain (stage wide) |
| grid identity | `reloc_v6/grids_time_v6_v0.4.1.md5` (md5 of every grid file) | stage manifest |
| picks | `checks/pick_timing_v6/Cascadia_picks_catalog_v4.1tc.csv` (md5 1b21ab1c) | to be replaced if the PI confirms more clock epochs |
| events | catalog v4.1 event list | unchanged |
| floor | `reloc_v5/topo/topo_15s_cons_merged_eft.bin` (D-B056, D-B068) | md5 checked by stage surface |
| review flags | `catalog_v5/v5.1_review/returned/review_*.csv` (ingest_review.py) | filled when the reviews come back |

## Stages and run time

Run time below is for 16 processes on the loaded machine, scaled from the v5.1 run (iter0 and iter1: 39 and 38 min
with 100 processes, about 65 process-hours each) and from the cluster test (NLLoc 1,200 events in 40-57 s with 4
processes; Loc2ssst about 26 ns per grid node and arrival).

| stage | what | time at 16 processes |
|---|---|---|
| surface, grids, wide, manifest | done today (grids 2026-11-25) | (see grid status) |
| obs | a2_make_obs.py from the final pick file into `obs_tiles_v6` | 2 min |
| smoke | three tiles (Cascades, Axial, Blanco), no refusal, no undocumented arrival drop, 0 events above the surface | 20 min |
| iter0 | NLLoc, no terms (LOCGAU 0.2 0.0 without LOCGAU2, D-B069; minimum 5 phases, D-B066; merged floor; fine then wide grids) | 4 h |
| terms | static terms from iter0 (a2_station_terms.py), as v5.1 | 1 min |
| iter1 | NLLoc with the iter0 static terms | 4 h |
| SSST (if chosen) | per tile block: Loc2ssst, then NLLoc with the corrected grids, one pass per CharDist | about 4.5 h per pass |
| deep | events rejected at the bottom face or at z' >= 78 km, z' -5..120 km with the wide grids (D-B067) | 10 min |
| coherence (if chosen) | only events with CC partners (D-B040 families); needs CC for every family first | per family |
| catalog | a2_parse_hyp.py, a2_build_catalog.py, b4_catalog.py, b4_acceptance.py, b4_lit_paired.py | 15 min |
| publish | reports committed, `reloc_v6/b4_v6.DONE` for session -18 | 1 min |

Static terms only: about 9 hours. Each SSST pass adds about 4.5 hours (one NLLoc pass plus Loc2ssst). With the
PI's leave to use 64 processes the NLLoc passes take about 1 hour each.

## Station terms: what the test supports so far

Method test on the v5.1 grids (Axial and JdF complete on 2026-11-25 11:35; Blanco, Mount St Helens and the coherence
arms finish in the background, see `reloc_v6/ssst_test/run_g51_coh.log`):

- Axial (1,200 events, 16 station-phases): the v5.1 static terms give the best CC consistency without coherence
  (pair-demeaned CC residual MAD 12.3 ms; repeater pairs 0.134 km apart). The SSST schedule (9999, 16, 8, 4, 2 km)
  makes it worse step by step: 13.5 ms after the static step, 15.2 ms at the end, repeaters 0.242 km apart,
  floor-limited events 40 -> 111, A-B events within 2 km of the seafloor 91.4 % -> 85.7 %. Adding NLL-coherence on
  top of the final SSST gives 11.2 ms and 0.089 km, the best of all arms.
- JdF (162 events, tier C): SSST frees the events from the floor (floor-limited 153 -> 23; median depth below the
  seafloor 0.02 -> 1.41 km). CC consistency is best at CharDist 16-8 km (35.7 ms against 39.0 ms static) and back
  to 40.9 ms at 2 km. Coherence does not apply (7 of 162 events have a compact pdf).

The options for the PI:

- **static (as v5.1)**: iter0, static terms, iter1.
- **regional SSST**: Loc2ssst with large CharDist only (static step, then 16 and 8 km). Close to static at Axial,
  better at JdF; small CharDist (4, 2 km) is not supported by the test.
- **coherence**: the step that sharpened Axial most, but it needs waveform CC for every family (today only the four
  test sets) and applies only to events with a compact pdf (ellipsoid len3 at most 5 km).

## Native NLL-SSST at catalog scale

Loc2ssst (NonLinLoc 7.1.07) has fixed limits: 100,000 events per run (`MAX_NUM_INPUT_FILES`) and 20,000 arrivals per
station-phase (`MAX_NUM_ARRIVALS_LOC2SSST`, more are dropped with an ERROR), and one output grid geometry
(`LSOUTGRID`) for all station-phases of a run. The whole catalog has 2.5 million arrivals, so SSST runs per region:

- blocks of tiles (25 km), each block's Loc2ssst reading the events of the block plus a halo of 3 x CharDist.
  Per tile the largest station-phase count is 15,334 arrivals (t07_32, Axial), below the limit; 13 station-phases
  exceed 20,000 over the whole catalog (B932.PB, B047.PB, AXEC1.OO, AXEC3.OO ...), so Axial blocks with halos must
  stay small;
- `LSOUTGRID` = the block's NLLoc search box. The tiles hold 97,436 station-phases in total (median 41 per tile,
  max 603). With the production box (tile +- 75 km, 175 x 175 x 86 nodes, 10.5 MB per grid) one iteration writes
  about 1 TB; with tile +- 15 km (55 x 55 x 86, 1.0 MB) about 100 GB. The previous iteration is the next one's
  input, so two iterations are on disk at a time;
- Loc2ssst cost: (LSGRID nodes) x (arrivals of the block) at about 26 ns each (measured: Axial 4.2e9 in 28.5 s and
  Blanco 5.7e9 in 36 s with 4 instances): with tile +- 15 km boxes and halos about 0.4 process-hours per iteration at
  LSGRID 4 km (CharDist 8-16 km), 3 at 2 km, 24 at 1 km;
- station sets per Loc2ssst instance must not contain a station code that is a substring of another set's string
  (`LSSTATIONS` is matched with strstr); b4_v6_ssst.station_sets checks this.

An alternative that needs no new grids: apply each event's source-specific correction (the same Gaussian-weighted
mean of its neighbours' residuals, computed at the event's previous position) to its pick times and rerun NLLoc
with the static grids. It is the classic SSST of Richards-Dinger and Shearer (2000) rather than NonLinLoc's grid
form; the correction does not follow the trial hypocentre inside one NLLoc search. Not built: the test does not
support small CharDist, so it is not needed now.

## Picks and clock corrections

The obs stage pins the pick file by md5 (`PICKS`, `PICKS_MD5` in the chain). When the PI confirms more clock epochs
(D-B090), the pick-timing toolkit writes a new pick file; the chain then reruns from obs (the grids do not depend
on the picks). The pick file the catalog was located with goes into the catalog bundle (D-B080).

## Location-review flags

ingest_review.py writes `review_flag_not_earthquake.csv`, `review_depth_unreliable.csv` and
`review_relocate.csv`. b4_catalog.py joins them by ID when the catalog is built. They are flags only: no hypocentre
is moved or removed (D-B055). A reviewer's corrected depth is kept as information, never written into `depth`.

## Catalog columns to add for session -18

All v5.1 columns stay. New columns:

| column | type | meaning |
|---|---|---|
| model_version | str | velocity model of the travel-time grids (v0.4.1) |
| station_terms | str | `static` (iter0 terms, as v5.1) or `ssst` |
| ssst_char_km | float | final Loc2ssst CharDist of the event's region (empty with static terms) |
| rms_unweighted | float | unweighted rms of the used arrivals, s (NLLoc's RMS is EDT-weighted, D-B069) |
| n_picks_retimed | int | picks of the event re-timed (D-B072/D-B075) |
| n_picks_clock_corrected | int | picks corrected for a station clock drift (D-B090) |
| shift_h_v51_km, shift_z_v51_km | float | change from catalog v5.1 (horizontal, depth; positive = deeper) |
| review_not_earthquake | bool | a reviewer flagged the event as not an earthquake |
| review_depth_unreliable | bool | a reviewer flagged the depth as unreliable or implausible |
| review_relocate | bool | a reviewer asked for a relocation |
| review_depth_km | float | the reviewer's depth, information only (never applied) |
| coherence_relocated | bool | relocated by NLL-coherence (only if that stage runs) |
| coherence_partners | int | partners stacked by NLL-coherence |

The README states the grid set (`grids_time_v6_v0.4.1.md5` md5), the pick file md5 and the floor file.
Magnitudes: the converged phase19 solution (D-B091) recomputed on the v6.0 hypocentres.

## Open before the full run

1. PI: station-term method (static, regional SSST, coherence later).
2. PI: the remaining clock-candidate epochs (D-B090), then the final pick file.
3. The v0.4.1 cluster test (`ssst_test/g041`), which runs automatically after the grids.
4. Process budget for the NLLoc passes (16 processes: about 4 hours per pass).

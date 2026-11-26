# termtest: station-term recipe test for the v6.0 relocation (PI request 2026-11-25 14:29)

Code: branch `reloc/v6-termtest` (worktree /home/mdenolle/wt_v6test, from origin/reloc/v6-prep 2d8cf6ff),
workflow/10_tomography/b4_v6_termtest*.py. Nothing here writes into other sessions' folders (reloc_v6/ssst_test,
tomo_v1/cc_dt, tomo_v1/cc_dt_all are read only).

Recipes: A = static station terms (v5.1 b4_v51_iter0) + NLL-coherence; B = NLL-SSST, CharDist 9999-16-8-4-2 km
(stopped at 2 km) + NLL-coherence. References: static alone, SSST-to-2-km alone. Method of the v6-prep test
(b4_v6_ssst.py), v5.1 grids (g51), picks v4.1tc.

## Decision rule (written 2026-11-25 14:50, before any score of these runs was seen)

- Per cluster and criterion, B - A with a 95 % paired bootstrap interval (2,000 replicates; resampling unit = event
  pair for the CC criteria, event for depths).
- Material difference: the interval excludes 0 AND |B - A| >= 5 % of A (held-out MAD, CC fit) or >= 0.2 km
  (median |depth - reference|).
- Criterion 1 (held-out stations, leak-free) is the primary one. A cluster counts for B (or A) when B (A) is
  materially better on P or S and not materially worse on the other phase.
- Recommend B only if it wins criterion 1 in at least 2 of the 3 clusters and is not materially worse on criterion 1
  in any cluster, and puts no event above the surface. Recommend A on the mirror condition. Otherwise "no material
  difference" -> A (simpler). Criteria 2 and 3 are reported and can only veto (a material loss on both in 2
  clusters).

## Layout

| path | what |
|---|---|
| `sets/` | events_<set>.csv, pairs_<set>.csv, README.md (b4_v6_termtest_sets.py) |
| `cc/` | CC dt of the three sets with the tested D-B078 code (b4_v6_termtest_cc.py): cc_<set>.parquet, dt_*.cc, qc/, snippets/ |
| `inputs/` | picks (v4.1tc) and v4 event rows of the sets |
| `runs/g51/<set>/<run>/` | run = full, fold0-2: obs, static_oct, ssst/it0-5 + g0-4, static_coherence, coherence |
| `runs/g51/<set>/grids_base/` | grid links (fine or wide per station-phase) |
| `control/nproc_<set>` | processes per NLLoc pass of the running chains (read before each pass) |
| `logs/` | cc.log, run_<set>_<run>.log, score.log |
| `score/` | scores and bootstrap tables (b4_v6_termtest_score.py) |
| `report/termtest_report.html` | report |

## Resume

    cd /home/mdenolle/wt_v6test/workflow/10_tomography
    PY="env LD_LIBRARY_PATH=/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/lib nice -n 10 /home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/bin/python"
    $PY b4_v6_termtest_sets.py                 # sets (done)
    $PY b4_v6_termtest_cc.py --stages tasks     # then snippets,measure,write,clock (detached, logs/cc.log)
    $PY b4_v6_termtest.py picks; $PY b4_v6_termtest.py prep
    /wd1/mdenolle_data/reloc_v6/termtest/run_chains.sh      # 12 detached chains; resumable (run.json per pass)
    $PY b4_v6_termtest_score.py                 # scores -> score/
    $PY b4_v6_termtest_report.py                # report/termtest_report.html

## Status log

- 14:40 sets built; CC tasks done; CC snippets + measure + write + clock launched (8 workers, logs/cc.log).
- 14:43 picks, obs of all 12 set-runs, grid links; 12 chains launched (puget 2, mtj 1, blancow 1 processes each).
- 15:02 puget and mtj snippets done (puget 24,294 ok of 24,298 channel reads; mtj 389 events with NC/BK merged from
  the D-B093 NCEDC cache). measure,write,clock for puget,mtj started in a second process
  (logs/cc_measure_puget_mtj.log); the first process (logs/cc.log) is stopped after the blancow snippets and
  blancow alone is measured, so no file is written twice.
- 15:00 control/balance.sh: mtj and blancow chains go to 2 processes once the puget fold chains and the CC reads are done.
- 15:02-15:05 the first CC process segfaulted in its blancow reader workers (libffi/python, dmesg) after it had run
  obspy's reader in the parent for the mtj NC merge (fork). Stopped; the ~80 blancow events it wrote moved to
  cc/snippets_discarded_blancow_segfault_run/; blancow relaunched in a fresh process (logs/cc_blancow.log), no
  crash. b4_v6_termtest_cc.py now uses a spawn pool.
- 15:04 puget and mtj CC done (cc_measure_puget_mtj.log): puget 7 accepted rows straddle the MH07/MG07 epochs.
- 15:36 only the mtj full chain left: control/nproc_mtj 8 for its coherence passes.
- 15:42 all 12 chains done (logs/run_<set>_<run>.log). 15:45 scores (score/, logs/score.log); 15:48 report
  (report/termtest_report.html; summary text workflow/10_tomography/notes/v6_termtest_summary.md).

## Result (2026-11-25 15:48)

Recommendation by the rule above: **B**. Held-out MAD (common rows), A vs B: puget P 85 vs 59, S 141 vs 90 ms;
mtj P 67 vs 56, S 118 vs 88 ms; blancow P 149 vs 107, S 264 vs 164 ms (all B - A intervals exclude 0). CC fit B
better in all three; repeaters mixed (blancow A tighter); depth B closer to Lange at blancow, no material difference
at puget (PNSN) and mtj (NCSN, 39 events). No event above the surface. The coherence stage relocates only Len3 <= 5 km
events (16-32 % puget, 42-56 % mtj, 0.4-2 % blancow in the fold runs), so most of the gap is static vs SSST; on rows
relocated by coherence in both arms B - A is -3.5 to -13 ms. Small copies in the repo: data/tomo_v1/v6_termtest/.
Disk: runs/ 33 GB (oct-trees of static_oct and ssst/it5, SSST grids g0-g4); safe to delete after the PI's review.

# Confirmatory test: SSST stopping distance (D-B095 follow-up, coordinator request 2026-11-25 ~15:50)

## Decision rule (written 2026-11-25 15:55, before any run or score of this test)

- Clusters: two new ones sharing no event with axial, blanco, msh, jdf, puget, mtj, blancow (sets2: one onshore,
  one offshore), chosen as before (core events nearest the median epicentre), CC with the same tested code.
- Arms per cluster and run (full, fold0-2): the one SSST chain 9999 -> 16 -> 8 -> 4 -> 2 km; stop at 16 km = it2,
  8 km = it3, 4 km = it4, 2 km = it5 (oct-trees saved on it2-it5); each stop followed by the coherence stage
  (arms coh16, coh8, coh4, coh2) and without it (ssst16, ssst8, ssst4, ssst2).
- Score per stop d (coherence arms, the recipe adopted in D-B095): S(d) = mean over the 2 clusters of
  [held-out MAD P + held-out MAD S] (ms; -18's fold scorer, 3 folds, common rows of all 8 arms of a cluster).
- Choice: d* = argmin S(d). If the next coarser stop d+ (2 -> 4 -> 8 -> 16 km) is not distinguishable from d*,
  i.e. the 95 % paired-bootstrap interval of S(d+) - S(d*) includes 0 (2,000 replicates; event pairs resampled
  within each cluster, the same resample for every arm and both phases), take d+ (one step only). The plain
  overlap of the two individual 95 % intervals is reported alongside.
- Reported, not part of the choice: the no-coherence arms (does coherence help per cluster?), CC fit (median
  per-pair residual), repeater separation, literature depths (PNSN / NCSN via ANSS ComCat), above-surface count
  (must be 0) and floor-limited count.

## Stopping-distance test: status log

- 15:53 sets2 built (b4_v6_termtest_sets.py --group 2): georgia 700 events (Strait of Georgia, CN/PB/UW/TA,
  ~15 km deep), gorda 800 events (family 1162, 2014 M6.8 Gorda plate, Z5/7D OBS + NC/PB/BK/TA); no shared event
  with the seven earlier sets. CC for both launched (logs/cc_group2.log, 6 workers, spawn pool).
- 15:55 picks, obs (full, fold0-2), grid links; 8 chains (run_chains2.sh: --arms ssst,coh_stops), gorda 3 and
  georgia 1 process per pass (control/balance2.sh raises georgia to 2 when the CC process ends). New: MemAvailable
  guard (no new pass below 35 GB).
- 16:14 group-2 CC done (georgia 83,374 ok rows, gorda 90,541 ok; gorda 693 events with NC/BK from the D-B093 NCEDC cache). georgia chains set to 3 processes (balance2 stopped).
- 16:16 georgia/full stopped at its first Loc2ssst: with 1 process the LSSTATIONS line exceeds NonLinLoc's 1,000-character buffer (b4_v6_ssst assertion); relaunched with 3 processes (resumes after it0).
- 16:20 gorda folds done; georgia chains to 5 processes per pass (4 chains x 5 + gorda full 3 = 23).
- 16:44 all 8 chains done.
- 16:47 scores (score_stop/, logs/score_stop.log): b4_v6_termtest_stop_score.py --nboot 2000; report section added
  (report/termtest_report.html, summary notes/v6_termtest_stop_summary.md).

## Stopping-distance result (2026-11-25 16:47)

Choice by the rule above: **4 km** (schedule 9999 -> 16 -> 8 -> 4 km). S(d) with coherence (mean over georgia,
gorda of held-out MAD P + S, ms): 16 km 81.0 [77.8, 83.7], 8 km 78.4 [75.7, 81.4], 4 km 76.8 [74.5, 79.9],
2 km 80.2 [77.4, 83.3]. 8 - 4 km = +1.6 [0.3, 2.7] (excludes 0, so 4 km is kept; the individual intervals overlap);
2 - 4 km = +3.4 [1.8, 4.6]. Coherence helps at georgia (P -15, S -21 ms at every stop), neutral at gorda (few
compact events). No event above the surface in any arm. Disk: runs/ for georgia + gorda adds the oct-trees of it2-it5.

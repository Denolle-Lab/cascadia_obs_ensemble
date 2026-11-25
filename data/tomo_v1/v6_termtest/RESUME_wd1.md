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

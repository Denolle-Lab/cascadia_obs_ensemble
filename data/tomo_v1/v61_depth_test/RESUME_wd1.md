# reloc_v61/depth_test: station-term recipe for catalog v6.1, scored on relative fit AND absolute depth (D-B100)

Coordinator request 2026-11-26 (~12:50 PST), PI approval of the test 2026-11-26 ~12:40 (D-B100). Branch
`reloc/v61-depth-test` (worktree /home/mdenolle/wt_v61_depth, from origin/main a3c247d1). Read only: reloc_v6/v60,
reloc_v6/heldout, reloc_v6/termtest, the grids, the picks, tomo_v1/cc_dt*, the literature pair tables. Nothing is
written outside /wd1/mdenolle_data/reloc_v61/depth_test and the worktree. Not touched: reloc_v6/control, the v6.0
outputs, other worktrees, the main tree.

## Code (workflow/10_tomography/)

| file | what |
|---|---|
| `b4_v61_grids.py` | NonLinLoc grid reader (trilinear value and gradient), parser of (concatenated) .hyp files |
| `b4_v61_depth_diag.py` | diagnosis on the v6.0 outputs (Axial tile t07_32, it0..it4, g0..g3) -> diag/ |
| `b4_v61_depth_sets.py` | fourth cluster gordad (Gorda Depression, Lange-paired events) -> sets/, inputs/, cc/ |
| `b4_v61_depth_test.py` | the test: runs (full + 3 held-out folds) of every arm, scores, decision -> runs/, score/ |
| `b4_v61_depth_score.py` | scores and the rule (called by `b4_v61_depth_test.py score`); post-hoc arms in their own tables |
| `b4_v61_depth_figs.py` | figures (diagnosis, scores) |
| `b4_v61_depth_report.py` | report_depth_test.html |
| `b4_v6_heldout.py` | (existing) small extension: held_out('full') = no station; coherence_run(name=...) |

## Diagnosis (2026-11-26 13:02, before any test run; details in diag/ and the report)

Reproduces D-B100 on the fixed set of 11,921 tier A-B Axial events of tile t07_32 (v6.0 tiers): within 2 km of the
seafloor it0 82.9 %, it1 76.6, it2 73.1, it3 69.2, it4 66.3 %; median depth below seafloor 1.63 -> 1.85 km. The
floor-limited share is small (1.6 % at it0) and the floor-limited events' residuals are not one-sided, so the floor is
not the cause. A linearised relocation with the cumulative Loc2ssst corrections at the it0 positions reproduces the
shift (median +0.22 km predicted vs +0.18 observed at it4; +0.10 vs +0.07 at it1). Far stations contribute ~0. Split
into parts: the part of the corrections common to all S arrivals of an event would deepen it by +0.28 km (mean), the
common P part by +0.07, the station-to-station pattern by -0.18. The S terms at the caldera stations are negative on
average relative to P (S - P common mode -0.040 s after the static pass, -0.062 s after the 4 km pass): S-P predicted
shorter -> events move away from the seafloor array. This is the joint hypocentre / station-correction null space of
JHD (Pujol 1988): for a compact cluster a common shift of the hypocentres and a matching set of station corrections fit
the arrivals equally well; the terms are computed from the residuals at the current positions, so the S-P misfit the
it0 positions leave (mean caldera S residual -0.03 s) is taken up by the terms, and each pass moves the cluster
further along that direction (S term at AXAS2 -0.20 s after the static pass, -0.28 s after the 4 km pass).

## Pre-registered decision rule (written 2026-11-26 13:09 PST, before any run or score of this test)

Clusters (four, each with a literature reference): **axial** (D-B078 set, 1,200 events; Waldhauser et al. 2020: 85 %
of events within 1.6 km and 90 % within 2.0 km of the seafloor), **msh** (D-B078 set, 1,436 events, 1,258 paired with
Hirao et al. 2025), **blancow** (termtest set, western Blanco 2012, 1,200 events, 891 paired with Lange et al. 2026,
absolute locations), **gordad** (new: the 503 catalog events paired with Lange et al. 2026 in the Gorda Depression
segment, longitude -127.1 to -126.0; 497 of them tier A-C in v6.0, so it is included; it is dropped from the depth
mean only if fewer than 100 paired tier A-C events are located in every arm of its full run).

Protocol: the v6.0 recipe of b4_v6_heldout.py (model v0.4.1 grids, fine 1 km or wide where the fine grid does not hold
the box; box = set's v5.1 epicentres +- 30 km; NLLoc as v6.0; Loc2ssst LSPHSTAT/WeightFloor of b4_v6_ssst; coherence
stage with partner rule R3; the Mendocino skip does not apply to these clusters). Runs: full (every station) and
fold0-2 (the stations dealt into 3 folds with numpy default_rng(78 + set index), index 0 axial, 2 msh, 6 blancow,
9 gordad; each fold reruns everything without the picks and CC rows of its stations).

Arms (10): one SSST chain it0 -> g0 (CharDist 9999) -> it1 -> g1 (16) -> it2 -> g2 (8) -> it3 -> g3 (4) -> it4, plus a
second chain nf (below), each with and without the coherence stage on its pass:
- `it0` no station terms; `it0_coh`
- `static` = it1 (one Loc2ssst static per station-phase, CharDist 9999 km); `static_coh`
- `ssst16` = it2 (SSST stopped at 16 km); `ssst16_coh`
- `ssst4` = it4 (SSST stopped at 4 km; `ssst4_coh` is the v6.0 recipe)
- `nf4` = the same schedule to 4 km, but after every Loc2ssst pass the part of the corrections that a common shift
  of the cluster could absorb is removed: per station-phase i a constant d_i = a_i . beta is subtracted from its
  correction (exact on the corrected time grids), with a_i = [dT/dx, dT/dy, dT/dz, 1] of station-phase i (base grid,
  arrival-weighted mean over the pass's events) and beta (4 numbers) chosen so that sum_e G_e' W_e (c_e - d) = 0
  over the pass's located events e (G_e: their arrivals' partial derivatives at their positions, W_e: NLLoc arrival
  weights, c_e: the Loc2ssst increments at their positions). To first order the corrections then cannot move the
  cluster as a whole; relative positions are free. No event is moved or clamped. `nf4_coh`.

Scores:
1. Relative fit (held-out, leak-free): pooled over the 3 folds, per cluster and phase (P, S), MAD of the held-out CC
   differential-time residuals after removing each pair's median training residual (session -18's
   b5_cmp_common.fold_score, as D-B095/D-B096), on the rows located in every arm.
2. Absolute depth (full run): event-paired depth offset ours - published (km, positive = ours deeper), median and
   p10/p90, msh vs Hirao et al. 2025, blancow and gordad vs Lange et al. 2026 (absolute), on the paired events with
   v6.0 tier A-C located in every arm of the full run. Axial: share of the set's events with v6.0 tier A-B within 1.6
   and 2.0 km below the seafloor (depth + merged surface), and as an offset in km: mean of [q85 - 1.6, q90 - 2.0],
   where q85, q90 are our 85th and 90th percentiles of depth below the seafloor (the shift that would put our
   distribution on Waldhauser's two quantiles).
3. Events above the solid surface (all runs).

Rule:
- (a) An arm with any event above the solid surface in any run is out.
- (b) Per cluster and phase, best = the arm with the lowest held-out MAD. An arm is out if, in any cluster and phase,
  the lower end of the 95 % paired-bootstrap interval of MAD(arm) - MAD(best) is above 5 % of MAD(best) (2,000
  resamples of event pairs, the same resample for every arm).
- (c) Among the remaining arms, the one with the smallest D = mean over the literature clusters (axial, msh, blancow,
  gordad) of |depth offset| wins.
- (d) Arms with D within 0.05 km of the smallest D tie; a tie goes to the simplest recipe, in this order (simplest
  first): it0, it0_coh, static, static_coh, ssst16, ssst16_coh, ssst4, ssst4_coh, nf4, nf4_coh (fewer Loc2ssst
  passes first; without coherence before with; existing code before new code).
- An arm whose runs do not all finish is reported as incomplete and is not eligible.
- Reported, not part of the rule: p10/p90 of the offsets, the Axial shares, per-cluster offsets, Lange relative
  (double-difference) depths, floor-limited shares, held-out rms, relocated counts, the nf beta and d_i.

Caveat: the published depths have their own velocity models, station sets and datums (Lange's datum is not stated and
is read as km below sea level; Hirao: km below sea level; Waldhauser: below the seafloor), so an offset of zero is not
proof of a correct depth. The rule uses them as the reference anyway (PI, D-B100); the decision is about which recipe
departs least from them, and the rule cannot see biases common to every arm (the model, v0.4.1).

## Status log

- 13:02 diagnosis run (diag/t07_32/: by_pass.csv, station_phases.csv, linearised.csv, summary.json).
- 13:05 gordad set built (503 events, 8,877 picks, 168 stations, 5,547 accepted CC rows, 2,469 pairs).
- 13:09 rule written (above).
- 13:11 16 chains launched (run_chains.sh; logs/run_<set>_<run>.log). 13:12 gordad/fold0 crashed on a race in the shared
  grids_base links (concurrent first calls); links verified complete for all four sets, chain relaunched.
- 13:14 NLLoc with full base grids costs ~1.5 GB per process (MemAvailable fell to 31 GB): control/max_procs lowered to
  14, then control/governor.sh adjusts it (MemAvailable < 42 GB: -4; > 55 GB: +2, max 24).
- 13:18 slot scheduler rewritten (msh chains had started it0 with 1 process each): every attempt locks all free slots
  and keeps what it may; msh has priority (others leave 6 slots free while an msh chain waits). All chains stopped,
  incomplete pass folders (no run.json / loc2ssst.json / nf.json) removed, chains relaunched 13:19:35. Code committed e853f9dd at 13:20.
- 13:16-13:18 diagnosis also on Puget t29_40, Gorda t22_09, Blanco t11_23 (diag/<tile>/); explanation
  diag_explanation.md.

- 13:22 first nf projection (axial/fold1/nf/g0): beta = common shift (dx -0.14, dy -0.02, dz -0.25 km, dt0 +14 ms) removed;
  summed right-hand side 0 after; corrected grids = raw - d_i to 3e-7 s, headers identical, uncorrected station-phases
  stay links. (nf.json 'beta_names' of these runs says s/km; the units are km, km, km, s; fixed in the code for reruns.)
- 13:21 regional diagnosis re-read with tier A-C medians (A-B has only 119 events at Blanco t11_23): Puget +0.50 km
  (linearised +0.52, common S +0.52), Blanco +0.44 (+0.35, +0.35), Gorda unstable (68 % of events move > 1 km).
- 13:26 passes held all their slots until the slowest NLLoc job ended (about half of the held slots idle). nll()
  now deals the events into the jobs by arrival count (snake order) and a chain
  that waited 3 min ignores the msh reserve. Chains stopped, incomplete pass folders removed, relaunched 13:26:58.
- 13:33 check: the it0 maximum-likelihood hypocentres equal those of reloc_v6/heldout (other job split) exactly on
  axial/fold0, blancow/fold1, msh/fold0; the pdf statistics (Len3, differences up to 3-4 km) depend on the split through
  NLLoc's random sampling, as in every earlier run (they enter the coherence stage's Len3 <= 5 km filter).
- 14:17 post-hoc arm cm4 (common mode per phase removed from every increment; not part of the registered decision) launched for blancow and gordad (run_extra.sh); registered nf4 note: its Hessian-weighted common-shift removal moved the typical blancow/gordad event (nf.json linearised medians -0.5 to -1 km at the static pass).
- 14:23 axial/full restarted with 8 processes for its last coherence stages (it had 1); cm4 chains paused meanwhile.
- 14:28 all 16 registered chains DONE (axial/full restarted 14:23 with 8 processes for its last coherence stages; post-hoc cm4 chains paused 14:23 to give axial the slots, partial cm folders removed). Scoring started 14:29; cm4 chains relaunched 14:29.
- 14:30 registered scores (score/, nboot 2000; logs/score.log). Decision (score/decision.json): only ssst16_coh passes the
  relative-fit gate in every cluster and phase -> winner ssst16_coh (SSST 9999 -> 16 km + coherence), D = 2.02 km
  (v6.0 recipe ssst4_coh 2.35, out at gordad: P +23 ms [+8, +39], S +49 ms [+17, +79]). Same winner without gordad and
  with the other gate reading (sensitivity, not part of the rule). 0 events above the surface in every arm and run.
- 14:52 all 16 post-hoc cm4 chains DONE; 14:54 final scoring (logs/score_final.log): registered decision identical to
  14:30 (score/decision_1430.json); post-hoc cm4 / cm4_coh D = 2.96 / 3.00 km, out by the gate at blancow and gordad
  (and at axial, msh without coherence); Axial -0.54 km with 21.4 % floor-limited (cm4). 14:55 report built.

## Result (2026-11-26 14:55)

Winner by the pre-registered rule: **ssst16_coh** = NLL-SSST 9999 -> 16 km (two Loc2ssst passes) + coherence (R3).
Only arm passing the relative-fit gate in all 4 clusters x 2 phases; D = 2.02 km (v6.0 recipe ssst4_coh 2.35, out at
gordad). Depth ssst16_coh / ssst4_coh / it0: Axial within 2.0 km 89.1 / 87.7 / 91.2 % (within 1.6: 56.4 / 54.0 /
66.4 %); MSH - Hirao +1.26 / +1.15 / +1.53 km; western Blanco - Lange -0.40 / -0.36 / +0.93; Gorda Depression - Lange
+6.24 / +7.66 / +2.40. nf4 (registered principled arm): best Axial depth (0.09 km), over-corrects offshore (blancow
-2.4 km), out at gordad. 0 events above the surface in every arm and run. Report: report_depth_test.html (text inputs
report_summary_input.txt, report_posthoc_input.txt, diag_explanation.md). Small copies in the repo:
data/tomo_v1/v61_depth_test/.

## Layout

| path | what |
|---|---|
| `diag/<tile>/` | diagnosis on v6.0 (by_pass.csv, station_phases.csv, linearised.csv, summary.json) |
| `sets/`, `inputs/`, `cc/` | gordad set (b4_v61_depth_sets.py); sets_summary.csv of the four clusters |
| `runs/<set>/<run>/` | ssst/it0-4, g0-3; nf/it1-4, g<k>_raw, g<k> (nf.json); cm/ (post-hoc); coh_it0, coh_it1, coh_it2, coh_it4, coh_nf4, coh_cm4 |
| `score/` | heldout.csv, depth.csv, surface.csv, decision.json (+ decision_1430.json), extra_*.csv/json, heldout_rows_<set>.parquet |
| `figs/` | PNG copies of the report figures |
| `control/` | max_procs, slots/ (flock), governor.sh/log |
| `logs/` | run_<set>_<run>.log, extra_<set>_<run>.log, diag_*.log, score*.log |

Disk: runs/ ~120 GB (oct-trees and corrected grids); safe to delete after the PI's review (scores and decision are in
score/).

## Resume

    cd /wd1/mdenolle_data/reloc_v61/depth_test && ./run_chains.sh          # resumable; SETS=... RUNS=... for a subset
    setsid nohup control/governor.sh > /dev/null 2>&1 < /dev/null &
    # stop: pkill -f '[b]4_v61_depth_test.py run'; then kill NLLoc/Loc2ssst whose cwd is under runs/, and remove pass
    # folders without run.json (NLLoc, coherence), loc2ssst.json (g*), nf.json (nf/g?) before relaunching
    cd /home/mdenolle/wt_v61_depth/workflow/10_tomography
    PY="env LD_LIBRARY_PATH=/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/lib nice -n 10 /home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/bin/python"
    $PY b4_v61_depth_test.py score --nboot 2000      # -> score/ (decision.json)
    $PY b4_v61_depth_report.py --implications /wd1/mdenolle_data/reloc_v61/depth_test/summary.md
    ./run_extra.sh                                                 # post-hoc cm4 (waits for each run's DONE)

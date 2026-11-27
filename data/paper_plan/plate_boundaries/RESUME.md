# Plate-boundary paper plan (D-B107, D-B109, D-B111): resume notes

Worktree /home/mdenolle/wt_pbplan, branch analysis/plate-boundaries. Code workflow/11_paper_plan/, small tables here,
page and large files /wd1/mdenolle_data/paper_plan/. Python: LD_LIBRARY_PATH=/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/lib
/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/bin/python, nice 10, at most 6 processes.

## Status log
- 2026-11-26 16:32 PST: first build on catalog v6.0 (cecab0a5); page plate_boundaries_plan.html.
- Fri Nov 27 09:38:33 PST 2026: coordinator (D-B111): Vp/Vs spread raw (factor 1.0, not 1.5); circularity control on the
  v0.4.2 synthetic ensemble required before any contrast is interpreted. Rule below written and committed before any
  synthetic contrast is computed.

- Fri Nov 27 09:42:27 PST 2026: scored (pb_circularity.py, rule unchanged): 105 contrasts testable; 39 pass 2 sigma_ens
  (Vp/Vs raw); 36 survive the rule; dropped by S95: GS Vp/Vs near-far, MSH Vp along, MSH Vp/Vs across. Spurious sd ~
  raw member sigma (median ratio 0.97); S95 median 0.90 of 2 sigma_ens; noise-floor clone sd <= 0.0024. Null truth not
  possible with the existing synthetic set (one truth carrying the v0.4.2 contrasts); cost estimate ~2.1
  process-hours per member. Table circularity_contrasts.csv; page section 3.
- Next: rerun everything on catalog v6.1 when the coordinator sends the paths of its clusters, Mc maps and fault
  surfaces (D-B111); pb_common.CAT, CLU, MCMAP, FLT/FLT_BIG are the places to change.

## Pre-registered rule: circularity control of the structure contrasts (written Fri Nov 27 09:38:33 PST 2026, before scoring)

Contrasts: every contrast of pb_structure.py, i.e. segment x variable (Vp, Vs, Vp/Vs) x type (near-far, across,
along), on the same node sets (v0.4.2 masks: rock and (checkerboard or DWS >= 100), P / S without Vs-reverted nodes /
both; distances to the segment's fault surfaces and v6.0 seismicity), depth-matched as in pb_structure.depth_matched.

For each contrast k:
1. D_real = the contrast in the deterministic model v0.4.2 (product values, as in structure_contrasts.csv).
2. sigma_ens = std of the contrast over the 100 real ensemble members (v0.4.2+ens.100) x factor: Vp 1.50, Vs 1.35,
   Vp/Vs 1.00 (D-B099, D-B111).
3. Synthetic (ensemble_v0.4.2/synth): D_truth = the contrast in the synthetic truth (synth/truth/truth_model.npz,
   v0.4.2 tomography x (1 + f)); D_rec,i = the contrast in synthetic member i (s000-s099, merged.npz vp and vs before
   the D-B073 node rule, like the truth); spurious s_i = D_rec,i - D_truth; S95 = 95th percentile of |s_i| over the
   members.
4. **Rule: contrast k counts (survives) only if |D_real| >= 2 sigma_ens AND |D_real| >= S95.** If fewer than 50
   synthetic members or the truth give no value on those nodes, k is "not testable" and does not survive.
5. Reported, not part of the rule: mean spurious contrast (bias) and whether it has the sign of D_real; recovery ratio
   mean(D_rec) / D_truth; spread of the noise-floor clones nf0-nf4 with s000; the rule with sigma_ens alone (old) for
   comparison.
6. Null truth: the synthetic set has one truth that carries the v0.4.2 contrasts. A truth without across-fault or
   along-strike contrast needs new synthetic inversions; if the existing members do not allow it, this is stated, not
   approximated.
7. The rule is not changed after the synthetic results are seen; any other threshold is reported as a variant only.

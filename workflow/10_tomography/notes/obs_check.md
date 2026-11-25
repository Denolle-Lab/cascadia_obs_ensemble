# OBS S-pick waveform check (follow-up to D-B010)

Date: 2026-11-22. Script: `../b1_obs_waveform_check.py` (reads params.yaml, writes a run manifest).
Outputs: `data/tomo_v1/obs_check/` (figures `wf_<group>_<station>.png`, `summary_*.png`,
`obs_check_station_stats.csv`, `obs_check_events.csv`, `obs_near_station_S_terms.csv`, manifest).
Raw windows (699 miniSEED, 20 MB): `/wd1/mdenolle_data/tomo_v1/obs_check/raw/` (not in git).

## Question

At deep-water OBS with strongly negative v4-NLL3D S terms (S observed 0.8-2.4 s before the 3D-model time),
is the deep-learning S pick the S wave, or a P-to-S conversion at the sediment-basement contact (Ps,
0.3-2.5 s after P) or another phase? At shelf OBS with large positive S terms, is the delay the size
expected from slow sediments?

## Data and method

* Picks: all weighted (EDT w > 0) iter1 picks of the v4-NLL3D run `iter1_vpvs176`, from the B1 phase cache
  (`b1_dataset/cache/phases_iter1_vpvs176.parquet`), not only the B1 subset, because the stations with the
  most negative terms have 5-50 S picks. `res0` = NLLoc residual + applied station delay = residual against
  the 3D model without station term. Predicted P = predicted S / 1.76 (the run's constant Vp/Vs).
* Stations: 16 deep-water OBS with S term < -0.8 s (W01, W02, W03 of 7A; J35A, J35C, M03C, FN13C, J58C, J67C,
  J34A, J11B, G20D, J20D, J28D of 7D; GB220, GB281 of Z5), 8 shelf OBS with S term > +0.5 s, 3 control OBS
  (BB300, BB320 of X9, J06B of 7D, |S term| < 0.15 s) and 3 control land stations (B045, B047, B049 of PB,
  coastal, |P|, |S term| < 0.1 s).
* Waveforms: up to 25 events per station (events with a P pick at the station first, then nearest), read
  from the local pnwstore archive through its sqlite index (EarthScope FDSN as fallback), 2-12 Hz.
  Horizontal orientation of the OBS is not known, so H1/H2 are plotted unrotated and the horizontal
  modulus is used. Eight events per station are plotted with the P pick, S pick, model P and S, and model
  S + v4-NLL3D term.
* Tests:
  1. Moveout. Picked S minus predicted P against predicted S-P. A conversion picked as S sits at a constant
     0.3-2.5 s with slope 0; a true S follows slope ~1 with intercept ~ the S term.
  2. Onset on each component. AIC onset on Z and on the horizontal modulus in pick -2.5/+3 s, and the H/Z
     RMS ratio in the first second after the pick. An S-to-P conversion (Sp) picked as S would show a Z
     onset at the pick, the horizontal onset later, and H/Z < 1.
  3. Residual against distance, and a station static from near events only (epicentral distance <= 60 km,
     n >= 10), compared with the v4-NLL3D term (all distances).

## Results

### Moveout and onset (from `obs_check_station_stats.csv`)

| station | group | water km | S term | P term | n S | median dist km | slope S-Ppred | median S-Ppred s | tH-pick s | H/Z | near-60 km static (n) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W01.7A | neg | 1.70 | -2.35 | - | 16 | 166 | 0.96 | 16.4 | -0.10 | 3.0 | none (1) | true S |
| J35A.7D | neg | 2.71 | -2.15 | - | 20 | 189 | 0.83 | 18.5 | -0.19 | 9.6 | none (0) | true S |
| M03C.7D | neg | 1.85 | -2.14 | -1.01 | 48 | 138 | 0.96 | 13.9 | -0.08 | 6.4 | none (0) | true S |
| W02.7A | neg | 2.13 | -1.99 | - | 21 | 128 | 0.89 | 13.5 | -0.12 | 2.5 | none (3) | true S |
| FN13C.7D | neg | 1.78 | -1.97 | - | 16 | 134 | 0.94 | 13.5 | -0.14 | 10.1 | none (2) | true S |
| J35C.7D | neg | 2.71 | -1.95 | - | 15 | 151 | 0.83 | 15.6 | -0.18 | 9.7 | none (5) | true S |
| J58C.7D | neg | 1.53 | -1.75 | -0.54 | 28 | 131 | 0.94 | 13.6 | -0.11 | 8.1 | none (3) | true S |
| J67C.7D | neg | 2.65 | -1.65 | -0.78 | 189 | 154 | 0.88 | 14.8 | -0.16 | 3.2 | none (0) | true S |
| J34A.7D | neg | 2.57 | -1.45 | -0.19 | 26 | 128 | 0.96 | 13.4 | (dead H) | - | none (1) | true S, picked on Z (horizontals dead after the first events) |
| W03.7A | neg | 2.52 | -1.40 | -0.94 | 44 | 126 | 0.94 | 12.8 | -0.10 | 4.1 | none (2) | true S |
| J11B.7D | neg | 3.06 | -1.31 | -0.30 | 484 | 124 | 0.88 | 12.0 | -0.02 | 13.2 | none (3); near-100: -0.90 | true S |
| GB220.Z5 | neg | 3.03 | -1.25 | -0.36 | 1638 | 137 | 0.88 | 13.1 | -0.08 | 1.7 | -0.78 (97) | true S |
| G20D.7D | neg | 2.98 | -1.00 | -0.01 | 3008 | 105 | 0.91 | 10.6 | -0.14 | 8.3 | -0.65 (537) | true S |
| GB281.Z5 | neg | 3.16 | -0.85 | -0.11 | 4524 | 113 | 0.84 | 11.4 | -0.16 | 11.1 | -0.49 (486) | true S |
| J20D.7D | neg | 2.94 | -0.92 | -0.28 | 705 | 123 | 0.91 | 12.1 | -0.06 | 15.4 | none (0); near-100: -0.65 | true S |
| J28D.7D | neg | 2.91 | -0.97 | -0.20 | 377 | 185 | 0.95 | 18.1 | +0.02 | 6.1 | none (2) | true S |
| FC03D.7D | shelf | 0.47 | +1.83 | - | 34 | 55 | 0.98 | 10.1 | -0.15 | 10.2 | +1.82 (21) | true S, sediment delay |
| FS15D.7D | shelf | 0.11 | +1.61 | +0.86 | 11 | 14 | - | 6.2 | -0.19 | 1.3 | +1.61 (11) | true S, sediment delay |
| J33A.7D | shelf | 0.40 | +1.13 | +0.76 | 31 | 67 | 0.99 | 10.0 | +0.16 | 7.0 | +1.07 (12) | true S, sediment delay |
| FS18B.7D | shelf | 0.21 | +1.12 | +0.17 | 180 | 30 | 0.98 | 6.7 | -0.24 | 13.4 | +1.14 (157) | true S, sediment delay |
| FS14D.7D | shelf | 0.21 | +1.03 | +0.30 | 401 | 21 | 1.02 | 5.7 | -0.20 | 2.8 | +1.02 (384) | true S, sediment delay |
| FS42D.7D | shelf | 0.14 | +0.80 | +0.07 | 964 | 36 | 1.00 | 7.2 | -0.15 | 7.7 | +0.79 (852) | true S, sediment delay |
| FS08D.7D | shelf | 0.20 | +0.69 | +0.13 | 2702 | 22 | 1.04 | 5.4 | -0.20 | 8.7 | +0.67 (2332) | true S, sediment delay |
| M01C.7D | shelf | 0.19 | +0.88 | +0.26 | 57 | 91 | 1.05 | 12.1 | -0.18 | 11.4 | +0.69 (12) | true S, sediment delay |
| BB320.X9 | ctrl OBS | 2.80 | -0.10 | -0.08 | 7544 | 46 | 0.94 | 5.5 | -0.11 | 12.5 | | true S |
| J06B.7D | ctrl OBS | 3.28 | -0.11 | -0.13 | 3844 | 87 | 0.93 | 9.0 | -0.20 | 6.3 | | true S |
| BB300.X9 | ctrl OBS | 2.95 | -0.13 | +0.01 | 6881 | 34 | 0.89 | 4.2 | -0.11 | 3.6 | | true S |
| B047.PB | ctrl land | | -0.09 | +0.06 | 24914 | 37 | 0.94 | 6.1 | -0.05 | 1.6 | | true S |
| B045.PB | ctrl land | | -0.08 | +0.06 | 14710 | 35 | 0.96 | 6.0 | -0.09 | 1.7 | | true S |
| B049.PB | ctrl land | | -0.09 | +0.06 | 12899 | 47 | 0.99 | 7.5 | -0.08 | 2.9 | | true S |

slope S-Ppred: slope of (picked S - predicted P) on predicted S-P. tH-pick: median AIC onset of the horizontal
modulus relative to the S pick (the AIC itself sits ~0.1-0.2 s before the onset on every group, controls
included). H/Z: horizontal/vertical RMS ratio in the first second after the pick (median).

### What the numbers show

* No S pick at the negative-term stations is a Ps conversion. The picks are 10-18 s (median) after the
  predicted P, never in the 0.3-2.5 s window, and their moveout slope on predicted S-P (0.83-0.96) is the
  same as at the control OBS (0.89-0.94) and land stations (0.94-0.99). Figure `summary_S_minus_Ppred.png`.
* The picks are not an Sp precursor on Z either. The horizontal S onset coincides with the pick
  (tH - pick -0.02 to -0.19 s, same as the controls), and the first second after the pick is dominated by
  the horizontals (H/Z 1.7-15). In the record sections the pick sits on the first large horizontal arrival;
  the model S time (no term) falls 1-2.5 s later, usually inside the S coda. J34A is the exception in kind
  only: its horizontals are dead for most of the deployment (long-period glitches only), and the picks are on
  the S onset on Z, still 13 s after P.
* The negative terms are mostly a path effect that grows with distance, not a station constant.
  The most negative stations (W01-W03, J35A/C, M03C, FN13C, J58C, J67C, J34A, J28D) have no or almost no picks
  within 60 km; their median pick distance is 125-190 km. Over all 255 OBS with a term, the S term correlates
  with the median S-pick distance (r = -0.47): median term +0.12 s for stations with median distance < 50 km,
  +0.07 s at 50-100 km, -0.41 s at 100-150 km, -0.50 s beyond 150 km. The control OBS show the same trend
  when their own far picks are binned (BB300: S residual -0.09 s within 40 km, -1.29 s beyond 100 km; J06B:
  +0.02 and -0.40). The residual against predicted S-P (`summary_Sres_vs_dist.png`) falls steadily with
  distance at deep-water OBS. P residuals fall too, but 2-5 times less (beyond 100 km: P -0.2 to -1.2 s,
  S -1.0 to -2.4 s), so the 3D model's Vs (Vp/1.76) is too slow along long paths in the oceanic plate, or
  Vp/Vs there is below 1.76, or the far offshore events are biased. That is structure for B2, not a
  station correction.
* Where near events exist, the deep-water static is much smaller than the v4-NLL3D term: G20D -0.65 (term
  -1.00), GB281 -0.49 (-0.85), GB220 -0.78 (-1.25), J11B -0.90 within 100 km (-1.31), J20D -0.65 within 100 km
  (-0.92). A near-60 km static is below -0.5 s at 6 OBS in > 1.5 km of water (G19B, G20D, G35B, J11D,
  GB220 off southern Oregon / Gorda, and NC27.NV), against 41 with the v4-NLL3D term, and no lower than
  -0.78 s. Those -0.5 to -0.8 s statics may be local structure (fast shallow crust) or a location bias of the
  nearby events; they are small enough to keep as statics.
* Shelf OBS: true S, delayed by slow sediment. Picks are on the horizontal S onset, the delay is a station
  constant (near-60 km static equals the v4-NLL3D term within 0.05 s at FS08D, FS14D, FS15D, FS18B, FS42D,
  FC03D; M01C, M08A differ by 0.2 s), P terms are near zero (median +0.06 s), and the S delay is 0.6-1.8 s.
  A delay dt = h (1/Vs_sed - 1/Vs_model) with Vs_sed 0.3-0.6 km/s and Vs_model ~2-3 km/s gives
  h ~ 0.2-0.4 km for 0.7 s and ~0.5-0.9 km for 1.6 s, the sediment thickness range of the northern California
  and Oregon shelf. The matching Ps - P delay for the same layer, h (1/Vs - 1/Vp), is 0.5-1.6 s; at the shelf
  sites the picks are 4-12 s after P, so no shelf S pick is on a Ps phase. The moveout intercepts (+0.4 to
  +2.0 s) agree with the terms.
* Z onset before the horizontal onset by 0.6-1.5 s (median tH - tZ) at J11B, GB281, J33A and BB320 may be an
  Sp precursor on Z; the picker did not take it (pick on the horizontal onset). The AIC on Z is noisy (B047, a
  land control, gives 2 s), so this is not used for any decision.

## Verdicts

* Deep-water OBS with negative terms (16 checked): all true S. None are Ps, none are Sp on Z.
* Shelf OBS with positive terms (8 checked): all true S, delayed by sediment; terms are station constants.
* Unclear: none. J34A is picked on Z because its horizontals are dead; the picks are still S.

## Proposed decisions

1. Keep all OBS S picks. Dropping deep-water OBS S is not supported: the picks are S.
2. Revise D-B010: the station term is right for the shelf but wrong in kind for the deep-water stations
   whose term comes from far events. Use a near-event S static instead of the v4-NLL3D term:
   * static = median no-term iter1 S residual of picks with epicentral distance <= 60 km (n >= 10);
     if fewer, the same within 100 km (n >= 10); if fewer, no static (S pick kept, uncorrected).
   * Counts on the B1 dataset: 95,804 OBS S picks get a near-60 static (172 OBS), 8,336 a near-100 static
     (31 OBS), 2,879 none (70 OBS, 47 of them in > 1.5 km of water, with only far picks). OBS with static
     < -0.5 s: 13 instead of 45 (none below -0.91 s; the -1.3 to -2.4 s terms all drop out). Table: `data/tomo_v1/obs_check/obs_near_station_S_terms.csv`
     (columns near40/60/100, n40/60/100, S_static_proposed, static_source).
   * The 92 OBS S picks without any v4-NLL3D term stay dropped (unchanged).
3. Leave the far-path S misfit (-1 to -2.4 s at > 100 km offshore) to the inversion. It is the main Vs signal
   offshore. Check the B2 starting model: Vp/Vs = 1.76 offshore may be too high in the oceanic plate; a
   forward run with Vp/Vs 1.70-1.73 west of the trench would tell. If the B2 residual QC (D-B012, 4 sigma_MAD on
   forward residuals) rejects many far OBS S picks, that is a model issue, not a pick issue.
4. Shelf statics (water < 0.5 km, median +0.57 s) are unchanged by the rule, because the near and all-distance
   terms agree there.
5. Optional for the station-static option (i-b): start the damped statics from the near-event value, not from
   the v4-NLL3D term.

## Caveats

* The v4-NLL3D locations of the far offshore events depend on the OBS picks; a location bias of those events
  would also project into distance-dependent S residuals. The near-event static avoids that.
* OBS horizontal orientation unknown; no R/T rotation. The onset test uses the horizontal modulus.
* 25 events per station at most; at W01, W02, FN13C, J35C, J35A all S picks are measured (15-21).

# Bayesian uncertainty for the tomoDD models and locations (design, D-B030 proposed)

Coordinator draft, 2026-11-22. Status: proposed; nothing run. Builds on D-B020 (OBS statics), D-B022/023
(balanced data), D-B026 (tomoDD, parallel serial runs), D-B028/031 (forward fixes), D-B029 (depth nodes).

## 1. What we want and why an ensemble

Posterior uncertainty of Vp, Vs, Vp/Vs on the inversion nodes and of every hypocentre, including the parts the
usual resolution tests miss: station effects, pick noise, the starting model, the regularisation, and the
remaining forward error. A Markov-chain sampler around tomoDD is out of reach (one run = 12 nonlinear iterations,
~30 min). We use **randomize-then-optimize (RTO) / randomized maximum likelihood**: draw every uncertain input
from its prior, perturb the data with draws of the noise, run the full inversion, keep the result. For a linear
Gaussian problem the ensemble is an exact posterior sample; here it is an approximation whose calibration we test
on synthetics (section 5) before believing it.

## 2. Uncertain inputs and their priors

Each member m draws all of the following independently.

| # | Input | Prior | Numbers it rests on |
|---|---|---|---|
| 1 | Starting model | mixture: v4-NLL3D model (D-B005) 50%; same with Vp/Vs 1.73 and 1.80 25% each; each plus a smooth random field (Vp and Vs separately, 3% std, Gaussian correlation 30 km laterally, 8 km in depth, tapered to 0 below 60 km) | spread between published models; D-B020 follow-up (Vp/Vs 1.76 possibly high offshore) |
| 2 | Station static, S at OBS | N(D-B031 P-6 static, s_stat), s_stat = bootstrap std of the near-event median (per station) plus 0.05 s | P-6 statics, 101 stations within 60 km |
| 3 | Station static, all other station-phases | N(0, 0.05 s) land P, N(0, 0.08) land S, N(0, 0.08) OBS P | P-4 residual spread after correction (p95 0.07-0.12 s P) |
| 4 | Forward-model error (P-4 residual) | per station-phase offset N(0, s_fwd) with s_fwd = robust std of the v2 end-to-end residual for that station type and distance bin | notes/forward_accuracy.md section 7 |
| 5 | Pick noise | per pick N(0, lambda_c x sigma_pick), sigma_pick from D-B008, lambda_c a class factor (land/OBS x P/S) drawn from its posterior given the residuals of the selected deterministic run (inverse-gamma, empirical Bayes) | residual RMS of the chosen run vs sigma classes |
| 6 | Starting hypocentres | NLL location PDF of v4-NLL3D (Gaussian with the iter1 covariance), depth clipped above the surface | iter1 sig_z, h_max |
| 7 | Regularisation | damping log-uniform on [700, 1500]; smoothing 15-30 (lateral), half that in depth | the scan in mendocino_gorda_v2 (to fill in) |

Items 2-4 enter as constant time shifts per station-phase added to the absolute times (differential times are
unchanged by a static), because tomoDD cannot solve statics. Integrating them out this way is what makes the model
spread include station effects. Item 5 perturbs absolute times pick by pick; the differential times are rebuilt from
the perturbed absolute times, so their noise is consistent.

Amendments proposed 2026-11-23 (notes/merge_design.md P-M5, P-M6): draw items 1, 2-3, 5 and 6 once for the whole
region (random field on one regional lattice, statics by station code, noise by pick ID, start hypocentres by event
ID) so member m of every tile shares its draw and tiles merge member by member; item 7 per tile, log-uniform on
[0.7, 1.5] x that tile's chosen damping.

## 3. One member

1. Draw items 1-7 (seeded RNG, seed = member index; draws saved in member/draws.json).
2. Build inputs with b2_tomodd_inputs.py plus a `--member` option that applies the draws (new MOD, shifted and
   perturbed absolute times, rebuilt dt.ct, perturbed event.dat).
3. Run b2_tomodd_run.py with the drawn damping and smoothing; back-transform (b2_tomodd_backtransform.py).
4. Keep: Vp, Vs, DWS on nodes; relocations; final residual statistics; the draws.

Same binary, flags and host for every member (D-B026). Numerical noise floor: 5 members with identical draws
except a 1e-6 s data perturbation; their spread (~0.2 km expected from the FMA test) is reported as the floor.

## 4. Cost

One full run: 26-31 min and 2.4-2.7 GB alone, ~1.2-1.65x slower with 32-64 at once (D-B026). 100 members at 40
concurrent: 3 waves x ~45 min = ~2.5 h per region, ~110 GB RAM. Disk ~0.4 GB per member (keep outputs, drop
Input_Files copies): 40 GB per region.

## 5. Calibration before real data (STATUS step 5)

A synthetic truth: the v4-NLL3D model plus a known random field (as item 1) plus known station statics (draw of
items 2-3), times from tomoDD-SE_syn on the real geometry, noise per item 5. Run the full 100-member ensemble on
it. Pass if, on resolved nodes (spike/checkerboard recovery >= 0.5, D-B018), the true value falls inside the 68%
and 95% intervals 60-76% and 90-98% of the time, and true hypocentres likewise. If the intervals are too narrow,
widen the priors that the data cannot constrain (statics, starting model) before going to real data.
Also repeat the D-B029 spike test at the chosen damping/smoothing.

## 6. Products

Per node: posterior mean, median, std, 5-95% of Vp, Vs, Vp/Vs, probability that dlnV > 0, resolved mask. Per event:
mean location and covariance from the member relocations (tomoDD, in-run). For B4 (all 172k events with NonLinLoc):
relocate in 20 member models drawn from the ensemble, P-1 grids each (D-B028), to carry velocity-model uncertainty
into every hypocentre; the spread feeds the fault-surface uncertainty and, through hypocentral distance, the
magnitude recomputation (STATUS step 9).

## 7. Proposed decisions (for D-B030)

* RTO ensemble of 100 members per region with the priors of section 2.
* Calibration on synthetics (section 5) is a gate: no real-data ensemble until coverage passes.
* B4 location uncertainty from relocation in 20 ensemble models.

Open for the PI: prior weights of the starting-model mixture (include Delph et al. 2018 where it covers the region?);
the 3% / 30 km random-field prior; 100 vs 200 members.

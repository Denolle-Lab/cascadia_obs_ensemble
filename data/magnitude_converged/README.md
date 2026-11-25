# Converged Route A magnitudes (D-B091), catalogs v4.1 and v5.1

Decision D-B091 (PI, 2026-11-25 10:24): run the phase19 station-term inversion to convergence and
recompute the v4.1 (paper) and v5.1 magnitudes. Branch `analysis/mag-converged` (from
`analysis/v51-mag` cd0334d6). Report: `magnitude_converged_report.html` (self-contained).
Large outputs: `/wd1/mdenolle_data/magnitude_converged/` (see `RESUME.md` there).

An independent verifier re-derived the results (scripts and outputs in
`/wd1/mdenolle_data/magnitude_converged/verify/`). Its verdict: fit to replace the paper's magnitudes.
- The solution is the true minimum.
- It is set by the data, not the damping: damping 1e-4 changes ML by at most 0.001; damping 1e-2 by
  0.026 at the 99th percentile.
- The anchor test is not circular. Fitting the offset to half of the anchors and testing on the other
  half gives an offshore gap of +0.116 (stored) and +0.009 (converged); the converged gap is closer to
  zero in 99.8% of 500 splits.

Its must-fix items are resolved below (gauge).

## What changed
Only the phase19 solver and its gauge. Amplitudes, SNR cut, station epochs, the Hutton-Boore term,
the 150 km pick selection, the 4-MAD rejection, the anchors, the Mw switch at 4.5 and the Mw_cal
rule are unchanged.

- `workflow/05_magnitude/phase19_solver.py` (new). Method `exact` (default) eliminates the event terms
  and solves the Schur complement of the station-phase terms directly (1,337 x 1,337 for v4.1), then
  refines twice on the full sparse system. It raises `NotConverged` unless the last refinement step is
  below 1e-8 and the relative normal-equation residual below 1e-10, and unless the gauge rows equal
  the null directions (`check_gauge()`).
- Gauge (verifier must-fix). Picks not linked to the main network are dropped before each pass
  (`main_component()`): this is the term 7D.J35C@2013-08-09|S, 9 picks on 9 one-pick events, so no ML
  is lost. One gauge row is used: the sum of the P terms = 0.
  - The stored design had two rows (sum P = 0, sum S = 0). The S row was absorbed by that isolated
    term (C = -303 in v4.1 and -295 in v5.1 converged; -255 and -234 in the legacy runs), which left
    the P-S difference free.
  - The earlier statement here, that the two rows fix the P-S difference, was wrong.
  - Without the isolated term, the two rows would have forced mean S = mean P: misfit +2,938, and
    regional ML down by up to 0.22. The convergence test would not have caught that.
  - ML is unchanged by the fix (`gauge_design_test.csv`; full chain, `gauge_ml_unchanged.csv`: v4.1
    max |dML| 0.0015, v5.1 0.0015).
  - The isolated term is no longer in the converged station-term tables. In the stored
    `phase19_station_terms_v51.csv` it is a gauge artefact (C about -234): ignore it. Method `lsqr` is a column-scaled LSQR (atol = btol = 1e-12,
  istop checked) kept as an independent check. Method `legacy3000` reproduces the stored runs.
- `phase19_preferred_magnitude.py`: `--solver`, `--outdir`, `--tag`; writes `<stem>.provenance.json`.
- `v51_magnitude_table.py`: `--pm`, `--out`, `--tag`, `--solver`, `--v4-mags`.
- `mag_converged.py` (patch, near, threads, gauge, compare, reconcile, mcb), `mag_converged_report.py`,
  `mag_converged_text.py`, and the driver `scale/run_magnitude_converged.sh` (resumable; about 40 min
  from scratch).

## Convergence proof
- Exact solve: last refinement step 5e-15 to 6e-14 and relative normal-equation residual 5e-14 to
  3e-13 (both passes, v4.1 and v5.1). Every pass has 1 connected piece, 1 numerical null direction and
  1 gauge row.
- Independent checks: the full phase19 run with `--solver lsqr` (column-scaled, istop 2 after 238 and 240
  iterations) gives the same v4.1 file as `exact`. A sparse LU of the full normal equations agreed to
  5.4e-7 (first version, two-row system).
- Threads (`threads_test.csv`): exact, 1 vs 12 BLAS threads, max |dML| 1.3e-15. Legacy: max 0.86,
  99th percentile 0.075; 98.8% of events differ by more than 1e-4. Only the 1-thread legacy run gives
  the stored offset (+0.413); a 12-thread rerun differs from the stored v4.1 file by up to 0.60.
- `convergence_curve.csv`: LSQR with the phase19 settings, on the stored system, is 0.07 (median) and
  0.66 (99th percentile) from the minimum at 3000 iterations. It reaches istop 2 at 4,600 iterations and
  is still 0.064 off at the 99th percentile there.
- On the new one-row system the same tolerances stop at istop 2 after 3,097 iterations, 0.22 off. The
  stopping test is the problem, not the gauge.
- Reconcile: on the stored v5.1 system, LSQR to istop 2 takes 4,863 iterations, as in D-B091.

## Results (near events, median converged minus stored ML)
| Region | v4.1 | v5.1 |
|---|---|---|
| Blanco | -0.67 | -0.70 |
| JdF ridge/plate | -0.29 | -0.51 |
| Gorda | -0.19 | -0.29 |
| Mendocino | -0.19 | -0.29 |
| Offshore forearc | -0.19 | -0.28 |
| Explorer | +0.01 | -0.08 |
| Endeavour | +0.06 | +0.09 |
| Axial | +0.07 | -0.06 |
| Onshore | +0.005 to +0.014 | +0.010 to +0.015 |

- The 3,500 "mapped" events drop by 0.2-0.45 because their Theil-Sen map is refitted (slope 1.15 -> 1.01).
- v5.1 - v4.1 with both converged: +0.001 overall, -0.004 offshore (stored pair: +0.047 and +0.070);
  only Axial changes (-0.24, relocation).
- ComCat ML anchors, v4.1, offshore and coastal labels: +0.12 -> +0.01. Scatter 0.211 -> 0.174
  (v5.1: 0.230 -> 0.175). Blanco +0.48 -> +0.06, Gorda +0.24 -> +0.02, Mendocino +0.46 -> +0.27.
- Tensor Mw 3-4.5, near events, offshore labels, ML - Mw (median, 95% bootstrap interval): v4.1 +0.11
  [-0.05, +0.20] -> -0.07 [-0.21, +0.02]; v5.1 +0.26 -> -0.02. Inconclusive: 43 events, overlapping
  intervals, and Gorda gets worse (-0.03 -> -0.22). The ComCat ML anchors are the stronger test.
- "Onshore +0.01" holds north of 42 N only. Northern California onshore near events (4,527; 3,708 of them
  "Other onshore") drop by -0.18 to -0.19 with the NC network level, and the onshore Mc box b goes from 0.92 to 1.01.
- Internal check: the Blanco near-event median ML was 2.23 / 2.07 / 2.12 / 1.61 in 2012 / 2013 / 2014 /
  2015 (stored) and is 1.55 / 1.39 / 1.62 / 1.56 (converged). The stored deployment levels of 2012-2014
  were 0.5-0.7 too high relative to 2015.
- phase19 numbers, v4.1: anchor offset +0.413 -> +0.356; Mw_cal offset (near) +0.04 -> +0.19;
  Mw_cal events 35 -> 27.

## Gauge (`gauge_deployments.csv`, `gauge_eigen.csv`)
- Every ocean-bottom deployment level is formally determined to better than 0.1 (NV 2014 0.085; the
  others 0.009-0.046). The stored offsets were a solver effect, not missing ties.
- The formal errors are optimistic. The verifier's event bootstrap gives 1.0-1.7 times larger values
  (NV 2014 0.12; the others 0.011-0.051).
- The thinly tied levels carry about 0.1 of model uncertainty. With picks within 100 km instead of
  150 km: Axial -0.15, Explorer +0.13, Endeavour -0.11.
- The "before" columns of `gauge_deployments.csv` come from different runs:
  - v4.1: a legacy rerun at 12 threads (the stored per-pick table was not kept). Against the stored
    file itself (`anchor_gap_stored_file`): X9 +0.89 -> +0.23, 7D Y2 +0.32 -> +0.03, 7D Y4
    +0.20 -> +0.02, Z5 2013 +0.25 -> -0.09.
  - v5.1: the stored run.
- Thin ties: NV 2014, 7A 2010, NV 2011, OO 2014 (Axial; 66 of its 22,053 events have land picks),
  NV 2010, NV 2013 and 7D Y1.
- X9 (Blanco) has 4 events with land picks. It is tied through 7D Y2 (3,800 shared events), 7D Y3
  (401) and Z5 2013 (108).
- The anchors set only the global offset; the cross-links set each deployment level. Dominated-anchor
  gaps (v4.1, stored file -> converged): 7D Y2 +0.32 -> +0.03, 7D Y4 +0.20 -> +0.02, Z5 2013 +0.25 -> -0.09,
  X9 +0.89 -> +0.23;
  Z5 2014 goes from -0.06 (stored file) to -0.13 and OO 2014 stays at +0.39 (open).
- The northern California land network NC moved by +0.15 (v4.1) against the other land networks.
  This is why Mendocino changes by -0.19 in every year, 2010 included.

## Mc and b
- Paper method (`mc_maps_v41/`): Blanco 2.2 -> 1.6, Gorda 2.1 -> 1.9 (borderline GFT95), Mendocino
  triple junction 1.7 -> 1.5, Endeavour-Explorer 1.8 -> 1.6, offshore 2.2 -> 1.7; onshore unchanged.
- The Blanco 2011-12 -> 2013-15 Mc improvement (2.6 -> 2.2) disappears (1.5 -> 1.6).
- The Blanco M >= 2.5 rate ratios for 2010 and 2016 go from 0.05 and 0.08 to 0.28 and 0.29.
- v5.1 piece estimators (`mc_b/`, `mc_b_change.csv`, `paired_b_converged.csv`): converged v5.1 and
  v4.1 b agree within 0.03 in every box except Axial (+0.10).

## Paper impact
`paper_impact.csv` and section 7 of the report list each figure, table and number, with stored and
converged values. Main items:
- Fig. 2a/2b, Figs. S3-S7 and Table S6 are remade in `/wd1/.../mc_maps_v41`.
- Figs. 1, 3 and 5-6 need remaking (symbol sizes; clustering rerun in `/wd1/.../clusters_zbz_v41`).
- Changed numbers include: 3,416 anchors with scatter 0.21 -> 0.17; the regional Mc ranges; the
  Blanco Mc improvement; the 5%/8% rate ratios; 27,555 -> 27,423 clustered events; the swarm shares;
  and the family sizes (Axial eruption 632 -> 599; June 2015 Blanco 755 -> 850 events).
- The manuscript itself is not edited.

## Missed paper items added after verification
- `v51_clusters_zbz.py`, `v51_interp_common.py` and `v51_struct_common.py` on main read the stored
  v5.1 magnitudes (switch below).
- Repo main.tex line 243 (Overleaf line 265), "Mc 0.75" (Axial) -> 0.85.
- The Methods sentence "inverted for event size, distance attenuation and station terms" is wrong for
  the near-event ML. The Hutton-Boore distance term is fixed and only the event and station-phase terms
  are inverted; the solver is not mentioned.
- Fig. 6B colours symbols by M: remake it with its colour bar.
- Eruption week. The text's 1,021 counts calendar weeks (Mon-Sun) of the analysis set: the stored v4.1
  clustering gives 1,024 that way and 999 in the 7-day bins of Fig. 6A; converged 998 and 975.
  "Below 15 per week within four weeks" still holds (7-day bins after the eruption: stored 45, 11, 4,
  2; converged 41, 5, 4, 1).

## Switching to the converged files (not done here)
- `workflow/10_tomography/params.yaml`, `catalog_versions.v4_1_magnitudes`:
  `/wd1/mdenolle_data/magnitude_converged/v4.1/cascadia_catalog_M_routeA_converged.csv`.
- v4 paper scripts that read `data/magnitude_v4/cascadia_catalog_M_routeA.csv` of the main checkout
  (`v4_mc_maps.py` MAG, `v4_clusters_zbz.py` MAG, `v4_figures.py` line 78, `build_v4_atlas.py`,
  `v4_swarms.py`): point them at the converged v4.1 file.
- `v51_clusters_zbz.py` MAG (line 86) and `v51_struct_common.py` MAG (line 29):
  `/wd1/mdenolle_data/magnitude_converged/v5.1/cascadia_catalog_v5.1_magnitudes_converged.csv`.
- `v51_clusters_zbz.py` MCB (line 90): `data/magnitude_converged/mc_b/mc_b_regional.csv`.
- `v51_interp_common.py`: line 81 should read `cascadia_catalog_v5.1_magnitudes_converged_slim.csv.gz`,
  with V51_MAG set to `<repo>/data/catalog_v5/v5.1_analysis/mag`.
- MD5s of the converged files: `MD5SUMS.txt` (here and under /wd1).

## Not touched
`data/magnitude_v4/`, `/wd1/.../v5.1_analysis/mag/`, the catalogs, Zenodo, other worktrees. The
converged v4.1 file is at `/wd1/mdenolle_data/magnitude_converged/v4.1/cascadia_catalog_M_routeA_converged.csv`
(not copied next to the stored file in the main tree).

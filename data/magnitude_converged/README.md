# Converged Route A magnitudes (D-B091), catalogs v4.1 and v5.1

Decision D-B091 (PI, 2026-11-25 10:24): run the phase19 station-term inversion to convergence and
recompute the v4.1 (paper) and v5.1 magnitudes. Branch `analysis/mag-converged` (from
`analysis/v51-mag` cd0334d6). Report: `magnitude_converged_report.html` (self-contained).
Large outputs: `/wd1/mdenolle_data/magnitude_converged/` (see `RESUME.md` there).

## What changed
Only the phase19 solver. Amplitudes, SNR cut, station epochs, the Hutton-Boore term, the 150 km pick
selection, the 4-MAD rejection, the anchors, the Mw switch at 4.5 and the Mw_cal rule are unchanged.

- `workflow/05_magnitude/phase19_solver.py` (new). Method `exact` (default) eliminates the event terms
  and solves the Schur complement of the station-phase terms directly (1,338 x 1,338 for v4.1), then
  refines twice on the full sparse system. It raises `NotConverged` unless the last refinement step is
  below 1e-8 and the relative normal-equation residual below 1e-10, or if an event with three or more
  picks is cut off from the main network. Method `lsqr` is a column-scaled LSQR (atol = btol = 1e-12,
  istop checked) kept as an independent check. Method `legacy3000` reproduces the stored runs.
- `phase19_preferred_magnitude.py`: `--solver`, `--outdir`, `--tag`; writes `<stem>.provenance.json`.
- `v51_magnitude_table.py`: `--pm`, `--out`, `--tag`, `--solver`, `--v4-mags`.
- `mag_converged.py` (patch, near, threads, gauge, compare, reconcile, mcb), `mag_converged_report.py`,
  `mag_converged_text.py`, and the driver `scale/run_magnitude_converged.sh` (resumable; about 40 min
  from scratch).

## Convergence proof
- Exact solve: last refinement step 3e-14 to 3e-13, relative normal-equation residual 1.7e-13 to 3.1e-13
  (both passes, v4.1 and v5.1).
- Independent checks (v4.1 first pass): a sparse LU of the full normal equations agrees to 5.4e-7; the
  column-scaled LSQR stops with istop 2 after 332 iterations and agrees to 1.7e-6. The full phase19 run
  with `--solver lsqr` gives the same file as `exact`.
- Threads (`threads_test.csv`): exact, 1 vs 12 BLAS threads, max |dML| 1.3e-15. Legacy: max 0.86,
  99th percentile 0.075; 98.8% of events differ by more than 1e-4. Only the 1-thread legacy run gives
  the stored offset (+0.413); a 12-thread rerun differs from the stored v4.1 file by up to 0.60.
- `convergence_curve.csv`: LSQR with the phase19 settings is 0.07 (median) and 0.65 (99th percentile)
  from the minimum at 3000 iterations. It reaches istop 2 at 4,591 iterations and is still 0.064 off
  at the 99th percentile there.

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
- Tensor Mw 3-4.5, near events, offshore labels, ML - Mw: v4.1 +0.11 -> -0.07, v5.1 +0.26 -> -0.02.
- Internal check: the Blanco near-event median ML was 2.23 / 2.07 / 2.12 / 1.61 in 2012 / 2013 / 2014 /
  2015 (stored) and is 1.55 / 1.39 / 1.62 / 1.56 (converged). The stored deployment levels of 2012-2014
  were 0.5-0.7 too high relative to 2015.
- phase19 numbers, v4.1: anchor offset +0.413 -> +0.356; Mw_cal offset (near) +0.04 -> +0.19;
  Mw_cal events 35 -> 27.

## Gauge (`gauge_deployments.csv`, `gauge_eigen.csv`)
- Every ocean-bottom deployment level is formally determined to better than 0.1 (NV 2014 0.085; the
  others 0.009-0.046). The stored offsets were a solver effect, not missing ties.
- Thin ties: NV 2014, 7A 2010, NV 2011, OO 2014 (Axial; 66 of its 22,053 events have land picks),
  NV 2010, NV 2013 and 7D Y1.
- X9 (Blanco) has 4 events with land picks. It is tied through 7D Y2 (3,800 shared events), 7D Y3
  (401) and Z5 2013 (108).
- The anchors set only the global offset; the cross-links set each deployment level. Dominated-anchor
  gaps (v4.1): 7D Y2 +0.30 -> +0.03, 7D Y4 +0.17 -> +0.02, Z5 2013 +0.22 -> -0.09, X9 +0.87 -> +0.23;
  Z5 2014 stays at -0.13 and OO 2014 at +0.39 (open).
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

## Not touched
`data/magnitude_v4/`, `/wd1/.../v5.1_analysis/mag/`, the catalogs, Zenodo, other worktrees. The
converged v4.1 file is at `/wd1/mdenolle_data/magnitude_converged/v4.1/cascadia_catalog_M_routeA_converged.csv`
(not copied next to the stored file in the main tree).

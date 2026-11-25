"""Text and tables of the converged-magnitude report (called by mag_converged_report.py).
Numbers are read from the tables at build time; the prose states what they show."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import mag_converged as mc


def _v(df, q, col):
    x = df.query(q)[col]
    return float(x.iat[0]) if len(x) else np.nan


def sections(t, tab, img, pi):
    runs, thr, cur = t("phase19_runs"), t("threads_test"), t("convergence_curve")
    g, ge = t("gauge_deployments"), t("gauge_eigen")
    anc, ten = t("comcat_anchors"), t("tensors")
    r41, r51, d5 = t("v41_by_region"), t("v51_by_region"), t("v51_minus_v41")
    rec, recm = t("reconcile_db091"), t("reconcile_misfit")
    mcb, pb = t("mc_b_change"), t("paired_b_converged")
    gdt, gmu, tbr, onsp = t("gauge_design_test"), t("gauge_ml_unchanged"), t("tensors_by_region"), t("v41_onshore_split")

    def ten_ci(sol, ev, cls):
        x = ten[(ten.solution == sol) & (ten.events == ev) & (ten.Mw_class == cls)]
        return (f"{x.ML_minus_Mw.iat[0]:+.2f} [{x.ci95_lo.iat[0]:+.2f}, {x.ci95_hi.iat[0]:+.2f}]"
                if len(x) and "ci95_lo" in x else "?")
    cm = json.load(open(f"{mc.SMALL}/compare_meta.json")) if os.path.exists(f"{mc.SMALL}/compare_meta.json") else {}
    gm = json.load(open(f"{mc.SMALL}/gauge_meta.json")) if os.path.exists(f"{mc.SMALL}/gauge_meta.json") else {}
    c41 = runs.set_index("run").loc["v4.1 converged"]
    l41 = runs.set_index("run").loc["v4.1 legacy rerun"]
    c51 = runs.set_index("run").loc["v5.1 converged"]
    ex = thr.set_index("solver").loc["exact"]
    lg = thr.set_index("solver").loc["legacy3000"]
    lr = cm.get("legacy_rerun_vs_stored", {})

    def reg(df, r, col="median"):
        return _v(df, f"region == '{r}'", col)

    def anc_med(sol, val, col="median"):
        return _v(anc, f"solution == '{sol}' and value == '{val}'", col)

    def ten_med(sol, ev, cls):
        return _v(ten, f"solution == '{sol}' and events == '{ev}' and Mw_class == '{cls}'", "ML_minus_Mw")

    d5ac = _v(d5, "region == 'all'", "d_conv")
    d5as = _v(d5, "region == 'all'", "d_stored")
    d5oc = _v(d5, "region == 'offshore and coastal labels'", "d_conv")
    d5os = _v(d5, "region == 'offshore and coastal labels'", "d_stored")
    d5xc = _v(d5, "region == 'Axial'", "d_conv")
    cp = cur[cur.solver.str.startswith("LSQR (phase19 settings")]
    cn = cur[cur.solver.str.startswith("LSQR (phase19 tolerances, new")]
    cs = cur[cur.solver.str.startswith("LSQR column")]
    at3000 = cp[cp.iter_lim == 3000].iloc[0]
    conv_row = cp[cp.istop.isin([1, 2])].head(1)
    s_conv = cs[cs.istop.isin([1, 2])].head(1)
    h = []
    h.append("<h1>Converged Route A magnitudes for catalogs v4.1 and v5.1</h1>")
    h.append("<p>Decision D-B091 (PI, 2026-11-25 10:24). Branch <code>analysis/mag-converged</code>; "
             "large outputs in <code>/wd1/mdenolle_data/magnitude_converged</code>. Only the phase19 solver "
             "changed: amplitudes, SNR cut, station epochs, the Hutton-Boore distance term, the 150 km pick "
             "selection, the 4-MAD rejection, the ComCat ML anchors, the Mw switch at 4.5 and the Mw_cal rule "
             "are those of the stored runs.</p>")
    o41 = _v(anc, "solution == 'v4.1 stored' and value == 'offshore and coastal labels'", "median")
    n41a = _v(anc, "solution == 'v4.1 converged' and value == 'offshore and coastal labels'", "median")
    n41, n51 = t("v41_by_region_near"), t("v51_by_region_near")
    h.append("<div class='box'><b>Independent verification.</b> A separate verifier re-derived the converged "
             "magnitudes (scripts and outputs in <code>/wd1/mdenolle_data/magnitude_converged/verify/</code>) and found "
             "them fit to replace the paper's magnitudes. Its checks: the solution is the true minimum, it is set by "
             "the data and not by the damping (damping 1e-4 changes ML by at most 0.001, damping 1e-2 by 0.026 at the "
             "99th percentile), and the anchor test is not circular (offset from half of the anchors, gap measured on the "
             "other half: offshore +0.116 stored, +0.009 converged, in 99.8% of 500 splits closer to zero). Its "
             "must-fix items are resolved here: one gauge row with a null-direction check (section 1b); the text that "
             "said two gauge rows fix the P-S difference is corrected; the isolated term no longer appears in the "
             "station-term tables.</div>")
    h.append("<div class='box'><b>Summary.</b><ul>"
             "<li>The phase19 inversion is now solved exactly: the 161,093 event terms are eliminated and the "
             "1,337 station-phase terms come from a direct solve of their Schur complement, "
             "followed by two refinement steps. The last refinement step is at most 6e-14 and the relative "
             "normal-equation residual at most 3e-13. Two independent solvers agree with it: a sparse LU "
             "factorisation of the full normal equations (5e-7) and a column-scaled LSQR, which converges in "
             f"about 250 iterations to the same output file. One and twelve BLAS threads give the same ML to {ex.max_abs:.0e}.</li>"
             f"<li>The stored solver, LSQR stopped at 3000 iterations, is {at3000.median_abs:.2f} from the "
             f"minimum for the median event and {at3000.p99_abs:.2f} at the 99th percentile. It is not "
             f"reproducible: 1 and 12 threads differ by up to {lg.max_abs:.2f} (99th percentile {lg.p99_abs:.3f}), "
             f"and a rerun at 12 threads differs from the stored v4.1 file by up to {lr.get('max', np.nan):.2f}.</li>"
             f"<li>v4.1 (the paper), converged minus stored ML, near events (median by region): Blanco "
             f"{reg(n41, 'Blanco'):+.2f}, JdF ridge/plate {reg(n41, 'JdF ridge/plate'):+.2f}, Gorda {reg(n41, 'Gorda'):+.2f}, "
             f"Mendocino {reg(n41, 'Mendocino'):+.2f}, offshore forearc {reg(n41, 'Offshore forearc'):+.2f}, Endeavour "
             f"{reg(n41, 'Endeavour'):+.2f}, Explorer {reg(n41, 'Explorer'):+.2f}, Axial {reg(n41, 'Axial'):+.2f}, onshore "
             f"+0.005 to +0.014. v5.1 near events: Blanco "
             f"{reg(n51, 'Blanco'):+.2f}, JdF {reg(n51, 'JdF ridge/plate'):+.2f}, Gorda and Mendocino {reg(n51, 'Gorda'):+.2f}, "
             f"offshore forearc {reg(n51, 'Offshore forearc'):+.2f}, Explorer {reg(n51, 'Explorer'):+.2f}, Axial "
             f"{reg(n51, 'Axial'):+.2f}, onshore +0.01 to +0.015 "
             "(the D-B091 numbers); v5.1 moves more than v4.1 because its stored run had stopped further from the "
             "minimum. The 3,500 'mapped' events (recorded only beyond 150 km) drop by 0.2-0.45 everywhere, onshore "
             "included, because the Theil-Sen line that maps them is refitted to the new near ML.</li>"
             f"<li>With both catalogs converged, v5.1 minus v4.1 is {d5ac:+.3f} "
             f"(stored pair {d5as:+.3f}) and "
             f"{d5oc:+.3f} offshore "
             f"(stored {d5os:+.3f}). The +0.07 offshore "
             "shift reported in the v5.1 magnitude piece was a solver artefact; only Axial keeps a real change "
             f"({d5xc:+.2f}, from the relocation).</li>"
             f"<li>External references, v4.1: ComCat ML anchor gap at offshore and coastal labels {o41:+.2f} -> "
             f"{n41a:+.2f}; anchor scatter 0.211 -> {c41.anchor_std:.3f}; tensor Mw 3-4.5 "
             f"at offshore labels, near events, ML - Mw {ten_med('v4.1 stored', 'near', '3-4.5, offshore and coastal labels'):+.2f}"
             f" -> {ten_med('v4.1 converged', 'near', '3-4.5, offshore and coastal labels'):+.2f} (v5.1 "
             f"{ten_med('v5.1 stored', 'near', '3-4.5, offshore and coastal labels'):+.2f} -> "
             f"{ten_med('v5.1 converged', 'near', '3-4.5, offshore and coastal labels'):+.2f}). The ComCat ML anchors "
             "favour the converged solution clearly; the tensor comparison does not decide between the two "
             "(43 events, overlapping 95% intervals, Gorda worse, section 4).</li>"
             "<li>'Onshore' is not uniform: near events at onshore labels south of 42 N (northern California, 4,527 "
             "events in v4.1, 3,708 of them labelled 'Other onshore') drop by -0.18 to -0.19 with the northern "
             "California network NC; north of 42 N they change by "
             "+0.01. The onshore Mc box b goes from 0.92 to 1.01.</li>"
             "<li>Gauge rows (verifier must-fix): the stored design had two gauge rows, of which the S row was "
             "absorbed by one isolated term (7D.J35C@2013-08-09|S, C = -303). Now the isolated picks are dropped, one "
             "gauge row is used, and the solver checks that the gauge rows equal the null directions. ML is unchanged "
             "(largest difference 0.0015 in the design test, see section 1b for the full-chain comparison).</li>"
             "<li>Gauge: every ocean-bottom deployment is tied to the land network well enough that its formal "
             "level error is below 0.1 (most below 0.03); a bootstrap gives errors 1.0-1.7 times larger (NV 2014 "
             "about 0.12), and the thinly tied ones carry about 0.1 of model uncertainty. The stored offsets came "
             "from the solver, not from weak ties. The ties that are thin are NV 2014, 7A 2010, NV 2011 and OO 2014 (Axial). X9 (Blanco) has "
             "only 3-4 events with land picks; 3,800 events shared with 7D Y2, 401 with 7D Y3 and 108 with Z5 2013 "
             "carry its level. In "
             "phase19 the anchors set one global offset only; the cross-links set every deployment level. The "
             "northern California land network NC also moved, by +0.15 against the other land networks (v4.1). That is "
             "why Mendocino changes by -0.19 in every year, 2010-2011 included.</li>"
             "<li>Mc (paper method, v4_mc_maps.py on v4.1): Blanco 2.2 -> 1.6, Gorda 2.1 -> 1.9, Mendocino triple "
             "junction 1.7 -> 1.5, Endeavour-Explorer 1.8 -> 1.6, offshore pooled 2.2 -> 1.7; onshore unchanged. The "
             "Blanco Mc 'improvement' between 2011-2012 and 2013-2015 (2.6 -> 2.2) disappears (1.5 -> 1.6), and the "
             "Blanco M &gt;= 2.5 rate of 2010 and 2016 relative to 2013-2015 rises from 5% and 8% to 28% and 29%. The "
             "paper's Fig. 2, Figs. S3-S7, Table S6, the "
             "clustering (Figs. 5-6, Table S5) and about 30 numbers change (list in the last section).</li>"
             "</ul></div>")

    # ---- 1 solver
    h.append("<h2>1. Solver change and convergence proof</h2>")
    h.append("<p>phase19 fits station ML = M_i + C_jp to the Hutton-Boore station magnitudes within 150 km, with "
             "damping 1e-3 and gauge rows of weight 1000 (section 1b for the gauge). The stored runs solved it with "
             "LSQR stopped at 3000 iterations (istop 7). LSQR is slow on this system for two reasons. First, the column norms range from 1 (an "
             "event with one pick) to 130 (a station-phase term with 16,876 picks), and the gauge rows have weight "
             "1000. Second, moving a whole "
             "deployment against its events changes the misfit little per unknown: the eigenvalue of that mode "
             "is the deployment's stiffness divided by its number of terms plus events (X9 0.25, Z5 2013 0.48, "
             "OO 2014 0.009, against 1.5e4 at the top of the data spectrum). Starting from zero station terms, "
             "LSQR resolves these modes last. Until it does, the large site terms of sediment-covered "
             "ocean-bottom stations stay in the event terms, so offshore ML comes out too high.</p>")
    h.append("<p>The new default, <code>phase19_solver.solve(method='exact')</code>, eliminates the event terms "
             "(their block of the normal equations is diagonal). The 1,337 x 1,337 Schur complement of the station "
             "terms is solved by a symmetric eigendecomposition, and two refinement steps follow on the full sparse "
             "system. It raises <code>NotConverged</code> unless the last refinement step is below 1e-8 and the "
             "relative normal-equation residual is below 1e-10, or unless the gauge rows equal the null directions "
             "(section 1b). "
             "<code>--solver lsqr</code> runs a column-scaled LSQR with atol = btol = 1e-12 and fails unless istop "
             "is 1 or 2. <code>--solver legacy3000</code> reproduces the stored runs.</p>")
    h.append(tab(runs, cols=["run", "solver", "istop", "itn", "r_norm", "normal_res_rel", "anchor_offset", "n_anchors",
                             "anchor_std", "mapped_intercept", "mapped_slope", "mapped_std", "mw_cal_near",
                             "mw_cal_mapped", "M_type_ML", "M_type_Mw", "M_type_Mw_cal"]))
    h.append("<p>Exact-solver diagnostics (both passes, v4.1 and v5.1, one gauge row): last refinement step 5e-15 "
             "to 6e-14, relative normal-equation residual 5e-14 to 3e-13; in every pass 1 connected piece, 1 "
             "numerical null direction, 1 gauge row. The full phase19 run with <code>--solver lsqr</code> "
             "(column-scaled, istop 2 after 238 and 240 iterations) gives the same v4.1 output file as the exact "
             "run (largest difference 0.000). Independent check on the first version (two-row system, v4.1 first "
             "pass): a sparse LU factorisation of the full 162,515 x 162,515 normal equations differs by at most "
             "5.4e-7.</p>")
    h.append("<h3>Thread count</h3><p>phase19 'near' ML of v4.1 (both passes and the anchor offset) in separate "
             "processes with 1 and 12 BLAS threads:</p>" + tab(thr, fmt="{:.3g}"))
    h.append(f"<p>The legacy solver at 1 thread gives the stored anchor offset (+0.4132; stored +0.413) and at 12 "
             f"threads +0.4055. A rerun at 12 threads differs from the stored v4.1 file by up to "
             f"{lr.get('max', np.nan):.2f} (99th percentile {lr.get('p99', np.nan):.3f}) on the near events.</p>")
    h.append("<h3>How far LSQR stops from the minimum</h3><p>LSQR on the final v4.1 pick set, stopped at increasing "
             "iteration counts, against the exact solution (ML with the anchor offset recomputed, near events):</p>"
             + tab(cur, fmt="{:.3g}"))
    if len(conv_row):
        r_ = conv_row.iloc[0]
        h.append(f"<p>With the phase19 tolerances LSQR reaches istop 2 after {int(r_.itn):,} iterations and is then "
                 f"still {r_.p99_abs:.3f} from the minimum at the 99th percentile (largest {r_.max_abs:.2f}). istop 2 "
                 "bounds the gradient, not the error: in a soft direction the error is the gradient divided by a "
                 "small eigenvalue. The D-B091 check (4,863 iterations on the stored v5.1 pick set) was therefore "
                 "close, but not exact (section 4).</p>")
    if len(cn) and cn.istop.isin([1, 2]).any():
        r2 = cn[cn.istop.isin([1, 2])].iloc[0]
        h.append(f"<p>The same tolerances on the new one-row system stop with istop 2 after {int(r2.itn):,} iterations, "
                 f"{r2.p99_abs:.2f} from the minimum at the 99th percentile: the stopping test is the problem, not "
                 "the gauge. The column-scaled LSQR reaches the minimum to 1e-8 in a few hundred iterations.</p>")
    h.append("<h3>1b. Gauge rows and null directions (verifier must-fix 1)</h3>"
             "<p>The data fix every M_i + C_jp but not a common shift: on each connected piece of the event / "
             "station-phase graph, all event terms +delta and all station terms -delta leave the misfit unchanged. "
             "The gauge needs one row per such null direction; any further row constrains the fit. The stored runs, "
             "and the first version of this work, used two rows (sum of the P terms = 0, sum of the S terms = 0). The "
             "graph had two pieces: the main network and one isolated S term, 7D.J35C@2013-08-09|S (9 picks on 9 "
             "one-pick events). The S row was absorbed by that term, which ended at C = -303 (v4.1) and -295 (v5.1); "
             "stored legacy runs -255 and -234. The two rows therefore fixed the two null directions, and the P - S "
             "level difference of the main network stayed free (mean S term - mean P term +0.454). The earlier text "
             "of this report said that the two rows fix the P - S difference; that was wrong. Without the isolated "
             "term, though, the two rows would have forced mean S = mean P: the misfit rises by about 2,900 and "
             "regional ML moves by up to -0.22. The convergence test of the solver does not see this, because the "
             "constrained problem is solved exactly too.</p>"
             "<p>Fix: picks not linked to the main network are dropped before each pass "
             "(<code>main_component()</code>; their events have fewer than 3 picks, so no ML is lost). ONE gauge row "
             "is used (sum of the P terms = 0, which keeps the P terms and the main-network S terms of the previous "
             "solution). <code>check_gauge()</code> counts the null directions from the graph (connected pieces) and "
             "numerically (eigenvalues of the reduced system below 1e-9 of the largest). It raises NotConverged "
             "unless both equal the number of gauge rows and the gauge rows are not orthogonal to the null space. The "
             "anchor offset absorbs any single gauge row, so ML does not depend on which one is chosen. The isolated "
             "term no longer appears in the converged station-term tables. The stored "
             "<code>phase19_station_terms_v51.csv</code> still lists it (C about -234); that value is a gauge "
             "artefact and should be ignored.</p>"
             "<p>Design test on the final v4.1 pick set (exact solver; the 9 isolated picks added back where kept):</p>"
             + tab(gdt) + "<p>The new full chain (both passes, rejection, anchor offset) against the previous "
             "converged run with two rows, unrounded near-event ML:</p>" + tab(gmu, fmt="{:.2e}"))
    h.append(img("fig_convergence", "LSQR error against iterations (v4.1, near events), each against the exact "
                 "solution of its own system. Blue: the stored system with the phase19 settings. Green: the new "
                 "one-row system with the same tolerances. Red: the column-scaled LSQR."))

    # ---- 2 gauge
    h.append("<h2>2. The gauge: how each ocean-bottom deployment is tied to the land network</h2>")
    h.append("<p>Groups: every seafloor station-epoch, grouped by network and deployment year (7D = Cascadia "
             "Initiative years Y1-Y4); everything on land is the reference. Per group, on the final pick set of the "
             "converged run: <b>n_events_with_land</b> = events with picks on the group and on land stations; "
             "<b>n_events_with_other_obs</b> = events shared with another ocean-bottom group; <b>n_anchors</b> = ComCat ML "
             "anchors among its near events (<b>dominated</b>: anchors with at least half of their picks on the "
             "group); <b>stiffness</b> k = misfit increase per delta^2 when all its terms move by +delta and the "
             "events are re-fitted (the Schur complement; equal to the sum over events of n_G n_other / n, an "
             "effective number of tie picks); <b>se_level</b> = formal standard error of the mean term of the group "
             "minus the mean land term (sigma = "
             f"{gm.get('v4.1', {}).get('sigma', np.nan):.3f}, from the full inverse of the reduced system); "
             "<b>dphi_opt_0.5</b> and <b>dphi_naive_0.5</b> = misfit increase for delta = 0.5 with the events "
             "re-fitted, and with every event that has a pick on the group moved by -delta (the +delta/-delta test; "
             f"total misfit {gm.get('v4.1', {}).get('phi', np.nan):,.0f}); <b>dC_minus_land</b> = change of the mean term "
             "(converged minus stored) relative to the land mean; <b>dML_dominated</b> = median ML change of the "
             "events with at least half of their picks on the group; <b>anchor_gap</b> = median ML - ComCat ML of "
             "the dominated anchors.</p>")
    cols = ["group", "n_terms", "n_picks", "n_events", "n_events_with_land", "n_events_with_other_obs", "n_anchors",
            "n_anchors_dominated", "stiffness", "rigid_mode_eig", "se_level", "dphi_opt_0.5", "dphi_naive_0.5",
            "dC_minus_land", "dML_dominated", "anchor_gap_unconv", "anchor_gap_conv", "main_region"]
    for lab in ("v4.1", "v5.1"):
        h.append(f"<h3>{lab}, ocean-bottom deployments</h3>" + tab(g[(g.catalog == lab) & (g.level == "deployment")], cols=cols))
    h.append("<h3>Land networks, each against the rest of the land network (networks with at least 1,000 picks)</h3>"
             "<p>Here 'land' in the column names means the land terms outside the network.</p>")
    for lab in ("v4.1", "v5.1"):
        h.append(f"<h3>{lab}, land networks</h3>" + tab(g[(g.catalog == lab) & (g.level == "land network")], cols=cols))
    h.append("<p>The 7D sub-arrays (J, FN, FS, G, M per year) are in <code>gauge_deployments.csv</code> (level "
             "sub-array).</p>")
    h.append("<p>Source of the 'before' columns (dC_minus_land, dML_dominated, anchor_gap_unconv): for v4.1, a "
             "legacy3000 rerun at 12 threads, because the stored run did not keep its per-pick table. "
             "anchor_gap_stored_file uses the stored v4.1 file itself. For v5.1, the stored v5.1 run.</p>"
             "<p>Uncertainty of the levels. The formal s.e. assumes independent Gaussian residuals and a correct "
             "model. The verifier's bootstrap (32 replicates, resampling events) gives errors 1.0-1.7 times the "
             "formal ones (7D Y2 0.016 vs 0.010, X9 0.023 vs 0.016, NV 2014 0.12 vs 0.085). A model test shifts the "
             "thinly tied levels by about 0.1: with picks within 100 km instead of 150 km, Axial -0.15, Explorer "
             "+0.13, Endeavour -0.11. Blanco, Gorda and Mendocino move by -0.04 to +0.04 in that test. "
             "The NC terms sit 0.19 below the other land networks (bootstrap s.d. 0.014).</p>")
    h.append("<p>What the table shows:</p><ul>"
             "<li>No deployment is free. The largest formal error of a deployment level is 0.085-0.091 (NV 2014, "
             "bootstrap 0.12, "
             "4 terms, 2 or fewer events with land picks); every other is 0.009-0.046 (bootstrap 0.011-0.051). A 0.5 shift of a deployment "
             "costs from 2 (NV 2014) to 6,900 (7D Y4) in misfit, out of about 72,000.</li>"
             "<li>Thin ties (stiffness below about 400): NV 2014 (k 8-10), 7A 2010 (69-75), NV 2011 (117-119; 0-2 "
             "events with land picks), OO 2014 (194-200; 66-71 of its 22,000 events have land picks), NV 2010 (244-256), "
             "NV 2013 (360-367) and 7D Y1 (376-406). Their levels rest on few links. OO 2014 carries the Axial "
             "catalog.</li>"
             "<li>X9 2012 (Blanco) has only 3-4 events with land picks. It is tied through 3,800 events shared with "
             "7D Y2, 401 with 7D Y3 and 108 with Z5 2013 (stiffness 5,300, s.e. 0.016); 7D Y2 in turn shares 2,615 "
             "events with land. It moved most (-0.67 to -0.71) because LSQR reaches this chain last, not because the "
             "data leave it free.</li>"
             "<li>Anchors versus cross-links. In phase19 the 3,416 anchors (84% onshore) set one global offset and do "
             "not enter the deployment levels; the cross-links set every level. The anchors are an independent check. "
             "Where a deployment has 9 or more dominated anchors, convergence reduces the gap. Against the stored v4.1 "
             "file: 7D Y2 +0.32 -> +0.03 (24 anchors), 7D Y4 +0.20 -> +0.02 (43), Z5 2013 +0.25 -> -0.09 (38), X9 "
             "+0.89 -> +0.23 (9). The 12-thread legacy rerun gives +0.30, +0.17, +0.22, +0.87. Z5 2014 gets worse: "
             "-0.06 in the stored file (-0.12 in the rerun) -> -0.13 (13 anchors). OO 2014 (Axial) stays at +0.39/+0.40 with only 4 anchors; the 2 anchors in "
             "the Axial region are +0.9 above ComCat in both solutions. Axial is the one place where the anchors "
             "disagree with the cross-link level, and its ties are thin (open problem 1).</li>"
             "<li>The land network is not one rigid block either. The northern California network NC (128 terms) moves "
             "by +0.15 (v4.1) and +0.25 (v5.1) against the other land networks, and BK by +0.10/+0.17. Events dominated "
             "by NC change by -0.17/-0.29. This is why Mendocino, Gorda and the offshore forearc change by the same "
             "-0.19 (v4.1) in every year, 2010 and 2011 included, when no ocean-bottom station was recording. NC is "
             "stiffly tied (s.e. 0.010; 21,030 events shared with the other land networks), so this too was the "
             "solver. Its 54 dominated anchors go from +0.39 to +0.24 above ComCat ML; the remaining +0.24 may be a "
             "scale difference between the NC local magnitudes in ComCat and the PNSN ones that dominate the anchor "
             "offset.</li></ul>")
    h.append("<h3>Near-null directions</h3><p>The smallest eigenvalues of the reduced system (station terms, events "
             "eliminated, global shift and the isolated J35C S term removed). Each is a pattern of station terms "
             "that costs that much misfit per unit norm; <b>energy_share</b> is the share of the pattern on one "
             "deployment and <b>coherent</b> = 1 for a rigid shift of that deployment, 0 for an internal "
             "see-saw.</p>" + tab(ge.head(12)) + tab(ge[ge.catalog == "v5.1"].head(8)))
    h.append("<p>The softest patterns lie inside one deployment: 7D Y3 (the FN Grays Harbor and J/M sub-arrays move "
             "against each other), 7D Y1 J stations against the rest, NV 2010, NV 2014 and OO 2014. No soft "
             "pattern moves the whole offshore network against the land network.</p>")
    h.append(img("fig_gauge_v41", "Left: formal standard error of each deployment level against its stiffness "
                 "(v4.1). Right: median ML change of the events dominated by a deployment against the eigenvalue of "
                 "its rigid mode in the full system; small values are the directions LSQR converges on last."))

    # ---- 3 v4.1
    h.append("<h2>3. Catalog v4.1 (the paper's magnitudes)</h2>")
    h.append("<p>New file <code>/wd1/mdenolle_data/magnitude_converged/v4.1/cascadia_catalog_M_routeA_converged.csv</code> "
             "(130,769 events, same columns and events as the stored <code>data/magnitude_v4/cascadia_catalog_M_routeA.csv</code>; "
             "the v4.1 patch is applied as in <code>utils/fix_catalog_v4.py</code>: 172113 removed, 900001-900003 "
             "copied from the stored file as ComCat Mw). Provenance: <code>.provenance.json</code> next to it (solver "
             "diagnostics of both passes, anchor offset and scatter, mapping fit, Mw_cal offsets, M_type counts, "
             "thread settings, MD5 of the stored and new files, git commit).</p>")
    h.append("<p>phase19 numbers, stored -> converged: anchor offset +0.413 -> "
             f"{c41.anchor_offset:+.3f}; anchor scatter 0.211 -> {c41.anchor_std:.3f}; 'mapped' fit "
             f"ML = -0.353 + 1.154 ML_inv (std 0.76) -> {c41.mapped_intercept:+.3f} + {c41.mapped_slope:.3f} ML_inv "
             f"(std {c41.mapped_std:.2f}); Mw_cal offset (near) +0.04 -> {c41.mw_cal_near:+.2f}, (mapped) +0.97 -> "
             f"{c41.mw_cal_mapped:+.2f}; M_type ML 130,652 / Mw 80 / Mw_cal 35 (v4.0 run) -> "
             f"{int(c41.M_type_ML):,} / {int(c41.M_type_Mw)} / {int(c41.M_type_Mw_cal)} (before the v4.1 patch). "
             "The near-event Mw_cal offset grows because the large offshore events now have lower ML.</p>")
    h.append("<h3>By region (converged minus stored ML, all events with ML)</h3>" + tab(r41))
    h.append("<h3>By region, near events (3 or more picks within 150 km; the D-B091 set)</h3>" + tab(n41))
    h.append("<h3>By region, mapped events</h3>" + tab(t("v41_by_region_mapped")))
    h.append("<h3>Near events at onshore labels, north and south of 42 N</h3>" + tab(onsp)
             + "<p>The onshore +0.01 holds north of 42 N only. South of 42 N (northern California) near events drop by "
             "-0.19, with the NC network level (section 2). The onshore box of the Mc maps changes b from 0.92 to "
             "1.01 (section 6).</p>")
    h.append("<h3>By ML method</h3>" + tab(t("v41_by_method")))
    h.append("<h3>By stored ML</h3>" + tab(t("v41_by_mag")))
    h.append("<h3>By distance to the nearest station with an amplitude (v4 hypocentre)</h3>" + tab(t("v41_by_dist")))
    h.append("<p>The change follows the region (the deployment that records the event), not the magnitude or the "
             "distance as such: the M 2-3 bin moves most (-0.61) because it is dominated by Blanco events, and the "
             "0-10 km bin least because it is dominated by onshore and Axial events.</p>")
    ry = t("v41_region_year")
    if ry is not None:
        h.append("<h3>Internal check: median ML of near events by region and year</h3><p>The ocean-bottom deployments "
                 "change from year to year, so a deployment level that is off shows as a jump of the regional median "
                 "between years. Blanco, stored: 2.23 (2012), 2.07 (2013), 2.12 (2014), 1.61 (2015, recorded by Z5 2014); "
                 "converged: 1.55, 1.39, 1.62, 1.56. The 2012-2014 deployments (X9, Z5 2013, 7D Y2/Y3) were 0.5-0.7 "
                 "too high relative to the 2015 one in the stored catalog; the converged levels remove the jump. "
                 "Rows with at least 50 events.</p>"
                 + tab(ry[ry.region.isin(["Blanco", "Gorda", "Mendocino", "JdF ridge/plate", "Offshore forearc",
                                          "Explorer", "Axial", "Puget/Georgia"])]))
    h.append("<h3>M_type transitions (rows stored, columns converged)</h3>" + tab(t("v41_M_type")))
    h.append(img("fig_map_change_v41", "v4.1, converged minus stored ML (near events), and the stations with "
                 "amplitudes by deployment."))

    # ---- external references
    h.append("<h2>4. External references and the D-B091 numbers</h2>")
    h.append("<h3>ComCat ML anchors (median ML - ComCat ML, near events)</h3>")
    a2 = anc[anc.group == "split"]
    h.append(tab(a2, cols=["solution", "value", "n", "median", "p25", "p75", "std"]))
    h.append(tab(anc[anc.group == "region"].pivot(index="value", columns="solution", values="median")
                 .reindex([r for r in mc.REG_ORDER if r in set(anc.value)]).reset_index()))
    h.append(img("fig_anchors", "Median ML - ComCat ML of the anchors by region, stored and converged; the "
                 "number under each label is the number of anchors."))
    h.append("<h3>ComCat moment tensors (median ML - Mw)</h3>")
    h.append(tab(ten.pivot_table(index=["events", "Mw_class"], columns="solution", values="ML_minus_Mw").reset_index()))
    h.append(tab(ten, cols=["solution", "events", "Mw_class", "n", "ML_minus_Mw", "ci95_lo", "ci95_hi"]))
    h.append("<h3>Tensor Mw 3-4.5, near events, by region (median ML - Mw)</h3>" + tab(tbr))
    h.append("<p>The tensor comparison is inconclusive. For near events at offshore labels with Mw 3-4.5 "
             f"(43 events), ML - Mw goes from {ten_ci('v4.1 stored', 'near', '3-4.5, offshore and coastal labels')} to "
             f"{ten_ci('v4.1 converged', 'near', '3-4.5, offshore and coastal labels')} (v4.1; median and bootstrap "
             f"95% interval) and from {ten_ci('v5.1 stored', 'near', '3-4.5, offshore and coastal labels')} to "
             f"{ten_ci('v5.1 converged', 'near', '3-4.5, offshore and coastal labels')} (v5.1). The intervals "
             "overlap, and by region Gorda gets worse (-0.03 -> -0.22, 10 events) while Mendocino and the offshore "
             "forearc get better. The tensors do not reject either solution; the ComCat ML anchors (549 offshore "
             "and coastal, 3,416 in all) are the stronger test. Onshore, ML - Mw stays at +0.05 to +0.12. Counted over all events, the offshore "
             "value goes to -0.21/-0.23 because of the 11-14 'mapped' tensor events: their ML comes from the "
             "Theil-Sen map of the inversion ML, whose slope fell from 1.15 to 1.01 when the near ML it is fitted to "
             "came down. phase19 already flags mapped ML above about 4 as lower bounds. Above Mw 4.5 the tensor "
             "Mw is used, so these values do not enter M.</p>")
    h.append("<h3>D-B091 on the stored pick sets</h3><p>These comparisons use the stored system (stored pick set, "
             "isolated term included, two gauge rows), which is what the stored runs and the D-B091 check solved; "
             "its exact solution gives the same ML as the new one-row system (column "
             "max_abs_dML_exact_new_vs_stored_system). On the pick set of the stored v5.1 run, the exact solver "
             "reproduces the D-B091 changes, and the D-B091 LSQR to istop 2 (4,863 iterations) is within 0.006 of it "
             "in every region except Axial (-0.03); its largest event error is 0.18. On the v4.1 pick set (the stored "
             "per-pick table was not kept, so a 12-thread legacy rerun stands in) the same LSQR is within 0.02 except "
             "Axial (+0.06). The data misfit drops by 274 (v5.1, 0.38%) and 184 (v4.1, 0.25%) from the stored to the "
             "exact solution. Redoing the first pass and the 4-MAD rejection with the exact solver changes only tens "
             "of picks per deployment and the regional medians by at most 0.02 (columns full_chain_minus_stored and "
             "rejection_effect). The larger Explorer and Endeavour changes in the all-event region tables come from "
             "the 'mapped' events (Explorer: 485 of 1,326), not from the rejection.</p>")
    h.append(tab(recm))
    h.append(tab(rec))
    for lab in ("v4.1", "v5.1"):
        x = t(f"reconcile_rejection_{lab}")
        if x is not None:
            h.append(f"<p>{lab}: picks kept only by the stored chain or only by the exact chain, by deployment</p>"
                     + tab(x))

    # ---- 5 v5.1
    h.append("<h2>5. Catalog v5.1</h2>")
    h.append("<p>New files in <code>/wd1/mdenolle_data/magnitude_converged/v5.1/</code>: "
             "<code>cascadia_catalog_M_v51_converged.csv</code> (phase19) and "
             "<code>cascadia_catalog_v5.1_magnitudes_converged.csv</code> (the per-row table of v51_magnitude_table.py, "
             "172,342 rows, with M_v4 / ML_v4 from the converged v4.1). phase19: anchor offset "
             f"{c51.anchor_offset:+.3f} (stored +0.420), anchor scatter {c51.anchor_std:.3f} (stored 0.230), M_type "
             f"ML {int(c51.M_type_ML):,} / Mw {int(c51.M_type_Mw)} / Mw_cal {int(c51.M_type_Mw_cal)}.</p>")
    h.append("<h3>By region (converged minus stored, all events with ML)</h3>" + tab(r51))
    h.append("<h3>By region, near events</h3>" + tab(n51))
    h.append("<h3>By region, mapped events</h3>" + tab(t("v51_by_region_mapped")))
    h.append("<h3>By distance to the nearest station (v5.1)</h3>" + tab(t("v51_by_dist")))
    h.append("<h3>v5.1 minus v4.1, stored pair against converged pair (same events)</h3>" + tab(d5))
    h.append(tab(t("v51_minus_v41_by_mag")))
    h.append("<p>With both catalogs converged, the relocation changes ML by +0.00 everywhere except Axial "
             f"({d5xc:+.2f}). At Axial the stations are a few km from the events, "
             "so the depth change of the relocation moves the Hutton-Boore distance term. The +0.05 to +0.15 "
             "offshore changes of the stored pair came from the two unconverged runs stopping at different "
             "points.</p>")
    h.append("<h3>M_type transitions v5.1 (rows stored, columns converged)</h3>" + tab(t("v51_M_type")))

    # ---- 6 Mc and b
    h.append("<h2>6. Mc and b</h2><p>Same estimators, grid and boxes as the v5.1 magnitude piece "
             "(<code>workflow/06_analysis/v51_mc_b.py</code>, which imports <code>v4_mc_maps.py</code>: MAXC+0.2, "
             "MBS, GFT90/95, the Woessner &amp; Wiemer best rule, 200 bootstrap resamples, Aki-Utsu b with the "
             "Shi &amp; Bolt error). Versions: A = v4.1 M on the v4 epicentres (the paper's Mc), B = v4.1 M on the "
             "v5.1 epicentres, C = v5.1 M on the v5.1 epicentres; here all with converged M. "
             "Version A stored equals the paper's <code>data/catalog_v4/mc_maps/mc_regional.csv</code> (n 130,772 "
             "includes 3 duplicated rows of 900001-900003 from the v4.1 catalog join, as in the stored run).</p>")
    if mcb is not None:
        sel = mcb[(mcb.period == "all")]
        h.append(tab(sel, cols=["version", "region", "n_stored", "mc_best_stored", "mc_best_conv", "d_mc_best",
                                "mc_maxc_stored", "mc_maxc_conv", "b_stored", "b_conv", "d_b"]))
        per = mcb[mcb.version.eq("A_v4M_v4loc") & mcb.period.ne("all") & mcb.region.isin(["BL", "GO", "MT", "EN", "OFF"])]
        h.append("<h3>Version A by period, offshore boxes</h3>" + tab(per, cols=["region", "period", "n_stored",
                 "mc_best_stored", "mc_best_conv", "b_stored", "b_conv"]))
    h.append("<h3>Paired b test, converged v5.1 M against converged v4.1 M (same events, common Mc)</h3>"
             + tab(pb, cols=["box", "mc_source", "mc", "n_events", "b_v41", "b_v51", "db", "sd_paired", "z",
                             "significant_bonferroni"]))
    h.append("<p>Offshore Mc drops by 0.2-0.6 with the magnitudes. With the paper's script (v4_mc_maps.py, "
             "<code>mc_maps_v41/mc_regional.csv</code>): Blanco 2.2 -> 1.6, Gorda 2.1 -> 1.9, Mendocino triple junction "
             "1.7 -> 1.5, Endeavour-Explorer 1.8 -> 1.6, offshore pooled 2.2 -> 1.7. The Gorda value is borderline: "
             "GFT95 is reached at 1.9 there, but not in version A of v51_mc_b.py (the same events plus 3 duplicated "
             "rows of 900001-900003), which falls back to GFT90 = 1.7 with b 1.05 instead of 1.22. "
             "Onshore Mc does not change. Two period statements of the paper change. The Blanco Mc of 2011-2012 "
             "(2.6 stored) was inflated by the same deployment levels: converged it is 1.5, against 1.6 in 2013-2015, "
             "so the 'offshore Mc improved by 0.2-0.4 after the first two years' result does not hold (EN +0.1, BL "
             "-0.1, GO 0.0). The rate of M &gt;= 2.5 Blanco events in 2010 and 2016, relative to 2013-2015, rises from 5% "
             "and 8% to 28% and 29%: the events of the ocean-bottom years came down by 0.6-0.7, those of 2010 and "
             "2016 (mapped, land stations only) by about 0.3. The Mendocino 2016 ratio stays at 10-11%. "
             "b changes by -0.07 to +0.12 in the offshore boxes. In the onshore box "
             "(everything outside the named boxes) b goes from 0.92 to 1.01 although its median ML change is +0.01: "
             "11% of its events drop by more than 0.1, half of them in northern California between 40 and 42 N "
             "(east of the triple-junction box, recorded by the Gorda and Mendocino ocean-bottom stations) and 12% "
             "of them 'mapped' events. Between converged v5.1 and converged v4.1 M (paired test, same events) b "
             "agrees within 0.03 in every box except Axial (+0.10 at Mc 0.1, the Axial relocation).</p>")
    for f, cap in ((f"{mc.OUT}/mc_maps_v41/fig1_mc_three_methods.png", "Paper Fig. 2a remade: Mc by three methods, "
                    "converged v4.1 M (v4 epicentres)."),
                   (f"{mc.OUT}/mc_maps_v41/fig3_mc_periods.png", "Paper Fig. 2b remade: Mc best by period, converged "
                    "v4.1 M."),
                   (os.path.normpath(os.path.join(mc.HERE, "../../paper_sciadv/figures/fig2a_mc_three_methods.png")),
                    "Paper Fig. 2a as it is now (stored M), for comparison."),
                   (f"{mc.OUT}/mc_b/fig2_b_and_change_v51.png", "v5.1 converged: b above Mc and the change of Mc and "
                    "b between converged v5.1 and converged v4.1 M on the v5.1 epicentres.")):
        h.append(img(f, cap))

    # ---- 7 paper impact
    h.append("<h2>7. Paper impact</h2><p>Every figure, table and number of the paper (Overleaf "
             "<code>sciadv/main.tex</code> and <code>supplementary.tex</code> at 8def2ea4b5, the repository copies "
             "on main, and the figure scripts) that uses Route A M, with the stored and the converged value or "
             "the figure to remake. The manuscript is not edited (manuscript changes are bullets only, by the PI). "
             "Clustering values: <code>v4_clusters_zbz.py</code> rerun on the converged v4.1 M (main run and 3 "
             "shuffles, no sensitivity variants) in <code>/wd1/mdenolle_data/magnitude_converged/clusters_zbz_v41</code>.</p>")
    h.append(tab(pi))
    extra = os.path.join(mc.SMALL, "paper_impact_text.html")
    if os.path.exists(extra):
        h.append(open(extra).read())

    # ---- 8 files, open problems
    h.append("<h2>8. Files, reproduction, open problems</h2>")
    h.append("<p>Code (branch analysis/mag-converged): <code>workflow/05_magnitude/phase19_solver.py</code> (new), "
             "<code>phase19_preferred_magnitude.py</code> (solver, --solver/--outdir/--tag, provenance), "
             "<code>v51_magnitude_table.py</code> (--pm/--out/--tag/--solver/--v4-mags), "
             "<code>mag_converged.py</code> (patch, near, threads, gauge, compare, reconcile, mcb), "
             "<code>mag_converged_report.py</code> and <code>mag_converged_text.py</code> (this page), "
             "<code>scale/run_magnitude_converged.sh</code> (driver). Resume notes: "
             "<code>/wd1/mdenolle_data/magnitude_converged/RESUME.md</code>.</p>")
    h.append("<h3>Open problems</h3><ol>"
             "<li>Axial (OO 2014): converged ML is still +0.39 above ComCat ML on its 4 dominated anchors and +0.9 on "
             "the 2 anchors of the Axial region. The OO level rests on 66-71 events with land picks (stiffness about "
             "200). Its level should be checked against the OOI/PNSN Axial catalogs before the Axial magnitudes are "
             "used for b or Mc.</li>"
             "<li>JdF ridge/plate and the 7D Y1/Y3 J stations: v4.1 near events move by -0.29 and v5.1 by -0.51, "
             "because the two stored runs stopped at different points; both converged runs agree (v5.1 - v4.1 = "
             "+0.005). The JdF plate has 4 ComCat anchors in total, so its level rests on the cross-links "
             "(7D Y1 stiffness about 400, s.e. 0.03) with no external check.</li>"
             "<li>Model uncertainty of the thinly tied levels: the 100 km pick-selection test moves Axial by -0.15, "
             "Explorer by +0.13 and Endeavour by -0.11 (verifier). These regions carry about 0.1 of uncertainty "
             "beyond the formal errors.</li>"
             "<li>'Mapped' events (3,522 in v4.1) follow the Theil-Sen map fitted to the near events; the map changed "
             "(slope 1.15 -> 1.01), so mapped ML above about 3.5 dropped more than the region shift. Their tensor "
             "comparison (ML - Mw about -0.2 at Mw 3-4.5) is worse than before.</li>"
             "<li>The near-event Mw_cal offset rose from +0.04 to +0.19 (v4.1) because large offshore ML dropped while "
             "the tensors did not change; Mw_cal events: 35 -> 27. The ML - Mw shortfall above Mw 4.5 is larger "
             "(-0.79 -> -1.15 for all, -0.36 -> -0.69 for near events), which strengthens the case for the Mw switch.</li>"
             "<li>The converged v4.1 file is not placed next to the stored one in the main tree (other worktrees are "
             "not touched); the clustering rerun has no sensitivity variants; the paper figures 1, 3, 5, 6 and the "
             "atlas are not remade.</li></ol>")
    return "\n".join(h)

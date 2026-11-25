"""Text of the v5.1 structure-seismicity report (numbers are read from the result tables)."""
from __future__ import annotations

import html

import numpy as np
import pandas as pd


def f(x, nd=2, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def iv(lo, hi, nd=2, sign=True):
    return f"[{f(lo, nd, sign)}, {f(hi, nd, sign)}]"


def build_html(c: dict) -> str:
    from v51_struct_report import CSS, fig_html, table_html
    ev, cov, czd, t1, t1e, t2, tb, t3, t3s, t4p, seg, segc, slabp, areas, syn, syns, mc, v4, ts, figs = (
        c[k] for k in ("ev", "cov", "czd", "t1", "t1e", "t2", "tb", "t3", "t3s", "t4p", "seg", "segc", "slabp",
                       "areas", "syn", "syns", "mc", "v4", "ts", "figs"))
    model = c["model"]
    g = cov.set_index("group")
    n_rows = int(g.loc["all rows (incl. no hypocentre)", "n"])
    n_loc = int(g.loc["located (loc_source nll3d/comcat)", "n"])
    n_eq = int(g.loc["located earthquakes", "n"])
    n_dg = int(g.loc["located earthquakes, depth_good", "n"])
    pP, pPS = g.loc["located earthquakes", "pct_resP"], g.loc["located earthquakes", "pct_resPS"]
    nP, nPS = int(g.loc["located earthquakes", "n_resP"]), int(g.loc["located earthquakes", "n_resPS"])
    nP8, nPS8 = int(g.loc["located earthquakes", "n_resP8"]), int(g.loc["located earthquakes", "n_resPS8"])
    n_noeq = n_loc - n_eq
    n_none = n_rows - n_loc
    ncellP = czd.ncell_P.sum()
    ncellPS = czd.ncell_PS.sum()

    def t2row(zone, var, events="all"):
        r = t2[(t2.zone == zone) & (t2["var"] == var) & (t2.events == events)]
        return r.iloc[0] if len(r) else None

    def synrange(zone, var):
        s = syn[(syn.zone == zone) & (syn["var"] == var)]
        return (s.event_mean_res.min(), s.event_mean_res.max(), s.rho_cov.min(), s.rho_cov.max()) if len(s) else (
            np.nan,) * 4

    def mcv(stat):
        r = mc[mc.statistic == stat]
        return r.iloc[0] if len(r) else None

    def t1row(var, setname="resPS", contrast="swarm - mainshock-aftershock"):
        r = t1[(t1.set == setname) & (t1["var"] == var) & (t1.contrast == contrast)]
        return r.iloc[0] if len(r) else None

    fam_ps = c["fam_ps"]
    n_sw_blocks = int(fam_ps[fam_ps.type == "swarm"].block.nunique())
    b_dVp, b_Vp, b_dVs, b_VpVs, b_dVpVs = (t2row("BL", v) for v in ("dVp", "Vp", "dVs", "VpVs", "dVpVs"))
    sm_dVp, sm_dVs, sm_VpVs = (t2row("BL", v, "all, field smoothed 20 km") for v in ("dVp", "dVs", "VpVs"))
    sP, sS, sR = synrange("BL", "dVp"), synrange("BL", "dVs"), synrange("BL", "dVpVs")
    r1 = t1row("VpVs_res")
    r1v = t1row("VpVs")
    r1dg = t1row("VpVs_res", "resPS_dg")
    r1st = t1row("VpVs_res", "resPS_strict")
    r1p = t1row("dVp_res")
    r1s = t1row("dVs_res")
    mcP, mcS, mcR = mcv("BL_dVp_res_mean"), mcv("BL_dVs_res_mean"), mcv("BL_VpVs_res_mean")
    mc1 = mcv("BL_swarm_minus_MA_VpVs_res")
    p = []
    A = p.append
    A("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
      "content='width=device-width,initial-scale=1'><title>Structure and seismicity v5.1</title>"
      f"<style>{CSS}</style></head><body>")
    A("<h1>Velocity structure and seismicity, catalog v5.1</h1>")
    A("<p class='cap'>Model v0.3.1, resolved nodes only. Catalog v5.1 with the v5.1 magnitudes, the v5.1 clustering "
      "and fault surfaces v1. 2026-11-25. Private working report; not peer reviewed. Code: "
      "<code>workflow/06_analysis/v51_struct*.py</code> on branch analysis/v51-struct.</p>")

    # ---------------- summary
    A("<div class='box'><h3 style='margin-top:0'>Summary</h3><ul>")
    A(f"<li><b>Testable sample.</b> {pP:.1f}% of the {n_eq:,} located earthquakes ({nP:,}) are in cells where Vp passes "
      f"the checkerboard tests, and {pPS:.1f}% ({nPS:,}) where Vp and Vs both pass, so Vp/Vs can be read. All of "
      "the Vp/Vs cells are at the Blanco transform, at the 15 and 18 km nodes (a few at 29 km). Vp alone is "
      "resolved in three places: the Blanco at 15, 18 and 29 km, the Gorda slab under Cape Mendocino at 29 km, and "
      "southwest Washington (Mount St. Helens, 1-9 km; Puget and the Cascades foothills, 15-18 km). The resolved "
      f"volume holds about {ncellP:.0f} independent checkerboard cells for Vp and {ncellPS:.0f} for Vp/Vs, summed over "
      "the resolved depth nodes. That cell "
      "count, not the number of events, limits every test below.</li>")
    A(f"<li><b>Q1, swarms against mainshock-aftershock sequences (Blanco only).</b> {int(r1.n_swarm)} swarm and "
      f"{int(r1.n_other)} mainshock-aftershock families have members in Vp/Vs-resolved cells. Their median Vp/Vs "
      f"values are {f(r1v.med_swarm, 3)} and {f(r1v.med_other, 3)}. The coverage-matched difference is "
      f"{f(r1['diff'], 3, True)} (95% block bootstrap {iv(r1.ci_lo, r1.ci_hi, 3)}, permutation p {f(r1.p_perm, 2)}). "
      f"The families fall in {int(r1.n_blocks)} blocks of 40 km, the swarms in {n_sw_blocks}, so a difference "
      f"smaller than about {f(max(abs(r1.null_lo), abs(r1.null_hi)), 2)} in Vp/Vs cannot be detected. Published "
      "differences between settings are 0.05-0.15. The data neither support nor exclude a fluid control of swarms.</li>")
    A(f"<li><b>Q2, where the events are.</b> At the Blanco, the nodes that hold events are slower than coverage and "
      f"depth predict. The event-weighted anomaly is {f(b_dVp.event_mean_res, 2, True)}% in Vp change "
      f"({iv(b_dVp.res_ci_lo, b_dVp.res_ci_hi)}) and {f(b_dVs.event_mean_res, 2, True)}% in Vs change "
      f"({iv(b_dVs.res_ci_lo, b_dVs.res_ci_hi)}). Randomized catalogs reach {iv(b_dVp.res_null_lo, b_dVp.res_null_hi)}% "
      f"and {iv(b_dVs.res_null_lo, b_dVs.res_null_hi)}%, and the checkerboard inversion errors "
      f"{iv(sP[0], sP[1])}% and {iv(sS[0], sS[1])}%. <b>But the signal lives at the scale of single nodes (about "
      "10 km), below the 40 km resolved cell.</b> With the field smoothed to the resolved scale (Gaussian, sigma "
      f"20 km), the anomaly becomes {f(sm_dVp.event_mean_res, 2, True)}% ({iv(sm_dVp.res_ci_lo, sm_dVp.res_ci_hi)}) for "
      f"Vp and {f(sm_dVs.event_mean_res, 2, True)}% ({iv(sm_dVs.res_ci_lo, sm_dVs.res_ci_hi)}) for Vs. Both intervals "
      "include zero. Under the rule that only resolved structure is interpreted, the Blanco result is not usable as "
      f"structure. Vp/Vs at the events is not clearly higher ({f(b_VpVs.event_mean_res, 3, True)}, "
      f"{iv(b_VpVs.res_ci_lo, b_VpVs.res_ci_hi, 3)}; smoothed {f(sm_VpVs.event_mean_res, 3, True)}, "
      f"{iv(sm_VpVs.res_ci_lo, sm_VpVs.res_ci_hi, 3)}). Nothing is found at the Mendocino triple junction or in "
      "Washington. b-value does not differ between slow and fast nodes in any zone.</li>")
    A("<li><b>Q3, depth extent.</b> A Vp level cannot be tested: only "
      f"{int(t3[(t3.test == 'Vp level at d90') & (t3.zone == 'BL')].n_columns.iloc[0])} Blanco columns, and 0-3 "
      "elsewhere, have resolved Vp on both nodes around their 90th-percentile depth. Against the slab (Slab2, no "
      "tomography needed): in the Mendocino box the deepest well-located events lie a median "
      f"{f(t3[(t3.zone == 'MT')]['median'].iloc[0], 1)} km below the slab top and do not follow its shape more closely "
      "than random pairing. In the Puget and onshore boxes they stop 27-31 km above the slab.</li>")
    tv = t4p[(t4p["var"] == "Vp") & (t4p.subset == "all")].iloc[0]
    A(f"<li><b>Q4, fault surfaces v1.</b> {int(tv.n_surfaces)} surfaces have at least 10 mesh vertices in "
      "Vp-resolved cells. Their normals are no more aligned with the horizontal Vp gradient than randomly rotated "
      f"surfaces (mean alignment {f(tv.align_mean, 2)} against {f(tv.align_null_mean, 2)}, p "
      f"{f(tv.align_p_greater, 2)}). They also do not sit at stronger gradients than other resolved nodes of the "
      f"same coverage (median ratio {f(tv.grad_ratio_med, 2)}, {iv(tv.grad_ratio_ci_lo, tv.grad_ratio_ci_hi, 2, False)}). "
      "At 40-60 km resolution a fault-scale contrast cannot be seen.</li>")
    sc = segc[(segc.x == "VpVs_events") & (segc.y == "swarm_share")].iloc[0]
    A("<li><b>Q5, Blanco segments.</b> The two western segments (West Blanco and Surveyor Depressions) have the "
      f"highest Vp/Vs at their events ({f(seg.VpVs_events.iloc[0], 2)} and {f(seg.VpVs_events.iloc[1], 2)}) and the "
      f"highest swarm shares ({f(seg.swarm_share.iloc[0], 2)}, {f(seg.swarm_share.iloc[1], 2)}). The Blanco Ridge "
      f"has the lowest Vp/Vs ({f(seg.VpVs_events.iloc[3], 2)}) and a swarm share of {f(seg.swarm_share.iloc[3], 2)}. "
      f"Over five segments the Spearman correlation is {f(sc.spearman, 2)} with exact p {f(sc.p_exact, 2)}: a "
      "pattern to test with relocated swarms and a finer model, not a result.</li>")
    A(f"<li><b>Multiple tests.</b> {len(ts)} tests are listed in tests_summary.csv. {int((ts.q_BH < 0.05).sum())} "
      "have a Benjamini-Hochberg q below 0.05. Only those whose block-bootstrap interval also excludes the no-effect "
      f"value count as passed. There are {int(ts.passed.sum())}, all node-scale Blanco Q2 statistics: "
      + "; ".join(sorted({html.escape(r.test) for _, r in ts[ts.passed].iterrows()})) + ". None passes at the "
      "resolved scale.</li>")
    A("</ul></div>")

    # ---------------- data
    A("<h2>1. Data and selection</h2><ul>")
    A(f"<li>Catalog v5.1: {n_rows:,} rows. Rows without a hypocentre (loc_source none) are left out of every "
      f"spatial analysis: {n_none}. Located: {n_loc:,}. Non-earthquakes (ComCat explosions, quarry blasts, other "
      f"events; Texada quarry_blast_likely; is_earthquake False in the v5.1 magnitudes) are left out: {n_noeq:,}. "
      f"That leaves {n_eq:,} located earthquakes.</li>")
    A(f"<li>Well-constrained depth (depth_good, the clusters piece's definition): tier A-C, depth_quality ok, not "
      f"floor_limited, far_only or deep_pass: {n_dg:,} events. Depth-sensitive variants use only these. Blanco "
      "depths outside the 2012-13 OBS deployment are depth_quality unconstrained, and the Blanco v5.1 depths are "
      "a median 2.9 km deeper than the Lange et al. (2026) OBS catalog (faults piece). Placement in the 15 or 18 km "
      "layer therefore depends on that bias.</li>")
    A("<li>Magnitudes: v5.1 M (mag piece). Mc per box from the clusters piece (v4 rule on v5.1 magnitudes). "
      "Clustering: the v5.1 nearest-neighbour families (types swarm, mainshock-aftershock, multiplet) and their "
      "classes. Fault surfaces v1: summary, assignment, and the 1 km mesh (constrained vertices only).</li>")
    A(f"<li>Model v0.3.1 ({model.shape[2]} x {model.shape[1]} x {model.shape[0]} nodes, 0.125 x 0.1 deg, depth -3 to "
      "125 km). Resolved Vp: resolution_p &gt; 0. Resolved Vs: resolution_s &gt; 0 and vs_reverted = 0. Vp/Vs needs both. "
      "These are the D-B018 checkerboard criteria (local correlation &gt;= 0.7 and amplitude &gt;= 0.5 at 40, 60 or 100 km "
      "cells, with pick noise, both pattern phases), merged under D-B049, and D-B051 for Vs. "
      f"Resolved nodes: {int(model.mP.sum())} for Vp, {int(model.mS.sum())} for Vs, {int(model.mPS.sum())} for both. "
      f"Resolved nodes in water or air by the v0.3.2 medium flag: {model.n_res_not_rock}. The D-B073 shallow-crust "
      "rule is not in v0.3.1 and is not used. Only the medium flag is read from v0.3.2.</li>")
    A("<li>An event belongs to the node nearest to its hypocentre (voxel boundaries halfway between nodes). "
      f"A stricter variant requires all eight corners of the enclosing grid cell to be resolved: {nP8:,} events for "
      f"Vp and {nPS8:,} for Vp/Vs.</li></ul>")

    A("<h2>2. Method: controls for circularity</h2>")
    A("<p>The events give the rays. Ray density (DWS) and resolution are therefore high where events are, and a "
      "tomographic error that follows the events would look like a structure-seismicity relation. Each test uses "
      "four controls:</p><ol>")
    A("<li><b>Coverage-matched anomaly.</b> Each resolved node gets a stratum: zone x depth node x DWS tercile (P DWS "
      "for Vp, S DWS for Vs and Vp/Vs). V_res is the node value minus the mean over the resolved nodes of its stratum. "
      "A relation in V_res cannot come from depth or from the amount of ray coverage alone. DWS quintiles give the "
      "same results (t2_density.csv, columns *_q5).</li>")
    A("<li><b>Randomized catalogs.</b> The event set of each occupied node is moved to a random resolved node of the "
      "same stratum (2,000 times). This keeps the event clustering and the coverage. For Q1, family types are "
      "permuted among families of the same stratum.</li>")
    A("<li><b>Block bootstrap.</b> 95% intervals come from resampling 40 km blocks, the finest resolved checkerboard "
      "cell, with all events and nodes of a block together (2,000 resamples). Node-level permutations treat "
      "neighbouring nodes as independent when they are not, so the bootstrap interval is the one to use for the "
      "effect size.</li>")
    A("<li><b>Synthetic inversion errors.</b> The checkerboard inversions of each tile (40, 60 and 100 km cells; "
      "pick noise; pattern phases 0 and 1; and inversions started from perturbed hypocentres, noise_ploc) were run "
      "on the real event and station geometry. The error, recovered minus true, follows the events only through the "
      "inversion. The same statistics on these error fields give the size of an artefact.</li></ol>")
    A("<p>Each event's catalog errors (horizontal sqrt((h_min^2 + h_max^2)/2)/1.515, vertical sig_z) are resampled "
      "50 times to test the sensitivity to location. This only reassigns events to nodes; no hypocentre in any "
      "product is moved. Multiple tests: Benjamini-Hochberg q values over all tests (tests_summary.csv).</p>")

    # ---------------- coverage
    A("<h2>3. Fraction of events in resolved cells</h2>")
    A(fig_html(figs["coverage"], "Left: resolved nodes at four depth nodes, with the events of each layer. Right: "
               "located earthquakes in resolved cells by catalog region."))
    cz = czd[["node_zone", "node_depth", "nodes_P", "nodes_PS", "area_P_km2", "ncell_P", "ncell_PS", "n_resP",
              "n_resPS", "n_depth_good", "n_aboveMc"]].copy()
    A("<p>By zone and depth node. ncell is the resolved area divided by the square of the finest resolved "
      "checkerboard cell: the number of independent velocity samples.</p>")
    A(table_html(cz, "{:.4g}"))
    cv = cov[~cov.group.str.startswith("region") & ~cov.group.str.startswith("tier")][
        ["group", "n", "n_resP", "pct_resP", "n_resPS", "pct_resPS", "n_resP8", "n_resPS8"]]
    A(table_html(cv, "{:.3g}"))
    A("<p>Swarm events are over-represented in Vp/Vs-resolved cells "
      f"({f(g.loc['class: swarm', 'pct_resPS'], 1)}% against {f(pPS, 1)}% overall). The Blanco swarms are what "
      "resolves the 15-18 km layer. Events on no fault surface are under-represented "
      f"({f(g.loc['fault assignment: none', 'pct_resPS'], 1)}%). Nothing is resolved at Axial, Endeavour, Explorer "
      "or the Juan de Fuca ridge and plate, in Gorda above 29 km, or in the offshore Oregon and Washington "
      "forearc.</p>")

    # ---------------- Blanco maps
    A("<h2>4. The Blanco at 15 and 18 km</h2>")
    A(fig_html(figs["blanco"], "Resolved nodes only: Vp change, Vs change and Vp/Vs at the 15 and 18 km nodes, with "
               "the events of each layer and the segment boundaries used in Q5."))
    nb = c["nodes_bl"]
    A(f"<p>Vp at these nodes is {f(nb['vp5'], 1)}-{f(nb['vp95'], 1)} km/s (5th-95th percentile; median "
      f"{f(nb['vp50'], 2)}), i.e. uppermost mantle velocities, and the Vp change from the start spans "
      f"{f(nb['dvp5'], 1, True)} to {f(nb['dvp95'], 1, True)}%. Vp/Vs at the {nb['nps']} Vp/Vs-resolved nodes spans "
      f"{f(nb['r5'], 2)}-{f(nb['r95'], 2)} (5th-95th percentile), with a median of {f(nb['r50'], 2)}. The starting model there is "
      "Kuna et al. (2019) 1-D, with Vp/Vs 1.73 in the mantle. Lange et al. (2026) report Vp/Vs above 2 in most of "
      "the fault zone, but higher up, in the crust; our model does not resolve the crust at the Blanco.</p>")

    # ---------------- Q1
    A("<h2>5. Q1: swarms and mainshock-aftershock sequences in different Vp/Vs settings?</h2>")
    A(fig_html(figs["t1"], "Coverage-matched Vp/Vs, Vp change and Vs change of each Blanco family (median over the "
               "members in Vp/Vs-resolved cells), by family type. Bars: medians."))
    t1s = t1[t1["var"].isin(["VpVs", "VpVs_res", "dVpVs_res", "dVp_res", "dVs_res"])][
        ["set", "zone", "var", "contrast", "n_swarm", "n_other", "n_blocks", "med_swarm", "med_other", "diff", "ci_lo",
         "ci_hi", "null_lo", "null_hi", "p_perm"]]
    A(table_html(t1s, "{:.3g}"))
    A("<p>Set resPS: families with at least 3 members in Vp/Vs-resolved cells. resPS_dg: the same with "
      "well-constrained depths only. resPS_strict: at least half of the members in resolved cells (this drops the "
      "large Blanco swarms, whose members spread over the unresolved 12 and 21 km layers). resP: Vp only, Blanco and "
      f"Washington. Main result: Vp/Vs swarm minus mainshock-aftershock {f(r1['diff'], 3, True)} "
      f"{iv(r1.ci_lo, r1.ci_hi, 3)}; depth_good only {f(r1dg['diff'], 3, True)} {iv(r1dg.ci_lo, r1dg.ci_hi, 3)}; "
      f"strict {f(r1st['diff'], 3, True)} {iv(r1st.ci_lo, r1st.ci_hi, 3)}. With location errors resampled, the "
      f"difference stays between {f(mc1.mc_p05, 3, True)} and {f(mc1.mc_p95, 3, True)} (5-95%). Vp and Vs changes: "
      f"{f(r1p['diff'], 2, True)}% {iv(r1p.ci_lo, r1p.ci_hi)} and {f(r1s['diff'], 2, True)}% "
      f"{iv(r1s.ci_lo, r1s.ci_hi)}.</p>")
    A("<p>Event level (events above Mc in Vp/Vs-resolved cells): the swarm share of clustered events in nodes above and "
      "below the coverage-matched mean.</p>")
    A(table_html(t1e, "{:.3g}"))
    A("<p>Caveat from the clusters piece's verifier: at the Blanco, a time-shuffled catalog reproduces 39% of the "
      "nearest-neighbour links, so part of the swarm and mainshock-aftershock labelling is not robust.</p>")

    # ---------------- Q2
    A("<h2>6. Q2: seismicity density and b-value against Vp/Vs and Vp change at fixed depth and coverage</h2>")
    A(fig_html(figs["t2"], "Mean coverage-matched anomaly at the events (each event counts once), by zone and "
               "variable. Blue: catalog with its block-bootstrap 95% interval. Grey band: 95% range of the "
               "randomized catalogs. Orange ticks: the same statistic on each checkerboard inversion error field "
               "(9 per zone)."))
    t2s = t2[t2.events.isin(["all", "all, field smoothed 20 km"])][["events"] + ["zone", "var", "n_nodes", "n_nodes_with_events", "n_events", "n_blocks",
                                  "rho_depth_only", "rho_cov", "rho_ci_lo", "rho_ci_hi", "rho_p", "event_mean_res",
                                  "res_ci_lo", "res_ci_hi", "res_null_lo", "res_null_hi", "res_p", "event_mean_res_q5"]]
    A(table_html(t2s, "{:.3g}"))
    A("<p>rho_depth_only: Spearman correlation between the event count of a node and the raw value, within zone x "
      "depth (coverage not matched). rho_cov: the same on the coverage-matched anomaly, averaged over strata. "
      "event_mean_res: the mean anomaly at the events. Units: % for changes, km/s for Vp and Vs, ratio for "
      "Vp/Vs. The table for well-constrained depths, events above Mc and background events only is in "
      "t2_density.csv. Their Blanco values have the same sign and size, e.g. depth_good Vp change "
      f"{f(t2row('BL', 'dVp', 'depth_good').event_mean_res, 2, True)}% "
      f"{iv(t2row('BL', 'dVp', 'depth_good').res_ci_lo, t2row('BL', 'dVp', 'depth_good').res_ci_hi)}.</p>")
    A("<p><b>Reading.</b> At the node scale, Blanco nodes with more events are slower in Vp and Vs at the same depth "
      f"and coverage: {f(b_dVp.event_mean_res, 1, True)}% (Vp) and {f(b_dVs.event_mean_res, 1, True)}% (Vs). That is "
      f"about {f(abs(b_dVp.event_mean_res) / max(abs(sP[0]), abs(sP[1])), 0)} and "
      f"{f(abs(b_dVs.event_mean_res) / max(abs(sS[0]), abs(sS[1])), 0)} times the largest checkerboard artefact, and "
      "the inversions started from perturbed hypocentres do not give larger artefacts. It survives the location Monte "
      f"Carlo at a smaller size ({f(mcP.mc_median, 2, True)}% and {f(mcS.mc_median, 2, True)}%) and DWS quintiles. "
      "It does not survive smoothing to the resolved scale (rows 'all, field smoothed 20 km' in t2_density.csv; "
      "open markers in the figure). The node-to-node spread of the Vp change at the Blanco is "
      f"{f(b_dVp.node_sd, 1)}%, but only {f(sm_dVp.node_sd, 1)}% after smoothing. Most of the variance, and the whole "
      "association with the events, sits at 10-20 km. The checkerboard tests (40, 60 and 100 km cells) do not certify "
      "that scale. Ten-kilometre structure there may be real (a damaged or hydrated transform mantle) or an inversion "
      "artefact that the checkerboard errors do not capture: those use true hypocentres and random noise, not the "
      "v5.1 Blanco depth bias. For the smoothed Vp/Vs and Vs, the randomized catalogs give p &lt; 0.001 while the "
      "block bootstrap includes zero. A smoothed field is strongly autocorrelated, so the node permutation is too "
      "narrow there, and the bootstrap is the interval to read.</p>")
    A(fig_html(figs["t2b"], "b-value difference between events in nodes above and below the coverage-matched mean, "
               "with block-bootstrap 95% intervals and the randomized-node null (grey)."))
    A(table_html(tb[["mask", "events", "zone", "var", "n_low", "n_high", "b_low", "b_high", "db_high_minus_low",
                     "ci_lo", "ci_hi", "null_lo", "null_hi", "p_perm", "n_blocks", "mc_boxes"]], "{:.3g}"))
    A("<p>b uses the Aki-Utsu estimator on M minus the Mc of the event's box (pooled excess magnitudes). Offshore "
      "ML is 0.3-0.5 above ComCat ML at the Blanco, Gorda and Mendocino (mag piece). Within one zone this offset is "
      "common to both groups, but D-B041 asks for an ML-Mw check before b is compared between settings, and none is "
      "attempted here.</p>")

    # ---------------- Q3
    A("<h2>7. Q3: does the depth extent of seismicity follow a Vp level or the slab?</h2>")
    A(fig_html(figs["t3"], "Left: 95th-percentile depth of well-constrained events in each model column with at "
               "least 30 of them, against the Slab2 depth. Right: Vp at the 90th-percentile depth, only where both "
               "bracketing depth nodes are resolved."))
    A(table_html(t3.drop(columns=[c_ for c_ in ("note",) if c_ in t3]), "{:.3g}"))
    A(table_html(t3s, "{:.4g}"))
    A("<p>The Vp-level question is untestable on the resolved model. The checkerboards resolve isolated depth "
      "nodes (1-9, 15, 18 and 29 km, with gaps at 12, 21 and 25 km that the tomography notes attribute to the "
      "vertical checker pattern). Almost no column has resolved nodes on both sides of its seismicity bottom. The 6 "
      "Blanco columns do not show a common Vp at D90 (spread larger than for depths placed in other columns). For the "
      "slab: in the Mendocino box, D95 is below the slab top in almost every column, and D95 minus slab depth spreads "
      "as much as with randomly paired columns. In the Puget and onshore boxes, D95 is shallower where the slab is "
      "deeper (Spearman " + f(t3[t3.zone == 'PU'].rho_d95_slab.iloc[0], 2) + " and " +
      f(t3[t3.zone == 'ON'].rho_d95_slab.iloc[0], 2) + "). Upper-plate seismicity there ends at a crustal depth, not "
      "at the slab.</p>")

    # ---------------- Q4
    A("<h2>8. Q4: do fault surfaces v1 align with velocity contrasts?</h2>")
    A(fig_html(figs["t4"], "Per surface: median alignment between the horizontal surface normal and the horizontal "
               "gradient of Vp, Vp change, Vp/Vs and Vp/Vs change at its constrained mesh vertices in resolved "
               "cells. 1 = the gradient is perpendicular to the surface (the surface follows a contrast). Black bar: "
               "mean over surfaces; grey band: 95% range of that mean when every surface is rotated to a random "
               "strike."))
    A(table_html(t4p, "{:.3g}"))
    A("<p>Gradients are central differences between resolved nodes only (about 10-11 km apart). The model resolves "
      "40-60 km cells, so it can show only whether a surface runs along a long-wavelength velocity boundary. "
      "Neither the absolute Vp (which includes the starting model's lateral structure) nor the Vp change shows "
      "alignment. Orientation is used for all surfaces, but the faults verifier found that 93 of the 142 surfaces "
      "change strike between detection reruns (median range 103 deg). The 'stable' rows, 10 surfaces for Vp, are "
      "the only ones whose orientation can be trusted. Their alignment is "
      f"{f(t4p[(t4p['var'] == 'Vp') & (t4p.subset == 'stable')].align_mean.iloc[0], 2)} against "
      f"{f(t4p[(t4p['var'] == 'Vp') & (t4p.subset == 'stable')].align_null_mean.iloc[0], 2)} "
      f"(p {f(t4p[(t4p['var'] == 'Vp') & (t4p.subset == 'stable')].align_p_greater.iloc[0], 2)}; "
      f"q {f(ts[ts.test == 'alignment |n.grad Vp| (stable surfaces)'].q_BH.iloc[0], 2)} after the correction for "
      "multiple tests), on 10 surfaces in 7 blocks.</p>")

    # ---------------- Q5
    A("<h2>9. Q5: region-specific questions</h2><h3>Blanco segments</h3>")
    A(fig_html(figs["t5"], "Blanco segments: median Vp/Vs at events in Vp/Vs-resolved cells against the swarm share "
               "of clustered events above Mc (left) and the b-value (right)."))
    A(table_html(seg[["segment", "name", "n_events", "n_depth_good", "n_resPS", "pct_resPS", "VpVs_events",
                      "VpVs_ci_lo", "VpVs_ci_hi", "dVp_events", "dVs_events", "depth_med", "depth_good_med",
                      "n_aboveMc", "swarm_share", "swarm_ci_lo", "swarm_ci_hi", "background_share", "n_fam5",
                      "swarm_frac_fam5", "b_value"]], "{:.3g}"))
    A(table_html(segc, "{:.3g}"))
    A("<p>Segments are longitude boxes along the transform, consistent with the literature names carried by the v1 "
      "Blanco surfaces (proposed decision). Only 3-24% of each segment's events are in Vp/Vs-resolved cells (the "
      "Cascadia Depression 3%). With five segments, the smallest possible two-sided exact p is 0.017, so a "
      "rank correlation cannot reach significance unless it is perfect. The pattern (higher Vp/Vs and more swarms in "
      "the west) agrees with the science-framing hypothesis, but the uncertainty intervals of neighbouring segments "
      "overlap.</p>")
    A("<h3>Forearc against slab; Mendocino triple junction; Mount St. Helens</h3>")
    A(table_html(slabp, "{:.3g}"))
    A("<p>Within each resolved zone, all events are on one side of the slab top: the 29 km Mendocino nodes hold "
      "slab events (a median 13 km below the Slab2 top), and the Washington nodes hold upper-plate events (the slab "
      "is about 55 km deep there). A forearc-slab comparison at matched depth and coverage is therefore not possible. "
      "The resolved Mendocino nodes are all north of 40.45 N, so the contrast across the Mendocino fault (Guo 2025) "
      "cannot be tested.</p>")
    A(table_html(areas, "{:.3g}"))
    A("<p>Mount St. Helens: Vp is resolved at 1-9 km, but over 1-2 blocks per depth. Neither the 1 km node "
      "(coverage-matched Vp change at the events "
      f"{f(areas.iloc[0].dVp_res_mean, 1, True)}%, randomized range {iv(areas.iloc[0].null_lo, areas.iloc[0].null_hi, 1)}%) "
      "nor the other depths differ from randomized catalogs. The test cannot separate the seismic column of the "
      "volcano from its surroundings at 40 km resolution.</p>")

    # ---------------- controls
    A("<h2>10. Circularity controls in detail</h2>")
    A("<h3>Checkerboard inversion errors</h3>")
    A(table_html(syns, "{:.3g}"))
    A("<p>err_rms: RMS error at resolved nodes (the checker amplitude is 8%). event_mean_res: the event-weighted "
      "coverage-matched mean of the error (what Q2 measures). swarm_minus_ma: the Q1 statistic on the error field. "
      "All 45 rows are in synth_artifact.csv.</p>")
    A("<h3>Location Monte Carlo (50 draws)</h3>")
    A(table_html(mc, "{:.3g}"))
    pr = mc[mc.statistic == "pct_resP"].iloc[0]
    A(f"<p>Resampling the locations changes the share of events in Vp-resolved cells from {f(pr.catalog, 1)}% to "
      f"{f(pr.mc_median, 1)}%. It shrinks the Blanco anomalies (Vp change {f(mcP.catalog, 2, True)} to "
      f"{f(mcP.mc_median, 2, True)}%, Vs change {f(mcS.catalog, 2, True)} to {f(mcS.mc_median, 2, True)}%) because "
      "events scattered by their errors land in neighbouring nodes. This is consistent with an association at the "
      "catalog positions and on scales near the location error.</p>")

    # ---------------- v4
    A("<h2>11. v4-era counterpart</h2>")
    A("<p>The earlier analysis, phase13_tomography.py (Route A era, 09-29), compared an earlier catalog with "
      "external Vs models (Delph et al. 2018; CRESCENT CVM Gen0). It used dlnVs relative to each model's depth mean "
      "and no resolution mask or coverage control. No v4 structure-seismicity analysis exists on our own tomography: "
      "v0.3.1 did not exist before v5.1. Its statistic, rerun on the v4 and the v5.1 hypocentres of the same events, "
      "shows what the relocation changed:</p>")
    A(table_html(v4, "{:.3g}"))
    cr = v4[v4.model == "CRESCENT_Gen0.nc"].set_index("group")
    de = v4[v4.model == "Delph2018.nc"].set_index("group")
    other = cr.drop(index=[i for i in cr.index if i == "Blanco" or i.startswith("v5.1")], errors="ignore")
    sp = cr[cr.index.str.startswith("v5.1")]
    A(f"<p>The relocation moves Blanco events into faster CRESCENT Vs ({f(cr.loc['Blanco', 'change'], 1, True)} "
      "points in median dlnVs) because they moved deeper (median "
      f"{f(cr.loc['Blanco', 'depth_med_v4'], 1)} to {f(cr.loc['Blanco', 'depth_med_v51'], 1)} km). In the other "
      f"regions the median changes by {f(other.change.min(), 1, True)} to {f(other.change.max(), 1, True)} points "
      f"(CRESCENT) and {f(de.change.min(), 1, True)} to {f(de.change.max(), 1, True)} points (Delph). These numbers "
      "mostly follow depth and geography. The external models are not affected by the circularity of our own "
      "tomography, but they have no resolution information at the event scale either.")
    if len(sp):
        A(f" At the {int(sp.n.iloc[0]):,} Vp/Vs-resolved v5.1 events inside CRESCENT, its dlnVs and our "
          f"coverage-matched Vs change correlate weakly (Spearman {f(sp.change.iloc[0], 2)}).")
    A(" CRESCENT Gen0 Vs is part of our starting model onshore (D-B005); at the Blanco the start is Kuna et al. "
      "(2019).</p>")

    # ---------------- untestable
    A("<h2>12. What cannot be tested</h2>")
    rg = cov[cov.group.isin([f"region: {r}" for r in ("Axial", "Endeavour", "Explorer", "JdF ridge/plate")])]
    ridge_pct = 100 * rg.n.sum() / n_eq
    un = pd.DataFrame([
        ("Vp/Vs anywhere outside the Blanco 15-18 km layer", "Vs is reverted (dws_s < 10) or unresolved at "
         f"{int(model.mP.sum() - model.mPS.sum())} of the {int(model.mP.sum())} Vp-resolved nodes"),
        (f"Axial, Endeavour, Explorer, JdF ridge and plate ({ridge_pct:.0f}% of located earthquakes)", "no resolved node (ridges_north "
         "left out of the merge, D-B050)"),
        ("Gorda ridge and plate, Mendocino above 29 km; offshore Oregon and Washington forearc", "no resolved node "
         "(s480 tiles barely recover checkerboards, D-B049)"),
        ("Crust at the Blanco (above 13.5 km)", "no resolved node at 1-12 km offshore"),
        ("Q3 depth extent against a Vp level", "0-6 columns per zone with resolved Vp bracketing D90/D95"),
        ("Forearc against slab at matched depth and coverage", "each resolved zone holds one population only"),
        ("Contrast across the Mendocino fault", "no resolved node south of 40.45 N"),
        ("Swarm diffusion or migration against Vp/Vs", "88% of swarms are unresolved in migration (clusters piece)"),
        ("Ensemble (posterior) uncertainty of the velocities (D-B037)", "the D-B030 ensemble is not run on v0.3.1; "
         "the intervals here are spatial bootstrap intervals, not model posteriors"),
        ("Fault-zone volume averages (D-B037)", "volume widths from spike half-widths not yet set; at 40-60 km "
         "resolution a fault-zone volume is a single node"),
    ], columns=["question", "why"])
    A(table_html(un))

    # ---------------- proposed decisions
    A("<h2>13. Proposed decisions</h2><ol>")
    for d in PROPOSED:
        A(f"<li>{html.escape(d)}</li>")
    A("</ol>")
    A("<h2>14. Open issues and limits</h2><ul>")
    for d in [OPEN[0].format(ncellP=ncellP, ncellPS=ncellPS)] + OPEN[1:]:
        A(f"<li>{html.escape(d)}</li>")
    A("</ul>")
    A("<h2>15. All tests</h2>")
    A(table_html(ts, "{:.3g}"))
    A("<h2>16. Files</h2><p>In data/catalog_v5/v5.1_analysis/struct/: coverage_*.csv, t1_*.csv, t2_*.csv, t3_*.csv, "
      "t4_*.csv, t5_*.csv, synth_artifact*.csv, mc_location*.csv, v4_counterpart_external_vs.csv, tests_summary.csv, "
      "nodes_resolved_v031.csv, events_in_resolved_cells.csv.gz, fig_*.png. On "
      "/wd1/mdenolle_data/catalog_v5/v5.1_analysis/struct/: events_struct_v51.parquet, nodes_v031.parquet, "
      "RESUME.md, logs.</p>")
    A("</body></html>")
    return "\n".join(p)


PROPOSED = [
    "Resolved means resolution_p > 0 for Vp, resolution_s > 0 and vs_reverted == 0 for Vs, and both for Vp/Vs, "
    "on v0.3.1 as published. The 12/21/25 km mask gaps are not bridged, and no DWS-only or tile-weight criterion "
    "is added.",
    "Events are sampled at the nearest node (voxel). The all-eight-corners rule is reported as a stricter count, not "
    "used for the tests.",
    "Circularity control: coverage-matched anomalies within zone x depth node x DWS tercile (P DWS for Vp, S DWS for "
    "Vs and Vp/Vs); randomized catalogs that move each node's event set to a random node of the same stratum; family "
    "types permuted within strata; 40 km block bootstrap (finest resolved checkerboard cell) for all intervals.",
    "Use the tiles' checkerboard inversion errors (recovered minus true, noise and noise_ploc, both pattern phases) "
    "as the artefact yardstick, interpolated from the tile lattices onto the merged grid.",
    "Event set: located earthquakes only (no loc_source none, no is_earthquake False). Depth-sensitive variants use "
    "depth_good (tier A-C, depth_quality ok, not floor_limited, far_only or deep_pass), the clusters piece's "
    "definition.",
    "Q1 family value: median over members in resolved cells, families with >= 3 such members; >= 50% (strict) and "
    "depth_good variants are reported.",
    "b-value: Aki-Utsu on M minus the Mc of the event's cluster box, events split at a coverage-matched anomaly of 0. "
    "Report only within a zone (D-B041).",
    "Blanco segments as longitude boxes (lat 42.4-45.0): WBD < -129.45 < SD < -128.95 < CD < -128.5 < BR < -127.45 "
    "< GD < -126.4, matching the v1 surfaces' literature names.",
    "Q3 depth extent: columns of one model node (0.125 x 0.1 deg) with >= 30 well-constrained events; Vp at D90/D95 "
    "only between two adjacent resolved depth nodes.",
    "Q4: constrained mesh vertices only; alignment of the horizontal normal with the horizontal gradient against "
    "random strike rotations; orientation results read only for the faults piece's 'stable' surfaces.",
    "Multiple testing: Benjamini-Hochberg over all tests in tests_summary.csv; report q with p.",
    "Test every relation also on the field smoothed to the resolved scale (Gaussian, sigma 20 km = half the finest "
    "resolved checkerboard cell); a relation that exists only at the node scale is reported as unresolved.",
    "Do not use this analysis for interpretation in the paper beyond the statement of what is untestable and the "
    "null results at the resolved scale, until the D-B077 re-inversion (v0.4.0) with its resolution tests (including "
    "a 20 km checkerboard or spike test at the Blanco) and its ensemble exists.",
]

OPEN = [
    "The resolved volume is small: about {ncellP:.0f} independent Vp cells and {ncellPS:.0f} Vp/Vs cells. The D-B049 gate keeps the "
    "s480 tiles at full weight, but their checkerboards barely recover, so Gorda, Oregon and Washington are largely "
    "unresolved. Any structure-seismicity claim outside the Blanco needs a less smoothed or re-inverted model "
    "(D-B077) with new resolution tests.",
    "The checkerboard masks are empty at 12, 21 and 25 km while neighbouring depths are resolved (tomography notes: "
    "probably the vertical checker sign changes). If the masks change, rerun every step.",
    "The Blanco low-Vp/low-Vs association at the events exists only at the node scale (10-20 km), below the 40 km "
    "resolved cell, so the resolution tests do not cover it. It could be real or could come from a systematic "
    "trade-off with the v5.1 Blanco depth bias (+2.9 km median against Lange 2026); the checkerboards use true "
    "hypocentres plus random perturbations, not a systematic depth error. A 20 km checkerboard or spike test and a "
    "synthetic test with a systematic depth offset, on the D-B077 model, would settle it.",
    "Vp/Vs at the Blanco (1.62-1.88 at resolved nodes) is close to the Kuna (2019) starting value of 1.73 in the "
    "mantle. Vs is reverted at most S-poor nodes (D-B051), so the Vp/Vs signal is weak by construction.",
    "Family labels at the Blanco are partly reproduced by time-shuffled catalogs (39% of links, clusters verifier). "
    "Q1 inherits that uncertainty.",
    "The fault surfaces used are the faults piece's committed v1 (6be76cfd). The verifier asked for relabelled Axial "
    "events and new flags (no effect here, because Axial has no resolved node) and for detection-level "
    "uncertainty: only stable surfaces should carry orientation results.",
    "No ensemble posterior (D-B030/D-B032) exists for v0.3.1, so the velocity uncertainty at a node is not in the "
    "intervals; the block bootstrap covers spatial sampling only.",
]

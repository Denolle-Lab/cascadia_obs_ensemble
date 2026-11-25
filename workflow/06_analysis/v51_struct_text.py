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
    fl, v4h, blk, tiles, bchk, t4s, t3c, prov = (c[k] for k in ("fl", "v4h", "blk", "tiles", "bchk", "t4s", "t3c",
                                                                 "prov"))
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
    ncellP_lo = czd.groupby("node_zone").ncell_P.max().sum()
    ncellPS_lo = czd.groupby("node_zone").ncell_PS.max().sum()
    bl_ps = czd[czd.node_zone == "BL"].set_index("node_depth").ncell_PS
    wa12 = czd[(czd.node_zone == "WA") & (czd.node_depth == 12)]
    n12_nodes = int(wa12.nodes_P.iloc[0]) if len(wa12) else 0
    n12_ev = int(wa12.n_resP.iloc[0]) if len(wa12) else 0
    eq_all = ev[ev.located & ev.is_earthquake]
    n_s_only_nodes = int((model.mS & ~model.mP).sum())
    n_s_only_ev = int((eq_all.resS & ~eq_all.resP).sum())

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

    def v4row(pos, var, zone="BL"):
        r = v4h[(v4h.positions == pos) & (v4h["var"] == var) & (v4h.zone == zone)]
        return r.iloc[0] if len(r) else None

    def blkrow(var, size):
        r = blk[(blk["var"] == var) & (blk.block_km == size)]
        return r.iloc[0] if len(r) else None

    def tilerow(tile, run, var):
        r = tiles[(tiles.tile == tile) & (tiles.run == run) & (tiles["var"] == var)]
        return r.iloc[0] if len(r) else None

    def flrow(zone, mask="resP"):
        r = fl[(fl.zone == zone) & (fl["mask"] == mask)]
        return r.iloc[0] if len(r) else None

    def bc(zone, var, events, split_by="all", group="all", mc_plus=0.0, mask="resP"):
        r = bchk[(bchk.zone == zone) & (bchk["var"] == var) & (bchk.events == events) & (bchk.split_by == split_by)
                 & (bchk.group == group) & np.isclose(bchk.mc_plus, mc_plus) & (bchk["mask"] == mask)]
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
    q1_half = 0.5 * (r1.ci_hi - r1.ci_lo)
    # v4-hypocentre control
    hv4P, hv4S = v4row("v4", "dVp"), v4row("v4", "dVs")
    hmixP, hmixS = v4row("v5.1 epicentre, v4 depth", "dVp"), v4row("v5.1 epicentre, v4 depth", "dVs")
    hepiP = v4row("v4 epicentre, v5.1 depth", "dVp")
    hsameP, hsameS = v4row("v5.1 (events with a v4 hypocentre)", "dVp"), v4row("v5.1 (events with a v4 hypocentre)",
                                                                             "dVs")
    # block placements
    bk = {(v, s): blkrow(v, s) for v in ("dVp", "dVs", "Vp", "Vs") for s in (40, 60)}
    # tile runs
    tr = {(t, r, v): tilerow(t, r, v) for t, r in (("blanco_m0", "d200_s20"), ("blanco_m0", "d200_s60"),
                                                   ("blanco_m0", "d200_s120"), ("blanco_m0t", "d200_s20"),
                                                   ("blanco_m0t", "d200_s60"), ("blanco_m0tnf", "d200_s20"))
          for v in ("dVp", "dVs")}
    keep = [tr[("blanco_m0", run, v)].event_mean_res / tr[("blanco_m0", "d200_s20", v)].event_mean_res
            for run in ("d200_s60", "d200_s120") for v in ("dVp", "dVs")]
    keep_lo, keep_hi = min(keep), max(keep)
    # multiple tests
    prim = ts[ts.family == "primary"]
    n_prim, n_all = len(prim), len(ts)
    pass_prim = prim[prim.passed]
    pass_all = ts[ts.passed_all]
    bl_pass = pass_prim[pass_prim.question.str.startswith("Q2 seismicity") & (pass_prim.zone == "BL")]
    n_bl_signals = bl_pass.signal.nunique()
    other_pass = pass_prim.drop(index=bl_pass.index)
    # b-value
    wa_bg = tb[(tb.zone == "WA") & (tb["var"] == "Vp") & (tb.events == "background")].iloc[0]
    wa_bg_d = tb[(tb.zone == "WA") & (tb["var"] == "dVp") & (tb.events == "background")].iloc[0]
    wa_all = tb[(tb.zone == "WA") & (tb["var"] == "Vp") & (tb.events == "all above Mc")].iloc[0]
    q_wa_bg = ts[(ts.zone == "WA") & (ts.test == "b(high Vp_res) - b(low), background only")].q_BH.iloc[0]
    q_wa_all = ts[(ts.zone == "WA") & (ts.test == "b(high Vp_res) - b(low)")].q_BH.iloc[0]
    wa_on = bc("WA", "Vp", "background", "box", "ON")
    wa_pu = bc("WA", "Vp", "background", "box", "PU")
    wa_mc2 = bc("WA", "Vp", "background", "all", "all", 0.2)
    wa_d1 = bc("WA", "Vp", "background", "node depth", "1-9 km")
    wa_d2 = bc("WA", "Vp", "background", "node depth", "12-18 km")
    wa_any = bc("WA", "Vp", "background")
    n_b_tests = int(ts.question.str.startswith("Q2 b-value").sum())
    n_b_bg = int(ts.question.eq("Q2 b-value (background events)").sum())
    # flags
    fbl, fmt_, fwa = flrow("BL"), flrow("MTJ"), flrow("WA")
    mtj_all, mtj_dg = t2row("MTJ", "dVp"), t2row("MTJ", "dVp", "depth_good")
    bl_dg = t2row("BL", "dVp", "depth_good")
    # Q3 columns with deep D95
    deep = t3c[(t3c.d95 > 50) & t3c.box.isin(["GO", "MT"])]
    go_brk = deep[deep.lat > 41.5]
    go_mtj = deep[deep.lat <= 41.5]
    msh = t3c[(t3c.lon > -122.45) & (t3c.lon < -122.0) & (t3c.lat >= 46.0) & (t3c.lat <= 46.45)]
    msh_deep = msh.sort_values("d95").iloc[-1] if len(msh) else None
    msh_d95_wo = np.nan
    if msh_deep is not None:
        em = ev[ev.located & ev.is_earthquake & ev.depth_good & (ev.iy == msh_deep.iy) & (ev.ix == msh_deep.ix)]
        msh_d95_wo = em[~em.review.fillna("").str.contains("P1:deep")].depth.quantile(0.95)
    t3rv = t3[t3.zone.astype(str).str.contains("without review")]
    # Q4 stable surfaces
    st_vp = t4s[(t4s["var"] == "Vp") & t4s.stable]
    n_st_strand = int(st_vp["flags"].fillna("").str.contains("Blanco strand").sum())
    st_norec = st_vp[st_vp["flags"].fillna("").str.contains("not recovered in 3 of 3")].surface_id.tolist()
    # Q5 shared node
    same_wbd_sd = abs(seg.VpVs_events.iloc[0] - seg.VpVs_events.iloc[1]) < 1e-6
    ebl = ev[ev.located & ev.is_earthquake & ev.resPS & (ev.node_zone == "BL")]
    segs = np.full(len(ebl), "", dtype=object)
    from v51_struct_common import BLANCO_SEGMENTS
    for k, _, lo, hi in BLANCO_SEGMENTS:
        segs[((ebl.lat >= 42.4) & (ebl.lat <= 45.0) & (ebl.lon >= lo) & (ebl.lon < hi)).values] = k
    ebl = ebl.assign(seg=segs)
    ebl = ebl[ebl.seg != ""]
    nseg = ebl.groupby("node").seg.nunique()
    shared = nseg[nseg > 1].index
    n_shared_nodes, n_shared_ev = len(shared), int(ebl.node.isin(shared).sum())
    # review sets
    rv = ev[ev.located & ev.is_earthquake].review.fillna("")
    evr = ev[ev.located & ev.is_earthquake]
    n_rv = {k: int(rv.str.contains(k).sum()) for k in ("P1:deep", "P2:", "P3:", "P4:far_only", "P4:deep_pass")}
    n_rv_res = {k: int((rv.str.contains(k) & evr.resP).sum()) for k in n_rv}
    n_p1_ref_res = int((rv.str.contains("P1:") & ~rv.str.contains("P1:deep") & evr.resP).sum())
    p3_res = evr[rv.str.contains("P3:") & evr.resP].node_zone.value_counts().to_dict()
    # v4 counterpart
    cr = v4[v4.model == "CRESCENT_Gen0.nc"].set_index("group")
    de = v4[v4.model == "Delph2018.nc"].set_index("group")
    crb = cr.loc["Blanco"]

    p = []
    A = p.append
    A("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
      "content='width=device-width,initial-scale=1'><title>Structure and seismicity v5.1</title>"
      f"<style>{CSS}</style></head><body>")
    A("<h1>Velocity structure and seismicity, catalog v5.1</h1>")
    A("<p class='cap'>Model v0.3.1, resolved nodes only. Catalog v5.1 with the v5.1 magnitudes, the v5.1 clustering "
      "and fault surfaces v1. 2026-11-25, revised the same day after an adversarial review (b-value statement, Q1 "
      "detection limit, controls of the Blanco node-scale result, multiple-testing families) and rerun on the "
      f"fixed sibling products (clusters {html.escape(prov.get('clusters commit', '?'))}, faults "
      f"{html.escape(prov.get('faults commit', '?'))}; v5.1 magnitudes with production M unchanged). Private working "
      "report; not peer reviewed. Code: <code>workflow/06_analysis/v51_struct*.py</code> on branch "
      "analysis/v51-struct; input versions and checksums in input_commits.json.</p>")

    # ---------------- context
    A("<div class='box'><h3 style='margin-top:0'>Context since this analysis was built</h3><ul>")
    A("<li>The PI adopted model v0.4.0 (v0.3.1 re-inverted with the re-timed picks, D-B082); v0.4.1 (with a surface "
      "floor in the inversion, D-B083) is being built. Catalog v6.0 (D-B079, NonLinLoc with SSST station terms) will "
      "supersede v5.1, and these v5.1 products will be redone on v6.0. Nothing here was rerun on v0.4.0, except "
      "that v0.4.0's Blanco tile (blanco_m0t, damping 200, smoothing 20) is one of the tile runs in section 6.</li>")
    A("<li>Waveform review of four v5.1 location problems (catalog_v5/v5.1_review): "
      f"<b>P1</b> Mount St Helens deep events, likely false (weekly artificial signal on analog UW channels): none of "
      f"the {n_rv['P1:deep']} is in a resolved cell (all deeper than 30 km; Washington is resolved only to 18 km), "
      "but they set the 95th-percentile depth (D95) of the Mount St Helens column in Q3 ("
      + (f"{int(msh_deep.n_review_P1deep)} of its {int(msh_deep.n)} well-constrained events; D95 "
         f"{f(msh_deep.d95, 0)} km with them, {f(msh_d95_wo, 0)} km without" if msh_deep is not None else "")
      + f"). {n_p1_ref_res} of the P1 shallow "
      "reference and Wednesday-window events are in Washington resolved cells at 1-9 km. "
      f"<b>P2</b> Brookings 2012 (two depth solutions, NonLinLoc took about 56 km): none of the {n_rv['P2:']} is in "
      "a resolved cell; they make the two Gorda columns near Brookings with D95 58-63 km in Q3. "
      f"<b>P3</b> tensor events (v5.1 a median 13.7 km deeper than ComCat): {n_rv_res['P3:']} in resolved cells "
      f"({', '.join(f'{k} {v}' for k, v in p3_res.items())}). "
      f"<b>P4</b> far_only events (moved from near the surface in v4 to 44-64 km): {n_rv_res['P4:far_only']} in "
      f"resolved cells, {int(fbl.n_far_only) if fbl is not None else 0} of them at the Blanco; the deep_pass events: "
      f"{n_rv_res['P4:deep_pass']}.</li>")
    A("<li>The magnitude piece found that its phase19 ML inversion stops before converging. Production M is "
      "unchanged (PI decision pending); converged, whole offshore regions would move (Blanco -0.68, Gorda and "
      "Mendocino -0.28, onshore +0.01). The b-value splits here compare events within one zone, so a regional "
      "shift does not change them to first order, but the set of events above the box Mc would change.</li>")
    A("</ul></div>")

    # ---------------- summary
    A("<div class='box'><h3 style='margin-top:0'>Summary</h3><ul>")
    A(f"<li><b>Testable sample.</b> {pP:.1f}% of the {n_eq:,} located earthquakes ({nP:,}) are in cells where Vp passes "
      f"the checkerboard tests, and {pPS:.1f}% ({nPS:,}) where Vp and Vs both pass, so Vp/Vs can be read. All of "
      "the Vp/Vs cells are at the Blanco transform, at the 15 and 18 km nodes (a few at 29 km). Vp alone is "
      "resolved in three places: the Blanco at 15, 18 and 29 km, the Gorda slab under Cape Mendocino at 29 km, and "
      "southwest Washington (Mount St. Helens, 1-9 km; Puget and the Cascades foothills, 12-18 km). Summed over the "
      f"resolved depth nodes, the resolved volume holds about {ncellP:.0f} independent checkerboard cells for Vp and "
      f"{ncellPS:.0f} for Vp/Vs. That is an upper bound: depth nodes 2-3 km apart (1-9 km at Mount St. Helens, 15 "
      "and 18 km at the Blanco) are counted as independent, although the 40-100 km checkerboards do not separate "
      f"them. Counting only the largest layer of each zone gives {ncellP_lo:.0f} and {ncellPS_lo:.1f} (the Blanco "
      f"Vp/Vs layer is {f(bl_ps.get(15.0), 1)} cells at 15 km and {f(bl_ps.get(18.0), 1)} at 18 km). That cell "
      "count, not the number of events, limits every test below.</li>")
    A(f"<li><b>Q1, swarms against mainshock-aftershock sequences (Blanco only).</b> {int(r1.n_swarm)} swarm and "
      f"{int(r1.n_other)} mainshock-aftershock families have members in Vp/Vs-resolved cells. Their median Vp/Vs "
      f"values are {f(r1v.med_swarm, 3)} and {f(r1v.med_other, 3)}. The coverage-matched difference is "
      f"{f(r1['diff'], 3, True)} (95% block bootstrap {iv(r1.ci_lo, r1.ci_hi, 3)}, permutation p {f(r1.p_perm, 2)}). "
      f"The families fall in {int(r1.n_blocks)} blocks of 40 km, the swarms in {n_sw_blocks}. The bootstrap interval "
      f"sets what can be excluded: it has a half-width of {f(q1_half, 2)}, so differences from {f(r1.ci_lo, 2, True)} "
      f"to {f(r1.ci_hi, 2, True)} are not excluded. That covers most of the 0.05-0.15 range of published "
      "differences between settings. (The permutation range, "
      f"{iv(r1.null_lo, r1.null_hi, 2)}, treats neighbouring families as independent and is not a detection "
      "limit.) The data neither support nor exclude a fluid control of swarms.</li>")
    A(f"<li><b>Q2, where the events are (node scale).</b> At the Blanco, the nodes that hold events are slower than "
      f"coverage and depth predict: {f(b_dVp.event_mean_res, 2, True)}% in Vp change "
      f"({iv(b_dVp.res_ci_lo, b_dVp.res_ci_hi)}) and {f(b_dVs.event_mean_res, 2, True)}% in Vs change "
      f"({iv(b_dVs.res_ci_lo, b_dVs.res_ci_hi)}), against {iv(b_dVp.res_null_lo, b_dVp.res_null_hi)}% and "
      f"{iv(b_dVs.res_null_lo, b_dVs.res_null_hi)}% for randomized catalogs and {iv(sP[0], sP[1])}% and "
      f"{iv(sS[0], sS[1])}% for the checkerboard inversion errors. <b>Four controls weaken it.</b> (a) With the "
      "catalog v4 hypocentres of the same events (not located in v0.3.1), the Vp part drops to "
      f"{f(hv4P.event_mean_res, 2, True)}% {iv(hv4P.res_ci_lo, hv4P.res_ci_hi)}, and to "
      f"{f(hmixP.event_mean_res, 2, True)}% with v5.1 epicentres and v4 depths: the Vp signal comes from the v5.1 "
      f"depths, which were computed in this model. The Vs part stays at {f(hv4S.event_mean_res, 2, True)}% "
      f"{iv(hv4S.res_ci_lo, hv4S.res_ci_hi)}. (b) Counting each occupied node once instead of each event gives "
      f"only {f(b_dVp.occ_mean_res, 2, True)}% and {f(b_dVs.occ_mean_res, 2, True)}%: a few busy nodes carry it. "
      "(c) Its bootstrap interval excludes zero in "
      f"{int(bk[('dVp', 40)].w_excludes_0)}/16 (Vp) and {int(bk[('dVs', 40)].w_excludes_0)}/16 (Vs) placements of "
      f"the 40 km block grid, and in {int(bk[('dVp', 60)].w_excludes_0)}/16 and {int(bk[('dVs', 60)].w_excludes_0)}"
      "/16 with 60 km blocks; the stratified Spearman versions exclude zero in "
      f"{int(bk[('dVp', 40)].s_excludes_0)}/16 and {int(bk[('dVs', 40)].s_excludes_0)}/16 at 40 km and "
      f"{int(bk[('dVp', 60)].s_excludes_0)}/16 and {int(bk[('dVs', 60)].s_excludes_0)}/16 at 60 km. (d) The "
      f"{len(bl_pass)} Blanco tests that pass are {n_bl_signals} correlated signals (Vp and Vs), each counted "
      "four times (change and absolute value, each as an event-weighted mean and a Spearman correlation).</li>")
    A(f"<li><b>Q2 at the resolved scale.</b> With the field smoothed to the resolved scale (Gaussian, sigma 20 km), "
      f"the anomaly is {f(sm_dVp.event_mean_res, 2, True)}% ({iv(sm_dVp.res_ci_lo, sm_dVp.res_ci_hi)}) for Vp and "
      f"{f(sm_dVs.event_mean_res, 2, True)}% ({iv(sm_dVs.res_ci_lo, sm_dVs.res_ci_hi)}) for Vs; both intervals "
      "include zero. This smoothing is applied after the inversion. Blanco tile inversions that smooth more "
      "themselves keep most of the anomaly at the events in value, though their intervals include zero (smoothing 60: "
      f"{f(tr[('blanco_m0', 'd200_s60', 'dVp')].event_mean_res, 2, True)}% "
      f"{iv(tr[('blanco_m0', 'd200_s60', 'dVp')].res_ci_lo, tr[('blanco_m0', 'd200_s60', 'dVp')].res_ci_hi)} / "
      f"{f(tr[('blanco_m0', 'd200_s60', 'dVs')].event_mean_res, 2, True)}% "
      f"{iv(tr[('blanco_m0', 'd200_s60', 'dVs')].res_ci_lo, tr[('blanco_m0', 'd200_s60', 'dVs')].res_ci_hi)}; "
      f"smoothing 120: {f(tr[('blanco_m0', 'd200_s120', 'dVp')].event_mean_res, 2, True)}% / "
      f"{f(tr[('blanco_m0', 'd200_s120', 'dVs')].event_mean_res, 2, True)}%, Vp / Vs change), and v0.4.0's "
      "Blanco tile with the re-timed picks reproduces it "
      f"({f(tr[('blanco_m0t', 'd200_s20', 'dVp')].event_mean_res, 2, True)}% / "
      f"{f(tr[('blanco_m0t', 'd200_s20', 'dVs')].event_mean_res, 2, True)}%, intervals excluding zero). The "
      "40-100 km checkerboards certify neither scale. Under the rule that only resolved structure is interpreted, the Blanco result is not usable "
      f"as structure. Vp/Vs at the events is not clearly higher ({f(b_VpVs.event_mean_res, 3, True)}, "
      f"{iv(b_VpVs.res_ci_lo, b_VpVs.res_ci_hi, 3)}; smoothed {f(sm_VpVs.event_mean_res, 3, True)}, "
      f"{iv(sm_VpVs.res_ci_lo, sm_VpVs.res_ci_hi, 3)}). Nothing passes at the Mendocino triple junction or in "
      f"Washington. At the Mendocino 29 km node the Vp change flips sign with well-constrained depths only "
      f"({f(mtj_all.event_mean_res, 2, True)}% for all events, {f(mtj_dg.event_mean_res, 2, True)}% for "
      f"depth_good): {f(fmt_.pct_depth_biased_deep, 0)}% of the resolved Mendocino events are depth_quality "
      "biased_deep.</li>")
    A(f"<li><b>Q2, b-value.</b> For all events above Mc, no split between slow and fast nodes passes (largest: "
      f"Washington Vp {f(wa_all.db_high_minus_low, 2, True)} {iv(wa_all.ci_lo, wa_all.ci_hi)}, p "
      f"{f(wa_all.p_perm, 3)}, q {f(q_wa_all, 2)}). <b>For background events only, the Washington Vp split passes "
      f"the pass rule</b>: b is {f(wa_bg.b_low, 2)} below and {f(wa_bg.b_high, 2)} above the coverage-matched mean "
      f"(difference {f(wa_bg.db_high_minus_low, 2, True)} {iv(wa_bg.ci_lo, wa_bg.ci_hi)}, p {f(wa_bg.p_perm, 3)}, q "
      f"{f(q_wa_bg, 3)} in the primary family). It holds in both Mc boxes (Other onshore {f(wa_on.b_low, 2)} vs "
      f"{f(wa_on.b_high, 2)}, Puget {f(wa_pu.b_low, 2)} vs {f(wa_pu.b_high, 2)}) and above Mc + 0.2 "
      f"({f(wa_mc2.b_low, 2)} vs {f(wa_mc2.b_high, 2)}); {int(wa_any.n_ML)} of its {int(wa_any.n_low + wa_any.n_high)} "
      f"events have ML, and {int(wa_any.n_review)} are in a waveform-review set. By node depth: 1-9 km "
      f"{f(wa_d1.b_low, 2) if wa_d1 is not None else 'n/a'} vs {f(wa_d1.b_high, 2) if wa_d1 is not None else 'n/a'}, "
      f"12-18 km {f(wa_d2.b_low, 2) if wa_d2 is not None else 'n/a'} vs "
      f"{f(wa_d2.b_high, 2) if wa_d2 is not None else 'n/a'}. It is one pass among {n_b_tests} b-value splits "
      f"({n_b_bg} of them background-only), and the same events split by Vp change do not differ "
      f"({f(wa_bg_d.db_high_minus_low, 2, True)} {iv(wa_bg_d.ci_lo, wa_bg_d.ci_hi)}). A lead to recheck on catalog "
      "v6.0 and model v0.4.x, not a result.</li>")
    A("<li><b>Q3, depth extent.</b> A Vp level cannot be tested: only "
      f"{int(t3[(t3.test == 'Vp level at d90') & (t3.zone == 'BL')].n_columns.iloc[0])} Blanco columns, and 0-3 "
      "elsewhere, have resolved Vp on both nodes around their 90th-percentile depth. Against the slab (Slab2, no "
      "tomography needed): in the Mendocino box the deepest well-located events lie a median "
      f"{f(t3[(t3.zone == 'MT')]['median'].iloc[0], 1)} km below the slab top and do not follow its shape more closely "
      "than random pairing. In the Puget and onshore boxes they stop 27-31 km above the slab. "
      f"{len(go_brk) + len(go_mtj)} Gorda-box columns have D95 deeper than 50 km where the slab is shallow or "
      f"absent: {len(go_brk)} near Brookings (D95 {f(go_brk.d95.min(), 0)}-{f(go_brk.d95.max(), 0)} km; review P2) "
      f"and {len(go_mtj)} at {f(go_mtj.lat.median(), 1)} N, {f(-go_mtj.lon.max(), 2)}-{f(-go_mtj.lon.min(), 2)} W "
      f"(D95 {f(go_mtj.d95.min(), 0)}-{f(go_mtj.d95.max(), 0)} km, no Slab2 there): possible depth errors.</li>")
    tv = t4p[(t4p["var"] == "Vp") & (t4p.subset == "all")].iloc[0]
    tst = t4p[(t4p["var"] == "Vp") & (t4p.subset == "stable")].iloc[0]
    A(f"<li><b>Q4, fault surfaces v1.</b> {int(tv.n_surfaces)} surfaces have at least 10 mesh vertices in "
      "Vp-resolved cells. Their normals are no more aligned with the horizontal Vp gradient than randomly rotated "
      f"surfaces (mean alignment {f(tv.align_mean, 2)} against {f(tv.align_null_mean, 2)}, p "
      f"{f(tv.align_p_greater, 2)}). They also do not sit at stronger gradients than other resolved nodes of the "
      f"same coverage (median ratio {f(tv.grad_ratio_med, 2)}, {iv(tv.grad_ratio_ci_lo, tv.grad_ratio_ci_hi, 2, False)}). "
      f"The {int(tst.n_surfaces)} stable surfaces give {f(tst.align_mean, 2)} against {f(tst.align_null_mean, 2)} "
      f"(p {f(tst.align_p_greater, 3)}); {n_st_strand} of them are Blanco strands whose strike comes from a map-view "
      "band search" + (f", and {', '.join(st_norec)} is not recovered in 3 of 3 detection reruns" if st_norec else "")
      + ". At 40-60 km resolution a fault-scale contrast cannot be seen.</li>")
    sc = segc[(segc.x == "VpVs_events") & (segc.y == "swarm_share")].iloc[0]
    A("<li><b>Q5, Blanco segments.</b> The two western segments (West Blanco and Surveyor Depressions) have the "
      f"highest Vp/Vs at their events ({f(seg.VpVs_events.iloc[0], 2)} and {f(seg.VpVs_events.iloc[1], 2)}) and the "
      f"highest swarm shares ({f(seg.swarm_share.iloc[0], 2)}, {f(seg.swarm_share.iloc[1], 2)}). The Blanco Ridge "
      f"has the lowest Vp/Vs ({f(seg.VpVs_events.iloc[3], 2)}) and a swarm share of {f(seg.swarm_share.iloc[3], 2)}. "
      f"Over five segments the Spearman correlation is {f(sc.spearman, 2)} with exact p {f(sc.p_exact, 2)}. The "
      "segment values are not independent either"
      + (f": the two western medians are the same node value ({f(seg.VpVs_events.iloc[0], 3)})" if same_wbd_sd
         else "")
      + f", and {n_shared_nodes} nodes hold {n_shared_ev} events of two segments. A pattern to test with relocated "
      "swarms and a finer model, not a result.</li>")
    A(f"<li><b>Multiple tests.</b> The primary family has {n_prim} tests (the 77 of the first version plus the "
      f"{n_b_bg} background-only b-value splits, which were computed and must be counted). "
      f"{int((prim.q_BH < 0.05).sum())} have a Benjamini-Hochberg q below 0.05; {len(pass_prim)} also have a "
      "block-bootstrap interval that excludes the no-effect value and count as passed: "
      f"{len(bl_pass)} node-scale Blanco Q2 statistics ({n_bl_signals} signals, Vp and Vs)"
      + (", and " + "; ".join(html.escape(f"{r.zone} {r.test}") for _, r in other_pass.iterrows())
         if len(other_pass) else "")
      + f". With every computed variant added ({n_all} tests; variant sets of Q1-Q3 and the Q5 area tests), "
      f"{len(pass_all)} pass, all in the same {pass_all.signal.nunique()} signals ("
      + "; ".join(f"{html.escape(s)} {n}" for s, n in pass_all.signal.value_counts().items())
      + "): the extra passes are the Blanco statistics repeated for depth_good, above-Mc and background events. "
      "None passes at the resolved scale. tests_summary.csv lists both families and a 'signal' column that groups "
      "tests measuring the same thing.</li>")
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
    A("<li>Flags of the events in Vp-resolved cells (flags_resolved_by_zone.csv): at the Blanco "
      f"{f(fbl.pct_depth_unconstrained, 0)}% have unconstrained depth, {f(fbl.pct_tier_D, 0)}% are tier D (median "
      f"h_max of these {f(fbl.tierD_h_max_med_km, 0)} km, i.e. more than half a bootstrap block) and "
      f"{int(fbl.n_far_only)} are far_only; at the Mendocino 29 km node {f(fmt_.pct_depth_biased_deep, 0)}% are "
      f"biased_deep and {f(fmt_.pct_tier_D, 0)}% tier D; in Washington {f(fwa.pct_tier_D, 0)}% are tier D (median "
      f"h_max {f(fwa.tierD_h_max_med_km, 0)} km) and {int(fwa.n_floor_limited)} floor_limited. The main tests use all "
      "located earthquakes and report depth_good variants. These agree at the Blanco (Vp change "
      f"{f(bl_dg.event_mean_res, 2, True)}% {iv(bl_dg.res_ci_lo, bl_dg.res_ci_hi)}) but not at the Mendocino node "
      f"(sign flip, {f(mtj_all.event_mean_res, 2, True)}% to {f(mtj_dg.event_mean_res, 2, True)}%).</li>")
    A("<li>Magnitudes: v5.1 M (mag piece; production M unchanged by its review fixes). Mc per box from the clusters "
      "piece (v4 rule on v5.1 magnitudes). Clustering: the v5.1 nearest-neighbour families (types swarm, "
      "mainshock-aftershock, multiplet) and their classes; family ids and types are unchanged by the clusters "
      "piece's review fixes. Fault surfaces v1: summary, assignment, and the 1 km mesh (constrained vertices only); "
      "the faults review fixes changed assignment labels (a new class for Axial, which has no resolved node) and "
      "surface flags, not the surfaces' geometry or stable flags.</li>")
    A(f"<li>Model v0.3.1 ({model.shape[2]} x {model.shape[1]} x {model.shape[0]} = "
      f"{model.shape[0] * model.shape[1] * model.shape[2]:,} nodes, 0.125 x 0.1 deg, depth -3 to "
      "125 km). Vp and its change are sampled where resolution_p &gt; 0. Vs, Vp/Vs and their changes are all sampled "
      "on one node set, the nodes resolved for both phases (resolution_p &gt; 0, resolution_s &gt; 0 and "
      "vs_reverted = 0), so the Vs and Vp/Vs tests use the same nodes and strata. "
      "These are the D-B018 checkerboard criteria (local correlation &gt;= 0.7 and amplitude &gt;= 0.5 at 40, 60 or 100 km "
      "cells, with pick noise, both pattern phases), merged under D-B049, and D-B051 for Vs. "
      f"Resolved nodes: {int(model.mP.sum())} for P, {int(model.mS.sum())} for S, {int(model.mPS.sum())} for both. "
      f"The {n_s_only_nodes} nodes resolved for S but not P (holding {n_s_only_ev} located earthquakes) are not used. "
      f"Resolved nodes in water or air by the v0.3.2 medium flag: {model.n_res_not_rock}. The D-B073 shallow-crust "
      "rule is not in v0.3.1 and is not used. Only the medium flag is read from v0.3.2. The masks are empty at 21 "
      f"and 25 km everywhere; at 12 km only {n12_nodes} Washington nodes are resolved (holding {n12_ev} events), and "
      "offshore nothing is resolved above 13.5 km.</li>")
    A("<li>An event belongs to the node nearest to its hypocentre (voxel boundaries halfway between nodes). "
      f"A stricter variant requires all eight corners of the enclosing grid cell to be resolved: {nP8:,} events for "
      f"Vp and {nPS8:,} for Vp/Vs.</li></ul>")

    A("<h2>2. Method: controls for circularity</h2>")
    A("<p>The events give the rays. Ray density (DWS) and resolution are therefore high where events are, and a "
      "tomographic error that follows the events would look like a structure-seismicity relation. The v5.1 "
      "hypocentres were also located in this model (plus the D-B073 shallow-crust rule), so a velocity error and a "
      "location error can trade off. Each test uses four controls:</p><ol>")
    A("<li><b>Coverage-matched anomaly.</b> Each resolved node gets a stratum: zone x depth node x DWS tercile (P DWS "
      "for Vp, S DWS for Vs and Vp/Vs). V_res is the node value minus the mean over the resolved nodes of its stratum. "
      "A relation in V_res cannot come from depth or from the amount of ray coverage alone. DWS quintiles give the "
      "same results (t2_density.csv, columns *_q5).</li>")
    A("<li><b>Randomized catalogs.</b> The event set of each occupied node is moved to a random resolved node of the "
      "same stratum (2,000 times). This keeps the event clustering and the coverage. For Q1, family types are "
      "permuted among families of the same stratum. These node-level permutations treat neighbouring nodes as "
      "independent, so their ranges are too narrow and are not used as detection limits.</li>")
    A("<li><b>Block bootstrap.</b> 95% intervals come from resampling 40 km blocks, the finest resolved checkerboard "
      "cell, with all events and nodes of a block together (2,000 resamples). This is the interval to read for the "
      "effect size. For the Blanco Q2 statistics the interval also depends on where the block grid falls (section "
      "6, t2_block_offsets*.csv).</li>")
    A("<li><b>Synthetic inversion errors.</b> The checkerboard inversions of each tile (40, 60 and 100 km cells; "
      "pick noise; pattern phases 0 and 1; and inversions started from perturbed hypocentres, noise_ploc) were run "
      "on the real event and station geometry. The error, recovered minus true, follows the events only through the "
      "inversion. The same statistics on these error fields give the size of an artefact.</li></ol>")
    A("<p>Added after review, for the Blanco Q2 result: the catalog v4 hypocentres of the same events (not located in "
      "v0.3.1; v4 depths have their own problems, D-B057) sampled in the same model; each occupied node counted once; "
      "16 placements of 40 and 60 km block grids; and the same statistic on Blanco tile inversions with stronger "
      "smoothing and with the re-timed picks. Each event's catalog errors (horizontal sqrt((h_min^2 + "
      "h_max^2)/2)/1.515, vertical sig_z) are resampled 50 times to test the sensitivity to location. None of this "
      "moves a hypocentre in any product. Multiple tests: Benjamini-Hochberg q within the primary family and over "
      "primary plus variants (tests_summary.csv, section 15).</p>")

    # ---------------- coverage
    A("<h2>3. Fraction of events in resolved cells</h2>")
    A(fig_html(figs["coverage"], "Left: resolved nodes at four depth nodes, with the events of each layer. Right: "
               "located earthquakes in resolved cells by catalog region."))
    cz = czd[["node_zone", "node_depth", "nodes_P", "nodes_PS", "area_P_km2", "ncell_P", "ncell_PS", "n_resP",
              "n_resPS", "n_depth_good", "n_aboveMc"]].copy()
    A("<p>By zone and depth node. ncell is the resolved area divided by the square of the finest resolved "
      "checkerboard cell: the number of independent velocity samples in that layer. Layers 2-3 km apart are not "
      "independent of each other, so the sum over layers is an upper bound.</p>")
    A(table_html(cz, "{:.4g}"))
    cv = cov[~cov.group.str.startswith("region") & ~cov.group.str.startswith("tier")][
        ["group", "n", "n_resP", "pct_resP", "n_resPS", "pct_resPS", "n_resP8", "n_resPS8"]]
    A(table_html(cv, "{:.3g}"))
    none_key = "fault assignment: none"
    A("<p>Swarm events are over-represented in Vp/Vs-resolved cells "
      f"({f(g.loc['class: swarm', 'pct_resPS'], 1)}% against {f(pPS, 1)}% overall). The Blanco swarms are what "
      "resolves the 15-18 km layer. Events on no fault surface (assignment none, now only in fitted regions) are "
      f"under-represented ({f(g.loc[none_key, 'pct_resPS'], 1) if none_key in g.index else 'n/a'}%). Nothing is "
      "resolved at Axial, Endeavour, Explorer or the Juan de Fuca ridge and plate, in Gorda above 29 km, or in the "
      "offshore Oregon and Washington forearc.</p>")
    A("<p>Catalog flags and waveform-review sets of the events in resolved cells:</p>")
    A(table_html(fl, "{:.3g}"))

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
      f"{iv(r1s.ci_lo, r1s.ci_hi)}. What can be excluded is set by the bootstrap interval (half-width "
      f"{f(q1_half, 2)} in Vp/Vs), not by the permutation range (null_lo, null_hi).</p>")
    A("<p>Event level (events above Mc in Vp/Vs-resolved cells): the swarm share of clustered events in nodes above and "
      "below the coverage-matched mean.</p>")
    A(table_html(t1e, "{:.3g}"))
    A("<p>Family labels at the Blanco (clusters piece, null test after its review): a catalog shuffled in time over "
      "the whole period reproduces 39% of the Blanco nearest-neighbour links (same-month shuffles 62-92% across "
      "regions), but only 12.8 swarms on average against 70 in the real catalog. Part of the mainshock-aftershock "
      "and multiplet labelling is therefore not robust; the swarms are.</p>")

    # ---------------- Q2
    A("<h2>6. Q2: seismicity density and b-value against Vp/Vs and Vp change at fixed depth and coverage</h2>")
    A(fig_html(figs["t2"], "Mean coverage-matched anomaly at the events (each event counts once), by zone and "
               "variable. Blue: catalog with its block-bootstrap 95% interval. Grey band: 95% range of the "
               "randomized catalogs. Orange ticks: the same statistic on each checkerboard inversion error field "
               "(9 per zone)."))
    t2s = t2[t2.events.isin(["all", "all, field smoothed 20 km"])][["events"] + ["zone", "var", "n_nodes", "n_nodes_with_events", "n_events", "n_blocks",
                                  "rho_depth_only", "rho_cov", "rho_ci_lo", "rho_ci_hi", "rho_p", "event_mean_res",
                                  "res_ci_lo", "res_ci_hi", "res_null_lo", "res_null_hi", "res_p", "event_mean_res_q5",
                                  "occ_mean_res", "occ_ci_lo", "occ_ci_hi"]]
    A(table_html(t2s, "{:.3g}"))
    A("<p>rho_depth_only: Spearman correlation between the event count of a node and the raw value, within zone x "
      "depth (coverage not matched). rho_cov: the same on the coverage-matched anomaly, averaged over strata. "
      "event_mean_res: the mean anomaly at the events. occ_mean_res: the same with each occupied node counted once. "
      "Units: % for changes, km/s for Vp and Vs, ratio for Vp/Vs. The rows for well-constrained depths, events "
      "above Mc and background events only are in t2_density.csv (variant family). Their Blanco values have the "
      "same sign and size, e.g. depth_good Vp change "
      f"{f(bl_dg.event_mean_res, 2, True)}% {iv(bl_dg.res_ci_lo, bl_dg.res_ci_hi)}. At the Mendocino node they "
      f"do not: {f(mtj_all.event_mean_res, 2, True)}% for all events, {f(mtj_dg.event_mean_res, 2, True)}% "
      f"{iv(mtj_dg.res_ci_lo, mtj_dg.res_ci_hi)} for depth_good.</p>")
    A("<p><b>Reading.</b> At the node scale, Blanco nodes with more events are slower in Vp and Vs at the same depth "
      f"and coverage: {f(b_dVp.event_mean_res, 1, True)}% (Vp) and {f(b_dVs.event_mean_res, 1, True)}% (Vs). That is "
      f"about {f(abs(b_dVp.event_mean_res) / max(abs(sP[0]), abs(sP[1])), 0)} and "
      f"{f(abs(b_dVs.event_mean_res) / max(abs(sS[0]), abs(sS[1])), 0)} times the largest checkerboard artefact, and "
      "the inversions started from perturbed hypocentres do not give larger artefacts. It survives the location Monte "
      f"Carlo at a smaller size ({f(mcP.mc_median, 2, True)}% and {f(mcS.mc_median, 2, True)}%) and DWS quintiles. "
      "The checkerboard errors use true hypocentres and random perturbations; they do not simulate a systematic "
      "depth error such as the v5.1 Blanco depth bias. The controls below show how much of it depends on choices "
      "that the checkerboards do not test.</p>")
    A("<h3>Control 1: catalog v4 hypocentres in the same model</h3>")
    v4s = v4h[v4h.zone == "BL"][["positions", "mask", "var", "n_events", "n_nodes_with_events", "n_blocks",
                                 "event_mean_res", "res_ci_lo", "res_ci_hi", "res_null_lo", "res_null_hi", "res_p",
                                 "occ_mean_res", "rho_cov"]]
    A(table_html(v4s, "{:.3g}"))
    A("<p>The v4 hypocentres (GraphDD, catalog v4) were not located in v0.3.1. Restricted to the same events, the "
      f"v5.1 positions give {f(hsameP.event_mean_res, 2, True)}% (Vp change) and {f(hsameS.event_mean_res, 2, True)}% "
      f"(Vs change); the v4 positions give {f(hv4P.event_mean_res, 2, True)}% {iv(hv4P.res_ci_lo, hv4P.res_ci_hi)} and "
      f"{f(hv4S.event_mean_res, 2, True)}% {iv(hv4S.res_ci_lo, hv4S.res_ci_hi)}. v4 epicentres with v5.1 depths give "
      f"{f(hepiP.event_mean_res, 2, True)}% and v5.1 epicentres with v4 depths {f(hmixP.event_mean_res, 2, True)}% "
      f"(Vp) and {f(hmixS.event_mean_res, 2, True)}% (Vs). The Vp association follows the v5.1 depths, which were "
      "computed in this model, so it may be a velocity-depth trade-off; the Vs association does not depend on which "
      "hypocentres are used. Rows for Mendocino and Washington are in t2_v4hypo.csv. This is also the v4 counterpart "
      "on our own tomography that the first version said did not exist. The v4 depths are not publishable (D-B057), "
      "so this control says which coordinate carries the signal, not which depth is right.</p>")
    A("<h3>Control 2: block-grid placement</h3>")
    A(table_html(blk, "{:.3g}"))
    A("<p>16 placements of the bootstrap block grid (origin shifted by 0, 1/4, 1/2 and 3/4 of a block in x and y) "
      "for 40 and 60 km blocks. w_excludes_0: placements where the interval of the event-weighted anomaly excludes "
      "zero; s_excludes_0: the same for the stratified Spearman correlation. The event-weighted interval depends on "
      "where the grid falls; the rank correlation does not. All placements: t2_block_offsets.csv.</p>")
    A("<h3>Control 3: tile inversions with stronger smoothing and with re-timed picks</h3>")
    A(table_html(tiles, "{:.3g}"))
    A("<p>The same event-weighted statistic on the v0.3.1 Blanco resolved nodes and strata, with the field of each "
      "tile run (trilinear from the tile lattice, interior nodes). blanco_m0 d200_s20 is the run merged into v0.3.1; "
      "blanco_m0t d200_s20 is the Blanco tile of v0.4.0. Stronger smoothing inside the inversion keeps "
      f"{f(100 * keep_lo, 0)}-{f(100 * keep_hi, 0)}% of the anomaly at the events in value (with intervals that "
      "then include zero) and lowers the node spread "
      "(node_sd); the post-hoc Gaussian smoothing removes most of it. So 'only at the single-node scale' was too "
      "strong: the anomaly sits at 10-40 km, below or at the finest checkerboard cell, and the 40-100 km "
      "checkerboards certify neither. The verifier's nearest-node mapping gave similar values (smoothing 60 "
      "-1.71/-2.19, smoothing 120 -1.19/-1.49, re-timed -2.04/-2.89).</p>")
    A("<h3>Resolved scale</h3>")
    A("<p>Rows 'all, field smoothed 20 km' in t2_density.csv and open markers in the figure. The node-to-node "
      f"spread of the Vp change at the Blanco is {f(b_dVp.node_sd, 1)}%, but only {f(sm_dVp.node_sd, 1)}% after "
      "the Gaussian smoothing. For the smoothed Vp/Vs and Vs, the randomized catalogs give p &lt; 0.001 while the "
      "block bootstrap includes zero; a smoothed field is strongly autocorrelated, so the node permutation is too "
      "narrow there, and the bootstrap is the interval to read. Ten-kilometre structure at the Blanco may be real "
      "(a damaged or hydrated transform mantle) or a trade-off with the v5.1 depths (control 1).</p>")
    A(fig_html(figs["t2b"], "b-value difference between events in nodes above and below the coverage-matched mean, "
               "with block-bootstrap 95% intervals and the randomized-node null (grey)."))
    A(table_html(tb[["mask", "events", "zone", "var", "n_low", "n_high", "b_low", "b_high", "db_high_minus_low",
                     "ci_lo", "ci_hi", "null_lo", "null_hi", "p_perm", "n_blocks", "mc_boxes"]], "{:.3g}"))
    A("<p>b uses the Aki-Utsu estimator on M minus the Mc of the event's box (pooled excess magnitudes). The "
      "background-only rows are part of the primary test family. <b>Washington, background events, Vp:</b> b "
      f"{f(wa_bg.b_low, 2)} below and {f(wa_bg.b_high, 2)} above the coverage-matched mean, "
      f"{f(wa_bg.db_high_minus_low, 2, True)} {iv(wa_bg.ci_lo, wa_bg.ci_hi)}, p {f(wa_bg.p_perm, 3)}, q "
      f"{f(q_wa_bg, 3)}: it passes. The first version of this report said that b does not differ in any zone; that "
      "was wrong for this row. Checks (t2_bvalue_checks.csv, point values):</p>")
    wchk = bchk[(bchk.zone == "WA") & (bchk["var"] == "Vp") & (bchk.events == "background")][
        ["split_by", "group", "mc_plus", "n_low", "n_high", "b_low", "b_high", "db_high_minus_low", "n_ML",
         "n_review"]]
    A(table_html(wchk, "{:.3g}"))
    A("<p>Offshore b-values carry the magnitude piece's open problem: its phase19 ML inversion stops before "
      "converging. With the stored (production) M, ComCat ML gaps on anchors with at least 3 near picks are +0.59 "
      "at the Blanco, +0.33 at Gorda and +0.54 at Mendocino; converged, they would be +0.10, +0.02 and +0.26, with "
      "whole offshore regions moving (Blanco -0.68, Gorda and Mendocino -0.28, onshore +0.01). Within one zone a "
      "uniform shift does not change b, but D-B041 still asks for an ML-Mw check before b is compared between "
      "settings, and none is attempted here.</p>")

    # ---------------- Q3
    A("<h2>7. Q3: does the depth extent of seismicity follow a Vp level or the slab?</h2>")
    A(fig_html(figs["t3"], "Left: 95th-percentile depth of well-constrained events in each model column with at "
               "least 30 of them, against the Slab2 depth. Right: Vp at the 90th-percentile depth, only where both "
               "bracketing depth nodes are resolved."))
    A(table_html(t3.drop(columns=[c_ for c_ in ("note",) if c_ in t3]), "{:.3g}"))
    A(table_html(t3s, "{:.4g}"))
    A("<p>The Vp-level question is untestable on the resolved model. The checkerboards resolve isolated depth "
      "nodes (1-9, 15, 18 and 29 km; at 12 km only four Washington nodes; nothing at 21 and 25 km, which the "
      "tomography notes attribute to the vertical checker pattern). Almost no column has resolved nodes on both sides "
      "of its seismicity bottom. The 6 Blanco columns do not show a common Vp at D90 (spread larger than for depths "
      "placed in other columns). For the slab: in the Mendocino box, D95 is below the slab top in almost every "
      "column, and D95 minus slab depth spreads as much as with randomly paired columns. In the Puget and onshore "
      "boxes, D95 is shallower where the slab is deeper (Spearman " + f(t3[t3.zone == 'PU'].rho_d95_slab.iloc[0], 2)
      + " and " + f(t3[t3.zone == 'ON'].rho_d95_slab.iloc[0], 2) + "). Upper-plate seismicity there ends at a "
      "crustal depth, not at the slab.</p>")
    dcols = t3c[(t3c.d95 > 50)][["iy", "ix", "lon", "lat", "box", "zone", "n", "d50", "d95", "slab_km",
                                 "n_review_P1deep", "n_review_P2"]]
    A("<p><b>Columns with D95 deeper than 50 km</b> (possible depth errors where the slab is shallow or absent):</p>")
    A(table_html(dcols, "{:.4g}"))
    msh_txt = ""
    if msh_deep is not None:
        msh_txt = (f" The Mount St Helens column at {f(msh_deep.lat, 2)} N, {f(-msh_deep.lon, 3)} W has D95 "
                   f"{f(msh_deep.d95, 0)} km with {int(msh_deep.n_review_P1deep)} of its {int(msh_deep.n)} "
                   "well-constrained events in the review's P1 deep set (likely false events); without them it is "
                   f"{f(msh_d95_wo, 0)} km.")
    A(f"<p>The {len(go_brk)} Gorda-box columns near Brookings (42.2 N) hold the 2012 swarm (review P2: predicted S-P "
      "is not monotonic with depth in the v5.1 grids, so about 7-9 km and about 56 km both fit the distant "
      f"stations, and NonLinLoc took the deep one). The {len(go_mtj)} Gorda-box columns at "
      f"{f(go_mtj.lat.median(), 1)} N, west of the triple junction, have D95 "
      f"{f(go_mtj.d95.min(), 0)}-{f(go_mtj.d95.max(), 0)} km with no Slab2 there; they are not in a review set "
      "and should be checked the same way." + msh_txt + " Without the P1 and P2 events (variant family):</p>")
    A(table_html(t3rv.drop(columns=[c_ for c_ in ("note",) if c_ in t3rv]), "{:.3g}"))

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
      "alignment. Orientation is used for all surfaces, but 93 of the 142 surfaces are unstable under detection "
      "reruns (strike range median 103 deg, against 6.5 deg for the 49 stable ones; faults piece). The 'stable' rows, "
      f"{int(tst.n_surfaces)} surfaces for Vp, are the only ones whose orientation can be trusted. Their alignment is "
      f"{f(tst.align_mean, 2)} against {f(tst.align_null_mean, 2)} "
      f"(p {f(tst.align_p_greater, 2)}; "
      f"q {f(ts[ts.test == 'alignment |n.grad Vp| (stable surfaces)'].q_BH.iloc[0], 2)} after the correction for "
      f"multiple tests), on {int(tst.n_surfaces)} surfaces in {int(tst.n_blocks)} blocks. {n_st_strand} of them "
      "are Blanco strands (strike from a map-view band search; no recovery test)"
      + (f"; {', '.join(st_norec)} is not recovered in 3 of 3 detection reruns" if st_norec else "")
      + ". Flags of every surface used are in t4_surfaces.csv (column flags).</p>")

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
      "rank correlation cannot reach significance unless it is perfect. The segment values are not independent: "
      + ("the West Blanco and Surveyor medians are the same node value "
         f"({f(seg.VpVs_events.iloc[0], 4)}), and " if same_wbd_sd else "")
      + f"{n_shared_nodes} nodes hold {n_shared_ev} events of two segments. The pattern (higher Vp/Vs and more "
      "swarms in the west) agrees with the science-framing hypothesis, but the uncertainty intervals of neighbouring "
      "segments overlap.</p>")
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
      "volcano from its surroundings at 40 km resolution. The review's P1 deep events lie below the resolved "
      "depths and do not enter it.</p>")

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
      "catalog positions and on scales near the location error. It resamples random errors only; a systematic "
      "depth error (control 1) is not simulated.</p>")

    # ---------------- v4
    A("<h2>11. v4-era counterpart</h2>")
    A("<p>The earlier analysis, phase13_tomography.py (Route A era, 09-29), compared an earlier catalog with "
      "external Vs models (Delph et al. 2018; CRESCENT CVM Gen0). It used dlnVs relative to each model's depth mean "
      "and no resolution mask or coverage control. Its statistic, rerun on the v4 and the v5.1 hypocentres of the "
      "same events, shows what the relocation changed. The v4 counterpart on our own tomography is control 1 of "
      "section 6.</p>")
    A(table_html(v4, "{:.3g}"))
    other = cr.drop(index=[i for i in cr.index if i == "Blanco" or i.startswith("v5.1")], errors="ignore")
    sp = cr[cr.index.str.startswith("v5.1")]
    A(f"<p>The relocation moves Blanco events into faster CRESCENT Vs ({f(crb['change'], 1, True)} points in median "
      f"dlnVs). This statistic uses only the {int(crb.n):,} of {int(crb.n_group_all):,} Blanco events that lie inside "
      "CRESCENT at both positions, and those moved deeper (median "
      f"{f(crb.depth_med_v4, 1)} to {f(crb.depth_med_v51, 1)} km). For all Blanco events the medians are "
      f"{f(crb.depth_med_v4_all, 1)} and {f(crb.depth_med_v51_all, 1)} km, with a paired median shift of "
      f"{f(crb.depth_shift_med_all, 1, True)} km. In the other "
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
        ("Mount St Helens deep seismicity (review P1)", "below the resolved depths; likely false events"),
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
    for d in [OPEN[0].format(ncellP=ncellP, ncellPS=ncellPS, ncellP_lo=ncellP_lo, ncellPS_lo=ncellPS_lo)] + \
            [o.format(wa_db=f(wa_bg.db_high_minus_low, 2, True), wa_ci=iv(wa_bg.ci_lo, wa_bg.ci_hi),
                      wa_q=f(q_wa_bg, 3), n_b=n_b_tests) for o in OPEN[1:]]:
        A(f"<li>{html.escape(d)}</li>")
    A("</ul>")
    A("<h2>15. All tests</h2>")
    A(f"<p>family primary: {n_prim} tests, q_BH within them; family variant: the other computed sets; q_BH_all over "
      f"all {n_all}. passed / passed_all: q below 0.05 and a block-bootstrap interval excluding the no-effect value "
      "(ref).</p>")
    A(table_html(ts, "{:.3g}"))
    A("<h2>16. Files</h2><p>In data/catalog_v5/v5.1_analysis/struct/: coverage_*.csv, flags_resolved_by_zone.csv, "
      "t1_*.csv, t2_*.csv (including t2_v4hypo.csv, t2_block_offsets*.csv, t2_tile_runs.csv, t2_bvalue_checks.csv), "
      "t3_*.csv, t4_*.csv, t5_*.csv, synth_artifact*.csv, mc_location*.csv, v4_counterpart_external_vs.csv, "
      "tests_summary.csv, nodes_resolved_v031.csv, events_in_resolved_cells.csv.gz, input_commits.json, fig_*.png. "
      "On /wd1/mdenolle_data/catalog_v5/v5.1_analysis/struct/: events_struct_v51.parquet, nodes_v031.parquet, "
      "RESUME.md, run_all.sh, logs, builder_tables (the first version's tables, for comparison).</p>")
    A("</body></html>")
    return "\n".join(p)


PROPOSED = [
    "Resolved means resolution_p > 0 for Vp; Vs and Vp/Vs both use the nodes resolved for both phases "
    "(resolution_p > 0, resolution_s > 0 and vs_reverted == 0), on v0.3.1 as published. The 12/21/25 km mask gaps "
    "are not bridged, and no DWS-only or tile-weight criterion is added.",
    "Events are sampled at the nearest node (voxel). The all-eight-corners rule is reported as a stricter count, not "
    "used for the tests.",
    "Circularity control: coverage-matched anomalies within zone x depth node x DWS tercile (P DWS for Vp, S DWS for "
    "Vs and Vp/Vs); randomized catalogs that move each node's event set to a random node of the same stratum; family "
    "types permuted within strata; 40 km block bootstrap (finest resolved checkerboard cell) for all intervals.",
    "Use the tiles' checkerboard inversion errors (recovered minus true, noise and noise_ploc, both pattern phases) "
    "as the artefact yardstick, interpolated from the tile lattices onto the merged grid.",
    "Event set: located earthquakes only (no loc_source none, no is_earthquake False). Depth-sensitive variants use "
    "depth_good (tier A-C, depth_quality ok, not floor_limited, far_only or deep_pass), the clusters piece's "
    "definition. A result that changes sign with depth_good (Mendocino Vp change) is reported as not robust.",
    "Q1 family value: median over members in resolved cells, families with >= 3 such members; >= 50% (strict) and "
    "depth_good variants are reported. What a null result excludes is stated from the block-bootstrap interval, "
    "never from the permutation range.",
    "b-value: Aki-Utsu on M minus the Mc of the event's cluster box, events split at a coverage-matched anomaly of 0. "
    "Report only within a zone (D-B041).",
    "Blanco segments as longitude boxes (lat 42.4-45.0): WBD < -129.45 < SD < -128.95 < CD < -128.5 < BR < -127.45 "
    "< GD < -126.4, matching the v1 surfaces' literature names.",
    "Q3 depth extent: columns of one model node (0.125 x 0.1 deg) with >= 30 well-constrained events; Vp at D90/D95 "
    "only between two adjacent resolved depth nodes.",
    "Q4: constrained mesh vertices only; alignment of the horizontal normal with the horizontal gradient against "
    "random strike rotations; orientation results read only for the faults piece's 'stable' surfaces.",
    "Multiple testing: Benjamini-Hochberg within a primary family that holds every computed main-set test, "
    "including the background-only b-value splits; q over primary plus variant sets reported alongside; a 'signal' "
    "column groups correlated tests, and passes are counted by signal as well as by test.",
    "Test every relation also on the field smoothed to the resolved scale (Gaussian, sigma 20 km = half the finest "
    "resolved checkerboard cell) and on tile inversions with stronger smoothing; a relation that the checkerboards "
    "do not certify is reported as unresolved.",
    "A node-scale event-velocity association is reported with four controls: hypocentres not located in the model "
    "(v4), each occupied node counted once, several block-grid placements, and tile runs with other smoothing or "
    "picks.",
    "Do not use this analysis for interpretation in the paper beyond the statement of what is untestable and the "
    "null results at the resolved scale (with their bootstrap intervals). Redo it on catalog v6.0 (D-B079) with "
    "model v0.4.x (D-B082/D-B083) once its resolution tests (including a 20 km checkerboard or spike test at the "
    "Blanco) and its ensemble exist.",
]

OPEN = [
    "The resolved volume is small: about {ncellP:.0f} independent Vp cells and {ncellPS:.0f} Vp/Vs cells summed over "
    "depth nodes, or {ncellP_lo:.0f} and {ncellPS_lo:.1f} counting the largest layer per zone. The D-B049 gate keeps "
    "the s480 tiles at full weight, but their checkerboards barely recover, so Gorda, Oregon and Washington are "
    "largely unresolved. Any structure-seismicity claim outside the Blanco needs a less smoothed or re-inverted model "
    "with new resolution tests.",
    "The checkerboard masks are empty at 21 and 25 km, and at 12 km except four Washington nodes, while neighbouring "
    "depths are resolved (tomography notes: probably the vertical checker sign changes). If the masks change, rerun "
    "every step (run_all.sh).",
    "The Blanco low-Vp/low-Vs association at the events is not certified by the 40-100 km resolution tests. Its Vp "
    "part follows the v5.1 depths (it nearly vanishes with the v4 hypocentres), so a trade-off with the v5.1 Blanco "
    "depths (+2.9 km median against Lange 2026; tensor events a median 13.7 km deeper than ComCat, review P3) is "
    "possible. Its Vs part persists with the v4 hypocentres and with the re-timed picks (v0.4.0's Blanco tile), and "
    "in value with stronger smoothing (whose intervals include zero). Counting each occupied node once leaves "
    "little of either part, and the event-weighted interval depends on the block-grid placement. A 20 km "
    "checkerboard or spike test and a synthetic with a systematic depth offset, on v0.4.x and catalog v6.0, would "
    "settle it.",
    "The Washington background-event b-value split by Vp ({wa_db} {wa_ci}, q {wa_q} in the primary family) passes "
    "the rule but is one of {n_b} b-value splits and is not matched by the Vp-change split. Recheck on catalog v6.0 "
    "and v0.4.x before any reading.",
    "Vp/Vs at the Blanco (1.62-1.88 at resolved nodes) is close to the Kuna (2019) starting value of 1.73 in the "
    "mantle. Vs is reverted at most S-poor nodes (D-B051), so the Vp/Vs signal is weak by construction.",
    "Family labels at the Blanco: a whole-period time shuffle reproduces 39% of the links but only 12.8 of 70 "
    "swarms (clusters piece null test); Q1 inherits that uncertainty for the mainshock-aftershock and multiplet "
    "labels.",
    "Magnitudes: the phase19 ML inversion is not converged (mag piece); production M is unchanged pending a PI "
    "decision. Converged, offshore ML moves by region (Blanco -0.68), which would change which events are above the "
    "box Mc in the Blanco b-value and segment statistics.",
    "Waveform review P1-P4 is pending human review. None of the P1 deep or P2 Brookings events is in a resolved "
    "cell, but they set the deepest Q3 columns; 30 far_only events (29 Blanco) and 10 tensor events are in resolved "
    "cells.",
    "No ensemble posterior (D-B030/D-B032) exists for v0.3.1, so the velocity uncertainty at a node is not in the "
    "intervals; the block bootstrap covers spatial sampling only.",
    "All v5.1 products will be redone on catalog v6.0 (D-B079, NonLinLoc with SSST station terms), with model v0.4.0 "
    "(adopted, D-B082) or v0.4.1 (D-B083, being built).",
]

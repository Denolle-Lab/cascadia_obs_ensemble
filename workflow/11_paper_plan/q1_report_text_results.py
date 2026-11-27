"""Result-dependent sections of the Q1 report. Numbers are read from the tables at build time; the wording was
written after the results were seen and checked against the pre-registered decision rules."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _p(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(v):
        return "-"
    return f"{v:.4f}" if v >= 0.0001 else f"{v:.1e}"


def _r(v, nd=2):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "-"
    return "-" if not np.isfinite(v) else f"{v:+.{nd}f}"


def _row(T, test):
    return T.set_index("test").loc[test]


def _sec(T, unit, metric, cov):
    k = T[(T.unit == unit) & (T.metric == metric) & (T.covariate == cov)]
    return k.iloc[0] if len(k) else None


def TXT_VERDICT(d, h):
    dec = d["dec"]
    T = d["T"]
    t1, t2, t4a, t4b, t3 = (_row(T, k) for k in ("T1", "T2", "T4a", "T4b", "T3"))
    hc = dec["hot_cold"].get("mcu_recluster_n100", {})
    nr = dec.get("null_rule") or {}
    lines = [
        f"<b>Thermal control (T1, T2, T4): {h['esc'](dec['verdict_thermal'])}.</b> "
        f"Plate age: cell rho {_r(t1.rho)} (n = {int(t1.n)}, one-sided p {_p(t1.p)}, Holm {_p(t1.p_adj)}, block 95 % "
        f"interval {_r(t1.block_lo)} to {_r(t1.block_hi)}); within boundary types {_r(t4a.rho)} (Holm {_p(t4a.p_adj)}). "
        f"Heat flow: rho {_r(t2.rho)} (n = {int(t2.n)}, Holm {_p(t2.p_adj)}, block {_r(t2.block_lo)} to "
        f"{_r(t2.block_hi)}); within types {_r(t4b.rho)} (Holm {_p(t4b.p_adj)}).",
        f"Time-shuffle null rule: {nr.get('n_above', '-')} of 4 hot segments above the null 95th percentile "
        f"({'passed' if nr.get('passed') else 'failed'}). Uniform-Mc reclustering (Mc 2.15, segments with >= 100 "
        f"events): hot median {h['fmt'](hc.get('hot_median'), 1)} %, cold median {h['fmt'](hc.get('cold_median'), 1)} %.",
        f"<b>Fluid signal (T3, in-situ Vp/Vs): {h['esc'](dec['verdict_vpvs'])}.</b> "
        f"{int(t3.get('n_swarm', 0) or 0)} swarm and {int(t3.get('n_ma', 0) or 0)} mainshock-aftershock families "
        f"carry a valid Vp/Vs; the test needs 20 of each.",
    ]
    return '<div class="verdict">' + "<br>".join(lines) + "</div>"


def TXT_CHANGES(d, h):
    info = d["info"]["main"]
    return [
        "Segment membership (an input, not a test choice): the D-B113 ridge axes (analysis/plate-boundaries ded23ee2, "
        "plate_axes_v1.csv, md5 " + str(info.get("axes_file_md5", "-"))[:8] + ") replaced the PB2002 traces at "
        "13:23-13:28, around the registration. Every number here uses the D-B113 segments: 1,046 events moved from the "
        "Gorda plate to the Gorda Ridge and 228 from the Juan de Fuca plate to its ridge (event-set counts, pb tables). "
        "The cell tests do not use segments except T4 (cell type) and the hot/cold contrast.",
        "Covariate source: the D-B114 per-event table (events_external_geo_v61.parquet, workflow/12_external_geo, "
        "branch analysis/external-geo) appeared at 13:36, after the registration. As registered, the decision uses it; "
        "the interim sampler (q1_covariates.py) is reported alongside. Column choices: age_myr; heat flow = hf_smooth "
        "(Gaussian-weighted median of GHFDB sites, sigma 15 km, at least 3 effective sites; none at Blanco); 600 C "
        "isotherm = iso600_plate_bsf (GDH1 plate model, ages floored at 0.1 Myr); spreading rate = full_rate_mmyr "
        "(the Seton et al. 2020 rate at which the crust formed, not present-day plate motion; MORVEL was not "
        "available); roughness = rough20 (20.5 km window, plane removed) where the window holds no land; sediment = "
        "sed_thk_m. The slab-surface temperature Tslab_D80 was added to the exploratory family; the registration "
        "named a slab-model isotherm, which the table does not provide.",
        "The external-data job printed its own exploratory segment correlations (for example swarm share against plate "
        "age, rho -0.77, on the pre-D-B113 segments). They were seen after this registration, were not used, and "
        "nothing here was adapted to them.",
        "Vp/Vs: the registration named an orthogonal-regression variant without saying how outliers are handled; it is "
        "computed on the points within 5 MAD of the L1 line. The time-shuffle null stores the swarm members of each "
        "shuffled catalog, so the per-unit null shares follow the current segment join (implementation only).",
    ]


def TXT_DATA(d, h):
    info = d["info"]["main"]
    V = d["V"]
    S = d["S"]
    cells = d["cells"]
    rows = [
        ("Catalog v6.1", "cascadia_catalog_v6.1.csv, md5 " + info.get("catalog_md5", "")[:8],
         f"{info['n_event_set']:,} events in the event set (tiers A-C, earthquakes, above the regional Mc)"),
        ("Clustering v6.1", "v51_clusters_zbz.py --catalog v6.1 (Zaliapin and Ben-Zion nearest-neighbour; nine regions)",
         "families typed swarm / mainshock-aftershock / multiplet; 12 sensitivity variants"),
        ("Mc maps v6.1", "magnitude_converged/mc_maps_v61 (non-earthquakes out)", "mc_best per 0.1 deg node (controls)"),
        ("Segments", "pb_common.assign_segments, D-B113 axes (recomputed in q1_metrics.py)",
         f"{len(S)} segments with >= 100 events"),
        ("Cells", "0.2 x 0.2 deg from 132 W, 38 N", f"{len(cells)} cells with >= 50 events"),
        ("External covariates", "D-B114 events_external_geo_v61.parquet; PROVENANCE.csv",
         "Seton et al. 2020 age and full rate (doi 10.1029/2020GC009214); GHFDB R2024 (10.5880/fidgeo.2024.014); "
         "GlobSed v3 (10.1029/2018GC008115); GEBCO 2026 + GMRT"),
        ("CC differential times", "tomo_v1/cc_dt_all (v5.1 families, D-B093)",
         f"{len(V):,} v5.1 families; {int(V.carried.sum()):,} carry over to one v6.1 family (>= 50 % of members)"),
    ]
    out = h["table"](pd.DataFrame(rows, columns=["input", "source", "use"]), left=("input", "source", "use"))
    return out


HOT = ["AX", "JR", "EN", "EXR", "GR", "BL", "SOV", "MSH"]
COLD = ["GP", "MFZ", "MTJ", "GS"]
RUNS = (["main", "v60", "mcu_count", "mcu_recluster", "mc_local", "ci_window"]
        + ["var_" + v for v in ["d3", "d3q", "df1.3", "df2.0", "df_est", "eta0_shuf5", "mc+0.3", "mc_best", "mc_year",
                                "noM_floor", "storedM", "tierABC"]])


def _hot_cold(Q):
    rows = []
    for run in RUNS:
        f = Q.BIG / "tables" / f"metrics_{run}.csv"
        if not f.exists():
            continue
        x = pd.read_csv(f)
        x = x[(x.unit == "seg") & (x.n >= 100)].set_index("unit_id").swarm_share
        hot, cold = x.reindex(HOT).dropna(), x.reindex(COLD).dropna()
        rows.append(dict(run=run, n_hot=len(hot), hot_median=hot.median(), hot_min=hot.min(), n_cold=len(cold),
                         cold_median=cold.median(), cold_max=cold.max(), separated=bool(hot.min() > cold.max())))
    return pd.DataFrame(rows)


def TXT_RESULTS(d, h):
    Q, table, figure, FIG, esc = h["Q"], h["table"], h["figure"], h["FIG"], h["esc"]
    T, S, N, V, cells = d["T"], d["S"], d["N"], d["V"], d["cells"]
    t1, t2, t4a, t4b = (_row(T, k) for k in ("T1", "T2", "T4a", "T4b"))
    nt = N[(N.null == "tshuf") & (N.unit == "seg")].set_index("unit_id")
    nm = N[(N.null == "mshuf") & (N.unit == "seg")].set_index("unit_id")
    s = S.copy()
    s["null_t95"] = s.unit_id.map(nt.null_p95)
    s["null_m_mean"] = s.unit_id.map(nm.null_mean)
    s["mig"] = [f"{int(a)}/{int(b)}" if b > 0 else "-" for a, b in zip(s.swarm10_migrating.fillna(0), s.n_swarm10.fillna(0))]
    s["group"] = ["hot" if u in HOT else "cold" if u in COLD else "" for u in s.unit_id]
    order = [x for x in Q.P.SEG_ORDER if x in set(s.unit_id)]
    s = s.set_index("unit_id").loc[order].reset_index()
    a = cells[cells.age_myr.notna()]
    out = []
    A = out.append
    A("<h3>4.1 Segments</h3>")
    A("<p>Swarm share above Mc separates two groups of segments. The ridges, the Blanco transform, Mount St Helens and the "
      "other Cascade arc events put 15-29 % of their earthquakes in swarms; the Gorda plate, Mendocino fault zone, "
      "triple junction, Gorda slab, the Juan de Fuca slab and the forearc put 0.7-4.7 %. The lowest segment of the "
      "hot group (Juan de Fuca ridge, 17.2 %) is six times the highest of the cold group (Gorda slab, 2.8 %). This "
      "contrast generated the hypothesis, so it is described here, not tested. b ranges from 0.74 (Juan de Fuca "
      "intraslab) and 0.80 (Axial) to 1.41 (Gorda Ridge) and 1.45 (Puget); it does not split along the hot and cold "
      "groups (Axial is among the lowest b, the Gorda Ridge among the highest). Productivity (direct offspring of "
      "events 1-2 units above Mc) is highest at Blanco (1.5) and Puget (1.4) and lowest at the Mendocino fault zone "
      "(0.25) and the Juan de Fuca slab (0.13): in swarm settings the chain-like families give each event offspring, so "
      "this measure does not isolate aftershock productivity.</p>")
    A(figure(FIG / "fig_q1_segments.png",
             "Swarm share, b and productivity per segment (D-B113 segments, >= 100 events). Bars: 95 % bootstrap "
             "intervals (families resampled as whole units for the swarm share; events for b; parent events for "
             "productivity). Black ticks: 95th percentile of the swarm share in 100 time-shuffled catalogs. Crosses: "
             "mean swarm share when magnitudes are shuffled within each clustering region and the families are "
             "retyped. Colours: batlowS by segment type with one marker per type."))
    cols = ["unit_id", "group", "seg_type_major", "n", "swarm_share", "swarm_share_lo", "swarm_share_hi", "bg_frac",
            "fam_mix", "b", "b_lo", "b_hi", "productivity", "swarm_dur_med_days", "mig", "age_myr", "heatflow_mw",
            "null_t95", "null_m_mean"]
    names = ["seg", "group", "type", "events", "swarm %", "lo", "hi", "background %", "swarm fam. share", "b", "lo", "hi",
             "productivity", "swarm duration (d)", "migrating / swarms n>=10", "age (Myr)", "heat flow", "time-null p95",
             "mag-null mean"]
    A(table(s, cols, names, nd={"swarm_share": 1, "swarm_share_lo": 1, "swarm_share_hi": 1, "bg_frac": 1,
                                "null_t95": 1, "null_m_mean": 1, "swarm_dur_med_days": 2, "age_myr": 2,
                                "heatflow_mw": 0}))
    A('<p class="muted">Swarm duration: median over swarm families with >= 5 events. Age and heat flow: median of the '
      "D-B114 values over the segment's events (empty when fewer than half carry a value).</p>")

    A("<h3>4.2 Spatial cells and the primary tests</h3>")
    A(f"<p>{len(cells)} cells of 0.2 degree hold at least 50 events. Plate age exists in {int(t1.n)} of them (oceanic "
      f"crust in the age grid: ridges, Blanco, the Mendocino fault zone, the Gorda plate and two triple-junction cells), "
      f"heat flow in {int(t2.n)} (mostly onshore and around the triple junction; the smoothed field is empty at Blanco "
      f"and at Axial).</p>")
    A(figure(FIG / "fig_q1_map.png", "Cells with >= 50 events: swarm share, plate age and heat flow (D-B114 values, "
             "median over the cell's events). Open squares: no value. Crameri batlow and lajolla."))
    A(f"<p><b>T1, plate age.</b> rho = {_r(t1.rho)} over {int(t1.n)} cells, one-sided p = {_p(t1.p)}, Holm-adjusted "
      f"{_p(t1.p_adj)}: significant with the predicted sign. The 1 x 1 degree block bootstrap gives a 95 % interval "
      f"of {_r(t1.block_lo)} to {_r(t1.block_hi)}, which includes 0. The {int(t1.n)} cells fall in "
      f"{a.block.nunique()} blocks, and the signal comes from one contrast. Of the {int((a.age_myr > 2).sum())} cells "
      f"older than 2 Myr, one is a western Blanco cell (2.9 Myr, swarm share 59 %); the other "
      f"{int((a.age_myr > 2).sum()) - 1} (southern Gorda plate, Mendocino fault zone, triple junction) have no swarm "
      f"events at all. The cells younger than 2 Myr (ridges and Blanco) have a median swarm share of "
      f"{a[a.age_myr < 2].swarm_share.median():.1f} %.</p>")
    bl = a[a.seg_major == "BL"]
    wt = json.loads(t4a.within_type) if isinstance(t4a.within_type, str) else {}
    wt_txt = "; ".join(f"{k} {('%+.2f' % v[0]) if v[0] is not None else 'constant'} ({v[1]} cells)" for k, v in wt.items())
    A(f"<p><b>T4a, plate age within boundary types.</b> Weighted within-type rho = {_r(t4a.rho)}, one-sided p = "
      f"{_p(t4a.p)} (Holm {_p(t4a.p_adj)}). Per type: {esc(wt_txt)}. The 31 transform cells are 25 Blanco and 6 Mendocino fault zone cells. Looked "
      f"at after the test (not registered): the {len(bl)} Blanco cells alone ({bl.age_myr.min():.1f}-"
      f"{bl.age_myr.max():.1f} Myr, swarm share {bl.swarm_share.min():.0f}-{bl.swarm_share.max():.0f} %) give rho "
      f"{spearmanr(bl.age_myr, bl.swarm_share)[0]:+.2f}, the opposite sign to the prediction. The quasi-binomial "
      f"model with type terms "
      f"gives {float(t4a.glm_coef_per_sd):+.2f} +- {float(t4a.glm_se):.2f} on the logit scale per standard deviation of "
      f"age (dispersion {float(t4a.glm_dispersion):.0f}).</p>")
    wt = json.loads(t4b.within_type) if isinstance(t4b.within_type, str) else {}
    wt_txt = "; ".join(f"{k} {('%+.2f' % v[0]) if v[0] is not None else 'constant'} ({v[1]} cells)" for k, v in wt.items())
    A(f"<p><b>T2 and T4b, heat flow.</b> rho = {_r(t2.rho)} over {int(t2.n)} cells, one-sided p = {_p(t2.p)}, Holm "
      f"{_p(t2.p_adj)}; block interval {_r(t2.block_lo)} to {_r(t2.block_hi)}. Within types {_r(t4b.rho)} (Holm "
      f"{_p(t4b.p_adj)}; {esc(wt_txt)}); the type-controlled model does not converge (the subduction and crustal "
      f"cells have no swarms, a separation), so it gives no effect size. With the interim heat-flow sampler (median "
      f"of GHFDB sites within 25 km) T2 is {_r(_row(d['Ti'], 'T2').rho) if d['Ti'] is not None else '-'}: the heat-flow "
      f"result depends on how sparse site values are spread to the events"
      + (f" (with the interim values the heat-flow sign holds in "
         f"{d['dec_int']['d1_detail']['T2']['controls_sign_ok']} of 22 control runs, against "
         f"{d['dec']['d1_detail']['T2']['controls_sign_ok']} with the D-B114 field)" if d.get("dec_int") else "")
      + ".</p>")
    A(figure(FIG / "fig_q1_scatter.png", "Cell swarm share against plate age and heat flow (primary covariates) and "
             "against roughness and sediment thickness (exploratory). rho and p: Spearman and its permutation p "
             "(one-sided where a sign was predicted)."))
    prim = T[T.family == "primary"].copy()
    prim["holm_sig"] = prim.significant
    A(table(prim, ["test", "covariate", "rho", "n", "p", "p_adj", "block_lo", "block_hi", "holm_sig", "sign_ok"],
            ["test", "covariate", "statistic", "n", "p (one-sided)", "Holm p", "block lo", "block hi",
             "significant (Holm)", "predicted sign"], nd={"rho": 3, "p": 4, "p_adj": 4}))
    A('<p class="muted">T3 statistic: difference of median Vp/Vs, swarm minus mainshock-aftershock families; empty '
      "because the test needs 20 families of each type.</p>")
    A("<h3>4.3 Secondary and exploratory tests</h3>")
    sec = T[T.family != "primary"].copy()
    A(table(sec, ["family", "unit", "metric", "covariate", "pred_sign", "rho", "n", "p", "p_adj", "significant"],
            ["family", "unit", "metric", "covariate", "predicted sign", "rho", "n", "p", "BH p", "significant (BH)"],
            nd={"rho": 3, "p": 4, "p_adj": 4}))
    A("<p>None of the secondary tests passes the Benjamini-Hochberg threshold. The two closest are b against age "
      "(cells, rho -0.35, BH 0.08, the predicted sign) and the segment swarm share against age (rho -0.79 over 8 "
      "oceanic segments, BH 0.08). Productivity against age has the opposite sign to the prediction (rho -0.61, 12 "
      "cells). Among the exploratory covariates, the swarm share falls with sediment thickness (cells, rho -0.66) and "
      "rises with roughness (rho +0.42). Both covariates follow the same geography as age (thin sediment and rough, "
      "young seafloor at the ridges and Blanco; thick sediment on the Gorda plate and near the margin), so they "
      "describe the same contrast and were not predicted. The 600 C isotherm depth has the ranks of plate age "
      "(rho -0.44, as registered it is not a separate test). The slab-surface temperature correlates with the "
      "segment swarm share (rho +0.83 over 8 segments) but not over cells (+0.19): the hot-slab segments are the "
      "arc segments far from the trench, so this is the arc-forearc contrast again.</p>")

    A("<h3>4.4 In-situ Vp/Vs of earthquake families</h3>")
    vv = V[V.valid & V.carried]
    ax = V[(V.block == "axial") & V.vpvs.notna()]
    A(f"<p>Of the {len(V):,} v5.1 families, {int(V.carried.sum()):,} carry over to one v6.1 family (that family "
      f"holds >= 50 % of their members). With the registered rules (CC-accepted P and S at >= 3 common stations per "
      f"pair, pairs within 5 km, >= 5 events and >= 30 points per family), {int(V.vpvs.notna().sum())} families give "
      f"an estimate and {int(V.valid.sum())} are valid (95 % half-width &lt;= 0.10, value within 1.4-2.5): "
      f"{int((vv.type61 == 'swarm').sum())} swarm, {int((vv.type61 == 'mainshock-aftershock').sum())} "
      f"mainshock-aftershock and {int((vv.type61 == 'multiplet').sum())} multiplet families. T3 is therefore not "
      f"testable. None of the valid families is on a ridge; 8 are in the Mendocino block.</p>")
    A("<p>Two measurement limits explain the low yield. First, regression dilution: within a family the demeaned P "
      "differential times spread over only about 10 ms (median interquartile range), close to the CC noise (median "
      "residual MAD 5 ms). The L1 slope then falls toward zero as noise grows: the median estimate rises from 0.30 in "
      "the lowest fifth of signal-to-noise to 1.67 in the highest fifth, where the values (1.6-1.8) are physically "
      "plausible. Second, at Axial the S-P time is about 0.5 s, so the S window (-0.3 to +1.0 s around S) contains P "
      f"energy; the {len(ax)} Axial estimates have a median of {ax.vpvs.median():.2f}, below any rock value, with "
      "half-widths near 0.2. Catalog picks (the registered fallback) are worse: their noise exceeds the spread, the "
      f"median estimate is {V.vpvs_cat.median():.2f} and {int(V.valid_cat.sum())} families are valid. The 11-25 "
      "feasibility count (621 families) counted catalog picks only and did not check precision.</p>")
    A(f"<p>For the record, not as a test: the valid swarm families have a median Vp/Vs of "
      f"{vv[vv.type61 == 'swarm'].vpvs.median():.2f} (n = {int((vv.type61 == 'swarm').sum())}) and the "
      f"mainshock-aftershock families {vv[vv.type61 == 'mainshock-aftershock'].vpvs.median():.2f} (n = "
      f"{int((vv.type61 == 'mainshock-aftershock').sum())}), the opposite order to the hypothesis, on too few "
      "families to mean anything.</p>")
    A(figure(FIG / "fig_q1_vpvs.png", "Left: family Vp/Vs (L1 slope of demeaned S on P CC differential times) against "
             "signal-to-noise; grey band: the registered valid range; crosses: Axial, where S-P is about 0.5 s. Right: "
             "valid families by v6.1 type, with 95 % bootstrap intervals; markers and colours by segment type."))
    return "\n".join(out)


def TXT_CONTROLS(d, h):
    Q, table, figure, FIG = h["Q"], h["table"], h["figure"], h["FIG"]
    C, N, dec = d["C"], d["N"], d["dec"]
    out = []
    A = out.append
    det = dec["d1_detail"]
    A(f"<p><b>Sign stability.</b> The cell rho of swarm share against plate age is negative in "
      f"{det['T1']['controls_sign_ok']} of {det['T1']['controls_n']} control runs; against heat flow it is positive "
      f"in {det['T2']['controls_sign_ok']} of {det['T2']['controls_n']} (v6.0 with the interim sampler is the "
      "exception). The size is less stable than the sign: with plate age, rho ranges from -0.06 (cells with >= 100 "
      "events, 27 cells) and -0.10 (clustering at Mc + 0.3) to -0.50 (cells with >= 30 events), and the runs that thin "
      "the cells (uniform Mc, local Mc, Cascadia Initiative window, 0.4 degree cells) give -0.25 to -0.34 with "
      "one-sided p of 0.04-0.06.</p>")
    A(figure(FIG / "fig_q1_controls.png", "Spearman rho of cell swarm share with plate age and heat flow in the main "
             "run (black, with the 1 x 1 degree block-bootstrap 95 % interval) and in the 22 control runs (blue: "
             "predicted sign; orange: opposite sign)."))
    c = C.pivot(index="run", columns="covariate", values=["rho", "p_one", "n"])
    c.columns = [f"{a}_{b}" for a, b in c.columns]
    c = c.reindex(list(dict.fromkeys(C.run))).reset_index()
    for k in ("n_age_myr", "n_heatflow_mw"):
        c[k] = c[k].astype(int)
    A(table(c, ["run", "rho_age_myr", "p_one_age_myr", "n_age_myr", "rho_heatflow_mw", "p_one_heatflow_mw",
                "n_heatflow_mw"], ["control run", "rho age", "p", "cells", "rho heat flow", "p", "cells"],
            nd={"rho_age_myr": 3, "p_one_age_myr": 4, "rho_heatflow_mw": 3, "p_one_heatflow_mw": 4}))
    A('<p class="muted">mcu_count: only M >= 2.15 events counted, main families. mcu_recluster: clustering redone '
      "with Mc 2.15 in every region. mc_local: events at or above the Mc map value of their 0.1 degree node. var_*: "
      "the 12 v6.1 clustering variants. offshore: cells with most events offshore. partial: partial rank "
      "correlation controlling for the Mc map value and the distance to the nearest station. ci_window: 2011-10 to "
      "2015-09 only. cell04: 0.4 degree cells. N30 / N100: other minimum event counts. v60: catalog v6.0, its "
      "clustering and the same segment rule (interim covariates on the v6.0 positions).</p>")
    hc = _hot_cold(Q)
    mr = d["m"]["mcu_recluster"]
    n_ax = int(mr[(mr.unit == "seg") & (mr.unit_id == "AX")].n.iloc[0])
    A("<p><b>Completeness, clustering parameters and catalog version (hot against cold segments).</b> In every run the "
      "lowest hot segment stays above the highest cold segment. With the clustering redone at a uniform Mc of 2.15 "
      f"(Axial keeps {n_ax} tier A-C events above it), "
      "the hot median is 23 % and the cold median 1.6 %, so the falsification rule on completeness does not apply. "
      "The segment swarm shares of v6.0 and v6.1 have a rank correlation of 0.96; Mount St Helens moves from 38 % "
      "to 28 % (family 144 retyped, D-B111 family instability).</p>")
    A(table(hc, ["run", "n_hot", "hot_median", "hot_min", "n_cold", "cold_median", "cold_max", "separated"],
            ["run", "hot segments", "hot median %", "hot min %", "cold segments", "cold median %", "cold max %",
             "min hot > max cold"], nd={"hot_median": 1, "hot_min": 1, "cold_median": 1, "cold_max": 1}))
    nt = N[(N.null == "tshuf") & (N.unit == "seg")].copy()
    nm = N[(N.null == "mshuf") & (N.unit == "seg")].set_index("unit_id")
    nt["mshuf_mean"] = nt.unit_id.map(nm.null_mean)
    nt["mshuf_p05"] = nt.unit_id.map(nm.null_p05)
    nt["below_mshuf_p05"] = nt.unit_id.map(nm.below_p05)
    order = [x for x in Q.P.SEG_ORDER if x in set(nt.unit_id)]
    nt = nt.set_index("unit_id").loc[order].reset_index()
    nr = dec.get("null_rule") or {}
    A(f"<p><b>Shuffled-catalog nulls.</b> Time shuffle (100 catalogs, event times permuted within each clustering "
      f"region, real thresholds): Axial, Blanco, the Gorda Ridge and Mount St Helens all lie above the 95th "
      f"percentile of their null swarm share, so the null rule passes ({nr.get('n_above', '-')} of 4). Dense "
      f"seismicity alone makes swarms at Axial (null mean 14.7 %, observed 29.2 %); elsewhere the null share is "
      f"0-3.5 %. Magnitude shuffle (200 draws, family membership and trees kept): in the cold segments the observed "
      f"swarm share is far below the magnitude-shuffle range (Gorda plate 0.7 % against a null mean of 56 %; "
      f"Mendocino fault zone 1.2 % against 12 %; triple junction 1.0 % against 13 %; Gorda slab 2.8 % against 12 %), "
      f"so their families carry a dominant mainshock far more often than random magnitudes would give. At Axial, "
      f"Mount St Helens, the Juan de Fuca ridge and the Cascade arc the observed share matches the magnitude-shuffle "
      f"mean: their families look like random draws in magnitude. Blanco and the Gorda Ridge sit in between "
      f"(below the 5th percentile, but at 25 % and 20 %).</p>")
    A(table(nt, ["unit_id", "n", "swarm_share", "null_mean", "null_p95", "above_p95", "mshuf_mean", "mshuf_p05",
                 "below_mshuf_p05"],
            ["seg", "events", "observed %", "time-null mean %", "time-null p95 %", "above p95",
             "mag-null mean %", "mag-null p05 %", "below mag-null p05"],
            nd={"swarm_share": 1, "null_mean": 1, "null_p95": 1, "mshuf_mean": 1, "mshuf_p05": 1}))
    A("<p><b>Network geometry and detection.</b> Every plate-age cell is offshore, so the offshore-only run equals the "
      "main run for T1. The partial correlation that removes the Mc map value and the nearest-station distance keeps "
      "rho at -0.37 (p 0.009). The Cascadia Initiative window (2011-10 to 2015-09) keeps the sign at -0.26 on 39 "
      "cells. Axial has the lowest Mc (0.35-0.65) because of the cabled array; under the uniform Mc of 2.15 its swarm "
      "share rises to 76 %, so its high share is not a product of its low Mc.</p>")
    return "\n".join(out)


def TXT_CLAIMS(d, h):
    dec = d["dec"]
    return (
        f'<div class="verdict"><b>Pre-registered outcome.</b> Thermal control: {dec["verdict_thermal"]} (T1 is '
        "significant with the predicted sign, the null rule passes and the sign holds in all 22 control runs, but "
        "the block-bootstrap interval of T1 includes 0, so D1 fails; T4 shows no age effect within boundary types). "
        "Not falsified: T1 and T2 both have the predicted sign, the hot group stays above the cold group at a uniform "
        f"Mc, and the null rule passes. Fluid signal: {dec['verdict_vpvs']}.</div>"
        "<h3>The paper can claim</h3><ul>"
        "<li>Slip style splits by boundary setting: 17-29 % of earthquakes above Mc are in swarms on the ridges, the "
        "Blanco transform and at Mount St Helens, against 0.7-2.8 % on the Gorda plate, the Mendocino fault zone, the "
        "triple junction and the Gorda slab. The split survives a uniform Mc of 2.15 (reclustered: hot median 23 %, "
        "cold median 1.6 %), all 12 clustering variants, catalog v6.0, and a time-shuffle null.</li>"
        "<li>In the cold settings, families carry a dominant mainshock far more often than random magnitude ordering "
        "would give (swarm share below the 5th percentile of a magnitude-shuffle null), while at Axial, Mount St "
        "Helens and the Juan de Fuca ridge the families are indistinguishable from random magnitude ordering.</li>"
        "<li>Across 45 oceanic 0.2 degree cells, the swarm share falls with plate age (Spearman -0.43, Holm-adjusted "
        "p 0.008); of the cells older than 2 Myr only one western Blanco cell has swarms. Stated as a correlation "
        "that tracks the boundary settings, with its block interval.</li>"
        "</ul><h3>The paper cannot claim</h3><ul>"
        "<li>That plate age or temperature sets the slip style beyond the boundary type: within types the age effect "
        "is nil (weighted rho +0.03), and the cell correlation rests on 14 one-degree blocks (block interval -0.78 to "
        "+0.29). Young lithosphere and the swarm-prone settings coincide here, so these data cannot separate the "
        "two.</li>"
        "<li>Any heat-flow control: T2 is not significant, measured heat flow is missing at Blanco and Axial, and the "
        "sign depends on how site values are spread.</li>"
        "<li>A fluid (Vp/Vs) control on swarms: T3 is not testable (7 swarm and 18 mainshock-aftershock families with "
        "a valid Vp/Vs).</li>"
        "<li>That sediment thickness or roughness control the style: those correlations are exploratory and follow "
        "the same geography as age.</li>"
        "<li>Anything about individual families other than large sequences (D-B111).</li></ul>")


def TXT_BULLETS(d, h):
    return [
        "[TO WRITE] Slip style by setting (v6.1, tier A-C earthquakes above the regional Mc, D-B113 segments): swarm "
        "share 29 % Axial, 28 % Mount St Helens, 25 % Blanco, 20 % Gorda Ridge, 17 % Juan de Fuca ridge; 0.7 % Gorda "
        "plate, 1.2 % Mendocino fault zone, 1.0 % triple junction, 2.8 % Gorda slab.",
        "[TO WRITE] Robustness: the split holds at a uniform Mc of 2.15 after reclustering (hot median 23 %, cold "
        "median 1.6 %), in all 12 clustering variants, in catalog v6.0 (segment rank correlation 0.96), and above a "
        "time-shuffle null (100 catalogs) at Axial, Blanco, the Gorda Ridge and Mount St Helens.",
        "[TO WRITE] Mainshock dominance: on the Gorda plate, Mendocino fault zone, triple junction and Gorda slab the "
        "swarm share is below the 5th percentile of a magnitude-shuffle null (Gorda plate 0.7 % against 56 %); at "
        "Axial and Mount St Helens it equals the null mean.",
        "[TO WRITE] Plate age (Seton et al. 2020): across 45 oceanic 0.2 degree cells the swarm share falls with age "
        "(Spearman -0.43, Holm p 0.008, pre-registered), same sign in 22 of 22 controls; block-bootstrap interval "
        "-0.78 to +0.29 and no age effect within boundary types: age tracks the setting contrast, it does not explain "
        "it.",
        "[TO WRITE] Heat flow (GHFDB R2024, smoothed): no significant relation (rho +0.30, Holm p 0.24); no measured "
        "heat flow on the Blanco transform.",
        "[TO WRITE] In-situ Vp/Vs of families from CC differential times (Lin and Shearer 2007): 27 of 2,097 families "
        "give a valid value; too few to compare swarms with mainshock-aftershock families; limited by the small spread "
        "of differential times and by S-P of 0.5 s at Axial.",
        "[TO WRITE] Methods: hypotheses, tests and decision rules registered before any covariate correlation "
        "(2026-11-27 13:26 PST); Holm over five primary tests; BH for secondary tests.",
        "[TO WRITE] Limitations: see the limitation bullets (family instability, Mc contrast, heat-flow coverage, "
        "Vp/Vs precision, no waveform double-difference catalog).",
    ]


def TXT_LIMITS(d, h):
    return [
        "Families shift between catalog versions (D-B111): the shares here are over many families and change little "
        "between v6.0 and v6.1 (rank 0.96), but Mount St Helens moves from 38 % to 28 % because one family was retyped.",
        "Mc differs between regions (0.65 at Axial to 2.15 on the Gorda Ridge and plate in the clustering; 0.35 at "
        "Axial and Mount St Helens in the segment Mc). The event set is above the regional Mc, and the uniform-Mc runs "
        "keep the contrast, but at Mc 2.15 Mount St Helens and several small segments drop below 100 events.",
        "The swarm and mainshock-aftershock types come from topology and magnitude rules (Zaliapin and Ben-Zion) "
        "applied to families of >= 5 events; families of 2-4 events are multiplets or mainshock-aftershock pairs by the "
        "magnitude gap only. The magnitude-shuffle null shows that random magnitudes alone make many families swarms.",
        "Plate age exists only on oceanic crust in the age grid, so the cell tests cover ridges, Blanco, the Mendocino "
        "fault zone, the Gorda plate and two triple-junction cells; the slab, the forearc and onshore settings enter "
        "only the segment contrast.",
        "Spatial dependence: 45 age cells in 14 one-degree blocks. Neighbouring cells share families and structure, so "
        "the cell count overstates the independent samples; the block interval is the honest uncertainty.",
        "Heat flow: GHFDB sites are sparse offshore, absent on the Blanco transform, and ridge-crest values are "
        "lowered by hydrothermal circulation; the smoothed field and the 25 km median give different results.",
        "Spreading rate is the rate at which the crust formed (Seton et al. 2020 grid), not present-day plate motion; "
        "it varies little here (60-69 mm/yr in the plate-age cells).",
        "Vp/Vs: the CC differential times within families spread over about 10 ms against about 5 ms of noise, so "
        "the slope is biased low except in the best-sampled families; at Axial the S window contains P energy (S-P "
        "about 0.5 s). A useful family Vp/Vs needs a P-free S window and families with a wider spread of station "
        "geometry.",
        "No waveform double-difference catalog: swarm migration and duration are provisional.",
        "Deep events under Mount St Helens are not interpreted (D-B111). The Mount St Helens event set holds 243 "
        "events within 25 km of the summit at any depth; 5 are deeper than 30 km and none of them is in a swarm. Without "
        "the 13 events deeper than 20 km the swarm share is 29.1 % (28.4 % with them).",
    ]


def TXT_FILES(d, h):
    Q = h["Q"]
    return ("<h2>Files</h2><ul>"
            "<li>Code: <code>workflow/11_paper_plan/q1_*.py</code>, driver <code>run_q1.sh</code> (branch "
            "science/q1-slip-style). The segment join is recomputed inside q1_metrics.py; rerun the driver after any "
            "change of the segment rule.</li>"
            f"<li>Tables: <code>{Q.BIG}/tables/</code> (metrics_*.csv per run, tests_external.csv, "
            "tests_interim.csv, controls_*.csv, nulls_units.csv, decision.json); small copies in "
            "<code>data/paper_plan/q1_slip_style/</code>.</li>"
            f"<li>Nulls: <code>{Q.BIG}/nulls/</code>; Vp/Vs: <code>{Q.BIG}/vpvs/vpvs_families.csv</code>; figures: "
            f"<code>{Q.BIG}/figures/</code>.</li>"
            "<li>Pre-registration: <code>data/paper_plan/q1_slip_style/RESUME.md</code> (commit 109915b8).</li></ul>")

#!/usr/bin/env python3
"""HTML page of the plate-boundary feasibility study and figure plan (D-B107):
/wd1/mdenolle_data/paper_plan/plate_boundaries_plan.html (self-contained, figures embedded).

Reads the tables of pb_structure.py, pb_seismicity.py, pb_struct_seis.py, pb_sensitivity.py and the figures of
pb_figures.py. Usage: pb_page.py ["time label from `date`"]
"""
from __future__ import annotations

import base64
import hashlib
import html
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb_common as P  # noqa: E402

PAGE = P.BIG / "plate_boundaries_plan.html"
FIG = P.BIG / "figures"

CSS = """
:root{--bg:#fbfbfa;--ink:#1d1d1b;--muted:#5d5c57;--rule:#d8d7d2;--card:#ffffff;--accent:#0b5cad;
--r:#1f6f3f;--m:#9a6a00;--n:#9b2c2c}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1a19;--ink:#ecebe6;--muted:#b4b3ab;
--rule:#3a3a37;--card:#222220;--accent:#7fb2ec;--r:#7fd19b;--m:#e6c069;--n:#f08a8a}}
:root[data-theme="dark"]{--bg:#1a1a19;--ink:#ecebe6;--muted:#b4b3ab;--rule:#3a3a37;--card:#222220;--accent:#7fb2ec;
--r:#7fd19b;--m:#e6c069;--n:#f08a8a}
body{background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;
max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:1.5em;margin:.4em 0}h2{font-size:1.2em;margin-top:1.8em;border-bottom:1px solid var(--rule)}
h3{font-size:1.02em;margin-top:1.3em}
table{border-collapse:collapse;font-size:12px;margin:.6em 0;display:block;overflow-x:auto;max-width:100%}
th,td{border-bottom:1px solid var(--rule);padding:3px 6px;text-align:right;white-space:nowrap}
th{text-align:center;color:var(--muted);font-weight:600;vertical-align:bottom}
td:first-child,th:first-child,td.l{text-align:left}
img{max-width:100%;background:#fff;border:1px solid var(--rule);border-radius:4px}
.cap{color:var(--muted);font-size:13px;margin:.2em 0 1.2em}
.box{background:var(--card);border:1px solid var(--rule);border-radius:6px;padding:10px 14px;margin:1em 0}
.resolved{color:var(--r);font-weight:600}.marginal{color:var(--m);font-weight:600}
.notresolved{color:var(--n);font-weight:600}
code{font-size:12.5px}li{margin:.3em 0}.small{font-size:13px;color:var(--muted)}
"""


def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


def img(name, cap):
    p = FIG / name
    return f'<img src="data:image/png;base64,{b64(p)}" alt="{html.escape(cap[:90])}"><p class="cap">{cap}</p>'


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt(v, f="{:.2f}"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        return f.format(v)
    return html.escape(str(v))


def table(df, fmts=None, left=()):
    fmts = fmts or {}
    h = ["<table><thead><tr>"] + [f"<th>{html.escape(str(c))}</th>" for c in df.columns] + ["</tr></thead><tbody>"]
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, str) and v.startswith("<"):
                s = v
            else:
                s = fmt(v, fmts.get(c, "{:.2f}"))
            cells.append(f'<td class="l">{s}</td>' if c in left else f"<td>{s}</td>")
        h.append("<tr>" + "".join(cells) + "</tr>")
    h.append("</tbody></table>")
    return "".join(h)


def ci_str(v, lo, hi, f="{:.2f}"):
    if not np.isfinite(v):
        return "–"
    if np.isfinite(lo) and np.isfinite(hi):
        return f"{f.format(v)} [{f.format(lo)}, {f.format(hi)}]"
    return f.format(v)


def cls_html(c):
    k = {"resolved": "resolved", "marginal": "marginal", "not resolved": "notresolved"}[c]
    return f'<span class="{k}">{c}</span>'


MSH_TEXT = ("38 % (107 of 282 events) to 28 % (69 of 243). Without the 131 analog_test_signal events the v6.0 share is "
            "still 38 % (103 of 268): the flags remove 14 analysis events, 4 of them swarm events. Of the 107 v6.0 "
            "swarm events, 60 stay swarm events; 13 drop to tier D, 4 carry the analog flag, 2 are not located, 1 "
            "falls below Mc; 11 move into multiplets, 11 into a mainshock-aftershock family and 5 into the background. "
            "The largest single change is family 144 (2011-01-28, Mmax 3.96): its v6.1 successor 142 no longer meets "
            "the chain criterion and is typed mainshock-aftershock. Families 1434 (2014-11) and 1987 (2015-08) break "
            "into multiplets. The drop is a reclassification of small and medium families, not the analog flags; "
            "this is the family instability D-B111 names.")


def axes_section():
    """D-B113: the bathymetric axes, their evaluation, the events moved and the changed metrics (when built)."""
    mv_f = P.OUT / "axes_moved_by_segment.csv"
    if P.AXES == "pb2002" or not mv_f.exists():
        return ""
    ev = pd.read_csv(P.ROOT / "data/paper_plan/plate_boundaries/plate_axes_v1_evaluation.csv")
    ev = ev[ev.n_events > 0]
    cols = ["section", "lat_min", "lat_max", "n_events", "chosen"] + [c for c in ev.columns if c.startswith(("median_km",
                                                                                                          "within15"))]
    mv = pd.read_csv(mv_f)
    mv = mv[(mv.moved_in > 0) | (mv.moved_out > 0)]
    chg = pd.read_csv(P.OUT / "axes_changed_metrics.csv")
    keep = ["n_events", "b", "pct_bg", "pct_MA", "pct_swarm", "bsf_med", "bsf_p90", "frac_below_z75", "n_surfaces",
            "dip_wmed", "burst_all", "nP_10", "nPS_10", "class"]
    chg = chg[chg.metric.isin(keep)]
    piv = chg.assign(v=[f"{fmt_v(b)} → {fmt_v(a)}" for b, a in zip(chg.before, chg.after)]).pivot(
        index="metric", columns="seg", values="v").reindex(keep)
    piv.index.name = "metric"
    cc = pd.read_csv(P.OUT / "axes_changed_contrasts.csv")
    n_moved = int(mv.moved_out.sum())
    if P.AXES == "v2":
        return v2_axes_html(mv, piv, cc, n_moved)
    return (f"<h3>Plate-boundary axes (D-B113)</h3><p>PB2002 places the southern Gorda Ridge 10-20 km west of its axial "
            "valley and of its seismicity, so the 15 km buffer missed it and its events fell in the Gorda plate group. "
            "The spreading axes and the Mendocino transform are re-picked from the bathymetry (code pb_axes.py): the "
            "D-B068 merged solid surface (/wd1/mdenolle_data/reloc_v5/topo/topo_15s_cons_merged.bin; GMRT multibeam "
            "where available, GMT earth_relief 01m elsewhere), profiles across the PB2002 trace every 2 km, two "
            "candidates per profile (axial valley: deepest point, 1.5 km smoothing, 10 km median filter; axial high: "
            "highest point within 15 km, 4 km smoothing, 18 km median filter); for the Mendocino transform the crest of "
            "the Mendocino Ridge and the base of its north-facing slope. PB2002 is replaced in a section only where its "
            "15 km buffer misses at least 10 % of the nearby tier A-C earthquakes and a bathymetric candidate keeps at "
            "least 10 percentage points more of them within 15 km without a larger median distance. The candidates "
            "come from the bathymetry; the seismicity only chooses among them. Chosen: the axial valley for the southern "
            "Gorda Ridge (40.45-41.6 N) and the axial high for the Juan de Fuca section 46.4-47.7 N; PB2002 elsewhere. "
            "The Mendocino transform keeps PB2002: its seismicity lies 8-12 km north of the trace, but the buffer "
            "already holds 66 % of the nearby events and the base-of-slope pick only 71 %; the rest are Gorda-plate "
            "lineaments north of the transform. Blanco events are assigned by the catalog region, so the Blanco "
            "trace does not enter the rule; Sovanco and the Explorer ridge keep PB2002 (unconstrained locations). "
            f"Buffer unchanged (15 km). {n_moved:,} tier A-C earthquakes change segment.</p>"
            + '<img src="data:image/png;base64,' + b64(FIG / "fig_axes.png") + '" alt="axes"><p class="cap">PB2002 '
            "(cyan) and the chosen bathymetric axes (yellow) on the shaded merged surface, with the tier A-C "
            "earthquakes.</p>" + table(ev[cols], fmts={c: "{:.2f}" for c in cols if c.startswith(("median", "within"))},
                                       left=("section", "chosen"))
            + '<img src="data:image/png;base64,' + b64(FIG / "fig_axes_before_after.png") + '" alt="before after">'
            '<p class="cap">Segments before (PB2002) and after (D-B113 axes), and the events that change segment.</p>'
            "<p>Events per segment before and after:</p>" + table(mv, left=("out_to", "in_from"))
            + "<p>Changed metrics of the affected segments (before → after):</p>" + table(piv.reset_index(),
                                                                                         left=("metric",))
            + "<p>Contrasts whose value or verdict changed (D-B111 rule):</p>"
            + table(cc, fmts={c: "{:+.3f}" for c in cc.columns if c.startswith("D_real")}, left=("var", "contrast")))


def v2_axes_html(mv, piv, cc, n_moved):
    """D-B117: adopted plate boundaries v2; numbers from data/paper_plan/plate_boundaries_v2/ and the comparison
    tables of pb_axes_compare.py / pb_axes_changes.py (previous axes: PB_AXES_OLD, D-B113 axes v1)."""
    import os as _os
    v2 = P.ROOT / "data/paper_plan/plate_boundaries_v2"
    off = pd.read_csv(v2 / "offsets_summary.csv")
    cor = pd.read_csv(v2 / "corners_adopted.csv")
    brk = pd.read_csv(v2 / "breaks_adopted.csv")
    mfz = off[(off.section == "MFZ") & (off.span == "section")].iloc[0]
    tj = cor[cor.name.str.startswith("Mendocino triple junction (PB2002")].iloc[0]
    bl = brk[(brk.section == "BL") & (brk.kind == "step-over")]
    gr = brk[(brk.section == "GR") & (brk.kind == "step-over")]
    blt = " and ".join(f"{r.offset_km:.1f} km at {abs(r.lon):.2f} W" for _, r in bl.iterrows())
    grt = ", ".join(f"{r.offset_km:.1f} km at {r.lat:.2f} N" for _, r in gr.iterrows())
    old = _os.environ.get("PB_AXES_OLD", "gmrt")
    fig_v2 = Path("/wd1/mdenolle_data/paper_plan/plate_boundaries_v2/figures/map_proposal.png")
    img_v2 = ('<img src="data:image/png;base64,' + b64(fig_v2) + '" alt="boundaries v2"><p class="cap">Plate '
              "boundaries v2 (report: /wd1/mdenolle_data/paper_plan/plate_boundaries_v2/report_boundaries.html).</p>"
              if fig_v2.exists() else "")
    return ("<h3>Plate boundaries v2 (D-B116 rules, adopted D-B117)</h3><p>The segment rule uses the adopted plate "
            "boundaries v2 (pbv2_adopt.py, data/paper_plan/plate_boundaries_v2/plate_axes_v2_adopted.csv; rules "
            "pre-registered in RULES.md). Straight segments chosen by BIC are fitted to axis picks of the relocated "
            "seismicity (v6.1 tier A-B, horizontal error ≤ 3 km) and compared with the bathymetric axes. "
            f"Mendocino transform: {mfz.seis_minus_pb2002_km:.1f} km north of PB2002 (95 %: {mfz.ci95_lo:.1f}-"
            f"{mfz.ci95_hi:.1f}), at the base of the north-facing slope of the Mendocino Ridge (seismicity minus "
            f"bathymetric pick {mfz.seis_minus_bathy_km:+.1f} km); it joins the PB2002 triple junction, which is kept, "
            f"by a {tj.connector_km:.1f} km connector (A7). Blanco transform: three segments with two step-overs, "
            f"{blt}. Gorda Ridge on its axial valley, stepping east by {grt}. Juan de Fuca Ridge on its bathymetric "
            "axis (axial high, Axial to Northern Symmetric), Endeavour from the MAPRIDGES v2602 trace (A6). Explorer, "
            "Sovanco and Nootka keep PB2002. Buffer 15 km and every other rule unchanged; events are never moved. "
            f"Against the previous axes ({old}: D-B113 axes v1), {n_moved:,} tier A-C earthquakes change segment.</p>"
            + img_v2
            + '<img src="data:image/png;base64,' + b64(FIG / "fig_axes_before_after.png") + '" alt="before after">'
            f'<p class="cap">Segments with the previous axes ({old}) and with boundaries v2, and the events that change '
            "segment.</p><p>Events per segment before and after:</p>" + table(mv, left=("out_to", "in_from"))
            + "<p>Changed metrics of the affected segments (before → after):</p>"
            + table(piv.reset_index(), left=("metric",))
            + "<p>Contrasts whose value or verdict changed (D-B111 rule):</p>"
            + table(cc, fmts={c: "{:+.3f}" for c in cc.columns if c.startswith("D_real")}, left=("var", "contrast"))
            + headline_html())


def headline_html():
    f = P.OUT / "axes_headline_changes.csv"
    if not f.exists():
        return ""
    h = pd.read_csv(f)
    x = h[h.outside_previous_ci]
    txt = ("none" if x.empty else "; ".join(f"{r.seg} {r.metric} {r.before:.2f} → {r.after:.2f} (previous 95 %: "
                                            f"{r.before_lo:.2f}-{r.before_hi:.2f})" for _, r in x.iterrows()))
    return ("<p>Headline metrics (b, Mc, swarm share, 90th percentile of the depth below the surface) whose new value "
            f"lies outside the previous 95 % bootstrap interval: {txt}.</p>")


def fmt_v(v):
    try:
        f = float(v)
        return f"{f:,.0f}" if abs(f) >= 100 or f == int(f) else f"{f:.2f}"
    except (TypeError, ValueError):
        return "–" if v is None or (isinstance(v, float)) else str(v)


def zone_note():
    f = P.OUT / "null_zone_check.csv"
    if not f.exists():
        return ""
    z = pd.read_csv(f)
    single = int(z.single_zone_new.fillna(False).astype(bool).sum()) if "single_zone_new" in z else 0
    d = z.assign(dd=(z.D0_new - z.D0_old).abs(), ds=(z.own_share_new - z.own_share_old))
    big = d[(d.dd > 0.05) | (d.ds < -0.15)]
    segs = sorted(set(big.seg), key=P.SEG_ORDER.index)
    det = "; ".join(f"{sg}: " + ", ".join(f"{r['var']} {r['contrast']} (Δ truth {r['D0_old']:+.2f} → {r['D0_new']:+.2f}, "
                                          f"own-zone share {r['own_share_old']:.2f} → {r['own_share_new']:.2f})"
                                          for _, r in big[big.seg == sg].iterrows())
                    for sg in segs) or "none"
    return ("<p><b>Null zones and the current axes.</b> The null truth's flattened zones were built on the PB2002 "
            "segments (null_zones.npz, reconstructed exactly); the members are scored on the current node sets. Zones "
            "are nearest-segment regions, so only about half of a contrast's nodes lie in its own zone (median "
            f"{z.own_share_old.median():.2f} with the PB2002 node sets, {z.own_share_new.median():.2f} with the current "
            "ones) and the null truth keeps zone-boundary contrasts (median |Δ truth| "
            f"{z.D0_old.abs().median():.3f} / {z.D0_new.abs().median():.3f} km/s or Vp/Vs units); {single} of "
            f"{int(z.single_zone_new.notna().sum())} current contrasts lie inside one zone (true no-contrast tests). "
            "The rule (recovered − truth) still measures the recovery error of a known truth. Contrasts whose null-truth "
            f"value moved by more than 0.05 with the axis change, or whose own-zone share fell by more than 0.15: {det}."
            "</p>")


def null_section(nt, circ):
    """Short (findings) and long (section 3) text of the null-truth control (pb_null_score.py output)."""
    import json as _json
    nj = Path("/wd1/mdenolle_data/tomo_v1/ensemble_v0.4.2_null/synth/truth/null_truth.json")
    info = _json.loads(nj.read_text()) if nj.exists() else {}
    nm = int(nt.n_null.max()) if len(nt) else 0
    vn_ = {"Vp": "Vp", "Vs": "Vs", "VpVs": "Vp/Vs"}
    cl = nt[nt.claimed]
    drop = nt[nt.survives_DB111 & ~nt.pass_null]
    lst = lambda x: "; ".join(f"{r.seg} {vn_[r['var']]} {r.contrast} {r.D_real:+.3f}" for _, r in x.iterrows()) or "none"  # noqa: E731
    med_ratio = float(np.nanmedian(nt.N95 / nt.S95))
    short = (f"Null-truth control (rule pre-registered 2026-11-27 10:13:48 PST, commit 6d45ad98; {nm} members): "
             f"{len(cl)} of the {int(nt.survives_DB111.sum())} D-B111 survivors are claimed (|Δ| also ≥ N95, the 95th "
             f"percentile of |recovered − truth| for a truth flattened by depth layer near the segments). Dropped by the null "
             f"truth: {lst(drop)}. Claimed: {lst(cl)}.")
    t = nt[["seg", "var", "contrast", "D_real", "sigma_ens", "S95", "D0_truth", "null_mean", "null_sd", "N95", "N95_rec",
            "paired_cal_minus_null_mean", "truth_cal_minus_null", "n_null", "survives_DB111", "pass_null", "claimed"]].copy()
    t["claimed"] = np.where(t.claimed, '<span class="resolved">claimed</span>',
                            np.where(t.survives_DB111, '<span class="marginal">dropped by N95</span>', "–"))
    t = t.rename(columns={"D_real": "Δ real", "sigma_ens": "σ ens", "D0_truth": "Δ null truth",
                          "null_mean": "null spurious mean", "null_sd": "null spurious sd",
                          "N95": "N95 = p95 |rec − truth|", "N95_rec": "p95 |rec| (variant)",
                          "paired_cal_minus_null_mean": "cal − null (paired members)",
                          "truth_cal_minus_null": "cal − null (truths)", "n_null": "members",
                          "survives_DB111": "D-B111", "pass_null": "≥ N95", "claimed": "verdict"})
    tab = table(t, fmts={c: "{:+.3f}" for c in ["Δ real", "Δ null truth", "null spurious mean",
                                                 "cal − null (paired members)", "cal − null (truths)"]} |
                {"σ ens": "{:.3f}", "S95": "{:.3f}", "null spurious sd": "{:.3f}", "N95 = p95 |rec − truth|": "{:.3f}",
                 "p95 |rec| (variant)": "{:.3f}"}, left=("var", "contrast"))
    nseg = ", ".join(f"{r['seg']} {r['nodes']:,}" for r in info.get("nodes_per_segment", []))
    cmp_f = P.OUT / "null_claimed_compare.csv"
    cmp_txt = ""
    if cmp_f.exists():
        cm = pd.read_csv(cmp_f)
        oc = [c for c in cm.columns if c.startswith("claimed_") and not c.endswith(P.AXES)][0]
        old_lab = oc.replace("claimed_", "")
        g_, l_ = cm[cm.change == "gained"], cm[cm.change == "lost"]
        lst2 = lambda x: "; ".join(f"{r.seg} {vn_[r['var']]} {r.contrast}" for _, r in x.iterrows()) or "none"  # noqa: E731
        cmp_txt = (f" Against the same 20 members scored on the previous axes ({old_lab}): "
                   f"{int(cm[oc].fillna(False).astype(bool).sum())} claimed there; gained here: {lst2(g_)}; "
                   f"lost here: {lst2(l_)}.")
        short = short + cmp_txt
    long = (f"<p><b>Null truth (D-B112).</b> Rule pre-registered in RESUME.md at Fri Nov 27 10:13:48 PST 2026 (commit "
            "6d45ad98), before the null truth was built: a contrast is claimed only if it passes the D-B111 rule and "
            "|Δ_real| ≥ N95, the 95th percentile of |Δ_rec,i − Δ_truth| over the null members (≥ 15 members). Null "
            "truth: the calibration truth (v0.4.2 x (1 + f), same random field, hypocentres and noise draws) with every "
            f"rock node within {info.get('zone_km', 45):.0f} km of a segment's faults or seismicity set to its depth-layer "
            f"mean in that segment's zone ({info.get('nodes_changed', 0):,} nodes; {nseg}; median change of Vp "
            f"{info.get('dvp_abs_median', float('nan')):.2f} km/s, p95 {info.get('dvp_abs_p95', float('nan')):.2f}). "
            f"Synthetic data from the frozen tomoDD-SE_syn binaries; {nm} members (s000-s{nm - 1:03d}, the same seeds "
            "as the calibration members) through M0, four tiles and the merge in "
            "/wd1/mdenolle_data/tomo_v1/ensemble_v0.4.2_null. The truth contrast is zero where both node sets lie in "
            f"one zone. N95 is a median {med_ratio:.2f} of S95.</p>" + zone_note()
            + '<img src="data:image/png;base64,' + b64(FIG / "fig_null.png") + '" alt="null truth"><p class="cap">'
            "Real contrasts with ±2σ of the real ensemble and ±N95 of the null-truth members (grey). Filled = claimed. "
            "Draft.</p>" + (f"<p>{cmp_txt.strip()}</p>" if cmp_txt else "") + tab)
    return short, long


def main():
    when = sys.argv[1] if len(sys.argv) > 1 else subprocess.run(["date"], capture_output=True, text=True).stdout.strip()
    git = lambda *a: subprocess.run(["git", "-C", str(P.ROOT), *a], capture_output=True, text=True).stdout.strip()  # noqa
    head, branch = git("rev-parse", "--short", "HEAD"), git("rev-parse", "--abbrev-ref", "HEAD")
    f = pd.read_csv(P.OUT / "structure_feasibility.csv")
    s = pd.read_csv(P.OUT / "seismicity_metrics.csv")
    ct = pd.read_csv(P.OUT / "structure_contrasts.csv")
    dl = pd.read_csv(P.OUT / "depth_vs_level.csv")
    sv = pd.read_csv(P.OUT / "swarm_vs_ma.csv")
    sens = pd.read_csv(P.OUT / "segment_rule_sensitivity.csv")
    circ = pd.read_csv(P.OUT / "circularity_contrasts.csv")
    sec = pd.read_csv(P.OUT / "sections_info.csv", index_col=0)
    sseg = pd.read_csv(P.OUT / "surface_segments.csv")
    m = f.merge(s.drop(columns=["name", "type", "n_events", "n_depth_good", "n_surfaces"]), on="seg")
    n_all = int(s.n_events.sum())
    L = P.CATLABEL
    ev = pd.read_parquet(P.BIG / f"events_segments_{P.TAG}.parquet",
                         columns=["ID", "seg", "abc", "is_eq", "review_P1", "event_flag", "lat", "lon", "depth", "M",
                                  "tier"])
    n_p1 = int((ev.abc & ev.is_eq & ev.review_P1).sum())
    n_analog = int(ev.event_flag.fillna("").str.contains("analog_test_signal").sum())
    n_flag = int((~ev.is_eq).sum())
    gp = s.set_index("seg")
    gp_good, gp_all = int(gp.loc["GP", "n_depth_good"]), int(gp.loc["GP", "n_events"])
    bl_sz = float(gp.loc["BL", "sig_z_med_good"])
    sn = sens.set_index("seg")
    dlg = dl[(dl.seg == "GS") & (dl.level == "z75")].iloc[0]
    dlb = dl[(dl.seg == "BL") & (dl.level == "z75")].iloc[0]
    dlm = dl[(dl.seg == "MFZ") & (dl.level == "z75")]
    mfz_link = (f"at MFZ none (p = {dlm.iloc[0].p_smaller:.2f})" if len(dlm) and dlm.iloc[0].p_smaller > 0.05
                else "at MFZ see the table")
    svb = sv[(sv.seg == "BL") & (sv["var"] == "VpVs")].iloc[0]
    svp = sv[(sv.seg == "PU") & (sv["var"] == "VpVs")].iloc[0]
    sv_ps = sv.dropna(subset=["p_two"])
    sv_pmin = float(sv_ps.p_two.min()) if len(sv_ps) else float("nan")
    # clustering input check: positions, magnitudes and tiers of the clustering table against the catalog
    zc = pd.read_csv(P.CLU / f"events_zbz_{P.TAG}.csv.gz", usecols=["event_id", "lat", "lon", "depth", "M", "tier"])
    zc = zc.merge(ev.rename(columns={"ID": "event_id"}), on="event_id", suffixes=("_z", ""))
    clu_same = all(((zc[k + "_z"] - zc[k]).abs() > 1e-3).sum() == 0 for k in ("lat", "lon", "depth")) and \
        int((zc.tier_z != zc.tier).sum()) == 0 and int(((zc.M_z - zc.M).abs() > 1e-6).sum()) == 0
    summ_v = pd.read_csv(P.FLT / "fault_surfaces_v1_summary.csv")
    ns = summ_v[summ_v.strand.astype(str).ne("True")]
    n_stable, n_ns = int(ns.stable.astype(str).eq("True").sum()), len(ns)
    swc = P.OUT / f"swarm_change_MSH_v60_{P.TAG}.csv"
    null_csv = P.OUT / "null_truth_contrasts.csv"
    VNOTE = {
        "v6.0": dict(
            status=("Catalog v6.0 is used as a stand-in: v6.1 (D-B103, D-B105) replaces it in the paper and every "
                    "number here is to be recomputed on it."),
            mags="v6.0 M: converged magnitudes of the v5.1 hypocentres, D-B105",
            fig2="ready on v6.0 (sec. 4); recompute on v6.1 with its own magnitudes.",
            faults=("Fault surfaces v1 were fitted on v5.1 positions; only 42 of 130 non-strand surfaces are stable "
                    "under detection reruns (v1 README). Dips and widths here are descriptive."),
            fig3="needs the fault surfaces refit on v6.1 (v1 is fitted on v5.1).",
            catalog=("Catalog v6.0, not v6.1: Axial depths about 0.2 km too deep and the SSST depth drift elsewhere "
                     "(Puget, Blanco) (D-B100, D-B103); magnitudes are the converged values of the v5.1 hypocentres "
                     "(D-B105). The paper text is still on v4.1. Every number here moves with v6.1.")),
        "v6.1": dict(
            status=("Catalog v6.1 (registered; D-B103, D-B104, D-B105, D-B108): these are the paper numbers "
                    "(D-B111). Its clustering, Mc maps and fault surfaces are the v6.1 production products; the v6.0 "
                    "feasibility study is kept at /wd1/mdenolle_data/paper_plan/plate_boundaries_plan.html."),
            mags="v6.1 M: converged magnitudes on the v6.1 hypocentres, D-B105",
            fig2="ready on v6.1 (sec. 4).",
            faults=(f"Fault surfaces v1 refit on v6.1; {n_stable} of {n_ns} non-strand surfaces are marked stable "
                    "under detection reruns. Dips and widths here are descriptive."),
            fig3="fault surfaces refit on v6.1: ready.",
            catalog=("Catalog v6.1 limits (README, D-B110): Axial tier A-B events 73 % within 2 km of the seafloor "
                     "(target 85 %; near-surface Vp/Vs of the model, fix in v6.2); flagged events fed the station "
                     "terms; 129 candidate clock epochs and the location-review verdicts wait for v6.2; no coherence "
                     "relocation at Mendocino. Paper stance (D-B111): deep events under Mount St Helens are not "
                     "interpreted; family-level claims only for large sequences; the seismicity-bottom link to the "
                     "7.5 km/s level is claimed for the Gorda slab only.")),
    }
    vnote = VNOTE.get(L, VNOTE["v6.1"])
    axes_html = axes_section()
    if null_csv.exists():
        null_short, null_long = null_section(pd.read_csv(null_csv), circ)
    else:
        null_short = ("The null-truth test (a truth flattened by depth layer near the segments, D-B112) has no scored table "
                      "(section 3).")
        null_long = ("<p><b>Null truth.</b> Every member of the calibration set inverts data from the same truth, "
                     "which carries the v0.4.2 contrasts, so that set cannot show whether coverage creates a contrast "
                     "from nothing. A null-truth ensemble (D-B112: depth-layer means within 45 km of the segments, "
                     "about 20 members, root /wd1/mdenolle_data/tomo_v1/ensemble_v0.4.2_null) is approved; its rule is "
                     "pre-registered in RESUME.md before scoring. No scored table in this output folder.</p>")
    swc_html = ""
    if swc.exists():
        t = pd.read_csv(swc)
        cols = [c for c in t.columns if c not in ("start",)]
        swc_html = ("<h3>Mount St Helens swarm share, v6.0 to " + L + ", family by family</h3>"
                    "<p>Each v6.0 swarm family with events in the Mount St Helens analysis set, the v6.1 families its "
                    "events fall in, and what became of each event (code pb_swarm_change.py). The share falls from "
                    + MSH_TEXT + "</p>" + table(t[cols]))

    # ---------------- segment table
    rows = []
    for _, r in m.iterrows():
        rows.append({
            "seg": r.seg, "segment": r["name"], "type": r.type, "events A-C": int(r.n_events),
            "depth": r.depth_class, "structure": cls_html(r["class"]),
            "P / PS nodes ≤10 km": f"{int(r.nP_10)} / {int(r.nPS_10)}",
            "PS depth span (km)": r.span_PS10 if isinstance(r.span_PS10, str) else "–",
            "checkerboard share P": fmt(r.cb_shareP_10),
            "Mc": fmt(r.Mc), "b [95 % CI]": ci_str(r.b, r.b_lo, r.b_hi),
            "background %": fmt(r.pct_bg, "{:.0f}"), "MA %": fmt(r.pct_MA, "{:.0f}"),
            "swarm %": fmt(r.pct_swarm, "{:.0f}"),
            "depth below surface p50 / p90 (km)": (f"{r.bsf_med:.1f} / {r.bsf_p90:.1f}" if np.isfinite(r.bsf_med)
                                                  else "–"),
            "below 7.5 km/s": fmt(r.frac_below_z75, "{:.2f}"),
            "surfaces (dip)": (f"{int(r.n_surfaces)} ({r.dip_wmed:.0f}°)" if r.n_surfaces > 0 and
                               np.isfinite(r.dip_wmed) else str(int(r.n_surfaces))),
            "migrating swarms": f"{int(r.n_swarm_fam_migrating)} + {int(r.n_dbscan_migrating)}",
            "burstiness": fmt(r.burst_all),
        })
    seg_table = table(pd.DataFrame(rows), left=("segment", "type", "depth", "PS depth span (km)"))

    # ---------------- feasibility details
    fd = f[["seg", "type", "n_surfaces", "ref_events", "nP_5", "nP_10", "nP_20", "nS_10", "nPS_5", "nPS_10", "nPS_20",
            "span_P10", "span_PS10", "cb_shareP_10", "cb_shareS_10", "cb_cell_P_med_km", "sd_Vp_10", "sd_Vs_10",
            "sd_VpVs_10", "ev_in_P", "ev_in_PS", "max_contrast_ratio", "class"]].copy()
    fd["class"] = fd["class"].map(cls_html)
    fd = fd.rename(columns={"n_surfaces": "surfaces", "ref_events": "reference events", "cb_shareP_10": "cb share P",
                            "cb_shareS_10": "cb share S", "cb_cell_P_med_km": "cb cell (km)",
                            "sd_Vp_10": "1σ Vp", "sd_Vs_10": "1σ Vs", "sd_VpVs_10": "1σ Vp/Vs",
                            "ev_in_P": "events on P nodes", "ev_in_PS": "events on PS nodes",
                            "max_contrast_ratio": "max |Δ|/σ", "span_P10": "P span (km)",
                            "span_PS10": "PS span (km)"})
    feas_table = table(fd, fmts={"1σ Vp": "{:.3f}", "1σ Vs": "{:.3f}", "1σ Vp/Vs": "{:.3f}", "max |Δ|/σ": "{:.1f}",
                                 "cb cell (km)": "{:.0f}"}, left=("type", "reference events", "P span (km)",
                                                                  "PS span (km)"))

    # ---------------- contrasts
    keep = f[f["class"].isin(["resolved", "marginal"])].seg.tolist()
    c2 = ct[ct.seg.isin(keep) & ct.value.notna()].copy()
    c2["unit"] = np.where(c2["var"] == "VpVs", "", "km/s")
    c2 = c2[["seg", "var", "contrast", "n1", "n2", "value", "sigma_members", "ratio", "frac_members_same_sign",
             "median_node_sd", "ratio_node_sd"]]
    c2 = c2.rename(columns={"value": "Δ (v0.4.2)", "sigma_members": "σ of Δ (members)", "ratio": "|Δ|/σ",
                            "frac_members_same_sign": "members same sign", "median_node_sd": "node 1σ",
                            "ratio_node_sd": "|Δ|/node 1σ"})
    c2["seg"] = pd.Categorical(c2.seg, P.SEG_ORDER, ordered=True)
    c2 = c2.sort_values(["seg", "var", "contrast"])
    c2["seg"] = c2.seg.astype(str)
    contrast_table = table(c2, fmts={"Δ (v0.4.2)": "{:+.3f}", "σ of Δ (members)": "{:.3f}", "|Δ|/σ": "{:.1f}",
                                     "node 1σ": "{:.3f}", "|Δ|/node 1σ": "{:.2f}"}, left=("var", "contrast"))

    # ---------------- circularity control
    ci_ = circ.copy()
    ci_["seg"] = pd.Categorical(ci_.seg, P.SEG_ORDER, ordered=True)
    ci_ = ci_.sort_values(["seg", "var", "contrast"])
    ci_["seg"] = ci_.seg.astype(str)
    ci_["verdict"] = np.where(ci_.survives, '<span class="resolved">survives</span>',
                              np.where(ci_.old_rule_2sigma, '<span class="marginal">dropped by S95</span>', "–"))
    ci_t = ci_[["seg", "var", "contrast", "D_real", "sigma_ens", "D_truth", "spur_mean", "spur_sd", "spur_p05",
                "spur_p95", "S95", "recovery", "noise_floor_sd", "verdict"]]
    ci_t = ci_t.rename(columns={"D_real": "Δ real", "sigma_ens": "σ ens", "D_truth": "Δ truth (synth)",
                                "spur_mean": "spurious mean", "spur_sd": "spurious sd", "spur_p05": "spurious p05",
                                "spur_p95": "spurious p95", "S95": "S95 = p95 |spurious|", "recovery": "recovery",
                                "noise_floor_sd": "noise floor sd"})
    circ_table = table(ci_t, fmts={c: "{:+.3f}" for c in ["Δ real", "Δ truth (synth)", "spurious mean",
                                                            "spurious p05", "spurious p95"]} |
                       {"σ ens": "{:.3f}", "spurious sd": "{:.3f}", "S95 = p95 |spurious|": "{:.3f}",
                        "noise floor sd": "{:.4f}"}, left=("var", "contrast"))
    n_test, n_old, n_surv = int(circ.testable.sum()), int(circ.old_rule_2sigma.sum()), int(circ.survives.sum())
    dropped = circ[circ.old_rule_2sigma & ~circ.survives]
    vn_ = {"Vp": "Vp", "Vs": "Vs", "VpVs": "Vp/Vs"}
    dropped_s = "; ".join(f"{r.seg} {vn_[r['var']]} {r.contrast} ({r.D_real:+.3f}, S95 {r.S95:.3f})"
                          for _, r in dropped.iterrows())
    sv_ = circ[circ.survives]

    def surv_list(segs):
        x = sv_[sv_.seg.isin(segs)]
        return "; ".join(f"{r.seg} {vn_[r['var']]} {r.contrast} {r.D_real:+.2f}" for _, r in x.iterrows()) or "none"
    med_ratio = float(np.median(circ.S95 / (2 * circ.sigma_ens)))
    rec_med, rec_lo, rec_hi = (float(np.percentile(sv_.recovery.dropna(), q)) for q in (50, 10, 90))
    ocean = {"JR", "GR", "BL", "MFZ", "SOV", "GP", "MTJ", "GS"}
    bl_c = circ[circ.seg == "BL"]
    bl_note = ("no surviving contrast (nothing reaches 2σ)" if not bl_c.old_rule_2sigma.any()
               else f"no surviving contrast ({int(bl_c.old_rule_2sigma.sum())} reach 2σ, none passes S95)"
               if not bl_c.survives.any() else f"{int(bl_c.survives.sum())} surviving contrasts")
    vv = sv_[sv_["var"] == "VpVs"]
    vpvs_note = ((("No Vp/Vs contrast survives on an oceanic boundary; " if not vv.seg.isin(ocean).any() else "")
                  + f"the Vp/Vs survivors are {', '.join(f'{r.seg} {r.contrast}' for _, r in vv.iterrows())} "
                  f"(|Δ| {vv.D_real.abs().min():.3f}-{vv.D_real.abs().max():.3f}).") if len(vv)
                 else "No Vp/Vs contrast survives.")
    big = sv_[sv_.seg.isin(["MTJ", "GS"]) & (sv_.contrast == "along") & sv_["var"].isin(["Vp", "Vs"])].D_real.abs()
    big_lo, big_hi = float(big.min()), float(big.max())
    sw_max = float(s.set_index("seg").loc[["MFZ", "GP", "MTJ", "GS"], "pct_swarm"].max())
    mg = s[(s.n_swarm_fam_migrating + s.n_dbscan_migrating) > 0]
    mig_txt = ", ".join(f"{r.seg} {int(r.n_swarm_fam_migrating)} zbz + {int(r.n_dbscan_migrating)} DBSCAN"
                        for _, r in mg.iterrows()) or "none flagged"

    # ---------------- seismicity metrics full
    sm = s[["seg", "type", "n_events", "n_with_M", "Mmax", "n_M4", "Mc", "Mc_lo", "Mc_hi", "Mc_map_median", "b", "b_lo",
            "b_hi", "n_above_Mc", "n_zbz", "pct_bg", "pct_bg_lo", "pct_bg_hi", "pct_MA", "pct_swarm", "pct_multiplet",
            "swarm_share_of_clustered", "n_fam_mainshock-aftershock", "n_fam_swarm", "n_fam_multiplet",
            "n_swarm_fam_ge10", "n_swarm_fam_migrating", "n_dbscan_swarmlike", "n_dbscan_migrating"]].copy()
    sm.columns = ["seg", "type", "events", "with M", "Mmax", "M≥4", "Mc", "Mc lo", "Mc hi", "Mc map", "b", "b lo",
                  "b hi", "n ≥ Mc", "n clustering", "bg %", "bg lo", "bg hi", "MA %", "swarm %", "multiplet %",
                  "swarm % of clustered", "MA fam", "swarm fam", "multiplet fam", "swarm fam ≥10", "migrating (zbz)",
                  "DBSCAN swarm-like", "migrating (DBSCAN)"]
    seis_table1 = table(sm, fmts={"bg %": "{:.0f}", "bg lo": "{:.0f}", "bg hi": "{:.0f}", "MA %": "{:.0f}",
                                  "swarm %": "{:.0f}", "multiplet %": "{:.0f}", "swarm % of clustered": "{:.0f}"},
                        left=("type",))
    sd_ = s[["seg", "depth_class", "n_depth_good", "sig_z_med_good", "bsf_p10", "bsf_med", "bsf_med_lo", "bsf_med_hi",
             "bsf_p90", "bsf_p90_lo", "bsf_p90_hi", "bsf_med_raw", "z75_bsf_med", "frac_with_z75", "frac_below_z75",
             "frac_below_z75_lo", "frac_below_z75_hi", "frac_below_z75_mem_p05", "frac_below_z75_mem_p95",
             "frac_with_z8", "frac_below_z8", "rel_slab_med", "dip_wmed", "dip_min", "dip_max", "seis_width_med_km",
             "seis_bottom_med_km", "frac_events_on_surfaces"]].copy()
    sd_.columns = ["seg", "depth class", "depth-good", "median σz", "bsf p10", "bsf p50", "p50 lo", "p50 hi", "bsf p90",
                   "p90 lo", "p90 hi", "bsf p50 (unconstrained)", "7.5 km/s level bsf", "share with level",
                   "below 7.5", "lo", "hi", "members p05", "members p95", "share with 8 km/s", "below 8",
                   "depth - Slab2", "dip (wtd)", "dip min", "dip max", "seismogenic width", "seismogenic bottom",
                   "share on surfaces"]
    seis_table2 = table(sd_, left=("depth class",))
    tm = s[["seg", "first", "last", "years_with_events", "rate_per_yr_geMc", "cov_all", "burst_all", "cov_bg",
            "burst_bg", "share_busiest_1pct_days", "largest_family_share"]].copy()
    tm.columns = ["seg", "first", "last", "years", "events ≥ Mc per yr", "COV all", "burstiness all", "COV bg",
                  "burstiness bg", "share on busiest 1 % of days", "largest family / events"]
    seis_table3 = table(tm, fmts={"events ≥ Mc per yr": "{:.0f}"}, left=("first", "last"))

    dl2 = dl.dropna(subset=["sd_real"])[["seg", "level", "n_columns", "median_d90_minus_level", "rho", "sd_real",
                                         "sd_null_med", "p_smaller", "sd_d90", "sd_level"]]
    dl2 = dl2.rename(columns={"median_d90_minus_level": "median D90 − level (km)", "rho": "Spearman ρ",
                              "sd_real": "sd(D90 − level)", "sd_null_med": "null sd (median)",
                              "p_smaller": "p", "sd_d90": "sd D90", "sd_level": "sd level"})
    depth_level_table = table(dl2, fmts={"p": "{:.3f}"}, left=("level",))
    sv2 = sv[(sv.n_swarm > 0) | (sv.n_MA > 0)][["seg", "var", "n_swarm", "n_MA", "n_multiplet", "diff", "null_lo",
                                                 "null_hi", "p_two", "sigma_members"]]
    sv2 = sv2.rename(columns={"diff": "swarm − MA (median)", "null_lo": "null 2.5 %", "null_hi": "null 97.5 %",
                              "p_two": "p", "sigma_members": "σ members"})
    swarm_table = table(sv2, fmts={"swarm − MA (median)": "{:+.3f}", "null 2.5 %": "{:+.3f}", "null 97.5 %": "{:+.3f}",
                                   "σ members": "{:.3f}"}, left=("var",))
    sens_table = table(sens)

    # ---------------- inputs
    inputs = [(f"catalog {L}", P.CAT), ("clustering events", P.CLU / f"events_zbz_{P.TAG}.csv.gz"),
              ("clustering families", P.CLU / f"families_zbz_{P.TAG}.csv"), (f"Mc map {L}", P.MCMAP / "mc_grid.csv"),
              ("model v0.4.2", P.MODEL), ("ensemble v0.4.2+ens.100", P.ENS),
              ("fault surfaces v1 mesh", P.FLT_BIG / "fault_surfaces_v1_vertices.csv.gz"),
              ("fault surfaces v1 summary", P.FLT / "fault_surfaces_v1_summary.csv")]
    inp = "".join(f"<li>{a}: <code>{p}</code> (md5 {md5(p)})</li>" for a, p in inputs)

    def g(seg, col):
        return m.set_index("seg").loc[seg, col]

    cc = ct[ct.seg.isin(f[f["class"].isin(["resolved", "marginal"])].seg) & ct.value.notna()]
    n_ct, n_2s, n_node = len(cc), int((cc.ratio >= 2).sum()), int((cc.ratio_node_sd > 1).sum())
    nf = cc[(cc.contrast == "near-far") & (cc.ratio >= 2)]
    vn = {"Vp": "Vp", "Vs": "Vs", "VpVs": "Vp/Vs"}
    nearfar = "; ".join(f"{r.seg} {vn[r['var']]} {r.value:+.2f}{'' if r['var'] == 'VpVs' else ' km/s'} "
                        f"({r.ratio:.1f}σ)" for _, r in nf.iterrows())
    res = f[f["class"] == "resolved"]
    res_pb = [x for x in res.seg if P.SEGMENTS[x][1] != "other"]
    mar = f[f["class"] == "marginal"].seg.tolist()
    nres = f[f["class"] == "not resolved"].seg.tolist()

    body = f"""
<h1>Plate-boundary segments: structure feasibility, seismicity style and figure plan ({L})</h1>
<p class="small">Draft for the Science Advances re-framing (D-B107, PI 2026-11-26 16:05). Built {html.escape(when)};
branch <code>{branch}</code> at <code>{head}</code> (worktree /home/mdenolle/wt_pbplan). Private working page; not
peer reviewed. {vnote['status']}</p>

<div class="box"><b>What this page answers.</b> D-B107 asks for a paper on fault-zone structure and seismicity style
at the plate boundaries, with velocity structure shown only near faults and only where the model resolves it. Before
figures are drawn: which boundaries does model v0.4.2 resolve, how large is the ensemble uncertainty there, how large
are the structure contrasts against it, and how do the boundaries differ in seismicity style? Section 5 is the figure
plan; draft mock-ups are in sections 4 and 6.</div>

<h2>1. Findings</h2>
<ul>
<li><b>Coverage.</b> The rule below sorts {n_all:,} located tier A-C earthquakes of {L} (flagged non-earthquakes
and the waveform-review P1 set left out) into 21 segments, 19 of them on plate boundaries. No tomography tile covers
the model columns of Axial Seamount, Endeavour or the Explorer segments (n_tiles = 0, DWS = 0), and the JdF plate
interior lies at the tile edges (largest P DWS 84 in its columns). So {', '.join(nres)} have no resolved node; their
style metrics stand alone.</li>
<li><b>Resolved for a fault-zone panel</b> (≥ 15 P and ≥ 10 P-and-S nodes within 10 km over ≥ 3 depth layers):
{', '.join(res_pb)} (plus the non-boundary groups OC and NAPA). <b>Marginal</b>: {', '.join(mar)} (Gorda ridge and
the northern subduction groups have P nodes only, FA 2 P-and-S nodes; JR has 13 P nodes at 5-7 km). The P-and-S depth span is
{g('BL', 'span_PS10')} km at Blanco, {g('MFZ', 'span_PS10')} km at the Mendocino fault zone,
{g('GP', 'span_PS10')} km in the Gorda plate, {g('MTJ', 'span_PS10')} km at the triple junction,
{g('PU', 'span_PS10')} km in Puget crust and {g('MSH', 'span_PS10')} km at Mount St Helens.</li>
<li><b>Scale.</b> The finest checkerboard cell that passes near every segment is 40 km, and most near-fault nodes
pass the mask on ray sampling alone (DWS ≥ 100): the checkerboard share within 10 km is {g('BL', 'cb_shareP_10'):.0%}
(P) and {g('BL', 'cb_shareS_10'):.0%} (S) at Blanco, {g('MSH', 'cb_shareP_10'):.0%} at Mount St Helens,
{g('PU', 'cb_shareP_10'):.0%} in Puget and below 10 % at MFZ, Gorda plate, MTJ and the Gorda slab. The panels can show
the structure of the plate-boundary zone on a 40 km scale, not a fault damage zone. Figure captions and the text have
to say this.</li>
<li><b>Ensemble 1σ near the faults</b> (median over nodes within 10 km, calibrated): Vp {g('MSH', 'sd_Vp_10'):.2f}
km/s at Mount St Helens to {g('BL', 'sd_Vp_10'):.2f} km/s at Blanco; Vp/Vs {g('MSH', 'sd_VpVs_10'):.3f} to
{g('BL', 'sd_VpVs_10'):.3f} (raw member spread: the synthetic calibration found Vp/Vs calibrated without
inflation, D-B111). The Blanco tile (smoothing 20) has the largest member spread.</li>
<li><b>Contrasts after the circularity control</b> (rule pre-registered 2026-11-27 09:38 PST, commit 6b64abf1,
before scoring: a contrast counts only if |Δ| ≥ 2σ of the real ensemble and |Δ| ≥ S95, the 95th percentile of
|recovered − truth| over the 100 synthetic members). Of {n_test} contrasts on resolved and marginal segments, {n_old}
pass 2σ alone (Vp/Vs with the raw σ) and {n_surv} survive both. Dropped by S95: {dropped_s}. The spurious contrasts
are small and centred: their sd is about the raw member σ and S95 is a median {med_ratio:.2f} of 2σ, so the synthetic
test removes few contrasts. Surviving on the oceanic boundaries: {surv_list(["GR", "BL", "MFZ", "GP"])}; at the
triple junction and the Gorda slab: {surv_list(["MTJ", "GS"])}; on land: {surv_list(["PU", "MSH", "CAS"])}. Blanco
has {bl_note}. {vpvs_note} The largest values ({big_lo:.1f}-{big_hi:.1f} km/s along strike at MTJ and in the Gorda
slab) follow the onshore-offshore transition, not a fault zone. {null_short}</li>
<li><b>Seismicity style</b> separates the boundary types more clearly than structure does. Swarm share of the events
above Mc: Axial {g('AX', 'pct_swarm'):.0f} %, Mount St Helens {g('MSH', 'pct_swarm'):.0f} %, Blanco
{g('BL', 'pct_swarm'):.0f} %, Gorda ridge {g('GR', 'pct_swarm'):.0f} %, against ≤ {sw_max:.1f} % at MFZ, Gorda plate,
MTJ and the Gorda slab. Mainshock-aftershock share is highest in the Gorda plate ({g('GP', 'pct_MA'):.0f} %).
b-values range from {g('IS', 'b'):.2f} (JdF intraslab) and {g('AX', 'b'):.2f} (Axial) to {g('PU', 'b'):.2f} (Puget
crust). Median depth below the surface: Axial {g('AX', 'bsf_med'):.1f} km, Blanco {g('BL', 'bsf_med'):.1f} km,
Mount St Helens {g('MSH', 'bsf_med'):.1f} km, Puget {g('PU', 'bsf_med'):.1f} km, Gorda slab
{g('GS', 'bsf_med'):.1f} km, MTJ {g('MTJ', 'bsf_med'):.1f} km, MFZ {g('MFZ', 'bsf_med'):.1f} km.</li>
<li><b>Depth against the 7.5 km/s level.</b> The model rarely reaches 8 km/s offshore (MFZ peaks near 7.9 km/s), so
the 8 km/s level is resolved in few columns; 7.5 km/s is resolved for {g('BL', 'frac_with_z75'):.0%} of the
depth-good Blanco events and {g('MTJ', 'frac_with_z75'):.0%} at MTJ. Share of events below it: MFZ
{g('MFZ', 'frac_below_z75'):.0%}, Gorda plate {g('GP', 'frac_below_z75'):.0%}, MTJ {g('MTJ', 'frac_below_z75'):.0%},
Blanco {g('BL', 'frac_below_z75'):.0%}, Gorda slab {g('GS', 'frac_below_z75'):.0%}. Per model column, the bottom of
the Gorda-slab seismicity (D90) follows the 7.5 km/s level (ρ = {dlg.rho:.2f}, p = {dlg.p_smaller:.3f} against permuted
levels, {int(dlg.n_columns)} columns); at Blanco ρ = {dlb.rho:.2f}, p = {dlb.p_smaller:.3f}; {mfz_link}. The paper claims
this link for the Gorda slab only (D-B111).</li>
<li><b>Swarms against mainshock-aftershock families</b> can be compared on resolved nodes only at Blanco (Vp/Vs:
{int(svb.n_swarm)} swarm against {int(svb.n_MA)} MA families) and in Puget ({int(svp.n_swarm)} against {int(svp.n_MA)}).
Neither differs (smallest p {sv_pmin:.2f} over Vp and Vp/Vs), as in the v5.1 analysis. A panel on this would report a
null result with few families.</li>
</ul>

<h2>2. Segment rule and segment table</h2>
<p>Rule (code <code>workflow/11_paper_plan/pb_common.py</code>, first match wins):</p>
<ol>{''.join(f'<li>{html.escape(x)}</li>' for x in P.SEGMENT_RULE)}</ol>
<p class="small">Events: located earthquakes of {L} (event_flag empty: {n_flag:,} flagged events left out, of them
{n_analog} analog_test_signal), tiers A-C, review P1 (Mount St Helens weekly-signal candidates, {n_p1} further tier A-C
events) left out. Fault surfaces v1 belong to the segment of most of their fit members at the {L} positions (table <code>surface_segments.csv</code>; {int((sseg.seg_share >= 0.8).sum())} of
{len(sseg)} surfaces have ≥ 80 % of their members in one segment). Threshold sensitivity is in section 7.</p>
{img("fig_segments_map.png", f"Segments on catalog {L} (tier A-C earthquakes, coloured by type) with the structure class "
     "of each segment (label fill: black resolved, grey marginal, white not resolved). Coastline: zero contour of the "
     "model surface. Draft.")}
<p>Segment table. Structure class: section 3. Depth class from the depth-good events: good (median σz ≤ 3 km), fair
(3-6 km), none (fewer than 50 depth-good events). Shares are of the events above the clustering Mc ({L} nearest-
neighbour run). Migrating swarms: zbz swarm families ≥ 10 events + DBSCAN swarm-like clusters flagged migrating.</p>
{seg_table}
{axes_html}

<h2>3. Structure feasibility</h2>
<p>Model v0.4.2 on its node grid (0.125° x 0.1°, about 10 x 11 km; depth nodes 2-6 km apart above 30 km).
Masks as used so far: rock and (checkerboard-resolved or DWS ≥ 100), P for Vp, S for Vs, both for Vp/Vs; nodes whose
Vs was reverted to the reference (D-B051) are left out of the S mask. Distance of a node: 3-D distance to the
nearest point of the segment's fault surfaces (1 km mesh) or of its seismicity (depth-good events where there are
≥ 50, else tier A-C). Ensemble 1σ: v0.4.2+ens.100 Vp_std (x 1.50) and Vs_std (x 1.35) as stored; Vp/Vs: the raw
member std (factor 1.0, D-B111). Class: <b>resolved</b> ≥ 15 P and ≥ 10 P-and-S nodes within 10 km over ≥ 3 depth layers;
<b>marginal</b> ≥ 5 P nodes within 10 km; else <b>not resolved</b>. Spans list depth layers with ≥ 3 nodes.</p>
{feas_table}
<h3>Contrasts against the ensemble</h3>
<p>Depth-matched contrasts (per depth layer, weighted by the smaller node count): <i>near-far</i> = nodes within
5 km minus nodes 10-20 km away; <i>across</i> = the two sides of the segment's strike line (PCA of its epicentres;
nodes within 3 km of the line left out; + = left of the strike direction); <i>along</i> = the third of the nodes within 10 km
at the strike-azimuth end (strike 0-180°, <code>segment_axes.csv</code>) minus the third at the other end. σ of Δ: the same contrast in each of the 100 members, std x calibration
factor. "members same sign" = share of members whose contrast has the sign of the v0.4.2 value.</p>
{contrast_table}
<h3>Circularity control (synthetic ensemble, D-B111)</h3>
<p><b>Rule, pre-registered</b> in <code>data/paper_plan/plate_boundaries/RESUME.md</code> at Fri Nov 27 09:38:33 PST
2026 (commit 6b64abf1, pushed before any synthetic contrast was computed): a contrast counts only if |Δ_real| ≥ 2σ_ens
(Vp x1.50, Vs x1.35, Vp/Vs x1.00) and |Δ_real| ≥ S95, where S95 is the 95th percentile of |Δ_rec,i − Δ_truth| over the
synthetic members; fewer than 50 members with a value means not testable.</p>
<p>Synthetic set: <code>/wd1/mdenolle_data/tomo_v1/ensemble_v0.4.2/synth</code>, one truth (v0.4.2 tomography x (1 + f),
3 % random field, 30 km / 8 km correlation, truth hypocentres drawn about v4-NLL3D), synthetic data on the real
event-station geometry and pick list, 100 members s000-s099 inverted from their own start draws, and five noise-floor
clones of s000. The same node sets and depth matching as the real contrasts; truth and members compared before the
D-B073 node rule. Spurious contrast = recovered − truth: it holds what ray sampling, the events' trade-offs and the
regularisation do to a contrast that is in the truth. All {n_test} contrasts were testable.</p>
{null_long}
{img("fig_circularity.png", "Real contrasts with ±2σ of the real ensemble and ±S95 of the synthetic spurious contrasts "
     "(grey). Filled = survives the pre-registered rule. Colours by segment type. Draft.")}
{circ_table}

<h2>4. Seismicity-style metrics</h2>
<p>Mc: maximum curvature + 0.2 on the segment's tier A-C magnitudes ({vnote['mags']}); b: Aki-Utsu for continuous magnitudes above Mc; both re-estimated in 500 event bootstrap draws
(95 % intervals). "Mc map": event-weighted median of the {L} Mc map cells (flagged events excluded). Clustering
shares: {L} nearest-neighbour clustering, events in its analysis set (above the regional Mc of that run), with a
bootstrap over units (one background event or one whole family).</p>
{img("fig_fingerprint.png", "Seismicity-style fingerprint of the plate-boundary segments (mock-up of main figure 2). "
     "Label: segment code and structure class (r resolved, m marginal, n not resolved). Draft.")}
<h3>Size, completeness, clustering</h3>{seis_table1}
<h3>Depth and geometry</h3>
<p>bsf = depth below the solid surface (ground or seafloor), depth-good events only (tier A-C, depth_quality ok, not
floor-limited, far-only or deep-pass). For segments without depth-good events the raw median is given in the
"unconstrained" column and should not be used. 7.5 and 8 km/s levels: depth of the first Vp crossing in the event's
model column between two P-resolved nodes; "members p05/p95" = the share recomputed with each member's level (raw
spread). Fault-surface columns: v1 surfaces of the segment (dip weighted by member count; seismogenic width = 2-98 %
depth range of depth-ok members).</p>
{seis_table2}
<h3>Temporal character</h3>
<p>Events ≥ the segment Mc over the whole period. COV = coefficient of variation of inter-event times (1 for a
Poisson process); burstiness = (σ − μ)/(σ + μ). The network changed with the OBS deployments (2011-2015) and Axial
enters tier A-C only from 2013-09, so rates and burstiness compare segments only roughly.</p>
{seis_table3}
{swc_html}

<h2>5. Figure plan</h2>
<p>Six main figures, each a set of small multiples with segments on a shared axis and one colour or marker per
boundary type (ridge, transform, intraplate, triple junction, subduction, crustal, arc). Velocity appears only in
figures 4 and 5 and only on resolved nodes near the segments. All on catalog v6.1 and model v0.4.2.</p>
<table><thead><tr><th>#</th><th>content</th><th>data, status</th><th>from the current figures</th></tr></thead><tbody>
<tr><td>1</td><td class="l">Plate boundaries and data: (A) v6.1 seismicity coloured by segment type with the segment
codes and their structure class; (B) stations; (C) margin-normal depth section; (D) weekly rate and station count.</td>
<td class="l">segment rule frozen; v6.1 catalog. Mock-up: segment map (sec. 2).</td>
<td class="l">fig1 stays; panel A recoloured by segment.</td></tr>
<tr><td>2</td><td class="l">Seismicity-style fingerprint: per segment b and Mc with 95 % CI; background /
mainshock-aftershock / swarm / multiplet shares; depth below surface p10-p50-p90; burstiness and largest-sequence
share; fault-surface dip. Segments grouped by type on one y axis.</td><td class="l">{vnote['fig2']} Mock-up:
fig_fingerprint.</td><td class="l">replaces fig5 D-E (family mix by region)
and the regional b/Mc numbers.</td></tr>
<tr><td>3</td><td class="l">Fault geometry per boundary: one small multiple per segment with surfaces (BL, MFZ, GR, GP,
MTJ, GS, PU, MSH, CAS): map of the surfaces with depth-good events and an along-strike depth view with the seismogenic
depth range; the Blanco strand panel as the BL member.</td><td class="l">{vnote['fig3']}</td><td class="l">fig3 and fig4 merged into one small-multiple figure; 3-D views to the
supplement.</td></tr>
<tr><td>4</td><td class="l">Fault-zone structure sections: fault-normal sections stacked along strike, shared axes
(±45 km, 0-40 km bsl) and colour scales, Vp and Vp/Vs with the ensemble 1σ (third row or contours), seismicity and the
7.5 km/s level; columns BL, MFZ, GP, MTJ/GS, PU, MSH (CAS optional).</td><td class="l">feasible for {', '.join(res_pb)};
mock-up for BL and MFZ (below). 40 km resolution stated in the caption.</td><td class="l">new.</td></tr>
<tr><td>5</td><td class="l">Structure against seismicity, with controls: (A) seismicity bottom (D90) against the
7.5 km/s level per model column (GS, BL, MFZ, MTJ) with the permutation null; (B) share of seismicity below the
7.5 km/s level per segment with the member range; (C) depth-matched contrasts against the member σ; (D) Vp/Vs at swarm
and mainshock-aftershock families (BL, PU), expected null.</td><td class="l">numbers in sec. 3 and 6; circularity
control done on the synthetic ensemble (sec. 3); null-truth synthetics and the location Monte Carlo still to do.</td>
<td class="l">new; replaces the v5.1 struct figures.</td></tr>
<tr><td>6</td><td class="l">Temporal character by boundary type: Axial 2015 eruption, West Blanco 2015 swarm, Gorda
plate sequences, MFZ; add a Mount St Helens swarm sequence; burstiness values in the panels.</td>
<td class="l">existing panels on v4.1; redo on v6.1.</td><td class="l">fig6 stays.</td></tr>
</tbody></table>
<p><b>Supplement.</b> S1 segment rule and threshold sensitivity (sec. 7); S2 Mc maps, three methods and by period
(current fig2a, fig2b, on v6.1 without flagged events, D-B105); S3 catalog comparison (current figS); S4
nearest-neighbour density plots by segment type (current fig5 A-C, regrouped); S5 structure feasibility: node counts,
checkerboard share and ensemble 1σ maps limited to the segment areas; S6 five individual ensemble members of the BL
and MFZ sections (what the 1σ means); S7 Mendocino perspective view (current perspective_mendocino_fault); S8
migrating swarms ({mig_txt}); S9 the unresolved boundaries
(Axial, Endeavour, Explorer, JdF plate): seismicity-only panels.</p>
<p><b>Leaves the main text:</b> fig2a and fig2b (Mc maps) move to the supplement; fig5 A-C move to S4; fig4 becomes a
panel of figure 3 or moves to the supplement. No full-size velocity maps or margin-normal model sections (D-B107).</p>

<h2>6. Mock-up: fault-normal sections, Blanco and Mendocino fault zone</h2>
{img("fig_sections_BL_MFZ.png", f"Model v0.4.2 on resolved nodes in 5 km bins of fault-normal distance (≥ 2 nodes per "
     f"bin), stacked along {sec.loc['BL', 'length_km']:.0f} km (Blanco) and {sec.loc['MFZ', 'length_km']:.0f} km (MFZ) "
     f"of the PB2002 transform trace and centred on the seismicity (trace offset {sec.loc['BL', 'offset_km']:+.1f} km "
     f"and {sec.loc['MFZ', 'offset_km']:+.1f} km). Rows: Vp, its 1σ (x1.50), Vp/Vs (P and S mask), its 1σ (raw, D-B111). "
     "Dots: depth-good tier A-C events of the segment. Dashed: 7.5 km/s. Draft.")}
<p>What the mock-up shows: within 10 km of the Blanco faults and seismicity the resolved layers span
{g('BL', 'span_P10')} km for Vp and {g('BL', 'span_PS10')} km for Vp/Vs and the seismicity sits in them; at MFZ
{g('MFZ', 'span_P10')} km (P) and {g('MFZ', 'span_PS10')} km (P and S), with {g('MFZ', 'frac_below_z75'):.0%} of the
seismicity below the 7.5 km/s level. Median Vp 1σ: Blanco {g('BL', 'sd_Vp_10'):.2f} km/s, MFZ
{g('MFZ', 'sd_Vp_10'):.2f} km/s. Structure within ±10 km of the faults is near the node spacing.</p>
<h3>Depth extent against the velocity level, per model column</h3>{depth_level_table}
<p class="small">Columns with ≥ 20 depth-good events and a resolved level. p = share of 1000 permutations of the level
among the segment's columns with a smaller sd(D90 − level).</p>
<h3>Swarm against mainshock-aftershock families on resolved nodes</h3>{swarm_table}
<p class="small">Families with ≥ 5 depth-good members, placed at the node nearest their median position. Null: type
labels permuted among families of the same depth layer (2000 draws). σ members: the same difference in each member x
calibration factor.</p>

<h2>7. Caveats</h2>
<ul>
<li>{vnote['catalog']}</li>
<li>No structure for Axial, Endeavour and Explorer (outside every tomography tile) or the JdF plate interior (tile
edges, DWS below 100).</li>
<li>Resolution: 40 km checkerboard cells and a 10 km node grid. Most nodes near the faults pass the mask on DWS ≥ 100
only, which shows ray sampling, not resolution. Ray coverage follows the seismicity, so velocity near events can be
biased in the same place the events are. The synthetic-ensemble control (section 3) bounds the error of contrasts
present in the truth; a null-truth synthetic, which would show contrasts created from nothing, has not been run.</li>
<li>Uncertainty: the ensemble 1σ for Vp and Vs is calibrated (x 1.50, x 1.35; D-B099); Vp/Vs uses the raw spread
(calibrated without inflation, D-B111). The member spread does not include smoothing bias from limited resolution;
the synthetic recovery ratios of the surviving contrasts (median {rec_med:.2f}, 10-90 % {rec_lo:.2f}-{rec_hi:.2f})
give its size for this truth.</li>
<li>Depths: the ridges other than Axial, Explorer and the JdF plate have depth_quality unconstrained; Gorda events are
mostly biased_deep ({gp_good:,} depth-good of {gp_all:,} Gorda-plate events); Blanco depth-good events have median
σz {bl_sz:.1f} km and a deep tail against Lange et al. (2026). Depth rows are marked fair or none accordingly.</li>
<li>Segment rule: PB2002 traces are approximate (MFZ seismicity lies {sec.loc['MFZ', 'offset_km']:+.0f} km and Blanco
{sec.loc['BL', 'offset_km']:+.0f} km from the trace, + north); the thresholds (MTJ radius, axis half-width, 41.5 N, ±5 km
about Slab2) are choices. MTJ changes from {sn.loc['MTJ', 'R_MTJ_KM=30.0']:,} to {sn.loc['MTJ', 'R_MTJ_KM=50.0']:,}
events between radii of 30 and 50 km; the interface-zone group SIF from {sn.loc['SIF', 'SLAB_IF_KM=3.0']:,} to
{sn.loc['SIF', 'SLAB_IF_KM=8.0']:,} between ±3 and ±8 km.</li>
<li>{vnote['faults']}</li>
<li>Clustering: the {L} run used its own regional Mc (AX, EN, BL, GO, MT, WA, OR, PU, ON), not the segment Mc.
Positions, magnitudes and tiers of its event table {'match' if clu_same else 'DO NOT match'} the catalog file read here
(checked when this page was built). Bootstrap intervals ignore correlation in time.</li>
<li>Temporal metrics depend on network changes; they are rough comparisons only.</li>
</ul>

<h3>Threshold sensitivity (tier A-C events per segment)</h3>{sens_table}

<h2>8. Reproduce</h2>
<p>Worktree /home/mdenolle/wt_pbplan, branch analysis/plate-boundaries (from origin/main 5b42fcd2; merged with
pipeline/end-to-end f21e87c6, which added the PB_* variables). Catalog {L}: <code>{P.CAT}</code>; tables
<code>{P.OUT}</code>; page and figures <code>{P.BIG}</code>{'; run with workflow/11_paper_plan/run_v61.sh' if L == 'v6.1' else ''}. Python:
<code>LD_LIBRARY_PATH=/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/lib
/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/bin/python</code>, nice 10, one process. In
<code>workflow/11_paper_plan/</code>: <code>pb_structure.py</code> (events with segments, levels, feasibility,
contrasts; about 40 s), <code>pb_seismicity.py</code>, <code>pb_struct_seis.py</code>,
<code>pb_sensitivity.py</code>, <code>pb_figures.py</code>, <code>pb_circularity.py</code> (after
pb_structure.py, which writes the contrast node sets), <code>pb_page.py "$(date)"</code>. Small tables:
<code>data/paper_plan/plate_boundaries/</code>; page, figures and the event table: /wd1/mdenolle_data/paper_plan/.
Colour helpers: <code>workflow/08_figures/scm_palettes.py</code> (copy of paper/model-figures 7aea31dd).</p>
<ul>{inp}</ul>
"""
    doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width, initial-scale=1">'
           f"<title>Plate-boundary paper plan</title><style>{CSS}</style></head><body>{body}</body></html>")
    PAGE.write_text(doc)
    P.log(f"wrote {PAGE} ({PAGE.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

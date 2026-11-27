#!/usr/bin/env python3
"""
HTML report of the converged Route A magnitudes (decision D-B091).

Reads the tables written by mag_converged.py (data/magnitude_converged/*.csv), the Mc / b
and clustering outputs under /wd1/mdenolle_data/magnitude_converged, and writes
data/magnitude_converged/magnitude_converged_report.html (self-contained: figures are
embedded) plus the figures as PNG in data/magnitude_converged/figs/ and the paper-impact
table paper_impact.csv.

Usage (from workflow/05_magnitude):  python mag_converged_report.py
"""
from __future__ import annotations

import base64
import html
import io
import json
import os

import numpy as np
import pandas as pd

import mag_converged as mc

SMALL = mc.SMALL
OUT = mc.OUT
FIGD = f"{SMALL}/figs"
PAPER = os.path.normpath(os.path.join(mc.HERE, "../../paper_sciadv/figures"))
PAPER_MC = os.path.normpath(os.path.join(mc.HERE, "../../data/catalog_v4/mc_maps"))
PAPER_ZBZ = os.path.normpath(os.path.join(mc.HERE, "../../data/catalog_v4/clusters_zbz"))
FIGS = {}


def t(name):
    p = f"{SMALL}/{name}.csv"
    return pd.read_csv(p) if os.path.exists(p) else None


def savefig(fig, name):
    import matplotlib.pyplot as plt
    os.makedirs(FIGD, exist_ok=True)
    fig.savefig(f"{FIGD}/{name}.png", dpi=130, bbox_inches="tight")
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    FIGS[name] = base64.b64encode(b.getvalue()).decode()


def img(path_or_name, cap, width="100%"):
    if path_or_name in FIGS:
        data = FIGS[path_or_name]
    elif os.path.exists(path_or_name):
        with open(path_or_name, "rb") as fh:
            data = base64.b64encode(fh.read()).decode()
    else:
        return f"<p class='miss'>(figure not available: {html.escape(path_or_name)})</p>"
    return (f"<figure><img src='data:image/png;base64,{data}' style='width:{width}'>"
            f"<figcaption>{cap}</figcaption></figure>")


def tab(df, fmt="{:.3f}", cols=None, index=False):
    if df is None:
        return "<p class='miss'>(table not available)</p>"
    d = df[cols] if cols else df
    if index:
        d = d.reset_index()

    def f(v):
        if isinstance(v, (bool, np.bool_)):
            return str(bool(v))
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else fmt.format(v)
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return html.escape(str(v))

    h = "<table><tr>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in d.columns) + "</tr>"
    for r in d.itertuples(index=False):
        h += "<tr>" + "".join(f"<td>{f(v)}</td>" for v in r) + "</tr>"
    return h + "</table>"


# ------------------------------------------------------------------ figures
def fig_convergence():
    import matplotlib.pyplot as plt
    c = t("convergence_curve")
    if c is None:
        return
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.2))
    for (sol, g), col in zip(c.groupby("solver", sort=False), ("C0", "C2", "C3")):
        g = g.drop_duplicates("itn")
        ax.plot(g.itn, g.median_abs, "o-", color=col, label=f"{sol}: median")
        ax.plot(g.itn, g.p99_abs, "s--", color=col, alpha=.7, label=f"{sol}: 99th percentile")
    ax.axvline(3000, color="0.4", lw=.8)
    ax.text(3050, 1.5, "phase19 iter_lim 3000", fontsize=7, color="0.3")
    ax.axhline(1e-4, color="0.6", lw=.6, ls=":")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("LSQR iterations")
    ax.set_ylabel("|ML - ML exact| (near events, v4.1)")
    ax.legend(fontsize=6.5, loc="lower left")
    ax.grid(alpha=.3, which="both")
    savefig(fig, "fig_convergence")


def fig_map():
    import matplotlib.pyplot as plt
    p = f"{OUT}/v4.1/v41_converged_vs_stored_events.csv.gz"
    if not os.path.exists(p):
        return
    e = pd.read_csv(p)
    s = pd.read_csv(f"{mc.V4}/cascadia_catalog_M_routeA.csv", usecols=["event_id", "evla", "evlo"])
    e = e.merge(s, on="event_id")
    e = e[e.ML_method_s == "near"]
    st = pd.read_csv(f"{mc.V4}/amp_distance_dataset_routeA.csv", usecols=["station", "stla", "stlo", "stel_m"]) \
        .drop_duplicates("station")
    st["dep"] = [mc.deployment(a, b) for a, b in zip(st.station, st.stel_m)]
    fig, ax = plt.subplots(1, 2, figsize=(11, 6.2), sharey=True)
    e = e.sort_values("dML", key=np.abs)
    sc = ax[0].scatter(e.evlo, e.evla, c=e.dML, s=1.2, cmap="RdBu_r", vmin=-0.8, vmax=0.8, rasterized=True)
    fig.colorbar(sc, ax=ax[0], shrink=.7, label="ML converged - ML stored (v4.1, near events)")
    land = st[st.dep == "land"]
    ax[1].scatter(land.stlo, land.stla, marker="^", s=6, c="0.6", label="land")
    deps = [d for d in sorted(st.dep.unique()) if d != "land"]
    cmap = plt.get_cmap("tab20")
    for i, d in enumerate(deps):
        g = st[st.dep == d]
        ax[1].scatter(g.stlo, g.stla, marker="v", s=14, color=cmap(i % 20), label=d, edgecolor="k", lw=.2)
    ax[1].legend(fontsize=6, loc="lower left", ncol=2, markerscale=1.2)
    ax[1].set_title("stations with amplitudes, by deployment", fontsize=8)
    for a_ in ax:
        a_.set_xlim(-131.5, -119.5)
        a_.set_ylim(39.0, 51.0)
        a_.set_aspect(1 / np.cos(np.radians(45)))
        a_.grid(alpha=.3)
        a_.set_xlabel("longitude")
    ax[0].set_ylabel("latitude")
    savefig(fig, "fig_map_change_v41")


def fig_gauge():
    import matplotlib.pyplot as plt
    g = t("gauge_deployments")
    if g is None:
        return
    g = g[(g.catalog == "v4.1") & (g.level == "deployment")]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    ax[0].scatter(g.stiffness, g.se_level, c="C0")
    for r in g.itertuples():
        ax[0].annotate(r.group, (r.stiffness, r.se_level), fontsize=6.5, xytext=(3, 2), textcoords="offset points")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("stiffness k (misfit increase / delta$^2$,\nrigid shift, events re-fitted)")
    ax[0].set_ylabel("formal s.e. of the deployment level vs land")
    ax[0].grid(alpha=.3)
    ax[1].scatter(g.rigid_mode_eig, g.dML_dominated, c="C3")
    for r in g.itertuples():
        ax[1].annotate(r.group, (r.rigid_mode_eig, r.dML_dominated), fontsize=6.5, xytext=(3, 2),
                       textcoords="offset points")
    ax[1].set_xscale("log")
    ax[1].axhline(0, color="0.5", lw=.6)
    ax[1].set_xlabel("k / (terms + events)\n(eigenvalue of the rigid mode in the full system)")
    ax[1].set_ylabel("median ML change of its events")
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    savefig(fig, "fig_gauge_v41")


def fig_anchors():
    import matplotlib.pyplot as plt
    a = t("comcat_anchors")
    if a is None:
        return
    a = a[a.group == "region"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, cat in zip(axs, ("v4.1", "v5.1")):
        x = a[a.solution.str.startswith(cat)].pivot(index="value", columns="solution", values="median")
        n = a[a.solution == f"{cat} stored"].set_index("value").n
        x = x.reindex([r for r in mc.REG_ORDER if r in x.index])
        i = np.arange(len(x))
        ax.bar(i - .2, x[f"{cat} stored"], .4, label="stored (3000 iterations)", color="0.6")
        ax.bar(i + .2, x[f"{cat} converged"], .4, label="converged (exact)", color="C0")
        ax.set_xticks(i, [f"{r}\n{int(n.get(r, 0))}" for r in x.index], rotation=60, fontsize=6.5)
        ax.axhline(0, color="k", lw=.6)
        ax.set_title(f"catalog {cat}: median ML - ComCat ML of the anchors", fontsize=8)
        ax.grid(alpha=.3, axis="y")
    axs[0].legend(fontsize=7)
    savefig(fig, "fig_anchors")


# ------------------------------------------------------------------ paper impact
ZBZ_NEW = f"{OUT}/clusters_zbz_v41"
MC_NEW = f"{OUT}/mc_maps_v41"


def _mc(R, reg, per="all", col="mc_best", var="all"):
    x = R[(R.region == reg) & (R.period == per) & (R.variant == var)]
    return float(x[col].iat[0]) if len(x) else np.nan


def _axial_weekly(d):
    ez = pd.read_csv(f"{d}/events_zbz.csv.gz", usecols=["region", "in_analysis", "time", "family_id"])
    ez["time"] = pd.to_datetime(ez.time, utc=True, format="ISO8601")
    t0, t1 = pd.Timestamp("2014-10-01", tz="UTC"), pd.Timestamp("2015-07-01", tz="UTC")
    ax = ez[(ez.region == "AX") & ez.in_analysis & ez.time.between(t0, t1)]
    wk = ax.set_index("time").resample("7D").size()
    fam = pd.read_csv(f"{d}/families_zbz.csv", usecols=["region", "type", "start"])
    fam["start"] = pd.to_datetime(fam.start, utc=True, format="ISO8601")
    sw = fam[(fam.region == "AX") & (fam.type == "swarm")]
    mo = sw.groupby(sw.start.dt.strftime("%Y-%m")).size()
    nov = wk[(wk.index >= "2014-11-01") & (wk.index < "2014-12-01")].mean()
    mar = wk[(wk.index >= "2015-03-08") & (wk.index < "2015-03-29")].mean()
    er = wk.max()
    cal = ax.set_index("time").resample("W-SUN").size().max()      # calendar weeks (the text's 1,021)
    post = wk[wk.index > wk.idxmax()].head(4).tolist()
    return (f"Nov 2014 {nov:.0f}/wk, mid-Mar 2015 {mar:.0f}/wk; eruption week {cal:,} (calendar week, Mon-Sun; "
            f"the text's quantity) or {er:,} (7-day bins of Fig. 6A); next four 7-day bins {post}; swarm families "
            f"Nov 2014 {mo.get('2014-11', 0)}, Mar 2015 {mo.get('2015-03', 0)}, Apr 2015 {mo.get('2015-04', 0)}")


def _bl_swarms(d):
    f = pd.read_csv(f"{d}/families_zbz.csv")
    s_ = f[(f.region == "BL") & (f.type == "swarm")]
    fc = pd.read_csv(f"{d}/faults_v0_cluster_counts.csv").set_index("surface_id").pct_swarm
    strands = "; ".join(f"{k} {fc.get(k, np.nan):.1f}" for k in ("BL004a", "BL006a", "BL015a", "BL011b", "BL012"))
    return (f"{len(s_)} BL swarms, median duration90 {24 * s_.duration90_days.median():.1f} h, extent "
            f"{s_.extent_km.median():.1f} km", f"swarm % of events above Mc: {strands}")


def _validation(d):
    v = pd.read_csv(f"{d}/validation_sequences.csv")
    keep = {"2015-06-01 Mw 5.8, 5.4, 5.9 Blanco triplet": "June 2015 Blanco family", "2015-04-24 Axial Seamount eruption onset":
            "Axial eruption family", "2012-02/04 Mw 5.8 and 6.0 Blanco": "2012 Blanco Mw 5.8/6.0 family"}
    out = []
    for lab, short in keep.items():
        r = v[v.label == lab]
        if len(r):
            r = r.iloc[0]
            out.append(f"{short}: id {r.family_id}, {r.type}, n {r.n} (above Mc), n_all {r.n_all}")
    types = "; ".join(f"{r.label[:24]} {r.type}" for r in v.itertuples())
    return " | ".join(out), types


def paper_impact():
    """Every paper item that uses Route A M (inventory of the Overleaf text at 8def2ea4b5 and the
    repository on main), with the stored and the converged value, or the action."""
    rows = []
    runs = t("phase19_runs")
    c41 = runs.set_index("run").loc["v4.1 converged"]
    Ro = pd.read_csv(f"{PAPER_MC}/mc_regional.csv")
    Rn = pd.read_csv(f"{MC_NEW}/mc_regional.csv") if os.path.exists(f"{MC_NEW}/mc_regional.csv") else None
    No = pd.read_csv(f"{PAPER_MC}/mc_nodes_by_region.csv")
    Nn = pd.read_csv(f"{MC_NEW}/mc_nodes_by_region.csv") if os.path.exists(f"{MC_NEW}/mc_nodes_by_region.csv") else None
    Go = pd.read_csv(f"{PAPER_MC}/mc_grid.csv")
    Gn = pd.read_csv(f"{MC_NEW}/mc_grid.csv") if os.path.exists(f"{MC_NEW}/mc_grid.csv") else None
    zo, zn = PAPER_ZBZ, ZBZ_NEW
    have_z = os.path.exists(f"{zn}/regions_zbz.csv")

    def add(loc, item, old, new, action="update"):
        rows.append(dict(location=loc, item=item, stored=old, converged=new, action=action))

    # ---- Results, catalog paragraph; Text S5; Methods
    add("main.tex 77-78; Text S5 (supp 174-181); Methods 435-450", "ComCat anchors and scatter",
        "3,416 anchors, scatter 0.21", f"3,416 anchors, scatter {c41.anchor_std:.2f} (offset +0.413 -> {c41.anchor_offset:+.3f})")
    add("main.tex 77-78; Text S5", "events with a magnitude", "130,769 (main) / 130,767 (Text S5)",
        "130,769 (unchanged)", "unchanged; Text S5 should say 130,769 (v4.1)")
    add("Methods 435-450 (not in text)", "M_type counts (v4.0 run, before patch)", "ML 130,652 / Mw 80 / Mw_cal 35",
        f"ML {int(c41.M_type_ML):,} / Mw {int(c41.M_type_Mw)} / Mw_cal {int(c41.M_type_Mw_cal)}",
        "update if quoted (v4.1 adds 3 Mw)")
    add("not in text (D-B041 input)", "Mw_cal offset (near / mapped)", "+0.04 / +0.97",
        f"{c41.mw_cal_near:+.2f} / {c41.mw_cal_mapped:+.2f}", "note")
    ten = t("tensors")
    def tv(sol, ev, cls):
        x = ten[(ten.solution == sol) & (ten.events == ev) & (ten.Mw_class == cls)]
        return float(x.ML_minus_Mw.iat[0]) if len(x) else np.nan
    add("not in text (driver.log; D-B041)", "ML - tensor Mw, all events: Mw 3-4.5 / 4.5-5.5 / 5.5-7.5",
        f"{tv('v4.1 stored', 'all', '3-4.5'):+.2f} / {tv('v4.1 stored', 'all', '4.5-5.5'):+.2f} / {tv('v4.1 stored', 'all', '5.5-7.5'):+.2f}",
        f"{tv('v4.1 converged', 'all', '3-4.5'):+.2f} / {tv('v4.1 converged', 'all', '4.5-5.5'):+.2f} / {tv('v4.1 converged', 'all', '5.5-7.5'):+.2f}",
        "note (near events only: +0.08 -> -0.02 at Mw 3-4.5)")
    add("main.tex 429 (Overleaf bullet)", "matched moment tensors used for the ML-Mw check", "112 (catalog v3 count)",
        "147 (v4.1)", "fix the count")
    if Rn is not None:
        on_o = [_mc(Ro, r) for r in ("WA", "PU", "ON", "OR", "MT")]
        on_n = [_mc(Rn, r) for r in ("WA", "PU", "ON", "OR", "MT")]
        of_o = [_mc(Ro, r) for r in ("EN", "GO", "BL")]
        of_n = [_mc(Rn, r) for r in ("EN", "GO", "BL")]
        add("main.tex 79-84", "pooled Mc: onshore and forearc (WA, PU, ON, OR, MT)",
            f"{min(on_o):.1f}-{max(on_o):.1f} ({', '.join(f'{v:.1f}' for v in on_o)})",
            f"{min(on_n):.1f}-{max(on_n):.1f} ({', '.join(f'{v:.1f}' for v in on_n)})")
        add("main.tex 79-84", "pooled Mc: offshore plate boundaries (EN, GO, BL)",
            f"{min(of_o):.1f}-{max(of_o):.1f} ({', '.join(f'{v:.1f}' for v in of_o)})",
            f"{min(of_n):.1f}-{max(of_n):.1f} ({', '.join(f'{v:.1f}' for v in of_n)})")
        add("main.tex 79-84", "offshore pooled Mc (OFF) / all", f"{_mc(Ro, 'OFF'):.1f} / {_mc(Ro, 'ALL'):.1f}",
            f"{_mc(Rn, 'OFF'):.1f} / {_mc(Rn, 'ALL'):.1f}", "note")
        add("main.tex 85-91", "Blanco Mc_best 2011-2012 -> 2013-2015",
            f"{_mc(Ro, 'BL', '2011-2012'):.1f} -> {_mc(Ro, 'BL', '2013-2015'):.1f}",
            f"{_mc(Rn, 'BL', '2011-2012'):.1f} -> {_mc(Rn, 'BL', '2013-2015'):.1f}")
        imp = [(_mc(Ro, r, '2011-2012') - _mc(Ro, r, '2013-2015'), _mc(Rn, r, '2011-2012') - _mc(Rn, r, '2013-2015'))
               for r in ("EN", "BL", "GO")]
        add("main.tex 85-91", "offshore Mc improvement 2011-12 -> 2013-15 (EN, BL, GO)",
            ", ".join(f"{a:+.1f}" for a, _ in imp), ", ".join(f"{b:+.1f}" for _, b in imp))
        for reg, per in (("BL", "2010"), ("BL", "2016"), ("MT", "2016")):
            add("main.tex 85-93", f"rate of M >= 2.5, {reg} {per} / 2013-2015", f"{_mc(Ro, reg, per, 'rr25'):.2f}",
                f"{_mc(Rn, reg, per, 'rr25'):.2f}", "update (absolute threshold)")
        ax_o = Ro[(Ro.region == "AX") & (Ro.period == "all") & (Ro.variant == "all")].iloc[0]
        ax_n = Rn[(Rn.region == "AX") & (Rn.period == "all") & (Rn.variant == "all")].iloc[0]
        add("Text S7 (supp 241-259)", "Axial pooled: GFT sigma; MAXC; node 10-90%",
            f"sigma {ax_o.sigma_boot:.2f}; MAXC {ax_o.mc_maxc:.1f}; nodes 0.4-1.0",
            f"sigma {ax_n.sigma_boot:.2f}; MAXC {ax_n.mc_maxc:.1f}; nodes "
            f"{float(Nn[(Nn.region == 'AX') & (Nn.period == 'all')]['10%'].iat[0]):.1f}-"
            f"{float(Nn[(Nn.region == 'AX') & (Nn.period == 'all')]['90%'].iat[0]):.1f}")
    if Nn is not None:
        def med(N, r):
            return float(N[(N.region == r) & (N.period == "all")]["50%"].iat[0])
        add("main.tex 79-84", "node medians, Puget Lowland and offshore Washington (PU, WA)",
            f"{med(No, 'PU'):.1f}, {med(No, 'WA'):.1f}", f"{med(Nn, 'PU'):.1f}, {med(Nn, 'WA'):.1f}")
        add("main.tex 79-84", "highest nodes (western Blanco, northern Gorda)",
            f"max {Go.mc_best.max():.1f}; GO max {float(No[(No.region == 'GO') & (No.period == 'all')]['max'].iat[0]):.1f}",
            f"max {Gn.mc_best.max():.1f} (western Blanco); GO max "
            f"{float(Nn[(Nn.region == 'GO') & (Nn.period == 'all')]['max'].iat[0]):.1f}",
            "rewrite: 'highest on the western Blanco transform (2.7-2.8)'")
        add("main.tex 79-84", "lowest nodes (Mount St. Helens, Axial)",
            f"ON min {float(No[(No.region == 'ON') & (No.period == 'all')]['min'].iat[0]):.1f}, AX min "
            f"{float(No[(No.region == 'AX') & (No.period == 'all')]['min'].iat[0]):.1f}",
            f"ON min {float(Nn[(Nn.region == 'ON') & (Nn.period == 'all')]['min'].iat[0]):.1f}, AX min "
            f"{float(Nn[(Nn.region == 'AX') & (Nn.period == 'all')]['min'].iat[0]):.1f}", "check (0.0-0.3 holds)")
        v_o = Go.dropna(subset=["mc_maxc", "mc_mbs", "mc_best"])
        v_n = Gn.dropna(subset=["mc_maxc", "mc_mbs", "mc_best"])
        f_ = lambda v: (((v.mc_maxc >= v.mc_mbs) & (v.mc_maxc >= v.mc_best)).mean(), (v.mc_mbs - v.mc_maxc).mean(),
                        (v.mc_best - v.mc_maxc).mean())
        a_, b_ = f_(v_o), f_(v_n)
        add("Text S7", "MAXC+0.2 most conservative (share of nodes); MBS - MAXC; best - MAXC (same criterion both)",
            f"{100 * a_[0]:.0f}%; {a_[1]:+.2f}; {a_[2]:+.2f} (text: 84%, -0.19, -0.14)",
            f"{100 * b_[0]:.0f}%; {b_[1]:+.2f}; {b_[2]:+.2f}", "update (text criterion may differ: compare in the same way)")
        add("Text S7", "QC-only minus all, Mc_best per node", "median 0.0, IQR +-0.1", "median 0.0, IQR +-0.1", "unchanged")
    for name, old, new in (("Fig. 2a", "fig2a_mc_three_methods.png", "fig1_mc_three_methods.png"),
                           ("Fig. 2b", "fig2b_mc_periods.png", "fig3_mc_periods.png")):
        add("paper_sciadv/figures", name, old, f"{MC_NEW}/{new}", "replace (remade with v4_mc_maps.py on the converged v4.1 M)")
    add("supplementary Figs. S3-S7", "mc_maps fig1-fig5", "data/catalog_v4/mc_maps/fig*.png", f"{MC_NEW}/fig*.png",
        "replace (fig4_regional_fmd still draws the stored clustering Mc line: rerun after the clustering)")
    add("Table S6", "Mc / b / rate-ratio table", "data/catalog_v4/mc_maps/mc_regional.csv", f"{MC_NEW}/mc_regional.csv", "replace")
    add("Fig. 1 A; Fig. 3 B-D; perspective_mendocino_fault.png", "symbol size by M (msize = 0.012 x 2.2^(M-1))",
        "stored M", "converged M: Blanco symbols about 1.7x smaller in diameter, JdF 1.3x, Gorda/Mendocino 1.2x",
        "remake (workflow/08_figures/v4_figures.py reads the Route A file from its own repo root)")
    add("Fig. 4", "Blanco strands", "loads M, does not use it", "-", "no change")
    add("Fig. S8 (catalog comparison), Fig. 7, Figs. S1-S2", "-", "no Route A M", "-", "no change")
    # ---- clustering
    if have_z:
        ro, rn = pd.read_csv(f"{zo}/regions_zbz.csv").set_index("region"), pd.read_csv(f"{zn}/regions_zbz.csv").set_index("region")
        so, sn = pd.read_csv(f"{zo}/setting_stats.csv").set_index("region"), pd.read_csv(f"{zn}/setting_stats.csv").set_index("region")
        add("main.tex 384-385; Text S6", "events kept by the per-region Mc", f"{int(ro.n_analysis.sum()):,}",
            f"{int(rn.n_analysis.sum()):,}")
        add("Text S6", "clustering Mc range (per region, MAXC+0.2, largest yearly)",
            f"{ro.Mc.min():.2f}-{ro.Mc.max():.2f}", f"{rn.Mc.min():.2f}-{rn.Mc.max():.2f}")
        add("Table S5; Fig. 5", "clustering Mc per region", "; ".join(f"{r} {v:.2f}" for r, v in ro.Mc.items()),
            "; ".join(f"{r} {v:.2f}" for r, v in rn.Mc.items()))
        add("Table S5; Fig. 5", "analysis-set events per region", "; ".join(f"{r} {v:,}" for r, v in ro.n_analysis.items()),
            "; ".join(f"{r} {v:,}" for r, v in rn.n_analysis.items()))
        add("main.tex 234-243", "background share AX, BL, GO ('about half')",
            ", ".join(f"{ro.pct_background[r]:.1f}" for r in ("AX", "BL", "GO")),
            ", ".join(f"{rn.pct_background[r]:.1f}" for r in ("AX", "BL", "GO")))
        add("main.tex 234-243", "families >= 5 that are swarms, AX, BL, GO ('two thirds')",
            ", ".join(f"{so.pct_swarm_of_ge5[r]:.1f}" for r in ("AX", "BL", "GO")),
            ", ".join(f"{sn.pct_swarm_of_ge5[r]:.1f}" for r in ("AX", "BL", "GO")))
        add("main.tex 234-243", "Mendocino triple junction: swarm share of families >= 5; events Mc+1..2 without offspring",
            f"{so.pct_swarm_of_ge5['MT']:.1f}%; {so.pct_Mc1to2_no_offspring['MT']:.1f}%",
            f"{sn.pct_swarm_of_ge5['MT']:.1f}%; {sn.pct_Mc1to2_no_offspring['MT']:.1f}%")
        add("main.tex 234-243", "background share WA, OR, PU, ON ('68-84%')",
            ", ".join(f"{ro.pct_background[r]:.1f}" for r in ("WA", "OR", "PU", "ON")),
            ", ".join(f"{rn.pct_background[r]:.1f}" for r in ("WA", "OR", "PU", "ON")))
        try:
            a1, b1 = _bl_swarms(zo)
            a2, b2 = _bl_swarms(zn)
            add("main.tex 245-250", "Blanco swarms: duration and extent ('4 hours', '4 km')", a1, a2)
            add("main.tex 245-250, 341-343", "swarm share on West Blanco / Surveyor / Cascadia Depression strands ('33-45%')",
                b1, b2)
        except Exception as ex:
            print("bl swarms:", ex)
        try:
            add("main.tex 258-265 (Overleaf); Fig. 6A", "Axial weekly events above Mc and swarm families per month "
                "(text: about 60 -> about 300 per week, 1,021 in the eruption week, below 15 per week within four weeks)",
                _axial_weekly(zo), _axial_weekly(zn))
        except Exception as ex:
            print("axial weekly:", ex)
        try:
            v1, t1 = _validation(zo)
            v2, t2 = _validation(zn)
            add("main.tex 224-232, 245-250, 270-275; Text S6", "validation families (IDs renumber)", v1, v2)
            add("main.tex 224-232; Text S6", "six large earthquakes are mainshock-aftershock; eruption is one swarm", t1, t2,
                "check")
        except Exception as ex:
            print("validation:", ex)
        sh_o = pd.read_csv(f"{zo}/shuffle_test.csv").groupby("region").mean(numeric_only=True)
        sh_n = pd.read_csv(f"{zn}/shuffle_test.csv").groupby("region").mean(numeric_only=True)
        add("Text S6", "bimodality (Ashman D) real; shuffled", f"{ro.ashman_D.min():.2f}-{ro.ashman_D.max():.2f}; "
            f"{sh_o.ashman_D.min():.2f}-{sh_o.ashman_D.max():.2f}", f"{rn.ashman_D.min():.2f}-{rn.ashman_D.max():.2f}; "
            f"{sh_n.ashman_D.min():.2f}-{sh_n.ashman_D.max():.2f}")
        add("Text S6", "shuffled events below the real threshold (frac_below_eta0, mean of 3 shuffles)",
            "; ".join(f"{r} {100 * v:.0f}%" for r, v in sh_o.frac_below_eta0.items()),
            "; ".join(f"{r} {100 * v:.0f}%" for r, v in sh_n.frac_below_eta0.items()),
            "update (text: 3-17%, Axial 34%)")
        cn = pd.read_csv(f"{zn}/completeness_b_df.csv").set_index("region").Mc
        co = pd.read_csv(f"{zo}/completeness_b_df.csv").set_index("region").Mc
        if Rn is not None:
            below = [r for r in cn.index if any(cn[r] < _mc(Rn, r, col=c) - 1e-6 for c in ("mc_maxc", "mc_mbs", "mc_best"))]
            add("Text S7", "regions where the clustering Mc is below one of the three map methods",
                "BL (MBS 2.5 > 2.45)", ", ".join(f"{r} ({cn[r]:.2f} vs MAXC {_mc(Rn, r, col='mc_maxc'):.1f}, MBS "
                                                 f"{_mc(Rn, r, col='mc_mbs'):.1f}, best {_mc(Rn, r):.1f})" for r in below) or "none")
    add("Figs. 5, 6; Table S5; family IDs 270, 346, 1758, 1803", "clustering products", "data/catalog_v4/clusters_zbz",
        ZBZ_NEW, "regenerate figures (v4_figures.py clusters/short) and re-identify the family IDs quoted in the text")
    add("Overleaf main.tex 265 (repo main.tex 243)", "'... to almost none above Mc 0.75' (Axial clustering Mc)", "0.75",
        "0.85", "update")
    add("Overleaf main.tex 435-438 (repo main.tex 405-408), Methods", "'inverted for event size, distance attenuation and "
        "station terms'", "wrong for the near-event ML (127,000 of 130,769 events): the Hutton-Boore distance term is "
        "fixed, only event and station-phase terms are inverted; the solver is not mentioned",
        "near ML: event + station-phase terms with the Hutton & Boore (1987) distance term fixed, solved exactly "
        "(one gauge row); the distance-attenuation inversion (phase3) is used only for the 3,522 'mapped' events",
        "correct the Methods sentence (bullet)")
    add("Fig. 6B (West Blanco swarm, June 2015)", "symbols coloured and sized by M (cpt 1.5-5.0); stars 'M x.x' for "
        "M >= 5", "stored M", "converged M: the ML events of the swarm drop by about 0.67; the three tensor Mw (5.80, "
        "5.44, 5.89) do not change", "remake (colour bar range may need 1.0-5.0)")
    add("main.tex 291-294 (Fig. 6D)", "'only events near the coast or above M 4 remain'", "stored M",
        "Mendocino ML -0.19 (v4.1 near)", "recheck on the remade Fig. 6D")
    add("main.tex 352-363", "'the magnitude scale ... the same everywhere'", "claim", "holds better: offshore anchor gap "
        "+0.12 -> +0.01", "keep; cite the converged solver in Methods")
    add("Overleaf main.tex 126", "comparison with Zhang & Niksejel 2025 (offshore Mc 1.5)", "ours 1.8-2.2",
        "ours 1.6-1.9 (EN 1.6, BL 1.6, GO 1.9 pooled)", "update the comparison")
    add("Table S4 rows GO004, GO011", "catalog maximum M", "M5.6, M4.1 (catalog magnitude column, not Route A)",
        "Route A: 144373 Mw 5.45; 103612 ML 3.85 -> see per-event file", "check (not Route A values)")
    add("Text S5", "Hutton-Boore named without \\cite", "-", "Hutton1987 is in references.bib", "cite")
    add("workflow/06_analysis/v51_clusters_zbz.py:86, :90 (main)", "v5.1 clustering reads the stored v5.1 magnitudes "
        "(MAG = /wd1/.../v5.1_analysis/mag/cascadia_catalog_v5.1_magnitudes.csv) and the stored mc_b table (MCB)",
        "stored", "converged: /wd1/mdenolle_data/magnitude_converged/v5.1/cascadia_catalog_v5.1_magnitudes_converged.csv; "
        "data/magnitude_converged/mc_b/mc_b_regional.csv", "repoint and rerun (PR body)")
    add("workflow/06_analysis/v51_struct_common.py:29 (main)", "v5.1 structure analysis MAG", "stored v5.1 magnitudes",
        "converged v5.1 table", "repoint and rerun (PR body)")
    add("workflow/06_analysis/v51_interp_common.py:21, :81 (main)", "v5.1 interpretation reads "
        "cascadia_catalog_v5.1_magnitudes_slim.csv.gz from V51_MAG (default the old worktree wt_a51_mag)",
        "stored slim table", "data/catalog_v5/v5.1_analysis/mag/cascadia_catalog_v5.1_magnitudes_converged_slim.csv.gz "
        "(new name, next to the stored one)", "repoint and rerun (PR body)")
    add("workflow/10_tomography/params.yaml catalog_versions.magnitude_table_files (main)", "catalog v5.x bundles take M from "
        "the v4.1 Route A file", "data/magnitude_v4/cascadia_catalog_M_routeA.csv (main checkout)",
        "/wd1/mdenolle_data/magnitude_converged/v4.1/cascadia_catalog_M_routeA_converged.csv", "switch when adopted (PR body)")
    add("Data availability; atlas; catalog v5 bundles; Zenodo", "Route A magnitudes in the released products", "stored",
        "converged", "rebuild when the PI adopts the converged file (not done here)")
    return pd.DataFrame(rows)


def mc_regions():
    return ["AX", "EN", "BL", "GO", "MT", "WA", "OR", "PU", "ON"]


# ------------------------------------------------------------------ HTML
CSS = """
:root { --fg:#1d2329; --bg:#ffffff; --mut:#5b6670; --line:#d9dee3; --acc:#1f5f99; --hl:#fff6d6; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --fg:#e4e8ec; --bg:#14181c; --mut:#9aa5ae;
  --line:#2e363d; --acc:#7ab3e6; --hl:#3a3416; } }
:root[data-theme="dark"] { --fg:#e4e8ec; --bg:#14181c; --mut:#9aa5ae; --line:#2e363d; --acc:#7ab3e6; --hl:#3a3416; }
body { font: 14px/1.5 system-ui, sans-serif; color: var(--fg); background: var(--bg); max-width: 1100px;
  margin: 0 auto; padding: 16px; }
h1 { font-size: 1.5em; } h2 { font-size: 1.2em; border-bottom: 1px solid var(--line); padding-top: .6em; }
h3 { font-size: 1.02em; }
table { border-collapse: collapse; font-size: 12px; margin: .4em 0 1em; display: block; overflow-x: auto; }
th, td { border: 1px solid var(--line); padding: 2px 6px; text-align: right; white-space: nowrap; }
th { background: var(--hl); } td:first-child, th:first-child { text-align: left; }
figure { margin: .6em 0 1.2em; } img { max-width: 100%; background: #fff; }
figcaption { font-size: 12px; color: var(--mut); }
.box { border-left: 3px solid var(--acc); padding: 4px 12px; background: var(--hl); }
.miss { color: var(--mut); font-style: italic; } code { font-size: 12px; }
"""


def main():
    os.makedirs(FIGD, exist_ok=True)
    for f in (fig_convergence, fig_map, fig_gauge, fig_anchors):
        try:
            f()
        except Exception as ex:  # a missing input must not stop the report
            print(f"figure {f.__name__} failed: {ex}")
    pi = paper_impact()
    pi.to_csv(f"{SMALL}/paper_impact.csv", index=False)
    from mag_converged_text import sections
    body = sections(t, tab, img, pi)
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Converged Route A magnitudes</title><style>{CSS}</style></head><body>{body}</body></html>")
    fo = f"{SMALL}/magnitude_converged_report.html"
    with open(fo, "w") as fh:
        fh.write(page)
    print(f"wrote {fo} ({len(page) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

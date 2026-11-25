#!/usr/bin/env python3
"""Figures and the self-contained HTML report of the v5.1 structure-seismicity analysis (v51_struct.py tables).

    python v51_struct_report.py
writes data/catalog_v5/v5.1_analysis/struct/fig_*.png, tests_summary.csv and v51_structure_seismicity_report.html.
"""
from __future__ import annotations

import base64
import html
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "utils"))
from v51_struct_common import BIG, BLANCO_SEGMENTS, OUT, ZONES, Model  # noqa: E402

try:
    from paper_style import use
    use()
except Exception:  # noqa: BLE001
    pass

INK, MUTED, GRID = "#222222", "#666666", "#d9d9d9"
TYPE_COL = {"swarm": "#D55E00", "mainshock-aftershock": "#0072B2", "multiplet": "#009E73", "background": "#8C8C8C"}
REAL, NULLC, ART = "#0072B2", "#8C8C8C", "#D55E00"
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200})


def rd(name):
    return pd.read_csv(OUT / name)


def png(fig, name):
    p = OUT / f"fig_{name}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


# ------------------------------------------------------------------ FDR
def bh(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    o = np.argsort(pv)
    r = pv[o] * len(pv) / (np.arange(len(pv)) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty_like(pv)
    out[o] = np.minimum(r, 1)
    q[ok] = out
    return q


def tests_summary():
    rows = []
    t1 = rd("t1_swarm_vs_ma.csv")
    for _, r in t1[(t1.set == "resPS") & t1["var"].isin(["VpVs_res", "dVpVs_res", "dVp_res", "dVs_res"])].iterrows():
        rows.append(dict(question="Q1 swarms vs mainshock-aftershock", zone=r.zone, test=f"{r.contrast}: {r['var']}",
                         n=f"{r.n_swarm} vs {r.n_other} families, {r.n_blocks} blocks", effect=r["diff"],
                         ci_lo=r.ci_lo, ci_hi=r.ci_hi, null_lo=r.null_lo, null_hi=r.null_hi, p=r.p_perm))
    t2 = rd("t2_density.csv")
    for _, r in t2[t2.events == "all"].iterrows():
        rows.append(dict(question="Q2 seismicity density", zone=r.zone,
                         test=f"event-weighted coverage-matched {r['var']}",
                         n=f"{r.n_events} events, {r.n_nodes} nodes, {r.n_blocks} blocks", effect=r.event_mean_res,
                         ci_lo=r.res_ci_lo, ci_hi=r.res_ci_hi, null_lo=r.res_null_lo, null_hi=r.res_null_hi,
                         p=r.res_p))
        rows.append(dict(question="Q2 seismicity density", zone=r.zone,
                         test=f"stratified Spearman(count, {r['var']}_res)",
                         n=f"{r.n_nodes} nodes, {r.n_strata} strata", effect=r.rho_cov, ci_lo=r.rho_ci_lo,
                         ci_hi=r.rho_ci_hi, null_lo=r.rho_null_lo, null_hi=r.rho_null_hi, p=r.rho_p))
    for _, r in t2[t2.events == "all, field smoothed 20 km"].iterrows():
        rows.append(dict(question="Q2 seismicity density, resolved scale", zone=r.zone,
                         test=f"event-weighted coverage-matched {r['var']}, field smoothed 20 km",
                         n=f"{r.n_events} events, {r.n_nodes} nodes, {r.n_blocks} blocks", effect=r.event_mean_res,
                         ci_lo=r.res_ci_lo, ci_hi=r.res_ci_hi, null_lo=r.res_null_lo, null_hi=r.res_null_hi,
                         p=r.res_p))
    tb = rd("t2_bvalue.csv")
    for _, r in tb[tb.events == "all above Mc"].iterrows():
        rows.append(dict(question="Q2 b-value", zone=r.zone, test=f"b(high {r['var']}_res) - b(low)",
                         n=f"{r.n_high} vs {r.n_low} events, {r.n_blocks} blocks", effect=r.db_high_minus_low,
                         ci_lo=r.ci_lo, ci_hi=r.ci_hi, null_lo=r.null_lo, null_hi=r.null_hi, p=r.p_perm))
    t3 = rd("t3_depth_extent.csv")
    for _, r in t3[t3.sd_real.notna()].iterrows():
        rows.append(dict(question="Q3 depth extent", zone=r.zone, test=f"{r.test} (smaller than null?)",
                         n=f"{int(r.n_columns)} columns", effect=r.sd_real, ci_lo=r.sd_ci_lo, ci_hi=r.sd_ci_hi,
                         null_lo=r.sd_null_lo, null_hi=r.sd_null_hi, p=r.p_smaller))
    t4 = rd("t4_pooled.csv")
    for _, r in t4[t4.align_mean.notna()].iterrows():
        rows.append(dict(question="Q4 fault surfaces", zone="all resolved", test=f"alignment |n.grad {r['var']}| "
                         f"({r.subset} surfaces)", n=f"{int(r.n_surfaces)} surfaces, {int(r.n_blocks)} blocks",
                         effect=r.align_mean, ci_lo=r.align_ci_lo, ci_hi=r.align_ci_hi, null_lo=r.align_null_lo,
                         null_hi=r.align_null_hi, p=r.align_p_greater))
    t5 = rd("t5_blanco_segment_corr.csv")
    for _, r in t5.iterrows():
        rows.append(dict(question="Q5 Blanco segments", zone="BL", test=f"Spearman({r.x}, {r.y})",
                         n=f"{r.n_segments} segments", effect=r.spearman, p=r.p_exact))
    df = pd.DataFrame(rows)
    df["q_BH"] = bh(df.p.values)
    # reference value of "no effect": 0 for differences, anomalies and correlations (Q1, Q2, Q5), the middle of the
    # null range for spreads and alignments (Q3, Q4). A test counts as passed only when q < 0.05 AND its block-
    # bootstrap interval excludes that reference (the permutation p is too optimistic for autocorrelated fields).
    ref = np.where(df.question.str.startswith(("Q3", "Q4")), 0.5 * (df.null_lo + df.null_hi), 0.0)
    df["ref"] = ref
    df["passed"] = (df.q_BH < 0.05) & ((df.ci_lo > ref) | (df.ci_hi < ref))
    df.to_csv(OUT / "tests_summary.csv", index=False, float_format="%.4g")
    return df


# ------------------------------------------------------------------ figures
def fig_coverage(model: Model, ev: pd.DataFrame, cov: pd.DataFrame):
    depths = [3, 15, 18, 29]
    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1.15])
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]),
           fig.add_subplot(gs[1, 1])]
    e = ev[ev.located & ev.is_earthquake]
    for a, z in zip(axs, depths):
        k = int(np.argmin(np.abs(model.dep - z)))
        lo = 0.5 * (model.dep[k - 1] + model.dep[k])
        hi = 0.5 * (model.dep[k] + model.dep[k + 1])
        ez = e[(e.depth >= lo) & (e.depth < hi)]
        a.plot(ez.lon, ez.lat, ",", color="#b0b0b0", rasterized=True)
        cat = np.where(model.mPS[k], 2, np.where(model.mP[k], 1, np.nan))
        a.pcolormesh(model.lon, model.lat, cat, cmap=matplotlib.colors.ListedColormap(["#56B4E9", "#0072B2"]),
                     vmin=0.5, vmax=2.5, shading="nearest", alpha=0.75)
        a.set_title(f"{model.dep[k]:g} km node ({lo:.1f}-{hi:.1f} km)", fontsize=7)
        a.set_xlim(-131.5, -120.5)
        a.set_ylim(39.3, 50.3)
        a.set_aspect(1 / np.cos(np.radians(45)))
        a.tick_params(labelsize=6)
    axs[0].text(0.02, 0.02, "light blue: Vp resolved\ndark blue: Vp and Vs resolved\ngrey: events in the layer",
                transform=axs[0].transAxes, fontsize=5.5, va="bottom")
    ax = fig.add_subplot(gs[:, 2])
    r = cov[cov.group.str.startswith("region: ")].copy()
    r["region"] = r.group.str.replace("region: ", "")
    r = r.sort_values("n")
    y = np.arange(len(r))
    ax.barh(y + 0.2, r.pct_resP, height=0.38, color="#56B4E9", label="Vp resolved")
    ax.barh(y - 0.2, r.pct_resPS, height=0.38, color="#0072B2", label="Vp/Vs resolved")
    for yy, (n, p1) in enumerate(zip(r.n, r.pct_resP)):
        ax.text(max(p1, 0) + 1, yy + 0.2, f"{p1:.1f}% of {n:,}", va="center", fontsize=5.5, color=MUTED)
    ax.set_yticks(y, r.region, fontsize=6)
    ax.set_xlabel("located earthquakes in resolved cells (%)")
    ax.set_xlim(0, 60)
    ax.legend(loc="lower right", fontsize=6)
    fig.tight_layout()
    return png(fig, "coverage")


def fig_blanco_maps(model: Model, ev: pd.DataFrame):
    e = ev[ev.located & ev.is_earthquake & (ev.region == "Blanco")]
    fig, ax = plt.subplots(3, 2, figsize=(7.2, 6.4), sharex=True, sharey=True)
    for c, z in enumerate((15, 18)):
        k = int(np.argmin(np.abs(model.dep - z)))
        lo, hi = 0.5 * (model.dep[k - 1] + model.dep[k]), 0.5 * (model.dep[k] + model.dep[k + 1])
        ez = e[(e.depth >= lo) & (e.depth < hi)]
        for r, (var, mask, cmap, vlim, lab) in enumerate((
                ("dVp", model.mP, "RdBu", 8, "Vp change from start (%)"),
                ("dVs", model.mPS, "RdBu", 8, "Vs change from start (%)"),
                ("VpVs", model.mPS, "PuOr_r", None, "Vp/Vs"))):
            a = ax[r, c]
            arr = np.where(mask[k], getattr(model, var)[k], np.nan)
            if vlim is None:
                im = a.pcolormesh(model.lon, model.lat, arr, cmap=cmap, vmin=1.55, vmax=1.95, shading="nearest")
            else:
                im = a.pcolormesh(model.lon, model.lat, arr, cmap=cmap, vmin=-vlim, vmax=vlim, shading="nearest")
            a.plot(ez.lon, ez.lat, ".", ms=0.6, color=INK, alpha=0.35, rasterized=True)
            for kk, _, l0, l1 in BLANCO_SEGMENTS:
                a.axvline(l0, color=MUTED, lw=0.4, ls=":")
                if r == 0 and c == 0:
                    a.text(0.5 * (l0 + l1), 44.9, kk, ha="center", fontsize=5.5, color=MUTED)
            a.set_xlim(-130.7, -126.4)
            a.set_ylim(42.6, 45.0)
            a.set_aspect(1 / np.cos(np.radians(44)))
            if r == 0:
                a.set_title(f"{z} km node (events {lo:.1f}-{hi:.1f} km)", fontsize=7)
            if c == 1:
                cb = fig.colorbar(im, ax=a, shrink=0.8, pad=0.02)
                cb.set_label(lab, fontsize=6)
                cb.ax.tick_params(labelsize=5.5)
    fig.tight_layout()
    return png(fig, "blanco_maps")


def fig_t1(model):
    f = rd("t1_families_resPS.csv")
    t1 = rd("t1_swarm_vs_ma.csv")
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for a, v, lab in zip(ax, ("VpVs_res", "dVp_res", "dVs_res"),
                         ("Vp/Vs, coverage-matched", "Vp change, coverage-matched (%)",
                          "Vs change, coverage-matched (%)")):
        for i, t in enumerate(["swarm", "mainshock-aftershock", "multiplet"]):
            d = f[f.type == t]
            x = i + np.random.default_rng(1).uniform(-0.15, 0.15, len(d))
            a.scatter(x, d[v], s=np.clip(d.n_res, 4, 60), color=TYPE_COL[t], alpha=0.7, edgecolor="white", lw=0.3)
            a.plot([i - 0.25, i + 0.25], [d[v].median()] * 2, color=INK, lw=1.2)
        r = t1[(t1.set == "resPS") & (t1["var"] == v) & (t1.contrast == "swarm - mainshock-aftershock")].iloc[0]
        a.set_title(f"swarm - MA: {r['diff']:+.3g} [{r.ci_lo:+.2g}, {r.ci_hi:+.2g}], p {r.p_perm:.2f}", fontsize=6)
        a.set_xticks([0, 1, 2], ["swarm", "MA", "multiplet"], fontsize=6.5)
        a.set_ylabel(lab, fontsize=6.5)
        a.axhline(0, color=GRID, lw=0.6, zorder=0)
    fig.suptitle("Blanco families with at least 3 members in Vp/Vs-resolved cells (marker size: members there)",
                 fontsize=7)
    fig.tight_layout()
    return png(fig, "t1_families")


def fig_t2(t2, syn):
    d = t2[t2.events == "all"].copy()
    d["label"] = d.zone + " " + d["var"]
    order = ["BL dVp", "BL Vp", "BL dVs", "BL Vs", "BL dVpVs", "BL VpVs", "MTJ dVp", "MTJ Vp", "WA dVp", "WA Vp"]
    d = d.set_index("label").loc[[o for o in order if o in set(d.label)]]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(width_ratios=[1, 1]))
    # (a) percent variables
    pc = [l for l in d.index if l.split()[1] in ("dVp", "dVs", "dVpVs")]
    for a, labels, title, unit in ((ax[0], pc, "changes from the start model", "%"),
                                   (ax[1], [l for l in d.index if l not in pc], "absolute values",
                                    "km/s or ratio")):
        y = np.arange(len(labels))[::-1]
        for yy, l in zip(y, labels):
            r = d.loc[l]
            a.plot([r.res_null_lo, r.res_null_hi], [yy, yy], color=NULLC, lw=5, alpha=0.45, solid_capstyle="butt")
            a.plot([r.res_ci_lo, r.res_ci_hi], [yy, yy], color=REAL, lw=1.2)
            a.plot(r.event_mean_res, yy, "o", color=REAL, ms=4)
            sm = t2[(t2.events == "all, field smoothed 20 km") & (t2.zone + " " + t2["var"] == l)]
            if len(sm):
                sm = sm.iloc[0]
                a.plot([sm.res_ci_lo, sm.res_ci_hi], [yy + 0.25, yy + 0.25], color=REAL, lw=0.8, ls=(0, (2, 1)))
                a.plot(sm.event_mean_res, yy + 0.25, "o", mfc="white", mec=REAL, ms=3.5)
            z, v = l.split()
            s = syn[(syn.zone == z) & (syn["var"] == v)]
            if len(s):
                a.plot(s.event_mean_res, [yy - 0.22] * len(s), "|", color=ART, ms=5, mew=1)
        a.axvline(0, color=GRID, lw=0.8, zorder=0)
        a.set_yticks(y, labels, fontsize=6.5)
        a.set_title(title, fontsize=7)
        a.set_xlabel(f"event-weighted coverage-matched anomaly ({unit})", fontsize=6.5)
    ax[0].text(0.02, 0.02, "filled dot + line: catalog, block-bootstrap 95%\nopen dot + dashed: field smoothed to "
               "the resolved scale (20 km)\ngrey band: randomized catalogs 95%\n"
               "orange ticks: checkerboard inversion errors", transform=ax[0].transAxes, fontsize=5.5)
    ax[0].texts[-1].set_position((0.0, -0.22))
    ax[0].texts[-1].set_va("top")
    fig.tight_layout()
    return png(fig, "t2_effects")


def fig_t2b(tb):
    d = tb.copy()
    d["label"] = d.zone + " " + d["var"] + np.where(d.events == "background", " (bg)", "")
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    y = np.arange(len(d))[::-1]
    for yy, (_, r) in zip(y, d.iterrows()):
        ax.plot([r.null_lo, r.null_hi], [yy, yy], color=NULLC, lw=5, alpha=0.45, solid_capstyle="butt")
        ax.plot([r.ci_lo, r.ci_hi], [yy, yy], color=REAL, lw=1.2)
        ax.plot(r.db_high_minus_low, yy, "o", color=REAL, ms=3.5)
    ax.axvline(0, color=GRID, lw=0.8, zorder=0)
    ax.set_yticks(y, d.label, fontsize=5.5)
    ax.set_xlabel("b(high anomaly) - b(low anomaly)", fontsize=6.5)
    ax.set_title("b-value split at the coverage-matched anomaly (events above Mc of their box)", fontsize=6.5)
    fig.tight_layout()
    return png(fig, "t2_bvalue")


def fig_t3():
    col = rd("t3_columns.csv")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    a = ax[0]
    cols = {"MT": "#D55E00", "PU": "#0072B2", "ON": "#8C8C8C", "WA": "#009E73", "OR": "#CC79A7", "GO": "#E69F00"}
    for b, d in col[np.isfinite(col.slab_km)].groupby("box"):
        a.scatter(d.slab_km, d.d95, s=6, color=cols.get(b, "#444444"), label=f"{b} ({len(d)})", alpha=0.8)
    lim = [0, 110]
    a.plot(lim, lim, color=MUTED, lw=0.6, ls="--")
    a.set_xlabel("Slab2 depth at the column (km)")
    a.set_ylabel("D95 of well-constrained depths (km)")
    a.legend(fontsize=5.5, loc="upper left")
    a.set_title("depth extent vs slab (columns of 0.125 x 0.1 deg, >= 30 events)", fontsize=6.5)
    a = ax[1]
    d = col[np.isfinite(col.Vp_d90)]
    a.scatter(d.d90, d.Vp_d90, s=12, color="#0072B2")
    for _, r in d.iterrows():
        a.text(r.d90, r.Vp_d90, f" {r.zone}", fontsize=5)
    a.set_xlabel("D90 (km)")
    a.set_ylabel("Vp at D90 (km/s, resolved nodes)")
    a.set_title(f"columns with resolved Vp bracketing D90: {len(d)}", fontsize=6.5)
    fig.tight_layout()
    return png(fig, "t3_depth")


def fig_t4():
    s = rd("t4_surfaces.csv")
    pool = rd("t4_pooled.csv")
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    vars_ = ["Vp", "dVp", "VpVs", "dVpVs"]
    rng = np.random.default_rng(2)
    for i, v in enumerate(vars_):
        d = s[s["var"] == v]
        x = i + rng.uniform(-0.18, 0.18, len(d))
        ax.scatter(x, d.align_med, s=np.clip(d.n_vertices / 6, 4, 50), c=np.where(d.stable, "#D55E00", "#8C8C8C"),
                   alpha=0.8, edgecolor="white", lw=0.3)
        r = pool[(pool["var"] == v) & (pool.subset == "all")].iloc[0]
        ax.fill_between([i - 0.3, i + 0.3], r.align_null_lo, r.align_null_hi, color=NULLC, alpha=0.3, lw=0)
        ax.plot([i - 0.3, i + 0.3], [r.align_mean] * 2, color=INK, lw=1.2)
    ax.set_xticks(range(len(vars_)), [f"{v}\n({(s['var'] == v).sum()} surfaces)" for v in vars_], fontsize=6.5)
    ax.set_ylabel("median |n_h . g_h| / |g_h| per surface", fontsize=6.5)
    ax.set_ylim(0, 1.02)
    ax.set_title("fault normal vs horizontal gradient; orange: stable surfaces; black bar: mean; "
                 "grey: 95% of random strikes", fontsize=6)
    fig.tight_layout()
    return png(fig, "t4_alignment")


def fig_t5():
    seg = rd("t5_blanco_segments.csv")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for a, (yv, ylo, yhi, lab) in zip(ax, (("swarm_share", "swarm_ci_lo", "swarm_ci_hi",
                                             "swarm events / clustered events (above Mc)"),
                                            ("b_value", None, None, "b-value (events above Mc)"))):
        a.errorbar(seg.VpVs_events, seg[yv],
                   xerr=[seg.VpVs_events - seg.VpVs_ci_lo, seg.VpVs_ci_hi - seg.VpVs_events],
                   yerr=None if ylo is None else [seg[yv] - seg[ylo], seg[yhi] - seg[yv]],
                   fmt="o", color="#0072B2", ms=4, lw=0.8, capsize=2)
        for _, r in seg.iterrows():
            a.text(r.VpVs_events, r[yv], f"  {r.segment}", fontsize=6)
        a.set_xlabel("median Vp/Vs at events in resolved cells")
        a.set_ylabel(lab, fontsize=6.5)
    fig.suptitle("Blanco segments (west to east: WBD, SD, CD, BR, GD); n = 5, block-bootstrap 95% intervals",
                 fontsize=7)
    fig.tight_layout()
    return png(fig, "t5_segments")


# ------------------------------------------------------------------ html
CSS = """
:root{--bg:#fbfbfa;--ink:#1d1d1b;--muted:#5d5c57;--rule:#d8d7d2;--card:#ffffff;--accent:#0b5cad}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1a19;--ink:#ecebe6;--muted:#b4b3ab;
--rule:#3a3a37;--card:#222220;--accent:#7fb2ec}}
:root[data-theme="dark"]{--bg:#1a1a19;--ink:#ecebe6;--muted:#b4b3ab;--rule:#3a3a37;--card:#222220;--accent:#7fb2ec}
body{background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;
max-width:1000px;margin:0 auto;padding:16px}
h1{font-size:1.5em;margin:.4em 0}h2{font-size:1.2em;margin-top:1.8em;border-bottom:1px solid var(--rule)}
h3{font-size:1.02em;margin-top:1.3em}
table{border-collapse:collapse;font-size:12.5px;margin:.6em 0;display:block;overflow-x:auto;max-width:100%}
th,td{border-bottom:1px solid var(--rule);padding:3px 7px;text-align:right;white-space:nowrap}
th{text-align:center;color:var(--muted);font-weight:600}td:first-child,th:first-child{text-align:left}
img{max-width:100%;background:#fff;border:1px solid var(--rule);border-radius:4px}
.cap{color:var(--muted);font-size:13px;margin:.2em 0 1.2em}
.box{background:var(--card);border:1px solid var(--rule);border-radius:6px;padding:10px 14px;margin:1em 0}
code{font-size:12.5px}li{margin:.25em 0}
"""


def table_html(df: pd.DataFrame, floatfmt="{:.3g}", max_rows=200):
    df = df.head(max_rows)
    h = ["<table><thead><tr>"] + [f"<th>{html.escape(str(c))}</th>" for c in df.columns] + ["</tr></thead><tbody>"]
    for _, r in df.iterrows():
        cells = []
        for v in r.values:
            if isinstance(v, (float, np.floating)):
                cells.append("" if not np.isfinite(v) else floatfmt.format(v))
            else:
                cells.append(html.escape(str(v)))
        h.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    h.append("</tbody></table>")
    return "".join(h)


def fig_html(p: Path, cap: str):
    return f'<img src="data:image/png;base64,{b64(p)}" alt="{html.escape(cap[:80])}"><p class="cap">{cap}</p>'


def main():
    model = Model()
    ev = pd.read_parquet(BIG / "events_struct_v51.parquet")
    cov = rd("coverage_fraction.csv")
    czd = rd("coverage_by_zone_depth.csv")
    t1 = rd("t1_swarm_vs_ma.csv")
    t1e = rd("t1_event_swarm_share.csv")
    t2 = rd("t2_density.csv")
    tb = rd("t2_bvalue.csv")
    t3 = rd("t3_depth_extent.csv")
    t3s = rd("t3_slab_position_by_box.csv")
    t4p = rd("t4_pooled.csv")
    seg = rd("t5_blanco_segments.csv")
    segc = rd("t5_blanco_segment_corr.csv")
    slabp = rd("t5_slab_position_dvp.csv")
    areas = rd("t5_areas_dvp.csv")
    syn = rd("synth_artifact.csv")
    syns = rd("synth_artifact_summary.csv")
    mc = rd("mc_location_summary.csv")
    v4 = rd("v4_counterpart_external_vs.csv")
    ts = tests_summary()
    fam_ps = rd("t1_families_resPS.csv")
    nd = pd.read_parquet(BIG / "nodes_v031.parquet")
    b = nd[(nd.zone == "BL") & nd.resP & nd.depth.isin([15.0, 18.0])]
    s = nd[(nd.zone == "BL") & nd.resPS]
    nodes_bl = dict(vp5=b.Vp.quantile(.05), vp50=b.Vp.median(), vp95=b.Vp.quantile(.95), dvp5=b.dVp.quantile(.05),
                    dvp95=b.dVp.quantile(.95), r5=s.VpVs.quantile(.05), r50=s.VpVs.median(),
                    r95=s.VpVs.quantile(.95), nps=len(s))
    figs = dict(coverage=fig_coverage(model, ev, cov), blanco=fig_blanco_maps(model, ev), t1=fig_t1(model),
                t2=fig_t2(t2, syn), t2b=fig_t2b(tb), t3=fig_t3(), t4=fig_t4(), t5=fig_t5())
    from v51_struct_text import build_html
    page = build_html(locals())
    out = OUT / "v51_structure_seismicity_report.html"
    out.write_text(page)
    print("wrote", out, f"{out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

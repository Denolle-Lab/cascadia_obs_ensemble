"""Fig. 2: seismicity-style fingerprint of the plate-boundary segments, as a dot matrix: one row per segment (grouped
by boundary type, colour and marker of the type), one column per metric on a shared row axis: events, Mc and b with
their 95 % bootstrap intervals, the family mix of the nearest-neighbour clustering, the swarm share with its interval,
the depth below the solid surface (p10-p50-p90) and the burstiness. Values from seismicity_metrics.csv (pb_seismicity.py);
structure class from structure_feasibility.csv. Segments of type "other" (not a plate boundary) are left out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import paper_style as S
from paper_v2 import data as D

SHORT = {"AX": "Axial Seamount", "JR": "JdF Ridge", "EN": "Endeavour", "EXR": "Explorer Ridge", "GR": "Gorda Ridge",
         "BL": "Blanco", "MFZ": "Mendocino FZ", "SOV": "Sovanco", "JP": "JdF plate", "GP": "Gorda plate",
         "EXP": "Explorer plate", "MTJ": "Triple junction", "GS": "Gorda slab", "FA": "Forearc", "SIF": "Interface zone",
         "IS": "JdF intraslab", "PU": "Puget-Georgia", "MSH": "Mount St Helens", "CAS": "Cascade arc"}
MIX = [("pct_swarm", "swarm", "#2b2b2b"), ("pct_MA", "mainshock-aftershock", "#7d7d7d"),
       ("pct_multiplet", "multiplet", "#b3b3b3"), ("pct_bg", "background", "#e1e1e1")]
CLASS_FILL = {"resolved": S.INK, "marginal": "#9a9a9a", "not resolved": "white"}


def short_name(seg, name):
    return SHORT.get(seg, str(name).split(" (")[0])


def make(cfg):
    s = D.segments(cfg)
    s = s[s.type != "other"].copy()
    s["_t"] = s.type.map({t: i for i, t in enumerate(S.TYPE_ORDER)})
    s["_o"] = np.arange(len(s))
    s = s.sort_values(["_t", "_o"]).reset_index(drop=True)
    n = len(s)
    y = np.arange(n)[::-1].astype(float)                    # first segment on top
    # columns: key, header, unit, kind, width (cm)
    cols = [("n_events", "Events", "tier A-C", "log", 1.55),
            ("Mc", "Mc", "95 % CI", "ci", 1.6),
            ("b", "b-value", "95 % CI", "ci", 1.8),
            ("mix", "Family mix", "% of events ≥ Mc", "mix", 2.45),
            ("pct_swarm", "Swarm share", "%, 95 % CI", "ci", 1.8),
            ("depth", "Depth below surface", "km, p10–p50–p90", "depth", 2.3),
            ("burst_all", "Burstiness", "events ≥ Mc", "dot", 1.4)]
    W = S.COL2
    lab_w, gap, top, bottom = 3.0, 0.24, 1.0, 1.42
    row_h = 0.40
    H = top + n * row_h + bottom
    fig = S.fig_cm(W, H)
    x = lab_w
    yl = (-0.6, n - 0.4)
    axes = []
    groups = [(t, s.index[s.type == t]) for t in S.TYPE_ORDER if (s.type == t).any()]
    for key, head, unit, kind, w in cols:
        ax = S.axes_cm(fig, x, bottom, w, n * row_h)
        ax.set_ylim(*yl)
        S.clean(ax, grid_axis="x", left=False)
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.tick_params(axis="x", length=2.0, pad=1.2)
        # group bands: alternate type groups on a faint band
        for gi, (t, idx) in enumerate(groups):
            if gi % 2 == 1:
                ax.axhspan(y[idx].min() - 0.5, y[idx].max() + 0.5, color="#f4f4f4", lw=0, zorder=0)
        for i, r in s.iterrows():
            col = S.TYPE_COL[r.type]
            mk = S.TYPE_MK[r.type]
            ms = 4.2 * S.TYPE_MS[mk]
            yy = y[i]
            if kind == "mix":
                left = 0.0
                for k, _, c in MIX:
                    v = float(r.get(k, np.nan))
                    if not np.isfinite(v) or v <= 0:
                        continue
                    ax.add_patch(Rectangle((left, yy - 0.28), v, 0.56, facecolor=c, edgecolor="white", lw=0.6,
                                           zorder=2))
                    left += v
                if not np.isfinite(r.get("pct_bg", np.nan)):
                    ax.text(2, yy, "–", va="center", fontsize=S.FS["small"], color=S.INK2)
                continue
            if kind == "depth":
                if r.depth_class in ("good", "fair", "poor") and np.isfinite(r.bsf_med):
                    ax.plot([r.bsf_p10, r.bsf_p90], [yy, yy], color=col, lw=1.3, alpha=0.55, zorder=2,
                            solid_capstyle="round")
                    hollow = r.depth_class != "good"
                    ax.plot(r.bsf_med, yy, mk, ms=ms, mfc="white" if hollow else col, mec=col if hollow else "white",
                            mew=0.8 if hollow else 0.4, zorder=3)
                else:
                    ax.text(1.0, yy, "no depth-good events", va="center", fontsize=S.FS["small"] - 0.5,
                            color=S.INK2)
                continue
            v = float(r.get(key, np.nan))
            if not np.isfinite(v):
                ax.text(0.03, yy, "–", transform=ax.get_yaxis_transform(), va="center", fontsize=S.FS["small"],
                        color=S.INK2)
                continue
            if kind == "ci":
                lo, hi = (r.get(f"{key}_lo", np.nan), r.get(f"{key}_hi", np.nan))
                if np.isfinite(lo) and np.isfinite(hi):
                    ax.plot([lo, hi], [yy, yy], color=col, lw=1.3, alpha=0.55, zorder=2, solid_capstyle="round")
            ax.plot(v, yy, mk, ms=ms, mfc=col, mec="white", mew=0.4, zorder=3)
        if kind == "log":
            ax.set_xscale("log")
            ax.set_xlim(40, 8e4)
            ax.set_xticks([1e2, 1e3, 1e4])
            ax.set_xticklabels(["10²", "10³", "10⁴"])
            ax.xaxis.set_minor_locator(__import__("matplotlib").ticker.NullLocator())
        elif key == "Mc":
            ax.set_xlim(0, 2.6)
            ax.set_xticks([0, 1, 2])
        elif key == "b":
            ax.set_xlim(0.6, 1.6)
            ax.set_xticks([0.75, 1.0, 1.25, 1.5])
            ax.axvline(1.0, color="#bdbdbd", lw=0.5, zorder=1)
        elif kind == "mix":
            ax.set_xlim(0, 100)
            ax.set_xticks([0, 50, 100])
            ax.get_xticklabels()[-1].set_horizontalalignment("right")
        elif key == "pct_swarm":
            ax.set_xlim(0, 50)
            ax.set_xticks([0, 20, 40])
        elif kind == "depth":
            ax.set_xlim(0, 62)
            ax.set_xticks([0, 20, 40, 60])
        elif key == "burst_all":
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.5, 1])
            ax.set_xticklabels(["0", "0.5", "1"])
        # header
        fig.text((x + w / 2) / W, (bottom + n * row_h + 0.42) / H, head, ha="center", va="bottom",
                 fontsize=S.FS["base"], fontweight="bold")
        fig.text((x + w / 2) / W, (bottom + n * row_h + 0.12) / H, unit, ha="center", va="bottom",
                 fontsize=S.FS["small"], color=S.INK2)
        axes.append(ax)
        x += w + gap
    # row labels: type marker, code, short name, structure class square
    lab = S.axes_cm(fig, 0.1, bottom, lab_w - 0.2, n * row_h)
    lab.set_xlim(0, 1)
    lab.set_ylim(*yl)
    lab.axis("off")
    for gi, (t, idx) in enumerate(groups):
        if gi % 2 == 1:
            lab.axhspan(y[idx].min() - 0.5, y[idx].max() + 0.5, xmin=0.0, xmax=1.0, color="#f4f4f4", lw=0, zorder=0)
    for i, r in s.iterrows():
        mk = S.TYPE_MK[r.type]
        lab.plot(0.05, y[i], mk, ms=4.2 * S.TYPE_MS[mk], mfc=S.TYPE_COL[r.type], mec="white", mew=0.4)
        lab.text(0.12, y[i], r.seg, va="center", fontsize=S.FS["base"], fontweight="bold")
        lab.text(0.33, y[i], short_name(r.seg, r["name"]), va="center", fontsize=S.FS["small"], color=S.INK)
        cl = r.get("class", "not resolved")
        lab.add_patch(Rectangle((0.925, y[i] - 0.17), 0.05, 0.34, facecolor=CLASS_FILL.get(cl, "white"),
                                edgecolor=S.INK, lw=0.5, zorder=3))
    fig.text((0.1 + (lab_w - 0.2) * 0.95) / W, (bottom + n * row_h + 0.12) / H, "structure", ha="center", va="bottom",
             fontsize=S.FS["small"], color=S.INK2)
    # legends along the bottom: types (row 1); family mix and structure class (row 2)
    hs = S.type_handles(ms=4.2)
    fig.legend(handles=hs, loc="lower left", bbox_to_anchor=(lab_w / W, 0.5 / H), ncol=len(hs), frameon=False,
               fontsize=S.FS["small"], handletextpad=0.25, columnspacing=1.1, borderaxespad=0)
    hm = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="none", label=lb) for _, lb, c in MIX]
    hcl = [Line2D([], [], ls="", marker="s", ms=4.2, mfc=CLASS_FILL[c], mec=S.INK, mew=0.5, label=f"{c}")
           for c in ("resolved", "marginal", "not resolved")]
    fig.legend(handles=hm, loc="lower left", bbox_to_anchor=(lab_w / W, 0.05 / H), ncol=4, frameon=False,
               fontsize=S.FS["small"], handletextpad=0.35, columnspacing=1.1, borderaxespad=0, handlelength=1.0,
               handleheight=0.8)
    fig.legend(handles=hcl, loc="lower right", bbox_to_anchor=((W - 0.2) / W, 0.05 / H), ncol=3, frameon=False,
               fontsize=S.FS["small"], handletextpad=0.25, columnspacing=1.1, borderaxespad=0,
)
    size = S.save(fig, cfg["out"], "fig2_fingerprint")
    segs = ", ".join(s.seg)
    cap = (f"Seismicity style of {n} plate-boundary segments of catalog {cfg['label']} (tier A-C earthquakes; flagged "
           "events and the waveform-review P1 set left out), grouped by boundary type (colour and marker). Events: "
           "number of events. Mc: maximum curvature + 0.2; b: Aki-Utsu estimate above Mc; bars: 95 % intervals over "
           "500 bootstrap draws. Family mix: shares of the events above the clustering Mc that are background, in "
           "mainshock-aftershock families, in swarms and in multiplets (nearest-neighbour clustering of the catalog); "
           "swarm share with its 95 % bootstrap interval over units. Depth below the solid surface (ground or "
           "seafloor) of depth-good events: p10-p90 bar and median (open symbol: depth class fair, median σz 3-6 km; "
           "no symbol: fewer than 50 depth-good events). Burstiness (σ − μ)/(σ + μ) of the inter-event times of events "
           "above the segment Mc. Square at the right of each label: structure class of model v0.4.2 near the "
           f"segment (filled resolved, grey marginal, open not resolved). Segments: {segs}.")
    return [dict(name="fig2_fingerprint", files=["fig2_fingerprint.pdf", "fig2_fingerprint.png"], size_cm=size,
                 caption=cap, before=["fig5_clustering.png"], before_plan=["fig_fingerprint.png"], inputs=["seismicity_metrics", "structure_feasibility"])]

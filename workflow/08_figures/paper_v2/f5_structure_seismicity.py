"""Fig. 5: structure against seismicity, with the controls.

(A-D) Seismicity bottom against the Vp 7.5 km/s level per model column (depth_vs_level_columns.csv: D90 of the
      depth-good events of a 0.125 x 0.1 deg column with >= 20 of them, against the depth of the first resolved Vp
      crossing of 7.5 km/s in that column), one panel per segment with enough columns; Spearman rho and the permutation
      p of sd(D90 - level) from depth_vs_level.csv (pb_struct_seis.py).
(E)   Share of the depth-good events below the 7.5 km/s level per segment, with the range over the ensemble members
      (5-95 %, each member's own level).
(F-H) Depth-matched structure contrasts (Vp, Vs, Vp/Vs) of the resolved and marginal segments against their controls:
      +-2 sigma of the real ensemble (bar), the synthetic spurious-contrast S95 (grey band; circularity control, D-B111)
      and, once scored, the null-truth N95 (hatched band, D-B112). Filled symbols pass the rule.
The null-truth hook: when the configuration's `null_truth_contrasts` table exists, its N95 bands are drawn and
`claimed` decides the fill; until then the fill is the D-B111 verdict and the caption says the null truth is pending.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import paper_style as S
from paper_v2 import data as D

CMK = {"near-far": "o", "across": "^", "along": "s"}
COFF = {"near-far": 0.0, "across": 0.27, "along": -0.27}
PANEL_SEGS = ["GS", "MTJ", "BL", "MFZ"]          # order of the column panels (those with >= 15 columns are drawn)


def make(cfg):
    typ = D.seg_type(cfg)
    segt = D.segments(cfg).set_index("seg")
    cols = pd.read_csv(cfg.path("depth_vs_level_columns"))
    dl = pd.read_csv(cfg.path("depth_vs_level"))
    dl = dl[dl.level == "z75"].set_index("seg")
    ct = D.contrasts(cfg)
    null_scored = bool(ct.null_scored.iloc[0])

    W = S.COL2
    H = 15.4
    fig = S.fig_cm(W, H)
    # ---------------- A-D: D90 against the level
    segs = [s for s in PANEL_SEGS if s in dl.index and dl.loc[s, "n_columns"] >= 15]
    pw, ph, x0, ytop = 2.95, 2.95, 1.05, H - 0.55
    lim = (5, 45)
    letters = iter("ABCDEFGHIJ")
    for i, sg in enumerate(segs):
        ax = S.axes_cm(fig, x0 + i * (pw + 0.28), ytop - ph, pw, ph)
        c = cols[(cols.seg == sg) & np.isfinite(cols.z75)]
        col = S.TYPE_COL[typ.get(sg, "other")]
        mk = S.TYPE_MK[typ.get(sg, "other")]
        ax.plot(lim, lim, color="#bdbdbd", lw=0.6, zorder=1)
        ax.scatter(c.z75, c.d90, s=np.clip(np.sqrt(c.n) * 0.9, 3, 26), c=col, marker=mk, edgecolors="white",
                   linewidths=0.35, zorder=3, alpha=0.9)
        ax.set_xlim(*lim)
        ax.set_ylim(lim[1], lim[0])
        ax.set_xticks([10, 20, 30, 40])
        ax.set_yticks([10, 20, 30, 40])
        S.clean(ax)
        ax.set_aspect("equal")
        r = dl.loc[sg]
        claim = r.p_smaller <= 0.01 and r.rho > 0.3
        ax.text(0.97, 0.97, f"{sg} · {int(r.n_columns)} columns\nρ = {r.rho:.2f} · p = {r.p_smaller:.3f}",
                transform=ax.transAxes, fontsize=S.FS["small"], ha="right", va="top",
                fontweight="bold" if claim else "normal")
        if i == 0:
            ax.set_ylabel("D90 of the column (km)")
        else:
            ax.tick_params(labelleft=False)
        ax.set_xlabel("7.5 km/s level (km)")
        S.letter(ax, next(letters), dx=-0.85 if i == 0 else -0.2, dy=0.08)
    # ---------------- E: share below the level per segment
    ex = x0 + len(segs) * (pw + 0.28) + 0.75
    ew = W - ex - 0.25
    ax = S.axes_cm(fig, ex, ytop - ph, ew, ph)
    sb = segt[(segt.frac_with_z75 >= 0.3) & np.isfinite(segt.frac_below_z75)].copy()
    sb = sb[sb.type != "other"]
    sb["_t"] = sb.type.map({t: k for k, t in enumerate(S.TYPE_ORDER)})
    sb = sb.sort_values("_t")
    yy = np.arange(len(sb))[::-1]
    for y, (sg, r) in zip(yy, sb.iterrows()):
        col = S.TYPE_COL[r.type]
        mk = S.TYPE_MK[r.type]
        ax.plot([100 * r.frac_below_z75_mem_p05, 100 * r.frac_below_z75_mem_p95], [y, y], color=col, lw=1.4,
                alpha=0.55, solid_capstyle="round")
        ax.plot(100 * r.frac_below_z75, y, mk, ms=4.5 * S.TYPE_MS[mk], mfc=col, mec="white", mew=0.4)
        ax.text(-3, y, sg, ha="right", va="center", fontsize=S.FS["small"], fontweight="bold")
    ax.set_ylim(-0.6, len(sb) - 0.4)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.tick_params(axis="y", left=False, labelleft=False)
    S.clean(ax, grid_axis="x", left=False)
    ax.set_xlabel("events below 7.5 km/s (%)")
    S.letter(ax, next(letters), dx=-0.85, dy=0.08)

    # ---------------- F-H: contrasts against their controls
    keep = segt[segt["class"].isin(["resolved", "marginal"]) & (segt.type != "other")].copy()
    keep["_t"] = keep.type.map({t: k for k, t in enumerate(S.TYPE_ORDER)})
    keep["_o"] = np.arange(len(keep))
    sg_list = keep.sort_values(["_t", "_o"]).index.tolist()
    n = len(sg_list)
    yb, hb = 1.6, H - 0.55 - ph - 1.25 - 1.6
    vars_ = [("Vp", "ΔVp (km/s)", (-0.75, 0.75)), ("Vs", "ΔVs (km/s)", (-0.5, 0.5)), ("VpVs", "ΔVp/Vs", (-0.07, 0.07))]
    cw = (W - 1.4 - 0.25 - 2 * 0.3) / 3
    for j, (v, lab, xl) in enumerate(vars_):
        ax = S.axes_cm(fig, 1.4 + j * (cw + 0.3), yb, cw, hb)
        d = ct[ct["var"] == v]
        for i, sg in enumerate(sg_list):
            y0 = n - 1 - i
            if i % 2 == 0:
                ax.axhspan(y0 - 0.5, y0 + 0.5, color="#f6f6f6", lw=0, zorder=0)
            for cn, off in COFF.items():
                r = d[(d.seg == sg) & (d.contrast == cn)]
                if r.empty or not np.isfinite(r.iloc[0].D_real):
                    continue
                r = r.iloc[0]
                y = y0 + off
                h = 0.2
                if np.isfinite(r.get("S95", np.nan)):
                    ax.add_patch(Rectangle((-r.S95, y - h / 2), 2 * r.S95, h, facecolor="#d4d4d4", edgecolor="none",
                                           zorder=1))
                if null_scored and np.isfinite(r.get("N95", np.nan)):
                    ax.add_patch(Rectangle((-r.N95, y - h / 2), 2 * r.N95, h, facecolor="none", edgecolor="#8c8c8c",
                                           hatch="//////", lw=0, zorder=1.2))
                col = S.TYPE_COL[typ.get(sg, "other")]
                passed = bool(r.claimed) if null_scored else bool(r.survives)
                ax.plot([r.D_real - 2 * r.sigma_ens, r.D_real + 2 * r.sigma_ens], [y, y], color=col, lw=0.9,
                        zorder=2, solid_capstyle="butt")
                ax.plot(r.D_real, y, CMK[cn], ms=3.6, mfc=col if passed else "white", mec=col, mew=0.8, zorder=3)
        ax.axvline(0, color="#9a9a9a", lw=0.5, zorder=1)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xlim(*xl)
        ax.set_yticks(range(n))
        ax.set_yticklabels(sg_list[::-1] if j == 0 else [])
        ax.tick_params(axis="y", length=0, pad=2)
        for t in ax.get_yticklabels():
            t.set_fontweight("bold")
        S.clean(ax, grid_axis="x", left=False)
        ax.set_xlabel(lab)
        S.letter(ax, next(letters), dx=-0.85 if j == 0 else -0.2, dy=0.08)
    # legend for F-H
    hl = [Line2D([], [], ls="", marker=CMK[c], ms=3.8, mfc="white", mec=S.INK, mew=0.8, label=c) for c in CMK]
    hl += [Line2D([], [], color=S.INK, lw=0.9, label="±2σ ensemble"),
           Rectangle((0, 0), 1, 1, facecolor="#d4d4d4", edgecolor="none", label="±S95 synthetic")]
    if null_scored:
        hl.append(Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="#8c8c8c", hatch="//////", lw=0,
                            label="±N95 null truth"))
    hl.append(Line2D([], [], ls="", marker="o", ms=3.8, mfc=S.INK, mec=S.INK, label="passes (filled)"))
    fig.legend(handles=hl, loc="lower center", bbox_to_anchor=(0.5, 0.5 / H), ncol=len(hl),
               frameon=False, fontsize=S.FS["small"], handletextpad=0.3, columnspacing=1.0, borderaxespad=0,
               handlelength=1.3)
    if not null_scored:
        fig.text((W - 0.25) / W, (yb + hb + 0.15) / H, "null-truth control (D-B112) pending", ha="right",
                 va="bottom", fontsize=S.FS["small"], color=S.INK2)
    th = S.type_handles(ms=4.0)
    fig.legend(handles=th, loc="lower center", bbox_to_anchor=(0.5, 0.08 / H), ncol=len(th), frameon=False,
               fontsize=S.FS["small"], handletextpad=0.25, columnspacing=1.1, borderaxespad=0)
    size = S.save(fig, cfg["out"], "fig5_structure_seismicity")
    n_pass = int(ct.claimed.sum()) if null_scored else int(ct.survives.sum())
    gs = dl.loc["GS"] if "GS" in dl.index else None
    cap = ("Structure against seismicity. (" + "-".join([chr(65), chr(64 + len(segs))]) + ") Seismicity bottom against "
           "the Vp 7.5 km/s level per model column (0.125° × 0.1°, at least 20 depth-good tier A-C events and a "
           "resolved level): D90, the 90th percentile of event depth, against the depth of the first Vp crossing of "
           "7.5 km/s between two resolved nodes; symbol size grows with the number of events; grey line 1:1. ρ: "
           "Spearman rank correlation; p: share of 1000 permutations of the levels among the segment's columns with a "
           "smaller spread of D90 − level" + (f" (GS: ρ = {gs.rho:.2f}, p = {gs.p_smaller:.3f})" if gs is not None else "")
           + f". ({chr(65 + len(segs))}) Share of the depth-good events below the 7.5 km/s level, segments with a "
           "resolved level for at least 30 % of their events; bar: 5-95 % over the ensemble members, each with its own "
           f"level. ({chr(66 + len(segs))}-{chr(68 + len(segs))}) Depth-matched contrasts of Vp, Vs and Vp/Vs of "
           "model v0.4.2 for the resolved and marginal segments: near−far (nodes within 5 km of the segment's faults "
           "and seismicity minus nodes 10-20 km away), across (the two sides of the segment's strike line) and along "
           "(the two ends of the segment); bar: ±2σ of the same contrast over the 100 ensemble members (× 1.50 Vp, "
           "× 1.35 Vs, × 1.00 Vp/Vs); grey band: ±S95, the 95th percentile of |recovered − true| contrast over the 100 "
           "members of the synthetic calibration ensemble (D-B111)" +
           ("; hatched band: ±N95 over the null-truth members (D-B112); filled: |Δ| ≥ 2σ, ≥ S95 and ≥ N95"
            if null_scored else "; filled: |Δ| ≥ 2σ and ≥ S95 (the null-truth control, D-B112, is pending)") +
           f"; {n_pass} contrasts pass. Colour and marker of the segment type as in Fig. 2.")
    return [dict(name="fig5_structure_seismicity", files=["fig5_structure_seismicity.pdf",
                                                          "fig5_structure_seismicity.png"],
                 size_cm=size, caption=cap, before=[], before_plan=["fig_circularity.png"], contrasts=dict(n_pass=n_pass, null_scored=null_scored),
                 inputs=["depth_vs_level_columns", "depth_vs_level", "seismicity_metrics", "structure_feasibility",
                         "circularity_contrasts", "null_truth_contrasts"])]

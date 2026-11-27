#!/usr/bin/env python3
"""Q2 figures (D-B115) from the q2_analysis.py / q2_circularity.py tables. Colourblind-safe: Crameri maps (vik for
signed residuals, batlow for sequential), batlowS categorical colours with a marker per boundary type (D-B104).
Writes <Q2_BIG>/figures/q2_*.png."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "08_figures"))
import pb_common as P  # noqa: E402
import q2_analysis as A  # noqa: E402
import q2_common as Q  # noqa: E402
from scm_palettes import scm_cmap, scm_path  # noqa: E402

FIG = Q.BIG / "figures"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "font.family": "DejaVu Sans",
                     "savefig.dpi": 200})


def batlow_s():
    out = []
    for line in scm_path("batlowS").read_text().splitlines():
        t = line.split()
        if len(t) == 2 and not t[0].startswith("#") and "/" in t[1]:
            out.append([float(v) / 255 for v in t[1].split("/")])
    return np.array(out)


_S = batlow_s()
TYPE_COL = {"ridge": _S[0], "transform": _S[4], "intraplate": _S[3], "triple junction": _S[11], "slab": _S[7]}
TYPE_MK = {"ridge": "o", "transform": "s", "intraplate": "D", "triple junction": "*", "slab": "v"}
CM_RES = scm_cmap("vik", reverse=False, bad="#ffffff")


def coast(ax):
    f = Q.BIG / "coast_i.txt"
    if not f.exists():
        return
    seg = []
    for ln in f.read_text().splitlines():
        if ln.startswith(">"):
            if len(seg) > 1:
                a = np.array(seg)
                ax.plot(a[:, 0], a[:, 1], color="0.35", lw=0.4, zorder=2)
            seg = []
        else:
            seg.append([float(v) for v in ln.split()[:2]])
    if len(seg) > 1:
        a = np.array(seg)
        ax.plot(a[:, 0], a[:, 1], color="0.35", lw=0.4, zorder=2)


def traces(ax):
    pts, sub, _ = P.pb2002()
    for k, st in (("OSR", dict(color="k", lw=0.9)), ("OTF", dict(color="k", lw=0.9, ls="--"))):
        xy = pts[k]
        # draw as points (densified traces are unordered between sections)
        ax.plot(xy[:, 0], xy[:, 1], ".", ms=0.6, color="k", zorder=3)
    ax.plot(sub[:, 0], sub[:, 1], "-", color="k", lw=0.8, zorder=3)
    ax.plot(sub[::25, 0], sub[::25, 1], ">", color="k", ms=2.0, zorder=3)


def unit_type(U):
    t = U.key.map(Q.OFFSHORE).fillna("")
    return t


def fig_map(U):
    """Residual maps (frame units, segment-independent): D90 - z(600 C, HSC) and D90 - z(7.5 km/s)."""
    F = U[U.grouping.eq("frame")]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.9), constrained_layout=True, sharey=True)
    norm = BoundaryNorm(np.arange(-15, 15.1, 2.5), CM_RES.N)
    for ax, (key, title) in zip(axs, (("HSC_600", "base - 600 C isotherm (half-space)"),
                                      ("V7.5", "base - 7.5 km/s level (v0.4.2)"),
                                      ("V8.0", "base - 8.0 km/s level (v0.4.2)"))):
        coast(ax)
        traces(ax)
        r = F.D - F[key]
        for (_, u), rv in zip(F.iterrows(), r):
            lo, la = u.lon - 0.25, u.lat - 0.25
            if np.isfinite(rv):
                ax.add_patch(Rectangle((lo, la), 0.5, 0.5, facecolor=CM_RES(norm(rv)), edgecolor="0.2", lw=0.3,
                                       zorder=4, hatch=None if u.key == "oceanic" else "////"))
            else:
                ax.add_patch(Rectangle((lo, la), 0.5, 0.5, facecolor="none", edgecolor="0.55", lw=0.4, ls=":",
                                       zorder=4))
        ax.set_xlim(-131.5, -121.5)
        ax.set_ylim(39.5, 50.0)
        ax.set_aspect(1 / np.cos(np.radians(44.5)))
        ax.set_title(title)
        ax.set_xlabel("longitude")
    axs[0].set_ylabel("latitude")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=CM_RES)
    cb = fig.colorbar(sm, ax=axs, shrink=0.75, pad=0.01)
    cb.set_label("D90 minus predicted level (km); > 0: seismicity deeper")
    fig.suptitle("Base of seismicity (D90) relative to each hypothesis, 0.5 deg bins (hatched: slab frame; "
                 "dotted outline: no resolved level)", fontsize=8)
    fig.savefig(FIG / "q2_map_residuals.png")
    plt.close(fig)


def equiv_age(mod, young, old, T=600.0):
    """Single-side age whose isotherm T lies at the depth of the mean geotherm of (young, old) (transform units)."""
    z = A.z_below_top(mod, T, young, old)
    tt = np.geomspace(0.005, 200, 4000)
    zt = A.z_below_top(mod, T, tt)
    return np.interp(z, zt, tt)


def fig_age_depth(U, E):
    """Depth below the seafloor against plate age, oceanic and transform units, with cooling-model isotherms."""
    S = U[U.grouping.eq("seg") & U.ttype.isin(["oceanic", "transform"])].copy()
    S["type"] = unit_type(S)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.7), constrained_layout=True, sharey=True)
    t = np.geomspace(0.01, 40, 400)
    bat = scm_cmap("batlow")
    tcol = {350.0: bat(0.15), 600.0: bat(0.5), 800.0: bat(0.85)}
    e = E[E.frame.eq("oceanic") & E.seg.isin(list(Q.OFFSHORE))]
    for ax, mod, ls in ((axs[0], "HSC", "-"), (axs[1], "GDH1", "--")):
        tr = e.ttype_seg.eq("transform").values
        x = e.age.values.copy()
        ok = tr & np.isfinite(e.age_young.values) & np.isfinite(e.age_old.values)
        x[ok] = equiv_age(mod, e.age_young.values[ok], e.age_old.values[ok])
        ax.scatter(x, e.bsf, s=1, color="0.78", lw=0, rasterized=True, label="events (depth-reliable)")
        for T in Q.ISOTHERMS:
            ax.plot(t, A.z_below_top(mod, T, t), ls, color=tcol[T], lw=1.4, label=f"{int(T)} C isotherm")
        for ty, g in S.groupby("type"):
            xx = g.age.values.copy()
            gt = g.ttype.eq("transform").values
            xx[gt] = equiv_age(mod, g.age_young.values[gt], g.age_old.values[gt])
            ax.vlines(xx, g.Dbsf_lo95, g.Dbsf_hi95, color=TYPE_COL[ty], lw=0.7, zorder=4)
            ax.plot(xx, g.D_bsf, TYPE_MK[ty], color=TYPE_COL[ty], ms=5, mec="k", mew=0.4, label=f"D90 {ty}",
                    zorder=5)
        ax.set_xscale("log")
        ax.set_xlim(0.02, 40)
        ax.set_ylim(40, 0)
        ax.set_xlabel("thermal age (Myr)")
        ax.set_title({"HSC": "half-space cooling (Tm 1350 C)", "GDH1": "plate model GDH1 (Tm 1450 C, 95 km)"}[mod])
    axs[0].set_ylabel("depth below the seafloor (km)")
    axs[1].legend(fontsize=6, loc="lower left", ncol=2, frameon=False)
    fig.text(0.5, -0.01, "Thermal age: plate age at the epicentre (Seton et al. 2020); transforms: the single-side age "
             "whose 600 C depth equals that of the mean of the two geotherms (ages 20 km either side of the trace).",
             ha="center", va="top", fontsize=6.5)
    fig.savefig(FIG / "q2_age_depth.png", bbox_inches="tight")
    plt.close(fig)


def fig_scatter(U, G):
    """D90 against each primary predictor, frames and boundary types."""
    S = U[U.grouping.eq("seg")].copy()
    S["type"] = unit_type(S)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), constrained_layout=True)
    for ax, key, lab in zip(axs, ("HSC_600", "GDH1_600", "V7.5"),
                            ("600 C isotherm, half-space (km bsl)", "600 C isotherm, GDH1 plate (km bsl)",
                             "7.5 km/s level, v0.4.2 (km bsl)")):
        ax.fill_between([0, 70], [-3, 67], [3, 73], color="0.9", zorder=0)
        ax.plot([0, 70], [0, 70], "k-", lw=0.6)
        for ty, g in S.groupby("type"):
            ok = np.isfinite(g[key])
            sd = g[f"{key}_sd"][ok] if f"{key}_sd" in g else None
            ax.vlines(g[key][ok], g.D_lo95[ok], g.D_hi95[ok], color=TYPE_COL[ty], lw=0.5)
            if sd is not None:
                ax.hlines(g.D[ok], g[key][ok] - 2 * sd, g[key][ok] + 2 * sd, color=TYPE_COL[ty], lw=0.5)
            ax.plot(g[key][ok], g.D[ok], TYPE_MK[ty], color=TYPE_COL[ty], ms=4.5, mec="k", mew=0.3, label=ty)
        ax.set_xlim(0, 70)
        ax.set_ylim(70, 0)
        ax.set_xlabel(lab)
        ax.set_aspect("equal")
    axs[0].set_ylabel("D90 of the unit (km bsl)")
    axs[0].legend(fontsize=6, frameon=False, loc="lower left")
    fig.suptitle("Units = segment x 0.5 deg bin. Grey band: registered tolerance +-3 km. Vertical bars: 95 % of the D90 "
                 "draws (bootstrap + depth Monte Carlo);\nhorizontal bars (7.5 km/s): +-2 sd of the member level "
                 "(raw member deviations x1.50).", fontsize=7)
    fig.savefig(FIG / "q2_scatter.png")
    plt.close(fig)


def fig_forest(G):
    groups = ["frame:oceanic", "frame:slab", "ridge", "transform", "intraplate", "triple junction", "Gorda slab (GS)",
              "slab", "seg:BL", "seg:MFZ", "seg:GP", "seg:MTJ", "seg:SIF", "seg:IS", "seg:AX", "seg:GR"]
    bat = scm_cmap("batlow")
    keys = [("HSC_350", bat(0.15), "o", "350 C HSC"), ("HSC_600", bat(0.5), "o", "600 C HSC"),
            ("HSC_800", bat(0.85), "o", "800 C HSC"), ("GDH1_600", bat(0.5), "^", "600 C GDH1"),
            ("V7.5", (0.1, 0.1, 0.1), "s", "7.5 km/s"), ("V8.0", (0.6, 0.6, 0.6), "D", "8.0 km/s")]
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    ax.axvspan(-3, 3, color="0.9", zorder=0)
    ax.axvline(0, color="k", lw=0.6)
    for gi, g in enumerate(groups):
        sub = G[G.group.eq(g)].set_index("level")
        for kj, (k, col, mk, lab) in enumerate(keys):
            if k not in sub.index or not np.isfinite(sub.loc[k, "bias"]):
                continue
            y = gi + (kj - 2.5) * 0.12
            r = sub.loc[k]
            ax.hlines(y, r.bias_lo95, r.bias_hi95, color=col, lw=0.8)
            ax.plot(r.bias, y, mk, color=col, ms=4, mec="k", mew=0.3, label=lab)
        n = int(sub.n_units_all.max()) if len(sub) else 0
        ax.text(26.5, gi, f"{n}", va="center", ha="right", fontsize=6.5)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([g.replace("seg:", "segment ").replace("frame:", "frame: ") for g in groups])
    ax.invert_yaxis()
    ax.set_xlim(-15, 27)
    ax.set_xlabel("bias = median over units of (D90 - predicted level) (km); 95 % interval of the draws")
    h, lab = ax.get_legend_handles_labels()
    keep = dict(zip(lab, h))
    ax.legend(keep.values(), keep.keys(), fontsize=6.5, ncol=6, loc="upper center", bbox_to_anchor=(0.45, 1.06),
              frameon=False)
    ax.text(26.5, -0.9, "units", ha="right", fontsize=6.5)
    fig.savefig(FIG / "q2_forest.png")
    plt.close(fig)


def fig_tstar(U):
    S = U[U.grouping.eq("seg")].copy()
    S["type"] = unit_type(S)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    ax = axs[0]
    for T in Q.ISOTHERMS:
        ax.axhline(T, color="0.6", lw=0.6, ls="--")
    order = ["ridge", "transform", "intraplate", "triple junction", "slab"]
    for i, ty in enumerate(order):
        g = S[S.type.eq(ty)]
        for j, (mod, dx) in enumerate((("HSC", -0.12), ("GDH1", 0.12))):
            v = g[f"Tbase_{mod}"].values
            ok = np.isfinite(v)
            ax.scatter(np.full(ok.sum(), i + dx) + np.random.default_rng(1).uniform(-0.05, 0.05, ok.sum()), v[ok],
                       marker=TYPE_MK[ty], color=TYPE_COL[ty] if mod == "HSC" else "white", edgecolor=TYPE_COL[ty]
                       if mod == "GDH1" else "k", s=18, lw=0.6)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=20)
    ax.set_ylabel("temperature at D90 (C)")
    ax.set_title("cooling-model temperature at the base (filled HSC, open GDH1;\nslab: coldest-slab bound = lower "
                 "bound on T)")
    ax = axs[1]
    for v in Q.LEVELS:
        ax.axhline(v, color="0.6", lw=0.6, ls="--")
    for i, ty in enumerate(order):
        g = S[S.type.eq(ty)]
        v = g.Vbase.values
        ok = np.isfinite(v)
        xj = np.full(ok.sum(), i) + np.random.default_rng(2).uniform(-0.1, 0.1, ok.sum())
        ax.vlines(xj, np.fmin(g.Vbase_lo95.values[ok], g.Vbase_hi95.values[ok]),
                  np.fmax(g.Vbase_lo95.values[ok], g.Vbase_hi95.values[ok]), color=TYPE_COL[ty], lw=0.5)
        ax.plot(xj, v[ok], TYPE_MK[ty], color=TYPE_COL[ty], mec="k", mew=0.3, ms=4.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=20)
    ax.set_ylabel("Vp of v0.4.2 at D90 (km/s)")
    ax.set_title("model Vp at the base (resolved nodes only;\nbar: Vp at the 2.5-97.5 % D90 draws)")
    fig.savefig(FIG / "q2_base_T_V.png")
    plt.close(fig)


CAT_ORDER = ["temperature-limited", "composition-limited", "both consistent", "mixed", "neither",
             "H_T supported, not preferred", "H_C supported, not preferred", "H_T supported (H_C not testable)",
             "H_C supported (H_T not testable)", "H_T not supported; H_C not testable",
             "H_C not supported; H_T not testable", "not testable"]
CAT_SHORT = {"temperature-limited": "T", "composition-limited": "C", "both consistent": "T+C", "mixed": "mix",
             "neither": "none", "H_T supported, not preferred": "T?", "H_C supported, not preferred": "C?",
             "H_T supported (H_C not testable)": "T (C n/t)", "H_C supported (H_T not testable)": "C (T n/t)",
             "H_T not supported; H_C not testable": "notT (C n/t)", "H_C not supported; H_T not testable":
             "notC (T n/t)", "not testable": "n/t"}


def fig_controls(answers: dict):
    groups = ["frame:oceanic", "frame:slab", "frame:oceanic+slab", "ridge", "ridge (no AX)", "transform", "intraplate",
              "triple junction", "slab", "Gorda slab (GS)", "seg:BL", "seg:MFZ", "seg:MTJ", "seg:SIF"]
    cfgs = [c for c in A.REGISTERED if c in answers]
    bat = scm_cmap("batlow")
    cols = {"temperature-limited": np.array(bat(0.85)[:3]), "composition-limited": np.array(bat(0.05)[:3]),
            "both consistent": np.array(bat(0.55)[:3]), "mixed": np.array(bat(0.3)[:3]),
            "neither": np.array([0.55, 0.55, 0.55])}
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for i, g in enumerate(groups):
        for j, c in enumerate(cfgs):
            a = answers[c]
            r = a[a.group.eq(g)]
            ans = r.answer.iloc[0] if len(r) else "not testable"
            col = cols.get(ans, np.array([0.93, 0.93, 0.93]))
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=col, edgecolor="white", lw=1.0))
            lum = 0.2126 * col[0] + 0.7152 * col[1] + 0.0722 * col[2]
            ax.text(j + 0.5, i + 0.5, CAT_SHORT.get(ans, ans), ha="center", va="center", fontsize=6,
                    color="white" if lum < 0.45 else "black")
    ax.set_xlim(0, len(cfgs))
    ax.set_ylim(len(groups), 0)
    ax.set_xticks(np.arange(len(cfgs)) + 0.5)
    ax.set_xticklabels(cfgs, rotation=0)
    ax.set_yticks(np.arange(len(groups)) + 0.5)
    ax.set_yticklabels([g.replace("seg:", "segment ").replace("frame:", "frame: ") for g in groups])
    ax.set_title("Registered answer per group under each configuration (main = decisions; others = variants and "
                 "controls)\nT temperature-limited, C composition-limited, T+C both consistent, none = neither,\n"
                 "? supported but not preferred, n/t not testable", fontsize=7)
    fig.savefig(FIG / "q2_controls.png")
    plt.close(fig)


def fig_circularity(C):
    C = C[C.status.eq("ok") & C.level.eq("V7.5")]
    groups = [g for g in ["frame:oceanic", "frame:slab", "frame:oceanic+slab", "transform", "seg:BL", "Gorda slab (GS)",
                          "slab", "triple junction"] if g in set(C.group)]
    fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    for i, g in enumerate(groups):
        r = C[C.group.eq(g)].iloc[0]
        ax.plot([r.rho_real_lo90, r.rho_real], [i, i], "-", color=_S[0], lw=2)
        ax.plot(r.rho_real, i, "o", color=_S[0], mec="k", ms=6, label="real rho (D90, z7.5); bar to block-bootstrap 5 %"
                if i == 0 else None)
        ax.plot(r.rho_truth, i - 0.22, "s", color=_S[4], mec="k", ms=4.5, label="synthetic truth rho" if i == 0 else None)
        ax.plot(r.rho_truth + r.A_median, i - 0.22, ">", color=_S[4], mec="k", ms=4.5,
                label="synthetic recovered (truth + median A)" if i == 0 else None)
        if np.isfinite(r.get("rho_null_p95", np.nan)):
            ax.plot([r.rho_null_median, r.rho_null_p95], [i + 0.22, i + 0.22], "-", color=_S[9], lw=2)
            ax.plot(r.rho_null_p95, i + 0.22, "|", color=_S[9], ms=9, mew=2,
                    label="null-truth members: median to 95th percentile" if i == 0 else None)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([g.replace("seg:", "segment ").replace("frame:", "frame: ") for g in groups])
    ax.invert_yaxis()
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman rho between D90 and the 7.5 km/s level (0.5 deg units)")
    ax.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.savefig(FIG / "q2_circularity.png", bbox_inches="tight")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    U = pd.read_csv(Q.OUT / "units_main.csv")
    G = pd.read_csv(Q.OUT / "groups_main.csv")
    E = pd.read_parquet(Q.BIG / "events_q2_main.parquet")
    fig_map(U)
    fig_age_depth(U, E)
    fig_scatter(U, G)
    fig_forest(G)
    fig_tstar(U)
    answers = {c: pd.read_csv(Q.OUT / f"answers_{c}.csv") for c in A.CONFIGS if (Q.OUT / f"answers_{c}.csv").exists()}
    fig_controls(answers)
    if (Q.OUT / "circularity_levels.csv").exists():
        fig_circularity(pd.read_csv(Q.OUT / "circularity_levels.csv"))
    Q.log("figures written to", FIG)


if __name__ == "__main__":
    main()

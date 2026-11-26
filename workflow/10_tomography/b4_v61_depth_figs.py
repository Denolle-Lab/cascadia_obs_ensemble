"""Track B, catalog v6.1 depth test: figures of the diagnosis (diag/) and of the test scores (score/), as PNG bytes.

Used by b4_v61_depth_report.py; also writes the PNG files next to the report (figs/).
"""
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path("/wd1/mdenolle_data/reloc_v61/depth_test")
DIAG = OUT / "diag"
FIGS = OUT / "figs"
CP, CS, CG, CR = "#2a78d6", "#eb6834", "#7a7a75", "#0b7a3e"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})


def png(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.png", dpi=150, bbox_inches="tight")
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return b.getvalue()


def fig_axial_passes(tile="t07_32"):
    """(a) Axial depth below the seafloor by pass; (b) caldera-station cumulative corrections by pass, P and S;
    (c) linearised depth shift split into common-P, common-S and station-pattern parts."""
    d = DIAG / tile
    P = pd.read_csv(d / "by_pass.csv")
    S = pd.read_csv(d / "station_phases.csv").set_index("key")
    L = pd.read_csv(d / "linearised.csv")
    L = L[L.tier.isin(["A", "B"])]
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.4))
    ab = P[P.tiers == "A-B"]
    a = ax[0]
    a.plot(ab.it, ab.within_2_0, "o-", color=CP, label="within 2.0 km")
    a.plot(ab.it, ab.within_1_6, "s-", color=CS, label="within 1.6 km")
    a.axhline(85, color=CS, ls=":", lw=1)
    a.axhline(90, color=CP, ls="--", lw=0.8)
    a.text(4.05, 90, "Waldhauser 2.0 km: 90 %", fontsize=7, va="bottom", ha="right", color=CP)
    a.text(4.05, 85, "Waldhauser 1.6 km: 85 %", fontsize=7, va="top", ha="right", color=CS)
    a.set_xticks(range(5))
    a.set_xticklabels(["it0\nno terms", "it1\nstatic", "it2\n16 km", "it3\n8 km", "it4\n4 km"])
    a.set_ylabel("tier A-B events below seafloor (%)")
    a.set_title("(a) Axial t07_32: share near the seafloor", loc="left")
    a.legend(frameon=False, fontsize=7, loc="lower left")
    a2 = a.twinx()
    a2.plot(ab.it, ab.median_bsf, "d--", color=CG, ms=4)
    a2.set_ylabel("median depth below seafloor (km)", color=CG)
    a2.spines["right"].set_visible(True)
    b = ax[1]
    cal = S[S.near & (S.n_it0 >= 1000)].copy()
    cum = np.cumsum(cal[[f"inc_g{k}_median" for k in range(4)]].to_numpy(), axis=1)
    for (key, r), c in zip(cal.iterrows(), cum):
        b.plot(range(1, 5), np.r_[c] * 1e3, "o-" if r.phase == "P" else "s-", color=CP if r.phase == "P" else CS,
               lw=1, ms=3, alpha=0.8)
        b.text(4.08, c[-1] * 1e3, key.replace(".OO", ""), fontsize=6, va="center", color=CP if r.phase == "P" else CS)
    for ph, col in (("P", CP), ("S", CS)):
        m = cal[cal.phase == ph]
        cm = np.cumsum(m[[f"inc_g{k}_median" for k in range(4)]].to_numpy(), axis=1)
        w = m.n_it0.to_numpy()
        b.plot(range(1, 5), (w @ cm) / w.sum() * 1e3, "-", color=col, lw=3, alpha=0.5,
               label=f"{ph}: pick-weighted mean")
    b.axhline(0, color="k", lw=0.6)
    b.set_xticks(range(1, 5))
    b.set_xticklabels(["after g0\n(static)", "g1\n16 km", "g2\n8 km", "g3\n4 km"])
    b.set_xlim(0.8, 4.9)
    b.set_ylabel("cumulative correction at the events (ms)")
    b.set_title("(b) caldera stations: terms by pass", loc="left")
    b.legend(frameon=False, fontsize=7, loc="lower left")
    c = ax[2]
    parts = [("common\nP", "dz_commonP"), ("common\nS", "dz_commonS"), ("station\npattern", "dz_rest"),
             ("sum\n(linear)", "dz_lin"), ("observed", "dz_obs")]
    x = np.arange(len(parts))
    for k, (it, off, col) in enumerate(((1, -0.2, "#9cc3ee"), (4, 0.2, CP))):
        v = [L[f"{p}_it{it}"].mean() for _, p in parts]
        c.bar(x + off, v, width=0.38, color=col, label=f"it{it} - it0")
        for xi, vi in zip(x + off, v):
            c.text(xi, vi + (0.01 if vi >= 0 else -0.01), f"{vi:+.2f}", ha="center",
                   va="bottom" if vi >= 0 else "top", fontsize=6.5)
    c.axhline(0, color="k", lw=0.6)
    c.set_xticks(x)
    c.set_xticklabels([p for p, _ in parts], fontsize=7.5)
    c.set_ylabel("mean depth shift (km, + deeper)")
    c.set_title("(c) what moves the events (linearised)", loc="left")
    c.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return png(fig, "diag_axial")


def fig_tradeoff(tiles):
    """per-event correlation between depth and a free common S term (inverse of G'WG with a column 1_S)."""
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.0))
    for tile, lab in tiles:
        f = DIAG / tile / "linearised.csv"
        if not f.exists():
            continue
        L = pd.read_csv(f)
        L = L[L.tier.isin(["A", "B", "C"])].dropna(subset=["corr_z_commonS"])
        ax[0].hist(L.corr_z_commonS, bins=np.linspace(-1, 1, 41), histtype="step", lw=1.4, density=True, label=lab)
        ax[1].hist(np.log10(L.sigz_ratio_commonS), bins=np.linspace(0, 1.5, 31), histtype="step", lw=1.4,
                   density=True, label=lab)
    ax[0].set_xlabel("correlation(depth, common S term)")
    ax[0].set_title("(a) depth trades with a common S term", loc="left")
    ax[1].set_xlabel("log10 [ sigma_z with a free common S term / sigma_z ]")
    ax[1].set_title("(b) depth uncertainty inflation", loc="left")
    ax[0].legend(frameon=False, fontsize=7)
    for a in ax:
        a.set_yticks([])
    fig.tight_layout()
    return png(fig, "diag_tradeoff")


def fig_scores(H, D, dec):
    """held-out MAD (relative to the best arm, %) against the depth score D per arm; one panel per phase."""
    arms = dec["arms"]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
    sets = list(dict.fromkeys(H.set))
    mk = dict(zip(sets, "osD^"))
    for k, ph in enumerate(("P", "S")):
        a = ax[k]
        for s in sets:
            h = H[(H.set == s) & (H.phase == ph)].set_index("arm")
            xs = [i for i, arm in enumerate(arms) if arm in h.index]
            rel = [100 * h.loc[arms[i], "diff_best_ms"] / h.loc[h.best.iloc[0], "mad_ms"] for i in xs]
            lo = [100 * h.loc[arms[i], "lo_ms"] / h.loc[h.best.iloc[0], "mad_ms"] for i in xs]
            hi = [100 * h.loc[arms[i], "hi_ms"] / h.loc[h.best.iloc[0], "mad_ms"] for i in xs]
            a.errorbar(np.array(xs) + 0.12 * (sets.index(s) - 1.5), rel,
                       yerr=[np.array(rel) - np.array(lo), np.array(hi) - np.array(rel)], fmt=mk[s], ms=4,
                       lw=0.8, capsize=0, label=s)
        a.axhline(5, color="k", ls=":", lw=0.8)
        a.set_xticks(range(len(arms)))
        a.set_xticklabels(arms, rotation=45, ha="right", fontsize=7.5)
        a.set_ylabel(f"held-out MAD {ph}: arm - best (% of best)")
        a.set_title(f"({'ab'[k]}) relative fit, {ph} (dotted: 5 % gate)", loc="left")
        a.set_yscale("symlog", linthresh=10)
        if k == 0:
            a.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    return png(fig, "scores_fit")


def fig_depth(Dp, dec):
    arms = dec["arms"]
    sets = [s for s in ("axial", "msh", "blancow", "gordad") if s in set(Dp.set)]
    fig, ax = plt.subplots(1, len(sets) + 1, figsize=(3.0 * (len(sets) + 1), 3.4), sharey=False)
    for k, s in enumerate(sets):
        a = ax[k]
        d = Dp[Dp.set == s].set_index("arm").reindex(arms)
        x = np.arange(len(arms))
        col = [CS if "coh" in arm else CP for arm in arms]
        a.bar(x, d.offset_km, color=col)
        if "p10_km" in d and s != "axial":
            a.errorbar(x, d.offset_km, yerr=[d.offset_km - d.p10_km, d.p90_km - d.offset_km], fmt="none",
                       ecolor="k", lw=0.6, alpha=0.5)
        a.axhline(0, color="k", lw=0.6)
        a.set_xticks(x)
        a.set_xticklabels(arms, rotation=60, ha="right", fontsize=6.5)
        lim = d.offset_km.abs().max()
        if s != "axial":
            a.set_ylim(min(0, d.offset_km.min()) - 0.15 * lim, d.offset_km.max() + 0.15 * lim)
        a.set_title(f"{s}: offset (km)" + (" [quantiles]" if s == "axial" else " [median]"), loc="left", fontsize=8)
    a = ax[-1]
    D = dec["D_km"]
    x = np.arange(len(arms))
    elig = set(dec["eligible"])
    a.bar(x, [D.get(arm, np.nan) for arm in arms], color=[CR if arm == dec["winner"] else (CG if arm in elig else
                                                                                         "#d9d8d4") for arm in arms])
    a.set_xticks(x)
    a.set_xticklabels(arms, rotation=60, ha="right", fontsize=6.5)
    a.set_title("D = mean |offset| (green: winner,\ngrey: eligible, light: out)", loc="left", fontsize=8)
    dm = [D[arm] for arm in arms if arm in D]
    a.set_ylim(min(dm) - 0.3, max(dm) + 0.1)
    fig.tight_layout()
    return png(fig, "scores_depth")


if __name__ == "__main__":
    fig_axial_passes()
    fig_tradeoff([("t07_32", "Axial t07_32"), ("t29_40", "Puget t29_40"), ("t22_09", "Gorda t22_09"),
                  ("t11_23", "Blanco t11_23")])
    print("figs ->", FIGS)

#!/usr/bin/env python3
"""Q1 (D-B115) figures. Crameri colour maps and the batlowS segment-type colours with a marker per type (D-B104,
as pb_figures.py): sequential batlow for shares, redundant marker coding for colour-vision deficiency.

  fig_q1_segments  swarm share, b and productivity per segment with 95 % intervals; time-shuffle null 95th percentile
  fig_q1_map       0.2 degree cells coloured by swarm share; plate age and heat flow (D-B114) at the same cells
  fig_q1_scatter   cell swarm share against plate age, heat flow, roughness, sediment (D-B114 values)
  fig_q1_controls  cell rank correlation (age, heat flow) in the main run and the 22 control runs
  fig_q1_vpvs      in-situ Vp/Vs: estimate against signal-to-noise; valid families by type
Writes PNGs to BIG/figures.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import xarray as xr  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "08_figures"))
import q1_common as Q  # noqa: E402
from pb_figures import TYPE_COL, TYPE_MK  # noqa: E402
from scm_palettes import scm_cmap  # noqa: E402

TAB = Q.BIG / "tables"
FIG = Q.BIG / "figures"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "font.family": "DejaVu Sans"})
CM = scm_cmap("batlow", bad="#ffffff")
SRC = "external" if (TAB / "tests_external.csv").exists() else "interim"


def coast(ax):
    f = Q.BIG / "covariates" / "gebco2026_cut.nc"
    if f.exists():
        d = xr.open_dataset(f).z
        d = d.sel(lon=slice(-132.2, -119.8), lat=slice(38.6, 51.6))[::8, ::8]
        ax.contour(d.lon, d.lat, d.values, levels=[0], colors="0.35", linewidths=0.5)


def fig_segments():
    m = pd.read_csv(TAB / f"segments_main_{SRC}.csv")
    m = m[m.seg_type_major != "other"].copy()
    nu = pd.read_csv(TAB / "nulls_units.csv") if (TAB / "nulls_units.csv").exists() else pd.DataFrame()
    order = [s for s in Q.P.SEG_ORDER if s in set(m.unit_id)]
    m = m.set_index("unit_id").loc[order].reset_index()
    fig, ax = plt.subplots(3, 1, figsize=(7.2, 6.4), sharex=True)
    x = np.arange(len(m))
    for i, r in m.iterrows():
        c, mk = TYPE_COL[r.seg_type_major], TYPE_MK[r.seg_type_major]
        ax[0].errorbar(x[i], r.swarm_share, yerr=[[r.swarm_share - r.swarm_share_lo], [r.swarm_share_hi - r.swarm_share]],
                       fmt=mk, color=c, ms=6, mec="k", mew=0.4, capsize=2, lw=0.8)
        if np.isfinite(r.b):
            ax[1].errorbar(x[i], r.b, yerr=[[r.b - r.b_lo], [r.b_hi - r.b]], fmt=mk, color=c, ms=5, mec="k", mew=0.4,
                           capsize=2, lw=0.8)
        if np.isfinite(r.get("productivity", np.nan)):
            ax[2].errorbar(x[i], r.productivity, yerr=[[r.productivity - r.productivity_lo],
                                                       [r.productivity_hi - r.productivity]],
                           fmt=mk, color=c, ms=5, mec="k", mew=0.4, capsize=2, lw=0.8)
    if len(nu):
        t = nu[(nu.null == "tshuf") & (nu.unit == "seg")].set_index("unit_id")
        mm = nu[(nu.null == "mshuf") & (nu.unit == "seg")].set_index("unit_id")
        ax[0].scatter(x, t.null_p95.reindex(order), marker="_", s=120, color="k", lw=1.2,
                      label="time-shuffle null, 95th percentile", zorder=5)
        ax[0].scatter(x, mm.null_mean.reindex(order), marker="x", s=18, color="0.45", lw=0.8,
                      label="magnitude-shuffle null, mean", zorder=5)
        ax[0].legend(loc="upper right", fontsize=6.5, frameon=False)
    for k in order:
        if k in Q.HOT:
            ax[0].text(order.index(k), -4, "hot", ha="center", va="top", fontsize=6, color="0.3")
        elif k in Q.COLD:
            ax[0].text(order.index(k), -4, "cold", ha="center", va="top", fontsize=6, color="0.3")
    ax[0].set_ylim(-8, None)
    ax[0].set_ylabel("swarm share above Mc (%)")
    ax[1].set_ylabel("b (Aki-Utsu)")
    ax[2].set_ylabel("direct offspring per\nevent at Mc+1 to Mc+2")
    ax[2].set_xticks(x, order, rotation=0)
    for a in ax:
        a.grid(axis="y", lw=0.3, color="0.85")
    hs = [plt.Line2D([], [], marker=TYPE_MK[t], color=TYPE_COL[t], ls="", mec="k", mew=0.4, label=t)
          for t in Q.P.TYPE_ORDER if t in set(m.seg_type_major)]
    ax[2].legend(handles=hs, ncol=4, fontsize=6.5, frameon=False, loc="upper left")
    ax[0].set_title("Slip style by segment, catalog v6.1: tier A-C earthquakes above the regional Mc, 95 % bootstrap "
                    "intervals", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_q1_segments.png", dpi=200)
    plt.close(fig)


def fig_map():
    c = pd.read_csv(TAB / f"cells_main_{SRC}.csv")
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 4.3), sharey=True)
    specs = [("swarm_share", "swarm share (%)", CM, None), ("age_myr", "plate age (Myr)", scm_cmap("lajolla"), "log"),
             ("heatflow_mw", "heat flow (mW/m$^2$)", scm_cmap("lajolla", reverse=True), "log")]
    from matplotlib.colors import LogNorm
    for a, (col, lab, cm, sc) in zip(ax, specs):
        coast(a)
        v = c[col].values
        ok = np.isfinite(v)
        norm = LogNorm(np.nanpercentile(v[ok & (v > 0)], 2), np.nanpercentile(v[ok], 98)) if sc == "log" else None
        a.scatter(c.lon[~ok], c.lat[~ok], s=10, marker="s",
                  facecolor="none", edgecolor="0.6", lw=0.4)
        sc_ = a.scatter(c.lon[ok], c.lat[ok], c=np.where(v[ok] > 0, v[ok], np.nan) if sc == "log" else v[ok],
                        s=16, marker="s", cmap=cm, norm=norm, edgecolor="k", lw=0.2)
        cb = fig.colorbar(sc_, ax=a, orientation="horizontal", pad=0.07, fraction=0.05)
        cb.set_label(lab)
        a.set_xlim(-131.8, -120)
        a.set_ylim(39, 51.2)
        a.set_aspect(1 / np.cos(np.radians(45)))
        a.tick_params(labelsize=6.5)
    ax[0].set_title("0.2 deg cells with >= 50 events\n(open: no value)", loc="left", fontsize=7.5)
    ax[1].set_title("D-B114 values: median over\nthe cell's events", loc="left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_q1_map.png", dpi=200)
    plt.close(fig)


def fig_scatter():
    c = pd.read_csv(TAB / f"cells_main_{SRC}.csv")
    T = pd.read_csv(TAB / f"tests_{SRC}.csv")
    covs = [("age_myr", True), ("heatflow_mw", True), ("rough_m", False), ("sed_m", True)]
    fig, ax = plt.subplots(1, 4, figsize=(7.4, 2.8), sharey=True)
    for a, (cov, lg) in zip(ax, covs):
        for t in Q.P.TYPE_ORDER:
            g = c[c.seg_type_major == t]
            a.scatter(g[cov], g.swarm_share, marker=TYPE_MK[t], color=TYPE_COL[t], s=14, edgecolor="k", lw=0.25,
                      label=t)
        if lg:
            a.set_xscale("log")
        r = T[(T.unit == "cell") & (T.metric == "swarm_share") & (T.covariate == cov)]
        if len(r):
            r = r.iloc[0]
            a.set_title(f"rho = {r.rho:.2f}, p = {r.p:.2g}, n = {int(r.n)}", loc="left", fontsize=7)
        a.set_xlabel(Q.COV_LABEL[cov])
        a.grid(lw=0.3, color="0.88")
    ax[0].set_ylabel("swarm share (%)")
    hs, ls = ax[0].get_legend_handles_labels()
    fig.legend(hs, ls, ncol=8, fontsize=6.5, frameon=False, loc="lower center")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "fig_q1_scatter.png", dpi=200)
    plt.close(fig)


def fig_controls():
    C = pd.read_csv(TAB / f"controls_{SRC}.csv")
    T = pd.read_csv(TAB / f"tests_{SRC}.csv").set_index("test")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 4.2), sharey=True)
    runs = list(dict.fromkeys(C.run))
    y = np.arange(len(runs) + 1)
    for a, (cov, test) in zip(ax, (("age_myr", "T1"), ("heatflow_mw", "T2"))):
        g = C[C.covariate == cov].set_index("run").reindex(runs)
        a.axvline(0, color="0.5", lw=0.6)
        a.scatter(g.rho, y[1:], color=np.where(g.sign_ok, "#1b5e8a", "#c0703a"), s=14, zorder=3)
        a.errorbar(T.loc[test, "rho"], 0, xerr=[[T.loc[test, "rho"] - T.loc[test, "block_lo"]],
                                               [T.loc[test, "block_hi"] - T.loc[test, "rho"]]],
                   fmt="D", color="k", ms=5, capsize=2)
        a.set_title(f"{Q.COV_LABEL[cov]}: predicted sign {'-' if Q.PRED_SIGN[cov] < 0 else '+'}", loc="left")
        a.set_xlabel("Spearman rho, cell swarm share")
        a.grid(axis="x", lw=0.3, color="0.88")
    ax[0].set_yticks(y, ["main (block 95 %)"] + runs, fontsize=6.5)
    ax[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG / "fig_q1_controls.png", dpi=200)
    plt.close(fig)


def fig_vpvs():
    V = pd.read_csv(Q.BIG / "vpvs" / "vpvs_families.csv")
    e = V[V.vpvs.notna()]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    a = ax[0]
    a.scatter(e.snr, e.vpvs, s=8, color="0.55", label="L1 estimate (all)")
    a.scatter(e.snr[e.valid], e.vpvs[e.valid], s=14, color="#1b5e8a", edgecolor="k", lw=0.3, label="valid")
    ax_ = e[e.block == "axial"]
    a.scatter(ax_.snr, ax_.vpvs, s=14, marker="x", color="#c0703a", label="Axial (S-P ~ 0.5 s)")
    a.axhspan(1.4, 2.5, color="0.93", zorder=0)
    a.set_xscale("log")
    a.set_xlabel("signal-to-noise: IQR of dtP' / MAD of residuals")
    a.set_ylabel("family Vp/Vs")
    a.legend(fontsize=6, frameon=False, loc="lower right")
    a.set_title("Estimates fall with signal-to-noise (regression dilution)", loc="left", fontsize=7.5)
    b = ax[1]
    f = V[V.valid & V.carried]
    types = ["swarm", "mainshock-aftershock", "multiplet"]
    for i, t in enumerate(types):
        g = f[f.type61 == t]
        for j, (_, r) in enumerate(g.iterrows()):
            b.errorbar(i + (j - len(g) / 2) * 0.03, r.vpvs, yerr=[[r.vpvs - r.vpvs_lo], [r.vpvs_hi - r.vpvs]],
                       fmt=TYPE_MK.get(r.seg_type, "o"), color=TYPE_COL.get(r.seg_type, "0.5"), ms=4, mec="k",
                       mew=0.3, lw=0.6)
    b.set_xticks(range(3), [f"{t}\n(n = {int((f.type61 == t).sum())})" for t in types])
    b.set_ylabel("Vp/Vs (valid families)")
    b.set_title("Valid families by v6.1 family type (T3 needs >= 20 each)", loc="left", fontsize=7.5)
    b.grid(axis="y", lw=0.3, color="0.88")
    fig.tight_layout()
    fig.savefig(FIG / "fig_q1_vpvs.png", dpi=200)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (fig_segments, fig_map, fig_scatter, fig_controls, fig_vpvs):
        try:
            f()
            Q.log(f"{f.__name__} done")
        except Exception as ex:  # a missing input must not stop the other figures
            Q.log(f"{f.__name__} FAILED: {type(ex).__name__}: {ex}")
            raise


if __name__ == "__main__":
    main()

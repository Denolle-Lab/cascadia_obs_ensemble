#!/usr/bin/env python3
"""Catalog v5.1 re-interpretation (piece "interp"): figures.

Reads the joined event table written by v51_interp_tables.py and the products of the pieces
faults and clusters. Writes PNGs to data/catalog_v5/v5.1_analysis/interp/figs/.

    python workflow/06_analysis/v51_interp_figs.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from v51_interp_common import (BIG, CLU, OUT, REGIONS, SLUG, VOLCANOES, V0, load_pb_steps, local_xy,  # noqa: E402
                               hav)

FIG = OUT / "figs"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "figure.dpi": 110, "savefig.dpi": 110})
CLS_COL = {"background": "#9a9a9a", "mainshock-aftershock": "#1f5fa8", "swarm": "#d9711a", "multiplet": "#2e8b57"}
GMT = "/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/bin/gmt"

SECT = {"Axial": 90, "Endeavour": 90, "Explorer": 90, "JdF ridge/plate": 20, "Blanco": 116, "Gorda": 90,
        "Mendocino": 90, "Offshore forearc": 90, "Puget/Georgia": 90, "Cascades": 0, "Other onshore": 90}


def coast():
    f = BIG / "coast_i.txt"
    if not f.exists():
        with open(f, "w") as fh:
            subprocess.run([GMT, "coast", "-R-134/-118/38/52", "-Di", "-W", "-M"], stdout=fh, check=True, cwd=BIG)
    segs, cur = [], []
    for line in f.read_text().splitlines():
        if line.startswith(">"):
            if cur:
                segs.append(np.array(cur))
            cur = []
        else:
            a = line.split()
            if len(a) >= 2:
                cur.append((float(a[0]), float(a[1])))
    if cur:
        segs.append(np.array(cur))
    return segs


COAST = coast()
PB = load_pb_steps()


def draw_base(ax, ext, pb=True, volc=True):
    for s in COAST:
        ax.plot(s[:, 0], s[:, 1], "-", color="#333", lw=0.4, zorder=3)
    if pb:
        colors = {"OSR": "#b2182b", "OTF": "#2166ac", "SUB": "#4d4d4d"}
        for _, r in PB.iterrows():
            ax.plot([r.lon1, r.lon2], [r.lat1, r.lat2], "-", color=colors.get(r.cls, "#999"), lw=0.8, alpha=0.7, zorder=3)
    if volc:
        for n, (la, lo) in VOLCANOES.items():
            if ext[0] <= lo <= ext[1] and ext[2] <= la <= ext[3]:
                ax.plot(lo, la, "^", ms=5, mfc="none", mec="k", mew=0.8, zorder=5)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect(1 / np.cos(np.radians((ext[2] + ext[3]) / 2)))


def traces(ax, kind):
    if kind == "v1":
        s = pd.read_csv(OUT / "surfaces_v1_interp.csv")
        for _, r in s.iterrows():
            bad = isinstance(r.interp_flags, str) and len(r.interp_flags) > 0
            ax.plot([r.top_edge_lon1, r.top_edge_lon2], [r.top_edge_lat1, r.top_edge_lat2], "-" if not bad else "--",
                    color="#7a0177", lw=1.1, zorder=4)
    else:
        t = pd.read_csv(V0 / "fault_surfaces_v0_traces.csv")
        for _, r in t.iterrows():
            ax.plot([r.top_edge_lon1, r.top_edge_lon2], [r.top_edge_lat1, r.top_edge_lat2], "-", color="#7a0177",
                    lw=1.1, zorder=4)


def fig_overview(c):
    E = c[c.is_eq]
    ext = (-131.5, -120.5, 39.5, 50.6)
    fig, axs = plt.subplots(1, 2, figsize=(11, 6.6), constrained_layout=True)
    for ax, (lab, la, lo, de) in zip(axs, [("catalog v4 (GENIE/GraphDD, datum unconfirmed)", "v4_lat", "v4_lon", "v4_depth"),
                                           ("catalog v5.1 (NonLinLoc, 3D model v0.3.1)", "lat", "lon", "depth")]):
        draw_base(ax, ext)
        o = np.argsort(-E[de].values)
        sc = ax.scatter(E[lo].values[o], E[la].values[o], c=E[de].values[o], s=0.3, cmap="viridis_r", vmin=0, vmax=50,
                        rasterized=True, zorder=2)
        traces(ax, "v0" if "v4" in lab else "v1")
        ax.set_title(lab)
    for r in REGIONS:
        g = E[E.region.eq(r)]
        axs[1].text(g.lon.median(), g.lat.median(), r, fontsize=7, ha="center", color="k",
                    bbox=dict(fc="white", ec="none", alpha=0.6, pad=0.5), zorder=6)
    fig.colorbar(sc, ax=axs, shrink=0.6, label="depth (km below sea level)")
    fig.suptitle(f"Located earthquakes ({len(E):,}; 623 rows without a hypocentre and 3,313 non-earthquakes left out).\n"
                 "Purple: fault-surface top edges (left v0, right v1; dashed = flagged). Red/blue/grey: PB2002 ridge/transform/subduction. Triangles: volcanoes.",
                 fontsize=8)
    fig.savefig(FIG / "fig_overview.png")
    plt.close(fig)


LIT = {  # literature depth ranges for the depth panels (km below sea level unless noted; from depth_literature.html)
    "Axial": (1.5, 3.5, "0-2 km bsf (+1.5 km seafloor)"), "Endeavour": (3.6, 6.1, "1.5-3.5 km bsf"),
    "Explorer": (5, 22, "5-22 km bsl"), "JdF ridge/plate": (2.5, 5.5, "above AMC, <~3 km bsf"),
    "Blanco": (4.7, 11.3, "Lange p5-p95"), "Gorda": (6, 15, "interior limit ~9.5-15 bsf"),
    "Mendocino": (15, 25, "slab double zone"), "Offshore forearc": (8, 23, "Newport 8-13, Fortuna 17-23"),
    "Puget/Georgia": (15, 25, "crust 15-25; slab 40-65"), "Cascades": (-2.2, 8, "MSH -2.2 to 8"),
    "Other onshore": (13, 30, "Scotts Mills 13-15; NW Calif slab 15-30")}


def fig_depths(c):
    E = c[c.is_eq]
    fig, axs = plt.subplots(3, 4, figsize=(11, 7.5), constrained_layout=True)
    for ax, r in zip(axs.ravel(), REGIONS):
        g = E[E.region.eq(r)]
        bins = np.arange(-4, 72, 1.0)
        ax.hist(g.v4_depth, bins=bins, histtype="step", color="#999", lw=1.2, label="v4 (all)", orientation="horizontal")
        ax.hist(g.depth, bins=bins, histtype="step", color="#1f5fa8", lw=1.2, label="v5.1 (all)", orientation="horizontal")
        gg = g[g.depth_good]
        if len(gg):
            ax.hist(gg.depth, bins=bins, histtype="stepfilled", color="#d9711a", alpha=0.45,
                    label="v5.1 depth_good", orientation="horizontal")
        lo, hi, lab = LIT[r]
        ax.axhspan(lo, hi, color="#2e8b57", alpha=0.15, lw=0)
        ax.text(0.98, 0.02, f"green: {lab}", transform=ax.transAxes, ha="right", va="bottom", fontsize=6)
        ax.set_ylim(70, -4)
        ax.set_title(f"{r} (n={len(g):,})")
        ax.set_xlabel("events per km")
        ax.set_ylabel("depth (km bsl)")
    axs.ravel()[0].legend(fontsize=6, loc="lower right", bbox_to_anchor=(1, 0.12))
    ax = axs.ravel()[-1]
    ax.axis("off")
    ax.text(0, 0.9, "depth_good: tier A-C, depth_quality ok,\nnot floor_limited / far_only / deep_pass.\n"
                    "Endeavour, Explorer and JdF have none\n(depth_quality unconstrained everywhere).\n"
                    "Green band: published range, shifted to km bsl\nwhere the source gives depth below seafloor.",
            va="top", fontsize=7)
    fig.savefig(FIG / "fig_depths_by_region.png")
    plt.close(fig)


def region_ext(g):
    lo = np.r_[g.lon.values, g.v4_lon.values]
    la = np.r_[g.lat.values, g.v4_lat.values]
    x0, x1 = np.nanpercentile(lo, [1, 99])
    y0, y1 = np.nanpercentile(la, [1, 99])
    px = max(0.08, 0.06 * (x1 - x0))
    py = max(0.06, 0.06 * (y1 - y0))
    return (x0 - px, x1 + px, y0 - py, y1 + py)


def fig_region(c, r):
    E = c[c.is_eq & c.region.eq(r)]
    ext = region_ext(E)
    lat0, lon0 = E.lat.median(), E.lon.median()
    az = np.radians(SECT[r])
    fig = plt.figure(figsize=(11, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])
    dd = []
    for la_, lo_ in [("v4_lat", "v4_lon"), ("lat", "lon")]:
        x_, y_ = local_xy(E[la_], E[lo_], lat0, lon0)
        dd.append(x_ * np.sin(az) + y_ * np.cos(az))
    dd = np.concatenate(dd)
    xlim = np.nanpercentile(dd, [0.5, 99.5]) + np.array([-3, 3])
    ytop = min(70, max(15, np.nanpercentile(np.r_[E.depth.values, E.v4_depth.values], 99) + 3))
    for k, (ver, la, lo, de, cls) in enumerate([("v4", "v4_lat", "v4_lon", "v4_depth", "cls4"),
                                                 ("v5.1", "lat", "lon", "depth", "cls")]):
        ax = fig.add_subplot(gs[0, k])
        draw_base(ax, ext)
        for cl in ["background", "multiplet", "mainshock-aftershock", "swarm"]:
            g = E[E[cls].eq(cl)]
            ax.scatter(g[lo], g[la], s=1.2 if cl == "background" else 2.0, c=CLS_COL[cl], lw=0, rasterized=True,
                       label=f"{cl} ({len(g):,})", zorder=2 if cl == "background" else 3)
        traces(ax, "v0" if ver == "v4" else "v1")
        ax.legend(fontsize=6, loc="lower left", markerscale=4, framealpha=0.8)
        ax.set_title(f"{r}, catalog {ver}: class from the {'v4' if ver == 'v4' else 'v5.1'} clustering "
                     f"({'v0' if ver == 'v4' else 'v1'} surfaces)")
        # section line
        L = 300
        xs = np.array([-L, L]) * np.sin(az)
        ys = np.array([-L, L]) * np.cos(az)
        ax.plot(lon0 + xs / (111.195 * np.cos(np.radians(lat0))), lat0 + ys / 111.195, ":", color="k", lw=0.7)
        axs = fig.add_subplot(gs[1, k])
        x, y = local_xy(E[la], E[lo], lat0, lon0)
        d = x * np.sin(az) + y * np.cos(az)
        good = E.depth_good.values
        fl = E.floor_limited.values | E.far_only.values | E.deep_pass.values
        axs.scatter(d[~good & ~fl], E[de].values[~good & ~fl], s=0.8, c="#bbbbbb", lw=0, rasterized=True,
                    label="tier D or depth_quality not ok")
        axs.scatter(d[fl], E[de].values[fl], s=1.0, c="#e6a23c", lw=0, rasterized=True,
                    label="floor_limited / far_only / deep_pass")
        axs.scatter(d[good], E[de].values[good], s=0.8, c="#1f3a68", lw=0, rasterized=True, label="depth_good")
        axs.set_ylim(ytop, -4)
        axs.set_xlim(*xlim)
        axs.set_xlabel(f"distance along azimuth {SECT[r]} deg from {lat0:.2f}N {lon0:.2f}E (km)")
        axs.set_ylabel(f"depth (km bsl), {ver}")
        axs.legend(fontsize=6, loc="lower left", markerscale=5)
        axs.set_title(f"section, all events of the region projected ({ver})")
    fig.savefig(FIG / f"fig_region_{SLUG[r]}.png")
    plt.close(fig)


KEY_SWARMS = [  # (family_id, title)
    (1872, "Axial Seamount: eruption onset swarm, 2015-04-24"),
    (427, "Blanco: 2012-10-02 swarm (largest family)"),
    (1170, "Blanco: 2014-03-11/12 sequence (v4 mainshock-aftershock, v5.1 swarm)"),
    (1932, "Blanco: 2015-06-01 Mw 5.8 / 5.4 / 5.9 triplet"),
    (296, "Explorer: 2012-02-04 M 5.6 sequence (migrating)"),
    (1066, "Juan de Fuca ridge: 2014-01-09 swarm"),
    (736, "Gorda: 2013-05-20 Escanaba Trough swarm"),
    (989, "Gorda: 2013-11-13 swarm (fast migration on 16 events)"),
    (150, "Mount St. Helens: 2011-01-28 swarm"),
    (830, "Mount St. Helens area: 2013-08-03 compact swarm at 13-15 km"),
    (1347, "Offshore forearc (Fortuna area): 2014-07-16 swarm"),
    (431, "Puget Lowland: 2012-10-05 swarm"),
]


def fig_swarm(c, fam, title, fa):
    F = fa[fa.family_id.eq(fam)].iloc[0]
    G = c[c.family_id.eq(fam) & c.is_eq].sort_values("t")
    if len(G) < 3:
        return None
    t0 = G.t.min()
    td = (G.t - t0).dt.total_seconds() / 86400
    lat0, lon0 = G.lat.median(), G.lon.median()
    x, y = local_xy(G.lat, G.lon, lat0, lon0)
    x4, y4 = local_xy(G.v4_lat, G.v4_lon, lat0, lon0)
    az = F.axis_az if np.isfinite(F.axis_az) else 90.0
    a = np.radians(az)
    along = x * np.sin(a) + y * np.cos(a)
    good = G.depth_good.values
    M = G.M51.fillna(G.M51.min() if G.M51.notna().any() else 0).values
    sz = 3 + 4 * np.clip(M - np.nanmin(M), 0, 6) ** 1.6
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.4), constrained_layout=True)
    ax = axs[0]
    ax.scatter(x4, y4, s=4, c="#cccccc", lw=0, label="v4 positions")
    sc = ax.scatter(x, y, s=sz, c=td, cmap="plasma", lw=0.2, edgecolors="k", label="v5.1")
    ax.set_aspect(1)
    lim = max(3, np.nanpercentile(np.abs(np.r_[x, y]), 97) * 1.2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("km east")
    ax.set_ylabel("km north")
    ax.legend(fontsize=6, loc="lower left")
    ax.set_title(f"map, centre {lat0:.3f}N {lon0:.3f}E")
    fig.colorbar(sc, ax=ax, shrink=0.8, label="days since first event")
    ax = axs[1]
    ax.scatter(td, along, s=sz, c=np.where(good, "#1f3a68", "#aaaaaa"), lw=0)
    ax.set_xlabel("days since first event")
    ax.set_ylabel(f"km along azimuth {az:.0f} deg")
    ax.set_title("space-time (dark: depth_good)")
    ax = axs[2]
    ax.scatter(td[~good], G.depth.values[~good], s=sz[~good], c="#bbbbbb", lw=0, label="depth not good")
    ax.scatter(td[good], G.depth.values[good], s=sz[good], c="#1f3a68", lw=0, label="depth_good")
    ax.invert_yaxis()
    ax.set_xlabel("days since first event")
    ax.set_ylabel("depth (km bsl)")
    ax.legend(fontsize=6)
    ax.set_title("depth-time")
    ax = axs[3]
    ax.scatter(td, G.M51, s=8, c=np.where(G.in_analysis.eq(True), "#d9711a", "#999999"), lw=0)
    ax.set_xlabel("days since first event")
    ax.set_ylabel("M (v5.1)")
    ax.set_title("magnitude (orange: above Mc)")
    for ax in axs[1:]:
        if td.max() > 5 * max(np.median(td), 1e-3):
            ax.set_xscale("symlog", linthresh=0.1)
            ax.set_xlim(0, td.max() * 1.3)
            ax.set_xlabel("days since first event (linear below 0.1 d, log above)")
    fig.suptitle(f"{title}. Family {fam}: {F.type}, {int(F.n_all)} events ({int(F.n)} above Mc), "
                 f"Mmax {F.Mmax:.1f}, start {str(F.start_all)[:16]} UTC, v4 status {F.v4_status}", fontsize=8)
    out = FIG / f"fig_swarm_{fam}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_msh(c):
    E = c[c.is_eq & (hav(c.lat.values, c.lon.values, 46.20, -122.19) < 15)].copy()
    E["how"] = E.t.dt.dayofweek * 24 + E.t.dt.hour + E.t.dt.minute / 60
    deep = E.depth > 30
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    ax = axs[0]
    ax.hist(E.how[~deep], bins=np.arange(0, 169, 1), color="#999", label=f"depth <= 30 km (n={int((~deep).sum()):,})",
            density=True)
    ax.hist(E.how[deep], bins=np.arange(0, 169, 1), histtype="step", color="#c0392b", lw=1.2,
            label=f"depth > 30 km (n={int(deep.sum())})", density=True)
    ax.set_xticks(np.arange(0, 169, 24))
    ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", ""])
    ax.set_xlabel("hour of the week (UTC)")
    ax.set_ylabel("density")
    ax.legend(fontsize=6)
    ax.set_title("events within 15 km of Mount St. Helens")
    ax = axs[1]
    ax.scatter(E.t[~deep], E.depth[~deep], s=0.5, c="#999", lw=0, rasterized=True)
    ax.scatter(E.t[deep], E.depth[deep], s=3, c=np.where(E[deep].tier.isin(["A", "B"]), "k", "#c0392b"), lw=0)
    ax.invert_yaxis()
    ax.set_ylabel("depth (km bsl)")
    ax.set_title("depth against time (red: deep, tiers C-D)")
    ax = axs[2]
    ax.scatter(E.v4_depth[deep], E.depth[deep], s=4, c="#c0392b", lw=0)
    ax.plot([0, 120], [0, 120], "k:", lw=0.7)
    ax.set_xlabel("v4 depth (km)")
    ax.set_ylabel("v5.1 depth (km bsl)")
    ax.set_title("deep events: v4 against v5.1")
    fig.savefig(FIG / "fig_msh_deep_weekly.png")
    plt.close(fig)


def fig_robustness():
    b = pd.read_csv(OUT / "t3b_boxes.csv")
    order = ["AX", "EN", "BL", "GO", "MT", "WA", "OR", "PU", "ON"]
    b = b.set_index("region").loc[order].reset_index()
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), constrained_layout=True)
    ax = axs[0]
    y = np.arange(len(b))
    for col, m, lab in [("bg_v4", "s", "v4"), ("bg_main", "o", "v5.1 main"), ("bg_tierABC", "v", "v5.1 tier A-C only"),
                        ("bg_eta0_shuf5", "^", "v5.1 conservative threshold"), ("bg_d3q", "x", "v5.1 3-D (good depths)")]:
        ax.plot(b[col], y, m, label=lab, ms=5, mfc="none" if m in "sv^" else None)
    ax.set_yticks(y)
    ax.set_yticklabels(b.region)
    ax.invert_yaxis()
    ax.set_xlabel("% of events above Mc that are background")
    ax.legend(fontsize=6, loc="lower right")
    ax.set_title("background share: main run and variants (clusters piece)")
    ax = axs[1]
    ax.barh(y, b.null_over_real_links, color=np.where(b.null_over_real_links > 0.5, "#c0392b", "#777"))
    ax.axvline(0.5, color="k", lw=0.6, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels(b.region)
    ax.invert_yaxis()
    ax.set_xlabel("linked fraction in time-shuffled catalogs / real linked fraction")
    ax.set_title("how much of the clustering a shuffled catalog reproduces (verifier)")
    fig.savefig(FIG / "fig_cluster_robustness.png")
    plt.close(fig)


def fig_surface_activity():
    s = pd.read_csv(OUT / "surfaces_v1_interp.csv")
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    regs = [r for r in REGIONS if r in set(s.cat_region)]
    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(regs):
        g = s[s.cat_region.eq(r)]
        axs[0].scatter(g.pct_background, g.pct_swarm, s=np.sqrt(g.n_events) * 2, color=cmap(i % 10), alpha=0.7,
                       label=r, edgecolors=np.where(g.J_sub80.ge(0.5), "k", "none"))
        axs[1].scatter(g.n_events, g.J_sub80, s=12, color=cmap(i % 10), alpha=0.8)
    for _, r in s[(s.pct_swarm > 30) | (s.n_events > 2000)].iterrows():
        axs[0].text(r.pct_background, r.pct_swarm, r.surface_id.replace("v1_", ""), fontsize=5)
    axs[0].set_xlabel("% background (events above Mc on the surface)")
    axs[0].set_ylabel("% in swarms")
    axs[0].legend(fontsize=6, ncol=2)
    axs[0].set_title("activity style per v1 surface (size ~ sqrt(members); black edge: recovered on 80% subsample)")
    axs[1].set_xscale("log")
    axs[1].axhline(0.5, color="k", ls=":", lw=0.6)
    axs[1].set_xlabel("fit members")
    axs[1].set_ylabel("Jaccard overlap, 80% subsample rerun (verifier)")
    axs[1].set_title("detection stability of each surface")
    fig.savefig(FIG / "fig_surface_activity.png")
    plt.close(fig)


def fig_axial_rate(c):
    A = c[c.is_eq & c.region.eq("Axial")]
    t = A.t
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.2), constrained_layout=True)
    w = A.set_index("t").resample("7D").size()
    axs[0].step(w.index, w.values, where="post", color="k")
    axs[0].axvline(pd.Timestamp("2015-04-24 05:00", tz="UTC"), color="#c0392b", lw=0.8)
    axs[0].set_ylabel("located events per week")
    axs[0].set_title("Axial: weekly count (red: eruption onset 2015-04-24)")
    ab = A[A.tier.isin(["A", "B"])]
    axs[1].hist(ab.bsf, bins=np.arange(-0.2, 5, 0.1), color="#1f5fa8", alpha=0.7, label=f"v5.1 tier A-B (n={len(ab):,})")
    axs[1].hist(ab.v4_depth + ab.surface_km, bins=np.arange(-4, 5, 0.1), histtype="step", color="#777",
                label="v4 read as km bsl, converted to bsf")
    axs[1].axvline(2.0, color="k", ls=":", lw=0.8)
    axs[1].set_xlabel("depth below seafloor (km)")
    axs[1].legend(fontsize=6)
    axs[1].set_title("Axial depth below seafloor; 90% within 2.0 km in Waldhauser et al. (2020)")
    fig.savefig(FIG / "fig_axial_rate_depth.png")
    plt.close(fig)


def main():
    c = pd.read_pickle(BIG / "events_joined.pkl")
    fa = pd.read_csv(CLU / "families_zbz_v51.csv")
    fig_overview(c)
    fig_depths(c)
    for r in REGIONS:
        fig_region(c, r)
    made = []
    for fam, title in KEY_SWARMS:
        o = fig_swarm(c, fam, title, fa)
        if o:
            made.append(fam)
    (OUT / "figs" / "key_swarms.json").write_text(json.dumps({str(k): t for k, t in KEY_SWARMS if k in made}, indent=1))
    fig_msh(c)
    fig_robustness()
    fig_surface_activity()
    fig_axial_rate(c)
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()

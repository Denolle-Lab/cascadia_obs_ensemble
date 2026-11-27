"""Supplementary figure: how the catalog versions differ, region by region (restyle of the old
figS_catalog_compare, workflow/11_compare/fig_compare.py).

Inputs: the catalog comparison of the catalog's production run (workflow/11_compare/compare_catalogs.py with
workflow/pipeline/configs/catalogs_v6.1.yaml; reference v4.1): summary.csv, anss.csv, registry.json and the common
event tables big/<catalog>.parquet. Path: config key `catalog_compare` when present, else
<faults>/../validation/catalog_compare (the v6.1 production folder).

A  depth p10-p50-p90 per region and catalog        B  depth minus ANSS for events with a station within
C  events above the local surface (%)                 max(depth, 5 km): median and MAD (>= 10 events)
D  depth change, E  horizontal shift of the latest catalog against the reference (same v4 events)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import paper_style as S
from paper_v2 import data as D

ORDER = ["Explorer", "Endeavour", "JdF ridge/plate", "Axial", "Puget/Georgia", "Offshore forearc", "Cascades",
         "Other onshore", "Blanco", "Gorda", "Mendocino"]
# catalog versions are ordered: the reference (GENIE + GraphDD, 1D) grey, the 3D NonLinLoc versions a blue ramp,
# darkest for the paper catalog; one marker per version as the second channel
RAMP = ["#8fb3d9", "#4a7fb8", "#16365e"]
REF_COL = "#9a9a9a"
MARKERS = ["o", "s", "D", "^", "v", "h"]


def compare_dir(cfg):
    if "catalog_compare" in cfg:
        return cfg.path("catalog_compare")
    return cfg.path("faults").parent / "validation" / "catalog_compare"


def styles(ids, ref):
    others = [c for c in ids if c != ref]
    ramp = RAMP[-len(others):] if len(others) <= len(RAMP) else [S.INK] * len(others)
    st = {ref: dict(color=REF_COL, marker="o", ms=3.6, mfc="white")}
    for k, c in enumerate(others):
        st[c] = dict(color=ramp[k], marker=MARKERS[k + 1], ms=3.4 if MARKERS[k + 1] != "D" else 3.0,
                     mfc=ramp[k])
    return st


def stepfill(ax, edges, v, col):
    """Histogram as a light wash under a thin step outline (no heavy blocks)."""
    x = np.repeat(edges, 2)[1:-1]
    y = np.repeat(v, 2)
    ax.fill_between(x, 0, y, color=col, alpha=0.18, lw=0, zorder=2)
    ax.plot(np.r_[edges[0], x, edges[-1]], np.r_[0, y, 0], color=col, lw=0.7, zorder=3)
    ax.set_ylim(0, float(v.max()) * 1.12)


def _dist_km(lat1, lon1, lat2, lon2):
    return D.hav(lat1, lon1, lat2, lon2)


def make(cfg):
    C = compare_dir(cfg)
    reg = json.loads((C / "registry.json").read_text())
    ref = reg["reference"]
    Sm, A, P = (pd.read_csv(C / f) for f in ("summary.csv", "anss.csv", "pairs.csv"))
    same = P[(P.against == ref) & (P.dh_p90 == 0) & (P.dz_p10 == 0) & (P.dz_p90 == 0)].catalog.tolist()
    ids = [e["id"] for e in reg["catalogs"] if e["id"] not in same]
    latest = ids[-1]
    st = styles(ids, ref)
    regs = [r for r in ORDER if r in set(Sm.region)]
    ny = len(regs)
    yv = {r: ny - 1 - i for i, r in enumerate(regs)}
    off = {c: (k - (len(ids) - 1) / 2) * 0.17 for k, c in enumerate(ids)}

    W = S.COL2
    left, pw, gap = 2.45, 4.75, 0.3
    top_h, bot_h = 6.3, 3.0
    y_bot, y_top = 0.75, 0.75 + bot_h + 1.25
    H = y_top + top_h + 1.15
    fig = S.fig_cm(W, H)
    axs = [S.axes_cm(fig, left + i * (pw + gap), y_top, pw, top_h) for i in range(3)]

    def frame(ax, xlim, xlabel, labels):
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.6, ny - 0.4)
        ax.set_yticks(range(ny))
        ax.set_yticklabels(regs[::-1] if labels else [])
        ax.tick_params(axis="y", length=0)
        for i in range(ny):
            ax.axhline(i, color=S.GRID, lw=S.LW["grid"], zorder=0)
        S.clean(ax, left=False)
        ax.set_xlabel(xlabel)

    # A: depth p10-p50-p90
    ax = axs[0]
    for c in ids:
        s = Sm[(Sm.catalog == c) & Sm.region.isin(regs)]
        for r in s.itertuples():
            y = yv[r.region] + off[c]
            ax.plot([r.depth_p10, r.depth_p90], [y, y], color=st[c]["color"], lw=0.8, solid_capstyle="butt", zorder=2)
            ax.plot(r.depth_p50, y, ls="", marker=st[c]["marker"], ms=st[c]["ms"], mfc=st[c]["mfc"],
                    mec=st[c]["color"], mew=0.6, zorder=3)
    xmax = float(np.ceil((Sm[Sm.region.isin(regs) & Sm.catalog.isin(ids)].depth_p90.max() + 2) / 5) * 5)
    frame(ax, (-5, xmax), "depth (km below sea level), p10–p50–p90", True)
    ax.axvline(0, color=S.AXIS, lw=0.4, zorder=1)

    # B: depth minus ANSS, near stations
    ax = axs[1]
    ax.axvline(0, color=S.AXIS, lw=0.4, zorder=1)
    for c in ids:
        g = A[(A.catalog == c) & (A.subset == "near") & A.region.isin(regs) & (A.n >= 10)]
        for r in g.itertuples():
            y = yv[r.region] + off[c]
            ax.plot([r.dz_p50 - r.dz_mad, r.dz_p50 + r.dz_mad], [y, y], color=st[c]["color"], lw=0.8,
                    solid_capstyle="butt", zorder=2)
            ax.plot(np.clip(r.dz_p50, -14.5, 14.5), y, ls="", marker=st[c]["marker"], ms=st[c]["ms"],
                    mfc=st[c]["mfc"], mec=st[c]["color"], mew=0.6, zorder=3)
    frame(ax, (-15, 15), "depth − ANSS (km), events near a station", False)
    nmiss = [r for r in regs if A[(A.region == r) & (A.subset == "near") & (A.n >= 10)].empty]
    for r in nmiss:
        ax.text(0.03, yv[r], "fewer than 10 events", transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=S.FS["small"], color=S.INK2, path_effects=S.halo(1.6), zorder=4)

    # C: above the surface
    ax = axs[2]
    for c in ids:
        s = Sm[(Sm.catalog == c) & Sm.region.isin(regs)]
        ax.plot(s.pct_above_surface, s.region.map(yv) + off[c], ls="", marker=st[c]["marker"], ms=st[c]["ms"],
                mfc=st[c]["mfc"], mec=st[c]["color"], mew=0.6, zorder=3)
    frame(ax, (-2, 75), "events above the local surface (%)", False)
    for ax, s in zip(axs, "ABC"):
        S.letter(ax, s, dx=-0.35 if s != "A" else -2.4, dy=0.12)

    # legend above the top row
    lab = {e["id"]: e for e in reg["catalogs"]}
    hs = [Line2D([], [], ls="-", lw=0.8, color=st[c]["color"], marker=st[c]["marker"], ms=st[c]["ms"],
                 mfc=st[c]["mfc"], mec=st[c]["color"], mew=0.6,
                 label=f"{c}{' (reference)' if c == ref else ''}") for c in ids]
    fig.legend(handles=hs, loc="lower left", ncol=len(ids), bbox_to_anchor=(left / W, (y_top + top_h + 0.62) / H),
               frameon=False, borderaxespad=0, columnspacing=1.6, handlelength=1.8)

    # D, E: latest against the reference (same v4 events)
    big = C / "big"
    t = pd.read_parquet(big / f"{latest}.parquet")
    r0 = pd.read_parquet(big / f"{ref}.parquet").drop_duplicates("v4_id").set_index("v4_id")
    m = t[t.located].dropna(subset=["v4_id"]).join(r0[["lat", "lon", "depth"]], on="v4_id", rsuffix="_r")
    m = m.dropna(subset=["lat_r"])
    m = m[np.isfinite(m.depth) & np.isfinite(m.depth_r)]
    dz = (m.depth - m.depth_r).to_numpy()
    dh = _dist_km(m.lat.to_numpy(), m.lon.to_numpy(), m.lat_r.to_numpy(), m.lon_r.to_numpy())
    dh = dh[np.isfinite(dh)]
    colL = st[latest]["color"]
    wD = 2 * pw + gap
    axD = S.axes_cm(fig, left, y_bot, wD, bot_h)
    axE = S.axes_cm(fig, left + wD + gap + 0.9, y_bot, pw - 0.9, bot_h)
    bins = np.arange(-20, 20.01, 0.5)
    hD, _ = np.histogram(np.clip(dz, -19.999, 19.999), bins)
    stepfill(axD, bins, 100 * hD / len(dz), colL)
    axD.set_xlim(-20, 20)
    axD.set_ylabel("share (%)")
    axD.set_xlabel(f"depth {latest} − {ref} (km; + = deeper)")
    S.clean(axD, grid_axis="y")
    axD.text(0.99, 0.97, f"n = {len(dz):,}; median {np.median(dz):+.1f} km", transform=axD.transAxes,
             ha="right", va="top", fontsize=S.FS["small"])
    axD.text(0.01, 0.97, f"≤ −20 km: {100 * np.mean(dz <= -20):.1f} %\n≥ +20 km: {100 * np.mean(dz >= 20):.1f} %",
             transform=axD.transAxes, ha="left", va="top", fontsize=S.FS["small"], linespacing=1.15)
    binsE = np.arange(0, 20.01, 0.25)
    hE, _ = np.histogram(np.clip(dh, 0, 19.999), binsE)
    stepfill(axE, binsE, 100 * hE / len(dh), colL)
    axE.set_xlim(0, 20)
    axE.set_xlabel(f"horizontal shift {latest} – {ref} (km)")
    axE.set_ylabel("share (%)")
    S.clean(axE, grid_axis="y")
    axE.text(0.99, 0.97, f"median {np.median(dh):.1f} km\n> 5 km: {100 * np.mean(dh > 5):.0f} %",
             transform=axE.transAxes, ha="right", va="top", fontsize=S.FS["small"], linespacing=1.15)
    S.letter(axD, "D", dx=-2.4, dy=0.12)
    S.letter(axE, "E", dx=-0.85, dy=0.12)
    size = S.save(fig, cfg["out"], "figS_catalog_compare")
    labs = "; ".join(f"{c}: {lab[c].get('method', '')}".rstrip(": ") for c in ids)
    cap = (f"Catalog versions compared region by region against the reference {ref} ({', '.join(ids)}). "
           f"(A) Depth below sea level: median (symbol) and 10th-90th percentiles (bar). (B) Depth minus the ANSS "
           f"depth for events with a station within max(depth, 5 km): median and median absolute deviation, regions "
           f"with at least 10 such events. (C) Share of events above the local solid surface. (D) Depth change and "
           f"(E) horizontal shift of {latest} against {ref} for the {len(dz):,} events located in both (linked by "
           f"the v4 event id); values beyond ±20 km in the end bins.")
    return [dict(name="figS_catalog_compare", files=["figS_catalog_compare.pdf", "figS_catalog_compare.png"],
                 size_cm=size, caption=cap, before=["figS_catalog_compare.png"],
                 inputs=[str(C / "summary.csv"), str(C / "anss.csv"), str(C / "pairs.csv"), str(C / "registry.json"),
                         str(big / f"{latest}.parquet"), str(big / f"{ref}.parquet")],
                 facts=dict(catalogs=ids, reference=ref, latest=latest, n_linked=int(len(dz)),
                            dz_median=round(float(np.median(dz)), 2), dh_median=round(float(np.median(dh)), 2),
                            dh_gt5_pct=round(float(100 * np.mean(dh > 5)), 1), methods=labs))]

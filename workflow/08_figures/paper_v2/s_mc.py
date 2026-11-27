"""Supplementary Mc maps on the catalog's converged magnitudes (v6.1: mc_maps_v61, workflow/06_analysis/v4_mc_maps.py).

figS_mc_methods  Mc at every 0.1 degree node by three methods: maximum curvature + 0.2, MBS, GFT 95/90 with the
                 Woessner and Wiemer (2005) rule (mc_best); one shared colour scale
figS_mc_periods  mc_best per network period (2010, 2011-2012, 2013-2015, 2016) and its range over the periods
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xarray as xr

import paper_style as S
from paper_v2 import data as D

REGION = [-131.0, -119.5, 39.0, 51.0]
MC_RANGE = (0.0, 3.0)
CM_RANGE = S.scm.scm_cmap("bilbao", reverse=True, lo=0.04, hi=0.94)


def _edges(c):
    c = np.asarray(c, float)
    d = np.diff(c)
    return np.r_[c[0] - d[0] / 2, c[:-1] + d / 2, c[-1] + d[-1] / 2]


def mc_layer(m, da, cmap, vmin, vmax):
    lon, lat = da.lon.values, da.lat.values
    v = np.ma.masked_invalid(da.values.astype(float))
    return m.ax.pcolormesh(_edges(lon), S.merc(_edges(lat)), v, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat",
                           zorder=2, rasterized=True, edgecolors="none")


def draw_stations(m, st):
    land, sea = st[~st.sea], st[st.sea]
    m.scatter(land.lon, land.lat, s=2.2, marker="^", c="#3a3a3a", edgecolors="none", zorder=9)
    m.scatter(sea.lon, sea.lat, s=4.0, marker="v", facecolors="white", edgecolors=S.INK, linewidths=0.3, zorder=9)


def base(fig, ax, ctx, labels, px, ticks, xt=None):
    m = S.MapAx(ax, REGION, labels=labels, ticks=ticks)
    if xt is not None:                       # narrow panels: fewer longitude labels, none at the frame edge
        ax.set_xticks(xt)
        ax.set_xticklabels([f"{abs(x):g}°W" for x in xt])
    ctx["relief"].draw(m, px=px)
    return m


def overlay(m, ctx, plate_scale=0.7):
    S.draw_coast(m, ctx["coast"], ctx["borders"], zorder=4)
    S.draw_plates(m, ctx["pb"], scale=plate_scale, zorder=6, teeth_km=60.0)


def station_table(cfg, years=None):
    s = D.stations(cfg)
    s = s[s.used]
    if years is not None:
        y0, y1 = years
        s = s[(s.epoch_start <= pd.Timestamp(f"{y1}-12-31")) & (s.epoch_end >= pd.Timestamp(f"{y0}-01-01"))]
    return s.drop_duplicates(["network", "station"])


def cbar(fig, mappable, x, y, w, label, ticks):
    cax = S.axes_cm(fig, x, y, w, 0.14)
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal", ticks=ticks)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=S.FS["small"], length=1.6, pad=1)
    cb.set_label(label, fontsize=S.FS["base"], labelpad=2)
    return cb


def fig_methods(cfg, ctx, ds, run):
    W = S.COL2
    left, gap, right = 0.95, 0.22, 0.2
    w = (W - left - right - 2 * gap) / 3
    h = float(S.map_height(REGION, w))
    top, bot = 0.75, 1.5
    H = top + h + bot
    fig = S.fig_cm(W, H)
    st = station_table(cfg)
    panels = [("mc_maxc", "maximum curvature + 0.2"), ("mc_mbs", "MBS"),
              ("mc_best", "GFT 95/90 (Woessner and Wiemer 2005 rule)")]
    stats = {}
    for i, (var, title) in enumerate(panels):
        ax = S.axes_cm(fig, left + i * (w + gap), bot, w, h)
        m = base(fig, ax, ctx, "WS" if i == 0 else "S", 900, (4.0, 2.0))
        pc = mc_layer(m, ds[var], S.CM_MC, *MC_RANGE)
        overlay(m, ctx)
        draw_stations(m, st)
        ax.set_title(title, loc="left", fontsize=S.FS["base"], pad=3)
        S.letter(ax, "ABC"[i], dx=-0.85 if i == 0 else -0.05, dy=0.32)
        v = ds[var].values
        stats[var] = dict(nodes=int(np.isfinite(v).sum()), median=round(float(np.nanmedian(v)), 2),
                          p10=round(float(np.nanpercentile(v, 10)), 2), p90=round(float(np.nanpercentile(v, 90)), 2))
    cbar(fig, pc, left + w + gap + 0.3, 0.8, w - 0.6, "Mc (magnitude of completeness)", np.arange(0, 3.01, 0.5))
    size = S.save(fig, cfg["out"], "figS_mc_methods")
    n_sea, n_land = int(st.sea.sum()), int((~st.sea).sum())
    cap = (f"Magnitude of completeness of catalog {run.get('catalog_label', cfg['label'])} at 0.1° nodes "
           f"(the {run.get('n_events', 250)} nearest events within {run.get('rmax_km', 50):g} km; flagged events "
           f"left out; {run.get('nboot', 200)} bootstrap draws), by three methods: (A) maximum curvature + 0.2, "
           f"(B) MBS, (C) GFT at 95 % goodness of fit, else 90 %, else maximum curvature (Woessner and Wiemer, 2005). "
           f"Nodes with Mc: {stats['mc_best']['nodes']:,}; median Mc (C) {stats['mc_best']['median']:.2f}. "
           f"Inverted triangles: {n_sea} ocean-bottom stations; small triangles: {n_land} land stations.")
    return dict(name="figS_mc_methods", files=["figS_mc_methods.pdf", "figS_mc_methods.png"], size_cm=size,
                caption=cap, before=["fig2a_mc_three_methods.png"], inputs=["mc_maps (mc_grid.nc, run_params.json)",
                                                                            "stations", "pb2002", "plate_axes"],
                facts=stats)


def fig_periods(cfg, ctx, dp, run):
    W = S.COL2
    left, gap, right = 0.95, 0.15, 0.2
    n = len(dp.period) + 1
    w = (W - left - right - (n - 1) * gap) / n
    h = float(S.map_height(REGION, w))
    top, bot = 0.75, 1.5
    H = top + h + bot
    fig = S.fig_cm(W, H)
    periods = [str(p) for p in dp.period.values]
    pers = run.get("periods", {})
    stats = {}
    for i, p in enumerate(periods):
        ax = S.axes_cm(fig, left + i * (w + gap), bot, w, h)
        m = base(fig, ax, ctx, "WS" if i == 0 else "S", 600, (4.0, 3.0), xt=[-128, -124])
        pc = mc_layer(m, dp.mc_best.sel(period=p), S.CM_MC, *MC_RANGE)
        overlay(m, ctx, 0.6)
        draw_stations(m, station_table(cfg, pers.get(p)))
        ax.set_title(p.replace("-", "–"), loc="left", fontsize=S.FS["base"], pad=3)
        S.letter(ax, "ABCDE"[i], dx=-0.85 if i == 0 else -0.05, dy=0.32)
        v = dp.mc_best.sel(period=p).values
        stats[p] = dict(nodes=int(np.isfinite(v).sum()), median=round(float(np.nanmedian(v)), 2))
    stack = dp.mc_best.values
    cnt = np.isfinite(stack).sum(0)
    with np.errstate(invalid="ignore"):
        rng = np.where(cnt >= 2, np.nanmax(stack, 0) - np.nanmin(stack, 0), np.nan)
    da = xr.DataArray(rng, coords=dict(lat=dp.lat, lon=dp.lon), dims=("lat", "lon"))
    ax = S.axes_cm(fig, left + (n - 1) * (w + gap), bot, w, h)
    m = base(fig, ax, ctx, "S", 600, (4.0, 3.0), xt=[-128, -124])
    pr = mc_layer(m, da, CM_RANGE, 0.0, 1.0)
    overlay(m, ctx, 0.6)
    draw_stations(m, station_table(cfg))
    ax.set_title("largest − smallest", loc="left", fontsize=S.FS["base"], pad=3)
    S.letter(ax, "E", dx=-0.05, dy=0.32)
    stats["range"] = dict(nodes=int(np.isfinite(rng).sum()), median=round(float(np.nanmedian(rng)), 2),
                          p90=round(float(np.nanpercentile(rng, 90)), 2))
    x0 = left + 0.5 * w
    cbar(fig, pc, x0, 0.8, 3 * (w + gap) - 0.2, "Mc (GFT 95/90, else maximum curvature)", np.arange(0, 3.01, 0.5))
    cbar(fig, pr, left + (n - 1) * (w + gap) + 0.25, 0.8, w - 0.5, "Mc range over the periods", [0, 0.5, 1.0])
    size = S.save(fig, cfg["out"], "figS_mc_periods")
    cap = (f"Magnitude of completeness of catalog {run.get('catalog_label', cfg['label'])} by network period "
           f"(mc_best: GFT 95/90, else maximum curvature; the {run.get('n_events_period', 150)} nearest events within "
           f"{run.get('rmax_km', 50):g} km): (A-D) " + ", ".join(f"{p} ({stats[p]['nodes']:,} nodes)" for p in periods)
           + f"; (E) largest minus smallest Mc at nodes with a value in at least two periods "
           f"({stats['range']['nodes']:,} nodes, median {stats['range']['median']:.2f}). Triangles: stations operating "
           f"in the period (E: all).")
    return dict(name="figS_mc_periods", files=["figS_mc_periods.pdf", "figS_mc_periods.png"], size_cm=size,
                caption=cap, before=["fig2b_mc_periods.png"],
                inputs=["mc_maps (mc_grid_periods.nc, run_params.json)", "stations", "pb2002", "plate_axes"],
                facts=stats)


def make(cfg):
    ctx = D.context(cfg)
    mc = cfg.path("mc_maps")
    run = json.loads((mc / "run_params.json").read_text())
    ds = xr.open_dataset(mc / "mc_grid.nc")
    dp = xr.open_dataset(mc / "mc_grid_periods.nc")
    return [fig_methods(cfg, ctx, ds, run), fig_periods(cfg, ctx, dp, run)]

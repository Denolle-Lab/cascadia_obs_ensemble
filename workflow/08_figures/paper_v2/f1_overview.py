"""Fig. 1: plate boundaries and data (D-B107 framing).

A  map of the margin: the paper set of the catalog (tier A-C earthquakes; flagged events and the review P1 set left
   out) coloured by the type of its plate-boundary segment, segment codes with their structure class, plate traces
   (plate boundaries v2 adopted, D-B117; PB2002 dashed for reference), NNR-MORVEL56 plate motions
B  stations used by the catalog: ocean-bottom stations coloured by their first year, land stations open
C  depth against longitude (depth-good events, same colours), Slab2 at three latitudes
D  events per week; E  operating stations per week (ocean-bottom, land): two panels on one time axis (no dual axis)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

import paper_style as S
from paper_v2 import data as D

REGION = [-131.0, -119.5, 39.0, 51.0]
NAME = "fig1_overview"
# label offsets (km east, km north) from the segment's centre (segment_axes.csv lon0, lat0); a leader line is drawn
# when the offset is not zero. Positions not listed sit on the centre. "at": absolute (lon, lat) instead.
LABEL_OFF = {
    "AX": (-58, -12), "JR": (-52, 2), "EN": (42, 6), "EXR": (8, -34), "SOV": (-52, 8), "EXP": (38, 30),
    "GR": (-55, 0), "BL": (-5, -42), "MFZ": (-10, -34), "JP": (0, 0), "GP": (-35, 40), "MTJ": (-48, -38),
    "GS": (62, 16), "FA": (-62, 2), "SIF": (-70, -30), "IS": (-58, 28), "PU": (48, 6), "MSH": (52, -14),
    "CAS": {"at": (-120.75, 47.1)},
}
CLASS_STYLE = {"resolved": dict(fc=S.INK, ec=S.INK, tc="white"),
               "marginal": dict(fc="#8f8f8f", ec="#8f8f8f", tc="white"),
               "not resolved": dict(fc="white", ec=S.INK, tc=S.INK)}
SLAB_LATS = (42.0, 45.0, 48.0)
T0, T1 = pd.Timestamp("2010-01-01"), pd.Timestamp("2017-01-01")


def _km_to_deg(lat, de, dn):
    return de / (D.KM * np.cos(np.radians(lat))), dn / D.KM


def seg_label(m, x, y, s, cls, fs=None):
    st = CLASS_STYLE.get(cls, CLASS_STYLE["not resolved"])
    return m.text(x, y, s, fontsize=fs or S.FS["small"], fontweight="bold", ha="center", va="center", color=st["tc"],
                  zorder=30, bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.25", fc=st["fc"], ec=st["ec"], lw=0.4))


def panel_map_events(fig, ax, cfg, ctx, E, segs, axes_tab):
    m = S.MapAx(ax, REGION, labels="WS")
    ctx["relief"].draw(m, px=1500)
    S.draw_coast(m, ctx["coast"], ctx["borders"])
    if cfg.get("plate_axes_reference") and ctx.get("pb_ref"):
        S.draw_plates_reference(m, ctx["pb_ref"])
    S.draw_plates(m, ctx["pb"], scale=0.9, zorder=6)
    # events: 'other' underneath, then every typed event by magnitude (large on top)
    oth = E[E.seg_type.eq("other") | E.seg_type.isna()]
    m.scatter(oth.lon, oth.lat, s=S.msize(oth.M, 0.75), c=S.TYPE_COL["other"], edgecolors="none", alpha=0.55,
              zorder=4, rasterized=True)
    ty = E[~E.index.isin(oth.index)].sort_values("M", na_position="first")
    col = ty.seg_type.map(S.TYPE_COL).to_numpy()
    big = ty.M.fillna(0).to_numpy() >= 4.0
    m.scatter(ty.lon[~big], ty.lat[~big], s=S.msize(ty.M[~big], 0.75), c=col[~big], edgecolors="none", alpha=0.85,
              zorder=5, rasterized=True)
    m.scatter(ty.lon[big], ty.lat[big], s=S.msize(ty.M[big], 0.75), c=col[big], edgecolors="white", linewidths=0.3,
              zorder=8, rasterized=True)
    # plate motions (NNR-MORVEL56): moving plate relative to the fixed one
    for mov, fix, lat, lon, lab, side in (("JF", "NA", 45.3, -128.3, "JdF–NA", "left"),
                                         ("PA", "NA", 39.22, -127.35, "PA–NA", "right"),
                                         ("JF", "PA", 47.3, -128.65, "JdF–PA", "above")):
        rate, az = S.draw_plate_motion(m, mov, fix, lat, lon, km_per_mm=2.6, label=False)
        L = rate * 2.6
        lat2 = lat + 0.5 * L * np.cos(np.radians(az)) / D.KM
        lon2 = lon + 0.5 * L * np.sin(np.radians(az)) / (D.KM * np.cos(np.radians(lat)))
        txt = f"{lab}\n{rate:.0f} mm/yr"
        if side == "above":
            m.text(lon2, lat2 + 0.2, txt, fontsize=S.FS["small"], ha="center", va="bottom", color=S.INK,
                   path_effects=S.halo(1.5), zorder=26, linespacing=1.0)
        elif side == "left":
            m.text(lon2 - 0.2, lat2 + 0.05, txt, fontsize=S.FS["small"], ha="right", va="center", color=S.INK,
                   path_effects=S.halo(1.5), zorder=26, linespacing=1.0)
        elif side == "below":
            m.text(lon2 + 0.12, lat2 - 0.12, txt, fontsize=S.FS["small"], ha="left", va="top", color=S.INK,
                   path_effects=S.halo(1.5), zorder=26, linespacing=1.0)
        else:
            m.text(lon2 + 0.18, lat2, txt, fontsize=S.FS["small"], ha="left", va="center", color=S.INK,
                   path_effects=S.halo(1.5), zorder=26, linespacing=1.0)
    # segment labels (plate-boundary segments only), style = structure class
    cls = dict(zip(segs.seg, segs["class"]))
    typ = dict(zip(segs.seg, segs.type))
    for _, r in axes_tab.iterrows():
        sg = r.seg
        if typ.get(sg, "other") == "other" or sg not in cls:
            continue
        off = LABEL_OFF.get(sg, (0, 0))
        if isinstance(off, dict):
            lx, ly = off["at"]
            seg_label(m, lx, ly, sg, cls[sg])
            continue
        dx, dy = _km_to_deg(r.lat0, *off)
        lx, ly = r.lon0 + dx, r.lat0 + dy
        if off != (0, 0):
            m.plot([r.lon0, lx], [r.lat0, ly], color=S.INK, lw=0.4, zorder=29, solid_capstyle="butt")
            m.scatter([r.lon0], [r.lat0], s=3.0, c=S.INK, zorder=29, edgecolors="none")
        seg_label(m, lx, ly, sg, cls[sg])
    m.scalebar(-122.75, 39.35, 100)
    m.north(-120.15, 49.95)
    return m


def panel_stations(fig, ax, cfg, ctx):
    m = S.MapAx(ax, REGION, labels="ES")
    ctx["relief"].draw(m, px=1500)
    S.draw_coast(m, ctx["coast"], ctx["borders"])
    S.draw_plates(m, ctx["pb"], scale=0.9, zorder=6)
    s = D.stations(cfg)
    s = s[s.used]
    g = s.groupby(["network", "station"]).agg(sea=("sea", "first"), t0=("epoch_start", "min"), lat=("lat", "first"),
                                              lon=("lon", "first")).reset_index()
    land, sea = g[~g.sea], g[g.sea].sort_values("t0")
    m.scatter(land.lon, land.lat, s=9, marker="^", facecolors="white", edgecolors="#3a3a3a", linewidths=0.4, zorder=9)
    years = np.arange(int(sea.t0.dt.year.min()), int(sea.t0.dt.year.max()) + 1)
    cols = S.scm.scm_colors("lipari", lo=0.06, hi=0.80, n=len(years))
    cmap = ListedColormap(cols)
    norm = BoundaryNorm(np.r_[years - 0.5, years[-1] + 0.5], cmap.N)
    sc = m.scatter(sea.lon, sea.lat, s=13, marker="v", c=sea.t0.dt.year, cmap=cmap, norm=norm, edgecolors=S.INK,
                   linewidths=0.35, zorder=10)
    # counts and the year key (inset over the open ocean, south-west)
    S.text_halo(ax, 0.97, 0.985, f"{len(sea)} ocean-bottom stations\n{len(land)} land stations",
                transform=ax.transAxes, ha="right", va="top", fs=S.FS["small"], zorder=30, linespacing=1.1)
    import matplotlib.patches as mpatches
    W, H = fig.get_size_inches() / S.CM
    b = ax.get_position()
    bw, bh = 3.7, 0.88                                             # key box (cm), lower left, over the open ocean
    ax.add_patch(mpatches.FancyBboxPatch((0.1 / (b.width * W), 0.07 / (b.height * H)), bw / (b.width * W),
                                         bh / (b.height * H), boxstyle="round,pad=0,rounding_size=0.01",
                                         transform=ax.transAxes, fc="white", ec="none", alpha=0.88, zorder=28))
    cax = fig.add_axes([b.x0 + 0.3 / W, b.y0 + 0.39 / H, (bw - 0.4) / W, 0.12 / H])
    cax.set_zorder(ax.get_zorder() + 1)
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal", ticks=years)
    cb.ax.tick_params(labelsize=S.FS["small"], length=1.5, pad=1)
    cb.outline.set_linewidth(0.4)
    cb.ax.set_xticklabels([str(y) for y in years])
    cb.ax.xaxis.set_label_position("top")
    cb.set_label("first year of the ocean-bottom station", fontsize=S.FS["small"], labelpad=2)
    return m, len(sea), len(land), years


def panel_depth(ax, cfg, E):
    Eg = E[E.depth_good]
    oth = Eg[Eg.seg_type.eq("other") | Eg.seg_type.isna()]
    ty = Eg[~Eg.index.isin(oth.index)]
    ax.scatter(oth.lon, oth.depth, s=0.5, c=S.TYPE_COL["other"], edgecolors="none", alpha=0.45, rasterized=True,
               zorder=2)
    ax.scatter(ty.lon, ty.depth, s=0.5, c=ty.seg_type.map(S.TYPE_COL).to_numpy(), edgecolors="none", alpha=0.6,
               rasterized=True, zorder=3)
    sl = D.slab_fn(cfg)
    lon = np.linspace(REGION[0], REGION[1], 600)
    dashes = {42.0: (0, (1, 0)), 45.0: (0, (4, 1.6)), 48.0: (0, (1.2, 1.2))}
    for lat in SLAB_LATS:
        z = sl(lon, np.full_like(lon, lat))
        ok = np.isfinite(z) & (z <= 62)
        if ok.sum() < 2:
            continue
        S.cased(ax, lon[ok], z[ok], color=S.INK, lw=0.7, ls=dashes[lat], zorder=6)
    ax.set_xlim(REGION[0], REGION[1])
    ax.set_ylim(60, -2)
    ax.set_ylabel("depth (km)")
    xt = np.arange(-130, -121, 2.0)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{abs(x):g}°W" for x in xt])
    S.clean(ax)
    ax.set_yticks([0, 20, 40, 60])
    # Slab2 key (lines differ by dash; one key in the empty upper right)
    x0, y0 = 0.03, 0.30
    for i, lat in enumerate(SLAB_LATS):
        yy = y0 - i * 0.11
        ax.plot([x0, x0 + 0.06], [yy, yy], transform=ax.transAxes, color=S.INK, lw=0.7, ls=dashes[lat], zorder=8)
        ax.text(x0 + 0.075, yy, f"Slab2 at {lat:g}°N", transform=ax.transAxes, fontsize=S.FS["small"], va="center",
                ha="left", zorder=8, path_effects=S.halo(1.4))
    return len(Eg)


def weekly(E):
    t = E.t.dt.tz_convert(None) if E.t.dt.tz is not None else E.t
    return pd.Series(1, index=pd.DatetimeIndex(t)).sort_index().resample("7D", origin=T0).sum()


def station_counts(cfg, days):
    s = D.stations(cfg)
    s = s[s.used]
    out = {}
    for sea, key in ((True, "sea"), (False, "land")):
        g = s[s.sea == sea]
        a0, a1 = g.epoch_start.to_numpy(), g.epoch_end.to_numpy()
        st = (g.network + "." + g.station).to_numpy()
        cnt = []
        for d in days:
            on = (a0 <= d) & (a1 >= d)
            cnt.append(len(np.unique(st[on])))
        out[key] = np.array(cnt)
    return out


def panel_rate(ax, E):
    w = weekly(E)
    x = w.index + pd.Timedelta(days=3.5)
    ax.fill_between(x, 0, w.values, step="mid", color=S.INK, alpha=0.10, lw=0, zorder=2)
    ax.step(x, w.values, where="mid", color=S.INK, lw=0.6, zorder=3)
    ax.set_xlim(T0, T1)
    ymax = float(w.max())
    ax.set_ylim(0, ymax * 1.08)
    S.clean(ax, grid_axis="y", left=False)
    ax.spines["right"].set_visible(True)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("events / week")
    ax.tick_params(labelbottom=False)
    i = int(np.argmax(w.values))
    ax.annotate(f"Axial Seamount eruption\n{w.index[i]:%b %Y}", xy=(x[i], w.values[i]), xytext=(-8, -2),
                textcoords="offset points", ha="right", va="top", fontsize=S.FS["small"], color=S.INK,
                arrowprops=dict(arrowstyle="-", lw=0.4, color=S.INK, shrinkA=0, shrinkB=1))
    return w


def panel_stations_time(ax, cfg):
    days = pd.date_range(T0, T1, freq="7D")
    c = station_counts(cfg, days.to_numpy())
    ax.fill_between(days, 0, c["sea"], color=S.STATION_COL["sea"], alpha=0.12, lw=0, zorder=2)
    ax.plot(days, c["sea"], color=S.STATION_COL["sea"], lw=0.9, zorder=3, label="ocean-bottom")
    ax.plot(days, c["land"], color=S.INK, lw=0.9, zorder=3, label="land")
    ax.set_xlim(T0, T1)
    ax.set_ylim(0, max(c["land"].max(), c["sea"].max()) * 1.38)
    S.clean(ax, grid_axis="y", left=False)
    ax.spines["right"].set_visible(True)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("stations")
    ax.legend(loc="upper left", ncol=2, handlelength=1.4, borderaxespad=0.1, bbox_to_anchor=(0.0, 1.02))
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return c


def legend_strip(fig, ax, mscale):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    hs = S.type_handles(with_other=True, ms=4.6)
    hs[-1] = Line2D([], [], ls="", marker="o", ms=3.6, mfc=S.TYPE_COL["other"], mec="white", mew=0.4,
                    label="other")
    hs += [Line2D([], [], color=S.INK, lw=S.LW["ridge"] * 0.9, label="plate boundary v2"),
           Line2D([], [], color="#4d4d4d", lw=0.5, ls=(0, (2.2, 1.8)), label="PB2002")]
    ax.legend(handles=hs, loc="upper left", ncol=len(hs), bbox_to_anchor=(0.0, 1.06), frameon=False,
              handletextpad=0.25, columnspacing=0.9, title=None, borderaxespad=0)
    # second row: segment label key and magnitude sizes
    W, H = fig.get_size_inches() / S.CM
    b = ax.get_position()
    w_cm, h_cm = b.width * W, b.height * H
    y = 0.22
    ax.text(0.0, y, "segment code, structure:", fontsize=S.FS["small"], va="center", ha="left")
    x = 0.155
    for k, lab in (("resolved", "resolved"), ("marginal", "marginal"), ("not resolved", "not resolved")):
        st = CLASS_STYLE[k]
        ax.text(x, y, "BL" if k == "resolved" else "GR" if k == "marginal" else "AX", fontsize=S.FS["small"],
                fontweight="bold", ha="center", va="center", color=st["tc"],
                bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.25", fc=st["fc"], ec=st["ec"], lw=0.4))
        ax.text(x + 0.016, y, lab, fontsize=S.FS["small"], va="center", ha="left")
        x += 0.016 + 0.0055 * len(lab) + 0.035
    xm = 0.585
    ax.text(xm, y, "magnitude", fontsize=S.FS["small"], va="center", ha="left")
    mags = (2, 3, 4, 5, 6)
    xx = xm + 0.075
    for mg in mags:
        d_cm = np.sqrt(S.msize(mg, mscale)) / 72 * 2.54
        ax.scatter([xx], [y], s=S.msize(mg, mscale), c="#9a9a9a", edgecolors="white", linewidths=0.3, clip_on=False)
        ax.text(xx + 0.5 * d_cm / w_cm + 0.006, y, f"{mg}", fontsize=S.FS["small"], va="center", ha="left")
        xx += (d_cm + 0.42) / w_cm


def make(cfg):
    ctx = D.context(cfg)
    E = D.events(cfg)
    segs = D.segments(cfg)
    axes_tab = pd.read_csv(cfg.path("segment_axes"))
    W = S.COL2
    mw = 7.75
    mh = float(S.map_height(REGION, mw))
    top, strip, gap_maps, low_h, bottom = 0.15, 0.95, 1.0, 3.35, 0.75
    H = top + strip + mh + gap_maps + low_h + bottom
    fig = S.fig_cm(W, H)
    xA, xB = 1.0, 1.0 + mw + 0.3
    y_maps = bottom + low_h + gap_maps
    axL = S.axes_cm(fig, xA - 0.1, y_maps + mh + 0.12, W - xA, strip - 0.12)
    axA = S.axes_cm(fig, xA, y_maps, mw, mh)
    axB = S.axes_cm(fig, xB, y_maps, mw, mh)
    axC = S.axes_cm(fig, xA, bottom, mw, low_h)
    hD = (low_h - 0.42) / 2
    axD = S.axes_cm(fig, xB, bottom + hD + 0.42, mw, hD)
    axE = S.axes_cm(fig, xB, bottom, mw, hD, sharex=axD)
    panel_map_events(fig, axA, cfg, ctx, E, segs, axes_tab)
    _, n_sea, n_land, years = panel_stations(fig, axB, cfg, ctx)
    n_dg = panel_depth(axC, cfg, E)
    w = panel_rate(axD, E)
    c = panel_stations_time(axE, cfg)
    legend_strip(fig, axL, 0.75)
    for ax, s, dx in ((axA, "A", -0.95), (axB, "B", -0.05), (axC, "C", -0.95), (axD, "D", -0.05), (axE, "E", -0.05)):
        S.letter(ax, s, dx=dx, dy=0.06)
    size = S.save(fig, cfg["out"], NAME)
    n = len(E)
    counts = E.seg_type.fillna("other").value_counts()
    cap = (f"Plate boundaries and data. (A) {n:,} tier A-C earthquakes of catalog {cfg['label']} (flagged events and "
           f"the waveform-review P1 set left out), coloured by the type of their segment; symbol size scales with "
           f"magnitude (events without a magnitude drawn at M 0); grey: events outside the plate-boundary segments. Labels: segment codes; label fill gives the "
           f"structure class of model v0.4.2 near the segment (black resolved, grey marginal, white not resolved). "
           f"Black lines: plate boundaries v2 (D-B117): Gorda Ridge on its axial valley, Juan de Fuca Ridge on its "
           f"bathymetric axis (Endeavour from MAPRIDGES), Blanco and Mendocino transforms as straight segments fitted "
           f"to the relocated seismicity with their step-overs, Mendocino joined to the PB2002 triple junction by a "
           f"connector; subduction trace PB2002, teeth on the upper plate" +
           ("; thin dashed grey: PB2002 ridges and transforms for reference" if cfg.get("plate_axes_reference") else "")
           + ". Arrows: NNR-MORVEL56 relative plate motions. Relief: GMRT. "
           f"(B) {n_sea} ocean-bottom stations (inverted triangles, colour = first year, {years[0]}-{years[-1]}) "
           f"and {n_land} land stations (open triangles) with picks in the catalog. (C) Depth against longitude for "
           f"the {n_dg:,} depth-good events, same colours; lines: Slab2 at 42, 45 and 48°N. (D) Events per week "
           f"(7-day bins). (E) Ocean-bottom and land stations operating per week.")
    return [dict(name=NAME, files=[f"{NAME}.pdf", f"{NAME}.png"], size_cm=size, caption=cap,
                 before=["fig1_overview.png"],
                 inputs=["events_segments", "seismicity_metrics", "structure_feasibility", "segment_axes", "stations",
                         "slab2", "pb2002", "plate_axes", "cache (GMRT mosaic)"],
                 facts=dict(n_events=n, n_depth_good=n_dg, n_sea=n_sea, n_land=n_land,
                            by_type={k: int(v) for k, v in counts.items()}, peak_week=str(w.idxmax().date()),
                            peak_week_events=int(w.max())))]

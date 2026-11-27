"""Fig. 3: fault geometry per plate boundary (fault surfaces v1 refit on the catalog of the configuration).

One small multiple per segment with fault surfaces (default: the plan's list BL, MFZ, GR, GP, MTJ, GS, PU, MSH, CAS,
kept where the segment has >= 3 surfaces; config key fig3_segments overrides), in the segment's own frame (the strike
line of segment_axes.csv: x along strike, y across, + y = left of the strike direction):

  plan view   light GMRT relief, depth-good tier A-C events of the segment (darker grey: fit members of a surface),
              the map footprint of every surface (rectangle corners projected to the map; steep surfaces draw as
              lines), coloured by dip; across direction exaggerated where the zone is narrow (factor printed)
  depth view  the same events along strike against depth below sea level, and for every surface its along-strike
              extent over its seismogenic depth range (2-98 % of the depth-good fit members), coloured by dip

Surfaces belong to the segment of most of their fit members (surface_segments.csv). The Blanco member is the strand
view of the old Fig. 4 (exaggerated across the transform).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection, PolyCollection

import paper_style as S
import scm_palettes as scm
from paper_v2 import data as D

PLAN_SEGS = ["BL", "MFZ", "GR", "GP", "MTJ", "GS", "PU", "MSH"]   # CAS (1,000 km of arc, 9 small surfaces) left out
WIDE = {"BL"}                                                        # double-width slot: the strand view
CM_DIP = scm.scm_cmap("acton", reverse=True, lo=0.12, hi=1.0)     # 90 deg darkest
DIP_NORM = Normalize(30, 90)
ZMAX = 45.0


def frame_of(ax_row):
    u = np.array([ax_row.ux, ax_row.uy], float)
    u /= np.hypot(*u)
    n = np.array([-u[1], u[0]])
    return dict(lon0=float(ax_row.lon0), lat0=float(ax_row.lat0), u=u, n=n, strike=float(ax_row.strike))


def to_frame(fr, lon, lat):
    x, y = D.local_xy(lon, lat, fr["lon0"], fr["lat0"])
    return x * fr["u"][0] + y * fr["u"][1], x * fr["n"][0] + y * fr["n"][1]


def from_frame(fr, a, c):
    xe = a * fr["u"][0] + c * fr["n"][0]
    yn = a * fr["u"][1] + c * fr["n"][1]
    return fr["lon0"] + xe / (D.KM * np.cos(np.radians(fr["lat0"]))), fr["lat0"] + yn / D.KM


def frame_relief(relief, fr, xr, yr, nx=520, strength=0.40, ve=8.0):
    """Shaded relief resampled into the segment frame (true km in both directions for the shading)."""
    from matplotlib.colors import LightSource
    from scipy.interpolate import RegularGridInterpolator
    ny = max(40, int(nx * (yr[1] - yr[0]) / (xr[1] - xr[0])))
    ny = min(ny, 900)
    a = np.linspace(xr[0], xr[1], nx)
    c = np.linspace(yr[1], yr[0], ny)                       # row 0 = top (largest across)
    A, C = np.meshgrid(a, c)
    lon, lat = from_frame(fr, A, C)
    f = RegularGridInterpolator((relief.lat, relief.lon), relief.z, bounds_error=False, fill_value=np.nan)
    z = f(np.c_[lat.ravel(), lon.ravel()]).reshape(A.shape)
    z = np.where(np.isfinite(z), z, np.nanmedian(z) if np.isfinite(z).any() else 0.0)
    az = (405.0 - fr["strike"]) % 360                        # light from geographic NW
    hs = LightSource(azdeg=az, altdeg=42).hillshade(z, vert_exag=ve, dx=(a[1] - a[0]) * 1000,
                                                     dy=abs(c[1] - c[0]) * 1000, fraction=1.0)
    base = S.elevation_colors(z)
    shade = 1.0 - strength * (1.0 - hs)
    shade = shade / np.percentile(shade, 65)
    return np.clip(base * shade[..., None], 0, 1)


def nice_km(L):
    for v in (5, 10, 20, 25, 50, 100, 200):
        if v >= L * 0.18:
            return v
    return 200


def load(cfg):
    fdir = cfg.path("faults")
    ss = pd.read_csv(cfg.path("surface_segments"))
    summ = pd.read_csv(fdir / "fault_surfaces_v1_summary.csv")
    mem = pd.read_csv(fdir / "fault_surfaces_v1_members.csv")
    rect = {f["properties"]["surface_id"]: np.asarray(f["geometry"]["coordinates"][0], float)
            for f in json.loads((fdir / "fault_surfaces_v1_rectangles.geojson").read_text())["features"]}
    keep = ["surface_id", "strike", "dip", "top_km", "bottom_km", "events_depth_ok_2pct_km", "events_depth_ok_98pct_km",
            "strand", "kind", "n_events"]
    s = ss[["surface_id", "seg", "seg_share"]].merge(summ[[k for k in keep if k in summ.columns]], on="surface_id")
    # surfaces of the suspect Mount St Helens deep population (members deeper than 30 km, weekly analog signal,
    # v5.1 review P1; D-B111: not interpreted) are left out of the figure and counted in the caption
    flags = summ.set_index("surface_id")["flags"].reindex(s.surface_id).fillna("").astype(str).values
    s["suspect"] = np.char.find(flags.astype(str), "deep population") >= 0
    return s, mem, rect


def make(cfg):
    S.use()
    segtab = D.segments(cfg)
    s, mem, rect = load(cfg)
    counts = s.seg.value_counts()                           # the analysis' surface count (n_surfaces)
    n_suspect = int(s.suspect.sum())
    s = s[~s.suspect]
    want = cfg.get("fig3_segments", PLAN_SEGS)
    order = [g for g in segtab.seg if g in want and counts.get(g, 0) >= 3]
    names = dict(zip(segtab.seg, segtab.name))
    types = dict(zip(segtab.seg, segtab.type))
    axes = pd.read_csv(cfg.path("segment_axes")).set_index("seg")
    E = D.events(cfg, depth_good=True)
    members = set(mem.event_id.astype(np.int64))
    relief = D.context(cfg)["relief"]

    ncol = 3
    slots, k = [], 0                                          # (segment, row, column, span)
    for g in order:
        span = 2 if g in WIDE else 1
        r, c = divmod(k, ncol)
        if c + span > ncol:
            k = (r + 1) * ncol
            r, c = divmod(k, ncol)
        slots.append((g, r, c, span))
        k += span
    nrow = slots[-1][1] + 1 if slots else 1
    W = S.COL2
    left, right, gapx = 0.95, 0.25, 0.72
    w1 = (W - left - right - (ncol - 1) * gapx) / ncol
    Hp, Hs, gin, top_t, xlab, gapy = 2.85, 1.35, 0.10, 0.42, 0.52, 0.22
    row_h = top_t + Hp + gin + Hs + xlab + gapy
    leg_h = 1.25
    H = nrow * row_h + leg_h + 0.15
    fig = S.fig_cm(W, H)
    letters = "ABCDEFGHIJKL"
    info = []
    for k, (seg, r, cidx, span) in enumerate(slots):
        w = span * w1 + (span - 1) * gapx
        x0 = left + cidx * (w1 + gapx)
        y_top = H - 0.1 - r * row_h - top_t
        axp = S.axes_cm(fig, x0, y_top - Hp, w, Hp)
        axd = S.axes_cm(fig, x0, y_top - Hp - gin - Hs, w, Hs)
        fr = frame_of(axes.loc[seg])
        e = E[E.seg == seg]
        ea, ec = to_frame(fr, e.lon.values, e.lat.values)
        sg = s[s.seg == seg].sort_values("dip")
        # along range: events p1-p99 and the surfaces
        corners = {sid: rect[sid] for sid in sg.surface_id if sid in rect}
        ca = np.concatenate([to_frame(fr, v[:, 0], v[:, 1])[0] for v in corners.values()]) if corners else np.array([])
        lo_a = min(np.percentile(ea, 1), ca.min() if ca.size else np.inf)
        hi_a = max(np.percentile(ea, 99), ca.max() if ca.size else -np.inf)
        pad = 0.06 * (hi_a - lo_a)
        xr = (lo_a - pad, hi_a + pad)
        L = xr[1] - xr[0]
        sc = w / L                                            # cm per km along strike
        onm = e.ID.astype(np.int64).isin(members).values
        ref = ec[onm] if onm.sum() >= 50 else ec              # the fit members set the across range
        need = (np.percentile(ref, 98) - np.percentile(ref, 2)) * 1.3 + 4.0
        full = Hp / sc                                        # across km shown at 1:1
        ex = 1 if need >= full else int(min(5, max(1, np.floor(full / need))))
        span = full / ex
        cmid = 0.5 * (np.percentile(ref, 2) + np.percentile(ref, 98))
        yr = (cmid - span / 2, cmid + span / 2)
        img = frame_relief(relief, fr, xr, yr, strength=0.40 / np.sqrt(ex))
        axp.imshow(img, extent=[xr[0], xr[1], yr[0], yr[1]], aspect="auto", interpolation="bilinear", zorder=0,
                   rasterized=True)
        axp.set_xlim(*xr)
        axp.set_ylim(*yr)
        for sel, col, sz, z in ((~onm, "#9a9a9a", 0.35, 2), (onm, "#4a4a4a", 0.35, 3)):
            axp.scatter(ea[sel], ec[sel], s=sz, c=col, lw=0, alpha=0.55, zorder=z, rasterized=True)
        # surfaces: map footprint (polygon) or line for steep ones
        polys, pcol, lines, lcol = [], [], [], []
        for _, row in sg.iterrows():
            v = corners.get(row.surface_id)
            if v is None:
                continue
            a, c = to_frame(fr, v[:4, 0], v[:4, 1])
            col = CM_DIP(DIP_NORM(row.dip))
            if row.dip >= 80:
                top2 = np.argsort(-v[:4, 2])[:2]              # shallowest two corners (z in m, up)
                lines.append(np.c_[a[top2], c[top2]])
                lcol.append(col)
            else:
                polys.append(np.c_[a, c])
                pcol.append(col)
        if polys:
            axp.add_collection(PolyCollection(polys, facecolors=[(*c[:3], 0.28) for c in pcol], edgecolors=pcol,
                                              linewidths=0.7, zorder=5))
        if lines:
            axp.add_collection(LineCollection(lines, colors="white", linewidths=1.9, zorder=5.5, alpha=0.85,
                                              capstyle="round"))
            axp.add_collection(LineCollection(lines, colors=lcol, linewidths=0.95, zorder=6, capstyle="round"))
        axp.set_xticks([])
        axp.set_yticks([])
        S.boxed(axp)
        # scale bar (along km) and north arrow in frame coordinates
        sb = nice_km(L)
        xb, yb = xr[0] + 0.05 * L, yr[0] + 0.07 * span
        hh = 0.06 / Hp * span
        for i in range(2):
            axp.add_patch(__import__("matplotlib").patches.Rectangle((xb + i * sb / 2, yb), sb / 2, hh,
                          facecolor=S.INK if i == 0 else "white", edgecolor=S.INK, lw=0.4, zorder=20))
        S.text_halo(axp, xb + sb / 2, yb + hh * 1.8, f"{sb:g} km", ha="center", va="bottom", zorder=21)
        # north arrow: direction of geographic north in the (exaggerated) display
        nA, nC = fr["u"][1], fr["n"][1]                       # north in (along, across) components
        dxd, dyd = nA * sc, nC * sc * ex                      # cm on the page
        nn = np.hypot(dxd, dyd)
        dxd, dyd = dxd / nn, dyd / nn
        cxa, cya = 0.93, 0.80
        Lc = 0.42
        p0 = (cxa - 0.5 * Lc * dxd / w, cya - 0.5 * Lc * dyd / Hp)
        p1 = (cxa + 0.5 * Lc * dxd / w, cya + 0.5 * Lc * dyd / Hp)
        axp.annotate("", xy=p1, xytext=p0, xycoords="axes fraction", zorder=22,
                     arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.2", lw=0.7, color=S.INK,
                                     shrinkA=0, shrinkB=0))
        S.text_halo(axp, p1[0] + 0.035 * dxd * Hp / w, p1[1] + 0.06 * dyd, "N", fs=S.FS["small"], weight="bold",
                    transform=axp.transAxes, ha="center", va="center", zorder=22)
        if ex > 1:
            S.text_halo(axp, 0.985, 0.04, f"across ×{ex}", transform=axp.transAxes, ha="right", va="bottom",
                        color=S.INK, zorder=21)
        # depth view
        axd.scatter(ea[~onm], e.depth.values[~onm], s=0.3, c="#9a9a9a", lw=0, alpha=0.5, rasterized=True, zorder=2)
        axd.scatter(ea[onm], e.depth.values[onm], s=0.3, c="#4a4a4a", lw=0, alpha=0.5, rasterized=True, zorder=3)
        dpol, dcol = [], []
        for _, row in sg.iterrows():
            v = corners.get(row.surface_id)
            if v is None:
                continue
            a, _ = to_frame(fr, v[:4, 0], v[:4, 1])
            z0 = row.events_depth_ok_2pct_km if np.isfinite(row.events_depth_ok_2pct_km) else row.top_km
            z1 = row.events_depth_ok_98pct_km if np.isfinite(row.events_depth_ok_98pct_km) else row.bottom_km
            dpol.append(np.array([[a.min(), z0], [a.max(), z0], [a.max(), z1], [a.min(), z1]]))
            dcol.append(CM_DIP(DIP_NORM(row.dip)))
        if dpol:
            fa = 0.16 if len(dpol) < 15 else 0.07
            axd.add_collection(PolyCollection(dpol, facecolors=[(*c[:3], fa) for c in dcol], edgecolors=dcol,
                                              linewidths=0.55, zorder=4))
        axd.set_xlim(*xr)
        axd.set_ylim(ZMAX, -2)
        axd.set_yticks([0, 20, 40])
        S.clean(axd, grid_axis="y")
        ve = (Hs / (ZMAX + 2)) / sc
        axd.set_xlabel(f"along strike (km); vertical exaggeration {ve:.1f}" if span > 1 else
                       f"along strike (km); VE {ve:.1f}", labelpad=1.0)
        if cidx == 0:
            axd.set_ylabel("depth (km)")
        # title: type marker, code, name; counts
        t = types[seg]
        axp.plot([0.08 / w], [1.0 + 0.16 / Hp], marker=S.TYPE_MK[t], ms=4.6 * S.TYPE_MS[S.TYPE_MK[t]],
                 mfc=S.TYPE_COL[t], mec="white", mew=0.4, transform=axp.transAxes, clip_on=False, ls="")
        axp.text(0.26 / w, 1.0 + 0.16 / Hp, f"{seg}  {short_name(names[seg], seg)}", transform=axp.transAxes,
                 fontsize=S.FS["base"], fontweight="bold", va="center", ha="left")
        S.text_halo(axp, 0.10 / w, 1.0 - 0.10 / Hp, f"{len(sg)} surfaces, {len(e):,} events", transform=axp.transAxes,
                    color=S.INK, va="top", ha="left", zorder=21)
        S.letter(axp, letters[k], dx=-0.62 if cidx == 0 else -0.5, dy=0.06)
        info.append(dict(seg=seg, surfaces=int(len(sg)), events=int(len(e)), ex=ex, ve=round(ve, 2),
                         dip_median=float(np.median(sg.dip)) if len(sg) else None))
    # legend row: dip colour bar, event shades, type markers
    cax = S.axes_cm(fig, left + 0.2, 0.72, 4.2, 0.13)
    import matplotlib.cm as mcm
    cb = fig.colorbar(mcm.ScalarMappable(norm=DIP_NORM, cmap=CM_DIP), cax=cax, orientation="horizontal",
                      ticks=[30, 45, 60, 75, 90])
    cb.outline.set_linewidth(0.4)
    cb.set_label("surface dip (°)", fontsize=S.FS["small"], labelpad=1.5)
    cax.tick_params(length=2, pad=1)
    lax = S.axes_cm(fig, left + 5.2, 0.3, W - left - 5.4, 0.75)
    lax.axis("off")
    lax.scatter([0.01], [0.62], s=6, c="#4a4a4a", lw=0)
    lax.text(0.025, 0.62, "fit member of a surface", va="center", fontsize=S.FS["small"])
    lax.scatter([0.01], [0.22], s=6, c="#9a9a9a", lw=0)
    lax.text(0.025, 0.22, "other depth-good event", va="center", fontsize=S.FS["small"])
    lax.plot([0.30, 0.345], [0.62, 0.62], color=CM_DIP(0.95), lw=1.0)
    lax.text(0.355, 0.62, "steep surface (dip ≥ 80°): top edge", va="center", fontsize=S.FS["small"])
    lax.add_patch(__import__("matplotlib").patches.Rectangle((0.30, 0.12), 0.045, 0.2, facecolor=(*CM_DIP(0.5)[:3], 0.28),
                                                               edgecolor=CM_DIP(0.5), lw=0.6, transform=lax.transAxes))
    lax.text(0.355, 0.22, "dipping surface: map footprint", va="center", fontsize=S.FS["small"])
    lax.text(0.66, 0.62, "depth view: seismogenic range", va="center", fontsize=S.FS["small"])
    lax.text(0.66, 0.22, "(2-98 % of depth-good members)", va="center", fontsize=S.FS["small"], color=S.INK2)
    lax.set_xlim(0, 1)
    lax.set_ylim(0, 1)
    name = "fig3_fault_geometry"
    size = S.save(fig, cfg["out"], name)
    ns = int(sum(i["surfaces"] for i in info))
    cap = (f"Fault surfaces v1 refit on catalog {cfg.get('label', '')} by plate-boundary segment "
           f"({ns} surfaces in {len(info)} segments). Top of each panel: plan view in the segment frame (x along the "
           "strike of the segment's epicentres, y across, north arrow), GMRT relief, depth-good tier A-C events "
           "(dark grey: fit members of a surface); steep surfaces (dip ≥ 80°) as their top edge, dipping surfaces as "
           "their map footprint, coloured by dip; the across direction is exaggerated where printed. Bottom: depth "
           "against along-strike distance with each surface's along-strike extent over the 2-98 % depth range of its "
           "depth-good members; VE = vertical exaggeration. " +
           (f"{n_suspect} surface of the suspect Mount St Helens deep population (members deeper than 30 km) is left "
            "out. " if n_suspect == 1 else f"{n_suspect} surfaces of the suspect Mount St Helens deep population are "
            "left out. " if n_suspect else "") +
           "; ".join(f"{i['seg']}: {i['surfaces']} surfaces, {i['events']:,} events" for i in info) + ".")
    return [dict(name=name, files=[f"{name}.pdf", f"{name}.png"], size_cm=size, caption=cap,
                 before=["fig3_fault_surfaces.png", "fig4_blanco_strands.png"],
                 inputs=["faults", "surface_segments", "segment_axes", "events_segments", "seismicity_metrics", "cache"],
                 panels=info)]


SHORT = {"GR": "Gorda Ridge", "BL": "Blanco transform", "MFZ": "Mendocino fault zone", "GP": "Gorda plate",
         "MTJ": "Triple junction", "GS": "Gorda slab", "PU": "Puget-Georgia", "MSH": "Mount St Helens",
         "CAS": "Cascade arc", "JR": "JdF ridge", "FA": "Forearc", "IS": "JdF intraslab", "EXP": "Explorer plate"}


def short_name(n, seg=None):
    return SHORT.get(seg, str(n))

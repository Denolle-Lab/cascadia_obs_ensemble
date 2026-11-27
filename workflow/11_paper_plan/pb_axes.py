#!/usr/bin/env python3
"""Better-located spreading axes and Mendocino transform trace for the segment rule (D-B113), picked from the
bathymetry: the D-B068 merged solid surface (/wd1/mdenolle_data/reloc_v5/topo/topo_15s_cons_merged.bin: GMRT multibeam
where available, GMT earth_relief 01m elsewhere; 15 arc-second grid).

PB2002 (Bird 2003) places the southern Gorda Ridge 10-20 km west of its axial valley and of its seismicity, so the
15 km ridge buffer of the segment rule misses it (the events fall in the Gorda plate group). Method:

* ridges (PB2002 PA-JF spreading steps, split into sections at the transform offsets): profiles across the PB2002 trace
  every 2 km, +-25 km at 0.25 km; two candidate picks per profile: the axial valley (deepest point of the profile
  smoothed with a 1.5 km Gaussian, median-filtered over 5 profiles = 10 km along the axis) and the axial high (highest
  point within 15 km of the trace, 4 km Gaussian, median filter over 9 profiles = 18 km);
* Mendocino transform (PB2002 transform steps south of 41 N, west of 124.9 W): north-south profiles, +-30 km; the crest
  of the Mendocino Ridge (highest point within 15 km of the trace, 2 km Gaussian) and the base of its north-facing slope
  (first point north of the crest where the slope falls below 15 % of the flank's steepest slope);
* choice per section, on evidence: PB2002 is replaced only where its 15 km buffer misses >= 10 % of the nearby tier A-C
  earthquakes of catalog v6.1 (within 30 km of the PB2002 section; Blanco and Axial boxes and events within 10 km of
  another transform left out; >= 30 events), by the bathymetric candidate that keeps the most of them within 15 km,
  if it keeps at least 10 percentage points more and its median distance to them is not larger; else PB2002. The candidates come from the bathymetry only; the
  seismicity only chooses among them.
* Blanco and Sovanco keep PB2002: Blanco events are assigned by the catalog region (the trace is not used), Sovanco
  events have unconstrained locations (checked and reported).

Writes data/paper_plan/plate_boundaries/plate_axes_v1.csv (lon, lat, kind OSR/OTF, section, source) and
plate_axes_v1_evaluation.csv; figure <PB_BIG>/figures/fig_axes.png. The catalog is the one of the PB_* variables
(v6.1 for the axes in git: run with workflow/11_paper_plan/run_v61.sh).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, map_coordinates, median_filter
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb_common as P  # noqa: E402

TOPO = Path("/wd1/mdenolle_data/reloc_v5/topo/topo_15s_cons_merged.bin")
NX, NY, X0, Y0, DX = 3841, 3601, -134.0, 37.5, 1 / 240           # .hdr: normal node registration, rows north first
OUT_AX = P.ROOT / "data/paper_plan/plate_boundaries/plate_axes_v1.csv"
OUT_EV = P.ROOT / "data/paper_plan/plate_boundaries/plate_axes_v1_evaluation.csv"
STEP_KM, W_KM, DU = 2.0, 25.0, 0.25
MIN_EV, MISS = 30, 0.10
CREST_KM = 15.0


def topo():
    z = np.fromfile(TOPO, dtype="<f8").reshape(NY, NX)[::-1]      # row 0 = south
    return lambda lon, lat: map_coordinates(z, [(np.asarray(lat) - Y0) / DX, (np.asarray(lon) - X0) / DX], order=1)


def to_xy(lon, lat):
    return ((np.asarray(lon) + 125.0) * P.KM * np.cos(np.radians(lat)), np.asarray(lat) * P.KM)


def to_lonlat(x, y):
    lat = y / P.KM
    return x / (P.KM * np.cos(np.radians(lat))) - 125.0, lat


def sections(xy, gap_km=3.0):
    """Split a densified trace (points in PB2002 order) into continuous pieces."""
    x, y = to_xy(xy[:, 0], xy[:, 1])
    d = np.hypot(np.diff(x), np.diff(y))
    cut = np.r_[0, np.flatnonzero(d > gap_km) + 1, len(xy)]
    return [xy[a:b] for a, b in zip(cut[:-1], cut[1:]) if b - a >= 3]


def resample(sec):
    x, y = to_xy(sec[:, 0], sec[:, 1])
    s = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    q = np.arange(0, s[-1] + 1e-9, STEP_KM)
    xq, yq = np.interp(q, s, x), np.interp(q, s, y)
    tx, ty = np.gradient(xq), np.gradient(yq)
    n = np.hypot(tx, ty)
    tx, ty = tx / np.where(n > 0, n, 1), ty / np.where(n > 0, n, 1)
    return xq, yq, tx, ty


def pick_ridge(sec, Z):
    xq, yq, tx, ty = resample(sec)
    nx, ny = ty, -tx                                   # normal; made to point east
    flip = nx < 0
    nx, ny = np.where(flip, -nx, nx), np.where(flip, -ny, ny)
    u = np.arange(-W_KM, W_KM + 1e-9, DU)
    px = xq[:, None] + u[None, :] * nx[:, None]
    py = yq[:, None] + u[None, :] * ny[:, None]
    lo, la = to_lonlat(px, py)
    z = Z(lo.ravel(), la.ravel()).reshape(lo.shape)
    valley = u[np.argmin(gaussian_filter1d(z, 1.5 / DU, axis=1), axis=1)]
    zc = np.where(np.abs(u)[None, :] <= CREST_KM, gaussian_filter1d(z, 4.0 / DU, axis=1), -np.inf)
    crest = u[np.argmax(zc, axis=1)]
    out = {}
    for nm, off, w in (("valley", valley, 5), ("crest", crest, 9)):
        off = median_filter(off, size=w, mode="nearest") if len(off) >= w else off
        out[nm] = np.c_[to_lonlat(xq + off * nx, yq + off * ny)]
    return out


def pick_mfz(sec, Z):
    """North-south profiles across the Mendocino transform: crest and base of the north-facing slope."""
    o = np.argsort(sec[:, 0])
    sec = sec[o]
    lon = np.arange(sec[0, 0], sec[-1, 0], STEP_KM / (P.KM * np.cos(np.radians(40.4))))
    lat0 = np.interp(lon, sec[:, 0], sec[:, 1])
    u = np.arange(-30, 30 + 1e-9, DU)
    la = lat0[:, None] + u[None, :] / P.KM
    z = Z(np.repeat(lon[:, None], len(u), 1).ravel(), la.ravel()).reshape(la.shape)
    zs = gaussian_filter1d(z, 2.0 / DU, axis=1)
    crest, base = np.zeros(len(lon)), np.zeros(len(lon))
    win = np.abs(u) <= 15
    for i in range(len(lon)):
        k = np.flatnonzero(win)[np.argmax(zs[i, win])]
        crest[i] = u[k]
        g = np.gradient(zs[i], DU)                     # m per km, + toward north
        north = np.arange(k, len(u))
        gmin = g[north].min()                          # steepest descent north of the crest (negative)
        j = north[np.argmin(g[north])]
        after = np.flatnonzero((np.arange(len(u)) > j) & (g > 0.15 * gmin))
        base[i] = u[after[0]] if len(after) else np.nan
    out = {}
    for nm, off in (("crest", crest), ("base", base)):
        off = pd.Series(off).interpolate(limit_direction="both").values
        off = median_filter(off, size=5, mode="nearest")
        out[nm] = np.c_[lon, lat0 + off / P.KM]
    return out


def med_dist(line, lon, lat):
    if len(lon) == 0:
        return np.nan, np.nan
    X, Y = to_xy(line[:, 0], line[:, 1])
    x, y = to_xy(lon, lat)
    # densify the candidate line to 0.5 km for the distance
    s = np.r_[0, np.cumsum(np.hypot(np.diff(X), np.diff(Y)))]
    q = np.arange(0, s[-1] + 1e-9, 0.5)
    d, _ = cKDTree(np.c_[np.interp(q, s, X), np.interp(q, s, Y)]).query(np.c_[x, y])
    return float(np.median(d)), float((d <= P.D_AXIS_KM).mean())


def main():
    Z = topo()
    pts, sub, tj = P.pb2002(axes="pb2002")
    c = P.load_catalog()                               # positions and flags only (no segment labels)
    e = c[c.is_eq & c.abc & ~c.review_P1 & ~c.region.isin(["Blanco", "Axial"])]
    rows, axes = [], []
    cand_all = {}
    # ridges
    for k, sec in enumerate(sections(pts["OSR"])):
        name = f"OSR{k:02d}"
        cands = {"PB2002": sec, **pick_ridge(sec, Z)}
        near = e[(P.dist_to_trace(e.lon.values, e.lat.values, sec) <= 30.0)
                 & (P.dist_to_trace(e.lon.values, e.lat.values, pts["OTF"]) > 10.0)]
        rows += evaluate(name, "OSR", sec, cands, near, axes, cand_all)
    # transforms: MFZ re-picked; the other transform steps keep PB2002
    otf = pts["OTF"]
    mfz = otf[(otf[:, 1] < 41.0) & (otf[:, 0] < -124.9)]
    rest = otf[~((otf[:, 1] < 41.0) & (otf[:, 0] < -124.9))]
    cands = {"PB2002": mfz, **pick_mfz(mfz, Z)}
    near = c[c.is_eq & c.abc & ~c.review_P1 & (c.lat < 41.0) & (c.lon < -124.9) & (c.lon > -127.6)]
    near = near[P.dist_to_trace(near.lon.values, near.lat.values, mfz) <= 30.0]
    rows += evaluate("MFZ", "OTF", mfz, cands, near, axes, cand_all)
    for k, sec in enumerate(sections(rest)):
        axes.append(pd.DataFrame(dict(lon=sec[:, 0], lat=sec[:, 1], kind="OTF", section=f"OTF{k:02d}", source="PB2002")))
    ev = pd.DataFrame(rows)
    ax = pd.concat(axes, ignore_index=True)
    OUT_AX.parent.mkdir(parents=True, exist_ok=True)
    ax.to_csv(OUT_AX, index=False, float_format="%.5f")
    ev.to_csv(OUT_EV, index=False, float_format="%.3f")
    P.log(ev.to_string())
    figure(pts, cand_all, ax, c, Z)


def evaluate(name, kind, sec, cands, near, axes, cand_all):
    res = {nm: med_dist(line, near.lon.values, near.lat.values) for nm, line in cands.items()}
    base = res["PB2002"][0]
    best = "PB2002"
    if len(near) >= MIN_EV and res["PB2002"][1] < 1 - MISS:       # the 15 km buffer misses >= 10 % of the events
        alt = max((nm for nm in res if nm != "PB2002" and np.isfinite(res[nm][1])), key=lambda nm: res[nm][1])
        if res[alt][1] - res["PB2002"][1] >= MISS and res[alt][0] <= base:
            best = alt
    line = cands[best]
    axes.append(pd.DataFrame(dict(lon=line[:, 0], lat=line[:, 1], kind=kind, section=name,
                                  source="PB2002" if best == "PB2002" else f"bathymetry ({best})")))
    cand_all[name] = cands
    row = dict(section=name, kind=kind, lat_min=float(sec[:, 1].min()), lat_max=float(sec[:, 1].max()),
               lon_min=float(sec[:, 0].min()), lon_max=float(sec[:, 0].max()), n_events=len(near), chosen=best)
    for nm, (md, fr) in res.items():
        row[f"median_km_{nm}"] = md
        row[f"within15_{nm}"] = fr
    return [row]


def figure(pts, cand_all, ax_df, c, Z):
    e = c[c.is_eq & c.abc & ~c.review_P1]
    boxes = {"S Gorda": (-127.9, -126.9, 40.35, 41.75), "N Gorda": (-127.6, -126.5, 41.6, 43.05),
             "JdF S": (-130.6, -129.4, 44.4, 46.2), "JdF N - Endeavour": (-129.9, -128.6, 46.1, 48.4),
             "Explorer": (-130.6, -128.4, 48.3, 50.3), "Mendocino": (-127.8, -124.5, 40.15, 40.8)}
    from matplotlib.colors import LightSource
    ls = LightSource(azdeg=315, altdeg=35)
    lon = X0 + DX * np.arange(NX)
    lat = Y0 + DX * np.arange(NY)
    zz = np.fromfile(TOPO, dtype="<f8").reshape(NY, NX)[::-1]
    fig, axs = plt.subplots(2, 3, figsize=(15, 11))
    for ax, (nm, (a, b, c0, d)) in zip(axs.ravel(), boxes.items()):
        i = (lon >= a) & (lon <= b)
        j = (lat >= c0) & (lat <= d)
        z = zz[np.ix_(j, i)]
        lo, hi = np.percentile(z, [2, 98])
        rgb = ls.shade(z, cmap=plt.cm.gray, vert_exag=0.05, blend_mode="overlay", vmin=lo, vmax=hi,
                       dx=DX * 111e3, dy=DX * 111e3)
        ax.imshow(rgb, extent=(lon[i][0], lon[i][-1], lat[j][0], lat[j][-1]), origin="lower",
                  aspect=1 / np.cos(np.radians((c0 + d) / 2)))
        g = e[(e.lon >= a) & (e.lon <= b) & (e.lat >= c0) & (e.lat <= d)]
        ax.scatter(g.lon, g.lat, s=0.6, c="#d7301f", lw=0, alpha=0.6)
        for k in ("OSR", "OTF"):
            xy = pts[k]
            m = (xy[:, 0] >= a) & (xy[:, 0] <= b) & (xy[:, 1] >= c0) & (xy[:, 1] <= d)
            ax.plot(xy[m, 0], xy[m, 1], ".", ms=1.2, color="cyan", label="PB2002" if k == "OSR" else None)
        for sec, h in ax_df.groupby("section"):
            if h.source.iloc[0] != "PB2002":
                ax.plot(h.lon, h.lat, "-", color="yellow", lw=1.4, label="bathymetric axis (chosen)")
        hd, lb = ax.get_legend_handles_labels()
        uniq = dict(zip(lb, hd))
        ax.legend(uniq.values(), uniq.keys(), fontsize=6.5, loc="lower right")
        ax.set_xlim(a, b)
        ax.set_ylim(c0, d)
        ax.set_title(nm, fontsize=9)
    fig.suptitle("Plate-boundary axes (D-B113): PB2002 (cyan) and the bathymetric axes chosen per section (yellow); "
                 "red: tier A-C earthquakes of v6.1; shaded D-B068 merged surface (GMRT where available)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    (P.BIG / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(P.BIG / "figures" / "fig_axes.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

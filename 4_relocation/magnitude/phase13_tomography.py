#!/usr/bin/env python3
"""Statistical relation between the ensemble seismicity and 3-D shear-velocity models of
Cascadia: Delph et al. (2018) (forearc; EMC Cascadia_ANT+RF_Delph2018) and the newer
CRESCENT Community Velocity Model Gen0 (He et al., 2026; whole margin).

For every event inside the model volume we sample the shear velocity and express it as a
perturbation dlnVs from the depth-mean (removing the 1-D increase of Vs with depth), so
positive = fast (e.g. the subducting slab / cold lithosphere) and negative = slow (e.g.
fluid/melt-rich forearc mantle wedge). We then ask, by event class (phase10), whether
seismicity preferentially occupies fast or slow structure relative to the whole-model
baseline (Kolmogorov-Smirnov test), and show a Vs cross-section with the events.

    python phase13_tomography.py --model ../../data/tomography/Delph2018.nc \
        ../../data/tomography/CRESCENT_Gen0.nc

With several models, each gets a figure row, and every later model is also re-evaluated
on the first model's footprint (same depth-mean reference) so their numbers compare.

Fetch the model first:  python utils/fetch_tomography.py
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import ks_2samp

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "utils"))
from paper_style import CLASS_COLORS, FULL, INK, MUTED, panel, save, use  # noqa: E402

use()

warnings.filterwarnings("ignore", "Mean of empty slice")

CLS = "../../data/magnitude/cascadia_catalog_classified.csv"
MODEL = "../../data/tomography/Delph2018.nc"
CRESCENT = "../../data/tomography/CRESCENT_Gen0.nc"
OUT = "../../data/magnitude/tomography_relation.png"


def coord(d, *names):
    for n in names:
        for c in list(d.coords) + list(d.dims):
            if n in c.lower():
                return c
    raise KeyError(names)


CLASSES = ["volcanic", "crustal-fault", "megathrust?", "intraslab"]
LABELS = {"Delph2018.nc": "Delph et al. (2018)",
          "CRESCENT_Gen0.nc": "CRESCENT CVM (He et al., 2026)"}


def load_dln(path, vs_var=None, box=None):
    """Vs perturbation (%) from the depth-mean of the model, optionally cropped to
    box=(lon0, lon1, lat0, lat1) BEFORE taking the depth-mean, so two models can be
    compared on the same footprint and reference."""
    d = xr.open_dataset(os.path.expanduser(path))
    vname = vs_var or next((v for v in d.data_vars if "vs" in v.lower()), None)
    if vname is None:
        raise SystemExit(f"no Vs variable found in {path}; data_vars="
                         f"{list(d.data_vars)}. Pass --vs-var explicitly.")
    cdep, clat, clon = coord(d, "depth"), coord(d, "lat"), coord(d, "lon")
    V = d[vname].transpose(cdep, clat, clon).values
    dep, lat, lon = d[cdep].values, d[clat].values, d[clon].values
    if lon.max() > 180:
        lon = np.where(lon > 180, lon - 360, lon)
    if box is not None:
        ki = (lon >= box[0]) & (lon <= box[1]); kj = (lat >= box[2]) & (lat <= box[3])
        V, lon, lat = V[:, kj][:, :, ki], lon[ki], lat[kj]
    depth_mean = np.nanmean(V, axis=(1, 2))
    dln = 100.0 * (V - depth_mean[:, None, None]) / depth_mean[:, None, None]
    return dep, lat, lon, dln


def sample(cls, dep, lat, lon, dln):
    interp = RegularGridInterpolator((dep, lat, lon), dln, bounds_error=False,
                                     fill_value=np.nan)
    c = cls[(cls.lon.between(lon.min(), lon.max()))
            & (cls.lat.between(lat.min(), lat.max()))
            & (cls.depth.between(dep.min(), dep.max()))].copy()
    c["dlnVs"] = interp(np.column_stack([c.depth, c.lat, c.lon]))
    return c.dropna(subset=["dlnVs"])


def report(name, c, base):
    ks = ks_2samp(c.dlnVs, base)
    print(f"\n== {name}: {len(c):,} events; median {c.dlnVs.median():+.2f}% vs baseline "
          f"{np.median(base):+.2f}% (KS D={ks.statistic:.3f})")
    for k in CLASSES:
        s = c[c.event_class == k]
        if len(s):
            print(f"  {k:14s} {s.dlnVs.median():+.2f}%  (n={len(s):5d}, "
                  f"KS p={ks_2samp(s.dlnVs, base).pvalue:.1e})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", nargs="+", default=[MODEL, CRESCENT],
                    help="one or more tomography netCDFs (one figure row each), e.g. "
                         "Delph2018.nc CRESCENT_Gen0.nc")
    ap.add_argument("--vs-var", default=None, help="Vs variable name (auto-detect if unset)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    cls_all = pd.read_csv(os.path.expanduser(CLS))
    rows = []
    for path in args.model:
        dep, lat, lon, dln = load_dln(path, args.vs_var)
        print(f"model {os.path.basename(path)}: depth {dep.min():.0f}..{dep.max():.0f} km, "
              f"lon {lon.min():.1f}..{lon.max():.1f}, lat {lat.min():.1f}..{lat.max():.1f}")
        c = sample(cls_all, dep, lat, lon, dln)
        base = dln[np.isfinite(dln)]
        report(os.path.basename(path), c, base)
        rows.append((LABELS.get(os.path.basename(path), os.path.basename(path)),
                     dep, lat, lon, dln, c, base))
    # common-footprint check: every later model re-referenced on the first one's box
    if len(rows) > 1:
        _, _, lat0, lon0, _, c0, _ = rows[0]
        box = (lon0.min(), lon0.max(), lat0.min(), lat0.max())
        for path in args.model[1:]:
            dep, lat, lon, dln = load_dln(path, args.vs_var, box=box)
            # the same events the first model samples, so only the model differs
            cs = sample(cls_all[cls_all.orid.isin(c0.orid)], dep, lat, lon, dln)
            report(f"{os.path.basename(path)} on the {os.path.basename(args.model[0])} "
                   "footprint, same events", cs, dln[np.isfinite(dln)])
            m = c0[["orid", "dlnVs"]].merge(cs[["orid", "dlnVs"]], on="orid")
            print(f"  event-by-event correlation with {os.path.basename(args.model[0])}: "
                  f"r={np.corrcoef(m.dlnVs_x, m.dlnVs_y)[0, 1]:.2f} (n={len(m):,})")

    # figure: one row per model -- [47-48N cross-section | colorbar | dlnVs PDF by class
    # (rotated: dlnVs on y, sharing the colorbar's range) | legend]. Axes are placed in
    # inches so the section's printed aspect (hence its vertical exaggeration) is exact.
    n = len(rows)
    VMAX = 8.0                                          # colorbar = PDF dlnVs range
    L, S, G1, CB, G2, P, G3 = 0.42, 3.86, 0.05, 0.08, 0.40, 0.95, 0.10
    H, TOP, GAP, BOT = 1.36, 0.18, 0.34, 0.40           # section 3.86 x 1.36 in
    W = FULL
    Ht = TOP + n * H + (n - 1) * GAP + BOT
    fig = plt.figure(figsize=(W, Ht))

    def put(x, y, w, h):                                # inches from the top-left
        return fig.add_axes([x / W, 1 - (y + h) / Ht, w / W, h / Ht])

    bins = np.linspace(-12, 12, 45)
    xlo = min(r[3].min() for r in rows); xhi = max(r[3].max() for r in rows)
    xlo, xhi = max(xlo, -129.0), min(xhi, -119.0)       # the margin, common to all rows
    for i, (label, dep, lat, lon, dln, c, base) in enumerate(rows):
        y0 = TOP + i * (H + GAP)
        axc = put(L, y0, S, H)
        cax = put(L + S + G1, y0, CB, H)
        axd = put(L + S + G1 + CB + G2, y0, P, H)
        swath = (lat >= 47) & (lat <= 48)
        xsec = np.nanmean(dln[:, swath, :], axis=1)
        im = axc.pcolormesh(lon, dep, xsec, cmap="RdBu", vmin=-VMAX, vmax=VMAX,
                            shading="auto", rasterized=True)
        ev = c[c.lat.between(47, 48)]
        for k in ["crustal-fault", "volcanic", "megathrust?", "intraslab"]:
            e = ev[ev.event_class == k]
            axc.scatter(e.lon, e.depth, s=1.0, color=CLASS_COLORS[k], edgecolor=INK,
                        linewidths=0.1, alpha=0.9, rasterized=True)
        axc.set_xlim(xlo, xhi); axc.set_ylim(80, 0)
        if i == n - 1:
            axc.set_xlabel("Longitude (°)")
        else:
            axc.tick_params(labelbottom=False)
        axc.set_ylabel("Depth (km)")
        axc.set_title(f"{label}, 47–48°N", loc="right", fontsize=7, color=MUTED)
        panel(axc, "abcdef"[2 * i], x=-0.075)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("dln$V_S$ (%)", labelpad=1); cb.outline.set_linewidth(0.5)
        cb.ax.tick_params(width=0.5, length=2)
        if i == 0:
            fig.canvas.draw()
            b = axc.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            hk, vk = (xhi - xlo) * 75.0 / b.width, 80.0 / b.height
            print(f"section {b.width:.2f} x {b.height:.2f} in: {hk:.0f} km/in across "
                  f"(75 km/deg at 47.5N), {vk:.0f} km/in down -> vertical "
                  f"exaggeration {hk / vk:.1f}x")

        # rotated PDF: dlnVs on y (same limits as the colorbar), density on x
        axd.hist(base, bins=bins, density=True, histtype="step", lw=0.9, color=INK,
                 orientation="horizontal", label="model baseline")
        for k in CLASSES:
            s = c[c.event_class == k]
            if len(s) > 30:
                axd.hist(s.dlnVs, bins=bins, density=True, histtype="step", lw=0.9,
                         color=CLASS_COLORS[k], orientation="horizontal",
                         label=f"{k} {s.dlnVs.median():+.1f}% ({len(s):,})")
        axd.axhline(0, color=MUTED, lw=0.5)
        axd.set_ylim(-VMAX, VMAX)
        axd.set_yticks(cb.get_ticks()); axd.tick_params(labelleft=False)
        axd.set_xlim(0, None)
        axd.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(2))
        if i == n - 1:
            axd.set_xlabel("Prob. density")
        panel(axd, "abcdef"[2 * i + 1], x=-0.04)
        axd.legend(loc="center left", bbox_to_anchor=(1.0 + G3 / P, 0.5),
                   handlelength=1.0, handletextpad=0.4, labelspacing=0.35,
                   borderaxespad=0, title="median (n)", title_fontsize=6.5,
                   alignment="left")

    print()
    save(fig, args.out)


if __name__ == "__main__":
    main()

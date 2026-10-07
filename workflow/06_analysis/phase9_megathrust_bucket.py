#!/usr/bin/env python3
"""A deliberately *generous* bucket of potentially megathrust-related seismicity.

Because both the plate-interface geometry and our focal depths are uncertain, an event
is flagged as *potentially megathrust-related* when its depth falls within a generous
band of the Slab2 interface. The band is the Slab2 model's own vertical uncertainty at
that location (cas_slab2_unc; Hayes et al. 2018, median ~13 km for Cascadia, larger
offshore) plus a margin for our relocation depth error and for the few-km offsets that
newer offshore imaging finds relative to Slab2 (e.g. CASIE21 / Carbotte et al. 2024).
This over-counts on purpose -- it is an upper bound on how much of the catalog *could*
be on the interface, not a claim that it is.

    python phase9_megathrust_bucket.py                 # margin = 5 km
    python phase9_megathrust_bucket.py --margin 10     # even more generous

Classes (where Slab2 exists beneath the event):
    megathrust?  |z_event - z_slab| <= unc + margin
    crustal      z_event shallower than interface - band  (upper plate)
    deeper       z_event deeper than interface + band     (intraslab / mantle)
    no-slab      no interface modeled beneath (outer rise, ridge/transform)
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "utils"))
from paper_style import use, FULL, panel, CLASS_COLORS, INK, save  # noqa: E402

use()

QC = "../../data/datasets_all_regions/origin_2010_2015_reloc_cog_ver3_cc_p_4_s_4_rms_2_5.csv"
DEP = "../../data/slab2/cas_slab2_dep.xyz"
UNC = "../../data/slab2/cas_slab2_unc.xyz"
OUT = "../../data/magnitude/megathrust_bucket.png"


def load_grid(path):
    g = pd.read_csv(os.path.expanduser(path), names=["lon", "lat", "v"])
    g["lon"] = np.where(g.lon > 180, g.lon - 360, g.lon)
    return g.dropna()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--margin", type=float, default=5.0,
                    help="km added to the Slab2 uncertainty (event-depth error slack)")
    args = ap.parse_args()

    qc = pd.read_csv(os.path.expanduser(QC))
    dep, unc = load_grid(DEP), load_grid(UNC)
    pts = qc[["lon", "lat"]].to_numpy()
    z_slab = np.abs(griddata(dep[["lon", "lat"]].to_numpy(), dep.v.to_numpy(), pts,
                             method="linear"))
    z_unc = griddata(unc[["lon", "lat"]].to_numpy(), unc.v.to_numpy(), pts,
                     method="linear")
    qc = qc.assign(z_slab=z_slab, z_unc=z_unc)
    qc["band"] = qc.z_unc + args.margin
    qc["dz"] = qc.depth - qc.z_slab                       # + = below interface

    has = qc.z_slab.notna() & qc.z_unc.notna()
    cls = np.full(len(qc), "no-slab", dtype=object)
    cls[has & (qc.dz.abs() <= qc.band)] = "megathrust?"
    cls[has & (qc.dz < -qc.band)] = "crustal"
    cls[has & (qc.dz > qc.band)] = "deeper"
    qc["cls"] = cls

    good = (qc.gap < 180) & (qc.s_picks >= 6) & (qc.depth > 0)
    print(f"margin={args.margin} km; band = Slab2 unc (median "
          f"{qc.z_unc.median():.0f} km) + margin")
    print("\nGENEROUS bucket over all QC events:")
    print(qc.cls.value_counts().to_string())
    nmt = (qc.cls == "megathrust?").sum()
    print(f"\npotentially megathrust-related: {nmt:,} / {len(qc):,} "
          f"({100*nmt/len(qc):.0f}%)  [well-constrained only: "
          f"{((qc.cls=='megathrust?') & good).sum():,}]")

    # figure: (a) large map by class (left); (b) smaller forearc cross-section with the
    # uncertainty band (top right) above a shared legend. Axes placed in inches.
    W, Ht = FULL, 5.05
    fig = plt.figure(figsize=(W, Ht))

    def put(x, y, w, h):                                # inches from the top-left
        return fig.add_axes([x / W, 1 - (y + h) / Ht, w / W, h / Ht])

    lon0, lon1, lat0, lat1 = -131.0, -121.0, 39.0, 51.0
    asp = 1 / np.cos(np.deg2rad(45.0))
    mw = 2.75; mh = mw * (lat1 - lat0) * asp / (lon1 - lon0)
    axm = put(0.42, 0.12, mw, mh)
    axc = put(0.42 + mw + 0.62, 0.12, W - (0.42 + mw + 0.62) - 0.05, 2.75)
    colors = {"megathrust?": CLASS_COLORS["megathrust?"],
              "crustal": CLASS_COLORS["crustal-fault"],
              "deeper": CLASS_COLORS["intraslab"],
              "no-slab": CLASS_COLORS["oceanic"]}
    draw = ["crustal", "no-slab", "deeper", "megathrust?"]   # background class first
    for c in draw:
        d = qc[qc.cls == c]
        axm.scatter(d.lon, d.lat, s=1.0, c=colors[c], alpha=0.5, linewidths=0,
                    label=f"{c} (n={len(d):,})", rasterized=True)
    axm.set_xlabel("Longitude (°)"); axm.set_ylabel("Latitude (°)")
    axm.set_xlim(lon0, lon1); axm.set_ylim(lat0, lat1)
    axm.set_aspect(asp)
    axm.add_patch(plt.Rectangle((-127, 44), 5.5, 5, facecolor=INK, alpha=0.06, lw=0,
                                zorder=0))                      # footprint of b
    axm.text(-126.9, 48.9, "section b\n(44–49°N)", fontsize=6.5, color="#666666",
             va="top", ha="left")
    panel(axm, "a")

    # (b) forearc cross-section with slab +/- band envelope
    fa = qc[qc.lat.between(44, 49) & good & qc.z_slab.notna()].sort_values("lon")
    xs = np.arange(-127, -121.5, 0.2)
    sl = [fa.loc[fa.lon.between(x, x+0.2), "z_slab"].median() for x in xs]
    bd = [fa.loc[fa.lon.between(x, x+0.2), "band"].median() for x in xs]
    xs2, sl, bd = xs+0.1, np.array(sl), np.array(bd)
    axc.fill_between(xs2, sl-bd, sl+bd, color=colors["megathrust?"], alpha=0.12,
                     lw=0, zorder=0, label="megathrust band (Slab2 unc + margin)")
    for c in draw:
        d = fa[fa.cls == c]
        axc.scatter(d.lon, d.depth, s=1.2, c=colors[c], alpha=0.6, linewidths=0,
                    rasterized=True)
    axc.plot(xs2, sl, "-", color=INK, lw=1.0, label="Slab2 interface")
    axc.set_ylim(55, 0); axc.set_xlim(-127, -121.5)
    axc.set_xlabel("Longitude (°)"); axc.set_ylabel("Depth (km)")
    panel(axc, "b", x=-0.1)

    # one legend for both panels, under the section
    h1, l1 = axm.get_legend_handles_labels()
    h2, l2 = axc.get_legend_handles_labels()
    leg = fig.legend(h1 + h2, l1 + l2, loc="upper left", ncol=1, markerscale=4,
                     handletextpad=0.3, labelspacing=0.45,
                     bbox_to_anchor=((0.42 + mw + 0.62) / W, 1 - (0.12 + 2.75 + 0.55) / Ht))
    for h in leg.legend_handles[:len(h1)]:
        h.set_alpha(1)

    outp = os.path.expanduser(OUT)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    print()
    save(fig, outp)


if __name__ == "__main__":
    main()

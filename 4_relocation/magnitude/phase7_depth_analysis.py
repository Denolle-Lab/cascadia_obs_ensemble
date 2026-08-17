#!/usr/bin/env python3
"""Depth structure of the ensemble catalog: how well is focal depth constrained, and
what does it say about crustal vs megathrust vs deeper seismicity?

Offshore OBS geometry constrains depth only moderately (large azimuthal gaps), so this
figure separates a *well-constrained* subset (gap < 180 deg, >= 6 S picks, depth > 0)
and colors an E-W cross-section by azimuthal gap so the reader can see where depth is
trustworthy. Two panels:
  (A) depth histograms by tectonic domain (well-constrained subset);
  (B) longitude-depth cross-section for the WA/OR forearc swath, colored by gap.

    python phase7_depth_analysis.py
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

QC = "../../data/datasets_all_regions/origin_2010_2015_reloc_cog_ver3_cc_p_4_s_4_rms_2_5.csv"
OUT = "../../data/magnitude/depth_analysis.png"


def domain(lon):
    if lon < -127: return "offshore ridge/transform"
    if lon < -125: return "deformation front"
    if lon < -123: return "forearc (coast)"
    return "arc / backarc"


def main():
    qc = pd.read_csv(os.path.expanduser(QC))
    good = qc[(qc.gap < 180) & (qc.s_picks >= 6) & (qc.depth > 0)].copy()
    good["domain"] = good["lon"].map(domain)
    print(f"{len(qc):,} QC events; {len(good):,} well-constrained "
          f"(gap<180, S>=6, depth>0; {100*len(good)/len(qc):.0f}%)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))

    # (A) depth histograms by domain
    order = ["offshore ridge/transform", "deformation front",
             "forearc (coast)", "arc / backarc"]
    colors = ["#4477aa", "#66ccee", "#ee6677", "#aa3377"]
    bins = np.arange(0, 60, 2.5)
    for dom, c in zip(order, colors):
        d = good.loc[good.domain == dom, "depth"]
        ax1.hist(d, bins=bins, orientation="horizontal", histtype="step",
                 lw=2, color=c, density=True, label=f"{dom} (n={len(d)})")
    ax1.set_ylim(55, 0)
    ax1.set_ylabel("focal depth (km)")
    ax1.set_xlabel("normalized count")
    ax1.set_title("(A) Depth by tectonic domain\n(well-constrained events)")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.axhspan(0, 15, color="0.9", zorder=0)
    ax1.text(ax1.get_xlim()[1]*0.97, 7.5, "upper crust", ha="right", fontsize=8, color="0.4")

    # (B) longitude-depth cross-section, forearc swath, colored by gap
    fa = good[good.lat.between(44, 49)]
    sc = ax2.scatter(fa.lon, fa.depth, c=fa.gap, s=6, cmap="viridis_r",
                     vmin=90, vmax=270, alpha=0.6, linewidths=0)
    # median depth vs longitude
    lons = np.arange(-127, -121, 0.5)
    med = [fa.loc[fa.lon.between(l, l+0.5), "depth"].median() for l in lons]
    ax2.plot(lons+0.25, med, "k-o", lw=2, ms=4, label="median event depth")

    # Slab2 megathrust interface (deepens landward), if the grid has been fetched
    slab = os.path.expanduser("../../data/slab2/cas_slab2_dep.xyz")
    if os.path.exists(slab):
        s = pd.read_csv(slab, names=["lon", "lat", "dep"])
        s["lon"] = np.where(s.lon > 180, s.lon - 360, s.lon)
        s = s.dropna()
        sfa = s[s.lat.between(44, 49)]
        xs = np.arange(-127, -121, 0.25)
        idep = [abs(sfa.loc[sfa.lon.between(x, x+0.25), "dep"].median()) for x in xs]
        ax2.plot(xs+0.125, idep, color="darkblue", lw=2.5, ls="--",
                 label="Slab2 megathrust interface")
        ax2.text(-122.6, 47, "megathrust\ninterface", color="darkblue", fontsize=8,
                 ha="center")
        ax2.text(-123.8, 8, "crustal upper-plate\nseismicity (above interface)",
                 color="0.25", fontsize=8, ha="center")
    ax2.set_ylim(55, 0)
    ax2.set_xlim(-127, -121)
    ax2.set_xlabel("longitude (deg)  ~  distance landward from trench")
    ax2.set_ylabel("focal depth (km)")
    ax2.set_title("(B) Forearc cross-section (44-49$^\\circ$N)\ncolored by azimuthal gap")
    cb = fig.colorbar(sc, ax=ax2, pad=0.02)
    cb.set_label("azimuthal gap (deg)  -  lower = better depth")
    ax2.legend(fontsize=8, loc="lower left")
    ax2.text(-126.9, 3, "west depths biased deep\n(high gap, west of array)",
             fontsize=8, color="0.35")

    fig.tight_layout()
    outp = os.path.expanduser(OUT)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    fig.savefig(outp, dpi=200, bbox_inches="tight")
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()

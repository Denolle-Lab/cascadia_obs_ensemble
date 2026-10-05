#!/usr/bin/env python3
"""Is the seismicity correlated with the roughness of the subducting (Juan de Fuca /
Gorda) plate?

The roughness of the incoming oceanic plate -- abyssal-hill fabric, seamounts,
propagator wakes -- is carried down-dip as the plate subducts and is thought to
modulate seismicity and coupling. Here we use seafloor bathymetric roughness offshore
as the proxy for subducting-plate roughness: roughness = local standard deviation of
bathymetry (a high-pass measure). We then test whether earthquakes concentrate in rough
vs smooth seafloor by comparing the roughness sampled at event locations to a random
ocean baseline, and we compare along-strike (latitude) profiles of trench roughness and
forearc seismicity.

    python phase12_slab_roughness.py

Resolution note: uses the 02m relief grid (~3.7 km), reliable but coarse -- finer grids
(--res 15s) resolve seamounts better where the network allows.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygmt
from scipy.ndimage import uniform_filter
from scipy.stats import ks_2samp

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "utils"))
from paper_style import FULL, MUTED, OURS, panel, save, use  # noqa: E402

use()

CLS = "../../data/magnitude/cascadia_catalog_classified.csv"
OUT = "../../data/magnitude/slab_roughness.png"
REGION = [-132, -123.5, 39.5, 50.5]                  # incoming-plate offshore box


def roughness_grid(res, win=5):
    """Local std of bathymetry (km) as a roughness proxy; NaN on land."""
    g = pygmt.datasets.load_earth_relief(resolution=res, region=REGION)
    z = g.values.astype(float) / 1000.0              # m -> km, elevation (neg = ocean)
    mean = uniform_filter(z, win, mode="nearest")
    var = uniform_filter(z * z, win, mode="nearest") - mean * mean
    rough = np.sqrt(np.clip(var, 0, None))
    rough[z >= 0] = np.nan                            # ocean only
    return g.coords["lon"].values, g.coords["lat"].values, rough


def sample(lons, lats, rough, x, y):
    ix = np.clip(np.searchsorted(lons, x) - 1, 0, len(lons) - 1)
    iy = np.clip(np.searchsorted(lats, y) - 1, 0, len(lats) - 1)
    return rough[iy, ix]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--res", default="02m")
    args = ap.parse_args()

    lons, lats, rough = roughness_grid(args.res)
    cls = pd.read_csv(os.path.expanduser(CLS))
    off = cls[(cls.lon.between(*REGION[:2])) & (cls.lat.between(*REGION[2:]))
              & (cls.event_class.isin(["oceanic", "megathrust?"]))].copy()
    off["rough"] = sample(lons, lats, rough, off.lon.values, off.lat.values)
    off = off.dropna(subset=["rough"])

    # random ocean baseline (same count, drawn uniformly at random over the wet grid)
    wet = np.argwhere(~np.isnan(rough))
    gen = np.random.default_rng(0)                   # seeded for reproducibility
    pick = wet[gen.integers(0, len(wet), size=len(off))]
    base = rough[pick[:, 0], pick[:, 1]]

    ks = ks_2samp(off.rough, base)
    med_e, med_b = np.nanmedian(off.rough), np.nanmedian(base)
    print(f"offshore events sampled: {len(off):,}")
    print(f"roughness at events median {med_e:.3f} km vs ocean baseline {med_b:.3f} km")
    print(f"KS test: D={ks.statistic:.3f}, p={ks.pvalue:.1e} "
          f"({'events rougher' if med_e > med_b else 'events smoother'})")

    fig, (axm, axr) = plt.subplots(1, 2, figsize=(FULL, 3.4),
                                   gridspec_kw={"width_ratios": [1, 1.1]})
    # (a) roughness map + seismicity
    im = axm.pcolormesh(lons, lats, rough, cmap="cividis",
                        vmin=0, vmax=np.nanpercentile(rough, 98), shading="auto",
                        rasterized=True)
    axm.scatter(off.lon, off.lat, s=0.8, c=OURS, alpha=0.5, linewidths=0,
                rasterized=True)                     # offshore EQ (see caption)
    axm.set_xlim(*REGION[:2]); axm.set_ylim(*REGION[2:])
    axm.set_aspect(1 / np.cos(np.radians(np.mean(REGION[2:]))))
    axm.set_xlabel("Longitude (°)"); axm.set_ylabel("Latitude (°)")
    panel(axm, "a")
    cb = fig.colorbar(im, ax=axm, extend="max", shrink=0.8, pad=0.02, aspect=25)
    cb.set_label("Seafloor roughness (km)")
    cb.outline.set_linewidth(0.4)

    # (b) roughness at events vs random baseline
    bins = np.linspace(0, np.nanpercentile(base, 99), 40)
    axr.hist(base, bins=bins, density=True, histtype="step", lw=1, color=MUTED,
             label=f"Random ocean points (median {med_b:.2f} km)")
    axr.hist(off.rough, bins=bins, density=True, histtype="step", lw=1, color=OURS,
             label=f"Earthquakes (median {med_e:.2f} km)")
    axr.axvline(med_b, color=MUTED, ls=":", lw=0.7)
    axr.axvline(med_e, color=OURS, ls=":", lw=0.7)
    axr.set_xlabel("Seafloor roughness (km)"); axr.set_ylabel("Probability density (1/km)")
    axr.set_xlim(bins[0], bins[-1])
    panel(axr, "b")
    from matplotlib.lines import Line2D
    axr.legend(handles=[Line2D([], [], color=MUTED, lw=1),
                        Line2D([], [], color=OURS, lw=1)],
               labels=[f"Random ocean points (median {med_b:.2f} km)",
                       f"Earthquakes (median {med_e:.2f} km)"], loc="upper right")

    fig.tight_layout(w_pad=0.8)
    save(fig, OUT)

if __name__ == "__main__":
    main()

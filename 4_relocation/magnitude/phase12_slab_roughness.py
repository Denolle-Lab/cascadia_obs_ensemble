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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygmt
from scipy.ndimage import uniform_filter
from scipy.stats import ks_2samp

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

    fig, (axm, axr) = plt.subplots(1, 2, figsize=(12.5, 6.4),
                                   gridspec_kw={"width_ratios": [1, 1.1]})
    # (A) roughness map + seismicity
    im = axm.pcolormesh(lons, lats, rough, cmap="cividis",
                        vmin=0, vmax=np.nanpercentile(rough, 97), shading="auto")
    axm.scatter(off.lon, off.lat, s=3, c="red", alpha=0.35, linewidths=0,
                label="offshore EQ")
    axm.set_xlim(*REGION[:2]); axm.set_ylim(*REGION[2:])
    axm.set_xlabel("longitude"); axm.set_ylabel("latitude")
    axm.set_title("(A) Seafloor roughness (subducting-plate proxy)\n+ offshore seismicity")
    fig.colorbar(im, ax=axm, label="roughness (km, local std)")
    axm.legend(fontsize=8, loc="lower left")

    # (B) roughness at events vs random baseline
    bins = np.linspace(0, np.nanpercentile(base, 99), 40)
    axr.hist(base, bins=bins, density=True, histtype="step", lw=2, color="gray",
             label=f"random ocean (med {med_b:.2f})")
    axr.hist(off.rough, bins=bins, density=True, histtype="step", lw=2, color="red",
             label=f"at earthquakes (med {med_e:.2f})")
    axr.axvline(med_b, color="gray", ls=":"); axr.axvline(med_e, color="red", ls=":")
    axr.set_xlabel("seafloor roughness (km)"); axr.set_ylabel("density")
    axr.set_title(f"(B) Roughness at events vs baseline\nKS D={ks.statistic:.2f}, "
                  f"p={ks.pvalue:.0e}")
    axr.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.expanduser(OUT), dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Statistical relation between the ensemble seismicity and the Delph et al. (2018)
3-D shear-velocity model of the Cascadia forearc (IRIS EMC Cascadia_ANT+RF_Delph2018).

For every event inside the model volume we sample the shear velocity and express it as a
perturbation dlnVs from the depth-mean (removing the 1-D increase of Vs with depth), so
positive = fast (e.g. the subducting slab / cold lithosphere) and negative = slow (e.g.
fluid/melt-rich forearc mantle wedge). We then ask, by event class (phase10), whether
seismicity preferentially occupies fast or slow structure relative to the whole-model
baseline (Kolmogorov-Smirnov test), and show a Vs cross-section with the events.

    python phase13_tomography.py

Fetch the model first:  python utils/fetch_tomography.py
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import ks_2samp

CLS = "../../data/magnitude/cascadia_catalog_classified.csv"
MODEL = "../../data/tomography/Delph2018.nc"
OUT = "../../data/magnitude/tomography_relation.png"
CLASS_COLORS = {"volcanic": "#ee7733", "megathrust?": "#cc3311",
                "crustal-fault": "#888888", "intraslab": "#4477aa"}


def coord(d, *names):
    for n in names:
        for c in list(d.coords) + list(d.dims):
            if n in c.lower():
                return c
    raise KeyError(names)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MODEL,
                    help="tomography netCDF (default Delph2018; e.g. CRESCENT CVM-Gen0)")
    ap.add_argument("--vs-var", default=None, help="Vs variable name (auto-detect if unset)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    d = xr.open_dataset(os.path.expanduser(args.model))
    vname = args.vs_var or next((v for v in d.data_vars if "vs" in v.lower()), None)
    if vname is None:
        raise SystemExit(f"no Vs variable found in {args.model}; data_vars="
                         f"{list(d.data_vars)}. Pass --vs-var explicitly.")
    vs = d[vname]                                    # (depth, lat, lon), km/s
    cdep, clat, clon = coord(d, "depth"), coord(d, "lat"), coord(d, "lon")
    dep, lat, lon = d[cdep].values, d[clat].values, d[clon].values
    vs = vs.transpose(cdep, clat, clon)
    V = vs.values
    if lon.max() > 180:
        lon = np.where(lon > 180, lon - 360, lon)
    print(f"model {os.path.basename(args.model)}: Vs '{vname}', "
          f"depth {dep.min():.0f}..{dep.max():.0f} km, lon {lon.min():.1f}..{lon.max():.1f}")

    # perturbation from the depth-mean (lateral anomaly), percent
    depth_mean = np.nanmean(V, axis=(1, 2))
    dln = 100.0 * (V - depth_mean[:, None, None]) / depth_mean[:, None, None]
    interp = RegularGridInterpolator((dep, lat, lon), dln, bounds_error=False,
                                     fill_value=np.nan)

    cls = pd.read_csv(os.path.expanduser(CLS))
    cls = cls[(cls.lon.between(lon.min(), lon.max()))
              & (cls.lat.between(lat.min(), lat.max()))
              & (cls.depth.between(dep.min(), dep.max()))].copy()
    cls["dlnVs"] = interp(np.column_stack([cls.depth, cls.lat, cls.lon]))
    cls = cls.dropna(subset=["dlnVs"])
    base = dln[np.isfinite(dln)]                      # whole-model baseline (~0 mean)

    print(f"events in model volume: {len(cls):,}")
    print(f"all events dlnVs median {cls.dlnVs.median():+.2f}% vs model baseline "
          f"{np.median(base):+.2f}%")
    ks = ks_2samp(cls.dlnVs, base)
    print(f"KS all events vs baseline: D={ks.statistic:.3f}, p={ks.pvalue:.1e}")
    print("\nby class (median dlnVs, n):")
    for c in ["volcanic", "crustal-fault", "megathrust?", "intraslab"]:
        s = cls[cls.event_class == c]
        if len(s):
            k = ks_2samp(s.dlnVs, base)
            print(f"  {c:14s} {s.dlnVs.median():+.2f}%  (n={len(s):5d}, KS p={k.pvalue:.1e})")

    # figure: (A) dlnVs distribution by class; (B) Vs cross-section (Puget swath)
    fig, (axd, axc) = plt.subplots(1, 2, figsize=(13, 5.6),
                                   gridspec_kw={"width_ratios": [1, 1.25]})
    bins = np.linspace(-12, 12, 45)
    axd.hist(base, bins=bins, density=True, histtype="step", lw=2, color="black",
             label="model baseline")
    for c in ["volcanic", "crustal-fault", "megathrust?", "intraslab"]:
        s = cls[cls.event_class == c]
        if len(s) > 30:
            axd.hist(s.dlnVs, bins=bins, density=True, histtype="step", lw=2,
                     color=CLASS_COLORS[c], label=f"{c} ({s.dlnVs.median():+.1f}%)")
    axd.axvline(0, color="0.6", lw=0.8)
    axd.set_xlabel("shear-velocity perturbation dlnVs (%)  [<0 slow/fluid  ·  >0 fast/slab]")
    axd.set_ylabel("density"); axd.set_title("(A) Velocity structure at events, by class")
    axd.legend(fontsize=8)

    # (B) cross-section: average dlnVs over lat 47-48N vs lon & depth, events overlaid
    swath = (lat >= 47) & (lat <= 48)
    xsec = np.nanmean(dln[:, swath, :], axis=1)       # (depth, lon)
    im = axc.pcolormesh(lon, dep, xsec, cmap="RdBu", vmin=-8, vmax=8, shading="auto")
    ev = cls[cls.lat.between(47, 48)]
    axc.scatter(ev.lon, ev.depth, s=6, c=[CLASS_COLORS.get(c, "k") for c in ev.event_class],
                edgecolor="k", linewidths=0.2)
    axc.set_ylim(80, 0); axc.set_xlabel("longitude"); axc.set_ylabel("depth (km)")
    axc.set_title("(B) Vs perturbation cross-section (47-48$^\\circ$N) + events")
    fig.colorbar(im, ax=axc, label="dlnVs (%)")

    fig.tight_layout()
    fig.savefig(os.path.expanduser(args.out), dpi=200, bbox_inches="tight")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

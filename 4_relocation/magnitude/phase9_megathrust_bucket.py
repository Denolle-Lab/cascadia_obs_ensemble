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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

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

    # figure: (A) map by class, (B) cross-section with the uncertainty band
    fig, (axm, axc) = plt.subplots(1, 2, figsize=(13, 6.2),
                                   gridspec_kw={"width_ratios": [1, 1.15]})
    colors = {"megathrust?": "#cc3311", "crustal": "#bbbbbb",
              "deeper": "#4477aa", "no-slab": "#eecc66"}
    for c in ["no-slab", "deeper", "crustal", "megathrust?"]:
        d = qc[qc.cls == c]
        axm.scatter(d.lon, d.lat, s=5, c=colors[c], alpha=0.5, linewidths=0,
                    label=f"{c} (n={len(d)})")
    axm.set_xlabel("longitude"); axm.set_ylabel("latitude")
    axm.set_title(f"(A) Generous megathrust bucket\n{nmt:,} events "
                  f"({100*nmt/len(qc):.0f}% of catalog)")
    axm.legend(fontsize=8, loc="lower left", markerscale=2)
    axm.set_xlim(-131, -121); axm.set_ylim(39, 51)

    # (B) forearc cross-section with slab +/- band envelope
    fa = qc[qc.lat.between(44, 49) & good & qc.z_slab.notna()].sort_values("lon")
    axc.scatter(fa.lon, fa.depth, s=7, c=[colors[c] for c in fa.cls], alpha=0.6,
                linewidths=0)
    xs = np.arange(-127, -121.5, 0.2)
    sl = [fa.loc[fa.lon.between(x, x+0.2), "z_slab"].median() for x in xs]
    bd = [fa.loc[fa.lon.between(x, x+0.2), "band"].median() for x in xs]
    xs2, sl, bd = xs+0.1, np.array(sl), np.array(bd)
    axc.plot(xs2, sl, "b-", lw=2, label="Slab2 interface")
    axc.fill_between(xs2, sl-bd, sl+bd, color="#cc3311", alpha=0.15,
                     label="megathrust band (unc + margin)")
    axc.set_ylim(55, 0); axc.set_xlim(-127, -121.5)
    axc.set_xlabel("longitude  ~  distance landward"); axc.set_ylabel("depth (km)")
    axc.set_title("(B) Forearc cross-section (44-49$^\\circ$N,\nwell-constrained)")
    axc.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    outp = os.path.expanduser(OUT)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    fig.savefig(outp, dpi=200, bbox_inches="tight")
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()

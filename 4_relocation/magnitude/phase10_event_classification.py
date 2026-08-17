#!/usr/bin/env python3
"""Partition the ensemble catalog into physically distinct populations, each carrying a
crude location + depth uncertainty, and write a labeled catalog.

Classes (in precedence order):
    volcanic       within --vol-radius km of a Holocene volcano (GVP) -- e.g. St Helens
    megathrust?    depth within the Slab2 interface +/- (Slab2 uncertainty + margin)
    crustal-fault  above the interface (upper plate), away from volcanoes
    intraslab      below the interface (subducting plate / mantle)
    oceanic        no modeled slab beneath (offshore ridge / transform)

Each event also gets its k>0 local magnitude (ML) and a reported horizontal location
uncertainty joined from the relocated catalog by origin time. Because focal depths are
only moderately constrained, the megathrust bucket is a generous upper bound, not an
assertion (see phase9).

    python phase10_event_classification.py
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

D = "../../data/datasets_all_regions"
QC = f"{D}/origin_2010_2015_reloc_cog_ver3_cc_p_4_s_4_rms_2_5.csv"
REL = f"{D}/Cascadia_relocated_catalog_ver_3.csv"
ML = "../../data/magnitude/cascadia_catalog_ML_kpos.csv"
VOLC = f"{D}/GVP_Volcano_List_Holocene_202504292212.csv"
DEP = "../../data/slab2/cas_slab2_dep.xyz"
UNC = "../../data/slab2/cas_slab2_unc.xyz"
OUT_CSV = "../../data/magnitude/cascadia_catalog_classified.csv"
OUT_FIG = "../../data/magnitude/event_classification.png"
COLORS = {"volcanic": "#ee7733", "megathrust?": "#cc3311", "crustal-fault": "#888888",
          "intraslab": "#4477aa", "oceanic": "#eecc66"}


def grid_at(path, pts):
    g = pd.read_csv(os.path.expanduser(path), names=["lon", "lat", "v"])
    g["lon"] = np.where(g.lon > 180, g.lon - 360, g.lon)
    g = g.dropna()
    return griddata(g[["lon", "lat"]].to_numpy(), g.v.to_numpy(), pts, method="linear")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--margin", type=float, default=5.0)
    ap.add_argument("--vol-radius", type=float, default=20.0)
    args = ap.parse_args()

    qc = pd.read_csv(os.path.expanduser(QC))
    qc["t"] = pd.to_datetime(qc["time"], unit="s")

    # ML by orid==event_id
    ml = pd.read_csv(os.path.expanduser(ML))
    qc = qc.merge(ml[["event_id", "ML"]], left_on="orid", right_on="event_id", how="left")

    # horizontal location uncertainty from the relocated catalog (join by origin time)
    rel = pd.read_csv(os.path.expanduser(REL))
    rel.columns = [c.strip() for c in rel.columns]
    rel["t"] = pd.to_datetime(rel["Origin Time (UTC)"], errors="coerce",
                              utc=True).dt.tz_localize(None)
    rel = rel.dropna(subset=["t"]).sort_values("t")
    qc = pd.merge_asof(qc.sort_values("t"), rel[["t", "Horizontal Uncertainity (km)"]],
                       on="t", direction="nearest", tolerance=pd.Timedelta("2s"))
    qc = qc.rename(columns={"Horizontal Uncertainity (km)": "h_unc_km"})

    # Slab2 interface + uncertainty
    pts = qc[["lon", "lat"]].to_numpy()
    qc["z_slab"] = np.abs(grid_at(DEP, pts))
    qc["z_unc"] = grid_at(UNC, pts)
    qc["dz"] = qc.depth - qc.z_slab
    band = qc.z_unc + args.margin

    # nearest Holocene volcano
    v = pd.read_csv(os.path.expanduser(VOLC), encoding="latin-1", skiprows=1)
    vlat = [c for c in v.columns if "atitude" in c][0]
    vlon = [c for c in v.columns if "ongitude" in c][0]
    vnm = [c for c in v.columns if c.strip() == "Volcano Name"][0]
    vc = v[(v[vlat].between(38, 52)) & (v[vlon].between(-131, -118))]
    dmin = np.full(len(qc), np.inf)
    vname = np.array([""] * len(qc), dtype=object)
    for _, r in vc.iterrows():
        d = np.sqrt(((qc.lat - r[vlat]) * 111) ** 2
                    + ((qc.lon - r[vlon]) * 111 * np.cos(np.radians(r[vlat]))) ** 2)
        upd = d.values < dmin
        dmin = np.where(upd, d.values, dmin)
        vname = np.where(upd, r[vnm], vname)
    qc["vol_dist_km"] = dmin
    qc["nearest_volcano"] = vname

    # classify (precedence: volcanic first)
    has = qc.z_slab.notna() & qc.z_unc.notna()
    cls = np.full(len(qc), "oceanic", dtype=object)
    cls[has & (qc.dz.abs() <= band)] = "megathrust?"
    cls[has & (qc.dz < -band)] = "crustal-fault"
    cls[has & (qc.dz > band)] = "intraslab"
    cls[qc.vol_dist_km < args.vol_radius] = "volcanic"
    qc["event_class"] = cls

    out = qc[["orid", "t", "lat", "lon", "depth", "ML", "h_unc_km", "nass",
              "gap", "rms", "z_slab", "z_unc", "dz", "vol_dist_km",
              "nearest_volcano", "event_class"]]
    out.to_csv(os.path.expanduser(OUT_CSV), index=False)
    print(f"wrote {OUT_CSV}")
    print("class counts:\n" + qc.event_class.value_counts().to_string())
    print(f"\nmedian horizontal uncertainty: {qc.h_unc_km.median():.1f} km "
          f"({qc.h_unc_km.notna().mean()*100:.0f}% joined)")

    # figure: map by class + volcano markers
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(13, 6.4),
                                  gridspec_kw={"width_ratios": [1.25, 1]})
    order = ["oceanic", "intraslab", "crustal-fault", "megathrust?", "volcanic"]
    for c in order:
        d = qc[qc.event_class == c]
        ax.scatter(d.lon, d.lat, s=5, c=COLORS[c], alpha=0.5, linewidths=0,
                   label=f"{c} (n={len(d):,})")
    ax.scatter(vc[vlon], vc[vlat], marker="^", s=70, facecolor="none",
               edgecolor="black", linewidths=1.1, label="Holocene volcano")
    ax.set_xlim(-131, -119.5); ax.set_ylim(39, 51)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title("(A) Event classification")
    ax.legend(fontsize=8, loc="lower left", markerscale=1.6)

    # (B) events near the top volcanoes
    top = (qc[qc.event_class == "volcanic"].groupby("nearest_volcano").size()
           .sort_values(ascending=False).head(8))
    axb.barh(top.index[::-1], top.values[::-1], color=COLORS["volcanic"])
    axb.set_xlabel("events within %g km" % args.vol_radius)
    axb.set_title("(B) Volcanic-seismicity by edifice")
    fig.tight_layout()
    fig.savefig(os.path.expanduser(OUT_FIG), dpi=200, bbox_inches="tight")
    print(f"wrote {OUT_FIG}")


if __name__ == "__main__":
    main()

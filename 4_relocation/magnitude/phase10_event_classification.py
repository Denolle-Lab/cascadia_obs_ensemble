#!/usr/bin/env python3
"""Partition the ensemble catalog into physically distinct populations, each carrying a
crude location + depth uncertainty, and write a labeled catalog.

Classes (in precedence order):
    volcanic       within --vol-radius km of a Holocene volcano (GVP) -- e.g. St Helens
    megathrust?    depth within the Slab2 interface +/- (Slab2 uncertainty + margin)
    crustal-fault  above the interface (upper plate), away from volcanoes
    intraslab      below the interface (subducting plate / mantle)
    oceanic        no modeled slab beneath (offshore ridge / transform)

Each event also gets its Route A (response-removed) local magnitude (ML) and a reported horizontal location
uncertainty joined from the relocated catalog by origin time. Because focal depths are
only moderately constrained, the megathrust bucket is a generous upper bound, not an
assertion (see phase9).

    python phase10_event_classification.py
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
from paper_style import CLASS_COLORS, FULL, INK, MUTED, panel, save, use  # noqa: E402

use()

D = "../../data/datasets_all_regions"
# ALL relocated events (not just the QC subset); qc_pass flags membership in the
# published quality-controlled catalog (>=4 P & >=4 S picks, RMS < 2.5 s, plus the
# shallow-event manual review) so users can filter to their taste.
QC = f"{D}/origin_2010_2015_reloc_cog_ver3_cc.csv"
QC_PASS = f"{D}/origin_2010_2015_reloc_cog_ver3_cc_p_4_s_4_rms_2_5.csv"
REL = f"{D}/Cascadia_relocated_catalog_ver_3.csv"
ML = "../../data/magnitude/cascadia_catalog_ML_routeA.csv"
VOLC = f"{D}/GVP_Volcano_List_Holocene_202504292212.csv"
DEP = "../../data/slab2/cas_slab2_dep.xyz"
UNC = "../../data/slab2/cas_slab2_unc.xyz"
OUT_CSV = "../../data/magnitude/cascadia_catalog_classified.csv"
OUT_FIG = "../../data/magnitude/event_classification.png"
COLORS = dict(CLASS_COLORS)                          # single source: utils/paper_style.py


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
    # qc_pass = membership in the published QC catalog (consistent with the paper)
    qc_orids = set(pd.read_csv(os.path.expanduser(QC_PASS))["orid"])
    qc["qc_pass"] = qc.orid.isin(qc_orids)

    # ML by orid==event_id
    ml = pd.read_csv(os.path.expanduser(ML))
    mlcols = ["event_id", "ML"] + [c for c in ("ML_unc",) if c in ml.columns]
    qc = qc.merge(ml[mlcols], left_on="orid", right_on="event_id", how="left")

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

    out_cols = ["orid", "t", "lat", "lon", "depth", "ML", "ML_unc", "h_unc_km",
                "nass", "p_picks", "s_picks", "gap", "rms", "qc_pass", "z_slab",
                "z_unc", "dz", "vol_dist_km", "nearest_volcano", "event_class"]
    out = qc[[c for c in out_cols if c in qc.columns]]
    out.to_csv(os.path.expanduser(OUT_CSV), index=False)
    print(f"qc_pass: {int(qc.qc_pass.sum()):,} of {len(qc):,} "
          f"({100*qc.qc_pass.mean():.0f}%) pass QC")
    print(f"wrote {OUT_CSV}")
    print("class counts:\n" + qc.event_class.value_counts().to_string())
    print(f"\nmedian horizontal uncertainty: {qc.h_unc_km.median():.1f} km "
          f"({qc.h_unc_km.notna().mean()*100:.0f}% joined)")

    # figure: (a) large map by class + volcano markers (left); (b) smaller bar chart of
    # the most active volcanoes (top right) above the map legend. Axes placed in inches.
    W, Ht = FULL, 5.05
    fig = plt.figure(figsize=(W, Ht))

    def put(x, y, w, h):                                # inches from the top-left
        return fig.add_axes([x / W, 1 - (y + h) / Ht, w / W, h / Ht])

    lon0, lon1, lat0, lat1 = -131.5, -119.5, 39.0, 51.0
    asp = 1 / np.cos(np.radians(45.0))
    mw = 3.25; mh = mw * (lat1 - lat0) * asp / (lon1 - lon0)
    ax = put(0.42, 0.12, mw, mh)
    xb = 0.42 + mw + 1.55                               # room for the volcano names
    axb = put(xb, 0.12, W - xb - 0.12, 2.6)
    order = ["crustal-fault", "oceanic", "intraslab", "megathrust?", "volcanic"]
    for c in order:
        d = qc[qc.event_class == c]
        ax.scatter(d.lon, d.lat, s=0.8, c=COLORS[c], alpha=0.5, linewidths=0,
                   rasterized=True, label=c)   # counts go in the caption
    ax.scatter(vc[vlon], vc[vlat], marker="^", s=9, facecolor="none",
               edgecolor=INK, linewidths=0.6, label="Holocene volcano")
    ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1)
    ax.set_aspect(asp)
    ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
    panel(ax, "a")
    h, l = ax.get_legend_handles_labels()
    leg = fig.legend(h, l, loc="upper left", handletextpad=0.3, labelspacing=0.45,
                     bbox_to_anchor=((0.42 + mw + 0.45) / W, 1 - (0.12 + 2.6 + 0.6) / Ht))
    for hh in leg.legend_handles[:-1]:
        hh.set_sizes([10]); hh.set_alpha(1)
    leg.legend_handles[-1].set_sizes([12])

    # (b) events near the top volcanoes
    top = (qc[qc.event_class == "volcanic"].groupby("nearest_volcano").size()
           .sort_values(ascending=False).head(8))
    axb.barh(top.index[::-1], top.values[::-1], color=COLORS["volcanic"], height=0.7)
    for yy, vv in enumerate(top.values[::-1]):
        axb.text(vv, yy, f" {vv:,}", va="center", ha="left", fontsize=6.5, color=MUTED)
    axb.set_xlim(0, top.values.max() * 1.22)
    axb.set_xlabel("Events within %g km of edifice" % args.vol_radius)
    axb.tick_params(axis="y", length=0)
    axb.xaxis.set_major_locator(plt.MaxNLocator(3))
    panel(axb, "b", x=-0.84)                          # left of the volcano names
    save(fig, OUT_FIG)


if __name__ == "__main__":
    main()

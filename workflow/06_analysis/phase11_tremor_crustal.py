"""Is the crustal seismicity related to tectonic tremor (ETS) in space or time?

Uses the classified catalog (crustal-fault events; phase10), the PNSN tremor GeoJSON,
and the ANSS/ComCat catalog as an independent check. Two panels:
  (A) spatial: crustal-fault events vs tremor density -- tremor is downdip (landward)
      of the upper-plate crustal seismicity, so they occupy distinct positions;
  (B) temporal: monthly crustal-earthquake rate (this catalog and ANSS) vs tremor rate,
      with correlation coefficients.

    python phase11_tremor_crustal.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "utils"))
from paper_style import FULL, INK, MUTED, OURS, REF, panel, save, use  # noqa: E402

use()

D = "../../data/datasets_all_regions"
CLS = "../../data/magnitude/cascadia_catalog_classified.csv"
TREMOR = f"{D}/pnsn_tremor.json"
ANSS = "../../data/datasets_anss/anss_2010-15.csv"
OUT = "../../data/magnitude/tremor_crustal.png"
WIN = slice("2010-01-01", "2015-07-01")


def load_tremor():
    t = json.load(open(os.path.expanduser(TREMOR)))
    lon = [f["geometry"]["coordinates"][0] for f in t["features"]]
    lat = [f["geometry"]["coordinates"][1] for f in t["features"]]
    tm = [f["properties"]["time"] for f in t["features"]]
    tr = pd.DataFrame({"lon": lon, "lat": lat, "time": tm})
    tr["t"] = pd.to_datetime(tr["time"], errors="coerce", utc=True).dt.tz_localize(None)
    return tr.dropna(subset=["t"]).query("'2010-01-01' <= t < '2015-07-01'")


def main():
    cls = pd.read_csv(os.path.expanduser(CLS), parse_dates=["t"])
    cr = cls[cls.event_class == "crustal-fault"]
    tr = load_tremor()
    anss = pd.read_csv(os.path.expanduser(ANSS), index_col=0)
    anss["t"] = pd.to_datetime(anss["time"], format="%Y-%m-%dT%H:%M:%S.%fZ",
                               errors="coerce")
    # ANSS crustal proxy: onshore forearc, shallow, resolvable magnitude
    ac = anss[(anss.longitude > -124) & (anss.depth < 30) & (anss.mag >= 1.0)
              & anss.t.between("2010-01-01", "2015-07-01")]

    fig = plt.figure(figsize=(FULL, 3.3))
    # columns: map | colorbar | (room for colorbar labels) | stacked time series
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 0.95, 4.0],
                          wspace=0.0, hspace=0.12, left=0.07, right=0.99,
                          bottom=0.13, top=0.95)
    axm = fig.add_subplot(gs[:, 0])
    axt = fig.add_subplot(gs[0, 2])
    axr = fig.add_subplot(gs[1, 2], sharex=axt)

    # (a) spatial: tremor density (hexbin) + crustal events
    hb = axm.hexbin(tr.lon, tr.lat, gridsize=60, cmap="cividis_r", mincnt=1,
                    extent=(-125, -121, 40, 49.5), linewidths=0, rasterized=True)
    axm.scatter(cr.lon, cr.lat, s=0.6, c=INK, alpha=0.4, linewidths=0,
                rasterized=True)                     # crustal-fault EQ (see caption)
    axm.set_xlim(-126, -121); axm.set_ylim(40, 49.5)
    axm.set_aspect(1 / np.cos(np.radians(44.75)))
    axm.set_xlabel("Longitude (°)"); axm.set_ylabel("Latitude (°)")
    panel(axm, "a")
    axm.set_xticks([-126, -124, -122])
    fig.canvas.draw()                                # place colorbar against the aspect-fixed map
    pm = axm.get_position()
    cax = fig.add_axes([pm.x1 + 0.008, pm.y0 + 0.2 * pm.height, 0.009, 0.6 * pm.height])
    cb = fig.colorbar(hb, cax=cax)
    cb.set_label("Tremor detections per cell")
    cb.outline.set_linewidth(0.4)

    # (b) monthly crustal EQ rate, (c) monthly tremor rate
    cr_m = cr.set_index("t").resample("MS").size().loc[WIN]
    ac_m = ac.set_index("t").resample("MS").size().loc[WIN]
    tr_m = tr.set_index("t").resample("MS").size().loc[WIN]
    axt.plot(cr_m.index, cr_m.values, "-o", color=OURS, ms=1.8, lw=0.9,
             label="This catalog")
    axt.plot(ac_m.index, ac_m.values, "-s", color=REF, ms=1.8, lw=0.9,
             label="ANSS")
    axt.set_ylabel("Earthquakes per month")
    axt.legend(loc="upper left", ncol=2)
    axt.tick_params(labelbottom=False)
    panel(axt, "b")
    axr.plot(tr_m.index, tr_m.values, "-", color=MUTED, lw=0.9)
    axr.set_ylabel("Tremor detections\nper month")
    panel(axr, "c")

    def r(a, b):
        j = pd.concat([a, b], axis=1).dropna()
        return j.iloc[:, 0].corr(j.iloc[:, 1]), j.iloc[:, 0].corr(j.iloc[:, 1], method="spearman")
    r1 = r(cr_m, tr_m); r2 = r(ac_m, tr_m)
    axr.xaxis.set_major_locator(mdates.YearLocator())
    axr.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axr.set_xlabel("Year")

    save(fig, OUT)
    print(f"crustal-fault EQ n={len(cr):,}; ANSS crustal n={len(ac):,}; tremor n={len(tr):,}")
    print(f"temporal corr (ours) Pearson {r1[0]:.2f} Spearman {r1[1]:.2f}; "
          f"(ANSS) Pearson {r2[0]:.2f} Spearman {r2[1]:.2f}")


if __name__ == "__main__":
    main()

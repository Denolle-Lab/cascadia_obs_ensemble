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

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

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

    fig, (axm, axt) = plt.subplots(1, 2, figsize=(13, 6.2),
                                   gridspec_kw={"width_ratios": [1, 1.2]})

    # (A) spatial: tremor density (hexbin) + crustal events
    axm.hexbin(tr.lon, tr.lat, gridsize=60, cmap="Purples", mincnt=1,
               extent=(-125, -121, 40, 49.5))
    axm.scatter(cr.lon, cr.lat, s=4, c="0.15", alpha=0.35, linewidths=0,
                label="crustal-fault EQ")
    axm.set_xlim(-125, -121); axm.set_ylim(40, 49.5)
    axm.set_xlabel("longitude"); axm.set_ylabel("latitude")
    axm.set_title("(A) Crustal seismicity (dots) vs tremor density (purple)")
    axm.legend(fontsize=8, loc="lower left")

    # (B) temporal: monthly rates + correlation
    cr_m = cr.set_index("t").resample("MS").size().loc[WIN]
    ac_m = ac.set_index("t").resample("MS").size().loc[WIN]
    tr_m = tr.set_index("t").resample("MS").size().loc[WIN]
    axt.plot(cr_m.index, cr_m.values, "-o", color="firebrick", ms=3,
             label="crustal EQ (this catalog)")
    axt.plot(ac_m.index, ac_m.values, "-s", color="darkorange", ms=3,
             label="crustal EQ (ANSS)")
    axt.set_ylabel("earthquakes / month")
    axb = axt.twinx()
    axb.plot(tr_m.index, tr_m.values, "-", color="purple", lw=1.6,
             label="tremor / month")
    axb.set_ylabel("tremor detections / month", color="purple")
    axb.tick_params(axis="y", labelcolor="purple")

    def r(a, b):
        j = pd.concat([a, b], axis=1).dropna()
        return j.iloc[:, 0].corr(j.iloc[:, 1]), j.iloc[:, 0].corr(j.iloc[:, 1], method="spearman")
    r1 = r(cr_m, tr_m); r2 = r(ac_m, tr_m)
    axt.set_title("(B) Monthly rates: crustal EQ vs tremor\n"
                  f"ours: r={r1[0]:.2f} (S={r1[1]:.2f}) · ANSS: r={r2[0]:.2f} (S={r2[1]:.2f})")
    axt.legend(fontsize=8, loc="upper left")
    axt.xaxis.set_major_locator(mdates.YearLocator())
    axt.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(os.path.expanduser(OUT), dpi=200, bbox_inches="tight")
    print(f"crustal-fault EQ n={len(cr):,}; ANSS crustal n={len(ac):,}; tremor n={len(tr):,}")
    print(f"temporal corr (ours) Pearson {r1[0]:.2f} Spearman {r1[1]:.2f}; "
          f"(ANSS) Pearson {r2[0]:.2f} Spearman {r2[1]:.2f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

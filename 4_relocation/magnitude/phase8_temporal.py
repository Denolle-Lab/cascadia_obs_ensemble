#!/usr/bin/env python3
"""Temporal context for the ensemble catalog (2010-2015):

  (A) monthly earthquake rate vs monthly tectonic-tremor rate (from the PNSN tremor
      GeoJSON), to place the regular seismicity against the ETS / slow-slip cycle;
  (B) offshore latitude-vs-time, exposing the central Juan de Fuca ridge / Axial
      Seamount coverage gap (no events at ~45-46.5 N even through the April 2015
      Axial eruption -- that domain is monitored by the OOI cabled array, while the
      Cascadia-Initiative OBS were being recovered).

    python phase8_temporal.py
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

QC = "../../data/datasets_all_regions/origin_2010_2015_reloc_cog_ver3_cc_p_4_s_4_rms_2_5.csv"
TREMOR = "../../data/datasets_all_regions/pnsn_tremor.json"
OUT = "../../data/magnitude/temporal_context.png"
AXIAL_ERUPTION = pd.Timestamp("2015-04-24")


def load_tremor(path):
    t = json.load(open(os.path.expanduser(path)))
    rows = [(f["properties"]["time"]) for f in t["features"]]
    tr = pd.DataFrame({"time": rows})
    tr["t"] = pd.to_datetime(tr["time"], errors="coerce", utc=True).dt.tz_localize(None)
    return tr.dropna(subset=["t"])


def main():
    qc = pd.read_csv(os.path.expanduser(QC))
    qc["t"] = pd.to_datetime(qc["time"], unit="s")     # catalog time is epoch seconds
    tr = load_tremor(TREMOR)
    win = (slice("2010-01-01", "2015-07-01"))
    eq_m = qc.set_index("t").resample("MS").size().loc[win]
    tr_m = tr.set_index("t").resample("MS").size().loc[win]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # (A) monthly earthquake rate vs tremor rate
    ax1.bar(eq_m.index, eq_m.values, width=25, color="firebrick", alpha=0.8,
            label="earthquakes / month")
    ax1.set_ylabel("earthquakes / month", color="firebrick")
    ax1.tick_params(axis="y", labelcolor="firebrick")
    axt = ax1.twinx()
    axt.plot(tr_m.index, tr_m.values, "-o", color="navy", ms=3, lw=1.3,
             label="tremor detections / month")
    axt.set_ylabel("tremor detections / month", color="navy")
    axt.tick_params(axis="y", labelcolor="navy")
    ax1.axvline(AXIAL_ERUPTION, color="green", ls="--", lw=1.5)
    ax1.text(AXIAL_ERUPTION, ax1.get_ylim()[1]*0.92, " Axial eruption\n Apr 2015",
             color="green", fontsize=8, va="top")
    ax1.set_title("(A) Monthly earthquake rate vs tectonic tremor (ETS proxy)")

    # (B) offshore latitude vs time -> the central-ridge / Axial gap
    off = qc[qc.lon < -126.5]
    ax2.scatter(off.t, off.lat, s=4, c="firebrick", alpha=0.4, linewidths=0)
    ax2.axhspan(45.0, 46.6, color="green", alpha=0.12)
    ax2.text(off.t.min(), 45.8, "  central JdF ridge / Axial: no events",
             color="green", fontsize=9, va="center")
    ax2.axvline(AXIAL_ERUPTION, color="green", ls="--", lw=1.5)
    ax2.set_ylabel("latitude (offshore events, lon < -126.5)")
    ax2.set_xlabel("time")
    ax2.set_ylim(39, 51)
    ax2.set_title("(B) Offshore seismicity in latitude-time: the Axial coverage gap")
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    outp = os.path.expanduser(OUT)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    fig.savefig(outp, dpi=200, bbox_inches="tight")
    n_axial = ((qc.lat.between(45.0, 46.6)) & (qc.lon < -129)).sum()
    print(f"wrote {outp}")
    print(f"catalog {qc.t.min():%Y-%m} to {qc.t.max():%Y-%m}; "
          f"central-ridge/Axial events (45-46.6N, lon<-129): {n_axial}")


if __name__ == "__main__":
    main()

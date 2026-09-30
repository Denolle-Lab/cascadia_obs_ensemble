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
import sys

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "utils"))
from paper_style import use, FULL, panel, INK, MUTED, OURS, REF, save  # noqa: E402

use()

QC = "../../data/datasets_all_regions/origin_2010_2015_reloc_cog_ver3_cc_p_4_s_4_rms_2_5.csv"
TREMOR = "../../data/datasets_all_regions/pnsn_tremor.json"
ASSOC = "../../data/datasets_all_regions/assoc_2010_2015_reloc_cog_ver3.csv"
ORIGIN = "../../data/datasets_all_regions/origin_2010_2015_reloc_cog_ver3.csv"
OUT = "../../data/magnitude/temporal_context.png"
AXIAL_ERUPTION = pd.Timestamp("2015-04-24")


def active_stations_per_month():
    """Unique stations contributing associated picks each month (network-size proxy);
    the station list has no deployment dates, so we count from the associations."""
    assoc = pd.read_csv(os.path.expanduser(ASSOC), usecols=["orid", "sta"])
    orig = pd.read_csv(os.path.expanduser(ORIGIN), usecols=["orid", "time"])
    m = assoc.merge(orig, on="orid", how="inner")
    m["t"] = pd.to_datetime(m["time"], unit="s")
    s = m.set_index("t").groupby(pd.Grouper(freq="MS"))["sta"].nunique()
    return s


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
    sta_m = active_stations_per_month().loc[win]

    fig, (axe, axr, axs, axl) = plt.subplots(
        4, 1, figsize=(FULL, 5.4), sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 0.8, 2.2]})
    erupt = dict(color=MUTED, ls="--", lw=0.6, zorder=3)

    # (a) monthly earthquake rate
    axe.bar(eq_m.index, eq_m.values, width=25, color=OURS, lw=0)
    axe.set_ylabel("Earthquakes\nper month")
    # (b) monthly tectonic-tremor detections (ETS proxy)
    axr.plot(tr_m.index, tr_m.values, "-", color=REF, lw=0.8)
    axr.set_ylabel("Tremor\nper month")
    # (c) active stations (unique stations with associated picks each month)
    axs.plot(sta_m.index, sta_m.values, "-", color=INK, lw=0.8)
    axs.set_ylabel("Active\nstations")
    axs.set_ylim(0, 260)
    for ax in (axe, axr, axs, axl):
        ax.axvline(AXIAL_ERUPTION, **erupt)
    axe.text(AXIAL_ERUPTION, 1.0, "Axial eruption ", transform=axe.get_xaxis_transform(),
             ha="right", va="top", fontsize=6.5, color=MUTED)

    # (d) offshore latitude vs time -> the central-ridge / Axial gap
    off = qc[qc.lon < -126.5]
    axl.axhspan(45.0, 46.6, color="0.92", lw=0, zorder=0)
    axl.scatter(off.t, off.lat, s=1.2, c=OURS, alpha=0.4, linewidths=0, rasterized=True)
    axl.text(off.t.min(), 45.8, " central JdF ridge / Axial: no events",
             color=MUTED, fontsize=6.5, va="center")
    axl.set_ylabel("Latitude (°)")
    axl.set_xlabel("Year")
    axl.set_ylim(39, 51)
    axl.xaxis.set_major_locator(mdates.YearLocator())
    axl.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax, l in zip((axe, axr, axs, axl), "abcd"):
        panel(ax, l, x=-0.07)
    fig.align_ylabels((axe, axr, axs, axl))

    fig.tight_layout(h_pad=0.4)
    outp = os.path.expanduser(OUT)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    save(fig, outp)
    n_axial = ((qc.lat.between(45.0, 46.6)) & (qc.lon < -129)).sum()
    print(f"catalog {qc.t.min():%Y-%m} to {qc.t.max():%Y-%m}; "
          f"central-ridge/Axial events (45-46.6N, lon<-129): {n_axial}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Diagnose why the ensemble catalog under-estimates large magnitudes.

Matching our local magnitudes to ANSS/ComCat shows the scale is compressed by ~2.6x
(ML vs Mw slope ~0.38, not 1): we DETECT the large events but grossly under-size them
(e.g. the 2014 M6.8 Ferndale earthquake is ML 4.6 in our catalog). The compression is
already present in the uncalibrated relative magnitude M_rel (slope ~0.37), so it comes
from the amplitude measurement, not the absolute calibration. The amplitudes in the
catalog were measured as the raw peak counts (max|trace|) WITHOUT removing the
instrument response or simulating a Wood-Anderson seismograph
(4_relocation/calculate_amplitudes.ipynb), so mixing heterogeneous OBS/land instruments
and their frequency responses biases the amplitude-vs-magnitude scaling.

    python phase15_magnitude_diagnostic.py
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ML = "../../data/magnitude/cascadia_catalog_ML_kpos.csv"
ANSS = "../../data/datasets_anss/anss_2010-15.csv"
WAMP = "../../data/datasets_all_regions/Cascadia_updated_catalog_picks_assignment_ver_3_w_amp.csv"
OUT = "../../data/magnitude/magnitude_diagnostic.png"


def main():
    ml = pd.read_csv(os.path.expanduser(ML))
    ml["t"] = pd.to_datetime(ml["otime"], errors="coerce", utc=True).dt.tz_localize(None)
    a = pd.read_csv(os.path.expanduser(ANSS), index_col=0)
    a["t"] = pd.to_datetime(a["time"], format="%Y-%m-%dT%H:%M:%S.%fZ", errors="coerce")
    a = a[a.mag >= 2.5]

    rows = []
    for _, e in a.iterrows():
        dt = (ml.t - e.t).abs().dt.total_seconds()
        n = ml[(dt < 45) & ((ml.evla - e.latitude).abs() < 0.6)
               & ((ml.evlo - e.longitude).abs() < 0.6)]
        if len(n):
            r = n.iloc[n.ML.argmax()]
            rows.append((e.mag, r.ML, r.M_rel))
    m = pd.DataFrame(rows, columns=["Mw", "ML", "M_rel"])
    if len(m) < 5:
        raise SystemExit(f"only {len(m)} ANSS<->catalog matches within the thresholds; "
                         "cannot fit. Check the catalog/ANSS inputs and time window.")
    sML = np.polyfit(m.Mw, m.ML, 1)
    sMr = np.polyfit(m.Mw, m.M_rel, 1)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.6))

    # (A) ML vs Mw compression
    axB0 = axA
    axB0.plot([2, 7], [2, 7], "k--", lw=1, label="1:1 (correct)")
    axB0.scatter(m.Mw, m.ML, s=14, c="firebrick", alpha=0.5, linewidths=0, label="our ML")
    xx = np.array([2.5, 7])
    axB0.plot(xx, sML[0] * xx + sML[1], "firebrick", lw=2,
              label=f"fit slope={sML[0]:.2f}")
    axB0.set_xlabel("ANSS moment magnitude Mw")
    axB0.set_ylabel("our local magnitude ML")
    axB0.set_title(f"(A) Magnitude compression (slope {sML[0]:.2f}, not 1)\n"
                   "M_rel slope %.2f -> compression is upstream of calibration" % sMr[0])
    axB0.legend(fontsize=8, loc="upper left"); axB0.set_xlim(2.5, 7); axB0.set_ylim(2, 7)

    # (B) per-station amplitude vs distance for the 2014 M6.8, OBS vs land
    w = pd.read_csv(os.path.expanduser(WAMP))
    w["t"] = pd.to_datetime(w["time"], errors="coerce", utc=True).dt.tz_localize(None)
    ev = w[(w.t.between("2014-03-10 05:17", "2014-03-10 05:20"))
           & (w.latitude.between(40.5, 41.2)) & (w.longitude.between(-125.5, -124.7))].copy()
    ev["dist"] = np.sqrt(((ev.latitude - ev.slatitude) * 111) ** 2
                         + ((ev.longitude - ev.slongitude) * 111 * np.cos(np.radians(40.8))) ** 2)
    ev["net"] = ev.station.str.split(".").str[0]
    obs_nets = {"Z5", "7D", "X9", "XX", "YO"}                 # Cascadia-Initiative OBS
    ev["is_obs"] = ev.net.isin(obs_nets)
    for lab, sel, c in [("OBS (seafloor)", ev.is_obs, "tab:blue"),
                        ("land", ~ev.is_obs, "tab:orange")]:
        d = ev[sel]
        axB.scatter(d.dist, np.log10(d.Amplitude.clip(1e-3)), s=28, c=c, alpha=0.7,
                    label=f"{lab} (n={len(d)})")
    axB.set_xlabel("hypocentral distance (km)")
    axB.set_ylabel("log10(raw peak amplitude, counts)")
    axB.set_title("(B) 2014 M6.8 Ferndale: OBS read ~2 log-units low\n"
                  "(raw counts, response NOT removed)")
    axB.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.expanduser(OUT), dpi=200, bbox_inches="tight")
    print(f"ML vs Mw slope={sML[0]:.2f}, M_rel vs Mw slope={sMr[0]:.2f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

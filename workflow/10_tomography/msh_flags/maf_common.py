"""Mount St Helens analog test-signal flags for catalog v6.1 (D-B108, PI 2026-11-26 16:09): shared helpers.

A weekly artificial signal (a calibration-type pulse: ~1.5 s negative box, rise to a plateau, plateau end ~27.5 s
after the onset, exponential return) is injected on the analog UW short-period channels near Mount St Helens on
Wednesdays around noon local time. GENIE associates its common-mode onsets into false events (v5.1 review, P1).

Read-only on the catalog, the picks and pnwstore; writes only under OUT. Flag only: nothing is deleted.

Waveforms: pnwstore short reads on the record-header time base (review_v51/rv_common.py, itself a copy of
b5_cc_common.py). Raw counts (no response removal), reduced to 10 Hz by block means: the signal is a slow, very
large offset (hundreds to thousands of counts), so the reduction keeps it and averages the seismic noise down.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "review_v51"))
import rv_common as RV  # noqa: E402  (pnwstore readers only)

OUT = Path("/wd1/mdenolle_data/reloc_v61/flags")
WORK = OUT / "work"
FIG = OUT / "fig"
CAT = Path("/wd1/mdenolle_data/catalog_v6/v6.0/cascadia_catalog_v6.0.csv")
PICKS = Path("/wd1/mdenolle_data/tomo_v1/checks/pick_timing_v6/Cascadia_picks_catalog_v4.1tc.csv")
STATIONS = Path("/wd1/mdenolle_data/reloc_v5/model_vpvs176/stations_GTSRCE.in")
TZ = "America/Los_Angeles"
NAMED = ["HSR.UW", "EDM.UW", "STD.UW"]               # the channels named by the v5.1 review (P1)
MSH_BOX = dict(lat=(46.05, 46.30), lon=(-122.35, -122.10))
MSH_SUMMIT = (46.191, -122.194)
FS = 10.0                                            # Hz of the block-mean series
T_PRE, T_POST = 10.0, 50.0                           # template window around the onset (s)
MASK = (0.0, 1.8)                                    # the negative box (clipped differently per station): masked
NPROC = 6
PY_NICE = 10


def picks():
    f = WORK / "picks.parquet"
    if f.exists():
        return pd.read_parquet(f)
    p = pd.read_csv(PICKS, skipinitialspace=True,
                    usecols=["event_ID", "pick_datetime", "station", "phase", "pick_prob"])
    p = p.rename(columns={"event_ID": "ID"})
    p["t"] = pd.to_datetime(p.pick_datetime, utc=True, format="ISO8601")
    p = p.drop(columns=["pick_datetime"])
    WORK.mkdir(parents=True, exist_ok=True)
    p.to_parquet(f)
    return p


def catalog():
    f = WORK / "catalog.parquet"
    if f.exists():
        return pd.read_parquet(f)
    c = pd.read_csv(CAT, low_memory=False,
                    usecols=["ID", "time", "lat", "lon", "depth", "tier", "nphs", "sig_z", "region", "event_flag",
                             "M", "v4_time", "v4_lat", "v4_lon", "v4_depth", "review_problems",
                             "review_auto_indicators", "deep_pass"])
    c["event_flag"] = c.event_flag.fillna("").astype(str)
    c.to_parquet(f)
    return c


def stations():
    rows = []
    for ln in open(STATIONS):
        s = ln.split()
        if len(s) >= 6 and s[0] == "GTSRCE":
            rows.append((s[1], float(s[3]), float(s[4]), float(s[5])))
    return pd.DataFrame(rows, columns=["station", "lat", "lon", "elev"]).drop_duplicates("station")


def local(ts):
    return pd.Timestamp(ts).tz_convert(TZ)


def read10(sta, t_lo, t_hi):
    """Vertical channel of sta ('EDM.UW') over [t_lo, t_hi] (pandas UTC): list of (t0 POSIX, x 10-Hz block means of
    the raw counts minus the segment median), status. Picker's band rule (HH > BH > EH)."""
    code, net = sta.split(".")
    try:
        rows = RV.index_rows(net, code, t_lo, t_hi)
    except Exception as e:
        return [], f"index_error:{type(e).__name__}"
    cb = RV.choose_band(rows)
    if cb is None:
        return [], "no_data"
    try:
        segs, info = RV.read_window(rows, cb[1], t_lo.timestamp() + 1, t_lo.timestamp(), t_hi.timestamp())
    except Exception as e:
        return [], f"read_error:{type(e).__name__}"
    out = []
    for s0, sr, d in segs:
        n = int(round(sr / FS))
        if n < 1 or len(d) < n * 40:
            continue
        m = len(d) // n
        x = d[:m * n].astype(np.float64).reshape(m, n).mean(1)
        out.append((s0, x - np.median(x)))
    return out, (f"ok {cb[1]}" if out else info.get("status", "no_samples"))


def template():
    """Empirical template (10 Hz, T_PRE before to T_POST after the onset), plateau normalised to 1, and its mask."""
    f = WORK / "template.npz"
    z = np.load(f)
    return z["t"], z["y"], z["w"]


def cc_series(x, tt, ty, tw):
    """Masked normalised cross-correlation of the template with every window of x (10 Hz, valid lags) and the
    least-squares template amplitude (counts of the plateau). Window k starts at sample k; the onset is at
    k + T_PRE * FS."""
    from scipy.signal import fftconvolve
    L = len(ty)
    if len(x) < L:
        return np.array([]), np.array([])
    nw = tw.sum()
    tm = (ty - (ty * tw).sum() / nw) * tw
    norm = np.sqrt((tm ** 2).sum())
    tn = tm / norm
    num = fftconvolve(x, tn[::-1], mode="valid")
    sx = fftconvolve(x, tw[::-1], mode="valid")
    sx2 = fftconvolve(x * x, tw[::-1], mode="valid")
    var = np.maximum(sx2 - sx * sx / nw, 1e-12)
    cc = num / np.sqrt(var)
    amp = num / norm                       # least-squares amplitude of the (plateau = 1) template
    return np.clip(cc, -1, 1), amp


def noise_sigma(x):
    """Robust standard deviation of the 10-Hz series' first differences / sqrt(2), floored at 0.5 count (a dead or
    quantised channel has a zero robust spread)."""
    d = np.diff(x)
    if len(d) < 10:
        return np.nan
    return max(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2), 0.5)

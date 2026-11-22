#!/usr/bin/env python3
"""Cut raw vertical-component windows around catalog v4 P picks at the Axial
Seamount OOI stations (network OO, AX*), for testing AxialPolCap (FM_ML).

For a sample of days, each station-day is read once from pnwstore (Z channel of
the preferred sensor: EHZ for the short-period stations, HHZ for AXCC1/AXEC2/AXBA1),
demeaned, decimated to 100 Hz (anti-alias FIR, zero phase), and a window from
-3 s to +2 s around each P pick is stored unfiltered (counts). Filtering, pick
shifts and noise windows are applied later in the notebook.

Output (npz): windows (N, 500) float32 at 100 Hz, P at sample 300; plus pick
metadata columns.

Environment: default pixi env + pnwstore on PYTHONPATH, e.g.
  LD_LIBRARY_PATH=.pixi/envs/default/lib PYTHONPATH=/wd1/mdenolle_data/pyext \
  .pixi/envs/default/bin/python workflow/09_polarity/axial_test_extract.py OUT.npz
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from obspy import UTCDateTime

PICKS = "data/catalog_v4/Cascadia_picks_catalog_v4.csv"
EVENTS = "data/catalog_v4/Cascadia_relocated_catalog_v4.csv"
SR = 100.0
PRE_S, POST_S = 3.0, 2.0
N_WIN = int((PRE_S + POST_S) * SR)
# sensor preference per station (Z channel); HH = broadband, EH = short period
CHAN_PREF = ["EHZ", "HHZ", "SHZ", "BHZ"]


def load_picks():
    p = pd.read_csv(PICKS, sep=", ", engine="python")
    p.columns = [c.strip() for c in p.columns]
    p = p[(p.phase == "P") & p.station.str.startswith("AX") & p.station.str.endswith(".OO")].copy()
    p["t"] = pd.to_datetime(p.pick_datetime.str.strip().str.rstrip("Z"))
    p["sta"] = p.station.str.split(".").str[0]
    p["day"] = p.t.dt.floor("D")
    return p


def choose_days(p, n_days, seed):
    """Stratified: ~1/3 2014, ~1/2 2015 (pre/co-eruption), rest after 2015-05."""
    rng = np.random.default_rng(seed)
    days = p.groupby("day").size()
    days = days[days >= 50].index
    strata = [days[days < "2015-01-01"],
              days[(days >= "2015-01-01") & (days < "2015-05-15")],
              days[days >= "2015-05-15"]]
    want = [n_days // 3, n_days // 2, n_days - n_days // 3 - n_days // 2]
    out = []
    for s, w in zip(strata, want):
        if len(s):
            out += list(rng.choice(s, size=min(w, len(s)), replace=False))
    return sorted(pd.Timestamp(d) for d in out)


def read_day(client, sta, day):
    t0 = UTCDateTime(day.to_pydatetime())
    rows = set(client._query(["location", "channel"], network="OO", station=sta,
                             year=t0.year, doy=t0.julday))
    chans = {r[1]: (r[0] or "") for r in rows}
    for ch in CHAN_PREF:
        if ch in chans:
            st = client.get_waveforms(network="OO", station=sta, location=chans[ch] or "--",
                                      channel=ch, year=t0.year, doy=t0.julday)
            if len(st):
                return st, ch
    return None, ""


def prep(st):
    """Demean and decimate to 100 Hz with a zero-phase FIR anti-alias filter
    (obspy's decimate uses a causal IIR, which distorts onsets)."""
    from scipy.signal import decimate
    st = st.copy()
    for tr in st:
        tr.stats.sampling_rate = round(tr.stats.sampling_rate, 3)
    st.merge(method=1, fill_value=None)
    st = st.split()
    for tr in st:
        x = tr.data.astype(np.float64)
        x -= x.mean()
        f = int(round(tr.stats.sampling_rate / SR))
        if f > 1 and abs(tr.stats.sampling_rate / f - SR) < 1e-6:
            if len(x) < 200:
                tr.data = np.array([], dtype=np.float64)
                continue
            x = decimate(x, f, ftype="fir", zero_phase=True)
            tr.data = x
            tr.stats.sampling_rate = SR
        else:
            tr.data = x
            tr.resample(SR, window="hann")
    return [tr for tr in st if tr.stats.npts > 0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--n-days", type=int, default=24)
    ap.add_argument("--max-per-day", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20261122)
    a = ap.parse_args(argv)

    from pnwstore.mseed import WaveformClient
    client = WaveformClient()
    p = load_picks()
    ev = pd.read_csv(EVENTS, sep=", ", engine="python")
    ev.columns = [c.strip() for c in ev.columns]
    days = choose_days(p, a.n_days, a.seed)
    print("days:", [str(d.date()) for d in days], flush=True)
    rng = np.random.default_rng(a.seed)

    W, meta = [], []
    for day in days:
        pd_day = p[p.day == day]
        for sta, g in pd_day.groupby("sta"):
            if len(g) > a.max_per_day:
                g = g.iloc[np.sort(rng.choice(len(g), a.max_per_day, replace=False))]
            try:
                st, ch = read_day(client, sta, day)
            except Exception as e:  # noqa: BLE001
                print(f"  {day.date()} {sta}: read failed {e}", flush=True)
                continue
            if st is None:
                print(f"  {day.date()} {sta}: no data", flush=True)
                continue
            st = prep(st)
            n_ok = 0
            for r in g.itertuples():
                tp = UTCDateTime(r.t.to_pydatetime())
                for tr in st:
                    if tr.stats.starttime <= tp - PRE_S and tr.stats.endtime >= tp + POST_S:
                        i0 = int(round((tp - PRE_S - tr.stats.starttime) * SR))
                        w = tr.data[i0:i0 + N_WIN]
                        if len(w) == N_WIN and np.all(np.isfinite(w)):
                            W.append(w.astype(np.float32))
                            meta.append((r.event_ID, sta, ch, str(r.t), r.pick_prob, r._4))
                            n_ok += 1
                        break
            print(f"  {day.date()} {sta} {ch}: {n_ok}/{len(g)} windows", flush=True)

    m = pd.DataFrame(meta, columns=["event_ID", "station", "channel", "pick_time", "pick_prob", "travel_time"])
    ev = ev.rename(columns=lambda c: c.split("(")[0].strip()).rename(columns={"ID": "event_ID", "datetime": "origin_time"})
    m = m.merge(ev[["event_ID", "origin_time", "latitude", "longitude", "depth", "magnitude", "Num_p_picks"]],
                how="left", on="event_ID")
    np.savez_compressed(a.out, windows=np.stack(W), sr=SR, p_index=int(PRE_S * SR),
                        meta=m.to_json(orient="split"))
    print(f"wrote {a.out}: {len(W)} windows", flush=True)


if __name__ == "__main__":
    sys.exit(main())

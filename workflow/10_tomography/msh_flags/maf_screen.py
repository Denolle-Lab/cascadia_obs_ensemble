"""Step 3: which channels carry the pulse? (station screen; before any event scoring)

Reference pulses: onsets found at EDM.UW by the scan (cc >= 0.95, amplitude / noise >= 20), 24 of them spread
evenly over the dates they occur on. Stations: every station with picks in the catalog within 300 km of the
Mount St Helens summit. For each station and reference pulse: read onset - 20 s .. onset + 60 s, best masked cc of
the template within +-2 s of the reference onset, template amplitude and amplitude / noise.
Signal channel: |cc| >= 0.9 (either polarity: some channels carry it inverted) and amplitude / noise >= 5 at half or more of the reference pulses with data.
Output: WORK/screen_rows.csv, WORK/screen.csv (one row per station), WORK/signal_stations.json.
"""
import json
import multiprocessing as mp
import os

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import maf_common as C  # noqa: E402

CC_MIN, SNR_MIN, LAG_S = 0.9, 5.0, 2.0
N_REF = 24
R_KM = 300.0


def hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def at_onset(sta, on):
    """Best cc of the template within +-LAG_S of onset `on` (POSIX) at station sta."""
    tt, ty, tw = C.template()
    t0 = pd.Timestamp(on - C.T_PRE - LAG_S - 10, unit="s", tz="UTC")
    t1 = pd.Timestamp(on + C.T_POST + LAG_S + 10, unit="s", tz="UTC")
    segs, st = C.read10(sta, t0, t1)
    if not segs:
        return dict(status=st)
    s0, x = max(segs, key=lambda s: len(s[1]))
    cc, amp = C.cc_series(x, tt, ty, tw)
    if len(cc) == 0:
        return dict(status="short")
    k = s0 + np.arange(len(cc)) / C.FS + C.T_PRE          # onset time of each lag
    m = np.abs(k - on) <= LAG_S
    if not m.any():
        return dict(status="gap")
    i = np.where(m)[0][np.argmax(np.abs(cc[m]))]
    sig = C.noise_sigma(x[:int(8 * C.FS)]) if len(x) > 8 * C.FS else C.noise_sigma(x)
    return dict(status="ok", cc=float(cc[i]), amp=float(amp[i]), snr=float(abs(amp[i]) / sig), dt=float(k[i] - on))


def one(a):
    sta, on = a
    try:
        r = at_onset(sta, on)
    except Exception as e:
        r = dict(status=f"error {type(e).__name__}")
    return dict(station=sta, onset=on, **r)


def main():
    sc = pd.read_csv(C.WORK / "scan_rows.csv")
    ref = sc[(sc.sta == "EDM.UW") & (sc.cc >= 0.95) & (sc.snr >= 20)].copy()
    ref["on"] = pd.to_datetime(ref.onset_utc, utc=True, format="ISO8601").map(lambda t: t.timestamp())
    ref = ref.sort_values("on")
    idx = np.unique(np.linspace(0, len(ref) - 1, N_REF).round().astype(int))
    ons = ref.on.values[idx]
    p = C.picks()
    st = C.stations()
    near = st[hav(C.MSH_SUMMIT[0], C.MSH_SUMMIT[1], st.lat, st.lon) <= R_KM]
    used = set(p.station.unique())
    stas = sorted(s for s in near.station if s in used and s.split(".")[1] not in ("NC", "BK"))
    jobs = [(s, o) for s in stas for o in ons]
    print(f"screen: {len(stas)} stations x {len(ons)} pulses", flush=True)
    with mp.get_context("spawn").Pool(C.NPROC) as pool:
        rows = pool.map(one, jobs, chunksize=4)
    r = pd.DataFrame(rows)
    r.to_csv(C.WORK / "screen_rows.csv", index=False)
    r["hit"] = (r.cc.abs() >= CC_MIN) & (r.snr >= SNR_MIN)
    r["hit_neg"] = r.hit & (r.cc < 0)
    g = r.groupby("station").agg(n_data=("status", lambda s: int((s == "ok").sum())), n_hit=("hit", "sum"),
                                 n_hit_neg=("hit_neg", "sum"), cc_med=("cc", "median"),
                                 amp_med=("amp", "median")).reset_index()
    g = g.merge(near[["station", "lat", "lon"]], on="station")
    g["dist_km"] = hav(C.MSH_SUMMIT[0], C.MSH_SUMMIT[1], g.lat, g.lon).round(1)
    g["signal"] = (g.n_data > 0) & (g.n_hit >= 0.5 * g.n_data)
    g = g.sort_values(["signal", "dist_km"], ascending=[False, True])
    g.to_csv(C.WORK / "screen.csv", index=False)
    S = sorted(g[g.signal].station)
    json.dump(dict(signal_stations=S, ref_onsets_utc=[pd.Timestamp(o, unit="s", tz="UTC").isoformat() for o in ons],
                   rule=f"|cc| >= {CC_MIN} and amplitude/noise >= {SNR_MIN} within +-{LAG_S} s of the reference onset "
                        f"at >= half of the reference pulses with data"),
              open(C.WORK / "signal_stations.json", "w"), indent=1)
    print(g.to_string())
    print("signal stations:", S)


if __name__ == "__main__":
    main()

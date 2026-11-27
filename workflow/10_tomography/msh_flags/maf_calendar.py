"""Step 4: pulse calendar and timing windows (from the scan; before any event scoring).

Pulses: scan detections (cc >= 0.9, amplitude / noise >= 5) at two or more of the scanned stations with onsets
within 2 s of each other (common mode: the signal is injected on all analog channels at once).

Device time: the pulse generator keeps the pre-2007 US daylight-saving rule (DST from the first Sunday of April to
the last Sunday of October): in the weeks between the 2007+ and the old change dates the pulses come one hour late
in local time (seen in March 2012, March-April 2013, early November 2011 and 2012). Device time = UTC - 8 h, + 1 h
inside the old DST period.

Weekly pulses: Wednesdays, device time 11:50-12:05 (pulse 1) and 12:10-12:25 (pulse 2). Era: first to last
Wednesday with a weekly pulse. For a Wednesday of the era where a pulse was not detected (data gap, a long
plateau joining the two pulses, a day of glitches), its onset is predicted by linear interpolation in device time
between the detected onsets of the same pulse on the nearest Wednesdays before and after.
Irregular pulses: common-mode detections outside the weekly windows (any day scanned).

Windows: detected onset - 40 s .. + 90 s; predicted onset - 120 s .. + 150 s.
Outputs: WORK/pulses.csv, WORK/windows.csv, WORK/calendar.json, FIG/calendar.png
"""
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import maf_common as C  # noqa: E402

CC_MIN, SNR_MIN, MERGE_S = 0.9, 5.0, 2.0
W_DET = (-40.0, 90.0)
W_PRED = (-120.0, 150.0)
P1 = (11 * 60 + 50, 12 * 60 + 5)
P2 = (12 * 60 + 10, 12 * 60 + 25)


def old_dst(ts):
    """True if UTC Timestamp ts is inside the pre-2007 US DST period (first Sunday of April 02:00 PST to last Sunday
    of October 02:00 PDT)."""
    y = ts.year
    a = pd.Timestamp(f"{y}-04-01")
    a = a + pd.Timedelta(days=(6 - a.dayofweek) % 7)
    o = pd.Timestamp(f"{y}-10-31")
    o = o - pd.Timedelta(days=(o.dayofweek + 1) % 7)
    start = pd.Timestamp(a.strftime("%Y-%m-%d") + " 10:00", tz="UTC")
    end = pd.Timestamp(o.strftime("%Y-%m-%d") + " 09:00", tz="UTC")
    return start <= ts < end


def device_time(ts):
    """UTC Timestamp -> naive device wall time."""
    return (ts + pd.Timedelta(hours=-8 + (1 if old_dst(ts) else 0))).tz_localize(None)


def merge(sc):
    h = sc[(sc.cc >= CC_MIN) & (sc.snr >= SNR_MIN)].copy()
    h["on"] = pd.to_datetime(h.onset_utc, utc=True, format="ISO8601").map(lambda t: t.timestamp())
    h = h.sort_values("on")
    groups, cur = [], None
    for r in h.itertuples():
        if cur is not None and r.on - cur["last"] <= MERGE_S:
            cur["stas"].add(r.sta)
            cur["ons"].append(r.on)
            cur["ccs"].append(r.cc)
            cur["last"] = r.on
        else:
            cur = dict(stas={r.sta}, ons=[r.on], ccs=[r.cc], last=r.on)
            groups.append(cur)
    rows = []
    for g in groups:
        on = float(np.median(g["ons"]))
        ts = pd.Timestamp(on, unit="s", tz="UTC")
        dv = device_time(ts)
        lo = C.local(ts)
        dm = dv.hour * 60 + dv.minute + dv.second / 60
        rows.append(dict(onset_utc=ts.isoformat(), onset=on, local=lo.strftime("%Y-%m-%d %H:%M:%S %Z"),
                         device=dv.strftime("%Y-%m-%d %H:%M:%S"), device_min=round(dm, 3), dow=int(lo.dayofweek),
                         n_sta=len(g["stas"]), stations=",".join(sorted(g["stas"])), cc_max=max(g["ccs"])))
    p = pd.DataFrame(rows)
    p["pulse"] = np.where((p.dow == 2) & (p.device_min >= P1[0]) & (p.device_min < P1[1]), 1,
                          np.where((p.dow == 2) & (p.device_min >= P2[0]) & (p.device_min < P2[1]), 2, 0))
    return p


def main():
    sc = pd.read_csv(C.WORK / "scan_rows.csv")
    p = merge(sc)
    p.to_csv(C.WORK / "pulses.csv", index=False)
    cm = p[p.n_sta >= 2]
    wk = cm[cm.pulse > 0].copy()
    wk["wed"] = pd.to_datetime(wk.device.str[:10])
    era = (wk.wed.min(), wk.wed.max())
    weds = pd.date_range(era[0], era[1], freq="7D")
    win = []
    for k in (1, 2):
        d = wk[wk.pulse == k].sort_values("wed").drop_duplicates("wed")
        x = d.wed.map(lambda t: t.value / 1e9 / 86400).values
        y = d.device_min.values
        have = set(d.wed)
        for r in d.itertuples():
            win.append(dict(kind="weekly", pulse=k, source="detected", onset=r.onset, onset_utc=r.onset_utc,
                            device=r.device, t_lo=r.onset + W_DET[0], t_hi=r.onset + W_DET[1], n_sta=r.n_sta))
        for w in weds:
            if w in have:
                continue
            xm = w.value / 1e9 / 86400
            ym = float(np.interp(xm, x, y))
            dv = w + pd.Timedelta(minutes=ym)
            # device wall time -> UTC (old DST rule): try both offsets, keep the consistent one
            for off in (7, 8):
                ts = (dv + pd.Timedelta(hours=off)).tz_localize("UTC")
                if (off == 7) == old_dst(ts):
                    break
            on = ts.timestamp()
            win.append(dict(kind="weekly", pulse=k, source="predicted", onset=on, onset_utc=ts.isoformat(),
                            device=dv.strftime("%Y-%m-%d %H:%M:%S"), t_lo=on + W_PRED[0], t_hi=on + W_PRED[1],
                            n_sta=0))
    for r in cm[cm.pulse == 0].itertuples():
        win.append(dict(kind="irregular", pulse=0, source="detected", onset=r.onset, onset_utc=r.onset_utc,
                        device=r.device, t_lo=r.onset + W_DET[0], t_hi=r.onset + W_DET[1], n_sta=r.n_sta))
    W = pd.DataFrame(win).sort_values("onset").reset_index(drop=True)
    W.to_csv(C.WORK / "windows.csv", index=False)
    # drift of the weekly pulses (device minutes per week), per pulse
    drift = {}
    for k in (1, 2):
        d = wk[wk.pulse == k]
        x = (d.wed - era[0]).dt.days.values / 7
        drift[k] = float(np.polyfit(x, d.device_min.values * 60, 1)[0]) if len(d) > 3 else None
    loc_min = wk["local"].map(lambda s: int(s[11:13]) * 60 + int(s[14:16]) + int(s[17:19]) / 60)
    lt_late = wk[(loc_min - wk.device_min) > 30]
    info = dict(era_first=str(era[0].date()), era_last=str(era[1].date()), n_weds=len(weds),
                n_detected_weekly={k: int((wk.pulse == k).sum()) for k in (1, 2)},
                n_predicted={k: int(((W.kind == "weekly") & (W.source == "predicted") & (W.pulse == k)).sum())
                             for k in (1, 2)},
                n_irregular=int((W.kind == "irregular").sum()),
                irregular=cm[cm.pulse == 0][["local", "device", "n_sta", "stations"]].to_dict("records"),
                single_station_detections=int((p.n_sta == 1).sum()),
                drift_s_per_week=drift,
                device_range_min={k: [float(wk[wk.pulse == k].device_min.min()),
                                      float(wk[wk.pulse == k].device_min.max())] for k in (1, 2)},
                late_local_weeks=sorted(set(lt_late.device.str[:10])),
                windows=dict(detected=W_DET, predicted=W_PRED))
    json.dump(info, open(C.WORK / "calendar.json", "w"), indent=1, default=str)
    print(json.dumps(info, indent=1, default=str))

    # figure: onset clock time against date, device and local
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.6), sharey=True)
    for ax, col, lab in ((axs[0], "local", "local time (PST/PDT)"), (axs[1], "device", "device time (pre-2007 DST rule)")):
        x = pd.to_datetime(cm[col].str[:19])
        y = x.dt.hour * 60 + x.dt.minute + x.dt.second / 60
        c = np.where(cm.pulse > 0, "#2a78d6", "#eb6834")
        ax.scatter(pd.to_datetime(cm.device.str[:10]), y / 60, s=7, c=c)
        ax.set_title(f"common-mode pulse onsets, {lab}", fontsize=9, loc="left")
        ax.grid(color="#d9d8d4", lw=0.4)
        ax.set_ylim(11, 13.5)
    pr = W[W.source == "predicted"]
    dv = pd.to_datetime(pr.device)
    axs[1].scatter(dv.dt.normalize(), (dv.dt.hour * 60 + dv.dt.minute + dv.dt.second / 60) / 60, s=12,
                   facecolors="none", edgecolors="#0b0b0b", lw=0.6, label="predicted (not detected)")
    axs[1].legend(fontsize=7, loc="upper right")
    axs[0].set_ylabel("hour of day")
    fig.text(0.01, 0.01, "Blue: weekly pulses (Wednesday); orange: irregular common-mode pulses (scan 11:00-13:30 local only).",
             fontsize=7.5, color="#6b6a66")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(C.FIG / "calendar.png", dpi=110)


if __name__ == "__main__":
    main()

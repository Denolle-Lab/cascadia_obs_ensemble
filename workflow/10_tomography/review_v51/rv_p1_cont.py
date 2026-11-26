"""P1 (Mount St Helens deep events): continuous data on Wednesdays 19:05-19:30 UTC against the Tuesdays before.

Outputs in OUT/P1_msh_deep/ctx/ (figures) and WORK/p1_cont/ (scan table):
  cont_<STA>.webp    for each Wednesday with a catalogued Wednesday-window event: the 19:05-19:30 UTC trace (2-15 Hz)
                     and the Tuesday before at the same clock time, one row each, common scale per station
  pulses.webp        the first P pick of each Wednesday-window event +-4 s at the analog stations and the
                     co-located / nearby digital stations (same window), aligned on the event's first P pick
  weekly_scan.webp   every Wednesday and Tuesday 2010-2016 at one analog and one digital station: peak / median of
                     the 2-15 Hz envelope in 19:05-19:30 UTC and the clock time of the peak
Reads are short pnwstore reads (rv_common.read_window, record-header time base).
"""
import json
import os
import sys
from multiprocessing import Pool

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import rv_common as C  # noqa: E402

CTX = C.OUT / "P1_msh_deep" / "ctx"
W = C.WORK / "p1_cont"
BAND = (2.0, 15.0)
ANALOG = ["HSR.UW", "EDM.UW", "STD.UW", "SHW.UW", "FL2.UW", "ELK.UW"]
DIGITAL = ["STD.CC", "SEP.CC", "NED.CC", "SUG.CC"]
SAVE = dict(format="webp", pil_kwargs={"quality": 88})
INK, MUTED, GRID = "#0b0b0b", "#6b6a66", "#d9d8d4"


def read(sta, t0, t1, raw=False):
    """Vertical trace of a station over [t0, t1] (pandas UTC): (t POSIX array, filtered y) or None, status.
    raw=True: demeaned counts instead of the band-pass."""
    code, net = sta.split(".")
    rows = C.index_rows(net, code, t0, t1)
    cb = C.choose_band(rows)
    if cb is None:
        return None, "no data"
    segs, info = C.read_window(rows, cb[1], t0.timestamp(), t0.timestamp() - 5, t1.timestamp() + 5)
    if not segs:
        return None, info["status"]
    ts, ys = [], []
    for s0, sr, d in segs:
        if len(d) < 4 * sr:
            continue
        y = (d - np.mean(d)) if raw else C.bandpass(d, sr, BAND)
        t = s0 + np.arange(len(y)) / sr
        m = (t >= t0.timestamp()) & (t <= t1.timestamp())
        ts.append(t[m]); ys.append(y[m])
        ts.append(np.array([np.nan])); ys.append(np.array([np.nan]))
    if not ts:
        return None, "short"
    return (np.concatenate(ts), np.concatenate(ys)), f"ok {cb[1]}"


def _scan_one(a):
    """Envelope peak / median (2-15 Hz) and the longest flat stretch of the raw trace in 19:05-19:30 UTC of one day.
    Flat: 1-s windows whose standard deviation is below 2 % of the median 1-s standard deviation of the window."""
    sta, day = a
    t0 = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=19, minutes=5)
    t1 = t0 + pd.Timedelta(minutes=25)
    out = dict(sta=sta, day=day)
    try:
        r, st = read(sta, t0, t1)
        rr, _ = read(sta, t0, t1, raw=True)
    except Exception as e:
        out["status"] = f"error {type(e).__name__}: {e}"[:120]
        return out
    if r is None or rr is None:
        out["status"] = st
        return out
    t, y = r
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(y) < 1000:
        out["status"] = "short"
        return out
    sr = 1.0 / np.median(np.diff(t))
    n = max(1, int(0.5 * sr))
    env = np.convolve(np.abs(y), np.ones(n) / n, mode="same")
    k = int(np.argmax(env))
    med = float(np.median(env))
    out.update(status="ok", ratio=float(env[k] / med) if med > 0 else np.nan,
               t_peak_min=float((t[k] - (t0.timestamp() - 5 * 60)) / 60.0), cover_min=float((t[-1] - t[0]) / 60))
    # quiet stretch: 1-s windows of the 2-15 Hz trace with std below 10 % of the median 1-s std (no seismic noise)
    m = int(round(sr))
    nb = len(y) // m
    if nb > 30:
        sd = y[:nb * m].reshape(nb, m).std(axis=1)
        q = sd < 0.1 * np.median(sd)
        best, cur, start, bstart = 0, 0, 0, 0
        for i, f in enumerate(q):
            if f:
                if cur == 0:
                    start = i
                cur += 1
                if cur > best:
                    best, bstart = cur, start
            else:
                cur = 0
        out["quiet_s"] = float(best)
        out["quiet_start_min"] = float((t[bstart * m] - (t0.timestamp() - 5 * 60)) / 60.0) if best else np.nan
    # step: largest |1-s mean of the raw trace - median of the 1-s means| over the median 1-s std
    tr, yr = rr
    ok = np.isfinite(yr)
    yr = yr[ok]
    nb = len(yr) // m
    if nb > 30:
        mu = yr[:nb * m].reshape(nb, m).mean(axis=1)
        sdr = yr[:nb * m].reshape(nb, m).std(axis=1)
        out["step_ratio"] = float(np.max(np.abs(mu - np.median(mu))) / max(np.median(sdr), 1e-9))
    return out


def weekly_scan(nproc=12):
    """Resumable: rows are appended to WORK/p1_cont/weekly_scan_rows.csv as they finish (spawned workers: no state
    inherited from this process)."""
    import csv
    import multiprocessing as mp
    f = W / "weekly_scan_rows.csv"
    days = pd.date_range("2010-01-01", "2016-12-31", freq="D")
    days = [d.strftime("%Y-%m-%d") for d in days if d.dayofweek in (1, 2)]
    jobs = [(s, d) for s in ("HSR.UW", "STD.UW", "STD.CC") for d in days]
    cols = ["sta", "day", "status", "ratio", "t_peak_min", "cover_min", "quiet_s", "quiet_start_min", "step_ratio"]
    done = set()
    if f.exists():
        old = pd.read_csv(f)
        done = set(zip(old.sta, old.day))
    todo = [j for j in jobs if j not in done]
    print(f"scan: {len(todo)} of {len(jobs)} station-days to do", flush=True)
    new = not f.exists()
    with open(f, "a", newline="") as fh, mp.get_context("spawn").Pool(nproc) as p:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        if new:
            w.writeheader()
        for i, r in enumerate(p.imap_unordered(_scan_one, todo, chunksize=2)):
            w.writerow(r)
            if i % 100 == 0:
                fh.flush()
                print(f"scan {i + 1}/{len(todo)}", flush=True)
    df = pd.read_csv(f)
    df.to_csv(W / "weekly_scan.csv", index=False)
    return df


def fig_scan(df, ev):
    df = df.copy()
    df["date"] = pd.to_datetime(df.day)
    df["dow"] = df.date.dt.dayofweek
    stas = [s for s in ("HSR.UW", "STD.UW", "STD.CC") if s in set(df.sta)]
    fig, axs = plt.subplots(2, len(stas), figsize=(16, 6.4), sharex=True, squeeze=False)
    evd = pd.to_datetime(ev.time, utc=True, format="ISO8601").dt.tz_localize(None).dt.normalize()
    for j, s in enumerate(stas):
        x = df[(df.sta == s) & (df.status == "ok")]
        for dw, lab, col, mk in ((2, "Wednesday", "#2a78d6", "o"), (1, "Tuesday", "#eb6834", "s")):
            y = x[x.dow == dw]
            axs[0, j].plot(y.date, y.quiet_s, mk, ms=3.2, color=col, alpha=0.8, label=lab)
            hi = y[y.quiet_s >= 10]
            axs[1, j].plot(hi.date, hi.quiet_start_min, mk, ms=3.5, color=col, alpha=0.85, label=f"{lab}, quiet stretch >= 10 s")
        for d in evd.unique():
            axs[0, j].axvline(d, color=MUTED, lw=0.4, alpha=0.5, zorder=0)
        axs[0, j].set_title(f"{s}: longest quiet stretch (s) in 19:05-19:30 UTC", loc="left", fontsize=9)
        axs[0, j].axhline(10, color=MUTED, lw=0.6, ls="--")
        axs[1, j].set_ylim(5, 30)
        axs[1, j].set_ylabel("start of the quiet stretch (min after 19:00 UTC)")
        axs[1, j].axhspan(14, 21, color=GRID, alpha=0.5, zorder=0)
        for ax in axs[:, j]:
            ax.grid(color=GRID, lw=0.4)
        n_ok = {dw: int(((x.dow == dw)).sum()) for dw in (1, 2)}
        n_hi = {dw: int(((x.dow == dw) & (x.quiet_s >= 10)).sum()) for dw in (1, 2)}
        axs[1, j].set_title(f"days with data: Wed {n_ok[2]}, Tue {n_ok[1]}; quiet stretch >= 10 s: Wed {n_hi[2]}, Tue {n_hi[1]}",
                            loc="left", fontsize=8.5)
    axs[0, 0].set_ylabel("longest quiet stretch (s)")
    axs[0, 0].legend(loc="upper left", fontsize=7.5)
    axs[1, 0].legend(loc="upper left", fontsize=7.5)
    fig.text(0.01, 0.005, "Grey vertical lines: dates of catalogued Wednesday 19:14-19:20 UTC events in the Mount St Helens box. "
             "Shaded band: 19:14-19:21 UTC. Quiet: 1-s windows of the 2-15 Hz trace with a standard deviation below 10 % of the median "
             "1-s standard deviation of the 25-min window. Dashed line: 10 s (threshold of the lower panels).", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(CTX / "weekly_scan.webp", dpi=90, **SAVE)
    plt.close(fig)


def fig_cont(sta, ev, dates):
    rows = []
    for d in dates:
        for k, lab in ((0, "Wed"), (-1, "Tue")):
            day = pd.Timestamp(d, tz="UTC") + pd.Timedelta(days=k)
            t0 = day + pd.Timedelta(hours=19, minutes=5)
            r, st = read(sta, t0, t0 + pd.Timedelta(minutes=25))
            rows.append((day, lab, t0, r, st))
    n = len(rows)
    scales = []
    for i in range(0, n, 2):          # one scale per Wednesday-Tuesday pair: 25 x the median |amplitude| of the pair
        a = [np.nanmedian(np.abs(r[1])) for _, _, _, r, _ in rows[i:i + 2] if r is not None]
        scales += [25 * np.nanmedian(a) if a else 1.0] * 2
    fig, ax = plt.subplots(figsize=(15.5, 0.33 * n + 1.2))
    for i, (day, lab, t0, r, st) in enumerate(rows):
        y0 = n - 1 - i
        ax.text(4.9, y0, f"{day:%Y-%m-%d} {lab}", ha="right", va="center", fontsize=7,
                color=INK if lab == "Wed" else MUTED)
        if r is None:
            ax.text(5.2, y0, st, fontsize=7, color="#b3261e", va="center")
            continue
        t, y = r
        ax.plot((t - (t0.timestamp() - 5 * 60)) / 60.0, y0 + np.clip(y / scales[i], -0.6, 0.6) * 0.75,
                color=INK if lab == "Wed" else "#8a8985", lw=0.35)
        e = ev[(pd.to_datetime(ev.time, utc=True, format="ISO8601") >= t0) &
               (pd.to_datetime(ev.time, utc=True, format="ISO8601") < t0 + pd.Timedelta(minutes=25))]
        for _, r_ in e.iterrows():
            tm = (pd.Timestamp(r_.time, tz="UTC") - (t0 - pd.Timedelta(minutes=5))).total_seconds() / 60
            ax.plot([tm, tm], [y0 + 0.42, y0 + 0.5], color="#2a78d6", lw=2)
    ax.set_xlim(5, 30)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_yticks([])
    ax.set_xlabel("minutes after 19:00 UTC")
    ax.grid(axis="x", color=GRID, lw=0.4)
    ax.set_title(f"{sta} vertical, {BAND[0]:g}-{BAND[1]:g} Hz zero-phase, 19:05-19:30 UTC; each Wednesday with a catalogued "
                 f"Wednesday-window event (black) above the Tuesday before (grey). One scale per Wednesday-Tuesday pair (25 x the median "
                 f"absolute amplitude of the pair), clipped. Blue ticks: origin times of catalogued events in the MSH box.",
                 loc="left", fontsize=8, wrap=True)
    fig.tight_layout()
    fig.savefig(CTX / f"cont_{sta.replace('.', '_')}.webp", dpi=90, **SAVE)
    plt.close(fig)


def fig_pulses(wev, picks):
    stas = ANALOG[:4] + DIGITAL[:3]
    evs = wev.sort_values("time")
    n = len(evs)
    fig, axs = plt.subplots(1, len(stas), figsize=(16.5, 0.3 * n + 1.4), sharey=True)
    for i, (_, e) in enumerate(evs.iterrows()):
        pk = picks[(picks.event_ID == e.ID) & (picks.phase == "P")]
        if not len(pk):
            continue
        tref = pd.to_datetime(pk.pick_datetime, utc=True, format="ISO8601").min()
        y0 = n - 1 - i
        for j, s in enumerate(stas):
            r, st = read(s, tref - pd.Timedelta(seconds=4), tref + pd.Timedelta(seconds=6))
            ax = axs[j]
            if r is None:
                ax.text(-3.8, y0, st, fontsize=6, color="#b3261e", va="center")
                continue
            t, y = r
            a = np.nanmax(np.abs(y)) or 1
            ax.plot(t - tref.timestamp(), y0 + 0.45 * y / a, color=INK, lw=0.45)
            mine = pk[pk.station == s]
            for _, p in mine.iterrows():
                tp = (pd.Timestamp(p.pick_datetime) - tref).total_seconds()
                ax.plot([tp, tp], [y0 - 0.45, y0 + 0.45], color="#eb6834", lw=1.4)
        axs[0].text(-4.3, y0, f"{e.ID}  {e.time[:10]}  {e.depth:.0f} km", ha="right", va="center", fontsize=6.6)
    for j, s in enumerate(stas):
        axs[j].set_title(s + (" (analog, UW)" if s in ANALOG else " (digital, CC)"), fontsize=8, loc="left")
        axs[j].set_xlim(-4, 6)
        axs[j].set_yticks([])
        axs[j].grid(axis="x", color=GRID, lw=0.4)
        axs[j].set_xlabel("s after first P pick")
    fig.suptitle("Wednesday 19:14-19:20 UTC events: vertical traces 2-15 Hz zero-phase, each row scaled to its own peak "
                 "(a quiet row is noise scaled up). Orange: P pick at that station.", x=0.01, ha="left", fontsize=8.5)
    fig.tight_layout(rect=(0.06, 0, 1, 0.97))
    fig.savefig(CTX / "pulses.webp", dpi=90, **SAVE)
    plt.close(fig)


def fig_raw(wev, picks, n=6):
    """Raw counts (demeaned, no filter) 30 s before to 80 s after the first P pick of some Wednesday-window events."""
    evs = wev.sort_values("time")
    evs = evs.iloc[np.unique(np.linspace(0, len(evs) - 1, n).round().astype(int))]
    stas = ["HSR.UW", "EDM.UW", "STD.UW", "STD.CC"]
    fig, axs = plt.subplots(len(evs), len(stas), figsize=(16, 1.55 * len(evs) + 0.8), squeeze=False, sharex=True)
    for i, (_, e) in enumerate(evs.iterrows()):
        pk = picks[(picks.event_ID == e.ID) & (picks.phase == "P")]
        tref = pd.to_datetime(pk.pick_datetime, utc=True, format="ISO8601").min()
        for j, s in enumerate(stas):
            ax = axs[i, j]
            r, st = read(s, tref - pd.Timedelta(seconds=30), tref + pd.Timedelta(seconds=80), raw=True)
            if r is None:
                ax.text(0.5, 0.5, st, transform=ax.transAxes, color="#b3261e", ha="center")
            else:
                ax.plot(r[0] - tref.timestamp(), r[1], color=INK, lw=0.4)
            ax.axvline(0, color="#eb6834", lw=0.8)
            ax.grid(color=GRID, lw=0.4)
            ax.tick_params(labelsize=6.5)
            if i == 0:
                ax.set_title(s + (" (analog, UW)" if s.endswith("UW") else " (digital, CC)"), loc="left", fontsize=8.5)
            if j == 0:
                ax.set_ylabel(f"{e.ID}\n{e.time[:10]}\n{e.depth:.0f} km", fontsize=7)
    for ax in axs[-1]:
        ax.set_xlabel("s after the event's first P pick")
    fig.suptitle("Raw counts (demeaned, no filter, no response removal), own scale per panel; orange: first P pick of the event",
                 x=0.01, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(CTX / "raw_examples.webp", dpi=90, **SAVE)
    plt.close(fig)


def main():
    CTX.mkdir(parents=True, exist_ok=True)
    W.mkdir(parents=True, exist_ok=True)
    c = C.catalog()
    box = c.loc_source.eq("nll3d") & c.lat.between(*C.MSH_BOX["lat"]) & c.lon.between(*C.MSH_BOX["lon"])
    ev = c[box]
    t = ev.t
    wed = ev[(t.dt.dayofweek == 2) & (t.dt.hour == 19) & t.dt.minute.between(14, 20)]
    picks = pd.read_csv(C.PICKS, skipinitialspace=True, usecols=["event_ID", "pick_datetime", "station", "phase"])
    picks = picks[picks.event_ID.isin(wed.ID)]
    dates = sorted(set(wed.t.dt.strftime("%Y-%m-%d")))
    what = sys.argv[1:] or ["cont", "pulses", "raw", "scan"]
    if "raw" in what:
        fig_raw(wed[wed.depth > 30], picks)
        print("raw done", flush=True)
    if "pulses" in what:
        fig_pulses(wed, picks)
        print("pulses done", flush=True)
    if "cont" in what:
        for s in ["HSR.UW", "STD.UW", "STD.CC", "SEP.CC"]:
            fig_cont(s, ev, dates)
            print("cont", s, flush=True)
    if "scan" in what:
        df = weekly_scan()
        fig_scan(df, wed)
        s = {}
        for sta, g in df[df.status == "ok"].groupby("sta"):
            g = g.assign(dow=pd.to_datetime(g.day).dt.dayofweek)
            s[sta] = {("Wed" if d == 2 else "Tue"): dict(
                days=int((g.dow == d).sum()), quiet_ge10=int(((g.dow == d) & (g.quiet_s >= 10)).sum()),
                quiet_ge10_in_14_21=int(((g.dow == d) & (g.quiet_s >= 10) & g.quiet_start_min.between(14, 21)).sum()),
                first=str(g[(g.dow == d) & (g.quiet_s >= 10)].day.min()), last=str(g[(g.dow == d) & (g.quiet_s >= 10)].day.max()))
                for d in (1, 2)}
        (W / "weekly_scan_summary.json").write_text(json.dumps(s, indent=1))
        print(json.dumps(s, indent=1), flush=True)


if __name__ == "__main__":
    main()

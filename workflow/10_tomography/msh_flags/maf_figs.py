"""Step 6: figures of the evidence page (after scoring).

  python maf_figs.py --examples ID1,ID2,... --real IDr1,IDr2
FIG/ev_<ID>.png: record section of one event, origin -40 s .. +80 s: left, raw counts (10-Hz block means, each
trace scaled to its own range); right, 2-15 Hz band-pass of the full-rate trace; pick ticks (P blue, S orange);
stations of the signal set S in dark ink, others in grey labels. FIG/depth.png, FIG/years.png.
"""
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import maf_common as C  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#6b6a66", "#d9d8d4"
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#c03a2b"


def hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def read_full(sta, a, b):
    code, net = sta.split(".")
    rows = C.RV.index_rows(net, code, a, b)
    cb = C.RV.choose_band(rows)
    if cb is None:
        return []
    segs, _ = C.RV.read_window(rows, cb[1], a.timestamp() + 1, a.timestamp(), b.timestamp())
    return segs


def fig_event(eid, ev, picks, S, title, max_sta=12):
    e = ev.loc[eid]
    t0 = pd.Timestamp(e.t_origin)
    pk = picks[picks.ID == eid].copy()
    st = C.stations().set_index("station")
    lat = e.lat if pd.notna(e.lat) else e.v4_lat
    lon = e.lon if pd.notna(e.lon) else e.v4_lon
    pk["dist"] = [hav(lat, lon, st.lat.get(s, np.nan), st.lon.get(s, np.nan)) for s in pk.station]
    stas = pk.sort_values("dist").drop_duplicates("station").station.tolist()
    stas = [s for s in stas if s.split(".")[1] not in ("NC", "BK")][:max_sta]
    a, b = t0 - pd.Timedelta(seconds=40), t0 + pd.Timedelta(seconds=80)
    fig, axs = plt.subplots(1, 2, figsize=(12, 0.55 * len(stas) + 1.4), sharey=True)
    for k, s in enumerate(stas):
        y0 = -k
        segs = read_full(s, a - pd.Timedelta(seconds=5), b + pd.Timedelta(seconds=5))
        col = INK if s in S else MUTED
        for s0, sr, d in segs:
            t = s0 + np.arange(len(d)) / sr - t0.timestamp()
            n = max(1, int(round(sr / C.FS)))
            m = len(d) // n
            if m < 20:
                continue
            x = d[:m * n].astype(float).reshape(m, n).mean(1)
            x = x - np.median(x)
            tx = s0 + np.arange(m) * n / sr - t0.timestamp()
            r = np.ptp(x) or 1
            axs[0].plot(tx, y0 + 0.85 * (x - x.min()) / r - 0.42, lw=0.6, color=col)
            if len(d) > 4 * sr:
                f = C.RV.bandpass(d, sr, (2.0, 15.0))
                q = np.percentile(np.abs(f), 99.5) or 1
                axs[1].plot(t, y0 + 0.42 * np.clip(f / q, -1, 1), lw=0.35, color=col)
        for r in pk[pk.station == s].itertuples():
            tp = r.t.timestamp() - t0.timestamp()
            for ax in axs:
                ax.plot([tp, tp], [y0 - 0.45, y0 + 0.45], color=BLUE if r.phase == "P" else ORANGE, lw=1.4)
    for ax in axs:
        ax.set_yticks([-k for k in range(len(stas))])
        ax.set_yticklabels([s + (" *" if s in S else "") for s in stas], fontsize=7)
        ax.set_xlim(-40, 80)
        ax.grid(color=GRID, lw=0.4)
        ax.set_xlabel("s after the origin time")
    axs[0].set_title("raw counts, 10-Hz block means (own scale per trace)", fontsize=8, loc="left")
    axs[1].set_title("2-15 Hz band-pass (clipped)", fontsize=8, loc="left")
    lo = C.local(t0)
    fig.suptitle(f"{title}: event {eid}, {lo:%Y-%m-%d %H:%M:%S %Z} ({lo:%a}), depth {e.depth:.1f} km, "
                 f"{int(e.n_in)} of {int(e.n_tot)} picks on the signal, {int(e.n_match)} of {int(e.n_wf)} waveform "
                 f"matches; * = signal channel", fontsize=8.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(C.FIG / f"ev_{eid}.png", dpi=100)
    plt.close(fig)


def fig_depth(ev):
    box = ev[ev.in_box & ev.depth.notna()]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    bins = np.arange(-4, 122, 2)
    groups = [("unflagged box events", box[box.cls == "none"], MUTED),
              ("probable (not flagged)", box[box.cls == "probable"], ORANGE),
              ("certain (flagged)", box[box.cls == "certain"], RED)]
    for lab, x, c in groups:
        ax.hist(x.depth, bins, histtype="step" if lab.startswith("unfl") else "stepfilled",
                alpha=1 if lab.startswith("unfl") else 0.75, color=c, lw=1.2,
                weights=np.ones(len(x)) / max(len(x), 1), label=f"{lab} (n={len(x)})")
    ax.set_yscale("log")
    ax.set_xlabel("v6.0 depth, km below sea level")
    ax.set_ylabel("fraction of the group")
    ax.legend(fontsize=7)
    ax.grid(color=GRID, lw=0.4)
    ax.set_title("Mount St Helens box (46.05-46.30 N, 122.35-122.10 W)", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(C.FIG / "depth.png", dpi=110)
    plt.close(fig)


def fig_years(ev):
    yrs = list(range(2010, 2017))
    fig, ax = plt.subplots(figsize=(7, 2.8))
    bot = np.zeros(len(yrs))
    for cls, c in (("certain", RED), ("probable", ORANGE), ("mixed", BLUE)):
        n = np.array([((ev.cls == cls) & (ev.year == y)).sum() for y in yrs])
        ax.bar(yrs, n, bottom=bot, color=c, label=cls, width=0.7)
        bot += n
    ax.legend(fontsize=7)
    ax.set_ylabel("events")
    ax.grid(color=GRID, lw=0.4, axis="y")
    fig.tight_layout()
    fig.savefig(C.FIG / "years.png", dpi=110)
    plt.close(fig)


def fig_allstations(onset_utc="2012-05-02T18:57:58.8"):
    """Raw traces at the analog and the nearby digital stations around one weekly pulse."""
    on = pd.Timestamp(onset_utc, tz="UTC")
    stas = ["HSR.UW", "SHW.UW", "EDM.UW", "STD.UW", "JUN.UW", "SOS.UW", "FL2.UW", "CDF.UW", "ELK.UW", "TDL.UW",
            "MTM.UW", "LVP.UW", "RVW.UW", "LCW.UW", "KMO.UW", "HBO.UW", "SEP.CC", "STD.CC", "SWFL.CC", "B204.PB"]
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(stas) + 1.0))
    a, b = on - pd.Timedelta(seconds=25), on + pd.Timedelta(seconds=65)
    for k, s in enumerate(stas):
        segs, st = C.read10(s, a, b)
        for s0, x in segs:
            t = s0 + np.arange(len(x)) / C.FS - on.timestamp()
            r = np.ptp(x) or 1
            ax.plot(t, -k + 0.85 * (x - x.min()) / r - 0.42, lw=0.7, color=INK if s.endswith("UW") else BLUE)
            ax.text(64, -k + 0.1, f"{np.ptp(x):.0f}", fontsize=6.5, color=MUTED, ha="right")
    ax.set_yticks([-k for k in range(len(stas))])
    ax.set_yticklabels(stas, fontsize=7)
    ax.set_xlim(-25, 65)
    ax.set_xlabel(f"s after the pulse onset {C.local(on):%Y-%m-%d %H:%M:%S %Z}")
    ax.set_title("raw counts, 10-Hz block means, own scale per trace (number: peak-to-peak counts); "
                 "black UW analog, blue digital", fontsize=8, loc="left")
    ax.grid(color=GRID, lw=0.4)
    fig.tight_layout()
    fig.savefig(C.FIG / "raw_allstations.png", dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default="")
    ap.add_argument("--real", default="")
    a = ap.parse_args()
    ev = pd.read_csv(C.WORK / "events_scored.csv", low_memory=False).set_index("ID")
    ev["t_origin"] = pd.to_datetime(ev.t_origin, utc=True, format="ISO8601")
    S = set(json.load(open(C.WORK / "signal_stations.json"))["signal_stations"])
    fig_depth(ev)
    fig_years(ev)
    fig_allstations()
    picks = C.picks()
    for lab, ids in (("flagged", a.examples), ("not flagged", a.real)):
        for i in [int(x) for x in ids.split(",") if x]:
            cls = ev.loc[i, "cls"]
            fig_event(i, ev, picks, S, f"{lab}, class {cls}")
            print("fig", i, flush=True)


if __name__ == "__main__":
    main()

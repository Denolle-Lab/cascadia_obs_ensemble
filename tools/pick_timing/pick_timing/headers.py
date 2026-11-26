"""scan-headers: MiniSEED record headers per channel-day (headers only, 64 bytes per record where the reader
allows it).

Per channel-day (vertical channels of the scanned bands by default):
* day-merge drift test (D-B072): records are joined into segments as libmseed/ObsPy join them (start within half a
  sample of the previous record's end, same rate); within a segment resid = header start - (segment start +
  samples so far / nominal rate) is the error a day-merged trace carries. absmax_resid >= drift_threshold_s
  (default 5 ms = half a sample at 100 Hz) marks a station-day whose picks need ``retime``.
* restarts and gaps: every record whose start is off the previous record's end by more than tear_samples
  samples (kind tear < 60 s, gap >= 60 s, overlap < 0). Consecutive days of one station are scanned in order so
  tears at midnight are seen. A free-running clock is re-synchronised at such a restart.
* timing information when present: blockette-1001 timing quality (tq), I/O flag 'clock locked', the
  time-correction field and its 'applied' bit. Many archives carry none of these (then tq = -1, locked 0).

Outputs: <tag>_channel_days.csv, <tag>_events.csv (one row per tear/gap/overlap), provenance JSON.
"""
from __future__ import annotations

import os
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .mseed import segments
from .readers import make_reader
from .tables import posix_to_iso, provenance, split_station, write_json

DAY = 86400.0


def channel_day_metrics(h: pd.DataFrame, prev_last=None, nominal=None, thr=0.005, tear_samples=0.5):
    """Metrics and events of one channel's records over one day (h: header table of that channel)."""
    h = h.sort_values("start").reset_index(drop=True)
    rate = float(h.rate.mode().iloc[0]) if len(h) else np.nan
    # tears (including one at midnight, against the previous day's last record) ...
    he = h if prev_last is None else pd.concat([prev_last[h.columns].to_frame().T.astype(h.dtypes.to_dict()), h],
                                               ignore_index=True)
    se = segments(he, tear_samples)
    events = []
    gaps = se.gap_s.values
    with np.errstate(invalid="ignore"):
        tears = np.where(np.abs(gaps) > tear_samples / np.where(se.rate.values > 0, se.rate.values, np.nan))[0]
    for k in tears:
        g = float(gaps[k])
        kind = "overlap" if g < 0 else ("tear" if g < 60 else "gap")
        events.append(dict(time_s=float(se.start.values[k]), gap_s=g, kind=kind,
                           locked_before=bool(se.locked.values[k - 1]) if k > 0 else None,
                           locked_after=bool(se.locked.values[k]), tq_before=int(se.tq.values[k - 1]) if k > 0 else -1,
                           tq_after=int(se.tq.values[k])))
    # ... and the day-merge metrics of this day alone (a day-reading picker starts counting at the day's first record)
    s = segments(h, tear_samples)
    res = s.resid.values
    t = s.start.values
    seg = s.seg.values
    first_t = np.array([t[seg == q][0] for q in np.unique(seg)])
    tseg0 = dict(zip(np.unique(seg), first_t))
    dt = t - np.array([tseg0[q] for q in seg])
    den = float(np.sum(dt * dt))
    slope = float(np.sum(res * dt) / den * DAY) if den > 0 else np.nan
    end = t + s.npts.values / np.where(s.rate.values > 0, s.rate.values, np.nan)
    dur = pd.Series(end).groupby(seg).max().values - pd.Series(t).groupby(seg).min().values
    big = np.where(np.abs(res) > thr)[0]
    m = dict(n_records=len(s), n_segments=int(len(np.unique(seg))), rate=rate, n_rates=int(s.rate.nunique()),
             npts=int(s.npts.sum()), max_resid_s=float(res.max()) if len(res) else np.nan,
             min_resid_s=float(res.min()) if len(res) else np.nan,
             absmax_resid_s=float(np.abs(res).max()) if len(res) else np.nan, slope_s_per_day=slope,
             apparent_rate=float(s.npts.sum() / dur.sum()) if dur.sum() > 0 else np.nan,
             longest_seg_h=float(dur.max() / 3600) if len(dur) else np.nan,
             first_t=posix_to_iso(t[0], 3) if len(t) else "", last_t=posix_to_iso(end[-1], 3) if len(t) else "",
             first_over_threshold=posix_to_iso(t[big[0]], 3) if len(big) else "",
             tq_median=float(np.median(s.tq)), tq_min=int(s.tq.min()), tq_max=int(s.tq.max()),
             locked_frac=float(s.locked.mean()), tcorr_nonzero=int((s.tcorr_s != 0).sum()),
             tcorr_median_s=float(np.median(s.tcorr_s)), corr_applied_frac=float(s.corr_applied.mean()),
             n_tears=sum(e["kind"] == "tear" for e in events), n_gaps=sum(e["kind"] == "gap" for e in events),
             n_overlaps=sum(e["kind"] == "overlap" for e in events),
             drifting=bool(len(res) and np.abs(res).max() >= thr))
    if nominal:
        m["rate_is_nominal"] = bool(abs(rate - nominal) < 1e-9)
    last = s.iloc[-1] if len(s) else None
    return m, events, last


def _scan_station(args):
    station, days, spec, sc, pattern = args
    reader = make_reader(spec)
    net, sta = split_station([station], pattern)
    net, sta = net[0], sta[0]
    bands = sorted({c[:2] for c in sc["channels"]})
    rows, evs = [], []
    prev = {}
    prev_day = None
    for day0 in days:
        if prev_day is not None and day0 - prev_day > DAY + 1:
            prev = {}  # not consecutive: no carry-over
        prev_day = day0
        base = dict(station=station, net=net, day=posix_to_iso(day0, 0)[:10])
        try:
            h = reader.day_heads(net, sta, day0, bands)
        except Exception as e:  # noqa: BLE001
            rows.append({**base, "status": f"error {type(e).__name__}: {e}"[:200]})
            continue
        if h is None or not len(h):
            rows.append({**base, "status": "no_data"})
            prev = {}
            continue
        h = h[h.cha.str[:2].isin(bands) & h.cha.str[-1].isin(list(sc["components"]))]
        h = h[h.valid_time] if "valid_time" in h.columns else h
        if not len(h):
            rows.append({**base, "status": "no_vertical"})
            continue
        for (loc, cha), hh in h.groupby(["loc", "cha"]):
            key = (loc, cha)
            try:
                m, ev, last = channel_day_metrics(hh, prev.get(key), sc.get("nominal_rate"),
                                                  sc["drift_threshold_s"], sc["tear_samples"])
            except Exception as e:  # noqa: BLE001
                rows.append({**base, "loc": loc, "cha": cha, "status": f"error {type(e).__name__}: {e}"[:200]})
                traceback.print_exc()
                continue
            prev[key] = last
            rows.append({**base, "loc": loc, "cha": cha, "status": "ok", **m})
            for e in ev:
                evs.append(dict(station=station, loc=loc, cha=cha, time=posix_to_iso(e["time_s"], 3), **e))
    return rows, evs


def build_tasks(cfg, a) -> dict:
    """{station: [day0 POSIX, ...]}"""
    pattern = cfg.get("station_id", "{sta}.{net}")
    tasks = {}
    if a.stations:
        d0 = pd.Timestamp(a.start).floor("D").timestamp()
        d1 = pd.Timestamp(a.end).timestamp()
        days = list(np.arange(d0, d1, DAY))
        for s in a.stations.split(","):
            tasks[s] = days
    if a.from_picks:
        from .tables import load
        pk = load(cfg, "picks")
        pk["day0"] = np.floor(pk.time_s / DAY) * DAY
        sd = pk.groupby(["station", "day0"]).size().reset_index()
        if a.sample != "all":
            key = pd.to_datetime(sd.day0, unit="s").dt.strftime("%G-%V" if a.sample == "weekly" else "%Y-%m")
            sd = sd[~sd.assign(k=key).duplicated(["station", "k"])]
        for s, g in sd.groupby("station"):
            tasks.setdefault(s, [])
            tasks[s] = sorted(set(tasks[s]) | set(g.day0))
    nets = {s: split_station([s], pattern)[0][0] for s in tasks}
    if a.networks:
        keep = set(a.networks.split(","))
        tasks = {s: d for s, d in tasks.items() if nets[s] in keep}
    if a.exclude_networks:
        drop = set(a.exclude_networks.split(","))
        tasks = {s: d for s, d in tasks.items() if nets[s] not in drop}
    return tasks


def cli_scan(cfg, a):
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(cfg, a)
    sc = dict(cfg["scan"], nominal_rate=cfg["retime"]["nominal_rate"])
    # split long station ranges into ~60-day jobs (carry-over lost only at job edges)
    jobs = []
    for s, days in tasks.items():
        for i in range(0, len(days), 60):
            jobs.append((s, days[i:i + 60], cfg["waveforms"], sc, cfg.get("station_id", "{sta}.{net}")))
    print(f"scan-headers: {len(tasks)} stations, {sum(len(d) for d in tasks.values())} station-days, {len(jobs)} jobs",
          flush=True)
    t0 = time.time()
    if cfg["workers"] > 1 and len(jobs) > 1:
        with Pool(cfg["workers"]) as p:
            res = list(p.imap(_scan_station, jobs, chunksize=1))
    else:
        res = [_scan_station(j) for j in jobs]
    rows = [r for rr, _ in res for r in rr]
    evs = [e for _, ee in res for e in ee]
    cd = pd.DataFrame(rows)
    ev = pd.DataFrame(evs, columns=["station", "loc", "cha", "time", "time_s", "gap_s", "kind", "locked_before",
                                    "locked_after", "tq_before", "tq_after"])
    f1 = out / f"{a.tag}_channel_days.csv"
    f2 = out / f"{a.tag}_events.csv"
    cd.to_csv(f1, index=False)
    ev.to_csv(f2, index=False)
    write_json(provenance({}, {"tasks": {s: [posix_to_iso(d[0], 0)[:10], posix_to_iso(d[-1], 0)[:10], len(d)]
                                         for s, d in tasks.items()},
                               "scan": sc, "waveforms": cfg["waveforms"], "channel_days": len(cd), "events": len(ev),
                               "seconds": round(time.time() - t0)}, md5_inputs=False),
               out / f"{a.tag}.provenance.json")
    nd = int(cd.get("drifting", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if len(cd) else 0
    print(f"{len(cd)} channel-days ({nd} with day-merge drift >= {sc['drift_threshold_s']} s), {len(ev)} tears/gaps "
          f"-> {f1}, {f2} ({time.time() - t0:.0f} s)", flush=True)
    return 0

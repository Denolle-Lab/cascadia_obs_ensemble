"""retime: re-time picks of a day-merging picker from their own MiniSEED record headers (D-B072 / D-B075 method).

Generalised from a project's pick re-timing scripts, with waveform access through a reader.

For every pick of a station-day to re-time:
 1. Rebuild the picker's day-trace layout: the reader's whole-day blocks (each read with ObsPy, as the picker
    read them), trimmed to [day, day + day_seconds), the picker's channel rule (first band of
    ``channel_priority`` with a vertical; ``alternate_rules`` are tried too), then Stream.merge. To know which
    original sample sits at each merged index, the merge is run on index traces (value = trace number * 1e8 +
    sample number, fill -1): ObsPy places segments by time only, so the layout is the picker's.
 2. The pick's merged index i = (pick time - merged start) * rate must be an integer within ``grid_tolerance``
    (the pick sits on the picker's sample grid; else flag off_grid). Merged index -> (trace k, sample q) ->
    MiniSEED record r and position p (cumulative npts of the records of that trace).
 3. Check: the ``value_window`` raw counts each side of the sample in the day trace equal those at the same
    position of a short read of records r-3..r+3 (value_mismatch otherwise); uniqueness recorded (n_match).
 4. True time = start of record r read alone + p / rate (each sample timed by its own record header).
    correction = true time - pick time. No rate model anywhere.
A station-day is corrected (``applied``) when the largest header-vs-count difference of its channel reaches
``apply_threshold_s`` (half a sample at 100 Hz); then every re-timed pick of that day gets its header time.

Flags: ok | ok_nonunique | in_fill (pick on a merge fill: timed from the nearest real sample) | off_grid |
value_mismatch | map_fail | no_data | rate_not_nominal | error. Picks that cannot be re-timed keep their time.
"""
from __future__ import annotations

import io
import os
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .mseed import parse_headers
from .readers import make_reader
from .tables import load, posix_to_iso, provenance, split_station, write_json

ENC = 1e8
DAY = 86400.0


def _vert_ok(cha, comps):
    return cha[-1] in comps


def choose_channel(chans: list[str], bands: list[str], comps: str):
    """First band of `bands` with a vertical channel (the picker's select_channels rule)."""
    if not any(_vert_ok(c, comps) for c in chans):
        return None
    for b in bands:
        if any(c.startswith(b) for c in chans):
            zs = [c for c in chans if c.startswith(b) and _vert_ok(c, comps)]
            return (b, comps) if zs else None
    return None


class DayTrace:
    """The picker's merged vertical day trace of one band + the map merged index -> (block trace, sample)."""

    def __init__(self, blocks, band, comps, day0, day_seconds, nominal):
        import obspy
        from obspy import Stream, Trace, UTCDateTime
        t0 = UTCDateTime(day0)
        self.pre = []
        for blk in blocks:
            if blk.cha and not (blk.cha.startswith(band) and _vert_ok(blk.cha, comps)):
                continue
            st = obspy.read(io.BytesIO(blk.data), format="MSEED")
            h = parse_headers(blk.data)
            for u in st:
                if not (u.stats.channel.startswith(band) and _vert_ok(u.stats.channel, comps)):
                    continue
                tr = Stream([u.copy()]).trim(starttime=t0, endtime=t0 + day_seconds)
                if len(tr) == 0 or tr[0].stats.npts == 0:
                    continue
                tr = tr[0]
                m = int(round((tr.stats.starttime - u.stats.starttime) * u.stats.sampling_rate))
                self.pre.append(dict(tr=tr, u=u, buf=blk.data, h=h, m=m, source=blk.source))
        self.ok = bool(self.pre)
        if not self.ok:
            return
        ids = sorted({p["tr"].id for p in self.pre})
        self.ids = ids
        self.multi_loc = len(ids) > 1
        rates = {p["tr"].stats.sampling_rate for p in self.pre}
        self.rate = rates.pop() if len(rates) == 1 else None
        self.rate_ok = self.rate is not None and abs(self.rate - nominal) == 0
        idx = Stream()
        for k, p in enumerate(self.pre):
            tr = p["tr"]
            idx += Trace(data=k * ENC + np.arange(tr.stats.npts, dtype=np.float64), header=tr.stats.copy())
        idx.merge(method=0, fill_value=-1.0)
        idx = [t for t in idx if t.id == ids[0]][0]
        self.start = idx.stats.starttime
        self.map = idx.data

    def day_absmax(self):
        """Largest |record header time - count time| over the records of this channel in the day."""
        if getattr(self, "_dam", None) is not None:
            return self._dam
        mx = 0.0
        for p in self.pre:
            u, h = p["u"], p["h"]
            ch = h[(h.cha == u.stats.channel) & (h["loc"] == u.stats.location)]
            tu = float(u.stats.starttime)
            j0 = np.where(np.abs(ch.start.values - tu) < 2e-6)[0]
            if not len(j0):
                continue
            hh = ch.iloc[j0[0]:]
            cum = np.r_[0, np.cumsum(hh.npts.values)][:len(hh)]
            keep = cum < u.stats.npts
            res = hh.start.values[keep] - (tu + cum[keep] / u.stats.sampling_rate)
            if len(res):
                mx = max(mx, float(np.abs(res).max()))
        self._dam = mx
        return mx

    def locate(self, i):
        v = self.map[i]
        if v < 0:
            return None
        k = int(v // ENC)
        pos = int(round(v - k * ENC))
        return k, pos + self.pre[k]["m"]

    def true_time(self, k, q, W=50):
        import obspy
        from numpy.lib.stride_tricks import sliding_window_view
        p = self.pre[k]
        u, h, buf = p["u"], p["h"], p["buf"]
        cha_h = h[(h.cha == u.stats.channel) & (h["loc"] == u.stats.location)]
        tu = float(u.stats.starttime)
        j0 = np.where(np.abs(cha_h.start.values - tu) < 2e-6)[0]
        if len(j0) == 0:
            return dict(flag="map_fail", why="trace start not a record start")
        j0 = j0[0]
        hh = cha_h.iloc[j0:]
        cum = np.r_[0, np.cumsum(hh.npts.values)]
        if cum[-1] < u.stats.npts:
            return dict(flag="map_fail", why="records shorter than trace")
        r = int(np.searchsorted(cum, q, side="right") - 1)
        if r >= len(hh) or q >= u.stats.npts:
            return dict(flag="map_fail", why="sample beyond trace")
        pos = int(q - cum[r])
        off = int(hh.offset.iloc[r])
        allnext = h.offset.values[h.offset.values > off]
        rec_end = int(allnext.min()) if len(allnext) else len(buf)
        rec = obspy.read(io.BytesIO(buf[off:rec_end]), format="MSEED")[0]
        if rec.stats.npts != int(hh.npts.iloc[r]):
            return dict(flag="map_fail", why="record npts mismatch")
        t_true = rec.stats.starttime + pos / rec.stats.sampling_rate
        val_ok = bool(rec.data[pos] == u.data[q])
        r_lo, r_hi = max(0, r - 3), min(len(hh) - 1, r + 3)
        o_lo = int(hh.offset.iloc[r_lo])
        allnext = h.offset.values[h.offset.values > int(hh.offset.iloc[r_hi])]
        o_hi = int(allnext.min()) if len(allnext) else len(buf)
        ss = obspy.read(io.BytesIO(buf[o_lo:o_hi]), format="MSEED").select(id=u.id)
        jj = int(cum[r] - cum[r_lo] + pos)
        a, b = max(0, q - W), min(u.stats.npts, q + W + 1)
        win = u.data[a:b]
        n_match, short_ok, short_dt = 0, False, np.nan
        if len(ss) >= 1:
            s0 = ss[0]
            ja, jb = jj - (q - a), jj + (b - q)
            if ja >= 0 and jb <= s0.stats.npts:
                short_ok = bool(np.array_equal(s0.data[ja:jb], win))
            if s0.stats.npts >= len(win):
                sw = sliding_window_view(s0.data, len(win))
                n_match = int(np.all(sw == win[None, :], axis=1).sum())
            short_dt = float((s0.stats.starttime + jj / s0.stats.sampling_rate) - t_true)
        flag = "ok" if (val_ok and short_ok and n_match == 1) else (
            "ok_nonunique" if (val_ok and short_ok) else "value_mismatch")
        return dict(flag=flag, t_true=float(t_true), record_start=str(rec.stats.starttime), record_pos=pos,
                    record_offset=off, n_match=n_match, short_minus_true_ms=short_dt * 1e3, file=p["source"],
                    channel=u.id)


def retime_day(reader, net, sta, day0, picks: pd.DataFrame, q: dict):
    """Re-time the picks (rows with pick_row, time_s, event_id, phase) of one station-day."""
    out = []
    base = dict(net=net, sta=sta, day=posix_to_iso(day0, 0)[:10])
    blocks = reader.day_blocks(net, sta, day0)
    chans = sorted({b.cha for b in blocks if b.cha})
    if not chans:  # readers without per-block channel (FDSN): from the headers
        chans = sorted(set().union(*[set(parse_headers(b.data).cha) for b in blocks])) if blocks else []
    cands = []
    for rule in [q["channel_priority"]] + list(q.get("alternate_rules", [])):
        ch = choose_channel(chans, rule, q["components"])
        if ch is not None and ch not in cands:
            cands.append(ch)
    if not cands:
        for pk in picks.itertuples():
            out.append({**base, "pick_row": pk.pick_row, "flag": "no_data", "why": "no vertical channel"})
        return out
    days = {}
    for ch in cands:
        d = DayTrace(blocks, ch[0], ch[1], day0, q["day_seconds"], q["nominal_rate"])
        if d.ok and all(d.ids[0] != o.ids[0] for o in days.values()):
            days[ch] = d
    if not days:
        for pk in picks.itertuples():
            out.append({**base, "pick_row": pk.pick_row, "flag": "no_data", "why": "no samples in the day"})
        return out
    tol = q["grid_tolerance"]
    for pk in picks.itertuples():
        from obspy import UTCDateTime
        tp = UTCDateTime(pk.time_s)
        rec = {**base, "pick_row": pk.pick_row, "event_id": pk.event_id, "phase": pk.phase,
               "t_pick": posix_to_iso(pk.time_s, 6)}
        grids = {}
        for ch, d in days.items():
            x = (tp - d.start) * d.rate if d.rate else np.nan
            i = int(round(x)) if np.isfinite(x) else -1
            grids[ch] = (abs(x - i) if np.isfinite(x) else 1.0, i)
        on = [ch for ch, (fr, i) in grids.items() if fr < tol and 0 <= i < len(days[ch].map)]
        rec["n_channel_candidates"] = len(days)
        rec["channel_choice"] = "+".join(d.ids[0] for d in days.values())
        if not on:
            ch = min(grids, key=lambda c: grids[c][0])
            rec.update(flag="off_grid", why=f"grid fraction {grids[ch][0]:.4f}", merged_start=str(days[ch].start))
            out.append(rec)
            continue
        if len(on) > 1:
            rec["why"] = "several candidate channels on grid; first rule used"
        ch = on[0]
        d = days[ch]
        i = grids[ch][1]
        rec.update(merged_index=i, merged_start=str(d.start), grid_frac=grids[ch][0], n_segments=len(d.pre),
                   multi_loc=d.multi_loc, rate_ok=d.rate_ok, day_absmax_s=d.day_absmax())
        if not d.rate_ok:
            rec.update(flag="rate_not_nominal", why=f"rate {d.rate}")
            out.append(rec)
            continue
        loc = d.locate(i)
        if loc is not None:
            res = d.true_time(*loc, W=q["value_window"])
            rec.update(res)
            if "t_true" in res:
                rec["correction_s"] = float(res["t_true"] - float(tp))
                rec["method"] = "record_header_sample"
        else:
            left, right = i - 1, i + 1
            while left >= 0 and d.map[left] < 0:
                left -= 1
            while right < len(d.map) and d.map[right] < 0:
                right += 1
            info = {}
            for side, j in (("left", left), ("right", right)):
                if 0 <= j < len(d.map):
                    r2 = d.true_time(*d.locate(j), W=q["value_window"])
                    if "t_true" in r2:
                        info[side] = (abs(i - j), float(r2["t_true"] + (i - j) / d.rate - float(tp)), r2["flag"])
            rec["fill_left_corr_s"] = info.get("left", (0, np.nan, ""))[1]
            rec["fill_right_corr_s"] = info.get("right", (0, np.nan, ""))[1]
            if info:
                side = min(info, key=lambda s: info[s][0])
                rec.update(flag="in_fill", why=f"merge fill; timed from nearest real sample ({side}, "
                                                f"{info[side][0]} samples)", correction_s=info[side][1],
                           method="nearest_real_sample")
                rec["t_true"] = float(tp) + info[side][1]
            else:
                rec.update(flag="in_fill", why="merge fill, no real sample")
        out.append(rec)
    return out


_G = {}


def _init(picks, spec, q, pattern):
    _G.update(picks=picks, spec=spec, q=q, pattern=pattern)


def _run_chunk(args):
    k, tasks, outdir = args
    fn = Path(outdir) / f"chunk_{k:05d}.csv.gz"
    if fn.exists():
        return k, "skip"
    reader = _G.get("reader") or make_reader(_G["spec"])
    _G["reader"] = reader
    P = _G["picks"]
    recs = []
    for station, day0 in tasks:
        net, sta = split_station([station], _G["pattern"])
        net, sta = net[0], sta[0]
        pk = P[(P.station == station) & (P.day0 == day0)]
        try:
            recs += retime_day(reader, net, sta, day0, pk, _G["q"])
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            for p in pk.itertuples():
                recs.append(dict(net=net, sta=sta, day=posix_to_iso(day0, 0)[:10], pick_row=p.pick_row, flag="error",
                                 why=f"{type(e).__name__}: {e}"[:200]))
    pd.DataFrame(recs).to_csv(str(fn) + ".tmp", index=False, compression="gzip")
    os.replace(str(fn) + ".tmp", fn)
    return k, len(recs)


def assemble(outdir, threshold_s: float) -> pd.DataFrame:
    """Merge chunk files; applied = re-timed (ok/ok_nonunique/in_fill with a correction) on a station-day whose
    largest header-vs-count difference reaches threshold_s."""
    parts = [pd.read_csv(f) for f in sorted(Path(outdir).glob("chunk_*.csv.gz"))]
    c = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["pick_row", "flag"])
    if "correction_s" not in c.columns:
        c["correction_s"] = np.nan
    if "day_absmax_s" not in c.columns:
        c["day_absmax_s"] = np.nan
    good = c.flag.isin(["ok", "ok_nonunique", "in_fill"]) & c.correction_s.notna()
    dmax = c.groupby(["net", "sta", "day"]).day_absmax_s.transform("max")
    c["applied"] = good & (dmax >= threshold_s)
    c["timing_correction_s"] = np.where(c.applied, c.correction_s, 0.0)
    return c.sort_values("pick_row").reset_index(drop=True)


def cli_retime(cfg, a):
    q = dict(cfg["retime"])
    q["components"] = q.get("components", cfg["scan"]["components"])
    out = Path(cfg["output_dir"])
    pk = load(cfg, "picks")
    pk["pick_row"] = np.arange(len(pk))
    pk["day0"] = np.floor(pk.time_s / DAY) * DAY
    if a.tasks:
        t = pd.read_csv(a.tasks)
        t["day0"] = pd.to_datetime(t.day).astype("int64") / 1e9
        sd = t[["station", "day0"]]
    elif a.scan:
        s = pd.read_csv(a.scan)
        s = s[s.status == "ok"]
        s = s[s.absmax_resid_s >= q["apply_threshold_s"]]
        s["day0"] = pd.to_datetime(s.day).astype("int64") / 1e9
        sd = s[["station", "day0"]].drop_duplicates()
    elif a.all_days:
        sd = pk[["station", "day0"]].drop_duplicates()
    else:
        raise SystemExit("retime: give --tasks, --scan or --all-days")
    sd = sd.merge(pk[["station", "day0"]].drop_duplicates(), on=["station", "day0"])
    sd = sd.sort_values(["day0", "station"]).reset_index(drop=True)
    pk = pk.merge(sd, on=["station", "day0"])
    outdir = out / a.tag / "chunks"
    outdir.mkdir(parents=True, exist_ok=True)
    CH = 40
    tasks = list(zip(sd.station, sd.day0))
    jobs = [(k, tasks[i:i + CH], str(outdir)) for k, i in enumerate(range(0, len(tasks), CH))]
    print(f"retime: {len(tasks)} station-days, {len(pk)} picks, {len(jobs)} chunks -> {outdir}", flush=True)
    t0 = time.time()
    args = (pk[["pick_row", "station", "day0", "time_s", "event_id", "phase"]], cfg["waveforms"], q,
            cfg.get("station_id", "{sta}.{net}"))
    if cfg["workers"] > 1 and len(jobs) > 1:
        with Pool(cfg["workers"], initializer=_init, initargs=args) as pool:
            for n, (k, r) in enumerate(pool.imap_unordered(_run_chunk, jobs), 1):
                if n % 20 == 0 or n == len(jobs):
                    print(f"  {n}/{len(jobs)} chunks {time.time() - t0:.0f} s", flush=True)
    else:
        _init(*args)
        for j in jobs:
            _run_chunk(j)
    c = assemble(outdir, q["apply_threshold_s"])
    fn = out / a.tag / "corrections.csv.gz"
    c.to_csv(fn, index=False)
    write_json(provenance({"picks": cfg["picks"]["path"]}, {"retime": q, "waveforms": cfg["waveforms"],
                                                            "station_days": len(tasks), "picks": len(pk),
                                                            "flags": c.flag.value_counts().to_dict(),
                                                            "applied": int(c.applied.sum())}),
               out / a.tag / "retime.provenance.json")
    print(f"{len(c)} picks re-timed: flags {c.flag.value_counts().to_dict()}, applied {int(c.applied.sum())} -> {fn}",
          flush=True)
    return 0

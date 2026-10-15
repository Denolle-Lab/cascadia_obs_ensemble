#!/usr/bin/env python3
"""
Route A amplitudes for every pick of the ver5 pick table, measured once per station-day.

route_a_wa_amplitudes.py measures one associated pick at a time: it fetches a padded
window around the pick and removes the response on it. For the 60.9 M picks of ver5
that is ~60 M fetches and deconvolutions. This script does the same measurement per
station-day: it reads the day once (one sensor only, chosen from the pnwstore index
before any data is read), removes the response once on the whole day and then measures
every pick of the day with array operations.

The processing chain is the one of route_a_wa_amplitudes.py (same sensor preference,
same response removal, Wood-Anderson simulation, 1 Hz high-pass, 0.05-2 Hz displacement
band, 100 Hz resampling, max over the components of the sensor). Two things differ:
  * the response is removed on the whole contiguous segment (hours) instead of a window
    padded by 100 s, so the long-period padding problem of the displacement band goes
    away. Picks closer than ~100 s to a data gap or a day boundary are flagged with
    `seg_margin_s` (seconds from the pick to the nearest segment edge);
  * the source distance is unknown before association, so the window cannot be scaled
    with distance here. Every pick gets the running maximum from 0.3 s before the pick
    to W s after it, for W in WINDOWS (1..15 s every 0.25 s, 16..20 s every 1 s, 25..60 s every 5 s). After
    association, `select` takes the column W = the Route A post-window rounded up to the
    next grid value (phase_window() in route_a_wa_amplitudes.py) and writes the
    raw_wa_amplitudes CSV that route_a_build_dataset.py reads.

Output: one parquet file per station-year, amps_{net}.{sta}.{year}.parquet, one row per
pick, keyed by `pick_id` (the pick_id column of the ver5 table on Zenodo; ver3 and ver4
ids are unchanged in ver5). Columns: see DATA_COLUMNS below and README_amps.md, which
`consolidate` writes next to the files.

Subcommands (run in this order; scale/run_day_amplitudes.sh drives them):
  plan         ver5 csv.gz -> per-shard pick lists and task lists (station-days)
  run          one shard: measure its station-days, resumable
  consolidate  shard chunks -> amps_{net}.{sta}.{year}.parquet + summary
  select       associated picks (pick_id + distance + phase) -> raw_wa_amplitudes CSV
  validate     compare with the per-pick v2 amplitudes in arrivals_v3.csv

Environment: the default pixi env (pyarrow) with pnwstore on PYTHONPATH, see
scale/run_day_amplitudes.sh.
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.join(HERE, "..", "..", "utils"))

# s after the pick. 0.25 s steps up to 15 s (every Route A P window is <= 15 s): rounding a
# P window up by < 0.25 s stays inside the 0.2 tSP margin Route A keeps before the S wave
# (a 1 s grid let S into short-distance P windows, up to +0.9 log10 units).
WINDOWS = ([round(1.0 + 0.25 * i, 2) for i in range(57)]          # 1.00 .. 15.00
           + [16.0, 17.0, 18.0, 19.0, 20.0] + [25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0])
PRE = 0.3                 # s before the pick (phase_window)
NOISE = (11.0, 1.0)       # noise RMS from tp-11 s to tp-1 s, as route_a_wa_amplitudes.py
SR = 100.0                # resampling rate (Hz), as --sample-rate there
MIN_SEG_S = 300.0         # shorter contiguous segments are not deconvolved
TAPER_S = 10.0            # cosine taper length at each segment end

def _wname(w):
    return f"{w:05.2f}s".replace(".", "p")          # 1.25 -> 01p25s


WA_COLS = [f"wa_mm_{_wname(w)}" for w in WINDOWS]
DISP_COLS = [f"disp_um_{_wname(w)}" for w in WINDOWS]
DATA_COLUMNS = (["pick_id", "phase"] + WA_COLS + DISP_COLS
                + ["wa_noise_mm", "disp_noise_um", "n_comp", "seg_margin_s",
                   "sensor", "epoch", "status"])


# ----------------------------------------------------------------------------- plan
def cmd_plan(a):
    import pyarrow as pa
    import pyarrow.csv as pacsv

    os.makedirs(a.out, exist_ok=True)
    t = time.time()
    cols = ["pick_id", "network", "station", "label", "pick_time"]
    tbl = pacsv.read_csv(
        a.picks,
        convert_options=pacsv.ConvertOptions(
            include_columns=cols,
            column_types={"pick_id": pa.int64(), "network": pa.string(),
                          "station": pa.string(), "label": pa.string(),
                          "pick_time": pa.string()}),
        read_options=pacsv.ReadOptions(block_size=1 << 26))
    df = tbl.to_pandas()
    del tbl
    print(f"read {len(df):,} picks in {time.time() - t:.0f} s", flush=True)
    if df["pick_id"].duplicated().any():
        sys.exit("pick_id is not unique")
    df["t"] = (pd.to_datetime(df["pick_time"], format="ISO8601", utc=True)
               .astype("int64") / 1e9)
    df = df.drop(columns="pick_time")
    df["day"] = (df["t"] // 86400).astype("int32")       # days since 1970-01-01
    df = df.sort_values(["network", "station", "day", "t"], kind="stable")
    df = df.reset_index(drop=True)

    tasks = (df.groupby(["network", "station", "day"], sort=False)
               .size().rename("n_picks").reset_index())
    tasks["task"] = (tasks["network"] + "." + tasks["station"] + "."
                     + pd.to_datetime(tasks["day"], unit="D").dt.strftime("%Y-%m-%d"))
    # interleave station-days over shards so every shard gets a similar mix of
    # stations and years; order inside a shard stays (station, day)
    tasks["shard"] = np.arange(len(tasks)) % a.nshard
    tasks.to_csv(os.path.join(a.out, "tasks.csv"), index=False)

    df = df.merge(tasks[["network", "station", "day", "shard"]],
                  on=["network", "station", "day"], how="left")
    sd = os.path.join(a.out, "shards")
    os.makedirs(sd, exist_ok=True)
    for s, g in df.groupby("shard"):
        g.drop(columns="shard").to_parquet(os.path.join(sd, f"picks_{s:03d}.parquet"),
                                           index=False)
    meta = dict(picks=os.path.abspath(a.picks), n_picks=int(len(df)),
                n_tasks=int(len(tasks)), nshard=a.nshard, windows=WINDOWS)
    json.dump(meta, open(os.path.join(a.out, "plan.json"), "w"), indent=1)
    print(f"{len(df):,} picks, {len(tasks):,} station-days, {a.nshard} shards "
          f"({time.time() - t:.0f} s)")
    print(tasks.groupby("network")["n_picks"].agg(["size", "sum"]).to_string())


# ------------------------------------------------------------------------------ run
class DayReader:
    """One sensor of one station-day: pnwstore (index first, then that sensor only),
    NC/BK from NCEDC."""

    def __init__(self, inv):
        from route_a_wa_amplitudes import SENSOR_PREF, GROUND_MOTION_INSTR
        self.inv = inv
        self.pref = SENSOR_PREF
        self.gm = GROUND_MOTION_INSTR
        self._client = None
        self.resp = Responses(inv)

    @property
    def client(self):
        if self._client is None:
            from pnwstore.mseed import WaveformClient
            self._client = WaveformClient()
        return self._client

    def has_response(self, seed_id, t):
        try:
            self.inv.get_response(seed_id, t)
            return True
        except Exception:
            return False

    def choose(self, net, sta, pairs, t):
        """pairs: iterable of (location, channel). Best sensor with a response."""
        rank = {k: i for i, k in enumerate(self.pref)}
        cands = {}
        for loc, ch in pairs:
            if len(ch) != 3 or ch[1] not in self.gm or ch[:2] not in rank:
                continue
            cands.setdefault((loc, ch[:2]), []).append(ch)
        no_resp = []
        for loc, bi in sorted(cands, key=lambda s: (rank[s[1]], s[0])):
            chans = [c for c in cands[(loc, bi)]
                     if self.has_response(f"{net}.{sta}.{loc}.{c}", t)]
            if chans:
                return loc, bi, ""
            no_resp.append(f"{loc}.{bi}")
        if not cands:
            return None, None, "no_ground_motion_channel"
        return None, None, "no_response:" + ",".join(no_resp)[:60]

    def read(self, net, sta, day_t0):
        from obspy import Stream, UTCDateTime
        t0 = UTCDateTime(day_t0)
        noon = t0 + 43200
        if net in ("NC", "BK"):
            from data_client import get_waveforms
            st = get_waveforms(net, sta, "?H?,?N?,?L?,?P?", t0, t0 + 86400,
                               source="pnwstore")
            loc, bi, why = self.choose(net, sta, {(tr.stats.location, tr.stats.channel)
                                                  for tr in st}, noon)
            if why:
                return Stream(), "", why
            st = Stream([tr for tr in st if tr.stats.location == loc
                         and tr.stats.channel[:2] == bi])
        else:
            rows = set(self.client._query(["location", "channel"], network=net,
                                          station=sta, year=t0.year, doy=t0.julday))
            if not rows:
                return Stream(), "", "no_data"
            loc, bi, why = self.choose(net, sta, {(r[0] or "", r[1]) for r in rows}, noon)
            if why:
                return Stream(), "", why
            st = self.client.get_waveforms(network=net, station=sta,
                                           location=loc or "--", channel=f"{bi}?",
                                           year=t0.year, doy=t0.julday)
        if len(st) == 0:
            return st, "", "no_data"
        return st, f"{loc}.{bi}", ""


RESP_GRID = 1 << 17       # response evaluated once per channel on this many points
PAD_S = 600.0             # zero padding at each segment end before the FFT (s)


class Responses:
    """Instrument responses per channel, evaluated once on a fine grid from 0 to the
    Nyquist of SR (as obspy does on its nfft grid) and interpolated onto each
    segment's FFT grid. Water level as obspy.signal.invsim.invert_spectrum, with the
    maximum taken over the fine grid."""

    def __init__(self, inv, water_level=60.0):
        self.inv = inv
        self.wl = water_level
        self.f = np.linspace(0.0, SR / 2.0, RESP_GRID + 1)
        self.cache = {}

    def inverse(self, seed_id, t, output):
        from obspy.signal.invsim import invert_spectrum
        resp = self.inv.get_response(seed_id, t)
        key = (seed_id, output, id(resp))
        if key not in self.cache:
            if len(self.cache) > 24:
                self.cache.clear()
            r = resp.get_evalresp_response_for_frequencies(self.f, output=output)
            r = np.asarray(r, np.complex128).copy()
            invert_spectrum(r, self.wl)
            self.cache[key] = r
        return self.cache[key]

    def on(self, f, seed_id, t, output):
        r = self.inverse(seed_id, t, output)
        return np.interp(f, self.f, r.real) + 1j * np.interp(f, self.f, r.imag)


def merge_split(st):
    """Contiguous segments of the day. Gaps stay gaps (not interpolated across hours).
    Some loggers write records whose sampling rate differs in the 7th digit
    (199.99998 vs 200.0) or with mixed dtypes, which obspy refuses to merge: round
    the rate when it is within 1e-6 of a whole number of mHz, cast to float64, and,
    if the merge still fails, keep the traces as they are (overlaps then only
    repeat the same samples, and measure() takes the max)."""
    for tr in st:
        sr = tr.stats.sampling_rate
        r = round(sr, 3)
        if sr != r and abs(sr - r) / r < 1e-6:
            tr.stats.sampling_rate = r
        tr.data = tr.data.astype(np.float64)
    try:
        st.merge(method=1)
    except Exception:
        st.sort()
    return st.split()


def process_segments(st, resp, hp, disp_band):
    """Response removal on each contiguous segment. Returns a list of
    (t0, wa_mm, disp_um) with float32 arrays at SR Hz.

    Reproduces, in one FFT per trace, the chain of route_a_wa_amplitudes.py:
    detrend/taper -> Trace.resample(SR) (obspy's default window='hann' weights the
    spectrum by cos^2(pi f / sr) even when sr == SR; this lowers WA peaks by up to
    2x on short-period sensors and is part of every published Route A amplitude) ->
    remove_response(VEL or DISP, water_level=60, pre_filt) -> WA simulation ->
    1 Hz causal high-pass / zero-phase displacement band-pass."""
    import scipy.fft as sfft
    from fractions import Fraction
    from scipy.signal import freqs as s_freqs, zpk2tf
    from obspy.signal.invsim import cosine_sac_taper
    from obspy.signal.filter import highpass, bandpass
    from route_a_wa_amplitudes import PAZ_WA

    nyq = SR / 2.0
    pre_filt = (disp_band[0] / 4.0, disp_band[0] / 2.0, 0.8 * nyq, 0.9 * nyq)
    b_wa, a_wa = zpk2tf(PAZ_WA["zeros"], PAZ_WA["poles"], PAZ_WA["gain"])
    st = merge_split(st)
    out = []
    for tr in st:
        sr = float(tr.stats.sampling_rate)
        n = tr.stats.npts
        if n / sr < MIN_SEG_S or sr < 10:
            continue
        x = tr.data.astype(np.float64)
        x -= x.mean()
        nt = int(round(TAPER_S * sr))                       # cosine taper at both ends
        w = 0.5 * (1 - np.cos(np.pi * np.arange(nt) / nt))
        x[:nt] *= w
        x[-nt:] *= w[::-1]
        ratio = Fraction(SR / sr).limit_denominator(1000)   # SR/sr = p/q
        p, q = ratio.numerator, ratio.denominator
        need = n + int(2 * PAD_S * sr)
        npad = sfft.next_fast_len(need, real=True)
        while npad % q or sfft.next_fast_len((npad * p) // q, real=True) != (npad * p) // q:
            npad = sfft.next_fast_len(npad + 1, real=True)
        nout = npad * p // q
        X = sfft.rfft(x, n=npad)
        del x
        f_in = np.arange(X.size) * (sr / npad)
        X *= 0.5 + 0.5 * np.cos(2 * np.pi * f_in / sr)       # obspy resample 'hann'
        Y = np.zeros(nout // 2 + 1, np.complex128)            # same df: just pad/cut
        m = min(Y.size, X.size)
        Y[:m] = X[:m] * (nout / npad)
        del X
        f = np.arange(Y.size) * (SR / nout)
        Y *= cosine_sac_taper(f, flimit=pre_filt)
        t = tr.stats.starttime
        V = Y * resp.on(f, tr.id, t, "VEL")
        V *= s_freqs(b_wa, a_wa, 2 * np.pi * f)[1] * PAZ_WA["sensitivity"]
        V[-1] = abs(V[-1])
        n100 = int(n * p // q)
        wa = sfft.irfft(V, n=nout)[:n100]
        del V
        D = Y * resp.on(f, tr.id, t, "DISP")
        D[-1] = abs(D[-1])
        del Y
        dsp = sfft.irfft(D, n=nout)[:n100]
        del D
        wa = highpass(wa, hp, SR, corners=4, zerophase=False)
        dsp = bandpass(dsp, disp_band[0], disp_band[1], SR, corners=4, zerophase=True)
        out.append((float(t.timestamp), (wa * 1e3).astype(np.float32),     # m -> mm
                    (dsp * 1e6).astype(np.float32)))                        # m -> um
    return out


def measure(segs, tp):
    """Running maxima, noise RMS and margins for pick times tp (epoch s)."""
    n = len(tp)
    nw = len(WINDOWS)
    wa = np.full((n, nw), np.nan, np.float32)
    dp = np.full((n, nw), np.nan, np.float32)
    wn = np.full(n, np.nan, np.float32)
    dn = np.full(n, np.nan, np.float32)
    ncomp = np.zeros(n, np.int8)
    margin = np.full(n, -np.inf, np.float32)
    ends = np.round((PRE + np.asarray(WINDOWS, float)) * SR).astype(int)
    L = ends[-1] + 1
    ar = np.arange(L)
    for t0, xw, xd in segs:
        npts = len(xw)
        t1 = t0 + (npts - 1) / SR
        inside = (tp >= t0) & (tp <= t1)
        if not inside.any():
            continue
        idx_p = np.nonzero(inside)[0]
        m = np.minimum(tp[idx_p] - t0, t1 - tp[idx_p]).astype(np.float32)
        margin[idx_p] = np.maximum(margin[idx_p], m)
        ilo = np.round((tp[idx_p] - PRE - t0) * SR).astype(np.int64)
        aw, ad = np.abs(xw), np.abs(xd)
        cw = np.concatenate([[0.0], np.cumsum(xw.astype(np.float64) ** 2)])
        cd = np.concatenate([[0.0], np.cumsum(xd.astype(np.float64) ** 2)])
        a0 = np.round((tp[idx_p] - NOISE[0] - t0) * SR).astype(np.int64)
        a1 = np.round((tp[idx_p] - NOISE[1] - t0) * SR).astype(np.int64)
        okn = (a0 >= 0) & (a1 > a0)
        for b in range(0, len(idx_p), 512):             # bounded memory
            sl = slice(b, b + 512)
            rows = idx_p[sl]
            g = ilo[sl, None] + ar[None, :]
            valid = (g >= 0) & (g < npts)
            gc_ = np.clip(g, 0, npts - 1)
            for arr, dst in ((aw, wa), (ad, dp)):
                A = np.where(valid, arr[gc_], -np.inf)
                A = np.maximum.accumulate(A, axis=1)[:, ends]
                # a window that starts before or ends after the segment is unmeasured
                A[~(valid[:, 0:1] & valid[:, ends])] = np.nan
                old = dst[rows]
                dst[rows] = np.fmax(old, A)
            got = valid[:, ends[0]] & valid[:, 0]
            ncomp[rows] += got.astype(np.int8)
        r = idx_p[okn]
        nn = (a1[okn] - a0[okn]).astype(np.float64)
        wn[r] = np.fmax(wn[r], np.sqrt((cw[a1[okn]] - cw[a0[okn]]) / nn))
        dn[r] = np.fmax(dn[r], np.sqrt((cd[a1[okn]] - cd[a0[okn]]) / nn))
    margin[~np.isfinite(margin)] = np.nan
    return wa, dp, wn, dn, ncomp, margin


TIMES = {}


def empty_frame(picks, status=""):
    out = pd.DataFrame({"pick_id": picks["pick_id"].to_numpy(),
                        "phase": picks["label"].astype(str).str.upper().to_numpy()})
    nan = np.full(len(out), np.nan, np.float32)
    for c in WA_COLS + DISP_COLS + ["wa_noise_mm", "disp_noise_um"]:
        out[c] = nan
    out["n_comp"] = np.zeros(len(out), np.int8)
    out["seg_margin_s"] = nan
    out["sensor"] = ""
    out["epoch"] = ""
    out["status"] = status
    return out


def run_task(reader, inv, net, sta, day, picks, a):
    from route_a_wa_amplitudes import epoch_id
    from obspy import UTCDateTime
    day_t0 = day * 86400.0
    out = empty_frame(picks)
    try:
        T0 = time.time()
        st, sensor, why = reader.read(net, sta, day_t0)
        TIMES['read'] = time.time() - T0
    except Exception as e:
        why, sensor, st = f"fetch:{str(e)[:60]}", "", None
    if why:
        out["status"] = why
        return out, why, sensor
    try:
        T0 = time.time()
        segs = process_segments(st, reader.resp, a.highpass, (a.disp_lo, a.disp_hi))
        TIMES['proc'] = time.time() - T0
    except Exception as e:
        why = f"resp:{str(e)[:60]}"
        out["status"] = why
        return out, why, sensor
    del st
    if not segs:
        out["status"] = "no_segment"
        return out, "no_segment", sensor
    wa, dp, wn, dn, nc, mg = measure(segs, picks["t"].to_numpy())
    out[WA_COLS] = wa
    out[DISP_COLS] = dp
    out["wa_noise_mm"] = wn
    out["disp_noise_um"] = dn
    out["n_comp"] = nc
    out["seg_margin_s"] = mg
    out["sensor"] = sensor
    out["epoch"] = epoch_id(inv, net, sta, UTCDateTime(day_t0 + 43200))
    out["status"] = np.where(nc > 0, "ok", "not_in_segment")
    return out, "ok", sensor


def cmd_run(a):
    from obspy import read_inventory
    import pyarrow as pa
    import pyarrow.parquet as pq

    sd = os.path.join(a.out, "shards")
    cd = os.path.join(a.out, "chunks")
    os.makedirs(cd, exist_ok=True)
    picks = pd.read_parquet(os.path.join(sd, f"picks_{a.shard:03d}.parquet"))
    logp = os.path.join(a.out, "logs", f"shard_{a.shard:03d}.csv")
    os.makedirs(os.path.dirname(logp), exist_ok=True)
    done = set()
    if os.path.exists(logp):
        done = set(pd.read_csv(logp, usecols=["task"])["task"])
    else:
        with open(logp, "w") as f:
            f.write("task,n_picks,n_ok,status,sensor,seconds,chunk\n")
    nchunk = len(glob.glob(os.path.join(cd, f"s{a.shard:03d}_*.parquet")))
    inv = read_inventory(a.inventory)
    reader = DayReader(inv)
    groups = picks.groupby(["network", "station", "day"], sort=False)
    todo = [(k, g) for k, g in groups
            if f"{k[0]}.{k[1]}.{pd.Timestamp(k[2], unit='D'):%Y-%m-%d}" not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"shard {a.shard}: {len(todo):,} station-days to do "
          f"({len(done):,} done)", flush=True)
    buf, logbuf = [], []

    def flush():
        nonlocal nchunk, buf, logbuf
        if not buf:
            return
        name = f"s{a.shard:03d}_{nchunk:05d}.parquet"
        tbl = pa.Table.from_pandas(pd.concat(buf, ignore_index=True), preserve_index=False)
        tmp = os.path.join(cd, name + ".tmp")
        pq.write_table(tbl, tmp, compression="zstd")
        os.replace(tmp, os.path.join(cd, name))          # chunk first, then log
        with open(logp, "a") as f:
            for r in logbuf:
                f.write(",".join(str(v) for v in r + [name]) + "\n")
        nchunk += 1
        buf, logbuf = [], []

    for (net, sta, day), g in todo:
        task = f"{net}.{sta}.{pd.Timestamp(day, unit='D'):%Y-%m-%d}"
        t = time.time()
        try:
            res, why, sensor = run_task(reader, inv, net, sta, day, g, a)
        except Exception as e:                      # never lose the shard to one day
            traceback.print_exc()
            why, sensor = f"error:{type(e).__name__}", ""
            res = empty_frame(g, why)
        res = res[DATA_COLUMNS]
        buf.append(res)
        n_ok = int((res["status"] == "ok").sum())
        why = str(why).replace(",", ";")
        logbuf.append([task, len(g), n_ok, why, sensor, f"{time.time() - t:.1f}"])
        print(f"{task} {len(g)} picks ok={n_ok} {why} {sensor} "
              f"read={TIMES.get('read', 0):.1f}s proc={TIMES.get('proc', 0):.1f}s "
              f"total={time.time() - t:.1f}s", flush=True)
        TIMES.clear()
        if len(logbuf) >= a.chunk_tasks:
            flush()
            gc.collect()
    flush()
    print(f"shard {a.shard}: finished", flush=True)


# ---------------------------------------------------------------------- consolidate
README = """# Route A amplitudes for the ver5 picks

One file per station-year, `amps_{{net}}.{{sta}}.{{year}}.parquet`, one row per pick of
`all_picks_all_regions_2010_2016_ver5.csv` (key `pick_id`; ver3 and ver4 ids unchanged).
Made by workflow/05_magnitude/route_a_day_amplitudes.py (commit {commit}).

Measurement (per station-day, one sensor: HH > BH > EH > SH > ... > HN, the first with a
response in station_inventory_v2): demean, {taper:g} s taper, resample to {sr:g} Hz, response
removal (water level 60, pre-filter {pf0:g}-{pf1:g} Hz), then
- Wood-Anderson (IASPEI, gain 2080) on ground velocity, high-pass {hp:g} Hz -> `wa_mm_*`
- displacement band-passed {d0:g}-{d1:g} Hz (zero phase) -> `disp_um_*`
Response removal runs on each contiguous segment of the day (>= {minseg:g} s), not on a
window around the pick.

| column | unit | meaning |
|---|---|---|
| pick_id | | ver5 pick id |
| phase | | P or S (picker label) |
| wa_mm_WWpWWs | mm | max abs Wood-Anderson displacement over the components, from 0.3 s before the pick to WW s after it |
| disp_um_WWpWWs | um | same for the {d0:g}-{d1:g} Hz displacement |
| wa_noise_mm, disp_noise_um | mm, um | RMS from 11 s to 1 s before the pick (max over components) |
| n_comp | | components with data over the 1 s window |
| seg_margin_s | s | time from the pick to the nearest end of its data segment (gap or day boundary); values under ~100 s may carry edge effects in disp_um_* |
| sensor | | location.band+instrument code used |
| epoch | | start date of the response epoch (per-deployment station terms) |
| status | | ok, or why there is no value (no_data, no_response:..., no_ground_motion_channel, no_segment, not_in_segment, fetch:..., resp:..., error:...) |

Windows (s; column wa_mm_01p25s is 1.25 s): {windows}. NaN where the window runs past the data.

Route A amplitude of an associated pick: take the column whose WW is the Route A post-window
(phase_window in route_a_wa_amplitudes.py: P min(1 + 0.03 r, 15, 0.8 tSP), S min(2 + 0.06 r, 60),
r = hypocentral distance in km) rounded UP to the next WW; SNR = wa_mm / wa_noise_mm.
`route_a_day_amplitudes.py select` does this and writes the CSV that route_a_build_dataset.py reads.
"""


def cmd_consolidate(a):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import subprocess

    plan = json.load(open(os.path.join(a.out, "plan.json")))
    logs = pd.concat([pd.read_csv(p) for p in
                      sorted(glob.glob(os.path.join(a.out, "logs", "shard_*.csv")))])
    tasks = pd.read_csv(os.path.join(a.out, "tasks.csv"))
    missing = set(tasks["task"]) - set(logs["task"])
    if missing and not a.allow_missing:
        sys.exit(f"{len(missing):,} station-days not done yet")
    od = os.path.join(a.out, "amps")
    os.makedirs(od, exist_ok=True)
    chunks = sorted(glob.glob(os.path.join(a.out, "chunks", "*.parquet")))
    # map chunk rows to station-years through the shard pick lists
    key = []
    for p in sorted(glob.glob(os.path.join(a.out, "shards", "picks_*.parquet"))):
        k = pd.read_parquet(p, columns=["pick_id", "network", "station", "day"])
        key.append(k)
    key = pd.concat(key, ignore_index=True)
    key["year"] = pd.to_datetime(key["day"], unit="D").dt.year.astype("int16")
    key = key.set_index("pick_id")[["network", "station", "year"]]
    parts = {}
    for p in chunks:
        d = pd.read_parquet(p)
        d = d.drop_duplicates("pick_id", keep="last")
        k = key.loc[d["pick_id"]]
        d["_f"] = (k["network"].to_numpy() + "." + k["station"].to_numpy() + "."
                   + k["year"].astype(str).to_numpy())
        for f, g in d.groupby("_f"):
            parts.setdefault(f, []).append(g.drop(columns="_f"))
    n = 0
    stat = []
    for f, gs in sorted(parts.items()):
        g = pd.concat(gs, ignore_index=True).drop_duplicates("pick_id", keep="last")
        g = g.sort_values("pick_id")
        for c in ("phase", "sensor", "epoch", "status"):
            g[c] = g[c].astype("category")
        pq.write_table(pa.Table.from_pandas(g, preserve_index=False),
                       os.path.join(od, f"amps_{f}.parquet"), compression="zstd")
        n += len(g)
        stat.append(g["status"].astype(str).value_counts().rename(f))
    st = pd.concat(stat, axis=1).fillna(0).astype(int).sum(axis=1).sort_values(ascending=False)
    try:
        commit = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = "?"
    nyq = SR / 2
    with open(os.path.join(od, "README_amps.md"), "w") as fh:
        fh.write(README.format(commit=commit, taper=TAPER_S, sr=SR, pf0=a.disp_lo / 4,
                               pf1=0.9 * nyq, hp=a.highpass, d0=a.disp_lo, d1=a.disp_hi,
                               minseg=MIN_SEG_S, windows=", ".join(f"{w:g}" for w in WINDOWS)))
    summ = [f"picks in plan : {plan['n_picks']:,}",
            f"rows written  : {n:,} in {len(parts):,} station-year files",
            f"station-days  : {len(logs):,} done of {len(tasks):,}", "status:"]
    summ += [f"  {k:<40s} {v:>12,}" for k, v in st.items()]
    open(os.path.join(od, "SUMMARY.txt"), "w").write("\n".join(summ) + "\n")
    print("\n".join(summ))
    if n != plan["n_picks"] and not a.allow_missing:
        sys.exit(f"row count {n:,} != plan {plan['n_picks']:,}")


# --------------------------------------------------------------------------- select
def window_for(phase, r_km):
    from route_a_wa_amplitudes import phase_window
    post = np.array([phase_window(p, r)[1] for p, r in zip(phase, r_km)])
    w = np.asarray(WINDOWS, float)
    i = np.searchsorted(w, post - 1e-9)          # first grid value >= post
    return np.minimum(i, len(w) - 1)


def load_amps(amps_dir, pick_ids, cols=None):
    files = sorted(glob.glob(os.path.join(amps_dir, "amps_*.parquet")))
    want = pd.Index(pd.unique(np.asarray(pick_ids)))
    out = []
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    import pyarrow as pa
    vs = pa.array(want.to_numpy())
    for f in files:
        t = pq.read_table(f, columns=cols)
        t = t.filter(pc.is_in(t["pick_id"], value_set=vs))
        if t.num_rows:
            out.append(t.to_pandas())
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)


def cmd_select(a):
    """Associated picks -> raw_wa_amplitudes CSV (route_a_wa_amplitudes.OUT_COLS).

    --arrivals needs pick_id, phase and either dist_hypo_km or (evla, evlo, evdp, stla,
    stlo, stel_m). arid, event_id, network, station are copied when present."""
    from route_a_wa_amplitudes import OUT_COLS, haversine_km
    arr = pd.read_csv(a.arrivals)
    if "dist_hypo_km" not in arr:
        arr["dist_hypo_km"] = np.hypot(
            haversine_km(arr["evla"], arr["evlo"], arr["stla"], arr["stlo"]),
            arr["evdp"] + arr["stel_m"] / 1000.0)
    arr = arr[arr["pick_id"].notna()].copy()
    arr["pick_id"] = arr["pick_id"].astype("int64")
    amps = load_amps(a.amps, arr["pick_id"])
    m = arr.merge(amps.drop(columns="phase"), on="pick_id", how="left")
    ph = m["phase"].astype(str).str.upper().to_numpy()
    iw = window_for(ph, m["dist_hypo_km"].to_numpy())
    wa = m[WA_COLS].to_numpy()[np.arange(len(m)), iw]
    dp = m[DISP_COLS].to_numpy()[np.arange(len(m)), iw]
    m["wa_amp_mm"] = wa
    m["disp_amp_um"] = dp
    m["snr"] = wa / m["wa_noise_mm"]
    m["window_s"] = np.asarray(WINDOWS)[iw]
    st = m["status"].astype(object).fillna("not_in_amps")
    m["reason"] = np.where((st == "ok") & np.isfinite(wa), "ok",
                           np.where(st == "ok", "window_outside_data", st))
    for c in OUT_COLS:
        if c not in m:
            m[c] = ""
    cols = OUT_COLS + ["pick_id", "window_s", "seg_margin_s"]
    m[cols].to_csv(a.out, index=False)
    print(f"{len(m):,} arrivals -> {a.out}")
    print(m["reason"].value_counts().head(10).to_string())


# ------------------------------------------------------------------------- validate
def cmd_validate(a):
    """Day-based amplitudes against the per-pick v2 run (arrivals_v3.csv)."""
    ref = pd.read_csv(a.arrivals, usecols=["arid", "pick_id", "phase", "distance_km",
                                           "wa_amp_mm", "disp_amp_um", "amp_snr",
                                           "amp_status", "amp_sensor", "network",
                                           "station"])
    ref = ref[(ref["amp_status"] == "ok") & ref["pick_id"].notna()]
    ref["pick_id"] = ref["pick_id"].astype("int64")
    if a.raw:   # hypocentral distances as used by the v2 run
        raw = pd.read_csv(a.raw, usecols=["arid", "dist_hypo_km"])
        ref = ref.merge(raw, on="arid", how="left")
    else:
        ref["dist_hypo_km"] = ref["distance_km"]
    amps = load_amps(a.amps, ref["pick_id"])
    m = ref.merge(amps, on="pick_id", how="inner", suffixes=("_ref", ""))
    m = m[m["status"] == "ok"]
    iw = window_for(m["phase_ref"].astype(str).str.upper().to_numpy(),
                    m["dist_hypo_km"].to_numpy())
    rows = np.arange(len(m))
    m["wa_new"] = m[WA_COLS].to_numpy()[rows, iw]
    m["disp_new"] = m[DISP_COLS].to_numpy()[rows, iw]
    m["dwa"] = np.log10(m["wa_new"] / m["wa_amp_mm"])
    m["ddisp"] = np.log10(m["disp_new"] / m["disp_amp_um"])
    m["same_sensor"] = m["sensor"].astype(str) == m["amp_sensor"].astype(str)
    q = lambda s: s.replace([np.inf, -np.inf], np.nan).dropna().quantile([.05, .25, .5, .75, .95]).round(3).tolist()
    print(f"matched picks: {len(m):,} (same sensor: {m['same_sensor'].mean():.1%})")
    for name, sub in (("all", m), ("same sensor", m[m["same_sensor"]]),
                      ("margin>=120 s", m[m["same_sensor"] & (m["seg_margin_s"] >= 120)])):
        print(f"{name:14s} n={len(sub):>8,}  log10(new/v2) WA q05..q95 {q(sub['dwa'])}"
              f"  disp {q(sub['ddisp'])}")
    if a.out:
        m.drop(columns=WA_COLS + DISP_COLS).to_csv(a.out, index=False)


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--picks", required=True, help="ver5 pick table (csv or csv.gz)")
    p.add_argument("--out", required=True)
    p.add_argument("--nshard", type=int, default=48)
    p = sub.add_parser("run")
    p.add_argument("--out", required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--inventory", default=os.path.join(HERE, "station_inventory_v2_slim.xml"))
    p.add_argument("--chunk-tasks", type=int, default=200,
                   help="station-days per output chunk (the resume granularity)")
    p.add_argument("--limit", type=int, default=0, help="at most N station-days (tests)")
    p = sub.add_parser("consolidate")
    p.add_argument("--out", required=True)
    p.add_argument("--allow-missing", action="store_true")
    p = sub.add_parser("select")
    p.add_argument("--amps", required=True, help="directory of amps_*.parquet")
    p.add_argument("--arrivals", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("validate")
    p.add_argument("--amps", required=True)
    p.add_argument("--arrivals", required=True, help="arrivals_v3.csv (Zenodo data record)")
    p.add_argument("--raw", default=None, help="raw_wa_amplitudes_v2.csv for dist_hypo_km")
    p.add_argument("--out", default=None)
    for name in ("run", "consolidate"):
        q = sub._name_parser_map[name]
        q.add_argument("--highpass", type=float, default=1.0)
        q.add_argument("--disp-lo", type=float, default=0.05)
        q.add_argument("--disp-hi", type=float, default=2.0)
    a = ap.parse_args(argv)
    {"plan": cmd_plan, "run": cmd_run, "consolidate": cmd_consolidate,
     "select": cmd_select, "validate": cmd_validate}[a.cmd](a)


if __name__ == "__main__":
    main()

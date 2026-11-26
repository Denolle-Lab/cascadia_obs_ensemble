"""Stage 2b of the all-family CC differential times (D-B093): NC/BK snippets from NCEDC FDSN.

NC and BK are not in pnwstore; the picker read them from NCEDC (utils/data_client.py routes NC/BK to
obspy Client("NCEDC")). One bulk dataselect request per event (all its NC/BK stations, channels HH?,BH?,EH?,
[first pick - 10 s, last pick + 10 s]); per pick the snippet [pick - 8 s, pick + 8 s] of the channel chosen by the
picker rule (HH > BH > EH, first band with a vertical; one location code per station-band: the first in sort
order that has the vertical). Time base: the record start times as served (NC/BK are GPS-timed permanent
stations; no header-vs-count anchoring, anchor_shift_ms = 0, hdr_dev_max_ms empty). No response removal (both
events of a pair use the same channel).

Output: <SNIP_NC>/<ID // 1000>/<ID>.{P,S}.mseed, <ID>.meta.csv (same layout as the pnwstore cache; meta written
last). No-data answers are written (status no_data); network errors are not (the event is retried on a rerun).

  python b5_ccall_ncedc.py [--threads 6] [--families ...]
"""
import argparse
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import Stream, UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b5_ccall_common as A  # noqa: E402
import b5_cc_common as C  # noqa: E402

_tl = threading.local()


def client():
    if not hasattr(_tl, "c"):
        _tl.c = Client("NCEDC", timeout=120)
    return _tl.c


def fetch(bulk, tries=5):
    for k in range(tries):
        try:
            return client().get_waveforms_bulk(bulk)
        except FDSNNoDataException:
            return Stream()
        except Exception as e:  # network / server: back off and retry
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))
            _tl.__dict__.pop("c", None)


def event_ncedc(eid, picks):
    d = A.snip_dir(A.SNIP_NC, eid)
    fmeta = d / f"{eid}.meta.csv"
    if fmeta.exists():
        return eid, "skip", 0
    t_start = time.time()
    bulk = []
    for sta, g in picks.groupby("station"):
        S, N = sta.split(".")
        bulk.append((N, S, "*", "HH?,BH?,EH?", UTCDateTime(g.t_pick.min() - C.SNIP_PRE - 2),
                     UTCDateTime(g.t_pick.max() + C.SNIP_POST + 2)))
    st = fetch(bulk)
    rows_out = []
    streams = {"P": Stream(), "S": Stream()}
    for sta, g in picks.groupby("station"):
        S, N = sta.split(".")
        base = dict(ID=eid, station=sta, source="ncedc")
        s1 = st.select(network=N, station=S)
        chans = pd.DataFrame({"channel": sorted({tr.stats.channel for tr in s1})})
        ch = C.choose_band(chans) if len(chans) else None
        if ch is None:
            for _, pk in g.iterrows():
                rows_out.append({**base, "phase": pk.phase, "status": "no_data"})
            continue
        band, zc, hz = ch
        locs = sorted({tr.stats.location for tr in s1.select(channel=zc)})
        loc = locs[0]
        s1 = s1.select(location=loc)
        hz = [c for c in hz if len(s1.select(channel=c))]
        for _, pk in g.iterrows():
            for cha in ([zc] if pk.phase == "P" else [zc] + hz):
                rec = {**base, "phase": pk.phase, "band": band, "channel": cha, "comp": "Z" if cha == zc else "H",
                       "n_horiz": len(hz), "location": loc}
                tp = pk.t_pick
                cand = [tr for tr in s1.select(channel=cha)
                        if tr.stats.starttime.timestamp <= tp <= tr.stats.endtime.timestamp]
                if not cand:
                    rec["status"] = "pick_in_gap"
                    rows_out.append(rec)
                    continue
                tr = cand[0].copy()
                sr = tr.stats.sampling_rate
                t_lo, t_hi = tp - C.SNIP_PRE, tp + C.SNIP_POST
                tr.trim(UTCDateTime(t_lo), UTCDateTime(t_hi))
                rec.update(status="ok", anchor_shift_ms=0.0, rate=sr, n_traces=len(s1.select(channel=cha)),
                           npts_read=int(tr.stats.npts), t_start=tr.stats.starttime.timestamp,
                           t_end=tr.stats.endtime.timestamp)
                rec["covers"] = bool(rec["t_start"] <= t_lo + 1.0 / sr and rec["t_end"] >= t_hi - 1.0 / sr)
                tr.data = tr.data.astype(np.float32) if tr.data.dtype.kind == "f" else tr.data.astype(np.int32)
                rec["trace_id"] = tr.id
                streams[pk.phase] += tr
                rows_out.append(rec)
    d.mkdir(parents=True, exist_ok=True)
    for ph, s in streams.items():
        if not len(s):
            continue
        f = d / f"{eid}.{ph}.mseed"
        for tr in s:
            tr.stats.pop("mseed", None)
        tmp = f.with_suffix(".tmp")
        ints = Stream([tr for tr in s if tr.data.dtype.kind != "f"])
        flts = Stream([tr for tr in s if tr.data.dtype.kind == "f"])
        with open(tmp, "wb") as fh:
            if len(ints):
                ints.write(fh, format="MSEED", encoding="STEIM2", reclen=4096)
            if len(flts):
                flts.write(fh, format="MSEED", encoding="FLOAT32", reclen=4096)
        os.replace(tmp, f)
    m = pd.DataFrame(rows_out)
    m["read_s"] = time.time() - t_start
    tmp = fmeta.with_suffix(".tmp")
    m.to_csv(tmp, index=False)
    os.replace(tmp, fmeta)
    return eid, "done", len(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--families", default="")
    a = ap.parse_args()
    logf = A.OUT / "logs" / "ncedc.log"
    logf.parent.mkdir(parents=True, exist_ok=True)
    ev = pd.read_parquet(A.TASKS / "events.parquet", columns=["ID", "T0", "family_id"])
    if a.families:
        ev = ev[ev.family_id.isin([int(x) for x in a.families.split(",")])]
    pk = pd.read_parquet(A.TASKS / "picks.parquet", columns=["ID", "station", "net", "phase", "t_hdr"])
    pk = pk[pk.ID.isin(ev.ID) & pk.net.isin(A.NCEDC_NETS)].rename(columns={"t_hdr": "t_pick"})
    order = ev.set_index("ID").T0
    ids = sorted(pk.ID.unique(), key=lambda i: order[i])
    todo = [i for i in ids if not (A.snip_dir(A.SNIP_NC, i) / f"{i}.meta.csv").exists()]
    A.log(f"ncedc: {len(ids)} events with NC/BK picks ({len(pk)} picks), {len(todo)} to fetch, {a.threads} threads",
          logf)
    A.write_state("ncedc", events=len(ids), picks=len(pk), todo_at_start=len(todo), done=False)
    groups = dict(tuple(pk.groupby("ID")))
    t0 = time.time()
    n = nerr = 0
    with ThreadPoolExecutor(a.threads) as ex:
        futs = {ex.submit(event_ncedc, int(i), groups[i]): i for i in todo}
        for f in as_completed(futs):
            n += 1
            try:
                f.result()
            except Exception:
                nerr += 1
                if nerr <= 20:
                    traceback.print_exc()
            if n % 500 == 0 or n == len(todo):
                rate = n / (time.time() - t0)
                A.log(f"{n}/{len(todo)} events, {nerr} errors, {rate:.2f} ev/s, ETA "
                      f"{(len(todo) - n) / max(rate, 1e-9) / 3600:.2f} h", logf)
                A.write_state("ncedc", fetched=n, errors=nerr, rate_ev_s=round(rate, 3))
    missing = [i for i in ids if not (A.snip_dir(A.SNIP_NC, i) / f"{i}.meta.csv").exists()]
    A.write_state("ncedc", done=not missing, missing=len(missing), errors=nerr, wall_s=round(time.time() - t0))
    A.log(f"ncedc finished: {n} fetched, {nerr} errors, {len(missing)} missing, {time.time() - t0:.0f} s", logf)


if __name__ == "__main__":
    main()

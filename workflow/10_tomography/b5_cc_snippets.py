"""Step 2 of the CC differential times: waveform snippets around every pick of the set events (cache).

For each event of a set and each of its picks: read [pick - 8 s, pick + 8 s] from pnwstore with short reads on
the record-header time base (b5_cc_common.read_channel). P: vertical of the picker's band (HH > BH > EH);
S: vertical and the horizontals of that band. No response removal (same channel for both events of a pair).

Output (cache, so a rerun does not re-read): /wd1/mdenolle_data/tomo_v1/cc_dt/snippets/<set>/<ID>.P.mseed,
<ID>.S.mseed (one trace per station-component, raw counts, start shifted to the pick record's header time) and
<ID>.meta.csv (one row per pick-channel: status, band, rate, anchor shift, header deviation, coverage, bytes).
The meta file is written last: an event with a meta file is done (resumable).

  python b5_cc_snippets.py --set msh [--workers 12] [--sample-pairs N --seed 0] [--ids 1,2,3]
  python b5_cc_snippets.py --set msh --band-map snippets/msh/band_map.json --redo-band-map   (consistent band)

Band per station (--band-map, from b5_cc_bandmap.py): the band with a vertical at the most picks of that station in
the set, so both events of a pair use the same channel (XD: BH at every pick instead of HH on some days).
"""
import argparse
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import Stream, UTCDateTime

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b5_cc_common as C  # noqa: E402


def event_snippets(args):
    name, eid, picks = args[:3]
    band_map = args[3] if len(args) > 3 else {}
    d = C.SNIP / name
    fmeta = d / f"{eid}.meta.csv"
    if fmeta.exists():
        return eid, "skip", 0
    t_start = time.time()
    b0 = C.FC.bytes_read
    rows_out = []
    streams = {"P": Stream(), "S": Stream()}
    for sta, g in picks.groupby("station"):
        S, N = sta.split(".")
        base = dict(ID=eid, station=sta)
        if N in C.NO_PNWSTORE:
            for _, pk in g.iterrows():
                rows_out.append({**base, "phase": pk.phase, "status": "no_pnwstore"})
            continue
        tlo = UTCDateTime(g.t_pick.min() - C.SNIP_PRE - 2)
        thi = UTCDateTime(g.t_pick.max() + C.SNIP_POST + 2)
        try:
            rows = C.index_rows(N, S, tlo, thi)
        except Exception as e:
            rows = pd.DataFrame()
            err = f"index_error:{type(e).__name__}"
        else:
            err = None
        ch = C.choose_band(rows, band_map.get(sta)) if len(rows) else None
        if ch is None:
            for _, pk in g.iterrows():
                rows_out.append({**base, "phase": pk.phase, "status": err or "no_data"})
            continue
        band, zc, hz = ch
        for _, pk in g.iterrows():
            chans = [zc] if pk.phase == "P" else [zc] + hz
            for cha in chans:
                rec = {**base, "phase": pk.phase, "band": band, "channel": cha, "comp": "Z" if cha == zc else "H",
                       "n_horiz": len(hz)}
                try:
                    tr, info = C.read_channel(rows, cha, pk.t_pick, pk.t_pick - C.SNIP_PRE, pk.t_pick + C.SNIP_POST)
                except OSError:
                    raise   # archive I/O problem: no meta file, the event is retried on the next run
                except Exception as e:
                    tr, info = None, dict(status=f"error:{type(e).__name__}:{str(e)[:80]}")
                rec.update(info)
                if tr is not None:
                    if tr.data.dtype.kind == "f":
                        tr.data = tr.data.astype(np.float32)
                    else:
                        tr.data = tr.data.astype(np.int32)
                    rec["trace_id"] = tr.id
                    streams[pk.phase] += tr
                rows_out.append(rec)
    d.mkdir(parents=True, exist_ok=True)
    for ph, st in streams.items():
        f = d / f"{eid}.{ph}.mseed"
        if len(st):
            for tr in st:
                tr.stats.pop("mseed", None)
            tmp = f.with_suffix(".tmp")
            enc = {tr.id: ("FLOAT32" if tr.data.dtype.kind == "f" else "STEIM2") for tr in st}
            # write per trace encoding: obspy takes one encoding per call -> write float and int separately
            ints = Stream([tr for tr in st if enc[tr.id] == "STEIM2"])
            flts = Stream([tr for tr in st if enc[tr.id] == "FLOAT32"])
            with open(tmp, "wb") as fh:
                if len(ints):
                    ints.write(fh, format="MSEED", encoding="STEIM2", reclen=4096)
                if len(flts):
                    flts.write(fh, format="MSEED", encoding="FLOAT32", reclen=4096)
            os.replace(tmp, f)
    m = pd.DataFrame(rows_out)
    m["read_s"] = time.time() - t_start
    m["bytes_read_event"] = C.FC.bytes_read - b0
    tmp = fmeta.with_suffix(".tmp")
    m.to_csv(tmp, index=False)
    os.replace(tmp, fmeta)
    return eid, "done", len(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--sample-pairs", type=int, default=0, help="only the events of N random given pairs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ids", default="")
    ap.add_argument("--band-map", default="", help="json {station: band}: one band per station for the set")
    ap.add_argument("--redo-band-map", action="store_true",
                    help="delete (then re-read) the cached events whose band differs from --band-map")
    a = ap.parse_args()
    band_map = json.loads(Path(a.band_map).read_text()) if a.band_map else {}
    out = C.OUT / a.set
    picks = pd.read_parquet(out / "picks.parquet")
    if a.ids:
        ids = [int(x) for x in a.ids.split(",")]
    elif a.sample_pairs:
        pr = pd.read_parquet(out / "pairs.parquet")
        pr = pr[pr.source == "given"].sample(a.sample_pairs, random_state=a.seed)
        ids = sorted(set(pr.ID1) | set(pr.ID2))
    else:
        ids = sorted(picks.ID.unique())
    if a.redo_band_map and band_map:
        n_redo = 0
        for i in ids:
            f = C.SNIP / a.set / f"{i}.meta.csv"
            if not f.exists():
                continue
            m = pd.read_csv(f)
            m = m[(m.comp == "Z") & m.station.isin(list(band_map))]
            if (m.band != m.station.map(band_map)).any():
                for x in (f, f.with_name(f"{i}.P.mseed"), f.with_name(f"{i}.S.mseed")):
                    x.unlink(missing_ok=True)
                n_redo += 1
        print(f"{n_redo} events re-read for the band map", flush=True)
    tasks = [(a.set, int(i), picks[picks.ID == i], band_map) for i in ids]
    (C.SNIP / a.set).mkdir(parents=True, exist_ok=True)
    print(f"{time.strftime('%H:%M:%S')} {a.set}: {len(tasks)} events, {a.workers} workers", flush=True)
    t1 = time.time()
    n = 0
    with Pool(a.workers) as pool:
        for eid, st, k in pool.imap_unordered(_safe, tasks, chunksize=1):
            n += 1
            if n % 50 == 0 or n == len(tasks):
                print(f"{time.strftime('%H:%M:%S')} {n}/{len(tasks)} events {time.time() - t1:.0f} s", flush=True)
    if not a.ids and not a.sample_pairs:
        done = all((C.SNIP / a.set / f"{i}.meta.csv").exists() for i in ids)
        if done:
            (C.SNIP / a.set / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S\n"))
            print("snippets done", flush=True)


def _safe(t):
    try:
        return event_snippets(t)
    except Exception:
        traceback.print_exc()
        return t[1], "error", 0


if __name__ == "__main__":
    main()

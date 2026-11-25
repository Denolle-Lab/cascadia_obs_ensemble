"""Stage 2 of the all-family CC differential times (D-B093): pnwstore waveform snippets of every scoped event.

Same reader as the D-B078 sets (b5_cc_snippets.event_snippets, b5_cc_common.read_channel): for each pick the
MiniSEED records overlapping [pick - 8 s, pick + 8 s], on the record-header time base; picker band rule
HH > BH > EH; P on the vertical, S on the vertical and the two horizontals. The pick time given to the reader is
the header-time pick t_hdr (= the v4.1t time; v4.1tc minus its clock correction). NC/BK picks are left to
b5_ccall_ncedc.py (meta status no_pnwstore here).

Differences from b5_cc_snippets: events are processed in origin-time order, in chunks, and the per-process byte
cache serves every read from 256 kB aligned blocks (LRU), so events close in time share reads of the same day
files. The bytes returned to the reader are identical. The D-B078 cache (cc_dt/snippets/<set>/) is used as is
(read only) for the events it holds.

Concurrency: in-flight chunks start at --workers (24) and rise by 4 up to --max-workers (48) while MemAvailable
stays above 60 GB; they drop when it falls below 30 GB, and no new chunk starts below 22 GB.

  python b5_ccall_snippets.py [--workers 24 --max-workers 48] [--ids-from tasks/events.parquet] [--limit N]
Output: <SNIP>/<ID // 1000>/<ID>.{P,S}.mseed, <ID>.meta.csv (meta written last = event done).
"""
import argparse
import os
import sys
import time
import traceback
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b5_ccall_common as A  # noqa: E402
import b5_cc_common as C  # noqa: E402
import b5_cc_snippets as CS  # noqa: E402

BLOCK = 256 << 10


class AlignedCache:
    """Byte-range reads served from BLOCK-aligned blocks of the file (LRU, maxbytes per process)."""

    def __init__(self, maxbytes=96 << 20):
        self.d = OrderedDict()
        self.size = 0
        self.max = maxbytes
        self.nread = 0
        self.bytes_read = 0

    def _block(self, fn, k):
        key = (fn, k)
        if key in self.d:
            self.d.move_to_end(key)
            return self.d[key]
        with open(fn, "rb") as f:
            f.seek(k * BLOCK)
            buf = f.read(BLOCK)
        self.nread += 1
        self.bytes_read += len(buf)
        self.d[key] = buf
        self.size += len(buf)
        while self.size > self.max and len(self.d) > 1:
            _, b = self.d.popitem(last=False)
            self.size -= len(b)
        return buf

    def read(self, fn, off, nb):
        if nb <= 0:
            return b""
        k0, k1 = off // BLOCK, (off + nb - 1) // BLOCK
        parts = [self._block(fn, k) for k in range(k0, k1 + 1)]
        buf = b"".join(parts) if len(parts) > 1 else parts[0]
        a = off - k0 * BLOCK
        return buf[a:a + nb]


def old_cache_index():
    """{ID: folder} of the D-B078 snippet cache (complete events only: meta file present)."""
    out = {}
    for s in ("axial", "blanco", "msh", "jdf"):
        d = A.SNIP_OLD / s
        if not d.exists():
            continue
        for f in d.glob("*.meta.csv"):
            out[int(f.name.split(".")[0])] = str(d)
    return out


_parse_headers = C.parse_headers


def safe_parse_headers(buf):
    """C.parse_headers, but a chunk that does not start on a record boundary and happens to look like one (a
    garbage date: ValueError in the year/day parse, seen on 7D FS*B 2012-13 day files with mixed record sizes)
    returns no record, so records_window falls back to reading the whole index block (its existing rule for a
    chunk that is not on a record boundary). The D-B078 reader raised and the channel got status error."""
    try:
        return _parse_headers(buf)
    except ValueError:
        return pd.DataFrame(columns=["offset", "reclen", "t", "npts", "rate", "cha"])


def init_worker():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    C.FC = AlignedCache()
    C.parse_headers = safe_parse_headers
    C.SNIP = A.SNIP          # event_snippets writes to C.SNIP / <name> / <ID>.*; name = shard


def run_chunk(chunk):
    out = []
    for eid, picks in chunk:
        try:
            out.append(CS.event_snippets((A.shard(eid), eid, picks, {})))
        except Exception:
            traceback.print_exc()
            out.append((eid, "error", 0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--max-workers", type=int, default=48)
    ap.add_argument("--chunk", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--families", default="", help="only the events of these families (test)")
    a = ap.parse_args()
    logf = A.OUT / "logs" / "snippets.log"
    logf.parent.mkdir(parents=True, exist_ok=True)
    ev = pd.read_parquet(A.TASKS / "events.parquet", columns=["ID", "T0", "family_id"])
    if a.families:
        ev = ev[ev.family_id.isin([int(x) for x in a.families.split(",")])]
    pk = pd.read_parquet(A.TASKS / "picks.parquet", columns=["ID", "station", "net", "phase", "t_hdr"])
    pk = pk[pk.ID.isin(ev.ID)].rename(columns={"t_hdr": "t_pick"})
    old = old_cache_index()
    ev = ev.sort_values("T0")
    todo = [i for i in ev.ID if i not in old and not (A.snip_dir(A.SNIP, i) / f"{i}.meta.csv").exists()]
    if a.limit:
        todo = todo[:a.limit]
    n_old = int(ev.ID.isin(list(old)).sum())
    A.log(f"snippets: {len(ev)} events, {n_old} in the D-B078 cache, {len(ev) - n_old - len(todo)} already done, "
          f"{len(todo)} to read; workers {a.workers}..{a.max_workers}", logf)
    A.write_state("snippets", events=len(ev), in_old_cache=n_old, todo_at_start=len(todo), done=False)
    groups = dict(tuple(pk.groupby("ID")))
    empty = pk.iloc[:0]
    chunks = [[(int(i), groups.get(i, empty)) for i in todo[k:k + a.chunk]] for k in range(0, len(todo), a.chunk)]
    for i in set(todo):
        A.snip_dir(A.SNIP, i).mkdir(parents=True, exist_ok=True)
    W = a.workers
    t0 = time.time()
    n_ev = n_err = 0
    last_adj = time.time()
    high_since = None
    with ProcessPoolExecutor(max_workers=a.max_workers, initializer=init_worker) as ex:
        it = iter(chunks)
        running = set()
        exhausted = False
        while running or not exhausted:
            mem = A.mem_available_gb()
            now = time.time()
            if mem > 60:
                high_since = high_since or now
            else:
                high_since = None
            if now - last_adj > 60:
                if high_since and now - high_since > 120 and W < a.max_workers:
                    W = min(a.max_workers, W + 4)
                    A.log(f"MemAvailable {mem:.0f} GB > 60 for 2 min: in-flight -> {W}", logf)
                    last_adj = now
                elif mem < 30 and W > 8:
                    W = max(8, W - 4)
                    A.log(f"MemAvailable {mem:.0f} GB < 30: in-flight -> {W}", logf)
                    last_adj = now
            while not exhausted and len(running) < W and A.mem_available_gb() > 22:
                c = next(it, None)
                if c is None:
                    exhausted = True
                    break
                running.add(ex.submit(run_chunk, c))
            if not running:
                time.sleep(5)
                continue
            done, running = wait(running, timeout=30, return_when=FIRST_COMPLETED)
            running = set(running)
            for f in done:
                for eid, st, k in f.result():
                    n_ev += 1
                    n_err += st == "error"
            if done and (n_ev % 600 < len(done) * a.chunk or not running):
                el = time.time() - t0
                rate = n_ev / el if el > 0 else 0
                eta = (len(todo) - n_ev) / rate / 3600 if rate > 0 else float("nan")
                A.log(f"{n_ev}/{len(todo)} events, {n_err} errors, {rate:.2f} ev/s, in-flight {W}, "
                      f"MemAvailable {mem:.0f} GB, ETA {eta:.2f} h", logf)
                A.write_state("snippets", read=n_ev, errors=n_err, rate_ev_s=round(rate, 3), eta_h=round(eta, 2),
                              in_flight=W)
    missing = [i for i in ev.ID if i not in old and not (A.snip_dir(A.SNIP, i) / f"{i}.meta.csv").exists()]
    A.write_state("snippets", done=not missing, missing=len(missing), read=n_ev, errors=n_err,
                  wall_s=round(time.time() - t0), max_in_flight=W)
    A.log(f"snippets finished: {n_ev} read, {n_err} errors, {len(missing)} missing, {time.time() - t0:.0f} s", logf)


if __name__ == "__main__":
    main()

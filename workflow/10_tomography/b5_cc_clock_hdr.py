"""Station clock check, part 2: MiniSEED record headers of a station over a date range (pnwstore, read only).

For every index row of the station's vertical channel(s) and every day, reads only the first 64 bytes of each record
(fixed record length from blockette 1000) and summarizes per day: records, clock-locked flag (I/O flags bit 5),
timing quality (blockette 1001), time-correction field and its 'applied' bit, and time tears (record start minus the
end of the previous record, > half a sample): a re-lock of a free-running clock shows as a tear of the accumulated
offset. Path mapping: b5_cc_common.real_path (pnwstore year -> disk).

  python b5_cc_clock_hdr.py --station MH07.XD --start 2015-10-01 --end 2016-07-01 [--channels BHZ,HHZ]
  -> /wd1/mdenolle_data/tomo_v1/cc_dt/clock/hdr_<STA>.csv (per day and channel), hdr_<STA>_tears.csv
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import UTCDateTime

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b5_cc_common as C  # noqa: E402


def heads(fn, off, nbytes):
    """First 64 bytes of each record of a byte range (fixed record length from the first record's B1000)."""
    with open(fn, "rb") as f:
        f.seek(off)
        h0 = f.read(64)
        rl = C._b1000_len(np.frombuffer(h0, np.uint8), 0)
        if not rl:
            return None, None
        n = nbytes // rl
        out = np.zeros((n, 64), np.uint8)
        for k in range(n):
            f.seek(off + k * rl)
            b = f.read(64)
            out[k, :len(b)] = np.frombuffer(b, np.uint8)
    return out, rl


def parse_heads(H):
    be_year = (H[:, 20].astype(int) << 8) | H[:, 21]
    big = (be_year > 1900) & (be_year < 2100)

    def u16(i):
        hi = np.where(big, H[:, i], H[:, i + 1]).astype(np.int64)
        lo = np.where(big, H[:, i + 1], H[:, i]).astype(np.int64)
        return (hi << 8) | lo

    def i16(i):
        v = u16(i)
        return np.where(v >= 32768, v - 65536, v)

    def i32(i):
        a = [H[:, i + k].astype(np.int64) for k in range(4)]
        v = np.where(big, (a[0] << 24) | (a[1] << 16) | (a[2] << 8) | a[3], (a[3] << 24) | (a[2] << 16) | (a[1] << 8) | a[0])
        return np.where(v >= 2 ** 31, v - 2 ** 32, v)

    year, doy = u16(20), u16(22)
    base = (pd.to_datetime((year * 1000 + doy).astype(str), format="%Y%j").values.astype("datetime64[s]").astype(np.int64))
    sec = base + H[:, 24].astype(np.int64) * 3600 + H[:, 25].astype(np.int64) * 60 + H[:, 26].astype(np.int64)
    rate = C._rate(i16(32), i16(34))
    act, io = H[:, 36], H[:, 37]
    tcorr = i32(40)
    bo = u16(46)
    tq = np.full(len(H), -1)
    usec = np.zeros(len(H))
    cur = bo.copy()
    for _ in range(3):
        ok = (cur > 0) & (cur + 8 <= 64)
        if not ok.any():
            break
        idx = np.where(ok)[0]
        p = cur[idx]
        bt = np.where(big[idx], (H[idx, p].astype(int) << 8) | H[idx, p + 1], (H[idx, p + 1].astype(int) << 8) | H[idx, p])
        nx = np.where(big[idx], (H[idx, p + 2].astype(int) << 8) | H[idx, p + 3], (H[idx, p + 3].astype(int) << 8) | H[idx, p + 2])
        m = bt == 1001
        tq[idx[m]] = H[idx[m], p[m] + 4]
        usec[idx[m]] = H[idx[m], p[m] + 5].astype(np.int8)
        new = np.zeros_like(cur)
        new[idx] = nx
        cur = new
    applied = (act & 0x02) != 0
    t = sec + u16(28) * 1e-4 + usec * 1e-6 + np.where(applied, 0, tcorr) * 1e-4
    return pd.DataFrame({"t": t, "npts": u16(30), "rate": rate, "locked": (io & 0x20) != 0, "tq": tq,
                         "tcorr_s": tcorr * 1e-4, "corr_applied": applied,
                         "cha": [bytes(r[15:18]).decode("ascii", "replace").strip() for r in H]})


def day_task(args):
    net, sta, day, chans = args
    t0 = UTCDateTime(day)
    rows = C.index_rows(net, sta, t0 + 60, t0 + 86340)
    out, tears = [], []
    if len(rows) == 0:
        return out, tears
    for cha in chans:
        rr = rows[rows.channel == cha]
        hs = []
        for r in rr.itertuples():
            H, rl = heads(C.real_path(r.filename), int(r.byteoffset), int(r.bytes))
            if H is None:
                continue
            h = parse_heads(H)
            hs.append(h[h.cha == cha])
        if not hs:
            continue
        h = pd.concat(hs).sort_values("t").reset_index(drop=True)
        h = h[(h.t >= t0.timestamp - 3600) & (h.t < t0.timestamp + 90000)]
        end = h.t.values + h.npts.values / np.where(h.rate.values > 0, h.rate.values, np.nan)
        gap = np.r_[np.nan, h.t.values[1:] - end[:-1]]
        sr = float(np.nanmedian(h.rate))
        tear = np.abs(gap) > 0.5 / sr
        for k in np.where(tear)[0]:
            tears.append(dict(station=f"{sta}.{net}", channel=cha, time=UTCDateTime(h.t.values[k]).isoformat(),
                              tear_s=float(gap[k]), locked_before=bool(h.locked.values[k - 1]),
                              locked_after=bool(h.locked.values[k]), tq_before=int(h.tq.values[k - 1]),
                              tq_after=int(h.tq.values[k])))
        out.append(dict(station=f"{sta}.{net}", channel=cha, day=day, records=len(h), rate=sr,
                        locked_frac=float(h.locked.mean()), tq_median=float(np.median(h.tq)), tq_min=int(h.tq.min()),
                        tq_max=int(h.tq.max()), tcorr_nonzero=int((h.tcorr_s != 0).sum()),
                        tcorr_median_s=float(np.median(h.tcorr_s)), corr_applied_frac=float(h.corr_applied.mean()),
                        n_tears=int(tear.sum()), tear_sum_s=float(np.nansum(np.where(tear, gap, 0)))))
    return out, tears


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--channels", default="BHZ,HHZ")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    sta, net = a.station.split(".")
    days = [d.strftime("%Y-%m-%d") for d in pd.date_range(a.start, a.end, freq="D", inclusive="left")]
    with Pool(a.workers) as p:
        res = p.map(day_task, [(net, sta, d, a.channels.split(",")) for d in days])
    out = pd.DataFrame([x for r in res for x in r[0]])
    tears = pd.DataFrame([x for r in res for x in r[1]])
    od = C.OUT / "clock"
    od.mkdir(parents=True, exist_ok=True)
    out.to_csv(od / f"hdr_{a.station}.csv", index=False)
    tears.to_csv(od / f"hdr_{a.station}_tears.csv", index=False)
    print(f"{a.station}: {len(out)} channel-days, {len(tears)} time tears", flush=True)


if __name__ == "__main__":
    main()

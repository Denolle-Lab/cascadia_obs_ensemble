"""MiniSEED (v2) fixed-header parsing without decoding data, and libmseed-style segment joining.

Ported from the D-B072 pick re-timing (prt_common.parse_headers, segments) and the clock-epoch header scan
(b5_cc_clock_hdr.parse_heads), merged into one vectorised parser that needs only the first 64 bytes of each
record. Start times follow libmseed: BTIME + blockette-1001 microseconds + the time-correction field when the
'time correction applied' activity bit is not set.

Columns of a header table: offset, reclen, net, sta, loc, cha, start (POSIX float s), npts, rate,
tq (blockette-1001 timing quality, -1 if absent), locked (I/O flag bit 5: clock locked), tcorr_s,
corr_applied, act, io, dq.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rate(f, m):
    f = f.astype(float)
    m = m.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((f > 0) & (m > 0), f * m,
               np.where((f > 0) & (m < 0), -f / np.where(m == 0, 1, m),
               np.where((f < 0) & (m > 0), -m / np.where(f == 0, 1, f),
               np.where((f < 0) & (m < 0), 1.0 / (f * m), 0.0))))


def b1000_reclen(b: np.ndarray, off: int = 0):
    """Record length from blockette 1000 of the record at `off` (None if not found in the first 64 bytes+)."""
    n = len(b)
    if off + 48 > n:
        return None
    be = 1900 < ((int(b[off + 20]) << 8) | int(b[off + 21])) < 2100
    def u16(i):
        return (int(b[i]) << 8 | int(b[i + 1])) if be else (int(b[i + 1]) << 8 | int(b[i]))
    nb = int(b[off + 39])
    bo = u16(off + 46)
    for _ in range(nb + 1):
        if bo == 0 or off + bo + 8 > n:
            break
        btype = u16(off + bo)
        if btype == 1000:
            return 2 ** int(b[off + bo + 6])
        bo = u16(off + bo + 2)
    return None


def record_offsets(buf: bytes) -> np.ndarray:
    """Offsets of the records in a byte buffer (fixed length fast path, else walk by blockette 1000)."""
    b = np.frombuffer(buf, dtype=np.uint8)
    n = len(b)
    if n < 48:
        return np.zeros(0, int)
    rl0 = b1000_reclen(b, 0)
    if rl0 and n % rl0 == 0:
        cand = np.arange(0, n, rl0)
        if all(b1000_reclen(b, int(o)) == rl0 for o in cand[:: max(1, len(cand) // 50)]):
            return cand
    offs, o = [], 0
    while o + 48 <= n:
        rl = b1000_reclen(b, o)
        if not rl:
            raise ValueError(f"no blockette 1000 at offset {o}")
        offs.append(o)
        o += rl
    return np.array(offs, int)


def parse_heads(H: np.ndarray, offsets=None) -> pd.DataFrame:
    """Parse an (n, 64) uint8 array of record heads."""
    H = np.asarray(H, dtype=np.uint8)
    nrec = len(H)
    if nrec == 0:
        return pd.DataFrame(columns=["offset", "net", "sta", "loc", "cha", "start", "npts", "rate", "tq", "locked",
                                     "tcorr_s", "corr_applied", "act", "io", "dq", "reclen"])
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
        v = np.where(big, (a[0] << 24) | (a[1] << 16) | (a[2] << 8) | a[3],
                     (a[3] << 24) | (a[2] << 16) | (a[1] << 8) | a[0])
        return np.where(v >= 2 ** 31, v - 2 ** 32, v)

    year, doy = u16(20), u16(22)
    ok = (year > 1900) & (year < 2100) & (doy >= 1) & (doy <= 366)
    yd = np.where(ok, year * 1000 + doy, 1970001)
    base = (pd.to_datetime(yd.astype(str), format="%Y%j").values.astype("datetime64[s]").astype(np.int64))
    sec = base + H[:, 24].astype(np.int64) * 3600 + H[:, 25].astype(np.int64) * 60 + H[:, 26].astype(np.int64)
    fract = u16(28)
    rate = _rate(i16(32), i16(34))
    act, io_, dq = H[:, 36], H[:, 37], H[:, 38]
    tcorr = i32(40)
    bo = u16(46)
    tq = np.full(nrec, -1)
    usec = np.zeros(nrec)
    rate100 = np.full(nrec, np.nan)
    reclen = np.zeros(nrec, int)
    cur = bo.copy()
    rows = np.arange(nrec)
    for _ in range(4):
        valid = (cur > 0) & (cur + 4 <= 64)
        if not valid.any():
            break
        idx = rows[valid]
        p = cur[idx]
        bt = np.where(big[idx], (H[idx, p].astype(int) << 8) | H[idx, p + 1], (H[idx, p + 1].astype(int) << 8) | H[idx, p])
        nx = np.where(big[idx], (H[idx, p + 2].astype(int) << 8) | H[idx, p + 3], (H[idx, p + 3].astype(int) << 8) | H[idx, p + 2])
        m = (bt == 1001) & (p + 6 <= 64)
        tq[idx[m]] = H[idx[m], p[m] + 4]
        usec[idx[m]] = H[idx[m], p[m] + 5].astype(np.int8)
        m = (bt == 1000) & (p + 7 <= 64)
        reclen[idx[m]] = 2 ** H[idx[m], p[m] + 6].astype(int)
        m = (bt == 100) & (p + 8 <= 64)
        if m.any():
            ii, pp = idx[m], p[m]
            raw = np.stack([H[ii, pp + 4 + k] for k in range(4)], 1)
            raw = np.where(big[ii][:, None], raw, raw[:, ::-1]).astype(np.uint8).copy()
            rate100[ii] = raw.view(">f4").ravel()
        new = np.zeros_like(cur)
        new[idx] = nx
        cur = new
    applied = (act & 0x02) != 0
    start = sec + fract * 1e-4 + usec * 1e-6 + np.where(applied, 0, tcorr) * 1e-4
    rate = np.where(np.isfinite(rate100), rate100, rate)
    txt = lambda a, b: [bytes(r[a:b]).decode("ascii", "replace").strip() for r in H]  # noqa: E731
    return pd.DataFrame({"offset": offsets if offsets is not None else np.arange(nrec) * 0,
                         "net": txt(18, 20), "sta": txt(8, 13), "loc": txt(13, 15), "cha": txt(15, 18),
                         "start": start, "npts": u16(30), "rate": rate, "tq": tq, "locked": (io_ & 0x20) != 0,
                         "tcorr_s": tcorr * 1e-4, "corr_applied": applied, "act": act, "io": io_, "dq": dq,
                         "reclen": reclen, "valid_time": ok})


def parse_headers(buf: bytes) -> pd.DataFrame:
    """Every record header of a MiniSEED byte buffer."""
    offs = record_offsets(buf)
    b = np.frombuffer(buf, dtype=np.uint8)
    if len(offs) == 0:
        return parse_heads(np.zeros((0, 64), np.uint8))
    pad = np.concatenate([b, np.zeros(64, np.uint8)])
    H = pad[offs[:, None] + np.arange(64)]
    return parse_heads(H, offs)


def read_heads_file(path: str, offset: int, nbytes: int) -> pd.DataFrame:
    """Header-only read of a byte range of a file (fixed record length from the first record): 64 bytes per
    record, so a day of records costs a few hundred kB of reads instead of the whole data."""
    with open(path, "rb") as f:
        f.seek(offset)
        h0 = np.frombuffer(f.read(64), np.uint8)
        rl = b1000_reclen(np.concatenate([h0, np.zeros(64, np.uint8)]), 0)
        if not rl:
            f.seek(offset)
            return parse_headers(f.read(nbytes))
        n = nbytes // rl
        H = np.zeros((n, 64), np.uint8)
        for k in range(n):
            f.seek(offset + k * rl)
            x = f.read(64)
            H[k, :len(x)] = np.frombuffer(x, np.uint8)
    return parse_heads(H, np.arange(n) * rl)


def segments(h: pd.DataFrame, tol_samples: float = 0.5) -> pd.DataFrame:
    """Join the records of ONE channel (time order) into contiguous segments as libmseed/ObsPy do: a record
    continues the previous one when |start - (prev start + prev npts / rate)| <= tol_samples / rate and the rate
    is unchanged. Adds: seg, gap_s (start minus expected start; NaN for the first record), cum (samples before the
    record in its segment), resid (header start - segment start - cum / nominal rate: the error of a time
    counted from the segment start at the nominal rate, i.e. what a day-merged trace carries)."""
    h = h.sort_values("start", kind="stable").reset_index(drop=True).copy()
    t = h.start.values
    rate = h.rate.values.astype(float)
    npts = h.npts.values.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        exp = np.r_[np.nan, t[:-1] + npts[:-1] / np.where(rate[:-1] > 0, rate[:-1], np.nan)]
        gap = t - exp
        cont = np.r_[False, (np.abs(gap[1:]) <= tol_samples / np.where(rate[1:] > 0, rate[1:], np.nan))
                     & (rate[1:] == rate[:-1])]
    seg = np.cumsum(~cont) - 1
    h["seg"] = seg
    h["gap_s"] = gap
    cum = np.zeros(len(h))
    resid = np.zeros(len(h))
    for s in np.unique(seg):
        m = np.where(seg == s)[0]
        c = np.r_[0.0, np.cumsum(npts[m][:-1])]
        cum[m] = c
        resid[m] = t[m] - t[m[0]] - c / rate[m[0]] if rate[m[0]] > 0 else np.nan
    h["cum"] = cum
    h["resid"] = resid
    return h

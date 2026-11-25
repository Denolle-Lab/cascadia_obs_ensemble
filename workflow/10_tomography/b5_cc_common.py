"""Waveform cross-correlation differential times for the D-B078 test sets (D-B040: CC for clustered families).

Shared helpers of b5_cc_*.py: paths, inputs, pnwstore short reads on the record-header time base, filtering and
the cross-correlation core. Read-only on pnwstore, the pick file, the catalog and the -18 cc_test folder.

Waveform route (same archive as the picker, the D-B072 pick re-timing and b1_obs_waveform_check.py): the local
pnwstore MiniSEED day files through their mseedindex sqlite tables (/wd1/PNWstore_sqlite/<year>.sqlite), path
/data/wdNN -> /auto/pnwstore1-wdNN. Every network of the four sets (OO, NV, Z5, 7D, UW, CC, PB, XD, YH, TA) is
there; NC/BK are not (the picker read them from NCEDC) and are skipped (flag no_pnwstore), no FDSN download.

Time base (D-B072/D-B075): never a merged day. For each pick only the MiniSEED records overlapping
[pick - SNIP_PRE, pick + SNIP_POST] are read (byte offsets from the index time-index, a small aligned chunk
around the interpolated offset). ObsPy joins those few records; the trace is then shifted so that the record
holding the pick sample is on its own header time (the definition used by the re-timing), and the largest
header-vs-count deviation of the other records in the snippet is stored (hdr_dev_max_ms).
"""
import io
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import obspy
from obspy import UTCDateTime, Stream

HERE = Path(__file__).resolve().parent
OUT = Path("/wd1/mdenolle_data/tomo_v1/cc_dt")
SNIP = OUT / "snippets"
SETDIR = Path("/wd1/mdenolle_data/tomo_v1/b5/cc_test")
RETIME = Path("/wd1/mdenolle_data/tomo_v1/checks/pick_retiming")
PICKS = RETIME / "Cascadia_picks_catalog_v4.1t.csv"
PICKS_MD5 = "762f18e8f181eaaaa247d367c2d2c302"
CORR = RETIME / "corrections.csv.gz"
CATALOG = Path("/wd1/mdenolle_data/catalog_v5/v5.1/cascadia_catalog_v5.1.csv")
INDEX = "/wd1/PNWstore_sqlite/{year}.sqlite"
SETS = ["axial", "blanco", "msh", "jdf"]

SNIP_PRE, SNIP_POST = 8.0, 8.0   # s read around each pick (CC window + max lag + filter margin + noise window)
NO_PNWSTORE = {"NC", "BK"}

# ------------------------------------------------------------------------------------------------ inputs


def load_events(name):
    ev = pd.read_csv(SETDIR / f"events_{name}.csv")
    ev["T0"] = [UTCDateTime(t).timestamp for t in ev.time]
    return ev


def load_given_pairs(name):
    return pd.read_csv(SETDIR / f"pairs_{name}.csv")


def timing_status(flag, applied):
    """Pick timing class from corrections.csv (D-B072): retimed (time changed), checked (re-timed, no change
    needed), unretimable (off_grid / rate_not_100 / error: kept at the v4.1 time), not_checked (station-day
    not in the re-timing tasks: no drift found by the scan, or not scanned)."""
    if flag is None or (isinstance(flag, float) and np.isnan(flag)):
        return "not_checked"
    if flag in ("off_grid", "rate_not_100", "error"):
        return "unretimable"
    return "retimed" if bool(applied) else "checked"


# ------------------------------------------------------------------------------------------------ pnwstore

_dbs = {}


def db(year):
    if year not in _dbs:
        _dbs[year] = sqlite3.connect(f"file:{INDEX.format(year=year)}?mode=ro", uri=True)
    return _dbs[year]


# pnwstore.constants.wd_mapper (year -> disk; the /data/wdNN of the index is not the disk), 2000-2023
WD = {2000: "wd05", 2001: "wd08", 2002: "wd07", 2003: "wd06", 2004: "wd05", 2005: "wd04", 2006: "wd03",
      2007: "wd02", 2008: "wd01", 2009: "wd00", 2010: "wd08", 2011: "wd02", 2012: "wd08", 2013: "wd07",
      2014: "wd06", 2015: "wd05", 2016: "wd05", 2017: "wd07", 2018: "wd02", 2019: "wd01", 2020: "wd00",
      2021: "wd10", 2022: "wd03", 2023: "wd09"}


def real_path(fn):
    """pnwstore.utils.pnwstore_filename_mapper: /data/wdXX/PNWyyyy/... -> /auto/pnwstore1-<wd of yyyy>/PNWyyyy/..."""
    return "/auto/pnwstore1-" + WD[int(fn.split("/")[5])] + fn[10:]


def index_rows(net, sta, t_lo, t_hi):
    """tsindex rows of a station overlapping [t_lo, t_hi] (UTCDateTime), all channels, both years if needed."""
    out = []
    for y in sorted({t_lo.year, t_hi.year}):
        if not os.path.exists(INDEX.format(year=y)):
            continue
        q = ("select network,station,location,channel,starttime,endtime,samplerate,filename,byteoffset,bytes,"
             "timeindex from tsindex where network=? and station=? and starttime<=? and endtime>=?")
        out.append(pd.read_sql(q, db(y), params=[net, sta, t_hi.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                                                 t_lo.strftime("%Y-%m-%dT%H:%M:%S.%f")]))
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


BANDS = ("HH", "BH", "EH")   # picker rule (elep_picker.select_channels hh_bh_eh): first band with a vertical


def choose_band(rows):
    """Band and component channels: the picker's rule HH > BH > EH (first band with a Z). Returns
    (band, z_channel, [horizontal channels]) or None. Horizontals: E/N or 1/2 of the same band."""
    if len(rows) == 0:
        return None
    ch = set(rows.channel)
    for b in BANDS:
        z = [c for c in ch if c[:2] == b and c[2] in "Z3"]
        if z:
            zc = b + "Z" if b + "Z" in ch else sorted(z)[0]
            hz = sorted(c for c in ch if c[:2] == b and c[2] in "EN12")
            return b, zc, hz
    return None


# MiniSEED fixed-header parse: copy of prt_common.parse_headers (D-B072 re-timing, scripts/prt_common.py), trimmed.
def _rate(f, m):
    f = f.astype(float)
    m = m.astype(float)
    return np.where((f > 0) & (m > 0), f * m,
           np.where((f > 0) & (m < 0), -f / np.where(m == 0, 1, m),
           np.where((f < 0) & (m > 0), -m / np.where(f == 0, 1, f),
           np.where((f < 0) & (m < 0), 1.0 / (f * m), 0.0))))


def _b1000_len(b, off):
    n = len(b)
    nb = b[off + 39]
    bo = int(b[off + 46]) << 8 | int(b[off + 47])
    for _ in range(int(nb) + 1):
        if bo == 0 or bo + off + 8 > n:
            break
        btype = int(b[off + bo]) << 8 | int(b[off + bo + 1])
        if btype == 1000:
            return 2 ** int(b[off + bo + 6])
        bo = int(b[off + bo + 2]) << 8 | int(b[off + bo + 3])
    return None


def parse_headers(buf):
    """Start time (POSIX float, BTIME + B1001 usec + uncorrected time correction, as libmseed), npts, rate,
    channel of every record in a MiniSEED buffer that starts on a record boundary (big-endian or little-endian)."""
    b = np.frombuffer(buf, dtype=np.uint8)
    n = len(b)
    offs, o = [], 0
    while o + 48 <= n:
        rl = _b1000_len(b, o)
        if not rl:
            break
        if o + rl > n:
            break
        offs.append(o)
        o += rl
    if not offs:
        return pd.DataFrame(columns=["offset", "reclen", "t", "npts", "rate", "cha"])
    offs = np.array(offs)
    H = b[offs[:, None] + np.arange(64)]
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
    hh, mm, ss = H[:, 24].astype(np.int64), H[:, 25].astype(np.int64), H[:, 26].astype(np.int64)
    fract, npts = u16(28), u16(30)
    rate = _rate(i16(32), i16(34))
    act = H[:, 36]
    tcorr = i32(40)
    bo = u16(46)
    base = (pd.to_datetime((year * 1000 + doy).astype(str), format="%Y%j").values.astype("datetime64[s]")
            .astype(np.int64))
    sec = base + hh * 3600 + mm * 60 + ss
    usec = np.zeros(len(offs))
    rate100 = np.full(len(offs), np.nan)
    cur = bo.copy()
    for _ in range(4):
        valid = (cur > 0) & (cur + 8 <= 4096)
        if not valid.any():
            break
        idx = np.where(valid)[0]
        pos = np.minimum(offs[idx] + cur[idx], n - 8)
        bt = np.where(big[idx], (b[pos].astype(int) << 8) | b[pos + 1], (b[pos + 1].astype(int) << 8) | b[pos])
        nxt = np.where(big[idx], (b[pos + 2].astype(int) << 8) | b[pos + 3], (b[pos + 3].astype(int) << 8) | b[pos + 2])
        m1001 = bt == 1001
        if m1001.any():
            usec[idx[m1001]] = b[pos[m1001] + 5].astype(np.int8)
        m100 = bt == 100
        if m100.any():
            ii, pp = idx[m100], pos[m100]
            raw = np.stack([b[pp + 4 + k] for k in range(4)], 1)
            raw = np.where(big[ii][:, None], raw, raw[:, ::-1]).astype(np.uint8).copy()
            rate100[ii] = raw.view(">f4").ravel()
        newcur = np.zeros_like(cur)
        newcur[idx] = nxt
        cur = newcur
    applied = (act & 0x02) != 0
    corr = np.where(applied, 0, tcorr) * 1e-4
    rate = np.where(np.isfinite(rate100), rate100, rate)
    t = sec.astype(np.float64) + fract * 1e-4 + usec * 1e-6 + corr
    reclen = np.diff(np.append(offs, o))
    return pd.DataFrame({"offset": offs, "reclen": reclen, "t": t, "npts": npts, "rate": rate,
                         "cha": [bytes(r[15:18]).decode("ascii", "replace").strip() for r in H]})


def _timeindex(row):
    keys, offs = [], []
    for x in str(row.timeindex).split(","):
        if "=>" not in x or x.startswith("latest"):
            continue
        a, b_ = x.split("=>")
        keys.append(float(a))
        offs.append(int(b_))
    if not keys:
        keys, offs = [UTCDateTime(row.starttime).timestamp], [int(row.byteoffset)]
    return np.array(keys), np.array(offs)


class FileCache:
    """Per-process cache of byte ranges already read (key: file, offset)."""

    def __init__(self, maxbytes=64 << 20):
        self.d = {}
        self.size = 0
        self.max = maxbytes
        self.nread = 0
        self.bytes_read = 0

    def read(self, fn, off, nb):
        k = (fn, off, nb)
        if k in self.d:
            return self.d[k]
        with open(fn, "rb") as f:
            f.seek(off)
            buf = f.read(nb)
        self.nread += 1
        self.bytes_read += len(buf)
        if self.size + len(buf) > self.max:
            self.d.clear()
            self.size = 0
        self.d[k] = buf
        self.size += len(buf)
        return buf


FC = FileCache()


def records_window(row, t_lo, t_hi, chunk=65536):
    """Bytes and parsed headers of the records of one index row that overlap [t_lo, t_hi] (POSIX floats).

    Uses the row's time index (hourly record offsets) to bound the search, reads an aligned chunk around the
    linearly interpolated offset, widens it until the records cover the window (or the bounds are reached)."""
    fn = real_path(row.filename)
    r0, r1 = int(row.byteoffset), int(row.byteoffset) + int(row.bytes)
    keys, offs = _timeindex(row)
    i = max(0, int(np.searchsorted(keys, t_lo, side="right")) - 1)
    j = int(np.searchsorted(keys, t_hi, side="right"))
    lo_b = int(offs[i])
    hi_b = int(offs[j]) if j < len(offs) else r1
    hi_b = min(max(hi_b, lo_b), r1)
    head = FC.read(fn, lo_b, 64)
    rl = _b1000_len(np.frombuffer(head, dtype=np.uint8), 0) or 512
    if hi_b - lo_b <= 4 * chunk:
        a, b_ = lo_b, hi_b
    else:
        # interpolate the byte position of t_lo between the bounding index keys
        k1 = keys[i + 1] if i + 1 < len(keys) else UTCDateTime(row.endtime).timestamp
        frac = (t_lo - keys[i]) / max(k1 - keys[i], 1e-3)
        g = lo_b + int(frac * (hi_b - lo_b))
        a = lo_b + ((max(g - chunk, lo_b) - lo_b) // rl) * rl
        b_ = min(a + 2 * chunk + int((t_hi - t_lo) / max(k1 - keys[i], 1e-3) * (hi_b - lo_b)), hi_b)
    for _ in range(12):
        buf = FC.read(fn, a, b_ - a)
        h = parse_headers(buf)
        if len(h) == 0 and (a, b_) != (lo_b, hi_b):
            a, b_ = lo_b, hi_b      # guessed chunk not on a record boundary: read the whole index block
            continue
        if len(h) == 0:
            return b"", h
        tend = h.t.values + h.npts.values / np.where(h.rate.values > 0, h.rate.values, np.nan)
        ok_lo = h.t.values[0] <= t_lo or a <= lo_b
        ok_hi = tend[-1] >= t_hi or b_ >= hi_b
        if ok_lo and ok_hi:
            break
        span = b_ - a
        if not ok_lo:
            a = max(lo_b, a - max(span, chunk))
            a = lo_b + ((a - lo_b) // rl) * rl
        if not ok_hi:
            b_ = min(hi_b, b_ + max(span, chunk))
    tend = h.t.values + h.npts.values / np.where(h.rate.values > 0, h.rate.values, np.nan)
    sel = np.where((h.t.values <= t_hi) & (tend >= t_lo))[0]
    if len(sel) == 0:
        return b"", h.iloc[:0]
    s0, s1 = sel[0], sel[-1]
    sub = h.iloc[s0:s1 + 1].copy()
    o0 = int(sub.offset.iloc[0])
    o1 = int(sub.offset.iloc[-1] + sub.reclen.iloc[-1])
    sub["offset"] -= o0
    return buf[o0:o1], sub


def read_channel(rows, cha, tp, t_lo, t_hi):
    """Snippet of one channel around pick time tp (POSIX float): records overlapping [t_lo, t_hi] of every index
    row of that channel (two rows at a day boundary). Returns (Trace or None, info dict).

    Time base: the trace holding tp is shifted so that the record holding the sample at tp starts at its own
    header time (D-B072 definition); hdr_dev_max_ms = largest |header - count| of the snippet's other records."""
    info = dict(status="ok")
    rr = rows[rows.channel == cha]
    if len(rr) == 0:
        info["status"] = "no_channel"
        return None, info
    st = Stream()
    hdrs = []
    for _, row in rr.iterrows():
        buf, h = records_window(row, t_lo, t_hi)
        if not len(buf):
            continue
        try:
            s = obspy.read(io.BytesIO(buf), format="MSEED")
        except Exception as e:  # corrupt record
            info["status"] = f"read_error:{type(e).__name__}"
            continue
        s = s.select(channel=cha)
        st += s
        h = h[h.cha == cha]
        hdrs.append(h)
    if len(st) == 0:
        info["status"] = info["status"] if info["status"] != "ok" else "no_records"
        return None, info
    h = pd.concat(hdrs, ignore_index=True).sort_values("t").drop_duplicates("t")
    locs = sorted({tr.stats.location for tr in st})
    if len(locs) > 1:
        info["multi_loc"] = ",".join(locs)
        st = st.select(location=locs[0])
    # trace holding tp
    cand = [tr for tr in st if tr.stats.starttime.timestamp <= tp <= tr.stats.endtime.timestamp]
    if not cand:
        info["status"] = "pick_in_gap"
        return None, info
    tr = cand[0]
    sr = tr.stats.sampling_rate
    t0 = tr.stats.starttime.timestamp
    hh = h[(h.t >= t0 - 2e-6) & (h.t <= tr.stats.endtime.timestamp + 1.0 / sr)]
    j0 = np.where(np.abs(hh.t.values - t0) < 2e-6)[0]
    if len(j0) == 0:
        info["status"] = "anchor_fail"
        return None, info
    hh = hh.iloc[j0[0]:]
    cum = np.concatenate([[0], np.cumsum(hh.npts.values)])[:len(hh)]
    keep = cum < tr.stats.npts
    hh, cum = hh[keep], cum[keep]
    dev = hh.t.values - (t0 + cum / sr)            # header - count of each record in the snippet
    r = int(np.searchsorted(t0 + cum / sr, tp, side="right") - 1)
    r = min(max(r, 0), len(dev) - 1)
    shift = float(dev[r])
    tr.stats.starttime = tr.stats.starttime + shift
    info.update(anchor_shift_ms=shift * 1e3, hdr_dev_max_ms=float(np.abs(dev - shift).max()) * 1e3,
                rate=sr, n_traces=len(st), npts_read=int(tr.stats.npts))
    # other traces (gaps) are dropped; coverage is checked at CC time
    tr.trim(UTCDateTime(t_lo), UTCDateTime(t_hi))
    info["t_start"] = tr.stats.starttime.timestamp
    info["t_end"] = tr.stats.endtime.timestamp
    info["covers"] = bool(info["t_start"] <= t_lo + 1.0 / sr and info["t_end"] >= t_hi - 1.0 / sr)
    return tr, info


# ------------------------------------------------------------------------------------------------ CC core


def prep(x, sr, band, taper_s=1.0):
    """Demean, detrend, cosine taper (taper_s each end), zero-phase 4-pole Butterworth band-pass (2 passes of
    2 corners). x: 1-D array of the full snippet; returns float64 array."""
    from scipy.signal import butter, sosfiltfilt, detrend
    y = detrend(np.asarray(x, dtype=np.float64), type="linear")
    n = len(y)
    m = int(min(taper_s * sr, n // 4))
    if m > 1:
        w = 0.5 * (1 - np.cos(np.pi * np.arange(m) / m))
        y[:m] *= w
        y[-m:] *= w[::-1]
    fmax = min(band[1], 0.45 * sr)
    sos = butter(2, [band[0], fmax], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, y)


def norm_cc(a, b, maxlag):
    """Normalized cross-correlation of a fixed window a (n samples) against b (n + 2*maxlag samples, centred on
    the same nominal position). a, b may be 2-D (components x samples): vector CC = sum over components of the
    products / sqrt(total energies). Returns cc[k] for lag k - maxlag (lag > 0: b's signal is later)."""
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    n = a.shape[1]
    L = 2 * maxlag + 1
    num = np.zeros(L)
    eb = np.zeros(L)
    for ai, bi in zip(a, b):
        num += np.correlate(bi, ai, mode="valid")
        c = np.concatenate([[0.0], np.cumsum(bi * bi)])
        eb += c[n:n + L] - c[:L]
    ea = float((a * a).sum())
    den = np.sqrt(ea * eb)
    with np.errstate(invalid="ignore", divide="ignore"):
        cc = np.where(den > 0, num / den, 0.0)
    return cc


def subsample_peak(cc, k):
    """Parabolic interpolation of the peak at integer index k: (fractional offset in samples, peak value)."""
    if k <= 0 or k >= len(cc) - 1:
        return 0.0, float(cc[k])
    y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
    d = y0 - 2 * y1 + y2
    if d >= 0:
        return 0.0, float(y1)
    p = 0.5 * (y0 - y2) / d
    return float(p), float(y1 - 0.25 * (y0 - y2) * p)

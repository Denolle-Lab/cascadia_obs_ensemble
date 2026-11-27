"""Step 2: pulse calendar from continuous data (window definition; before any event scoring).

Every day 2010-01-01..2016-12-31, 11:00-13:30 local time (America/Los_Angeles: PST/PDT handled by the time zone),
EDM.UW (all weekdays: Wednesday against the other days as control) and, on Wednesdays, also HSR, STD, SHW and
ELK.UW. Masked normalised cross-correlation with the template (maf_template.py) at every 0.1-s lag; peaks with
cc >= 0.5, at least 60 s apart, are kept with their onset time, cc, template amplitude and amplitude / noise.
Resumable: rows appended to WORK/scan_rows.csv; a station-day with a row is done.

  python maf_scan.py [--nproc 6]
"""
import argparse
import csv
import multiprocessing as mp
import os

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import maf_common as C  # noqa: E402

COLS = ["sta", "day", "dow", "status", "cover_min", "n_peaks", "onset_utc", "onset_local_min", "cc", "amp", "snr"]
WIN = ("11:00", "13:30")
CC_KEEP = 0.5


def peaks(cc, min_sep):
    """Greedy: highest cc first, then suppress +-min_sep samples."""
    order = np.argsort(cc)[::-1]
    taken = np.zeros(len(cc), bool)
    out = []
    for i in order:
        if cc[i] < CC_KEEP:
            break
        if taken[i]:
            continue
        out.append(i)
        taken[max(0, i - min_sep):i + min_sep + 1] = True
    return out


def scan_one(a):
    sta, day = a
    tt, ty, tw = C.template()
    lt0 = pd.Timestamp(f"{day} {WIN[0]}", tz=C.TZ)
    lt1 = pd.Timestamp(f"{day} {WIN[1]}", tz=C.TZ)
    t0, t1 = lt0.tz_convert("UTC"), lt1.tz_convert("UTC")
    base = dict(sta=sta, day=day, dow=int(lt0.dayofweek))
    try:
        segs, st = C.read10(sta, t0, t1)
    except Exception as e:
        return [dict(base, status=f"error {type(e).__name__}"[:60])]
    if not segs:
        return [dict(base, status=st)]
    cover = sum(len(x) for _, x in segs) / C.FS / 60
    rows = []
    for s0, x in segs:
        cc, amp = C.cc_series(x, tt, ty, tw)
        if len(cc) == 0:
            continue
        sig = C.noise_sigma(x)
        for i in peaks(cc, int(60 * C.FS)):
            on = s0 + i / C.FS + C.T_PRE
            ont = pd.Timestamp(on, unit="s", tz="UTC")
            lo = C.local(ont)
            rows.append(dict(base, status="ok", cover_min=round(cover, 2), onset_utc=ont.isoformat(),
                             onset_local_min=round(lo.hour * 60 + lo.minute + lo.second / 60 + lo.microsecond / 6e7, 4),
                             cc=round(float(cc[i]), 4), amp=round(float(amp[i]), 2),
                             snr=round(float(abs(amp[i]) / sig), 2)))
    if not rows:
        rows = [dict(base, status="ok", cover_min=round(cover, 2))]
    for r in rows:
        r["n_peaks"] = len([x for x in rows if "cc" in x])
    return rows


def jobs():
    days = pd.date_range("2010-01-01", "2016-12-31", freq="D")
    out = [("EDM.UW", d.strftime("%Y-%m-%d")) for d in days]
    for s in ("HSR.UW", "STD.UW", "SHW.UW", "ELK.UW"):
        out += [(s, d.strftime("%Y-%m-%d")) for d in days if d.dayofweek == 2]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=C.NPROC)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--confirm", action="store_true",
                    help="second pass: days (any weekday) with an EDM.UW detection also at HSR, STD, SHW, ELK.UW")
    a = ap.parse_args()
    f = C.WORK / "scan_rows.csv"
    done = set()
    if f.exists():
        old = pd.read_csv(f, usecols=["sta", "day"])
        done = set(zip(old.sta, old.day))
    jj = jobs()
    if a.confirm:
        sc = pd.read_csv(f)
        hit = sorted(set(sc[(sc.sta == "EDM.UW") & (sc.cc >= 0.9) & (sc.snr >= 5)].day))
        jj = [(s, d) for d in hit for s in ("HSR.UW", "STD.UW", "SHW.UW", "ELK.UW")]
    todo = [j for j in jj if j not in done]
    # Wednesdays first, so the calendar is usable early
    todo.sort(key=lambda j: (pd.Timestamp(j[1]).dayofweek != 2, j[1], j[0]))
    if a.limit:
        todo = todo[:a.limit]
    print(f"scan: {len(todo)} station-days to do", flush=True)
    new = not f.exists()
    with open(f, "a", newline="") as fh, mp.get_context("spawn").Pool(a.nproc) as p:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        for i, rows in enumerate(p.imap_unordered(scan_one, todo, chunksize=2)):
            for r in rows:
                w.writerow(r)
            if i % 50 == 0:
                fh.flush()
                print(f"scan {i + 1}/{len(todo)}", flush=True)
    print("scan done", flush=True)


if __name__ == "__main__":
    main()

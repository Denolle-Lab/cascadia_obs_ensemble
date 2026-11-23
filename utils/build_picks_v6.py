#!/usr/bin/env python3
"""Build the ELEP pick table ver6 and its Route A amplitudes (workflow/01_picking/V5_PLAN.md §12).

ver6 = ver5, byte for byte (so every ver5 pick_id is unchanged), followed by the picks of the
v6 fill: the station-days of 122-123°W, 40-50°N, 2010-2016 with no pick in ver5, picked with
option (a) by workflow/01_picking/pick_and_amps.py, which also measured their amplitudes.
New rows use the ver3-ver5 columns and conventions and continue the pick_id.

Dropped (PI decision, 2026-10-26): station-days whose data are broken into short fragments,
i.e. at least 20 picks of which fewer than half have an amplitude (the amplitude engine only
deconvolves segments of >= 300 s). Gluing the fragments leaves a step at every join and the
picker fires on them (CN.HNBB, Oct 2014 - Apr 2015: ~3,500 picks a day, against ~30 on clean
days). They are listed in ver6_dropped_station_days.csv. Days with more than 10 times the
station's median pick count are kept and flagged (`burst`) in ver6_added_station_days.csv.

Writes, in --out:
    all_picks_all_regions_2010_2016_ver6.csv.gz   the table (ver5 .gz + one gzip member)
    ver6_added_picks.csv.gz                       the new rows only, with the header
    ver6_added_station_days.csv                   one row per added station-day: network, station,
                                                  day, n_picks, pick-id range, amp_ok_frac, burst
    ver6_dropped_station_days.csv                 the fragmented station-days left out
    ver6_summary.txt, MD5SUMS
    amps/amps_{net}.{sta}.{year}.parquet          amplitudes of the new picks, keyed by pick_id
                                                  (same columns as amps_v5)

    python utils/build_picks_v6.py --out /wd1/mdenolle_data/picks_v6
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_picks_v4 import COLUMNS, ver3_location  # noqa: E402

VER5 = Path("/wd1/mdenolle_data/picks_v5/all_picks_all_regions_2010_2016_ver5.csv.gz")
VER5_N = 60_918_847                       # picks in ver5 (ver5_summary.txt); next pick_id
FILL = Path("/wd1/mdenolle_data/picks_v6_east")
CATEGORY = "v6_east_fill"
MIN_PICKS, MIN_AMP_FRAC, BURST = 20, 0.5, 10


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def last_pick_id(path: Path) -> int:
    last = b""
    with gzip.open(path, "rb") as f:
        for line in f:
            last = line
    return int(next(csv.reader([last.decode()]))[COLUMNS.index("pick_id")])


def station_days(fill: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    log = pd.read_csv(fill / "log.csv", dtype=str)
    ok = log[log.status == "ok"].copy()
    ok["n_picks"] = ok.n_a.astype(int)
    ok["amp_ok_frac"] = (ok.n_amp_ok.astype(int) / ok.n_picks).round(3)
    med = ok.groupby(["network", "station"]).n_picks.transform("median")
    ok["burst"] = ok.n_picks > BURST * med.clip(lower=10)
    drop = (ok.n_picks >= MIN_PICKS) & (ok.amp_ok_frac < MIN_AMP_FRAC)
    keep = ok[~drop].sort_values(["network", "station", "day"]).reset_index(drop=True)
    keep["first_pick_id"] = VER5_N + keep.n_picks.cumsum() - keep.n_picks
    keep["last_pick_id"] = keep.first_pick_id + keep.n_picks - 1
    cols = ["network", "station", "day", "n_picks", "amp_ok_frac", "burst"]
    return keep, ok[drop][cols].sort_values(["network", "station", "day"])


def zero_past_end(df):
    """A window that ends exactly at the end of the data segment can come out as 0 instead of
    NaN (2 picks of UW.LAB in 2011). The values are running maxima, so a 0 after a positive
    value is such a window: set it to NaN, as the longer windows after it."""
    for pre in ("wa_mm_", "disp_um_"):
        cols = [c for c in df.columns if c.startswith(pre)]
        A = df[cols].to_numpy(np.float32)
        bad = (A == 0) & (np.fmax.accumulate(np.nan_to_num(A), axis=1) > 0)
        if bad.any():
            A[bad] = np.nan
            df[cols] = A


def amps_station_year(job):
    """Re-key one station-year of per-day amplitude files from `row` to pick_id."""
    fill, out, net, sta, year, days = job
    parts = []
    for day, first, n in days:
        a = pd.read_parquet(fill / "amps" / f"{net}.{sta}.{day}.parquet")
        assert len(a) == n and (a["row"].to_numpy() == np.arange(n)).all(), (net, sta, day)
        parts.append(a.rename(columns={"row": "pick_id"}).assign(pick_id=a["row"] + first))
    df = pd.concat(parts, ignore_index=True)
    df["pick_id"] = df.pick_id.astype("int64")
    zero_past_end(df)
    df.to_parquet(out / "amps" / f"amps_{net}.{sta}.{year}.parquet", index=False,
                  compression="zstd")
    return len(df), int((df.status == "ok").sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("/wd1/mdenolle_data/picks_v6"))
    ap.add_argument("--fill", type=Path, default=FILL)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    (args.out / "amps").mkdir(parents=True, exist_ok=True)

    last = last_pick_id(VER5)
    assert last == VER5_N - 1, f"ver5 ends at pick_id {last}, expected {VER5_N - 1}"
    keep, dropped = station_days(args.fill)
    print(f"fill: {len(keep):,} station-days kept ({keep.n_picks.sum():,} picks), "
          f"{len(dropped):,} dropped ({dropped.n_picks.sum():,} picks)", flush=True)

    member = args.out / "ver6_added_picks.member.gz"   # new rows, no header
    next_id = VER5_N
    with gzip.open(member, "wt", newline="", compresslevel=6) as fo:
        w = csv.writer(fo, lineterminator="\n")
        for r in keep.itertuples(index=False):
            x = pd.read_csv(args.fill / "picks" / f"{r.network}.{r.station}.{r.day}.csv",
                            dtype=str, keep_default_na=False, na_values=[""])
            assert len(x) == r.n_picks and next_id == r.first_pick_id, (r.network, r.station, r.day)
            for p in x.itertuples(index=False):
                w.writerow([next_id, next_id, p.network, p.station, ver3_location(p.location),
                            p.band_inst, p.label, p.trace_starttime, p.trigger_onset, p.pick_time,
                            p.trigger_offset, p.max_prob, p.thresh_prob, next_id,
                            f"{p.network}.{p.station}."])
                next_id += 1
    print(f"picks written: {next_id - VER5_N:,}", flush=True)

    out = args.out / "all_picks_all_regions_2010_2016_ver6.csv.gz"
    with open(out, "wb") as fo:
        for src in (VER5, member):
            with open(src, "rb") as fi:
                shutil.copyfileobj(fi, fo, 1 << 24)
    delta = args.out / "ver6_added_picks.csv.gz"
    with open(delta, "wb") as fo:
        fo.write(gzip.compress((",".join(COLUMNS) + "\n").encode()))
        with open(member, "rb") as fi:
            shutil.copyfileobj(fi, fo, 1 << 24)
    member.unlink()

    keep["year"] = keep.day.str[:4]
    jobs = [(args.fill, args.out, n, s, y,
             list(zip(g.day, g.first_pick_id, g.n_picks)))
            for (n, s, y), g in keep.groupby(["network", "station", "year"])]
    n_amp = n_ok = 0
    with Pool(args.workers) as pool:
        for a, b in pool.imap_unordered(amps_station_year, jobs):
            n_amp += a
            n_ok += b
    assert n_amp == next_id - VER5_N, (n_amp, next_id - VER5_N)
    print(f"amplitudes: {len(jobs)} station-year files, {n_amp:,} rows, {n_ok:,} ok", flush=True)

    keep[["network", "station", "day", "n_picks", "first_pick_id", "last_pick_id", "amp_ok_frac",
          "burst"]].assign(category=CATEGORY).to_csv(args.out / "ver6_added_station_days.csv", index=False)
    dropped.to_csv(args.out / "ver6_dropped_station_days.csv", index=False)
    with gzip.open(delta, "rt") as f:
        ph = pd.read_csv(f, usecols=["label"], dtype=str)
    lines = [f"ver5 picks: {VER5_N:,}", f"ver6 picks: {next_id:,}",
             f"added: {next_id - VER5_N:,} picks on {len(keep):,} station-days "
             f"({keep.station.nunique()} stations, {keep.network.nunique()} networks), "
             f"122-123W 40-50N 2010-2016",
             f"added P / S: {(ph.label == 'P').sum():,} / {(ph.label == 'S').sum():,}",
             f"with an amplitude: {n_ok:,} ({n_ok / n_amp:.1%})",
             f"flagged burst (> {BURST}x station median, kept): {int(keep.burst.sum()):,} station-days, "
             f"{keep[keep.burst].n_picks.sum():,} picks",
             f"dropped (fragmented data: >= {MIN_PICKS} picks, < {MIN_AMP_FRAC:.0%} with amplitude): "
             f"{len(dropped):,} station-days, {dropped.n_picks.sum():,} picks", "",
             "added picks by network and year:",
             keep.pivot_table(index="network", columns="year", values="n_picks", aggfunc="sum",
                              fill_value=0, margins=True).to_string(), "",
             "added station-days by network and year:",
             keep.assign(n=1).pivot_table(index="network", columns="year", values="n", aggfunc="sum",
                                          fill_value=0, margins=True).to_string()]
    (args.out / "ver6_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:7]))

    names = [out.name, delta.name, "ver6_added_station_days.csv", "ver6_dropped_station_days.csv"]
    (args.out / "MD5SUMS").write_text("".join(f"{md5(args.out / n)}  {n}\n" for n in names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

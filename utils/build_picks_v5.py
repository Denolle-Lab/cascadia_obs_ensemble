#!/usr/bin/env python3
"""Build the ELEP pick table ver5: ver4 plus the offshore gap fill (1_picking/V5_PLAN.md §10).

ver5 = ver4, byte for byte (so every ver4 pick_id is unchanged), followed by the picks
of the v5 gap fill: the offshore station-days of 2010-2016 that have no pick in ver4,
picked with option (a), the model set of the region runs (1_picking/dryrun_models.py
--configs a). New rows use the ver3/ver4 columns and conventions (numeric location
codes as floats, station_id NET.STA.) and continue the pick_id.

The gzip file is two members: ver4's .gz copied unchanged, then the new rows. Any gzip
reader (zcat, pandas) reads them as one CSV. ver5_added_picks.csv.gz holds the new
rows alone, with the header, for those who already have ver4.

Writes, in --out:
    all_picks_all_regions_2010_2016_ver5.csv.gz   the table
    ver5_added_picks.csv.gz                       the new rows only (ver4 + these = ver5)
    ver5_added_station_days.csv                   one row per added station-day: category,
                                                  network, station, day, n_picks, pick_id range
    ver5_summary.txt                              counts per network, year and phase
    MD5SUMS

    python utils/build_picks_v5.py --out /wd1/mdenolle_data/picks_v5
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_picks_v4 import COLUMNS, ver3_location  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VER4 = ROOT / "data" / "picks_v4" / "all_picks_all_regions_2010_2015_ver4.csv.gz"
VER4_N = 49_541_718                       # picks in ver4 (ver4_summary.txt); next pick_id
FILL = Path("/wd1/mdenolle_data/picks_v5_fill")
CATEGORY = "v5_offshore_fill"


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def ver4_last_pick_id() -> int:
    """pick_id of the last ver4 row (reads the whole file: ~1 min)."""
    last = b""
    with gzip.open(VER4, "rb") as f:
        for line in f:
            last = line
    return int(next(csv.reader([last.decode()]))[COLUMNS.index("pick_id")])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("/wd1/mdenolle_data/picks_v5"))
    ap.add_argument("--fill", type=Path, default=FILL)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    last = ver4_last_pick_id()
    assert last == VER4_N - 1, f"ver4 ends at pick_id {last}, expected {VER4_N - 1}"

    log = pd.read_csv(args.fill / "log.csv", dtype=str)
    ok = log[log.status == "ok"].sort_values(["network", "station", "day"])
    print(f"fill: {len(ok):,} station-days with picks "
          f"({log.status.str.split(':').str[0].value_counts().to_dict()})", flush=True)

    member = args.out / "ver5_added_picks.member.gz"   # new rows, no header
    added, next_id = [], VER4_N
    with gzip.open(member, "wt", newline="", compresslevel=6) as fo:
        w = csv.writer(fo, lineterminator="\n")
        for r in ok.itertuples(index=False):
            f = args.fill / "picks" / f"{r.network}.{r.station}.{r.day}.csv"
            x = pd.read_csv(f, dtype=str, keep_default_na=False, na_values=[""])
            x = x[x.config == "a"]
            if not len(x):
                continue
            first = next_id
            for p in x.itertuples(index=False):
                loc = ver3_location(p.location)
                w.writerow([next_id, next_id, p.network, p.station, loc, p.band_inst, p.label,
                            p.trace_starttime, p.trigger_onset, p.pick_time, p.trigger_offset,
                            p.max_prob, p.thresh_prob, next_id, f"{p.network}.{p.station}."])
                next_id += 1
            added.append((CATEGORY, r.network, r.station, r.day, next_id - first, first, next_id - 1,
                          f.name))

    out = args.out / "all_picks_all_regions_2010_2016_ver5.csv.gz"
    with open(out, "wb") as fo:
        for src in (VER4, member):
            with open(src, "rb") as fi:
                shutil.copyfileobj(fi, fo, 1 << 24)
    delta = args.out / "ver5_added_picks.csv.gz"
    with open(delta, "wb") as fo:
        fo.write(gzip.compress((",".join(COLUMNS) + "\n").encode()))
        with open(member, "rb") as fi:
            shutil.copyfileobj(fi, fo, 1 << 24)
    member.unlink()

    man = pd.DataFrame(added, columns=["category", "network", "station", "day", "n_picks",
                                       "first_pick_id", "last_pick_id", "file"])
    man.to_csv(args.out / "ver5_added_station_days.csv", index=False)
    man["year"] = man.day.str[:4]
    with gzip.open(delta, "rt") as f:
        ph = pd.read_csv(f, usecols=["network", "label"], dtype=str)
    lines = [f"ver4 picks: {VER4_N:,}", f"ver5 picks: {next_id:,}",
             f"added: {next_id - VER4_N:,} picks on {len(man):,} station-days "
             f"({man.station.nunique()} stations, {man.network.nunique()} networks)",
             f"added P / S: {(ph.label == 'P').sum():,} / {(ph.label == 'S').sum():,}", "",
             "added picks by network and year:",
             man.pivot_table(index="network", columns="year", values="n_picks", aggfunc="sum",
                             fill_value=0, margins=True).to_string(), "",
             "added station-days by network and year:",
             man.assign(n=1).pivot_table(index="network", columns="year", values="n", aggfunc="sum",
                                         fill_value=0, margins=True).to_string()]
    (args.out / "ver5_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:4]))

    names = [out.name, delta.name, "ver5_added_station_days.csv"]
    (args.out / "MD5SUMS").write_text("".join(f"{md5(args.out / n)}  {n}\n" for n in names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

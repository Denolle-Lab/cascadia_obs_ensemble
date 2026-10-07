#!/usr/bin/env python3
"""Build the ver4 ELEP pick table: ver3 plus the per-day picks the ver3 merge left out.

ver4 = ver3, byte for byte (so every ver3 pick_id is unchanged), followed by the
picks of every station-day (network, station, UTC day) that has no pick at all in
ver3. A station-day already in ver3, from any run, is never added again, so two
runs are never mixed on one station-day. New rows use the ver3 columns and
conventions (v1 rows: empty band_inst and trigger fields; station_id NET.STA.;
numeric location codes written as floats, e.g. 2.0) and continue the pick_id.

Sources, in priority order (see workflow/01_picking/legacy/README.md):

    v2_region    picks_{year}_{122-129,123-127_EH,122-123_46-50,127-129_46-50}/
    v2_2013      picks_2013_122-123_46-50/picks_2013_122-123_46-50/  (nested, not merged in ver3)
    v1           picks_{2011..2015}/                                  (first run)
    v2_40-46     picks_2010_122-123_40-46/                             (run of 2025-09, after ver3)

Writes, in --out:
    all_picks_all_regions_2010_2015_ver4.csv   the table
    ver4_added_station_days.csv                one row per added station-day: category, source,
                                               pick_id range, file and its UTC date (fmtime)
    ver4_summary.txt                           counts per source, year and station

    python utils/build_picks_v4.py --out data/picks_v4
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import os
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PICKDIR = Path("/wd1/hbito_data/data")
VER3 = ROOT / "data" / "datasets_all_regions" / "all_picks_all_regions_2010_2015_ver3.csv"
YEARS = range(2010, 2016)
REGIONS = ["122-129", "123-127_EH", "122-123_46-50", "127-129_46-50"]
# newest file date the ver3 merge read (from the file dates; the merge itself is not in the repo)
VER3_MERGED = "2025-03-05"
COLUMNS = ["", "Unnamed: 0", "network", "station", "location", "band_inst", "label",
           "trace_starttime", "trigger_onset", "pick_time", "trigger_offset", "max_prob",
           "thresh_prob", "pick_id", "station_id"]


def sources():
    """(source, kind, file) for every per-day file, in priority order."""
    for y in YEARS:
        for r in REGIONS:
            for f in sorted(glob.glob(str(PICKDIR / f"picks_{y}_{r}" / "*.csv"))):
                yield "v2_region", "v2", f
    for f in sorted(glob.glob(str(PICKDIR / "picks_2013_122-123_46-50" / "picks_2013_122-123_46-50" / "*.csv"))):
        yield "v2_2013", "v2", f
    for y in range(2011, 2016):
        for f in sorted(glob.glob(str(PICKDIR / f"picks_{y}" / "*.csv"))):
            yield "v1", "v1", f
    for f in sorted(glob.glob(str(PICKDIR / "picks_2010_122-123_40-46" / "*.csv"))):
        yield "v2_40-46", "v2", f


def category(source: str, f: str) -> str:
    """Why the station-day is missing from ver3 (data/PICKS_VER4.md); drop by category there."""
    if source == "v2_region":
        return "eh_run_incomplete_at_merge" if "_EH" in f else "edge_run_dropped"
    return {"v2_2013": "2013_nested_folder", "v1": "v1_not_in_ver3",
            "v2_40-46": "new_run_2010_40-46N"}[source]


def file_date(f: str) -> str:
    """UTC modification date of a per-day file."""
    return dt.datetime.fromtimestamp(os.path.getmtime(f), dt.timezone.utc).date().isoformat()


def ver3_location(loc) -> str:
    """ver3 wrote numeric location codes as floats ('02' -> '2.0')."""
    loc = "" if pd.isna(loc) else str(loc).strip()
    return f"{float(loc):.1f}" if loc.isdigit() else loc


def rows_v2(x: pd.DataFrame):
    for r in x.itertuples(index=False):
        yield [r.network, r.station, ver3_location(r.location), r.band_inst, r.label,
               r.trace_starttime, r.trigger_onset, r.pick_time, r.trigger_offset,
               r.max_prob, r.thresh_prob]


def rows_v1(x: pd.DataFrame):
    for col, label in (("trace_p_arrival", "P"), ("trace_s_arrival", "S")):
        for r in x.itertuples(index=False):
            t = getattr(r, col)
            if pd.isna(t) or not str(t).strip():
                continue
            yield [r.station_network_code, r.station_code, ver3_location(r.station_location_code),
                   "", label, r.trace_start_time, " ", t, "", "", ""]


def station_day(kind: str, x: pd.DataFrame):
    if kind == "v2":
        return x.network.iloc[0], x.station.iloc[0], str(x.pick_time.iloc[0])[:10]
    t = x.trace_p_arrival.dropna()
    t = t[t.str.strip() != ""]
    t = t if len(t) else x.trace_s_arrival.dropna()
    return x.station_network_code.iloc[0], x.station_code.iloc[0], str(t.iloc[0])[:10]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "picks_v4")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "all_picks_all_regions_2010_2015_ver4.csv"

    print("reading ver3 station-days ...", flush=True)
    v3 = pd.read_csv(VER3, usecols=["network", "station", "pick_time"], dtype=str)
    n3 = len(v3)
    have = set(zip(v3.network, v3.station, v3.pick_time.str[:10]))
    del v3
    print(f"  {n3:,} picks on {len(have):,} station-days")

    print(f"copying ver3 -> {out} ...", flush=True)
    shutil.copyfile(VER3, out)

    added, next_id = [], n3
    skipped = {"in_ver3": 0, "empty": 0, "unreadable": 0}
    with open(out, "a", newline="") as fo:
        w = csv.writer(fo, lineterminator="\n")
        for source, kind, f in sources():
            if kind == "v2":   # NET_STA_YYYYMMDD_YYYYMMDD.csv: skip without reading
                net, sta, d = os.path.basename(f).split("_")[:3]
                if (net, sta, f"{d[:4]}-{d[4:6]}-{d[6:]}") in have:
                    skipped["in_ver3"] += 1
                    continue
            try:
                x = pd.read_csv(f, dtype=str, keep_default_na=False, na_values=[""])
            except Exception:
                skipped["unreadable"] += 1
                continue
            if len(x) == 0:
                skipped["empty"] += 1
                continue
            try:
                key = station_day(kind, x)
            except (AttributeError, IndexError):
                skipped["unreadable"] += 1
                continue
            if key in have:
                skipped["in_ver3"] += 1
                continue
            have.add(key)
            first = next_id
            for row in (rows_v2(x) if kind == "v2" else rows_v1(x)):
                w.writerow([next_id, next_id] + row + [next_id, f"{row[0]}.{row[1]}."])
                next_id += 1
            if next_id > first:
                added.append((category(source, f), source, *key, next_id - first, first, next_id - 1,
                              os.path.relpath(f, PICKDIR), file_date(f)))

    man = pd.DataFrame(added, columns=["category", "source", "network", "station", "day", "n_picks",
                                       "first_pick_id", "last_pick_id", "file", "fmtime"])
    man["after_ver3"] = man.fmtime > VER3_MERGED
    man.to_csv(args.out / "ver4_added_station_days.csv", index=False)
    man["year"] = man.day.str[:4]
    lines = [f"ver3 picks: {n3:,}", f"ver4 picks: {next_id:,}",
             f"added: {next_id - n3:,} picks on {len(man):,} station-days",
             f"per-day files skipped: {skipped}", "",
             "added picks by source and year:",
             man.pivot_table(index="source", columns="year", values="n_picks", aggfunc="sum",
                             fill_value=0).to_string(), "",
             "added station-days by station (top 40):",
             man.groupby(["network", "station", "source"]).agg(days=("day", "size"), picks=("n_picks", "sum"))
                .sort_values("picks", ascending=False).head(40).to_string()]
    (args.out / "ver4_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

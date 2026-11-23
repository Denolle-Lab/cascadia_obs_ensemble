#!/usr/bin/env python3
"""Catalog v4: Ian McBrearty's final GENIE + GraphDD catalog on the ver6 picks, with one fix.

Input (as received, 2026-11-21, not modified):
    data/Cascadia_relocated_file_ver_1.csv   172,340 events
    data/Cascadia_picks_ver_1.csv            2,503,189 associated picks
Output, same format byte for byte except the fixed dates:
    data/catalog_v4/Cascadia_relocated_catalog_v4.csv
    data/catalog_v4/Cascadia_picks_catalog_v4.csv
    data/catalog_v4/FIXES.md

Fix: 21 events (339 picks) are dated one day early. Each has its origin a few seconds before
midnight and its picks in the first 5-12 s of the next day. Their picks match no ver6 pick;
shifted by +1 day, all 339 match a ver6 pick exactly (audit of 2026-11-21). The origin time
and every pick time of these events are moved one day later; travel times are unchanged.
The other 2,502,850 picks already match ver6 exactly.

    python utils/fix_catalog_v4.py
"""
from datetime import datetime, timedelta
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
EVENTS_IN = DATA / "Cascadia_relocated_file_ver_1.csv"
PICKS_IN = DATA / "Cascadia_picks_ver_1.csv"
OUT = DATA / "catalog_v4"
EVENTS_OUT = OUT / "Cascadia_relocated_catalog_v4.csv"
PICKS_OUT = OUT / "Cascadia_picks_catalog_v4.csv"

SHIFT = {2877, 18916, 26927, 28027, 44216, 45770, 54911, 71159, 74839, 78814, 79142, 79237,
         82525, 82760, 82950, 83004, 106164, 106234, 117862, 149294, 171280}
N_PICKS_SHIFTED = 339
FMT = "%Y-%m-%dT%H:%M:%S.%fZ"
SEP = ", "


def plus_one_day(s):
    return (datetime.strptime(s, FMT) + timedelta(days=1)).strftime(FMT)


def rewrite(src, dst, id_col, time_col):
    """Copy src to dst, shifting the time field of the rows whose id is in SHIFT."""
    n = 0
    with open(src) as fi, open(dst, "w") as fo:
        fo.write(fi.readline())
        for line in fi:
            f = line.rstrip("\n").split(SEP)
            if int(f[id_col]) in SHIFT:
                f[time_col] = plus_one_day(f[time_col])
                line = SEP.join(f) + "\n"
                n += 1
            fo.write(line)
    return n


def main():
    OUT.mkdir(exist_ok=True)
    ne = rewrite(EVENTS_IN, EVENTS_OUT, 0, 2)          # ID, raw_ID, datetime, ...
    npk = rewrite(PICKS_IN, PICKS_OUT, 0, 2)           # event_ID, raw_event_ID, pick_datetime, ...
    assert ne == len(SHIFT) and npk == N_PICKS_SHIFTED, (ne, npk)
    (OUT / "FIXES.md").write_text(
        "# Catalog v4: changes from the files as received\n\n"
        f"Source: `{EVENTS_IN.name}`, `{PICKS_IN.name}` (I. McBrearty, received 2026-11-21).\n"
        "Built with `utils/fix_catalog_v4.py`.\n\n"
        f"- {ne} events and their {npk} picks moved one day later (origin time and pick times; "
        "travel times unchanged). They were dated one day early: origin a few seconds before "
        "midnight, picks just after it. After the shift every one of their picks matches a ver6 "
        "pick exactly.\n"
        f"  Event IDs: {', '.join(map(str, sorted(SHIFT)))}.\n\n"
        "Nothing else is changed. The format is still that of the files as received; the "
        "conversion to the origins/arrivals/picks standard of the v3 record comes later.\n")
    print(f"shifted {ne} events and {npk} picks -> {OUT}")


if __name__ == "__main__":
    main()

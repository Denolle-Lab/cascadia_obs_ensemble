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

Patch v4.1 (2026-11-22, data/catalog_v4/fixes/m66_2016/, branch fix/m66-2016): the 2016 offshore
events that v4 missed are added and one spurious event is removed.
  - 3 events added, IDs 900001-900003: Mw 6.6 of 2016-12-08 (us20007z6r) and the Gorda pair of
    2016-01-30 (M 5.1 us20004vq3, M 5.0 us20004vq9). ComCat hypocentre and magnitude, our ELEP
    repicks associated to the ComCat origin (283 picks). raw_ID and detection_value are nan;
    spatial_sigma(m) is the distance between our NLLoc 1D epicentre and ComCat.
  - Event 172114 (the M 4.7 aftershock, us20007z7b) keeps its v4 hypocentre and gains the repicks on
    station-phases it did not have; their travel times are taken from the v4 origin time.
  - Event 172113 and its 5 picks are removed: its P picks are the Mw 6.6 Pn at 708-797 km.
  - cascadia_catalog_M_routeA.csv: 172113 removed; 900001-900003 added with M = ComCat Mww.

    python utils/fix_catalog_v4.py [--data DIR] [--no-patch]
"""
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PATCH = REPO / "data" / "catalog_v4" / "fixes" / "m66_2016"
REMOVE = {172113}

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


def t(s):
    return datetime.strptime(s.strip(), FMT)


def f4(x):
    return "nan" if pd.isna(x) or x == "" else f"{float(x):.4f}"


def patch_m66(events_f, picks_f, mag_f):
    """Patch v4.1: drop REMOVE, add the 900001-900003 rows and picks, add repicks to 172114."""
    cat = pd.read_csv(PATCH / "catalog_v4_patch_2016.csv", dtype={"raw_ID": str})
    new = pd.read_csv(PATCH / "picks_v4_patch_2016.csv")
    ev_lines = events_f.read_text().splitlines(keepends=True)
    pk_lines = picks_f.read_text().splitlines(keepends=True)
    head_e, ev = ev_lines[0], [l for l in ev_lines[1:] if int(l.split(SEP, 1)[0]) not in REMOVE]
    head_p = pk_lines[0]
    pk = [l for l in pk_lines[1:] if int(l.split(SEP, 1)[0]) not in REMOVE]
    n_rm_e, n_rm_p = len(ev_lines) - 1 - len(ev), len(pk_lines) - 1 - len(pk)

    # 172114: v4 hypocentre kept; repicks on station-phases it lacks, travel time from the v4 origin
    old_ids = set(new.event_ID) - set(cat.ID)
    added = {}
    for eid in old_ids:
        row = next(l for l in ev if int(l.split(SEP, 1)[0]) == eid).rstrip("\n").split(SEP)
        ot, raw = t(row[2]), row[1]
        have = {(l.split(SEP)[4], l.split(SEP)[5]) for l in pk if int(l.split(SEP, 1)[0]) == eid}
        g = new[(new.event_ID == eid) & ~new.apply(lambda r: (r.station, r.phase) in have, axis=1)]
        rows = [SEP.join([str(eid), raw, r.pick_datetime, f"{(t(r.pick_datetime) - ot).total_seconds():.4f}",
                          r.station, r.phase, "nan", f4(r.pick_prob)]) + "\n" for r in g.itertuples()]
        last = max(i for i, l in enumerate(pk) if int(l.split(SEP, 1)[0]) == eid)
        pk[last + 1:last + 1] = rows
        row[7] = str(int(row[7]) + int((g.phase == "P").sum()))
        row[8] = str(int(row[8]) + int((g.phase == "S").sum()))
        ev = [SEP.join(row) + "\n" if int(l.split(SEP, 1)[0]) == eid else l for l in ev]
        added[eid] = (int((g.phase == "P").sum()), int((g.phase == "S").sum()))

    for r in cat.itertuples():
        ev.append(SEP.join([str(r.ID), "nan", r.datetime, f4(r.latitude), f4(r.longitude), f4(r._6), f4(r.magnitude),
                            str(r.Num_p_picks), str(r.Num_s_picks), f4(r._10), "nan"]) + "\n")
        for p in new[new.event_ID == r.ID].itertuples():
            pk.append(SEP.join([str(r.ID), "nan", p.pick_datetime, f4(p._4), p.station, p.phase, "nan", f4(p.pick_prob)]) + "\n")
    events_f.write_text(head_e + "".join(ev))
    picks_f.write_text(head_p + "".join(pk))

    # preferred magnitudes (Route A table): drop REMOVE, add the new events as ComCat Mww
    m = pd.read_csv(mag_f)
    m = m[~m.event_id.isin(REMOVE | set(cat.ID))]
    add = pd.DataFrame({"event_id": cat.ID, "otime": pd.to_datetime(cat.datetime).astype(str), "evla": cat.latitude,
                        "evlo": cat.longitude, "evdp": cat["depth(km)"], "M": cat.magnitude, "M_unc": 0.1, "M_type": "Mw",
                        "Mw_mt": cat.magnitude, "comcat_id": cat.location_source.str.extract(r"ComCat (\S+)")[0]})
    pd.concat([m, add], ignore_index=True).to_csv(mag_f, index=False)
    return n_rm_e, n_rm_p, cat, new, added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=REPO / "data", help="folder with the ver_1 inputs (shared data tree)")
    ap.add_argument("--no-patch", action="store_true", help="build v4.0 only (no m66_2016 patch)")
    a = ap.parse_args()
    DATA = a.data
    EVENTS_IN, PICKS_IN = DATA / "Cascadia_relocated_file_ver_1.csv", DATA / "Cascadia_picks_ver_1.csv"
    OUT = DATA / "catalog_v4"
    EVENTS_OUT, PICKS_OUT = OUT / "Cascadia_relocated_catalog_v4.csv", OUT / "Cascadia_picks_catalog_v4.csv"
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
    if a.no_patch:
        return
    mag_f = DATA / "magnitude_v4" / "cascadia_catalog_M_routeA.csv"
    n_rm_e, n_rm_p, cat, new, added = patch_m66(EVENTS_OUT, PICKS_OUT, mag_f)
    n_new = int(new.event_ID.isin(cat.ID).sum())
    rows = "\n".join(f"  - {r.ID}: {r.datetime[:19]}, {r.latitude:.4f}, {r.longitude:.4f}, {r._6} km, M {r.magnitude}, "
                     f"{r.Num_p_picks} P + {r.Num_s_picks} S; {r.location_source}" for r in cat.itertuples())
    add = "; ".join(f"{k}: +{p} P, +{s} S" for k, (p, s) in added.items())
    with open(OUT / "FIXES.md", "a") as fo:
        fo.write(
            "\n## v4.1 (2026-11-22): 2016 offshore events\n\n"
            "Files in `fixes/m66_2016/` (README there). The 2016 Mw 6.6 Mendocino earthquake and three other "
            "M 4.7-5.1 offshore events of 2016 were missing or poorly picked in v4, because ver6 has no NC/BK "
            "picks in 2016. Their waveforms were downloaded again, repicked with the same ELEP ensemble and "
            "associated to the ComCat origins. Our NLLoc epicentres are biased west (no station west of "
            "the events, nearest at 156 km) and their depths are unresolved, so the added rows carry the "
            "ComCat hypocentre and magnitude.\n\n"
            f"- {len(cat)} events added ({n_new} picks; `raw_ID`, `detection_value` and `amplitude(mm)` are nan; "
            "`spatial_sigma(m)` is the distance from our NLLoc 1D epicentre to ComCat):\n" + rows + "\n"
            f"- Event 172114 (M 4.7 aftershock, us20007z7b) keeps its v4 hypocentre; repicks added on "
            f"station-phases it lacked ({add}), travel times from the v4 origin time.\n"
            f"- Event 172113 and its {n_rm_p} picks removed ({n_rm_e} event): its P picks are the Pn of the Mw 6.6 at "
            "708-797 km, within 2 s of iasp91.\n"
            "- `magnitude_v4/cascadia_catalog_M_routeA.csv`: 172113 removed, the 3 new events added as Mw (ComCat Mww).\n\n"
            "The added picks are not ver6 picks (no `pick_id`, no Route A amplitude).\n")
    print(f"patch v4.1: -{n_rm_e} event/-{n_rm_p} picks, +{len(cat)} events/+{n_new} picks, 172114 {add}")


if __name__ == "__main__":
    main()

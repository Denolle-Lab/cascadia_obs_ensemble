#!/usr/bin/env python3
"""Family-by-family account of a segment's swarm-share change between two catalog versions (default Mount St Helens,
v6.0 -> v6.1): which v6.0 swarm events are still swarm events in v6.1, and why the others are not.

  pb_swarm_change.py [--seg MSH] [--old v60 --new v61]

Event set as in pb_seismicity.py: segment events, tiers A-C, not flagged, not review P1, in the clustering analysis
set (above the regional Mc of that clustering run). Reads the events_segments tables written by pb_structure.py for
both versions and the two clusterings. Writes <new tables>/swarm_change_<seg>_<old>_<new>.csv (one row per old swarm
family) and prints the summary.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

W = Path("/wd1/mdenolle_data")
V = {"v60": dict(seg=W / "paper_plan/events_segments_v60.parquet", clu=W / "catalog_v6/v6.0_analysis/clusters",
                 cat=W / "catalog_v6/v6.0/cascadia_catalog_v6.0.csv"),
     "v61": dict(seg=W / "paper_plan/v6.1/events_segments_v61.parquet", clu=W / "catalog_v6/v6.1_analysis/clusters",
                 cat=W / "catalog_v6/v6.1/cascadia_catalog_v6.1.csv", out=W / "paper_plan/v6.1/tables")}
COLS = ["ID", "seg", "abc", "is_eq", "review_P1", "M", "tier"]


def load(tag):
    e = pd.read_parquet(V[tag]["seg"])[COLS]
    z = pd.read_csv(V[tag]["clu"] / f"events_zbz_{tag}.csv.gz", usecols=["event_id", "in_analysis", "family_id", "cls"])
    return e.merge(z.rename(columns={"event_id": "ID"}), on="ID", how="outer")


def analysis(a, seg):
    return a[a.seg.eq(seg) & a.abc.eq(True) & a.is_eq.eq(True) & a.review_P1.eq(False) & a.in_analysis.eq(True)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", default="MSH")
    ap.add_argument("--old", default="v60")
    ap.add_argument("--new", default="v61")
    a = ap.parse_args()
    o, n = load(a.old), load(a.new)
    flags = pd.read_csv(V[a.new]["cat"], usecols=["ID", "event_flag"], low_memory=False)
    analog = set(flags[flags.event_flag.fillna("").str.contains("analog_test_signal")].ID)
    so, sn = analysis(o, a.seg), analysis(n, a.seg)
    nn = n.set_index("ID")

    def fate(i):
        if i in analog:
            return "analog_test_signal flag"
        if i not in nn.index or pd.isna(nn.at[i, "seg"]):
            return "not located"
        r = nn.loc[i]
        if r.seg != a.seg:
            return "other segment"
        if not r.abc == True:               # noqa: E712  (numpy bools; NaN counts as False)
            return "tier D"
        if r.review_P1 == True:             # noqa: E712
            return "review P1"
        if not r.in_analysis == True:       # noqa: E712
            return "below Mc"
        return f"{a.new} {r.cls if isinstance(r.cls, str) else 'background'}"
    sw = so[so.cls.eq("swarm")].copy()
    sw["fate"] = [fate(i) for i in sw.ID]
    sw["family_new"] = [nn.at[i, "family_id"] if i in nn.index else None for i in sw.ID]
    fo = pd.read_csv(V[a.old]["clu"] / f"families_zbz_{a.old}.csv", usecols=["family_id", "start", "Mmax", "n", "type"])
    fn = pd.read_csv(V[a.new]["clu"] / f"families_zbz_{a.new}.csv", usecols=["family_id", "type", "n"])
    rows = []
    for fid, g in sw.groupby("family_id"):
        f = fo[fo.family_id == fid].iloc[0]
        new_f = sorted({int(x) for x in g.family_new.dropna() if x >= 0})
        rows.append(dict(family_old=int(fid), start=f.start, Mmax=f.Mmax, n_segment_analysis=len(g),
                         families_new=";".join(map(str, new_f)),
                         types_new=";".join(fn.set_index("family_id").loc[new_f, "type"].astype(str)) if new_f else "",
                         **g.fate.value_counts().to_dict()))
    t = pd.DataFrame(rows).fillna(0).sort_values("n_segment_analysis", ascending=False)
    out = V[a.new].get("out", W / "paper_plan")
    t.to_csv(Path(out) / f"swarm_change_{a.seg}_{a.old}_{a.new}.csv", index=False)
    sh = lambda s: 100 * s.cls.eq("swarm").mean()  # noqa: E731
    so_na = so[~so.ID.isin(analog)]
    print(f"{a.seg}: swarm share {a.old} {sh(so):.1f} % ({so.cls.eq('swarm').sum()} / {len(so)}), "
          f"{a.new} {sh(sn):.1f} % ({sn.cls.eq('swarm').sum()} / {len(sn)}); "
          f"{a.old} without the {a.new} analog flags {sh(so_na):.1f} % ({so_na.cls.eq('swarm').sum()} / {len(so_na)})")
    print("fate of the", len(sw), f"{a.old} swarm events:", sw.fate.value_counts().to_dict())
    left = so[~so.ID.isin(sn.ID)]
    print(f"left the {a.new} analysis set:", pd.Series([fate(i) for i in left.ID]).value_counts().to_dict(),
          f"of {len(left)}; entered: {int((~sn.ID.isin(so.ID)).sum())}")
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()

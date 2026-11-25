"""Short-lived space-time clusters (swarm scan of v4_swarms.py) in catalog v5.1, compared with v4.

Same DBSCAN-style scan and parameters as v4_swarms.py (functions imported from it): epicentres in km plus time
scaled at 2 km/day, link distance 3 km, core events with >= 5 links, clusters of >= 15 events with a 5-95 %
duration of at most 30 days, each compared with the rate within 10 km over the 90 days before and after.
Events: the located earthquakes of catalog v5.1 (no rows without a hypocentre, no explosions or quarry
blasts), all magnitudes. n_qc counts tier A-C events (v4: events with >= 4 P and >= 4 S picks).

Added for v5.1: depth range from well-constrained depths only (depth_good, see v51_clusters_zbz.py), tier
mix, migration and sqrt(t) diffusion fit from the tier A-C members, overlap with the v5.1 nearest-neighbour
families, and the match with the v4 clusters (kept, split, merged, new, dissolved) by shared event IDs.

Outputs in data/catalog_v5/v5.1_analysis/clusters/: dbscan_clusters_v51.csv, dbscan_members_v51.csv,
dbscan_v4_status.csv, dbscan_v51_status.csv.
Run after v51_clusters_zbz.py (step main):  python v51_swarms.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v4_swarms as S  # noqa: E402
import v51_clusters_zbz as Z  # noqa: E402

V4S = Z.ROOT / "data/catalog_v4/swarms"


def load() -> pd.DataFrame:
    df = Z.prepare(Z.load_all())
    df["latitude"], df["longitude"] = df.lat, df.lon
    df["qc"] = df.tierABC
    df["x"] = (df.longitude - S.LON0) * S.KX
    df["y"] = (df.latitude - S.LAT0) * S.KY
    df["day"] = (df.time - Z.T0) / pd.Timedelta(days=1)
    return df.sort_values("time", kind="stable").reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eps", type=float, default=3.0)
    ap.add_argument("--kmday", type=float, default=2.0)
    ap.add_argument("--core", type=int, default=5)
    ap.add_argument("--min-n", type=int, default=15)
    ap.add_argument("--max-dur", type=float, default=30.0)
    ap.add_argument("--rad", type=float, default=10.0)
    ap.add_argument("--bg", type=float, default=90.0)
    a = ap.parse_args(argv)
    out = Z.OUT
    c = load()
    lab = S.cluster(c, a.eps, a.kmday, a.core)
    t = S.describe(c, lab, a.rad, a.bg, a.min_n, a.max_dur)
    keep = np.isin(lab, t["cluster"].to_numpy())
    mem = pd.DataFrame({"event_id": c["event_id"][keep].values, "cluster": lab[keep]})
    # extra columns per cluster
    ev = pd.read_csv(Z.OUT / "events_zbz_v51.csv.gz", usecols=["event_id", "family_id", "family_type", "in_analysis"])
    fam = pd.read_csv(Z.OUT / "families_zbz_v51.csv", usecols=["family_id", "type", "n", "n_all"])
    cc = c.set_index("event_id")
    rows = []
    for k, g in mem.groupby("cluster"):
        e = cc.loc[g.event_id].reset_index().sort_values("t_yr")
        z = e.depth[e.depth_good]
        d = dict(cluster=k, n_tierD=int((e.tier == "D").sum()), n_depth_good=len(z),
                 depth_good_med=float(z.median()) if len(z) else np.nan,
                 depth_good_q10=float(z.quantile(0.1)) if len(z) else np.nan,
                 depth_good_q90=float(z.quantile(0.9)) if len(z) else np.nan,
                 region_v51=e.region_v51.mode().iloc[0], box=e.region.mode().iloc[0],
                 Mmin=float(e.M.min()) if e.M.notna().any() else np.nan)
        d.update(Z.migration_details(e[e.tierABC]))
        z5 = g.merge(ev, on="event_id", how="left")
        fc = z5.family_id[z5.family_id > 0].value_counts()
        d.update(n_above_Mc=int(z5.in_analysis.sum()), frac_in_zbz_families=round(float((z5.family_id > 0).mean()), 3),
                 n_zbz_families=len(fc), main_zbz_family=int(fc.index[0]) if len(fc) else -1,
                 frac_in_main_zbz_family=round(fc.iloc[0] / len(g), 3) if len(fc) else 0.0)
        rows.append(d)
    x = pd.DataFrame(rows)
    t = t.merge(x, on="cluster", how="left")
    t["main_zbz_type"] = t.main_zbz_family.map(fam.set_index("family_id").type).fillna("background")
    t.to_csv(out / "dbscan_clusters_v51.csv", index=False, float_format="%.4f")
    mem.to_csv(out / "dbscan_members_v51.csv", index=False)
    # comparison with the v4 scan
    m4 = pd.read_csv(V4S / "v4_cluster_members.csv").rename(columns={"cluster": "f4"})
    m5 = mem.rename(columns={"cluster": "f5"})
    t4 = pd.read_csv(V4S / "v4_clusters.csv")
    ov, r4, r5 = Z.match_families(m4, m5, None, None)
    r4 = r4.merge(t4[["cluster", "region", "volcano", "n", "start", "dur_days", "swarm_like"]].rename(
        columns={"cluster": "f4"}), on="f4", how="left")
    r5 = r5.merge(t[["cluster", "region", "volcano", "n", "start", "dur_days", "swarm_like"]].rename(
        columns={"cluster": "f5"}), on="f5", how="left")
    # fate of the events of dissolved v4 clusters
    exc = Z.load_all().set_index("event_id").excluded
    dm = m4[m4.f4.isin(r4[r4.status == "dissolved"].f4)].copy()
    # size and 5-95 % duration of every DBSCAN label, to say why a v4 cluster has no v5.1 counterpart
    lb = pd.DataFrame({"event_id": c.event_id.values, "lab": lab, "day": c.day.values})
    lb = lb[lb.lab >= 0]
    st = lb.groupby("lab").day.agg(n="size", dur=lambda d: np.quantile(d, 0.95) - np.quantile(d, 0.05))
    why_lab = pd.Series(np.where(st.n < a.min_n, f"in a cluster of < {a.min_n} events",
                                 np.where(st.dur > a.max_dur, f"in a cluster lasting > {a.max_dur:.0f} days",
                                          "in a kept cluster")), index=st.index)
    lb["why"] = lb.lab.map(why_lab)
    why = lb.set_index("event_id").why
    fate = dm.event_id.map(exc).fillna("not in v5.1")
    dm["fate"] = np.where(fate.ne(""), fate, dm.event_id.map(why).fillna("not clustered"))
    r4 = r4.merge(dm.groupby("f4").fate.agg(lambda s: ";".join(f"{k}:{v}" for k, v in s.value_counts().items()))
                  .rename("dissolved_member_fate"), on="f4", how="left")
    r4.to_csv(out / "dbscan_v4_status.csv", index=False, float_format="%.3f")
    r5.to_csv(out / "dbscan_v51_status.csv", index=False, float_format="%.3f")
    print(f"{len(c):,} events -> {len(t)} clusters; {t.n.sum():,} events in them; "
          f"swarm-like {int((t.swarm_like == True).sum())}")  # noqa: E712
    print(t.groupby("region").agg(n_clusters=("n", "size"), events=("n", "sum"),
                                  swarm_like=("swarm_like", lambda s: int((s == True).sum()))).to_string())  # noqa: E712
    print("v4 clusters:", r4.status.value_counts().to_dict())
    print("v5.1 clusters:", r5.status.value_counts().to_dict())


if __name__ == "__main__":
    main()

"""Short-lived space-time clusters (swarm scan of v4_swarms.py) in catalog v5.1, compared with v4.

Same DBSCAN-style scan and parameters as v4_swarms.py (functions imported from it): epicentres in km plus time
scaled at 2 km/day, link distance 3 km, core events with >= 5 links, clusters of >= 15 events with a 5-95 %
duration of at most 30 days, each compared with the rate within 10 km over the 90 days before and after.
Events: the located earthquakes of catalog v5.1 (no rows without a hypocentre, no explosions or quarry
blasts), all magnitudes. n_qc counts tier A-C events (v4: events with >= 4 P and >= 4 S picks).

Added for v5.1: depth range from well-constrained depths only (depth_good, see v51_clusters_zbz.py; the v4
columns depth_med / depth_iqr use every depth and are renamed depth_any_quality_med / _iqr), tier
mix, migration and sqrt(t) diffusion fit from the tier A-C members (permutation p-value, mig_p_perm), overlap
with the v5.1 nearest-neighbour families, and the match with the v4 clusters (kept, split, merged, new,
dissolved) by shared event IDs.

Outputs in the clustering folder of the catalog version (v51_clusters_zbz.configure): dbscan_clusters_<tag>.csv,
dbscan_members_<tag>.csv, and per comparison run <ref>: dbscan_<ref>_status.csv and dbscan_<tag>_status.csv (first
comparison run) or dbscan_<tag>_status_vs_<ref>.csv. v5.1 (--catalog v5.1 --mag stored): the v4 scan of the stored
magnitudes (frozen copy); v6.0: the v5.1 scan (primary) and the v4 scan on main. The scan does not use magnitudes
(only the descriptions do), so these comparisons show the location change.
Run after v51_clusters_zbz.py (step main):  python v51_swarms.py [--catalog v6.0]
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


def swarm_refs() -> dict:
    """Reference DBSCAN scans: key -> (members file, clusters file); the cluster column is 'cluster' in both."""
    out = {}
    for key, R in Z.REFS.items():
        d = R.get("swarms")
        if d is None:
            continue
        d = Path(d)
        if (d / "v4_cluster_members.csv").exists():
            out[key] = (d / "v4_cluster_members.csv", d / "v4_clusters.csv")
        elif (d / f"dbscan_members_{R.get('swarms_tag', 'v51')}.csv").exists():   # v51; v6.x refs name their tag
            t = R.get("swarms_tag", "v51")
            out[key] = (d / f"dbscan_members_{t}.csv", d / f"dbscan_clusters_{t}.csv")
    return out


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
    ap.add_argument("--catalog", default="v5.1", choices=sorted(Z.CATALOGS))
    ap.add_argument("--mag", default=None, help="v5.1 only: as v51_clusters_zbz.py (stored = the committed run)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--big", default=None)
    a = ap.parse_args(argv)
    Z.configure(a.catalog, a.mag, a.out, a.big)
    out, tag = Z.OUT, Z.TAG
    c = load()
    lab = S.cluster(c, a.eps, a.kmday, a.core)
    t = S.describe(c, lab, a.rad, a.bg, a.min_n, a.max_dur)
    # v4_swarms.describe takes depth_med / depth_iqr over every member, whatever its depth quality
    # (floor_limited, far_only, deep_pass included): labelled here; use depth_good_* for depths
    t = t.rename(columns={"depth_med": "depth_any_quality_med", "depth_iqr": "depth_any_quality_iqr"})
    keep = np.isin(lab, t["cluster"].to_numpy())
    mem = pd.DataFrame({"event_id": c["event_id"][keep].values, "cluster": lab[keep]})
    # extra columns per cluster
    ev = pd.read_csv(Z.OUT / Z.EVF, usecols=["event_id", "family_id", "family_type", "in_analysis"])
    fam = pd.read_csv(Z.OUT / Z.FAMF, usecols=["family_id", "type", "n", "n_all"])
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
    t.to_csv(out / f"dbscan_clusters_{tag}.csv", index=False, float_format="%.4f")
    mem.to_csv(out / f"dbscan_members_{tag}.csv", index=False)
    # size and 5-95 % duration of every DBSCAN label, to say why a reference cluster has no counterpart
    lb = pd.DataFrame({"event_id": c.event_id.values, "lab": lab, "day": c.day.values})
    lb = lb[lb.lab >= 0]
    st = lb.groupby("lab").day.agg(n="size", dur=lambda d: np.quantile(d, 0.95) - np.quantile(d, 0.05))
    why_lab = pd.Series(np.where(st.n < a.min_n, f"in a cluster of < {a.min_n} events",
                                 np.where(st.dur > a.max_dur, f"in a cluster lasting > {a.max_dur:.0f} days",
                                          "in a kept cluster")), index=st.index)
    lb["why"] = lb.lab.map(why_lab)
    why = lb.set_index("event_id").why
    exc = Z.load_all().set_index("event_id").excluded
    status = {}
    for k_ref, (ref, (fm, fc)) in enumerate(swarm_refs().items()):
        sfx = "" if k_ref == 0 else f"_vs_{ref}"
        m4 = pd.read_csv(fm).rename(columns={"cluster": "f4"})
        m5 = mem.rename(columns={"cluster": "f5"})
        t4 = pd.read_csv(fc)
        ov, r4, r5 = Z.match_families(m4, m5, None, None, ref=ref, new=tag)
        r4 = r4.merge(t4[["cluster", "region", "volcano", "n", "start", "dur_days", "swarm_like"]].rename(
            columns={"cluster": "f4"}), on="f4", how="left")
        r5 = r5.merge(t[["cluster", "region", "volcano", "n", "start", "dur_days", "swarm_like"]].rename(
            columns={"cluster": "f5"}), on="f5", how="left")
        # fate of the events of dissolved reference clusters
        dm = m4[m4.f4.isin(r4[r4.status == "dissolved"].f4)].copy()
        fate = dm.event_id.map(exc).fillna(f"not in {Z.CATV}")
        dm["fate"] = np.where(fate.ne(""), fate, dm.event_id.map(why).fillna("not clustered"))
        r4 = r4.merge(dm.groupby("f4").fate.agg(lambda s: ";".join(f"{k}:{v}" for k, v in s.value_counts().items()))
                      .rename("dissolved_member_fate"), on="f4", how="left")
        r4.to_csv(out / f"dbscan_{ref}_status.csv", index=False, float_format="%.3f")
        r5.to_csv(out / f"dbscan_{tag}_status{sfx}.csv", index=False, float_format="%.3f")
        status[ref] = (r4, r5)
    print(f"{len(c):,} events -> {len(t)} clusters; {t.n.sum():,} events in them; "
          f"swarm-like {int((t.swarm_like == True).sum())}")  # noqa: E712
    print(t.groupby("region").agg(n_clusters=("n", "size"), events=("n", "sum"),
                                  swarm_like=("swarm_like", lambda s: int((s == True).sum()))).to_string())  # noqa: E712
    for ref, (r4, r5) in status.items():
        print(f"{ref} clusters:", r4.status.value_counts().to_dict())
        print(f"{Z.CATV} clusters vs {ref}:", r5.status.value_counts().to_dict())


if __name__ == "__main__":
    main()

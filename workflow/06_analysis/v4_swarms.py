"""Short-lived event clusters (swarm candidates) in catalog v4.

Clusters are built in space-time, not depth (v4 depths are poorly constrained):
x, y in km plus time scaled by --kmday (km per day). Two events are linked when
they are within --eps in that space. Core events have at least --core links;
clusters are connected core events plus their linked border events (DBSCAN
without sklearn).

Network coverage changes over months to years. To keep that from creating
"swarms", each cluster is compared with the rate within --rad km of its
centroid over the --bg days before and after it (cluster events excluded).

Swarm-like vs mainshock-aftershock: a cluster is swarm-like when its largest
event is not in the first 10% of events and the two largest magnitudes differ
by less than 0.5 (Vidale & Shearer 2006 style criteria, simplified).

Output: data/catalog_v4/swarms/v4_clusters.csv (one row per cluster) and
v4_cluster_members.csv (event_id, cluster).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.stats import theilslopes

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data"
sys.path.append(str(ROOT / "workflow" / "05_magnitude"))
import mag_paths  # noqa: E402  (converged Route A magnitudes by default, D-B091)

MAG = mag_paths.v41()        # --mag / $CASCADIA_MAG_V41: converged (default), stored or a path
LON0, LAT0 = -125.25, 45.0
KX = 111.195 * np.cos(np.radians(LAT0))
KY = 111.195
# most specific first (the first box that holds the centroid names the cluster)
REGIONS = [
    ("Axial Seamount", [-130.35, -129.75, 45.75, 46.2]),
    ("Mendocino triple junction", [-125.6, -123.2, 39.6, 41.4]),
    ("Puget Lowland", [-123.6, -121.4, 46.6, 48.6]),
    ("Endeavour - Explorer", [-130.4, -127.2, 47.4, 49.8]),
    ("Blanco transform", [-130.6, -126.6, 42.6, 45.3]),
    ("Gorda plate", [-128.8, -123.8, 40.0, 43.2]),
    ("Offshore Washington", [-127.6, -123.2, 46.2, 49.2]),
    ("Offshore Oregon", [-126.8, -123.2, 42.8, 46.4]),
]

# Cascade volcanoes (summit lon, lat), for tagging onshore clusters
VOLCANOES = {
    "Mount Baker": (-121.813, 48.777), "Glacier Peak": (-121.113, 48.112), "Mount Rainier": (-121.760, 46.853),
    "Mount St. Helens": (-122.180, 46.200), "Mount Adams": (-121.490, 46.206), "Mount Hood": (-121.695, 45.374),
    "Mount Jefferson": (-121.800, 44.674), "Three Sisters": (-121.768, 44.103), "Newberry": (-121.229, 43.722),
    "Crater Lake": (-122.110, 42.940), "Medicine Lake": (-121.550, 41.590), "Mount Shasta": (-122.193, 41.409),
    "Lassen Peak": (-121.505, 40.492),
}


def nearest_volcano(lon: float, lat: float) -> tuple[str, float]:
    d = {k: float(np.hypot((lon - v[0]) * KX, (lat - v[1]) * KY)) for k, v in VOLCANOES.items()}
    k = min(d, key=d.get)
    return k, d[k]


def region_of(lon: float, lat: float) -> str:
    for name, (w, e, s, n) in REGIONS:
        if w <= lon <= e and s <= lat <= n:
            return name
    return "Other"


def load() -> pd.DataFrame:
    c = pd.read_csv(D / "catalog_v4" / "Cascadia_relocated_catalog_v4.csv", skipinitialspace=True)
    c.columns = [x.strip() for x in c.columns]
    c = c.rename(columns={"ID": "event_id", "depth(km)": "depth", "spatial_sigma(m)": "sigma_m"})
    c["time"] = pd.to_datetime(c["datetime"].str.strip(), utc=True)
    m = pd.read_csv(MAG, usecols=["event_id", "M", "M_type"])
    c = c.merge(m, on="event_id", how="left")
    c["qc"] = (c["Num_p_picks"] >= 4) & (c["Num_s_picks"] >= 4)
    c["x"] = (c["longitude"] - LON0) * KX
    c["y"] = (c["latitude"] - LAT0) * KY
    c["day"] = (c["time"] - pd.Timestamp("2010-01-01", tz="UTC")) / pd.Timedelta(days=1)
    return c.sort_values("time").reset_index(drop=True)


def cluster(c: pd.DataFrame, eps: float, kmday: float, core: int) -> np.ndarray:
    P = np.c_[c["x"], c["y"], c["day"] * kmday]
    t = cKDTree(P)
    pairs = t.query_pairs(eps, output_type="ndarray")
    n = len(c)
    deg = np.bincount(pairs.ravel(), minlength=n)
    is_core = deg >= core
    cc = pairs[is_core[pairs[:, 0]] & is_core[pairs[:, 1]]]
    g = coo_matrix((np.ones(len(cc)), (cc[:, 0], cc[:, 1])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    lab = np.where(is_core, lab, -1)
    # border events join the cluster of a linked core event
    for a, b in pairs:
        if lab[a] < 0 and is_core[b]:
            lab[a] = lab[b]
        elif lab[b] < 0 and is_core[a]:
            lab[b] = lab[a]
    _, lab2 = np.unique(lab, return_inverse=True)
    return np.where(lab < 0, -1, lab2 - (1 if (lab < 0).any() else 0))


def describe(c: pd.DataFrame, lab: np.ndarray, rad: float, bg: float, min_n: int, max_dur: float) -> pd.DataFrame:
    tree = cKDTree(np.c_[c["x"], c["y"]])
    rows = []
    for k in np.unique(lab[lab >= 0]):
        idx = np.flatnonzero(lab == k)
        if len(idx) < min_n:
            continue
        e = c.iloc[idx]
        d = e["day"].to_numpy()
        d05, d95 = np.quantile(d, [0.05, 0.95])
        dur = max(d95 - d05, 0.1)
        if dur > max_dur:
            continue
        x0, y0 = e["x"].median(), e["y"].median()
        near = np.array(tree.query_ball_point([x0, y0], rad))
        dn = c["day"].to_numpy()[near]
        out = ~np.isin(near, idx)
        before = out & (dn >= d.min() - bg) & (dn < d.min())
        after = out & (dn > d.max()) & (dn <= d.max() + bg)
        span = min(bg, d.min() - 0) + min(bg, c["day"].max() - d.max())
        bg_rate = (before.sum() + after.sum()) / max(span, 1.0)          # events/day around the cluster
        ratio = (len(idx) / dur) / max(bg_rate, 1.0 / max(span, 1.0))
        M = e["M"].to_numpy()
        has_m = np.isfinite(M)
        if has_m.sum() >= 2:
            o = np.argsort(-np.where(has_m, M, -9))
            mmax, m2 = M[o[0]], M[o[1]]
            first_frac = (np.argsort(np.argsort(d))[o[0]]) / (len(idx) - 1)
            swarm = (mmax - m2 < 0.5) and (first_frac > 0.1)
        else:
            mmax = m2 = first_frac = np.nan
            swarm = np.nan
        # migration: distance from the first five events vs time
        xs, ys = e["x"].to_numpy(), e["y"].to_numpy()
        r = np.hypot(xs - np.median(xs[:5]), ys - np.median(ys[:5]))
        mig = theilslopes(r, d)[0] if len(idx) >= 10 and np.ptp(d) > 0 else np.nan
        lo, la = e["longitude"].median(), e["latitude"].median()
        vname, vdist = nearest_volcano(lo, la)
        rows.append(dict(
            cluster=int(k), region=region_of(lo, la),
            volcano=vname if vdist <= 30 else "", volcano_km=round(vdist, 1) if vdist <= 30 else np.nan,
            n=len(idx), n_qc=int(e["qc"].sum()),
            start=e["time"].min().strftime("%Y-%m-%d %H:%M"), end=e["time"].max().strftime("%Y-%m-%d %H:%M"),
            dur_days=round(dur, 2), lon=round(e["longitude"].median(), 4), lat=round(e["latitude"].median(), 4),
            depth_med=round(e["depth"].median(), 1), depth_iqr=round(e["depth"].quantile(.75) - e["depth"].quantile(.25), 1),
            extent_km=round(float(np.hypot(np.ptp(xs), np.ptp(ys))), 1),
            Mmax=round(mmax, 2) if np.isfinite(mmax) else np.nan, M2=round(m2, 2) if np.isfinite(m2) else np.nan,
            Mmax_pos=round(first_frac, 2) if np.isfinite(first_frac) else np.nan,
            n_with_M=int(has_m.sum()), swarm_like=swarm,
            bg_rate_per_day=round(bg_rate, 3), rate_ratio=round(ratio, 1),
            migration_km_per_day=round(mig, 2) if np.isfinite(mig) else np.nan,
        ))
    return pd.DataFrame(rows).sort_values(["rate_ratio"], ascending=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eps", type=float, default=3.0, help="link distance in km (space-time)")
    ap.add_argument("--kmday", type=float, default=2.0, help="km per day for the time axis")
    ap.add_argument("--core", type=int, default=5, help="links needed for a core event")
    ap.add_argument("--min-n", type=int, default=15)
    ap.add_argument("--max-dur", type=float, default=30.0, help="max 5-95%% duration in days")
    ap.add_argument("--rad", type=float, default=10.0, help="radius (km) for the background rate")
    ap.add_argument("--bg", type=float, default=90.0, help="days before/after for the background rate")
    ap.add_argument("--out", default=str(D / "catalog_v4" / "swarms"))
    mag_paths.add_argument(ap, "v4.1")
    a = ap.parse_args(argv)
    global MAG
    MAG = mag_paths.v41(a.mag)
    print(f"magnitudes: {MAG}", flush=True)
    c = load()
    lab = cluster(c, a.eps, a.kmday, a.core)
    t = describe(c, lab, a.rad, a.bg, a.min_n, a.max_dur)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "v4_clusters.csv", index=False)
    keep = np.isin(lab, t["cluster"].to_numpy())
    pd.DataFrame({"event_id": c["event_id"][keep], "cluster": lab[keep]}).to_csv(out / "v4_cluster_members.csv", index=False)
    print(f"{len(c):,} events -> {len(t)} clusters (n>={a.min_n}, <= {a.max_dur} d); "
          f"{t['n'].sum():,} events in them; swarm-like {int((t['swarm_like'] == True).sum())}")
    print(t.groupby("region").agg(n_clusters=("n", "size"), events=("n", "sum"),
                                  swarm_like=("swarm_like", lambda s: int((s == True).sum()))).to_string())
    pd.set_option("display.width", 250)
    print(t[t.volcano != ""][["volcano", "volcano_km", "n", "start", "dur_days", "depth_med", "Mmax", "swarm_like", "rate_ratio"]].to_string(index=False))
    print(t.head(25)[["region", "n", "n_qc", "start", "dur_days", "lat", "lon", "depth_med", "extent_km",
                      "Mmax", "M2", "Mmax_pos", "swarm_like", "rate_ratio", "migration_km_per_day"]].to_string(index=False))


if __name__ == "__main__":
    main()

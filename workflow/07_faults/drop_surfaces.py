"""Remove surfaces from the fault-surface product (all files), then recompute the assignment table.

    python workflow/07_faults/drop_surfaces.py --reason "quarry blasts" ON012 ON020 ...

Removed surfaces are listed in fault_surfaces_v0_removed.csv with the reason. Their events become
unassigned; surface ids of the others do not change. The map is redrawn from the vertex file.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

V0 = Path(__file__).resolve().parents[2] / "data" / "faults" / "v0"
STEM = "fault_surfaces_v0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--reason", required=True)
    a = ap.parse_args()
    drop = set(a.ids)
    summ = pd.read_csv(V0 / f"{STEM}_summary.csv")
    miss = drop - set(summ.surface_id)
    if miss:
        raise SystemExit(f"not in the product: {sorted(miss)}")
    rem = summ[summ.surface_id.isin(drop)][["surface_id", "n_events", "lon", "lat", "strike", "dip"]
                                           + [c for c in ("lit_class", "lit_match") if c in summ]].assign(reason=a.reason)
    rf = V0 / f"{STEM}_removed.csv"
    rem = pd.concat([pd.read_csv(rf), rem]) if rf.exists() else rem
    rem.to_csv(rf, index=False)
    summ[~summ.surface_id.isin(drop)].to_csv(V0 / f"{STEM}_summary.csv", index=False)
    for suf in ("vertices", "members", "literature", "traces"):
        f = V0 / f"{STEM}_{suf}.csv"
        if f.exists():
            d = pd.read_csv(f)
            d[~d.surface_id.isin(drop)].to_csv(f, index=False)
    f = V0 / f"{STEM}_ensemble.csv.gz"
    if f.exists():
        d = pd.read_csv(f)
        d[~d.surface_id.isin(drop)].to_csv(f, index=False, compression="gzip")
    g = json.load(open(V0 / f"{STEM}.geojson"))
    g["features"] = [x for x in g["features"] if x["properties"]["surface_id"] not in drop]
    (V0 / f"{STEM}.geojson").write_text(json.dumps(g))
    # assignment: events per region from the old table, assigned recounted from the members file
    asg = pd.read_csv(V0 / f"{STEM}_assignment.csv")
    mem = pd.read_csv(V0 / f"{STEM}_members.csv")
    n_by = mem.surface_id.str[:2].value_counts()
    asg["assigned"] = asg.region.map(n_by).fillna(0).astype(int)
    asg["fraction"] = (asg.assigned / asg.events).round(3)
    asg.to_csv(V0 / f"{STEM}_assignment.csv", index=False)
    print(f"removed {len(drop)}; {len(g['features'])} surfaces left; "
          f"{asg.assigned.sum():,} of {asg.events.sum():,} events on a fault ({asg.assigned.sum() / asg.events.sum():.0%})")
    figure()


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    vx = pd.read_csv(V0 / f"{STEM}_vertices.csv")
    g = json.load(open(V0 / f"{STEM}.geojson"))
    views = [("Whole margin", [-131, -119.5, 39, 51]), ("Blanco transform", [-130.6, -126.6, 42.6, 45.3]),
             ("Mendocino and Gorda", [-128.0, -123.2, 39.8, 42.5])]
    fig, axs = plt.subplots(1, 3, figsize=(17, 6.5), constrained_layout=True)
    polys, vals = [], []
    for f in g["features"]:
        for poly in f["geometry"]["coordinates"]:
            tri = np.array(poly[0][:3])
            polys.append(tri[:, :2]); vals.append(-tri[:, 2].mean() / 1000)
    for ax, (title, b) in zip(axs, views):
        ax.scatter(vx.lon, vx.lat, s=0.05, c="0.85", lw=0, rasterized=True)
        pc = PolyCollection(polys, array=np.array(vals), cmap="viridis_r", clim=(0, 30), edgecolors="none", alpha=0.85)
        ax.add_collection(pc)
        ax.set_xlim(b[0], b[1]); ax.set_ylim(b[2], b[3]); ax.set_aspect(1 / np.cos(np.radians((b[2] + b[3]) / 2)))
        ax.set_title(title, fontsize=10)
    fig.colorbar(pc, ax=axs, label="Surface depth (km, catalog datum)", shrink=0.7)
    fig.suptitle(f"Fault surfaces v0 from catalog v4 ({len(g['features'])} rectangular faults)", fontsize=11)
    fig.savefig(V0 / f"{STEM}_map.png", dpi=170)


if __name__ == "__main__":
    main()

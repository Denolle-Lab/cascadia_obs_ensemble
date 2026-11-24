"""B5 sanity check: the whole-catalog B5 solution (b5_relocate.py) against the D-B048 test solution (b5_test, tomoDD
variant dd) on the Blanco swarm cases f0399, f1803, f1051 (b5_test.cases). Same code, damping and variant; the
differences come from the differential-time graph (whole catalog: more neighbours, 3D distances on the sphere),
the cluster (component or chunk, more events), the centroid restoration (B5) and the frozen binary (D-B054).

Metrics on the events both relocated (common set): position differences with each solution's centroid removed
(horizontal, vertical; median and 90th percentile), median |difference| of event-pair distances, both solutions'
shift from B4, PCA of the common set (sigma1-3, strike, dip). The run's catalog may be partial (collect --partial).

  python b5_check_test.py RUN [--catalog PATH] [--cases f0399 ...]
Output: data/tomo_v1/b5/<RUN>/check_b5_test.md and its manifest.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from b5_common import KM_DEG, P, enu, pca, write_manifest
from manifest import REPO

B = P["b5"]
BT = P["b5_test"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--catalog")
    ap.add_argument("--cases", nargs="*", default=list(BT["cases"]))
    a = ap.parse_args()
    d = Path(B["out"]) / a.run
    cf = Path(a.catalog) if a.catalog else next(p for p in (d / f"catalog_b5_{a.run}.csv",
                                                             d / f"catalog_b5_{a.run}_partial.csv") if p.exists())
    cat = pd.read_csv(cf, usecols=["ID", "lat", "lon", "depth", "b4_lat", "b4_lon", "b4_depth", "loc_source",
                                   "b5_cluster"]).set_index("ID")
    L = [f"# B5 run {a.run} against the D-B048 test (tomoDD dd) on the Blanco cases", "",
         f"Catalog {cf}. Common set = events relocated by both. Differences in km; 'c.r.' = each solution's centroid "
         "removed.", "",
         "| case | test relocated | B5 relocated (of the case) | common | B5 clusters | pair | h diff c.r. med (p90) | "
         "z diff c.r. med (p90) | pair-distance diff med | centroid shift h / z |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    shapes, inputs = [], [cf]
    for case in a.cases:
        f = Path(BT["out"]) / case / "tomodd" / "dd" / "reloc.csv"
        if not f.exists():
            continue
        inputs.append(f)
        t = pd.read_csv(f)
        t = t[t.relocated].set_index("ID")
        ids_case = pd.read_csv(Path(BT["out"]) / case / "data" / "events.csv").ID
        b5 = cat.reindex(ids_case)
        b5r = b5[b5.loc_source == "b5_tomodd"]
        com = t.index.intersection(b5r.index)
        if len(com) < 3:
            continue
        T_, S_, B4 = t.loc[com], b5r.loc[com], cat.loc[com]
        lat0, lon0 = float(B4.b4_lat.median()), float(B4.b4_lon.median())
        xt = enu(T_.lat, T_.lon, T_.depth, lat0, lon0)
        xs = enu(S_.lat, S_.lon, S_.depth, lat0, lon0)
        x4 = enu(B4.b4_lat, B4.b4_lon, B4.b4_depth, lat0, lon0)
        i, j = np.triu_indices(len(com), 1)
        if len(i) > 2_000_000:
            k = np.random.default_rng(0).choice(len(i), 2_000_000, replace=False)
            i, j = i[k], j[k]
        head = f"| {case} | {len(t)} | {len(b5r)} of {len(ids_case)} | {len(com)} | {b5r.b5_cluster.nunique()} |"
        for name, X1, X0 in (("B5 - test", xs, xt), ("B5 - B4", xs, x4), ("test - B4", xt, x4)):
            c1, c0 = X1.mean(0), X0.mean(0)
            r = (X1 - c1) - (X0 - c0)
            dh, dz = np.hypot(r[:, 0], r[:, 1]), np.abs(r[:, 2])
            pdif = np.abs(np.linalg.norm(X1[i] - X1[j], axis=1) - np.linalg.norm(X0[i] - X0[j], axis=1))
            L.append(f"{head} {name} | {np.median(dh):.3f} ({np.quantile(dh, 0.9):.3f}) | "
                     f"{np.median(dz):.3f} ({np.quantile(dz, 0.9):.3f}) | {np.median(pdif):.3f} | "
                     f"{np.hypot(*(c1 - c0)[:2]):.3f} / {(c1 - c0)[2]:+.3f} |")
            head = "| | | | | |"
        for name, X in (("B4", x4), ("test dd", xt), ("B5", xs)):
            s, _, strike, dip = pca(X)
            shapes.append(f"| {case} | {name} | {s[0]:.2f} / {s[1]:.2f} / {s[2]:.2f} | {strike:.0f} / {dip:.0f} | "
                          f"{np.percentile(X[:, 2], 10):.1f} / {np.median(X[:, 2]):.1f} / "
                          f"{np.percentile(X[:, 2], 90):.1f} |")
    L += ["", "Shape of the common set:", "", "| case | solution | PCA sigma1 / 2 / 3 | strike / dip | depth p10 / "
          "med / p90 |", "|---|---|---|---|---|"] + shapes
    od = REPO / B["summary"] / a.run
    od.mkdir(parents=True, exist_ok=True)
    (od / "check_b5_test.md").write_text("\n".join(L) + "\n")
    write_manifest(f"b5_check_test_{a.run}", od, inputs=inputs, outputs=[od / "check_b5_test.md"],
                   params={"cases": a.cases})
    print("\n".join(L))


if __name__ == "__main__":
    main()

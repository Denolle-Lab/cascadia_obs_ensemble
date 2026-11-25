"""D-B078 test sets: the event and pair lists of the clusters on which the B5 code is chosen (tomoDD + D-B081 floor vs
GrowClust3D on the NonLinLoc grids vs NLL-SSST-coherence), for the waveform cross-correlation differential times
(D-B040, built in another session) and for the three relocations.

Sets (catalog v5.1, B5 prep/split of run b5_v5.1_s3, which match production b5_v5.1: same prep, same split rule):
* axial: the 1,200 core events of split chunk c000001_p001 nearest its median epicentre (rule of b5_wtest.make_subset);
* blanco: cluster c000010 (core events);  msh: cluster c000007 (Mount St Helens, onshore, well recorded);
* jdf: cluster c000032 (JdF ridge, most events on the floor: the floor test case).
Pairs: the catalog differential-time pairs among each set's events (prep/pairs.parquet), with their number of common
station-phases. Positions and origin times: catalog v5.1 (B4 run b4_v51_iter1, full precision).

  python b5_cc_sets.py   ->  /wd1/mdenolle_data/tomo_v1/b5/cc_test/{events,pairs}_<set>.csv, README.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import load_params, write_manifest  # noqa: E402

B = load_params()["b5"]
RUN = Path(B["out"]) / "b5_v5.1_s3"
OUT = Path(B["out"]) / "cc_test"
SETS = {"axial": ("c000001_p001", 1200), "blanco": ("c000010", None), "msh": ("c000007", None), "jdf": ("c000032", None)}


def main():
    ev = pd.read_parquet(RUN / "prep" / "events.parquet")
    mem = pd.read_parquet(RUN / "split" / "members.parquet")
    pr = pd.read_parquet(RUN / "prep" / "pairs.parquet")
    OUT.mkdir(parents=True, exist_ok=True)
    L = ["# D-B078 test sets (b5_cc_sets.py)", "", "| set | cluster | events | pairs | lat | lon | depth median (km) |",
         "|---|---|---|---|---|---|---|"]
    counts = {}
    for name, (cid, n) in SETS.items():
        m = mem[(mem.cid == cid) & mem.core]
        e = ev.iloc[np.sort(m.ev.values)]
        if n:
            la0, lo0 = e.lat.median(), e.lon.median()
            d = np.hypot((e.lat - la0) * 111.2, (e.lon - lo0) * 111.2 * np.cos(np.radians(la0)))
            e = e.iloc[np.sort(np.argsort(d.values)[:n])]
        idx = set(e.index)
        p = pr[pr.e1.isin(idx) & pr.e2.isin(idx)].copy()
        p["ID1"], p["ID2"] = ev.ID.values[p.e1], ev.ID.values[p.e2]
        cols = ["ID1", "ID2", "n_dt", "sep_km"]
        e[["ID", "time", "lat", "lon", "depth", "region", "tier"]].to_csv(OUT / f"events_{name}.csv", index=False,
                                                                          float_format="%.8g")
        p[cols].to_csv(OUT / f"pairs_{name}.csv", index=False, float_format="%.6g")
        counts[name] = dict(cluster=cid, events=len(e), pairs=len(p))
        L.append(f"| {name} | {cid} | {len(e):,} | {len(p):,} | {e.lat.median():.3f} | {e.lon.median():.3f} | "
                 f"{e.depth.median():.2f} |")
    L += ["", "Events: ID, v5.1 origin time and hypocentre (km bsl). Pairs: the catalog-dt pairs among them (B5 prep: "
          "10 nearest neighbours within 10 km, >= 8 common station-phases; n_dt = catalog dt rows, sep_km = 3D separation)."]
    (OUT / "README.md").write_text("\n".join(L) + "\n")
    write_manifest("b5_cc_sets", OUT, inputs=[RUN / "prep" / "events.parquet", RUN / "split" / "members.parquet",
                                              RUN / "prep" / "pairs.parquet"],
                   outputs=sorted(OUT.glob("*_*.csv")), params=dict(sets=SETS), counts=counts)
    print("\n".join(L))


if __name__ == "__main__":
    main()

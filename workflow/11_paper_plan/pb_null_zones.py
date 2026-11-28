#!/usr/bin/env python3
"""Where the contrasts of the current axes fall in the null truth's flattened zones (D-B112). The zones were built on
the PB2002 segments (null_zones.npz in the null root's synth/truth/); the contrasts are scored on the current node
sets. Per contrast: the share of its nodes in its own segment's zone and the null-truth contrast (zero only when all
nodes lie in one zone), for the PB2002 node sets (the zones' own segmentation; built once into
<PB_BIG>/pb2002_nodes/ with pb_structure.py and PB_AXES=pb2002) and the current ones. Writes
<PB_OUT>/null_zone_check.csv (columns *_old = PB2002 node sets, *_new = current)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pb_common as P  # noqa: E402
from pb_circularity import contrast  # noqa: E402

NULL = Path(os.environ.get("PB_NULL_ROOT", "/wd1/mdenolle_data/tomo_v1/ensemble_v0.4.2_null")) / "synth" / "truth"


def node_sets_pb2002():
    d = P.BIG / "pb2002_nodes"
    f = d / "contrast_nodes.npz"
    if not f.exists():
        (d / "tables").mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "PB_AXES": "pb2002", "PB_BIG": str(d), "PB_OUT": str(d / "tables")}
        subprocess.run([sys.executable, str(HERE / "pb_structure.py")], env=env, cwd=HERE, check=True,
                       stdout=subprocess.DEVNULL)
    return f


def rows(nodes_file, zmap, zsegs, tr):
    nodes = np.load(nodes_file)
    layer = nodes["layer"]
    n = layer.size
    out = []
    for key in [k[:-2] for k in nodes.files if k.endswith("|1")]:
        seg, var, cn = key.split("|")
        i1, i2 = nodes[key + "|1"], nodes[key + "|2"]
        if len(i1) == 0 or len(i2) == 0:
            continue
        zz = np.r_[zmap[i1], zmap[i2]]
        own = zsegs.index(seg) if seg in zsegs else -9
        d = contrast(tr[var], layer, i1, i2, n)
        out.append(dict(seg=seg, var=var, contrast=cn, own_share=float(np.mean(zz == own)),
                        single_zone=bool(len(np.unique(zz)) == 1 and zz[0] >= 0),
                        D0=float(d[0]) if d is not None else np.nan))
    return pd.DataFrame(out).set_index(["seg", "var", "contrast"])


def main():
    z = np.load(NULL / "null_zones.npz")
    zmap, zsegs = z["zone"].reshape(-1), list(z["segs"])
    t = np.load(NULL / "truth_model.npz")
    tr = {"Vp": np.transpose(t["vp"], (2, 1, 0)).reshape(1, -1).astype(float),
          "Vs": np.transpose(t["vs"], (2, 1, 0)).reshape(1, -1).astype(float)}
    tr["VpVs"] = tr["Vp"] / tr["Vs"]
    old = rows(node_sets_pb2002(), zmap, zsegs, tr)
    new = rows(P.BIG / "contrast_nodes.npz", zmap, zsegs, tr)
    m = old.join(new, lsuffix="_old", rsuffix="_new", how="outer")
    m.to_csv(P.OUT / "null_zone_check.csv")
    P.log(f"median own-zone share PB2002 {m.own_share_old.median():.2f}, {P.AXES} {m.own_share_new.median():.2f}; "
          f"median |D0| {m.D0_old.abs().median():.3f} / {m.D0_new.abs().median():.3f}; single-zone contrasts "
          f"{int(m.single_zone_new.fillna(False).sum())} of {int(m.single_zone_new.notna().sum())}")
    big = m[(m.D0_new - m.D0_old).abs() > 0.05]
    P.log(big.round(3).to_string())


if __name__ == "__main__":
    main()

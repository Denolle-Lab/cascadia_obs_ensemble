"""Model-vs-method test: Ian's v4 1D models on the same NLL grid as the 3D model.

Keeps the 3D model's air / water nodes (same topography, same station placement) and replaces every
solid node with the 1D profile (depth below sea level; rock above sea level takes the top velocity;
linear gradients between nodes as in data/vel_*.csv, Vs = Vp/sqrt(3)). Also writes the list of
stations within R km of a point, for scale/run_reloc_a1_grids.sh (STATIONS=...).

Usage: python a2_model_1d_test.py pnsn_wa [--lat 45.45 --lon -121.75 --r 270]
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
B = Path("/wd1/mdenolle_data/reloc_v5")
BASE = B / "model_vpvs176"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--lat", type=float, default=45.45)
    ap.add_argument("--lon", type=float, default=-121.75)
    ap.add_argument("--r", type=float, default=270.0)
    a = ap.parse_args()
    vel = pd.read_csv(REPO / f"data/vel_{a.model}.csv")
    vel.columns = [c.strip() for c in vel.columns]
    out = B / f"model_ian_{a.model}"
    out.mkdir(exist_ok=True)
    hdr = (BASE / "cascadia3d.P.mod.hdr").read_text()
    nx, ny, nz = map(int, hdr.split()[:3])
    z0, dz = float(hdr.split()[5]), float(hdr.split()[8])
    z = z0 + dz * np.arange(nz)
    zz = np.clip(z, 0, None)
    vp1 = np.interp(zz, vel.depth, vel.vp)        # node values with linear gradients in between
    vs1 = np.interp(zz, vel.depth, vel.vs)
    for ph, v1 in (("P", vp1), ("S", vs1)):
        s = np.fromfile(BASE / f"cascadia3d.{ph}.mod.buf", "<f4").reshape(nx, ny, nz)
        v = dz / s
        air = np.isclose(v, 0.3)
        water = np.isclose(v, 1.5) if ph == "P" else np.isclose(v, 0.4)
        solid = ~(air | water)
        v = np.where(solid, v1[None, None, :], v)
        (dz / v).astype("<f4").tofile(out / f"cascadia3d.{ph}.mod.buf")
        (out / f"cascadia3d.{ph}.mod.hdr").write_text(hdr)
        print(ph, "solid nodes", solid.mean().round(3), "v range", v.min(), v.max())
    st = (BASE / "stations_GTSRCE.in").read_text().splitlines()
    keep = []
    for l in st:
        t = l.split()
        la, lo = float(t[3]), float(t[4])
        if np.hypot((la - a.lat) * 111.2, (lo - a.lon) * 111.2 * np.cos(np.radians(a.lat))) <= a.r:
            keep.append(l)
    (out / "stations_GTSRCE.in").write_text("\n".join(keep) + "\n")
    print(len(keep), "stations within", a.r, "km ->", out)


if __name__ == "__main__":
    main()

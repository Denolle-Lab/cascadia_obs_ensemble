"""Forward-accuracy test: NonLinLoc Grid2Time (Podvin-Lecomte) against analytic travel times (flat models).

Models (velocity depends on depth only):
  grad   v(z) = 5.0 + 0.03 z km/s (P-like); exact t = arccosh(1 + g^2 R^2 / (2 v_s v_r)) / g
  layer  6.0 km/s over 8.0 km/s at 30 km; exact t = min(direct, Moho head wave)
Source (the 'station') at the surface (z = 0) or at 3 km (seafloor-like); receivers ('events') at 2-40 km
depth and 5-200 km offset. Cell sizes 1, 0.5, 0.25 km; cell velocity at the cell centre ('c') or at the
lower-left node as in a1_build_model.py ('n', the production convention).

  python b2_fwd_pltest.py   -> OUT/pltest/summary.csv
"""
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np
import pandas as pd

from b2_fwd_common import OUT, PROJ, grid2time, nll_time

X0, Y0 = 0.0, -500.0          # source position in the LAMBERT frame (km)
WD = OUT / "pltest"


def vel(model):
    if model == "grad":
        return lambda lo, la, x, y, z: 5.0 + 0.03 * np.maximum(z, 0.0)
    return lambda lo, la, x, y, z: np.where(z < 30.0, 6.0, 8.0)


def exact(model, zs, x, zr):
    if model == "grad":
        g, v0 = 0.03, 5.0
        R2 = x ** 2 + (zr - zs) ** 2
        return np.arccosh(1 + g ** 2 * R2 / (2 * (v0 + g * zs) * (v0 + g * zr))) / g
    v1, v2, h = 6.0, 8.0, 30.0
    direct = np.hypot(x, zr - zs) / v1
    ci = np.sqrt(1 - (v1 / v2) ** 2)
    xc = (h - zs) * (v1 / v2) / ci + (h - zr) * (v1 / v2) / ci
    head = np.where(x > xc, x / v2 + ((h - zs) + (h - zr)) * ci / v1, np.inf)
    return np.minimum(direct, head)


def job(a):
    model, zs, dx, conv = a
    lon, lat = PROJ(X0 * 1e3, Y0 * 1e3, inverse=True)
    label = f"{model}_z{zs:g}_d{dx:g}_{conv}"
    root = grid2time(WD / label, label, lat, lon, zs, "P", vel(model), (X0 - 5, X0 + 205, Y0 - 12, Y0 + 12),
                     dx, (-5.0, 45.0), eft=False, centred=(conv == "c"))
    xs = np.r_[5, 10, 15, 20, 25, 35, 50, 75, 100, 125, 150, 175, 200].astype(float)
    zr = np.r_[2, 5, 10, 15, 20, 25, 35, 40].astype(float)
    X, Z = np.meshgrid(xs, zr, indexing="ij")
    X, Z = X.ravel(), Z.ravel()
    t = nll_time(root, X0 + X, np.full_like(X, Y0), Z)
    return pd.DataFrame(dict(model=model, zs=zs, dx=dx, conv=conv, x=X, zr=Z, t_nll=t, t_exact=exact(model, zs, X, Z)))


def main():
    jobs = list(product(["grad", "layer"], [0.0, 3.0], [1.0, 0.5, 0.25], ["c", "n"]))
    with ProcessPoolExecutor(12) as ex:
        d = pd.concat(list(ex.map(job, jobs)))
    d["err"] = d.t_nll - d.t_exact
    d["rel_pct"] = 100 * d.err / d.t_exact
    d.to_csv(WD / "all.csv", index=False)
    d["xbin"] = pd.cut(d.x, [0, 25, 50, 100, 150, 200])
    s = d.pivot_table(index=["model", "zs", "conv", "dx"], columns="xbin", values="err", aggfunc="median",
                      observed=True).round(3)
    s["rel_pct_median"] = d.groupby(["model", "zs", "conv", "dx"]).rel_pct.median().round(3)
    s.to_csv(WD / "summary.csv")
    pd.set_option("display.width", 200)
    print(s.to_string())


if __name__ == "__main__":
    main()

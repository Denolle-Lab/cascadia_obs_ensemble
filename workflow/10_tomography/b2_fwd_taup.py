"""Forward-accuracy test: exact spherical 1D reference (ObsPy TauP) for the laterally homogeneous MOD (oned).

The 1D model is what tomoDD sees for OUT/mods/oned.MOD: velocity linear in depth between the nodes, the
z = 0 value above sea level (cellvel clamps depth at 0), ak135 below 410 km. TauP needs the surface at the
top of the model, so the model starts at 2 km above sea level (depth' = z + 2 km, planet radius 6373 km, sea
level at r = 6371 km as in tomoDD). First arrival of p/P (s/S), source and receiver at their true depths.
Distances: t_taup_oned_sph = great circle on the 6371 km sphere from geographic coordinates (what tomoDD
computes), t_taup_oned_ell = WGS-84 geodesic distance (the true geometry; NLL's LAMBERT grid has it).

  python b2_fwd_taup.py          -> OUT/nll/taup_oned.csv
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from obspy.taup import TauPyModel
from obspy.taup.taup_create import build_taup_model
from pyproj import Geod

from b2_fwd_common import OUT, RE, read_mod

TOP = 2.0
AK = __import__("obspy").__path__[0] + "/taup/data/ak135f_no_mud.nd"
WD = OUT / "taup"


def write_nd():
    m = read_mod(OUT / "mods" / "oned.MOD")
    z, vp, vs = m["z"], m["vp"][5, 5, :], m["vs"][5, 5, :]
    v0p, v0s = np.interp(0.0, z, vp), np.interp(0.0, z, vs)
    rows = [(0.0, v0p, v0s), (TOP, v0p, v0s)]
    rows += [(zz + TOP, a, b) for zz, a, b in zip(z, vp, vs) if 0 < zz <= 400]
    lines = []
    for d, a, b in rows:
        lines.append(f"{d:9.3f} {a:7.4f} {b:7.4f} 3.0000")
        if abs(d - (47 + TOP)) < 1e-6:                  # label only (no discontinuity at the 'moho')
            lines += ["mantle", f"{d:9.3f} {a:7.4f} {b:7.4f} 3.0000"]
    ak = [l for l in open(AK).read().splitlines()]
    seen410 = False
    for l in ak:
        t = l.split()
        if len(t) == 1:
            if t[0] in ("outer-core", "inner-core"):
                lines.append(t[0])
            continue
        d = float(t[0])
        if d < 410:
            continue
        if d == 410 and not seen410:            # first (upper) 410 line skipped, start at the discontinuity
            seen410 = True
            continue
        lines.append(f"{d + TOP:9.3f} {float(t[1]):7.4f} {float(t[2]):7.4f} {float(t[3]):.4f}")
    WD.mkdir(parents=True, exist_ok=True)
    (WD / "oned_top2.nd").write_text("\n".join(lines) + "\n")
    build_taup_model(str(WD / "oned_top2.nd"), output_folder=str(WD))
    return WD / "oned_top2.npz"


_M = {}


def first(args):
    npz, phase, zs, zr, deg = args
    if "m" not in _M:
        _M["m"] = TauPyModel(model=str(npz))
    pl = ["p", "P"] if phase == "P" else ["s", "S"]
    for eps in (0.0, 1e-3, -1e-3, 5e-3):            # TauP occasionally fails at layer-boundary depths
        try:
            arr = _M["m"].get_travel_times(source_depth_in_km=zs + TOP + eps, distance_in_degree=deg,
                                           phase_list=pl, receiver_depth_in_km=zr + TOP)
            return min(a.time for a in arr) if arr else np.nan
        except Exception:
            continue
    return np.nan


def main():
    npz = write_nd()
    p = pd.read_csv(OUT / "subset/picks.csv")
    la1, lo1, la2, lo2 = (np.radians(p[c].values) for c in ("lat", "lon", "sta_lat", "sta_lon"))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    deg_sph = np.degrees(2 * np.arcsin(np.sqrt(a)))
    _, _, dist_m = Geod(ellps="WGS84").inv(p.lon.values, p.lat.values, p.sta_lon.values, p.sta_lat.values)
    deg_ell = dist_m / 1e3 / (RE * np.pi / 180)
    out = p[["event_ID", "code", "phase"]].copy()
    for col, deg in (("t_taup_oned_sph", deg_sph), ("t_taup_oned_ell", deg_ell)):
        jobs = [(npz, ph, zs, -el, dg) for ph, zs, el, dg in
                zip(p.phase, p.depth_bsl_km, p.sta_elev_km, deg)]
        with ProcessPoolExecutor(12) as ex:
            out[col] = list(ex.map(first, jobs, chunksize=200))
        print(col, "NaN", int(out[col].isna().sum()), flush=True)
    out.to_csv(OUT / "nll" / "taup_oned.csv", index=False)


if __name__ == "__main__":
    main()

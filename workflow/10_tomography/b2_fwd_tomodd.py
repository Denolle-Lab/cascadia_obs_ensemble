"""Forward-accuracy test, step 3b: tomoDD-SE forward times (1 iteration, initial residuals) for the subset.

  python b2_fwd_tomodd.py RUN [RUN ...]
RUN = <model>[_ellcoord] with <model> = mod_check01 | mod (inputs/MOD) | any OUT/mods/<model>.MOD; the suffix
_ellcoord rescales event, station and MOD-node coordinates with b2_fwd_mods.ellipsoid_coords (tomoDD's sphere
then has WGS-84 distances). 'proposed' = pointz2 MOD with ellipsoid coordinates (the recommended set-up).
Output OUT/tomodd_runs/<RUN>/pred.csv (column t_tomodd).
"""
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from b2_fwd_common import INPUTS, OUT, read_mod, tomodd_forward, write_mod
from b2_fwd_mods import ellipsoid_coords

CENTER = (41.3, -125.0)
ALIASES = {"mod_check01": "mod", "proposed": "pointz2_ellcoord"}


def run(name):
    spec = ALIASES.get(name, name)
    ell = spec.endswith("_ellcoord")
    model = spec.replace("_ellcoord", "")
    mod = INPUTS / "MOD" if model == "mod" else OUT / "mods" / f"{model}.MOD"
    sub = OUT / "subset"
    p = pd.read_csv(sub / "picks.csv")
    e = pd.read_csv(sub / "events.csv")
    st = pd.read_csv(sub / "stations.csv")
    wd = OUT / "tomodd_runs" / name
    center = CENTER
    if ell:
        wd.mkdir(parents=True, exist_ok=True)
        m = read_mod(mod)
        m["lat"], m["lon"] = ellipsoid_coords(m["lat"], m["lon"])
        mod = wd / "MOD_ellcoord"
        write_mod(mod, m)
        e["lat"], e["lon"] = ellipsoid_coords(e.lat.values, e.lon.values)
        st["sta_lat"], st["sta_lon"] = ellipsoid_coords(st.sta_lat.values, st.sta_lon.values)
    out = tomodd_forward(wd, p, e, st, mod, center=center)
    return name, int(out.t_tomodd.isna().sum())


def main():
    with ProcessPoolExecutor(min(8, len(sys.argv) - 1)) as ex:
        for name, nnan in ex.map(run, sys.argv[1:]):
            print(name, "done; missing", nnan, flush=True)


if __name__ == "__main__":
    main()

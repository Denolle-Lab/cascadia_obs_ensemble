"""MTJ shallow-anomaly diagnosis (notes/mtj_shallow.md): shared readers for tomoDD-SE run outputs.

* read_vel(run): per-iteration Vp, Vs, DWS-P, DWS-S from Output_Files/tomoDD.vel ([nx, ny, nz] incl. boundary
  nodes; DWS = norm_abs = sum over rows of |weighted derivative| as tomoDD computes it before THRES)
* grid(run): geographic node lon/lat/z (Output_Files/model_geo.npz, else back-transform of the MOD nodes)
* read_res(run): tomoDD.res (final residuals per datum), with station names and absolute/differential flag
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import REPO, load_params  # noqa: E402
from b2_fwd_common import read_mod  # noqa: E402

T = load_params()["tomodd"]
BASE = Path(T["runs"]) / "mendocino_gorda_v2"
TESTS = BASE / "mtj_tests"
FIG = REPO / "data/tomo_v1/b2_mtj"
INPUTS = BASE / "inputs"
# MTJ box of the finding (geographic)
MTJ = dict(lat=(40.0, 40.6), lon=(-125.2, -124.0), zmax=5.0)


def run_dir(tag):
    p = Path(tag)
    if p.is_absolute():
        return p
    return TESTS / tag if (TESTS / tag).exists() else BASE / tag


def inverse_coords(lat, lon):
    fw = json.loads((INPUTS / "forward.json").read_text())
    clat, clon = fw["center"]
    return (clat + (np.asarray(lat, float) - clat) / fw["lat_scale_M_over_R"],
            clon + (np.asarray(lon, float) - clon) / fw["lon_scale_N_over_R"])


def grid(run=None):
    m = read_mod((run_dir(run) if run else INPUTS) / "MOD")
    lat = inverse_coords(m["lat"], np.full_like(m["lat"], -125.0))[0]
    lon = inverse_coords(np.full_like(m["lon"], 41.3), m["lon"])[1]
    return dict(lon=lon, lat=lat, z=m["z"], vp0=m["vp"], vs0=m["vs"])


def read_vel(run):
    """Per-iteration blocks of tomoDD.vel -> dict(iter -> dict(vp, vs, dwsp, dwss)), arrays [nx, ny, nz]."""
    run = run_dir(run)
    m = read_mod(run / "MOD")
    nx, ny, nz = len(m["lon"]), len(m["lat"]), len(m["z"])
    lines = (run / "Output_Files" / "tomoDD.vel").read_text(errors="replace").splitlines()
    out = {}
    hdr = re.compile(r"(P-wave velocity|S-wave velocity|DWS for P-wave|DWS for S-wave) at iteration:\s+(\d+)")
    key = {"P-wave velocity": "vp", "S-wave velocity": "vs", "DWS for P-wave": "dwsp", "DWS for S-wave": "dwss"}
    i = 0
    while i < len(lines):
        h = hdr.search(lines[i])
        if not h:
            i += 1
            continue
        k, it = key[h.group(1)], int(h.group(2))
        width = 7 if k in ("vp", "vs") else 10
        rows = lines[i + 1: i + 1 + ny * nz]
        a = np.array([[float(r[j * width:(j + 1) * width]) if "*" not in r[j * width:(j + 1) * width] else np.inf
                       for j in range(nx)] for r in rows])
        out.setdefault(it, {})[k] = a.reshape(nz, ny, nx).transpose(2, 1, 0)
        i += 1 + ny * nz
    return out


def read_res(run, final=True):
    """tomoDD.res (or tomoDD-SE.res.ini) -> DataFrame with sta code, dt, c1, c2, idx, qual, res_ms, wt, offs."""
    run = run_dir(run)
    f = run / "Output_Files" / "tomoDD.res" if final else run / "tomoDD-SE.res.ini"
    df = pd.read_csv(f, sep=r"\s+", skiprows=1, header=None,
                     names=["code", "dt", "c1", "c2", "idx", "qual", "res_ms", "wt", "offs"])
    df["absolute"] = df.c1 == df.c2
    df["phase"] = np.where(df.idx.isin([1, 3]), "P", "S")
    return df


def stations():
    return pd.read_csv(INPUTS / "station_codes.csv")

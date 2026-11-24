"""D-B048 test (tomoDD-SE with frozen velocity vs GrowClust3D): shared paths, frames and data format.

Frames
* B4 frame: NLL LAMBERT x, y (km; lat0 45, lon0 -125.8, parallels 41 / 49, WGS-84) and the Earth-flattened depth
  z' = Re ln(Re / (Re - z)) of the B4 travel-time grids (b4_time_grids.py). GrowClust3D reads those grids, so its
  event depths go in as z' and come out as z' (mapped back with z = Re (1 - exp(-z' / Re)), as b4_catalog.py).
* tomoDD frame: tomoDD-SE's sphere (R = 6371 km) with the D-B028 P-2 ellipsoid rescaling of latitude and longitude
  about the case centre (b2_fwd_mods.ellipsoid_coords); depth is true depth (tomoDD traces rays on the sphere).

Case data (one folder per case, <out>/<case>/data/), written by b5_prep.py (real swarm) or b5_synth.py:
  events.csv   ID, time (start origin time, UTC), lat, lon, depth (km bsl, true), x_km, y_km, zf_km, h_min, h_max,
               sig_z, tier, mag
  picks.csv    event_ID, station, phase, tt (pick - start origin time, s), cls, sigma, w
  stations.csv station, lat, lon, zsrc_km (+ down, as GTSRCE), code (S0000.., tomoDD keeps 7 characters)
  dt.csv       id1, id2, station, phase, tt1, tt2, w   (the one set of differential times fed to both codes)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import REPO, load_params, write_manifest  # noqa: E402,F401
from b2_fwd_common import RE, eft_depth, lambert_km, read_nll_hdr  # noqa: E402,F401

P = load_params()
B5 = P["b5_test"]
OUT = Path(B5["out"])
GRIDS = Path(B5["time_grids"])
PREFIX = B5["grid_prefix"]
ERR_P = {int(k): float(v) for k, v in P["pick_errors"]["P_sigma_s"].items()}
S_OVER_P = float(P["pick_errors"]["S_over_P"])
KM_DEG = 111.19


def unflatten(zf):
    return RE * (1 - np.exp(-np.asarray(zf, float) / RE))


def case_dir(case):
    return OUT / case


def data_dir(case):
    return OUT / case / "data"


def load_case(case):
    d = data_dir(case)
    return (pd.read_csv(d / "events.csv"), pd.read_csv(d / "picks.csv"), pd.read_csv(d / "stations.csv"),
            pd.read_csv(d / "dt.csv"))


def grid_stations():
    """Stations of the B4 grids (GTSRCE label, lat, lon, zsrc km + down)."""
    man = json.loads((GRIDS / "b4_time_grids_time_b4_v0.3.0.manifest.json").read_text())
    src = [i["path"] for i in man["inputs"] if i["path"].endswith("stations_GTSRCE.in")][0]
    rows = [l.split() for l in Path(src).read_text().splitlines() if l.startswith("GTSRCE")]
    return pd.DataFrame({"station": [r[1] for r in rows], "lat": [float(r[3]) for r in rows],
                         "lon": [float(r[4]) for r in rows], "zsrc_km": [float(r[5]) for r in rows]})


def grid_root(phase, station):
    return GRIDS / f"{PREFIX}.{phase}.{station}.time"


def pick_class(prob):
    prob = pd.Series(prob)
    return pd.Series(np.select([prob >= 0.6, prob >= 0.3, prob.isna()], [2, 1, 1], 0), index=prob.index)


def pick_sigma(cls, phase):
    return pd.Series(cls).map(ERR_P).values * np.where(np.asarray(phase) == "S", S_OVER_P, 1.0)


def weight(sigma):
    return (0.05 / np.asarray(sigma, float)) ** 2      # B5 weight (D-B008 quality, no tomography balance weight)


class Grids:
    """Trilinear lookup in the B4 time grids (x, y Lambert km, z' km), cached per station-phase."""

    def __init__(self):
        self.c = {}

    def get(self, phase, station):
        k = (phase, station)
        if k not in self.c:
            root = grid_root(phase, station)
            g = read_nll_hdr(root)
            a = np.memmap(f"{root}.buf", dtype="<f4", mode="r", shape=(g["nx"], g["ny"], g["nz"]))
            self.c[k] = (g, a)
        return self.c[k]

    def inside(self, phase, station, x, y, zf, margin):
        g, _ = self.get(phase, station)
        x1, y1 = g["x0"] + g["dx"] * (g["nx"] - 1), g["y0"] + g["dx"] * (g["ny"] - 1)
        z1 = g["z0"] + g["dx"] * (g["nz"] - 1)
        return ((x >= g["x0"] + margin) & (x <= x1 - margin) & (y >= g["y0"] + margin) & (y <= y1 - margin)
                & (zf >= g["z0"]) & (zf <= z1 - margin))

    def time(self, phase, station, x, y, zf):
        g, a = self.get(phase, station)
        x, y, zf = (np.atleast_1d(np.asarray(v, float)) for v in (x, y, zf))
        fx, fy, fz = (x - g["x0"]) / g["dx"], (y - g["y0"]) / g["dx"], (zf - g["z0"]) / g["dx"]
        ix = np.clip(np.floor(fx).astype(int), 0, g["nx"] - 2)
        iy = np.clip(np.floor(fy).astype(int), 0, g["ny"] - 2)
        iz = np.clip(np.floor(fz).astype(int), 0, g["nz"] - 2)
        wx, wy, wz = fx - ix, fy - iy, fz - iz
        out = np.zeros(len(x))
        for dx_, ax in ((0, 1 - wx), (1, wx)):
            for dy_, ay in ((0, 1 - wy), (1, wy)):
                for dz_, az in ((0, 1 - wz), (1, wz)):
                    out += ax * ay * az * a[ix + dx_, iy + dy_, iz + dz_]
        return out


def diff_times(ev, pk, D):
    """Catalog differential times (as b2_tomodd_inputs.diff_times): each event with up to max_neighbours nearest
    events within max_sep_km (3D, start hypocentres, true depth); pair kept with >= min_common common
    station-phases; weight = min of the two pick weights. Returns a long table id1, id2, station, phase, tt1, tt2, w."""
    from scipy.spatial import cKDTree
    lat0 = np.deg2rad(ev.lat.mean())
    xyz = np.c_[ev.lon.values * KM_DEG * np.cos(lat0), ev.lat.values * KM_DEG, ev.depth.values]
    ids = ev.ID.values
    dd, ii = cKDTree(xyz).query(xyz, k=D["max_neighbours"] + 1, distance_upper_bound=D["max_sep_km"])
    pairs = set()
    for a in range(len(ids)):
        for d_, b in zip(dd[a, 1:], ii[a, 1:]):
            if np.isfinite(d_):
                pairs.add((min(a, b), max(a, b)))
    key = {k: dict(zip(zip(v.station, v.phase), zip(v.tt, v.w))) for k, v in pk.groupby("event_ID")}
    rows = []
    n_cand = len(pairs)
    for a, b in sorted(pairs):
        A, B_ = key.get(ids[a], {}), key.get(ids[b], {})
        common = [k for k in A if k in B_]
        if len(common) < D["min_common"]:
            continue
        rows += [(ids[a], ids[b], k[0], k[1], A[k][0], B_[k][0], min(A[k][1], B_[k][1])) for k in common]
    dt = pd.DataFrame(rows, columns=["id1", "id2", "station", "phase", "tt1", "tt2", "w"])
    return dt, n_cand


def write_case(case, ev, pk, st, dt, extra=None, inputs=(), params=None):
    d = data_dir(case)
    d.mkdir(parents=True, exist_ok=True)
    ev.to_csv(d / "events.csv", index=False, float_format="%.6f")
    pk.to_csv(d / "picks.csv", index=False, float_format="%.5f")
    st.to_csv(d / "stations.csv", index=False, float_format="%.6f")
    dt.to_csv(d / "dt.csv", index=False, float_format="%.5f")
    pairs = dt[["id1", "id2"]].drop_duplicates()
    linked = pd.unique(pairs.values.ravel())
    summ = dict(case=case, events=len(ev), picks=len(pk), P=int((pk.phase == "P").sum()),
                S=int((pk.phase == "S").sum()), stations=len(st), dt_pairs=len(pairs), dt_data=len(dt),
                events_linked=int(len(linked)), dt_P=int((dt.phase == "P").sum()), dt_S=int((dt.phase == "S").sum()),
                **(extra or {}))
    (d / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    write_manifest(f"b5_data_{case}", d, inputs=inputs, params=params,
                   outputs=[d / f for f in ("events.csv", "picks.csv", "stations.csv", "dt.csv", "summary.json")],
                   counts=summ)
    return summ


def pca(xyz):
    """Principal axes of a point cloud (n x 3, east, north, down km). Returns sigmas (descending), the axes
    (columns), strike (deg, of the plane normal to the smallest axis) and dip (deg)."""
    X = np.asarray(xyz, float)
    X = X - X.mean(axis=0)
    w, v = np.linalg.eigh(np.cov(X.T))
    o = np.argsort(w)[::-1]
    w, v = w[o], v[:, o]
    n = v[:, 2] if v[2, 2] <= 0 else -v[:, 2]          # normal pointing up (down component <= 0)
    dip = np.degrees(np.arccos(abs(n[2])))
    strike = (np.degrees(np.arctan2(n[0], n[1])) + 90.0) % 180.0   # dip direction - 90, folded to 0-180
    return np.sqrt(np.maximum(w, 0)), v, strike, dip


def frame_centre(lat, lon, clear_km=0.05):
    """tomoDD frame centre (wlat, wlon): the median epicentre rounded to 0.001 deg plus 0.0005 deg, moved by
    0.003 deg steps until no event lies within clear_km. An event at or within metres of tomoDD's Cartesian origin
    gives NaN ray coordinates and a segfault (f0399: event exactly at the centre; synth_tdd: 3 m)."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    c = [round(float(np.median(lat)), 3) + 0.0005, round(float(np.median(lon)), 3) + 0.0005]
    for _ in range(100):
        d = np.hypot((lat - c[0]) * KM_DEG, (lon - c[1]) * KM_DEG * np.cos(np.radians(c[0])))
        if d.min() >= clear_km:
            break
        c = [c[0] + 0.003, c[1] + 0.003]
    return round(c[0], 4), round(c[1], 4)


def enu(lat, lon, depth, lat0, lon0):
    """Local east, north (km) about (lat0, lon0) and depth (km)."""
    return np.c_[(np.asarray(lon) - lon0) * KM_DEG * np.cos(np.radians(lat0)), (np.asarray(lat) - lat0) * KM_DEG,
                 np.asarray(depth, float)]

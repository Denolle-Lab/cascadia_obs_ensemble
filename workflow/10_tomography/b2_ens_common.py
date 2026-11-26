"""Track B Bayesian ensemble (D-B030, D-B032; notes/bayes_design.md, notes/ensemble_calibration.md): shared helpers for
the ensemble of ONE merged-model configuration (M0 + the merged tiles), params tomodd.ensemble.

A configuration (tomodd.ensemble.configs.<name>) names the M0 run and the tile runs (region = its inputs, tag = the
deterministic run, smoothing, damping, frozen binary class), so the same code runs v0.3.1's configuration or a later
one (v0.4.x on re-timed picks). Environment overrides: B2_ENS_CONFIG (configuration), B2_ENS_ROOT (output root).
Replaces the single-region version of 2026-11-22 (mendocino_gorda_v2, never run at scale).

Layout under ROOT (params tomodd.ensemble.root):
  prior/                    shared prior ingredients (b2_ens_member.py prep): pick/event/station unions, sigma tables,
                            per-region base data, D-B005 bases, Delph component, merge geometry
  synth/truth/              synthetic truth (b2_ens_synth.py); synth/base_<stage>.parquet the synthetic data
  synth/s000.. synth/nf0..  synthetic members and noise-floor clones; real/r000..  real-data members
  <member>/draws.json, draws.npz, <stage>/{inputs,run,DONE}, merged.npz, MERGED
  chain.state, chain.log, RESUME.md, stats/, products/
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import REPO, load_params  # noqa: E402,F401

P = load_params()
T = P["tomodd"]
E = T["ensemble"]
CONFIG = os.environ.get("B2_ENS_CONFIG", E["config"])
CFG = E["configs"][CONFIG]
ROOT = Path(os.environ.get("B2_ENS_ROOT", E["root"]))
PRIOR = ROOT / "prior"
RUNS = Path(T["runs"])
PR = E["prior"]
STAGES = ["m0"] + list(CFG["tiles"])                 # tomoDD stages of one member
PY_DEFAULT = "/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default"
PY_MODELS = "/home/mdenolle/wt_tomo_res/.pixi/envs/models"
ITEMS = {"component": 1, "rf_p": 2, "rf_s": 3, "static": 4, "fwd": 5, "lam": 6, "noise": 7, "hypo": 8, "damp": 9,
         "sc": 10, "eps": 11}


def stage_cfg(stage):
    return CFG["m0"] if stage == "m0" else CFG["tiles"][stage]


def det_run(stage):
    c = stage_cfg(stage)
    return RUNS / c["region"] / c["tag"]


def det_inputs(stage):
    return RUNS / stage_cfg(stage)["region"] / "inputs"


def binary(stage, kind="main"):
    """Frozen binary of the stage's size class. The freeze is a parameter: configs.<c>.bin_dir overrides
    tomodd.ensemble.bin_dir (e.g. frozen_2026-11-25, the D-B081 floor build, once approved for the inversions)."""
    cls = stage_cfg(stage)["bin"]
    d = Path(CFG.get("bin_dir", E["bin_dir"]))
    return str(d / (f"tomoDD-SE_{cls}" if kind == "main" else f"tomoDD-SE_syn_{cls}"))


if CFG.get("floor"):          # D-B081 surface.dat per stage: not implemented in the ensemble yet (assessment only)
    raise SystemExit("configs.<c>.floor is not implemented in b2_ens_member.py yet (needs a conservative floor at "
                     ">= 30 s spacing per stage: tomoDD's surface.dat holds at most 1001 x 1001 nodes)")


def member_dir(kind, name):
    return ROOT / kind / name


def rng(seed, item, extra=0):
    """Independent stream per member seed and item, so noise-floor clones share every item except eps."""
    return np.random.default_rng([int(seed), ITEMS[item], int(extra)])


def jdump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


# ---------------------------------------------------------------- random field on a Lambert km lattice
def proj():
    from pyproj import Proj
    return Proj(proj="lcc", lat_0=45.0, lon_0=-125.8, lat_1=41.0, lat_2=49.0, ellps="WGS84")   # = the NLL grid


def rf_lattice():
    """Regular Lambert lattice (km) covering every tile and M0 node box with margin, step rf.dx_km; depth =
    tomodd.node_z. Boundary nodes of tomoDD (8 deg outside) are not sampled: the MOD padding copies the interior."""
    d = float(PR["rf"]["dx_km"])
    x = np.arange(-620.0, 570.0 + d / 2, d)
    y = np.arange(-810.0, 830.0 + d / 2, d)
    return x, y, np.array(T["node_z"], float)


def _axis_factor(coords_km, L):
    dd = coords_km[:, None] - coords_km[None, :]
    C = np.exp(-(dd / L) ** 2)
    w, V = np.linalg.eigh(C)
    return V * np.sqrt(np.clip(w, 0.0, None))


def random_field(g):
    """Zero-mean Gaussian field on rf_lattice() [x, y, z], separable correlation exp(-(d/L)^2) (L lateral 30 km, depth
    8 km), marginal std rf.std, multiplied by a depth taper (1 above taper[0], cosine to 0 at taper[1])."""
    x, y, z = rf_lattice()
    R = PR["rf"]
    Ax, Ay, Az = _axis_factor(x, R["len_lat_km"]), _axis_factor(y, R["len_lat_km"]), _axis_factor(z, R["len_z_km"])
    W = g.standard_normal((len(x), len(y), len(z)))
    f = np.einsum("ia,jb,kc,abc->ijk", Ax, Ay, Az, W, optimize=True)
    t0, t1 = R["taper_km"]
    tap = np.where(z <= t0, 1.0, np.where(z >= t1, 0.0, 0.5 * (1 + np.cos(np.pi * (z - t0) / (t1 - t0)))))
    return (R["std"] * f * tap[None, None, :]).astype("float32")


def sample_field(f, lon, lat, z):
    """Bilinear in Lambert x, y and exact on the depth levels (z must be values of tomodd.node_z) -> same shape as lon."""
    x, y, zl = rf_lattice()
    X, Y = proj()(np.asarray(lon, float), np.asarray(lat, float))
    X, Y = X / 1e3, Y / 1e3
    d = x[1] - x[0]
    fx, fy = (X - x[0]) / d, (Y - y[0]) / d
    i = np.clip(np.floor(fx).astype(int), 0, len(x) - 2)
    j = np.clip(np.floor(fy).astype(int), 0, len(y) - 2)
    wx, wy = np.clip(fx - i, 0, 1), np.clip(fy - j, 0, 1)
    k = np.clip(np.searchsorted(zl, np.asarray(z, float)), 0, len(zl) - 1)
    assert np.allclose(zl[k], z), "sample_field: depth not on tomodd.node_z"
    return ((1 - wx) * (1 - wy) * f[i, j, k] + wx * (1 - wy) * f[i + 1, j, k] + (1 - wx) * wy * f[i, j + 1, k]
            + wx * wy * f[i + 1, j + 1, k])


# ---------------------------------------------------------------- tomoDD files
def read_dws(vel_path, shape):
    """Last 'DWS for P-wave' and 'DWS for S-wave' blocks of tomoDD.vel -> (dws_p, dws_s) [nx, ny, nz]."""
    nx, ny, nz = shape
    lines = Path(vel_path).read_text().splitlines()
    out = {}
    for ph in ("P", "S"):
        idx = [i for i, l in enumerate(lines) if l.strip().startswith(f"DWS for {ph}-wave")]
        if not idx:
            out[ph] = None
            continue
        vals = []
        for l in lines[idx[-1] + 1: idx[-1] + 1 + ny * nz]:
            toks = re.findall(r"\*+|[-\d.]+(?:[eE][-+]?\d+)?", l)
            vals.append([np.inf if t.startswith("*") else float(t) for t in toks])
        out[ph] = np.array(vals, float).reshape(nz, ny, nx).transpose(2, 1, 0)
    return out["P"], out["S"]


def read_res_abs(path):
    r = pd.read_csv(path, sep=r"\s+")
    r = r[r.C1 == r.C2].copy()
    r["phase"] = np.where(r.IDX == 3, "P", "S")
    return r.rename(columns={"STA": "code", "C1": "event_ID"}).assign(res=lambda d: d.RES / 1000.0)[
        ["event_ID", "code", "phase", "res"]]


def read_reloc(path):
    r = pd.read_csv(path, sep=r"\s+", header=None)
    return r.rename(columns={0: "ID", 1: "lat", 2: "lon", 3: "depth"})[["ID", "lat", "lon", "depth"]].set_index("ID")


def parse_absolute(path):
    rows, ev = [], None
    for line in open(path):
        if line.startswith("#"):
            ev = int(line.split()[1])
            continue
        c, t, w, ph = line.split()
        rows.append((ev, c, ph, w, float(t)))
    return pd.DataFrame(rows, columns=["event_ID", "code", "phase", "w", "t"])


def parse_dt(path):
    rows, pair = [], None
    for line in open(path):
        if line.startswith("#"):
            pair = tuple(int(v) for v in line.split()[1:3])
            continue
        c, t1, t2, w, ph = line.split()
        rows.append((pair[0], pair[1], c, ph, w, float(t1), float(t2)))
    return pd.DataFrame(rows, columns=["ev1", "ev2", "code", "phase", "w", "t1", "t2"])


def write_absolute(path, d, col):
    """d in file order (event_ID, code, phase, w text, time column); 6 decimals so 1e-6 s reaches tomoDD."""
    out, last = [], None
    ev, code, ph, w, t = d.event_ID.values, d.code.values, d.phase.values, d.w.values, d[col].values
    for i in range(len(d)):
        if ev[i] != last:
            out.append(f"#  {ev[i]}")
            last = ev[i]
        out.append(f"{code[i]:<7s} {t[i]:11.6f} {w[i]:>6s} {ph[i]}")
    Path(path).write_text("\n".join(out) + "\n")


def write_dt(path, d, c1, c2):
    out, last = [], None
    e1, e2, code, ph, w, t1, t2 = (d[k].values for k in ("ev1", "ev2", "code", "phase", "w", c1, c2))
    for i in range(len(d)):
        if (e1[i], e2[i]) != last:
            out.append(f"#  {e1[i]}  {e2[i]}")
            last = (e1[i], e2[i])
        out.append(f"{code[i]:<7s} {t1[i]:11.6f} {t2[i]:11.6f} {w[i]:>6s} {ph[i]}")
    Path(path).write_text("\n".join(out) + "\n")


def write_event_dat(path, det_event_dat, hyp):
    """det event.dat lines with lat, lon, depth replaced by hyp (indexed by ID; tomoDD coordinates)."""
    out = []
    for l in Path(det_event_dat).read_text().splitlines():
        t = l.split()
        r = hyp.loc[int(t[9])]
        out.append(f"{t[0]}  {t[1]} {r.lat:10.6f} {r.lon:11.6f} {r.depth:8.3f} {' '.join(t[5:])}")
    Path(path).write_text("\n".join(out) + "\n")

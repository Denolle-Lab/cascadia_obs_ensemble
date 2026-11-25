"""Track B Bayesian ensemble (D-B030, D-B032; notes/bayes_design.md, notes/ensemble_calibration.md): shared helpers for
the ensemble of ONE merged-model configuration (M0 + the merged tiles), params tomodd.ensemble.

A configuration (tomodd.ensemble.configs.<name>) names the M0 run and the tile runs (region = its inputs, tag = the
deterministic run, smoothing, damping, frozen binary class, mask region), so the same code runs v0.3.1's configuration or
a later one (v0.4.1: re-timed picks and the D-B081 floor in every stage). Optional per configuration: root (output
root), bin_dir (frozen binaries), runtime_files, floor (D-B083 surface.dat per member stage, Air_dep
tomodd.floor.air_dep_km), merge_product (tomodd.merge.<key>: grid, taper, floor, reference of the merge),
calibration_root (the root whose passed synthetic calibration gates this configuration's real phase), calibration (the
explicit acceptance of a failed calibration with inflated Vp / Vs spreads, D-B094: calibration_acceptance), dataset (the b1b
dataset every stage must use, checked), prior (overrides of tomodd.ensemble.prior, merged key by key: e.g. the start
model's random-field width rf.std / rf.len_lat_km / rf.len_z_km / rf.long, a pending PI decision).
Environment overrides: B2_ENS_CONFIG (configuration), B2_ENS_ROOT (output root).
Replaces the single-region version of 2026-11-22 (mendocino_gorda_v2, never run at scale).

Layout under ROOT (params tomodd.ensemble.root):
  prior/                    shared prior ingredients (b2_ens_member.py prep): pick/event/station unions, sigma tables,
                            per-region base data, D-B005 bases, Delph component, merge geometry
  synth/truth/              synthetic truth (b2_ens_synth.py); synth/base_<stage>.parquet the synthetic data
  synth/s000.. synth/nf0..  synthetic members and noise-floor clones; real/r000..  real-data members
  <member>/draws.json, draws.npz, <stage>/{inputs,run,DONE}, merged.npz, MERGED
  chain.state, chain.log, RESUME.md, stats/, products/
"""
import copy
import fcntl
import hashlib
import json
import os
import re
import sys
import time
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
ROOT = Path(os.environ.get("B2_ENS_ROOT", CFG.get("root", E["root"])))
PRIOR = ROOT / "prior"
RUNS = Path(T["runs"])


def _merge(a, b):
    """Deep merge of dict b onto a copy of dict a (b's leaves win; a None leaf in b removes the key)."""
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if v is None:
            out.pop(k, None)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


PR = _merge(E["prior"], CFG.get("prior"))            # effective prior of this configuration
STAGES = ["m0"] + list(CFG["tiles"])                 # tomoDD stages of one member
FLOOR = bool(CFG.get("floor"))                       # D-B083: the D-B081 floor in every stage of every member
PY_DEFAULT = "/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default"
PY_MODELS = "/home/mdenolle/wt_tomo_res/.pixi/envs/models"
ITEMS = {"component": 1, "rf_p": 2, "rf_s": 3, "static": 4, "fwd": 5, "lam": 6, "noise": 7, "hypo": 8, "damp": 9,
         "sc": 10, "eps": 11, "rf_p_long": 12, "rf_s_long": 13}


def stage_cfg(stage):
    return CFG["m0"] if stage == "m0" else CFG["tiles"][stage]


def det_run(stage):
    c = stage_cfg(stage)
    return RUNS / c["region"] / c["tag"]


def det_inputs(stage):
    return RUNS / stage_cfg(stage)["region"] / "inputs"


def binary(stage, kind="main"):
    """Frozen binary of the stage's size class. The freeze is a parameter: configs.<c>.bin_dir overrides
    tomodd.ensemble.bin_dir (v0.4.1: frozen_2026-11-25_inv/bin, the D-B081 floor build of P-F1)."""
    cls = stage_cfg(stage)["bin"]
    d = Path(CFG.get("bin_dir", E["bin_dir"]))
    return str(d / (f"tomoDD-SE_{cls}" if kind == "main" else f"tomoDD-SE_syn_{cls}"))


def runtime_files():
    return Path(CFG.get("runtime_files", E["runtime_files"]))


def air_dep_km(summ):
    """Air_dep of a member stage: the deterministic inputs' value, or tomodd.floor.air_dep_km (-1000 km) when the
    configuration has the floor (tomoDD tests Air_dep on the raw update, before the floor, and would then drop the
    whole iteration's update: it must not trigger; D-B081, D-B083)."""
    return float(T["floor"]["air_dep_km"]) if FLOOR else float(summ["air_dep_km"])


def check_config():
    """Configuration consistency: dataset of every stage, floor spacing of every stage when the floor is on."""
    bad = []
    for s in STAGES:
        R = T["regions"][stage_cfg(s)["region"]]
        if CFG.get("dataset") and R.get("b1b") != CFG["dataset"]:
            bad.append(f"{s}: region {stage_cfg(s)['region']} uses b1b {R.get('b1b')}, configuration {CFG['dataset']}")
        if FLOOR and R.get("floor_spacing_s") is None:
            bad.append(f"{s}: floor on but region {stage_cfg(s)['region']} has no floor_spacing_s")
    if bad:
        raise SystemExit("configuration " + CONFIG + ": " + "; ".join(bad))


def product_version():
    """Model version of the configuration's products (product_version before any '+'), e.g. 0.4.1."""
    return str(CFG["product_version"]).split("+")[0]


# ---------------------------------------------------------------- calibration of the real phase (D-B030, D-B094)
INFLATABLE = ("Vp", "Vs")        # D-B094: only the velocity spreads; hypocentres and Vp/Vs are never inflated


def calibration_root():
    """Root whose synthetic calibration gates this configuration's real phase (configs.<c>.calibration_root, else ROOT)."""
    return Path(CFG.get("calibration_root") or ROOT)


def calibration_entry():
    """configs.<c>.calibration (D-B094): the explicit acceptance of a FAILED synthetic calibration with inflated spreads,
    {accepted: <decision>, inflation: {Vp: f, Vs: f}, coverage_md5: <md5 of the accepted synth/coverage.json>}, or {}."""
    return CFG.get("calibration") or {}


def inflation():
    """{quantity: factor} the products apply to the Vp / Vs spreads (D-B094), {} without an accepted calibration."""
    return {k: float(v) for k, v in (calibration_entry().get("inflation") or {}).items()}


def calibration_acceptance(cov, cov_path):
    """Why the accepted-calibration entry does NOT let the real phase start after a failed calibration (None = it does).
    cov: the calibration root's synth/coverage.json. Conditions: a decision id in `accepted`; factors for Vp and/or Vs
    only, each finite and >= 1; every failed verdict item inflated (so a failed hypocentre or any other item still
    stops the chain); coverage_md5 = the md5 of that coverage.json (the acceptance holds for that verdict only)."""
    acc = calibration_entry()
    if not acc:
        return f"no configs.{CONFIG}.calibration entry (an accepted failure needs the PI's decision, e.g. D-B094)"
    if not str(acc.get("accepted") or "").strip():
        return f"configs.{CONFIG}.calibration.accepted (decision id) missing"
    fac = acc.get("inflation") or {}
    if not fac:
        return f"configs.{CONFIG}.calibration.inflation missing"
    bad = [k for k in fac if k not in INFLATABLE]
    if bad:
        return f"inflation of {bad}: only {list(INFLATABLE)} may be inflated (D-B094)"
    badf = [k for k, v in fac.items() if not (isinstance(v, (int, float)) and np.isfinite(v) and v >= 1.0)]
    if badf:
        return f"inflation factors of {badf} not finite numbers >= 1"
    if not cov:
        return f"{cov_path} missing (the calibration has not run)"
    failed = sorted(k for k, v in (cov.get("verdict") or {}).items() if not v)
    if cov.get("result") == "FAIL" and not failed:
        return f"{cov_path}: result FAIL without a failed verdict item"
    not_inflated = [k for k in failed if k not in fac]
    if not_inflated:
        return f"failed items not covered by the inflation: {not_inflated}"
    if acc.get("coverage_md5") != md5(cov_path):
        return (f"coverage_md5 {acc.get('coverage_md5')} != md5 of {cov_path} ({md5(cov_path)}): the acceptance "
                "was for another calibration verdict")
    return None


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


class Lock:
    """Exclusive flock on a lock file holding the owner's pid, host and start time (per member stage, per chain).
    The kernel releases it when the process ends, so a lock left by a killed process is never stale. acquire()
    returns False (and keeps no lock) when another live process holds it."""

    def __init__(self, path):
        self.path, self.fd = Path(path), None

    def acquire(self, what=""):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps(dict(pid=os.getpid(), pgid=os.getpgid(0), host=os.uname().nodename, what=what,
                                     start=time.strftime("%F %T"))).encode())
        self.fd = fd
        return True

    def owner(self):
        try:
            return self.path.read_text()
        except OSError:
            return ""

    def release(self):
        if self.fd is not None:
            os.close(self.fd)                       # closing the descriptor drops the flock
            self.fd = None


def is_locked(path):
    """True if another process holds the flock on path (tested without keeping it)."""
    if not Path(path).exists():
        return False
    lk = Lock(path)
    if lk.acquire("probe"):
        lk.release()
        return False
    return True


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


def random_field(g, spec=None):
    """Zero-mean Gaussian field on rf_lattice() [x, y, z], separable correlation exp(-(d/L)^2) (L lateral
    spec.len_lat_km, depth spec.len_z_km), marginal std spec.std, multiplied by a depth taper (1 above taper[0], cosine
    to 0 at taper[1]; spec.taper_km, default rf.taper_km). spec default: prior rf (D-B030: 3 %, 30 km, 8 km)."""
    x, y, z = rf_lattice()
    R = PR["rf"] if spec is None else spec
    Ax, Ay, Az = _axis_factor(x, R["len_lat_km"]), _axis_factor(y, R["len_lat_km"]), _axis_factor(z, R["len_z_km"])
    W = g.standard_normal((len(x), len(y), len(z)))
    f = np.einsum("ia,jb,kc,abc->ijk", Ax, Ay, Az, W, optimize=True)
    t0, t1 = R.get("taper_km", PR["rf"]["taper_km"])
    tap = np.where(z <= t0, 1.0, np.where(z >= t1, 0.0, 0.5 * (1 + np.cos(np.pi * (z - t0) / (t1 - t0)))))
    return (R["std"] * f * tap[None, None, :]).astype("float32")


def prior_field(seed, phase):
    """The start model's relative perturbation f for phase P or S of one draw (member seed or the truth seed): the
    rf component (stream rf_p / rf_s; unchanged draws whatever the optional component) plus, if prior rf.long is set
    with std > 0, an independent long-wavelength component (stream rf_p_long / rf_s_long; std, len_lat_km, len_z_km,
    optional taper_km). The prior width is a parameter (PI decision pending), not code."""
    k = phase.lower()
    f = random_field(rng(seed, f"rf_{k}"))
    L = PR["rf"].get("long")
    if L and float(L.get("std", 0)) > 0:
        f = (f + random_field(rng(seed, f"rf_{k}_long"), L)).astype("float32")
    return f


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


def floor_log(log_text):
    """tomoDD-SE D-B081 log lines: '>>>Floor: ID depth floor' (an update stopped 1 m under the floor; iteration =
    count of 'Iteration N finished' lines before it + 1), '>>>Floor not met' (error), '>>>Warning: negative depth'
    (air quakes: none expected with Air_dep -1000 km)."""
    it, hits, aq, notmet = 1, [], [], []
    for ln in log_text.splitlines():
        if ln.startswith(">>>Floor: "):
            t = ln.split()
            hits.append((it, int(t[1]), float(t[2]), float(t[3])))
        elif ln.startswith(">>>Floor not met"):
            notmet.append(ln.strip())
        elif ln.startswith(">>>Warning: negative depth"):
            aq.append((it, int(ln.split()[-1])))
        else:
            m = re.match(r"Iteration\s+(\d+) finished", ln)
            if m:
                it = int(m.group(1)) + 1
    return (pd.DataFrame(hits, columns=["iteration", "ID", "depth", "floor"]),
            pd.DataFrame(aq, columns=["iteration", "ID"]), notmet)


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

"""Track B Bayesian ensemble (D-B030, D-B032; notes/bayes_design.md, notes/ensemble_calibration.md): shared helpers.

* paths of an ensemble: <runs>/<region>/ensemble_<name>/ (members m000.., noise-floor clones nf0..), the shared
  prior ingredients <runs>/<region>/ensemble_prior/ (b2_ens_member.py prep)
* smooth Gaussian random field on the (irregular) node grid, separable kernel exp(-(d/L)^2)
* Brocher (2005) eq. 9 Vp(Vs), as workflow/08_relocation/a1_build_model.py (origin/reloc/track-a)
* tomoDD output readers: final DWS from tomoDD.vel, absolute residuals from tomoDD.res
* writers for event.dat / absolute.dat / dt.ct (formats as b2_tomodd_inputs.py, times with 6 decimals so a 1e-6 s
  perturbation reaches tomoDD's single-precision read)
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import REPO, load_params  # noqa: E402

P = load_params()
T = P["tomodd"]
REGION = "mendocino_gorda_v2"
BASE = Path(T["runs"]) / REGION
INPUTS = BASE / "inputs"
BASELINE = BASE / "d1000_s20"                  # deterministic run used for the pick-noise class factors and truth
PRIOR = BASE / "ensemble_prior"
FIGS = REPO / "data/tomo_v1/b2_ensemble"
PY = [str(REPO / ".pixi/envs/default/bin/python")]
DIST_BINS = [0, 25, 50, 100, 150, 200, 1e4]     # forward-error bins (b2_fwd_common.BINS, last bin open)

# ---------------------------------------------------------------- prior constants (D-B030, PI 2026-11-22)
PRIOR_CFG = dict(
    mixture={"v4_176": 0.40, "v4_173": 0.20, "v4_180": 0.20, "delph": 0.20},   # Delph only if coverage is meaningful
    mixture_no_delph={"v4_176": 0.50, "v4_173": 0.25, "v4_180": 0.25},
    vpvs_ref=1.76,
    rf_std=0.03, rf_len_lat_km=30.0, rf_len_z_km=8.0, rf_taper_km=(50.0, 60.0),   # 1 above 50 km, cosine to 0 at 60
    static_add_s=0.05,                         # OBS S: bootstrap std of the near-event median + 0.05 s
    static_sd={"land_P": 0.05, "land_S": 0.08, "obs_P": 0.08, "obs_S_nostatic": 0.08},
    damp_range=(700.0, 1500.0),                # log-uniform
    smooth_range=(15.0, 30.0),                 # uniform, lateral (lon = lat); depth = half
    delph_taper_km=30.0, delph_min_share=0.10,  # Delph used if >= 10% of the baseline DWS lies in its footprint
    h68_factor=1.515,                          # 68% 2-D radius / sigma (chi2_2): sigma_h = h_max / 1.515
    eps_floor_s=1e-6,
)
ITEMS = {"model": 1, "rf_p": 2, "rf_s": 3, "static": 4, "fwd": 5, "lam": 6, "noise": 7, "hypo": 8, "reg": 9, "eps": 10}


def rng(member_seed, item, extra=0):
    """Independent stream per member and item, so clones share every item except the one perturbed."""
    return np.random.default_rng([int(member_seed), ITEMS[item], int(extra)])


def ens_dir(name):
    return BASE / f"ensemble_{name}"


def brocher_vp(vs, vpvs_min=1.76):
    """Brocher (2005) eq. 9, floored at vpvs_min * Vs (a1_build_model.py, VPVS_MIN 1.76)."""
    vp = 0.9409 + 2.0947 * vs - 0.8206 * vs ** 2 + 0.2683 * vs ** 3 - 0.0251 * vs ** 4
    return np.maximum(vp, vpvs_min * vs)


# ---------------------------------------------------------------- random field
def _axis_factor(coords_km, L):
    d = coords_km[:, None] - coords_km[None, :]
    C = np.exp(-(d / L) ** 2)
    w, V = np.linalg.eigh(C)
    return V * np.sqrt(np.clip(w, 0.0, None))


def random_field(g, lon, lat, z, std, Lh, Lz, taper):
    """Zero-mean Gaussian field on the node grid (nx, ny, nz), separable correlation exp(-(d/L)^2),
    unit marginal std times `std`, multiplied by a depth taper (1 above taper[0], cosine to 0 at taper[1])."""
    lat0 = np.radians(np.mean(lat))
    x = (lon - lon[0]) * 111.19 * np.cos(lat0)
    y = (lat - lat[0]) * 111.19
    Ax, Ay, Az = _axis_factor(x, Lh), _axis_factor(y, Lh), _axis_factor(np.asarray(z, float), Lz)
    W = g.standard_normal((len(lon), len(lat), len(z)))
    f = np.einsum("ia,jb,kc,abc->ijk", Ax, Ay, Az, W, optimize=True)
    z = np.asarray(z, float)
    t0, t1 = taper
    tap = np.where(z <= t0, 1.0, np.where(z >= t1, 0.0, 0.5 * (1 + np.cos(np.pi * (z - t0) / (t1 - t0)))))
    return std * f * tap[None, None, :]


def repad(m):
    """Boundary nodes from the interior as b2_tomodd_inputs.write_mod: lateral copy, top copy, bottom Vp kept with
    the Vp/Vs of the deepest interior node."""
    vp, r = m["vp"].copy(), (m["vp"] / m["vs"]).copy()
    for a in (vp, r):
        a[0], a[-1] = a[1], a[-2]
        a[:, 0], a[:, -1] = a[:, 1], a[:, -2]
        a[:, :, 0] = a[:, :, 1]
    r[:, :, -1] = r[:, :, -2]
    return {**m, "vp": vp, "vs": vp / r}


# ---------------------------------------------------------------- tomoDD outputs
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
        blk = lines[idx[-1] + 1: idx[-1] + 1 + ny * nz]
        vals = []
        for l in blk:
            toks = re.findall(r"\*+|[-\d.]+(?:[eE][-+]?\d+)?", l)
            vals.append([np.inf if t.startswith("*") else float(t) for t in toks])
        a = np.array(vals, float)
        out[ph] = a.reshape(nz, ny, nx).transpose(2, 1, 0)
    return out["P"], out["S"]


def read_res_abs(path):
    r = pd.read_csv(path, sep=r"\s+")
    r = r[r.C1 == r.C2].copy()
    r["phase"] = np.where(r.IDX == 3, "P", "S")
    return r.rename(columns={"STA": "code", "C1": "event_ID"}).assign(res=lambda d: d.RES / 1000.0)[
        ["event_ID", "code", "phase", "res", "QUAL"]]


def read_reloc(path):
    r = pd.read_csv(path, sep=r"\s+", header=None)
    return r.rename(columns={0: "ID", 1: "lat", 2: "lon", 3: "depth"})[["ID", "lat", "lon", "depth"]].set_index("ID")


# ---------------------------------------------------------------- writers
def write_event_dat(path, ev, nd=6):
    """ev: ID, origin_time, lat, lon (already in tomoDD coordinates), depth_bsl_km. Format of b2_tomodd_inputs.py."""
    with open(path, "w") as f:
        for r in ev.itertuples():
            t = pd.Timestamp(r.origin_time)
            cs = int(round((t.second + t.microsecond / 1e6) * 100))
            f.write(f"{t.strftime('%Y%m%d')}  {t.hour:02d}{t.minute:02d}{cs:04d} {r.lat:10.{nd}f} {r.lon:11.{nd}f} "
                    f"{r.depth_bsl_km:8.3f} 1.0 0.0 0.0 0.0 {r.ID:10d} 0\n")


def write_absolute(path, pk, col):
    """pk sorted by event; columns event_ID, code, phase, w and the time column."""
    out = []
    ev = pk.event_ID.values
    code, ph, w, t = pk.code.values, pk.phase.values, pk.w.values, pk[col].values
    last = None
    for i in range(len(pk)):
        if ev[i] != last:
            out.append(f"#  {ev[i]}")
            last = ev[i]
        out.append(f"{code[i]:<7s} {t[i]:11.6f} {w[i]:6.3f} {ph[i]}")
    Path(path).write_text("\n".join(out) + "\n")


def read_dt_pairs(path):
    """dt.ct -> DataFrame (pair index, ev1, ev2, code, phase, w as text) in file order."""
    rows = []
    pair = None
    for line in open(path):
        if line.startswith("#"):
            pair = tuple(int(v) for v in line.split()[1:3])
            continue
        sta, _, _, wt, ph = line.split()
        rows.append((pair[0], pair[1], sta, ph, wt))
    return pd.DataFrame(rows, columns=["ev1", "ev2", "code", "phase", "w"])


def write_dt(path, pairs, tmap):
    """pairs from read_dt_pairs; tmap: dict (event_ID, code, phase) -> time."""
    out = []
    last = None
    for e1, e2, sta, ph, wt in pairs.itertuples(index=False):
        if (e1, e2) != last:
            out.append(f"#  {e1}  {e2}")
            last = (e1, e2)
        out.append(f"{sta:<7s} {tmap[(e1, sta, ph)]:11.6f} {tmap[(e2, sta, ph)]:11.6f} {wt} {ph}")
    Path(path).write_text("\n".join(out) + "\n")

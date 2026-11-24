"""Forward-accuracy test (tomoDD-SE vs NonLinLoc), shared helpers. See notes/forward_accuracy.md.

* paths, projection (= NLL LAMBERT grid of the v4-NLL3D model), the Mendocino-Gorda inputs
* tomoDD MOD reader and a re-implementation of tomoDD's model lookup (cellvel/vel3: trilinear in lon, lat,
  depth on the node velocities, Vs = Vp / (Vp/Vs) at nodes, depth clamped to >= 0)
* NLL grid reader and trilinear time lookup (as NLLoc does), Grid2Time driver for velocity functions
* tomoDD 1-iteration forward driver (initial residuals in tomoDD-SE.res.ini -> predicted times)
* bias tables per phase x station type x distance bin
"""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Proj

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import tomodd_bin  # noqa: E402

REGION = Path("/wd1/mdenolle_data/tomo_v1/b2_tomodd/mendocino_gorda")
INPUTS = REGION / "inputs"
CHECK01 = REGION / "check01"
OUT = Path("/wd1/mdenolle_data/tomo_v1/b2_tomodd/fwd_test")
NLL_MODEL = Path("/wd1/mdenolle_data/reloc_v5/model_vpvs176")
NLL_TIME = Path("/wd1/mdenolle_data/reloc_v5/time_vpvs176")
NLL_BIN = Path("/wd1/mdenolle_data/reloc_v5/src/NonLinLoc/src/bin")
TOMODD_BIN = Path("/wd1/mdenolle_data/tomo_v1/bin/tomoDD-SE_O3nofs_e10k")
PROJ = Proj(proj="lcc", lat_0=45.0, lon_0=-125.8, lat_1=41.0, lat_2=49.0, ellps="WGS84")
TRANS = ("TRANSFORM  LAMBERT RefEllipsoid WGS-84  LatOrig 45.000000  LongOrig -125.800000  "
         "FirstStdParal 41.000000  SecondStdParal 49.000000  RotCW 0.000000")
RE = 6371.0                     # tomoDD Earth radius (km)
BINS = [0, 25, 50, 100, 150, 200, 300]


# ------------------------------------------------------------------ tomoDD model
def read_mod(path):
    """simul2000-style MOD -> dict(lon, lat, z, vp[nx,ny,nz], vs[nx,ny,nz])."""
    tok = Path(path).read_text().split()
    nx, ny, nz = int(tok[1]), int(tok[2]), int(tok[3])
    i = 4
    lon = np.array(tok[i:i + nx], float); i += nx
    lat = np.array(tok[i:i + ny], float); i += ny
    z = np.array(tok[i:i + nz], float); i += nz
    n = nx * ny * nz
    vp = np.array(tok[i:i + n], float).reshape(nz, ny, nx).transpose(2, 1, 0); i += n
    r = np.array(tok[i:i + n], float).reshape(nz, ny, nx).transpose(2, 1, 0)
    return dict(lon=lon, lat=lat, z=z, vp=vp, vs=vp / r, bld=float(tok[0]))


def write_mod(path, m, vpvs=None):
    """Write a MOD from dict(lon, lat, z, vp, vs) (all nodes incl. boundary nodes)."""
    r = m["vp"] / m["vs"] if vpvs is None else vpvs
    lines = [f"{m.get('bld', 0.01)} {len(m['lon'])} {len(m['lat'])} {len(m['z'])}",
             " ".join(f"{v:.4f}" for v in m["lon"]), " ".join(f"{v:.4f}" for v in m["lat"]),
             " ".join(f"{v:.3f}" for v in m["z"])]
    for arr, fmt in ((m["vp"], "{:.4f}"), (r, "{:.5f}")):
        for k in range(len(m["z"])):
            for j in range(len(m["lat"])):
                lines.append(" ".join(fmt.format(v) for v in arr[:, j, k]))
    Path(path).write_text("\n".join(lines) + "\n")


def _axis_weights(nodes, q):
    i = np.clip(np.searchsorted(nodes, q, side="right") - 1, 0, len(nodes) - 2)
    w = np.clip((q - nodes[i]) / (nodes[i + 1] - nodes[i]), 0.0, 1.0)
    return i, w


def mod_velocity(m, phase, lon, lat, z, clamp_surface=True):
    """tomoDD lookup: trilinear velocity at (lon, lat, depth km bsl); depth < 0 -> 0 (cellvel)."""
    v = m["vp"] if phase == "P" else m["vs"]
    z = np.maximum(z, 0.0) if clamp_surface else z
    ix, wx = _axis_weights(m["lon"], lon)
    iy, wy = _axis_weights(m["lat"], lat)
    iz, wz = _axis_weights(m["z"], z)
    out = 0.0
    for dx_, ax in ((0, 1 - wx), (1, wx)):
        for dy_, ay in ((0, 1 - wy), (1, wy)):
            for dz_, az in ((0, 1 - wz), (1, wz)):
                out = out + ax * ay * az * v[ix + dx_, iy + dy_, iz + dz_]
    return out


# ------------------------------------------------------------------ NLL grids
def read_nll_hdr(root):
    h = Path(f"{root}.hdr").read_text().splitlines()
    t = h[0].split()
    nx, ny, nz = map(int, t[:3])
    x0, y0, z0, dx, dy, dz = map(float, t[3:9])
    return dict(nx=nx, ny=ny, nz=nz, x0=x0, y0=y0, z0=z0, dx=dx, kind=t[9])


def nll_time(root, x, y, z):
    """Trilinear lookup in an NLL TIME grid (x, y km in the LAMBERT frame, z km down). NaN outside."""
    g = read_nll_hdr(root)
    a = np.memmap(f"{root}.buf", dtype="<f4", mode="r", shape=(g["nx"], g["ny"], g["nz"]))
    fx, fy, fz = (x - g["x0"]) / g["dx"], (y - g["y0"]) / g["dx"], (z - g["z0"]) / g["dx"]
    ok = (fx >= 0) & (fy >= 0) & (fz >= 0) & (fx <= g["nx"] - 1) & (fy <= g["ny"] - 1) & (fz <= g["nz"] - 1)
    ix = np.clip(np.floor(fx).astype(int), 0, g["nx"] - 2)
    iy = np.clip(np.floor(fy).astype(int), 0, g["ny"] - 2)
    iz = np.clip(np.floor(fz).astype(int), 0, g["nz"] - 2)
    wx, wy, wz = fx - ix, fy - iy, fz - iz
    out = np.zeros(len(x))
    for dx_, ax in ((0, 1 - wx), (1, wx)):
        for dy_, ay in ((0, 1 - wy), (1, wy)):
            for dz_, az in ((0, 1 - wz), (1, wz)):
                out += ax * ay * az * a[ix + dx_, iy + dy_, iz + dz_]
    out[~ok] = np.nan
    return out


def lambert_km(lon, lat):
    x, y = PROJ(np.asarray(lon), np.asarray(lat))
    return np.asarray(x) / 1e3, np.asarray(y) / 1e3


def eft_depth(z):
    """Earth-flattening transform of depth (km bsl), Re = 6371 km."""
    return RE * np.log(RE / (RE - np.asarray(z, float)))


def grid2time(wd, label, lat, lon, zsrc, phase, vel_fn, extent, dx, z_range, eft=False, centred=True):
    """Build a SLOW_LEN grid from vel_fn(lon, lat, x_km, y_km, depth_km) and run Grid2Time (Podvin-Lecomte).

    extent: (x0, x1, y0, y1) km in the LAMBERT frame; z_range (z0, z1) km (flat depths if eft).
    centred: velocity of each PL cell taken at the cell centre (the PL code applies the value stored at
    node (i, j, k) to the cell [i, i+1] x [j, j+1] x [k, k+1]); False = value at the node (as a1_build_model.py).
    eft: Earth-flattening transform (depth and velocity) of the model, source depth transformed too.
    Returns the time-grid root path.
    """
    wd = Path(wd)
    wd.mkdir(parents=True, exist_ok=True)
    root = wd / f"fwd.{phase}.{label}.time"
    if Path(f"{root}.hdr").exists():
        return root
    x0, x1, y0, y1 = extent
    z0, z1 = z_range
    xs = x0 + dx * np.arange(int(round((x1 - x0) / dx)) + 1)
    ys = y0 + dx * np.arange(int(round((y1 - y0) / dx)) + 1)
    zs = z0 + dx * np.arange(int(round((z1 - z0) / dx)) + 1)
    off = dx / 2 if centred else 0.0
    zq_flat = zs + off
    zq = RE * (1 - np.exp(-zq_flat / RE)) if eft else zq_flat
    scale = np.exp(zq_flat / RE) if eft else np.ones_like(zq_flat)
    buf = np.empty((len(xs), len(ys), len(zs)), "<f4")
    for i, xv in enumerate(xs):
        X = np.full(len(ys), xv + off)
        lo, la = PROJ(X * 1e3, (ys + off) * 1e3, inverse=True)
        LO = np.repeat(lo[:, None], len(zs), 1)
        LA = np.repeat(la[:, None], len(zs), 1)
        XX = np.repeat(X[:, None], len(zs), 1)
        YY = np.repeat((ys + off)[:, None], len(zs), 1)
        Z = np.repeat(zq[None, :], len(ys), 0)
        v = vel_fn(LO.ravel(), LA.ravel(), XX.ravel(), YY.ravel(), Z.ravel()).reshape(len(ys), len(zs)) * scale[None, :]
        buf[i] = dx / v
    mroot = wd / f"mod_{label}.{phase}.mod"
    buf.tofile(f"{mroot}.buf")
    Path(f"{mroot}.hdr").write_text(f"{len(xs)} {len(ys)} {len(zs)}  {x0:f} {y0:f} {z0:f}  {dx:f} {dx:f} {dx:f} "
                                    f"SLOW_LEN FLOAT\n{TRANS}\n")
    zs_src = float(eft_depth(zsrc)) if eft else zsrc
    ctl = wd / f"g2t_{label}_{phase}.in"
    ctl.write_text(f"CONTROL 1 54321\nTRANS LAMBERT WGS-84 45.0 -125.8 41.0 49.0 0.0\n"
                   f"GTFILES mod_{label} fwd {phase}\nGTMODE GRID3D ANGLES_NO\n"
                   f"GTSRCE {label} LATLON {lat:.6f} {lon:.6f} {zs_src:.4f} 0.0\nGT_PLFD 1.0e-3 0\n")
    with open(wd / f"g2t_{label}_{phase}.log", "w") as f:
        subprocess.run([str(NLL_BIN / "Grid2Time"), ctl.name], cwd=wd, stdout=f, stderr=subprocess.STDOUT,
                       check=True)
    Path(f"{mroot}.buf").unlink()                      # keep only the time grid
    for c in wd.glob(f"fwd.{phase}.mod.*"):            # Grid2Time's copy of the model
        c.unlink()
    return root


# ------------------------------------------------------------------ tomoDD forward
INP_TEMPLATE = (CHECK01 / "tomoDD-SE.inp")


def tomodd_forward(wd, picks, events, stations, mod_path, center=(41.3, -125.0), air_dep=-1.914):
    """1-iteration tomoDD-SE run; predicted time = observed - initial residual (tomoDD-SE.res.ini).

    picks: event_ID, code, phase, tt, w; events: ID, origin_time, lat, lon, depth_bsl_km;
    stations: code, sta_lat, sta_lon, sta_elev_km. Returns picks with column t_tomodd.
    """
    wd = Path(wd)
    if (wd / "pred.csv").exists():
        return pd.read_csv(wd / "pred.csv")
    (wd / "Input_Files").mkdir(parents=True, exist_ok=True)
    (wd / "Output_Files").mkdir(exist_ok=True)
    with open(wd / "Input_Files/station.dat", "w") as f:
        for r in stations.itertuples():
            f.write(f"{r.code} {r.sta_lat:.6f} {r.sta_lon:.6f} {r.sta_elev_km * 1000.0:.1f}\n")
    with open(wd / "Input_Files/event.dat", "w") as f:
        for r in events.itertuples():
            t = pd.Timestamp(r.origin_time)
            cs = int(round((t.second + t.microsecond / 1e6) * 100))
            f.write(f"{t.strftime('%Y%m%d')}  {t.hour:02d}{t.minute:02d}{cs:04d} {r.lat:10.6f} {r.lon:11.6f} "
                    f"{r.depth_bsl_km:8.3f} 1.0 0.0 0.0 0.0 {r.ID:10d} 0\n")
    dt = []
    with open(wd / "Input_Files/absolute.dat", "w") as f:
        for eid, d in picks.groupby("event_ID"):
            f.write(f"#  {eid}\n")
            for r in d.itertuples():
                f.write(f"{r.code:<7s} {r.tt:9.4f} {1.0:6.3f} {r.phase}\n")
    # one dummy differential pair (the run needs a non-empty dt.ct; it does not change the initial residuals)
    ids = picks.groupby("event_ID").size().index[:2]
    a, b = (picks[picks.event_ID == i].set_index(["code", "phase"]).tt for i in ids)
    com = a.index.intersection(b.index)
    dt = [f"#  {ids[0]}  {ids[1]}"] + [f"{k[0]:<7s} {a[k]:9.4f} {b[k]:9.4f} 1.000 {k[1]}" for k in com]
    if len(com) == 0:                                  # fall back: same event twice is not allowed; skip pair
        dt = []
    (wd / "Input_Files/dt.ct").write_text("\n".join(dt) + "\n")
    shutil.copy(mod_path, wd / "MOD")
    for f in ("ak135.15.SKS", "layer-16.dat"):
        shutil.copy(TOMODD_BIN.parent / f, wd / f)
    inp = INP_TEMPLATE.read_text()
    lines = inp.splitlines()
    for i, l in enumerate(lines):
        if l.startswith("* OBSCC"):
            lines[i + 1] = f"    0      0     {air_dep:.3f}"
        if l.startswith("* wlat"):
            lines[i + 1] = f"  {center[0]}  {center[1]}   0"
    (wd / "tomoDD-SE.inp").write_text("\n".join(lines) + "\n")
    with open(wd / "stdout.txt", "w") as so:
        subprocess.run([tomodd_bin("bin", wd), "tomoDD-SE.inp"], cwd=wd, stdout=so, stderr=subprocess.STDOUT)
    res = pd.read_csv(wd / "tomoDD-SE.res.ini", sep=r"\s+")
    res = res[res.C1 == res.C2]
    res = res.rename(columns={"STA": "code", "C1": "event_ID", "DT": "tt_res"})
    res["phase"] = np.where(res.IDX == 3, "P", "S")
    res["t_tomodd"] = res.tt_res - res["RES"] / 1000.0
    out = picks.merge(res[["event_ID", "code", "phase", "t_tomodd"]], on=["event_ID", "code", "phase"], how="left")
    out.to_csv(wd / "pred.csv", index=False)
    for f in ("Output_Files",):
        shutil.rmtree(wd / f, ignore_errors=True)
    return out


# ------------------------------------------------------------------ tables
def bias_table(df, col, by=("phase", "stype")):
    d = df[np.isfinite(df[col])].copy()
    d["dist"] = pd.cut(d.epi_dist_km, BINS)
    g = d.groupby(list(by) + ["dist"], observed=True)[col]
    t = pd.DataFrame({"n": g.size(), "median": g.median(), "p95_abs": g.apply(lambda x: np.percentile(np.abs(x), 95))})
    tot = d.groupby(list(by), observed=True)[col]
    tt = pd.DataFrame({"n": tot.size(), "median": tot.median(),
                       "p95_abs": tot.apply(lambda x: np.percentile(np.abs(x), 95))})
    tt["dist"] = "all"
    tt = tt.set_index("dist", append=True)
    return pd.concat([t, tt]).sort_index()


def summary_row(df, col):
    """Medians per phase/type for short (0-25), mid (50-100) and far (150-200 km) bins + overall."""
    d = df[np.isfinite(df[col])]
    row = {}
    for ph in ("P", "S"):
        for st in ("land", "obs"):
            e = d[(d.phase == ph) & (d.stype == st)]
            for a, b in ((0, 25), (50, 100), (150, 200)):
                x = e[(e.epi_dist_km > a) & (e.epi_dist_km <= b)][col]
                row[f"{ph} {st} {a}-{b}"] = round(float(x.median()), 3) if len(x) else np.nan
        x = d[d.phase == ph][col]
        row[f"{ph} all"] = round(float(x.median()), 3)
        row[f"{ph} p95|d|"] = round(float(np.percentile(np.abs(x), 95)), 3)
    return row


def md_table(df, floatfmt=".3f"):
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(f"{v:{floatfmt}}" if isinstance(v, (float, np.floating)) else str(v)
                                      for v in r) + " |")
    return "\n".join(out)

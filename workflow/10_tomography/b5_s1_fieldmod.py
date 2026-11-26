"""S1 test for B5 on catalog v5.1 (PI go 2026-11-25): tomoDD MOD built from the 1 km NonLinLoc velocity field of v5.1
(work_time_b4_v0.3.1m_sc21/field_1.{P,S}.buf: merged surface, water 1.5 / 0.4 km/s, air 0.3 km/s, D-B073 shallow crust)
on a graded node mesh, against the node MOD of the netCDF model (b5_relocate.write_mod: rock above the seafloor).
One tomoDD iteration per variant gives tomoDD's calculated differential times at the v5.1 positions
(tomoDD-SE.res.ini: calc = DT - RES/1000); they are compared row by row with the NonLinLoc grids of v5.1
(time_b4_v0.3.1m_sc21), which were built from the same 1 km field.

Graded mesh (km about the cluster's event centre, then lon/lat and the D-B028 P-2 rescaling about the frame centre of
the cluster's run): `fine_km` nodes over the event footprint + margin_km, steps growing by `grow` up to max_step_km out
to the stations + pad_km; depth nodes every 0.5 km from above the highest station to 10 km, 1 km to 20 km, then the
model's depths to 80 km; edges as b5_relocate.write_mod, the bottom edge at 150 km (tomoDD's index map: bld 0.001 needs
the depth span / bld below izkms = 200000). Velocities: trilinear in velocity between the field's cell centres, Earth
flattening undone (v = v' exp(-z'/Re)). Node coordinates rounded to bld (RaySPDR2.f: node positions multiples of bld).

  python b5_s1_fieldmod.py CID [CID ...]      -> <run>/diag/s1/<cid>/{nodes,field}/, compare.parquet, summary.json
"""
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from b2_fwd_common import NLL_MODEL, RE, eft_depth, lambert_km, read_nll_hdr  # noqa: E402
from b2_fwd_mods import ellipsoid_coords  # noqa: E402
import b5_relocate as BR  # noqa: E402

RUN = Path("/wd1/mdenolle_data/tomo_v1/b5/b5_v5.1_s1")
WORK = Path("/wd1/mdenolle_data/reloc_v5/work_time_b4_v0.3.1m_sc21")
GRIDS = "/wd1/mdenolle_data/reloc_v5/time_b4_v0.3.1m_sc21"
OUT = RUN / "diag" / "s1"
MESH = dict(fine_km=1.0, margin_km=3.0, grow=1.6, max_step_km=20.0, pad_km=15.0, z_fine_km=0.5, z_fine_to=10.0,
            z_mid_to=20.0, z_deep=[22, 25, 29, 34, 40, 47, 55, 65, 80], z_bottom=150.0, bld=0.001)
KM = 111.19
LAT = read_nll_hdr(NLL_MODEL / "cascadia3d.P.mod")
SH = (LAT["nx"], LAT["ny"], LAT["nz"])
FIELD = {ph: np.memmap(WORK / f"field_1.{ph}.buf", dtype="<f4", mode="r", shape=SH) for ph in "PS"}
ROCK = {ph: np.memmap(WORK / f"field_1.{ph}.rock.buf", dtype="<f4", mode="r", shape=SH) for ph in "PS"}   # fluids -> rock below
MX = BR.B["tomodd"]["mod_max"]
TAGS = ("nodes", "field", "hybrid", "rock")


def field_v(ph, lon, lat, z, rock=False):
    """True velocity (km/s) of the 1 km field (rock=True: the rock-everywhere field) at (lon, lat, true depth z):
    trilinear in velocity on cell centres."""
    x, y = lambert_km(np.ravel(lon), np.ravel(lat))
    zf = eft_depth(np.ravel(z))
    d = LAT["dx"]
    f = [(x - LAT["x0"]) / d - 0.5, (y - LAT["y0"]) / d - 0.5, (zf - LAT["z0"]) / d - 0.5]
    i = [np.clip(np.floor(g).astype(int), 0, n - 2) for g, n in zip(f, SH)]
    a = [np.clip(g - k, 0.0, 1.0) for g, k in zip(f, i)]
    A = (ROCK if rock else FIELD)[ph]
    v = np.zeros(len(x))
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                w = ((a[0] if di else 1 - a[0]) * (a[1] if dj else 1 - a[1]) * (a[2] if dk else 1 - a[2]))
                v += w * d / A[i[0] + di, i[1] + dj, i[2] + dk]
    return (v * np.exp(-zf / RE)).reshape(np.shape(lon))


def axis(lo_c, hi_c, lo_s, hi_s, step):
    """Graded node axis (km): step over the events + margin, growing steps out to the stations + pad."""
    m = MESH
    a = np.arange(np.floor((lo_c - m["margin_km"]) / step) * step, hi_c + m["margin_km"] + step, step)
    left, right, s = [], [], step
    x = a[0]
    while x > lo_s - m["pad_km"]:
        s = min(s * m["grow"], m["max_step_km"])
        x -= s
        left.append(x)
    s, x = step, a[-1]
    while x < hi_s + m["pad_km"]:
        s = min(s * m["grow"], m["max_step_km"])
        x += s
        right.append(x)
    return np.r_[left[::-1], a, right]


def depth_nodes(z_top):
    m = MESH
    for zstep, zto in ((m["z_fine_km"], m["z_fine_to"]), (m["z_fine_km"], 8.0), (1.0, m["z_fine_to"])):
        z = np.r_[np.arange(z_top, zto + 1e-6, zstep), np.arange(np.floor(zto) + 1, m["z_mid_to"] + 1e-6, 1.0),
                  m["z_deep"]]
        z = np.unique(np.round(z, 3))
        if len(z) + 2 <= MX["nz"]:
            return z
    raise ValueError("depth nodes do not fit maxnz")


def write_field_mod(path, ev_lat, ev_lon, st_lat, st_lon, st_z, center, hybrid=False, rock=False):
    """hybrid: the full field (water, air) only at the fine nodes over the events; the rock-everywhere field elsewhere,
    so that rays reach distant stations through rock, as NonLinLoc's station rock patch makes them."""
    lat_c, lon_c = float(np.mean(ev_lat)), float(np.mean(ev_lon))
    kx = KM * np.cos(np.radians(lat_c))
    ex, ey = (ev_lon - lon_c) * kx, (ev_lat - lat_c) * KM
    sx, sy = (st_lon - lon_c) * kx, (st_lat - lat_c) * KM
    for step in (MESH["fine_km"], 1.5, 2.0, 3.0, 4.0):
        xi = axis(ex.min(), ex.max(), min(sx.min(), ex.min()), max(sx.max(), ex.max()), step)
        yi = axis(ey.min(), ey.max(), min(sy.min(), ey.min()), max(sy.max(), ey.max()), step)
        if len(xi) + 2 <= MX["nx"] and len(yi) + 2 <= MX["ny"]:
            break
    else:
        raise ValueError("graded mesh does not fit maxnx/maxny")
    lon = lon_c + xi / kx
    lat = lat_c + yi / KM
    z_top = np.floor((np.min(st_z) - 0.5) / MESH["z_fine_km"]) * MESH["z_fine_km"]
    zi = depth_nodes(min(z_top, -1.0))
    LO, LA, Z = np.meshgrid(lon, lat, zi, indexing="ij")
    vp = field_v("P", LO, LA, Z, rock=rock)
    vs = field_v("S", LO, LA, Z, rock=rock)
    if hybrid:
        fx = (xi >= ex.min() - MESH["margin_km"] - 1e-6) & (xi <= ex.max() + MESH["margin_km"] + 1e-6)
        fy = (yi >= ey.min() - MESH["margin_km"] - 1e-6) & (yi <= ey.max() + MESH["margin_km"] + 1e-6)
        out_ = ~(fx[:, None, None] & fy[None, :, None]) & np.ones(LO.shape, bool)
        vp[out_] = field_v("P", LO[out_], LA[out_], Z[out_], rock=True)
        vs[out_] = field_v("S", LO[out_], LA[out_], Z[out_], rock=True)
    r = vp / vs
    T = BR.T
    E = T["edge_deg"]
    xn, yn = np.r_[lon[0] - E, lon, lon[-1] + E], np.r_[lat[0] - E, lat, lat[-1] + E]
    zt, zb = T["edge_z"][0], MESH["z_bottom"]
    zn = np.r_[zt, zi, zb]
    vp = np.pad(vp, ((1, 1), (1, 1), (0, 0)), mode="edge")
    r = np.pad(r, ((1, 1), (1, 1), (0, 0)), mode="edge")
    dv = T["deep_vp"]
    vbot = float(np.interp(zb, sorted(dv), [dv[k] for k in sorted(dv)]))
    vp = np.concatenate([vp[:, :, :1], vp, np.full(vp.shape[:2] + (1,), vbot)], axis=2)
    r = np.concatenate([r[:, :, :1], r, r[:, :, -1:]], axis=2)
    assert np.isfinite(vp).all() and np.isfinite(r).all()
    yw, xw = ellipsoid_coords(yn, xn, *center)
    b = MESH["bld"]
    xw, yw, zw = (np.round(np.asarray(v) / b) * b for v in (xw, yw, zn))
    assert (np.diff(xw) > 0).all() and (np.diff(yw) > 0).all() and (np.diff(zw) > 0).all()
    span = max(xw[-1] - xw[0], yw[-1] - yw[0], zw[-1] - zw[0]) / b
    assert span < 200000, f"index map {span:.0f} > izkms"
    lines = [f"{b:g} {len(xw)} {len(yw)} {len(zw)}", " ".join(f"{v:.3f}" for v in xw),
             " ".join(f"{v:.3f}" for v in yw), " ".join(f"{v:.3f}" for v in zw)]
    for arr, fmt in ((vp, "{:.4f}"), (r, "{:.5f}")):
        for k in range(len(zw)):
            for j in range(len(yw)):
                lines.append(" ".join(fmt.format(v) for v in arr[:, j, k]))
    Path(path).write_text("\n".join(lines) + "\n")
    fluid = (vp[1:-1, 1:-1, 1:-1] < 1.6)
    return dict(nx=len(xw), ny=len(yw), nz=len(zw), fine_km=step, z_nodes=[float(v) for v in zi],
                fluid_nodes=int(fluid.sum()), air_nodes=int((vp[1:-1, 1:-1, 1:-1] < 0.5).sum()))


def _cp(a, b):
    """copy a to b, overwriting a read-only copy from an earlier run"""
    b = Path(b)
    if b.exists():
        b.chmod(0o644)
    shutil.copyfile(a, b)


def forward_dir(src, d, mod=None):
    """Copy a cluster's tomoDD inputs to d with one iteration of the first set (initial residuals in res.ini)."""
    (d / "Input_Files").mkdir(parents=True, exist_ok=True)
    (d / "Output_Files").mkdir(exist_ok=True)
    for f in ("MOD", "ak135.15.SKS", "layer-16.dat"):
        _cp(src / f, d / f)
    for f in (src / "Input_Files").iterdir():
        _cp(f, d / "Input_Files" / f.name)
    L = (src / "tomoDD-SE.inp").read_text().splitlines()
    i = next(k for k, l in enumerate(L) if l.startswith("*  ISTART"))
    L[i + 1] = re.sub(r"(\d+)\s*$", "1", L[i + 1])                      # NSET 1
    j = next(k for k, l in enumerate(L) if l.startswith("* NITER"))
    L[j + 1] = re.sub(r"^\s*\d+", "  1", L[j + 1])                       # NITER 1, first set only
    del L[j + 2:j + 5]
    if (d / "tomoDD-SE.inp").exists():
        (d / "tomoDD-SE.inp").chmod(0o644)
    (d / "tomoDD-SE.inp").write_text("\n".join(L) + "\n")
    if mod is not None:
        _cp(mod, d / "MOD")


def run_tomodd(d, exe):
    t0 = time.time()
    with open(d / "stdout.txt", "w") as so:
        p = subprocess.run([exe, "tomoDD-SE.inp"], cwd=d, stdout=so, stderr=subprocess.STDOUT, timeout=7200)
    return dict(exit=p.returncode, wall_s=round(time.time() - t0, 1), res_ini=(d / "tomoDD-SE.res.ini").exists())


def read_res(d):
    r = pd.read_csv(d / "tomoDD-SE.res.ini", sep=r"\s+", skiprows=1, header=None,
                    names=["STA", "DT", "C1", "C2", "IDX", "QUAL", "RES", "WT", "OFFS"])
    r = r[(r.C1 != r.C2) & r.IDX.isin([3, 4])].copy()
    r["RES"] = pd.to_numeric(r.RES, errors="coerce")
    r["ph"] = np.where(r.IDX == 3, "P", "S")
    r["calc_t"] = r.DT - r.RES / 1000.0
    return r


def main(cids):
    P = RUN / "prep"
    st = pd.read_csv(P / "stations.csv").set_index("code")
    ev = pd.read_parquet(P / "events.parquet").set_index("ID")
    G = BR.Grids(GRIDS)
    summ = {}
    rms = lambda v: float(np.sqrt(np.nanmean(np.square(v))))
    mad = lambda v: float(np.nanmedian(np.abs(v - np.nanmedian(v))))
    for cid in cids:
        src = RUN / "clusters" / cid
        info = json.loads((src / "run.json").read_text())
        exe, center = info["bin"], info["center"]
        sd = pd.read_csv(src / "Input_Files" / "station.dat", sep=r"\s+", header=None)
        codes = sd[0].values
        ids = pd.read_csv(src / "Input_Files" / "event.dat", sep=r"\s+", header=None)[9].values
        e = ev.loc[ids]
        s = st.loc[codes]
        W = OUT / cid
        out = {"center": center, "n_events": len(e), "n_sta": len(s)}
        W.mkdir(parents=True, exist_ok=True)
        mods = {"nodes": None, "field": W / "MOD.field", "hybrid": W / "MOD.hybrid", "rock": W / "MOD.rock"}
        out["mesh"] = write_field_mod(mods["field"], e.lat.values, e.lon.values, s.lat.values, s.lon.values,
                                      s.zsrc_km.values, center)
        out["mesh_hybrid"] = write_field_mod(mods["hybrid"], e.lat.values, e.lon.values, s.lat.values, s.lon.values,
                                             s.zsrc_km.values, center, hybrid=True)
        out["mesh_rock"] = write_field_mod(mods["rock"], e.lat.values, e.lon.values, s.lat.values, s.lon.values,
                                           s.zsrc_km.values, center, rock=True)
        for tag in TAGS:
            d = W / tag
            forward_dir(src, d, mods[tag])
            out[tag] = run_tomodd(d, exe)
            log = (d / "stdout.txt").read_text(errors="replace")
            out[tag]["warnings"] = sorted({l.strip()[:90] for l in log.splitlines() if re.search(
                r"(?i)warn|error|nan|stop|segment|fault|negative depth", l)})[:20]
            print(cid, tag, out[tag]["exit"], out[tag]["wall_s"], "s", flush=True)
        rows = []
        for tag in TAGS:
            if not out[tag]["res_ini"]:
                continue
            r = read_res(W / tag)
            sta = st.loc[r.STA.values, "station"].values
            t = {}
            for side in ("C1", "C2"):
                tt = np.full(len(r), np.nan)
                for (sn, ph), g in pd.DataFrame({"s": sta, "p": r.ph.values}).groupby(["s", "p"]).groups.items():
                    g = np.asarray(list(g))
                    ii = r[side].values[g]
                    tt[g] = G.time(ph, sn, e.x_km.reindex(ii).values, e.y_km.reindex(ii).values,
                                   e.zf_km.reindex(ii).values)
                t[side] = tt
            r["calc_n"] = t["C1"] - t["C2"]
            r["d"] = r.calc_t - r.calc_n
            for col, calc in (("res_t", "calc_t"), ("res_n", "calc_n")):
                rr = r.DT - r[calc]
                r[col] = rr - rr.groupby([r.C1, r.C2]).transform("median")
            r["tag"] = tag
            rows.append(r)
        R = pd.concat(rows)
        R.to_parquet(W / "compare.parquet")
        stats = {}
        for (tag, ph), g in R.groupby(["tag", "ph"]):
            stats[f"{tag}_{ph}"] = dict(n=len(g), tomodd_nan=int(g.calc_t.isna().sum()),
                                        d_med_ms=round(float(np.nanmedian(g.d)) * 1e3, 1),
                                        d_mad_ms=round(mad(g.d) * 1e3, 1), d_rms_ms=round(rms(g.d) * 1e3, 1),
                                        d_p99_ms=round(float(np.nanpercentile(np.abs(g.d), 99)) * 1e3, 1),
                                        res_tomodd=round(rms(g.res_t), 4), res_grids=round(rms(g.res_n), 4))
        out["stats"] = stats
        summ[cid] = out
        (W / "summary.json").write_text(json.dumps(out, indent=1))
        print(json.dumps({cid: stats}, indent=1), flush=True)
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])

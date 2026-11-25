"""Track B, model v0.3.2: how well do the node values carry the D-B073 rule and the surface, against the exact field
rule of the v5.1 production fields (the v0.3.2 NonLinLoc grids)? (main session request 2026-11-25, items 3 and 5)

Node variants, each turned into a 1 km field by b4_time_grids.build_field with the merged water/air mask and the
field-level rule switched off (TG.SC = None), so the only difference to production is how the nodes carry the rule
or the surface:
  v031      v0.3.1 nodes (no rule; rock values above the surface)
  ruleall   the rule at every offshore node <= 2.5 km below the seafloor, above-seafloor nodes included (rock values
            above the surface; the first amendment's variant)
  v032rock  v0.3.2's rule (rock nodes only), above-surface nodes filled with the v0.3.1 (tomography) values
  v032water v0.3.2 as stored, water nodes Vp 1.5 / Vs 0.4 km/s (D-B005 water), air nodes = tomography values: what a
            tomoDD MOD with water above the seafloor would hold
Production = the v5.1 fields (exact field rule).
Outputs in <WORK>/compare/: profiles.csv (Vs and Vp along depth at named sites: production cell values, field rule at
the point, trilinear node values of each variant), tt.csv (Grid2Time 1 km travel times at events 2-10 km below the
seafloor and 5-50 km from OBS stations, every variant and production), summary.json.

  python b2_v032_compare.py fields | tt | summary        (models env)
"""
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b2_model_v032 as M  # noqa: E402
import b4_time_grids as TG  # noqa: E402
import velocity_model_tools as RV  # noqa: E402
from b2_fwd_common import eft_depth, nll_time  # noqa: E402

OUT = M.WORK / "compare"
VARIANTS = ["v031", "ruleall", "v032rock", "v032water"]
SITES = {"Axial caldera": (45.95, -130.01), "Endeavour": (47.95, -129.10), "Juan de Fuca ridge flank": (46.5, -128.5),
         "Blanco transform": (44.0, -129.3), "Gorda plate": (41.3, -126.6), "Cascadia slope off Oregon": (44.6, -125.0),
         "Washington shelf": (47.4, -124.6), "Mendocino slope": (40.35, -124.80)}
STATIONS = ["AXCC1.OO", "AXAS1.OO", "G03B.7D", "G19B.7D", "FS06B.7D", "FS08B.7D", "J25C.7D", "J49C.7D", "M08A.7D",
            "J73A.7D", "NC89.NV", "BB350.X9", "BB200.X9"]


def variant_nodes(name):
    nc = xr.open_dataset(M.STORE / f"v{M.VERSION}.nc")
    vp_t, vs_t = nc.Vp_tomography.values.astype("f8"), nc.Vs_tomography.values.astype("f8")
    med = nc.medium.values
    surf = nc.surface_elevation.values.astype("f8")[None]
    off = nc.offshore.values.astype(bool)[None]
    Z = nc.depth.values[:, None, None]
    SC = M.SC
    if name == "v031":
        vp, vs = vp_t, vs_t
    elif name == "ruleall":
        r = off & (Z > 0) & (Z + surf <= SC["thickness_km"])
        vp, vs = vp_t, np.where(r, np.minimum(vs_t, vp_t / SC["vpvs"]), vs_t)
    elif name == "v032rock":
        vp = vp_t
        vs = np.where(med == 0, nc.Vs.values, vs_t)
    elif name == "v032water":
        vp = np.where(med == 1, 1.5, np.where(med == 0, nc.Vp.values, vp_t))
        vs = np.where(med == 1, 0.4, np.where(med == 0, nc.Vs.values, vs_t))
    else:
        raise ValueError(name)
    ds = nc[["Vp", "Vs"]].copy()
    ds["Vp"] = (nc.Vp.dims, vp.astype("f4"))
    ds["Vs"] = (nc.Vs.dims, vs.astype("f4"))
    return ds


def cmd_fields():
    TG.SC = None                                      # field rule off: the nodes carry the rule (or not)
    for v in VARIANTS:
        w = OUT / f"field_{v}"
        w.mkdir(parents=True, exist_ok=True)
        if v != "v032water":                          # Vp nodes = v0.3.1 at rock-relevant nodes: production P field
            for f in ("field_1.P.buf", "field_1.P.rock.buf"):
                if not (w / f).exists():
                    os.symlink(M.PROD / f, w / f)
        TG.build_field(variant_nodes(v), 1.0, w)
        print("field", v, flush=True)


def fields_of(v, ph):
    if v == "production":
        return M.PROD / f"field_1.{ph}.buf"
    return OUT / f"field_{v}" / f"field_1.{ph}.buf"


def g2t_job(a):
    v, label, lat, lon, z, ph = a
    wd = OUT / "tt" / f"{v}_{label}_{ph}"
    root = wd / f"t_{v}.{ph}.{label}.time"
    if Path(f"{root}.hdr").exists():
        return str(root)
    wd.mkdir(parents=True, exist_ok=True)
    sx, sy = (c / 1e3 for c in TG.proj()(lon, lat))
    H = 70
    nx, ny, nz = TG.field_shape(1.0)
    i0 = max(0, int(np.floor(sx - H - TG.G["x0"]))); i1 = min(nx - 1, int(np.ceil(sx + H - TG.G["x0"])))
    j0 = max(0, int(np.floor(sy - H - TG.G["y0"]))); j1 = min(ny - 1, int(np.ceil(sy + H - TG.G["y0"])))
    r = TG.g2t(wd, v, label, lat, lon, z, ph, fields_of(v, ph), (nx, ny, nz), 1.0, i0, i1, j0, j1)
    return str(r)


def cmd_tt():
    st = {s[0]: s for s in TG.gtsrce()}
    use = [s for s in STATIONS if s in st]
    print("stations", use, "missing", [s for s in STATIONS if s not in st], flush=True)
    jobs = [(v, lab, st[lab][1], st[lab][2], st[lab][3], ph) for v in ["production"] + VARIANTS for lab in use
            for ph in ("P", "S")]
    with ProcessPoolExecutor(16) as ex:
        roots = list(ex.map(g2t_job, jobs))
    from b4_surface import surface_m
    rows = []
    g = np.random.default_rng(3)
    for lab in use:
        _, la, lo, zs = st[lab]
        sx, sy = (c / 1e3 for c in TG.proj()(lo, la))
        for dist in (5, 10, 20, 30, 50):
            for az in np.arange(0, 360, 30) + g.uniform(0, 30):
                ex_, ey_ = sx + dist * np.sin(np.radians(az)), sy + dist * np.cos(np.radians(az))
                elon, elat = TG.proj()(ex_ * 1e3, ey_ * 1e3, inverse=True)
                sf = -float(surface_m(np.array([elon]), np.array([elat]))[0]) / 1e3
                if sf <= 0:                            # epicentre on land: not an offshore path
                    continue
                for dz in (2, 4, 6, 10):
                    rows.append(dict(station=lab, sta_depth=zs, dist_km=dist, az=az, lat=elat, lon=elon,
                                     seafloor_km=sf, below_seafloor_km=dz, depth=sf + dz, x=ex_, y=ey_))
    ev = pd.DataFrame(rows)
    for v in ["production"] + VARIANTS:
        for ph in ("P", "S"):
            t = np.full(len(ev), np.nan)
            for lab in use:
                m = (ev.station == lab).values
                root = OUT / "tt" / f"{v}_{lab}_{ph}" / f"t_{v}.{ph}.{lab}.time"
                t[m] = nll_time(root, ev.x.values[m], ev.y.values[m], eft_depth(ev.depth.values[m]))
            ev[f"t_{v}_{ph}"] = t
    ev.to_csv(OUT / "tt.csv", index=False)
    print(len(ev), "event-station paths", flush=True)


def cmd_profiles():
    rows = []
    z = np.arange(-0.5, 12.01, 0.25)
    for name, (la, lo) in SITES.items():
        LA, LO = np.full(len(z), la), np.full(len(z), lo)
        f = RV.velocity_at(M.STORE, M.VERSION, LA, LO, z, mode="field")
        n = RV.velocity_at(M.STORE, M.VERSION, LA, LO, z, mode="nll")
        d = pd.DataFrame(dict(site=name, lat=la, lon=lo, depth=z, seafloor_km=-f["surface_km"], medium=f["medium"],
                              Vs_production=n["Vs"], Vp_production=n["Vp"], Vs_field=f["Vs"], Vp_field=f["Vp"]))
        for v in VARIANTS:
            ds = variant_nodes(v)
            d[f"Vs_nodes_{v}"] = RV.trilinear(ds, "Vs", LA, LO, z)
            d[f"Vp_nodes_{v}"] = RV.trilinear(ds, "Vp", LA, LO, z)
        rows.append(d)
    p = pd.concat(rows)
    p.round(4).to_csv(OUT / "profiles.csv", index=False)
    return p


def cmd_summary():
    p = cmd_profiles()
    ev = pd.read_csv(OUT / "tt.csv")
    S = {"profiles": {}, "tt": {}}
    for name, d in p.groupby("site", sort=False):
        rk = (d.medium == 0) & (d.depth - d.seafloor_km <= 4.0) & (d.depth - d.seafloor_km >= 0)
        S["profiles"][name] = {"seafloor_km": round(float(d.seafloor_km.iloc[0]), 3)}
        for v in VARIANTS:
            e = (d[f"Vs_nodes_{v}"] - d.Vs_field)[rk] / d.Vs_field[rk]
            S["profiles"][name][f"Vs_{v}_minus_field_pct_0to4km_below_seafloor"] = [
                round(100 * float(np.nanmedian(e)), 2), round(100 * float(np.nanmax(np.abs(e))), 2)]
    for v in VARIANTS:
        for ph in ("P", "S"):
            dt = ev[f"t_{v}_{ph}"] - ev[f"t_production_{ph}"]
            S["tt"][f"{v}_{ph}"] = dict(median=round(float(dt.median()), 4), p5=round(float(dt.quantile(0.05)), 4),
                                        p95=round(float(dt.quantile(0.95)), 4), max_abs=round(float(dt.abs().max()), 4),
                                        by_dist={int(k): round(float(x.median()), 4) for k, x in dt.groupby(ev.dist_km)},
                                        by_depth={int(k): round(float(x.median()), 4)
                                                  for k, x in dt.groupby(ev.below_seafloor_km)})
    S["tt"]["n_paths"] = int(len(ev))
    S["tt"]["stations"] = sorted(ev.station.unique())
    (OUT / "summary.json").write_text(json.dumps(S, indent=1))
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    {"fields": cmd_fields, "tt": cmd_tt, "summary": cmd_summary}[sys.argv[1]]()

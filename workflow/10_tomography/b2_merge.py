"""Track B: merge the tiles into one velocity model (notes/merge_design.md M1-M3; D-B044..D-B047).

Deterministic merge of the tiles restarted from M0 (D-B046), each at its chosen run (damping 200, smoothing from
data/tomo_v1/b2_merge/tiles_m0/choice.csv), on one geographic grid (params tomodd.merge.product.grid: the M0 extent,
0.125 x 0.1 deg, the shared depth nodes):

  V(x) = V_ref(x) (1 + sum_k w_k(x) dlnV_k(x) / sum_k w_k(x)),   Vp and Vs separately
  V_ref = D-B005 (nearest column) x (1 + M0's change from its own start), Vp/Vs floor start_vpvs_min: the rule
          that built every tile's start, so where no tile changes anything the merged model is the tiles' start
  dlnV_k = V_k / V0_k - 1, tile k's change from its own start (tomoDD trilinear lookup on its geographic nodes)
  w_k = t_k u_k: t_k = product over the other tiles j of sin^2(pi/2 min(1, d_kj / W_kj)), d_kj the distance to the
        edges of tile k's events box that lie inside tile j's box, W_kj the width of their overlap (0 outside box
        k); u_k = 1 on tile k's resolved nodes (checkerboard mask at its damping), cosine taper over taper_km,
        floor r outside (M3). No inverse-variance factor.
Extra variables: resolution_p, resolution_s (finest resolved checkerboard cell, km, over the covering tiles;
0 = not resolved: interpret only where > 0, D-B037), n_tiles (tile boxes covering the node), dws_p, dws_s
(weighted DWS of the covering tiles; 0 = no ray).

  pixi run -e models python workflow/10_tomography/b2_merge.py VERSION "DESCRIPTION" [--nll] [--status S]
  pixi run -e models python workflow/10_tomography/b2_merge.py --dry-run OUT.nc     # test, no registration
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import REPO, load_params  # noqa: E402
from b2_fwd_common import mod_velocity, read_mod  # noqa: E402
from b2_t6_joint import dist_km, facing_edge_points, inside, overlap_width_km, res_window  # noqa: E402
import b2_model_export as EX  # noqa: E402
import b2_tomodd_inputs as TI  # noqa: E402

T = load_params()["tomodd"]
M = T["merge"]
C = M["product"]
RUNS = Path(T["runs"])
RES = REPO / "data" / "tomo_v1" / "b2_res_tests"


def tile_runs(spec=None):
    """[(tile region, tag, mask region, damp)] from --runs "region:tag:maskregion:damp,...", else the phase-1 choice."""
    if spec:
        return [(r, t, m, float(d)) for r, t, m, d in (x.split(":") for x in spec.split(","))]
    ch = pd.read_csv(REPO / C["choice"])
    return [(r.region, r.tag, r.region, 200) for r in ch.itertuples()]


def box(region):
    r = T["regions"][region]
    return [*r["events_lat"], *r["events_lon"]]


def load_mask(region, damp):
    for f in (RES / region / f"d{damp:g}" / "resolution_mask.npz", RES / region / "resolution_mask.npz"):
        if f.exists():
            return np.load(f), f
    return None, None


def reference(lon, lat, z):
    """V_ref on the grid: D-B005 nearest column x (1 + M0 change), floor on Vp/Vs; deep nodes as the builder."""
    zm = z[z <= 80]
    VP, VS = TI.sample_model(lon, lat, zm, 1)
    bad = np.isnan(VP[:, :, 0])                      # columns outside the D-B005 grid: nearest valid column (as
    if bad.any():                                    # b2_tomodd_inputs.write_mod)
        from scipy import ndimage
        idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
        VP, VS = VP[idx[0], idx[1]], VS[idx[0], idx[1]]
    fp, fs = TI.sample_run(C["reference"], lon, lat, zm)
    sp, ss = TI.sample_run(C["reference"], lon, lat, zm, which="start")
    VP, VS = VP * fp / sp, VS * fs / ss
    fl = float(T.get("start_vpvs_min", 0))
    VS = np.where(VP / VS < fl, VP / fl, VS)
    deep = [d for d in z if d > 80]
    r80 = (VP / VS)[:, :, -1:]
    VP = np.concatenate([VP] + [np.full(VP.shape[:2] + (1,), T["deep_vp"][int(d)]) for d in deep], axis=2)
    VS = np.concatenate([VS] + [np.full(VS.shape[:2] + (1,), T["deep_vp"][int(d)]) / r80 for d in deep], axis=2)
    return VP, VS                                      # [x, y, z]


def run_fields(region, tag):
    run = RUNS / region / tag
    g = np.load(run / "Output_Files" / "model_geo.npz")
    s = read_mod(run / "MOD")
    base = dict(lon=g["lon"], lat=g["lat"], z=g["z"])
    D = EX.dws(run, g["vp"].shape)
    return (run, dict(base, vp=s["vp"], vs=s["vs"]), dict(base, vp=g["vp"], vs=g["vs"]),
            {ph: dict(base, vp=D[ph], vs=D[ph]) for ph in D})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("version", nargs="?")
    ap.add_argument("description", nargs="?")
    ap.add_argument("--nll", action="store_true")
    ap.add_argument("--status", default="candidate")
    ap.add_argument("--dry-run", help="write the merged netCDF here, do not register")
    ap.add_argument("--runs", help='test: "region:tag:maskregion:damp,..." instead of the phase-1 choice')
    a = ap.parse_args()
    G = C["grid"]
    lon = np.round(np.arange(G["lon"][0], G["lon"][1] + G["lon"][2] / 2, G["lon"][2]), 4)
    lat = np.round(np.arange(G["lat"][0], G["lat"][1] + G["lat"][2] / 2, G["lat"][2]), 4)
    z = np.array(T["node_z"], float)
    VP0, VS0 = reference(lon, lat, z)
    LO, LA, Z = np.meshgrid(lon, lat, z, indexing="ij")
    flo, fla, fz = LO.ravel(), LA.ravel(), Z.ravel()
    tiles = tile_runs(a.runs)
    boxes = {t: box(t) for t, *_ in tiles}
    num = {ph: np.zeros(flo.size) for ph in "PS"}
    den = {ph: np.zeros(flo.size) for ph in "PS"}
    dws_acc = {ph: np.zeros(flo.size) for ph in "ps"}
    res = {ph: np.full(flo.size, np.inf) for ph in "PS"}
    ncov = np.zeros(flo.size, int)
    used = []
    for region, tag, mreg, damp in tiles:
        bk = boxes[region]
        ink = inside(bk, fla, flo)
        ncov += ink
        t = np.where(ink, 1.0, 0.0)
        for other, bj in boxes.items():
            if other == region:
                continue
            W = overlap_width_km(bk, bj)
            if W <= 0:
                continue
            edge = facing_edge_points(bk, bj)
            if len(edge):
                t[ink] *= np.sin(0.5 * np.pi * np.minimum(1.0, dist_km(fla[ink], flo[ink], edge) / W)) ** 2
        run, m0, m1, D = run_fields(region, tag)
        mk, mf = load_mask(mreg, damp)
        for ph in "PS":
            sel = ink & (t > 0)
            u, rs = res_window(mk, ph, fla[sel], flo[sel], fz[sel], C["taper_km"], C["floor"])
            v0 = mod_velocity(m0, ph, flo[sel], fla[sel], fz[sel], clamp_surface=False)
            v1 = mod_velocity(m1, ph, flo[sel], fla[sel], fz[sel], clamp_surface=False)
            w = t[sel] * u
            num[ph][sel] += w * (v1 / v0 - 1)
            den[ph][sel] += w
            dws_acc[ph.lower()][sel] += w * mod_velocity(D[ph.lower()], "P", flo[sel], fla[sel], fz[sel],
                                                          clamp_surface=False) if ph.lower() in D else 0.0
            if mk is not None:
                a_ = mk["finest_p" if ph == "P" else "finest_s"]
                ix = np.abs(mk["lon"][None, :] - flo[sel][:, None]).argmin(1)
                iy = np.abs(mk["lat"][None, :] - fla[sel][:, None]).argmin(1)
                iz = np.abs(mk["z"][None, :] - fz[sel][:, None]).argmin(1)
                f = a_[ix, iy, iz].astype(float)
                cur = res[ph][sel]
                res[ph][sel] = np.where(f > 0, np.minimum(cur, f), cur)
        used.append(dict(region=region, tag=tag, run=str(run), mask=str(mf) if mf else None,
                         model_geo_sha256=EX.sha256(run / "Output_Files" / "model_geo.npz"),
                         mask_sha256=EX.sha256(mf) if mf else None))
        print(f"{region} {tag}: {int(ink.sum())} grid points in its box; mask {mf}", flush=True)
    shp = LO.shape
    out = {}
    for ph, V0 in (("P", VP0), ("S", VS0)):
        d = np.where(den[ph] > 0, num[ph] / np.where(den[ph] > 0, den[ph], 1), 0.0).reshape(shp)
        out[ph] = V0 * (1 + d)
    vp, vs = out["P"], out["S"]
    tr = lambda a: np.ascontiguousarray(np.asarray(a).reshape(shp).transpose(2, 1, 0)).astype("float32")
    dsum = {ph: np.where(den[ph.upper()] > 0, dws_acc[ph] / np.where(den[ph.upper()] > 0, den[ph.upper()], 1), 0)
            for ph in "ps"}
    src = "data-derived: merge of tomoDD-SE tile inversions (b2_merge.py)"
    V = {
        "Vp": (tr(vp), dict(long_name="P-wave velocity", display_name="Vp (km/s)", units="km/s", source=src)),
        "Vs": (tr(vs), dict(long_name="S-wave velocity", display_name="Vs (km/s)", units="km/s", source=src)),
        "VpVs": (tr(vp / vs), dict(long_name="Vp/Vs ratio", display_name="Vp/Vs", units="1", source="Vp / Vs")),
        "Vp_start": (tr(VP0), dict(long_name="P-wave velocity of the reference (starting) model",
                                   display_name="Vp start (km/s)", units="km/s",
                                   source=f"D-B005 x (1 + change of {C['reference']}), the tiles' start rule")),
        "Vs_start": (tr(VS0), dict(long_name="S-wave velocity of the reference (starting) model",
                                   display_name="Vs start (km/s)", units="km/s",
                                   source=f"D-B005 x (1 + change of {C['reference']}), the tiles' start rule")),
        "Vp_perturbation": (tr(100 * (vp / VP0 - 1)), dict(long_name="Vp change from the reference model",
                                                           display_name="dVp/Vp (%)", units="percent",
                                                           source="100 (Vp / Vp_start - 1)")),
        "Vs_perturbation": (tr(100 * (vs / VS0 - 1)), dict(long_name="Vs change from the reference model",
                                                           display_name="dVs/Vs (%)", units="percent",
                                                           source="100 (Vs / Vs_start - 1)")),
        "dws_p": (tr(dsum["p"]), dict(long_name="merge-weighted derivative weight sum of P rays",
                                      display_name="DWS P", units="1", source="tiles' tomoDD.vel; 0 = no ray")),
        "dws_s": (tr(dsum["s"]), dict(long_name="merge-weighted derivative weight sum of S rays",
                                      display_name="DWS S", units="1", source="tiles' tomoDD.vel; 0 = no ray")),
        "resolution_p": (tr(np.where(np.isfinite(res["P"]), res["P"], 0)), dict(
            long_name="finest resolved checkerboard cell for Vp over the covering tiles", display_name="Vp res (km)",
            units="km", source="tiles' checkerboard masks (D-B018); 0 = not resolved, do not interpret")),
        "resolution_s": (tr(np.where(np.isfinite(res["S"]), res["S"], 0)), dict(
            long_name="finest resolved checkerboard cell for Vs over the covering tiles", display_name="Vs res (km)",
            units="km", source="tiles' checkerboard masks (D-B018); 0 = not resolved, do not interpret")),
        "n_tiles": (tr(ncov).astype("int16"), dict(long_name="number of tile boxes covering the node",
                                                   display_name="tiles", units="1", source="tile events boxes")),
    }
    ds = xr.Dataset({k: (("depth", "latitude", "longitude"), a, at) for k, (a, at) in V.items()},
                    coords=EX.coords(z, lat, lon))
    prov = dict(run="merge: " + ", ".join(f"{u['region']}/{u['tag']}" for u in used), tiles=json.dumps(used),
                reference=C["reference"], weights=json.dumps({k: C[k] for k in ("taper_km", "floor")}),
                start_model=f"D-B005 x (1 + change of {C['reference']})")
    if a.dry_run:
        EX.add_attrs(ds, "0.0.0-dryrun", "dry run", prov, "b2_merge.py --dry-run")
        ds.to_netcdf(a.dry_run, format="NETCDF4_CLASSIC")
        print(f"dry run written to {a.dry_run}")
        return
    EX.add_attrs(ds, a.version, a.description, prov, "b2_merge.py")
    EX.export(ds, a.version, a.description, a.nll, a.status)


if __name__ == "__main__":
    main()

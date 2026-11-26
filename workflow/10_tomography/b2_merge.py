"""Track B: merge the tiles into one velocity model (notes/merge_design.md M1-M3; D-B044..D-B047, D-B049..D-B051).

Deterministic merge of the tiles restarted from M0 (D-B046), each at its chosen run (damping 200, smoothing from
data/tomo_v1/b2_merge/tiles_m0/choice.csv) except the tiles in tomodd.merge.exclude_tiles (D-B050: ridges_north,
approved for relocation only, D-B038; its box no longer tapers the others), on one geographic grid (params
tomodd.merge.product.grid: the M0 extent, 0.125 x 0.1 deg, the shared depth nodes):

  V(x) = R(x) V_tiles(x) + (1 - R(x)) V_ref(x)   where a tile covers x (sum_k w_k > 0),   Vp and Vs separately
       = V_ref(x)                                elsewhere
  V_tiles = sum_k w_k V_k / sum_k w_k, the tile blend
  R = max_k u'_k (D-B049), per phase (the P mask for Vp, the S mask for Vs): u'_k = tile k's resolution window
      without the floor (1 on its resolved nodes, cosine taper to 0 over taper_km), so a node that no tile resolves
      falls back to V_ref (M0); n.b. the tile weights w_k keep the floor. A tile without a mask would have R = 0
      everywhere, so it stops the merge (--runs tests: a warning and the attribute tiles_without_mask).
      As amended (PI 2026-11-24): only tiles merged at smoothing W1 <= tomodd.merge.gate_max_smooth (100) are
      gated (blanco, s20); tiles at s480 count with u = u' = 1 in their box, because checkerboards at that
      smoothing barely recover while their smooth changes are small (< 1 % rms) and improved Gorda depths
  V_k = tile k's model at x (tomoDD trilinear lookup on its geographic nodes) = V0_k (1 + dlnV_k), V0_k its own
        start and dlnV_k = V_k / V0_k - 1 its change: a resolved node covered by one tile equals that tile exactly
  V_ref = D-B005 (nearest column) x (1 + M0's change from its own start), Vp/Vs floor start_vpvs_min: the rule that
          built every tile's start. It equals V0_k only where tile k's nodes sit on this lattice (mendocino_gorda_c,
          ridges_north); blanco, oregon and washington nodes are offset in longitude, and V_ref (1 + dlnV_k) there
          missed V_k by up to 50 % near the surface (v0.3.0; PR #35 review), so it is not used to carry dlnV_k.
  Vp_start, Vs_start = R sum_k w_k V0_k / sum_k w_k + (1 - R) V_ref (V_ref off the tiles): the perturbations are
        the tiles' changes where R = 1 and 0 where R = 0.
  w_k = t_k u_k: t_k = product over the other tiles j of sin^2(pi/2 min(1, d_kj / W_kj)), d_kj the distance to the
        edges of tile k's events box that lie inside tile j's box, W_kj the width of their overlap (0 outside box
        k); u_k = 1 on tile k's resolved nodes (checkerboard mask at its damping), cosine taper over taper_km,
        floor r outside (M3). No inverse-variance factor.
  Vs check (D-B051; params tomodd.merge.vpvs_min, dws_s_min), after R: at covered nodes with Vp/Vs < vpvs_min or
        dws_s < dws_s_min, Vs = Vs_start = V_ref's Vs; where Vp/Vs is then still < vpvs_min (Vp changed),
        Vs = Vp / vpvs_min.
Extra variables: resolution_p, resolution_s (finest resolved checkerboard cell, km, over the covering tiles;
0 = not resolved: interpret only where > 0, D-B037), n_tiles (tile boxes covering the node), dws_p, dws_s
(weighted DWS of the covering tiles; 0 = no ray), tile_weight_p, tile_weight_s (R; 0 = the reference), vs_reverted
(int8 with CF flag_values: 0 unchanged, 1 Vs reverted to the reference, 2 reverted and capped at Vp / vpvs_min; counts
in the attributes, with the excluded tiles and the share of covered nodes that fell back to the reference per phase).

  pixi run -e models python workflow/10_tomography/b2_merge.py VERSION "DESCRIPTION" [--nll] [--status S]
  pixi run -e models python workflow/10_tomography/b2_merge.py --dry-run OUT.nc     # test, no registration
"""
import argparse
import json
import re
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
EXCL = list(M.get("exclude_tiles") or [])          # D-B050: tiles left out of the merge
GATE = float(M.get("gate_max_smooth", 100))        # D-B049 as amended (PI 2026-11-24): only tiles merged at a smoothing
                                                   # W1 <= GATE are gated by their checkerboard mask; at s480 the
                                                   # checkerboards barely recover (median amp 0.005-0.13) although the
                                                   # tiles' smooth changes (< 1 % rms) are what improved Gorda depths


def smooth_w1(tag):
    m = re.search(r"_s(\d+)", tag)
    return float(m.group(1)) if m else float("inf")


def tile_runs(spec=None):
    """([(tile region, tag, mask region, damp)], [excluded tiles]) from --runs "region:tag:maskregion:damp,..." (used
    as given, nothing excluded), else the phase-1 choice without the tiles in exclude_tiles (D-B050), each of which
    must name a tile of choice.csv (its `tile` column, e.g. ridges_north, not the region ridges_north_m0)."""
    if spec:
        return [(r, t, m, float(d)) for r, t, m, d in (x.split(":") for x in spec.split(","))], []
    ch = pd.read_csv(REPO / C["choice"])
    unknown = sorted(set(EXCL) - set(ch.tile))
    if unknown:
        raise SystemExit(f"tomodd.merge.exclude_tiles {unknown} match no tile of {C['choice']} ({', '.join(ch.tile)})")
    dropped = [t for t in ch.tile if t in EXCL]
    print(f"excluded from the merge (D-B050): {', '.join(dropped) or 'none'}", flush=True)
    # optional column mask_region (v0.4.0: the re-timed tiles use the resolution masks of the v0.3.1 tiles, whose
    # event, station and pick sets they share); default the tile's own region
    mreg = lambda r: r.mask_region if isinstance(getattr(r, "mask_region", None), str) else r.region
    return [(r.region, r.tag, mreg(r), 200) for r in ch.itertuples() if r.tile not in EXCL], dropped


def box(region):
    r = T["regions"][region]
    return [*r["events_lat"], *r["events_lon"]]


def load_mask(region, damp, tag):
    """Checkerboard mask of the mask region at this damping. The top-level file is the mask_default_damp mask, so it
    is used only at that damping (as b2_t6_joint, b2_tile_overlap); a missing mask gives (None, None): floor weight."""
    f = RES / region / f"d{damp:g}" / "resolution_mask.npz"
    if not f.exists() and damp == M["mask_default_damp"]:
        f = RES / region / "resolution_mask.npz"
    if not f.exists():
        return None, None
    mk = np.load(f)
    if "damp" in mk.files and float(mk["damp"]) != damp:      # masks that store their damping and smoothing
        raise SystemExit(f"{f} was built at damping {float(mk['damp']):g}, the run uses {damp:g}")
    s = re.search(r"_s(\d+)", tag)
    if "mask_smooth" in mk.files and s and float(mk["mask_smooth"]) != float(s.group(1)):
        print(f"WARNING: {f} was built at smoothing {float(mk['mask_smooth']):g}, the run {tag} uses {s.group(1)}")
    return mk, f


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


def run_fields(run):
    run = Path(run)
    g = np.load(run / "Output_Files" / "model_geo.npz")
    s = read_mod(run / "MOD")
    base = dict(lon=g["lon"], lat=g["lat"], z=g["z"])
    if (run / "dws.npz").exists():                  # ensemble members keep the last DWS blocks only (b2_ens_member.trim);
        z = np.load(run / "dws.npz")                  # tomoDD's '****' overflow there is inf: a large finite value here
        D = {ph: np.where(np.isfinite(z[ph]), z[ph], 1e9).astype(float) for ph in ("p", "s") if z[ph].ndim == 3}
    else:
        D = EX.dws(run, g["vp"].shape)
    return (run, dict(base, vp=s["vp"], vs=s["vs"]), dict(base, vp=g["vp"], vs=g["vs"]),
            {ph: dict(base, vp=D[ph], vs=D[ph]) for ph in D})


def grid_axes():
    G = C["grid"]
    lon = np.round(np.arange(G["lon"][0], G["lon"][1] + G["lon"][2] / 2, G["lon"][2]), 4)
    lat = np.round(np.arange(G["lat"][0], G["lat"][1] + G["lat"][2] / 2, G["lat"][2]), 4)
    return lon, lat, np.array(T["node_z"], float)


def geometry(tiles, lon, lat, z, allow_nomask=False):
    """Merge weights that depend only on the tile boxes and the resolution masks (the same for every ensemble member):
    per tile the grid points it covers (sel), w_k = t_k u_k and u'_k per phase, the finest resolved cell; plus
    n_tiles. tiles: [dict(region, run, mask_region, mask_damp, tag)]; region gives the events box."""
    LO, LA, Z = np.meshgrid(lon, lat, z, indexing="ij")
    flo, fla, fz = LO.ravel(), LA.ravel(), Z.ravel()
    boxes = {t["region"]: box(t["region"]) for t in tiles}
    ncov = np.zeros(flo.size, int)
    geo, nomask = [], []
    for tl in tiles:
        region, tag = tl["region"], tl["tag"]
        mk, mf = load_mask(tl["mask_region"], tl["mask_damp"], tag)
        gated = smooth_w1(tag) <= GATE                  # D-B049 as amended: gate only lightly smoothed tiles
        if mk is None and gated:  # D-B049: no mask -> u' = 0, R = 0: the tile's velocities would drop out everywhere
            msg = (f"{region} {tag}: no resolution mask for {tl['mask_region']} at damping {tl['mask_damp']:g} under "
                   f"{RES}; with R = 0 its velocities would be dropped everywhere (D-B049)")
            if not allow_nomask:
                raise SystemExit(msg)
            print(f"WARNING: {msg}", flush=True)
            nomask.append(region)
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
        sel = ink & (t > 0)
        g = dict(tile=tl, sel=sel, gated=bool(gated), mask=str(mf) if mf else None, n_box=int(ink.sum()),
                 mask_sha256=EX.sha256(mf) if mf else None)
        for ph in "PS":
            if gated:
                u, _ = res_window(mk, ph, fla[sel], flo[sel], fz[sel], C["taper_km"], C["floor"])
            else:                                       # not gated: full weight in its box (u = u' = 1)
                u = np.ones(int(sel.sum()))
            g[f"w_{ph}"] = t[sel] * u
            # u'_k = the window without the floor: u = r + (1 - r) u' (res_window), 0 where the tile has no mask
            g[f"r_{ph}"] = np.clip((u - C["floor"]) / (1 - C["floor"]), 0, 1)
            g[f"fin_{ph}"] = None
            if mk is not None:
                a_ = mk["finest_p" if ph == "P" else "finest_s"]
                ix = np.abs(mk["lon"][None, :] - flo[sel][:, None]).argmin(1)
                iy = np.abs(mk["lat"][None, :] - fla[sel][:, None]).argmin(1)
                iz = np.abs(mk["z"][None, :] - fz[sel][:, None]).argmin(1)
                g[f"fin_{ph}"] = a_[ix, iy, iz].astype(float)
        geo.append(g)
        print(f"{region} {tag}: {int(ink.sum())} grid points in its box; mask {mf}; "
              f"{'gated by its mask' if gated else 'not gated (smoothing > ' + format(GATE, 'g') + ')'}", flush=True)
    return dict(geo=geo, ncov=ncov, shape=LO.shape, flo=flo, fla=fla, fz=fz, nomask=nomask)


def blend(GEO, VP0, VS0, quiet=False):
    """D-B049 blend of the tiles' runs (each tile's run = tl['run']) on the reference VP0/VS0 [x, y, z], then the
    D-B051 Vs check. Returns out (P, S), start (P, S), Rg, dsum, res, fall, vs_rev and the check counts."""
    flo, fla, fz, shp = GEO["flo"], GEO["fla"], GEO["fz"], GEO["shape"]
    num = {ph: np.zeros(flo.size) for ph in "PS"}      # sum_k w_k V_k
    num0 = {ph: np.zeros(flo.size) for ph in "PS"}     # sum_k w_k V0_k
    den = {ph: np.zeros(flo.size) for ph in "PS"}
    dws_acc = {ph: np.zeros(flo.size) for ph in "ps"}
    res = {ph: np.full(flo.size, np.inf) for ph in "PS"}
    R = {ph: np.zeros(flo.size) for ph in "PS"}        # max_k u'_k (D-B049)
    used = []
    for g in GEO["geo"]:
        tl, sel = g["tile"], g["sel"]
        run, m0, m1, D = run_fields(tl["run"])
        for ph in "PS":
            v0 = mod_velocity(m0, ph, flo[sel], fla[sel], fz[sel], clamp_surface=False)
            v1 = mod_velocity(m1, ph, flo[sel], fla[sel], fz[sel], clamp_surface=False)
            w = g[f"w_{ph}"]
            R[ph][sel] = np.maximum(R[ph][sel], g[f"r_{ph}"])
            num[ph][sel] += w * v1
            num0[ph][sel] += w * v0
            den[ph][sel] += w
            dws_acc[ph.lower()][sel] += w * mod_velocity(D[ph.lower()], "P", flo[sel], fla[sel], fz[sel],
                                                          clamp_surface=False) if ph.lower() in D else 0.0
            if g[f"fin_{ph}"] is not None:
                f = g[f"fin_{ph}"]
                cur = res[ph][sel]
                res[ph][sel] = np.where(f > 0, np.minimum(cur, f), cur)
        used.append(dict(region=tl["region"], tag=tl["tag"], run=str(run), mask=g["mask"], gated=g["gated"],
                         model_geo_sha256=EX.sha256(run / "Output_Files" / "model_geo.npz"),
                         mask_sha256=g["mask_sha256"]))
    out, start, Rg, fall = {}, {}, {}, {}
    for ph, VR in (("P", VP0), ("S", VS0)):
        cov = (den[ph] > 0).reshape(shp)
        dd = np.where(den[ph] > 0, den[ph], 1).reshape(shp)
        Rg[ph] = np.where(cov, R[ph].reshape(shp), 0.0)
        # D-B049: a tile's velocity counts only where the checkerboard resolves it; elsewhere M0's reference
        out[ph] = np.where(cov, Rg[ph] * num[ph].reshape(shp) / dd + (1 - Rg[ph]) * VR, VR)
        start[ph] = np.where(cov, Rg[ph] * num0[ph].reshape(shp) / dd + (1 - Rg[ph]) * VR, VR)
        fall[ph] = (float((Rg[ph][cov] == 0).mean()), float(((Rg[ph][cov] > 0) & (Rg[ph][cov] < 1)).mean()))
        if not quiet:
            print(f"{ph}: {int(cov.sum())} covered nodes, {100 * fall[ph][0]:.1f} % on the reference (R = 0), "
                  f"{100 * fall[ph][1]:.1f} % partly (0 < R < 1)", flush=True)
    dsum = {ph: np.where(den[ph.upper()] > 0, dws_acc[ph] / np.where(den[ph.upper()] > 0, den[ph.upper()], 1), 0)
            for ph in "ps"}
    # D-B051 (replaces the Vp/Vs floor of 1.5): the tiles pass on Vp/Vs down to 1.22 at unresolved nodes (v0.3.0),
    # and B4 relocates in this field. At covered nodes with Vp/Vs < vpvs_min or S-ray coverage dws_s < dws_s_min,
    # Vs (and Vs_start) = the reference Vs, which keeps the P field the tiles fit; where Vp then still gives
    # Vp/Vs < vpvs_min, Vs = Vp / vpvs_min. Nodes no tile covers are the reference already and keep 0; so do nodes
    # where R_S = 0 (Vs is the reference by D-B049), unless capped: vs_reverted marks the nodes this step changed.
    vmin, dmin = float(M["vpvs_min"]), float(M["dws_s_min"])
    cov_s = (den["S"] > 0).reshape(shp)
    low_r, low_d = cov_s & (out["P"] / out["S"] < vmin), cov_s & (dsum["s"].reshape(shp) < dmin)
    rev = low_r | low_d
    out["S"], start["S"] = np.where(rev, VS0, out["S"]), np.where(rev, VS0, start["S"])
    cap = rev & (out["P"] / out["S"] < vmin)
    out["S"] = np.where(cap, out["P"] / vmin, out["S"])
    chg = rev & ((Rg["S"] > 0) | cap)
    # D-B051 asks for uint8; int8 ("byte") here, so the Zarr store and the netCDF-4 Classic file (no unsigned types,
    # b2_model_export.write) hold the same type; values 0-2 either way
    vs_rev = chg.astype("int8") + cap.astype("int8")
    if not quiet:
        print(f"Vs check (Vp/Vs < {vmin:g} or dws_s < {dmin:g}): met at {int(rev.sum())} covered nodes; Vs changed at "
              f"{int(chg.sum())} ({int((chg & low_r).sum())} on Vp/Vs, {int((chg & low_d).sum())} on dws_s), "
              f"{int(cap.sum())} of them capped at Vp / {vmin:g}", flush=True)
    return dict(out=out, start=start, Rg=Rg, dsum=dsum, res=res, fall=fall, vs_rev=vs_rev, used=used, den=den,
                rev=rev, chg=chg, cap=cap, low_r=low_r, low_d=low_d, vmin=vmin, dmin=dmin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("version", nargs="?")
    ap.add_argument("description", nargs="?")
    ap.add_argument("--nll", action="store_true")
    ap.add_argument("--status", default="candidate")
    ap.add_argument("--dry-run", help="write the merged netCDF here, do not register")
    ap.add_argument("--runs", help='test: "region:tag:maskregion:damp,..." instead of the phase-1 choice')
    ap.add_argument("--product", default="product", help="key of tomodd.merge holding choice, reference, grid, taper_km, "
                    "floor (default product = v0.3.x; product_v040 = the re-timed tiles)")
    a = ap.parse_args()
    global C
    C = M[a.product]
    assert 0 <= C["floor"] < 1, f"tomodd.merge.product.floor = {C['floor']}: must be in [0, 1) (R divides by 1 - floor)"
    lon, lat, z = grid_axes()
    VP0, VS0 = reference(lon, lat, z)
    tiles, dropped = tile_runs(a.runs)
    tl = [dict(region=r, tag=t, run=RUNS / r / t, mask_region=m, mask_damp=d) for r, t, m, d in tiles]
    GEO = geometry(tl, lon, lat, z, allow_nomask=bool(a.runs))
    B = blend(GEO, VP0, VS0)
    out, start, Rg, dsum, res, fall, vs_rev, used = (B[k] for k in ("out", "start", "Rg", "dsum", "res", "fall",
                                                                     "vs_rev", "used"))
    den, rev, chg, cap, low_r, low_d, vmin, dmin = (B[k] for k in ("den", "rev", "chg", "cap", "low_r", "low_d",
                                                                   "vmin", "dmin"))
    ncov, nomask, shp = GEO["ncov"], GEO["nomask"], GEO["shape"]
    tr = lambda a: np.ascontiguousarray(np.asarray(a).reshape(shp).transpose(2, 1, 0)).astype("float32")
    vp, vs, vp0, vs0 = out["P"], out["S"], start["P"], start["S"]
    src = "data-derived: merge of tomoDD-SE tile inversions (b2_merge.py)"
    start_src = {ph: f"D-B049: R x (the tiles' starts, D-B005 x (1 + change of {C['reference']}) at their own nodes, "
                     f"blended with the merge weights) + (1 - R) x the reference (that rule on this grid), R = "
                     f"tile_weight_{ph.lower()}; the reference where R = 0 or no tile covers" for ph in "PS"}
    start_src["S"] += (f"; D-B051: the reference Vs wherever Vp/Vs < {vmin:g} or dws_s < {dmin:g} (vs_reverted > 0 "
                       "marks where that changed Vs)")
    V = {
        "Vp": (tr(vp), dict(long_name="P-wave velocity", display_name="Vp (km/s)", units="km/s", source=src)),
        "Vs": (tr(vs), dict(long_name="S-wave velocity", display_name="Vs (km/s)", units="km/s", source=src)),
        "VpVs": (tr(vp / vs), dict(long_name="Vp/Vs ratio", display_name="Vp/Vs", units="1", source="Vp / Vs")),
        "Vp_start": (tr(vp0), dict(long_name="P-wave velocity of the starting model",
                                   display_name="Vp start (km/s)", units="km/s", source=start_src["P"])),
        "Vs_start": (tr(vs0), dict(long_name="S-wave velocity of the starting model",
                                   display_name="Vs start (km/s)", units="km/s", source=start_src["S"])),
        "Vp_perturbation": (tr(100 * (vp / vp0 - 1)), dict(long_name="Vp change from the starting model",
                                                           display_name="dVp/Vp (%)", units="percent",
                                                           source="100 (Vp / Vp_start - 1)")),
        "Vs_perturbation": (tr(100 * (vs / vs0 - 1)), dict(long_name="Vs change from the starting model",
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
        "tile_weight_p": (tr(Rg["P"]), dict(
            long_name="share of the tiles' blend in Vp (R = max over the covering tiles of the P resolution window "
                      "without floor)", display_name="tile weight P", units="1",
            source="D-B049: Vp = R Vp_tiles + (1 - R) Vp_reference; 0 = the reference (M0), do not interpret")),
        "tile_weight_s": (tr(Rg["S"]), dict(
            long_name="share of the tiles' blend in Vs (R = max over the covering tiles of the S resolution window "
                      "without floor)", display_name="tile weight S", units="1",
            source="D-B049: Vs = R Vs_tiles + (1 - R) Vs_reference; 0 = the reference (M0), do not interpret; "
                   "before the vs_reverted check")),
        "vs_reverted": (tr(vs_rev).astype("int8"), dict(
            long_name="Vs set to the reference Vs at this node", display_name="Vs reverted", units="1",
            flag_values=np.array([0, 1, 2], "int8"),
            flag_meanings="unchanged reverted_to_reference reverted_and_capped",
            source=f"D-B051: 1 = Vp/Vs < {vmin:g} or dws_s < {dmin:g}, Vs set to the reference Vs (Vs_start); "
                   f"2 = also capped at Vp / {vmin:g}; 0 = unchanged, including nodes whose Vs was the reference "
                   "already (tile_weight_s = 0) (params tomodd.merge.vpvs_min, dws_s_min). Stored as int8 (byte) in "
                   "the Zarr store and the netCDF file: netCDF-4 Classic has no unsigned types")),
    }
    ds = xr.Dataset({k: (("depth", "latitude", "longitude"), a, at) for k, (a, at) in V.items()},
                    coords=EX.coords(z, lat, lon))
    prov = dict(run="merge: " + ", ".join(f"{u['region']}/{u['tag']}" for u in used), tiles=json.dumps(used),
                reference=C["reference"], weights=json.dumps({k: C[k] for k in ("taper_km", "floor")}),
                start_model=f"D-B005 x (1 + change of {C['reference']}) at each tile's nodes")
    merge_attrs = dict(                                                       # add_attrs replaces ds.attrs
        merge_rules="D-B049 resolution-gated merge V = R V_tiles + (1 - R) V_ref; D-B050 excluded tiles; "
                    "D-B051 Vs reverted to the reference where Vp/Vs < vpvs_min or dws_s < dws_s_min",
        excluded_tiles=", ".join(dropped) if dropped else ("none (--runs)" if a.runs else "none"),
        gated_tiles=", ".join(u_["region"] for u_ in used if u_["gated"]) or "none", gate_max_smooth=GATE,
        tiles_without_mask=", ".join(nomask) or "none",                     # --runs only (else SystemExit)
        covered_nodes_p=int((den["P"] > 0).sum()), covered_nodes_s=int((den["S"] > 0).sum()),
        reference_fallback_fraction_p=round(fall["P"][0], 4), reference_fallback_fraction_s=round(fall["S"][0], 4),
        reference_partial_fraction_p=round(fall["P"][1], 4), reference_partial_fraction_s=round(fall["S"][1], 4),
        vpvs_min=vmin, dws_s_min=dmin, vs_check_nodes=int(rev.sum()), vs_reverted_nodes=int(chg.sum()),
        vs_reverted_vpvs_nodes=int((chg & low_r).sum()), vs_reverted_dws_s_nodes=int((chg & low_d).sum()),
        vs_capped_nodes=int(cap.sum()))
    if a.dry_run:
        EX.add_attrs(ds, "0.0.0-dryrun", "dry run", prov, "b2_merge.py --dry-run").attrs.update(merge_attrs)
        ds.to_netcdf(a.dry_run, format="NETCDF4_CLASSIC")
        print(f"dry run written to {a.dry_run}")
        return
    EX.add_attrs(ds, a.version, a.description, prov, "b2_merge.py").attrs.update(merge_attrs)
    EX.export(ds, a.version, a.description, a.nll, a.status)


if __name__ == "__main__":
    main()

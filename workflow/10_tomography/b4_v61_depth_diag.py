"""Track B, catalog v6.1 (D-B100): why do the NLL-SSST station terms of catalog v6.0 move depths? Diagnosis on the v6.0
outputs (read only): the Axial tile t07_32 (23,616 events, the OOI cabled array), passes it0 (no terms) .. it4
(SSST stopped at 4 km) and the Loc2ssst corrections g0 (CharDist 9999 km: one static per station-phase), g1 (16 km),
g2 (8 km), g3 (4 km).

What it computes:
1. per pass, for a fixed event set (the tile's Axial events with v6.0 tier A-B, LOCATED in every pass): depth below
   the seafloor (true depth + merged surface), share within 1.6 / 2.0 km, floor-limited share, median depth;
2. per station-phase: distance and azimuth from the cluster, depth sensitivity dT/dz at the cluster centroid (base time
   grid, central differences), it0 mean residual, the Loc2ssst increments g0..g3 and their sum at the event positions;
3. per event, the linearised relocation the corrections imply: dm = -(G'WG)^-1 G'W c, with G = [dT/dx, dT/dy, dT/dz, 1]
   at the it0 position (base grids), W the NLLoc arrival weights of it0 and c the cumulative correction at the it0
   position; dz is split exactly into the contributions of station-phase groups (caldera P, caldera S, far P, far S)
   and into the parts of c that are common to all P, common to all S (each phase's weighted mean over the event's
   arrivals) and the rest;
4. the it0 residual pattern of floor-limited and free events (does the floor make the terms one-sided?).

  python b4_v61_depth_diag.py [--tile t07_32] -> /wd1/mdenolle_data/reloc_v61/depth_test/diag/
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "08_relocation"))
from b4_catalog import floor_at, read_floor, unflatten, surface_km  # noqa: E402
from b4_v61_grids import Grid, parse_hyp  # noqa: E402

V60 = Path("/wd1/mdenolle_data/reloc_v6/v60")
CAT = Path("/wd1/mdenolle_data/reloc_v6/catalog/catalog_b4_v6.0.csv")
OUT = Path("/wd1/mdenolle_data/reloc_v61/depth_test/diag")
FLOOR = Path("/wd1/mdenolle_data/reloc_v5/topo/topo_15s_cons_merged_eft.bin")
PASSES = 5
NEAR_KM = 10.0          # caldera stations (OOI AX* except AXBA1) are within ~5 km of the cluster


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_passes(tile, ids):
    evs, phs = {}, {}
    for k in range(PASSES):
        ev, ph = parse_hyp(V60 / f"it{k}" / "hyp" / f"{tile}.hyp", ids=ids)
        ev = ev[ev.status == "LOCATED"].drop_duplicates("ID").set_index("ID")
        evs[k], phs[k] = ev, ph[ph.ID.isin(ev.index)]
        log(f"it{k}: {len(ev)} located, {len(phs[k])} arrivals")
    return evs, phs


def frame(ev, fl):
    e = ev.copy()
    e["depth_t"] = unflatten(e.depth)
    e["surface_km"] = surface_km(e.lat.values, e.lon.values)
    e["bsf"] = e.depth_t + e.surface_km                     # km below the solid surface (seafloor)
    e["h_floor"] = e.depth.to_numpy(float) + floor_at(fl, e.lat.values, e.lon.values) / 1e3
    e["floor_limited"] = e.h_floor < 1.5 * e.ns_z
    return e


def wavg(v, w):
    return float(np.sum(v * w) / np.sum(w)) if len(v) and np.sum(w) > 0 else 0.0


def corrections(tile, keys, pts):
    """increments g0..g3 (Loc2ssst SSST_TIMECORR grids) of each station-phase at points pts (N x 3); 0 where the
    station-phase kept its previous grid (no corrected grid written in that pass)."""
    out = {}
    for k in range(PASSES - 1):
        for key in keys:
            f = V60 / f"g{k}" / tile / f"cascadia3d.{key}.ssst.hdr"
            out[(k, key)] = Grid(f).at(pts) if f.exists() else np.zeros(len(pts))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", default="t07_32")
    ap.add_argument("--region", default="Axial")
    ap.add_argument("--near-km", type=float, default=NEAR_KM)
    a = ap.parse_args()
    od = OUT / a.tile
    od.mkdir(parents=True, exist_ok=True)
    cat = pd.read_csv(CAT, usecols=["ID", "region", "tile", "tier", "depth", "surface_km"], low_memory=False)
    sel = cat[(cat.tile == a.tile) & (cat.region == a.region)]
    ids = set(sel.ID)
    tier = sel.set_index("ID").tier
    log(f"{a.tile} {a.region}: {len(ids)} events in v6.0")
    evs, phs = load_passes(a.tile, ids)
    fl = read_floor(FLOOR)
    fr = {k: frame(evs[k], fl) for k in range(PASSES)}
    common = sorted(set.intersection(*[set(fr[k].index) for k in range(PASSES)]))
    log(f"located in every pass: {len(common)}")

    # 1. per-pass depth statistics on a fixed event set (v6.0 tiers)
    rows = []
    for tl, tt in (("A-B", "AB"), ("A-C", "ABC"), ("A-D", "ABCD")):
        idx = [i for i in common if tier.get(i) in list(tt)]
        for k in range(PASSES):
            e = fr[k].loc[idx]
            rows.append(dict(tiers=tl, it=k, n=len(e), within_1_6=100 * (e.bsf <= 1.6).mean(),
                             within_2_0=100 * (e.bsf <= 2.0).mean(), median_bsf=e.bsf.median(),
                             floor_limited=100 * e.floor_limited.mean(),
                             dz_med=(e.depth_t - fr[0].loc[idx].depth_t).median(),
                             rms_med=e.rms.median()))
    P = pd.DataFrame(rows)
    P.round(3).to_csv(od / "by_pass.csv", index=False)
    print(P.round(3).to_string())

    # 2. station-phases: geometry, depth sensitivity, residuals, corrections
    ph0 = phs[0].copy()
    ph0["key"] = ph0.phase + "." + ph0.sta
    keys = sorted(ph0.key.unique())
    base = {}
    for key in keys:
        f = V60 / "base" / a.tile / f"cascadia3d.{key}.time.hdr"
        if f.exists():
            base[key] = Grid(f)
    e0 = fr[0].loc[common]
    cen = np.array([e0.x.median(), e0.y.median(), e0.z.median()])
    srows = []
    for key in keys:
        g = base.get(key)
        if g is None:
            continue
        sx, sy, sz = g.sta[1:]
        dist = float(np.hypot(sx - cen[0], sy - cen[1]))
        az = float(np.degrees(np.arctan2(sx - cen[0], sy - cen[1])) % 360)
        gr = g.grad(cen)[0]
        r = ph0[ph0.key == key]
        srows.append(dict(key=key, phase=key[0], sta=key[2:], n_it0=len(r), dist_km=dist, az=az,
                          dTdx=gr[0], dTdy=gr[1], dTdz=gr[2], res_it0_mean=float(np.average(r.res, weights=r.weight)) if r.weight.sum() > 0 else np.nan,
                          res_it0_median=float(r.res.median())))
    S = pd.DataFrame(srows).set_index("key")

    # corrections at each event's it0 position
    pts = e0[["x", "y", "z"]].to_numpy()
    inc = corrections(a.tile, list(S.index), pts)
    for k in range(PASSES - 1):
        S[f"inc_g{k}_median"] = [float(np.median(inc[(k, key)])) for key in S.index]
    S["cum_g3_median"] = S[[f"inc_g{k}_median" for k in range(PASSES - 1)]].sum(axis=1)
    S["near"] = S.dist_km <= a.near_km
    S.round(5).to_csv(od / "station_phases.csv")
    print(S[S.n_it0 >= 50].round(3).to_string())

    # 3. linearised relocation per event from the cumulative corrections (at it0 positions), and its attribution
    pos = {i: j for j, i in enumerate(common)}
    ph0 = ph0[ph0.ID.isin(pos) & ph0.key.isin(S.index) & (ph0.weight > 0)].reset_index(drop=True)
    ph0["j"] = ph0.ID.map(pos)
    # gradients at each arrival's event position
    G = np.zeros((len(ph0), 4))
    G[:, 3] = 1.0
    for key, r in ph0.groupby("key"):
        G[r.index.to_numpy(), :3] = base[key].grad(pts[r.j.to_numpy()])
    cum = {}
    for k in range(1, PASSES):
        c = np.zeros(len(ph0))
        for key, r in ph0.groupby("key"):
            ii = r.index.to_numpy()
            c[ii] = sum(inc[(j, key)][r.j.to_numpy()] for j in range(k))
        cum[k] = c
    ph0["near"] = ph0.key.map(S.near).to_numpy()
    groups = {"caldera P": (ph0.near & (ph0.phase == "P")).to_numpy(),
              "caldera S": (ph0.near & (ph0.phase == "S")).to_numpy(),
              "far P": (~ph0.near & (ph0.phase == "P")).to_numpy(),
              "far S": (~ph0.near & (ph0.phase == "S")).to_numpy()}
    w = ph0.weight.to_numpy()
    isP = (ph0.phase == "P").to_numpy()
    lin = []
    order = np.argsort(ph0.j.to_numpy(), kind="stable")
    jj = ph0.j.to_numpy()[order]
    starts = np.searchsorted(jj, np.arange(len(common)))
    ends = np.r_[starts[1:], len(jj)]
    for n, i in enumerate(common):
        ix = order[starts[n]:ends[n]]
        if len(ix) < 5:
            continue
        Gi, Wi = G[ix], w[ix]
        A = Gi.T @ (Wi[:, None] * Gi)
        try:
            K = -np.linalg.solve(A, (Gi * Wi[:, None]).T)          # 4 x n: dm = K c
        except np.linalg.LinAlgError:
            continue
        row = dict(ID=i, n_arr=len(ix), n_P=int(isP[ix].sum()))
        # depth / common-S-term trade-off: sigma_z with a free common S term over sigma_z without (same weights)
        sv = (~isP[ix]).astype(float)
        if 0 < sv.sum() < len(ix):
            Ga = np.c_[Gi, sv]
            try:
                Ca = np.linalg.inv(Ga.T @ (Wi[:, None] * Ga))
                C0 = np.linalg.inv(A)
                row["sigz_ratio_commonS"] = float(np.sqrt(Ca[2, 2] / C0[2, 2]))
                row["corr_z_commonS"] = float(Ca[2, 4] / np.sqrt(Ca[2, 2] * Ca[4, 4]))
            except np.linalg.LinAlgError:
                pass
        for k in range(1, PASSES):
            c = cum[k][ix]
            dm = K @ c
            row[f"dz_lin_it{k}"] = dm[2]
            row[f"dz_obs_it{k}"] = fr[k].loc[i, "depth_t"] - fr[0].loc[i, "depth_t"]
            if k in (1, PASSES - 1):
                for gname, gm in groups.items():
                    row[f"dz_{gname.replace(' ', '_')}_it{k}"] = K[2] @ (c * gm[ix])
                # common P / common S / rest (each phase's weighted mean over this event's arrivals)
                mP, mS = isP[ix], ~isP[ix]
                cp = np.where(mP, wavg(c[mP], Wi[mP]), 0.0)
                cs = np.where(mS, wavg(c[mS], Wi[mS]), 0.0)
                row[f"dz_commonP_it{k}"] = K[2] @ cp
                row[f"dz_commonS_it{k}"] = K[2] @ cs
                row[f"dz_rest_it{k}"] = K[2] @ (c - cp - cs)
                row[f"SmP_common_it{k}"] = wavg(c[mS], Wi[mS]) - wavg(c[mP], Wi[mP])
        lin.append(row)
    L = pd.DataFrame(lin).set_index("ID")
    L["tier"] = tier.reindex(L.index)
    L["floor_limited_it0"] = fr[0].floor_limited.reindex(L.index)
    L["bsf_it0"] = fr[0].bsf.reindex(L.index)
    L.round(5).to_csv(od / "linearised.csv")
    ab = L[L.tier.isin(list("AB"))]
    summ = {}
    for k in range(1, PASSES):
        summ[f"it{k}"] = dict(dz_obs_median=float(ab[f"dz_obs_it{k}"].median()),
                              dz_lin_median=float(ab[f"dz_lin_it{k}"].median()),
                              corr=float(np.corrcoef(ab[f"dz_obs_it{k}"], ab[f"dz_lin_it{k}"])[0, 1]))
    for k in (1, PASSES - 1):
        summ[f"attribution_it{k}"] = {c.replace(f"_it{k}", ""): float(ab[c].median()) for c in ab.columns
                                      if c.startswith("dz_") and c.endswith(f"_it{k}") and "obs" not in c
                                      and "lin" not in c}
        summ[f"attribution_mean_it{k}"] = {c.replace(f"_it{k}", ""): float(ab[c].mean()) for c in ab.columns
                                           if c.startswith("dz_") and c.endswith(f"_it{k}")}
        summ[f"S_minus_P_common_median_it{k}"] = float(ab[f"SmP_common_it{k}"].median())

    # 4. it0 residuals of floor-limited and free events at caldera stations, by phase
    r0 = phs[0].copy()
    r0["key"] = r0.phase + "." + r0.sta
    r0["near"] = r0.key.map(S.near)
    r0["fl"] = r0.ID.map(fr[0].floor_limited)
    r0["tier"] = r0.ID.map(tier)
    res = (r0[r0.tier.isin(list("AB"))].groupby(["near", "phase", "fl"])
           .apply(lambda g: pd.Series(dict(n=len(g), mean=np.average(g.res, weights=g.weight) if g.weight.sum() > 0 else np.nan, median=g.res.median())))
           .reset_index())
    res.round(4).to_csv(od / "it0_residuals_floor.csv", index=False)
    summ["it0_residuals_floor"] = res.round(4).to_dict(orient="records")
    summ["by_pass_AB"] = P[P.tiers == "A-B"].round(3).to_dict(orient="records")
    if "sigz_ratio_commonS" in ab:
        summ["sigz_ratio_commonS_median"] = float(ab.sigz_ratio_commonS.median())
        summ["corr_z_commonS_median"] = float(ab.corr_z_commonS.median())
    summ["centroid_xyz"] = cen.tolist()
    summ["events_common"] = len(common)
    (od / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float))
    log(f"-> {od}")


if __name__ == "__main__":
    main()

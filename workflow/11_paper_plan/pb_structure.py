#!/usr/bin/env python3
"""Structure feasibility per plate-boundary segment (D-B107): which segments does model v0.4.2 resolve near the
faults, how certain is it there, and how large are across-fault and along-strike contrasts against the ensemble?

Per segment (pb_common.SEGMENTS; events = located earthquakes of v6.0, tiers A-C, waveform-review P1 left out):
* reference set = the segment's fault-surface v1 mesh vertices (surfaces whose fit members mostly fall in the
  segment) together with its seismicity (depth-good events where the segment has >= 50 of them, else tier A-C);
  a node's distance is the 3-D distance to the nearest reference point;
* resolved nodes within 5, 10 and 20 km (masks of pb_common.Model: P, S and both), share passing the checkerboard
  tests, depth range, median calibrated ensemble 1 sigma (Vp x1.50, Vs x1.35, Vp/Vs raw, D-B111);
* contrasts, depth-matched (per depth layer, then averaged with weights min(n1, n2)):
    near - far     nodes within 5 km against nodes 10-20 km away (fault-zone signature);
    across         nodes within 20 km on the two sides of the segment's strike line (PCA of epicentres; nodes within
                   3 km of the line left out);
    along          first against last third (along strike) of the nodes within 10 km;
  each contrast is recomputed in every ensemble member (same nodes); its 1 sigma is the member spread times the
  calibration factor; |contrast| / sigma is reported.
* the 8 km/s depth of each event's model column, where the crossing lies between two P-resolved nodes.

Writes data/paper_plan/plate_boundaries/structure_feasibility.csv, structure_contrasts.csv, segment_axes.csv and
/wd1/mdenolle_data/paper_plan/events_segments_v60.parquet (events with segment, z8 and slab columns).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb_common as P  # noqa: E402

VARS = ["Vp", "Vs", "VpVs"]
MASKS = {"P": "mP", "S": "mS", "PS": "mPS"}


def local_xy(lon, lat, lon0, lat0):
    return ((np.asarray(lon) - lon0) * P.KM * np.cos(np.radians(lat0)), (np.asarray(lat) - lat0) * P.KM)


def seg_axis(e: pd.DataFrame):
    """Centroid and strike of a segment from the PCA of its epicentres (tier A-C)."""
    lon0, lat0 = float(e.lon.median()), float(e.lat.median())
    x, y = local_xy(e.lon, e.lat, lon0, lat0)
    X = np.c_[x, y]
    w, V = np.linalg.eigh(np.cov(X.T))
    u = V[:, 1]                                       # major axis (unit, east/north)
    if u[0] < 0 or (u[0] == 0 and u[1] < 0):          # orient along the strike azimuth in [0, 180)
        u = -u
    strike = (np.degrees(np.arctan2(u[0], u[1])) + 360) % 360
    aspect = float(np.sqrt(w[1] / max(w[0], 1e-9)))
    return dict(lon0=lon0, lat0=lat0, ux=u[0], uy=u[1], strike=strike, aspect=aspect,
                length_km=float(np.ptp(X @ u)), width_km=float(np.ptp(X @ np.r_[-u[1], u[0]])))


def depth_matched(vals, layer, g1, g2):
    """Depth-matched mean difference g1 - g2 of `vals` ([..., n_nodes]) over layers holding both groups."""
    num, den = 0.0, 0.0
    for k in np.unique(layer[g1 | g2]):
        a, b = g1 & (layer == k), g2 & (layer == k)
        if a.sum() == 0 or b.sum() == 0:
            continue
        w = min(a.sum(), b.sum())
        num = num + w * (vals[..., a].mean(-1) - vals[..., b].mean(-1))
        den += w
    if den == 0:
        return None, 0
    return num / den, den


def xr_shallow():
    import xarray as xr
    return xr.open_dataset(P.MODEL)["shallow_crust_node"].values == 1


def main():
    P.BIG.mkdir(parents=True, exist_ok=True)
    P.OUT.mkdir(parents=True, exist_ok=True)
    c = P.assign_segments(P.load_catalog())
    m = P.Model()
    iy, ix = m.column_index(c.lon.values, c.lat.values)
    for lev, tag in ((8.0, "z8"), (7.5, "z75")):
        zz = m.z8(lev)
        c[tag] = zz[iy, ix]
        c[f"{tag}_bsf"] = c[tag] + m.surf[iy, ix]       # level depth below the node-column surface
        c[f"rel_{tag}"] = c.depth - c[tag]              # > 0: event below the level
    c.to_parquet(P.BIG / f"events_segments_{P.TAG}.parquet")
    E = c[c.is_eq & c.abc & ~c.review_P1]
    summ, mem, vert = P.fault_tables()
    sseg = P.surface_segments(summ, mem, c.set_index("ID").seg)
    sseg.to_csv(P.OUT / "surface_segments.csv", index=False)
    names, mem_vals = P.load_members()
    P.log(f"members: {len(names)}")
    # level depths in every member (same P mask): their spread is the structure uncertainty of the level
    zl = {tag: np.stack([m.z8(lev, mem_vals["Vp"][i]) for i in range(len(names))])
          for lev, tag in ((8.0, "z8"), (7.5, "z75"))}
    np.savez_compressed(P.BIG / "level_depth_members.npz", names=np.array(names), **zl)
    rock = m.rock
    flat = lambda a: a.reshape(-1)                     # noqa: E731
    xyz = m.xyz.reshape(-1, 3)
    layer = np.broadcast_to(np.arange(len(m.dep))[:, None, None], m.Vp.shape).reshape(-1)
    LOf, LAf = flat(np.broadcast_to(m.LO, m.Vp.shape)), flat(np.broadcast_to(m.LA, m.Vp.shape))
    depf = flat(m.D)
    rows, crow, axes = [], [], []
    node_sets = {}                                     # contrast node sets, reused by pb_circularity.py
    scn = flat(np.asarray(xr_shallow(), bool))
    for seg in P.SEG_ORDER:
        e = E[E.seg == seg]
        if len(e) == 0:
            continue
        good = e[e.depth_good]
        ev = good if len(good) >= 50 else e
        srf = sseg[sseg.seg == seg].surface_id
        v = vert[vert.surface_id.isin(srf)]
        ref_ev = P.ecef(ev.lon.values, ev.lat.values, ev.depth.values)
        ref_f = P.ecef(v.lon.values, v.lat.values, v.depth_km.values) if len(v) else np.zeros((0, 3))
        ref = np.vstack([ref_ev, ref_f])
        dist = np.full(xyz.shape[0], np.inf)
        rk = flat(rock)
        d, _ = cKDTree(ref).query(xyz[rk], distance_upper_bound=25.0)
        dist[rk] = d
        dfl = np.full(xyz.shape[0], np.inf)
        if len(ref_f):
            d2, _ = cKDTree(ref_f).query(xyz[rk], distance_upper_bound=25.0)
            dfl[rk] = d2
        ax = seg_axis(e)
        ax.update(seg=seg, n_events=len(e), n_ref_events=len(ev), ref_events="depth-good" if ev is good else "tier A-C",
                  n_surfaces=int(len(srf)), n_vertices=int(len(v)))
        axes.append(ax)
        x, y = local_xy(LOf, LAf, ax["lon0"], ax["lat0"])
        s_along = x * ax["ux"] + y * ax["uy"]
        s_norm = -x * ax["uy"] + y * ax["ux"]
        row = dict(seg=seg, name=P.SEGMENTS[seg][0], type=P.SEGMENTS[seg][1], n_events=len(e),
                   n_depth_good=int(e.depth_good.sum()), n_surfaces=int(len(srf)),
                   ref_events=ax["ref_events"])
        for mk, attr in MASKS.items():
            msk = flat(getattr(m, attr))
            for r in (5, 10, 20):
                row[f"n{mk}_{r}"] = int((msk & (dist <= r)).sum())
            if len(ref_f):
                row[f"n{mk}_10_faults"] = int((msk & (dfl <= 10)).sum())
        for mk, cb in (("P", m.cbP), ("S", m.cbS)):
            msk = flat(getattr(m, MASKS[mk]))
            sel = msk & (dist <= 10)
            row[f"cb_share{mk}_10"] = float(flat(cb)[sel].mean()) if sel.any() else np.nan
        cbsel = flat(m.cbP) & (dist <= 10)
        row["cb_cell_P_med_km"] = float(np.median(flat(m.res_p)[cbsel])) if cbsel.any() else np.nan
        row["cb_cell_P_min_km"] = float(np.min(flat(m.res_p)[cbsel])) if cbsel.any() else np.nan
        selP = flat(m.mP) & (dist <= 10)
        selPS = flat(m.mPS) & (dist <= 10)
        selS = flat(m.mS) & (dist <= 10)
        row["depth_min_P10"] = float(depf[selP].min()) if selP.any() else np.nan
        row["depth_max_P10"] = float(depf[selP].max()) if selP.any() else np.nan
        row["layers_P10"] = int(len(np.unique(layer[selP])))
        row["layers_PS10"] = int(len(np.unique(layer[selPS])))
        # depth layers holding >= 3 resolved nodes within 10 km (the usable depth span)
        for mk, sel in (("P", selP), ("PS", selPS)):
            kk, nn = np.unique(layer[sel], return_counts=True)
            kk = kk[nn >= 3]
            row[f"span_{mk}10"] = (f"{m.dep[kk.min()]:.0f}-{m.dep[kk.max()]:.0f}" if len(kk) else "")
        row["ev_depth_p10"] = float(np.percentile(ev.depth, 10))
        row["ev_depth_p90"] = float(np.percentile(ev.depth, 90))
        for vname, sel in (("Vp", selP), ("Vs", selS), ("VpVs", selPS)):
            sd = flat(m.sd(vname))
            row[f"sd_{vname}_10"] = float(np.nanmedian(sd[sel])) if sel.any() else np.nan
            vv = flat(getattr(m, vname))
            row[f"med_{vname}_10"] = float(np.nanmedian(vv[sel])) if sel.any() else np.nan
        # share of segment events whose own node column/depth is P-resolved (nearest node)
        miz = np.clip(np.searchsorted(0.5 * (m.dep[1:] + m.dep[:-1]), e.depth.values), 0, len(m.dep) - 1)
        eiy, eix = m.column_index(e.lon.values, e.lat.values)
        row["ev_in_P"] = float(m.mP[miz, eiy, eix].mean())
        row["ev_in_PS"] = float(m.mPS[miz, eiy, eix].mean())
        rows.append(row)
        # contrasts
        for vname in VARS:
            msk = flat(m.mask(vname))
            det = flat(getattr(m, vname))
            mv = mem_vals[vname].reshape(len(names), -1)
            defs = {
                "near-far": (msk & (dist <= 5), msk & (dist > 10) & (dist <= 20)),
                "across": (msk & (dist <= 20) & (s_norm > 3), msk & (dist <= 20) & (s_norm < -3)),
            }
            sel10 = msk & (dist <= 10)
            if sel10.sum() >= 6:
                q1, q2 = np.quantile(s_along[sel10], [1 / 3, 2 / 3])
                defs["along"] = (sel10 & (s_along >= q2), sel10 & (s_along <= q1))
            for cname, (g1, g2) in defs.items():
                key = f"{seg}|{vname}|{cname}"
                node_sets[key + "|1"], node_sets[key + "|2"] = np.flatnonzero(g1), np.flatnonzero(g2)
                dd, wsum = depth_matched(det[None, :], layer, g1, g2)
                if dd is None:
                    crow.append(dict(seg=seg, var=vname, contrast=cname, n1=int(g1.sum()), n2=int(g2.sum()),
                                     note="no shared depth layer"))
                    continue
                dm, _ = depth_matched(mv, layer, g1, g2)
                dm = dm[np.isfinite(dm)]
                sig = float(np.std(dm) * P.INFL[vname]) if len(dm) > 5 else np.nan
                med_node_sd = float(np.nanmedian(flat(m.sd(vname))[g1 | g2]))
                crow.append(dict(seg=seg, var=vname, contrast=cname, n1=int(g1.sum()), n2=int(g2.sum()),
                                 weight=float(wsum), value=float(dd[0]), sigma_members=sig,
                                 ratio=float(abs(dd[0]) / sig) if sig and np.isfinite(sig) and sig > 0 else np.nan,
                                 member_median=float(np.median(dm)) if len(dm) else np.nan,
                                 frac_members_same_sign=float(np.mean(np.sign(dm) == np.sign(dd[0]))) if len(dm)
                                 else np.nan,
                                 median_node_sd=med_node_sd,
                                 frac_nodes_vs_rule=float(scn[g1 | g2].mean()) if vname != "Vp" else 0.0,
                                 ratio_node_sd=float(abs(dd[0]) / med_node_sd) if med_node_sd > 0 else np.nan))
        P.log(seg, row["nP_10"], row["nPS_10"], row["sd_Vp_10"])
    np.savez_compressed(P.BIG / "contrast_nodes.npz", layer=layer, **node_sets)
    f = pd.DataFrame(rows)
    ct = pd.DataFrame(crow)
    # classification
    best = ct.dropna(subset=["ratio"]).groupby("seg").ratio.max() if len(ct) else pd.Series(dtype=float)
    f["max_contrast_ratio"] = f.seg.map(best)

    def cls(r):
        if r.nP_10 >= 15 and r.nPS_10 >= 10 and r.layers_PS10 >= 3:
            return "resolved"
        if r.nP_10 >= 5:
            return "marginal"
        return "not resolved"
    f["class"] = f.apply(cls, axis=1)
    f.to_csv(P.OUT / "structure_feasibility.csv", index=False, float_format="%.4g")
    ct.to_csv(P.OUT / "structure_contrasts.csv", index=False, float_format="%.4g")
    pd.DataFrame(axes).to_csv(P.OUT / "segment_axes.csv", index=False, float_format="%.5g")
    P.log(f[["seg", "n_events", "nP_5", "nP_10", "nP_20", "nPS_10", "layers_PS10", "sd_Vp_10", "sd_VpVs_10",
             "max_contrast_ratio", "class"]].to_string())


if __name__ == "__main__":
    main()

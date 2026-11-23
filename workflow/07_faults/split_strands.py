"""Split composite transform surfaces into strands (map-view line detection), then refit them.

Some v0 surfaces on the Blanco transform are single planes fitted across several right-stepping
strands (the literature check: East Blanco and Surveyor depressions). This script takes the events
of those surfaces, plus unassigned events inside their footprint, and finds strands in map view,
where v4 locations are good (depths are not):

  * sequential RANSAC lines: lines through a seed and one event within 10 km, azimuth within
    --az-window deg of --azimuth (plate motion), scored by events within --tol km in a 25 km ball;
  * growth along the line: events within --tol, kept while connected along strike (gaps < --gap km),
    line refit, repeat; a strand needs >= --n-min events and >= 5 km length;
  * reassignment to the nearest strand (within --tol and inside its extent + 2 km), then merging of
    collinear strands (azimuths within 8 deg, offset < --tol, along-strike gap < --gap);
  * each strand gets a 3D rectangle with the same fit and uncertainty as fault_surfaces_planes.py;
    stability: detection rerun with two other seeds and tol -/+ 0.3 km (median Jaccard overlap).

The parent surfaces are removed (fault_surfaces_v0_removed.csv, reason "split into strands") and
the strands are added to every v0 file, with ids <parent><a, b, ...> after the parent that gave
most of their events. Literature fields are inherited from the parent and flagged for re-checking.

    python workflow/07_faults/split_strands.py BL001 BL002 BL006 BL009 BL010 BL020
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as MPath
from scipy.spatial import ConvexHull, cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fault_surfaces_v0 as v0  # noqa: E402
import fault_surfaces_planes as fp  # noqa: E402

V0 = v0.D / "faults" / "v0"
STEM = "fault_surfaces_v0"


def az_of(d):
    return np.degrees(np.arctan2(d[0], d[1])) % 180


def az_diff(a, b):
    d = abs(a - b) % 180
    return min(d, 180 - d)


def detect_lines(XY, a, rng, tol):
    N = len(XY)
    tree = cKDTree(XY)
    free = np.ones(N, bool)
    tried = np.zeros(N, bool)
    nb = tree.query_ball_point(XY, 3.0, return_length=True)
    order = np.argsort(-(nb * rng.uniform(0.8, 1.25, N)))
    lines = []
    for s in order:
        if not free[s] or tried[s]:
            continue
        ball = np.array(tree.query_ball_point(XY[s], 25.0)); ball = ball[free[ball]]
        near = ball[np.linalg.norm(XY[ball] - XY[s], axis=1) <= 10.0]
        if len(ball) < a.n_min or len(near) < 2:
            tried[s] = True; continue
        best = (0, None)
        Q = XY[ball] - XY[s]
        for _ in range(a.trials):
            j = rng.choice(near)
            d = XY[j] - XY[s]
            L = np.linalg.norm(d)
            if L < 1.0 or az_diff(az_of(d), a.azimuth) > a.az_window:
                continue
            d = d / L
            cnt = int((np.abs(Q @ np.array([-d[1], d[0]])) <= tol).sum())
            if cnt > best[0]:
                best = (cnt, d)
        if best[0] < a.n_min:
            tried[s] = True; continue
        d, c0, members, ext = best[1], XY[s].copy(), None, 25.0
        for _ in range(10):
            cand = np.array(tree.query_ball_point(c0, max(25.0, 0.75 * ext + 25.0))); cand = cand[free[cand]]
            Q = XY[cand] - c0
            cand = cand[np.abs(Q @ np.array([-d[1], d[0]])) <= tol]
            if len(cand) < a.n_min:
                break
            u = (XY[cand] - c0) @ d
            o = np.argsort(u); us = u[o]
            cuts = np.flatnonzero(np.diff(us) > a.gap)
            pieces = np.split(o, cuts + 1)
            us0 = (XY[s] - c0) @ d
            seg = min(pieces, key=lambda p: 0 if u[p].min() - 1e-9 <= us0 <= u[p].max() + 1e-9 else
                      min(abs(u[p].min() - us0), abs(u[p].max() - us0)) + 1e3)
            cand = cand[seg]
            if len(cand) < a.n_min:
                break
            c1 = XY[cand].mean(0)
            w, V = np.linalg.eigh(np.cov((XY[cand] - c1).T))
            d1 = V[:, np.argmax(w)]
            if az_diff(az_of(d1), a.azimuth) > a.az_window:
                break
            c0, d, ext = c1, d1, np.ptp((XY[cand] - c1) @ d1)
            if members is not None and len(cand) == len(members) and (np.sort(cand) == np.sort(members)).all():
                break
            members = cand
        if members is None or len(members) < a.n_min or np.ptp((XY[members] - c0) @ d) < 5.0:
            tried[s] = True; continue
        free[members] = False
        lines.append(members)
    return lines


def line_geom(XY, m):
    c = XY[m].mean(0)
    w, V = np.linalg.eigh(np.cov((XY[m] - c).T))
    d = V[:, np.argmax(w)]
    u = (XY[m] - c) @ d
    return c, d, u.min(), u.max()


def reassign_lines(XY, lines, a, tol):
    for _ in range(2):
        best = np.full(len(XY), np.inf); owner = -np.ones(len(XY), int)
        for i, m in enumerate(lines):
            c, d, u0, u1 = line_geom(XY, m)
            Q = XY - c
            u, w = Q @ d, np.abs(Q @ np.array([-d[1], d[0]]))
            ok = (w <= tol) & (u >= u0 - 2) & (u <= u1 + 2)
            better = ok & (w < best)
            best[better] = w[better]; owner[better] = i
        lines = [np.flatnonzero(owner == i) for i in range(len(lines))]
        lines = [m for m in lines if len(m) >= a.n_min]
    return lines


def separation(XY, A, B):
    """Ashman's D and separation (km) across strike where two near-parallel strands overlap along strike.
    Returns None when they are not parallel (> 12 deg), overlap < 3 km, or have < 20 events each there."""
    ca, da, _, _ = line_geom(XY, A); cb, db, _, _ = line_geom(XY, B)
    if az_diff(az_of(da), az_of(db)) > 12:
        return None
    d = da + np.sign(da @ db) * db; d /= np.linalg.norm(d); n = np.array([-d[1], d[0]])
    ua, ub = XY[A] @ d, XY[B] @ d
    lo, hi = max(ua.min(), ub.min()), min(ua.max(), ub.max())
    if hi - lo < 3:
        return None
    wa, wb = (XY[A] @ n)[(ua >= lo) & (ua <= hi)], (XY[B] @ n)[(ub >= lo) & (ub <= hi)]
    if len(wa) < 20 or len(wb) < 20:
        return None
    sep = abs(wa.mean() - wb.mean())
    return dict(D=float(np.sqrt(2) * sep / np.sqrt(wa.var() + wb.var())), sep_km=float(sep), overlap_km=float(hi - lo),
                sd_a=float(wa.std()), sd_b=float(wb.std()), n_a=int(len(wa)), n_b=int(len(wb)))


def merge_lines(XY, lines, a, tol):
    changed = True
    while changed:
        changed = False
        for i in range(len(lines)):
            ci, di, a0, a1 = line_geom(XY, lines[i])
            for j in range(i + 1, len(lines)):
                cj, dj, b0, b1 = line_geom(XY, lines[j])
                sp = separation(XY, lines[i], lines[j])
                if sp is not None and sp["D"] < a.min_d:       # parallel and overlapping but not separable
                    lines[i] = np.union1d(lines[i], lines[j]); lines.pop(j)
                    changed = True
                    break
                if az_diff(az_of(di), az_of(dj)) > 8:
                    continue
                if abs((cj - ci) @ np.array([-di[1], di[0]])) > tol:
                    continue
                uj = (XY[lines[j]] - ci) @ di
                gap = max(uj.min() - a1, a0 - uj.max(), 0)
                if gap > a.gap:
                    continue
                lines[i] = np.union1d(lines[i], lines[j]); lines.pop(j)
                changed = True
                break
            if changed:
                break
    return lines


def run(XY, a, seed, tol):
    rng = np.random.default_rng(seed)
    L = reassign_lines(XY, detect_lines(XY, a, rng, tol), a, tol)
    return reassign_lines(XY, merge_lines(XY, L, a, tol), a, tol)


def trace_row(r, v):
    KM = v0.KM
    s = np.radians(r["strike"]); eu = np.array([np.sin(s), np.cos(s)])
    kx = KM * np.cos(np.radians(r["lat"]))
    xy = np.c_[(v.lon - r["lon"]) * kx, (v.lat - r["lat"]) * KM]
    top = (v.depth_km <= v.depth_km.min() + 0.6).to_numpy()
    u = xy[top] @ eu
    tv = v[top].iloc[[int(np.argmin(u)), int(np.argmax(u))]]
    d = np.radians(max(r["dip"], 1)); dd = np.radians(r["strike"] + 90)
    off = r["depth_km"] / np.tan(d)
    cx, cy = -off * np.sin(dd), -off * np.cos(dd)
    L = r["length_km"] / 2
    ll = lambda p: (round(r["lon"] + p[0] / kx, 4), round(r["lat"] + p[1] / KM, 4))
    a_, b_ = ll((cx - L * eu[0], cy - L * eu[1])), ll((cx + L * eu[0], cy + L * eu[1]))
    return dict(top_edge_lon1=round(tv.lon.iloc[0], 4), top_edge_lat1=round(tv.lat.iloc[0], 4),
                top_edge_lon2=round(tv.lon.iloc[1], 4), top_edge_lat2=round(tv.lat.iloc[1], 4),
                sealevel_trace_lon1=a_[0], sealevel_trace_lat1=a_[1], sealevel_trace_lon2=b_[0], sealevel_trace_lat2=b_[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--azimuth", type=float, default=111.0, help="expected strand azimuth (deg), plate motion")
    ap.add_argument("--az-window", type=float, default=20.0)
    ap.add_argument("--tol", type=float, default=1.5, help="half-width of a strand in map view (km)")
    ap.add_argument("--gap", type=float, default=4.0)
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--n-min", type=int, default=30)
    ap.add_argument("--min-d", type=float, default=2.0, help="Ashman's D needed to keep overlapping parallel strands apart")
    ap.add_argument("--pool-name", default="")
    ap.add_argument("--draws", type=int, default=100)
    a = ap.parse_args()
    fa = argparse.Namespace(tol_h=1.5, tol_z=3.0, tol_k=1.5, pad=0.5, grid=1.0, draws=a.draws)

    summ = pd.read_csv(V0 / f"{STEM}_summary.csv")
    mem = pd.read_csv(V0 / f"{STEM}_members.csv")
    lit = pd.read_csv(V0 / f"{STEM}_literature.csv")
    parents = set(a.ids)
    if parents - set(summ.surface_id):
        raise SystemExit(f"not in product: {sorted(parents - set(summ.surface_id))}")
    c, _ = v0.load(1.0, 1e9)
    c = c[c.region != "AX"].reset_index(drop=True)
    pid = c.event_id.map(mem.set_index("event_id").surface_id)
    in_par = pid.isin(parents).to_numpy()
    # footprint of the parents' events (map view, 2 km buffer); add unassigned events inside it
    lon0, lat0 = c.longitude[in_par].median(), c.latitude[in_par].median()
    kx = v0.KM * np.cos(np.radians(lat0))
    XYall = np.c_[(c.longitude - lon0) * kx, (c.latitude - lat0) * v0.KM]
    hp = XYall[in_par][ConvexHull(XYall[in_par]).vertices]
    cen = hp.mean(0); hp = cen + (hp - cen) * (1 + 2.0 / np.linalg.norm(hp - cen, axis=1).mean())
    inside = MPath(hp).contains_points(XYall)
    pool = np.flatnonzero(in_par | (inside & pid.isna().to_numpy()))
    XY = XYall[pool]
    print(f"{in_par.sum():,} events in {len(parents)} parent surfaces + {len(pool) - in_par.sum():,} unassigned in the footprint")
    lines = run(XY, a, 0, a.tol)
    print(f"{len(lines)} strands; {sum(len(m) for m in lines):,} of {len(pool):,} pool events assigned")
    alt = [run(XY, a, 101, a.tol), run(XY, a, 202, a.tol), run(XY, a, 0, a.tol - 0.3), run(XY, a, 0, a.tol + 0.3)]

    rel, cf, sl = v0.relief(), v0.cfm(), v0.slab()
    rng = np.random.default_rng(0)
    pl = lit.set_index("surface_id")
    new_feats, new_rows, vrows, erows, mrows, trows, lrows = [], [], [], [], [], [], []
    count = {}
    strands = sorted(lines, key=lambda m: -len(m))
    for m in strands:
        idx = pool[m]
        s = fp.surface_rect(c, idx, fa, cf, sl, rel, rng)
        par = pid.iloc[idx].dropna()
        parent = par.mode()[0] if len(par) else sorted(parents)[0]
        count[parent] = count.get(parent, 0) + 1
        sid = f"{parent}{chr(96 + count[parent])}"
        p = s["props"]
        J = []
        for L in alt:
            lab = -np.ones(len(pool), int)
            for k, mm in enumerate(L):
                lab[mm] = k
            la = lab[m]; la = la[la >= 0]
            if len(la) == 0:
                J.append(0.0); continue
            B = np.flatnonzero(lab == np.bincount(la).argmax())
            J.append(len(np.intersect1d(m, B)) / len(np.union1d(m, B)))
        p.update(surface_id=sid, cluster_jaccard_median=round(float(np.median(J)), 2), stable=bool(np.median(J) >= 0.5),
                 kind="steep zone (fault candidate)" if p["dip"] >= 45 else "dipping zone",
                 strand_azimuth=round(float(az_of(line_geom(XY, m)[1])), 1), split_from=parent)
        pr = pl.loc[parent]
        p.update(lit_match=f"Strand split from {parent}: {pr.lit_match}", lit_confidence=pr.lit_confidence,
                 lit_class="known structure", lit_new_candidate=bool(pr.lit_new_candidate))
        props = dict(version="v0", catalog="Cascadia catalog v4", **p)
        props["normal_xyz"] = [round(float(x), 4) for x in s["normal"]]
        tri = [[[[round(float(s["lon"][i]), 5), round(float(s["lat"][i]), 5), round(float(-s["dep"][i] * 1000), 1)]
                 for i in (t0, t1, t2, t0)]] for t0, t1, t2 in s["tris"]]
        new_feats.append(dict(type="Feature", properties=props, geometry=dict(type="MultiPolygon", coordinates=tri)))
        new_rows.append(props)
        el = s["surface_elev_km"]
        v = pd.DataFrame({"surface_id": sid, "vertex": np.arange(len(s["lon"])), "lon": np.round(s["lon"], 5),
                          "lat": np.round(s["lat"], 5), "depth_km": np.round(s["dep"], 3), "sigma_boot_km": np.round(s["sb"], 3),
                          "sigma_loc_km": np.round(s["sl"], 3), "sigma_method_km": np.round(s["sm"], 3),
                          "sigma_total_km": np.round(s["st"], 3), "constrained": s["constrained"].astype(int),
                          "surface_elev_km": np.round(el, 3)})
        vrows.append(v)
        W = np.round(s["Wt"], 3)
        erows.append(pd.DataFrame({"surface_id": sid, "vertex": np.tile(np.arange(W.shape[1]), W.shape[0]),
                                   "draw": np.repeat(np.arange(W.shape[0]), W.shape[1]), "normal_offset_km": W.ravel()}))
        mrows.append(pd.DataFrame({"event_id": c.event_id.values[idx], "surface_id": sid}))
        trows.append(dict(surface_id=sid, region="BL", n_events=p["n_events"], lon=p["lon"], lat=p["lat"], depth_km=p["depth_km"],
                          top_km=p["top_km"], bottom_km=p["bottom_km"], strike=p["strike"], dip=p["dip"],
                          dip_68=";".join(map(str, p["dip_68"])), length_km=p["length_km"], width_km=p["width_km"],
                          stable=p["stable"], kind=p["kind"], **trace_row(p, v), nearest_cfm=p["nearest_cfm"],
                          nearest_cfm_km=p["nearest_cfm_km"], nearest_cfm_dip=p["nearest_cfm_dip"],
                          slab2_minus_depth_km=p["slab2_minus_depth_km"]))
        lrows.append(dict(surface_id=sid, lit_match=p["lit_match"], lit_confidence=p["lit_confidence"], lit_class="known structure",
                          lit_new_candidate=p["lit_new_candidate"],
                          lit_evidence=f"Map-view strand (azimuth {p['strand_azimuth']} deg) from the events of {parent}; re-check against the literature.",
                          lit_refs=pr.lit_refs, lit_notes_file=pr.lit_notes_file))
        print(f"  {sid}: {p['n_events']} events, strike {p['strike']}, strand azimuth {p['strand_azimuth']}, dip {p['dip']}, "
              f"{p['length_km']} x {p['width_km']} km, jaccard {p['cluster_jaccard_median']}")

    sid_of = {i: f["properties"]["surface_id"] for i, f in enumerate(new_feats)}
    sep_rows = []
    for i in range(len(strands)):
        for j in range(i + 1, len(strands)):
            sp = separation(XY, strands[i], strands[j])
            if sp:
                sep_rows.append(dict(pool=a.pool_name, a=sid_of[i], b=sid_of[j], **{k: round(v, 2) for k, v in sp.items()}))
    for f in new_feats:
        sid = f["properties"]["surface_id"]
        ds = [r["D"] for r in sep_rows if sid in (r["a"], r["b"])]
        f["properties"]["min_D_to_parallel"] = round(min(ds), 2) if ds else None
    for r_ in new_rows:
        r_["min_D_to_parallel"] = next(f["properties"]["min_D_to_parallel"] for f in new_feats if f["properties"]["surface_id"] == r_["surface_id"])
    sf = V0 / f"{STEM}_strand_separation.csv"
    old = pd.read_csv(sf) if sf.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame(sep_rows)]).to_csv(sf, index=False)
    print(pd.DataFrame(sep_rows).to_string(index=False) if sep_rows else "no overlapping parallel strands")
    # write: remove parents, add strands
    rem = summ[summ.surface_id.isin(parents)][["surface_id", "n_events", "lon", "lat", "strike", "dip", "lit_class", "lit_match"]].assign(
        reason=f"split into strands ({', '.join(f['properties']['surface_id'] for f in new_feats)})")
    rf = V0 / f"{STEM}_removed.csv"
    pd.concat([pd.read_csv(rf), rem]).to_csv(rf, index=False)
    ns = pd.DataFrame(new_rows)
    for col in ns.columns:
        if ns[col].map(lambda x: isinstance(x, list)).any():
            ns[col] = ns[col].map(lambda x: ";".join(map(str, x)) if isinstance(x, list) else x)
    pd.concat([summ[~summ.surface_id.isin(parents)], ns], ignore_index=True).to_csv(V0 / f"{STEM}_summary.csv", index=False)
    vx = pd.read_csv(V0 / f"{STEM}_vertices.csv")
    pd.concat([vx[~vx.surface_id.isin(parents)]] + vrows).to_csv(V0 / f"{STEM}_vertices.csv", index=False)
    en = pd.read_csv(V0 / f"{STEM}_ensemble.csv.gz")
    pd.concat([en[~en.surface_id.isin(parents)]] + erows).to_csv(V0 / f"{STEM}_ensemble.csv.gz", index=False, compression="gzip")
    pd.concat([mem[~mem.surface_id.isin(parents)]] + mrows).to_csv(V0 / f"{STEM}_members.csv", index=False)
    tr = pd.read_csv(V0 / f"{STEM}_traces.csv")
    pd.concat([tr[~tr.surface_id.isin(parents)], pd.DataFrame(trows)]).to_csv(V0 / f"{STEM}_traces.csv", index=False)
    pd.concat([lit[~lit.surface_id.isin(parents)], pd.DataFrame(lrows)]).to_csv(V0 / f"{STEM}_literature.csv", index=False)
    g = json.load(open(V0 / f"{STEM}.geojson"))
    g["features"] = [f for f in g["features"] if f["properties"]["surface_id"] not in parents] + new_feats
    (V0 / f"{STEM}.geojson").write_text(json.dumps(g))
    asg = pd.read_csv(V0 / f"{STEM}_assignment.csv")
    m2 = pd.read_csv(V0 / f"{STEM}_members.csv")
    asg["assigned"] = asg.region.map(m2.surface_id.str[:2].value_counts()).fillna(0).astype(int)
    asg["fraction"] = (asg.assigned / asg.events).round(3)
    asg.to_csv(V0 / f"{STEM}_assignment.csv", index=False)
    print(f"{len(g['features'])} surfaces; {asg.assigned.sum():,} of {asg.events.sum():,} events on a fault")


if __name__ == "__main__":
    main()

"""Fault surfaces v0 from catalog v4 seismicity, whole margin.

v0 tests the method and the uncertainty steps on the v4 locations as they are. It is
not for interpretation: absolute depths are biased shallow (see
data/catalog_v4/next_steps), the depth datum is unconfirmed, and v4 has no relative
relocation. v1 reruns this on the 3D-model relocation (Track A); v2 adds our own focal
mechanisms computed on our own locations.

Steps
  1. Events: QC (>= 4 P and >= 4 S), catalog depth >= 0, and spatial_sigma below a
     per-region cut (the --sig-quantile of the region, capped at --sig-cap km).
     spatial_sigma behaves like an absolute error (it is 3-20 km while events scatter
     1-3 km about the fitted surfaces), so a relative cut per region is used.
  2. Clusters: DBSCAN in 3D (depth weighted by --zscale), without sklearn.
  3. Recursive splitting until each piece is planar (sqrt(l3/l2) <= --planarity, where
     l1 >= l2 >= l3 are the variances along the principal axes) and shorter than
     --max-len km. Each split tries (a) two planes (k-planes, k = 2, started from the
     two sides of the best plane), which separates parallel strands and step-overs, and
     (b) a cut across strike at the median, which separates segments; the one with the
     smaller summed normal variance wins.
  4. Surface: in the frame of the best plane (u along strike, v down dip, w normal), in
     a local projection centred on the cluster, fit w = f(u, v), plane or quadratic by
     BIC, on a --grid km (u, v) grid inside the buffered convex hull of the events. A
     vertex is "constrained" when at least 5 events lie within 2.5 km of it in (u, v).
  5. Uncertainty, reported separately and combined:
       sigma_boot    bootstrap of the events (--draws draws)
       sigma_loc     Monte Carlo over location errors: each event moved by N(0, s_i) on
                     x, y and z with s_i = spatial_sigma (taken as 1-sigma per axis).
                     Since spatial_sigma is larger than the observed scatter, this is a
                     worst case.
       sigma_method  spread over surface-model choices (plane, quadratic, with and without
                     trimming events beyond 2.5 RMS)
       sigma_total   the three in quadrature
     Strike and dip intervals (68%, 95%) come from the bootstrap; the same with the
     location Monte Carlo added are given as *_with_loc_mc (worst case). Clustering
     stability: the whole pipeline is rerun over a 3 x 3 grid of DBSCAN settings; each
     surface reports the median Jaccard overlap with its best-matching piece and the
     range of that piece's strike and dip.
  6. Checks: zone width (RMS normal residual) against the median spatial_sigma; nearest
     CRESCENT CFM fault (3D distance); vertical distance to Slab2. No focal-mechanism
     check: published mechanisms (ComCat) belong to other catalogs' locations and
     velocity models. Velocity-model and datum systematics are not computed in v0;
     surface_elev_km is given at every vertex (depth below sea level = depth_km -
     surface_elev_km if the catalog depth is below the local surface).

Output (data/faults/v0/): fault_surfaces_v0.geojson (MultiPolygon Z triangles per surface:
lon, lat, elevation in m, negative down, as in the CRESCENT CFM), _summary.csv,
_vertices.csv, _ensemble.csv.gz, _members.csv, _map.png, README.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as MPath
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, cKDTree

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data"
BOX = [-131.0, -119.5, 39.0, 51.0]
LON0 = -125.25
KM = 111.195
RE = 6371.0
# most specific first; (prefix, name, box)
REGIONS = [
    ("AX", "Axial Seamount", [-130.35, -129.75, 45.75, 46.2]),
    ("MT", "Mendocino triple junction", [-125.6, -123.2, 39.6, 41.4]),
    ("PU", "Puget Lowland", [-123.6, -121.4, 46.6, 48.6]),
    ("EN", "Endeavour - Explorer", [-130.4, -127.2, 47.4, 49.8]),
    ("BL", "Blanco transform", [-130.6, -126.6, 42.6, 45.3]),
    ("GO", "Gorda plate", [-128.8, -123.8, 40.0, 43.2]),
    ("WA", "Offshore Washington", [-127.6, -123.2, 46.2, 49.2]),
    ("OR", "Offshore Oregon", [-126.8, -123.2, 42.8, 46.4]),
]
OTHER = ("ON", "Onshore and other")
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- geometry
def sinu(lon, lat):
    """Sinusoidal projection (km) about LON0: local distances kept, fine for clustering."""
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    return (lon - LON0) * KM * np.cos(np.radians(lat)), (lat - 45.0) * KM


def ecef(lon, lat, dep):
    lo, la = np.radians(lon), np.radians(lat)
    r = RE - np.asarray(dep, float)
    return np.c_[r * np.cos(la) * np.cos(lo), r * np.cos(la) * np.sin(lo), r * np.sin(la)]


def pca(P):
    c = P.mean(0)
    w, V = np.linalg.eigh(np.cov((P - c).T))
    o = np.argsort(w)[::-1]
    return c, np.maximum(w[o], 0), V[:, o]


def up_normal(n):
    n = n / np.linalg.norm(n)
    return -n if n[2] > 0 else n          # z is down: an upward normal has n_z <= 0


def strike_dip(n):
    """Right-hand-rule strike and dip (deg) of the plane with normal n (x east, y north, z down)."""
    n = up_normal(n)
    dip = np.degrees(np.arccos(min(1.0, abs(n[2]))))
    dipdir = np.degrees(np.arctan2(n[0], n[1])) % 360
    return (dipdir - 90) % 360, dip


def interval_stats(strikes, dips):
    """2.5, 16, 50, 84, 97.5 percentiles of strike (folded about the median; modulo 180 when steep) and dip."""
    if np.median(dips) > 75:
        ref = np.median(strikes % 180)
        s = (strikes % 180 - ref + 90) % 180 - 90 + ref
    else:
        ref = np.median(strikes)
        s = (strikes - ref + 180) % 360 - 180 + ref
    q = lambda a: np.percentile(a, [2.5, 16, 50, 84, 97.5])
    return q(s) % 360, q(dips)


# ---------------------------------------------------------------- data
def region_of(lon, lat):
    pre = np.full(len(lon), OTHER[0], object)
    done = np.zeros(len(lon), bool)
    for p, _, (w, e, s, n) in REGIONS:
        m = ~done & (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)
        pre[m] = p; done |= m
    return pre


def load(q, cap):
    c = pd.read_csv(D / "catalog_v4" / "Cascadia_relocated_catalog_v4.csv", skipinitialspace=True)
    c.columns = [x.strip() for x in c.columns]
    c = c.rename(columns={"ID": "event_id", "depth(km)": "depth", "spatial_sigma(m)": "sigma_m"})
    c["sig"] = pd.to_numeric(c["sigma_m"], errors="coerce") / 1000.0
    c = c[(c.Num_p_picks >= 4) & (c.Num_s_picks >= 4) & (c.depth >= 0) & c.sig.notna()].copy()
    c["region"] = region_of(c.longitude.values, c.latitude.values)
    cut = c.groupby("region")["sig"].quantile(q).clip(upper=cap)
    c = c[c.sig <= c.region.map(cut)].copy()
    c["x"], c["y"] = sinu(c.longitude, c.latitude)
    c["z"] = c["depth"]
    print("sigma cut per region (km): " + ", ".join(f"{k} {v:.1f}" for k, v in cut.items()))
    print(f"selected {len(c):,} events (QC, depth >= 0, sigma below the regional cut)")
    return c.reset_index(drop=True), cut


def relief():
    import pygmt
    g = pygmt.datasets.load_earth_relief(resolution="01m", region=[BOX[0] - .2, BOX[1] + .2, BOX[2] - .2, BOX[3] + .2],
                                         registration="gridline")
    f = RegularGridInterpolator((g.lat.values, g.lon.values), g.values / 1000.0, bounds_error=False, fill_value=np.nan)
    return lambda lon, lat: f(np.c_[lat, lon])


def slab():
    s = pd.read_csv(D / "slab2" / "cas_slab2_dep.xyz", header=None, names=["lon", "lat", "z"])
    s["lon"] = np.where(s.lon > 180, s.lon - 360, s.lon)
    p = s.pivot_table(index="lat", columns="lon", values="z")
    f = RegularGridInterpolator((p.index.values, p.columns.values), -p.values, bounds_error=False, fill_value=np.nan)
    return lambda lon, lat: f(np.c_[lat, lon])


def cfm():
    g = json.load(open(D / "faults" / "crescent_cfm" / "crescent_cfm_crustal_3d_5km.geojson"))
    pts, owner = [], []
    for k, f in enumerate(g["features"]):
        for poly in f["geometry"]["coordinates"]:
            for lon, lat, z in poly[0]:
                pts.append((lon, lat, -z / 1000.0)); owner.append(k)
    pts = np.array(pts)
    return cKDTree(ecef(pts[:, 0], pts[:, 1], pts[:, 2])), np.array(owner), [f["properties"] for f in g["features"]]


# ---------------------------------------------------------------- clustering
def dbscan(P, eps, core):
    pairs = cKDTree(P).query_pairs(eps, output_type="ndarray")
    n = len(P)
    deg = np.bincount(pairs.ravel(), minlength=n)
    is_core = deg >= core
    cc = pairs[is_core[pairs[:, 0]] & is_core[pairs[:, 1]]]
    _, lab = connected_components(coo_matrix((np.ones(len(cc)), (cc[:, 0], cc[:, 1])), shape=(n, n)), directed=False)
    lab = np.where(is_core, lab, -1)
    a, b = pairs[:, 0], pairs[:, 1]
    m = (lab[a] < 0) & is_core[b]; lab[a[m]] = lab[b[m]]
    m = (lab[b] < 0) & is_core[a]; lab[b[m]] = lab[a[m]]
    return lab


def two_planes(Q, w):
    lab = w > 0
    for _ in range(20):
        if lab.sum() < 3 or (~lab).sum() < 3:
            return None
        d = []
        for side in (False, True):
            c, _, V = pca(Q[lab == side])
            d.append(np.abs((Q - c) @ V[:, 2]))
        new = d[1] < d[0]
        if (new == lab).all():
            break
        lab = new
    return lab


def cost(Q, lab):
    return sum(pca(Q[lab == s])[1][2] * (lab == s).sum() for s in (False, True) if (lab == s).sum() >= 3)


def split(P, idx, a, out, level=0):
    if len(idx) < a.n_min:
        return
    c, var, V = pca(P[idx])
    ratio = np.sqrt(var[2] / max(var[1], 1e-9))
    L = 2 * 1.96 * np.sqrt(var[0])
    if (ratio <= a.planarity and L <= a.max_len) or len(idx) < 2 * a.n_min or level > 14:
        if ratio <= a.planarity:
            out.append(idx)
        return
    Q = P[idx] - c
    cands = []
    if L <= a.max_len or ratio > a.planarity:
        lab = two_planes(Q, Q @ V[:, 2])
        if lab is not None and min(lab.sum(), (~lab).sum()) >= a.n_min // 2:
            cands.append(lab)
    u = Q @ V[:, 0]
    cands.append(u > np.median(u))
    best = min(cands, key=lambda l: cost(Q, l))
    if ratio <= a.planarity:            # planar but too long: cut across strike
        best = cands[-1]
    split(P, idx[~best], a, out, level + 1)
    split(P, idx[best], a, out, level + 1)


def pieces(c, a, eps=None, core=None):
    eps, core = eps or a.eps, core or a.core
    lab = dbscan(np.c_[c.x, c.y, c.z * a.zscale], eps, core)
    P = np.c_[c.x, c.y, c.z]
    out = []
    for k in np.unique(lab[lab >= 0]):
        idx = np.flatnonzero(lab == k)
        if len(idx) >= a.n_min:
            split(P, idx, a, out)
    return out


# ---------------------------------------------------------------- surface fit
def design(u, v, quad):
    return np.c_[np.ones_like(u), u, v, u * u, u * v, v * v] if quad else np.c_[np.ones_like(u), u, v]


def fit_w(u, v, w, quad, trim=False):
    A = design(u, v, quad)
    coef = np.linalg.lstsq(A, w, rcond=None)[0]
    if trim:
        r = w - A @ coef
        keep = np.abs(r) <= 2.5 * np.sqrt(np.mean(r ** 2))
        coef = np.linalg.lstsq(A[keep], w[keep], rcond=None)[0]
    return coef


def bic(u, v, w, quad):
    A = design(u, v, quad)
    rss = np.sum((w - A @ np.linalg.lstsq(A, w, rcond=None)[0]) ** 2)
    return len(w) * np.log(rss / len(w)) + A.shape[1] * np.log(len(w))


def surface(c, idx, a, cfmk, slabf, reliefk):
    lon, lat, dep, sig = (c[k].values[idx] for k in ("longitude", "latitude", "z", "sig"))
    lon0, lat0 = np.median(lon), np.median(lat)
    kx = KM * np.cos(np.radians(lat0))
    P = np.c_[(lon - lon0) * kx, (lat - lat0) * KM, dep]          # local projection centred on the cluster
    c0, var, V = pca(P)
    planarity = float(np.sqrt(var[2] / var[1]))
    if planarity > a.planarity or np.sqrt(var[1]) < 1.5:
        return None
    n = up_normal(V[:, 2])
    strike, dip = strike_dip(n)
    s = np.radians(strike)
    eu = np.array([np.sin(s), np.cos(s), 0.0])
    ev = np.cross(n, eu)
    if ev[2] < 0:
        ev = -ev
    Q = P - c0
    u, v, w = Q @ eu, Q @ ev, Q @ n
    quad = bic(u, v, w, True) < bic(u, v, w, False) - 10
    hp = np.c_[u, v][ConvexHull(np.c_[u, v]).vertices]
    cen = hp.mean(0)
    hp = cen + (hp - cen) * (1 + 1.0 / max(np.linalg.norm(hp - cen, axis=1).mean(), 1.0))
    gu = np.arange(np.floor(u.min()) - 1, np.ceil(u.max()) + 2, a.grid)
    gv = np.arange(np.floor(v.min()) - 1, np.ceil(v.max()) + 2, a.grid)
    GU, GV = np.meshgrid(gu, gv)
    inside = MPath(hp).contains_points(np.c_[GU.ravel(), GV.ravel()]).reshape(GU.shape)
    if inside.sum() < 4:
        return None
    vu, vv = GU[inside], GV[inside]
    coef = fit_w(u, v, w, quad)
    vw = design(vu, vv, quad) @ coef
    constrained = np.array([len(x) for x in cKDTree(np.c_[u, v]).query_ball_point(np.c_[vu, vv], 2.5)]) >= 5
    rms = float(np.sqrt(np.mean((w - design(u, v, quad) @ coef) ** 2)))

    def draw(boot, perturb):
        W, S, Dp = [], [], []
        for _ in range(a.draws):
            j = RNG.integers(0, len(idx), len(idx)) if boot else np.arange(len(idx))
            Pj = P[j] + (RNG.normal(size=(len(j), 3)) * sig[j, None] if perturb else 0.0)
            Qj = Pj - c0
            W.append(design(vu, vv, quad) @ fit_w(Qj @ eu, Qj @ ev, Qj @ n, quad) - vw)
            sj, dj = strike_dip(pca(Pj)[2][:, 2])
            S.append(sj); Dp.append(dj)
        return np.array(W), np.array(S), np.array(Dp)
    Wb, Sb, Db = draw(True, False)
    Wl, _, _ = draw(False, True)
    Wt, St, Dt = draw(True, True)
    variants = [design(vu, vv, q) @ fit_w(u, v, w, q, t) for q in (False, True) for t in (False, True)]
    sb, sl, sm = Wb.std(0), Wl.std(0), np.std(variants, axis=0)
    st = np.sqrt(sb ** 2 + sl ** 2 + sm ** 2)
    s_int, d_int = interval_stats(Sb, Db)
    s_mc, d_mc = interval_stats(St, Dt)
    X = c0 + np.outer(vu, eu) + np.outer(vv, ev) + np.outer(vw, n)
    vlon, vlat = X[:, 0] / kx + lon0, X[:, 1] / KM + lat0
    vid = -np.ones(GU.shape, int); vid[inside] = np.arange(inside.sum())
    tris = []
    for i in range(GU.shape[0] - 1):
        for k in range(GU.shape[1] - 1):
            q4 = vid[i, k], vid[i, k + 1], vid[i + 1, k + 1], vid[i + 1, k]
            if min(q4) >= 0:
                tris += [(q4[0], q4[1], q4[2]), (q4[0], q4[2], q4[3])]
    if not tris:
        return None
    clon, clat = c0[0] / kx + lon0, c0[1] / KM + lat0
    dist, j = cfmk[0].query(ecef([clon], [clat], [c0[2]])[0])
    fp = cfmk[2][cfmk[1][j]]
    sd = float(slabf([clon], [clat])[0] - c0[2])
    r2 = lambda x: round(float(x), 1)
    return dict(
        idx=idx, lon=vlon, lat=vlat, dep=X[:, 2], tris=np.array(tris), constrained=constrained,
        sb=sb, sl=sl, sm=sm, st=st, Wt=Wt, normal=n, surface_elev_km=reliefk(vlon, vlat),
        props=dict(
            region=c.region.values[idx[0]] if len(set(c.region.values[idx])) == 1 else pd.Series(c.region.values[idx]).mode()[0],
            n_events=int(len(idx)), lon=round(float(clon), 4), lat=round(float(clat), 4), depth_km=round(float(c0[2]), 2),
            strike=r2(strike), dip=r2(dip),
            strike_68=[r2(s_int[1]), r2(s_int[3])], strike_95=[r2(s_int[0]), r2(s_int[4])],
            dip_68=[r2(d_int[1]), r2(d_int[3])], dip_95=[r2(d_int[0]), r2(d_int[4])],
            strike_68_with_loc_mc=[r2(s_mc[1]), r2(s_mc[3])], dip_68_with_loc_mc=[r2(d_mc[1]), r2(d_mc[3])],
            model="quadratic" if quad else "plane", planarity=round(planarity, 3),
            length_km=r2(2 * 1.96 * np.sqrt(var[0])), width_km=r2(2 * 1.96 * np.sqrt(var[1])),
            rms_normal_km=round(rms, 2), median_sigma_km=round(float(np.median(sig)), 2),
            width_to_error=round(rms / float(np.median(sig)), 2),
            sigma_total_median_km=round(float(np.median(st)), 2), sigma_boot_median_km=round(float(np.median(sb)), 2),
            sigma_loc_median_km=round(float(np.median(sl)), 2), sigma_method_median_km=round(float(np.median(sm)), 2),
            constrained_fraction=round(float(constrained.mean()), 2), n_vertices=int(len(vu)), n_triangles=int(len(tris)),
            nearest_cfm=fp.get("name"), nearest_cfm_km=r2(dist), nearest_cfm_dip=fp.get("dip"),
            slab2_minus_depth_km=r2(sd) if np.isfinite(sd) else None,
        ))


def stability(c, surfaces, a):
    """Rerun the whole pipeline over a 3 x 3 grid of DBSCAN settings; match each surface to its best piece."""
    lab_sets = []
    for eps in (a.eps - 0.5, a.eps, a.eps + 0.5):
        for core in (a.core - 2, a.core, a.core + 2):
            lab = -np.ones(len(c), int)
            for k, idx in enumerate(pieces(c, a, eps, core)):
                lab[idx] = k
            lab_sets.append(lab)
    P = np.c_[c.x, c.y, c.z]
    for s in surfaces:
        A = s["idx"]; J, S, Dp = [], [], []
        for lab in lab_sets:
            la = lab[A]; la = la[la >= 0]
            if len(la) == 0:
                J.append(0.0); continue
            B = np.flatnonzero(lab == np.bincount(la).argmax())
            J.append(len(np.intersect1d(A, B)) / len(np.union1d(A, B)))
            st, dp = strike_dip(pca(P[B])[2][:, 2]); S.append(st); Dp.append(dp)
        p = s["props"]
        p["cluster_jaccard_median"] = round(float(np.median(J)), 2)
        if S:
            si, di = interval_stats(np.array(S), np.array(Dp))
            p["strike_settings_range"] = [round(float(si[0]), 1), round(float(si[4]), 1)]
            p["dip_settings_range"] = [round(float(di[0]), 1), round(float(di[4]), 1)]
        dz = p["slab2_minus_depth_km"]
        if p["dip"] >= 45:
            p["kind"] = "steep zone (fault candidate)"
        elif dz is not None and abs(dz) <= 5 and p["dip"] <= 25:
            p["kind"] = "near-slab layer (not a single fault)"
        else:
            p["kind"] = "dipping zone"
        p["stable"] = bool(p["cluster_jaccard_median"] >= 0.5)


# ---------------------------------------------------------------- output
def write(c, surfaces, out, cut, a):
    out.mkdir(parents=True, exist_ok=True)
    feats, rows, vrows, erows, mrows = [], [], [], [], []
    count = {}
    for s in surfaces:
        r = s["props"]["region"]; count[r] = count.get(r, 0) + 1
        sid = f"{r}{count[r]:03d}"
        s["props"]["surface_id"] = sid
        p = dict(version="v0", catalog="Cascadia catalog v4", **s["props"])
        p["normal_xyz"] = [round(float(x), 4) for x in s["normal"]]
        tri = [[[[round(float(s["lon"][i]), 5), round(float(s["lat"][i]), 5), round(float(-s["dep"][i] * 1000), 1)]
                 for i in (t0, t1, t2, t0)]] for t0, t1, t2 in s["tris"]]
        feats.append(dict(type="Feature", properties=p, geometry=dict(type="MultiPolygon", coordinates=tri)))
        rows.append(p)
        el = s["surface_elev_km"]
        for i in range(len(s["lon"])):
            vrows.append((sid, i, round(float(s["lon"][i]), 5), round(float(s["lat"][i]), 5), round(float(s["dep"][i]), 3),
                          round(float(s["sb"][i]), 3), round(float(s["sl"][i]), 3), round(float(s["sm"][i]), 3),
                          round(float(s["st"][i]), 3), int(s["constrained"][i]),
                          round(float(el[i]), 3) if np.isfinite(el[i]) else np.nan))
        W = np.round(s["Wt"], 3)
        erows.append(pd.DataFrame({"surface_id": sid, "vertex": np.tile(np.arange(W.shape[1]), W.shape[0]),
                                   "draw": np.repeat(np.arange(W.shape[0]), W.shape[1]), "normal_offset_km": W.ravel()}))
        mrows += [(int(e), sid) for e in c.event_id.values[s["idx"]]]
    gj = dict(type="FeatureCollection", name="fault_surfaces_v0",
              crs=dict(type="name", properties=dict(name="urn:ogc:def:crs:OGC:1.3:CRS84")), features=feats)
    (out / "fault_surfaces_v0.geojson").write_text(json.dumps(gj))
    summ = pd.DataFrame(rows)
    for col in summ.columns:
        if summ[col].map(lambda v: isinstance(v, list)).any():
            summ[col] = summ[col].map(lambda v: ";".join(map(str, v)) if isinstance(v, list) else v)
    summ.to_csv(out / "fault_surfaces_v0_summary.csv", index=False)
    pd.DataFrame(vrows, columns=["surface_id", "vertex", "lon", "lat", "depth_km", "sigma_boot_km", "sigma_loc_km",
                                 "sigma_method_km", "sigma_total_km", "constrained", "surface_elev_km"]
                 ).to_csv(out / "fault_surfaces_v0_vertices.csv", index=False)
    pd.concat(erows).to_csv(out / "fault_surfaces_v0_ensemble.csv.gz", index=False, compression="gzip")
    pd.DataFrame(mrows, columns=["event_id", "surface_id"]).to_csv(out / "fault_surfaces_v0_members.csv", index=False)
    pd.Series(cut, name="sigma_cut_km").to_csv(out / "fault_surfaces_v0_sigma_cuts.csv")
    return summ


def figure(c, surfaces, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    views = [("Whole margin", [-131, -119.5, 39, 51]), ("Blanco transform", [-130.6, -126.6, 42.6, 45.3]),
             ("Mendocino and Gorda", [-128.0, -123.2, 39.8, 42.5])]
    fig, axs = plt.subplots(1, 3, figsize=(16, 6.5), constrained_layout=True)
    for ax, (title, b) in zip(axs, views):
        sel = c.longitude.between(b[0], b[1]) & c.latitude.between(b[2], b[3])
        ax.scatter(c.longitude[sel], c.latitude[sel], s=0.3, c="0.78", lw=0, rasterized=True)
        polys, vals = [], []
        for s in surfaces:
            for t in s["tris"]:
                polys.append(np.c_[s["lon"][t], s["lat"][t]]); vals.append(s["dep"][t].mean())
        pc = PolyCollection(polys, array=np.array(vals), cmap="viridis_r", clim=(0, 30), edgecolors="none")
        ax.add_collection(pc)
        if title != "Whole margin":
            for s in surfaces:
                p = s["props"]
                if b[0] <= p["lon"] <= b[1] and b[2] <= p["lat"] <= b[3]:
                    ax.text(p["lon"], p["lat"], p["surface_id"], fontsize=5.5, ha="center", va="center")
        ax.set_xlim(b[0], b[1]); ax.set_ylim(b[2], b[3]); ax.set_aspect(1 / np.cos(np.radians((b[2] + b[3]) / 2)))
        ax.set_title(title, fontsize=10)
    fig.colorbar(pc, ax=axs, label="Surface depth (km, catalog datum)", shrink=0.7)
    fig.suptitle("Fault surfaces v0 from catalog v4 (method test, not for interpretation)", fontsize=11)
    fig.savefig(out / "fault_surfaces_v0_map.png", dpi=200)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sig-quantile", type=float, default=0.6, help="keep events below this quantile of spatial_sigma per region")
    ap.add_argument("--sig-cap", type=float, default=20.0, help="upper limit of the per-region cut (km)")
    ap.add_argument("--eps", type=float, default=2.5, help="DBSCAN link distance (km)")
    ap.add_argument("--core", type=int, default=8)
    ap.add_argument("--zscale", type=float, default=0.5, help="depth weight in the clustering")
    ap.add_argument("--n-min", type=int, default=50)
    ap.add_argument("--planarity", type=float, default=0.5)
    ap.add_argument("--max-len", type=float, default=40.0, help="longest planar piece before a cut across strike (km)")
    ap.add_argument("--grid", type=float, default=1.0, help="(u, v) grid spacing (km)")
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--out", default=str(D / "faults" / "v0"))
    a = ap.parse_args(argv)
    c, cut = load(a.sig_quantile, a.sig_cap)
    rel, cf, sl = relief(), cfm(), slab()
    pc = pieces(c, a)
    print(f"{len(pc)} planar pieces with >= {a.n_min} events")
    surfaces = [s for s in (surface(c, idx, a, cf, sl, rel) for idx in pc) if s is not None]
    surfaces.sort(key=lambda s: (s["props"]["region"], -s["props"]["n_events"]))
    stability(c, surfaces, a)
    out = Path(a.out)
    summ = write(c, surfaces, out, cut, a)
    figure(c, surfaces, out)
    print(f"{len(summ)} surfaces")
    print(summ.groupby("region").agg(n=("surface_id", "size"), steep=("kind", lambda k: (k.str.startswith("steep")).sum()),
                                     stable=("stable", "sum"), events=("n_events", "sum")).to_string())
    pd.set_option("display.width", 250)
    print(summ[["surface_id", "kind", "stable", "n_events", "lon", "lat", "depth_km", "strike", "dip", "dip_68",
                "length_km", "rms_normal_km", "sigma_total_median_km", "cluster_jaccard_median", "nearest_cfm",
                "nearest_cfm_km"]].to_string(index=False))


if __name__ == "__main__":
    main()

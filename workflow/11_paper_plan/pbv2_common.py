"""Plate boundaries v2 (D-B116): shared inputs, section frames, seismicity-axis picks and the straight-segment fit.

The rules are pre-registered in data/paper_plan/plate_boundaries_v2/RULES.md (written before any fit). This module only
reads; pbv2_fit.py, pbv2_bias.py, pbv2_figures.py and pbv2_report.py write.

Frames: each section has a straight reference line p0 -> p1 in an azimuthal-equidistant projection centred on the
section (pyproj, WGS84); s = distance along p0 -> p1 (km), u = signed distance across it (km), positive to the north
for transforms and to the east for ridges. Every dataset (events, PB2002, bathymetric picks, other models) is projected
with the same function, so offsets between them carry no projection error.

Python: LD_LIBRARY_PATH=/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/lib
/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/bin/python -I (the scripts read downloaded files: -I).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Proj
from scipy.ndimage import convolve1d

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import pb_common as P  # noqa: E402

WD1 = Path("/wd1/mdenolle_data")
CAT = WD1 / "catalog_v6/v6.1/cascadia_catalog_v6.1.csv"
CMP = WD1 / "catalog_v6/v6.1_analysis/validation/catalog_compare/big"     # v4.1 v5.1 v6.0 v6.1 rows by event_id
FAULTS = WD1 / "catalog_v6/v6.1_analysis/faults"
EXT = WD1 / "external_geo"
ANSS = Path("/home/mdenolle/cascadia_obs_ensemble/data/magnitude_v4/anss_2010-16.csv")
LANGE = WD1 / "tomo_v1/depth_literature/pairs"
TOPO = Path("/wd1/mdenolle_data/reloc_v5/topo/topo_15s_cons_merged.bin")
OUT = ROOT / "data/paper_plan/plate_boundaries_v2"                       # small tables (git)
BIG = Path(os.environ.get("PBV2_BIG", WD1 / "paper_plan/plate_boundaries_v2"))
SEED = 20261127
NBOOT = 300
BIN = 5.0          # km, along-strike bins
BW = 1.5           # km, kernel bandwidth across strike
DU = 0.1           # km, density grid
MIN_EV = 10        # events per bin for a pick
LMIN = 30.0        # km, minimum segment length
MIN_PK_SEG = 4
KMAX = 5
MIN_PICKS = 8      # fewer picks (or < 50 % coverage): bathymetry-only section
H_MAX, ELL3 = 3.0, 10.0
CI_YEARS = {"pre-CI": ("2010-01-01", "2011-07-01"), "CI Y1": ("2011-07-01", "2012-07-01"),
            "CI Y2": ("2012-07-01", "2013-07-01"), "CI Y3": ("2013-07-01", "2014-07-01"),
            "CI Y4": ("2014-07-01", "2015-10-31"), "post-CI": ("2015-10-31", "2017-01-01")}


@dataclass
class Section:
    code: str
    name: str
    kind: str                 # 'transform' or 'ridge'
    p0: tuple
    p1: tuple
    uwin: tuple
    pb_class: str             # PB2002 step class compared with
    bathy: str                # bathymetric reference: valley | base | crest | mapridges
    k_forced: int | None = None

    def __post_init__(self):
        lon_c, lat_c = (self.p0[0] + self.p1[0]) / 2, (self.p0[1] + self.p1[1]) / 2
        self.proj = Proj(proj="aeqd", lat_0=lat_c, lon_0=lon_c, ellps="WGS84")
        x0, y0 = self.proj(*self.p0)
        x1, y1 = self.proj(*self.p1)
        t = np.array([x1 - x0, y1 - y0]) / 1000.0
        self.length = float(np.hypot(*t))
        self.t = t / self.length
        n = np.array([-self.t[1], self.t[0]])                      # left normal
        if (self.kind == "transform" and n[1] < 0) or (self.kind == "ridge" and n[0] < 0):
            n = -n
        self.n = n
        self.o = np.array([x0, y0]) / 1000.0

    def su(self, lon, lat):
        x, y = self.proj(np.asarray(lon, float), np.asarray(lat, float))
        d = np.stack([np.asarray(x) / 1000.0 - self.o[0], np.asarray(y) / 1000.0 - self.o[1]], -1)
        return d @ self.t, d @ self.n

    def lonlat(self, s, u):
        s, u = np.asarray(s, float), np.asarray(u, float)
        x = (self.o[0] + s * self.t[0] + u * self.n[0]) * 1000.0
        y = (self.o[1] + s * self.t[1] + u * self.n[1]) * 1000.0
        return self.proj(x, y, inverse=True)

    @property
    def edges(self):
        return np.arange(0.0, self.length + BIN - 1e-9, BIN)

    @property
    def centres(self):
        e = self.edges
        return (e[:-1] + e[1:]) / 2


# p0 -> p1: the along-strike direction; u > 0 north (transforms) or east (ridges)
SECTIONS = {
    "MFZ": Section("MFZ", "Mendocino transform", "transform", (-127.45, 40.44), (-125.00, 40.33), (-20, 25), "OTF", "base"),
    "GR": Section("GR", "Gorda Ridge", "ridge", (-127.45, 40.40), (-126.60, 43.00), (-35, 30), "OSR", "valley"),   # A1
    "BL": Section("BL", "Blanco transform", "transform", (-126.60, 43.00), (-130.45, 44.52), (-30, 20), "OTF", "mapridges", 2),
    "JRS": Section("JRS", "Juan de Fuca Ridge, Cleft to Axial", "ridge", (-130.45, 44.50), (-130.00, 45.95), (-25, 25), "OSR", "crest"),
    "JRN": Section("JRN", "Juan de Fuca Ridge, Axial to Endeavour", "ridge", (-130.00, 45.95), (-128.90, 48.40), (-25, 25), "OSR", "crest", 0),
}
SIGN = {"transform": "north", "ridge": "east"}


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ events
def load_v61() -> pd.DataFrame:
    cols = ["ID", "time", "lat", "lon", "depth", "loc_source", "tier", "region", "h_max", "ell_len3", "gap",
            "dist_min", "event_flag", "nphs", "M"]
    c = pd.read_csv(CAT, usecols=cols, low_memory=False)
    c = c[c.loc_source.eq("nll3d") & c.event_flag.isna()].copy()
    c["t"] = pd.to_datetime(c.time, utc=True, format="ISO8601")
    c["day"] = c.t.dt.floor("D")
    c["S0"] = c.tier.isin(["A", "B"]) & (c.h_max <= H_MAX) & (c.ell_len3 <= ELL3)
    c["S1"] = c.tier.isin(["A", "B"]) & (c.h_max <= 1.5) & (c.ell_len3 <= 6.0)
    c["S2"] = c.tier.isin(["A", "B", "C"]) & (c.h_max <= H_MAX) & (c.ell_len3 <= ELL3)
    c["ci_year"] = ""
    for k, (a, b) in CI_YEARS.items():
        c.loc[(c.t >= pd.Timestamp(a, tz="UTC")) & (c.t < pd.Timestamp(b, tz="UTC")), "ci_year"] = k
    return c.reset_index(drop=True)


def version_positions(ids) -> dict:
    """lon, lat of the same events in catalogs v4.1, v5.1, v6.0, v6.1 (compare_catalogs.py tables, by event_id)."""
    out = {}
    for v in ("v4.1", "v5.1", "v6.0", "v6.1"):
        t = pd.read_parquet(CMP / f"{v}.parquet", columns=["event_id", "lat", "lon", "located"])
        t = t.set_index("event_id").reindex(ids)
        out[v] = t
    return out


# ------------------------------------------------------------------ reference traces
def pb2002_points():
    pts, sub, tj = P.pb2002(axes="pb2002")
    return pts, sub, tj


def mapridges():
    import geopandas as gpd
    g = []
    for f, typ in (("RS", "ridge"), ("TF", "transform"), ("TZ", "transform zone")):
        d = gpd.read_file(EXT / f"pbv2_mapridges/MAPRIDGESv2602_{f}.gpkg").cx[-132:-123, 39:52]
        nm = d["Name_Lit"] if "Name_Lit" in d else d["Name"]
        for name, geom in zip(nm, d.geometry):
            for ln in getattr(geom, "geoms", [geom]):
                xy = np.asarray(ln.coords)[:, :2]
                g.append(dict(name=name if isinstance(name, str) else "", type=typ, xy=xy))
    return g


def hasterok():
    import geopandas as gpd
    d = gpd.read_file(EXT / "pbv2_hasterok2022/plate_boundaries.shp").cx[-132:-123, 39:52]
    g = []
    for _, r in d.iterrows():
        for ln in getattr(r.geometry, "geoms", [r.geometry]):
            g.append(dict(name=r["feature"] or "", type=r["type"], plates=f"{r['plate1']}-{r['plate2']}",
                          xy=np.asarray(ln.coords)[:, :2]))
    return g


def morvel56_jf():
    """NNR-MORVEL56 Juan de Fuca plate outline (LAT LON order in the file)."""
    import zipfile
    with zipfile.ZipFile(EXT / "pbv2_nnrmorvel56/NnrMRVL_PltBndsLatLon.zip") as z:
        name = [n for n in z.namelist() if n.split("/")[-1].upper() == "JF"][:1] or \
               [n for n in z.namelist() if n.split("/")[-1].upper().startswith("JF")]
        txt = z.read(name[0]).decode("ascii", "replace")
    rows = []
    for ln in txt.splitlines():
        p = ln.replace(",", " ").split()
        try:
            la, lo = float(p[0]), float(p[1])
        except (ValueError, IndexError):
            continue
        rows.append((lo - 360 if lo > 180 else lo, la))
    return np.asarray(rows)


def densify(xy, step_km=0.5):
    return P._densify(np.asarray(xy, float), step_km)


# ------------------------------------------------------------------ bathymetry picks (D-B113 code reused)
def bathy_picks():
    """D-B113 bathymetric candidates from pb_axes.py on the PB2002 sections: dict name -> (N, 2) lon/lat."""
    import pb_axes as A
    Z = A.topo()
    pts, _, _ = pb2002_points()
    out = {"valley": [], "crest": []}
    for sec in A.sections(pts["OSR"]):
        c = A.pick_ridge(sec, Z)
        out["valley"].append(c["valley"])
        out["crest"].append(c["crest"])
    otf = pts["OTF"]
    mfz = otf[(otf[:, 1] < 41.0) & (otf[:, 0] < -124.9)]
    m = A.pick_mfz(mfz, Z)
    out = {k: np.vstack(v) for k, v in out.items()}
    out["base"], out["mfz_crest"] = m["base"], m["crest"]
    return out


def topo_grid():
    nx, ny, x0, y0, dx = 3841, 3601, -134.0, 37.5, 1 / 240
    z = np.fromfile(TOPO, dtype="<f8").reshape(ny, nx)[::-1]
    return x0 + dx * np.arange(nx), y0 + dx * np.arange(ny), z


# ------------------------------------------------------------------ picks
def _kernel():
    h = np.arange(-4 * BW, 4 * BW + DU / 2, DU)
    k = np.exp(-0.5 * (h / BW) ** 2)
    return k / k.sum()


KER = _kernel()


def bin_picks(sec: Section, s, u, w=None, min_ev=MIN_EV):
    """Peak of the across-strike density in each 5 km bin; NaN where < min_ev (weighted) events or the peak lies
    within 1 km of the window edge. Returns picks (nbins,), counts (nbins,)."""
    s, u = np.asarray(s, float), np.asarray(u, float)
    w = np.ones(len(s)) if w is None else np.asarray(w, float)
    e = sec.edges
    grid = np.arange(sec.uwin[0], sec.uwin[1] + DU / 2, DU)
    ib = np.searchsorted(e, s, side="right") - 1
    iu = np.round((u - sec.uwin[0]) / DU).astype(int)
    ok = (ib >= 0) & (ib < len(e) - 1) & (iu >= 0) & (iu < len(grid)) & (w > 0)
    H = np.zeros((len(e) - 1, len(grid)))
    np.add.at(H, (ib[ok], iu[ok]), w[ok])
    n = H.sum(1)
    D = convolve1d(H, KER, axis=1, mode="constant")
    pk = grid[D.argmax(1)]
    bad = (n < min_ev) | (pk < sec.uwin[0] + 1.0) | (pk > sec.uwin[1] - 1.0)
    pk = np.where(bad, np.nan, pk)
    return pk, n


def day_weights(days: pd.Series, rng, nboot):
    """Day-block bootstrap: per sample, the multiplicity of each event's UTC day."""
    codes, uniq = pd.factorize(days)
    nd = len(uniq)
    for _ in range(nboot):
        cnt = np.bincount(rng.integers(0, nd, nd), minlength=nd)
        yield cnt[codes].astype(float)


# ------------------------------------------------------------------ straight-segment fit (dynamic programming)
def _cost_matrix(s, u, w):
    n = len(s)
    sc = s - s.mean()
    cs = lambda v: np.r_[0.0, np.cumsum(v)]                      # noqa: E731
    W, S, SS, U, SU, UU = cs(w), cs(w * sc), cs(w * sc * sc), cs(w * u), cs(w * sc * u), cs(w * u * u)
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    g = lambda A: A[j + 1] - A[i]                                 # noqa: E731
    Sw, Ss, Sss, Su, Ssu, Suu = g(W), g(S), g(SS), g(U), g(SU), g(UU)
    with np.errstate(invalid="ignore", divide="ignore"):
        det = Sw * Sss - Ss ** 2
        b = (Sw * Ssu - Ss * Su) / det
        a = (Su - b * Ss) / Sw
        chi = Suu - a * Su - b * Ssu
    span = s[None, :] - s[:, None] + BIN
    valid = (j - i + 1 >= MIN_PK_SEG) & (span >= LMIN) & (det > 0)
    chi = np.where(valid, np.maximum(chi, 0.0), np.inf)
    return chi, a, b, s.mean()


def fit_segments(s, u, sig, K):
    """Best K-break fit: list of segments (i0, i1, a, b) with u = a + b (s - smean), chi2. None if impossible."""
    s, u, sig = map(np.asarray, (s, u, sig))
    n = len(s)
    if n == 0:
        return None
    C, A, B, sm = _cost_matrix(s, u, 1.0 / sig ** 2)
    F = np.full((K + 1, n), np.inf)
    bp = np.zeros((K + 1, n), int)
    F[0] = C[0]
    for k in range(1, K + 1):
        for jj in range(n):
            cand = F[k - 1][:jj] + C[1:jj + 1, jj]
            if len(cand) == 0:
                continue
            m = int(np.argmin(cand))
            F[k, jj], bp[k, jj] = cand[m], m + 1
    if not np.isfinite(F[K, n - 1]):
        return None
    segs, jj = [], n - 1
    for k in range(K, -1, -1):
        i0 = bp[k, jj] if k > 0 else 0
        segs.append((i0, jj, A[i0, jj], B[i0, jj]))
        jj = i0 - 1
    segs = segs[::-1]
    return dict(segs=segs, chi2=float(F[K, n - 1]), smean=float(sm), s=s, u=u, sig=sig, K=K)


def bic_select(s, u, sig, length):
    n = len(s)
    kmax = int(min(KMAX, np.floor(length / LMIN) - 1))
    rows, best = [], None
    for K in range(0, max(kmax, 0) + 1):
        f = fit_segments(s, u, sig, K)
        if f is None:
            continue
        bic = f["chi2"] + (3 * K + 2) * np.log(n) + K * np.log(n)
        rows.append(dict(K=K, chi2=f["chi2"], bic=bic, n=n))
        if best is None or bic < best[1]:
            best = (f, bic)
    return (best[0] if best else None), pd.DataFrame(rows)


def seg_geometry(f):
    """Segments with s extents (break = midpoint between the last pick of one and the first of the next) and lines."""
    s, segs = f["s"], f["segs"]
    out = []
    for k, (i0, i1, a, b) in enumerate(segs):
        s0 = s[i0] - BIN / 2 if k == 0 else (s[segs[k - 1][1]] + s[i0]) / 2
        s1 = s[i1] + BIN / 2 if k == len(segs) - 1 else (s[i1] + s[segs[k + 1][0]]) / 2
        out.append(dict(k=k, s0=float(s0), s1=float(s1), a=float(a), b=float(b), sm=f["smean"], i0=int(i0),
                        i1=int(i1), npk=int(i1 - i0 + 1)))
    return out


def line_u(seg, s):
    return seg["a"] + seg["b"] * (np.asarray(s, float) - seg["sm"])


def eval_fit(f, sgrid):
    g = seg_geometry(f)
    u = np.full(len(sgrid), np.nan)
    for k, sg in enumerate(g):
        lo = -np.inf if k == 0 else sg["s0"]
        hi = np.inf if k == len(g) - 1 else sg["s1"]
        m = (sgrid >= lo) & (sgrid < hi)
        u[m] = line_u(sg, sgrid[m])
    return u, g


def step_offsets(g):
    return [line_u(g[k + 1], g[k]["s1"]) - line_u(g[k], g[k]["s1"]) for k in range(len(g) - 1)]


def wls_line(s, u, sig=None):
    """Straight line u = a + b (s - sm) by least squares; returns a, b, sm, cov, rms."""
    s, u = np.asarray(s, float), np.asarray(u, float)
    sm = s.mean()
    X = np.c_[np.ones(len(s)), s - sm]
    w = np.ones(len(s)) if sig is None else 1.0 / np.asarray(sig) ** 2
    XtW = X.T * w
    cov0 = np.linalg.inv(XtW @ X)
    beta = cov0 @ XtW @ u
    r = u - X @ beta
    dof = max(len(s) - 2, 1)
    rms = float(np.sqrt(np.sum(r ** 2) / dof))
    cov = cov0 * (rms ** 2 if sig is None else max(np.sum(w * r ** 2) / dof, 1.0))
    return float(beta[0]), float(beta[1]), float(sm), cov, rms


def line_se(cov, sm, s):
    s = np.asarray(s, float) - sm
    return np.sqrt(cov[0, 0] + 2 * s * cov[0, 1] + s * s * cov[1, 1])


def intersect(p, d, q, e):
    """Intersection of lines p + t d and q + v e (2-D); returns point and t, v."""
    M = np.array([[d[0], -e[0]], [d[1], -e[1]]])
    if abs(np.linalg.det(M)) < 1e-12:
        return None
    t, v = np.linalg.solve(M, np.asarray(q) - np.asarray(p))
    return np.asarray(p) + t * np.asarray(d), t, v


def km_proj(lon_c, lat_c):
    return Proj(proj="aeqd", lat_0=lat_c, lon_0=lon_c, ellps="WGS84")


def kde_peak_1d(x, w=None, lo=-40, hi=40):
    grid = np.arange(lo, hi + DU / 2, DU)
    iu = np.round((np.asarray(x) - lo) / DU).astype(int)
    w = np.ones(len(iu)) if w is None else np.asarray(w)
    ok = (iu >= 0) & (iu < len(grid))
    H = np.zeros(len(grid))
    np.add.at(H, iu[ok], w[ok])
    return float(grid[np.argmax(np.convolve(H, KER, mode="same"))]) if H.sum() > 0 else np.nan

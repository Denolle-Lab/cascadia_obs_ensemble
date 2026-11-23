"""Nearest-neighbor clustering of catalog v4 (Zaliapin & Ben-Zion method).

Classifies every catalog v4 event as background (single), member of a
mainshock-aftershock family, member of a swarm, or member of a doublet /
multiplet. Completeness, b, and the threshold are set region by region.

Method
------
For each event j, its nearest neighbor (parent) is the earlier event i that
minimises the space-time-magnitude proximity (Baiesi & Paczuski 2004;
Zaliapin et al. 2008):

    eta_ij = t_ij * r_ij**df * 10**(-b_i * m_i)        (t_ij > 0)

with t in years, r in km, m_i the magnitude of the parent and b_i the b-value
of the parent's region. The rescaled time and distance are
T = t * 10**(-q b m_i) and R = r**df * 10**(-(1-q) b m_i), q = 0.5, so that
log eta = log T + log R.

Choices (all documented in the report):
* Regions: the nine boxes of the project (AX, MT, PU, EN, BL, GO, WA, OR, ON),
  each event in the first box that holds it. Mc, b and eta0 are regional.
  The parent search runs over the whole catalog, but a parent in another
  region is allowed only within the link distance cap below. Sequences that
  cross a box edge (e.g. 2014 M6.5 Vancouver Island, whose aftershocks fall
  in the EN box) then stay in one family, while distant large events in a
  neighbouring region do not become the nearest neighbor of small
  background events. A family takes the region of its largest event.
* Distance: epicentral (2-D, great circle). v4 absolute depths are biased
  shallow, so 3-D distance is only a variant (d3). Distances are floored at
  --rmin km (default 0.5 km) because v4 epicentres are not better than that.
* Link distance cap: a link with eta < eta0 is kept only when
  r <= max(--rcap-min, 3 L(m_i)) with L the subsurface rupture length of
  Wells & Coppersmith (1994), log10 L = -2.44 + 0.59 M (10 km floor). Without
  it, M6.5 parents link events 300+ km away within days. The cap does not
  change the nearest-neighbor search or the eta distribution.
* Magnitude: the Route-A magnitude table (ML calibrated to ComCat, Mw from
  moment tensors above 4.5). The catalog magnitude column is not used.
* Completeness: Mc per region and year by maximum curvature + 0.2
  (years with >= 200 events); the region Mc is the largest yearly value, so
  one Mc holds for the whole 2010-2016 window. A b-value-stability Mc
  (Woessner & Wiemer 2005 style) is reported as a check. The clustering runs
  on events with M >= Mc of their region ("analysis set").
* b-value: Aki (1965) maximum likelihood for continuous magnitudes above Mc,
  per region.
* df: 1.6 in all regions (main run), the value used by Zaliapin & Ben-Zion
  (2013a, 2016). The correlation dimension of the epicentres above Mc
  (Grassberger-Procaccia, 3-30 km) is reported per region; it is lower
  (0.8-1.4) because the offshore seismicity lies on narrow transform and
  ridge strands, and because the full catalog is itself clustered. Variants:
  df_est (regional estimate clipped to [1.2, 2.0]), df1.3, df2.0.
* Threshold eta0: two-component 1-D Gaussian mixture on log10 eta of the
  analysis-set events of each region; eta0 is where the two weighted
  components are equal. The KDE valley is reported as a check. Variant
  eta0_shuf5 instead sets eta0 per region so that only 5 % of the
  nearest-neighbor links in the time-shuffled catalogs fall below it.
* Families: links with eta < eta0 (and within the distance cap) joined into
  trees (connected components). Events with no such link either way are
  background singles.
* Events below Mc or without M are attached afterwards: each one gets its
  nearest-neighbor parent among earlier analysis-set events (eta depends only
  on the parent's magnitude, so the child needs no magnitude). If the link
  passes the threshold and the cap it joins the parent's family (role suffix
  "_belowMc"); otherwise it is background. These events do not change the
  family classification.

Family classification (n = analysis-set events in the family):
* n = 2-4: "multiplet" (doublet, triplet...) when the two largest magnitudes
  differ by less than 0.3; otherwise "mainshock-aftershock".
* n >= 5: three swarm indicators,
    S1  no dominant event: M1 - M2 < 0.5
    S2  late peak: >= 20 % of the family's events occur before the largest
    S3  chain-like tree: mean leaf depth >= 2 and fewer than half of the
        family are direct offspring of the largest event
  "swarm" when at least two hold, else "mainshock-aftershock".
  Roles: largest event = mainshock, earlier = foreshock, later = aftershock;
  in swarms all events are "swarm"; in multiplets all are "multiplet".
These follow the topology criteria of Zaliapin & Ben-Zion (2013b, 2016):
burst-like families are shallow, bushy trees around one large event;
swarm-like families are deep trees with no dominant event; and the magnitude
gap / largest-event-position criteria of Vidale & Shearer (2006).

Validation: time-shuffled catalogs (event times permuted among the events of
each region; locations and magnitudes kept) to show the clustered mode
disappears; known sequences; comparison with v4_swarms.py clusters
(done in v4_clusters_zbz_report.py).

Outputs in data/catalog_v4/clusters_zbz/ (see --out). Run:
    python workflow/06_analysis/v4_clusters_zbz.py                 # everything
    python workflow/06_analysis/v4_clusters_zbz.py --faults-only   # redo fault join
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import gaussian_kde, spearmanr, theilslopes

ROOT = Path(__file__).resolve().parents[2]
MAIN = Path("/home/mdenolle/cascadia_obs_ensemble")
CAT = MAIN / "data/catalog_v4/Cascadia_relocated_catalog_v4.csv"
MAG = MAIN / "data/magnitude_v4/cascadia_catalog_M_routeA.csv"
FAULTS = ROOT / "data/faults/v0"
SWARMS = ROOT / "data/catalog_v4/swarms"
OUT = ROOT / "data/catalog_v4/clusters_zbz"

REGIONS = [  # most specific first
    ("AX", "Axial Seamount", [-130.35, -129.75, 45.75, 46.2]),
    ("MT", "Mendocino triple junction", [-125.6, -123.2, 39.6, 41.4]),
    ("PU", "Puget Lowland", [-123.6, -121.4, 46.6, 48.6]),
    ("EN", "Endeavour-Explorer", [-130.4, -127.2, 47.4, 49.8]),
    ("BL", "Blanco transform", [-130.6, -126.6, 42.6, 45.3]),
    ("GO", "Gorda ridge and plate", [-128.8, -123.8, 40.0, 43.2]),
    ("WA", "Offshore Washington", [-127.6, -123.2, 46.2, 49.2]),
    ("OR", "Offshore Oregon", [-126.8, -123.2, 42.8, 46.4]),
]
REGION_NAMES = {k: n for k, n, _ in REGIONS} | {"ON": "Onshore and other"}
REG_ORDER = ["AX", "EN", "BL", "GO", "MT", "WA", "OR", "PU", "ON"]
RE = 6371.0
Q = 0.5
T0 = pd.Timestamp("2010-01-01", tz="UTC")




# ---------------------------------------------------------------- loading
def load() -> pd.DataFrame:
    cat = pd.read_csv(CAT, skipinitialspace=True)
    cat.columns = [c.strip() for c in cat.columns]
    mag = pd.read_csv(MAG, usecols=["event_id", "M", "M_type"])
    df = cat[["ID", "datetime", "latitude", "longitude", "depth(km)"]].rename(
        columns={"ID": "event_id", "latitude": "lat", "longitude": "lon", "depth(km)": "depth"})
    df = df.merge(mag, on="event_id", how="left")
    df["time"] = pd.to_datetime(df["datetime"], utc=True, format="ISO8601")
    df["t_yr"] = (df["time"] - T0).dt.total_seconds() / (365.25 * 86400)
    df["year"] = df["time"].dt.year
    reg = np.full(len(df), "ON", dtype=object)
    for key, _, (a, b, c, d) in REGIONS[::-1]:
        s = df.lon.between(a, b) & df.lat.between(c, d)
        reg[s.values] = key
    df["region"] = reg
    return df.sort_values("t_yr", kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------- completeness, b, df
def mc_maxc(m: np.ndarray, corr: float = 0.2) -> float:
    m = m[np.isfinite(m)]
    if len(m) < 50:
        return np.nan
    h, e = np.histogram(m, bins=np.arange(-2, 8, 0.1))
    return float(e[np.argmax(h)] + 0.05 + corr)


def b_aki(m: np.ndarray, mc: float) -> tuple[float, float, int]:
    m = m[np.isfinite(m) & (m >= mc)]
    if len(m) < 20:
        return np.nan, np.nan, len(m)
    b = 1.0 / (np.log(10) * (m.mean() - mc))
    db = 2.3 * b * b * np.sqrt(((m - m.mean()) ** 2).sum() / (len(m) * (len(m) - 1)))  # Shi & Bolt 1982
    return float(b), float(db), len(m)


def mc_bstab(m: np.ndarray) -> float:
    """Smallest Mc where b(Mc) agrees with the mean b over Mc..Mc+0.5 within its uncertainty."""
    m = m[np.isfinite(m)]
    grid = np.arange(np.floor(np.nanmin(m) * 10) / 10, np.nanmax(m) - 1.0, 0.1)
    bs = np.array([b_aki(m, g)[:2] for g in grid])
    for i, g in enumerate(grid):
        j = (grid >= g) & (grid <= g + 0.5 + 1e-9)
        if np.isnan(bs[i, 0]) or j.sum() < 5 or np.isnan(bs[j, 0]).any():
            continue
        if abs(bs[j, 0].mean() - bs[i, 0]) <= bs[i, 1]:
            return float(round(g, 2))
    return np.nan


def xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def corr_dim(lat, lon, rng, rlo=3.0, rhi=30.0, nmax=4000) -> float:
    idx = rng.choice(len(lat), size=min(nmax, len(lat)), replace=False)
    u = xyz(lat[idx], lon[idx])
    d = 2 * RE * np.arcsin(np.clip(np.sqrt(((u[:, None, :] - u[None, :, :]) ** 2).sum(-1)) / 2, 0, 1))
    d = d[np.triu_indices(len(idx), 1)]
    rr = np.logspace(np.log10(rlo), np.log10(rhi), 12)
    c = np.array([(d < r).mean() for r in rr])
    ok = c > 0
    return float(np.polyfit(np.log10(rr[ok]), np.log10(c[ok]), 1)[0])


def completeness_table(df: pd.DataFrame, nmin_year: int = 200) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, yrows = [], []
    rng = np.random.default_rng(0)
    for r in REG_ORDER:
        d = df[df.region == r]
        mcy = {}
        for y, g in d.groupby("year"):
            n = g.M.notna().sum()
            mcy[y] = mc_maxc(g.M.values)
            yrows.append(dict(region=r, year=y, n=len(g), n_M=n, Mc_maxc=mcy[y]))
        used = [y for y in mcy if (d.year == y).sum() >= nmin_year and np.isfinite(mcy[y])]
        mc = max(mcy[y] for y in used) if used else mc_maxc(d.M.values)
        b, db, nb = b_aki(d.M.values, mc)
        a = d[d.M >= mc]
        dfr = corr_dim(a.lat.values, a.lon.values, rng)
        rows.append(dict(region=r, name=REGION_NAMES[r], n=len(d), n_M=int(d.M.notna().sum()),
                         Mc_all=mc_maxc(d.M.values), Mc_bstab=mc_bstab(d.M.values), Mc=round(mc, 2),
                         years_used=";".join(map(str, used)), n_above=nb, b=b, b_err=db,
                         df_est=dfr, df_est_clip=float(np.clip(dfr, 1.2, 2.0)), df=1.6))
    return pd.DataFrame(rows), pd.DataFrame(yrows)



# ---------------------------------------------------------------- nearest neighbors
_G: dict = {}  # arrays shared with forked workers


def rupture_cap(m, rcap_min=10.0):
    return np.maximum(rcap_min, 3.0 * 10 ** (-2.44 + 0.59 * m))


def _nn_chunk(rng_):
    """Nearest earlier parent for children rng_[0]:rng_[1] of _G (see nn_search)."""
    a, b_ = rng_
    G = _G
    tc, uc, zc, dfc = G["tc"][a:b_], G["uc"][a:b_], G["zc"][a:b_], G["dfc"][a:b_]
    tp, up, zp, bmp = G["tp"], G["up"], G["zp"], G["bmp"]
    prefix, rmin, d3 = G["prefix"], G["rmin"], G["d3"]
    regc, regp, capp = G["regc"][a:b_], G["regp"], G["capp"]
    nc = b_ - a
    par = np.full(nc, -1)
    out = np.full((nc, 5), np.nan)
    sub = 256
    for s0 in range(0, nc, sub):
        sl = slice(s0, min(s0 + sub, nc))
        npar = (a + sl.stop) if prefix else np.searchsorted(tp, tc[sl].max(), side="left")
        if npar == 0:
            continue
        dt = tc[sl, None] - tp[None, :npar]
        if prefix:
            bad = np.arange(npar)[None, :] >= np.arange(a + sl.start, a + sl.stop)[:, None]
        else:
            bad = dt <= 0
        dt = np.where(bad, 1.0, np.maximum(dt, 1e-9))
        ch = np.sqrt(((uc[sl, None, :] - up[None, :npar, :]) ** 2).sum(-1))
        r = 2 * RE * np.arcsin(np.clip(ch / 2, 0, 1))
        if d3:
            r = np.sqrt(r ** 2 + (zc[sl, None] - zp[None, :npar]) ** 2)
        r = np.maximum(r, rmin)
        # parents from another region only within the rupture-length cap
        bad = bad | ((regc[sl, None] != regp[None, :npar]) & (r > capp[None, :npar]))
        dt = np.where(bad, 1.0, dt)
        lr_all = np.log10(r)
        le = np.log10(dt) + dfc[sl, None] * lr_all - bmp[None, :npar]
        le[bad] = np.inf
        k = np.argmin(le, axis=1)
        rows = np.arange(le.shape[0])
        lev = le[rows, k]
        ok = np.isfinite(lev)
        idx = np.arange(sl.start, sl.stop)[ok]
        kk = k[ok]
        par[idx] = kk
        lt = np.log10(dt[rows, k])[ok]
        lr = lr_all[rows, k][ok]
        out[idx, 0] = lev[ok]
        out[idx, 1] = lt - Q * bmp[kk]
        out[idx, 2] = dfc[sl][ok] * lr - (1 - Q) * bmp[kk]
        out[idx, 3] = r[rows, k][ok]
        out[idx, 4] = dt[rows, k][ok]
    return par, out


def nn_search(tc, uc, zc, dfc, tp, up, zp, bmp, rmin, d3, prefix, workers, regc, regp, capp, chunk=2000):
    """For each child, the earlier parent minimising eta.

    tc, tp: times (yr, sorted); uc, up: unit vectors; zc, zp: depth (km);
    dfc: df per child; bmp: b_i*m_i per parent; regc, regp: region codes;
    capp: distance cap per parent (applies only to parents in another region). prefix=True when children are
    the parents (parent index < child index).
    Returns parent index (-1 none) and array [log eta, log T, log R, r km, dt yr].
    """
    _G.update(regc=regc, regp=regp, capp=capp, tc=tc, uc=uc, zc=zc, dfc=dfc, tp=tp, up=up, zp=zp, bmp=bmp, prefix=prefix, rmin=rmin, d3=d3)
    n = len(tc)
    rngs = [(a, min(a + chunk, n)) for a in range(0, n, chunk)]
    # balance: prefix chunks at the end are costlier; ProcessPool handles the queue
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as ex:
        res = list(ex.map(_nn_chunk, rngs))
    return np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res])


# ---------------------------------------------------------------- threshold
def gmm1d(x: np.ndarray, iters: int = 500) -> dict:
    x = x[np.isfinite(x)]
    mu = np.quantile(x, [0.2, 0.8]).astype(float)
    sd = np.array([x.std() / 2] * 2)
    w = np.array([0.5, 0.5])
    for _ in range(iters):
        p = w / (sd * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x[:, None] - mu) / sd) ** 2)
        g = p / p.sum(1, keepdims=True)
        nk = g.sum(0)
        w2, mu2 = nk / len(x), (g * x[:, None]).sum(0) / nk
        sd2 = np.sqrt((g * (x[:, None] - mu2) ** 2).sum(0) / nk)
        if np.allclose(mu2, mu, atol=1e-6) and np.allclose(sd2, sd, atol=1e-6):
            w, mu, sd = w2, mu2, sd2
            break
        w, mu, sd = w2, mu2, sd2
    o = np.argsort(mu)
    w, mu, sd = w[o], mu[o], sd[o]
    grid = np.linspace(mu[0], mu[1], 2001)
    f = [w[k] / sd[k] * np.exp(-0.5 * ((grid - mu[k]) / sd[k]) ** 2) for k in (0, 1)]
    thr = float(grid[np.argmin(np.abs(f[0] - f[1]))])
    ashman = float(np.sqrt(2) * abs(mu[1] - mu[0]) / np.sqrt(sd[0] ** 2 + sd[1] ** 2))
    # KDE valley between the two means (check)
    kde = gaussian_kde(x if len(x) < 20000 else np.random.default_rng(1).choice(x, 20000, replace=False))
    kv = kde(grid)
    valley = float(grid[np.argmin(kv)])
    return dict(w1=w[0], mu1=mu[0], sd1=sd[0], w2=w[1], mu2=mu[1], sd2=sd[1],
                log_eta0=thr, log_eta0_kde=valley, ashman_D=ashman)


# ---------------------------------------------------------------- families
def tree_metrics(members: np.ndarray, parent: np.ndarray, linked: np.ndarray, m: np.ndarray):
    """members: sorted global indices (time order). parent: global parent index or -1."""
    pos = {g: k for k, g in enumerate(members)}
    n = len(members)
    depth = np.zeros(n, int)
    nchild = np.zeros(n, int)
    par_local = np.full(n, -1)
    for k, g in enumerate(members):
        p = parent[g]
        if linked[g] and p in pos:
            par_local[k] = pos[p]
            depth[k] = depth[pos[p]] + 1
            nchild[pos[p]] += 1
    leaves = nchild == 0
    imax = int(np.nanargmax(m[members]))
    return dict(mean_leaf_depth=float(depth[leaves].mean()) if leaves.any() else 0.0,
                max_depth=int(depth.max()),
                main_offspring_frac=float((par_local == imax).sum() / max(n - 1, 1)),
                mean_branching=float(nchild[~leaves].mean()) if (~leaves).any() else 0.0,
                imax_local=imax)


def migration(t_day, x, y):
    if len(t_day) < 10:
        return np.nan, np.nan, np.nan
    k0 = max(3, len(t_day) // 10)
    x0, y0 = x[:k0].mean(), y[:k0].mean()
    d = np.hypot(x - x0, y - y0)
    tt = t_day - t_day[0]
    if np.ptp(tt) <= 0:
        return np.nan, np.nan, np.nan
    rho, p = spearmanr(tt, d)
    sl = theilslopes(d, tt)[0]
    return float(sl), float(rho), float(p)


def classify(n, dM, frac_before, tm):
    if n < 5:
        return "multiplet" if dM < 0.3 else "mainshock-aftershock", (dM < 0.3, False, False)
    s1 = dM < 0.5
    s2 = frac_before >= 0.2
    s3 = (tm["mean_leaf_depth"] >= 2.0) and (tm["main_offspring_frac"] < 0.5)
    return ("swarm" if (s1 + s2 + s3) >= 2 else "mainshock-aftershock"), (s1, s2, s3)



# ---------------------------------------------------------------- one full run
def select(df, ctab, variant):
    """Analysis-set mask, magnitude used, df per event, b per event."""
    c = ctab.set_index("region")
    mm = df.M.values.copy()
    mc = df.region.map(c.Mc).values + (0.3 if variant == "mc+0.3" else 0.0)
    if variant == "noM_floor":
        mm = np.where(np.isnan(mm), mc, mm)
    if variant == "mc_year":
        mcy = np.array([c.Mc_year[r].get(y, np.nan) for r, y in zip(df.region, df.year)])
        mc = np.where(np.isfinite(mcy), mcy, mc)
    sel = np.isfinite(mm) & (mm >= mc - 1e-9)
    if variant in ("df1.3", "df2.0"):
        dfe = np.full(len(df), float(variant[2:]))
    elif variant == "df_est":
        dfe = df.region.map(c.df_est_clip).values
    else:
        dfe = df.region.map(c.df).values
    be = df.region.map(c.b).values
    return sel, mm, mc, dfe, be


def build(df, ctab, variant, a, thr_override=None, shuffle_seed=None):
    """Run one variant on the whole catalog. Returns events, families, region thresholds."""
    sel, mm, mcv, dfe, be = select(df, ctab, variant)
    A = df[sel].copy()
    A["m"] = mm[sel]
    A["dfe"], A["be"] = dfe[sel], be[sel]
    if shuffle_seed is not None:  # permute times among the events of each region
        rng = np.random.default_rng(shuffle_seed)
        for r, g in A.groupby("region"):
            A.loc[g.index, "t_yr"] = rng.permutation(g.t_yr.values)
        A = A.sort_values("t_yr", kind="stable")
    A = A.reset_index().rename(columns={"index": "row"})
    u = xyz(A.lat.values, A.lon.values)
    t, z, m = A.t_yr.values, A.depth.values, A.m.values
    bm = A.be.values * m
    rc = A.region.map({r: k for k, r in enumerate(REG_ORDER)}).values
    capm = rupture_cap(m, a.rcap_min)
    par, o = nn_search(t, u, z, A.dfe.values, t, u, z, bm, a.rmin, variant == "d3", True, a.workers,
                       rc, rc, capm)
    reg = A.region.values
    thr = {}
    for r in REG_ORDER:
        g = gmm1d(o[reg == r, 0])
        if thr_override is not None:
            g["log_eta0"] = thr_override[r]
        thr[r] = g
    le0 = np.array([thr[r]["log_eta0"] for r in reg])
    cap = rupture_cap(m[np.maximum(par, 0)], a.rcap_min)
    below = np.isfinite(o[:, 0]) & (o[:, 0] < le0)
    linked = below & (o[:, 3] <= cap)
    if shuffle_seed is not None:
        return dict(log_eta=o[:, 0], log_T=o[:, 1], log_R=o[:, 2], region=reg, below=below, linked=linked)
    n = len(A)
    ii = np.where(linked)[0]
    M_ = coo_matrix((np.ones(len(ii)), (ii, par[ii])), shape=(n, n))
    _, lab = connected_components(M_, directed=False)
    sizes = np.bincount(lab)
    fam_local = np.where(sizes[lab] >= 2, lab, -1)
    role = np.array(["single"] * n, dtype=object)
    fid = np.full(n, -1)
    fam_rows = []
    order = pd.Series(np.arange(n))[fam_local >= 0].groupby(fam_local[fam_local >= 0])
    # family ids ordered by start time
    starts = sorted(((g.values[0], c) for c, g in order), key=lambda x: x[0])
    groups = {c: g.values for c, g in order}
    for F, (_, c) in enumerate(starts, start=1):
        mem = groups[c]
        tm = tree_metrics(mem, par, linked, m)
        ms = np.sort(m[mem])[::-1]
        dM = float(ms[0] - ms[1])
        k = tm["imax_local"]
        frac_before = k / len(mem)
        typ, flags = classify(len(mem), dM, frac_before, tm)
        fid[mem] = F
        if typ in ("swarm", "multiplet"):
            role[mem] = typ
        else:
            role[mem[:k]] = "foreshock"
            role[mem[k + 1:]] = "aftershock"
            role[mem[k]] = "mainshock"
        la, lo = A.lat.values[mem], A.lon.values[mem]
        x = (lo - lo.mean()) * 111.195 * np.cos(np.radians(la.mean()))
        y = (la - la.mean()) * 111.195
        dc = np.hypot(x, y)
        tday = (t[mem] - t[mem[0]]) * 365.25
        v, rho, p = migration(tday, x, y)
        regs = pd.Series(reg[mem]).value_counts()
        fam_rows.append(dict(
            family_id=F, region=reg[mem[k]], regions=";".join(f"{q}:{w}" for q, w in regs.items()),
            n=len(mem), Mmax=float(ms[0]), M2=float(ms[1]), dM=dM,
            mainshock_id=int(A.event_id.values[mem[k]]),
            start=A.time.values[mem[0]], end=A.time.values[mem[-1]], duration_days=float(tday[-1]),
            duration90_days=float(np.quantile(tday, 0.9)),
            lon=float(lo.mean()), lat=float(la.mean()), lon_main=float(lo[k]), lat_main=float(la[k]),
            depth_med=float(np.median(A.depth.values[mem])),
            extent_km=float(2 * np.quantile(dc, 0.95)), type=typ,
            S1_no_dominant=flags[0], S2_late_peak=flags[1], S3_chain=flags[2],
            frac_before_largest=frac_before, t_largest_days=float(tday[k]),
            mean_leaf_depth=tm["mean_leaf_depth"], max_depth=tm["max_depth"],
            main_offspring_frac=tm["main_offspring_frac"], mean_branching=tm["mean_branching"],
            migration_km_day=v, migration_rho=rho, migration_p=p,
            migrating=bool(np.isfinite(rho) and rho >= 0.5 and p < 0.01 and v > 0)))
    fam = pd.DataFrame(fam_rows)
    ev = pd.DataFrame(dict(
        event_id=A.event_id.values, region=reg, in_analysis=True, M=m,
        parent_id=np.where(par >= 0, A.event_id.values[np.maximum(par, 0)], -1),
        log_eta=o[:, 0], log_T=o[:, 1], log_R=o[:, 2], nn_dist_km=o[:, 3], nn_dt_days=o[:, 4] * 365.25,
        linked=linked, family_id=fid, role=role))
    # ---- attach events below Mc / without M
    O = df[~sel]
    if len(O):
        pp, oo = nn_search(O.t_yr.values, xyz(O.lat.values, O.lon.values), O.depth.values, dfe[~sel],
                           t, u, z, bm, a.rmin, variant == "d3", False, a.workers,
                           O.region.map({r: k for k, r in enumerate(REG_ORDER)}).values, rc, capm)
        le0o = O.region.map({r: thr[r]["log_eta0"] for r in thr}).values
        capo = rupture_cap(m[np.maximum(pp, 0)], a.rcap_min)
        lk = np.isfinite(oo[:, 0]) & (oo[:, 0] < le0o) & (oo[:, 3] <= capo)
        pf = np.where(pp >= 0, fid[np.maximum(pp, 0)], -1)
        ftype = dict(zip(fam.family_id, fam.type)) if len(fam) else {}
        tmain = df.set_index("event_id").t_yr
        mt = dict(zip(fam.family_id, tmain.reindex(fam.mainshock_id).values)) if len(fam) else {}
        orole = np.where(~lk, "single_belowMc", "linked_to_single_belowMc").astype(object)
        to = O.t_yr.values
        for q in np.where(lk & (pf >= 0))[0]:
            ty = ftype[pf[q]]
            if ty == "swarm":
                orole[q] = "swarm_belowMc"
            elif ty == "multiplet":
                orole[q] = "multiplet_belowMc"
            else:
                orole[q] = "foreshock_belowMc" if to[q] < mt[pf[q]] else "aftershock_belowMc"
        ev2 = pd.DataFrame(dict(
            event_id=O.event_id.values, region=O.region.values, in_analysis=False, M=O.M.values,
            parent_id=np.where(pp >= 0, A.event_id.values[np.maximum(pp, 0)], -1),
            log_eta=oo[:, 0], log_T=oo[:, 1], log_R=oo[:, 2], nn_dist_km=oo[:, 3],
            nn_dt_days=oo[:, 4] * 365.25, linked=lk, family_id=np.where(lk, pf, -1), role=orole))
        ev = pd.concat([ev, ev2], ignore_index=True)
    regt = pd.DataFrame([dict(region=r, n_analysis=int((reg == r).sum()),
                              Mc=float(np.nanmax(mcv[sel & (df.region.values == r)])) if variant != "mc_year"
                              else np.nan,
                              n_below_eta0=int(below[reg == r].sum()),
                              n_cut_by_cap=int((below & ~linked)[reg == r].sum()), **thr[r])
                         for r in REG_ORDER])
    regt = regt.merge(ctab[["region", "b", "df"]], on="region")
    return ev, fam, regt, thr


def summarize(ev, fam, reg):
    rows = []
    for r in REG_ORDER:
        e = ev[(ev.region == r) & ev.in_analysis]
        f = fam[fam.region == r] if len(fam) else fam
        rr = reg[reg.region == r].iloc[0]
        rows.append(dict(
            region=r, n_analysis=len(e), Mc=rr.Mc, b=round(rr.b, 2), df=round(rr.df, 2),
            log_eta0=round(rr.log_eta0, 2), ashman_D=round(rr.ashman_D, 2),
            pct_background=round(100 * (e.role == "single").mean(), 1),
            pct_MA=round(100 * e.role.isin(["mainshock", "foreshock", "aftershock"]).mean(), 1),
            pct_swarm=round(100 * (e.role == "swarm").mean(), 1),
            pct_multiplet=round(100 * (e.role == "multiplet").mean(), 1),
            n_families=len(f), n_MA=int((f.type == "mainshock-aftershock").sum()) if len(f) else 0,
            n_MA_ge5=int(((f.type == "mainshock-aftershock") & (f.n >= 5)).sum()) if len(f) else 0,
            n_swarm=int((f.type == "swarm").sum()) if len(f) else 0,
            n_multiplet=int((f.type == "multiplet").sum()) if len(f) else 0))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- faults
def fault_join(ev, fam, out):
    mem = pd.read_csv(FAULTS / "fault_surfaces_v0_members.csv")
    summ = pd.read_csv(FAULTS / "fault_surfaces_v0_summary.csv", usecols=["surface_id", "region", "lit_match"])
    e = ev.merge(mem, on="event_id", how="left")
    # per family: fraction of members (all, incl. below-Mc) on each surface
    ff = e[e.family_id > 0].groupby("family_id")
    rows = []
    for F, g in ff:
        vc = g.surface_id.value_counts()
        fr = (vc / len(g)).round(3)
        rows.append(dict(family_id=F, n_all=len(g), frac_on_faults=round(g.surface_id.notna().mean(), 3),
                         main_fault=vc.index[0] if len(vc) else "",
                         fault_ids=";".join(f"{k}:{v}" for k, v in fr.items())))
    fj = pd.DataFrame(rows)
    fam2 = fam.drop(columns=[c for c in ["n_all", "frac_on_faults", "main_fault", "fault_ids"] if c in fam],
                    errors="ignore").merge(fj, on="family_id", how="left")
    # per fault: background vs clustered (analysis set only, and all)
    ea = e[e.surface_id.notna()].merge(fam[["family_id", "type"]], on="family_id", how="left")
    ea["cls"] = np.where(ea.family_id > 0, ea.type, "background")
    pf = []
    for sid, g in ea.groupby("surface_id"):
        a = g[g.in_analysis]
        d = dict(surface_id=sid, n_events=len(g), n_analysis=len(a))
        for c in ["background", "mainshock-aftershock", "swarm", "multiplet"]:
            d[f"n_{c}"] = int((a.cls == c).sum())
        d["pct_background"] = round(100 * d["n_background"] / len(a), 1) if len(a) else np.nan
        d["pct_swarm"] = round(100 * d["n_swarm"] / len(a), 1) if len(a) else np.nan
        d["n_families"] = int(a.family_id[a.family_id > 0].nunique())
        pf.append(d)
    pf = pd.DataFrame(pf).merge(summ, on="surface_id", how="left")
    pf.to_csv(out / "faults_v0_cluster_counts.csv", index=False)
    return fam2, pf


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--rmin", type=float, default=0.5, help="distance floor (km)")
    ap.add_argument("--rcap-min", type=float, default=10.0, help="minimum link distance cap (km)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--nshuffle", type=int, default=3)
    ap.add_argument("--variants", default="d3,mc+0.3,df_est,df1.3,df2.0,noM_floor,mc_year,eta0_shuf5",
                    help="comma list of sensitivity variants ('' for none)")
    ap.add_argument("--faults-only", action="store_true", help="redo only the fault join")
    a = ap.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    if a.faults_only:
        ev = pd.read_csv(out / "events_zbz.csv.gz")
        fam = pd.read_csv(out / "families_zbz.csv")
        fam, _ = fault_join(ev, fam, out)
        fam.to_csv(out / "families_zbz.csv", index=False, float_format="%.4f")
        return
    df = load()
    print(f"{len(df)} events, {df.M.notna().sum()} with M", flush=True)
    ctab, ytab = completeness_table(df)
    ctab["Mc_year"] = [ytab[ytab.region == r].set_index("year").Mc_maxc.to_dict() for r in ctab.region]
    ctab.drop(columns="Mc_year").to_csv(out / "completeness_b_df.csv", index=False, float_format="%.3f")
    ytab.to_csv(out / "completeness_by_year.csv", index=False, float_format="%.2f")
    print(ctab.drop(columns="Mc_year").to_string(), flush=True)

    ev, fam, reg, thr = build(df, ctab, "main", a)
    fam, pf = fault_join(ev, fam, out)
    summ = summarize(ev, fam, reg)
    print(summ.to_string(), flush=True)

    # shuffled catalogs
    srows, shuf_all = [], {}
    for sd in range(a.nshuffle):
        s = build(df, ctab, "main", a, shuffle_seed=100 + sd,
                  thr_override={r: thr[r]["log_eta0"] for r in thr})
        for r in REG_ORDER:
            k = s["region"] == r
            le = s["log_eta"][k]
            g = gmm1d(le)
            srows.append(dict(region=r, seed=sd, n=int(k.sum()),
                              frac_below_eta0=float(np.nanmean(le < thr[r]["log_eta0"])),
                              frac_linked=float(s["linked"][k].mean()),
                              ashman_D=g["ashman_D"], mu1=g["mu1"], mu2=g["mu2"], w1=g["w1"]))
            shuf_all.setdefault(r, []).append(le[np.isfinite(le)])
        if sd == 0:
            np.savez_compressed(out / "shuffle_eta_seed0.npz", log_eta=s["log_eta"], log_T=s["log_T"],
                                log_R=s["log_R"], region=s["region"].astype(str))
    sh = pd.DataFrame(srows)
    sh.to_csv(out / "shuffle_test.csv", index=False, float_format="%.4f")
    print(sh.groupby("region").mean(numeric_only=True).to_string(), flush=True)
    thr_shuf = {r: float(np.quantile(np.concatenate(v), 0.05)) for r, v in shuf_all.items()}
    reg["log_eta0_shuf5"] = reg.region.map(thr_shuf)
    reg.to_csv(out / "thresholds_zbz.csv", index=False, float_format="%.4f")

    ev = ev.merge(df[["event_id", "time", "lat", "lon", "depth"]], on="event_id", how="left")
    ev.sort_values("event_id").to_csv(out / "events_zbz.csv.gz", index=False, float_format="%.4f")
    fam.to_csv(out / "families_zbz.csv", index=False, float_format="%.4f")
    summ.to_csv(out / "regions_zbz.csv", index=False)

    # sensitivity variants
    vs = [v for v in a.variants.split(",") if v]
    allv = [summ.assign(variant="main")]
    main_role = ev[ev.in_analysis].set_index("event_id").role
    for v in vs:
        ev_v, fam_v, reg_v, _ = build(df, ctab, v, a, thr_override=thr_shuf if v == "eta0_shuf5" else None)
        sv = summarize(ev_v, fam_v, reg_v).assign(variant=v)
        e2 = ev_v[ev_v.in_analysis].set_index("event_id")
        com = main_role.index.intersection(e2.index)
        same = pd.Series((main_role[com] == "single").values == (e2.role[com] == "single").values, index=com)
        agree = same.groupby(e2.region[com]).mean()
        sv["agree_bg_vs_clustered_with_main"] = sv.region.map(agree).round(3)
        allv.append(sv)
        print(v, "\n", sv.to_string(), flush=True)
    pd.concat(allv).to_csv(out / "sensitivity_zbz.csv", index=False)
    json.dump(dict(rmin=a.rmin, rcap_min=a.rcap_min, q=Q, nshuffle=a.nshuffle, variants=vs, df_main=1.6,
                   catalog=str(CAT), magnitudes=str(MAG)),
              open(out / "run_params.json", "w"), indent=1)


if __name__ == "__main__":
    main()

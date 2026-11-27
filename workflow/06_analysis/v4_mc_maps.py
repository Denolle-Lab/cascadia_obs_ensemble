"""Maps of the magnitude of completeness (Mc) for the Cascadia catalog v4.

Three standard estimators are computed at every node of a 0.1 deg grid, on the
nearest N events that have a magnitude (Route A M):

  MAXC  maximum curvature: Mc = centre of the most populated 0.1 bin of the
        non-cumulative FMD, plus +0.2 (Wiemer & Wyss 2000; correction from
        Woessner & Wiemer 2005).
  MBS   b-value stability (Cao & Gao 2002), in the form of Woessner & Wiemer
        (2005, eq. 4): b(Mc) from Aki-Utsu, b_ave = mean of b over the five
        bins Mc ... Mc+0.4 (a 0.5 window); Mc is the lowest cutoff with
        |b_ave - b(Mc)| <= delta_b, delta_b being the Shi & Bolt (1982) error.
  GFT   goodness of fit (Wiemer & Wyss 2000): for each cutoff, fit a and b
        (Aki-Utsu) to M >= Mc, compare the synthetic and observed cumulative
        counts, R = 100 - 100 * sum|B_i - S_i| / sum B_i. Mc90 and Mc95 are
        the lowest cutoffs with R >= 90 and R >= 95.
  best  the combination rule of Woessner & Wiemer (2005): Mc95 if it exists,
        else Mc90, else MAXC+0.2.

Uncertainty: 200 bootstrap resamples of the node's magnitudes (multinomial draws
on the 0.1 bin histogram, the same as resampling events with replacement);
sigma_boot is the standard deviation of the bootstrap Mc_best (Woessner &
Wiemer 2005). b and b_sigma are Aki-Utsu (Aki 1965; Utsu 1965) with the Shi &
Bolt (1982) error, fitted above Mc_best on the full node sample.

Magnitudes are rounded to 0.1 before any calculation (Aki-Utsu then uses the
bin edge Mc - 0.05). A b fit needs at least NMIN_B = 30 events above the cutoff.

Spatial sampling: grid every 0.1 deg over lon -131 to -119.5, lat 39 to 51. At
each node, the nearest N events (default 250) are used if they all lie within
RMAX = 50 km (epicentral, great circle); otherwise the node is left empty. The
period maps use N = 150 by default (fewer events per period; the bootstrap sigma
shows the cost). Events of any QC are used by default; --qc keeps events with
at least 4 P and 4 S picks.

Regions (most specific first) are those of v4_clusters_zbz.py; the regional FMD
figure compares the pooled-regional Mc from the three methods with the Mc used
in clusters_zbz (completeness_b_df.csv).

The FMD, MAXC, Aki-Utsu/Shi-Bolt and b-stability functions are adapted from
AgentSeismo (Z. Zhang), studies/gibraltar_2010_2011/02_assessment/mc_extended.py;
here they are vectorised over a histogram so the bootstrap runs on all
resamples at once, and the MBS test uses the Shi-Bolt error instead of a fixed
0.03 tolerance. GFT is new.

References
  Aki, K. (1965). Bull. Earthq. Res. Inst. Univ. Tokyo 43, 237-239.
  Utsu, T. (1965). Geophys. Bull. Hokkaido Univ. 13, 99-103.
  Shi, Y. & Bolt, B. A. (1982). BSSA 72(5), 1677-1687. doi:10.1785/BSSA0720051677
  Wiemer, S. & Wyss, M. (2000). BSSA 90(4), 859-869. doi:10.1785/0119990114
  Cao, A. & Gao, S. S. (2002). GRL 29(9), 1334. doi:10.1029/2001GL013775
  Woessner, J. & Wiemer, S. (2005). BSSA 95(2), 684-698. doi:10.1785/0120040007
  Mignan, A. & Woessner, J. (2012). CORSSA. doi:10.5078/corssa-00180805

Other catalogs (D-B102, D-B104): --catalog takes a catalog bundle (v5.1, v6.0
schema: ID, time, lat, lon and its own magnitude columns) instead of the v4
catalog and the --mag file, so that each catalog's Mc map uses its own
hypocentres and its own magnitudes (--mag-col, default M; never recomputed).
Selection: located events (lat and lon set) with a magnitude, any tier. The QC
variant counts P and S picks per event in the bundle's own picks file (--picks).

Not earthquakes (D-B105): --exclude-flagged drops events with an event_flag
(quarry_blast_likely, anss_explosion, anss_quarry_blast, anss_other_event; see
workflow/10_tomography/b4_catalog.py). A bundle uses its own event_flag column;
--flags names other files with ID and event_flag columns (the IDs flagged in any
of them are dropped), which the v4 catalog needs since it has no flag column.

Usage
  python v4_mc_maps.py                 # compute everything, then plot
  python v4_mc_maps.py --stage plot    # figures only, from the saved CSVs
  python v4_mc_maps.py --mag stored --out <dir>   # the maps before D-B091 (default: converged magnitudes)
  python v4_mc_maps.py --catalog <bundle>/cascadia_catalog_v6.0.csv \\
      --picks <bundle>/Cascadia_picks_catalog_v4.1tc.csv.gz --exclude-flagged --out <dir>   # catalog v6.0
  python v4_mc_maps.py --exclude-flagged --flags <v5.1 catalog> <v6.0 catalog> --out <dir>   # v4.1, D-B105
"""
from __future__ import annotations

import argparse
import json
import zlib
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "workflow" / "05_magnitude"))
import mag_paths  # noqa: E402  (converged Route A magnitudes by default, D-B091)

MAIN = Path("/home/mdenolle/cascadia_obs_ensemble")
CAT = MAIN / "data/catalog_v4/Cascadia_relocated_catalog_v4.csv"
MAG = mag_paths.v41()        # --mag / $CASCADIA_MAG_V41: converged (default), stored or a path
STA = Path("/wd1/mdenolle_data/picks_share/stations/stations_ver5_v6.csv")
ZBZ = ROOT / "data/catalog_v4/clusters_zbz/completeness_b_df.csv"
OUT = ROOT / "data/catalog_v4/mc_maps"
# --catalog: a bundle with its own hypocentres and magnitudes replaces CAT + MAG (set in main())
BUNDLE: dict | None = None
LABEL = "v4"
# --exclude-flagged (D-B105): {"files": event_flag sources, "ids": flagged IDs, or None for the bundle's own column}
FLAGS: dict | None = None
SEL: dict = {}               # selection counts of the last load_catalog calls (run_params.json)

REGIONS = [  # most specific first (same as v4_clusters_zbz.py)
    ("AX", "Axial Seamount", [-130.35, -129.75, 45.75, 46.2]),
    ("MT", "Mendocino triple junction", [-125.6, -123.2, 39.6, 41.4]),
    ("PU", "Puget Lowland", [-123.6, -121.4, 46.6, 48.6]),
    ("EN", "Endeavour-Explorer", [-130.4, -127.2, 47.4, 49.8]),
    ("BL", "Blanco transform", [-130.6, -126.6, 42.6, 45.3]),
    ("GO", "Gorda ridge and plate", [-128.8, -123.8, 40.0, 43.2]),
    ("WA", "Offshore Washington", [-127.6, -123.2, 46.2, 49.2]),
    ("OR", "Offshore Oregon", [-126.8, -123.2, 42.8, 46.4]),
]
OFFSHORE = ["EN", "BL", "GO", "WA", "OR"]
REG_ORDER = ["AX", "EN", "BL", "GO", "MT", "WA", "OR", "PU", "ON"]
REGION_NAMES = {k: n for k, n, _ in REGIONS} | {"ON": "Onshore and other"}
PERIODS = {"2010": (2010, 2010), "2011-2012": (2011, 2012),
           "2013-2015": (2013, 2015), "2016": (2016, 2016)}

DM = 0.1
MLO, MHI = -2.5, 7.5
MBINS = np.round(np.arange(MLO, MHI + DM / 2, DM), 2)   # bin centres
LN10 = np.log(10.0)
NMIN_B = 30
CORR = 0.2
WIN = 5
RE = 6371.0
REGION_MAP = [-131.0, -119.5, 39.0, 51.0]


# ---------------------------------------------------------------- data
def pick_counts(path: Path) -> pd.DataFrame:
    """Number of P and S picks per event in a picks file (event_ID, phase), as Num_p_picks / Num_s_picks."""
    p = pd.read_csv(path, skipinitialspace=True, usecols=lambda c: c.strip() in ("event_ID", "phase"))
    p.columns = [c.strip() for c in p.columns]
    k = p.assign(phase=p.phase.str.strip()).groupby(["event_ID", "phase"]).size().unstack(fill_value=0)
    return k.reindex(columns=["P", "S"], fill_value=0).rename(columns={"P": "Num_p_picks", "S": "Num_s_picks"})


def flagged_ids(files) -> set:
    """IDs with an event_flag (not an earthquake, D-B071 / D-B105) in any of the files (columns ID, event_flag)."""
    ids = set()
    for f in files:
        c = pd.read_csv(f, usecols=["ID", "event_flag"], low_memory=False)
        ids |= set(c.ID[c.event_flag.fillna("").astype(str).str.strip() != ""].astype(int))
    return ids


def describe_flags() -> dict:
    if not FLAGS:
        return {}
    files = FLAGS["files"] or [BUNDLE["catalog"]]
    return {"exclude_flagged": True,
            "flags_from": [str(f) for f in files], "flags_md5": [mag_paths.md5(f) for f in files],
            "flags_rule": "event_flag set (not empty) in any of flags_from"}


def load_bundle(qc: bool = False) -> pd.DataFrame:
    """Events of a catalog bundle (--catalog) with its own hypocentres (lat, lon, time) and magnitudes (--mag-col):
    located events with a magnitude; flagged events dropped with --exclude-flagged. Counts go to SEL['selection']
    (SEL['selection_qc'] for the QC variant) and from there to run_params.json."""
    b = BUNDLE
    cat = pd.read_csv(b["catalog"], low_memory=False)
    miss = {"ID", "time", "lat", "lon", b["mag_col"]} - set(cat.columns)
    if miss:
        raise SystemExit(f"{b['catalog']}: missing columns {sorted(miss)}")
    df = pd.DataFrame({"event_id": cat.ID, "datetime": cat.time, "lat": cat.lat, "lon": cat.lon,
                       "M": cat[b["mag_col"]].astype(float)})
    if FLAGS and FLAGS["files"]:
        flag = cat.ID.isin(FLAGS["ids"])
    elif "event_flag" in cat:
        flag = cat.event_flag.fillna("").astype(str).str.strip() != ""
    elif FLAGS:
        raise SystemExit(f"{b['catalog']}: no event_flag column; give --flags")
    else:
        flag = pd.Series(False, index=cat.index)
    loc = df.lat.notna() & df.lon.notna()
    keep = loc & df.M.notna()
    n = {"rows": len(df), "no_hypocentre": int((~loc).sum()), "located_no_magnitude": int((loc & df.M.isna()).sum()),
         "flagged_with_magnitude": int((keep & flag).sum()), "exclude_flagged": bool(FLAGS)}
    if FLAGS:
        keep &= ~flag
    if qc:
        k = pick_counts(b["picks"])
        df = df.join(k, on="event_id")
        q = (df.Num_p_picks.fillna(0) >= 4) & (df.Num_s_picks.fillna(0) >= 4)
        n["below_4P_4S"] = int((keep & ~q).sum())
        keep &= q
    n["used"] = int(keep.sum())
    if "magnitude_source" in cat:
        n["magnitude_source"] = {str(s): int(c) for s, c in cat.magnitude_source[keep].value_counts().items()}
    SEL["selection_qc" if qc else "selection"] = n
    return df[keep]


def describe_bundle() -> dict:
    b = BUNDLE
    return {"catalog": str(b["catalog"]), "catalog_md5": mag_paths.md5(b["catalog"]), "catalog_label": LABEL,
            "magnitudes": str(b["catalog"]), "magnitudes_column": b["mag_col"],
            "magnitudes_md5": mag_paths.md5(b["catalog"]), "epicentres": "lat, lon of the catalog",
            "picks": str(b["picks"]) if b["picks"] else None,
            "picks_md5": mag_paths.md5(b["picks"]) if b["picks"] else None}


def load_catalog(qc: bool = False) -> pd.DataFrame:
    if BUNDLE:
        df = load_bundle(qc).copy()
    else:
        cat = pd.read_csv(CAT, skipinitialspace=True)
        cat.columns = [c.strip() for c in cat.columns]
        mag = pd.read_csv(MAG, usecols=["event_id", "M"])
        df = cat.rename(columns={"ID": "event_id", "latitude": "lat", "longitude": "lon"})
        df = df.merge(mag, on="event_id", how="left")
        if qc:
            df = df[(df.Num_p_picks >= 4) & (df.Num_s_picks >= 4)]
        df = df[df.M.notna()].copy()
        if FLAGS:
            fl = df.event_id.isin(FLAGS["ids"])
            SEL["selection_qc" if qc else "selection"] = {
                "rows": len(cat), "with_magnitude": len(df), "flagged_with_magnitude": int(fl.sum()),
                "exclude_flagged": True, "used": int((~fl).sum())}
            df = df[~fl].copy()
    df["year"] = pd.to_datetime(df["datetime"], utc=True, format="ISO8601").dt.year
    df["ibin"] = np.clip(np.round((df.M.values - MLO) / DM).astype(int), 0, len(MBINS) - 1)
    reg = np.full(len(df), "ON", dtype=object)
    for key, _, (a, b, c, d) in REGIONS[::-1]:
        s = df.lon.between(a, b) & df.lat.between(c, d)
        reg[s.values] = key
    df["region"] = reg
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- Mc estimators
def mc_counts(C: np.ndarray, m: np.ndarray) -> dict:
    """All three Mc estimators for each row of a histogram matrix C (S x K),
    bin centres m (K). Returns arrays of length S."""
    C = np.atleast_2d(C).astype(float)
    S, K = C.shape
    n = np.cumsum(C[:, ::-1], 1)[:, ::-1]                    # N(>= m_j)
    s1 = np.cumsum((C * m)[:, ::-1], 1)[:, ::-1]
    s2 = np.cumsum((C * m * m)[:, ::-1], 1)[:, ::-1]
    ok = n >= NMIN_B
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = s1 / n
        b = 1.0 / (LN10 * (mean - (m - DM / 2)))              # Aki-Utsu
        var = np.clip(s2 - n * mean ** 2, 0, None)
        sb = 2.30 * b ** 2 * np.sqrt(var / (n * (n - 1)))    # Shi & Bolt (1982)
    b[~ok] = np.nan
    sb[~ok] = np.nan
    rows = np.arange(S)

    def first(cond):
        has = cond.any(1)
        j = np.argmax(cond, 1)
        return np.where(has, m[np.minimum(j, K - 1)], np.nan), np.where(has, j, -1)

    # MAXC (+0.2)
    maxc = m[np.argmax(C, 1)] + CORR

    # MBS, Woessner & Wiemer (2005) eq. 4
    if K >= WIN:
        bave = sliding_window_view(b, WIN, axis=1).mean(-1)
        L = bave.shape[1]
        with np.errstate(invalid="ignore"):
            cond = np.abs(bave - b[:, :L]) <= sb[:, :L]
        cond &= np.isfinite(bave)
        mbs, _ = first(cond)
    else:
        mbs = np.full(S, np.nan)

    # GFT, Wiemer & Wyss (2000), cumulative counts
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        a = np.log10(n) + b * m
        synth = 10.0 ** (a[:, :, None] - b[:, :, None] * m[None, None, :])  # S, j, k
        upper = np.triu(np.ones((K, K), bool))                               # k >= j
        num = np.where(upper, np.abs(n[:, None, :] - synth), 0.0).sum(-1)
        den = np.where(upper, n[:, None, :], 0.0).sum(-1)
        R = 100.0 - 100.0 * num / den
    R[~ok] = np.nan
    with np.errstate(invalid="ignore"):
        g90, _ = first(R >= 90)
        g95, _ = first(R >= 95)

    best = np.where(np.isfinite(g95), g95, np.where(np.isfinite(g90), g90, maxc))
    jb = np.clip(np.round((best - m[0]) / DM).astype(int), 0, K - 1)
    return dict(mc_maxc=maxc, mc_mbs=mbs, mc_gft90=g90, mc_gft95=g95, mc_best=best,
                b=b[rows, jb], b_sigma=sb[rows, jb], n_above=n[rows, jb])


def mc_with_bootstrap(ibin: np.ndarray, nboot: int, seed: int) -> dict:
    """Point estimates on the full sample + bootstrap standard deviations."""
    h = np.bincount(ibin, minlength=len(MBINS))
    nz = np.nonzero(h)[0]
    lo, hi = nz[0], nz[-1] + 1
    hi = max(hi, lo + WIN + 1)
    h = h[lo:hi]
    m = MBINS[lo:hi]
    N = int(h.sum())
    rng = np.random.default_rng(seed)
    C = np.vstack([h, rng.multinomial(N, h / N, size=nboot)])
    r = mc_counts(C, m)
    out = {k: float(v[0]) for k, v in r.items()}
    with np.errstate(invalid="ignore"):
        for k in ("mc_maxc", "mc_mbs", "mc_best"):
            v = r[k][1:]
            out["sigma_" + k[3:]] = float(np.nanstd(v)) if np.isfinite(v).sum() > 10 else np.nan
        g = np.where(np.isfinite(r["mc_gft95"][1:]), r["mc_gft95"][1:], r["mc_gft90"][1:])
        out["sigma_gft"] = float(np.nanstd(g)) if np.isfinite(g).sum() > 10 else np.nan
    out["sigma_boot"] = out.pop("sigma_best")
    out["n"] = N
    return out


# ---------------------------------------------------------------- grid
def xyz(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


_G = {}


def _node(args):
    k, idx, rad = args
    r = mc_with_bootstrap(_G["ibin"][idx], _G["nboot"], seed=k)
    r["radius_km"] = rad
    return k, r


def mc_grid(df: pd.DataFrame, n_ev: int, rmax: float, nboot: int, step: float,
            nproc: int) -> pd.DataFrame:
    lons = np.round(np.arange(REGION_MAP[0], REGION_MAP[1] + step / 2, step), 3)
    lats = np.round(np.arange(REGION_MAP[2], REGION_MAP[3] + step / 2, step), 3)
    LO, LA = np.meshgrid(lons, lats)
    LO, LA = LO.ravel(), LA.ravel()
    tree = cKDTree(xyz(df.lon.values, df.lat.values))
    chord = 2 * np.sin(rmax / (2 * RE))
    rows = []
    jobs = []
    if len(df) >= n_ev:
        d, ii = tree.query(xyz(LO, LA), k=n_ev, distance_upper_bound=chord * 1.0000001)
        d = np.atleast_2d(d)
        ii = np.atleast_2d(ii)
        good = np.isfinite(d[:, -1])
        for k in np.nonzero(good)[0]:
            rad = 2 * RE * np.arcsin(d[k, -1] / 2)
            jobs.append((int(k), ii[k], float(rad)))
    _G["ibin"] = df.ibin.values
    _G["nboot"] = nboot
    res = {}
    if jobs:
        with Pool(nproc) as p:
            for k, r in p.imap_unordered(_node, jobs, chunksize=16):
                res[k] = r
    cols = ["mc_maxc", "mc_mbs", "mc_gft90", "mc_gft95", "mc_best", "sigma_boot", "b",
            "b_sigma", "sigma_maxc", "sigma_mbs", "sigma_gft", "n_above"]
    for k in range(len(LO)):
        r = res.get(k)
        row = dict(lon=LO[k], lat=LA[k], n=r["n"] if r else 0,
                   radius_km=r["radius_km"] if r else np.nan)
        for c in cols:
            row[c] = r[c] if r else np.nan
        rows.append(row)
    g = pd.DataFrame(rows)
    return g


def to_xr(g: pd.DataFrame):
    import xarray as xr
    g = g.set_index(["lat", "lon"]).sort_index()
    ds = xr.Dataset.from_dataframe(g)
    ds.lon.attrs.update(units="degrees_east")
    ds.lat.attrs.update(units="degrees_north")
    return ds


# ---------------------------------------------------------------- regional
def regional_table(df: pd.DataFrame, nboot: int) -> pd.DataFrame:
    rows = []
    d0 = df[df.year.between(2013, 2015)]
    for per, (y0, y1) in [("all", (2010, 2016))] + list(PERIODS.items()):
        dp = df[df.year.between(y0, y1)]
        for reg in REG_ORDER + ["OFF", "ALL"]:
            if reg == "ALL":
                d = dp
            elif reg == "OFF":   # offshore plate-boundary regions pooled (no AX, no MT)
                d = dp[dp.region.isin(OFFSHORE)]
            else:
                d = dp[dp.region == reg]
            row = dict(region=reg, period=per, n=len(d))
            dref = d0 if reg == "ALL" else (d0[d0.region.isin(OFFSHORE)] if reg == "OFF"
                                            else d0[d0.region == reg])
            row |= rate_ratio(MBINS[d.ibin.values], y1 - y0 + 1, MBINS[dref.ibin.values], 3.0)
            if len(d) >= 100:
                row |= mc_with_bootstrap(d.ibin.values, nboot, seed=zlib.crc32(f"{reg}{per}".encode()))
            rows.append(row)
    return pd.DataFrame(rows)


def rate_ratio(m_p, years_p, m_ref, years_ref, mags=(2.0, 2.5, 3.0)):
    """Annual rate of events >= M in the period divided by the rate in the
    reference period (2013-2015, densest network). A check that needs no GR fit:
    near 1 where both periods are complete at M, well below 1 where the period
    misses events of that size (or where the true rate changed: swarms and
    aftershock sequences make it noisy)."""
    out = {}
    for M in mags:
        nr = (m_ref >= M - 1e-9).sum()
        out[f"rr{int(M * 10):02d}"] = (((m_p >= M - 1e-9).sum() / years_p) / (nr / years_ref)
                                       if nr >= 10 else np.nan)
    return out


def node_summary(g: pd.DataFrame, period: str) -> pd.DataFrame:
    """Percentiles of node Mc_best per region."""
    import pandas as pd
    gg = g[g.mc_best.notna()].copy()
    reg = np.full(len(gg), "ON", dtype=object)
    for key, _, (a, b, c, d) in REGIONS[::-1]:
        s = gg.lon.between(a, b) & gg.lat.between(c, d)
        reg[s.values] = key
    gg["region"] = reg
    out = gg.groupby("region").mc_best.describe(percentiles=[0.1, 0.5, 0.9])
    out = out[["count", "min", "10%", "50%", "90%", "max"]].reindex(REG_ORDER)
    out.insert(0, "period", period)
    return out.reset_index()


# ---------------------------------------------------------------- compute driver
def compute(a):
    OUT.mkdir(parents=True, exist_ok=True)
    import xarray as xr
    meta = dict(n_events=a.n, n_events_period=a.n_period, rmax_km=a.rmax, nboot=a.nboot,
                step_deg=a.step, nmin_b=NMIN_B, maxc_corr=CORR, mbs_window_bins=WIN,
                dm=DM, periods=PERIODS, **(describe_bundle() if BUNDLE else mag_paths.describe(MAG)), **describe_flags())
    df = load_catalog(qc=False)
    print(f"events with M: {len(df)}", flush=True)

    g = mc_grid(df, a.n, a.rmax, a.nboot, a.step, a.nproc)
    g.to_csv(OUT / "mc_grid.csv", index=False, float_format="%.4f")
    print("grid all:", g.mc_best.notna().sum(), "nodes", flush=True)

    dq = load_catalog(qc=True)
    gq = mc_grid(dq, a.n, a.rmax, a.nboot, a.step, a.nproc)
    gq.to_csv(OUT / "mc_grid_qc.csv", index=False, float_format="%.4f")
    print("grid qc:", gq.mc_best.notna().sum(), "nodes", flush=True)
    if SEL:
        meta |= {k: SEL[k] for k in ("selection", "selection_qc")}
        print("selection:", json.dumps(SEL["selection"]), flush=True)

    dsets = []
    summ = [node_summary(g, "all")]
    for per, (y0, y1) in PERIODS.items():
        gp = mc_grid(df[df.year.between(y0, y1)].reset_index(drop=True), a.n_period, a.rmax,
                     a.nboot, a.step, a.nproc)
        gp.to_csv(OUT / f"mc_grid_{per}.csv", index=False, float_format="%.4f")
        print(f"grid {per}:", gp.mc_best.notna().sum(), "nodes", flush=True)
        dsets.append(to_xr(gp).expand_dims(period=[per]))
        summ.append(node_summary(gp, per))
    if a.yearly:
        for y in range(2010, 2017):
            gp = mc_grid(df[df.year == y].reset_index(drop=True), a.n_period, a.rmax,
                         a.nboot, a.step, a.nproc)
            gp.to_csv(OUT / f"mc_grid_{y}.csv", index=False, float_format="%.4f")
            print(f"grid {y}:", gp.mc_best.notna().sum(), "nodes", flush=True)

    ds = to_xr(g)
    ds["mc_best_qc"] = to_xr(gq)["mc_best"]
    ds.attrs.update(title=f"Cascadia catalog {LABEL} magnitude of completeness",
                    source="workflow/06_analysis/v4_mc_maps.py", params=json.dumps(meta))
    ds.to_netcdf(OUT / "mc_grid.nc")
    dp = xr.concat(dsets, dim="period")
    dp.attrs.update(title=f"Cascadia catalog {LABEL} Mc per period",
                    source="workflow/06_analysis/v4_mc_maps.py", params=json.dumps(meta))
    dp.to_netcdf(OUT / "mc_grid_periods.nc")

    pd.concat(summ).to_csv(OUT / "mc_nodes_by_region.csv", index=False, float_format="%.3f")
    rt = regional_table(df, a.nboot)
    rq = regional_table(dq, a.nboot)
    rq["variant"] = "qc"
    rt["variant"] = "all"
    pd.concat([rt, rq]).to_csv(OUT / "mc_regional.csv", index=False, float_format="%.4f")
    (OUT / "run_params.json").write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------- figures
OKABE = {"maxc": "#E69F00", "mbs": "#0072B2", "gft": "#009E73", "zbz": "#CC79A7"}


def _stations(period=None):
    s = pd.read_csv(STA)
    s["t0"] = pd.to_datetime(s.epoch_start, errors="coerce")
    s["t1"] = pd.to_datetime(s.epoch_end, errors="coerce").fillna(pd.Timestamp("2100-01-01"))
    if period is not None:
        y0, y1 = PERIODS[period]
        s = s[(s.t0 <= pd.Timestamp(f"{y1}-12-31")) & (s.t1 >= pd.Timestamp(f"{y0}-01-01"))]
    s = s.drop_duplicates(["network", "station"])
    return s


def _grid_da(g, col):
    import xarray as xr
    da = g.set_index(["lat", "lon"])[col].to_xarray().sortby("lat").sortby("lon")
    return da.astype("float32")


def _map_panel(fig, da, cpt, title, stations=None, frame_left=True, frame_bottom=True):
    import map_context as mc
    reg = REGION_MAP
    proj = "M5.2c"
    fl = "W" if frame_left else "w"
    fb = "S" if frame_bottom else "s"
    fig.basemap(region=reg, projection=proj, frame=[f"{fl}{fb}en", "xa4f1", "ya2f1"])
    fig.coast(land="gray92", water="white", shorelines="0.25p,gray40", resolution="i")
    fig.grdimage(da, cmap=cpt, nan_transparent=True)
    fig.coast(shorelines="0.25p,gray30", resolution="i", borders=["1/0.2p,gray50", "2/0.15p,gray60"])
    mc.draw_boundaries(fig, reg, pen_scale=0.6)
    if stations is not None and len(stations):
        sf = stations[stations.seafloor.astype(str) == "True"]
        ld = stations[stations.seafloor.astype(str) != "True"]
        fig.plot(x=ld.lon, y=ld.lat, style="t0.09c", fill="gray20", pen="0.1p,white")
        fig.plot(x=sf.lon, y=sf.lat, style="i0.10c", fill="white", pen="0.25p,black")
    fig.text(position="TL", text=title, font="7p,Helvetica-Bold,black", offset="0.1c/-0.1c",
             fill="white@20", clearance="0.05c/0.05c")


def plot(a):
    import pygmt
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import map_context as mc
    mc.PB2002 = str(ROOT / "data/tectonics/PB2002_steps.json")
    pygmt.config(FONT_ANNOT_PRIMARY="6p", FONT_LABEL="7p", MAP_FRAME_TYPE="plain",
                 MAP_TICK_LENGTH_PRIMARY="0.06c", MAP_FRAME_PEN="0.5p", FORMAT_GEO_MAP="ddd")
    g = pd.read_csv(OUT / "mc_grid.csv")
    sta = _stations()
    lo, hi = 0.0, 3.0
    cpt = OUT / "mc.cpt"
    pygmt.makecpt(cmap="batlow", series=[lo, hi, 0.1], continuous=True, output=str(cpt))

    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"{name}.{ext}", dpi=300)
        print("wrote", name)

    # (1) three methods
    fig = pygmt.Figure()
    g = g.assign(mc_gftb=g.mc_best)
    panels = [("mc_maxc", "a) MAXC + 0.2"), ("mc_mbs", "b) MBS"),
              ("mc_best", "c) GFT 95/90 (W&W 2005 rule)")]
    for i, (col, t) in enumerate(panels):
        if i:
            fig.shift_origin(xshift="5.5c")
        _map_panel(fig, _grid_da(g, col), str(cpt), t, sta, frame_left=(i == 0))
    fig.colorbar(cmap=str(cpt), position="JBC+w8c/0.22c+h+o-5.5c/0.7c",
                 frame=["xa0.5f0.1+lMc (Route A M)"])
    save(fig, "fig1_mc_three_methods")

    # (2) bootstrap sigma and b
    fig = pygmt.Figure()
    scpt = OUT / "sigma.cpt"
    pygmt.makecpt(cmap="lajolla", series=[0, 0.5, 0.05], continuous=True, reverse=True, output=str(scpt))
    _map_panel(fig, _grid_da(g, "sigma_boot"), str(scpt), "a) bootstrap sigma of Mc", sta)
    fig.colorbar(cmap=str(scpt), position="JBC+w4.5c/0.2c+h+o0/0.7c",
                 frame=["xa0.1f0.05+lsigma(Mc)"])
    fig.shift_origin(xshift="6.2c")
    bcpt = OUT / "b.cpt"
    pygmt.makecpt(cmap="vik", series=[0.6, 1.4, 0.05], continuous=True, output=str(bcpt))
    _map_panel(fig, _grid_da(g, "b"), str(bcpt), "b) b-value above Mc", sta, frame_left=False)
    fig.colorbar(cmap=str(bcpt), position="JBC+w4.5c/0.2c+h+o0/0.7c", frame=["xa0.2f0.1+lb"])
    save(fig, "fig2_sigma_b")

    # (3) periods + difference
    fig = pygmt.Figure()
    gps = {p: pd.read_csv(OUT / f"mc_grid_{p}.csv") for p in PERIODS}
    for i, p in enumerate(PERIODS):
        if i:
            fig.shift_origin(xshift="5.5c")
        _map_panel(fig, _grid_da(gps[p], "mc_best"), str(cpt), f"{p}", _stations(p),
                   frame_left=(i == 0))
    fig.colorbar(cmap=str(cpt), position="JBC+w8c/0.22c+h+o-8.25c/0.7c",
                 frame=["xa0.5f0.1+lMc best (GFT95/90, else MAXC)"])
    stack = np.stack([gps[p].mc_best.values for p in PERIODS])
    cnt = np.isfinite(stack).sum(0)
    with np.errstate(invalid="ignore"):
        diff = np.where(cnt >= 2, np.nanmax(stack, 0) - np.nanmin(stack, 0), np.nan)
    gd = gps["2010"][["lon", "lat"]].assign(diff=diff)
    dcpt = OUT / "diff.cpt"
    pygmt.makecpt(cmap="bilbao", series=[0, 1.0, 0.1], continuous=True, reverse=True, output=str(dcpt))
    fig.shift_origin(xshift="5.9c")
    _map_panel(fig, _grid_da(gd, "diff"), str(dcpt), "worst - best period", sta,
               frame_left=False)
    fig.colorbar(cmap=str(dcpt), position="JBC+w4.5c/0.2c+h+o0/0.7c",
                 frame=["xa0.2f0.1+lMc(worst) - Mc(best)"])
    save(fig, "fig3_mc_periods")
    gd.to_csv(OUT / "mc_grid_period_range.csv", index=False, float_format="%.3f")

    # (5) QC variant minus all
    gq = pd.read_csv(OUT / "mc_grid_qc.csv")
    gdq = g[["lon", "lat"]].assign(d=gq.mc_best.values - g.mc_best.values)
    qcpt = OUT / "qcdiff.cpt"
    pygmt.makecpt(cmap="vik", series=[-0.6, 0.6, 0.1], continuous=True, output=str(qcpt))
    fig = pygmt.Figure()
    _map_panel(fig, _grid_da(gdq, "d"), str(qcpt), "Mc(QC 4P+4S) - Mc(all)", sta)
    fig.colorbar(cmap=str(qcpt), position="JBC+w4.5c/0.2c+h+o0/0.7c", frame=["xa0.2f0.1+ldelta Mc"])
    save(fig, "fig5_qc_minus_all")

    plot_fmd(a)


def plot_fmd(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "font.family": "DejaVu Sans"})
    df = load_catalog(qc=False)
    rt = pd.read_csv(OUT / "mc_regional.csv")
    rt = rt[(rt.variant == "all") & (rt.period == "all")].set_index("region")
    zbz = pd.read_csv(ZBZ).set_index("region")
    fig, axs = plt.subplots(3, 3, figsize=(7.2, 6.6), sharex=True)
    for ax, reg in zip(axs.ravel(), REG_ORDER):
        d = df[df.region == reg]
        h = np.bincount(d.ibin.values, minlength=len(MBINS))
        cum = np.cumsum(h[::-1])[::-1]
        nz = h > 0
        ax.semilogy(MBINS[cum > 0], cum[cum > 0], "o", ms=2.2, color="k", label="cumulative")
        ax.semilogy(MBINS[nz], h[nz], "o", ms=2.2, mfc="white", mec="0.45", mew=0.5,
                    label="per 0.1 bin")
        r = rt.loc[reg]
        for key, col, lab in [("mc_maxc", OKABE["maxc"], "MAXC+0.2"), ("mc_mbs", OKABE["mbs"], "MBS"),
                              ("mc_best", OKABE["gft"], "GFT (best)")]:
            if np.isfinite(r[key]):
                ax.axvline(r[key], color=col, lw=1.1, ls="-" if key != "mc_mbs" else "--",
                           label=f"{lab} {r[key]:.1f}")
        if reg in zbz.index:
            ax.axvline(zbz.loc[reg, "Mc"], color=OKABE["zbz"], lw=1.0, ls=":",
                       label=f"clusters_zbz {zbz.loc[reg, 'Mc']:.2f}")
        if np.isfinite(r["b"]):
            mm = np.linspace(r["mc_best"], MBINS[cum > 0].max(), 20)
            aval = np.log10(r["n_above"]) + r["b"] * r["mc_best"]
            ax.semilogy(mm, 10 ** (aval - r["b"] * mm), "-", color=OKABE["gft"], lw=0.6, alpha=0.7)
        ax.set_title(f"{reg}  {REGION_NAMES[reg]}\nN={len(d)}, b={r['b']:.2f}", fontsize=6.5)
        ax.legend(fontsize=5.2, frameon=True, framealpha=0.9, edgecolor="none", loc="lower left", handlelength=1.5)
        ax.set_xlim(-1.5, 5.5)
        ax.set_ylim(0.8, None)
        ax.grid(alpha=0.2, which="major")
    for ax in axs[-1]:
        ax.set_xlabel("M (Route A)")
    for ax in axs[:, 0]:
        ax.set_ylabel("Number of events")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig4_regional_fmd.{ext}", dpi=300)
    print("wrote fig4_regional_fmd")


def main():
    global MAG, OUT, ZBZ, BUNDLE, LABEL, FLAGS
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", choices=["compute", "plot", "all"], default="all")
    p.add_argument("--n", type=int, default=250, help="events per node (all-data map)")
    p.add_argument("--n-period", type=int, default=150, help="events per node (period maps)")
    p.add_argument("--rmax", type=float, default=50.0, help="maximum node radius, km")
    p.add_argument("--nboot", type=int, default=200)
    p.add_argument("--step", type=float, default=0.1, help="grid step, deg")
    p.add_argument("--nproc", type=int, default=32)
    p.add_argument("--yearly", action="store_true", help="also write per-year grids")
    mag_paths.add_argument(p, "v4.1")
    p.add_argument("--out", type=Path, default=None, help=f"output folder (default {OUT.relative_to(ROOT)})")
    p.add_argument("--zbz", type=Path, default=None,
                   help=f"clustering Mc table for fig4 (default {ZBZ.relative_to(ROOT)})")
    p.add_argument("--catalog", type=Path, default=None,
                   help="catalog bundle CSV with its own hypocentres and magnitudes (v5.1/v6.0 schema: ID, time, lat, "
                        "lon, M); replaces the v4 catalog and --mag (D-B102, D-B104)")
    p.add_argument("--mag-col", default="M", help="magnitude column of --catalog (default M)")
    p.add_argument("--picks", type=Path, default=None,
                   help="picks file of --catalog (event_ID, phase): P and S counts for the QC variant (4 P + 4 S)")
    p.add_argument("--exclude-flagged", action="store_true",
                   help="drop events flagged as not earthquakes (event_flag set: blasts, explosions; D-B105): the "
                        "catalog's own event_flag, or the IDs flagged in --flags")
    p.add_argument("--flags", type=Path, nargs="+", default=None,
                   help="files with ID and event_flag columns (e.g. the v5.1 and v6.0 catalogs); with --exclude-flagged "
                        "the IDs flagged in any of them are dropped; needed for the v4 catalog, which has no flag column")
    p.add_argument("--label", default=None, help="catalog name in titles (default: from the --catalog file name)")
    a = p.parse_args()
    if a.catalog:
        if a.mag:
            p.error("--catalog uses the catalog's own magnitudes (--mag-col); --mag applies to the v4 catalog only")
        if a.stage != "plot" and not a.picks:
            p.error("--catalog needs --picks (the catalog's picks file) for the QC variant")
        BUNDLE = dict(catalog=a.catalog, mag_col=a.mag_col, picks=a.picks)
        LABEL = a.label or a.catalog.stem.split("_")[-1]
    elif a.label:
        LABEL = a.label
    if a.flags and not a.exclude_flagged:
        p.error("--flags goes with --exclude-flagged")
    if a.exclude_flagged:
        if not a.catalog and not a.flags:
            p.error("--exclude-flagged on the v4 catalog needs --flags (it has no event_flag column)")
        FLAGS = dict(files=a.flags or [], ids=flagged_ids(a.flags) if a.flags else None)
        print(f"excluding flagged events: {'IDs flagged in ' + ', '.join(map(str, a.flags)) if a.flags else 'own event_flag'}"
              + (f" ({len(FLAGS['ids']):,} IDs)" if a.flags else ""), flush=True)
    MAG = mag_paths.v41(a.mag)
    OUT = a.out or OUT
    ZBZ = a.zbz or ZBZ
    if BUNDLE:
        print(f"catalog {LABEL}: {a.catalog} (magnitudes: column {a.mag_col}; picks: {a.picks})", flush=True)
    else:
        print(f"magnitudes: {MAG}", flush=True)
    if a.stage in ("compute", "all"):
        compute(a)
    if a.stage in ("plot", "all"):
        plot(a)


if __name__ == "__main__":
    main()

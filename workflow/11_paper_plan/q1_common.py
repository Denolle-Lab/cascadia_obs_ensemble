"""Q1 slip style (D-B115): shared inputs, event sets, units and metrics.

Question: what sets how a plate boundary releases slip, in swarms or in mainshock-aftershock sequences? The
pre-registration (hypotheses, metrics, covariates, tests, decision rules; written before any covariate correlation)
is data/paper_plan/q1_slip_style/RESUME.md. This module reads; the q1_*.py scripts write.

Inputs (read only): catalog v6.1 (and v6.0 for the stability control), its nearest-neighbour clustering
(v51_clusters_zbz.py --catalog v6.x: events_zbz_<tag>.csv.gz, families_zbz_<tag>.csv, variants/), the Mc maps without
non-earthquakes, the plate-boundary segment rule of pb_common.py (D-B113 axes; the segment join is recomputed here
from the catalog every run, so a new axis file or rule only needs a rerun), the per-event covariates of
q1_covariates.py (interim) or of the D-B114 job.

Event set (pre-registered): located earthquakes (event_flag empty), tiers A-C, not in the waveform-review set P1,
in the analysis set of the clustering (M >= regional Mc of that run).

Environment: Q1_OUT (small tables, default data/paper_plan/q1_slip_style), Q1_BIG (report, figures, large tables,
default /wd1/mdenolle_data/paper_plan/q1_slip_style), PB_AXES (gmrt = D-B113 axes, default; pb2002 = old rule).
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import pb_common as P  # noqa: E402  (segment rule, geometry; PB_* catalog variables are not used here)

WD1 = Path("/wd1/mdenolle_data")
OUT = Path(os.environ.get("Q1_OUT", ROOT / "data/paper_plan/q1_slip_style"))
BIG = Path(os.environ.get("Q1_BIG", WD1 / "paper_plan/q1_slip_style"))
SEED = 20261127

CATS = {
    "v61": dict(label="v6.1", csv=WD1 / "catalog_v6/v6.1/cascadia_catalog_v6.1.csv",
                clu=WD1 / "catalog_v6/v6.1_analysis/clusters", mcmap=WD1 / "magnitude_converged/mc_maps_v61",
                segpq=WD1 / "paper_plan/v6.1/events_segments_v61.parquet"),
    "v60": dict(label="v6.0", csv=WD1 / "catalog_v6/v6.0/cascadia_catalog_v6.0.csv",
                clu=WD1 / "catalog_v6/v6.0_analysis/clusters", mcmap=WD1 / "magnitude_converged/mc_maps_v60",
                segpq=None),
}
VARIANTS = ["d3", "d3q", "df1.3", "df2.0", "df_est", "eta0_shuf5", "mc+0.3", "mc_best", "mc_year", "noM_floor",
            "storedM", "tierABC"]

# pre-registered constants (RESUME.md)
MC_U = 2.15                    # uniform Mc: largest regional Mc of the v6.1 clustering (GO)
CELL_DEG = 0.2
CELL_LON0, CELL_LAT0 = -132.0, 38.0
NMIN_CELL = 50
NMIN_SEG = 100
HOT = ["AX", "JR", "EN", "EXR", "GR", "BL", "SOV", "MSH"]
COLD = ["GP", "MFZ", "MTJ", "GS"]
CI_WINDOW = ("2011-10-01", "2015-09-30")      # Cascadia Initiative OBS years (control)
NB = 500


def log(*a):
    print(*a, flush=True)


def md5(path, n=1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(n), b""):
            h.update(b)
    return h.hexdigest()


def tobool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().eq("true")


# ------------------------------------------------------------------ catalog and segments
CAT_COLS = ["ID", "time", "lat", "lon", "depth", "loc_source", "tier", "region", "sig_z", "depth_quality",
            "event_flag", "floor_limited", "far_only", "deep_pass", "surface_km", "M", "magnitude_source",
            "review_problems", "dist_min", "n_sta", "gap"]


def load_catalog(tag: str = "v61") -> pd.DataFrame:
    """Located rows of the catalog with the flags of the pre-registered event set and the segment of each event
    (pb_common.assign_segments, axes PB_AXES)."""
    c = pd.read_csv(CATS[tag]["csv"], usecols=CAT_COLS, low_memory=False)
    c = c[c.loc_source.ne("none")].copy()
    for f in ("floor_limited", "far_only", "deep_pass"):
        c[f] = tobool(c[f])
    c["is_eq"] = c.event_flag.isna()
    c["abc"] = c.tier.isin(["A", "B", "C"])
    c["depth_good"] = c.abc & c.depth_quality.eq("ok") & ~c.floor_limited & ~c.far_only & ~c.deep_pass
    c["review_P1"] = c.review_problems.fillna("").str.contains("P1")
    c["t"] = pd.to_datetime(c.time, utc=True, format="ISO8601")
    c["offshore"] = c.surface_km < 0
    c = P.assign_segments(c.reset_index(drop=True))
    return c


def load_zbz(tag: str = "v61", variant: str | None = None):
    """Clustering event table (analysis set and attached events) and family table of the main run or a variant.
    Variant tables carry no parent links (productivity is main-run only); cls is rebuilt as in build()."""
    d = CATS[tag]["clu"]
    if variant in (None, "main"):
        z = pd.read_csv(d / f"events_zbz_{tag}.csv.gz", usecols=["event_id", "region", "in_analysis", "M", "parent_id",
                                                                  "linked", "family_id", "family_type", "role", "cls"],
                        low_memory=False)
        f = pd.read_csv(d / f"families_zbz_{tag}.csv", low_memory=False)
    else:
        z = pd.read_csv(d / "variants" / f"events_{variant}.csv.gz", low_memory=False)
        f = pd.read_csv(d / "variants" / f"families_{variant}.csv", low_memory=False)
    return normalise_zbz(z), f


def normalise_zbz(z: pd.DataFrame) -> pd.DataFrame:
    z = z.rename(columns={"event_id": "ID", "region": "zbz_region", "M": "M_zbz"}).copy()
    z["in_analysis"] = tobool(z.in_analysis) if z.in_analysis.dtype != bool else z.in_analysis
    z["linked"] = tobool(z.linked) if z.linked.dtype != bool else z.linked
    z["family_type"] = z.family_type.fillna("")
    if "cls" not in z.columns:
        z["cls"] = np.where(z.family_id > 0, z.family_type, np.where(z.linked, "linked_to_single", "background"))
    return z


def regional_mc(tag: str = "v61") -> dict:
    r = pd.read_csv(CATS[tag]["clu"] / "regions_zbz.csv")
    return dict(zip(r.region, r.Mc))


def mc_map_values(lon, lat, tag="v61", period=None) -> np.ndarray:
    """mc_best of the 0.1 degree Mc map node nearest to each epicentre (v6.x map without non-earthquakes)."""
    f = CATS[tag]["mcmap"] / ("mc_grid.csv" if period is None else f"mc_grid_{period}.csv")
    g = pd.read_csv(f)
    g = g[np.isfinite(g.mc_best)]
    key = lambda lo, la: np.round(np.asarray(lo) * 10).astype(int) * 100000 + np.round(np.asarray(la) * 10).astype(int)  # noqa: E731
    s = pd.Series(g.mc_best.values, index=key(g.lon.values, g.lat.values))
    s = s[~s.index.duplicated()]
    return pd.Series(key(lon, lat)).map(s).values


def add_cells(e: pd.DataFrame, deg: float = CELL_DEG, col: str = "cell") -> pd.DataFrame:
    ix = np.floor((e.lon.values - CELL_LON0) / deg).astype(int)
    iy = np.floor((e.lat.values - CELL_LAT0) / deg).astype(int)
    e[col] = [f"{a}_{b}" for a, b in zip(ix, iy)]
    e[col + "_lon"] = CELL_LON0 + (ix + 0.5) * deg
    e[col + "_lat"] = CELL_LAT0 + (iy + 0.5) * deg
    return e


def event_table(tag: str = "v61", variant: str | None = None, cat: pd.DataFrame | None = None,
                zbz: tuple | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All located rows of the catalog joined with the clustering (main or variant). Returns (events, families)."""
    c = load_catalog(tag) if cat is None else cat
    z, f = load_zbz(tag, variant) if zbz is None else zbz
    e = c.merge(z, on="ID", how="left", validate="1:1")
    e["in_analysis"] = e.in_analysis.fillna(False).astype(bool)
    e["event_set"] = e.is_eq & e.abc & ~e.review_P1 & e.in_analysis
    e = add_cells(e)
    e = add_cells(e, 0.4, "cell04")
    return e, f


# ------------------------------------------------------------------ metrics
def mc_maxc(m, corr=0.2):
    m = m[np.isfinite(m)]
    if len(m) < 50:
        return np.nan
    h, ed = np.histogram(m, bins=np.arange(-2, 8, 0.1))
    return float(ed[np.argmax(h)] + 0.05 + corr)


def b_aki(m, mc, nmin=50):
    m = m[np.isfinite(m) & (m >= mc - 1e-9)]
    if len(m) < nmin:
        return np.nan, len(m)
    return float(1.0 / (np.log(10) * (m.mean() - mc))), len(m)


def ci(a, lo=2.5, hi=97.5):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if len(a) < 20:
        return np.nan, np.nan
    return float(np.percentile(a, lo)), float(np.percentile(a, hi))


def family_units(e: pd.DataFrame, unit: str) -> pd.Series:
    """Unit of each family = the unit of most of its event-set members."""
    g = e[e.event_set & (e.family_id > 0)].groupby("family_id")[unit]
    return g.agg(lambda s: s.value_counts().index[0])


def offspring_counts(e: pd.DataFrame) -> pd.Series:
    """Direct offspring per event: linked analysis-set children (main run only; needs parent_id)."""
    if "parent_id" not in e.columns:
        return pd.Series(dtype=float)
    ch = e[e.in_analysis & e.linked & (e.parent_id >= 0)]
    return ch.groupby("parent_id").size()


def unit_metrics(e: pd.DataFrame, fam: pd.DataFrame, unit: str, mcreg: dict, nmin: int,
                 boot: bool = True, rng=None, mc_count: float | None = None, full: bool = True) -> pd.DataFrame:
    """Pre-registered metrics M1-M7 per unit. e: event table (event_table); the event set is e.event_set.
    mc_count: count only event-set events with M >= mc_count (uniform-Mc control) for M1-M3.
    full: also b, productivity, durations (main run)."""
    rng = rng or np.random.default_rng(SEED)
    E = e[e.event_set].copy()
    if mc_count is not None:
        E = E[E.M >= mc_count - 1e-9]
    fu = family_units(e, unit)
    fam = fam.copy()
    fam["unit"] = fam.family_id.map(fu)
    nfam_col = "n"
    off = offspring_counts(e) if full else pd.Series(dtype=float)
    Eb = e[e.is_eq & e.abc & ~e.review_P1 & e.M.notna()] if full else None
    rows = []
    for u, g in E.groupby(unit):
        n = len(g)
        if n < nmin:
            continue
        cls = g.cls.fillna("background").values
        r = {unit: u, "n": n}
        r["swarm_share"] = 100 * float(np.mean(cls == "swarm"))
        r["bg_frac"] = 100 * float(np.mean(cls == "background"))
        r["ma_share"] = 100 * float(np.mean(cls == "mainshock-aftershock"))
        r["multiplet_share"] = 100 * float(np.mean(cls == "multiplet"))
        clu = cls[cls != "background"]
        r["swarm_of_clustered"] = 100 * float(np.mean(clu == "swarm")) if len(clu) else np.nan
        f = fam[(fam.unit == u) & (fam[nfam_col] >= 5)]
        nsw, nma = int((f.type == "swarm").sum()), int((f.type == "mainshock-aftershock").sum())
        r["n_fam_swarm5"], r["n_fam_ma5"] = nsw, nma
        r["fam_mix"] = nsw / (nsw + nma) if (nsw + nma) >= 3 else np.nan
        if boot:
            fid = g.family_id.fillna(-1).values
            unitk = np.where(fid > 0, fid, -1 - np.arange(n))
            uu, inv = np.unique(unitk, return_inverse=True)
            nn = np.bincount(inv).astype(float)
            sw = np.bincount(inv, weights=(cls == "swarm").astype(float))
            bs = []
            for _ in range(NB):
                k = rng.integers(0, len(uu), len(uu))
                bs.append(100 * sw[k].sum() / nn[k].sum())
            r["swarm_share_lo"], r["swarm_share_hi"] = ci(bs)
        if full:
            # b: all tier A-C earthquakes of the unit with M (any M), Mc by maximum curvature + 0.2
            m = Eb.loc[Eb[unit] == u, "M"].values
            mc = mc_maxc(m)
            b, nb = b_aki(m, mc) if np.isfinite(mc) else (np.nan, 0)
            r.update(Mc_unit=mc, b=b, n_b=nb)
            if np.isfinite(b) and boot:
                bb = []
                for _ in range(NB):
                    mm = rng.choice(m, len(m))
                    q = mc_maxc(mm)
                    bb.append(b_aki(mm, q)[0] if np.isfinite(q) else np.nan)
                r["b_lo"], r["b_hi"] = ci(bb)
            # productivity: parents with M - Mc_region in [1, 2)
            mcr = g.zbz_region.map(mcreg).values
            dm = g.M_zbz.values - mcr
            par = g[(dm >= 1.0) & (dm < 2.0)]
            r["n_parents"] = len(par)
            if len(par) >= 20 and len(off):
                k_ = par.ID.map(off).fillna(0).values
                r["productivity"] = float(k_.mean())
                if boot:
                    pb = [k_[rng.integers(0, len(k_), len(k_))].mean() for _ in range(NB)]
                    r["productivity_lo"], r["productivity_hi"] = ci(pb)
            else:
                r["productivity"] = np.nan
            fs = fam[(fam.unit == u) & (fam.type == "swarm") & (fam[nfam_col] >= 5)]
            dcol = "duration_all_days" if "duration_all_days" in fs.columns else "duration_days"
            r["swarm_dur_med_days"] = float(fs[dcol].median()) if len(fs) else np.nan
            f10 = fam[(fam.unit == u) & (fam.type == "swarm") & (fam[nfam_col] >= 10)]
            if len(f10):
                mig = tobool(f10.get("migrating", pd.Series(False, index=f10.index)))
                if "mig_migrating" in f10.columns:
                    mig = mig | tobool(f10.mig_migrating)
                r["n_swarm10"], r["swarm10_migrating"] = len(f10), int(mig.sum())
                r["D90_med"] = float(f10.D90_m2s.median()) if "D90_m2s" in f10.columns else np.nan
            else:
                r["n_swarm10"], r["swarm10_migrating"] = 0, 0
        # unit descriptors
        r["seg_type_major"] = g.seg_type.value_counts().index[0]
        r["seg_major"] = g.seg.value_counts().index[0]
        r["seg_purity"] = float(g.seg.value_counts(normalize=True).iloc[0])
        r["offshore_frac"] = float(g.offshore.mean())
        r["lon"], r["lat"] = float(g.lon.median()), float(g.lat.median())
        r["dist_min_med"] = float(g.dist_min.median())
        r["mc_map_med"] = float(np.nanmedian(g.mc_map)) if "mc_map" in g.columns and g.mc_map.notna().any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ covariates
COVS = ["age_myr", "heatflow_mw", "z600_km", "rate_mmyr", "rough_m", "sed_m", "Tslab_C"]
COV_LABEL = {"age_myr": "plate age (Myr)", "heatflow_mw": "heat flow (mW/m2)", "z600_km": "600 C isotherm depth (km)",
             "rate_mmyr": "spreading rate (mm/yr)", "rough_m": "seafloor roughness (m)", "sed_m": "sediment (m)",
             "Tslab_C": "slab-surface temperature (C)",
             "vpvs_fam": "family Vp/Vs"}
PRED_SIGN = {"age_myr": -1, "heatflow_mw": +1, "z600_km": -1, "rate_mmyr": +1, "rough_m": 0, "sed_m": 0, "Tslab_C": 0,
             "vpvs_fam": +1}


def covariate_file(tag: str = "v61") -> Path:
    return BIG / "covariates" / f"covariates_interim_{tag}.parquet"


EXTERNAL_TABLE = Path(os.environ.get("Q1_EXTERNAL", WD1 / "external_geo/derived/events_external_geo_v61.parquet"))


def external_event_table() -> Path | None:
    """The D-B114 per-event covariate table (workflow/12_external_geo/eg_build.py; provenance
    /wd1/mdenolle_data/external_geo/PROVENANCE.csv), if it exists."""
    return EXTERNAL_TABLE if EXTERNAL_TABLE.exists() else None


def unit_covariates(e: pd.DataFrame, cov: pd.DataFrame, unit: str, cols=COVS, minfrac=0.5) -> pd.DataFrame:
    """Median of each per-event covariate over the unit's event-set events (NaN when < minfrac carry a value)."""
    E = e[e.event_set][["ID", unit]].merge(cov, on="ID", how="left")
    out = {}
    for c_ in cols:
        if c_ not in E.columns:
            continue
        g = E.groupby(unit)[c_]
        med, frac = g.median(), g.apply(lambda s: s.notna().mean())
        out[c_] = med.where(frac >= minfrac)
    return pd.DataFrame(out).reset_index()

"""Shared loaders and statistics for the v5.1 structure-seismicity analysis (v51_struct.py).

Inputs (read only):
* catalog v5.1 (/wd1/mdenolle_data/catalog_v5/v5.1/cascadia_catalog_v5.1.csv), the v5.1 magnitudes (mag piece),
  the v5.1 nearest-neighbour clustering (clusters piece) and the fault surfaces v1 (faults piece);
* velocity model v0.3.1 (netCDF). Only nodes that pass the checkerboard tests are used: resolution_p > 0 for Vp,
  resolution_s > 0 and vs_reverted == 0 for Vs, both for Vp/Vs (D-B018, D-B049, D-B051). The D-B073 shallow-crust
  rule is not in v0.3.1 and is never applied here. The `medium` flag of v0.3.2 (rock/water/air against the merged
  solid surface, D-B068) is read only to check that no resolved node lies in water or air.

Conventions: depth km below sea level, positive down. "Coverage-matched anomaly" V_res of a node variable is V
minus the mean of V over the resolved nodes of the same stratum (zone x depth node x DWS tercile), so it removes
what the ray coverage and the depth alone predict.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CAT = Path("/wd1/mdenolle_data/catalog_v5/v5.1/cascadia_catalog_v5.1.csv")
MAG = Path("/wd1/mdenolle_data/catalog_v5/v5.1_analysis/mag/cascadia_catalog_v5.1_magnitudes.csv")
CLU = Path("/home/mdenolle/wt_a51_clusters/data/catalog_v5/v5.1_analysis/clusters")
FLT = Path("/home/mdenolle/wt_a51_faults/data/catalog_v5/v5.1_analysis/faults")
FLT_BIG = Path("/wd1/mdenolle_data/catalog_v5/v5.1_analysis/faults")
MODELS = Path("/wd1/mdenolle_data/tomo_v1/models/cascadia_obs_tomo")
MODEL = MODELS / "v0.3.1.nc"
MODEL_MEDIUM = MODELS / "v0.3.2.nc"          # only its `medium` flag (rock / water / air) is read
SLAB = Path("/home/mdenolle/cascadia_obs_ensemble/data/slab2/cas_slab2_dep.xyz")
TOMO_RUNS = Path("/wd1/mdenolle_data/tomo_v1/b2_tomodd")
# waveform review of four v5.1 location problems (P1 Mount St Helens deep events, P2 Brookings 2012, P3 tensor
# events, P4 deep_pass / far_only); read only to flag events, never to change them
REVIEW = Path("/wd1/mdenolle_data/catalog_v5/v5.1_review/review_v51_events.csv")
EXT_MODELS = Path("/home/mdenolle/cascadia_obs_ensemble/data/tomography")
V4_CAT = Path("/home/mdenolle/cascadia_obs_ensemble/data/catalog_v4")
OUT = ROOT / "data/catalog_v5/v5.1_analysis/struct"
BIG = Path("/wd1/mdenolle_data/catalog_v5/v5.1_analysis/struct")

KM = 111.195
BLOCK_KM = 40.0          # bootstrap block = the finest resolved checkerboard cell (D-B018)
LON0, LAT0 = -125.0, 44.0
COSREF = np.cos(np.radians(44.5))
DLON, DLAT = 0.125, 0.1

ZONES = {"BL": "Blanco transform (15, 18, 29 km)",
         "MTJ": "Mendocino triple junction, Gorda slab (29 km)",
         "WA": "SW Washington Cascades and Puget forearc (1-18 km)"}

# Blanco segments: longitude boxes (lat 42.4-45.0) consistent with the literature names that the fault surfaces v1
# carry (v1_BL001/002/014 West Blanco Depression; BL003/007 Surveyor; BL004/006/010 Cascadia Depression;
# BL005/009/015/017 Blanco Ridge; BL008/011/013 Gorda Depression).
BLANCO_SEGMENTS = [("WBD", "West Blanco Depression", -130.6, -129.45),
                   ("SD", "Surveyor Depression", -129.45, -128.95),
                   ("CD", "Cascadia Depression", -128.95, -128.5),
                   ("BR", "Blanco Ridge", -128.5, -127.45),
                   ("GD", "Gorda Depression", -127.45, -126.4)]

RNG_SEED = 20261125


def log(*a):
    print(*a, flush=True)


def _bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().eq("true")


# ---------------------------------------------------------------- model
def zone_of(lon, lat) -> np.ndarray:
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    z = np.full(lon.shape, "other", dtype=object)
    z[(lon <= -126.4) & (lat >= 42.4) & (lat <= 45.0)] = "BL"
    z[(lat < 42.0) & (lon > -126.4)] = "MTJ"
    z[(lon > -124.0) & (lat >= 45.0)] = "WA"
    return z


def block_of(lon, lat) -> np.ndarray:
    x = (np.asarray(lon, float) - LON0) * KM * COSREF
    y = (np.asarray(lat, float) - LAT0) * KM
    return (np.floor(x / BLOCK_KM).astype(int) * 1000 + np.floor(y / BLOCK_KM).astype(int))


class Model:
    """v0.3.1 on its node grid with the resolution masks and per-node strata."""

    def __init__(self, path: Path = MODEL):
        import xarray as xr
        ds = xr.open_dataset(path)
        self.ds = ds
        self.lon = ds.longitude.values
        self.lat = ds.latitude.values
        self.dep = ds.depth.values
        g = lambda v: ds[v].values.astype(float)            # noqa: E731  [depth, lat, lon]
        self.Vp, self.Vs, self.VpVs = g("Vp"), g("Vs"), g("VpVs")
        self.Vp0, self.Vs0 = g("Vp_start"), g("Vs_start")
        self.dVp, self.dVs = g("Vp_perturbation"), g("Vs_perturbation")
        self.dVpVs = 100.0 * (self.VpVs / (self.Vp0 / self.Vs0) - 1.0)
        self.dws_p, self.dws_s = g("dws_p"), g("dws_s")
        self.res_p, self.res_s = g("resolution_p"), g("resolution_s")
        self.vs_rev = ds["vs_reverted"].values
        medium = None
        try:
            medium = xr.open_dataset(MODEL_MEDIUM)["medium"].values
        except Exception as e:  # noqa: BLE001
            log("medium flag not read:", e)
        self.medium = medium
        rock = np.ones_like(self.Vp, bool) if medium is None else (medium == 0)
        self.mP = (self.res_p > 0) & rock
        self.mS = (self.res_s > 0) & (self.vs_rev == 0) & rock
        self.mPS = self.mP & self.mS
        self.n_res_not_rock = int(((self.res_p > 0) & ~rock).sum() + ((self.res_s > 0) & ~rock).sum())
        nz, ny, nx = self.Vp.shape
        self.shape = (nz, ny, nx)
        LA, LO = np.meshgrid(self.lat, self.lon, indexing="ij")
        self.zone2 = zone_of(LO, LA)                       # [lat, lon]
        self.block2 = block_of(LO, LA)
        self.LO, self.LA = LO, LA
        self.strata()

    def strata(self):
        """Stratum id per node: zone x depth node x DWS tercile (P DWS for P variables, S DWS for S variables)."""
        nz, ny, nx = self.shape
        zone3 = np.broadcast_to(self.zone2, self.shape)
        self.zone3 = zone3
        for ph, mask, dws in (("P", self.mP, self.dws_p), ("S", self.mPS, self.dws_s)):
            bin_ = np.full(self.shape, -1, int)
            for k in range(nz):
                for zn in ZONES:
                    sel = mask[k] & (self.zone2 == zn)
                    n = int(sel.sum())
                    if n == 0:
                        continue
                    if n < 9:
                        bin_[k][sel] = 0
                        continue
                    lv = np.log10(np.maximum(dws[k][sel], 1e-3))
                    q = np.quantile(lv, [1 / 3, 2 / 3])
                    bin_[k][sel] = np.searchsorted(q, lv, side="right")
            zi = np.vectorize({"BL": 1, "MTJ": 2, "WA": 3, "other": 4}.get)(zone3)
            kk = np.arange(nz)[:, None, None] * np.ones(self.shape, int)
            st = np.where(mask & (bin_ >= 0), zi * 10000 + kk * 10 + bin_, -1)
            setattr(self, f"dwsbin_{ph}", bin_)
            setattr(self, f"stratum_{ph}", st)
        # coverage-matched anomalies
        for v, ph in (("Vp", "P"), ("dVp", "P"), ("Vs", "S"), ("dVs", "S"), ("VpVs", "S"), ("dVpVs", "S")):
            a = getattr(self, v)
            st = getattr(self, f"stratum_{ph}")
            res = np.full(self.shape, np.nan)
            for s in np.unique(st[st >= 0]):
                sel = st == s
                res[sel] = a[sel] - a[sel].mean()
            setattr(self, f"{v}_res", res)

    def smoothed_res(self, var: str, ph: str, sigma_km: float = 20.0) -> np.ndarray:
        """Coverage-matched anomaly of `var` after a Gaussian lateral smoothing (sigma_km) over the resolved nodes
        of the same depth layer and zone: keeps only what varies on the scale of the resolved checkerboard cells."""
        mask = self.mP if ph == "P" else self.mPS
        a = getattr(self, var)
        sm = np.full(self.shape, np.nan)
        x = (self.LO - LON0) * KM * np.cos(np.radians(self.LA))
        y = (self.LA - LAT0) * KM
        for k in range(self.shape[0]):
            for zn in ZONES:
                sel = mask[k] & (self.zone2 == zn)
                if not sel.any():
                    continue
                xs, ys, vs = x[sel], y[sel], a[k][sel]
                d2 = (xs[:, None] - xs[None, :]) ** 2 + (ys[:, None] - ys[None, :]) ** 2
                w = np.exp(-0.5 * d2 / sigma_km ** 2)
                sm[k][sel] = (w @ vs) / w.sum(1)
        st = getattr(self, f"stratum_{ph}")
        res = np.full(self.shape, np.nan)
        for s_ in np.unique(st[st >= 0]):
            m_ = st == s_
            res[m_] = sm[m_] - sm[m_].mean()
        return res

    def node_index(self, lon, lat, depth):
        lon, lat, depth = (np.asarray(x, float) for x in (lon, lat, depth))
        ix = np.round((lon - self.lon[0]) / DLON).astype(int)
        iy = np.round((lat - self.lat[0]) / DLAT).astype(int)
        # nearest depth node (uneven spacing): voxel boundaries at mid-depths
        mids = 0.5 * (self.dep[1:] + self.dep[:-1])
        iz = np.searchsorted(mids, depth)
        inside = (ix >= 0) & (ix < len(self.lon)) & (iy >= 0) & (iy < len(self.lat)) & np.isfinite(depth)
        ix, iy, iz = np.clip(ix, 0, len(self.lon) - 1), np.clip(iy, 0, len(self.lat) - 1), np.clip(iz, 0,
                                                                                                    len(self.dep) - 1)
        return ix, iy, iz, inside

    def corners_resolved(self, lon, lat, depth, mask):
        """True when every corner node of the trilinear cell around the point is resolved (strict variant)."""
        lon, lat, depth = (np.asarray(x, float) for x in (lon, lat, depth))
        fx = (lon - self.lon[0]) / DLON
        fy = (lat - self.lat[0]) / DLAT
        fz = np.interp(depth, self.dep, np.arange(len(self.dep)))
        x0 = np.clip(np.floor(fx).astype(int), 0, len(self.lon) - 2)
        y0 = np.clip(np.floor(fy).astype(int), 0, len(self.lat) - 2)
        z0 = np.clip(np.floor(fz).astype(int), 0, len(self.dep) - 2)
        ok = np.ones(lon.shape, bool)
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    ok &= mask[z0 + dz, y0 + dy, x0 + dx]
        inside = (fx >= 0) & (fx <= len(self.lon) - 1) & (fy >= 0) & (fy <= len(self.lat) - 1)
        return ok & inside

    def node_table(self) -> pd.DataFrame:
        nz, ny, nx = self.shape
        K, J, I = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
        d = pd.DataFrame(dict(iz=K.ravel(), iy=J.ravel(), ix=I.ravel(), depth=self.dep[K.ravel()],
                              lat=self.lat[J.ravel()], lon=self.lon[I.ravel()],
                              zone=self.zone3.ravel(), block=np.broadcast_to(self.block2, self.shape).ravel(),
                              resP=self.mP.ravel(), resS=self.mS.ravel(), resPS=self.mPS.ravel(),
                              res_p_km=self.res_p.ravel(), res_s_km=self.res_s.ravel(),
                              dws_p=self.dws_p.ravel(), dws_s=self.dws_s.ravel(),
                              dwsbin_P=self.dwsbin_P.ravel(), dwsbin_S=self.dwsbin_S.ravel(),
                              stratum_P=self.stratum_P.ravel(), stratum_S=self.stratum_S.ravel()))
        for v in ("Vp", "Vs", "VpVs", "dVp", "dVs", "dVpVs", "Vp0", "Vs0"):
            d[v] = getattr(self, v).ravel()
        for v in ("Vp", "dVp", "Vs", "dVs", "VpVs", "dVpVs"):
            d[f"{v}_res"] = getattr(self, f"{v}_res").ravel()
        if self.medium is not None:
            d["medium"] = self.medium.ravel()
        d["node"] = np.arange(len(d))
        return d

    def gradients(self, var: str, mask) -> np.ndarray:
        """Central-difference gradient (per km) [3, nz, ny, nx] (east, north, down) of a node variable, only where
        the node and both neighbours along the axis are in `mask`; NaN elsewhere."""
        a = getattr(self, var)
        nz, ny, nx = self.shape
        g = np.full((3, nz, ny, nx), np.nan)
        dxk = DLON * KM * np.cos(np.radians(self.lat))[None, :, None]
        dyk = DLAT * KM
        ok = mask[:, :, 1:-1] & mask[:, :, 2:] & mask[:, :, :-2]
        g[0][:, :, 1:-1] = np.where(ok, (a[:, :, 2:] - a[:, :, :-2]) / (2 * dxk), np.nan)
        ok = mask[:, 1:-1] & mask[:, 2:] & mask[:, :-2]
        g[1][:, 1:-1] = np.where(ok, (a[:, 2:] - a[:, :-2]) / (2 * dyk), np.nan)
        ok = mask[1:-1] & mask[2:] & mask[:-2]
        dz = (self.dep[2:] - self.dep[:-2])[:, None, None]
        g[2][1:-1] = np.where(ok, (a[2:] - a[:-2]) / dz, np.nan)
        return g


# ---------------------------------------------------------------- slab
def slab_depth_fn():
    """Slab2 depth (km, positive down) at (lon, lat) by bilinear interpolation on its 0.05 deg grid; NaN off grid."""
    from scipy.interpolate import RegularGridInterpolator
    s = pd.read_csv(SLAB, header=None, names=["lon", "lat", "z"])
    s["lon"] = np.where(s.lon > 180, s.lon - 360, s.lon)
    lo = np.sort(s.lon.unique())
    la = np.sort(s.lat.unique())
    grid = s.pivot_table(index="lat", columns="lon", values="z", dropna=False).reindex(index=la, columns=lo).values
    f = RegularGridInterpolator((la, lo), -grid, bounds_error=False, fill_value=np.nan)
    return lambda lon, lat: f(np.column_stack([np.asarray(lat, float), np.asarray(lon, float)]))


# ---------------------------------------------------------------- events
def load_events(model: Model) -> pd.DataFrame:
    cols = ["ID", "time", "lat", "lon", "depth", "loc_source", "tier", "region", "h_min", "h_max", "sig_z",
            "depth_quality", "event_flag", "floor_limited", "far_only", "deep_pass", "five_phase", "surface_km",
            "v4_lat", "v4_lon", "v4_depth", "dist_min", "gap"]
    c = pd.read_csv(CAT, usecols=cols, low_memory=False)
    m = pd.read_csv(MAG, usecols=["ID", "M", "M_type", "is_earthquake", "ml_caution"])
    c = c.merge(m, on="ID", how="left", validate="1:1")
    c["review"] = review_labels(c.ID)
    for f in ("floor_limited", "far_only", "deep_pass", "five_phase"):
        c[f] = _bool(c[f])
    c["is_earthquake"] = _bool(c["is_earthquake"])
    c["located"] = ~c.loc_source.eq("none")
    c["depth_good"] = (c.tier.isin(["A", "B", "C"]) & c.depth_quality.eq("ok") & ~c.floor_limited & ~c.far_only
                       & ~c.deep_pass & c.located)
    z = pd.read_csv(CLU / "events_zbz_v51.csv.gz",
                    usecols=["event_id", "region", "in_analysis", "family_id", "family_type", "role", "background",
                             "cls"]).rename(columns={"event_id": "ID", "region": "box"})
    c = c.merge(z, on="ID", how="left", validate="1:1")
    reg = pd.read_csv(CLU / "regions_zbz.csv")
    c["Mc_box"] = c.box.map(dict(zip(reg.region, reg.Mc)))
    a = pd.read_csv(FLT / "fault_surfaces_v1_assignment.csv.gz",
                    usecols=["event_id", "assignment", "surface_id", "nearest_surface", "distance_km"]).rename(
        columns={"event_id": "ID", "distance_km": "fault_dist_km"})
    c = c.merge(a, on="ID", how="left", validate="1:1")
    sl = slab_depth_fn()
    c["slab_km"] = np.nan
    loc = c.located.values
    c.loc[loc, "slab_km"] = sl(c.lon.values[loc], c.lat.values[loc])
    c["zone"] = zone_of(c.lon.values, c.lat.values)
    c["block"] = block_of(c.lon.values, c.lat.values)
    c["sigma_h"] = np.sqrt((c.h_min ** 2 + c.h_max ** 2) / 2) / 1.515
    attach_nodes(c, model)
    return c


def review_labels(ids: pd.Series) -> pd.Series:
    """'P1:deep;P4:deep_pass'-style labels of the waveform-review sets an event belongs to ('' when none)."""
    try:
        rv = pd.read_csv(REVIEW, usecols=["ID", "problem", "role"])
    except FileNotFoundError:
        log("review list not found:", REVIEW)
        return pd.Series("", index=ids.index)
    lab = (rv.problem + ":" + rv.role).groupby(rv.ID).agg(lambda s: ";".join(sorted(set(s))))
    return ids.map(lab).fillna("")


def input_provenance() -> dict:
    """Git commit of each sibling worktree and size / mtime / md5 of every input file (written by the prep step)."""
    import hashlib
    import subprocess
    out = {}
    for name, p in (("clusters", CLU), ("faults", FLT), ("this piece", ROOT)):
        try:
            out[f"{name} commit"] = subprocess.run(["git", "-C", str(p), "rev-parse", "--short", "HEAD"],
                                                   capture_output=True, text=True, check=True).stdout.strip()
        except Exception as e:  # noqa: BLE001
            out[f"{name} commit"] = f"unknown ({e})"
    files = [CAT, MAG, CLU / "events_zbz_v51.csv.gz", CLU / "regions_zbz.csv", CLU / "families_zbz_v51.csv",
             FLT / "fault_surfaces_v1_assignment.csv.gz", FLT / "fault_surfaces_v1_summary.csv",
             FLT_BIG / "fault_surfaces_v1_vertices.csv.gz", MODEL, MODEL_MEDIUM, REVIEW, SLAB]
    import time as _t
    for f in files:
        if not f.exists():
            out[str(f)] = "missing"
            continue
        h = hashlib.md5()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        st = f.stat()
        out[str(f)] = dict(bytes=st.st_size, mtime=_t.strftime("%Y-%m-%dT%H:%M:%S", _t.localtime(st.st_mtime)),
                           md5=h.hexdigest())
    return out


def attach_nodes(c: pd.DataFrame, model: Model, lon=None, lat=None, depth=None, prefix=""):
    """Nearest-node sampling of the resolved model (in place). Rows without a hypocentre get resolved = False."""
    lon = c.lon.values if lon is None else lon
    lat = c.lat.values if lat is None else lat
    depth = c.depth.values if depth is None else depth
    ix, iy, iz, inside = model.node_index(np.nan_to_num(lon), np.nan_to_num(lat), depth)
    inside &= c.located.values
    node = (iz * model.shape[1] + iy) * model.shape[2] + ix
    c[prefix + "node"] = np.where(inside, node, -1)
    for nm, mask in (("resP", model.mP), ("resS", model.mS), ("resPS", model.mPS)):
        c[prefix + nm] = inside & mask[iz, iy, ix]
    if prefix:
        return
    c["iz"], c["iy"], c["ix"] = iz, iy, ix
    c["node_depth"] = np.where(inside, model.dep[iz], np.nan)
    for v in ("Vp", "dVp", "Vp_res", "dVp_res"):
        c[v] = np.where(c.resP, getattr(model, v)[iz, iy, ix], np.nan)
    for v in ("Vs", "dVs", "VpVs", "dVpVs", "Vs_res", "dVs_res", "VpVs_res", "dVpVs_res"):
        c[v] = np.where(c.resPS, getattr(model, v)[iz, iy, ix], np.nan)
    c["dws_p"] = np.where(inside, model.dws_p[iz, iy, ix], np.nan)
    c["dws_s"] = np.where(inside, model.dws_s[iz, iy, ix], np.nan)
    c["stratum_P"] = np.where(c.resP, model.stratum_P[iz, iy, ix], -1)
    c["stratum_S"] = np.where(c.resPS, model.stratum_S[iz, iy, ix], -1)
    c["node_zone"] = np.where(inside, model.zone3[iz, iy, ix], "")
    c["resP8"] = model.corners_resolved(np.nan_to_num(lon), np.nan_to_num(lat), depth, model.mP) & c.located.values
    c["resPS8"] = model.corners_resolved(np.nan_to_num(lon), np.nan_to_num(lat), depth, model.mPS) & c.located.values


# ---------------------------------------------------------------- statistics
def block_bootstrap(blocks: np.ndarray, stat, n: int = 2000, seed: int = RNG_SEED, strata=None):
    """Resample whole blocks with replacement (within `strata` of blocks when given); stat(idx) -> float or array.
    Returns an array [n, ...] of the statistic."""
    rng = np.random.default_rng(seed)
    ub, inv = np.unique(blocks, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(ub))]
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(ub), len(ub))
        idx = np.concatenate([members[i] for i in pick]) if len(pick) else np.array([], int)
        out.append(stat(idx))
    return np.array(out, dtype=float)


def ci(a, q=(2.5, 97.5)):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return (np.nan, np.nan)
    return tuple(np.percentile(a, q))


def perm_p(real: float, null: np.ndarray) -> float:
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    if len(null) == 0 or not np.isfinite(real):
        return np.nan
    return float((1 + np.sum(np.abs(null) >= abs(real))) / (1 + len(null)))


def permute_within(labels: np.ndarray, strata: np.ndarray, rng) -> np.ndarray:
    out = labels.copy()
    for s in np.unique(strata):
        i = np.flatnonzero(strata == s)
        out[i] = labels[rng.permutation(i)]
    return out


def b_aki(excess: np.ndarray) -> float:
    """Aki-Utsu b from magnitudes minus their completeness (continuous magnitudes: no binning term)."""
    excess = np.asarray(excess, float)
    excess = excess[np.isfinite(excess) & (excess >= 0)]
    if len(excess) < 30 or excess.mean() <= 0:
        return np.nan
    return float(np.log10(np.e) / excess.mean())


def fmt(x, nd=2, sign=False):
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"

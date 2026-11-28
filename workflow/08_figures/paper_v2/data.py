"""Inputs of the paper figures v2: the configuration (paper_figures_v2.yaml) and loaders for every layer.

Every path comes from the configuration; nothing here hard-codes a catalog version, a segment list or a table name,
so the figures follow a rerun of the plate-boundary analysis (segment fix D-B113, null truth D-B112) by pointing the
configuration at its output. Segment codes, names and types come from seismicity_metrics.csv; structure classes from
structure_feasibility.csv.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]          # repository root of this checkout
KM = 111.195
RE = 6371.0


class Cfg(dict):
    """The configuration with ${key} expansion; paths resolved against the repository root."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def path(self, k, must=True):
        p = Path(self[k])
        p = p if p.is_absolute() else ROOT / p
        if must and not p.exists():
            raise FileNotFoundError(f"config '{k}': {p}")
        return p

    def opt(self, k):
        """Path of an optional input, or None when it does not exist (yet)."""
        try:
            return self.path(k, must=True)
        except (FileNotFoundError, KeyError):
            return None


def load_config(path) -> Cfg:
    raw = yaml.safe_load(Path(path).read_text())
    pat = re.compile(r"\$\{(\w+)\}")

    def expand(v, depth=0):
        if isinstance(v, str):
            for _ in range(10):
                n = pat.sub(lambda m: str(raw[m.group(1)]), v)
                if n == v:
                    break
                v = n
            return v
        if isinstance(v, list):
            return [expand(x) for x in v]
        if isinstance(v, dict):
            return {k: expand(x) for k, x in v.items()}
        return v
    cfg = Cfg({k: expand(v) for k, v in raw.items()})
    cfg["config_path"] = str(Path(path).resolve())
    return cfg


# ------------------------------------------------------------------ geometry
def hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * RE * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def local_xy(lon, lat, lon0, lat0):
    return ((np.asarray(lon, float) - lon0) * KM * np.cos(np.radians(lat0)), (np.asarray(lat, float) - lat0) * KM)


COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def compass(az):
    return COMPASS[int(np.round((az % 360) / 22.5)) % 16]


# ------------------------------------------------------------------ catalog and plate-boundary tables
@lru_cache(maxsize=2)
def _events(path):
    return pd.read_parquet(path)


def events(cfg, depth_good=False):
    """Located earthquakes of the catalog with their segment (events_segments parquet of the plate-boundary
    analysis): tier A-C, flagged non-earthquakes and the waveform-review P1 set left out (the paper set)."""
    c = _events(str(cfg.path("events_segments")))
    E = c[c.is_eq & c.abc & ~c.review_P1]
    if depth_good:
        E = E[E.depth_good]
    return E


def segments(cfg):
    """Segment table: seismicity metrics with the structure class, in the analysis order (types grouped)."""
    s = pd.read_csv(cfg.path("seismicity_metrics"))
    f = pd.read_csv(cfg.path("structure_feasibility"))
    keep = [c for c in ("seg", "class", "nP_10", "nPS_10", "span_PS10", "cb_shareP_10", "sd_Vp_10", "sd_VpVs_10")
            if c in f.columns]
    return s.merge(f[keep], on="seg", how="left")


def seg_type(cfg):
    s = segments(cfg)
    return dict(zip(s.seg, s.type))


def contrasts(cfg):
    """Every structure contrast with its verdicts: circularity control (D-B111: survives) and, when the null-truth
    table exists (D-B112), the null-truth columns (N95, pass_null, claimed). `status` is one of claimed /
    survives (null pending) / fails null / not claimed."""
    c = pd.read_csv(cfg.path("circularity_contrasts"))
    nt = cfg.opt("null_truth_contrasts")
    c["null_scored"] = nt is not None
    if nt is not None:
        n = pd.read_csv(nt)
        cols = [k for k in ("seg", "var", "contrast", "N95", "pass_null", "claimed", "testable_null", "null_mean",
                            "D0_truth") if k in n.columns]
        c = c.merge(n[cols], on=["seg", "var", "contrast"], how="left")
        c["status"] = np.where(c.claimed.fillna(False).astype(bool), "claimed",
                               np.where(c.survives.astype(bool), "fails null", "not claimed"))
    else:
        c["N95"] = np.nan
        c["claimed"] = False
        c["status"] = np.where(c.survives.astype(bool), "survives (null pending)", "not claimed")
    return c


# ------------------------------------------------------------------ model
class Model:
    """Model v0.4.2 on its node grid [depth, lat, lon] with the masks of the plate-boundary analysis
    (pb_common.Model): rock and (checkerboard-resolved or DWS >= dws_min); P for Vp, S for Vs (nodes with Vs reverted
    to the reference left out, D-B051), both for Vp/Vs. Ensemble 1 sigma: Vp_std, Vs_std as stored (calibrated,
    D-B099), Vp/Vs raw (D-B111)."""

    def __init__(self, cfg):
        import xarray as xr
        ds = xr.open_dataset(cfg.path("model"))
        en = xr.open_dataset(cfg.path("ensemble"))
        self.lon, self.lat, self.dep = (ds[c].values.astype(float) for c in ("longitude", "latitude", "depth"))
        g = lambda d, v: d[v].values.astype(float)            # noqa: E731
        self.Vp, self.Vs, self.VpVs = g(ds, "Vp"), g(ds, "Vs"), g(ds, "VpVs")
        self.surf = g(ds, "surface_elevation")                # km, + up
        rock = ds["medium"].values == 0
        dmin = float(cfg.get("dws_min", 100))
        self.rock = rock
        self.mP = rock & ((g(ds, "resolution_p") > 0) | (g(ds, "dws_p") >= dmin))
        self.mS = rock & ((g(ds, "resolution_s") > 0) | (g(ds, "dws_s") >= dmin)) & (ds["vs_reverted"].values == 0)
        self.mPS = self.mP & self.mS
        infl = cfg.get("inflation", {"VpVs": 1.0})
        self.sd = {"Vp": g(en, "Vp_std"), "Vs": g(en, "Vs_std"), "VpVs": g(en, "VpVs_std") * float(infl["VpVs"])}
        self.LA, self.LO = np.meshgrid(self.lat, self.lon, indexing="ij")
        self.version = ds.attrs.get("version", Path(cfg["model"]).stem)

    def mask(self, var):
        return {"Vp": self.mP, "Vs": self.mS, "VpVs": self.mPS}[var]

    def field(self, var):
        return {"Vp": self.Vp, "Vs": self.Vs, "VpVs": self.VpVs}[var]


def load_members(cfg):
    """Ensemble member grids [member, depth, lat, lon]: Vp (tomography) and Vp/Vs (Vs with the D-B073 node rule),
    as in pb_common.load_members."""
    root = cfg.path("members")
    names = sorted(p.name for p in root.iterdir() if (p / "merged.npz").exists())
    vp, vs = [], []
    for n in names:
        z = np.load(root / n / "merged.npz")
        vp.append(np.transpose(z["vp"], (2, 1, 0)))
        vs.append(np.transpose(z["vs_rule"], (2, 1, 0)))
    vp, vs = np.stack(vp).astype("float32"), np.stack(vs).astype("float32")
    return names, {"Vp": vp, "VpVs": vp / vs}


# ------------------------------------------------------------------ context
def slab_fn(cfg):
    """Slab2 depth (km, + down) at (lon, lat); NaN off the grid."""
    from scipy.interpolate import RegularGridInterpolator
    s = pd.read_csv(cfg.path("slab2"), header=None, names=["lon", "lat", "z"])
    s["lon"] = np.where(s.lon > 180, s.lon - 360, s.lon)
    lo, la = np.sort(s.lon.unique()), np.sort(s.lat.unique())
    grid = s.pivot_table(index="lat", columns="lon", values="z", dropna=False).reindex(index=la, columns=lo).values
    f = RegularGridInterpolator((la, lo), -grid, bounds_error=False, fill_value=np.nan)
    return lambda lon, lat: f(np.column_stack([np.ravel(lat), np.ravel(lon)])).reshape(np.shape(lon))


def slab_grid(cfg):
    s = pd.read_csv(cfg.path("slab2"), header=None, names=["lon", "lat", "z"])
    s["lon"] = np.where(s.lon > 180, s.lon - 360, s.lon)
    g = s.pivot_table(index="lat", columns="lon", values="z", dropna=False)
    return g.columns.to_numpy(), g.index.to_numpy(), -g.to_numpy()


def stations(cfg):
    s = pd.read_csv(cfg.path("stations"))
    for c in ("epoch_start", "epoch_end"):
        s[c] = pd.to_datetime(s[c], errors="coerce", utc=True).dt.tz_localize(None)
    s["epoch_end"] = s.epoch_end.fillna(pd.Timestamp("2030-01-01"))
    s["sea"] = s.seafloor.astype(str).str.lower().eq("true")
    s["used"] = s.in_ver5.astype(str).str.lower().eq("true") | s.in_v6.astype(str).str.lower().eq("true")
    return s


def context(cfg):
    """Coast, borders, PB2002 polylines and the GMRT relief (built once into the cache)."""
    import paper_style as S
    cache = Path(cfg["cache"])
    cache.mkdir(parents=True, exist_ok=True)
    return dict(coast=S.gmt_lines(cache, "coast_h", "-Dh -W"), borders=S.gmt_lines(cache, "borders_h", "-Dh -N1 -N2"),
                pb=S.pb2002_lines(cfg.path("pb2002"), axes=cfg.opt("plate_axes")),
                pb_ref=S.pb2002_lines(cfg.path("pb2002"), oceanic_only=True), relief=S.Relief(cache))


def threads():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, "2")

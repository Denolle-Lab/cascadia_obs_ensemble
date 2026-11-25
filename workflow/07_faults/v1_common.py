"""Shared pieces of fault surfaces v1 (catalog v5.1): paths, event selection, parameters.

Event selection for the fit (PI choice, 2026-11-25): tiers A-C located by NonLinLoc (loc_source nll3d), without
floor_limited, far_only or deep_pass events, without ComCat explosions, quarry blasts or other non-earthquakes
(event_flag anss_*) and without the Texada quarry_blast_likely events. Axial Seamount stays out, as in v0
(published fault maps there are better than this catalog allows). Rows without a hypocentre (loc_source none) are
left out of everything spatial.

Per-event location error (convention of workflow/10_tomography, b2_ens_common.h68_factor): sigma_h = h / 1.515 with
h the 68 % horizontal semi-axis (chi2 with 2 degrees of freedom), sigma_z = sig_z. The catalog gives the two
horizontal semi-axes but not their azimuth, so one horizontal sigma with the same total variance is used:
sigma_h = sqrt((h_min^2 + h_max^2) / 2) / 1.515.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fault_surfaces_v0 as v0  # noqa: E402

ROOT = HERE.parents[1]
CAT = Path("/wd1/mdenolle_data/catalog_v5/v5.1/cascadia_catalog_v5.1.csv")
OUT = ROOT / "data" / "catalog_v5" / "v5.1_analysis" / "faults"          # small outputs (git)
BIG = Path("/wd1/mdenolle_data/catalog_v5/v5.1_analysis/faults")         # large outputs
WORK = BIG / "work"
V0DIR = ROOT / "data" / "faults" / "v0"
H68 = 1.515
NONQUAKE = "anss_explosion|anss_quarry_blast|anss_other_event|anss_sonic_boom|quarry_blast_likely"
TAG = "v1"
CATLABEL = "Cascadia catalog v5.1"
SHIFT_FLAG = 10.0   # flag surfaces whose members moved > 10 km in depth from v4 to v5.1 (v1_build, v1_report)


def params(**kw):
    """The v0 detection parameters (fault_surfaces_planes.py defaults), unchanged for v1."""
    p = dict(tol_h=1.5, tol_z=3.0, tol_k=1.5, ball=20.0, max_reach=150.0, gap=4.0, trials=150, min_seed_nb=6,
             n_min=30, merge_angle=15.0, min_dip=30.0, planarity=0.5, pad=0.5, grid=1.0, draws=100)
    p.update(kw)
    return argparse.Namespace(**p)


def strand_params(**kw):
    """The v0 strand parameters (split_strands.py defaults)."""
    p = dict(azimuth=111.0, az_window=20.0, tol=1.5, gap=4.0, trials=300, n_min=30, min_d=2.0)
    p.update(kw)
    return argparse.Namespace(**p)


def _bool(s):
    return s.astype(str).str.strip().eq("True")


def load():
    """All catalog rows with fit/assignment columns. Returns (c, counts)."""
    c = pd.read_csv(CAT, low_memory=False).rename(columns={"ID": "event_id"})
    for k in ("floor_limited", "far_only", "deep_pass", "five_phase"):
        c[k] = _bool(c[k])
    c["located"] = c.loc_source.isin(["nll3d", "comcat"]) & c.lat.notna() & c.depth.notna()
    c["nonquake"] = c.event_flag.fillna("").str.contains(NONQUAKE)
    c["longitude"], c["latitude"], c["z"] = c.lon, c.lat, c.depth
    lo, la = c.lon.fillna(-125.0).to_numpy(), c.lat.fillna(45.0).to_numpy()
    c["fregion"] = np.where(c.located, v0.region_of(lo, la), "")
    c["x"], c["y"] = v0.sinu(lo, la)
    sh = np.sqrt((c.h_min ** 2 + c.h_max ** 2) / 2.0) / H68
    c["sh"] = sh
    c["sz"] = c.sig_z
    c["sigma_fallback"] = c.sh.isna() | c.sz.isna()
    c.loc[c.sigma_fallback, "sh"] = 1.5                      # nominal v0 relative errors where the catalog has none
    c.loc[c.sigma_fallback, "sz"] = 3.0                      # (the 2 loc_source comcat rows)
    reason = np.full(len(c), "", object)
    rules = [("no hypocentre", ~c.located),
             ("not an earthquake (event_flag)", c.nonquake),
             ("tier D or U", ~c.tier.isin(["A", "B", "C"])),
             ("not located by NonLinLoc here", c.loc_source != "nll3d"),
             ("floor_limited", c.floor_limited),
             ("far_only", c.far_only),
             ("deep_pass", c.deep_pass),
             ("Axial Seamount (left out, as in v0)", c.fregion == "AX")]
    counts = []
    for name, m in rules:
        m = m.to_numpy() & (reason == "")
        reason[m] = name
        counts.append((name, int(m.sum())))
    c["fit_excl"] = reason
    c["fit"] = reason == ""
    counts.append(("used for the fit", int(c.fit.sum())))
    t = pd.to_datetime(c.time, errors="coerce", utc=True)
    c["t"] = t
    loc = t.dt.tz_convert("America/Los_Angeles")
    c["weekday_daytime"] = (loc.dt.dayofweek < 5) & (loc.dt.hour >= 8) & (loc.dt.hour < 17)
    c["year"] = t.dt.year
    return c, counts


def fit_set(c):
    f = c[c.fit].reset_index(drop=True).copy()
    f["region"] = f.fregion
    return f

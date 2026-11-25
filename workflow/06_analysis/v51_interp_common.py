"""Shared inputs and helpers for the catalog v5.1 re-interpretation (piece "interp").

Reads (never writes) the catalog v5.1, the outputs of the pieces mag, faults and clusters
(their worktrees, committed state), their verifiers' tables on /wd1, and the v4-era products
(catalog v4 clusters, fault surfaces v0 and their literature matches) of this worktree.

Paths can be overridden with environment variables (V51_CAT, V51_MAG, V51_FAULTS, V51_CLUSTERS).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WD1 = Path("/wd1/mdenolle_data/catalog_v5")
CAT = Path(os.environ.get("V51_CAT", WD1 / "v5.1" / "cascadia_catalog_v5.1.csv"))
MAG = Path(os.environ.get("V51_MAG", "/home/mdenolle/wt_a51_mag/data/catalog_v5/v5.1_analysis/mag"))
FLT = Path(os.environ.get("V51_FAULTS", "/home/mdenolle/wt_a51_faults/data/catalog_v5/v5.1_analysis/faults"))
CLU = Path(os.environ.get("V51_CLUSTERS", "/home/mdenolle/wt_a51_clusters/data/catalog_v5/v5.1_analysis/clusters"))
VER_F = WD1 / "v5.1_analysis" / "verify_faults"
VER_C = WD1 / "v5.1_analysis" / "verify_clusters"
VER_M = WD1 / "v5.1_analysis" / "verify_mag"
REVIEW = WD1 / "v5.1_review"          # waveform review of four v5.1 location problems (read only)
V4C = ROOT / "data" / "catalog_v4" / "clusters_zbz"
V4S = ROOT / "data" / "catalog_v4" / "swarms"
V0 = ROOT / "data" / "faults" / "v0"
OUT = ROOT / "data" / "catalog_v5" / "v5.1_analysis" / "interp"
BIG = WD1 / "v5.1_analysis" / "interp"
SLAB2 = Path("/home/mdenolle/cascadia_obs_ensemble/data/slab2/cas_slab2_dep.xyz")
PB_STEPS = ROOT / "data" / "tectonics" / "PB2002_steps.json"

REGIONS = ["Axial", "Endeavour", "Explorer", "JdF ridge/plate", "Blanco", "Gorda", "Mendocino",
           "Offshore forearc", "Puget/Georgia", "Cascades", "Other onshore"]
SLUG = {r: r.lower().replace("/", "_").replace(" ", "_") for r in REGIONS}

# Cascade volcanoes and Axial (summit or caldera centre; deg). Public summit coordinates.
VOLCANOES = {
    "Mount Baker": (48.777, -121.813), "Glacier Peak": (48.112, -121.113),
    "Mount Rainier": (46.853, -121.760), "Mount St. Helens": (46.191, -122.194),
    "Mount Adams": (46.206, -121.490), "Mount Hood": (45.374, -121.696),
    "Mount Jefferson": (44.674, -121.800), "Three Sisters": (44.103, -121.769),
    "Newberry": (43.722, -121.229), "Crater Lake": (42.930, -122.120),
    "Mount McLoughlin": (42.445, -122.315), "Medicine Lake": (41.611, -121.554),
    "Mount Shasta": (41.409, -122.193), "Lassen Peak": (40.488, -121.505),
    "Axial Seamount": (45.950, -130.010),
}


def hav(lat1, lon1, lat2, lon2):
    """Great-circle distance, km."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def local_xy(lat, lon, lat0, lon0):
    """Local equirectangular projection, km (x east, y north)."""
    x = (np.asarray(lon) - lon0) * 111.195 * np.cos(np.radians(lat0))
    y = (np.asarray(lat) - lat0) * 111.195
    return x, y


def tobool(s):
    return s.astype(str).str.strip().eq("True")


def load_events() -> pd.DataFrame:
    """Catalog v5.1 joined with the v5.1 magnitudes, the clustering classes and the fault assignment."""
    c = pd.read_csv(CAT, low_memory=False)
    for col in ["floor_limited", "far_only", "deep_pass", "five_phase", "above_surface"]:
        c[col] = tobool(c[col])
    c["located"] = c.loc_source.ne("none")
    m = pd.read_csv(MAG / "cascadia_catalog_v5.1_magnitudes_slim.csv.gz")
    m = m[["ID", "M", "M_type", "ML", "is_earthquake", "mt_match", "Mw_mt"]].rename(
        columns={"M": "M51", "M_type": "M51_type", "ML": "ML51", "Mw_mt": "Mw_mt51"})
    c = c.rename(columns={"M": "M41", "M_type": "M41_type"}).merge(m, on="ID", how="left")
    c["is_earthquake"] = tobool(c.is_earthquake)
    z = pd.read_csv(CLU / "events_zbz_v51.csv.gz")
    z = z.rename(columns={"event_id": "ID", "region": "box51", "M": "M_zbz"})
    c = c.merge(z[["ID", "box51", "in_analysis", "family_id", "family_type", "role", "background", "cls",
                   "log_eta"]], on="ID", how="left")
    a = pd.read_csv(FLT / "fault_surfaces_v1_assignment.csv.gz")
    a = a.rename(columns={"event_id": "ID", "surface_id": "surf_v1", "region": "fbox"})
    if "on_surface_strict" not in a.columns:
        a["on_surface_strict"] = False
    a["on_surface_strict"] = tobool(a.on_surface_strict)
    c = c.merge(a[["ID", "assignment", "surf_v1", "distance_km", "z", "n_compatible", "fbox", "on_surface_strict"]],
                on="ID", how="left")
    c["on_surface_strict"] = c.on_surface_strict.eq(True)
    # Axial was left out of the fault fit (as in v0): neither on nor off a surface. The revised faults product labels
    # it "not fitted (Axial Seamount left out)"; the first build labelled it "none" (relabelled here).
    ax = (c.assignment.eq("none") | c.assignment.astype(str).str.startswith("not fitted")) & c.fbox.eq("AX")
    c.loc[ax, "assignment"] = "region not fitted (Axial)"
    v4 = pd.read_csv(V4C / "events_zbz.csv.gz", usecols=["event_id", "in_analysis", "family_id", "role"])
    f4 = pd.read_csv(V4C / "families_zbz.csv", usecols=["family_id", "type"])
    v4 = v4.merge(f4, on="family_id", how="left").rename(
        columns={"event_id": "ID", "in_analysis": "in_analysis4", "family_id": "family_id4", "role": "role4",
                 "type": "family_type4"})
    v4["cls4"] = np.where(v4.family_id4.ge(0) & v4.family_type4.notna(), v4.family_type4, "background")
    c = c.merge(v4[["ID", "in_analysis4", "family_id4", "family_type4", "cls4"]], on="ID", how="left")
    c["t"] = pd.to_datetime(c.time, utc=True, format="ISO8601", errors="coerce")
    c["t4"] = pd.to_datetime(c.v4_time, utc=True, format="ISO8601", errors="coerce")
    c["bsf"] = c.depth + c.surface_km          # depth below the local ground/seafloor (km)
    c["is_eq"] = c.located & c.is_earthquake
    c["depth_good"] = (c.is_eq & c.tier.isin(["A", "B", "C"]) & c.depth_quality.eq("ok") & ~c.floor_limited
                       & ~c.far_only & ~c.deep_pass)
    c["fit_like"] = (c.is_eq & c.tier.isin(["A", "B", "C"]) & ~c.floor_limited & ~c.far_only & ~c.deep_pass)
    return c


def load_surfaces() -> pd.DataFrame:
    s = pd.read_csv(FLT / "fault_surfaces_v1_summary.csv")
    s = s.rename(columns={"flags": "flt_flags"})
    tr = pd.read_csv(FLT / "fault_surfaces_v1_traces.csv")
    s = s.merge(tr[["surface_id", "top_edge_lon1", "top_edge_lat1", "top_edge_lon2", "top_edge_lat2"]],
                on="surface_id", how="left")
    if "recovery_J_sub80" in s.columns:      # revised faults product carries the verifier's detection reruns
        s = s.rename(columns={"recovery_J_sub80": "J_sub80", "recovery_J_perturbed": "J_perturbed",
                              "recovery_J_tierAB": "J_tierAB"})
    else:
        rec = pd.read_csv(VER_F / "recovery_all.csv").rename(columns={"sid": "surface_id"})
        s = s.merge(rec[["surface_id", "J_sub80", "J_perturbed", "J_tierAB"]], on="surface_id", how="left")
    cc = pd.read_csv(CLU / "faults_v1_cluster_counts.csv")
    s = s.merge(cc[["surface_id", "n_events_joined", "n_analysis", "pct_background", "pct_swarm",
                    "n_families", "n_swarm_families_all"]], on="surface_id", how="left")
    return s


def load_v0_traces() -> pd.DataFrame:
    t = pd.read_csv(V0 / "fault_surfaces_v0_traces.csv")
    lit = pd.read_csv(V0 / "fault_surfaces_v0_literature.csv")
    return t.merge(lit[["surface_id", "lit_match", "lit_confidence", "lit_class"]], on="surface_id", how="left")


def load_pb_steps() -> pd.DataFrame:
    """PB2002 plate-boundary steps near Cascadia: midpoints with step class (OSR ridge, OTF transform, SUB...)."""
    import json
    js = json.loads(PB_STEPS.read_text())
    rows = []
    for f in js["features"]:
        p = f["properties"]
        lo = (p["STARTLONG"] + p["FINALLONG"]) / 2
        la = (p["STARTLAT"] + p["FINALLAT"]) / 2
        if -135 < lo < -118 and 37 < la < 53:
            rows.append(dict(lon=lo, lat=la, cls=p["STEPCLASS"], bound=p["PLATEBOUND"],
                             lon1=p["STARTLONG"], lat1=p["STARTLAT"], lon2=p["FINALLONG"], lat2=p["FINALLAT"]))
    return pd.DataFrame(rows)


def nearest_boundary(lat, lon, pb: pd.DataFrame):
    """Distance (km) and class of the nearest PB2002 step midpoint (steps are ~10-30 km long)."""
    lat = np.atleast_1d(lat)
    lon = np.atleast_1d(lon)
    d = hav(lat[:, None], lon[:, None], pb.lat.values[None, :], pb.lon.values[None, :])
    k = np.argmin(d, axis=1)
    return d[np.arange(len(lat)), k], pb.cls.values[k], pb.bound.values[k]


def nearest_volcano(lat, lon):
    names = list(VOLCANOES)
    la = np.array([VOLCANOES[n][0] for n in names])
    lo = np.array([VOLCANOES[n][1] for n in names])
    lat = np.atleast_1d(lat)
    lon = np.atleast_1d(lon)
    d = hav(lat[:, None], lon[:, None], la[None, :], lo[None, :])
    k = np.argmin(d, axis=1)
    return d[np.arange(len(lat)), k], np.array(names)[k]


def q(x, p):
    x = pd.Series(x).dropna()
    return float(np.percentile(x, p)) if len(x) else np.nan


def med_rng(x, nd=1):
    x = pd.Series(x).dropna()
    if not len(x):
        return "n/a"
    return f"{np.median(x):.{nd}f} [{np.percentile(x, 10):.{nd}f}, {np.percentile(x, 90):.{nd}f}]"

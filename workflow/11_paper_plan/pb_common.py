"""Shared inputs, geometry and the plate-boundary segment rule for the paper-plan feasibility study (D-B107).

Reads only: catalog v6.0, its v6.0 clustering, the v6.0 Mc maps, the fault surfaces v1 (fitted on v5.1), model
v0.4.2 and its ensemble v0.4.2+ens.100 (statistics file and the 100 member grids), PB2002 plate boundaries and Slab2.
Writes nothing (the scripts pb_*.py write).

Other catalog versions (end-to-end pipeline, D-B110): environment variables replace the v6.0 inputs and the output
folders; unset, every path is the v6.0 one below, so the committed feasibility study reproduces. PB_CATALOG (catalog
CSV), PB_TAG (file tag of its clustering and of the event table, v60), PB_LABEL (catalog name in titles, v6.0),
PB_CLUSTERS (clustering folder: events_zbz_<tag>.csv.gz, families_zbz_<tag>.csv, dbscan_*_<tag>.csv), PB_MCMAP (Mc map
folder), PB_FAULTS and PB_FAULTS_BIG (fault surfaces v1: small tables, mesh vertices), PB_MODEL, PB_ENS, PB_MEMBERS
(model, ensemble statistics, member grids), PB_SYNTH (synthetic calibration ensemble), PB_OUT (small tables), PB_BIG (page, figures, large tables).

Conventions: depth km below sea level, positive down; "bsf" = depth below the local solid surface (ground or
seafloor) = depth + surface_km. Distances are straight-line 3-D distances in an Earth-centred frame (radius 6371 km
minus depth) unless called horizontal.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WD1 = Path("/wd1/mdenolle_data")
_env = os.environ.get
CAT = Path(_env("PB_CATALOG", WD1 / "catalog_v6/v6.0/cascadia_catalog_v6.0.csv"))
TAG = _env("PB_TAG", "v60")                              # file tag of the clustering and of events_segments_<tag>
CATLABEL = _env("PB_LABEL", "v6.0")                      # catalog name in titles
CLU = Path(_env("PB_CLUSTERS", WD1 / "catalog_v6/v6.0_analysis/clusters"))
MCMAP = Path(_env("PB_MCMAP", WD1 / "magnitude_converged/mc_maps_v60"))
FLT_BIG = Path(_env("PB_FAULTS_BIG", WD1 / "catalog_v5/v5.1_analysis/faults"))
FLT = Path(_env("PB_FAULTS", ROOT / "data/catalog_v5/v5.1_analysis/faults"))
MODELS = WD1 / "tomo_v1/models/cascadia_obs_tomo"
MODEL = Path(_env("PB_MODEL", MODELS / "v0.4.2.nc"))
ENS = Path(_env("PB_ENS", MODELS / "v0.4.2+ens.100.nc"))
MEMBERS = Path(_env("PB_MEMBERS", WD1 / "tomo_v1/ensemble_v0.4.2/real"))
INFL = {"Vp": 1.50, "Vs": 1.35, "VpVs": 1.00}   # D-B099; Vp/Vs calibrated without inflation (D-B111: coverage 0.68/0.97)
SYNTH = Path(_env("PB_SYNTH", WD1 / "tomo_v1/ensemble_v0.4.2/synth"))   # synthetic calibration ensemble (one truth, s000-s099, nf0-nf4)
PB2002 = ROOT / "data/tectonics/PB2002_steps.json"
SLAB = Path("/home/mdenolle/cascadia_obs_ensemble/data/slab2/cas_slab2_dep.xyz")
OUT = Path(_env("PB_OUT", ROOT / "data/paper_plan/plate_boundaries"))  # small tables (git)
BIG = Path(_env("PB_BIG", WD1 / "paper_plan"))                          # page, figures, large tables
RE = 6371.0
KM = 111.195
SEED = 20261126

# ------------------------------------------------------------------ segments
# code: (name, type, colour group order). Types: ridge, transform, intraplate, triple junction, subduction, crustal,
# arc, other (not a plate-boundary segment; counted, not compared).
SEGMENTS = {
    "AX": ("Axial Seamount (JdF ridge)", "ridge"),
    "JR": ("JdF ridge, other segments", "ridge"),
    "EN": ("Endeavour segment", "ridge"),
    "EXR": ("Explorer ridge", "ridge"),
    "GR": ("Gorda ridge", "ridge"),
    "BL": ("Blanco transform", "transform"),
    "MFZ": ("Mendocino fault zone", "transform"),
    "SOV": ("Sovanco transform", "transform"),
    "JP": ("JdF plate interior", "intraplate"),
    "GP": ("Gorda plate interior", "intraplate"),
    "EXP": ("Explorer plate / Nootka", "intraplate"),
    "MTJ": ("Mendocino triple junction", "triple junction"),
    "GS": ("Gorda slab beneath the margin (S of 41.5 N)", "subduction"),
    "FA": ("Forearc upper plate (N of 41.5 N)", "subduction"),
    "SIF": ("Interface zone (N of 41.5 N)", "subduction"),
    "IS": ("JdF intraslab (N of 41.5 N)", "subduction"),
    "PU": ("Puget-Georgia crust", "crustal"),
    "MSH": ("Mount St Helens", "arc"),
    "CAS": ("Cascade arc, other", "arc"),
    "OC": ("Other onshore crust", "other"),
    "NAPA": ("North America-Pacific, south of the MTJ", "other"),
}
SEG_ORDER = list(SEGMENTS)
TYPE_ORDER = ["ridge", "transform", "intraplate", "triple junction", "subduction", "crustal", "arc", "other"]
MSH = (46.191, -122.194)            # summit (public coordinates)
R_MTJ_KM = 40.0                      # horizontal radius about the PB2002 triple junction
D_AXIS_KM = 15.0                     # horizontal half-width about a PB2002 ridge or transform trace
R_MSH_KM = 25.0                      # horizontal radius about the Mount St Helens summit
SLAB_IF_KM = 5.0                     # |depth - Slab2| < 5 km: interface zone; > +5 km below: intraslab
SLAB_CRUST_KM = 10.0                 # onshore crust: more than 10 km above Slab2 (or no Slab2)
LAT_GS = 41.5                        # south of it, east of the trench and outside MTJ: Gorda slab group GS

SEGMENT_RULE = [
    f"Located events of catalog {CATLABEL} (any tier), first match wins; distances are horizontal.",
    "AX: catalog region Axial. BL: catalog region Blanco. EN: catalog region Endeavour.",
    f"MTJ: within {R_MTJ_KM:.0f} km of the PB2002 triple junction (southern end of the JF\\NA subduction trace).",
    f"MFZ: within {D_AXIS_KM:.0f} km of a PB2002 PA-JF transform step, south of 41 N.",
    f"Ridges: within {D_AXIS_KM:.0f} km of a PB2002 PA-JF spreading-ridge step: GR south of 43 N, JR 43-48.35 N, "
    "EXR north of 48.35 N.",
    f"Explorer region, rest: SOV within {D_AXIS_KM:.0f} km of a transform step, else EXP.",
    "West of the PB2002 subduction trace (oceanic plate): GP south of 43 N, JP north of it.",
    "South of the triple junction latitude (outside MTJ and MFZ): NAPA.",
    f"East of the trace, by depth relative to Slab2 (rel = depth - slab): Cascades region: MSH within "
    f"{R_MSH_KM:.0f} km of the summit, else CAS. Puget/Georgia and Other onshore: crust (PU, OC) if "
    f"rel < -{SLAB_CRUST_KM:.0f} km or no Slab2. Every other event east of the trace: GS south of {LAT_GS} N; "
    f"north of it IS if rel >= +{SLAB_IF_KM:.0f} km, SIF if |rel| < {SLAB_IF_KM:.0f} km, FA otherwise (also "
    "without Slab2).",
]


def log(*a):
    print(*a, flush=True)


def tobool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().eq("true")


def ecef(lon, lat, dep=0.0):
    lo, la = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    r = RE - np.asarray(dep, float)
    return np.stack([r * np.cos(la) * np.cos(lo), r * np.cos(la) * np.sin(lo), r * np.sin(la)], axis=-1)


def hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * RE * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ------------------------------------------------------------------ plate boundaries, slab
def pb2002():
    """PB2002 steps in the study area: dict class -> (N, 2) lon/lat points (steps are densified at about 1 km),
    plus the trench trace (sorted by latitude) and the triple junction (southern end of the subduction trace)."""
    d = json.loads(PB2002.read_text())
    pts = {"OSR": [], "OTF": [], "SUB": [], "CTF": []}
    for f in d["features"]:
        p = f["properties"]
        if not (-133 < p["STARTLONG"] < -119 and 37 < p["STARTLAT"] < 53):
            continue
        cl = p["STEPCLASS"]
        key = {"OSR": "OSR", "OTF": "OTF", "SUB": "SUB", "CTF": "CTF"}.get(cl)
        if key is None:
            continue
        xy = np.asarray(f["geometry"]["coordinates"], float)
        pts[key].append(xy)
    pts = {k: np.vstack(v) for k, v in pts.items() if v}
    sub = pts["SUB"][np.argsort(pts["SUB"][:, 1])]
    tj = sub[0].copy()
    return pts, sub, tj


def dist_to_trace(lon, lat, xy):
    """Horizontal distance (km) of points to the nearest point of a densified trace."""
    from scipy.spatial import cKDTree
    t = cKDTree(ecef(xy[:, 0], xy[:, 1]))
    d, _ = t.query(ecef(lon, lat))
    return d


def trench_lon(lat, sub):
    return np.interp(np.asarray(lat, float), sub[:, 1], sub[:, 0])


def slab_depth_fn():
    """Slab2 depth (km, positive down) at (lon, lat), bilinear on its 0.05 deg grid; NaN off grid."""
    from scipy.interpolate import RegularGridInterpolator
    s = pd.read_csv(SLAB, header=None, names=["lon", "lat", "z"])
    s["lon"] = np.where(s.lon > 180, s.lon - 360, s.lon)
    lo, la = np.sort(s.lon.unique()), np.sort(s.lat.unique())
    grid = s.pivot_table(index="lat", columns="lon", values="z", dropna=False).reindex(index=la, columns=lo).values
    f = RegularGridInterpolator((la, lo), -grid, bounds_error=False, fill_value=np.nan)
    return lambda lon, lat: f(np.column_stack([np.asarray(lat, float), np.asarray(lon, float)]))


# ------------------------------------------------------------------ catalog
CAT_COLS = ["ID", "time", "lat", "lon", "depth", "loc_source", "tier", "region", "h_min", "h_max", "sig_z",
            "depth_quality", "event_flag", "floor_limited", "far_only", "deep_pass", "five_phase", "surface_km",
            "M", "M_type", "review_problems", "review_auto_indicators", "nphs", "gap"]


def load_catalog() -> pd.DataFrame:
    c = pd.read_csv(CAT, usecols=CAT_COLS, low_memory=False)
    c = c[c.loc_source.ne("none")].copy()
    for f in ("floor_limited", "far_only", "deep_pass", "five_phase"):
        c[f] = tobool(c[f])
    c["is_eq"] = c.event_flag.isna()
    c["abc"] = c.tier.isin(["A", "B", "C"])
    c["depth_good"] = c.abc & c.depth_quality.eq("ok") & ~c.floor_limited & ~c.far_only & ~c.deep_pass
    c["bsf"] = c.depth + c.surface_km
    c["t"] = pd.to_datetime(c.time, utc=True, format="ISO8601")
    c["review_P1"] = c.review_problems.fillna("").str.contains("P1")
    return c.reset_index(drop=True)


def assign_segments(c: pd.DataFrame) -> pd.DataFrame:
    """Add seg (code), seg_type, d_osr, d_otf, d_tj, trench_lon, slab_km, rel_slab (see SEGMENT_RULE)."""
    pts, sub, tj = pb2002()
    lon, lat = c.lon.values, c.lat.values
    d_osr = dist_to_trace(lon, lat, pts["OSR"])
    d_otf = dist_to_trace(lon, lat, pts["OTF"])
    d_tj = hav(lat, lon, tj[1], tj[0])
    tl = trench_lon(lat, sub)
    sl = slab_depth_fn()(lon, lat)
    rel = c.depth.values - sl
    reg = c.region.values
    seg = np.full(len(c), "", dtype=object)

    def put(mask, code):
        m = (seg == "") & mask
        seg[m] = code
    put(reg == "Axial", "AX")
    put(reg == "Blanco", "BL")
    put(reg == "Endeavour", "EN")
    put(d_tj <= R_MTJ_KM, "MTJ")
    put((d_otf <= D_AXIS_KM) & (lat < 41.0), "MFZ")
    near_r = d_osr <= D_AXIS_KM
    put(near_r & (lat < 43.0), "GR")
    put(near_r & (lat >= 43.0) & (lat < 48.35), "JR")
    put(near_r & (lat >= 48.35), "EXR")
    put((reg == "Explorer") & (d_otf <= D_AXIS_KM), "SOV")
    put(reg == "Explorer", "EXP")
    put(lat < tj[1], "NAPA")
    west = lon < tl
    put(west & (lat < 43.0), "GP")
    put(west, "JP")
    put((reg == "Cascades") & (hav(lat, lon, MSH[0], MSH[1]) <= R_MSH_KM), "MSH")
    put(reg == "Cascades", "CAS")
    crust = ~(rel >= -SLAB_CRUST_KM)           # rel < -10 or no Slab2
    put((reg == "Puget/Georgia") & crust, "PU")
    put((reg == "Other onshore") & crust, "OC")
    put(lat < LAT_GS, "GS")
    put(rel >= SLAB_IF_KM, "IS")
    put(np.abs(rel) < SLAB_IF_KM, "SIF")
    put(np.ones(len(c), bool), "FA")
    c = c.copy()
    c["seg"] = seg
    c["seg_type"] = c.seg.map({k: v[1] for k, v in SEGMENTS.items()})
    c["d_osr"], c["d_otf"], c["d_tj"] = d_osr, d_otf, d_tj
    c["trench_lon"], c["slab_km"], c["rel_slab"] = tl, sl, rel
    return c


# ------------------------------------------------------------------ model and ensemble
class Model:
    """v0.4.2 on its node grid with the feasibility masks and the ensemble statistics.

    Masks (as used so far for the v0.4.2 figures): rock AND (checkerboard-resolved OR DWS >= 100); P for Vp, S for Vs,
    both for Vp/Vs. Vs nodes reverted to the reference (vs_reverted > 0, D-B051) are left out of the S mask here (their
    Vs is not from the data)."""

    def __init__(self):
        import xarray as xr
        ds = xr.open_dataset(MODEL)
        en = xr.open_dataset(ENS)
        self.lon, self.lat, self.dep = ds.longitude.values, ds.latitude.values, ds.depth.values
        g = lambda d, v: d[v].values.astype(float)            # noqa: E731  [depth, lat, lon]
        self.Vp, self.Vs, self.VpVs = g(ds, "Vp"), g(ds, "Vs"), g(ds, "VpVs")
        self.dVp, self.dVs = g(ds, "Vp_perturbation"), g(ds, "Vs_perturbation")
        self.Vp0, self.Vs0 = g(ds, "Vp_start"), g(ds, "Vs_start")
        self.dws_p, self.dws_s = g(ds, "dws_p"), g(ds, "dws_s")
        self.res_p, self.res_s = g(ds, "resolution_p"), g(ds, "resolution_s")
        self.vs_rev = ds["vs_reverted"].values
        self.medium = ds["medium"].values
        self.surf = ds["surface_elevation"].values.astype(float)    # km, up
        rock = self.medium == 0
        self.rock = rock
        self.mP = rock & ((self.res_p > 0) | (self.dws_p >= 100))
        self.mS_any = rock & ((self.res_s > 0) | (self.dws_s >= 100))
        self.mS = self.mS_any & (self.vs_rev == 0)
        self.mPS = self.mP & self.mS
        self.cbP, self.cbS = rock & (self.res_p > 0), rock & (self.res_s > 0) & (self.vs_rev == 0)
        self.sdVp, self.sdVs = g(en, "Vp_std"), g(en, "Vs_std")                       # calibrated (D-B099)
        self.sdVpVs = g(en, "VpVs_std") * INFL["VpVs"]                                  # raw spread (D-B111)
        self.ens_med = {"Vp": g(en, "Vp_median"), "Vs": g(en, "Vs_median"), "VpVs": g(en, "VpVs_median")}
        LA, LO = np.meshgrid(self.lat, self.lon, indexing="ij")
        self.LA, self.LO = LA, LO
        D = self.dep[:, None, None] * np.ones(self.Vp.shape)
        self.D = D
        self.xyz = ecef(np.broadcast_to(LO, D.shape), np.broadcast_to(LA, D.shape), D)   # [z, y, x, 3]
        self.attrs = dict(ds.attrs)

    def mask(self, var):
        return {"Vp": self.mP, "dVp": self.mP, "Vs": self.mS, "dVs": self.mS, "VpVs": self.mPS}[var]

    def sd(self, var):
        return {"Vp": self.sdVp, "Vs": self.sdVs, "VpVs": self.sdVpVs}[var]

    def column_index(self, lon, lat):
        ix = np.clip(np.round((np.asarray(lon) - self.lon[0]) / 0.125).astype(int), 0, len(self.lon) - 1)
        iy = np.clip(np.round((np.asarray(lat) - self.lat[0]) / 0.1).astype(int), 0, len(self.lat) - 1)
        return iy, ix

    def z8(self, level=8.0, V=None):
        """Per column: the shallowest depth where Vp reaches `level`, interpolated between two adjacent rock nodes
        that are both in the P mask; NaN where that crossing is not resolved, or where Vp is at or above the level
        already at the top rock node (no crossing). V: another Vp grid (an ensemble member) on the same nodes."""
        return level_depth(self.Vp if V is None else V, self.mP, self.rock, self.dep, level)


def level_depth(V, mP, rock, dep, level):
    nz = V.shape[0]
    out = np.full(V.shape[1:], np.nan)
    done = np.zeros(V.shape[1:], bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        for k in range(nz - 1):
            both = rock[k] & rock[k + 1]
            cross = both & (V[k] < level) & (V[k + 1] >= level) & ~done
            ok = cross & mP[k] & mP[k + 1]
            w = (level - V[k]) / (V[k + 1] - V[k])
            out[ok] = dep[k] + w[ok] * (dep[k + 1] - dep[k])
            done |= cross | (both & (V[k] >= level))
    return out


def load_synth(names=None):
    """Synthetic ensemble: truth and members, [.., depth, lat, lon], tomography values before the D-B073 node rule
    (as the calibration compares them). Returns truth dict, member names, member dict (Vp, Vs, VpVs)."""
    t = np.load(SYNTH / "truth" / "truth_model.npz")
    tr = {"Vp": np.transpose(t["vp"], (2, 1, 0)).astype(float), "Vs": np.transpose(t["vs"], (2, 1, 0)).astype(float)}
    tr["VpVs"] = tr["Vp"] / tr["Vs"]
    names = names or sorted(p.name for p in SYNTH.iterdir() if p.name[0] == "s" and (p / "merged.npz").exists())
    vp, vs = [], []
    for n in names:
        z = np.load(SYNTH / n / "merged.npz")
        vp.append(np.transpose(z["vp"], (2, 1, 0)))
        vs.append(np.transpose(z["vs"], (2, 1, 0)))
    vp, vs = np.stack(vp).astype(float), np.stack(vs).astype(float)
    return tr, names, {"Vp": vp, "Vs": vs, "VpVs": vp / vs}


def load_members(names=None):
    """Ensemble member grids, [member, depth, lat, lon]: Vp (tomography), Vs (with the D-B073 node rule), Vp/Vs."""
    names = names or sorted(p.name for p in MEMBERS.iterdir() if (p / "merged.npz").exists())
    vp, vs = [], []
    for n in names:
        z = np.load(MEMBERS / n / "merged.npz")
        vp.append(np.transpose(z["vp"], (2, 1, 0)))
        vs.append(np.transpose(z["vs_rule"], (2, 1, 0)))
    vp, vs = np.stack(vp).astype(float), np.stack(vs).astype(float)
    return names, {"Vp": vp, "Vs": vs, "VpVs": vp / vs}


# ------------------------------------------------------------------ faults
def fault_tables():
    summ = pd.read_csv(FLT / "fault_surfaces_v1_summary.csv")
    mem = pd.read_csv(FLT / "fault_surfaces_v1_members.csv")
    vert = pd.read_csv(FLT_BIG / "fault_surfaces_v1_vertices.csv.gz")
    return summ, mem, vert


def surface_segments(summ, mem, seg_of_id: pd.Series) -> pd.DataFrame:
    """Segment of each fault surface = the segment of most of its fit members (v6.0 positions), with the share."""
    m = mem.copy()
    m["seg"] = m.event_id.map(seg_of_id)
    g = m.dropna(subset=["seg"]).groupby("surface_id").seg
    top = g.agg(lambda s: s.value_counts().index[0])
    share = g.agg(lambda s: s.value_counts(normalize=True).iloc[0])
    out = summ[["surface_id", "region", "n_events", "strike", "dip", "dip_depth_ok", "width_km", "length_km",
                "top_km", "bottom_km", "events_depth_ok_2pct_km", "events_depth_ok_98pct_km", "stable", "kind",
                "strand", "flags", "frac_depth_ok"]].copy()
    out["seg"] = out.surface_id.map(top)
    out["seg_share"] = out.surface_id.map(share)
    return out

#!/usr/bin/env python3
"""Compare earthquake catalogs across location and velocity-model iterations.

Every catalog listed in catalogs.yaml is read into one table layout, events are linked to the catalog v4
event IDs (directly, or by origin time for catalogs with their own IDs), and the same metrics are computed
for each catalog, overall and per region:

  counts       events, events linked to v4, relocated events (tier A-D or located), tier counts
  depth        10/50/90% depth (km below sea level); share above the local surface (topography/bathymetry)
  shift        horizontal and vertical change from the reference catalog (same v4 event), median and p90
  ANSS         ComCat events matched to the reference by time (<= 3 s) and distance (<= 50 km): median and
               spread of the horizontal offset and depth difference, for all matches and for "near" events
               (nearest station with a pick closer than max(depth, 5 km))
  faults       for each fault surface v0 (data/faults/v0, members = v4 event IDs), the RMS distance of its
               events to their best-fitting plane: a catalog that sharpens structure lowers it
  pairs        each catalog against its parent (shift statistics)

Outputs (small, in git): data/catalog_compare/
  summary.csv          one row per catalog x region (incl. "All")
  anss.csv             ANSS check per catalog x region x subset
  faults.csv           plane-fit thickness per catalog x surface;  faults_summary.csv: median per catalog
  pairs.csv            catalog vs parent and vs reference
  registry.json        catalogs.yaml as read, with row counts and file md5 (for the atlas and the paper)
  SUMMARY.md           tables for reading
Common tables (large, not in git): /wd1/mdenolle_data/catalog_compare/<id>.parquet

  python workflow/11_compare/compare_catalogs.py [--only ID ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
MAIN = Path("/home/mdenolle/cascadia_obs_ensemble")          # shared data tree (large files are gitignored)
OUT = REPO / "data" / "catalog_compare"
BIG = Path("/wd1/mdenolle_data/catalog_compare")
STATIONS = Path("/wd1/mdenolle_data/picks_share/stations/stations_ver5_v6.csv")
REGION_SRC = Path("/wd1/mdenolle_data/reloc_v5/catalog/catalog_v5_trackA_iter1_vpvs176.csv")
R_EARTH = 6371.0
COLS = ["event_id", "v4_id", "time", "lat", "lon", "depth", "tier", "located", "nphs", "gap", "rms"]


def path(p):
    p = Path(p)
    if p.is_absolute():
        return p
    for root in (REPO, MAIN):
        if (root / p).exists():
            return root / p
    return REPO / p


def md5(p, n=1 << 24):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while b := f.read(n):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ readers -> common layout
def read_v4(f):
    c = pd.read_csv(f, skipinitialspace=True)
    c.columns = [x.strip() for x in c.columns]
    return pd.DataFrame({"event_id": c.ID, "v4_id": c.ID, "time": pd.to_datetime(c.datetime.str.strip(), utc=True),
                         "lat": c.latitude, "lon": c.longitude, "depth": c["depth(km)"], "tier": "",
                         "located": True, "nphs": c.Num_p_picks + c.Num_s_picks, "gap": np.nan, "rms": np.nan})


def read_v3(f):
    c = pd.read_csv(f)
    return pd.DataFrame({"event_id": c.orid, "v4_id": np.nan, "time": pd.to_datetime(c.time, utc=True),
                         "lat": c.latitude, "lon": c.longitude, "depth": c.depth, "tier": "",
                         "located": True, "nphs": c.n_arrivals, "gap": c.gap, "rms": c.rms})


def read_nll_catalog(f):
    c = pd.read_csv(f)
    loc = c.loc_source.eq("nll3d")
    return pd.DataFrame({"event_id": c.ID, "v4_id": c.ID, "time": pd.to_datetime(c.time, utc=True, format="ISO8601"),
                         "lat": c.lat, "lon": c.lon, "depth": c.depth, "tier": c.tier.fillna(""),
                         "located": loc, "nphs": c.nphs, "gap": c.gap, "rms": c.rms})


def read_nll_run(f):
    c = pd.read_csv(f)
    ok = c.status.eq("LOCATED")
    return pd.DataFrame({"event_id": c.ID, "v4_id": c.ID, "time": pd.to_datetime(c.time, utc=True, format="ISO8601"),
                         "lat": c.lat, "lon": c.lon, "depth": c.depth, "tier": "", "located": ok,
                         "nphs": c.nphs, "gap": c.gap, "rms": c.rms})[ok.to_numpy()]


def read_tomodd_reloc(f):
    """tomoDD/hypoDD .reloc: ID LAT LON DEPTH X Y Z EX EY EZ YR MO DY HR MI SC MAG NCCP NCCS NCTP NCTS RCC RCT CID.
    Assumes the tomoDD event IDs are catalog v4 IDs (check b2_tomodd_inputs.py before registering a run)."""
    c = pd.read_csv(f, sep=r"\s+", header=None)
    t = pd.to_datetime(dict(year=c[12], month=c[13], day=c[14], hour=c[15], minute=c[16]), utc=True) + pd.to_timedelta(c[17], unit="s")
    return pd.DataFrame({"event_id": c[0], "v4_id": c[0], "time": t, "lat": c[1], "lon": c[2], "depth": c[3],
                         "tier": "", "located": True, "nphs": c[19] + c[20], "gap": np.nan, "rms": c[22]})


READERS = {"v4": read_v4, "v3_origins": read_v3, "nll_catalog": read_nll_catalog, "nll_run": read_nll_run,
           "tomodd_reloc": read_tomodd_reloc}


# ------------------------------------------------------------------ geometry helpers
def dist_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def xyz(lat, lon, dep, lat0=45.0, lon0=-125.25):
    return np.c_[np.radians(lon - lon0) * R_EARTH * np.cos(np.radians(lat0)), np.radians(lat - lat0) * R_EARTH, -dep]


def match_by_time(a, b, dt=3.0, dmax=50.0):
    """For each row of a, the index of the row of b closest in origin time within dt s and dmax km (or -1)."""
    tb = b.time.astype("int64").to_numpy() / 1e9
    o = np.argsort(tb); tb = tb[o]
    ta = a.time.astype("int64").to_numpy() / 1e9
    out = np.full(len(a), -1)
    best = np.full(len(a), np.inf)
    i0 = np.searchsorted(tb, ta - dt); i1 = np.searchsorted(tb, ta + dt)
    for k in range(int((i1 - i0).max()) if len(a) else 0):
        j = i0 + k
        ok = j < i1
        jj = o[np.minimum(j, len(o) - 1)]
        d = dist_km(a.lat.to_numpy(), a.lon.to_numpy(), b.lat.to_numpy()[jj], b.lon.to_numpy()[jj])
        dtt = np.abs(ta - tb[np.minimum(j, len(tb) - 1)])
        better = ok & (d <= dmax) & (dtt < best)
        out[better], best[better] = jj[better], dtt[better]
    return out


class Surface:
    """Topography/bathymetry (km above sea level) at any epicentre, from GMT earth_relief 01m."""

    def __init__(self):
        import pygmt
        z = pygmt.datasets.load_earth_relief(resolution="01m", region=[-133, -117, 37, 53], registration="gridline")
        self.lon, self.lat, self.z = z.lon.values, z.lat.values, z.values / 1000.0

    def __call__(self, lat, lon):
        i = np.clip(np.round((np.asarray(lat) - self.lat[0]) / (self.lat[1] - self.lat[0])).astype(int), 0, len(self.lat) - 1)
        j = np.clip(np.round((np.asarray(lon) - self.lon[0]) / (self.lon[1] - self.lon[0])).astype(int), 0, len(self.lon) - 1)
        return self.z[i, j]


# ------------------------------------------------------------------ shared inputs
def pick_stations():
    """Per v4 event: coordinates of the stations with picks (for the nearest-station distance)."""
    f = MAIN / "data" / "catalog_v4" / "Cascadia_picks_catalog_v4.csv"
    p = pd.read_csv(f, skipinitialspace=True, usecols=["event_ID", "station"])
    s = pd.read_csv(STATIONS)
    s["station_key"] = s.station + "." + s.network
    s = s.groupby("station_key")[["lat", "lon"]].median()
    p = p.join(s, on="station").dropna(subset=["lat"])
    return p.rename(columns={"event_ID": "v4_id", "lat": "slat", "lon": "slon"})[["v4_id", "slat", "slon"]]


def nearest_station(cat, ps):
    m = ps.merge(cat[["v4_id", "lat", "lon"]].dropna(subset=["v4_id"]).astype({"v4_id": int}), on="v4_id")
    m["d"] = dist_km(m.lat.to_numpy(), m.lon.to_numpy(), m.slat.to_numpy(), m.slon.to_numpy())
    return m.groupby("v4_id").d.min()


def regions():
    r = pd.read_csv(REGION_SRC, usecols=["ID", "region", "v4_lat", "v4_lon"])
    return r


def assign_region(cat, reg):
    """Region of the linked v4 event; events without a v4 link take the region of the nearest v4 epicentre."""
    from scipy.spatial import cKDTree
    out = cat.v4_id.map(reg.set_index("ID").region)
    miss = out.isna().to_numpy()
    if miss.any():
        tree = cKDTree(np.c_[reg.v4_lon * np.cos(np.radians(45)), reg.v4_lat])
        _, k = tree.query(np.c_[cat.lon[miss] * np.cos(np.radians(45)), cat.lat[miss]])
        out[miss] = reg.region.to_numpy()[k]
    return out.fillna("Other onshore")


# ------------------------------------------------------------------ metrics
def q(x, p):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.percentile(x, p)) if len(x) else np.nan


def summary_rows(cid, c, ref):
    rows = []
    r = ref.drop_duplicates("v4_id").set_index("v4_id")
    for reg, g in [("All", c)] + sorted(c.groupby("region")):
        loc = g[g.located]
        m = loc.dropna(subset=["v4_id"]).join(r[["lat", "lon", "depth"]], on="v4_id", rsuffix="_ref").dropna(subset=["lat_ref"])
        dh = dist_km(m.lat.to_numpy(), m.lon.to_numpy(), m.lat_ref.to_numpy(), m.lon_ref.to_numpy())
        dz = (m.depth - m.depth_ref).to_numpy()
        row = dict(catalog=cid, region=reg, n_events=len(g), n_linked_v4=int(g.v4_id.notna().sum()),
                   n_located=len(loc), depth_p10=q(loc.depth, 10), depth_p50=q(loc.depth, 50), depth_p90=q(loc.depth, 90),
                   pct_above_surface=100 * float(loc.above_surface.mean()) if len(loc) else np.nan,
                   n_vs_ref=len(m), dh_ref_p50=q(dh, 50), dh_ref_p90=q(dh, 90), dz_ref_p50=q(dz, 50),
                   dz_ref_iqr=q(dz, 75) - q(dz, 25), rms_p50=q(loc.rms, 50), nphs_p50=q(loc.nphs, 50))
        for t in "ABCDU":
            row[f"tier_{t}"] = int((g.tier == t).sum())
        rows.append(row)
    return rows


def anss_rows(cid, c, anss_link):
    """anss_link: v4_id -> ANSS lat, lon, depth (matched once on the reference)."""
    rows = []
    m = c[c.located].dropna(subset=["v4_id"]).join(anss_link, on="v4_id", rsuffix="_anss").dropna(subset=["lat_anss"])
    m["dh"] = dist_km(m.lat.to_numpy(), m.lon.to_numpy(), m.lat_anss.to_numpy(), m.lon_anss.to_numpy())
    m["dz"] = m.depth - m.depth_anss
    m["near"] = m.dist_min <= np.maximum(m.depth_anss.clip(lower=0), 5.0)
    for reg, g in [("All", m)] + sorted(m.groupby("region")):
        for sub, h in (("all", g), ("near", g[g.near])):
            mad = 1.4826 * np.median(np.abs(h.dz - h.dz.median())) if len(h) else np.nan
            rows.append(dict(catalog=cid, region=reg, subset=sub, n=len(h), dh_p50=q(h.dh, 50), dh_p90=q(h.dh, 90),
                             dz_p50=q(h.dz, 50), dz_mad=mad, abs_dz_p50=q(np.abs(h.dz), 50)))
    return rows


def fault_rows(cid, c, members, nmin=20):
    rows = []
    loc = c[c.located].dropna(subset=["v4_id"]).drop_duplicates("v4_id").set_index("v4_id")
    for sid, g in members.groupby("surface_id"):
        e = loc.reindex(g.event_id).dropna(subset=["lat"])
        if len(e) < nmin:
            continue
        X = xyz(e.lat.to_numpy(), e.lon.to_numpy(), e.depth.to_numpy())
        X = X - X.mean(0)
        _, s, vt = np.linalg.svd(X, full_matrices=False)
        d = X @ vt[2]
        n = vt[2] * np.sign(vt[2][2] or 1)
        dip = float(np.degrees(np.arccos(abs(n[2]))))
        rows.append(dict(catalog=cid, surface_id=sid, n=len(e), rms_plane_km=float(np.sqrt(np.mean(d ** 2))),
                         p50_abs_km=float(np.median(np.abs(d))), dip=dip))
    return rows


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=str(Path(__file__).with_name("catalogs.yaml")))
    ap.add_argument("--only", nargs="*", help="recompute these catalog ids only (others read from the parquet cache)")
    a = ap.parse_args()
    reg_y = yaml.safe_load(open(a.registry))
    cats = {e["id"]: e for e in reg_y["catalogs"]}
    ref_id = reg_y["reference"]
    OUT.mkdir(parents=True, exist_ok=True); BIG.mkdir(parents=True, exist_ok=True)

    surf = Surface()
    ps = pick_stations()
    reg = regions()
    members = pd.read_csv(path("data/faults/v0/fault_surfaces_v0_members.csv"))

    tables = {}
    order = [ref_id] + [k for k in cats if k != ref_id]
    for cid in order:
        e = cats[cid]
        f = path(e["file"])
        cache = BIG / f"{cid}.parquet"
        if a.only is not None and cid not in a.only and cid != ref_id and cache.exists():
            tables[cid] = pd.read_parquet(cache); continue
        if not f.exists():
            print(f"{cid}: {f} missing, skipped"); continue
        c = READERS[e["adapter"]](f)[COLS].reset_index(drop=True)
        if not e.get("keyed_v4", True):
            k = match_by_time(c, tables[ref_id])
            c["v4_id"] = np.where(k >= 0, tables[ref_id].v4_id.to_numpy()[np.maximum(k, 0)], np.nan)
        c["region"] = assign_region(c, reg).to_numpy()
        c["surface_km"] = surf(c.lat, c.lon)
        c["above_surface"] = c.depth < -c.surface_km
        c["dist_min"] = c.v4_id.map(nearest_station(c, ps))
        c.to_parquet(cache)
        tables[cid] = c
        e["_rows"], e["_md5"] = len(c), md5(f)
        print(f"{cid}: {len(c):,} events, {int(c.located.sum()):,} located, {int(c.v4_id.notna().sum()):,} linked to v4")

    ref = tables[ref_id]
    anss = pd.read_csv(path("data/magnitude_v4/anss_2010-16.csv"))
    anss = pd.DataFrame({"time": pd.to_datetime(anss.time, utc=True), "lat": anss.latitude, "lon": anss.longitude,
                         "depth": anss.depth})
    k = match_by_time(anss, ref)
    ok = k >= 0
    anss_link = pd.DataFrame({"lat": anss.lat[ok].to_numpy(), "lon": anss.lon[ok].to_numpy(), "depth": anss.depth[ok].to_numpy()},
                             index=pd.Index(ref.v4_id.to_numpy()[k[ok]], name="v4_id"))
    anss_link = anss_link[~anss_link.index.duplicated()]
    print(f"ANSS: {len(anss):,} events, {len(anss_link):,} matched to {ref_id}")

    S, A, F, P = [], [], [], []
    for cid, c in tables.items():
        S += summary_rows(cid, c, ref)
        A += anss_rows(cid, c, anss_link)
        F += fault_rows(cid, c, members)
        for other, kind in ((cats[cid].get("parent"), "parent"), (ref_id, "reference")):
            if other in tables and other != cid:
                o = tables[other][tables[other].located].dropna(subset=["v4_id"]).drop_duplicates("v4_id").set_index("v4_id")
                m = c[c.located].dropna(subset=["v4_id"]).join(o[["lat", "lon", "depth"]], on="v4_id", rsuffix="_o").dropna(subset=["lat_o"])
                dh = dist_km(m.lat.to_numpy(), m.lon.to_numpy(), m.lat_o.to_numpy(), m.lon_o.to_numpy())
                dz = (m.depth - m.depth_o).to_numpy()
                P.append(dict(catalog=cid, against=other, kind=kind, n=len(m), dh_p50=q(dh, 50), dh_p90=q(dh, 90),
                              dz_p50=q(dz, 50), dz_p10=q(dz, 10), dz_p90=q(dz, 90),
                              pct_dh_gt5=100 * float(np.mean(dh > 5)) if len(m) else np.nan))
    S, A, F, P = map(pd.DataFrame, (S, A, F, P))
    common = F.groupby("surface_id").catalog.nunique()
    common = common[common == F.catalog.nunique()].index
    FS = (F[F.surface_id.isin(common)].groupby("catalog")
          .agg(n_surfaces=("surface_id", "size"), rms_plane_p50=("rms_plane_km", "median"), p50_abs_p50=("p50_abs_km", "median"))
          .reset_index())
    for name, df in (("summary", S), ("anss", A), ("faults", F), ("faults_summary", FS), ("pairs", P)):
        df.to_csv(OUT / f"{name}.csv", index=False, float_format="%.3f")
    reg_out = {k: v for k, v in reg_y.items() if k != "catalogs"}
    reg_out["catalogs"] = [{kk: (str(vv) if kk == "date" else vv) for kk, vv in cats[c].items()} for c in order if c in tables]
    json.dump(reg_out, open(OUT / "registry.json", "w"), indent=1, default=str)
    write_md(order, tables, cats, S, A, FS, P, ref_id, len(anss_link))
    print(f"wrote {OUT}")


def write_md(order, tables, cats, S, A, FS, P, ref_id, n_anss):
    ids = [c for c in order if c in tables]
    L = ["# Catalog comparison", "", f"Reference: **{ref_id}**. Built by `workflow/11_compare/compare_catalogs.py` "
         f"from `catalogs.yaml`. Depths in km below sea level. ANSS events matched to {ref_id}: {n_anss:,}.", "",
         "| catalog | velocity model | method | status |", "|---|---|---|---|"]
    L += [f"| {c} | {cats[c]['velocity']} | {cats[c]['method']} | {cats[c]['status']} |" for c in ids]
    s = S[S.region == "All"].set_index("catalog").loc[ids]
    a = A[(A.region == "All")].pivot(index="catalog", columns="subset", values=["dz_p50", "dz_mad", "dh_p50", "n"]).reindex(ids)
    fs = FS.set_index("catalog").reindex(ids)
    L += ["", "## Whole catalog", "",
          "| catalog | events | located | depth p10/p50/p90 | above surface | shift from ref. h p50/p90, dz p50 | ANSS dz p50 (MAD), all / near | ANSS dh p50 | fault-plane RMS (km) |",
          "|---|---:|---:|---|---:|---|---|---:|---:|"]
    for c in ids:
        r = s.loc[c]
        L.append(f"| {c} | {r.n_events:,} | {r.n_located:,} | {r.depth_p10:.1f} / {r.depth_p50:.1f} / {r.depth_p90:.1f} | "
                 f"{r.pct_above_surface:.1f}% | {r.dh_ref_p50:.1f} / {r.dh_ref_p90:.1f}, {r.dz_ref_p50:+.1f} | "
                 f"{a.loc[c, ('dz_p50', 'all')]:+.1f} ({a.loc[c, ('dz_mad', 'all')]:.1f}) / {a.loc[c, ('dz_p50', 'near')]:+.1f} ({a.loc[c, ('dz_mad', 'near')]:.1f}) | "
                 f"{a.loc[c, ('dh_p50', 'all')]:.1f} | {fs.loc[c, 'rms_plane_p50']:.2f} |")
    L += ["", f"Fault-plane RMS: median over the {int(fs.n_surfaces.max()) if fs.n_surfaces.notna().any() else 0} fault surfaces "
          "with >= 20 located members in every catalog. The surfaces were fitted to v4 locations and their members chosen on v4, "
          "so this number favours v4; double-difference catalogs (v3, v4, tomoDD) are also sharper than single-event "
          "locations (NonLinLoc) by construction. Compare like with like.", "", "## By region: median depth / above surface / ANSS dz (near)", "",
          "| region | " + " | ".join(ids) + " |", "|---|" + "---|" * len(ids)]
    an = A[A.subset == "near"].set_index(["region", "catalog"])
    ss = S.set_index(["region", "catalog"])
    for reg in [r for r in S.region.unique() if r != "All"]:
        cells = []
        for c in ids:
            if (reg, c) not in ss.index:
                cells.append(""); continue
            r = ss.loc[(reg, c)]
            dz = an.loc[(reg, c), "dz_p50"] if (reg, c) in an.index else np.nan
            cells.append(f"{r.depth_p50:.1f} / {r.pct_above_surface:.0f}% / {dz:+.1f}")
        L.append(f"| {reg} | " + " | ".join(cells) + " |")
    L += ["", "## Each catalog against its parent", "", "| catalog | against | n | dh p50 / p90 | dz p10 / p50 / p90 | dh > 5 km |", "|---|---|---:|---|---|---:|"]
    for r in P.itertuples():
        L.append(f"| {r.catalog} | {r.against} ({r.kind}) | {r.n:,} | {r.dh_p50:.1f} / {r.dh_p90:.1f} | {r.dz_p10:+.1f} / {r.dz_p50:+.1f} / {r.dz_p90:+.1f} | {r.pct_dh_gt5:.0f}% |")
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

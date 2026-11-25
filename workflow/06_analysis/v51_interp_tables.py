#!/usr/bin/env python3
"""Catalog v5.1 re-interpretation (piece "interp"): region-by-region tables.

For each catalog region (column `region`, set from the v4 epicentre) this computes
  T1 location changes v4 -> v5.1 (counts, tiers, flags, depths, shifts, errors)
  T2 structure sharpness (nearest-neighbour distances on the same events, v4 against v5.1)
  T3 clustering classes above Mc (v5.1 and v4) and box-level robustness (verifier nulls, variants)
  T4 fault surfaces v1 per region (members, stability, verifier recovery, flags, activity)
  T5 swarms and sequences (families of the clusters piece with setting: plate boundary, volcano, surface)
  T6 depth against ANSS/ComCat (paired, v4 and v5.1)
  T7 named depth benchmarks from the literature (v5.1 numbers on boxes defined here)
Large joined event table: /wd1/.../v5.1_analysis/interp/events_joined.pkl. Small tables: data/catalog_v5/v5.1_analysis/interp/.

    python workflow/06_analysis/v51_interp_tables.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from v51_interp_common import (BIG, CLU, FLT, OUT, REGIONS, VER_C, V0, VOLCANOES, hav, load_events, load_pb_steps,
                               load_surfaces, local_xy, nearest_boundary, nearest_volcano, q, MAG)

OUT.mkdir(parents=True, exist_ok=True)
BIG.mkdir(parents=True, exist_ok=True)


def t1_locations(c):
    rows = []
    for r in REGIONS:
        g = c[c.region.eq(r)]
        L = g[g.located]
        E = g[g.is_eq]
        G = g[g.depth_good]
        tiers = L.tier.value_counts()
        row = dict(region=r, rows=len(g), located=len(L), no_hypocentre=int((~g.located).sum()),
                   non_earthquake=int((g.located & ~g.is_earthquake).sum()), earthquakes=len(E),
                   pctA=100 * tiers.get("A", 0) / max(len(L), 1), pctB=100 * tiers.get("B", 0) / max(len(L), 1),
                   pctC=100 * tiers.get("C", 0) / max(len(L), 1), pctD=100 * tiers.get("D", 0) / max(len(L), 1),
                   floor_limited=int(L.floor_limited.sum()), far_only=int(L.far_only.sum()),
                   deep_pass=int(L.deep_pass.sum()), five_phase=int(L.five_phase.sum()),
                   dq_ok=int(L.depth_quality.eq("ok").sum()),
                   dq_unconstrained=int(L.depth_quality.eq("unconstrained").sum()),
                   dq_biased_deep=int(L.depth_quality.eq("biased_deep").sum()),
                   depth_good=len(G),
                   v4_depth_med=q(E.v4_depth, 50), v4_depth_p10=q(E.v4_depth, 10), v4_depth_p90=q(E.v4_depth, 90),
                   depth_med=q(E.depth, 50), depth_p10=q(E.depth, 10), depth_p90=q(E.depth, 90),
                   bsf_med=q(E.bsf, 50), bsf_p10=q(E.bsf, 10), bsf_p90=q(E.bsf, 90),
                   good_depth_med=q(G.depth, 50), good_depth_p10=q(G.depth, 10), good_depth_p90=q(G.depth, 90),
                   good_v4_depth_med=q(G.v4_depth, 50), good_bsf_med=q(G.bsf, 50),
                   shift_h_med=q(E.shift_h_km, 50), shift_h_p90=q(E.shift_h_km, 90),
                   shift_z_med=q(E.shift_z_km, 50), shift_z_p10=q(E.shift_z_km, 10), shift_z_p90=q(E.shift_z_km, 90),
                   good_shift_z_med=q(G.shift_z_km, 50), good_shift_h_med=q(G.shift_h_km, 50),
                   sig_z_med=q(E.sig_z, 50), h_max_med=q(E.h_max, 50), dist_min_med=q(E.dist_min, 50),
                   surface_med=q(E.surface_km, 50),
                   v4_above_surface_pct=100 * float((E.v4_depth < -E.surface_km).mean()) if len(E) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def nn_k(x, y, z, k):
    pts = np.c_[x, y] if z is None else np.c_[x, y, z]
    tr = cKDTree(pts)
    d, _ = tr.query(pts, k=k + 1)
    return d[:, k]


def t2_sharpness(c):
    rows = []
    for r in REGIONS:
        G = c[c.depth_good & c.region.eq(r) & c.v4_depth.notna()]
        E = c[c.is_eq & c.region.eq(r)]
        row = dict(region=r, n_good=len(G), n_eq=len(E))
        if len(G) > 20:
            lat0, lon0 = G.lat.median(), G.lon.median()
            x5, y5 = local_xy(G.lat, G.lon, lat0, lon0)
            x4, y4 = local_xy(G.v4_lat, G.v4_lon, lat0, lon0)
            row.update(nn5_3d_v4=np.median(nn_k(x4, y4, G.v4_depth.values, 5)),
                       nn5_3d_v51=np.median(nn_k(x5, y5, G.depth.values, 5)),
                       nn1_epi_good_v4=np.median(nn_k(x4, y4, None, 1)),
                       nn1_epi_good_v51=np.median(nn_k(x5, y5, None, 1)),
                       depth_iqr_good_v4=q(G.v4_depth, 75) - q(G.v4_depth, 25),
                       depth_iqr_good_v51=q(G.depth, 75) - q(G.depth, 25))
        if len(E) > 20:
            # oct-tree lattice: events sharing an identical epicentre (NonLinLoc maximum-likelihood cell)
            dup = E.duplicated(["lat", "lon"], keep=False)
            U = E.drop_duplicates(["lat", "lon"])
            xu, yu = local_xy(U.lat, U.lon, U.lat.median(), U.lon.median())
            row.update(share_shared_epicentre=100 * float(dup.mean()),
                       share_shared_epicentre_AB=100 * float(dup[E.tier.isin(["A", "B"])].mean()) if E.tier.isin(["A", "B"]).any() else np.nan,
                       n_distinct_epicentres=len(U), n_distinct_depths=int(E.depth.round(4).nunique()),
                       nn1_distinct_epi_v51=np.median(nn_k(xu, yu, None, 1)) if len(U) > 2 else np.nan)
            lat0, lon0 = E.lat.median(), E.lon.median()
            x5, y5 = local_xy(E.lat, E.lon, lat0, lon0)
            x4, y4 = local_xy(E.v4_lat, E.v4_lon, lat0, lon0)
            row.update(nn1_epi_all_v4=np.median(nn_k(x4, y4, None, 1)), nn1_epi_all_v51=np.median(nn_k(x5, y5, None, 1)))
        rows.append(row)
    return pd.DataFrame(rows)


def t3_clusters(c):
    rows = []
    for r in REGIONS:
        A = c[c.region.eq(r) & c.in_analysis.eq(True)]
        A4 = c[c.region.eq(r) & c.in_analysis4.eq(True)]
        n = len(A)
        n4 = len(A4)
        cl = A.cls.value_counts()
        cl4 = A4.cls4.value_counts()
        rows.append(dict(region=r, n_above_Mc=n, pct_background=100 * cl.get("background", 0) / max(n, 1),
                         pct_MA=100 * cl.get("mainshock-aftershock", 0) / max(n, 1),
                         pct_swarm=100 * cl.get("swarm", 0) / max(n, 1),
                         pct_multiplet=100 * cl.get("multiplet", 0) / max(n, 1),
                         n_above_Mc_v4=n4, pct_background_v4=100 * cl4.get("background", 0) / max(n4, 1),
                         pct_swarm_v4=100 * cl4.get("swarm", 0) / max(n4, 1),
                         pct_MA_v4=100 * cl4.get("mainshock-aftershock", 0) / max(n4, 1),
                         tierD_share_above_Mc=100 * float(A.tier.eq("D").mean()) if n else np.nan,
                         bg_share_tierD=100 * float(A[A.tier.eq("D")].cls.eq("background").mean()) if A.tier.eq("D").any() else np.nan,
                         bg_share_tierABC=100 * float(A[A.tier.isin(list("ABC"))].cls.eq("background").mean()) if n else np.nan))
    t = pd.DataFrame(rows)
    fa = pd.read_csv(CLU / "families_zbz_v51.csv")
    fa5 = fa[fa.n.ge(5)]
    k = fa.groupby(["region_v51", "type"]).size().unstack(fill_value=0)
    k5 = fa5.groupby(["region_v51", "type"]).size().unstack(fill_value=0)
    t = t.merge(k.add_prefix("fam_").reset_index().rename(columns={"region_v51": "region"}), on="region", how="left")
    t = t.merge(k5.add_prefix("fam5_").reset_index().rename(columns={"region_v51": "region"}), on="region", how="left")
    return t.fillna({c_: 0 for c_ in t.columns if c_.startswith("fam")})


def t3b_boxes():
    """Box-level robustness: shuffled-to-real link ratio (verifier), variants (clusters piece)."""
    nul = pd.read_csv(VER_C / "v03_nulls_summary.csv")
    nul = nul[nul.null.eq("region")][["region", "frac_linked_mean", "real_frac_linked", "null_over_real_links",
                                       "D_max", "real_D"]]
    s = pd.read_csv(CLU / "sensitivity_zbz.csv")
    piv = s.pivot_table(index="region", columns="variant", values="pct_background").reset_index()
    keep = ["region", "main", "tierABC", "eta0_shuf5", "d3q", "v41M", "mc_best"]
    piv = piv[[k for k in keep if k in piv.columns]].rename(columns=lambda x: x if x == "region" else f"bg_{x}")
    v4 = pd.read_csv(V0.parents[1] / "catalog_v4" / "clusters_zbz" / "regions_zbz.csv")
    v4 = v4.rename(columns={"pct_background": "bg_v4"})[["region", "bg_v4"]] if "pct_background" in v4.columns else None
    out = nul.merge(piv, on="region", how="outer")
    if v4 is not None:
        out = out.merge(v4, on="region", how="left")
    return out


def surface_region(c, s):
    m = pd.read_csv(FLT / "fault_surfaces_v1_members.csv").rename(columns={"event_id": "ID"})
    m = m.merge(c[["ID", "region", "shift_z_km", "depth_quality", "depth", "v4_depth"]], on="ID", how="left")
    reg = m.groupby("surface_id").region.agg(lambda x: x.value_counts().index[0]).rename("cat_region")
    dz = m[m.depth_quality.eq("ok")].groupby("surface_id").shift_z_km.median().rename("members_dz_ok_med")
    nok = m[m.depth_quality.eq("ok")].groupby("surface_id").size().rename("members_n_ok")
    dz_all = m.groupby("surface_id").shift_z_km.median().rename("members_dz_med")
    s = s.merge(reg, on="surface_id", how="left").merge(dz, on="surface_id", how="left").merge(
        nok, on="surface_id", how="left").merge(
        dz_all, on="surface_id", how="left")
    return s


def interp_flags(s, c):
    """Flags required by the faults verifier, added to each surface (not changing the faults product)."""
    fl = []
    for _, r in s.iterrows():
        f = []
        if r.surface_id == "v1_GO022":
            f.append("Brookings 2012 swarm: members moved +30.6 km to 52.8 km, below the slab top (15-20 km, Slab2); depth suspect")
        if r.surface_id == "v1_ON028":
            f.append("42 events at 23-38 km under Mount St. Helens: same suspect deep population as removed v1x_06")
        big_ok = pd.notna(r.members_dz_ok_med) and abs(r.members_dz_ok_med) > 10 and r.members_n_ok >= 10
        big_all = pd.notna(r.median_shift_z_km) and abs(r.median_shift_z_km) > 10 and r.frac_depth_ok >= 0.5
        if (big_ok or big_all) and r.surface_id != "v1_GO022":
            f.append(f"members moved {r.median_shift_z_km:+.1f} km in depth v4->v5.1 (median; depth-ok members "
                     f"{r.members_dz_ok_med:+.1f} km)")
        if pd.notna(r.J_sub80) and r.J_sub80 < 0.5:
            f.append("not recovered on a random 80% subsample (verifier)")
        fl.append("; ".join(f))
    s["interp_flags"] = fl
    return s


def t4_faults(c, s):
    rows = []
    for r in REGIONS:
        S = s[s.cat_region.eq(r)]
        g = c[c.region.eq(r) & c.is_eq]
        fs = g[g.fit_like & ~g.region.eq("Axial")]
        on = g.assignment.isin(["fit member", "assigned"])
        rows.append(dict(region=r, surfaces=len(S), steep=int(S.dip.ge(45).sum()), members=int(S.n_events.sum()),
                         stable=int(S.stable.astype(str).eq("True").sum()),
                         recovered_sub80=int(S.J_sub80.ge(0.5).sum()),
                         recovered_tierAB=int(S.J_tierAB.ge(0.5).sum()),
                         depth_ok_majority=int(S.frac_depth_ok.ge(0.5).sum()),
                         earthquakes=len(g), on_fault=int(on.sum()),
                         ambiguous=int(g.assignment.eq("ambiguous").sum()),
                         none=int(g.assignment.eq("none").sum()),
                         not_fitted=int(g.assignment.eq("region not fitted (Axial)").sum()),
                         pct_on_fault=100 * float(on.mean()) if len(g) else np.nan,
                         fit_like=len(fs)))
    t = pd.DataFrame(rows)
    v0 = pd.read_csv(V0 / "fault_surfaces_v0_summary.csv", usecols=["surface_id", "rms_normal_km", "n_events", "region"])
    v0m = pd.read_csv(V0 / "fault_surfaces_v0_members.csv").rename(columns={"event_id": "ID"})
    v0m = v0m.merge(c[["ID", "region"]], on="ID", how="left")
    reg0 = v0m.groupby("surface_id").region.agg(lambda x: x.value_counts().index[0]).rename("cat_region")
    v0 = v0.merge(reg0, on="surface_id", how="left")
    t = t.merge(v0.groupby("cat_region").agg(surfaces_v0=("surface_id", "size"), members_v0=("n_events", "sum"),
                                             rms_normal_v0=("rms_normal_km", "median")).reset_index().rename(
        columns={"cat_region": "region"}), on="region", how="left")
    t = t.merge(s.groupby("cat_region").rms_normal_km.median().rename("rms_normal_v1").reset_index().rename(
        columns={"cat_region": "region"}), on="region", how="left")
    return t


def t5_families(c, pb):
    fa = pd.read_csv(CLU / "families_zbz_v51.csv")
    dpb, cpb, bpb = nearest_boundary(fa.lat_main.values, fa.lon_main.values, pb)
    dv, nv = nearest_volcano(fa.lat_main.values, fa.lon_main.values)
    fa["pb_km"], fa["pb_class"], fa["pb_bound"] = dpb, cpb, bpb
    fa["volc_km"], fa["volcano"] = dv, nv
    # v4 type of the main v4 family
    f4 = pd.read_csv(V0.parents[1] / "catalog_v4" / "clusters_zbz" / "families_zbz.csv", usecols=["family_id", "type", "n_all"])
    f4 = f4.rename(columns={"family_id": "v4_family_main", "type": "v4_type", "n_all": "v4_n_all"})
    fa = fa.merge(f4, on="v4_family_main", how="left")
    return fa


def t6_anss(c):
    f = "/wd1/mdenolle_data/catalog_v5/v5.1_analysis/mag/anss_2010-16.csv"
    a = pd.read_csv(f)
    a = a[a.type.eq("earthquake") & a.depth.notna()]
    # drop fixed depths (us/at with round values, or no depth error)
    fixed = a.depthError.isna() | a.depthError.le(0) | (a.net.isin(["us", "at", "pt"]) & a.depth.isin([0, 5, 10, 33, 35]))
    a = a[~fixed].copy()
    a["t"] = pd.to_datetime(a.time, utc=True, format="ISO8601")
    out = []
    for ver, tcol, lat, lon, dep in [("v5.1", "t", "lat", "lon", "depth"), ("v4", "t4", "v4_lat", "v4_lon", "v4_depth")]:
        E = c[c.is_eq & c[tcol].notna()].sort_values(tcol)
        te = E[tcol].values.astype("datetime64[ns]").astype(np.int64) / 1e9
        ta = a.t.values.astype("datetime64[ns]").astype(np.int64) / 1e9
        i = np.searchsorted(te, ta)
        best = np.full(len(a), -1)
        bestdt = np.full(len(a), 99.0)
        for off in (-2, -1, 0, 1):
            j = np.clip(i + off, 0, len(te) - 1)
            dt = np.abs(te[j] - ta)
            dd = hav(a.latitude.values, a.longitude.values, E[lat].values[j], E[lon].values[j])
            ok = (dt < 3) & (dd < 20) & (dt < bestdt)
            best[ok] = j[ok]
            bestdt[ok] = dt[ok]
        m = best >= 0
        x = E.iloc[best[m]][["ID", "region", "tier", dep, "depth_good", "dist_min"]].copy()
        x["anss_depth"] = a.depth.values[m]
        x["anss_net"] = a.net.values[m]
        x["dz"] = x[dep] - x.anss_depth
        x["ver"] = ver
        out.append(x.rename(columns={dep: "our_depth"}))
    P = pd.concat(out)
    rows = []
    for r in REGIONS:
        row = dict(region=r)
        for ver in ["v4", "v5.1"]:
            X = P[P.ver.eq(ver) & P.region.eq(r)]
            Xg = X[X.tier.isin(list("AB"))]
            row[f"n_{ver}"] = len(X)
            row[f"dz_med_{ver}"] = q(X.dz, 50)
            row[f"absdz_med_{ver}"] = q(X.dz.abs(), 50)
            row[f"n_AB_{ver}"] = len(Xg)
            row[f"dz_med_AB_{ver}"] = q(Xg.dz, 50)
        # paired on the same events
        both = P[P.ver.eq("v4")].merge(P[P.ver.eq("v5.1")], on="ID", suffixes=("_4", "_5"))
        b = both[both.region_5.eq(r)]
        row["n_paired"] = len(b)
        row["dz_paired_v4"] = q(b.dz_4, 50)
        row["dz_paired_v51"] = q(b.dz_5, 50)
        row["absdz_paired_v4"] = q(b.dz_4.abs(), 50)
        row["absdz_paired_v51"] = q(b.dz_5.abs(), 50)
        row["nets"] = ",".join(b.anss_net_5.value_counts().index[:3].astype(str)) if len(b) else ""
        rows.append(row)
    return pd.DataFrame(rows), P


def t7_benchmarks(c):
    """Named depth benchmarks (literature values from the depth-literature report; boxes defined here)."""
    E = c[c.is_eq]
    rows = []

    def add(name, sel, lit, datum, col="depth", note=""):
        X = E[sel]
        rows.append(dict(benchmark=name, n=len(X), v51_med=q(X[col], 50), v51_p10=q(X[col], 10),
                         v51_p90=q(X[col], 90), v4_med=q(X["v4_depth"] + (X["surface_km"] if col == "bsf" else 0), 50),
                         literature=lit, datum=datum, note=note))
    ac = E.tier.isin(list("ABC"))
    ab = E.tier.isin(list("AB"))
    # Axial
    sel = E.region.eq("Axial") & ab
    X = E[sel]
    rows.append(dict(benchmark="Axial caldera, tier A-B: share within 2.0 km below seafloor", n=len(X),
                     v51_med=100 * float((X.bsf <= 2.0).mean()), v51_p10=np.nan, v51_p90=np.nan,
                     v4_med=np.nan, literature="90 % (Waldhauser et al. 2020); 85 % within 1.6 km", datum="% of events",
                     note="v4-NLL3D 79 %; B4 at v0.3.0 12 %"))
    add("Axial caldera, tier A-B, depth below seafloor", sel, "0-2 km; ring faults to the reservoir roof at ~1.1-2.3 km", "bsf", col="bsf")
    endv = E.lat.between(47.85, 48.10) & E.lon.between(-129.2, -129.0)
    add("Endeavour axial box (47.85-48.10N, 129.2-129.0W), all tiers", endv, "1.5-3.5 km (Wilcock 2002; Van Ark 2007)", "bsf", col="bsf")
    sea = E.region.eq("Explorer") & E.surface_km.lt(-1.5)
    add("Explorer seaward of the front (seafloor deeper than 1.5 km)", sea, "5-22 km (Hutchinson et al. 2019, 2023)", "bsl")
    add("Explorer seaward, nearest station < 30 km", sea & E.dist_min.lt(30), "5-22 km", "bsl")
    jdf = E.region.eq("JdF ridge/plate")
    add("JdF ridge/plate, all tiers", jdf, "above the AMC at 2.0-2.7 km; crust 6.7-8 km (indirect)", "bsf", col="bsf")
    add("Blanco, tier A-B, X9 years (2012-09 to 2013-10)", E.region.eq("Blanco") & ab & E.t.between(pd.Timestamp("2012-09-01", tz="UTC"), pd.Timestamp("2013-11-01", tz="UTC")),
        "West 0-8, east 3-13 km bsf; Lange p5-p95 4.7-11.3 km", "bsf", col="bsf")
    gint = E.region.eq("Gorda") & E.lon.between(-127.0, -125.0) & E.lat.between(40.5, 42.5)
    add("Gorda interior (127-125W, 40.5-42.5N), nearest station < 10 km", gint & E.dist_min.lt(10), "faulting limit ~9.5-15 km bsf (thermal)", "bsf", col="bsf")
    add("Gorda interior, all", gint, "~9.5-15 km bsf", "bsf", col="bsf")
    gr = E.region.eq("Gorda") & E.pb_class.eq("OSR") & E.pb_km.lt(10) & ab
    add("Gorda Ridge axis (within 10 km of the PB2002 ridge), tier A-B", gr, "~3-7 km bsf (Jones & Johnson 1978)", "bsf", col="bsf")
    add("Gorda Ridge axis, tier A-B, nearest station < 15 km", gr & E.dist_min.lt(15), "~3-7 km bsf", "bsf", col="bsf")
    fort = hav(E.lat.values, E.lon.values, 40.6, -124.1) < 6
    add("Fortuna cluster (6 km of 40.6N 124.1W), tier A-C", fort & ac, "17-23 km (Alongi et al. 2021)", "not stated")
    pug = E.region.eq("Puget/Georgia") & ab & E.depth.lt(35) & E.lat.between(47.0, 48.0) & E.lon.between(-122.7, -121.9)
    add("Central Puget Lowland crust, tier A-B, depth < 35 km", pug, "15-25 km, mean 17.6 km (Blakely 2002)", "not stated (surface ~0)")
    msh = hav(E.lat.values, E.lon.values, 46.20, -122.19) < 10
    add("Mount St. Helens (10 km of summit), tier A-B", msh & ab, "-2.2 to 2 km bsl, deeper events 2-8 km from 2016 (Hirao 2025); paired +0.76 km (paired_v5.1.md)", "bsl")
    add("Mount St. Helens (10 km of summit), deeper than 30 km, all tiers", msh & E.depth.gt(30), "none expected; SHZ base ~16-20 km", "bsl")
    return pd.DataFrame(rows)


def msh_weekly(c):
    """Time-of-week of deep events near Mount St. Helens (faults piece finding)."""
    E = c[c.is_eq & (hav(c.lat.values, c.lon.values, 46.20, -122.19) < 15)].copy()
    E["dow"] = E.t.dt.dayofweek
    E["hm"] = E.t.dt.hour + E.t.dt.minute / 60
    E["deep"] = E.depth > 30
    wed = E.dow.eq(2) & E.hm.between(19 + 14 / 60, 19 + 20 / 60)
    return dict(n=len(E), deep=int(E.deep.sum()), deep_wed_window=int((E.deep & wed).sum()),
                shallow_wed_window=int((~E.deep & wed).sum()),
                deep_tiers=E[E.deep].tier.value_counts().to_dict(),
                deep_sigz_med=q(E[E.deep].sig_z, 50), deep_distmin_med=q(E[E.deep].dist_min, 50),
                deep_v4_depth_med=q(E[E.deep].v4_depth, 50), deep_depth_med=q(E[E.deep].depth, 50),
                deep_dow=E[E.deep].dow.value_counts().sort_index().to_dict(),
                deep_by_year=E[E.deep].t.dt.year.value_counts().sort_index().to_dict())


MODEL_NODES = [-3, -1, 1, 3, 5, 7, 9, 12, 15, 18, 21, 25, 29, 34, 40, 47, 55, 65, 80]


def t8_node_pileups(c):
    """Events piled on the depth nodes of model v0.3.1 (linear interpolation between nodes makes gradient kinks)."""
    E = c[c.is_eq]
    rows = []
    for r in REGIONS:
        g = E[E.region.eq(r)].depth.values
        for z in MODEL_NODES:
            win = np.abs(g - z) <= 0.25
            ring = (np.abs(g - z) > 0.25) & (np.abs(g - z) <= 1.25)
            exp = ring.sum() * 0.5 / 2.0        # ring is 2 km wide, window 0.5 km
            if win.sum() >= 30 and win.sum() > 1.5 * max(exp, 1):
                rows.append(dict(region=r, node_km=z, n_window=int(win.sum()), n_expected=round(float(exp), 1),
                                 excess=int(win.sum() - exp), excess_pct_of_region=100 * (win.sum() - exp) / len(g)))
    return pd.DataFrame(rows)


def region_extras(c):
    E = c[c.is_eq]
    x = {}
    ab = E.tier.isin(["A", "B"])
    A = E[E.region.eq("Axial")]
    for ver, lo in [("v4", "v4_lon"), ("v5.1", "lon")]:
        Aab = A[A.tier.isin(["A", "B"])]
        x[f"axial_east_west_ratio_{ver}"] = float((Aab[lo] > -130.0).sum() / max((Aab[lo] <= -130.0).sum(), 1))
    m = A.set_index("t").resample("MS").size()
    x["axial_monthly"] = {str(k)[:7]: int(v) for k, v in m.items() if v > 0}
    x["axial_bsf_AB_p90"] = q(A[A.tier.isin(["A", "B"])].bsf, 90)
    x["axial_floor_limited"] = int(A.floor_limited.sum())
    # Blanco segments (longitude bins of the depth-literature report)
    B = E[E.region.eq("Blanco")].copy()
    bins = [-180, -129.3, -128.75, -128.45, -127.1, 0]
    labs = ["West (WBD/EBD)", "Surveyor", "Cascadia Depression", "Blanco Ridge", "Gorda Depression"]
    B["seg"] = pd.cut(B.lon, bins, labels=labs)
    rows = []
    for sname, g in B.groupby("seg", observed=True):
        a = g[g.in_analysis.eq(True)]
        gg = g[g.depth_good]
        rows.append(dict(segment=sname, n=len(g), n_above_Mc=len(a),
                         pct_bg=100 * float(a.cls.eq("background").mean()) if len(a) else np.nan,
                         pct_swarm=100 * float(a.cls.eq("swarm").mean()) if len(a) else np.nan,
                         pct_MA=100 * float(a.cls.eq("mainshock-aftershock").mean()) if len(a) else np.nan,
                         n_good=len(gg), good_bsf_med=q(gg.bsf, 50), good_bsf_p90=q(gg.bsf, 90),
                         good_v4_med=q(gg.v4_depth + gg.surface_km, 50),
                         y2012=int(g.t.dt.year.eq(2012).sum()), y2013=int(g.t.dt.year.eq(2013).sum()),
                         y2014=int(g.t.dt.year.eq(2014).sum()), y2015=int(g.t.dt.year.eq(2015).sum())))
    pd.DataFrame(rows).to_csv(OUT / "blanco_segments.csv", index=False, float_format="%.2f")
    x["blanco_segments"] = rows
    # Puget: crust vs slab; near-surface events
    Pg = E[E.region.eq("Puget/Georgia")]
    a = Pg[Pg.in_analysis.eq(True) & Pg.depth_good]
    x["puget_bg_deep30"] = 100 * float(a[a.depth >= 30].cls.eq("background").mean())
    x["puget_bg_shallow30"] = 100 * float(a[a.depth < 30].cls.eq("background").mean())
    x["puget_n_deep30_good"] = int((Pg.depth_good & Pg.depth.ge(30)).sum())
    tl = E.t.dt.tz_convert("America/Los_Angeles")
    wd = (tl.dt.dayofweek < 5) & tl.dt.hour.between(8, 16)
    for r in ["Puget/Georgia", "Other onshore", "Cascades", "Offshore forearc"]:
        g = E.region.eq(r) & (E.bsf < 1.0)
        x[f"near_surface_{r}"] = dict(n=int(g.sum()), weekday_daytime=float(wd[g].mean()) if g.any() else np.nan,
                                      region_weekday_daytime=float(wd[E.region.eq(r)].mean()))
    # Gorda: ridge vs interior
    G = E[E.region.eq("Gorda")]
    ridge = G.pb_class.eq("OSR") & G.pb_km.lt(15)
    for lab, sel in [("ridge", ridge), ("interior", ~ridge)]:
        a = G[sel & G.in_analysis.eq(True)]
        x[f"gorda_{lab}"] = dict(n=int(sel.sum()), n_above_Mc=len(a),
                                 pct_swarm=100 * float(a.cls.eq("swarm").mean()) if len(a) else np.nan,
                                 pct_MA=100 * float(a.cls.eq("mainshock-aftershock").mean()) if len(a) else np.nan,
                                 pct_bg=100 * float(a.cls.eq("background").mean()) if len(a) else np.nan,
                                 depth_med=q(G[sel].depth, 50), bsf_near_station=q(G[sel & G.dist_min.lt(10)].bsf, 50),
                                 n_near_station=int((sel & G.dist_min.lt(10)).sum()))
    # Explorer / Endeavour / JdF: floor-limited and deep shares
    for r in ["Endeavour", "Explorer", "JdF ridge/plate"]:
        g = E[E.region.eq(r)]
        x[f"ridge_{r}"] = dict(n=len(g), floor_limited=int(g.floor_limited.sum()), deeper_8bsf=int((g.bsf > 8).sum()),
                              within_0p25_of_floor=int((g.bsf < 0.25).sum()), bsf_med=q(g.bsf, 50),
                              v4_bsf_med=q(g.v4_depth + g.surface_km, 50), sig_z_med=q(g.sig_z, 50),
                              dist_min_med=q(g.dist_min, 50), years=g.t.dt.year.value_counts().sort_index().to_dict())
    # Mendocino: slab double zone layers and depth vs ANSS by distance
    M = E[E.region.eq("Mendocino") & E.depth_good]
    x["mendocino_good_depth_hist"] = np.histogram(M.depth, bins=np.arange(0, 45, 2.5))[0].tolist()
    # Cascades: Mount St. Helens and Hood
    for v in ["Mount St. Helens", "Mount Hood", "Mount Rainier", "Mount Baker", "Mount Shasta", "Lassen Peak",
              "Glacier Peak", "Mount Adams", "Three Sisters", "Newberry", "Crater Lake", "Medicine Lake", "Mount McLoughlin",
              "Mount Jefferson"]:
        la, lo = VOLCANOES[v]
        d = hav(E.lat.values, E.lon.values, la, lo)
        g = E[d < 15]
        a = g[g.in_analysis.eq(True)]
        x[f"volcano_{v}"] = dict(n=len(g), n_AB=int(g.tier.isin(["A", "B"]).sum()), depth_med=q(g.depth, 50),
                                 depth_AB_med=q(g[g.tier.isin(["A", "B"])].depth, 50),
                                 v4_depth_med=q(g.v4_depth, 50), n_above_Mc=len(a),
                                 pct_bg=100 * float(a.cls.eq("background").mean()) if len(a) else np.nan,
                                 n_deeper_30=int((g.depth > 30).sum()), dist_min_med=q(g.dist_min, 50))
    return x


def main():
    c = load_events()
    pb = load_pb_steps()
    c["pb_km"], c["pb_class"], c["pb_bound"] = np.nan, None, None
    L = c.located.values
    idx = np.where(L)[0]
    for k in range(0, len(idx), 20000):
        ii = idx[k:k + 20000]
        d, cl, bd = nearest_boundary(c.lat.values[ii], c.lon.values[ii], pb)
        c.loc[c.index[ii], "pb_km"] = d
        c.loc[c.index[ii], "pb_class"] = cl
        c.loc[c.index[ii], "pb_bound"] = bd
    c.to_pickle(BIG / "events_joined.pkl")
    t1 = t1_locations(c)
    t2 = t2_sharpness(c)
    t3 = t3_clusters(c)
    t3b = t3b_boxes()
    s = surface_region(c, load_surfaces())
    s = interp_flags(s, c)
    t4 = t4_faults(c, s)
    fa = t5_families(c, pb)
    t6, P = t6_anss(c)
    t7 = t7_benchmarks(c)
    mw = msh_weekly(c)
    t8 = t8_node_pileups(c)
    t8.to_csv(OUT / "t8_node_pileups.csv", index=False, float_format="%.2f")
    print(t8.to_string())
    ex = region_extras(c)
    (OUT / "region_extras.json").write_text(json.dumps(ex, indent=1, default=str))
    print(json.dumps(ex, indent=1, default=str)[:6000])
    for name, t in [("t1_locations", t1), ("t2_sharpness", t2), ("t3_clusters", t3), ("t3b_boxes", t3b),
                    ("t4_faults", t4), ("t6_anss_depth", t6), ("t7_benchmarks", t7)]:
        t.to_csv(OUT / f"{name}.csv", index=False, float_format="%.3f")
    s.drop(columns=["normal_xyz"], errors="ignore").to_csv(OUT / "surfaces_v1_interp.csv", index=False, float_format="%.3f")
    fa.to_csv(OUT / "families_v51_setting.csv", index=False, float_format="%.4f")
    P.to_csv(BIG / "anss_pairs.csv.gz", index=False, float_format="%.3f")
    (OUT / "msh_weekly.json").write_text(json.dumps(mw, indent=1, default=str))
    print(t1.to_string())
    print(t2.to_string())
    print(t3.to_string())
    print(t3b.to_string())
    print(t4.to_string())
    print(t6.to_string())
    print(t7.to_string())
    print(mw)


if __name__ == "__main__":
    main()

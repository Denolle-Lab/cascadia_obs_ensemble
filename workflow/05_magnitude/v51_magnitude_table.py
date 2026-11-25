#!/usr/bin/env python3
"""
Catalog v5.1 magnitudes: one row per v5.1 event, joined to the v5.1 flags.

Reads the outputs of scale/run_magnitude_v51.sh (phase2 anchors, phase19 preferred
magnitude, the amplitude dataset) and writes:

  <data>/cascadia_catalog_v5.1_magnitudes.csv   every v5.1 row (172,342), columns below
  <data>/phase19_near_picks_v51.parquet         per-pick station ML, station term and
                                                residual of the phase19 inversion (with
                                                the pick-timing correction of the pick)
  <data>/phase19_station_terms_v51.csv          station-phase terms C (phase19)

Rules:
  * Located rows (loc_source nll3d / comcat) take the phase19 preferred magnitude
    (M, M_unc, M_type ML / Mw / Mw_cal) computed with their v5.1 hypocentre.
  * Located rows without a Route A ML (no ver6 picks with amplitudes: the v4.1
    additions 900001-900003) keep the ComCat tensor Mw carried by v4.1 (M_type Mw).
  * Tensor matches lost by the relocation (event matched to a ComCat tensor in v4.1, v5.1
    origin now beyond phase18's 5 s / 50 km): the v4.1 tensor is carried (mt_match "v4.1"),
    and M = tensor Mw when Mw >= 4.5, as phase19 does for its own matches. The event is
    the same set of picks; the tensor magnitude does not depend on our hypocentre.
  * Rows without a hypocentre (loc_source none): M is empty. The documented fallback
    ML_v4hyp is computed from their amplitudes at the v4 hypocentre with the v5.1
    phase19 station terms and anchor offset held fixed (median of the station
    magnitudes within 150 km, >= 3 picks); for those with a matched ComCat tensor
    in v4.1, Mw_mt_v4 is carried. These values are never written to M.
  * Explosions and quarry blasts keep their magnitude; is_earthquake is False for
    them (event_flag anss_explosion, anss_quarry_blast, anss_other_event,
    anss_sonic_boom, quarry_blast_likely). They are dropped from Mc and b fits.

  * ml_caution (rows with an ML): semicolon-separated reasons why the ML may not hold,
    empty when none applies:
      deep_pass      the hypocentre comes from the deep NonLinLoc pass;
      tierD_moved30  tier D and the hypocentre moved more than 30 km (3D) from v4;
      weak_terms     every 'near' pick of the event sits on a phase19 station-phase term
                     fitted from fewer than 20 picks (term weakly tied to the network).
    The first two make ML follow a poorly constrained distance; depth_quality (copied
    from the v5.1 catalog: ok, unconstrained, biased_deep) is the catalog's own flag.

Columns: ID, time, lat, lon, depth, depth_quality, loc_source, tier, region,
offshore (surface_km < 0), event_flag, is_earthquake, floor_limited, far_only, deep_pass,
five_phase, M, M_unc, M_type, ML, ML_unc, ML_method, ml_caution, n_ML (picks), n_sta_ML (stations), ML_inv,
ML_inv_unc, n_obs_inv, Mw_mt, comcat_id, mt_match (v5.1 = phase18 match on the v5.1
origin, v4.1 = carried), anchor_ml (ComCat ML of the anchor match),
anss_id, anss_mag, anss_magtype, anss_dt_s, anss_dist_km (nearest ANSS event of any
type, 15 s / 50 km, one ANSS event per catalog event), M_v4, M_type_v4, ML_v4,
ML_method_v4 (catalog v4.1 Route A, v4 hypocentres), ML_v4hyp, ML_v4hyp_unc,
n_ML_v4hyp, Mw_mt_v4 (fallback, rows without a hypocentre only).

Usage:
  python v51_magnitude_table.py --data /wd1/mdenolle_data/catalog_v5/v5.1_analysis/mag --suffix _v51
  # converged phase19 (D-B091), written next to nothing stored:
  python v51_magnitude_table.py --data /wd1/.../v5.1_analysis/mag --suffix _v51 \
      --pm <dir>/cascadia_catalog_M_v51_converged.csv --out <dir> --tag _converged \
      --v4-mags <dir>/cascadia_catalog_M_routeA_converged.csv
  --solver sets the phase19 solver of the near inversion redone here (default: phase19's,
  "exact"); --v4-mags replaces M_v4, M_type_v4, ML_v4, ML_method_v4 (otherwise copied from
  the v5.1 catalog, i.e. the stored v4.1 Route A values) by those of the given phase19 file.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import phase19_preferred_magnitude as p19
from phase18_moment_tensor_match import hutton_boore, RMAX

V51 = "/wd1/mdenolle_data/catalog_v5/v5.1/cascadia_catalog_v5.1.csv"
NONQUAKE = ("anss_explosion", "anss_quarry_blast", "anss_other_event", "anss_sonic_boom",
            "quarry_blast_likely")
R_EARTH_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def near_inversion(ds):
    """phase19.event_ml, returning the retained picks with residuals and the terms."""
    near = ds[ds.dist_hypo_km <= RMAX]
    cnt = near.sp.value_counts()
    near = near[near.sp.isin(cnt[cnt >= p19.MIN_STA_OBS].index)]
    M, C, res = p19.invert(near)
    mad = 1.4826 * np.median(np.abs(res - np.median(res)))
    near = near[np.abs(res) < p19.REJECT_MAD * mad]
    M, C, res = p19.invert(near)
    near = near.assign(res=res, C=C.reindex(near.sp).to_numpy(), M_ev=M.reindex(near.event_id).to_numpy())
    return near, M, C


def match_anss(cat, anss, dt_s=15.0, dist_km=50.0):
    """Each ANSS event -> the nearest (epicentral) located event within dt_s; one ANSS
    event per catalog event (the closest in space)."""
    ev = cat.dropna(subset=["t"]).sort_values("t")
    te = ev["t"].astype("int64").to_numpy() / 1e9
    ta = anss["t"].astype("int64").to_numpy() / 1e9
    rows = []
    for i in range(len(anss)):
        lo, hi = np.searchsorted(te, ta[i] - dt_s), np.searchsorted(te, ta[i] + dt_s)
        if hi <= lo:
            continue
        d = haversine_km(anss.latitude.iat[i], anss.longitude.iat[i],
                         ev.lat.to_numpy()[lo:hi], ev.lon.to_numpy()[lo:hi])
        j = int(np.argmin(d))
        if d[j] <= dist_km:
            rows.append(dict(ID=int(ev.ID.iat[lo + j]), anss_id=anss.id.iat[i],
                             anss_mag=anss.mag.iat[i], anss_magtype=anss.magType.iat[i],
                             anss_type=anss.type.iat[i],
                             anss_dt_s=round(float(te[lo + j] - ta[i]), 2),
                             anss_dist_km=round(float(d[j]), 1)))
    m = pd.DataFrame(rows)
    return m.sort_values("anss_dist_km").drop_duplicates("ID")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--suffix", default="_v51")
    ap.add_argument("--pm", default=None, help="phase19 output (default <data>/cascadia_catalog_M<suffix>.csv)")
    ap.add_argument("--out", default=None, help="output directory (default --data)")
    ap.add_argument("--tag", default="", help="appended to the output file stems")
    ap.add_argument("--solver", default=None, help="phase19 solver (default phase19's)")
    ap.add_argument("--v4-mags", default=None, help="phase19 file whose M, M_type, ML, ML_method replace the v4.1 columns")
    a = ap.parse_args(argv)
    D, S = a.data, a.suffix
    O, T = a.out or a.data, a.tag
    p19.D, p19.SUFFIX = D, S
    if a.solver:
        p19.SOLVER = a.solver

    cat = pd.read_csv(V51)
    pm = pd.read_csv(a.pm or f"{D}/cascadia_catalog_M{S}.csv")
    anchors = pd.read_csv(f"{D}/route_b_ml_anchors{S}.csv")
    inv = pd.read_csv(f"{D}/cascadia_catalog_ML{S}.csv", usecols=["event_id", "n_obs"])

    # ---- phase19 near inversion again: per-pick residuals, stations, terms
    ds = pd.read_csv(f"{D}/amp_distance_dataset{S}.csv",
                     usecols=["arid", "event_id", "station", "phase", "dist_hypo_km", "log10A"])
    ds["m"] = hutton_boore(ds.log10A, ds.dist_hypo_km)
    ds["sp"] = ds.station + "|" + ds.phase
    near, Mev, C = near_inversion(ds)
    gse = near.groupby("event_id").res
    se = 1.4826 * gse.apply(lambda v: np.median(np.abs(v - v.median()))) / np.sqrt(gse.size())
    ok = gse.size() >= 3
    an = anchors[["event_id", "ml"]].join(Mev[ok.reindex(Mev.index).fillna(False)].rename("m"),
                                          on="event_id", how="inner").join(se.rename("se"), on="event_id")
    off = float(np.median(an.ml - an.m))
    s_anchor = float(np.sqrt(max(np.var(an.ml - an.m - off) - np.mean(an.se ** 2), 0)))
    chk = pm.set_index("event_id").ML.reindex(Mev.index) - (Mev + off)
    print(f"anchor offset {off:+.3f}; check against phase19 ML (near): "
          f"max |diff| {np.nanmax(np.abs(chk[pm.set_index('event_id').ML_method.reindex(Mev.index) == 'near'])):.4f}")
    arr = pd.read_csv(f"{D}/arrivals_v51.csv", usecols=["arid", "pick_id", "timing_correction_s"])
    near = near.merge(arr, on="arid", how="left")
    near.to_parquet(f"{O}/phase19_near_picks{S}{T}.parquet", index=False)
    terms = C.rename("C").rename_axis("sp").reset_index()
    terms[["station", "phase"]] = terms.sp.str.split("|", expand=True)
    terms = terms.merge(near.groupby("sp").size().rename("n_obs").reset_index(), on="sp")
    terms[["station", "phase", "C", "n_obs"]].to_csv(f"{O}/phase19_station_terms{S}{T}.csv", index=False)
    nsta = near.assign(sta=near.station.str.split("@").str[0]).groupby("event_id").sta.nunique()

    # ---- per-event table
    out = cat[["ID", "time", "lat", "lon", "depth", "depth_quality", "loc_source", "tier", "region",
               "surface_km", "event_flag", "floor_limited", "far_only", "deep_pass", "five_phase",
               "shift_h_km", "shift_z_km"]].copy()
    out["offshore"] = pd.Series(np.where(out.surface_km < 0, "True", "False"), index=out.index).where(
        out.surface_km.notna(), "")
    flag = out.event_flag.fillna("")
    out["is_earthquake"] = ~flag.apply(lambda s: any(k in s.split(";") for k in NONQUAKE))
    v4 = cat[["ID", "M", "M_type", "ML", "ML_method", "Mw_mt", "comcat_id"]].rename(
        columns={"M": "M_v4", "M_type": "M_type_v4", "ML": "ML_v4", "ML_method": "ML_method_v4",
                 "Mw_mt": "Mw_mt_v4", "comcat_id": "comcat_id_v4"})
    if a.v4_mags:
        r4 = pd.read_csv(a.v4_mags, usecols=["event_id", "M", "M_type", "ML", "ML_method"]).set_index("event_id")
        for c_ in ("M", "M_type", "ML", "ML_method"):
            v4[f"{c_}_v4"] = v4.ID.map(r4[c_])
        print(f"v4.1 columns M, M_type, ML, ML_method from {a.v4_mags} ({int(v4.M_v4.notna().sum()):,} with M)")
    pmc = pm.rename(columns={"event_id": "ID"})[
        ["ID", "M", "M_unc", "M_type", "ML", "ML_unc", "n_ML", "ML_method", "Mw_mt", "comcat_id",
         "ML_inv", "ML_inv_unc"]]
    out = out.merge(pmc, on="ID", how="left").merge(v4, on="ID", how="left")
    out = out.merge(inv.rename(columns={"event_id": "ID", "n_obs": "n_obs_inv"}), on="ID", how="left")
    out["n_sta_ML"] = out.ID.map(nsta)
    out = out.merge(anchors[["event_id", "ml"]].rename(columns={"event_id": "ID", "ml": "anchor_ml"}),
                    on="ID", how="left")

    located = out.loc_source.isin(["nll3d", "comcat"])
    # located, no Route A ML, tensor Mw carried by v4.1 (900001-900003)
    carry = located & out.M.isna() & out.M_type_v4.eq("Mw")
    out.loc[carry, "M"] = out.loc[carry, "M_v4"]
    out.loc[carry, "M_unc"] = p19.MW_MT_UNC
    out.loc[carry, "M_type"] = "Mw"
    out.loc[carry, "Mw_mt"] = out.loc[carry, "Mw_mt_v4"]
    out.loc[carry, "comcat_id"] = out.loc[carry, "comcat_id_v4"]
    print(f"located without Route A ML, tensor Mw carried from v4.1: {int(carry.sum())} "
          f"({out.loc[carry, 'ID'].tolist()})")
    out["mt_match"] = np.where(out.Mw_mt.notna(), "v5.1", "")
    out.loc[carry, "mt_match"] = "v4.1"
    # tensor association lost by the relocation: the event (same picks) was matched in v4.1
    # but its v5.1 origin is now beyond 5 s / 50 km of the ComCat origin. The tensor still
    # belongs to the event; carry it (unless phase18 gave that tensor to another event).
    lost = (located & out.Mw_mt.isna() & out.Mw_mt_v4.notna()
            & ~out.comcat_id_v4.isin(out.comcat_id.dropna()))
    out.loc[lost, "Mw_mt"] = out.loc[lost, "Mw_mt_v4"]
    out.loc[lost, "comcat_id"] = out.loc[lost, "comcat_id_v4"]
    out.loc[lost, "mt_match"] = "v4.1"
    big = lost & (out.Mw_mt >= p19.M_SWITCH)
    out.loc[big, "M"] = out.loc[big, "Mw_mt"]
    out.loc[big, "M_unc"] = p19.MW_MT_UNC
    out.loc[big, "M_type"] = "Mw"
    print(f"tensor matches lost by the relocation, carried from v4.1: {int(lost.sum())} "
          f"({out.loc[lost, 'ID'].tolist()}); of them Mw >= {p19.M_SWITCH}: {int(big.sum())}")
    assert out.loc[~located, "M"].isna().all()

    # ---- fallback for rows without a hypocentre (v4 hypocentre, fixed terms)
    fb_path = f"{D}/amp_distance_dataset_nohypo_v4hyp.csv"
    out["ML_v4hyp"] = np.nan
    out["ML_v4hyp_unc"] = np.nan
    out["n_ML_v4hyp"] = 0
    if os.path.exists(fb_path):
        fb = pd.read_csv(fb_path, usecols=["event_id", "station", "phase", "dist_hypo_km", "log10A"])
        fb["m"] = hutton_boore(fb.log10A, fb.dist_hypo_km)
        fb["sp"] = fb.station + "|" + fb.phase
        fb = fb[(fb.dist_hypo_km <= RMAX) & fb.sp.isin(C.index)]
        fb["mc"] = fb.m - C.reindex(fb.sp).to_numpy()
        g = fb.groupby("event_id").mc
        f = pd.DataFrame({"ML_v4hyp": g.median() + off, "n": g.size(),
                          "se": 1.4826 * g.apply(lambda v: np.median(np.abs(v - v.median()))) / np.sqrt(g.size())})
        f = f[f.n >= 3]
        f["ML_v4hyp_unc"] = np.sqrt(f.se ** 2 + s_anchor ** 2)
        idx = out.ID.map(f.ML_v4hyp)
        out["ML_v4hyp"] = idx.where(~located)
        out["ML_v4hyp_unc"] = out.ID.map(f.ML_v4hyp_unc).where(~located)
        out["n_ML_v4hyp"] = out.ID.map(f.n).where(~located).fillna(0).astype(int)
    nh = ~located
    print(f"rows without a hypocentre: {int(nh.sum())}; fallback ML_v4hyp: "
          f"{int(out.loc[nh, 'ML_v4hyp'].notna().sum())}; with a v4.1 tensor Mw: "
          f"{int((nh & out.M_type_v4.eq('Mw')).sum())}; had a v4.1 M: {int(out.loc[nh, 'M_v4'].notna().sum())}")

    # ---- ml_caution: hypocentre-sensitive or weakly tied ML (see docstring)
    has_ml = out.ML.notna()
    moved = np.hypot(out.shift_h_km, out.shift_z_km)
    nterm = near.groupby("sp").size()
    weak_ev = near.assign(w=near.sp.map(nterm) < 20).groupby("event_id").w.all()
    reasons = {
        "deep_pass": has_ml & out.deep_pass.astype(str).eq("True"),
        "tierD_moved30": has_ml & out.tier.eq("D") & (moved > 30.0),
        "weak_terms": has_ml & out.ID.map(weak_ev).fillna(False).astype(bool),
    }
    out["ml_caution"] = ""
    for k, s in reasons.items():
        cur = out.loc[s, "ml_caution"]
        out.loc[s, "ml_caution"] = cur.where(cur.eq(""), cur + ";") + k
    print("ml_caution: " + ", ".join(f"{k} {int(s.sum()):,}" for k, s in reasons.items())
          + f"; any {int(out.ml_caution.ne('').sum()):,} of {int(has_ml.sum()):,} rows with ML")
    print(f"depth_quality: {out.depth_quality.value_counts().to_dict()}")

    # ---- nearest ANSS event (any type) for the ComCat comparison
    anss = pd.read_csv(f"{D}/anss_2010-16.csv", index_col=0)
    anss["t"] = pd.to_datetime(anss["time"], format="%Y-%m-%dT%H:%M:%S.%fZ", utc=True, errors="coerce")
    anss = anss.dropna(subset=["t", "mag"])
    c = out[located].copy()
    c["t"] = pd.to_datetime(c.time, utc=True, format="ISO8601")
    m = match_anss(c, anss)
    out = out.merge(m, on="ID", how="left")
    print(f"ANSS events matched (any type): {len(m):,}")

    cols = ["ID", "time", "lat", "lon", "depth", "depth_quality", "loc_source", "tier", "region", "offshore",
            "event_flag", "is_earthquake", "floor_limited", "far_only", "deep_pass", "five_phase",
            "M", "M_unc", "M_type", "ML", "ML_unc", "ML_method", "ml_caution", "n_ML", "n_sta_ML",
            "ML_inv", "ML_inv_unc", "n_obs_inv", "Mw_mt", "comcat_id", "mt_match", "anchor_ml",
            "anss_id", "anss_mag", "anss_magtype", "anss_type", "anss_dt_s", "anss_dist_km",
            "M_v4", "M_type_v4", "ML_v4", "ML_method_v4",
            "ML_v4hyp", "ML_v4hyp_unc", "n_ML_v4hyp", "Mw_mt_v4"]
    out["Mw_mt_v4"] = out.Mw_mt_v4.where(~located)
    out = out[cols]
    for c_ in ("n_ML", "n_sta_ML", "n_obs_inv"):
        out[c_] = out[c_].astype("Int64")
    fo = f"{O}/cascadia_catalog_v5.1_magnitudes{T}.csv"
    out.round({"M": 3, "M_unc": 3, "ML": 3, "ML_unc": 3, "ML_inv": 3, "ML_inv_unc": 3,
               "ML_v4hyp": 3, "ML_v4hyp_unc": 3}).to_csv(fo, index=False)
    print(f"wrote {fo}: {len(out):,} rows; with M {int(out.M.notna().sum()):,}; "
          f"M_type {out.M_type.value_counts().to_dict()}; ML_method {out.ML_method.value_counts().to_dict()}")


if __name__ == "__main__":
    main()

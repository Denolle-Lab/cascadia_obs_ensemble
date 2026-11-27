"""Per-segment summaries of the external geophysical data and first-look comparisons (D-B114).

Cheap to rerun when the segment definitions change: it joins the per-event table written by eg_build.py
(derived/events_external_geo_<tag>.parquet) to the paper-plan event table (EXTGEO_SEGTABLE: ID, seg, seg_type,
depth_good, bsf, is_eq) on ID, and the seismicity metrics (EXTGEO_METRICS: seg, pct_swarm, ...) on seg.

Writes derived/segments_external_geo_<tag>.csv (one row per segment), derived/first_look_<tag>.csv and a copy of both
in the repository (data/external_geo/, small tables).

Definitions
  heat-flow coverage   GHFDB R2024 sites whose horizontal distance to the nearest event of the segment is <= 20 km
                       (hf_sites20) and <= 50 km (hf_sites50); their median / 25th / 75th percentile value
  values at events     medians over the segment's events (all located events unless stated)
  deepest 10 %         depth_good earthquakes; bsf_p90 = 90th percentile of depth below the seafloor; the isotherm
                       depths are the medians at the events deeper than bsf_p90; frac_below_iso600 = share of
                       depth_good events deeper (bsf) than the 600 degC isotherm (bsf) at their own epicentre
  fabric               axial mean of fabric_az20 (double-angle), weighted by aniso20; R = resultant length (0-1)
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))   # run with python -I (downloads are untrusted data)
import eg_common as C  # noqa: E402

METRICS = Path(os.environ.get("EXTGEO_METRICS", "/wd1/mdenolle_data/paper_plan/v6.1/tables/seismicity_metrics.csv"))
REPO_OUT = Path(__file__).resolve().parents[2] / "data/external_geo"
OFFSHORE_TYPES = ("ridge", "transform", "intraplate", "triple junction")


def q(x, p):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return np.percentile(x, p) if x.size else np.nan


def axial_mean(az, w):
    ok = np.isfinite(az) & np.isfinite(w)
    if not ok.any():
        return np.nan, np.nan
    a, w = np.radians(2 * az[ok]), w[ok]
    c, s = (w * np.cos(a)).sum() / w.sum(), (w * np.sin(a)).sum() / w.sum()
    return (np.degrees(np.arctan2(s, c)) / 2) % 180, float(np.hypot(c, s))


def main():
    ev = pd.read_parquet(C.DER / f"events_external_geo_{C.CATTAG}.parquet")
    seg = pd.read_parquet(C.SEGTABLE, columns=["ID", "seg", "seg_type", "depth_good", "is_eq", "bsf"])
    d = ev.drop(columns=["bsf"]).merge(seg, on="ID", how="inner")
    C.log(f"joined {len(d)} of {len(ev)} events to {C.SEGTABLE.name}; segments {d.seg.nunique()}")
    sites = pd.read_csv(C.DER / "ghfdb_sites_area.csv")
    S = C.unit_xyz(sites.lon.values, sites.lat.values)
    met = pd.read_csv(METRICS) if METRICS.exists() else pd.DataFrame(columns=["seg"])

    rows, fl = [], []
    for s, g in d.groupby("seg", sort=False):
        r = dict(seg=s, seg_type=g.seg_type.iloc[0], n_events=len(g), n_eq=int(g.is_eq.sum()),
                 n_depth_good=int(g.depth_good.sum()))
        tree = cKDTree(C.unit_xyz(g.lon.values, g.lat.values))
        dist = C.chord_to_km(tree.query(S, k=1)[0])
        for R in (20, 50):
            m = dist <= R
            r[f"hf_sites{R}"] = int(m.sum())
            r[f"hf_med{R}"], r[f"hf_p25_{R}"], r[f"hf_p75_{R}"] = (q(sites.q[m], p) for p in (50, 25, 75))
        r["hf_refs20"] = ";".join(sites.ref[dist <= 20].value_counts().index[:5].astype(str))
        r["frac_ev_hf_smooth"] = float(np.isfinite(g.hf_smooth).mean())
        r["frac_ev_oceanic_age"] = float(np.isfinite(g.age_myr).mean())
        for k in ("age_myr", "age_min10", "age_max10", "age_misfit_myr", "full_rate_mmyr", "sed_thk_m",
                  "crust1_thk_km", "water_depth_km", "rough5", "rough10", "rough20", "directfrac10", "landfrac20",
                  "aniso20", "hf_smooth", "q_gdh1", "q_hsc", "iso600_hsc_bsf", "iso600_plate_bsf", "slab_km",
                  "Tslab_D80", "Tslab_D80new"):
            r[f"{k}_med"] = q(g[k], 50)
        r["age_myr_p10"], r["age_myr_p90"] = q(g.age_myr, 10), q(g.age_myr, 90)
        r["hf_over_gdh1_med"] = q(g.hf_smooth / g.q_gdh1, 50)
        r["fabric_az20_mean"], r["fabric_R"] = axial_mean(g.fabric_az20.values.astype(float), g.aniso20.values.astype(float))
        # deepest 10 % of depth_good earthquakes vs isotherms from age
        gd = g[g.depth_good & g.is_eq & np.isfinite(g.bsf)]
        r["n_dg_eq"] = len(gd)
        if len(gd) >= 20:
            p90 = q(gd.bsf, 90)
            deep = gd[gd.bsf >= p90]
            r["bsf_p50"], r["bsf_p90"] = q(gd.bsf, 50), p90
            for m in ("hsc", "plate"):
                r[f"iso600_{m}_deep10_med"] = q(deep[f"iso600_{m}_bsf"], 50)
                r[f"iso800_{m}_deep10_med"] = q(deep[f"iso800_{m}_bsf"], 50)
                ok = np.isfinite(gd[f"iso600_{m}_bsf"])
                r[f"frac_below_iso600_{m}"] = float((gd.bsf[ok] > gd[f"iso600_{m}_bsf"][ok]).mean()) if ok.any() else np.nan
            # isotherm from the older side of a +-10 km box (transforms juxtapose two ages)
            r["iso600_hsc_agemax10_deep10_med"] = q(C.iso_hsc(deep.age_max10.values, 600), 50)
            r["iso600_plate_agemax10_deep10_med"] = q(C.iso_plate(deep.age_max10.values, 600), 50)
            r["age_deep10_med"] = q(deep.age_myr, 50)
            r["age_max10_deep10_med"] = q(deep.age_max10, 50)
            r["frac_deep10_oceanic_age"] = float(np.isfinite(deep.age_myr).mean())
        rows.append(r)
    out = pd.DataFrame(rows)
    keep = [c for c in ("seg", "name", "pct_swarm", "pct_swarm_lo", "pct_swarm_hi", "swarm_share_of_clustered",
                        "pct_MA", "pct_bg", "n_zbz") if c in met.columns]
    if len(keep) > 1:
        out = out.merge(met[keep], on="seg", how="left")
    order = {k: i for i, k in enumerate(met.seg)} if "seg" in met else {}
    out = out.sort_values("seg", key=lambda s: s.map(order).fillna(99)).reset_index(drop=True)
    f = C.DER / f"segments_external_geo_{C.CATTAG}.csv"
    out.to_csv(f, index=False, float_format="%.4g")
    C.log(f"wrote {f}")

    # ---------------------------------------------------------------- first-look comparisons
    fl = []
    for s in ("GP", "BL", "MFZ", "GR", "JR", "AX", "MTJ"):
        r = out[out.seg == s]
        if r.empty or "bsf_p90" not in r or not np.isfinite(r.bsf_p90.iloc[0]):
            continue
        r = r.iloc[0]
        fl.append(dict(test="deep10_vs_iso600", seg=s, n=r.n_dg_eq, bsf_p50=r.bsf_p50, bsf_p90=r.bsf_p90,
                       age_deep10_med=r.age_deep10_med, age_max10_deep10_med=r.age_max10_deep10_med,
                       iso600_hsc=r.iso600_hsc_deep10_med, iso600_plate=r.iso600_plate_deep10_med,
                       iso800_hsc=r.iso800_hsc_deep10_med, iso600_hsc_agemax10=r.iso600_hsc_agemax10_deep10_med,
                       iso600_plate_agemax10=r.iso600_plate_agemax10_deep10_med,
                       frac_below_iso600_hsc=r.frac_below_iso600_hsc, frac_below_iso600_plate=r.frac_below_iso600_plate,
                       frac_deep10_oceanic_age=r.frac_deep10_oceanic_age))
    if "pct_swarm" in out:
        off = out[out.seg_type.isin(OFFSHORE_TYPES) & np.isfinite(out.pct_swarm)]
        for x in ("hf_med20", "hf_med50", "hf_smooth_med", "hf_over_gdh1_med", "rough5_med", "rough10_med",
                  "rough20_med", "age_myr_med", "sed_thk_m_med"):
            for ycol in ("pct_swarm", "swarm_share_of_clustered"):
                if ycol not in off:
                    continue
                m = np.isfinite(off[x]) & np.isfinite(off[ycol])
                if m.sum() >= 4:
                    rho, p = spearmanr(off[x][m], off[ycol][m])
                    fl.append(dict(test=f"spearman_{ycol}_vs_{x}", n=int(m.sum()), rho=rho, p=p,
                                   segs=",".join(off.seg[m])))
    fl = pd.DataFrame(fl)
    f2 = C.DER / f"first_look_{C.CATTAG}.csv"
    fl.to_csv(f2, index=False, float_format="%.4g")
    C.log(f"wrote {f2}")
    REPO_OUT.mkdir(parents=True, exist_ok=True)
    for src in (f, f2):
        shutil.copy(src, REPO_OUT / src.name)
    return out, fl


if __name__ == "__main__":
    main()

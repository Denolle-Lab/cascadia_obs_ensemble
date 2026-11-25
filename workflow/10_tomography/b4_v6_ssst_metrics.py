"""Track B, catalog v6.0 preparation (D-B092): scores of the NLL-SSST test (b4_v6_ssst.py) on the D-B078 clusters.

Per set and arm (v5.1 catalog, static, ssst it0..itN, coherence), LOCATED events only, depths in the true frame
(Earth-flattening undone, b4_catalog.unflatten):
* residuals: rms of the NLLoc residuals (arrivals with weight > 0) pooled by phase and per station-phase (median over
  station-phases with >= 10 arrivals); the SSST arms fit these residuals by construction, so they are reported, not
  used to rank;
* CC consistency (independent of every arm's fit): for each accepted CC row (cc_<set>.parquet: above, same rate, no
  station-clock epoch inside the pair) at a station-phase used by both events, r = [dt_cc + T0v51_1 - T0v51_2] -
  [OT_1 - OT_2] - [(TTpred + Tcorr)_1 - (TTpred + Tcorr)_2]: the observed differential arrival time (CC, on the v5.1
  origin times the CC files use) minus the arm's predicted one. r_pair = r minus the pair's median r (origin-time
  offset), rms over rows = how well the arm's relative positions explain the CC times; the same with the catalog dt
  (dt_cat) of the same rows for reference;
* sharpness: median 3D distance of the CC repeater pairs (qc/<set>_repeaters.csv: median CC >= 0.95 over >= 6
  station-phases, real separations ~0.1 km), median nearest-neighbour distance, robust spread (MAD) in x, y, z;
* depths against the literature (pair tables of b4_lit_paired.py: Hirao et al. 2025 at MSH, Lange et al. 2026 at
  Blanco, km bsl): median and p10/p90 of catalog - published depth, tiers A-D and A-B;
* floor and surface: located events above the merged solid surface (D-B055, must be 0), floor_limited (maximum
  likelihood within 1.5 final oct-tree cells of the NLLoc floor, as b4_catalog.py);
* Axial: share of tier A-B events within 1.6 and 2.0 km below the seafloor (v5.1: 81 % within 2 km; target 85 %).
Tiers: a2_build_catalog.tier with the run's own values.

  python b4_v6_ssst_metrics.py [--grids g51] [--sets axial,blanco,msh,jdf]
      -> /wd1/mdenolle_data/reloc_v6/ssst_test/<grids>/metrics_<set>.json and scores_<grids>.csv
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "08_relocation"))
import a2_parse_hyp as PH  # noqa: E402
from a2_build_catalog import tier  # noqa: E402
from b4_catalog import floor_at, read_floor, unflatten, surface_km  # noqa: E402
from b4_v6_ssst import OUT, SETS, SCHEDULE, CC_DIR, SETS_DIR, FLOOR, lambert  # noqa: E402

V51_CAT = Path("/wd1/mdenolle_data/reloc_v5/catalog/catalog_b4_b4_v51_iter1.csv")
V51_LOC = Path("/wd1/mdenolle_data/reloc_v5/runs/b4_v51_iter1/locations.csv")
PAIRS = Path("/wd1/mdenolle_data/tomo_v1/depth_literature/pairs")
FLOOR_CELLS = 1.5
LIT = {"msh": ("Hirao et al. 2025", "msh_hirao2025_pairs.csv"), "blanco": ("Lange et al. 2026", "blanco_lange2026_abs_pairs.csv")}


def read_event_file(f):
    """(event row, phase rows) of one per-event NLLoc .hyp file."""
    rows = PH.parse(f)
    if len(rows) != 1:
        return None, []
    ev = rows[0]
    ph, on = [], False
    for line in open(f):
        if line.startswith("PHASE "):
            on = True
            continue
        if line.startswith("END_PHASE"):
            break
        if on and ">" in line:
            a, b = line.split(">")
            t, u = a.split(), b.split()
            ph.append(dict(ID=ev.get("ID"), sta=t[0], phase=t[4], ttpred=float(u[0]), res=float(u[1]),
                           weight=float(u[2]), sdist=float(u[6]), tcorr=float(u[11])))
    return ev, ph


def read_arm(rd):
    """events and phases of an NLLoc run folder (per-event files of job*/ or e*/)."""
    files = [f for f in sorted(rd.glob("*/loc.*.grid0.loc.hyp")) if ".sum." not in f.name]
    evs, phs = [], []
    for f in files:
        ev, ph = read_event_file(f)
        if ev is None or "ID" not in ev:
            continue
        evs.append(ev)
        phs += ph
    ev = pd.DataFrame(evs)
    ph = pd.DataFrame(phs)
    if len(ev) and ev.ID.duplicated().any():         # coherence reruns write one file per event; keep LOCATED
        ev = ev.sort_values("status").drop_duplicates("ID", keep="first")
    return ev, ph


def located_frame(ev, v4):
    """LOCATED events with true depth, surface, floor height, tier."""
    e = ev[ev.status == "LOCATED"].copy()
    e["depth_t"] = unflatten(e.depth)
    e["surface_km"] = surface_km(e.lat, e.lon)
    fl = read_floor(FLOOR)
    e["h_floor"] = e.depth.to_numpy(float) + floor_at(fl, e.lat, e.lon) / 1e3
    e["floor_limited"] = e.h_floor < FLOOR_CELLS * e.ns_z
    e["above_surface"] = e.depth_t < -e.surface_km
    e = e.merge(v4[["ID", "v4_depth"]], on="ID", how="left")
    tt = e.assign(depth=e.depth_t)
    e["tier"], _ = tier(tt)
    x, y = lambert()(e.lon.values, e.lat.values)
    e["x"], e["y"] = x / 1e3, y / 1e3
    return e


def v51_arm(ids):
    """the v5.1 catalog rows of the set (production: tile boxes, whole-catalog static terms)."""
    c = pd.read_csv(V51_CAT)
    c = c[c.ID.isin(ids) & (c.loc_source == "nll3d")].copy()
    loc = pd.read_csv(V51_LOC, usecols=["ID", "ns_z", "depth", "rms", "nphs"]).rename(columns={"depth": "depth_zf"})
    c = c.merge(loc[["ID", "ns_z", "depth_zf"]], on="ID", how="left")
    c["depth_t"] = c.depth
    x, y = lambert()(c.lon.values, c.lat.values)
    c["x"], c["y"] = x / 1e3, y / 1e3
    return c


def cc_rows(s):
    c = pd.read_parquet(CC_DIR / f"cc_{s}.parquet")
    c = c[(c.status == "ok") & c.above.astype(bool) & ~c.rate_mismatch.astype(bool)]
    if "straddles_clock_epoch" in c:
        c = c[~c.straddles_clock_epoch.fillna(False).astype(bool)]
    if "clock_dt_s" in c:
        c = c[c.clock_dt_s.fillna(0).abs() < 1e-9]
    return c[["ID1", "ID2", "station", "phase", "dt_cc", "dt_cat", "T0_1", "T0_2", "cc_max"]]


def ot_seconds(t):
    return pd.to_datetime(t).astype("int64") / 1e9


def cc_score(e, ph, cc):
    """rms of the pair-demeaned residual of the CC (and catalog) differential times against the arm (docstring)."""
    if not len(e) or not len(ph):
        return {}
    ot = pd.Series(ot_seconds(e.time).to_numpy(), index=e.ID.to_numpy())
    p = ph[ph.ID.isin(e.ID)].copy()
    p["pred"] = p.ttpred + p.tcorr
    p = p.drop_duplicates(["ID", "sta", "phase"]).set_index(["ID", "sta", "phase"]).pred
    c = cc[cc.ID1.isin(ot.index) & cc.ID2.isin(ot.index)].copy()
    k1 = list(zip(c.ID1, c.station, c.phase))
    k2 = list(zip(c.ID2, c.station, c.phase))
    have = np.array([(a in p.index) and (b in p.index) for a, b in zip(k1, k2)], bool)
    c = c[have]
    if not len(c):
        return {}
    p1 = p.loc[[k for k, h in zip(k1, have) if h]].to_numpy()
    p2 = p.loc[[k for k, h in zip(k2, have) if h]].to_numpy()
    t0 = (pd.to_datetime(c.T0_1).astype("int64") - pd.to_datetime(c.T0_2).astype("int64")).to_numpy() / 1e9
    dot = ot.loc[c.ID1].to_numpy() - ot.loc[c.ID2].to_numpy()
    out = {"cc_rows": int(len(c)), "cc_pairs": int(c.groupby(["ID1", "ID2"]).ngroups)}
    for lab, col in (("cc", "dt_cc"), ("cat", "dt_cat")):
        r = c[col].to_numpy() + t0 - dot - (p1 - p2)
        d = pd.DataFrame({"pair": list(zip(c.ID1, c.ID2)), "r": r, "phase": c.phase.to_numpy()})
        n = d.groupby("pair").r.transform("size")
        d = d[n >= 3]                                # pair median needs a few rows
        rr = d.r - d.groupby("pair").r.transform("median")
        out[f"{lab}_rms_pair_ms"] = round(float(np.sqrt(np.mean(rr ** 2))) * 1e3, 1)
        out[f"{lab}_mad_pair_ms"] = round(float(np.median(np.abs(rr))) * 1e3, 1)
        for phz in ("P", "S"):
            m = d.phase == phz
            out[f"{lab}_mad_pair_{phz}_ms"] = round(float(np.median(np.abs(rr[m]))) * 1e3, 1) if m.any() else None
        out[f"{lab}_rows_scored"] = int(len(d))
    return out


def sharp(e, s):
    out = {}
    if len(e) < 3:
        return out
    xyz = np.c_[e.x, e.y, e.depth_t]
    from scipy.spatial import cKDTree
    d, _ = cKDTree(xyz).query(xyz, k=2)
    out["nn_median_km"] = round(float(np.median(d[:, 1])), 3)
    for k, v in zip("xyz", xyz.T):
        out[f"mad_{k}_km"] = round(float(np.median(np.abs(v - np.median(v)))), 3)
    rp = pd.read_csv(CC_DIR / "qc" / f"{s}_repeaters.csv")
    pos = pd.DataFrame(xyz, index=e.ID.to_numpy(), columns=list("xyz"))
    rp = rp[rp.ID1.isin(pos.index) & rp.ID2.isin(pos.index)]
    if len(rp):
        dd = np.linalg.norm(pos.loc[rp.ID1].to_numpy() - pos.loc[rp.ID2].to_numpy(), axis=1)
        out["repeater_pairs"] = int(len(rp))
        out["repeater_dist_median_km"] = round(float(np.median(dd)), 3)
        out["repeater_dist_p90_km"] = round(float(np.percentile(dd, 90)), 3)
    return out


def resid(ph, e):
    p = ph[ph.ID.isin(e.ID) & (ph.weight > 0)]
    out = {}
    for phz in ("P", "S"):
        q = p[p.phase == phz]
        if len(q):
            out[f"rms_{phz}_s"] = round(float(np.sqrt(np.mean(q.res ** 2))), 4)
            g = q.groupby("sta").res.agg(lambda r: np.sqrt(np.mean(r ** 2)) if len(r) >= 10 else np.nan).dropna()
            out[f"rms_{phz}_sta_median_s"] = round(float(g.median()), 4) if len(g) else None
    return out


def lit(e, s):
    if s not in LIT:
        return {}
    ref, f = LIT[s]
    pr = pd.read_csv(PAIRS / f)
    m = pr.merge(e[["ID", "depth_t", "tier"]], on="ID")
    out = {"reference": ref}
    for lab, tt in (("A-D", list("ABCD")), ("A-B", list("AB"))):
        d = m[m.tier.isin(tt)]
        if len(d):
            x = d.depth_t - d.ref_depth
            out[lab] = dict(n=int(len(d)), med=round(float(np.median(x)), 2), p10=round(float(np.percentile(x, 10)), 2),
                            p90=round(float(np.percentile(x, 90)), 2))
    return out


def surface_scores(e, s):
    out = {"located": int(len(e)), "above_surface": int(e.above_surface.sum()),
           "floor_limited": int(e.floor_limited.sum()) if "floor_limited" in e else None,
           "tiers": {k: int(v) for k, v in e.tier.value_counts().sort_index().items()},
           "depth_t_median_km": round(float(np.median(e.depth_t)), 2),
           "below_seafloor_median_km": round(float(np.median(e.depth_t + e.surface_km)), 2)}
    if s == "axial":
        ab = e[e.tier.isin(["A", "B"])]
        bs = ab.depth_t + ab.surface_km
        out["axial_AB"] = int(len(ab))
        out["axial_within_1.6_km"] = round(100 * float((bs <= 1.6).mean()), 1) if len(ab) else None
        out["axial_within_2.0_km"] = round(100 * float((bs <= 2.0).mean()), 1) if len(ab) else None
    return out


def score(e, ph, s, cc):
    return {**surface_scores(e, s), **resid(ph, e), **sharp(e, s), **cc_score(e, ph, cc),
            "literature": lit(e, s)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", default="g51")
    ap.add_argument("--sets", default=",".join(SETS))
    a = ap.parse_args()
    v4 = PH.read_v4()
    rows = []
    for s in a.sets.split(","):
        d = OUT / a.grids / s
        ids = set(pd.read_csv(SETS_DIR / f"events_{s}.csv").ID)
        cc = cc_rows(s)
        res = {}
        arms = [("static", d / "static")] + [(f"ssst_it{k}", d / "ssst" / f"it{k}") for k in range(len(SCHEDULE) + 1)]
        arms += [("coherence", d / "coherence"), ("static_coherence", d / "static_coherence")]
        # v5.1 production: positions from the catalog, residuals from its own run's per-event files are not kept;
        # the CC score uses the NLLoc phase predictions, so v5.1 gets the location scores only
        c51 = v51_arm(ids)
        c51["floor_limited"] = c51.floor_limited.astype(bool)
        c51["above_surface"] = c51.above_surface.astype(bool)
        res["v5.1"] = {**surface_scores(c51, s), **sharp(c51, s), "literature": lit(c51, s)}
        def arm_events(lab, rd):
            ev, ph = read_arm(rd)
            if lab in ("coherence", "static_coherence"):   # events not relocated by coherence keep their location
                ev5, ph5 = read_arm(d / "ssst" / f"it{len(SCHEDULE)}" if lab == "coherence" else d / "static_oct")
                keep = ~ev5.ID.isin(ev.ID)
                ev = pd.concat([ev, ev5[keep]], ignore_index=True)
                ph = pd.concat([ph, ph5[ph5.ID.isin(ev5.ID[keep])]], ignore_index=True)
            return ev, ph

        for lab, rd in arms:
            if not (rd / "run.json").exists():
                continue
            ev, ph = arm_events(lab, rd)
            e = located_frame(ev, v4)
            r = score(e, ph, s, cc)
            r["events"] = int(len(ev))
            if (rd.parent.parent / "ssst").exists() and lab.startswith("ssst_it"):
                k = int(lab[len("ssst_it"):])
                g = rd.parent / f"g{k}" / "loc2ssst.json"
                if g.exists():
                    j = json.loads(g.read_text())
                    r["loc2ssst"] = {k2: j[k2] for k2 in ("char_dist_km", "lsgrid_km", "events_accepted",
                                                          "corrected_grids", "negative_nodes", "wall_s")}
                    r["loc2ssst"]["neg_max_dist_km"] = max([v.get("neg_max_dist_km", 0) for v in j["grids"].values()]
                                                           or [0])
            if lab in ("coherence", "static_coherence"):
                r["coherence_run"] = {k: v for k, v in json.loads((rd / "run.json").read_text()).items()
                                      if k not in ("failed", "grids")}
            res[lab] = r
        base = {}
        for lab, rd in arms:
            if (rd / "run.json").exists():
                ev, _ = arm_events(lab, rd)
                base[lab] = located_frame(ev, v4).set_index("ID")[["x", "y", "depth_t"]]
        base["v5.1"] = c51.set_index("ID")[["x", "y", "depth_t"]]
        for lab in base:
            if lab == "static" or "static" not in base:
                continue
            j = base[lab].join(base["static"], rsuffix="_s", how="inner")
            dh = np.hypot(j.x - j.x_s, j.y - j.y_s)
            dz = j.depth_t - j.depth_t_s
            res[lab]["vs_static"] = dict(n=int(len(j)), dh_median_km=round(float(np.median(dh)), 3),
                                         dz_median_km=round(float(np.median(dz)), 3),
                                         dz_absp90_km=round(float(np.percentile(np.abs(dz), 90)), 3))
        (OUT / a.grids / f"metrics_{s}.json").write_text(json.dumps(res, indent=1, default=float))
        for lab, r in res.items():
            flat = {"set": s, "arm": lab}
            for k, v in r.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        if not isinstance(v2, dict):
                            flat[f"{k}.{k2}"] = v2
                        else:
                            for k3, v3 in v2.items():
                                flat[f"{k}.{k2}.{k3}"] = v3
                else:
                    flat[k] = v
            rows.append(flat)
        print(s, "done", flush=True)
    pd.DataFrame(rows).to_csv(OUT / a.grids / f"scores_{a.grids}_{a.sets.replace(',', '_')}.csv", index=False)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plate boundaries v2, adopted version (D-B117, PI 2026-11-27 19:09 PST).

D-B117 adopts the D-B116 proposal with two changes: (1) the JdF ridge Axial-Endeavour follows the bathymetric axis
(CoAxial-Northern Symmetric axial high) and the straight Axial-Endeavour line becomes a flagged alternative that the
segment rule does not use; (2) the PB2002 triple junction is kept and v2 changes only the Mendocino transform west of
it. Implementation notes A6 (Endeavour from MAPRIDGES) and A7 (Mendocino join) are in RULES.md.

Reads the proposal (BIG/proposed_boundaries_v1.geojson, BIG/work/fit.pkl, OUT/corners.csv, OUT/breaks.csv); writes
  BIG/adopted_boundaries_v2.geojson
  OUT/plate_axes_v2_adopted.csv              traces for pb_common PB_AXES=v2 (default since D-B117), no TJ row
  OUT/corners_adopted.csv, OUT/breaks_adopted.csv
  OUT/segment_change_v1_v2adopted.csv        v1 (gmrt) -> v2 adopted, by from -> to pair (v6.1 tier A-C earthquakes)
  OUT/segment_totals_adopted.csv             per-segment totals for pb2002, gmrt (v1), v2proposal, v2 (adopted)
Usage: python -I pbv2_adopt.py
"""
from __future__ import annotations

import copy
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("PB_CATALOG", "/wd1/mdenolle_data/catalog_v6/v6.1/cascadia_catalog_v6.1.csv")
os.environ.setdefault("PB_LABEL", "v6.1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pbv2_common as C  # noqa: E402
from pbv2_common import P, SECTIONS, log  # noqa: E402
from pbv2_fit import feat  # noqa: E402

BEND_THR = 2.0      # km, bend / step-over threshold for bathymetric lines (2 x the 1 km pick sigma, rule 4)
N_JRN_FIT = 4       # A6: the first four segments of the bathymetric BIC fit (Axial-CoAxial-Northern Symmetric)


def u_of(g, s):
    return g["a"] + g["b"] * (np.asarray(s, float) - g["sm"])


def seg_feature(sec, g, ps, pu, **props):
    s = np.linspace(g["s0"], g["s1"], 30)
    lo, la = sec.lonlat(s, u_of(g, s))
    m = (ps >= g["s0"]) & (ps <= g["s1"])
    d = u_of(g, ps[m]) - pu[m]
    unc = float(np.median(C.line_se(g["cov"], g["sm"], s))) if g.get("cov") is not None else None
    return feat("LineString", np.c_[lo, la], uncertainty_km=None if unc is None else round(unc, 2),
                offset_from_pb2002_km=round(float(np.median(d)), 2) if m.sum() >= 3 else None,
                max_abs_offset_from_pb2002_km=round(float(np.max(np.abs(d))), 2) if m.sum() >= 3 else None,
                offset_sign="+ = " + C.SIGN[sec.kind], length_km=round(g["s1"] - g["s0"], 1), **props)


def jdf_adopted(F):
    """A6: Axial-CoAxial-Northern Symmetric from the bathymetric BIC fit, Endeavour from MAPRIDGES."""
    r = F["res"]["JRN"]
    sec = SECTIONS["JRN"]
    segs = []
    for g in r["bathy_fit"]["geo"][:N_JRN_FIT]:
        segs.append(dict(s0=g["s0"], s1=g["s1"], a=g["a"], b=g["b"], sm=g["sm"], cov=g["cov_fit"], npk=g["npk"],
                         data="bathymetry (axial high, D-B113 picks, BIC fit)"))
    en = next(f for f in C.mapridges() if f["name"] == "Endeavour")
    xy = C.densify(en["xy"], 1.0)
    s, u = sec.su(xy[:, 0], xy[:, 1])
    a, b, sm, cov, rms = C.wls_line(s, u)
    e = dict(s0=float(s.min()), s1=float(min(s.max(), sec.length)), a=a, b=b, sm=sm, cov=cov, npk=0,
             data="bathymetry (MAPRIDGES v2602 Endeavour trace, A6)", rms=rms)
    sb = max(segs[-1]["s1"], e["s0"])
    segs[-1]["s1"], e["s0"] = sb, sb
    segs.append(e)
    ax = r["centroids"]["AX"]
    segs[0]["s0"] = ax["s"]                                   # JRN starts at the Axial node (A4)
    breaks = []
    for k in range(len(segs) - 1):
        x = segs[k]["s1"]
        off = float(u_of(segs[k + 1], x) - u_of(segs[k], x))
        lo, la = sec.lonlat(x, (u_of(segs[k], x) + u_of(segs[k + 1], x)) / 2)
        breaks.append(dict(section="JRN", break_k=k, s_break=x, offset_km=off,
                           kind="step-over" if abs(off) >= BEND_THR else "bend",
                           origin="A6 junction (Northern Symmetric - Endeavour)" if k == len(segs) - 2 else
                           "BIC fit (bathymetry)", lon=float(lo), lat=float(la)))
    return segs, breaks, ax


def mfz_join(F, sub, tj):
    """A7: the Mendocino trace ends one 5 km bin west of its trench crossing X; a straight connector J -> TJ."""
    r = F["res"]["MFZ"]
    sec = SECTIONS["MFZ"]
    L = copy.deepcopy(r["lines"])
    last = L[-1]
    sX = last["s1"]
    sJ = sX - C.BIN
    last["s1"] = sJ
    X = np.array(sec.lonlat(sX, u_of(last, sX)), float)
    J = np.array(sec.lonlat(sJ, u_of(last, sJ)), float)
    st, ut = map(float, sec.su(tj[0], tj[1]))
    uJ = float(u_of(last, sJ))
    conn = float(np.hypot(st - sJ, ut - uJ))
    d_line = np.array([1.0, last["b"]]) / np.hypot(1.0, last["b"])
    d_conn = np.array([st - sJ, ut - uJ]) / conn
    bend = float(np.degrees(np.arccos(np.clip(d_line @ d_conn, -1, 1))))
    # the connector stays west of the PB2002 deformation front (all points but its end)
    q = np.linspace(0, 1, 200)[:-1]
    cl = J[0] + q * (tj[0] - J[0])
    ca = J[1] + q * (tj[1] - J[1])
    margin = P.trench_lon(ca, sub) - cl                       # > 0: west of the trench
    sX_, uX_ = map(float, sec.su(*X))
    area = 0.5 * abs((sX_ - sJ) * (ut - uJ) - (st - sJ) * (uX_ - uJ))
    info = dict(J_lon=J[0], J_lat=J[1], X_lon=X[0], X_lat=X[1], connector_km=conn, bend_deg=bend,
                min_margin_west_of_trench_km=float(np.min(margin) * P.KM * np.cos(np.radians(40.33))),
                dist_X_to_TJ_km=float(P.hav(X[1], X[0], tj[1], tj[0])),
                triangle_JXTJ_km2=area, foot_of_TJ_east_of_trench=True)
    return L, info


def main():
    F = pickle.load(open(C.BIG / "work" / "fit.pkl", "rb"))
    gj = json.loads((C.BIG / "proposed_boundaries_v1.geojson").read_text())
    pts, sub, tj = C.pb2002_points()
    tj = np.asarray(tj, float)
    segs, jbreaks, ax = jdf_adopted(F)
    Lm, join = mfz_join(F, sub, tj)
    log("JRN adopted segments:", [(round(g["s0"], 1), round(g["s1"], 1)) for g in segs])
    log("MFZ join:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in join.items()})
    out = []
    nlast = len(Lm) - 1
    rM, sM = F["res"]["MFZ"], SECTIONS["MFZ"]
    for f in gj["features"]:
        p = f["properties"]
        if p.get("section_code") == "JRN":
            continue                                          # replaced below (A6), chord re-added as alternative
        if p["type"] == "transform" and p.get("section_code") == "MFZ" and p.get("segment") == nlast:
            g = dict(Lm[-1])
            nf = seg_feature(sM, g, rM["ps"], rM["pu"], type="transform", section=p["section"], section_code="MFZ",
                             segment=nlast, source=p["source"], data=p.get("data"), n_picks=p.get("n_picks"),
                             flag=(p.get("flag") or "") + "; A7 (D-B117): ends 5 km west of the deformation front, "
                                  "joined to the PB2002 triple junction by a connector")
            nf["properties"]["uncertainty_km"] = p.get("uncertainty_km")
            out.append(nf)
            continue
        if f["geometry"]["type"] == "Point" and p["type"] == "triple junction":
            q = copy.deepcopy(f)
            q["properties"]["type"] = "triple junction (D-B116 proposal, not adopted)"
            out.append(q)
            continue
        out.append(f)
    # Mendocino connector and the PB2002 triple junction
    out.append(feat("LineString", [[join["J_lon"], join["J_lat"]], [float(tj[0]), float(tj[1])]],
                    type="transform (connector to the PB2002 triple junction)", section="Mendocino transform",
                    section_code="MFZ", segment=99, source="A7 join (D-B117)", length_km=round(join["connector_km"], 2),
                    bend_deg=round(join["bend_deg"], 1),
                    flag="A7: straight connector from J (5 km west of the trench crossing) to the PB2002 triple junction"))
    out.append(feat("Point", [float(tj[0]), float(tj[1])], type="triple junction", name="Mendocino triple junction",
                    source="PB2002 (kept, D-B117): southern end of the JF\\NA subduction trace",
                    join=f"Mendocino trace to J ({join['J_lon']:.4f}, {join['J_lat']:.4f}), connector "
                         f"{join['connector_km']:.2f} km, bend {join['bend_deg']:.1f} deg"))
    # JdF Axial-Endeavour, adopted (A6)
    r, sec = F["res"]["JRN"], SECTIONS["JRN"]
    for k, g in enumerate(segs):
        last = k == len(segs) - 1
        out.append(seg_feature(sec, g, r["ps"], r["pu"], type="spreading ridge", section=sec.name,
                               section_code="JRN", segment=k, source="bathymetry", data=g["data"],
                               n_picks=int(g["npk"]),
                               flag="D-B117 adopted: bathymetric axis (seismicity too sparse)" +
                                    ("; A6: Endeavour segment from the MAPRIDGES trace" if last else "")))
    for b in jbreaks:
        k, x = b["break_k"], b["s_break"]
        lo, la = sec.lonlat([x, x], [u_of(segs[k], x), u_of(segs[k + 1], x)])
        out.append(feat("LineString", np.c_[lo, la], type=b["kind"], section=sec.name, section_code="JRN",
                        break_k=k, origin=b["origin"], source="bathymetry", offset_km=round(b["offset_km"], 2)))
    lo0, la0 = sec.lonlat(segs[0]["s0"], u_of(segs[0], segs[0]["s0"]))
    out.append(feat("LineString", [[ax["lon"], ax["lat"]], [float(lo0), float(la0)]], type="step-over",
                    section="Juan de Fuca Ridge at Axial", section_code="JRN", break_k=98, origin="A4 Axial node",
                    source="A4 Axial node (seismicity centroid)",
                    offset_km=round(float(P.hav(ax["lat"], ax["lon"], float(la0), float(lo0))), 2)))
    ch = F["res"]["JRN"]["lines"][0]
    s = np.linspace(ch["s0"], ch["s1"], 30)
    lo, la = sec.lonlat(s, u_of(ch, s))
    out.append(feat("LineString", np.c_[lo, la],
                    type="spreading ridge (alternative: straight Axial-Endeavour line, not used by the segment rule)",
                    section=sec.name, section_code="JRN_alt", segment=0, source="seismicity (AX-EN centroids)",
                    flag="D-B117: drawn only; the adopted trace is the bathymetric axis"))
    gj2 = dict(type="FeatureCollection", name="adopted_boundaries_v2",
               properties=dict(gj["properties"], decision="D-B117 (adopts D-B116 with two changes)",
                               status="adopted for the paper analyses (PB_AXES default v2)",
                               changes="JdF Axial-Endeavour = bathymetric axis (A6); PB2002 triple junction kept, "
                                       "Mendocino joined by a connector (A7)"),
               features=out)
    (C.BIG / "adopted_boundaries_v2.geojson").write_text(json.dumps(gj2, indent=1))
    with open(C.BIG / "work" / "adopted.pkl", "wb") as fh:
        pickle.dump(dict(jrn=segs, mfz=Lm, join=join), fh)
    axes_file(out)
    # corners and breaks, adopted
    cor = pd.read_csv(C.OUT / "corners.csv")
    tjrow = cor[cor.name.str.contains("triple")].iloc[0].to_dict()
    cor = cor[~cor.name.str.contains("triple")]
    prop = dict(tjrow, name="Mendocino triple junction, D-B116 proposal (not adopted)")
    new = dict(name="Mendocino triple junction (PB2002, kept)", a="MFZ", b="trench (PB2002)", lon=float(tj[0]),
               lat=float(tj[1]), n_draws=0, note="PB2002 point (D-B117); Mendocino joined by the A7 connector",
               dist_to_pb2002_tj_km=0.0, **{k: v for k, v in join.items()})
    ca = pd.concat([cor, pd.DataFrame([new, prop])], ignore_index=True)
    ca.to_csv(C.OUT / "corners_adopted.csv", index=False, float_format="%.4f")
    br = pd.read_csv(C.OUT / "breaks.csv")
    br = pd.concat([br[br.section != "JRN"], pd.DataFrame(jbreaks)], ignore_index=True)
    br.to_csv(C.OUT / "breaks_adopted.csv", index=False, float_format="%.3f")
    seg_change()
    log(ca.to_string())
    log(pd.DataFrame(jbreaks).to_string())


def axes_file(feats):
    """plate_axes_v2_adopted.csv (format of plate_axes_v1.csv): ridges -> OSR; transforms, step-overs and the
    triple-junction connector -> OTF; alternatives, strands and bends left out; no TJ row (PB2002 junction)."""
    rows = []
    for f in feats:
        p = f["properties"]
        if f["geometry"]["type"] != "LineString" or "alternative" in p["type"] or "strand" in p["type"]:
            continue
        kind = "OSR" if p["type"] == "spreading ridge" else ("OTF" if (p["type"].startswith("transform") or
                                                                     p["type"] == "step-over") else None)
        if kind is None:
            continue
        code = p.get("section_code") or "PB2002N"
        name = f"{code}_step{p['break_k']}" if p["type"] == "step-over" else f"{code}{p.get('segment', '')}"
        raw = np.asarray(f["geometry"]["coordinates"], float)
        xy = np.vstack([C.densify(raw, 1.0), raw[-1:]])            # keep both end points (short connectors)
        rows.append(pd.DataFrame(dict(lon=xy[:, 0], lat=xy[:, 1], kind=kind, section=name,
                                      source=p.get("source") or p.get("origin", ""))))
    a = pd.concat(rows, ignore_index=True)
    assert a.groupby("section").size().min() >= 2
    a.to_csv(C.OUT / "plate_axes_v2_adopted.csv", index=False, float_format="%.5f")


def seg_change():
    c = P.load_catalog()
    c = c[c.is_eq & c.abc]
    segs = {ax: P.assign_segments(c, axes=ax).seg for ax in ("pb2002", "gmrt", "v2proposal", "v2")}
    tot = pd.DataFrame({k: v.value_counts() for k, v in segs.items()}).fillna(0).astype(int)
    tot["v2_minus_gmrt"] = tot["v2"] - tot["gmrt"]
    tot.index.name = "segment"
    tot.loc[[i for i in P.SEG_ORDER if i in tot.index]].to_csv(C.OUT / "segment_totals_adopted.csv")
    a, b = segs["gmrt"], segs["v2"]
    ct = pd.crosstab(a.rename("from_v1_gmrt"), b.rename("to_v2_adopted")).stack()
    mv = ct[(ct > 0) & (ct.index.get_level_values(0) != ct.index.get_level_values(1))].sort_values(ascending=False)
    mv = mv.rename("events").reset_index()
    mv.to_csv(C.OUT / "segment_change_v1_v2adopted.csv", index=False)
    log(f"v1 -> v2 adopted: {int((a != b).sum()):,} of {len(c):,} tier A-C earthquakes change segment")
    log(mv.to_string())
    log(tot.to_string())


if __name__ == "__main__":
    main()

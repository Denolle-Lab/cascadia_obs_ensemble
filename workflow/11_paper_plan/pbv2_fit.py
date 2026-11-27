#!/usr/bin/env python3
"""Plate boundaries v2 (D-B116), steps 2-3: seismicity axis per section, straight segments chosen by BIC, comparison
with the bathymetric axis, ridge-transform corners, triple junction, deformation zones, GeoJSON and plate_axes_v2.csv.

Rules: data/paper_plan/plate_boundaries_v2/RULES.md (pre-registered 2026-11-27 14:01 PST before the first fit;
addendum A1-A4 written after the first fit, 14:11 PST, flagged in the outputs).

Outputs
  BIG/proposed_boundaries_v1.geojson     proposal (lines, step-overs, corners, triple junction, zones, alternatives)
  BIG/work/fit.pkl                       events, picks, bootstrap picks and lines per section (pbv2_bias, figures)
  OUT/picks.csv, segments.csv, bic.csv, breaks.csv, corners.csv, zones.csv (small tables, git)
  OUT/plate_axes_v2.csv                  traces for pb_common PB_AXES=v2 (the segment rule itself is unchanged)
Usage: python -I pbv2_fit.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pbv2_common as C  # noqa: E402
from pbv2_common import P, SECTIONS, log  # noqa: E402

DEC = 10.0          # km, decimation of the bathymetric picks (D-B113 median-filter length)
GAP = 30.0          # km, end gap without S0 picks that triggers A2
SNAP_KM = 30.0
ENDS = {("GR", 0, "s0"): "GR-MFZ corner (RTI)", ("MFZ", 0, "s0"): "GR-MFZ corner (RTI)",
        ("GR", -1, "s1"): "GR-Blanco corner (RTI)", ("BL", 0, "s0"): "GR-Blanco corner (RTI)",
        ("BL", -1, "s1"): "Blanco-JdF corner (RTI, Cleft)", ("JRS", 0, "s0"): "Blanco-JdF corner (RTI, Cleft)",
        ("MFZ", -1, "s1"): "Mendocino triple junction"}


# ------------------------------------------------------------------ helpers
def decimate(sec, s, u, grp, u_ref=None):
    """One bathymetric pick per 10 km bin: the median of one candidate group (nearest to u_ref where given, else
    the group with most picks)."""
    e = np.arange(0, sec.length + DEC, DEC)
    out = []
    for a, b in zip(e[:-1], e[1:]):
        m = (s >= a) & (s < b)
        if m.sum() < 2:
            continue
        best = None
        for g in np.unique(grp[m]):
            mg = m & (grp == g)
            if mg.sum() < 2:
                continue
            um = float(np.median(u[mg]))
            sc = abs(um - u_ref((a + b) / 2)) if u_ref is not None else -mg.sum()
            if best is None or sc < best[0]:
                best = (sc, float(np.median(s[mg])), um)
        if best:
            out.append(best[1:])
    return np.array(out).reshape(-1, 2)


def section_bathy(sec, cands, mr):
    """Bathymetric candidate points of the section in its frame: s, u, group label."""
    if sec.bathy == "mapridges":
        xy, grp = [], []
        for f in mr:
            if f["type"] == "transform" and f["name"].startswith("Blanco"):
                d = C.densify(f["xy"], 1.0)
                xy.append(d)
                grp += [f["name"]] * len(d)
        xy, grp = np.vstack(xy), np.array(grp)
    else:
        xy = cands[sec.bathy]
        s0, u0 = sec.su(xy[:, 0], xy[:, 1])
        grp = np.cumsum(np.r_[0, np.hypot(np.diff(s0), np.diff(u0)) > 4.0])   # one group per PB2002 piece
    s, u = sec.su(xy[:, 0], xy[:, 1])
    m = (s >= -2) & (s <= sec.length + 2) & (u >= sec.uwin[0]) & (u <= sec.uwin[1])
    return s[m], u[m], np.asarray(grp)[m]


def pb2002_su(sec, pts):
    xy = pts[sec.pb_class]                              # PB2002 steps, already densified (pb_common)
    s, u = sec.su(xy[:, 0], xy[:, 1])
    m = (s >= -5) & (s <= sec.length + 5) & (u >= sec.uwin[0] - 15) & (u <= sec.uwin[1] + 15)
    return s[m], u[m]


def u_at(geo, x):
    g = next((g for g in geo if g["s0"] <= x <= g["s1"]), geo[0] if x < geo[0]["s0"] else geo[-1])
    return C.line_u(g, x)


def band_of(UB):
    return np.nanpercentile(UB, [2.5, 16, 50, 84, 97.5], axis=0)


# ------------------------------------------------------------------ one section
def fit_section(code, sec, ev, cat, cands, mr, pts, rng):
    s, u = sec.su(ev.lon.values, ev.lat.values)
    m = (s >= 0) & (s < sec.length) & (u >= sec.uwin[0]) & (u <= sec.uwin[1])
    e, s, u = ev[m], s[m], u[m]
    pk, n = C.bin_picks(sec, s, u)
    sig_loc = float(np.median(e.h_max)) if len(e) else 1.0
    B = np.full((C.NBOOT, len(pk)), np.nan)
    for b, w in enumerate(C.day_weights(e.day, rng, C.NBOOT)):
        B[b], _ = C.bin_picks(sec, s, u, w)
    okb = np.isfinite(B).mean(0) >= 0.5
    with np.errstate(all="ignore"):
        lo, hi = np.nanpercentile(np.where(okb, B, np.nan), [16, 84], axis=0)
    sig_boot = (hi - lo) / 2
    keep = np.isfinite(pk) & okb
    sc = sec.centres
    npk = int(keep.sum())
    cover = npk * C.BIN / sec.length
    mode = "seismicity" if (npk >= C.MIN_PICKS and cover >= 0.5) else "bathymetry"
    log(f"{code}: {len(e):,} S0 events, {npk} picks, coverage {cover:.2f}, sigma_loc {sig_loc:.2f} km -> {mode}")
    bs, bu, bg = section_bathy(sec, cands, mr)
    ps, pu = pb2002_su(sec, pts)
    e2 = cat[cat.S2]
    s2, u2 = sec.su(e2.lon.values, e2.lat.values)
    m2 = (s2 >= 0) & (s2 < sec.length) & (u2 >= sec.uwin[0]) & (u2 <= sec.uwin[1])
    pk2, n2 = C.bin_picks(sec, s2[m2], u2[m2])
    r = dict(sec=sec, code=code, events=e, s=s, u=u, pk=pk, n=n, B=B, sig_boot=sig_boot, keep=keep,
             sig_loc=sig_loc, mode=mode, bs=bs, bu=bu, bg=bg, ps=ps, pu=pu, cover=cover, pk2=pk2, n2=n2,
             sig_loc2=float(np.median(e2.h_max.values[m2])) if m2.any() else 1.0)
    if mode == "seismicity":
        ss, uu = sc[keep], pk[keep]
        sg = np.sqrt(sig_boot[keep] ** 2 + sig_loc ** 2)
    else:
        d = decimate(sec, bs, bu, bg)
        ss, uu, sg = d[:, 0], d[:, 1], np.ones(len(d))
    fbest, tab = C.bic_select(ss, uu, sg, sec.length)
    K_bic = int(fbest["K"]) if fbest else -1
    K = sec.k_forced if (sec.k_forced is not None and code != "JRN") else K_bic
    tab["section"], tab["data"] = code, mode
    tab["chosen_bic"], tab["drawn"] = tab.K.eq(K_bic), tab.K.eq(K) & (code != "JRN")
    log(tab.to_string(index=False))
    f = C.fit_segments(ss, uu, sg, K)
    sgrid = np.arange(0, sec.length + 0.5, 1.0)
    u_fit, geo = C.eval_fit(f, sgrid)
    for g in geo:
        g["data"] = mode
    if mode == "seismicity":
        UB, GB, BRK, OFF = [], [], [], []
        for b in range(C.NBOOT):
            kb = np.isfinite(B[b]) & keep
            fb = C.fit_segments(sc[kb], B[b][kb], np.sqrt(sig_boot[kb] ** 2 + sig_loc ** 2), K) if kb.sum() >= 4 * (K + 1) else None
            if fb is None:
                continue
            ub, gb = C.eval_fit(fb, sgrid)
            for g in gb:
                g["data"] = mode
            UB.append(ub)
            GB.append(gb)
            BRK.append([g["s1"] for g in gb[:-1]])
            OFF.append(C.step_offsets(gb))
        UB = np.array(UB)
        BRK, OFF = np.array(BRK).reshape(len(UB), K), np.array(OFF).reshape(len(UB), K)
    else:
        UB, GB = param_draws(geo, ss, uu, sgrid, rng)
        BRK = OFF = np.zeros((len(UB), K))
    r.update(f=f, K=K, K_bic=K_bic, bic=tab, geo=geo, sgrid=sgrid, u_fit=u_fit, UB=UB, GB=GB, BRK=BRK, OFF=OFF,
             fit_s=ss, fit_u=uu, fit_sig=sg)
    r["band"] = band_of(UB)
    return r


def param_draws(geo, ss, uu, sgrid, rng, n=C.NBOOT):
    UB, GB = [], []
    fits = []
    for g in geo:
        m = (ss >= g["s0"]) & (ss <= g["s1"])
        fits.append(C.wls_line(ss[m], uu[m]))
        g["cov_fit"] = fits[-1][3]
    for _ in range(n):
        gb = []
        for g, (a_, b_, sm, cov, _r) in zip(geo, fits):
            a2, b2 = rng.multivariate_normal([a_, b_], cov)
            gb.append({**g, "a": a2, "b": b2, "sm": sm})
        GB.append(gb)
        ub = np.full(len(sgrid), np.nan)
        for k, g in enumerate(gb):
            mm = (sgrid >= (-np.inf if k == 0 else g["s0"])) & (sgrid < (np.inf if k == len(gb) - 1 else g["s1"]))
            ub[mm] = C.line_u(g, sgrid[mm])
        UB.append(ub)
    return np.array(UB), GB


def fill_end_gaps(r, rng):
    """A2: a section end with no S0 pick over > 30 km gets a straight segment from the S2 picks in that span."""
    sec, geo = r["sec"], r["geo"]
    if r["mode"] != "seismicity":
        return
    sc = sec.centres
    added = []
    for side in ("start", "end"):
        gap = (0.0, geo[0]["s0"]) if side == "start" else (geo[-1]["s1"], sec.length)
        if gap[1] - gap[0] <= GAP:
            continue
        m = np.isfinite(r["pk2"]) & (sc >= gap[0]) & (sc <= gap[1])
        if m.sum() < 4:
            log(f"{r['code']}: {side} gap {gap[1] - gap[0]:.0f} km, {m.sum()} S2 picks: not filled")
            continue
        a_, b_, sm, cov, rms = C.wls_line(sc[m], r["pk2"][m])
        g = dict(k=-1, s0=gap[0], s1=gap[1], a=a_, b=b_, sm=sm, i0=-1, i1=-1, npk=int(m.sum()),
                 data="seismicity (tier A-C)", cov=cov, rms=rms)
        log(f"{r['code']}: {side} gap {gap[1] - gap[0]:.0f} km filled from {m.sum()} S2 picks (A2), rms {rms:.2f} km")
        if side == "start":
            geo.insert(0, g)
        else:
            geo.append(g)
        added.append(g)
        r.setdefault("a2", []).append(dict(side=side, s0=gap[0], s1=gap[1], n_picks=int(m.sum()), rms=rms))
    for k, g in enumerate(geo):
        g["k"] = k
    if not added:
        return
    for i, gb in enumerate(r["GB"]):
        it = iter(gb)
        new = []
        for g in geo:
            if g["data"] == "seismicity (tier A-C)":
                a2, b2 = rng.multivariate_normal([g["a"], g["b"]], g["cov"])
                new.append({**g, "a": a2, "b": b2})
            else:
                new.append(next(it))
        r["GB"][i] = new


def jrn_chord(r, cat, rng):
    """A3: JRN = the straight line through the Axial (S0) and Endeavour (S2) seismicity centroids."""
    sec = r["sec"]
    ax = cat[cat.S0 & cat.region.eq("Axial")]
    en = cat[cat.S2 & cat.region.eq("Endeavour")]

    def chord(wa=None, we=None):
        la = np.average(ax.lat, weights=wa), np.average(ax.lon, weights=wa)
        le = np.average(en.lat, weights=we), np.average(en.lon, weights=we)
        (sa, ua), (se, ue) = [tuple(map(float, sec.su(lo, lt))) for lt, lo in (la, le)]
        return dict(a=ua, b=(ue - ua) / (se - sa), sm=sa), (la, le), (sa, ua, se, ue)
    c0, cent, su = chord()
    UB, GB = [], []
    for wa, we in zip(C.day_weights(ax.day, rng, C.NBOOT), C.day_weights(en.day, rng, C.NBOOT)):
        cb, _, _ = chord(wa, we)
        GB.append([dict(k=0, s0=su[0], s1=sec.length, data="seismicity (AX-EN centroids)", **cb)])
        UB.append(cb["a"] + cb["b"] * (r["sgrid"] - cb["sm"]))
    r["bathy_fit"] = dict(geo=r["geo"], UB=r["UB"], K=r["K_bic"])       # bathymetric BIC fit = the alternative
    r["geo"] = [dict(k=0, s0=su[0], s1=sec.length, data="seismicity (AX-EN centroids)", i0=-1, i1=-1, npk=2, **c0)]
    r["GB"], r["UB"] = GB, np.array(UB)
    r["band"] = band_of(r["UB"])
    r["u_fit"] = c0["a"] + c0["b"] * (r["sgrid"] - c0["sm"])
    r["K"] = 0
    r["mode"] = "seismicity (AX-EN centroids)"
    r["centroids"] = dict(AX=dict(lat=float(cent[0][0]), lon=float(cent[0][1]), n=len(ax), s=su[0], u=su[1]),
                          EN=dict(lat=float(cent[1][0]), lon=float(cent[1][1]), n=len(en), s=su[2], u=su[3]))
    log(f"JRN chord: AX {cent[0]} (n={len(ax)}), EN {cent[1]} (n={len(en)})")


def compare_bathy(r):
    """Rule 5: per segment, the straight bathymetric line over the segment vs the drawn seismicity line."""
    sec, geo = r["sec"], r["geo"]
    dB = decimate(sec, r["bs"], r["bu"], r["bg"], lambda x: u_at(geo, x))
    r["bathy_dec"] = dB
    rows = []
    hw_all = (r["band"][3] - r["band"][1]) / 2
    for g in geo:
        m = (dB[:, 0] >= g["s0"]) & (dB[:, 0] <= g["s1"])
        q = np.array([g["s0"], (g["s0"] + g["s1"]) / 2, g["s1"]])
        us = C.line_u(g, q)
        hw = C.line_se(g["cov"], g["sm"], q) if "cov" in g else np.interp(q, r["sgrid"], hw_all)
        row = dict(section=r["code"], segment=g["k"], data=g["data"], s0=g["s0"], s1=g["s1"],
                   length_km=g["s1"] - g["s0"], n_picks=g["npk"], a=g["a"], b=g["b"], sm=g["sm"],
                   sig_loc=r["sig_loc"], u_start=us[0], u_mid=us[1], u_end=us[2], sig_seis_mid=hw[1])
        g["bathy"] = None
        if g["data"] == "bathymetry":
            g["bathy"] = dict(a=g["a"], b=g["b"], sm=g["sm"], cov=g["cov_fit"])
            row.update(agree=True, source="bathymetry", bathy_n=g["npk"],
                       bathy_se_mid=float(C.line_se(g["cov_fit"], g["sm"], q)[1]))
        elif m.sum() >= 3:
            a_, b_, sm, cov, rms = C.wls_line(dB[m, 0], dB[m, 1])
            ub_ = a_ + b_ * (q - sm)
            se = C.line_se(cov, sm, q)
            g["bathy"] = dict(a=a_, b=b_, sm=sm, cov=cov)
            dev = np.abs(us - ub_)
            tol = 2 * np.sqrt(hw ** 2 + se ** 2)
            agree = bool(np.all(dev <= tol))
            row.update(bathy_a=a_, bathy_b=b_, bathy_sm=sm, bathy_rms=rms, bathy_n=int(m.sum()), bathy_se_mid=se[1],
                       dev_start=dev[0], dev_mid=dev[1], dev_end=dev[2], tol_max=float(tol.max()),
                       max_excess=float(np.max(dev - tol)), agree=agree, source="both" if agree else "seismicity")
        else:
            row.update(agree=np.nan, bathy_n=int(m.sum()), source="seismicity")
        g["source"] = row["source"]
        rows.append(row)
    return rows


def breaks_of(r):
    sec, geo = r["sec"], r["geo"]
    rows = []
    thr = 2 * (r["sig_loc"] if r["mode"] == "seismicity" else 1.0)
    first_fit = next((i for i, g in enumerate(geo) if g["data"] == r["mode"]), 0)
    for k in range(len(geo) - 1):
        sb = geo[k]["s1"]
        off0 = float(C.line_u(geo[k + 1], sb) - C.line_u(geo[k], sb))
        a2 = "tier A-C" in geo[k]["data"] + geo[k + 1]["data"]
        row = dict(section=r["code"], break_k=k, s_break=sb, offset_km=off0,
                   kind="step-over" if abs(off0) >= thr else "bend", origin="A2 junction" if a2 else "BIC fit")
        kb = k - first_fit
        if r["mode"] == "seismicity" and not a2 and 0 <= kb < r["BRK"].shape[1]:
            for nm, arr in (("s", r["BRK"][:, kb]), ("off", r["OFF"][:, kb])):
                p = np.percentile(arr, [2.5, 16, 50, 84, 97.5])
                row.update({f"{nm}_p2.5": p[0], f"{nm}_p16": p[1], f"{nm}_p84": p[3], f"{nm}_p97.5": p[4]})
        lo, la = sec.lonlat(sb, (C.line_u(geo[k], sb) + C.line_u(geo[k + 1], sb)) / 2)
        row.update(lon=float(lo), lat=float(la))
        rows.append(row)
    return rows


# ------------------------------------------------------------------ main
def main():
    rng = np.random.default_rng(C.SEED)
    C.BIG.mkdir(parents=True, exist_ok=True)
    (C.BIG / "work").mkdir(exist_ok=True)
    C.OUT.mkdir(parents=True, exist_ok=True)
    cat = C.load_v61()
    ev = cat[cat.S0]
    log(f"v6.1 S0 events: {len(ev):,}")
    pts, sub, tj_pb = C.pb2002_points()
    cands = C.bathy_picks()
    mr = C.mapridges()
    res, pick_rows, seg_rows, bic_rows, brk_rows = {}, [], [], [], []
    for code, sec in SECTIONS.items():
        r = fit_section(code, sec, ev, cat, cands, mr, pts, rng)
        fill_end_gaps(r, rng)
        if code == "JRN":
            jrn_chord(r, cat, rng)
        seg_rows += compare_bathy(r)
        brk_rows += breaks_of(r)
        bic_rows.append(r["bic"])
        for i in range(len(r["pk"])):
            pick_rows.append(dict(section=code, s=sec.centres[i], n_S0=r["n"][i], pick_u=r["pk"][i],
                                  sig_boot=r["sig_boot"][i], used=bool(r["keep"][i]), n_S2=r["n2"][i],
                                  pick_u_S2=r["pk2"][i]))
        res[code] = r
    ax = res["JRN"]["centroids"]["AX"]
    axial = dict(name="Axial node", lon=ax["lon"], lat=ax["lat"], n_draws=0, a="JRS", b="JRN",
                 note="Axial seismicity centroid (A4)")
    lines = {}
    for code, r in res.items():
        L = []
        for g in r["geo"]:
            ln = g["bathy"] if (g["source"] in ("both", "bathymetry") and g["bathy"] is not None) else g
            L.append(dict(s0=g["s0"], s1=g["s1"], a=ln["a"], b=ln["b"], sm=ln["sm"], source=g["source"], k=g["k"],
                          data=g["data"]))
        lines[code] = L
    corners = corners_tj(res, lines, sub)
    corners.insert(3, axial)
    snap_to_corners(res, lines, corners)
    pd.DataFrame(corners).to_csv(C.OUT / "corners.csv", index=False, float_format="%.4f")
    zones = deformation_zones(cat, res, lines, corners, sub)
    seg = pd.DataFrame(seg_rows)
    brk = pd.DataFrame(brk_rows)
    feats = build_features(res, lines, corners, zones, seg, brk, pts)
    gj = dict(type="FeatureCollection", name="proposed_boundaries_v1",
              properties=dict(decision="D-B116", rules="data/paper_plan/plate_boundaries_v2/RULES.md",
                              catalog="v6.1 S0 (tier A-B, h_max <= 3 km, ell_len3 <= 10 km, earthquakes)",
                              sign="offset_from_pb2002_km: proposal minus PB2002, + north (transforms) or east (ridges)",
                              status="proposal for PI review; not adopted (PB_AXES default unchanged)"),
              features=feats)
    (C.BIG / "proposed_boundaries_v1.geojson").write_text(json.dumps(gj, indent=1))
    pd.DataFrame(pick_rows).to_csv(C.OUT / "picks.csv", index=False, float_format="%.3f")
    pd.concat(bic_rows).to_csv(C.OUT / "bic.csv", index=False, float_format="%.2f")
    brk.to_csv(C.OUT / "breaks.csv", index=False, float_format="%.3f")
    seg.to_csv(C.OUT / "segments.csv", index=False, float_format="%.3f")
    axes_v2(feats, corners)
    keep = {}
    for code, r in res.items():
        keep[code] = {k: v for k, v in r.items() if k not in ("sec", "events", "f")}
        keep[code]["ids"] = r["events"].ID.values
        keep[code]["lines"] = lines[code]
    with open(C.BIG / "work" / "fit.pkl", "wb") as fh:
        pickle.dump(dict(res=keep, corners=corners, zones=zones), fh)
    pd.set_option("display.width", 250)
    log(seg[["section", "segment", "data", "s0", "s1", "n_picks", "source", "agree", "dev_mid", "tol_max"]].to_string())
    log(brk.to_string())
    log(pd.DataFrame(corners).to_string())


# ------------------------------------------------------------------ corners and the triple junction
def seg_xy(sec, ln, proj, ext=150.0):
    s = np.array([ln["s0"] - ext, ln["s1"] + ext])
    lo, la = sec.lonlat(s, ln["a"] + ln["b"] * (s - ln["sm"]))
    x, y = proj(lo, la)
    return np.c_[np.asarray(x) / 1e3, np.asarray(y) / 1e3]


def draw_line(r, L, k, i):
    """Draw i of the proposed line of segment k: bootstrap line (seismicity) or parametric (bathymetric, A2)."""
    ln = L[k]
    g = r["geo"][k]
    if ln["source"] in ("both", "bathymetry") and g.get("bathy") is not None:
        rr = np.random.default_rng(C.SEED + 7919 * i + 31 * k)
        a, b = rr.multivariate_normal([g["bathy"]["a"], g["bathy"]["b"]], g["bathy"]["cov"])
        return {**ln, "a": a, "b": b, "sm": g["bathy"]["sm"]}
    gb = r["GB"][i % len(r["GB"])]
    return {**ln, "a": gb[k]["a"], "b": gb[k]["b"], "sm": gb[k]["sm"]}


def corners_tj(res, lines, sub):
    specs = [("GR-MFZ corner (RTI)", "GR", 0, "MFZ", 0), ("GR-Blanco corner (RTI)", "GR", -1, "BL", 0),
             ("Blanco-JdF corner (RTI, Cleft)", "BL", -1, "JRS", 0)]
    rows = []
    for name, A, ka, Bc, kb in specs:
        ka2, kb2 = ka % len(lines[A]), kb % len(lines[Bc])
        sec, ln = res[A]["sec"], lines[A][ka2]
        sx = ln["s1"] if ka == -1 else ln["s0"]
        lo, la = sec.lonlat(sx, ln["a"] + ln["b"] * (sx - ln["sm"]))
        proj = C.km_proj(float(lo), float(la))
        pa, pb = seg_xy(sec, ln, proj), seg_xy(res[Bc]["sec"], lines[Bc][kb2], proj)
        x = C.intersect(pa[0], pa[1] - pa[0], pb[0], pb[1] - pb[0])[0]
        pts = []
        for i in range(C.NBOOT):
            qa = seg_xy(sec, draw_line(res[A], lines[A], ka2, i), proj)
            qb = seg_xy(res[Bc]["sec"], draw_line(res[Bc], lines[Bc], kb2, i), proj)
            y = C.intersect(qa[0], qa[1] - qa[0], qb[0], qb[1] - qb[0])
            if y is not None:
                pts.append(y[0])
        rows.append(corner_row(name, A, ka2, Bc, kb2, x, np.array(pts), proj))
    r = res["MFZ"]
    k = len(lines["MFZ"]) - 1
    proj = C.km_proj(-124.7, 40.33)
    T = C.densify(sub[sub[:, 1] < 41.5], 0.5)
    tx, ty = proj(T[:, 0], T[:, 1])
    T = np.c_[np.asarray(tx) / 1e3, np.asarray(ty) / 1e3]

    def hit(ln):
        p = seg_xy(r["sec"], ln, proj, ext=200.0)
        best = None
        for i in range(len(T) - 1):
            y = C.intersect(p[0], p[1] - p[0], T[i], T[i + 1] - T[i])
            if y is not None and 0 <= y[2] <= 1 and (best is None or np.hypot(*y[0]) < np.hypot(*best)):
                best = y[0]
        return best
    x = hit(lines["MFZ"][k])
    pts = np.array([q for q in (hit(draw_line(r, lines["MFZ"], k, i)) for i in range(C.NBOOT)) if q is not None])
    row = corner_row("Mendocino triple junction", "MFZ", k, "trench (PB2002)", -1, x, pts, proj)
    _, _, tj = C.pb2002_points()
    tjx, tjy = proj(float(tj[0]), float(tj[1]))
    row["dist_to_pb2002_tj_km"] = float(np.hypot(tjx / 1e3 - x[0], tjy / 1e3 - x[1]))
    row["note"] = "trench position (PB2002) taken as exact; its uncertainty is not included"
    rows.append(row)
    return rows


def corner_row(name, A, ka, Bc, kb, x, pts, proj):
    lo, la = proj(x[0] * 1e3, x[1] * 1e3, inverse=True)
    row = dict(name=name, a=A, seg_a=ka, b=Bc, seg_b=kb, lon=float(lo), lat=float(la), n_draws=len(pts))
    if len(pts) >= 10:
        w, v = np.linalg.eigh(np.cov((pts - np.median(pts, 0)).T))
        row.update(sig_major_km=float(np.sqrt(max(w[1], 0))), sig_minor_km=float(np.sqrt(max(w[0], 0))),
                   az_major_deg=float(np.degrees(np.arctan2(v[0, 1], v[1, 1])) % 180),
                   sig_e_km=float(np.std(pts[:, 0])), sig_n_km=float(np.std(pts[:, 1])))
    return row


def snap_to_corners(res, lines, corners):
    """End segments are extended or trimmed to the corner (on both straight lines) when it lies within 30 km of the
    segment end along strike; otherwise the end is kept and the corner flagged. JRN starts at the Axial node (A3/A4);
    JRS ends at the node's along-strike position (A4)."""
    cn = {c["name"]: c for c in corners}
    for (code, kk, side), nm in ENDS.items():
        sec = res[code]["sec"]
        ln = lines[code][kk % len(lines[code])]
        sc, _ = sec.su(cn[nm]["lon"], cn[nm]["lat"])
        sc = float(sc)
        if abs(sc - ln[side]) <= SNAP_KM:
            ln[side] = sc
        else:
            cn[nm]["flag"] = (str(cn[nm].get("flag") or "") + f" {code} end not snapped ({sc - ln[side]:+.0f} km)").strip()
    ax = cn["Axial node"]
    lines["JRN"][0]["s0"] = res["JRN"]["centroids"]["AX"]["s"]
    sa, _ = res["JRS"]["sec"].su(ax["lon"], ax["lat"])
    lines["JRS"][-1]["s1"] = float(sa)


# ------------------------------------------------------------------ deformation zones (RULES 6)
def hdr_polygons(lon, lat, lon_c, lat_c, mass=0.8, bw=5.0, min_area=100.0, simp=2.0):
    from scipy.ndimage import gaussian_filter
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    proj = C.km_proj(lon_c, lat_c)
    x, y = proj(np.asarray(lon), np.asarray(lat))
    x, y = np.asarray(x) / 1e3, np.asarray(y) / 1e3
    xe = np.arange(x.min() - 25, x.max() + 26, 1.0)
    ye = np.arange(y.min() - 25, y.max() + 26, 1.0)
    H, _, _ = np.histogram2d(x, y, [xe, ye])
    D = gaussian_filter(H, bw)
    v = np.sort(D.ravel())[::-1]
    cm = np.cumsum(v) / v.sum()
    lev = v[np.searchsorted(cm, mass)]
    xc, yc = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2
    fig = plt.figure()
    cs = plt.contour(xc, yc, D.T, [lev])
    polys = [Polygon(p).buffer(0) for p in cs.allsegs[0] if len(p) >= 4]
    plt.close(fig)
    U = unary_union(polys).simplify(simp)
    out = []
    for g in [g for g in getattr(U, "geoms", [U]) if g.area >= min_area]:
        lo, la = proj(np.asarray(g.exterior.xy[0]) * 1e3, np.asarray(g.exterior.xy[1]) * 1e3, inverse=True)
        out.append(dict(lonlat=np.c_[lo, la], area_km2=float(g.area), n_events=int(shapely.contains_xy(g, x, y).sum())))
    return out, float(lev)


def line_lonlat(res, lines, code):
    sec = res[code]["sec"]
    out = []
    for ln in lines[code]:
        s = np.linspace(ln["s0"], ln["s1"], 60)
        lo, la = sec.lonlat(s, ln["a"] + ln["b"] * (s - ln["sm"]))
        out.append(np.c_[lo, la])
    return out


def deformation_zones(cat, res, lines, corners, sub):
    from shapely.geometry import Polygon
    zones = []
    allxy = np.vstack([np.vstack(line_lonlat(res, lines, c)) for c in lines])
    cn = {c["name"]: (c["lon"], c["lat"]) for c in corners}
    g1, g2, tj = cn["GR-MFZ corner (RTI)"], cn["GR-Blanco corner (RTI)"], cn["Mendocino triple junction"]
    gr = np.vstack(line_lonlat(res, lines, "GR"))
    mf = np.vstack(line_lonlat(res, lines, "MFZ"))
    gr = gr[(gr[:, 1] > g1[1]) & (gr[:, 1] < g2[1])]
    mf = mf[(mf[:, 0] > g1[0]) & (mf[:, 0] < tj[0])]
    tr = sub[(sub[:, 1] > tj[1]) & (sub[:, 1] < g2[1])]
    trn_lon = float(np.interp(g2[1], sub[:, 1], sub[:, 0]))
    ring = np.vstack([[g1], gr[np.argsort(gr[:, 1])], [g2], [[trn_lon, g2[1]]], tr[np.argsort(-tr[:, 1])], [tj],
                      mf[np.argsort(-mf[:, 0])], [g1]])
    plate = Polygon(ring).buffer(0)
    e = cat[cat.S0]
    inside = shapely.contains_xy(plate, e.lon.values, e.lat.values)
    d = P.dist_to_trace(e.lon.values, e.lat.values, C.densify(allxy, 0.5))
    g = e[inside & (d > 10.0)]
    pz, lev = hdr_polygons(g.lon.values, g.lat.values, -125.9, 41.3)
    for i, z in enumerate(pz):
        zones.append(dict(name=f"Gorda plate interior deformation zone {i + 1}", region="GP", selection="S0",
                          confidence="medium", level=lev, n_input=len(g), **z))
    x = cat[cat.S2 & (cat.lon > -131.0) & (cat.lon < -126.8) & (cat.lat > 48.45) & (cat.lat < 50.8)]
    x = x[x.lon.values < P.trench_lon(x.lat.values, sub)]
    pz, lev = hdr_polygons(x.lon.values, x.lat.values, -128.6, 49.4)
    for i, z in enumerate(pz):
        zones.append(dict(name=f"Explorer plate edge deformation zone {i + 1}", region="EXP", selection="S2",
                          confidence=f"low (tier A-C; S0 has {int(x.S0.sum())} events here)", level=lev,
                          n_input=len(x), **z))
    pd.DataFrame([{k: v for k, v in z.items() if k != "lonlat"} for z in zones]).to_csv(
        C.OUT / "zones.csv", index=False, float_format="%.4g")
    return zones


# ------------------------------------------------------------------ GeoJSON and plate_axes_v2.csv
def pb_offset(ln, ps, pu):
    """Median and max |offset| (proposal minus PB2002) over the segment, from the PB2002 points in its span."""
    m = (ps >= ln["s0"]) & (ps <= ln["s1"])
    if m.sum() < 3:
        return np.nan, np.nan
    d = (ln["a"] + ln["b"] * (ps[m] - ln["sm"])) - pu[m]
    return float(np.median(d)), float(np.max(np.abs(d)))


def build_features(res, lines, corners, zones, seg, brk, pts):
    F = []
    for code, L in lines.items():
        r = res[code]
        sec = r["sec"]
        kind = "transform" if sec.kind == "transform" else "spreading ridge"
        hw_all = (r["band"][3] - r["band"][1]) / 2
        for k, ln in enumerate(L):
            s = np.linspace(ln["s0"], ln["s1"], 30)
            lo, la = sec.lonlat(s, ln["a"] + ln["b"] * (s - ln["sm"]))
            srow = seg[(seg.section == code) & (seg.segment == k)].iloc[0]
            g = r["geo"][k]
            if ln["source"] in ("both", "bathymetry"):
                unc = float(np.median(C.line_se(g["bathy"]["cov"], g["bathy"]["sm"], s)))
            elif "cov" in g:
                unc = float(np.median(C.line_se(g["cov"], g["sm"], s)))
            else:
                unc = float(np.median(np.interp(s, r["sgrid"], hw_all)))
            off, offmax = pb_offset(ln, r["ps"], r["pu"])
            flags = []
            if srow["source"] == "seismicity" and srow.get("agree") is False:
                flags.append("bathymetric and seismicity axes disagree (alternative feature given)")
            if srow["source"] == "seismicity" and g.get("bathy") is None:
                flags.append("no bathymetric axis in this span")
            if g["data"] == "bathymetry":
                flags.append("seismicity too sparse: bathymetry only")
            if "tier A-C" in g["data"]:
                flags.append("A2: fitted to tier A-C picks (no tier A-B picks in this span)")
            if "AX-EN" in g["data"]:
                flags.append("A3: straight line through the Axial and Endeavour seismicity centroids (PI); "
                             "seismicity between them too sparse for an axis")
            F.append(feat("LineString", np.c_[lo, la], type=kind, section=sec.name, section_code=code, segment=k,
                          source=srow["source"], data=g["data"], uncertainty_km=round(unc, 2),
                          uncertainty_note="68 % half-width across strike: day-block bootstrap of the seismicity "
                                           "line, or standard error of a straight-line fit (bathymetric / A2 lines)",
                          offset_from_pb2002_km=None if np.isnan(off) else round(off, 2),
                          max_abs_offset_from_pb2002_km=None if np.isnan(offmax) else round(offmax, 2),
                          offset_sign="+ = " + C.SIGN[sec.kind], length_km=round(ln["s1"] - ln["s0"], 1),
                          n_picks=int(srow["n_picks"]), flag="; ".join(flags)))
            if srow["source"] == "seismicity" and g.get("bathy") is not None:
                bb = g["bathy"]
                lo, la = sec.lonlat(s, bb["a"] + bb["b"] * (s - bb["sm"]))
                F.append(feat("LineString", np.c_[lo, la], type=kind + " (bathymetric alternative)",
                              section=sec.name, section_code=code, segment=k, source="bathymetry",
                              uncertainty_km=round(float(srow["bathy_se_mid"]), 2),
                              flag="alternative: %.1f km from the seismicity line at mid-segment" % srow["dev_mid"]))
        if code == "JRN":
            bf = r["bathy_fit"]
            for g in bf["geo"]:
                s = np.linspace(g["s0"], g["s1"], 30)
                lo, la = sec.lonlat(s, C.line_u(g, s))
                F.append(feat("LineString", np.c_[lo, la], type="spreading ridge (bathymetric alternative)",
                              section=sec.name, section_code=code, segment=g["k"], source="bathymetry",
                              flag=f"bathymetric BIC fit (K={bf['K']}) of the axial-high picks; disagrees with the "
                                   "AX-EN line"))
        for b in brk[brk.section == code].to_dict("records"):
            k, sb = b["break_k"], b["s_break"]
            u0 = L[k]["a"] + L[k]["b"] * (sb - L[k]["sm"])
            u1 = L[k + 1]["a"] + L[k + 1]["b"] * (sb - L[k + 1]["sm"])
            lo, la = sec.lonlat([sb, sb], [u0, u1])
            props = dict(type=b["kind"], section=sec.name, section_code=code, break_k=k, origin=b["origin"],
                         source=r["mode"],
                         offset_km=round(float(u1 - u0), 2))
            for q in ("s_p16", "s_p84", "off_p16", "off_p84", "off_p2.5", "off_p97.5"):
                if q in b and np.isfinite(b[q]):
                    props[q] = round(float(b[q]), 2)
            F.append(feat("LineString", np.c_[lo, la], **props))
    ax = next(c for c in corners if c["name"] == "Axial node")
    ln = lines["JRS"][-1]
    sec = res["JRS"]["sec"]
    lo, la = sec.lonlat(ln["s1"], ln["a"] + ln["b"] * (ln["s1"] - ln["sm"]))
    F.append(feat("LineString", [[float(lo), float(la)], [ax["lon"], ax["lat"]]], type="step-over",
                  section="Juan de Fuca Ridge at Axial", section_code="JRS", break_k=99, origin="A4 Axial node",
                  offset_km=round(float(P.hav(float(la), float(lo), ax["lat"], ax["lon"])), 2)))
    for c in corners:
        F.append(feat("Point", [c["lon"], c["lat"]], type="triple junction" if "triple" in c["name"] else "corner",
                      name=c["name"],
                      source="intersection of the proposed straight traces" if c.get("n_draws") else c.get("note", ""),
                      uncertainty_km=c.get("sig_major_km"), uncertainty_minor_km=c.get("sig_minor_km"),
                      uncertainty_az_deg=c.get("az_major_deg"), flag=c.get("flag"),
                      dist_to_pb2002_tj_km=c.get("dist_to_pb2002_tj_km")))
    for z in zones:
        F.append(feat("Polygon", z["lonlat"], type="deformation zone", name=z["name"], source="seismicity",
                      selection=z["selection"], confidence=z["confidence"], area_km2=round(z["area_km2"]),
                      n_events=z["n_events"], rule="80 % highest-density region, Gaussian kernel 5 km"))
    tr = pd.read_csv(C.FAULTS / "fault_surfaces_v1_traces.csv")
    for t in tr[(tr.region == "BL") & (tr.n_events >= 100)].itertuples():
        F.append(feat("LineString", [[t.top_edge_lon1, t.top_edge_lat1], [t.top_edge_lon2, t.top_edge_lat2]],
                      type="transform strand (fault surface v1, v6.1)", section="Blanco transform",
                      surface_id=t.surface_id, n_events=int(t.n_events), strike=float(t.strike), dip=float(t.dip),
                      source="seismicity (fault surface top edge)"))
    jrn_end = lines["JRN"][-1]
    _, lat_end = res["JRN"]["sec"].lonlat(jrn_end["s1"], jrn_end["a"] + jrn_end["b"] * (jrn_end["s1"] - jrn_end["sm"]))
    for k in ("OSR", "OTF"):
        xy = pts[k]
        m = xy[:, 1] > float(lat_end) + 0.02
        for i, piece in enumerate(_pieces(xy[m])):
            F.append(feat("LineString", piece, type=("spreading ridge" if k == "OSR" else "transform"),
                          section="West Valley, Sovanco, Explorer (PB2002 kept)", section_code=f"PB2002N{k}", segment=i,
                          source="PB2002", flag="not resolved: too few well-located earthquakes"))
    return F


def _pieces(xy, gap=3.0):
    if len(xy) < 2:
        return []
    d = P.hav(xy[:-1, 1], xy[:-1, 0], xy[1:, 1], xy[1:, 0])
    cut = np.r_[0, np.flatnonzero(d > gap) + 1, len(xy)]
    return [xy[a:b] for a, b in zip(cut[:-1], cut[1:]) if b - a >= 2]


def feat(gtype, coords, **props):
    coords = np.asarray(coords, float)
    if gtype == "Point":
        g = dict(type="Point", coordinates=[round(float(coords[0]), 5), round(float(coords[1]), 5)])
    elif gtype == "LineString":
        g = dict(type="LineString", coordinates=[[round(float(a), 5), round(float(b), 5)] for a, b in coords])
    else:
        ring = [[round(float(a), 5), round(float(b), 5)] for a, b in coords]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        g = dict(type="Polygon", coordinates=[ring])
    clean = {}
    for k, v in props.items():
        if isinstance(v, (np.floating, float)):
            v = None if not np.isfinite(v) else round(float(v), 4)
        elif isinstance(v, np.integer):
            v = int(v)
        elif isinstance(v, np.bool_):
            v = bool(v)
        clean[k] = v
    return dict(type="Feature", geometry=g, properties=clean)


def axes_v2(feats, corners):
    """plate_axes_v2.csv in the plate_axes_v1.csv format (lon, lat, kind OSR/OTF, section, source) plus the triple
    junction (kind TJ). Ridges -> OSR; transforms and step-overs -> OTF; PB2002 north of the JRN end kept."""
    rows = []
    for f in feats:
        p = f["properties"]
        if f["geometry"]["type"] != "LineString" or "alternative" in p["type"] or "strand" in p["type"]:
            continue
        kind = {"spreading ridge": "OSR", "transform": "OTF", "step-over": "OTF"}.get(p["type"])
        if kind is None:
            continue
        xy = C.densify(np.asarray(f["geometry"]["coordinates"]), 1.0)
        code = p.get("section_code") or "PB2002N"
        if p["type"] == "step-over":
            name = f"{code}_step{p['break_k']}"
        else:
            name = f"{code}{p['segment']}" if p.get("segment") is not None else code
        rows.append(pd.DataFrame(dict(lon=xy[:, 0], lat=xy[:, 1], kind=kind, section=name, source=p.get("source") or p.get("origin", ""))))
    tj = next(c for c in corners if "triple" in c["name"])
    rows.append(pd.DataFrame(dict(lon=[tj["lon"]], lat=[tj["lat"]], kind=["TJ"], section=["TJ"],
                                  source=["intersection MFZ x trench"])))
    pd.concat(rows, ignore_index=True).to_csv(C.OUT / "plate_axes_v2.csv", index=False, float_format="%.5f")


if __name__ == "__main__":
    main()

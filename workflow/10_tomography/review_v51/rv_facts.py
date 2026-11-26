"""Factual lines of one event view, generated from the numbers (no verdicts).

Every number is computed here from the event cache (rv_event.py) and the v5.1 travel-time grids. Residuals use each
hypothesis' refitted origin time (rv_event docstring) unless stated.
"""
import numpy as np
import pandas as pd

import rv_common as C
import rv_event as E


def _fmt(x, nd=2, sign=False):
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def sp_depth(J, sta, sp_obs):
    """Depth (km bsl) below the v5.1 epicentre at which predicted S-P at sta equals sp_obs in the v5.1 grids
    (1 km steps of z', linear interpolation), or a statement that no depth matches."""
    x, y = J["nll"]["x"], J["nll"]["y"]
    zps = np.arange(-2.0, 121.0, 1.0)
    tp = C.tt_profile("P", sta, x, y, zps)
    ts = C.tt_profile("S", sta, x, y, zps)
    tc = C.station_terms()
    sp = ts + tc.get((sta, "S"), 0.0) - tp - tc.get((sta, "P"), 0.0)
    ok = np.isfinite(sp)
    if ok.sum() < 3:
        return None, None
    z, sp = C.ieft(zps[ok]), sp[ok]
    if sp_obs < sp.min():
        return None, f"shorter than predicted for any depth below the v5.1 epicentre (minimum {sp.min():.2f} s at {z[np.argmin(sp)]:.0f} km)"
    if sp_obs > sp.max():
        return None, f"longer than predicted for any depth to {z.max():.0f} km below the v5.1 epicentre (maximum {sp.max():.2f} s)"
    i = int(np.argmin(np.abs(sp - sp_obs)))
    # refine linearly with the neighbour on the right side
    j = i + 1 if (i + 1 < len(sp) and (sp[i] - sp_obs) * (sp[i + 1] - sp_obs) <= 0) else max(i - 1, 0)
    if j != i and sp[j] != sp[i]:
        zz = z[i] + (sp_obs - sp[i]) * (z[j] - z[i]) / (sp[j] - sp[i])
    else:
        zz = z[i]
    return float(zz), None


def p_span_shallow(J, P, depth=5.0):
    """Predicted P time span over the P-picked stations for a source `depth` km below the v5.1 epicentre."""
    x, y = J["nll"]["x"], J["nll"]["y"]
    zp = float(C.eft(depth))
    tc = C.station_terms()
    v = []
    for st in P[P.phase == "P"].sta:
        t, _ = C.tt("P", st, x, y, zp, 0)
        v.append(t + tc.get((st, "P"), 0.0))
    v = np.array(v, float)
    v = v[np.isfinite(v)]
    return float(v.max() - v.min()) if len(v) >= 2 else np.nan


CLOCK_STA = ("MH07.XD", "MG07.XD")
CLOCK_P0, CLOCK_P1 = pd.Timestamp("2015-11-21", tz="UTC"), pd.Timestamp("2016-05-03", tz="UTC")


def facts(J, row, problem):
    P = pd.DataFrame(J["phases"])
    hy = J["hyp"]
    hs = [h for h in E.HYPS if h in hy]
    L = []
    if not len(P):
        return ["The NonLinLoc output of this event has no phase lines."]
    P["tobs_ts"] = pd.to_datetime(P.tobs, utc=True, format="ISO8601")
    # 1 fit
    L.append("Residual rms with each hypothesis' origin time refitted in the v5.1 grids: "
             + ", ".join(f"{h} {_fmt(hy[h]['rms_fit'])} s" for h in hs)
             + f" ({hy['v5.1']['n']} picks); with the catalog origin times: "
             + ", ".join(f"{h} {_fmt(hy[h]['rms_ot'])} s" for h in hs) + ".")
    med = {h: float(np.nanmedian(np.abs(P[f"resfit_{h}"]))) for h in hs}
    L.append("Median |observed - predicted| (OT refit): " + ", ".join(f"{h} {_fmt(med[h])} s" for h in hs) + ".")
    # 2 nearest stations
    sts = list(dict.fromkeys(P.sort_values("dist").sta))[:3]
    X = P[P.sta.isin(sts)]
    L.append(f"At the {len(sts)} nearest stations with picks ("
             + ", ".join(f"{s} {P[P.sta == s].dist.iloc[0]:.0f} km" for s in sts)
             + f"; {len(X)} picks), mean |observed - predicted| (OT refit): "
             + ", ".join(f"{h} {_fmt(np.nanmean(np.abs(X[f'resfit_{h}'])))} s" for h in hs) + ".")
    for ph in ("P", "S"):
        Y = P[P.phase == ph].sort_values("dist").head(3)
        if len(Y) >= 2:
            best = {h: np.nanmean(np.abs(Y[f"resfit_{h}"])) for h in hs}
            order = sorted(best, key=best.get)
            if len(order) >= 2 and np.isfinite(best[order[0]]) and np.isfinite(best[order[1]]):
                vals = ", ".join(f"{h} {best[h]:.2f} s" for h in order if np.isfinite(best[h]))
                if best[order[1]] - best[order[0]] < 0.05:
                    L.append(f"{ph} at the {len(Y)} nearest stations with {'an' if ph == 'S' else 'a'} {ph} pick "
                             f"({', '.join(Y.sta)}): mean |res| (OT refit) {vals}; {order[0]} and {order[1]} within 0.05 s.")
                else:
                    L.append(f"{ph} at the {len(Y)} nearest stations with {'an' if ph == 'S' else 'a'} {ph} pick "
                             f"({', '.join(Y.sta)}): mean |res| (OT refit) {vals}; {order[0]} smallest by "
                             f"{best[order[1]] - best[order[0]]:.2f} s.")
    # 3 S-P at the nearest station with both
    both = [s for s in P.sort_values("dist").sta.unique() if set(P[P.sta == s].phase) >= {"P", "S"}]
    for s in both[:2]:
        pp, ss = P[(P.sta == s) & (P.phase == "P")].iloc[0], P[(P.sta == s) & (P.phase == "S")].iloc[0]
        sp_obs = (ss.tobs_ts - pp.tobs_ts).total_seconds()
        pr = {h: (ss[f"tt_{h}"] + ss.tcorr) - (pp[f"tt_{h}"] + pp.tcorr) for h in hs}
        z, msg = sp_depth(J, s, sp_obs)
        tail = (f" In the v5.1 grids this S-P is reached {z:.1f} km below the v5.1 epicentre." if z is not None
                else (f" Observed S-P is {msg}." if msg else ""))
        L.append(f"S-P at {s} ({pp.dist:.0f} km): observed {sp_obs:.2f} s; predicted "
                 + ", ".join(f"{_fmt(pr[h])} s ({h})" for h in hs) + "." + tail)
    if not both:
        L.append("No station has both a P and an S pick.")
    # 4 down-weighted arrivals
    lw = P[P.weight < 0.1]
    if len(lw):
        L.append(f"NonLinLoc gave weight < 0.1 to {len(lw)} of {len(P)} arrivals: "
                 + ", ".join(f"{r.sta} {r.phase} ({r.dist:.0f} km, res {r.res:+.2f} s)" for r in lw.head(8).itertuples())
                 + (" ..." if len(lw) > 8 else "") + ".")
    # 5 P-pick span
    PP = P[P.phase == "P"]
    if len(PP) >= 3 and PP.dist.max() < 120:
        span = (PP.tobs_ts.max() - PP.tobs_ts.min()).total_seconds()
        pspan = {h: float((PP[f"tt_{h}"] + PP.tcorr).max() - (PP[f"tt_{h}"] + PP.tcorr).min()) for h in hs}
        s5 = p_span_shallow(J, P, 5.0)
        L.append(f"P picks at {len(PP)} stations {PP.dist.min():.0f}-{PP.dist.max():.0f} km from the v5.1 epicentre span "
                 f"{span:.2f} s; predicted span " + ", ".join(f"{h} {pspan[h]:.2f} s" for h in hs)
                 + f"; a source 5 km below the v5.1 epicentre would give {s5:.2f} s.")
    # 6 hypocentre differences
    d51 = hy["v5.1"]
    for h in ("v4", "ComCat"):
        if h in hy and hy[h]["depth"] is not None:
            dz = d51["depth"] - hy[h]["depth"]
            dh = float(C.hav(d51["lat"], d51["lon"], hy[h]["lat"], hy[h]["lon"]))
            dt = (pd.Timestamp(d51["OT"]) - pd.Timestamp(hy[h]["OT"])).total_seconds() if hy[h]["OT"] else np.nan
            L.append(f"v5.1 minus {h}: depth {dz:+.1f} km, epicentre {dh:.1f} km apart, origin time {dt:+.2f} s.")
    # 7 scatter
    f = C.WORK / "events" / f"{J['ID']}.scat.npy"
    if f.exists():
        s = np.load(f)
        q = np.percentile(s[:, 2], [16, 50, 84])
        L.append(f"NonLinLoc scatter depth: median {q[1]:.1f} km, 16-84 % range {q[0]:.1f}-{q[2]:.1f} km "
                 f"(catalog sig_z {row.sig_z:.1f} km).")
    # 7b station clock errors found by the CC-dt QC
    ck = P[P.sta.isin(CLOCK_STA)]
    if len(ck) and CLOCK_P0 <= pd.Timestamp(hy["v5.1"]["OT"]) < CLOCK_P1:
        L.append("Picks at " + ", ".join(f"{r.sta} {r.phase} (res {r.res:+.2f} s, weight {r.weight:.2f})" for r in ck.itertuples())
                 + ": the CC-dt QC found record-header clock errors at MH07.XD and MG07.XD on some days of 2015-11-21 to "
                 "2016-05-02; on 10 days each the v5.1 residuals there are offset by a median +0.45 s (MH07.XD) and "
                 "-0.37 s (MG07.XD) relative to the other XD stations (P1 context section).")
    # 8 unpicked stations with data inside the picked distance range
    up = J.get("unpicked", {})
    wi = J.get("wf_info", {})
    dmaxP = P[P.phase == "P"].dist.max() if (P.phase == "P").any() else P.dist.max()
    quiet = [f"{s} {d['dist']:.0f} km" for s, d in sorted(up.items(), key=lambda kv: kv[1]["dist"])
             if d["dist"] is not None and d["dist"] <= dmaxP and str(wi.get(s, {}).get("status", "")).startswith("ok")]
    if quiet:
        L.append(f"Stations with data but no pick, closer than the farthest P pick ({dmaxP:.0f} km): " + ", ".join(quiet) + ".")
    # 9 data
    nod = [f"{s} ({i.get('status', '')})" for s, i in wi.items() if not str(i.get("status", "")).startswith("ok")]
    L.append(f"Waveforms: {len(wi) - len(nod)} of {len(wi)} displayed stations with data"
             + (f"; without: {'; '.join(nod)}" if nod else "") + ".")
    if J.get("stations_picked_not_shown"):
        L.append(f"{len(J['stations_picked_not_shown'])} picked stations are not drawn (more than {E.MAX_STA}): "
                 + ", ".join(J["stations_picked_not_shown"][:20]) + (" ..." if len(J["stations_picked_not_shown"]) > 20 else "") + ".")
    return L

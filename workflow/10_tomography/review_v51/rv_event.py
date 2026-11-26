"""Per-event data of the v5.1 location review: picks and residuals for three hypocentres in the v5.1 grids, the
NonLinLoc scatter, and waveforms read from pnwstore. Cached under WORK/events/<ID>.{json,npz} (resumable).

Hypotheses (all in the SAME v5.1 travel-time grids and station terms, so differences come from the location only):
  v5.1    NonLinLoc maximum-likelihood point of run b4_v51_iter1 (deep pass b4_v51_deep), its origin time
  v4      catalog v4 hypocentre (v4_lat, v4_lon, v4_depth, v4_time)
  ComCat  the matched ComCat hypocentre (tensor match, else the ANSS match of the magnitude table), when one exists
For each hypothesis the predicted arrival is OT + T(station, phase, hypocentre) + station term (D-B069 terms of
b4_v51_iter0), T from the grid NLLoc used for that arrival in the v5.1 location (header rule on the event's LOCGRID
box), the other grid when the hypocentre falls outside it. Two origin times are used: the hypothesis' own origin time
(catalog OT) and a refitted one: OT + weighted mean of (observed - predicted), weights 1 / (pick error^2 + 0.2^2).
"""
import json
import time
import traceback

import numpy as np
import pandas as pd

import rv_common as C

HYPS = ("v5.1", "v4", "ComCat")
MAX_STA = 36          # waveform stations from the picked ones (24 nearest + 12 spread beyond when more)
N_UNPICKED = 6        # nearest stations without a pick, shown when they have data


def _f(x, nd=3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(x) else round(x, nd)


def band_for(dmed, mag):
    """Filter band (Hz) by the median distance of the displayed stations and the magnitude. The low corner stays at
    1 Hz or above: below it the ocean microseism (0.1-0.5 Hz) dominates the OBS and coastal records."""
    if mag is not None and np.isfinite(mag) and mag >= 4.5:
        return (1.0, 5.0)
    if dmed < 40:
        return (2.0, 15.0)
    if dmed < 150:
        return (1.0, 10.0)
    return (1.0, 6.0)


def hypotheses(row, H):
    out = {}
    out["v5.1"] = dict(lat=H["lat"], lon=H["lon"], depth=float(C.ieft(H["zp"])), zp=H["zp"], x=H["x"], y=H["y"],
                       OT=H["OT"], src="NonLinLoc v5.1 (b4_v51_deep)" if row.deep_pass else "NonLinLoc v5.1 (b4_v51_iter1)")
    if np.isfinite(row.v4_lat) and np.isfinite(row.v4_depth):
        x, y = C.lambert(row.v4_lon, row.v4_lat)
        out["v4"] = dict(lat=float(row.v4_lat), lon=float(row.v4_lon), depth=float(row.v4_depth),
                         zp=float(C.eft(row.v4_depth)), x=float(x), y=float(y), OT=row.t_v4, src="catalog v4")
    cc = C.comcat_for(int(row.ID))
    if cc is not None and np.isfinite(cc["lat"]):
        x, y = C.lambert(cc["lon"], cc["lat"])
        out["ComCat"] = dict(lat=cc["lat"], lon=cc["lon"], depth=cc["depth"], zp=float(C.eft(cc["depth"])),
                             x=float(x), y=float(y), OT=cc["t"], src=f"ComCat {cc['id']} ({cc['src']})", info=cc)
    return out


def predict(P, box, hy, row):
    """Adds per hypothesis tt_<h>, pred_<h> (catalog OT), res_<h>, resfit_<h> columns; returns (P, otfit dict)."""
    P = P.copy()
    P["root"] = [C.serving_root(box, ph, st) for ph, st in zip(P.phase, P.sta)]
    w = 1.0 / (P.err.values ** 2 + C.MODEL_ERR ** 2)
    otfit = {}
    for h, d in hy.items():
        tts, used = [], []
        for ph, st, rt in zip(P.phase, P.sta, P.root):
            v, u = C.tt(ph, st, d["x"], d["y"], d["zp"], rt)
            tts.append(v)
            used.append(u)
        P[f"tt_{h}"] = tts
        P[f"root_{h}"] = used
        tobs = (P.tobs - d["OT"]).dt.total_seconds().values
        res = tobs - P[f"tt_{h}"].values - P.tcorr.values
        P[f"res_{h}"] = res
        ok = np.isfinite(res)
        shift = float(np.sum(w[ok] * res[ok]) / np.sum(w[ok])) if ok.any() else np.nan
        otfit[h] = shift
        P[f"resfit_{h}"] = res - shift
    return P, otfit


def rms(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else np.nan


def unpicked_candidates(lat, lon, picked, dmax, t_lo, t_hi, n=N_UNPICKED):
    S = C.stations()
    d = C.hav(lat, lon, S.lat.values, S.lon.values)
    order = np.argsort(d)
    out, nq = [], 0
    for i in order:
        if d[i] > dmax or len(out) >= n or nq >= 3 * n:
            break
        st = S.index[i]
        if st in picked:
            continue
        net, code = S.net.iloc[i], S.code.iloc[i]
        if net in C.NO_PNWSTORE:
            continue
        nq += 1
        rows = C.index_rows(net, code, t_lo, t_hi)
        if len(rows) and C.choose_band(rows) is not None:
            out.append(st)
    return out


def read_station(sta, t0, t1, t_anchor):
    """Raw traces (Z and horizontals of the picker's band) of one station: dict comp -> [(t0, sr, data)], info."""
    code, net = sta.split(".")
    info = dict(sta=sta)
    if net in C.NO_PNWSTORE:
        info["status"] = "no data: network not in pnwstore (NC/BK were read from NCEDC by the picker)"
        return {}, info
    lo, hi = pd.Timestamp(t0, unit="s", tz="UTC"), pd.Timestamp(t1, unit="s", tz="UTC")
    rows = C.index_rows(net, code, lo, hi)
    cb = C.choose_band(rows)
    if cb is None:
        info["status"] = "no data: no HH/BH/EH vertical in the pnwstore index for this window"
        return {}, info
    band, zc, hz = cb
    info["band"] = band
    out = {}
    stat = []
    for comp, cha in [("Z", zc)] + [(c[-1], c) for c in hz]:
        segs, inf = C.read_window(rows, cha, t_anchor, t0, t1)
        stat.append(f"{cha}:{inf['status']}")
        if segs:
            out[cha] = segs
        if "anchor_shift_ms" in inf and comp == "Z":
            info["anchor_shift_ms"] = round(inf["anchor_shift_ms"], 2)
    info["channels"] = " ".join(stat)
    info["status"] = "ok" if any(k[-1] in "Z3" for k in out) else ("horizontals only" if out else "no data: no records in the window")
    return out, info


def compute(eid, force=False, read_waveforms=True):
    """Builds WORK/events/<eid>.json (+ .npz waveforms). Returns the json dict."""
    fj = C.WORK / "events" / f"{eid}.json"
    if fj.exists() and not force:
        return json.loads(fj.read_text())
    fj.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    c = C.catalog()
    row = c.loc[eid]
    out = dict(ID=int(eid), problems=[])
    hf, sf, wd = C.find_hyp(eid, row)
    note = []
    if hf is None:
        hf = C.sum_block(eid, wd)
        note.append("no per-event .hyp in the workdir (file name shared with another event); hypocentre and phases from "
                    "loc.sum.grid0.loc.hyp, no scatter")
    if hf is None:
        out["error"] = "no NonLinLoc output found"
        fj.write_text(json.dumps(out))
        return out
    H = C.parse_hyp(hf)
    P = H["phases"]
    if len(P) == 0:
        note.append("the .hyp has no phase lines")
    box = C.control_box(wd)
    hy = hypotheses(row, H)
    P, otfit = predict(P, box, hy, row)
    S = C.stations()
    P["lat"] = [S.lat.get(s, np.nan) for s in P.sta]
    P["lon"] = [S.lon.get(s, np.nan) for s in P.sta]
    P["dist"] = C.hav(H["lat"], H["lon"], P.lat.values, P.lon.values)
    P["hdist"] = np.hypot(P.dist, hy["v5.1"]["depth"] + np.array([S.z_km.get(s, 0.0) * 0 for s in P.sta]))
    P = P.sort_values(["dist", "phase"]).reset_index(drop=True)
    out["hyp_file"] = str(hf)
    out["scat_file"] = str(sf) if sf is not None else None
    out["notes"] = note
    # scatter (flattened -> true depth, Lambert -> lon/lat)
    scat = None
    if sf is not None:
        a = C.read_scatter(sf, max_n=2000)
        lon, lat = C.proj()(a[:, 0] * 1e3, a[:, 1] * 1e3, inverse=True)
        scat = np.c_[lon, lat, C.ieft(a[:, 2]), a[:, 3]].astype(np.float32)
    # hypotheses summary
    hyo = {}
    for h, d in hy.items():
        sel = np.isfinite(P[f"res_{h}"].values)
        hyo[h] = dict(lat=_f(d["lat"], 4), lon=_f(d["lon"], 4), depth=_f(d["depth"], 2), src=d["src"],
                      OT=str(d["OT"]) if d["OT"] is not None else None, ot_refit_shift=_f(otfit[h], 3),
                      rms_ot=_f(rms(P.loc[sel, f"res_{h}"]), 3), rms_fit=_f(rms(P.loc[sel, f"resfit_{h}"]), 3),
                      n=int(sel.sum()),
                      rms_fit_P=_f(rms(P.loc[sel & (P.phase == "P").values, f"resfit_{h}"]), 3),
                      rms_fit_S=_f(rms(P.loc[sel & (P.phase == "S").values, f"resfit_{h}"]), 3),
                      wide_used=int(np.sum(np.array(P[f"root_{h}"].values) == 1)),
                      no_grid=int(np.sum(~sel)))
        if h == "ComCat":
            hyo[h]["info"] = {k: (str(v) if not isinstance(v, (int, float)) else _f(v, 4)) for k, v in d["info"].items()}
    out["hyp"] = hyo
    out["nll"] = dict(status=H.get("status"), rms=H.get("rms"), nphs=H.get("nphs"), gap=H.get("gap"),
                      dist=H.get("dist"), zp=H["zp"], x=H["x"], y=H["y"])
    # check of the grid reader against NLLoc's own TTpred at the v5.1 point
    out["tt_check_max_s"] = _f(np.nanmax(np.abs(P["tt_v5.1"] - P.ttpred)) if len(P) else np.nan, 5)
    cols = ["sta", "phase", "dist", "err", "tobs", "ttpred", "res", "tcorr", "weight", "root"] + \
           [f"{k}_{h}" for h in hy for k in ("tt", "res", "resfit")]
    PP = P[cols].copy()
    PP["tobs"] = PP.tobs.astype(str)
    out["phases"] = json.loads(PP.to_json(orient="records", double_precision=4))
    # ------------------------------------------------------------ waveform stations
    picked = list(dict.fromkeys(P.sta))
    if len(picked) > MAX_STA:
        far = picked[24:]
        keep = picked[:24] + [far[i] for i in np.unique(np.linspace(0, len(far) - 1, MAX_STA - 24).astype(int))]
    else:
        keep = picked
    dmax_keep = float(P[P.sta.isin(keep)].dist.max()) if len(P) else 50.0
    ot = hy["v5.1"]["OT"]
    tP, tS = [], []
    for h, d in hy.items():
        if d["OT"] is None:
            continue
        sh = otfit[h] if np.isfinite(otfit[h]) else 0.0
        off = (d["OT"] - ot).total_seconds() + sh
        for ph, lst in (("P", tP), ("S", tS)):
            x = P[P.phase == ph]
            lst += list(off + x[f"tt_{h}"] + x.tcorr)
    tobs_rel = (P.tobs - ot).dt.total_seconds()
    allt = np.array([v for v in tP + tS + list(tobs_rel) if np.isfinite(v)])
    pmax = max([v for v in tP if np.isfinite(v)] + [1.0])
    pre = float(np.clip(0.25 * pmax, 5.0, 20.0))
    rel_lo = float(min(0.0, np.nanmin(allt) if len(allt) else 0.0) - pre)
    rel_hi = float((np.nanmax(allt) if len(allt) else 30.0) + max(8.0, 0.15 * pmax))
    rel_hi = min(rel_hi, rel_lo + 320.0)
    t0 = ot.timestamp() + rel_lo - 3.0
    t1 = ot.timestamp() + rel_hi + 3.0
    unp = []
    if read_waveforms:
        unp = unpicked_candidates(H["lat"], H["lon"], set(picked), max(dmax_keep, 10.0),
                                  pd.Timestamp(t0, unit="s", tz="UTC"), pd.Timestamp(t1, unit="s", tz="UTC"))
    # predictions at the unpicked stations (P and S, each hypothesis, refit OT)
    upred = {}
    for st in unp:
        dd = {}
        for ph in ("P", "S"):
            rt = C.serving_root(box, ph, st)
            for h, d in hy.items():
                v, _ = C.tt(ph, st, d["x"], d["y"], d["zp"], rt)
                if d["OT"] is None or not np.isfinite(v):
                    continue
                dd[f"{ph}_{h}"] = _f((d["OT"] - ot).total_seconds() + otfit[h] + v + C.station_terms().get((st, ph), 0.0), 3)
        upred[st] = dict(pred=dd, dist=_f(C.hav(H["lat"], H["lon"], S.lat[st], S.lon[st]), 2))
    out["unpicked"] = upred
    out["window_rel"] = [rel_lo, rel_hi]
    dmed = float(np.median([P[P.sta == s].dist.iloc[0] for s in keep])) if keep else 50.0
    M = row.M if np.isfinite(row.M) else (hy.get("ComCat", {}).get("info", {}) or {}).get("mag", np.nan)
    out["band"] = band_for(dmed, M)
    # waveforms
    wf_info, arrays = {}, {}
    if read_waveforms:
        nc = [s for s in keep + unp if s.split(".")[1] in C.NO_PNWSTORE]
        nc_data, nc_info = C.read_ncedc(nc, t0, t1)
        for st in keep + unp:
            if st in nc:
                segs, info = nc_data.get(st, {}), nc_info.get(st, dict(sta=st, status="no data"))
            else:
                pp = P[(P.sta == st) & (P.phase == "P")]
                anchor = pp.tobs.iloc[0].timestamp() if len(pp) else ot.timestamp() + (upred.get(st, {}).get("pred", {}).get("P_v5.1") or 0.0)
                try:
                    segs, info = read_station(st, t0, t1, anchor)
                except Exception as e:
                    segs, info = {}, dict(sta=st, status=f"error: {type(e).__name__}: {e}")
            wf_info[st] = info
            for cha, sl in segs.items():
                for k, (ts, sr, data) in enumerate(sl):
                    key = f"{st}|{cha}|{k}"
                    arrays[key] = data
                    arrays[key + "|meta"] = np.array([ts - ot.timestamp(), sr], dtype=np.float64)
        np.savez_compressed(C.WORK / "events" / f"{eid}.npz", **arrays)
    out["stations_shown"] = keep + unp
    out["stations_picked_not_shown"] = [s for s in picked if s not in keep]
    out["wf_info"] = wf_info
    if scat is not None:
        np.save(C.WORK / "events" / f"{eid}.scat.npy", scat)
    out["has_scatter"] = scat is not None
    out["elapsed_s"] = round(time.time() - t_start, 1)
    fj.write_text(json.dumps(out, default=str))
    return out


def load_waveforms(eid):
    f = C.WORK / "events" / f"{eid}.npz"
    if not f.exists():
        return {}
    z = np.load(f)
    out = {}
    for k in z.files:
        if k.endswith("|meta"):
            continue
        st, cha, i = k.split("|")
        rel0, sr = z[k + "|meta"]
        out.setdefault(st, {}).setdefault(cha, []).append((float(rel0), float(sr), z[k]))
    return out


if __name__ == "__main__":
    import sys
    for e in sys.argv[1:]:
        try:
            d = compute(int(e), force=True)
            print(e, d.get("elapsed_s"), d.get("tt_check_max_s"), {s: i.get("status") for s, i in d.get("wf_info", {}).items()})
        except Exception:
            traceback.print_exc()
